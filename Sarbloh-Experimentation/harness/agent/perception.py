"""Perception: everything the agent is shown about a game frame, in one file. Host side only.

Replaces the old ``vision.py`` (the picture) and ``intuition.py`` (letters, objects, change lines). Coordinates are
always (row, column), top-left = (0, 0), the order of the ``act`` click "6 r c".

Three layers:

1. Text views (kept from ``intuition.py``): the board as colour letters with numbered rows and columns; objects as
   4-connected one-colour regions whose ids stay the same across the steps of a level (``Tracker``); one change
   line per step.
2. The briefing, the winner of the perception lab (``experiments/*_perception_lab``, fusion variant), ported to work on
   the grid instead of a screenshot. A blueprint layout of the frame (terrain, panels, HUD, composite "things"),
   then MEASURED facts computed by code (reach, enclosure, tethers and midpoints, alignment, look-alikes, copies,
   lopsided colours) and GUESSES (a role per thing, each with a one-action test). A thing is named after the
   object-list id that covers most of it, so the text, the picture and the change lines share one set of ids.
3. The picture: the lab's map (blueprint, role tags, look-alike threads, midpoint rings, tile lattice) with the HUD drawn
   enlarged on the right. The briefing is not drawn into it: it travels as text.

    ascii(grid, box) / legend(colours) / rows_of(grid)    text helpers
    segment(grid) -> (objects, background);  Tracker().observe(grid) -> (objects, change line or None)
    brief(grid, objects, actions) -> Briefing             the briefing analysis
    picture(briefing, cell) -> PNG;  render(grid, cell) -> PNG of the plain frame
    Scene(...).text(full=...)                             what the agent reads; Scene.to_json() feeds ``observe()``
    image_message(text, png, step)                        the user message that carries text and picture

Deviations from the lab version, each for correctness or to bound the output on busy frames: facts name things by id; repeated
facts are dropped; "share one open space" is checked with a flood fill; alignment skips overlapping pairs; a line
whose two ends touch the same thing is not a tether; things with no evidence get no guess; look-alikes and straight
paths are checked among the largest things only (``MAX_PAIR_THINGS``, ``MAX_PATH_THINGS``); the tile-grid test leaves
out status boxes and keeps the level's previous grid while it still fits (both stop it flipping as you play: ls20
lost its grid after one move, ka59 whenever the self touched another thing).
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import math
import threading
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# =====================================================================================================================
# 1. Colours and the letter board
# =====================================================================================================================

PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 255, 255), (204, 204, 204), (153, 153, 153), (102, 102, 102),   # 0 white, 1 light grey, 2 grey, 3 dark grey
    (51, 51, 51), (0, 0, 0), (229, 58, 163), (255, 123, 204),             # 4 charcoal, 5 black, 6 magenta, 7 pink
    (249, 60, 49), (30, 147, 255), (136, 216, 241), (255, 220, 0),        # 8 red, 9 blue, 10 sky blue, 11 yellow
    (255, 133, 27), (146, 18, 49), (79, 204, 48), (163, 86, 214),         # 12 orange, 13 dark red, 14 green, 15 purple
)
LETTERS = "WwgGcBMPRbSYOrNp"
NAMES = ("white", "light grey", "grey", "dark grey", "charcoal", "black", "magenta", "pink", "red", "blue", "sky blue",
         "yellow", "orange", "dark red", "green", "purple")
Box = tuple[int, int, int, int]   # r0, c0, r1, c1 inclusive
Cell = tuple[int, int]            # (row, column)


def rows_of(grid: Any) -> list[list[int]]:
    grid = getattr(grid, "grid", grid)
    return grid.tolist() if hasattr(grid, "tolist") else [list(map(int, r)) for r in grid]


def letter(v: int) -> str:
    return LETTERS[int(v) & 15]


def legend(colours: Any) -> str:
    """``W=white(0) R=red(8)`` for the given colour values, in value order."""
    return " ".join(f"{LETTERS[c]}={NAMES[c]}({c})" for c in sorted({int(c) & 15 for c in colours}))


def ascii(grid: Any, box: Box | None = None) -> str:
    """The grid (or ``box`` of it) as letters. Two header lines give the column number (tens, ones); each row starts
    with its row number."""
    rows = rows_of(grid)
    if not rows:
        return "(empty grid)"
    r0, c0, r1, c1 = box or (0, 0, len(rows) - 1, len(rows[0]) - 1)
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(len(rows) - 1, r1), min(len(rows[0]) - 1, c1)
    cols = range(c0, c1 + 1)
    out = ["    " + "".join(str(c // 10 % 10) for c in cols), "    " + "".join(str(c % 10) for c in cols)]
    out += [f"{r:>3} " + "".join(letter(rows[r][c]) for c in cols) for r in range(r0, r1 + 1)]
    return "\n".join(out)


def background(rows: list[list[int]]) -> int:
    return Counter(v for r in rows for v in r).most_common(1)[0][0] if rows and rows[0] else 0


# =====================================================================================================================
# 2. Objects (segmentation), stable ids and change lines
# =====================================================================================================================

@dataclass
class Obj:
    id: int
    colour: int
    cells: frozenset[tuple[int, int]]   # (r, c)
    bbox: Box
    hash: str                           # shape only (translation-invariant); colour is separate
    corners: int                        # vertices of the outline: a rectangle has 4, an L 6
    parent: int | None = None           # smallest object whose box holds this one
    children: list[int] = field(default_factory=list)
    adjacent: list[int] = field(default_factory=list)
    hud: bool = False                   # lies in the 3-cell band along a frame edge: likely a counter or bar

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def hw(self) -> tuple[int, int]:
        return self.bbox[2] - self.bbox[0] + 1, self.bbox[3] - self.bbox[1] + 1

    @property
    def solid(self) -> bool:
        h, w = self.hw
        return self.size == h * w

    def label(self) -> str:
        """``R 3x3`` for a filled rectangle, else ``R 7px``."""
        h, w = self.hw
        return f"{letter(self.colour)} {h}x{w}" if self.solid else f"{letter(self.colour)} {self.size}px"

    def at(self) -> str:
        r0, c0, r1, c1 = self.bbox
        return f"r{r0} c{c0}" if (r0, c0) == (r1, c1) else f"r{r0}-{r1} c{c0}-{c1}"

    def row(self) -> str:
        rel = []
        if self.parent is not None:
            rel.append(f"in {self.parent}")
        if self.children:
            rel.append("has " + ",".join(map(str, self.children[:8])) + ("+" if len(self.children) > 8 else ""))
        if self.adjacent:
            rel.append("adj " + ",".join(map(str, self.adjacent[:8])) + ("+" if len(self.adjacent) > 8 else ""))
        return (f"{self.id:>3} {self.label():<9} {self.at():<17} #{self.hash} {self.corners}c"
                + (" hud" if self.hud else "") + (" | " + "; ".join(rel) if rel else ""))

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "colour": self.colour, "letter": letter(self.colour), "name": NAMES[self.colour & 15],
                "size": self.size, "bbox": list(self.bbox), "hash": self.hash, "corners": self.corners,
                "parent": self.parent, "children": list(self.children), "adjacent": list(self.adjacent),
                "hud": self.hud, "cells": sorted(self.cells) if self.size <= 256 else None}


def _shape_hash(cells: frozenset[tuple[int, int]], r0: int, c0: int) -> str:
    norm = sorted((r - r0, c - c0) for r, c in cells)
    return hashlib.blake2b(repr(norm).encode(), digest_size=3).hexdigest()


def _corners(cells: frozenset[tuple[int, int]]) -> int:
    """Outline vertices: 2x2 windows with 1 or 3 cells inside are one vertex, the two diagonal patterns are two."""
    seen: set[tuple[int, int]] = set()
    n = 0
    for r, c in cells:
        for wr in (r - 1, r):
            for wc in (c - 1, c):
                if (wr, wc) in seen:
                    continue
                seen.add((wr, wc))
                a, b = (wr, wc) in cells, (wr, wc + 1) in cells
                d, e = (wr + 1, wc) in cells, (wr + 1, wc + 1) in cells
                k = a + b + d + e
                if k in (1, 3):
                    n += 1
                elif k == 2 and a == e:
                    n += 2
    return n


def segment(grid: Any, bg: int | None = None) -> tuple[list[Obj], int]:
    """Objects in reading order of their first cell, ids 1, 2, ...; the most common colour is the background."""
    rows = rows_of(grid)
    if not rows or not rows[0]:
        return [], 0
    bg = background(rows) if bg is None else bg
    h, w = len(rows), len(rows[0])
    label = [[0] * w for _ in range(h)]
    objs: list[Obj] = []
    for sr in range(h):
        for sc in range(w):
            if label[sr][sc] or rows[sr][sc] == bg:
                continue
            colour, oid = rows[sr][sc], len(objs) + 1
            stack, cells = [(sr, sc)], []
            label[sr][sc] = oid
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= nr < h and 0 <= nc < w and not label[nr][nc] and rows[nr][nc] == colour:
                        label[nr][nc] = oid
                        stack.append((nr, nc))
            cs = frozenset(cells)
            rs, ks = [p[0] for p in cells], [p[1] for p in cells]
            box = (min(rs), min(ks), max(rs), max(ks))
            band = 3
            hud = (box[2] < band or box[0] >= h - band or box[3] < band or box[1] >= w - band)
            objs.append(Obj(oid, int(colour), cs, box, _shape_hash(cs, box[0], box[1]), _corners(cs), hud=hud))
    adj: dict[int, set[int]] = {o.id: set() for o in objs}
    for r in range(h):
        for c in range(w):
            a = label[r][c]
            if not a:
                continue
            for nr, nc in ((r + 1, c), (r, c + 1)):
                if nr < h and nc < w:
                    b = label[nr][nc]
                    if b and b != a:
                        adj[a].add(b)
                        adj[b].add(a)
    for o in objs:
        o.adjacent = sorted(adj[o.id])
    if len(objs) <= 600:  # containment is O(n^2): skip it on noise-heavy frames
        area = {o.id: o.hw[0] * o.hw[1] for o in objs}
        for o in objs:
            r0, c0, r1, c1 = o.bbox
            best = None
            for p in objs:
                if p is o or area[p.id] <= area[o.id]:
                    continue
                if p.bbox[0] <= r0 and p.bbox[1] <= c0 and p.bbox[2] >= r1 and p.bbox[3] >= c1:
                    if best is None or area[p.id] < area[best.id]:
                        best = p
            if best is not None:
                o.parent = best.id
                best.children.append(o.id)
    return objs, bg


def _renumber(objs: list[Obj], mapping: dict[int, int]) -> None:
    for o in objs:
        o.id = mapping[o.id]
    for o in objs:
        o.parent = mapping.get(o.parent) if o.parent is not None else None
        o.children = sorted(mapping[c] for c in o.children)
        o.adjacent = sorted(mapping[a] for a in o.adjacent)
    objs.sort(key=lambda o: o.id)


def _direction(dr: int, dc: int) -> str:
    parts = []
    if dr:
        parts.append(f"{'up' if dr < 0 else 'down'} {abs(dr)}")
    if dc:
        parts.append(f"{'left' if dc < 0 else 'right'} {abs(dc)}")
    return ", ".join(parts)


class Tracker:
    """Keeps object ids stable across the steps of a level, and describes each transition by object."""

    MAX_ITEMS = 8

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.prev: list[Obj] | None = None
        self.prev_rows: list[list[int]] | None = None
        self.next_id = 1

    def observe(self, grid: Any) -> tuple[list[Obj], str | None]:
        rows = rows_of(grid)
        objs, _ = segment(rows)
        if self.prev is None or self.prev_rows is None or len(rows) != len(self.prev_rows) \
                or (rows and len(rows[0]) != len(self.prev_rows[0])):
            _renumber(objs, {o.id: o.id for o in objs})
            self.next_id = len(objs) + 1
            self.prev, self.prev_rows = objs, rows
            return objs, None
        text = self._match(self.prev, objs, self.prev_rows, rows)
        self.prev, self.prev_rows = objs, rows
        return objs, text

    def _match(self, old: list[Obj], new: list[Obj], prev_rows: list[list[int]], rows: list[list[int]]) -> str:
        changed = sum(1 for r in range(len(rows)) for c in range(len(rows[0])) if rows[r][c] != prev_rows[r][c])
        mapping: dict[int, int] = {}
        left_old = {o.id: o for o in old}
        left_new = {o.id: o for o in new}
        events: list[tuple[bool, str]] = []   # (hud, text)

        def pair(n: Obj, o: Obj) -> None:
            mapping[n.id] = o.id
            left_old.pop(o.id, None)
            left_new.pop(n.id, None)

        by_key: dict[tuple, list[Obj]] = {}
        for o in old:
            by_key.setdefault((o.colour, o.hash, o.bbox), []).append(o)
        for n in new:                                     # 1. unchanged
            cand = [o for o in by_key.get((n.colour, n.hash, n.bbox), []) if o.id in left_old]
            if cand:
                pair(n, cand[0])
        for n in sorted(left_new.values(), key=lambda o: -o.size):   # 2. moved: same colour and shape
            cand = [o for o in left_old.values() if o.colour == n.colour and o.hash == n.hash]
            if cand:
                o = min(cand, key=lambda o: abs(o.bbox[0] - n.bbox[0]) + abs(o.bbox[1] - n.bbox[1]))
                pair(n, o)
                events.append((n.hud, f"obj {o.id} {n.label()} moved {_direction(n.bbox[0] - o.bbox[0], n.bbox[1] - o.bbox[1])}"
                                      f" -> {n.at()}"))
        for n in list(left_new.values()):                # 3. recoloured: same shape and place
            cand = [o for o in left_old.values() if o.hash == n.hash and o.bbox == n.bbox]
            if cand:
                o = cand[0]
                pair(n, o)
                events.append((n.hud, f"obj {o.id} {o.label()} at {o.at()} recoloured "
                                      f"{letter(o.colour)}->{letter(n.colour)}"))
        for n in sorted(left_new.values(), key=lambda o: -o.size):   # 4. reshaped: same colour, overlapping cells
            cand = [(len(o.cells & n.cells), o) for o in left_old.values() if o.colour == n.colour]
            cand = [x for x in cand if x[0] > 0]
            if cand:
                _, o = max(cand, key=lambda x: x[0])
                pair(n, o)
                verb = "grew" if n.size > o.size else "shrank" if n.size < o.size else "changed shape"
                events.append((n.hud, f"obj {o.id} {letter(n.colour)} {verb} {o.size}->{n.size} cells, now {n.at()}"))
        for o in sorted(left_old.values(), key=lambda o: o.id):   # 5. vanished
            events.append((o.hud, f"obj {o.id} {o.label()} vanished from {o.at()}"))
        for n in sorted(left_new.values(), key=lambda o: (o.bbox[0], o.bbox[1])):   # 6. appeared
            mapping[n.id] = self.next_id
            self.next_id += 1
            events.append((n.hud, f"obj {mapping[n.id]} {n.label()} appeared at {n.at()}"))
        _renumber(new, mapping)
        if not changed:
            return "no change"
        events.sort(key=lambda e: e[0])   # the play area first, HUD after
        texts = [t + (" (hud)" if hud else "") for hud, t in events]
        more = len(texts) - self.MAX_ITEMS
        body = "; ".join(texts[:self.MAX_ITEMS]) + (f"; +{more} more" if more > 0 else "")
        if changed > len(rows) * len(rows[0]) * 3 // 5:
            body = f"most of the screen changed ({len(texts)} object changes)"
        return (body or "pixels changed with no object change") + f" ({changed} cells changed)"


def change_boxes(prev: Any, grid: Any, margin: int = 3) -> list[Box]:
    """Boxes around the changed cells, expanded by ``margin`` and clipped; boxes that touch are merged."""
    a, b = rows_of(prev), rows_of(grid)
    if not a or not b or len(a) != len(b) or len(a[0]) != len(b[0]):
        return []
    h, w = len(b), len(b[0])
    cells = [(r, c) for r in range(h) for c in range(w) if a[r][c] != b[r][c]]
    if not cells:
        return []
    if len(cells) > 1500:
        rs, cs = [p[0] for p in cells], [p[1] for p in cells]
        return [(min(rs), min(cs), max(rs), max(cs))]
    boxes = [(max(0, r - margin), max(0, c - margin), min(h - 1, r + margin), min(w - 1, c + margin)) for r, c in cells]
    merged = True
    while merged:
        merged = False
        out: list[Box] = []
        for bx in boxes:
            for i, ob in enumerate(out):
                if bx[0] <= ob[2] + 1 and ob[0] <= bx[2] + 1 and bx[1] <= ob[3] + 1 and ob[1] <= bx[3] + 1:
                    out[i] = (min(bx[0], ob[0]), min(bx[1], ob[1]), max(bx[2], ob[2]), max(bx[3], ob[3]))
                    merged = True
                    break
            else:
                out.append(bx)
        boxes = out
    return sorted(boxes, key=lambda bx: -(bx[2] - bx[0] + 1) * (bx[3] - bx[1] + 1))


# =====================================================================================================================
# 3. The briefing: layout, measured facts, role guesses
# =====================================================================================================================

NB4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NB8 = NB4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
MAX_PAIR_THINGS = 60    # look-alikes are checked among this many of the largest things (pairs grow as n^2)
MAX_PATH_THINGS = 20    # straight-path checks among this many of the largest main objects
MAX_TINY = 300          # more tiny pieces than this is noise, not a dotted figure
ROLE_TESTS = {"YOU": "press one arrow key: it should move one tile and the gauge should drop",
              "GOAL": "get the right thing into it and see if the level ends",
              "MODIFIER": "step onto it, then check whether the inventory icon changed",
              "INVENTORY": "watch it after touching the modifier",
              "MOVES LEFT": "take any action and check it shrinks by one unit",
              "LIVES": "watch it when the gauge runs out or you fail",
              "HANDLE (click)": "click an empty spot once: the selected handle should jump there",
              "CARRIED THING": "move one handle: it should slide to the new midpoint",
              "SCENERY": "ignore unless something else fails"}
ROLE_COLOURS = {"YOU": (0, 230, 0), "GOAL": (255, 60, 60), "MODIFIER": (255, 160, 0), "INVENTORY": (0, 200, 255),
                "MOVES LEFT": (255, 255, 0), "LIVES": (255, 120, 200), "HANDLE (click)": (0, 255, 160),
                "CARRIED THING": (200, 120, 255), "SCENERY": (160, 160, 160)}
TURN_WORDS = {"rot90ccw": "turned 90 deg anticlockwise", "rot180": "turned 180 deg",
              "rot90cw": "turned 90 deg clockwise", "flipLR": "mirrored left-right",
              "rot90ccw+flipLR": "turned 90 deg anticlockwise then mirrored", "rot180+flipLR": "mirrored top-bottom",
              "rot90cw+flipLR": "turned 90 deg clockwise then mirrored"}


@dataclass(eq=False)
class Thing:
    """One thing of the layout: a composite of touching pieces of different colours, an icon (pieces of one colour
    inside a panel), a dotted figure, a thin line or a HUD bar."""
    kind: str                          # object | icon | dotted | line | bar
    cells: list[Cell]
    colors: Counter
    r0: int
    c0: int
    r1: int
    c1: int
    panel: int | None = None           # index of the panel it sits in
    on: int = -1                       # the terrain colour around it
    hud: bool = False                  # at the screen edge, or a bar: a counter, a gauge, an inventory
    name: str = ""                     # colour phrase + shape word, e.g. "blue top / orange bottom square"
    id: int | None = None              # the object-list id that covers most of it
    short: str = ""                    # its two main colours and shape: "blue/orange square"
    ref: str = ""                      # how the briefing names it: "obj 7 (blue/orange square)"
    _mask: np.ndarray | None = field(default=None, repr=False)

    @property
    def h(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def w(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def cy(self) -> float:
        return sum(y for y, _ in self.cells) / len(self.cells)

    @property
    def cx(self) -> float:
        return sum(x for _, x in self.cells) / len(self.cells)

    def mask(self) -> np.ndarray:
        """The shape in its box (read-only: shared by every check)."""
        if self._mask is None:
            m = np.zeros((self.h, self.w), bool)
            for y, x in self.cells:
                m[y - self.r0, x - self.c0] = True
            self._mask = m
        return self._mask

    def colgrid(self, g: np.ndarray) -> np.ndarray:
        out = np.full((self.h, self.w), -1, int)
        for y, x in self.cells:
            out[y - self.r0, x - self.c0] = g[y, x]
        return out


@dataclass
class Panel:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int
    border: bool          # touches the frame border


@dataclass
class Layout:
    g: np.ndarray
    terrain: list[int]    # colours that each cover more than 8% of the frame, most common first
    things: list[Thing]   # in reading order of their box corner
    panels: list[Panel]


def _bbox(cells: list[Cell]) -> Box:
    ys, xs = [c[0] for c in cells], [c[1] for c in cells]
    return min(ys), min(xs), max(ys), max(xs)


def _components(mask: np.ndarray, nb: tuple = NB8) -> list[list[Cell]]:
    """Connected regions of ``mask``, each a sorted list of cells, in reading order of their first cell."""
    m = mask.tolist()
    h, w = len(m), len(m[0]) if m else 0
    seen = [[False] * w for _ in range(h)]
    out = []
    for r in range(h):
        for c in range(w):
            if not m[r][c] or seen[r][c]:
                continue
            seen[r][c] = True
            stack, cells = [(r, c)], []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in nb:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and m[yy][xx] and not seen[yy][xx]:
                        seen[yy][xx] = True
                        stack.append((yy, xx))
            out.append(sorted(cells))
    return out


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """``mask`` plus the cells the outside cannot reach in 4-connected steps (its holes)."""
    pad = np.pad(mask, 1).tolist()
    h, w = len(pad), len(pad[0])
    outside = [[False] * w for _ in range(h)]
    outside[0][0] = True
    stack = [(0, 0)]
    while stack:
        y, x = stack.pop()
        for dy, dx in NB4:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and not pad[yy][xx] and not outside[yy][xx]:
                outside[yy][xx] = True
                stack.append((yy, xx))
    return ~np.array(outside, dtype=bool)[1:-1, 1:-1]


def _is_thin(cells: list[Cell]) -> bool:
    s = set(cells)
    return all(sum((y + dy, x + dx) in s for dy, dx in NB8) <= 2 for y, x in cells)


def _norm(cells: list[Cell]) -> list[Cell]:
    y0, x0 = min(c[0] for c in cells), min(c[1] for c in cells)
    return sorted((y - y0, x - x0) for y, x in cells)


def build_layout(g: np.ndarray) -> Layout:
    """The lab's ``scene.build_scene`` on a grid. Terrain is every colour above 8% of the frame; the rest is cut into
    one-colour pieces, which become lines, bars, composites (touching pieces of different colours), icons (pieces of
    one colour set inside one panel) and dotted figures (three or more tiny pieces close together)."""
    H, W = g.shape
    counts = Counter(g.ravel().tolist())
    terrain = [c for c, n in counts.most_common() if n / g.size > 0.08]
    fg = ~np.isin(g, terrain)
    pieces = [(col, cells) for col in sorted(set(g[fg].tolist())) for cells in _components((g == col) & fg)]

    # thin long pieces: tethers (lines) and screen-edge bars; long solid strips are bars too
    lines, bars, rest = [], [], []
    for col, cells in pieces:
        r0, c0, r1, c1 = _bbox(cells)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        on_edge = r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1
        if len(cells) >= 5 and _is_thin(cells):
            (bars if (h == 1 or w == 1) and on_edge else lines).append(cells)
        elif max(h, w) >= 6 * min(h, w) and max(h, w) >= 8 and len(cells) == h * w:
            bars.append(cells)
        else:
            rest.append((col, cells))

    # composites: pieces of different colours that touch (8-neighbours) are one thing
    owner = {cell: i for i, (_, cells) in enumerate(rest) for cell in cells}
    parent = list(range(len(rest)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (y, x), i in owner.items():
        for dy, dx in NB8:
            j = owner.get((y + dy, x + dx))
            if j is not None and j != i and rest[i][0] != rest[j][0]:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(rest)):
        groups.setdefault(find(i), []).append(i)
    comps = [[c for i in idx for c in rest[i][1]] for idx in groups.values()]

    # panels: compact terrain boxes that are not the main field (HUD boxes, lock boxes)
    panels: list[Panel] = []
    for col in terrain:
        for cells in _components(g == col, NB4):
            r0, c0, r1, c1 = _bbox(cells)
            box = (r1 - r0 + 1) * (c1 - c0 + 1)
            if box > 0.12 * g.size or min(r1 - r0, c1 - c0) < 2:
                continue
            m = np.zeros((r1 - r0 + 1, c1 - c0 + 1), bool)
            for y, x in cells:
                m[y - r0, x - c0] = True
            if _fill_holes(m).sum() / box >= 0.9:
                panels.append(Panel(col, r0, c0, r1, c1, r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1))

    def panel_of(cells: list[Cell]) -> int | None:
        for k, p in enumerate(panels):
            if all(p.r0 <= y <= p.r1 and p.c0 <= x <= p.c1 for y, x in cells):
                return k
        return None

    # icons: pieces with the same colours inside the same panel are one thing, unless they are identical copies
    raw = [(cells, panel_of(cells)) for cells in comps]
    colours = [{int(g[y, x]) for y, x in cells} for cells, _ in raw]
    used: set[int] = set()
    merged: list[tuple[list[Cell], int | None, str]] = []
    for i, (ci, pi) in enumerate(raw):
        if i in used:
            continue
        grp = [i]
        if pi is not None:
            for j in range(i + 1, len(raw)):
                cj, pj = raw[j]
                if j in used or pj != pi or colours[j] != colours[i]:
                    continue
                if len(ci) == len(cj) and _norm(ci) == _norm(cj):
                    continue   # an identical copy stays separate: a counter, lives
                grp.append(j)
        used.update(grp)
        merged.append(([c for k in grp for c in raw[k][0]], pi, "icon" if len(grp) > 1 else "object"))

    # dotted figures: three or more tiny thin pieces of the same colours, at most 2 cells apart, outside panels
    tiny = [k for k, (cells, pi, _) in enumerate(merged) if len(cells) <= 4 and pi is None and _is_thin(cells)]
    dgroups: dict[int, list[int]] = {}
    if len(tiny) <= MAX_TINY:
        up = {k: k for k in tiny}

        def root(a: int) -> int:
            while up[a] != a:
                a = up[a]
            return a

        tcols = {k: {int(g[y, x]) for y, x in merged[k][0]} for k in tiny}
        for ia, a in enumerate(tiny):
            for b in tiny[ia + 1:]:
                if tcols[a] == tcols[b] and min(max(abs(y1 - y2), abs(x1 - x2)) for y1, x1 in merged[a][0]
                                                for y2, x2 in merged[b][0]) <= 2:
                    up[root(a)] = root(b)
        for k in tiny:
            dgroups.setdefault(root(k), []).append(k)
    final: list[tuple[list[Cell], int | None, str]] = []
    consumed: set[int] = set()
    for ks in dgroups.values():
        if len(ks) >= 3:
            final.append(([c for k in ks for c in merged[k][0]], None, "dotted"))
            consumed.update(ks)
    final += [item for k, item in enumerate(merged) if k not in consumed]
    final += [(cells, None, "line") for cells in lines]
    final += [(cells, panel_of(cells), "bar") for cells in bars]

    terr = np.isin(g, terrain)
    things = []
    for cells, pi, kind in final:
        r0, c0, r1, c1 = _bbox(cells)
        t = Thing(kind, cells, Counter(int(g[y, x]) for y, x in cells), r0, c0, r1, c1, panel=pi)
        # the terrain colour around it: the most common terrain cell in its box grown by one, itself left out
        ya, yb, xa, xb = max(r0 - 1, 0), min(r1 + 2, H), max(c0 - 1, 0), min(c1 + 2, W)
        around = terr[ya:yb, xa:xb].copy()
        for y, x in cells:
            around[y - ya, x - xa] = False
        ring = Counter(g[ya:yb, xa:xb][around].tolist())
        t.on = ring.most_common(1)[0][0] if ring else -1
        things.append(t)
    things.sort(key=lambda t: (t.r0, t.c0))
    for t in things:
        t.name = f"{_colour_phrase(t, g)} {_shape_word(t.mask(), t.kind)}"
    return Layout(g, terrain, things, panels)


def _shape_word(m: np.ndarray, kind: str) -> str:
    h, w = m.shape
    n = int(m.sum())
    if kind == "line":
        return "dotted line" if n < max(h, w) else "line"
    if kind == "bar":
        return "bar"
    if kind == "dotted":
        ys, xs = np.nonzero(m)
        d = np.hypot(ys - ys.mean(), xs - xs.mean())
        return "dotted ring" if d.mean() >= 2 and d.std() / d.mean() < 0.25 else "dotted cluster"
    if n == h * w:
        if max(h, w) >= 4 * min(h, w):
            return "bar"
        return "square" if h == w else "rectangle"
    if h == w and h % 2 == 1:
        r = h // 2
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        if np.array_equal(m, (yy == 0) | (xx == 0)):
            return "plus"
        if np.array_equal(m, (np.abs(yy) + np.abs(xx)) <= r):
            return "diamond"
        if np.array_equal(m, (np.abs(yy) + np.abs(xx) <= r) & ~((yy == 0) & (xx == 0))):
            return "hollow diamond"
        corners_cut = m.copy()
        corners_cut[[0, 0, -1, -1], [0, -1, 0, -1]] = True
        if corners_cut.all():
            return "disc"
    holes = int(_fill_holes(m).sum() - n)
    if holes > 0:
        return "ring" if holes >= n // 4 else f"shape with {holes} hole cell(s)"
    if n <= 8:
        return {3: "L-tromino", 4: "tetromino", 5: "pentomino"}.get(n, f"{n}-cell shape")
    return f"irregular shape ({n} cells)"


def _colour_phrase(t: Thing, g: np.ndarray) -> str:
    cg = t.colgrid(g)
    cols = [c for c, _ in t.colors.most_common()]
    if len(cols) == 1:
        return NAMES[cols[0]]
    h, w = cg.shape
    top, bot = cg[: h // 2], cg[(h + 1) // 2:]
    tc, bc = set(top[top >= 0].ravel().tolist()), set(bot[bot >= 0].ravel().tolist())
    rows_pure = all(len(set(row[row >= 0].tolist())) <= 1 for row in cg)
    if rows_pure and len(tc) == 1 and len(bc) == 1 and tc != bc:
        return f"{NAMES[tc.pop()]} top / {NAMES[bc.pop()]} bottom"
    if h % 2 and w % 2 and cg[h // 2, w // 2] >= 0 and t.colors[int(cg[h // 2, w // 2])] == 1:
        centre = int(cg[h // 2, w // 2])
        return f"{'/'.join(NAMES[c] for c in cols if c != centre)} with {NAMES[centre]} centre"
    cy, cx = (h - 1) / 2, (w - 1) / 2   # where each colour sits inside the thing, in compass words
    parts = []
    for c in cols:
        places = set()
        for y, x in zip(*np.nonzero(cg == c)):
            dy, dx = y - cy, x - cx
            if abs(dy) < 0.5 and abs(dx) < 0.5:
                places.add("centre")
            elif abs(dy) >= abs(dx):
                places.add("top" if dy < 0 else "bottom")
            else:
                places.add("left" if dx < 0 else "right")
        where = ", ".join(p for p in ("top", "right", "bottom", "left", "centre") if p in places)
        parts.append(f"{NAMES[c]} ({where})")
    return " + ".join(parts)


def _is_hud(t: Thing, lay: Layout) -> bool:
    H, W = lay.g.shape
    if t.kind == "bar":
        return True
    if t.panel is not None:
        p = lay.panels[t.panel]
        if p.r0 <= 2 or p.c0 <= 2 or p.r1 >= H - 3 or p.c1 >= W - 3:
            return True
    return t.r0 <= 1 or t.c0 <= 1 or t.r1 >= H - 2 or t.c1 >= W - 2


Lattice = tuple[int, int, int, int]   # tile size, row offset, column offset, floor colour


def _lattice(lay: Layout, hint: Lattice | None = None) -> Lattice | None:
    """The tile the floor is built from: the size of the largest compact multi-colour square, if more than 75% of the
    floor's terrain edges fall on its grid lines. ``hint`` (the lattice of the level's previous frame) is kept when no
    square qualifies but the edges still fit it: the self stops looking like a tile while it touches another thing.
    Status boxes (boxes on the screen edge with something in them) are left out: their gauges change every step."""
    g = lay.g
    terr = np.isin(g, lay.terrain)
    for k in {t.panel for t in lay.things if t.panel is not None}:
        p = lay.panels[k]
        if p.border:
            terr[p.r0:p.r1 + 1, p.c0:p.c1 + 1] = False

    def fits(u: int, ro: int, co: int, fl: int) -> bool:
        a, b = g[:, :-1], g[:, 1:]          # left and right neighbours; b sits at column 1..W-1
        edge = (a != b) & ((a == fl) | (b == fl)) & terr[:, :-1] & terr[:, 1:]
        hits = int((edge & ((np.arange(1, g.shape[1]) - co) % u == 0)[None, :]).sum())
        tot = int(edge.sum())
        a, b = g[:-1, :], g[1:, :]          # upper and lower neighbours; b sits at row 1..H-1
        edge = (a != b) & ((a == fl) | (b == fl)) & terr[:-1, :] & terr[1:, :]
        hits += int((edge & ((np.arange(1, g.shape[0]) - ro) % u == 0)[:, None]).sum())
        tot += int(edge.sum())
        return bool(tot) and hits / tot > 0.75

    cands = [t for t in lay.things if t.kind == "object" and t.h == t.w and t.h >= 3 and len(t.colors) >= 2]
    for t in sorted(cands, key=lambda t: -t.size):
        if fits(t.h, t.r0 % t.h, t.c0 % t.h, t.on):
            return t.h, t.r0 % t.h, t.c0 % t.h, t.on
    return hint if hint is not None and fits(*hint) else None


def _gap(t: Thing, cell: Cell) -> int:
    """Steps from ``cell`` to the box of ``t`` (0 inside it)."""
    y, x = cell
    return max(t.r0 - y, 0, y - t.r1) + max(t.c0 - x, 0, x - t.c1)


def _ends(line: Thing) -> list[Cell]:
    s = set(line.cells)
    return [c for c in line.cells if sum((c[0] + dy, c[1] + dx) in s for dy, dx in NB8) <= 1]


# --- MEASURED facts (X3 visual routines, X4 look-alikes, X2 broken symmetry) -----------------------------------------

def _reach(lay: Layout, me: Thing | None, lattice: Lattice | None) -> list[str]:
    """R1: from the most self-like thing, where can it go? In tile steps when the floor has a tile grid; otherwise
    one open space, straight paths between the main objects, and the walls."""
    if me is None:
        return []
    g = lay.g
    H, W = g.shape
    floor = me.on
    occ = np.zeros(g.shape, bool)
    for t in lay.things:
        if not t.hud:
            for y, x in t.cells:
                occ[y, x] = True
    walk = (g == floor) | occ
    if lattice:
        u = lattice[0]
        start = (me.r0, me.c0)
        dist = {start: 0}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-u, 0), (u, 0), (0, -u), (0, u)):
                nb = (r + dr, c + dc)
                if nb not in dist and 0 <= nb[0] <= H - u and 0 <= nb[1] <= W - u \
                        and walk[nb[0]:nb[0] + u, nb[1]:nb[1] + u].all():
                    dist[nb] = dist[(r, c)] + 1
                    queue.append(nb)
        steps = np.full(g.shape, -1)
        for (r, c), k in sorted(dist.items(), key=lambda kv: -kv[1]):   # nearer tiles overwrite farther ones
            steps[r:r + u, c:c + u] = k
        facts = [f"Floor is a {u}x{u} tile grid."]
        for t in lay.things:
            if t is me or t.kind == "line":
                continue
            ks = steps[[y for y, _ in t.cells], [x for _, x in t.cells]]
            ks = ks[ks >= 0]
            if ks.size:
                facts.append(f"{t.ref}: reachable in {int(ks.min())} steps")
            elif not t.hud:
                p = lay.panels[t.panel] if t.panel is not None else None
                door = [(k, (r, c)) for (r, c), k in dist.items() if p is not None and _touches(p, r, c, u)]
                if door:
                    k, (r, c) = min(door)
                    facts.append(f"{t.ref}: sits inside a {NAMES[p.color]} box that is NOT plain floor; the floor "
                                 f"reaches the box's {_side(p, r, c, u)} side in {k} steps (the only way in)")
                else:
                    facts.append(f"{t.ref}: NOT reachable on the floor")
        return facts
    where = f"{NAMES[floor]} ({(g == floor).mean():.0%} of screen)" if floor >= 0 else "unknown"
    facts = [f"no tile grid -> free movement. Open space = {where}."]
    sal = [t for t in lay.things if not t.hud and t.kind != "line" and t.size >= 9]
    if len(sal) >= 2:
        area = np.full(g.shape, -1)
        for k, cells in enumerate(_components(walk, NB4)):
            for y, x in cells:
                area[y, x] = k
        n_areas = len({int(area[t.cells[0]]) for t in sal})
        checked = sorted(sorted(sal, key=lambda t: -t.size)[:MAX_PATH_THINGS], key=lambda t: (t.r0, t.c0))
        rock = (g != floor) & ~occ
        blocked = []
        for i, a in enumerate(checked):
            for b in checked[i + 1:]:
                n = _rock_on_path(rock, a, b)
                if n:
                    blocked.append(f"{a.ref} -> {b.ref} ({n} rock cells)")
        space = (f"all {len(sal)} main objects share one open space" if n_areas == 1
                 else f"the {len(sal)} main objects are in {n_areas} separate open areas")
        paths = "ALL CLEAR of rock" if not blocked else "clear except: " + "; ".join(blocked[:6]) + (
            f"; +{len(blocked) - 6} more" if len(blocked) > 6 else "")
        facts.append(f"{space}; straight paths between them are {paths}")
    walls = [c for c in lay.terrain if c != floor]
    if walls:
        facts.append(f"the {', '.join(NAMES[c] for c in walls)} areas are not open floor (likely walls)")
    return facts


def _touches(p: Panel, r: int, c: int, u: int) -> bool:
    """The tile (r, c) of size u is within one tile of the box and overlaps it on the other axis."""
    gap_r = max(p.r0 - (r + u - 1), r - p.r1, 0)
    gap_c = max(p.c0 - (c + u - 1), c - p.c1, 0)
    return (gap_r <= u and gap_c == 0) or (gap_c <= u and gap_r == 0)


def _side(p: Panel, r: int, c: int, u: int) -> str:
    if r >= p.r1:
        return "bottom"
    if r + u <= p.r0:
        return "top"
    return "left" if c + u <= p.c0 else "right"


def _rock_on_path(rock: np.ndarray, a: Thing, b: Thing) -> int:
    """Rock cells swept by the smaller footprint of the two moving in a straight line from ``a`` to ``b``."""
    H, W = rock.shape
    hh, ww = min(a.h, b.h), min(a.w, b.w)
    ay, ax, by, bx = a.cy, a.cx, b.cy, b.cx
    hit = np.zeros(rock.shape, bool)
    for t in np.linspace(0, 1, 60):
        cy, cx = ay + (by - ay) * t, ax + (bx - ax) * t
        r0, c0 = int(round(cy - hh / 2 + 0.5)), int(round(cx - ww / 2 + 0.5))
        ys = slice(min(max(r0, 0), H), min(max(r0 + hh, 0), H))
        xs = slice(min(max(c0, 0), W), min(max(c0 + ww, 0), W))
        hit[ys, xs] |= rock[ys, xs]
    return int(hit.sum())


def _enclosures(lay: Layout) -> list[str]:
    """R2: hollow things (and dotted figures) and the space inside them; which things would fit it exactly."""
    owner = np.full(lay.g.shape, -1)
    for k, t in enumerate(lay.things):
        for y, x in t.cells:
            owner[y, x] = k
    facts = []
    for k, t in enumerate(lay.things):
        m = t.mask()
        if t.h - 2 < 2 or not (t.kind == "dotted" or _fill_holes(m).sum() > m.sum()):
            continue
        ih, iw = t.h - 2, t.w - 2
        inside = owner[t.r0 + 1:t.r1, t.c0 + 1:t.c1]
        empty = not ((inside >= 0) & (inside != k)).any()
        fits = [q.ref for q in lay.things if q is not t and q.kind == "object" and q.h == ih and q.w == iw]
        facts.append(f"{t.ref} encloses {'an EMPTY' if empty else 'a'} {ih}x{iw} space"
                     + (f" -- exactly the size of: {', '.join(fits)}" if fits else ""))
    return facts


def _tethers(lay: Layout) -> tuple[list[str], list[int]]:
    """R3: thin lines and the two things at their ends; a thing tied to exactly two others that sits at their
    midpoint is held between them. Returns the facts and the held things (indices)."""
    solids = [t for t in lay.things if t.kind != "line"]
    index = {id(t): k for k, t in enumerate(lay.things)}
    facts, links = [], []
    for ln in [t for t in lay.things if t.kind == "line"]:
        ends = _ends(ln)[:2]
        if len(ends) < 2 or not solids:
            continue
        a, b = (min(solids, key=lambda o: _gap(o, e)) for e in ends)
        if a is not b:
            links.append((a, b))
            facts.append(f"a {ln.name} connects {a.ref} to {b.ref}")
    tied: dict[int, list[Thing]] = {}
    for a, b in links:
        tied.setdefault(index[id(a)], []).append(b)
        tied.setdefault(index[id(b)], []).append(a)
    held = []
    for k, others in tied.items():
        if len(others) != 2:
            continue
        o, (a, b) = lay.things[k], others
        err = math.hypot(o.cy - (a.cy + b.cy) / 2, o.cx - (a.cx + b.cx) / 2)
        if err <= 1.0:
            held.append(k)
            facts.append(f"{o.ref} sits at the MIDPOINT of {a.ref} and {b.ref} (error {err:.1f} cells) -> it is held "
                         "between them; moving either end should move it to the new midpoint")
    return facts, held


def _alignments(lay: Layout, first: set[int]) -> list[str]:
    """R4: pairs of main things that share columns (one above the other) or rows (side by side). Pairs with a thing
    in ``first`` (the likely self and goals) come first."""
    sal = [k for k, t in enumerate(lay.things) if not t.hud and t.kind != "line"]
    found = []
    for n, i in enumerate(sal):
        for j in sal[n + 1:]:
            a, b = lay.things[i], lay.things[j]
            colov = min(a.c1, b.c1) - max(a.c0, b.c0) + 1
            rowov = min(a.r1, b.r1) - max(a.r0, b.r0) + 1
            if colov > 0 and rowov > 0:
                continue   # their boxes overlap: one is on or in the other, not beside it
            if colov >= 0.6 * min(a.w, b.w):
                rel = "directly above" if a.cy < b.cy else "directly below"
                text = f"{a.ref} is {rel} {b.ref} (shared columns {max(a.c0, b.c0)}-{min(a.c1, b.c1)})"
            elif rowov >= 0.6 * min(a.h, b.h):
                rel = "left of" if a.cx < b.cx else "right of"
                text = f"{a.ref} is level with and {rel} {b.ref} (shared rows {max(a.r0, b.r0)}-{min(a.r1, b.r1)})"
            else:
                continue
            found.append((not ({i, j} & first), text))
    return [text for _, text in sorted(found, key=lambda f: f[0])]


def _downsample(m: np.ndarray, s: int) -> np.ndarray | None:
    h, w = m.shape
    if h % s or w % s:
        return None
    blocks = m.reshape(h // s, s, w // s, s)
    if not (blocks.all(axis=(1, 3)) | ~blocks.any(axis=(1, 3))).all():
        return None
    return blocks.all(axis=(1, 3))


def _d4(m: np.ndarray) -> list[tuple[str, np.ndarray]]:
    out = []
    for k, nm in enumerate(("", "rot90ccw", "rot180", "rot90cw")):
        r = np.rot90(m, k)
        out.append((nm or "same", r))
        out.append(((nm + "+" if nm else "") + "flipLR", np.fliplr(r)))
    return out


@dataclass
class Likeness:
    """How two things look alike (X4). EXACT: same shape and colours. STATE: same shape, other colours. TURNED: the
    big one, turned, mirrored or drawn ``scale`` times bigger (``how``), is exactly the small one."""
    kind: str
    big: int
    small: int
    scale: int
    how: tuple[str, ...]


def _relate(things: list[Thing], i: int, j: int) -> Likeness | None:
    a, b = things[i], things[j]
    big, small = (i, j) if a.mask().size >= b.mask().size else (j, i)
    mb, ms = things[big].mask(), things[small].mask()
    for s in (1, 2, 3):
        dm = mb if s == 1 else _downsample(mb, s)
        if dm is None or dm.shape not in (ms.shape, ms.shape[::-1]):
            continue
        matches = tuple(nm for nm, t in _d4(dm) if t.shape == ms.shape and np.array_equal(t, ms))
        if not matches:
            continue
        if "same" in matches:
            kind = ("EXACT" if set(a.colors) == set(b.colors) else "STATE") if s == 1 else "TURNED"
            return Likeness(kind, big, small, s, ("same",))
        return Likeness("TURNED", big, small, s, matches)
    return None


def _ids(things: list[Thing], ks: list[int]) -> str:
    """``obj 3, 5, 7``."""
    return "obj " + ", ".join(str(things[k].id) if things[k].id is not None else "?" for k in ks)


def _group(things: list[Thing], ks: list[int]) -> str:
    """One thing by its ref; several as ``obj 3, 5, 7 (red square; blue square)``."""
    if len(ks) == 1:
        return things[ks[0]].ref
    shorts = list(dict.fromkeys(things[k].short for k in ks))
    return f"{_ids(things, ks)} ({'; '.join(shorts[:4])}{'; ...' if len(shorts) > 4 else ''})"


def _same_shapes(things: list[Thing], ks: list[int]) -> tuple[list[str], list[str], list[list[int]]]:
    """X4's EXACT and STATE pairs, grouped by shape: things drawn exactly alike are copies (a count); one shape in
    different colours is one kind of thing in different states. Returns (copy facts, state facts, state groups)."""
    by_shape: dict[tuple, list[int]] = {}
    for k in ks:
        m = things[k].mask()
        by_shape.setdefault((m.shape, m.tobytes()), []).append(k)
    copies, states, groups = [], [], []
    for members in by_shape.values():
        if len(members) < 2:
            continue
        by_colour: dict[frozenset, list[int]] = {}
        for k in members:
            by_colour.setdefault(frozenset(things[k].colors), []).append(k)
        for same in by_colour.values():
            if len(same) >= 2:
                t = things[same[0]]
                copies.append(f"{len(same)} identical {t.name} ({t.h}x{t.w}): {_ids(things, same)} -> a count of "
                              "something (lives, items, tiles)")
        if len(by_colour) >= 2:
            looks = " vs ".join("/".join(NAMES[c] for c, _ in things[g[0]].colors.most_common())
                                for g in by_colour.values())
            states.append(f"{_group(things, members)} have the same shape in different colours ({looks})")
            groups.append(members)
    return copies, states, groups


def _turned(things: list[Thing], rel: dict[tuple[int, int], Likeness], ks: set[int]) -> tuple[list[str],
                                                                                              list[list[int]]]:
    """X4's TURNED pairs, grouped by the two shapes and the turn: "obj 9, 10 are obj 3, 4 drawn 2x bigger"."""
    found: dict[tuple, tuple[list[int], list[int]]] = {}
    for (i, j), lk in rel.items():
        if lk.kind != "TURNED" or i not in ks or j not in ks:
            continue
        mb, ms = things[lk.big].mask(), things[lk.small].mask()
        bigs, smalls = found.setdefault((mb.shape, mb.tobytes(), ms.shape, ms.tobytes(), lk.scale, lk.how), ([], []))
        bigs += [lk.big] if lk.big not in bigs else []
        smalls += [lk.small] if lk.small not in smalls else []
    facts, groups = [], []
    for (*_, s, how), (bigs, smalls) in found.items():
        big, small, one = _group(things, bigs), _group(things, smalls), len(bigs) == 1
        if how == ("same",):
            facts.append(f"{big} {'is' if one else 'are'} {small} drawn {s}x bigger")
        else:
            scale = f" (drawn {s}x bigger)" if s > 1 else ""
            facts.append(f"{big}{scale}, {' OR '.join(TURN_WORDS[m] for m in how)}, "
                         f"{'becomes' if one else 'become'} exactly {small}")
        groups.append(bigs + smalls)
    return facts, groups


def _symmetry_break(t: Thing, g: np.ndarray) -> bool:
    """X2: the shape is symmetric but its colours are not (designers use this to mark a direction or a state)."""
    if len(t.colors) < 2:
        return False
    m, cg = t.mask(), t.colgrid(g)
    ops = [np.fliplr, np.flipud] + ([np.rot90] if m.shape[0] == m.shape[1] else [])
    return any(np.array_equal(f(m), m) and not np.array_equal(f(cg), cg) for f in ops)


# --- GUESSES (X7 role ledger) ---------------------------------------------------------------------------------------

def _roles(lay: Layout, rel: dict[tuple[int, int], Likeness], me: Thing | None,
           lattice: Lattice | None, keyboard: bool, clicking: bool) -> dict[int, tuple[str, float]]:
    """X7: each thing is scored against fixed roles from game-design priors and the facts above; a thing's top role
    and its share of the score. Only things with some evidence get a role."""
    g, things = lay.g, lay.things
    objs = [k for k, t in enumerate(things) if t.kind != "line"]
    score: dict[int, dict[str, float]] = {k: defaultdict(float) for k in objs}

    def add(k: int, role: str, pts: float) -> None:
        score[k][role] += pts

    hud_copies = Counter((things[k].name, things[k].h, things[k].w) for k in objs if things[k].hud)
    for k in objs:
        t = things[k]
        copies = hud_copies[(t.name, t.h, t.w)]
        if t.kind == "bar" and t.hud:
            add(k, "MOVES LEFT", 3)                       # a long bar at the screen edge is a gauge
        if t.hud and copies >= 2:
            add(k, "LIVES", 3)                            # identical copies at the screen edge are a count
        if t.hud and t.kind in ("icon", "object") and copies == 1:
            add(k, "INVENTORY", 2)                        # a single icon in a status box
        if not t.hud and keyboard and t is me:
            add(k, "YOU", 3)                              # compact, multi-coloured, unique, on the floor
            if lattice and lattice[0] == t.h:
                add(k, "YOU", 1)                          # as big as a floor tile: it walks tile by tile
        if not t.hud and t.size <= 12 and t is not me and _symmetry_break(t, g):
            add(k, "MODIFIER", 2)                         # a small symbol with lopsided colours
        if t.kind == "dotted" or (_fill_holes(t.mask()).sum() > t.size and not t.hud):
            if any(things[q].h == t.h - 2 and things[q].w == t.w - 2 for q in objs if q != k):
                add(k, "GOAL", 3)                         # a hollow outline that fits another thing exactly
        if t.panel is not None and not t.hud:
            add(k, "GOAL", 1)                             # shown inside a special box in the play area
    for (i, j), lk in rel.items():
        kind = lk.kind
        if kind == "TURNED":
            for x, y in ((i, j), (j, i)):
                if things[x].hud and not things[y].hud:
                    add(x, "INVENTORY", 2)                # a HUD icon shaped like a thing in the play area ...
                    add(y, "GOAL", 2)                     # ... which asks for that item, in this orientation
        elif kind == "STATE":
            for x in (i, j):
                add(x, "HANDLE (click)" if clicking else "SCENERY", 2)   # an on/off pair
    lines = [t for t in things if t.kind == "line"]
    if clicking and lines and objs:
        ends: Counter = Counter()
        for ln in lines:
            for e in _ends(ln):
                ends[min(objs, key=lambda q: _gap(things[q], e))] += 1
        for k, n in ends.items():
            if n >= 2:
                add(k, "CARRIED THING", 3)                # both tethers end on it: it hangs between the anchors
            else:
                add(k, "HANDLE (click)", 2)               # a tether starts here: an anchor you can grab
        for k in objs:
            if score[k].get("GOAL", 0) >= 3:
                body = [q for q in objs if score[q].get("CARRIED THING")]
                if body and set(things[k].colors) & set(things[body[0]].colors):
                    add(k, "GOAL", 1)                     # the same colour as the carried thing
    roles = {}
    for k in objs:
        if score[k]:
            role, pts = max(score[k].items(), key=lambda kv: kv[1])
            roles[k] = (role, pts / (sum(score[k].values()) + 1.0))
    return roles


FACT_KINDS = ("reach", "inside", "tether", "align", "alike", "copies", "lopsided")


def _share(sizes: list[int], room: int) -> list[int]:
    """Lines per fact kind: one each in turn until ``room`` is used, so one busy kind cannot crowd out the rest."""
    quota = [0] * len(sizes)
    while room > 0 and any(q < n for q, n in zip(quota, sizes)):
        for i, n in enumerate(sizes):
            if room > 0 and quota[i] < n:
                quota[i] += 1
                room -= 1
    return quota


@dataclass
class Briefing:
    """The briefing analysis of one frame: the text tiers and what the picture draws."""
    controls: str                                   # "arrow keys", "clicks", "arrow keys and clicks", ...
    facts: dict[str, list[str]]                     # MEASURED by kind (``FACT_KINDS``): computed, true by construction
    guesses: list[str]                              # GUESSES: one line per role, each with a one-action test
    layout: Layout
    floor: int
    lattice: Lattice | None
    roles: dict[int, tuple[str, float]]             # thing index -> (role, share of its score)
    threads: list[tuple[int, int, str]]             # look-alikes to draw: (thing, thing, TURNED | STATE)
    held: list[int]                                 # things held at the midpoint of two others

    def lines(self, limit: int | None = None) -> list[str]:
        """The briefing as text, MEASURED first. With ``limit``, at most that many fact and guess lines: up to a
        third for guesses, the rest shared fairly between the fact kinds; what is left out is counted."""
        kinds = [k for k in FACT_KINDS if self.facts.get(k)]
        sizes = [len(self.facts[k]) for k in kinds]
        quota, ng = sizes, len(self.guesses)
        if limit is not None:
            ng = min(ng, limit // 3)
            quota = _share(sizes, limit - ng)
            ng = min(len(self.guesses), limit - sum(quota))
        out = [f"BRIEFING ({self.controls})", "MEASURED (certain):"]
        for k, q in zip(kinds, quota):
            out += [f"- {f}" for f in self.facts[k][:q]]
        if not kinds:
            out.append("- (nothing measured)")
        if sum(quota) < sum(sizes):
            out.append(f"- (+{sum(sizes) - sum(quota)} more facts: `observe().briefing` in ipython)")
        out.append("GUESSES (check with one action):")
        out += [f"- {x}" for x in self.guesses[:ng]] or ["- (none)"]
        if ng < len(self.guesses):
            out.append(f"- (+{len(self.guesses) - ng} more guesses: `observe().briefing` in ipython)")
        return out


def brief(grid: Any, objects: list[Obj], actions: Any = (1, 2, 3, 4), prev: Briefing | None = None) -> Briefing:
    """The briefing of ``grid``. ``objects`` are the frame's object list (things are named by its ids);
    ``actions`` are the legal action ids (1-4 arrow keys, 6 click), which decide the YOU and HANDLE guesses. ``prev``
    is the briefing of the level's previous frame: its tile grid is kept while it still fits."""
    g = np.array(rows_of(grid), dtype=np.int16)
    lay = build_layout(g)
    things = lay.things
    owner = {cell: o.id for o in objects for cell in o.cells}
    for k, t in enumerate(things):
        t.hud = _is_hud(t, lay)
        ids = Counter(owner[c] for c in t.cells if c in owner)
        t.id = ids.most_common(1)[0][0] if ids else None
        t.short = f"{'/'.join(NAMES[c] for c, _ in t.colors.most_common(2))} {_shape_word(t.mask(), t.kind)}"
        t.ref = f"{f'obj {t.id}' if t.id is not None else f'thing {k + 1}'} ({t.short})"
    actions = {int(a) for a in actions}
    keyboard, clicking = bool(actions & {1, 2, 3, 4}), 6 in actions
    cands = sorted((t for t in things if t.kind == "object" and not t.hud and len(t.colors) >= 2 and t.size >= 9),
                   key=lambda t: (-(t.h == t.w), -t.size))
    me = cands[0] if cands else None
    lattice = _lattice(lay, prev.lattice if prev is not None else None)
    world = [t for t in things if not t.hud]
    floor = me.on if me else (max(lay.terrain, key=lambda c: sum(t.on == c for t in world)) if lay.terrain
                              else background(rows_of(grid)))

    # look-alikes among the largest things (pairs grow as n^2), then the role guesses that use them
    pool = [k for k, t in enumerate(things) if t.kind != "line"]
    if len(pool) > MAX_PAIR_THINGS:
        pool = sorted(sorted(pool, key=lambda k: -things[k].size)[:MAX_PAIR_THINGS])
    rel = {}
    for n, i in enumerate(pool):
        for j in pool[n + 1:]:
            lk = _relate(things, i, j)
            if lk:
                rel[(i, j)] = lk
    roles = _roles(lay, rel, me, lattice, keyboard, clicking)
    by_role: dict[str, list[int]] = {}
    for k in sorted(roles, key=lambda k: -roles[k][1]):
        by_role.setdefault(roles[k][0], []).append(k)
    guesses = []
    for role, ks in by_role.items():
        lo, hi = (f"{p:.0%}" for p in (roles[ks[-1]][1], roles[ks[0]][1]))
        pct = lo if lo == hi else f"{lo[:-1]}-{hi}"
        guesses.append(f"{_group(things, ks)} = {role} ({pct}) - test: {ROLE_TESTS[role]}")

    x4 = [k for k, t in enumerate(things) if t.kind in ("object", "icon", "dotted") and t.size >= 2]
    copies, states, state_groups = _same_shapes(things, x4)
    turned, turned_groups = _turned(things, rel, set(x4))
    tether_facts, held = _tethers(lay)
    key = {k for k, (role, _) in roles.items() if role in ("YOU", "GOAL")} | ({things.index(me)} if me else set())
    facts = {"reach": _reach(lay, me, lattice), "inside": _enclosures(lay), "tether": tether_facts,
             "align": _alignments(lay, key), "alike": turned + states, "copies": copies,
             "lopsided": [f"{t.ref} has a symmetric shape but lopsided colours (marks a direction/state)"
                          for t in things if t.kind != "line" and not t.hud and _symmetry_break(t, g)]}
    threads = []   # each look-alike group drawn as one chain, in reading order: n - 1 threads, not n^2
    for kind, groups in (("TURNED", turned_groups), ("STATE", state_groups)):
        for grp in groups:
            order = sorted(grp, key=lambda k: (things[k].r0, things[k].c0))
            threads += [(a, b, kind) for a, b in zip(order, order[1:])]
    controls = " and ".join(w for w, on in (("arrow keys", keyboard), ("clicks", clicking)) if on) or "other keys"
    return Briefing(controls, {k: list(dict.fromkeys(v)) for k, v in facts.items()}, guesses, lay, int(floor),
                    lattice, roles, threads, held)


# =====================================================================================================================
# 4. The picture: the map and HUD panel, without the briefing text
# =====================================================================================================================

PICTURE_TITLE = "Map: number = object id, ? = guessed role"
_DRAW_LOCK = threading.Lock()   # PIL fonts share one FreeType face: draw from one game thread at a time
_FONT_FILES = {False: ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"),
               True: ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf")}


@lru_cache(maxsize=None)
def _font(size: int, bold: bool = False) -> Any:
    """DejaVu (Linux, or the copy inside matplotlib), Arial on Windows, else Pillow's own font."""
    paths = list(_FONT_FILES[bold])
    spec = importlib.util.find_spec("matplotlib")   # finds the package without importing it
    if spec and spec.origin:
        paths.append(str(Path(spec.origin).parent / "mpl-data" / "fonts" / "ttf" /
                         ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")))
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _caption(text: str, width: int, size: int = 15) -> Image.Image:
    img = Image.new("RGB", (width, size + 20), (250, 250, 247))
    ImageDraw.Draw(img).text((10, 10), text, fill=(20, 20, 20), font=_font(size + 1, True))
    return img


def _titled(img: Image.Image, title: str, size: int = 16) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + size + 12), (250, 250, 247))
    ImageDraw.Draw(out).text((4, 4), title, fill=(20, 20, 20), font=_font(size, True))
    out.paste(img, (0, size + 12))
    return out


def _stack(imgs: list[Image.Image], across: bool, gap: int = 12) -> Image.Image:
    if across:
        size = (sum(i.width for i in imgs) + gap * (len(imgs) - 1), max(i.height for i in imgs))
    else:
        size = (max(i.width for i in imgs), sum(i.height for i in imgs) + gap * (len(imgs) - 1))
    out = Image.new("RGB", size, (250, 250, 247))
    pos = 0
    for i in imgs:
        out.paste(i, (pos, 0) if across else (0, pos))
        pos += (i.width if across else i.height) + gap
    return out


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render(grid: Any, cell: int = 4) -> bytes:
    """The plain frame as a PNG, each grid cell a ``cell``-pixel square."""
    rows = rows_of(grid)
    if not rows or not rows[0]:
        raise ValueError("cannot render an empty grid")
    rgb = np.array(PALETTE, np.uint8)[np.array(rows, dtype=np.int64) & 15]
    k = max(1, int(cell))
    return _png(Image.fromarray(np.repeat(np.repeat(rgb, k, 0), k, 1)))


def picture(b: Briefing, cell: int = 10) -> bytes:
    """The map as a PNG: floor light, other terrain dark and hatched, panels pale yellow, the HUD grey-blue; things
    in their own colours, boxed in the colour of their guessed role and tagged "id: ROLE?"; dashed threads join
    look-alikes (magenta: turned or resized, blue: recoloured); a yellow ring marks a thing held at a midpoint; thin
    lines show the floor's tile grid. The HUD things are drawn again, enlarged, in a panel on the right."""
    with _DRAW_LOCK:
        return _picture(b, max(4, int(cell)))


def _picture(b: Briefing, cell: int) -> bytes:
    lay, g = b.layout, b.layout.g
    H, W = g.shape
    things = lay.things
    hud = np.zeros(g.shape, bool)
    for t in things:
        if t.hud:
            if t.panel is not None:
                p = lay.panels[t.panel]
                hud[p.r0:p.r1 + 1, p.c0:p.c1 + 1] = True
            for y, x in t.cells:
                hud[y, x] = True
    panel = np.zeros(g.shape, bool)
    for p in lay.panels:
        if not p.border:
            panel[p.r0:p.r1 + 1, p.c0:p.c1 + 1] = True
    wall = np.isin(g, lay.terrain) & (g != b.floor) & ~panel & ~hud
    base = np.empty((H, W, 3), np.uint8)
    base[:] = (236, 231, 216)
    base[panel & ~hud] = (255, 244, 200)
    base[wall] = (70, 72, 80)
    base[hud] = (205, 205, 215)
    img = Image.fromarray(np.repeat(np.repeat(base, cell, 0), cell, 1))
    d = ImageDraw.Draw(img)
    for y, x in zip(*np.nonzero(wall)):
        d.line([x * cell, (y + 1) * cell - 1, (x + 1) * cell - 1, y * cell], fill=(100, 102, 112))
    if b.lattice:
        u, ro, co, _ = b.lattice
        for r in range(ro, H, u):
            d.line([(0, r * cell), (W * cell, r * cell)], fill=(210, 205, 190))
        for c in range(co, W, u):
            d.line([(c * cell, 0), (c * cell, H * cell)], fill=(210, 205, 190))
    for t in things:
        for y, x in t.cells:
            d.rectangle([x * cell, y * cell, (x + 1) * cell - 1, (y + 1) * cell - 1], fill=PALETTE[int(g[y, x])],
                        outline=(60, 60, 60))
    half = cell / 2
    for i, j, kind in b.threads:
        a, o = things[i], things[j]
        col = (255, 0, 255) if kind == "TURNED" else (0, 170, 255)
        (x0, y0), (x1, y1) = (a.cx * cell + half, a.cy * cell + half), (o.cx * cell + half, o.cy * cell + half)
        for t in np.linspace(0, 1, 40)[::2]:
            d.line([(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t),
                    (x0 + (x1 - x0) * (t + 0.025), y0 + (y1 - y0) * (t + 0.025))], fill=col, width=3)
    tag = _font(14, True)
    for k, t in enumerate(things):
        if t.kind == "line":
            continue
        role = b.roles.get(k)
        col = ROLE_COLOURS.get(role[0], (220, 220, 220)) if role else (220, 220, 220)
        d.rectangle([t.c0 * cell - 2, t.r0 * cell - 2, (t.c1 + 1) * cell + 1, (t.r1 + 1) * cell + 1], outline=col,
                    width=3)
        num = str(t.id) if t.id is not None else "?"
        lab = f"{num}: {role[0]}?" if role else num
        tx = min(max(0, t.c0 * cell), W * cell - 8 * len(lab))
        ty = t.r0 * cell - 18 if t.r0 > 2 else (t.r1 + 1) * cell + 2
        d.text((tx, ty), lab, fill=col, font=tag, stroke_width=3, stroke_fill=(0, 0, 0))
    for k in b.held:
        t = things[k]
        d.ellipse([t.cx * cell - 0.4 * cell, t.cy * cell - 0.4 * cell, t.cx * cell + 1.4 * cell,
                   t.cy * cell + 1.4 * cell], outline=(255, 255, 0), width=3)
    side = [_caption("HUD", 190)]
    room = H * cell + 28 - side[0].height
    shown_hud = [k for k, t in enumerate(things) if t.hud]
    for n, k in enumerate(shown_hud):
        t = things[k]
        z = max(3, min(12, 150 // max(t.h, t.w)))
        im = Image.new("RGB", (190, t.h * z + 8), (60, 60, 70))
        dd = ImageDraw.Draw(im)
        cg = t.colgrid(g)
        for y in range(t.h):
            for x in range(t.w):
                if cg[y, x] >= 0:
                    dd.rectangle([4 + x * z, 4 + y * z, 4 + (x + 1) * z - 2, 4 + (y + 1) * z - 2],
                                 fill=PALETTE[int(cg[y, x])])
        role = b.roles.get(k)
        what = role[0] if role else t.short
        block = _titled(im, f"{t.id if t.id is not None else '?'}: {what}", size=12)
        if block.height + 12 > room:
            side.append(_caption(f"+{len(shown_hud) - n} more", 190, 12))
            break
        side.append(block)
        room -= block.height + 12
    if not shown_hud:
        side.append(_caption("(none)", 190, 12))
    return _png(_stack([_titled(img, PICTURE_TITLE), _stack(side, across=False)], across=True))


# =====================================================================================================================
# 5. What the agent reads
# =====================================================================================================================

@dataclass
class Scene:
    """One state as the agent is shown it. ``prev`` is the grid of the last observation sent (the baseline of the
    changed region); ``briefing`` is shown in the full view, at most ``briefing_lines`` lines of it."""
    step: int | None
    rows: list[list[int]]
    objects: list[Obj]
    change: str | None = None                 # the newest change line: "#12 A1(up): obj 4 R 3x3 moved up 5 -> ..."
    prev: list[list[int]] | None = None
    status: str = ""                          # "game ls20 | level 1 of 7 (0 done) | ..." from the host
    briefing: Briefing | None = None
    briefing_lines: int = 30

    def colours(self) -> set[int]:
        return {v for r in self.rows for v in r}

    def text(self, full: bool = False, *, max_objects: int = 40, budget_tokens: int = 800, show_ascii: bool = True,
             segmentation: bool = True, margin: int = 3) -> str:
        """The full view (briefing, every object, the whole board), or after an act the short view: the objects in
        the changed region and that region as letters, within ``budget_tokens`` (chars / 4)."""
        head = [f"[state after step #{self.step}]" if self.step is not None else "[state]"]
        if self.status:
            head[0] += " " + self.status
        if self.change:
            head.append(f"last step: {self.change}")
        bg = background(self.rows)
        if full or self.prev is None:
            parts = self.briefing.lines(self.briefing_lines) if self.briefing is not None else []
            if segmentation:
                objs = self.objects[:max_objects]
                parts.append(f"objects ({len(self.objects)}, background {letter(bg)} {NAMES[bg]}): id colour size "
                             "where #shape-hash corners [hud] | relations")
                parts += ["  " + o.row() for o in objs]
                if len(self.objects) > len(objs):
                    parts.append(f"  +{len(self.objects) - len(objs)} more: `observe().objects` in ipython")
            if show_ascii:
                parts.append("board:")
                parts.append(ascii(self.rows))
            parts.append("legend: " + legend(self.colours()))
            return "\n".join(head + parts)
        budget = budget_tokens * 4 - sum(len(x) + 1 for x in head)
        boxes = change_boxes(self.prev, self.rows, margin)
        crops, shown = [], set()
        if show_ascii:
            for bx in boxes:
                txt = f"changed region r{bx[0]}-{bx[2]} c{bx[1]}-{bx[3]}:\n" + ascii(self.rows, bx)
                if len(txt) + 1 > budget * 3 // 4 - sum(len(c) + 1 for c in crops):
                    crops.append(f"changed region r{bx[0]}-{bx[2]} c{bx[1]}-{bx[3]}: too large to show here; "
                                 f"`print(observe().ascii({bx[0]}, {bx[1]}, {bx[2]}, {bx[3]}))` in ipython")
                    continue
                crops.append(txt)
                shown |= {self.rows[r][c] for r in range(bx[0], bx[2] + 1) for c in range(bx[1], bx[3] + 1)}
        rows: list[str] = []
        if segmentation and boxes:
            near = [o for o in self.objects if any(o.bbox[0] <= bx[2] and bx[0] <= o.bbox[2] and o.bbox[1] <= bx[3]
                                                   and bx[1] <= o.bbox[3] for bx in boxes)]
            near = [o for o in near if o.size < len(self.rows) * len(self.rows[0]) // 2][:max_objects]
            if near:
                rows = [f"objects in the changed region ({len(near)} of {len(self.objects)}): id colour size where "
                        "#shape-hash corners [hud] | relations"] + ["  " + o.row() for o in near]
        tail = ["legend: " + legend(shown)] if shown else []
        room = budget - sum(len(x) + 1 for x in crops + tail)
        while rows and sum(len(x) + 1 for x in rows) > room:
            rows.pop()
            if len(rows) == 1:
                rows = []
        return "\n".join(head + rows + crops + tail)

    def to_json(self) -> dict[str, Any]:
        """What ``observe()`` returns in the REPL (the host adds the full text). No number grid: colours are letters."""
        bg = background(self.rows)
        return {"step": self.step, "status": self.status, "change": self.change, "background": letter(bg),
                "legend": legend(self.colours()), "letters": ["".join(letter(v) for v in r) for r in self.rows],
                "objects": [o.to_json() for o in self.objects],
                "briefing": self.briefing.lines() if self.briefing is not None else []}


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def image_message(text: str, png: bytes, step: int | None) -> dict[str, Any]:
    """A user message: the observation text, then the picture. ``_image_step`` marks it for the context builder,
    which keeps only the newest picture."""
    return {"role": "user", "_kind": "observation", "_image_step": step,
            "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": data_url(png)}}]}
