"""E008 perception, text channel: the game frame as letters, objects and changes. Pure Python, no host access.

Written by us, following the output format of Tufa Labs' ``grid_utils.py`` / ``segmentation.py`` (one letter per
colour, objects with a shape hash, corners, containment and adjacency). No Tufa code is copied: their licence is
UNCONFIRMED. Coordinates are always (row, column), top-left = (0, 0), the order of the ``act`` click "6 r c".

    ascii(grid, box=None)       -> letter grid with row and column labels (box = (r0, c0, r1, c1), inclusive)
    segment(grid)               -> (objects, background): 4-connected single-colour regions, background skipped
    Tracker().observe(grid)     -> (objects with ids stable across steps, change line or None)
    change_boxes(prev, grid, m) -> boxes around the changed cells, ``m`` cells of margin, merged when they touch
    Scene(...).text()           -> the observation text the agent reads after every act
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

LETTERS = "WwgGcBMPRbSYOrNp"
NAMES = ("white", "light grey", "grey", "dark grey", "charcoal", "black", "magenta", "pink", "red", "blue", "sky blue",
         "yellow", "orange", "dark red", "green", "purple")
Box = tuple[int, int, int, int]   # r0, c0, r1, c1 inclusive


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


@dataclass
class Scene:
    """What the agent is shown about one state. ``prev`` is the grid before the last act (None at level start)."""
    step: int | None
    rows: list[list[int]]
    objects: list[Obj]
    change: str | None = None                 # the last step's change line
    prev: list[list[int]] | None = None
    level_start: bool = False
    status: str = ""                          # "level 1/7 · NOT_FINISHED · ..." from the host

    def colours(self) -> set[int]:
        return {v for r in self.rows for v in r}

    def text(self, *, max_objects: int = 40, budget_tokens: int = 800, show_ascii: bool = True,
             segmentation: bool = True, margin: int = 3) -> str:
        head = [f"[state after step #{self.step}]" if self.step is not None else "[state]"]
        if self.status:
            head[0] += " " + self.status
        if self.change and not self.level_start:
            head.append(f"last step: {self.change}")
        bg = background(self.rows)
        parts: list[str] = []
        if self.level_start:
            if segmentation:
                objs = [o for o in self.objects][:max_objects]
                parts.append(f"objects ({len(self.objects)}, background {letter(bg)} {NAMES[bg]}): id colour size "
                             "where #shape-hash corners [hud] | relations")
                parts += ["  " + o.row() for o in objs]
                if len(self.objects) > len(objs):
                    parts.append(f"  +{len(self.objects) - len(objs)} more: `scene.objects` in ipython")
            if show_ascii:
                parts.append("board:")
                parts.append(ascii(self.rows))
            parts.append("legend: " + legend(self.colours()))
            return "\n".join(head + parts)
        budget = budget_tokens * 4 - sum(len(x) + 1 for x in head)
        boxes = change_boxes(self.prev, self.rows, margin) if self.prev is not None else []
        crops, shown = [], set()
        if show_ascii:
            for bx in boxes:
                txt = f"changed region r{bx[0]}-{bx[2]} c{bx[1]}-{bx[3]}:\n" + ascii(self.rows, bx)
                if len(txt) + 1 > budget * 3 // 4 - sum(len(c) + 1 for c in crops):
                    crops.append(f"changed region r{bx[0]}-{bx[2]} c{bx[1]}-{bx[3]}: too large to show here; "
                                 f"`print(scene.ascii({bx[0]}, {bx[1]}, {bx[2]}, {bx[3]}))` in ipython")
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
        return {"step": self.step, "status": self.status, "change": self.change, "level_start": self.level_start,
                "background": background(self.rows), "letters": ["".join(letter(v) for v in r) for r in self.rows],
                "objects": [o.to_json() for o in self.objects]}
