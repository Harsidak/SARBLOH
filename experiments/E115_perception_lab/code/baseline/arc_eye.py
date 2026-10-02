#!/usr/bin/env python3
"""
arc_eye.py -- screenshot -> symbolic state for grid-based puzzle games.

Single-file perception pipeline. The analyst (human or model) never looks at the
pixels; it reads only the text report this program prints.

Pipeline
  1. store      copy screenshot into a session folder with a frame index + action tag
  2. lattice    infer cell pitch and origin from colour-edge positions (no hard-coded size)
  3. sample     median colour of each cell's inner region -> one symbol per cell
  4. palette    greedy colour clustering, persisted per session so symbols stay stable
  5. parse      terrain vs objects, connected components, holes, symmetry, D4 shape hash,
                composite objects (touching, different colours), dotted chains, bars
  6. diff       cell-level change map + object tracking (translation / rotation / reshape)
                against the previous frame of the session

Usage
  python arc_eye.py ingest shot.png --action LEFT      # store + analyse + diff vs previous
  python arc_eye.py ingest shot.png --new              # start a fresh session (new game/level)
  python arc_eye.py report 3                           # reprint report of frame 3
  python arc_eye.py diff 2 5                           # diff any two stored frames
  python arc_eye.py log                                # action history

Dependencies: numpy, Pillow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

log = logging.getLogger("arc_eye")

# --------------------------------------------------------------------------- config
EDGE_T = 60.0           # sum |dRGB| across a pixel boundary that counts as a real colour edge
GRID_N = 64            # HARD-CODED board size: every frame is forced to exactly 64 x 64 cells
PITCH_SLACK = 0.15     # pitch searched only within +-15% of image_size / GRID_N
PITCH_STEP = 0.005
PITCH_TOL = 0.75        # px tolerance for an edge to sit on a lattice line
CLUSTER_DIST = 38.0     # RGB distance under which two cell colours are the same symbol
TERRAIN_FRAC = 0.08     # colours covering more than this fraction of cells are terrain
SMALL_FRAGMENT = 12     # terrain-coloured components at most this big are still objects
MIXED_STD = 22.0        # inner-pixel std above this marks a cell as visually mixed
MAX_MASK_PRINT = (12, 16)

NAME_SYMBOL = {
    "white": "W", "lightgray": "#", "gray": "=", "darkgray": ":", "black": "K",
    "red": "R", "orange": "O", "yellow": "Y", "green": "G", "blue": "B", "cyan": "C",
    "purple": "P", "magenta": "M", "pink": "p", "brown": "N", "maroon": "m", "navy": "n",
}
SPARE_SYMBOLS = "abcdefghijstuvwxyz0123456789&$@%"
HUES = {
    "red": (230, 40, 40), "orange": (255, 140, 20), "yellow": (255, 220, 0),
    "green": (40, 200, 60), "blue": (30, 140, 255), "cyan": (0, 220, 220),
    "purple": (160, 80, 220), "magenta": (230, 50, 200), "pink": (255, 150, 200),
    "brown": (140, 80, 30), "maroon": (110, 25, 60), "navy": (20, 30, 120),
}


def colour_name(rgb: Iterable[float]) -> str:
    r, g, b = (float(v) for v in rgb)
    if max(r, g, b) - min(r, g, b) < 25:
        lum = (r + g + b) / 3
        for lim, nm in ((30, "black"), (85, "darkgray"), (150, "gray"), (215, "lightgray")):
            if lum < lim:
                return nm
        return "white"
    return min(HUES, key=lambda k: sum((a - c) ** 2 for a, c in zip((r, g, b), HUES[k])))


# --------------------------------------------------------------------------- palette
@dataclass
class Palette:
    entries: list[dict] = field(default_factory=list)   # {sym, rgb, name}

    @classmethod
    def load(cls, path: Path) -> "Palette":
        return cls(json.loads(path.read_text())) if path.exists() else cls()

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.entries, indent=1))

    def match(self, rgb: np.ndarray) -> str | None:
        best, bd = None, CLUSTER_DIST
        for e in self.entries:
            d = float(np.linalg.norm(np.asarray(e["rgb"]) - rgb))
            if d < bd:
                best, bd = e["sym"], d
        return best

    def add(self, rgb: np.ndarray, background: bool = False) -> str:
        used = {e["sym"] for e in self.entries}
        name = colour_name(rgb)
        if background and "." not in used:
            sym = "."
        else:
            sym = NAME_SYMBOL.get(name, "?")
            if sym in used or sym == "?":
                sym = next(s for s in SPARE_SYMBOLS if s not in used)
        self.entries.append({"sym": sym, "rgb": [round(float(v)) for v in rgb], "name": name})
        return sym

    def info(self, sym: str) -> dict:
        return next((e for e in self.entries if e["sym"] == sym), {"sym": sym, "rgb": None, "name": "unknown"})


# --------------------------------------------------------------------------- lattice
@dataclass
class Lattice:
    pitch: float
    x0: float
    y0: float
    rows: int
    cols: int
    score: float


def _edge_positions(im: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.abs(np.diff(im, axis=axis)).sum(axis=2)
    counts = (d > EDGE_T).sum(axis=0 if axis == 1 else 1).astype(float)
    pos = np.nonzero(counts >= 2)[0]
    return pos.astype(float) + 1.0, counts[pos]


def _axis_fit(pos: np.ndarray, w: np.ndarray, P: float) -> tuple[float, float]:
    if len(pos) == 0:
        return 0.0, 0.0
    ph = np.mod(pos, P)
    cand = ph[np.argsort(-w)[:40]]
    dist = np.abs(ph[:, None] - cand[None, :])
    dist = np.minimum(dist, P - dist)
    s = (w[:, None] * (dist <= PITCH_TOL)).sum(axis=0)
    k = int(np.argmax(s))
    # refine offset as weighted circular mean of inliers
    inl = dist[:, k] <= PITCH_TOL
    ang = ph[inl] / P * 2 * np.pi
    off = (np.angle(np.sum(w[inl] * np.exp(1j * ang))) / (2 * np.pi) * P) % P
    return float(s[k] / w.sum()), float(off)


def _refine(px, wx, py, wy, P, ox, oy, iters: int = 4):
    """Weighted least squares: pos = off_axis + k * P over inlier edges of both axes.
    The coarse search only knows P to within the tolerance band; drift of 0.03px/cell
    accumulates to >2px across 64 cells, so this step is mandatory."""
    for _ in range(iters):
        rows, rhs, wts = [], [], []
        for pos, w, off, ax in ((px, wx, ox, 0), (py, wy, oy, 1)):
            k = np.round((pos - off) / P)
            res = pos - (off + k * P)
            inl = np.abs(res) <= PITCH_TOL
            for kk, pp, ww in zip(k[inl], pos[inl], w[inl]):
                rows.append([kk, 1.0 if ax == 0 else 0.0, 1.0 if ax == 1 else 0.0])
                rhs.append(pp)
                wts.append(ww)
        if len(rows) < 6:
            break
        A, b, sw = np.array(rows), np.array(rhs), np.sqrt(np.array(wts))
        sol, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
        P, ox, oy = float(sol[0]), float(sol[1]) % float(sol[0]), float(sol[2]) % float(sol[0])
    return P, ox, oy


def fit_lattice(im: np.ndarray) -> Lattice:
    H, W, _ = im.shape
    px, wx = _edge_positions(im, 1)
    py, wy = _edge_positions(im, 0)
    tot = wx.sum() + wy.sum()
    if tot == 0:
        raise ValueError("no colour edges found: blank image?")
    best = None
    lo = min(W, H) / GRID_N * (1 - PITCH_SLACK)
    hi = max(W, H) / GRID_N * (1 + PITCH_SLACK)
    for P in np.arange(lo, hi, PITCH_STEP):
        sx, _ = _axis_fit(px, wx, P)
        sy, _ = _axis_fit(py, wy, P)
        raw = (sx * wx.sum() + sy * wy.sum()) / tot
        chance = min(1.0, 2 * PITCH_TOL / P)
        norm = (raw - chance) / (1 - chance + 1e-9)
        if best is None or norm > best[0] + 1e-9:
            best = (norm, P)
    norm, P = best
    _, ox = _axis_fit(px, wx, P)
    _, oy = _axis_fit(py, wy, P)
    P, ox, oy = _refine(px, wx, py, wy, P, ox, oy)
    # include a leading partial cell only if at least half of it is visible
    x0 = ox - P if ox >= 0.5 * P else ox
    y0 = oy - P if oy >= 0.5 * P else oy
    cols = int(np.floor((W - x0) / P + 0.5))
    rows = int(np.floor((H - y0) / P + 0.5))
    return Lattice(float(P), float(x0), float(y0), rows, cols, float(norm))


# --------------------------------------------------------------------------- frame
@dataclass
class Frame:
    grid: np.ndarray            # rows x cols of symbols
    lattice: Lattice
    mixed: list[tuple[int, int]]
    image_size: tuple[int, int]
    fit_notes: list[str] = field(default_factory=list)


def _is_chrome_line(rgb: np.ndarray, ok: np.ndarray) -> bool:
    """A screen-edge strip: (almost) fully visible and one flat colour end to end."""
    if ok.mean() < 0.9:
        return False
    v = rgb[ok]
    return bool(np.linalg.norm(v - np.median(v, axis=0), axis=1).max() < 20)


def _force_axis(rgb, valid, vis, axis, notes, name):
    """Trim or pad one axis to exactly GRID_N lines.
    Trim order: screen-edge chrome strips first, then the less-visible (cut-off) edge line.
    Returns arrays plus the index shift applied at the start."""
    shift = 0
    while rgb.shape[axis] > GRID_N:
        n = rgb.shape[axis]
        first = (np.take(rgb, 0, axis), np.take(valid, 0, axis))
        last = (np.take(rgb, n - 1, axis), np.take(valid, n - 1, axis))
        key_first = (0 if _is_chrome_line(*first) else 1, vis[0])
        key_last = (0 if _is_chrome_line(*last) else 1, vis[-1])
        if key_first < key_last:
            why = "chrome strip" if key_first[0] == 0 else f"cut-off ({vis[0]:.0%} visible)"
            notes.append(f"dropped first {name} ({why})")
            rgb, valid, vis = np.delete(rgb, 0, axis), np.delete(valid, 0, axis), vis[1:]
            shift += 1
        else:
            why = "chrome strip" if key_last[0] == 0 else f"cut-off ({vis[-1]:.0%} visible)"
            notes.append(f"dropped last {name} ({why})")
            rgb, valid, vis = np.delete(rgb, n - 1, axis), np.delete(valid, n - 1, axis), vis[:-1]
    while rgb.shape[axis] < GRID_N:
        notes.append(f"padded last {name} with '?' (image too small)")
        pad_shape = list(rgb.shape); pad_shape[axis] = 1
        rgb = np.concatenate([rgb, np.zeros(pad_shape)], axis)
        vshape = list(valid.shape); vshape[axis] = 1
        valid = np.concatenate([valid, np.zeros(vshape, bool)], axis)
        vis = np.append(vis, 0.0)
    return rgb, valid, vis, shift


def sample_frame(path: Path, palette: Palette) -> Frame:
    im = np.asarray(Image.open(path).convert("RGB")).astype(float)
    H, W, _ = im.shape
    lat = fit_lattice(im)
    P = lat.pitch
    cols_rgb = np.zeros((lat.rows, lat.cols, 3))
    valid = np.zeros((lat.rows, lat.cols), bool)
    mixed = []
    m = max(1, int(round(P * 0.22)))
    for r in range(lat.rows):
        for c in range(lat.cols):
            ya, yb = int(round(lat.y0 + r * P)) + m, int(round(lat.y0 + (r + 1) * P)) - m
            xa, xb = int(round(lat.x0 + c * P)) + m, int(round(lat.x0 + (c + 1) * P)) - m
            ya2, yb2, xa2, xb2 = max(ya, 0), min(yb, H), max(xa, 0), min(xb, W)
            if yb2 - ya2 < max(1, (yb - ya) * 0.3) or xb2 - xa2 < max(1, (xb - xa) * 0.3):
                continue
            px = im[ya2:yb2, xa2:xb2].reshape(-1, 3)
            cols_rgb[r, c] = np.median(px, axis=0)
            valid[r, c] = True
            if px.std(axis=0).max() > MIXED_STD:
                mixed.append((r, c))
    # ---- hard 64x64: trim / pad the sampled lattice
    notes: list[str] = []
    visr = np.array([(min(H, lat.y0 + (r + 1) * P) - max(0, lat.y0 + r * P)) / P for r in range(lat.rows)])
    visc = np.array([(min(W, lat.x0 + (c + 1) * P) - max(0, lat.x0 + c * P)) / P for c in range(lat.cols)])
    cols_rgb, valid, visr, sr = _force_axis(cols_rgb, valid, visr, 0, notes, "row")
    cols_rgb, valid, visc, sc = _force_axis(cols_rgb, valid, visc, 1, notes, "column")
    mixed = [(r - sr, c - sc) for r, c in mixed if 0 <= r - sr < GRID_N and 0 <= c - sc < GRID_N]
    lat = Lattice(lat.pitch, lat.x0 + sc * P, lat.y0 + sr * P, GRID_N, GRID_N, lat.score)
    # palette: assign by frequency so the dominant colour of the first frame becomes '.'
    flat = cols_rgb[valid]
    keys, inv, cnt = np.unique(np.round(flat / 4) * 4, axis=0, return_inverse=True, return_counts=True)
    first = not palette.entries
    for i in np.argsort(-cnt):
        if palette.match(keys[i]) is None:
            palette.add(keys[i], background=first and i == np.argsort(-cnt)[0])
    grid = np.full((lat.rows, lat.cols), "?", dtype="<U1")
    for r, c in zip(*np.nonzero(valid)):
        grid[r, c] = palette.match(cols_rgb[r, c]) or palette.add(cols_rgb[r, c])
    assert grid.shape == (GRID_N, GRID_N), grid.shape
    return Frame(grid, lat, mixed, (W, H), notes)


# --------------------------------------------------------------------------- objects
@dataclass
class Obj:
    id: int
    sym: str
    size: int
    r0: int
    c0: int
    r1: int
    c1: int
    cy: float
    cx: float
    mask: list[str]
    shape: str          # exact shape hash
    d4: str             # rotation/reflection-invariant hash
    sym_flags: str
    holes: int
    on: str             # symbol surrounding the object
    terrain_fragment: bool = False
    chrome: bool = False        # spans a full screen edge: UI/border, not game content

    @property
    def h(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def w(self) -> int:
        return self.c1 - self.c0 + 1


def _components(mask: np.ndarray, conn8: bool = False) -> list[list[tuple[int, int]]]:
    R, C = mask.shape
    seen = np.zeros_like(mask, bool)
    nb = [(1, 0), (-1, 0), (0, 1), (0, -1)] + ([(1, 1), (1, -1), (-1, 1), (-1, -1)] if conn8 else [])
    out = []
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        q, cells = deque([(r, c)]), []
        seen[r, c] = True
        while q:
            y, x = q.popleft()
            cells.append((int(y), int(x)))
            for dy, dx in nb:
                yy, xx = y + dy, x + dx
                if 0 <= yy < R and 0 <= xx < C and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    q.append((yy, xx))
        out.append(cells)
    return out


def _hash(a: np.ndarray) -> str:
    return hashlib.md5(f"{a.shape}".encode() + np.packbits(a).tobytes()).hexdigest()[:8]


def _d4(a: np.ndarray) -> list[np.ndarray]:
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out += [r, np.fliplr(r)]
    return out


def _sym_flags(a: np.ndarray) -> str:
    f = []
    if np.array_equal(a, np.fliplr(a)): f.append("mirrorLR")
    if np.array_equal(a, np.flipud(a)): f.append("mirrorUD")
    if a.shape[0] == a.shape[1] and np.array_equal(a, np.rot90(a)): f.append("rot90")
    elif np.array_equal(a, np.rot90(a, 2)): f.append("rot180")
    return ",".join(f) or "none"


def _holes(a: np.ndarray) -> int:
    pad = np.pad(~a, 1, constant_values=True)
    outside = _components(pad)
    border = [comp for comp in outside if any(y in (0, pad.shape[0] - 1) or x in (0, pad.shape[1] - 1) for y, x in comp)]
    return int(pad.sum() - sum(len(b) for b in border))


def extract_objects(grid: np.ndarray, palette: Palette) -> tuple[list[Obj], dict]:
    R, C = grid.shape
    counts = Counter(grid.ravel())
    total = grid.size
    terrain = {s for s, n in counts.items() if n / total > TERRAIN_FRAC and s != "?"}
    objs: list[Obj] = []
    regions: dict[str, list[tuple]] = {}
    for sym in sorted(counts):
        if sym == "?":
            continue
        for cells in _components(grid == sym):
            ys, xs = zip(*cells)
            if sym in terrain and len(cells) > SMALL_FRAGMENT:
                regions.setdefault(sym, []).append((len(cells), min(ys), min(xs), max(ys), max(xs)))
                continue
            r0, c0, r1, c1 = min(ys), min(xs), max(ys), max(xs)
            a = np.zeros((r1 - r0 + 1, c1 - c0 + 1), bool)
            for y, x in cells:
                a[y - r0, x - c0] = True
            ring = [grid[y, x] for y in range(max(r0 - 1, 0), min(r1 + 2, R)) for x in range(max(c0 - 1, 0), min(c1 + 2, C))
                    if not (r0 <= y <= r1 and c0 <= x <= c1)]
            on = Counter(ring).most_common(1)[0][0] if ring else "?"
            objs.append(Obj(
                id=0, sym=sym, size=len(cells), r0=r0, c0=c0, r1=r1, c1=c1,
                cy=round(float(np.mean(ys)), 1), cx=round(float(np.mean(xs)), 1),
                mask=["".join(sym if v else "." for v in row) for row in a],
                shape=_hash(a), d4=min(_hash(t) for t in _d4(a)), sym_flags=_sym_flags(a),
                holes=_holes(a), on=on, terrain_fragment=sym in terrain))
    objs.sort(key=lambda o: (o.r0, o.c0))
    for i, o in enumerate(objs):
        o.id = i
        edge_col = o.w == 1 and o.c0 in (0, C - 1) and o.h >= 0.9 * R
        edge_row = o.h == 1 and o.r0 in (0, R - 1) and o.w >= 0.9 * C
        o.chrome = bool(edge_col or edge_row)
    return objs, {"terrain": sorted(terrain), "regions": regions, "counts": dict(counts)}


def composites(objs: list[Obj], grid: np.ndarray) -> list[list[int]]:
    """Groups of objects whose cells touch (8-neighbourhood) and have different colours."""
    owner = {}
    for o in objs:
        for i, row in enumerate(o.mask):
            for j, ch in enumerate(row):
                if ch != ".":
                    owner[(o.r0 + i, o.c0 + j)] = o.id
    parent = list(range(len(objs)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for (y, x), a in owner.items():
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                b = owner.get((y + dy, x + dx))
                if b is not None and b != a and objs[a].sym != objs[b].sym:
                    parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for o in objs:
        groups.setdefault(find(o.id), []).append(o.id)
    return [g for g in groups.values() if len(g) > 1]


def chains(objs: list[Obj], max_gap: int = 2, max_size: int = 6) -> list[dict]:
    """Same-colour small pieces spaced along a line (dotted lines, trails, tethers)."""
    small = [o for o in objs if o.size <= max_size]
    by_sym: dict[str, list[Obj]] = {}
    for o in small:
        by_sym.setdefault(o.sym, []).append(o)
    out = []
    for sym, lst in by_sym.items():
        n = len(lst)
        adj = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                a, b = lst[i], lst[j]
                gy = max(0, max(a.r0, b.r0) - min(a.r1, b.r1) - 1)
                gx = max(0, max(a.c0, b.c0) - min(a.c1, b.c1) - 1)
                if max(gy, gx) <= max_gap:
                    adj[i].append(j)
                    adj[j].append(i)
        seen = set()
        for i in range(n):
            if i in seen:
                continue
            comp, q = [], [i]
            seen.add(i)
            while q:
                k = q.pop()
                comp.append(k)
                for m in adj[k]:
                    if m not in seen:
                        seen.add(m)
                        q.append(m)
            if len(comp) < 3:
                continue
            pts = np.array([(lst[k].cy, lst[k].cx) for k in comp])
            mu = pts.mean(0)
            u, s, vt = np.linalg.svd(pts - mu)
            d = vt[0]
            t = (pts - mu) @ d
            a_, b_ = pts[np.argmin(t)], pts[np.argmax(t)]
            resid = float(np.abs((pts - mu) @ vt[1]).max()) if len(vt) > 1 else 0.0
            out.append({"sym": sym, "pieces": len(comp), "ids": sorted(lst[k].id for k in comp),
                        "end_a": (round(float(a_[0]), 1), round(float(a_[1]), 1)),
                        "end_b": (round(float(b_[0]), 1), round(float(b_[1]), 1)),
                        "straight": resid < 1.0, "max_off_line": round(resid, 2)})
    return out


def glyphs(objs: list[Obj], grid: np.ndarray, gap: int = 1) -> list[dict]:
    """Same-colour pieces separated by at most `gap` cells, read as one figure
    (icons, rings, letters). Returns only groups with >1 piece."""
    n = len(objs)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            a, b = objs[i], objs[j]
            if a.sym != b.sym or a.terrain_fragment != b.terrain_fragment:
                continue
            gy = max(0, max(a.r0, b.r0) - min(a.r1, b.r1) - 1)
            gx = max(0, max(a.c0, b.c0) - min(a.c1, b.c1) - 1)
            if max(gy, gx) <= gap:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for o in objs:
        groups.setdefault(find(o.id), []).append(o.id)
    out = []
    for ids in groups.values():
        if len(ids) < 2:
            continue
        os_ = [objs[i] for i in ids]
        r0, c0 = min(o.r0 for o in os_), min(o.c0 for o in os_)
        r1, c1 = max(o.r1 for o in os_), max(o.c1 for o in os_)
        a = np.zeros((r1 - r0 + 1, c1 - c0 + 1), bool)
        for o in os_:
            for i, row in enumerate(o.mask):
                for j, ch in enumerate(row):
                    if ch != ".":
                        a[o.r0 - r0 + i, o.c0 - c0 + j] = True
        out.append({"ids": sorted(ids), "sym": os_[0].sym, "bbox": (r0, c0, r1, c1), "mask": a,
                    "holes": _holes(a), "symflags": _sym_flags(a), "d4": min(_hash(t) for t in _d4(a))})
    return out


def nearest_to(point: tuple[float, float], objs: list[Obj], exclude: set[int], k: int = 2) -> list[tuple[int, float]]:
    def dist(o: Obj) -> float:
        dy = max(o.r0 - point[0], 0, point[0] - o.r1)
        dx = max(o.c0 - point[1], 0, point[1] - o.c1)
        return float(np.hypot(dy, dx))
    cand = sorted(((o.id, dist(o)) for o in objs if o.id not in exclude and not o.chrome), key=lambda t: t[1])
    return [(i, round(d, 1)) for i, d in cand[:k]]


# --------------------------------------------------------------------------- diff / tracking
def track(prev: list[dict], cur: list[Obj]) -> list[str]:
    lines, used = [], set()
    curd = [asdict(o) for o in cur]

    def cen(o):
        return np.array([o["cy"], o["cx"]])
    rules = [
        ("same", lambda a, b: a["sym"] == b["sym"] and a["shape"] == b["shape"]),
        ("turned", lambda a, b: a["sym"] == b["sym"] and a["d4"] == b["d4"]),
        ("reshaped", lambda a, b: a["sym"] == b["sym"]),
    ]
    unmatched_prev = list(range(len(prev)))
    for tag, ok in rules:
        nxt = []
        for i in unmatched_prev:
            a = prev[i]
            cands = [(float(np.linalg.norm(cen(a) - cen(b))), j) for j, b in enumerate(curd) if j not in used and ok(a, b)]
            if tag == "reshaped":
                cands = [c for c in cands if c[0] <= 8]
            if not cands:
                nxt.append(i)
                continue
            d, j = min(cands)
            used.add(j)
            b = curd[j]
            dy, dx = round(b["cy"] - a["cy"], 1), round(b["cx"] - a["cx"], 1)
            if tag == "same" and dy == 0 and dx == 0:
                continue
            extra = ""
            if tag == "reshaped":
                extra = f" size {a['size']}->{b['size']} bbox {a['r1']-a['r0']+1}x{a['c1']-a['c0']+1}->{b['r1']-b['r0']+1}x{b['c1']-b['c0']+1}"
            if tag == "turned":
                extra = " (same shape up to rotation/reflection)"
            lines.append(f"  {tag.upper():8s} {a['sym']} prev#{a['id']} -> cur#{b['id']}  move dy={dy:+} dx={dx:+}{extra}")
        unmatched_prev = nxt
    for i in unmatched_prev:
        a = prev[i]
        lines.append(f"  VANISHED {a['sym']} prev#{a['id']} size={a['size']} at ({a['r0']},{a['c0']})-({a['r1']},{a['c1']})")
    for j, b in enumerate(curd):
        if j not in used:
            lines.append(f"  APPEARED {b['sym']} cur#{b['id']} size={b['size']} at ({b['r0']},{b['c0']})-({b['r1']},{b['c1']})")
    return lines


def cell_diff(a: np.ndarray, b: np.ndarray) -> list[str]:
    if a.shape != b.shape:
        return [f"  grid shape changed {a.shape} -> {b.shape}; cell diff skipped"]
    ch = a != b
    n = int(ch.sum())
    lines = [f"  {n} cells changed"]
    for comp in sorted(_components(ch, conn8=True), key=len, reverse=True):
        ys, xs = zip(*comp)
        tr = Counter(f"{a[y, x]}->{b[y, x]}" for y, x in comp)
        lines.append(f"  region ({min(ys)},{min(xs)})-({max(ys)},{max(xs)}) n={len(comp)}  " +
                     " ".join(f"{k}x{v}" for k, v in tr.most_common(6)))
    return lines


# --------------------------------------------------------------------------- report
def render(frame: Frame, palette: Palette, objs: list[Obj], meta: dict, idx: int, action: str | None,
           prev: dict | None) -> str:
    g, lat = frame.grid, frame.lattice
    out = [f"=== FRAME {idx}  action={action or '-'}  image={frame.image_size[0]}x{frame.image_size[1]}px ==="]
    out.append(f"LATTICE pitch={lat.pitch:.3f}px origin=({lat.x0:.2f},{lat.y0:.2f}) grid={lat.rows}x{lat.cols} "
               f"fit={lat.score:.3f}  mixed_cells={len(frame.mixed)}"
               + (f" e.g. {frame.mixed[:8]}" if frame.mixed else ""))
    out.append("FIT NOTES: " + ("; ".join(frame.fit_notes) if frame.fit_notes else "lattice was exactly 64x64, nothing trimmed"))
    out.append("PALETTE")
    for s, n in sorted(meta["counts"].items(), key=lambda t: -t[1]):
        e = palette.info(s)
        role = "terrain" if s in meta["terrain"] else "object"
        out.append(f"  '{s}' {e['name']:9s} rgb={e['rgb']} cells={n} ({role})")
    out.append("GRID")
    C = g.shape[1]
    out.append("     " + "".join(str(c // 10 % 10) for c in range(C)))
    out.append("     " + "".join(str(c % 10) for c in range(C)))
    for r in range(g.shape[0]):
        out.append(f"{r:3d}  " + "".join(g[r]))
    out.append("TERRAIN REGIONS (colour: n_regions; largest bboxes)")
    for s, regs in meta["regions"].items():
        regs = sorted(regs, reverse=True)
        out.append(f"  '{s}': {len(regs)} regions; " +
                   "; ".join(f"n={n} ({a},{b})-({c},{d})" for n, a, b, c, d in regs[:5]))
    out.append("OBJECTS")
    for o in objs:
        tag = (" [terrain-fragment]" if o.terrain_fragment else "") + (" [SCREEN-EDGE/CHROME]" if o.chrome else "")
        bar = " [BAR]" if max(o.h, o.w) >= 6 * min(o.h, o.w) and max(o.h, o.w) >= 6 else ""
        out.append(f"  #{o.id:<3d}'{o.sym}' {palette.info(o.sym)['name']:9s} size={o.size:<4d} bbox=({o.r0},{o.c0})-({o.r1},{o.c1}) "
                   f"{o.h}x{o.w} centre=({o.cy},{o.cx}) on='{o.on}' holes={o.holes} sym={o.sym_flags} "
                   f"shape={o.shape} d4={o.d4}{tag}{bar}")
        if o.h <= MAX_MASK_PRINT[0] and o.w <= MAX_MASK_PRINT[1] and o.size > 1:
            for row in o.mask:
                out.append("        " + row)
    comps = composites(objs, g)
    if comps:
        out.append("COMPOSITES (touching objects of different colours)")
        for grp in comps:
            os_ = [objs[i] for i in grp]
            r0, c0 = min(o.r0 for o in os_), min(o.c0 for o in os_)
            r1, c1 = max(o.r1 for o in os_), max(o.c1 for o in os_)
            out.append(f"  ids={grp} colours={''.join(o.sym for o in os_)} bbox=({r0},{c0})-({r1},{c1}) {r1-r0+1}x{c1-c0+1}")
            for r in range(r0, r1 + 1):
                out.append("        " + "".join(g[r, c] if any(o.r0 <= r <= o.r1 and o.c0 <= c <= o.c1 and o.mask[r - o.r0][c - o.c0] != "." for o in os_) else "." for c in range(c0, c1 + 1)))
    gl = glyphs(objs, g)
    if gl:
        out.append("GLYPHS (same-colour pieces <=1 cell apart, read as one figure)")
        for G in gl:
            r0, c0, r1, c1 = G["bbox"]
            out.append(f"  '{G['sym']}' ids={G['ids']} bbox=({r0},{c0})-({r1},{c1}) {r1-r0+1}x{c1-c0+1} "
                       f"cells={int(G['mask'].sum())} holes={G['holes']} sym={G['symflags']} d4={G['d4']}")
            if G["mask"].shape[0] <= MAX_MASK_PRINT[0] and G["mask"].shape[1] <= MAX_MASK_PRINT[1]:
                for row in G["mask"]:
                    out.append("        " + "".join(G["sym"] if v else "." for v in row))
    ch = chains(objs)
    if ch:
        out.append("CHAINS (same-colour pieces strung along a line)")
        for c in ch:
            ex = set(c["ids"])
            out.append(f"  '{c['sym']}' pieces={c['pieces']} straight={c['straight']} (max off-line {c['max_off_line']}) "
                       f"end_a={c['end_a']} near {nearest_to(c['end_a'], objs, ex)}  "
                       f"end_b={c['end_b']} near {nearest_to(c['end_b'], objs, ex)}")
            out.append(f"      pieces ids={c['ids']}")
            pcs = [(objs[i].cy, objs[i].cx) for i in c["ids"]]
            onl = [o.id for o in objs if o.id not in ex and o.size > 6 and not o.chrome and
                   min(max(o.r0 - y, 0, y - o.r1) + max(o.c0 - x, 0, x - o.c1) for y, x in pcs) <= 1.5]
            if onl:
                out.append(f"      objects sitting on the chain: {onl}")
    sal = [(f"#{o.id}'{o.sym}'", o.cy, o.cx) for o in objs if o.size >= 5 and not o.chrome and not o.terrain_fragment]
    for G in gl:
        if G["mask"].sum() >= 5 and G["sym"] not in {"#"}:
            r0, c0, r1, c1 = G["bbox"]
            sal.append((f"glyph'{G['sym']}'{G['ids'][0]}", (r0 + r1) / 2, (c0 + c1) / 2))
    if len(sal) > 1:
        out.append("RELATIONS between salient things (centre-to-centre; angle: 0=right, 90=up)")
        for i in range(len(sal)):
            for j in range(i + 1, len(sal)):
                a, b = sal[i], sal[j]
                dy, dx = b[1] - a[1], b[2] - a[2]
                ang = float(np.degrees(np.arctan2(-dy, dx)))
                out.append(f"  {a[0]:>14s} ({a[1]:.1f},{a[2]:.1f}) -> {b[0]:<14s} ({b[1]:.1f},{b[2]:.1f})  "
                           f"dy={dy:+.1f} dx={dx:+.1f} dist={np.hypot(dy, dx):.1f} angle={ang:+.0f}")
    if prev is not None:
        out.append(f"DIFF vs frame {prev['idx']} (action that led here: {action or '-'})")
        out += cell_diff(np.array(prev["grid"]), g)
        out.append("TRACKING")
        tr = track(prev["objects"], objs)
        out += tr if tr else ["  no object moved or changed"]
    return "\n".join(out)


# --------------------------------------------------------------------------- session
class Session:
    def __init__(self, root: Path):
        self.root = root
        self.frames = root / "frames"
        self.frames.mkdir(parents=True, exist_ok=True)
        self.pal_path = root / "palette.json"
        self.log_path = root / "log.jsonl"

    def reset(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.__init__(self.root)

    def count(self) -> int:
        return len(list(self.frames.glob("*.json")))

    def load(self, idx: int) -> dict:
        p = self.frames / f"{idx:04d}.json"
        if not p.exists():
            raise FileNotFoundError(f"frame {idx} not in session {self.root}")
        return json.loads(p.read_text())

    def ingest(self, img: Path, action: str | None) -> str:
        if not img.exists():
            raise FileNotFoundError(img)
        idx = self.count()
        stored = self.frames / f"{idx:04d}_{(action or 'start').replace(' ', '_')}{img.suffix.lower()}"
        shutil.copy2(img, stored)
        pal = Palette.load(self.pal_path)
        fr = sample_frame(stored, pal)
        objs, meta = extract_objects(fr.grid, pal)
        prev = self.load(idx - 1) if idx > 0 else None
        rep = render(fr, pal, objs, meta, idx, action, prev)
        pal.save(self.pal_path)
        rec = {"idx": idx, "action": action, "image": stored.name, "lattice": asdict(fr.lattice),
               "grid": fr.grid.tolist(), "objects": [asdict(o) for o in objs]}
        (self.frames / f"{idx:04d}.json").write_text(json.dumps(rec))
        (self.frames / f"{idx:04d}.txt").write_text(rep)
        with self.log_path.open("a") as f:
            f.write(json.dumps({"idx": idx, "action": action, "image": stored.name}) + "\n")
        return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=Path, default=Path(__file__).resolve().parent / "arc_session")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest"); s.add_argument("image", type=Path)
    s.add_argument("--action", default=None); s.add_argument("--new", action="store_true")
    s = sub.add_parser("report"); s.add_argument("idx", type=int)
    s = sub.add_parser("diff"); s.add_argument("a", type=int); s.add_argument("b", type=int)
    sub.add_parser("log")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    ses = Session(a.session)
    try:
        if a.cmd == "ingest":
            img = a.image.resolve()
            if a.new:
                # stage the image outside the session first: it may live inside the folder being wiped
                import tempfile
                tmp = Path(tempfile.mkdtemp()) / img.name
                shutil.copy2(img, tmp)
                ses.reset()
                img = tmp
            print(ses.ingest(img, a.action))
        elif a.cmd == "report":
            print((ses.frames / f"{a.idx:04d}.txt").read_text())
        elif a.cmd == "diff":
            A, B = ses.load(a.a), ses.load(a.b)
            print(f"=== DIFF frame {a.a} -> {a.b} ===")
            print("\n".join(cell_diff(np.array(A["grid"]), np.array(B["grid"]))))
            print("TRACKING")
            cur = [Obj(**o) for o in B["objects"]]
            print("\n".join(track(A["objects"], cur)) or "  no object moved or changed")
        elif a.cmd == "log":
            if ses.log_path.exists():
                print(ses.log_path.read_text())
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
