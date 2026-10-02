"""
scene.py -- shared perception core for experiments X1..X8.

screenshot --> exact 64x64 ARC palette grid --> scene objects.
Every experiment uses the SAME segmentation; experiments differ only in what they compute on top of
it and how they present it. That isolates the variable we are testing: the presentation/inference.
"""
from __future__ import annotations

import hashlib
import os
import sys
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "baseline"))
from arc_eye import fit_lattice  # noqa: E402  (edge-based lattice fit, validated earlier)

N = 64
PAL_HEX = ["#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333", "#000000", "#E53AA3", "#FF7BCC",
           "#F93C31", "#1E93FF", "#88D8F1", "#FFDC00", "#FF851B", "#921231", "#4FCC30", "#A356D6"]
PAL = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in PAL_HEX], dtype=np.uint8)
NAMES = ["white", "light grey", "grey", "dark grey", "charcoal", "black", "magenta", "pink", "red",
         "blue", "light blue", "yellow", "orange", "maroon", "green", "purple"]
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


FONT_CANDIDATES = {
    False: [FONT_PATH, "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf", "/Library/Fonts/Arial.ttf", "DejaVuSans.ttf"],
    True: [FONT_BOLD, "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf", "/Library/Fonts/Arial Bold.ttf", "DejaVuSans-Bold.ttf"],
}


def font(size: int, bold: bool = False):
    """Works on Linux, Windows and macOS; falls back to Pillow's built-in scalable font."""
    for path in FONT_CANDIDATES[bold]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ----------------------------------------------------------------------------- screenshot -> grid
def load_grid(path: str) -> tuple[np.ndarray, list[str]]:
    """Return (64x64 int grid, notes). Palette-snapped; trims a surplus edge line by least visibility."""
    im = np.asarray(Image.open(path).convert("RGB")).astype(float)
    H, W, _ = im.shape
    lat = fit_lattice(im)
    P = lat.pitch
    m = max(1, int(round(P * 0.25)))
    g = np.full((lat.rows, lat.cols), -1, int)
    for r in range(lat.rows):
        for c in range(lat.cols):
            ya, yb = max(0, int(round(lat.y0 + r * P)) + m), min(H, int(round(lat.y0 + (r + 1) * P)) - m)
            xa, xb = max(0, int(round(lat.x0 + c * P)) + m), min(W, int(round(lat.x0 + (c + 1) * P)) - m)
            if yb - ya < 1 or xb - xa < 1:
                continue
            px = np.median(im[ya:yb, xa:xb].reshape(-1, 3), 0)
            g[r, c] = int(np.argmin(((PAL.astype(float) - px) ** 2).sum(1)))
    notes = [f"lattice pitch {P:.3f}px, raw {lat.rows}x{lat.cols}"]
    vr = [(min(H, lat.y0 + (r + 1) * P) - max(0, lat.y0 + r * P)) / P for r in range(lat.rows)]
    vc = [(min(W, lat.x0 + (c + 1) * P) - max(0, lat.x0 + c * P)) / P for c in range(lat.cols)]
    while g.shape[0] > N:
        drop = 0 if vr[0] < vr[-1] else -1
        g = np.delete(g, drop, 0); vr.pop(drop); notes.append(f"dropped {'first' if drop == 0 else 'last'} row (cut off)")
    while g.shape[1] > N:
        drop = 0 if vc[0] < vc[-1] else -1
        g = np.delete(g, drop, 1); vc.pop(drop); notes.append(f"dropped {'first' if drop == 0 else 'last'} column (cut off)")
    if g.shape != (N, N):
        pad = np.full((N, N), -1, int); pad[:g.shape[0], :g.shape[1]] = g; g = pad; notes.append("padded")
    return g, notes


def to_image(g: np.ndarray, cell: int = 8, dim: float | np.ndarray = 0.0, grid_lines: bool = False) -> Image.Image:
    """Render grid. dim: 0..1 scalar or per-cell array (fade toward mid grey)."""
    rgb = PAL[np.clip(g, 0, 15)].astype(float)
    rgb[g < 0] = (40, 40, 40)
    d = np.broadcast_to(np.asarray(dim, float), g.shape)[..., None]
    rgb = rgb * (1 - d) + np.array([128, 128, 128]) * d
    img = Image.fromarray(np.repeat(np.repeat(rgb.astype(np.uint8), cell, 0), cell, 1))
    if grid_lines:
        dr = ImageDraw.Draw(img)
        for i in range(0, N * cell, cell):
            dr.line([(i, 0), (i, N * cell)], fill=(70, 70, 70)); dr.line([(0, i), (N * cell, i)], fill=(70, 70, 70))
    return img


# ----------------------------------------------------------------------------- components
NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
NB4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def components(mask: np.ndarray, conn8: bool = True) -> list[list[tuple[int, int]]]:
    nb = NB8 if conn8 else NB4
    seen = np.zeros(mask.shape, bool); out = []
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        q = deque([(r, c)]); seen[r, c] = True; cells = []
        while q:
            y, x = q.popleft(); cells.append((int(y), int(x)))
            for dy, dx in nb:
                yy, xx = y + dy, x + dx
                if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; q.append((yy, xx))
        out.append(cells)
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    pad = np.pad(mask, 1)
    outside = np.zeros(pad.shape, bool); q = deque([(0, 0)]); outside[0, 0] = True
    while q:
        y, x = q.popleft()
        for dy, dx in NB4:
            yy, xx = y + dy, x + dx
            if 0 <= yy < pad.shape[0] and 0 <= xx < pad.shape[1] and not pad[yy, xx] and not outside[yy, xx]:
                outside[yy, xx] = True; q.append((yy, xx))
    return ~outside[1:-1, 1:-1]


def shape_hash(mask: np.ndarray) -> str:
    return hashlib.md5(str(mask.shape).encode() + np.packbits(mask).tobytes()).hexdigest()[:8]


def d4(mask: np.ndarray) -> list[tuple[str, np.ndarray]]:
    out = []
    for k, nm in enumerate(["", "rot90ccw", "rot180", "rot90cw"]):
        r = np.rot90(mask, k)
        out.append((nm or "same", r)); out.append(((nm + "+" if nm else "") + "flipLR", np.fliplr(r)))
    return out


# ----------------------------------------------------------------------------- scene objects
@dataclass
class Obj:
    id: int
    kind: str                      # object | icon | dotted | line | bar
    cells: list[tuple[int, int]]
    colors: Counter
    r0: int; c0: int; r1: int; c1: int
    on: int = -1                   # terrain colour it sits on
    panel: int | None = None       # id of enclosing panel (HUD box) if any
    name: str = ""
    desc: str = ""
    tags: dict = field(default_factory=dict)

    @property
    def h(self): return self.r1 - self.r0 + 1
    @property
    def w(self): return self.c1 - self.c0 + 1
    @property
    def size(self): return len(self.cells)
    @property
    def cy(self): return float(np.mean([c[0] for c in self.cells]))
    @property
    def cx(self): return float(np.mean([c[1] for c in self.cells]))

    def mask(self) -> np.ndarray:
        m = np.zeros((self.h, self.w), bool)
        for y, x in self.cells:
            m[y - self.r0, x - self.c0] = True
        return m

    def colgrid(self, g: np.ndarray) -> np.ndarray:
        out = np.full((self.h, self.w), -1, int)
        for y, x in self.cells:
            out[y - self.r0, x - self.c0] = g[y, x]
        return out

    def label(self) -> str:
        return f"{self.name}"


@dataclass
class Panel:
    id: int
    color: int
    r0: int; c0: int; r1: int; c1: int
    border: bool          # touches the frame border


@dataclass
class Scene:
    g: np.ndarray
    terrain: list[int]
    objects: list[Obj]
    panels: list[Panel]
    notes: list[str]


def _is_thin(cells) -> bool:
    s = set(cells)
    return all(sum((y + dy, x + dx) in s for dy, dx in NB8) <= 2 for y, x in cells)


def build_scene(g: np.ndarray, notes: list[str] | None = None) -> Scene:
    cnt = Counter(g.ravel().tolist()); cnt.pop(-1, None)
    terrain = [c for c, n in cnt.most_common() if n / g.size > 0.08]
    fg = ~np.isin(g, terrain) & (g >= 0)

    # 1. per-colour pieces
    pieces = []
    for col in sorted(set(g[fg].tolist())):
        for cells in components((g == col) & fg, conn8=True):
            pieces.append((col, cells))

    # 2. thin long things: tethers / lines vs HUD bars on the frame edge
    lines, bars, rest = [], [], []
    for col, cells in pieces:
        ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
        h, w = max(ys) - min(ys) + 1, max(xs) - min(xs) + 1
        on_edge = min(ys) == 0 or min(xs) == 0 or max(ys) == N - 1 or max(xs) == N - 1
        if len(cells) >= 5 and _is_thin(cells):
            if (h == 1 or w == 1) and on_edge:
                bars.append((col, cells))
            else:
                lines.append((col, cells))
        elif max(h, w) >= 6 * min(h, w) and max(h, w) >= 8 and len(cells) == h * w:
            bars.append((col, cells))
        else:
            rest.append((col, cells))

    # 3. composites: touching pieces of different colours
    owner = {}
    for i, (col, cells) in enumerate(rest):
        for y, x in cells:
            owner[(y, x)] = i
    parent = list(range(len(rest)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
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

    # 4. panels = compact terrain boxes (HUD boxes, lock boxes) that are not the main field
    panels: list[Panel] = []
    for col in terrain:
        for cells in components(g == col, conn8=False):
            m = np.zeros(g.shape, bool)
            for y, x in cells:
                m[y, x] = True
            mf = fill_holes(m)
            ys, xs = np.nonzero(mf)
            r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
            box = (r1 - r0 + 1) * (c1 - c0 + 1)
            if mf.sum() / box >= 0.9 and box <= 0.12 * g.size and min(r1 - r0, c1 - c0) >= 2:
                border = r0 == 0 or c0 == 0 or r1 == N - 1 or c1 == N - 1
                panels.append(Panel(len(panels), col, int(r0), int(c0), int(r1), int(c1), border))

    def panel_of(cells):
        for p in panels:
            if all(p.r0 <= y <= p.r1 and p.c0 <= x <= p.c1 for y, x in cells):
                return p.id
        return None

    # 5. icons: same-colour pieces inside the same panel, unless they are identical copies (counters)
    raw = [(cells, panel_of(cells)) for cells in comps]
    used = set(); merged = []
    for i, (ci, pi) in enumerate(raw):
        if i in used:
            continue
        cols_i = {int(g[y, x]) for y, x in ci}
        grp = [i]
        if pi is not None:
            for j in range(i + 1, len(raw)):
                cj, pj = raw[j]
                if j in used or pj != pi or {int(g[y, x]) for y, x in cj} != cols_i:
                    continue
                if len(ci) == len(cj) and _norm(ci) == _norm(cj):
                    continue  # identical copy -> keep separate (counter / lives)
                grp.append(j)
        for k in grp:
            used.add(k)
        merged.append(([c for k in grp for c in raw[k][0]], pi, "icon" if len(grp) > 1 else "object"))

    # 6. dotted figures: many tiny same-colour pieces with <=1 cell gaps, outside panels
    tiny = [k for k, (cells, pi, _) in enumerate(merged) if len(cells) <= 4 and pi is None and _is_thin(cells)]
    parent2 = {k: k for k in tiny}

    def f2(a):
        while parent2[a] != a:
            a = parent2[a]
        return a
    for a in tiny:
        for b in tiny:
            if a < b and {int(g[y, x]) for y, x in merged[a][0]} == {int(g[y, x]) for y, x in merged[b][0]}:
                if min(max(abs(y1 - y2), abs(x1 - x2)) for y1, x1 in merged[a][0] for y2, x2 in merged[b][0]) <= 2:
                    parent2[f2(a)] = f2(b)
    dgroups: dict[int, list[int]] = {}
    for k in tiny:
        dgroups.setdefault(f2(k), []).append(k)
    final_cells = []
    consumed = set()
    for ks in dgroups.values():
        if len(ks) >= 3:
            final_cells.append(([c for k in ks for c in merged[k][0]], None, "dotted"))
            consumed.update(ks)
    for k, item in enumerate(merged):
        if k not in consumed:
            final_cells.append(item)
    for col, cells in lines:
        final_cells.append((cells, None, "line"))
    for col, cells in bars:
        final_cells.append((cells, panel_of(cells), "bar"))

    objs = []
    for cells, pi, kind in final_cells:
        ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
        o = Obj(0, kind, cells, Counter(int(g[y, x]) for y, x in cells), min(ys), min(xs), max(ys), max(xs), panel=pi)
        ring = [int(g[y, x]) for y in range(max(o.r0 - 1, 0), min(o.r1 + 2, N)) for x in range(max(o.c0 - 1, 0), min(o.c1 + 2, N))
                if (y, x) not in set(cells) and g[y, x] in terrain]
        o.on = Counter(ring).most_common(1)[0][0] if ring else -1
        objs.append(o)
    objs.sort(key=lambda o: (o.r0, o.c0))
    for i, o in enumerate(objs):
        o.id = i
        o.name, o.desc = name_object(o, g)
    return Scene(g, terrain, objs, panels, notes or [])


def _norm(cells):
    y0 = min(c[0] for c in cells); x0 = min(c[1] for c in cells)
    return sorted((y - y0, x - x0) for y, x in cells)


# ----------------------------------------------------------------------------- naming (geon vocabulary)
def shape_word(m: np.ndarray, kind: str) -> str:
    h, w = m.shape; n = int(m.sum())
    if kind == "line":
        return "dotted line" if n < max(h, w) else "line"
    if kind == "bar":
        return "bar"
    if kind == "dotted":
        ys, xs = np.nonzero(m)
        d = np.hypot(ys - ys.mean(), xs - xs.mean())
        return "dotted ring" if d.mean() >= 2 and d.std() / d.mean() < 0.25 else "dotted cluster"
    holes = int(fill_holes(m).sum() - n)
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
        if np.array_equal(m, (yy == 0) | (xx == 0)):
            return "plus"
        corners_cut = m.copy(); corners_cut[[0, 0, -1, -1], [0, -1, 0, -1]] = True
        if corners_cut.all():
            return "disc"
    if holes > 0:
        return "ring" if holes >= n // 4 else f"shape with {holes} hole cell(s)"
    if n <= 8:
        return {3: "L-tromino", 4: "tetromino", 5: "pentomino"}.get(n, f"{n}-cell shape")
    return f"irregular shape ({n} cells)"


def colour_phrase(o: Obj, g: np.ndarray) -> str:
    cg = o.colgrid(g)
    cols = [c for c, _ in o.colors.most_common()]
    if len(cols) == 1:
        return NAMES[cols[0]]
    h, w = cg.shape
    top, bot = cg[: h // 2], cg[(h + 1) // 2:]
    tc, bc = set(top[top >= 0].ravel()), set(bot[bot >= 0].ravel())
    rows_pure = all(len(set(row[row >= 0].tolist())) <= 1 for row in cg)
    if rows_pure and len(tc) == 1 and len(bc) == 1 and tc != bc:
        return f"{NAMES[tc.pop()]} top / {NAMES[bc.pop()]} bottom"
    if h % 2 and w % 2 and cg[h // 2, w // 2] >= 0 and o.colors[cg[h // 2, w // 2]] == 1:
        main = [c for c in cols if c != cg[h // 2, w // 2]]
        return f"{'/'.join(NAMES[c] for c in main)} with {NAMES[cg[h // 2, w // 2]]} centre"
    # where does each colour sit inside the object? (compass words)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    parts = []
    for c in cols:
        places = set()
        for y, x in zip(*np.nonzero(cg == c)):
            dy, dx = y - cy, x - cx
            if abs(dy) < 0.5 and abs(dx) < 0.5: places.add("centre")
            elif abs(dy) >= abs(dx): places.add("top" if dy < 0 else "bottom")
            else: places.add("left" if dx < 0 else "right")
        order = ["top", "right", "bottom", "left", "centre"]
        parts.append(f"{NAMES[c]} ({', '.join(p for p in order if p in places)})")
    return " + ".join(parts)


def name_object(o: Obj, g: np.ndarray) -> tuple[str, str]:
    sw = shape_word(o.mask(), o.kind)
    cp = colour_phrase(o, g)
    if o.kind == "object" and len(o.colors) > 1:
        # composite: describe by shape of the union
        pass
    name = f"{cp} {sw}"
    desc = f"{name}, {o.h}x{o.w}, at row {o.r0}-{o.r1}, col {o.c0}-{o.c1}"
    return name, desc


# ----------------------------------------------------------------------------- lattice unit
def lattice_unit(scene: Scene) -> tuple[int, int, int] | None:
    """Detect the game's movement/tile unit from the largest compact multi-colour object, and check
    that terrain boundaries align with it. Returns (unit, row_offset, col_offset) or None."""
    cands = [o for o in scene.objects if o.kind == "object" and o.h == o.w and o.h >= 3 and len(o.colors) >= 2]
    g = scene.g
    for o in sorted(cands, key=lambda o: -o.size):
        u = o.h; ro, co = o.r0 % u, o.c0 % u
        # boundary alignment: fraction of terrain-colour changes that sit on lattice lines
        hits = tot = 0
        fl = o.on
        for r in range(N):
            for c in range(1, N):
                a, b = g[r, c - 1], g[r, c]
                if a != b and fl in (a, b) and a in scene.terrain and b in scene.terrain:
                    tot += 1; hits += (c - co) % u == 0
        for c in range(N):
            for r in range(1, N):
                a, b = g[r - 1, c], g[r, c]
                if a != b and fl in (a, b) and a in scene.terrain and b in scene.terrain:
                    tot += 1; hits += (r - ro) % u == 0
        if tot and hits / tot > 0.75:
            return u, ro, co
        scene.notes.append(f"lattice test unit={u}: {hits}/{tot} floor edges on lattice")
    return None


def summary(scene: Scene) -> str:
    return "\n".join(f"  [{o.id}] {o.kind:6s} {o.desc}" for o in scene.objects)


if __name__ == "__main__":
    g, notes = load_grid(sys.argv[1])
    sc = build_scene(g, notes)
    print(notes, "terrain:", [NAMES[t] for t in sc.terrain])
    print(summary(sc))
    print("panels:", [(p.id, NAMES[p.color], p.r0, p.c0, p.r1, p.c1, p.border) for p in sc.panels])
    print("lattice unit:", lattice_unit(sc))
