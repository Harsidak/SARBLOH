"""Grid perception shared by the host (``act`` change summaries) and the REPL (``wm.objects`` and friends).

Pure Python on lists of lists; numpy arrays are accepted and converted. Nothing here spends an action or talks to
the host, so the host imports it as ``harness.runtime.skills.perception`` and the kernel as ``perception``.

    objects(grid)                  -> [Obj(colour, cells, bbox, size, shape)], one per 4-connected single-colour region
    summarize_change(before, after) -> "9 px changed: colour 12 5x5 moved (+0,-5); colour 11 1px 11->3 at (40,61)"
    state_key(grid)                -> a short hash of the grid (equal grids, equal keys)
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

Grid = list[list[int]]


def as_rows(grid: Any) -> Grid:
    grid = getattr(grid, "grid", grid)  # an Observation
    return grid.tolist() if hasattr(grid, "tolist") else [list(r) for r in grid]


def state_key(grid: Any) -> str:
    rows = as_rows(grid)
    return hashlib.blake2b(bytes(int(c) & 0xFF for r in rows for c in r) + bytes([len(rows) & 0xFF]),
                           digest_size=8).hexdigest()


def background(grid: Any) -> int:
    """The most common colour."""
    rows = as_rows(grid)
    return Counter(c for r in rows for c in r).most_common(1)[0][0] if rows else 0


@dataclass
class Obj:
    colour: int
    cells: frozenset[tuple[int, int]]   # (x, y)
    bbox: tuple[int, int, int, int]     # x0, y0, x1, y1 (inclusive)

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def shape(self) -> frozenset[tuple[int, int]]:
        """Cells relative to the bbox corner: equal shapes compare equal wherever they are."""
        return frozenset((x - self.bbox[0], y - self.bbox[1]) for x, y in self.cells)

    @property
    def wh(self) -> tuple[int, int]:
        return self.bbox[2] - self.bbox[0] + 1, self.bbox[3] - self.bbox[1] + 1

    def __repr__(self) -> str:
        w, h = self.wh
        return f"Obj(colour={self.colour}, size={self.size}, {w}x{h} at x={self.bbox[0]}..{self.bbox[2]}, " \
               f"y={self.bbox[1]}..{self.bbox[3]})"


def objects(grid: Any, bg: int | None = None, diagonal: bool = False, min_size: int = 1,
            region: set[tuple[int, int]] | None = None) -> list[Obj]:
    """Connected single-colour regions, largest first. ``bg`` (default: the most common colour) is skipped; pass
    ``bg=-1`` to keep every colour. ``region`` limits the search to components touching those (x, y) cells."""
    rows = as_rows(grid)
    if bg is None:
        bg = background(rows)
    h = len(rows)
    w = len(rows[0]) if h else 0
    seen = [[False] * w for _ in range(h)]
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)] + ([(1, 1), (1, -1), (-1, 1), (-1, -1)] if diagonal else [])
    seeds = sorted(region, key=lambda p: (p[1], p[0])) if region is not None else \
        [(x, y) for y in range(h) for x in range(w)]
    out = []
    for sx, sy in seeds:
        if not (0 <= sx < w and 0 <= sy < h) or seen[sy][sx] or rows[sy][sx] == bg:
            continue
        colour = rows[sy][sx]
        stack, cells = [(sx, sy)], []
        seen[sy][sx] = True
        while stack:
            x, y = stack.pop()
            cells.append((x, y))
            for dx, dy in steps:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and rows[ny][nx] == colour:
                    seen[ny][nx] = True
                    stack.append((nx, ny))
        if len(cells) >= min_size:
            xs, ys = [c[0] for c in cells], [c[1] for c in cells]
            out.append(Obj(int(colour), frozenset(cells), (min(xs), min(ys), max(xs), max(ys))))
    out.sort(key=lambda o: -o.size)
    return out


def diff_cells(before: Any, after: Any) -> list[tuple[int, int, int, int]]:
    a, b = as_rows(before), as_rows(after)
    if len(a) != len(b) or (a and len(a[0]) != len(b[0])):
        return []
    return [(x, y, int(a[y][x]), int(b[y][x])) for y in range(len(a)) for x in range(len(a[y])) if a[y][x] != b[y][x]]


def _describe(o: Obj) -> str:
    w, h = o.wh
    return f"colour {o.colour} {w}x{h}" if o.size == w * h else f"colour {o.colour} {o.size}px"


def summarize_change(before: Any, after: Any, max_items: int = 6) -> dict[str, Any]:
    """What changed between two grids, as objects rather than pixels.

    Returns ``{"changed": n_pixels, "text": "...", "moves": [(colour, dx, dy)], "bbox": (x0, y0, x1, y1) | None}``.
    An object that keeps its colour and shape but not its place is reported as moved; everything else in the
    changed area is reported as a colour change inside a box.
    """
    a, b = as_rows(before), as_rows(after)
    if len(a) != len(b) or (a and len(a[0]) != len(b[0])):
        return {"changed": -1, "text": f"grid size changed {len(a)}x{len(a[0]) if a else 0} -> "
                                       f"{len(b)}x{len(b[0]) if b else 0}", "moves": [], "bbox": None}
    cells = diff_cells(a, b)
    if not cells:
        return {"changed": 0, "text": "no change", "moves": [], "bbox": None}
    xs, ys = [c[0] for c in cells], [c[1] for c in cells]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    if len(cells) > 1500:
        return {"changed": len(cells), "text": f"{len(cells)} px changed (most of the screen) in x={bbox[0]}..{bbox[2]}, "
                                               f"y={bbox[1]}..{bbox[3]}", "moves": [], "bbox": bbox}
    region = {(x, y) for x, y, _, _ in cells}
    bg_a, bg_b = background(a), background(b)
    olds = [o for o in objects(a, bg=bg_a, region=region) if o.size <= 600]
    news = [o for o in objects(b, bg=bg_b, region=region) if o.size <= 600]
    parts: list[str] = []
    moves: list[tuple[int, int, int]] = []
    explained: set[tuple[int, int]] = set()
    used: set[int] = set()
    for o in olds:
        # A move is a short hop: far-apart vanish/appear pairs (single pixels, noise) stay colour changes.
        reach = max(6, 3 * max(o.bbox[2] - o.bbox[0] + 1, o.bbox[3] - o.bbox[1] + 1))
        match = [j for j, n in enumerate(news) if j not in used and n.colour == o.colour and n.shape == o.shape
                 and n.bbox != o.bbox and abs(n.bbox[0] - o.bbox[0]) + abs(n.bbox[1] - o.bbox[1]) <= reach]
        if len(match) >= 1:
            # nearest candidate of the same colour and shape
            j = min(match, key=lambda j: abs(news[j].bbox[0] - o.bbox[0]) + abs(news[j].bbox[1] - o.bbox[1]))
            n = news[j]
            used.add(j)
            dx, dy = n.bbox[0] - o.bbox[0], n.bbox[1] - o.bbox[1]
            moves.append((o.colour, dx, dy))
            parts.append(f"{_describe(o)} at ({o.bbox[0]},{o.bbox[1]}) moved ({dx:+d},{dy:+d})")
            explained |= o.cells | n.cells
    rest = [c for c in cells if (c[0], c[1]) not in explained]
    if rest:
        by_pair = Counter((old, new) for _, _, old, new in rest)
        for (old, new), n in by_pair.most_common(max_items):
            pts = [(x, y) for x, y, o, nw in rest if (o, nw) == (old, new)]
            px, py = [p[0] for p in pts], [p[1] for p in pts]
            where = f"at ({px[0]},{py[0]})" if n == 1 else f"in x={min(px)}..{max(px)}, y={min(py)}..{max(py)}"
            parts.append(f"{n}px {old}->{new} {where}")
        if len(by_pair) > max_items:
            parts.append(f"+{len(by_pair) - max_items} more colour pairs")
    return {"changed": len(cells), "text": f"{len(cells)} px changed: " + "; ".join(parts[:max_items + 2]),
            "moves": moves, "bbox": bbox}
