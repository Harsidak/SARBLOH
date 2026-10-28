"""Geometry on boards, pre-imported by the kernel: turn and mirror a board or a piece of one, find what transform
turns one shape into another, find a shape on the board, check symmetry, spot a scrolled view, walk rays, find paths,
track the numbers on screen and the other movers, find the moves that undo each other, check and search the agent's
own ``step()`` model, map what this level has tried and the world of a scrolling level, show how earlier levels were
won and rank what makes something happen.

A board is what ``obs.board`` holds (a list of strings, one letter per cell), a list of lists or a numpy array; a
function that returns a board returns the same kind it was given (numbers stay numbers). Positions are (row, column),
boxes are [r0, c0, r1, c1] inclusive, as in ``obs.objects``. An object dict from ``obs.objects`` (with its ``cells``)
is accepted wherever a piece is. Nothing here spends a move.

Offered in the prompt (the axes the games are built from: transforms, reversibility, propagation, other agents,
goals):
    rotate(b, 1); flip(b, "h")
    relate(a, b)          # which turn / mirror / scale / recolour turns a into b, and how well
    find(piece, board)    # where a piece is on the board, turned and mirrored copies too (recolour=True: any colours)
    symmetry(b)           # mirror and turn symmetry, and the cells that break it
    camera(before, after) # did the whole view scroll, and by how much
    ray(b, r, c, dr, dc); reach(b, start, passable, goal)
    counters()            # the letters whose cell count changes move after move (bars, counters, lives)
    track(letter)         # where each object of a letter was after every move, next to the move sent
    undo_pairs()          # which moves brought the board back to an earlier one, and which never did
    verify(step)          # how many of this level's moves your step() model gets right, and where it is wrong
    atlas()               # the boards this level has seen, the moves tried and never tried from each, paths back
    world(you)            # a scrolling level as one map, with you on it
    wins()                # the board before each earlier level's winning move
    cause(what, you)      # the conditions that tell the moves that changed something from those that did not
    search(start, step, is_goal, moves)   # breadth-first search over your own step() model
Kept for the tools above and the curator, not offered: transpose, shift, crop, paste, scale, recolor, line.
"""

from __future__ import annotations

import copy
import time
from collections import Counter, deque
from typing import Any, Callable, Iterable

import numpy as np

EMPTY = "."   # a cell outside a piece made from an object's cells; ``find`` and ``relate`` treat it as "any"
_TURNS = ("same", "rotate 90", "rotate 180", "rotate 270", "mirror left-right", "mirror top-bottom", "transpose",
          "anti-transpose")


# --- boards in and out ---------------------------------------------------------------------------------------------

def _arr(g: Any) -> np.ndarray:
    """A board, piece or object dict as a 2-D array of one-character strings."""
    if isinstance(g, dict) and g.get("cells"):
        cells = [tuple(x) for x in g["cells"]]
        r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
        h, w = max(r for r, _ in cells) - r0 + 1, max(c for _, c in cells) - c0 + 1
        a = np.full((h, w), EMPTY, dtype="<U1")
        for r, c in cells:
            a[r - r0, c - c0] = str(g.get("letter", "X"))[:1]
        return a
    if isinstance(g, dict):
        raise ValueError("this object has no cells (objects over 256 cells have cells=None): crop it from the board")
    if isinstance(g, np.ndarray):
        if g.ndim != 2:
            return np.array([list(str(x)) for x in g], dtype="<U1")
        if g.dtype.kind in "iub":   # numbers: one character each, 10.. as letters, so 12 never becomes "1"
            return np.vectorize(_cell, otypes=["<U1"])(g) if g.size else np.zeros(g.shape, "<U1")
        return g.astype("<U1")
    rows = list(g)
    if not rows:
        return np.zeros((0, 0), dtype="<U1")
    if len({len(r) for r in rows}) > 1:
        raise ValueError("every row of a board or piece must have the same length; use '.' for an empty cell")
    if all(isinstance(r, str) for r in rows):   # rows of letters: read the text straight in, one cell per letter
        w = len(rows[0])
        if w == 0:
            return np.zeros((len(rows), 0), dtype="<U1")
        return np.frombuffer("".join(rows).encode("utf-32-le"), dtype="<U1").reshape(len(rows), w).copy()
    return np.array([[_cell(v) for v in r] for r in rows], dtype="<U1")


def _cell(v: Any) -> str:
    s = str(v)
    if len(s) == 1:
        return s
    if isinstance(v, (int, np.integer)) and 10 <= int(v) <= 35:
        return chr(ord("a") + int(v) - 10)   # an int grid: 10.. as one character, so shapes stay one cell wide
    return s[:1]


def _uncell(s: str) -> int:
    """The number ``_cell`` wrote as one character ("7" -> 7, "c" -> 12)."""
    return int(s) if s.isdigit() else ord(s) - ord("a") + 10


def _numbers(like: Any) -> bool:
    """Whether ``like`` is a grid of numbers 0..35 (a numeric numpy array, or lists of ints), the ones that go
    through one character each and come back."""
    if isinstance(like, np.ndarray):
        return like.ndim == 2 and like.dtype.kind in "iu" and (like.size == 0 or (like.min() >= 0 and like.max() <= 35))
    if isinstance(like, (list, tuple)) and like and isinstance(like[0], (list, tuple, np.ndarray)):
        return bool(any(len(r) for r in like)) and all(
            isinstance(v, (int, np.integer)) and not isinstance(v, bool) and 0 <= v <= 35 for r in like for v in r)
    return False


def _out(a: np.ndarray, like: Any) -> Any:
    """``a`` back in the kind of ``like``: a list of strings for a list of strings or an object, numbers for a grid
    of numbers, else lists."""
    if isinstance(like, dict) or (isinstance(like, (list, tuple)) and like and isinstance(like[0], str)):
        return ["".join(r) for r in a.tolist()]
    if _numbers(like):
        n = np.vectorize(_uncell, otypes=[int])(a) if a.size else np.zeros(a.shape, int)
        return n.astype(like.dtype) if isinstance(like, np.ndarray) else n.tolist()
    if isinstance(like, np.ndarray):
        return a
    return a.tolist()


def _background(a: np.ndarray) -> str:
    vals, counts = np.unique(a, return_counts=True)
    return str(vals[int(np.argmax(counts))]) if len(vals) else EMPTY


def _variants(a: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [("same", a), ("rotate 90", np.rot90(a, -1)), ("rotate 180", np.rot90(a, 2)),
            ("rotate 270", np.rot90(a, 1)), ("mirror left-right", a[:, ::-1]), ("mirror top-bottom", a[::-1, :]),
            ("transpose", a.T), ("anti-transpose", np.rot90(a, 2).T)]


# --- transforms ----------------------------------------------------------------------------------------------------

def rotate(g: Any, turns: int = 1) -> Any:
    """The board turned clockwise by 90 degrees ``turns`` times (negative: anticlockwise)."""
    return _out(np.rot90(_arr(g), -int(turns)), g)


def flip(g: Any, axis: str = "h") -> Any:
    """The board mirrored: "h" left-right (columns reversed), "v" top-bottom (rows reversed)."""
    a = _arr(g)
    if axis not in ("h", "v"):
        raise ValueError('axis is "h" (left-right) or "v" (top-bottom)')
    return _out(a[:, ::-1] if axis == "h" else a[::-1, :], g)


def transpose(g: Any) -> Any:
    """Rows become columns."""
    return _out(_arr(g).T, g)


def shift(g: Any, dr: int, dc: int, fill: Any = None) -> Any:
    """The board moved down ``dr`` rows and right ``dc`` columns (negative: up, left); what comes in is ``fill``
    (default: the most common letter)."""
    a = _arr(g)
    fill = _background(a) if fill is None else _cell(fill)
    out = np.full(a.shape, fill, dtype="<U1")
    h, w = a.shape
    dr, dc = int(dr), int(dc)
    if abs(dr) < h and abs(dc) < w:
        out[max(dr, 0):h + min(dr, 0), max(dc, 0):w + min(dc, 0)] = a[max(-dr, 0):h - max(dr, 0),
                                                                      max(-dc, 0):w - max(dc, 0)]
    return _out(out, g)


def crop(g: Any, r0: int, c0: int, r1: int, c1: int) -> Any:
    """Rows r0..r1 and columns c0..c1 (inclusive, like a bbox)."""
    return _out(_arr(g)[int(r0):int(r1) + 1, int(c0):int(c1) + 1], g)


def paste(g: Any, piece: Any, r: int, c: int, skip: Any = EMPTY) -> Any:
    """The board with ``piece`` written at row ``r``, column ``c`` (its top-left corner); cells of the piece equal to
    ``skip`` leave the board as it is. The part outside the board is cut off."""
    a, p = _arr(g).copy(), _arr(piece)
    h, w = a.shape
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            rr, cc = int(r) + i, int(c) + j
            if 0 <= rr < h and 0 <= cc < w and (skip is None or p[i, j] != str(skip)):
                a[rr, cc] = p[i, j]
    return _out(a, g)


def scale(g: Any, k: float) -> Any:
    """The board scaled by ``k``: 2, 3 ... repeat each cell k x k times; 1/2, 1/3 ... keep one cell of each block."""
    a = _arr(g)
    if k >= 1:
        n = int(round(k))
        return _out(np.repeat(np.repeat(a, n, axis=0), n, axis=1), g)
    n = int(round(1 / k))
    return _out(a[::n, ::n], g)


def recolor(g: Any, mapping: dict[Any, Any]) -> Any:
    """The board with letters replaced by ``mapping`` ({"R": "B"}); letters not in it stay."""
    a = _arr(g)
    out = a.copy()
    for old, new in mapping.items():
        out[a == _cell(old)] = _cell(new)
    return _out(out, g)


# --- comparing shapes ----------------------------------------------------------------------------------------------

def _match(a: np.ndarray, b: np.ndarray, recolour: bool) -> tuple[float, dict[str, str]]:
    """How well two arrays of the same size agree: the share of cells that agree (an EMPTY cell agrees with an
    EMPTY cell only), with the most common colour map when ``recolour``."""
    care = (a != EMPTY) | (b != EMPTY)
    n = int(care.sum())
    if n == 0:
        return 1.0, {}
    if not recolour:
        return float(((a == b) & care).sum()) / n, {}
    pairs = Counter((x, y) for x, y in zip(a[care].tolist(), b[care].tolist()))
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for (x, y), _ in pairs.most_common():
        if x not in mapping and y not in used and (x == EMPTY) == (y == EMPTY):
            mapping[x], used = y, used | {y}
    agree = sum(cnt for (x, y), cnt in pairs.items() if mapping.get(x) == y)
    return agree / n, {x: y for x, y in mapping.items() if x != y}


def relate(a: Any, b: Any, recolour: bool = True, top: int = 3) -> list[dict[str, Any]]:
    """Which transform turns piece ``a`` into piece ``b``: each of the 8 turns and mirrors, at scales 1, 2, 3, 1/2 and
    1/3, with a colour map when ``recolour``. Returns the best ``top`` as dicts: ``transform`` (e.g. "rotate 90"),
    ``scale``, ``colors`` (the letters it changes), ``match`` (share of cells that agree, 1.0 = exact); for two
    objects also ``by`` [rows, columns] from a's box to b's. Best first; an empty list when no size fits."""
    pa, pb = _arr(a), _arr(b)
    found = []
    for k in (1, 2, 3, 1 / 2, 1 / 3):
        sa = _arr(scale(pa, k)) if k != 1 else pa
        for name, v in _variants(sa):
            if v.shape != pb.shape:
                continue
            m, cmap = _match(v, pb, recolour)
            found.append({"transform": name, "scale": k if k >= 1 else f"1/{int(round(1 / k))}",
                          "colors": cmap, "match": round(m, 3)})
    found.sort(key=lambda d: (-d["match"], _TURNS.index(d["transform"]), len(d["colors"])))
    if isinstance(a, dict) and isinstance(b, dict) and a.get("bbox") and b.get("bbox"):
        by = [b["bbox"][0] - a["bbox"][0], b["bbox"][1] - a["bbox"][1]]
        for d in found:
            d["by"] = by
    return found[:max(1, int(top))]


def find(piece: Any, board: Any, turns: bool = True, min_match: float = 1.0, limit: int = 20,
         recolour: bool = False) -> list[dict[str, Any]]:
    """Where ``piece`` is on ``board``: every place where at least ``min_match`` of its cells agree (cells equal to "."
    match anything), also turned and mirrored copies when ``turns``. With ``recolour`` a copy drawn in other colours
    counts too (each colour of the piece one other colour, never the background): the way to find a target, a key or
    a panel that shows the shape you control. Returns dicts ``at`` [r, c] (top-left corner), ``bbox``, ``transform``
    and ``match`` (and ``colors``, the colour changes, with ``recolour``), best first, at most ``limit``."""
    from numpy.lib.stride_tricks import sliding_window_view
    bd, p0 = _arr(board), _arr(piece)
    if recolour:
        return _find_recoloured(p0, bd, turns, min_match, limit)
    out, seen = [], set()
    for name, p in (_variants(p0) if turns else [("same", p0)]):
        h, w = p.shape
        if h == 0 or w == 0 or h > bd.shape[0] or w > bd.shape[1] or (p.shape, p.tobytes()) in seen:
            continue
        seen.add((p.shape, p.tobytes()))
        care = p != EMPTY
        n = int(care.sum())
        if n == 0:
            continue
        win = sliding_window_view(bd, (h, w))
        score = ((win == p) & care).sum(axis=(2, 3)) / n
        for r, c in zip(*np.nonzero(score >= min_match - 1e-9)):
            out.append({"at": [int(r), int(c)], "bbox": [int(r), int(c), int(r) + h - 1, int(c) + w - 1],
                        "transform": name, "match": round(float(score[r, c]), 3)})
    out.sort(key=lambda d: (-d["match"], _TURNS.index(d["transform"]), d["at"]))
    return out[:max(1, int(limit))]


def _find_recoloured(p0: np.ndarray, bd: np.ndarray, turns: bool, min_match: float,
                     limit: int) -> list[dict[str, Any]]:
    """``find`` with each colour of the piece free to be one other colour: per colour, the share of its cells that
    agree with the colour under its first cell."""
    from numpy.lib.stride_tricks import sliding_window_view
    names, codes = np.unique(bd, return_inverse=True)
    codes = codes.reshape(bd.shape)
    bg = _background(bd)
    out, seen = [], set()
    for name, p in (_variants(p0) if turns else [("same", p0)]):
        h, w = p.shape
        if h == 0 or w == 0 or h > bd.shape[0] or w > bd.shape[1] or (p.shape, p.tobytes()) in seen:
            continue
        seen.add((p.shape, p.tobytes()))
        letters = [x for x in np.unique(p).tolist() if x != EMPTY]
        n = int((p != EMPTY).sum())
        if not letters:
            continue
        win = sliding_window_view(codes, (h, w))
        agree = np.zeros(win.shape[:2])
        firsts = []
        for x in letters:
            rr, cc = np.nonzero(p == x)
            vals = win[:, :, rr, cc]
            agree += (vals == vals[:, :, :1]).sum(axis=2)
            firsts.append(vals[:, :, 0])
        score = agree / n
        ok = score >= min_match - 1e-9
        for i, x in enumerate(letters):   # one colour per piece colour, and the background only for itself
            if x != bg:
                ok &= names[firsts[i]] != bg
            for j in range(i):
                ok &= firsts[i] != firsts[j]
        for r, c in zip(*np.nonzero(ok)):
            colors = {x: str(names[f[r, c]]) for x, f in zip(letters, firsts) if str(names[f[r, c]]) != x}
            out.append({"at": [int(r), int(c)], "bbox": [int(r), int(c), int(r) + h - 1, int(c) + w - 1],
                        "transform": name, "match": round(float(score[r, c]), 3), "colors": colors})
    out.sort(key=lambda d: (-d["match"], len(d["colors"]), _TURNS.index(d["transform"]), d["at"]))
    return out[:max(1, int(limit))]


def symmetry(g: Any, limit: int = 20) -> dict[str, Any]:
    """How symmetric a board or piece is about its own centre: for "mirror left-right", "mirror top-bottom",
    "rotate 180" (and "rotate 90", "transpose" when square) the share of cells that agree and up to ``limit``
    cells [r, c, letter now, letter its mirror has] that break it. Crop the region first to test a part."""
    a = _arr(g)
    tests = {"mirror left-right": a[:, ::-1], "mirror top-bottom": a[::-1, :], "rotate 180": np.rot90(a, 2)}
    if a.shape[0] == a.shape[1]:
        tests.update({"rotate 90": np.rot90(a, -1), "transpose": a.T})
    out = {}
    for name, m in tests.items():
        bad = np.argwhere(a != m)
        out[name] = {"match": round(1 - len(bad) / max(1, a.size), 3),
                     "breaks": [[int(r), int(c), str(a[r, c]), str(m[r, c])] for r, c in bad[:limit]]}
    return out


def camera(before: Any, after: Any, max_shift: int = 16, shifts: Iterable[Any] | None = None) -> dict[str, Any]:
    """Did the whole view move? The shift (rows, columns) that makes ``after`` agree best with ``before`` moved by it,
    counted over the cells that are not background in either; every shift up to ``max_shift`` cells is tried, or only
    the (rows, columns) pairs in ``shifts``, as long as the part that still overlaps holds at least half of those
    cells. Returns ``by`` [rows, columns] ([0, 0] when no shift explains the change
    better than none), ``match`` (share of those cells that agree after the shift) and ``still`` (the share with no
    shift). Board ``after`` = ``shift(before, *by)`` where the view scrolled."""
    a, b = _arr(before), _arr(after)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("camera needs two boards of the same size")
    bg = _background(a)
    h, w = a.shape

    def score(dr: int, dc: int) -> float:
        pa = a[max(-dr, 0):h - max(dr, 0), max(-dc, 0):w - max(dc, 0)]
        pb = b[max(dr, 0):h + min(dr, 0), max(dc, 0):w + min(dc, 0)]
        care = (pa != bg) | (pb != bg)
        n = int(care.sum())
        if n < need:   # a shift that leaves only a sliver of the view overlapping proves nothing
            return 0.0
        return float(((pa == pb) & care).sum()) / n if n else 0.0

    need = 0
    still = score(0, 0)
    need = 0.5 * int(((a != bg) | (b != bg)).sum())
    best, by = still, (0, 0)
    m = max(0, min(int(max_shift), h - 1, w - 1))
    tries = [(dr, dc) for dr in range(-m, m + 1) for dc in range(-m, m + 1)] if shifts is None else \
        [(int(s[0]), int(s[1])) for s in shifts if abs(int(s[0])) < h and abs(int(s[1])) < w]
    for dr, dc in tries:
        if (dr, dc) != (0, 0):
            s = score(dr, dc)
            if s > best + 1e-9:
                best, by = s, (dr, dc)
    if by != (0, 0) and best < max(0.6, still + 0.15):
        best, by = still, (0, 0)
    return {"by": [by[0], by[1]], "match": round(best, 3), "still": round(still, 3)}


# --- lines and paths -----------------------------------------------------------------------------------------------

def line(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """The cells on the straight line from (r0, c0) to (r1, c1), both ends included."""
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 >= r0 else -1), (1 if c1 >= c0 else -1)
    err, out = dc - dr, []
    while True:
        out.append((r0, c0))
        if (r0, c0) == (r1, c1):
            return out
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c0 += sc
        if e2 < dc:
            err += dc
            r0 += sr


def ray(g: Any, r: int, c: int, dr: int, dc: int, through: Iterable[Any] | None = None) -> dict[str, Any]:
    """Walks from (r, c) in steps of (dr, dc) while the cells are in ``through`` (default: the background letter).
    Returns ``cells`` (the free cells walked, the start not included), ``hit`` [r, c, letter] of the first cell that
    stopped it, or None when it left the board."""
    a = _arr(g)
    ok = {_background(a)} if through is None else {_cell(x) for x in through}
    h, w = a.shape
    cells = []
    rr, cc = int(r) + int(dr), int(c) + int(dc)
    while 0 <= rr < h and 0 <= cc < w:
        if a[rr, cc] not in ok:
            return {"cells": cells, "hit": [rr, cc, str(a[rr, cc])]}
        cells.append((rr, cc))
        rr, cc = rr + int(dr), cc + int(dc)
    return {"cells": cells, "hit": None}


_DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def reach(g: Any, start: tuple[int, int], passable: Any, goal: Any = None, step: int = 1,
          size: tuple[int, int] = (1, 1)) -> dict[str, Any]:
    """Shortest paths on the board for a block of ``size`` (rows, columns) whose top-left corner is at ``start``,
    moving ``step`` cells up, down, left or right, onto cells whose letters are in ``passable`` (letters, or a function
    letter -> bool). ``goal``: a corner (r, c), a list of corners, a function (r, c) -> bool, or letters ("Y"): the
    block covers a cell of one of them (those letters count as passable). Returns ``path`` (direction names, None
    when the goal cannot be reached, [] with no goal), ``corners`` along it, ``reachable`` (how many corners can be
    reached) and ``dist`` (corner -> moves, for every reachable corner)."""
    a = _arr(g)
    goal_letters = {_cell(x) for x in goal} if isinstance(goal, str) else None
    ok = passable if callable(passable) else (lambda x, s={_cell(p) for p in passable}: x in s)
    if goal_letters:
        ok = (lambda x, base=ok: x in goal_letters or bool(base(x)))
    h, w = a.shape
    bh, bw = int(size[0]), int(size[1])
    free = np.vectorize(lambda x: bool(ok(x)), otypes=[bool])(a) if a.size else np.zeros((0, 0), bool)

    def fits(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + bh <= h and c + bw <= w and bool(free[r:r + bh, c:c + bw].all())

    if goal is None:
        is_goal = lambda r, c: False
    elif goal_letters is not None:
        is_goal = lambda r, c: bool(np.isin(a[r:r + bh, c:c + bw], list(goal_letters)).any())
    elif callable(goal):
        is_goal = goal
    else:
        targets = {tuple(goal)} if len(goal) == 2 and all(isinstance(x, (int, np.integer)) for x in goal) \
            else {tuple(x) for x in goal}
        is_goal = lambda r, c: (r, c) in targets
    s = (int(start[0]), int(start[1]))
    prev: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {s: None}
    dist = {s: 0}
    q, end = deque([s]), None
    while q:
        cur = q.popleft()
        if goal is not None and is_goal(*cur):
            end = cur
            break
        for name, (dr, dc) in _DIRS.items():
            nxt = (cur[0] + dr * int(step), cur[1] + dc * int(step))
            if nxt not in prev and fits(*nxt):
                prev[nxt], dist[nxt] = (cur, name), dist[cur] + 1
                q.append(nxt)
    path: list[str] | None = [] if goal is None else None
    corners = [s]
    if end is not None:
        path, corners, cur = [], [], end
        while prev[cur] is not None:
            corners.append(cur)
            cur, name = prev[cur]
            path.append(name)
        corners.append(s)
        path.reverse()
        corners.reverse()
    return {"path": path, "corners": corners, "reachable": len(dist), "dist": dist}


# --- the numbers on screen -----------------------------------------------------------------------------------------

def counters(rows: list[dict[str, Any]] | None = None, min_share: float = 0.3) -> list[dict[str, Any]]:
    """The letters whose number of cells changes from move to move, as bars, counters and lives do (the background
    letter is left out). ``rows``: the
    moves to read (default ``moves()``, this level). For each letter that changed on at least ``min_share`` of the
    moves: ``letter``, ``first`` and ``now`` (cells), ``per_move`` (the most common change per move), ``changed_on``
    (how many moves changed it) and ``odd`` (up to 10 moves [i, change] that changed it differently)."""
    if rows is None:
        from observation import moves
        rows = moves()
    rows = [r for r in rows if r.get("before") and r.get("after") and r.get("move") != "reset"]
    if not rows:
        return []
    count = lambda b: Counter("".join(b))
    deltas: dict[str, list[tuple[Any, int]]] = {}
    for r in rows:
        cb, ca = count(r["before"]), count(r["after"])
        for x in set(cb) | set(ca):
            if ca[x] != cb[x]:
                deltas.setdefault(x, []).append((r.get("i"), ca[x] - cb[x]))
    first, last = count(rows[0]["before"]), count(rows[-1]["after"])
    bg = first.most_common(1)[0][0] if first else None
    out = []
    for x, ds in deltas.items():
        if x == bg or len(ds) < max(1, min_share * len(rows)):
            continue
        per = Counter(d for _, d in ds).most_common(1)[0][0]
        out.append({"letter": x, "first": first[x], "now": last[x], "per_move": per, "changed_on": len(ds),
                    "moves": len(rows), "odd": [[i, d] for i, d in ds if d != per][:10]})
    out.sort(key=lambda d: -d["changed_on"])
    return out


# --- other movers and undoing --------------------------------------------------------------------------------------

def _level_moves(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if rows is None:
        from observation import moves
        rows = moves()
    return [r for r in rows if r.get("before") and r.get("after")]


def _boxes(board: list[str], letter: str) -> list[list[int]]:
    """The boxes of the 4-connected areas of ``letter`` on the board, top to bottom."""
    cells = {(r, c) for r, row in enumerate(board) for c, x in enumerate(row) if x == letter}
    out = []
    while cells:
        stack = [cells.pop()]
        rs, cs = [], []
        while stack:
            r, c = stack.pop()
            rs.append(r)
            cs.append(c)
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n in cells:
                    cells.remove(n)
                    stack.append(n)
        out.append([min(rs), min(cs), max(rs), max(cs)])
    return sorted(out)


def track(letter: str, rows: list[dict[str, Any]] | None = None, last: int = 30) -> list[dict[str, Any]]:
    """Where the areas of ``letter`` were after each of the last ``last`` moves of this level, next to the move sent:
    the way to see whether something chases you, runs away, copies your moves (or mirrors them) or moves only on some
    moves. ``rows``: the moves to read (default ``moves()``). Each entry: ``i``, ``move``, ``boxes`` (the areas after
    the move) and ``by`` [rows, columns] the area nearest to where one was before moved (None when none moved or the
    number of areas changed)."""
    letter = _cell(letter)
    out = []
    for r in _level_moves(rows)[-max(1, int(last)):]:
        if r.get("move") == "reset":
            continue
        before, after = _boxes(r["before"], letter), _boxes(r["after"], letter)
        by = None
        if before and len(before) == len(after):
            for a in before:   # the first area that is not where it was: its nearest area after the move
                if a in after:
                    continue
                b = min(after, key=lambda x: abs(x[0] - a[0]) + abs(x[1] - a[1]))
                by = [b[0] - a[0], b[1] - a[1]]
                break
        out.append({"i": r.get("i"), "move": r.get("move"), "boxes": after[:10], "by": by})
    return out


def _masked_key(rows: list[dict[str, Any]]) -> Callable[[list[str]], str]:
    """A board as one string with the bars and counters (``counters()``) turned into the background, so a bar that
    shrinks every move does not make every board new. A letter is masked only inside the box where it changed, and
    only when that box is small: on a scrolling view every letter's count changes, and the scenery must stay. And
    only a bar: a letter whose count changes on most moves, nearly always the same way (it shrinks or it fills),
    not one that goes up and down as things are switched on and off."""
    bars = set()
    for c in counters(rows, min_share=0.6):
        ups = downs = 0
        for r in rows:
            d = "".join(r["after"]).count(c["letter"]) - "".join(r["before"]).count(c["letter"])
            ups, downs = ups + (d > 0), downs + (d < 0)
        if max(ups, downs) >= 0.9 * (ups + downs):
            bars.add(c["letter"])
    if not rows or not bars:
        return lambda b: "\n".join(b)
    a0 = _arr(rows[0]["before"])
    bg, (h, w) = _background(a0), a0.shape
    hit = {x: np.zeros((h, w), bool) for x in bars}
    for r in rows:
        a, b = _arr(r["before"]), _arr(r["after"])
        if a.shape == b.shape == (h, w):
            for x in bars:
                hit[x] |= (a == x) != (b == x)
    boxes = {}
    for x, m in hit.items():
        if m.any():
            rs, cs = np.nonzero(m)
            r0, c0, r1, c1 = int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())
            if (r1 - r0 + 1) * (c1 - c0 + 1) <= h * w / 8:
                boxes[x] = (r0, c0, r1, c1)

    def key(b: list[str]) -> str:
        out = list(b)
        for x, (r0, c0, r1, c1) in boxes.items():
            for r in range(r0, min(r1 + 1, len(out))):
                s = out[r]
                out[r] = s[:c0] + s[c0:c1 + 1].replace(x, bg) + s[c1 + 1:]
        return "\n".join(out)
    return key


def _is_over(r: dict[str, Any]) -> bool:
    return str(r.get("state") or "").upper().startswith("GAME_OVER")


def _tries(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """The moves split into tries: a reset or a lost level starts a new one."""
    tries: list[list[dict[str, Any]]] = [[]]
    for r in rows:
        if r.get("move") == "reset":
            tries.append([])
            continue
        tries[-1].append(r)
        if _is_over(r):
            tries.append([])
    return [t for t in tries if t]


def undo_pairs(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Which moves of this level brought the board back to an earlier one, and which changes never came back: the way
    to see which moves are safe to try (they can be undone) and which are one-way. The letters of the bars and counters
    (``counters()``) are left out, so a bar that shrinks every move does not make every board new. ``rows``: the moves
    to read (default ``moves()``). Returns ``pairs`` [{"first", "then", "times"}] (a move right after another that
    put the board back as it was before the first), ``returns`` (up to 10 longer runs of moves [i, ...] that came back
    to an earlier board) and ``one_way`` [{"move", "times"}] (moves that changed the board, where the board before
    them was never seen again on that try)."""
    rows = _level_moves(rows)
    key = _masked_key(rows)
    tries = _tries(rows)   # a new try's boards are not compared with the old ones
    pairs: Counter = Counter()
    returns: list[list[Any]] = []
    sent: Counter = Counter()
    back: Counter = Counter()
    for t in tries:
        befores = [key(r["before"]) for r in t]
        afters = [key(r["after"]) for r in t]
        first_seen: dict[str, int] = {}
        for j, b in enumerate(befores):
            first_seen.setdefault(b, j)
        for j, r in enumerate(t):
            if befores[j] == afters[j]:
                continue   # nothing changed: neither undone nor one-way
            sent[r.get("move")] += 1
            if afters[j] in first_seen and first_seen[afters[j]] <= j or any(a == befores[j] for a in afters[j + 1:]):
                back[r.get("move")] += 1   # it went back to an earlier board, or its board came back later
            if j > 0 and afters[j] == befores[j - 1] and befores[j - 1] != afters[j - 1]:
                pairs[(t[j - 1].get("move"), r.get("move"))] += 1
            k = first_seen.get(afters[j])
            if k is not None and k < j - 1 and len(returns) < 10:
                returns.append([x.get("i") for x in t[k:j + 1]])
    return {"pairs": [{"first": a, "then": b, "times": n} for (a, b), n in pairs.most_common(20)],
            "returns": returns,
            "one_way": [{"move": m, "times": n} for m, n in sent.most_common() if back[m] == 0]}


# --- checking a model and reading the record of this level ---------------------------------------------------------

def _plain_rows(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [r for r in _level_moves(rows) if r.get("move") != "reset"]


def _areas(board: list[str], skip: Iterable[str] = ()) -> list[tuple[str, set[tuple[int, int]]]]:
    """The 4-connected one-letter areas of the board, the letters in ``skip`` left out, top to bottom."""
    skip = set(skip)
    h = len(board)
    seen: set[tuple[int, int]] = set()
    out = []
    for r0 in range(h):
        for c0, x in enumerate(board[r0]):
            if x in skip or (r0, c0) in seen:
                continue
            seen.add((r0, c0))
            stack, cells = [(r0, c0)], set()
            while stack:
                r, c = stack.pop()
                cells.add((r, c))
                for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= n[0] < h and 0 <= n[1] < len(board[n[0]]) and n not in seen and board[n[0]][n[1]] == x:
                        seen.add(n)
                        stack.append(n)
            out.append((x, cells))
    return out


def _click_cell(m: Any) -> tuple[int, int] | None:
    parts = str(m).split()
    if len(parts) == 3 and parts[0] == "6":
        try:
            return int(parts[1]), int(parts[2])
        except ValueError:
            return None
    return None


def _middle(cells: set[tuple[int, int]]) -> tuple[int, int]:
    mr = sum(r for r, _ in cells) / len(cells)
    mc = sum(c for _, c in cells) / len(cells)
    return min(cells, key=lambda p: ((p[0] - mr) ** 2 + (p[1] - mc) ** 2, p))


def _cell_box(cells: set[tuple[int, int]]) -> list[int]:
    return [min(r for r, _ in cells), min(c for _, c in cells), max(r for r, _ in cells), max(c for _, c in cells)]


def verify(step: Callable[[Any, Any], Any], rows: list[dict[str, Any]] | None = None,
           read: Callable[[Any], Any] | None = None, move: Callable[[Any], Any] | None = None,
           limit: int = 10) -> dict[str, Any]:
    """Check your ``step()`` model against every move of this level, the lost tries included: for each recorded move
    it runs ``step(board before, move)`` and compares what comes back with the board after. ``step`` gets the board
    as ``obs.board`` holds it (a list of strings) and the move as written in a ``play()`` path ("1", "6 12 30"); give
    ``read(board)`` to turn a board into your own state (then it must return ``read(board after)``) and ``move(m)``
    to turn the move into yours. The bars and counters (``counters()``) are left out of the board compares.
    Returns ``moves`` (how many were checked), ``right``, ``by_move`` {move: [right, checked]}, ``wrong`` (up to
    ``limit``: ``i``, ``move`` and ``cells`` [r, c, yours, real] up to 10, or ``yours`` and ``real`` for your own
    state) and ``errors`` (up to ``limit`` moves where it raised, with the error)."""
    rows = _plain_rows(rows)
    key = _masked_key(rows)

    def board(b: Any) -> np.ndarray:
        a = _arr(b)
        return _arr(key(["".join(x) for x in a]).split("\n")) if a.size else a

    right, by_move, wrong, errors = 0, {}, [], []
    for r in rows:
        m = r.get("move")
        tally = by_move.setdefault(str(m), [0, 0])
        tally[1] += 1
        try:
            got = step(read(list(r["before"])) if read else list(r["before"]), move(m) if move else m)
            want = read(list(r["after"])) if read else None
        except Exception as exc:  # noqa: BLE001 - reported, the check goes on
            if len(errors) < limit:
                errors.append({"i": r.get("i"), "move": m, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if read:
            good = freeze(got) == freeze(want)
            miss = {"yours": got, "real": want}
        elif got is None:
            good, miss = False, {"cells": "step() returned None"}
        else:
            if isinstance(got, list) and all(isinstance(x, str) for x in got):   # rows of letters: compare as text
                good = key(got) == key(list(r["after"]))
            else:
                ga, wa = board(got), board(r["after"])
                good = ga.shape == wa.shape and bool((ga == wa).all())
            miss = {}
            if not good and len(wrong) < limit:
                ga, wa = board(got), board(r["after"])
                miss = {"cells": [[int(y), int(x), str(ga[y, x]), str(wa[y, x])]
                                  for y, x in np.argwhere(ga != wa)[:10]]
                        if ga.shape == wa.shape else f"your board is {ga.shape[0]}x{ga.shape[1]}, "
                                                     f"the real one {wa.shape[0]}x{wa.shape[1]}"}
        if good:
            right += 1
            tally[0] += 1
        elif len(wrong) < limit:
            wrong.append({"i": r.get("i"), "move": m, **miss})
    return {"moves": len(rows), "right": right, "by_move": by_move, "wrong": wrong, "errors": errors}


def atlas(rows: list[dict[str, Any]] | None = None, moves: Iterable[Any] | None = None, to: int | None = None,
          limit: int = 10) -> dict[str, Any]:
    """What this level has already tried, as a map: every different board seen on it (bars and counters left out)
    is a place, numbered in the order first seen (0 = the level's first board), and every move sent from a place
    is a road to where it led. ``moves``: the moves you can send (default: every move sent in this game, and the
    spots clicked on this level). Returns ``places`` (how many), ``now`` (the place of the board now), ``tried_here`` {move: place it led to,
    "same" when nothing changed, "GAME_OVER"}, ``untried_here`` (moves never sent from the board now), ``open`` (up to
    ``limit`` places nearest the start with moves never sent from them: ``place``, ``untried``, ``path`` from the
    level's first board), ``never_clicked`` (areas on the board now that no click on this level touched: ``letter``,
    ``box`` and ``click`` "6 r c" at its middle) and ``losing`` (up to ``limit`` [place, move] that lost the level).
    With ``to``: ``path`` (the moves from the level's first board to that place, None when no road leads there) and
    ``board`` (that place's board)."""
    given = rows is not None
    rows = _plain_rows(rows)
    if moves is None:
        if given:
            sent = [r.get("move") for r in rows]
        else:
            from observation import moves as all_moves
            sent = [r.get("move") for r in all_moves(all_levels=True)]
        moves = [m for m in dict.fromkeys(str(x) for x in sent) if m != "reset" and _click_cell(m) is None]
        # and every spot clicked on this level, so "which clicks were never tried from here" is answered too
        moves += list(dict.fromkeys(str(r.get("move")) for r in rows if _click_cell(r.get("move"))))[:60]
    moves = [str(m) for m in moves]
    if not rows:
        return {"places": 0, "now": None, "tried_here": {}, "untried_here": moves, "open": [], "never_clicked": [],
                "losing": []}
    key = _masked_key(rows)
    ids: dict[str, int] = {}
    boards: list[list[str]] = []

    def place(b: list[str]) -> int:
        k = key(b)
        if k not in ids:
            ids[k] = len(boards)
            boards.append(list(b))
        return ids[k]

    roads: dict[int, dict[str, Any]] = {}
    losing: list[list[Any]] = []
    clicks: set[tuple[int, int]] = set()
    now = place(rows[0]["before"])
    for r in rows:
        a = place(r["before"])
        m = str(r.get("move"))
        if _click_cell(m):
            clicks.add(_click_cell(m))
        if _is_over(r):
            roads.setdefault(a, {})[m] = "GAME_OVER"
            if len(losing) < limit and [a, m] not in losing:
                losing.append([a, m])
            now = 0
            continue
        b = place(r["after"])
        roads.setdefault(a, {}).setdefault(m, "same" if a == b else b)
        now = b
    dist: dict[int, int] = {0: 0}
    back: dict[int, tuple[int, str]] = {}
    q = deque([0])
    while q:
        a = q.popleft()
        for m, b in roads.get(a, {}).items():
            if isinstance(b, int) and b not in dist:
                dist[b] = dist[a] + 1
                back[b] = (a, m)
                q.append(b)

    def path_to(p: int) -> list[str] | None:
        if p not in dist:
            return None
        out = []
        while p != 0:
            p, m = back[p]
            out.append(m)
        return out[::-1]

    here = roads.get(now, {})
    opened = sorted((p for p in range(len(boards)) if p in dist and any(m not in roads.get(p, {}) for m in moves)),
                    key=lambda p: (dist[p], p))[:limit]
    bg = _background(_arr(boards[now]))
    never = [(x, cells) for x, cells in _areas(boards[now], skip={bg}) if not cells & clicks]
    out = {"places": len(boards), "now": now, "tried_here": here, "untried_here": [m for m in moves if m not in here],
           "open": [{"place": p, "untried": [m for m in moves if m not in roads.get(p, {})], "path": path_to(p)}
                    for p in opened],
           "never_clicked": [{"letter": x, "box": _cell_box(cells), "click": "6 %d %d" % _middle(cells)}
                             for x, cells in sorted(never, key=lambda t: -len(t[1]))[:max(limit, 20)]],
           "losing": losing}
    if to is not None:
        out["path"] = path_to(int(to))
        out["board"] = boards[int(to)] if 0 <= int(to) < len(boards) else None
    return out


_CAMERA_SEEN: dict[tuple[Any, ...], dict[str, Any]] = {}


def _scroll_guesses(a: np.ndarray, b: np.ndarray, m: int, top: int = 3) -> list[tuple[int, int]]:
    """The likeliest scrolls, cheaply: the row shifts that best line up the colour counts of each row, crossed with
    the column shifts that best line up the counts of each column. ``camera`` then checks only these."""
    a, b = a.view(np.uint32), b.view(np.uint32)   # one-letter strings as their code points: fast compares
    letters = np.unique(np.concatenate([a.ravel(), b.ravel()]))
    ha = np.stack([(a == x).sum(axis=1) for x in letters], 1)
    hb = np.stack([(b == x).sum(axis=1) for x in letters], 1)
    va = np.stack([(a == x).sum(axis=0) for x in letters], 1)
    vb = np.stack([(b == x).sum(axis=0) for x in letters], 1)

    def best(pa: np.ndarray, pb: np.ndarray) -> list[int]:
        n = len(pa)
        cost = {}
        for d in range(-m, m + 1):   # after[i + d] lines up with before[i]
            x, y = pa[max(-d, 0):n - max(d, 0)], pb[max(d, 0):n + min(d, 0)]
            cost[d] = float(np.abs(x - y).sum()) / max(1, len(x))
        return sorted(cost, key=lambda d: (cost[d], abs(d)))[:top]

    return [(dr, dc) for dr in best(ha, hb) for dc in best(va, vb)]


def world(you: str | None = None, rows: list[dict[str, Any]] | None = None, dark: str | None = None,
          max_shift: int = 16) -> dict[str, Any]:
    """The whole level as one map, for a game whose view scrolls or shows only the part near you: each board of this
    level is placed where it belongs, using the scroll between moves (``camera``). Cells that stay put while the view
    scrolls (a status bar) are left off the map from the second scroll on. ``dark``: the letter of the cells you
    cannot see (the dark around a lit patch); they count as unseen, so the map keeps what was lit before. A board
    that does not follow from the one before by a scroll (a teleport, a new room) starts a new piece of map, unless
    that same board was seen before. ``you``: your letter, to put you on the map.
    Returns ``pieces`` (how many separate maps), ``piece`` (the one the view is on now), ``map`` (that piece as rows
    of letters, "?" where never seen, each cell as last seen), ``origin`` [row, column] (where the top-left of the
    view is on that map now), ``scrolls`` (moves that scrolled the view), ``you`` (your box on the map now) and
    ``visited`` (how many different places on this piece your box has had)."""
    rows = _plain_rows(rows)
    if not rows:
        return {"pieces": 0, "piece": None, "map": [], "origin": None, "scrolls": 0, "you": None, "visited": 0}
    key = _masked_key(rows)
    letter = _cell(you) if you is not None else None
    unseen = _cell(dark) if dark is not None else None
    pieces: list[dict[str, Any]] = []
    screens: dict[str, tuple[int, tuple[int, int]]] = {}
    fixed = None
    cur: dict[str, Any] = {}
    scrolls = 0
    visits: dict[int, set[tuple[int, int]]] = {}

    def paint(b: list[str]) -> None:
        a = _arr(b)
        h, w = a.shape
        pc = pieces[cur["piece"]]
        R, C = cur["origin"]
        g, (R0, C0) = pc["grid"], pc["off"]
        top, left = min(R0, R), min(C0, C)
        bottom, right = max(R0 + g.shape[0], R + h), max(C0 + g.shape[1], C + w)
        if (top, left, bottom, right) != (R0, C0, R0 + g.shape[0], C0 + g.shape[1]):
            big = np.full((bottom - top, right - left), "?", dtype="<U1")
            big[R0 - top:R0 - top + g.shape[0], C0 - left:C0 - left + g.shape[1]] = g
            pc["grid"], pc["off"] = big, (top, left)
            g, (R0, C0) = big, (top, left)
        view = g[R - R0:R - R0 + h, C - C0:C - C0 + w]
        keep = fixed if fixed is not None and fixed.shape == a.shape else np.zeros(a.shape, bool)
        if unseen is not None:
            keep = keep | (a == unseen)
        view[...] = np.where(keep, view, a)
        screens.setdefault(key(b), (cur["piece"], cur["origin"]))
        if letter is not None:
            cells = np.argwhere(a == letter)
            if len(cells):
                visits.setdefault(cur["piece"], set()).add((int(cells[:, 0].min()) + R, int(cells[:, 1].min()) + C))

    def jump(b: list[str]) -> None:
        hit = screens.get(key(b))
        if hit:
            cur["piece"], cur["origin"] = hit
        else:
            a = _arr(b)
            pieces.append({"grid": np.full(a.shape, "?", dtype="<U1"), "off": (0, 0)})
            cur["piece"], cur["origin"] = len(pieces) - 1, (0, 0)
        paint(b)

    prev = None
    for r in rows:
        before, after = list(r["before"]), list(r["after"])
        if prev is None or key(before) != key(prev):
            jump(before)
        if _is_over(r):
            prev = None
            continue
        if before != after:
            a, b = _arr(before), _arr(after)
            ck = (hash("\n".join(before)), hash("\n".join(after)), int(max_shift), unseen)
            if ck not in _CAMERA_SEEN:
                if len(_CAMERA_SEEN) > 50000:
                    _CAMERA_SEEN.clear()
                same = a.shape == b.shape
                diff = (a != b) if same else None
                wide = False
                if diff is not None and diff.mean() > 0.05:   # a scroll changes cells across most of the view
                    rs, cs = np.nonzero(diff.any(axis=1))[0], np.nonzero(diff.any(axis=0))[0]
                    wide = rs[-1] - rs[0] + 1 >= 0.75 * a.shape[0] or cs[-1] - cs[0] + 1 >= 0.75 * a.shape[1]
                m = max(0, min(int(max_shift), a.shape[0] - 1, a.shape[1] - 1))
                guesses = _scroll_guesses(a, b, m) if wide else [(0, 0)]
                # a few cells changed, or nothing points to a scroll: something moved on a still view
                cam = camera(before, after, shifts=guesses) if guesses[0] != (0, 0) else {"by": [0, 0]}
                if not same:
                    cam = {"by": [0, 0], "match": 0.0, "still": 0.0}
                elif cam["by"] == [0, 0]:   # how much of the view stayed the same
                    if unseen is not None:   # in the dark only the cells lit on both boards can disagree
                        care = (a != unseen) & (b != unseen)
                    else:
                        bg = _background(a)
                        care = (a != bg) | (b != bg)
                    n = int(care.sum())
                    still = round(float(((a == b) & care).sum()) / n, 3) if n else 1.0
                    cam = {"by": [0, 0], "match": still, "still": still}
                _CAMERA_SEEN[ck] = cam
            cam = _CAMERA_SEEN[ck]
            dr, dc = cam["by"]
            if (dr, dc) != (0, 0):
                scrolls += 1
                moved = _arr(shift(before, dr, dc))
                still = (a == b) & (moved != b) & (b != _background(b))
                # cells that stayed put on every scroll so far (a status bar, you at the middle of the view)
                fixed = still if fixed is None or fixed.shape != still.shape else fixed & still
                cur["origin"] = (cur["origin"][0] - dr, cur["origin"][1] - dc)
                paint(after)
            elif cam["still"] < 0.5:
                jump(after)
            else:
                paint(after)
        prev = after
    if "piece" not in cur:
        jump(list(rows[-1]["after"]))
    pc = pieces[cur["piece"]]
    R, C = cur["origin"]
    R0, C0 = pc["off"]
    you_box = None
    if letter is not None and prev is not None:
        cells = np.argwhere(_arr(prev) == letter)
        if len(cells):
            you_box = [int(cells[:, 0].min()) + R - R0, int(cells[:, 1].min()) + C - C0,
                       int(cells[:, 0].max()) + R - R0, int(cells[:, 1].max()) + C - C0]
    return {"pieces": len(pieces), "piece": cur["piece"], "map": ["".join(x) for x in pc["grid"]],
            "origin": [R - R0, C - C0], "scrolls": scrolls, "you": you_box,
            "visited": len(visits.get(cur["piece"], ()))}


def wins(rows: list[dict[str, Any]] | None = None, limit: int = 5) -> list[dict[str, Any]]:
    """How the earlier levels of this game were won, to look for the same on this one. For each won level:
    ``level`` (1 = the first), ``moves`` (moves sent on it, lost tries included), ``move`` (the winning move),
    ``before`` (the board just before it), ``last`` (the ``limit`` moves before it: ``i``, ``move``, ``change``) and,
    for a winning click, ``clicked`` (the letter and box of the area clicked). The board after the winning move is
    already the next level, so line ``before`` up with this level's boards (``relate``, ``find``) to see what has to
    be true here when you win. ``rows``: the whole game's moves (default ``moves(all_levels=True)``)."""
    if rows is None:
        from observation import moves
        rows = moves(all_levels=True)
    rows = [r for r in rows if r.get("before")]
    out, start = [], 0
    for j, r in enumerate(rows):
        nxt = rows[j + 1] if j + 1 < len(rows) else None
        won = str(r.get("state") or "").upper() == "WIN" or (
            nxt is not None and (nxt.get("level") or 0) > (r.get("level") or 0))
        if not won:
            continue
        mine = rows[start:j + 1]
        win = {"level": (r.get("level") or 0) + 1, "moves": sum(1 for x in mine if x.get("move") != "reset"),
               "move": r.get("move"), "before": list(r["before"]),
               "last": [{"i": x.get("i"), "move": x.get("move"), "change": x.get("change")}
                        for x in mine[-1 - max(0, int(limit)):-1]]}
        cell = _click_cell(r.get("move"))
        if cell and 0 <= cell[0] < len(r["before"]) and 0 <= cell[1] < len(r["before"][cell[0]]):
            x = r["before"][cell[0]][cell[1]]
            area = next((cells for y, cells in _areas(r["before"]) if cell in cells), {cell})
            win["clicked"] = {"letter": x, "box": _cell_box(area)}
        out.append(win)
        start = j + 1
    return out


def cause(what: Any, you: str | None = None, rows: list[dict[str, Any]] | None = None,
          limit: int = 10) -> list[dict[str, Any]]:
    """What makes something happen: this level's moves are split into those that changed ``what`` and those that did
    not, and simple conditions are ranked by how well they tell the two apart. ``what``: a box [r0, c0, r1, c1] (a
    cell in it changed) or a letter (where its cells are changed). The conditions: the move sent; the letters in the
    box before the move (when ``what`` is a box); and with ``you`` (your letter): each letter next to you before the
    move, each letter on the cells you moved onto (as they were before the move), and whether you moved. Returns up to
    ``limit`` dicts, best first: ``if`` (the condition), ``changed`` [moves with it that changed it, all moves that
    changed it], ``not_changed`` [moves with it that did not, all moves that did not] and ``score`` (the share of
    changes that had it minus the share of non-changes that had it; 1.0 = it explains every change)."""
    rows = _plain_rows(rows)
    box = [int(x) for x in what] if isinstance(what, (list, tuple)) and len(what) == 4 else None
    letter = None if box else _cell(what)
    me = _cell(you) if you is not None else None
    pos: Counter = Counter()
    neg: Counter = Counter()
    n_pos = n_neg = 0
    for r in rows:
        a, b = _arr(r["before"]), _arr(r["after"])
        if a.shape != b.shape:
            continue
        if box:
            ra, rb = a[box[0]:box[2] + 1, box[1]:box[3] + 1], b[box[0]:box[2] + 1, box[1]:box[3] + 1]
            hit = bool((ra != rb).any())
        else:
            hit = bool(((a == letter) != (b == letter)).any())
        feats = {f"move {r.get('move')}"}
        if box:
            feats |= {f"box had {x}" for x in np.unique(ra)}
        if me is not None:
            mine_a, mine_b = a == me, b == me
            if mine_a.any():
                ring = np.zeros_like(mine_a)
                ring[1:] |= mine_a[:-1]
                ring[:-1] |= mine_a[1:]
                ring[:, 1:] |= mine_a[:, :-1]
                ring[:, :-1] |= mine_a[:, 1:]
                feats |= {f"next to {x}" for x in np.unique(a[ring & ~mine_a])}
            onto = mine_b & ~mine_a
            feats |= {f"moved onto {x}" for x in np.unique(a[onto]) if x != me}
            feats.add("you moved" if (mine_a != mine_b).any() else "you did not move")
        if hit:
            n_pos += 1
            pos.update(feats)
        else:
            n_neg += 1
            neg.update(feats)
    if not n_pos:
        return []
    ranked = []
    for f in set(pos) | set(neg):
        score = pos[f] / n_pos - (neg[f] / n_neg if n_neg else 0.0)
        ranked.append({"if": f, "changed": [pos[f], n_pos], "not_changed": [neg[f], n_neg], "score": round(score, 3)})
    ranked.sort(key=lambda d: (-d["score"], -d["changed"][0], d["if"]))
    return ranked[:max(1, int(limit))]


# --- search over a model -------------------------------------------------------------------------------------------

_PLAIN = frozenset((str, int, float, bool, type(None)))


def freeze(s: Any) -> Any:
    """A hashable value that is equal for equal states and different for different ones: numpy arrays by their bytes
    (their repr shortens big arrays, so two boards could look the same), lists and tuples as tuples, dicts and sets
    sorted, an object by its fields. ``search`` keys its states with it."""
    if isinstance(s, np.ndarray):
        return ("ndarray", s.shape, s.dtype.str, s.tobytes())
    if isinstance(s, (list, tuple)):
        if all(map(_PLAIN.__contains__, map(type, s))):   # a row of letters or numbers, or a board as strings
            return s if type(s) is tuple else tuple(s)
        return tuple(freeze(x) for x in s)
    if isinstance(s, dict):
        return ("dict", tuple(sorted(((freeze(k), freeze(v)) for k, v in s.items()), key=repr)))
    if isinstance(s, (set, frozenset)):
        return ("set", tuple(sorted((freeze(x) for x in s), key=repr)))
    if isinstance(s, (str, bytes, int, float, bool, type(None), np.generic)):
        return s
    if hasattr(s, "__dict__") and (type(s).__hash__ in (None, object.__hash__)):
        return (type(s).__name__, freeze(vars(s)))   # a class without its own hash: equal fields, same state
    try:
        hash(s)
        return s
    except TypeError:
        return repr(s)


class _StepChangedInput(Exception):
    pass


def search(start: Any, step: Callable[[Any, Any], Any], is_goal: Callable[[Any], bool], moves: Iterable[Any],
           key: Callable[[Any], Any] | None = None, max_states: int = 200000, max_s: float = 20.0) -> dict[str, Any]:
    """Breadth-first search over your own model: ``step(state, move)`` returns the next state (or None when the
    move is not possible), ``is_goal(state)`` says when it is won, ``moves`` are the moves to try from every state.
    Any state works (a board, numpy array, tuple, dict, set, object); two states are the same when their contents
    are equal, or when ``key(state)`` gives the same value. A ``step()`` that changes the state it was given is
    noticed: the search starts again and gives ``step()`` a copy of the state every time (``copied`` True).
    Returns ``path`` (the shortest list of moves, None if not found), ``states`` (how many were seen), ``reason``
    ("found", "exhausted", "max_states" or "max_s"), ``end`` (the goal state) and ``copied``."""
    moves = list(moves)
    k = key if key is not None else freeze
    t0 = time.time()

    def call(fn: Callable[..., Any], *args: Any) -> Any:
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 - the error goes on, with the move that raised it
            if hasattr(exc, "add_note"):
                exc.add_note(f"search: raised by step(state, {args[1]!r})" if fn is step else
                             "search: raised by is_goal(state)")
            raise

    def run(copied: bool) -> dict[str, Any]:
        start = copy.deepcopy(pristine)   # the caller's state is never handed to step()
        seen = {k(start): None}
        if call(is_goal, start):
            return {"path": [], "states": 1, "reason": "found", "end": start, "copied": copied}
        q = deque([(start, [], k(start))])
        while q:
            if len(seen) >= max_states:
                return {"path": None, "states": len(seen), "reason": "max_states", "end": None, "copied": copied}
            if time.time() - t0 > max_s:
                return {"path": None, "states": len(seen), "reason": "max_s", "end": None, "copied": copied}
            state, path, sk = q.popleft()
            for m in moves:
                nxt = call(step, copy.deepcopy(state) if copied else state, m)
                if nxt is None:
                    continue
                kk = k(nxt)
                if kk in seen:
                    continue
                seen[kk] = None
                if call(is_goal, nxt):
                    if not copied and k(state) != sk:
                        raise _StepChangedInput
                    return {"path": path + [m], "states": len(seen), "reason": "found", "end": nxt, "copied": copied}
                q.append((nxt, path + [m], kk))
            if not copied and k(state) != sk:   # step() wrote into the state it was given: start again with copies
                raise _StepChangedInput
        return {"path": None, "states": len(seen), "reason": "exhausted", "end": None, "copied": copied}

    pristine = copy.deepcopy(start)   # the start as given, for a second try if step() changes its input
    try:
        return run(False)
    except _StepChangedInput:
        print("search: your step() changes the state it is given, so every state was copied before step() "
              "(slower). Build a new state in step() instead.")
        return run(True)
