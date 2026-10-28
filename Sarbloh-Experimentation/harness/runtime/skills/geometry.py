"""Geometry on boards, pre-imported by the kernel: turn, mirror, shift, crop, paste and scale a board or a piece of
one, find what transform turns one shape into another, find a shape on the board, check symmetry, spot a scrolled
view, walk lines and rays, find paths, track the numbers on screen, and search the agent's own ``step()`` model.

A board is what ``obs.board`` holds (a list of strings, one letter per cell) or a list of lists; a function that
returns a board returns the same kind it was given. Positions are (row, column), boxes are [r0, c0, r1, c1]
inclusive, as in ``obs.objects``. An object dict from ``obs.objects`` (with its ``cells``) is accepted wherever a
piece is. Nothing here spends a move.

    rotate(b, 1); flip(b, "h"); shift(b, 0, 3); crop(b, 2, 2, 9, 9); paste(b, piece, r, c); scale(b, 2)
    relate(a, b)          # which turn / mirror / scale / recolour turns a into b, and how well
    find(piece, board)    # where a piece is on the board, turned and mirrored copies too
    symmetry(b)           # mirror and turn symmetry, and the cells that break it
    camera(before, after) # did the whole view scroll, and by how much
    line(r0, c0, r1, c1); ray(b, r, c, dr, dc); reach(b, start, passable, goal)
    counters()            # the letters whose cell count changes move after move (bars, counters, lives)
    search(start, step, is_goal, moves)   # breadth-first search over your own step() model
"""

from __future__ import annotations

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
        return g.astype("<U1") if g.ndim == 2 else np.array([list(str(x)) for x in g], dtype="<U1")
    rows = list(g)
    if not rows:
        return np.zeros((0, 0), dtype="<U1")
    if len({len(r) for r in rows}) > 1:
        raise ValueError("every row of a board or piece must have the same length; use '.' for an empty cell")
    if isinstance(rows[0], str):
        return np.array([list(r) for r in rows], dtype="<U1")
    return np.array([[_cell(v) for v in r] for r in rows], dtype="<U1")


def _cell(v: Any) -> str:
    s = str(v)
    if len(s) == 1:
        return s
    if isinstance(v, (int, np.integer)) and 10 <= int(v) <= 35:
        return chr(ord("a") + int(v) - 10)   # an int grid: 10.. as one character, so shapes stay one cell wide
    return s[:1]


def _out(a: np.ndarray, like: Any) -> Any:
    """``a`` back in the kind of ``like``: a list of strings for a list of strings or an object, else lists."""
    if isinstance(like, dict) or (isinstance(like, (list, tuple)) and like and isinstance(like[0], str)):
        return ["".join(r) for r in a.tolist()]
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


def find(piece: Any, board: Any, turns: bool = True, min_match: float = 1.0, limit: int = 20) -> list[dict[str, Any]]:
    """Where ``piece`` is on ``board``: every place where at least ``min_match`` of its cells agree (cells equal to "."
    match anything), also turned and mirrored copies when ``turns``. Returns dicts ``at`` [r, c] (top-left corner),
    ``bbox``, ``transform`` and ``match``, best first, at most ``limit``."""
    from numpy.lib.stride_tricks import sliding_window_view
    bd, p0 = _arr(board), _arr(piece)
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
    the (rows, columns) pairs in ``shifts``. Returns ``by`` [rows, columns] ([0, 0] when no shift explains the change
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
        return float(((pa == pb) & care).sum()) / n if n else 0.0

    still = score(0, 0)
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
    letter -> bool). ``goal``: a corner (r, c), a list of corners, or a function (r, c) -> bool. Returns ``path``
    (direction names, None when the goal cannot be reached, [] with no goal), ``corners`` along it, ``reachable``
    (how many corners can be reached) and ``dist`` (corner -> moves, for every reachable corner)."""
    a = _arr(g)
    ok = passable if callable(passable) else (lambda x, s={_cell(p) for p in passable}: x in s)
    h, w = a.shape
    bh, bw = int(size[0]), int(size[1])
    free = np.vectorize(lambda x: bool(ok(x)), otypes=[bool])(a) if a.size else np.zeros((0, 0), bool)

    def fits(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + bh <= h and c + bw <= w and bool(free[r:r + bh, c:c + bw].all())

    if goal is None:
        is_goal = lambda r, c: False
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


# --- search over a model -------------------------------------------------------------------------------------------

def search(start: Any, step: Callable[[Any, Any], Any], is_goal: Callable[[Any], bool], moves: Iterable[Any],
           key: Callable[[Any], Any] | None = None, max_states: int = 200000, max_s: float = 20.0) -> dict[str, Any]:
    """Breadth-first search over your own model: ``step(state, move)`` returns the next state (or None when the
    move is not possible), ``is_goal(state)`` says when it is won, ``moves`` are the moves to try from every state.
    ``key(state)`` must give a hashable value (default: the state itself, or its repr if it cannot be hashed).
    Returns ``path`` (the shortest list of moves, None if not found), ``states`` (how many were seen), ``reason``
    ("found", "exhausted", "max_states" or "max_s") and ``end`` (the goal state)."""
    moves = list(moves)

    def k(s: Any) -> Any:
        if key is not None:
            return key(s)
        try:
            hash(s)
            return s
        except TypeError:
            return repr(s)

    t0 = time.time()
    seen = {k(start): None}
    q = deque([(start, [])])
    if is_goal(start):
        return {"path": [], "states": 1, "reason": "found", "end": start}
    while q:
        if len(seen) >= max_states:
            return {"path": None, "states": len(seen), "reason": "max_states", "end": None}
        if time.time() - t0 > max_s:
            return {"path": None, "states": len(seen), "reason": "max_s", "end": None}
        state, path = q.popleft()
        for m in moves:
            nxt = step(state, m)
            if nxt is None:
                continue
            kk = k(nxt)
            if kk in seen:
                continue
            seen[kk] = None
            if is_goal(nxt):
                return {"path": path + [m], "states": len(seen), "reason": "found", "end": nxt}
            q.append((nxt, path + [m]))
    return {"path": None, "states": len(seen), "reason": "exhausted", "end": None}
