# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""The kernel's geometry tools (``geo``): the transforms keep the board's kind and do what their names say; relate names
the turn, mirror, scale and colour change between two pieces (and the shift between two objects); find finds turned
and near copies; symmetry reports the cells that break it; camera finds a scrolled view and ignores a still one; line,
ray and reach walk the board (reach with a block size and step, and None when walled off); counters finds the bar that
shrinks per move; search finds the shortest path over a model and stops at its caps; the live kernel has ``geo``."""
import tempfile
from pathlib import Path

import numpy as np

from harness.runtime import kernel as kernel_mod
from harness.runtime.skills import geometry as geo

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# --- transforms -----------------------------------------------------------------------------------------------------
b = ["ab", "cd"]
ok("rotate_cw", geo.rotate(b) == ["ca", "db"] and geo.rotate(b, -1) == ["bd", "ac"] and geo.rotate(b, 4) == b)
ok("flip", geo.flip(b, "h") == ["ba", "dc"] and geo.flip(b, "v") == ["cd", "ab"])
ok("transpose", geo.transpose(b) == ["ac", "bd"])
ok("lists_stay_lists", geo.rotate([[1, 2], [3, 4]]) == [[3, 1], [4, 2]])
board = ["....", ".R..", "....", "...."]
ok("shift_fills_background", geo.shift(board, 1, 2) == ["....", "....", "...R", "...."])
ok("crop_inclusive", geo.crop(board, 1, 1, 2, 2) == ["R.", ".."])
ok("paste_skips_dots", geo.paste(board, ["B.", "BB"], 2, 0) == ["....", ".R..", "B...", "BB.."])
ok("paste_cut_at_edge", geo.paste(board, ["BB"], 3, 3) == ["....", ".R..", "....", "...B"])
ok("scale_up_down", geo.scale(["ab"], 2) == ["aabb", "aabb"] and geo.scale(["aabb", "aabb"], 1 / 2) == ["ab"])
try:
    geo.rotate(["ab", "c"])
    ok("ragged_board_explained", False)
except ValueError as exc:
    ok("ragged_board_explained", "same length" in str(exc))
ok("recolor",geo.recolor(board, {"R": "G", ".": "K"}) == ["KKKK", "KGKK", "KKKK", "KKKK"])

# --- relate ---------------------------------------------------------------------------------------------------------
L = ["R.", "R.", "RR"]
best = geo.relate(L, geo.rotate(L, 1))[0]
ok("relate_rotation", best["transform"] == "rotate 90" and best["match"] == 1.0 and best["scale"] == 1, str(best))
best = geo.relate(L, geo.flip(L, "h"))[0]
ok("relate_mirror", best["transform"] == "mirror left-right" and best["match"] == 1.0, str(best))
best = geo.relate(L, geo.scale(L, 2))[0]
ok("relate_scale", best["scale"] == 2 and best["match"] == 1.0 and best["transform"] == "same", str(best))
best = geo.relate(L, geo.recolor(L, {"R": "B"}))[0]
ok("relate_recolour", best["colors"] == {"R": "B"} and best["match"] == 1.0, str(best))
ok("relate_no_recolour", geo.relate(L, geo.recolor(L, {"R": "B"}), recolour=False)[0]["match"] < 1.0)
o1 = {"letter": "R", "bbox": [2, 3, 4, 4], "cells": [(2, 3), (3, 3), (4, 3), (4, 4)]}
o2 = {"letter": "R", "bbox": [7, 10, 9, 11], "cells": [(7, 10), (8, 10), (9, 10), (9, 11)]}
best = geo.relate(o1, o2)[0]
ok("relate_objects_by", best["transform"] == "same" and best["by"] == [5, 7] and best["match"] == 1.0, str(best))
try:
    geo.relate({"letter": "R", "cells": None}, o2)
    ok("relate_big_object_explains", False)
except ValueError as exc:
    ok("relate_big_object_explains", "crop it from the board" in str(exc))

# --- find -----------------------------------------------------------------------------------------------------------
field = ["......", ".RR...", ".R....", "......", "...BB.", "....B."]
hits = geo.find(["RR", "R."], field)
ok("find_exact", hits[0]["at"] == [1, 1] and hits[0]["transform"] == "same" and hits[0]["match"] == 1.0, str(hits))
hits = geo.find(["BB", ".B"], field, turns=False)
ok("find_dot_is_any", any(h["at"] == [4, 3] for h in hits), str(hits))
hits = geo.find(["RR", ".R"], field)
ok("find_turned", any(h["at"] == [1, 1] and h["transform"] != "same" for h in hits), str(hits))
ok("find_near", geo.find(["RR", "RR"], field, turns=False, min_match=0.75)[0]["at"] == [1, 1])
ok("find_none", geo.find(["GG"], field) == [])

# --- symmetry -------------------------------------------------------------------------------------------------------
sym = geo.symmetry(["ABA", "CDC"])
ok("symmetry_lr", sym["mirror left-right"]["match"] == 1.0 and not sym["mirror left-right"]["breaks"], str(sym))
sym = geo.symmetry(["ABA", "CDE"])
ok("symmetry_breaks", sym["mirror left-right"]["breaks"] == [[1, 0, "C", "E"], [1, 2, "E", "C"]], str(sym))
ok("symmetry_square_only_turns", "rotate 90" in geo.symmetry(["AB", "BA"]) and "rotate 90" not in sym)

# --- camera ---------------------------------------------------------------------------------------------------------
scene = [("." * 20) for _ in range(20)]
scene = geo.paste(scene, ["RRR", "R.R"], 3, 4)
scene = geo.paste(scene, ["BBBB"], 12, 9)
scene = geo.paste(scene, ["G", "G"], 8, 15)
moved = geo.shift(scene, -2, 3)
cam = geo.camera(scene, moved)
ok("camera_finds_scroll", cam["by"] == [-2, 3] and cam["match"] == 1.0 and cam["still"] < 0.5, str(cam))
one = geo.paste(scene, ["...", "..."], 3, 4)
one = geo.paste(one, ["RRR", "R.R"], 3, 6)
cam = geo.camera(scene, one)
ok("camera_still_when_one_object_moves", cam["by"] == [0, 0], str(cam))

# --- lines and paths ------------------------------------------------------------------------------------------------
ok("line_diagonal", geo.line(0, 0, 3, 3) == [(0, 0), (1, 1), (2, 2), (3, 3)])
ok("line_ends", geo.line(2, 5, 2, 2) == [(2, 5), (2, 4), (2, 3), (2, 2)])
r = geo.ray(field, 1, 0, 0, 1)
ok("ray_hits", r["cells"] == [] and r["hit"] == [1, 1, "R"], str(r))
r = geo.ray(field, 0, 0, 0, 1)
ok("ray_leaves", len(r["cells"]) == 5 and r["hit"] is None, str(r))
maze = ["S.#....",
        ".##.##.",
        "...#...",
        "##...#G"]
p = geo.reach(maze, (0, 0), ".SG", (3, 6))
ok("reach_path", p["path"] == ["down", "down", "right", "right", "down", "right", "right", "up", "right", "right", "down"]
   and p["corners"][-1] == (3, 6) and p["dist"][(3, 6)] == 11, str(p["path"]))
p = geo.reach(maze, (0, 0), ".S", (3, 6))
ok("reach_none", p["path"] is None and p["reachable"] > 1)
ok("reach_no_goal", geo.reach(maze, (0, 0), ".SG")["path"] == [])
big = ["......", "......", "..##..", "......", "......", "......"]
p = geo.reach(big, (0, 0), ".", (4, 4), step=2, size=(2, 2))
ok("reach_block_step", p["path"] is not None and len(p["path"]) == 4 and (2, 2) not in p["dist"], str(p["path"]))

# --- counters -------------------------------------------------------------------------------------------------------
rows = []
for i in range(6):
    bar = "B" * (10 - i) + "." * i
    after = "B" * (9 - i) + "." * (i + 1)
    rows.append({"i": i, "move": "1", "before": [bar, "R.........", "." * 10],
                 "after": [after, "R........." if i != 3 else ".R........", "." * 10]})
cs = geo.counters(rows)
ok("counters_bar", cs and cs[0]["letter"] == "B" and cs[0]["per_move"] == -1 and cs[0]["changed_on"] == 6
   and cs[0]["first"] == 10 and cs[0]["now"] == 4 and not cs[0]["odd"], str(cs))
ok("counters_skip_rare_and_background", all(c["letter"] not in "R." for c in cs))

# --- search ---------------------------------------------------------------------------------------------------------
step = lambda s, m: s + m if 0 <= s + m <= 20 else None
res = geo.search(0, step, lambda s: s == 7, [1, 3, -1])
ok("search_shortest", res["reason"] == "found" and sum(res["path"]) == 7 and len(res["path"]) == 3, str(res))
res = geo.search(0, step, lambda s: s == 99, [1])
ok("search_exhausted", res["path"] is None and res["reason"] == "exhausted" and res["states"] == 21)
res = geo.search(0, lambda s, m: s + m, lambda s: False, [1], max_states=50)
ok("search_capped", res["reason"] == "max_states" and res["states"] == 50)
res = geo.search([0, 0], lambda s, m: [s[0] + m, s[1]] if s[0] + m <= 3 else None, lambda s: s[0] == 3, [1])
ok("search_unhashable_state", res["path"] == [1, 1, 1], str(res))

# The kinds of state the agent's step() models used (10-27 run): numpy boards, dicts with sets, tuples with dicts.
D4 = {"1": (-1, 0), "2": (1, 0), "3": (0, -1), "4": (0, 1)}
big = np.zeros((64, 64), int)
big[30, 30] = 5


def np_step(b, m):
    r, c = map(int, np.argwhere(b == 5)[0])
    n = b.copy()
    n[r, c], n[r + D4[m][0], c + D4[m][1]] = 0, 5
    return n


res = geo.search(big, np_step, lambda b: b[33, 33] == 5, "1234")
ok("search_numpy_board", res["reason"] == "found" and len(res["path"]) == 6 and not res["copied"], str(res)[:200])
ok("freeze_numpy_differs", geo.freeze(big) != geo.freeze(np_step(big, "1")) and geo.freeze(big) == geo.freeze(big.copy()))
small = [[0] * 8 for _ in range(8)]
small[0][0] = 1


def mutating(b, m):   # no copy: writes into the state it was given
    r, c = next((i, j) for i in range(8) for j in range(8) if b[i][j] == 1)
    dr, dc = D4[m]
    if not (0 <= r + dr < 8 and 0 <= c + dc < 8):
        return None
    b[r][c], b[r + dr][c + dc] = 0, 1
    return b


res = geo.search(small, mutating, lambda b: b[3][3] == 1, "1234")
ok("search_step_changes_input", res["reason"] == "found" and len(res["path"]) == 6 and res["copied"]
   and small[0][0] == 1, str(res)[:200])
start = {"p": (0, 0), "keys": set()}


def key_step(s, m):
    p = (s["p"][0] + D4[m][0], s["p"][1] + D4[m][1])
    if not (0 <= p[0] < 6 and 0 <= p[1] < 6):
        return None
    return {"keys": set(s["keys"]) | ({p} if p in {(2, 3), (4, 1)} else set()), "p": p}   # keys in another order


res = geo.search(start, key_step, lambda s: len(s["keys"]) == 2 and s["p"] == (5, 5), "1234")
ok("search_dict_with_set", res["reason"] == "found" and len(res["path"]) == 14, str(res)[:200])
res = geo.search((0, {"a": 1}), lambda s, m: (s[0] + 1, {"a": s[1]["a"] + int(m)}) if s[0] < 4 else None,
                 lambda s: s[1]["a"] == 4, "01")
ok("search_tuple_with_dict", res["path"] == ["1", "1", "1"], str(res))
try:
    geo.search(0, lambda s, m: s + {"1": 1}[m], lambda s: s == 3, ["1", "5"])
    ok("search_error_names_move", False, "no error")
except KeyError as exc:
    ok("search_error_names_move", "step(state, '5')" in " ".join(getattr(exc, "__notes__", [])), repr(exc))

# --- number grids stay numbers ---------------------------------------------------------------------------------------
ok("rotate_int_lists", geo.rotate([[1, 12], [3, 4]]) == [[3, 1], [4, 12]], str(geo.rotate([[1, 12], [3, 4]])))
f = geo.flip(np.array([[1, 12], [3, 15]]), "h")
ok("flip_numpy_keeps_10_to_15", isinstance(f, np.ndarray) and f.dtype.kind == "i" and f.tolist() == [[12, 1], [15, 3]],
   repr(f))
rc = geo.recolor(np.array([[0, 7], [11, 0]], dtype=np.int8), {7: 3})
ok("recolor_numpy_keeps_dtype", rc.dtype == np.int8 and rc.tolist() == [[0, 3], [11, 0]], repr(rc))
ok("negative_numbers_fall_back", geo.rotate([[-1, 2]]) == [["-"], ["2"]], str(geo.rotate([[-1, 2]])))
ok("find_int_12_not_1", geo.find([[12]], [[0, 0, 0], [0, 1, 0]], turns=False) == [])

# --- reach to a color ---------------------------------------------------------------------------------------------------
p = geo.reach(["....Y", ".BBB.", "....."], (2, 0), ".", "Y")
ok("reach_letter_goal", p["path"] is not None and p["corners"][-1] == (0, 4) and len(p["path"]) == 6, str(p["path"]))
ok("reach_letter_missing", geo.reach(["....", "...."], (0, 0), ".", "Y")["path"] is None)

# --- track ------------------------------------------------------------------------------------------------------------
def board_with(*cells):
    b = [["."] * 8 for _ in range(8)]
    for r, c, x in cells:
        b[r][c] = x
    return ["".join(r) for r in b]


trows, me, it = [], (4, 4), (0, 0)
for i, m in enumerate("4422"):
    nme = (me[0] + D4[m][0], me[1] + D4[m][1])
    nit = (it[0] + (nme[0] > it[0]) - (nme[0] < it[0]), it[1] + (nme[1] > it[1]) - (nme[1] < it[1]))   # chases me
    trows.append({"i": i, "move": m, "before": board_with((*me, "Y"), (*it, "R")),
                  "after": board_with((*nme, "Y"), (*nit, "R"))})
    me, it = nme, nit
tr = geo.track("R", trows)
ok("track_chaser", [t["by"] for t in tr] == [[1, 1], [1, 1], [1, 1], [1, 1]] and tr[-1]["boxes"] == [[4, 4, 4, 4]],
   str(tr))
ok("track_you", [t["by"] for t in geo.track("Y", trows)] == [[0, 1], [0, 1], [1, 0], [1, 0]])

# --- undo_pairs ---------------------------------------------------------------------------------------------------------
def ub(pos, bar, door="D"):
    b = [list("B" * bar + "." * (8 - bar))] + [["."] * 8 for _ in range(6)]
    b[pos[0]][pos[1]] = "Y"
    b[6][7] = door
    return ["".join(r) for r in b]


seq = [((3, 3), "4", (3, 4)), ((3, 4), "3", (3, 3)), ((3, 3), "2", (4, 3)), ((4, 3), "1", (3, 3))]
urows = [{"i": i, "move": m, "state": "NOT_FINISHED", "before": ub(a, 8 - i), "after": ub(b, 7 - i)}
         for i, (a, m, b) in enumerate(seq)]
urows.append({"i": 4, "move": "5", "state": "NOT_FINISHED", "before": ub((3, 3), 4), "after": ub((3, 3), 3, ".")})
u = geo.undo_pairs(urows)
pairs = {(p["first"], p["then"]) for p in u["pairs"]}
ok("undo_pairs_found", {("4", "3"), ("2", "1")} <= pairs, str(u))
ok("undo_one_way", [o["move"] for o in u["one_way"]] == ["5"], str(u))
ok("undo_bar_ignored", "B" not in str(u["pairs"]) and u["pairs"], str(u))

# --- what the prompt offers -------------------------------------------------------------------------------------------
from harness.agent import prompts  # noqa: E402

sysp = prompts.game_system(act_max=100)
ok("prompt_offers_kept_tools", all(f"geo.{n}(" in sysp for n in
                                   ("rotate", "flip", "relate", "find", "symmetry", "camera", "ray", "reach",
                                    "counters", "track", "undo_pairs", "search")), "")
ok("prompt_drops_tools", not any(f"geo.{n}(" in sysp for n in
                                 ("transpose", "shift", "crop", "paste", "scale", "recolor", "line")))

# --- in the live kernel -----------------------------------------------------------------------------------------------
k = kernel_mod.Kernel(Path(tempfile.mkdtemp()) / "work", lambda req: {}, env={"SARBLOH_FILE_GUARD": "1"})
k.start()
try:
    r = k.execute('print(geo.rotate(["ab", "cd"]), geo.camera(["R.", ".."], [".R", ".."])["by"])')
    ok("kernel_has_geo", "['ca', 'db']" in r.stdout, r.render(600))
    names = k.user_names() or {}
    ok("geo_not_a_user_name", "geo" not in str(names), str(names))
finally:
    k.close()

print("================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
raise SystemExit(1 if FAILS else 0)
