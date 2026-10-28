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
ok("lists_stay_lists", geo.rotate([[1, 2], [3, 4]]) == [["3", "1"], ["4", "2"]])
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
