# --- path shim: these suites live in tests/ but import the agent from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_ROOT, _os.path.join(_ROOT, 'eval'), _os.path.join(_ROOT, 'sarbloh', 'legacy')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""
Isolation test harness for COMPONENT 1: EYES (StateEncoder).

Doctrine: the component STAYS in my_agent.py. This file only imports it and
feeds hand-built grids whose correct answer we know by construction, so a
failure is visible for the eyes alone -- no game engine, no LLM, no planner.

Bar is NOT "perfect", it is "passes this fixed set". Never delete a case.
Run:  python test_eyes.py
"""
import sys
import numpy as np

from my_agent import StateEncoder

E = StateEncoder()
FAILS = []
WARNS = []


def check(name, got, want):
    if got == want:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}: got {got!r}, want {want!r}")
        FAILS.append(name)


def warn(name, cond, detail):
    tag = "ok  " if cond else "WARN"
    if not cond:
        WARNS.append(name)
    print(f"  {tag}  {name}: {detail}")


def g(rows):
    return np.array(rows, dtype=int)


print("== detect_background ==")
# All-zero interior, no border dominance -> bg 0
check("bg_all_zero", E.detect_background(g([[0, 0, 0], [0, 0, 0], [0, 0, 0]])), 0)
# Avatar pixel does not change the dominant background
check("bg_with_avatar", E.detect_background(g([[0, 0, 0], [0, 2, 0], [0, 0, 0]])), 0)
# Empty grid edge case
check("bg_empty", E.detect_background(np.zeros((0, 0), dtype=int)), 0)

print("\n== objects: single avatar ==")
grid = g([[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 2, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]])
objs = E.objects(grid, bg=0)
check("avatar_count", len(objs), 1)
if objs:
    o = objs[0]
    check("avatar_color", o.color, 2)
    check("avatar_size", o.size, 1)
    check("avatar_bbox", o.bbox, (2, 3, 2, 3))
    check("avatar_center", o.center, (2.0, 3.0))
    check("avatar_shape", o.shape_class, "single_pixel")

print("\n== objects: no non-background object ==")
check("empty_objs", len(E.objects(g([[0, 0], [0, 0]]), bg=0)), 0)

print("\n== 4-connectivity splits diagonals ==")
diag = g([[3, 0], [0, 3]])
check("diag_two_objects", len(E.objects(diag, bg=0)), 2)

print("\n== classify_shape ==")
check("hline", E.objects(g([[0, 0, 0, 0], [0, 3, 3, 3], [0, 0, 0, 0]]), bg=0)[0].shape_class, "horizontal_line")
check("vline", E.objects(g([[0, 0], [0, 5], [0, 5], [0, 5]]), bg=0)[0].shape_class, "vertical_line")
check("square", E.objects(g([[4, 4], [4, 4]]), bg=0)[0].shape_class, "square")
check("rectangle", E.objects(g([[4, 4, 4], [4, 4, 4]]), bg=0)[0].shape_class, "rectangle")
hollow = g([[0, 0, 0, 0, 0, 0],
            [0, 7, 7, 7, 7, 0],
            [0, 7, 0, 0, 7, 0],
            [0, 7, 0, 0, 7, 0],
            [0, 7, 7, 7, 7, 0],
            [0, 0, 0, 0, 0, 0]])
check("hollow", E.objects(hollow, bg=0)[0].shape_class, "hollow_shape")

print("\n== encode_transition ==")
prev = g([[0, 2, 0], [0, 0, 0]])
nxt = g([[0, 0, 0], [0, 2, 0]])            # avatar moved down
warn("transition_move", "movement/swap" in E.encode_transition(prev, nxt, "ACTION2"),
     E.encode_transition(prev, nxt, "ACTION2"))
warn("transition_nochange", "No change" in E.encode_transition(prev, prev, "ACTION1"),
     E.encode_transition(prev, prev, "ACTION1"))
warn("transition_resize", "resized" in E.encode_transition(prev, g([[0], [0]]), "ACTION4"),
     E.encode_transition(prev, g([[0], [0]]), "ACTION4"))

print("\n== full wall border (bordered maps) ==")
# Border all 8, interior 0 with one avatar. This is the ReachGoal-style bordered
# map. If bg is detected as the border color, the whole interior collapses into
# ONE phantom object and bg flips to the wall color, corrupting every downstream
# reader (LLM prompts, MCTS leaf value, object-scoped DSL ops).
bordered = g([[8, 8, 8, 8, 8],
              [8, 0, 0, 0, 8],
              [8, 0, 2, 0, 8],
              [8, 0, 0, 0, 8],
              [8, 8, 8, 8, 8]])
check("bordered_bg_is_0", E.detect_background(bordered), 0)
objs_b = E.objects(bordered)
check("bordered_obj_count", len(objs_b), 2)                       # wall ring + avatar
check("bordered_colors", sorted(o.color for o in objs_b), [2, 8])
check("bordered_avatar_isolated",
      any(o.color == 2 and o.size == 1 for o in objs_b), True)
check("bordered_wall_is_one_ring",
      [o.size for o in objs_b if o.color == 8], [16])

# Bordered maze: interior floor 0 with interior walls 8 (still <50% of interior).
maze = g([[8, 8, 8, 8, 8, 8],
          [8, 0, 0, 0, 0, 8],
          [8, 0, 8, 8, 0, 8],
          [8, 0, 0, 0, 0, 8],
          [8, 2, 0, 0, 0, 8],
          [8, 8, 8, 8, 8, 8]])
check("maze_bg_is_0", E.detect_background(maze), 0)
check("maze_avatar_isolated",
      any(o.color == 2 and o.size == 1 for o in E.objects(maze)), True)

print("\n== the frame rule must NOT over-fire ==")
# The border color also FILLS the interior -> it is the genuine background.
filled = g([[8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8],
            [8, 8, 2, 8, 8],
            [8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8]])
check("filled_bg_is_8", E.detect_background(filled), 8)
# Border color dominates the interior (>50%) even though another color is present.
mostly = g([[8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8],
            [8, 3, 8, 3, 8],
            [8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8]])
check("mostly_border_bg_is_8", E.detect_background(mostly), 8)
# One LARGE centered object on a plain background. The object owns a majority of
# the interior, so an interior-majority vote alone would call it the background;
# only the border/interior DENSITY ratio distinguishes it from a wall ring.
blocky = np.zeros((10, 10), dtype=int)
blocky[2:8, 2:8] = 3
check("large_object_is_not_bg", E.detect_background(blocky), 0)
check("large_object_is_one_object", len(E.objects(blocky)), 1)
# Speckled frame (a wall ring with gaps/doors) still reads as a frame.
noisy = g([[8, 0, 8, 0, 8],
           [0, 1, 1, 1, 8],
           [8, 1, 1, 1, 0],
           [0, 1, 1, 1, 8],
           [8, 0, 8, 0, 8]])
check("noisy_border_bg_is_majority", E.detect_background(noisy), 1)
# Interior too small to vote (single pixel) -> border stays background.
tiny = g([[8, 8, 8],
          [8, 2, 8],
          [8, 8, 8]])
check("tiny_interior_bg_is_8", E.detect_background(tiny), 8)
# No interior color reaches a majority -> border stays background (no guessing).
split = g([[8, 8, 8, 8, 8, 8],
           [8, 1, 1, 3, 3, 8],
           [8, 1, 1, 3, 3, 8],
           [8, 1, 1, 3, 3, 8],
           [8, 1, 1, 3, 3, 8],
           [8, 8, 8, 8, 8, 8]])
check("split_interior_bg_is_8", E.detect_background(split), 8)

print("\n== thick chrome band is not the background ==")
# A wide surround/HUD band (color 3) around an off-centre play field (color 4).
# The band wins the 1-px border vote outright, but it is over-represented on the
# border relative to its own area while the field owns MORE of the grid. The
# 1-px-interior frame rule above is blind to this: peel one row and the band is
# still there. Constructed by hand -- these are the proportions that make the
# question well-posed, not a copy of any game's frame.
chrome = np.full((20, 20), 3, dtype=int)
chrome[3:17, 4:20] = 4                       # field: 224 px vs band 176 px
check("chrome_band_bg_is_field", E.detect_background(chrome), 4)
objs_c = E.objects(chrome)
check("chrome_band_no_phantom", max(o.size for o in objs_c), 176)   # the band, not the field
# Same layout, but the band also owns the majority of the grid -> nothing better
# to choose, so the border winner stands (no guessing).
big_band = np.full((20, 20), 3, dtype=int)
big_band[6:14, 6:14] = 4                     # field 64 px, band 336 px
check("dominant_band_stays_bg", E.detect_background(big_band), 3)
# Two large panels, neither over-represented on the border -> border winner stands.
panels = np.zeros((20, 20), dtype=int)
panels[:, :11] = 1
panels[:, 11:] = 2
check("two_panels_border_winner", E.detect_background(panels), 1)
# A genuine background with objects on it is never over-represented on its own
# border, so the chrome rule must not fire here.
plain = np.zeros((20, 20), dtype=int)
plain[2:6, 2:6] = 7
plain[10:18, 10:18] = 9
check("plain_bg_unaffected", E.detect_background(plain), 0)

print("\n================ SUMMARY ================")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
sys.exit(1 if FAILS else 0)
