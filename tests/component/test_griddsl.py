# --- path shim: these suites live in tests/ but import the agent from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'eval')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""
Isolation test harness for COMPONENT 2a: THE WORLD MODEL'S VOCABULARY (GridDSL)
plus its compiler (compile_program).

Doctrine: the component STAYS in my_agent.py. This file only imports it and
feeds hand-built grids whose correct answer we know by construction. No game
engine, no synthesizer, no planner -- a failure here is the DSL's alone.

Why this component matters: every op is a CLAIM about physics that the
synthesizer will certify against real transitions and the planner will then
trust for unscored lookahead. An op that is subtly wrong does not fail loudly;
it gets certified on a few samples and then silently mispredicts during
planning. So the bar is exactness on the cases below, plus four invariants
that must hold for EVERY op:

  I1  shape-preserving          (the DSL is declared shape-preserving)
  I2  never mutates its input   (the Timeline holds those arrays)
  I3  preserves dtype           (downstream does integer color compares)
  I4  total on bad arguments    (returns identity, never a wrong prediction)

Bar is NOT "perfect", it is "passes this fixed set". Never delete a case.
Run:  python test_griddsl.py
"""
import sys
import itertools
import inspect

import numpy as np

from my_agent import GridDSL, compile_program, StateEncoder

D = GridDSL
FAILS = []
WARNS = []


def check(name, got, want):
    if got == want:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}: got {got!r}, want {want!r}")
        FAILS.append(name)


def check_grid(name, got, want):
    want = np.asarray(want)
    if isinstance(got, np.ndarray) and got.shape == want.shape and np.array_equal(got, want):
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}:\n    got\n{got}\n    want\n{want}")
        FAILS.append(name)


def warn(name, cond, detail):
    tag = "ok  " if cond else "WARN"
    if not cond:
        WARNS.append(name)
    print(f"  {tag}  {name}: {detail}")


def g(rows):
    return np.array(rows, dtype=int)


# A 4x4 scene reused across ops: background 0, an avatar 5, a wall 7.
SCENE = g([[0, 0, 0, 0],
           [0, 5, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 7, 0]])

print("== identity ==")
check_grid("identity_unchanged", D.identity(SCENE, 0), SCENE)

print("\n== translate (whole grid, bg fill) ==")
# Everything shifts down one row; the vacated top row fills with bg. The wall
# moves too -- translate is whole-grid physics by definition, not object-scoped.
check_grid("translate_down1", D.translate(SCENE, 0, dr=1, dc=0),
           [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 5, 0, 0],
            [0, 0, 0, 0]])
check_grid("translate_right1", D.translate(SCENE, 0, dr=0, dc=1),
           [[0, 0, 0, 0],
            [0, 0, 5, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 7]])
check_grid("translate_zero_is_identity", D.translate(SCENE, 0, dr=0, dc=0), SCENE)
# Shifted entirely off the grid -> nothing survives, all background.
check_grid("translate_off_grid", D.translate(SCENE, 0, dr=9, dc=0), np.zeros((4, 4), dtype=int))
# bg is honoured as the fill color, not hardcoded to 0.
check_grid("translate_fills_with_bg", D.translate(g([[1, 1], [1, 2]]), 1, dr=1, dc=0),
           [[1, 1], [1, 1]])

check_grid("translate_up_left", D.translate(SCENE, 0, dr=-1, dc=-1),
           [[5, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 7, 0, 0],
            [0, 0, 0, 0]])

# translate was rewritten from a per-pixel Python loop to slice-copies (it runs
# thousands of times per action during synthesis). Behaviour must be IDENTICAL,
# so check it against the original loop over the entire shift space, including
# every off-grid case, on a grid with no symmetry to hide an error behind.
def _translate_reference(gr, bg, dr, dc):
    out = np.full_like(gr, bg)
    h, w = gr.shape
    for r in range(h):
        for c in range(w):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = gr[r, c]
    return out

rng = np.random.RandomState(0)
asym = rng.randint(0, 5, size=(5, 7))
_bad = [(dr, dc) for dr in range(-8, 9) for dc in range(-9, 10)
        if not np.array_equal(D.translate(asym, 9, dr=dr, dc=dc), _translate_reference(asym, 9, dr, dc))]
check("translate_matches_reference_over_shift_space", _bad, [])

print("\n== recolor ==")
check_grid("recolor_5_to_3", D.recolor(SCENE, 0, src=5, dst=3),
           [[0, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 7, 0]])
check_grid("recolor_absent_color_noop", D.recolor(SCENE, 0, src=9, dst=3), SCENE)
check_grid("recolor_missing_args_noop", D.recolor(SCENE, 0), SCENE)

print("\n== reflect / rotate ==")
check_grid("reflect_h", D.reflect_h(g([[1, 2, 3]]), 0), [[3, 2, 1]])
check_grid("reflect_v", D.reflect_v(g([[1], [2], [3]]), 0), [[3], [2], [1]])
# rotate90 is CLOCKWISE (documented in GridDSL.help()).
check_grid("rotate90_clockwise", D.rotate90(g([[1, 2], [3, 4]]), 0), [[3, 1], [4, 2]])
check_grid("rotate90_four_times_identity",
           D.rotate90(D.rotate90(D.rotate90(D.rotate90(SCENE, 0), 0), 0), 0), SCENE)
# Non-square cannot rotate inside a shape-preserving DSL -> honest identity.
nonsq = g([[1, 2, 3], [4, 5, 6]])
check_grid("rotate90_nonsquare_identity", D.rotate90(nonsq, 0), nonsq)

print("\n== gravity ==")
FALL = g([[2, 0],
          [0, 3],
          [0, 0]])
check_grid("gravity_down", D.gravity(FALL, 0, direction="down"), [[0, 0], [0, 0], [2, 3]])
check_grid("gravity_up", D.gravity(FALL, 0, direction="up"), [[2, 3], [0, 0], [0, 0]])
check_grid("gravity_left", D.gravity(FALL, 0, direction="left"), [[2, 0], [3, 0], [0, 0]])
check_grid("gravity_right", D.gravity(FALL, 0, direction="right"), [[0, 2], [0, 3], [0, 0]])
# Stacking order is preserved (2 stays above 6), and a full column is unchanged.
STACK = g([[2, 4],
           [0, 4],
           [6, 4]])
check_grid("gravity_down_preserves_order", D.gravity(STACK, 0, direction="down"),
           [[0, 4], [2, 4], [6, 4]])
check_grid("gravity_empty_column_stays_bg", D.gravity(np.zeros((3, 2), dtype=int), 0), np.zeros((3, 2), dtype=int))
check_grid("gravity_default_is_down", D.gravity(FALL, 0), D.gravity(FALL, 0, direction="down"))
# I4: an unrecognised direction must not silently pick some other direction.
check_grid("gravity_bad_direction_identity", D.gravity(FALL, 0, direction="sideways"), FALL)

print("\n== flood_replace ==")
ROOM = g([[7, 7, 7, 7],
          [7, 0, 0, 7],
          [7, 0, 0, 7],
          [7, 7, 7, 7]])
check_grid("flood_fills_interior", D.flood_replace(ROOM, 7, r=1, c=1, dst=4),
           [[7, 7, 7, 7],
            [7, 4, 4, 7],
            [7, 4, 4, 7],
            [7, 7, 7, 7]])
# 4-connectivity: a diagonal-only neighbour is a different region.
DIAG = g([[1, 0],
          [0, 1]])
check_grid("flood_is_4_connected", D.flood_replace(DIAG, 0, r=0, c=0, dst=9), [[9, 0], [0, 1]])
check_grid("flood_same_color_noop", D.flood_replace(ROOM, 7, r=1, c=1, dst=0), ROOM)
check_grid("flood_out_of_bounds_noop", D.flood_replace(ROOM, 7, r=99, c=0, dst=4), ROOM)
check_grid("flood_missing_dst_noop", D.flood_replace(ROOM, 7, r=1, c=1), ROOM)

print("\n== translate_color (object-scoped) ==")
# ONLY the 5 moves; the wall 7 stays; the vacated cell becomes bg.
check_grid("translate_color_down", D.translate_color(SCENE, 0, color=5, dr=1, dc=0),
           [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 5, 0, 0],
            [0, 0, 7, 0]])
# Moving off the grid deletes those cells (they cannot be represented).
check_grid("translate_color_off_grid_deletes", D.translate_color(SCENE, 0, color=5, dr=-9, dc=0),
           [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 7, 0]])
check_grid("translate_color_missing_color_noop", D.translate_color(SCENE, 0, dr=1), SCENE)
check_grid("translate_color_absent_color_noop", D.translate_color(SCENE, 0, color=9, dr=1), SCENE)
# A multi-cell object moves rigidly.
BAR = g([[0, 0, 0],
         [6, 6, 0],
         [0, 0, 0]])
check_grid("translate_color_rigid", D.translate_color(BAR, 0, color=6, dr=0, dc=1),
           [[0, 0, 0], [0, 6, 6], [0, 0, 0]])

print("\n== step_color (one cell, or not at all) ==")
# The move that succeeds: one cell only, unlike slide_color.
check_grid("step_moves_exactly_one", D.step_color(SCENE, 0, color=5, direction="down"),
           [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 5, 0, 0],
            [0, 0, 7, 0]])
# THE case this op exists for: an obstacle directly ahead -> no change at all.
# translate_color would move through it; slide_color would already be flush.
WALLED = g([[0, 0],
            [0, 5],
            [0, 7]])
check_grid("step_blocked_by_object_is_identity", D.step_color(WALLED, 0, color=5, direction="down"), WALLED)
# The grid edge blocks exactly like an object -- the avatar is not deleted.
EDGE = g([[0, 5],
          [0, 0]])
check_grid("step_blocked_by_edge_is_identity", D.step_color(EDGE, 0, color=5, direction="up"), EDGE)
check_grid("step_up", D.step_color(SCENE, 0, color=5, direction="up"),
           [[0, 5, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 7, 0]])
check_grid("step_left", D.step_color(SCENE, 0, color=5, direction="left"),
           [[0, 0, 0, 0],
            [5, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 7, 0]])
# A multi-cell object steps rigidly and is NOT blocked by its own trailing cell.
PAIRV = g([[0, 6, 0],
           [0, 6, 0],
           [0, 0, 0]])
check_grid("step_multicell_not_self_blocking", D.step_color(PAIRV, 0, color=6, direction="down"),
           [[0, 0, 0], [0, 6, 0], [0, 6, 0]])
# ALL cells must be free: one blocked cell blocks the whole object.
PAIRH = g([[0, 0, 0],
           [6, 6, 0],
           [0, 7, 0]])
check_grid("step_partial_block_blocks_all", D.step_color(PAIRH, 0, color=6, direction="down"), PAIRH)
check_grid("step_missing_color_noop", D.step_color(SCENE, 0, direction="down"), SCENE)
check_grid("step_absent_color_noop", D.step_color(SCENE, 0, color=9, direction="down"), SCENE)
check_grid("step_bad_direction_identity", D.step_color(SCENE, 0, color=5, direction="sideways"), SCENE)
# It is a strictly finer hypothesis than its neighbours: agrees with them where
# the move is free, differs where it is blocked. That gap is its whole purpose.
FREE1 = g([[0, 0, 0],
           [0, 5, 0],
           [0, 0, 0]])
check("step_agrees_with_translate_when_free",
      np.array_equal(D.step_color(FREE1, 0, color=5, direction="down"),
                     D.translate_color(FREE1, 0, color=5, dr=1, dc=0)), True)
check("step_differs_from_translate_when_blocked",
      np.array_equal(D.step_color(WALLED, 0, color=5, direction="down"),
                     D.translate_color(WALLED, 0, color=5, dr=1, dc=0)), False)

print("\n== step_color stride (sprite lattice movement) ==")
# Real games move objects by a lattice stride of 3-7 px, not 1. A stride-1-only
# vocabulary cannot express their movement at all, so stride is load-bearing.
LANE = np.zeros((8, 8), dtype=int)
LANE[1, 1] = 5
check_grid("stride_moves_k_pixels", D.step_color(LANE, 0, color=5, direction="down", stride=4),
           np.where(np.arange(64).reshape(8, 8) == 5 * 8 + 1, 5, 0))
check_grid("stride_1_matches_default",
           D.step_color(LANE, 0, color=5, direction="down", stride=1),
           D.step_color(LANE, 0, color=5, direction="down"))
# A stride that would leave the grid is blocked, exactly like a stride of 1.
check_grid("stride_off_grid_is_identity", D.step_color(LANE, 0, color=5, direction="up", stride=4), LANE)
# The jump is rigid: only the DESTINATION must be clear. An obstacle strictly
# between origin and destination does not block a sprite that teleports a cell.
HOP = np.zeros((8, 8), dtype=int)
HOP[1, 1] = 5
HOP[3, 1] = 7                      # sits between origin and destination
hopped = D.step_color(HOP, 0, color=5, direction="down", stride=4)
check("stride_jumps_over_intermediate", int(hopped[5, 1]), 5)
check("stride_intermediate_untouched", int(hopped[3, 1]), 7)
# But a blocked DESTINATION still refuses the whole move.
HOPB = np.zeros((8, 8), dtype=int)
HOPB[1, 1] = 5
HOPB[5, 1] = 7
check_grid("stride_blocked_destination_is_identity",
           D.step_color(HOPB, 0, color=5, direction="down", stride=4), HOPB)
# A multi-cell sprite moves rigidly by the stride and is not blocked by itself
# even when the move OVERLAPS its own previous footprint.
SPRITE = np.zeros((8, 8), dtype=int)
SPRITE[1:4, 1:4] = 6               # 3x3 sprite, stride 2 -> overlaps itself
moved = D.step_color(SPRITE, 0, color=6, direction="down", stride=2)
want_sprite = np.zeros((8, 8), dtype=int)
want_sprite[3:6, 1:4] = 6
check_grid("stride_overlapping_self_move", moved, want_sprite)
# I4 again: nonsense strides decline rather than guess.
check_grid("stride_zero_is_identity", D.step_color(LANE, 0, color=5, direction="down", stride=0), LANE)
check_grid("stride_negative_is_identity", D.step_color(LANE, 0, color=5, direction="down", stride=-3), LANE)
check_grid("stride_none_is_identity", D.step_color(LANE, 0, color=5, direction="down", stride=None), LANE)

print("\n== vacate (the colour uncovered where the mover was) ==")
# WHY THIS ARGUMENT EXISTS: "the mover leaves background behind" is an
# assumption, not a law. On a board with a floor tile distinct from the outer
# chrome, an avatar stepping off a cell uncovers the FLOOR. Measured on the dev
# games, that one assumption is the difference between 0 and 8 (game, action)
# pairs whose transitions a single DSL rule reproduces EXACTLY.
# Floor 3, chrome/background 0, avatar 5, wall 7.
FLOOR = g([[0, 0, 0, 0],
           [0, 3, 3, 3],
           [0, 3, 5, 3],
           [0, 3, 7, 3]])
# translate_color: the source cell becomes `vacate`, not bg.
check_grid("vacate_translate_leaves_floor",
           D.translate_color(FLOOR, 0, color=5, dr=0, dc=-1, vacate=3),
           [[0, 0, 0, 0],
            [0, 3, 3, 3],
            [0, 5, 3, 3],
            [0, 3, 7, 3]])
# STRICT GENERALISATION: omitted, None, and vacate==bg must all be the old
# behaviour, bit for bit. This is what makes the argument safe to add.
check_grid("vacate_none_equals_omitted",
           D.translate_color(FLOOR, 0, color=5, dr=0, dc=-1, vacate=None),
           D.translate_color(FLOOR, 0, color=5, dr=0, dc=-1))
check_grid("vacate_bg_equals_default",
           D.translate_color(FLOOR, 0, color=5, dr=0, dc=-1, vacate=0),
           D.translate_color(FLOOR, 0, color=5, dr=0, dc=-1))
check_grid("vacate_step_none_equals_omitted",
           D.step_color(FLOOR, 0, color=5, direction="up", vacate=None),
           D.step_color(FLOOR, 0, color=5, direction="up"))
check_grid("vacate_step_bg_equals_default",
           D.step_color(FLOOR, 0, color=5, direction="up", vacate=0),
           D.step_color(FLOOR, 0, color=5, direction="up"))
# THE POINT OF THE ARGUMENT, and the bug it hid: step_color decides "blocked"
# from `(g != bg) & (g != color)`. With bg reported as the chrome colour, every
# floor cell reads as a wall and the op is identity EVERYWHERE -- dead weight in
# exactly the games vacate was added for. The uncovered colour is floor, so it
# must be passable.
check_grid("vacate_uncovered_colour_is_passable",
           D.step_color(FLOOR, 0, color=5, direction="up", vacate=3),
           [[0, 0, 0, 0],
            [0, 3, 5, 3],
            [0, 3, 3, 3],
            [0, 3, 7, 3]])
# Without vacate the same press is refused, because floor 3 != bg 0 reads wall.
# Kept as a case, not a bug: it documents why the argument had to widen `others`.
check_grid("vacate_absent_makes_floor_a_wall",
           D.step_color(FLOOR, 0, color=5, direction="up"), FLOOR)
# WALLS ARE STILL WALLS. vacate widens FREE by exactly one colour; anything else
# non-background blocks as before, and a blocked press is a WHOLE-GRID identity
# -- no half-move that paints vacate at the source and drops the mover.
check_grid("vacate_does_not_unblock_a_wall",
           D.step_color(FLOOR, 0, color=5, direction="down", vacate=3), FLOOR)
check_grid("vacate_blocked_by_edge_is_identity",
           D.step_color(FLOOR, 0, color=5, direction="down", stride=3, vacate=3), FLOOR)
# A multi-cell mover uncovers floor across its whole footprint, and overlapping
# its own previous cells does not overwrite the mover with vacate.
SPRITEF = np.full((8, 8), 3, dtype=int)
SPRITEF[1:4, 1:4] = 6
movedf = D.step_color(SPRITEF, 0, color=6, direction="down", stride=2, vacate=3)
wantf = np.full((8, 8), 3, dtype=int)
wantf[3:6, 1:4] = 6
check_grid("vacate_overlapping_self_move_keeps_mover", movedf, wantf)
# vacate == color is degenerate but must not corrupt: the mover paints its own
# colour behind it, so the footprint grows rather than moves. Total, not wrong.
smear = D.step_color(FLOOR, 0, color=5, direction="up", vacate=5)
check("vacate_equals_color_is_total", smear.shape == FLOOR.shape and smear.dtype == FLOOR.dtype, True)
# A float vacate COERCES rather than declining -- _as_int truncates, same as it
# does for dr/dc/stride. Pinned so the convention stays uniform across args.
check_grid("vacate_float_truncates_like_other_int_args",
           D.translate_color(FLOOR, 0, color=5, dr=0, dc=-1, vacate=3.7),
           D.translate_color(FLOOR, 0, color=5, dr=0, dc=-1, vacate=3))
# I4: non-numeric vacate values decline rather than guess a colour.
for badv in ([1], {"a": 1}, "x", object()):
    tag = type(badv).__name__
    check_grid(f"vacate_bad_{tag}_translate_defaults_to_bg",
               D.translate_color(FLOOR, 0, color=5, dr=0, dc=-1, vacate=badv),
               D.translate_color(FLOOR, 0, color=5, dr=0, dc=-1))
    check_grid(f"vacate_bad_{tag}_step_defaults_to_bg",
               D.step_color(FLOOR, 0, color=5, direction="up", vacate=badv),
               D.step_color(FLOOR, 0, color=5, direction="up"))
# 0-size grid: `g != vac` must not be the numpy trap that `g == 'x'` is.
EMPTY0 = np.zeros((0, 0), dtype=int)
for vv in (3, None, "x"):
    check("vacate_zero_size_translate_%s" % vv,
          D.translate_color(EMPTY0, 0, color=5, dr=1, dc=0, vacate=vv).shape, (0, 0))
    check("vacate_zero_size_step_%s" % vv,
          D.step_color(EMPTY0, 0, color=5, direction="down", vacate=vv).shape, (0, 0))

print("\n== slide_color (rigid slide until blocked) ==")
FREE = g([[0, 0],
          [0, 5],
          [0, 0],
          [0, 0]])
check_grid("slide_to_wall", D.slide_color(FREE, 0, color=5, direction="down"),
           [[0, 0], [0, 0], [0, 0], [0, 5]])
BLOCKED = g([[0, 0],
             [0, 5],
             [0, 0],
             [0, 7]])
check_grid("slide_stops_before_obstacle", D.slide_color(BLOCKED, 0, color=5, direction="down"),
           [[0, 0], [0, 0], [0, 5], [0, 7]])
# Already flush against the wall -> no movement, not an error.
FLUSH = g([[0, 0],
           [0, 0],
           [0, 5]])
check_grid("slide_already_flush", D.slide_color(FLUSH, 0, color=5, direction="down"), FLUSH)
check_grid("slide_up", D.slide_color(FREE, 0, color=5, direction="up"),
           [[0, 5], [0, 0], [0, 0], [0, 0]])
check_grid("slide_left", D.slide_color(FREE, 0, color=5, direction="left"),
           [[0, 0], [5, 0], [0, 0], [0, 0]])
check_grid("slide_absent_color_noop", D.slide_color(FREE, 0, color=9, direction="down"), FREE)
check_grid("slide_missing_color_noop", D.slide_color(FREE, 0, direction="down"), FREE)
# I4: unrecognised direction must not silently slide downward.
check_grid("slide_bad_direction_identity", D.slide_color(FREE, 0, color=5, direction="sideways"), FREE)
# A two-cell object slides rigidly and stops when EITHER cell would be blocked.
PAIR = g([[0, 5, 5],
          [0, 0, 0],
          [0, 7, 0]])
check_grid("slide_pair_stops_on_first_contact", D.slide_color(PAIR, 0, color=5, direction="down"),
           [[0, 0, 0], [0, 5, 5], [0, 7, 0]])

print("\n== swap_colors / delete_color ==")
check_grid("swap_two_colors", D.swap_colors(SCENE, 0, a=5, b=7),
           [[0, 0, 0, 0],
            [0, 7, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 5, 0]])
check_grid("swap_same_color_noop", D.swap_colors(SCENE, 0, a=5, b=5), SCENE)
check_grid("swap_missing_args_noop", D.swap_colors(SCENE, 0, a=5), SCENE)
check_grid("delete_color", D.delete_color(SCENE, 0, color=7),
           [[0, 0, 0, 0],
            [0, 5, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]])
check_grid("delete_missing_color_noop", D.delete_color(SCENE, 0), SCENE)
check_grid("delete_bg_noop", D.delete_color(SCENE, 0, color=0), SCENE)

print("\n== remap_colors: whole-palette maps, applied SIMULTANEOUSLY ==")
# Why this op exists: the mechanical games are colour permutations, and a
# permutation has no representation as a chain of recolors -- the second rule
# reads the first one's output. Every mask here is read off the SOURCE grid, so
# the result cannot depend on the order the entries happen to be iterated in.
CYC = g([[1, 2, 3],
         [1, 2, 3],
         [0, 0, 0]])
# The 3-cycle 1->2->3->1: the case the old DSL could not express AT ALL.
check_grid("remap_three_cycle", D.remap_colors(CYC, 0, mapping={1: 2, 2: 3, 3: 1}),
           [[2, 3, 1], [2, 3, 1], [0, 0, 0]])
# ... and it is a genuine cycle: applying it three times is the identity.
_r3 = CYC
for _ in range(3):
    _r3 = D.remap_colors(_r3, 0, mapping={1: 2, 2: 3, 3: 1})
check_grid("remap_three_cycle_order_three", _r3, CYC)
# The collision a sequential chain would produce, pinned as the counter-example:
# recolor 1->2 then recolor 2->3 turns the ORIGINAL 1s into 3s. remap does not.
_seq = D.recolor(D.recolor(CYC, 0, src=1, dst=2), 0, src=2, dst=3)
check("sequential_recolor_collides", np.array_equal(
    _seq, D.remap_colors(CYC, 0, mapping={1: 2, 2: 3})), False)
check_grid("remap_no_collision", D.remap_colors(CYC, 0, mapping={1: 2, 2: 3}),
           [[2, 3, 3], [2, 3, 3], [0, 0, 0]])
# Iteration order must not matter -- the same map spelled backwards, same answer.
check_grid("remap_order_independent",
           D.remap_colors(CYC, 0, mapping={3: 1, 2: 3, 1: 2}),
           D.remap_colors(CYC, 0, mapping={1: 2, 2: 3, 3: 1}))
# It subsumes the two ops it generalises, exactly.
check_grid("remap_subsumes_swap", D.remap_colors(SCENE, 0, mapping={5: 7, 7: 5}),
           D.swap_colors(SCENE, 0, a=5, b=7))
check_grid("remap_subsumes_recolor", D.remap_colors(SCENE, 0, mapping={5: 3}),
           D.recolor(SCENE, 0, src=5, dst=3))
# A two-config TOGGLE: the map is its own inverse, so applying it twice returns.
TOG = {1: 3, 3: 1, 2: 0, 0: 2}
check_grid("remap_toggle_is_involution",
           D.remap_colors(D.remap_colors(CYC, 0, mapping=TOG), 0, mapping=TOG), CYC)
# Colours absent from the map are untouched; an identity entry is a no-op.
check_grid("remap_leaves_unmapped_alone", D.remap_colors(CYC, 0, mapping={1: 9}),
           [[9, 2, 3], [9, 2, 3], [0, 0, 0]])
check_grid("remap_identity_entry_noop", D.remap_colors(CYC, 0, mapping={2: 2}), CYC)
# String keys/values survive a JSON round-trip of a program spec.
check_grid("remap_string_keys_coerced", D.remap_colors(CYC, 0, mapping={"1": "2"}),
           D.remap_colors(CYC, 0, mapping={1: 2}))
# Totality: the map is refused WHOLE, never applied in part -- a half-applied
# permutation is a different hypothesis from the one that was certified.
check_grid("remap_empty_noop", D.remap_colors(CYC, 0, mapping={}), CYC)
check_grid("remap_none_noop", D.remap_colors(CYC, 0), CYC)
check_grid("remap_not_a_dict_noop", D.remap_colors(CYC, 0, mapping=[(1, 2)]), CYC)
check_grid("remap_bad_entry_refuses_whole_map",
           D.remap_colors(CYC, 0, mapping={1: 2, 2: "x"}), CYC)
check_grid("remap_bad_key_refuses_whole_map",
           D.remap_colors(CYC, 0, mapping={"x": 2}), CYC)

print("\n== invariants over EVERY registered op ==")
# One plausible argument set per op. The point is not the semantics (covered
# above) but that no op can corrupt shape, dtype, or the caller's array.
ARGS = {
    "identity": {},
    "translate": {"dr": 1, "dc": -1},
    "recolor": {"src": 5, "dst": 3},
    "reflect_h": {}, "reflect_v": {}, "rotate90": {},
    "gravity": {"direction": "down"},
    "flood_replace": {"r": 0, "c": 0, "dst": 4},
    "translate_color": {"color": 5, "dr": 1, "dc": 0, "vacate": 3},
    "step_color": {"color": 5, "direction": "down", "vacate": 3},
    "slide_color": {"color": 5, "direction": "down"},
    "swap_colors": {"a": 5, "b": 7},
    "remap_colors": {"mapping": {5: 7, 7: 3}},
    "delete_color": {"color": 7},
}
check("all_ops_have_invariant_args", sorted(ARGS), sorted(D.OPS))
for name, fn in sorted(D.OPS.items()):
    src = SCENE.copy()
    out = fn(src, 0, **ARGS[name])
    check(f"I1_shape_{name}", isinstance(out, np.ndarray) and out.shape == SCENE.shape, True)
    check(f"I2_no_mutation_{name}", np.array_equal(src, SCENE), True)
    check(f"I3_dtype_{name}", out.dtype == SCENE.dtype, True)

print("\n== I4: every op is total on a nonsense argument ==")
# A wrong prediction is worse than no prediction: an op handed an argument it
# cannot honour must return the grid unchanged rather than guess.
BAD = {
    "translate": {"dr": None, "dc": 0},
    "recolor": {"src": None, "dst": None},
    "gravity": {"direction": "diagonal"},
    "flood_replace": {"r": -5, "c": 0, "dst": 1},
    "translate_color": {"color": None, "dr": 1, "dc": 0},
    "step_color": {"color": 5, "direction": "diagonal"},
    "slide_color": {"color": 5, "direction": "diagonal"},
    "swap_colors": {"a": None, "b": 3},
    "remap_colors": {"mapping": {5: None}},
    "delete_color": {"color": None},
}
for name, bad in sorted(BAD.items()):
    try:
        out = D.OPS[name](SCENE.copy(), 0, **bad)
        ok = isinstance(out, np.ndarray) and np.array_equal(out, SCENE)
    except Exception as e:
        ok = False
        out = f"raised {type(e).__name__}: {e}"
    check(f"I4_total_{name}", ok, True)

# Named regressions for the five I4 defects the single-BAD-case table above did
# NOT catch. Each of these raised before 2026-07-29; a raise inside a planner
# rollout kills the whole search, not just the one op call.
for _n, _kw in [
    ("recolor", {"src": 1, "dst": "x"}),
    ("swap_colors", {"a": "x", "b": 1}),
    ("flood_replace", {"r": None, "c": None, "dst": 1}),
    ("flood_replace", {"r": 0, "c": 0, "dst": "x"}),
    ("translate_color", {"color": 5, "dr": None, "dc": 0}),
    ("delete_color", {"color": "x"}),
    ("step_color", {"color": "x", "direction": "down"}),
    ("slide_color", {"color": "x", "direction": "down"}),
    ("translate", {"dr": [1], "dc": 0}),
]:
    try:
        _o = D.OPS[_n](SCENE.copy(), 0, **_kw)
        _ok = isinstance(_o, np.ndarray) and np.array_equal(_o, SCENE)
    except Exception as e:
        _ok = f"raised {type(e).__name__}"
    check(f"I4_nonnumeric_{_n}_{'_'.join(str(v) for v in _kw.values())}", _ok, True)

# The 0-size grid is its own case: `g == 'x'` returns all-False on a normal int
# array but RAISES on a 0-size one, so a colour guard that only checks None
# passes every test above and still dies on an empty grid.
for _n, _kw in [("translate_color", {"color": "x", "dr": 0, "dc": 1}),
                ("delete_color", {"color": "x"}),
                ("recolor", {"src": "x", "dst": 1})]:
    _empty = np.zeros((0, 0), dtype=np.int32)
    try:
        _o = D.OPS[_n](_empty.copy(), 0, **_kw)
        _ok = isinstance(_o, np.ndarray) and _o.shape == (0, 0)
    except Exception as e:
        _ok = f"raised {type(e).__name__}"
    check(f"I4_empty_grid_{_n}", _ok, True)

print("\n== I1-I4 over the FULL argument space of every op ==")
# The hand-picked BAD case above catches one bad argument per op; it missed
# five real defects (recolor/flood_replace/translate_color/swap_colors raising
# on a non-numeric colour, and `g == 'x'` raising rather than returning
# all-False on a 0-size grid). This sweep is the exhaustive version: every op
# x every combination of a fixed pool of good AND malformed values x every
# grid shape the agent can hand it, including 64x64 (the real frame size) and
# 0x0. Nothing here is random -- the pools are fixed, so this is a labeled
# test set that stays labeled, not a fuzz that reports something different
# each run.
POOL = {
    "dr": [-3, 0, 2, None, "x", 1.5, True, [1]], "dc": [-3, 0, 2, None, "x", 1.5],
    "color": [None, 0, 1, 9, "x", [1]],
    "direction": ["down", "up", "left", "right", "sideways", None, 3],
    "stride": [1, 3, 0, -2, None, "x", 2.7],
    "src": [None, 0, 1, "x"], "dst": [None, 1, "x"],
    "r": [0, 1, -1, None, "x", 99], "c": [0, 1, -1, None, "x"],
    "a": [None, 0, 1, "x"], "b": [None, 1, 2, "x"],
    "vacate": [None, 0, 3, "x"],
    # remap's argument is a whole CONTAINER, so the malformed shapes it has to
    # survive are different in kind: wrong type, empty, unhashable/non-numeric
    # keys and values, and a self-referential entry.
    "mapping": [None, {}, {1: 2}, {1: 2, 2: 3, 3: 1}, {1: 1}, {0: 9},
                {"1": "2"}, {1: None}, {None: 1}, {1: "x"}, {"x": 1},
                {1: [2]}, {1: 2.5}, {2.5: 1}, {1: True},
                [(1, 2)], "x", 7, [1, 2]],
}
_SWEEP_SHAPES = [(1, 1), (3, 3), (4, 6), (1, 5), (5, 1), (0, 0), (64, 64)]
_swept = 0
for _name, _fn in sorted(D.OPS.items()):
    _ps = [p for p in inspect.signature(_fn).parameters if p not in ("g", "bg")]
    _bad_cases = []
    for _shape in _SWEEP_SHAPES:
        _base = (np.arange(int(np.prod(_shape))) % 4).reshape(_shape).astype(np.int32)
        for _combo in itertools.product(*[POOL.get(p, [None, 1, "x"]) for p in _ps]):
            _kw = dict(zip(_ps, _combo))
            _g = _base.copy()
            _swept += 1
            try:
                _out = _fn(_g, 0, **_kw)
            except Exception as e:
                _bad_cases.append(f"{_shape}{_kw} raised {type(e).__name__}")
                continue
            if not isinstance(_out, np.ndarray):
                _bad_cases.append(f"{_shape}{_kw} returned {type(_out).__name__}")
            elif _out.shape != _shape:
                _bad_cases.append(f"{_shape}{_kw} -> shape {_out.shape}")
            elif _out.dtype != _base.dtype:
                _bad_cases.append(f"{_shape}{_kw} -> dtype {_out.dtype}")
            elif not np.array_equal(_g, _base):
                _bad_cases.append(f"{_shape}{_kw} mutated the input")
    check(f"sweep_{_name}", _bad_cases[:3], [])
print(f"  (swept {_swept} op/argument/shape combinations)")

print("\n== degenerate grid shapes must not crash lookahead ==")
# Ops are called inside planner rollouts; an exception there kills the search,
# not just the op. 1x1 and empty grids are the shapes with no interior at all.
for shape in [(1, 1), (0, 0), (1, 5), (5, 1)]:
    tiny = np.zeros(shape, dtype=int)
    broke = []
    for name, fn in sorted(D.OPS.items()):
        try:
            out = fn(tiny.copy(), 0, **ARGS[name])
            if not (isinstance(out, np.ndarray) and out.shape == shape):
                broke.append(name)
        except Exception as e:
            broke.append(f"{name}:{type(e).__name__}")
    check(f"degenerate_{shape[0]}x{shape[1]}", broke, [])

print("\n== translate is NOT invertible (data is lost at the edge) ==")
# Planners must not assume they can undo a shift: the pixels pushed off the
# grid are gone, and the vacated strip is background. Pinned so that a future
# "optimisation" cannot quietly treat translate as a reversible move.
there = D.translate(SCENE, 0, dr=1, dc=0)
back = D.translate(there, 0, dr=-1, dc=0)
check("translate_roundtrip_is_lossy", np.array_equal(back, SCENE), False)
check_grid("translate_roundtrip_loses_bottom_row", back,
           [[0, 0, 0, 0],
            [0, 5, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]])
# The object-scoped move IS reversible when nothing leaves the grid -- this is
# the property that makes translate_color safe for backward search.
check_grid("translate_color_roundtrip_exact",
           D.translate_color(D.translate_color(SCENE, 0, color=5, dr=1, dc=1), 0, color=5, dr=-1, dc=-1),
           SCENE)

print("\n== ops are deterministic ==")
for name, fn in sorted(D.OPS.items()):
    a = fn(SCENE.copy(), 0, **ARGS[name])
    b = fn(SCENE.copy(), 0, **ARGS[name])
    check(f"deterministic_{name}", np.array_equal(a, b), True)

print("\n== compile_program ==")
enc = StateEncoder()
# A one-rule program gated on ACTION2 must fire for ACTION2 and no other.
prog = [{"action": "ACTION2", "op": "translate_color", "args": {"color": 5, "dr": 1, "dc": 0}}]
fn = compile_program(prog, enc)
check("compile_returns_callable", callable(fn), True)
check_grid("compile_applies_on_matching_action", fn(SCENE, "ACTION2"),
           D.translate_color(SCENE, 0, color=5, dr=1, dc=0))
# An action no rule claims is UNKNOWN, not a no-op. This case used to expect
# SCENE back; "that action changes nothing" is a claim the program never made,
# and asserting it capped WorldModelManager.verify at the share of its sample the
# program happened to cover -- 0.32 on the dev games, against a 0.8 threshold.
check("compile_skips_other_action", fn(SCENE, "ACTION1"), None)
check("compile_covers_claimed_action", "ACTION2" in fn.covers, True)
check("compile_covers_excludes_unclaimed", "ACTION1" in fn.covers, False)
check("compile_covers_normalises_action6",
      "ACTION6" in compile_program(
          [{"action": "ACTION6", "op": "identity", "args": {}}], enc).covers, True)
check_grid("compile_action6_variant_hits_base_rule",
           compile_program([{"action": "ACTION6", "op": "delete_color",
                             "args": {"color": 7}}], enc)(SCENE, "ACTION6_r1_c2"),
           D.delete_color(SCENE, 0, color=7))
# A wildcard rule fires for every action.
wild = compile_program([{"action": "*", "op": "delete_color", "args": {"color": 7}}], enc)
check_grid("compile_wildcard_fires", wild(SCENE, "ACTION4"), D.delete_color(SCENE, 0, color=7))
check("compile_wildcard_covers_everything", wild.covers, None)
# Rules compose in order: delete the wall, then move the avatar.
seq = compile_program([{"action": "*", "op": "delete_color", "args": {"color": 7}},
                       {"action": "*", "op": "translate_color", "args": {"color": 5, "dr": 2, "dc": 0}}], enc)
check_grid("compile_rules_compose_in_order", seq(SCENE, "ACTION1"),
           [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 5, 0, 0]])
# Rejected at COMPILE time, so a bad program can never be trusted as a model.
check("compile_unknown_op_is_none", compile_program([{"op": "teleport"}], enc), None)
check("compile_empty_is_none", compile_program([], enc), None)
check("compile_not_a_list_is_none", compile_program({"op": "identity"}, enc), None)
check("compile_rule_not_a_dict_is_none", compile_program(["identity"], enc), None)
# Bad ARGS survive compilation (only the op name is checked), so the runtime
# fallback must degrade to identity rather than raise into the planner.
badargs = compile_program([{"action": "*", "op": "translate", "args": {"nope": 1}}], enc)
check("compile_badargs_still_callable", callable(badargs), True)
check_grid("compile_badargs_runtime_identity", badargs(SCENE, "ACTION1"), SCENE)
# The compiled transition must not mutate the grid it was handed.
probe = SCENE.copy()
fn(probe, "ACTION2")
check_grid("compile_does_not_mutate_input", probe, SCENE)
# A list input is accepted (compile_program calls np.asarray).
check_grid("compile_accepts_list_input", wild(SCENE.tolist(), "ACTION1"),
           D.delete_color(SCENE, 0, color=7))

print("\n================ SUMMARY ================")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
sys.exit(1 if FAILS else 0)
