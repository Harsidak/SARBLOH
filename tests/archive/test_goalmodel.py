# --- path shim: these suites live in tests/ but import the agent from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_ROOT, _os.path.join(_ROOT, 'eval'), _os.path.join(_ROOT, 'sarbloh', 'legacy')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""
Isolation test harness for COMPONENT 5.12: GOAL MODEL (the order over
world-states).

Why this component gets its own harness: every other component answers a
question about WHAT IS. This one is the only thing in the agent that says WHICH
STATE IS BETTER, and a goal model that is confidently wrong is strictly worse
than no goal model at all -- it converts an unbiased walk (which is how the
agent's completed levels were actually won) into a biased walk toward a decoy
that the livelock breaker structurally cannot see, because every pursuit is a
different path to the same wrong place.

So the bar here is not "the goals are good". It is:

    the component can never REMOVE a choice, only reorder one;
    with no evidence it is a strict no-op, byte-identical to the agent
    without it; and a decoy that the record already refutes never costs a
    single scored action to reject.

Doctrine: the component STAYS in my_agent.py. This file only imports it and
feeds hand-built inputs whose correct answer is known by construction -- no game
engine, no LLM. Sections follow the architecture doc's harness contract
(Docs/goalmodel_architecture.md section 8):

    1. no-evidence no-op            6. falsification retires a hypothesis
    2. DECOY REFUTED BY HISTORY     7. channel priority (A over B)
    3. THE CLOCK TRAP               8. observational equivalence + MDL
    4. frontier permuted not cut    9. wiring vs the real choose_action
    5. cross-level transfer

Bar is NOT "perfect", it is "passes this fixed set". Never delete a case.
Run:  python test_goalmodel.py
"""
import os
import random
import sys

import numpy as np

from my_agent import (GoalModel, GoalPredicate, Readout, Feature, GridStats,
                      Timeline, StateGraph, DeadActionTracker, MyAgent,
                      GameState, base_action)

FAILS = []
WARNS = []


def check(name, got, want):
    if got == want:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}: got {got!r}, want {want!r}")
        FAILS.append(name)


def ok(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}" + (f": {detail}" if detail else ""))
    else:
        print(f"  FAIL  {name}: {detail}")
        FAILS.append(name)


def warn(name, cond, detail):
    tag = "ok  " if cond else "WARN"
    if not cond:
        WARNS.append(name)
    print(f"  {tag}  {name}: {detail}")


# ---------------------------------------------------------------- helpers
def blank(h=12, w=12, bg=0):
    return np.full((h, w), bg, dtype=np.int32)


def drive(gm, steps):
    """Feed (prev, action, nxt, reward, terminal) tuples through the REAL write
    path: append to the Timeline first (GoalModel reads the record, it never
    writes it), then observe."""
    for prev, action, nxt, reward, terminal in steps:
        gm.timeline.append(prev, action, nxt, reward, 0, terminal)
        gm.observe(prev, action, nxt, reward, terminal, 0)
    gm.certify()


def new_model(avatar=None, seed=0):
    tl = Timeline()
    return GoalModel(timeline=tl, avatar_fn=(lambda: avatar) if avatar is not None
                     else None, seed=seed)


def pin(gm, ro, direction=1, channel="A", alpha=5.0, beta=1.0, rng=(0.0, 4.0)):
    """Hand-build ONE feature with a known verdict, and freeze certification.

    Certification is not being tested here -- cases 2, 3 and 8 do that from the
    record. These cases test the CONSUMERS (potential / rank_frontier / target),
    which need a pool whose contents are known by construction rather than
    derived. Without the freeze `certify()` correctly re-judges the pinned
    feature back to 'no evidence' (there is no record behind it) and every
    consumer assertion would pass vacuously against an inert model."""
    f = Feature(ro, alpha=alpha, beta=beta)
    f.direction, f.channel, f.reason = direction, channel, "pinned"
    n = len(gm.timeline.entries)
    gm._feat[ro.key] = f
    gm._vp[ro.key] = [0.0] * n
    gm._vn[ro.key] = [0.0] * n
    gm._range[ro.key] = list(rng)
    gm._cert_at = (gm._n_seen, gm._pool_ver)
    return f


# ================================================================
print("\n=== 1. NO-EVIDENCE NO-OP (the correctness floor) ===")
# ================================================================
gm = new_model()
g = blank()
g[3, 3] = 4
check("empty_pool_has_no_target", gm.target(g), None)
check("empty_pool_potential_is_zero", gm.potential(g), (0.0, 0.0))
items = [(1, "ACTION1"), (2, "ACTION3"), (3, "ACTION2"), (4, "ACTION4")]
check("rank_frontier_is_identity_without_grids", gm.rank_frontier(items), items)
check("rank_frontier_is_identity_with_grids",
      gm.rank_frontier(items, grid_of=lambda it: g), items)
check("frontier_ranker_is_None_when_inert", gm.frontier_ranker(), None)
ok("explain_says_it_has_no_goal", "no adoptable goal" in gm.explain(), gm.explain())
check("pursuit_budget_floors_at_BUDGET_MIN", gm.pursuit_budget(), GoalModel.BUDGET_MIN)

# Totality of the whole vocabulary: nothing in it may ever raise, on any input.
weird = [blank(1, 1), np.zeros((0, 0), dtype=np.int32), blank(3, 3) - 5,
         np.full((4, 4), 99, dtype=np.int32)]
bad = []
for ro in [Readout("count", (3,)), Readout("exists", (3,)), Readout("cell", (9, 9)),
           Readout("regions", (3,)), Readout("objects", (3, 2)),
           Readout("dist", (1, 2)), Readout("contact", (1, 2)),
           Readout("align", (1, 2)), Readout("agree", (0, 1)),
           Readout("nonsense", (1,))]:
    for w in weird:
        try:
            v = ro.value(GridStats(w))
            float(v)
        except Exception as e:                                  # pragma: no cover
            bad.append((ro.describe(), w.shape, repr(e)))
ok("every_readout_is_total", not bad, f"raised on {bad[:3]}")

# A GoalPredicate is total too -- it is handed live frames by a caller that has
# no idea which readout is inside it.
p = GoalPredicate(Readout("count", (3,)), "<=", 0, ("count", (3,)))
ok("predicate_is_total_on_junk",
   p.holds(None) is False and p.holds("nonsense") is False and p.holds(blank()) is True,
   "None/str -> False, a grid with no colour 3 -> True")


# ================================================================
print("\n=== 2. A DECOY IS REFUTED BY HISTORY (the case that matters most) ===")
# ================================================================
# The world: avatar=5, BLUE=3 at (2,2) is the real goal, RED=2 at (9,9) is a
# decoy. The agent touched RED twice and got nothing; it touched BLUE once and
# the level completed. Nothing here is a scored action -- the actions were
# already spent, so this whole verdict is free.
AV, BLUE, RED = 5, 3, 2


def world(ar, ac):
    g = blank()
    g[2, 2] = BLUE
    g[9, 9] = RED
    g[ar, ac] = AV
    return g


gm = new_model(avatar=AV, seed=1)
steps = []
walk = [(6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (7, 5), (8, 5), (8, 6), (8, 7),
        (8, 8), (9, 8),                 # <- adjacent to RED  (no reward)
        (8, 8), (7, 8), (7, 7), (7, 6), (8, 6), (9, 6), (9, 7), (9, 8),
        (9, 8), (9, 8), (9, 8)]         # <- adjacent to RED again (no reward)
walk = [(6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (7, 5), (8, 5), (8, 6), (8, 7),
        (8, 8), (9, 8), (8, 8), (7, 8), (7, 7), (7, 6), (7, 5), (7, 4), (8, 4),
        (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (8, 8), (7, 8), (6, 8), (5, 8),
        (4, 8), (3, 8), (3, 7), (3, 6), (3, 5), (3, 4), (3, 3), (3, 2)]
acts = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
for i in range(len(walk) - 1):
    a = acts[i % 4]
    steps.append((world(*walk[i]), a, world(*walk[i + 1]), 0.0, False))
# (3,2) is adjacent to BLUE at (2,2). The NEXT action completes the level, and
# the engine reports it on a frame whose whole layout has already been swapped
# -- exactly the real shape, which is why Channel A reads the PRE-outcome frame.
after = blank()
after[7, 7] = AV
steps.append((world(3, 2), "ACTION1", after, 1.0, False))
drive(gm, steps)

RED_K, BLUE_K = ("contact", (AV, RED)), ("contact", (AV, BLUE))
ok("both_contact_features_were_proposed",
   RED_K in gm._feat and BLUE_K in gm._feat, f"pool={sorted(gm._feat)[:6]}...")
ok("reaching_RED_is_refuted_by_the_record", gm._refuted(RED_K, +1) is True,
   "contact(avatar,red) hit its maximum twice and no score event ever followed")
ok("reaching_BLUE_is_NOT_refuted", gm._refuted(BLUE_K, +1) is False,
   "the one time contact(avatar,blue) peaked, the level completed")
fb = gm._feat.get(BLUE_K)
ok("BLUE_is_adopted_as_a_positive_coordinate",
   fb is not None and fb.direction == 1 and fb.channel == "A",
   f"{fb.describe() if fb else None}")
fr = gm._feat.get(RED_K)
ok("RED_is_never_adopted_as_a_positive_coordinate",
   fr is None or fr.direction != 1,
   f"{fr.describe() if fr else None} -- learning it as NEGATIVE is allowed and "
   f"is the correct reading of the record; pursuing it is not")
# Zero scored actions were spent to reach that verdict: certify() only replays.
before_steps = gm._steps
gm.certify()
gm.certify()
check("certify_costs_no_actions", gm._steps, before_steps)
ok("certify_is_idempotent",
   gm._feat[BLUE_K].direction == 1 and gm._feat[BLUE_K].alpha == fb.alpha,
   "a second pass changes nothing")

# --- 2b. EVIDENCE THAT IS ONLY A BASE RATE IS NOT EVIDENCE -------------------
# The same disproportionality rule that makes sacred cells safe. A readout that
# is at its maximum on the winning frame -- and on almost every other frame too,
# because the thing it counts is nearly always on screen -- has told us nothing.
# Without this the first 25-game run adopted 11-19 hypotheses per game, every one
# of them at the untested prior, and the frontier was being steered by base rates.
gm = new_model(seed=13)
COMMON, RARE = 1, 4


def frame(i):
    """COMMON sits at its maximum on 27 of 30 frames -- including the winning one.
    RARE sits at its maximum on exactly one frame: the winning one."""
    g = blank(8, 8)
    n_common = 1 if i in (5, 11, 17) else 3
    g[0, :n_common] = COMMON
    g[2, :(4 if i == 29 else 1)] = RARE
    return g


steps = [(frame(i), acts[i % 4], frame(i + 1), 1.0 if i == 29 else 0.0, False)
         for i in range(30)]
drive(gm, steps)
ex_c = gm._extremity(("count", (COMMON,)), [29])
ex_r = gm._extremity(("count", (RARE,)), [29])
ok("a_readout_at_its_max_on_the_win_AND_almost_everywhere_is_NOT_confirmed",
   gm._confirmed(("count", (COMMON,))) == 0,
   f"marked={ex_c[0]:.2f} base={ex_c[1]:.2f} -> margin {ex_c[0]-ex_c[1]:.2f} "
   f"< {GoalModel.EXTREME_MARGIN}")
ok("a_readout_at_its_max_ONLY_at_the_win_IS_confirmed",
   gm._confirmed(("count", (RARE,))) == 1,
   f"marked={ex_r[0]:.2f} base={ex_r[1]:.2f} -> margin {ex_r[0]-ex_r[1]:.2f}")


# ================================================================
print("\n=== 3. THE CLOCK TRAP (section 5.1 -- the second gate) ===")
# ================================================================
# ZERO score events on record. Two readouts that both look like progress:
#   (a) KEY=7 -- a counter that only ever falls, and only under ACTION2;
#   (b) BAR=9 -- a progress bar that grows by one cell on EVERY action. It is
#       monotone, irreversible AND bounded, so it satisfies every structural
#       test at once. It is a clock.
KEY, BAR = 7, 9


def clocked(keys, bar):
    g = blank(12, 28)
    for i in range(keys):
        g[1, i] = KEY
    for i in range(bar):
        g[11, i] = BAR
    return g


gm = new_model(seed=2)
steps, keys, bar = [], 4, 0
prev = clocked(keys, bar)
for i in range(24):
    a = acts[i % 4]
    if a == "ACTION2" and keys > 0:
        keys -= 1
    bar += 1
    nxt = clocked(keys, bar)
    steps.append((prev, a, nxt, 0.0, False))
    prev = nxt
drive(gm, steps)

check("no_score_events_on_record",
      any(e.reward > 0 for e in gm.timeline.entries), False)
fk = gm._feat.get(("count", (KEY,)))
fb = gm._feat.get(("count", (BAR,)))
ok("the_real_counter_is_adopted",
   fk is not None and fk.direction == -1 and fk.channel == "B",
   f"{fk.describe() if fk else None}")
ok("the_step_counter_is_REJECTED",
   fb is None or (fb.direction == 0 and fb.reason == "clock"),
   f"{fb.describe() if fb else None}")
ok("the_clock_separator_is_what_rejected_it",
   gm._clock_ok(("count", (BAR,))) is False and gm._clock_ok(("count", (KEY,))) is True,
   "the bar advances under every action; the key counter only under ACTION2")
# The bar passes every STRUCTURAL test -- that is the whole point of the trap.
d, why = gm._structural(("count", (BAR,)))
ok("the_trap_is_real_the_bar_passes_every_structural_test",
   d == 1 and why == "monotone",
   "monotone + irreversible + bounded; only the clock separator can see it")

# The mirror, kept as an explicit WARN rather than papered over: a GENUINE
# progress coordinate that also happens to advance on every action is not
# separable from a clock with the information available here. Accepted for the
# same reason as `every_step_blinker_reads_as_hud` in test_stategraph.py -- the
# failure it protects against is much worse than the opportunity it costs.
gm2 = new_model(seed=3)
steps, keys = [], 25
prev = clocked(keys, 0)
for i in range(24):
    a = acts[i % 4]
    keys -= 1                       # a real key really is collected every action
    nxt = clocked(keys, 0)
    steps.append((prev, a, nxt, 0.0, False))
    prev = nxt
drive(gm2, steps)
fk2 = gm2._feat.get(("count", (KEY,)))
warn("genuine_coordinate_that_moves_every_step_reads_as_a_clock",
     fk2 is not None and fk2.direction == -1,
     "a real counter that advances under EVERY action is indistinguishable from "
     "a step counter here; rejected on purpose (5.1 residual blind spot)")


# ================================================================
print("\n=== 4. THE FRONTIER IS PERMUTED, NEVER TRUNCATED (guard 1) ===")
# ================================================================
# Property-style over many seeds: a randomly-wrong potential over randomly-built
# candidates must still be a permutation. This is the structural floor -- it is
# what bounds the worst case of an arbitrarily wrong goal at "a permutation of an
# order that was already arbitrary".
bad_sets, bad_lens, permuted, live = 0, 0, 0, 0
for seed in range(60):
    rng = random.Random(seed)
    gm = new_model(seed=seed)
    grids = {}
    cands = []
    for i in range(rng.randint(2, 12)):
        g = blank(8, 8)
        g[rng.randrange(8), rng.randrange(8)] = rng.randrange(1, 6)
        grids[i] = g
        cands.append((i, acts[rng.randrange(4)]))
    # Force a live, arbitrary pool: random readouts, random directions, random
    # credibilities. None of it is earned -- that is the point.
    for k in range(rng.randint(1, 5)):
        ro = Readout(rng.choice(["count", "exists", "regions"]), (rng.randrange(1, 6),))
        pin(gm, ro, direction=rng.choice([-1, 1]), channel=rng.choice(["A", "B"]),
            alpha=rng.uniform(1, 9), beta=rng.uniform(1, 9),
            rng=(0.0, float(rng.randint(1, 9))))
    # A pool whose every draw landed below the pursuit floor is legitimately
    # inert (rank_frontier is then identity by contract) -- counted, not failed,
    # so the vacuity guard below stays honest without asserting a coin flip.
    if gm._active():
        live += 1
    out = gm.rank_frontier(cands, grid_of=lambda c: grids.get(c[0]))
    if set(out) != set(cands):
        bad_sets += 1
    if len(out) != len(cands):
        bad_lens += 1
    if out != cands:
        permuted += 1
check("rank_frontier_preserves_the_set_over_60_seeds", bad_sets, 0)
check("rank_frontier_preserves_the_length_over_60_seeds", bad_lens, 0)
ok("the_property_test_is_not_vacuous", permuted > 0 and live >= 50,
   f"{live} of 60 pools were live; {permuted} actually reordered the frontier")

# Unscorable candidates keep their relative order and go LAST -- an unrankable
# pair is not a demoted pair.
gm = new_model(seed=7)
pin(gm, Readout("count", (4,)), direction=1, channel="A", alpha=8, beta=1,
    rng=(0.0, 4.0))
hi, lo = blank(6, 6), blank(6, 6)
hi[0, :4] = 4
c = [("x", None), ("lo", lo), ("y", None), ("hi", hi)]
out = gm.rank_frontier(c, grid_of=lambda it: it[1])
check("higher_potential_ranks_first", out[0][0], "hi")
check("unscorable_go_last_in_original_order", [t[0] for t in out[2:]], ["x", "y"])

# And the graph-level contract: a ranker is a permutation of the SHORTEST-distance
# frontier pairs, and a ranker that is not a permutation is ignored outright.
sg = StateGraph()
a0, a1, a2 = blank(4, 4), blank(4, 4), blank(4, 4)
a1[0, 0] = 1
a2[0, 1] = 1
sg.ensure_state(a0, ["ACTION1", "ACTION2", "ACTION3"])
sg.record(a0, "ACTION1", a1)
sg.ensure_state(a1, ["ACTION1", "ACTION2", "ACTION3"])
random.seed(11)
base = sg.nearest_untested_grid(a0, DeadActionTracker())
random.seed(11)
same = sg.nearest_untested_grid(a0, DeadActionTracker())
check("nearest_untested_is_deterministic_under_a_fixed_seed", base, same)
seen_actions = set()
for _ in range(30):
    r = sg.nearest_untested_grid(a0, DeadActionTracker(),
                                 rank=lambda cs: sorted(cs, key=lambda c: c[1],
                                                        reverse=True))
    seen_actions.add(r[1])
ok("a_ranker_can_only_pick_a_genuinely_untested_action",
   seen_actions <= {"ACTION2", "ACTION3"},
   f"picked {sorted(seen_actions)}; ACTION1 is already recorded from a0")
r = sg.nearest_untested_grid(a0, DeadActionTracker(), rank=lambda cs: cs[:1])
ok("a_ranker_that_drops_candidates_is_ignored", r is not None and r[1] in
   ("ACTION2", "ACTION3"), f"{r}")
r = sg.nearest_untested_grid(a0, DeadActionTracker(), rank=lambda cs: 1 / 0)
ok("a_ranker_that_raises_is_ignored", r is not None and r[1] in
   ("ACTION2", "ACTION3"), f"{r}")


# ================================================================
print("\n=== 5. CROSS-LEVEL TRANSFER (the two-tier lifecycle) ===")
# ================================================================
# RHAE's denominator spans every level of a game, so a level-1-only completion is
# worth ~3% and the score lives in layouts the agent has never seen. A learned
# win-condition is the only object in the system that can cross a level boundary
# -- so the RULES persist and only the LAYOUT bindings reset.
gm = new_model(seed=4)
ro_rule = Readout("count", (3,))
ro_pos = Readout("cell", (0, 5))
ro_panel = Readout("agree", (0, 1))
for ro in (ro_rule, ro_pos, ro_panel):
    f = Feature(ro, alpha=6, beta=1)
    f.direction, f.channel = -1, "A"
    gm._feat[ro.key] = f
    gm._vp[ro.key] = []
    gm._vn[ro.key] = []
    gm._range[ro.key] = [0.0, 5.0]
gm._retired.add(("count", (11,)))
gm.new_level()
ok("a_confirmed_RULE_survives_the_level_boundary",
   ro_rule.key in gm._feat and gm._feat[ro_rule.key].alpha == 6
   and gm._feat[ro_rule.key].direction == -1,
   "colour semantics are a property of the game, not of the layout")
ok("POSITIONAL_bindings_do_not_survive",
   ro_pos.key not in gm._feat and ro_panel.key not in gm._feat,
   "those coordinates belong to the old map")
check("per_level_value_ranges_are_cleared", gm._range, {})
ok("a_refuted_feature_stays_refuted_on_the_next_level",
   ("count", (11,)) in gm._retired, "retirement is permanent")
gm._propose(blank())     # a proposal round on the new level
ok("a_retired_feature_is_never_re_proposed",
   ("count", (11,)) not in gm._feat, "the pool cannot resurrect it")


# ================================================================
print("\n=== 6. FALSIFICATION RETIRES A HYPOTHESIS ===")
# ================================================================
# Satisfying a predicate without the predicted score event is a FAILURE. This is
# `_llm_goal`'s "arrived, no level-up -> drop the hint" generalised to every
# hypothesis, and it is what stops a plausible decoy absorbing a whole level.
gm = new_model(seed=5)
ro = Readout("count", (6,))
f = Feature(ro, alpha=1, beta=1)
f.direction, f.channel = -1, "B"
gm._feat[ro.key] = f
gm._vp[ro.key] = []
gm._vn[ro.key] = []
gm._range[ro.key] = [0.0, 3.0]
attempts = 0
target_grid = blank(6, 6)
target_grid[0, 0] = 6
while ro.key in gm._feat and attempts < 20:
    attempts += 1
    pred = GoalPredicate(ro, "<=", 0, ro.key)
    # 4th slot = the alpha warrant (case 11). False on purpose here: refutation
    # never needs one, so this loop must retire the hypothesis exactly as it did
    # before the warrant existed.
    gm._pursuit = (ro.key, pred, gm._steps, False)
    gm._steps += 1
    gm._settle_pursuit(blank(6, 6), 0.0)     # predicate satisfied, NO score event
ok("a_predicate_satisfied_without_a_score_event_retires_the_hypothesis",
   ro.key not in gm._feat and attempts <= 8, f"retired after {attempts} attempts")
ok("the_retirement_is_permanent", ro.key in gm._retired, "moved to _retired")
gm._propose(target_grid)
ok("later_unrelated_evidence_does_not_resurrect_it", ro.key not in gm._feat,
   "a proposal round on a board that contains colour 6 re-adds nothing")
# ... and the symmetric half: a score event during a pursuit is a confirmation.
ro2 = Readout("count", (8,))
f2 = Feature(ro2, alpha=1, beta=1)
f2.direction, f2.channel = -1, "B"
gm._feat[ro2.key] = f2
gm._pursuit = (ro2.key, GoalPredicate(ro2, "<=", 0, ro2.key), gm._steps, True)
gm._settle_pursuit(blank(6, 6), 1.0)
check("a_score_event_during_a_pursuit_confirms_it", gm._feat[ro2.key].alpha, 2.0)
# Same event WITHOUT the warrant is not a confirmation -- case 11 (f) is where
# that asymmetry is stated; asserted here too so the two halves of `_settle_pursuit`
# are pinned in the same place.
f3 = Feature(Readout("count", (9,)), alpha=1, beta=1)
f3.direction, f3.channel = -1, "B"
gm._feat[f3.ro.key] = f3
gm._pursuit = (f3.ro.key, GoalPredicate(f3.ro, "<=", 0, f3.ro.key), gm._steps, False)
gm._settle_pursuit(blank(6, 6), 1.0)
check("an_undirected_score_event_confirms_nothing", gm._feat[f3.ro.key].alpha, 1.0)


# ================================================================
print("\n=== 7. CHANNEL PRIORITY: A OUTRANKS B, ALWAYS ===")
# ================================================================
# An intrinsic (structural) signal may never outrank a confirmed one, whatever
# its credibility -- CDE's clipping law: an intrinsic bonus must never be able to
# reverse the sign of the extrinsic signal.
gm = new_model(seed=6)
roA, roB = Readout("count", (4,)), Readout("count", (5,))
# A is barely credible but CONFIRMED; B is near-certain but only STRUCTURAL.
fA = pin(gm, roA, direction=1, channel="A", alpha=1.05, beta=1.0, rng=(0.0, 4.0))
fB = pin(gm, roB, direction=1, channel="B", alpha=99.0, beta=1.0, rng=(0.0, 4.0))
gA, gB = blank(6, 6), blank(6, 6)
gA[0, :4] = 4          # A loves this one; B is indifferent
gB[1, :4] = 5          # B loves this one; A is indifferent
ok("channel_A_decides_the_order",
   gm.potential(gA)[0] > gm.potential(gB)[0],
   f"phi(A-good)={gm.potential(gA)[0]:.6f} > phi(B-good)={gm.potential(gB)[0]:.6f} "
   f"even though B's credibility is {fB.post_mean():.2f} vs A's {fA.post_mean():.2f}")
ok("B_still_breaks_ties_underneath_A",
   gm.potential(gB)[0] > gm.potential(blank(6, 6))[0],
   "B is not discarded, only dominated")
check("credibility_reported_is_channel_A's_when_A_is_live",
      round(gm.potential(gA)[1], 4), round(fA.post_mean(), 4))
# Thompson sampling must prefer A too, on every draw.
picks = set()
for _ in range(200):
    t = gm.target(gA)
    if t is not None:
        picks.add(t.key)
check("target_selection_never_prefers_B_over_A", picks, {roA.key})
ok("thompson_sampling_declines_when_the_confirmed_sample_is_weak",
   len(picks) == 1,
   "a low draw on A yields no target at all rather than promoting B -- "
   "'A outranks B' has to hold in selection, not only in the potential")


# ================================================================
print("\n=== 8. OBSERVATIONAL EQUIVALENCE + MDL TIE-BREAK ===")
# ================================================================
# Two readouts with identical value vectors over the whole record ARE the same
# hypothesis however differently they are written. Keeping both would double
# their weight in the potential and double the cost of every future backtest.
gm = new_model(seed=8)
DOT = 6


def dots(n):
    g = blank(10, 10)
    for i in range(n):
        g[2 * i, 0] = DOT          # n isolated single cells -> count == objects(.,1)
    return g


steps, n = [], 5
prev = dots(n)
for i in range(20):
    a = acts[i % 4]
    if a == "ACTION3" and n > 0:
        n -= 1
    nxt = dots(n)
    steps.append((prev, a, nxt, 0.0, False))
    prev = nxt
drive(gm, steps)
k_count, k_obj = ("count", (DOT,)), ("objects", (DOT, 1))
ok("the_two_readouts_really_are_observationally_equal",
   gm._vn.get(k_count) is None or gm._vn.get(k_obj) is None
   or gm._vn[k_count] == gm._vn[k_obj],
   "count(dot) == objects(dot, area 1) on every frame of this record")
ok("only_the_shorter_description_survives",
   (k_count in gm._feat) and (k_obj not in gm._feat),
   f"count dl={Readout('count', (DOT,)).dl} beat "
   f"objects dl={Readout('objects', (DOT, 1)).dl}")
check("pool_stays_within_MAX_FEATURES",
      len(gm._feat) <= GoalModel.MAX_FEATURES, True)
# Certification cost stays bounded as the record grows: a second pass with no new
# evidence does no work at all.
stamp = gm._cert_at
gm.certify()
check("certify_short_circuits_with_no_new_evidence", gm._cert_at, stamp)

# The cap is enforced even when candidates flood in, and it drops the WEAKEST.
gm = new_model(seed=9)
for i in range(GoalModel.MAX_FEATURES * 2):
    ro = Readout("count", (i,))
    f = Feature(ro, alpha=1.0 + i, beta=1.0)
    f.direction, f.channel = 1, "B"
    gm._feat[ro.key] = f
    gm._vn[ro.key] = [float(i)]
    gm._vp[ro.key] = [0.0]
gm._cap()
check("cap_trims_to_MAX_FEATURES", len(gm._feat), GoalModel.MAX_FEATURES)
ok("cap_keeps_the_most_credible",
   min(f.post_mean() for f in gm._feat.values())
   > max([1.0 / 2.0], default=0), "the alpha=1 features went first")


# ================================================================
print("\n=== 9. WIRING vs THE REAL MyAgent.choose_action ===")
# ================================================================
class Frame:
    def __init__(self, grid, actions, st=GameState.NOT_FINISHED, lvl=0):
        self.frame = [np.asarray(grid).tolist()]
        self.available_actions = actions
        self.state = st
        self.levels_completed = lvl


def spy(agent):
    calls = {"observe": [], "level": [], "terminal": 0}
    real_obs = agent.goals.observe
    real_lvl = agent.goals.note_level_complete

    def obs(prev, action, nxt, reward=0.0, terminal=False, level=0):
        calls["observe"].append((action, reward, terminal))
        if terminal:
            calls["terminal"] += 1
        return real_obs(prev, action, nxt, reward, terminal, level)

    def lvl(frame, level=0):
        calls["level"].append(level)
        return real_lvl(frame, level)

    agent.goals.observe = obs
    agent.goals.note_level_complete = lvl
    return calls


w0, w1, w2 = blank(8, 8), blank(8, 8), blank(8, 8)
w1[0, 0] = 1
w2[0, 1] = 1

os.environ["ARC_AGENT_SEED"] = "0"
ag = MyAgent(game_id="goalmodel-wiring")
calls = spy(ag)
ag.choose_action([Frame(w0, [1, 2, 3])], Frame(w0, [1, 2, 3]))
check("no_observe_before_there_is_a_transition", calls["observe"], [])
ag.choose_action([Frame(w1, [1, 2, 3])], Frame(w1, [1, 2, 3]))
ok("observe_fires_on_an_ordinary_transition", len(calls["observe"]) == 1,
   f"{calls['observe']}")
f_lvl = Frame(w2, [1, 2, 3], lvl=1)
ag.choose_action([f_lvl], f_lvl)
ok("observe_fires_on_a_score_event",
   len(calls["observe"]) == 2 and calls["observe"][-1][1] == 1.0,
   f"{calls['observe']}")
check("note_level_complete_fires_on_level_up", calls["level"], [0])
f_dead = Frame(w1, [1, 2, 3], st=GameState.GAME_OVER)
ag.choose_action([f_dead], f_dead)
ok("observe_fires_on_GAME_OVER_and_is_flagged_terminal", calls["terminal"] == 1,
   f"{calls['observe'][-1] if calls['observe'] else None}")
ok("the_terminal_flag_reaches_the_TIMELINE",
   any(e.terminal for e in ag.timeline.entries),
   "deaths are ground truth, not a GoalModel private note")
ok("features_survive_the_level_boundary_in_the_real_agent",
   ag.goals._level_start >= 0 and ag.goals.timeline is ag.timeline,
   "one record, read by the goal model, never copied")

# ---- THE DECISIVE ASSERTION -------------------------------------------------
# With an empty pool the agent must be byte-identical to the agent without this
# component. Shipping it cannot regress anything, and that is a test, not a
# promise.
def run_sequence(disable_goals):
    if disable_goals:
        os.environ["ARC_NO_GOALS"] = "1"
    else:
        os.environ.pop("ARC_NO_GOALS", None)
    os.environ["ARC_AGENT_SEED"] = "0"
    agent = MyAgent(game_id="goalmodel-identity")
    out = []
    board = blank(8, 8)
    for i in range(60):
        board = board.copy()
        board[(i * 3) % 8, (i * 5) % 8] = 1 + (i % 4)
        f = Frame(board, [1, 2, 3, 4])
        out.append(str(agent.choose_action([f], f)))
    return out, agent


seq_off, _ = run_sequence(True)
seq_on, ag_on = run_sequence(False)
os.environ.pop("ARC_NO_GOALS", None)
ok("EMPTY_POOL_MEANS_A_BYTE_IDENTICAL_ACTION_SEQUENCE", seq_off == seq_on,
   f"{sum(1 for a, b in zip(seq_off, seq_on) if a != b)} of {len(seq_off)} differ")
# ... and say plainly what that run actually proved: the pool WAS inert, so this
# is the empty-pool identity claim and not a lucky agreement between two live
# rankers. A live ranker is allowed to change the sequence; that is its job.
ok("the_identity_run_really_had_an_inert_pool",
   ag_on.goals.frontier_ranker() is None,
   f"pool={len(ag_on.goals._feat)} features, none adoptable: {ag_on.goals.explain()}")

# ... and the reason it can be identical: the goal model never touches the shared
# RNG stream. Thompson sampling draws from a PRIVATE generator.
gm = new_model(seed=12)
ro = Readout("count", (4,))
f = Feature(ro, alpha=5, beta=2); f.direction, f.channel = 1, "A"
gm._feat[ro.key] = f
gm._range[ro.key] = [0.0, 4.0]
g4 = blank(6, 6); g4[0, :2] = 4
random.seed(99)
np.random.seed(99)
want = [random.random(), float(np.random.rand())]
random.seed(99)
np.random.seed(99)
for _ in range(50):
    gm.target(g4)
    gm.potential(g4)
    gm.rank_frontier([(1, "ACTION1"), (2, "ACTION2")], grid_of=lambda c: g4)
got = [random.random(), float(np.random.rand())]
check("the_goal_model_consumes_no_global_randomness", got, want)

# The frontier ranker stays None until something is actually adoptable, which is
# what makes the identity above hold in a real run rather than by luck.
ag2 = MyAgent(game_id="goalmodel-inert")
check("frontier_ranker_is_None_on_a_fresh_agent", ag2.goals.frontier_ranker(), None)


# ---------------------------------------------------------------------------
# CASE 10: an UNTESTED Channel-B hypothesis must not move the potential.
#
# Not a design preference -- a measured one. The first 25-game paired run
# (goals-off vs goals-on, identical code, seed 0) found A=0 and retired=0 in
# every one of the 25 games: `target()` has no call site yet, so `_settle_pursuit`
# never runs, so no posterior ever leaves Beta(1,1). Every feature steering the
# frontier was a structural guess at p=0.50, and the effect was a coin flip
# (tr87 -11.1 / wa30 -3.0 / m0r0 -2.7 re-spend against sp80 +10.5 / cd82 +7.1;
# dev mean -0.37pts). sp80 showed the hazard directly: no-ops -13.3 while
# re-spend +10.5 -- idling traded for re-treading, under a wrong coordinate.
# ---------------------------------------------------------------------------
print("\n=== CASE 10: unearned belief must not steer ===")

ro10 = Readout("count", (4,))
g10 = blank(6, 6); g10[0, :2] = 4


def steps10():
    """Six real transitions so `pin` has a record to size its value vectors to."""
    out, g = [], blank(6, 6)
    for i in range(6):
        nx = blank(6, 6)
        nx[0, :min(i + 1, 4)] = 4
        out.append((g, "ACTION1", nx, 0.0, False))
        g = nx
    return out


gm = new_model(avatar=1, seed=10)
drive(gm, steps10())
fB = pin(gm, ro10, direction=1, channel="B", alpha=1.0, beta=1.0)
ok("an_untested_B_feature_does_not_steer", not fB.steers(),
   f"alpha+beta={fB.alpha + fB.beta}")
check("untested_B_contributes_zero_potential", gm.potential(g10)[0], 0.0)
check("untested_B_leaves_the_ranker_inert", gm.frontier_ranker(), None)

# ...but it may still be PURSUED. That asymmetry is the whole point: pursuit is
# how a hypothesis earns evidence, the potential is where earned belief is spent.
# Gate both and Channel B deadlocks -- never pursued, so never updated, so never
# qualified. This assertion is what would catch that "cleanup".
#
# Selection is a Thompson draw against PURSUIT_FLOOR, so one call is a coin flip
# by design; over 20 draws from the private (seeded) RNG a uniform Beta(1,1) that
# never clears 0.35 has probability 0.35**20 ~ 3e-10. Failure here means the gate
# leaked into `target()`, not that the draw was unlucky.
pursued = any(gm.target(g10) is not None for _ in range(20))
ok("an_untested_B_feature_can_still_be_pursued", pursued,
   "target() must NOT apply the steers() gate")

# One settled pursuit is enough to earn a vote -- the gate is a threshold on
# evidence existing, not a high bar.
fB.alpha += 1.0
ok("B_steers_once_its_posterior_has_moved", fB.steers(),
   f"alpha+beta={fB.alpha + fB.beta}")
ok("earned_B_now_contributes_potential", gm.potential(g10)[0] != 0.0,
   f"phi={gm.potential(g10)[0]}")

# Channel A is exempt: `_confirmed` already paid for it with real score events
# plus the base-rate margin, so its evidence sits in the ADOPTION, not the Beta.
gm2 = new_model(avatar=1, seed=11)
drive(gm2, steps10())
fA = pin(gm2, ro10, direction=1, channel="A", alpha=1.0, beta=1.0)
ok("an_untested_A_feature_steers_immediately", fA.steers(),
   "confirmation by score event IS the evidence")
ok("untested_A_contributes_potential", gm2.potential(g10)[0] != 0.0,
   f"phi={gm2.potential(g10)[0]}")


# ---------------------------------------------------------------------------
# CASE 11: the ALPHA WARRANT -- refutation is free, adoption is expensive.
#
# Wiring `target()` to a call site (Section 9 step 6) is what makes the Beta
# layer non-inert, but the naive wiring has a fatal bug: arm a predicate every
# step, and any score event landing inside the budget window awards alpha to a
# hypothesis that did NOTHING. The reward came from whatever the agent was doing
# anyway. Under that rule every candidate rides the agent's own luck into the
# pool and the posteriors measure nothing.
#
# The fix is asymmetric on purpose (Section 5 guard 2 made literal):
#   alpha  needs `note_directed()` -- proof the model CHANGED a decision;
#   beta   needs nothing -- a predicate that came true and paid no score is
#          refuted no matter who made it true.
# So an armed-but-unused pursuit is a PURE REFUTATION ENGINE. It can only ever
# shrink the pool, which is why Half 1 of the wiring is safe by construction.
# ---------------------------------------------------------------------------
print("\n=== CASE 11: alpha needs a warrant, beta does not ===")

ro11 = Readout("count", (4,))


def arm(seed, alpha=20.0, beta=1.0, channel="B"):
    """A model with ONE near-certain feature and a pursuit on the hook.

    Armed through the real `target()` path rather than by assigning `_pursuit`:
    the tuple's shape (and the `directed=False` it must start at) is exactly what
    is under test here."""
    gm = new_model(avatar=1, seed=seed)
    drive(gm, steps10())
    f = pin(gm, ro11, direction=1, channel=channel, alpha=alpha, beta=beta)
    g = blank(6, 6); g[0, :2] = 4          # count(4) = 2 -> predicate is ">= 3"
    pred = None
    for _ in range(20):                     # Thompson draw; ~1.0 each try at a=20
        pred = gm.target(g)
        if pred is not None:
            break
    return gm, f, pred


def step(gm, nxt, reward):
    """One real transition through `observe` -- the only caller of
    `_settle_pursuit`. `drive` is unusable here: its trailing `certify()` would
    re-judge the pinned feature out of the pool and every assertion below would
    pass vacuously against an empty model."""
    prev = blank(6, 6)
    gm.timeline.append(prev, "ACTION1", nxt, reward, 0, False)
    gm.observe(prev, "ACTION1", nxt, reward, False, 0)


gm11, f11, pred11 = arm(21)
ok("target_arms_a_pursuit", pred11 is not None and gm11.pursuing(),
   f"pred={pred11}")
check("a_fresh_pursuit_starts_undirected", gm11._pursuit[3], False)

# (a) armed but never used + a score event -> NO alpha. The hypothesis explained
#     nothing, so it is neither confirmed nor blamed: the pursuit just closes.
a0 = f11.alpha
win = blank(6, 6); win[0, :4] = 4          # predicate holds AND score arrives
step(gm11, win, reward=1.0)
check("undirected_pursuit_earns_no_alpha", f11.alpha, a0)
check("undirected_pursuit_still_closes", gm11.pursuing(), False)

# (b) the same event, but the model actually redirected the agent -> alpha.
gm11b, f11b, _ = arm(22)
a0 = f11b.alpha
ok("note_directed_reports_success_while_pursuing", gm11b.note_directed() is True)
check("note_directed_sets_the_warrant", gm11b._pursuit[3], True)
step(gm11b, win, reward=1.0)
check("directed_pursuit_earns_alpha", f11b.alpha, a0 + 1.0)

# (c) refutation needs NO warrant. Predicate satisfied, no score -> beta, both
#     ways. This is the half that makes bare arming worth doing.
for tag, directed in (("undirected", False), ("directed", True)):
    gmx, fx, _ = arm(23)
    if directed:
        gmx.note_directed()
    b0 = fx.beta
    step(gmx, win, reward=0.0)
    check(f"{tag}_pursuit_is_refuted_without_score", fx.beta, b0 + 1.0)
    check(f"{tag}_refutation_closes_the_pursuit", gmx.pursuing(), False)

# (d) neither satisfied nor scored -> the pursuit stays open (its budget window
#     is the point: one step is not a verdict).
gm11c, f11c, _ = arm(24)
step(gm11c, blank(6, 6), reward=0.0)       # count(4)=0, predicate ">= 3" is false
check("an_unresolved_pursuit_stays_open", gm11c.pursuing(), True)

# (e) `note_directed` outside a pursuit is a no-op that reports failure, so a
#     consumer can never manufacture a warrant for a hypothesis that is not on
#     the hook.
gm11d = new_model(avatar=1, seed=25)
check("note_directed_is_False_with_nothing_armed", gm11d.note_directed(), False)

# (f) refutation ALONE can never switch the re-ranker on. A Channel-B feature
#     beaten down to Beta(1,2) sits at post_mean 1/3 = 0.333, below the 0.35
#     pursuit floor -- so `steering()` stays False and only a score-CONFIRMED
#     (Channel A) feature ever opens the consumer gate. That is the property
#     that bounds the risk of Half 2 of this wiring.
gm11e = new_model(avatar=1, seed=26)
drive(gm11e, steps10())
fref = pin(gm11e, ro11, direction=1, channel="B", alpha=1.0, beta=2.0)
ok("a_purely_refuted_feature_sits_below_the_pursuit_floor",
   fref.post_mean() < GoalModel.PURSUIT_FLOOR, f"post_mean={fref.post_mean():.3f}")
check("refutation_alone_does_not_enable_steering", gm11e.steering(), False)


# ---------------------------------------------------------------------------
# CASE 12: `_goal_rank` is a PERMUTATION, never a filter (Section 5 guard 1).
#
# This is the risky half: "the goal is the nearest distinctive object" is the
# decoy walk the architecture names as THE hazard, and `non_goal_colors` is the
# scar it already left. Potential-based shaping (Ng/Harada/Russell 1999) is what
# bounds the damage -- an arbitrarily wrong Phi can only PERMUTE an order, never
# remove a choice -- but only if the code really is a permutation. These cases
# hold that line, plus the three gates that keep it inert until it has earned
# the right to speak.
# ---------------------------------------------------------------------------
print("\n=== CASE 12: the target re-ranker is order-only ===")

from my_agent import ReactivePlanner


class StubGoals:
    """A GoalModel-shaped oracle with a scripted potential."""

    def __init__(self, phi, steer=True, boom=False):
        self.phi, self.steer, self.boom = phi, steer, boom
        self.directed = 0
        self.calls = 0

    def steering(self):
        if self.boom:
            raise RuntimeError("broken steering")
        return self.steer

    def potential(self, grid):
        self.calls += 1
        return (self.phi[self.calls - 1], 0.0)

    def note_directed(self):
        self.directed += 1
        return True


def planner(goals=None, avatar=1):
    p = ReactivePlanner()
    p.avatar_color = avatar
    p.goals = goals
    return p


g12 = blank(8, 8); g12[4, 4] = 1           # avatar at (4,4)
CANDS = [(False, 2, 3, (4, 6)), (False, 5, 3, (0, 0)), (True, 1, 40, (7, 7))]

# (a) same SET, same LENGTH, whatever Phi says -- over randomised pools.
rnd = random.Random(12)
worst = None
for trial in range(200):
    n = rnd.randint(2, 6)
    pool = [(rnd.random() < 0.3, rnd.randint(0, 20), rnd.randint(1, 50),
             (rnd.randint(0, 7), rnd.randint(0, 7))) for _ in range(n)]
    out = planner(StubGoals([rnd.uniform(-5, 5) for _ in range(n)]))._goal_rank(g12, pool)
    if len(out) != n or sorted(map(str, out)) != sorted(map(str, pool)):
        worst = (pool, out)
        break
ok("goal_rank_is_a_permutation_over_200_random_pools", worst is None, f"{worst}")

# (b) `structural` stays the FIRST key. A wall or HUD strip must never be
#     promoted above a discrete object no matter how much Phi likes it -- that
#     ordering was validated against real games and is not the goal model's to
#     overrule.
#     Phi adores the structural candidate and prefers the FARTHER object over the
#     nearer one, so both halves of the claim are under load: the wall stays last,
#     and distance really does yield to Phi inside the non-structural class.
sg = StubGoals([-9.0, 5.0, 99.0])
out12 = planner(sg)._goal_rank(g12, CANDS)
check("structural_candidates_are_never_promoted", out12[-1], CANDS[2])
ok("phi_reorders_only_within_a_structural_class", out12[0] is CANDS[1],
   f"head={out12[0]}")
check("changing_the_destination_claims_the_warrant", sg.directed, 1)

# A Phi TIE must fall back to the incumbent order, not to an arbitrary one: the
# sort key ends in `i` for exactly this reason. A model with no opinion between
# two candidates has to leave "nearest" -- the validated prior -- in charge.
sgt = StubGoals([-9.0, -9.0, 99.0])
outt = planner(sgt)._goal_rank(g12, CANDS)
check("a_phi_tie_keeps_the_validated_nearest_first_order", outt[0], CANDS[0])
check("a_phi_tie_claims_no_warrant", sgt.directed, 0)

# (c) Phi that agrees with the incumbent order must NOT claim a warrant: nothing
#     was redirected, so there is nothing to be credited for.
sg2 = StubGoals([5.0, 1.0, 0.0])
out12b = planner(sg2)._goal_rank(g12, CANDS)
check("agreeing_with_the_incumbent_keeps_the_order", out12b[0], CANDS[0])
check("agreeing_claims_no_warrant", sg2.directed, 0)

# (d) the three inert paths. Each returns the input list untouched, and (this is
#     the load-bearing part) `potential` is never even consulted.
sg3 = StubGoals([0.0] * 3, steer=False)
check("inert_when_no_feature_has_earned_steering",
      planner(sg3)._goal_rank(g12, CANDS), CANDS)
check("inert_gate_does_not_evaluate_potential", sg3.calls, 0)

os.environ["ARC_NO_GOALS"] = "1"
sg4 = StubGoals([9.0, 0.0, 0.0])
check("ARC_NO_GOALS_disables_the_re_ranker",
      planner(sg4)._goal_rank(g12, CANDS), CANDS)
check("ARC_NO_GOALS_does_not_evaluate_potential", sg4.calls, 0)
del os.environ["ARC_NO_GOALS"]

check("no_goal_model_means_no_re_ranking",
      planner(None)._goal_rank(g12, CANDS), CANDS)
check("a_single_candidate_is_never_re_ranked",
      planner(StubGoals([9.0]))._goal_rank(g12, CANDS[:1]), CANDS[:1])

# (e) a BROKEN goal model must cost the agent exactly nothing. This is the
#     failure mode that matters most in a 9h submission: an exception in a
#     re-ranker must degrade to the validated order, not to a crash.
check("a_raising_goal_model_leaves_the_order_untouched",
      planner(StubGoals([0.0] * 3, boom=True))._goal_rank(g12, CANDS), CANDS)


class HalfBroken(StubGoals):
    def potential(self, grid):
        raise ValueError("no potential")


check("a_raising_potential_leaves_the_order_untouched",
      planner(HalfBroken([0.0] * 3))._goal_rank(g12, CANDS), CANDS)

# (f) `_imagine_at` is TOTAL. It feeds `potential()` on grids the agent never
#     saw, so every degenerate input has to return an array rather than raise.
p12 = planner(StubGoals([0.0]))
ok("imagine_moves_the_avatar", p12._imagine_at(g12, (0, 0))[0, 0] == 1)
ok("imagine_clears_the_old_avatar_cell", p12._imagine_at(g12, (0, 0))[4, 4] == 0)
ok("imagine_preserves_shape", p12._imagine_at(g12, (0, 0)).shape == g12.shape)
ok("imagine_does_not_mutate_the_input", g12[4, 4] == 1 and g12[0, 0] == 0)
ok("imagine_is_a_no_op_when_the_avatar_is_absent",
   np.array_equal(planner(StubGoals([0.0]), avatar=7)._imagine_at(g12, (0, 0)), g12))

blk = blank(8, 8); blk[0:3, 0:3] = 1       # a 3x3 avatar body, not a single cell
for tgt in ((0, 0), (7, 7), (-3, -3), (99, 99)):
    try:
        r = p12._imagine_at(blk, tgt)
        ok(f"imagine_is_total_at_{tgt}", r.shape == blk.shape)
    except Exception as e:                  # noqa: BLE001 -- totality is the test
        ok(f"imagine_is_total_at_{tgt}", False, repr(e))

# (g) end to end through `_nearest_target` with a REAL GoalModel: the chosen cell
#     must still be one of the candidates. Phi picks WHICH object, never invents
#     a destination.
gm12 = new_model(avatar=1, seed=27)
drive(gm12, steps10())
pin(gm12, Readout("count", (4,)), direction=1, channel="A", alpha=5.0, beta=1.0)
ok("a_confirmed_A_feature_opens_the_consumer_gate", gm12.steering())
p12b = planner(gm12)
board = blank(8, 8); board[4, 4] = 1; board[1, 1] = 4; board[6, 6] = 5
tgt = p12b._nearest_target(board, 4, 4)
ok("nearest_target_returns_a_real_object_cell", tgt in ((1, 1), (6, 6)), f"{tgt}")


# ---------------------------------------------------------------------------
# CASE 13: the MyAgent wiring itself (Section 9 step 6).
#
# The link is one assignment, and the prior measurement is the reason it gets a
# test: `target()` had no call site, so A=0 and retired=0 in all 25 games and the
# entire Bayesian layer was dead code that LOOKED alive. A silently-severed
# reference reproduces exactly that null result, so the reference is asserted
# rather than read.
# ---------------------------------------------------------------------------
print("\n=== CASE 13: the consumer link survives the lifecycle ===")

ag13 = MyAgent(game_id="goalmodel-wiring")
ok("the_planner_sees_the_agents_goal_model", ag13.rplanner.goals is ag13.goals)

# `reset()` re-runs `__init__` via the parent, which would sever a MyAgent-owned
# reference on every GAME_OVER -- the planner would then silently stop consuming
# the model partway through a run. Same hazard the timeline reference already
# has an explicit save/restore for.
ag13.rplanner.reset()
ok("the_link_survives_a_planner_reset", ag13.rplanner.goals is ag13.goals)
ag13.rplanner.new_level()
ok("the_link_survives_a_new_level", ag13.rplanner.goals is ag13.goals)

# And a fresh agent stays inert: no feature has earned anything yet, so the
# re-ranker must not be steering on step 0.
check("a_fresh_agent_is_not_steering", ag13.goals.steering(), False)
check("a_fresh_agent_has_nothing_on_the_hook", ag13.goals.pursuing(), False)


print("\n================ SUMMARY ================")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
sys.exit(1 if FAILS else 0)
