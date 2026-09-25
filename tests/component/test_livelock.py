# --- path shim: these suites live in tests/ but import the agent from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_ROOT, _os.path.join(_ROOT, 'eval'), _os.path.join(_ROOT, 'sarbloh', 'legacy')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""
Isolation test harness for COMPONENT 5.11: LIVELOCK BREAKER
(+ the DeadActionTracker click granularity and ClickPlanner.fresh_cell that the
escalation ladder stands on).

Doctrine: the components STAY in my_agent.py. This file only imports them and
feeds hand-built inputs whose correct answer we know by construction, so a
failure is visible for the breaker alone -- no game engine, no LLM.

The end-to-end section builds a STUCK WORLD (an environment whose frame never
changes, whatever you do) and drives the real MyAgent.choose_action against it.
That is the whole point of the component: on such a world the agent must stop
re-spending actions it has already proven inert, and must escalate rather than
hammer. The control run with ARC_NO_BREAKER=1 proves the new code is what does
it -- without the control, "the agent behaves" says nothing about why.

Bar is NOT "perfect", it is "passes this fixed set". Never delete a case.
Run:  python test_livelock.py
"""
import os
import sys

import numpy as np

from my_agent import (LivelockBreaker, DeadActionTracker, ClickPlanner, MyAgent,
                      GameState, GameAction, base_action,
                      NOOP_TRIP, NOOP_HARD, SPEND_REPEAT_CAP, MAX_BREAK_RESETS,
                      NOVEL_PATIENCE, ESCALATE_BFS_EVERY)

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


# ======================================================================
print("== LivelockBreaker: the trip conditions ==")
# ======================================================================
b = LivelockBreaker()
check("fresh_not_tripped", b.tripped(), False)
check("fresh_level_actions", b.level_actions, 0)

# NOOP_TRIP consecutive inert actions is exactly the trip point -- not one before.
b = LivelockBreaker()
for _ in range(NOOP_TRIP - 1):
    b.update(1, "ACTION1", changed=False)
check("one_short_of_trip", b.tripped(), False)
b.update(1, "ACTION1", changed=False)
check("at_trip", b.tripped(), True)
check("trip_run_len", b.run, NOOP_TRIP)

# A single real change clears the streak: the layer IS learning again.
b.update(1, "ACTION1", changed=True)
check("change_clears_run", b.run, 0)
check("change_clears_trip", b.tripped(), False)
check("change_counted", b.level_changes, 1)

# The HARD invariant is independent of the streak. An ENGINE-forced restart
# (GAME_OVER) clears the run -- the frame is replaced wholesale -- but must NOT
# buy the level another 30 free actions: the agent did not choose it and has
# learned nothing from it.
b = LivelockBreaker()
for _ in range(NOOP_HARD - 5):
    b.update(1, "ACTION1", changed=False)
b.note_reset()
check("reset_clears_run", b.run, 0)
for _ in range(4):
    b.update(1, "ACTION1", changed=False)
check("hard_not_yet", b.tripped(), False)      # run=4 < TRIP, actions=29 < HARD
check("hard_run_below_trip", b.run < NOOP_TRIP, True)
b.update(1, "ACTION1", changed=False)
check("hard_invariant_trips", b.tripped(), True)   # actions==HARD, zero changes
check("hard_still_below_trip_run", b.run < NOOP_TRIP, True)

# A BREAKER-chosen restart is the opposite case: it was spent precisely in order
# to land on a different frame, so the invariant gets a fresh window to observe
# the result. Without this the hard clause re-trips on the very next action and
# the entire reset budget burns in consecutive steps -- measured, and the reason
# the window exists at all.
b = LivelockBreaker()
for _ in range(NOOP_HARD):
    b.update(1, "ACTION1", changed=False)
check("breaker_reset_precondition", b.tripped(), True)
b.note_reset(by_breaker=True)
check("breaker_reset_opens_window", b.tripped(), False)
for _ in range(NOOP_TRIP - 1):
    b.update(1, "ACTION1", changed=False)
check("window_not_tripped_early", b.tripped(), False)
for _ in range(NOOP_HARD - NOOP_TRIP + 1):
    b.update(1, "ACTION1", changed=False)
check("window_trips_again", b.tripped(), True)
ok("level_actions_keep_accumulating", b.level_actions >= NOOP_HARD * 2,
   f"level_actions={b.level_actions} (the RESET budget gate reads this, not the window)")

# A level that HAS moved is judged only by the streak, never by the hard clause.
b = LivelockBreaker()
b.update(1, "ACTION1", changed=True)
for _ in range(NOOP_HARD * 2):
    b.update(1, "ACTION2", changed=False)
ok("moved_level_uses_streak", b.tripped(), "long inert run still trips")
b.update(1, "ACTION2", changed=True)
check("moved_level_recovers", b.tripped(), False)

print("\n== LivelockBreaker: what is NOT a scored action ==")
b = LivelockBreaker()
b.update(1, "RESET", changed=False)
b.update(1, "", changed=False)
b.update(1, None, changed=False)
check("reset_not_counted", b.level_actions, 0)
check("reset_not_streaked", b.run, 0)
check("reset_not_ledgered", b.spent(1, "RESET"), 0)
check("reset_never_stale", b.stale(1, "RESET"), False)

print("\n== LivelockBreaker: the spend ledger ==")
b = LivelockBreaker()
check("unspent_is_zero", b.spent(7, "ACTION1"), 0)
check("unspent_not_stale", b.stale(7, "ACTION1"), False)
for i in range(SPEND_REPEAT_CAP - 1):
    b.update(7, "ACTION1", changed=False)
check("below_cap_not_stale", b.stale(7, "ACTION1"), False)
b.update(7, "ACTION1", changed=False)
check("at_cap_is_stale", b.stale(7, "ACTION1"), True)
check("spend_count", b.spent(7, "ACTION1"), SPEND_REPEAT_CAP)

# Same action, DIFFERENT state: a separate ledger entry. The pair is the unit --
# an action that is inert here may be the only thing that works elsewhere.
check("other_state_not_stale", b.stale(8, "ACTION1"), False)
# Same state, different action.
check("other_action_not_stale", b.stale(7, "ACTION2"), False)

# A pair that has EVER changed the world is never stale, however often re-spent.
b = LivelockBreaker()
b.update(9, "ACTION3", changed=True)
for _ in range(SPEND_REPEAT_CAP * 5):
    b.update(9, "ACTION3", changed=False)
check("productive_pair_never_stale", b.stale(9, "ACTION3"), False)
ok("productive_pair_counted", b.spent(9, "ACTION3") == SPEND_REPEAT_CAP * 5 + 1,
   f"spent={b.spent(9, 'ACTION3')}")

# Clicks are ledgered at FULL coordinate granularity: two different pixels are
# two different actions, which is the whole reason a click game can livelock
# while every base-action check says "ACTION6 still works".
b = LivelockBreaker()
for _ in range(SPEND_REPEAT_CAP):
    b.update(1, "ACTION6_r4_c4", changed=False)
check("click_cell_stale", b.stale(1, "ACTION6_r4_c4"), True)
check("click_other_cell_fresh", b.stale(1, "ACTION6_r4_c8"), False)

# No state hash (the grid could not be hashed) degrades to "no opinion" rather
# than to a wrong one.
b = LivelockBreaker()
for _ in range(SPEND_REPEAT_CAP * 3):
    b.update(None, "ACTION1", changed=False)
check("no_hash_not_stale", b.stale(None, "ACTION1"), False)
check("no_hash_spent_zero", b.spent(None, "ACTION1"), 0)
ok("no_hash_still_streaks", b.run == SPEND_REPEAT_CAP * 3,
   "the streak needs no hash, only the ledger does")

print("\n== LivelockBreaker: level / reset lifecycle ==")
b = LivelockBreaker()
for _ in range(SPEND_REPEAT_CAP):
    b.update(1, "ACTION1", changed=False)
b.note_reset()
ok("ledger_survives_reset", b.stale(1, "ACTION1"),
   "same layout after an intra-level restart -> an inert pair is still inert")
b.new_level()
check("ledger_cleared_on_level", b.stale(1, "ACTION1"), False)
check("level_actions_cleared", b.level_actions, 0)
check("resets_cleared", b.resets, 0)

print("\n== LivelockBreaker: the RESET rung is budgeted ==")
b = LivelockBreaker()
check("may_not_reset_young_level", b.may_reset(), False)
for _ in range(NOOP_HARD):
    b.update(1, "ACTION1", changed=False)
check("may_reset_old_level", b.may_reset(), True)
# Only breaker-chosen restarts consume the budget.
for _ in range(MAX_BREAK_RESETS * 2):
    b.note_reset(by_breaker=False)
check("engine_resets_are_free", b.may_reset(), True)
check("engine_resets_uncounted", b.resets, 0)

# Each breaker restart must be EARNED again: the window guard stops the rung
# firing repeatedly through the `stale` path while the level age stands still.
b2 = LivelockBreaker()
for _ in range(NOOP_HARD):
    b2.update(1, "ACTION1", changed=False)
check("first_reset_allowed", b2.may_reset(), True)
b2.note_reset(by_breaker=True)
check("second_reset_needs_new_window", b2.may_reset(), False)
for _ in range(NOOP_HARD - 1):
    b2.update(1, "ACTION1", changed=False)
check("window_still_short", b2.may_reset(), False)
b2.update(1, "ACTION1", changed=False)
check("second_reset_earned", b2.may_reset(), True)

for _ in range(MAX_BREAK_RESETS):
    b.note_reset(by_breaker=True)
check("reset_budget_exhausted", b.may_reset(), False)
check("reset_budget_counted", b.resets, MAX_BREAK_RESETS)


# ======================================================================
print("\n== DeadActionTracker: keys behave exactly as before ==")
# ======================================================================
d = DeadActionTracker()
for _ in range(d.MIN_TRIALS - 1):
    d.update("ACTION1", changed=False)
check("key_alive_below_trials", d.is_dead("ACTION1"), False)
d.update("ACTION1", changed=False)
check("key_dead_at_trials", d.is_dead("ACTION1"), True)
d.update("ACTION1", changed=True)
check("one_change_whitelists", d.is_dead("ACTION1"), False)

print("\n== DeadActionTracker: clicks are per CELL ==")
d = DeadActionTracker()
for _ in range(d.CELL_MIN_TRIALS):
    d.update("ACTION6_r1_c1", changed=False)
check("cell_dead", d.cell_is_dead("ACTION6_r1_c1"), True)
check("other_cell_alive", d.cell_is_dead("ACTION6_r2_c2"), False)
# REGRESSION GUARD: callers filter `valid` with is_dead, and `valid` carries the
# bare action name. If a dead CELL ever made the bare ACTION6 read dead, a
# click-only game would have its only modality filtered away entirely.
check("bare_action6_never_dead", d.is_dead("ACTION6"), False)
check("dead_cell_bare_still_alive", d.is_dead("ACTION6_r1_c1"), False)
# A cell that changed something is never dead.
d.update("ACTION6_r3_c3", changed=True)
for _ in range(d.CELL_MIN_TRIALS * 3):
    d.update("ACTION6_r3_c3", changed=False)
check("reactive_cell_never_dead", d.cell_is_dead("ACTION6_r3_c3"), False)

print("\n== DeadActionTracker: click-surface exhaustion ==")
d = DeadActionTracker()
check("no_clicks_not_exhausted", d.clicks_exhausted(), False)
d.update("ACTION6_r1_c1", changed=False)
check("undersampled_not_exhausted", d.clicks_exhausted(), False)
for _ in range(d.CELL_MIN_TRIALS):
    d.update("ACTION6_r1_c1", changed=False)
check("all_dead_is_exhausted", d.clicks_exhausted(), True)
d.update("ACTION6_r9_c9", changed=True)
check("one_live_cell_unexhausts", d.clicks_exhausted(), False)
# Key actions must not leak into the click verdict.
d2 = DeadActionTracker()
for _ in range(d2.MIN_TRIALS):
    d2.update("ACTION1", changed=False)
check("keys_dont_exhaust_clicks", d2.clicks_exhausted(), False)


# ======================================================================
print("\n== ClickPlanner.fresh_cell: coverage refines, never repeats ==")
# ======================================================================
def _click(cp, action_str, changed=False):
    cp.update(action_str, changed=changed, reward=0.0)


cp = ClickPlanner()
grid = np.zeros((16, 16), dtype=np.int32)
first = cp.fresh_cell(grid)
ok("fresh_returns_click", first is not None and first.startswith("ACTION6_r"),
   str(first))
# The first rung IS the ordinary QUANT lattice, so the breaker and the coverage
# policy agree about what "unexplored" means until the lattice runs out.
lat = set(cp._lattice(16, 16))
r, c = int(first.split("_")[1][1:]), int(first.split("_")[2][1:])
check("fresh_starts_on_lattice", (r, c) in lat, True)

# Exhaust the whole QUANT lattice; fresh_cell must then REFINE rather than repeat.
for (rr, cc) in sorted(lat):
    _click(cp, f"ACTION6_r{rr}_c{cc}")
nxt = cp.fresh_cell(grid)
ok("refines_past_lattice", nxt is not None, str(nxt))
r2, c2 = int(nxt.split("_")[1][1:]), int(nxt.split("_")[2][1:])
check("refined_is_off_lattice", (r2, c2) in lat, False)
check("refined_is_unclicked", (r2, c2) in cp._exact, False)

# 300 further draws must every one of them be a pixel never clicked before --
# this is the exact property the measured pathology violated (~70 cells absorbing
# ~1300 clicks).
seen_repeat = None
for _ in range(300):
    a = cp.fresh_cell(grid)
    if a is None:
        break
    rr, cc = int(a.split("_")[1][1:]), int(a.split("_")[2][1:])
    if (rr, cc) in cp._exact:
        seen_repeat = (rr, cc)
        break
    _click(cp, a)
check("never_repeats_a_pixel", seen_repeat, None)

# Lethal cells are skipped: new information is not worth dying for twice.
cp2 = ClickPlanner()
small = np.zeros((8, 8), dtype=np.int32)
for (rr, cc) in cp2._lattice(8, 8):
    cp2.mark_loss(f"ACTION6_r{rr}_c{cc}")
a = cp2.fresh_cell(small)
if a is not None:
    rr, cc = int(a.split("_")[1][1:]), int(a.split("_")[2][1:])
    check("skips_lethal_cells", cp2.is_lethal(rr, cc), False)
else:
    ok("skips_lethal_cells", True, "whole board lethal -> None")

# True exhaustion: every pixel clicked -> None (the caller must change modality).
cp3 = ClickPlanner()
tiny = np.zeros((4, 4), dtype=np.int32)
for rr in range(4):
    for cc in range(4):
        _click(cp3, f"ACTION6_r{rr}_c{cc}")
check("exhausted_returns_none", cp3.fresh_cell(tiny), None)

# Totality: fresh_cell never raises on junk input.
check("fresh_none_grid", cp3.fresh_cell(None), None)
check("fresh_1d_grid", cp3.fresh_cell(np.zeros(5, dtype=np.int32)), None)
check("fresh_empty_grid", ClickPlanner().fresh_cell(np.zeros((0, 0), dtype=np.int32)), None)

# new_level wipes the exact-pixel memory with the stats it belongs to.
cp4 = ClickPlanner()
_click(cp4, "ACTION6_r2_c2")
ok("exact_recorded", (2, 2) in cp4._exact, str(sorted(cp4._exact)))
cp4.new_level()
check("exact_cleared_on_level", (2, 2) in cp4._exact, False)
cp5 = ClickPlanner()
_click(cp5, "ACTION6_r2_c2")
cp5.reset()
ok("exact_survives_intra_level_reset", (2, 2) in cp5._exact,
   "same layout after GAME_OVER -> the pixel is still known")


# ======================================================================
print("\n== END-TO-END: a stuck world (every action is a masked no-op) ==")
# ======================================================================
# MyAgent seeds its exploration RNG from WALL TIME unless ARC_AGENT_SEED is set,
# which made this section flicker: the same code passed and failed on successive
# runs. Pin it -- a flaky case cannot back a claim about a fix.
os.environ["ARC_AGENT_SEED"] = "0"


class StuckFrame:
    """A frame that NEVER changes, whatever action is taken."""
    def __init__(self, g, actions, lvl=0, st=None):
        self.frame = g.tolist()
        self.available_actions = list(actions)
        self.levels_completed = lvl
        self.state = st if st is not None else GameState.NOT_FINISHED


def drive(agent, grid, actions, n):
    """Run n real choose_action steps against a world that never moves.
    Returns the list of chosen action strings."""
    out = []
    for _ in range(n):
        f = StuckFrame(grid, actions)
        agent.choose_action([f], f)
        out.append(agent.last_action_str)
    return out


def worst_repeat(seq):
    """How often the single most-repeated action was chosen."""
    counts = {}
    for a in seq:
        counts[a] = counts.get(a, 0) + 1
    return max(counts.values()) if counts else 0


def wasted(agent):
    """Scored actions spent on (state, action) pairs ALREADY proven inert.

    The RHAE currency: every one of these is an action that could not possibly
    have taught the agent anything, because the games are deterministic and the
    pair's outcome was already on record. Comparable between runs -- the ledger
    is kept even when the breaker is disabled, so only the ACTING differs."""
    return sum(max(0, n - SPEND_REPEAT_CAP)
               for (n, ch) in agent.breaker._spend.values() if not ch)


N = 120
board = np.zeros((16, 16), dtype=np.int32)
board[4, 4] = 2
board[9, 11] = 3

# --- movement-style stuck world: keys only ------------------------------
os.environ.pop("ARC_NO_BREAKER", None)
ag = MyAgent(game_id="livelock-keys")
keys = [1, 2, 3, 4]
seq = drive(ag, board, keys, N)
ok("keys_breaker_tripped", ag.breaker.escalations > 0,
   f"escalations={ag.breaker.escalations}")
ok("keys_emits_reset", "RESET" in seq,
   "a fixed point under every key must end in a level restart, not more keys")
# The restarts must be SPACED, not spent in consecutive steps. The step after a
# RESET is a replay and is not counted, so without the window guard the level age
# never advances and the whole budget goes in a handful of actions.
_reset_at = [i for i, a in enumerate(seq) if a == "RESET"]
ok("keys_resets_are_spaced",
   all(b_ - a_ >= NOOP_HARD for a_, b_ in zip(_reset_at, _reset_at[1:])),
   f"RESET at steps {_reset_at}")
ok("keys_reset_budget_respected", len(_reset_at) <= MAX_BREAK_RESETS,
   f"{len(_reset_at)} restarts in {N} actions")
# The invariant the goal states: no single proven-dead action may be repeated
# past a threshold. Four keys over 120 steps would be ~30 each if it just cycled.
ok("keys_no_hammering", worst_repeat([a for a in seq if a != "RESET"]) <= N // 3,
   f"worst repeat = {worst_repeat([a for a in seq if a != 'RESET'])} of {N}")

os.environ["ARC_NO_BREAKER"] = "1"
ag_ctl = MyAgent(game_id="livelock-keys-ctl")
seq_ctl = drive(ag_ctl, board, keys, N)
os.environ.pop("ARC_NO_BREAKER", None)
check("control_never_escalates", ag_ctl.breaker.escalations, 0)
check("control_never_resets", "RESET" in seq_ctl, False)
# NOTE on what this world can and cannot show: with only four keys and a frame
# that never moves, EVERY action is a re-spend after the first 4*CAP, so the
# breaker cannot beat the control on repeat count -- both cycle the four keys
# evenly (the graph explorer already balances by global usage). The difference
# the breaker makes here is categorical, not quantitative: it notices, and it
# escalates to a restart. The quantitative win lives in the click world below,
# where the action space is large enough for escalation to have somewhere to go.
ok("keys_wasted_not_worse", wasted(ag) <= wasted(ag_ctl),
   f"breaker wasted {wasted(ag)} vs control {wasted(ag_ctl)} inert re-spends")

# --- click-only stuck world ---------------------------------------------
ag2 = MyAgent(game_id="livelock-clicks")
seq2 = drive(ag2, board, [6], N)
clicks2 = [a for a in seq2 if a.startswith("ACTION6")]
ok("clicks_all_distinct", len(set(clicks2)) == len(clicks2),
   f"{len(set(clicks2))} distinct of {len(clicks2)} clicks")
ok("clicks_breaker_active", ag2.breaker.escalations > 0,
   f"escalations={ag2.breaker.escalations}")

os.environ["ARC_NO_BREAKER"] = "1"
ag2_ctl = MyAgent(game_id="livelock-clicks-ctl")
seq2_ctl = drive(ag2_ctl, board, [6], N)
os.environ.pop("ARC_NO_BREAKER", None)
clicks2_ctl = [a for a in seq2_ctl if a.startswith("ACTION6")]
ok("control_clicks_repeat", len(set(clicks2_ctl)) < len(clicks2_ctl),
   f"control reaches {len(set(clicks2_ctl))} distinct of {len(clicks2_ctl)}")
ok("clicks_wasted_drops", wasted(ag2) < wasted(ag2_ctl),
   f"breaker wasted {wasted(ag2)} vs control {wasted(ag2_ctl)} inert re-spends")

# --- a world that DOES move must be left alone ---------------------------
# The breaker is a circuit breaker, not a policy: while the layer is learning it
# must not fire at all, or it would displace a planner that is working.
class RespondingWorld:
    """A tiny real movement game: the avatar moves in the direction pressed.

    Deliberately TWO-DIMENSIONAL. The first version of this fixture walked one
    pixel along a single row on every action, which the StateGraph's periodicity
    mask correctly classified as a clock (row 4 changed with a modal gap of 1,
    every transition) -- so `changed_masked` reported no world change and the
    breaker tripped on a world that was visibly moving. The fixture was wrong,
    not the breaker, but see `clock_periodic_mover_reads_as_hud` below: the
    limitation is real and is recorded rather than papered over.
    """
    DIRS = {"ACTION1": (-1, 0), "ACTION2": (1, 0),
            "ACTION3": (0, -1), "ACTION4": (0, 1)}

    def __init__(self, n=16):
        self.n = n
        self.pos = [n // 2, n // 2]

    def step(self, action):
        dr, dc = self.DIRS.get(str(action), (0, 0))
        self.pos[0] = max(0, min(self.n - 1, self.pos[0] + dr))
        self.pos[1] = max(0, min(self.n - 1, self.pos[1] + dc))

    def grid(self):
        g = np.zeros((self.n, self.n), dtype=np.int32)
        g[self.pos[0], self.pos[1]] = 2
        return g


ag3 = MyAgent(game_id="livelock-moving")
world = RespondingWorld()
for i in range(60):
    f = StuckFrame(world.grid(), [1, 2, 3, 4])
    ag3.choose_action([f], f)
    world.step(ag3.last_action_str)
ok("moving_world_counted_changes", ag3.breaker.level_changes >= 30,
   f"level_changes={ag3.breaker.level_changes} of 60 "
   f"(edge presses against the wall are genuine no-ops)")
# The breaker must stay OUT OF THE WAY of a working layer -- but "out of the way"
# is not "silent". An avatar shoved into a wall produces a genuine run of inert
# actions, and breaking that IS the job: the escalation picks a different key.
# The invariant is therefore rarity, not absence. (This case originally asserted
# absence and flickered run to run, which is what exposed the unpinned seed.)
ok("moving_world_escalation_is_rare", ag3.breaker.escalations <= 6,
   f"escalations={ag3.breaker.escalations} in 60 productive actions")
ok("moving_world_never_restarts", ag3.breaker.resets == 0,
   "a level that is moving must never be thrown away")
ok("moving_world_no_hard_trip", ag3.breaker.win_changes > 0,
   f"win_changes={ag3.breaker.win_changes}: the hard invariant never applies here")

# KNOWN LIMITATION, recorded permanently. A mover that advances on a perfectly
# regular clock ALONG ONE ROW is, to the HUD mask, indistinguishable from a
# timer: MASK_LINE_PERIODIC sees a modal gap covering 100% of gaps. The agent
# then reads a moving world as static. Real games move in 2-D and only when the
# matching action is pressed, so gaps are irregular -- but if this ever becomes
# a FAIL rather than a WARN, the mask has changed and the breaker's premise with
# it. Do not delete: this case is why the fixture above is two-dimensional.
from my_agent import StateGraph as _SG                     # noqa: E402
_sg = _SG()
_prev = None
for i in range(60):
    _g = np.zeros((16, 16), dtype=np.int32)
    _g[4, (4 + i) % 16] = 2
    if _prev is not None:
        _sg.record(_prev, "ACTION1", _g)
        _sg.ensure_state(_g, ["ACTION1", "ACTION2"])
    _prev = _g
_m = _sg.hud_mask()
_masked_row4 = _m is not None and bool(_m[4].all())
warn("clock_periodic_mover_reads_as_hud", _masked_row4,
     f"row-4 mover masked={_masked_row4} "
     f"(expected True today; a 1-D clock-periodic mover is read as a timer)")
_a = np.zeros((16, 16), dtype=np.int32); _a[4, 1] = 2
_b = np.zeros((16, 16), dtype=np.int32); _b[4, 2] = 2      # the mover advanced
warn("clock_periodic_mover_change_hidden", not _sg.changed_masked(_a, _b),
     "consequence: a real move inside the masked row reports as no change")
# ... and the same motion OUTSIDE the masked row is still seen, so the mask is
# narrow enough to be a row verdict rather than a blanket one.
_c = np.zeros((16, 16), dtype=np.int32); _c[9, 1] = 2
_d = np.zeros((16, 16), dtype=np.int32); _d[9, 2] = 2
ok("unmasked_row_change_still_seen", _sg.changed_masked(_c, _d),
   "motion outside the clock row is unaffected")

# --- the breaker must not interrupt a protected sequence ------------------
# A ClickPlanner search replays "RESET + path + button" and hashes the result to
# record an edge; a foreign action mid-replay records a FALSE transition.
ag4 = MyAgent(game_id="livelock-protected")
ag4.cplanner._buttons = [(2, 2), (6, 6)]
ag4.cplanner._enter_search(ag4.cplanner._buttons)
ok("protected_while_searching", ag4._protected_sequence(),
   "search in flight")
from collections import deque as _deque
ag5 = MyAgent(game_id="livelock-protected-plan")
ag5._click_plan = _deque([(1, 1)])
ok("protected_while_planning", ag5._protected_sequence(), "LLM click plan queued")
ag5._discard_click_plan()
check("unprotected_when_idle", ag5._protected_sequence(), False)

# --- coverage exhausted: prefer a restart over grinding pixels ------------
# When every cell ever sampled is proven inert, the surface is very likely
# globally inert. Probing the remaining ~4000 pixels one scored action at a time
# is a worse experiment than restarting the level -- but ONLY while a restart is
# actually available, because with the budget spent, refining is still strictly
# better than repeating.
def _stuck_click_agent(name):
    a = MyAgent(game_id=name)
    for i in range(NOOP_HARD):                      # age the level, all inert
        a.breaker.update(1, f"ACTION6_r{i}_c0", changed=False)
    for i in range(NOOP_HARD):                      # ... and prove those cells dead
        for _ in range(a.dead.CELL_MIN_TRIALS):
            a.dead.update(f"ACTION6_r{i}_c0", changed=False)
    return a


ag7 = _stuck_click_agent("livelock-exhausted")
ok("exhausted_precondition", ag7.dead.clicks_exhausted() and ag7.breaker.may_reset(),
   "surface proven inert and a restart is affordable")
check("exhausted_prefers_reset",
      ag7._escalate(board, ["ACTION6"], h=1), "RESET")

ag8 = _stuck_click_agent("livelock-exhausted-nobudget")
for _ in range(MAX_BREAK_RESETS):
    ag8.breaker.note_reset(by_breaker=True)
check("no_budget_may_reset", ag8.breaker.may_reset(), False)
_alt = ag8._escalate(board, ["ACTION6"], h=1)
ok("no_budget_falls_back_to_refining",
   _alt is not None and _alt.startswith("ACTION6_r"),
   f"{_alt} -- refining still beats repeating once restarts are gone")

# A live click cell means the surface is NOT exhausted: keep probing pixels.
ag9 = _stuck_click_agent("livelock-live-cell")
ag9.dead.update("ACTION6_r0_c9", changed=True)
check("live_cell_unexhausts", ag9.dead.clicks_exhausted(), False)
_alt9 = ag9._escalate(board, ["ACTION6"], h=1)
ok("live_surface_keeps_clicking",
   _alt9 is not None and _alt9.startswith("ACTION6_r"), str(_alt9))

# --- RESET must never be the first action of a level ---------------------
# The engine reads a RESET taken while its internal action_count is 0 as a FULL
# reset, which wipes levels_completed -- so the rung is gated on level age.
ag6 = MyAgent(game_id="livelock-young")
seq6 = drive(ag6, board, [1, 2, 3, 4], NOOP_HARD - 1)
check("no_reset_before_hard", "RESET" in seq6, False)

# ======================================================================
print("\n== NOVELTY GATE: `stale` may not outrank a layer that is still learning ==")
# ======================================================================
# WHY THIS SECTION EXISTS. The first shipped version of the post-filter treated
# `tripped` (the world is frozen) and `stale` (this one pair was re-spent) as
# equally strong evidence. Measured on the 25-game diagnostic, that cost a game
# its only level-up: a planner working a frontier re-enters known states
# constantly, and the post-filter kept overriding it. The repair is to require
# that the agent has ALSO stopped reaching new states before the weak evidence is
# allowed to take control. These cases pin that distinction.

b = LivelockBreaker()
check("fresh_level_is_learning", b.learning(), True)
for i in range(NOVEL_PATIENCE - 1):
    b.update(1, "ACTION1", changed=True)
ok("still_learning_just_under_patience", b.learning(),
   f"since_novel={b.since_novel} < {NOVEL_PATIENCE}")
b.update(1, "ACTION1", changed=True)
ok("patience_expires_on_schedule", not b.learning(),
   f"since_novel={b.since_novel} >= {NOVEL_PATIENCE}")
b.note_novel()
check("novel_state_restores_learning", b.learning(), True)
# A world that CHANGES every action but cycles the same states is exactly the
# pathology `tripped` cannot see -- novelty is the signal that catches it.
for _ in range(NOVEL_PATIENCE + 5):
    b.update(1, "ACTION1", changed=True)
ok("changing_but_not_novel_stops_learning", not b.learning(),
   "frame changed every action, no new state -> not learning")
check("changing_world_never_trips", b.tripped(), False)
# ... and the strong evidence is NOT gated: a frozen world trips regardless.
b2 = LivelockBreaker()
for _ in range(NOOP_TRIP):
    b2.update(1, "ACTION1", changed=False)
ok("frozen_world_trips_while_learning", b2.tripped() and b2.learning(),
   f"tripped={b2.tripped()} learning={b2.learning()} -- `tripped` is ungated "
   "by design: a frozen world cannot be teaching anyone anything")
# A level-up clears the novelty clock along with everything else.
b2.new_level()
check("new_level_resets_novelty", b2.since_novel, 0)

# End-to-end: the agent must actually WIRE novelty to the state graph growing.
ag10 = MyAgent(game_id="livelock-novelty")
w10 = RespondingWorld()
for _ in range(40):
    f = StuckFrame(w10.grid(), [1, 2, 3, 4])
    ag10.choose_action([f], f)
    w10.step(ag10.last_action_str)
ok("moving_world_stays_in_learning", ag10.breaker.learning(),
   f"since_novel={ag10.breaker.since_novel} after 40 productive actions")
ag11 = MyAgent(game_id="livelock-novelty-stuck")
drive(ag11, board, [1, 2, 3, 4], NOVEL_PATIENCE + NOOP_HARD)
ok("stuck_world_leaves_learning", not ag11.breaker.learning(),
   f"since_novel={ag11.breaker.since_novel} on a world that never moves")

# The gate must be readable off the post-filter itself, not just the breaker: a
# stale pair on a LEARNING agent must not be escalated, the same pair once
# learning has lapsed must be.
ag12 = MyAgent(game_id="livelock-gate")
_H = 12345
for _ in range(SPEND_REPEAT_CAP):
    ag12.breaker.update(_H, "ACTION1", changed=False)
ok("gate_pair_is_stale", ag12.breaker.stale(_H, "ACTION1"), "precondition")
ag12.breaker.note_novel()
ok("stale_suppressed_while_learning",
   ag12.breaker.stale(_H, "ACTION1") and ag12.breaker.learning(),
   "stale is true but learning() vetoes it at the call site")
ag12.breaker.since_novel = NOVEL_PATIENCE
ok("stale_fires_once_learning_lapses",
   ag12.breaker.stale(_H, "ACTION1") and not ag12.breaker.learning(),
   "both halves of the call-site condition now hold")

# ======================================================================
print("\n== BFS COST CAP: the frontier search may not run every step ==")
# ======================================================================
# Rung 1 of the ladder is a BFS over the whole state graph. Once escalation was
# firing it ran on EVERY step -- measured at ~7x the per-action cost on a game
# with a large graph, which on a fixed wall clock loses more actions than the
# rung wins. Two guards: a per-step memo shared with the routing ladder, and a
# throttle between escalation-initiated searches.

ag13 = MyAgent(game_id="livelock-bfs-memo")
_bfs_calls = [0]
_real_nug = ag13.sgraph.nearest_untested_grid


def _counting_nug(*a, **k):
    _bfs_calls[0] += 1
    return _real_nug(*a, **k)


ag13.sgraph.nearest_untested_grid = _counting_nug
ag13.steps_total = 100
ag13._graph_explore(board, ["ACTION1"])
_first = _bfs_calls[0]
ag13._graph_explore(board, ["ACTION1"])
check("memo_suppresses_second_search_same_step", _bfs_calls[0], _first)
ag13.steps_total = 101
ag13._graph_explore(board, ["ACTION1"])
ok("memo_expires_next_step", _bfs_calls[0] > _first,
   f"{_bfs_calls[0]} searches after stepping the clock")
# The memo caches the FRONTIER step, not the filtered answer, so a caller with a
# different `valid` set still gets its own correct verdict.
ag13.steps_total = 200
ag13.sgraph.nearest_untested_grid = lambda *a, **k: ([], "ACTION2")
_wide = ag13._graph_explore(board, ["ACTION1", "ACTION2"])
_narrow = ag13._graph_explore(board, ["ACTION1"])
check("memo_respects_caller_valid_set", (_wide, _narrow), ("ACTION2", None))

ag14 = MyAgent(game_id="livelock-bfs-throttle")
_calls14 = [0]


def _nug14(*a, **k):
    _calls14[0] += 1
    return None                      # frontier empty -> rung 1 always falls through


ag14.sgraph.nearest_untested_grid = _nug14
for i in range(ESCALATE_BFS_EVERY * 4):
    ag14.steps_total = 1000 + i
    ag14._ge_memo = None             # nothing else asked this step
    ag14._escalate(board, ["ACTION1"], h=None)
# Budget: 4 throttle windows, and each _graph_explore makes up to TWO searches
# (the imagination-pruned pass plus the completeness fallback when it returns
# nothing). The number that matters is that it is bounded by the window count,
# not by the escalation count -- unthrottled this would be 32 windows' worth.
ok("escalate_bfs_throttled", _calls14[0] <= 4 * 2,
   f"{_calls14[0]} searches over {ESCALATE_BFS_EVERY * 4} escalations "
   f"(<= 2 per {ESCALATE_BFS_EVERY}-step window; unthrottled would be ~64)")
# Throttled does not mean starved: the ladder still returns an action every step.
ag15 = MyAgent(game_id="livelock-bfs-throttle-serves")
ag15.sgraph.nearest_untested_grid = lambda *a, **k: None
_served = [ag15._escalate(board, ["ACTION1", "ACTION2"], h=None)
           for _ in range(ESCALATE_BFS_EVERY)]
ok("throttled_ladder_still_serves", all(x is not None for x in _served),
   f"{_served[:4]}... -- rungs 2+ cover the steps between searches")
# And a result another caller already paid for this step is taken regardless of
# the throttle: the memo makes it free.
ag16 = MyAgent(game_id="livelock-bfs-free-memo")
ag16.steps_total = 500
ag16._esc_bfs_at = 500               # throttle is CLOSED
ag16._ge_memo = (500, "ACTION3")
check("free_memo_used_despite_throttle",
      ag16._escalate(board, ["ACTION3"], h=None), "ACTION3")

print("\n================ SUMMARY ================")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
sys.exit(1 if FAILS else 0)
