# --- path shim: these suites live in tests/ but import the agent from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'eval')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""test_verifier.py -- COMPONENT 2.5 CandidateVerifier (system_v2 3.3, the trust boundary).

The property under test, stated once so every case can be read against it:

    A candidate is accepted ONLY IF it reproduces the RECORDED LOG, HUD-masked,
    over a slice that spans the whole history -- and no candidate, however
    pathological, can stall the agent or crash it.

The gate this replaces judged `claimed[-5:]`: the five newest transitions. That
cannot refute a model which fits only the recent past, and "fits only the recent
past" is precisely what a synthesiser fitted on recent frames produces. Section 3
below is that case, and it is the reason this file exists.

Run:  PYTHONIOENCODING=utf-8 python test_verifier.py
Never delete a case.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ARC_NO_LLM", "1")

from my_agent import (                                    # noqa: E402
    CandidateVerifier, Verdict, Transition, WorldModelManager, StateEncoder,
    VERIFY_PROMOTE_THRESHOLD, MIN_CHANGED_FOR_VERIFY, VERIFY_MAX_SAMPLES,
)

FAILS = []
CHECKS = [0]


def check(name, got, want):
    CHECKS[0] += 1
    if got != want:
        FAILS.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL {name}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {name}")


def near(name, got, want, tol=1e-9):
    CHECKS[0] += 1
    if abs(got - want) > tol:
        FAILS.append(f"{name}: got {got!r}, want ~{want!r}")
        print(f"  FAIL {name}: got {got!r}, want ~{want!r}")
    else:
        print(f"  ok   {name}")


# ---------------------------------------------------------------- fixtures
def grid(n=8, fill=0):
    return np.full((n, n), fill, dtype=np.int8)


def shift_right(g, action):
    """Ground truth: ACTION3 moves the single 5-pixel one column right, WRAPPING.

    Wrapping and not clamping, deliberately. A clamped pixel parks against the
    wall and every later transition becomes a no-op, so a log of 40 is really 7
    transitions and 33 frames of nothing -- against which the identity function
    scores 0.83 and "verified" means nothing. A saturating fixture is a fixture
    that tests the fixture.
    """
    out = np.asarray(g).copy()
    ys, xs = np.where(out == 5)
    if len(ys) and action == "ACTION3":
        out[ys[0], xs[0]] = 0
        out[ys[0], (xs[0] + 1) % out.shape[1]] = 5
    return out


def make_log(n=20, action="ACTION3", hud_row=None, world=shift_right):
    """A synthetic transition log. `hud_row` ticks a row nothing models.

    The clock writes 1..3 and never 5, so it cannot be mistaken for the avatar.
    """
    log, g = [], grid()
    g[3, 0] = 5
    for i in range(n):
        nxt = world(g, action)
        if hud_row is not None:
            nxt = nxt.copy()
            nxt[hud_row, :] = (i % 3) + 1
        log.append(Transition(prev=g.copy(), action=action, next=nxt.copy(),
                              reward=0.0, diff_encoding="", timestamp=float(i)))
        g = nxt
    return log


def covering(fn, actions=None):
    """Attach the `.covers` contract compile_program sets."""
    fn.covers = None if actions is None else frozenset(actions)
    return fn


print("=" * 70)
print("1. the contract: a correct model is accepted, identity is not")
print("=" * 70)
V = CandidateVerifier()
log = make_log(20)
v_good = V.check(covering(lambda g, a: shift_right(g, a)), log)
check("correct_model_accepted", bool(v_good), True)
near("correct_model_accuracy", v_good.accuracy, 1.0)
check("correct_model_reason", v_good.reason, "verified")
check("correct_model_no_errors", v_good.errors, 0)

v_id = V.check(covering(lambda g, a: np.asarray(g).copy()), log)
check("identity_rejected", bool(v_id), False)
check("identity_reason", v_id.reason, "falsified")

v_none = V.check(None, log)
check("no_candidate_rejected", bool(v_none), False)
check("no_candidate_reason", v_none.reason, "no-candidate")

check("verdict_is_falsy_when_rejected", bool(Verdict(accepted=False)), False)
check("verdict_is_truthy_when_accepted", bool(Verdict(accepted=True)), True)

print()
print("=" * 70)
print("2. thin evidence is not a refutation")
print("=" * 70)
# Fewer claimed transitions than MIN_CHANGED_FOR_VERIFY says nothing about the
# candidate. It must not read as 'wrong' -- that is how a correct partial program
# gets banned before it has been seen.
thin = make_log(MIN_CHANGED_FOR_VERIFY - 1)
v_thin = V.check(covering(lambda g, a: shift_right(g, a)), thin)
check("thin_rejected", bool(v_thin), False)
check("thin_reason", v_thin.reason, "thin-evidence")
check("thin_tested_nothing", v_thin.n_tested, 0)

# A model that CLAIMS one action is judged on that action only.
mixed = make_log(12, action="ACTION3") + make_log(12, action="ACTION1")
only3 = covering(lambda g, a: shift_right(g, a), ["ACTION3"])
v_part = V.check(only3, mixed)
check("partial_model_accepted", bool(v_part), True)
check("partial_model_claimed_only_its_own", v_part.n_claimed, 12)

print()
print("=" * 70)
print("3. THE REGRESSION: a recency-overfit model must NOT pass")
print("=" * 70)
# A candidate that has MEMORISED the newest 5 transitions and is wrong on
# everything else. The old gate sampled exactly those 5 and promoted it at 1.00.
long_log = make_log(40)


def recency_overfit(g, a):
    for t in long_log[-5:]:
        if np.array_equal(np.asarray(g), t.prev):
            return t.next.copy()
    return np.asarray(g).copy()          # wrong everywhere else


v_over = V.check(covering(recency_overfit), long_log)
check("recency_overfit_rejected", bool(v_over), False)
check("recency_overfit_reason", v_over.reason, "falsified")

# ...and the pre-2026-08-06 gate is shown to accept it, so the case is a real
# difference and not a test of nothing.
enc = StateEncoder()
wmm = WorldModelManager(llm=None, encoder=enc)
old = wmm._verify_last5(covering(recency_overfit), long_log, {})
near("old_gate_would_have_accepted_it", old, 1.0)
check("old_gate_above_threshold", old >= VERIFY_PROMOTE_THRESHOLD, True)

print()
print("=" * 70)
print("4. sample selection: spans the log, deterministic, oldest first")
print("=" * 70)
short = make_log(10)
sel_short = V.select(short)
check("short_log_tested_whole", len(sel_short), 10)
check("short_log_chronological", [id(t) for t in sel_short],
      [id(t) for t in short])

big = make_log(300)
sel_big = V.select(big)
check("big_log_capped", len(sel_big), VERIFY_MAX_SAMPLES)
check("big_log_deterministic", [id(t) for t in V.select(big)],
      [id(t) for t in sel_big])
check("big_log_includes_oldest", sel_big[0] is big[0], True)
check("big_log_includes_newest", sel_big[-1] is big[-1], True)
# The point of the stride: coverage of the WHOLE log, not a window at either end.
# Positions come from an identity map -- list.index() on a record holding numpy
# arrays compares them elementwise and raises "truth value is ambiguous".
pos = {id(t): i for i, t in enumerate(big)}
idxs = [pos[id(t)] for t in sel_big]
check("big_log_spans_history",
      max(idxs) - min(idxs) >= int(0.9 * (len(big) - 1)), True)
check("history_tested_before_recent", idxs[0] < idxs[-1], True)
check("selection_has_no_duplicates", len(set(idxs)), len(idxs))

print()
print("=" * 70)
print("5. no candidate can crash or stall the agent")
print("=" * 70)


def raiser(g, a):
    raise RuntimeError("candidates may be hostile")


v_raise = V.check(covering(raiser), log)
check("raising_candidate_rejected", bool(v_raise), False)
check("raising_candidate_errors_counted", v_raise.errors > 0, True)

v_wrongtype = V.check(covering(lambda g, a: "not an array"), log)
check("wrong_return_type_rejected", bool(v_wrongtype), False)

v_wrongshape = V.check(covering(lambda g, a: np.zeros((3, 3), dtype=np.int8)), log)
check("wrong_shape_rejected", bool(v_wrongshape), False)

v_returns_none = V.check(covering(lambda g, a: None), log)
check("none_return_rejected", bool(v_returns_none), False)


def slow(g, a):
    time.sleep(0.05)
    return shift_right(g, a)


t0 = time.perf_counter()
v_slow = V.check(covering(slow), make_log(60), budget_ms=120.0)
el = (time.perf_counter() - t0) * 1000.0
check("slow_candidate_rejected", bool(v_slow), False)
check("slow_candidate_reason", v_slow.reason, "budget")
check("slow_candidate_flagged_stalled", v_slow.stalled, True)
check("slow_candidate_bounded", el < 1500.0, True)


def hangs(g, a):
    time.sleep(30.0)
    return g


t0 = time.perf_counter()
v_hang = V.check_guarded(covering(hangs), log, guard_ms=250.0)
el = (time.perf_counter() - t0) * 1000.0
check("hanging_candidate_rejected", bool(v_hang), False)
check("hanging_candidate_bounded_by_guard", el < 3000.0, True)
check("guard_returns_a_verdict", isinstance(v_hang, Verdict), True)
# The guard must not change the answer for a well-behaved candidate.
v_guard_ok = V.check_guarded(covering(lambda g, a: shift_right(g, a)), log)
check("guard_accepts_correct_model", bool(v_guard_ok), True)

print()
print("=" * 70)
print("6. HUD masking: the fit mask and the verify mask are the same mask")
print("=" * 70)
# A world model that is exactly right about the world and knows nothing about the
# clock on row 7. Judged raw it is 0.00; judged under the mask it fitted with, it
# is 1.00. Getting this backwards demoted correct models on every game with a
# live HUD -- see world-model-hud-mask-asymmetry.
hud_log = make_log(20, hud_row=7)
world_only = covering(lambda g, a: shift_right(g, a))
v_raw = V.check(world_only, hud_log, mask_for=None)
check("unmasked_comparison_refutes_correct_model", bool(v_raw), False)

# masked_diff does `d & ~mask`, so a TRUE cell is one to IGNORE. Getting this
# polarity backwards silently compares raw and every masked case still "passes".
mask = np.zeros((8, 8), dtype=bool)
mask[7, :] = True                        # True = ignore (the HUD row)
v_masked = V.check(world_only, hud_log, mask_for={"ACTION3": mask})
check("masked_comparison_accepts_it", bool(v_masked), True)
near("masked_accuracy", v_masked.accuracy, 1.0)

# The mask must not launder a model that is wrong about the WORLD.
v_masked_bad = V.check(covering(lambda g, a: np.asarray(g).copy()), hud_log,
                       mask_for={"ACTION3": mask})
check("mask_does_not_excuse_a_wrong_model", bool(v_masked_bad), False)

print()
print("=" * 70)
print("7. early abort stops paying for a candidate that cannot recover")
print("=" * 70)
calls = [0]


def wrong_and_counted(g, a):
    calls[0] += 1
    return np.asarray(g).copy()


big_log = make_log(60)
V.check(covering(wrong_and_counted), big_log)
# threshold 0.8 over a 48-sample slice tolerates 9 wrong; the 10th is decisive.
budgeted = int((1.0 - VERIFY_PROMOTE_THRESHOLD) * VERIFY_MAX_SAMPLES) + 1
check("aborted_at_the_decisive_sample", calls[0], budgeted)
check("abort_is_cheaper_than_full_sweep", calls[0] < VERIFY_MAX_SAMPLES, True)

# A candidate at exactly the threshold is accepted; one below is not.
def fails_k_times(k):
    state = {"n": 0}

    def f(g, a):
        state["n"] += 1
        if state["n"] <= k:
            return np.asarray(g).copy()      # wrong
        return shift_right(g, a)
    return covering(f)


exact = make_log(10)                          # 10 samples, tolerate 2 wrong
v_at = V.check(fails_k_times(2), exact)
near("at_threshold_accuracy", v_at.accuracy, 0.8)
check("at_threshold_accepted", bool(v_at), True)
v_below = V.check(fails_k_times(3), exact)
check("below_threshold_rejected", bool(v_below), False)

print()
print("=" * 70)
print("8. dry_run: the unscored executor")
print("=" * 70)
start = grid(); start[3, 0] = 5
states = CandidateVerifier.dry_run(lambda g, a: shift_right(g, a), start,
                                   ["ACTION3"] * 3)
check("dry_run_length", len(states), 3)
check("dry_run_moved_the_pixel", int(np.where(states[-1] == 5)[1][0]), 3)
check("dry_run_did_not_mutate_start", int(np.where(start == 5)[1][0]), 0)
check("dry_run_horizon_respected",
      len(CandidateVerifier.dry_run(lambda g, a: shift_right(g, a), start,
                                    ["ACTION3"] * 9, horizon=2)), 2)
# "I don't model this" must come back as None, never as "nothing happens".
check("dry_run_none_when_model_declines",
      CandidateVerifier.dry_run(lambda g, a: None, start, ["ACTION3"]), None)
check("dry_run_none_on_raise",
      CandidateVerifier.dry_run(raiser, start, ["ACTION3"]), None)
check("dry_run_none_on_shape_change",
      CandidateVerifier.dry_run(lambda g, a: np.zeros((2, 2)), start,
                                ["ACTION3"]), None)
check("dry_run_empty_plan", CandidateVerifier.dry_run(
    lambda g, a: shift_right(g, a), start, []), None)
check("dry_run_no_model", CandidateVerifier.dry_run(None, start, ["ACTION3"]),
      None)

print()
print("=" * 70)
print("9. degenerate input never raises")
print("=" * 70)
ok = True
for args in ((covering(lambda g, a: g), []), (covering(lambda g, a: g), None),
             (None, None), (covering(lambda g, a: g), make_log(1))):
    try:
        r = V.check(*args)
        ok = ok and isinstance(r, Verdict) and not r.accepted
    except Exception as e:                                   # pragma: no cover
        ok = False
        print(f"    raised on {args!r}: {e}")
check("degenerate_inputs_return_a_verdict", ok, True)

print()
print("=" * 70)
print("10. wiring: WorldModelManager routes through the verifier")
print("=" * 70)
check("manager_owns_a_verifier", isinstance(wmm.verifier, CandidateVerifier), True)
check("manager_has_verdict_slot", hasattr(wmm, "last_verdict"), True)
# The kill switch must restore the old sampling exactly.
os.environ["ARC_NO_VERIFIER"] = "1"
try:
    off = wmm.verify(covering(recency_overfit), long_log)
finally:
    os.environ.pop("ARC_NO_VERIFIER", None)
on = wmm.verify(covering(recency_overfit), long_log)
check("kill_switch_restores_old_gate", off >= VERIFY_PROMOTE_THRESHOLD, True)
check("verifier_on_refutes_it", on < VERIFY_PROMOTE_THRESHOLD, True)
check("verdict_recorded_when_on", isinstance(wmm.last_verdict, Verdict), True)

print()
print("=" * 70)
print(f"CHECKS: {CHECKS[0]}   HARD FAILS: {len(FAILS)}")
for f in FAILS:
    print("  " + f)
print("=" * 70)
sys.exit(1 if FAILS else 0)
