# --- path shim: these suites live in tests/ but import the agent from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_ROOT, _os.path.join(_ROOT, 'eval'), _os.path.join(_ROOT, 'sarbloh', 'legacy')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""
Isolation test harness for the SCORING LAYER: eval/official_score.py.

Doctrine: the component stays where it lives; this file only imports it and
feeds hand-built inputs whose correct answer we know by construction.

WHY THIS FILE EXISTS. Every harness in this repo carried its own transcription
of the RHAE formula off the paper, and all of them were measuring a quantity the
competition does not compute. The gap was ~400x on real recorded runs. A metric
nobody tests is a metric nobody should steer on, so the scoring layer now gets
the same treatment as any other component.

THE INVARIANTS THIS FILE EXISTS TO PIN (read out of arc_agi/scorecard.py, not
guessed):
  * `Card.set_levels_completed` appends to actions_by_level on ANY change,
    INCLUDING a drop to 0 after full_reset()                          (L710)
  * `_calculate_score` reads the i-th entry as "level i+1, COMPLETED",
    ignoring the level number actually recorded                       (L477-491)
  * weights are 1-indexed (`level_index=level_idx + 1`)               (L487)
  * the denominator iterates `range(len(baseline_actions))` -- the BASELINE
    list sets it, never the frame's win_levels                        (L475)
  * per-level cap 115.0, applied to (baseline/actions)**2 * 100       (L170)
  * per-environment cap `min(score, max_weights/total_weights*100)`   (L206)
=> AN AGENT THAT BANKS LEVEL 1 ONCE AND THEN CHURNS full_reset() SCORES 100.

`official` must reproduce that faithfully -- bugs included -- because it is what
the leaderboard pays. `honest` must NOT, because it is what we steer capability
on. Both are pinned here.

Bar is NOT "perfect", it is "passes this fixed set". Never delete a case.
Run:  python test_metrics.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))

from official_score import (LevelEvents, official_score, honest_score,  # noqa: E402
                            score_run, load_baselines, DEFAULT_INDEX,
                            _FallbackCalc, _OfficialCalc)

FAILS = []
WARNS = []


def check(name, got, want):
    if got == want:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}: got {got!r}, want {want!r}")
        FAILS.append(name)


def close(name, got, want, tol=1e-6):
    if abs(got - want) <= tol:
        print(f"  PASS  {name}  ({got:.6f})")
    else:
        print(f"  FAIL  {name}: got {got!r}, want {want!r}")
        FAILS.append(name)


def ok(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
    else:
        print(f"  FAIL  {name}" + (f"  [{detail}]" if detail else ""))
        FAILS.append(name)


def warn(name, cond, detail=""):
    if not cond:
        print(f"  WARN  {name}" + (f"  [{detail}]" if detail else ""))
        WARNS.append(name)
    else:
        print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))


AR25 = [17, 22, 103, 29, 29, 159, 152, 66]          # 8 levels, real baselines


# =====================================================================
print("\n--- 1. LevelEvents mirrors Card.set_levels_completed -----------------")
# =====================================================================
ev = LevelEvents()
for (lv, a) in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 9)]:
    ev.observe(lv, a)
check("events_appended_only_on_change", ev.events, [(1, 3), (2, 9)])
check("first_reach_is_monotonic", ev.first_reach, {1: 3, 2: 9})
check("no_wipes_when_monotonic", ev.wipes, 0)
check("real_levels_tracks_peak", ev.real_levels, 2)

# A drop is an EVENT. This is the whole finding.
ev = LevelEvents()
for (lv, a) in [(1, 10), (0, 20), (1, 30), (0, 40)]:
    ev.observe(lv, a)
check("drop_to_zero_is_recorded_as_an_event",
      ev.events, [(1, 10), (0, 20), (1, 30), (0, 40)])
check("wipes_counted", ev.wipes, 2)
check("peak_unaffected_by_wipes", ev.real_levels, 1)
check("first_reach_records_only_the_FIRST_bank", ev.first_reach, {1: 10})

# observe() reports whether it fired, so callers can log the event.
ev = LevelEvents()
check("observe_returns_False_on_no_change", ev.observe(0, 1), False)
check("observe_returns_True_on_change", ev.observe(1, 2), True)


# =====================================================================
print("\n--- 2. official_score reproduces the leaderboard, bugs included -----")
# =====================================================================
# (a) honest agent: banks level 1 at action 150, stuck for the remaining budget.
s = official_score([(1, 150)], 1500, AR25)
close("A_single_real_level_scores_weighted_share", s.score, 0.0357, tol=5e-4)
check("A_credited_levels", s.levels_completed, 1)
ok("A_level1_score_is_capped_ratio",
   abs(s.level_scores[0] - min((17 / 150) ** 2 * 100, 115.0)) < 1e-9,
   f"{s.level_scores[0]:.4f}")
ok("A_uncompleted_levels_score_zero", all(x == 0.0 for x in s.level_scores[1:]))

# (b) THE EXPLOIT: identical real progress, but full_reset churn after each death.
seq, a, out = [], 150, []
for _ in range(8):
    out.append((1, a)); a += 20
    out.append((0, a)); a += 20
s = official_score(out, 1500, AR25)
close("B_reset_churn_scores_a_perfect_environment", s.score, 100.0, tol=1e-6)
check("B_scorer_credits_8_levels_for_1_real_level", s.levels_completed, 8)
ok("B_every_churn_cycle_hits_the_115_cap",
   all(x == 115.0 for x in s.level_scores[1:]),
   f"{[round(x, 1) for x in s.level_scores]}")

# (c) a genuinely good agent: 4 real levels at human speed.
s = official_score([(1, 17), (2, 39), (3, 142), (4, 171)], 1500, AR25)
close("C_four_real_levels_at_human_speed", s.score, 27.7778, tol=5e-4)
ok("C_env_cap_binds_at_completed_weight_share",
   abs(s.score - (1 + 2 + 3 + 4) / 36 * 100) < 5e-4,
   f"cap = {(1 + 2 + 3 + 4) / 36 * 100:.4f}")

# The env cap must actually bind: four levels ALL above human speed still cannot
# exceed the weighted fraction completed.
s = official_score([(1, 1), (2, 2), (3, 3), (4, 4)], 1500, AR25)
close("C_superhuman_cannot_beat_the_env_cap", s.score, 10 / 36 * 100, tol=5e-4)

# Zero progress scores zero, and does not divide by zero.
s = official_score([], 1500, AR25)
close("empty_run_scores_zero", s.score, 0.0)
check("empty_run_credits_nothing", s.levels_completed, 0)


# =====================================================================
print("\n--- 3. honest_score refuses to pay for churn ------------------------")
# =====================================================================
ev = LevelEvents()
a = 150
for _ in range(8):
    ev.observe(1, a); a += 20
    ev.observe(0, a); a += 20
r = score_run(ev, 1500, AR25, win_levels=8)
close("churn_official_is_100", r["official"], 100.0, tol=1e-6)
close("churn_honest_is_one_level_only", r["honest"], 0.0357, tol=5e-4)
close("honest_rhae_is_the_same_number_as_a_fraction",
      r["honest_rhae"], r["honest"] / 100.0, tol=1e-12)
check("churn_real_levels_is_1", r["real_levels"], 1)
check("churn_credited_levels_is_8", r["credited_levels"], 8)
check("churn_wipes_counted", r["wipes"], 8)
ok("churn_inflation_is_large", r["inflation"] > 1000, f"{r['inflation']:.0f}x")

# No churn -> the two numbers agree exactly (modulo the 0-100 vs 0-1 scale).
ev = LevelEvents()
ev.observe(1, 150)
r = score_run(ev, 1500, AR25, win_levels=8)
close("clean_run_official_equals_honest", r["official"], r["honest"], tol=1e-9)
close("clean_run_inflation_is_1", r["inflation"], 1.0, tol=1e-9)
ok("both_scores_are_in_the_same_units",
   r["official"] > 0 and abs(r["official"] - r["honest"]) < 1e-9,
   f"official={r['official']:.6f} honest={r['honest']:.6f}")
check("clean_run_no_wipes", r["wipes"], 0)

# Honest charges deaths and resets to the level they were spent on.
ev = LevelEvents()
ev.observe(1, 50)          # level 1 cost 50
ev.observe(0, 60)          # wiped
ev.observe(1, 300)         # replay -- free, level 1 already banked
ev.observe(2, 400)         # level 2 cost 400-50 = 350, replay included
r = score_run(ev, 1500, AR25, win_levels=8)
check("honest_charges_replay_to_the_next_level",
      r["honest_level_actions"][:2], [50, 350])
check("honest_counts_two_real_levels", r["real_levels"], 2)


# =====================================================================
print("\n--- 4. the denominator comes from the BASELINES, not win_levels -----")
# =====================================================================
ev = LevelEvents()
ev.observe(1, 100)
r = score_run(ev, 1500, AR25, win_levels=8)
check("denominator_is_len_baselines", r["levels_denominator"], 8)
ok("no_warning_when_they_agree", r["warn"] is None, str(r["warn"]))

# cn04 really does ship win_levels=5 with 6 baselines. Using 5 would inflate
# every cn04 score by 21/15 = 1.4x. The scorer must follow the baselines AND say so.
r = score_run(ev, 1500, [16, 54, 69, 318, 208, 114], win_levels=5)
check("cn04_shape_uses_6_not_5", r["levels_denominator"], 6)
ok("cn04_shape_warns", r["warn"] is not None and "win_levels=5" in r["warn"],
   str(r["warn"]))

# A game with no baselines is scored 0 by the competition -- not skipped.
r = score_run(ev, 1500, [], win_levels=8)
close("no_baselines_scores_zero", r["official"], 0.0)
ok("no_baselines_warns", r["warn"] is not None, str(r["warn"]))


# =====================================================================
print("\n--- 5. the fallback calculator matches the official one -------------")
# =====================================================================
# The fallback exists so the harness runs without arc_agi installed. It must
# never drift: whenever the real class IS importable, pin them against each
# other on every case above.
if _OfficialCalc is None:
    warn("official_calculator_importable", False, "arc_agi absent; fallback untested")
else:
    cases = [
        ([(1, 150)], 1500, AR25),
        (out, 1500, AR25),
        ([(1, 17), (2, 39), (3, 142), (4, 171)], 1500, AR25),
        ([], 1500, AR25),
        ([(1, 5)], 900, [16, 54, 69, 318, 208, 114]),
        ([(1, 1), (2, 2), (3, 3)], 50, [7, 28, 30, 20, 37, 45]),
    ]

    def _fallback(events, total, base):
        calc = _FallbackCalc()
        prev = 0
        for i, b in enumerate(base):
            if i < len(events):
                _l, at = events[i]
                acts, done = at - prev, True
                prev = at
            else:
                acts, done = total - prev, False
                prev = total
            calc.add_level(i + 1, done, acts, b)
        return calc.to_score()

    worst = 0.0
    for (e, t, b) in cases:
        a1 = official_score(e, t, b).score
        a2 = _fallback(e, t, b).score
        worst = max(worst, abs(a1 - a2))
    ok("fallback_matches_official_on_every_case", worst < 1e-9,
       f"max abs delta {worst:.2e} over {len(cases)} cases")


# =====================================================================
print("\n--- 6. the real games_index is well formed --------------------------")
# =====================================================================
if not os.path.exists(DEFAULT_INDEX):
    warn("games_index_present", False, DEFAULT_INDEX)
else:
    idx = load_baselines(DEFAULT_INDEX)
    check("games_index_has_25_games", len(idx), 25)
    ok("every_game_has_baselines",
       all(m["baseline_steps"] for m in idx.values()),
       str([g for g, m in idx.items() if not m["baseline_steps"]]))
    ok("every_baseline_is_positive",
       all(all(x > 0 for x in m["baseline_steps"]) for m in idx.values()))
    mismatch = {g: (m["win_levels"], len(m["baseline_steps"]))
                for g, m in idx.items() if m["win_levels"] != len(m["baseline_steps"])}
    # This is a KNOWN data quirk, not a regression -- pin it so a silent change
    # to games_index.json shows up here instead of moving every score.
    check("known_win_levels_mismatches", mismatch, {"cn04": (5, 6)})


print("\n================ SUMMARY ================")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
sys.exit(1 if FAILS else 0)
