# --- path shim: these suites live in tests/ but import the agent from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'eval')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""
Isolation tests for the OFFICIAL SCORING PATH -- the one Kaggle_test.py now uses.

WHY THIS FILE EXISTS
    test_metrics.py pins eval/official_score.py, our RECONSTRUCTION of the
    leaderboard number: it builds a Card by hand and reads _calculate_score. That
    is only half the chain. The other half is `Scorecard.update_scorecard`, which
    decides WHICH CARD PLAY each frame lands in -- and it splits on full_reset:

        arc_agi/scorecard.py:839   if action is RESET:
                                       if full_reset: new_play(...)   <- new row
                                       else:          reset(...)      <- +1 action
        arc_agi/scorecard.py:241   EnvironmentScoreList.score = MAX over plays

    So a full_reset does NOT append a phantom completed level to the play that
    banked one. It starts a fresh play, and the game keeps the best play. The
    reconstruction, which models the whole run as ONE play with the drops
    appended, therefore OVERSTATES a churning agent -- and that overstatement is
    the "reset churn pays" story this repo has been carrying.

    Case 2 below is that claim, executed.

AND ONE STEP FURTHER UP: on the leaderboard the wipe cannot even happen.
    arc_agi/api.py:316-334, competition_mode: a RESET arriving while the engine's
    `_action_count == 0` is NOT performed at all. The API returns the current
    frame and books the update -- the action is spent, nothing changes. Case 3
    pins the engine-side trigger for that guard against the real game.

Bar is NOT "perfect", it is "passes this fixed set". Never delete a case.
Run:  python test_scorecard.py
"""
import os
import sys
import glob

import numpy as np

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
        print(f"  PASS  {name}  {detail}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


def warn(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}  {detail}")
    else:
        print(f"  WARN  {name}  {detail}")
        WARNS.append(name)


try:
    from arc_agi.models import EnvironmentInfo
    from arc_agi.scorecard import EnvironmentScorecard, ScorecardManager
    from arcengine import ActionInput, FrameDataRaw, GameAction, GameState
    HAVE = True
except Exception as e:                                      # pragma: no cover
    HAVE = False
    print(f"arc_agi/arcengine not importable ({e}) -- nothing to test")

from official_score import LevelEvents, official_score, honest_score   # noqa: E402

GID = "lp85-305b61c3"
GUID = "guid-under-test"
BASE = [33, 22, 31, 23, 33, 34, 73, 173]        # lp85's real human baselines


def _frame(action_id, levels, full_reset=False, state=None):
    f = FrameDataRaw(game_id=GID,
                     state=state or GameState.NOT_FINISHED,
                     levels_completed=levels, win_levels=len(BASE),
                     action_input=ActionInput(id=GameAction.from_id(action_id)),
                     guid=GUID, full_reset=full_reset)
    f.frame = [np.zeros((2, 2), dtype=np.int32)]   # non-empty: the wrapper's gate
    return f


def _play(script):
    """Feed a scripted frame stream through the REAL Scorecard and score it.

    `script` is a list of (action_id, levels_completed, full_reset). Returns
    (official_env_score, per_play_scores, LevelEvents, total_actions_the_card_saw).
    """
    mgr = ScorecardManager()
    card_id = mgr.new_scorecard(source_url=None, tags=None, api_key="t",
                                opaque=None, competition_mode=False)
    mgr.add_game(card_id, GUID)
    ev = LevelEvents()
    n = 0
    for action_id, levels, full in script:
        mgr.update_scorecard(GUID, _frame(action_id, levels, full), full)
        if action_id != 0:
            n += 1
        ev.observe(levels, n)
    sc = mgr.get_scorecard(card_id, "t")
    info = EnvironmentInfo(game_id=GID, baseline_actions=list(BASE))
    card = EnvironmentScorecard.from_scorecard(sc, [info])
    env = card.find_environment("lp85")
    return env.score, [round(r.score, 4) for r in env.runs], ev, n


def _one_level_run(n_actions, start_levels=0):
    """RESET, then n-1 ordinary actions, then one that banks a level."""
    out = [(0, start_levels, True)]
    out += [(6, start_levels, False)] * (n_actions - 1)
    out += [(6, start_levels + 1, False)]
    return out


if HAVE:
    # =================================================================
    print("\n--- 1. a clean run: no churn, the two models must agree ------------")
    # =================================================================
    score, plays, ev, n = _play(_one_level_run(60))
    recon = official_score(ev.events, n, BASE).score
    hon = honest_score(ev.first_reach, n, BASE).score
    check("clean_one_play", len(plays), 1)
    ok("clean_scorecard_matches_reconstruction", abs(score - recon) < 1e-9,
       f"scorecard {score:.4f} vs reconstruction {recon:.4f}")
    ok("clean_scorecard_matches_honest", abs(score - hon) < 1e-9,
       f"scorecard {score:.4f} vs honest {hon:.4f}")
    # (33/60)**2 * 100 = 30.25 for level 1, weight 1 of sum(1..8)=36 -> 0.8403.
    ok("clean_level1_scores_what_the_formula_says", abs(score - 0.84028) < 1e-4,
       f"score {score:.5f}")
    CLEAN = score

    # =================================================================
    print("\n--- 2. THE CLAIM: full_reset churn does NOT pay ---------------------")
    # =================================================================
    # Bank level 1, wipe, re-bank, wipe, re-bank. Under Card semantics each wipe
    # opens a NEW PLAY, so the card holds three one-level plays and keeps the best.
    churn = _one_level_run(60) + _one_level_run(60) + _one_level_run(60)
    score, plays, ev, n = _play(churn)
    recon = official_score(ev.events, n, BASE).score
    hon = honest_score(ev.first_reach, n, BASE).score
    check("churn_opens_a_play_per_wipe", len(plays), 3)
    ok("churn_scorecard_is_the_best_play", abs(score - max(plays)) < 1e-3,
       f"score {score:.4f}, plays {plays}")
    ok("churn_buys_nothing_over_one_clean_play", abs(score - CLEAN) < 1e-9,
       f"three wipes scored {score:.5f}; one clean play scores {CLEAN:.5f}")
    # 2 wipes -> the reconstruction credits 3 completed levels (weights 1+2+3)
    # where the card holds 3 separate one-level plays. ~8.6x here; the ratio grows
    # with the number of wipes, which is why a churning run looked like progress.
    ok("reconstruction_overstates_churn", recon > 3 * score,
       f"reconstruction {recon:.4f} vs scorecard {score:.4f} = "
       f"{recon / score:.1f}x over {ev.wipes} wipes -- THIS is the "
       f"'reset churn pays' artifact, and it lives in OUR model, not theirs")
    ok("honest_is_not_fooled_either", hon <= score + 1e-9,
       f"honest {hon:.4f} <= scorecard {score:.4f}")

    # =================================================================
    print("\n--- 3. a level-reset (not full) stays in the same play --------------")
    # =================================================================
    # RESET with actions already taken -> level_reset: same play, +1 action, and
    # the banked level survives.
    script = _one_level_run(60) + [(6, 1, False), (0, 1, False), (6, 1, False)]
    score, plays, ev, n = _play(script)
    check("level_reset_keeps_one_play", len(plays), 1)
    ok("level_reset_keeps_the_level", score > 0.0, f"score {score:.4f}")
    ok("level_reset_costs_an_action", ev.wipes == 0, f"wipes {ev.wipes}")

    # =================================================================
    print("\n--- 4. more level events than baselines -> the scorer returns 0 -----")
    # =================================================================
    # _calculate_score bails with "Human baseline actions size mismatch" when a
    # card holds more level events than the game has baselines. A churning agent
    # can reach that in ONE play if the wipes do not split it -- worth pinning,
    # because it turns a scoring game into a zero.
    long_run = [(0, 0, True)] + [(6, i // 10 + 1, False) for i in range(100)]
    mgr = ScorecardManager()
    cid = mgr.new_scorecard(source_url=None, tags=None, api_key="t", opaque=None)
    mgr.add_game(cid, GUID)
    for a, lv, fr in long_run:
        mgr.update_scorecard(GUID, _frame(a, lv, fr), fr)
    sc = mgr.get_scorecard(cid, "t")
    info = EnvironmentInfo(game_id=GID, baseline_actions=BASE[:3])   # only 3
    env = EnvironmentScorecard.from_scorecard(sc, [info]).find_environment("lp85")
    msgs = [r.message for r in env.runs if r.message]
    ok("size_mismatch_is_reported_not_silent", any("size mismatch" in (m or "")
                                                   for m in msgs), f"{msgs}")
    check("size_mismatch_scores_zero", round(env.score, 6), 0.0)

    # =================================================================
    print("\n--- 5. the engine trigger competition_mode guards against -----------")
    # =================================================================
    # arcengine: handle_reset() full-resets when _action_count == 0, and set_level
    # (i.e. every level change) sets _action_count = 0. That is the exact
    # condition arc_agi/api.py:330 checks before REFUSING the reset.
    hits = sorted(glob.glob(os.path.join(_ROOT, "eval", "real_games", "lp85",
                                         "*", "lp85.py")))
    if not hits:
        warn("real_game_available", False, "eval/real_games/lp85 not found")
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_lp85_probe", hits[0])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from arcengine import ARCBaseGame
        cls = next(o for o in vars(mod).values()
                   if isinstance(o, type) and issubclass(o, ARCBaseGame)
                   and o is not ARCBaseGame)
        import inspect
        g = cls(seed=0) if "seed" in inspect.signature(cls).parameters else cls()
        g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        check("engine_action_count_is_0_after_reset", g._action_count, 0)
        f = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        ok("second_reset_is_a_FULL_reset", bool(f.full_reset),
           "this is the frame that would wipe the score locally, and the one "
           "competition_mode refuses outright")
        g.perform_action(ActionInput(id=GameAction.ACTION6,
                                     data={"x": 1, "y": 1}), raw=True)
        f = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        ok("reset_after_a_real_action_is_a_level_reset", not bool(f.full_reset),
           "an action in between makes RESET safe -- that is the whole gate")

    # =================================================================
    print("\n--- 6. Kaggle_test's inline scorer == eval/official_score.py --------")
    # =================================================================
    # On Kaggle only Kaggle_test.py ships, so its inline fallback is what runs.
    # It must agree with the module the dev box uses, or the two environments
    # measure different things.
    import Kaggle_test as KT
    cases = [
        ([], 0), ([(1, 60)], 60), ([(1, 60), (0, 60), (1, 120)], 120),
        ([(1, 20), (2, 45), (3, 900)], 900), ([(1, 1)], 1),
    ]
    worst = 0.0
    for events, total in cases:
        a = LevelEvents()
        b = KT._LevelEvents()
        for lv, at in events:
            a.observe(lv, at)
            b.observe(lv, at)
        from official_score import score_run as _real_score_run
        x = _real_score_run(a, total, BASE, len(BASE))
        y = KT._score_run(b, total, BASE, len(BASE))
        for key in ("official", "honest", "wipes", "real_levels",
                    "credited_levels", "levels_denominator"):
            worst = max(worst, abs(float(x[key]) - float(y[key])))
    ok("inline_scorer_matches_official_score_module", worst < 1e-9,
       f"max abs delta {worst:.2e} over {len(cases)} cases")
    ok("inline_level_score_matches", abs(KT._level_score(33, 60)
                                         - __import__("official_score").level_score(33, 60)) < 1e-9)

    # =================================================================
    print("\n--- 7. Kaggle_test's OfficialScorecard == eval/scoreboard.py --------")
    # =================================================================
    # The SAME objects exist twice on purpose: there is no eval/ directory on
    # Kaggle, and importing one from the other is one of the two bugs that made
    # Kaggle_test.py unrunnable there. Drift is prevented by THIS check, not by
    # an import -- so the two copies must return byte-identical read-outs when
    # driven by the same frame stream.
    import scoreboard as SB

    def _drive(cls, script, comp=False):
        board = cls(competition_mode=comp)
        if not board.ok:
            return None
        board.mgr.add_game(board.card_id, GUID)
        for a, lv, fr in script:
            board.mgr.update_scorecard(GUID, _frame(a, lv, fr), fr)
        board.register(EnvironmentInfo(game_id=GID, baseline_actions=list(BASE)))
        return board.game("lp85"), board.overall()

    for tag, script in (("clean", _one_level_run(60)),
                        ("churn", _one_level_run(60) * 3),
                        ("nolevel", [(0, 0, True)] + [(6, 0, False)] * 20)):
        a = _drive(SB.OfficialScorecard, script)
        b = _drive(KT.OfficialScorecard, script)
        ok(f"scoreboard_copies_agree_{tag}", a == b,
           f"score {a[0]['score']:.5f}, {a[0]['plays']} play(s)" if a == b
           else f"DRIFT\n    eval/scoreboard.py {a}\n    Kaggle_test.py    {b}")

    # make_env_info is the other half: it decides which baseline list the
    # calculator divides by, and a length mismatch there silently scores 0.
    hits = sorted(glob.glob(os.path.join(_ROOT, "eval", "real_games", "lp85",
                                         "*", "lp85.py")))
    if not hits:
        warn("real_game_available_for_env_info", False, "eval/real_games/lp85")
    else:
        meta = {"baseline_steps": list(BASE)}
        ia, sa = SB.make_env_info(hits[0], "lp85", meta)
        ib, sb = KT.make_env_info(hits[0], "lp85", meta)
        check("make_env_info_same_source", sa, sb)
        check("make_env_info_same_baselines",
              list(ia.baseline_actions), list(ib.baseline_actions))
        check("make_env_info_same_class_name", ia.class_name, ib.class_name)
        ok("make_env_info_prefers_metadata_json", sa == "metadata.json",
           f"source {sa!r} -- the calculator divides by metadata's "
           f"baseline_actions, so this is the list that must win")

    # The predicate competition mode is built on. It reads the ENGINE's counter,
    # so it must be false the moment a real action has been spent.
    ok("reset_would_wipe_is_false_without_a_game",
       SB.reset_would_wipe(object()) is False)


print("\n================ SUMMARY ================")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
sys.exit(1 if FAILS else 0)
