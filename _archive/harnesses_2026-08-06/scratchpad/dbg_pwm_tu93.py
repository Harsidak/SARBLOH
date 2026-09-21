"""Instrument PatchWorldModel.predict_noop on tu93: why is it dormant?

For every predict_noop call, replicate the internals WITHOUT memoization and
record where the test fails:
  - unknown  : some live cell's (action, patch) never seen
  - underseen: patch seen but < MIN_SEEN
  - nowitness: all patches known but no always-ineffective witness patch
Also scores two relaxation variants per call:
  - AGNOSTIC : known-test sums patch counts across ALL actions (witness stays per-action)
  - MINSEEN1 : MIN_SEEN=1 for the known test (witness unchanged)

Run from ARC_AGI EXPERMENTATIONS/:
    PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 python -u scratchpad/dbg_pwm_tu93.py [gid] [steps]
"""
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from local_eval import run_game, default_agent_factory  # noqa: E402
import my_agent as ma  # noqa: E402

DIAG = {
    "calls": 0, "true_strict": 0, "true_agnostic": 0, "true_minseen1": 0,
    "fail": Counter(),            # first-failure reason per call (strict)
    "unknown_hist": Counter(),    # bucketed count of not-known cells per call
    "examples": [],
}


def factory(game_id):
    agent = default_agent_factory(game_id)
    pwm = agent.pwm
    agn = {}          # patch_bytes -> total occurrences across actions

    orig_learn = pwm.learn

    def learn(prev, action, nxt, mask=None):
        n0 = pwm.trained
        orig_learn(prev, action, nxt, mask)
        if pwm.trained > n0:      # it actually recorded
            pat = pwm._patches(prev, mask)
            for i in pwm._live(mask, prev.shape):
                b = pat[i].tobytes()
                agn[b] = agn.get(b, 0) + 1

    def predict_noop(grid, action, mask=None):
        a = ma.base_action(action)
        if a == "ACTION6":
            return False
        pat = pwm._patches(grid, mask)
        stats = pwm.stats
        n_unknown = n_underseen = 0
        witness = False
        agn_known = True
        ms1_known = True
        for i in pwm._live(mask, grid.shape):
            b = pat[i].tobytes()
            s = stats.get((a, b))
            tot = 0 if s is None else s[0] + s[1]
            if s is None:
                n_unknown += 1
            elif tot < pwm.MIN_SEEN:
                n_underseen += 1
            if tot < 1:
                ms1_known = False
            if agn.get(b, 0) < pwm.MIN_SEEN:
                agn_known = False
            if s is not None and s[0] == 0 and s[1] >= pwm.WITNESS_MIN:
                witness = True
        strict_known = (n_unknown + n_underseen) == 0
        res = strict_known and witness
        DIAG["calls"] += 1
        if res:
            DIAG["true_strict"] += 1
        if agn_known and witness:
            DIAG["true_agnostic"] += 1
        if ms1_known and witness:
            DIAG["true_minseen1"] += 1
        if not strict_known:
            DIAG["fail"]["unknown" if n_unknown else "underseen"] += 1
        elif not witness:
            DIAG["fail"]["nowitness"] += 1
        nk = n_unknown + n_underseen
        bucket = 0 if nk == 0 else (1 if nk <= 2 else (5 if nk <= 5 else
                  (20 if nk <= 20 else (100 if nk <= 100 else 999))))
        DIAG["unknown_hist"][bucket] += 1
        if len(DIAG["examples"]) < 8 and nk not in [e[0] for e in DIAG["examples"]]:
            DIAG["examples"].append(
                (nk, a, n_unknown, n_underseen, witness, agn_known, ms1_known))
        return res

    pwm.predict_noop = predict_noop
    factory.agent = agent
    return agent


if __name__ == "__main__":
    gid = sys.argv[1] if len(sys.argv) > 1 else "tu93"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    import glob as _g
    gf = _g.glob(os.path.join(ROOT, "eval", "real_games", gid, "*", gid + ".py"))[0]
    r = run_game(gf, factory, seed=0, max_actions=steps)
    print(r)
    ag = factory.agent
    print(f"\npwm.trained={ag.pwm.trained}  patches={len(ag.pwm.stats)}  "
          f"noop_skips={len(ag.sgraph._noop_skip)}")
    print(f"calls={DIAG['calls']}  TRUE strict={DIAG['true_strict']}  "
          f"agnostic={DIAG['true_agnostic']}  minseen1={DIAG['true_minseen1']}")
    print(f"failure reasons: {dict(DIAG['fail'])}")
    print(f"not-known-cells histogram (bucket<=): {dict(sorted(DIAG['unknown_hist'].items()))}")
    print("examples (nk, action, n_unknown, n_underseen, witness, agn_known, ms1_known):")
    for e in DIAG["examples"]:
        print("  ", e)
    # How many known patches are certified witnesses at all?
    w = sum(1 for s in ag.pwm.stats.values() if s[0] == 0 and s[1] >= ag.pwm.WITNESS_MIN)
    known = sum(1 for s in ag.pwm.stats.values() if s[0] + s[1] >= ag.pwm.MIN_SEEN)
    print(f"stats: {len(ag.pwm.stats)} patches, {known} known, {w} certified witnesses")
