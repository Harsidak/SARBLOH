"""Step 0: is O3 starved of DATA, or fitted on the WRONG BASIS?

Reads the episode dumps written by `ARC_O3_DUMP=<dir>` (raw boards + how the episode
ended) and answers three questions offline, at zero cost in scored actions:

  Q1  How many of the 20 current features earn a direction from the WIN alone?
      (should reproduce the live `features_with_a_direction=3` on lp85)
  Q2  How many of those ALSO move the same way on DEATH trajectories? A feature that
      rises on the way to a win AND on the way to a death is not progress, it is
      elapsed time -- and with one win it is indistinguishable from progress. This is
      the number A1 (learn from deaths) exists to remove.
  Q3  Does an OBJECT-LEVEL basis have more monotone structure than the colour
      histogram? This is C1. If the object basis finds directions where the histogram
      finds none, the problem was never the data.

Usage:
    python scratchpad/diag_o3.py scratchpad/o3dump
"""
import os
import sys
import glob
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("ARC_AGENT_SEED", "0")

from my_agent import ProgressModel, StateEncoder      # noqa: E402

MONOTONE_MIN = ProgressModel.MONOTONE_MIN
MIN_TRACE = ProgressModel.MIN_TRACE

FEAT_NAMES = ([f"colour[{i}]" for i in range(ProgressModel.N_COLORS)]
              + ["boundary", "extent", "h-sym", "v-sym"])

# ---------------------------------------------------------------- object basis (C1)
OBJ_NAMES = ["n_objects", "n_colours", "largest_frac", "mean_size", "size_spread",
             "n_singletons", "bg_frac", "spread", "n_shape_classes", "touching_pairs"]
_ENC = StateEncoder()


def object_features(grid: np.ndarray) -> np.ndarray:
    """A TOTAL object-level descriptor. Every component is scale-free, same contract as
    ProgressModel.features -- the only thing that changes is WHAT is being described:
    the things on the board rather than the colours of its pixels."""
    f = np.zeros(len(OBJ_NAMES), dtype=np.float64)
    try:
        g = np.asarray(grid)
        n = float(g.size)
        bg = _ENC.detect_background(g)
        objs = _ENC.objects(g, bg)
        f[0] = min(1.0, len(objs) / 64.0)
        f[1] = len(set(int(o.color) for o in objs)) / 16.0
        sizes = np.array([o.size for o in objs], dtype=np.float64) if objs else np.zeros(1)
        f[2] = float(sizes.max()) / n
        f[3] = float(sizes.mean()) / n
        f[4] = float(sizes.std()) / n
        f[5] = float((sizes == 1).sum()) / max(1.0, len(objs))
        f[6] = float((g == bg).sum()) / n
        if objs:
            cs = np.array([o.center for o in objs], dtype=np.float64)
            f[7] = float(cs.std()) / max(g.shape)
            f[8] = len(set(o.shape_class for o in objs)) / 8.0
            # adjacency: how many object pairs touch (a proxy for "assembled")
            occ = g != bg
            touch = float((occ[:, 1:] & occ[:, :-1]).sum() + (occ[1:, :] & occ[:-1, :]).sum())
            f[9] = touch / max(1.0, float(occ.sum()))
    except Exception:
        return np.zeros(len(OBJ_NAMES), dtype=np.float64)
    return np.clip(f, 0.0, 1.0)


# ---------------------------------------------------------------- monotonicity
def agreement(A: np.ndarray) -> np.ndarray:
    """Per feature: signed agreement in [-1, 1], or 0 when fit() would abstain.

    Exactly fit()'s rule -- net displacement sets the wanted direction, and the share of
    steps that MOVED and agreed with it must clear MONOTONE_MIN. Reimplemented rather
    than called so the diagnostic can be applied to a basis fit() has never seen."""
    out = np.zeros(A.shape[1], dtype=np.float64)
    if A.shape[0] < MIN_TRACE:
        return out
    net = A[-1] - A[0]
    d = np.diff(A, axis=0)
    for j in range(A.shape[1]):
        moved = d[:, j][d[:, j] != 0.0]
        if net[j] == 0.0 or moved.size < MIN_TRACE - 1:
            continue
        want = 1.0 if net[j] > 0 else -1.0
        ag = float((np.sign(moved) == want).mean())
        if ag >= MONOTONE_MIN:
            out[j] = want * ag
    return out


def matrix(grids, featfn) -> np.ndarray:
    return np.vstack([featfn(g) for g in grids]) if len(grids) else np.zeros((0, 1))


def report(tag, names, featfn, wins, deaths):
    print(f"\n  --- basis: {tag}  ({len(names)} features) ---")
    win_ag = [agreement(matrix(w, featfn)) for w in wins]
    death_ag = [agreement(matrix(d, featfn)) for d in deaths]
    if not win_ag:
        # No win in this game -- still worth knowing whether DEATHS alone contain
        # monotone structure, because that is the signal A1 works from.
        if death_ag:
            D = np.vstack(death_ag)
            shared = int((np.abs(np.sign(D).sum(axis=0)) >= max(2, 0.7 * len(death_ag))).sum())
            print(f"    no win. deaths={len(death_ag)}  "
                  f"features monotone on >=70% of deaths (same sign): {shared}")
        return None
    W = np.vstack(win_ag)
    wmask = np.abs(W).max(axis=0) > 0
    print(f"    Q1  wins={len(win_ag)}  features with a direction from the WIN: "
          f"{int(wmask.sum())}/{len(names)}")
    if not death_ag:
        print("    (no deaths recorded -- Q2 not answerable)")
        return None
    D = np.vstack(death_ag)
    wsign = np.sign(W.sum(axis=0))
    conf, disc = [], []
    for j in range(len(names)):
        if not wmask[j]:
            continue
        same = int((np.sign(D[:, j]) == wsign[j]).sum())
        rate = same / len(death_ag)
        (conf if rate >= 0.5 else disc).append((names[j], rate, float(wsign[j])))
    print(f"    Q2  deaths={len(death_ag)}")
    print(f"        CONFOUNDED (same direction on >=50% of deaths -> elapsed time, "
          f"not progress): {len(conf)}")
    for nm, r, s in conf:
        print(f"            {nm:<14} win_dir={s:+.0f}  matches {r*100:4.0f}% of deaths")
    print(f"        DISCRIMINATIVE (rises to a win, not to a death): {len(disc)}")
    for nm, r, s in disc:
        print(f"            {nm:<14} win_dir={s:+.0f}  matches {r*100:4.0f}% of deaths")
    return len(disc)


def dedupe(traces):
    """Distinct trajectories only.

    lp85 records 11 GAME_OVERs but only 7 distinct graph paths -- five deaths walk the
    SAME path. Counting them as 11 independent observations would inflate every
    agreement statistic here, which is the exact error this diagnostic exists to catch
    elsewhere. n is the number of distinct paths, not the number of episodes."""
    seen, out = set(), []
    for t in traces:
        k = t.tobytes()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def main(root):
    files = sorted(glob.glob(os.path.join(root, "*.npz")))
    if not files:
        print(f"no dumps under {root}")
        return
    by_game = defaultdict(lambda: {"win": [], "death": []})
    for p in files:
        base = os.path.basename(p)[:-4]
        gid, _n, outcome = base.split("_", 2)
        z = np.load(p, allow_pickle=True)
        by_game[gid][outcome].append((z["walk"], z["graph"]))

    for gid in sorted(by_game):
        eps = by_game[gid]
        print("=" * 74)
        print(f"GAME {gid}:  win episodes={len(eps['win'])}  "
              f"death episodes={len(eps['death'])}")
        for outcome in ("win", "death"):
            for walk, graph in eps[outcome][:14]:
                print(f"    {outcome:<5} walk={walk.shape[0]:<5} graph={graph.shape[0]}")
        # Two bases, because they answer different questions. GRAPH is what fit() is fed
        # (deliberate steps only, but heavily deduplicated). WALK is every board the
        # planner stood on -- noisier per step, far more independent episodes.
        for src, idx in (("graph (what fit() sees)", 1), ("walk (every frame)", 0)):
            wins = dedupe([e[idx] for e in eps["win"] if e[idx].shape[0] >= MIN_TRACE])
            deaths = dedupe([e[idx] for e in eps["death"] if e[idx].shape[0] >= MIN_TRACE])
            print(f"\n  ################ trace source: {src} -- "
                  f"distinct wins={len(wins)} distinct deaths={len(deaths)}")
            report("current (colour histogram + 4)", FEAT_NAMES,
                   ProgressModel.features, wins, deaths)
            report("C1 object-level", OBJ_NAMES, object_features, wins, deaths)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "o3dump"))
