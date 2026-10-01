"""B2 falsification: do MONOTONE, NON-PERIODIC counter cells actually exist?

B2's claim is that a game's own HUD already contains a progress meter that can be
grounded with ZERO wins. The claim has three parts and each can be refuted offline from
the ARC_O3_DUMP episodes, before any agent code is written:

  P1  Do cells exist that change several times within an episode and NEVER revert?
      (monotone cells -- candidate counters)
  P2  Of those, how many are NOT clock-periodic? A cell that ticks once per action is a
      timer and carries no information about progress. This reuses the same periodicity
      test StateGraph's HUD mask already validates on tr87/tu93/ls20: a clock's change
      gaps are dominated by ONE gap length.
  P3  Can the SIGN be resolved? "3 of 5 collected" and "5 lives left" are both monotone;
      only the direction differs, and getting it backwards is worse than abstaining. The
      proposed discriminator is the death labels: if a counter sits at its LOW extreme
      when episodes end in GAME_OVER, then low is bad and up is progress.

If P1 or P2 comes back empty, B2 is dead and no code should be written for it.

Usage:  python scratchpad/diag_b2.py scratchpad/o3dump
"""
import os
import sys
import glob
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("ARC_AGENT_SEED", "0")

from my_agent import StateGraph      # noqa: E402  (for the periodicity constants)

MIN_CHANGES = 3          # a cell that ticked twice is not yet a counter
PERIODIC_SHARE = StateGraph.MASK_LINE_PERIODIC
MIN_CHG_FOR_PERIOD = StateGraph.MASK_LINE_MIN_CHG


def analyse(frames):
    """Per cell over one episode: changes, whether it ever REVISITS a value, and the
    multiset of gaps between changes.

    "Never revisits" and not "moves monotonically in colour value". A counter on screen
    is a rendered digit, and the glyphs for 2, 3, 4 are not ordered in colour space --
    testing sign(delta) asks whether the SPRITE ATLAS happens to be sorted, which is a
    fact about the artist, not about the game. The order-theoretic version (a progress
    cell advances through states and never returns to one it has left) is what a counter,
    a filling bar and a one-way door all satisfy, in any palette."""
    A = np.asarray(frames, dtype=np.int16)
    if A.shape[0] < MIN_CHANGES + 1:
        return None
    T, H, W = A.shape
    d = A[1:] != A[:-1]
    changes = d.sum(axis=0)
    revisit = np.zeros((H, W), dtype=bool)
    lastt = np.full((H, W), -1, dtype=np.int32)
    gaps = defaultdict(lambda: defaultdict(int))
    # seen[(v)] -> boolean grid of "cell has held value v at some earlier time"
    seen_vals: dict = {}
    prev = A[0]
    for v in np.unique(prev):
        seen_vals[int(v)] = (prev == v)
    for t in range(1, T):
        cur = A[t]
        hit = cur != prev
        for v in np.unique(cur[hit]) if hit.any() else ():
            m = seen_vals.get(int(v))
            if m is not None:
                revisit |= hit & (cur == v) & m
        for v in np.unique(cur):
            m = seen_vals.get(int(v))
            seen_vals[int(v)] = (cur == v) if m is None else (m | (cur == v))
        if hit.any():
            idx = np.argwhere(hit)
            for (i, j) in idx:
                pt = int(lastt[i, j])
                if pt >= 0:
                    gaps[(int(i), int(j))][t - pt] += 1
                lastt[i, j] = t
        prev = cur
    return changes, revisit.astype(np.int32), gaps, A


def periodic(g):
    tot = sum(g.values())
    return tot + 1 >= MIN_CHG_FOR_PERIOD and max(g.values()) / tot >= PERIODIC_SHARE


def main(root):
    files = sorted(glob.glob(os.path.join(root, "*.npz")))
    by_game = defaultdict(list)
    for p in files:
        base = os.path.basename(p)[:-4]
        gid, _n, outcome = base.split("_", 2)
        by_game[gid].append((outcome, np.load(p, allow_pickle=True)["walk"]))

    for gid in sorted(by_game):
        eps = by_game[gid]
        print("=" * 74)
        print(f"GAME {gid}: {len(eps)} episodes")
        # A cell is a counter candidate only if it behaves monotonically in EVERY
        # episode it moved in -- one revert anywhere disqualifies it. This is the strict
        # reading on purpose: a "counter" that sometimes goes backwards is a sprite.
        mono_ok = None
        mono_bad = None
        per_cell_gaps = defaultdict(lambda: defaultdict(int))
        n_used = 0
        for outcome, frames in eps:
            r = analyse(frames)
            if r is None:
                continue
            n_used += 1
            changes, reversals, gaps, _A = r
            good = (changes >= MIN_CHANGES) & (reversals == 0)
            bad = reversals > 0
            mono_ok = good if mono_ok is None else (mono_ok | good)
            mono_bad = bad if mono_bad is None else (mono_bad | bad)
            for k, g in gaps.items():
                for gap, c in g.items():
                    per_cell_gaps[k][gap] += c
        if mono_ok is None:
            print("   no usable episodes")
            continue
        cand = mono_ok & ~mono_bad
        n_cand = int(cand.sum())
        print(f"   P1  never-revisiting cells (>= {MIN_CHANGES} changes, no value ever repeats, "
              f"across {n_used} episodes): {n_cand}")
        if not n_cand:
            print("   P2  -- (nothing to test)")
            continue
        cells = [tuple(x) for x in np.argwhere(cand)]
        nonper = [c for c in cells if not periodic(per_cell_gaps[c])]
        print(f"   P2  of those, NOT clock-periodic (event counters, not timers): "
              f"{len(nonper)}")
        if nonper:
            rows = sorted(set(c[0] for c in nonper))
            print(f"       rows touched: {rows[:12]}{'...' if len(rows) > 12 else ''}")
            # P3: where does the counter sit when the episode ends badly?
            deaths = [f for o, f in eps if o == "death"]
            wins = [f for o, f in eps if o == "win"]
            for label, group in (("death", deaths), ("win", wins)):
                if not group:
                    continue
                fin, rng = [], []
                for frames in group:
                    A = np.asarray(frames, dtype=np.int16)
                    if A.shape[0] < 2:
                        continue
                    for (i, j) in nonper[:200]:
                        col = A[:, i, j]
                        lo, hi = int(col.min()), int(col.max())
                        if hi > lo:
                            fin.append((int(col[-1]) - lo) / (hi - lo))
                            rng.append(hi - lo)
                if fin:
                    print(f"       P3  {label}: final value sits at "
                          f"{np.mean(fin)*100:.0f}% of its observed range "
                          f"(n={len(fin)} cell-episodes)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "o3dump"))
