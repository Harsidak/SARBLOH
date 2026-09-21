"""Why did remap_colors never promote on a real game?

Read-only probe. Runs the real agent for a short budget, then -- OUTSIDE the
loop, on the transitions it collected -- asks the synthesizer three questions
per action:

  1. does the data even PROPOSE a colour map?   (_remap_candidates)
  2. how well does that map score?              (_score)
  3. what rule wins instead, and by how much?   (the full candidate sweep)

Nothing here changes the agent: the probe subclasses MyAgent only to keep a
handle on the live instance, exactly as diag_eval does.

Usage:
  PYTHONHASHSEED=0 python scratchpad/probe_remap.py [ids...] [--actions 400]
"""
import os
import sys
import glob
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

import local_eval as LE                                    # noqa: E402
from my_agent import MyAgent, base_action                  # noqa: E402

_LAST = {}


class ProbeAgent(MyAgent):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        _LAST["agent"] = self


def factory(game_id):
    return ProbeAgent(game_id=game_id)


def report(gid, ag):
    syn = ag.world_model.enumerator
    ev = syn.evidence(ag.transitions)
    print(f"\n=== {gid}: {len(ag.transitions)} transitions, "
          f"{sum(1 for t in ag.transitions if t.changed)} changed, "
          f"{len(ev)} actions with evidence ===")
    if not ev:
        print("  (no fittable evidence at all -- click-only or nothing moved)")
        return
    for act, (samples, mask) in sorted(ev.items()):
        use = [(t.prev, t.next) for t in samples][-5:]
        if len(use) < 2:
            print(f"  {act}: only {len(use)} sample(s)")
            continue
        bgs = [syn.encoder.detect_background(p) for p, _ in use]
        maps = syn._remap_candidates(use, mask)
        line = f"  {act}: {len(use)} samples, mask={'yes' if mask is not None else 'RAW'}"
        if not maps:
            print(line + " | remap PROPOSED: nothing")
        else:
            for m in maps:
                s = syn._score("remap_colors", {"mapping": m}, use, bgs, mask=mask)
                short = dict(list(m.items())[:6])
                print(line + f" | remap {short}{'...' if len(m) > 6 else ''}"
                             f" scores {s}/{len(use)}")
                line = " " * len(line)
        best_s, best = 0, None
        for op, args in syn._candidate_ops(use, sorted({c for t in samples
                                                        for c in t.palette}),
                                           bgs[0], mask=mask):
            sc = syn._score(op, args, use, bgs, floor=best_s, mask=mask)
            if sc > best_s:
                best_s, best = sc, (op, args)
        print(f"      winner: {best[0] if best else None} "
              f"{str(best[1])[:70] if best else ''} -> {best_s}/{len(use)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*")
    ap.add_argument("--actions", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    files = [f for f in LE._discover_real_games()
             if not a.games
             or os.path.splitext(os.path.basename(f))[0] in set(a.games)]
    for gf in files:
        gid = os.path.splitext(os.path.basename(gf))[0]
        LE.run_game(gf, factory, seed=a.seed, max_actions=a.actions, verbose=False)
        ag = _LAST.get("agent")
        if ag is not None:
            try:
                report(gid, ag)
            except Exception as e:
                print(f"  {gid}: probe failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
