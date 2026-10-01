"""Evidence probe for COMPONENT 1: is there a peelable chrome margin?

The 643-frame corpus showed the outer HUD/chrome margin winning the border vote,
so the play-field floor comes back as ONE phantom object (lp85 1962px = 48% of
the grid). Before writing any play-area rule, measure the candidate:

    peel edge rows/cols that are UNIFORM (a single color across the whole line),
    repeatedly, from all four sides.

A HUD band is a solid strip; a play field with anything in it is not. This says
nothing game-specific. What we need to know is (a) how often it peels at all,
(b) what it does to bg coverage and to the largest "object", and (c) that it
does NOT eat the play field (guard: keep >=25% of the area).

Run:  PYTHONIOENCODING=utf-8 python scratchpad/probe_playarea.py [n_steps]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.dirname(HERE)
sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, os.path.join(AGENT_DIR, "eval"))

from local_eval import _discover_real_games
from my_agent import StateEncoder
from probe_bg_corpus import collect

E = StateEncoder()
N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
MIN_KEEP = 0.25


def peel(grid):
    """Strip uniform edge lines from all four sides. Returns (r0, c0, r1, c1)."""
    h, w = grid.shape
    r0, c0, r1, c1 = 0, 0, h, w
    changed = True
    while changed and (r1 - r0) > 2 and (c1 - c0) > 2:
        changed = False
        for _ in range(4):
            if r1 - r0 > 2 and len(np.unique(grid[r0, c0:c1])) == 1:
                r0 += 1; changed = True
            if r1 - r0 > 2 and len(np.unique(grid[r1 - 1, c0:c1])) == 1:
                r1 -= 1; changed = True
            if c1 - c0 > 2 and len(np.unique(grid[r0:r1, c0])) == 1:
                c0 += 1; changed = True
            if c1 - c0 > 2 and len(np.unique(grid[r0:r1, c1 - 1])) == 1:
                c1 -= 1; changed = True
    if (r1 - r0) * (c1 - c0) < MIN_KEEP * grid.size:
        return 0, 0, h, w
    return r0, c0, r1, c1


def stats(grid):
    bg = E.detect_background(grid)
    cov = float((grid == bg).sum()) / grid.size
    objs = E.objects(grid, bg=bg)
    largest = max((o.size for o in objs), default=0)
    return bg, cov, len(objs), largest


def main():
    for gf in _discover_real_games():
        gid, grids = collect(gf, N_STEPS)
        if not grids:
            print(f"  {gid:<6} no frames")
            continue
        n_peeled = 0
        rows = []
        for g in grids:
            r0, c0, r1, c1 = peel(g)
            sub = g[r0:r1, c0:c1]
            if sub.shape != g.shape:
                n_peeled += 1
            rows.append((stats(g), stats(sub), sub.shape))
        (bg0, cov0, n0, big0), (bg1, cov1, n1, big1), shp = rows[-1]
        print(f"  {gid:<6} frames={len(grids):<3} peeled={n_peeled:<3} "
              f"| full  bg={bg0} cov={cov0:.2f} objs={n0:<4} big={big0:<5} "
              f"| play {str(shp):<9} bg={bg1} cov={cov1:.2f} objs={n1:<4} big={big1}",
              flush=True)


if __name__ == "__main__":
    main()
