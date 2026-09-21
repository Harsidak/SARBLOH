"""Evidence probe for COMPONENT 1: what IS the giant phantom object?

Peeling chrome was the wrong hypothesis (probe_playarea.py: helps lp85, hurts
cn04/ka59/ft09/tn36/sk48, leaves bp35/tr87/m0r0/dc22 untouched). So look at the
structure directly: for the top colors by pixel count, report how that color is
laid out -- number of 4-connected components, largest component, and how much of
the grid border it owns.

A real BACKGROUND is spread out: it owns a lot of area and touches everywhere.
Chrome is a band: high border share, small interior share. A floor mis-labelled
as an object shows up as one huge component of a color that is NOT the detected
bg. This tells us which of those three things the agent is currently choosing.

Run:  PYTHONIOENCODING=utf-8 python scratchpad/probe_bgcandidates.py [n_steps]
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
N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 15


def components(mask):
    """Sizes of 4-connected components of a boolean mask, descending."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q, n = [(r, c)], 0
                seen[r, c] = True
                while q:
                    cr, cc = q.pop()
                    n += 1
                    for nr, nc in ((cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)):
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                out.append(n)
    out.sort(reverse=True)
    return out


def report(gid, grid):
    h, w = grid.shape
    bg = E.detect_background(grid)
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    u, cnt = np.unique(grid, return_counts=True)
    order = np.argsort(-cnt)[:4]
    print(f"  {gid:<6} shape={grid.shape} detected_bg={bg}")
    for i in order:
        col = int(u[i])
        m = grid == col
        comps = components(m)
        bshare = float((m & border).sum()) / int(border.sum())
        print(f"      color {col:<3} px={int(cnt[i]):<5} ({cnt[i]/grid.size:.2f})  "
              f"comps={len(comps):<4} largest={comps[0]:<5} "
              f"border_share={bshare:.2f}{'   <-- detected bg' if col == bg else ''}")


def main():
    for gf in _discover_real_games():
        gid, grids = collect(gf, N_STEPS)
        if grids:
            report(gid, grids[-1])


if __name__ == "__main__":
    main()
