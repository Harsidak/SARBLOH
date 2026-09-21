"""Blast-radius probe for StateEncoder.detect_background.

Collects a REAL frame corpus (random actions in every competition game), then
compares the OLD background rule against the CURRENT one frame by frame. Cheap
(env stepping only, no agent thinking) and it answers the question a unit test
cannot: on real frames, how often does the frame rule fire, and did this change
actually move anything?

Run:  PYTHONIOENCODING=utf-8 python scratchpad/probe_bg_corpus.py [n_steps]
"""
import os
import sys
import glob
import random
import logging

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.dirname(HERE)
sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, os.path.join(AGENT_DIR, "eval"))

from arc_agi.local_wrapper import LocalEnvironmentWrapper
from arc_agi.models import EnvironmentInfo
from local_eval import _class_name_from_file, _discover_real_games
from my_agent import StateEncoder, MyAgent

E = StateEncoder()
N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def old_detect_background(grid):
    """The rule as it stood before the FRAME_CONCENTRATION fix (2026-07-29)."""
    if grid.size == 0:
        return 0
    h, w = grid.shape
    if h == 0 or w == 0:
        return 0
    border = (np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
              if h > 2 and w > 2 else grid.flatten())
    if len(border) == 0:
        return 0
    u, c = np.unique(border, return_counts=True)
    bbg = int(u[c.argmax()])
    bf = c.max() / len(border)
    if bf >= 0.85 and h > 2 and w > 2:
        it = grid[1:-1, 1:-1]
        if it.size >= 4:
            iu, ic = np.unique(it, return_counts=True)
            ibg = int(iu[ic.argmax()])
            inside = int(ic[iu == bbg].sum())
            if ibg != bbg and ic.max() > it.size * 0.5 and inside < it.size * 0.5:
                return ibg
    if bf > 0.5:
        return bbg
    u, c = np.unique(grid, return_counts=True)
    return int(u[c.argmax()]) if len(u) else 0


def collect(game_file, n_steps, seed=0):
    from arcengine import GameAction
    gid = os.path.splitext(os.path.basename(game_file))[0]
    info = EnvironmentInfo(game_id=gid, class_name=_class_name_from_file(game_file),
                           local_dir=os.path.dirname(os.path.abspath(game_file)))
    lg = logging.getLogger("probe")
    lg.setLevel(logging.ERROR)
    w = LocalEnvironmentWrapper(info, lg, scorecard_id="probe", seed=seed)
    f0 = w.observation_space
    if f0 is None:
        return gid, []
    rng = random.Random(seed)
    grids, frame = [], f0
    avail = [a for a in (getattr(f0, "available_actions", None) or [])]
    for _ in range(n_steps):
        g = MyAgent._grid_from(frame)
        if g is not None:
            grids.append(g)
        act = rng.choice(avail) if avail else GameAction.ACTION1
        name = getattr(act, "name", str(act))
        data = {}
        if name in ("ACTION6", "ACTION7"):
            data = {"x": rng.randrange(0, 64), "y": rng.randrange(0, 64)}
        try:
            resp = w.step(act, data=data)
        except Exception:
            break
        if resp is None:
            break
        frame = resp
    return gid, grids


def main():
    total = changed = fired_new = fired_old = 0
    rows = []
    for gf in _discover_real_games():
        gid, grids = collect(gf, N_STEPS)
        c = f_new = f_old = 0
        example = None
        for g in grids:
            o, n = old_detect_background(g), E.detect_background(g)
            # "fired" = the frame rule overrode the plain border-majority answer
            plain = _plain(g)
            f_old += (o != plain)
            f_new += (n != plain)
            if o != n:
                c += 1
                if example is None:
                    example = (g.shape, o, n)
        total += len(grids)
        changed += c
        fired_new += f_new
        fired_old += f_old
        rows.append((gid, len(grids), c, f_old, f_new, example))
        print(f"  {gid:<6} frames={len(grids):<4} bg_changed={c:<4} "
              f"frame_rule_fired old={f_old:<4} new={f_new:<4} {example or ''}", flush=True)
    print(f"\nTOTAL frames={total}  bg_changed={changed} "
          f"({100.0 * changed / max(1, total):.1f}%)  "
          f"frame_rule_fired old={fired_old} new={fired_new}")


def _plain(grid):
    """Border-majority answer with NO frame rule -- the thing the rule overrides."""
    if grid.size == 0:
        return 0
    h, w = grid.shape
    border = (np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
              if h > 2 and w > 2 else grid.flatten())
    if len(border) == 0:
        return 0
    u, c = np.unique(border, return_counts=True)
    if c.max() / len(border) > 0.5:
        return int(u[c.argmax()])
    u, c = np.unique(grid, return_counts=True)
    return int(u[c.argmax()]) if len(u) else 0


if __name__ == "__main__":
    main()
