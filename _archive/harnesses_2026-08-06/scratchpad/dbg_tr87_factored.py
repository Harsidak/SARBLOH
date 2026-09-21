import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from local_eval import run_game, GAMES_DIR, default_agent_factory

def diff_grids(g1, g2):
    if g1.shape != g2.shape:
        return []
    diffs = []
    for r in range(g1.shape[0]):
        for c in range(g1.shape[1]):
            if g1[r, c] != g2[r, c]:
                diffs.append((r, c, g1[r, c], g2[r, c]))
    return diffs

def factory(game_id):
    agent = default_agent_factory(game_id)
    orig_choose = agent.choose_action.__func__

    def patched_choose(self, frames, latest_frame):
        step = agent.action_counter
        action = orig_choose(self, frames, latest_frame)
        if step > 0 and step <= 20:
            grid_prev = self._parse_grid(frames[-1], frames) if len(frames) > 0 else self._parse_grid(latest_frame, frames)
            grid_now = self._parse_grid(latest_frame, frames)
            diffs = diff_grids(grid_prev, grid_now)
            print(f"[{step}] action={agent.last_action_str} diffs: {diffs}")
        return action

    agent.choose_action = patched_choose.__get__(agent)
    return agent

if __name__ == "__main__":
    gid = "tr87"
    steps = 25
    import glob as _g
    hits = _g.glob(os.path.join(ROOT, "eval", "real_games", gid, "*", gid + ".py"))
    if not hits:
        print("Game not found!")
        sys.exit(1)
    gf = hits[0]
    print(f"Running {gf}")
    r = run_game(gf, factory, seed=0, max_actions=steps)
    print(r)
