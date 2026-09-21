"""Instrument choose_action routing on any game: which path drives each action.

Logs per-action:
  [step] PATH=<path_name> action=<action_str> mech=<bool>

Run from ARC_AGI EXPERMENTATIONS/:
    PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 python -u scratchpad/dbg_routing.py [gid] [steps]
"""
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from local_eval import run_game, GAMES_DIR, default_agent_factory  # noqa: E402
import my_agent as ma  # noqa: E402

COUNTS = Counter()


def factory(game_id):
    agent = default_agent_factory(game_id)
    rp = agent.rplanner
    orig_choose = agent.choose_action.__func__

    orig_graph_explore = agent._graph_explore.__func__

    def patched_graph_explore(self, grid, valid):
        r = orig_graph_explore(self, grid, valid)
        if r is not None:
            self._graph_tag = "graph"
        return r

    agent._graph_explore = patched_graph_explore.__get__(agent)
    agent._graph_tag = None

    def patched_choose(self, frames, latest_frame):
        step = agent.action_counter
        mech = rp.looks_mechanical() if rp.avatar_color is not None else None
        agent._graph_tag = None

        action = orig_choose(self, frames, latest_frame)
        action_str = agent.last_action_str

        if agent._graph_tag == "graph":
            path = "graph-explore"
        elif mech:
            path = "novelty(mech)"
        else:
            path = "execplanner-or-novelty"

        COUNTS[path] += 1
        if step < 20 or step % 50 == 0:
            print(f"[{step:4d}] PATH={path:25s} action={action_str:12s} mech={mech}")

        return action

    agent.choose_action = patched_choose.__get__(agent)
    factory.agent = agent
    return agent


if __name__ == "__main__":
    gid = sys.argv[1] if len(sys.argv) > 1 else "keydoor"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    gf = os.path.join(GAMES_DIR, gid + ".py")
    if not os.path.exists(gf):
        import glob as _g
        hits = _g.glob(os.path.join(ROOT, "eval", "real_games", gid, "*", gid + ".py"))
        gf = hits[0]
    r = run_game(gf, factory, seed=0, max_actions=steps)
    print(r)
    print(f"\nRouting distribution: {dict(sorted(COUNTS.items()))}")
