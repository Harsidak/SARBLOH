import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from local_eval import run_game, default_agent_factory

def factory(game_id):
    agent = default_agent_factory(game_id)
    orig_choose = agent.choose_action.__func__

    def patched_choose(self, frames, latest_frame):
        # We just want to count the sprites.
        import collections
        c = collections.Counter()
        for s in latest_frame.sprites:
            c[s.name] += 1
        print(f"Level {self.level_steps}: {c}")
        sys.exit(0)

    agent.choose_action = patched_choose.__get__(agent)
    return agent

if __name__ == "__main__":
    gid = "tr87"
    import glob as _g
    hits = _g.glob(os.path.join(ROOT, "eval", "real_games", gid, "*", gid + ".py"))
    if hits:
        run_game(hits[0], factory, seed=0, max_actions=10)
