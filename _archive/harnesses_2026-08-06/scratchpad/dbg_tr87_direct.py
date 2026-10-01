import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from arc_agi.game import Game
from local_eval import default_agent_factory

def diff_grids(g1, g2):
    if g1.shape != g2.shape:
        return []
    diffs = []
    for r in range(g1.shape[0]):
        for c in range(g1.shape[1]):
            if g1[r, c] != g2[r, c]:
                diffs.append((r, c, g1[r, c], g2[r, c]))
    return diffs

gid = "tr87"
gf = os.path.join(ROOT, "eval", "real_games", gid, "1", gid + ".py")
if not os.path.exists(gf):
    import glob as _g
    hits = _g.glob(os.path.join(ROOT, "eval", "real_games", gid, "*", gid + ".py"))
    if hits: gf = hits[0]

game = Game(gf, seed=0)
agent = default_agent_factory(gid)

print("Starting game...")
grid_prev = None

for step in range(20):
    obs = game.get_observation()
    frames = obs['frames']
    latest_frame = frames[-1]
    
    grid_now = agent._parse_grid(latest_frame, frames)
    if grid_prev is not None:
        diffs = diff_grids(grid_prev, grid_now)
        print(f"[{step}] diffs: {diffs}")
    
    grid_prev = grid_now
    
    action_str = agent.choose_action(frames, latest_frame)
    print(f"[{step}] Chosen action: {action_str}")
    
    # find the action object
    action = None
    for a in latest_frame.available_actions:
        if a.name == action_str:
            action = a
            break
            
    if action is None:
        print("Invalid action returned")
        break
        
    game.step(action)
    if game.is_done():
        print("Game Over")
        break
