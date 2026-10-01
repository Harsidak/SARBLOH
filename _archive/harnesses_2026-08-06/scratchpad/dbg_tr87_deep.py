"""
Deep diagnostic for tr87 mechanics.

tr87 is a PUZZLE game with:
- Puzzle pieces (nxkictbbvzt*) displayed in two rows: upper row (zvojhrjxxm) 
  and lower row (ztgmtnnufb)
- A cursor (qvtymdcqear) that highlights one piece in the lower row
- ACTION3/4: move cursor left/right among lower-row pieces
- ACTION1/2: rotate the currently-selected lower-row piece (cycle through 7 variants)
- Win condition: upper row pattern matches lower row pattern (bsqsshqpox)
- Each action costs 1 from a budget (upmkivwyrxz); budget=0 -> lose

The agent sees a 64x64 grid with:
- Background color 2 (teal)
- Color blocks (10=orange, 7=blue, 11=purple) as grid backgrounds
- Puzzle pieces rendered as 5x5 sprites of color 5 with various shapes
- Cursor shown as bracket-like sprites above/below the selected piece

KEY INSIGHT: The grid CHANGES on every action (cursor moves or piece rotates),
but the debug showed "diffs: []" for all 20 actions. This means _parse_grid 
is failing or the frames aren't being compared correctly.

Let's trace what the raw frames look like.
"""

import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from local_eval import run_game, GAMES_DIR, default_agent_factory

def factory(game_id):
    agent = default_agent_factory(game_id)
    orig_choose = agent.choose_action.__func__
    orig_update = agent.update.__func__
    step_counter = [0]
    
    def patched_update(self, *args, **kwargs):
        result = orig_update(self, *args, **kwargs)
        step_counter[0] += 1
        step = step_counter[0]
        if step <= 30:
            # Print the agent's internal state
            grid = getattr(self, 'last_grid', None)
            if grid is not None:
                unique = np.unique(grid)
                print(f"  [update {step}] grid shape={grid.shape} unique_colors={sorted(unique.tolist())}")
                # Check what the parsed grid looks like - find non-background regions
                mask = (grid != 2) & (grid != 3)  # not teal, not padding
                if mask.any():
                    rows, cols = np.where(mask)
                    print(f"    non-bg region: rows [{rows.min()}-{rows.max()}] cols [{cols.min()}-{cols.max()}]")
            
            # Print StateGraph info
            sg = getattr(self, 'sgraph', None)
            if sg is not None:
                print(f"    sgraph: states={len(sg._nodes)} edges={sum(len(v) for v in sg._nodes.values())}")
            
            # Print PWM info  
            pwm = getattr(self, 'pwm', None)
            if pwm is not None:
                print(f"    pwm: temporal_stats={len(pwm.temporal_stats)} single_stats={len(pwm.single_stats)} trained={pwm.trained}")
                
            # Print what ExecPlanner thinks
            rp = getattr(self, 'rplanner', None)
            if rp is not None:
                print(f"    rplanner: mechanical={rp.looks_mechanical()} avatar={rp.avatar_color}")
                
        return result
    
    def patched_choose(self, frames, latest_frame):
        step = step_counter[0]
        action = orig_choose(self, frames, latest_frame)
        if step <= 30:
            astr = self.last_action_str
            grid = self.last_grid
            # Compare current grid to previous  
            prev = getattr(self, '_prev_dbg_grid', None)
            if prev is not None and prev.shape == grid.shape:
                diff = (prev != grid)
                n_diff = diff.sum()
                print(f"  [choose {step}] action={astr} grid_diffs={n_diff}")
                if n_diff > 0 and n_diff < 50:
                    rows, cols = np.where(diff)
                    for r, c in zip(rows[:10], cols[:10]):
                        print(f"    ({r},{c}): {prev[r,c]} -> {grid[r,c]}")
            else:
                print(f"  [choose {step}] action={astr} (first frame)")
            self._prev_dbg_grid = grid.copy()
        return action
    
    agent.update = patched_update.__get__(agent)
    agent.choose_action = patched_choose.__get__(agent)
    return agent

if __name__ == "__main__":
    import glob as _g
    gid = "tr87"
    hits = _g.glob(os.path.join(ROOT, "eval", "real_games", gid, "*", gid + ".py"))
    if not hits:
        print("Game not found!")
        sys.exit(1)
    gf = hits[0]
    print(f"Running {gf}")
    r = run_game(gf, factory, seed=0, max_actions=40)
    print(f"\nResult: {r}")
