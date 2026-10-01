"""
Diagnostic: render tr87 level 1 grid and identify icon regions.
Goal: understand the grid layout so we can build a visual pattern matcher.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from local_eval import run_game, default_agent_factory
import glob

def factory(game_id):
    agent = default_agent_factory(game_id)
    orig_choose = agent.choose_action.__func__
    call_count = [0]
    
    def patched_choose(self, frames, latest_frame):
        call_count[0] += 1
        step = call_count[0]
        grid = self._parse_grid(latest_frame, frames)
        
        if step == 1:
            print(f"Grid shape: {grid.shape}")
            print(f"Unique colors: {sorted(np.unique(grid).tolist())}")
            
            # Find all color-5 (icon) pixels
            mask5 = (grid == 5)
            ys, xs = np.where(mask5)
            if len(ys) > 0:
                print(f"\nColor-5 pixels: {len(ys)} total")
                print(f"  Y range: {ys.min()}-{ys.max()}")
                print(f"  X range: {xs.min()}-{xs.max()}")
            
            # Find icon regions by clustering color-5 pixels
            # Simple approach: group by proximity
            regions = _find_icon_regions(grid)
            print(f"\nFound {len(regions)} icon regions:")
            for i, (r_min, c_min, r_max, c_max) in enumerate(regions):
                h = r_max - r_min + 1
                w = c_max - c_min + 1
                pattern = grid[r_min:r_max+1, c_min:c_max+1]
                fingerprint = tuple((pattern == 5).ravel().tolist())
                print(f"  Icon {i}: rows [{r_min}-{r_max}] cols [{c_min}-{c_max}] size {h}x{w} fp_hash={hash(fingerprint) % 10000}")
            
            # Group by Y
            row_groups = {}
            for i, (r_min, c_min, r_max, c_max) in enumerate(regions):
                y_center = (r_min + r_max) // 2
                # Round to nearest row
                matched = False
                for ry in row_groups:
                    if abs(y_center - ry) < 4:
                        row_groups[ry].append(i)
                        matched = True
                        break
                if not matched:
                    row_groups[y_center] = [i]
            
            print(f"\nRows (by Y center):")
            for ry in sorted(row_groups.keys()):
                idxs = row_groups[ry]
                cols = [regions[i][1] for i in idxs]
                print(f"  y~{ry}: {len(idxs)} icons, x-positions: {sorted(cols)}")
            
            # Find the bar/separator regions (color 0 with specific patterns)
            mask0 = (grid == 0)
            print(f"\nColor-0 pixels: {mask0.sum()}")
            
            # Find color-3 horizontal bars
            mask3 = (grid == 3)
            # Look for horizontal runs of color 3 that are >5 pixels wide
            bars = []
            for r in range(grid.shape[0]):
                in_run = False
                run_start = 0
                for c in range(grid.shape[1]):
                    if mask3[r, c]:
                        if not in_run:
                            in_run = True
                            run_start = c
                    else:
                        if in_run:
                            run_len = c - run_start
                            if run_len >= 8:  # bars are 11 pixels wide
                                bars.append((r, run_start, c-1, run_len))
                            in_run = False
                if in_run:
                    run_len = grid.shape[1] - run_start
                    if run_len >= 8:
                        bars.append((r, run_start, grid.shape[1]-1, run_len))
            
            # Deduplicate bars (they span 2 rows)
            unique_bars = []
            for r, c1, c2, l in bars:
                dup = False
                for ur, uc1, uc2, ul in unique_bars:
                    if abs(r - ur) <= 1 and abs(c1 - uc1) <= 1:
                        dup = True
                        break
                if not dup:
                    unique_bars.append((r, c1, c2, l))
            
            print(f"\nHorizontal bars (color 3/0 runs):")
            for r, c1, c2, l in unique_bars:
                print(f"  row {r}, cols [{c1}-{c2}], len {l}")
            
            # Find cursor brackets (look for color 0 in bottom section near icons)
            # The cursor is made of qvtymdcqear sprites which have color 0
            print(f"\nLooking for cursor (color 0 patterns near bottom icons)...")
            for r in range(30, grid.shape[0]):
                row_0 = np.where(grid[r] == 0)[0]
                if len(row_0) >= 3 and len(row_0) <= 10:
                    print(f"  Row {r}: color-0 at cols {row_0.tolist()}")
            
            # Print a compact view of the grid
            print(f"\n--- Grid compact view (every 2nd row/col, 32x32) ---")
            alpha = ".0123456789abcde"
            for r in range(0, min(grid.shape[0], 64), 1):
                row_str = ""
                for c in range(0, min(grid.shape[1], 64), 1):
                    v = int(grid[r, c])
                    if v < len(alpha):
                        row_str += alpha[v]
                    else:
                        row_str += chr(ord('A') + v - 16) if v < 42 else '?'
                print(f"  {r:2d}: {row_str}")
        
        # Just do random actions to keep the game going
        action = orig_choose(self, frames, latest_frame)
        return action
    
    agent.choose_action = patched_choose.__get__(agent)
    return agent


def _find_icon_regions(grid):
    """Find all 5x5-ish icon regions (clusters of color 5)."""
    mask = (grid == 5).astype(np.int32)
    if not mask.any():
        return []
    
    # Simple flood fill to find connected components
    visited = np.zeros_like(mask, dtype=bool)
    regions = []
    
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c] and not visited[r, c]:
                # BFS flood fill
                component = []
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    component.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                            if mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                
                ys = [p[0] for p in component]
                xs = [p[1] for p in component]
                r_min, r_max = min(ys), max(ys)
                c_min, c_max = min(xs), max(xs)
                h = r_max - r_min + 1
                w = c_max - c_min + 1
                if h <= 6 and w <= 6 and len(component) >= 5:
                    regions.append((r_min, c_min, r_max, c_max))
    
    return regions


if __name__ == "__main__":
    gid = "tr87"
    hits = glob.glob(os.path.join(ROOT, "eval", "real_games", gid, "*", gid + ".py"))
    if not hits:
        print("Game not found!")
        sys.exit(1)
    gf = hits[0]
    print(f"Running {gf}\n")
    r = run_game(gf, factory, seed=0, max_actions=5)
    print(f"\nResult: {r}")
