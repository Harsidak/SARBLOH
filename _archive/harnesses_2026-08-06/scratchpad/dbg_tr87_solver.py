"""
Prototype CursorPuzzleSolver for tr87 - v4.
KEY FIX: Icons are randomly rotated. Must use rotation-invariant matching.
Generate all 4 rotations of each fingerprint and match across rotations.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from local_eval import run_game, default_agent_factory
import glob


def find_icon_regions(grid, icon_color=5):
    """Find all 5x5-ish icon regions (clusters of icon_color)."""
    mask = (grid == icon_color).astype(np.int32)
    if not mask.any():
        return []
    visited = np.zeros_like(mask, dtype=bool)
    regions = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c] and not visited[r, c]:
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


def icon_fingerprint(grid, bbox, icon_color=5):
    """Extract a fingerprint: the 5x5 binary mask of icon_color pixels as a 2D numpy array."""
    r_min, c_min, r_max, c_max = bbox
    patch = grid[r_min:r_max+1, c_min:c_max+1]
    return (patch == icon_color).astype(np.int8)


def canonical_fingerprint(fp):
    """Return the canonical (rotation-invariant) version of a fingerprint.
    The canonical form is the lexicographically smallest among all 4 rotations."""
    rotations = [fp]
    current = fp
    for _ in range(3):
        current = np.rot90(current)
        rotations.append(current.copy())
    return min(rotations, key=lambda x: tuple(x.ravel().tolist()))


def fp_hash(fp):
    """Hash a fingerprint for display."""
    return hash(tuple(fp.ravel().tolist())) % 10000


def group_into_rows(regions):
    """Group icon regions by Y coordinate."""
    rows = {}
    for i, (r_min, c_min, r_max, c_max) in enumerate(regions):
        y_center = (r_min + r_max) // 2
        matched_key = None
        for ry in rows:
            if abs(y_center - ry) < 4:
                matched_key = ry
                break
        if matched_key is not None:
            rows[matched_key].append(i)
        else:
            rows[y_center] = [i]
    return rows


def find_cursor_position(grid, ctrl_regions, cursor_color=0):
    H = grid.shape[0]
    for r in range(H // 2, H - 1):
        cols_0 = np.where(grid[r] == cursor_color)[0]
        if 3 <= len(cols_0) <= 10:
            bracket_center = (cols_0[0] + cols_0[-1]) / 2
            for i, (r_min, c_min, r_max, c_max) in enumerate(ctrl_regions):
                icon_center = (c_min + c_max) / 2
                if abs(bracket_center - icon_center) < 3:
                    return i
    return 0


def find_block_colors(grid, regions):
    block_colors = []
    for r_min, c_min, r_max, c_max in regions:
        r_start = max(0, r_min - 2)
        r_end = min(grid.shape[0], r_max + 3)
        c_start = max(0, c_min - 2)
        c_end = min(grid.shape[1], c_max + 3)
        patch = grid[r_start:r_end, c_start:c_end]
        vals, counts = np.unique(patch, return_counts=True)
        mask = (vals != 5) & (vals != 2) & (vals != 0) & (vals != 1) & (vals != 3)
        if mask.any():
            filtered_vals = vals[mask]
            filtered_counts = counts[mask]
            block_colors.append(int(filtered_vals[filtered_counts.argmax()]))
        else:
            block_colors.append(-1)
    return block_colors


def solve_puzzle(grid):
    """Given a tr87 grid, analyze the puzzle and build a mapping from reference to target."""
    regions = find_icon_regions(grid)
    if not regions:
        return None, "no icon regions found"
    
    row_groups = group_into_rows(regions)
    sorted_rows = sorted(row_groups.keys())
    
    if len(sorted_rows) < 2:
        return None, "not enough rows"
    
    ref_row_y = sorted_rows[-2]
    ctrl_row_y = sorted_rows[-1]
    rule_row_ys = sorted_rows[:-2]
    
    ref_idxs = sorted(row_groups[ref_row_y], key=lambda i: regions[i][1])
    ctrl_idxs = sorted(row_groups[ctrl_row_y], key=lambda i: regions[i][1])
    
    block_colors = find_block_colors(grid, regions)
    
    # Determine which block color is the "input" side (same as reference row)
    # and which is the "output" side (same as control row)
    ref_block = block_colors[ref_idxs[0]] if ref_idxs else -1
    ctrl_block = block_colors[ctrl_idxs[0]] if ctrl_idxs else -1
    
    print(f"Reference row: {len(ref_idxs)} icons, block_color={ref_block}")
    print(f"Control row: {len(ctrl_idxs)} icons, block_color={ctrl_block}")
    
    # Build rules using CANONICAL (rotation-invariant) fingerprints
    # For each rule row: icons in ref_block color = input side, icons in ctrl_block color = output side
    # The rule maps canonical(input_fp) -> exact(output_fp)
    rules = {}  # canonical_hash -> output_fp (exact, as seen on grid)
    
    for ry in rule_row_ys:
        row_idxs = sorted(row_groups[ry], key=lambda i: regions[i][1])
        if len(row_idxs) < 2:
            continue
        
        # Split by block color: ref_block color = input, ctrl_block color = output
        input_idxs = [i for i in row_idxs if block_colors[i] == ref_block]
        output_idxs = [i for i in row_idxs if block_colors[i] == ctrl_block]
        
        input_idxs.sort(key=lambda i: regions[i][1])
        output_idxs.sort(key=lambda i: regions[i][1])
        
        n_pairs = min(len(input_idxs), len(output_idxs))
        for k in range(n_pairs):
            in_fp = icon_fingerprint(grid, regions[input_idxs[k]])
            out_fp = icon_fingerprint(grid, regions[output_idxs[k]])
            in_canon = canonical_fingerprint(in_fp)
            out_canon = canonical_fingerprint(out_fp)
            
            in_key = tuple(in_canon.ravel().tolist())
            out_val = tuple(out_canon.ravel().tolist())
            rules[in_key] = out_val
            print(f"  Rule: input canon={fp_hash(in_canon)} -> output canon={fp_hash(out_canon)}")
    
    print(f"\nTotal rules: {len(rules)}")
    
    # Match reference icons to rules
    targets = []  # target CANONICAL fingerprint for each control position
    for i, ref_idx in enumerate(ref_idxs):
        ref_fp = icon_fingerprint(grid, regions[ref_idx])
        ref_canon = canonical_fingerprint(ref_fp)
        ref_key = tuple(ref_canon.ravel().tolist())
        
        if ref_key in rules:
            target_canon_tuple = rules[ref_key]
            targets.append(target_canon_tuple)
            print(f"  Ref {i}: canon={fp_hash(ref_canon)} -> target canon={hash(target_canon_tuple) % 10000}")
        else:
            targets.append(None)
            print(f"  Ref {i}: canon={fp_hash(ref_canon)} -> NO RULE!")
    
    # Check current control icons
    ctrl_fps_canon = []
    for i, ctrl_idx in enumerate(ctrl_idxs):
        ctrl_fp = icon_fingerprint(grid, regions[ctrl_idx])
        ctrl_canon = canonical_fingerprint(ctrl_fp)
        ctrl_canon_tuple = tuple(ctrl_canon.ravel().tolist())
        ctrl_fps_canon.append(ctrl_canon_tuple)
        
        if i < len(targets) and targets[i] is not None:
            if ctrl_canon_tuple == targets[i]:
                print(f"  Control {i}: CORRECT ✓")
            else:
                print(f"  Control {i}: needs cycling (current canon={hash(ctrl_canon_tuple) % 10000})")
        else:
            print(f"  Control {i}: no target")
    
    cursor_pos = find_cursor_position(grid, [regions[i] for i in ctrl_idxs])
    print(f"\nCursor at: {cursor_pos}")
    
    return {
        "ctrl_idxs": ctrl_idxs,
        "targets": targets,
        "ctrl_fps_canon": ctrl_fps_canon,
        "cursor_pos": cursor_pos,
        "regions": regions,
    }, "OK"


def factory(game_id):
    agent = default_agent_factory(game_id)
    orig_choose = agent.choose_action.__func__
    call_count = [0]
    state = {"phase": "analyze", "targets": None, "cursor": 0, "pos": 0, "cycles": 0}
    
    def patched_choose(self, frames, latest_frame):
        call_count[0] += 1
        step = call_count[0]
        grid = self._parse_grid(latest_frame, frames)
        
        if grid is None:
            return orig_choose(self, frames, latest_frame)
        
        if state["phase"] == "analyze":
            print(f"\n=== ANALYZING (step {step}) ===")
            result, msg = solve_puzzle(grid)
            if result is None or not any(t is not None for t in result["targets"]):
                print(f"Analysis failed: {msg}")
                state["phase"] = "fallback"
                return orig_choose(self, frames, latest_frame)
            
            state["targets"] = result["targets"]
            state["ctrl_idxs"] = result["ctrl_idxs"]
            state["regions"] = result["regions"]
            state["cursor"] = result["cursor_pos"] or 0
            state["pos"] = 0
            state["cycles"] = 0
            state["n_icons"] = len(result["ctrl_idxs"])
            state["phase"] = "goto_start"
        
        if state["phase"] == "goto_start":
            if state["cursor"] > 0:
                state["cursor"] -= 1
                return self.parse_action("ACTION3")
            state["phase"] = "solve"
        
        if state["phase"] == "solve":
            pos = state["pos"]
            n = state["n_icons"]
            
            if pos >= n:
                print(f"  Step {step}: all positions processed!")
                state["phase"] = "fallback"
                return orig_choose(self, frames, latest_frame)
            
            target = state["targets"][pos] if pos < len(state["targets"]) else None
            
            if target is None:
                print(f"  Step {step}: pos {pos} no target, skipping")
                state["pos"] += 1
                state["cycles"] = 0
                if pos < n - 1:
                    return self.parse_action("ACTION4")
                state["phase"] = "fallback"
                return orig_choose(self, frames, latest_frame)
            
            # Check current fingerprint
            ctrl_idx = state["ctrl_idxs"][pos]
            ctrl_fp = icon_fingerprint(grid, state["regions"][ctrl_idx])
            ctrl_canon = tuple(canonical_fingerprint(ctrl_fp).ravel().tolist())
            
            if ctrl_canon == target:
                print(f"  Step {step}: pos {pos} MATCHES ✓ (after {state['cycles']} cycles)")
                state["pos"] += 1
                state["cycles"] = 0
                if pos < n - 1:
                    return self.parse_action("ACTION4")
                print("  ALL DONE!")
                state["phase"] = "fallback"
                return orig_choose(self, frames, latest_frame)
            
            if state["cycles"] >= 7:
                print(f"  Step {step}: pos {pos} exhausted 7 cycles!")
                state["pos"] += 1
                state["cycles"] = 0
                if pos < n - 1:
                    return self.parse_action("ACTION4")
                state["phase"] = "fallback"
                return orig_choose(self, frames, latest_frame)
            
            print(f"  Step {step}: cycling pos {pos} (attempt {state['cycles']+1})")
            state["cycles"] += 1
            return self.parse_action("ACTION1")
        
        return orig_choose(self, frames, latest_frame)
    
    agent.choose_action = patched_choose.__get__(agent)
    return agent


if __name__ == "__main__":
    gid = "tr87"
    hits = glob.glob(os.path.join(ROOT, "eval", "real_games", gid, "*", gid + ".py"))
    if not hits:
        print("Game not found!")
        sys.exit(1)
    gf = hits[0]
    print(f"Running {gf}\n")
    r = run_game(gf, factory, seed=0, max_actions=80)
    print(f"\nResult: {r}")
