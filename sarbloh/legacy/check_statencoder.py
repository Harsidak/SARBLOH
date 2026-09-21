"""StateEncoder in isolation, driven by the real ls20 game.

COMPONENT 1 of my_agent.py ("EYES") is copied VERBATIM below -- class StateEncoder
plus the two constants and the one dataclass it depends on. Nothing else from
my_agent.py is imported, so what this prints is exactly what the agent's eyes
report and nothing downstream can be blamed for it.

Two different things are called "level" here; keep them apart:
  * GAME level    -- ls20's level 1, the first playable board.
  * ENCODING level -- the `level=` argument of encode_state (0..3), a verbosity
    dial. Level 1 is the one the agent actually feeds to the LLM.

Run:
    PYTHONIOENCODING=utf-8 python -u check_statencoder.py
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import glob
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "eval"))

# ======================================================================
# ---- VERBATIM FROM my_agent.py ---------------------------------------
# my_agent.py:123-124
FRAME_CONCENTRATION = 4.0            # a wall ring must be >=4x denser on the border than inside
BORDER_CONCENTRATION = 1.5           # chrome: >=1.5x over-represented on the border vs. its own area


# my_agent.py:261-268
@dataclass
class GridObject:
    color: int
    size: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    shape_class: str
    mask: np.ndarray


# my_agent.py:321-504
class StateEncoder:
    def __init__(self):
        self.shape_types = ['single_pixel', 'horizontal_line', 'vertical_line',
                            'square', 'rectangle', 'hollow_shape', 'irregular_blob']
        self._obj_cache: Dict[Any, List[GridObject]] = {}

    def detect_background(self, grid: np.ndarray) -> int:
        if grid.size == 0:
            return 0
        h, w = grid.shape
        if h == 0 or w == 0:
            return 0
        border = (np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
                  if h > 2 and w > 2 else grid.flatten())
        if len(border) == 0:
            return 0
        unique, counts = np.unique(border, return_counts=True)
        border_bg = int(unique[counts.argmax()]) if len(unique) > 0 else 0
        border_frac = counts.max() / len(border) if len(border) > 0 else 0.0

        # A FRAME (wall ring) is NOT the background -- the floor inside it is.
        # Frame vs. fill is decided by CONCENTRATION, not by absolute counts: a
        # frame color is many times denser on the border than in the interior,
        # while a genuine background is about as dense inside as out. The density
        # ratio is what separates "wall ring around a floor" from "background
        # around one large centered object" -- the two are indistinguishable by
        # counts alone, and the majority-interior test on its own gets the second
        # case wrong. Without this rule, bordered rooms/mazes flip bg to the wall
        # color and
        # collapse the whole floor into one phantom object, corrupting every
        # downstream reader (LLM prompts, MCTS value, object-scoped DSL ops).
        if border_frac > 0.5 and h > 2 and w > 2:
            interior = grid[1:-1, 1:-1]
            if interior.size >= 4:
                iu, ic = np.unique(interior, return_counts=True)
                interior_bg = int(iu[ic.argmax()])
                inside_frac = float(ic[iu == border_bg].sum()) / interior.size
                if (interior_bg != border_bg
                        and ic.max() > interior.size * 0.5
                        and inside_frac * FRAME_CONCENTRATION < border_frac):
                    return interior_bg

        if border_frac > 0.5:
            # Same principle as above, one scale up: the border winner may be a
            # thick CHROME band (surround / HUD panel) rather than a 1-px ring,
            # in which case the 1-px-interior test above cannot see it. A chrome
            # color is OVER-REPRESENTED on the border relative to how much of the
            # grid it actually owns (border_share / area_share is high), and some
            # other color owns strictly more of the grid -- that other color is
            # the play field. A genuine background is not over-represented on its
            # own border, so this cannot fire on one. Both conditions are
            # required: over-representation alone would misfire on frames that
            # are legitimately split into two large panels, where no color beats
            # the border winner on area and there is nothing better to choose.
            area_share = float((grid == border_bg).sum()) / grid.size
            if area_share > 0 and border_frac >= BORDER_CONCENTRATION * area_share:
                unique_g, counts_g = np.unique(grid, return_counts=True)
                cand = int(unique_g[counts_g.argmax()])
                if cand != border_bg and counts_g.max() > area_share * grid.size:
                    return cand
            return border_bg
        unique, counts = np.unique(grid, return_counts=True)
        return int(unique[counts.argmax()]) if len(unique) > 0 else 0

    def objects(self, grid: np.ndarray, bg: Optional[int] = None) -> List[GridObject]:
        """Cached object extraction (Issue 3: MCTS expands many nodes on the same
        grid -- flood fill must not be recomputed per expansion)."""
        if bg is None:
            bg = self.detect_background(grid)
        key = (grid.shape, int(bg), grid.tobytes())
        hit = self._obj_cache.get(key)
        if hit is not None:
            return hit
        objs = self.flood_fill_objects(grid, bg)
        if len(self._obj_cache) > 256:
            self._obj_cache.clear()
        self._obj_cache[key] = objs
        return objs

    def flood_fill_objects(self, grid: np.ndarray, bg_color: int, connectivity: int = 4) -> List[GridObject]:
        visited = np.zeros_like(grid, dtype=bool)
        objects: List[GridObject] = []
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                if not visited[r, c] and grid[r, c] != bg_color:
                    color = int(grid[r, c])
                    mask = np.zeros_like(grid, dtype=bool)
                    q = [(r, c)]
                    visited[r, c] = True
                    mask[r, c] = True
                    size = 0
                    rmin, cmin, rmax, cmax = r, c, r, c
                    while q:
                        cr, cc = q.pop(0)
                        size += 1
                        rmin, rmax = min(rmin, cr), max(rmax, cr)
                        cmin, cmax = min(cmin, cc), max(cmax, cc)
                        neigh = [(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)]
                        if connectivity == 8:
                            neigh += [(cr-1, cc-1), (cr-1, cc+1), (cr+1, cc-1), (cr+1, cc+1)]
                        for nr, nc in neigh:
                            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == color:
                                visited[nr, nc] = True
                                mask[nr, nc] = True
                                q.append((nr, nc))
                    center = (rmin + (rmax - rmin) / 2.0, cmin + (cmax - cmin) / 2.0)
                    shape_class = self.classify_shape(mask, rmin, cmin, rmax, cmax, size)
                    objects.append(GridObject(color, size, (rmin, cmin, rmax, cmax), center, shape_class, mask))
        return objects

    def classify_shape(self, mask, rmin, cmin, rmax, cmax, size) -> str:
        h = rmax - rmin + 1
        w = cmax - cmin + 1
        if size == 1:
            return 'single_pixel'
        if h == 1 and w > 1 and size == w:
            return 'horizontal_line'
        if w == 1 and h > 1 and size == h:
            return 'vertical_line'
        if size == h * w:
            return 'square' if h == w else 'rectangle'
        if h > 2 and w > 2:
            inner = mask[rmin+1:rmax, cmin+1:cmax]
            if not np.any(inner) and size == (2*h + 2*w - 4):
                return 'hollow_shape'
        return 'irregular_blob'

    def rle_encode(self, mask: np.ndarray) -> str:
        flat = mask.flatten()
        if len(flat) == 0:
            return ""
        runs, curr, count = [], flat[0], 1
        for val in flat[1:]:
            if val == curr:
                count += 1
            else:
                runs.append(f"{count}{'T' if curr else 'F'}")
                curr, count = val, 1
        runs.append(f"{count}{'T' if curr else 'F'}")
        return ",".join(runs)

    def encode_state(self, grid: np.ndarray, level: int = 1, requested_objects: List[int] = None) -> str:
        h, w = grid.shape
        bg = self.detect_background(grid)
        unique, counts = np.unique(grid, return_counts=True)
        color_counts = dict(zip([int(x) for x in unique], [int(x) for x in counts]))
        l0 = f"Grid: {h}x{w}. BG: {bg}. Colors: {color_counts}."
        if level == 0:
            return l0
        objects = self.objects(grid, bg)
        obj_strs = [f"Obj{i}(C:{o.color}, S:{o.size}, BB:{o.bbox}, Shp:{o.shape_class})"
                    for i, o in enumerate(objects)]
        l1 = l0 + " Objects: [" + ", ".join(obj_strs) + "]"
        if level == 1:
            return l1
        if level == 2 and requested_objects:
            details = [f"Obj{i}_RLE:{self.rle_encode(objects[i].mask)}"
                       for i in requested_objects if i < len(objects)]
            return l1 + " Details: " + " ".join(details)
        if level == 3:
            if h <= 10 and w <= 10:
                grid_str = "\n".join(" ".join(map(str, row)) for row in grid)
                return l1 + f"\nFull Grid:\n{grid_str}"
            sh, sw = max(1, h // 10), max(1, w // 10)
            ds = grid[::sh, ::sw]
            grid_str = "\n".join(" ".join(map(str, row)) for row in ds)
            return l1 + f"\nDownsampled Grid:\n{grid_str}"
        return l1

    def encode_transition(self, prev: np.ndarray, nxt: np.ndarray, action: str) -> str:
        if prev.shape != nxt.shape:
            return f"Action {action} resized grid {prev.shape} -> {nxt.shape}"
        diff = nxt != prev
        n = int(np.sum(diff))
        if n == 0:
            return f"Action {action}: No change."
        added, ac = np.unique(nxt[diff], return_counts=True)
        removed, rc = np.unique(prev[diff], return_counts=True)
        if set(added.tolist()) == set(removed.tolist()) and len(added) > 0:
            return f"Action {action}: movement/swap of {n} px, colors {added.tolist()}."
        add_d = dict(zip([int(x) for x in added], [int(x) for x in ac]))
        rem_d = dict(zip([int(x) for x in removed], [int(x) for x in rc]))
        return f"Action {action}: {n} px changed. Added: {add_d}, Removed: {rem_d}."


# my_agent.py:7326-7338 -- how the agent turns a FrameData into the array it
# hands the encoder. Copied because the encoder is useless without it.
def _grid_from(frame_obj) -> Optional[np.ndarray]:
    raw = getattr(frame_obj, "grid", None)
    if raw is None:
        raw = getattr(frame_obj, "frame", None)
    if raw is None:
        return None
    try:
        arr = np.array(raw, dtype=np.int32)
    except Exception:
        return None
    if arr.ndim == 3:               # arcengine FrameData.frame is a LIST of grids
        arr = arr[-1]               # (animation frames); the last one is current
    return arr if arr.size and arr.ndim == 2 else None
# ---- END VERBATIM ----------------------------------------------------
# ======================================================================


def open_game(gid="ls20", seed=0):
    from scoreboard import make_env_info
    from arc_agi.local_wrapper import LocalEnvironmentWrapper
    hits = glob.glob(os.path.join(HERE, "eval", "real_games", gid, "*", f"{gid}.py"))
    if not hits:
        raise SystemExit(f"no game module for {gid}")
    game_file = sorted(hits)[0]
    info, src = make_env_info(game_file, gid, {})
    lg = logging.getLogger("probe")
    lg.setLevel(logging.ERROR)
    wrapper = LocalEnvironmentWrapper(info, lg, seed=seed, scorecard_id="local")
    return wrapper, game_file, src


def dump_objects(enc, grid, bg, limit=None):
    objs = enc.objects(grid, bg)
    print(f"  objects found: {len(objs)}")
    print(f"  {'#':>4}  {'color':>5} {'size':>6}  {'bbox (r0,c0,r1,c1)':<24} "
          f"{'center (r,c)':<16} shape")
    shown = objs if limit is None else objs[:limit]
    for i, o in enumerate(shown):
        bb = f"({o.bbox[0]},{o.bbox[1]},{o.bbox[2]},{o.bbox[3]})"
        ct = f"({o.center[0]:.1f},{o.center[1]:.1f})"
        print(f"  {i:>4}  {o.color:>5} {o.size:>6}  {bb:<24} {ct:<16} {o.shape_class}")
    if limit is not None and len(objs) > limit:
        print(f"  ... {len(objs) - limit} more")
    return objs


def main():
    gid = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    wrapper, game_file, base_src = open_game(gid)
    frame = wrapper.observation_space
    grid = _grid_from(frame)
    enc = StateEncoder()

    print("=" * 78)
    print(f"StateEncoder isolation probe -- {gid}")
    print("=" * 78)
    print(f"game module : {os.path.relpath(game_file, HERE)}")
    print(f"baselines   : {base_src}")
    print(f"game level  : {getattr(frame, 'levels_completed', '?')} completed "
          f"(win_levels={getattr(frame, 'win_levels', '?')})  "
          f"state={getattr(frame, 'state', '?')}")
    print(f"available   : {getattr(frame, 'available_actions', None)}")
    raw = getattr(frame, "frame", None)
    try:
        raw_shape = np.array(raw, dtype=np.int32).shape
    except Exception:
        raw_shape = "?"
    print(f"raw frame   : {raw_shape}  -> encoder sees {grid.shape}")

    bg = enc.detect_background(grid)
    uniq, cnt = np.unique(grid, return_counts=True)
    print()
    print("-- 1. detect_background -------------------------------------------")
    h, w = grid.shape
    border = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
    bu, bc = np.unique(border, return_counts=True)
    print(f"  border winner : color {int(bu[bc.argmax()])} "
          f"({bc.max()}/{len(border)} = {bc.max()/len(border):.3f} of the border)")
    print(f"  whole-grid    : " + ", ".join(
        f"{int(u)}:{int(c)} ({c/grid.size:.3f})" for u, c in zip(uniq, cnt)))
    print(f"  CHOSEN bg     : {bg}"
          + ("  <-- border winner" if bg == int(bu[bc.argmax()])
             else "  <-- OVERRIDDEN (frame/chrome rule fired)"))

    print()
    print("-- 2. objects (flood fill, 4-connectivity, non-bg) -----------------")
    objs = dump_objects(enc, grid, bg, limit=40)
    from collections import Counter
    print("  by shape class: " + dict(Counter(o.shape_class for o in objs)).__repr__())
    print("  by color      : " + dict(Counter(o.color for o in objs)).__repr__())
    if objs:
        big = max(objs, key=lambda o: o.size)
        print(f"  largest       : color {big.color}, {big.size} px, "
              f"{big.size/grid.size:.1%} of the grid, shape {big.shape_class}")

    print()
    print("-- 3. encode_state(level=0) ---------------------------------------")
    print(enc.encode_state(grid, level=0))

    print()
    print("-- 4. encode_state(level=1)  <-- what the agent actually emits -----")
    s1 = enc.encode_state(grid, level=1)
    print(s1)
    print(f"  [{len(s1)} chars, ~{len(s1)//4} tokens]")

    print()
    print("-- 5. encode_state(level=3) tail (downsampled grid) ----------------")
    s3 = enc.encode_state(grid, level=3)
    print(s3[len(s1):].strip() or "(empty)")

    print()
    print("-- 6. encode_transition, one step per available action -------------")
    for a in list(getattr(frame, "available_actions", []) or []):
        name = getattr(a, "name", str(a))
        w2, _, _ = open_game(gid)           # a fresh game per probe: unscored view
        f0 = w2.observation_space
        g0 = _grid_from(f0)
        try:
            if name.upper().startswith("ACTION6") or "6" in name:
                f1 = w2.step(a, x=32, y=32)
                name = f"{name}(x=32,y=32)"
            else:
                f1 = w2.step(a)
        except TypeError:
            f1 = w2.step(a)
        except Exception as e:
            print(f"  {name}: step failed: {type(e).__name__}: {e}")
            continue
        g1 = _grid_from(f1)
        print("  " + enc.encode_transition(g0, g1, name))
        b1 = enc.detect_background(g1)
        n1 = len(enc.objects(g1, b1))
        print(f"      -> bg {bg}->{b1}, objects {len(objs)}->{n1}")


if __name__ == "__main__":
    main()
