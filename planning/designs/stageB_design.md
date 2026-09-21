# Stage B — state-graph BFS with frame-hash dedup (the click world-model)

## Model
- **Transition model = the env.** A live RESET restores the level's exact initial state, so any
  button-index path can be replayed. (RESET refills the StepCounter budget too.)
- **State = frame hash** `hash((grid.shape, grid.tobytes()))`, fed in via `update(grid=...)`.
- **Actions = clustered buttons** (Stage A: `_detect_buttons` → 1 rep per sprite).
- **Goal = WIN**, detected implicitly: a winning test-click increments the score → the outer loop
  calls `new_level()`, which resets search. No explicit goal model needed.

## Why it beats blind product-BFS
lp85 LEFT button yields only ~5 distinct states then PARKS at a wall (fixed point). Product BFS
re-clicks the parked button (no-ops) and re-explores equivalent sequences; state-dedup visits each
distinct board once and finds the SHORTEST winning path.

## Bookkeeping (one action per act() call; result arrives via update() next turn)
- `_G: {state: {button_idx: child_state}}`   learned edges
- `_dist: {state: (button_idx, ...)}`        BFS path from root (shortest)
- `_queue: deque[state]`                      BFS frontier
- `_root: state | None`
- `_plan: [ "RESET" | int, ... ]`, `_plan_i`  the sequence currently being emitted
- `_pending: ("root",) | ("expand", parent, b) | None`  what the finished plan tests
- `_dead_edges: set[(state, b)]`              budget-aborted / over-long expansions
- `_last_hash`                                hash of most recent grid (set in update)

## Loop
1. enter search → `_pending=("root",)`, `_plan=["RESET"]`.
2. emit plan step-by-step; when `_plan_i==len(_plan)`, call `_process_result()` using `_last_hash`:
   - root: `_root=h; _dist[h]=(); _queue=[h]`
   - expand: `_G[parent][b]=h`; if h new: `_dist[h]=_dist[parent]+(b,); _queue.append(h)`
3. `_build_next_plan()`: pop front of queue, first untried & non-dead button b:
   `path=_dist[s]`; if `len(path)+1 > MAX_REPLAY`: dead; else `_plan=["RESET",*path,b]`,
   `_pending=("expand",s,b)`. Queue empty → `_search_done`, fall back to coverage.
4. GAME_OVER mid-replay → outer loop calls `reset()`: mark `_pending` edge dead, clear plan.

## Guards / floor
- Gated behind coverage (`discover` first) — floor preserved.
- Exhaustion / hash-noise blowup → falls back to coverage.
- New MAX_REPLAY ~30 (lp85 ~52 button-clicks/level budget; only button clicks cost).
