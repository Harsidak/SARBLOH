What it is: a Python function `step(state, move)` that returns the next state, the way the game would. The state is a small hashable value, for example a tuple of positions plus a frozenset of open doors, so it can go into a `set` and a search can use it.
Used when: you know some rules of the game and want to plan without spending moves. Every search (bfs, weighted_search, beam_search, sequence_search) runs on it.
How it looks:
```python
DIRS = {"1": (-1, 0), "2": (1, 0), "3": (0, -1), "4": (0, 1)}
def step(state, move):
    (r, c), keys = state
    dr, dc = DIRS[move]
    nr, nc = r + dr * STRIDE, c + dc * STRIDE
    if blocked(nr, nc):
        return state
    return ((nr, nc), keys)
```
Keep `STRIDE`, `blocked()` and the other rules as separate names, so you can fix one rule when the game disagrees.
