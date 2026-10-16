What it is: a function `score(state)` that returns a number for how close a state is to the goal. Examples: minus the distance from the block to the target, the number of cells that match a pattern, the number of objects in place.
Used when: you compare candidate moves or move lists, in beam search, as the A* guess, or when you choose between clicks. A good score turns a hard search into a short greedy one.
How it looks:
```python
def score(state):
    (r, c), _ = state
    return -(abs(r - TR) + abs(c - TC))
```
For pattern goals, count matches: `sum(a == b for a, b in zip(cells_now, cells_wanted))`.
