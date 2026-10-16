What it is: a search that keeps only the best K states at each depth, rated by a score function, and expands only those. It is fast but can miss a path that looks worse at first.
Used when: there are too many states for BFS and you have a score that says how close a state is to the goal (cells matched, distance to the target, objects in place).
How it looks:
```python
def beam(start, moves, score, is_goal, width=50, depth=40):
    layer = [(start, [])]
    for _ in range(depth):
        nxt = {}
        for s, path in layer:
            for m in moves:
                n = step(s, m)
                if is_goal(n):
                    return path + [m]
                nxt.setdefault(n, path + [m])
        layer = sorted(nxt.items(), key=lambda x: -score(x[0]))[:width]
    return None
```
