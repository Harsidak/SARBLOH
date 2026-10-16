What it is: breadth-first search. It tries all 1-move paths, then all 2-move paths, and so on, so the first path that reaches the goal is a shortest one. It uses `collections.deque` and a `seen` set.
Used when: you move something to a target, around walls, and every move costs the same. Also used to check whether a goal can be reached at all.
How it looks:
```python
from collections import deque
def bfs(start, is_goal, moves, limit=200000):
    q, seen = deque([(start, [])]), {start}
    while q and len(seen) < limit:
        s, path = q.popleft()
        if is_goal(s):
            return path
        for m in moves:
            n = step(s, m)
            if n not in seen:
                seen.add(n)
                q.append((n, path + [m]))
    return None
```
If it returns None, one rule in `step()` is probably wrong, or the goal is.
