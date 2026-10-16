What it is: shortest-path search with a priority queue (`heapq`). Dijkstra takes the cheapest state first. A* adds a guess of the cost still to go, such as the Manhattan distance to the target, so it looks at fewer states.
Used when: moves cost different amounts (a slide moves many cells, a click jumps), or BFS is too slow because there are many states and you have a good distance guess.
How it looks:
```python
import heapq, itertools
def astar(start, is_goal, moves, cost, guess):
    tie = itertools.count()
    q, best = [(guess(start), next(tie), 0, start, [])], {start: 0}
    while q:
        _, _, g, s, path = heapq.heappop(q)
        if is_goal(s):
            return path
        for m in moves:
            n, g2 = step(s, m), g + cost(s, m)
            if g2 < best.get(n, 1e9):
                best[n] = g2
                heapq.heappush(q, (g2 + guess(n), next(tie), g2, n, path + [m]))
    return None
```
With `guess = lambda s: 0` it is Dijkstra.
