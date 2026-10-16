What it is: trying every short list of moves, for example all lists of length 1 to 4 with `itertools.product`, running each through `step()`, and keeping the best one by a score.
Used when: the state is hard to write as a small tuple, a move affects many objects at once, or you want the best short combination (for example which buttons to press in which order). It grows fast: 4 moves at length 6 is 4096 lists, at length 10 a million.
How it looks:
```python
import itertools
def best_sequence(state, moves, max_len, score):
    best = (score(state), [])
    for n in range(1, max_len + 1):
        for seq in itertools.product(moves, repeat=n):
            s = state
            for m in seq:
                s = step(s, m)
            v = score(s)
            if v > best[0]:
                best = (v, list(seq))
    return best
```
