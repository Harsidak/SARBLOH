What it is: sending moves from code one at a time, checking the board after each one against the model, and stopping as soon as they differ.
Used when: you have a path from a search but are not fully sure of the model. The loop spends only the moves that matched, and you fix `step()` at the first surprise. When the model is already trusted, the whole path goes in one `act(path)` instead.
How it looks:
```python
s = state_of(observe())
for m in path:
    want = step(s, m)
    r = await act([m], quiet=True)
    s = state_of(observe())
    if r.level_up or r.done or s != want:
        print("stopped at", m, "expected", want, "got", s)
        break
```
