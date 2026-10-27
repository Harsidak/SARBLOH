What it is: a short summary of what moves changed: which objects moved (and by how much), turned, changed colour, appeared or disappeared. The kernel already has it as `frame_diff(before, after)`, so call it instead of writing your own.
Used when: after a `play()`, to compare the real board with what `step()` predicted, and to learn what an unknown move does.
How it looks:
```python
d = frame_diff()                          # the board before the newest move -> the board now
for o in d["moved"]:
    print(o["letter"], o["size"], "by", o["by"])
d = frame_diff(m["before"], m["after"])   # any move m from moves()
```
If only the move bar or a counter at the edge changed, the move did nothing on the board.
