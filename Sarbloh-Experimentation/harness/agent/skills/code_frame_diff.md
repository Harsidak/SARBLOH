What it is: a short summary of what moves changed: which objects moved (and by how much), appeared, disappeared or changed size, and whether only the HUD changed. It uses `match()` from the object_tracking skill.
Used when: after a `play()`, to compare the real board with what `step()` predicted, and to learn what an unknown move does.
How it looks:
```python
def diff(old, new):
    out, changed = [], []
    for o, n in match(old.objects, new.objects):
        dr, dc = n["bbox"][0] - o["bbox"][0], n["bbox"][1] - o["bbox"][1]
        if (dr, dc) != (0, 0) or n["size"] != o["size"]:
            out.append(f'{o["letter"]} {o["id"]}: moved {dr},{dc} size {o["size"]}->{n["size"]}')
            changed.append(n)
    only_hud = bool(changed) and all(n["hud"] for n in changed)
    return out, only_hud
```
If only HUD objects changed, the move did nothing useful on the board.
