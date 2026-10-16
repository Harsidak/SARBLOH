What it is: matching each object in a new `observe()` to the same object in the old one. Match by `hash` first (same shape and color), then by color, then by the nearest `bbox`. Exact positions alone fail when things move.
Used when: you want to know which object moved and how far, which one you control, or which object a click changed.
How it looks:
```python
def match(old, new):
    pairs, used = [], set()
    for o in old:
        free = [n for n in new if n["id"] not in used]
        cands = [n for n in free if n["hash"] == o["hash"]] or [n for n in free if n["letter"] == o["letter"]]
        if cands:
            n = min(cands, key=lambda n: abs(n["bbox"][0] - o["bbox"][0]) + abs(n["bbox"][1] - o["bbox"][1]))
            used.add(n["id"])
            pairs.append((o, n))
    return pairs
```
Objects with no match appeared or disappeared.
