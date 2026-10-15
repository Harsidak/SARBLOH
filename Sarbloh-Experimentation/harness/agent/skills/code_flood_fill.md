What it is: a fill from one cell to every cell it can reach through open floor. It gives the set of reachable cells, or splits the board into rooms.
Used when: you want to know which objects can be reached, which region is closed off, which cells count as floor, or which cells form one connected area of a color.
How it looks:
```python
def flood(board, start, passable):
    H, W = len(board), len(board[0])
    stack, seen = [start], {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in seen and passable(board[nr][nc]):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen
```
