What it is: sending moves from code one at a time, checking the board after each one against the model, and stopping as soon as they differ. `play()` does this loop for you.
Used when: you have a path from a search. The loop spends only the moves that matched, and you fix `step()` at the first surprise. A correct model costs nothing extra, so use it for trusted paths too.
How it looks:
```python
def predict(board, move):          # the board you expect after the move
    return render(step(state_of_board(board), move))

r = await play(path, predict=predict)
if r.stopped == "surprise":
    print(r.surprise["move"], r.surprise["cells"][:5])   # fix that rule in step(), then search again

# predict only the cells you care about, for example where the player lands:
r = await play(path, predict=lambda board, m: {player_after(board, m): "R"})

# or compare your own state instead of the board:
r = await play(path, predict=step, state=state_of)
```
