What it is: one or two test moves chosen so that their result tells two of your ideas apart. Before the move, compute what each idea predicts; after it, keep the idea that matched.
Used when: you are not sure what a move does or what wins, and a search on the wrong model would waste many moves. Once you know what you control, what each move does and the goal, probes stop paying back and searching the model is cheaper.
How it looks:
```python
s = state_of(observe())
pred_a = step_a(s, "1")   # idea A: move 1 slides until a wall
pred_b = step_b(s, "1")   # idea B: move 1 goes one cell
r = await play(["1"])
now = state_of(r.obs)
print("A" if now == pred_a else "B" if now == pred_b else "neither")
```
