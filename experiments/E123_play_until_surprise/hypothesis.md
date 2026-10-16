# E123 — `play()`: send a whole plan and stop only at the first surprise

- **Date opened:** 2026-10-16
- **Axis varied:** action policy (a kernel helper the model calls; no gate, no new rule)
- **Baseline:** `Sarbloh-Experimentation/harness` as of a8e0479, the owner's 45-minute run of 2026-10-16 on the 15
  public games (notebook `kaggle/experimental/sarbloh_experimentation.ipynb`)
- **Split:** experimentation games (public, smoke-test only; not evidence for the paper)
- **Status:** code written 2026-10-16 at the owner's go ("implement fix 1"). Not run yet.

## Problem (measured in the earlier run, before this change)

The median `act()` call sends 1 move, and 71% of reply tokens go to levels that are never cleared. The prompt
already asks for the loop "act one move, compare with `step()`, stop at the first difference" (`prompts.py`
"Act loop", skill `code_act_loop`), but the model rarely writes it. So each move costs a whole turn: a reply, a
prefill and a thinking budget.

## Mechanism

A ready-made kernel function, `await play(path, predict)`, runs that loop for the model:
1. Before each move it asks `predict(board, move)` what the board will look like. It may return a whole board, a
   dict `{(r, c): letter}` with only the cells it cares about, or `None` (this move is not checked).
2. It sends that one move through the normal `act()`. Same legality check, same stops, same memory sync.
3. It reads the new board (without asking the host for a new picture) and compares. Cells of `hud` objects are
   skipped by default, because counters change on every move.
4. It stops at the first mismatch, a level up, GAME_OVER or the end of the game. It prints one line per move,
   then on a surprise the differing cells and a small window around them.

With `state=` the comparison uses the model's own state instead (`state(obs)` before and after,
`predict(state, move)` compared with `==`), matching the `code_act_loop` skill.

Moves made by `play()` are logged as normal `act` events with `"via": "play"`, so `speed.py` can count them. A new
metric, moves per cell, works for both `act()` and `play()`.

## Prediction

- Median moves per `ipython` cell that moves at all rises from 1 to **5 or more**.
- Levels cleared in the 45-minute run on the 15 games is **at least the baseline's**, at equal tok/s.

## Kill criterion

Kill if, in one 45-minute run: fewer than 20% of cells that move use `play()`, **or** levels cleared fall below the
baseline's at equal tok/s.

## Cost noted in advance

Each move is its own host request, so the host builds the observation (and the picture when vision is on) once
per move instead of once per call. That is host CPU in milliseconds per move, not model time. If the trace shows
it is large, the fix is to build the observation once at the end of the cell.
