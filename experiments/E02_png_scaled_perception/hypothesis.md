# E02 — Frame representation: deltas vs whole-frame image
- **Axis varied:** perception

> Logged retrospectively on 2026-10-01 from the owner's report. Qualitative result; no held-out RHAE yet (Rule 2 open).

## Mechanism
Original idea: send only changed cells to save tokens. Tested against rendering the full frame as a PNG scaled 10x.

## Why it should work
Compute was not the constraint. Some actions change the game globally (ls20 level 1: stepping on `+` affects the whole game), which a cell-delta view hides; a whole-frame image keeps the global picture.

## Prediction
The PNG arm forms the main understanding of the game from the image; the delta-only arm misses global effects.
