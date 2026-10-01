# E004 verdict — INCONCLUSIVE, superseded by E005 (2026-09-28)

**Run:** `runs/prime_local_E004_v1` (Qwen3.5-4B, ls20, `--cell-cap 8 --auto-refine 15 --refine-cooldown-min 5`).
The transcript covers 9.4 min and has no `end` event: the process was stopped before the 60-min kill check, so the
kill criterion was never evaluated.

Counted from the transcript: 38 turns, 39 code blocks (13 errored), 59 actions, level 0/7, 5 compactions, 2 auto-refine
reviews (1 applied memory, host-written, partly false: e.g. "Actions remaining per level: 5"), 0 agent-written harness
entries, 0 `rlm.spawn`.

Predictions: 1 missed (1 entry, not ≥5), 2 missed (every post-compaction window began with a reset), 3 met
(`cell_cap_hits` > 0; the cap bound 3 times), 4 not predicted.

**Why superseded, not killed:** the E005 audit (`experiments/E005_prime_fidelity/audit.md`) found that the behaviour
E004 measured came from our port, not from Prime Agent or only from the model: compaction kept only the task
message plus a summary (upstream keeps the newest 20k tokens verbatim), the cap message used "cell" for a code block
while the ARC prompt used it for a grid square (the model filed the cap under "Falsified Hypotheses" and reset to
"clean the cells"), and `arc.transitions()` was described like a tuple (8 indexing errors). Measuring forced
reflection on top of a broken L1 would not isolate anything. E005 fixes the port first; the cap and the reflection
lock stay in the code, off by default.
