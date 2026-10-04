# E121 — SFT traces from the exact wire, behind a `tracing` switch

- **Date opened:** 2026-10-05
- **Axis varied:** adaptation (training data for LoRA SFT); no change to how the agent plays
- **Baseline:** `Sarbloh-Experimentation/harness/trace.py` as of commit bba2626 (`sft_levels`: one row per level, rebuilt
  from the transcript's message log)
- **Split:** none for the data check (it is a correctness test); the first Kaggle run is on the experimentation games
- **Status:** written 2026-10-05 at the owner's order, code in the same change. Not run on Kaggle yet.

## Why

LoRA SFT trains the model on what it was shown and what it answered. The old export rebuilt each level from the
logged messages, but the agent rewrites its context while it plays, so the rows were not what the model saw:

1. **Level starts:** every level-up resets the context (`_reset_context`: level-up note, full memory, full board). The
   export put the original task message first instead, and the new level's board was missing from its row.
2. **Rewrites:** a drain removes old thinking, old pictures and memory messages; a compaction replaces the history with
   a summary; a trim cuts old tool outputs. The transcript keeps the originals, so the rows held text the model never
   saw at that point and lacked the summary it did see.
3. **Pictures:** the transcript keeps only a `[png of step N]` marker; the picture (map + briefing panel) was lost.
4. **Reasoning effort:** set per level as a chat-template argument, not recorded with the rows.
5. **Labels:** only per level (`level_cleared`, `rhae`); cut-off replies, failed cells, refused acts, repeated
   (state, action) pairs and GAME_OVER moves were all training targets.

And on the competition rerun none of this is seen, so it should cost nothing there.

## Mechanism

**Switch.** `tracing` in `harness/config.py` (default True). Off: no `transcript.jsonl`, no picture files, no
`trace.py` reports or SFT export. `run.py` also forces it off on the competition rerun. The notebook sets
`"tracing": not SUBMISSION`, where `SUBMISSION` is Kaggle's `KAGGLE_IS_COMPETITION_RERUN`.

**Agent (logging only; the messages and the requests are not touched).** With tracing on:
- after every context rewrite (level-up reset, drain, compaction, trim, overflow reset) one `context` event holds the
  whole new message list as stored, thinking included;
- every picture sent is saved as `games/<game>/frames/img_NNNNN.png` and its message names the file (`_image_file`);
- every reply's event carries the reasoning effort it was asked with and `wire_sha`, a digest of the exact request;
- the old builder (`append_only` off) also logs each memory block it pins (`pinned` event).

**trace.py.** Replays the transcript: the stored messages are the last `context` snapshot plus every message appended
after it; each turn's request is rebuilt with the same `context.build` the agent used and checked against `wire_sha`.
Turns whose request extends the previous one (the append-only case between rewrites) form one row, so every
assistant message in a row was trained on exactly its preceding messages. Per reply a `flags` list
(`cut_off`, `tool_error`, `cell_error`, `repeat`, `game_over`, `no_tool_call`) and `train: false` for the bad ones
(all but `no_tool_call`). Row fields: level, `level_cleared`, `rhae`, `segment_start` (start / level_up / drain /
compaction / trim / overflow), `template_kwargs`, `wire_verified`. Transcripts without snapshots fall back to the old
per-level export. The reports (steps.md, report.md, trace.json, trace.md) are unchanged.

## Prediction

- Every rebuilt request matches its `wire_sha` (100% `wire_verified`) on the scripted local session and on the first
  Kaggle run.
- The switch off writes no transcript, frames or SFT file and changes neither the requests nor the score.
- Tracing on costs under 1% of wall clock (one JSON digest per turn, a PNG write per picture, a snapshot per rewrite).

## Kill criterion

Kill (go back to the old export) if, on the first Kaggle run, fewer than 99% of turns are `wire_verified` and the
mismatch cannot be fixed in one day, or if tracing on costs more than 3% of wall clock or more than 5 GB of
`/kaggle/working`.

## Measurement

- `tests/unit/test_experimentation_trace.py`: a scripted session on ls20 (drains, compactions, a forced level-up
  reset); every request the model received must equal the request rebuilt from the transcript.
- Kaggle: `trace.md` reports rows, turns, verified share, flagged share and frame bytes.
