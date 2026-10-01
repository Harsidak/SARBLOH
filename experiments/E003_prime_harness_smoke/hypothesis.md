# E003 — Prime Agent harness smoke test (Gemma 4 31B NVFP4, 1 hour)

- **Date opened:** 2026-09-27
- **Axis varied:** substrate (the whole harness: Prime Agent RLM loop replaces the Duck)
- **Baseline:** none for score. This is a plumbing gate, not a score claim. E001 (Duck) stays the score control.
- **Split:** public offline environment files on Kaggle (smoke only; public-set numbers are never evidence, §4.3)

## Mechanism

`Sarbloh-Prime/prime/` runs the Prime Agent design (arXiv 2608.23552) on ARC-AGI-3. The upstream `rlm` kernel
(`Sarbloh-Prime/rlm/`, MIT, commit 2d24ad4) is the persistent REPL and the only model tool (`ipython`). The game is
exposed in the REPL as `arc` through typed host requests (`prime/arc_host.py`). The Python host loop
(`prime/session.py`) adds autonomous continuation, compaction, the Continual Harness digest, and `rlm.spawn`
children. The model is Gemma-4-31B-IT-NVFP4 on vLLM 0.19.0 with a transformers 5.5.0 overlay (`prime/vllm.py`).

## Why it should work

Locally, the same code path ran end to end with a scripted model and with Qwen3.5-4B on llama.cpp (native tool
calls, 15 actions committed, a compaction, a subagent round trip). What is unverified is only the Kaggle side:
the Gemma 4 serving stack and throughput on the RTX PRO 6000.

## Prediction

Committed before the run:
1. vLLM starts on `gemma_fast` or `gemma_safe` and the smoke test returns `tool_mode=native`.
2. All 6 games run for the full budget without a session crash; each commits at least 20 actions.
3. Aggregate throughput is at least 300 tok/s at concurrency 8.
4. Score: at least 1 level completed across the 6 games. This is not a gate, only recorded.

## Kill criterion

Binding. If vLLM fails on all three profiles, or more than 2 of the 6 sessions crash, the serving stack or the
harness is wrong. Fix it before any scored run; do not tune the agent.

## Measurement

- `LEDGER_ROW` from the notebook: profile, tool_mode, throughput, turns, tool calls, native vs fenced, levels,
  actions, wall.
- `prime_run/results.json`: per-session end reason, compactions, cell errors, children.
- Transcripts in `prime_run/games/*/transcript.jsonl`, read for failure modes before the next change.
