# E109 — SGLang (Pennyroyal, RTX Pro 6000 build) instead of vLLM

- **Date opened:** 2026-10-01
- **Axis varied:** budget (serving; backlog E73/E83 neighbourhood)
- **Baseline:** E008 arm A on vLLM 0.19, Kaggle run `banwait13/sarbloh-prime` v354374653 (2026-10-01): profile
  `qwen_fp8_vision`, probe 42.6 tok/s single stream and 315 tok/s over 8 streams; in-game 168 tok/s over 6 games
  (vllm-server.log average); ls20: 25 turns, 16 actions, 1 level in 90 min while sharing the GPU with 5 other games.
- **Split:** none. One public game (ls20), owner decision 2026-10-01 ("for the inference run check on ls20 game on
  cloud"). A serving smoke test, not a score result.

## In simple words

The model is the brain. The server is the mouth it talks through. Today the mouth (vLLM) is slow: about 28 words a
second per game when 6 games share it. SGLang is a different mouth. Someone built a version of it made for exactly
our graphics card (the RTX Pro 6000). It also has a trick, "guess ahead" (speculative decoding with the model's own
MTP head): a small part of the model guesses the next 3 words, and the big part checks them all at once. When the
guesses are right, 3-4 words come out for the price of one. Same brain, faster mouth, more turns per hour.

## Mechanism

- `prime/llm/sglang.py` (new, ours): installs the offline SGLang wheelhouse `dfranzen/pennyroyal-v253` (SGLang
  0.5.19 fork `d00d88e`, torch 2.13 cu130, flashinfer 0.6.17) into its own venv in `/tmp/sarbloh-sglang` with the
  wheelhouse's own `uv`, sets the CUDA 13 toolkit that ships in the wheels as `CUDA_HOME`, links `libcuda.so`, and
  returns the server environment. It plugs into the existing runtime hook (`ModelSpec.runtimes`), so the watchdog,
  smoke tests, gate and teardown of `prime/llm/vllm.py` are reused unchanged.
- `prime/llm/vllm.py`: a runtime may bring its own Python (`python`) and its metric names (`backend: sglang`); the
  watchdog reads `sglang:*` counters and, when /metrics does not answer, the health endpoint.
- `prime/llm/qwen.py`: three new profiles on the same weights (`Qwen/Qwen3.8-27B-FP8`, Kaggle snapshot
  `fumiyauchiyama/qwen3-8-27b-fp8-hf-snapshot`): `sglang_fp8_mtp_vision` (MTP speculative decoding, 3 steps, + image),
  `sglang_fp8_vision` (no speculation), `sglang_fp8` (text only). The chain for this run is those three, then the
  vLLM profiles as a fallback so the run still yields a trace. Flag names come from the reference launcher that runs
  on this exact wheel (`kaggle/reference/arc-agi-3-milestone-2-solution.ipynb`, cell 12); the Flash-Next-only flags
  (AutoRound, external draft model, FR-Spec token map, PLE offload, MoE runner) are dropped because our model is the
  dense-MLP hybrid 27B (48 linear-attention + 16 full-attention layers, 1 MTP layer; config checked on HF 2026-10-01).
- Nothing in the agent changes. The E109 run uses the E008 prompts (control), so the agent numbers on ls20 compare.

## Why it should work

Decode speed is the binding constraint (2026-10-01 analysis: ~6,400 completion tokens per action, ~4 minutes per
turn, every game ended on the wall clock). The reference team reports this build as the fastest serving stack on
this card; MTP speculation multiplies decode tokens per forward pass when the draft is accepted. More tokens per
second means more turns, and so more actions and levels, inside the same 12 hours.

## Prediction

Committed before the run.
- An SGLang profile starts, passes the native tool-call smoke test and the image smoke test.
- Probe: single stream ≥ 60 tok/s (≥ 1.4x vLLM's 42.6) and 8 streams ≥ 440 tok/s (≥ 1.4x vLLM's 315).
- ls20 runs to the end of its wall clock with 0 server restarts and 0 LLM failures.

## Kill criterion

Committed before the run. Binding.

> Kill (for this wheel and model) if no SGLang profile passes start + smoke tests in this run, or if the 8-stream
> probe is below 1.2x vLLM (378 tok/s) on the profile that started. A start failure is recorded with the engine's
> error lines (server_events.jsonl `start_failed`), not retried blindly.

## Measurement

- `summary.json` → `vllm_profile`, `vllm_attempts` (which SGLang profile started, seconds to start), `throughput`.
- `server_events.jsonl`: installed / runtime_ready / ready / freeze events.
- ls20 agent numbers (turns, completion tokens per turn, seconds per turn) from `results.json` and the transcript,
  compared with today's ls20 (shared GPU, so only seconds per turn and tokens per second are comparable).

## Licence

The wheelhouse dataset's licence is "unknown" on Kaggle (the fork's upstream, SGLang, is Apache-2.0). Fine for an
experiment; **UNCONFIRMED for a prize submission** until the fork's licence is checked. We attach the wheels; we do
not copy the reference launcher code (the installer and flags here are written by us from its documented settings).
