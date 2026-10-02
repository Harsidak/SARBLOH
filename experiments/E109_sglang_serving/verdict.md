# E109 verdict: INCONCLUSIVE on ls20. Infrastructure failure, superseded by E112

Run: `banwait13/sarbloh-experimentation`, 2026-10-01 21:12-21:39 UTC. ls20 only, concurrency 1, 400 actions, 60 min
game wall. The status is CANCEL_ACKNOWLEDGED at about 16 min of play. Output is in
`runs/kaggle_sarbloh-experimentation_20261002_035050/`.

## What happened (server_events.jsonl + vllm-server.log)
| UTC | event |
| --- | --- |
| 21:13:42 | Pennyroyal venv installed in 46 s |
| 21:13:52 | `sglang_fp8_mtp_vision` start_failed: `--linear-attn-decode-backend flashinfer on SM100+ requires --mamba-ssm-dtype bfloat16` |
| 21:22:28 | `sglang_fp8_vision` ready after 516 s (weight load 374 s cold, prefill CUDA graphs 104 s) |
| 21:22:51 | probe: **45.3 tok/s single stream, 342 tok/s at 8 streams** (vLLM 0.19 baseline: 42.6 / 315) |
| 21:25:09 | watchdog "frozen" -> server killed, restarted (ready 21:26:04) |
| 21:30:07 | watchdog "frozen" again -> fallback to vLLM `qwen_fp8_vision` (install 119 s, ready 21:35:56) |
| ~21:39 | cancelled: 0 levels, **1 action**, 3 LLM calls (24k prompt / 10.6k completion tokens) |

## Root cause: the watchdog, not SGLang
- **The server was not frozen.** At the moment of the first kill, `vllm-server.log` shows one request decoding
  steadily at 43.4-43.5 tok/s, with the context at 12.4k tokens and growing 40 tokens per log line.
- **Why the watchdog thought it was.** Its progress signal is `prompt_tokens_total + generation_tokens_total`.
  SGLang increments these only when a request *finishes* (`observe_one_finished_request`). A single thinking turn
  of more than 120 s (4720 tokens at 45 tok/s is about 105 s, plus prefill) never moved the counter, so the
  watchdog called it frozen.
- **The stack dump is a symptom.** The "Fatal Python error: Aborted" stacks are our own dump signal on kill.
- **Why it only showed up with SGLang.** vLLM updates its counters every step, so vLLM runs never hit this.
- **Fix (2026-10-02, both `Sarbloh-Arc` and `Sarbloh-Experimentation` `prime/llm/vllm.py`).** `token_counters` also
  counts `sglang:realtime_tokens_total`, which is incremented every decode iteration. The test is
  `test_prime_sglang.py::sglang_long_request_moves`.

## What ls20 tells us about the model and agent
Not enough to score anything; the agent got one action in. Its behaviour in the turns it had was sound:
- Turn 1 inspected the scene for free.
- Turn 2 mapped the grey maze in ipython.
- Turn 3 proposed hypotheses: player identity (h1/h2), floor/walls (h3), goal = reach the orange/blue target (h4)
  and HUD bar = budget (h5). It then took one diagnostic action (A2).
- That action only changed the HUD (the yellow bar shrank 84->82), which already speaks against h1 and h2 as
  stated.

Per-turn cost on this stack was 2-6k completion tokens, i.e. 45-130 s per turn at 45 tok/s.

## Verdict
- **Killed as an FP8 serving path.** It is no faster than vLLM untuned (45 vs 43 tok/s), and MTP was never tested
  because of the missing flag.
- **Superseded by E112.** NVFP4 + MTP5 on the same SGLang build measured 125 tok/s single stream and 1162 tok/s at 10
  thinking streams.
- **The ls20 question moves to the E112 ls20 games run** (`banwait13/sarbloh-prime` v15, pushed 2026-10-02 03:55
  local), on the same settings: ls20, concurrency 1, 400 actions, 60 min.
