# E112 verdict: KEPT. NVFP4 + MTP on SGLang clears 500 tok/s at 10 streams (2x headroom)

Run: `banwait13/sarbloh-prime` v14, 2026-10-02 03:33-03:48 local (Kaggle 22:03-22:18 UTC), RTX Pro 6000
Blackwell. The working tree from git 4d356a4 (dirty) was embedded in the notebook. Raw output is in `results.json`
and `runs/kaggle_sarbloh-prime_20261002_034833/`. v13 never measured anything (wheel names, see MISTAKES 2026-10-02).

| profile | start s | d1 | d8 | d10 | **think10** | accept len | prefill tok/s | agent later-turn s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sglang_nvfp4_mtp_vision` (NEXTN 3) | 175* | 126.4 | 532.4 | 1026.7 | **1070.2** | 2.70-2.80 | 10655 | 1.80 |
| `sglang_nvfp4_mtp5_vision` (NEXTN 5) | 82 | 124.7 | 539.1 | 1030.0 | **1162.4** | 2.85-3.21 | 10642 | 1.67 |
| `sglang_nvfp4_vision` (no spec) | 89 | 68.2 | 366.8 | 595.9 | **590.1** | - | 11109 | 2.32 |
| `sglang_nvfp4_mtp_frspec_vision` | failed | | | | | | | |

\* The first launch includes the venv install (63 s) and the page-cache warm-up.

How the numbers are measured: tok/s = sum of `usage.completion_tokens` / wall clock, including TTFT.
- `think10`: 10 parallel thinking-on puzzle requests with the Duck's sampling, up to 2048 tokens each, stopping at
  EOS. This is the agent's real decode shape.
- `d10`: 512 forced tokens per stream.

## Against the prediction
- **Prediction: >= 500 tok/s with MTP. Result: confirmed.** think10 reached 1070 (NEXTN 3) and 1162 (NEXTN 5).
- **No speculation, predicted at ~350-450: higher.** It measured 590-596, which on its own also clears 500.
- **MTP acceptance length, predicted >= 2.0: confirmed.** It measured 2.7-3.2, and 5 steps beat 3 on thinking text.
- All three working profiles passed the native tool-call smoke test, the image smoke test and the thinking probe.

## Defects and open questions
- **FR-Spec is incompatible with this checkpoint.**
  - The error is `mat1 and mat2 shapes cannot be multiplied (10x5120 and 2560x65536)` in draft CUDA graph capture.
  - Cause: the hot-token map slices the lm_head, but here the lm_head is packed NVFP4 (5120/2 = 2560 bytes per row).
  - The arm is dropped; MTP alone already gives 2x headroom.
- **d8 is anomalous and UNCONFIRMED.**
  - Per stream it reads 66 tok/s, against 103 at d10 and 125 at d1, in all three arms.
  - The likely cause is warm-up (the first multi-stream batch right after d1), not a batch-8 penalty, but this is
    unmeasured.
- **The tokenizer hash (0997f410…) differs from the reference's FR-Spec hash.** This matters only for FR-Spec.
- **Quality on ARC is NOT measured.** The next step is a games run on `sglang_nvfp4_mtp5_vision`, against the
  Duck's 6 RHAE on NVFP4/vLLM.
- **This is a serving bench, not a scored run,** so there is no RHAE ledger row (CLAUDE.md §4.1 covers scored runs).

## Recommendation
- Put `sglang_nvfp4_mtp5_vision` first in `SGLANG_CHAIN`, then `sglang_nvfp4_mtp_vision`, then `sglang_nvfp4_vision`,
  then the vLLM FP8 profiles.
- Remove FR-Spec from `SERVE_BENCH`.
- Use 5-6 MTP steps for the agent: thinking text is where the agent spends its tokens, and that is where NEXTN 5 wins.

## Follow-up: ls20 games run on `sglang_nvfp4_mtp5_vision` (2026-10-02, sarbloh-prime v15)
Settings: E008 agent, ls20 only, concurrency 1, 400 actions, 60 min game wall. They match E109's ls20 run. The
watchdog fix (`sglang:realtime_tokens_total`) was in. Output is in `runs/kaggle_sarbloh-prime_20261002_050009/`.

| | E109 SGLang FP8 (2026-10-01) | E008 vLLM FP8 (6 games sharing, 2026-10-01) | **E112 SGLang NVFP4 + MTP5** |
| --- | --- | --- | --- |
| server start | 516 s, then 2 false freezes, then fallback to vLLM | ~350 s | **176 s** (+61 s install), **0 restarts / 0 freezes** |
| decode tok/s in game | 43-45 (1 stream) | ~28 per game (168 shared over 6) | **151 mean per stream** (3436 decode logs at 1 running request), accept len **3.81** |
| turns / LLM calls | 3 calls in 16 min | 25 turns in 90 min | **101 turns / 126 calls in 60 min** |
| actions | 1 | 16 | **122** |
| levels | 0 | 1 | **1 of 7**: L1 in **15 actions (human 22)**, L2 107 actions not cleared (human 123) |
| score | 0 | - | 3.57 (RHAE 0.0357) |

## Reading
- **Serving is fixed.** No crash, freeze or fallback, and every one of the 126 calls succeeded.
- **Speed is about 5x E008's per-game speed.** The agent gets about 4x the turns per hour.
- **The time is still almost all decoding.** The run produced 522k completion tokens; at 151 tok/s that is about
  3460 s of the 3600 s wall. Prefill is cheap: 1.8M new tokens at about 10k tok/s is about 3 min. Prefix-cache hit
  was only 36%, because of 11 compactions and new state pushed every turn.
- **One game wastes most of the GPU.** It ran one request at a time for 99.9% of decode steps. The bench shows the
  card does about 1100 tok/s at 10 streams, so one game uses about 14% of the capacity.
  - The lever is **concurrency**: run 8-10 games at once, at about 100 tok/s each instead of 151.
  - A 10-game wave would then cost about the same wall time per game as this single game did.
- **The agent, not serving, now limits ls20.**
  - L1 was cleared efficiently: 15 actions against the human's 22, which scores at the cap.
  - On L2 the agent spent about 45 min and 107 actions without clearing it, and the run ended on the wall clock.
  - With a 3 h per-game wall (the submission config) it would get about 300 turns.
  - Whether L2 falls is an agent question for the E008 line, not E112.
- **Public game, one seed:** a smoke result, not evidence (CLAUDE.md §4.2-4.3).
