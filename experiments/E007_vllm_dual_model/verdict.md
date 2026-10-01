# E007 verdict

## v3 (2026-09-30, kernel `banwait13/sarbloh-vllm-bench`, outputs `runs/kaggle_sarbloh-vllm-bench_20260930_210107/`)

RTX PRO 6000 Blackwell, vLLM 0.19.0 wheelhouse, 10 streams. No games were played, so there is no RHAE claim.

| Profile | Start | Tools | Decode 1 / 10 streams (tok/s) | Spec accept | Cold prefill (tok/s) | agent_like turn 1 / later (s) | Prefix hit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen_nvfp4_mtp_pc | **failed** | | | | | | |
| qwen_nvfp4_mtp | **failed** | | | | | | |
| qwen_fp8_mtp | 290 s | native | 82.1 / **648.4** | 0.54 | 8548 | 33.3 / 8.1 | n/a |
| qwen_fp8 (Duck flags) | 98 s | native | 42.6 / 388.5 | n/a | 8682 | 16.5 / **2.0** | 0.63 |
| gemma_turbo (ngram) | 237 s | native | 36.1 / 336.3 | 0.21 | 6439 | 17.2 / 4.8 | 0.80 |
| gemma_fast | 89 s | native | 35.4 / 337.5 | n/a | 6392 | 17.6 / 4.5 | 0.80 |

**Stress tests** (10 min, 10 sessions, engine SIGSTOPped after 120 s):

| Model and profile | OK | Failed | Detected after | Ready again after | Stack dump | Restarts |
| --- | --- | --- | --- | --- | --- | --- |
| qwen_fp8_mtp | 240 | **0** | 120 s | 173 s | yes | 1 |
| gemma_fast | 396 | **0** | 120 s | 173 s | yes | 1 |

**Against the prediction and the kill criteria:**
- **Watchdog redesign: KEPT.** Both freezes were detected at 120 s (prediction: 150 s or less) and recovered in 173 s
  (prediction: 10 min or less), with 0 unrecovered requests and a faulthandler stack dump. This replaces E006's
  30-minute stalls.
- **Speed.**
  - Qwen best: 648 tok/s at 10 streams, 2.8x E006's 233. PASS.
  - Gemma best: 337 tok/s at 10 streams, above 233. PASS.
- **Prefix-cache hit rate of at least 50%.** Met by qwen_fp8 (0.63) and Gemma (0.80). The MTP profiles ran without
  prefix caching.
- **Killed:** `qwen_nvfp4_mtp_pc` and `qwen_nvfp4_mtp` failed to start. The error was `There is no module or
  parameter named 'lm_head.weight_scale' in Qwen3_5ForCausalLM`. The overseer66 model is a copy of
  `unsloth/Qwen3.8-27B-NVFP4`, a mixed compressed-tensors quant whose FP8 lm_head vLLM 0.19 cannot load.
- **Killed:** `gemma_turbo`. It is not faster than `gemma_fast` at 10 streams (336 vs 338). N-gram acceptance was
  only 0.21 and the startup is 2.7x longer. `gemma_fast` stays first in the Gemma chain.
- **Finding:** on the agent-shaped workload, prefix caching beats MTP. qwen_fp8 finished 10 sessions of 5 turns in
  49 s, against 96 s for qwen_fp8_mtp. MTP wins raw decode, but an append-only agent is prefill-bound.

## v4: Qwen3.8-Flash-Next NVFP4 (owner decision). Pending.

See the hypothesis.md addendum. The profiles are `qwen_flash` (Duck/Keith recipe), `qwen_flash_pc` and `qwen_flash_pc_64k`.

### v4 result (2026-09-30, outputs `runs/kaggle_sarbloh-vllm-bench_20260930_222723/`)

Runtime unpack and patch took 111 s. Every pin check passed.

| Profile | Start | Tools | Decode 1 / 10 streams (tok/s) | Spec accept | Cold prefill (tok/s) | agent_like turn 1 / later (s) | Prefix hit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen_flash | 1084 s | native | **119.4** / 290.5 | 0.55 | **13258** | 11.2 / 15.9 | n/a |
| qwen_flash_pc | 871 s | native | 134.9 / 295.7 | 0.56 | 13708 | 20.3 / 19.6 | **0.087** |
| qwen_flash_pc_64k | **failed** in `_initialize_kv_caches` (65k does not fit) | | | | | | |

**Stress (`qwen_flash_pc`): FAILED.** 8 requests OK, 52 failed. The freeze was detected at 120 s and the stacks were
dumped. The restart then died at weight load with `ValueError: could not determine the shape of object type
'torch.storage.UntypedStorage'`, the fallback to `qwen_flash` failed at once, and the watchdog gave up.
- Hypothesis (UNCONFIRMED): the SIGKILLed server's PLE CPU-offload state (shared memory / offload worker) survives
  the kill.
- Keith's `serving_teardown.py` (1,500 lines) cleans this up; our `kill()` does not.

**Against the v4 prediction and the kill criteria:**
- `qwen_flash` starts and uses tools natively: PASS.
- Decode at 10 streams is at least 648: **FAIL** (290). Single-stream decode (119) and prefill (13.3k) are the best
  measured.
- `qwen_flash_pc` later turn at most 4 s: **FAIL** (19.6 s; prefix caching gets almost no hits on this hybrid model).
- Stress, 0 failed requests: **FAIL**.
- **Killed:** `qwen_flash_pc_64k` (did not start) and `qwen_flash_pc` (slower on agent_like). The chain is now `["qwen_flash"]`.
- **Open, not killed:** Flash-Next as the Qwen model. It starts, but freeze recovery is broken and 32k is below what
  the agent's compaction needs.
- **Next (NOT BUILT):**
  - port the teardown of the PLE offload into a runtime `cleanup` hook;
  - try `--kv-cache-dtype fp8` with fewer seqs for 64k;
  - re-bench.
