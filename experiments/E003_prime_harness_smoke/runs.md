# E003 run log

## v1 — 2026-09-27 18:55 UTC — FAILED before any game (not a scored run, no ledger row)
- Kernel `banwait13/sarbloh-prime` v1, RTX PRO 6000 (sm120), internet off, commit 4400ab0.
- Install OK in 119 s: vLLM 0.19.0, torch 2.10.0+cu128, transformers 5.5.0, hub 1.33.0 (overlay works).
- Gemma-4-31B NVFP4 weights load OK: 29.96 GiB in 63 s; torch.compile 42 s. vLLM picks `NvFp4LinearBackend.FLASHINFER_CUTLASS`.
- All three profiles die at FlashInfer JIT of `fp4_gemm_cutlass_sm120`: `/usr/bin/ld: cannot find -lcuda`.
  Cause: the image has only the driver's `libcuda.so.1`; no unversioned `libcuda.so`, no CUDA stub. Profiles cannot
  help because all share the NVFP4 backend.
- The notebook printed only the APIServer traceback ("See root cause above"); the cause was in `vllm-server.log`.
- Fix (prime/vllm.py): symlink `libcuda.so` -> the driver lib, prepend its dir to LIBRARY_PATH for the server;
  failure print now shows the engine's error lines. UNCONFIRMED until v2 runs.

## v2 — 2026-09-27 ~19:40 UTC — RAN, scored 0.0005 mean (ledger row written)
- Kernel v2, commit a26b167, RTX PRO 6000, internet off. Outputs: `runs/kaggle_sarbloh-prime_v2_20260928_011233/`.
- **libcuda fix CONFIRMED:** `gemma_fast` up in 215 s, `tool_mode=native`, 0 restarts. 35.4 tok/s single, 275 tok/s @ 8.
- Games: 6/6 ran, 0 crashes, all ended `action_budget` (300 each). 1/40 levels (vc33 L1), mean score 0.0005.
- Wall 842 s of 3600: the action cap ended the run at 14 min; 46 min unused.
- 233 turns, 230 tool calls (229 native, 1 fenced), 21 cell errors. 5.95M prompt / 99k completion tokens.
- **Prime-specific machinery never fired:** compactions 0 (context peaked ~40k, threshold 96k), children 0 (no
  `rlm.spawn`), `rlm.harness` calls 0 (no memory, skill, prompt note or refinement written). Reasoning present on
  47/233 turns only.
- Action spending: 1,522/1,800 actions (85%) went in turns that spent >=10 actions each via Python loops. Worst:
  ft09 spent 279 actions in one cell (`for y, x: if grid==8: arc.step(6,x,y)`). Resets: vc33 24, sp80 16, ls20 6.
- ls20: the model read the step-counter bar on row 61 as the goal ("paint the row") and optimised it.
