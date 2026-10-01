# E001 — Duck + fast vLLM (kaggle/main.ipynb)

- **Date opened:** 2026-09-25
- **Axis varied:** budget (inference throughput), with harness parity as a precondition
- **Baseline:** the Duck, Kaggle offline, ~4.5 (owner's own run of the TufaLabs notebook, all 25 public games, UNCONFIRMED exact value)
- **Split:** public env files on Kaggle Save & Run. This is a smoke test and parity check, not evidence (CLAUDE.md §4.3).

## Mechanism

`kaggle/main.ipynb` runs **Tufa Labs' bundle unchanged**: `jeroencottaar/taaf-kaggle-source-share`, with its
pickled Benchmark and HarnessSolver and the same solver settings, verified locally on 2026-09-25 to equal the
pickle. Only the serving layer changes:
1. The bundle's `setup_commands.json` is not run. Our launcher starts vLLM and exports byte-identical `LOCAL_ANALYZER_*` settings.
2. Profile chain: `fast` (MTP×3, no prefix cache, 32 seqs, 8192 batched tokens), then `fast_nospec` (batching
   knobs only), then `duck` (exact bundle flags). A profile counts only if it starts **and** parses a tool call.
3. A throughput probe (single-stream plus 16-way aggregate tok/s), a watchdog (4 failures → restart, at most 2), and guaranteed teardown.

The ported `sarbloh.harness` (`kaggle/experimental/001_sarbloh.ipynb`) is **not** part of E001. It comes after E001 passes.

## Why it should work

The Duck's score is limited by wall clock: 28 games share a 9 h budget, and many games time out. Faster decoding
gives each game more analyzer turns, which means more levels reached before the deadline. The DuckQwen notebook
reported this profile as its fastest.

## Prediction

> (a) Parity: with `profile=duck`, the offline mean is within ±1.0 of the Duck's ~4.5.
> (b) With `profile=fast`, single-stream tok/s is at least 1.5× the duck profile, and the offline mean is at least as high as (a).

## Kill criterion

> The fast profile does not come up, or it brings no tok/s gain, on 2 runs: drop it and keep `duck`.
> The parity run scores below 3.0: the harness port is at fault. Stop and diff it against taaf before any further experiment.

## Measurement

`summary.json` from the Kaggle run: mean_score, per_game, vllm_profile, and the smoke tok/s from the log. Record the
`nvidia-smi` output in CLAUDE.md §2.
