# Experiments, 2026-10-01

Today's runs. One row per experiment: name, short description, result, verdict. The details (hypothesis, config,
results, verdict) live in `experiments/E###_*/`, which follows `experiments/E000_TEMPLATE/`.

**Code:** `Sarbloh-Experimentation/` (a copy of `Sarbloh-Arc/`; the submission code is not touched).
**Games:** the same 5 public dev games in every run: ls20 and tr87 (keyboard), lp85 and su15 (click), tu93
(keyboard+click). Held-out games are never run.

| Where | How | Score from |
| --- | --- | --- |
| Local (plumbing, Qwen3.5-4B) | `.\scripts\serve_llm.ps1`, then `uv run python Sarbloh-Experimentation/prime/run.py --games ls20 tr87 lp85 su15 tu93 --experiment E###_x` | `runs/experimentation_local/results.json` (SDK scorecard) + `trace.md` |
| Kaggle (Qwen3.8-27B-FP8, RTX Pro 6000) | set `EXPERIMENT` / `EXPERIMENT_OVERRIDES` in `kaggle/experimental/sarbloh_experimentation.ipynb`, then `uv run python scripts/kaggle_push.py experimentation` | `experimentation-output` -> `prime_run/results.json` + `LEDGER_ROW` |

A local 4B score is not evidence. It only shows that the code runs. 5 public games are a smoke signal, not a
held-out result.

| # | Experiment | Short description | Local | Kaggle | Verdict |
| --- | --- | --- | --- | --- | --- |
