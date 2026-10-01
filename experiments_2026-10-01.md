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
| 1 | [E021_versioned_goal](experiments/E021_versioned_goal/) | Goal kept as numbered versions ("g1 v3") with the reason each was replaced; up to 3 open (1 active, rivals), refuted ones kept in view; last 5 goal changes shown; goal locked once level 1 is won; switch `agent.memory.goal_versioning` | ls20: 4B run plumbing pass (11 actions, 0 levels, 0 act errors). History + lock checked by a scripted model on ls20. Tests 75/75 + 26/26 suites | not run | USEFUL (owner); score effect still needs Kaggle A vs B |
| 2 | [E022_level_review](experiments/E022_level_review/) | Score-change memory: after every level-up a hidden review reads the whole level and writes how it was won, the rules, wasted actions and notes for the next level into the working memory, the lessons graph and levels.jsonl; switch `agent.memory.level_review` | Tests only (llama.cpp stopped): scripted model on ls20, 1 review per level-up, bad reply logged and skipped. Tests 40/40 + 26/26 suites | not run | PLUMBING PASS; effect not measured |
