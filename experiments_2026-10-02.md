# Experiments, 2026-10-02

Today's runs. One row per experiment: name, short description, result, verdict. The details (hypothesis, config,
results, verdict) live in `experiments/E###_*/`, which follows `experiments/E000_TEMPLATE/`.

**Code:** `Sarbloh-Experimentation/` (a copy of `Sarbloh-Arc/`; the submission code is not touched).
**Games:** the same 5 public dev games in every run: ls20, tr87, lp85, su15, tu93. Held-out games are never run.

| # | Experiment | Short description | Local | Kaggle | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | [E018_wrong_rulebook](experiments/E018_wrong_rulebook/) | Refuted rules go to a separate cross-game "wrong rulebook" the moment the agent refutes them. It is shown as its own block, and a hypothesis that restates one gets a note and is counted. Verified rules keep the lessons → skills path. Skills are named by their use case (≤ 10 words). Switches `agent.memory.wrong_rulebook`, `agent.memory.skill_names` | tests 34/34; existing suites 0 fails up to the missing-`sarbloh` import (not caused by E018) | not run | PENDING (needs Kaggle A vs B) |
