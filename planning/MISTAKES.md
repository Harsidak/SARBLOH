# Mistakes log

Append-only. Anything that cost more than an hour. Newest first.

The entry that matters is the **cause**, not the symptom. "Run crashed" is not an entry. "Reward function failed
silently because the Kaggle environment swallows exceptions in the scoring callback" is an entry.

---

## 2026-09-21 — Reorg moved files without running the tests

**Cost:** the test suite was silently un-runnable for three days. Found 2026-09-24 when setting up the uv env.

**What happened:** the SARBLOH reorg moved `my_agent.py` to `sarbloh/legacy/`, `Kaggle_test.py` to `sarbloh/harness/`
and the suites to `tests/component/`. Every suite's path shim, the runner's glob and `eval/components.py` still
pointed at the old locations: `run_all.py` reported "no suites selected", and each suite failed on import.
The documented command (`pytest tests/`) cannot run these suites at all, because they are scripts, not pytest tests.

**Cause:** the reorg was checked by eye, not by running `tests/run_all.py` afterwards, and the documented test
command had never been executed.

**Rule adopted:** any commit that moves files runs `uv run python tests/run_all.py` and must stay 16/16 green. Every
command documented in `CLAUDE.md` §6 has been executed at least once (verified 2026-09-24).

---

## Prior Kaggle competition — RL forced where hardware made it suboptimal

**Cost:** the competition result; LoRA SFT, not RL, produced the best score.

**What happened:** the owner, whose research background is RL, applied RL to a Kaggle problem because the papers
argue RL is what drives advanced agentic systems. Under Kaggle's hardware limits RL underperformed, and LoRA SFT
won.

**Cause:** the method was chosen from the literature, which assumes cluster-scale compute, rather than from the
competition's actual GPU and wall-clock budget, and it was not tested early against a simpler baseline.

**Rule adopted:** method follows hardware. RL has to beat an SFT or no-training baseline inside the Kaggle budget,
in a measured run, before it is adopted. The default training route is LoRA SFT on verified trajectories.
(`CLAUDE.md` §0)

---

## 2026-08 — Tests moved to the archive

**Cost:** repository verifiability, for roughly six weeks.

**What happened:** fifteen test files were moved to `_archive/tests_2026-08-10/` rather than fixed. Four tests
remained live against an 830 KB agent codebase.

**Cause:** tests were treated as blocking work rather than as the thing that makes work cumulative. Under
schedule pressure the cheapest-looking move was to remove the blocker.

**Rule adopted:** a broken test is fixed or deleted with a recorded reason. Never archived. (`CLAUDE.md` §4.6)

---

## 2026-08 — Results recorded without attribution

**Cost:** three months of experimental results are not reconstructable.

**What happened:** fourteen result files accumulated in `scores/` — `battle2`, `champion`, `memo`,
`llmagent_route` — with no mapping from any of them to the configuration or hypothesis that produced it.

**Cause:** logging was treated as a write-up step to be done later rather than as part of the run.

**Rule adopted:** the ledger row is written by the run. A run that is not logged did not happen. (`CLAUDE.md` §4.1)

---

## 2026-09-21 — Wall-clock ceiling assumed, not verified

**Cost:** low, caught early.

**What happened:** planning proceeded on a 9-hour Kaggle ceiling. The ARC Prize testing policy states 12 hours.

**Cause:** a working figure carried forward across sessions without being checked against the source.

**Rule adopted:** competition constants live in `CLAUDE.md` §2 with a verification date, and anything unverified
is marked UNCONFIRMED rather than used silently.
