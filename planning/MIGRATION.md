# Migration — move list

Everything here is a move **you** execute in File Explorer. This session has file read/write on the folder but
no shell on your machine, so it can create files and directories but cannot move, rename or delete them.

The skeleton is already in place. Nothing below overwrites anything; every destination directory exists.

**Order matters.** Group 0 before `git init`. Everything else after the initial commit.

---

## Group 0 — before `git init`

| # | From | To |
| --- | --- | --- |
| 1 | `avo\` | `resources\repos\avo\` |

Reason in `GIT_SETUP.md` §1. Do not skip.

---

## Group 1 — freeze the monolithic agents

| # | From | To |
| --- | --- | --- |
| 2 | `my_agent.py` | `sarbloh\legacy\my_agent.py` |
| 3 | `my_agent_llm.py` | `sarbloh\legacy\my_agent_llm.py` |
| 4 | `my_agent original.py` | `sarbloh\legacy\my_agent_original.py` *(rename — drop the space)* |
| 5 | `check_statencoder.py` | `sarbloh\legacy\check_statencoder.py` |
| 6 | `Kaggle_test.py` | `sarbloh\harness\kaggle_test_legacy.py` |

Frozen, not refactored. See `CLAUDE.md` §3 (legacy) for the extraction protocol and log.

---

## Group 2 — reclaim the tests

| # | From | To |
| --- | --- | --- |
| 7 | all 15 files in `_archive\tests_2026-08-10\` | `tests\component\` |
| 8 | `tests\test_replay.py`, `test_scorecard.py`, `test_verifier.py` | `tests\component\` |
| 9 | `tests\run_all.py` | `tests\run_all.py` *(stays)* |

Expect failures on first run. Fix them or delete with a recorded reason. Do not move any of them back.

---

## Group 3 — eval spine

| # | From | To |
| --- | --- | --- |
| 10 | `heldout_split.json` | `eval\splits\heldout_split.json` |
| 11 | `component_suite.json` | `eval\suites\component_suite.json` |
| 12 | `_archive\harnesses_2026-08-06\rhae_eval.py` | `eval\probes\rhae_eval.py` |
| 13 | `_archive\harnesses_2026-08-06\probe_*.py`, `dbg_*.py` | `eval\probes\` |

`eval/` itself does not move. It is the one part of the repository that was already right.

---

## Group 4 — resources

| # | From | To |
| --- | --- | --- |
| 14 | `Docs\*.pdf` (5 files) | `resources\papers\` |
| 15 | `Docs\_challenge_paper.txt`, `_comp_research.txt`, `_ewm_paper.txt` | `resources\papers\` |
| 16 | `Docs\research\*.md` (4 files) | `resources\papers\notes\` |
| 17 | `Docs\arc_agi_3_architecture_flowchart.svg` | `resources\papers\` |

---

## Group 5 — architecture notes become decisions

| # | From | To |
| --- | --- | --- |
| 18 | `Docs\goalmodel_architecture.md` | `planning\decisions\ADR-0002-goalmodel.md` |
| 19 | `Docs\stategraph_architecture.md` | `planning\decisions\ADR-0003-stategraph.md` |
| 20 | `Docs\routing_architecture.md` | `planning\decisions\ADR-0004-routing.md` |
| 21 | `Docs\system_v2_architecture.md` | `planning\decisions\ADR-0005-system-v2.md` |
| 22 | `Docs\world_model_objectives.md` | `planning\decisions\ADR-0006-world-model-objectives.md` |
| 23 | `Docs\factored_model_design.md`, `stageB_design.md`, `feedback_loop_plan.txt` | `planning\decisions\` |

These were already architecture decision records; they were just not labelled or dated. Add a status line
(accepted / superseded) to the top of each as you touch it. Several are probably superseded already — saying so
is more useful than leaving them ambiguous.

---

## Group 6 — history

| # | From | To |
| --- | --- | --- |
| 24 | all 14 files in `scores\` | `history\legacy_scores\` |
| 25 | `SESSION_CONTEXT.md` | `planning\SESSION_CONTEXT_2026-08.md` |

`SESSION_CONTEXT.md` is 100 KB. It is a transcript, not a context file — `CLAUDE.md` replaces its operational
role. Harvest anything still true out of it into `CLAUDE.md` or an ADR, then leave the rest as an archive
document.

---

## Group 7 — scratch

| # | From | To |
| --- | --- | --- |
| 26 | `lp85\`, `tu93\`, `vc33\` | `runs\` |
| 27 | `agent_checkpoint.pkl` | `runs\` |
| 28 | `v-o-i-d.ipynb`, `v-o-i-d-agent -original.ipynb` | `notebooks\` |
| 29 | `__pycache__\` | **delete** |

---

## Group 8 — archive stays put

`_archive\dead_files\`, `_archive\logs\`, `_archive\runs\`, and whatever remains in
`_archive\harnesses_2026-08-06\` after Group 3 stay exactly where they are. `_archive/runs/` and
`_archive/logs/` are gitignored — roughly 20 MB of traces that belong on disk and in the zip backup, not in git
history.

Nothing in `_archive/` is authoritative. If code there is needed, it is read and reimplemented, not restored.

---

## After the moves

```bash
python -m pytest tests/ -q          # expect red. that is the point.
python -m eval.scoreboard
```

The first green suite is the moment this repository becomes cumulative again. Until then every result is a
claim, not a measurement.
