# E021 local plumbing smoke, ls20 (2026-10-01)

**Plumbing only. A 4B score is not evidence about the 27B.** Not a scored run, so no ledger row (same rule as the E008
local smoke). The ledger row comes with a Kaggle run.

- Model: Qwen3.5-4B Q4_K_M + mmproj (`scripts/serve_llm.ps1 -Vision`), llama.cpp, 16k context.
- Command: `uv run python Sarbloh-Experimentation/prime/run.py --games ls20 --max-actions 30 --minutes 12 --vision
  --ctx 16384 --goal-versioning --experiment E021_versioned_goal --out runs/experimentation_local/E021_ls20`
- Code: working tree on top of 109bebb (E021 changes uncommitted). Config hash `a42273c69097`.
- Output: `runs/experimentation_local/E021_ls20/` (`results.json`, `trace.md`, `games/ls20-*/transcript.jsonl`,
  `report.md`), console log `runs/e021_ls20_local.log`.

## Tests (before the run)
- `tests/unit/test_prime_goals.py` (new): 51 checks, 0 fails. It covers parsing, the string and list forms, the max of
  3 open goals, one active goal, all or nothing, reproposal flagging, the block, recall, save and load, level-up, new
  game, the switch-off path (E008), the act schema and the prompt lines. It also has a scripted end-to-end run (a fake
  model plays the real ls20 through `AgentSession` with the switch on): goal events, the reproposal note, a bad goal
  id refused without spending an action, the E021 system prompt, the schema and the pinned block.
- `tests/run_all.py` (default, Sarbloh-Arc): 24/24 suites green, 1583 checks, 2 known warnings.
- The prime suites against the copy (`PRIME_DIR=Sarbloh-Experimentation`, switch off): memory, context, curator,
  tools, fidelity, perception_views and llm_server all 0 fails. So arm A (E008) behaves as before.

## Run
| | |
| --- | --- |
| Exit | 0, stopped at the 12 min soft deadline (`cancelled`) |
| Actions | 11 of 30, level 0 of 7, score 0.0 |
| act calls / act argument errors | 11 / **0** |
| LLM calls / failures | 27 / 0 (171k prompt, 18.5k completion tokens) |
| Compactions / curator runs / curator errors | 7 / 6 / 0 |
| Hypothesis events | 21 |
| Goal events | 1: `g1 v1 active new` "move object 2 to touch the red objects (16-18) on the right edge" (turn 1) |
| Goals open at once (max) | 1 |
| Re-proposed refuted goals | 0 (none refuted) |
| System prompt has the 3 E021 goal lines | yes |

## What it shows
- The new path runs on a real model without errors. The 4B used the string form, which went through `update_goals` and
  became g1 v1 (active). The event, `working.json` (`goals`, `next_g`) and the `report.md` goal table are all correct.
- The 4B never kept a rival goal, never changed its goal and never sent a list. So the list path (rivals, switch,
  refute) is only covered by the tests, not by a live model. Whether the 27B uses it is the Kaggle question.
- Compared with the E008 local smoke (Sarbloh-Arc, same settings): no new error type. That smoke had 2 goal changes
  in 9 actions, this one had 1 goal and 0 changes in 11 actions. With a 4B model and one run each, the difference
  means nothing.

## Amendment 1 (2026-10-01): history with reasons, lock after level 1

No new live run: the owner stopped the llama.cpp server, and the 4B never wins level 1, so it could not reach the
lock anyway. Checked by tests instead (scripted model on the real ls20):
- `tests/unit/test_prime_goals.py`: 75 checks, 0 fails. New checks: each history entry keeps its reason, demotion
  says which goal became active, a new version keeps the old text and the reason, the block lists the last 5 changes
  in write order, rivals cleared at a level-up keep their history (`[cleared]`), the lock (string, list and invalid
  list ignored, the rest of the act written, `update_goals` refuses, survives save/load, still marked won at level 2),
  no lock when level 1 is won without an active goal (the first goal set then locks), a new game unlocks, the prompt
  lock line.
- `tests/unit/test_prime_level_review.py` (E021 + E022 on, forced level-up): the act result says the goal is now
  locked, a later goal edit gets the lock note and a `goal_locked` event while its actions run, the goal stays "reach
  the exit", and the pinned block shows the lock line and the reason of the cleared rival.
