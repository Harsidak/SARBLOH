# E006: dedicated tools, planner and REPL world model

- **Date opened:** 2026-09-29
- **Axis varied:** action policy (interface), with verification (world-model check) and consolidation (memory tools)
- **Baseline:** E005 10-game run (`toolset: ipython`, Gemma-4-31B-IT-NVFP4): 1/76 levels, 350 resets, a 320-action cell
- **Split:** tune (public games, never the held-out eight)

## Mechanism

The model gets eight native tools instead of one (`prime/agent/tools.py`, config `agent.toolset: "dedicated"`):
- `act(actions, expect)`: at most 5 actions per call, with a required prediction. Each call returns an object-level
  change summary (`perception.summarize_change`). `arc.step` and `arc.reset` in the REPL are refused for the root.
- `reset_level(reason)`, `remember(kind, title, content, evidence)`, `recall(query)`, `delegate(name, task)` and
  `message(to, text)` wrap existing harness calls. `plan(goal, phase, hypotheses, steps)` lives in `prime/agent/Planner.py`.
- `ipython` is unchanged. The new REPL module `wm` (`runtime/skills/worldmodel.py`) holds the agent-written
  `step()`, replays every transition to check it (`wm.check`) and searches it (`wm.plan`). If a model is registered,
  `act` checks each step against it and stops at the first wrong prediction.
- `trace.py` writes one report per game covering actions and causes, beliefs over time, living code, waste and run
  context, plus `sft_levels.jsonl` (one row per level, tagged `level_cleared` and `rhae`).

## Why it should work

In E005 every action was a line of code, so the agent wrote loops that spent 320 actions blind and used 350 resets as
experiments. A capped `act` with a required `expect` makes each action a stated test, and a required `reason` makes each
reset a deliberate choice. Memory, planning and delegation were buried in a long prompt and were used once in 10 games;
as tools they are visible at every turn.

## Prediction

Committed before the run.
- Local plumbing (Qwen3.5-4B, ls20, stop after 3 levels): all eight tools are called at least once, and no act call
  spends more than 5 actions. This is not a score claim: a 4B model's score is not evidence.
- Kaggle (Gemma-4-31B, same 10 games as E005): the largest single spend drops from 320 to at most 5 by construction.
  Resets drop from 8% of actions to at most 4%. At least 2 of 76 levels are solved (E005: 1).

## Kill criterion

Binding. Kill the dedicated toolset if, on the Kaggle 10-game rerun, it clears fewer levels than E005 (fewer than 1 of 76),
**or** more than 20% of `act` calls fail on argument errors. Either means the interface costs more than it saves.

## Measurement

- `trace.py` report: tool-call counts, act batch sizes, resets with reasons, repeated (state, action) pairs, living-code
  ratio, and actions after the last new belief.
- Secondary: turns to the first `wm.check` pass, and the share of act steps checked by a world model.
- Seeds: 1 (plumbing). The Kaggle rerun is the owner's action.
