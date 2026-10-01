# E004 — Forced Continual Harness + per-cell action cap (local Qwen, ls20, 1–2 h)

- **Date opened:** 2026-09-28
- **Axis varied:** harness control (L1/L3 made host-driven instead of model-opt-in; L2 action spending capped)
- **Baseline:** E003 v2 (Gemma-4-31B, Kaggle): 0 compactions, 0 spawns, 0 `rlm.harness` writes, 85% of actions
  spent in >=10-action cells.
- **Split:** local OFFLINE `eval/real_games` ls20, plumbing/behaviour only. A 4B score is not evidence (§6).

## Mechanism
E003 showed every Prime-specific mechanism is opt-in, and the model never opts in. E004 moves the decision to the host:
1. `max_actions_per_cell` (8): `ArcHost` refuses the 9th `arc.step` in one cell. No blind loops.
2. Reflection checkpoints: every 25 actions, on level-up, on GAME_OVER, the host injects a REFLECT message. Until
   the harness file changes (a memory/skill/prompt note written or updated), `arc.step` is refused. Max 2 re-asks,
   then released (logged as `reflection_skipped`) so a model that cannot comply does not deadlock.
3. Counters: `reflections`, `reflection_skipped`, `harness_entries` per kind, `cell_cap_hits` in session stats.
Compaction already triggers at 12k tokens locally (16k context), so a 1–2 h run exercises L1.

## Prediction (committed before the run)
1. >=5 harness entries written over the run (memories + skills), >=1 entry *updated* later (self-improvement).
2. >=1 compaction, and after it the model reads its harness/REPL state rather than re-exploring from scratch.
3. No cell spends more than 8 actions; `cell_cap_hits` > 0 shows the cap binding at least once.
4. Score: not predicted (4B model).

## Kill criterion
If after 60 min Qwen has 0 successful harness writes with checkpoints forcing it (all `reflection_skipped`), the
forcing mechanism does not work for this model class: stop and redesign (host-written memory), do not extend.

## Measurement
`runs/prime_local_E004/games/ls20-*/transcript.jsonl` read live, turn by turn; `harness/harness_state.json`
diffed over time; `session.json` counters.

## Amendment (2026-09-28, written as the main run started)
The 5-min check (`runs/prime_local_E004_check`) found (a) our relative harness path made the kernel write to
`work/runs/...` so the host never saw writes (local only; Kaggle paths were absolute), (b) `create_skill` needs a
Python `reference` the prompt never mentioned, and (c) upstream has a **host-driven auto /refine** subsystem
(`core/refinement/refinement.ts`, triggers in `agent-session.ts`: every 25 turns + after compaction, 20 min
cooldown, review gate then JSON edits applied by the host) that our port omitted. That omission, not only
Gemma, explains E003's 0 harness writes.
Main run therefore tests the upstream mechanism (`prime/refine.py`), not the lock checkpoint (kept, off):
`--cell-cap 8 --auto-refine 15 --refine-cooldown-min 5 --minutes 90 --max-actions 400`, Qwen3.5-4B, thinking on.
Predictions 1–3 unchanged; entries now come from auto-refine (`source: refine`) and/or the agent.
Kill criterion restated: 0 applied refine edits after 60 min => stop and diagnose.
