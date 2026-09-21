# ARC-AGI-3 — specification snapshot

**Retrieved 2026-09-21.** Sources: [docs.arcprize.org/games](https://docs.arcprize.org/games),
[arc-agi on PyPI](https://pypi.org/project/arc-agi/), [ARC-AGI Toolkit](https://github.com/arcprize/arc-agi),
[ARC-AGI-3 technical report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf).

This is a snapshot, not a substitute for the SDK. Where this file and the installed toolkit disagree, the
toolkit wins and this file is corrected.

## Environment

- Hand-crafted interactive environments testing abstraction and reasoning. Turn-based.
- Grid: up to **64x64**, each cell an integer **0-15**. Origin `(0,0)` top-left, `(x,y)` ordering.
- Each environment is a **series of levels**. A level ends on its win condition. Early levels teach mechanics;
  later levels require composing them.
- Deliberation is permitted — the benchmark prioritises offline reasoning over real-time reflex.
- Example environment IDs: `ls20` (agent reasoning), `ft09` (elementary logic), `vc33` (orchestration).
  Full list via `arcprize.org/tasks` or the list-environments endpoint.

## Actions

- `GameAction`: **ACTION1-ACTION4** directional, **ACTION5** game-specific, **ACTION6** complex (carries
  coordinate `data`). Documentation also references an undo action.
- **UNCONFIRMED:** the exact enum members and whether undo is ACTION7 or separate in toolkit 0.9.9. Resolve with
  `list(GameAction)` against the installed SDK and correct this file in the same commit.
- Available actions vary by environment; read them from `action_space`.

## Observation

- `FrameDataRaw` carrying `state`, `levels_completed`, and frame data.
- `GameState`: WIN, GAME_OVER (verify the in-progress members against the SDK).

## Toolkit

- Package `arc-agi`, version **0.9.9** (2026-06-10). Python **>=3.12**. Licence **MIT**.
- `pip install arc-agi` / `uv add arc-agi`. API key optional, from `three.arcprize.org`.

### `Arcade`
Entry point. Constructor takes `arc_api_key`, `arc_base_url`, `operation_mode`, `environments_dir`,
`recordings_dir`. Methods: `make()`, `get_environments()`, `create_scorecard()`, `get_scorecard()`,
`close_scorecard()`, `listen_and_serve()`.

### `EnvironmentWrapper`
Properties `observation_space`, `action_space`, `info`. Methods `reset()`, `step(action, data, reasoning)`.

### Operation modes
| Mode | Meaning |
| --- | --- |
| NORMAL | local + API |
| ONLINE | API only |
| OFFLINE | local only |
| COMPETITION | official leaderboard participation |

OFFLINE and COMPETITION share the interface. Local replicas in `eval/real_games/` and the private submission run
through the same agent loop.

### Rendering
`"terminal"` (FPS-bounded), `"terminal-fast"` (unbounded), `"human"` (matplotlib), or a custom renderer.
Removing terminal rendering is reported to give large throughput gains — relevant to the wall-clock budget.

### Recordings
JSONL.

## Scoring — RHAE

Relative Human Action Efficiency. Per level: the ratio of human to agent action count, **squared**, weighted
linearly by level index, capped at **1.15x** to prevent outlier exploitation.

Human baseline = the upper-median best human action count, i.e. the third-best performer among completers.
Derived from 486 participants across 414 candidate environments. Median attempt 7.4 minutes; successful runs
averaged 8.1 minutes. Participants worked under soft 20-minute and hard 30-minute per-environment limits within
90-minute sessions.

All environments are verified 100% solvable by humans with no task-specific training.

**Implication:** the baseline already contains human mechanics-discovery overhead. The agent does not lose by
exploring; it loses by exploring the same thing twice.

## Why agents underperform (from the technical report)

1. **Knowledge-bound reasoning** — reasoning models excel only where domain knowledge and verification exist.
2. **Exploration deficits** — agents struggle to discover unknown unknowns without being told the win condition.
3. **Context management** — 64x64 grids over long traces exhaust token budgets; naive rolling windows fail.
4. **Overfitting** — countered by out-of-distribution private environments with limited mechanical overlap with
   the public set.

Item 3 is the direct justification for `sarbloh.surat`. Item 2 is the direct justification for information-gain
action selection in `sarbloh.jugat`.
