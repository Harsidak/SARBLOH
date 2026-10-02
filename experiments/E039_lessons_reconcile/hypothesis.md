# E039 — Lessons reconcile: after every level won, the lessons graph is corrected against the wrong rulebook

- **Date opened:** 2026-10-02
- **Axis varied:** abstraction / consolidation. Backlog E39 ("causal graph induction over object interactions",
  status kept: "works, needs improvement"): the lessons graph is that sub-memory, and this is the improvement the
  owner asked for.
- **Baseline:** E018 arm B (`wrong_rulebook: true`, `skill_names: use_case`) with `lessons_reconcile: false`. Arm B
  of E039 = the same with `lessons_reconcile: true`. Same code, same games.
- **Split:** none yet. Local tests and plumbing only. Not a result.

## Mechanism

Owner's statement (2026-10-02): "Implement E39, that is, improve the lessons graph after every level by comparing both
the lessons and the wrong rulebook to finally update the lessons. The wrong rulebook should be logged constantly, but
after one level is complete both the lessons and the wrong rulebook should be compared and then the lessons updated."

In E008 a lesson, once written, is never corrected: a rule promoted at level 1 stays even if the agent refutes it at
level 3, and two wordings of one rule stay as two lessons. E039 adds, behind `agent.memory.lessons_reconcile` in
`Sarbloh-Experimentation/harness/`:

1. **When.** After every level won: after the E022 review (if on), before the curator. One hidden JSON call, thinking
   off, like the curator (`harness/agent/lessons_reconcile.py`). The agent never sees it; it sees the corrected lessons.
2. **Input.** The lessons (rule and goal nodes, not shapes) with their ids and where they came from: this game's
   first (newest first), then other games' by use, under `lessons_reconcile_tokens` (2000). The wrong rulebook of
   this game (E018), under half that. The newest E022 review, if any.
3. **Output and how it is applied** (`GameMemory.reconcile_lessons`, `LessonsGraph.remove` / `merge`):
   - `drop`: a lesson a wrong rule shows false, or useless. Only a lesson of this game alone can be dropped.
   - `rewrite`: new text for a partly wrong or vague lesson. A lesson other games also taught may only be narrowed.
     The node keeps its type, games, levels, uses and edges (its id changes with its text).
   - `merge`: several lessons saying the same thing become one, with all their games, levels, uses and edges.
   - Guards: only ids the pass was shown; at most 10 changes per pass; a bad item is skipped and logged, the rest
     applied; a reply that does not parse twice is logged as an error and the game carries on.
4. **Logs.** Event `lessons_reconcile` per level, with each change's old and new text and the reason; stats
   `reconciles`, `reconcile_errors`, `lessons_dropped`, `lessons_rewritten`, `lessons_merged`.

Without the E018 rulebook the pass still runs on the lessons alone (merges and rewrites, no wrong rules).

## Why it should work

The lessons graph feeds the pinned "Lessons" block and the curator's skills. A false lesson there makes the agent
act on a wrong rule in the next level and the next game, which costs actions; a duplicate takes room a different
lesson could use. The wrong rulebook holds exactly the evidence needed to find the false ones, and a level-up is the
natural time to look: the level's facts are settled and no action is spent.

## Prediction

Committed before the run.

- **Local (tests, no model):** with the switch off nothing changes. With it on: the pass reads this game's lessons
  first with ids and the wrong rules; drop / rewrite / merge apply with edges and games kept; a lesson of other games
  is never dropped; unknown ids and items over the limit are refused; an unparseable reply raises.
- **Kaggle (not run; owner's call):** vs the baseline on the 5 dev games, fewer `wrong_rule_reproposed` events at
  levels 2+, no fewer levels won, and at least one applied change per game that wins 2 or more levels. Pass time
  (`duration_s`) under 60 s per level (UNCONFIRMED budget).

## Kill criterion

Committed before the run. Binding.

> Local: kill as a bug if the switch-off path changes any existing suite result, or a lesson of another game is
> dropped. Kaggle (when it runs): kill if levels won drop by 2 or more over the 5 games, or if the pass applies no
> change in any game (it does nothing). If neither and RHAE is within noise, INCONCLUSIVE.

## Measurement

- **Tests:** `tests/unit/test_prime_lessons_reconcile.py` (28 checks) and the E018 suite.
- **Kaggle (later):** `kaggle_push.py experimentation` with `EXPERIMENT_OVERRIDES` from `config.yaml`; stats and
  `lessons_reconcile` events from `prime_run/`, `lessons.json` before and after, per-game RHAE via
  `eval/official_score.py`.
