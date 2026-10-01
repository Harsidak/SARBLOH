# E021 — Explicit versioned goal hypothesis

- **Date opened:** 2026-10-01
- **Axis varied:** substrate (backlog E21)
- **Baseline:** E008_perception_memory as it is (`agent.memory.goal_versioning: False`), same code, same game
- **Split:** none. This run is local, on the public game ls20 only (owner decision 2026-10-01: "implement it locally
  only on ls20"). It is a plumbing smoke, not a result.

## Mechanism

In E008 the goal is one string (`WorkingMemory.goal`). `set_goal` overwrites it silently: the old text and the
reason it changed are lost. Only a won level writes a log line. The prompt says "Keep several goal guesses"
(`prompts.py`), but the `act` tool takes one string, so the agent cannot do that.

E021 adds, behind `agent.memory.goal_versioning` in `Sarbloh-Experimentation/prime/`:
- `WorkingMemory.goals`: up to 3 open goals `g1..g3`. Each has text, status (`active` | `candidate` | `refuted`),
  version, evidence steps, why and a history. Exactly one is active. `WorkingMemory.goal` mirrors the active text,
  so the level-up, levels.jsonl and the lessons graph work as before.
- `WorkingMemory.update_goals`: `act`'s `goal` takes a string (a new version of the active goal, as before) or a list
  (add a rival, switch the active one, refute one). It is all-or-nothing, like the hypotheses. A refuted goal that is
  proposed again is flagged (`reproposed`) and logged; it is not blocked.
- `confirm_goal` at a level-up: the active goal is marked won at that level and stays active; the rival goals are
  cleared; the refuted goals stay for the game.
- The goal block shows the active goal, its rivals and the last 2 refuted goals, under cap `goal_versioned` (250 tokens).
- `recall` scope `goal` shows every version of every goal.
- The `goal` schema of `act`, three prompt lines and the curator's `<goal>` view change only when the switch is on.
- Every goal change is logged as a `goal` event with `goal_id`, `version`, `status`, `kind` (new | version | switch |
  refuted | status) and `reproposed`.

## Why it should work

RHAE squares the action ratio, so actions spent chasing a wrong goal, or one already shown wrong, are pure loss.
When the rivals and the refuted goals are on screen, the agent can choose actions that tell the rivals apart and
does not go back to a refuted goal. Fewer wasted actions per level means a higher RHAE.

## Prediction

Committed before the run.
- **Local (Qwen3.5-4B, ls20, plumbing only):** the agent writes at least one goal through the new path, the
  events are logged, no crash, no new act error type. No score prediction: a 4B score is not evidence.
- **Kaggle (not run today; owner's call):** vs arm A on the 5 dev games, fewer actions to clear level 1, the same
  or more levels, 0 re-proposed refuted goals.

## Kill criterion

Committed before the run. Binding.

> Local: kill (as a bug, fix and rerun) if the run crashes in the new code, or if act errors caused by the `goal`
> argument appear that do not appear with the switch off.
> Kaggle (when it runs): kill if (1) fewer than 2 goals are ever open in a game on 4 of the 5 games (the feature is
> not used), or (2) the act error rate rises by more than 5 points vs arm A. If arm B is not better than arm A on
> levels or on actions per level, the verdict is INCONCLUSIVE, not KEPT (5 public games, one run each).

## Measurement

Evaluation systems used:
- **Local:** `Sarbloh-Experimentation/prime/run.py` on llama.cpp (Qwen3.5-4B Q4_K_M). Score from the SDK scorecard in
  `runs/experimentation_local/results.json`; behaviour from `trace.md` and the `events.jsonl` `goal` events.
- **Tests:** `tests/unit/test_prime_goals.py` (new) and the existing prime suites, run against
  `Sarbloh-Experimentation` with `PRIME_DIR=Sarbloh-Experimentation`.
- **Kaggle (later):** `kaggle_push.py experimentation`, `prime_run/results.json` + `LEDGER_ROW`; per-game RHAE via
  `eval/official_score.py`.
- **Secondary metrics:** goal versions per level, goals open at once, re-proposed refuted goals, act errors from `goal`.

## Amendment 1 (owner, 2026-10-01, after the first local run), written before the code

The owner restated E21: the agent keeps its guess about the win condition as a numbered entry ("goal v3: get the
blue block onto the red cell"). When evidence changes the guess it writes v4 and keeps v3 with the reason it was
replaced, so it neither changes the goal silently mid-plan nor drifts back to a goal it already ruled out. Two
additions:

1. **Goal history in view.** Each history entry now keeps the reason (`why`) with the version it belongs to. The
   goal block shows the last `goal_history_shown` (5) goal changes with their reasons, so when the agent thinks its
   last goal was wrong it can read the 4-5 before it without a tool call. `recall` scope `goal` still shows all of
   them. The cap `goal_versioned` goes from 250 to 400 tokens to fit them.
2. **Goal locked after level 1** (`agent.memory.goal_lock_after_level: 1`, 0 = never). Level 1 is for exploration.
   Once level 1 is won, the goal that won it is kept for the rest of the game: the agent's `goal` argument is ignored
   (the act still runs, with a note), and the block says the goal is locked. Exception (our assumption, UNCONFIRMED
   with the owner): if level 1 was won with no active goal, the agent may set one, which then locks. A new game
   unlocks. The host still marks the goal "won" at later level-ups.

**Added predictions.** Local: the history lines and the lock note appear with no new act error type; a goal edit
after level 1 never changes the goal memory (checked by tests on the real ls20 with a scripted model, because the
4B does not win level 1). Kaggle: the agent sends 0 goal edits after level 1 on most games once it reads the note.
**Added kill criterion (binding):** kill as a bug if a goal edit after level 1 changes the goal memory, or if the
lock makes an act fail (it must only ignore the goal argument).
