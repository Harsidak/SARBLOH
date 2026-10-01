# E021 — verdict

- **Date:** 2026-10-01 (local part, with amendment 1)
- **Outcome:** **USEFUL** (owner's verdict, 2026-10-01). The mechanism is kept in `Sarbloh-Experimentation` behind
  `agent.memory.goal_versioning`. There is no score evidence yet: the effect on RHAE is measured only by Kaggle arm A
  vs arm B.
- **Git SHA:** 109bebb + uncommitted E021 changes

## What E021 is now

The agent keeps its guess about the win condition as numbered goals ("g1 v3 [active] get the blue block onto the red
cell"). A new guess writes v4 and keeps v3 with the reason it was replaced. Up to 3 goals are open (1 active, the others
rivals); refuted goals stay in view; re-proposing a refuted goal is flagged. Amendment 1 (owner):
- the goal block shows the last 5 goal changes with their reasons (`recall` scope goal shows all, including rivals
  cleared at a level-up);
- level 1 is for exploration: once level 1 is won, the goal that won it is locked for the rest of the game (act's
  `goal` is ignored with a note, the act still runs).

## Result

| Run | Games | Levels | Actions | RHAE | act arg errors | Goals open (max) |
| --- | --- | --- | --- | --- | --- | --- |
| local 4B, switch on (before amendment 1) | ls20 | 0/7 | 11 | 0.0 | 0 | 1 |
| tests, scripted model on the real ls20 (after amendment 1) | ls20 | forced level-up | 4-5 | n/a | 0 | 2 |
| Kaggle 27B, arm A / arm B | 5 dev | not run | | | | |

Evaluation: SDK scorecard and goal events of the local run (`local_smoke.md`); `tests/unit/test_prime_goals.py`
(75 checks) and the E021 part of `tests/unit/test_prime_level_review.py` (lock note, `goal_locked` event, history
with reasons in the pinned block, lock line in the prompt); `tests/run_all.py` 26/26 suites; the 7 prime suites against
the copy with the switch off. There is no held-out number and no Kaggle number.

## Did the prediction hold

- Local prediction (the new path runs, events are logged, no crash, no new act error type): **yes.**
- Local kill criterion (a crash in the new code, or `goal` argument errors): **not hit.**
- Amendment 1 prediction (history lines and lock note appear, a goal edit after level 1 never changes the goal
  memory): **yes, in tests.** The 4B does not win level 1, so the lock is shown by a scripted model, not a live one.
- Amendment 1 kill criterion (a goal edit after level 1 changes the goal memory, or the lock makes an act fail):
  **not hit.**

## Next

Owner's call: commit, update `banwait13/sarblohagent`, then run arm A (`goal_versioning: false`) and arm B
(`EXPERIMENT_OVERRIDES = {"agent": {"memory": {"goal_versioning": True}}}`) with
`kaggle/experimental/sarbloh_experimentation.ipynb` on the 5 dev games. Read the `goal` and `goal_locked` events and
the act errors, then apply the Kaggle kill criteria in `hypothesis.md`.
