# E022 — verdict

- **Date:** 2026-10-01 (local part)
- **Outcome:** **PLUMBING PASS, effect not measured.** The review runs, writes into the working memory, the lessons
  graph and `levels.jsonl`, and never stops the run. Whether it saves actions on later levels is a Kaggle question.
- **Git SHA:** 109bebb + uncommitted E022 changes

## Result

| Run | Games | Levels | Reviews | Review errors | RHAE |
| --- | --- | --- | --- | --- | --- |
| tests, scripted model on the real ls20, good reply | ls20 | forced level-up | 1 | 0 | n/a |
| tests, scripted model, bad reply | ls20 | forced level-up | 0 | 1 (logged, run went on) | n/a |
| local 4B | | not run (server stopped by the owner) | | | |
| Kaggle 27B, arm A / arm B | 5 dev | not run | | | |

Evaluation: `tests/unit/test_prime_level_review.py` (40 checks), `tests/run_all.py` (26/26 suites), the 7 prime suites
against the copy with the switch off. Details in `local_tests.md`. There is no live-model, held-out or Kaggle number.

## Did the prediction hold

- Local prediction (after a level-up the review runs once; its notes reach the pinned block, its rules the lessons
  graph and its row `levels.jsonl`; a bad reply does not stop the run): **yes, in tests.**
- Local kill criterion (crash, nothing written on a valid reply, behaviour changed with the switch off): **not hit.**

## Next

Owner's call: commit, update `banwait13/sarblohagent`, then arm A (`level_review: false`) and arm B
(`EXPERIMENT_OVERRIDES = {"agent": {"memory": {"level_review": True}}}`), or E021 + E022 together
(`{"agent": {"memory": {"goal_versioning": True, "level_review": True}}}`), on the 5 dev games. Read the
`level_review` events (duration, errors) and actions per level for levels 2+, then apply the kill criteria.
