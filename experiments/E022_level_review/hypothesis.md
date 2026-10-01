# E022 — Score-change memory: a review of the whole level at every level-up

- **Date opened:** 2026-10-01
- **Axis varied:** substrate / consolidation (backlog E22, "score-delta memory", as restated by the owner)
- **Baseline:** E008_perception_memory as it is (`agent.memory.level_review: False`), same code, same game. E022 is
  independent of E021: both switches can be on.
- **Split:** none yet. Local plumbing on the public game ls20 only, as for E021. Not a result.

## Mechanism

Owner's statement: "Every time the level counter changes, it should update the level-specific findings in the memory,
reviewing what happened in the whole level and updating the different memories accordingly."

In E008 a level-up is mechanical: verified hypotheses and findings become lesson nodes as the agent wrote them, then
plan, hypotheses and findings are erased (`GameMemory.on_level_up`). The curator that runs after it sees only the last
25 steps. Nothing looks at the whole level: which actions won it, which were wasted, what was learned.

E022 adds, behind `agent.memory.level_review` in `Sarbloh-Experimentation/prime/`:
- `prime/agent/level_review.py`: one hidden JSON call (thinking off, like the curator) after every level-up. Input:
  every step of the level just won (the timeline rows of that level, resets and GAME_OVERs included, clipped to
  `level_review_steps_tokens` by keeping the first and the last steps), every plan the agent wrote in it, and what its
  memory held when the level ended (goal(s), hypotheses, findings, from `levels.jsonl`), plus earlier reviews.
  Output: `summary` (how the level was won), `win_condition`, `rules` (verified or refuted), `mistakes` (wasted
  actions and why) and `carry_over` (at most 3 notes for the next level).
- The review updates the memories:
  - **working memory**: a new pinned block "Level reviews" shows the last 2 reviews (win condition, carry-over notes,
    mistakes) under cap `levels` (250 tokens); it is kept for the game and erased at a new game;
  - **lessons graph**: the rules become `rule` / `refuted` nodes and the win condition a `goal` node
    (source `level_review`), linked to shapes named "obj N";
  - **levels log**: a `review` row in `levels.jsonl`; `recall` scopes `findings` and `all` return it.
  - The goal memory is not touched (E021's lock applies when both switches are on).
- A failed review (unparseable reply, overflow twice) is logged and skipped; it never stops the run.

## Why it should work

RHAE squares the action ratio per level, and later levels compose the mechanics of earlier ones. A review of the
whole level gives the next level the win condition and the cost of its mistakes in a few lines, so the agent spends
fewer actions rediscovering them. The per-step lessons from E008 miss facts the agent never wrote down.

## Prediction

Committed before the run.
- **Local (tests + scripted model on the real ls20):** after a level-up, the review runs once, its notes appear in
  the pinned block, its rules in the lessons graph and its row in `levels.jsonl`; a bad review reply does not stop
  the run. The 4B does not win level 1 locally, so the live path after a level-up is covered by the scripted run.
- **Kaggle (not run; owner's call):** vs arm A on the 5 dev games, fewer actions on levels 2+ per level won.

## Kill criterion

Committed before the run. Binding.

> Local: kill as a bug if the review crashes the session, writes nothing on a valid reply, or changes behaviour with
> the switch off. Kaggle (when it runs): kill if more than 1 in 4 reviews fail to parse, or if a review adds more
> than 60 s per level-up on the 27B. If arm B is not better than arm A on actions per level for levels 2+, the verdict
> is INCONCLUSIVE.

## Measurement

Evaluation systems used:
- **Tests:** `tests/unit/test_prime_level_review.py` (new): unit checks and a scripted end-to-end run on the real
  ls20 that wins level 1 with a fake model. The existing prime suites against the copy (`PRIME_DIR=Sarbloh-Experimentation`).
- **Local:** `Sarbloh-Experimentation/prime/run.py` on llama.cpp only if the owner asks (the owner stopped the server
  on 2026-10-01).
- **Kaggle (later):** `kaggle_push.py experimentation`, `prime_run/results.json` + `LEDGER_ROW`; per-game RHAE via
  `eval/official_score.py`; `level_review` events in `events.jsonl` (duration, errors, counts written).
