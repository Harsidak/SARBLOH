# E022 local checks (2026-10-01)

**Plumbing only, no model.** No live run: the owner stopped the llama.cpp server, and the 4B never wins level 1, so it
would never reach a review. Not a scored run, so no ledger row.

- Code: `Sarbloh-Experimentation/prime/` on top of 109bebb (uncommitted): `agent/level_review.py` (new),
  `memory/lifecycle.py` (`level_record`, `add_level_review`, `_promote`, recall), `memory/working.py` (`reviews`, block
  "levels"), `agent/context.py` (block order), `agent/agent.py` (`_maybe_review`, before the curator), `agent/prompts.py`
  (one sentence), `config.py` (`level_review` False, `level_review_max_tokens` 2048, `level_review_steps_tokens` 3000,
  cap `levels` 250), `run.py` (`--level-review`).

## Tests
- `tests/unit/test_prime_level_review.py` (new): 40 checks, 0 fails.
  - Unit: default off; block order; the level record (steps, plans, memory at the end); the review input holds the
    whole level; step clipping keeps the first and last steps; a review goes into the working memory (block "levels",
    last 2 reviews, under cap), the lessons graph (rule, refuted and goal nodes, "obj N" linked to the shape) and
    `levels.jsonl` (recall scope findings and all); proposed rules are dropped; an empty review is refused; the goal is
    not touched; save/load; a new game clears the reviews; switch off: no block, prompt unchanged.
  - The call: thinking off, no tools; a reply that never parses raises after 2 tries; on overflow the input is halved.
  - End to end, a scripted model on the real ls20 with E021 + E022 on and a forced level-up: one review of level 1,
    run before the curator and before the next agent turn; its input has the level's steps and plans; the next pinned
    memory shows "Level reviews" with the win condition; the system prompt has the E022 sentence. With a bad reply:
    a `level_review` error event, the session goes on.
- `tests/run_all.py`: 26/26 suites green, 1647 checks, 2 known warnings.
- The 7 prime suites against the copy with both switches off (`PRIME_DIR=Sarbloh-Experimentation`): 0 fails.
