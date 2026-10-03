# E118 — skills written at every level-up, reasoning by level, optional plan

- **Date opened:** 2026-10-03
- **Axis varied:** consolidation (skills) and action policy (reasoning per level)
- **Baseline:** the next dev run of `Sarbloh-Experimentation/` before this change (perception and act-to-verify
  prompts included). Note: that run has not happened yet, so the next dev run carries the perception change, the
  act-to-verify prompts and this change together, and cannot separate them.
- **Split:** dev games only (ls20 tr87 lp85 su15 tu93). This is a public-set smoke run, not evidence.
- **Status:** written 2026-10-03 at the owner's request ("Upgrade #2"), code in the same change. Unit suites only.

## Mechanism

All in `Sarbloh-Experimentation/harness/`. No experiment ids in the code (owner rule).

1. **One toolset.** The `toolset` setting is gone. The agent always has `ipython` (read-only `observe()`), `act` and
   `recall`. The upstream one-tool REPL arm and everything only it used are removed: the `arc` REPL module, the
   Continual Harness digest and auto-refine (`refine.py`), the forced reflection checkpoint, the per-cell action cap
   and the REPL's action path in `ArcHost`. Experiment ids are removed from comments, names and prompts.
2. **Skills at every level-up, before memory is cleared.** When an `act` clears a level, the hidden curator runs
   *before* `GameMemory.on_level_up` erases the level's plan, hypotheses and findings, so it reads them. At a level-up
   its prompt asks for 2 to 5 specific skills (one mechanic, control, goal pattern or procedure each, taken from what
   the level taught), and it states: "Do not think or reason much." (the call also runs with thinking off). Other
   curator passes keep their limit of 2 skills.
3. **`plan` is optional** in the `act` schema and in `GameMemory.write_act` (a missing plan keeps the old one). The
   prompts ask the agent to write a plan often. The `ipython` prompts say that variables and functions persist for the
   whole game, so the agent should write code once, change it when the game shows something new, and then act.
4. **Reasoning by level.** The level start messages (game start, level-up, after a compaction) say, for levels 1 and 2,
   to reason extensively; for level 3 on, to reason, but to start from the skills and the functions already written in
   `ipython`, changing them where the level differs (the word "extensive" is not used).

## Why it should work

Later levels compose the mechanics of earlier ones. Today a level's verified rules become one-line lessons, and the
curator writes at most 2 skills, from memory that the level-up has already erased. Skills written from the full level
memory, plus a push to reuse them and the agent's own code from level 3 on, should cut the actions spent re-learning
known mechanics on later levels. Heavy reasoning on levels 1 and 2 costs wall clock but no actions.

## Prediction

Committed before the run, against the baseline dev run:

- actions per cleared level on levels 3 and later fall by at least 20%;
- levels cleared stay the same or rise;
- at least 2 curated skills per level-up on average, and at least one `recall` of scope "skills" or one reused
  `ipython` function per game that reaches level 3.

## Kill criterion

Kill if, on the dev run, any of these holds:

- fewer levels cleared in total than the baseline;
- a timeout on any game that did not time out in the baseline (the extra reasoning on levels 1 and 2 cost the clock);
- curator errors on more than a quarter of level-ups.

## Measurement

- Levels cleared, actions per level, RHAE per game, timeouts.
- `stats`: `curator_runs`, `curator_errors`, skills written per level-up (the `curator` events), `recalls`.
- The transcript: whether the agent names a skill or calls a function it wrote on an earlier level.
