# E117 — verify by acting, not by writing checks

- **Date opened:** 2026-10-03
- **Axis varied:** verification
- **Baseline:** the E116 dev run (X8 perception), the same code with the current prompts
- **Split:** dev games only (ls20 tr87 lp85 su15 tu93). This is a public-set smoke run, not evidence.
- **Status:** prompt changes written on 2026-10-03 at the owner's request, before the E116 run and before step 0.
  Not tested. Because the E116 dev run will now carry both changes, it cannot separate their effects.

## The problem

The agent spends a lot of time writing Python and thinking before it moves. Some of that is useful: perception, and a
search when the path is long. Some of it is the agent trying to prove a rule before it uses it. In an unknown game
nothing can prove a rule in advance. Not code (an if-else check breaks on a game nobody has seen), and not a second
LLM (that is only more reasoning). The only thing that can show a rule is right is the game itself, when a move
does what the rule said it would.

The games are deterministic, so one move gives an exact answer, and the host already records every move and
flags repeats. Reasoning costs no actions, but it does cost wall clock, and a timeout scores 0.

## Mechanism

Prompt wording only, in `Sarbloh-Experimentation/harness/agent/prompts.py`. The owner's snippets
(`ROLE_OBJECTIVE_COMMUNICATION`, `GAME_INTUITION`, `IPYTHON_BRIEFING`) are not touched.

1. `LEVEL_METHOD` step 4 says "When the rules you need are verified, write a search in `ipython`". This becomes:
   when you know enough to reach the goal, move toward it, and write a search in `ipython` only when the path is
   long or not obvious.
2. `LEVEL_METHOD` step 3 gets one sentence: a move toward the goal is also a test. Say in `plan` what you expect it
   to do, and if the change lines show something else, stop and think again.
3. `EXAMPLE` drops "You check the path in `ipython`". The agent sees that object 4 needs 6 more moves, marks both
   hypotheses verified with evidence [0], and sends the moves.

What stays the same:

- The meaning of "verified" in `MEMORY`: a rule that correctly predicted moves made after it was written. That is
  already verification by acting.
- The memory gate. Only verified or refuted rules become lessons at a level up (`memory/lifecycle.py`). That is
  where a wrong belief does lasting damage, and the check costs no reasoning, because it is a record of what moves
  did.
- The `RULES` bullet "do not count cells in your head, ask `ipython`". That is about seeing the board exactly, not
  about proving rules, and changing it would mix two effects.

A correction to what I said earlier: the forced reflection checkpoint (`arc_host.py:160`) is off by default
(`reflect_every_actions: None`) and the experimentation notebook does not turn it on. It is not part of the
pressure, and this experiment does not touch it.

## Step 0: measure before changing anything

Before any code, read the E116 dev run transcript:

- `ipython` calls per `act`, thinking tokens per `act`, and wall-clock seconds per action, for each game;
- by hand, label 30 `ipython` cells as one of: looking at the board, checking a rule, searching a path, other.

If checking cells are under a fifth of the `ipython` cells, and no game ran out of time, the pressure is not where
the time goes. In that case E117 is closed without a run, and `verdict.md` records the counts.

## Prediction

Committed before the run, against the E116 dev run:

- `ipython` calls per `act` fall by at least 30%, and wall-clock seconds per action fall by at least 20%;
- levels cleared stay the same or rise;
- actions per cleared level rise by no more than 15%. Some moves that used to be checked in Python will now be
  made and checked by the game. That is the intended trade, but it must stay small.

## Kill criterion

Kill if, on the dev run, any of these holds:

- the agent clears fewer levels in total than the E116 run;
- actions per cleared level rise by more than 25%;
- `ipython` calls per `act` fall by less than 10%, which means the wording did not change the behaviour.

## Measurement

- Levels cleared, actions per level and RHAE per game, against the E116 dev run.
- From `stats` and the trace: `ipython` calls per `act`, thinking tokens per `act`, seconds per action, timeouts.
- From the transcript: how often a `plan` states an expected result, and how often a surprising change line leads
  to a refuted hypothesis in the next `act`.
