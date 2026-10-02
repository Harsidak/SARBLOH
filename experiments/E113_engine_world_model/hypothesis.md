# E113: Engine-grounded world model (the agent writes the game in `arcengine`)

- **Date opened:** 2026-10-02 (logged, NOT started; parked by the owner until ~30 RHAE)
- **Axis varied:** hypotheses / world model representation
- **Baseline:** the best harness at the time the experiment starts (at the time of logging: E006 dedicated tools,
  plus whatever world-model path is current), same backbone, same games
- **Split:** Phase A on held-out/unseen games only; Phase B tune for development, held-out for the report

## Origin

Owner observation, 2026-10-02, unlogged and not reproducible as it stands: Claude was given the ARC-AGI-3 game
authoring context (the `arcengine` API, which lets anyone build their own ARC-AGI-3 games) and a single screenshot of
ls20, and was asked to "create your own game". It wrote a game that reproduced ls20's dynamics.

**Contamination caveat.** ls20 is a public game. Its source or play-throughs may be in a frontier model's training
data, so that observation cannot separate understanding from recall. This experiment must be judged on games the
model cannot have seen (see Phase A).

## Mechanism

Give the agent's `ipython` REPL the `arcengine` package (`ARCBaseGame`, `Level`, `Sprite`, `Camera`, `GameAction`),
plus a one-page API card and a worked example (`Sarbloh-Arc/Scratchpad.py` already is one). The agent's world model
then becomes a **candidate game definition** in the engine's own vocabulary (sprites, levels, collisions, win
condition) instead of a free-form `step()` in plain Python (E09 / Schema / EWM).

The harness gets two new host-side functions:

1. `clone_check(game_cls)`: instantiate the candidate game at the recorded level start, replay the recorded action
   sequence, and compare every produced frame with the recorded frame (exact match, the existing `parkh` rule).
   Report the first mismatching transition and the differing cells. Free: no real action is spent.
2. `clone_plan(game_cls, goal)`: BFS over the candidate game's states to the level-win condition. The plan goes to
   `act` in batches. Halt-on-misprediction stays on.

## Why it should work

1. **Smaller, correct hypothesis space.** Every ARC-AGI-3 game is a program in this engine. A candidate written in
   it can only express what the engine can express, so the search runs over the space the authors actually used,
   not over all of Python.
2. **Certification and planning are free.** A clone that replays every recorded transition exactly is a perfect
   simulator for the states seen so far. Search inside it costs zero actions, which raises RHAE directly.
3. **Transfer.** Later levels are usually new `Level` objects built from the same sprites and rules, so a certified
   clone of level N gives a strong prior for level N+1.

## Prediction (committed before the run)

- **Phase A (reconstruction probe, no game actions):** on unseen games, given the first frame plus the first 10
  recorded transitions, a frontier model reaches >= 80% exact transition replay, and the open backbone (Qwen/Gemma
  class) reaches >= 50%. UNCONFIRMED: no measurement exists; these numbers are calculated guesses.
- **Phase B (in harness):** vs the baseline, a lower action count on levels >= 2 (where transfer should show), and
  certification earlier (`certified_at_action`) on levels where a clone passes.

> Expected held-out RHAE delta: not committed until Phase A passes. Phase A decides whether Phase B runs.

## Kill criterion (binding)

> Kill if the open backbone reaches < 30% exact replay on unseen games in Phase A after two prompt iterations
> (API card / example changes only). The idea is then frontier-only and out of scope for the Kaggle submission.

> Phase B: kill if levels >= 2 show no action reduction vs baseline on held-out games.

## Measurement

**Phase A: offline, no game actions, minutes of GPU time.**

- Inputs per item: the first frame plus 10 recorded transitions from our runs (frames are in `ArcHost` /
  `arc.transitions()`), the API card and the `Scratchpad.py` example.
- Output: a game class.
- Score: the share of the 10 transitions reproduced exactly; the first-mismatch index; whether the code runs at
  all.
- Games:
  - **Unseen set (the only one that counts):** games the owner authors in `arcengine`, plus mutated public games
    (recoloured, rearranged levels, changed rule constants).
  - **Public set (reported separately, contamination-flagged):** for comparison only.
- Models: one frontier reference (for the ceiling only, never trained on) and the submission backbone.

**Phase B: games.**

- Metrics: held-out RHAE; actions per level on levels >= 2; `certified_at_action`; the share of real actions spent
  after a passing clone exists; clone-check pass rate per level.
- Seeds: 3, reported with variance.

## Practical constraints

- **Offline Kaggle:** `arcengine` must be installed in the competition kernel from a bundled wheel (same route as
  the other wheels). Check its licence before including it in the submission.
- **Context cost:** the API card plus the example must fit within about 2k tokens. Measure it.
- **Sandbox:** the candidate game runs in the kernel. It must not be able to touch the real `ArcHost` game object.
  `clone_check` builds the clone from the class and never from the live game.

## Dependencies and timing

Parked by owner decision: run when the main harness reaches ~30 RHAE and there is spare compute. Phase A can run
earlier as a standalone probe because it spends no game actions and needs no working agent loop.

Related: E09 (`step()` in the REPL), E51/E54 (backtest, halt on misprediction), E55 (planning in the model), the
"Raven" harness self-evolution idea (also parked until ~30 RHAE).
