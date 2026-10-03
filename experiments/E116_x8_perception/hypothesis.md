# E116 — X8 perception in the agent (one `observe()`)

- **Date opened:** 2026-10-03
- **Axis varied:** perception
- **Baseline:** the E008 perception of `Sarbloh-Experimentation/` before this change (`vision.py` + `intuition.py`,
  `scene.*` in the REPL), on the same 5 public dev games
- **Split:** dev games only (ls20 tr87 lp85 su15 tu93). This is a public-set smoke run, not evidence.

## Mechanism

`harness/agent/vision.py` and `harness/agent/intuition.py` are replaced by one file, `harness/agent/perception.py`:

1. The letter board (ASCII, rows and columns numbered), the object list (segmentation with stable ids) and the change
   lines are kept from `intuition.py`.
2. E115's winner, X8 fusion (`experiments/E115_perception_lab/code/experiments/X8_fusion/`), is ported to work on
   the grid. It gives a **briefing**, made of MEASURED facts (reach, enclosure, tethers and midpoints, alignment,
   look-alikes, copies, broken symmetry) and GUESSES (role guesses, each with a one-action test). It also gives a
   **picture**: the blueprint map with role tags, threads and a HUD side panel, without the briefing text.
3. The briefing names objects by the ids in the object list, so there is a single set of ids in the text, the
   picture and the change lines.
4. In `ipython`, `scene.*` (about 10 functions) is replaced by one function, `observe()`. It returns the state for
   the agent's code (`obs.board`, `obs.objects`, `obs.change`, `obs.history`, `obs.ascii(...)`). The host then
   sends the full state (briefing + objects + board as text, and the picture) as the next message, at most once
   per state.
5. Bug fix: `scene.change` was always None (`Scene(change=None)` in `agent.py`). The newest change line now
   reaches both the text and `obs.change`.

After every `act` the host still sends the short view: the change lines, then the changed region, now with the X8
picture. The full view is sent at a level start, after a compaction, and on `observe()`.

## Why it should work

E115 measured that computed relations (reach, inside, tether midpoints, look-alikes) are facts a model cannot
reliably see for itself, and X8 scored 25/28 against 3/28 for the old eye on the two dev screenshots. If the agent
reads those facts at each level start, it should need fewer test actions to find the controls and the goal.
Because the guesses are labelled as guesses and each comes with a test, a wrong guess should cost one action
rather than a whole wrong plan.

## Prediction

Committed before the run: on the 5 dev games, fewer actions to the first level-up on at least 3 of 5 games, and
no game that loses a level it cleared before. The RHAE delta is not predicted: 5 public games cannot show it.

## Kill criterion

Kill if, on the dev run, any of these holds:

- the agent clears fewer levels in total than the baseline run;
- perception takes more than 2 s per act (median), measured from the `perception` timing in `stats`;
- the agent stops using `act` because it keeps calling `observe()`, i.e. more than 3 `observe()` calls per act on
  average.

## Measurement

- Levels cleared and actions per level, against the last dev run of the baseline perception.
- `stats`: `observe_calls`, `observe_pushes`, `perception_errors`, `perception_s_max`, `perception_s_total`.
- The transcript: whether the model uses ids from the briefing, and whether it tests its guesses.

## Local checks before the run (2026-10-03, no model)

- `tests/unit/test_e116_perception.py` (50 checks) and the E116 part of `tests/unit/test_prime_tools.py` (real ls20,
  real kernel, scripted model: `observe()` at the start sends nothing new; called twice after an act it sends the
  full state once; `obs.change` holds the newest change line).
- 225 real frames (25 offline games, start + 8 random actions each): 0 errors; briefing median 42 ms, max 0.38 s
  (lp85); picture median 40 ms; full text median 7.6k chars, max 11k.
- Found and fixed while checking: X8's tile-grid test flipped during play (ls20 lost its 5x5 grid after one move,
  because the gauge in the status box adds off-grid edges; ka59 lost its 3x3 grid whenever the self touched another
  thing). Status boxes are now left out of the test and the level's previous grid is kept while it still fits: 0
  flips on the 225 frames.
- Found 2026-10-03 on tr87: "X is directly above Y" facts filled 27 of the 30 briefing lines shown (68 in all;
  bp35 had 792). A cap of 12 per frame was tried and removed the same day: it deleted facts from
  `observe().briefing` too. Now every alignment fact is kept, those touching the guessed self and goal come first,
  and `Briefing.lines()` shares the shown lines between the fact kinds, so one kind cannot fill them. The rest are
  counted as "(+N more facts: `observe().briefing` in ipython)".
