# E115 — Perception lab: eight ways to read one screenshot (X1-X8)

- **Date opened:** 2026-10-02 (run offline in `D:\tmp\scarcthpad\perception_lab`, logged here the same day)
- **Axis varied:** perception
- **Baseline:** E0 = `code/baseline/arc_eye.py`, the old perception module (about 3,500 text tokens per frame)
- **Split:** two dev screenshots (ls20 level 1, r11l level 1) for design and grading; a held-out check on the 23
  other official games, which were never used to design anything. Public games, so this is a smoke test, not RHAE
  evidence (CLAUDE.md §5 eval).

The plan was written before the code (`research/04_new_experiment_plan.md`, which replaced
`research/02_experiment_plan.md`). The answer keys (`results/answer_keys.md`) were written and checked in the engine
before any experiment ran. X8 was added after X1-X7 had been graded, so it is not pre-registered.

## Mechanism

Each experiment is one Python file that takes a single screenshot and returns an image plus a short text for the
model. All of them share the same front end, `code/experiments/scene.py`: screenshot to an exact 64x64 palette
grid to a list of objects. So the only thing that changes between experiments is what is computed on top of the
objects and how it is shown.

| ID | Name | What it computes | Research basis |
| --- | --- | --- | --- |
| X1 | Exploded inventory | pulls every object out, enlarges it, spaces them apart, stacks copies as xN | VLMs are blind (2407.06581), V* (2312.14135) |
| X2 | Surprise map | ranks cells by how unpredictable their 3x3 neighbourhood is; flags broken symmetry | AIM saliency, MDL / CompressARC (2512.06104) |
| X3 | Visual routines | reachability by flood fill, what is inside what, line tracing to attached objects, midpoints, alignment | Ullman 1984 |
| X4 | Look-alike threads | pairs of objects that are equal under rotation, flip, scale or recolour, with the transform named | Visual Sketchpad (2406.09403) |
| X5 | Foresight strip | imagined next frame per action from simple priors; scored against the engine afterwards | EMPA (2107.12544) |
| X6 | Blueprint | clean schematic: floor, walls, tile lattice, named shapes, HUD moved to a side panel | Marr, Biederman |
| X7 | Role ledger | a role guess per object (you, goal, handle, gauge, lives...) with a confidence and a one-action test | EMPA, contingency awareness (1811.01483) |
| X8 | Fusion | X6 map as the base, X3 and X4 facts as MEASURED, X7 roles as GUESSES with their tests; X5 dropped | built from the X1-X7 grades |

Tools: `code/harness/shot2grid.py` (screenshot to grid), `code/experiments/holdout_check.py` (engine-graded
held-out check). Game files are not copied: they come from the official 25 games
(github.com/BDR-Pro/arc-prize-2026-arc-agi-3) and `arc-agi` (Python 3.12 or newer).

## Why it should work

Vision-language models often fail to turn what their vision encoder sees into words, mostly for small, close or
relational things (reachability, enclosure, "is this line attached to that"). If code computes those relations
and hands the model plain sentences and a clean picture, the model should understand a game from its first
frame, and so spend fewer actions finding out what it controls and what the goal is.

## Prediction

At least one presentation scores clearly above E0 on the 28-point rubric (7 answer-key items x 2 points per game),
at a lower text cost. Relation facts (X3, X4) should hold on the held-out games, because they are computed rather
than guessed.

## Kill criterion

Drop a presentation if it scores no better than E0 (3/28) on the two screenshots, or if its claims are wrong on
about half or more of the held-out games.

## Measurement

- Rubric score out of 28 against `results/answer_keys.md`, plus an intuition score from 1 to 5 and the text cost.
- Held-out, graded by the engine: grid accuracy, X7 "YOU" hit rate, X7 click-handle precision, X7 gauge hit rate,
  X5 keyboard foresight accuracy.
- Caveat: the grader was the same agent that built the experiments and knows both games. The blind-reader grading
  in the plan was not done.
