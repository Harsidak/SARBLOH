# Phase 2 plan — perception experiments (awaiting approval)

## Shared harness (built first, used by every experiment)

- Interface: each perception module = `perceive(history) -> text`, where history is the list of
  (frame, action) pairs seen so far. Same input for all modules; only the output differs.
- Data: offline engine. **Dev set** = 30 synthetic + ~30 community games (tuning allowed).
  **Test set** = 25 official games (reported once, no tuning). Snapshots at several levels via
  `set_level`, each with a short history from an exploration policy.
- Ground truth (from the engine, never from me): exact grid, sprite instances, clickable sprites,
  controllable sprite (deepcopy + try each action), UI/HUD cells, true next frame for every action.
  Goal/rule answer keys for ~8 test games written from game source *before* any module is scored.
- Metrics:
  - M1 object recovery — instance F1 vs engine sprites (automatic)
  - M2 self identification — does the output single out the controllable/clickable thing (automatic + reader)
  - M3 effect prediction — a blind reader predicts what an action will change; graded vs engine
  - M4 blind-reader QA — a fresh agent that sees ONLY the text answers: what do I control, what is
    the goal, what is the counter/HUD, what are the rules; graded against the answer key
  - M5 cost — tokens per frame
  - M6 finale — closed-loop play on test games with the top 2-3 modules vs baseline:
    levels solved, actions used vs human baseline (RHAE)

## Experiments

| ID | Name | Hypothesis | Output |
|---|---|---|---|
| E0 | arc_eye (current) | baseline | grid + components + diffs |
| E1 | Raw grid | floor: what most LLM agents get | 64 rows of hex digits |
| E2 | Object sentences | objects beat grids (Xu 2023) | one sentence per object, no grid |
| E3 | Zoom-out map | 64x64 overwhelms readers; the game has a natural unit | coarse map in game units + legend |
| E4 | Inverse graphics (compression) | shortest exact redraw recovers designer's sprites | sprite library + placements, verified pixel-exact |
| E5 | Layout parse | screen = play area + status panels | panels first, then contents |
| E6 | Analogy finder | ARC goals hide in matches | all matches up to move/rotate/flip/scale/recolour |
| E7 | Common-fate grouping | objects = what moves together; clocks change regardless | multi-frame objects, clock/HUD flags |
| E8 | Poke-test (contingency) | agentness is interventional | "you are X", action->effect table, inert things |
| E9 | Event diary | full grids waste context | changes only + running memory |
| E10 | Hypothesis ledger | perception should be a falsifiable theory | ranked role hypotheses + evidence + predictions checked next frame |
| E11 | Annotated picture (optional) | a labelled image may beat text | code-drawn labelled image |
| E12 | Synthesis | best parts combine | final module, test-set only |

Order: harness -> E0, E1, E2 -> E3, E4, E5, E6 -> E7, E8, E9 -> E10 -> (E11) -> E12 -> M6 finale -> report.
