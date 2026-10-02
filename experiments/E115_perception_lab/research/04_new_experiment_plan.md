# Round 2 experiment plan (supersedes 02_experiment_plan.md) — awaiting approval

Rules: one Python file per experiment in experiments/XN_name/; input = one screenshot; output =
image(s) for a VLM + short text. Run on both screenshots (ls20 L1, r11l L1). Engine only where an
experiment needs ground truth (X5 foresight check, answer keys).

| ID | Name | Mechanism (research basis) | Output |
|---|---|---|---|
| X1 | Exploded inventory | pull every object out, enlarge, space apart; identical objects stacked xN (BlindTest spacing, V*) | contact-sheet image + legend |
| X2 | Surprise map | predict scene from its own regularities (tiles, fills, repeats, symmetry); show only what can't be predicted, ranked in bits (AIM, MDL, pop-out) | residual image + top-k list |
| X3 | Visual routines | flood-fill reachability, inside/outside, curve tracing, wall openings (Ullman) | one panel per routine + relation sentences |
| X4 | Correspondence threads | all pairs equal up to move/rotate/flip/scale/recolour, plus the transform needed to make near-matches equal (analogy) | threads drawn on image + "A -> B needs rot90" |
| X5 | Foresight strip | imagine the next frame per action from core-knowledge priors; verify against engine (visual foresight) | strip of imagined frames + accuracy vs engine |
| X6 | Blueprint redraw | inverse graphics to a clean schematic: walls/floor/objects/HUD panel, shapes named by a geon vocabulary (Marr, Biederman) | schematic image + named parts |
| X7 | Role ledger | per object role distribution (self, goal, obstacle, item, counter, life, switch) from design-grammar priors + cheapest test action (EMPA) | annotated image + ranked hypotheses |
| X8 | Fusion briefing | best of X1-X7 in one image + <=150 words | decided after results |

Evaluation (honest): blind readers = fresh agents that never saw the games or my work, given only one
experiment's output, answer: what do I control, what is the goal, what is the counter/HUD, what
should I do first and why. Graded vs answer keys written from engine/source before reading outputs.
Plus automatic: X5 foresight accuracy, X7 role accuracy vs engine, output token/image cost.
