# Perception experiments X1-X8 — results

Inputs: the two screenshots only (ls20 level 1, r11l level 1). Every experiment = one Python file in
experiments/; outputs (image + text) in results/XN/. Answer keys (results/answer_keys.md) were written and
verified in the engine BEFORE any experiment ran. Grader = me, with a fixed rubric (7 items x 2 points per
game, max 28) + an intuition score. Caveat: I know both games; I graded only what each output states.

## Scores on the two screenshots

| Exp | ls20 /14 | r11l /14 | Total /28 | Intuition 1-5 | Text tokens | Notes |
|---|---|---|---|---|---|---|
| E0 arc_eye (old) | 2 | 1 | 3 | 1 | ~3,500 | flagged r11l's move counter as screen chrome (-1) |
| X1 exploded inventory | 2 | 1 | 3 | 2 | ~180 | clean, but no meaning attached |
| X2 surprise map | 3 | 1 | 4 | 2 | ~270 | 90% of screen is texture; caught the lopsided plus |
| X3 visual routines | 7 | 8 | 15 | 4 | ~240 | found r11l's midpoint tether + ring fits a 5x5 thing |
| X4 look-alike threads | 4 | 2 | 6 | 3 | ~140 | key icon = lock glyph turned 90 deg, drawn 2x |
| X5 foresight | 4 | 8 | 12 | 4 | ~140 | ls20 4/4 moves + gauge right; r11l 3-click plan solved the level |
| X6 blueprint | 4 | 4 | 8 | 3 | ~100 | best map for navigation; no meaning |
| X7 role ledger | 11 | 9 | 20 | 5 | ~380 | every role right on both screenshots |
| X8 fusion (built after grading) | 13 | 12 | 25 | 5 | ~370 | measured facts + guesses with tests, one image |

## Held-out check (23 official games never used to design anything; graded by the engine)
- Screenshot -> grid: 23/23 games at 100% cell accuracy.
- X7 "YOU": 6/14 keyboard games correct (2 more had nothing move on the first press).
- X5 keyboard foresight: fully right in 1/16 games (tu93); the "moves by its own size" prior is usually wrong.
- X7 click targets: 7/29 labelled handles actually respond (good in dc22 2/2, sb26 4/4, ka59 1/1; bad in lp85 0/17).
- X7 gauge ("moves left"): 10/20.
=> The guesses in X5/X7 were effectively fitted to the two screenshots. The MEASURED facts (X3, X4)
   are true by construction and do not have this problem.

## Conclusions
1. Pixel->grid is solved. The bottleneck is meaning.
2. What moved understanding most: computed relations (X3: reach, inside, tethers/midpoint, alignment;
   X4: same-shape-under-rotation/scale). They state facts a model cannot reliably see for itself.
3. Role labels on the picture (X7) give the fastest "gist", but single-frame role guesses are wrong about
   half the time on unseen games -> they must be shown as guesses with a one-action test, never as facts.
4. Single-frame dynamics priors (X5) do not generalise; one real action would settle them.
5. Best package: X8 = blueprint map + measured facts (X3, X4) + labelled guesses with tests (X7).
