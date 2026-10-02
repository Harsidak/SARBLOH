# E115 — verdict

- **Date closed:** 2026-10-02
- **Outcome:** KEPT (X8 is the winner), with the caveats below
- **Git SHA:** not committed when run (scratch folder); logged on top of 6649ab6

## Result

Two dev screenshots, rubric out of 28 (`results/REPORT.md`, outputs in `results/XN/`).

| Exp | ls20 /14 | r11l /14 | Total /28 | Intuition 1-5 | Text tokens | Note |
| --- | --- | --- | --- | --- | --- | --- |
| E0 arc_eye (old) | 2 | 1 | 3 | 1 | ~3,500 | called r11l's move counter screen chrome (-1) |
| X1 exploded inventory | 2 | 1 | 3 | 2 | ~180 | clean, but no meaning attached |
| X2 surprise map | 3 | 1 | 4 | 2 | ~270 | 90% of the screen is texture; caught the lopsided plus |
| X3 visual routines | 7 | 8 | 15 | 4 | ~240 | found r11l's midpoint tether and the ring that fits a 5x5 thing |
| X4 look-alike threads | 4 | 2 | 6 | 3 | ~140 | key icon = lock glyph turned 90 degrees, drawn 2x |
| X5 foresight | 4 | 8 | 12 | 4 | ~140 | ls20 4/4 moves and gauge right; r11l 3-click plan solved the level |
| X6 blueprint | 4 | 4 | 8 | 3 | ~100 | best map for navigation; no meaning |
| X7 role ledger | 11 | 9 | 20 | 5 | ~380 | every role right on both screenshots |
| **X8 fusion** | **13** | **12** | **25** | **5** | ~370 | measured facts plus guesses with tests, in one image |

Held-out check on 23 official games, graded by the engine (`results/holdout/holdout_results.json`):

| Check | Result |
| --- | --- |
| Screenshot to grid | 23/23 games at 100% cell accuracy |
| X7 "YOU" | 6/14 keyboard games right (2 more had nothing move on the first press) |
| X5 keyboard foresight | fully right in 1/16 games (tu93) |
| X7 click handles | 7/29 labelled handles really respond (dc22 2/2, sb26 4/4, ka59 1/1; lp85 0/17) |
| X7 gauge ("moves left") | 10/20 |

## Did the prediction hold

Partly. Several arms beat E0 at a tenth of its text cost, and X8 scored 25/28 against E0's 3/28. But the role
and dynamics guesses (X5, X7) were in effect fitted to the two dev screenshots: on unseen games they are right
about half the time or less. The measured relations (X3, X4) are true by construction and do not have this
problem. X8 was designed after seeing the X1-X7 grades and the held-out check, and it was graded on the same two
screenshots, so its 25/28 is optimistic. It has not been run on the held-out games as a whole package.

## Did it hit the kill criterion

- X1 (3/28) and X2 (4/28) did no better than E0: killed as standalone presentations.
- X5 and X7 fail the held-out half of the criterion as fact sources. They survive only inside X8, as labelled
  guesses that each come with a one-action test.
- X3, X4 and X6 survive as parts of X8.

## What this changes

1. Turning the screenshot into a grid is solved. The bottleneck is meaning.
2. Computed relations (reach, inside, tethers and midpoints, alignment, same shape under rotation or scale) are
   what moved understanding most. They state facts a model cannot reliably see for itself.
3. Role guesses from a single frame are wrong about half the time on unseen games, so they must be shown as
   guesses with a test, never as facts.
4. One real action settles what single-frame dynamics guesses cannot.

Next: wire X8 into the agent's perception (`Sarbloh/harness/agent/vision.py` and the act observation) as a
switchable option, and measure it in the harness on games it was not designed on. Until then this is not RHAE
evidence.

## Ledger

No row in `history/ledger.jsonl`: this was a perception probe on screenshots, not a scored RHAE run.
