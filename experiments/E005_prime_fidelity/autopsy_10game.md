# E005: autopsy of the 10-game Kaggle run (2026-09-28)

Run: `runs/kaggle_sarbloh-prime_20260928_144039/prime_run` (kernel `banwait13/sarbloh-prime`), config hash
`dd1f4e5f5001`, Gemma-4-31B-IT-NVFP4, 10 public games (none held-out), 500 actions per game, ~50 min wall clock.
Git SHA of the dataset used is UNCONFIRMED (HEAD at download: 8acdd3d).

## Result
| game | levels | actions | resets | GAME_OVER | turns | biggest single cell | end |
|---|---|---|---|---|---|---|---|
| lp85 | **1/8** | 500 (L0: 231 vs human 17) | 0 | 0 | 26 | 231 actions | action budget |
| bp35 | 0/9 | 500 | 31 | 5 | 62 | 65 | action budget |
| sc25 | 0/6 | 500 | 15 | 4 | 48 | 41 | action budget |
| sp80 | 0/6 | 500 | 62 | 2 | 83 | 46 | action budget |
| su15 | 0/9 | 500 | 71 | 7 | 60 | 40 | action budget |
| tn36 | 0/7 | 500 | 16 | 8 | 50 | 62 | action budget |
| tr87 | 0/6 | 500 | 78 | 0 | 41 | **320** | action budget |
| tu93 | 0/9 | 500 | 9 | 16 | 144 | 52 | action budget |
| ls20 | 0/7 | 295 | 41 | 0 | 54 | 22 | wall clock |
| wa30 | 0/9 | 154 | 27 | 0 | 36 | 14 | wall clock |

Mean score 0.0015; 1/76 levels; 4,449 actions; 16.3 M prompt tokens, 0.34 M output; 46 cell errors out of 597;
0 subagents; 15 compactions; ~13 harness refinements in total.

## Findings
1. **The plumbing works.** Tool calls were native in 597 of 597 cases, there were no crashes, and compaction and
   auto-refine fired. Every action is traced to a turn and a cell. The harness is not what failed.
2. **Brute force burns the budget.** The agent writes loops that spend actions blindly: tr87 spent 320 actions in
   one cell (every sequence of 1/2 up to length 5, with a reset in between), lp85 spent 197 in one cell (clicked
   one pixel per row per colour, because of a `break` bug), and bp35 clicked (0,0) 64 times. Once the agent can
   write a loop, it stops treating actions as costly.
3. **Understanding fails before execution does.** In ls20 the agent saw the bottom bar change (b→3) on its first
   move and dismissed it; that bar is the step counter. It then spent 40 resets guessing target cells ("move the
   block onto the 0 at (31,7)?"). In tu93 and sp80 it guessed goals one by one ("what if maroon must overlap
   blue?"). None of the 10 games reached a correct goal hypothesis before the budget ran out. This supports the
   owner's thesis (E105): the missing part is understanding the goal and dynamics.
4. **Resets are used as experiments.** There were 350 resets in 4,449 actions (8%); tr87 had 78 and su15 had 71.
   Each costs an action.
5. **The one level-up was luck plus search.** In lp85 only colour 8 changed anything, and clicking it twice
   cleared level 0: 231 actions against a human baseline of 17, so RHAE ≈ (17/231)² ≈ 0.005 for that level.
6. **Row and column confusion.** The agent mixed up `(row, col)` and `(x, y)`; ls20's reasoning names targets as
   (row, col) while the API takes (x, y). UNCONFIRMED as a cause of failure, but it recurs.
7. **Memory learned trivia.** The only ls20 refinement was "1=Up, 2=Down, 3=Left, 4=Right". No goal or mechanic
   was ever written to memory.
8. **Slow games hit the clock.** ls20 and wa30 had long reasoning (~1.8k characters per turn) and ~27k-token
   prompts, and ran out of wall clock at 295 and 154 actions.

## So what
- Execution (tool calling, stepping, reading the grid) works. Understanding (goal, hazards, counters) does not.
  E105 (oracle briefing on ls20) is the direct test: given the understanding, can it execute?
- Cheap guard to test next: an action-spend cap per cell. It already exists as `max_actions_per_cell` (E004), but it
  was off (`None`) in this run. Turning it on would have stopped findings 2 and 4 at the source.
