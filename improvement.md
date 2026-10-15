# Benchmark Improvement Tracker

This document tracks iterative performance, game-by-game results, and comparative improvements across experimental runs.

**Target**: Achieve **3x to 5x improvement** in score and level completion velocity over the Experiment 1 baseline before concluding iterations.

---

## 1. Run Progress & Multiplier Summary

| Exp # | Date / Timestamp | Run Directory | Wall Time | Levels Won | Mean Score | Total Actions | Level Velocity | Score Multiplier vs Exp 1 |
|---|---|---|---|---|---|---|---|---|
| **1 (Baseline)** | 2026-10-15 15:11 | `runs/kaggle_sarbloh-experimentation_20261015_151105` | 28.0 min (1,680.9s) | **19 / 108** | **5.7449** | 1,682 | 40.72 lev/hr | **1.0x (Baseline)** |
| **2** | 2026-10-16 00:32 | `runs/kaggle_sarbloh-experimentation_20261016_003252` | 31.5 min (1,892.2s) | **2 / 108** | **0.4233** | 158 | 3.81 lev/hr | **0.074x (-92.6%)** |
| **3** | *(Pending)* | | | | | | | |

---

## 2. Experiment 1 — Baseline Benchmark

- **Timestamp**: 2026-10-15 15:11:05
- **Experiment Config**: `experimentation_45min_15games` (Wall Time: 28.0 minutes / 1,680.9s)
- **Model / Infrastructure**: `intel-qwen3.8-flash-next-w4a16-autoround` via SGLang + EAGLE speculative decoding
- **Summary**: 15 games evaluated, 19 levels completed out of 108, Mean Score **5.7449**, Total Actions: **1,682**, Total Turns: **638**.

### Per-Game Breakdown Table

| Game ID | Levels Won | Score | Actions | Turns | State |
|---|---|---|---|---|---|
| `bp35-0a0ad940` | 0 / 9 | 0.00 | 86 | 32 | cancelled |
| `cn04-2fe56bfb` | 0 / 6 | 0.00 | 49 | 19 | cancelled |
| `ka59-38d34dbb` | 0 / 7 | 0.00 | 52 | 20 | cancelled |
| `lf52-271a04aa` | 0 / 10 | 0.00 | 3 | 7 | cancelled |
| `g50t-5849a774` | 0 / 7 | 0.00 | 33 | 24 | cancelled |
| `ls20-9607627b` | 1 / 7 | 1.69 | 152 | 52 | gave_up |
| `m0r0-492f87ba` | 1 / 6 | 4.76 | 272 | 58 | cancelled |
| `r11l-495a7899` | 1 / 6 | 4.76 | 37 | 39 | cancelled |
| `cd82-fb555c5d` | 2 / 6 | 5.71 | 105 | 62 | cancelled |
| `re86-8af5384d` | 2 / 8 | 8.03 | 129 | 49 | cancelled |
| `ar25-0c556536` | 2 / 8 | 8.14 | 113 | 37 | cancelled |
| `s5i5-18d95033` | 2 / 8 | 8.33 | 165 | 50 | cancelled |
| `lp85-305b61c3` | 3 / 8 | 13.66 | 109 | 53 | cancelled |
| `dc22-fdcac232` | 2 / 6 | 14.29 | 199 | 78 | cancelled |
| `ft09-0d8bbf25` | 3 / 6 | 16.80 | 178 | 58 | cancelled |

### Verified Bottlenecks in Experiment 1
1. **Scheduler Slot Starvation (21–23 min wait)**:
   - Games with 9–10 levels (`lf52`, `bp35`) had low $\text{gain}$ values ($\text{gain} = \frac{L+1}{n(n+1)/2} = 0.018 - 0.022$) and were starved without aging.
   - `lf52` waited **1,420 seconds (23.67 min)**; `bp35` waited **1,316 seconds (21.94 min)**.
2. **Strict Hypothesis Validation Exception**:
   - `lf52` passed `status: "testing"`, triggering a fatal `ValueError` in `WorkingMemory` that blocked the entire move batch. 0 moves executed in that turn.
3. **Action Oscillation & Self-Destructive Testing**:
   - In `bp35`, the agent executed `A3, A4` 25 times in one turn testing survive-to-0, triggering `GAME_OVER` and resetting the board.
4. **Goal Blindness on Transitions**:
   - `m0r0` took 247 actions on Level 1 with 0 hypotheses recorded. `ka59` and `g50t` stated no goal across all turns.

---

## 3. Experiment 2 — Comparative Benchmark

- **Timestamp**: 2026-10-16 00:32:52
- **Run Directory**: `runs/kaggle_sarbloh-experimentation_20261016_003252`
- **Experiment Config**: `experimentation_45min_15games` (Wall Time: 31.5 minutes / 1,892.2s)
- **Model / Infrastructure**: `intel-qwen3.8-flash-next-w4a16-autoround` via SGLang + EAGLE speculative decoding
- **Summary**: 15 games evaluated, **2 levels** completed out of 108, Mean Score **0.4233**, Total Actions: **158**, Total Turns: **2,098**.

### Per-Game Breakdown Table

| Game ID | Levels Won | Score | Actions | Turns | State |
|---|---|---|---|---|---|
| `bp35-0a0ad940` | 0 / 9 | 0.00 | 4 | 77 | cancelled |
| `cn04-2fe56bfb` | 0 / 6 | 0.00 | 8 | 283 | cancelled |
| `ka59-38d34dbb` | 0 / 7 | 0.00 | 16 | 60 | cancelled |
| `lf52-271a04aa` | 0 / 10 | 0.00 | 8 | 16 | cancelled |
| `g50t-5849a774` | 0 / 7 | 0.00 | 12 | 158 | cancelled |
| `ls20-9607627b` | 1 / 7 | 3.57 | 21 | 48 | cancelled |
| `m0r0-492f87ba` | 0 / 6 | 0.00 | 5 | 186 | cancelled |
| `r11l-495a7899` | 0 / 6 | 0.00 | 5 | 236 | cancelled |
| `cd82-fb555c5d` | 0 / 6 | 0.00 | 23 | 149 | cancelled |
| `re86-8af5384d` | 0 / 8 | 0.00 | 1 | 111 | cancelled |
| `ar25-0c556536` | 1 / 8 | 2.78 | 30 | 103 | cancelled |
| `s5i5-18d95033` | 0 / 8 | 0.00 | 0 | 106 | cancelled |
| `lp85-305b61c3` | 0 / 8 | 0.00 | 9 | 81 | cancelled |
| `dc22-fdcac232` | 0 / 6 | 0.00 | 15 | 175 | cancelled |
| `ft09-0d8bbf25` | 0 / 6 | 0.00 | 1 | 309 | cancelled |

### Head-to-Head Comparison: Exp 1 vs Exp 2

| Metric | Exp 1 (Baseline) | Exp 2 (Recent Run) | Delta / Change | Impact |
|---|---|---|---|---|
| **Levels Cleared** | **19 / 108** | **2 / 108** | **-17 (-89.5%)** | Severe drop |
| **Mean Score** | **5.7449** | **0.4233** | **-5.3216 (-92.6%)** | Severe collapse |
| **Total Actions** | 1,682 | 158 | -1,524 (-90.6%) | Agent stopped acting |
| **Total Turns** | 638 | 2,098 | +1,460 (+228.8%) | Turns exploded |
| **`act()` Calls** | 781 | 75 | -706 (-90.4%) | Model barely called `act()` |
| **Idle Turns Share** | 43.7% | 96.7% | +53.0% | 96.7% of turns did 0 actions |
| **Prompt Tokens** | 25,111,081 | 104,363,192 | +79,252,111 (4.15x) | Token waste exploded |
| **Completion Tokens** | 1,097,651 | 657,874 | -439,777 | Output was brief confusion |
| **Wait for Slot %** | 31.7% | 41.7% | +10.0% | High scheduler contention |

### Root Cause Analysis of Experiment 2 Regression
1. **The `hide_code: True` Blindness Trap**:
   - `hide_code: True` stripped all assistant code from the context, replacing each cell with a stub comment `# [All the code you write lives in your ipython kernel...]`.
   - Simultaneously, **99.2% of all execution cells (2,072 out of 2,088)** produced `[no output]` (because variables/functions were assigned or inspection had no explicit `print()`).
   - The model saw alternating pairs of `[code stub comment]` and `[no output]`, causing acute context amnesia. It could not recall what it had defined, whether commands ran, or why output was missing.
2. **Infinite Diagnostic Loops**:
   - In games like `ft09` (309 turns, 1 action), `cn04` (283 turns, 8 actions), and `s5i5` (106 turns, 0 actions), the model spent hundreds of consecutive turns running probe prints (`print('hi')`, `print('hi2')`, `dir(obs)`), believing the kernel had died or hung.
3. **Turn Inflation and Cache Bloat**:
   - Each useless turn appended another user/assistant exchange to the context, swelling history to 80k+ tokens without making game progress, resulting in over 104 million prompt tokens.

---

## 4. Key Performance Indicators (KPIs) for Success

To satisfy the **3x to 5x improvement** criterion:
- **Mean Score Target**: $\ge 17.2$ (3x) to $\ge 28.7$ (5x) [Exp 1 baseline: 5.74]
- **Levels Completed Target**: $\ge 45$ to $60+$ levels [Exp 1 baseline: 19]
- **Zero-Score Games**: Reduced from 5 games down to $\le 1$
- **Scheduler Max Wait**: Reduced from 1,420s down to $< 90\text{s}$ per turn
