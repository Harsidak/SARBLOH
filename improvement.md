# Benchmark Improvement Tracker

This document tracks iterative performance, game-by-game results, and comparative improvements across experimental runs.

**Target**: Achieve **3x to 5x improvement** in score and level completion velocity over the Experiment 1 baseline before concluding iterations.

---

## 1. Run Progress & Multiplier Summary

| Exp # | Date / Timestamp | Run Directory | Wall Time | Levels Won | Mean Score | Total Actions | Level Velocity | Score Multiplier vs Exp 1 |
|---|---|---|---|---|---|---|---|---|
| **1 (Baseline)** | 2026-10-15 15:11 | `runs/kaggle_sarbloh-experimentation_20261015_151105` | 28.0 min (1,680.9s) | **19 / 108** | **5.7449** | 1,682 | 40.72 lev/hr | **1.0x (Baseline)** |
| **2** | 2026-10-16 00:32 | `runs/kaggle_sarbloh-experimentation_20261016_003252` | 31.5 min (1,892.2s) | **2 / 108** | **0.4233** | 158 | 3.81 lev/hr | **0.074x (-92.6%)** |
| **3** | 2026-10-16 02:42 | `runs/kaggle_sarbloh-experimentation_20261016_024252` | 30.4 min (1,823.0s) | **18 / 108** | **6.2742** | 1,651 | 35.57 lev/hr | **1.09x (+14.8x over Exp 2)** |
| **4 (with play)** | 2026-10-16 13:36 | `runs/kaggle_sarbloh-experimentation_20261016_133640` | 32.6 min (1,959.1s) | **21 / 108** | **8.4520** | 1,436 | 38.61 lev/hr | **1.47x (+34.7% over Exp 3)** |
| **5** | *(Pending)* | | | | | | | *(Target: 3.0x - 5.0x)* |

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

## 4. Experiment 3 — Recovery & Baseline Surpassed

- **Timestamp**: 2026-10-16 02:42:52 (downloaded from `kaggle_sarbloh-experimentation_20261016_024252`)
- **Key Changes Tested**: `hide_code` disabled (full code visibility in context restored); `prompts.py` streamlined with `short_objects` and `observe_view` options.
- **Summary**: 15 games evaluated, **18 levels cleared** out of 108, Mean Score **6.2742** (surpassing baseline 5.7449 by +9.2%, recovering 14.8x over Exp 2), Total Actions: **1,651**, Total Turns: **590** (2.80 actions/turn).

### Per-Game Breakdown Table

| Game ID | Levels Cleared | Score | Actions | Turns | State |
|---|---|---|---|---|---|
| `ar25-0c556536` | 2 / 8 | 8.33 | 174 | 47 | stopped |
| `bp35-0a0ad940` | 0 / 9 | 0.00 | 71 | 29 | stopped |
| `cd82-fb555c5d` | 2 / 6 | 8.59 | 115 | 51 | stopped |
| `cn04-2fe56bfb` | 0 / 6 | 0.00 | 176 | 65 | wall_clock |
| `dc22-fdcac232` | 2 / 6 | 14.29 | 168 | 72 | stopped |
| `ft09-0d8bbf25` | 3 / 6 | 28.57 | 147 | 42 | stopped |
| `g50t-5849a774` | 0 / 7 | 0.00 | 57 | 20 | stopped |
| `ka59-38d34dbb` | 0 / 7 | 0.00 | 33 | 18 | stopped |
| `lf52-271a04aa` | 1 / 10 | 1.82 | 21 | 19 | stopped |
| `lp85-305b61c3` | 1 / 8 | 2.78 | 66 | 39 | stopped |
| `ls20-9607627b` | 1 / 7 | 3.57 | 134 | 40 | wall_clock |
| `m0r0-492f87ba` | 1 / 6 | 4.76 | 66 | 43 | stopped |
| `r11l-495a7899` | 1 / 6 | 4.76 | 13 | 19 | stopped |
| `re86-8af5384d` | 3 / 8 | 15.83 | 343 | 63 | stopped |
| `s5i5-18d95033` | 1 / 8 | 0.81 | 67 | 23 | stopped |

### Head-to-Head 3-Way Comparison

| Game ID | Exp 1 (Baseline) | Exp 2 (Regression) | Exp 3 (Recent Run) | Exp 3 vs Baseline $\Delta$ |
|---|---|---|---|---|
| `ar25` | 2 lev (8.14) | 1 lev (2.78) | **2 lev (8.33)** | +0.19 pts |
| `bp35` | 0 lev (0.00) | 0 lev (0.00) | **0 lev (0.00)** | 0.00 pts |
| `cd82` | 2 lev (5.71) | 0 lev (0.00) | **2 lev (8.59)** | **+2.88 pts** |
| `cn04` | 0 lev (0.00) | 0 lev (0.00) | **0 lev (0.00)** | 0.00 pts |
| `dc22` | 2 lev (14.29) | 0 lev (0.00) | **2 lev (14.29)** | 0.00 pts |
| `ft09` | 3 lev (16.80) | 0 lev (0.00) | **3 lev (28.57)** | **+11.77 pts** |
| `g50t` | 0 lev (0.00) | 0 lev (0.00) | **0 lev (0.00)** | 0.00 pts |
| `ka59` | 0 lev (0.00) | 0 lev (0.00) | **0 lev (0.00)** | 0.00 pts |
| `lf52` | 0 lev (0.00) | 0 lev (0.00) | **1 lev (1.82)** | **+1 level (+1.82 pts)** |
| `lp85` | 3 lev (13.66) | 0 lev (0.00) | **1 lev (2.78)** | -10.88 pts |
| `ls20` | 1 lev (1.69) | 1 lev (3.57) | **1 lev (3.57)** | **+1.88 pts** |
| `m0r0` | 1 lev (4.76) | 0 lev (0.00) | **1 lev (4.76)** | 0.00 pts |
| `r11l` | 1 lev (4.76) | 0 lev (0.00) | **1 lev (4.76)** | 0.00 pts |
| `re86` | 2 lev (8.03) | 0 lev (0.00) | **3 lev (15.83)** | **+1 level (+7.80 pts)** |
| `s5i5` | 2 lev (8.33) | 0 lev (0.00) | **1 lev (0.81)** | -7.52 pts |
| **TOTALS / MEAN** | **19 lev (5.74)** | **2 lev (0.42)** | **18 lev (6.27)** | **+0.53 pts (+9.2%)** |

---

## 5. Experiment 4 — The `play()` Feature Benchmark

- **Timestamp**: 2026-10-16 13:36:40
- **Run Directory**: `runs/kaggle_sarbloh-experimentation_20261016_133640`
- **Key Change**: Added `play(path, predict=..., state=...)` kernel helper (E123). Allows executing multi-move plans one step at a time, halting automatically on the first unexpected mismatch ("surprise"), level up, or game over.
- **Summary**: 
  - **Levels Won**: **21 / 108** (Highest all-time: +3 over Exp 3, +2 over Exp 1)
  - **Mean Score**: **8.4520** (Highest all-time: +34.7% over Exp 3, +47.1% over Exp 1)
  - **Tokens Wasted on Uncleared Levels**: Dropped from 71.4% to **61.5%** (-9.9% waste reduction)
  - **Output Tokens per Cleared Level**: Dropped from 69,928 down to **63,521** (-9.2% token cost reduction)
  - **Median Turns to Clear Level 1**: Dropped from 13.0 turns to **10.0 turns** (-23%)
  - **Wall Time**: 32.6 minutes (1,959.1s) across 628 turns and 1,436 actions.

### Per-Game Breakdown Table

| Game ID | Levels Cleared | Score | Actions | Turns | State |
|---|---|---|---|---|---|
| `ar25-0c556536` | 1 / 8 | 2.78 | 41 | 25 | cancelled |
| `bp35-0a0ad940` | 0 / 9 | 0.00 | 65 | 39 | cancelled |
| `cd82-fb555c5d` | 0 / 6 | 0.00 | 32 | 21 | cancelled |
| `cn04-2fe56bfb` | 2 / 6 | 13.45 | 152 | 105 | wall_clock |
| `dc22-fdcac232` | 2 / 6 | 14.29 | 129 | 52 | cancelled |
| `ft09-0d8bbf25` | 3 / 6 | 28.57 | 176 | 37 | cancelled |
| `g50t-5849a774` | 0 / 7 | 0.00 | 17 | 18 | cancelled |
| `ka59-38d34dbb` | 0 / 7 | 0.00 | 54 | 24 | cancelled |
| `lf52-271a04aa` | 1 / 10 | 1.82 | 141 | 43 | cancelled |
| `lp85-305b61c3` | 3 / 8 | 16.67 | 73 | 51 | cancelled |
| `ls20-9607627b` | 1 / 7 | 3.57 | 80 | 29 | cancelled |
| `m0r0-492f87ba` | 1 / 6 | 4.76 | 144 | 46 | cancelled |
| `r11l-495a7899` | 1 / 6 | 4.76 | 24 | 34 | cancelled |
| `re86-8af5384d` | 4 / 8 | 27.78 | 203 | 57 | cancelled |
| `s5i5-18d95033` | 2 / 8 | 8.33 | 105 | 47 | cancelled |

### Head-to-Head 4-Way Comparison

| Game ID | Exp 1 (Baseline) | Exp 3 (091657) | Exp 4 (New Run 133640) | Exp 4 vs Exp 3 $\Delta$ | Exp 4 vs Exp 1 $\Delta$ |
|---|---|---|---|---|---|
| `ar25` | 2 lev (8.14) | 2 lev (8.33) | **1 lev (2.78)** | -5.55 pts | -5.36 pts |
| `bp35` | 0 lev (0.00) | 0 lev (0.00) | **0 lev (0.00)** | 0.00 pts | 0.00 pts |
| `cd82` | 2 lev (5.71) | 2 lev (8.59) | **0 lev (0.00)** | -8.59 pts | -5.71 pts |
| `cn04` | 0 lev (0.00) | 0 lev (0.00) | **2 lev (13.45)** | **+13.45 pts (+2 lev)** | **+13.45 pts (+2 lev)** |
| `dc22` | 2 lev (14.29) | 2 lev (14.29) | **2 lev (14.29)** | 0.00 pts | 0.00 pts |
| `ft09` | 3 lev (16.80) | 3 lev (28.57) | **3 lev (28.57)** | 0.00 pts | **+11.77 pts** |
| `g50t` | 0 lev (0.00) | 0 lev (0.00) | **0 lev (0.00)** | 0.00 pts | 0.00 pts |
| `ka59` | 0 lev (0.00) | 0 lev (0.00) | **0 lev (0.00)** | 0.00 pts | 0.00 pts |
| `lf52` | 0 lev (0.00) | 1 lev (1.82) | **1 lev (1.82)** | 0.00 pts | **+1.82 pts** |
| `lp85` | 3 lev (13.66) | 1 lev (2.78) | **3 lev (16.67)** | **+13.89 pts (+2 lev)** | **+3.01 pts** |
| `ls20` | 1 lev (1.69) | 1 lev (3.57) | **1 lev (3.57)** | 0.00 pts | **+1.88 pts** |
| `m0r0` | 1 lev (4.76) | 1 lev (4.76) | **1 lev (4.76)** | 0.00 pts | 0.00 pts |
| `r11l` | 1 lev (4.76) | 1 lev (4.76) | **1 lev (4.76)** | 0.00 pts | 0.00 pts |
| `re86` | 2 lev (8.03) | 3 lev (15.83) | **4 lev (27.78)** | **+11.95 pts (+1 lev)** | **+19.75 pts (+2 lev)** |
| `s5i5` | 2 lev (8.33) | 1 lev (0.81) | **2 lev (8.33)** | **+7.52 pts (+1 lev)** | 0.00 pts |
| **TOTALS / MEAN** | **19 lev (5.74)** | **18 lev (6.27)** | **21 lev (8.452)** | **+2.18 pts (+34.7%)** | **+2.71 pts (+47.1%)** |

---

## 6. Key Performance Indicators (KPIs) for Success

To satisfy the **3x to 5x improvement** criterion:
- **Mean Score Target**: $\ge 17.2$ (3x) to $\ge 28.7$ (5x) [Current: 8.452 (1.47x)]
- **Levels Completed Target**: $\ge 45$ to $60+$ levels [Current: 21 levels]
- **Zero-Score Games**: Reduced from 4–5 games down to $\le 1$ [Current: 4 games: `bp35`, `g50t`, `ka59`, `cd82`]
- **Scheduler Max Wait**: Reduced from 1,146s down to $< 120\text{s}$ per turn [Current: up to 1,146s]
