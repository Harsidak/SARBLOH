# Benchmark Improvement Tracker

This document tracks iterative performance, game-by-game results, and comparative improvements across experimental runs.

**Target**: Achieve **3x to 5x improvement** in score and level completion velocity over the Experiment 1 baseline before concluding iterations.

---

## 1. Run Progress & Multiplier Summary

| Exp # | Date / Timestamp | Run Directory | Wall Time | Levels Won | Mean Score | Total Actions | Level Velocity | Score Multiplier vs Exp 1 |
|---|---|---|---|---|---|---|---|---|
| **1 (Baseline)** | 2026-10-15 15:11 | `runs/kaggle_sarbloh-experimentation_20261015_151105` | 28.0 min (1,680.9s) | **19 / 108** | **5.7449** | 1,682 | 40.72 lev/hr | **1.0x (Baseline)** |
| **2** | *(Pending)* | | | | | | | *(Target: 3.0x - 5.0x)* |
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

## 3. Experiment 2 — Comparative Benchmark *(Template)*

- **Timestamp**: *(TBD)*
- **Experiment Config**: `experimentation_45min_15games`
- **Changes Deployed**: *(TBD — e.g. Anti-starvation scheduler aging, hypothesis status normalization, oscillation breaker, goal prompt guardrails)*

### Per-Game Breakdown Table

| Game ID | Levels Won | Score | Actions | Turns | State |
|---|---|---|---|---|---|
| `bp35-0a0ad940` | - / 9 | - | - | - | - |
| `cn04-2fe56bfb` | - / 6 | - | - | - | - |
| `ka59-38d34dbb` | - / 7 | - | - | - | - |
| `lf52-271a04aa` | - / 10 | - | - | - | - |
| `g50t-5849a774` | - / 7 | - | - | - | - |
| `ls20-9607627b` | - / 7 | - | - | - | - |
| `m0r0-492f87ba` | - / 6 | - | - | - | - |
| `r11l-495a7899` | - / 6 | - | - | - | - |
| `cd82-fb555c5d` | - / 6 | - | - | - | - |
| `re86-8af5384d` | - / 8 | - | - | - | - |
| `ar25-0c556536` | - / 8 | - | - | - | - |
| `s5i5-18d95033` | - / 8 | - | - | - | - |
| `lp85-305b61c3` | - / 8 | - | - | - | - |
| `dc22-fdcac232` | - / 6 | - | - | - | - |
| `ft09-0d8bbf25` | - / 6 | - | - | - | - |

### Head-to-Head Comparison: Exp 1 vs Exp 2

| Game ID | Levels Won (Exp 1 $\rightarrow$ Exp 2) | Score (Exp 1 $\rightarrow$ Exp 2) | Action Efficiency $\Delta$ | Observations / Outcome |
|---|---|---|---|---|
| `bp35-0a0ad940` | 0/9 $\rightarrow$ | 0.00 $\rightarrow$ | | |
| `cn04-2fe56bfb` | 0/6 $\rightarrow$ | 0.00 $\rightarrow$ | | |
| `ka59-38d34dbb` | 0/7 $\rightarrow$ | 0.00 $\rightarrow$ | | |
| `lf52-271a04aa` | 0/10 $\rightarrow$ | 0.00 $\rightarrow$ | | |
| `g50t-5849a774` | 0/7 $\rightarrow$ | 0.00 $\rightarrow$ | | |
| `ls20-9607627b` | 1/7 $\rightarrow$ | 1.69 $\rightarrow$ | | |
| `m0r0-492f87ba` | 1/6 $\rightarrow$ | 4.76 $\rightarrow$ | | |
| `r11l-495a7899` | 1/6 $\rightarrow$ | 4.76 $\rightarrow$ | | |
| `cd82-fb555c5d` | 2/6 $\rightarrow$ | 5.71 $\rightarrow$ | | |
| `re86-8af5384d` | 2/8 $\rightarrow$ | 8.03 $\rightarrow$ | | |
| `ar25-0c556536` | 2/8 $\rightarrow$ | 8.14 $\rightarrow$ | | |
| `s5i5-18d95033` | 2/8 $\rightarrow$ | 8.33 $\rightarrow$ | | |
| `lp85-305b61c3` | 3/8 $\rightarrow$ | 13.66 $\rightarrow$ | | |
| `dc22-fdcac232` | 2/6 $\rightarrow$ | 14.29 $\rightarrow$ | | |
| `ft09-0d8bbf25` | 3/6 $\rightarrow$ | 16.80 $\rightarrow$ | | |
| **TOTALS / MEAN** | **19 $\rightarrow$** | **5.74 $\rightarrow$** | | |

---

## 4. Key Performance Indicators (KPIs) for Success

To satisfy the **3x to 5x improvement** criterion:
- **Mean Score Target**: $\ge 17.2$ (3x) to $\ge 28.7$ (5x) [Exp 1 baseline: 5.74]
- **Levels Completed Target**: $\ge 45$ to $60+$ levels [Exp 1 baseline: 19]
- **Zero-Score Games**: Reduced from 5 games down to $\le 1$
- **Scheduler Max Wait**: Reduced from 1,420s down to $< 90\text{s}$ per turn
