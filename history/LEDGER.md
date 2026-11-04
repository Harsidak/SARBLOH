# SARBLOH ledger

Append-only. Newest first. One row per **scored** run. Every row cites its split; a number without a split is
not a result.

| Date | Exp | Hypothesis (one line) | Split | RHAE | Delta | Seeds | Git SHA | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-02 | E112 ls20 | Qwen3.8-27B NVFP4 + MTP5 on SGLang plays ls20 (serving check, E008 agent) | public smoke, 1 game (ls20) | 0.0357 (1/7 lv; L1 15 vs human 22 actions) | n/a (E109 SGLang FP8 got 1 action in 16 min; no prior ls20-only E008 run) | 1 | 4d356a4 dirty (embedded) | Serving pass: 0 freezes, 151 tok/s/stream in-game, 101 turns/60 min. Agent: L2 unsolved after 107 actions, ended on wall clock |
| 2026-09-28 | E005 10-game | Faithful Prime port plays 10 public games (Gemma-4-31B NVFP4) | public, 10 games | 0.0015 (1/76 lv) | +0.001 vs E003 (different games) | 1 | UNCONFIRMED | Harness pass; agent fail: brute-force loops, 350 resets, no goal found. See E005/autopsy_10game.md |
| 2026-09-28 | E003 v2 | Prime harness runs on Kaggle with Gemma-4-31B NVFP4 | public smoke, 6 games | 0.0005 (1/40 lv) | — | 1 | a26b167 | Plumbing pass; agent fail. 0 compactions, 0 spawns, 0 harness writes: ran as bare Gemma+REPL |
| 2026-09-21 | — | Repository reorganised under SARBLOH. No score. | — | — | — | — | — | — |

---

## Pre-ledger period

Results from before 2026-09-21 live in `legacy_scores/` and are **not attributable**: fourteen result files
(`battle2`, `champion`, `memo`, `llmagent_*`) with no mapping to the config or hypothesis that produced them.
They are preserved as evidence that work happened, not as evidence of what worked. Nothing in them may be cited
as a baseline until it is reproduced under the protocol.
