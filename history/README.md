# history/ — what actually moved the score

| File | Role |
| --- | --- |
| `LEDGER.md` | Human-readable, append-only, newest first. |
| `ledger.jsonl` | Machine-readable, one row per scored run. Plots and the paper read this. |
| `baselines.json` | Named reference points every delta is measured against. |
| `legacy_scores/` | Pre-reorg results from `scores/`. Preserved, **unattributed** — no config or hypothesis maps to them. |

A run that is not in the ledger did not happen. Logging is part of the run, not a step afterwards; the August
2026 result set is unattributable precisely because it was going to be written up later.
