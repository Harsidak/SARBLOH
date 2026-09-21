# eval/ — the scoring spine

This directory predates the reorganisation and is the healthiest part of the repository. It stays.

| File | Role |
| --- | --- |
| `official_score.py` | Official RHAE computation. The only scorer whose output goes in the ledger. |
| `bench.py` | Benchmark driver across environments. |
| `scoreboard.py` | Standings across runs. |
| `components.py` | Component-level evaluation. |
| `agent_loader.py` | Agent resolution for the bench. |
| `real_games/` | 24 local environment replicas + `games_index.json`. |
| `splits/` | Frozen tune / held-out split. |
| `suites/` | Component suites. |
| `probes/` | Diagnostic probes reclaimed from `_archive/harnesses_2026-08-06/`. |

## The protocol

The split in `splits/heldout_split.json` is frozen. Tune on the tune set; report on the held-out set. A number
quoted without naming its split is not a result.

Public environments are contaminated — the models were released after the public games shipped — and ARC Prize
states the private set has limited mechanical overlap with them. Treat public RHAE as a smoke test, never as
evidence.
