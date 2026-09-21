# ARC Prize 2026 — rules and constraints snapshot

**Retrieved 2026-09-21.** Sources: [ARC Prize 2026](https://arcprize.org/competitions/2026),
[ARC-AGI-3 competition](https://arcprize.org/competitions/2026/arc-agi-3),
[ARC Prize testing policy](https://arcprize.org/policy),
[Milestone Prize #1](https://arcprize.org/blog/arc-prize-2026-milestone-1).

## Calendar

| Event | Date |
| --- | --- |
| Competition launch | 25 Mar 2026 |
| ARC-AGI-3 Milestone #1 | 30 Jun 2026 |
| ARC-AGI-3 Milestone #2 | **30 Sep 2026** |
| Final submissions | **2 Nov 2026** |
| Paper deadline | 8 Nov 2026 |
| Results | 4 Dec 2026 |

## Prizes — ARC-AGI-3 track ($850K)

- **Grand Prize $700K** — first eligible agent scoring 100% on evaluation; carries over if unclaimed.
- **Top Score $75K** — 1st $40K, 2nd $15K, 3rd $10K, 4th-5th $5K each.
- **Milestone $75K** — open-source submissions by each milestone date: $25K / $10K / $2.5K.

## Hard constraints

- Entries through the **Kaggle competition only**, as a Kaggle notebook.
- **No internet access during evaluation.** API-based systems (GPT, Claude, and the like) cannot be used.
- **Solutions must complete within 12 hours on Kaggle.** (Testing policy; supersedes any earlier working figure.)
- **All authored code must be open-sourced under a permissive public-domain licence** to be prize-eligible.
  Third-party code requires at least an open-source licence permitting public sharing.
- **UNCONFIRMED:** exact GPU allocation for this track. Check the competition Code and Data tabs and record here.

## Evaluation sets

| Tier | Access |
| --- | --- |
| Public | Fully open |
| Semi-private | Frontier model testing via zero-retention APIs; limited leakage acknowledged |
| Private | Competition leaderboards only |

Verified-testing tolerance for ARC-AGI-3: public and semi-private scores must agree within **±15 percentage
points**. Community submissions are unverified by default; exceptional open-source submissions may be selected
for verification.

## Milestone #1 results — the open-weight field

| Place | Entry | Model | Method |
| --- | --- | --- | --- |
| 1 | Tufa Labs, "The Duck" | Qwen 3.6 27B FP8, local | Python REPL; context eviction for infinite play; multimodal perception (renders, ASCII, segmentation). Reported hand-crafted tools *hurt* — improvisation did better. |
| 2 | Reki | Gemma-4-31B, local | VLM returning JSON actions per step; reflection memory; JSON self-repair; legal-action constraints. Built on the official GPT-OSS-120B template. Ablation-friendly via env vars. |
| 3 | forge (Md Boktiar Mahbub Murad) | Gemma-4-31B, local | Vision-to-JSON in a configurable framework; action generators, arbiters, confidence prompts. **Winning configuration disabled the extra machinery.** |

Two of the three top entries scored best with their elaborate components turned off. Treat every added component
as guilty until an ablation proves otherwise.

## Context: unverified public-set claims

NVIDIA AVO (100.00 RHAE), Schema / Impossible Research (98.98), VISTA (100.00) — all public set, all
self-reported, all driven by frontier API models with unbounded reasoning. None verified by ARC Prize. None
usable as a target under Kaggle's offline constraint. Useful only as sources of mechanism.
