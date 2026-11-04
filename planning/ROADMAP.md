# SARBLOH roadmap

Anchored on the official calendar, not on preference.

| Gate | Date | Days from 21 Sep |
| --- | --- | --- |
| Milestone Prize #2 (open-source) | 30 Sep 2026 | 9 |
| Final submission | 2 Nov 2026 | 42 |
| Paper track | 8 Nov 2026 | 48 |
| Results | 4 Dec 2026 | 74 |

## Phase 1 — spine (21-24 Sep)

Ranks 1-3 of the experiment shortlist, built as one system.

- [ ] `worldmodel`: induce executable `step()` from the transition record
- [ ] `parkh`: exact backtest over all recorded transitions, binary verdict
- [ ] `agent`: halt-on-misprediction commitment policy
- [ ] `surat`: lossless frame store with explicit retrieval
- [ ] One public environment completed end to end

**Gate:** if `step()` cannot backtest clean on the majority of public levels by 24 Sep, the executable
representation is not viable on a 27B backbone and the fallback is the natural-language rulebook. Decide on the
date, not later.

## Phase 2 — milestone submission (25-30 Sep)

- [ ] Harden, licence audit, `resources/repos/avo` excluded from the tree
- [ ] Open-source the repository
- [ ] **Submit to Milestone #2 on 30 Sep regardless of score**

The milestone rewards open-sourcing, not winning. A dated leaderboard placement is the cheapest verified number
available and it is six weeks ahead of the final submission.

## Phase 3 — the differentiator (1-12 Oct)

- [ ] `memory`: three clocks with explicit promotion and eviction
- [ ] Consolidation policy: episodic to semantic, verified-only
- [ ] Falsified-hypothesis log
- [ ] Held-out split frozen and enforced in `eval/bench.py`

**Gate:** re-derivation events must fall measurably against the single-store baseline. If they do not, the
three-clock thesis is wrong and the paper's central claim needs rewriting while there is still time.

## Phase 4 — exploration economics (13-24 Oct)

- [ ] k-parallel hypotheses
- [ ] Information-gain action selection
- [ ] Undo-scaffolded probing
- [ ] Macro-action induction

## Phase 5 — freeze (25 Oct - 1 Nov)

- [ ] Budget governor, early abort, stagnation supervisor
- [ ] Backbone sweep (Qwen 3.6 27B FP8 vs Gemma-4-31B)
- [ ] Full private-set-sized dry run inside the 12-hour ceiling
- [ ] Freeze 1 Nov, submit 2 Nov

## Phase 6 — paper (3-8 Nov)

Written from `history/ledger.jsonl`. If the ledger is thin, the paper is thin; there is no recovering that in
five days.

## Explicitly out of scope until 3 Nov

LoRA test-time training on verified transitions, SSM frame-history encoders, synthetic environment pretraining.
These have the highest research ceiling and are the natural centre of the phase-two programme. They are excluded
by decision, not oversight. Revisit after submission.
