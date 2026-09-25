# Experiment backlog

The candidate space, ranked. **A directory under `experiments/` is created only when an experiment is
opened** — this file is the catalogue, not the work. Copy `E000_TEMPLATE/` when you open one, and set the
status here in the same commit.

Status values: `open` · `running` · `kept` · `killed` · `inconclusive` · `superseded` · blank = not started.

## Revisions since first draft (2026-09-20)

Three findings from the 22 Sep reconnaissance change this list. Read them before picking anything.

1. **The open-weight state of the art is 1.21% RHAE.** Tufa Labs won Milestone #1 at that score; their own
   25-game / 20-pass benchmark means 1.6002 ± 0.4475. Any target above ~5 RHAE on the public set is not a
   stretch goal, it is a misunderstanding of the field. Re-read every "expected delta" below in that light.
2. **E14 is already built and open-sourced.** Tufa's Duck harness *is* the FIFO baseline, with a complete
   25x20 run and a viewer in the repository. Do not rebuild it. Fork it, reproduce the run, and use their
   published numbers as `B000_fifo`. This saves roughly two days and makes every later comparison a
   controlled ablation against the reigning winner on their own harness.
3. **Wall clock is 12 hours, not 9.** Budget experiments (E63-E65, E75) re-scope accordingly.

A fourth finding is not a revision but a warning. Three independent teams — Tufa (hand-crafted tools hurt),
forge (won with its own machinery disabled), and the Nemotron winner (plain cross-entropy beat every complex
scheme) — all report that elaborate additions lose to a simple thing tuned well. **Every experiment below must
be switchable off from `config.yaml`.** A module that cannot be disabled produces no ablation and therefore no
paper.

## Method

Diagnosis before permutation. The prerequisite for picking anything off this list is E-DIAG below; without it
you are permuting blind, and the calendar will run out before the list does.

| ID | Experiment | Axis | Rank | Status |
| --- | --- | --- | --- | --- |
| E-DIAG | **Failure autopsy of the Duck's published run** — which of the 25 games never clear level 1, and is the cause failed induction or lost context? Binary answer, decides whether the memory thesis holds. | diagnosis | **0** | |
| E01 | Lossless frame store outside context with agent-chosen `inspect` at frame/region/pixel level | substrate | 3 | |
| E02 | Frame-delta encoding — store changed cells only | perception |  | |
| E03 | Object-token / run-length grid serialisation | perception |  | |
| E04 | PNG render only, VLM path | perception |  | |
| E05 | Dual-channel: ASCII grid plus render in one turn | perception |  | |
| E06 | Deterministic connected-component segmentation before the model sees anything | perception |  | |
| E07 | Object permanence — persistent IDs matched across frames | substrate |  | |
| E08 | Transition tuple database, `(s,a,s')`, queryable | substrate |  | |
| E09 | Executable `step()` file as the sole persistent model | substrate | 1 | |
| E10 | Natural-language rulebook, no code — the Phase 1 fallback | substrate |  | |
| E11 | Dual representation, `step()` plus rulebook, disagreements flagged | substrate |  | |
| E12 | Three-clock memory: frame / level / environment, separate update rates | clock | 4 | |
| E13 | Surprise-gated writes — store a frame only on misprediction | consolidation |  | |
| E14 | FIFO eviction baseline — **superseded: fork Tufa's Duck instead** | consolidation | — | |
| E15 | MDL refactor pass — periodically rewrite memory to shortest faithful form | consolidation |  | |
| E16 | Append-only timeline plus periodic episodic-to-semantic consolidation | consolidation | 4 | |
| E17 | Verified-only semantic store — nothing promoted before backtest | consolidation | 4 | |
| E18 | Falsified-hypothesis log — refuted rules never re-proposed | consolidation | 4 | |
| E19 | Action-outcome index keyed by (object class, action) | retrieval |  | |
| E20 | Persistent 64x64 semantic annotation overlay | substrate |  | |
| E21 | Explicit versioned goal hypothesis | substrate |  | |
| E22 | Score-delta memory — every score change as a labelled supervision event | substrate |  | |
| E23 | Undo as checkpoint primitive | substrate |  | |
| E24 | Level-boundary carryover ablation — reset vs carry mechanics | clock |  | |
| E25 | Run-clock prior file persisting across environments in one run | clock |  | |
| E26 | Library learning — extract reusable primitives from solved levels | abstraction |  | |
| E27 | Game-specific DSL induction, then `step()` inside it | abstraction |  | |
| E28 | Refactor-for-generality pass on a schedule | abstraction |  | |
| E29 | Grounding / mechanism split — objects solved separately from rules | abstraction |  | |
| E30 | Frozen grounding — fix object vocabulary early | abstraction |  | |
| E31 | Contradiction-triggered re-grounding | abstraction |  | |
| E32 | Shared cross-environment primitive vocabulary | abstraction |  | |
| E33 | Analogy retrieval — match to nearest solved mechanic | retrieval |  | |
| E34 | Mechanic taxonomy classifier: push / toggle / collect / gate / timer / sort | abstraction |  | |
| E35 | Parameterised schema slots instantiated per environment | abstraction |  | |
| E36 | Explicit tutorial mode — spend freely on early levels by design | budget |  | |
| E37 | Level difficulty inference for exploration allocation | budget |  | |
| E38 | Counterfactual annotation — record what was expected and why it was wrong | abstraction |  | |
| E39 | Causal graph induction over object interactions | abstraction |  | |
| E40 | Invariant discovery as hard constraints on hypotheses | abstraction |  | |
| E41 | Symmetry detection — translation, reflection, colour permutation | abstraction |  | |
| E42 | Failure abstraction — generalise plan failures into reusable warnings | abstraction |  | |
| E43 | Skill cards — natural-language skill written after each cleared level | abstraction |  | |
| E44 | Executable skill library callable by the planner | abstraction |  | |
| E45 | Subgoal decomposition memory — persist the tree, not the trace | abstraction |  | |
| E46 | Macro-action induction from repeated verified sequences | action policy | 7 | |
| E47 | MDL-ranked hypotheses — shortest `step()` that backtests clean | hypotheses |  | |
| E48 | Cross-model agreement — Qwen and Gemma propose, keep the intersection | hypotheses |  | |
| E49 | Simulator self-play — adversarial action sequences before committing | verification |  | |
| E50 | Abstraction audit — `step()` length and refactor count vs held-out RHAE | abstraction |  | |
| E51 | Exact backtest gate — no commit until all transitions replay exactly | verification | 1 | |
| E52 | Partial-credit backtest at k% recency-weighted | verification |  | |
| E53 | Probabilistic world model with per-rule confidence | verification |  | |
| E54 | Halt-on-misprediction executor | verification | 1 | |
| E55 | BFS inside the certified simulator | action policy | 2 | |
| E56 | A* with a heuristic induced from the goal hypothesis | action policy |  | |
| E57 | Beam search over plans, re-ranked on simulated outcome | action policy |  | |
| E58 | Information-gain action selection — maximise bits per committed action | action policy | 5 | |
| E59 | k-parallel hypotheses maintained simultaneously | hypotheses | 5 | |
| E60 | Ensemble vote — act on agreement, probe on disagreement | hypotheses |  | |
| E61 | Thompson sampling over hypotheses | hypotheses |  | |
| E62 | Expected-RHAE planning objective, not step count | action policy |  | |
| E63 | Human-baseline estimation per level — makes E62 implementable | budget |  | |
| E64 | Early abort on environments showing no modelling progress | budget | 8 | |
| E65 | Marginal-gain compute allocation across the 12 hours | budget | 8 | |
| E66 | Stagnation supervisor detecting unproductive cycles | budget | 8 | |
| E67 | Seeded restart on stagnation | budget |  | |
| E68 | Commitment-length sweep: 1 / 3 / 5 / full plan prefix | action policy |  | |
| E69 | Undo-scaffolded probing — probe, observe, revert, record | action policy | 6 | |
| E70 | Proposer / critic split across two passes | hypotheses |  | |
| E71 | Small local verifier model for backtest triage | verification |  | |
| E72 | Speculative execution of a high-confidence prefix while re-inducing | budget |  | |
| E73 | Batched hypothesis inference on the 96 GB card | budget |  | |
| E74 | KV-cache reuse for the static prompt across turns | budget |  | |
| E75 | Budget governor with hard accounting and graceful degradation | budget | 8 | |
| E76 | LoRA TTT on the run's own backtested transitions — **phase two** | adaptation | — | |
| E77 | LoRA TTT on symmetry-augmented verified transitions — **phase two** | adaptation | — | |
| E78 | Prompt-cache warm start per environment | adaptation |  | |
| E79 | In-context skill injection at environment start | adaptation |  | |
| E80 | Distilled System-1 change detector — real state change vs animation | adaptation |  | |
| E81 | Distilled object-role classifier | adaptation |  | |
| E82 | Distilled no-op predictor before every commit | adaptation | 10 | |
| E83 | Speculative decoding with a small draft model | budget |  | |
| E84 | Quantisation ablation: FP8 vs INT4 vs AWQ | budget |  | |
| E85 | Backbone comparison: Qwen 3.6 27B FP8 vs Gemma-4-31B vs ensemble | adaptation | 9 | |
| E86 | Reasoning-token budget sweep | budget |  | |
| E87 | Self-consistency sampling on `step()` induction | hypotheses | 9 | |
| E88 | Phase-tied temperature — high exploring, low executing | hypotheses |  | |
| E89 | Synthetic ARC-AGI-3-like environment generator — **phase two** | adaptation | — | |
| E90 | Offline RL on logged trajectories — **phase two** | adaptation | — | |
| E91 | Behaviour cloning on human action traces — **phase two** | adaptation | — | |
| E92 | Mechanic-curriculum fine-tune, coarse to fine — **phase two** | adaptation | — | |
| E93 | SSM / Mamba frame-history encoder — **phase two**, [[sgrpo]] crossover | adaptation | — | |
| E94 | State-space memory as context replacing file retrieval — **phase two** | adaptation | — | |
| E95 | Latent vs explicit memory ablation — **phase two**, the paper's central comparison | adaptation | — | |
| E96 | Recursive sub-solver for bounded grid subproblems — **phase two** | adaptation | — | |
| E97 | Neural backtest surrogate for triage — **phase two** | verification | — | |
| E98 | Learned exploration policy over the action space — **phase two** | adaptation | — | |
| E99 | Meta-controller selecting harness mode per environment | hypotheses |  | |
| E100 | Full-stack integration under the held-out protocol — the submission | integration |  | |


## Ranked shortlist

The ten to actually run, in order. Ranks in the table above refer to this list.

| Rank | Experiments | What it establishes |
| --- | --- | --- |
| **0** | E-DIAG | Whether the memory thesis survives contact with the champion's own failure data |
| 1 | E09 + E51 + E54 | Executable world model, exact backtest, halt on misprediction — the spine |
| 2 | E55 | Planning becomes free; the core RHAE lever |
| 3 | E01 | Removes the named context-exhaustion failure |
| 4 | E12 + E16 + E17 + E18 | Three-clock memory with verified-only promotion — the novel contribution |
| 5 | E58 + E59 | Bits per action, and no premature commitment |
| 6 | E69 | Undo as an experimental primitive — unexploited by every published harness |
| 7 | E46 | Attacks the action count directly on later levels |
| 8 | E64 + E65 + E66 + E75 | Converts a timeout into a score |
| 9 | E85 + E87 | Cheapest reliability gain available |
| 10 | E82 | Removes wasted actions at minimal inference cost |

Phase-two items (marked **—**) are excluded by decision, not oversight. Revisit 3 Nov.

## Rules

1. Nothing is opened without a `hypothesis.md` stating mechanism, prediction and kill criterion **first**.
2. Nothing is reported off the tune split. Held-out only.
3. Any delta below the measured noise floor is not a result. Record it as noise; do not quietly drop it.
4. Kill criteria are binding. A killed experiment gets a `verdict.md` and a status change here.
5. Negative results carry the same weight as positive ones. An ablation containing only wins is not an ablation.
