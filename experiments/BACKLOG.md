# Experiment backlog

The candidate space, ranked. **A directory under `experiments/` is created only when an experiment is
opened** — this file is the catalogue, not the work. Copy `E000_TEMPLATE/` when you open one, and set the
status here in the same commit.

Status values: `open` · `running` · `complete` · `kept` · `killed` · `inconclusive` · `superseded` · `deprecated` (dropped without running) · blank = not started.

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
| E-DIAG | **Failure autopsy of the Duck's published run** — which of the 25 games never clear level 1, and is the cause failed induction or lost context? Binary answer, decides whether the memory thesis holds. | diagnosis | **0** | complete |
| E01 | Lossless frame store outside context with agent-chosen `inspect` at frame/region/pixel level | substrate | 3 | kept |
| E02 | Frame-delta encoding — store changed cells only | perception |  | kept |
| E03 | Object-token / run-length grid serialisation | perception |  | kept |
| E04 | PNG render only, VLM path | perception |  | superseded |
| E05 | Dual-channel: ASCII grid plus render in one turn | perception |  | superseded |
| E06 | Deterministic connected-component segmentation before the model sees anything | perception |  | superseded |
| E07 | Object permanence — persistent IDs matched across frames | substrate |  | kept |
| E08 | Transition tuple database, `(s,a,s')`, queryable | substrate |  | kept |
| E09 | Executable `step()` file as the sole persistent model | substrate | 1 | kept |
| E10 | Natural-language rulebook, no code — the Phase 1 fallback | substrate |  | kept |
| E11 | Dual representation, `step()` plus rulebook, disagreements flagged | substrate |  | kept |
| E12 | Three-clock memory: frame / level / environment, separate update rates | clock | 4 | kept |
| E13 | Surprise-gated writes — store a frame only on misprediction | consolidation |  | killed |
| E14 | FIFO eviction baseline — **superseded: fork Tufa's Duck instead** | consolidation | — | superseded |
| E15 | MDL refactor pass — periodically rewrite memory to shortest faithful form | consolidation |  | kept |
| E16 | Append-only timeline plus periodic episodic-to-semantic consolidation | consolidation | 4 | kept |
| E17 | Verified-only semantic store — nothing promoted before backtest | consolidation | 4 | killed |
| E18 | Falsified-hypothesis log — refuted rules never re-proposed | consolidation | 4 | |
| E19 | Action-outcome index keyed by (object class, action) | retrieval |  | kept |
| E20 | Persistent 64x64 semantic annotation overlay | substrate |  | kept |
| E21 | Explicit versioned goal hypothesis | substrate |  | |
| E22 | Score-delta memory — every score change as a labelled supervision event | substrate |  | |
| E23 | Undo as checkpoint primitive | substrate |  | killed |
| E24 | Level-boundary carryover ablation — reset vs carry mechanics | clock |  | kept |
| E25 | Run-clock prior file persisting across environments in one run | clock |  | deprecated |
| E26 | Library learning — extract reusable primitives from solved levels | abstraction |  | kept |
| E27 | Game-specific DSL induction, then `step()` inside it | abstraction |  | |
| E28 | Refactor-for-generality pass on a schedule | abstraction |  | |
| E29 | Grounding / mechanism split — objects solved separately from rules | abstraction |  | |
| E30 | Frozen grounding — fix object vocabulary early | abstraction |  | |
| E31 | Contradiction-triggered re-grounding | abstraction |  | killed |
| E32 | Shared cross-environment primitive vocabulary | abstraction |  | |
| E33 | Analogy retrieval — match to nearest solved mechanic | retrieval |  | |
| E34 | Mechanic taxonomy classifier: push / toggle / collect / gate / timer / sort | abstraction |  | kept |
| E35 | Parameterised schema slots instantiated per environment | abstraction |  | deprecated |
| E36 | Explicit tutorial mode — spend freely on early levels by design | budget |  | |
| E37 | Level difficulty inference for exploration allocation | budget |  | |
| E38 | Counterfactual annotation — record what was expected and why it was wrong | abstraction |  | kept |
| E39 | Causal graph induction over object interactions | abstraction |  | kept |
| E40 | Invariant discovery as hard constraints on hypotheses | abstraction |  | killed |
| E41 | Symmetry detection — translation, reflection, colour permutation | abstraction |  | |
| E42 | Failure abstraction — generalise plan failures into reusable warnings | abstraction |  | deprecated |
| E43 | Skill cards — natural-language skill written after each cleared level | abstraction |  | kept |
| E44 | Executable skill library callable by the planner | abstraction |  | kept |
| E45 | Subgoal decomposition memory — persist the tree, not the trace | abstraction |  | killed |
| E46 | Macro-action induction from repeated verified sequences | action policy | 7 | |
| E47 | MDL-ranked hypotheses — shortest `step()` that backtests clean | hypotheses |  | deprecated |
| E48 | Cross-model agreement — Qwen and Gemma propose, keep the intersection | hypotheses |  | deprecated |
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
| E59 | k-parallel hypotheses maintained simultaneously | hypotheses | 5 | deprecated |
| E60 | Ensemble vote — act on agreement, probe on disagreement | hypotheses |  | deprecated |
| E61 | Thompson sampling over hypotheses | hypotheses |  | |
| E62 | Expected-RHAE planning objective, not step count | action policy |  | |
| E63 | Human-baseline estimation per level — makes E62 implementable | budget |  | |
| E64 | Early abort on environments showing no modelling progress | budget | 8 | |
| E65 | Marginal-gain compute allocation across the 12 hours | budget | 8 | |
| E66 | Stagnation supervisor detecting unproductive cycles | budget | 8 | |
| E67 | Seeded restart on stagnation | budget |  | |
| E68 | Commitment-length sweep: 1 / 3 / 5 / full plan prefix | action policy |  | |
| E69 | Undo-scaffolded probing — probe, observe, revert, record | action policy | 6 | |
| E70 | Proposer / critic split across two passes | hypotheses |  | killed |
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
| E83 | Speculative decoding with a small draft model | budget |  | kept |
| E84 | Quantisation ablation: FP8 vs INT4 vs AWQ | budget |  | kept |
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
| E001 | **Duck + fast vLLM** — `kaggle/main.ipynb`: Tufa's bundle unchanged, our profile-chain launcher (fast → fast_nospec → duck), watchdog. Gate: ≥ the Duck's ~4.5 offline | serving | **S** | open |
| E003 | **Prime Agent harness smoke** — `Sarbloh-Prime/` (ADR-0002), Gemma-4-31B NVFP4, `kaggle/experimental/002_prime.ipynb`, 1 hour, 6 public games. Gate: vLLM up + no session crashes | substrate | **S** | complete |
| E101 | **FreeToken serving** (arXiv 2608.16157) instead of vLLM on Kaggle — *later*. MoE-only expert offload: only pays if the backbone is an MoE too big for 96 GB. Its semantic-anchor state checkpoints at tool/thinking boundaries are separately testable in vLLM terms (prefix reuse under Duck eviction) | serving |  | |
| E102 | **Wave scheduling for skill transfer** (Prime): play games in waves of k (1 / 2-3 / all) so reflections from wave n reach wave n+1 through `global_harness/`. Today all games start at once, so no game can benefit from another's skills. Optimisation, not a bug | self-improvement |  | deprecated |
| E103 | **Locked `global_harness/` writes** (Prime): concurrent sessions write one file; lost updates UNCONFIRMED. Measure first (count refinements written vs present), then add a file lock | self-improvement |  | |
| E104 | **Qwen Coder backbone in the Prime harness**: same config as E005, Gemma-4-31B swapped for a Qwen Coder model (exact checkpoint UNCONFIRMED). Prime + Gemma is the baseline | backbone |  | deprecated |
| E105 | **Oracle briefing: understanding vs execution** (Prime, ls20 only): the owner writes a full briefing (goal, action effects, dynamics, hazards, per-level notes) that goes into the task message. Arms: none (baseline) / full briefing (optionally goal-only). Clears levels with the briefing -> execution is fine, understanding is the bottleneck; fails even with it -> fix the harness/execution first | diagnosis | **next** | kept |

| E106 | **Timeline in TOON instead of JSON** — same timeline sub-memory, serialised as TOON (Token-Oriented Object Notation) rather than JSON in the REPL. Measure tokens per timeline entry and whether the agent reads it as accurately | substrate |  | open |
| E107 | **LoRA SFT pipeline smoke** — `Alignment/lora_sft.py` (was `sft/`) on Kaggle (T4, Qwen3.5-2B, 3 Prime-style trajectories) ends in a PASS result card: masks, LoRA path, adapter save + reload all checked. Prerequisite for SFT on our own trajectories | adaptation |  | **kept** (2026-10-01: Kaggle T4 PASS, loss 1.149→0.003, reload exact, 3/3 generations exact; see verdict.md) |
| E108 | **LoRA RL (GRPO) pipeline smoke** — `Alignment/lora_rl.py` in the shared `Alignment/alignment.ipynb` beside SFT (T4, Qwen3.5-2B, 3 prompts, verifiable `tool_match` reward, KL to the adapter-disabled reference) ends in a PASS card: reward recognises its references, greedy reward rises ≥ 0.1, adapter save + reload checked. Makes the RL-vs-SFT comparison of CLAUDE.md §0 runnable | adaptation |  | kept: SFT+RL PASS on T4 in 30 min; RL gain is format only (see verdict) |
| E109 | **SGLang serving (Pennyroyal RTX Pro 6000 build)** instead of vLLM: same Qwen3.8-27B-FP8 weights, MTP speculative decoding with the model's own MTP layer, offline wheelhouse `dfranzen/pennyroyal-v253` (licence unknown: UNCONFIRMED for a submission). Profiles `sglang_fp8_mtp_vision` -> `sglang_fp8_vision` -> `sglang_fp8` -> vLLM. Gate: 8-stream probe >= 1.2x vLLM's 315 tok/s; inference check on ls20 on Kaggle | budget | **next** | open |
| E110 | **Short-thinking prompts** (`agent.prompt_version: e110`): time paragraph (thinking costs minutes), never count cells in thinking (ipython), BFS + batch the path, HUD strip rule, "the game is solvable", level-up carry-over. From the 2026-10-01 run: 6,367 completion tokens per action, 9.4% of turns cut at the output limit. Ideas from the Taaf agent prompts, our wording | action policy |  | open |
| E111 | **Priority scheduler** (`scheduler.enabled`): all games alive, `slots` on the GPU; a slot goes to the waiting game with the best next-level weight x hope (hope halves at 80k tokens on one level); quantum 4 calls, re-compete at level-up. Replaces fixed 90-min waves (E65/E64 made concrete) | budget |  | open |
| E113 | **Engine-grounded world model** (owner idea, 2026-10-02): give the REPL the `arcengine` game-authoring API so the agent writes its world model as a candidate *game definition* (sprites, levels, collisions, win condition) instead of a free-form `step()`; host `clone_check` replays recorded actions in the clone (exact frame match), `clone_plan` BFS-searches it. Phase A = offline reconstruction probe on unseen/authored games (no actions); Phase B = in-harness. Origin: Claude rebuilt ls20 from one screenshot (public game, contamination-flagged). See `E113_engine_world_model/hypothesis.md` | hypotheses |  | parked (until ~30 RHAE) |
| E114 | **Harness self-evolution** (Raven / HarnessBank, arXiv 2609.33439 §4): frozen backbone; an Evolver diagnoses trace failures and proposes patches to Mem / Plan / Cap / Act (window, prompts, tools, vision render, act policy), each with a declared activation predicate; a quality-diversity gene bank (edit category x pathology); candidates pass validity -> activation -> paired-gain screen (t >= 1.96 over n games, K_att=3) before full-train eval; the selected harness is frozen and scored on a sealed held-out split. Raven's multi-agent DAG part is out of scope. Budget ~645 game attempts at default settings, so cut first. See `E114_harness_self_evolution/hypothesis.md` | harness (all axes) |  | parked (until ~30 RHAE) |
| E115 | **Perception lab X1-X8** (2026-10-02): eight single-screenshot presentations on one shared screenshot->grid->objects front end (exploded inventory, surprise map, visual routines, look-alike threads, foresight, blueprint, role ledger, fusion). X8 fusion (blueprint map + measured relations + role guesses each with a one-action test) scored 25/28 vs old arc_eye 3/28 on ls20/r11l L1; held-out 23 games: grid 23/23, but role guesses ~50% and foresight 1/16. Next: X8 in the harness. See `E115_perception_lab/verdict.md` | perception |  | kept (X8) |
| E120 | **GLM-4.6V as the agent model** (owner, 2026-10-04): vision-first MoE (108B / 12B active, MIT, 128k) with native image tool calling, on the strength of the Gemini image result (raw grid lost; +image 12-24 actions). Stage 1 perception-only probe vs Flash-Next (5 questions per level-1 frame, owner's key; kill if GLM < +10%); stage 2 A/B on ls20/tr87/lp85 with the picture in the act tool result in both arms (kill if no RHAE win). No Kaggle dataset yet; HF has NVFP4 (vivek51) and AWQ (cyankiwi). See `E120_glm/hypothesis.md` | model |  | open |
| E122 | **Guardrails** (owner, 2026-10-16): catalogue of the 21 guardrails in the TAAF/Duck agent (turn yield 60 s, 1024-token tool output cap, turn ends on first move, 30 s sandboxed cell with import allowlist, terminal-state latch, action validation, context trim, watchdog) vs ours; candidate ports #19 latch and #11 output cap, measured from transcripts first. See `E122_guardrails/hypothesis.md` | budget |  | open |

## Status log — 2026-10-01

Recorded from the owner's report. These verdicts are qualitative (playthrough / harness observation), **not**
held-out RHAE numbers — Rule 2 is not yet satisfied for any of them. Re-measure on held-out once the submission
baseline exists.

- **Kept (implemented and observed to work):** E105 (oracle briefing), E02, E03, E01, E09 (step() lives in the
  REPL), E10 + E11 (rulebook + step() both in memory architecture), E12 (three clocks → questions / goals /
  lessons / timeline / hypothesis sub-memories), E15 (compaction), E16 (timeline sub-memory), E19, E24 (carryover
  helps), E26 / E43 / E44 (auto-refine), E34 (agent reads ARC actions better), E38 (lessons sub-memory),
  E39 (works, needs improvement — sub-memory is model-specific), E84 (NVFP4 best on RTX 6000).
- **Superseded:** E04, E05 (covered by E105 oracle/Gemini run), E06 (covered by E02), E14.
- **Killed:** E08 — a queryable transition database adds retrieval latency; JSON held in the REPL is faster.
  Follow-up opened as E106 (TOON instead of JSON).
- **Deprecated:** E25 (all games run together, so no cross-environment prior), E35, E42, E47, E48, E59, E60,
  E102 (no use), E104.

## Status log — 2026-10-04

Owner's decisions. Qualitative; no held-out RHAE yet.

- **Kept (implemented):** E07 (object permanence), E08 (re-opened: transitions implemented as JSON in the REPL,
  not a queryable database — supersedes the 2026-10-01 kill), E20 (annotation overlay), E83 (speculative decoding).
- **Killed (by decision, harness simplicity):** E13, E17, E23, E31, E40 (skills already do this job), E45 (adds
  harness complexity), E70 (adds time and complexity).

## Ranked shortlist

The ten to actually run, in order. Ranks in the table above refer to this list.

**Stage S (2026-09-25, before everything below):** E001 must reproduce the Duck at ≥ ~4.5 offline on the fast
serving path. Until it does, no agent experiment has a trustworthy baseline.

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
