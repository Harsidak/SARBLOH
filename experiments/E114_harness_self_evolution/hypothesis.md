# E114: Harness self-evolution (Raven / HarnessBank method, frozen backbone)

- **Date opened:** 2026-10-02 (logged, NOT started; parked by the owner until ~30 RHAE)
- **Axis varied:** the whole harness (memory, planning, capability/perception, action policy), with the model frozen
- **Baseline:** `h0` = the best hand-tuned SARBLOH harness at the time the experiment starts, same backbone
- **Split:** evolution on a training split only; the report comes from a sealed held-out split that the search never
  sees

## Source

EverMind AI, *Raven: The Harness of Harnesses for Composable Agentic Intelligence*, arXiv 2609.33439 (27 Sep 2026),
Section 4 "Agent Harness Self-Evolution" (built on their earlier HarnessBank work, ref. [39] in the paper).

**Only Section 4 is in scope.** Raven's multi-agent orchestration (a Host Agent building a DAG of specialist agents)
does not fit ARC-AGI-3: one game, one action stream, scored per action. Nothing there benefits from fan-out.

**What the source reports** (Section 7.2, Fig. 14; frozen Qwen3.6-27B, held-out Pass@1, 3 attempts per task):
- AppWorld 41.3 → 56.7
- BrowseComp+ 16.9 → 30.8
- LiveCodeBench 58.1 → 71.8
- Omni-MATH 54.3 → 66.0
- Terminal-Bench-2 36.1 → 45.4
- GDPval 43.7 → 52.9
- SWE-bench Verified 47.4 → 52.6, which did NOT pass their paired-gain criterion

The authors also state that evolved gains "can be unstable, transfer only partially to held-out tasks, and depend
strongly on the model that executes the harness" (Section 4.3, citing HarnessDev).

## Mechanism (the paper's method, mapped to SARBLOH)

**1. A harness is the model's policy, `h = (Mem, Plan, Cap, Act; θ)`.** The model weights stay fixed; only θ
(prompts, instructions, configuration, code in the editable modules) changes.

| Raven role | SARBLOH editable locations (`X_edit`) |
| --- | --- |
| Mem: builds the working context, decides what to drop | sliding window size, compaction rule, sub-memory injection and caps, wipe-vs-keep on reset |
| Plan: turn guidance | system/ARC prompts (`prompts.py`), continuation and planner nudges |
| Cap: tools exposed | tool set and schemas (`tools.py`), vision render variant, `intuition.py` views |
| Act: response handling, completion | `act` batch cap, halt-on-misprediction, stagnation supervisor thresholds |

**2. Fixed kernel (`K_fixed`, never edited):** `ArcHost` (action counting, the budget, scoring), the
`arc.transitions()` record, `trace.py` instrumentation, the evaluation protocol. Every candidate must leave these
byte-identical. This is what keeps evolution from "improving" the score by changing how it is measured.

**3. Diagnosis-driven proposals.**
- An Evolver Agent reads the parent's failure traces (`trace.py` reports, transcripts) and writes a diagnosis:
  symptom, trace evidence, edit location, expected behaviour change.
- It then proposes a patch with a semantic descriptor and a **declared activation predicate**: the logged event
  proving the new mechanism actually ran. Example: "the patched reset rule fired on at least one GAME_OVER".
- The Evolver may be a different, stronger model than the frozen Task Agent. It never changes the Task Agent's
  weights.

**4. The gene bank.** The bank is a grid of cells (edit category × diagnosed pathology), holding the best harness for
each cell. That is quality-diversity search, so a fix for a narrow failure is kept even when it isn't the global best.
- Edit categories: prompt, knowledge, runtime, config.
- Pathology labels (to be drawn from our autopsies): blind action loops, repeated (state, action) pairs, counter
  overrun into GAME_OVER, understanding lost at compaction, `act` format rejections, premature goal commitment,
  HUD mistaken for a puzzle piece.
- The parent each round is the best training-utility harness among `h0` and the bank. Proposals either synthesize a
  new intervention or recombine changes from bank donors.

**5. Gated screening.** For each round, sample `n` training games without replacement. A candidate must pass, in
order:
1. **validity:** a complete, protocol-valid log;
2. **activation:** its predicate fired at least once;
3. **paired gain:** per game, `gamma_i = score(candidate) - score(parent)`, averaged over `K_att` attempts. The
   statistic `t = mean(gamma) / (sd(gamma)/sqrt(n))` must exceed **1.96**.

The first `N_keep` survivors get a full-training evaluation and compete for their bank cell. The search stops after
`R_round` rounds, or after `P` rounds with no cell change.

**6. Final test.** Freeze the selected harness. Report `Util(selected; held-out) - Util(h0; held-out)` under the same
protocol. Held-out results never feed back into the search.

## Why it should work here

Most of our gains so far came from hand edits found by reading traces: dedicated tools, required `expect`, reset
reasons, quote stripping. That is diagnosis-driven harness evolution done by a human, one edit per day. The method
automates exactly that loop and adds the controls a human skips: a declared activation check and a paired
significance test against the parent. It only works once there is gradient. Below roughly 30 RHAE most games score 0,
every candidate ties with the parent, and paired gains are all zero (owner decision, 2026-10-02).

## Prediction (committed before the run)

> The selected harness beats `h0` on held-out mean RHAE by **at least +3 points**, AND at least one surviving edit is
> in a pathology cell that hand-tuning never touched. UNCONFIRMED: no ARC-AGI-3 evidence exists. The +3 is a
> calculated guess scaled from the source's 5-15 point Pass@1 gains on other benchmarks, discounted for ARC's sparser
> score.

## Kill criterion (binding)

> Kill if, after the full search budget, the selected harness shows < +1 RHAE on held-out vs `h0`, OR no candidate
> passes paired screening in 3 consecutive rounds (stagnation with zero survivors).

## Measurement

- **Score per task:** per-game RHAE normalised to [0, 1], `K_att = 3` attempts per game (same as the source).
- **Split:** the public games split into training (about 15) and sealed held-out (about 10), plus owner-authored /
  synthetic games for the held-out side if available. Fixed before the first round; recorded in `config.yaml`.
  Public games are contamination-prone for any frontier Evolver: record which model proposed each edit.
- **Budget.** The cost in scored game attempts is `K_att * [(1 + R_round * N_keep) * |D_tr| + R_round * J * n]`.
  With `|D_tr|=15, K_att=3, R_round=4, N_keep=2, J=4, n=5`, that is `3 * [(1+8)*15 + 4*4*5] = 3 * [135+80] = 645`
  game attempts. At our throughput this is days of GPU time: the search settings must be cut to fit the Kaggle/GPU
  budget before starting. Smaller `J` and `n` are the first levers.
- **Logs:**
  - each candidate's diagnosis, patch, descriptor cell and activation predicate;
  - screening statistic and pass/fail reason;
  - the bank state per round;
  - Evolver tokens and wall clock, reported separately from deployment cost (the source's convention).
- **Report:**
  - held-out mean RHAE with variance;
  - the list of surviving edits with their pathology cells;
  - the source's caveat tested directly: does the gain survive on the held-out split?

## Practical constraints

- **Determinism of the protocol:** fixed backbone version, sampling settings, action budget, wall clock per game and
  infrastructure retry rule across all candidates. Utilities measured under different environments are not compared
  (the source's rule).
- **Infrastructure failures are not agent failures.** A vLLM/SGLang crash (16% of calls lost on 2026-09-30) must be
  retried under the declared rule or the log marked invalid. It must never count as a candidate's low score.
- **Every editable module must be switchable from config,** as BACKLOG already requires. The Evolver edits
  config/prompt/runtime files, not the kernel.

## Dependencies and timing

Parked by owner decision until the main harness reaches about 30 RHAE. Needs: a stable serving stack (E109/E112),
`trace.py` pathology reports, and a fixed train/held-out split. Related: E113 (engine-grounded world model, also
parked), E107/E108 (alignment pipelines; evolution changes the harness, alignment changes the weights, and the two
are complementary and must be ablated separately).
