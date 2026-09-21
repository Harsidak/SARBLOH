# SYSTEM V2 — "Coder-Champion" Architecture for ARC-AGI-3

**Target: beat the 0.30 champion on the official benchmark. Hard floor: 0.10.**
(A 0.10 target alone would be designing to lose — `my_agent original.py` already scores 0.30.
The floor exists as the abort criterion, not the ambition.)

---

## 0. What the research says (evidence base)

| Source | Approach | Score | Lesson |
|---|---|---|---|
| **Duck Harness** (Tufa Labs, **Milestone 1 winner**, Kaggle-legal, Qwen 3.6 27B FP8 — our exact model class) | Minimal coding harness: LLM in a Python REPL; observations as Python variables; image + text grid; oldest-message eviction | Best M1 submission | **Hand-crafted tools HURT the model; letting it improvise worked better.** Harness dictates cost; model capability dictates solvability. |
| **RGB Agent + Executable World Models** (arXiv 2605.05138 — already in `Docs/_ewm_paper.txt`) | LLM writes *executable* world models / policies as code; near-human action efficiency on preview | **58.12%** (community, unlimited compute) | The LLM's output must be **code that runs**, not prose that hints. |
| **Symbolica** | Recursive agentic orchestration through code | 36.08% (community) | Same family: agent-writes-code. |
| **Stochastic Goose** (preview winner) | CNN + RL predicting which actions change the frame | 12.58% preview → **0.25% full benchmark** | The overfit catastrophe. Tuning to known games collapses on unseen ones. |
| **Our own history** | Component-locked symbolic agent + oracle-LLM + GoalModel | 0.30 (original) → 0.15 (evolved) | Green harnesses ≠ score. Oracle-style LLM calls (38/game, ~26s each) steered nothing. Deleting the ugly-but-winning heuristics (`ChurnExploiter`) halved the score. |

**Convergent conclusion.** Every top result — at frontier scale and at our 27B scale — is an
**LLM-as-coder** system: the model reads the game as a programming problem and emits code
(a policy, a world model, a solver) that deterministic machinery then executes and verifies.
Our current agent uses the LLM as an **oracle** (describe the rule in JSON), which measured
zero steering effect. That is the single biggest architectural error to correct.

---

## 1. Design principles (non-negotiable, learned the hard way)

1. **The official score is the only fitness function.** No change ships unless it beats the
   current champion's official number. Harnesses are regression insurance, never evidence of
   improvement.
2. **Champion/challenger, always.** `my_agent original.py` (0.30) is the champion. V2 is a
   challenger built AROUND it, not a replacement of it. The champion's behavior is the
   guaranteed floor — which is how 0.10 is trivially secured and 0.30 is the number to beat.
3. **LLM writes code; symbolic machinery verifies and executes.** Never prose, never JSON
   "hypotheses". Code that fails to reproduce observed transitions is rejected automatically.
4. **No per-game logic anywhere.** Stochastic Goose's 12.58%→0.25% collapse is what tuning
   to known games buys. Dev/held-out split maintained; held-out is the reported number.
5. **RHAE math rules the budget.** `level_score = (human/ai)²` — at 2× human actions you keep
   19% of the score; at 10×, 0.8%. Exploration must be cheap, and once a solution is found it
   must be replayed near-optimally, not re-discovered.

---

## 2. System overview

```
                    ┌──────────────────────────────────────────────┐
                    │              ORCHESTRATOR (per game)          │
                    │  budget ledger · mode arbiter · score tracker │
                    └──────┬───────────────────────────┬───────────┘
                           │                           │
              ┌────────────▼───────────┐   ┌───────────▼────────────────┐
              │  LANE A — CHAMPION     │   │  LANE B — CODER (new)       │
              │  the 0.30 symbolic     │   │  Qwen-27B in a Python REPL  │
              │  agent, unmodified:    │   │  · sees grid as image+text  │
              │  ChurnExploiter,       │   │  · game state as variables  │
              │  planners, StateGraph, │   │  · writes candidate         │
              │  enumerative synth     │   │    policy/world-model CODE  │
              └────────────┬───────────┘   └───────────┬────────────────┘
                           │                           │
                           │               ┌───────────▼────────────────┐
                           │               │  SANDBOX EXECUTOR           │
                           │               │  · runs candidate against   │
                           │               │    RECORDED transitions     │
                           │               │  · exact replay check       │
                           │               │  · rejects on mismatch      │
                           │               └───────────┬────────────────┘
                           │                           │ verified code only
                    ┌──────▼───────────────────────────▼───────────┐
                    │            ACTION EMITTER                     │
                    │  one action/frame · replay cache for solved   │
                    │  levels · efficiency governor (RHAE-aware)    │
                    └───────────────────────────────────────────────┘
```

Two lanes, one arbiter. Lane A is the proven 0.30 agent, untouched. Lane B is the
research-backed coder loop. The orchestrator decides, per game and per level, which lane
drives — and the decision rule is designed so that V2 can never score below the champion.

---

## 3. Component specifications

### 3.1 Lane A — the Champion (frozen)

`my_agent original.py`, byte-preserved as a module. **No refactors, no cleanups.** Its ugly
heuristics are carrying measured score. The only permitted additions are the three verified
correctness fixes, each individually A/B-measured on the official benchmark before adoption:
bordered-background detection, HUD-masked verifier comparisons, `remap_colors`. Any of the
three that does not beat 0.30 stays out.

### 3.2 Lane B — the Coder loop (the new capability)

Modelled directly on Duck Harness (winner, same model class, same Kaggle constraints):

- **REPL environment.** A persistent Python namespace per game. Frames arrive as numpy
  arrays; helpers expose them as labeled images (grid rendered with coordinates) + compact
  text. The LLM interacts by emitting code cells, not chat.
- **Minimal toolset.** Research finding: hand-crafted tools *hurt*. Provide only: frame
  history access, an `act(action)` function, a scratch filesystem, and the recorded
  transition log. No bespoke perception APIs — let the model write its own.
- **The deliverable is code.** The model is prompted to produce, per level:
  1. a `world_step(grid, action) -> grid` candidate (executable world model), and/or
  2. a `policy(grid) -> action` candidate (executable policy),
  refined across REPL turns as new frames arrive.
- **Context discipline.** Oldest-eviction keeps the context small (Duck's approach); a
  running scratchpad file holds the model's own notes so evicted knowledge isn't lost.
- **Budget.** LLM latency at ~27B is ~25-30s/call (measured in our diagnostics). Cap coder
  turns per level (~8-12) and per game (~40); every turn must end in runnable code or it
  counts double against the cap.

### 3.3 Sandbox Executor & Verifier (the trust boundary)

The reason Lane B cannot poison the score the way oracle-LLM never could help it:

- Every candidate from Lane B runs against the **recorded transition log** (the same
  ground truth StateGraph already collects). A `world_step` must reproduce all logged
  `(prev, action, next)` HUD-masked; a `policy` is dry-run in imagination against the
  verified world model before a single scored action is spent.
- **Rejection is silent and total** — a failed candidate never touches the game. This is
  the executable-world-models insight: verification is mechanical because the output is code.
- Resource-capped execution (time/memory) so a pathological candidate cannot stall the run.

### 3.4 Orchestrator & Mode Arbiter (where the floor is guaranteed)

Per level, three phases:

1. **PROBE (Lane A drives).** The champion plays exactly as it does today; the transition
   log accumulates. Cost: zero relative to champion behavior.
2. **CODE (Lane B thinks in background).** While Lane A acts, the coder consumes the log
   and drafts candidates. Background thread, exactly like the current LLM synth path.
3. **EXECUTE (arbitration).** Lane B's verified policy drives ONLY when it (a) passed the
   replay check and (b) predicts level completion within a budget Lane A's current pace
   would not beat. One demotion rule: two consecutive wrong world-step predictions in live
   play → Lane B benched for the level, Lane A resumes. (EWMA-style demotion, kept from
   current architecture — it worked.)

**Floor argument.** If Lane B never produces verified code, the system IS the champion —
0.30 by construction, floor 0.10 satisfied with 3× margin. Lane B can only add score, never
subtract more than the bounded actions spent under a verified-then-falsified policy
(≤ 2 wrong predictions per level by the demotion rule).

### 3.5 Replay Cache & Efficiency Governor (the RHAE multiplier)

The overlooked score lever. RHAE squares the action ratio, so *how* a level is completed
matters as much as *whether*:

- When any lane completes a level, store the exact action sequence.
- On intra-game restarts (deaths on later levels), **replay** solved levels from cache at
  optimal length instead of re-solving.
- When Lane B holds a verified world model, plan the *shortest* action path in imagination
  (BFS over the model — free, unscored) before spending scored actions.

### 3.6 Measurement Rig (the missing discipline, now structural)

- `bench.sh`: runs the official scorer on dev and held-out splits, 2 seeds, and writes
  `scores/<git-hash>.json`. **No merge without this file showing ≥ champion.**
- Kill-switches for every subsystem (`ARC_NO_CODER`, existing `ARC_NO_*`) so every claim
  ("Lane B added X") is one paired run away from verification.
- The report per change: official score, per-game deltas, actions-per-completed-level.

---

## 4. Build order (each step officially benchmarked before the next)

| # | Step | Gate to proceed |
|---|---|---|
| 1 | Restore champion as `my_agent.py` verbatim; tag it; run official bench | reproduces ~0.30 |
| 2 | Graft the 3 correctness fixes one at a time, A/B each | each keeps ≥ 0.30 or is dropped |
| 3 | Build REPL + sandbox executor + verifier (no LLM wiring yet); unit-harness the verifier | replay check exact on synthetic logs |
| 4 | Wire Lane B coder loop with hard caps; arbitration rule as §3.4 | official ≥ champion; `ARC_NO_CODER` pair confirms Lane B ≥ 0 effect |
| 5 | Replay cache + shortest-path governor | actions-per-completed-level drops; official ≥ previous |
| 6 | Iterate on coder *prompting only* (EWM-paper style prompt optimization) | held-out score is the reported number, every time |

Steps 1-2 secure ≥0.30 (and thus the 0.10 floor) in week one. Steps 3-6 are the upside.

---

## 5. Risks, honestly

- **27B ceiling.** Duck Harness with the same model class solved some games consistently
  and others not at all — model capability, not harness, gated solvability. Lane B will not
  crack the abstract games; its job is to convert the *legible* ones near-optimally.
- **Latency.** ~25-30s/coder turn is real; caps in §3.2 keep worst-case wall-clock inside
  Kaggle's budget (~40 turns × 30s ≈ 20 min/game LLM time, comparable to today's usage that
  bought nothing — now it buys verified code).
- **The seduction of the substrate.** GoalModel/ProgressModel/Timeline stay OUT of V2 until
  the day a paired official run shows they add score. Nothing from the 0.15 branch returns
  without a number.

---

## 6. Sources

- Duck Harness (Tufa Labs, Milestone 1 winner): https://tufalabs.ai/research/duck-harness/
- Executable World Models: arXiv 2605.05138 (local: `Docs/_ewm_paper.txt`)
- ARC Prize 2026 Milestone 1 results: https://arcprize.org/blog/arc-prize-2026-milestone-1
- Stochastic Goose preview post-mortem: https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings
- ARC-AGI-3 technical report: https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf
- Local evidence: `Docs/goalmodel_architecture.md` §11 (paired-run methodology), diagnostic
  evals in this repo (oracle-LLM zero-steering, 0.30→0.15 regression).
