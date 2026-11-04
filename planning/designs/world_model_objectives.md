# Objective functions for every world model in the agent

**Date:** 2026-08-01. **Status:** audit + specification. Every claim below is tagged
either MEASURED (backed by a run recorded in this repo) or SPEC (a rule now enforced
in code).

## Why this document exists

LeCun's caveat about world models is the operative one here: a world model is only
useful if you can say what it is being *optimised for*. Without an objective, a
predictor optimises nothing — it will be accurate about things that do not matter,
confident about things it has never been graded on, and it will keep running because
nothing is empowered to switch it off.

This agent has **six** things that are world models in the relevant sense. Until now
exactly two of them had an objective wired to a decision. The rest were either graded
on a number nothing consumed, or not graded at all. That is not a theoretical
complaint — it is the direct, measured explanation for the two largest wastes of
budget in the agent.

## The four objectives

Any world model in this agent must answer all four. A model missing any one of them
is a liability, not an asset.

| | Objective | The question | What it gates |
|---|---|---|---|
| **O1** | FIDELITY | Is the model TRUE? | Whether the model may be used at all |
| **O2** | COST | What does acting on it SPEND? | Which plan is chosen among equals |
| **O3** | PROGRESS | What is a GOOD state? | What the model is aimed at |
| **O4** | SAFETY | What must NEVER happen? | A hard constraint, never a weighted term |

Two rules make these robust rather than decorative:

1. **O1 must be falsifiable and must have teeth.** "Accuracy 0.7" is not an objective;
   "below this threshold the model is deleted and its plans are discarded" is. A model
   that cannot be switched off by its own error signal is not being graded.
2. **O4 is a constraint, not a term.** Nothing may buy its way past a score wipe or a
   known-lethal action at any exchange rate.

O2 deserves emphasis in *this* competition. RHAE is
`min(1.15, human_actions / ai_actions) ** 2` — it **squares** the action ratio, so
10× human actions scores 1%. Any world model whose only objective is accuracy will
happily spend scored actions to become more accurate. That is the single failure
pattern behind the agent's measured ≈0.0001 RHAE.

---

## The audit

### 1. `StateGraph` (COMPONENT 5.8) — exact observed transitions

- **O1** — exact by construction (deterministic games, observed edges only). The real
  risk is *staleness*, handled by HUD masking via change-gap periodicity.
- **O2** — **absent.** Explores "nearest untested" by hop count, which is a proxy for
  cost but is not priced against alternatives.
- **O3** — the untested frontier; each unknown `(state, action)` is one unit.
- **O4** — `DeadActionTracker`.

MEASURED: 23.7% of dev actions, 54.0% new-state rate, **all 3 reward events**. This is
the agent's most productive component and the one whose objectives are most nearly
complete. Its missing O2 is a refinement, not a defect.

### 2. `PatchWorldModel` (COMPONENT 5.9) — learned local 5×5 rules

- **O1** — present and correctly shaped: `MIN_SEEN = 2` occurrences before a patch
  counts as known, `WITNESS_MIN = 2` all-ineffective occurrences to certify a blocker,
  and an **exhaustion fallback** that re-offers model-pruned pairs for real testing.
  That fallback is what makes the model safe: a wrong prediction can only ever
  *reorder* work, never permanently hide a solution.
- **O2** — implicit and correct: imagination is unscored, so pruning a certified no-op
  is a pure saving.
- **O3/O4** — inherited from StateGraph.

This is the template the rest of the agent should copy.

### 3. `WorldModelManager` + `GridDSL` + `EnumerativeSynthesizer` — synthesised programs

- **O1** — an EWMA `accuracy` with `DEMOTE_ACCURACY = 0.35`. It has teeth (demote +
  ban rules) but it grades **one-step pixel prediction**, which is not what any
  decision needs.
- **O2** — **absent.** Synthesis runs every step regardless of what it costs.
- **O3** — **absent.** The model is aimed at reproducing frames, not at reaching
  anything.
- **O4** — n/a.

MEASURED and decisive: fixing the HUD-mask asymmetry cut demotions from 57 to 4 — a
14× improvement in the O1 number — **with every level score unchanged**. A model whose
fidelity can improve 14× while changing no outcome is being graded on a quantity
nothing consumes. Separately, per-action synthesis cost ~1s/action on tu93 while
completing nothing. This is the clearest case in the codebase of LeCun's point.

### 4. `MCTSPlanner` (COMPONENT 4) — planning through model 3

- **O1** — **absent.** MCTS plans through `WorldModelManager.active_model` with no
  fidelity precondition of its own. `ExploitGate` (5 consecutive hits to enter, rolling
  mean < 0.5 to exit) gates *whether* to call MCTS, but not whether the model it
  rolls out is trustworthy at depth. One-step accuracy does not certify a 60-step
  rollout.
- **O2** — **absent.** No action-cost term at all.
- **O3** — novelty bonus, not task progress.

MEASURED: 9.2% of dev actions, **50.5% of all think time (348s of 690s), and zero
rewards**. A/B removal: `levels 1 -> 1`, wall 2020s → 1719s, think 498.9s → 221.9s.
**Resolution: default off, opt-in via `ARC_MCTS=1`.** A world model with no fidelity
gate and no cost objective explored a fiction for half the compute budget and returned
nothing. This is the prediction the framework makes, and it is what the measurement
found.

### 5. `ClickPlanner._G` (COMPONENT 5.6) — the click world model

Previously: `_G` was **built and never used for movement**. Every frontier probe was
priced as a fresh replay from the root — `["RESET", *path, button]` — so discovering
one edge cost `depth + 2` **scored** actions even when its parent was the board already
on screen.

SPEC, now enforced in code (see the objective block on the class):

- **O1** — every navigation plan carries `_expect`, the board it predicts at each step.
  A mismatch deletes the offending edge, abandons the plan, and re-plans. The pending
  probe is **not** recorded, because attributing a result to a parent we are
  demonstrably not standing on is silent graph corruption — strictly worse than a
  wasted action.
- **O2** — every probe is priced in scored actions and the cheapest route wins:
  *navigate* `[*path_from_here, button]` versus *teleport*
  `["RESET", *path_from_root, button]`. Planning through `_G` is free; only emission
  is paid for.
- **O3** — the frontier is the value function: an untried `(state, button)` pair is
  worth one unit, a recorded one is worth zero and is never re-spent.
- **O4** — `_dead_edges` are never planned through, and a RESET is never emitted while
  the engine's action counter reads 0 (that is a `full_reset`: every level already won
  is destroyed).

MEASURED (`test_clickplanner.py` case 9, a deterministic 2-button toy with a 3×3 state
lattice): **27 scored actions versus a teleport-only lower bound of 72**, with
`nav=13 teleport=5` and 45 actions avoided. Real-game effect is recorded in
`routing_architecture.md`.

### 6. `GoalModel` (COMPONENT 5.12)

- **O1** — alpha/beta counts per predicate.
- **O3** — this component *is* O3 for the rest of the agent: it is the only thing that
  tries to answer "what is a good state?" from data.

MEASURED: the loop closes (beta 242, retired 24) but **alpha is structurally
unreachable without a level-up** — the positive signal only ever arrives on a level
completion, which is exactly the event the agent cannot produce. The objective is
correctly specified and starved of data.

---

## What the audit concludes

Ranked by measured cost, the agent's world-model problems are not modelling problems:

1. **MCTS** — no O1, no O2 → 50.5% of think time, 0 rewards. *Resolved: opt-in.*
2. **DSL synthesis** — O1 grades a quantity no decision consumes → 14× fidelity gain,
   zero outcome change. *Open.*
3. **ClickPlanner** — had no O2 → paid `depth + 2` scored actions per edge.
   *Resolved: navigate-not-teleport, measured 2.7× cheaper on the toy.*
4. **GoalModel** — objective correct, data starved. *Open, and blocked on 1–3.*

The pattern is one sentence: **every component that spent budget without an O2 term
spent it badly, and every component whose O1 was not wired to a decision improved its
O1 without improving anything else.** Accuracy was never the binding constraint.

## The rule going forward

No world model is added to this agent, and no existing one is extended, without
stating its four objectives and a measurement that would falsify it. A model that
cannot be switched off by its own error signal does not ship.

---

## Addendum, 2026-08-02 — auditing O1 found two claims with no teeth

Both were found by measurement (`scratchpad/diag_o1.py`, lp85 + sb26, seed 0), and one
hypothesis that *sounded* right died on contact with the data. Recorded because the dead
one is as useful as the live ones.

**DEAD: "ClickPlanner hashes the raw grid while StateGraph masks the HUD, so a ticking
counter inflates the state space."** lp85 episode 1 showed 251 distinct states from 256
edges — a graph where nothing is ever revisited, exactly the signature a HUD counter
produces. It is not one. Masking the 64 most-variable cells changes the distinct-state
count by **zero** (251 → 251); only ~200 of 4096 cells vary at all, and the count does
not collapse until essentially every varying cell is masked. The states are genuinely
distinct. Corroborated by `falsified=0` across 256 navigations: a ticking counter would
break replay predictions constantly. **The unmasked hash is not a defect.** Do not
"fix" it.

**LIVE 1 — a resource limit was being filed as a proof.** `_build_next_plan` gives up
for three reasons and the caller read all three as *"this button set is refuted"*:

| exit | what it means | may it refute? |
|---|---|---|
| frontier emptied | every reachable state had every button tried | **yes — a claim about the GAME** |
| `len(_dist) > MAX_SEARCH_STATES` | we ran out of room to remember states | no — our memory budget |
| every frontier node > `MAX_REPLAY` | we cannot afford to reach the probe | no — our action budget |

lp85 episode 1 reached 251 states against a cap of 250. That cap trip banned a 3-button
set and — via the subset rule, which is *correct for real proofs* — also banned the
2-button subset that had been working. A falsification you can manufacture by running
out of memory is the textbook O1 failure: **not falsifiable with teeth.** Fixed by
splitting the record into `_exhausted_sets` (proof; propagates to subsets, because a
subset can only reach boards the superset already reached) and `_suspended_sets` (budget
stop; binds the identical set only, because a subset spans a *smaller* space and may
exhaust honestly where the superset could not).

**LIVE 2 — the planner disagreed with itself about button identity.**
`_detect_buttons` merges reactive coords within `BUTTON_CLUSTER_DIST = 10` px into one
physical button; `_abstraction_is_live` compared button sets by **exact pixel**. On sb26
that let detector jitter manufacture a fresh "abstraction" eight times over at
distances of 2, 2, 3, 4, 6, 7 and 7 px — the same two buttons, each re-detected slightly
off, each buying another search episode. Fixed by comparing sets under the detector's
own equivalence. A component must not use two different notions of identity for the
same object.

---

## O3 is the next work, and it is currently a coverage objective wearing a progress label

Section 5 above states ClickPlanner's O3 in the agent's own words: *"the frontier is the
value function: an untried `(state, button)` pair is worth one unit."* That is a
**coverage** objective. Every untried pair scores the same, so all 251 of lp85's states
are equally attractive and the search has no opinion about which board is closer to
winning. This is the LeCun failure exactly: a faithful (O1), cheap (O2), safe (O4)
simulator with nothing to optimise through it.

It is also where the score is. RHAE's denominator spans every level, so on lp85 (8
levels, Σi = 36) the *entire* remaining efficiency headroom on level 1 is +3.33 points
while one more completed level is +5.56 and the rest of the game is +94.4. And 23 of 25
games complete zero levels, where efficiency work is worth exactly nothing.

**SPEC — a progress score over states, used to ORDER the frontier and never to prune it.**

- **O3 signal, grounded.** When a level *is* completed, its trajectory is ground truth.
  Extract cheap grid features (per-colour counts, connected-component counts, occupied
  bounding boxes, symmetry) and record which moved monotonically toward the win. Score
  candidate states by agreement with those directions. This is the only channel that can
  transfer across levels, which the RHAE denominator makes mandatory.
- **O3 signal, ungrounded** — the 23-game case, where no win has ever been observed. Two
  generic priors: *irreversibility depth* (states no observed action returns from —
  puzzle progress is typically one-way) and *consolidation* (fewer components, larger
  uniform regions). Priors, explicitly, not truths.
- **O1 ON O3 (the teeth).** Every completed trajectory is a backtest: replay it and check
  the progress score was monotone non-decreasing along it. A prior that voted wrong is
  down-weighted, and below a threshold it is switched off. Per the rule above, a progress
  model that cannot be disabled by its own error signal does not ship.
- **O4.** O3 reorders the frontier. It never prunes a state, never marks an edge dead,
  never gates a plan. A wrong O3 costs ordering; it cannot make a reachable level
  unreachable. (Same constraint the GoalModel architecture puts on its re-ranker.)
### Built and measured, 2026-08-02 — the model works, the SIGNAL is the wall

`ProgressModel` (COMPONENT 5.55) ships against the spec below: 20 features, fitted at
`new_level()`, ordering the frontier, retiring itself on a failed backtest, strict no-op
before the first win. `test_progress.py` 67 checks, 0 fails.

What the first real measurement says (lp85, seed 0, `CP_DEBUG=1`):

```
[CP] O3 fit=True src=graph graph=5 walk=95 fits=1 ready=True features_with_a_direction=3
```

**Only 3 of 20 features earn a direction, and the score does not move (`honest` 0.3352,
unchanged).** The cause is the grounding trajectory, not the model. lp85's level-1 trace is
95 frames of which ~89 are *coverage clicks* — a random walk that happened to end well. A
random walk has no monotone structure, and `MONOTONE_MIN = 0.70` correctly refuses to
invent one: fitting on it gave **2** features. Switching to `_graph_trace()` — the shortest
button path from the root to the board the level was won from, every step deliberate —
gives 5 clean frames and **3** features.

The honest reading: O3 is now specified, wired, falsifiable and safe, and it is **starved**
in exactly the way GoalModel's alpha channel was. It can only learn from wins, and wins are
what the agent does not have. Tuning `MONOTONE_MIN` down would manufacture features out of
noise and is the one change explicitly ruled out — that threshold is the only thing making
the score mean anything. The work is on the INPUT: longer and cleaner winning trajectories,
or a channel that needs no win at all (the ungrounded prior, written but gated behind
`ARC_O3_PRIOR=1` and deliberately unmeasured until it is measured *alone*).

- **O2 layering.** `_build_next_plan` currently keys `best` on `(cost, queue_order)` —
  cost PRIMARY. That contradicts the table at the top of this document, where O2 decides
  "which plan among **equals**" and O3 decides what the model is aimed at. The key
  becomes `(-progress_bucket, cost, queue_order)`: progress first, cost breaking ties
  *within* a bucket. Bucketed and not raw, so the planner cannot be talked into walking
  20 actions for a rounding difference in progress.

### Addendum, 2026-08-03 — the starvation, measured; and O3 was never ordering anything

The user's diagnosis was that O3's starvation and `GoalModel`'s alpha starvation are the
same failure. They are: `ProgressModel.fit()` has exactly one call site (inside
`ClickPlanner.new_level()`) and `GoalModel`'s alpha is written only by `_settle_pursuit`.
Both are gated on a level completion. `FrameData` carries no score field, so completion
and `GAME_OVER` are the only extrinsic signals that exist.

**The measurement.** A default-off dump (`ARC_O3_DUMP=<dir>`, `ARC_O3_DUMP_TAG=<gid>`)
persists every episode's raw boards labelled by how it ended; `scratchpad/diag_o3.py` and
`scratchpad/diag_b2.py` read them offline. Three tier-1 click games, seed 0, 36 episodes.
All three scored identically to baseline with the dump on, so the instrumentation is a
verified no-op.

| game | wins | distinct deaths (graph/walk) | features with a direction | confounded with elapsed time | discriminative |
|---|---|---|---|---|---|
| lp85 | 1 | 6 / 11 | 3 (graph), 2 (walk) | 2 @ 100% of deaths, both bases | 1 (graph), **0** (walk) |
| r11l | 0 | 4 / 19 | — | — | death-consensus 4 / **1** of 20 |
| sb26 | 0 | 3 / 4 | — | — | death-consensus 0 / 2 of 20 |

Two findings, one of which is bigger than the work it was meant to justify.

**1. Most of what O3 learned was elapsed time.** Of the 3 features lp85's win taught, 2
move the *same direction* on 100% of the distinct death paths. On the walk basis — the
fallback `fit()` uses whenever the graph cannot reconstruct a path — 2 of 2 do, i.e. the
entire learned direction was time. With a single win, "rose because we won" and "rose
because the board advanced" are not separable; a control group is what separates them.
This is A1, and it is implemented: `fit_death()` absorbs `GAME_OVER` trajectories as a
control that can only ever SUBTRACT (see the method docstring for why the asymmetry is
what makes it safe to ship on evidence from failures).

**2. `bucket()` made O3 a no-op — it has never ordered anything.** Across 199 real lp85
level-2 boards the scores span **0.036**, while one absolute bucket step is
`1/BUCKETS = 0.125`. Every state landed in the *same* bucket, so `-prog` was a constant
in the frontier's sort key and O2 decided everything — O3 had silently degenerated back
into the coverage objective it was written to replace. The fault was measuring a small
*relative* difference against a fixed *absolute* grid; a progress score has no natural
unit. `buckets()` now ranks the whole frontier against the spread actually present in it.
Bucketing is kept — its purpose (O2 breaks ties among comparable states) survives intact.

This is also why A1 and C1 could not show up in a score on their own: both were feeding a
quantizer that discarded their output.

**3. B2 (a HUD progress meter, grounded with zero wins) is REFUTED — not built.** The
first test looked for cells monotone in *colour value*, found none, and was itself wrong:
a rendered counter is a digit glyph, and 2→3→4 is not ordered in colour space. Retested
order-theoretically ("never revisits a value"), lp85 yields 10 non-periodic candidate
cells at rows 43–46. They are constant for frames 0–90 of the 95-frame win episode and
change only at frames 91/93/94 — the level-up animation, a row of level-marker icons. It
reports the win *after* the win: it would backtest perfectly and guide nothing. r11l and
sb26 yield zero candidates. No agent code was written for B2.

**A1-as-avoidance is separately refuted.** The tempting stronger form — use `−death` as a
progress direction on the 23 games that never win — does not hold: across 19 distinct
r11l deaths only 1 of 20 features shares a direction, and 0 of 10 object features do.
Deaths agree about very little, so there is no avoidance direction to ride. Only the
control form shipped.
