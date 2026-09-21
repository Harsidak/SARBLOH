# GoalModel (Component 5.12) — Architecture for ARC-AGI-3

*The sixth lock. Numbered 5.12 in `my_agent.py` because `COMPONENT 6` is already
`ACT & MAIN AGENT`; "Component 6" in conversation means the sixth component to get a
permanent isolation harness, after Eyes, GridDSL, Synth, Livelock and StateGraph.*

**Thesis.** Every component built so far answers a question about **what is** — Eyes: what is
on the screen; GridDSL/Synth: how the screen changes; Livelock: am I repeating myself;
StateGraph: which world-state is this and what have I not tried here. **Not one of them
answers "which state is better."** GoalModel is that single missing relation: it is an
**order over world-states**, learned online from the record, carrying a calibrated
credibility, and exposed to the rest of the agent as a **re-ranking** — never as a gate.
It never chooses an action, never names a pixel, and never removes anything from the
exploration frontier. Its entire safety argument is that a *wrong* goal can only permute
an order that was previously arbitrary.

---

## 1. Why this component is load-bearing

### 1.1 The observed failure has exactly two shapes, and both are "no objective"

Full 25-game diagnostic, seed 0, 1500 actions/game (`scratchpad/diag/sac_*`):

| shape | games | masked-noop % | re-spend % | world model promoted | reading |
|---|---|---|---|---|---|
| **Churn** | tn36, vc33, s5i5, r11l, su15, dc22 | 0.1 – 9 | **78 – 88** | 0/0 on five of six | almost every action *changes the world*, and the agent keeps re-entering states it has already been in |
| **Inert** | lf52, sb26, lp85, wa30, g50t, sp80, m0r0, sc25, tu93 | **19 – 93** | 23 – 79 | healthy | the world model predicts correctly, and a fifth to a half of chosen actions still do nothing |

Churn is not blindness — StateGraph's mask is validated and the hashes are exact. It is
motion without a destination. Inert is not a broken model — `verify` sits at 1.00. It is
the difference between *"what does this action do"* and *"which action gets me closer to
winning"*: only the first question is currently asked anywhere in the system.

Totals: **4 levels completed across the 17 dev games, 0 across the 8 held-out games.**

### 1.2 RHAE makes "explore until you stumble onto the win" mathematically unavailable

`level_score = min(1.15, human / ai) ** 2`. The square is the whole story:

| agent actions vs human | level score | fraction of the 1.3225 cap |
|---|---|---|
| 0.87× (the cap) | 1.3225 | 100% |
| 1× | 1.00 | 76% |
| **2×** | **0.25** | **19%** |
| 4× | 0.0625 | 4.7% |
| 10× | 0.01 | 0.8% |
| 65× (lp85 L1 today: 161 vs 33) | 0.0002 | 0.02% |

A human solves **all 25 public games in 14,799 actions total** — a mean of 592 per game
against a 1500-action budget. **The budget was never the binding constraint; the ratio is.**
An agent that plays *twice as carefully as a human* still forfeits 81% of the score. There
is no action count at which information-gathering-by-doing is affordable, which is why every
scored action must be spent *on* a goal rather than *looking for* one.

### 1.3 The denominator: level-1-only completions are worth almost nothing

`game_score = Σ_{completed i} level_score_i · i / Σ_{i=1..win_levels} i`. The weight is the
1-indexed level number and **the denominator spans every level, including the ones never
reached**. So:

- tu93 (`win_levels=9`, denominator 45): the two levels it reaches today cap the game at
  `(1.3225·1 + 1.3225·2)/45 = 0.088` **even under flawless play**.
- lp85 (8 levels, denominator 36): a perfect level 1 is worth **3.7%**.
- The score lives in levels 5–9, which are *layouts the agent has never seen*.

This is the argument that settles the component's design. Frontier exploration cannot
transfer across a level boundary — every layout is new, and StateGraph is correctly
discarded per level. **A learned win-condition is the only object in the system that can
survive a level boundary**, because the *rules* persist even though the *layout* does not.
Any architecture that cannot generalise from level 1 to level 6 is capped near zero by the
denominator regardless of how well it plays.

### 1.4 The seed already exists and is currently used only defensively

`StateGraph._out_chg` is a per-cell count of changes at score and terminal moments, added
for the sacred-cell guard. It answers "which cells may I never mask." **Read positively it
answers "which cells decide the score"** — a candidate-generator for progress features that
costs nothing extra to collect and is already validated by `test_stategraph.py`.

Symmetrically: `_cellmask` masks cells whose change *rate* exceeds `MASK_RATE`. A **counter**
— keys collected, targets remaining, a level's move tally — changes only on events, so its
rate is low and it is **never masked**. It is already inside the state hash. The graph simply
treats `counter=2` and `counter=3` as two unrelated nodes. The information the game is
volunteering about its own objective is being received and discarded. Imposing an *order*
on it is the whole job.

---

## 2. ARC-AGI-3 constraints that dictate the design

| Constraint | Design consequence |
|---|---|
| `level_score` squares the action ratio | Goals are pursued **only through plans verified unscored inside the world model**. Reasoning is free; acting is not. A hypothesis may never "try itself out" in the environment. |
| Denominator spans all `win_levels`; weight = level index | Features are **game-scoped and persist across levels**; only their *bindings* (which colour, which cell) are level-scoped. Two-tier lifecycle is mandatory, not an optimisation. |
| Level 1 begins with **zero** reward observations | The model must produce a usable order **before it has ever seen a win**, from terminals and structure alone — and must be a **strict no-op** when even that evidence is absent. |
| Objective is hidden and never stated | The vocabulary of progress features is **fixed and game-general**. No per-game branch, ever. Which feature is right is *learned*; the set of candidates is not. |
| Human baselines are an offline eval artifact | The pursuit budget can **never** reference `baseline_steps`. Budget is derived from the level's own history (actions since the last confirmed potential gain). |
| Engine is deterministic | A hypothesis refuted by the record is refuted **forever**. Backtesting the Timeline is exact, not statistical. |
| RESET counts against the budget; the level layout is unchanged after it | Deaths are cheap, plentiful, and arrive *before* any win. Terminal evidence is the first evidence channel that ever fires and must be first-class. |
| 22 of 25 games complete zero levels | The component must **degrade to today's behaviour** with an empty pool. Shipping it cannot cost anything. |
| Kaggle: in-process LLM, budgeted, background | GoalModel is **entirely local and LLM-free**. The theorizer may later *propose into* the pool; proposals get no credibility discount and no bonus — identical certification path. |

---

## 3. Data model

Four objects. Everything else is derived and rebuildable.

**a) `Readout`** — a total function `grid → scalar`, drawn from a **fixed, closed vocabulary**,
parameterised by concrete arguments discovered at runtime. Total: never raises, returns a
sentinel when out of scope (the DSL discipline already in force). The vocabulary:

| readout | value | what it can express |
|---|---|---|
| `count(c)` | cells of colour `c` | items remaining / collected |
| `objects(class k)` | connected components of shape-class `k` | enemies alive, blocks placed |
| `exists(c)` | 0/1 | a door opened, a key consumed |
| `contact(a, b)` | adjacent cell-pairs between colours | avatar touching goal |
| `dist(a, b)` | Manhattan distance of centroids | walk-to-goal (today's *entire* goal notion, demoted to one member of a pool) |
| `align(a, b)` | shared rows + shared cols | line-up / aiming puzzles |
| `regions(c)` | connected components of `c` | merging, splitting, filling |
| `cell(r, c)` | raw value at a low-rate cell | **counters and score displays** (§1.4) |
| `agree(panel_i, panel_j)` | matching cells between two detected panels | **"make this look like that"** — the churn games' hypothesis |

`agree` is the speculative one. Eyes already resolves multi-panel frames correctly (the
"phantom objects are real panels" finding), so the input exists; whether it explains the
churn games is a claim the harness must settle, not an assumption (§9 step 5).

**b) `Feature`** — a `Readout` plus a **direction** (`+1` progress increases it, `-1`
decreases it), plus `Beta(α, β)` credibility, plus its **evidence channel** (§4.1), plus its
**description length** in the MDL sense (argument count + constant sizes).

**c) `Pool`** — at most `MAX_FEATURES` live features, ordered by posterior mean credibility,
deduplicated by **observational equivalence**: two features whose value vectors over the
entire Timeline are identical are the *same hypothesis*, and the shorter description
survives. This is what keeps a full re-certification affordable as the record grows.

**d) The record** — GoalModel **owns no history**. It reads `Timeline` (append-only, survives
GAME_OVER, segmented by level, never cleared) and `StateGraph` (`adj`, `sacred_cells()`,
per-action change statistics). This is deliberate: the Timeline is already the certified
ground truth every model is backtested against, and a second copy could disagree with it.

**Invariant:** every credibility value is reproducible by replaying the Timeline through
`certify()`. Nothing in the pool is ever updated in a way replay cannot reproduce — the same
rule that keeps StateGraph exact under an online-learned mask.

---

## 4. Mechanism

### 4.1 Three evidence channels, in strict priority

Priority is a safety ordering, taken from CDE's clipping law (an intrinsic bonus must never
be able to reverse the sign of the extrinsic signal, arXiv:2509.09675):

**Channel A — CONFIRMED (extrinsic).** Score events and `LEVEL_COMPLETE` frames. `_out_chg`
and `sacred_cells()` narrow the candidate cells to those the engine itself implicated, so
the candidate set is small before a single readout is evaluated. A feature is *confirmed*
when its value is systematically extreme on the frames the score moved from, and that
extremity is **disproportionate** — not explained by elapsed time and not explained by the
readout's own base rate (§4.1a). Confirmed features outrank all others absolutely.

> **Implementation note (found by measurement, not by design).** Only the **pre-outcome**
> frame may be read. The engine reports a score change on the frame *after* the whole
> layout has been swapped, so on a level-up the post-frame is a foreign board and its
> readout value is a layout artifact rather than progress. The informative side is the
> state the win was achieved *from*.

### 4.1a The base-rate margin — the third analogue of the sacred-cell rule

"Systematically extreme at the outcome" is **not** evidence on its own. `exists(goal_colour)`
is at its maximum on every frame the agent died on — and on 99% of every other frame too,
because the goal is nearly always on screen. Judged against a nominal midpoint that reads
as decisive.

The first 25-game run built without this term adopted **11–19 hypotheses per game**, every
one of them still at the untested prior, and the exploration frontier was being steered by
base rates. The fix is the rule StateGraph already uses for sacred cells: compare the rate
**at the outcome** against the rate **overall** and require a margin. A pure base-rate
readout scores margin 0 by construction, exactly as a pure clock scores sacred-margin 0.

This is now a design law of its own, and it applies to **both** channels:

> **Extremity is only evidence when it is disproportionate.** Every adoption test in this
> component compares a marked-frame statistic against the same statistic over the whole
> record, and requires a margin. There are no absolute thresholds anywhere.

**Channel B — STRUCTURAL (intrinsic).** Available before any win, which is the only regime
22 of 25 games have ever been in:

- **Monotone-irreversible.** A readout that has only ever moved one way, *and* whose reverse
  transition appears nowhere in `StateGraph.adj`, is a candidate progress coordinate. Games
  are built so progress does not undo itself: doors stay open, collected things stay
  collected.
- **Terminal-extremal.** A readout that is systematically extreme on the frames the agent
  died on is a *negative* coordinate. This is the first signal that ever fires — deaths
  precede wins in every game.
- **Bounded-approaching.** A readout with a visible saturation point (`agree → total cells`,
  `count(target) → 0`) supplies a dense gradient where a binary predicate supplies none.

**Channel C — NONE.** `Φ ≡ 0`, `target() → None`, and the frontier order is returned exactly
as received. **The agent behaves byte-identically to today.** This is not a fallback; it is
the component's correctness floor and the ninth harness case.

### 4.2 The potential, and why it is a potential

`Φ(s) = Σ_f  w_f · direction_f · normalise(readout_f(s))`, with `w_f` the posterior mean of
`Beta(α_f, β_f)` and Channel-A features weighted above any Channel-B feature.

Φ is used **only as potential-based shaping** (Ng, Harada & Russell, 1999): it re-orders
choices and never alters the set of choices. That is the formal reason a wrong goal cannot
be worse than no goal — under potential-based shaping the optimal policy set is invariant,
so the worst case of a completely wrong Φ is a permutation of an exploration order that was
previously arbitrary anyway.

### 4.3 Selection: Thompson sampling, never argmax

Pursuit selects a feature by **sampling** `Beta(α_f, β_f)` (AB-MCTS-A, arXiv:2503.04412),
not by taking the maximum. A hypothesis that keeps failing loses budget share smoothly and
automatically, with no hand-tuned confidence threshold to get wrong, and a correct
hypothesis that was merely unlucky is never permanently discarded. Credibility updates come
**only from real transitions** — a world-model rollout that "confirms" a goal confirms
nothing.

### 4.4 The pursuit loop (where the RHAE saving actually happens)

1. `certify()` — replay the Timeline, prune refuted features, dedupe, re-rank. **Unscored.**
2. Sample a feature; emit `target()` as a **state predicate** (`Φ ≥ k`, `count(c) == 0`),
   never a pixel coordinate. A predicate survives a change of layout; a coordinate does not,
   and that was exactly why the old `_llm_goal` could not transfer.
3. ExecPlanner searches for a plan satisfying the predicate **inside the world model**.
   **Unscored.** No plan found → no scored action is spent; return to the frontier.
4. Before executing, the hypothesis records a **falsifiable prediction**: "satisfying this
   predicate produces a score event."
5. Execute the plan, capped by `pursuit_budget()`, aborting on the first divergence between
   predicted and observed frame (the EWM plan-executor discipline: execution *is* an online
   test of the model).
6. Score event → `α += 1`. Predicate satisfied with **no** score event → `β += 1`, and the
   hypothesis is retired if it drops below the pursuit floor. Divergence → the world model
   is at fault, not the goal; credibility is untouched.

Steps 1–4 are free under RHAE. Only step 5 costs, and it costs a bounded number of actions
on a plan that has already been shown to work in a model certified against the whole record.

### 4.5 Budget

`pursuit_budget()` scales with posterior credibility and with **actions since the last
confirmed potential gain**, and with nothing else — the human baseline is not available at
runtime and must never appear in this expression. A weak hypothesis buys a 3-action probe;
only a hypothesis that has already paid off buys a long plan.

---

## 5. THE hazard to design against

**A wrong goal pursued confidently is strictly worse than no goal at all.**

This is the exact analogue of StateGraph §5's "over-masking is fatal, under-masking is
cheap", and it is not hypothetical — the codebase already carries the scar. `_nearest_target`
("the goal is the nearest distinctive object") kept walking into decoys, which is why
`non_goal_colors` had to be added: a patch that only ever learns *negatively*, one wasted
pursuit at a time, and only for colours.

Why wrong-and-confident beats no-goal downward:

- No goal is an *unbiased* walk. It wastes actions, but it retains a real chance of
  stumbling onto the objective — which is precisely how the agent's current 4 levels were
  completed.
- A wrong goal is a *biased* walk toward a fixed attractor. The agent takes a different
  path to the same decoy fifty times, so **Livelock cannot see it** — every pursuit is a
  distinct trajectory and no state repeats often enough to trip the breaker.
- Under a squared action ratio, the bias costs twice: it inflates total actions *and*
  removes the accidental-discovery chance that made the unbiased walk survivable.

**The guard, in four parts** (each an analogue of the disproportionality rule that made
sacred cells safe):

1. **Re-ranker, not a gate — the structural floor.** GoalModel may reorder the untested
   frontier. It may **never remove a pair from it**. Set-equality of input and output is a
   harness assertion, not a convention. This bounds the worst case of an arbitrarily wrong Φ
   at "a permutation," which is the strongest safety property available and is why this
   component can ship before it is any good.

   > **Correction to this argument, found in the build.** The frontier order was *not*
   > arbitrary. `nearest_untested` picks the **least-globally-used** action with a random
   > tie-break, and that is a load-bearing mechanism, not a default: plain name-ordering
   > hammered ACTION1 for 284 of 297 presses on tr87, and the random tie-break exists
   > because a strict round-robin stamps a fake clock period onto world rows and fools the
   > periodicity mask. Permuting it freely can therefore degrade the HUD mask that every
   > other component depends on.
   >
   > So the permutation is deliberately **narrow**: Φ is a function of the *state*, and it
   > reorders only among frontier pairs at the **same shortest BFS distance**. The
   > least-globally-used order is applied first and the Φ sort is stable, so the exploration
   > balance survives underneath as the tie-break, and the shortest-path guarantee is
   > untouched. When Φ has no opinion, `nearest_untested` takes its original early-return
   > path and is byte-identical — including its consumption of the shared RNG stream.
2. **Refutation is free; adoption is expensive.** A candidate must be consistent with the
   **entire** recorded history — every level, every death — before it may direct one scored
   action. Most decoys are already refuted by the record: the agent has usually touched them
   and got nothing. That refutation costs zero actions, because the actions were already
   spent.
3. **Falsification-first.** A hypothesis must make a risky prediction to receive budget, and
   satisfying the predicate *without* the predicted outcome is counted as a failure. Today's
   `_llm_goal` already does the special case ("arrived, no level-up → drop the hint");
   generalising it to every hypothesis is what stops a plausible-looking decoy from
   absorbing a whole level.
4. **Thompson sampling, never argmax** (§4.3) — no threshold to mis-tune, graceful decay,
   no permanent loss of an unlucky-but-correct hypothesis.

### 5.1 The specific trap this component must not fall into

**A step-counter is a perfect fake progress coordinate.** It is monotone, irreversible, and
bounded — it satisfies every Channel-B test simultaneously. An agent that adopts it will
conclude it is making progress on literally every action it takes, and will therefore never
abandon any plan.

The separator is the one StateGraph already computes: **a real progress coordinate advances
under *some* actions and not others; a clock advances under all of them.** The argument is
identical in form to the sacred-cell disproportionality rule — compare the rate under the
implicated action against the rate over all actions, and a pure clock scores zero margin by
construction. (In the build the per-action rates are computed over the *readout's* own value
vector rather than read out of `StateGraph._act`, which is per-cell; same statistic, correct
granularity, and no new coupling.)

> **The separator is a VETO, not an adoption rule.** It asks exactly one question: *is there
> an action under which this does not advance?* The tempting stronger form — "and some
> action must advance it *often*" — is wrong, because it rejects sparse positional readouts:
> `contact(avatar, goal)` moves on a handful of transitions in a whole level and is not
> thereby a clock. Adoption is Channel A's and Channel B's job. This test only removes the
> one candidate class that would otherwise pass every one of theirs.
>
> Click-only games (10 of 25) never well-sample a second base action, because every action
> is `ACTION6`. The same test with the action partition collapsed still works: a per-action
> clock moves on *every* transition, so a low global move rate cannot be one.

**Residual blind spot, to be pinned as a WARN rather than papered over:** a genuine progress
coordinate that also happens to advance on every action is statistically indistinguishable
from a clock with the information available here. This is the exact mirror of
`every_step_blinker_reads_as_hud` in `test_stategraph.py`, and it is accepted for the same
reason — the failure it protects against is much worse than the opportunity it costs.

---

## 6. Interfaces (frozen contracts)

**Write path**
- `observe(prev, action, nxt, reward, terminal, level)` — one **real** transition. The only
  input that may move credibility.
- `note_level_complete(frame, level)` — the winning frame; supplies a target exemplar and
  promotes the features that predicted it.
- `certify()` — backtest the pool against the Timeline; prune, dedupe, re-rank. **Unscored,
  callable freely, must be idempotent.**

**Read path**
- `potential(grid) -> (phi, credibility)` — the order and how much to trust it.
- `target(grid) -> GoalPredicate | None` — a predicate with `holds(grid) -> bool` and
  `describe() -> str`. Never a coordinate. `None` whenever nothing is adoptable.
- `rank_frontier(candidates) -> candidates` — **the safe consumer.** Same set, new order.
- `pursuit_budget() -> int` — max scored actions before re-certification is required.
- `explain() -> str` — the current top hypothesis in words, for the diag harness. A goal
  model that cannot say what it believes cannot be debugged.

**Lifecycle (the part that differs from every other component)**
- **Features persist for the whole game**, across level boundaries and across RESET.
- **Bindings reset per level** — which panel pair, which cell holds the counter. The rules
  survive; the layout does not.
- This two-tier lifecycle is what §1.3 requires, and it is the one place GoalModel must
  *not* copy StateGraph.

> **Where the cut actually falls.** The draft put *colour* on the level-scoped side. The
> build puts it on the game-scoped side, because that is where the rest of the agent already
> puts it: `rplanner` deliberately carries `avatar_color` and the per-action displacements
> across a level-up and forgets only coordinates, and the palette in these games means the
> same thing on every level. The honest rule is **colour is a rule, position is a layout** —
> so `cell(r, c)` and `agree(panel_i, panel_j)` are dropped at a level boundary and
> everything keyed on colours is kept, along with its credibility.
>
> Also dropped per level: the observed value **ranges** used to normalise Φ (a new board has
> a new scale), and the set of candidates that lost the pool cap or lost a dedupe tie-break
> — a new layout can make two readouts that were indistinguishable here distinguishable
> there, so they are worth re-asking exactly once per level.

---

## 7. What it must NOT do (anti-responsibilities)

1. **Never choose an action.** It returns an order and a predicate; the orchestrator acts.
2. **Never remove a pair from the untested frontier.** Re-rank only (§5 guard 1).
3. **Never emit a goal as a pixel coordinate.** Always a predicate over readouts, so it
   transfers to an unseen layout.
4. **Never update credibility from an imagined transition.** Only real outcomes count.
5. **Never carry a layout-specific binding across a level boundary.**
6. **Never let an intrinsic (Channel B) signal outrank a confirmed (Channel A) one.**
7. **Never branch on a game id, and never grow the readout vocabulary to fit one game.**
   Which feature is right is learned; the candidate set is fixed.
8. **Never call the LLM**, and never treat an LLM-proposed feature as pre-certified.
9. **Never own history.** Timeline and StateGraph are the record; a private copy could
   disagree with the thing every other model is certified against.

---

## 8. Isolation harness contract (`test_goalmodel.py`)

Hand-built inputs with answers known by construction. No game engine, no LLM. Bar is
"passes this fixed set", never deleted, never shrunk.

1. **No-evidence no-op.** Empty pool → `target()` is `None`, `potential()` is `(0, 0)`,
   `rank_frontier` returns its input **in the original order**.
2. **THE case that matters most — a decoy is refuted by history at zero action cost.**
   Timeline where the avatar contacted red twice with no reward and blue once with a score
   event: assert "reach red" never becomes adoptable, "reach blue" does, and that this
   verdict is reached inside `certify()` without a single scored action.
3. **The clock trap (§5.1) — the second gate.** A record with **zero** score events
   containing (a) a counter advancing only under ACTION2 and (b) a step counter advancing
   under every action: assert (a) is adopted as a progress coordinate and (b) is **not**.
   Then the mirror case as an explicit **WARN**: a genuine coordinate that advances under
   all actions is not separable, and the harness records that rather than hiding it.
4. **Frontier is permuted, never truncated.** For randomised pools and randomised Φ, assert
   `set(out) == set(in)` and `len(out) == len(in)`. Property-style over many seeds.
5. **Cross-level transfer.** A feature confirmed on level 1 survives `new_level()`; its
   binding does not; a feature refuted on level 1 stays refuted on level 2.
6. **Falsification retires a hypothesis.** Satisfying a predicate N times with no score event
   drives credibility below the pursuit floor within a bounded number of attempts, and the
   hypothesis is not resurrected by later unrelated evidence.
7. **Priority ordering.** A Channel-B feature pointing one way and a Channel-A feature
   pointing the other: Channel A wins, always, regardless of the B feature's credibility.
8. **Observational-equivalence pruning + MDL tie-break.** Two readouts with identical value
   vectors over the Timeline collapse to one, and the shorter description survives. Pool size
   stays ≤ `MAX_FEATURES`; `certify()` cost stays bounded as the Timeline grows.
9. **Wiring, against the real `MyAgent.choose_action`.** A `Frame` stub plus spies: `observe`
   fires on every transition including score events and GAME_OVER; `note_level_complete`
   fires on level-up; and — the decisive assertion — **with an empty pool, the action
   sequence is identical to the pre-component agent** for a fixed seed. Shipping this
   component cannot regress anything, and that is a test, not a promise.

Cases the build added, because the implementation surfaced hazards the design had not
named:

2b. **Evidence that is only a base rate is not evidence** (§4.1a). A readout at its maximum
    on the winning frame *and* on almost every other frame is **not** confirmed; a readout at
    its maximum only on the winning frame is. This is the case that stops the pool filling
    with base rates.

4b. **The graph-level permutation contract.** Against the real `StateGraph.nearest_untested`:
    a ranker can only ever return a genuinely untested action; a ranker that drops a
    candidate is ignored outright; a ranker that raises is ignored outright; and with no
    ranker the search is deterministic under a fixed seed. The component defends against its
    own consumer being wrong, not only against itself being wrong.

9b. **The goal model consumes no global randomness.** Fifty `target()` / `potential()` /
    `rank_frontier()` calls leave both `random` and `np.random` byte-identical. This is
    *why* case 9 can hold: Thompson sampling draws from a private generator, so shipping the
    component cannot desync a seeded run. Case 9 also asserts that the pool really was inert
    during the identity run, so the claim is "empty pool ⇒ identical" and not "two live
    rankers happened to agree".

---

## 9. Build order

Each step lands only when the previous one is green, and each is reported as **what changed
/ unit-test result / real-game score**.

1. **Skeleton + safety floor.** Readout vocabulary, `observe`, `certify`, `rank_frontier`.
   `Φ ≡ 0`, `target() → None`. Gate: harness cases 1, 4, 9 — including the byte-identical
   action sequence. Nothing else may proceed until case 9 passes.
2. **Channel A (confirmed).** Score/terminal-driven confirmation, seeded from `_out_chg` and
   `sacred_cells()` read positively. Gate: case 2.
3. **Channel B (structural) + the clock separator.** Monotone-irreversible,
   terminal-extremal, bounded-approaching, gated by the per-action disproportionality test.
   Gate: case 3 with its WARN.
4. **Credibility, selection, budget.** Beta priors, Thompson sampling, falsification
   retirement, `pursuit_budget`. Gates: cases 5, 6, 7, 8.
5. **`agree(panel_i, panel_j)`.** The churn-game hypothesis, added last precisely because it
   is the least certain. It ships only if the harness gates hold and the churn games
   (tn36, r11l, su15 — dev only) move on re-spend %.
6. **Consumer wiring.** `rank_frontier` into StateGraph's `nearest_untested` ordering;
   `target` into ExecPlanner's unscored search; `explain()` into `diag_eval`.

**Verification ladder, in order, no step skipped:**
`python my_agent.py` self-test → all six unit harnesses green (`test_eyes`, `test_griddsl`,
`test_synth`, `test_livelock`, `test_stategraph`, `test_goalmodel`) → toys (keydoor 2/2,
reachgoal 3/3) → tier-1 real games → the 17 dev games via `diag_compare`. **The 8 held-out
games are never inspected while tuning.**

---

## 10. Falsifiable predictions

The design is only worth building if it can be wrong. It predicts, on the **dev** games:

1. **Churn collapses before levels appear.** Re-spend % on tn36 / r11l / su15 / vc33-class
   games falls materially at unchanged level counts. If re-spend does not move, the ordering
   is not being consulted or Φ is uninformative — and step 5 is refuted regardless of how the
   code reads.
2. **Inert games spend fewer no-op actions.** wa30 / g50t / m0r0 / tu93 masked-noop % falls;
   these games already have a correct world model, so a target is the only missing input.
3. **Any level completed a second time costs fewer actions than the first.** This is the
   cross-level transfer claim (§1.3) and the only one that can move RHAE past the
   denominator. tu93 (2/9) and lp85 (1/8) are the first places to look.
4. **No regression anywhere.** By construction (§5 guard 1 + case 9) no game may lose a
   level. If one does, the component has violated its own contract and is reverted, not
   patched.

Prediction 4 is the one that matters most. The other three are why it is worth trying;
that one is why it is safe to.

---

## 11. Build record — what the predictions actually returned

### 11.1 First measurement (2026-07-31): controlled pair, 25 games, seed 0

The `sac` baseline is a different build, so a delta against it would contain the refactor as
well as the goal model. The measurement is therefore a **paired run of IDENTICAL code**,
`ARC_NO_GOALS=1` (goff) against goals-on (gon), 5 shards each, `--timeout 390`.

| | DEV (17) | HELD-OUT (8) |
|---|---|---|
| levels | 4 -> 4 **(+0)** | 0 -> 0 (+0) |
| mean masked-noop | 19.7% -> 19.1% (-0.7) | 11.7% -> 10.9% (-0.9) |
| mean re-spend | 50.1% -> 49.9% (-0.2) | 44.8% -> 45.1% (+0.3) |
| REGRESSED | none | none |

Against §10:

- **#4 (no regression) — CONFIRMED.** No game lost a level on either split. The safety
  contract of §5 guard 1 + case 9 held in the wild, which is the claim that made the
  component safe to try at all.
- **#1 (churn collapses) — REFUTED on its own named games.** tn36 / r11l / su15 / vc33 came
  out byte-identical (87.6 / 78.2 / 86.3 / 88.2, unchanged). The pool never went live on any
  of them, so the ordering was never consulted. §10 committed to calling that a refutation
  "regardless of how the code reads", and it is recorded as one.
- **#2 (inert games spend fewer no-ops) — 2 of 4.** m0r0 -5.1 and wa30 -1.5 for; g50t +0.5
  and tu93 +2.5 against.
- **#3 (second completion cheaper) — NO EVIDENCE.** No game completed any level twice, so
  the cross-level transfer claim of §1.3 remains entirely untested.

Nine rows hit the 390s cap and are truncated at different action counts (both sets ran
concurrently under CPU contention); they are symmetric but not comparable, and are excluded
below. On the **16 clean rows**, 10 are byte-identical and 5 moved re-spend by >=2pts:
**tr87 -11.1, wa30 -3.0, m0r0 -2.7** against **sp80 +10.5, cd82 +7.1**. Dev mean -0.37pts.
Helps as often as it hurts, at comparable magnitude: a coin flip, not an effect.

### 11.2 Why — the finding that matters more than the table

`A=0` and `retired=0` in **all 25 games**. The model never confirmed a goal and never refuted
one. The cause is not a tuning miss, it is a disconnected wire: **`target()` has no call
site** (§9 step 6 is deferred), so `_pursuit` is never set, so `_settle_pursuit` returns
immediately, so `alpha`/`beta` never move. Every feature stayed frozen at Beta(1,1),
`post_mean()` was a constant 0.5, and `potential()` therefore degenerated into an
*unweighted vote of unconfirmed structural guesses*. The Bayesian layer described in §4 was
inert by construction in the shipped wiring.

sp80 is §5's hazard appearing in real data rather than in argument: masked-noop fell 13.3pts
while re-spend **rose** 10.5. The agent stopped idling and spent every freed action
re-treading transitions it had already seen — pulled toward a wrong coordinate. "A wrong
goal is worse than no goal" is no longer a design worry; it is a measured row.

### 11.3 The gate this forced — `Feature.steers()`

Steering on a belief that has never been tested is exactly what the project's standing rule
forbids. So the potential now admits a feature only if:

    channel == "A"  or  (alpha + beta) > 2.0

Channel A steers on adoption, because `_confirmed` already paid for it with real score events
*and* the §4.1a base-rate margin — A's evidence lives in the adoption, not in the Beta.
Channel B is a structural hypothesis with no outcome behind it, and stays out of the
potential until a pursuit outcome has actually moved its posterior.

**`target()` deliberately does NOT apply the gate.** Pursuit is how a hypothesis earns
evidence; the potential is where earned belief is spent. Gating both would deadlock Channel B
forever — never pursued, so never updated, so never qualified. `test_goalmodel.py` case 10
asserts the asymmetry in both directions so that a future "cleanup" cannot quietly close it.

The gate makes a falsifiable claim of its own, and it was tested rather than argued: with
`A=0` in every game and Channel B barred from the potential, `_active()` must be False
everywhere, so `frontier_ranker()` must return None and `rank_frontier` must be the
identity — i.e. **the gated run must reproduce goals-off exactly**. Third 25-game run
(`scratchpad/run_gate_diag.sh`, seed 0):

- **13 games identical to goals-off on all 9 recorded metrics** (levels, actions, noop%,
  maxrun, re-spend%, cells, clicks, promotions, demotions).
- **12 games differ, and every one of them is a wall-clock-TIMEOUT row** — truncated at a
  different action count because the gate run was alone on the CPU instead of sharing it
  with a paired set. Not a behavioural difference; the same artifact as
  `eval-wallclock-modern-standby`.
- **0 non-truncated rows differ.** The contract holds exactly where it is observable.
- Levels 4 -> 4, REGRESSED: none. `frontier_ranker()` was None in all 25 games.

The honest cost is recorded too: the gate gives up tr87's -11.1 re-spend along with sp80's
+10.5 and cd82's +7.1. That is the correct trade for a coin flip, not a sacrifice.

### 11.4 What this makes the component, today

Honestly: **a proven-safe substrate with no proven benefit yet.** Prediction 4 is the only one
that passed, and with the gate in place the component's measured contribution is zero by
design rather than by luck. It stays wired because §9 step 6 — pursuit, the half that makes
the Beta layer mean anything — is the next measured step, and it needs this substrate. It
does not get claimed as a win until predictions 1-3 return something.
