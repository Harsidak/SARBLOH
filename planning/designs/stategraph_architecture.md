# StateGraph (Component 5.8) — Architecture for ARC-AGI-3

**Thesis.** StateGraph is the agent's *exact, HUD-invariant world memory and exploration
frontier*. It answers three questions and nothing else: **have I seen this world-state
before?**, **what have I not yet tried here?**, and **what is the shortest known path to the
nearest untried action?** It never chooses an action, never crosses a level boundary, and
never masks a score-relevant cell. Every design rule below exists to keep those three
answers *exact* under a benchmark that actively tries to make them wrong.

---

## 1. Why this component is load-bearing

The HUD mask that lives here feeds **every masked comparison in the system**: the
world-model verifier (`hud_mask(precise=True)`), the livelock breaker and the learner
(`changed_masked`), and the state hash itself. A subtle error here is not local — it
silently corrupts the two most recent fixes (verifier + breaker). That is why it is the
next lock, and why its harness matters more than its code.

---

## 2. ARC-AGI-3 constraints that dictate the design

| Constraint | Design consequence |
|---|---|
| Deterministic engine: same (state, action) → same next frame | An observed edge is **true forever**. Exploration never re-tests a known pair. This determinism is the premise the whole graph rests on. |
| 64×64 grid, 16 colors, HUD/timer strips tick every action | Raw frame hashes never repeat → "everything is novel." Cells that tick regardless of the agent **must be masked out of the identity hash**, or the graph is useless. |
| 1500-action budget; RHAE squares the action ratio | Every scored action must buy new information. The `untested` frontier is the mechanism that stops re-spending known pairs. |
| Objective and controls are hidden | The graph is policy-agnostic: it records what happened, it does not assume what matters. |
| ≤5 keys + undo + **parametric click (ACTION6)** | ACTION6 is excluded from adjacency (too many per-cell variants would explode the graph). Clicks are deferred to the ClickPlanner. |
| Intra-level RESET returns to the **same** layout | Graph **persists across RESET** — every edge is still true, so a restart replays into known territory for free. |
| Level-up swaps the **whole** layout | Graph is **discarded and rebuilt per level**. No edge survives a level boundary. |

---

## 3. Data model (single source of truth)

- `adj: hash → {action: next_hash}` — the exact transition function over *world*-states.
- `untested: hash → set(actions)` — the exploration frontier.
- `grids: hash → representative raw grid` — needed for BFS no-op prediction and re-hashing.
- **Append-only raw logs** (`_ensures`, `_trans`) — the ground truth. Everything derived
  (adj/untested/grids) is a *view* of these under the current mask, so it can be rebuilt
  exactly when the mask changes.
- Mask statistics: per-cell change counts, per-row/col change + **gap histograms**,
  per-action change stats.

**Invariant:** derived structures are never edited in a way that can't be reproduced by
replaying the raw logs. This is what keeps the graph exact despite an *online-learned* mask.

---

## 4. The state hash — HUD invariance (the crux)

A world-state's identity is its **non-HUD cells**: `hash = hash(grid with masked cells
blanked)`. Two frames that differ only in a timer are the **same node**.

The mask serves two consumers with opposite needs, so it is computed once and exposed at
two granularities:

- **`cellmask` (precise)** — a cell is HUD iff it changes in > `MASK_RATE` of transitions
  **and** under ≥2 distinct well-sampled actions. This is what the **rule verifier**
  consumes: it must not excuse a real world cell just because it shares a row with a timer.
- **`mask` (broad)** — `cellmask` **plus clock-periodic row/col strips** (a line whose
  change-gaps are dominated by one period is a clock). This is what the **state hash**
  consumes: deliberately over-broad, because painting a flickering row is cheap insurance
  against hash churn.

**Exact-rebuild rule:** when the mask changes, discard adj/untested/grids and replay the
raw logs under the new mask. Never migrate hashes in place. (Already implemented in
`_update_mask`; keep it — it is the correctness backbone.)

---

## 5. THE hazard to design against (ARC-AGI-3 specific)

The mask's premise is "cells that tick every action are HUD." **Some *world* cells
legitimately tick every action:** a blinking goal, a pulsing hazard, an animated portal, a
flashing key. The current defenses (≥2-action confirmation, periodic-strip rule) do **not**
fully cover a single animated cell that ticks under all actions with a regular period — it
can be eaten by the mask.

**Design law: the mask must be conservative — under-masking is cheap, over-masking is
fatal.**
- Under-mask (treat an animation as world): the graph thinks two states differ when only an
  animation differed → wastes *some* exploration. Recoverable.
- Over-mask (treat a goal/hazard as HUD): the graph **cannot tell a win-state from a
  loss-state** → the agent is blind to the only cells that decide the score. Unrecoverable.

**Required guard (new): sacred cells.** A cell (or strip) is **never** eligible for the mask
if it has ever been coincident with a **score change** or a **GAME_OVER** transition. Feed
the graph the reward/terminal signal it currently ignores, and let it protect those cells
absolutely. This is the single most important ARC-AGI-3 addition to this component.

---

## 6. Interfaces (frozen contracts)

**Write path**
- `record(prev, action, next)` — learn one edge + update mask statistics.
- `ensure_state(grid, valid)` — register a frame and its untried non-click actions.
- `note_outcome(reward, terminal)` *(new)* — mark the current transition's cells as
  score-relevant / terminal, feeding the sacred-cell guard.

**Read path**
- `changed_masked(prev, next) -> bool` — did the *world* change? (breaker + learner)
- `hud_mask(precise: bool) -> mask | None` — the learned mask. (verifier)
- `nearest_untested(grid, dead, noop) -> (path, action) | None` — shortest unscored path to
  the frontier; `None` only at true exhaustion, and the caller must retry without `noop`
  because a world model may be wrong. (orchestrator exploration)

**Lifecycle**
- Persist across intra-level RESET. New instance per level.

---

## 7. What it must NOT do (anti-responsibilities)

1. Never choose an action — it exposes the frontier; the orchestrator picks (least-globally-
   used, random tie-break, already implemented).
2. Never carry state across a level boundary.
3. Never mask a score-relevant or terminal-coincident cell.
4. Never migrate hashes on a mask change — always rebuild from raw logs.
5. Never admit ACTION6 into adjacency.

---

## 8. Isolation harness contract (`test_stategraph.py`)

The harness is the deliverable that "locks" this component. Required cases:

1. **HUD invariance** — two frames differing only in a ticking timer strip hash equal; a
   sequence with a per-action timer does **not** explode `untested`.
2. **Sacred-cell guard** *(the case that matters most)* — a blinking goal cell and a pulsing
   hazard cell that tick under all actions are **NOT** masked once flagged by
   `note_outcome`; assert they remain in the identity hash.
3. **Exactness / order-independence** — the same set of edges inserted in any order yields
   an identical graph; a mask rebuild reproduces adj/untested bit-for-bit.
4. **Frontier correctness** — `nearest_untested` returns a genuine shortest path; returns
   `None` only at true exhaustion; `noop`-skipped pairs are re-offered on the `None` retry.
5. **RESET persistence vs level reset** — edges survive an intra-level RESET; a new level
   starts empty.
6. **ACTION6 exclusion** — click actions never enter adjacency or the frontier.
7. **Budget bounds** — `MAX_BFS_STATES` / `MAX_LOG` respected; behavior graceful at the cap.

Every case is constructed by hand with a known answer — no game engine, no LLM. Bar is
"passes this fixed set," never deleted.

---

## 9. Build order

1. Add `note_outcome` + the sacred-cell guard (the one real gap).
2. Write `test_stategraph.py` covering §8, with case 2 as the gate.
3. Keep all existing harnesses green; then re-run the LLM-free `--real` eval and confirm the
   mask still collapses HUD games without newly blinding any scored game.
