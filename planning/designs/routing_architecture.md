# The routing ladder: who spends the actions

`MyAgent.choose_action` is one function with ~13 exits, and until 2026-08-01 the
identity of the branch that produced an action died with the call. Every claim
about "the planner" was therefore unfalsifiable. This document is the map, the
instrument, and the measurements.

## 1. The ladder as it stands

Order matters: the first branch to set `action_str` wins, and every branch below
it is unreachable for that step.

| tag | branch | condition |
|---|---|---|
| `boot` | probe an unlearned move | `untried` and (not `ready` or every 6th step) |
| `react.*` | `ExecPlanner.act` (COMPONENT 5.7) | `ready` ∧ ¬`looks_mechanical` ∧ **¬`click_only`** ∧ `rand > ε` |
| `click.search` | ClickPlanner env-as-simulator BFS | `click_only` ∧ `cplanner.searching` |
| `click.plan` | LLM-reasoned click plan | `click_only` ∧ plan queued |
| `click.hold` | idle on an inert cell while a plan generates | `click_only` ∧ `_click_inflight` |
| `click.probe` | occasional ACTION5/7 probe | `click_only` ∧ 15% |
| `click.cover` | coverage lattice (the floor) | `click_only` otherwise |
| `mcts` | `MCTSPlanner.search` | `gate.should_exploit()` ∧ promoted model |
| `graph` / `rand` / `novelty` | `_explore_action` sub-exits | otherwise |
| `+click` | bare `ACTION6` re-coordinated | any branch yielding bare `ACTION6` |
| `X>esc.Y` | livelock breaker overruled branch X with rung Y | post-filter |

`react.*` sub-tags name which of `ExecPlanner.act`'s nine exits fired:
`plan` (committed T1 queue), `exper` (discriminating experiment), `goal` / `goal1`
(BFS to target, green / red), `frontier` / `frontier1` (BFS to unvisited),
`bfs1` / `greedy` (inherited `ReactivePlanner`), `noavatar` / `notarget` / `nomoves`.
The distinction is not cosmetic: "committed a modelled plan" and "fell through to
the inherited greedy" are different agents wearing the same name in a score table.

## 2. The instrument

`MyAgent._route` / `_route_ms` and `ReactivePlanner._sub` are **pure assignments** —
no branch reads them, so tagging cannot change a decision. `scratchpad/route_diag.py`
aggregates them per game:

- `n`, `share` — actions spent by that branch
- `ms`, `p95`, `secs` — cost to DECIDE. RHAE ignores thinking time; the 9h Kaggle
  wall clock does not, so this decides how many games are reachable at all.
- `chg%` — share of the branch's actions that changed the HUD-masked world
- `new%` — share that reached a state never seen before. Strictly harder than
  `chg%`: re-treading a known state moves pixels and teaches nothing.
- `rew` — score events credited to the branch
- `dbl_reset`, `score_wipes` — the RESET/score invariant (§3), checked empirically
  across every emitter rather than argued per call site

`chg%` and `new%` are the point. A branch can be fast, busy, and worthless.

Run: `sh scratchpad/run_route_diag.sh` (17 dev games, 4 shards; the 8 held-out
games are deliberately absent — no routing change may be derived from them).

## 3. The RESET/score contract (arcengine)

Read out of `site-packages/arcengine/base_game.py`, not inferred:

- L279 — `_action_count` increments only on **non**-RESET actions
- L160 — `set_level` zeroes `_action_count`, and `level_reset()` calls `set_level`
- L313 — `handle_reset`: `_action_count == 0` → `full_reset()`
- L321 — `full_reset` sets `_score = 0`

**Therefore two RESETs with no action between them destroy every level already won.**

`ClickPlanner._enter_search` staged a lone `["RESET"]`; `_build_next_plan` then
staged `["RESET", *path, b]` — back-to-back. Measured on lp85 (seed 0, `CP_DEBUG=1`):
the agent won level 1 and wiped it **six times** in a single run, finishing 1/8.

The agent cannot read `_action_count` (on Kaggle the engine is remote), so
`ClickPlanner._acts_since_reset` mirrors it. `note_action()` is called from
`choose_action` for **every** executed action — RESET included, which `update()`
never sees — and is zeroed on level-up (the engine's `set_level` zeroes its counter
with no RESET emitted). `_search_step` then skips a RESET when the mirror reads 0:
at that point the board already IS the level's initial state, so the RESET is a
no-op in intent as well as a score-destroying one.

The other three RESET emitters are safe, and the reasons are worth recording:
`_escalate` rung 4 is gated by `may_reset()`, which requires
`win_actions >= NOOP_HARD` and `win_actions` counts only non-RESET actions;
GAME_OVER is safe because the death-causing action incremented the counter;
NOT_PLAYED is safe because there is no score to lose yet. Any NEW emitter must
consult the mirror — and be verified with `dbl_reset` / `score_wipes`, not argued.

## 4. Measurements

### lp85 (click-only, seed 0) — the two fixes

| | before | after RESET fix | after both fixes |
|---|---|---|---|
| wall clock | 100s | 150s (timeout) | **64.5s** (full 1500 actions) |
| total decide time | 1.9s | 68.4s | **1.3s** |
| `score_wipes` | 6 | 0 | **0** |
| `react.*` share | 5.5% | 84% | **0%** |
| levels | 1/8 | 1/8 | 1/8 |

The middle column is the interesting one. Removing the score wipe let the agent
keep level 1 and spend its budget on level 2 — where `ExecPlanner` seized the
ladder and burned 60 of 150 wall-clock seconds at 70ms/decision (p95 395ms) for a
2.2% new-state yield. Branch (b) had no `click_only` exclusion, so branch (b2) —
the one built for clicks — was unreachable whenever (b) returned anything.

Neither fix raised `levels_completed` on lp85. The wipe fix protects wins rather
than creating them; the routing fix buys wall clock, which matters for the 9h
Kaggle budget rather than for RHAE directly.

### Dev sweep (17 games, seed 0, 2026-08-01)

```
DEV levels=4  score_wipes=0  dbl_reset=0   think=690s of 4909s wall (14.1%)
```

Levels match the pre-change baseline exactly (lp85 1/8, sp80 1/6, tu93 2/9), so
neither fix regressed anything, and the RESET/score invariant now holds across all
17 games rather than the one it was found on.

Per-branch totals over 18,504 dev actions (abridged to the branches that matter):

| route | share | chg% | new% | rew | think |
|---|---|---|---|---|---|
| `graph` | 23.7% | 89.8 | **54.0** | **3** | 188s |
| `click.search` | 21.3% | 99.2 | **2.0** | 1 | 0.7s |
| `rand+click` | 9.6% | 87.3 | 4.0 | 0 | 11s |
| `mcts` | 9.2% | 38.6 | 11.6 | **0** | **254s** |
| `react.exper` | 6.8% | 85.4 | 32.7 | 0 | 62s |
| `rand` | 5.0% | 25.8 | **0.7** | 0 | 15s |
| `react.plan` | **0.2%** | 97.8 | **97.8** | 0 | 0.1s |

`graph` (StateGraph BFS) is the workhorse: best yield of any major branch and the
source of all three reward events. `click.search` is the #2 action consumer at a
2.0% new-state yield — the lp85 re-treading generalises. `rand` + `rand+click` are
~15% of actions at near-zero yield (su15 is 97% `rand+click`): on the 14 mixed-action
games every `click.*` branch is gated behind `click_only`, so clicks there arrive
only through random exploration or escalation rung 3.

### MCTS paired A/B (COMPONENT 4) — the branch is pure cost

`mcts*` branches took 2,217 actions (12%) and **348s of the 690s of dev thinking
(50.5%)** for **zero reward events**. The A/B (seed 0, the 8 dev games where the
branch fires; the other 9 cannot differ):

```
TOTAL (8 games)   levels 1 -> 1   REGRESSED: none   GAINED: none
  wall clock  2020s -> 1719s     (-301s, -15%)
  think time  498.9s -> 221.9s   (-277s, -55.5%)
```

sb26 went 127.8s -> 5.0s of thinking; dc22 80.2s -> 32.7s; sp80 kept its level at
73.0s -> 33.1s. tr87 and wa30 got *slower* in wall clock despite lower think time,
which is env/trajectory variance rather than deciding cost — the one result that
does not fit the clean story.

**MCTS is therefore OPT-IN (`ARC_MCTS=1`), default off.** Kept behind a switch
rather than deleted: the evidence is dev-only. Verified with self-tests both ways
and toys 2/2 (reachgoal 3/3).

### The CERTIFY green-gate (why `react.plan` is 0.2% of actions)

Splitting `ExecPlanner.act`'s exits by whether `certify()` returned green:

| | actions |
|---|---|
| green (`react.goal` 74 + `react.plan` 45 + `frontier` 0) | **119** |
| red (`react.exper` 1260 + `goal1` 337 + `frontier1` 61) | **1658** |

The executable world model is certified only **~7%** of the time. `_commit` is
gated on green, so a red theory can only ever take one vetted step and never
commits a multi-step plan — the agent spends **1260 actions falsifying its model
against 45 executing it**, a 28:1 ratio.

The mechanism is a ratchet: `certify()` sets `_cert_green = False` on any
unexplained transition (my_agent.py:2596) but restores it only at 2589, inside
`if key != self._cert_key and len(entries) - _last_replay_at >= CERT_REPLAY_GAP`.
The key is displacement + walls + obstacle colours + anomaly/excusal counts +
level, so a theory that has CONVERGED — stopped being revised — and was falsified
once stays red for the rest of the level, whether or not the offending entry was
later excused.

**A/B result (`ARC_NO_CERT=1` forces green; 8 ExecPlanner-driven dev games):**

```
levels 3 -> 2   REGRESSED: ['tu93']   GAINED: ['ls20']
think time 212.9s -> 161.1s
```

**The gate stays. No code change.** The red verdict is EARNED, not stale — and
this corrects the "ratchet" reading above. `_anomalies` and `_excused` are part of
the cert key and `_experiment` grows them, so the key *does* change, replays *do*
fire, and green *does* get chances to return; `CERT_REPLAY_GAP` throttles those
replays rather than blocking them. The theory stays red because it keeps genuinely
failing to explain real transitions, which matches the long-standing finding that
the displacement model does not transfer to real games. Forcing green costs tu93
both its levels.

ls20 gaining a level under a forced-green model is real but weak evidence — one
game, one seed, and the only game in the set where acting on an uncertified model
paid. Not a foundation to build on without replication.

## 5. Phase 3: the explore fallback (least-spent) — did NOT prove out

With MCTS off, branch (c) always routes to `_explore_action`, whose fallback is
`if rand < eps or active_model is None: random.choice(alive)`. Most dev games never
promote a model, so that single line decided **~15% of all dev actions** at a 0.7%
new-state yield. Replacing it with the least-spent `(state, action)` rule that
escalation rung 2 already uses:

```
TOTAL (7 games)   levels 0 -> 0   REGRESSED: none   GAINED: none
  wall clock  1690s -> 1869s     (+179s, +11%)
  think time  129.7s -> 180.5s   (+50.8s, +39%)
  branch yield  rand 2.3% new -> least 6.5% new
```

| game | n -> n | new% -> new% |
|---|---|---|
| sb26 | 1074 -> 976 | 4.4 -> **25.3** |
| sc25 | 801 -> 62 | 0.1 -> **43.5** |
| bp35 | 215 -> **1269** | 8.4 -> **0.1** |
| lf52 | 944 -> 944 | 0.0 -> 0.0 |
| su15 | 1459 -> 1478 | 2.6 -> 2.2 |

The 3x aggregate yield is carried entirely by sb26 and sc25. On bp35 it inverted:
the branch grew 6x and yielded 0.1%, because least-spent keeps electing actions the
ledger has not seen while the graph is starved. lf52 -- the 0.0% case it was aimed
at -- did not move. Levels 0 -> 0 proves nothing either way: none of these games
completes a level, which is exactly why they are the games where this branch fires.

**Made OPT-IN (`ARC_LEASTSPENT=1`), default off** -- the same treatment MCTS got,
for the opposite reason: MCTS had proof of harm, this has no proof of benefit.
Part of the cost is self-inflicted (it hashes `grid`, and `choose_action` hashes
the same grid again for the breaker); fix that before any re-measurement.

## 5. Open defects this surfaced

1. ~~**`ClickPlanner` has no world model.**~~ **CLOSED 2026-08-01.**
   `_build_next_plan` staged `["RESET", *path, b]`, so discovering ONE edge cost
   `depth + 2` **scored** actions. It built a perfectly good state graph and then
   paid real currency for every edge — exactly the "spend scored actions to gather
   information" pattern RHAE squares.

   **Fix shipped:** the graph is now *used*, not merely collected. `_nav_dists()`
   BFSes the observed edges outward from the board on screen (imagination, zero
   scored actions), and `_build_next_plan` prices every frontier probe by both
   routes — *navigate* `[*path_from_here, b]` versus *teleport*
   `["RESET", *path_from_root, b]` — and takes the cheaper. Standing on a frontier
   node, a probe now costs **1** action instead of `depth + 2`. The four objectives
   the model is optimised against (O1 fidelity / O2 cost / O3 progress / O4 safety)
   are stated on the class and audited in `Docs/world_model_objectives.md`.

   O1 has teeth: a navigation plan carries `_expect`, the board it predicts at each
   step; a mismatch deletes the offending edge, abandons the plan, and re-plans
   **without recording the in-flight probe** — attributing a result to a parent we
   are not standing on is silent graph corruption, strictly worse than a wasted
   action.

   **Measured.** Unit (`test_clickplanner.py` cases 7–9, 30 cases, was 21):
   deterministic 2-button toy over a 3×3 state lattice — **27 scored actions vs a
   teleport-only lower bound of 72**, `nav=13 teleport=5`, 45 actions avoided.
   Real (lp85, seed 0, 1500 actions, official scorer): **honest 0.0988 → 0.3352
   (3.4×)**, same 1/8 levels, 0 wipes; the level is simply banked far sooner.
   `click.search` fell from **72% → 12%** of the budget.

   **What this surfaced (new, see 4).** With search no longer starving the budget,
   the dominant consumer on lp85 is now `click.cover` at ~87%, doing nothing after
   the search exhausts.
1b. **Search exhaustion was read as "planning is over".** Follow-on from 1, measured
   the same day. On lp85 level 2 the graph over the detected button set was explored
   to exhaustion — `states=60 edges=120 dead=1` in **180 actions** — and the
   remaining **1320** went to coverage. But exhausting the graph over button set B
   only proves the level is unreachable *through B*; lp85 is a sliding puzzle and
   only **two** buttons had ever been detected. The abstraction was refuted, not the
   method.

   **Change:** exhaustion now records B in `_exhausted_sets` and hands back to
   coverage, which keeps discovering reactive cells; `act()` re-enters search as soon
   as a detected set contains a button no refuted set had. Bounded by construction —
   each re-entry needs a strictly new button and buttons are capped.

   **A regression this caused, and the fix.** The first version re-entered through the
   unchanged `_enter_search`, which opens with a RESET to root the graph. On lp85 that
   RESET was read as a `full_reset` and **wiped a banked level** (`SCORE REGRESSION
   1->0 prev_action='RESET'`) even though the planner's `_acts_since_reset` mirror read
   300. The run scored `official 0.6258 / honest 0.3352 / wipes 1` — i.e. the entire
   apparent gain was churn, and capability had not moved at all. The mirror is not
   trustworthy mid-level, so re-entry now emits **no RESET**: the graph is rooted on the
   board already on screen (`_adopt_root`), and teleport is disabled for that episode
   because RESET does not return to such a root. The number of RESETs worth gambling a
   banked level on is zero. Pinned by `test_clickplanner.py` §11.

1c. **`click.cover>esc.cell` spends a third of the click budget on nothing.** New
   instrumentation in `eval/rhae_eval.py` reports per-route `chg%` (did the action
   alter the frame at all) and `new%` (did it produce a never-seen frame), because
   spend alone cannot distinguish working from fidgeting. Pooled over lp85+r11l:

   | route | actions | share | chg% | new% |
   |---|---|---|---|---|
   | `click.search` | 1612 | 53.7% | **96.2%** | 42.2% |
   | `click.cover>esc.cell` | 1010 | 33.7% | **1.5%** | 1.4% |
   | `click.cover` | 378 | 12.6% | 36.5% | 34.1% |

   `click.search` is the most productive route in the agent by a wide margin. The
   livelock breaker's escape cell is the least — 1010 actions that leave the board
   untouched 98.5% of the time. This is the same defect as 2 below, seen from the
   spend side rather than the code side.

2. **`fresh_cell` is a uniform random pixel sampler.** The escalation's
   modality-switch rung ignores everything `ClickPlanner` knows about which cells
   are interactive (`_stats`, `_exact`, `_reactive`). On lp85 it was the #2 budget
   consumer at a 1.9% change rate.
3. **MCTS explores a fiction.** When the world model makes no claim, the child node
   copies the parent's state and scores the maximum leaf value, so depth along
   unknown branches is meaningless. `_leaf_value` also carries no reward term —
   it is a pure novelty-seeker. Whether this is worth repairing depends on how
   often `mcts` fires at all; see §4.
