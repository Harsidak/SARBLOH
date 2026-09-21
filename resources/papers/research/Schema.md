1. Core Philosophy
Schema treats the world model as an executable program, not an implicit neural representation. The agent acts like a physicist: it observes raw data, forms hypotheses about what entities exist and how they interact, tests those hypotheses against evidence, and revises both its ontology (what exists) and its laws (how things change) when predictions fail.
Key insight from the paper: "The latent world representation is a program, not a vector — so it is interpretable (a text file you can read and diff), verifiable (replayable against recorded reality, belief by belief), and searchable (a program is a simulator; planning inside it is free)."
2. Architecture Overview
plain
┌─────────────────────────────────────────────────────────────┐
│                    OUTER LOOP (Agent-Environment)              │
│                                                              │
│   OBSERVE  →  DELIBERATE  →  EXECUTE  →  RECORD              │
│      ↑_________________________________________|             │
│                                                              │
│   • OBSERVE: Receive 64×64 grid, 16 colors, valid actions  │
│   • DELIBERATE: Open-ended reasoning (inner loop)            │
│   • EXECUTE: Run committed action queue with per-step checks │
│   • RECORD: Append-only timeline of all interactions         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              INNER LOOP (Deliberation / Thinking)            │
│                                                              │
│   THEORIZE  →  CERTIFY  →  PLAN  →  COMMIT                   │
│      ↑_________|                                              │
│      (mismatch: fix the bug and loop back)                   │
│                                                              │
│   • THEORIZE (write_code): Edit world model as step(state,   │
│     action) program; jointly encode state grounding + rules    │
│   • CERTIFY (run_backtest): Replay ALL recorded transitions;  │
│     exact match or pointed bug                               │
│   • PLAN (run_bfs): Search inside certified program;         │
│     zero real environment cost                               │
│   • COMMIT (commit_actions): Only channel from thinking to   │
│     action; queue sent to environment                        │
└─────────────────────────────────────────────────────────────┘
3. Persistent Memory (The Agent's "Weights")
Three append-only/immutable files that survive across deliberations:
Table
File	Purpose	Nature
world_model.py	Current theory of the game: extract_state(), step(), is_goal()	Editable — revised when predictions fail
notes.md	Working hypotheses, observations, reasoning scratchpad	Editable — agent's working memory
Timeline	Complete interaction history: (obs, action, next_obs, reward, flags)	Append-only, immutable — ground truth
4. The Two Levels of Abstraction
Schema solves two problems jointly in a single editable program:
Level 1: State Grounding (VIGA lineage)
plain
observation (raw pixels)  →  state program
What the world is: Identifies objects, quantities, names, properties
Invents entities from raw grid without labels: "the 5×5 cyan+maroon block is the player"
Nothing tells the agent which pixels count as player, wall, counter — it must infer this
Level 2: Mechanism Discovery (WorldCoder lineage)
plain
(state, action, state')  →  transition program step()
How the world moves: The rule written as executable code
Discovers hidden mechanisms: spring walls, refuel rings, color rotators, portal swaps
Critical: These are jointly encoded. When a rule refuses to stay consistent with experiments, the counterexample can indict the representation itself — not just the rule. The agent can revise what the state is (e.g., "the cart is a valid landing cell, not an obstacle") and then update the transition rule to match.
5. The Four-Stage Outer Loop (Detailed)
Stage 1: OBSERVE
Receives: 64×64 grid of 16 color indices, set of legal actions
No object list, no rule sheet, no stated goal, no shaped reward
The environment supplies only raw pixels
Stage 2: DELIBERATE
Open-ended reasoning episode that ends only in a committed queue of actions
Runs the inner loop (theorize → certify → plan → commit) until actions are ready
Can also design exploratory actions when multiple hypotheses remain consistent
Stage 3: EXECUTE
Runs the committed action queue
Per-step self-check: Every real transition is checked against the theory's prediction
Single mismatch stops execution — plan is voided, agent returns to deliberation
Stage 4: RECORD
Appends every real transition to the Timeline
Immutable: The agent can revise hypotheses, but cannot alter observations or actions taken
Serves as ground truth for all future backtests
6. The Inner Loop: DELIBERATE (Detailed)
6.1 THEORIZE (write_code)
LLM (Claude Opus 4.8 / GPT-5.6 Sol / Fable 5) writes/updates world_model.py
Contains three functions:
extract_state(grid) → structured state with objects, properties, relations
step(state, action) → deterministic transition function
is_goal(state) → goal predicate (must be inferred from interaction)
When uncertain, makes falsifiable hypotheses
Can revise either state representation or transition rule when contradicted
6.2 CERTIFY (run_backtest)
Replays every recorded transition through the current world_model.py
Checks: exact grid match on non-terminal steps + level_up/dead/win flags on EVERY step
Output: 393/393 exact or X mismatches with pointed bug location
Scale: Hundreds of exact 64×64 frames verified in one operation
If mismatch: return to THEORIZE with counterexample
6.3 PLAN (run_bfs)
Only runs if backtest is 100% green
Breadth-first search inside the certified program — not the real environment
Explores 10³–10⁴ modeled states at zero real action cost
Returns shortest action sequence to goal
6.4 COMMIT (commit_actions)
The only channel from thinking to action
Sends action queue to environment
After commit, remaining plan is immediately discarded if any prediction error occurs
7. Action for Discovery (Targeted Experimentation)
When the agent has multiple candidate rules consistent with history:
Identifies competing hypotheses about game mechanics
Searches for an action that produces different predictions under each hypothesis
Commits that discriminating action
Compares observation with competing predictions
Updates step() with the falsified/confirmed rule
Re-runs backtest
This is efficient under RHAE (which squares excess actions): the best experiment resolves the most uncertainty with the fewest real interactions.
8. Reality Outranks the Model
During execution:
Every real transition is checked against the theory's prediction
Mismatch → stop execution, void remaining plan
Return to deliberation with the mismatched transition as counterexample
Must revise model to account for it before planning resumes
This prevents "dreaming" — the agent cannot execute a plan in a model that doesn't match reality.
9. Key Design Principles
Table
Principle	Implementation
Programs as representations	world_model.py is text, not vectors — readable, diffable, editable
Complete-history verification	run_backtest checks ALL transitions, not just recent ones
Joint state+rule revision	Counterexample can indict representation OR rule
Separation of thinking and acting	Only commit_actions touches environment
Append-only ground truth	Timeline is immutable; model is provisional
Model-based planning	BFS in verified simulator costs zero real actions
Targeted exploration	Actions chosen to discriminate hypotheses, not random
10. Model Pairing & Fallback Strategy
Table
Pairing	Public Set Score	Notes
Claude Opus 4.8 + Fable 5	98.98%	Primary pair; 14 games Opus, 11 games Fable retained
GPT-5.6 Sol xhigh + Sol max	95.35%	Secondary pair; 15 games xhigh, 10 games max retained
Fallback rule: Run primary model first; if game scores < 80% RHAE, rerun with fallback; retain higher per-game score.
Controlled comparison (same models, different harness):
Claude Code (generic harness): 42.83%
Schema harness: 98.98%
Gain from process alone: +56.15 percentage points
11. Performance Characteristics
19/25 games score exactly 100% RHAE (Claude pairing)
Median game score: 100%
Residual error concentrated in 6 games (range 89.87–99.10)
Where model is exact: 1.6–5.0× fewer actions than human baseline
Example: M0R0 Level 4 — agent 42 actions vs human 500 actions
Why efficient: Agent pays real actions once to discover mechanism, then plans inside model for free. Humans rediscover mechanics through trial-and-error on every level.
12. Verification Status & Limitations
Table
Aspect	Status
Public set scores (98.98%, 95.35%)	Self-reported, not ARC Prize verified
Semi-private set performance	Unknown — no official measurement
Public→Semi-private extrapolation	Not justified (Sol Max drops ~5.5 points)
Full evaluation artifacts	Released for Claude pair; Sol pair pending
