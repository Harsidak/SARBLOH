# ULTIMATE PROMPT: ARC-AGI-3 COMPETITION AGENT
# VERSION: 3.0 (POST-MORTEM CORRECTED)
# COMPETITION: ARC-AGI-3 (Interactive Hidden-State Grid World)
# COMPUTE: 1x NVIDIA RTX 6000 (48GB VRAM), 9-hour wall-clock limit
# LLM: Qwen 3.6 27B via vLLM (FP8), localhost:8000
# ENVIRONMENT: Kaggle Notebook (offline after setup, internet during setup)
# SCORING: RHAE = human_actions / agent_actions (higher is better)

---

## 1. EXECUTIVE SUMMARY

Build an agent that plays an interactive puzzle game where the rules are hidden. 
The agent must DISCOVER rules by experimenting, then EXPLOIT them to solve efficiently.

The winning strategy is NOT to call the LLM on every step. The LLM is an ORACLE 
used sparingly (~3-5 times per level) to generate Python world models. The agent 
must run fast Python MCTS and exploration logic for 99% of its runtime.

---

## 2. HARD CONSTRAINTS (Non-negotiable)

| Constraint | Value | Violation Consequence |
|---|---|---|
| Total runtime | 9 hours wall-clock | Auto-submission at cutoff |
| Actions per level | 1,500 hard cap | Level forfeited if exceeded |
| GPU VRAM | 48GB total | OOM = crash = zero score |
| LLM calls | Max ~10 per level ideally 3-5 | Too many = timeout before finish |
| LLM timeout | 5 seconds per call | Timeout must fallback gracefully |
| Internet | Available during setup only | Model must be pre-loaded or from Kaggle Dataset |
| Notebook state | Fragile (kernel restarts) | Must checkpoint every 15 minutes |

---

## 3. ARCHITECTURE: THE 6 COMPONENTS

You must implement exactly these 6 components. No more, no less.

### COMPONENT 1: EYES (StateEncoder)
**Mission:** Compress any grid into <500 tokens of structured text.

**Requirements:**
- `detect_background(grid)` → int: Detect background color dynamically. 
  *DO NOT hardcode 0.* Use border-pixel majority or global frequency.
- `flood_fill_objects(grid, bg_color, connectivity=4)` → List[GridObject]:
  Extract connected components. Each object has: color, size, bbox, center, shape_class.
- `classify_shape(mask, ...)` → str: Classify as single_pixel, horizontal_line, 
  vertical_line, square, rectangle, hollow_shape, irregular_blob.
- `encode_state(grid, level=1)` → str:
  * L0: Global stats (size, bg, color counts, symmetry) → ~30 tokens
  * L1: L0 + object list → ~100 tokens
  * L2: L1 + RLE-encoded pixel detail for requested objects → ~300 tokens
  * L3: Full grid only if ≤10x10, else downsampled → ~4000 tokens
- `encode_transition(prev, next, action)` → str:
  Describe diff: pixel count changed, color transitions, movement vectors, 
  merge/split detection. Must be <100 tokens.

**CRITICAL:** All text output must use ACTUAL newlines (`\n`), not escaped 
newlines (`\\n`). The LLM reads this text. Escaped newlines produce garbled prompts.

### COMPONENT 2: MEMORY (MemoryManager)
**Mission:** Store history without exceeding LLM context budget or VRAM.

**Requirements:**
- 4-tier memory OUTSIDE the LLM context:
  * L0 Working: Last 20 transitions (loaded into prompt)
  * L1 Episodes: Summaries of old transitions (loaded into prompt)
  * L2 Hypothesis: Current rule guess (loaded into prompt)
  * L3 Archive: Vector embeddings of past levels (retrieved on demand)
- Context budget: 12,000 tokens for the LLM prompt.
- Compression trigger: When prompt exceeds 90% budget, compress L0→L1.
- L1 compression: MUST be asynchronous or batched. 
  *DO NOT block the action loop waiting for LLM summarization.* 
  If LLM is busy, use a simple rule-based summary (e.g., "Steps 1-10: tested ACTION1-ACTION5, no reward").
- L3 Archive: Use sentence-transformers (all-MiniLM-L6-v2) on CPU. 
  FAISS or flat numpy index. Retrieval <100ms.
- `get_context_for_llm(current_state_desc)` → str: Assembles prompt sections.

**CRITICAL:** The action loop must never wait for memory compression. 
If the LLM summarizer times out, fall back to a hardcoded template summary.

### COMPONENT 3: BRAIN (LLMClient)
**Mission:** Interface to Qwen via vLLM. Used sparingly.

**Requirements:**
- `query(prompt, temperature, max_tokens)` → str: 
  HTTP POST to localhost:8000/v1/chat/completions. JSON mode optional.
  On timeout or error: return empty string, log error, disable LLM for 60s.
- `generate_world_model(observations: str)` → WorldModel:
  Prompt: "Given these state transitions, write a Python function 
  `def transition(state_grid, action_str)` that predicts the next grid.
  Return JSON with 'hypothesis' (NL) and 'code' (Python)."
  *Temperature 0.2, max_tokens 1024.*
  On failure: return identity fallback (`def transition(s,a): return s.copy()`).
- `reflect_on_failure(trajectory, reason)` → str:
  Prompt LLM to critique failure and update hypothesis. 
  *Temperature 0.5. Called only on GAME_OVER or when stuck.*

**EXPLICITLY DO NOT implement:**
- `explore()` — Do NOT ask LLM to generate actions during exploration. 
  Exploration is random/curiosity-driven, not LLM-driven.
- `evaluate()` — Do NOT ask LLM to score states for MCTS. 
  The value function must come from a small neural network or heuristics, 
  not a 27B model call per MCTS node.

**CRITICAL:** The LLM is called at most:
1. Once per level to generate initial world model (~step 30-50)
2. Once per level to refine world model if accuracy <60%
3. Once per level on GAME_OVER to reflect
4. Optionally once more if stuck for >100 steps

### COMPONENT 4: PLANNER (MCTSPlanner + WorldModelManager)
**Mission:** Select actions efficiently using a learned internal simulator.

**Requirements:**

**WorldModelManager:**
- `update_from_observations(transitions: List[Transition])`:
  Format last 30 transitions as text. Call LLM.generate_world_model().
  Compile the returned Python code with `exec()` in a restricted namespace 
  (`{'np': np}` only).
  Verify the compiled function immediately: run it on the last 5 observed 
  (state, action) pairs and compare predicted next_state to actual next_state.
  If exact match on ≥4/5: promote to ACTIVE_MODEL.
  If not: discard, keep previous model (or identity).
- `predict(state, action)` → next_state:
  If active_model exists and compiled: run it.
  Else: return state.copy() (identity fallback).
- `accuracy`: Rolling average of exact-match predictions over last 20 steps.

**MCTSPlanner:**
- `search(root_grid, valid_actions, n_iterations=100)` → action_str:
  Standard MCTS with UCT selection.
  **EXPANSION:** Use the policy network (Component 5) to generate action priors.
  **SIMULATION:** Use WorldModelManager.predict() for rollouts. 
    *ZERO LLM calls during simulation.*
  **EVALUATION:** Use a small value network (Component 5) or simple heuristic 
    (e.g., state novelty + reward signal) to evaluate leaf nodes.
  **BACKPROP:** Update visit counts and values.
  Return the most-visited child action.

**CRITICAL:** MCTS must complete 100 iterations in <2 seconds. 
If the world model is slow, reduce iterations to 50. 
If no world model exists, MCTS uses random rollouts (still no LLM calls).

**CRITICAL:** The value network MUST be trained. If you include a value head 
in the policy network, you MUST train it with TD-lambda or MCTS self-play 
bootstrapping. A randomly initialized value head makes MCTS worse than 
random selection.

### COMPONENT 5: LEARNER (ExplorationPolicy + ICM + StateCounter)
**Mission:** Drive exploration when the world model is weak. 
**EXPLICITLY DELETE GRPO. It does not work for this problem.**

**Requirements:**

**ExplorationPolicy (CNN + Actor):**
- Input: (1, 64, 64) grid tensor
- Encoder: Small CNN (3-4 conv layers) → 256-dim feature vector
- Actor: Linear(256, num_base_actions) → action logits
- **NO value head unless you train it.** If you can't train it properly, 
  delete it and use visit-count + reward heuristic for MCTS leaf evaluation.
- Forward pass must be <10ms on GPU.

**ICM (Intrinsic Curiosity Module):**
- Forward model: predict next feature from (current_feature + action)
- Inverse model: predict action from (current_feature, next_feature)
- `intrinsic_reward(state_feat, next_feat, action)` → float:
  MSE of forward prediction. Used as bonus during exploration.
- Train ICM online with Adam, lr=1e-4, after every 10 steps.

**StateCounter:**
- `simhash(grid)` → int: Downsample grid to 16x16, hash bytes.
- `get_bonus(grid)` → float: 1/sqrt(visit_count). 
  Pure count-based novelty. No neural network.

**Combined Reward during EXPLORATION:**
total_reward = extrinsic_reward + 0.3 * icm_reward + 0.2 * count_bonus
plain

**CRITICAL:** Do NOT use GRPO. Do NOT use PPO with old log probs. 
The policy is trained with simple REINFORCE or A2C on recent transitions, 
or just use the policy as a prior for MCTS without online updates. 
The simplest working approach: train the policy offline on synthetic data, 
or don't train it online at all — just use it as a fixed prior.

**CRITICAL:** If you use online training, the policy must be updated 
AFTER the level is complete, not during. Updating during the level makes 
MCTS use a moving target and destabilizes the value function.

### COMPONENT 6: ACT (MyAgent)
**Mission:** Orchestrate everything. Implement the `Agent` interface.

**Requirements:**

**Interface methods:**
- `is_done(frames, latest_frame)` → bool:
  Return True if `latest_frame.state == WIN` or `action_counter >= 1500`.
- `choose_action(frames, latest_frame)` → GameAction:
  The main loop. Must complete in <100ms (excluding occasional LLM calls).

**Main loop logic:**
Parse grid from latest_frame
If last_grid exists:
a. Compute diff, store transition in L0 memory
b. Compute reward (1.0 if WIN else 0.0)
c. Update ICM with (last_state, action, current_state)
d. Update StateCounter
e. If world_model.active_model exists:
Predict next_state from world_model
Compare to actual. Update rolling accuracy.
If accuracy > 80% for 10 steps: set mode = EXPLOIT
If accuracy < 60%: set mode = EXPLORE
f. If action_counter % 30 == 0 and mode == EXPLORE:
Trigger world_model.update_from_observations() in BACKGROUND THREAD
If latest_frame.state == GAME_OVER:
Call LLM.reflect_on_failure()
Return RESET
If mode == EXPLOIT and world_model.active_model exists:
Run MCTS with world_model rollouts
Return best action
Else (mode == EXPLORE):
Sample action from ExplorationPolicy (with entropy)
OR use epsilon-greedy random + curiosity bonus
Return action
Every 15 minutes: checkpoint agent state to disk
plain

**CRITICAL:** The winning action's reward must be captured. 
When `is_done` returns True because of WIN, the environment may not call 
`choose_action` again. Therefore, `choose_action` must detect WIN on the 
*current* frame and assign the reward to the *previous* action's rollout 
BEFORE returning. Do not wait for the next call.

**CRITICAL:** Action6 coordinates. If the action space includes spatial 
actions (ACTION6 with x,y), the policy must output:
- Base action logits: [ACTION1, ACTION2, ..., ACTION7]
- Spatial coordinates: only relevant if base action is ACTION6
MCTS must treat ACTION6 as ONE node, not 4096 nodes. The coordinate is 
a parameter, not a separate action branch.

**CRITICAL:** Parse the actual valid actions from the environment. 
Do not assume ACTION1-ACTION7 exist. Query `GameAction` enum or environment 
metadata if available.

---

## 4. EXECUTION PHASES (The 9-Hour Budget)

| Phase | Time | Activity | LLM Calls |
|---|---|---|---|
| BOOT | 0-10 min | Load model, init components | 0 |
| LEVEL EXPLORE | 10-40 min | Random/curiosity exploration, build dataset | 0 |
| LEVEL MODEL | 40-50 min | Generate & verify world model | 1 |
| LEVEL EXPLOIT | 50-120 min | MCTS with world model, solve level | 0 |
| CHECKPOINT | Every 15 min | Save state to disk | 0 |
| EMERGENCY | Any | If stuck >100 steps or GAME_OVER | 1 (reflect) |

**Per-level target:** 50-200 actions, 10-60 minutes.
**Total levels target:** 10-15 levels in 9 hours.

---

## 5. VULNERABILITY ANALYSIS (Known Failure Modes)

### V1: LLM Call Frequency Death Spiral
**Symptom:** Agent makes 2+ LLM calls per action. 5s × 2 × 1500 = 4.1 hours per level.
**Mitigation:** Hard limit LLM calls to 5 per level. Use a counter. 
If exceeded, agent must operate without LLM for remainder of level.

### V2: Identity Model False Confidence
**Symptom:** LLM fails to generate world model → falls back to identity. 
Environment ignores invalid actions (no change). Identity model is 70% "accurate." 
Agent switches to EXPLOIT mode and does nothing.
**Mitigation:** Accuracy must be computed only on actions that actually 
changed the state. If action caused no change, exclude from accuracy calculation. 
Also require accuracy >80% on ≥10 *distinct* state transitions, not just 10 steps.

### V3: Missing Winning Action Reward
**Symptom:** Final winning action is stored in replay buffer with reward=0.0. 
Environment calls is_done → True, then resets. assign_reward() is never called.
**Mitigation:** In `choose_action`, if `latest_frame.state == WIN`, 
immediately assign reward=1.0 to the pending rollout BEFORE returning.

### V4: Context Window Blocking
**Symptom:** Memory compression calls LLM synchronously. Action loop blocks 
for 5 seconds. Agent burns 20% of time budget waiting.
**Mitigation:** Compression must be async or rule-based. Never block the 
action loop for LLM summarization.

### V5: Value Network Randomness
**Symptom:** MCTS uses untrained value head. Leaf evaluations are random noise. 
MCTS performs worse than greedy.
**Mitigation:** Either (a) train value head with TD-lambda on real 
transitions, or (b) delete value head and use simple heuristic: 
leaf_value = reward_signal + 0.1 * state_novelty.

### V6: ICM Moving Target
**Symptom:** Policy encoder is updated during level, changing feature space. 
ICM forward model was trained on old features. Prediction error spikes. 
Agent thinks everything is novel.
**Mitigation:** Freeze encoder during level. Only update actor head 
(if at all). Or don't update policy online.

### V7: Kaggle Kernel Death
**Symptom:** Notebook kernel restarts. vLLM process dies. Agent loses all 
state including world model and memory.
**Mitigation:** Checkpoint every 15 minutes to `/kaggle/working/`. 
On restart, reload checkpoint and model weights from Kaggle Dataset path.

### V8: OOM During LLM + Training
**Symptom:** vLLM uses 30GB for model + 12GB for KV cache. ICM + policy 
use 4GB. During GRPO update (if implemented), gradient computation pushes 
total over 48GB.
**Mitigation:** Do NOT implement GRPO. Do NOT do large backward passes 
during the level. If training is needed, do it between levels or after 
competition.

### V9: Action Coordinate Mismatch
**Symptom:** MCTS selects "ACTION6_x5_y10" but environment expects 
`GameAction.ACTION6` with coordinates set separately. Parse fails or 
coordinates are stale.
**Mitigation:** Store coordinates as agent instance variables. 
`parse_action()` sets `self.action_x` and `self.action_y`. 
Verify environment reads these before step execution.

### V10: Background Thread Exceptions
**Symptom:** `threading.Thread(target=world_model.update...)` crashes silently. 
World model is never updated. Agent explores forever.
**Mitigation:** Background threads must have try/except blocks and set 
a `self.world_model_ready` flag. Main loop checks flag before switching 
to EXPLOIT mode.

---

## 6. ANTI-PATTERNS (What NOT to Build)

| Anti-Pattern | Why It Fails |
|---|---|
| **GRPO / PPO with old log probs** | Action space and rewards are non-stationary across levels. Importance sampling explodes. |
| **LLM-evaluated MCTS** | 5s × 100 nodes = 8 minutes per action. Budget dead in first level. |
| **Flat MLP policy on 4096-dim vector** | Destroys spatial structure. Needs 10x more data than CNN. |
| **Online policy updates during level** | MCTS uses moving target. Value function becomes meaningless. |
| **Synchronous LLM summarization** | Blocks action loop. Burns 20% of time budget. |
| **Hardcoded background color = 0** | Fails on levels where 0 is a meaningful color. |
| **Assuming fixed action space** | Environment may have level-specific actions. |

---

## 7. TESTING & ACCEPTANCE CRITERIA

Before declaring the agent complete, verify:

1. **EYES:** Encode 64x64 random grid → <500 tokens. Human can sketch it.
2. **MEMORY:** Store 1000 transitions. Context prompt <12k tokens. Retrieval <100ms.
3. **LLM:** Generate world model from 10 synthetic transitions. Function compiles. 
   Predicts next state correctly on 8/10 held-out transitions.
4. **MCTS:** Run 100 iterations on 10x10 grid in <2 seconds. No LLM calls.
5. **EXPLOIT GATE:** With identity model, agent stays in EXPLORE mode. 
   With accurate model, switches to EXPLOIT.
6. **INTEGRATION:** Agent plays a synthetic "move red square to target" game. 
   Discovers rule in <50 actions. Solves in <20 additional actions.
7. **KAGGLE:** Notebook runs end-to-end without internet after initial setup. 
   Checkpoint file created and reloadable.

---

## 8. DELIVERABLES
arc_agi3_agent/
├── agent/
│   ├── init.py
│   ├── eyes.py          # Component 1
│   ├── memory.py        # Component 2
│   ├── brain.py         # Component 3
│   ├── planner.py       # Component 4 (MCTS + WorldModel)
│   ├── learner.py       # Component 5 (CNN + ICM + Counter)
│   └── act.py           # Component 6 (MyAgent)
├── notebooks/
│   ├── 01_setup.ipynb   # Download model, start vLLM
│   ├── 02_test.ipynb     # Unit tests for all components
│   └── 03_submit.ipynb   # Competition submission
├── checkpoints/         # Created at runtime
├── config.yaml          # All hyperparameters
└── requirements.txt
plain

---

## 9. FINAL REMINDER

The goal is NOT to build a fancy RL system. The goal is to build a 
**scientific method machine** that:

1. Experiments cheaply (Python, no LLM)
2. Generalizes observations into a Python world model (LLM, rarely)
3. Plans efficiently using the world model (MCTS, fast)
4. Solves the level in fewer actions than a human

**Simplicity beats complexity. Speed beats sophistication.**