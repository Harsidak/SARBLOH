# ARC-AGI-3 Comprehensive Research Dossier
**Date**: 2026-07-15 | **Goal**: Win ARC-AGI-3 with maximum RHAE

---

## Table of Contents
1. [Benchmark Specification](#1-benchmark-specification)
2. [RHAE Scoring Deep-Dive](#2-rhae-scoring-deep-dive)
3. [Current SOTA & Leaderboard](#3-current-sota--leaderboard)
4. [Key Research Papers](#4-key-research-papers)
5. [Top Agent Architectures](#5-top-agent-architectures)
6. [Techniques Directly Applicable to Our Agent](#6-techniques-directly-applicable-to-our-agent)
7. [Strategic Recommendations](#7-strategic-recommendations)
8. [GitHub Repositories](#8-github-repositories)

---

## 1. Benchmark Specification

**Source Paper**: "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence" -- ARC Prize Foundation
**arXiv**: [2603.24621](https://arxiv.org/abs/2603.24621)

### What ARC-AGI-3 Is
- **Interactive, turn-based environments** (NOT static grid puzzles like ARC-AGI-1/2)
- 135 environments, 25 games per competition run, difficulty-calibrated via human testers
- 64x64 grids, 16-color palette
- No instructions, no tutorials, no explicit goals
- Agent must: **Explore -> Model -> Set Goals -> Plan -> Act**

### Core Knowledge Priors (the only assumptions permitted)
| Prior | Description |
|:------|:------------|
| **Objectness** | Persistent entities that move, collide, interact |
| **Geometry & Topology** | Symmetry, rotation, inside/outside, connectivity |
| **Physics** | Momentum, gravity, bouncing |
| **Agentness** | Intent-driven behavior in the environment |

### Action Interface
- **Movement actions**: up, down, left, right
- **Click action**: coordinate-based cell selection
- **RESET**: restart current level (costs an action)
- Turn-based: 1 action per step, state changes only in response to actions

### Environment Design
- Multiple levels per game, increasing in complexity
- Knowledge from early levels often needed for later levels
- Games are **deterministic** -- same action from same state -> same result
- Some games: walk-avatar-to-goal mazes
- Some games: click-only (toggle, select, arrange)
- Some games: **rotation, timer, counter mechanics** (structurally different!)

---

## 2. RHAE Scoring Deep-Dive

### The Formula

```
level_score = min(1.15, human_actions / ai_actions)^2

game_score = Sum_completed(level_score_i * i) / Sum_{1..win_levels} i
```

### Why the Square Kills You

| AI/Human Ratio | Raw Ratio | Score (squared) |
|:--------------|:----------|:----------------|
| 1x (match human) | 1.00 | **100%** |
| 1.15x cap bonus | 1.15 | **100%** (capped) |
| 2x | 0.50 | **25%** |
| 3x | 0.33 | **11%** |
| 5x | 0.20 | **4%** |
| 10x | 0.10 | **1%** |

### Hard Cutoff
If your agent exceeds **5x the human action count** on any level, that level is **terminated with score zero**.

### Key Implications
1. **Information-gathering actions are scored** -- every exploratory move burns RHAE
2. **Plan through a model, not through the real environment**
3. **Later levels are weighted more** (the x i factor) -- failing level 8 costs far more than failing level 1
4. **Dead clicks / redundant actions are catastrophic**

---

## 3. Current SOTA & Leaderboard

### ARC-AGI-3 Official Leaderboard (July 2026)
| Model/Agent | Score | Notes |
|:-----------|:------|:------|
| GPT-5.6 Sol | **7.8%** | Best frontier LLM |
| Humans | **100%** | Full solve rate |
| EWM (GPT-5.5 high) | 58.12% RHAE* | 15/25 games solved (on public games only, research) |
| EWM (GPT-5.4 high) | 41.29% RHAE* | 8/25 games solved (research) |

*The EWM 58% is on public games only (not the hidden test set). On the actual Kaggle leaderboard, scores are much lower.

### Kaggle Competition Milestone Winners
| Place | Agent | Model | RHAE |
|:------|:------|:------|:-----|
| 1st | **"The Duck"** (Tufa Labs) | Qwen 3.6 27B FP8 | Milestone 1 winner |
| 2nd | **Reki** | Gemma-4-31B | Vision-LLM-as-policy |
| 3rd | **forge** | Vision-centric | Local model inference |

### Preview Competition (2025)
| Place | Agent | Approach | Score |
|:------|:------|:---------|:------|
| 1st | **StochasticGoose** | CNN + online RL | 12.58% |
| 2nd | **Blind Squirrel** | Training-free graph exploration | 6.71% |

---

## 4. Key Research Papers

### TIER 1: Directly Relevant to ARC-AGI-3

#### P1. "Executable World Models for ARC-AGI-3 in the Era of Coding Agents"
- **arXiv**: [2605.05138](https://arxiv.org/abs/2605.05138)
- **Author**: Sergey Rodionov
- **GitHub**: [astroseger/arc-3-agents-baseline1](https://github.com/astroseger/arc-3-agents-baseline1)
- **Result**: 58.12% mean RHAE (GPT-5.5), 15/25 public games
- **Key Ideas**:
  - Agent maintains **executable Python world model** -- code that encodes the game's dynamics
  - **Verification loop**: test world model against observed transitions before acting
  - **Refactoring loop**: simplify the model (MDL-like bias) as evidence accumulates
  - **Plan executor**: simulate plans in world model, execute in real game, halt on prediction mismatch
  - **No game-specific code** -- same agent/prompts for all games
- **Architecture**: Observe -> Model (Python code) -> Verify -> Refactor -> Plan -> Execute
- **Weakness**: Premature commitment to incorrect world model. Proposed fix: multi-model protocols, competing hypotheses

#### P2. "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence"
- **arXiv**: [2603.24621](https://arxiv.org/abs/2603.24621)
- **Authors**: ARC Prize Foundation
- **Key contribution**: Official benchmark specification, RHAE scoring definition, Core Knowledge priors, human baseline methodology

#### P3. "Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks"
- **Authors**: Rudakov, Shock, Cowley (Dec 2025)
- **Key Ideas**:
  - **Training-free** -- no neural network needed
  - Represents environment as a **directed graph**: nodes = unique states, edges = actions
  - Tracks visited states and tested actions
  - Prioritizes **shortest path to untested state-action pairs**
  - Segments visual frames into meaningful components
  - **Ranked 3rd on private leaderboard** in preview competition
- **Directly applicable**: This is essentially what your ExecPlanner's T2 directed frontier exploration does

#### P4. "Code World Models for General Game Playing"
- **arXiv**: [2510.04542](https://arxiv.org/abs/2510.04542)
- **Authors**: Google DeepMind
- **Key Ideas**:
  - LLM translates game rules -> executable Python code ("Code World Model")
  - CWM provides: state transition, legal move enumeration, termination checks
  - Enables **MCTS planning** through the code model
  - LLM also generates **heuristic value functions** and **hidden-state inference functions**
  - Outperformed Gemini 2.5 Pro in 9/10 games
- **Relevance**: The architecture pattern (code as world model + MCTS planning) is the blueprint for ARC-AGI-3

### TIER 2: Foundational / Inspirational

#### P5. "WorldCoder: Model-Based LLM Agent Framework"
- **Venue**: NeurIPS 2024
- **Key Ideas**:
  - LLM synthesizes Python world model from interaction data
  - Uses **optimistic planning** -- invents reward functions it believes achievable
  - Plans via depth-limited value iteration or MCTS through the code model
  - More sample-efficient than deep RL, more compute-efficient than reactive LLM agents

#### P6. "Less is More: Recursive Reasoning with Tiny Networks"
- **arXiv**: [2510.04871](https://arxiv.org/abs/2510.04871)
- **Author**: A. Jolicoeur-Martineau (Samsung SAIL)
- **GitHub**: [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- **Key Ideas**:
  - **7M parameter** two-layer network achieves 45% on ARC-AGI-1
  - **Recursive reasoning**: model updates its predicted answer over time (iterative refinement)
  - Heavy data augmentation (geometric/color transforms)
  - Test-time: report most common answer across augmented inputs
- **Relevance**: Shows tiny networks + recursion can compete. Inspires the "test-time adaptation" paradigm.

#### P7. "Active Inference as the Test-Time Scaling Law for Physical AI Agents"
- **arXiv**: [2606.22813](https://arxiv.org/abs/2606.22813)
- **Key Ideas**:
  - Active inference as dynamic policy updating at test time
  - Agent resolves prediction errors by updating beliefs/policies in real-time
  - "Learning beyond training" -- reinforcing new instances discovered during test-time
  - 36%+ improvement in inference efficiency over standard RL baselines

#### P8. Grid-JEPA (Joint Embedding Predictive Architecture for Grid Worlds)
- **Source**: HuggingFace community research
- **Key Ideas**:
  - Adapts I-JEPA for discrete grid worlds with 1x1 patch embeddings
  - Integrates RSSM (Dreamer-style) world model for persistent hidden state
  - Predicts transformations in **latent space** (not pixel reconstruction)
  - Per-task LoRA adapters for test-time training
  - Supports "imagination rollouts" for sequential planning

---

## 5. Top Agent Architectures

### Architecture A: "The Duck" (Tufa Labs) -- Milestone 1 Winner

```
Qwen 3.6 27B FP8 (Local inference, no internet)
|
Perception:
  - Rendered image of grid
  - Raw ASCII grid representation
  - Segmentation tool (zoom into regions)
|
Interaction:
  - Python REPL sandbox
  - Writes & executes Python code
  - Game state exposed as Python variables
|
Context Management:
  - "Infinite play via eviction"
  - Removes oldest messages, keeps system + most recent interactions
|
Philosophy:
  - Generic harness, LLM drives reasoning
  - No heavy heuristics
```

### Architecture B: Executable World Models (Rodionov)

```
External Controller (scripted, not LLM)
  - Passes observations to coding agent
  - Triggers refactoring prompts
  - Monitors level status
|
Coding Agent (GPT-5.5 / Codex CLI)
  - Fills in Python world model templates
  - world_model_engine.py (dynamics)
  - world_model_state_io.py (state I/O)
  - world_model_main_planner.py (planning)
|
Verification Infrastructure
  - World-model verifier (replay transitions)
  - Planner verifier (can plans reach goal?)
  - Plan executor (sim + real, halt on diff)
|
Refactoring Loop
  - Replace special cases with shared rules
  - MDL-like simplicity bias via prompt
  - Must pass verifier after each refactor
```

### Architecture C: Reki (2nd Place, Milestone 1)

```
Gemma-4-31B (Local, vision-language model)
|
Vision-Action Loop:
  - Renders recent frames as labeled images
  - Gemma returns JSON: {plan, actions[1-4]}
|
Reflection Memory:
  - Refreshed every ~10 steps
  - Model analyzes progress, refines strategy
|
Numpy Heuristic Fallback:
  - Prioritize small, rare-colored objects
  - "Dead-signature" heuristic:
    If clicking X never changes state ->
    mark X as dead, never click again
```

### Architecture D: Blind Squirrel (Training-Free Graph Explorer)

```
NO neural network, NO training
|
State Graph:
  - Nodes = unique environment frames
  - Edges = (state, action) -> next_state
  - Track all visited state-action pairs
|
Exploration Policy:
  - BFS to nearest untested state-action pair
  - Visual salience heuristic for action rank
  - Frame segmentation for object detection
|
Result: 6.71% score, outperformed most LLMs
```

### Architecture E: StochasticGoose (CNN + Online RL)

```
4-layer CNN (small, fast)
|
Predicts: P(action causes frame change)
Bias selection toward actions that yield
new state information
|
Online RL: learns transition dynamics
as it explores each game
|
Result: 12.58%, 18 levels solved
```

---

## 6. Techniques Directly Applicable to Our Agent

Given our constraints (no internet, local Qwen 27B as fallback, primary work done by no-LLM planners, 1500-action budget), here are the most impactful techniques ranked by feasibility and expected lift:

### CRITICAL: Fix Goal Inference (Your Current Bottleneck)

**Problem**: `_nearest_target` picks "nearest distinctive color" which is wrong for:
- Layered games with multiple colored objects
- Games where the modal color is walls, not background
- Games with non-walk-to-goal mechanics (rotation, timers, counters)

**Solution from Research**: Implement **span-based goal candidacy**:
1. **Structural fill exclusion**: Don't count colors that form the dominant connected fill (walls/background)
2. **Object segmentation**: Identify discrete objects (connected components of non-background color)
3. **Goal hypothesis testing**: Try multiple goal candidates, track which one leads to level completion
4. **Dead-signature tracking** (from Reki): If attempting to reach object X repeatedly fails to change state, deprioritize it

### CRITICAL: Game Type Classification

Before committing to a planner, classify the game:

```python
def classify_game(frames, action_results):
    """Classify game type from initial exploration."""

    # Click-only: frame changes on click but NOT on movement
    if click_causes_change and not movement_causes_change:
        return "CLICK_ONLY"   # -> ClickPlanner

    # Walk-to-goal: avatar moves, distinct goal region exists
    if avatar_detected and goal_candidate_exists:
        return "WALK_TO_GOAL" # -> ExecPlanner

    # Rotation/Timer: specific pixel patterns rotate or decrement
    if rotation_pattern_detected or counter_pattern_detected:
        return "MECHANICAL"   # -> needs frame-predicting world model

    return "UNKNOWN"          # -> graph exploration fallback
```

### HIGH VALUE: Directed State Graph (from Blind Squirrel)

Your ExecPlanner's T2 already does directed frontier exploration. Enhance it:
1. **Hash each frame** for O(1) state comparison
2. **Build explicit state graph** -- never revisit a (state, action) pair
3. **BFS to nearest untested (state, action)** -- guarantees no wasted exploration
4. **Track which actions cause state changes** -- immediately identifies "dead" actions

### HIGH VALUE: Plan-Then-Execute with Verification (from EWM)

Your T1 already does this! But EWM's specific innovations to adopt:
1. **Online mismatch detection**: After each action, compare predicted state vs observed state. If they diverge -> **stop immediately**, don't waste more actions
2. **Refactoring**: After failures, simplify the model (remove special cases, find shared rules)
3. **Multi-hypothesis tracking**: Maintain 2-3 competing world models, pick the one with fewest mismatch errors

### MEDIUM VALUE: Frame Delta Prediction (CNN-based)

Instead of full-frame prediction, predict **what changes**:
```python
delta = next_frame - current_frame  # Sparse! Most cells don't change
```
- Train a tiny CNN (4 layers, ~100K params) online as the agent explores
- Input: (current_frame, action) -> Output: predicted delta
- Use this to simulate plans without real actions
- This is what StochasticGoose proved works

### MEDIUM VALUE: Dead-Click Detection (from Reki)

```python
class DeadActionTracker:
    def __init__(self):
        self.action_results = {}  # (state_hash, action) -> [changed: bool]

    def record(self, state_hash, action, state_changed):
        key = (state_hash, action)
        self.action_results.setdefault(key, []).append(state_changed)

    def is_dead(self, state_hash, action, threshold=3):
        """If action never caused change in 3+ tries, it's dead."""
        key = (state_hash, action)
        results = self.action_results.get(key, [])
        return len(results) >= threshold and not any(results)
```

### MEDIUM VALUE: Context Window Eviction (from The Duck)

For the Qwen 27B fallback:
- Keep system prompt + most recent N messages
- Pop oldest messages when approaching context limit
- This enables "infinite play" without context degradation

### LOWER PRIORITY BUT POWERFUL: MCTS Planning

If you build a frame-predicting world model (CNN or code-based):
```
MCTS over the world model:
  Selection: UCT to pick promising action sequences
  Expansion: apply action in world model
  Simulation: rollout to depth D using simple heuristic
  Backpropagation: update action values
```
- All search happens in the model -- 0 real actions spent
- Only the best plan gets executed for real

---

## 7. Strategic Recommendations

### Priority Order for Implementation

```
P0 (DO NOW): Game-type classifier + goal inference fix
  -> Directly unblocks the 23 games scoring 0
  -> Low implementation cost, high expected lift

P1 (NEXT): Explicit state graph + dead-action tracking
  -> Eliminates all wasted exploration actions
  -> Your T2 already has the bones; extend it

P2 (BUILD): Frame-delta CNN for online transition learning
  -> Small CNN trained per-game as agent explores
  -> Enables planning through the model for ALL game types
  -> Handles rotation, timer, counter games that ExecPlanner can't

P3 (ENHANCE): Multi-hypothesis world model tracking
  -> Don't commit to one goal/model too early
  -> Track 2-3 candidates, pick based on prediction accuracy

P4 (OPTIMIZE): MCTS planning through learned world model
  -> Maximum RHAE: all exploration is unscored (in model)
  -> Only execute the shortest verified plan
```

### The Key Insight from All Research

> **"The fundamental loop is: Observe -> Model -> Verify -> Refactor -> Plan -> Execute.
> All computation in the model is FREE. Only real actions cost RHAE."**
> -- Rodionov (EWM paper)

Your ExecPlanner already implements this loop! The gap is:
1. **Goal inference** -- you're picking the wrong targets
2. **Game diversity** -- your displacement model can't represent rotation/timer games
3. **Wasted exploration** -- without a state graph, you revisit states

### What 100% RHAE Would Require

Based on the research, achieving 100% would require:
1. **Perfect game-type classification** (first few actions)
2. **Correct goal inference** (from level structure, not just nearest color)
3. **Accurate transition model** (predicts frame changes for all game types)
4. **Optimal planning** (finds the human-efficient path, not just any path)
5. **Zero wasted actions** (no exploration that doesn't serve the plan)

This is essentially: build the right world model fast, plan optimally through it, execute only the optimal plan.

---

## 8. GitHub Repositories

### Essential Repos

| Repository | Description | Link |
|:-----------|:------------|:-----|
| **EWM Baseline** | Executable World Models -- 58% RHAE on public games | [astroseger/arc-3-agents-baseline1](https://github.com/astroseger/arc-3-agents-baseline1) |
| **ARC-AGI-3 Starter** | Official Kaggle starter kit | [arcprize/ARC-AGI-3-Kaggle-Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter) |
| **The Duck Harness** | Tufa Labs milestone 1 winner | [Tufalabs/duck-harness](https://github.com/Tufalabs/duck-harness) |
| **Tufa Labs ARC-AGI-3** | StochasticGoose + The Duck codebase | [tufa-labs/arc-agi-3](https://github.com/tufa-labs/arc-agi-3) |
| **Tiny Recursive Models** | 7M param, 45% on ARC-AGI-1 | [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) |

### Papers (arXiv Links)

| Paper | arXiv ID |
|:------|:---------|
| ARC-AGI-3 Benchmark Specification | [2603.24621](https://arxiv.org/abs/2603.24621) |
| Executable World Models for ARC-AGI-3 | [2605.05138](https://arxiv.org/abs/2605.05138) |
| Code World Models for General Game Playing | [2510.04542](https://arxiv.org/abs/2510.04542) |
| Less is More: Recursive Reasoning with Tiny Networks | [2510.04871](https://arxiv.org/abs/2510.04871) |
| Active Inference as Test-Time Scaling Law | [2606.22813](https://arxiv.org/abs/2606.22813) |
| Generating Code World Models with MCTS | [2405.15383](https://arxiv.org/abs/2405.15383) |

---

## Appendix A: The 25 Public Games -- What We Know

From the EWM paper results, game difficulty varies enormously:
- **Easy** (solved by EWM with high RHAE): ft09, tn36, and ~13 others
- **Hard** (0% or near-0%): ls20 (rotation+timer), tr87, tu93, and ~10 others
- **Click-only**: ~10 of 25 games -- handled by ClickPlanner

The critical gap is the **movement games that aren't simple walk-to-goal**:
- ls20: Rotation + step-timer (avatar 3px, modal color is walls)
- tr87, tu93: Likely similar mechanical diversity

**These need a frame-predicting world model, not displacement+BFS.**

---

## Appendix B: Competition Constraints Reminder

| Constraint | Value |
|:-----------|:------|
| Internet | **None** (offline only) |
| Model | Local only (Qwen 27B on Kaggle GPU) |
| Runtime | 6 hours total |
| Action budget | 1500 per game (practical), 5x human = hard cutoff per level |
| Games | 25 per run |
| RESET | Costs an action, restarts current level |
| Memory | No cross-game state persistence |
