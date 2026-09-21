# ARC-AGI-3 Agent Improvement Plan — Based on Research

## Research Summary

Full research dossier saved at:
- [ARC_AGI_3_COMPREHENSIVE_RESEARCH.md](file:///c:/Users/Banwa/Desktop/IWT/CODING/PROJECTS/Kaggle/ARC_AGI%20EXPERMENTATIONS/Research/ARC_AGI_3_COMPREHENSIVE_RESEARCH.md)
- [PAPERS_INDEX.md](file:///c:/Users/Banwa/Desktop/IWT/CODING/PROJECTS/Kaggle/ARC_AGI%20EXPERMENTATIONS/Research/PAPERS_INDEX.md)

Key findings from 8 papers, 5 top agent architectures, and 5 GitHub repos:

| Source | Score | Key Technique |
|:-------|:------|:-------------|
| EWM (Rodionov) | **58% RHAE** | Executable Python world model + verify + refactor |
| The Duck (Tufa Labs) | Milestone #1 | Qwen 27B + Python REPL + context eviction |
| Blind Squirrel | 6.7% | Training-free directed state graph |
| StochasticGoose | 12.6% | CNN predicting P(action causes change) |
| Reki | 2nd place | Vision-LLM + dead-signature heuristic |

**Our ExecPlanner already implements the core loop** (Observe → Model → Verify → Plan → Execute). The gaps are:
1. Goal inference picks wrong targets
2. Displacement model can't handle rotation/timer games
3. No state graph → wasted exploration revisiting states

---

## Open Questions

> [!IMPORTANT]
> **Should we run the tr87/tu93 structural probe first?**
> This was the work in progress before the research sweep. It tells us whether those games are walk-to-goal (fixable with goal inference) or mechanically different (needs frame-prediction CNN).
> **My recommendation**: Yes, do it — it's ~15 minutes and directly determines whether we go P0-only or P0+P2.

> [!IMPORTANT]
> **Budget allocation**: The 1500-action budget means we can afford ~5-10 "free" exploration actions to classify the game type, then must commit to a plan. How many exploration actions should the classifier use before committing to a planner?

---

## Proposed Changes

### P0 (DO NOW): Game-Type Classifier + Goal Inference Fix

This directly unblocks the 23 games currently scoring 0.

#### [MODIFY] my_agent.py — Add game-type classifier

After the first few actions, classify into:
- `CLICK_ONLY` → existing ClickPlanner
- `WALK_TO_GOAL` → ExecPlanner with fixed goal inference
- `MECHANICAL` → frame-prediction world model (P2)
- `UNKNOWN` → graph exploration fallback

#### [MODIFY] my_agent.py — Fix `_nearest_target` goal inference

Replace "nearest distinctive color" with **span-based goal candidacy**:
1. Flood-fill to identify the dominant connected region (walls/background)
2. Exclude that color from goal candidates
3. Identify discrete objects via connected components
4. Rank by: (a) size (small = likely interactable), (b) rarity, (c) distance
5. Support **multi-goal hypothesis tracking** — try top-3 candidates

---

### P1 (NEXT): Explicit State Graph + Dead-Action Tracking

#### [MODIFY] my_agent.py — Add `StateGraph` class

```python
class StateGraph:
    """Directed graph of environment states and transitions."""
    def __init__(self):
        self.states = {}      # hash -> frame
        self.edges = {}       # (hash, action) -> next_hash
        self.untested = set() # (hash, action) pairs not yet tried
    
    def add_state(self, frame):
        h = hash(frame.tobytes())
        if h not in self.states:
            self.states[h] = frame
            for action in ALL_ACTIONS:
                self.untested.add((h, action))
        return h
    
    def record_transition(self, from_hash, action, to_hash):
        self.edges[(from_hash, action)] = to_hash
        self.untested.discard((from_hash, action))
    
    def nearest_untested(self, current_hash):
        """BFS from current state to nearest untested action."""
        # Returns path of actions to get there + the untested action
```

#### [MODIFY] my_agent.py — Add `DeadActionTracker`

Track (state, action) pairs that never cause state changes. After 3 failures, mark as dead.

---

### P2 (BUILD): Frame-Delta CNN for Online Transition Learning

This handles rotation, timer, and counter games that ExecPlanner can't model.

#### [NEW] Inline CNN in my_agent.py

- 4-layer CNN, ~100K parameters
- Input: (current_frame_tensor, action_one_hot) → Output: predicted_delta
- Trained online from observed transitions during exploration
- Used for planning: simulate action sequences, compare to goal state

> [!WARNING]
> This is the most complex change. May need to be built incrementally:
> 1. First: frame hashing + state comparison (P1)
> 2. Then: delta storage from observed transitions
> 3. Then: CNN training loop
> 4. Finally: MCTS planning through the CNN

---

### P3 (ENHANCE): Multi-Hypothesis World Model

Don't commit to a single goal or model. Track 2-3 hypotheses:
- **Goal candidates**: top-3 objects by rarity/size/distance
- **Transition models**: displacement vs. frame-prediction
- **Selection**: pick the hypothesis with fewest prediction errors after N steps

---

### P4 (OPTIMIZE): MCTS Planning Through World Model

Once we have a working frame-prediction model (P2), add MCTS:
- Selection: UCT (Upper Confidence Trees)
- Expansion: apply action in world model
- Simulation: rollout to depth D
- Backpropagation: update action values
- **All search is FREE** — 0 real actions consumed

---

## Verification Plan

### Non-Negotiable Regression Suite
Every change must preserve:
- `reachgoal` 3/3
- `keydoor` 2/2
- `lp85` >= 1/8
- Self-tests pass

### New Validation
- Run structural probe on tr87/tu93 (before P0)
- Run full 25-game RHAE after each priority tier
- Compare against run001 baseline (cn04 + lp85 only)

### Expected Lift
| Tier | Expected Games Gained | RHAE Impact |
|:-----|:---------------------|:------------|
| P0 | +3-5 walk-to-goal games | +5-15% mean RHAE |
| P1 | +2-3 via efficiency gains | +3-8% mean RHAE |
| P2 | +5-8 mechanical games | +15-25% mean RHAE |
| P3+P4 | +2-4 optimization | +5-10% mean RHAE |
