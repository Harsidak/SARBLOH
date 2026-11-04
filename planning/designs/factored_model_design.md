# Factored / History-Window PatchWorldModel — Design

## Problem

tu93 and tr87 have **hidden, history-dependent state**:

- **tu93**: track-navigation. Moves are valid only onto track pixels (value 2).
  `sdiguidlbg` stores the avatar's FULL rotation history; phase-2 world updates
  consume it. The same visible grid can permit different actions depending on
  prior history. Result: StateGraph exhausts 37 visible states with 0
  nondeterminism conflicts, finds no reward, yet human L1 = 19 actions.

- **tr87**: sprite × cursor product space grows ~0.5 new states/step. Masking
  works but the raw state count overwhelms exploration.

Single-frame PatchWorldModel (`predict_noop` from `(grid, action)`) provably
cannot disambiguate these because the same 5×5 patch + action can yield
different outcomes depending on history.

## Approach: Frame-Stack Conditioning (PWM only)

Apply temporal conditioning to **PatchWorldModel only** — NOT to StateGraph hash.

### Why PWM-only

- StateGraph's value is **exact reproducibility** of single-frame transitions.
  Making it k-frame-keyed fragments the graph (same visible state + different
  history = different nodes), defeating BFS on mechanical games.
- The **prediction layer** is where temporal reasoning belongs: "will this action
  be effective given recent history?" PWM gains temporal context; StateGraph
  stays exact-per-visible-state for navigation.
- Memory: temporal patches grow by factor k but the space is sparse.

### Temporal Patch Format

Current: `(action, patch_5x5.tobytes()) -> [eff_count, ineff_count]`

Proposed: `(action, (patch_t-k, patch_t-k+1, ..., patch_t).tobytes()) -> [eff, ineff]`

For each unmasked cell at position (r, c):
```
temporal_patch(r, c) = concat([grid_{t-i}[r-K:r+K+1, c-K:c+K+1] for i in range(k, -1, -1)])
```
where K=2 (5×5 spatial window) and k = temporal depth.

**Flattened size**: (k+1) × 25 int32 values per cell. With k=4, that's 125
values × 4 bytes = 500 bytes per patch key. The stats dict stays sparse because
most cells don't change between frames (temporal patches collapse to repeated
identical 5×5 subpatches for static regions → same key across time steps).

### Window Size k

- tu93 human L1 = 19 actions. Most hidden-state effects (rotation memory)
  should surface within 4-8 actions of context.
- **Start with k=4**. Profile: if too few patches converge, try k=2; if
  disambiguation is poor, try k=8.
- **Memory budget**: k=4 means storing the last 5 grids (4 history + current).
  At 64×64 × 4 bytes = 16KB per grid → 80KB ring buffer. Trivial.

### Integration Points

#### PatchWorldModel changes

```python
class PatchWorldModel:
    HISTORY_K = 4                    # temporal depth (0 = current only = v2)

    def __init__(self):
        self.stats: Dict[Tuple[str, bytes], list] = {}
        self.trained = 0
        self._history: deque = deque(maxlen=self.HISTORY_K + 1)

    def _temporal_patches(self, mask):
        \"\"\"Stack the last k+1 grids' 5×5 patches into temporal patches.\"\"\"
        grids = list(self._history)
        # Pad with the earliest available if history is short
        while len(grids) < self.HISTORY_K + 1:
            grids.insert(0, grids[0])
        # Per-cell: concat k+1 spatial patches → one temporal patch
        all_patches = [self._patches(g, mask) for g in grids]
        return np.concatenate(all_patches, axis=1)  # (n_cells, (k+1)*25)

    def push_frame(self, grid):
        \"\"\"Called every step to maintain the history ring buffer.\"\"\"
        self._history.append(grid.copy())

    def learn(self, action, nxt, mask=None):
        # prev is the SECOND-TO-LAST in history; nxt is the latest
        if len(self._history) < 2:
            return
        prev = self._history[-2]
        # ... rest same as v2 but using _temporal_patches ...

    def predict_noop(self, action, mask=None):
        # Uses _temporal_patches on current history
        # ... same known + witness logic as v2 ...
```

#### MyAgent integration

```python
def choose_action(self, frames, latest_frame):
    grid = self._parse_grid(latest_frame, frames)
    self.pwm.push_frame(grid)  # maintain history ring buffer
    # ... rest unchanged ...
```

#### StateGraph: NO CHANGES

StateGraph continues to hash single frames. The noop predicate passed to
`nearest_untested` now uses temporal PWM internally but the StateGraph
interface is unchanged:
```python
noop = lambda g, a: self.pwm.predict_noop(a, self.sgraph._mask)
```

### Level/Reset Handling

- **Level-up**: `pwm._history.clear()` — new layout, old history is irrelevant.
- **GAME_OVER reset**: `pwm._history.clear()` — the level restarts from initial
  state, so prior history is stale. Stats (learned patches) persist (same layout).

### Verification Plan

1. **tu93 hidden-state test**: Run tu93 with k=4. Compare the temporal patch
   stats to the single-frame stats:
   - Do temporal patches that were always-ineffective under single-frame
     hashing split into effective/ineffective under temporal hashing?
   - Does `predict_noop` start firing (TRUE strict > 0)?

2. **Regression gates**: self-tests, reachgoal 3/3, keydoor 2/2, lp85 ≥1/8.
   Temporal PWM must be a strict improvement — never worse than single-frame.

3. **Fallback**: if temporal patches are too sparse to accumulate MIN_SEEN,
   fall back to single-frame prediction for cells whose temporal patch is
   unknown but single-frame patch is known. This hybrid approach ensures
   no regression.

## Open Work (after this design is validated)

- **Factored state for tr87**: sprite × cursor decomposition. The avatar and
  the cursor are independent objects — a factored model that hashes them
  separately would compress tr87's state space exponentially. This is a
  separate design from frame-stacking.
- **Frame-delta encoding**: instead of stacking raw frames, stack DELTAS
  (frame_t - frame_{t-1}). This compresses static regions even more and
  highlights the temporal signal (which cells changed recently).
