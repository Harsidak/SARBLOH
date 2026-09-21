"""
ARC-AGI-3 Agent  -  ITERATION 6 (Offline, in-process, no network LLM)

HARNESS REALITY (from v-o-i-d.ipynb):
  * This whole file is written to /kaggle/working/my_agent.py and run as `myagent`
    inside ARC-AGI-3-Agents. The framework talks to the game at http://gateway:8001;
    OUR code must make NO network calls of its own. There is NO vLLM/localhost.
  * All weights are mounted from a Kaggle dataset and loaded IN-PROCESS from local
    paths (see CONFIG). sentence-transformers + faiss-cpu are installed from wheels.
  * Hardware: 1x RTX 6000 Pro (Blackwell, 96GB). Qwen ~27B loads int8 (bitsandbytes)
    when available, else bf16 (~54GB, still fits). It is PRELOADED in a background
    thread at agent init so the first budgeted call doesn't stall the action loop.

DESIGN (mapped to roadmap.md):
  * World models are a VERIFIABLE DSL. PRIMARY synthesis is ENUMERATIVE and needs
    NO LLM and NO GPU -- it searches a small DSL program space against real,
    state-changing observations and only trusts a program that reproduces them
    (roadmap C4, V2, S9). Object-scoped ops (translate_color / slide_color / ...)
    cover the "a distinct-colored avatar moves" family that dominates ARC-AGI-3.
  * The local Qwen is a BUDGETED fallback (<=5 calls/level, roadmap V1): launched
    only when already loaded, generation hard-capped by max_time, and the launcher
    thread is covered by a watchdog + generation counter so a stall can never
    wedge the agent. LLM candidates are verified against FRESH transitions.
  * GRPO/PPO deleted (roadmap C5/S6/V8). No online training, no value head (V5).
    Exploration = count-based novelty with per-level epsilon decay.
  * MCTS: heuristic leaf value (novelty + activity), ACTION6 is ONE parametric
    node keyed to (cached) object centers, ZERO LLM calls in the loop (C4, V9).
  * Promotion needs verify >= 0.8 on changed transitions; a promoted model whose
    live accuracy (EWMA) collapses is DEMOTED and its rules banned so enumeration
    is forced to the next hypothesis instead of rediscovering the same one.
"""

import os
import re
import ast
import json
import time
import math
import random
import traceback
import pickle
import itertools
from collections import deque
from typing import Any, List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
import threading

import numpy as np
try:
    import torch                       # required on Kaggle (LLM); optional for local smoke tests
except ImportError:
    torch = None

# ==========================================
# CONFIG  (local dataset paths; NO network)
# ==========================================
ARC_DATA_ROOT = "/kaggle/input/notebooks/banwait13/datasets-for-arc-agi"
# Resolve at runtime; env var lets you repoint without editing the file.
LLM_MODEL_PATH = os.environ.get("ARC_LLM_PATH", "/kaggle/input/datasets/banwait13/models")
EMBED_MODEL_PATH = os.environ.get(
    "ARC_EMBED_PATH", f"{ARC_DATA_ROOT}/models/all-MiniLM-L6-v2")

EMBED_DEVICE = "cpu"                 # MiniLM is tiny; keep the 96GB GPU for the 27B LLM
LLM_LOAD_8BIT = False                # 54GB BF16 in 96GB VRAM -- quantization is pure risk

# LLM THINKING BUDGET (raised 2026-08-06).
#
# Measured: a full official eval finishes in ~2h of the 9h Kaggle allowance, so
# ~7h of wall clock was being left on the table while the LLM was throttled to
# 5 calls per level and 90s of thinking. RHAE charges ACTIONS, not seconds --
# internal reasoning is free, button presses are not. Under-spending compute to
# save time we are not short of is the one trade with no upside.
#
# The per-call and per-level caps below are deliberately generous; the thing
# that actually keeps the run inside 9h is LLM_SESSION_BUDGET_S, a global
# ledger over generate() wall time (see LocalLLM.budget_left). Bounding the
# total is safe; bounding each call just truncates thoughts mid-sentence.
def _envf(name, default):
    try:
        return float(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return float(default)


LLM_CALL_BUDGET_PER_LEVEL = int(_envf("ARC_LLM_CALLS_PER_LEVEL", 20))   # was 5
LLM_MAX_NEW_TOKENS = int(_envf("ARC_LLM_MAX_TOKENS", 2048))             # was 512
LLM_GEN_MAX_TIME_S = _envf("ARC_LLM_GEN_MAX_S", 240.0)                  # was 90
# Bounded repair rounds for a proposal that fails to PARSE, COMPILE or VERIFY.
#
# The old path asked once and dropped whatever came back without telling the
# model what was wrong. Most of those failures are one-line format mistakes an
# instruct model fixes immediately when shown the error -- so the information
# was there, we just threw it away.
#
# The loop takes NO environment action. RHAE squares the ACTION ratio and does
# not charge for thinking, so extra rounds cost exactly nothing on the
# leaderboard; the only budget they spend is LLM_SESSION_BUDGET_S, which
# generate() already caps. That asymmetry is the whole reason this is a repair
# loop over a recorded log and not a ReAct loop in the live environment.
#
# 0 restores the old one-shot behaviour.
LLM_REPAIR_ROUNDS = int(_envf("ARC_LLM_REPAIR_ROUNDS", 2))

# The watchdog MUST outlast a legitimate synthesis plus one wait on the generate
# lock, or it abandons its own healthy threads.
#
# This was a flat 540s back when one synthesis meant ONE generate call, and the
# rule of thumb next to it said "> 2x LLM_GEN_MAX_TIME_S". The repair loop broke
# that arithmetic: a healthy synthesis is now up to LLM_REPAIR_ROUNDS + 1
# generations, so a fixed 540 would have orphaned a perfectly good third round
# on Kaggle, silently and only there. Derive it instead of restating it -- the
# +1 is the lock wait. test_llm_repair.py asserts the invariant so raising
# LLM_REPAIR_ROUNDS can never quietly reintroduce the bug.
SYNTH_MAX_INFLIGHT_S = _envf("ARC_SYNTH_INFLIGHT_S",
                             (LLM_REPAIR_ROUNDS + 2) * LLM_GEN_MAX_TIME_S)
# Total generate() seconds for the whole submission. 9h session, ~2h of which
# is the agent itself; 6h of LLM leaves ~1h of margin. Calls run in background
# threads, so this is a ceiling on concurrent thinking, not added latency.
LLM_SESSION_BUDGET_S = _envf("ARC_LLM_SESSION_BUDGET_S", 6.0 * 3600.0)
CONTEXT_TOKEN_BUDGET = int(_envf("ARC_CONTEXT_TOKENS", 12000))          # roadmap C2
CHECKPOINT_INTERVAL_S = 900          # 15 min (roadmap V7)

FRAME_CONCENTRATION = 4.0            # a wall ring must be >=4x denser on the border than inside
BORDER_CONCENTRATION = 1.5           # chrome: >=1.5x over-represented on the border vs. its own area

MIN_CHANGED_SAME_ACTION = 4          # trigger A: >=4 changed transitions for ONE action
MIN_CHANGED_TOTAL = 8                # trigger B: >=8 changed total ...
MIN_ACTIONS_WITH_MIN_SAMPLES = 2     # ... with >=2 actions having >= MIN_SAMPLES_PER_RULE
MIN_SAMPLES_PER_RULE = 2             # don't trust a per-action rule from a single example
FIT_MEMO_MAX = 4096                  # per-action fit cache; ~7 entries added per synthesis
SYNTH_DEADLINE_S = 5.0               # tail cap on ONE action's fit; ~20x observed cost
MIN_CHANGED_FOR_VERIFY = 4           # verifier needs this much evidence to score at all
VERIFY_PROMOTE_THRESHOLD = 0.8       # >=4/5 exact match on changed transitions
DEMOTE_ACCURACY = 0.35               # EWMA accuracy below this -> demote + ban rules

# CandidateVerifier (COMPONENT 2.5) -- the trust boundary. A candidate is judged
# against the RECORDED LOG, not the last handful of frames: a model that fits only
# the recent past is exactly the failure the old last-5 sample could not see.
VERIFY_MAX_SAMPLES = 48              # cap the log slice so verification stays cheap
VERIFY_RECENT_SHARE = 0.5            # ...half of it the newest, half strided history
VERIFY_BUDGET_MS = 1500.0            # wall cap per candidate, checked between calls
VERIFY_GUARD_MS = 4000.0             # hard join timeout for UNTRUSTED (exec'd) code

EPSILON_START = 0.3
EPSILON_FLOOR = 0.05
EPSILON_DECAY = 0.99                 # per step within a level

# Livelock circuit-breaker (COMPONENT 5.11). RHAE squares the action ratio, so a
# scored action that neither changes the world nor tests an unknown outcome is
# pure loss -- and measurement showed the agent spends most of its budget on
# exactly those. The invariant: the current layer must keep producing NEW
# information, or control escalates.
NOOP_TRIP = 10                       # consecutive HUD-masked no-ops -> escalate
NOOP_HARD = 30                       # a level may not spend this many with ZERO change
SPEND_REPEAT_CAP = 2                 # times one (state, action) may be re-spent inertly
MAX_BREAK_RESETS = 3                 # level restarts the breaker may spend per level
# ...but "the world did not move" and "the agent learned nothing" are NOT the same
# claim, and conflating them is what made the breaker hijack layers that were
# working. A search legitimately re-visits an inert (state, action) on the way to
# somewhere new; that is a cost of search, not a livelock. So the WEAK evidence
# (`stale`: this exact pair was already spent inertly) may only take control while
# the agent has ALSO acquired no new state for NOVEL_PATIENCE actions. The STRONG
# evidence (`tripped`: the world is frozen outright) still fires unconditionally --
# a frozen world cannot be teaching anyone anything.
NOVEL_PATIENCE = 30                  # actions without a NEW state before `stale` may fire
ESCALATE_BFS_EVERY = 8               # escalations between graph-frontier BFS attempts

# Force offline for HF/transformers so a stray lookup can never hang on the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch is not None else "cpu"

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers/faiss not found. L3 memory -> flat storage.")

try:
    from arcengine import FrameData, GameAction, GameState
    HAS_ARCENGINE = True
except ImportError:
    HAS_ARCENGINE = False
    class GameAction:                      # noqa: N801  (local self-test stand-in)
        ACTION1 = "ACTION1"; ACTION2 = "ACTION2"; ACTION3 = "ACTION3"
        ACTION4 = "ACTION4"; ACTION5 = "ACTION5"; ACTION6 = "ACTION6"
        ACTION7 = "ACTION7"; RESET = "RESET"
    class GameState:                       # noqa: N801
        NOT_PLAYED = "NOT_PLAYED"; NOT_FINISHED = "NOT_FINISHED"
        WIN = "WIN"; GAME_OVER = "GAME_OVER"
    FrameData = Any

try:
    from agents.agent import Agent         # ARC-AGI-3-Agents framework (Kaggle rerun only)
except ImportError:
    class Agent:                           # noqa: N801
        def __init__(self, *a, **k):
            self.game_id = k.get("game_id", "local")
            self.action_counter = 0

VALID_ACTION_NAMES = {f"ACTION{i}" for i in range(1, 8)}


# ==========================================
# DATA STRUCTURES
# ==========================================

_TRANSITION_SERIAL = itertools.count()


@dataclass
class Transition:
    prev: np.ndarray
    action: str
    next: np.ndarray
    reward: float
    diff_encoding: str
    timestamp: float
    # Memo slots. A Transition is an immutable RECORD: prev/next are never
    # written to after construction (every consumer copies first), so anything
    # derived purely from them can be computed once. This matters because the
    # whole 200-deep buffer is rescanned for `changed` twice per action and for
    # `palette` once per synthesis -- profiling ls20 (300 actions) charged ~6.5s
    # to np.array_equal and ~112k np.unique calls to exactly those rescans.
    _changed: Optional[bool] = field(default=None, init=False, repr=False, compare=False)
    _palette: Optional[frozenset] = field(default=None, init=False, repr=False, compare=False)
    # Process-unique serial. `timestamp` is NOT usable as an identity: it comes
    # from time.time(), whose resolution on Windows is ~16ms while the agent
    # records transitions far faster than that, so two records routinely share
    # one value. The synthesis memo keys on identity and a collision there would
    # silently reuse another action's rule.
    _seq: int = field(default_factory=lambda: next(_TRANSITION_SERIAL),
                      init=False, repr=False, compare=False)

    @property
    def changed(self) -> bool:
        if self._changed is None:
            self._changed = bool(self.prev.shape != self.next.shape
                                 or not np.array_equal(self.prev, self.next))
        return self._changed

    @property
    def palette(self) -> frozenset:
        """Every colour appearing on either side of the transition."""
        if self._palette is None:
            self._palette = frozenset(int(x) for x in np.unique(self.prev)) | \
                            frozenset(int(x) for x in np.unique(self.next))
        return self._palette

@dataclass
class WorldModel:
    hypothesis: str
    source: str                       # "enumerative" | "llm-dsl" | "llm-code"
    spec: Any                         # program list OR code string
    fn: Optional[Callable] = None
    accuracy: float = 0.0

@dataclass
class GridObject:
    color: int
    size: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    shape_class: str
    mask: np.ndarray


def base_action(action: str) -> str:
    return "ACTION6" if str(action).startswith("ACTION6") else str(action)


def masked_diff(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Per-cell difference of two grids with HUD/timer cells zeroed out.

    Returns None when the grids cannot be compared cell-wise (shape change).
    A `mask` whose shape does not match is ignored rather than raising: masks
    are learned online and a level change can resize the grid under them.

    The world model compared grids RAW while StateGraph.changed_masked, the
    PatchWorldModel and the state hash all masked. That asymmetry meant one
    score/timer pixel ticking was enough to call a correct movement rule wrong,
    so the two halves of the agent disagreed about what "the same state" means.
    This is the single definition both now use."""
    if a.shape != b.shape:
        return None
    d = a != b
    if mask is not None and mask.shape == d.shape:
        d = d & ~mask
    return d


def masked_equal(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> bool:
    """Grid equality up to HUD cells -- see masked_diff."""
    d = masked_diff(a, b, mask)
    return d is not None and not bool(d.any())


def synth_ready(transitions: List[Transition]) -> bool:
    """Trigger for synthesis: >=4 changed transitions for the SAME action, OR
    >=8 changed total with at least 2 actions each having >=2 samples."""
    per: Dict[str, int] = {}
    for t in transitions:
        if t.changed:
            b = base_action(t.action)
            per[b] = per.get(b, 0) + 1
    if not per:
        return False
    if max(per.values()) >= MIN_CHANGED_SAME_ACTION:
        return True
    return (sum(per.values()) >= MIN_CHANGED_TOTAL
            and sum(1 for v in per.values() if v >= MIN_SAMPLES_PER_RULE) >= MIN_ACTIONS_WITH_MIN_SAMPLES)


# ==========================================
# COMPONENT 1: EYES (State Encoder)
# ==========================================

class StateEncoder:
    def __init__(self):
        self.shape_types = ['single_pixel', 'horizontal_line', 'vertical_line',
                            'square', 'rectangle', 'hollow_shape', 'irregular_blob']
        self._obj_cache: Dict[Any, List[GridObject]] = {}

    def detect_background(self, grid: np.ndarray) -> int:
        if grid.size == 0:
            return 0
        h, w = grid.shape
        if h == 0 or w == 0:
            return 0
        border = (np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
                  if h > 2 and w > 2 else grid.flatten())
        if len(border) == 0:
            return 0
        unique, counts = np.unique(border, return_counts=True)
        border_bg = int(unique[counts.argmax()]) if len(unique) > 0 else 0
        border_frac = counts.max() / len(border) if len(border) > 0 else 0.0

        # A FRAME (wall ring) is NOT the background -- the floor inside it is.
        # Frame vs. fill is decided by CONCENTRATION, not by absolute counts: a
        # frame color is many times denser on the border than in the interior,
        # while a genuine background is about as dense inside as out. The density
        # ratio is what separates "wall ring around a floor" from "background
        # around one large centered object" -- the two are indistinguishable by
        # counts alone, and the majority-interior test on its own gets the second
        # case wrong. Without this rule, bordered rooms/mazes flip bg to the wall
        # color and
        # collapse the whole floor into one phantom object, corrupting every
        # downstream reader (LLM prompts, MCTS value, object-scoped DSL ops).
        if border_frac > 0.5 and h > 2 and w > 2:
            interior = grid[1:-1, 1:-1]
            if interior.size >= 4:
                iu, ic = np.unique(interior, return_counts=True)
                interior_bg = int(iu[ic.argmax()])
                inside_frac = float(ic[iu == border_bg].sum()) / interior.size
                if (interior_bg != border_bg
                        and ic.max() > interior.size * 0.5
                        and inside_frac * FRAME_CONCENTRATION < border_frac):
                    return interior_bg

        if border_frac > 0.5:
            # Same principle as above, one scale up: the border winner may be a
            # thick CHROME band (surround / HUD panel) rather than a 1-px ring,
            # in which case the 1-px-interior test above cannot see it. A chrome
            # color is OVER-REPRESENTED on the border relative to how much of the
            # grid it actually owns (border_share / area_share is high), and some
            # other color owns strictly more of the grid -- that other color is
            # the play field. A genuine background is not over-represented on its
            # own border, so this cannot fire on one. Both conditions are
            # required: over-representation alone would misfire on frames that
            # are legitimately split into two large panels, where no color beats
            # the border winner on area and there is nothing better to choose.
            area_share = float((grid == border_bg).sum()) / grid.size
            if area_share > 0 and border_frac >= BORDER_CONCENTRATION * area_share:
                unique_g, counts_g = np.unique(grid, return_counts=True)
                cand = int(unique_g[counts_g.argmax()])
                if cand != border_bg and counts_g.max() > area_share * grid.size:
                    return cand
            return border_bg
        unique, counts = np.unique(grid, return_counts=True)
        return int(unique[counts.argmax()]) if len(unique) > 0 else 0

    def objects(self, grid: np.ndarray, bg: Optional[int] = None) -> List[GridObject]:
        """Cached object extraction (Issue 3: MCTS expands many nodes on the same
        grid -- flood fill must not be recomputed per expansion)."""
        if bg is None:
            bg = self.detect_background(grid)
        key = (grid.shape, int(bg), grid.tobytes())
        hit = self._obj_cache.get(key)
        if hit is not None:
            return hit
        objs = self.flood_fill_objects(grid, bg)
        if len(self._obj_cache) > 256:
            self._obj_cache.clear()
        self._obj_cache[key] = objs
        return objs

    def flood_fill_objects(self, grid: np.ndarray, bg_color: int, connectivity: int = 4) -> List[GridObject]:
        visited = np.zeros_like(grid, dtype=bool)
        objects: List[GridObject] = []
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                if not visited[r, c] and grid[r, c] != bg_color:
                    color = int(grid[r, c])
                    mask = np.zeros_like(grid, dtype=bool)
                    q = [(r, c)]
                    visited[r, c] = True
                    mask[r, c] = True
                    size = 0
                    rmin, cmin, rmax, cmax = r, c, r, c
                    while q:
                        cr, cc = q.pop(0)
                        size += 1
                        rmin, rmax = min(rmin, cr), max(rmax, cr)
                        cmin, cmax = min(cmin, cc), max(cmax, cc)
                        neigh = [(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)]
                        if connectivity == 8:
                            neigh += [(cr-1, cc-1), (cr-1, cc+1), (cr+1, cc-1), (cr+1, cc+1)]
                        for nr, nc in neigh:
                            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == color:
                                visited[nr, nc] = True
                                mask[nr, nc] = True
                                q.append((nr, nc))
                    center = (rmin + (rmax - rmin) / 2.0, cmin + (cmax - cmin) / 2.0)
                    shape_class = self.classify_shape(mask, rmin, cmin, rmax, cmax, size)
                    objects.append(GridObject(color, size, (rmin, cmin, rmax, cmax), center, shape_class, mask))
        return objects

    def classify_shape(self, mask, rmin, cmin, rmax, cmax, size) -> str:
        h = rmax - rmin + 1
        w = cmax - cmin + 1
        if size == 1:
            return 'single_pixel'
        if h == 1 and w > 1 and size == w:
            return 'horizontal_line'
        if w == 1 and h > 1 and size == h:
            return 'vertical_line'
        if size == h * w:
            return 'square' if h == w else 'rectangle'
        if h > 2 and w > 2:
            inner = mask[rmin+1:rmax, cmin+1:cmax]
            if not np.any(inner) and size == (2*h + 2*w - 4):
                return 'hollow_shape'
        return 'irregular_blob'

    def rle_encode(self, mask: np.ndarray) -> str:
        flat = mask.flatten()
        if len(flat) == 0:
            return ""
        runs, curr, count = [], flat[0], 1
        for val in flat[1:]:
            if val == curr:
                count += 1
            else:
                runs.append(f"{count}{'T' if curr else 'F'}")
                curr, count = val, 1
        runs.append(f"{count}{'T' if curr else 'F'}")
        return ",".join(runs)

    def encode_state(self, grid: np.ndarray, level: int = 1, requested_objects: List[int] = None) -> str:
        h, w = grid.shape
        bg = self.detect_background(grid)
        unique, counts = np.unique(grid, return_counts=True)
        color_counts = dict(zip([int(x) for x in unique], [int(x) for x in counts]))
        l0 = f"Grid: {h}x{w}. BG: {bg}. Colors: {color_counts}."
        if level == 0:
            return l0
        objects = self.objects(grid, bg)
        obj_strs = [f"Obj{i}(C:{o.color}, S:{o.size}, BB:{o.bbox}, Shp:{o.shape_class})"
                    for i, o in enumerate(objects)]
        l1 = l0 + " Objects: [" + ", ".join(obj_strs) + "]"
        if level == 1:
            return l1
        if level == 2 and requested_objects:
            details = [f"Obj{i}_RLE:{self.rle_encode(objects[i].mask)}"
                       for i in requested_objects if i < len(objects)]
            return l1 + " Details: " + " ".join(details)
        if level == 3:
            if h <= 10 and w <= 10:
                grid_str = "\n".join(" ".join(map(str, row)) for row in grid)
                return l1 + f"\nFull Grid:\n{grid_str}"
            sh, sw = max(1, h // 10), max(1, w // 10)
            ds = grid[::sh, ::sw]
            grid_str = "\n".join(" ".join(map(str, row)) for row in ds)
            return l1 + f"\nDownsampled Grid:\n{grid_str}"
        return l1

    def encode_transition(self, prev: np.ndarray, nxt: np.ndarray, action: str) -> str:
        if prev.shape != nxt.shape:
            return f"Action {action} resized grid {prev.shape} -> {nxt.shape}"
        diff = nxt != prev
        n = int(np.sum(diff))
        if n == 0:
            return f"Action {action}: No change."
        added, ac = np.unique(nxt[diff], return_counts=True)
        removed, rc = np.unique(prev[diff], return_counts=True)
        if set(added.tolist()) == set(removed.tolist()) and len(added) > 0:
            return f"Action {action}: movement/swap of {n} px, colors {added.tolist()}."
        add_d = dict(zip([int(x) for x in added], [int(x) for x in ac]))
        rem_d = dict(zip([int(x) for x in removed], [int(x) for x in rc]))
        return f"Action {action}: {n} px changed. Added: {add_d}, Removed: {rem_d}."


# ==========================================
# THE DSL (verifiable grid transforms)
# ==========================================

def _as_int(x):
    """Coerce an op argument to int, or None if it is not a number.

    Op totality (I4): an op handed an argument it cannot honour returns the grid
    unchanged rather than guessing or raising. A raise would kill a planner
    rollout mid-search; a guess would be certified by _fits on a handful of
    samples and then trusted for unscored lookahead, which is worse.
    Note `g == 'x'` on an int array raises on 0-size grids instead of returning
    all-False, so colour arguments need this guard too, not just offsets.
    """
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


class GridDSL:
    @staticmethod
    def identity(g, bg): return g.copy()

    @staticmethod
    def translate(g, bg, dr=0, dc=0):
        # An op that cannot honour its arguments returns the grid unchanged: a
        # confidently WRONG prediction is worse than no prediction, because the
        # synthesizer certifies on a few samples and the planner then trusts the
        # result for unscored lookahead.
        dr, dc = _as_int(dr), _as_int(dc)
        if dr is None or dc is None:
            return g.copy()
        out = np.full_like(g, bg)
        h, w = g.shape
        if abs(dr) >= h or abs(dc) >= w:
            return out                      # everything shifted off the grid
        # Slice-copy rather than a per-pixel loop: this op is evaluated
        # thousands of times per action during synthesis and lookahead.
        sr = slice(0, h - dr) if dr >= 0 else slice(-dr, h)
        tr = slice(dr, h) if dr >= 0 else slice(0, h + dr)
        sc = slice(0, w - dc) if dc >= 0 else slice(-dc, w)
        tc = slice(dc, w) if dc >= 0 else slice(0, w + dc)
        out[tr, tc] = g[sr, sc]
        return out

    @staticmethod
    def recolor(g, bg, src=None, dst=None):
        out = g.copy()
        src, dst = _as_int(src), _as_int(dst)
        if src is None or dst is None:
            return out
        out[g == src] = dst
        return out

    @staticmethod
    def reflect_h(g, bg): return g[:, ::-1].copy()

    @staticmethod
    def reflect_v(g, bg): return g[::-1, :].copy()

    @staticmethod
    def rotate90(g, bg):
        r = np.rot90(g, k=-1)
        return r.copy() if r.shape == g.shape else g.copy()

    @staticmethod
    def gravity(g, bg, direction="down"):
        # Unknown direction -> identity, never a silently substituted one.
        if direction not in ("down", "up", "left", "right"):
            return g.copy()
        out = np.full_like(g, bg)
        if direction in ("down", "up"):
            for c in range(g.shape[1]):
                col = g[:, c]
                vals = col[col != bg]
                if direction == "down":
                    out[g.shape[0] - len(vals):, c] = vals
                else:
                    out[:len(vals), c] = vals
        else:
            for r in range(g.shape[0]):
                row = g[r, :]
                vals = row[row != bg]
                if direction == "right":
                    out[r, g.shape[1] - len(vals):] = vals
                else:
                    out[r, :len(vals)] = vals
        return out

    @staticmethod
    def flood_replace(g, bg, r=0, c=0, dst=None):
        out = g.copy()
        h, w = g.shape
        r, c, dst = _as_int(r), _as_int(c), _as_int(dst)
        if r is None or c is None or dst is None:
            return out
        if not (0 <= r < h and 0 <= c < w):
            return out
        target = g[r, c]
        if target == dst:
            return out
        stack, seen = [(r, c)], set()
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in seen or not (0 <= cr < h and 0 <= cc < w) or g[cr, cc] != target:
                continue
            seen.add((cr, cc))
            out[cr, cc] = dst
            stack += [(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)]
        return out

    # ---- object(color)-scoped ops: "the avatar/object of color X does Y" ----

    @staticmethod
    def translate_color(g, bg, color=None, dr=0, dc=0, vacate=None):
        """Move ONLY the cells of `color` by (dr, dc).

        `vacate` is the colour left behind, defaulting to bg. It exists because
        "the mover leaves background behind" is an assumption, not a law: in a
        game with a floor tile, a carpet, or any board colour distinct from the
        border, the avatar uncovers the FLOOR when it steps off a cell. Measured
        on the dev games, that one wrong assumption is the whole difference
        between 0 and 8 (game, action) rules that reproduce the observed
        transitions EXACTLY -- see _observed_vacate, which reads the value off
        the data rather than guessing it."""
        color, dr, dc = _as_int(color), _as_int(dr), _as_int(dc)
        if color is None or dr is None or dc is None:
            return g.copy()
        vac = _as_int(vacate)
        out = g.copy()
        src = (g == color)
        out[src] = bg if vac is None else vac
        h, w = g.shape
        rs, cs = np.nonzero(src)
        nr, nc = rs + dr, cs + dc
        keep = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
        out[nr[keep], nc[keep]] = color
        return out

    @staticmethod
    def slide_color(g, bg, color=None, direction="down"):
        """Rigid-slide ALL cells of `color` in `direction` until any of them would
        hit a wall or a non-bg cell of another color (sokoban/ice-floor movement)."""
        color = _as_int(color)
        if color is None:
            return g.copy()
        step = {"down": (1, 0), "up": (-1, 0), "left": (0, -1), "right": (0, 1)}.get(direction)
        if step is None:
            return g.copy()             # unknown direction -> identity, not "down"
        dr, dc = step
        h, w = g.shape
        rs, cs = np.nonzero(g == color)
        if len(rs) == 0:
            return g.copy()
        others = (g != bg) & (g != color)
        k = 0
        while True:
            nr, nc = rs + (k + 1) * dr, cs + (k + 1) * dc
            if ((nr < 0) | (nr >= h) | (nc < 0) | (nc >= w)).any():
                break
            if others[nr, nc].any():
                break
            k += 1
        out = g.copy()
        out[rs, cs] = bg
        out[rs + k * dr, cs + k * dc] = color
        return out

    @staticmethod
    def step_color(g, bg, color=None, direction="down", stride=1, vacate=None):
        """Move ALL cells of `color` exactly `stride` pixels in `direction`, or
        not at all if any destination cell is off-grid or held by another object.

        `stride` is not decoration: these games draw objects as sprites on a
        lattice, so one move is k pixels (3-7 measured across the real games),
        not one. A stride-1-only vocabulary cannot express their movement at all.

        This is keyboard-movement physics, and it is the only hypothesis in the
        DSL that predicts a BLOCKED press as 'no change'. translate_color would
        walk through the wall and slide_color would cross the whole board; both
        agree with step_color on the presses that DID move something, which is
        exactly why the distinction has to exist in the vocabulary rather than
        be discovered from samples that never contain a blocked press.

        `vacate` is the colour uncovered where the object was; see
        translate_color for why it is not simply bg. It also widens what counts
        as FREE: a cell holding the uncovered colour is floor, not wall, so the
        mover may step onto it. Without that, a board whose floor differs from
        the detected background is wall-to-wall by construction and this op is
        identity everywhere -- precisely the case (a floor tile distinct from
        the outer chrome) the argument was added for. Walls are still walls:
        any other non-background colour blocks as before."""
        color, stride = _as_int(color), _as_int(stride)
        if color is None or stride is None or stride < 1:
            return g.copy()
        vac = _as_int(vacate)
        step = {"down": (1, 0), "up": (-1, 0), "left": (0, -1), "right": (0, 1)}.get(direction)
        if step is None:
            return g.copy()
        dr, dc = step[0] * stride, step[1] * stride
        h, w = g.shape
        rs, cs = np.nonzero(g == color)
        if len(rs) == 0:
            return g.copy()
        nr, nc = rs + dr, cs + dc
        if ((nr < 0) | (nr >= h) | (nc < 0) | (nc >= w)).any():
            return g.copy()                        # would leave the grid -> blocked
        others = (g != bg) & (g != color)
        if vac is not None:
            others &= (g != vac)                   # the uncovered floor is walkable
        if others[nr, nc].any():
            return g.copy()                        # another object is in the way
        out = g.copy()
        out[rs, cs] = bg if vac is None else vac
        out[nr, nc] = color
        return out

    @staticmethod
    def swap_colors(g, bg, a=None, b=None):
        out = g.copy()
        ai, bi = _as_int(a), _as_int(b)
        if ai is None or bi is None:
            return out
        out[g == ai] = bi
        out[g == bi] = ai
        return out

    @staticmethod
    def remap_colors(g, bg, mapping=None):
        """Apply an arbitrary src->dst colour map SIMULTANEOUSLY.

        The mechanical majority of these games are colour PERMUTATIONS: a tile
        cycles red->green->blue->red, a two-config toggle flips a whole palette
        at once. The DSL could express only a pairwise swap or a single recolor,
        so a 3-cycle had no representation at all -- and chaining recolors cannot
        substitute, because the second rule reads the FIRST one's output and
        collides (recolor 1->2 then 2->3 turns the original 1s into 3s).

        The fix is to read every mask off the SOURCE grid and write once, which
        makes the op order-free by construction and subsumes swap_colors (a
        2-cycle) and recolor (a 1-entry map) as special cases.

        Totality (I4): a mapping that is not a dict, is empty, or carries a single
        entry this op cannot honour returns the grid unchanged. The whole map is
        refused rather than partially applied -- a half-applied permutation is a
        different hypothesis from the one proposed, and it would be certified by
        _fits on a handful of samples and then trusted for unscored lookahead."""
        out = g.copy()
        if not isinstance(mapping, dict) or not mapping:
            return out
        pairs = []
        for s, d in mapping.items():
            si, di = _as_int(s), _as_int(d)
            if si is None or di is None:
                return out
            pairs.append((si, di))
        for si, di in pairs:
            if si != di:
                out[g == si] = di       # `g`, never `out`: that is the simultaneity
        return out

    @staticmethod
    def delete_color(g, bg, color=None):
        out = g.copy()
        color = _as_int(color)
        if color is not None:
            out[g == color] = bg
        return out

    OPS: Dict[str, Callable] = {}

    @classmethod
    def help(cls) -> str:
        return (
            "Ops (grid + optional args; all shape-preserving):\n"
            "  identity()                         no change\n"
            "  translate(dr, dc)                  shift all non-bg pixels\n"
            "  recolor(src, dst)                  repaint color src as dst\n"
            "  reflect_h() / reflect_v()          mirror horizontally / vertically\n"
            "  rotate90()                         rotate clockwise (square grids)\n"
            "  gravity(direction=down|up|left|right)\n"
            "  flood_replace(r, c, dst)           bucket-fill region at (r,c) with dst\n"
            "  translate_color(color, dr, dc[, vacate])  move ONLY that color's cells\n"
            "  step_color(color, direction, stride[, vacate])  move that color `stride` px, or not at all if blocked\n"
            "     vacate = colour uncovered where the object was (default: background;\n"
            "     set it when the mover walks over a floor tile rather than empty space)\n"
            "  slide_color(color, direction)      rigid-slide that color until blocked\n"
            "  swap_colors(a, b)                  exchange two colors\n"
            "  remap_colors(mapping)              apply a whole src->dst colour map AT ONCE\n"
            "     (expresses swaps, 3-cycles and full permutations in one rule;\n"
            "      every mask is read off the source grid, so order cannot matter)\n"
            "  delete_color(color)                erase a color to background\n"
        )

GridDSL.OPS = {
    "identity": GridDSL.identity, "translate": GridDSL.translate,
    "recolor": GridDSL.recolor, "reflect_h": GridDSL.reflect_h,
    "reflect_v": GridDSL.reflect_v, "rotate90": GridDSL.rotate90,
    "gravity": GridDSL.gravity, "flood_replace": GridDSL.flood_replace,
    "translate_color": GridDSL.translate_color, "step_color": GridDSL.step_color,
    "slide_color": GridDSL.slide_color,
    "swap_colors": GridDSL.swap_colors, "remap_colors": GridDSL.remap_colors,
    "delete_color": GridDSL.delete_color,
}


def compile_program(program: List[dict], encoder: StateEncoder) -> Optional[Callable]:
    """DSL rule list -> PARTIAL transition(grid, action). No exec(). None if op
    unknown. The compiled function returns None for an action no rule claims, and
    carries `.covers` = the claimed action set (None means a wildcard rule, i.e.
    everything).

    Returning the grid UNCHANGED for an unclaimed action, as this used to, is an
    assertion the program never made -- "that action does nothing" -- and it was
    the binding constraint on promotion. `verify` scores the last 5 changed
    transitions whatever their action, so a per-action program was judged on
    actions it had no rule for, and each of those scored wrong by construction.
    Measured over the dev games, the mean share of the verify sample a program
    actually claimed was 0.32 against a 0.8 threshold: 6 of the 7 games that
    synthesised anything could not have promoted however good their rules were.
    The same silent no-op also reached the planners, where an unclaimed action
    reads as a certified dead end and stops being tried."""
    if not isinstance(program, list) or not program:
        return None
    for rule in program:
        if not isinstance(rule, dict) or rule.get("op") not in GridDSL.OPS:
            return None
    wild = any(rule.get("action", "*") == "*" for rule in program)
    covers = frozenset(base_action(r.get("action", "*")) for r in program
                       if r.get("action", "*") != "*")

    def transition(grid: np.ndarray, action: str) -> Optional[np.ndarray]:
        g = np.asarray(grid)
        act_base = base_action(action)
        if not wild and act_base not in covers:
            return None
        bg = encoder.detect_background(g)
        out = g.copy()
        for rule in program:
            want = rule.get("action", "*")
            if want not in ("*", act_base):
                continue
            try:
                out = GridDSL.OPS[rule["op"]](out, bg, **(rule.get("args", {}) or {}))
            except Exception:
                return g.copy()
            if not isinstance(out, np.ndarray) or out.shape != g.shape:
                return g.copy()
        return out
    transition.covers = None if wild else covers
    return transition


def _reject_unsafe(tree) -> str:
    """'' if this AST is safe to exec in the restricted namespace, else why not.

    compile_python restricts __builtins__, which means an escape like `open(...)`
    fails at CALL time -- after the candidate was accepted and handed to the
    verifier, inside a thread we then have to abandon. Refusing at PARSE time is
    both cheaper and stricter: the code never reaches exec at all.
    """
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            return ("imports are not allowed; `np`, `math` and `dsl` are already "
                    "in scope and nothing else is available")
        if isinstance(n, ast.Attribute) and n.attr.startswith("__"):
            return f"attribute access to {n.attr!r} is not allowed (no dunders)"
        if isinstance(n, ast.Name) and n.id.startswith("__"):
            return f"the name {n.id!r} is not allowed (no dunders)"
    return ""


def compile_python_checked(code: str) -> Tuple[Optional[Callable], str]:
    """compile_python, but it says WHY it refused.

    The reason is not a log message. It is fed straight back to the model as the
    repair prompt, so it has to name the defect precisely enough to act on --
    "syntax error on line 3: expected ':'" is repairable, "compile failed" is not.
    """
    if not isinstance(code, str) or not code.strip():
        return None, "no code was returned"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return None, f"syntax error on line {e.lineno}: {e.msg}"
    except Exception as e:
        return None, f"the code could not be parsed: {type(e).__name__}: {e}"
    if not any(isinstance(n, ast.FunctionDef) and n.name == "transition"
               for n in tree.body):
        return None, ("no top-level function named `transition` was defined; it must "
                      "be `def transition(state_grid, action_str):` and return a "
                      "numpy array of the same shape")
    unsafe = _reject_unsafe(tree)
    if unsafe:
        return None, unsafe
    allowed = {"np": np, "math": math, "dsl": GridDSL, "__builtins__": {
        "range": range, "len": len, "min": min, "max": max, "abs": abs,
        "int": int, "float": float, "enumerate": enumerate, "zip": zip,
    }}
    local: Dict[str, Any] = {}
    try:
        exec(code, allowed, local)  # noqa: S102 - restricted + verified before trust
    except Exception as e:
        return None, f"the code raised while being defined: {type(e).__name__}: {e}"
    fn = local.get("transition")
    if not callable(fn):
        return None, "`transition` was defined but is not callable"
    return fn, ""


def compile_python(code: str) -> Optional[Callable]:
    """Sandboxed fallback for free-form LLM `transition`. Restricted namespace."""
    fn, err = compile_python_checked(code)
    if err:
        print(f"compile_python failed: {err}")
    return fn


# ---- reading one LLM reply --------------------------------------------------
# Everything below exists because a model's reply is a STRING, not a data
# structure, and the gap between the two is where proposals were being lost.


def _fenced_blocks(text: str) -> List[Tuple[str, str]]:
    """(language, body) for every ``` fence, in order of appearance."""
    out, i = [], 0
    while True:
        s = text.find("```", i)
        if s < 0:
            break
        nl = text.find("\n", s)
        if nl < 0:
            break
        lang = text[s + 3:nl].strip().lower()
        e = text.find("```", nl)
        if e < 0:
            out.append((lang, text[nl + 1:]))      # unterminated: take the rest
            break
        out.append((lang, text[nl + 1:e]))
        i = e + 3
    return out


def _balanced_objects(text: str) -> List[str]:
    """Every top-level {...} span, brace-balanced and string-aware, longest first.

    `find("{") ... rfind("}")` -- the old reader -- spans from the first brace in
    a prose sentence to the last brace of an unrelated object and hands back
    something that was never JSON. This yields each candidate separately so the
    real one can be tried on its own.
    """
    out, depth, start, in_str, esc = [], 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start:i + 1])
    if depth and start >= 0:
        out.append(text[start:])
    out.sort(key=len, reverse=True)
    return out


def _loads_lenient(s: str) -> dict:
    """json.loads, tolerating the one thing instruct models always get wrong.

    Asked for a multi-line function inside a JSON string field, a model emits
    REAL newlines. That is invalid JSON -- json.loads raises "Invalid control
    character" -- and it is not an edge case, it is the common case. Escaping
    control characters that occur INSIDE strings recovers the proposal without
    touching anything outside them.
    """
    try:
        return json.loads(s)
    except Exception:
        pass
    out, in_str, esc = [], False, False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            if ch in "\n\r\t":
                out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch])
                continue
        elif ch == '"':
            in_str = True
        out.append(ch)
    return json.loads("".join(out))


def _raw_code_field(text: str) -> str:
    """Last resort: pull the "code" string out by hand.

    Reached when the JSON is unreadable even leniently -- typically because the
    python body contains an unescaped quote, which closes the string early.
    Losing the whole proposal over a quote is worse than reading it coarsely,
    since whatever comes out still has to pass ast.parse and the verifier.
    """
    m = re.search(r'"code"\s*:\s*"', text)
    if not m:
        return ""
    body = text[m.end():]
    end = body.rfind('"')
    if end <= 0:
        return ""
    # Only the escapes a model actually emits. Backslash-unescaping is
    # deliberately NOT done: it would corrupt a body that already has real
    # newlines, and python rules rarely contain a literal backslash.
    return (body[:end].replace('\\n', '\n').replace('\\t', '\t')
            .replace('\\"', '"'))


def extract_candidate(text: str) -> Tuple[Optional["WorldModel"], str]:
    """Read one LLM reply into a WorldModel, or say exactly what was wrong.

    Returns (candidate, reason). Exactly one is ever non-empty. The reason is
    the repair prompt, so it is written to the model, not to the log.

    The reader is deliberately generous about FORM and strict about CONTENT: it
    will find the answer in a fenced block, behind a paragraph of preamble, or
    in JSON that does not quite parse -- and will then refuse it anyway unless
    it defines a real `transition` or a well-formed rule list. Being strict
    about form as well only loses correct answers to cosmetic mistakes.
    """
    if not isinstance(text, str) or not text.strip():
        return None, "the reply was empty"

    hypothesis, prog, code = "", None, ""
    # 1. JSON, wherever it lives: the whole reply, a ```json fence, or a
    #    balanced span buried in prose.
    spans = ([text] + [b for lang, b in _fenced_blocks(text) if lang in ("json", "")]
             + _balanced_objects(text))
    for span in spans:
        try:
            obj = _loads_lenient(span)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        hypothesis = str(obj.get("hypothesis", "") or "")
        if obj.get("program") is not None:
            prog = obj["program"]
        if obj.get("code"):
            code = str(obj["code"])
        if prog is not None or code:
            break
    # 2. It meant to send code but its JSON was unreadable.
    if prog is None and not code and '"code"' in text:
        code = _raw_code_field(text)
    # 3. No JSON at all. A bare function is still a correct answer to a coding
    #    prompt, and the old reader discarded every one of them.
    if prog is None and not code:
        blocks = [b for lang, b in _fenced_blocks(text)
                  if lang in ("python", "py", "") and "def transition" in b]
        if blocks:
            code = blocks[0]
        elif "def transition" in text:
            code = text[text.index("def transition"):]

    if prog is not None:
        if not isinstance(prog, list):
            return None, 'the "program" field must be a JSON list of rule objects'
        if not prog:
            return None, 'the "program" list was empty; give at least one rule'
        if any(not (isinstance(r, dict) and r.get("op")) for r in prog):
            return None, ('every entry in "program" must be an object with '
                          '"action", "op" and "args" keys')
        return WorldModel(hypothesis, "llm-dsl", prog), ""

    if code:
        _fn, err = compile_python_checked(code)
        if err:
            return None, err
        return WorldModel(hypothesis, "llm-code", code), ""

    return None, ('the reply contained neither a "program" list nor a `transition` '
                  'function; answer with JSON only, no commentary')


class LLMProposalStats:
    """How often does the LLM actually produce a model that survives the gate?

    The LLM half cannot be exercised on the dev box -- no weights fit -- so
    Kaggle is the only place it ever runs, and the log is the only instrument
    there. An unmeasured accept rate is an unknown one, and "the LLM is not
    helping" stays an opinion until the rejections are counted by reason.
    """

    def __init__(self):
        self.attempts = 0
        self.accepted = 0
        self.rounds_used: List[int] = []
        self.by_reason: Dict[str, int] = {}

    def attempt(self) -> None:
        self.attempts += 1

    def accepted_on(self, rounds: int) -> None:
        self.accepted += 1
        self.rounds_used.append(int(rounds))

    def rejected(self, reason: str) -> None:
        key = (reason or "unknown").strip().split("\n")[0][:60]
        self.by_reason[key] = self.by_reason.get(key, 0) + 1

    def rate(self) -> float:
        return self.accepted / self.attempts if self.attempts else 0.0

    def report(self) -> str:
        avg = sum(self.rounds_used) / len(self.rounds_used) if self.rounds_used else 0.0
        top = sorted(self.by_reason.items(), key=lambda kv: -kv[1])[:3]
        tail = "; ".join(f"{k} x{v}" for k, v in top) or "no rejections recorded"
        return (f"[LLM] proposals {self.accepted}/{self.attempts} accepted "
                f"({100 * self.rate():.0f}%), {avg:.1f} round(s) when accepted. "
                f"Top rejections: {tail}")


LLM_STATS = LLMProposalStats()


# ==========================================
# COMPONENT 2.5: CANDIDATE VERIFIER  (the trust boundary, system_v2 3.3)
# ==========================================
# One gate for every candidate world model, whatever produced it -- the enumerator,
# the DSL, the LLM. This is the "unify the INTERFACE, not the algorithms" decision:
# candidate -> verifier -> executor, with the verifier owning the contract so no
# producer can define its own idea of "verified".
#
# What it fixes about the gate it replaces (WorldModelManager.verify):
#
#   1. It judged `claimed[-5:]`. Five samples, all of them the newest. A model that
#      fit only the recent past scored 1.00 and was promoted; the log that would
#      have falsified it was sitting right there, unread. This one tests a slice
#      spanning the WHOLE log -- strided history first, newest last -- so a
#      recency-overfit candidate is refuted by the oldest evidence, early and cheap.
#   2. No resource cap at all, against a spec (v2 3.3) that requires one. A
#      pathological candidate could stall the run between two scored actions.
#   3. A bare float came back, so a rejection carried no reason and nothing about
#      candidate quality was ever legible. Verdict says WHY.
#
# Honest limit on the cap: `budget_ms` is checked BETWEEN predictions, so it bounds
# a slow candidate, not one that hangs inside a single call. Untrusted code (the
# exec() path) must go through check_guarded(), which pays for a daemon thread to
# get a real wall bound. Trusted DSL programs are total by construction and use the
# direct path.
class Verdict:
    """Why a candidate was accepted or refused. Never raises; never lies."""
    __slots__ = ("accepted", "accuracy", "n_tested", "n_claimed", "reason",
                 "errors", "ms", "stalled")

    def __init__(self, accepted=False, accuracy=0.0, n_tested=0, n_claimed=0,
                 reason="", errors=0, ms=0.0, stalled=False):
        self.accepted, self.accuracy = bool(accepted), float(accuracy)
        self.n_tested, self.n_claimed = int(n_tested), int(n_claimed)
        self.reason, self.errors = str(reason), int(errors)
        self.ms, self.stalled = float(ms), bool(stalled)

    def __bool__(self) -> bool:
        return self.accepted

    def __repr__(self) -> str:
        return (f"Verdict({'ACCEPT' if self.accepted else 'REJECT'} "
                f"acc={self.accuracy:.2f} n={self.n_tested}/{self.n_claimed} "
                f"err={self.errors} {self.ms:.0f}ms {self.reason})")


class CandidateVerifier:
    """Judge a candidate `transition(grid, action) -> grid` against a logged history.

    `mask_for` maps a base action to the HUD mask its evidence was FITTED under
    (masked_diff convention: a TRUE cell is one to IGNORE). Passing the fitting
    mask in is not a convenience -- comparing under a different mask than the fit
    used is the asymmetry that once demoted correct models on every game with a
    live clock, so the mask travels with the evidence.
    """

    def __init__(self, min_evidence: int = MIN_CHANGED_FOR_VERIFY,
                 max_samples: int = VERIFY_MAX_SAMPLES,
                 budget_ms: float = VERIFY_BUDGET_MS):
        self.min_evidence = int(min_evidence)
        self.max_samples = max(1, int(max_samples))
        self.budget_ms = float(budget_ms)
        self.last: Optional[Verdict] = None

    # -- sample selection ---------------------------------------------------
    def select(self, claimed: List["Transition"]) -> List["Transition"]:
        """Test order: strided history FIRST, newest LAST.

        Deterministic (a stride, not a draw) so a verdict is reproducible, and
        falsification-ordered so early abort does the most work: a candidate fitted
        on recent frames dies on the old ones, and dies on the first few of them.
        """
        n = len(claimed)
        if n <= self.max_samples:
            return list(claimed)          # chronological IS oldest-first
        n_recent = max(1, int(round(self.max_samples * VERIFY_RECENT_SHARE)))
        n_hist = self.max_samples - n_recent
        recent = claimed[-n_recent:]
        hist_pool = claimed[:-n_recent]
        if n_hist <= 0 or not hist_pool:
            return list(recent)
        step = max(1, len(hist_pool) // n_hist)
        hist = hist_pool[::step][:n_hist]
        return hist + recent

    # -- the gate -----------------------------------------------------------
    def check(self, fn: Optional[Callable], transitions: List["Transition"],
              mask_for: Optional[Dict[str, Optional[np.ndarray]]] = None,
              threshold: float = VERIFY_PROMOTE_THRESHOLD,
              budget_ms: Optional[float] = None) -> Verdict:
        t0 = time.perf_counter()
        mask_for = mask_for or {}
        budget = self.budget_ms if budget_ms is None else float(budget_ms)

        def done(**kw) -> Verdict:
            v = Verdict(ms=(time.perf_counter() - t0) * 1000.0, **kw)
            self.last = v
            return v

        if fn is None or not callable(fn):
            return done(reason="no-candidate")
        covers = getattr(fn, "covers", None)
        claimed = [t for t in (transitions or [])
                   if covers is None or base_action(t.action) in covers]
        if len(claimed) < self.min_evidence:
            return done(n_claimed=len(claimed), reason="thin-evidence")

        sample = self.select(claimed)
        n = len(sample)
        # A candidate may be wrong at most this many times and still clear the bar;
        # one more and no remaining outcome can save it, so stop paying for it.
        max_wrong = int(math.floor((1.0 - threshold) * n + 1e-9))
        correct = wrong = errors = 0
        for i, t in enumerate(sample):
            if (time.perf_counter() - t0) * 1000.0 > budget:
                return done(accuracy=correct / max(1, i), n_tested=i,
                            n_claimed=len(claimed), errors=errors,
                            reason="budget", stalled=True)
            try:
                pred = fn(np.asarray(t.prev).copy(), t.action)
                ok = (isinstance(pred, np.ndarray)
                      and pred.shape == np.asarray(t.next).shape
                      and masked_equal(pred, t.next,
                                       mask_for.get(base_action(t.action))))
            except Exception:
                errors += 1
                ok = False
            if ok:
                correct += 1
            else:
                wrong += 1
                if wrong > max_wrong:
                    return done(accuracy=correct / (i + 1), n_tested=i + 1,
                                n_claimed=len(claimed), errors=errors,
                                reason="falsified")
        acc = correct / max(1, n)
        return done(accepted=acc >= threshold, accuracy=acc, n_tested=n,
                    n_claimed=len(claimed), errors=errors,
                    reason="verified" if acc >= threshold else "below-threshold")

    def check_guarded(self, fn: Optional[Callable], transitions: List["Transition"],
                      mask_for: Optional[Dict[str, Optional[np.ndarray]]] = None,
                      threshold: float = VERIFY_PROMOTE_THRESHOLD,
                      guard_ms: float = VERIFY_GUARD_MS) -> Verdict:
        """check() with a real wall bound, for code we did not write.

        The between-calls budget cannot stop a candidate that hangs inside ONE
        call, so untrusted candidates are verified on a daemon thread we are
        willing to abandon. Abandoning is safe because check() is pure: it reads
        copies and mutates nothing the agent will look at again.
        """
        out: List[Verdict] = []

        def run():
            try:
                out.append(self.check(fn, transitions, mask_for, threshold,
                                      budget_ms=guard_ms))
            except Exception:
                out.append(Verdict(reason="verifier-crash"))

        th = threading.Thread(target=run, daemon=True)
        t0 = time.perf_counter()
        th.start()
        th.join(guard_ms / 1000.0)
        if not out:
            v = Verdict(reason="guard-timeout", stalled=True,
                        ms=(time.perf_counter() - t0) * 1000.0)
            self.last = v
            return v
        self.last = out[0]
        return out[0]

    # -- the executor half (v2 3.3: "dry-run in imagination") ---------------
    @staticmethod
    def dry_run(world_step: Callable, start: np.ndarray, plan: List[str],
                horizon: Optional[int] = None) -> Optional[List[np.ndarray]]:
        """Roll a plan forward through a VERIFIED model. Unscored, so it is free.

        Returns the predicted states, or None the moment the model declines to
        predict. None is not failure -- it is the model saying "I don't model
        this", and a caller that reads it as "nothing happens" is the exact bug
        compile_program's silent no-op used to cause. Total: never raises.
        """
        if world_step is None or start is None or not plan:
            return None
        cur = np.asarray(start)
        out: List[np.ndarray] = []
        for act in plan[:horizon or len(plan)]:
            try:
                nxt = world_step(cur.copy(), act)
            except Exception:
                return None
            if not isinstance(nxt, np.ndarray) or nxt.shape != cur.shape:
                return None
            out.append(nxt)
            cur = nxt
        return out or None


# ==========================================
# ENUMERATIVE SYNTHESIZER  (primary, no LLM, no GPU)
# ==========================================

class EnumerativeSynthesizer:
    """Search a small DSL program space for a per-action rule set that exactly
    reproduces the observed state-changing transitions. Deterministic, ~ms."""

    def __init__(self, encoder: StateEncoder, mask_fn: Optional[Callable] = None):
        self.encoder = encoder
        # Late-bound so the mask keeps improving after construction (StateGraph
        # rebuilds it every MASK_CHECK_EVERY transitions) and so a rebuilt graph
        # does not leave a stale array behind. None => compare raw.
        self.mask_fn = mask_fn
        # (action, sample identities, palette, banned, mask) -> rule or None.
        # See synthesize() for why this is an equivalence, not an approximation.
        # ARC_NO_FITMEMO=1 disables it; the trajectory must be identical either
        # way, which is exactly what makes that switch worth having.
        self._fit_memo: Dict[tuple, Optional[dict]] = {}
        self._memo_on = not os.environ.get("ARC_NO_FITMEMO")

    def hud_mask(self) -> Optional[np.ndarray]:
        if self.mask_fn is None:
            return None
        try:
            m = self.mask_fn()
        except Exception:
            return None
        return m if isinstance(m, np.ndarray) else None

    def evidence(self, transitions: List[Transition]) -> Dict[str, tuple]:
        """action -> (samples, mask it is judged under).

        The HUD mask's premise is "these cells tick no matter what you do, so
        they say nothing about your action". For an action whose EVERY observed
        change lies inside the mask, that premise is false: those cells are the
        only thing that ever says anything about it. Masking there does not
        remove noise, it removes the whole signal -- measured on the dev games,
        it erased an exactly-fitting rule on sp80 ACTION5 (5/5) and tu93 ACTION2
        (4/5) while helping seven other (game, action) pairs. So an action with
        no live evidence at all, and only such an action, is judged raw.

        Both the fitter and the verifier read the split from here, so a rule
        cannot be learned under one comparison and scored under another."""
        mask = self.hud_mask()
        per: Dict[str, list] = {}
        for t in transitions:
            if not t.changed or t.prev.shape != t.next.shape:
                continue
            per.setdefault(base_action(t.action), []).append(t)
        out: Dict[str, tuple] = {}
        for act, ts in per.items():
            live = [t for t in ts if not masked_equal(t.prev, t.next, mask)]
            out[act] = (live, mask) if len(live) >= MIN_SAMPLES_PER_RULE else (ts, None)
        return out

    def _changed_colors(self, samples) -> List[int]:
        cols = set()
        for prev, nxt in samples:
            d = prev != nxt
            cols.update(int(x) for x in np.unique(prev[d]))
            cols.update(int(x) for x in np.unique(nxt[d]))
        return sorted(cols)

    def _observed_shift(self, samples, col: int):
        """Read the displacement of `col` straight off the samples.

        Measured on real transitions: these games move objects by a sprite
        lattice stride of 3-7 px, while a hardcoded candidate list of +-1/+-2
        cannot express any of them -- which is why no object-scoped rule fit ANY
        action in ANY of the 25 games. Only rigid moves count (identical pixel
        count before and after), so a centroid that drifts because cells
        appeared or vanished cannot invent a shift."""
        deltas = []
        for prev, nxt in samples:
            pr, pc = np.nonzero(prev == col)
            nr, nc = np.nonzero(nxt == col)
            if len(pr) == 0 or len(pr) != len(nr):
                continue
            deltas.append((float(nr.mean() - pr.mean()), float(nc.mean() - pc.mean())))
        if not deltas:
            return None
        dr = int(round(float(np.median([d[0] for d in deltas]))))
        dc = int(round(float(np.median([d[1] for d in deltas]))))
        return None if (dr == 0 and dc == 0) else (dr, dc)

    def _shift_candidates(self, samples, col: int) -> List[Tuple[int, int]]:
        """Every displacement of `col` the samples support -- a small SEARCH, not
        a point estimate.

        _observed_shift is a centroid, and it declines outright unless the pixel
        count is identical before and after. That rules out the whole class of
        moves where the object LEAVES ITS OWN COLOUR BEHIND (a trail, a growing
        snake, a painted path): the count grows, so the centroid is refused and
        no data-derived candidate is emitted at all. Measured on 50 dev
        (game, action) windows, an exhaustive colour x shift x vacate sweep fit 5
        exactly while the proposed set fit 0, and 3 of those 5 were exactly this
        shape -- vacate == color, which is unreachable from a rigid-move
        estimator by construction.

        The bounding box supplies the missing estimates: for a rigid move both
        corners shift together, and for a trail the trailing corner stays put
        while the leading corner advances by the stride. Taking BOTH corners plus
        the centroid covers rigid moves, growth, and erosion with at most three
        hypotheses per colour -- the cost of a full sweep (palette x 32 shifts)
        would be ~3500 candidates per action, which the per-action time budget
        cannot pay."""
        out: List[Tuple[int, int]] = []
        cen = self._observed_shift(samples, col)
        if cen is not None:
            out.append(cen)
        lo: List[Tuple[int, int]] = []
        hi: List[Tuple[int, int]] = []
        for prev, nxt in samples:
            pr, pc = np.nonzero(prev == col)
            nr, nc = np.nonzero(nxt == col)
            if len(pr) == 0 or len(nr) == 0:
                continue
            lo.append((int(nr.min() - pr.min()), int(nc.min() - pc.min())))
            hi.append((int(nr.max() - pr.max()), int(nc.max() - pc.max())))
        for group in (hi, lo):          # leading edge first: it carries the stride
            if not group:
                continue
            d = (int(round(float(np.median([x[0] for x in group])))),
                 int(round(float(np.median([x[1] for x in group])))))
            if d != (0, 0) and d not in out:
                out.append(d)
        return out

    def _vacate_candidates(self, samples, col: int, dr: int, dc: int,
                           bg0: Optional[int]) -> List[Optional[int]]:
        """The colours worth trying behind the mover, cheapest-first.

        _observed_vacate returns ONE modal colour, which is right only when the
        object uncovers a uniform floor. Two cases it cannot produce:
        `vacate == col` (the object leaves itself behind -- nothing is uncovered,
        so the modal read has no cells to look at and returns None), and boards
        where the uncovered cells are mixed. Both appeared in the dev sweep.

        None is kept FIRST so that a rule which already fit keeps fitting with
        the identical argument list -- this widens the search without moving any
        existing hypothesis."""
        obs = self._observed_vacate(samples, col, dr, dc)
        out: List[Optional[int]] = [None]
        for v in (obs, int(col), bg0):
            if v is None or v == bg0:
                continue                 # v == bg0 is bit-identical to None
            if v not in out:
                out.append(v)
        return out

    def _observed_vacate(self, samples, col: int, dr: int, dc: int) -> Optional[int]:
        """What actually appears in the cells the object LEAVES.

        translate_color/step_color default to leaving the detected background
        behind, which is wrong for any board with a floor distinct from the
        border: the avatar uncovers the floor, not the void. Measured over the
        dev games' 46 (game, action) pairs, the best single-colour axis-aligned
        rule fits EXACTLY in 0 of them while the same rule with the right vacate
        colour fits exactly in 8 -- so this is not a refinement, it is the
        difference between having a world model and not having one.

        Read from the data like _observed_shift: modal colour of the next frame
        at the cells the object occupied and no longer covers. Returns None when
        it agrees with nothing (no uncovered cells anywhere)."""
        vals: List[int] = []
        for prev, nxt in samples:
            if prev.shape != nxt.shape:
                continue
            src = (prev == col)
            rs, cs = np.nonzero(src)
            if len(rs) == 0:
                continue
            h, w = prev.shape
            uncovered = src.copy()
            nr, nc = rs + dr, cs + dc
            keep = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
            uncovered[nr[keep], nc[keep]] = False   # still covered after the move
            if uncovered.any():
                vals.extend(int(x) for x in nxt[uncovered])
        if not vals:
            return None
        v, c = np.unique(np.asarray(vals), return_counts=True)
        return int(v[int(np.argmax(c))])

    def _remap_candidates(self, samples, mask=None) -> List[dict]:
        """Colour maps the samples SUPPORT, read straight off the changed cells.

        A permutation transition writes its own answer into the data: every cell
        that changed says "src became dst", so the map does not have to be
        searched, only counted. That is the whole reason remap_colors is
        proposable at all -- the space of colour maps is |palette|^|palette|,
        which no enumeration could ever sweep.

        Two candidates, cheapest bet first:
          - the MODAL map (each source colour to whatever it most often became),
          - the same map with low-agreement sources dropped, which is what a real
            permutation looks like once an animation frame or a single stray
            pixel has voted against it.
        Masked cells never vote: a HUD digit rolling 3->4 is not a world rule,
        and the verifier would not score it either -- the fitter and the verifier
        have to read the same comparison (see `evidence`).

        Nothing here is trusted. Both candidates go through _score exactly like
        every swept rule, so a spatial transition that happens to produce a tidy
        histogram is rejected on the grids themselves."""
        votes: Dict[int, Dict[int, int]] = {}
        for prev, nxt in samples:
            if prev.shape != nxt.shape:
                continue
            d = prev != nxt
            if mask is not None and getattr(mask, "shape", None) == d.shape:
                d = d & ~mask
            if not d.any():
                continue
            for s, t in zip(prev[d].tolist(), nxt[d].tolist()):
                per = votes.setdefault(int(s), {})
                per[int(t)] = per.get(int(t), 0) + 1
        if not votes:
            return []
        modal: Dict[int, int] = {}
        strict: Dict[int, int] = {}
        for src, per in votes.items():
            dst = max(per, key=lambda k: (per[k], -k))
            if dst == src:
                continue                       # a no-op entry is not a rule
            modal[src] = dst
            if per[dst] >= 0.9 * sum(per.values()):
                strict[src] = dst
        out: List[dict] = []
        for m in (modal, strict):
            if m and m not in out:
                out.append(dict(m))
        return out

    def _candidate_ops(self, samples, palette: List[int], bg0: Optional[int] = None,
                       mask=None):
        cands = []
        # Global geometry first (whole-grid physics).
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    cands.append(("translate", {"dr": dr, "dc": dc}))
        for d in ("down", "up", "left", "right"):
            cands.append(("gravity", {"direction": d}))
        cands += [("reflect_h", {}), ("reflect_v", {}), ("rotate90", {})]
        # Object(color)-scoped: enumerate ONLY over colors seen changing.
        moved = self._changed_colors(samples)
        if bg0 is None:
            bg0 = self.encoder.detect_background(samples[0][0]) if samples else 0
        for col in moved:
            if col == bg0:
                continue                       # "moving the background" is not an object rule
            # Data-derived displacement first: it is the only candidate that can
            # express a lattice stride, and it costs one hypothesis instead of a
            # combinatorial sweep. Blocked-aware form leads for the reason below.
            # ARC_NO_STRIDE reverts to the pre-2026-07-29 candidate set (fixed
            # +-1/+-2 shifts only). Debug switch, same family as ARC_NO_GRAPH /
            # ARC_NO_MECH_GATE / ARC_NO_PWM: it is the only way to attribute a
            # wall-clock or score delta to the data-derived displacement rather
            # than to trajectory divergence, which changes everything downstream.
            # The displacement and the colour left behind are both SEARCHED over
            # the small set the data supports, not point-estimated. A single
            # (shift, vacate) estimate proposed a fitting rule in 0 of 50 dev
            # windows where an exhaustive sweep found 5; the two candidate
            # helpers recover those without paying sweep cost. Bounded at
            # 3 shifts x 4 vacates x 2 ops per colour.
            shifts = [] if os.environ.get("ARC_NO_STRIDE") else self._shift_candidates(samples, col)
            for odr, odc in shifts:
                axis = ("down" if odr > 0 else "up") if odc == 0 else ("right" if odc > 0 else "left")
                for v in self._vacate_candidates(samples, col, odr, odc, bg0):
                    extra = {} if v is None else {"vacate": v}
                    if (odr == 0) != (odc == 0):    # axis-aligned -> expressible as a step
                        cands.append(("step_color",
                                      {"color": col, "direction": axis,
                                       "stride": max(abs(odr), abs(odc)), **extra}))
                    cands.append(("translate_color",
                                  {"color": col, "dr": odr, "dc": odc, **extra}))
            # step_color is tried BEFORE translate_color: on the samples that
            # actually moved they make identical predictions, but step_color
            # additionally predicts a blocked press as no-change. Unchanged
            # transitions never reach the sample set, so the tie cannot be
            # broken by evidence -- prefer the hypothesis that is right in more
            # states rather than the one that walks through walls.
            for d in ("down", "up", "left", "right"):
                cands.append(("step_color", {"color": col, "direction": d}))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1),
                           (-2, 0), (2, 0), (0, -2), (0, 2)):
                cands.append(("translate_color", {"color": col, "dr": dr, "dc": dc}))
            for d in ("down", "up", "left", "right"):
                cands.append(("slide_color", {"color": col, "direction": d}))
            cands.append(("delete_color", {"color": col}))
        # Whole-palette rules. These come AFTER the object-scoped moves on
        # purpose: on a movement game an exact object rule short-circuits the
        # candidate loop before a colour map is ever scored, so adding them here
        # cannot move a rule that already fits -- it can only supply one where
        # the DSL previously had no way to say what happened.
        for mapping in self._remap_candidates(samples, mask):
            cands.append(("remap_colors", {"mapping": mapping}))
        for i, a in enumerate(moved):
            for b in moved[i + 1:]:
                cands.append(("swap_colors", {"a": a, "b": b}))
        # Recolor last; src must be a color that actually changed.
        for s in moved:
            for d in palette:
                if s != d:
                    cands.append(("recolor", {"src": int(s), "dst": int(d)}))
        return cands

    def _fits(self, op: str, args: dict, samples, bgs, mask=None) -> bool:
        """`bgs` is the per-sample background, computed ONCE by the caller.

        It is a property of the sample, not of the candidate, so recomputing it
        inside the candidate loop was doing the same ~700 flood/vote passes per
        synthesis: profiling ls20 charged 66.2s of a 137.2s run to
        detect_background, all of it from here."""
        fn = GridDSL.OPS[op]
        for (prev, nxt), bg in zip(samples, bgs):
            try:
                out = fn(prev.copy(), bg, **args)
            except Exception:
                return False
            if not (isinstance(out, np.ndarray) and masked_equal(out, nxt, mask)):
                return False
        return True

    def _score(self, op: str, args: dict, samples, bgs, floor: int = 0, mask=None) -> int:
        """How many of `samples` this rule reproduces exactly.

        _fits is all-or-nothing, but the promotion gate this feeds
        (WorldModelManager.verify) accepts VERIFY_PROMOTE_THRESHOLD = 0.8. The
        two bars disagreed, and the strict one ran first: measured on the dev
        games, tr87 action 3 and dc22 action 3 each had a proposed rule that was
        exactly right on 4 of 5 samples and was thrown away by the 5th, so the
        action ended up with no rule at all and `synthesize` returned None for
        the whole game. A single animation frame or a HUD counter tick is enough
        to be that 5th sample.

        `floor` abandons a candidate as soon as it can no longer beat the best
        score so far, which keeps the widened candidate set affordable: the
        returned value is then only a lower bound, but a losing candidate's exact
        score is never used."""
        fn = GridDSL.OPS[op]
        hits, n = 0, len(samples)
        for i, ((prev, nxt), bg) in enumerate(zip(samples, bgs)):
            if hits + (n - i) <= floor:
                return hits                    # cannot overtake the incumbent
            try:
                out = fn(prev.copy(), bg, **args)
            except Exception:
                continue
            if isinstance(out, np.ndarray) and masked_equal(out, nxt, mask):
                hits += 1
        return hits

    def synthesize(self, transitions: List[Transition], banned: Optional[set] = None) -> Optional[WorldModel]:
        banned = banned or set()
        # A transition whose ONLY change is masked is a HUD tick: it says nothing
        # about the world and is unfittable by construction, so it used to eat one
        # of the five slots the action gets. `evidence` drops those -- and decides
        # per action whether masking is legitimate at all.
        ev = self.evidence(transitions)
        palette = set()
        for samples, _ in ev.values():
            for t in samples:
                palette |= t.palette
        by_action = {act: (samples, m) for act, (samples, m) in ev.items()}
        if not by_action:
            return None
        # The per-action fit is a PURE function of (window, palette, mask, banned),
        # and exactly one action gains a sample per step -- yet this loop re-ran
        # the whole search for every action, every action. Profiling ka59 (601
        # actions) charged 344.8s of 398.7s to synthesize, 294.5s of it to _score
        # across 301,050 calls. Memoising on the identity of those inputs returns
        # the SAME rule for the same inputs by construction; the only thing it
        # removes is the recomputation.
        pal_key = frozenset(palette)
        ban_key = frozenset(banned)      # exact, not len(): a demotion can swap
                                         # one banned rule for another
        program = []
        for act, (tsamples, mask) in by_action.items():
            use_t = tsamples[-5:]
            if len(use_t) < MIN_SAMPLES_PER_RULE:
                continue
            key = (act, tuple(t._seq for t in use_t), pal_key, ban_key,
                   None if mask is None else mask.tobytes())
            if self._memo_on and key in self._fit_memo:
                hit = self._fit_memo[key]
                if hit is not None:
                    # copy the args dict too: callers downstream own the rule
                    program.append({**hit, "args": dict(hit["args"])})
                continue
            use = [(t.prev, t.next) for t in use_t]
            bgs = [self.encoder.detect_background(p) for p, _ in use]
            cands = self._candidate_ops(use, sorted(palette), bgs[0], mask=mask)
            # Accept the BEST-scoring rule, not the first exactly-fitting one,
            # and hold it to the same bar the promotion gate uses rather than a
            # stricter private one -- see _score. An exact fit still short-
            # circuits in candidate order, so any action that already got a rule
            # gets the identical rule; this can only add rules where there were
            # none.
            need = len(use)
            allow = max(MIN_SAMPLES_PER_RULE, math.ceil(VERIFY_PROMOTE_THRESHOLD * need))
            best_score, best = 0, None
            # The candidate set is data-dependent (palette size, object count),
            # so its size is not bounded by anything in this file. This is the
            # only wall-clock floor under one action's decision; it is set ~20x
            # above the cost measured on the dev games, so it does not fire in
            # normal play -- it exists so a pathological frame costs seconds
            # rather than the game. Checked every 32 candidates because
            # time.perf_counter() is not free at this call frequency.
            _t_end, _timed_out = time.perf_counter() + SYNTH_DEADLINE_S, False
            for _ci, (op, args) in enumerate(cands):
                if (_ci & 31) == 0 and _ci and time.perf_counter() > _t_end:
                    _timed_out = True
                    break
                # The serialise-and-look-up is only meaningful once a rule has
                # been demoted; `banned` is empty for most of a run, and paying
                # ~700 json.dumps per synthesis for an empty set cost 12.8s of
                # the 137.2s ls20 profile.
                if banned and json.dumps({"action": act, "op": op, "args": args},
                                         sort_keys=True) in banned:
                    continue
                s = self._score(op, args, use, bgs, floor=best_score, mask=mask)
                if s > best_score:
                    best_score, best = s, (op, args)
                    if s == need:
                        break
            rule = None
            if best is not None and best_score >= allow:
                rule = {"action": act, "op": best[0], "args": dict(best[1])}
                program.append({**rule, "args": dict(rule["args"])})
            # A truncated search is NOT the answer to this key -- caching it
            # would make the deadline permanent for that window and break the
            # "same inputs, same output" property the memo rests on.
            if self._memo_on and not _timed_out:
                if len(self._fit_memo) >= FIT_MEMO_MAX:
                    self._fit_memo.clear()  # windows advance; old keys are dead
                self._fit_memo[key] = rule
        if not program:
            return None
        fn = compile_program(program, self.encoder)
        if fn is None:
            return None
        desc = "; ".join(f"{r['action']}->{r['op']}{r['args']}" for r in program)
        return WorldModel(hypothesis=f"Enumerated: {desc}", source="enumerative", spec=program, fn=fn)


# ==========================================
# COMPONENT 3: BRAIN  (in-process local LLM, offline, lazy, budgeted)
# ==========================================

def _resolve_model_dir(path: str) -> str:
    """Accept either the model dir or a parent containing it."""
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            if "config.json" in files:
                return root
            if root.count(os.sep) - path.count(os.sep) >= 2:
                dirs[:] = []
    return path


class LocalLLM:
    """Loads Qwen from a local directory with transformers. NO network. Preloaded
    in a background thread; a missing/OOM model never blocks or crashes the agent.
    Load is lock-guarded so preload + synth threads can't double-load 27B weights."""

    def __init__(self, model_path: str = LLM_MODEL_PATH):
        self.model_path = _resolve_model_dir(model_path)
        self.model = None
        self.tokenizer = None
        self._load_attempted = False
        self._load_lock = threading.Lock()
        self._gen_lock = threading.Lock()      # serialize generate() across threads
        # Global thinking ledger. Per-call caps are generous on purpose; THIS is
        # what keeps a 9h session from being eaten by a 27B model.
        self._budget_s = LLM_SESSION_BUDGET_S
        self._spent_s = 0.0
        self._calls = 0
        self._exhausted_announced = False
        self.available = (torch is not None
                          and os.path.isfile(os.path.join(self.model_path, "config.json")))
        if not self.available:
            print(f"[LLM] no config.json under {model_path} -> LLM disabled "
                  f"(agent runs on the no-LLM core).")

    def is_ready(self) -> bool:
        return self.model is not None and self.budget_left() > 0.0

    def budget_left(self) -> float:
        return max(0.0, self._budget_s - self._spent_s)

    def stats(self) -> dict:
        """For the run report: thinking actually spent vs budgeted. A run that
        reports 0 calls has an LLM that never ran, which is not the same thing
        as an LLM that did not help."""
        return {"calls": self._calls, "spent_s": round(self._spent_s, 1),
                "budget_s": round(self._budget_s, 1),
                "left_s": round(self.budget_left(), 1)}

    def ensure_loaded(self) -> bool:
        if self.model is not None:
            return True
        if not self.available:
            return False
        with self._load_lock:
            if self.model is not None:
                return True
            if self._load_attempted:
                return False
            self._load_attempted = True
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
                t0 = time.time()
                cfg = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
                print(f"[LLM] model_type={getattr(cfg, 'model_type', '?')}")
                # NOTE: no model_type rewriting. transformers 5.14.1 supports
                # qwen3_5 natively; forcing 'qwen2' loads the WRONG architecture.

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, local_files_only=True)

                dev = {"": 0} if torch.cuda.is_available() else "cpu"
                base = dict(local_files_only=True, device_map=dev, low_cpu_mem_usage=True)
                try:                                   # transformers >=5
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path, dtype=torch.bfloat16, **base)
                except TypeError:                      # older API
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path, torch_dtype=torch.bfloat16, **base)
                self.model.eval()

                try:
                    import fla  # noqa: F401
                    fast = "fla fast-path ON"
                except ImportError:
                    fast = "fla ABSENT (torch fallback; slower per token)"
                mem = self.model.get_memory_footprint() / 1e9
                print(f"[LLM] loaded {self.model_path} bf16 {mem:.1f}GB "
                      f"in {time.time() - t0:.0f}s | {fast}")

                # Smoke test AT LOAD TIME so a broken model fails here, loudly,
                # instead of silently returning "" on every budgeted call.
                if not self._smoke():
                    raise RuntimeError("smoke test produced no output")
                return True
            except Exception as e:
                print(f"[LLM] load failed ({type(e).__name__}: {e}); running without LLM.")
                traceback.print_exc()
                self.model = None
                self.tokenizer = None
                self.available = False
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False

    def _smoke(self) -> bool:
        try:
            ids = self.tokenizer("Reply with the word OK.", return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**ids, max_new_tokens=8, do_sample=False,
                                          pad_token_id=self.tokenizer.eos_token_id)
            txt = self.tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
            print(f"[LLM] smoke: {txt.strip()[:40]!r}")
            return True
        except Exception as e:
            print(f"[LLM] smoke failed: {e}")
            return False

    def generate(self, prompt: str, max_new_tokens: int = LLM_MAX_NEW_TOKENS,
                 temperature: float = 0.2) -> str:
        if not self.ensure_loaded():
            return ""
        # One generation at a time: click-plan / theorize / synth threads are
        # independently gated and WILL otherwise overlap on one CUDA model.
        with self._gen_lock:
            # Re-check inside the lock: a thread can queue behind a long call
            # and find the session budget gone by the time it gets the GPU.
            left = self.budget_left()
            if left <= 0.0:
                if not self._exhausted_announced:
                    self._exhausted_announced = True
                    print(f"[LLM] session thinking budget exhausted after "
                          f"{self._calls} calls / {self._spent_s:.0f}s; the "
                          f"no-LLM core carries the rest of the run.")
                return ""
            t0 = time.time()
            try:
                return self._generate_inner(prompt, max_new_tokens, temperature,
                                            max_time=min(LLM_GEN_MAX_TIME_S, left))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("[LLM] OOM during generate; skipping this call.")
                return ""
            except Exception as e:
                print(f"[LLM] generate failed: {type(e).__name__}: {e}")
                return ""
            finally:
                # Charge failures too: a call that OOMs still burned the clock.
                self._spent_s += time.time() - t0
                self._calls += 1

    def _encode(self, prompt: str):
        msgs = [{"role": "user", "content": prompt}]
        for kw in ({"enable_thinking": False}, {}):     # Qwen kwarg is optional
            try:
                enc = self.tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True,
                    return_tensors="pt", return_dict=True, **kw)
                return dict(enc)
            except TypeError:
                continue
            except Exception:
                break
        # Fallbacks: templated string, then raw prompt.
        try:
            text = self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)
        except Exception:
            text = prompt
        return dict(self.tokenizer(text, return_tensors="pt"))

    def _generate_inner(self, prompt: str, max_new_tokens: int, temperature: float,
                        max_time: float = None) -> str:
        enc = {k: v.to(self.model.device) for k, v in self._encode(prompt).items()
               if hasattr(v, "to")}
        in_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens,
                max_time=(LLM_GEN_MAX_TIME_S if max_time is None else max_time),
                do_sample=temperature > 0, temperature=max(temperature, 1e-2),
                pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][in_len:], skip_special_tokens=True)

    @staticmethod
    def _extract_json(text: str) -> dict:
        try:
            return json.loads(text)
        except Exception:
            s, e = text.find("{"), text.rfind("}")
            if 0 <= s < e:
                return json.loads(text[s:e + 1])
            raise

    def _synth_prompt(self, observations: str, feedback: str = "") -> str:
        """The one prompt, first attempt and every repair round alike, so the
        model never has to re-derive the output format from scratch."""
        p = f"""You are reverse-engineering the hidden rules of a grid puzzle.
Observed action -> effect transitions:
{observations}

{GridDSL.help()}
Return JSON ONLY. Prefer the FEWEST ops that explain the data:
{{"hypothesis": "one sentence", "program": [{{"action": "ACTION1", "op": "gravity", "args": {{"direction": "down"}}}}]}}
Use "action": "*" for a rule applying to every action. If no op fits, return instead:
{{"hypothesis": "...", "code": "def transition(state_grid, action_str):\\n    return state_grid.copy()"}}
The code form is exec'd in a RESTRICTED namespace: `np`, `math` and `dsl` are
already available, `import` is forbidden, no dunder names, and it must return a
numpy array with the SAME shape as state_grid."""
        if feedback:
            p += (f"\n\nYour PREVIOUS attempt was REJECTED: {feedback}\n"
                  "Return a corrected answer in the same format. Do not repeat "
                  "the rejected answer.")
        return p

    def propose_world_model(self, observations: str, feedback: str = ""
                            ) -> Tuple[Optional[WorldModel], str]:
        """ONE proposal round. Returns (candidate, why_not) -- never both.

        A None candidate with an EMPTY reason means the model said nothing at
        all: exhausted budget, OOM, or a failed generation. That is categorically
        different from a bad answer, and the caller must stop rather than spend
        its repair rounds re-asking a model that cannot reply.
        """
        raw = self.generate(self._synth_prompt(observations, feedback),
                            max_new_tokens=LLM_MAX_NEW_TOKENS, temperature=0.2)
        if not raw or not raw.strip():
            return None, ""
        return extract_candidate(raw)

    def synthesize_world_model(self, observations: str) -> Optional[WorldModel]:
        """One-shot proposal, kept for callers that don't run the repair loop."""
        return self.propose_world_model(observations)[0]

    def reflect(self, trajectory: str, failure_reason: str) -> str:
        prompt = f"""Trajectory:
{trajectory}
Failure: {failure_reason}
Give an updated one-sentence hypothesis of the puzzle rules.
Return JSON: {{"updated_hypothesis": "..."}}"""
        raw = self.generate(prompt, max_new_tokens=200, temperature=0.5)
        if not raw:
            return "Reflection unavailable."
        try:
            return self._extract_json(raw).get("updated_hypothesis", "Reflection failed.")
        except Exception:
            return "Reflection failed."

    def plan_clicks(self, board_text: str, observations: str, reactive: str,
                    hypothesis: str) -> List[Tuple[int, int]]:
        """Reasoning layer for click-only games (CSP / logic puzzles the coverage
        planner cannot solve blind). Given the board and what clicks have been
        observed to do, return an ORDERED list of (row, col) pixel cells to click
        toward completing the level. Empty list => no useful plan (caller falls back
        to coverage). Coords are 0-63; caller clamps + validates regardless."""
        prompt = f"""You are solving an interactive grid puzzle by CLICKING cells.
The board is a {board_text.count(chr(10)) + 1}-row grid; each char is a cell colour
('.' = background). Rows are numbered from the top (0), columns from the left (0).
A click is at a (row, col) coordinate.

Current board:
{board_text}

What clicks have done so far (action -> effect):
{observations or "nothing observed yet"}
Cells already found to react to a click (row,col): {reactive or "none yet"}
Working hypothesis of the rules: {hypothesis}

Reason about the GOAL (what a completed board looks like) and the click MECHANIC,
then give the shortest click sequence that makes progress toward completing the
level. Return JSON ONLY, at most 12 clicks, coordinates in [0,63]:
{{"goal": "one sentence", "clicks": [[row, col], [row, col]]}}"""
        raw = self.generate(prompt, max_new_tokens=LLM_MAX_NEW_TOKENS, temperature=0.3)
        if not raw:
            return []
        try:
            res = self._extract_json(raw)
        except Exception:
            return []
        out: List[Tuple[int, int]] = []
        for pair in (res.get("clicks") or [])[:12]:
            try:
                r, c = int(pair[0]), int(pair[1])
            except (TypeError, ValueError, IndexError):
                continue
            out.append((r, c))
        return out

    def theorize(self, board_text: str, theory: str, evidence: str,
                 trouble: str) -> Optional[dict]:
        """Schema-harness Phase 2 THEORIZE step (movement games): given the
        board, the locally-learned theory, and why the local loop is stuck,
        propose a BOUNDED revision -- a goal hypothesis and/or grounding edits
        (never free-form code). The caller certifies the proposal against the
        full timeline before adoption (ExecPlanner.propose_theory), so a
        hallucinated theory can never drive planning."""
        prompt = f"""You are reverse-engineering a grid game in which an avatar moves with UP/DOWN/LEFT/RIGHT.
Current board, one char per cell ('.' = background; other chars are colour
indices 0-9/a-f). Rows are numbered from the top (0), columns from the left (0):
{board_text}

Theory learned so far from real transitions:
{theory}

Recent observed transitions:
{evidence or "none"}

Why local reasoning is stuck: {trouble}

Reason about (1) the LEVEL GOAL -- which cell the avatar must reach -- and
(2) whether any learned grounding is wrong: a colour wrongly marked as a wall,
a wall colour not yet marked, or object colours that are decoys, not the goal.
Return JSON ONLY (use null for "goal" if unsure; colours are INTEGER indices):
{{"hypothesis": "one sentence",
 "goal": [row, col],
 "not_obstacle_colors": [],
 "obstacle_colors": [],
 "decoy_colors": []}}"""
        raw = self.generate(prompt, max_new_tokens=LLM_MAX_NEW_TOKENS, temperature=0.3)
        if not raw:
            return None
        try:
            res = self._extract_json(raw)
        except Exception:
            return None
        return res if isinstance(res, dict) else None


_LLM_SINGLETON: Optional[LocalLLM] = None

def get_local_llm() -> LocalLLM:
    """One 27B model per PROCESS, shared across agent instances/games."""
    global _LLM_SINGLETON
    if _LLM_SINGLETON is None:
        _LLM_SINGLETON = LocalLLM()
    return _LLM_SINGLETON


_EMBEDDER = None
_EMBEDDER_TRIED = False

def get_embedder():
    """One MiniLM per process (agents may be constructed once per game)."""
    global _EMBEDDER, _EMBEDDER_TRIED
    if _EMBEDDER is None and not _EMBEDDER_TRIED and HAS_SENTENCE_TRANSFORMERS:
        _EMBEDDER_TRIED = True
        try:
            _EMBEDDER = SentenceTransformer(EMBED_MODEL_PATH, device=EMBED_DEVICE)
        except Exception as e:
            print(f"Embedder init failed ({e}); L3 -> flat storage.")
            _EMBEDDER = None
    return _EMBEDDER


# ==========================================
# COMPONENT 2: MEMORY  (embedder on CPU; rule-based compression)
# ==========================================

class MemoryManager:
    def __init__(self):
        self.l0_working: List[Transition] = []
        self.l1_episodes: List[str] = []
        self.l2_hypothesis: str = "No rules discovered yet."
        self.max_tokens = CONTEXT_TOKEN_BUDGET
        self.embedder = get_embedder()
        self.l3_archive_index = None
        self.l3_archive_texts: List[str] = []
        if self.embedder is not None:
            try:
                self.l3_archive_index = faiss.IndexFlatL2(384)
            except Exception:
                self.l3_archive_index = None

    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) / 0.75)

    def add_transition(self, t: Transition):
        self.l0_working.append(t)
        if self.estimate_tokens(self.get_context_for_llm()) > self.max_tokens * 0.9:
            self.compress_l0_to_l1()
        if len(self.l1_episodes) > 5:
            self.embed_and_archive(self.l1_episodes.pop(0))

    def compress_l0_to_l1(self):
        if not self.l0_working:
            return
        chunk = self.l0_working[:10]
        self.l0_working = self.l0_working[10:]
        acts = sorted({t.action for t in chunk})
        summary = (f"Steps: tried {acts} over {len(chunk)} moves, "
                   f"{sum(1 for t in chunk if t.changed)} changed the grid, "
                   f"{sum(1 for t in chunk if t.reward > 0)} rewarded.")
        self.l1_episodes.append(summary)

    def embed_and_archive(self, text: str):
        self.l3_archive_texts.append(text)
        if self.embedder is not None and self.l3_archive_index is not None:
            try:
                emb = self.embedder.encode([text])[0]
                self.l3_archive_index.add(np.array([emb], dtype=np.float32))
            except Exception:
                pass

    def retrieve_similar(self, query: str, k: int = 3) -> List[str]:
        if not self.l3_archive_texts:
            return []
        if self.embedder is not None and self.l3_archive_index is not None:
            try:
                emb = self.embedder.encode([query])[0]
                _, idx = self.l3_archive_index.search(np.array([emb], dtype=np.float32),
                                                      min(k, len(self.l3_archive_texts)))
                return [self.l3_archive_texts[i] for i in idx[0] if i < len(self.l3_archive_texts)]
            except Exception:
                pass
        return self.l3_archive_texts[-k:]

    def observation_digest(self, transitions: List[Transition], max_items: int = 15) -> str:
        """Compact, LLM-ready summary of recent CHANGED transitions (keeps prompts small)."""
        changed = [t for t in transitions if t.changed][-max_items:]
        return "\n".join(f"{t.diff_encoding}" for t in changed) or "No state changes observed yet."

    def get_context_for_llm(self, current_state_desc: str = "") -> str:
        ctx = "=== Hypothesis ===\n" + self.l2_hypothesis + "\n\n"
        if current_state_desc and self.l3_archive_texts:
            sim = self.retrieve_similar(current_state_desc)
            if sim:
                ctx += "=== Relevant Past Episodes ===\n" + "\n".join(sim) + "\n\n"
        if self.l1_episodes:
            ctx += "=== Episode Summaries ===\n" + "\n".join(self.l1_episodes) + "\n\n"
        if self.l0_working:
            ctx += "=== Recent Working Memory ===\n"
            for t in self.l0_working[-10:]:
                ctx += f"Action: {t.action} -> {t.diff_encoding}\n"
        return ctx


# ==========================================
# COMPONENT 4: PLANNER (WorldModel + MCTS)
# ==========================================

class WorldModelManager:
    def __init__(self, llm: LocalLLM, encoder: StateEncoder, mask_fn: Optional[Callable] = None):
        self.llm = llm
        self.encoder = encoder
        self.enumerator = EnumerativeSynthesizer(encoder, mask_fn=mask_fn)
        self.active_model: Optional[WorldModel] = None
        self.banned_rules: set = set()
        self._lock = threading.Lock()
        # Per-action mask the last verify judged under, so the ONLINE accuracy can
        # reuse the very same comparison -- see judge().
        self._mask_for: Dict[str, Optional[np.ndarray]] = {}
        # The one gate every candidate passes, whatever produced it.
        self.verifier = CandidateVerifier()
        self.last_verdict: Optional[Verdict] = None

    def _compile(self, wm: WorldModel) -> Optional[Callable]:
        if wm.source in ("enumerative", "llm-dsl"):
            return compile_program(wm.spec, self.encoder)
        return compile_python(wm.spec)

    def verify(self, fn: Callable, transitions: List[Transition],
               guarded: bool = False) -> float:
        """Accuracy over the transitions the model CLAIMS -- see compile_program.

        MIN_CHANGED_FOR_VERIFY is applied twice: once to the game (is there
        enough changed evidence to judge anything at all) and once to the claimed
        subset (does this model have enough of its OWN domain on record). A
        program that claims one action and predicts it correctly is a correct
        model OF THAT ACTION; scoring it on the actions it never claimed measured
        nothing about it and rejected every partial program -- which is every
        program the enumerator produces, since its rules are per-action.

        Comparison is HUD-masked, and the "changed" filter with it, for the same
        reason synthesize drops HUD-only samples: a rule is judged on the world it
        models, and a transition that only ticked a timer is not evidence for or
        against it. The mask is taken PER ACTION from the same
        EnumerativeSynthesizer.evidence split the fitter used, so a rule can never
        be learned under one comparison and scored under another.

        The SAMPLE is chosen by CandidateVerifier (COMPONENT 2.5), not by this
        method: it used to be `claimed[-5:]`, the five newest, which cannot refute
        a model that fits only the recent past however much older evidence is on
        record. `ARC_NO_VERIFIER=1` restores the old five-sample gate so the change
        is one paired run away from being measured.
        """
        ev = self.enumerator.evidence(transitions)
        mask_for = {act: m for act, (_, m) in ev.items()}
        self._mask_for = mask_for
        keep = {id(t) for samples, _ in ev.values() for t in samples}
        changed = [t for t in transitions if id(t) in keep]   # temporal order kept
        if len(changed) < MIN_CHANGED_FOR_VERIFY:
            return 0.0
        if os.environ.get("ARC_NO_VERIFIER"):
            return self._verify_last5(fn, changed, mask_for)
        v = (self.verifier.check_guarded(fn, changed, mask_for) if guarded
             else self.verifier.check(fn, changed, mask_for))
        self.last_verdict = v
        # `thin-evidence` is "not enough of its own domain on record", which is not
        # the same claim as "predicts wrongly" -- both score 0.0 to the caller, as
        # before, but only one of them is about the candidate.
        return v.accuracy if v.reason != "thin-evidence" else 0.0

    def _verify_last5(self, fn: Callable, changed: List[Transition],
                      mask_for: Dict[str, Optional[np.ndarray]]) -> float:
        """The pre-2026-08-06 gate, kept intact as the control arm."""
        covers = getattr(fn, "covers", None)
        claimed = changed if covers is None else \
            [t for t in changed if base_action(t.action) in covers]
        if len(claimed) < MIN_CHANGED_FOR_VERIFY:
            return 0.0
        sample = claimed[-5:]
        correct = 0
        for t in sample:
            try:
                pred = fn(t.prev.copy(), t.action)
                if isinstance(pred, np.ndarray) and masked_equal(
                        pred, t.next, mask_for.get(base_action(t.action))):
                    correct += 1
            except Exception:
                pass
        return correct / len(sample)

    def judge(self, prev: np.ndarray, action: str, nxt: np.ndarray) -> Optional[float]:
        """Score the ACTIVE model on one live transition: 1.0, 0.0, or None.

        None means "this transition says nothing about this model" -- either the
        model claims nothing about the action (predict_or_none), or nothing
        happened in the world it models.

        The comparison is the per-action one the rule was VERIFIED under, taken
        from the snapshot verify() left behind. This closes the same asymmetry as
        verify itself, in the place where it does the most damage: the online
        accuracy is what DEMOTES a model and bans its rules, so while it compared
        raw, one timer pixel ticking alongside a real move scored a perfect model
        0.0, and a HUD-only tick scored it 0.0 with nothing having happened at
        all. A correct model decayed past DEMOTE_ACCURACY and banned its own
        correct rules on any game with a live HUD.

        An action the last verify never saw falls back to the graph's mask: the
        world model models the world, not the chrome. A rule that survives on
        masked-only evidence (the evidence() premise check) then never gets
        scored here -- masked_equal is True for it, so it returns None -- which
        is deliberate: such a rule cannot mislead the planners, since novelty and
        dead-action pruning run off the masked state hash rather than the model."""
        if self.active_model is None:
            return None
        if prev is None or nxt is None or prev.shape != nxt.shape:
            return None      # a resize is a level discontinuity, not a bad prediction
        mask = self._mask_for.get(base_action(action), self.enumerator.hud_mask())
        if masked_equal(prev, nxt, mask):
            return None
        pred = self.predict_or_none(prev, action)
        if pred is None:
            return None
        return 1.0 if masked_equal(pred, nxt, mask) else 0.0

    def _promote(self, wm: WorldModel, fn: Callable, score: float):
        wm.fn = fn
        wm.accuracy = score
        with self._lock:
            self.active_model = wm
        print(f"[world-model] PROMOTED ({wm.source}, verify={score:.2f}): {wm.hypothesis}")

    def demote(self, wm: WorldModel):
        """Live accuracy collapsed: drop the model and ban its rules so the
        enumerator is forced to the NEXT hypothesis, not the same one again."""
        with self._lock:
            if self.active_model is wm:
                self.active_model = None
        if isinstance(wm.spec, list):
            for rule in wm.spec:
                try:
                    self.banned_rules.add(json.dumps(rule, sort_keys=True))
                except Exception:
                    pass
        print(f"[world-model] DEMOTED (acc={wm.accuracy:.2f}): {wm.hypothesis}")

    def reset_level(self):
        with self._lock:
            self.active_model = None
        self.banned_rules.clear()

    def try_enumerative(self, transitions: List[Transition]) -> bool:
        """Synchronous, cheap. Returns True if a model was promoted."""
        wm = self.enumerator.synthesize(transitions, self.banned_rules)
        if wm is None or wm.fn is None:
            return False
        score = self.verify(wm.fn, transitions)
        if score >= VERIFY_PROMOTE_THRESHOLD:
            self._promote(wm, wm.fn, score)
            return True
        return False

    def synthesize_with_llm(self, observations: str,
                            transitions_provider: Callable[[], List[Transition]]):
        """Background-thread safe LLM fallback (roadmap V10), with a BOUNDED
        repair loop. Verifies against FRESH transitions at completion time, so a
        slow generation can't get promoted on stale evidence.

        Why a loop. The old path asked once and dropped whatever came back --
        unparseable, uncompilable or simply wrong -- without ever telling the
        model which of the three had happened. Most of those are one-line format
        mistakes, so the information needed to fix them existed and was
        discarded. Each round here feeds the EXACT failure back and asks again.

        Why it is safe under RHAE. Nothing in this method takes an environment
        action; it reads a recorded log. RHAE squares the ACTION ratio and does
        not charge for thinking, so the extra rounds are free on the
        leaderboard. That is precisely the property a ReAct loop -- which acts in
        order to observe -- would not have.

        What the loop must NOT do is soften the trust boundary. Every round
        still ends at CandidateVerifier under the same threshold, exec'd code is
        still verified behind the thread guard, and a model that never verifies
        is never promoted however many rounds it is given.
        """
        LLM_STATS.attempt()
        feedback, last_reason = "", "no proposal"
        try:
            for rnd in range(LLM_REPAIR_ROUNDS + 1):
                candidate, why = self.llm.propose_world_model(observations, feedback)
                if candidate is None:
                    if not why:
                        # The model said nothing: budget gone, OOM, or a failed
                        # generation. Re-asking cannot help, and each retry would
                        # still occupy the GPU and the generate lock.
                        last_reason = "no reply from the model"
                        break
                    feedback = last_reason = why
                    continue
                fn, cerr = ((self._compile(candidate), "")
                            if candidate.source == "llm-dsl"
                            else compile_python_checked(candidate.spec))
                if fn is None:
                    feedback = last_reason = (
                        cerr or "those ops did not compile into a usable rule; "
                                "check the op names and argument names")
                    continue
                transitions = transitions_provider() or []
                # Free-form LLM python went through exec(); it is the one candidate
                # source we cannot assume terminates, so it is verified behind the
                # thread guard. DSL programs are total by construction.
                untrusted = candidate.source not in ("enumerative", "llm-dsl")
                score = self.verify(fn, transitions, guarded=untrusted)
                if score >= VERIFY_PROMOTE_THRESHOLD:
                    self._promote(candidate, fn, score)
                    LLM_STATS.accepted_on(rnd + 1)
                    return
                v = self.last_verdict
                feedback = last_reason = (
                    f"the rule reproduced only {score:.0%} of the recorded "
                    f"transitions, and {VERIFY_PROMOTE_THRESHOLD:.0%} is required"
                    + (f" (verifier: {v.reason})" if v and v.reason else ""))
                print(f"[world-model] LLM candidate rejected "
                      f"({v if v else f'verify={score:.2f}'}).")
            LLM_STATS.rejected(last_reason)
        except Exception:
            LLM_STATS.rejected("exception during synthesis")
            traceback.print_exc()

    def claims(self, action: str) -> bool:
        """Does the active model have a rule for this action? Unclaimed is
        UNKNOWN, not inert -- the distinction the silent no-op used to erase."""
        with self._lock:
            m = self.active_model
        if not (m and m.fn):
            return False
        covers = getattr(m.fn, "covers", None)
        return covers is None or base_action(action) in covers

    def predict_or_none(self, state: np.ndarray, action: str) -> Optional[np.ndarray]:
        """The model's successor, or None where it makes no claim. Callers that
        need an array regardless use predict(); callers that would otherwise read
        'unknown' as 'nothing happens' -- MCTS expansion, the novelty lookahead,
        the live-accuracy tracker -- use this one."""
        with self._lock:
            m = self.active_model
        if m and m.fn:
            try:
                nxt = m.fn(state.copy(), action)
                if isinstance(nxt, np.ndarray) and nxt.shape == state.shape:
                    return nxt
            except Exception:
                pass
        return None

    def predict(self, state: np.ndarray, action: str) -> np.ndarray:
        nxt = self.predict_or_none(state, action)
        return state.copy() if nxt is None else nxt


class ExploitGate:
    """Hysteresis gate (Issue 4): enter EXPLOIT only after ENTER_STREAK consecutive
    correct one-step predictions; fall back to EXPLORE when the rolling accuracy
    over the last 10 changed steps drops below EXIT_AVG."""
    ENTER_STREAK = 5
    EXIT_AVG = 0.5

    def __init__(self):
        self.window: List[float] = []
        self.streak = 0
        self.mode = "EXPLORE"

    def update(self, hit: float):
        self.window.append(hit)
        self.window = self.window[-20:]
        self.streak = self.streak + 1 if hit >= 0.7 else 0
        if self.mode == "EXPLORE" and self.streak >= self.ENTER_STREAK:
            self.mode = "EXPLOIT"
        elif (self.mode == "EXPLOIT" and len(self.window) >= 10
              and float(np.mean(self.window[-10:])) < self.EXIT_AVG):
            self.mode = "EXPLORE"
            self.streak = 0

    def should_exploit(self) -> bool:
        return self.mode == "EXPLOIT"


class MCTSNode:
    __slots__ = ("state", "parent", "action", "children", "visits", "value_sum",
                 "prior", "expanded", "unknown")

    def __init__(self, state, parent=None, action=None, prior=1.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[str, "MCTSNode"] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.expanded = False
        self.unknown = False        # the model made no claim about this edge

    def value(self):
        return self.value_sum / self.visits if self.visits else 0.0

    def uct_select(self, c_puct=1.414):
        best, best_child = -1e18, None
        for child in self.children.values():
            u = c_puct * child.prior * math.sqrt(self.visits + 1) / (1 + child.visits)
            s = child.value() + u
            if s > best:
                best, best_child = s, child
        return best_child


class MCTSPlanner:
    """Heuristic-value MCTS. No neural value head, no LLM. ACTION6 = one node."""
    def __init__(self, world_model: WorldModelManager, encoder: StateEncoder, counter: "StateCounter"):
        self.world_model = world_model
        self.encoder = encoder
        self.counter = counter

    def _candidate_actions(self, grid: np.ndarray, valid: List[str]) -> List[str]:
        acts = [a for a in valid if not a.startswith("ACTION6")]
        if any(a.startswith("ACTION6") for a in valid):
            centers = [(int(o.center[0]), int(o.center[1]))
                       for o in self.encoder.objects(grid)[:3]]   # cached (Issue 3)
            if not centers:
                centers = [(grid.shape[0] // 2, grid.shape[1] // 2)]
            acts += [f"ACTION6_r{r}_c{c}" for r, c in centers]
        return acts or ["ACTION1"]

    def _leaf_value(self, node: "MCTSNode") -> float:
        """An edge the model makes no claim about is UNKNOWN, not inert.

        A copied parent state scores with the PARENT's novelty, which is the
        pessimistic reading and silently deletes those actions from the search --
        so a program with a rule for one action would narrow the agent to that
        action. Optimism-under-uncertainty is the standard fix: 1.0 is exactly
        peek_bonus' own upper bound (an unvisited state), so an unknown edge is
        ranked as the most promising thing the model could have predicted, never
        more."""
        grid = node.state
        bg = self.encoder.detect_background(grid)
        activity = float(np.mean(grid != bg))
        novelty = 1.0 if node.unknown else self.counter.peek_bonus(grid)
        return novelty + 0.1 * activity

    def search(self, root_grid: np.ndarray, valid_actions: List[str], n_iterations=60) -> str:
        root = MCTSNode(root_grid)
        for _ in range(n_iterations):
            node, path = root, [root]
            while node.expanded and node.children:
                nxt = node.uct_select()
                if nxt is None:
                    break
                node = nxt
                path.append(node)
            if not node.expanded:
                cands = self._candidate_actions(node.state, valid_actions)
                prior = 1.0 / len(cands)
                for a in cands:
                    ns = self.world_model.predict_or_none(node.state, a)
                    child = MCTSNode(node.state if ns is None else ns,
                                     parent=node, action=a, prior=prior)
                    child.unknown = ns is None
                    node.children[a] = child
                node.expanded = True
            val = self._leaf_value(node)
            for n in reversed(path):
                n.visits += 1
                n.value_sum += val
        if not root.children:
            return valid_actions[0] if valid_actions else "ACTION1"
        return max(root.children.items(), key=lambda kv: kv[1].visits)[0]


# ==========================================
# COMPONENT 5: LEARNER (count-based novelty only; no training)
# ==========================================

class StateCounter:
    def __init__(self):
        self.counts: Dict[int, int] = {}

    def simhash(self, grid: np.ndarray) -> int:
        if grid.shape[0] > 16 or grid.shape[1] > 16:
            sh, sw = max(1, grid.shape[0] // 16), max(1, grid.shape[1] // 16)
            grid = grid[::sh, ::sw]
        return hash(grid.tobytes())

    def get_bonus(self, grid: np.ndarray) -> float:
        h = self.simhash(grid)
        self.counts[h] = self.counts.get(h, 0) + 1
        return 1.0 / math.sqrt(self.counts[h])

    def peek_bonus(self, grid: np.ndarray) -> float:
        h = self.simhash(grid)
        return 1.0 / math.sqrt(self.counts.get(h, 0) + 1)


# ==========================================
# COMPONENT 5.5: GOAL INFERENCE + REACTIVE PLANNER
# ==========================================
#
# Why this exists: the real ARC-AGI-3 frames are a small logical grid rendered
# SCALED into 64x64 (an 8x8 level -> 8x8 pixel blocks; the avatar jumps `scale`
# pixels per action). The enumerative DSL assumes 1-2px moves, so it never fits
# the physics and the agent falls back to random walk (the ~0.2% failure mode).
#
# ReactivePlanner sidesteps that two ways:
#   1) It LEARNS the avatar's per-action displacement DIRECTLY from the observed
#      centroid shift, so it is scale-invariant (an 8px jump is just (dr,dc)=(8,0)).
#   2) It INFERS the goal as the nearest distinctive object the avatar can reach,
#      learns which colors are obstacles by bumping into them, and moves greedily
#      toward the goal with a wall-escape when a preferred move is blocked.
# No LLM, no GPU -- pure geometry, ~microseconds/step.

class ReactivePlanner:
    STUCK_LIMIT = 3          # centroid unchanged this many steps -> try next-best move

    def __init__(self):
        self.action_disp: Dict[str, Tuple[int, int]] = {}   # action -> (dr, dc) of avatar
        self.avatar_color: Optional[int] = None
        self.obstacle_colors: set = set()
        self.non_goal_colors: set = set()      # distinctive objects that gave NO reward (decoys)
        self._last_center: Optional[Tuple[float, float]] = None
        self._stuck = 0
        self._rank_offset = 0
        self._target: Optional[Tuple[int, int]] = None
        self._visits: Dict[Tuple[int, int], int] = {}   # lattice cell -> times seen
        # Which return path of act() produced the last action (read-only tag; see
        # MyAgent._route). act() has nine exits and they are NOT equivalent --
        # "committed a modelled plan" and "fell through to the inherited greedy"
        # are different agents wearing the same name in the score table.
        self._sub = ""
        # The order over world-states (COMPONENT 5.12), set by MyAgent. Read-only
        # here except for the pursuit warrant: this planner is the consumer that
        # `_goal_rank` re-orders, never an owner of goal evidence.
        self.goals = None
        # Game-type evidence (dual-purpose: gathered from actions we take anyway).
        # explained = a changed frame was a clean avatar translation; unexplained =
        # the frame changed in a way no translation accounts for (rotation, sprite
        # cycling, counters). Mostly-unexplained games are MECHANICAL: driving them
        # with a displacement model burns scored actions on a false premise.
        self.explained_moves = 0
        self.unexplained_changes = 0
        # An action whose displacement REVERSES sign is not a movement key --
        # it's a wrapping selector/cursor (tr87: ACTION3 = (0,+28) ... (0,-7) at
        # the wrap). Real movement keys keep a fixed direction; magnitude may
        # jitter (non-integer render scale) but the sign never flips.
        self.disp_contradictions = 0

    def reset(self):
        # Game-TYPE evidence survives GAME_OVER: whether this game is mechanical
        # is a property of the game, not of the attempt. Wiping it made the
        # classifier flip back to "movement" after every lose() (tr87's 128-move
        # budget), handing the turn back to ExecPlanner on a false premise.
        em, uc, dcon = (self.explained_moves, self.unexplained_changes,
                        self.disp_contradictions)
        gm = getattr(self, "goals", None)   # MyAgent-owned; survives every reset
        self.__init__()
        self.explained_moves, self.unexplained_changes = em, uc
        self.disp_contradictions = dcon
        self.goals = gm

    def new_level(self):
        """Layout changed: forget positional pursuit state but KEEP learned
        physics (avatar color + per-action displacement + obstacle colors)."""
        self._last_center = None
        self._stuck = 0
        self._rank_offset = 0
        self._target = None
        self._visits = {}

    # ---- learning ----
    @staticmethod
    def _centroids_by_color(grid, bg):
        out = {}
        for color in np.unique(grid):
            c = int(color)
            if c == bg:
                continue
            rs, cs = np.nonzero(grid == c)
            out[c] = (rs.mean(), cs.mean(), len(rs))
        return out

    def _infer_moved_color(self, prev, nxt, bg):
        cp, cn = self._centroids_by_color(prev, bg), self._centroids_by_color(nxt, bg)
        best, best_shift = None, 0.0
        for color, (pr, pc, pn) in cp.items():
            if color not in cn:
                continue
            nr, ncc, nn = cn[color]
            shift = abs(pr - nr) + abs(pc - ncc)
            # A single move translates the avatar by ~one avatar-size (the render
            # scale). Reject implausibly large jumps: those are level-swap
            # discontinuities, not moves. Cap at 2x the object's own bbox.
            size = max(1, min(pn, nn)) ** 0.5          # ~side length of the block
            if shift > 3.0 * max(size, self._obj_span(prev, int(color))):
                continue
            # prefer a color that translated while keeping most of its mass
            if shift > best_shift and min(pn, nn) >= 0.5 * max(pn, nn):
                best, best_shift = color, shift
        return best if best_shift >= 1.0 else None

    @staticmethod
    def _obj_span(grid, color):
        rs, cs = np.nonzero(grid == color)
        if len(rs) == 0:
            return 1.0
        return float(max(rs.max() - rs.min() + 1, cs.max() - cs.min() + 1))

    def update(self, prev, action, nxt, encoder, reward: float = 0.0):
        action = base_action(action)
        if prev is None or nxt is None or prev.shape != nxt.shape:
            return
        bg = encoder.detect_background(prev)
        if np.array_equal(prev, nxt):
            # No change: if this action normally moves the avatar, it was blocked.
            if action in self.action_disp and self.avatar_color is not None:
                self._mark_obstacle(prev, action, bg)
            return
        color = self._infer_moved_color(prev, nxt, bg)
        if color is None:
            self.unexplained_changes += 1
            return
        pr, pc, _ = self._centroids_by_color(prev, bg)[color]
        nr, ncc, _ = self._centroids_by_color(nxt, bg)[color]
        dr, dc = int(round(nr - pr)), int(round(ncc - pc))
        if dr == 0 and dc == 0:
            return
        # Diagonal-jump guard: movement actions translate along ONE axis (any
        # off-axis component is centroid-rounding noise <=1px). A strongly
        # diagonal displacement is a scene EVENT (key pickup / recolor shifting
        # the color's centroid), not physics -- learning it corrupts action_disp
        # and the ensuing sign-flips misclassify the game as mechanical
        # (keydoor regression 177 -> 939 actions, 2026-07-16). tr87's genuine
        # wrapping selectors are axis-aligned, so this does not mask them.
        if min(abs(dr), abs(dc)) > 1:
            self.unexplained_changes += 1
            return
        self.explained_moves += 1
        self.avatar_color = color
        if action in self.action_disp:
            odr, odc = self.action_disp[action]
            if dr * odr < 0 or dc * odc < 0:    # nonzero component flipped sign
                self.disp_contradictions += 1
        self.action_disp[action] = (dr, dc)
        # Goal inference from reward: the avatar just stepped onto whatever color
        # sat under its new center in `prev`. If that yielded NO reward, it's a
        # decoy, not the goal -- stop targeting it.
        if reward <= 0:
            arr, acc = np.nonzero(nxt == color)
            if len(arr):
                cr, cc = int(round(arr.mean())), int(round(acc.mean()))
                if 0 <= cr < prev.shape[0] and 0 <= cc < prev.shape[1]:
                    under = int(prev[cr, cc])
                    border = self._border_color(prev)
                    if under not in (bg, border, color) and under not in self.obstacle_colors:
                        self.non_goal_colors.add(under)

    def _mark_obstacle(self, grid, action, bg_hint):
        """A blocked move means a wall sits just past the avatar's leading edge
        (or it hit the grid boundary). Sample the band one cell beyond the bbox
        edge in the move direction; the dominant DISTINCTIVE color there is the
        wall. Never mark the play background or the letterbox border -- doing so
        would make BFS treat every empty cell as blocked (the 0.2%-style stall)."""
        dr, dc = self.action_disp[action]
        mask = (grid == self.avatar_color)
        rs, cs = np.nonzero(mask)
        if len(rs) == 0:
            return
        h, w = grid.shape
        # backgrounds we must NEVER treat as obstacles, derived consistently.
        vals, counts = np.unique(grid, return_counts=True)
        bg = int(vals[counts.argmax()])
        border = self._border_color(grid)
        r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
        step_r, step_c = int(np.sign(dr)), int(np.sign(dc))
        if step_c > 0:
            band = [(r, c1 + 1) for r in range(r0, r1 + 1)]
        elif step_c < 0:
            band = [(r, c0 - 1) for r in range(r0, r1 + 1)]
        elif step_r > 0:
            band = [(r1 + 1, c) for c in range(c0, c1 + 1)]
        elif step_r < 0:
            band = [(r0 - 1, c) for c in range(c0, c1 + 1)]
        else:
            return
        tally: Dict[int, int] = {}
        for r, c in band:
            if 0 <= r < h and 0 <= c < w:
                v = int(grid[r, c])
                if v not in (bg, border, self.avatar_color):
                    tally[v] = tally.get(v, 0) + 1
        if tally:                       # empty band => blocked by boundary, mark nothing
            self.obstacle_colors.add(max(tally, key=tally.get))

    # ---- acting ----
    def ready(self, valid) -> bool:
        if self.avatar_color is None:
            return False
        return sum(1 for a in valid if base_action(a) in self.action_disp) >= 1

    def looks_mechanical(self) -> bool:
        """Game-type classifier verdict, from evidence actions we took anyway:
        True when most frame changes are NOT avatar translations (rotation /
        sprite-cycling / counter games like ls20, tr87, tu93). The threshold
        needs real evidence (>=8 changed frames) so walk-to-goal games that
        merely animate a little are never misclassified."""
        if os.environ.get("ARC_NO_MECH_GATE"):      # debug bisect switch
            return False
        if self.disp_contradictions >= 3:           # wrapping selector, not movement
            return True
        total = self.explained_moves + self.unexplained_changes
        return total >= 8 and self.unexplained_changes / total > 0.75

    def _border_color(self, grid):
        h, w = grid.shape
        if h < 2 or w < 2:
            return None
        border = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
        vals, counts = np.unique(border, return_counts=True)
        return int(vals[counts.argmax()])

    def _nearest_target(self, grid, ar, ac):
        border = self._border_color(grid)
        # candidate goal colors: not background-ish, not the avatar, not a known obstacle
        vals, counts = np.unique(grid, return_counts=True)
        bg = int(vals[counts.argmax()])   # most common overall = play-area background
        h, w = grid.shape
        cands = []                        # (structural, nearest_dist, count, cell)
        for color, cnt in zip(vals, counts):
            c = int(color)
            if c in (bg, border, self.avatar_color) or c in self.obstacle_colors \
                    or c in self.non_goal_colors:
                continue
            rs, cs = np.nonzero(grid == c)
            # Span-based candidacy (the real-game fix): a colour stretching across
            # ~the whole board (interior walls, HUD strips) or owning a large mass
            # fraction is STRUCTURE, not a goal object. Demote it -- never pick it
            # while any discrete object color exists -- but keep it as a last
            # resort so sparse boards still yield a target.
            span = max((int(rs.max()) - int(rs.min()) + 1) / h,
                       (int(cs.max()) - int(cs.min()) + 1) / w)
            structural = span >= 0.9 or cnt / grid.size >= 0.25
            d = np.abs(rs - ar) + np.abs(cs - ac)
            i = int(d.argmin())
            cands.append((structural, int(d[i]), int(cnt), (int(rs[i]), int(cs[i]))))
        if not cands:
            return None
        # Non-structural first, then nearest (the validated old order), then rarer.
        cands.sort(key=lambda t: (t[0], t[1], t[2]))
        return self._goal_rank(grid, cands)[0][3]

    def _imagine_at(self, grid, cell):
        """The board as it would look with the avatar standing on `cell`.

        UNSCORED imagination, used only to ORDER candidates. Anti-responsibility 4
        survives intact because nothing here reaches `observe`: no credibility is
        ever updated from an imagined transition, only from a real one.

        Deliberately crude -- it relocates the avatar and changes NOTHING else. So
        it is exact for the positional readouts (`dist`, `contact`, `align`,
        `cell`) and a no-op for the rest (`count`, `regions`, `agree`), and that
        is precisely the right split when the question is 'which cell should I
        walk to': a readout that cannot tell the candidates apart should not get
        a vote on which one to walk to."""
        rs, cs = np.nonzero(grid == self.avatar_color)
        if len(rs) == 0:
            return grid
        vals, counts = np.unique(grid, return_counts=True)
        bg = int(vals[counts.argmax()])
        dr = int(cell[0]) - int(round(rs.mean()))
        dc = int(cell[1]) - int(round(cs.mean()))
        g = grid.copy()
        g[rs, cs] = bg
        nr, nc = rs + dr, cs + dc
        keep = (nr >= 0) & (nr < g.shape[0]) & (nc >= 0) & (nc < g.shape[1])
        g[nr[keep], nc[keep]] = self.avatar_color
        return g

    def _goal_rank(self, grid, cands):
        """COMPONENT 5.12 as a TARGET re-ranker: same set, new order.

        This is where the goal model belongs, because "the goal is the nearest
        distinctive object" is exactly the decoy walk Section 5 names as THE
        hazard, and `non_goal_colors` is the scar it already left -- a patch that
        only ever learns NEGATIVELY, one wasted pursuit at a time, and only for
        colours. The goal model is the positive form of that same learning.

        Three properties hold no matter what the potential says:
          - it never removes a candidate (guard 1: a permutation, never a filter),
            so the worst case of an arbitrarily wrong Phi is a reordering;
          - `structural` stays the FIRST key, so a wall or a HUD strip can never
            be promoted above a discrete object -- that ordering was validated
            against real games and is not the goal model's to overrule;
          - it does nothing at all unless `steering()`, i.e. some feature holds an
            EARNED posterior at or above the pursuit floor. An UNEARNED potential
            was MEASURED to be a coin flip (Feature.steers), so it does not get to
            pick the destination here either.
        Distance drops below Phi only inside a `structural` class, and only once
        that gate is open: "nearest" is a prior about where goals tend to be, and
        an earned belief about what a goal IS outranks a prior about where."""
        if (self.goals is None or len(cands) < 2
                or os.environ.get("ARC_NO_GOALS")):
            return cands
        try:
            if not self.goals.steering():
                return cands
            phi = [self.goals.potential(self._imagine_at(grid, t[3]))[0]
                   for t in cands]
        except Exception:
            return cands            # a broken re-ranker must cost the agent nothing
        order = sorted(range(len(cands)), key=lambda i: (cands[i][0], -phi[i], i))
        out = [cands[i] for i in order]
        if out[0] is not cands[0]:
            # The model CHANGED the destination, so it may now be credited -- or
            # blamed -- for what follows. Without this the pursuit stays
            # observation-only and can only ever lose credibility.
            self.goals.note_directed()
        return out

    def _scale(self, grid):
        """Render scale = size of the avatar block (1 logical cell). The lattice
        step for planning; makes everything scale-invariant."""
        rs, cs = np.nonzero(grid == self.avatar_color)
        if len(rs) == 0:
            return 1
        return max(1, min(rs.max() - rs.min() + 1, cs.max() - cs.min() + 1))

    def _bfs_action(self, grid, valid, ar, ac) -> Optional[str]:
        """Shortest path on the scale-lattice, avoiding learned obstacle colors.
        Each block is a point at its center; a move is valid if the destination
        center isn't an obstacle. Returns the first action of the path."""
        moves = [(a, self.action_disp[base_action(a)]) for a in valid
                 if base_action(a) in self.action_disp]
        if not moves:
            return None
        h, w = grid.shape
        border = self._border_color(grid)
        vals, counts = np.unique(grid, return_counts=True)
        bg = int(vals[counts.argmax()])
        obstacles = set(self.obstacle_colors)
        if border is not None:
            obstacles.add(border)
        targets = {int(v) for v in vals
                   if int(v) not in (bg, border, self.avatar_color)
                   and int(v) not in self.obstacle_colors
                   and int(v) not in self.non_goal_colors}
        if not targets:
            return None

        def blocked(r, c):
            return not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) in obstacles

        start = (int(round(ar)), int(round(ac)))
        from collections import deque
        seen = {start}
        q = deque([(start, None)])
        while q:
            (r, c), first = q.popleft()
            if int(grid[r, c]) in targets and first is not None:
                return first
            for a, (dr, dc) in moves:
                nr, nc = r + dr, c + dc
                if (nr, nc) in seen or blocked(nr, nc):
                    continue
                seen.add((nr, nc))
                q.append(((nr, nc), a if first is None else first))
        return None

    def act(self, grid, valid, encoder) -> Optional[str]:
        if self.avatar_color is None:
            return None
        rs, cs = np.nonzero(grid == self.avatar_color)
        if len(rs) == 0:
            return None
        ar, ac = rs.mean(), cs.mean()
        target = self._nearest_target(grid, ar, ac)
        self._target = target
        if target is None:
            self._sub = "notarget"
            return None
        tr, tc = target

        # Prefer a true shortest path around known obstacles (also near-optimal
        # action count -> better efficiency score). Greedy is the fallback that
        # bumps unknown walls to LEARN they are obstacles.
        bfs = self._bfs_action(grid, valid, ar, ac)
        if bfs is not None:
            self._last_center = (ar, ac)
            self._sub = "bfs1"
            return bfs

        moves = [a for a in valid if base_action(a) in self.action_disp]
        if not moves:
            self._sub = "nomoves"
            return None

        # stuck detection: centroid hasn't moved -> rotate to the next-best action
        center = (ar, ac)
        if self._last_center is not None and \
                abs(center[0] - self._last_center[0]) < 0.5 and \
                abs(center[1] - self._last_center[1]) < 0.5:
            self._stuck += 1
        else:
            self._stuck = 0
            self._rank_offset = 0
        self._last_center = center
        if self._stuck >= self.STUCK_LIMIT:
            self._rank_offset += 1
            self._stuck = 0

        # Visited-lattice memory: greedy oscillation (A<->B at a wall while BFS
        # has no route) is self-defeating if revisits cost. Rank moves first by
        # how often we've SEEN the destination cell, then by goal distance.
        scale = max(1, self._scale(grid))
        cell = (int(round(ar / scale)), int(round(ac / scale)))
        self._visits[cell] = self._visits.get(cell, 0) + 1

        def rank(a):
            dr, dc = self.action_disp[base_action(a)]
            dest = (int(round((ar + dr) / scale)), int(round((ac + dc) / scale)))
            dist = abs((ar + dr) - tr) + abs((ac + dc) - tc)
            return (self._visits.get(dest, 0), dist)

        moves.sort(key=rank)
        self._sub = "greedy"
        return moves[self._rank_offset % len(moves)]


# ==========================================
# COMPONENT 5.7: EXECUTABLE WORLD MODEL PLANNER (RHAE-critical)
# ==========================================
#
# The two techniques the SOTA "Executable World Models" paper and the competition
# dossier both prescribe, aimed squarely at RHAE's SQUARED action-ratio penalty --
# spending SCORED actions to gather information is fatal (10x human actions -> 1%):
#
#   T1  EXECUTABLE WORLD MODEL + UNSCORED PLAN-THEN-EXECUTE.  Learn the transition
#       dynamics from observation, then search a WHOLE path to the goal INSIDE the
#       model (test-time compute, zero scored actions), commit to it, and execute
#       one vetted step per turn. Re-plan only when reality FALSIFIES the model (a
#       committed step didn't do what the model predicted) -- never oscillate,
#       never take a scored random move once a plan exists.
#
#   T2  FALSIFICATION / CURIOSITY-DRIVEN EXPLORATION.  When no goal path is known
#       yet, don't flail (epsilon-random burns scored actions): steer to the
#       nearest UNVISITED lattice cell so the model is completed in the fewest
#       scored actions -- the second-biggest RHAE lever after not-searching-with-
#       real-actions.
#
# ExecPlanner SUBCLASSES ReactivePlanner: it inherits the validated physics
# learning (per-action displacement, avatar/obstacle/decoy colours, goal-from-
# reward) and ADDS a POSITIONAL wall map (precise, per-cell -- vs the parent's
# coarse per-COLOUR obstacles), a committed plan queue, falsification-driven
# replanning, and directed frontier exploration. It degrades to the parent's
# greedy behaviour when it has nothing better, so the validated toys don't regress.

class ExecPlanner(ReactivePlanner):
    MAX_BFS = 6000          # lattice cells expanded before abandoning a search
    TIE_EPS = 0.05          # tiny random tie-break -> breaks deterministic livelocks
    STUCK_ESCAPE = 6        # steps of no-progress on goal-pursuit before exploring
    ESCAPE_BURST = 4        # steps of directed exploration once triggered
    EXP_TRIES = 2           # re-tests of a disputed (cell, action) before it's an anomaly
    CERT_REPLAY_GAP = 5     # min timeline growth between full-history replays (cost cap)

    def __init__(self):
        super().__init__()
        self.walls: set = set()                    # LOGICAL cells known blocked (bumped)
        self._plan: deque = deque()                # committed (expected_cell, action) steps
        self._cur_scale = 1
        self._last_cell: Optional[Tuple[int, int]] = None
        self._stuck_n = 0
        self._escape_n = 0
        # --- Schema-harness Phase 1 state (CERTIFY / experiments) ---
        self.timeline = None                       # set by MyAgent; append-only ground truth
        self._anomalies: set = set()               # (cell, action): consistently out-of-theory
        self._excused: set = set()                 # timeline indices proven one-off events
        self._cert_key = None                      # theory fingerprint of the last replay
        self._cert_idx = 0                         # timeline entries already replayed
        self._cert_green = True
        self._cert_fail = None                     # (timeline index, entry) counterexample
        self._last_replay_at = -10**9              # timeline length at last full replay
        self._exp = None                           # ((cell, action), tries_left) experiment
        self._unwalled: set = set()                # colours already un-marked this level
        # --- Schema-harness Phase 2 state (LLM theorizer) ---
        self._llm_goal: Optional[Tuple[int, int]] = None   # adopted goal hint (pixel cell)

    def reset(self):
        # __init__ re-runs on reset (via the parent), which would sever the
        # MyAgent-owned timeline reference and forget which (cell, action) pairs
        # were accepted as anomalies -- both are properties of the GAME/level
        # record, not of the attempt (same layout after GAME_OVER -> RESET).
        tl = getattr(self, "timeline", None)
        anom = getattr(self, "_anomalies", set())
        exc = getattr(self, "_excused", set())
        unw = getattr(self, "_unwalled", set())
        lg = getattr(self, "_llm_goal", None)      # same layout after GAME_OVER -> keep
        super().reset()     # re-inits fully but preserves game-type evidence
        self.timeline = tl
        self._anomalies = anom
        self._excused = exc
        self._unwalled = unw
        self._llm_goal = lg

    def new_level(self):
        super().new_level()
        # Layout changed: positional walls + committed plan are stale; physics stays.
        self.walls = set()
        self._plan = deque()
        self._last_cell = None
        self._stuck_n = 0
        self._escape_n = 0
        # Positional certification state belongs to the old layout. The timeline
        # itself is append-only; the level tag keeps old evidence out of scope.
        self._anomalies = set()
        self._excused = set()
        self._cert_key = None
        self._cert_idx = 0
        self._cert_green = True
        self._cert_fail = None
        self._last_replay_at = -10**9
        self._exp = None
        self._unwalled = set()
        self._llm_goal = None      # hint reasoned about the old layout

    # ---- lattice helpers ----
    def _log(self, rc) -> Tuple[int, int]:
        # FLOOR, not round: avatar-block centroids sit exactly on N.5 scale
        # boundaries, and banker's rounding (round(5.5)=6, round(6.5)=6) collapses
        # adjacent grid cells onto one logical cell -- which corrupts the goal-cell
        # match and desyncs the committed plan. Floor gives a clean bijective lattice.
        s = max(1, self._cur_scale)
        return (int(rc[0] // s), int(rc[1] // s))

    def _obstacle_set(self, grid):
        # The play background must NEVER be treated as a wall -- when it equals the
        # letterbox border colour (common: both 0), adding the border would block
        # every cell and BFS would always fail (the border-poisoning stall).
        vals, counts = np.unique(grid, return_counts=True)
        bg = int(vals[counts.argmax()])
        border = self._border_color(grid)
        obs = set(self.obstacle_colors)
        if border is not None and border != bg:
            obs.add(border)
        obs.discard(bg)
        if self.avatar_color is not None:
            obs.discard(self.avatar_color)
        return obs

    def _blocked(self, grid, r, c, obstacles) -> bool:
        h, w = grid.shape
        if not (0 <= r < h and 0 <= c < w):
            return True
        if self._log((r, c)) in self.walls:          # learned by falsification (bump)
            return True
        return int(grid[r, c]) in obstacles           # generalised by colour

    def _bfs_path(self, grid, valid, ar, ac, is_goal) -> Optional[List[str]]:
        """UNSCORED search through the learned model: shortest action path from the
        avatar to the first cell satisfying is_goal, routing around known walls and
        obstacle colours. Returns the whole path (T1 commits to it)."""
        moves = [(a, self.action_disp[base_action(a)]) for a in valid
                 if base_action(a) in self.action_disp]
        if not moves:
            return None
        obstacles = self._obstacle_set(grid)
        # Key the frontier by EXACT pixel (like the validated parent BFS): rounding
        # to logical cells can collapse adjacent lattice steps onto one cell (banker's
        # rounding) and sever connectivity. Walls/visits stay in logical space.
        start = (int(round(ar)), int(round(ac)))
        seen = {start}
        q = deque([(start, [])])
        while q:
            (r, c), path = q.popleft()
            if path and is_goal(r, c):
                return path
            if len(seen) > self.MAX_BFS:
                break
            for a, (dr, dc) in moves:
                # An accepted anomaly = this action's outcome at this cell is
                # KNOWN not to follow the theory -- never route a plan through it.
                if self._anomalies and (self._log((r, c)), base_action(a)) in self._anomalies:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in seen or self._blocked(grid, nr, nc, obstacles):
                    continue
                seen.add((nr, nc))
                q.append(((nr, nc), path + [a]))
        return None

    def _commit(self, ar, ac, path):
        """Record the plan as (expected_from_cell, action) steps so act() can verify
        the avatar is still where the model predicted before executing each step."""
        seq = deque()
        r, c = int(round(ar)), int(round(ac))
        for a in path:
            seq.append((self._log((r, c)), a))
            dr, dc = self.action_disp[base_action(a)]
            r, c = r + dr, c + dc
        self._plan = seq

    # ---- CERTIFY (Schema Phase 1): full-history backtest of the movement theory ----
    def _explain(self, entry) -> Optional[bool]:
        """Does the CURRENT theory (per-action displacement + positional walls +
        obstacle colours) reproduce this recorded transition?
          None  = out of scope (non-movement action, avatar invisible, scene
                  event, or an accepted anomaly);
          True  = prediction matches reality;
          False = counterexample -- the record falsifies the theory."""
        a = base_action(entry.action)
        if self.avatar_color is None or a not in self.action_disp:
            return None
        m = entry.motion(self.avatar_color)
        if m is None:
            return None
        pr, pc, dr, dc = m
        if min(abs(dr), abs(dc)) > 1:       # diagonal jump = scene event, not motion
            return None
        self._cur_scale = max(1, self._scale(entry.prev))
        if (self._log((int(round(pr)), int(round(pc)))), a) in self._anomalies:
            return None                     # documented exception, out of scope
        ed, ec = self.action_disp[a]
        tr, tc = int(round(pr)) + ed, int(round(pc)) + ec
        if self._blocked(entry.prev, tr, tc, self._obstacle_set(entry.prev)):
            ped, pec = 0, 0                 # theory: move vetoed, avatar stays
        else:
            ped, pec = ed, ec
        # <=1px slack absorbs centroid rounding on multi-pixel sprites.
        # bool(): numpy comparisons yield np.bool_, and np.False_ `is False` is
        # False -- the caller's identity check would silently never fire.
        return bool(abs(dr - ped) <= 1 and abs(dc - pec) <= 1)

    def certify(self) -> bool:
        """The Schema CERTIFY gate: the movement theory must reproduce EVERY
        in-scope recorded transition of the current level before act() may
        commit a multi-step plan through it. Incremental (new entries checked
        as they arrive); any revision of the theory itself -- displacement,
        walls, obstacle colours, anomalies, excusals -- forces a full replay,
        because a revised theory must be re-verified against ALL evidence, not
        just the transition that prompted the revision. Full replays are
        throttled to every CERT_REPLAY_GAP entries so displacement jitter can't
        turn certification into an O(n^2) per-step cost."""
        if os.environ.get("ARC_NO_CERT"):   # debug bisect switch: always green
            return True
        tl = self.timeline
        if tl is None or not tl.entries:
            return True
        level = tl.entries[-1].level
        key = (self.avatar_color, tuple(sorted(self.action_disp.items())),
               frozenset(self.obstacle_colors), frozenset(self.walls),
               len(self._anomalies), len(self._excused), level)
        if key != self._cert_key and \
                len(tl.entries) - self._last_replay_at >= self.CERT_REPLAY_GAP:
            self._cert_key = key
            self._cert_idx = 0
            self._cert_fail = None
            self._cert_green = True
            self._last_replay_at = len(tl.entries)
        for i in range(self._cert_idx, len(tl.entries)):
            e = tl.entries[i]
            if e.level != level or i in self._excused:
                continue
            if self._explain(e) is False:
                self._cert_green = False
                self._cert_fail = (i, e)
        self._cert_idx = len(tl.entries)
        return self._cert_green

    def _experiment(self, grid, valid, ar, ac) -> Optional[str]:
        """Discriminating experiment (Schema 'action for discovery'): when the
        record contradicts the theory, the single most informative real action
        is to RE-RUN the disputed (cell, action) -- one scored step splits
        'one-off scene event' (excuse the recorded entry, theory stands) from
        'consistent rule the theory lacks' (after EXP_TRIES reproductions the
        pair becomes an anomaly: certification puts it out of scope and the
        planner stops routing through it). Steps toward the disputed cell are
        taken singly (no committed queue -- the theory is red here)."""
        if self._cert_fail is None:
            return None
        _, e = self._cert_fail
        a = base_action(e.action)
        m = e.motion(self.avatar_color) if self.avatar_color is not None else None
        if m is None or a not in self.action_disp:
            self._cert_fail = None
            return None
        cell = self._log((int(round(m[0])), int(round(m[1]))))
        if self._exp is None or self._exp[0] != (cell, a):
            self._exp = ((cell, a), self.EXP_TRIES)
        if self._exp[1] <= 0:
            # Reproduced EXP_TRIES times: a consistent rule outside the theory.
            # Joint state+rule revision (VIGA / WorldCoder lineage, Schema sec.4):
            # before accepting an anomaly, check whether the counterexample
            # indicts the REPRESENTATION -- the colour-obstacle / wall grounding
            # -- rather than the displacement rule. Only when no revision
            # applies does the (cell, action) become a behavioural anomaly.
            revised = self._revise_representation(e)
            self._exp = None
            self._cert_fail = None
            if not revised:
                self._anomalies.add((cell, a))
            return None
        if self._log((int(round(ar)), int(round(ac)))) == cell:
            exp_a = next((v for v in valid if base_action(v) == a), None)
            if exp_a is not None:
                self._exp = (self._exp[0], self._exp[1] - 1)
                return exp_a
            return None
        path = self._bfs_path(grid, valid, ar, ac,
                              lambda r, c: self._log((r, c)) == cell)
        return path[0] if path else None

    def _revise_representation(self, entry) -> bool:
        """VIGA/WorldCoder joint revision: a reproduced counterexample can mean
        the STATE GROUNDING is wrong, not the transition rule.
          - Theory said BLOCKED but the avatar moved through: the colour marked
            as an obstacle is actually walkable (Schema's 'the cart is a valid
            landing cell') -> un-mark it (once per level) / drop the stale wall.
          - Theory said FREE but the avatar stayed: an invisible barrier the
            colour layer can't see -> record a positional wall.
        Returns True when a revision was made; certification then re-verifies
        the WHOLE record against the revised grounding (full replay)."""
        a = base_action(entry.action)
        m = entry.motion(self.avatar_color) if self.avatar_color is not None else None
        if m is None or a not in self.action_disp:
            return False
        pr, pc, dr, dc = m
        ed, ec = self.action_disp[a]
        self._cur_scale = max(1, self._scale(entry.prev))
        tr, tc = int(round(pr)) + ed, int(round(pc)) + ec
        h, w = entry.prev.shape
        if not (0 <= tr < h and 0 <= tc < w):
            return False
        blocked_pred = self._blocked(entry.prev, tr, tc, self._obstacle_set(entry.prev))
        moved = max(abs(dr), abs(dc)) > 1
        if blocked_pred and moved:
            c = int(entry.prev[tr, tc])
            if c in self.obstacle_colors and c not in self._unwalled:
                self.obstacle_colors.discard(c)
                self._unwalled.add(c)
                self.walls.discard(self._log((tr, tc)))
                return True
            wcell = self._log((tr, tc))
            if wcell in self.walls:
                self.walls.discard(wcell)
                return True
            return False
        if not blocked_pred and not moved:
            self.walls.add(self._log((tr, tc)))
            return True
        return False

    # ---- Schema Phase 2: LLM theorizer proposals, gated by the backtest ----
    def _backtest_mismatches(self) -> int:
        """Full-replay mismatch count of the CURRENT theory against the current
        level's timeline (excused entries skipped). The theorizer's acceptance
        metric: a proposal must not explain LESS of the record than the theory
        it replaces."""
        tl = self.timeline
        if tl is None or not tl.entries:
            return 0
        level = tl.entries[-1].level
        n = 0
        for i, e in enumerate(tl.entries):
            if e.level != level or i in self._excused:
                continue
            if self._explain(e) is False:
                n += 1
        return n

    def propose_theory(self, proposal: dict, grid) -> bool:
        """Adopt an LLM-proposed theory revision ONLY if the full-history
        backtest does not get worse (Schema: the model is provisional, the
        Timeline is ground truth -- a proposal the record contradicts is
        reverted wholesale). Grounding edits are applied tentatively; the goal
        hint needs no backtest because it only REDIRECTS planning, which stays
        green-gated and per-step vetted. Returns True if anything was adopted."""
        if not isinstance(proposal, dict):
            return False
        adopted = False
        snap = (set(self.obstacle_colors), set(self.non_goal_colors), set(self.walls))
        before = self._backtest_mismatches()
        changed = False
        for c in proposal.get("not_obstacle_colors") or []:
            try:
                c = int(c)
            except (TypeError, ValueError):
                continue
            if c in self.obstacle_colors:
                self.obstacle_colors.discard(c)
                changed = True
        for c in proposal.get("obstacle_colors") or []:
            try:
                c = int(c)
            except (TypeError, ValueError):
                continue
            if c != self.avatar_color and c not in self.obstacle_colors:
                self.obstacle_colors.add(c)
                changed = True
        for c in proposal.get("decoy_colors") or []:
            try:
                c = int(c)
            except (TypeError, ValueError):
                continue
            if c != self.avatar_color:
                self.non_goal_colors.add(c)      # only affects target choice
        if changed:
            after = self._backtest_mismatches()
            if after > before:
                # The record contradicts the proposal: revert EVERYTHING it
                # touched (decoys included -- an untrustworthy theory gets no
                # partial credit).
                self.obstacle_colors, self.non_goal_colors, self.walls = snap
            else:
                adopted = True
                # Revised grounding => certification must re-verify the whole
                # record before the next multi-step commit.
                self._cert_key = None
                self._last_replay_at = -10**9
        g = proposal.get("goal")
        if isinstance(g, (list, tuple)) and len(g) == 2:
            try:
                r, c = int(g[0]), int(g[1])
            except (TypeError, ValueError):
                r = c = -1
            h, w = grid.shape
            if 0 <= r < h and 0 <= c < w:
                self._llm_goal = (r, c)
                adopted = True
        return adopted

    # ---- learning: positional wall on a falsified (blocked) move ----
    def update(self, prev, action, nxt, encoder, reward: float = 0.0):
        super().update(prev, action, nxt, encoder, reward)   # physics + colour obstacles
        if prev is None or nxt is None or prev.shape != nxt.shape:
            return
        a = base_action(action)
        if a in self.action_disp and self.avatar_color is not None and np.array_equal(prev, nxt):
            # The move was VETOED by the world -> the cell just past the avatar's
            # centre is a wall. Record it POSITIONALLY (precise) and drop the plan;
            # act() will replan around it. This is the falsification -> replan loop.
            self._cur_scale = max(1, self._scale(prev))
            rs, cs = np.nonzero(prev == self.avatar_color)
            if len(rs):
                dr, dc = self.action_disp[a]
                self.walls.add(self._log((rs.mean() + dr, cs.mean() + dc)))
            self._plan.clear()
        # Resolve a pending discriminating experiment on the outcome of its
        # disputed action (this transition is the newest timeline entry).
        if (self._exp is not None and self.timeline is not None
                and self.timeline.entries and self.avatar_color is not None):
            (cell, ea), _tries = self._exp
            if a == ea:
                i = len(self.timeline.entries) - 1
                e = self.timeline.entries[i]
                m = e.motion(self.avatar_color)
                if m is not None and \
                        self._log((int(round(m[0])), int(round(m[1])))) == cell:
                    if self._explain(e) is not False:
                        # Theory held on the re-test: the recorded contradiction
                        # was a one-off scene event -- excuse that entry only.
                        if self._cert_fail is not None:
                            self._excused.add(self._cert_fail[0])
                        self._exp = None
                        self._cert_fail = None

    # ---- acting: committed plan -> plan to goal -> frontier -> greedy ----
    def act(self, grid, valid, encoder) -> Optional[str]:
        if self.avatar_color is None:
            self._sub = "noavatar"
            return None
        rs, cs = np.nonzero(grid == self.avatar_color)
        if len(rs) == 0:
            self._sub = "noavatar"
            return None
        ar, ac = rs.mean(), cs.mean()
        # CERTIFY before anything else (mutates _cur_scale per replayed entry,
        # so re-derive the live scale after it).
        green = self.certify()
        self._cur_scale = max(1, self._scale(grid))
        cur = self._log((int(round(ar)), int(round(ac))))
        valid_moves = {a for a in valid if base_action(a) in self.action_disp}
        if not green:
            # Reality outranks the model: an outstanding counterexample voids
            # the committed queue (Schema's per-step mismatch rule).
            self._plan.clear()

        # Progress tracking: a genuine step changes our logical cell. If goal-pursuit
        # leaves the cell unchanged for STUCK_ESCAPE steps it is livelocked (e.g. a
        # non-integer render scale makes the fixed-displacement model oscillate).
        if cur == self._last_cell:
            self._stuck_n += 1
        else:
            self._stuck_n = 0
        self._last_cell = cur
        self._visits[cur] = self._visits.get(cur, 0) + 1

        # (1) T1 -- consume the committed plan WHILE reality still matches the model.
        if self._plan:
            exp_cell, a = self._plan[0]
            if exp_cell == cur and a in valid_moves:
                self._plan.popleft()
                self._sub = "plan"
                return a
            self._plan.clear()          # desync (interrupt / falsified) -> replan below

        # (1b) Theory red -> spend the next scored action on the discriminating
        #      experiment (re-test the disputed transition), not on pursuing a
        #      goal through a model the record has already falsified.
        if not green:
            exp = self._experiment(grid, valid, ar, ac)
            if exp is not None:
                self._sub = "exper"
                return exp

        # Schema Phase 2: an adopted theorizer GOAL HINT outranks the local
        # nearest-distinctive-colour heuristic (the documented real-game gap:
        # layered games' goals are not "nearest odd colour"). The hint only
        # redirects planning -- commits stay green-gated and per-step vetted --
        # and is dropped the moment reality discredits it (reached without a
        # level-up, or pursuit livelocks below).
        target = None
        if self._llm_goal is not None:
            if self._log(self._llm_goal) == cur:
                self._llm_goal = None       # arrived, no level-up -> hint was wrong
            else:
                target = self._llm_goal
        if target is None:
            target = self._nearest_target(grid, ar, ac)
        self._target = target

        # T2 -- livelock escape: goal-pursuit stopped making progress, so the belief
        # "I have a path to the goal" is FALSIFIED. Spend a short DIRECTED-exploration
        # burst to reach new ground (not epsilon-random -- RHAE squares wasted actions),
        # then resume pursuit from a fresh position/angle.
        if self._escape_n == 0 and self._stuck_n >= self.STUCK_ESCAPE:
            self._escape_n = self.ESCAPE_BURST
            self._plan.clear()
            self._llm_goal = None   # pursuit livelocked -> the hint is falsified too
        exploring = self._escape_n > 0
        if exploring:
            self._escape_n -= 1

        # (2) T1 -- goal known & not escaping: plan a WHOLE path to it through the
        #     learned model (unscored search) and commit. Optimal actions -> best RHAE.
        if target is not None and not exploring:
            tcell = self._log(target)
            path = self._bfs_path(grid, valid, ar, ac,
                                  lambda r, c: self._log((r, c)) == tcell)
            if path:
                if green:
                    self._commit(ar, ac, path)
                    self._sub = "goal"
                    return self._plan.popleft()[1]
                # ONLY PLAN WHEN GREEN: the theory has an outstanding
                # counterexample, so a committed queue would be built on a
                # falsified model -- take one vetted step instead; every real
                # transition re-enters certification before the next step.
                self._sub = "goal1"
                return path[0]
            # Goal known but no MODELLED path (unknown walls / lattice offset): defer
            # to the parent's validated greedy pursuit -- it drives toward the target,
            # bumping and LEARNING obstacles, with stuck-rotation to escape livelocks.
            return super().act(grid, valid, encoder)

        # (3) T2 -- no goal yet, or escaping a livelock: steer to the nearest UNVISITED
        #     cell (directed exploration) so the model is completed in the fewest
        #     SCORED actions.
        path = self._bfs_path(grid, valid, ar, ac,
                              lambda r, c: self._visits.get(self._log((r, c)), 0) == 0)
        if path:
            if green:
                self._commit(ar, ac, path)
                self._sub = "frontier"
                return self._plan.popleft()[1]
            self._sub = "frontier1"
            return path[0]      # red: single vetted step, no committed queue

        # (4) Fallback -- whole reachable region explored / boxed in: parent's
        #     least-visited greedy with stuck-rotation breaks the livelock.
        self._escape_n = 0
        return super().act(grid, valid, encoder)


# ==========================================
# COMPONENT 5.55: PROGRESS MODEL (objective O3)
# ==========================================
#
# The agent's world models were faithful (O1), cheap (O2) and safe (O4) and still
# scored ~0, because none of them could answer the only question that pays:
# WHICH BOARD IS CLOSER TO WINNING? ClickPlanner's O3 was "an untried (state, button)
# pair is worth one unit" -- a COVERAGE objective. Every untried pair scores the same,
# so lp85's 251 explored level-2 boards were all equally attractive and the search had
# no opinion about any of them.
#
# Why this is where the score is: RHAE's denominator spans EVERY level of a game, so on
# lp85 (8 levels, sum(i) = 36) the entire remaining efficiency headroom on level 1 is
# +3.33 points, one more completed level is +5.56, and the rest of the game is +94.4.
# 23 of 25 games complete zero levels, where efficiency work is worth exactly nothing.
#
# THE SIGNAL PROBLEM. The only unambiguous "good state" is a level completion, and that
# is the very event the agent cannot produce -- GoalModel's alpha channel starved for
# exactly this reason. So O3 here has two channels, and they are not equals:
#
#   GROUNDED   -- when a level IS completed, the trajectory that reached it is ground
#                 truth. Features that moved MONOTONICALLY along it are what progress
#                 looked like, in this game. lp85 completes level 1, so level 2 is
#                 searched with a progress model fitted on level 1. This is also the
#                 only channel that can TRANSFER across levels, which the RHAE
#                 denominator makes mandatory rather than nice.
#   PRIOR      -- with no win ever observed (23 of 25 games), all that is left is a
#                 generic guess: puzzles resolve toward CONSOLIDATED boards (fewer
#                 colour boundaries). Labelled a prior because that is what it is, and
#                 kept OFF by default (ARC_O3_PRIOR=1) until it is measured on its own.
#
# O1 ON O3 (the teeth -- the doc's rule is that a model which cannot be switched off by
# its own error signal does not ship): every later winning trajectory BACKTESTS the
# fitted model. If the score was not monotone along a trajectory that demonstrably WAS
# progress, the model is wrong and retires itself.
#
# O4: this component only ORDERS the frontier. It never prunes a state, never marks an
# edge dead, never vetoes a plan. A wrong O3 costs ordering; it cannot make a reachable
# level unreachable. Before the first win it returns a constant, so it is a strict no-op
# until it has evidence.

class ProgressModel:
    """O3: a scalar over boards, higher = closer to winning. Learned, falsifiable, and
    incapable of blocking anything (see the component note above)."""

    N_COLORS = 16
    # C1 -- THE BASIS. The first 20 slots describe the board's PIXELS; the last 10
    # describe the THINGS ON IT. Progress in these puzzles is object-level ("3 of the 5
    # blocks are on targets"), and a colour histogram can only approximate that by
    # accident -- which is what the measurement showed: on lp85's walk basis the
    # histogram earned 2 directions and BOTH were confounded with elapsed time (0 usable
    # features), while the object descriptor earned 1 that matched 0% of deaths. The
    # object half is not richer, it is CLEANER. Both halves are kept: the A1 control
    # deletes whichever turn out to be time, so carrying the histogram costs nothing but
    # arithmetic, and dropping it would be a guess in the other direction.
    N_OBJ = 10
    N_FEAT = N_COLORS + 4 + N_OBJ   # colour histogram + boundary/extent/syms + objects
    MIN_TRACE = 4                   # shorter than this says nothing about a direction
    MONOTONE_MIN = 0.70             # step-agreement needed to call a feature monotone
    BACKTEST_MIN = 0.55             # below this on a later win, the model retires
    BUCKETS = 8                     # progress is bucketed so O2 still breaks ties
    DEATH_MIN = 2                   # distinct death paths before the control may act

    def __init__(self):
        self._dir = np.zeros(self.N_FEAT, dtype=np.float64)     # signed weight per feature
        self._fits = 0
        self._retired = False
        self._backtests: List[float] = []                       # evidence, not decoration
        # A1 -- THE CONTROL GROUP. Same extraction, run on trajectories that ended in a
        # GAME_OVER. A feature that climbs on the way to a win AND on the way to a death
        # is not describing progress, it is describing elapsed time; with a single win
        # those two are otherwise indistinguishable. Measured on lp85 (scratchpad/
        # diag_o3.py): of the 3 features the win taught, 2 moved the SAME way on 100% of
        # the distinct death paths, and on the walk-trace basis 2 of 2 did -- i.e. the
        # entire learned direction was elapsed time. Deaths are also the plentiful label:
        # 46 per tier-1 sweep against 1 win.
        self._dir_death = np.zeros(self.N_FEAT, dtype=np.float64)
        self._death_fits = 0
        # Distinct paths only. lp85 records 11 GAME_OVERs but replays just 6 distinct
        # graph paths; without this the control would be a vote by whichever path the
        # search happened to repeat most, not by independent evidence.
        self._death_sigs: set = set()
        self._eff: Optional[np.ndarray] = None                  # cache, see effective()

    # -- features ---------------------------------------------------------------
    @staticmethod
    def features(grid: np.ndarray) -> np.ndarray:
        """A small, TOTAL, scale-free descriptor of a board. Every component is a
        fraction in [0, 1], so no feature can dominate the score by unit choice."""
        f = np.zeros(ProgressModel.N_FEAT, dtype=np.float64)
        try:
            g = np.asarray(grid)
            if g.ndim != 2 or g.size == 0:
                return f
            n = float(g.size)
            flat = np.clip(g.ravel().astype(np.int64), 0, ProgressModel.N_COLORS - 1)
            counts = np.bincount(flat, minlength=ProgressModel.N_COLORS)
            f[:ProgressModel.N_COLORS] = counts[:ProgressModel.N_COLORS] / n
            k = ProgressModel.N_COLORS
            h, w = g.shape
            # CONSOLIDATION: fraction of adjacent cell pairs whose colours differ.
            # A cheap stand-in for "how many objects" that costs two numpy compares
            # instead of a connected-components pass on every frontier state.
            pairs = h * max(0, w - 1) + max(0, h - 1) * w
            if pairs:
                diff = float((g[:, 1:] != g[:, :-1]).sum() + (g[1:, :] != g[:-1, :]).sum())
                f[k] = diff / pairs
            # EXTENT of the non-background content (background = most common colour).
            bg = int(np.argmax(counts))
            occ = g != bg
            if occ.any():
                rs, cs = np.where(occ)
                f[k + 1] = ((rs.max() - rs.min() + 1) * (cs.max() - cs.min() + 1)) / n
            # SYMMETRY: many puzzles resolve toward an aligned/mirrored configuration.
            f[k + 2] = float((g == g[:, ::-1]).mean())
            f[k + 3] = float((g == g[::-1, :]).mean())
            # C1: the object half. Costs one flood fill (~0.8ms, cached 256-deep in
            # StateEncoder), which is the same order as the histogram half above.
            if ProgressModel._OBJ_ON:
                f[k + 4:] = ProgressModel._object_block(g, n)
        except Exception:
            return np.zeros(ProgressModel.N_FEAT, dtype=np.float64)
        return f

    # A shared encoder, because its flood-fill cache is keyed by grid bytes and bounded
    # at 256 entries -- a per-call StateEncoder would throw that cache away every frame.
    _ENC: Optional["StateEncoder"] = None
    _OBJ_ON = not bool(os.environ.get("ARC_NO_OBJFEAT"))     # C1 kill switch

    @staticmethod
    def _object_block(g: np.ndarray, n: float) -> np.ndarray:
        """Ten scale-free descriptors of the OBJECTS on the board. Same contract as the
        rest of the vector: every entry is a fraction in [0, 1], and this is TOTAL --
        the caller's except clause is a backstop, not the plan.

        Background comes from detect_background, NOT from the histogram argmax used by
        `extent` above. The two disagree exactly where it matters: on a framed or
        multi-panel board the commonest colour is often the chrome, and handing that to
        the flood fill collapses the whole play field into one phantom object -- the
        failure detect_background's border/concentration rules exist to prevent."""
        o = np.zeros(ProgressModel.N_OBJ, dtype=np.float64)
        if ProgressModel._ENC is None:
            ProgressModel._ENC = StateEncoder()
        bg = ProgressModel._ENC.detect_background(g)
        objs = ProgressModel._ENC.objects(g, bg)
        if not objs:
            return o
        sizes = np.array([obj.size for obj in objs], dtype=np.float64)
        o[0] = min(1.0, len(objs) / 64.0)                    # how many things
        o[1] = len(set(int(obj.color) for obj in objs)) / float(ProgressModel.N_COLORS)
        o[2] = float(sizes.max()) / n                        # biggest thing
        o[3] = float(sizes.mean()) / n
        o[4] = float(sizes.std()) / n                        # uniformity of the things
        o[5] = float((sizes == 1).sum()) / float(len(objs))   # loose debris
        o[6] = float((g == bg).sum()) / n                     # how much is empty
        cs = np.array([obj.center for obj in objs], dtype=np.float64)
        o[7] = float(cs.std()) / float(max(g.shape))          # scattered vs gathered
        o[8] = len(set(obj.shape_class for obj in objs)) / 8.0
        occ = g != bg
        tot = float(occ.sum())
        if tot:                                               # assembled vs apart
            o[9] = float((occ[:, 1:] & occ[:, :-1]).sum()
                         + (occ[1:, :] & occ[:-1, :]).sum()) / tot
        return np.clip(o, 0.0, 1.0)

    # -- fitting ----------------------------------------------------------------
    def fit(self, trace: List[np.ndarray]) -> bool:
        """Learn from a trajectory that ENDED IN A LEVEL COMPLETION.

        A feature counts as progress only if it moved consistently in ONE direction:
        net displacement gives the direction, and at least MONOTONE_MIN of the steps
        that moved at all must agree with it. Features that wander are given weight 0
        rather than a small weight -- a noisy feature with a small vote is still a vote,
        and the whole point is that the score means something."""
        if self._retired or len(trace) < self.MIN_TRACE:
            return False
        A = np.vstack([np.asarray(t, dtype=np.float64).reshape(-1) for t in trace])
        if A.shape[1] != self.N_FEAT:
            return False
        # Backtest FIRST: grade the model as it stands on this trajectory, before this
        # trajectory is allowed to influence it. Grading a model on data it has already
        # absorbed measures memorisation, not progress.
        if self._fits:
            score = self.backtest(trace)
            self._backtests.append(score)
            if score < self.BACKTEST_MIN:
                self._retired = True          # O1 teeth: switched off by its own error
                return False
        vote = self._votes(A)
        if not np.any(vote):
            return False
        self._dir = (self._dir * self._fits + vote) / (self._fits + 1)
        self._fits += 1
        self._eff = None
        return True

    @classmethod
    def _votes(cls, A: np.ndarray) -> np.ndarray:
        """Signed per-feature agreement over one trajectory, or 0 where it abstains.

        Shared by the win channel and the death control so the two are measured by
        IDENTICAL rules -- a control judged by a laxer test than the thing it controls
        would subtract weights it never earned the right to."""
        vote = np.zeros(A.shape[1], dtype=np.float64)
        net = A[-1] - A[0]
        d = np.diff(A, axis=0)
        for j in range(A.shape[1]):
            moved = d[:, j][d[:, j] != 0.0]
            if net[j] == 0.0 or moved.size < cls.MIN_TRACE - 1:
                continue
            want = 1.0 if net[j] > 0 else -1.0
            agree = float((np.sign(moved) == want).mean())
            if agree >= cls.MONOTONE_MIN:
                vote[j] = want * agree
        return vote

    def fit_death(self, trace: List[np.ndarray]) -> bool:
        """A1: absorb a trajectory that ended in a GAME_OVER, as a CONTROL.

        Deliberately NOT symmetric with fit(). This channel can only ever SUBTRACT --
        it cancels win-features that deaths share -- so it never proposes a direction of
        its own and cannot make the model point somewhere wrong. The worst a completely
        bogus death channel can do is zero the whole direction, which returns O3 to the
        constant it was before any win, i.e. to a no-op. That asymmetry is what lets this
        ship on evidence from failures without a safety argument about failures.

        (The tempting stronger version -- treat -death as a progress direction on the 23
        games that never win -- was MEASURED and does not hold: across 19 distinct r11l
        deaths only 1 of 20 features shares a direction, and 0 of 10 object features do.
        Deaths agree about very little, so there is no avoidance direction to ride.)"""
        if self._retired or len(trace) < self.MIN_TRACE:
            return False
        try:
            A = np.vstack([np.asarray(t, dtype=np.float64).reshape(-1) for t in trace])
        except Exception:
            return False
        if A.shape[1] != self.N_FEAT:
            return False
        sig = A.round(6).tobytes()
        if sig in self._death_sigs:
            return False                      # a replayed path is not new evidence
        self._death_sigs.add(sig)
        vote = self._votes(A)
        self._dir_death = ((self._dir_death * self._death_fits + vote)
                           / (self._death_fits + 1))
        self._death_fits += 1
        self._eff = None
        return True

    def effective(self) -> np.ndarray:
        """The direction actually scored with: the win direction, minus every component
        the deaths show to be elapsed time.

        Computed lazily rather than folded into fit() because the evidence arrives in the
        wrong order -- on lp85 the win is episode 1 and all 11 deaths follow it, so a
        control applied at fit time would have had nothing to control with."""
        if self._eff is not None:
            return self._eff
        eff = self._dir
        if self._death_fits >= self.DEATH_MIN:
            confounded = (np.sign(self._dir) != 0) & \
                         (np.sign(self._dir) == np.sign(self._dir_death))
            if confounded.any():
                eff = self._dir.copy()
                eff[confounded] = 0.0
        self._eff = eff
        return eff

    # -- use --------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._fits > 0 and not self._retired

    def score(self, feat: Optional[np.ndarray]) -> float:
        if feat is None or not self.ready:
            return 0.0
        try:
            v = np.asarray(feat, dtype=np.float64).reshape(-1)
            if v.size != self.N_FEAT:
                return 0.0
            return float(np.dot(self.effective(), v))
        except Exception:
            return 0.0

    # Two boards whose scores differ by less than this are the same board as far as O3
    # is concerned. An absolute floor, not a tuned threshold: it exists only to stop
    # float noise on numerically identical boards from being dressed up as a ranking.
    MIN_SPREAD = 1e-6

    def buckets(self, feats: List[Optional[np.ndarray]]) -> List[int]:
        """Coarse progress ranks over a SET of candidate boards, scaled to the spread
        that is actually present among them.

        The absolute version (bucket() below) was measured to be a no-op: across 199
        lp85 level-2 boards the scores span 0.036 while one bucket step is 1/BUCKETS =
        0.125, so every state landed in the SAME bucket and `-prog` was a constant in
        the frontier's sort key. O3 was ordering nothing -- it had quietly degenerated
        back into the coverage objective it was written to replace. The fault was never
        the direction, it was measuring a small relative difference against a fixed
        absolute grid: a progress score has no natural unit, so only the ordering of the
        candidates in front of us carries meaning.

        Bucketing is KEPT, because its purpose survives: O2 must still break ties among
        states of comparable progress, or the planner can be talked into a long walk for
        a rounding difference. What changes is that the grid is now derived from the
        candidates instead of assumed.

        An undescribed board (None) ranks with the WORST known one -- it is not known to
        be good, and O4 is unaffected either way, since a low rank only delays a probe
        and never cancels it."""
        n = len(feats)
        if not self.ready or n == 0:
            return [0] * n
        scores = [self.score(f) if f is not None else None for f in feats]
        known = [s for s in scores if s is not None]
        if not known:
            return [0] * n
        lo, hi = min(known), max(known)
        if hi - lo < self.MIN_SPREAD:
            return [0] * n              # genuinely indistinguishable: say nothing
        return [0 if s is None else int(round((s - lo) / (hi - lo) * self.BUCKETS))
                for s in scores]

    def bucket(self, feat: Optional[np.ndarray]) -> int:
        """Coarse rank, so that O2 (cost) still decides among states of EQUAL progress
        -- which is what the objectives table says O2 is for. A raw score would let the
        planner be talked into a 20-action walk for a rounding difference."""
        if not self.ready:
            return 0
        return int(round(self.score(feat) * self.BUCKETS))

    def backtest(self, trace: List[np.ndarray]) -> float:
        """Fraction of steps along a KNOWN-GOOD trajectory on which the score rose.
        A progress model that does not rise along a path to a win is simply wrong."""
        if not self.ready or len(trace) < 2:
            return 1.0
        s = [self.score(t) for t in trace]
        d = [b - a for a, b in zip(s, s[1:])]
        moved = [x for x in d if x != 0.0]
        if not moved:
            return 0.0                        # says nothing anywhere = no signal at all
        return float(sum(1 for x in moved if x > 0) / len(moved))


# ==========================================
# COMPONENT 5.6: CLICK PLANNER (ACTION6 games)
# ==========================================
#
# 10 of the 25 real games are click-only (available_actions=[6]) -- boards of
# tiles where clicking mutates state (e.g. ft09 cycles neighbor colors,
# lights-out style). ReactivePlanner cannot act there, and the old fallback
# emitted a bare "ACTION6" that resolved to clicking the grid CENTER forever.
# This planner clicks like an experimenter instead:
#   1) segment the frame into objects -> candidate click targets,
#   2) click every candidate at least once (discovery),
#   3) learn per-cell effect (frame changed? reward?), then keep clicking
#      ACTIVE cells with the LOWEST click count -- uniform coverage of the
#      reactive surface beats hammering one tile, and for cyclic mechanics
#      revisiting is required, so cells are never blacklisted,
#   4) a small epsilon of uniform-random clicks covers non-object targets.

class ClickPlanner:
    """Policy for click-only games (available_actions == [6]).

    There is no single click mechanic across the real games -- probing shows at
    least three families: a LOCALIZED reactive region with lose-on-bad-click
    (ft09), a handful of SPARSE inert-elsewhere buttons (lp85: 2 of 4096 cells),
    and near-universal reactivity / cursor games (r11l, vc33). Frame-change is a
    game-specific, noisy signal (vc33 changes on every click); the only reliable
    reward is level completion, which is SPARSE. So the planner is built around
    coverage + credit assignment that PERSISTS across the many GAME_OVER restarts
    a click game induces, and around learning which cells are lethal.
    """
    MAX_TARGETS = 48   # candidate objects per decision (guards vs noisy frames)
    QUANT = 4          # px; cells sharing a quantized key are "the same target"

    # -- click world-model / state-graph search (the click analogue of
    #    ReactivePlanner). For small-action-space deterministic puzzles (e.g. lp85:
    #    2 buttons slide blocks toward goals) the env IS a replayable simulator: a
    #    live RESET restores the level's exact initial state AND refills the budget
    #    (verified: level_reset() leaves action_count>0, so back-to-back RESETs never
    #    full-reset the score). So we do a breadth-first search over BOARD STATES:
    #    each state = a frame hash, each edge = a button click discovered by replaying
    #    "RESET + path-to-parent + button" and hashing the result. Deduping by state
    #    (unlike a blind product-of-buttons enumeration) collapses no-op re-clicks and
    #    the wall-parked fixed points lp85 slides into, and finds the SHORTEST winning
    #    path. Winning is detected implicitly -- a winning click bumps the score, the
    #    outer loop calls new_level(), search restarts fresh on the next layout.
    #
    # -- THE OBJECTIVE FUNCTIONS THE MODEL IS OPTIMISED AGAINST -----------------
    #
    # A world model with no objective is just a predictor, and a predictor with no
    # objective optimises nothing -- it will happily be accurate about things that
    # do not matter and confident about things it has never been graded on. The
    # four objectives below are the whole specification of what `_G` is FOR. Each
    # one is measurable and each one is wired to a specific decision; none of them
    # is a free-floating "accuracy" number.
    #
    #  O1 FIDELITY -- is the model TRUE?
    #     `_G` holds only OBSERVED transitions, so it is exact by construction.
    #     The danger is not estimation error, it is STALENESS: a recorded edge is
    #     only valid if the game really is deterministic and really has no hidden
    #     state (tu93 proved some games do). So every navigation plan carries the
    #     state it EXPECTS to be in after each step (`_expect`), and a mismatch is
    #     treated as falsification: the offending edge is deleted, the plan is
    #     abandoned, and the planner re-plans. The model is never "probably right".
    #
    #  O2 COST -- actions are the CURRENCY, and RHAE squares the ratio.
    #     Every candidate probe is priced in SCORED ACTIONS and the cheapest wins.
    #     Planning through `_G` is free; only emission is paid for. This is the
    #     objective the old code was blind to: it priced every probe as a fresh
    #     replay from the root, so discovering one edge cost depth+2 actions even
    #     when the parent state was the board already on screen (cost 1).
    #
    #  O3 PROGRESS -- what is a GOOD state?
    #     Two things, and only the second is really an answer.
    #     (a) INFORMATION: a (state, untried button) pair is worth one unit, a pair
    #         whose outcome is recorded is worth zero and is never re-spent. This
    #         keeps the search from re-buying what it already knows.
    #     (b) PROGRESS PROPER (ProgressModel, COMPONENT 5.55): how close the board is
    #         to winning, learned from the trajectory of a level the agent actually
    #         completed. (a) alone is a COVERAGE objective -- every untried pair
    #         scores the same, so lp85's 251 explored level-2 boards were all equally
    #         attractive and the search had no opinion about any of them. That is the
    #         whole reason a faithful, cheap, safe model still scored ~0.
    #     The frontier is ordered by (b) first and priced by O2 second. Before the
    #     first level completion (b) returns a constant and the ordering is exactly
    #     the old cost-first one, so O3 is a strict no-op until it has evidence.
    #
    #  O4 SAFETY -- a hard CONSTRAINT, never a weighted term.
    #     A plan is rejected outright if it traverses a known-lethal edge
    #     (`_dead_edges`) or would emit a RESET while the engine's action counter
    #     reads 0 (a full_reset: every level already won is destroyed). These are
    #     not traded off against O1-O3 at any exchange rate.
    #
    # Composite: maximise  E[information] / scored_actions,  subject to (O1 fresh
    # AND O4 safe). Everything below is that sentence in code.
    MIN_BUTTONS_FOR_SEARCH = 2 # a real button puzzle has >=2 controls; a lone
                               # reactive cell (e.g. lp85 detects 1) is not a
                               # searchable action space -> stay on coverage, which
                               # is the validated floor. Prevents the search from
                               # displacing a coverage win it cannot improve on.
    MAX_BUTTONS = 6            # only search when the reactive set is this small
    MIN_DISCOVER_CLICKS = 60   # coverage clicks before trusting the button set
    MAX_REPLAY = 30            # deepest button-path to replay from RESET (< the ~52
                               # button-click/level budget; only button clicks cost)
    MAX_SEARCH_STATES = 250    # hard cap on distinct boards before bailing to coverage.
                               # A true button puzzle's early levels have few distinct
                               # boards; a blow-up past this means the frame hash is NOT
                               # a stable state signature (some games render a live
                               # step-counter INTO the 64x64 frame, so every board hashes
                               # unique and dedup fails). Bailing keeps the coverage floor.

    def __init__(self):
        # Non-RESET actions since the last RESET, mirroring the engine's
        # `_action_count` -- see note_action(). Starts at 0: before the game's
        # first action the engine's counter is 0 too, and the opening RESET is
        # SUPPOSED to be a full reset (there is no score to lose yet).
        self._acts_since_reset = 0
        # quantized cell -> [clicks, frame-changes, rewards, losses]
        self._stats: Dict[Tuple[int, int], List[int]] = {}
        # quantized cell -> the EXACT (r,c) that actually changed the frame there.
        # Buttons are often reactive only at a specific pixel, not the cell centre,
        # so the search MUST replay the exact coord (using the centre clicks a dead
        # spot and the whole sequence no-ops).
        self._reactive: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._buttons: List[Tuple[int, int]] = []     # exact coords, level-persistent
        # EXACT pixels clicked. _stats is quantized to QUANT-sized cells, which is
        # the right grain for credit assignment but hides the fact that a cell has
        # only ever been probed at its centre -- fresh_cell needs the finer memory
        # to refine coverage instead of repeating it.
        self._exact: set = set()
        # O3. Lives on the PLANNER, not on the search: what progress looks like is a
        # property of the GAME, so it must survive new_level() -- level 1's trajectory
        # is the only evidence level 2 will ever get. `_trace` is the current level's
        # feature history; `_feat` (per-search, cleared in _reset_search) is the
        # descriptor of each state in the graph so the frontier can be ranked.
        self._pm = ProgressModel()
        self._trace: List[np.ndarray] = []
        self._o3_off = bool(os.environ.get("ARC_NO_O3"))
        self._o3_prior = bool(os.environ.get("ARC_O3_PRIOR"))
        self._death_off = bool(os.environ.get("ARC_NO_DEATH"))   # A1 kill switch
        # DIAGNOSTIC ONLY (ARC_O3_DUMP=<dir>): persist the RAW BOARDS of every episode
        # together with how it ended, so the question "is O3 starved of data or fitted on
        # the wrong basis?" can be answered offline, at zero cost in scored actions.
        # Raw grids and not features, because deciding the basis (C1) requires recomputing
        # descriptors that do not exist yet. Off by default and touched nowhere else.
        self._o3_dump = os.environ.get("ARC_O3_DUMP") or ""
        self._dump_grids: List[np.ndarray] = []
        self._dump_state_grid: Dict[int, np.ndarray] = {}
        self._dump_n = 0
        self._reset_search(keep_buttons=False)

    def note_action(self, action_str: str) -> None:
        """Mirror the engine's action counter, which decides what a RESET MEANS.

        arcengine/base_game.py: `_action_count` increments on every NON-reset
        action (L279) and is zeroed by `set_level` (L160) -- which `level_reset`
        itself calls. `handle_reset` (L313) then reads
        `_action_count == 0 -> full_reset()`, and `full_reset` sets `_score = 0`.
        So TWO RESETS WITH NO ACTION BETWEEN THEM SILENTLY DESTROY EVERY LEVEL
        ALREADY WON. Measured on lp85: the agent won level 1 six times and wiped
        it six times, finishing at 1/8.

        The planner cannot read that counter -- on Kaggle the engine is remote --
        so it tracks the same quantity from its own action stream. EVERY action
        must land here, not just this planner's: an escalation click between two
        search RESETs moves the engine's counter, and a search that missed it
        would skip a RESET it genuinely needed and then record an edge under the
        wrong parent state."""
        if action_str == "RESET":
            self._acts_since_reset = 0
        elif action_str:
            self._acts_since_reset += 1

    def _reset_search(self, keep_buttons: bool):
        if not keep_buttons:
            self._buttons = []
            self._reactive = {}
        # Always start a level in COVERAGE (discover), NEVER in search -- even when
        # buttons are already known. Entering search emits a RESET as the level's
        # first action, and the engine treats a RESET while its internal
        # action_count==0 (true at the start of every level, since advancing a level
        # zeroes it) as a FULL reset: that wipes levels_completed back to 0, silently
        # throwing away every level already won. This was the lp85 0/8 regression --
        # coverage won level 1, new_level() re-entered search, its first RESET
        # full-reset the score. Coverage-first guarantees >=MIN_DISCOVER_CLICKS real
        # clicks before any search RESET, so RESET does a level_reset (keeps score).
        self._phase = "discover"
        self._total_clicks = 0
        self._search_done = False
        # Button sets whose graph was explored to exhaustion WITHOUT solving the
        # level -- a PROOF that the level is unreachable through that set. Cleared
        # here because a new level is a new layout: a set refuted against the old
        # board says nothing about the new one.
        self._exhausted_sets: List[frozenset] = []
        # Button sets that search STOPPED on for a resource reason (state cap,
        # everything out of replay range). These are NOT refutations and must never
        # be filed as such -- see `_stop_reason`.
        self._suspended_sets: List[frozenset] = []
        # Why the last _build_next_plan() gave up. "" while it is succeeding.
        self._stop_reason = ""
        # O3 descriptor per graph state (hash -> feature vector). Per-search: the
        # hashes are meaningless once the layout changes.
        self._feat: Dict[int, np.ndarray] = {}
        # -- state-graph BFS bookkeeping (populated on _enter_search) --
        self._G: Dict[int, Dict[int, int]] = {}          # state -> {button_idx: child_state}
        self._dist: Dict[int, Tuple[int, ...]] = {}       # state -> shortest button-path from root
        self._queue: "deque[int]" = deque()               # BFS frontier of states
        self._root: Optional[int] = None
        # True only while the root is the board a RESET produces. Teleport plans are
        # legal exactly then; a re-entered search is rooted where we stand instead.
        self._root_is_reset = True
        self._dead_edges: set = set()                     # (state, button_idx) that lose / over-deep
        self._plan: List[Any] = []                        # current replay: ["RESET", int, int, ...]
        self._plan_i = 0
        self._pending: Optional[Tuple] = None             # ("root",) | ("expand", parent, button_idx)
        # -- world-model navigation (O1/O2) --
        self._cur: Optional[int] = None                   # hash of the board on screen NOW
        self._expect: Optional[List[int]] = None          # states a nav plan predicts, per step
        self._nav_plans = 0                               # counters: evidence, not decoration
        self._reset_plans = 0
        self._saved = 0                                   # scored actions navigation avoided
        self._falsified = 0                               # edges deleted by an O1 mismatch

    def _graph_trace(self) -> List[np.ndarray]:
        """The PURPOSEFUL path to the board the level was won from, read out of `_G`.

        The wall-clock trace (`_trace`) is every board the planner stood on, and on lp85
        that is 95 frames of which ~89 are coverage clicks -- a random walk that happened
        to end well. Fitting O3 on it earned a direction for 2 features out of 20,
        because a random walk has no monotone structure to find and MONOTONE_MIN
        correctly refuses to invent one.

        The graph knows better: `_dist[_cur]` is the shortest BUTTON PATH from the root to
        the state we were standing on when the level completed. Walking it gives a
        trajectory in which every step was a deliberate, effective action -- shorter, but
        every frame is signal. Returns [] if the path cannot be reconstructed EXACTLY: an
        approximate trajectory is worse than none, because O3 would then learn from a
        fiction with full confidence."""
        path = self._dist.get(self._cur) if self._cur is not None else None
        if self._root is None or path is None:
            return []
        out, st = [], self._root
        for b in (None, *path):
            if b is not None:
                nxt = self._G.get(st, {}).get(b)
                if nxt is None or nxt == -1:
                    return []
                st = nxt
            f = self._feat.get(st)
            if f is None:
                return []
            out.append(f)
        return out

    def _dump_episode(self, outcome: str) -> None:
        """DIAGNOSTIC (ARC_O3_DUMP). Persist the episode that just ended, labelled by how
        it ended. Both outcomes are written -- the whole point of the measurement is to
        compare what rises on the way to a WIN against what rises on the way to a DEATH,
        and a feature that rises on both is not progress, it is elapsed time."""
        if not self._o3_dump or not self._dump_grids:
            return
        try:
            os.makedirs(self._o3_dump, exist_ok=True)
            # The purposeful path (deliberate steps only), when the graph can produce one
            # EXACTLY -- same reconstruction rule as _graph_trace, so the diagnostic sees
            # precisely what fit() would have seen.
            gpath: List[np.ndarray] = []
            path = self._dist.get(self._cur) if self._cur is not None else None
            if self._root is not None and path is not None:
                st, ok = self._root, True
                for b in (None, *path):
                    if b is not None:
                        nxt = self._G.get(st, {}).get(b)
                        if nxt is None or nxt == -1:
                            ok = False
                            break
                        st = nxt
                    g = self._dump_state_grid.get(st)
                    if g is None:
                        ok = False
                        break
                    gpath.append(g)
                if not ok:
                    gpath = []
            tag = os.environ.get("ARC_O3_DUMP_TAG", "run")
            self._dump_n += 1
            np.savez_compressed(
                os.path.join(self._o3_dump, f"{tag}_{self._dump_n:03d}_{outcome}.npz"),
                walk=np.asarray(self._dump_grids, dtype=np.uint8),
                graph=(np.asarray(gpath, dtype=np.uint8) if gpath
                       else np.zeros((0, 1, 1), dtype=np.uint8)),
                outcome=np.array(outcome),
            )
        except Exception:
            pass
        self._dump_grids = []

    def new_level(self):
        # O3 GROUNDING. This is the ONLY moment the agent is ever told "that was
        # progress". The trajectory just finished ended in a level completion, so it is
        # ground truth about what progress looks like in THIS game -- and the next level
        # is where it gets spent. Fit before the wipe below: `_reset_search` is about to
        # discard the search, and `_trace` belongs to the level that just ended.
        if not self._o3_off:
            # Prefer the graph path (every step deliberate) and fall back to the
            # wall-clock trace only when the graph cannot produce one -- which is the
            # case whenever the level was won during coverage, before any search began.
            gt = self._graph_trace()
            _src = "graph" if len(gt) >= self._pm.MIN_TRACE else "walk"
            _fitted = self._pm.fit(gt if _src == "graph" else self._trace)
            if self._DBG:
                import sys
                print(f"[CP] O3 fit={_fitted} src={_src} graph={len(gt)} "
                      f"walk={len(self._trace)} fits={self._pm._fits} "
                      f"ready={self._pm.ready} "
                      f"features_with_a_direction={int(np.count_nonzero(self._pm._dir))} "
                      f"deaths={self._pm._death_fits} "
                      f"after_control={int(np.count_nonzero(self._pm.effective()))} "
                      f"backtests={[round(b, 3) for b in self._pm._backtests]}",
                      file=sys.stderr)
        self._dump_episode("win")
        self._trace = []
        # Real level-up: the board layout changes, so per-cell stats are stale -> wipe.
        # But button POSITIONS are UI arrows, level-invariant in practice: KEEP them so
        # L2+ skip re-discovery and search immediately (if wrong, search exhausts and
        # coverage re-discovers from the fresh stats).
        self._stats = {}
        self._exact = set()        # new layout -> every pixel is new information again
        # Advancing a level calls base_game.set_level, which zeroes `_action_count`
        # with no RESET emitted -- the mirror only sees actions, so it is told here.
        self._acts_since_reset = 0
        self._reset_search(keep_buttons=True)

    def reset(self):
        # A1. This trajectory ended in a GAME_OVER. It is still a LABEL -- and it is the
        # plentiful one: 46 deaths per tier-1 sweep against 1 win. It is fed to the death
        # CONTROL, not to the progress channel, so it can only cancel win-features that
        # deaths share (see ProgressModel.fit_death). The old code deleted it outright,
        # which was right about the sign and wrong about the information: with one win,
        # "this feature rose because we won" and "this feature rose because the board
        # advanced" cannot be told apart without a control, and measurement said 2 of the
        # 3 features O3 had learned on lp85 were the latter.
        if not self._o3_off and not self._death_off:
            gt = self._graph_trace()
            self._pm.fit_death(gt if len(gt) >= self._pm.MIN_TRACE else self._trace)
        self._dump_episode("death")
        self._trace = []
        # Intra-level restart (after GAME_OVER). The layout is IDENTICAL on restart,
        # so which cells are reactive / safe / lethal is still valid -- KEEP it.
        # (The old code aliased reset->new_level, wiping this every loss, which made
        # the planner permanently amnesiac on lose-happy games like ft09.)
        if self._phase == "search" and self._pending is not None and self._pending[0] == "expand":
            # A GAME_OVER interrupted the replay of "RESET + path + b". The path prefix
            # was already validated as non-lethal & deterministic, so the death is on
            # the final edge being tested: mark (parent, b) dead so search never retries
            # it. The framework has already emitted its own RESET, so drop the aborted
            # plan; the next act() builds the next plan from the frontier.
            _, parent, b = self._pending
            self._G.setdefault(parent, {})[b] = -1        # sentinel: explored, leads nowhere
            self._dead_edges.add((parent, b))
        if self._phase == "search":
            self._pending = None
            self._plan = []
            self._plan_i = 0
            # The restart moved the board, so a navigation plan's predictions and
            # the cached "where we are" are both stale (O1: never plan from a state
            # we have not observed).
            self._expect = None
            self._cur = None

    @property
    def searching(self) -> bool:
        return bool(self._buttons) and self._phase == "search" and not self._search_done

    def _key(self, r: int, c: int) -> Tuple[int, int]:
        return (int(r) // self.QUANT, int(c) // self.QUANT)

    BUTTON_CLUSTER_DIST = 10   # px; reactive coords within this are ONE button sprite

    def _detect_buttons(self) -> List[Tuple[int, int]]:
        """Discrete actions of a button-driven puzzle: ONE representative exact coord
        per physical button sprite. A single button occupies several QUANT cells
        (lp85's LEFT slide is a 4-cell sprite at cols 2-6, rows 30-34), so the raw
        `self._reactive` coords over-count buttons ~3x. That inflation is fatal to the
        search: branching factor n caps the reachable depth via n**depth <=
        SEQ_FANOUT_CAP (6 fake buttons -> depth 3 only; 2 real buttons -> depth 8), so
        a 5-slide puzzle is unsearchable until the duplicates are merged. Cluster the
        reactive coords by spatial proximity (Chebyshev < BUTTON_CLUSTER_DIST) and
        return one coord per cluster. Replay still uses the EXACT pixel (cell centres
        are frequently dead spots)."""
        coords = sorted(self._reactive.values())
        clusters: List[List[Tuple[int, int]]] = []
        for rc in coords:
            for cl in clusters:
                if any(max(abs(rc[0] - m[0]), abs(rc[1] - m[1])) < self.BUTTON_CLUSTER_DIST
                       for m in cl):
                    cl.append(rc)
                    break
            else:
                clusters.append([rc])
        # representative = the cluster member closest to the cluster centroid (a
        # robust interior pixel, least likely to sit on a sprite edge / dead border).
        reps = []
        for cl in clusters:
            cr = sum(m[0] for m in cl) / len(cl)
            cc = sum(m[1] for m in cl) / len(cl)
            reps.append(min(cl, key=lambda m: (m[0] - cr) ** 2 + (m[1] - cc) ** 2))
        return sorted(reps)

    def _abstraction_is_live(self, buttons: List[Tuple[int, int]]) -> bool:
        """Is this button set still worth spending the budget on?

        Two kinds of "no", and they propagate differently -- which is the whole point.

        REFUTED (`_exhausted_sets`, a PROOF). Exhausting the graph over set B shows the
        level is not reachable through B. Any SUBSET of B is therefore also refuted: it
        can only reach boards B could already reach, so re-searching it is guaranteed
        waste. Proofs propagate downward.

        SUSPENDED (`_suspended_sets`, a BUDGET STOP). Search over B hit the state cap or
        ran out of replay range. That says nothing whatever about the game -- and so it
        says nothing about B's subsets either. A subset spans a SMALLER space and may
        well exhaust honestly where B could not. Only the identical set is blocked, and
        only because re-running the identical search with the identical strategy would
        reproduce the identical stop.

        Either way a set containing a button no blocked set had is a genuinely different
        model of the action space, untested, and worth the budget.

        "A button no blocked set had" is judged with the SAME equivalence `_detect_buttons`
        uses to merge reactive coords into one physical button (Chebyshev <
        BUTTON_CLUSTER_DIST). Comparing by exact pixel instead lets detector jitter
        manufacture new abstractions out of the same buttons: measured on sb26 (2026-08-02),
        eight search entries whose sets sat 2-7 px apart -- inside the clustering radius,
        so the detector itself calls them one button -- each passing as a fresh model of the
        action space. A planner that merges (58,22) and (58,18) into one button when
        detecting them must not treat them as two when deciding what it has already tried."""
        bs = [tuple(b) for b in buttons]
        if any(self._covers(refuted, bs) for refuted in self._exhausted_sets):
            return False
        # Suspensions bind only the identical set, so coverage must run BOTH ways.
        return not any(self._covers(susp, bs) and self._covers(bs, susp)
                       for susp in self._suspended_sets)

    @classmethod
    def _covers(cls, outer, inner) -> bool:
        """Does every button in `inner` coincide with one in `outer`? (Chebyshev
        proximity, the detector's own notion of button identity.)"""
        return all(any(max(abs(i[0] - o[0]), abs(i[1] - o[1])) < cls.BUTTON_CLUSTER_DIST
                       for o in outer)
                   for i in inner)

    @staticmethod
    def _hash(grid: np.ndarray) -> int:
        return hash((grid.shape, grid.tobytes()))

    _DBG = bool(os.environ.get("CP_DEBUG"))

    def _enter_search(self, buttons: List[Tuple[int, int]], from_reset: bool = True):
        """Switch from coverage to state-graph BFS.

        `from_reset=True` (the level's FIRST entry): the opening plan is a lone RESET
        that establishes the root as the level's initial board, which is what makes
        the teleport route `["RESET", *path_from_root, b]` meaningful.

        `from_reset=False` (RE-entry after a button set was refuted): emit NOTHING.
        The root is adopted from the board already on screen on the next step. This is
        not a convenience -- it is O4. A RESET is a full_reset whenever the engine's
        action counter reads 0, which destroys every level already won, and mid-level
        the agent cannot know that counter reliably (measured: a re-entry RESET wiped
        a banked lp85 level even with the mirror reading 300 actions since the last
        RESET). The only safe number of RESETs to gamble on is zero. Teleport is
        disabled for such an episode because RESET does not return to this root."""
        self._buttons = buttons
        self._phase = "search"
        self._search_done = False
        self._G = {}
        self._dist = {}
        self._queue = deque()
        self._root = None
        self._root_is_reset = bool(from_reset)
        self._stop_reason = ""          # per-episode: never read a previous episode's
        self._dead_edges = set()
        self._pending = ("root",) if from_reset else None
        self._plan = ["RESET"] if from_reset else []
        self._plan_i = 0
        self._search_actions = 0
        self._cur = None
        self._expect = None
        self._nav_plans = self._reset_plans = self._saved = self._falsified = 0
        if self._DBG:
            import sys
            print(f"[CP] enter_search buttons={buttons} after {self._total_clicks} "
                  f"clicks from_reset={from_reset}", file=sys.stderr)

    def _adopt_root(self, h: int) -> None:
        """Make an OBSERVED board a frontier node of the graph. Used to root a
        re-entered search where we stand, and to re-attach after navigation lands
        somewhere disconnected -- an unreachable frontier is not exhaustion."""
        # Only legal while teleport is off. With teleport live, `_dist` IS the
        # replay path from the post-RESET board, and giving a disconnected state an
        # empty path would stage `["RESET", b]` for a board RESET does not produce.
        # (That episode never needs this anyway: RESET reaches the whole graph.)
        if h is None or h in self._dist or self._root_is_reset:
            return
        if self._root is None:
            self._root = h
        self._dist[h] = ()
        self._G.setdefault(h, {})
        self._queue.append(h)

    def _process_result(self, h: int):
        """A replay plan finished; `h` is the hash of the board it produced. Record
        the root, or the edge (parent -> h) that the tested button just revealed."""
        kind = self._pending[0]
        if kind == "root":
            self._root = h
            self._dist[h] = ()
            self._queue.append(h)
        elif kind == "expand":
            _, parent, b = self._pending
            self._G.setdefault(parent, {})[b] = h
            if h not in self._dist:                        # a NEW board -> enqueue to expand
                self._dist[h] = self._dist[parent] + (b,)
                self._queue.append(h)
            # else: no-op / merges into a known state -> dedup drops it (this is the
            # win over blind product-BFS: parked/looping states are visited once).
        self._pending = None

    def _untried(self, s: int) -> Optional[int]:
        """The lowest-indexed button never tried from state `s` (O3: one unit of
        information), or None when `s` is fully expanded. O4 filters lethal edges."""
        tried = self._G.get(s, {})
        for cand in range(len(self._buttons)):
            if cand not in tried and (s, cand) not in self._dead_edges:
                return cand
        return None

    def _nav_dists(self) -> Dict[int, Tuple[int, ...]]:
        """IMAGINATION (O2): BFS over the ALREADY-OBSERVED edges of `_G`, outward
        from the board currently on screen, returning state -> button path.

        This is the world model being *used* rather than merely collected. It reads
        only recorded transitions, so it costs ZERO scored actions and cannot be
        wrong about anything it has not seen -- an unreachable state simply does not
        appear, and the caller falls back to the RESET replay. Sentinel (-1) and
        lethal children are excluded (O4)."""
        src = self._cur
        if src is None:
            return {}
        out: Dict[int, Tuple[int, ...]] = {src: ()}
        q: "deque[int]" = deque([src])
        while q:
            s = q.popleft()
            for b, child in self._G.get(s, {}).items():
                if child == -1 or child in out or (s, b) in self._dead_edges:
                    continue
                out[child] = out[s] + (b,)
                q.append(child)
        return out

    def _build_next_plan(self) -> bool:
        """Pick the frontier probe with the LOWEST SCORED-ACTION COST and stage it.

        This is objective O2 doing the work. Every (state, untried-button) pair on
        the frontier is worth the same one unit of information (O3), so the only
        thing left to optimise is what it COSTS, and there are two routes to the
        same probe:

          (a) TELEPORT  ["RESET", *shortest_path_from_root, button]
              -- what this planner used to do unconditionally. Price:
              len(path) + 1, plus 1 more for the RESET unless we are already at
              the level's initial state.
          (b) NAVIGATE  [*path_from_HERE, button]
              -- walk the known edges of `_G` from the board on screen. Price:
              len(path) + 1, and NO RESET at all.

        (b) is usually far cheaper and is frequently 1 action, because after
        expanding a state the board on screen IS a frontier node. Paying depth+2 to
        re-derive a board already in front of us is exactly the "spend scored
        actions to gather information" pattern RHAE squares.

        Returns False when no probe can be staged, and sets `_stop_reason` to WHY.
        The distinction is not bookkeeping -- it decides whether the caller may file
        the button set as REFUTED (a claim about the game) or merely SUSPENDED (a
        claim about our budget). See the three sites below.
        """
        if len(self._dist) > self.MAX_SEARCH_STATES:
            # BUDGET, NOT PROOF. We ran out of room to remember states. The unexplored
            # part of the space is unexplored, not shown to be goalless.
            self._stop_reason = "cap"
            return False
        # The board in front of us is always a legal place to search FROM. Adopting
        # it costs nothing and is what lets a RESET-free episode start, and recover
        # if navigation ever lands off the known component.
        self._adopt_root(self._cur)
        # Retire fully-expanded nodes from the head so the frontier stays small.
        while self._queue and self._untried(self._queue[0]) is None:
            self._queue.popleft()
        if not self._queue:
            # PROOF. Every state we could reach through these buttons has had every
            # button tried, and none of them won the level. This is the only one of
            # the three exits that says something about the GAME.
            self._stop_reason = "exhausted"
            return False

        nav = self._nav_dists()
        # A RESET is free when the engine's counter already reads 0 -- but only
        # because it is then a no-op in intent AND skipped by _search_step. It is
        # never *emitted* in that state (O4: that would be a full_reset).
        reset_cost = 0 if self._acts_since_reset == 0 else 1
        # (-progress, cost, queue_order, state, button, nav_path). O3 leads and O2
        # breaks ties, which is what the objectives table has always said: O2 decides
        # "which plan among EQUALS". Keying on cost first -- the old order -- meant the
        # search was aimed at nothing and merely aimed cheaply. Progress is BUCKETED, so
        # cost still decides among states of comparable progress and the planner cannot
        # be talked into a long walk for a rounding difference.
        best = None
        # Ranks are computed over the WHOLE frontier at once, because a progress score
        # only means something relative to the alternatives on offer (see
        # ProgressModel.buckets -- the per-state absolute version ranked every lp85
        # level-2 board identically and made this term a constant).
        if self._o3_off:
            prog_of: Dict[int, int] = {}
        else:
            _qs = list(self._queue)
            prog_of = dict(zip(_qs, self._pm.buckets([self._feat.get(s) for s in _qs])))
        for order, s in enumerate(self._queue):
            b = self._untried(s)
            if b is None:
                continue
            prog = prog_of.get(s, 0)
            # Teleport exists only if the root really is the post-RESET board. On a
            # re-entered search it is wherever we happened to be standing, so RESET
            # would land somewhere else entirely -- and emitting one at all is the
            # wipe hazard this episode was created to avoid.
            root_path = self._dist.get(s) if self._root_is_reset else None
            if root_path is not None and len(root_path) + 1 <= self.MAX_REPLAY:
                cand = (-prog, len(root_path) + 1 + reset_cost, order, s, b, None)
                if best is None or cand[:3] < best[:3]:
                    best = cand
            here = nav.get(s)
            if here is not None and len(here) + 1 <= self.MAX_REPLAY:
                cand = (-prog, len(here) + 1, order, s, b, here)
                if best is None or cand[:3] < best[:3]:
                    best = cand
        if best is None:
            # BUDGET, NOT PROOF. The frontier is non-empty -- there are states with
            # untried buttons -- but every one of them is further than MAX_REPLAY
            # from both the root and the board on screen. We cannot AFFORD the probe;
            # we have not shown the probe is pointless.
            self._stop_reason = "unreachable"
            return False

        self._stop_reason = ""
        _negprog, cost, _order, s, b, path = best
        if path is None:
            self._plan = ["RESET", *self._dist[s], b]
            self._expect = None
            self._reset_plans += 1
        else:
            self._plan = [*path, b]
            # O1: what the model PREDICTS the board will be after each navigation
            # step. Recorded now so _search_step can falsify it against reality.
            exp, st = [], self._cur
            for btn in path:
                st = self._G[st][btn]
                exp.append(st)
            self._expect = exp
            self._nav_plans += 1
            teleport = len(self._dist.get(s, ())) + 1 + reset_cost
            self._saved += max(0, teleport - cost)
        self._plan_i = 0
        self._pending = ("expand", s, b)
        return True

    def _search_step(self, grid: np.ndarray) -> Optional[str]:
        """Emit one action of the current replay plan. `grid` is the board resulting
        from the previous action; when a plan's last click has landed we hash it to
        learn the edge, then stage the next plan. Returns None when search is spent."""
        # The board on screen, hashed. This is the world model's ANCHOR: everything
        # below plans relative to where we actually are, not where a replay from the
        # root would put us.
        self._cur = self._hash(grid)
        # O3: remember what this board LOOKS like, so the frontier can be ranked by
        # progress later. Keyed by the same hash the graph uses, so a state and its
        # descriptor can never drift apart.
        if not self._o3_off and self._cur not in self._feat:
            self._feat[self._cur] = ProgressModel.features(grid)
        if self._o3_dump and self._cur not in self._dump_state_grid:
            try:
                self._dump_state_grid[self._cur] = np.asarray(grid, dtype=np.uint8).copy()
            except Exception:
                pass

        # O1 FALSIFICATION. A navigation plan predicted this board. If reality
        # disagrees, the edge we walked is not what `_G` recorded -- the game has
        # hidden state, or the frame carries a counter, or a click was swallowed.
        # Delete the offending edge and re-plan rather than record the pending probe
        # under a parent we are demonstrably not standing on (silent graph
        # corruption is far worse than a wasted action).
        if self._expect is not None and 0 < self._plan_i <= len(self._expect):
            step_i = self._plan_i - 1
            if self._cur != self._expect[step_i]:
                # The edge just walked was (parent -> predicted). We still know the
                # parent for every step but the first (whose parent was the board we
                # planned from, whose hash we no longer hold -- there we just drop
                # the plan and re-plan from the board that is really on screen).
                if step_i > 0:
                    parent = self._expect[step_i - 1]
                    bad = self._plan[step_i]
                    if isinstance(bad, int):
                        self._G.get(parent, {}).pop(bad, None)
                self._falsified += 1
                if self._DBG:
                    import sys
                    print(f"[CP] O1 falsified at nav step {step_i} "
                          f"(total {self._falsified})", file=sys.stderr)
                self._plan, self._plan_i, self._expect = [], 0, None
                self._pending = None

        # A finished plan's result has arrived -> learn from it. (`_cur` is that
        # board's hash, already computed above -- re-hashing it every step is pure
        # cost on a 64x64 frame.)
        if self._pending is not None and self._plan_i >= len(self._plan):
            self._process_result(self._cur)
        # Need a fresh plan?
        if self._plan_i >= len(self._plan):
            if not self._build_next_plan():
                # O1 AT THE ABSTRACTION LEVEL. A fully explored graph over button set
                # B is not "planning is finished" -- it is a PROOF that the level is
                # unreachable through B, i.e. B itself is falsified as sufficient. The
                # honest response is to widen B, not to stop modelling. Record B as
                # refuted and hand back to coverage, which keeps discovering reactive
                # cells; `act()` re-enters search the moment coverage finds a button
                # none of the refuted sets contained. Re-entry is bounded because each
                # one needs a strictly new button and buttons are capped at MAX_BUTTONS.
                #
                # ...BUT ONLY WHEN THE GRAPH REALLY WAS EXHAUSTED. `_build_next_plan`
                # also gives up when it hits the state cap or when every frontier node
                # is out of replay range. Those are statements about OUR BUDGET, not
                # about the game, and filing them as refutations is a falsification
                # claim with no teeth -- the exact failure mode O1 exists to prevent.
                # Measured on lp85: one episode reached 251 states (cap 250) and the
                # cap trip banned a 3-button set AND, by the subset rule, the 2-button
                # set that was working. A budget stop is recorded as SUSPENDED, which
                # bans re-running the identical search and nothing else.
                if self._stop_reason == "exhausted":
                    self._exhausted_sets.append(frozenset(self._buttons))
                else:
                    self._suspended_sets.append(frozenset(self._buttons))
                self._search_done = True               # this set is done; coverage next
                self._phase = "discover"
                self._expect = None
                if self._DBG:
                    import sys
                    print(f"[CP] search STOP({self._stop_reason or 'exhausted'}) "
                          f"states={len(self._dist)} "
                          f"edges={sum(len(v) for v in self._G.values())} "
                          f"dead={len(self._dead_edges)} actions={self._search_actions} "
                          f"nav={self._nav_plans} teleport={self._reset_plans} "
                          f"saved={self._saved} falsified={self._falsified}",
                          file=sys.stderr)
                return None
        while True:
            step = self._plan[self._plan_i]
            self._plan_i += 1
            if step != "RESET":
                break
            # A RESET now would be read as a FULL reset (score -> 0) -- see
            # note_action. It is also a no-op in INTENT: nothing has moved since
            # the last RESET, so the board already IS the level's initial state
            # and replaying from it is exactly where we are. Skipping is both
            # necessary and free, and it saves one scored action on every plan.
            if self._acts_since_reset == 0:
                if self._plan_i >= len(self._plan):
                    # The plan was a lone RESET (root probe) and we are already AT
                    # the root, so the board in hand IS its result. Re-enter: the
                    # top of this function settles `_pending` against `grid` and
                    # stages the next plan. Returning None instead would drop
                    # through to coverage, and the root would then be hashed from
                    # a board some later click had already moved.
                    return self._search_step(grid)
                continue
            self._search_actions += 1
            return "RESET"                             # replay from the level's initial state
        self._search_actions += 1
        b = self._buttons[step]
        return f"ACTION6_r{b[0]}_c{b[1]}"

    @staticmethod
    def _parse(action_str: str) -> Optional[Tuple[int, int]]:
        if not action_str.startswith("ACTION6_"):
            return None
        try:
            parts = action_str.split("_")
            return int(parts[1][1:]), int(parts[2][1:])
        except Exception:
            return None

    def update(self, action_str: str, changed: bool, reward: float):
        rc = self._parse(action_str)
        if rc is None:
            return
        st = self._stats.setdefault(self._key(*rc), [0, 0, 0, 0])
        st[0] += 1
        st[1] += 1 if changed else 0
        st[2] += 1 if reward > 0 else 0
        self._total_clicks += 1
        self._exact.add(rc)                          # exact-pixel coverage memory
        if changed:
            self._reactive[self._key(*rc)] = rc      # exact coord that worked here

    def mark_loss(self, action_str: str):
        """The click in `action_str` immediately preceded a GAME_OVER: blame its
        cell so future ranking backs off from it (persists across the restart)."""
        rc = self._parse(action_str)
        if rc is None:
            return
        st = self._stats.setdefault(self._key(*rc), [0, 0, 0, 0])
        st[3] += 1

    def is_lethal(self, r: int, c: int) -> bool:
        """True if a click at/near (r,c) has previously caused a GAME_OVER. Lets an
        external planner (e.g. an LLM click plan) refuse a target coverage already
        learned is fatal, so a suggested sequence can never score below the floor."""
        st = self._stats.get(self._key(r, c))
        return bool(st and st[3] > 0)

    def inert_cell(self) -> Optional[Tuple[int, int]]:
        """A representative coord of a cell that's been clicked, NEVER changed the
        frame, and never lost -- clicking it is a safe no-op that holds the board
        state STILL. Used to idle without mutating the board while a background LLM
        plan (reasoned about the current board) is still generating, so the plan
        lands on the same board it was computed from. None if none learned yet."""
        best = None
        for (kr, kc), st in self._stats.items():
            clicks, changes, rewards, losses = st
            if clicks > 0 and changes == 0 and losses == 0:
                cand = (kr * self.QUANT + self.QUANT // 2, kc * self.QUANT + self.QUANT // 2)
                # prefer the most-confirmed-inert cell (highest click count)
                if best is None or clicks > best[0]:
                    best = (clicks, cand)
        return best[1] if best else None

    def _lattice(self, h: int, w: int) -> List[Tuple[int, int]]:
        """Cell-centre grid at QUANT resolution -- guarantees sparse-button games
        (whole interactive surface = a couple of cells) get systematically swept,
        which pure object-centre targeting would miss."""
        half = self.QUANT // 2
        return [(r, c) for r in range(half, h, self.QUANT)
                for c in range(half, w, self.QUANT)]

    # Coverage refinement strides, coarse to fine. The QUANT lattice is a sample,
    # not a sweep: a 32x32 board is only 64 cells and the budget is 1500, so once
    # every cell has been touched, rank() can do nothing but re-order cells it has
    # already tried -- measured as ~70 distinct cells absorbing ~1300 clicks. Each
    # finer stride is genuinely new information, because reactivity is a property
    # of the PIXEL: a button live at one pixel of a QUANT cell is invisible to a
    # probe of that cell's centre.
    REFINE_STRIDES = (2, 1)

    def fresh_cell(self, grid: np.ndarray) -> Optional[str]:
        """A click whose exact pixel has NEVER been clicked, coarse-to-fine.

        Returns None only when every pixel of the board has been clicked -- at
        which point clicking truly cannot learn anything more and the caller must
        change modality. Lethal cells are skipped: a coord that already caused a
        GAME_OVER is not new information worth dying for again.
        """
        if grid is None or grid.ndim != 2:
            return None
        h, w = grid.shape
        for stride in (self.QUANT,) + self.REFINE_STRIDES:
            half = stride // 2
            cands = [(r, c)
                     for r in range(half, h, stride)
                     for c in range(half, w, stride)
                     if (r, c) not in self._exact and not self.is_lethal(r, c)]
            if cands:
                r, c = random.choice(cands)
                return f"ACTION6_r{r}_c{c}"
        return None

    def act(self, grid: np.ndarray, encoder: StateEncoder, epsilon: float) -> str:
        # O3: the level's trajectory, as feature vectors. Recorded HERE and not in
        # _search_step so that coverage clicks are in it too -- the trace has to be the
        # whole path to the level completion, not just the searched part of it.
        if not self._o3_off:
            self._trace.append(ProgressModel.features(grid))
        if self._o3_dump:
            try:
                self._dump_grids.append(np.asarray(grid, dtype=np.uint8).copy())
            except Exception:
                pass
        # If a small button set has been discovered, drive the state-graph BFS
        # (env-as-simulator via RESET). Otherwise cover the board to discover it.
        if self.searching:
            step = self._search_step(grid)
            if step is not None:
                return step                            # None => search just exhausted; fall through
        # O4 (hard constraint, not a term): _enter_search emits a RESET as its first
        # action, and the engine reads a RESET at action_count == 0 as a FULL reset --
        # every level already won is destroyed. Never enter search standing on a fresh
        # counter, no matter how attractive the button set looks.
        if (self._phase == "discover" and self._acts_since_reset > 0
                and self._total_clicks >= self.MIN_DISCOVER_CLICKS):
            buttons = self._detect_buttons()
            if (self.MIN_BUTTONS_FOR_SEARCH <= len(buttons) <= self.MAX_BUTTONS
                    and self._abstraction_is_live(buttons)):
                # A RESET is only safe for the level's FIRST entry, where coverage
                # has guaranteed real clicks since the level began. Every re-entry
                # runs RESET-free (O4: never gamble a banked level on a counter we
                # cannot observe). BOTH stop records count here: a first episode that
                # ended on the state cap leaves `_exhausted_sets` empty, and keying the
                # guard on that list alone would let the re-entry emit a RESET.
                first_entry = not (self._exhausted_sets or self._suspended_sets)
                self._enter_search(buttons, from_reset=first_entry)
                step = self._search_step(grid)
                if step is not None:
                    return step
        return self._coverage_click(grid, encoder, epsilon)

    def _coverage_click(self, grid: np.ndarray, encoder: StateEncoder, epsilon: float) -> str:
        h, w = grid.shape
        # Exploration: bias toward a lattice cell we've NEVER clicked (remembered
        # coverage) rather than un-remembered uniform-random that re-hits cells.
        if random.random() < epsilon:
            untried = [rc for rc in self._lattice(h, w)
                       if self._key(*rc) not in self._stats]
            if untried:
                r, c = random.choice(untried)
                return f"ACTION6_r{r}_c{c}"
            return f"ACTION6_r{random.randrange(h)}_c{random.randrange(w)}"

        # Candidates: object centres (small first -- the board panel is the biggest
        # object and rarely interactive) UNION the coverage lattice.
        objs = sorted(encoder.objects(grid), key=lambda o: o.size)[:self.MAX_TARGETS]
        cands = [(int(round(o.center[0])), int(round(o.center[1]))) for o in objs]
        cands += self._lattice(h, w)
        seen: set = set()
        uniq: List[Tuple[int, int]] = []
        for rc in cands:
            k = self._key(*rc)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(rc)
        if not uniq:
            uniq = [(h // 2, w // 2)]

        def rank(rc):
            st = self._stats.get(self._key(*rc))
            if st is None:
                return (0, 0.0, 0, random.random())          # never tried: coverage
            clicks, changes, rewards, losses = st
            loss_rate = losses / clicks if clicks else 0.0
            if rewards > 0:                                    # exploit what paid off
                return (-1, -rewards, loss_rate, random.random())
            dead = 0 if changes > 0 else 1                     # changed-safe before inert
            # non-rewarding tried cells: least lethal, then least explored, last.
            return (1 + dead, loss_rate, clicks, random.random())

        r, c = min(uniq, key=rank)
        return f"ACTION6_r{r}_c{c}"


# ==========================================
# COMPONENT 5.8: STATE GRAPH + DEAD-ACTION TRACKER
# ==========================================
#
# The games are DETERMINISTIC: the same action from the same frame always
# yields the same next frame, so an explicit graph of observed transitions is
# an EXACT world model -- no learning error, no training. It makes exploration
# DIRECTED: a (state, action) pair whose outcome is already recorded is never
# re-spent, and when the current state is exhausted the agent walks already-
# proven edges to the nearest state that still has an untested action (Blind
# Squirrel's whole strategy -- 6.7% in the preview with nothing else). RHAE
# squares wasted actions, so eliminating re-tests is pure gain. If a hidden
# timer/counter makes states never repeat, every state is novel and the graph
# degrades gracefully to "try each action once per new state".
#
# ACTION6 clicks are excluded throughout: they are parameterized by pixel, so
# per-state bookkeeping over the bare action is meaningless -- the ClickPlanner
# owns that surface with its own per-cell stats.

class DeadActionTracker:
    """Flags actions that have NEVER changed the frame anywhere in the level.

    Per-(state, action) inertness is already captured by the graph (a no-change
    transition is a self-loop edge, never re-suggested). What the graph cannot
    see is an action that is inert from EVERY state (e.g. an unused ACTION5 in
    this game): after MIN_TRIALS no-change outcomes with zero changes ever,
    exploration stops spending scored actions on it. One observed change
    whitelists the action for the rest of the level.

    CLICKS ARE TRACKED PER CELL, not per base action. A click is parameterized
    by pixel, so "ACTION6 is dead" is almost never true and was the wrong
    question: the useful fact is that THIS cell has been proven inert, and that
    EVERY cell on record is inert (the coverage surface is exhausted, so more
    clicking cannot learn anything). Both are reported separately from is_dead,
    whose ACTION6 answer stays False -- callers filter the bare action name out
    of `valid`, and suppressing clicks wholesale there would silently disarm the
    only modality a click-only game has.
    """
    MIN_TRIALS = 3
    CELL_MIN_TRIALS = 2             # a click cell is judged on fewer samples than a
                                    # key: there are hundreds of them and each costs
                                    # a scored action to sample.

    def __init__(self):
        self._no_change: Dict[str, int] = {}
        self._changed: set = set()
        self._cell_no_change: Dict[str, int] = {}
        self._cell_changed: set = set()

    def update(self, action: str, changed: bool):
        a = base_action(action)
        if a == "ACTION6":
            key = str(action)           # the full ACTION6_rR_cC -- one cell
            if changed:
                self._cell_changed.add(key)
            else:
                self._cell_no_change[key] = self._cell_no_change.get(key, 0) + 1
            return
        if changed:
            self._changed.add(a)
        else:
            self._no_change[a] = self._no_change.get(a, 0) + 1

    def is_dead(self, action: str) -> bool:
        a = base_action(action)
        if a == "ACTION6":
            return False
        return a not in self._changed and self._no_change.get(a, 0) >= self.MIN_TRIALS

    def cell_is_dead(self, action: str) -> bool:
        """This exact click cell has been sampled enough and never did anything."""
        key = str(action)
        return (key not in self._cell_changed
                and self._cell_no_change.get(key, 0) >= self.CELL_MIN_TRIALS)

    def clicks_exhausted(self) -> bool:
        """Every click cell ever tried is proven inert -- the click surface as
        sampled so far is dead, so escalation must widen it or change modality
        rather than re-rank the same cells."""
        seen = set(self._cell_no_change) | self._cell_changed
        if not seen:
            return False
        return all(self.cell_is_dead(k) for k in seen)


class StateGraph:
    """Exact directed graph of observed frames and transitions (per level).

    Persists across intra-level RESETs on purpose: the level layout is
    unchanged, so every recorded edge is still true and a restart replays into
    known territory instead of rediscovering it with scored actions.

    HUD/TIMER NORMALIZATION: games like tr87 (move-budget bar) and ls20 (timer)
    mutate a few fixed cells on EVERY action, so raw frame hashes never repeat
    and the graph degrades to "everything is novel". Cells that change in most
    transitions are HUD, not world -- they are learned online and masked out of
    the state hash. Raw frames and ensure-events are logged so the whole graph
    can be REBUILT under a new mask (exactness is preserved, nothing is guessed).
    """
    MAX_BFS_STATES = 4000               # 1500-action budget -> never near this
    MASK_RATE = 0.5                     # cell changing in >50% of transitions = HUD
    MASK_CHECK_EVERY = 25               # recompute mask (maybe rebuild) this often
    MASK_MIN_PER_ACT = 5                # samples before an action confirms/vetoes
    # HUD strips (timer / step-counter / budget bar) are CLOCKS: they tick as a
    # function of elapsed actions, NOT of what the agent does -- so the gaps
    # between a HUD line's changes are perfectly regular (tr87's bar: exactly
    # every 2nd transition; a timer: every transition). World lines (avatar
    # paths, cycled sprites, cursors) change when specific actions are taken,
    # which under randomized balanced exploration gives irregular gaps. A line
    # is HUD iff it changed often enough AND one gap length dominates.
    MASK_LINE_MIN_CHG = 10              # changes before periodicity is judged
    MASK_LINE_PERIODIC = 0.9            # modal gap must cover >=90% of gaps
    MAX_LOG = 4000                      # raw-log cap (budget is 1500 actions)
    # SACRED CELLS. Over-masking is the one UNRECOVERABLE error in this
    # component: a masked cell is invisible to the state hash, so the agent
    # literally cannot tell a winning frame from a losing one. Under-masking
    # only wastes exploration. `note_outcome` feeds the graph the reward /
    # terminal signal it otherwise ignores, and cells that change
    # DISPROPORTIONATELY at those moments are pinned out of the mask forever.
    #
    # Disproportionately -- not merely "changed at an outcome". A HUD timer
    # changes at EVERY transition, so it changes at every death too; protecting
    # on bare coincidence would pin the timer and destroy the mask on exactly
    # the games that need it most. Comparing the outcome rate against the
    # ordinary rate cancels that: a pure clock scores margin 0 by construction.
    # The guard is non-vacuous precisely where masking is at stake, since only
    # cells above MASK_RATE can be masked at all -- e.g. a goal that flickers in
    # 60% of frames but changes EVERY time the score moves. The residual blind
    # spot is a world cell that also ticks on literally every transition; no
    # statistic available here separates it from a clock (pinned as a WARN in
    # test_stategraph.py, alongside the symmetric case in test_livelock.py).
    SACRED_OUT_RATE = 0.75              # must change in >=75% of outcome transitions
    SACRED_MARGIN = 0.25                # ... and that much oftener than it normally does
    SACRED_MIN_OUT = 1                  # outcomes on record before the guard may fire

    def __init__(self):
        self.adj: Dict[int, Dict[str, int]] = {}    # hash -> {action: next_hash}
        self.untested: Dict[int, set] = {}          # hash -> actions never tried there
        self._ensures: List[Tuple[np.ndarray, tuple]] = []   # raw (grid, actions) log
        self._trans: List[Tuple[np.ndarray, str, np.ndarray]] = []
        self._change: Optional[np.ndarray] = None   # per-cell change counts
        self._rowchg: Optional[np.ndarray] = None   # transitions where row had a change
        self._colchg: Optional[np.ndarray] = None
        self._row_last: Optional[np.ndarray] = None  # transition idx of last row change
        self._col_last: Optional[np.ndarray] = None
        self._row_gaps: List[Dict[int, int]] = []   # per row: gap length -> count
        self._col_gaps: List[Dict[int, int]] = []
        # Per-action stats: action -> [cell counts, row counts, col counts, n].
        # HUD ticks under EVERY action; world cells the current policy happens to
        # hammer (graph explore repeating ACTION1 on a sprite-cycler) change under
        # SOME actions only. Requiring the rate under each well-sampled action
        # keeps the mask a property of the game, not of the policy.
        self._act: Dict[str, list] = {}
        self._mask: Optional[np.ndarray] = None     # True = ignore cell in the hash
        self._cellmask: Optional[np.ndarray] = None  # per-cell only, no row/col strips
        self._out_chg: Optional[np.ndarray] = None  # per-cell changes at OUTCOMES
        self._out_n = 0                             # outcome transitions on record
        self._sacred: Optional[np.ndarray] = None   # True = never maskable (score-relevant)
        self._n = 0
        self.grids: Dict[int, np.ndarray] = {}      # hash -> a raw grid of that state
        # (state, action) pairs a world model predicted to be no-ops: treated as
        # tested by imagination (unscored). Cleared on mask rebuilds (stale hashes).
        self._noop_skip: set = set()

    def _hash(self, grid: np.ndarray) -> int:
        m = self._mask
        if m is not None and m.shape == grid.shape:
            g = grid.copy()
            g[m] = -1
            return hash((g.shape, g.tobytes()))
        return hash((grid.shape, grid.tobytes()))

    def hud_mask(self, precise: bool = False) -> Optional[np.ndarray]:
        """The learned HUD mask (True = ignore this cell), or None.

        Public because the graph is the only component that accumulates the
        per-cell/per-action statistics the mask needs; the world model reads it
        from here instead of re-deriving a second, differently-wrong mask.

        `precise=True` drops the clock-periodic row/col strips and returns only
        the per-cell verdict -- what a rule verifier needs. See _update_mask."""
        return self._cellmask if precise else self._mask

    def sacred_cells(self) -> Optional[np.ndarray]:
        """Cells proven score-relevant by note_outcome (True = never maskable)."""
        return self._sacred

    def note_outcome(self, reward: float = 0.0, terminal: bool = False,
                     prev: Optional[np.ndarray] = None,
                     nxt: Optional[np.ndarray] = None) -> None:
        """Feed the graph the engine's SCORE / TERMINAL signal for one transition.

        The graph is otherwise policy- and reward-agnostic on purpose, but the
        mask is not a neutral summary: it decides which cells the agent is
        allowed to perceive. A cell that moves when the score moves, or when the
        run ends, is by definition one the agent must keep seeing -- so the one
        thing the graph must know about outcomes is WHICH CELLS were involved.

        Defaults to the most recently recorded transition (the one that produced
        the outcome). Explicit grids are accepted because the caller sometimes
        holds a transition the graph never recorded: a level-up is deliberately
        not fed to record() (it is a discontinuity, not a move) and a click
        transition is excluded from adjacency entirely -- both still carry
        perfectly good evidence about which cells decide the score."""
        if reward <= 0 and not terminal:
            return
        if prev is None or nxt is None:
            if not self._trans:
                return
            prev, _a, nxt = self._trans[-1]
        if prev.shape != nxt.shape:
            return
        if self._out_chg is None or self._out_chg.shape != prev.shape:
            self._out_chg = np.zeros(prev.shape, dtype=np.float32)
            self._out_n = 0
        self._out_chg += (prev != nxt)
        self._out_n += 1
        # Outcomes are rare and decisive: re-judge the mask NOW rather than
        # letting up to MASK_CHECK_EVERY more actions run while a cell that was
        # just proven score-relevant is still being ignored by the state hash.
        self._update_mask()

    def _sacred_mask(self) -> Optional[np.ndarray]:
        """Cells that change far oftener at outcomes than they normally do."""
        if (self._out_chg is None or self._out_n < self.SACRED_MIN_OUT
                or self._change is None
                or self._change.shape != self._out_chg.shape):
            return None
        out_rate = self._out_chg / self._out_n
        # Deliberately the rate over ALL transitions, not over the non-outcome
        # ones. Netting the outcomes out would be sharper statistics but the
        # counts come from different places -- a click or a level-up transition
        # reaches note_outcome and never reaches record() -- so the subtraction
        # can under-count ordinary changes, inflate the margin, and pin a clock.
        # That is the one failure this guard must not have.
        all_rate = self._change / max(1, self._n)
        s = ((out_rate >= self.SACRED_OUT_RATE)
             & ((out_rate - all_rate) >= self.SACRED_MARGIN))
        return s if s.any() else None

    def changed_masked(self, prev: np.ndarray, nxt: np.ndarray) -> bool:
        """Did the WORLD change (ignoring HUD/timer cells)? A move that only
        ticked the HUD did nothing the agent should credit as an effect."""
        d = masked_diff(prev, nxt, self._mask)
        return True if d is None else bool(d.any())

    def _ensure(self, h: int, acts) -> None:
        if h not in self.untested:
            self.untested[h] = set()
            self.adj[h] = {}
        for a in acts:
            if a not in self.adj[h]:
                self.untested[h].add(a)

    def ensure_state(self, grid: np.ndarray, valid: List[str]):
        """Register the current frame's state; unseen actions become untested."""
        acts = tuple(sorted({base_action(a) for a in valid
                             if base_action(a) != "ACTION6"}))
        if len(self._ensures) < self.MAX_LOG:
            self._ensures.append((grid, acts))
        h = self._hash(grid)
        self.grids.setdefault(h, grid)
        self._ensure(h, acts)

    def record(self, prev_grid: np.ndarray, action: str, next_grid: np.ndarray):
        a = base_action(action)
        if a == "ACTION6":
            return
        if len(self._trans) < self.MAX_LOG:
            self._trans.append((prev_grid, a, next_grid))
        if prev_grid.shape == next_grid.shape:
            if self._change is None or self._change.shape != prev_grid.shape:
                self._change = np.zeros(prev_grid.shape, dtype=np.float32)
                self._rowchg = np.zeros(prev_grid.shape[0], dtype=np.float32)
                self._colchg = np.zeros(prev_grid.shape[1], dtype=np.float32)
                self._row_last = np.full(prev_grid.shape[0], -1, dtype=np.int64)
                self._col_last = np.full(prev_grid.shape[1], -1, dtype=np.int64)
                self._row_gaps = [{} for _ in range(prev_grid.shape[0])]
                self._col_gaps = [{} for _ in range(prev_grid.shape[1])]
                self._act = {}
                self._n = 0
            d = prev_grid != next_grid
            self._change += d
            rhit, chit = d.any(axis=1), d.any(axis=0)
            self._rowchg += rhit
            self._colchg += chit
            for i in np.nonzero(rhit)[0]:
                if self._row_last[i] >= 0:
                    gap = self._n - int(self._row_last[i])
                    self._row_gaps[i][gap] = self._row_gaps[i].get(gap, 0) + 1
                self._row_last[i] = self._n
            for j in np.nonzero(chit)[0]:
                if self._col_last[j] >= 0:
                    gap = self._n - int(self._col_last[j])
                    self._col_gaps[j][gap] = self._col_gaps[j].get(gap, 0) + 1
                self._col_last[j] = self._n
            per = self._act.get(a)
            if per is None:
                per = self._act[a] = [np.zeros(prev_grid.shape, dtype=np.float32),
                                      np.zeros(prev_grid.shape[0], dtype=np.float32),
                                      np.zeros(prev_grid.shape[1], dtype=np.float32), 0]
            per[0] += d
            per[1] += d.any(axis=1)
            per[2] += d.any(axis=0)
            per[3] += 1
            self._n += 1
            if self._n % self.MASK_CHECK_EVERY == 0:
                self._update_mask()
        hp, hn = self._hash(prev_grid), self._hash(next_grid)
        self.grids.setdefault(hp, prev_grid)
        self.grids.setdefault(hn, next_grid)
        self._ensure(hp, ())                        # no-op if already known
        self.adj[hp][a] = hn
        self.untested[hp].discard(a)

    def _update_mask(self):
        if self._change is None:
            return                      # nothing comparable recorded yet
        n = max(1, self._n)
        new = (self._change / n) > self.MASK_RATE
        # Per-cell confirmation (needs >=2 well-sampled actions): a HUD cell
        # ticks under all of them; a cell only the current policy hammers
        # (repeated ACTION1 on a sprite-cycler) is quiet under another action.
        sampled = [p for p in self._act.values() if p[3] >= self.MASK_MIN_PER_ACT]
        if len(sampled) >= 2:
            for cc, rc, colcnt, na in sampled:
                new &= (cc / na) > self.MASK_RATE
        else:
            new[:] = False
        # Score-relevant cells are removed from BOTH granularities, and removed
        # again after the strips are painted -- a sacred cell sitting inside a
        # HUD row must survive the row being masked around it.
        sac = self._sacred_mask()
        self._sacred = sac
        if sac is not None:
            new &= ~sac
        # Keep the per-CELL verdict before the strips go on: it is the precise
        # one. The strips below are deliberately over-broad -- they exist to stop
        # a flickering row from churning the state hash, and painting the whole
        # row is cheap insurance there. A rule verifier cannot afford that: a
        # world cell that merely shares a row with a timer would be excused from
        # every prediction. Two consumers, two granularities, one statistic.
        self._cellmask = new.copy() if new.any() else None
        # HUD strips: clock-periodic rows/cols (see MASK_LINE_* above).
        def _periodic(gaps: Dict[int, int]) -> bool:
            tot = sum(gaps.values())
            return (tot + 1 >= self.MASK_LINE_MIN_CHG
                    and max(gaps.values()) / tot >= self.MASK_LINE_PERIODIC)
        new[[i for i, g in enumerate(self._row_gaps) if g and _periodic(g)], :] = True
        new[:, [j for j, g in enumerate(self._col_gaps) if g and _periodic(g)]] = True
        if sac is not None:
            new &= ~sac
        if not new.any():
            new = None
        old = self._mask
        if (old is None and new is None) or \
                (old is not None and new is not None and np.array_equal(old, new)):
            return
        self._mask = new
        # Every stored hash may have changed -> rebuild EXACTLY from the raw logs.
        self.adj = {}
        self.untested = {}
        self.grids = {}
        self._noop_skip = set()
        for grid, acts in self._ensures:
            h = self._hash(grid)
            self.grids.setdefault(h, grid)
            self._ensure(h, acts)
        for prev, a, nxt in self._trans:
            hp, hn = self._hash(prev), self._hash(nxt)
            self.grids.setdefault(hp, prev)
            self.grids.setdefault(hn, nxt)
            self._ensure(hp, ())
            self.adj[hp][a] = hn
            self.untested[hp].discard(a)

    def nearest_untested_grid(self, grid: np.ndarray,
                              dead: Optional[DeadActionTracker] = None,
                              noop=None, rank=None):
        return self.nearest_untested(self._hash(grid), dead, noop, rank)

    def nearest_untested(self, h: int, dead: Optional[DeadActionTracker] = None,
                         noop=None, rank=None):
        """UNSCORED BFS through known edges to the closest state that still has
        an untested (non-dead) action. Returns (action_path_to_it, untested_action)
        or None if the whole recorded graph is exhausted.

        `noop(grid, action) -> bool` is an optional world-model predicate: pairs
        it confidently predicts to be no-ops are tested by IMAGINATION instead of
        by a scored action (sticky in _noop_skip). Callers must fall back to a
        noop-less pass when this returns None -- the model may be wrong, so
        skipped pairs are re-offered before declaring true exhaustion.

        `rank(cands) -> cands` is an optional RE-ORDERING of the (state, action)
        pairs found at the SHORTEST frontier distance (COMPONENT 5.12). It is a
        permutation and never a filter: the shortest-path guarantee is untouched,
        only the choice AMONG equally-close frontier pairs changes. With
        `rank=None` this method takes the original early-return path and is
        byte-identical -- including its consumption of the shared RNG stream --
        to the version without a goal model."""
        seen = {h}
        q = deque([(h, [])])
        hits: List[tuple] = []          # (path, cur, avail) at the shortest depth
        limit = -1
        while q:
            cur, path = q.popleft()
            if limit >= 0 and len(path) > limit:
                break                   # past the shortest frontier distance
            avail = [a for a in self.untested.get(cur, ())
                     if dead is None or not dead.is_dead(a)]
            if noop is not None and avail:
                g = self.grids.get(cur)
                if g is not None:
                    keep = []
                    for a in avail:
                        if (cur, a) in self._noop_skip:
                            continue
                        if noop(g, a):
                            self._noop_skip.add((cur, a))
                        else:
                            keep.append(a)
                    avail = keep
            if avail:
                # Least-GLOBALLY-used first. Plain sorted() hammered ACTION1
                # forever on games where every frame is novel (tr87: 284 of 297
                # presses) -- depth-first down one button. Balancing usage
                # explores the action product space and feeds every action
                # enough samples for the HUD-mask veto. Ties break RANDOMLY
                # (seeded RNG): a fixed name-order tie-break degenerates into a
                # strict A1-A2-A3-A4 round-robin, which stamps a fake clock
                # period onto world rows and fools the periodicity mask.
                if rank is None:
                    return path, min(avail, key=lambda a: (
                        self._act.get(a, (0, 0, 0, 0))[3], random.random()))
                hits.append((path, cur, avail))
                limit = len(path)
                continue
            if len(seen) > self.MAX_BFS_STATES:
                break
            for a, nxt in self.adj.get(cur, {}).items():
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, path + [a]))
        if not hits:
            return None
        # Pre-sort by the SAME least-globally-used rule, then let the ranker
        # reorder: a stable sort means the goal model expresses a preference and
        # the exploration-balance tie-break survives underneath it.
        pmap = {}
        cands = []
        for path, cur, avail in hits:
            for a in sorted(avail, key=lambda x: (
                    self._act.get(x, (0, 0, 0, 0))[3], random.random())):
                pmap[(cur, a)] = path
                cands.append((cur, a))
        try:
            ordered = list(rank(cands))
        except Exception:
            ordered = cands
        if set(ordered) != set(cands) or len(ordered) != len(cands):
            ordered = cands             # a ranker that is not a permutation is ignored
        cur, a = ordered[0]
        return pmap[(cur, a)], a


class PatchWorldModel:
    """Learned LOCAL rewrite rules (COMPONENT 5.9): an executable world model.

    ARC-AGI-3 dynamics are local (moving onto track cells, walls, cursor steps,
    sprite cycles), so a cell's next value is usually a function of the action
    and the cell's 5x5 neighborhood. Those rules are learned from every recorded
    transition and -- unlike the exact StateGraph -- GENERALIZE across positions:
    a wall learned at one spot predicts a never-tried move into a wall elsewhere.
    Use: predict an untested action's outcome with unscored compute instead of
    spending a scored action to observe it (the RHAE currency exchange).

    The prediction target is transition EFFECTIVENESS (did any unmasked cell
    change), not per-cell values: games with global side-effects (tu93's
    platforms move by a history-dependent rule after every VALID move) make
    per-cell outcomes unpredictable, but ineffectiveness is still decided at one
    local site (the avatar pressed against a non-track cell). A (state, action)
    is called a no-op only when EVERY unmasked cell's patch is known (seen >=
    MIN_SEEN times under that action) AND the state contains a BLOCKING WITNESS:
    a patch that has only ever preceded ineffective transitions (>= WITNESS_MIN
    times). Unknown contexts always fall back to real testing, and StateGraph's
    caller-side exhaustion fallback re-offers model-skipped pairs, so
    completeness is never lost to a wrong rule.
    HUD cells (StateGraph mask) are neutralized both as patch context and in the
    effectiveness label -- a ticking budget bar would otherwise make every
    transition look effective (the same trap that broke raw state hashing).
    """
    K = 2                    # neighborhood radius -> 5x5 patches
    MIN_SEEN = 2             # occurrences before a patch counts as known
    WITNESS_MIN = 2          # all-ineffective occurrences to certify a blocker
    PAD = -2                 # out-of-grid / HUD fill (never a real color)
    MEMO_REFRESH = 25        # re-evaluate a cached False after this much new data

    def __init__(self):
        # (action, patch_bytes) -> [effective_count, ineffective_count]
        self.stats: Dict[Tuple[str, bytes], list] = {}
        self.trained = 0
        self._memo: Dict[Tuple[int, str], Tuple[int, bool]] = {}

    def _patches(self, grid: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        g = np.array(grid, dtype=np.int32)          # own copy; PAD needs signed
        if mask is not None and mask.shape == g.shape:
            g[mask] = self.PAD
        k = self.K
        p = np.pad(g, k, mode="constant", constant_values=self.PAD)
        w = np.lib.stride_tricks.sliding_window_view(p, (2 * k + 1, 2 * k + 1))
        return np.ascontiguousarray(w.reshape(g.size, -1))

    @staticmethod
    def _live(mask: Optional[np.ndarray], shape) -> np.ndarray:
        if mask is not None and mask.shape == shape:
            return np.nonzero(~mask.ravel())[0]
        return np.arange(int(np.prod(shape)))

    def learn(self, prev: np.ndarray, action: str, nxt: np.ndarray,
              mask: Optional[np.ndarray] = None):
        a = base_action(action)
        if a == "ACTION6" or prev.shape != nxt.shape:
            return
        d = prev != nxt
        if mask is not None and mask.shape == d.shape:
            d = d & ~mask
        eff = bool(d.any())
        pat = self._patches(prev, mask)
        stats = self.stats
        for i in self._live(mask, prev.shape):
            key = (a, pat[i].tobytes())
            s = stats.get(key)
            if s is None:
                s = stats[key] = [0, 0]
            s[0 if eff else 1] += 1
        self.trained += 1

    def predict_noop(self, grid: np.ndarray, action: str,
                     mask: Optional[np.ndarray] = None) -> bool:
        """True iff every live cell's context is known AND a blocking witness
        (an always-ineffective patch) is present."""
        a = base_action(action)
        if a == "ACTION6":
            return False
        key = (hash(grid.tobytes()), a)
        memo = self._memo.get(key)
        if memo is not None and (memo[1] or
                                 self.trained - memo[0] < self.MEMO_REFRESH):
            return memo[1]
        pat = self._patches(grid, mask)
        stats = self.stats
        witness = False
        result = True
        for i in self._live(mask, grid.shape):
            s = stats.get((a, pat[i].tobytes()))
            if s is None or s[0] + s[1] < self.MIN_SEEN:
                result = False                  # unknown context -> test for real
                break
            if s[0] == 0 and s[1] >= self.WITNESS_MIN:
                witness = True
        result = result and witness
        self._memo[key] = (self.trained, result)
        return result


# ==========================================
# COMPONENT 5.11: LIVELOCK BREAKER  (scored-action circuit breaker)
# ==========================================
#
# Measured pathology, whole-budget scale: on the real games most scored actions
# buy nothing. Click games exhaust their coverage lattice after ~60-70 cells and
# then re-click those same cells for the remaining ~1400 actions; movement games
# re-spend (state, action) pairs whose outcome is already on record; some levels
# never move the world at all and still burn the full budget. RHAE squares the
# action ratio, so this is the single largest source of lost score -- larger
# than any planner improvement, because it wastes the budget the planner needs.
#
# The breaker owns exactly ONE question -- "is the current layer still learning?"
# -- and answers it from two facts that need no game knowledge: consecutive
# HUD-masked no-ops, and how often a (state, action) pair has been re-spent
# without effect. It never chooses a game-specific action; it only tells the
# orchestrator to escalate, and the ESCALATION LADDER lives in MyAgent because
# only there are the graph, the dead-action tracker and the click surface all in
# scope.
#
# Determinism is what makes the ledger sound: the same action from the same
# frame yields the same next frame, so a pair already observed to be inert can
# never become useful. SPEND_REPEAT_CAP is 2 rather than 1 because that premise
# holds only for VISIBLE state -- games with history-dependent hidden state
# (a cycle counter the frame doesn't show) can legitimately differ on a second
# visit, and one free re-test is the price of not assuming they can't.

class LivelockBreaker:
    """Tracks whether scored actions are still buying information.

    Per level: consecutive masked no-ops, whether the world has EVER moved, and
    a ledger of (state hash, action) -> [times spent, ever changed]. Everything
    it reports is a fact about the run, not a policy -- see MyAgent._escalate
    for what is done about it.
    """

    def __init__(self):
        self.new_level()

    def new_level(self):
        self.run = 0                 # consecutive masked no-ops
        self.level_actions = 0
        self.level_changes = 0
        # The HARD invariant is judged over a WINDOW, not the whole level, and the
        # window restarts when the BREAKER restarts the level. That is not a
        # loophole: the breaker spent a RESET precisely in order to land on a
        # different frame, so it has to be allowed to observe the result. Without
        # it the invariant re-trips on the very next action and the whole reset
        # budget burns in consecutive steps, which is the opposite of escalating.
        # An ENGINE-forced restart (GAME_OVER) gets no such window -- the agent
        # did not choose it and has learned nothing from it.
        self.win_actions = 0
        self.win_changes = 0
        self.resets = 0
        self.escalations = 0
        # Actions since the agent last reached a state it had NEVER seen. This is
        # the "is anyone still learning?" signal, and it is deliberately separate
        # from the change counters: a game can change its frame every single action
        # (a spinning HUD-free animation) while cycling the same six states, and it
        # can also sit visually still while the agent discovers new structure.
        self.since_novel = 0
        self._spend: Dict[Tuple[int, str], List[int]] = {}

    # -- bookkeeping -------------------------------------------------------
    def update(self, state_hash: Optional[int], action: str, changed: bool):
        """Record one SCORED action and what it did to the masked world."""
        if not action or action == "RESET":
            # A restart is not evidence about the layer's productivity: it is the
            # escalation itself, and the frame it lands on is chosen by the
            # engine. Counting it as a change would clear the streak for free.
            return
        self.level_actions += 1
        self.win_actions += 1
        self.since_novel += 1
        if changed:
            self.run = 0
            self.level_changes += 1
            self.win_changes += 1
        else:
            self.run += 1
        if state_hash is not None:
            e = self._spend.setdefault((state_hash, str(action)), [0, 0])
            e[0] += 1
            e[1] |= int(bool(changed))

    def note_reset(self, by_breaker: bool = False):
        """A restart happened. The streak clears (the frame is replaced
        wholesale) but the spend ledger does NOT: the layout is identical after
        an intra-level restart, so a pair proven inert is still inert. Only
        restarts the BREAKER chose count against its reset budget -- a GAME_OVER
        the engine forced is not the breaker spending anything."""
        self.run = 0
        if by_breaker:
            self.resets += 1
            self.win_actions = 0     # a new window to observe what the restart did
            self.win_changes = 0

    def note_novel(self):
        """The agent reached structure it had never seen. Called by the owner of
        the evidence (MyAgent watches the state graph grow) rather than inferred
        here, so the breaker keeps its single job: report facts about spend."""
        self.since_novel = 0

    # -- questions ---------------------------------------------------------
    def learning(self) -> bool:
        """Has anything genuinely new arrived recently? While this holds, the weak
        `stale` evidence must not take control away from the driving layer: a
        planner working through a frontier re-enters known states constantly, and
        interrupting it there is how a working solve gets destroyed."""
        return self.since_novel < NOVEL_PATIENCE

    def tripped(self) -> bool:
        """Is the current layer stuck? Either it has produced nothing for
        NOOP_TRIP actions in a row, or the world has not moved ONCE in the last
        NOOP_HARD actions (the hard invariant, measured over the window since the
        last breaker-chosen restart -- see note_reset)."""
        return (self.run >= NOOP_TRIP
                or (self.win_changes == 0 and self.win_actions >= NOOP_HARD))

    def spent(self, state_hash: Optional[int], action: str) -> int:
        if state_hash is None:
            return 0
        e = self._spend.get((state_hash, str(action)))
        return e[0] if e else 0

    def stale(self, state_hash: Optional[int], action: str) -> bool:
        """Has this exact (state, action) already been spent to no effect, at
        least SPEND_REPEAT_CAP times? Re-spending it is provably worthless in a
        deterministic game."""
        if state_hash is None or not action or action == "RESET":
            return False
        e = self._spend.get((state_hash, str(action)))
        return bool(e and e[1] == 0 and e[0] >= SPEND_REPEAT_CAP)

    def may_reset(self) -> bool:
        """A level restart is the last rung. Three independent guards:

        - LEVEL AGE. The engine reads a RESET taken while its internal
          action_count is 0 as a FULL reset, which wipes levels_completed (see
          ClickPlanner). Only the level's own age makes RESET safe.
        - WINDOW. The restart must have been EARNED by observing NOOP_HARD fresh
          actions that changed nothing. Without this, the rung is also reachable
          through the `stale` path, and since the step after a RESET is not
          counted (it is a replay, not a move), the level age never advances --
          measured as three RESETs in five steps, the whole budget gone at once.
        - BUDGET. A fixed point must not become a RESET loop.
        """
        return (self.resets < MAX_BREAK_RESETS
                and self.level_actions >= NOOP_HARD
                and self.win_actions >= NOOP_HARD)


# ==========================================
# COMPONENT 5.10: TIMELINE  (append-only ground truth; Schema-harness Phase 1)
# ==========================================
#
# The Schema-harness discipline: keep an APPEND-ONLY, immutable record of every
# real transition. Learned models are provisional and get revised; the Timeline
# never does -- it is the ground truth every model must be CERTIFIED against
# (full-history backtest) before the agent is allowed to commit a multi-step
# plan inside that model. It deliberately survives GAME_OVER -> RESET
# (deterministic games: evidence from a failed attempt still constrains the
# theory) and is only SEGMENTED by level tag -- never cleared -- on level-up,
# because a new layout obsoletes positional facts but not the record itself.

class TimelineEntry:
    # `terminal` is part of the ground truth, not a GoalModel private note: deaths
    # are the FIRST outcome evidence any game ever produces (22 of 25 games have
    # never seen a win), so a record that cannot say "this is the frame we died
    # on" is missing the only signal those games emit. Recorded here rather than
    # in a second, parallel log so every model is certified against one history.
    __slots__ = ("prev", "action", "nxt", "reward", "level", "terminal", "_mv")

    def __init__(self, prev, action, nxt, reward, level, terminal=False):
        self.prev = prev
        self.action = action
        self.nxt = nxt
        self.reward = reward
        self.level = level
        self.terminal = bool(terminal)
        self._mv = None      # cached (avatar_color, motion) -- replay cost amortiser

    def motion(self, color):
        """Observed centroid motion of `color` across this transition, or None
        when the colour is absent on either side / shapes differ (out of a
        movement theory's scope). Cached per colour: backtests replay the whole
        history every time the theory is revised, and the np.nonzero scans are
        the dominant cost."""
        if self._mv is not None and self._mv[0] == color:
            return self._mv[1]
        res = None
        if (self.prev is not None and self.nxt is not None
                and self.prev.shape == self.nxt.shape):
            prs, pcs = np.nonzero(self.prev == color)
            nrs, ncs = np.nonzero(self.nxt == color)
            if len(prs) and len(nrs):
                pr, pc = prs.mean(), pcs.mean()
                res = (pr, pc, nrs.mean() - pr, ncs.mean() - pc)
        self._mv = (color, res)
        return res


class Timeline:
    def __init__(self):
        self.entries: List[TimelineEntry] = []

    def append(self, prev, action, nxt, reward, level, terminal=False):
        self.entries.append(TimelineEntry(prev, action, nxt, reward, level, terminal))

    def __len__(self):
        return len(self.entries)


# ==========================================
# COMPONENT 5.12: GOAL MODEL  (the order over world-states)
# ==========================================
#
# Every component before this one answers a question about WHAT IS -- Eyes: what
# is on the screen; GridDSL/Synth: how the screen changes; Livelock: am I
# repeating myself; StateGraph: which world-state is this and what have I not
# tried here. NONE of them answers "which state is BETTER". GoalModel is that one
# missing relation: an ORDER over world-states, learned online from the Timeline,
# carrying a calibrated credibility, and exposed to the rest of the agent as a
# RE-RANKING -- never as a gate. Architecture: Docs/goalmodel_architecture.md.
#
# THE HAZARD (why it is a re-ranker and not a planner): a wrong goal pursued
# confidently is strictly WORSE than no goal at all. No goal is an UNBIASED walk
# -- it wastes actions but keeps a real chance of stumbling onto the objective,
# which is how the agent's completed levels were actually won. A wrong goal is a
# BIASED walk toward a fixed attractor, and the livelock breaker cannot see it:
# every pursuit is a different path to the same decoy, so no state repeats often
# enough to trip it. The codebase already carries this scar -- `non_goal_colors`
# exists because `_nearest_target` kept walking into decoys.
#
# So the structural floor is: GoalModel may PERMUTE the untested frontier and may
# never REMOVE a pair from it. Under potential-based shaping (Ng, Harada &
# Russell 1999) the optimal policy set is invariant, so the worst case of an
# arbitrarily wrong potential is a permutation of an order that was already
# arbitrary. With an empty pool the agent is byte-identical to the one without
# this component -- asserted in test_goalmodel.py, not promised in a comment.
#
# THE TRAP (5.1): a step counter is a PERFECT fake progress coordinate -- it is
# monotone, irreversible AND bounded, so it satisfies every structural test at
# once, and an agent that adopts it concludes it is making progress on literally
# every action. The separator is the same disproportionality rule that made
# sacred cells safe: a real progress coordinate advances under SOME actions and
# not others; a clock advances under all of them.
#
# LIFECYCLE (the one place this must NOT copy StateGraph): the readout vocabulary
# and its credibilities persist for the WHOLE GAME, across level boundaries. RHAE's
# denominator spans every level of a game, so a level-1-only completion is worth
# ~3%; the score lives in layouts the agent has never seen, and a learned
# win-condition is the only object in the system that can cross a level boundary.
# What resets per level is POSITIONAL bindings (which cell holds the counter,
# which panels) and the value RANGES used to normalise -- the same split the rest
# of the agent already makes, where `rplanner` keeps avatar colour and per-action
# displacements across a level-up and forgets only coordinates.

_NAN = float("nan")


def _cc_areas(mask: np.ndarray, cap: int) -> Optional[List[int]]:
    """4-connected component areas of a boolean mask; None when out of scope.

    O(k) in the number of SET cells, not in the grid size: a colour occupying a
    few sprites costs a few steps. Past `cap` cells it refuses to answer rather
    than running a 4096-cell flood fill -- certify() replays the entire Timeline,
    so a readout nobody can afford is a readout that does not exist."""
    rs, cs = np.nonzero(mask)
    if len(rs) == 0:
        return []
    if len(rs) > cap:
        return None
    todo = set(zip(rs.tolist(), cs.tolist()))
    areas = []
    while todo:
        stack = [todo.pop()]
        a = 0
        while stack:
            r, c = stack.pop()
            a += 1
            for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nb in todo:
                    todo.discard(nb)
                    stack.append(nb)
        areas.append(a)
    return areas


def _bands(n: int, const: set, min_len: int = 2) -> List[Tuple[int, int]]:
    """Maximal runs of NON-separator indices, each at least `min_len` long."""
    out, start = [], None
    for i in range(n):
        if i in const:
            if start is not None and i - start >= min_len:
                out.append((start, i))
            start = None
        elif start is None:
            start = i
    if start is not None and n - start >= min_len:
        out.append((start, n))
    return out


class GridStats:
    """Memoised substrate for the readout vocabulary.

    Built ONCE per grid and shared by every feature that reads it: a full-history
    backtest re-scans thousands of frames, so the per-grid scans -- not the
    per-feature arithmetic -- decide whether certification is affordable at all.
    Every entry point is lazy: a feature that never asks for connected components
    never pays for them."""
    CC_MAX_CELLS = 600
    MAX_PANELS = 6

    __slots__ = ("g", "_counts", "_cent", "_cc", "_lines", "_contact", "_panels")

    def __init__(self, g: np.ndarray):
        self.g = g
        self._counts = None
        self._cent: Dict[int, Any] = {}
        self._cc: Dict[int, Any] = {}
        self._lines: Dict[int, Any] = {}
        self._contact: Dict[tuple, float] = {}
        self._panels = None

    def counts(self) -> np.ndarray:
        if self._counts is None:
            g = self.g
            hi = int(g.max()) + 1 if g.size else 1
            self._counts = np.bincount(np.maximum(g, 0).ravel(),
                                       minlength=max(16, hi))
        return self._counts

    def count(self, c: int) -> float:
        cn = self.counts()
        return float(cn[c]) if 0 <= c < len(cn) else 0.0

    def centroid(self, c: int):
        if c not in self._cent:
            rs, cs = np.nonzero(self.g == c)
            self._cent[c] = (float(rs.mean()), float(cs.mean())) if len(rs) else None
        return self._cent[c]

    def cc(self, c: int) -> Optional[List[int]]:
        if c not in self._cc:
            self._cc[c] = _cc_areas(self.g == c, self.CC_MAX_CELLS)
        return self._cc[c]

    def lines(self, c: int):
        if c not in self._lines:
            m = self.g == c
            self._lines[c] = (frozenset(np.nonzero(m.any(axis=1))[0].tolist()),
                              frozenset(np.nonzero(m.any(axis=0))[0].tolist()))
        return self._lines[c]

    def contact(self, a: int, b: int) -> float:
        k = (a, b)
        if k not in self._contact:
            ma, mb = self.g == a, self.g == b
            n = int((ma[:-1] & mb[1:]).sum() + (ma[1:] & mb[:-1]).sum()
                    + (ma[:, :-1] & mb[:, 1:]).sum() + (ma[:, 1:] & mb[:, :-1]).sum())
            self._contact[k] = float(n)
        return self._contact[k]

    def panels(self) -> List[Tuple[int, int, int, int]]:
        """Equal-shaped rectangular panels split by uniform separator lines.

        The multi-panel layout ("make this one look like that one") is a real
        ARC-AGI-3 shape -- Eyes already resolves those frames correctly, which is
        why the big 'phantom objects' turned out to be panels. Ragged splits are
        refused: `agree` compares two panels cell-by-cell, so equal shape is part
        of BEING a panel set, not a precondition to check later."""
        if self._panels is None:
            g = self.g
            h, w = g.shape
            rconst = {i for i in range(h) if bool((g[i] == g[i, 0]).all())}
            cconst = {j for j in range(w) if bool((g[:, j] == g[0, j]).all())}
            rb, cb = _bands(h, rconst), _bands(w, cconst)
            out = []
            if len(rb) * len(cb) >= 2:
                out = [(r0, r1, c0, c1) for (r0, r1) in rb for (c0, c1) in cb]
            if out:
                sh = (out[0][1] - out[0][0], out[0][3] - out[0][2])
                if len(out) > self.MAX_PANELS or not all(
                        (p[1] - p[0], p[3] - p[2]) == sh for p in out):
                    out = []
            self._panels = out
        return self._panels


class Readout:
    """A TOTAL function grid -> scalar from a FIXED, CLOSED vocabulary.

    Closed on purpose (anti-responsibility 7): WHICH readout is right is learned,
    the candidate set is not. A vocabulary that grows to fit a game is a per-game
    branch wearing a hat. Total on purpose (the DSL discipline already in force):
    out of scope returns NaN, never an exception, so a backtest can never be
    aborted by a frame the readout was not designed for."""
    __slots__ = ("kind", "args", "dl")
    _DL = {"exists": 2, "count": 2, "cell": 3, "regions": 3, "dist": 3,
           "align": 4, "contact": 4, "objects": 4, "agree": 5}
    # Readouts whose natural floor is 0. Used by normalise() so that "drive this
    # to zero" has a dense gradient even before zero has ever been observed.
    _FLOOR0 = ("count", "exists", "regions", "objects", "dist", "contact",
               "align", "agree")

    def __init__(self, kind: str, args=()):
        self.kind = kind
        self.args = tuple(int(a) for a in args)
        self.dl = self._DL.get(kind, 9) + len(self.args)

    @property
    def key(self) -> tuple:
        return (self.kind, self.args)

    def has_floor(self) -> bool:
        return self.kind in self._FLOOR0

    def positional(self) -> bool:
        """Is this readout bound to a LAYOUT rather than to the game's rules?

        Colours are game-scoped here (the palette means the same thing on every
        level -- the same reason `rplanner` carries avatar_color across a
        level-up); coordinates and panel indices are not."""
        return self.kind in ("cell", "agree")

    def value(self, st: Optional[GridStats]) -> float:
        if st is None:
            return _NAN
        k, a = self.kind, self.args
        try:
            if k == "count":
                return st.count(a[0])
            if k == "exists":
                return 1.0 if st.count(a[0]) > 0 else 0.0
            if k == "cell":
                g = st.g
                if a[0] >= g.shape[0] or a[1] >= g.shape[1]:
                    return _NAN
                return float(g[a[0], a[1]])
            if k == "regions":
                cc = st.cc(a[0])
                return _NAN if cc is None else float(len(cc))
            if k == "objects":
                cc = st.cc(a[0])
                return _NAN if cc is None else float(sum(1 for x in cc if x == a[1]))
            if k == "dist":
                p, q = st.centroid(a[0]), st.centroid(a[1])
                if p is None or q is None:
                    return _NAN
                return float(abs(p[0] - q[0]) + abs(p[1] - q[1]))
            if k == "contact":
                return st.contact(a[0], a[1])
            if k == "align":
                (ra, ca), (rb, cb) = st.lines(a[0]), st.lines(a[1])
                if not ra or not rb:
                    return _NAN
                return float(len(ra & rb) + len(ca & cb))
            if k == "agree":
                ps = st.panels()
                if a[0] >= len(ps) or a[1] >= len(ps):
                    return _NAN
                p, q = ps[a[0]], ps[a[1]]
                A, B = st.g[p[0]:p[1], p[2]:p[3]], st.g[q[0]:q[1], q[2]:q[3]]
                if A.shape != B.shape or A.size == 0:
                    return _NAN
                return float((A == B).sum())
        except Exception:
            return _NAN
        return _NAN

    def describe(self) -> str:
        return f"{self.kind}({','.join(str(x) for x in self.args)})"


class GoalPredicate:
    """A goal as a STATE PREDICATE, never as a pixel coordinate.

    A coordinate cannot survive a change of layout -- that is exactly why the old
    `_llm_goal` could not transfer to the next level, and RHAE's denominator says
    the score lives in levels the agent has never seen."""
    __slots__ = ("ro", "op", "thr", "key")

    def __init__(self, ro: Readout, op: str, thr: float, key: tuple):
        self.ro, self.op, self.thr, self.key = ro, op, float(thr), key

    def holds(self, grid) -> bool:
        try:
            g = np.asarray(grid)
        except Exception:
            return False
        v = self.ro.value(GridStats(g)) if g.ndim == 2 else _NAN
        if v != v:                                  # NaN -> out of scope
            return False
        return v >= self.thr if self.op == ">=" else v <= self.thr

    def describe(self) -> str:
        return f"{self.ro.describe()} {self.op} {self.thr:g}"


class Feature:
    """A Readout + a learned DIRECTION + Beta credibility + its evidence channel.

    Direction starts at 0 ("no opinion") and is assigned only by certification.
    A feature with direction 0 contributes nothing to the potential and can never
    be pursued -- which is what makes an un-evidenced pool a strict no-op."""
    __slots__ = ("ro", "direction", "channel", "alpha", "beta", "reason")

    def __init__(self, ro: Readout, alpha: float = 1.0, beta: float = 1.0):
        self.ro = ro
        self.direction = 0          # +1 progress increases it, -1 decreases it
        self.channel = ""           # "A" confirmed, "B" structural, "" none
        self.alpha, self.beta = alpha, beta
        self.reason = ""

    @property
    def key(self) -> tuple:
        return self.ro.key

    def post_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def steers(self) -> bool:
        """May this feature move the potential AT ALL?

        Measured, not assumed (25-game paired run, goals-off vs goals-on): every
        adopted feature in every game sat at the untested Beta(1,1) prior, because
        the ONLY writer to alpha/beta is `_settle_pursuit`, which runs only when
        `target()` has set a pursuit -- and `target()` has no call site yet. So the
        Bayesian layer was inert and `potential()` degenerated to an unweighted
        vote of unconfirmed structural guesses. The result was a coin flip:
        tr87 -11.1 re-spend, wa30 -3.0, m0r0 -2.7 against sp80 +10.5, cd82 +7.1;
        dev mean -0.37pts, i.e. noise. sp80 is the hazard in the data -- no-ops
        fell 13.3pts while re-spend ROSE 10.5, the agent trading idling for
        re-treading because it was pulled toward a wrong coordinate.

        So: CHANNEL A steers on adoption, because `_confirmed` already paid for it
        with real score events and the base-rate margin. CHANNEL B is a structural
        HYPOTHESIS with no outcome behind it, and it stays out of the potential
        until a pursuit outcome has actually moved its posterior off the prior."""
        return self.channel == "A" or (self.alpha + self.beta) > 2.0

    def describe(self) -> str:
        arrow = "+" if self.direction > 0 else ("-" if self.direction < 0 else "?")
        return (f"{arrow}{self.ro.describe()} [{self.channel or '-'}] "
                f"p={self.post_mean():.2f} ({self.reason})")


class GoalModel:
    """COMPONENT 5.12 -- the order over world-states.

    Owns NO history (anti-responsibility 9): it reads the Timeline and the
    StateGraph, and everything it stores is a derived view that a replay can
    reproduce exactly. That is the same rule that keeps StateGraph exact under an
    online-learned mask, and it exists for the same reason -- a second, private
    copy of the record could disagree with the thing every other model is
    certified against."""

    MAX_FEATURES = 24               # live pool cap; dedupe keeps it reachable
    CANDIDATE_EVERY = 25            # steps between candidate proposals (backfill cost)
    COLORS_MAX = 6                  # candidate colours per proposal round
    CELLS_MAX = 4                   # candidate counter cells per proposal round
    REPLAY_MAX = 2000               # backtest window (budget is 1500 actions)

    # -- the clock separator (5.1). A real progress coordinate advances under SOME
    #    actions and not others; a clock advances under all of them.
    CLOCK_MIN_PER_ACT = 4           # samples before an action's rate counts
    CLOCK_QUIET = 0.25              # SOME action must move it this rarely
    CLOCK_MIN_MOVES = 2             # a constant readout carries no order at all
    # Fallback when the action partition cannot separate anything (click-only
    # games: every action is ACTION6, so no second action is ever well sampled).
    # Same test with the partition collapsed: a per-action clock moves at rate 1.
    CLOCK_MIN_TOTAL = 12
    CLOCK_QUIET_GLOBAL = 0.35

    # -- evidence thresholds
    EXTREME_HIGH = 0.9              # rank within the record's value distribution
    EXTREME_LOW = 0.1
    EXTREME_MARGIN = 0.25           # ... and that much beyond the BASE rate
    MIN_TERMINALS = 2               # deaths before terminal-extremal may fire
    B_MIN_MOVES = 3                 # moves before monotonicity is evidence
    REFUTE_MIN = 2                  # arrivals at the goal extreme with no score

    # -- pursuit
    PURSUIT_FLOOR = 0.35            # sampled credibility below this -> no target
    RETIRE_FLOOR = 0.15             # posterior mean below this -> retired forever
    BUDGET_MIN = 3                  # a weak hypothesis buys a 3-action probe
    BUDGET_MAX = 40
    STALE_SCALE = 120.0             # actions since the last confirmed gain

    # Channel A outranks every Channel B feature absolutely (anti-responsibility
    # 6). Implemented lexicographically: B can only break a tie in A.
    LEX_EPS = 1e-3

    def __init__(self, timeline: Optional[Timeline] = None,
                 sgraph_fn: Optional[Callable] = None,
                 avatar_fn: Optional[Callable] = None,
                 seed: int = 0):
        self.timeline = timeline
        self._sgraph_fn = sgraph_fn
        self._avatar_fn = avatar_fn
        # PRIVATE rng. Thompson sampling must not consume the global stream --
        # shipping this component would otherwise desync every seeded run and the
        # "byte-identical with an empty pool" floor would be untestable.
        self._rng = random.Random(seed)

        self._feat: Dict[tuple, Feature] = {}
        self._vp: Dict[tuple, List[float]] = {}     # value at entry.prev
        self._vn: Dict[tuple, List[float]] = {}     # value at entry.nxt
        self._range: Dict[tuple, List[float]] = {}  # per-level [lo, hi] for normalise
        self._retired: set = set()                  # never resurrected
        self._capped: set = set()                   # lost the pool cap this level
        self._n_seen = 0                            # timeline entries vectorised
        self._pool_ver = 0
        self._cert_at = (-1, -1)
        self._stcache: Dict[int, GridStats] = {}
        self._level_start = 0                       # timeline index of this level
        self._last_propose = -10 ** 9
        self._steps = 0
        self._gain_at = 0                           # step of the last confirmed gain
        # (key, predicate, started_step, directed) -- `directed` is the alpha
        # warrant: True only once the hypothesis actually changed a decision.
        self._pursuit: Optional[tuple] = None
        # Pure diagnostics (never read by any decision). The first measurement of
        # this component reported A=0 / retired=0 and it was NOT obvious from the
        # outside whether the loop was closing at all -- these are what make the
        # difference between "pursued and lost" and "never pursued" visible.
        self._tally = {"armed": 0, "directed": 0, "alpha": 0, "beta": 0,
                       "expired": 0}
        self._wins: List[np.ndarray] = []

    # ---------------- write path ----------------

    def observe(self, prev, action, nxt, reward=0.0, terminal=False, level=0):
        """One REAL transition. The only input that may move credibility.

        The caller must already have appended it to the Timeline -- GoalModel
        reads the record, it does not write it. Imagined transitions never reach
        here (anti-responsibility 4): a world-model rollout that 'confirms' a goal
        confirms nothing."""
        self._steps += 1
        if reward > 0:
            self._gain_at = self._steps
        self._settle_pursuit(nxt, reward)
        self._extend()
        if self._steps - self._last_propose >= self.CANDIDATE_EVERY:
            self._last_propose = self._steps
            self._propose(nxt)

    def note_level_complete(self, frame, level=0):
        """The winning frame: a target exemplar, and the strongest evidence any
        game ever emits."""
        if frame is not None:
            try:
                self._wins.append(np.array(frame, dtype=np.int32))
            except Exception:
                pass
            if len(self._wins) > 8:
                self._wins.pop(0)
        self._gain_at = self._steps

    def new_level(self):
        """Two-tier lifecycle: RULES survive, LAYOUT does not.

        Features keyed on colours (and their credibilities) persist -- they are
        the game's rules and are the only object in the agent that can cross a
        level boundary, which RHAE's denominator makes mandatory. Positional
        readouts (`cell`, `agree`) and the per-level value ranges are dropped:
        those coordinates belong to the old map."""
        for k in [k for k, f in self._feat.items() if f.ro.positional()]:
            self._drop(k)
        self._range.clear()
        self._capped.clear()        # new layout -> these bindings are worth re-asking
        self._pursuit = None
        self._level_start = self._n_seen
        self._last_propose = -10 ** 9
        self._pool_ver += 1

    # ---------------- certification (UNSCORED, idempotent) ----------------

    def certify(self):
        """Backtest the pool against the Timeline: prune, dedupe, re-rank.

        Free under RHAE -- reasoning costs no actions -- so it may be called as
        often as the caller likes. Idempotent by construction: with no new
        Timeline entries and no pool change it does nothing at all."""
        self._extend()
        stamp = (self._n_seen, self._pool_ver)
        if stamp == self._cert_at:
            return
        self._cert_at = stamp
        for f in self._feat.values():
            self._judge(f)
        self._dedupe()
        self._cap()

    # ---------------- read path ----------------

    def potential(self, grid) -> Tuple[float, float]:
        """(phi, credibility). Potential-based shaping ONLY -- it re-orders
        choices and never alters the set of choices."""
        live = [f for f in self._feat.values()
                if f.direction and f.channel and f.steers()]
        if not live or grid is None:
            return 0.0, 0.0
        st = self._stats_for(grid)
        pa = pb = wa = wb = 0.0
        cred_a = cred_b = 0.0
        for f in live:
            v = f.ro.value(st)
            if v != v:
                continue
            x = self._normalise(f, v)
            w = f.post_mean()
            if f.channel == "A":
                pa += w * f.direction * x
                wa += w
                cred_a = max(cred_a, w)
            else:
                pb += w * f.direction * x
                wb += w
                cred_b = max(cred_b, w)
        phi = (pa / wa if wa else 0.0)
        phi += self.LEX_EPS * (pb / wb if wb else 0.0)
        return phi, (cred_a if wa else cred_b)

    def target(self, grid) -> Optional[GoalPredicate]:
        """A predicate to pursue, or None whenever nothing is adoptable.

        Selection is THOMPSON SAMPLING, never argmax (AB-MCTS-A): a hypothesis
        that keeps failing loses budget share smoothly with no hand-tuned
        threshold to mis-set, and a correct hypothesis that was merely unlucky is
        never permanently discarded.

        NOTE the deliberate asymmetry with `potential()`: this filter does NOT
        require `steers()`. Pursuit is how a hypothesis EARNS evidence; the
        potential is where earned belief is SPENT. Gating both on `steers()`
        would deadlock Channel B forever -- it could never be pursued, so its
        posterior could never move, so it could never qualify to steer."""
        self.certify()
        live = [f for f in self._feat.values() if f.direction and f.channel]
        if not live or grid is None:
            return None
        best, best_s = None, -1.0
        for f in live:
            s = self._rng.betavariate(max(1e-3, f.alpha), max(1e-3, f.beta))
            if f.channel == "A":
                s = 1.0 + s          # A outranks every B sample absolutely
            if s > best_s:
                best_s, best = s, f
        if best is None or (best_s if best.channel != "A" else best_s - 1.0) < self.PURSUIT_FLOOR:
            return None
        st = self._stats_for(grid)
        cur = best.ro.value(st)
        if cur != cur:
            return None
        thr = cur + best.direction               # readouts are integer-valued
        if best.direction < 0 and best.ro.has_floor():
            thr = max(0.0, thr)
            if thr >= cur:
                return None                      # already at the floor
        pred = GoalPredicate(best.ro, ">=" if best.direction > 0 else "<=",
                             thr, best.key)
        self._pursuit = (best.key, pred, self._steps, False)
        self._tally["armed"] += 1
        return pred

    def pursuing(self) -> bool:
        """Is a hypothesis currently on the hook? The caller arms a new one only
        when this is False, so an armed predicate always gets its budget window
        before the next is sampled."""
        return self._pursuit is not None

    def note_directed(self) -> bool:
        """The live hypothesis actually CHANGED what the agent did.

        This is the warrant `_settle_pursuit` requires before awarding alpha, and
        the asymmetry is deliberate. A predicate that was merely ARMED and then
        happened to be true when a score event landed has explained nothing --
        the reward came from whatever the agent was doing anyway, and crediting
        it would confirm decoys wholesale. Refutation needs no such warrant: a
        predicate that came true and paid nothing is refuted no matter who made
        it true. That is Section 5 guard 2 made literal -- refutation is free,
        adoption is expensive.

        Credit goes to the SAMPLED feature even though the order that redirected
        the agent is the pool's aggregate Phi. That is the standard Thompson
        rule: the sampled arm is the arm that acted."""
        if self._pursuit is None:
            return False
        k, p, s, was = self._pursuit
        self._pursuit = (k, p, s, True)
        if not was:
            self._tally["directed"] += 1
        return True

    def steering(self) -> bool:
        """Public form of `_active`: has ANY feature earned the right to order
        world-states? False is the correctness floor wherever it is asked -- the
        consumer then keeps its own validated order, unchanged."""
        return self._active()

    def rank_frontier(self, candidates, grid_of: Optional[Callable] = None):
        """THE SAFE CONSUMER: same set, new order.

        It may permute the untested frontier and may NEVER remove a pair from it
        (guard 1). set(out) == set(in) and len(out) == len(in) are harness
        assertions, not conventions. Candidates whose grid is unknown keep their
        relative order and go last -- an unscorable pair is not a demoted pair."""
        items = list(candidates)
        if grid_of is None or not self._active() or len(items) < 2:
            return items
        phis = []
        for it in items:
            try:
                g = grid_of(it)
            except Exception:
                g = None
            phis.append(self.potential(g)[0] if g is not None else None)
        order = sorted(range(len(items)),
                       key=lambda i: (0 if phis[i] is not None else 1,
                                      -(phis[i] or 0.0), i))
        return [items[i] for i in order]

    def frontier_ranker(self) -> Optional[Callable]:
        """A ranker for StateGraph.nearest_untested, or None when the pool has no
        opinion. None is not a fallback -- it is the correctness floor: the graph
        then takes its original early-return path and the agent is byte-identical
        to the one without this component."""
        if not self._active():
            return None
        sg = self._sgraph_fn() if self._sgraph_fn else None
        if sg is None:
            return None
        # `sg.grids` is REPLACED wholesale on a HUD-mask rebuild, so the lookup
        # has to read the attribute at call time. Binding the dict here would
        # leave the ranker scoring a hash table that no longer keys the graph.
        return lambda cands: self.rank_frontier(
            cands, grid_of=lambda c: sg.grids.get(c[0]))

    def pursuit_budget(self) -> int:
        """Max scored actions before re-certification is required.

        Scales with credibility and with actions since the last confirmed gain,
        and with NOTHING else -- the human baseline is an offline eval artifact
        and must never appear in this expression."""
        cred = max((f.post_mean() for f in self._feat.values()
                    if f.direction and f.channel), default=0.0)
        stale = max(0, self._steps - self._gain_at)
        decay = 1.0 / (1.0 + stale / self.STALE_SCALE)
        return int(max(self.BUDGET_MIN,
                       min(self.BUDGET_MAX, round(self.BUDGET_MAX * cred * decay))))

    def explain(self) -> str:
        """A goal model that cannot say what it believes cannot be debugged."""
        self.certify()
        live = sorted((f for f in self._feat.values() if f.direction and f.channel),
                      key=lambda f: (f.channel != "A", -f.post_mean(), f.ro.dl))
        if not live:
            return (f"no adoptable goal (pool={len(self._feat)}, "
                    f"retired={len(self._retired)})")
        return " | ".join(f.describe() for f in live[:3])

    # ---------------- internals ----------------

    def _active(self) -> bool:
        self.certify()
        return any(f.direction and f.channel and f.steers()
                   and f.post_mean() >= self.PURSUIT_FLOOR
                   for f in self._feat.values())

    def _stats_for(self, g) -> Optional[GridStats]:
        if g is None:
            return None
        k = id(g)
        st = self._stcache.get(k)
        if st is None or st.g is not g:
            if len(self._stcache) > 8:
                self._stcache.clear()
            st = GridStats(g)
            self._stcache[k] = st
        return st

    def _entries(self):
        tl = self.timeline
        return tl.entries if tl is not None else []

    def _drop(self, key):
        self._feat.pop(key, None)
        self._vp.pop(key, None)
        self._vn.pop(key, None)
        self._range.pop(key, None)

    def _extend(self):
        """Vectorise any Timeline entries not yet seen, for every live feature."""
        ent = self._entries()
        if self._n_seen >= len(ent):
            return
        keys = list(self._feat)
        for i in range(self._n_seen, len(ent)):
            e = ent[i]
            sp, sn = self._stats_for(e.prev), self._stats_for(e.nxt)
            for k in keys:
                ro = self._feat[k].ro
                a, b = ro.value(sp), ro.value(sn)
                self._vp[k].append(a)
                self._vn[k].append(b)
                self._note_range(k, a)
                self._note_range(k, b)
        self._n_seen = len(ent)

    def _backfill(self, keys):
        """Replay the record for NEWLY added features only.

        This is the operation the invariant rests on: every credibility value is
        reproducible by replaying the Timeline, so a feature that arrives late is
        judged on the whole history rather than on the part it happened to see."""
        ent = self._entries()
        lo = max(0, len(ent) - self.REPLAY_MAX)
        for k in keys:
            self._vp[k] = [_NAN] * lo
            self._vn[k] = [_NAN] * lo
        for i in range(lo, len(ent)):
            e = ent[i]
            sp, sn = self._stats_for(e.prev), self._stats_for(e.nxt)
            for k in keys:
                ro = self._feat[k].ro
                a, b = ro.value(sp), ro.value(sn)
                self._vp[k].append(a)
                self._vn[k].append(b)
                if i >= self._level_start:
                    self._note_range(k, a)
                    self._note_range(k, b)
        self._n_seen = len(ent)

    def _note_range(self, k, v):
        if v != v:
            return
        r = self._range.get(k)
        if r is None:
            self._range[k] = [v, v]
        else:
            if v < r[0]:
                r[0] = v
            if v > r[1]:
                r[1] = v

    def _normalise(self, f: Feature, v: float) -> float:
        r = self._range.get(f.key)
        lo, hi = (r[0], r[1]) if r else (v, v)
        if f.ro.has_floor():
            lo = min(lo, 0.0)
        if hi <= lo:
            return 0.5
        return float(min(1.0, max(0.0, (v - lo) / (hi - lo))))

    # -- candidate generation (bindings are discovered here, per level) --

    def _add(self, ro: Readout, new: list):
        k = ro.key
        if k in self._feat or k in self._retired or k in self._capped:
            return
        if len(self._feat) + len(new) >= self.MAX_FEATURES:
            return
        self._feat[k] = Feature(ro)
        new.append(k)

    def _propose(self, grid):
        """Generate candidate readouts from the live frame and the StateGraph.

        The vocabulary is fixed (anti-responsibility 7); what is discovered here
        is only which ARGUMENTS are worth binding it to on this board.

        Stops once the pool is full. Every new feature costs a full-history
        replay to judge it on the same record as everything else, so a pool that
        proposed, capped and re-proposed the same readout every 25 steps would
        pay that replay forever and learn nothing -- `_capped` closes that loop
        until the next level makes the bindings worth reconsidering."""
        self._extend()
        if grid is None or len(self._feat) >= self.MAX_FEATURES:
            return
        new: List[tuple] = []
        st = self._stats_for(grid)
        cn = st.counts()
        present = [c for c in range(len(cn)) if cn[c] > 0]
        if not present:
            return
        bg = max(present, key=lambda c: cn[c])
        cand = sorted((c for c in present if c != bg), key=lambda c: cn[c])
        cand = cand[:self.COLORS_MAX]
        av = None
        if self._avatar_fn is not None:
            try:
                av = self._avatar_fn()
            except Exception:
                av = None
        for c in cand:
            self._add(Readout("count", (c,)), new)
            self._add(Readout("exists", (c,)), new)
            if cn[c] <= GridStats.CC_MAX_CELLS:
                self._add(Readout("regions", (c,)), new)
                areas = st.cc(c)
                if areas:
                    modal = max(set(areas), key=areas.count)
                    self._add(Readout("objects", (c, modal)), new)
            if av is not None and c != av:
                self._add(Readout("dist", (av, c)), new)
                self._add(Readout("contact", (av, c)), new)
                self._add(Readout("align", (av, c)), new)
        # Counters and score displays. `_cellmask` masks only HIGH-rate cells, so
        # a cell that changes only on EVENTS (keys collected, targets remaining)
        # is never masked -- it is already inside the state hash, the graph simply
        # treats counter=2 and counter=3 as two unrelated nodes. The information
        # the game volunteers about its own objective is being received and
        # discarded; imposing an ORDER on it is this component's whole job.
        for (r, c) in self._counter_cells():
            self._add(Readout("cell", (r, c)), new)
        # `agree` -- the multi-panel "make this look like that" hypothesis. Added
        # here but deliberately the least certain member of the vocabulary; it
        # earns its place through the same certification path as everything else.
        ps = st.panels()
        if len(ps) >= 2:
            self._add(Readout("agree", (0, 1)), new)
            if len(ps) >= 3:
                self._add(Readout("agree", (0, 2)), new)
        if new:
            self._backfill(new)
            self._pool_ver += 1

    def _counter_cells(self) -> List[Tuple[int, int]]:
        """Cells that change RARELY but non-trivially, plus cells the engine
        itself implicated at a score or terminal moment.

        `_out_chg` was collected for the sacred-cell guard, which reads it
        defensively ('which cells may I never mask'). Read positively it answers
        'which cells decide the score' -- a candidate generator that costs nothing
        extra to collect and is already validated by test_stategraph.py."""
        sg = self._sgraph_fn() if self._sgraph_fn else None
        if sg is None or getattr(sg, "_change", None) is None:
            return []
        out: List[Tuple[int, int]] = []
        chg, n = sg._change, max(1, sg._n)
        rate = chg / n
        low = (chg >= 2) & (rate < 0.15)
        oc = getattr(sg, "_out_chg", None)
        if oc is not None and oc.shape == chg.shape:
            low |= (oc >= 1) & (rate < 0.5)
        rs, cs = np.nonzero(low)
        if len(rs) == 0:
            return []
        order = np.argsort(-chg[rs, cs])[:self.CELLS_MAX]
        for i in order:
            out.append((int(rs[i]), int(cs[i])))
        return out

    # -- certification internals --

    def _judge(self, f: Feature):
        """Assign the feature its evidence channel and direction, from the record
        alone. Channel A (score/terminal confirmed) outranks Channel B
        (structural) absolutely; with neither, the feature has no opinion and
        contributes nothing."""
        key = f.key
        vp, vn = self._vp.get(key), self._vn.get(key)
        ent = self._entries()
        if not vp or len(vp) != len(ent):
            f.channel, f.direction, f.reason = "", 0, "unvectorised"
            return
        if not self._clock_ok(key):
            f.channel, f.direction, f.reason = "", 0, "clock"
            return
        d = self._confirmed(key)
        if d:
            if self._refuted(key, d):
                f.channel, f.direction, f.reason = "", 0, "refuted"
                return
            if f.channel != "A":
                f.channel, f.reason = "A", "confirmed"
            f.direction = d
            return
        d, why = self._structural(key)
        if d and not self._refuted(key, d):
            if f.channel != "B":
                f.channel, f.reason = "B", why
            f.direction = d
            return
        f.channel, f.direction = "", 0
        f.reason = f.reason or "no evidence"

    def _sample_idx(self):
        ent = self._entries()
        lo = max(0, len(ent) - self.REPLAY_MAX)
        return range(lo, len(ent))

    def _clock_ok(self, key) -> bool:
        """THE 5.1 SEPARATOR. A real progress coordinate advances under SOME
        actions and not others; a clock advances under all of them. A step
        counter is monotone, irreversible AND bounded, so it passes every
        structural test at once -- this is the only test it cannot pass.

        Deliberately a VETO and not an adoption rule, so the question it asks is
        the narrow one: IS THERE AN ACTION UNDER WHICH THIS DOES NOT ADVANCE?
        Demanding that some action also advance it OFTEN would be the wrong test
        -- a sparse positional readout like `contact(avatar, goal)` moves on a
        handful of transitions in a whole level and is not thereby a clock.
        Adoption is Channel A's and Channel B's job; this only removes the one
        candidate class that would otherwise pass every one of their tests."""
        ent = self._entries()
        vp, vn = self._vp[key], self._vn[key]
        acts: Dict[str, List[int]] = {}
        moves = tot = 0
        for i in self._sample_idx():
            a, b = vp[i], vn[i]
            if a != a or b != b:
                continue
            s = acts.setdefault(base_action(ent[i].action), [0, 0])
            s[1] += 1
            tot += 1
            if a != b:
                s[0] += 1
                moves += 1
        if moves < self.CLOCK_MIN_MOVES:
            return False
        sampled = [s for s in acts.values() if s[1] >= self.CLOCK_MIN_PER_ACT]
        if len(sampled) >= 2:
            return min(m / n for m, n in sampled) <= self.CLOCK_QUIET
        # Click-only games never well-sample a second base action (every action is
        # ACTION6). Same test with the partition collapsed: a per-action clock
        # moves on every transition, so a global move rate this low cannot be one.
        if tot < self.CLOCK_MIN_TOTAL:
            return False
        return (moves / tot) <= self.CLOCK_QUIET_GLOBAL

    def _extremity(self, key, idxs, use_prev=True):
        """(marked, base): where the MARKED frames sit inside the record's value
        distribution, and where an unremarkable frame sits. 0.0 = the minimum,
        1.0 = the maximum.

        The BASE is the whole point, and it is the same disproportionality rule
        that makes sacred cells safe. `exists(goal_colour)` is 1.0 on every frame
        the agent died on -- and on 99% of every other frame too, because the goal
        is nearly always on screen. Judged against a nominal 0.5 that reads as
        'systematically extreme at death' and fills the pool with hypotheses that
        are really just base rates; a first 25-game run with the base term missing
        adopted 11-19 features per game, every one of them at the untested prior.
        Judged against its own base rate a pure base-rate readout scores margin 0
        by construction -- exactly as a pure clock scores sacred-margin 0."""
        vp, vn = self._vp[key], self._vn[key]
        pool = [v for i in self._sample_idx() for v in (vp[i], vn[i]) if v == v]
        if len(pool) < 4:
            return None
        lo, hi = min(pool), max(pool)
        if hi <= lo:
            return None
        span = hi - lo
        src = vp if use_prev else vn
        marks = [src[i] for i in idxs if src[i] == src[i]]
        if not marks:
            return None
        return (sum((m - lo) / span for m in marks) / len(marks),
                sum((v - lo) / span for v in pool) / len(pool))

    def _verdict(self, ex) -> int:
        """Turn (marked, base) into a direction, or 0 when the evidence is only a
        base rate. High-at-the-outcome means the outcome sits at the TOP, so for a
        death that is a NEGATIVE coordinate and for a win a positive one -- the
        caller supplies the sign."""
        if ex is None:
            return 0
        m, b = ex
        if m >= self.EXTREME_HIGH and (m - b) >= self.EXTREME_MARGIN:
            return 1
        if m <= self.EXTREME_LOW and (b - m) >= self.EXTREME_MARGIN:
            return -1
        return 0

    def _confirmed(self, key) -> int:
        """CHANNEL A. Score events and winning frames.

        Only the PRE-outcome frame is read. On a level-up the engine reports the
        score on the frame AFTER the whole layout was swapped, so `nxt` there is a
        foreign board and its readout value is a layout artifact, not progress.
        The informative side is the state the win was achieved FROM."""
        ent = self._entries()
        idx = [i for i in self._sample_idx() if ent[i].reward > 0]
        if not idx:
            return 0
        # A win sitting at the TOP of the distribution means higher is better.
        return self._verdict(self._extremity(key, idx, use_prev=True))

    def _structural(self, key) -> Tuple[int, str]:
        """CHANNEL B. Available before any win -- the only regime 22 of 25 games
        have ever been in. Every branch is already gated by the clock separator."""
        ent = self._entries()
        vp, vn = self._vp[key], self._vn[key]
        up = dn = 0
        for i in self._sample_idx():
            a, b = vp[i], vn[i]
            if a != a or b != b or a == b:
                continue
            if b > a:
                up += 1
            else:
                dn += 1
        # Terminal-extremal: the first evidence any game ever emits. Deaths
        # precede wins everywhere, and a readout that is systematically extreme on
        # the frames we died on is a NEGATIVE coordinate.
        term = [i for i in self._sample_idx() if ent[i].terminal]
        if len(term) >= self.MIN_TERMINALS:
            # Sign flip: a death sitting at the TOP means lower is better.
            v = self._verdict(self._extremity(key, term, use_prev=True))
            if v:
                return -v, "terminal-extremal"
        # Monotone-irreversible: games are built so progress does not undo itself
        # -- doors stay open, collected things stay collected. The reverse
        # transition appearing NOWHERE in the record is the exact evidence.
        if up >= self.B_MIN_MOVES and dn == 0:
            return 1, "monotone"
        if dn >= self.B_MIN_MOVES and up == 0:
            return -1, "monotone"
        return 0, ""

    def _refuted(self, key, direction) -> bool:
        """REFUTATION IS FREE; ADOPTION IS EXPENSIVE.

        If the record already shows the readout sitting at the extreme this
        direction points to, repeatedly, and no score event ever happened there,
        then 'go that way' is refuted -- and it was refuted at ZERO action cost,
        because those actions were already spent. This is what stops the agent
        walking into a decoy fifty times by a different route each time, which is
        the failure the livelock breaker structurally cannot see.

        Gated on having seen at least one score event: you cannot falsify 'this
        wins' before you know what winning looks like, and 22 of 25 games are in
        exactly that regime."""
        ent = self._entries()
        idxs = list(self._sample_idx())
        if not any(ent[i].reward > 0 for i in idxs):
            return False
        vn = self._vn[key]
        vals = [vn[i] for i in idxs if vn[i] == vn[i]]
        if not vals:
            return False
        goal = max(vals) if direction > 0 else min(vals)
        arrivals = [i for i in idxs if vn[i] == goal]
        if len(arrivals) < self.REFUTE_MIN:
            return False
        return not any(ent[i].reward > 0 for i in arrivals)

    def _dedupe(self):
        """Observational equivalence + MDL tie-break.

        Two readouts with identical value vectors over the entire record ARE the
        same hypothesis, however differently they are written; keeping both would
        double their weight in the potential and double the cost of every future
        backtest. The shorter description survives (bottom-up MDL synthesis)."""
        seen: Dict[Any, tuple] = {}
        for k in sorted(self._feat, key=lambda x: (str(x[0]), x[1])):
            v = self._vn.get(k)
            if not v:
                continue
            sig = hash(np.asarray(v, dtype=np.float64).tobytes())
            prev = seen.get(sig)
            if prev is None:
                seen[sig] = (k, self._feat[k].ro.dl)
                continue
            # The loser joins `_capped`, not `_retired`: it is not WRONG, it is
            # the same hypothesis written at greater length. Without this the
            # next proposal round re-adds it, the round after that de-dupes it
            # again, and the pool oscillates -- paying a full-history replay
            # every 25 steps and double-counting the hypothesis in Phi in
            # between. `new_level` clears it, because a new layout can make two
            # readouts that were indistinguishable here distinguishable there.
            pk, pdl = prev
            if self._feat[k].ro.dl < pdl:
                self._capped.add(pk)
                self._drop(pk)
                seen[sig] = (k, self._feat[k].ro.dl)
            else:
                self._capped.add(k)
                self._drop(k)

    def _cap(self):
        if len(self._feat) <= self.MAX_FEATURES:
            return
        order = sorted(self._feat.values(),
                       key=lambda f: (f.channel != "A", not f.direction,
                                      -f.post_mean(), f.ro.dl))
        for f in order[self.MAX_FEATURES:]:
            self._capped.add(f.key)
            self._drop(f.key)

    # -- falsification --

    def _settle_pursuit(self, nxt, reward):
        """FALSIFICATION-FIRST. A hypothesis must make a risky prediction to
        receive budget, and satisfying the predicate WITHOUT the predicted score
        event counts as a failure. `_llm_goal` already did the special case
        ('arrived, no level-up -> drop the hint'); generalising it to every
        hypothesis is what stops a plausible decoy absorbing a whole level."""
        if self._pursuit is None:
            return
        key, pred, started, directed = self._pursuit
        f = self._feat.get(key)
        if f is None:
            self._pursuit = None
            return
        if reward > 0:
            # ALPHA REQUIRES A WARRANT (`note_directed`). An armed-but-unpursued
            # predicate that is merely true when a score event lands has explained
            # nothing, and crediting it would let every candidate ride the agent's
            # own luck into the pool. Undirected -> the pursuit closes with no
            # verdict either way, exactly like a budget expiry.
            if directed:
                f.alpha += 1.0
                self._tally["alpha"] += 1
            self._pursuit = None
            return
        if pred.holds(nxt):
            f.beta += 1.0
            self._tally["beta"] += 1
            self._pursuit = None
            if f.post_mean() < self.RETIRE_FLOOR:
                self._retired.add(key)      # never resurrected
                self._drop(key)
                self._pool_ver += 1
            return
        if self._steps - started > self.pursuit_budget():
            self._pursuit = None            # ran out of budget; no verdict either way
            self._tally["expired"] += 1


# ==========================================
# COMPONENT 5.13: REPLAY CACHE + EFFICIENCY GOVERNOR
# ==========================================

class ReplayCache:
    """Never solve the same level twice. (system_v2_architecture.md 3.5)

    Why this is worth more than it looks
    ------------------------------------
    The scorer charges level k every action since level k-1 was FIRST reached.
    So when the agent banks level 1, dies on level 2 and re-walks level 1 from
    scratch, the re-walk is billed to LEVEL 2 -- and RHAE squares the ratio.
    Measured on this agent, deaths are routine and re-walks are re-discovered
    by the same stochastic planners that found the route the first time, at
    similar cost. Every one of those actions is paid for twice and scores once.

    Three parts, all unscored except the replay itself:

    1. RECORD. Keep (grid, action) for every step since the last level-up.
    2. GOVERN. On a level-up, compress the segment before storing it: if the
       same masked state appears twice, everything between the two visits was
       a cycle in the observed world and is dropped. This is a shortest-path
       extraction over a trajectory we already own -- free, and typically the
       difference between the route we stumbled into and the route we needed.
    3. REPLAY, VERIFIED. After a death, re-issue the stored route one action at
       a time, and before each one check that the CURRENT frame still matches
       the frame that action was recorded from, HUD-masked. First mismatch and
       the whole replay is abandoned in silence, control returns to the
       planners, and the route is retired.

    Verification is the whole safety argument. Games with hidden state (tu93)
    or randomised layouts will diverge, and divergence costs at most one action
    before the cache steps aside -- so the downside is bounded by construction
    while the upside is the entire re-walk. Comparison is HUD-masked with the
    mask that exists AT REPLAY TIME, applied to both sides, because the mask is
    learned online and a stored hash would go stale the moment it improved.
    """

    MAX_SEGMENT = 1200          # a route longer than this is not worth replaying
    MIN_REPLAY = 4              # below this the re-walk is cheaper than the risk

    def __init__(self):
        self.routes: Dict[int, list] = {}      # level -> [(grid, action_str), ...]
        self._seg: list = []                   # steps since the last level-up
        self._replay: list = []                # the route currently being re-issued
        self._i = 0
        self._level = -1                       # level being re-solved; -1 = frontier
        self.retired: set = set()              # routes that diverged; never retried
        # Counters -- a component that cannot report what it did cannot be A/B'd.
        self.stats = {"stored": 0, "compressed_out": 0, "replayed": 0,
                      "diverged": 0, "completed": 0, "frontier": 0}

    # ---- record -------------------------------------------------------
    def note_action(self, grid: np.ndarray, action_str: str) -> None:
        """One scored step. Stored as int8: colours are 0..15, and a 1500-step
        segment of int32 grids is 24MB for no reason.

        Replayed actions are recorded too. They are part of the route to
        wherever this life ends up, and omitting them would bank a truncated
        route that starts in the middle of the level.
        """
        if len(self._seg) > self.MAX_SEGMENT:
            return
        try:
            self._seg.append((np.asarray(grid, dtype=np.int8).copy(), action_str))
        except (TypeError, ValueError):
            self._seg = []                     # unrecordable frame -> no route

    def note_level_up(self, level: int) -> None:
        """Bank the route to `level`, compressed. Called on a score RISE only."""
        seg, self._seg = self._seg, []
        if seg and level not in self.routes and level not in self.retired:
            squeezed = self._compress(seg)
            self.routes[level] = squeezed
            self.stats["stored"] += 1
            self.stats["compressed_out"] += len(seg) - len(squeezed)
        if self._level == level:               # a replay reached its target
            self.stats["completed"] += 1
        # After a wipe we may hold routes for levels ahead of us; chain to the
        # next one. On an ordinary level-up there is no such route and this is
        # a no-op.
        self._arm(level + 1)

    def note_death(self, level_now: int) -> None:
        """GAME_OVER. arcengine level_reset()s, so the LEVEL restarts and every
        action spent reaching the point of death is spent again. That re-walk
        is what this replays -- minus the action that did the killing.
        """
        seg = self._compress(self._seg)[:-1]
        self._seg = []
        self._replay, self._i, self._level = [], 0, -1
        if len(seg) >= self.MIN_REPLAY:
            self._replay = seg
            self.stats["frontier"] += 1

    def note_wipe(self, level_now: int) -> None:
        """levels_completed DROPPED -- a full_reset wiped the banked levels, and
        every one of them has to be walked again. Those walks are the routes we
        already own."""
        self._seg = []
        self._arm(level_now + 1)

    def _arm(self, level: int) -> None:
        self._replay, self._i, self._level = [], 0, -1
        route = self.routes.get(level)
        if route and level not in self.retired:
            self._replay, self._level = route, level

    # ---- govern -------------------------------------------------------
    @staticmethod
    def _compress(seg: list) -> list:
        """Drop every cycle: if a state recurs, the actions between the visits
        returned the world to where it already was, so they bought nothing.

        Compares raw here (not masked) on purpose: at record time the mask may
        still be half-learned, and dropping a step wrongly is the one error this
        class cannot recover from cheaply. Raw equality is the conservative
        choice -- it compresses less and never invents a shortcut. A live HUD
        simply means nothing compresses, which costs only the compression.
        """
        out: list = []
        seen: Dict[bytes, int] = {}
        for grid, act in seg:
            key = grid.tobytes() + bytes(str(grid.shape), "ascii")
            if key in seen:
                del out[seen[key]:]                       # excise the loop
                seen = {}
                for i, (g2, _a2) in enumerate(out):
                    seen[g2.tobytes() + bytes(str(g2.shape), "ascii")] = i
            seen[key] = len(out)
            out.append((grid, act))
        return out

    # ---- replay -------------------------------------------------------
    def replaying(self) -> bool:
        return bool(self._replay) and self._i < len(self._replay)

    def next_action(self, grid: np.ndarray, valid: set,
                    mask: Optional[np.ndarray] = None) -> Optional[str]:
        """The next recorded action, or None if the route no longer applies.

        A route that does not reproduce its own recorded state is a route to a
        world that no longer exists. Retiring it on the first mismatch is what
        keeps the worst case at one action.
        """
        if not self.replaying():
            return None
        want, act = self._replay[self._i]
        cur = np.asarray(grid, dtype=np.int8)
        if not masked_equal(cur, want, mask) or base_action(act) not in {
                base_action(v) for v in valid}:
            self.abandon("diverged")
            return None
        self._i += 1
        self.stats["replayed"] += 1
        return act

    def abandon(self, why: str = "diverged") -> None:
        """Retire a LEVEL route that diverged: the world it described is gone,
        and re-issuing it would pay the same wasted action every life. A
        frontier route (level -1) is not retired -- it is rebuilt from the next
        life anyway."""
        self.stats["diverged"] += 1
        if self._level >= 0:
            self.retired.add(self._level)
            self.routes.pop(self._level, None)
        self._replay, self._i, self._level = [], 0, -1

    def new_game(self) -> None:
        self.__init__()


# ==========================================
# COMPONENT 6: ACT & MAIN AGENT
# ==========================================

class MyAgent(Agent):
    MAX_ACTIONS = 1500

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        gid = getattr(self, "game_id", "local")
        # Kaggle: fresh time-based randomness per game. Local eval: a caller can pin
        # ARC_AGENT_SEED to make the exploration RNG reproducible (the game wrapper's
        # own --seed does NOT control the agent's clicks), so results are comparable
        # across code changes instead of flickering on luck.
        _seed_env = os.environ.get("ARC_AGENT_SEED")
        if _seed_env is not None:
            seed = int(_seed_env) + (hash(gid) % 1_000_000)
        else:
            seed = int(time.time() * 1e6) + (hash(gid) % 1_000_000)
        random.seed(seed); np.random.seed(seed % (2**32 - 1))
        if torch is not None:
            torch.manual_seed(seed)

        self.encoder = StateEncoder()
        self.llm = get_local_llm()
        self.memory = MemoryManager()
        # The HUD mask is learned by the graph (only it keeps the per-cell /
        # per-action statistics), and read by the world model through this
        # closure. Late-bound on purpose: `self.sgraph` does not exist yet here,
        # is REPLACED on a new level (see the reset path), and its mask is
        # rebuilt every MASK_CHECK_EVERY transitions -- a snapshot passed by
        # value would be stale within 20 steps.
        self.world_model = WorldModelManager(
            self.llm, self.encoder,
            mask_fn=lambda: self.sgraph.hud_mask()
            if getattr(self, "sgraph", None) is not None else None)
        self.counter = StateCounter()
        self.planner = MCTSPlanner(self.world_model, self.encoder, self.counter)
        self.rplanner = ExecPlanner()   # executable world model + unscored plan (COMPONENT 5.7)
        # Append-only ground truth (COMPONENT 5.10). Owned here, never cleared:
        # ExecPlanner certifies its movement theory against it before planning.
        self.timeline = Timeline()
        self.rplanner.timeline = self.timeline
        self.cplanner = ClickPlanner()
        self.sgraph = StateGraph()      # exact transition graph (COMPONENT 5.8)
        self.dead = DeadActionTracker()
        self.breaker = LivelockBreaker()  # scored-action circuit breaker (5.11)
        self.replay = ReplayCache()       # never solve the same level twice (5.13)
        # Local rewrite rules (COMPONENT 5.9). Persists across levels ON PURPOSE:
        # the rules are the game's physics, not the level's layout.
        self.pwm = PatchWorldModel()
        # The order over world-states (COMPONENT 5.12). Persists across levels ON
        # PURPOSE and more strongly than anything else here: RHAE's denominator
        # spans every level, so a learned win-condition is the only object that
        # can pay off on a layout the agent has never seen. Its RNG is private --
        # Thompson sampling must not perturb the shared exploration stream.
        self.goals = GoalModel(
            timeline=self.timeline,
            sgraph_fn=lambda: getattr(self, "sgraph", None),
            avatar_fn=lambda: self.rplanner.avatar_color,
            seed=seed % (2 ** 31 - 1))
        # The consumer link (Section 9 step 6). The planner READS this order to
        # break ties between target candidates and writes back exactly one thing
        # -- the pursuit warrant -- so ownership stays here and the planner keeps
        # working untouched if it is ever None.
        self.rplanner.goals = self.goals
        self.gate = ExploitGate()

        self.transitions: List[Transition] = []
        self.last_grid: Optional[np.ndarray] = None
        self.last_action_str: str = ""
        # Masked hash of the frame the last action was taken FROM: the ledger key
        # the breaker needs. Kept here rather than re-hashed later because a mask
        # rebuild between the two calls would key the spend under a stale hash.
        self._prev_hash: Optional[int] = None
        self.last_score = 0
        self.action_x = 0
        self.action_y = 0
        self.level_steps = 0
        # Monotone across levels -- level_steps restarts, so it cannot key a cache
        # (level 2 step 1 would read level 1 step 1's answer).
        self.steps_total = 0
        self._ge_memo: Optional[Tuple[int, Optional[str]]] = None
        self._esc_bfs_at = -ESCALATE_BFS_EVERY
        self.epsilon = EPSILON_START

        self.llm_calls_this_level = 0
        self.synth_inflight = False
        self.synth_started_at = 0.0
        self.synth_generation = 0
        self._last_enum_at = -1

        # LLM click-reasoning plan (click-only CSP games). A budgeted background
        # call fills an ordered queue of pixel targets; choose_action consumes one
        # per step. Empty / LLM-absent => the ClickPlanner coverage floor drives.
        self._click_plan: "deque[Tuple[int, int]]" = deque()
        self._click_inflight = False
        self._click_src_hash: Optional[int] = None
        self._last_click_plan_at = -10 ** 9
        # Schema Phase 2: budgeted background THEORIZER for movement games. A
        # completed proposal is adopted on the MAIN thread (the backtest mutates
        # certification state act() also uses), gated by the full-timeline
        # backtest in ExecPlanner.propose_theory.
        self._theory_proposal: Optional[dict] = None
        self._theory_inflight = False
        self._last_theorize_at = -10 ** 9
        self._win_recorded = False
        self._stats_reported = False       # LLM accept rate: print the end-of-game
                                           # line once, not on every is_done poll
        self.last_checkpoint = time.time()

        # ---- ROUTING INSTRUMENTATION (read-only) ----------------------------
        # `_route` names the ladder branch that produced the last action and
        # `_route_ms` what deciding it cost. Nothing below READS either field, so
        # tagging cannot change a decision -- they exist because "which planner is
        # responsible for this action, and was it worth its latency" was
        # unanswerable from the outside: choose_action is one function and the
        # branch identity dies with the call. Consumed by scratchpad/route_diag.py.
        self._route = ""
        self._route_ms = 0.0

        # Preload the 27B in the background so the first budgeted call is instant.
        if self.llm.available and not self.llm.is_ready():
            threading.Thread(target=self.llm.ensure_loaded, daemon=True).start()

    # ---- helpers ----
    @staticmethod
    def _epsilon_for(step: int) -> float:
        return max(EPSILON_FLOOR, EPSILON_START * (EPSILON_DECAY ** step))   # Issue 5

    def _reset_level_state(self):
        self.transitions.clear()
        self.last_grid = None
        self.last_action_str = ""
        self._prev_hash = None
        self.breaker.note_reset()
        self.llm_calls_this_level = 0
        self.synth_inflight = False
        self.synth_generation += 1        # orphan any in-flight synth thread
        self.level_steps = 0
        self.epsilon = EPSILON_START
        self._last_enum_at = -1
        self._win_recorded = False
        self._theory_proposal = None
        self._theory_inflight = False
        self._last_theorize_at = -10 ** 9
        self.world_model.reset_level()
        self.rplanner.reset()
        self.cplanner.reset()
        self._discard_click_plan()
        self.gate = ExploitGate()

    def _synth_watchdog(self):
        """Issue 6: a stalled LLM thread can't be killed, but it CAN be orphaned --
        clear the inflight flag and bump the generation so its completion is ignored
        for flag bookkeeping (a late candidate still verifies on fresh data)."""
        if self.synth_inflight and time.time() - self.synth_started_at > SYNTH_MAX_INFLIGHT_S:
            print("[synth] watchdog: abandoning stalled LLM synthesis thread")
            self.synth_generation += 1
            self.synth_inflight = False

    # ---- LLM click reasoning (click-only CSP games) ----
    _CLICK_PLAN_MIN_INTERVAL = 40   # steps between plan requests (throttle the budget)
    _CLICK_PLAN_MIN_OBS = 20        # coverage clicks before the LLM has data to reason on

    def _discard_click_plan(self):
        self._click_plan.clear()
        self._click_src_hash = None

    @staticmethod
    def _grid_to_text(grid: np.ndarray, max_side: int = 64) -> str:
        """Compact char-grid for the LLM: one char per cell, '.' for the modal
        background, 0-9/a-z for other colours. Pixel coords are preserved (no
        downscaling), so a coord the LLM returns maps straight to a click."""
        g = grid
        if g.shape[0] > max_side or g.shape[1] > max_side:
            g = g[:max_side, :max_side]
        vals, counts = np.unique(g, return_counts=True)
        bg = int(vals[counts.argmax()]) if len(vals) else 0
        alpha = "0123456789abcdefghijklmnopqrstuvwxyz"
        rows = []
        for row in g:
            rows.append("".join("." if int(v) == bg else alpha[int(v) % len(alpha)]
                                for v in row))
        return "\n".join(rows)

    def _maybe_request_click_plan(self, grid: np.ndarray):
        """Fire a budgeted background LLM call to plan clicks. No-op unless the LLM
        is loaded, budget remains, no plan/request is already pending, and enough
        board evidence exists. choose_action never blocks on this."""
        if (not self.llm.is_ready() or self._click_inflight or self._click_plan
                or self.llm_calls_this_level >= LLM_CALL_BUDGET_PER_LEVEL
                or self.level_steps - self._last_click_plan_at < self._CLICK_PLAN_MIN_INTERVAL
                or self.cplanner._total_clicks < self._CLICK_PLAN_MIN_OBS):
            return
        self._click_inflight = True
        self._last_click_plan_at = self.level_steps
        self.llm_calls_this_level += 1
        gen = self.synth_generation
        src_hash = int(hash((grid.shape, grid.tobytes())))
        board_text = self._grid_to_text(grid)
        obs = self.memory.observation_digest(self.transitions)
        reactive = ", ".join(f"({r},{c})" for r, c in
                             list(self.cplanner._reactive.values())[:12])
        hyp = self.memory.l2_hypothesis

        def _job():
            try:
                clicks = self.llm.plan_clicks(board_text, obs, reactive, hyp)
            except Exception as e:
                print(f"[click-plan] failed: {e}")
                clicks = []
            finally:
                # Only adopt if this request wasn't orphaned by a level-reset.
                if gen == self.synth_generation and clicks:
                    h, w = grid.shape
                    self._click_plan = deque(
                        (max(0, min(h - 1, int(r))), max(0, min(w - 1, int(c))))
                        for r, c in clicks)
                    self._click_src_hash = src_hash
                self._click_inflight = False
        threading.Thread(target=_job, daemon=True).start()

    # ---- Schema Phase 2: budgeted LLM theorizer (movement games) ----
    _THEORIZE_AFTER = 120       # scored actions on a level before the theorizer may fire
    _THEORIZE_INTERVAL = 120    # min steps between theorize requests (budget throttle)

    def _maybe_request_theory(self, grid: np.ndarray):
        """Fire a budgeted background THEORIZE call when the local loop is
        demonstrably stuck (many scored actions on this level, no level-up).
        The result is stored as a pending proposal; choose_action adopts it on
        the main thread via ExecPlanner.propose_theory, which backtests it
        against the full timeline. choose_action never blocks on this; with no
        LLM (local eval) this is a no-op, so the measured no-LLM core is
        untouched."""
        if (not self.llm.is_ready() or self._theory_inflight
                or self._theory_proposal is not None
                or self.llm_calls_this_level >= LLM_CALL_BUDGET_PER_LEVEL
                or self.level_steps < self._THEORIZE_AFTER
                or self.level_steps - self._last_theorize_at < self._THEORIZE_INTERVAL):
            return
        rp = self.rplanner
        if rp.avatar_color is None:
            return
        self._theory_inflight = True
        self._last_theorize_at = self.level_steps
        self.llm_calls_this_level += 1
        gen = self.synth_generation
        board_text = self._grid_to_text(grid)
        theory = (f"avatar_color={rp.avatar_color}, "
                  f"action_displacements={rp.action_disp}, "
                  f"obstacle_colors={sorted(rp.obstacle_colors)}, "
                  f"decoy_colors={sorted(rp.non_goal_colors)}, "
                  f"positional_walls={len(rp.walls)} cells, "
                  f"anomalous_cells={sorted(rp._anomalies)}")
        obs = self.memory.observation_digest(self.transitions)
        trouble = ("certification RED: a recorded transition contradicts the theory"
                   if not rp._cert_green else
                   f"{self.level_steps} actions on this level without a level-up; "
                   f"target pursuit is not finding the goal")

        def _job():
            try:
                res = self.llm.theorize(board_text, theory, obs, trouble)
            except Exception as e:
                print(f"[theorize] failed: {e}")
                res = None
            finally:
                # Only adopt if this request wasn't orphaned by a level change.
                if gen == self.synth_generation and res:
                    self._theory_proposal = res
                self._theory_inflight = False
        threading.Thread(target=_job, daemon=True).start()

    def _next_click_from_plan(self, grid: np.ndarray) -> Optional[str]:
        """Pop the next planned target. The plan is an ORDERED sequence reasoned
        about the board as it was when the request fired, so it is only valid if it
        STARTS from that same board. We validate the start ONCE: on the first
        consume, if the live board no longer hashes to the request-time snapshot
        (coverage mutated it while the LLM was generating), the ordered assumption is
        broken -> drop the whole plan and fall back to coverage. After the start is
        validated, each planned click legitimately mutates the board, so we stop
        re-checking. Lethal targets coverage already learned are skipped so a plan
        can never trip a GAME_OVER coverage would have avoided (score below floor)."""
        if not self._click_plan:
            return None
        if self._click_src_hash is not None:               # start not yet validated
            if int(hash((grid.shape, grid.tobytes()))) != self._click_src_hash:
                self._discard_click_plan()                 # stale start -> abandon
                return None
            self._click_src_hash = None                    # validated; don't re-check
        while self._click_plan:
            r, c = self._click_plan.popleft()
            if self.cplanner.is_lethal(r, c):
                continue                                   # skip a known-fatal target
            return f"ACTION6_r{r}_c{c}"
        return None

    def _valid_actions(self, latest_frame) -> List[str]:
        avail = getattr(latest_frame, "available_actions", None)
        if avail:
            out = []
            for a in avail:
                name = getattr(a, "name", None)          # arcengine GameAction enum
                if name is None:
                    try:
                        name = f"ACTION{int(a)}"          # raw API action id
                    except (TypeError, ValueError):
                        name = str(a)
                if name in VALID_ACTION_NAMES:
                    out.append(name)
            if out:
                return out
        return [f"ACTION{i}" for i in range(1, 8)]

    @staticmethod
    def _read_score(latest_frame) -> int:
        # arcengine >=0.9.3 renamed score -> levels_completed; support both.
        score = getattr(latest_frame, "levels_completed", None)
        if score is None:
            score = getattr(latest_frame, "score", 0)
        try:
            return int(score or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _grid_from(frame_obj) -> Optional[np.ndarray]:
        raw = getattr(frame_obj, "grid", None)
        if raw is None:
            raw = getattr(frame_obj, "frame", None)
        if raw is None:
            return None
        try:
            arr = np.array(raw, dtype=np.int32)
        except Exception:
            return None
        if arr.ndim == 3:               # arcengine FrameData.frame is a LIST of grids
            arr = arr[-1]               # (animation frames); the last one is current
        return arr if arr.size and arr.ndim == 2 else None

    def _parse_grid(self, latest_frame, frames) -> Optional[np.ndarray]:
        grid = self._grid_from(latest_frame)
        if grid is None and frames:
            for f in reversed(frames[:-1] if frames and frames[-1] is latest_frame else frames):
                grid = self._grid_from(f)
                if grid is not None:
                    break
        return grid

    def parse_action(self, action_str: str):
        if action_str.startswith("ACTION6"):
            r = c = None
            try:
                parts = action_str.split("_")
                r = int(parts[1][1:])
                c = int(parts[2][1:])
            except Exception:
                if self.last_grid is not None:
                    r, c = self.last_grid.shape[0] // 2, self.last_grid.shape[1] // 2
                else:
                    r = c = 0
            self.action_x, self.action_y = int(c), int(r)     # API convention: x=col, y=row
            act = GameAction.ACTION6
            if hasattr(act, "set_data"):
                try:
                    act.set_data({"x": int(c), "y": int(r)})
                except Exception:
                    pass
            return act
        return getattr(GameAction, action_str, GameAction.ACTION1)

    def _graph_explore(self, grid: np.ndarray, valid: List[str]) -> Optional[str]:
        """Graph-directed exploration (COMPONENT 5.8): the single step that makes
        progress toward the nearest (state, action) pair with an UNKNOWN outcome.
        Recomputed every STEP -- the games are deterministic, so re-running BFS
        from the live state stays consistent, and if reality diverges (hidden
        timer -> novel state) the novel state's own untested actions take over.
        Within one step the answer cannot change (same grid, same graph, same
        dead set), so the second caller reuses the first one's search: the routing
        ladder and the livelock escalation both ask, and on a game with a large
        graph that duplicate BFS was the dominant per-action cost."""
        if os.environ.get("ARC_NO_GRAPH"):          # debug bisect switch
            return None
        memo = self._ge_memo
        if memo is not None and memo[0] == self.steps_total:
            step = memo[1]
        else:
            noop = None
            if not os.environ.get("ARC_NO_PWM"):    # debug bisect switch
                noop = lambda g, a: self.pwm.predict_noop(g, a, self.sgraph._mask)
            # COMPONENT 5.12: a PERMUTATION of the equally-close frontier pairs,
            # never a filter -- the shortest-path guarantee and the set of
            # candidates are both untouched. `frontier_ranker` returns None
            # whenever the pool has no adoptable opinion, and the graph then takes
            # its original early-return path, so an inert goal model costs exactly
            # nothing (test_goalmodel.py case 9).
            rank = None
            if not os.environ.get("ARC_NO_GOALS"):  # debug bisect switch
                rank = self.goals.frontier_ranker()
            res = self.sgraph.nearest_untested_grid(grid, self.dead, noop=noop,
                                                    rank=rank)
            if res is None and noop is not None:
                # Imagination may have pruned everything. Completeness fallback:
                # re-offer the model-skipped pairs for real testing.
                res = self.sgraph.nearest_untested_grid(grid, self.dead, rank=rank)
            if res is None:
                step = None
            else:
                path, a = res
                step = path[0] if path else a
            # Cache the FRONTIER step, not the answer: `valid` is per-caller, so
            # filtering after the cache keeps the memo an optimisation and never a
            # behaviour change.
            self._ge_memo = (self.steps_total, step)
        return step if step is not None and step in valid else None

    # -- livelock escalation (COMPONENT 5.11) ------------------------------
    def _protected_sequence(self) -> bool:
        """Is a deterministic multi-step sequence mid-flight?

        The ClickPlanner's env-as-simulator search replays "RESET + path + button"
        and HASHES the result to record a graph edge; an LLM click plan is an
        ordered list reasoned about one specific board. Injecting a foreign action
        into either does not merely waste a step -- it desyncs the replay, so the
        next observed frame is attributed to the wrong edge and the search records
        a FALSE transition. Those layers own their own exhaustion tests
        (`_search_done`, plan drain), so the breaker stands off while they run."""
        return bool(self.cplanner.searching or self._click_plan)

    def _escalate(self, grid: np.ndarray, valid: List[str],
                  h: Optional[int]) -> Optional[str]:
        """The escalation ladder: climb until something can still teach us.

        Each rung is strictly more desperate than the last, and every rung is
        phrased in terms the agent can check on ANY game -- an unknown outcome, an
        unproven action, an unclicked pixel, a restart. Returns None when even the
        last rung has nothing, in which case the caller keeps the planner's own
        choice (escalation must never make the agent act LESS informedly than the
        layer it interrupted)."""
        # (1) An outcome the graph has never observed, reached over proven edges.
        #     BFS over the frontier is by far the most expensive rung, and once
        #     escalation is firing it would run on EVERY step -- measured at ~7x
        #     the per-action cost on a game with a large graph, which on a fixed
        #     wall clock costs more actions than the rung wins. It is throttled to
        #     one attempt per ESCALATE_BFS_EVERY escalations; the cheap rungs below
        #     cover the steps in between, so the ladder loses nothing but its worst
        #     case. (A memoised result already computed this step is free -- take
        #     it regardless of the throttle.)
        fresh_memo = self._ge_memo is not None and self._ge_memo[0] == self.steps_total
        if fresh_memo or self.steps_total - self._esc_bfs_at >= ESCALATE_BFS_EVERY:
            if not fresh_memo:
                self._esc_bfs_at = self.steps_total
            a = self._graph_explore(grid, valid)
            if a is not None and not self.breaker.stale(h, a):
                self._route = "esc.graph"
                return a
        # (2) A non-click action not yet proven inert, least-spent first. The graph
        #     may have nothing to say (novel frame every step under a hidden
        #     counter) while a plain untried button still exists here.
        keyed = [x for x in valid if base_action(x) != "ACTION6"]
        fresh = [x for x in keyed
                 if not self.dead.is_dead(x) and not self.breaker.stale(h, x)]
        if fresh:
            self._route = "esc.key"
            return min(fresh, key=lambda x: (self.breaker.spent(h, x), random.random()))
        # (3) Modality switch: a pixel the click surface has never probed. This is
        #     the rung that matters for click-only games, whose coverage lattice is
        #     exhausted long before the budget is -- see ClickPlanner.fresh_cell.
        #     Keys come FIRST (rung 2) on mixed games, which is the "coverage
        #     exhausted -> try keys" half of the escalation; the other half is the
        #     skip below.
        if any(base_action(x) == "ACTION6" for x in valid):
            # When EVERY cell ever sampled is proven inert, the click surface is
            # very likely globally inert, and grinding the remaining pixels one at
            # a time is a poor use of thousands of scored actions -- a restart is
            # the cheaper experiment. Only when a restart is actually available:
            # with the budget spent, refining is still strictly better than
            # repeating, so this never costs a rung.
            if not (self.dead.clicks_exhausted() and self.breaker.may_reset()):
                cell = self.cplanner.fresh_cell(grid)
                if cell is not None and not self.breaker.stale(h, cell):
                    self._route = "esc.cell"
                    return cell
        # (4) Fixed point under everything on offer: restart the level rather than
        #     keep hammering. Budgeted and only once the level is old enough that
        #     the engine reads RESET as a level_reset (ClickPlanner documents why
        #     an early RESET would wipe levels_completed).
        if self.breaker.may_reset():
            self.breaker.note_reset(by_breaker=True)
            print(f"[livelock] fixed point after {self.breaker.level_actions} "
                  f"actions ({self.breaker.run} inert in a row) -> RESET")
            self._route = "esc.reset"
            return "RESET"
        return None

    def _explore_action(self, grid: np.ndarray, valid: List[str]) -> str:
        # Directed beats random: never re-spend a scored action on a known outcome.
        a = self._graph_explore(grid, valid)
        if a is not None:
            self._route = "graph"
            return a
        # Graph exhausted (or action not currently valid): fall back to novelty /
        # random, skipping actions proven inert everywhere this level.
        alive = [a for a in valid if not self.dead.is_dead(a)] or valid
        if random.random() < self.epsilon or self.world_model.active_model is None:
            # LEAST-SPENT, not uniform random. This line decides ~15% of every dev
            # action -- most games never promote a world model, so `active_model is
            # None` holds and the novelty branch below is unreachable -- and uniform
            # random measured a 0.7% new-state yield against `graph`'s 54.0%.
            # Re-spending a (state, action) pair whose outcome is already on record
            # buys nothing, and RHAE squares wasted actions.
            #
            # This is the SAME rule escalation rung 2 already applies
            # (`min(fresh, key=(spent, random))`): one principle, two sites, rather
            # than a new heuristic. The random tie-break keeps it unbiased among
            # equally-unspent actions, so with an empty ledger it degrades exactly
            # to the uniform choice it replaces.
            #
            # OPT-IN (`ARC_LEASTSPENT=1`), DEFAULT OFF -- it did not prove out.
            # Paired A/B (2026-08-01, seed 0, the 7 dev games where this branch
            # fires): levels 0 -> 0 (none of those games completes a level, so the
            # test had no power there), branch yield 2.3% -> 6.5% new states, but
            # wall clock +179s (+11%) and think time +50.8s (+39%).
            #
            # The aggregate yield is carried entirely by sb26 (4.4 -> 25.3%) and
            # sc25 (0.1 -> 43.5%). On bp35 it went the other way and badly: the
            # branch grew 215 -> 1269 actions at a 0.1% yield, because least-spent
            # keeps electing actions the ledger has not seen while the graph is
            # starved. lf52 -- the 0.0%-yield case this was aimed at -- did not
            # move at all. Part of the added cost is a double hash: this hashes
            # `grid`, and choose_action hashes the same grid again for the breaker.
            # Fix that before re-measuring if this is ever revisited.
            if os.environ.get("ARC_LEASTSPENT"):
                try:
                    _h = self.sgraph._hash(grid)
                    self._route = "least"
                    return min(alive, key=lambda a: (self.breaker.spent(_h, a),
                                                     random.random()))
                except Exception:
                    pass        # a broken ledger must never cost a real action
            self._route = "rand"
            return random.choice(alive)
        self._route = "novelty"
        best_a, best_nov = alive[0], -1.0
        for a in alive:
            nxt = self.world_model.predict_or_none(grid, a)
            # Unknown => assume novel (peek_bonus' own upper bound), so a partial
            # model cannot suppress the very actions it says nothing about.
            nov = 1.0 if nxt is None else self.counter.peek_bonus(nxt)
            if nov > best_nov:
                best_nov, best_a = nov, a
        return best_a

    def _checkpoint(self):
        path = "/kaggle/working/agent_checkpoint.pkl" if os.path.isdir("/kaggle/working") else "agent_checkpoint.pkl"
        try:
            with open(path, "wb") as f:
                pickle.dump({"action_counter": getattr(self, "action_counter", 0),
                             "hypothesis": self.memory.l2_hypothesis,
                             "mode": self.gate.mode}, f)
        except Exception as e:
            print(f"Checkpoint failed: {e}")
        self.last_checkpoint = time.time()

    def _record_transition(self, prev, action, nxt, reward, terminal=False):
        # Ground truth first (append-only; level tag = the level the action was
        # taken IN, i.e. before this transition's reward is applied).
        self.timeline.append(prev, action, nxt, reward, self.last_score, terminal)
        diff = self.encoder.encode_transition(prev, nxt, action)
        t = Transition(prev, action, nxt, reward, diff, time.time())
        self.transitions.append(t)
        if len(self.transitions) > 200:
            self.transitions.pop(0)
        self.memory.add_transition(t)

    # ---- interface ----
    def is_done(self, frames, latest_frame) -> bool:
        """The SECOND entry point the harness calls, and the second one that can
        end a game by raising.

        choose_action got an exception wall; this did not, and the asymmetry was
        the bug. Everything inside the WIN branch can throw -- `_parse_grid`
        indexes a frame whose shape the environment chose, `np.array_equal`
        compares two grids that a level change may have resized, `_record_transition`
        walks the memory. A raise here kills the game exactly as dead as a raise
        in choose_action, and takes every level it had banked with it.

        The two failure directions are NOT symmetric, so the recovery is not
        either. Returning True wrongly ends a game that still had actions left;
        returning False wrongly costs one extra loop iteration, which the
        action cap then stops. So bookkeeping is best-effort and only the two
        cheap, total predicates decide the answer."""
        try:
            won = latest_frame.state == GameState.WIN
        except Exception:
            won = False                      # can't read the state: keep playing
        if won:
            # Roadmap V3 fix: capture the transition that CAUSED the win with the
            # REAL final grid from the environment -- never a duplicated last_grid.
            try:
                if not self._win_recorded and self.last_grid is not None and self.last_action_str:
                    self._win_recorded = True
                    final = self._parse_grid(latest_frame, frames)
                    if final is not None and final.shape == self.last_grid.shape \
                            and not np.array_equal(final, self.last_grid):
                        self._record_transition(self.last_grid, self.last_action_str, final, 1.0)
                    elif self.transitions:
                        # WIN frame carries no (new) grid: credit the previous
                        # action's already-recorded transition instead of fabricating one.
                        self.transitions[-1].reward = 1.0
                    self.memory.l1_episodes.append(f"WIN via {self.last_action_str}.")
            except Exception as e:
                # The win is real whether or not we managed to file the paperwork.
                print(f"[is_done] recovered while recording the win: {type(e).__name__}: {e}")
            if LLM_STATS.attempts:
                print(LLM_STATS.report())
            return True
        out_of_actions = getattr(self, "action_counter", 0) >= self.MAX_ACTIONS
        if out_of_actions and LLM_STATS.attempts and not self._stats_reported:
            self._stats_reported = True
            print(LLM_STATS.report())
        return out_of_actions

    def choose_action(self, frames, latest_frame):
        """Exception wall around the whole decision.

        The local harness catches a raise from here and records the cell as a
        STALL, so a crash locally costs one row. The Kaggle-side loop does not:
        one unhandled exception ends the game, and with it every level that game
        was ever going to bank. Nothing in ~8,000 lines of planners, graph
        rebuilds and numpy indexing is worth that trade, so the fallback is to
        press a legal button and keep playing.

        The recovery deliberately clears `last_action_str`. The next call learns
        from (last_grid, last_action_str) -> grid; after a failed step that pair
        is a lie -- the action recorded is not the one the fallback issued -- and
        teaching the world model a fabricated transition is worse than skipping
        one. Clearing it makes the learn block skip exactly one step."""
        try:
            return self._choose_action(frames, latest_frame)
        except Exception as e:
            self._crashes = getattr(self, "_crashes", 0) + 1
            if self._crashes <= 5:        # a crash loop must not flood the log
                print(f"[choose_action] recovered from {type(e).__name__}: {e}")
                traceback.print_exc()
            self.last_action_str = ""
            self._route = "crash"
            try:
                valid = self._valid_actions(latest_frame) or ["ACTION1"]
            except Exception:
                valid = ["ACTION1"]
            pick = random.choice(valid)
            if base_action(pick) == "ACTION6" and "_" not in pick:
                # A bare ACTION6 with no coordinate lands wherever parse_action's
                # own fallback puts it; centre it explicitly instead.
                g = self.last_grid
                if g is not None:
                    pick = f"ACTION6_r{g.shape[0] // 2}_c{g.shape[1] // 2}"
            return self.parse_action(pick)

    def _choose_action(self, frames, latest_frame):
        if time.time() - self.last_checkpoint > CHECKPOINT_INTERVAL_S:
            self._checkpoint()
            # Kaggle is the only place the LLM half ever runs, and this log is
            # the only instrument there. Print it periodically rather than once
            # at the end, so a run that is killed by the wall clock still says
            # what the model was producing.
            if LLM_STATS.attempts:
                print(LLM_STATS.report())
        self._synth_watchdog()

        grid = self._parse_grid(latest_frame, frames)
        if grid is None:
            return GameAction.ACTION1
        score = self._read_score(latest_frame)

        # 1) Learn from the previous step. Reward = a level was completed.
        if self.last_grid is not None and self.last_action_str:
            reward = 1.0 if score > self.last_score else 0.0
            # Terminality is known HERE, on the same frame, so the record carries
            # it from the start rather than being back-patched later. Deaths are
            # the only outcome evidence 22 of the 25 games have ever produced.
            terminal = latest_frame.state == GameState.GAME_OVER
            self._record_transition(self.last_grid, self.last_action_str, grid,
                                    reward, terminal)
            # The order over world-states (COMPONENT 5.12) sees EVERY real
            # transition -- score events and deaths included -- and only real
            # ones: an imagined rollout that "confirms" a goal confirms nothing.
            self.goals.observe(self.last_grid, self.last_action_str, grid,
                               reward, terminal, self.last_score)
            # ONE definition of "did anything happen" for every consumer below.
            # The dead-action tracker already asked the graph; the click planner
            # used to ask np.array_equal instead, so on a game with a live HUD it
            # credited a ticking timer as the click having done something -- the
            # same asymmetry the world model had, in the component that decides
            # which cells are worth clicking again.
            world_changed = self.sgraph.changed_masked(self.last_grid, grid)
            # A RESET is not a move: it is the escalation itself, and the frame it
            # lands on is the engine's choice. Feeding it to the physics learners
            # teaches a bogus "action" and puts a RESET edge in the graph that
            # BFS would then try to walk (it is never in `valid`, so the path
            # silently dead-ends).
            replaying = self.last_action_str == "RESET"
            # The engine's action counter decides whether the NEXT RESET is a
            # level reset or a score-destroying full reset, so it must see every
            # action -- RESET included, which `cplanner.update` below never sees.
            self.cplanner.note_action(self.last_action_str)
            # BEFORE sgraph.record, which may rebuild the HUD mask and with it every
            # state hash. `_prev_hash` was taken under the mask that is still live
            # right here, so keying the ledger now is the only way the key and the
            # verdict come from the same mask epoch. (A rebuild still orphans older
            # entries -- they were keyed under a mask that no longer exists -- but
            # that is self-healing and can only make `stale` too permissive, never
            # too aggressive.)
            if not replaying:
                self.breaker.update(self._prev_hash, self.last_action_str,
                                    changed=world_changed or reward > 0)
            # Only learn physics WITHIN a level. A level-up swaps the whole grid
            # (new avatar spawn), so prev->nxt is a discontinuity, not a move --
            # feeding it to the planner corrupts the learned displacements.
            if reward == 0 and not replaying:
                self.rplanner.update(self.last_grid, self.last_action_str, grid, self.encoder, reward)
                # Exact world model: record the observed edge (deterministic games,
                # so this is ground truth, not an estimate).
                self.sgraph.record(self.last_grid, self.last_action_str, grid)
                self.dead.update(self.last_action_str, changed=world_changed)
                self.pwm.learn(self.last_grid, self.last_action_str, grid,
                               self.sgraph._mask)
            if not replaying:
                self.cplanner.update(self.last_action_str,
                                     changed=world_changed, reward=reward)
            self.counter.get_bonus(grid)
            m = self.world_model.active_model
            if m is not None:
                # judge() owns all three exemptions: an action the model never
                # claimed, a transition where nothing happened in the world it
                # models, and the HUD cells it was never asked to predict. It
                # scores under the SAME per-action comparison verify used --
                # scoring raw here demoted correct models on every live-HUD game.
                hit = self.world_model.judge(self.last_grid, self.last_action_str, grid)
                if hit is not None:
                    m.accuracy = 0.9 * m.accuracy + 0.1 * hit
                    self.gate.update(hit)
                    if m.accuracy < DEMOTE_ACCURACY:
                        self.world_model.demote(m)
                        self.gate = ExploitGate()

        if os.environ.get("CP_DEBUG") and score < self.last_score:
            import sys
            print(f"[CP] SCORE REGRESSION {self.last_score}->{score} "
                  f"prev_action={self.last_action_str!r} state={latest_frame.state} "
                  f"cp_phase={getattr(self.cplanner,'_phase',None)} "
                  f"searching={self.cplanner.searching}", file=sys.stderr)

        # Level-up: refresh per-level budgets, keep the verified physics model.
        if score != self.last_score:
            # The frame the score moved on is score-relevant BY DEFINITION: pin
            # its changed cells out of the HUD mask before anything else looks
            # at them. On a level-up the graph is discarded three lines down, so
            # what this really protects is a score REGRESSION -- there the graph,
            # and the mask it carries, survive into the rest of the level.
            if self.last_grid is not None and self.last_action_str:
                self.sgraph.note_outcome(reward=1.0, prev=self.last_grid, nxt=grid)
            if score > self.last_score:
                print(f"[level] score {self.last_score} -> {score}; refreshing per-level budgets")
                self.llm_calls_this_level = 0
                self.level_steps = 0
                self.gate = ExploitGate()     # re-earn EXPLOIT on the new layout
                # Keep learned physics (avatar + per-action displacement) across
                # levels, but the layout changed -> forget positional stuck/target
                # AND the dead-cell blacklist (those coords belong to the old map).
                self.rplanner.new_level()
                self.cplanner.new_level()
                # The winning frame is the strongest evidence the game ever
                # emits. Recorded BEFORE the two-tier reset, which keeps the
                # learned win-condition and drops only its positional bindings.
                self.goals.note_level_complete(self.last_grid, self.last_score)
                self.goals.new_level()
                # New layout -> every recorded state/edge belongs to the old map.
                self.sgraph = StateGraph()
                self.dead = DeadActionTracker()
                self.breaker.new_level()
                self._discard_click_plan()
                self._theory_proposal = None      # reasoned about the old layout
                self._last_theorize_at = -10 ** 9
                # Bank the route that just worked (5.13). Must run BEFORE
                # last_score moves so the level number recorded is the one the
                # route actually reaches.
                self.replay.note_level_up(score)
            else:
                # A DROP means full_reset() wiped the banked levels. Everything
                # below us has to be walked again -- and we already know how.
                self.replay.note_wipe(score)
            self.last_score = score

        # 2) Terminal handling.
        if latest_frame.state == GameState.GAME_OVER:
            # Whatever moved on the frame we died on decides the score, so it may
            # never be masked away. The graph survives GAME_OVER -> RESET (same
            # layout), so this evidence accumulates over the level's deaths.
            if self.last_grid is not None and self.last_action_str:
                self.sgraph.note_outcome(terminal=True, prev=self.last_grid, nxt=grid)
            # Blame the click that killed us so the ClickPlanner backs off that cell
            # on restart (its stats survive the reset -- same layout, same hazards).
            if self.last_action_str.startswith("ACTION6"):
                self.cplanner.mark_loss(self.last_action_str)
            if self.llm.is_ready() and self.llm_calls_this_level < LLM_CALL_BUDGET_PER_LEVEL:
                self.llm_calls_this_level += 1
                self.memory.l2_hypothesis = self.llm.reflect(
                    self.memory.observation_digest(self.transitions), "GAME_OVER")
            # The level restarts from the top, so the walk back to here is about
            # to be paid for a second time. Arm it as a replay (5.13).
            self.replay.note_death(score)
            self._reset_level_state()
            return GameAction.RESET
        if latest_frame.state == GameState.NOT_PLAYED:
            return GameAction.RESET

        valid = self._valid_actions(latest_frame)
        self.level_steps += 1
        self.steps_total += 1
        self.epsilon = self._epsilon_for(self.level_steps)
        # Register the current state so its untried actions are known to the graph.
        # (Survives intra-level RESETs by design -- the layout is unchanged.)
        # The graph gaining a node here is the agent's most honest "I have never
        # been in this state before", and it is the novelty signal the breaker
        # uses to decide whether anyone is still learning. It has to be sampled at
        # ensure_state, NOT at record(): record only registers the SOURCE hash,
        # which ensure_state already added on the previous step, so the counter
        # there never moves. (A mask rebuild re-keys every state and so reads as
        # novelty -- correctly: the agent's notion of "state" just changed.)
        _nodes_before = len(self.sgraph.adj)
        self.sgraph.ensure_state(grid, valid)
        if len(self.sgraph.adj) > _nodes_before:
            self.breaker.note_novel()

        # Click-only games (available_actions == [6]) can never learn a movement
        # world-model, and (b2) always sets the action for them, so the MCTS/
        # world-model path never runs -- firing world-model synth there would waste
        # the per-level LLM budget on a model nothing consumes. Route that budget to
        # click reasoning instead.
        #
        # The test is "clicks are the ONLY thing on offer", which is what the line
        # above always claimed. It used to read "no arrow keys, and a click exists",
        # which also swallowed action sets like {5,6,7}: those games have real
        # non-click actions with learnable effects, and the branch skipped world-model
        # synthesis for them entirely. Measured on the dev set, that silently
        # excluded the single cleanest world model the enumerator finds anywhere --
        # a rule for ACTION5 that verifies at 1.00 with full coverage.
        # (`bool(valid)` keeps an empty action set out of the branch: the empty set
        # is a subset of everything, and "nothing on offer" is not a click game.)
        click_only = bool(valid) and {base_action(a) for a in valid} <= {"ACTION6"}

        # Schema Phase 2 (THEORIZE): adopt a completed theorizer proposal on the
        # main thread, then maybe fire the next budgeted background request.
        # Movement games only -- mechanical games are graph-driven and click
        # games route the same budget to click reasoning instead.
        if not click_only and not os.environ.get("ARC_NO_THEORIZE"):
            if self._theory_proposal is not None:
                prop, self._theory_proposal = self._theory_proposal, None
                if self.rplanner.propose_theory(prop, grid):
                    print(f"[theorize] adopted: {str(prop.get('hypothesis', ''))[:100]}")
                else:
                    print("[theorize] proposal rejected by full-history backtest")
            if not self.rplanner.looks_mechanical():
                self._maybe_request_theory(grid)

        # 3) Discover the world model. Enumeration first (cheap, sync, re-run only
        #    when NEW evidence arrived); LLM as budgeted background fallback.
        n_changed = sum(1 for t in self.transitions if t.changed)
        if (not click_only and self.world_model.active_model is None
                and n_changed != self._last_enum_at and synth_ready(self.transitions)):
            self._last_enum_at = n_changed
            if not self.world_model.try_enumerative(self.transitions):
                if (self.llm.is_ready() and not self.synth_inflight
                        and self.llm_calls_this_level < LLM_CALL_BUDGET_PER_LEVEL):
                    self.synth_inflight = True
                    self.synth_started_at = time.time()
                    self.llm_calls_this_level += 1
                    gen = self.synth_generation
                    obs = self.memory.observation_digest(self.transitions)
                    def _job():
                        try:
                            self.world_model.synthesize_with_llm(obs, lambda: list(self.transitions))
                        finally:
                            if gen == self.synth_generation:
                                self.synth_inflight = False
                    threading.Thread(target=_job, daemon=True).start()

        # 3.5) Arm a goal hypothesis (COMPONENT 5.12, Section 4.4). This is the
        #      ONLY writer path to the Beta posteriors: `_settle_pursuit` returns
        #      immediately unless a pursuit is on the hook, so without this call
        #      the whole credibility layer is inert -- which is exactly what the
        #      first measurement found (A=0, retired=0, all 25 games).
        #
        #      Arming alone costs nothing behaviourally. Alpha needs the
        #      `note_directed` warrant, which only fires when the model actually
        #      CHANGED a destination, so an armed-but-unused predicate can only
        #      ever be REFUTED. That is the safe half by construction: it can
        #      shrink the candidate pool, never inflate it.
        if (not os.environ.get("ARC_NO_GOALS")) and not self.goals.pursuing():
            try:
                self.goals.target(grid)
            except Exception:
                pass        # a broken hypothesis must never cost a real action

        # 4) Act.
        #    Priority: (a) bootstrap -- try each move action once to learn its
        #    displacement; (b) reactive goal-seek once physics is known (scale-
        #    invariant, the main solver); (c) fall back to world-model EXPLOIT /
        #    curiosity EXPLORE for non-movement games.
        action_str = None
        self._route, _route_t0 = "", time.perf_counter()

        # (a0) REPLAY CACHE (COMPONENT 5.13). Top of the ladder, because a
        #      VERIFIED route to a state the agent has already reached is
        #      strictly better than re-deriving it -- and RHAE bills the
        #      re-derivation to the NEXT level, where it is squared. The route
        #      is checked against the live frame (HUD-masked) inside
        #      next_action, so a world that has changed costs one action and
        #      then the planners take over exactly as they would have.
        if not os.environ.get("ARC_NO_REPLAY") and self.replay.replaying():
            ra = self.replay.next_action(grid, valid, self.sgraph.hud_mask())
            if ra:
                self._route = "replay"
                self._route_ms = (time.perf_counter() - _route_t0) * 1000.0
                self.replay.note_action(grid, ra)
                self.last_grid, self.last_action_str = grid, ra
                self._prev_hash = self.sgraph._hash(grid)
                return self.parse_action(ra)

        move_actions = [a for a in valid if a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")]
        untried = [a for a in move_actions if a not in self.rplanner.action_disp]
        # (a) Keep probing UNLEARNED moves until their displacement is known. A move
        #     blocked on its first try (avatar spawned against a wall) must be retried
        #     later from open space, else e.g. "up" is never learned and snake mazes
        #     become unsolvable. Rotate so each unlearned move gets fresh positions.
        if untried and (not self.rplanner.ready(valid) or self.level_steps % 6 == 0):
            action_str = untried[self.level_steps % len(untried)]
            self._route = "boot"
        elif (self.rplanner.ready(valid) and not self.rplanner.looks_mechanical()
                and not click_only and random.random() > self.epsilon):
            # `not click_only` because (b2) below -- the branch actually BUILT for
            # clicks -- is only reachable when this one declines. On a click-only
            # game ExecPlanner has no movement key to model, so it ends up
            # treating ACTION6 as a displacement and emitting a bare "ACTION6"
            # that the fixer downstream re-coordinates; measured on lp85 that path
            # took 84% of the actions at 70ms each (60s of the wall clock) for a
            # 2.2% new-state yield, while the ClickPlanner it displaced got 8%.
            # Game-type classifier: MECHANICAL games (rotation/cycling/counters --
            # frame changes a displacement model can't explain) must NOT be driven
            # by ExecPlanner; they fall through to (c), where the state graph
            # explores untested (state, action) pairs systematically.
            # ExecPlanner (COMPONENT 5.7) is the PRIMARY driver: T1 unscored optimal
            # planning + T2 directed frontier / falsification-driven livelock escape.
            # The decaying-epsilon gate keeps a small escape hatch to the count-based
            # novelty explorer (path c) for pathologies the planner can't break on its
            # own (e.g. keydoor's non-integer render scale). Epsilon decays per level,
            # so this costs few scored actions once physics is learned.
            self.rplanner._sub = ""
            action_str = self.rplanner.act(grid, valid, self.encoder)   # (b) goal-seek
            self._route = "react." + (self.rplanner._sub or "none")

        # (b2) Click-driven games: no movement actions at all, but ACTION6 is
        #      available. Three layers, ONE active at a time (search XOR LLM-plan
        #      XOR coverage), with coverage as the guaranteed floor:
        #        - the ClickPlanner's env-as-simulator search when it's running
        #          (its deterministic RESET-replay must not be interleaved);
        #        - otherwise an LLM-reasoned click plan for the CSP/logic puzzles
        #          blind coverage can't crack (budgeted, filled in the background);
        #        - otherwise coverage discovers the reactive surface.
        if action_str is None and click_only:
            others = [a for a in valid if not a.startswith("ACTION6")]
            if self.cplanner.searching:
                action_str = self.cplanner.act(grid, self.encoder, self.epsilon)
                self._route = "click.search"
            else:
                # Only reason with the LLM while the search isn't driving (keeps
                # one reasoning layer active); coverage evidence feeds the prompt.
                self._maybe_request_click_plan(grid)
                action_str = self._next_click_from_plan(grid)
                if action_str is not None:
                    self._route = "click.plan"
                if action_str is None and self._click_inflight:
                    # A plan is generating (reasoned about THIS board). Idle on a
                    # known-inert cell so the board doesn't move under it -- then it
                    # lands on a matching board and its ordered start validates.
                    hold = self.cplanner.inert_cell()
                    if hold is not None:
                        action_str = f"ACTION6_r{hold[0]}_c{hold[1]}"
                        self._route = "click.hold"
                # Don't inject a stray ACTION5/7 mid-plan/replay -- it desyncs the
                # deterministic sequences the plan / search depend on.
                if action_str is None and others and random.random() < 0.15:
                    action_str = random.choice(others)   # probe ACTION5/7 occasionally
                    self._route = "click.probe"
                elif action_str is None:
                    action_str = self.cplanner.act(grid, self.encoder, self.epsilon)
                    self._route = "click.cover"

        if action_str is None:                           # (c)
            # MCTS IS OPT-IN (`ARC_MCTS=1`), and the default is OFF. It was the
            # agent's single largest consumer of THINKING time -- 348s of 690s
            # across the 17 dev games, 50.5% -- for 12% of the actions, ZERO
            # reward events, and an 11.6% new-state yield against `graph`'s 54.0%.
            #
            # The paired A/B that decided it (2026-08-01, seed 0, the 8 dev games
            # where the branch fires; the other 9 cannot differ): levels 1 -> 1,
            # REGRESSED none, think time 498.9s -> 221.9s (-55.5%), wall clock
            # -301s. Removing it costs nothing measurable and buys back half the
            # agent's compute. sb26 alone went 127.8s -> 5.0s of thinking.
            #
            # Kept behind a switch rather than deleted: the evidence is dev-only,
            # and `_explore_action` -- which is what runs instead -- is the same
            # fallback the branch already had. Re-enable to re-measure.
            if (self.gate.should_exploit() and self.world_model.active_model is not None
                    and os.environ.get("ARC_MCTS")):
                action_str = self.planner.search(grid, valid, n_iterations=60)
                self._route = "mcts"
            else:
                action_str = self._explore_action(grid, valid)

        # Any path that yields a bare "ACTION6" (random explore, MCTS edge cases)
        # would resolve to a fixed center click -- give it real coordinates.
        if action_str == "ACTION6":
            action_str = self.cplanner.act(grid, self.encoder, self.epsilon)
            self._route += "+click"

        # (d) LIVELOCK CIRCUIT-BREAKER (COMPONENT 5.11). Deliberately the LAST
        #     word, and a post-filter rather than another branch in the routing
        #     ladder: the pathology is not that one planner picks badly, it is
        #     that whichever layer is driving stops learning and nothing notices.
        #     Checking the chosen action instead of the chooser covers every path
        #     above -- present and future -- with one rule.
        #     The two triggers are NOT equally strong evidence and are no longer
        #     treated as if they were. `tripped` (the world is frozen) fires
        #     unconditionally. `stale` (this one pair was re-spent inertly) is
        #     ordinary search cost, so it may only take control once the agent has
        #     stopped reaching new states as well -- measured, not assumed.
        h = self.sgraph._hash(grid)
        if (not os.environ.get("ARC_NO_BREAKER")
                and not self._protected_sequence()
                and (self.breaker.tripped()
                     or (self.breaker.stale(h, action_str)
                         and not self.breaker.learning()))):
            _pre = self._route
            alt = self._escalate(grid, valid, h)
            if alt is not None and alt != action_str:
                self.breaker.escalations += 1
                action_str = alt
                # Keep BOTH names: "who was overruled > who overruled it". A bare
                # escalation tag would hide the planner whose choice was thrown
                # away, which is the half that says where the ladder is leaking.
                self._route = f"{_pre}>{self._route}"
            else:
                self._route = _pre

        self._route_ms = (time.perf_counter() - _route_t0) * 1000.0
        self.replay.note_action(grid, action_str)      # 5.13: build this life's route
        self.last_grid = grid
        self.last_action_str = action_str
        self._prev_hash = h
        return self.parse_action(action_str)


# ==========================================
# SELF-TESTS (offline; no LLM / no sentence-transformers required)
# ==========================================
if __name__ == "__main__":
    print("Running component isolation tests...")
    enc = StateEncoder()
    g = np.zeros((10, 10), dtype=np.int32); g[2:5, 2:5] = 1
    assert "square" in enc.encode_state(g).lower()

    # Object cache: same grid -> same cached list object (Issue 3)
    assert enc.objects(g) is enc.objects(g)

    # --- ReactivePlanner: scale-invariant learning + BFS pathing ---
    rp = ReactivePlanner()
    # Avatar = color 2 (a 2x2 block, i.e. scale 2), jumps 2 px right under ACTION4.
    p = np.zeros((12, 12), dtype=np.int32); p[2:4, 2:4] = 2
    n = np.zeros((12, 12), dtype=np.int32); n[2:4, 4:6] = 2
    rp.update(p, "ACTION4", n, enc)
    assert rp.avatar_color == 2 and rp.action_disp["ACTION4"] == (0, 2)   # learned 2px, scale-invariant
    # A blocked move (no change) while a wall color sits in the destination -> obstacle learned.
    pb = np.zeros((12, 12), dtype=np.int32); pb[2:4, 2:4] = 2; pb[2:4, 4:6] = 8
    rp.update(pb, "ACTION4", pb, enc)
    assert 8 in rp.obstacle_colors
    # BFS routes AROUND a wall: avatar bottom-left, goal top-right, wall column between,
    # gap at the top. Teach all four moves first (scale 1 here for clarity).
    rp2 = ReactivePlanner(); rp2.avatar_color = 2
    rp2.action_disp = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}
    rp2.obstacle_colors = {8}
    board = np.zeros((5, 5), dtype=np.int32)
    board[1:5, 2] = 8            # wall column 2, rows 1..4 (gap at row 0)
    board[4, 0] = 2              # avatar bottom-left
    board[0, 4] = 4              # goal top-right
    a = rp2.act(board, ["ACTION1", "ACTION2", "ACTION3", "ACTION4"], enc)
    assert a == "ACTION1", f"BFS should head up toward the gap, got {a}"   # not blindly right into the wall

    # --- ExecPlanner (COMPONENT 5.7): executable world model + unscored planning ---
    M4 = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
    DISP = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}
    # T1: commit a WHOLE path around a wall (not just the first step). A letterbox ring
    # (colour 5) is the border so the interior background (0) stays walkable; a colour-8
    # wall splits the interior with a single gap on the top interior row.
    ep = ExecPlanner(); ep.avatar_color = 2; ep.action_disp = dict(DISP); ep.obstacle_colors = {8}
    eb = np.zeros((20, 20), dtype=np.int32); eb[0, :] = eb[-1, :] = eb[:, 0] = eb[:, -1] = 5
    eb[2:19, 10] = 8                     # interior wall down col 10 (gap at interior row 1)
    eb[18, 1] = 2                        # avatar: interior bottom-left
    eb[1, 18] = 4                        # goal:   interior top-right (across the wall)
    a = ep.act(eb, M4, enc)
    assert a in ("ACTION1", "ACTION4"), f"ExecPlanner must head toward the gap/goal, got {a}"
    assert len(ep._plan) > 1, "T1: a whole multi-step path must be committed, not one step"
    # T1 falsification: a blocked move (no frame change) records a POSITIONAL wall cell
    # and DROPS the committed plan, so act() replans around it next turn.
    ep2 = ExecPlanner(); ep2.avatar_color = 2; ep2.action_disp = {"ACTION4": (0, 1)}
    pb = np.zeros((6, 6), dtype=np.int32); pb[2, 2] = 2
    ep2._plan = deque([((0, 0), "ACTION4")])                 # a stale committed step
    ep2.update(pb, "ACTION4", pb, enc)                       # ACTION4 vetoed (no change)
    assert (2, 3) in ep2.walls, f"a blocked move must learn the wall cell, got {ep2.walls}"
    assert len(ep2._plan) == 0, "a falsified move must drop the committed plan"
    # T2 curiosity: no goal on the board -> steer to an UNVISITED cell (directed), never
    # return None-and-flail. Letterbox ring keeps the interior walkable.
    ep3 = ExecPlanner(); ep3.avatar_color = 2; ep3.action_disp = dict(DISP)
    fb = np.zeros((12, 12), dtype=np.int32); fb[0, :] = fb[-1, :] = fb[:, 0] = fb[:, -1] = 5
    fb[6, 6] = 2                                             # only the avatar, no goal
    af = ep3.act(fb, M4, enc)
    assert af in M4, f"T2 frontier must pick a directed move, got {af}"

    # --- Schema Phase 1 (COMPONENT 5.10): Timeline + CERTIFY + experiments ---
    # Grids: avatar colour 2, single pixel (scale 1). tg1->tgJ is a 5px jump the
    # theory (disp (0,1)) cannot explain -- and too large for the parent to
    # re-learn as a displacement, so it stays a stable counterexample.
    tg0 = np.zeros((6, 8), dtype=np.int32); tg0[3, 1] = 2
    tg1 = np.zeros((6, 8), dtype=np.int32); tg1[3, 2] = 2
    tgJ = np.zeros((6, 8), dtype=np.int32); tgJ[3, 7] = 2
    tgR = np.zeros((6, 8), dtype=np.int32); tgR[3, 3] = 2
    # (1) Consistent history certifies GREEN; a contradicting entry turns RED.
    tl = Timeline()
    ep4 = ExecPlanner(); ep4.avatar_color = 2; ep4.action_disp = {"ACTION4": (0, 1)}
    ep4.timeline = tl
    tl.append(tg0, "ACTION4", tg1, 0.0, 0)
    assert ep4.certify(), "a consistent record must certify green"
    tl.append(tg1, "ACTION4", tgJ, 0.0, 0)
    assert not ep4.certify(), "a contradicting record must certify red"
    # (2) RED voids the committed queue, and the next scored action is the
    #     DISCRIMINATING EXPERIMENT (re-run the disputed action at its cell).
    ep4._plan = deque([((3, 2), "ACTION4")])
    a4 = ep4.act(tg1, ["ACTION4"], enc)
    assert len(ep4._plan) == 0, "red certification must void the committed queue"
    assert a4 == "ACTION4", f"red -> re-test the disputed (cell, action), got {a4}"
    # (3) Re-test outcome A: theory HOLDS -> the recorded contradiction was a
    #     one-off scene event; the entry is excused and certification is green.
    tl2 = Timeline()
    ep5 = ExecPlanner(); ep5.CERT_REPLAY_GAP = 0
    ep5.avatar_color = 2; ep5.action_disp = {"ACTION4": (0, 1)}; ep5.timeline = tl2
    tl2.append(tg1, "ACTION4", tgJ, 0.0, 0)
    assert not ep5.certify()
    assert ep5.act(tg1, ["ACTION4"], enc) == "ACTION4"
    tl2.append(tg1, "ACTION4", tgR, 0.0, 0)          # re-test: moved 1px = theory
    ep5.update(tg1, "ACTION4", tgR, enc)
    assert 0 in ep5._excused, "a held re-test must excuse the one-off entry"
    assert ep5.certify(), "excused entries must leave certification green"
    # (4) Re-test outcome B: the contradiction REPRODUCES -> after EXP_TRIES the
    #     (cell, action) is an anomaly: out of certification scope, and BFS
    #     refuses to route plans through it.
    tl3 = Timeline()
    ep6 = ExecPlanner(); ep6.CERT_REPLAY_GAP = 0
    ep6.avatar_color = 2; ep6.action_disp = {"ACTION4": (0, 1)}; ep6.timeline = tl3
    tl3.append(tg1, "ACTION4", tgJ, 0.0, 0)
    assert not ep6.certify()
    for _ in range(ExecPlanner.EXP_TRIES):
        assert ep6.act(tg1, ["ACTION4"], enc) == "ACTION4"   # issue the experiment
        tl3.append(tg1, "ACTION4", tgJ, 0.0, 0)              # ...jump reproduces
        ep6.update(tg1, "ACTION4", tgJ, enc)
    ep6.act(tg1, ["ACTION4"], enc)                # tries exhausted -> accept anomaly
    assert ((3, 2), "ACTION4") in ep6._anomalies, "reproduced contradiction -> anomaly"
    assert ep6.certify(), "anomalous (cell, action) must be out of certification scope"
    # (5) VIGA/WorldCoder joint revision: 'predicted-wall but reality walked
    #     through' indicts the REPRESENTATION (the colour-obstacle abstraction),
    #     not the rule -- the colour is un-marked, no anomaly is recorded, and
    #     the whole record re-certifies green under the revised grounding.
    gV1 = np.zeros((8, 10), dtype=np.int32)
    gV1[3:5, 2:4] = 2       # avatar 2x2 (scale 2), centroid (3.5, 2.5)
    gV1[3:5, 4:6] = 8       # colour marked as wall -- actually walkable (a cart)
    gV2 = np.zeros((8, 10), dtype=np.int32)
    gV2[3:5, 4:6] = 2       # the avatar drove straight onto it (dc = +2)
    tl4 = Timeline()
    ep7 = ExecPlanner(); ep7.CERT_REPLAY_GAP = 0
    ep7.avatar_color = 2; ep7.action_disp = {"ACTION4": (0, 2)}
    ep7.obstacle_colors = {8}; ep7.timeline = tl4
    tl4.append(gV1, "ACTION4", gV2, 0.0, 0)
    assert not ep7.certify(), "walked-through-'wall' must be a counterexample"
    for _ in range(ExecPlanner.EXP_TRIES):
        assert ep7.act(gV1, ["ACTION4"], enc) == "ACTION4"   # experiment re-tests
        tl4.append(gV1, "ACTION4", gV2, 0.0, 0)              # ...and it reproduces
        ep7.update(gV1, "ACTION4", gV2, enc)
    ep7.act(gV1, ["ACTION4"], enc)                # exhausted -> revise, not anomaly
    assert 8 not in ep7.obstacle_colors, "the colour-obstacle grounding must be indicted"
    assert not ep7._anomalies, "representation revision must pre-empt the anomaly"
    assert ep7.certify(), "the record must re-certify green under the revised grounding"

    # --- Schema Phase 2: budgeted LLM theorizer, backtest-gated adoption ---
    # (a) REJECT: the record shows a move into colour 7 was VETOED (avatar 2x2,
    #     disp (0,2) -- only "7 is a wall" explains the veto). A proposal
    #     un-marking 7 increases mismatches and must be reverted wholesale.
    gP1 = np.zeros((8, 10), dtype=np.int32)
    gP1[3:5, 2:4] = 2       # avatar 2x2 (scale 2)
    gP1[3:5, 4:6] = 7       # a REAL wall
    tlP = Timeline()
    ep8 = ExecPlanner(); ep8.CERT_REPLAY_GAP = 0
    ep8.avatar_color = 2; ep8.action_disp = {"ACTION4": (0, 2)}
    ep8.obstacle_colors = {7}; ep8.timeline = tlP
    tlP.append(gP1, "ACTION4", gP1, 0.0, 0)       # veto: avatar stayed put
    assert ep8.certify()
    assert not ep8.propose_theory({"not_obstacle_colors": [7], "goal": None,
                                   "decoy_colors": [9]}, gP1), \
        "a proposal the record contradicts must be rejected"
    assert 7 in ep8.obstacle_colors and 9 not in ep8.non_goal_colors, \
        "a rejected proposal must be reverted wholesale (no partial credit)"
    # (b) ACCEPT: the record shows the avatar drove THROUGH colour 8 (red);
    #     un-marking 8 explains the record -> adopted, and the goal hint lands.
    tlQ = Timeline()
    ep9 = ExecPlanner(); ep9.CERT_REPLAY_GAP = 0
    ep9.avatar_color = 2; ep9.action_disp = {"ACTION4": (0, 2)}
    ep9.obstacle_colors = {8}; ep9.timeline = tlQ
    tlQ.append(gV1, "ACTION4", gV2, 0.0, 0)       # walked through the "wall"
    assert not ep9.certify()
    assert ep9.propose_theory({"not_obstacle_colors": [8], "goal": [1, 8],
                               "decoy_colors": [9]}, gV1)
    assert 8 not in ep9.obstacle_colors and 9 in ep9.non_goal_colors
    assert ep9.certify(), "adoption must re-certify against the full record"
    assert ep9._llm_goal == (1, 8)
    # (c) The adopted goal hint REDIRECTS planning; reaching the hinted cell
    #     without a level-up drops it (reality outranks the hypothesis).
    epA = ExecPlanner(); epA.avatar_color = 2; epA.action_disp = dict(DISP)
    hb = np.zeros((12, 12), dtype=np.int32)
    hb[0, :] = hb[-1, :] = hb[:, 0] = hb[:, -1] = 5
    hb[6, 2] = 2                                  # avatar; NO distinctive goal colour
    assert epA.propose_theory({"goal": [6, 9]}, hb)
    aH = epA.act(hb, M4, enc)
    assert aH == "ACTION4", f"the hinted goal must drive planning, got {aH}"
    assert epA._target == (6, 9)
    hb2 = np.zeros((12, 12), dtype=np.int32)
    hb2[0, :] = hb2[-1, :] = hb2[:, 0] = hb2[:, -1] = 5
    hb2[6, 9] = 2                                 # avatar AT the hinted cell
    epA.act(hb2, M4, enc)
    assert epA._llm_goal is None, "reaching the hint without reward must drop it"
    # (d) LocalLLM.theorize parses a JSON proposal out of noisy generation.
    class _StubLLM(LocalLLM):
        def __init__(self):
            super().__init__(model_path="/nonexistent")
        def generate(self, *a, **k):
            return ('noise {"hypothesis": "h", "goal": [3, 4], '
                    '"not_obstacle_colors": [8]} tail')
    resT = _StubLLM().theorize("....", "t", "e", "stuck")
    assert resT and resT["goal"] == [3, 4] and resT["not_obstacle_colors"] == [8]

    # --- ClickPlanner: reward-exploit > coverage > active-safe > dead, avoid lethal ---
    # 8x8 grid at QUANT=4 -> exactly 4 lattice keys (0,0),(0,1),(1,0),(1,1); check the
    # quantized KEY of each pick (object-centre coords may share a key, so don't assert
    # exact px). New contract differs from the old one on purpose (see class docstring).
    cg = np.zeros((8, 8), dtype=np.int32)
    keyof = lambda s: ClickPlanner()._key(*ClickPlanner._parse(s))
    # (1) A cell that PAID OFF (reward) is re-selected over untried coverage cells.
    cp = ClickPlanner()
    cp.update("ACTION6_r2_c2", changed=True, reward=1.0)      # key (0,0) rewarded
    assert keyof(cp.act(cg, enc, epsilon=0.0)) == (0, 0), "reward cell must be exploited"
    # (2) Among non-rewarding TRIED cells (all lattice keys used up), active > dead.
    cp2 = ClickPlanner()
    cp2.update("ACTION6_r2_c2", changed=False, reward=0.0)    # (0,0) dead
    cp2.update("ACTION6_r2_c6", changed=True, reward=0.0)     # (0,1) active
    cp2.update("ACTION6_r6_c2", changed=False, reward=0.0)    # (1,0) dead
    cp2.update("ACTION6_r6_c6", changed=False, reward=0.0)    # (1,1) dead
    assert keyof(cp2.act(cg, enc, epsilon=0.0)) == (0, 1), "active cell preferred over dead"
    # (3) A lethal (loss-marked) active cell loses to an equally-active clean cell.
    cp3 = ClickPlanner()
    cp3.update("ACTION6_r2_c2", changed=True, reward=0.0); cp3.mark_loss("ACTION6_r2_c2")  # (0,0) active+lethal
    cp3.update("ACTION6_r2_c6", changed=True, reward=0.0)     # (0,1) active clean
    cp3.update("ACTION6_r6_c2", changed=False, reward=0.0)    # (1,0) dead
    cp3.update("ACTION6_r6_c6", changed=False, reward=0.0)    # (1,1) dead
    assert keyof(cp3.act(cg, enc, epsilon=0.0)) == (0, 1), "lethal cell must be avoided"
    # (4) Intra-level reset() KEEPS board knowledge; new_level() wipes it.
    cp3.reset();      assert cp3._stats, "reset() must preserve click stats across restarts"
    cp3.new_level();  assert not cp3._stats, "new_level() must wipe stats on real level-up"

    # --- ClickPlanner state-graph BFS: env-as-simulator world model over buttons ---
    # Mirror real lp85: two well-separated buttons, each a MULTI-CELL sprite whose
    # several reactive coords must cluster into ONE button (raw coords over-count
    # buttons ~3x, which throttled the old blind product-BFS).
    A, B = "ACTION6_r30_c2", "ACTION6_r30_c58"     # cluster representatives (interior pixels)

    def _register_buttons(cp):
        # Feed BOTH the click stats and the engine-counter mirror, exactly as
        # MyAgent.choose_action does for every executed action. A planner whose
        # mirror never moved would believe no action had been spent since the
        # last RESET and would skip the RESET that establishes the search root.
        def _click(cell, n=20):
            for _ in range(n):
                cp.update(cell, changed=True, reward=0.0)
                cp.note_action(cell)
        # button B (RIGHT sprite, cols ~58): two adjacent reactive cells -> 1 cluster
        _click("ACTION6_r30_c58"); _click("ACTION6_r34_c58")
        # button A (LEFT sprite, cols ~2-6): three adjacent cells -> 1 cluster
        _click("ACTION6_r30_c2"); _click("ACTION6_r30_c6"); _click("ACTION6_r34_c2")

    cps = ClickPlanner()
    gg = np.zeros((64, 64), dtype=np.int32)
    assert not cps.searching and cps.act(gg, enc, 0.0).startswith("ACTION6_r")  # coverage until discovered
    _register_buttons(cps)
    assert cps._detect_buttons() == [(30, 2), (30, 58)], \
        f"multi-cell sprites must cluster to 2 buttons, got {cps._detect_buttons()}"

    # A deterministic 1-D slide puzzle as the "env": button A (index 0) slides toward
    # position 0 and PARKS there (a wall fixed point); button B (index 1) slides toward
    # goal position 3 and parks there. RESET returns to position 0 (the root). Each
    # position renders to a distinct grid so states hash apart. This is the exact shape
    # the state-graph BFS must handle: no-op parked clicks + reconverging paths that a
    # blind product-BFS would re-explore combinatorially.
    GOAL = 3
    pos = [0]
    def _sim_grid():
        g = np.zeros((64, 64), dtype=np.int32); g[0, 0] = pos[0]; return g
    def _apply(a):
        if a == "RESET":            pos[0] = 0
        elif a == A:                pos[0] = max(pos[0] - 1, 0)       # slide left, park at 0
        elif a == B:                pos[0] = min(pos[0] + 1, GOAL)    # slide right, park at goal

    emitted = []
    for _ in range(400):
        a = cps.act(_sim_grid(), enc, 0.0)
        emitted.append(a)
        cps.note_action(a)          # the agent mirrors every executed action
        _apply(a)
        if cps._search_done:
            break
    assert emitted[0] == "RESET", f"search must establish the root via a RESET first, got {emitted[0]}"
    assert A in emitted and B in emitted, "both buttons must be probed by the BFS"
    # Dedup is the whole point: 4 reachable boards (positions 0..3) despite the parked
    # no-op re-clicks and the A/B paths that reconverge. A blind product-BFS would keep
    # re-expanding those; the state graph visits each distinct board exactly once.
    assert len(cps._dist) == 4, f"state graph must dedup to 4 distinct boards, got {len(cps._dist)}"
    assert cps._search_done and not cps.searching, "BFS must exhaust the reachable graph then yield to coverage"
    # Every discovered edge is deterministic-correct: from every non-root board, button A
    # steps one closer to 0 (recorded child hash matches the pos-1 board).
    root_h = cps._root
    assert cps._dist[root_h] == () and cps._G[root_h][1] != root_h, "B from root must reach a NEW board"

    # Post-exhaustion the planner is a pure coverage clicker again (never spins on RESET).
    assert cps.act(_sim_grid(), enc, 0.0).startswith("ACTION6_r"), "exhausted search -> coverage clicks"

    # GAME_OVER mid-replay: reset() must blame the final (parent, button) edge as dead
    # so the BFS never retries the losing click, and drop the interrupted plan.
    cpd = ClickPlanner(); _register_buttons(cpd)
    cpd._enter_search(cpd._detect_buttons())
    cpd.note_action(cpd.act(np.zeros((64, 64), dtype=np.int32), enc, 0.0))   # RESET (root plan)
    cpd.note_action(cpd.act(_sim_grid(), enc, 0.0))               # processes root, stages an expand
    assert cpd._pending is not None and cpd._pending[0] == "expand", "should be probing an edge"
    parent, b = cpd._pending[1], cpd._pending[2]
    cpd.reset()                                                    # simulate GAME_OVER on that edge
    assert (parent, b) in cpd._dead_edges, "a fatal replay edge must be marked dead"
    assert cpd._plan == [] and cpd._pending is None, "the interrupted plan must be dropped"

    # Button positions persist across level-up (UI arrows are level-invariant), but a
    # new level starts in COVERAGE, never search: a search RESET as the level's first
    # action is a FULL reset that wipes levels_completed (the lp85 0/8 regression).
    cps.new_level()
    assert cps._buttons == [(30, 2), (30, 58)] and not cps.searching, "new level must not start in search"
    assert cps.act(gg, enc, 0.0).startswith("ACTION6_r"), "first action of a level must be a click, not RESET"
    # search re-engages only after re-accumulating discover clicks (>=1 click => the
    # engine's action_count>0, so a later RESET is a level_reset that keeps the score).
    for _ in range(cps.MIN_DISCOVER_CLICKS):
        cps.update("ACTION6_r30_c2", changed=True, reward=0.0)
        cps.note_action("ACTION6_r30_c2")
    assert cps.act(gg, enc, 0.0) == "RESET" and cps.searching, "search resumes after coverage budget"

    # --- StateGraph + DeadActionTracker (COMPONENT 5.8) ---
    sg = StateGraph()
    g0 = np.zeros((4, 4), dtype=np.int32)
    g1 = g0.copy(); g1[0, 0] = 1
    g2 = g0.copy(); g2[1, 1] = 2
    h0 = sg._hash(g0)
    sg.ensure_state(g0, ["ACTION1", "ACTION2", "ACTION6"])
    assert "ACTION6" not in sg.untested[h0], "clicks must never enter the graph"
    res = sg.nearest_untested_grid(g0)
    assert res is not None and res[0] == [] and res[1] in ("ACTION1", "ACTION2"), \
        f"current state's own untested action first (tie-break random), got {res}"
    sg.record(g0, "ACTION1", g1)                                  # h0 --A1--> h1
    sg.ensure_state(g1, ["ACTION1", "ACTION2"])
    sg.record(g1, "ACTION1", g2)                                  # h1 --A1--> h2
    res = sg.nearest_untested_grid(g0)
    assert res == ([], "ACTION2"), "untested at the CURRENT state beats a walk"
    sg.record(g0, "ACTION2", g0)                                  # A2 inert at h0 (self-loop)
    # h0 exhausted -> BFS must walk the proven A1 edge to h1, where A2 is untested.
    res = sg.nearest_untested_grid(g0)
    assert res == (["ACTION1"], "ACTION2"), f"BFS to nearest untested failed: {res}"

    # HUD normalization: a cell mutating on EVERY transition (timer / budget bar)
    # must be masked out of the hash, so world-identical frames collapse to ONE
    # state and inert moves read as self-loops instead of endless novelty.
    sgh = StateGraph()
    def _hud(world, t):
        g = np.zeros((6, 6), dtype=np.int32); g[0, 0] = world; g[5, 5] = t; return g
    frames = [_hud(0, t) for t in range(StateGraph.MASK_CHECK_EVERY + 2)]
    acts3 = ["ACTION1", "ACTION2", "ACTION3"]
    sgh.ensure_state(frames[0], acts3)
    for t in range(StateGraph.MASK_CHECK_EVERY):    # A1/A2 alternate, both only tick HUD
        sgh.record(frames[t], "ACTION1" if t % 2 == 0 else "ACTION2", frames[t + 1])
        sgh.ensure_state(frames[t + 1], acts3)
    assert sgh._mask is not None and sgh._mask[5, 5] and not sgh._mask[0, 0], \
        "the always-changing HUD cell (and only it) must be masked"
    assert len(sgh.untested) == 1, \
        f"world-identical frames must collapse to 1 state, got {len(sgh.untested)}"
    assert not sgh.changed_masked(frames[0], frames[1]), "HUD-only change = no world change"
    wchg = frames[0].copy(); wchg[0, 0] = 9
    assert sgh.changed_masked(frames[0], wchg), "a real world change must still count"
    res = sgh.nearest_untested_grid(frames[-1])
    assert res == ([], "ACTION3"), f"collapsed state: A1/A2 tested (self-loops), A3 next, got {res}"
    # HUD strip: a draining budget bar changes a DIFFERENT cell each transition
    # (no single cell crosses MASK_RATE) but its row changes every time -> the
    # whole row is masked and world-identical frames still collapse.
    sgb = StateGraph()
    def _bar(world, t):
        g = np.zeros((6, 40), dtype=np.int32); g[0, 0] = world
        g[5, :40 - t] = 8                                         # bar drains right-to-left
        return g
    bframes = [_bar(0, t) for t in range(StateGraph.MASK_CHECK_EVERY + 2)]
    sgb.ensure_state(bframes[0], ["ACTION1", "ACTION2"])
    for t in range(StateGraph.MASK_CHECK_EVERY):    # bar drains under BOTH actions
        sgb.record(bframes[t], "ACTION1" if t % 2 == 0 else "ACTION2", bframes[t + 1])
        sgb.ensure_state(bframes[t + 1], ["ACTION1", "ACTION2"])
    assert sgb._mask is not None and sgb._mask[5].all() and not sgb._mask[0, 0], \
        "the draining-bar row (and not the world) must be masked"
    assert len(sgb.untested) == 1, \
        f"bar-only changes must collapse to 1 state, got {len(sgb.untested)}"
    # Policy-conditioned world changes must NOT be masked: a sprite cycled by A1
    # but untouched by A2 changes under only ONE action -> it is world, not HUD.
    sgp = StateGraph()
    spr = 0
    a1_steps = {0, 1, 3, 6, 10, 15, 21}     # irregular gaps -> not a periodic clock
    for t in range(StateGraph.MASK_CHECK_EVERY):
        a = np.zeros((6, 6), dtype=np.int32); a[2, 2] = 3 + spr
        b = a.copy()
        if t in a1_steps:
            spr = 1 - spr
            b[2, 2] = 3 + spr                           # A1 cycles the sprite
        sgp.record(a, "ACTION1" if t in a1_steps else "ACTION2", b)  # A2 quiet
    assert sgp._mask is None or not sgp._mask[2, 2], \
        "a cell quiet under one well-sampled action must stay unmasked"

    # --- PatchWorldModel (COMPONENT 5.9): local rules predict untested actions ---
    pwm = PatchWorldModel()
    def _cor(n_track, pos):
        # corridor: track cells (2) at cols 1..n_track, avatar (7) at pos, row 3
        g = np.zeros((7, 15), dtype=np.int32)
        g[3, 1:n_track + 1] = 2
        g[3, pos] = 7
        return g
    for _ in range(PatchWorldModel.MIN_SEEN):
        for p in range(1, 8):                         # long corridor: valid right-moves
            pwm.learn(_cor(10, p), "ACTION4", _cor(10, p + 1))
        pwm.learn(_cor(5, 5), "ACTION4", _cor(5, 5))  # short corridor: blocked at end
    assert not pwm.predict_noop(_cor(10, 9), "ACTION4"), \
        "a move onto track must not be predicted as a no-op"
    assert pwm.predict_noop(_cor(10, 10), "ACTION4"), \
        "blocked-at-wall must generalize from the short corridor to an unseen position"
    assert not pwm.predict_noop(np.random.randint(0, 9, (7, 15)), "ACTION4"), \
        "unknown patches must never be called no-ops"
    # Graph integration: predicted no-ops are tested by imagination (unscored),
    # and a noop-less pass re-offers them (completeness fallback).
    sgn = StateGraph()
    gN = np.zeros((4, 4), dtype=np.int32)
    sgn.ensure_state(gN, ["ACTION3", "ACTION4"])
    hN = sgn._hash(gN)
    res = sgn.nearest_untested(hN, noop=lambda g, a: a == "ACTION4")
    assert res == ([], "ACTION3"), f"predicted no-op must be skipped, got {res}"
    sgn.record(gN, "ACTION3", gN)
    assert sgn.nearest_untested(hN, noop=lambda g, a: a == "ACTION4") is None, \
        "imagination-pruned graph reads exhausted"
    assert sgn.nearest_untested(hN) == ([], "ACTION4"), \
        "fallback without the model re-offers the skipped pair"

    # A globally-dead action is excluded from graph suggestions.
    dt = DeadActionTracker()
    for _ in range(DeadActionTracker.MIN_TRIALS):
        dt.update("ACTION2", changed=False)
    assert dt.is_dead("ACTION2")
    assert sg.nearest_untested(h0, dt) is None, "dead actions must not be suggested"
    dt.update("ACTION2", changed=True)                            # one change whitelists it
    assert not dt.is_dead("ACTION2")
    for _ in range(9):
        dt.update("ACTION6_r1_c1", changed=False)
    assert not dt.is_dead("ACTION6_r1_c1"), "clicks are never globally dead"

    # --- Game-type classifier: mechanical games bypass the displacement planner ---
    rpm = ReactivePlanner()
    base_g = np.zeros((16, 16), dtype=np.int32); base_g[4, 4] = 3
    for k in range(8):        # 8 changed frames, none a clean translation (cycling)
        nxt_g = np.zeros((16, 16), dtype=np.int32); nxt_g[4, 4] = 3 + (k % 2) + 1
        rpm.update(base_g, "ACTION1", nxt_g, enc)
        base_g = nxt_g
    assert rpm.looks_mechanical(), "8/8 unexplained changes must classify as mechanical"
    rpw = ReactivePlanner()   # clean translations -> walk-to-goal, planner stays on
    wg = np.zeros((12, 12), dtype=np.int32); wg[2:4, 2:4] = 2
    for _ in range(8):
        nw = np.roll(wg, 2, axis=1)
        rpw.update(wg, "ACTION4", nw, enc)
        wg = nw
    assert not rpw.looks_mechanical(), "clean translations must NOT classify as mechanical"
    # A wrapping selector (tr87 cursor) LOOKS like clean translations, but its
    # displacement flips sign at every wrap -> contradictions flag it mechanical.
    rpc = ReactivePlanner()
    cg = np.zeros((12, 40), dtype=np.int32)
    pos = [2, 12, 22]         # 3 selector slots; cursor wraps 22 -> 2
    for k in range(12):
        a = np.zeros_like(cg); a[5, pos[k % 3]:pos[k % 3] + 8] = 6
        b = np.zeros_like(cg); b[5, pos[(k + 1) % 3]:pos[(k + 1) % 3] + 8] = 6
        rpc.update(a, "ACTION3", b, enc)
    assert rpc.disp_contradictions >= 3 and rpc.looks_mechanical(), \
        f"sign-flipping selector must classify mechanical ({rpc.disp_contradictions} flips)"

    # --- Span-based goal candidacy: structure never outranks a discrete object ---
    rps = ReactivePlanner(); rps.avatar_color = 2
    sb = np.zeros((20, 20), dtype=np.int32)
    sb[10, :] = 7            # full-width interior wall (span 100%): structural
    sb[2, 2] = 2             # avatar
    sb[17, 17] = 4           # the real goal object -- far away, 1 pixel
    tgt = rps._nearest_target(sb, 2.0, 2.0)
    assert tgt == (17, 17), f"structural colour must be demoted, got {tgt}"
    # ...but on a board with ONLY structure left, it is still a last-resort target.
    sb2 = np.zeros((20, 20), dtype=np.int32); sb2[10, :] = 7; sb2[2, 2] = 2
    assert rps._nearest_target(sb2, 2.0, 2.0) is not None, "structure stays a last resort"

    # DSL primitives + no-exec compile
    prog = [{"action": "*", "op": "translate", "args": {"dr": 1, "dc": 0}}]
    fn = compile_program(prog, enc)
    assert fn is not None and np.array_equal(fn(g, "ACTION1")[3, 2:5], np.array([1, 1, 1]))
    assert compile_program([{"op": "no_such_op"}], enc) is None

    # Object-scoped ops
    og = np.zeros((6, 6), dtype=np.int32); og[1, 1] = 2; og[4, 4] = 3
    moved = GridDSL.translate_color(og, 0, color=2, dr=0, dc=1)
    assert moved[1, 2] == 2 and moved[1, 1] == 0 and moved[4, 4] == 3   # obstacle untouched
    slid = GridDSL.slide_color(og, 0, color=2, direction="down")
    assert slid[5, 1] == 2 and slid[1, 1] == 0                          # slid to the wall
    blocked = np.zeros((6, 6), dtype=np.int32); blocked[0, 0] = 2; blocked[3, 0] = 5
    slid2 = GridDSL.slide_color(blocked, 0, color=2, direction="down")
    assert slid2[2, 0] == 2 and slid2[3, 0] == 5                        # stops above the 5
    swapped = GridDSL.swap_colors(og, 0, a=2, b=3)
    assert swapped[1, 1] == 3 and swapped[4, 4] == 2

    # Synthesis trigger (Issue 2): 4 same-action OR 8 total with >=2 actions of >=2
    def _t(act):
        p = np.zeros((2, 2), dtype=np.int32); n = np.ones((2, 2), dtype=np.int32)
        return Transition(p, act, n, 0.0, "", time.time())
    assert synth_ready([_t("ACTION1")] * 3) is False
    assert synth_ready([_t("ACTION1")] * 4) is True
    assert synth_ready([_t("ACTION1")] * 3 + [_t("ACTION2")] * 3) is False       # 6 total
    assert synth_ready([_t(f"ACTION{i}") for i in (1, 2, 3, 4)] * 2) is True     # 8 total, 4x2

    # Build a ground-truth "ACTION3 = gravity down" world and recover it WITHOUT any LLM
    truth = compile_program([{"action": "ACTION3", "op": "gravity", "args": {"direction": "down"}}], enc)
    trans, cur = [], np.zeros((8, 8), dtype=np.int32)
    for i in range(6):
        cur = cur.copy(); cur[0, i % 8] = (i % 3) + 1     # scatter some pixels up top
        nxt = truth(cur, "ACTION3")
        trans.append(Transition(cur, "ACTION3", nxt, 0.0, enc.encode_transition(cur, nxt, "ACTION3"), time.time()))
        cur = nxt
    syn = EnumerativeSynthesizer(enc)
    recovered = syn.synthesize(trans)
    assert recovered is not None and "gravity" in recovered.hypothesis, "enumerator failed to recover rule"

    # Recover an OBJECT-scoped rule: "ACTION2 moves the color-2 avatar right by 1"
    # (a static color-3 obstacle defeats the global translate hypothesis)
    ctrans = []
    for i in range(1, 6):
        p = np.zeros((8, 8), dtype=np.int32); p[3, i] = 2; p[7, 7] = 3
        n = np.zeros((8, 8), dtype=np.int32); n[3, i + 1] = 2; n[7, 7] = 3
        ctrans.append(Transition(p, "ACTION2", n, 0.0, enc.encode_transition(p, n, "ACTION2"), time.time()))
    rec2 = syn.synthesize(ctrans)
    assert rec2 is not None and rec2.spec[0]["op"] in ("step_color", "translate_color") \
        and rec2.spec[0]["args"]["color"] == 2, \
        f"expected an object-scoped rule on color 2, got {rec2.spec if rec2 else None}"
    # Every sample above is an UNOBSTRUCTED move, so step_color and
    # translate_color fit them identically -- the samples cannot break the tie.
    # What distinguishes them is the state the samples never contain: the avatar
    # pressed against the edge. A world model used for unscored lookahead must
    # not predict that the avatar walks off the board and vanishes, so require
    # the recovered rule to be the one that keeps it.
    edge = np.zeros((8, 8), dtype=np.int32); edge[3, 7] = 2; edge[7, 7] = 3
    assert np.array_equal(rec2.fn(edge, "ACTION2"), edge), \
        f"recovered rule loses the avatar at the edge: {rec2.spec}"

    # Verifier: correct model promotes, identity trap rejected (roadmap V2)
    wmm = WorldModelManager(LocalLLM(model_path="/nonexistent"), enc)
    assert wmm.llm.available is False                      # missing model dir -> LLM cleanly disabled
    assert wmm.verify(recovered.fn, trans) >= VERIFY_PROMOTE_THRESHOLD
    identity = compile_program([{"action": "*", "op": "identity"}], enc)
    assert wmm.verify(identity, trans) < VERIFY_PROMOTE_THRESHOLD
    assert wmm.try_enumerative(trans) is True and wmm.active_model is not None

    # A PARTIAL model is judged on its OWN domain. Add changed transitions for an
    # action the program has no rule for: the old verifier sampled the last 5
    # changed transitions whatever their action, so these unclaimed ones scored
    # wrong and dragged a perfect ACTION3 rule to 0. This is the coverage defect
    # (mean claimed share 0.32 on the dev games) in miniature.
    other = []
    for i in range(5):
        p = np.zeros((8, 8), dtype=np.int32); p[1, i] = 4
        n = np.zeros((8, 8), dtype=np.int32); n[2, i] = 4
        other.append(Transition(p, "ACTION1", n, 0.0, "", time.time()))
    assert recovered.fn.covers == frozenset({"ACTION3"})
    assert wmm.verify(recovered.fn, trans + other) >= VERIFY_PROMOTE_THRESHOLD, \
        "a correct per-action rule must survive transitions it never claimed"
    # ...but thin evidence INSIDE its own domain still cannot promote it.
    assert wmm.verify(recovered.fn, trans[:2] + other) == 0.0
    # Unknown must stay distinguishable from inert all the way to the planners.
    gz = np.zeros((8, 8), dtype=np.int32); gz[0, 3] = 1
    assert wmm.claims("ACTION3") is True and wmm.claims("ACTION1") is False
    assert wmm.predict_or_none(gz, "ACTION1") is None
    assert np.array_equal(wmm.predict(gz, "ACTION1"), gz)     # array for callers that need one
    assert wmm.predict_or_none(gz, "ACTION3") is not None

    # ---- HUD masking: judge a world model on the WORLD --------------------
    # The rec2 avatar move plus a one-cell timer at (0,0) that ticks every step.
    # Under raw comparison the correct rule is wrong on EVERY sample (one cell
    # off), so the enumerator ends up with no rule at all; under the mask the
    # graph already learns for its state hash, the same rule is exact. That
    # asymmetry was the bug: verify/_score used bare array_equal while
    # StateGraph.changed_masked, PatchWorldModel and the state hash all masked.
    htrans = []
    for i in range(1, 6):
        p = np.zeros((8, 8), dtype=np.int32); p[3, i] = 2; p[7, 7] = 3
        n = np.zeros((8, 8), dtype=np.int32); n[3, i + 1] = 2; n[7, 7] = 3
        p[0, 0] = 5 + (i % 2)                       # timer, ticks under any action
        n[0, 0] = 5 + ((i + 1) % 2)
        htrans.append(Transition(p, "ACTION2", n, 0.0,
                                 enc.encode_transition(p, n, "ACTION2"), time.time()))
    hud = np.zeros((8, 8), dtype=bool); hud[0, 0] = True
    assert masked_equal(htrans[0].prev, htrans[0].prev)
    assert not masked_equal(htrans[0].prev, htrans[0].next)
    assert masked_equal(htrans[0].prev, htrans[0].next, hud) is False   # avatar moved too
    assert masked_diff(np.zeros((2, 2), dtype=np.int32), np.zeros((3, 3), dtype=np.int32)) is None
    assert EnumerativeSynthesizer(enc).synthesize(htrans) is None, \
        "unmasked synthesis should be defeated by the timer -- that is the defect"
    mrec = EnumerativeSynthesizer(enc, mask_fn=lambda: hud).synthesize(htrans)
    assert mrec is not None and mrec.spec[0]["args"].get("color") == 2, \
        f"masked synthesis lost the avatar rule: {mrec.spec if mrec else None}"
    mwmm = WorldModelManager(LocalLLM(model_path="/nonexistent"), enc, mask_fn=lambda: hud)
    assert mwmm.verify(mrec.fn, htrans) == 1.0, \
        f"masked verify of an exact rule: {mwmm.verify(mrec.fn, htrans)}"
    # A tick with no world change is not evidence: masked, it is not a sample at
    # all, so it can neither support a rule nor veto one.
    hud_only = []
    for i in range(5):
        p = np.zeros((8, 8), dtype=np.int32); p[3, 2] = 2; p[0, 0] = 5 + (i % 2)
        n = p.copy(); n[0, 0] = 5 + ((i + 1) % 2)
        hud_only.append(Transition(p, "ACTION4", n, 0.0, "", time.time()))
    # ACTION4 here has NOTHING but masked evidence, so the premise check below
    # judges it raw and it does get a rule -- describing the timer. That is the
    # deliberate side of the trade: the alternative (stay masked, learn nothing)
    # is what erased sp80 ACTION5 and tu93 ACTION2, and the costs are not
    # symmetric. A wrong rule here predicts the HUD correctly and cannot fool the
    # planners, because novelty and dead-action pruning both run off the masked
    # state hash, not off the model; a missing rule loses the model outright.
    # (This case asserted None before the premise check existed.)
    hud_rule = EnumerativeSynthesizer(enc, mask_fn=lambda: hud).synthesize(hud_only)
    assert hud_rule is not None and hud_rule.spec[0]["action"] == "ACTION4"
    assert mwmm.verify(mrec.fn, htrans + hud_only) == 1.0, \
        "HUD-only transitions must not dilute a correct model's score"
    # The guard that matters: once the SAME action also has live evidence, the
    # mask applies again and the HUD-only samples are dropped, so the rule is
    # about the world and not about the timer.
    mixed = [Transition(t.prev, "ACTION4", t.next, 0.0, "", time.time()) for t in htrans]
    mev = EnumerativeSynthesizer(enc, mask_fn=lambda: hud).evidence(hud_only + mixed)
    assert mev["ACTION4"][1] is hud and len(mev["ACTION4"][0]) == len(mixed), \
        "live evidence must re-enable the mask and drop the pure ticks"

    # An action whose EVERY change is inside the mask is the case where the
    # mask's premise fails, so it must be judged RAW -- otherwise masking erases
    # the only signal that action has (measured: sp80 ACTION5 5/5 and tu93
    # ACTION2 4/5 were erased outright). Here ACTION9 only ever moves a sprite
    # that lives in the masked row, so `hud` must not apply to it, while ACTION2
    # (which has live evidence) must still be masked.
    wide = np.zeros((8, 8), dtype=bool); wide[0, :] = True    # a whole HUD row
    inrow = []
    for i in range(1, 6):
        p = np.zeros((8, 8), dtype=np.int32); p[0, i] = 7; p[4, 4] = 3
        n = np.zeros((8, 8), dtype=np.int32); n[0, i + 1] = 7; n[4, 4] = 3
        inrow.append(Transition(p, "ACTION9", n, 0.0, "", time.time()))
    esyn = EnumerativeSynthesizer(enc, mask_fn=lambda: wide)
    ev = esyn.evidence(inrow + htrans)
    assert ev["ACTION9"][1] is None, "an all-masked action must fall back to raw"
    assert len(ev["ACTION9"][0]) == 5, "raw fallback must keep every sample"
    assert ev["ACTION2"][1] is wide, "an action with live evidence stays masked"
    prog = esyn.synthesize(inrow + htrans)
    assert prog is not None and {r["action"] for r in prog.spec} == {"ACTION9", "ACTION2"}, \
        f"expected a rule for both actions, got {prog.spec if prog else None}"
    # ...and the verifier must score each action under the SAME comparison.
    ewmm = WorldModelManager(LocalLLM(model_path="/nonexistent"), enc, mask_fn=lambda: wide)
    assert ewmm.verify(prog.fn, inrow + htrans) == 1.0, \
        f"per-action masks disagree between fitter and verifier: " \
        f"{ewmm.verify(prog.fn, inrow + htrans)}"
    # The graph owns the mask; the world model only reads it through mask_fn.
    # `precise` drops the row/col strips -- on the dev games that component is
    # empty, which is why the strips (and the premise check above) matter.
    assert StateGraph().hud_mask() is None
    assert StateGraph().hud_mask(precise=True) is None

    # Demotion bans the rule so enumeration must find a DIFFERENT hypothesis
    wmm.demote(wmm.active_model)
    assert wmm.active_model is None and len(wmm.banned_rules) > 0
    again = wmm.enumerator.synthesize(trans, wmm.banned_rules)
    assert again is None or again.spec != recovered.spec

    # ExploitGate hysteresis (Issue 4): 5 consecutive hits to enter, <0.5 avg to exit
    gate = ExploitGate()
    for _ in range(4):
        gate.update(1.0)
    assert not gate.should_exploit()
    gate.update(1.0)
    assert gate.should_exploit()
    for _ in range(10):
        gate.update(0.0)
    assert not gate.should_exploit()

    # Epsilon decay (Issue 5)
    assert MyAgent._epsilon_for(0) == EPSILON_START
    assert MyAgent._epsilon_for(10_000) == EPSILON_FLOOR
    assert MyAgent._epsilon_for(50) < MyAgent._epsilon_for(10)

    # MCTS returns a valid action, ACTION6 as one node, no LLM
    wmm2 = WorldModelManager(LocalLLM(model_path="/nonexistent"), enc)
    assert wmm2.try_enumerative(trans) is True
    planner = MCTSPlanner(wmm2, enc, StateCounter())
    act = planner.search(g, ["ACTION1", "ACTION2", "ACTION6"], n_iterations=20)
    assert act.startswith("ACTION")

    # Agent-level checks (stand-in Agent, no LLM/embedder needed)
    agent = MyAgent(game_id="local-test")
    assert agent.llm.available is False or True            # constructed either way

    # parse_action: ACTION6 coordinates, x=col / y=row
    a6 = agent.parse_action("ACTION6_r3_c5")
    assert a6 == GameAction.ACTION6 and agent.action_x == 5 and agent.action_y == 3

    # _valid_actions accepts enums / ints / strings, drops RESET and junk
    class _F: pass
    fr = _F(); fr.available_actions = [1, 2, "ACTION3", "RESET", "JUNK"]
    assert agent._valid_actions(fr) == ["ACTION1", "ACTION2", "ACTION3"]

    # _read_score prefers arcengine >=0.9.3 levels_completed, falls back to score
    fs = _F(); fs.levels_completed = 3
    assert MyAgent._read_score(fs) == 3
    fs2 = _F(); fs2.score = 2
    assert MyAgent._read_score(fs2) == 2
    assert MyAgent._read_score(_F()) == 0

    # _parse_grid handles arcengine's 3D frame (list of animation grids)
    fr2 = _F(); fr2.frame = [np.zeros((4, 4), dtype=int).tolist(), (np.ones((4, 4), dtype=int) * 7).tolist()]
    got = agent._parse_grid(fr2, [fr2])
    assert got is not None and got[0, 0] == 7

    # is_done (WIN): records the CAUSING transition with the REAL final grid
    prev = np.zeros((4, 4), dtype=np.int32); prev[0, 0] = 2
    final = np.zeros((4, 4), dtype=np.int32); final[3, 3] = 2
    agent.last_grid = prev; agent.last_action_str = "ACTION5"
    fwin = _F(); fwin.state = GameState.WIN; fwin.frame = final.tolist(); fwin.levels_completed = 1
    assert agent.is_done([fwin], fwin) is True
    t_last = agent.transitions[-1]
    assert t_last.reward == 1.0 and np.array_equal(t_last.next, final) and t_last.action == "ACTION5"
    n_before = len(agent.transitions)
    agent.is_done([fwin], fwin)                            # second call must not double-record
    assert len(agent.transitions) == n_before

    # is_done (WIN, no grid in frame): retro-tags the previous transition's reward
    agent2 = MyAgent(game_id="local-test-2")
    agent2._record_transition(prev, "ACTION4", final, 0.0)
    agent2.last_grid = final; agent2.last_action_str = "ACTION4"
    fwin2 = _F(); fwin2.state = GameState.WIN               # no .frame / .grid at all
    assert agent2.is_done([fwin2], fwin2) is True
    assert agent2.transitions[-1].reward == 1.0

    # Synth watchdog (Issue 6): stale inflight flag is cleared, generation bumped
    agent.synth_inflight = True
    agent.synth_started_at = time.time() - SYNTH_MAX_INFLIGHT_S - 1
    gen0 = agent.synth_generation
    agent._synth_watchdog()
    assert agent.synth_inflight is False and agent.synth_generation == gen0 + 1

    # --- LLM click-reasoning plan: trigger -> background fill -> ordered consume ---
    # The real 27B never loads in local eval (its path is a Kaggle mount), so this is
    # the ONLY place the click-plan pipeline is exercised: a mock LLM stands in for it.
    class _MockLLM:
        available = True
        def is_ready(self): return True
        def plan_clicks(self, board_text, observations, reactive, hypothesis):
            # returns one in-range and one OUT-of-range coord to prove clamping
            return [(5, 7), (2, 3), (999, -4)]

    ca = MyAgent(game_id="click-plan-test")
    ca.llm = _MockLLM()
    ca.level_steps = 100                                   # past the throttle interval
    ca.cplanner._total_clicks = ca._CLICK_PLAN_MIN_OBS     # enough evidence to reason
    board = np.zeros((32, 32), dtype=np.int32); board[5, 7] = 4; board[2, 3] = 6

    # grid->text keeps pixel coords and marks background as '.'
    txt = ca._grid_to_text(board)
    assert txt.count("\n") == 31 and txt.splitlines()[5][7] != "." and txt.splitlines()[0][0] == "."

    ca._maybe_request_click_plan(board)
    t_wait = time.time()
    while ca._click_inflight and time.time() - t_wait < 5:
        time.sleep(0.01)
    assert not ca._click_inflight and len(ca._click_plan) == 3, "plan must be filled off-thread"
    assert ca.llm_calls_this_level == 1, "a plan request must consume exactly one budget unit"
    # consumed in ORDER, out-of-range coord clamped into [0,31]
    assert ca._next_click_from_plan(board) == "ACTION6_r5_c7"
    assert ca._next_click_from_plan(board) == "ACTION6_r2_c3"
    assert ca._next_click_from_plan(board) == "ACTION6_r31_c0"     # (999,-4) clamped
    assert ca._next_click_from_plan(board) is None                 # exhausted -> caller falls to coverage

    # budget exhausted => no new request fires (coverage floor stays in control)
    ca._discard_click_plan()
    ca.llm_calls_this_level = LLM_CALL_BUDGET_PER_LEVEL
    ca.level_steps += ca._CLICK_PLAN_MIN_INTERVAL + 1
    ca._maybe_request_click_plan(board)
    assert not ca._click_inflight and not ca._click_plan, "no plan when budget is spent"

    # a real (absent) LLM never triggers the pipeline: is_ready() is False locally
    ca.llm = get_local_llm()
    ca.llm_calls_this_level = 0
    ca._maybe_request_click_plan(board)
    assert not ca._click_plan, "LLM-absent must leave coverage as the sole click policy"

    # stale-start guard: a plan reasoned about board A must be DROPPED if the live
    # board has changed by the time it lands (coverage mutated it while generating).
    ca._click_plan = deque([(1, 1), (2, 2)])
    ca._click_src_hash = int(hash((board.shape, board.tobytes())))
    other = board.copy(); other[0, 0] = 9
    assert ca._next_click_from_plan(other) is None and not ca._click_plan, \
        "a plan must not execute from a board it wasn't reasoned about"
    # matching start validates ONCE, then ordered clicks proceed as the board mutates
    ca._click_plan = deque([(1, 1), (2, 2)])
    ca._click_src_hash = int(hash((board.shape, board.tobytes())))
    assert ca._next_click_from_plan(board) == "ACTION6_r1_c1"       # validates on match
    assert ca._next_click_from_plan(other) == "ACTION6_r2_c2"       # board moved -> still runs
    # lethal target in a plan is skipped (never scores below the coverage floor)
    ca.cplanner.update("ACTION6_r3_c3", changed=True, reward=0.0); ca.cplanner.mark_loss("ACTION6_r3_c3")
    ca._click_plan = deque([(3, 3), (7, 7)]); ca._click_src_hash = None
    assert ca._next_click_from_plan(board) == "ACTION6_r7_c7", "a known-lethal target must be skipped"

    # --- Blocker 2: the choose_action (b2) wiring, driven with a mock LLM ---
    class _CF:                                    # minimal click-only frame
        def __init__(self, g, lvl=0, st=None):
            self.frame = g.tolist(); self.available_actions = [6]
            self.levels_completed = lvl
            self.state = st if st is not None else GameState.NOT_FINISHED
    cb = MyAgent(game_id="click-choose-test")
    cb.llm = _MockLLM()
    fg = np.zeros((16, 16), dtype=np.int32); fg[4, 4] = 2
    # (i) a filled plan is emitted BY choose_action as a real ACTION6 click
    cb._click_plan = deque([(8, 8)])
    cb._click_src_hash = int(hash((fg.shape, fg.tobytes())))
    out = cb.choose_action([_CF(fg)], _CF(fg))
    assert out == GameAction.ACTION6 and cb.action_x == 8 and cb.action_y == 8, \
        "choose_action must emit the planned click for a click-only game"
    # (ii) when the ClickPlanner SEARCH is driving, the LLM plan is suppressed
    cs = MyAgent(game_id="click-suppress-test"); cs.llm = _MockLLM()
    _register_buttons(cs.cplanner); cs.cplanner._enter_search(cs.cplanner._detect_buttons())
    assert cs.cplanner.searching
    cs._click_plan = deque([(8, 8)])
    _ = cs.choose_action([_CF(fg)], _CF(fg))
    assert list(cs._click_plan) == [(8, 8)], "search must not consume the LLM plan (one layer at a time)"
    # (iii) a real level-up clears any pending plan
    cl = MyAgent(game_id="click-levelup-test"); cl.llm = _MockLLM()
    cl.last_score = 0; cl.last_grid = fg.copy(); cl.last_action_str = "ACTION6_r8_c8"
    cl._click_plan = deque([(8, 8)])
    _ = cl.choose_action([_CF(fg, lvl=1)], _CF(fg, lvl=1))
    assert not cl._click_plan, "level-up must discard a stale plan from the previous layout"

    print("All isolation tests passed. Agent iteration 6 is ready (offline / in-process).")
