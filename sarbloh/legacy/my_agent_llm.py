"""ARC-AGI-3 agent: LLM-as-coder (Schema / VIGA / WorldCoder), no symbolic planners.

WHAT THIS IS
------------
The agent's only representation of a game is ONE editable Python file it writes
itself, `world_model.py`, containing exactly four functions:

    parse_state(grid: np.ndarray) -> dict     # VIGA layer: pixels -> variables
    render(state: dict) -> np.ndarray         # VIGA verifier: variables -> pixels
    step(state: dict, action: str) -> dict    # WorldCoder layer: the mechanism
    is_goal(state: dict) -> bool              # inferred from reward, never given

State grounding and mechanism discovery live in the SAME file on purpose: when a
prediction fails, the repair may indict either the rule (`step`) or the
representation (`parse_state`), and the model is free to change either.

FOUR TOOLS, NO MORE (see Deliberator.TOOLS, guarded by an assert):
    write_code(prompt)      the LLM rewrites/edits world_model.py + notes.md
    run_backtest()          the three exact checks; returns a POINTED BUG
    run_bfs()               search inside the certified model; zero real actions
    commit_actions([...])   THE ONLY channel from thinking to the game

OUTER LOOP  observe -> deliberate -> execute -> record
INNER LOOP  theorize(write_code) -> certify(run_backtest) -> plan(run_bfs) -> commit

THE CHECKS (pure Python, no LLM, no learned judge)
--------------------------------------------------
CHECK 1  RECONSTRUCTION   render(parse_state(g)) == g, exactly, for EVERY frame.
                          This is our Blender: value equality, no photometric
                          loss, no VLM judge. Free and self-supervised.
CHECK 0  ABSTRACTION      guards CHECK 1 against the degenerate solution
                          `parse_state = lambda g: {"raw": g}`. Non-volatile
                          state may not be a verbatim copy of the grid, and may
                          not be constant across frames that differ.
CHECK 2  REPLAY           step(parse_state(g), a) == parse_state(g') for EVERY
                          recorded transition, compared with volatile keys masked.
CHECK 3  GOAL             is_goal fires on exactly the frames where the score
                          rose, and nowhere else.

Order is 1 -> 0 -> 2 -> 3: reconstruction is the cheapest and most foundational,
CHECK 0 exists only to close a hole in CHECK 1, and replay is meaningless over a
representation that failed either.

`certify()` never returns a bare bool. It returns a Report whose `.bug` names the
check, the Timeline index, the action and the exact differing cell or key --
"transition #47 ACTION2: predicted state['ay']=12, actual 11". A bare False
teaches the model nothing; a counterexample is the entire input to the next
write_code call.

HUD MASKING IS A DECLARATION, NOT A HEURISTIC
---------------------------------------------
Dict keys starting with `_` are VOLATILE. `render` must still reproduce them (so
CHECK 1 stays byte-exact), but CHECK 2 and the execution self-check ignore them.
So the model DECLARES what it cannot predict and our checker holds it to that
declaration -- replacing the hand-written change-rate HUD heuristics, which were
measured wrong. CHECK 0 stops a model from declaring everything volatile.

SCOPED AMENDMENT TO THE "no exec() in agent code" RULE
------------------------------------------------------
This agent's whole premise is executing model-authored Python, so it does exec,
under COMPONENT 3 (Sandbox) only: AST screen with an import whitelist (numpy IS
allowed -- refusing `np` is what made the old JSON-oracle prompt fail), curated
`__builtins__`, no filesystem/network/dunder access, per-call deadline on a
daemon thread with leaked-thread accounting, and size caps on returned states.
Size caps rather than setrlimit: RLIMIT_AS is process-wide and would cap the 27B
model's own allocations.

WHAT IS DELIBERATELY ABSENT
---------------------------
No GoalModel, ProgressModel, MCTS, ClickPlanner, MemoryManager tiers or
PatchWorldModel -- each measured at zero contribution. No hand-written
perception: the observer is a PROMPT, and Python only checks its output. No
bespoke DSL and no JSON hypothesis schema: LLM output is runnable Python. No
per-game branch and no game-id conditional, anywhere.

STEP 0 (PREREQUISITE, run on Kaggle before believing any score)
---------------------------------------------------------------
Qwen3.5/3.6 without flash-linear-attention falls back to a torch path and a call
costs 216-336 s, which makes a multi-turn deliberation unaffordable. Build the
wheels into the dataset the same way sentence-transformers/faiss already are:

    # on any networked machine, matching the Kaggle python/torch/cuda:
    pip download flash-linear-attention causal-conv1d -d wheels/ --no-deps
    # then upload wheels/ into the dataset, and in the notebook:
    !pip install --no-index --find-links=/kaggle/input/<dataset>/wheels \
        flash-linear-attention causal-conv1d

    # measure, do not assume:
    !python /kaggle/working/my_agent.py --bench-llm

`--bench-llm` prints `fla` presence, the model's memory footprint and
tokens/sec. Run it before and after the wheels, and again with the FP8 weights
(~27 GB) against the current bf16 (~53.8 GB), by repointing ARC_LLM_PATH.

ENV KNOBS: ARC_LLM_PATH ARC_WORK_DIR ARC_TURNS ARC_GEN_MAX_S ARC_CERT_MAX_S
ARC_BFS_MAX_S ARC_BFS_NODES ARC_PLAN_MAX ARC_LLM_BUDGET_S ARC_NO_LLM
ARC_NO_OPTIMISM ARC_NO_DISCRIM ARC_VERBOSE ARC_AGENT_SEED

The number to beat is 0.0009 (official arc_agi.scorecard, the only scorer).
"""

import os
import re
import ast
import sys
import json
import time
import math
import random
import hashlib
import threading
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- Dependencies & Fallbacks -------------------------------------------------
try:
    from arcengine import FrameData, GameAction, GameState
    HAS_ARCENGINE = True
except ImportError:
    HAS_ARCENGINE = False

    class GameAction:  # local-test stand-ins; the real enum wins when present
        ACTION1 = "ACTION1"; ACTION2 = "ACTION2"; ACTION3 = "ACTION3"
        ACTION4 = "ACTION4"; ACTION5 = "ACTION5"; ACTION6 = "ACTION6"
        ACTION7 = "ACTION7"; RESET = "RESET"

    class GameState:
        NOT_PLAYED = "NOT_PLAYED"; NOT_FINISHED = "NOT_FINISHED"
        WIN = "WIN"; GAME_OVER = "GAME_OVER"

    FrameData = Any

try:
    from agents.agent import Agent
except ImportError:
    class Agent:
        def __init__(self, *a, **k):
            self.game_id = k.get("game_id", "local")
            self.action_counter = 0

try:
    import torch
except ImportError:
    torch = None


# ============================================================================
# COMPONENT 0: CONFIG
# ============================================================================

def _envs(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _envf(name: str, default: float) -> float:
    try:
        return float(_envs(name, str(default)))
    except ValueError:
        return default


def _envi(name: str, default: int) -> int:
    try:
        return int(float(_envs(name, str(default))))
    except ValueError:
        return default


def _envb(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() not in ("0", "false", "no", "off")


ARC_DATA_ROOT = "/kaggle/input/notebooks/banwait13/datasets-for-arc-agi"
LLM_MODEL_PATH = _envs("ARC_LLM_PATH", "/kaggle/input/datasets/banwait13/models")
WORK_DIR = _envs("ARC_WORK_DIR", "/kaggle/working/arc_agent")

MAX_ACTIONS_DEFAULT = _envi("ARC_MAX_ACTIONS", 1500)
VERBOSE = _envb("ARC_VERBOSE", True)
AGENT_SEED = _envi("ARC_AGENT_SEED", 0)

# --- LLM ---
LLM_MAX_NEW_TOKENS = _envi("ARC_MAX_NEW_TOKENS", 3072)
LLM_GEN_MAX_TIME_S = _envf("ARC_GEN_MAX_S", 200.0)
LLM_TEMPERATURE = _envf("ARC_TEMP", 0.6)
# Each refuted repair raises the temperature. Sixteen repair calls at a fixed 0.6
# returned near-identical files: the same question sampled the same way answers the
# same way, so the extra fifteen generations bought nothing.
REPAIR_TEMP_STEP = _envf("ARC_REPAIR_TEMP_STEP", 0.12)
REPAIR_TEMP_MAX = _envf("ARC_REPAIR_TEMP_MAX", 1.0)
LLM_BUDGET_S = _envf("ARC_LLM_BUDGET_S", 1800.0)   # per game, CEILING (see below)
DISABLE_LLM = _envb("ARC_NO_LLM", False)

# --- the session: every game shares ONE wall clock ---
# LLM_BUDGET_S is a ceiling, not an allowance, and the difference is a whole third
# of the leaderboard. 25 games x 1800s is 12.5 h of generation -- `bench_llm` has
# been printing exactly that number all along -- inside a session that is 9 h long.
# A run that honours the ceiling therefore never reaches the last games of the
# ordering at all, and under RHAE an unrun game scores precisely what a failed one
# does: its levels stay in the denominator and contribute nothing.
# So the writer's allowance is derived, per game, from what the clock has LEFT
# divided by the games still to come. A game that finishes early hands its unused
# time to the others instead of losing it; a game that overruns is paid for by the
# rest, visibly, in the log line new_game() prints.
SESSION_BUDGET_S = _envf("ARC_SESSION_S", 8.5 * 3600.0)
SESSION_GAMES = _envi("ARC_SESSION_GAMES", 25)
# Held back for the write-out (stats, submission). Never handed to a writer.
SESSION_RESERVE_S = _envf("ARC_SESSION_RESERVE_S", 300.0)
# The writer's share of a game's wall clock. The remainder pays for the engine,
# certification, BFS and the sandbox, which are cheap but not free.
LLM_SHARE_OF_GAME = _envf("ARC_LLM_SHARE", 0.75)

# --- crash containment ---
# The harness drives choose_action in a bare loop. An exception there ends the
# game and, on Kaggle, plausibly the run -- so a bug in one game's deliberation
# takes every game after it to zero as well.
CRASH_TRACEBACKS = _envi("ARC_CRASH_TRACEBACKS", 3)   # printed in full, then counted

# --- deliberation ---
# There is NO "deliberate every N actions" knob any more. A timer is what made
# the agent spend real actions to pass the time between rounds: on the measured
# click game that was 1446 of 1500 actions, 98% of them world-no-ops, for zero
# levels completed. Deliberation
# is gated on the EVIDENCE changing instead, and nothing is ever played merely to
# let the clock advance -- see MyAgent.choose_action.
DELIBERATE_MAX_TURNS = _envi("ARC_TURNS", 10)
NO_OPTIMISM = _envb("ARC_NO_OPTIMISM", False)
NO_DISCRIM = _envb("ARC_NO_DISCRIM", False)
PROBE_BATCH = _envi("ARC_PROBE_BATCH", 3)   # informative actions bought per round
NO_INERT_FILTER = _envb("ARC_NO_INERT", False)   # kill switch for the no-op memory
STOP_WHEN_IDLE = _envb("ARC_STOP_WHEN_IDLE", False)  # opt-in: stop instead of fidgeting
IDLE_STOP_AFTER = _envi("ARC_IDLE_STOP", 25)

# --- certification ---
CERT_MAX_S = _envf("ARC_CERT_MAX_S", 20.0)
CERT_MAX_FRAMES = _envi("ARC_CERT_FRAMES", 60)
CERT_MAX_TRANSITIONS = _envi("ARC_CERT_TRANSITIONS", 150)
CERT_MIN_ACCURACY = _envf("ARC_CERT_MIN_ACC", 1.0)    # replay must be exact

# --- sandbox ---
SANDBOX_CALL_S = _envf("ARC_SANDBOX_CALL_S", 3.0)
SANDBOX_MAX_LEAKS = _envi("ARC_SANDBOX_MAX_LEAKS", 6)
SANDBOX_MAX_STATE_ELEMS = _envi("ARC_MAX_STATE_ELEMS", 200_000)
SANDBOX_MAX_CODE_CHARS = _envi("ARC_MAX_CODE_CHARS", 40_000)

# --- planning ---
BFS_MAX_S = _envf("ARC_BFS_MAX_S", 10.0)
BFS_MAX_NODES = _envi("ARC_BFS_NODES", 20_000)
BFS_MAX_DEPTH = _envi("ARC_BFS_DEPTH", 40)
PLAN_MAX_LEN = _envi("ARC_PLAN_MAX", 24)
CLICK_STRIDE = _envi("ARC_CLICK_STRIDE", 8)
CLICK_MAX_CANDIDATES = _envi("ARC_CLICK_MAX", 48)
# How many targets a model must name before the blind stride lattice is dropped
# from the search. The lattice is a sweep of the whole screen: adding it to the
# model's own targets is what made the click branching factor ~48 and capped BFS
# at depth two. See Planner.action_space.
CLICK_MIN_TARGETS = _envi("ARC_CLICK_MIN_TARGETS", 3)

# One deliberation = up to TURNS write_code+certify rounds, plus one optimism
# round, plus one BFS. The watchdog is DERIVED from the parts it guards; a future
# extra repair round therefore cannot silently invalidate it (the assert fails
# loudly instead of the watchdog firing mid-generation on Kaggle only).
DELIBERATE_MAX_S = (DELIBERATE_MAX_TURNS + 1) * (LLM_GEN_MAX_TIME_S + CERT_MAX_S) + BFS_MAX_S
assert DELIBERATE_MAX_S >= (DELIBERATE_MAX_TURNS + 1) * (LLM_GEN_MAX_TIME_S + CERT_MAX_S), \
    "DELIBERATE_MAX_S must cover every round it guards"
assert CERT_MAX_S > SANDBOX_CALL_S, "certification budget must allow at least one sandbox call"

# What one game gets if the session is divided evenly and nothing overruns. This
# is the number the per-game ceiling has to be read against, and the reason the
# allowance is computed at runtime rather than written down here: the even split
# is only correct until the first game finishes at a different time than planned.
PER_GAME_WALL_S = (SESSION_BUDGET_S - SESSION_RESERVE_S) / max(1, SESSION_GAMES)
assert 0.0 < LLM_SHARE_OF_GAME <= 1.0, "the writer's share of a game must be a fraction"
assert SESSION_BUDGET_S > SESSION_RESERVE_S, "the reserve cannot exceed the session"
assert LLM_GEN_MAX_TIME_S <= PER_GAME_WALL_S, (
    "a single generation (%.0fs) may not exceed one game's entire share of the "
    "session (%.0fs) -- raise ARC_SESSION_S, lower ARC_GEN_MAX_S, or run fewer games"
    % (LLM_GEN_MAX_TIME_S, PER_GAME_WALL_S))


class SessionClock:
    """One wall clock, shared by every game in the process.

    It starts when this module is imported, which on Kaggle is once, before the
    first game -- so the minutes the 27B spends loading are charged to the session
    honestly instead of appearing out of nowhere later.

    `share_s()` answers "how much time may this game spend?" as: what is left,
    less the write-out reserve, over the games that still have to run (this one
    included). It is a ceiling that shrinks when a game overruns and grows when
    one finishes early, which is the whole point -- a fixed per-game budget cannot
    do either.
    """

    def __init__(self, total_s: float = SESSION_BUDGET_S,
                 games: int = SESSION_GAMES,
                 reserve_s: float = SESSION_RESERVE_S) -> None:
        self.t0 = time.time()
        self.total_s = float(total_s)
        self.games = max(1, int(games))
        self.reserve_s = max(0.0, float(reserve_s))
        self.games_started = 0

    def elapsed_s(self) -> float:
        return time.time() - self.t0

    def remaining_s(self) -> float:
        """Spendable time left: reserve already deducted, never negative."""
        return max(0.0, self.total_s - self.elapsed_s() - self.reserve_s)

    def games_left(self) -> int:
        """Games still to run, counting the one being played."""
        return max(1, self.games - max(0, self.games_started - 1))

    def start_game(self) -> int:
        self.games_started += 1
        return self.games_started

    def share_s(self) -> float:
        return self.remaining_s() / self.games_left()

    def stats(self) -> Dict[str, Any]:
        return {"elapsed_h": round(self.elapsed_s() / 3600.0, 2),
                "remaining_h": round(self.remaining_s() / 3600.0, 2),
                "total_h": round(self.total_s / 3600.0, 2),
                "game": self.games_started, "of": self.games,
                "share_s": round(self.share_s(), 1)}


_CLOCK: Optional[SessionClock] = None
_CLOCK_LOCK = threading.Lock()


def session_clock() -> SessionClock:
    """The process-wide clock. Created on first use, i.e. at the first game."""
    global _CLOCK
    with _CLOCK_LOCK:
        if _CLOCK is None:
            _CLOCK = SessionClock()
        return _CLOCK


def config_report() -> Dict[str, Any]:
    """The budget the run believes it has, printed once at the first game.

    On Kaggle the log is the only instrument, and every expensive surprise so far
    has been a number nobody printed: 18 tok/s, 1446 filler actions, 12.5 h of
    planned generation in a 9 h session. This is that class of number, up front.
    """
    clock = session_clock()
    ceil_h = LLM_BUDGET_S * SESSION_GAMES / 3600.0
    rep = {
        "session_h": round(SESSION_BUDGET_S / 3600.0, 2),
        "games": SESSION_GAMES,
        "per_game_wall_s": round(PER_GAME_WALL_S, 1),
        "writer_share": LLM_SHARE_OF_GAME,
        "writer_allowance_s": round(min(LLM_BUDGET_S,
                                        PER_GAME_WALL_S * LLM_SHARE_OF_GAME), 1),
        "writer_ceiling_s": LLM_BUDGET_S,
        "ceiling_x_games_h": round(ceil_h, 2),
        "binds": "session clock" if ceil_h > SESSION_BUDGET_S / 3600.0 else "per-game ceiling",
        "max_actions": MAX_ACTIONS_DEFAULT,
        "gen_max_s": LLM_GEN_MAX_TIME_S,
        "gen_max_tokens": LLM_MAX_NEW_TOKENS,
        "deliberate_max_s": round(DELIBERATE_MAX_S, 1),
        "deliberate_turns": DELIBERATE_MAX_TURNS,
        "probe_batch": PROBE_BATCH,
        "click_min_targets": CLICK_MIN_TARGETS,
        "seed": AGENT_SEED,
    }
    log("CONFIG " + json.dumps(rep, sort_keys=True))
    if rep["binds"] == "session clock":
        log("CONFIG note: the per-game ceiling would need %.1f h of generation for "
            "%d games but the session is %.1f h -- the clock will cut each game's "
            "writer to ~%.0fs. This is the intended behaviour, not a warning."
            % (ceil_h, SESSION_GAMES, SESSION_BUDGET_S / 3600.0,
               rep["writer_allowance_s"]))
    return rep

MOVE_ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7")
CLICK_ACTION = "ACTION6"
CLICK_RE = re.compile(r"^ACTION6_r(\d+)_c(\d+)$", re.IGNORECASE)
_CLICK_LOOSE_RE = re.compile(r"^ACTION6[\s_,]*R?(\d+)[\s_,]*C?(\d+)$", re.IGNORECASE)


def norm_action(s: Any) -> str:
    """The single spelling of an action string.

    Case matters in "ACTION6_r12_c30", so upper-casing an action would turn a
    click into an unrecognisable name that silently degrades to a plain button
    press with no coordinates. Every action passes through here exactly once, and
    a loose match also accepts the spellings a model might write by hand.
    """
    t = str(s).strip().upper()
    m = _CLICK_LOOSE_RE.match(t)
    if m:
        return "ACTION6_r%d_c%d" % (int(m.group(1)), int(m.group(2)))
    return t


def log(*parts: Any) -> None:
    if VERBOSE:
        print("[llm-agent]", *parts, flush=True)


# ============================================================================
# COMPONENT 1: CANONICALIZATION & DIFF UTILITIES
# ============================================================================
# A model that returns int8 where we stored int64, or a list where we stored a
# tuple, is not wrong about the game. Comparison therefore goes through one
# normalizer, so the checker refutes mechanisms and not dtypes.

def _norm(obj: Any, depth: int = 0) -> Any:
    if depth > 12:
        return ("deep",)
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (bool, np.bool_)):
        return int(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if math.isnan(f):
            return "nan"
        return round(f, 6)
    if isinstance(obj, np.ndarray):
        a = obj
        if a.dtype == bool:
            a = a.astype(np.int64)
        elif np.issubdtype(a.dtype, np.integer):
            a = a.astype(np.int64)
        elif np.issubdtype(a.dtype, np.floating):
            a = np.round(a.astype(np.float64), 6)
        else:
            return ("arr", tuple(a.shape), repr(a.tolist()))
        return ("arr", tuple(a.shape), a.tobytes())
    if isinstance(obj, dict):
        items = [(_norm(k, depth + 1), _norm(v, depth + 1)) for k, v in obj.items()]
        items.sort(key=lambda kv: repr(kv[0]))
        return ("dict", tuple(items))
    if isinstance(obj, (list, tuple)):
        # tuple == list on purpose: a model that returns [r, c] instead of
        # (r, c) has not made a claim about the game.
        return ("seq", tuple(_norm(x, depth + 1) for x in obj))
    if isinstance(obj, (set, frozenset)):
        return ("set", tuple(sorted((repr(_norm(x, depth + 1)) for x in obj))))
    return ("repr", repr(obj))


def canon(obj: Any) -> str:
    """Stable digest of a normalized structure. Used for state identity."""
    return hashlib.blake2b(repr(_norm(obj)).encode("utf-8", "replace"),
                           digest_size=16).hexdigest()


def mask_volatile(obj: Any, depth: int = 0) -> Any:
    """Drop `_`-prefixed dict keys at every level. The model's declaration of
    what it cannot predict; enforced here rather than guessed from pixel rates."""
    if depth > 12:
        return obj
    if isinstance(obj, dict):
        return {k: mask_volatile(v, depth + 1) for k, v in obj.items()
                if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, (list, tuple)):
        return [mask_volatile(x, depth + 1) for x in obj]
    return obj


def state_size(obj: Any, depth: int = 0) -> int:
    """Element count, for the size cap. Cheap, approximate, bounded."""
    if depth > 12:
        return 1
    if isinstance(obj, np.ndarray):
        return int(obj.size)
    if isinstance(obj, dict):
        return 1 + sum(state_size(v, depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return 1 + sum(state_size(x, depth + 1) for x in obj)
    return 1


def _fmt(v: Any, cap: int = 60) -> str:
    if isinstance(v, np.ndarray):
        s = "array%s" % (tuple(v.shape),)
    else:
        s = repr(v)
    return s if len(s) <= cap else s[:cap] + "..."


def first_difference(pred: Any, actual: Any, path: str = "state") -> str:
    """The pointed part of a pointed bug: where two states first disagree."""
    if isinstance(pred, np.ndarray) or isinstance(actual, np.ndarray):
        pa, aa = np.asarray(pred), np.asarray(actual)
        if pa.shape != aa.shape:
            return "%s: predicted shape %s, actual shape %s" % (path, pa.shape, aa.shape)
        try:
            bad = np.argwhere(pa.astype(np.int64) != aa.astype(np.int64))
        except (ValueError, TypeError):
            return "%s: arrays differ (non-numeric dtype)" % path
        if len(bad) == 0:
            return ""
        cells = []
        for idx in bad[:4]:
            t = tuple(int(x) for x in idx)
            cells.append("%s%s predicted=%s actual=%s"
                         % (path, list(t), pa[t], aa[t]))
        extra = "" if len(bad) <= 4 else " (+%d more cells)" % (len(bad) - 4)
        return "; ".join(cells) + extra
    if isinstance(pred, dict) and isinstance(actual, dict):
        pk, ak = set(pred), set(actual)
        missing = sorted(str(k) for k in ak - pk)
        extra = sorted(str(k) for k in pk - ak)
        if missing:
            return "%s: predicted state is MISSING key(s) %s" % (path, missing[:5])
        if extra:
            return "%s: predicted state has EXTRA key(s) %s" % (path, extra[:5])
        for k in sorted(ak, key=repr):
            if canon(pred[k]) != canon(actual[k]):
                return first_difference(pred[k], actual[k], "%s[%r]" % (path, k))
        return ""
    if isinstance(pred, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(pred) != len(actual):
            return "%s: predicted length %d, actual length %d" % (path, len(pred), len(actual))
        for i, (p, a) in enumerate(zip(pred, actual)):
            if canon(p) != canon(a):
                return first_difference(p, a, "%s[%d]" % (path, i))
        return ""
    if canon(pred) == canon(actual):
        return ""
    return "%s: predicted %s, actual %s" % (path, _fmt(pred), _fmt(actual))


def to_grid(obj: Any) -> Optional[np.ndarray]:
    """Frame -> 2-D int grid. Mirrors the engine's shapes: `.grid` or `.frame`,
    and the last plane when a frame arrives as a stack."""
    arr = obj
    if hasattr(obj, "grid"):
        arr = obj.grid
    elif hasattr(obj, "frame"):
        arr = obj.frame
    if arr is None:
        return None
    try:
        a = np.asarray(arr)
    except Exception:
        return None
    if a.dtype == object or a.size == 0:
        return None
    while a.ndim > 2:
        a = a[-1]
    if a.ndim != 2:
        return None
    return a.astype(np.int64, copy=False)


def diff_cells(a: np.ndarray, b: np.ndarray, cap: int = 8) -> List[Tuple[int, int, int, int]]:
    if a.shape != b.shape:
        return []
    bad = np.argwhere(a != b)
    out = []
    for idx in bad[:cap]:
        r, c = int(idx[0]), int(idx[1])
        out.append((r, c, int(a[r, c]), int(b[r, c])))
    return out


_HEX = "0123456789abcdef"


def grid_block(g: np.ndarray) -> str:
    """Compact text grid: one hex digit per cell (ARC-AGI-3 uses 16 colours)."""
    rows = []
    for r in range(g.shape[0]):
        rows.append("".join(_HEX[int(v) % 16] for v in g[r]))
    return "\n".join(rows)


def grid_labeled(g: np.ndarray) -> str:
    """The labeled view: axis rulers + a colour histogram.

    NOTE: the spec asks for a labeled IMAGE. The Kaggle model is loaded as a
    text-only causal LM, so there is no image channel to attach to; this is the
    labeled rendering that channel would have carried. If a multimodal
    checkpoint is ever pointed at ARC_LLM_PATH, this is the one place to add a
    PNG. Nothing here parses objects -- it is a rendering of raw pixels.
    """
    h, w = g.shape
    tens = "    " + "".join(str((c // 10) % 10) if c % 10 == 0 else " " for c in range(w))
    ones = "    " + "".join(str(c % 10) for c in range(w))
    lines = [ "grid %dx%d (row, col; values are hex colour ids)" % (h, w), tens, ones]
    for r in range(h):
        lines.append("%3d " % r + "".join(_HEX[int(v) % 16] for v in g[r]))
    vals, counts = np.unique(g, return_counts=True)
    hist = ", ".join("%s:%d" % (_HEX[int(v) % 16], int(n))
                     for v, n in sorted(zip(vals.tolist(), counts.tolist()),
                                        key=lambda t: -t[1])[:12])
    lines.append("colour counts: " + hist)
    return "\n".join(lines)


def ensure_dir(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except OSError:
        alt = os.path.join(".", os.path.basename(path.rstrip("/\\")) or "arc_agent")
        try:
            os.makedirs(alt, exist_ok=True)
            return alt
        except OSError:
            return "."


# ============================================================================
# COMPONENT 2: TIMELINE  (append-only; the memory advantage over a human)
# ============================================================================

@dataclass
class Transition:
    idx: int
    prev_key: str
    action: str
    next_key: str
    reward: float
    terminal: bool
    level: int
    is_reset: bool


class Timeline:
    """Every real interaction, forever, in order.

    Append-only by contract: `record` is the only mutator, frames are interned
    and never rewritten, and nothing else in this file edits `steps`. That is
    what makes certification possible -- context compaction can wipe the
    conversation, but the evidence survives, so a model promoted on turn 30 is
    still tested against transition 3.
    """

    def __init__(self) -> None:
        self._steps: List[Transition] = []
        self._frames: Dict[str, np.ndarray] = {}
        self._frame_level: Dict[str, int] = {}
        self._frame_order: List[str] = []
        # frame key -> action -> (times tried, times the grid changed). This is
        # the no-op memory: in the measured run 98% of actions changed nothing and
        # 100% of
        # them came from a frame already seen five or more times, so "I have
        # pressed this here and it did nothing" is the cheapest fact in the run.
        self._effects: Dict[str, Dict[str, Tuple[int, int]]] = {}

    # --- writing ---
    def _intern(self, grid: np.ndarray, level: int) -> str:
        k = canon(grid)
        if k not in self._frames:
            self._frames[k] = np.array(grid, dtype=np.int64, copy=True)
            self._frames[k].setflags(write=False)
            self._frame_level[k] = level
            self._frame_order.append(k)
        return k

    def record(self, prev_grid: np.ndarray, action: str, next_grid: np.ndarray,
               reward: float, terminal: bool, level: int) -> Transition:
        is_reset = str(action).upper().startswith("RESET")
        t = Transition(idx=len(self._steps),
                       prev_key=self._intern(prev_grid, level),
                       action=str(action),
                       next_key=self._intern(next_grid, level + (1 if reward > 0 else 0)),
                       reward=float(reward), terminal=bool(terminal),
                       level=int(level), is_reset=is_reset)
        self._steps.append(t)
        if not t.is_reset:
            tries, changes = self._effects.setdefault(t.prev_key, {}).get(t.action, (0, 0))
            self._effects[t.prev_key][t.action] = (
                tries + 1, changes + (1 if t.next_key != t.prev_key else 0))
        return t

    def note_frame(self, grid: np.ndarray, level: int) -> str:
        """Register a frame with no transition attached (e.g. the first one)."""
        return self._intern(grid, level)

    # --- reading ---
    def __len__(self) -> int:
        return len(self._steps)

    @property
    def steps(self) -> Tuple[Transition, ...]:
        return tuple(self._steps)          # copy: callers cannot rewrite history

    def grid(self, key: str) -> np.ndarray:
        return self._frames[key]

    def pair(self, t: Transition) -> Tuple[np.ndarray, np.ndarray]:
        return self._frames[t.prev_key], self._frames[t.next_key]

    def frames(self, level: Optional[int] = None) -> List[np.ndarray]:
        keys = self._frame_order if level is None else \
            [k for k in self._frame_order if self._frame_level[k] == level]
        return [self._frames[k] for k in keys]

    def for_level(self, level: int) -> List[Transition]:
        return [t for t in self._steps if t.level == level]

    def transitions_for_check(self, level: Optional[int]) -> List[Transition]:
        """Modelled transitions only: RESET is not a mechanism of the game."""
        src = self._steps if level is None else self.for_level(level)
        return [t for t in src if not t.is_reset]

    # --- the no-op memory (what acting HERE is already known to buy) ----------
    def effects_at(self, grid: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """action -> (tried, changed) for this exact frame."""
        return dict(self._effects.get(canon(grid), {}))

    def inert(self, grid: np.ndarray, action: str) -> bool:
        """Has this action been played from THIS frame and never moved the world?

        Only ever a PREFERENCE, never a ban: a game with hidden state (a counter
        the pixels do not show) can answer differently the sixth time, so the
        probe de-prioritises inert actions and still plays one when nothing else
        is left. What it must not do is spend the budget re-pressing a button
        whose local answer is already recorded.

        The key is the EXACT frame, so a game with a live HUD counter gives every
        step a fresh key and this never fires -- which is visible in the report as
        `unique frames ~= actions` and is the signal to key on the certified
        model's own volatile-masked state instead. The measured click game produced
        14 unique frames in 1500 actions, so the exact frame is the right key there.
        """
        if NO_INERT_FILTER:
            return False
        tries, changes = self._effects.get(canon(grid), {}).get(str(action), (0, 0))
        return tries > 0 and changes == 0

    def novelty(self, grid: np.ndarray, action: str) -> Tuple[int, int]:
        """Sort key for buying evidence: untried first, then never-inert, then
        least-tried. Lower is better."""
        tries, changes = self._effects.get(canon(grid), {}).get(str(action), (0, 0))
        if tries == 0:
            return (0, 0)
        if NO_INERT_FILTER or changes > 0:
            return (1, tries)
        return (2, tries)

    # --- sampling ---
    def certify_sample(self, level: Optional[int],
                       limit: int = CERT_MAX_TRANSITIONS) -> List[Transition]:
        """Oldest-first stride, with every reward/terminal transition forced in.

        Deliberately NOT the most recent k: a model fitted to the recent past
        cannot be refuted by the recent past. Scoring `claimed[-5:]` is exactly
        how recency-overfit models used to certify at 1.00 and then collapse.
        """
        ts = self.transitions_for_check(level)
        if len(ts) <= limit:
            return ts
        forced = {t.idx for t in ts if t.reward > 0 or t.terminal}
        room = max(1, limit - len(forced))
        stride = max(1, len(ts) // room)
        picked = {t.idx for t in ts[::stride][:room]} | forced
        return [t for t in ts if t.idx in picked][:limit]

    def frame_sample(self, level: Optional[int],
                     limit: int = CERT_MAX_FRAMES) -> List[np.ndarray]:
        fs = self.frames(level)
        if len(fs) <= limit:
            return fs
        stride = max(1, len(fs) // limit)
        return fs[::stride][:limit]

    # --- evidence for the prompt ---
    def volatile_cell_hint(self, level: Optional[int], top: int = 12) -> str:
        """Cells that change under many different actions. INFORMATION for the
        prompt only -- never a gate. Rate-threshold HUD masks were measured
        wrong; the model must declare its own volatile keys and CHECK 1/2 judge
        that declaration."""
        ts = self.transitions_for_check(level)
        if not ts:
            return "(no transitions yet)"
        counts: Dict[Tuple[int, int], int] = {}
        for t in ts:
            a, b = self.pair(t)
            if a.shape != b.shape:
                continue
            for r, c in np.argwhere(a != b)[:400]:
                counts[(int(r), int(c))] = counts.get((int(r), int(c)), 0) + 1
        if not counts:
            return "(no cell ever changed)"
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
        n = len(ts)
        return ", ".join("(%d,%d) in %d/%d" % (r, c, k, n) for (r, c), k in ranked)

    def action_effect_summary(self, level: Optional[int], per_action: int = 2) -> str:
        ts = self.transitions_for_check(level)
        if not ts:
            return "(no transitions yet)"
        by_action: Dict[str, List[Transition]] = {}
        for t in ts:
            by_action.setdefault(t.action, []).append(t)
        out = []
        for a in sorted(by_action):
            group = by_action[a]
            changed = 0
            examples = []
            for t in group:
                pa, pb = self.pair(t)
                cells = diff_cells(pa, pb, cap=6)
                if cells:
                    changed += 1
                if len(examples) < per_action and cells:
                    examples.append("#%d: " % t.idx + ", ".join(
                        "(%d,%d) %s->%s" % (r, c, _HEX[v0 % 16], _HEX[v1 % 16])
                        for r, c, v0, v1 in cells))
            line = "%s x%d, changed the grid %d/%d" % (a, len(group), changed, len(group))
            if examples:
                line += " | " + " ; ".join(examples)
            out.append(line)
        return "\n".join(out)

    def reward_summary(self) -> str:
        rs = [t for t in self._steps if t.reward > 0]
        if not rs:
            return "(the score has never gone up yet)"
        return "; ".join("#%d after %s (level %d)" % (t.idx, t.action, t.level) for t in rs[:8])


# ============================================================================
# COMPONENT 3: SANDBOX  (the scoped exec amendment lives here and nowhere else)
# ============================================================================

class SandboxError(Exception):
    """Candidate code misbehaved: rejected, raised, timed out or grew too big.

    Carries a LABEL as well as a message. A run once reported 54 of 63 rejections
    as a bare `REJECTED`, which is un-diagnosable: a syntax error, a forbidden
    import, a missing function and a poisoned sandbox are four different problems
    with four different fixes, and the histogram that was supposed to localise the
    failure lumped them into one bucket. Every raise site names its own kind.
    """

    def __init__(self, message: str, label: str = "REJECTED") -> None:
        super().__init__(message)
        self.label = label


class SandboxTimeout(SandboxError):
    def __init__(self, message: str, label: str = "TIMEOUT") -> None:
        super().__init__(message, label)


class Deadline:
    """Shared clock. Checked between transitions and between BFS nodes, so
    well-behaved-but-slow code is stopped cleanly rather than abandoned."""

    def __init__(self, seconds: float) -> None:
        self.t0 = time.time()
        self.seconds = float(seconds)

    def remaining(self) -> float:
        return self.seconds - (time.time() - self.t0)

    def expired(self) -> bool:
        return self.remaining() <= 0.0


# Modules the model may import. numpy is IN: the old JSON-oracle prompt told the
# model "imports are not allowed; np", i.e. refused the one tool it actually
# knows how to use, and 0% of its proposals were ever accepted.
IMPORT_WHITELIST = {
    "numpy", "math", "itertools", "collections", "heapq", "copy",
    "functools", "operator", "bisect", "dataclasses", "typing", "re",
}

# Names that make an escape possible even without imports.
NAME_BLACKLIST = {
    "eval", "exec", "compile", "open", "input", "__import__", "globals",
    "locals", "vars", "breakpoint", "exit", "quit", "memoryview",
    "help", "license", "credits", "reload",
}

def _safe_import(name: str, globals_: Any = None, locals_: Any = None,
                 fromlist: Any = (), level: int = 0) -> Any:
    """The runtime half of the import whitelist. The AST screen rejects a
    forbidden `import` statement before the code ever runs; this stops the one it
    cannot see -- an import reached indirectly at call time."""
    root = str(name).split(".")[0]
    if level != 0 or root not in IMPORT_WHITELIST:
        raise ImportError("import of %r is not permitted; permitted modules are %s"
                          % (name, sorted(IMPORT_WHITELIST)))
    return __import__(name, globals_, locals_, fromlist or (), level)


SAFE_BUILTINS = {
    "__import__": _safe_import,
    "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool, "bytes": bytes,
    "callable": callable, "chr": chr, "complex": complex, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "format": format, "frozenset": frozenset, "getattr": getattr, "hasattr": hasattr,
    "hash": hash, "hex": hex, "id": id, "int": int, "isinstance": isinstance,
    "issubclass": issubclass, "iter": iter, "len": len, "list": list, "map": map,
    "max": max, "min": min, "next": next, "object": object, "oct": oct, "ord": ord,
    "pow": pow, "print": print, "range": range, "repr": repr, "reversed": reversed,
    "round": round, "set": set, "setattr": setattr, "slice": slice, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "type": type, "zip": zip,
    "True": True, "False": False, "None": None,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError, "ZeroDivisionError": ZeroDivisionError,
    "AttributeError": AttributeError, "StopIteration": StopIteration,
    "RuntimeError": RuntimeError, "NotImplementedError": NotImplementedError,
    "__build_class__": __build_class__, "__name__": "world_model",
}

REQUIRED_DEFS = ("parse_state", "render", "step", "is_goal")


def screen_code(src: str) -> Optional[Tuple[str, str]]:
    """Static screen. Returns (LABEL, reason), or None if the code may run.

    The label is the diagnosis: a reader looking at `{SCREEN SYNTAX: 40}` knows to
    fix the writer's output format, and one looking at `{SCREEN MISSING DEF: 40}`
    knows the prompt's contract is not landing. A bare `REJECTED: 54` says only
    that something went wrong 54 times.
    """
    if not src or not src.strip():
        return ("SCREEN EMPTY", "the reply contained no code at all")
    if len(src) > SANDBOX_MAX_CODE_CHARS:
        return ("SCREEN SIZE",
                "code is %d chars, limit is %d" % (len(src), SANDBOX_MAX_CODE_CHARS))
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return ("SCREEN SYNTAX line %s" % e.lineno,
                "SyntaxError: %s (line %s: %r)"
                % (e.msg, e.lineno, (e.text or "").strip()[:120]))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in IMPORT_WHITELIST:
                    return ("SCREEN IMPORT %s" % a.name.split(".")[0],
                            "import of %r is not permitted; permitted modules are %s"
                            % (a.name, sorted(IMPORT_WHITELIST)))
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in IMPORT_WHITELIST:
                return ("SCREEN IMPORT %s" % (root or "relative"),
                        "from-import of %r is not permitted; permitted modules are %s"
                        % (node.module, sorted(IMPORT_WHITELIST)))
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            return ("SCREEN GLOBAL",
                    "global/nonlocal is not permitted (the model must be a pure function)")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return ("SCREEN DUNDER %s" % node.attr,
                        "attribute %r is not permitted" % node.attr)
        elif isinstance(node, ast.Name):
            if node.id in NAME_BLACKLIST:
                return ("SCREEN NAME %s" % node.id,
                        "use of %r is not permitted" % node.id)
            if node.id.startswith("__") and node.id.endswith("__") and node.id != "__name__":
                return ("SCREEN DUNDER %s" % node.id, "name %r is not permitted" % node.id)
    missing = [d for d in REQUIRED_DEFS
               if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                          and n.name == d for n in tree.body)]
    if missing:
        return ("SCREEN MISSING DEF %s" % ",".join(missing),
                "the file must define %s at top level; missing: %s" % (
                    ", ".join(REQUIRED_DEFS), ", ".join(missing)))
    return None


class Sandbox:
    """Runs candidate functions under a per-call deadline on a daemon thread.

    Python cannot kill a thread, so a call that blows its deadline is ABANDONED
    and counted. After SANDBOX_MAX_LEAKS abandoned threads the sandbox stops
    accepting work, because leaked spinners would otherwise starve the run --
    a pathological candidate must not stall the game.
    """

    def __init__(self) -> None:
        self.leaked = 0
        self.calls = 0
        self.time_s = 0.0

    @property
    def poisoned(self) -> bool:
        return self.leaked >= SANDBOX_MAX_LEAKS

    def compile_module(self, src: str) -> Dict[str, Any]:
        screened = screen_code(src)
        if screened is not None:
            label, reason = screened
            raise SandboxError(reason, label)
        env: Dict[str, Any] = {"__builtins__": dict(SAFE_BUILTINS), "np": np, "numpy": np}
        code = compile(src, "<world_model>", "exec")
        self.guarded(lambda: exec(code, env), timeout=SANDBOX_CALL_S * 2, what="module import")
        for d in REQUIRED_DEFS:
            if not callable(env.get(d)):
                raise SandboxError("%s is not callable after import" % d,
                                   "CONTRACT %s not callable" % d)
        return env

    def guarded(self, fn: Callable[[], Any], timeout: float = SANDBOX_CALL_S,
                what: str = "call") -> Any:
        if self.poisoned:
            # Not a fact about THIS candidate: an earlier one leaked threads and
            # the sandbox is closed. Labelling it as a plain rejection makes the
            # histogram blame every later model for the first one's spinner.
            raise SandboxError("sandbox disabled after %d leaked threads" % self.leaked,
                               "SANDBOX POISONED")
        box: Dict[str, Any] = {}
        t0 = time.time()

        def runner() -> None:
            try:
                box["v"] = fn()
            except BaseException as e:                     # noqa: BLE001 - candidate code
                box["e"] = e
                box["tb"] = traceback.format_exc(limit=4)

        th = threading.Thread(target=runner, daemon=True, name="sandbox")
        th.start()
        th.join(max(0.05, timeout))
        self.calls += 1
        self.time_s += time.time() - t0
        if th.is_alive():
            self.leaked += 1
            raise SandboxTimeout("%s exceeded %.2fs (abandoned; %d leaked so far)"
                                 % (what, timeout, self.leaked),
                                 "TIMEOUT %s" % what)
        if "e" in box:
            e = box["e"]
            raise SandboxError("%s raised %s: %s" % (what, type(e).__name__, e),
                               "RAISED %s in %s" % (type(e).__name__, what))
        return box.get("v")


class CandidateModel:
    """A compiled world_model.py: the four functions plus their provenance."""

    def __init__(self, code: str, sandbox: Sandbox, notes: str = "",
                 origin: str = "llm") -> None:
        self.code = code
        self.notes = notes
        self.origin = origin
        self.sandbox = sandbox
        self.env = sandbox.compile_module(code)
        self.digest = canon(code)
        self.report: Optional["Report"] = None

    # each call is deadline-guarded and size-capped
    def _call(self, name: str, *args: Any) -> Any:
        fn = self.env[name]
        out = self.sandbox.guarded(lambda: fn(*args), what="%s()" % name)
        if state_size(out) > SANDBOX_MAX_STATE_ELEMS:
            raise SandboxError("%s() returned %d elements, cap is %d"
                               % (name, state_size(out), SANDBOX_MAX_STATE_ELEMS),
                               "CONTRACT %s too big" % name)
        return out

    def parse(self, grid: np.ndarray) -> Any:
        g = np.array(grid, dtype=np.int64, copy=True)
        out = self._call("parse_state", g)
        if not isinstance(out, dict):
            raise SandboxError("parse_state must return a dict, got %s" % type(out).__name__,
                               "CONTRACT parse_state -> %s" % type(out).__name__)
        return out

    def render(self, state: Any) -> np.ndarray:
        out = self._call("render", state)
        a = np.asarray(out)
        if a.ndim != 2:
            raise SandboxError("render must return a 2-D grid, got shape %s" % (a.shape,),
                               "CONTRACT render ndim=%d" % a.ndim)
        return a.astype(np.int64, copy=False)

    def step(self, state: Any, action: str) -> Any:
        out = self._call("step", state, action)
        if not isinstance(out, dict):
            raise SandboxError("step must return a dict, got %s" % type(out).__name__,
                               "CONTRACT step -> %s" % type(out).__name__)
        return out

    def is_goal(self, state: Any) -> bool:
        out = self._call("is_goal", state)
        # np.False_ is not False: normalize before anything believes it.
        return bool(out)


# ============================================================================
# COMPONENT 4: CERTIFIER  (three exact checks, one pointed bug)
# ============================================================================

@dataclass
class Report:
    ok: bool
    check: str = ""
    bug: str = ""
    accuracy: float = 0.0            # CHECK 2 exact-replay accuracy
    n_frames: int = 0
    n_transitions: int = 0
    suspect_representation: bool = False
    stats: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        if self.ok:
            return ("CERTIFIED: reconstruction %d/%d frames, replay %d/%d transitions, "
                    "goal check passed" % (self.n_frames, self.n_frames,
                                           self.n_transitions, self.n_transitions))
        return "%s FAILED: %s" % (self.check or "CHECK", self.bug)


class Certifier:
    """Pure Python. No LLM, no learned judge, no thresholds to tune.

    Every candidate world model passes through here, whatever wrote it. The
    output is a counterexample, because a counterexample is the only thing the
    next write_code call can act on.
    """

    def __init__(self, timeline: Timeline) -> None:
        self.timeline = timeline
        self.check2_fail_streak = 0
        self.n_certified = 0
        self.n_rejected = 0
        # label -> count, and one worked example per label. Printed in stats(), so
        # the dominant failure mode is legible from the agent's own report instead
        # of needing the harness to reconstruct it.
        self.reject_kinds: Dict[str, int] = {}
        self.reject_examples: Dict[str, str] = {}

    def note_rejection(self, rep: Report) -> None:
        """Every rejection, whoever produced it, lands in the histogram.

        Compilation failures never reached `certify()` at all, so they were
        counted only by the harness -- as a single bare `REJECTED` bucket that hid
        the difference between a syntax error and a poisoned sandbox.
        """
        self.n_rejected += 1
        key = rep.check or "UNLABELLED"
        self.reject_kinds[key] = self.reject_kinds.get(key, 0) + 1
        self.reject_examples.setdefault(key, rep.bug[:300])

    def certify(self, model: CandidateModel, level: Optional[int] = None,
                budget_s: float = CERT_MAX_S) -> Report:
        dl = Deadline(budget_s)
        frames = self.timeline.frame_sample(level)
        trans = self.timeline.certify_sample(level)
        rep = Report(ok=False, n_frames=len(frames), n_transitions=len(trans))
        # The checks short-circuit, but the tally must not: every exit path runs
        # the accounting below, so a rejection is always counted.
        try:
            self._run_checks(model, frames, trans, rep, dl)
        except SandboxError as e:
            rep.ok = False
            rep.check = getattr(e, "label", "CRASH")
            rep.bug = str(e)
        if rep.ok:
            self.n_certified += 1
            self.check2_fail_streak = 0
        else:
            self.note_rejection(rep)
        model.report = rep
        return rep

    def _run_checks(self, model: CandidateModel, frames: List[np.ndarray],
                    trans: List[Transition], rep: Report, dl: Deadline) -> None:
        parsed = self._check1_reconstruction(model, frames, rep, dl)
        if not rep.ok:
            return
        self._check0_abstraction(model, frames, parsed, rep)
        if not rep.ok:
            return
        self._check2_replay(model, trans, rep, dl)
        if not rep.ok:
            return
        self._check3_goal(model, trans, rep, dl)

    # --- CHECK 1: RECONSTRUCTION (VIGA) -------------------------------------
    def _check1_reconstruction(self, model: CandidateModel, frames: List[np.ndarray],
                               rep: Report, dl: Deadline) -> List[Any]:
        parsed: List[Any] = []
        for i, g in enumerate(frames):
            if dl.expired():
                rep.ok, rep.check = False, "TIMEOUT"
                rep.bug = ("CHECK 1 ran out of time after %d/%d frames -- parse_state/render "
                           "must be fast (simple numpy, no search)" % (i, len(frames)))
                return parsed
            st = model.parse(g)
            parsed.append(st)
            out = model.render(st)
            if out.shape != g.shape:
                rep.ok, rep.check = False, "CHECK 1 RECONSTRUCTION"
                rep.bug = ("frame %d: render(parse_state(grid)) has shape %s but the grid is %s"
                           % (i, out.shape, g.shape))
                return parsed
            if not np.array_equal(out, g):
                cells = diff_cells(out, g)
                rep.ok, rep.check = False, "CHECK 1 RECONSTRUCTION"
                rep.bug = ("frame %d: render(parse_state(grid)) != grid at %d cell(s). "
                           "First: %s. Your state is DROPPING or MISPLACING what draws "
                           "these cells; add the missing variable(s) to parse_state and "
                           "draw them in render (a `_`-prefixed key is fine if you cannot "
                           "predict it)." % (i, int(np.sum(out != g)),
                                             ", ".join("(%d,%d) rendered=%s actual=%s"
                                                       % (r, c, _HEX[v0 % 16], _HEX[v1 % 16])
                                                       for r, c, v0, v1 in cells)))
                rep.stats["reconstructed_frames"] = i
                return parsed
        rep.ok = True
        rep.stats["reconstructed_frames"] = len(frames)
        return parsed

    # --- CHECK 0: ABSTRACTION (closes the degenerate hole) -------------------
    def _check0_abstraction(self, model: CandidateModel, frames: List[np.ndarray],
                            parsed: List[Any], rep: Report) -> None:
        """`parse_state = lambda g: {"raw": g}` with `render = lambda s: s["raw"]`
        satisfies CHECK 1 perfectly and abstracts nothing; declaring everything
        volatile would then satisfy CHECK 2 vacuously. Two conditions:
        (a) no non-volatile value may be a verbatim copy of the grid;
        (b) the non-volatile part must distinguish frames that differ."""
        if not frames:
            rep.ok = True
            return
        for i, (g, st) in enumerate(zip(frames, parsed)):
            gk = canon(g)
            for k, v in st.items():
                if isinstance(k, str) and k.startswith("_"):
                    continue
                if isinstance(v, np.ndarray) and v.shape == g.shape and canon(v) == gk:
                    rep.ok, rep.check = False, "CHECK 0 ABSTRACTION"
                    rep.bug = ("frame %d: state[%r] is a verbatim copy of the grid. "
                               "parse_state must ABSTRACT -- name the objects, their "
                               "positions, colours and any counters; do not store the "
                               "pixels. render() rebuilds the pixels from those "
                               "variables." % (i, k))
                    return
        masked = [canon(mask_volatile(st)) for st in parsed]
        raw = [canon(g) for g in frames]
        if len(set(raw)) > 1 and len(set(masked)) == 1:
            rep.ok, rep.check = False, "CHECK 0 ABSTRACTION"
            rep.bug = ("every recorded frame collapses to the SAME non-volatile state, "
                       "yet the frames differ. You have declared the whole game volatile "
                       "(`_`-prefixed), so replay would pass without predicting anything. "
                       "Move whatever actually changes -- positions, counts, which cells "
                       "are filled -- into non-volatile keys, and keep `_` for timers and "
                       "score digits only.")
            return
        rep.ok = True
        rep.stats["distinct_states"] = len(set(masked))
        rep.stats["distinct_frames"] = len(set(raw))

    # --- CHECK 2: REPLAY (WorldCoder) ---------------------------------------
    def _check2_replay(self, model: CandidateModel, trans: List[Transition],
                       rep: Report, dl: Deadline) -> None:
        if not trans:
            rep.ok = True
            rep.accuracy = 1.0
            rep.stats["replayed"] = 0
            return
        good = 0
        for n, t in enumerate(trans):
            if dl.expired():
                rep.ok, rep.check = False, "TIMEOUT"
                rep.bug = ("CHECK 2 ran out of time after %d/%d transitions -- step() "
                           "must be fast (no search inside the model)" % (n, len(trans)))
                return
            g0, g1 = self.timeline.pair(t)
            s0 = model.parse(g0)
            s1 = model.parse(g1)
            pred = model.step(s0, t.action)
            pm, am = mask_volatile(pred), mask_volatile(s1)
            if canon(pm) == canon(am):
                good += 1
                continue
            rep.accuracy = good / float(len(trans))
            rep.ok, rep.check = False, "CHECK 2 REPLAY"
            self.check2_fail_streak += 1
            diff = first_difference(pm, am) or "(states differ only in volatile keys?)"
            note = ""
            if self.check2_fail_streak >= 2:
                # The diagnostic rule: repeated replay failures over a
                # reconstruction that passes usually mean the variables are
                # wrong, not the rule. Say so, explicitly.
                note = (" SUSPECT THE STATE REPRESENTATION, NOT THE RULE: CHECK 1 passes, "
                        "so your variables can redraw the screen, but they may not be the "
                        "variables the game actually updates. Consider re-grounding "
                        "parse_state (different objects, relative instead of absolute "
                        "coordinates, an extra hidden counter) before patching step again.")
            rep.suspect_representation = self.check2_fail_streak >= 2
            rep.bug = ("transition #%d, action %s: step(parse_state(prev)) != "
                       "parse_state(next). %s. (%d/%d earlier transitions did match.)%s"
                       % (t.idx, t.action, diff, good, len(trans), note))
            rep.stats["replayed"] = good
            return
        rep.accuracy = 1.0
        rep.ok = rep.accuracy >= CERT_MIN_ACCURACY
        rep.stats["replayed"] = good
        if not rep.ok:
            rep.check = "CHECK 2 REPLAY"
            rep.bug = "replay accuracy %.3f below required %.3f" % (rep.accuracy, CERT_MIN_ACCURACY)

    # --- CHECK 3: GOAL ------------------------------------------------------
    def _check3_goal(self, model: CandidateModel, trans: List[Transition],
                     rep: Report, dl: Deadline) -> None:
        """is_goal must fire on exactly the frames where the score rose. With no
        reward observed yet it must simply not fire on everything seen so far --
        otherwise BFS would 'reach the goal' at depth 0, forever."""
        wins = [t for t in trans if t.reward > 0]
        losses = [t for t in trans if t.reward <= 0]
        for t in wins:
            if dl.expired():
                break
            st = model.parse(self.timeline.pair(t)[1])
            if not model.is_goal(st):
                rep.ok, rep.check = False, "CHECK 3 GOAL"
                rep.bug = ("transition #%d (action %s) SCORED a point, but is_goal() is "
                           "False on the state after it. is_goal must be True exactly when "
                           "the level has just been completed -- work out what is true of "
                           "that state and of no other." % (t.idx, t.action))
                return
        neg = losses[::max(1, len(losses) // 40)][:40]
        for t in neg:
            if dl.expired():
                break
            st = model.parse(self.timeline.pair(t)[1])
            if model.is_goal(st):
                rep.ok, rep.check = False, "CHECK 3 GOAL"
                rep.bug = ("transition #%d (action %s) did NOT score, yet is_goal() is True "
                           "on the state after it. is_goal is too loose: it would make the "
                           "planner stop where nothing was won." % (t.idx, t.action))
                return
        rep.ok = True
        rep.stats["goal_positives"] = len(wins)
        rep.stats["goal_negatives"] = len(neg)


# ============================================================================
# COMPONENT 5: PLANNER  (run_bfs -- search inside the model, zero real actions)
# ============================================================================

@dataclass
class Plan:
    actions: List[str] = field(default_factory=list)
    expected: List[str] = field(default_factory=list)   # masked canon after each action
    reached: bool = False
    # True only when the search ran OUT OF STATES rather than out of budget. The
    # difference decides what the optimism prompt should ask for: exhaustion is a
    # proof that is_goal is unsatisfiable under these rules, while a budget cut-off
    # says nothing about is_goal at all.
    exhausted: bool = False
    reason: str = ""
    nodes: int = 0
    distinct: int = 0
    depth: int = 0
    seconds: float = 0.0

    def __bool__(self) -> bool:
        return self.reached and bool(self.actions)

    def summary(self) -> str:
        return ("plan len=%d reached=%s nodes=%d distinct=%d depth=%d %.2fs %s"
                % (len(self.actions), self.reached, self.nodes, self.distinct,
                   self.depth, self.seconds, self.reason))


def collect_coords(obj: Any, h: int, w: int, depth: int = 0,
                   out: Optional[List[Tuple[int, int]]] = None) -> List[Tuple[int, int]]:
    """Click targets taken from the model's OWN state: any (row, col)-looking
    pair inside it. Generic -- there is no game-id branch here and no
    hand-written notion of what a button is."""
    if out is None:
        out = []
    if depth > 8 or len(out) > 4 * CLICK_MAX_CANDIDATES:
        return out
    if isinstance(obj, dict):
        keys = {str(k).lower(): k for k in obj}
        for a, b in (("y", "x"), ("row", "col"), ("r", "c"), ("ry", "rx")):
            if a in keys and b in keys:
                try:
                    r, c = int(obj[keys[a]]), int(obj[keys[b]])
                    if 0 <= r < h and 0 <= c < w:
                        out.append((r, c))
                except (TypeError, ValueError):
                    pass
        for v in obj.values():
            collect_coords(v, h, w, depth + 1, out)
        return out
    if isinstance(obj, (list, tuple)):
        if (len(obj) == 2 and all(isinstance(v, (int, np.integer)) and
                                  not isinstance(v, bool) for v in obj)):
            r, c = int(obj[0]), int(obj[1])
            if 0 <= r < h and 0 <= c < w:
                out.append((r, c))
        for v in obj:
            collect_coords(v, h, w, depth + 1, out)
    return out


class Planner:
    """Breadth-first search through `step`, stopping at `is_goal`.

    Every node costs zero scored actions -- this is the whole point under RHAE,
    which squares the action ratio: thinking is free, button presses are not.
    """

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox
        self.total_nodes = 0
        self.total_s = 0.0

    def action_space(self, model: CandidateModel, state: Any,
                     available: List[str], shape: Tuple[int, int]) -> List[str]:
        """The branching factor of the search, and therefore its reach.

        DETECTED BUTTONS **OR** THE LATTICE, never both. The coordinates the model
        names in its own state are the things it believes are there; a stride
        lattice is a blind sweep of the screen. Appending the sweep to the buttons
        used to give a 64x64 click game ~48 children per node, so a 20k-node
        budget bought a depth of two and five certified models produced zero plans
        in five searches. With the model's own targets the same budget reaches
        depth five or six, which is where a click puzzle's answer lives.

        The lattice is not deleted: it is the fallback for a model too vague to
        name any target (and the padding while it names only one or two), because
        a search over nothing at all is worse than a search over a sweep.
        """
        h, w = shape
        acts = [a for a in available if a.upper() in MOVE_ACTIONS]
        if not any(a.upper() == CLICK_ACTION for a in available):
            return acts
        targets: List[Tuple[int, int]] = []
        for rc in collect_coords(state, h, w):
            if rc not in targets:
                targets.append(rc)
        detected = len(targets)
        if detected < CLICK_MIN_TARGETS:
            for r in range(0, h, max(1, CLICK_STRIDE)):
                for c in range(0, w, max(1, CLICK_STRIDE)):
                    if (r, c) not in targets:
                        targets.append((r, c))
        for r, c in targets[:CLICK_MAX_CANDIDATES]:
            acts.append("ACTION6_r%d_c%d" % (r, c))
        return acts

    def run_bfs(self, model: CandidateModel, grid: np.ndarray, available: List[str],
                budget_s: float = BFS_MAX_S, max_nodes: int = BFS_MAX_NODES,
                max_depth: int = BFS_MAX_DEPTH) -> Plan:
        dl = Deadline(budget_s)
        plan = Plan()
        t0 = time.time()
        try:
            root = model.parse(grid)
        except SandboxError as e:
            plan.reason = "parse_state failed on the live grid: %s" % e
            return plan
        try:
            if model.is_goal(root):
                plan.reached, plan.reason = True, "already at goal (is_goal true at depth 0)"
                return plan
        except SandboxError as e:
            plan.reason = "is_goal failed: %s" % e
            return plan

        shape = (int(grid.shape[0]), int(grid.shape[1]))
        start_key = canon(mask_volatile(root))
        frontier = deque([(root, [], [])])
        seen = {start_key}
        nodes = 0
        deepest = 0
        while frontier:
            if dl.expired():
                plan.reason = "BFS hit its %.1fs budget after %d nodes" % (budget_s, nodes)
                break
            if nodes >= max_nodes:
                plan.reason = "BFS hit its %d-node budget" % max_nodes
                break
            state, acts, keys = frontier.popleft()
            if len(acts) >= max_depth:
                continue
            deepest = max(deepest, len(acts))
            for a in self.action_space(model, state, available, shape):
                if dl.expired():
                    break
                nodes += 1
                try:
                    nxt = model.step(state, a)
                except SandboxError:
                    continue          # an action the model refuses to model
                k = canon(mask_volatile(nxt))
                if k in seen:
                    continue
                seen.add(k)
                nacts, nkeys = acts + [a], keys + [k]
                try:
                    hit = model.is_goal(nxt)
                except SandboxError:
                    hit = False
                if hit:
                    plan.reached = True
                    plan.actions = nacts[:PLAN_MAX_LEN]
                    plan.expected = nkeys[:PLAN_MAX_LEN]
                    plan.reason = "goal reached at depth %d" % len(nacts)
                    frontier.clear()
                    break
                if len(nacts) < max_depth:
                    frontier.append((nxt, nacts, nkeys))
        else:
            if not plan.reached:
                plan.exhausted = True
                if not plan.reason:
                    plan.reason = ("the model's reachable state space is EXHAUSTED (%d distinct "
                                   "states) and none of them satisfies is_goal" % len(seen))
        plan.nodes = nodes
        plan.distinct = len(seen)
        plan.depth = deepest
        plan.seconds = time.time() - t0
        self.total_nodes += nodes
        self.total_s += plan.seconds
        if not plan.reached and not plan.reason:
            plan.reason = "no goal state reachable within budget"
        return plan

    def disagreement_action(self, models: List[CandidateModel], grid: np.ndarray,
                            available: List[str]) -> Tuple[Optional[str], str]:
        """[F] Discriminating experiment: when several models all survive the
        backtest, the informative action is the one they predict DIFFERENTLY.
        RHAE squares excess actions, so buy the most refutation per press."""
        if len(models) < 2:
            return None, "only one surviving model; nothing to discriminate"
        shape = (int(grid.shape[0]), int(grid.shape[1]))
        states = []
        for m in models:
            try:
                states.append(m.parse(grid))
            except SandboxError:
                states.append(None)
        pool: List[str] = []
        for m, s in zip(models, states):
            if s is None:
                continue
            for a in self.action_space(m, s, available, shape):
                if a not in pool:
                    pool.append(a)
        best, best_n = None, 1
        for a in pool:
            outs = set()
            for m, s in zip(models, states):
                if s is None:
                    continue
                try:
                    outs.add(canon(mask_volatile(m.step(s, a))))
                except SandboxError:
                    outs.add("error")
            if len(outs) > best_n:
                best, best_n = a, len(outs)
        if best is None:
            return None, "all %d surviving models agree on every action" % len(models)
        return best, "%s splits the %d surviving models into %d predictions" % (
            best, len(models), best_n)


# ============================================================================
# COMPONENT 6: PROMPTS  (this is the perception layer -- there is no other)
# ============================================================================
# Guardrail 1: the observer is a PROMPT. Nothing in this file parses objects,
# finds sprites, or names a background colour. Python only CHECKS what the model
# writes. If perception is wrong, the fix belongs in the text below.

SYSTEM_PROMPT = """You are an expert Python programmer working out how an unfamiliar \
grid video game works. You do not describe the game in prose: you WRITE A PROGRAM that \
explains it, and an automatic checker runs your program against every frame and every \
transition ever observed.

Write the COMPLETE contents of world_model.py inside ONE ```python code block. It must \
define exactly these four top-level functions:

    def parse_state(grid):        # grid is a 2-D numpy int array -> return a dict
    def render(state):            # dict -> 2-D numpy int array, identical to the grid
    def step(state, action):      # dict, action string -> the dict after that action
    def is_goal(state):           # dict -> True exactly when a level was just completed

HOW YOU ARE GRADED (every check runs over the WHOLE recorded history, not the last few
frames, and all four must pass before your model is allowed to choose any action):

CHECK 1 RECONSTRUCTION: render(parse_state(g)) must equal g exactly, cell for cell, for
    every frame. This is how your perception is verified: if your variables can redraw
    the screen, they did not throw information away.
CHECK 0 ABSTRACTION: your state may NOT contain a verbatim copy of the grid, and its
    non-volatile part may not be identical across frames that differ. Storing the pixels
    is not an answer.
CHECK 2 REPLAY: step(parse_state(prev), action) must equal parse_state(next) for every
    recorded transition, comparing all non-volatile keys.
CHECK 3 GOAL: is_goal must be True on the state after every transition that scored a
    point, and False after every transition that did not.

VOLATILE KEYS: a dict key whose name starts with an underscore is VOLATILE. render() must
still draw it (CHECK 1 is exact), but CHECK 2 ignores it. Use `_`-names for things you
cannot predict -- score digits, timers, animation counters, blinking cursors. Do NOT hide
the game in volatile keys: CHECK 0 rejects a state that predicts nothing.

TWO PRESSURES ON YOUR REPRESENTATION:
  PARSIMONY -- the fewest named variables and the shortest code that still reconstructs
    every frame. Name the things the game is about (positions, colours, counts, which
    cells are filled, what is held), not the pixels.
  UTILITY -- a grounding only counts if a mechanism can be written over it. If you cannot
    express step() in terms of your variables, the variables are wrong: change parse_state
    rather than piling special cases into step().

ACTION STRINGS you may receive in step(): "ACTION1".."ACTION5" and "ACTION7" are simple
button presses; a click is "ACTION6_r<row>_c<col>", e.g. "ACTION6_r12_c30". step() must
never raise: for an action you do not model, return the state unchanged.

HARD RULES:
- Output the whole file every time. No diffs, no ellipses, no "unchanged" placeholders.
- `import numpy as np` is allowed and encouraged. Also math, itertools, collections,
  heapq, copy, functools, operator, bisect, re. Nothing else. No file, network or OS
  access, no randomness, no module-level mutable state, no printing.
- Be deterministic and fast: parse_state/render/step are each called thousands of times
  and are killed after a couple of seconds. Do not put a search or a solver inside step();
  a separate planner searches through your step() for you.
- State values must be plain data: ints, floats, bools, strings, tuples, lists, nested
  dicts, or small numpy arrays.
- After the code block you MAY add one ```notes block (at most 40 lines) with your working
  conclusions about the game. It is handed back to you next time you are called, so write
  what you would want to know. If you are asked to name an experiment, put a single line
  `EXPERIMENT: <action string>` in that notes block.
"""


@dataclass
class Context:
    """Everything the prompt is built from. Assembled from the Timeline only."""
    game_id: str
    level: int
    grid: np.ndarray
    available: List[str]
    action_summary: str
    volatile_hint: str
    reward_summary: str
    n_transitions: int
    notes: str = ""
    code: str = ""
    extra: str = ""


def _evidence(ctx: Context) -> str:
    parts = [
        "GAME: %s   LEVEL: %d   recorded transitions: %d" % (
            ctx.game_id, ctx.level, ctx.n_transitions),
        "AVAILABLE ACTIONS this level: %s" % ", ".join(ctx.available),
        "",
        "CURRENT FRAME",
        grid_labeled(ctx.grid),
        "",
        "COMPACT FRAME (one hex digit per cell, same data):",
        grid_block(ctx.grid),
        "",
        "WHAT EACH ACTION HAS DONE SO FAR (raw cell diffs, oldest first):",
        ctx.action_summary,
        "",
        "CELLS THAT CHANGE MOST OFTEN (candidates for `_`-volatile keys; this is a hint, "
        "not a rule -- you decide):",
        ctx.volatile_hint,
        "",
        "WHEN THE SCORE WENT UP: %s" % ctx.reward_summary,
    ]
    if ctx.notes.strip():
        parts += ["", "YOUR NOTES FROM EARLIER:", ctx.notes.strip()[:4000]]
    return "\n".join(parts)


def prompt_author(ctx: Context) -> str:
    return _evidence(ctx) + """

TASK: write world_model.py from scratch. Start by grounding the frame: decide which
variables the screen is made of so that render() can rebuild it exactly, then write the
mechanism for each action from the diffs above, then your best guess at is_goal (nobody
tells you the objective -- infer it from what scored, or if nothing has scored yet, from
what the game looks like it is asking for). It is better to be exactly right about a small
part of the game than approximately right about all of it: CHECK 1 and CHECK 2 are exact.
"""


# The repair FRAMINGS, in escalation order. A run produced sixteen repair calls
# whose replies were near-identical (5233 / 5229 / 5335 chars, same opening
# comment block): asking the same question at the same temperature gets the same
# answer, so sixteen calls were one call billed sixteen times. Each framing asks
# for a DIFFERENT KIND of change, so a refusal to move is at least a refusal to
# move in a named direction.
REPAIR_FRAMINGS = (
    """TASK: fix that specific failure and output the complete corrected file. Do not weaken the
model to dodge the check (do not declare the whole game volatile, do not make is_goal
constant, do not store the raw grid). Every other check must keep passing.""",

    """TASK: your last fix did not work, so stop patching step(). RE-GROUND parse_state instead:
choose DIFFERENT variables for the same screen -- relative coordinates instead of absolute,
one object where you had three, an extra counter you have not represented at all -- and
rewrite the mechanism over those. Output the complete file.""",

    """TASK: the file is too ambitious to be exactly right. SIMPLIFY IT: model LESS of the game,
and model that part exactly. Delete the rules you are unsure of (step() may return the state
unchanged for any action you do not understand yet); keep only what the recorded diffs force.
A small model that passes every check beats a rich one that fails one. Output the complete
file.""",

    """TASK: abandon the current file. It has been refuted several times in a row, which usually
means its whole framing is wrong -- the objects are not the game's objects. Start from the
frame again as if you had never seen your own code: what is on this screen, what does each
action do to it, what would completing this level look like? Output the complete new file.""",
)


def prompt_repair(ctx: Context, report: Report, attempt: int = 0,
                  refuted: Sequence[Tuple[str, str]] = ()) -> str:
    """The counterexample, plus everything already tried and how far to move.

    `refuted` is the (label, bug) of every attempt already dead THIS round. It is
    in the prompt for one reason: a writer that cannot see its own dead ends
    re-emits them, and each re-emission costs a full generation.
    """
    head = _evidence(ctx)
    hint = ""
    if report.suspect_representation:
        hint = ("\nThe checker notes that CHECK 1 passes while CHECK 2 keeps failing. "
                "SUSPECT THE STATE REPRESENTATION, NOT THE RULE: your variables can "
                "redraw the screen but may not be the variables the game updates. "
                "Prefer re-grounding parse_state over adding another branch to step().\n")
    history = ""
    if refuted:
        history = "\nALREADY REFUTED THIS ROUND -- do not send any of these back:\n" + \
            "\n".join("  attempt %d: %s -- %s" % (i + 1, lbl, (bug or "").strip()[:200])
                      for i, (lbl, bug) in enumerate(refuted)) + \
            "\nA file byte-identical to one of those is rejected WITHOUT being run, and you " \
            "will simply be asked again. Change something that matters.\n"
    return head + """

YOUR CURRENT world_model.py:
```python
%s
```

THE CHECKER REFUTED IT:
  %s
%s%s
%s
""" % (ctx.code.strip(), report.summary(), hint, history,
       REPAIR_FRAMINGS[min(attempt, len(REPAIR_FRAMINGS) - 1)])


def prompt_optimism(ctx: Context, plan: Plan) -> str:
    """[E] WorldCoder's optimism: a model that admits no win is EVIDENCE OF ERROR,
    not a dead end. Search failure is a refutation of the model, not of the game.

    WHICH PART of the model it refutes depends on how the search failed, and the
    two cases want opposite revisions:

      EXHAUSTED -- the search ran out of states. That is a PROOF that no state
        your rules can build satisfies is_goal, so is_goal is the prime suspect:
        it is describing something unreachable, or the wrong thing entirely. The
        checks barely constrain it while nothing has scored yet (it only has to be
        False on what has been seen), so it is the cheapest thing to change and the
        most likely to be wrong. Five certified models produced zero plans in five
        searches before this was said out loud.

      BUDGET -- the search was cut off. is_goal is not implicated at all; the win
        is just further away than the search can reach, usually because the state
        carries too many click targets and the branching factor eats the budget.
    """
    if plan.exhausted:
        focus = """
The search EXHAUSTED the state space. That is a PROOF about your file: no state your
step() can build from here satisfies your is_goal. One of the two is wrong, and is_goal is
the first suspect -- while nothing has scored yet the checker only requires it to be False
on the states already seen, so a plausible-looking goal condition can easily describe a
configuration your rules can never produce (a cell colour that never appears, a counter
that nothing increments, an object at a coordinate nothing can reach).

REVISE is_goal FIRST. Ask: what visibly changes when this kind of puzzle is solved, and can
my step() actually produce that change? If is_goal is right, then a rule is missing -- an
action you modelled as a no-op really does something, or a mechanism only fires in a
situation not yet observed."""
    else:
        focus = """
The search ran out of BUDGET rather than out of states, so nothing here refutes is_goal --
the win is simply deeper than the search could reach. Make it shallower: your state should
name FEWER click targets (every coordinate in it is a branch of the search, and a
screen-wide sweep of them buys nothing), and prefer rules that reach the goal in a handful
of actions over rules that need dozens."""
    return _evidence(ctx) + """

YOUR CURRENT world_model.py PASSES EVERY CHECK:
```python
%s
```

But a breadth-first search through your own step() reaches no winning state.
The search says: %s
%s

Your model is therefore incomplete: the game IS winnable, so something your model forbids
is actually possible, or something it ignores actually matters.

TASK: propose the REVISION that makes winning possible while still passing every check
against the recorded history, and output the complete file. You may rewrite is_goal, step,
or parse_state -- whichever the diagnosis above points at. Then, in the notes block, name
the single action whose outcome would confirm or refute your revision, on one line:
    EXPERIMENT: ACTION3
(or e.g. `EXPERIMENT: ACTION6_r20_c41`). One action only -- it will be played for real, and
wasted actions are expensive.
""" % (ctx.code.strip(), plan.reason, focus)


def prompt_levelup(ctx: Context, kept: str) -> str:
    """[H] A level-up is an EDIT, not a rewrite. WorldCoder's transfer story: the
    same program, minimally revised, is why level 2 should cost fewer turns."""
    return _evidence(ctx) + """

YOU JUST COMPLETED A LEVEL. The layout has changed but the rules almost certainly carry
over. Here is the model that solved the previous level -- it is your starting point, not a
draft to be thrown away:
```python
%s
```

TASK: EDIT it minimally for the new level and output the complete file. Change what the
new frame forces you to change (different object positions, an extra object, a new colour,
a bigger board) and keep everything you already established about the mechanism. If a rule
turns out to be level-specific, generalise it rather than replacing it.
""" % (kept.strip(),)


# ============================================================================
# COMPONENT 7: LLM CODER  (in-process; no network, no localhost, no vLLM)
# ============================================================================

def _resolve_model_dir(path: str) -> Optional[str]:
    """Accept either the model directory or a parent that contains it."""
    if not path or not os.path.isdir(path):
        return None
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    try:
        for name in sorted(os.listdir(path)):
            sub = os.path.join(path, name)
            if os.path.isfile(os.path.join(sub, "config.json")):
                return sub
            if os.path.isdir(sub):
                for name2 in sorted(os.listdir(sub)):
                    sub2 = os.path.join(sub, name2)
                    if os.path.isfile(os.path.join(sub2, "config.json")):
                        return sub2
    except OSError:
        return None
    return None


CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
NOTES_FENCE_RE = re.compile(r"```notes\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
EXPERIMENT_RE = re.compile(r"EXPERIMENT:\s*(ACTION\d(?:_r\d+_c\d+)?)", re.IGNORECASE)


def extract_code(text: str) -> Optional[str]:
    """Recover the file from the reply. Deliberately forgiving: valid models used
    to be lost at the READER, not the writer -- an unfenced answer or a stray
    prose line is not a modelling error."""
    if not text:
        return None
    blocks = [b for b in CODE_FENCE_RE.findall(text) if "def parse_state" in b]
    if blocks:
        return blocks[-1].strip("\n")
    blocks = [b for b in CODE_FENCE_RE.findall(text)]
    if blocks:
        best = max(blocks, key=len)
        if "def " in best:
            return best.strip("\n")
    if "def parse_state" in text:
        lines = text.splitlines()
        start = 0
        for i, ln in enumerate(lines):
            if ln.startswith(("import ", "from ", "def ")):
                start = i
                break
        end = len(lines)
        for j in range(len(lines) - 1, start, -1):
            s = lines[j].strip()
            if s and (lines[j][:1] in (" ", "\t") or s.startswith(("def ", "return", "import ", "from "))):
                end = j + 1
                break
        return "\n".join(lines[start:end]).strip("\n")
    return None


def extract_notes(text: str) -> str:
    if not text:
        return ""
    m = NOTES_FENCE_RE.findall(text)
    if m:
        return m[-1].strip()
    tail = CODE_FENCE_RE.sub("", text).strip()
    return tail[-2000:]


def extract_experiment(text: str, available: List[str]) -> Optional[str]:
    """One note line, not a schema. The model's answer is still Python; this is
    the single action it wants played, in the free-form notes it already writes."""
    for m in EXPERIMENT_RE.findall(text or ""):
        a = norm_action(m)
        base = a.split("_")[0]
        if base in [x.upper() for x in available] or base in MOVE_ACTIONS + (CLICK_ACTION,):
            return a
    return None


class LLMCoder:
    """Loads the local Qwen from a Kaggle dataset path and generates code.

    OUR CODE MAKES NO NETWORK CALLS. `local_files_only=True` everywhere; a
    missing model means `available == False` and the agent degrades to probing
    rather than crashing (which is also how the offline self-tests run).
    """

    def __init__(self, model_path: str = LLM_MODEL_PATH) -> None:
        self.model_dir = _resolve_model_dir(model_path)
        self.requested_path = model_path
        self.tok = None
        self.model = None
        self.fla_present: Optional[bool] = None
        self.conv1d_present: Optional[bool] = None
        self.footprint_gb = 0.0
        self.last_out_tokens = 0
        self.load_s = 0.0
        self.calls = 0
        self.failures = 0
        self.gen_s = 0.0
        self.out_tokens = 0
        self.spent_s = 0.0
        self.budget_s = LLM_BUDGET_S
        self.lock = threading.Lock()
        self.load_error = ""
        self.enabled = bool(self.model_dir) and not DISABLE_LLM
        if not self.model_dir:
            self.load_error = "no config.json under %r" % model_path

    # --- availability ---
    @property
    def available(self) -> bool:
        return self.enabled and self.load_error == ""

    def has_budget(self, need_s: float = 0.0) -> bool:
        """Two ceilings, and the session's is the one that cannot be argued with.

        The per-game ledger is the fair share; the session clock is the wall. A
        call that cannot finish before the wall is a call whose cost is paid in
        full and whose answer is never read, so it is refused here rather than
        started and abandoned.
        """
        if not self.available:
            return False
        if (self.spent_s + need_s) >= self.budget_s:
            return False
        left = session_clock().remaining_s()
        return left > 0.0 and need_s <= left

    @property
    def avg_call_s(self) -> float:
        """What the NEXT call is expected to cost, from what calls have cost."""
        return self.gen_s / self.calls if self.calls else 0.0

    @property
    def tokens_per_s(self) -> float:
        return self.out_tokens / self.gen_s if self.gen_s > 0 else 0.0

    # --- loading ---
    def load(self) -> bool:
        if self.model is not None:
            return True
        if not self.enabled:
            return False
        t0 = time.time()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            self.load_error = "transformers unavailable: %s" % e
            return False
        # The fast-path probe: without flash-linear-attention Qwen3.5 runs a torch
        # fallback and a single call costs minutes, which is what makes multi-turn
        # deliberation unaffordable. Report it loudly; do not silently accept it.
        try:
            import fla  # noqa: F401
            self.fla_present = True
        except Exception:
            self.fla_present = False
        try:
            import causal_conv1d  # noqa: F401
            self.conv1d_present = True
        except Exception:
            self.conv1d_present = False
        log("loading LLM from", self.model_dir,
            "| fla", "PRESENT" if self.fla_present else "ABSENT (torch fallback; slow)",
            "| causal_conv1d", "PRESENT" if self.conv1d_present else "ABSENT")
        try:
            self.tok = AutoTokenizer.from_pretrained(
                self.model_dir, local_files_only=True, trust_remote_code=True)
            kw = dict(local_files_only=True, trust_remote_code=True,
                      low_cpu_mem_usage=True)
            if torch is not None:
                kw["device_map"] = {"": 0} if torch.cuda.is_available() else "cpu"
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_dir, dtype=torch.bfloat16, **kw)
                except TypeError:      # older transformers spell it torch_dtype
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_dir, torch_dtype=torch.bfloat16, **kw)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, **kw)
            self.model.eval()
            try:
                self.footprint_gb = self.model.get_memory_footprint() / 1e9
            except Exception:
                self.footprint_gb = 0.0
        except Exception as e:                      # noqa: BLE001
            self.load_error = "%s: %s" % (type(e).__name__, e)
            log("LLM load FAILED:", self.load_error)
            return False
        self.load_s = time.time() - t0
        log("LLM ready in %.1fs, footprint %.1f GB" % (self.load_s, self.footprint_gb))
        return True

    # --- generation ---
    def _encode(self, system: str, user: str):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for kwargs in ({"enable_thinking": False}, {}):
            try:
                enc = self.tok.apply_chat_template(
                    msgs, add_generation_prompt=True, return_tensors="pt",
                    return_dict=True, **kwargs)
                return enc
            except TypeError:
                continue
            except Exception:
                break
        text = system + "\n\n" + user + "\n\n"
        return self.tok(text, return_tensors="pt")

    def _generate_inner(self, system: str, user: str, max_new_tokens: int,
                        max_time: float, temperature: Optional[float] = None) -> str:
        """The model call itself, kept separate from the accounting around it.

        `Kaggle_test.py --fake-llm` REPLACES this attribute to rehearse the
        plumbing without a GPU, so everything that must still run under a stub --
        the budget gate, the lock, code extraction, certification -- lives in
        `generate()` and not in here."""
        enc = self._encode(system, user)
        if torch is not None:
            dev = getattr(self.model, "device", None)
            if dev is not None:
                enc = {k: v.to(dev) for k, v in dict(enc).items()
                       if hasattr(v, "to")}
        n_in = int(dict(enc)["input_ids"].shape[-1])
        temp = LLM_TEMPERATURE if temperature is None else float(temperature)
        gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=temp > 0,
                      temperature=max(1e-5, temp), top_p=0.95,
                      max_time=float(max_time))
        if getattr(self.tok, "pad_token_id", None) is not None:
            gen_kw["pad_token_id"] = self.tok.pad_token_id
        elif getattr(self.tok, "eos_token_id", None) is not None:
            gen_kw["pad_token_id"] = self.tok.eos_token_id
        if torch is not None:
            with torch.no_grad():
                out = self.model.generate(**enc, **gen_kw)
        else:
            out = self.model.generate(**enc, **gen_kw)
        new = out[0][n_in:]
        self.last_in_tokens = n_in
        self.last_out_tokens = int(new.shape[-1])
        return self.tok.decode(new, skip_special_tokens=True)

    def generate(self, system: str, user: str,
                 max_new_tokens: int = LLM_MAX_NEW_TOKENS,
                 max_time: float = LLM_GEN_MAX_TIME_S,
                 temperature: Optional[float] = None) -> Optional[str]:
        if not self.load():
            return None
        # Reserve what this call is about to cost, not zero. Gating on the budget
        # already spent let 22 calls of ~90s each run up 1980s against a declared
        # 1800s ceiling: the last call is always allowed, however long it is.
        if not self.has_budget(self.avg_call_s):
            log("LLM budget exhausted (%.0fs of %.0fs; a call costs ~%.0fs)"
                % (self.spent_s, self.budget_s, self.avg_call_s))
            return None
        with self.lock:
            t0 = time.time()
            self.last_in_tokens = self.last_out_tokens = 0
            try:
                inner = self._generate_inner
                if hasattr(inner, "__self__"):
                    text = inner(system, user, max_new_tokens, max_time, temperature)
                else:
                    # An externally installed stub. Harnesses hand a writer ONE
                    # prompt, so join the two turns rather than silently passing
                    # `user` as `max_new_tokens`.
                    text = inner(system + "\n\n" + user)
                self.out_tokens += self.last_out_tokens
            except Exception as e:                  # noqa: BLE001
                self.failures += 1
                log("generation FAILED: %s: %s" % (type(e).__name__, e))
                self.spent_s += time.time() - t0
                self.gen_s += time.time() - t0
                return None
            dt = time.time() - t0
            self.calls += 1
            self.gen_s += dt
            self.spent_s += dt
            log("LLM call %d: %d in, %d tok out, %.1fs (%.1f tok/s) T=%.2f"
                % (self.calls, self.last_in_tokens, self.last_out_tokens, dt,
                   self.last_out_tokens / dt if dt > 0 else 0.0,
                   LLM_TEMPERATURE if temperature is None else temperature))
            return text

    # `Kaggle_test.py` times the load by wrapping `ensure_loaded`; same call.
    def ensure_loaded(self) -> bool:
        return self.load()

    def new_game(self) -> None:
        """Zero the PER-GAME ledger, keep the loaded weights, and re-derive this
        game's writer allowance from the clock the whole session shares.

        The 53 GB of weights are per process; the budget is per game -- but it is
        the SHARE, not the ceiling. Handing every game LLM_BUDGET_S is how 25 x
        1800s becomes 12.5 h of generation inside a 9 h session, and the games at
        the end of the ordering then never run at all.
        """
        # What a call has actually cost, measured on the PREVIOUS game, before the
        # ledger that measured it is zeroed. On game one there is no measurement,
        # so the generation cap stands in for it.
        one_call = self.avg_call_s if self.calls else LLM_GEN_MAX_TIME_S
        self.calls = self.failures = self.out_tokens = 0
        self.gen_s = self.spent_s = 0.0
        clock = session_clock()
        clock.start_game()
        share = clock.share_s() * LLM_SHARE_OF_GAME
        self.budget_s = max(0.0, min(LLM_BUDGET_S, share))
        # Rounding must not silently leave a game with a writer it can never call:
        # if the session can still afford one generation, this game gets one.
        if self.budget_s < one_call <= clock.remaining_s():
            self.budget_s = one_call
        log("game %d/%d: writer budget %.0fs (share %.0fs of a %.0fs slot, ceiling "
            "%.0fs); session %.2f h left of %.2f h"
            % (clock.games_started, clock.games, self.budget_s, share,
               clock.share_s(), LLM_BUDGET_S, clock.remaining_s() / 3600.0,
               clock.total_s / 3600.0))

    def stats(self) -> Dict[str, Any]:
        return {"calls": self.calls, "failures": self.failures,
                "gen_s": round(self.gen_s, 1), "out_tokens": self.out_tokens,
                "budget_s": round(self.budget_s, 1),
                "budget_used": round(self.spent_s / self.budget_s, 2)
                if self.budget_s > 0 else None,
                "tok_per_s": round(self.tokens_per_s, 2),
                "load_s": round(self.load_s, 1),
                "footprint_gb": round(self.footprint_gb, 1),
                "fla": self.fla_present, "model_dir": self.model_dir,
                "error": self.load_error}


# `LocalLLM` is the name every harness in this repo instruments (Kaggle_test.py's
# Probe resolves the class by name to time the load, count the calls and keep the
# raw answers -- on Kaggle that log is the ONLY place we see what the 27B said).
LocalLLM = LLMCoder

_LLM_SINGLETON: Optional[LLMCoder] = None
_LLM_LOCK = threading.Lock()


def get_local_llm() -> LLMCoder:
    """One writer PER PROCESS, not per game.

    A 27B model takes minutes to load and ~54 GB; constructing an LLMCoder inside
    every MyAgent would pay that for each of the 25 games, and on Kaggle the
    second load would also have to fit beside the first. The per-game *budget*
    still resets -- see `new_game()`."""
    global _LLM_SINGLETON
    with _LLM_LOCK:
        if _LLM_SINGLETON is None:
            _LLM_SINGLETON = LLMCoder()
        return _LLM_SINGLETON


# ============================================================================
# COMPONENT 8: DELIBERATOR  (the inner loop; exactly four tools)
# ============================================================================

def replay_accuracy(model: CandidateModel, timeline: Timeline,
                    level: Optional[int]) -> float:
    """Informational only: how well a model explains transitions it was not
    certified on (i.e. earlier levels). Never a gate."""
    ts = timeline.certify_sample(level, limit=60)
    if not ts:
        return float("nan")
    good = 0
    for t in ts:
        try:
            g0, g1 = timeline.pair(t)
            if canon(mask_volatile(model.step(model.parse(g0), t.action))) == \
               canon(mask_volatile(model.parse(g1))):
                good += 1
        except SandboxError:
            pass
    return good / float(len(ts))


class Deliberator:
    """theorize -> certify -> plan -> commit.

    The four tools below are the entire interface between thinking and the game.
    `commit_actions` is the ONLY way an action reaches the environment -- probes,
    experiments, plans, resets, everything routes through it, which is what makes
    that invariant checkable instead of aspirational.
    """

    def __init__(self, agent: "MyAgent") -> None:
        self.agent = agent
        self.llm = agent.llm
        self.certifier = agent.certifier
        self.planner = agent.planner
        self.sandbox = agent.sandbox
        self.model: Optional[CandidateModel] = None          # the certified model
        self.survivors: List[CandidateModel] = []            # for [F]
        self.code = ""
        self.notes = ""
        self.rounds = 0
        self.turns = 0
        self.n_plans = 0
        self.n_optimism = 0
        self.n_discrim = 0
        self.n_probes = 0
        self.n_repeat_replies = 0    # replies byte-identical to one already refuted
        self.last_report: Optional[Report] = None
        self.last_replan: Optional[Tuple[str, str]] = None   # (model digest, frame)
        self.TOOLS: Dict[str, Callable[..., Any]] = {
            "write_code": self.write_code,
            "run_backtest": self.run_backtest,
            "run_bfs": self.run_bfs,
            "commit_actions": self.commit_actions,
        }
        # Guardrail 3, mechanically: four tools, forever.
        assert len(self.TOOLS) == 4, "the toolset is fixed at four"

    # ---------------- TOOL 1: write_code ----------------
    def write_code(self, prompt: str,
                   temperature: Optional[float] = None) -> Tuple[Optional[str], str, str]:
        """The LLM rewrites world_model.py (and its notes). Returns (code, notes, raw)."""
        self.turns += 1
        raw = self.llm.generate(SYSTEM_PROMPT, prompt, temperature=temperature) or ""
        code = extract_code(raw)
        notes = extract_notes(raw)
        if code:
            self.agent.persist(code, notes)
        return code, notes, raw

    # ---------------- TOOL 2: run_backtest ----------------
    def run_backtest(self, code: str, notes: str = "") -> Tuple[Optional[CandidateModel], Report]:
        """Compile and run the three checks. The Report carries a pointed bug."""
        try:
            model = CandidateModel(code, self.sandbox, notes=notes)
        except SandboxError as e:
            # SandboxTimeout is a SandboxError, and both now carry their own
            # label, so one branch is enough and neither can lose its diagnosis.
            rep = Report(ok=False, check=getattr(e, "label", "REJECTED"), bug=str(e))
            self.certifier.note_rejection(rep)
            log("backtest:", rep.summary())
            return None, rep
        rep = self.certifier.certify(model, level=self.agent.level)
        self.last_report = rep
        log("backtest:", rep.summary())
        return (model if rep.ok else None), rep

    # ---------------- TOOL 3: run_bfs ----------------
    def run_bfs(self, model: CandidateModel) -> Plan:
        plan = self.planner.run_bfs(model, self.agent.grid, self.agent.available)
        self.n_plans += 1
        log("bfs:", plan.summary())
        return plan

    # ---------------- TOOL 4: commit_actions ----------------
    def commit_actions(self, actions: List[str], expected: Optional[List[str]] = None,
                       why: str = "") -> List[str]:
        """The only channel from thinking to the game."""
        return self.agent.enqueue(actions, expected, why)

    # ---------------- the inner loop ----------------
    def deliberate(self) -> List[str]:
        ag = self.agent
        dl = Deadline(DELIBERATE_MAX_S)
        self.rounds += 1

        # No writer available (offline, missing weights, spent budget). If a model
        # was already certified, keep EXPLOITING it -- re-certify against the newer
        # evidence and plan again; only fall back to probing with nothing to plan
        # through. This is not a planner, and it never authors code.
        #
        # The (code, frame) guard is what makes this affordable. Certification is
        # up to CERT_MAX_S and BFS another BFS_MAX_S; the writer's budget is spent
        # long before the action budget is, so without the guard the last ~1450
        # actions of a game would each pay 30s to re-derive an answer that cannot
        # have changed -- BFS is deterministic, so the same model searched from the
        # same frame returns the same plan.
        if not self.llm.has_budget():
            stamp = (self.model.digest if self.model else "", canon(ag.grid))
            if self.model is not None and self.code and stamp != self.last_replan:
                self.last_replan = stamp
                model, _rep = self.run_backtest(self.code, self.notes)
                if model is not None:
                    self.model = model
                    plan = self.run_bfs(model)
                    if plan:
                        return self.commit_actions(plan.actions, plan.expected,
                                                   "bfs plan (no writer available)")
            return self.probe("no LLM available (%s)"
                              % (self.llm.load_error or "budget spent"), n=PROBE_BATCH)

        ctx = ag.context()
        if ag.pending_levelup and self.code:
            prompt = prompt_levelup(ctx, self.code)     # [H] edit, never rewrite
            ag.pending_levelup = False
        elif self.model is not None and self.code:
            ctx.code = self.code
            prompt = prompt_repair(ctx, self.last_report or Report(
                ok=False, check="STALE", bug="the model no longer covers the newest "
                "transitions; extend it to explain them too"))
        elif self.code:
            ctx.code = self.code
            prompt = prompt_repair(ctx, self.last_report or Report(
                ok=False, check="UNKNOWN", bug="previous file did not certify"))
        else:
            prompt = prompt_author(ctx)

        # The repair loop ESCALATES rather than repeating itself. Sixteen repair
        # calls once produced near-identical files, which is one call billed
        # sixteen times: same prompt, same temperature, same answer. Each failed
        # attempt therefore raises the sampling temperature, moves to a framing
        # that asks for a different KIND of change, and is listed back to the
        # writer as already dead. A byte-identical reply is not even backtested.
        certified: Optional[CandidateModel] = None
        refuted: List[Tuple[str, str]] = []
        digests: set = set()
        attempt = 0
        for turn in range(DELIBERATE_MAX_TURNS):
            if dl.expired() or not self.llm.has_budget():
                break
            temp = min(REPAIR_TEMP_MAX, LLM_TEMPERATURE + REPAIR_TEMP_STEP * attempt)
            code, notes, _raw = self.write_code(prompt, temperature=temp)
            if not code:
                log("turn %d: no usable code in the reply" % (turn + 1))
                attempt += 1
                continue
            if canon(code) in digests:
                self.n_repeat_replies += 1
                log("turn %d: byte-identical to an attempt already refuted; "
                    "escalating instead of backtesting it again" % (turn + 1))
                attempt += 1
                ctx.code = code
                prompt = prompt_repair(ctx, self.last_report or Report(
                    ok=False, check="REPEAT",
                    bug="you returned a file already refuted this round"),
                    attempt=attempt, refuted=refuted)
                continue
            digests.add(canon(code))
            self.code, self.notes = code, notes or self.notes
            model, rep = self.run_backtest(code, notes)
            if model is not None:
                certified = model
                self.model = model
                if all(m.digest != model.digest for m in self.survivors):
                    self.survivors.append(model)
                del self.survivors[:-4]
                break
            refuted.append((rep.check or "REJECTED", rep.bug))
            attempt += 1
            ctx.code = code
            prompt = prompt_repair(ctx, rep, attempt=attempt, refuted=refuted[-4:])

        if certified is None:
            # Nothing survived. Prefer an experiment that splits the models that
            # DID survive earlier ([F]); otherwise probe.
            return self.discriminate_or_probe("no model certified this round")

        # A certified model: plan inside it for free.
        plan = self.run_bfs(certified)
        if plan:
            xacc = replay_accuracy(certified, ag.timeline, None)
            log("plan committed (cross-level replay %.2f)" % xacc)
            return self.commit_actions(plan.actions, plan.expected, "bfs plan")

        # [E] OPTIMISM TRIGGER: no reachable goal means the MODEL is wrong.
        if not NO_OPTIMISM and self.llm.has_budget() and not dl.expired():
            self.n_optimism += 1
            ctx.code = self.code
            code, notes, raw = self.write_code(prompt_optimism(ctx, plan))
            if code:
                model2, rep2 = self.run_backtest(code, notes)
                if model2 is not None:
                    self.code, self.notes = code, notes or self.notes
                    self.model = model2
                    if all(m.digest != model2.digest for m in self.survivors):
                        self.survivors.append(model2)
                    del self.survivors[:-4]
                    plan2 = self.run_bfs(model2)
                    if plan2:
                        return self.commit_actions(plan2.actions, plan2.expected,
                                                   "optimistic revision")
                    exp = extract_experiment(raw, ag.available)
                    if exp:
                        return self.commit_actions([exp], None, "optimism experiment")
                else:
                    exp = extract_experiment(raw, ag.available)
                    if exp:
                        return self.commit_actions([exp], None,
                                                   "experiment from an uncertified revision")

        return self.discriminate_or_probe("certified model admits no winning path")

    # ---------------- experiment selection ----------------
    def discriminate_or_probe(self, why: str) -> List[str]:
        if not NO_DISCRIM and len(self.survivors) >= 2:
            act, reason = self.planner.disagreement_action(
                self.survivors, self.agent.grid, self.agent.available)
            if act:
                self.n_discrim += 1
                log("discriminating experiment:", reason)
                return self.commit_actions([act], None, "discriminate: " + reason)
        return self.probe(why, n=PROBE_BATCH)

    def candidates(self) -> List[str]:
        """Every action that could be played right now, one entry per click cell.

        Generic: the bases the engine offers, plus -- for a click game -- the
        coordinates the certified model itself names, then a stride lattice. No
        game id, no hand-written button finder.
        """
        ag = self.agent
        pool = list(ag.available) or ["ACTION1"]
        order = {a: i for i, a in enumerate(pool)}   # not pool.index: list.sort()
        # Count by BASE action: every click is a different string, so keying the
        # tally on the full name would leave ACTION6 permanently "untried" and
        # replay cell (0,0) forever.
        pool.sort(key=lambda a: (ag.base_count(a), order[a]))
        out: List[str] = []
        for base in pool:
            if base.upper() != CLICK_ACTION:
                out.append(base)
                continue
            h, w = ag.grid.shape
            cells: List[Tuple[int, int]] = []
            if self.model is not None:
                try:
                    cells = collect_coords(self.model.parse(ag.grid), h, w)
                except SandboxError:
                    cells = []
            step = max(1, CLICK_STRIDE)
            cells += [(r, c) for r in range(0, h, step) for c in range(0, w, step)]
            seen: set = set()
            for r, c in cells:
                if (r, c) in seen:
                    continue
                seen.add((r, c))
                out.append("ACTION6_r%d_c%d" % (r, c))
        return out

    def informative_actions(self, n: int) -> List[str]:
        """The n most novel actions available from THIS frame, best first.

        "Novel" is the Timeline's no-op memory: an action never tried here beats
        one tried and known to work, which beats one tried and known to do
        nothing. An empty list means acting from here cannot teach us anything --
        which the caller must handle by thinking, not by pressing a button.
        """
        ag = self.agent
        cands = self.candidates()
        # Novelty first, then the agent's own tally, then the order candidates()
        # produced. The tally is what keeps a click sweep MOVING before any
        # transition has been recorded -- without it the most novel cell is the
        # same cell every time and the sweep replays (0,0) forever.
        pos = {a: i for i, a in enumerate(cands)}
        ranked = sorted(cands, key=lambda a: (ag.timeline.novelty(ag.grid, a),
                                              ag.action_counts.get(a, 0), pos[a]))
        picks: List[str] = []
        for a in ranked:
            if a in picks:
                continue
            if ag.timeline.inert(ag.grid, a):
                break          # ranked, so everything from here on is inert too
            picks.append(a)
            if len(picks) >= max(1, n):
                break
        return picks

    def probe(self, why: str, n: int = 1) -> List[str]:
        """The information-gathering floor: buy the transitions we do not have.

        It is NOT a planner and must never grow into one -- the coverage sweeps
        and movement trial-and-error it replaces are exactly what RHAE punishes.
        What it now refuses to do is spend a scored action on an outcome already
        recorded from this very frame; when every action's local answer is known
        the fallback still plays one (the engine demands an action) but it is
        counted as IDLE, so "the agent is fidgeting" is a number in the report
        instead of 1446 anonymous no-ops.
        """
        ag = self.agent
        self.n_probes += 1
        picks = self.informative_actions(max(1, n))
        if picks:
            return self.commit_actions(picks, None, "probe (%s)" % why)
        cands = self.candidates() or ["ACTION1"]
        ag.idle_actions += 1
        return self.commit_actions([cands[0]], None,
                                   "idle: every action's outcome here is already "
                                   "recorded (%s)" % why)

    def stats(self) -> Dict[str, Any]:
        return {"rounds": self.rounds, "llm_turns": self.turns, "plans": self.n_plans,
                "optimism": self.n_optimism, "discriminate": self.n_discrim,
                # Generations that came back byte-identical to a file already
                # refuted this round: paid for, and worth nothing.
                "repeat_replies": self.n_repeat_replies,
                "probes": self.n_probes, "certified": self.certifier.n_certified,
                "rejected": self.certifier.n_rejected,
                # WHY they were rejected. A run that reports 54 rejections and one
                # bucket has measured nothing; this is the difference between
                # "fix the writer's output format" and "fix the mechanism".
                "reject_kinds": dict(sorted(self.certifier.reject_kinds.items(),
                                            key=lambda kv: -kv[1])),
                "reject_examples": dict(self.certifier.reject_examples),
                "survivors": len(self.survivors),
                "has_model": self.model is not None}


# ============================================================================
# COMPONENT 9: AGENT  (outer loop: observe -> deliberate -> execute -> record)
# ============================================================================

_ROUTE_TAGS = (
    ("bfs plan", "plan"),
    ("optimistic revision", "plan_optimism"),
    ("optimism experiment", "experiment_optimism"),
    ("experiment from", "experiment_optimism"),
    ("discriminate", "experiment_discrim"),
    # `idle` outranks the probe reasons on purpose: the idle fallback quotes the
    # reason it was asked for, and "we were fidgeting" is the fact worth counting.
    ("idle", "probe_idle"),
    ("bootstrap", "probe_bootstrap"),
    ("no new evidence", "probe_evidence"),
    ("engine state", "reset"),
    ("probe", "probe"),
)


def route_tag(why: str) -> str:
    """Which branch spent this action. The harness ledger reads `_route`, so
    every action is attributable: a score with no attribution cannot steer
    anything, and neither can a cost."""
    w = (why or "").lower()
    for needle, tag in _ROUTE_TAGS:
        if needle in w:
            return tag
    return "other"


def read_score(latest_frame: Any) -> int:
    for attr in ("levels_completed", "score"):
        v = getattr(latest_frame, attr, None)
        if isinstance(v, (int, float, np.integer, np.floating)):
            return int(v)
    return 0


class MyAgent(Agent):
    """The class the harness imports. `Kaggle_test.py` and the notebook both do
    `from my_agent import MyAgent`, drive `choose_action`/`is_done`, increment
    `action_counter` themselves and read `last_action_str` for attribution."""

    MAX_ACTIONS = MAX_ACTIONS_DEFAULT

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        gid = kwargs.get("game_id") or getattr(self, "game_id", None) or "local"
        self.game_id = gid
        self.rng = random.Random(AGENT_SEED + (hash(str(gid)) & 0xFFFF))
        if not hasattr(self, "action_counter"):
            self.action_counter = 0
        self.last_action_str = ""
        self._route = "init"           # read by the harness ledger for attribution
        self.t0 = time.time()
        # Print the budget the run believes it has, once, before the first game
        # spends any of it. `llm.new_game()` below is what advances the counter.
        if session_clock().games_started == 0:
            config_report()

        # persistent state (the agent's "weights" for THIS game)
        self.timeline = Timeline()
        self.sandbox = Sandbox()
        self.certifier = Certifier(self.timeline)
        self.planner = Planner(self.sandbox)
        self.llm = get_local_llm()   # process singleton: weights loaded ONCE
        self.llm.new_game()          # ...but the per-game LLM budget starts fresh
        self.delib = Deliberator(self)

        self.queue: deque = deque()          # (action, expected_masked_canon, why)
        self.grid = np.zeros((1, 1), dtype=np.int64)
        self.available: List[str] = []
        self.level = 0                       # banked levels; level under play is +1
        self.last_score = 0
        self.action_counts: Dict[str, int] = {}
        self.pending: Optional[Tuple[np.ndarray, str, int, Optional[str]]] = None
        self.pending_levelup = False
        self.mispredictions = 0
        self.plan_aborts = 0
        self.resets = 0
        self.reported = False
        self.actions_at_level_start = 0
        self.turns_at_level_start = 0
        self.deliberated_at: Optional[Tuple[int, int, int, int]] = None
        self.idle_actions = 0        # played only because the engine wants one
        self.crashes = 0             # exceptions the crash wall absorbed
        self.level_costs: List[int] = []
        self.level_turns: List[int] = []

        # [H] state is per GAME: a fresh game starts from no model and no notes.
        self.work_dir = ensure_dir(os.path.join(
            WORK_DIR, re.sub(r"[^A-Za-z0-9_.-]", "_", str(gid))))
        for name in ("world_model.py", "notes.md"):
            p = os.path.join(self.work_dir, name)
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        log("agent ready for %s (work dir %s, LLM %s)"
            % (gid, self.work_dir, "on" if self.llm.available else
               "OFF: " + (self.llm.load_error or "disabled")))

    # ---------------- persistence ----------------
    def persist(self, code: str, notes: str) -> None:
        """world_model.py and notes.md on disk: inspectable after the run, and the
        only two artefacts that carry a game's learning. The MODEL never touches
        the filesystem -- we write on its behalf."""
        try:
            with open(os.path.join(self.work_dir, "world_model.py"), "w",
                      encoding="utf-8") as f:
                f.write(code)
            if notes:
                with open(os.path.join(self.work_dir, "notes.md"), "w",
                          encoding="utf-8") as f:
                    f.write(notes)
        except OSError as e:
            log("persist failed (harmless):", e)

    # ---------------- the commit channel ----------------
    def enqueue(self, actions: List[str], expected: Optional[List[str]] = None,
                why: str = "") -> List[str]:
        acts = [norm_action(a) for a in (actions or []) if str(a).strip()]
        acts = acts[:PLAN_MAX_LEN]
        exp = list(expected or [])
        for i, a in enumerate(acts):
            self.queue.append((a, exp[i] if i < len(exp) else None, why))
        if acts:
            log("commit %d action(s) [%s]%s | %s"
                % (len(acts), ",".join(acts[:8]), "..." if len(acts) > 8 else "", why))
        return acts

    # ---------------- context for the prompts ----------------
    def context(self) -> Context:
        lvl = self.level
        return Context(
            game_id=str(self.game_id),
            level=lvl + 1,
            grid=self.grid,
            available=list(self.available),
            action_summary=self.timeline.action_effect_summary(lvl),
            volatile_hint=self.timeline.volatile_cell_hint(lvl),
            reward_summary=self.timeline.reward_summary(),
            n_transitions=len(self.timeline.transitions_for_check(lvl)),
            notes=self.delib.notes,
            code=self.delib.code,
        )

    # ---------------- engine adapters ----------------
    def valid_actions(self, latest_frame: Any) -> List[str]:
        out: List[str] = []
        for a in (getattr(latest_frame, "available_actions", None) or []):
            name = getattr(a, "name", None)
            if not name:
                try:
                    name = "ACTION%d" % int(a)
                except (TypeError, ValueError):
                    name = str(a)
            name = str(name).upper()
            if name.startswith("RESET"):
                continue           # RESET is issued by us, never planned through
            if name not in out:
                out.append(name)
        return out

    def to_game_action(self, action_str: str) -> Any:
        s = str(action_str).upper()
        m = CLICK_RE.match(s)
        if m:
            r, c = int(m.group(1)), int(m.group(2))
            act = getattr(GameAction, "ACTION6", "ACTION6")
            try:
                act.set_data({"x": c, "y": r})      # engine wants x=col, y=row
            except AttributeError:
                pass
            return act
        return getattr(GameAction, s, getattr(GameAction, "ACTION1", s))

    # ---------------- the per-step self-check ----------------
    def self_check(self, prev_grid: np.ndarray, action: str, new_grid: np.ndarray,
                   expected: str) -> None:
        """[G] Reality outranks the model. ONE mismatch voids the whole remaining
        plan and hands the offending transition back as the counterexample."""
        model = self.delib.model
        if model is None:
            return
        try:
            actual_state = model.parse(new_grid)
            actual = canon(mask_volatile(actual_state))
        except SandboxError:
            actual, actual_state = None, None
        if actual == expected:
            return
        self.mispredictions += 1
        dropped = len(self.queue)
        if dropped:
            self.plan_aborts += 1
            self.queue.clear()
        diff = ""
        try:
            pred = model.step(model.parse(prev_grid), action)
            if actual_state is not None:
                diff = first_difference(mask_volatile(pred), mask_volatile(actual_state))
        except SandboxError as e:
            diff = "and the model then failed on the live frame: %s" % e
        self.delib.last_report = Report(
            ok=False, check="CHECK 2 REPLAY (live)",
            bug=("the plan was voided: your model predicted the wrong outcome for %s just "
                 "now, on the real game. %s. This transition is #%d in the history and "
                 "every future backtest includes it."
                 % (action, diff or "the states differ", max(0, len(self.timeline) - 1))),
            suspect_representation=self.certifier.check2_fail_streak >= 2)
        # A refutation from reality is the most valuable evidence there is: think
        # again on the very next action.
        self.deliberated_at = None
        log("MISPREDICTION on %s -> voided %d queued action(s)" % (action, dropped))

    # ---------------- bootstrap ----------------
    def base_count(self, action: str) -> int:
        """How often the underlying button has been pressed, clicks aggregated."""
        base = str(action).split("_")[0].upper()
        return sum(v for k, v in self.action_counts.items()
                   if k.split("_")[0].upper() == base)

    # ---------------- when to think ----------------
    def evidence_stamp(self) -> Tuple[int, int, int, int]:
        """What the agent knows, as four counters.

        Deliberation is gated on this CHANGING rather than on a clock. Anything
        that could make a new round produce a different answer moves it: another
        transition, a frame never seen before, a banked level, a refutation. If
        none of them moved, the writer would be re-reading the same evidence, and
        the old build's answer to that was to burn a real action waiting.
        """
        return (len(self.timeline),
                len(self.timeline.frames()),
                self.level,
                self.mispredictions)

    def think(self) -> List[str]:
        """One deliberation, inline and blocking. Records the evidence it read."""
        self.deliberated_at = self.evidence_stamp()
        return self.delib.deliberate()

    def bootstrap_target(self) -> int:
        """Author the first model only after each button has been pressed once.

        A model authored with zero transitions certifies vacuously (CHECK 2 has
        nothing to refute it with), so BFS would plan through a fantasy. This is
        the smallest fixed evidence purchase that prevents that.
        """
        n = len([a for a in self.available if a.upper() in MOVE_ACTIONS])
        if any(a.upper() == CLICK_ACTION for a in self.available):
            n += 3
        return min(8, max(2, n))

    # ---------------- OUTER LOOP ----------------
    def choose_action(self, frames: Any, latest_frame: Any) -> Any:
        """The crash wall, and the only reason it is a separate method.

        Everything below is ordinary code that can contain a bug. This line is the
        promise that such a bug costs one action instead of the rest of the
        session: the harness calls `choose_action` in a bare loop with no
        try/except of its own, so an exception raised here ends the game -- and on
        Kaggle, plausibly the run, taking every game after this one to zero along
        with it. A wrong action is recoverable evidence. A traceback is not.
        """
        try:
            return self._choose_action_inner(frames, latest_frame)
        except Exception as e:                              # noqa: BLE001
            self.crashes += 1
            log("choose_action RAISED (%d this game): %s: %s"
                % (self.crashes, type(e).__name__, e))
            if self.crashes <= CRASH_TRACEBACKS:
                log(traceback.format_exc(limit=8))
            try:
                return self.to_game_action(self.safe_action())
            except Exception:                               # noqa: BLE001
                # Even the fallback failed. Return the one action every game has.
                self.last_action_str, self._route = "ACTION1", "crash-fallback"
                return getattr(GameAction, "ACTION1", "ACTION1")

    def safe_action(self) -> str:
        """A legal action to play when deliberation has just crashed.

        Deliberately ignorant: it consults no model, no queue and no timeline,
        because each of those is a candidate for being the reason we are here. It
        still registers the action as pending, so the step is recorded and the next
        deliberation learns from it rather than losing the transition.
        """
        self.queue.clear()
        self.pending = None
        moves = [a.upper() for a in (self.available or list(MOVE_ACTIONS))
                 if a.upper() != CLICK_ACTION]
        if moves:
            act = moves[self.rng.randrange(len(moves))]
        else:
            h, w = (self.grid.shape[0], self.grid.shape[1]) \
                if getattr(self.grid, "ndim", 0) >= 2 else (64, 64)
            act = "ACTION6_r%d_c%d" % (self.rng.randrange(max(1, h)),
                                       self.rng.randrange(max(1, w)))
        self.last_action_str = act
        self._route = "crash-fallback"
        try:
            self.pending = (np.array(self.grid, dtype=np.int64, copy=True),
                            act, self.level, None)
        except Exception:                                   # noqa: BLE001
            self.pending = None
        return act

    def _choose_action_inner(self, frames: Any, latest_frame: Any) -> Any:
        g = to_grid(latest_frame)
        if g is None:
            g = self.grid
        score = read_score(latest_frame)
        gstate = getattr(latest_frame, "state", None)
        av = self.valid_actions(latest_frame)
        if av:
            self.available = av
        elif not self.available:
            self.available = list(MOVE_ACTIONS)

        # --- RECORD (learn from the previous step, now that we can see its result)
        if self.pending is not None:
            prev_grid, prev_act, prev_level, expected = self.pending
            reward = 1.0 if score > self.last_score else 0.0
            terminal = (gstate == GameState.GAME_OVER)
            self.timeline.record(prev_grid, prev_act, g, reward, terminal, prev_level)
            self.action_counts[prev_act] = self.action_counts.get(prev_act, 0) + 1
            self.pending = None
            if reward > 0:
                # LEVEL UP: keep the model, hand it back for editing.
                self.pending_levelup = True
                self.queue.clear()
                self.certifier.check2_fail_streak = 0
                self.deliberated_at = None         # a new layout earns a fresh round
                self.level_costs.append(self.action_counter - self.actions_at_level_start)
                self.level_turns.append(self.delib.turns - self.turns_at_level_start)
                self.actions_at_level_start = self.action_counter
                self.turns_at_level_start = self.delib.turns
                log("LEVEL UP -> %d banked (%d actions, %d LLM turns)"
                    % (int(score), self.level_costs[-1], self.level_turns[-1]))
            elif expected is not None:
                self.self_check(prev_grid, prev_act, g, expected)
        elif g is not None:
            self.timeline.note_frame(g, int(score))

        self.grid = g
        self.last_score = int(score)
        self.level = int(score)

        # --- RESET only when the engine has stranded us, and never twice running:
        # two RESETs with no action between them is a full_reset, which wipes
        # every level already banked.
        if gstate in (GameState.GAME_OVER, GameState.NOT_PLAYED):
            if not self.last_action_str.upper().startswith("RESET"):
                self.queue.clear()
                self.resets += 1
                self.delib.commit_actions(["RESET"], None, "engine state=%s" % gstate)

        # --- DELIBERATE (only when nothing is committed)
        #
        # Deliberation happens HERE, inline and blocking. Under RHAE thinking is
        # free and button presses are squared, so the agent may take as long as it
        # likes inside this call and must never play an action to pass the time.
        #
        # The gate is the EVIDENCE, not a clock. A deliberation that found nothing
        # will find nothing again against an identical record, so a round is paid
        # for only when something has actually been learned since the last one --
        # a new transition, a new frame, a level-up, a refutation. What used to sit
        # in the `else` below was a filler probe on a doubling timer; in the measured
        # run it spent 1446 of 1500 actions, 98% of them world-no-ops, for zero levels.
        # It is deleted, not tuned:
        #
        #     else:
        #         self.delib.probe("buying evidence for the next deliberation")
        #
        if not self.queue:
            if len(self.timeline.transitions_for_check(self.level)) < self.bootstrap_target():
                self.delib.probe("bootstrap: one press per button before theorising",
                                 n=PROBE_BATCH)
            elif self.evidence_stamp() != self.deliberated_at:
                self.think()
            elif self.delib.informative_actions(1):
                # No new evidence since the last round, but something here is
                # still worth pressing: buy a batch of it, so one expensive round
                # of generation is amortised over several real transitions.
                self.delib.probe("no new evidence since the last round", n=PROBE_BATCH)
            else:
                # Every action's outcome from this exact frame is already
                # recorded, so acting cannot teach us anything. Think again
                # instead -- free under RHAE -- and let the writer's temperature
                # escalation make the repeat non-degenerate.
                self.think()
        if not self.queue:
            self.delib.probe("nothing committed")

        # --- EXECUTE
        act, expected, why = self.queue.popleft()
        self.pending = (np.array(g, dtype=np.int64, copy=True), act, self.level, expected)
        self.last_action_str = act
        self._route = route_tag(why)
        return self.to_game_action(act)

    # backwards-compatible name used by older drivers
    def parse_action(self, action_str: str) -> Any:
        return self.to_game_action(action_str)

    def act(self, frame_data: Any) -> Any:
        return self.choose_action([frame_data], frame_data)

    def is_done(self, frames: Any, latest_frame: Any) -> bool:
        """Same wall as choose_action: the action cap is the answer of last resort,
        because a raise here also ends the run."""
        try:
            return self._is_done_inner(frames, latest_frame)
        except Exception as e:                              # noqa: BLE001
            log("is_done RAISED: %s: %s" % (type(e).__name__, e))
            return int(getattr(self, "action_counter", 0)) >= self.MAX_ACTIONS

    def _is_done_inner(self, frames: Any, latest_frame: Any) -> bool:
        if getattr(latest_frame, "state", None) == GameState.WIN:
            self.report("WIN")
            return True
        if getattr(self, "action_counter", 0) >= self.MAX_ACTIONS:
            self.report("out of actions")
            return True
        # The session wall. This is not giving up: the clock is spent either way,
        # and returning here lets the harness write its results out in the reserve
        # instead of being killed mid-generation with nothing on disk.
        if session_clock().remaining_s() <= 0.0:
            self.report("session wall clock spent (%.2f h)"
                        % (session_clock().elapsed_s() / 3600.0))
            return True
        # OPT-IN (ARC_STOP_WHEN_IDLE=1), and off by default because giving up
        # guarantees a zero for this game. The case for it is arithmetic: RHAE
        # scores a completed level on (human actions / our actions) SQUARED, so
        # actions spent fidgeting before a win do not merely waste time, they
        # divide the eventual score. An agent that stops after 60 actions and one
        # that stops after 1500 both score 0 today -- but only the first can still
        # score well if the writer later finds the answer. Set the flag to measure
        # which is true here instead of arguing about it.
        if STOP_WHEN_IDLE and self.idle_actions >= IDLE_STOP_AFTER:
            self.report("idle: %d actions with nothing left to learn" % self.idle_actions)
            return True
        return False

    # ---------------- reporting ----------------
    def stats(self) -> Dict[str, Any]:
        elapsed = time.time() - self.t0
        banked = len(self.level_costs)
        return {
            "game": str(self.game_id),
            "levels_banked": banked,
            "actions": int(getattr(self, "action_counter", 0)),
            "actions_per_level": [int(x) for x in self.level_costs],
            "llm_turns_per_level": [int(x) for x in self.level_turns],
            "transitions": len(self.timeline),
            "mispredictions": self.mispredictions,
            "plan_aborts": self.plan_aborts,
            "resets": self.resets,
            # Actions played only because the engine demands one. This is the
            # number the filler probe used to hide: it should be a handful, and if
            # it is hundreds the agent is fidgeting and the run is telling you so.
            "idle_actions": self.idle_actions,
            # Exceptions the crash wall absorbed. Must be 0. Anything else is a
            # bug that a fallback action papered over, and the traceback for the
            # first few is above in the log.
            "crashes": self.crashes,
            "wall_s": round(elapsed, 1),
            "proj_25_games_h": round(elapsed * SESSION_GAMES / 3600.0, 2),
            "session": session_clock().stats(),
            "sandbox": {"calls": self.sandbox.calls,
                        "time_s": round(self.sandbox.time_s, 1),
                        "leaked": self.sandbox.leaked},
            "bfs": {"nodes": self.planner.total_nodes,
                    "time_s": round(self.planner.total_s, 1)},
            "deliberator": self.delib.stats(),
            "llm": self.llm.stats(),
        }

    def report(self, why: str = "") -> None:
        if self.reported:
            return
        self.reported = True
        s = self.stats()
        log("=== %s finished (%s) ===" % (self.game_id, why))
        log(json.dumps(s, indent=2, default=str))
        try:
            with open(os.path.join(self.work_dir, "stats.json"), "w", encoding="utf-8") as f:
                json.dump(s, f, indent=2, default=str)
        except OSError:
            pass


# The notebook ships this file as /kaggle/working/my_agent.py, so the class name
# is what the harness imports. The old scaffold's name stays as an alias.
ARCAgent = MyAgent


# ============================================================================
# COMPONENT 10: SELF-TESTS  (offline, no LLM, no engine -- GATE 1)
# ============================================================================
# These check the CHECKER, which is the only part of this agent that must be
# right for the rest to mean anything: a certifier that accepts a wrong model is
# worse than no certifier. Everything here runs on hand-built frames.

TOY_GOOD = '''
import numpy as np

def parse_state(grid):
    g = np.asarray(grid)
    ys, xs = np.where(g == 3)
    ay, ax = int(ys[0]), int(xs[0])
    gy, gx = np.where(g == 4)
    if len(gy):
        goal = (int(gy[0]), int(gx[0]))
    else:
        goal = (ay, ax)              # the avatar is standing on the goal
    return {"ay": ay, "ax": ax, "goal": goal, "_hud": int(g[0, 0])}

def render(state):
    g = np.zeros((8, 8), dtype=int)
    g[0, 0] = state["_hud"]
    gy, gx = state["goal"]
    g[gy, gx] = 4
    g[state["ay"], state["ax"]] = 3
    return g

def step(state, action):
    s = dict(state)
    ay, ax = s["ay"], s["ax"]
    if action == "ACTION1":
        ay -= 1
    elif action == "ACTION2":
        ay += 1
    elif action == "ACTION3":
        ax -= 1
    elif action == "ACTION4":
        ax += 1
    s["ay"] = min(7, max(1, ay))
    s["ax"] = min(7, max(0, ax))
    return s

def is_goal(state):
    return (state["ay"], state["ax"]) == tuple(state["goal"])
'''

TOY_DROPS_GOAL = TOY_GOOD.replace(
    '    g[gy, gx] = 4\n', '')                      # forgets to draw the goal marker

TOY_BAD_STEP = TOY_GOOD.replace(
    '        ax += 1', '        ax += 2')           # right moves two cells

TOY_GOAL_NEVER = TOY_GOOD.replace(
    '    return (state["ay"], state["ax"]) == tuple(state["goal"])', '    return False')

TOY_GOAL_ALWAYS = TOY_GOOD.replace(
    '    return (state["ay"], state["ax"]) == tuple(state["goal"])', '    return True')

TOY_DEGENERATE_VOLATILE = '''
import numpy as np

def parse_state(grid):
    return {"_raw": np.asarray(grid).copy()}

def render(state):
    return state["_raw"]

def step(state, action):
    return dict(state)

def is_goal(state):
    return False
'''

TOY_DEGENERATE_RAW = TOY_DEGENERATE_VOLATILE.replace("_raw", "raw")


class ToyEnv:
    """8x8 world: avatar (colour 3) walks to the goal marker (colour 4); cell
    (0,0) is an unpredictable HUD counter. Hand-built, so the expected verdicts
    are known independently of the agent."""

    H = W = 8
    GOAL = (4, 7)

    def __init__(self) -> None:
        self.ay, self.ax = 4, 1
        self.t = 0
        self.score = 0
        self.won = False

    def grid(self) -> np.ndarray:
        g = np.zeros((self.H, self.W), dtype=np.int64)
        g[0, 0] = 8 + (self.t % 8)          # never 3 or 4: no colour collision
        g[self.GOAL] = 4
        g[self.ay, self.ax] = 3
        return g

    def step(self, action: str) -> None:
        ay, ax = self.ay, self.ax
        if action == "ACTION1":
            ay -= 1
        elif action == "ACTION2":
            ay += 1
        elif action == "ACTION3":
            ax -= 1
        elif action == "ACTION4":
            ax += 1
        self.ay = min(7, max(1, ay))
        self.ax = min(7, max(0, ax))
        self.t += 1
        if not self.won and (self.ay, self.ax) == self.GOAL:
            self.won = True
            self.score += 1

    def frame(self) -> Any:
        env = self

        class F:
            grid = env.grid()
            levels_completed = env.score
            state = GameState.NOT_FINISHED
            available_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
        return F()


def _toy_timeline(n: int = 14) -> Timeline:
    tl = Timeline()
    env = ToyEnv()
    acts = ["ACTION4", "ACTION1", "ACTION4", "ACTION2", "ACTION3", "ACTION4"]
    tl.note_frame(env.grid(), 0)
    for i in range(n):
        a = acts[i % len(acts)]
        g0, s0 = env.grid(), env.score
        env.step(a)
        tl.record(g0, a, env.grid(), 1.0 if env.score > s0 else 0.0, False, 0)
    return tl


def _toy_win_timeline() -> Timeline:
    """A timeline that definitely contains the scoring transition."""
    tl = Timeline()
    env = ToyEnv()
    tl.note_frame(env.grid(), 0)
    for a in ["ACTION1", "ACTION2"] + ["ACTION4"] * 6:
        g0, s0 = env.grid(), env.score
        env.step(a)
        tl.record(g0, a, env.grid(), 1.0 if env.score > s0 else 0.0, False, 0)
    assert any(t.reward > 0 for t in tl.steps), "fixture must contain a win"
    return tl


def _certify(code: str, tl: Timeline) -> Report:
    sb = Sandbox()
    try:
        model = CandidateModel(code, sb)
    except SandboxError as e:
        return Report(ok=False, check=getattr(e, "label", "REJECTED"), bug=str(e))
    return Certifier(tl).certify(model, level=0)


# --- the tests -------------------------------------------------------------

def t_good_model_certifies() -> None:
    rep = _certify(TOY_GOOD, _toy_win_timeline())
    assert rep.ok, "the correct model must certify, got: " + rep.summary()
    assert rep.n_frames > 3 and rep.n_transitions >= 8, rep.summary()
    assert rep.accuracy == 1.0


def t_dropped_sprite_fails_check1_at_the_right_cell() -> None:
    rep = _certify(TOY_DROPS_GOAL, _toy_timeline())
    assert not rep.ok and "CHECK 1" in rep.check, rep.summary()
    assert "(4,7)" in rep.bug, "the bug must name the cell that went missing: " + rep.bug
    assert "actual=4" in rep.bug, rep.bug


def t_wrong_step_fails_check2_with_the_transition_index() -> None:
    rep = _certify(TOY_BAD_STEP, _toy_timeline())
    assert not rep.ok and "CHECK 2" in rep.check, rep.summary()
    assert re.search(r"transition #\d+", rep.bug), rep.bug
    assert "ACTION4" in rep.bug and "'ax'" in rep.bug, rep.bug


def t_goal_too_tight_fails_check3() -> None:
    rep = _certify(TOY_GOAL_NEVER, _toy_win_timeline())
    assert not rep.ok and "CHECK 3" in rep.check, rep.summary()
    assert "SCORED" in rep.bug, rep.bug


def t_goal_too_loose_fails_check3() -> None:
    rep = _certify(TOY_GOAL_ALWAYS, _toy_win_timeline())
    assert not rep.ok and "CHECK 3" in rep.check, rep.summary()
    assert "did NOT score" in rep.bug, rep.bug


def t_degenerate_states_fail_check0() -> None:
    rep = _certify(TOY_DEGENERATE_VOLATILE, _toy_timeline())
    assert not rep.ok and "CHECK 0" in rep.check, \
        "an all-volatile raw-grid state must not certify: " + rep.summary()
    rep2 = _certify(TOY_DEGENERATE_RAW, _toy_timeline())
    assert not rep2.ok and "CHECK 0" in rep2.check, rep2.summary()
    assert "verbatim copy" in rep2.bug, rep2.bug


def t_volatile_keys_are_ignored_by_replay_but_not_by_reconstruction() -> None:
    """The HUD is unpredictable. Declared volatile, the model certifies (proved
    above). Declared NON-volatile, replay must refute it -- otherwise the mask is
    doing nothing and CHECK 2 is not exact."""
    code = TOY_GOOD.replace('"_hud"', '"hud"')
    rep = _certify(code, _toy_timeline())
    assert not rep.ok and "CHECK 2" in rep.check, \
        "a non-volatile HUD must be refuted by replay: " + rep.summary()
    assert "'hud'" in rep.bug, rep.bug


def t_diagnostic_rule_escalates_to_the_representation() -> None:
    tl = _toy_timeline()
    sb = Sandbox()
    cert = Certifier(tl)
    for i in range(3):
        rep = cert.certify(CandidateModel(TOY_BAD_STEP, sb), level=0)
        assert not rep.ok
        if i == 0:
            assert not rep.suspect_representation, "one failure is not a pattern"
        else:
            assert rep.suspect_representation, "repeated CHECK 2 failures must escalate"
            assert "SUSPECT THE STATE REPRESENTATION" in rep.bug, rep.bug


def t_sandbox_rejects_escapes() -> None:
    base = TOY_GOOD
    cases = {
        "import os\n" + base: "import",
        base.replace("    g = np.asarray(grid)",
                     "    g = np.asarray(open('x').read())"): "open",
        base.replace("    g = np.asarray(grid)", "    g = grid.__class__"): "__class__",
        base.replace("def is_goal(state):", "def not_is_goal(state):"): "is_goal",
        "def parse_state(grid):\n    return {}\n": "missing",
    }
    for code, needle in cases.items():
        screened = screen_code(code)
        assert screened is not None, "escape not screened: " + needle
        label, reason = screened
        assert needle in reason or "missing" in reason, "%s -> %s" % (needle, reason)
        assert label.startswith("SCREEN "), "every screen rejection is labelled: " + label
    # numpy must be ALLOWED: refusing `np` is what sank the old JSON oracle.
    assert screen_code(TOY_GOOD) is None, screen_code(TOY_GOOD)


def t_no_rejection_is_ever_unlabelled() -> None:
    """54 of 63 rejections once came back as a bare `REJECTED`, so the histogram
    that was supposed to localise the failure localised nothing. Each of these is a
    different problem with a different fix, and each must SAY which it is."""
    tl = _toy_win_timeline()
    cases = {
        "def parse_state(grid):\n  return {\n": "SCREEN SYNTAX",
        "import os\n" + TOY_GOOD: "SCREEN IMPORT os",
        "def parse_state(grid):\n    return {}\n": "SCREEN MISSING DEF",
        "": "SCREEN EMPTY",
        TOY_GOOD.replace("def parse_state(grid):",
                         "def parse_state(grid):\n    return 7\ndef _dead(grid):"):
            "CONTRACT parse_state -> int",
    }
    for code, want in cases.items():
        rep = _certify(code, tl)
        assert not rep.ok, want
        assert rep.check.startswith(want), "%r -> %r" % (want, rep.check)
        assert rep.bug, "a labelled rejection still needs the pointed bug"
        assert "REJECTED" != rep.check and rep.check, rep.check

    # ...and the histogram is kept by the certifier itself, not only the harness.
    cert = Certifier(tl)
    for lbl in ("SCREEN SYNTAX line 2", "SCREEN SYNTAX line 2", "SCREEN IMPORT os"):
        cert.note_rejection(Report(ok=False, check=lbl, bug="because " + lbl))
    assert cert.reject_kinds == {"SCREEN SYNTAX line 2": 2, "SCREEN IMPORT os": 1}
    assert cert.n_rejected == 3
    assert cert.reject_examples["SCREEN IMPORT os"].startswith("because")

    # A check failure keeps carrying WHICH transition and WHAT differed.
    bad = _certify(TOY_BAD_STEP, tl)
    assert bad.check == "CHECK 2 REPLAY" and re.search(r"transition #\d+", bad.bug)
    assert "predicted" in bad.bug, bad.bug


def t_sandbox_times_out_a_spinner() -> None:
    code = TOY_GOOD.replace("    g = np.asarray(grid)",
                            "    while True:\n        pass")
    rep = _certify(code, _toy_timeline())
    assert not rep.ok and rep.check.startswith(("TIMEOUT", "CRASH")), rep.summary()
    assert "exceeded" in rep.bug or "abandoned" in rep.bug, rep.bug


def t_bfs_finds_the_shortest_path() -> None:
    sb = Sandbox()
    model = CandidateModel(TOY_GOOD, sb)
    env = ToyEnv()
    plan = Planner(sb).run_bfs(model, env.grid(), ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
    assert plan, plan.summary()
    assert len(plan.actions) == 6, "shortest route is 6 rights, got " + str(plan.actions)
    assert set(plan.actions) == {"ACTION4"}, plan.actions
    assert len(plan.expected) == len(plan.actions)
    assert plan.distinct > 6 and plan.nodes > 6, plan.summary()


def t_bfs_reports_unreachable_instead_of_lying() -> None:
    """The optimism trigger's input: a model whose goal cannot be reached must say
    so, because that fact is evidence the model is wrong."""
    code = TOY_GOOD.replace("        ax += 1", "        pass")
    sb = Sandbox()
    model = CandidateModel(code, sb)
    plan = Planner(sb).run_bfs(model, ToyEnv().grid(), ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
    assert not plan.reached, plan.summary()
    assert "EXHAUSTED" in plan.reason or "budget" in plan.reason, plan.reason


def t_discriminating_action_splits_the_survivors() -> None:
    sb = Sandbox()
    models = [CandidateModel(TOY_GOOD, sb), CandidateModel(TOY_BAD_STEP, sb)]
    act, reason = Planner(sb).disagreement_action(
        models, ToyEnv().grid(), ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
    assert act == "ACTION4", "only the rightward move separates these two: %s (%s)" % (act, reason)


def t_timeline_is_append_only_and_samples_the_whole_history() -> None:
    tl = _toy_win_timeline()
    steps = tl.steps
    assert isinstance(steps, tuple), "callers must not be handed a mutable history"
    n = len(tl)
    steps_copy = list(steps)
    steps_copy.clear()
    assert len(tl) == n, "history changed after a caller mutated its copy"
    g = tl.grid(tl.steps[0].prev_key)
    try:
        g[0, 0] = 99
        raise AssertionError("interned frames must be read-only")
    except ValueError:
        pass
    sample = tl.certify_sample(0, limit=4)
    assert any(t.reward > 0 for t in sample), \
        "the sample must always include the scoring transition"
    assert sample[0].idx == min(t.idx for t in sample), "sample must be oldest-first"


def t_canon_normalizes_noise_but_not_meaning() -> None:
    assert canon({"a": (1, 2)}) == canon({"a": [1, 2]}), "tuple/list is not a claim"
    assert canon({"a": np.int8(3)}) == canon({"a": 3}), "dtype is not a claim"
    assert canon({"a": True}) == canon({"a": 1})
    assert canon({"a": 1, "b": 2}) == canon({"b": 2, "a": 1}), "key order is not a claim"
    assert canon({"a": 1}) != canon({"a": 2}), "values ARE a claim"
    assert canon({"a": np.zeros((2, 2))}) != canon({"a": np.zeros((2, 3))})
    assert mask_volatile({"a": 1, "_t": 5, "b": {"_u": 1, "v": 2}}) == {"a": 1, "b": {"v": 2}}
    assert not isinstance(bool(np.False_), np.bool_)     # np.False_ is not False


def t_pointed_bugs_point() -> None:
    d = first_difference({"ay": 4, "ax": 3}, {"ay": 4, "ax": 2})
    assert "'ax'" in d and "predicted 3" in d and "actual 2" in d, d
    d2 = first_difference(np.zeros((2, 2), int), np.array([[0, 0], [0, 7]]))
    assert "[1, 1]" in d2 and "actual=7" in d2, d2
    d3 = first_difference({"a": 1}, {"a": 1, "b": 2})
    assert "MISSING" in d3, d3
    assert first_difference({"a": 1}, {"a": 1}) == ""


def t_toolset_is_four_and_commit_is_the_only_channel() -> None:
    import tempfile
    globals()["WORK_DIR"] = tempfile.mkdtemp(prefix="arc_selftest_")
    agent = MyAgent(game_id="selftest-tools")
    assert len(agent.delib.TOOLS) == 4, agent.delib.TOOLS
    assert set(agent.delib.TOOLS) == {"write_code", "run_backtest", "run_bfs", "commit_actions"}
    # every path that emits an action must go through enqueue()
    src = open(__file__, encoding="utf-8").read()
    body = src.split("class MyAgent(Agent):", 1)[1].split("COMPONENT 10", 1)[0]
    assert body.count("self.queue.append(") == 1, \
        "the queue must only be filled by enqueue()"
    assert "def enqueue" in body


def t_end_to_end_offline_plan_and_bank_a_level() -> None:
    """The commander path with no LLM: a pre-certified model is re-certified
    against live evidence, BFS plans through it, the plan executes under the
    per-step self-check, and a level is banked."""
    import tempfile
    globals()["WORK_DIR"] = tempfile.mkdtemp(prefix="arc_selftest_")
    env = ToyEnv()
    agent = MyAgent(game_id="selftest-e2e")
    agent.MAX_ACTIONS = 40
    agent.delib.model = CandidateModel(TOY_GOOD, agent.sandbox)
    agent.delib.code = TOY_GOOD
    frames = [env.frame()]
    while agent.action_counter < 40 and not agent.level_costs:
        agent.choose_action(frames, frames[-1])
        env.step(agent.last_action_str)
        frames.append(env.frame())
        agent.action_counter += 1
    agent.choose_action(frames, frames[-1])      # let it observe the reward
    assert agent.level_costs, "the agent must bank the toy level within 40 actions"
    assert agent.action_counter <= 20, "banked in %d actions (expected ~10)" % agent.action_counter
    assert agent.mispredictions == 0, "a correct model must not mispredict"
    assert agent.delib.n_plans >= 1 and agent.delib.n_probes >= 2
    assert len(agent.timeline) >= 6


def t_misprediction_voids_the_plan() -> None:
    """Reality outranks the model: with a wrong step(), the first mismatch must
    drop the rest of the plan rather than play it out."""
    import tempfile
    globals()["WORK_DIR"] = tempfile.mkdtemp(prefix="arc_selftest_")
    env = ToyEnv()
    agent = MyAgent(game_id="selftest-void")
    agent.MAX_ACTIONS = 25
    # a model that believes ACTION4 moves two cells: BFS plans 3 rights, reality
    # delivers one, so the second queued action must never be played.
    agent.delib.model = CandidateModel(TOY_BAD_STEP, agent.sandbox)
    agent.delib.code = TOY_BAD_STEP
    agent.certifier.check2_fail_streak = 0
    frames = [env.frame()]
    for _ in range(12):
        agent.choose_action(frames, frames[-1])
        env.step(agent.last_action_str)
        frames.append(env.frame())
        agent.action_counter += 1
    # the wrong model cannot certify against real transitions, so the agent must
    # notice: either the backtest rejected it or the live self-check fired.
    assert agent.certifier.n_rejected >= 1 or agent.mispredictions >= 1, \
        "a wrong model must be caught by the backtest or by the live self-check"
    assert agent.plan_aborts == 0 or agent.mispredictions >= 1


TOY_UNREACHABLE_GOAL = TOY_GOOD.replace(
    '    return (state["ay"], state["ax"]) == tuple(state["goal"])',
    '    return state["ax"] == 99')          # certifies, but BFS can never satisfy it


class _StubLLM(LLMCoder):
    """A scripted writer. Exercises the real deliberate() loop -- prompts,
    extraction, certification, BFS, commit -- without the 27B model."""

    def __init__(self, replies: List[str]) -> None:
        super().__init__("")
        self.replies = list(replies)
        self.prompts: List[str] = []
        self.enabled, self.load_error = True, ""

    def load(self) -> bool:
        return True

    def generate(self, system: str, user: str, **kw: Any) -> Optional[str]:
        self.prompts.append(user)
        self.calls += 1
        return self.replies.pop(0) if self.replies else None


def _fenced(code: str, notes: str = "") -> str:
    out = "Here is my model.\n```python\n%s\n```" % code
    if notes:
        out += "\n```notes\n%s\n```" % notes
    return out


def _run_stubbed(replies: List[str], cap: int = 40) -> Tuple["MyAgent", ToyEnv, _StubLLM]:
    import tempfile
    globals()["WORK_DIR"] = tempfile.mkdtemp(prefix="arc_selftest_")
    env = ToyEnv()
    agent = MyAgent(game_id="selftest-stub")
    agent.MAX_ACTIONS = cap
    stub = _StubLLM(replies)
    agent.llm = stub
    agent.delib.llm = stub
    frames = [env.frame()]
    while agent.action_counter < cap and not agent.level_costs:
        agent.choose_action(frames, frames[-1])
        env.step(agent.last_action_str)
        frames.append(env.frame())
        agent.action_counter += 1
    agent.choose_action(frames, frames[-1])       # observe the final reward
    return agent, env, stub


def t_inner_loop_repairs_from_a_pointed_bug_and_banks_a_level() -> None:
    """theorize -> certify -> theorize again from the counterexample -> plan ->
    commit. The second prompt must contain the pointed bug, not a bare failure."""
    agent, _env, stub = _run_stubbed([
        _fenced(TOY_BAD_STEP),                                  # refuted by CHECK 2
        _fenced(TOY_GOOD, "right moves exactly one cell"),      # certifies
    ])
    assert stub.calls >= 2, "the refutation must trigger a second attempt"
    repair = stub.prompts[1]
    assert "REFUTED" in repair and re.search(r"transition #\d+", repair), repair[-600:]
    assert "state['ax']: predicted" in repair, "the repair prompt must carry the diff"
    assert "YOUR CURRENT world_model.py" in repair, "the model must see its own code"
    assert agent.delib.model is not None and agent.delib.n_plans >= 1
    assert agent.level_costs, "the repaired model must bank the level"
    assert agent.mispredictions == 0, "the certified model must not mispredict"
    assert os.path.isfile(os.path.join(agent.work_dir, "world_model.py"))
    assert os.path.isfile(os.path.join(agent.work_dir, "notes.md"))


def t_optimism_trigger_fires_when_no_win_is_reachable() -> None:
    """[E] A certified model that admits no winning path is evidence the MODEL is
    wrong; the agent must say so and ask for a revision, not give up."""
    agent, _env, stub = _run_stubbed([
        _fenced(TOY_UNREACHABLE_GOAL),      # passes all three checks; BFS finds nothing
        _fenced(TOY_GOOD),                  # the revision
    ])
    assert agent.delib.n_optimism >= 1, "the optimism trigger must fire"
    opt = stub.prompts[1]
    assert "reaches no winning state" in opt, opt[-500:]
    assert "EXHAUSTED" in opt or "budget" in opt, opt[-500:]
    assert "EXPERIMENT:" in opt, "the revision must be asked to name one experiment"
    assert agent.level_costs, "the revised model must then bank the level"


def t_unusable_replies_do_not_stall_the_game() -> None:
    """A writer that emits prose, or nothing at all, must still leave the agent
    playing: commit_actions is the only channel, and it always gets something."""
    agent, _env, stub = _run_stubbed(["I think the avatar should move right.", "", None], cap=24)
    assert agent.action_counter >= 20, "the agent must keep acting"
    assert agent.delib.n_probes >= 1
    assert agent.delib.model is None
    # One round must buy more than one action. This is what replaced the doubling
    # timer: a round that found nothing commits a BATCH of informative actions, so
    # generation is amortised without ever playing an action to pass the time.
    assert agent.delib.rounds <= agent.action_counter / 2, \
        "%d deliberations in %d actions: the probe batch is not amortising" % (
            agent.delib.rounds, agent.action_counter)


def t_no_action_is_ever_spent_to_pass_time() -> None:
    """The measured finding, as a test. 1446 of 1500 actions were filler on a doubling
    timer; the gate is now the EVIDENCE, and a probe must buy a transition nobody
    has. `route_tag` is the ledger the harness reads, so assert on that."""
    src = open(__file__, encoding="utf-8").read().split("COMPONENT 10", 1)[0]
    live = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "buying evidence for the next deliberation" not in live, \
        "the filler probe is back (it is kept only as a comment, on purpose)"
    assert "delib_backoff" not in live, "the doubling timer is back"
    assert route_tag("idle: every action's outcome here is already recorded (x)") \
        == "probe_idle", "an idle action must be attributable as idle"

    import tempfile
    globals()["WORK_DIR"] = tempfile.mkdtemp(prefix="arc_selftest_")
    agent = MyAgent(game_id="selftest-idle")
    agent.grid = np.zeros((4, 4), dtype=np.int64)
    agent.available = ["ACTION1", "ACTION2"]
    # Nothing recorded yet: both actions are novel, and a batch of 2 is distinct.
    assert agent.delib.informative_actions(2) == ["ACTION1", "ACTION2"]
    assert agent.idle_actions == 0
    # Record both as world-no-ops from this exact frame. Acting here can now teach
    # us nothing, so the probe must SAY that rather than press a button silently.
    for a in ("ACTION1", "ACTION2"):
        agent.timeline.record(agent.grid, a, agent.grid, 0.0, False, 0)
    assert agent.timeline.inert(agent.grid, "ACTION1")
    assert agent.delib.informative_actions(2) == []
    agent.delib.probe("nothing left")
    assert agent.idle_actions == 1, "the fidget must be counted, not hidden"
    assert route_tag(agent.queue[0][2]) == "probe_idle", agent.queue[0][2]
    # ...and an action that DID something stays informative, tried or not.
    agent.timeline.record(agent.grid, "ACTION2", np.ones((4, 4), dtype=np.int64),
                          0.0, False, 0)
    assert not agent.timeline.inert(agent.grid, "ACTION2")
    assert agent.delib.informative_actions(2) == ["ACTION2"]


def t_the_writer_reserves_what_a_call_costs() -> None:
    """22 calls of ~90s ran up 1980s against a declared 1800s ceiling, because the
    gate read only what was already spent. The next call must be priced in."""
    llm = LLMCoder("")
    llm.enabled, llm.load_error = True, ""
    llm.budget_s = 100.0
    llm.calls, llm.gen_s, llm.spent_s = 2, 80.0, 80.0      # 40s a call, 80s spent
    assert llm.avg_call_s == 40.0
    assert llm.has_budget(), "80s of 100s with nothing reserved still reads as open"
    assert not llm.has_budget(llm.avg_call_s), "80 + a 40s call overruns the 100s ceiling"
    llm.spent_s = 40.0
    assert llm.has_budget(llm.avg_call_s), "40 + 40 fits in 100s"


def t_repair_escalates_instead_of_asking_the_same_question() -> None:
    """Sixteen repair calls once returned near-identical files. A repair must vary:
    a hotter temperature, a framing that asks for a different KIND of change, and
    the dead ends listed back so they cannot be re-emitted for free."""
    tl = _toy_win_timeline()
    ctx = Context(game_id="demo", level=1, grid=ToyEnv().grid(),
                  available=["ACTION1"], action_summary=tl.action_effect_summary(0),
                  volatile_hint=tl.volatile_cell_hint(0),
                  reward_summary=tl.reward_summary(), n_transitions=len(tl),
                  code=TOY_GOOD)
    rep = _certify(TOY_BAD_STEP, tl)
    prompts = [prompt_repair(ctx, rep, attempt=i) for i in range(len(REPAIR_FRAMINGS) + 2)]
    assert len(set(prompts[:len(REPAIR_FRAMINGS)])) == len(REPAIR_FRAMINGS), \
        "each attempt must ask for a different kind of change"
    assert "RE-GROUND parse_state" in prompts[1]
    assert "SIMPLIFY IT" in prompts[2]
    assert "abandon the current file" in prompts[3]
    assert prompts[-1] == prompts[len(REPAIR_FRAMINGS) - 1], \
        "past the last framing it must stay on the last one, not crash"

    # the dead ends, listed back, with the pointed bug intact
    with_hist = prompt_repair(ctx, rep, attempt=1,
                              refuted=[("CHECK 2 REPLAY", "transition #3 went wrong"),
                                       ("SCREEN SYNTAX line 4", "SyntaxError: bad")])
    assert "ALREADY REFUTED THIS ROUND" in with_hist
    assert "transition #3 went wrong" in with_hist and "SCREEN SYNTAX line 4" in with_hist
    assert "rejected WITHOUT being run" in with_hist

    # the temperature actually reaches the writer, and rises with the attempt.
    # MethodType, because generate() routes an UNBOUND stub down the one-prompt
    # harness path -- see t_harness_hooks_exist_and_the_stub_shape_is_handled.
    import types
    seen: List[Optional[float]] = []
    llm = LLMCoder("")
    llm.enabled, llm.load_error = True, ""
    llm.load = lambda: True                       # type: ignore[method-assign]
    llm._generate_inner = types.MethodType(       # type: ignore[method-assign]
        lambda self, system, user, max_new_tokens, max_time, temperature=None:
        (seen.append(temperature), "```python\n%s\n```" % TOY_GOOD)[1], llm)
    for a in range(3):
        llm.generate("SYS", "USER",
                     temperature=min(REPAIR_TEMP_MAX,
                                     LLM_TEMPERATURE + REPAIR_TEMP_STEP * a))
    assert seen == sorted(seen) and seen[0] < seen[-1], seen
    assert seen[-1] <= REPAIR_TEMP_MAX


def t_a_repeated_reply_is_not_backtested_twice() -> None:
    """A writer that re-emits the same file must not be paid for it twice: the
    duplicate is counted, the framing escalates, and the certifier never sees it."""
    agent, _env, stub = _run_stubbed([
        _fenced(TOY_BAD_STEP),      # refuted by CHECK 2
        _fenced(TOY_BAD_STEP),      # ...and returned again, verbatim
        _fenced(TOY_GOOD),          # then a real change
    ])
    assert agent.delib.n_repeat_replies >= 1, "the duplicate must be counted"
    backtests = agent.certifier.n_certified + agent.certifier.n_rejected
    assert backtests <= 2, "the duplicate was backtested again: %d backtests" % backtests
    assert agent.level_costs, "and the loop must still get to the working model"


def t_level_up_hands_back_the_model_for_editing() -> None:
    agent, _env, stub = _run_stubbed([_fenced(TOY_GOOD)], cap=40)
    assert agent.level_costs, "must bank the toy level first"
    agent.pending_levelup = True
    agent.delib.llm.replies = [_fenced(TOY_GOOD)]
    ctx = agent.context()
    p = prompt_levelup(ctx, agent.delib.code)
    assert "EDIT it minimally" in p and "def parse_state" in p
    assert "your starting point" in p, "the level-up prompt must frame the model as a base"
    assert "rules almost certainly carry" in p


def t_derived_watchdog_covers_its_parts() -> None:
    assert DELIBERATE_MAX_S >= DELIBERATE_MAX_TURNS * (LLM_GEN_MAX_TIME_S + CERT_MAX_S)
    assert PLAN_MAX_LEN > 0 and BFS_MAX_DEPTH >= PLAN_MAX_LEN


def t_extractors_recover_real_answers() -> None:
    """Valid models used to be lost at the READER, not the writer."""
    body = "def parse_state(grid):\n    return {}\ndef render(s):\n    return s\n" \
           "def step(s, a):\n    return s\ndef is_goal(s):\n    return False\n"
    assert extract_code("prose\n```python\n%s```\ntrailing" % body).startswith("def parse_state")
    assert extract_code("```\n%s```" % body).startswith("def parse_state")
    assert extract_code("here it is:\n" + body).startswith("def parse_state")
    assert extract_code("no code at all") is None
    assert "keep going" in extract_notes("```python\nx=1\n```\n```notes\nkeep going\n```")
    assert extract_experiment("blah\nEXPERIMENT: ACTION6_r3_c9\n", ["ACTION6"]) == "ACTION6_r3_c9"
    assert extract_experiment("EXPERIMENT: ACTION2", ["ACTION1", "ACTION2"]) == "ACTION2"
    assert extract_experiment("no line here", ["ACTION1"]) is None


def t_click_coordinates_survive_the_commit_channel() -> None:
    """Regression: upper-casing an action turns ACTION6_r3_c9 into a name nothing
    recognises, which degrades silently into a coordinate-less button press."""
    import tempfile
    globals()["WORK_DIR"] = tempfile.mkdtemp(prefix="arc_selftest_")
    agent = MyAgent(game_id="selftest-click")
    agent.enqueue(["ACTION6_r3_c9", "action6 r12 c30", "ACTION6_R7_C1", "action2"])
    got = [a for a, _e, _w in agent.queue]
    assert got == ["ACTION6_r3_c9", "ACTION6_r12_c30", "ACTION6_r7_c1", "ACTION2"], got
    for a in got[:3]:
        m = CLICK_RE.match(a)
        assert m, "committed click %r is unparseable" % a
    assert norm_action("ACTION6_r0_c0") == "ACTION6_r0_c0"


def t_click_probe_advances_instead_of_repeating_one_cell() -> None:
    """Regression: on a click-only game the probe tally was keyed on the full
    action string, so ACTION6 always looked untried and cell (0,0) was replayed
    for the entire budget."""
    import tempfile
    globals()["WORK_DIR"] = tempfile.mkdtemp(prefix="arc_selftest_")
    agent = MyAgent(game_id="selftest-clickprobe")
    agent.grid = np.zeros((64, 64), dtype=np.int64)
    agent.available = ["ACTION6"]
    seen = []
    for _ in range(6):
        agent.delib.probe("test")
        act, _e, _w = agent.queue.popleft()
        seen.append(act)
        agent.action_counts[act] = agent.action_counts.get(act, 0) + 1
    assert len(set(seen)) == 6, "the click probe must sweep, not repeat: %s" % seen
    assert agent.base_count("ACTION6") == 6, agent.base_count("ACTION6")
    assert agent.base_count("ACTION6_r0_c0") == 6, "clicks aggregate under one base"


def t_click_action_space_is_generic() -> None:
    """Click targets come from the model's own state plus a lattice. No game-id
    branch, no hand-written button finder."""
    sb = Sandbox()
    model = CandidateModel(TOY_GOOD, sb)
    st = model.parse(ToyEnv().grid())
    acts = Planner(sb).action_space(model, st, ["ACTION6"], (8, 8))
    assert all(a.startswith("ACTION6_r") for a in acts), acts
    assert "ACTION6_r4_c7" in acts, "the goal coordinate in the state must be a candidate"
    assert len(acts) <= CLICK_MAX_CANDIDATES
    # Guardrail 4, mechanically: no game id may appear in the agent's own code.
    # Scan only the agent half of the file -- this test names the ids itself.
    agent_src = open(__file__, encoding="utf-8").read().split("COMPONENT 10", 1)[0]
    for gid in ("lp8" + "5", "tr8" + "7", "tu9" + "3", "ls2" + "0",
                "cn0" + "4", "sc2" + "5", "ar2" + "5", "r11" + "l"):
        assert gid not in agent_src, "per-game branch found for " + gid


TOY_THREE_BUTTONS = '''
import numpy as np

def parse_state(grid):
    g = np.asarray(grid)
    return {"buttons": [(1, 1), (2, 2), (3, 3)], "_raw_sum": int(g.sum())}

def render(state):
    return np.zeros((8, 8), dtype=int)

def step(state, action):
    return dict(state)

def is_goal(state):
    return False
'''


def t_click_space_is_buttons_or_lattice_never_both() -> None:
    """The branching factor IS the search depth. A model that names its targets
    must be searched over THOSE, not over them plus a screen-wide sweep: five
    certified models found zero plans while every node had ~48 children."""
    sb = Sandbox()
    lattice_n = len(range(0, 64, CLICK_STRIDE)) ** 2

    named = CandidateModel(TOY_THREE_BUTTONS, sb)
    st = named.parse(np.zeros((64, 64), dtype=np.int64))
    acts = Planner(sb).action_space(named, st, ["ACTION6"], (64, 64))
    assert acts == ["ACTION6_r1_c1", "ACTION6_r2_c2", "ACTION6_r3_c3"], acts
    assert len(acts) < lattice_n, "the blind lattice must be gone once targets are named"

    vague = CandidateModel(TOY_GOOD, sb)                    # names exactly one coord
    st2 = vague.parse(ToyEnv().grid())
    acts2 = Planner(sb).action_space(vague, st2, ["ACTION6"], (8, 8))
    assert "ACTION6_r4_c7" in acts2, "the one named target must survive"
    assert len(acts2) > 1, "a model too vague to name targets still gets the lattice"
    assert len(acts2) <= CLICK_MAX_CANDIDATES


def t_optimism_asks_for_is_goal_when_the_search_exhausted() -> None:
    """A search that ran out of STATES proves no state the rules can build
    satisfies is_goal; a search that ran out of BUDGET proves nothing about it.
    The two must not ask for the same revision."""
    tl = _toy_timeline()
    ctx = Context(game_id="demo", level=1, grid=ToyEnv().grid(),
                  available=["ACTION6"], action_summary=tl.action_effect_summary(0),
                  volatile_hint=tl.volatile_cell_hint(0),
                  reward_summary=tl.reward_summary(), n_transitions=len(tl),
                  code=TOY_GOOD)
    out = prompt_optimism(ctx, Planner(Sandbox()).run_bfs(
        CandidateModel(TOY_GOAL_NEVER, Sandbox()), ctx.grid, ["ACTION1", "ACTION2"]))
    assert "EXHAUSTED" in out and "REVISE is_goal FIRST" in out, out[-900:]

    cut = Plan(reached=False, exhausted=False,
               reason="BFS hit its 10.0s budget after 20000 nodes")
    out2 = prompt_optimism(ctx, cut)
    assert "BUDGET" in out2 and "FEWER click targets" in " ".join(out2.split()), out2[-900:]
    assert "REVISE is_goal FIRST" not in out2, "a budget cut-off does not refute is_goal"
    for o in (out, out2):
        assert "reaches no winning state" in o and "EXPERIMENT:" in o


def t_the_writer_is_loaded_once_per_process() -> None:
    """A 27B model per game means 25 loads of ~54 GB. Two agents must SHARE the
    writer, while the per-game budget ledger still starts from zero."""
    import tempfile
    globals()["WORK_DIR"] = tempfile.mkdtemp(prefix="arc_selftest_")
    a = MyAgent(game_id="selftest-singleton-a")
    a.llm.calls, a.llm.spent_s = 7, 123.0
    b = MyAgent(game_id="selftest-singleton-b")
    assert a.llm is b.llm, "the writer must be a process singleton"
    assert b.llm is get_local_llm()
    assert b.llm.calls == 0 and b.llm.spent_s == 0.0, "per-game ledger must reset"
    assert a.delib.llm is a.llm, "the deliberator must hold the same writer"


def t_harness_hooks_exist_and_the_stub_shape_is_handled() -> None:
    """Kaggle_test.py resolves the writer by the name `LocalLLM`, times
    `ensure_loaded`, wraps `generate`, and `--fake-llm` replaces
    `_generate_inner` with a ONE-prompt stub. Each of those must land."""
    assert LocalLLM is LLMCoder
    llm = LLMCoder("")
    llm.enabled, llm.load_error = True, ""
    llm.load = lambda: True                       # type: ignore[method-assign]
    seen: List[str] = []

    def fake_inner(prompt, max_new_tokens=0, temperature=0.0):
        seen.append(prompt)
        return "canned"
    llm._generate_inner = fake_inner              # type: ignore[method-assign]
    out = llm.generate("SYS", "USER")
    assert out == "canned", out
    assert seen and "SYS" in seen[0] and "USER" in seen[0], \
        "a one-prompt stub must receive BOTH turns, not `user` as a token count"
    assert llm.calls == 1 and llm.gen_s >= 0.0
    assert llm.ensure_loaded() is True


def t_the_session_clock_divides_what_is_left_not_what_was_planned() -> None:
    """25 games x the 1800s ceiling is 12.5 h of generation in a 9 h session, so
    the last games of the ordering never run -- and an unrun game scores exactly
    what a failed one does. The share must fall out of the clock, and a game that
    finishes early must hand its unused time to the rest."""
    c = SessionClock(total_s=1000.0, games=4, reserve_s=100.0)
    assert c.games_left() == 4, c.games_left()
    assert abs(c.share_s() - 225.0) < 2.0, c.share_s()   # (1000-100)/4

    c.start_game()
    assert c.games_left() == 4 and abs(c.share_s() - 225.0) < 2.0
    c.start_game()
    assert c.games_left() == 3, "three games remain once the second has begun"
    # Nothing was spent, so the three survivors now get MORE than the even split.
    assert c.share_s() > 225.0, c.share_s()

    c.t0 -= 800.0                                        # 800s consumed
    assert abs(c.remaining_s() - 100.0) < 2.0, c.remaining_s()
    assert abs(c.share_s() - 33.3) < 2.0, c.share_s()
    c.t0 -= 500.0                                        # past the wall
    assert c.remaining_s() == 0.0, "remaining time is never negative"


def t_a_games_writer_budget_is_the_share_and_the_wall_is_absolute() -> None:
    """new_game() derives the allowance; has_budget() refuses a call the session
    cannot finish. Gating on spend alone once let 22 calls of ~90s run up 1980s
    against a declared 1800s ceiling."""
    global _CLOCK
    saved = _CLOCK
    try:
        _CLOCK = SessionClock(total_s=4000.0, games=4, reserve_s=0.0)
        llm = LLMCoder("")
        llm.enabled, llm.load_error = True, ""
        llm.new_game()
        # (4000/4) * 0.75 = 750s, well under the 1800s ceiling -> the clock binds.
        assert abs(llm.budget_s - 1000.0 * LLM_SHARE_OF_GAME) < 5.0, llm.budget_s
        assert llm.budget_s < LLM_BUDGET_S, "the ceiling must not be the allowance"

        llm.spent_s = llm.budget_s - 1.0
        assert not llm.has_budget(50.0), "must not start a call it cannot pay for"
        assert llm.has_budget(0.0), "a free check is still allowed"

        # The session wall outranks a per-game ledger that still looks healthy.
        llm.spent_s = 0.0
        _CLOCK.t0 -= 3990.0
        assert not llm.has_budget(100.0), "no call may outlive the session"
        _CLOCK.t0 -= 100.0
        assert not llm.has_budget(), "a spent session reports no writer at all"
    finally:
        _CLOCK = saved


def t_an_exhausted_session_ends_the_game_instead_of_being_killed_in_it() -> None:
    """Returning at the wall is not giving up -- the clock is spent either way,
    and stopping lets the write-out happen inside the reserve."""
    global _CLOCK
    saved = _CLOCK
    import tempfile
    globals()["WORK_DIR"] = tempfile.mkdtemp(prefix="arc_selftest_")
    try:
        _CLOCK = SessionClock(total_s=600.0, games=2, reserve_s=60.0)
        ag = MyAgent(game_id="selftest-session-wall")
        ag.action_counter = 0
        assert ag.is_done([], None) is False, "a live session keeps playing"
        _CLOCK.t0 -= 600.0
        assert ag.is_done([], None) is True, "a spent session ends the game"
    finally:
        _CLOCK = saved


def t_a_crash_costs_one_action_not_the_rest_of_the_session() -> None:
    """The harness drives choose_action in a bare loop. An exception there ends
    the game and, on Kaggle, plausibly the run -- so every game after it scores
    zero too. A wrong action is recoverable; a traceback is not."""
    import tempfile
    globals()["WORK_DIR"] = tempfile.mkdtemp(prefix="arc_selftest_")
    ag = MyAgent(game_id="selftest-crash")
    ag.available = ["ACTION1", "ACTION2", "ACTION6"]
    ag.grid = np.zeros((8, 8), dtype=np.int64)

    boom = {"n": 0}

    def explode(frames, latest_frame):
        boom["n"] += 1
        raise RuntimeError("deliberation went wrong")
    ag._choose_action_inner = explode          # type: ignore[method-assign]

    out = ag.choose_action([], None)
    assert boom["n"] == 1 and out is not None, out
    assert ag.crashes == 1, ag.crashes
    assert ag.last_action_str in ("ACTION1", "ACTION2"), ag.last_action_str
    assert ag._route == "crash-fallback", ag._route
    assert ag.pending is not None, "the step must still be recorded and learned from"
    assert ag.stats()["crashes"] == 1, "the count has to reach the report"

    # Click-only games have no movement button to fall back on: the fallback must
    # still be a legal, in-bounds click rather than a bare ACTION6.
    ag.available = ["ACTION6"]
    ag.choose_action([], None)
    m = CLICK_RE.match(ag.last_action_str)
    assert m, ag.last_action_str
    assert 0 <= int(m.group(1)) < 8 and 0 <= int(m.group(2)) < 8, ag.last_action_str

    # And a crash inside is_done falls back to the action cap, never propagates.
    ag._is_done_inner = explode                # type: ignore[method-assign]
    ag.action_counter = ag.MAX_ACTIONS
    assert ag.is_done([], None) is True
    ag.action_counter = 0
    assert ag.is_done([], None) is False


def t_the_config_banner_names_the_binding_ceiling() -> None:
    """Every expensive surprise so far was a number nobody printed. The report
    must say which budget binds, and the numbers in it must be consistent."""
    rep = config_report()
    assert rep["writer_allowance_s"] <= rep["writer_ceiling_s"]
    assert rep["writer_allowance_s"] <= rep["per_game_wall_s"]
    assert abs(rep["ceiling_x_games_h"]
               - LLM_BUDGET_S * SESSION_GAMES / 3600.0) < 1e-6
    binds_clock = rep["ceiling_x_games_h"] > rep["session_h"]
    assert rep["binds"] == ("session clock" if binds_clock else "per-game ceiling")
    assert json.loads(json.dumps(rep)) == rep, "the banner must be JSON-clean"
    # The shipped defaults are the case that matters: 25 x 1800s cannot fit in 8.5 h.
    if LLM_BUDGET_S == 1800.0 and SESSION_GAMES == 25:
        assert binds_clock, "the shipped ceiling must be recognised as unaffordable"


SELFTESTS = [v for k, v in sorted(globals().items()) if k.startswith("t_") and callable(v)]


def run_selftests() -> int:
    failures = 0
    for fn in SELFTESTS:
        t0 = time.time()
        try:
            fn()
            print("PASS  %-52s %5.2fs" % (fn.__name__, time.time() - t0))
        except AssertionError as e:
            failures += 1
            print("FAIL  %-52s %s" % (fn.__name__, e))
        except Exception:                            # noqa: BLE001
            failures += 1
            print("ERROR %-52s\n%s" % (fn.__name__, traceback.format_exc(limit=6)))
    print("\n%d/%d passed" % (len(SELFTESTS) - failures, len(SELFTESTS)))
    return 1 if failures else 0


# ============================================================================
# COMPONENT 11: --bench-llm  (STEP 0: measure, do not assume)
# ============================================================================

BENCH_PROMPT = ("Write a Python function that takes a 2-D numpy integer array and returns "
                "a dict describing every connected region of equal colour: its colour, its "
                "bounding box and its cell count. Then explain your choice of data "
                "structure in three sentences.")


def bench_llm(rounds: int = 2, max_new_tokens: int = 512) -> int:
    """Print the two numbers that decide whether this architecture is affordable:
    whether the flash-linear-attention fast path is present, and tokens/sec.

    A deliberation is up to %d generations. At the pre-wheel 216-336 s per call
    that is over half an hour of thinking per action -- unaffordable. Run this
    before the wheels, after the wheels, and once more against FP8 weights (point
    ARC_LLM_PATH at them).
    """ % DELIBERATE_MAX_TURNS
    llm = get_local_llm()
    print("model path requested : %s" % llm.requested_path)
    print("model dir resolved   : %s" % llm.model_dir)
    if not llm.load():
        print("LOAD FAILED: %s" % llm.load_error)
        return 1
    print("flash-linear-attention: %s" % ("PRESENT (fast path)" if llm.fla_present
                                          else "ABSENT -> torch fallback, expect 200s+ calls"))
    print("causal_conv1d         : %s" % ("PRESENT" if getattr(llm, "conv1d_present", False)
                                          else "ABSENT"))
    print("dtype/footprint       : %.1f GB, load %.1fs" % (llm.footprint_gb, llm.load_s))
    if torch is not None and torch.cuda.is_available():
        print("device                : %s" % torch.cuda.get_device_name(0))
    for i in range(max(1, rounds)):
        t0 = time.time()
        before = llm.out_tokens
        out = llm.generate(SYSTEM_PROMPT, BENCH_PROMPT, max_new_tokens=max_new_tokens,
                           max_time=LLM_GEN_MAX_TIME_S)
        dt = time.time() - t0
        n = llm.out_tokens - before
        tag = "warmup" if i == 0 else "round %d" % i
        print("%-8s %4d tokens in %6.1fs = %6.2f tok/s%s"
              % (tag, n, dt, n / dt if dt > 0 else 0.0, "" if out else "  (GENERATION FAILED)"))
    print("\noverall %.2f tok/s over %d call(s)" % (llm.tokens_per_s, llm.calls))
    per_call = llm.gen_s / max(1, llm.calls)
    # The projection is now bounded by the LLM BUDGET, not by an actions-per-round
    # timer: deliberation is gated on the evidence changing, and generate()
    # reserves what a call costs, so a game cannot spend more than LLM_BUDGET_S on
    # thinking however many actions it plays.
    # The binding budget is the SESSION's, not the per-game ceiling: the clock
    # divides what is left by the games still to come, so quote the allowance the
    # writer will actually get and say which of the two ceilings binds.
    allowance = min(LLM_BUDGET_S, PER_GAME_WALL_S * LLM_SHARE_OF_GAME)
    delib_s = (DELIBERATE_MAX_TURNS + 1) * per_call
    print("at this rate one deliberation (<=%d generations) costs ~%.0fs."
          % (DELIBERATE_MAX_TURNS + 1, delib_s))
    print("the per-game ceiling is %.0fs (%.1f h over %d games); the session is "
          "%.1f h, so the writer's allowance is ~%.0fs -> ~%.1f deliberations "
          "per game. Binding ceiling: %s."
          % (LLM_BUDGET_S, LLM_BUDGET_S * SESSION_GAMES / 3600.0, SESSION_GAMES,
             SESSION_BUDGET_S / 3600.0, allowance,
             allowance / max(1e-9, delib_s),
             "the session clock" if LLM_BUDGET_S > PER_GAME_WALL_S * LLM_SHARE_OF_GAME
             else "the per-game budget"))
    print("GATE: report tok/s before wheels, after wheels, and for FP8 vs bf16.")
    return 0


def main(argv: List[str]) -> int:
    if "--bench-llm" in argv:
        rounds = 2
        for i, a in enumerate(argv):
            if a == "--rounds" and i + 1 < len(argv):
                rounds = int(argv[i + 1])
        return bench_llm(rounds=rounds)
    if "--show-prompt" in argv:
        env = ToyEnv()
        tl = _toy_timeline()
        ctx = Context(game_id="demo", level=1, grid=env.grid(),
                      available=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                      action_summary=tl.action_effect_summary(0),
                      volatile_hint=tl.volatile_cell_hint(0),
                      reward_summary=tl.reward_summary(),
                      n_transitions=len(tl))
        print(SYSTEM_PROMPT)
        print("=" * 78)
        print(prompt_author(ctx))
        return 0
    return run_selftests()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
