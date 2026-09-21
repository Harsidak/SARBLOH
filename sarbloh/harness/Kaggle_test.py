"""Kaggle_test.py -- diagnostic evaluation harness for my_agent.py.

WHAT IT IS
    A single, self-contained file that plays real ARC-AGI-3 games with MyAgent
    and reports (a) the competition metric RHAE, and (b) the diagnostics that
    say WHERE the action budget went and WHY a level was not banked -- which is
    the information you actually need to improve the agent.

WHICH AGENT
    Whatever `/kaggle/working/my_agent.py` contains. TWO builds are supported and
    the harness adapts to the one it imports, by capability and never by name:

      * the symbolic build       -- planners (ExecPlanner/ClickPlanner/StateGraph),
                                    a single-prompt writer, JSON answers;
      * the LLM-as-coder build   -- one Deliberator, a two-part (system, user)
                                    writer, answers that are runnable Python.

    Everything that differs is isolated in four places, each commented where it
    sits: llm_dir() / llm_generate() (the writer's API), attribute_agent() (who
    spent the action), world_changed() (whose HUD mask), and _planner_internals()
    (what the agent believes). Attribute access on the agent is defensive by
    policy: this harness must never be the reason a paid Kaggle session dies, and
    an AttributeError twenty minutes in costs a session.

    It needs nothing from this repo except `my_agent` -- on Kaggle those two
    files are the whole harness. (On the dev box it will also pick up
    eval/official_score.py if it is there, purely so this file and eval/bench.py
    cannot drift; see "Scoring" below.) Human baselines for all 25 public games
    are EMBEDDED below, so a game uploaded without its metadata.json is still
    scored.

HOW TO RUN ON KAGGLE (the two-cell route this file is designed for)
    Cell A -- paste this whole file under a writefile header:

        %%writefile /kaggle/working/Kaggle_test.py
        <this entire file>

    Cell B -- run it from the terminal (bang-shell):

        !cd /kaggle/working && python Kaggle_test.py

    That is all. The script re-execs itself once with PYTHONHASHSEED=0 and -u so
    runs are reproducible and output streams live; it finds my_agent.py, the game
    files and the baselines on its own.

    START WITH THE PREFLIGHT -- it spends no actions and takes seconds:

        !cd /kaggle/working && python Kaggle_test.py --preflight

    It reports the agent file, the games, the GPU, and -- the thing that actually
    goes wrong -- whether the LLM checkpoint at LLM_MODEL_PATH exists, what
    model_type/quantization it declares, and whether the installed transformers
    can be expected to load it. A FAILing preflight aborts the run (--force
    overrides), because a run started then measures the wrong thing.

    Then the real run. Before the first game the LLM is loaded in the FOREGROUND
    and probed with one generate(), so a load failure is a loud traceback at
    second 30 rather than a silent "the LLM was never called" 20 minutes later:

        !cd /kaggle/working && python Kaggle_test.py --games all --max-actions 600
        !cd /kaggle/working && python Kaggle_test.py --games lp85 wa30 --trace full
        !cd /kaggle/working && python Kaggle_test.py --env ARC_NO_THEORIZE=1   # ablation
        !cd /kaggle/working && python Kaggle_test.py --no-llm-warmup           # core only
        !cd /kaggle/working && python Kaggle_test.py --fake-llm                # see below
        !cd /kaggle/working && python Kaggle_test.py --no-competition-mode     # raw reset rules

    --fake-llm marks the model as loaded and answers every prompt with a canned
    reply in the shape THIS agent parses (a JSON object, or a trivial Python
    program that passes every check). It tests NOTHING about the model -- it
    rehearses the plumbing (do the budget gates fire? are calls counted? does the
    reply survive the reader and reach an action?) on a machine that cannot hold
    27B, so a Kaggle session is not spent finding that out.

    Other routes: paste the file into a plain cell and it runs `main()`
    (CONFIG is then the single source of truth -- argv is ignored); or run it
    locally from `ARC_AGI EXPERMENTATIONS/`:

        PYTHONIOENCODING=utf-8 python -u Kaggle_test.py --games lp85

WHY IT MATTERS THAT THIS RUNS ON KAGGLE
    The 27B Qwen cannot be loaded on the dev box, so Kaggle is the ONLY place the
    LLM half of the agent can be exercised. This harness therefore instruments
    the LLM explicitly: did it load, how long did it take, how much VRAM, how many
    budgeted calls fired, how long each took, how many theorizer proposals
    survived the backtest gate, and whether any of it preceded a level-up. If the
    LLM is absent (local run) everything still works -- the no-LLM core is
    measured exactly as before and the LLM section reads "not loaded".

WHAT IT REPORTS PER GAME
    * levels banked / win_levels, terminal state, wall time, stall reason
    * RHAE per level (ai actions vs human baseline) and the game score
    * BUDGET LEDGER: every scored action attributed to the subsystem that chose
      it (bootstrap / exec / click-coverage / click-search / llm-click-plan /
      state-graph / mcts / novelty / reset), with each subsystem's no-op rate and
      how many level-ups it produced. This is the single most useful table: RHAE
      squares the action ratio, so whichever row is big and scores nothing is the
      thing to fix.
    * ACTION ECONOMY: no-op rate (world unchanged after a scored action),
      state-revisit / livelock rate, novel-state rate, GAME_OVERs, RESETs spent,
      per-action think time (p50/p95/max) and the implied 25-game session cost.
    * LLM section: load/warm state, calls by kind, latency, theorizer adopted vs
      rejected, click plans -- plus the RAW last few exchanges (prompt size, the
      model's actual answer) and "did it help": how many calls were followed by a
      level-up within 30 actions. A model that answers fluently but off-schema
      looks identical to a broken one in the aggregates; the raw text separates
      them. Agent log markers ([level]/[theorize]/[world-model]...) too.
    * FINDINGS: ranked, human-readable diagnoses derived from all of the above.

AND AT THE END
    NEXT STEPS -- the per-game findings rolled up across games and ranked by
    severity then breadth, each mapped to the concrete change it implies. That
    list is the point of running this.

OUTPUTS
    <OUT_DIR>/kaggle_test_report.json   full machine-readable report, rewritten
                                        after EVERY game so a killed session
                                        still leaves everything measured so far
    <OUT_DIR>/traces/trace_<gid>_*.jsonl  per-step trace (see CONFIG["TRACE"])

Scoring -- THE OFFICIAL NUMBER IS NOT COMPUTED BY THIS FILE
    The run is handed to the competition's own scorecard objects and we read back
    what they produce. Concretely, this is the same chain `arc_agi` runs when a
    scorecard is closed on the leaderboard:

        ScorecardManager.new_scorecard()             one card for the whole run
        LocalEnvironmentWrapper(..., scorecard_manager=mgr)
            -> EnvironmentWrapper._set_last_response()   on EVERY frame
            -> ScorecardManager.update_scorecard(guid, frame, frame.full_reset)
            -> Scorecard.new_play / reset / take_action / set_levels_completed
        EnvironmentScorecard.from_scorecard(card, [EnvironmentInfo, ...])
            -> EnvironmentScoreCalculator.add_level(...) -> .to_score()

    So plays, resets, per-level action splits, the 115 cap, the level-index
    weighting and the max-over-plays / mean-over-environments aggregation are all
    THEIR code. Human baselines come from each game's `metadata.json`
    (`baseline_actions`) -- the same file `Arcade._scan_for_environments` reads --
    and only fall back to the table embedded below if that file is absent.

    Two consequences worth knowing before you read the report:
      * a full_reset (two RESETs with no action between) does NOT append a
        phantom completed level: it starts a NEW PLAY, and a game's score is the
        MAX over plays. Reset churn therefore cannot inflate the real scorecard.
      * if a game produces MORE level-change events than it has baselines, the
        official scorer returns 0 for that game with the message "Human baseline
        actions size mismatch". The report prints that message.

    `honest` (capability: each level counted once, same weights, same cap) is
    still reported beside it, and even that only does the BOOKKEEPING -- the
    arithmetic is EnvironmentScoreCalculator's. The RHAE formula is NOT written
    down in this file, on purpose: every copy of it in this repo was eventually
    wrong, once by ~400x.
"""
import os
import sys
import glob
import json
import time
import random
import inspect
import logging
import threading
import traceback
import subprocess
from collections import Counter, defaultdict

import numpy as np

# --------------------------------------------------------------------------
# Scoring imports. THE OFFICIAL calculator, from the installed competition
# package. If this import fails nothing here can be scored -- and that is
# reported as such rather than papered over with a local formula.
try:
    from arc_agi.models import EnvironmentInfo as _EnvInfo
    from arc_agi.scorecard import (EnvironmentScorecard, EnvironmentScoreCalculator,
                                   ScorecardManager)
    HAVE_SCORECARD = True
except Exception as _e:                                     # pragma: no cover
    _EnvInfo = EnvironmentScorecard = EnvironmentScoreCalculator = None
    ScorecardManager = None
    HAVE_SCORECARD = False
    _SCORECARD_IMPORT_ERROR = _e

# The `honest` side (each level counted once) needs a little bookkeeping around
# that calculator. On the dev box we import it from eval/official_score.py so
# this harness and eval/bench.py cannot drift; on Kaggle -- where only this file
# and my_agent.py exist -- the equivalent is defined inline below, delegating
# every piece of arithmetic to EnvironmentScoreCalculator.
try:
    _HERE0 = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # pasted into a notebook cell: no __file__
    _HERE0 = os.getcwd()
_EVAL_DIR = os.path.join(_HERE0, "eval")
if os.path.isdir(_EVAL_DIR) and _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
try:
    from official_score import LevelEvents, score_run, level_score   # noqa: E402
    SCORER_SRC = "eval/official_score.py (shared with eval/bench.py)"
except Exception:
    LevelEvents = score_run = level_score = None    # bound at the end of the
    SCORER_SRC = "inline (arc_agi.scorecard)"       # inline-scorer block below

# ============================== CONFIG ======================================
# Everything is tunable here. When run as a script, CLI flags override these;
# when pasted into a notebook cell, this dict is the single source of truth.
CONFIG = {
    # Which games. A list of ids, a path to a game .py, or one of the presets:
    #   "quick5"   one game per type (default; ~20 min at 1500 actions)
    #   "all"      all 25 public games
    #   "click"    the 6-only games      "movement" the no-click games
    #   "mixed"    everything else
    "GAMES": "quick5",

    # RNG seed: seeds random / np.random and is exported as ARC_AGENT_SEED.
    "SEED": 0,

    # Per-game budget of REAL (scored) actions.
    "MAX_ACTIONS": 1500,

    # Play by the LEADERBOARD's reset rules (arc_agi/api.py, competition_mode):
    # a RESET sent when the engine's action count is 0 -- the one that would
    # full_reset() and wipe every banked level -- is REFUSED. It still costs an
    # action; the world simply does not change. Set False to measure the raw
    # local semantics instead, where that RESET really does wipe the score.
    "COMPETITION_MODE": True,

    # Per-game wall-clock cap in seconds (0 = none) and a cap for the whole run.
    # On a hit the game is scored with whatever it banked; the run continues.
    "TIME_CAP_SECS": 0,
    "TOTAL_TIME_BUDGET_SECS": 0,

    # Paths. "" = auto-discover (see the finders below).
    "AGENT_DIR": "",            # dir containing my_agent.py
    "GAMES_DIR": "",            # root of the game files
    "BASELINES_FILE": "",       # games_index.json; embedded table used if absent
    "OUT_DIR": "",              # report + traces; "" = /kaggle/working else cwd

    # Extra env vars set BEFORE my_agent is imported. Ablation switches live
    # here, e.g. {"ARC_NO_THEORIZE": "1"} or {"ARC_NO_GRAPH": "1"}.
    "ENV": {},

    # Progress line every N actions (0 = quiet).
    "PROGRESS_EVERY": 250,

    # Per-step trace: "none" | "steps" (compact, recommended) | "full" (adds
    # planner internals). Grids: "none" | "keyframes" (level-ups, deaths, first
    # and last frame) | "all" (big: ~4KB/step).
    "TRACE": "steps",
    "TRACE_GRIDS": "keyframes",

    # The agent prints a lot. QUIET_AGENT hides it live but still counts the
    # markers and keeps the last AGENT_LOG_TAIL lines for the per-game report.
    "QUIET_AGENT": True,
    "AGENT_LOG_TAIL": 12,

    # Load the LLM in the FOREGROUND before the first game and probe it with one
    # generate(). Costs a few minutes once, but turns "the LLM was never called"
    # into a precise answer. Set False to measure the no-LLM core alone.
    "WARM_LLM": True,

    # Stop after the preflight report / ignore a failing preflight.
    "PREFLIGHT_ONLY": False,
    "FORCE": False,

    # Rehearse the LLM half with canned answers and no model (dev box only).
    "FAKE_LLM": False,

    # Per-call LLM exchanges to print per game (0 = none). The raw answer is the
    # only way to see "it replied, but not in the schema the agent parses".
    "SHOW_LLM_CALLS": 3,

    # Re-exec once with PYTHONHASHSEED=0 / -u for reproducibility. Script only.
    "AUTO_REEXEC": True,

    # The fast attention/conv kernels (flash-linear-attention, causal-conv1d).
    # ABSENT is the difference between ~18 tok/s and a usable decode rate, which
    # is the difference between the writer being consulted 22 times in nine hours
    # and being consulted often enough to matter. Reporting is unconditional;
    # INSTALLING is opt-in, because a pip install inside a paid session can also
    # break a working torch. "" for WHEELS_DIR = search the known dataset paths.
    "INSTALL_KERNELS": False,
    "WHEELS_DIR": "",
}
# ============================================================================

# Human baselines + metadata for the 25 public games, embedded so this file is
# self-sufficient on Kaggle (the official environment_files ship the games but
# NOT games_index.json). id -> (win_levels, available_actions, baseline_steps).
# A real games_index.json, if found, overrides this.
EMBEDDED_INDEX = {
    'ar25': (8, [1, 2, 3, 4, 5, 6, 7], [17, 22, 103, 29, 29, 159, 152, 66]),
    'bp35': (9, [3, 4, 6, 7], [15, 72, 36, 31, 31, 48, 86, 155, 163]),
    'cd82': (6, [1, 2, 3, 4, 5, 6], [41, 8, 30, 21, 19, 17]),
    'cn04': (5, [1, 2, 3, 4, 5, 6], [16, 54, 69, 318, 208, 114]),
    'dc22': (6, [1, 2, 3, 4, 6], [64, 117, 59, 78, 324, 550]),
    'ft09': (6, [6], [17, 19, 15, 21, 65, 26]),
    'g50t': (7, [1, 2, 3, 4, 5], [51, 175, 86, 52, 96, 48, 67]),
    'ka59': (7, [1, 2, 3, 4, 6], [39, 175, 86, 47, 21, 132, 326]),
    'lf52': (10, [1, 2, 3, 4, 6, 7], [24, 81, 74, 86, 118, 148, 189, 116, 150, 225]),
    'lp85': (8, [6], [33, 22, 31, 23, 33, 34, 73, 173]),
    'ls20': (7, [1, 2, 3, 4], [21, 123, 39, 92, 54, 108, 109]),
    'm0r0': (6, [1, 2, 3, 4, 5, 6], [30, 209, 83, 86, 436, 126]),
    'r11l': (6, [6], [7, 28, 30, 20, 37, 45]),
    're86': (8, [1, 2, 3, 4, 5], [28, 38, 198, 57, 84, 117, 328, 221]),
    's5i5': (8, [6], [19, 57, 85, 203, 82, 30, 76, 56]),
    'sb26': (8, [5, 6, 7], [18, 16, 15, 15, 31, 24, 17, 17]),
    'sc25': (6, [1, 2, 3, 4, 6], [39, 5, 32, 33, 66, 41]),
    'sk48': (8, [1, 2, 3, 4, 6, 7], [15, 32, 35, 113, 304, 42, 63, 92]),
    'sp80': (6, [1, 2, 3, 4, 5, 6], [11, 18, 17, 172, 102, 152]),
    'su15': (9, [6, 7], [18, 28, 50, 151, 18, 31, 50, 179, 41]),
    'tn36': (7, [6], [23, 22, 26, 37, 25, 56, 61]),
    'tr87': (6, [1, 2, 3, 4], [37, 30, 39, 29, 63, 119]),
    'tu93': (9, [1, 2, 3, 4], [19, 15, 34, 42, 76, 91, 47, 23, 31]),
    'vc33': (7, [6], [6, 13, 31, 59, 92, 24, 82]),
    'wa30': (9, [1, 2, 3, 4, 5], [125, 58, 259, 113, 499, 58, 186, 134, 132]),
}

# One representative game per type: click-only, mixed, pure movement, and the
# two mechanical games the state-graph work was validated on.
QUICK5 = ["lp85", "cn04", "wa30", "tr87", "ls20"]

# Subsystem labels used by the budget ledger, in report order.
#
# TWO vocabularies, because two agents are measured with this harness and the
# ledger must name what actually spent the action in each:
#   * the symbolic build (my_agent.py) has no self-attribution, so the Probe
#     infers the producer from which patched planner returned something;
#   * the LLM-as-coder build (my_agent_llm.py) sets `agent._route` itself
#     (route_tag()), which is authoritative -- see attribute_src().
# Unknown tags still print, they just carry no help text.
SRC_ORDER = ["bootstrap", "exec", "click_search", "llm_click", "click_cov",
             "graph", "mcts", "novelty",
             "plan", "plan_optimism", "experiment_optimism", "experiment_discrim",
             "probe_bootstrap", "probe_evidence", "probe", "probe_idle",
             "reset", "init", "other"]
SRC_HELP = {
    # --- symbolic build ---
    "bootstrap":    "probing an unlearned move to learn its displacement",
    "exec":         "ExecPlanner (5.7) goal-seek / unscored plan execution",
    "click_search": "ClickPlanner env-as-simulator sequence search",
    "llm_click":    "LLM-reasoned click plan",
    "click_cov":    "ClickPlanner coverage / reactive floor",
    "graph":        "StateGraph BFS to the nearest untested (state, action)",
    "mcts":         "MCTS over the promoted world model (EXPLOIT)",
    "novelty":      "count-based novelty / epsilon-random fallback",
    # --- LLM-as-coder build (agent._route) ---
    "plan":                 "BFS through a CERTIFIED world model (the paying route)",
    "plan_optimism":        "plan through a model revised after 'no goal reachable'",
    "experiment_optimism":  "single action the LLM named to test its revision",
    "experiment_discrim":   "action on which two surviving models disagree",
    "probe_bootstrap":      "one press per button before the first model is authored",
    "probe_evidence":       "informative action bought when a round found nothing new",
    "probe":                "least-tried action: the floor when nothing certified",
    "probe_idle":           "FIDGET: every action's outcome from this frame was "
                            "already recorded and one was played anyway",
    # --- both ---
    "reset":        "RESET after GAME_OVER (scored, and it costs)",
    "init":         "agent set no route for this action (pre-first-deliberation)",
    "other":        "no instrumented producer claimed this action",
}

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # pasted into a notebook cell: no __file__
    HERE = os.getcwd()

_OUT = sys.stdout                       # real stdout; survives the agent-log capture


def say(*parts, **kw):
    """print() that always reaches the real console, even while agent stdout is
    redirected into the capture buffer."""
    msg = kw.get("sep", " ").join(str(p) for p in parts)
    try:
        _OUT.write(msg + kw.get("end", "\n"))
        _OUT.flush()
    except Exception:
        pass


def _in_notebook():
    return "ipykernel" in sys.modules


def _fmt_secs(s):
    s = int(round(s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _pct(x):
    return "n/a" if x is None else f"{100.0 * x:.0f}%"


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


# ===================== inline scorer (Kaggle fallback) ======================
# Used ONLY when eval/official_score.py is not importable -- i.e. on Kaggle,
# where this file ships alone. It is deliberately not a copy of the RHAE
# formula: every number below comes out of EnvironmentScoreCalculator, the class
# the competition ships. What IS duplicated here is bookkeeping -- which level
# events happened and how many actions each level cost -- and the two harnesses
# are cross-checked against each other whenever both are available.
class _LevelEvents:
    """Mirror of Card.set_levels_completed: record (levels_completed, actions)
    on every CHANGE, drops included."""

    def __init__(self):
        self.events = []
        self._cur = 0
        self.first_reach = {}      # level -> cumulative actions, monotonic
        self._peak = 0

    def observe(self, levels_completed, action_counter):
        lv = int(levels_completed or 0)
        if lv == self._cur:
            return False
        self.events.append((lv, int(action_counter)))
        self._cur = lv
        while lv > self._peak:
            self._peak += 1
            self.first_reach.setdefault(self._peak, int(action_counter))
        return True

    @property
    def wipes(self):
        return sum(1 for i, (lv, _) in enumerate(self.events)
                   if i > 0 and lv < self.events[i - 1][0])

    @property
    def real_levels(self):
        return self._peak

    def to_json(self):
        return {"events": self.events, "wipes": self.wipes,
                "real_levels": self.real_levels,
                "first_reach": dict(self.first_reach)}


def _new_calc():
    return EnvironmentScoreCalculator(id=None, resets=0, guid="local", state=None)


def _score_from_reach(reach, total_actions, baselines):
    """One add_level() per baseline, then to_score(). `reach` maps a level's
    0-based position to the cumulative action count at which it was credited;
    every arithmetic step after that is the calculator's."""
    calc = _new_calc()
    prev = 0
    for i, baseline in enumerate(baselines):
        at = reach.get(i)
        if at is not None:
            acts, done = at - prev, True
            prev = at
        else:
            acts, done = max(0, total_actions - prev), False
            prev = total_actions
        calc.add_level(level_index=i + 1, completed=done, actions_taken=acts,
                       baseline_actions=baseline)
    return calc.to_score()


def _score_run(ev, total_actions, baselines, win_levels=None):
    """Same keys as official_score.score_run(). `official` here is the
    RECONSTRUCTED leaderboard number (one play, drops appended); the authoritative
    one comes from the real Scorecard and overwrites it in main()."""
    baselines = list(baselines or [])
    if not baselines or not HAVE_SCORECARD:
        return {"official": 0.0, "honest": 0.0, "honest_rhae": 0.0,
                "real_levels": ev.real_levels, "credited_levels": 0,
                "wipes": ev.wipes, "inflation": 0.0, "levels_denominator": 0,
                "official_level_scores": [], "honest_level_actions": [],
                "honest_level_scores": [],
                "warn": ("no baselines: the competition scores this run 0"
                         if not baselines else "arc_agi.scorecard unavailable")}
    o = _score_from_reach({i: at for i, (_lv, at) in enumerate(ev.events)},
                          total_actions, baselines)
    h = _score_from_reach({k - 1: v for k, v in ev.first_reach.items()},
                          total_actions, baselines)
    warn = None
    if win_levels and win_levels != len(baselines):
        warn = (f"win_levels={win_levels} but {len(baselines)} baselines; "
                f"denominator follows the baselines")
    return {
        "official": o.score, "honest": h.score, "honest_rhae": h.score / 100.0,
        "real_levels": ev.real_levels, "credited_levels": o.levels_completed,
        "wipes": ev.wipes,
        "inflation": (o.score / h.score) if h.score > 0 else (
            float("inf") if o.score > 0 else 0.0),
        "levels_denominator": len(baselines),
        "official_level_scores": [round(x, 2) for x in o.level_scores],
        "honest_level_actions": h.level_actions,
        "honest_level_scores": [round(x, 2) for x in h.level_scores],
        "warn": warn,
    }


def _level_score(baseline_actions, actions_taken):
    """One level's score, 0..100, from the official calculator."""
    if not actions_taken or actions_taken <= 0 or not HAVE_SCORECARD:
        return 0.0
    calc = _new_calc()
    calc.add_level(level_index=1, completed=True, actions_taken=int(actions_taken),
                   baseline_actions=int(baseline_actions))
    return calc.to_score().level_scores[0]


if LevelEvents is None:                 # eval/official_score.py not importable
    LevelEvents, score_run, level_score = _LevelEvents, _score_run, _level_score


# ======================= startup: determinism / encoding ====================
def _prepare_process(cfg):
    """Make stdout UTF-8 (Kaggle logs and Windows consoles both need it) and,
    for the script route, re-exec ONCE with PYTHONHASHSEED=0 and -u.

    Why re-exec: MyAgent derives its RNG seed from ARC_AGENT_SEED + hash(game_id),
    and Python randomizes str hashes per process -- without a pinned hash seed the
    same command gives different runs. `!python Kaggle_test.py` cannot set env
    vars, so the script pins them for you. -u also stops Kaggle's redirected
    stdout from block-buffering the progress lines."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if _in_notebook() or not cfg.get("AUTO_REEXEC", True):
        return
    if os.environ.get("KTEST_REEXEC") == "1":
        return
    if os.environ.get("PYTHONHASHSEED") == "0" and not sys.flags.hash_randomization:
        return
    try:
        script = os.path.abspath(__file__)
    except NameError:
        return
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8",
               KTEST_REEXEC="1")
    say("[setup] re-running with PYTHONHASHSEED=0 -u (reproducibility)")
    # subprocess, not os.execve: on Windows exec* detaches from the parent's
    # pipes and the child's output is lost when stdout is redirected.
    try:
        import subprocess
        rc = subprocess.call([sys.executable, "-u", script] + sys.argv[1:], env=env)
    except Exception as e:                       # keep going un-pinned rather than die
        say(f"[setup] re-run failed ({e}); continuing without a pinned hash seed")
        return
    sys.exit(rc)


# =========================== discovery helpers ==============================
def find_agent_dir(cfg):
    """Make `from my_agent import MyAgent` resolve, wherever we are running."""
    cands = [cfg["AGENT_DIR"], os.environ.get("ARC_AGENT_DIR", ""), HERE,
             "/kaggle/working", os.getcwd()]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "my_agent.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return c
    raise FileNotFoundError(
        "my_agent.py not found. Run the `%%writefile /kaggle/working/my_agent.py` "
        "cell first, or set CONFIG['AGENT_DIR'].")


def find_games_root(cfg):
    """First existing directory that could hold game files. Supports both the
    eval/real_games layout (<root>/<id>/<hash>/<id>.py) and a flat <root>/<id>.py."""
    cands = [cfg["GAMES_DIR"], os.environ.get("ARC_GAMES_DIR", ""),
             os.path.join(HERE, "eval", "real_games"),
             os.path.join(os.getcwd(), "eval", "real_games"),
             "/kaggle/working/real_games", "/kaggle/working/eval/real_games"]
    for pat in ("/kaggle/input/competitions/*/environment_files",
                "/kaggle/input/*/environment_files",
                "/kaggle/input/*/real_games",
                "/kaggle/input/*/*/real_games",
                "/kaggle/input/*/*/*/real_games"):
        cands += sorted(glob.glob(pat))
    for c in cands:
        if c and os.path.isdir(c):
            return c
    raise FileNotFoundError(
        "No games root found. Set CONFIG['GAMES_DIR'] (or $ARC_GAMES_DIR), attach "
        "the competition dataset, or upload eval/real_games as a Kaggle dataset.")


def find_game_file(root, gid):
    for pat in (os.path.join(root, gid, "*", f"{gid}.py"),
                os.path.join(root, gid, f"{gid}.py"),
                os.path.join(root, f"{gid}.py"),
                os.path.join(root, "*", gid, "*", f"{gid}.py")):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def load_index(cfg, root):
    """id -> {win_levels, available_actions, baseline_steps}. A real
    games_index.json wins; otherwise the table embedded in this file is used."""
    cands = [cfg.get("BASELINES_FILE", ""),
             os.path.join(root, "games_index.json"),
             os.path.join(os.path.dirname(root), "games_index.json"),
             os.path.join(HERE, "eval", "real_games", "games_index.json"),
             "/kaggle/working/games_index.json"]
    if os.path.isdir("/kaggle/input"):
        for depth in ("*", "*/*", "*/*/*", "*/*/*/*"):
            cands += sorted(glob.glob(f"/kaggle/input/{depth}/games_index.json"))
    for idx in cands:
        if idx and os.path.exists(idx):
            try:
                with open(idx, "r", encoding="utf-8") as f:
                    data = json.load(f)
                out = {g["id"]: {"win_levels": int(g.get("win_levels", 0) or 0),
                                 "available_actions": list(g.get("available_actions", [])),
                                 "baseline_steps": list(g.get("baseline_steps", []))}
                       for g in data}
                return out, idx
            except Exception as e:
                say(f"[setup] ignoring unreadable {idx}: {e}")
    out = {gid: {"win_levels": w, "available_actions": aa, "baseline_steps": bs}
           for gid, (w, aa, bs) in EMBEDDED_INDEX.items()}
    return out, "embedded table (this file)"


def resolve_games(spec, index):
    """CONFIG['GAMES'] -> a concrete list of ids / file paths."""
    if isinstance(spec, str):
        key = spec.lower()
        if key == "quick5":
            return list(QUICK5)
        if key == "all":
            return sorted(index)
        if key == "click":
            return sorted(g for g, m in index.items()
                          if set(m["available_actions"]) == {6})
        if key == "movement":
            return sorted(g for g, m in index.items()
                          if 6 not in m["available_actions"])
        if key == "mixed":
            return sorted(g for g, m in index.items()
                          if 6 in m["available_actions"]
                          and set(m["available_actions"]) != {6})
        return [spec]
    return list(spec)


def class_name_from_file(path):
    import importlib.util
    from arcengine import ARCBaseGame
    spec = importlib.util.spec_from_file_location(
        "_probe_" + os.path.splitext(os.path.basename(path))[0], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, obj in vars(mod).items():
        if isinstance(obj, type) and issubclass(obj, ARCBaseGame) and obj is not ARCBaseGame:
            return name
    raise RuntimeError(f"No ARCBaseGame subclass in {path}")


# ====================== the OFFICIAL scorecard =============================
def make_env_info(game_file, gid, meta):
    """The competition's EnvironmentInfo for one game.

    `metadata.json` sits beside every shipped game and is what
    Arcade._scan_for_environments() loads; its `baseline_actions` is the list the
    official scorer divides by, and its game_id carries the version suffix that
    keys the scorecard. Prefer it. Fall back to the table embedded in this file
    only when the game was uploaded without its metadata."""
    d = os.path.dirname(os.path.abspath(game_file))
    mpath = os.path.join(d, "metadata.json")
    if os.path.isfile(mpath):
        try:
            with open(mpath, "r", encoding="utf-8") as f:
                info = _EnvInfo.model_validate_json(f.read())
            info.local_dir = d          # the file's own local_dir is stale
            if not info.baseline_actions:
                info.baseline_actions = list(meta.get("baseline_steps") or [])
            return info, "metadata.json"
        except Exception as e:
            say(f"    [scorecard] unreadable {mpath}: {e}")
    info = _EnvInfo(game_id=gid, class_name=class_name_from_file(game_file),
                    baseline_actions=list(meta.get("baseline_steps") or []))
    info.local_dir = d
    return info, "embedded table"


class OfficialScorecard:
    """The competition's scorecard, driven by the competition's own objects.

    ONE card for the whole run -- that is what a submission is -- handed to every
    LocalEnvironmentWrapper. From then on the wrapper calls
    ScorecardManager.update_scorecard() on every frame, exactly as the hosted API
    does, so plays, resets, action counts and level events are recorded by THEIR
    code. `compute()` closes the loop with EnvironmentScorecard.from_scorecard().

    Nothing in here decides anything about the score. That is the point.
    """

    API_KEY = "local"

    def __init__(self, competition_mode=True):
        self.ok = HAVE_SCORECARD
        self.infos = []
        self.mgr = None
        self.card_id = None
        self.competition_mode = bool(competition_mode)
        if not self.ok:
            return
        self.mgr = ScorecardManager()
        # competition_mode is the flag api.py reads before refusing a
        # score-wiping RESET; set it so the card matches the run we play.
        self.card_id = self.mgr.new_scorecard(source_url=None, tags=None,
                                              api_key=self.API_KEY, opaque=None,
                                              competition_mode=self.competition_mode)

    def register(self, info):
        if self.ok and not any(i.game_id == info.game_id for i in self.infos):
            self.infos.append(info)

    def raw(self):
        return self.mgr.get_scorecard(self.card_id, self.API_KEY) if self.ok else None

    def compute(self):
        """-> EnvironmentScorecard (the object close_scorecard() returns)."""
        sc = self.raw()
        if sc is None:
            return None
        try:
            return EnvironmentScorecard.from_scorecard(sc, self.infos)
        except Exception as e:
            say(f"  [scorecard] from_scorecard failed: {type(e).__name__}: {e}")
            return None

    def game(self, gid, card=None):
        """The official numbers for one game, as a plain dict (or None)."""
        card = card or self.compute()
        if card is None:
            return None
        env = card.find_environment(gid)
        if env is None:
            return None
        best = max(env.runs, key=lambda r: r.score) if env.runs else None
        return {
            "score": env.score,                 # max over plays -- the game's score
            "levels_completed": env.levels_completed,
            "actions": env.actions,
            "resets": env.resets,
            "plays": len(env.runs),
            "level_count": env.level_count,
            "level_scores": list(best.level_scores or []) if best else [],
            "level_actions": list(best.level_actions or []) if best else [],
            "baselines": list(best.level_baseline_actions or []) if best else [],
            "message": best.message if best else None,
            "per_play": [round(r.score, 4) for r in env.runs],
        }

    @staticmethod
    def _field(obj, name):
        """pydantic's computed_field usually materialises as a property, but a
        plain method survives on some versions -- accept either."""
        v = getattr(obj, name, None)
        return v() if callable(v) else v

    def overall(self, card=None):
        card = card or self.compute()
        if card is None:
            return None
        return {"score": card.score,            # mean over environments
                "environments": self._field(card, "total_environments"),
                "levels_completed": self._field(card, "total_levels_completed"),
                "levels": self._field(card, "total_levels"),
                "actions": self._field(card, "total_actions"),
                "per_game": {e.id: round(e.score, 4) for e in card.environments}}


# ========================= agent stdout capture =============================
class LogCapture:
    """Swallows (or mirrors) the agent's prints, counts its [tag] markers and
    keeps a tail. Background LLM threads print too, and sys.stdout is global, so
    this sees them as well -- which is exactly what we want for LLM triage."""

    def __init__(self, quiet, tail_n):
        self.quiet, self.tail_n = quiet, tail_n
        self.markers = Counter()
        self.lines = []
        self.notable = []               # every [LLM]/[theorize] line, verbatim
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, s):
        if not self.quiet:
            _OUT.write(s)
        with self._lock:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                self._ingest(line)
        return len(s)

    def _ingest(self, line):
        line = line.rstrip()
        if not line:
            return
        self.lines.append(line)
        if len(self.lines) > 4000:
            del self.lines[:2000]
        if line.startswith("[") and "]" in line[:24]:
            tag = line[1:line.index("]")]
            self.markers[tag] += 1
            if tag in ("LLM", "theorize", "click-plan", "synth", "world-model"):
                if len(self.notable) < 200:
                    self.notable.append(line)

    def flush(self):
        if not self.quiet:
            try:
                _OUT.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    def tail(self):
        return self.lines[-self.tail_n:] if self.tail_n else []


# ============================ instrumentation ===============================
class Probe:
    """Monkeypatches my_agent so every scored action can be attributed to the
    subsystem that produced it, and every LLM interaction is timed and counted.

    my_agent.py is NOT modified: each patch wraps the original and is skipped
    silently if the attribute does not exist, so the harness keeps working as
    the agent evolves."""

    def __init__(self):
        self.lock = threading.Lock()
        self._undo = []                  # (class, attr, original)
        self._undo_mod = []              # (module, attr, original) -- module functions
        self._ev = []                    # per-step (tag, produced_something)
        # Where the agent is RIGHT NOW, so a call made on a background thread
        # can be pinned to the action index / level it was reasoning about.
        self.ctx = {"game": "", "i": 0, "lvl": 0}
        self.llm = dict(self.FRESH_LLM, patched=False, load_attempted=False,
                        loaded=False, load_secs=None, load_error=None)
        self._kind = threading.local()
        self._n_producers = 0       # instrumented action producers actually patched
        self.src_from_agent = False  # set by install(): see attribute_agent()

    # Counters that reset per game (the model itself is a process singleton).
    FRESH_LLM = {
        "calls": 0, "empty": 0, "errors": 0, "gen_secs": [], "out_chars": 0,
        "prompt_chars": 0, "out_tokens": 0, "in_tokens": 0,
        "by_kind": Counter(), "kind_ok": Counter(),
        "theory_proposed": 0, "theory_adopted": 0, "theory_rejected": 0,
        "click_plans": 0, "click_plan_targets": 0, "log": [],
        # --- LLM-as-coder build: the gates that decide whether it can score ---
        # A reply is worthless unless code comes out of it, the code certifies,
        # and the certified model admits a plan. Each of those is a separate
        # number here so a zero says WHICH gate closed.
        "code_replies": 0, "no_code": 0,
        "backtests": 0, "models_certified": 0,
        "reject_by_check": Counter(),      # CHECK 0/1/2/3, REJECTED, TIMEOUT
        "plans": 0, "plans_nonempty": 0, "plan_actions": 0,
    }

    def reset_game_counters(self):
        with self.lock:
            for k, v in self.FRESH_LLM.items():
                self.llm[k] = Counter() if isinstance(v, Counter) else (
                    list(v) if isinstance(v, list) else v)

    def snapshot_llm(self):
        with self.lock:
            d = dict(self.llm)
        for k, v in list(d.items()):
            if isinstance(v, Counter):
                d[k] = dict(v)
        d["log"] = list(d["log"])
        return d

    # ---- event recording (main thread only) ----
    def step_begin(self):
        self._ev = []

    def ev(self, tag, produced):
        self._ev.append((tag, bool(produced) and produced is not None))

    def attribute_agent(self, agent, action_str, is_reset):
        """Who spent this action -- inferred, or self-reported?

        Use the agent's own `_route` ONLY when no producer could be instrumented,
        which is how the LLM-as-coder build presents itself (one Deliberator, no
        planner classes to patch). Its labels carry things no wrapper could infer:
        'this probe was bought to fill the deliberation backoff' is invisible from
        outside.

        NOT for the symbolic build, even though it also sets `_route`. Its
        vocabulary is a different, finer-grained and composed one ('react.bfs',
        'click.cover', 'esc.graph>mcts'); swapping it in would silently retire
        every SRC_HELP entry, every SRC_ORDER row and the explore-delegates-to-
        graph nuance below, i.e. it would change what the ledger MEANS without
        changing any agent. One vocabulary per build, decided at install time."""
        if self.src_from_agent:
            route = getattr(agent, "_route", None)
            if isinstance(route, str) and route:
                return route
        return self.attribute(action_str, is_reset)

    def attribute(self, action_str, is_reset):
        """Which subsystem produced this action? The LAST instrumented callee
        that returned something wins, because choose_action tries its layers in
        order and takes the first non-None -- except `_explore_action`, which
        internally delegates to the state graph, so a successful graph call
        inside it is credited to the graph.

        A RESET is only labelled 'reset' when NO subsystem claimed it: the
        ClickPlanner's env-as-simulator replay issues RESETs deliberately, and
        charging those to the death-recovery row would hide where the budget
        really goes."""
        graph_ok = any(t == "graph" and ok for t, ok in self._ev)
        for tag, ok in reversed(self._ev):
            if not ok:
                continue
            if tag == "explore":
                return "graph" if graph_ok else "novelty"
            return tag
        if is_reset:
            return "reset"
        return "bootstrap" if action_str else "other"

    # ---- installation ----
    def install(self, mod):
        # The symbolic build: attribution has to be INFERRED from which planner
        # returned something, because that agent does not label its own actions.
        self._wrap_src(getattr(mod, "ExecPlanner", None), "act", "exec")
        self._wrap_src(getattr(mod, "MCTSPlanner", None), "search", "mcts")
        self._wrap_src(getattr(mod, "MyAgent", None), "_graph_explore", "graph")
        self._wrap_src(getattr(mod, "MyAgent", None), "_explore_action", "explore")
        self._wrap_src(getattr(mod, "MyAgent", None), "_next_click_from_plan", "llm_click")
        self._wrap_click(getattr(mod, "ClickPlanner", None))
        self._wrap_theory(getattr(mod, "ExecPlanner", None))
        # Either build: the writer is instrumented by whichever name it is
        # exported under (my_agent.py: LocalLLM; my_agent_llm.py: LLMCoder,
        # aliased to LocalLLM). Both are tried so the alias is not load-bearing.
        self._wrap_llm(getattr(mod, "LocalLLM", None) or getattr(mod, "LLMCoder", None))
        # The LLM-as-coder build: it labels its own actions (`_route`), so the
        # only things left to instrument are the inner-loop gates.
        self._wrap_delib(getattr(mod, "Deliberator", None))
        self._wrap_prompts(mod)
        # Nothing to infer from means the agent has to speak for itself.
        self.src_from_agent = (self._n_producers == 0)

    def attribution_note(self):
        """One line for the run header: which vocabulary the ledger will use."""
        if self.src_from_agent:
            return "agent._route (self-reported by the agent)"
        return f"inferred from {self._n_producers} instrumented producers"

    def uninstall(self):
        for cls, name, orig in reversed(self._undo):
            try:
                setattr(cls, name, orig)
            except Exception:
                pass
        for holder, name, orig in reversed(self._undo_mod):
            try:
                setattr(holder, name, orig)
            except Exception:
                pass
        self._undo = []
        self._undo_mod = []

    def _patch(self, cls, name, factory):
        """Returns whether the patch was actually applied -- install() counts the
        producers, and 'zero producers' is what tells the two builds apart."""
        if cls is None:
            return False
        orig = getattr(cls, name, None)
        if orig is None or getattr(orig, "_ktest", False):
            return False
        new = factory(orig)
        new._ktest = True
        # Wrappers are written `def w(self, *a, **k)`, which ERASES the signature
        # of the method they replace. llm_generate() dispatches on that signature:
        # with it erased the coder build's generate(system, user) was called with a
        # single argument, raised TypeError, returned '', and the preflight then
        # reported a FALSE "LLM reader FAILED" -- while the real run parsed 20 of
        # 22 replies perfectly. Instrumentation must be invisible.
        try:
            new.__signature__ = inspect.signature(orig)
        except (TypeError, ValueError):
            pass
        try:
            setattr(cls, name, new)
            self._undo.append((cls, name, orig))
            return True
        except Exception:
            return False

    def _wrap_src(self, cls, name, tag):
        probe = self

        def factory(orig):
            def w(self, *a, **k):
                r = orig(self, *a, **k)
                probe.ev(tag, r)
                return r
            return w
        if self._patch(cls, name, factory):
            self._n_producers += 1

    def _wrap_click(self, cls):
        """ClickPlanner.act serves two very different roles; separate them,
        because 'search' is a deliberate plan and 'coverage' is a blind sweep."""
        probe = self

        def factory(orig):
            def w(self, *a, **k):
                searching = False
                try:
                    searching = bool(getattr(self, "searching", False))
                except Exception:
                    pass
                r = orig(self, *a, **k)
                probe.ev("click_search" if searching else "click_cov", r)
                return r
            return w
        if self._patch(cls, "act", factory):
            self._n_producers += 1

    def _wrap_theory(self, cls):
        """Phase-2 theorizer value: proposals in vs proposals that survived the
        full-timeline backtest."""
        probe = self

        def factory(orig):
            def w(self, *a, **k):
                r = orig(self, *a, **k)
                with probe.lock:
                    probe.llm["theory_proposed"] += 1
                    if r:
                        probe.llm["theory_adopted"] += 1
                    else:
                        probe.llm["theory_rejected"] += 1
                return r
            return w
        self._patch(cls, "propose_theory", factory)

    def _wrap_delib(self, cls):
        """The LLM-as-coder inner loop, gate by gate.

        `write_code` -> did a reply contain code at all;
        `run_backtest` -> did that code survive CHECK 0/1/2/3 (and if not, which
        check killed it -- a Report, not a bare bool, is the whole reason this is
        measurable);
        `run_bfs` -> did the certified model admit a plan.
        A zero anywhere here localises the failure to one gate, which is what a
        Kaggle session is for."""
        if cls is None:
            return
        probe = self

        def code_factory(orig):
            def w(self, *a, **k):
                r = orig(self, *a, **k)
                code = r[0] if isinstance(r, tuple) else r
                with probe.lock:
                    if code:
                        probe.llm["code_replies"] += 1
                    else:
                        probe.llm["no_code"] += 1
                return r
            return w
        self._patch(cls, "write_code", code_factory)

        def backtest_factory(orig):
            def w(self, *a, **k):
                r = orig(self, *a, **k)
                model, rep = r if isinstance(r, tuple) else (r, None)
                with probe.lock:
                    probe.llm["backtests"] += 1
                    if model is not None:
                        probe.llm["models_certified"] += 1
                    else:
                        probe.llm["reject_by_check"][
                            str(getattr(rep, "check", "?")) or "?"] += 1
                return r
            return w
        self._patch(cls, "run_backtest", backtest_factory)

        def bfs_factory(orig):
            def w(self, *a, **k):
                r = orig(self, *a, **k)
                acts = list(getattr(r, "actions", []) or [])
                with probe.lock:
                    probe.llm["plans"] += 1
                    if acts:
                        probe.llm["plans_nonempty"] += 1
                        probe.llm["plan_actions"] += len(acts)
                return r
            return w
        self._patch(cls, "run_bfs", bfs_factory)

    def _wrap_prompts(self, mod):
        """Tag WHICH prompt the next generation is answering.

        The LLM-as-coder build has ONE call site (`write_code`), so `by_kind`
        would otherwise say `generate` for everything and hide the thing worth
        knowing: whether the writer is still authoring from scratch or already
        iterating on repairs. The builders run immediately before the call, so
        tagging at build time attributes correctly without touching the agent."""
        for name, kind in (("prompt_author", "author"),
                           ("prompt_repair", "repair"),
                           ("prompt_optimism", "optimism"),
                           ("prompt_levelup", "levelup")):
            orig = getattr(mod, name, None)
            if orig is None or getattr(orig, "_ktest", False):
                continue
            probe = self

            def make(orig=orig, kind=kind):
                def w(*a, **k):
                    probe._kind.kind = kind
                    return orig(*a, **k)
                w._ktest = True
                return w
            try:
                setattr(mod, name, make())
                self._undo_mod.append((mod, name, orig))
            except Exception:
                pass

    def _wrap_llm(self, cls):
        if cls is None:
            return
        probe = self
        self.llm["patched"] = True

        def load_factory(orig):
            def w(self, *a, **k):
                if getattr(self, "model", None) is not None:
                    # Already up (warm-up, an earlier game, or --fake-llm): the
                    # timing wrapper below would report a 0s "load" and, worse,
                    # leave 'loaded' False for the whole run.
                    with probe.lock:
                        probe.llm["loaded"] = True
                    return orig(self, *a, **k)
                t0 = time.time()
                with probe.lock:
                    probe.llm["load_attempted"] = True
                ok = orig(self, *a, **k)
                dt = time.time() - t0
                with probe.lock:
                    if probe.llm["load_secs"] is None or ok:
                        probe.llm["load_secs"] = round(dt, 1)
                    probe.llm["loaded"] = bool(ok) or probe.llm["loaded"]
                return ok
            return w
        self._patch(cls, "ensure_loaded", load_factory)

        def gen_factory(orig):
            def w(self, prompt, *a, **k):
                t0 = time.time()
                kind = getattr(probe._kind, "kind", "generate")
                ctx = dict(probe.ctx)
                err = None
                # The two builds differ in ARITY: generate(prompt, ...) vs
                # generate(system, user, ...). Sum every string argument instead
                # of reading the first one, or the whole system prompt (~10k
                # chars a call) silently vanishes from the accounting.
                n_prompt = sum(len(x) for x in ((prompt,) + a) if isinstance(x, str))
                n_prompt += sum(len(v) for v in k.values() if isinstance(v, str))
                try:
                    out = orig(self, prompt, *a, **k)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    out = ""
                dt = time.time() - t0
                with probe.lock:
                    if err:
                        probe.llm["errors"] += 1
                    probe.llm["calls"] += 1
                    probe.llm["gen_secs"].append(round(dt, 2))
                    probe.llm["prompt_chars"] += n_prompt
                    probe.llm["out_chars"] += len(out or "")
                    # Tokens, not chars, are what tok/s is measured in -- and
                    # tok/s is the number that decides whether 25 games fit in a
                    # 9h session. The writer exposes the count per call.
                    probe.llm["out_tokens"] += int(getattr(self, "last_out_tokens", 0) or 0)
                    probe.llm["in_tokens"] += int(getattr(self, "last_in_tokens", 0) or 0)
                    probe.llm["by_kind"][kind] += 1
                    if out:
                        probe.llm["kind_ok"][kind] += 1
                    else:
                        probe.llm["empty"] += 1
                    # Keep the raw exchange: on Kaggle this is the ONLY place we
                    # ever see what the 27B actually answered, and a plausible-
                    # looking but unparseable answer is the failure mode that
                    # otherwise shows up as a silent zero.
                    if len(probe.llm["log"]) < 60:
                        probe.llm["log"].append({
                            "at": ctx["i"], "lvl": ctx["lvl"], "kind": kind,
                            "secs": round(dt, 2), "prompt_chars": n_prompt,
                            "out_chars": len(out or ""), "error": err,
                            "out": (out or "")[:400],
                        })
                if err:
                    # my_agent's generate() swallows its own exceptions; if one
                    # escapes it is a harness-visible bug, so surface it and
                    # keep the game alive rather than killing the run.
                    say(f"    [ktest] LLM {kind} raised {err}")
                return out
            return w
        self._patch(cls, "generate", gen_factory)

        # Tag each budgeted call site so by_kind says WHICH reasoning fired.
        for meth, kind in (("synthesize_world_model", "world_model"),
                           ("reflect", "reflect"),
                           ("plan_clicks", "click_plan"),
                           ("theorize", "theorize")):
            def kind_factory(orig, kind=kind):
                def w(self, *a, **k):
                    prev = getattr(probe._kind, "kind", None)
                    probe._kind.kind = kind
                    try:
                        r = orig(self, *a, **k)
                    finally:
                        probe._kind.kind = prev
                    if kind == "click_plan":
                        with probe.lock:
                            probe.llm["click_plans"] += 1
                            probe.llm["click_plan_targets"] += len(r or [])
                    return r
                return w
            self._patch(cls, meth, kind_factory)


PROBE = Probe()


# ============================== preflight ===================================
def preflight(MA, agent_dir, games_root, index_src, games, cfg):
    """Everything that can be checked WITHOUT spending an action, printed as
    PASS/WARN/FAIL. Run this first: a wrong dataset path or a missing wheel is a
    30-second discovery here and a wasted 20-minute run otherwise."""
    rep, worst = [], 0

    def add(level, name, detail):
        nonlocal worst
        worst = max(worst, {"PASS": 0, "WARN": 1, "FAIL": 2}[level])
        rep.append({"level": level, "check": name, "detail": detail})

    ap = os.path.join(agent_dir, "my_agent.py")
    add("PASS", "agent", f"{ap}  ({os.path.getsize(ap) // 1024}KB, modified "
                         f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(ap)))})")

    missing = [g for g in games if not os.path.isfile(str(g))
               and find_game_file(games_root, str(g)) is None]
    add("FAIL" if len(missing) == len(games) else ("WARN" if missing else "PASS"),
        "games", f"{len(games) - len(missing)}/{len(games)} found under {games_root}"
                 + (f"; MISSING {' '.join(missing)}" if missing else ""))
    add("PASS" if "embedded" not in index_src else "WARN", "baselines", index_src)

    # The scorer itself. Without it there is no official number at all, and a run
    # that produces only our reconstruction is exactly what this file exists to
    # stop happening.
    add("PASS" if HAVE_SCORECARD else "FAIL", "scorer",
        "arc_agi.scorecard (EnvironmentScorecard + EnvironmentScoreCalculator); "
        f"honest side via {SCORER_SRC}" if HAVE_SCORECARD else
        f"arc_agi.scorecard did NOT import ({_SCORECARD_IMPORT_ERROR}); nothing "
        f"can be scored officially")
    with_meta = 0
    for g in games:
        gfp = g if os.path.isfile(str(g)) else find_game_file(games_root, str(g))
        if gfp and os.path.isfile(os.path.join(
                os.path.dirname(os.path.abspath(gfp)), "metadata.json")):
            with_meta += 1
    add("PASS" if with_meta == len(games) else "WARN", "baseline-src",
        f"{with_meta}/{len(games)} games ship metadata.json, whose baseline_actions "
        f"is what the official scorer divides by"
        + ("" if with_meta == len(games) else
           "; the rest fall back to the table embedded in this file"))

    # --- torch / GPU ---
    has_cuda = False
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
        if has_cuda:
            p = torch.cuda.get_device_properties(0)
            free = torch.cuda.mem_get_info()[0] / 1e9
            add("PASS" if free > 60 else "WARN", "gpu",
                f"{p.name}, {p.total_memory / 1e9:.0f}GB total, {free:.0f}GB free "
                f"(torch {torch.__version__})")
        else:
            add("WARN", "gpu", f"no CUDA device (torch {torch.__version__}); the "
                               f"LLM half cannot run, the no-LLM core still can")
    except Exception as e:
        add("WARN", "gpu", f"torch not importable: {e}")

    # --- the LLM half of the pipeline ---
    # A missing model is only FATAL where the LLM could actually have run: on a
    # CPU dev box the no-LLM core is exactly what we mean to measure.
    lvl_no_llm = "FAIL" if has_cuda else "WARN"
    path = getattr(MA, "LLM_MODEL_PATH", "")
    try:
        import transformers
        tv = transformers.__version__
    except Exception as e:
        tv = None
        add(lvl_no_llm, "transformers", f"not importable: {e}")
    if not path or not os.path.isdir(path):
        add(lvl_no_llm, "llm-path", f"{path!r} is not a directory -- attach the model "
                                    f"dataset or set $ARC_LLM_PATH"
                                    + ("" if has_cuda else "  (no GPU here anyway)"))
    else:
        cfgf = os.path.join(path, "config.json")
        if not os.path.isfile(cfgf):
            add(lvl_no_llm, "llm-path",
                f"no config.json in {path} (my_agent resolves the model dir at "
                f"import; it found nothing)")
        else:
            mt, arch = "?", "?"
            try:
                with open(cfgf, "r", encoding="utf-8") as f:
                    j = json.load(f)
                mt = j.get("model_type", "?")
                arch = ",".join(j.get("architectures") or []) or "?"
                quant = (j.get("quantization_config") or {}).get("quant_method")
            except Exception:
                quant = None
            wt = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)
                     if f.endswith((".safetensors", ".bin"))) / 1e9
            add("PASS", "llm-path", f"{path}  model_type={mt} arch={arch} "
                                    f"weights={wt:.0f}GB"
                                    + (f" quant={quant}" if quant else " (unquantized)"))
            if quant and tv:
                add("WARN", "llm-quant", f"checkpoint is {quant}-quantized; plain "
                                         f"transformers may refuse it -- if the load "
                                         f"fails below, that is why")
            if tv:
                add("PASS", "transformers", f"{tv}")
    ep = getattr(MA, "EMBED_MODEL_PATH", "")
    if ep:
        add("PASS" if os.path.isdir(ep) else "WARN", "embed-path",
            ep + ("" if os.path.isdir(ep) else "  (absent; the agent must degrade "
                                               "to its non-embedding path)"))

    # --- the fast decode kernels ---
    # Cheap to check and expensive to get wrong: with these absent the writer
    # decodes at ~18 tok/s, one answer costs ~90s, and the LLM half of a nine-hour
    # session is 22 calls. Checked here so `--preflight` answers it in seconds,
    # without loading 27B of weights. Report only -- warm_llm() does the install.
    if callable(getattr(MA, "get_local_llm", None)):
        have, missing = _kernel_state()
        if not missing:
            add("PASS", "kernels", f"{', '.join(have)} importable (fast decode)")
        else:
            found = find_wheels(cfg)
            add("WARN", "kernels",
                f"{', '.join(missing)} NOT importable -- decode falls back to "
                f"torch (~18 tok/s measured). "
                + (f"wheels ARE present in {found[0][0]}; re-run with "
                   f"--install-kernels" if found else
                   "no offline wheel found either; add one to the dataset or "
                   "pip install online"))

    # --- the entry points the framework itself uses ---
    # A submission that fails HERE fails on the leaderboard for reasons no game
    # result would explain, so it is checked before anything is spent.
    ag_cls = getattr(MA, "MyAgent", None)
    if ag_cls is None:
        add("FAIL", "agent-api", "the module exports no MyAgent class")
    else:
        missing = [m for m in ("choose_action", "is_done")
                   if not callable(getattr(ag_cls, m, None))]
        bases = [b.__name__ for b in getattr(ag_cls, "__mro__", [])[1:]]
        add("FAIL" if missing else "PASS", "agent-api",
            (f"MyAgent is missing {missing}" if missing else
             f"MyAgent(choose_action, is_done) over {bases[0] if bases else 'object'}"
             f"; MAX_ACTIONS={getattr(ag_cls, 'MAX_ACTIONS', '?')}"))

    # --- the exec sandbox, on THIS kernel ---
    # The LLM-as-coder build runs generated code behind an AST screen, an import
    # whitelist and a curated builtins dict. That is exactly the kind of thing
    # that passes on the dev box and fails on a different Python, and it would
    # then look like 'the model never writes anything usable'. Certify a known-
    # good trivial program here so the two failures cannot be confused.
    if hasattr(MA, "CandidateModel") and hasattr(MA, "Sandbox"):
        try:
            m = MA.CandidateModel(FAKE_PROGRAM_CODE, MA.Sandbox())
            probe_grid = np.array([[1, 2], [3, 0]], dtype=np.int64)
            st = m.parse(probe_grid)
            back = np.asarray(m.render(st))
            m.step(st, "ACTION1")
            ok = np.array_equal(back, probe_grid)
            add("PASS" if ok else "FAIL", "sandbox",
                "generated code compiles and round-trips a grid"
                if ok else "render() did not reproduce the probe grid")
        except Exception as e:
            add("FAIL", "sandbox", f"a known-good program was REJECTED: "
                                   f"{type(e).__name__}: {e}")

    say("")
    say("  PREFLIGHT")
    for r in rep:
        mark = {"PASS": " ok ", "WARN": "warn", "FAIL": "FAIL"}[r["level"]]
        say(f"    [{mark}] {r['check']:<14} {r['detail']}")
    return rep, worst


def llm_dir(llm):
    """Where the writer thinks its weights are.

    The two builds spell this differently (`model_path` vs `model_dir`, the
    latter already RESOLVED to the directory holding config.json), and reading
    the wrong one is an AttributeError in the middle of a paid Kaggle session,
    not a wrong string. So never touch the attribute directly."""
    for attr in ("model_dir", "model_path", "requested_path"):
        v = getattr(llm, attr, None)
        if v:
            return str(v)
    return "?"


def llm_generate(llm, system, user, max_new_tokens=48):
    """Call whichever `generate` the loaded agent exports.

        my_agent.py       generate(prompt, max_new_tokens=, temperature=)
        my_agent_llm.py   generate(system, user, max_new_tokens=, max_time=)

    Passing the wrong shape does not raise cleanly -- with the two-argument
    writer, `max_new_tokens=32` positionally lands in `user`, so the model is
    asked to answer the integer 32. The signature decides, and unknown kwargs
    are dropped rather than guessed at."""
    import inspect
    try:
        params = inspect.signature(llm.generate).parameters
    except (TypeError, ValueError):
        params = {}
    kw = {}
    if "max_new_tokens" in params:
        kw["max_new_tokens"] = max_new_tokens
    if "temperature" in params:
        kw["temperature"] = 0.0
    # Two text arguments, or one? Decided by the NAME of the second positional
    # parameter -- `len(params) >= 2` is true of the single-prompt writer too
    # (its second parameter is max_new_tokens), which is exactly the mistake
    # this helper exists to prevent.
    names = [n for n, p in params.items()
             if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    two_part = len(names) >= 2 and names[1] not in (
        "max_new_tokens", "temperature", "max_time", "max_tokens")
    if two_part:
        return llm.generate(system, user, **kw)
    try:
        return llm.generate((system + "\n\n" + user).strip(), **kw)
    except TypeError as e:
        # The signature lied -- an instrumentation wrapper erased it, or a build
        # changed shape. One retry in the other shape beats reporting a broken
        # reader on the strength of a call that never reached the model.
        if "positional argument" not in str(e):
            raise
        say(f"  LLM          signature said one prompt but generate() wants two "
            f"({e}); retrying as generate(system, user)")
        return llm.generate(system, user, **kw)


def llm_parse_probe(MA, text):
    """Can the loaded agent READ its own writer's answer?

    This is the whole point of the probe: the reader, not the model, is where
    answers were being lost. Each build has a different reader (JSON object vs a
    fenced Python program), so ask the agent for its own."""
    for name in ("extract_candidate", "extract_code"):
        fn = getattr(MA, name, None)
        if callable(fn):
            try:
                got = fn(text)
            except Exception:
                got = None
            return (bool(got), "python code via %s()" % name)
    for holder in (getattr(MA, "LocalLLM", None), getattr(MA, "LLMCoder", None)):
        fn = getattr(holder, "_extract_json", None)
        if callable(fn):
            try:
                return bool(fn(text)), "JSON via LocalLLM._extract_json()"
            except Exception:
                return False, "JSON via LocalLLM._extract_json()"
    return None, "no reader found on the agent module"


# ---------------------------------------------------------------------------
# Fast kernels. (import name, pip name, what its absence costs)
#
# `fla` is imported by the model's own modelling code; when the import fails
# transformers silently falls back to a pure-torch path, so the ONLY symptom is
# the decode rate. A measured 18.1 tok/s meant a 1600-token answer cost ~90s and
# 22 calls consumed 100% of the wall clock -- the writer was not slow because the
# model is big, it was slow because these two wheels were not installed.
FAST_KERNELS = (
    ("fla", "flash-linear-attention", "linear-attention decode"),
    ("causal_conv1d", "causal-conv1d", "the short conv in each block"),
)

# Where offline wheels live on Kaggle. First match wins; $ARC_WHEELS overrides.
WHEEL_DIRS = (
    "/kaggle/input/notebooks/banwait13/datasets-for-arc-agi/wheels",
    "/kaggle/input/datasets-for-arc-agi/wheels",
    "/kaggle/input/arc-agi-wheels",
    "/kaggle/input/notebooks/banwait13/datasets-for-nemotron-banwait/wheels",
)


def _kernel_state():
    """(importable-set, one-line summary) for the fast kernels, right now."""
    have, missing = [], []
    for mod, pip, _why in FAST_KERNELS:
        try:
            __import__(mod)
            have.append(mod)
        except Exception:
            missing.append(pip)
    return have, missing


def kernel_tag():
    """A short key naming which kernels are live, e.g. 'fla+causal_conv1d' or
    'none'. tok/s is only comparable within one tag, so every measurement is
    filed under it -- that is what makes a before/after pair a pair."""
    have, _ = _kernel_state()
    return "+".join(have) if have else "none"


def find_wheels(cfg=None):
    """Directories that actually contain a wheel for a missing kernel.

    Returns [(dir, [wheel filenames])]. A directory that exists but holds no
    matching wheel is not reported: 'the dataset is mounted' and 'the wheel is
    in it' are different facts and only the second one installs."""
    cands = []
    for d in ((cfg or {}).get("WHEELS_DIR"), os.environ.get("ARC_WHEELS"),
              os.path.join(HERE, "wheels")) + WHEEL_DIRS:
        if d and d not in cands:
            cands.append(str(d))
    stems = tuple(pip.replace("-", "_") for _m, pip, _w in FAST_KERNELS)
    out = []
    for d in cands:
        if not os.path.isdir(d):
            continue
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        hits = [n for n in names
                if n.endswith(".whl") and n.lower().startswith(stems)]
        if hits:
            out.append((d, hits))
    return out


def fast_kernels(cfg, install=None):
    """Report -- and, only when explicitly asked, install -- the fast kernels.

    MUST run before the model is loaded: `LLMCoder.load()` probes for these
    imports and picks its attention path there, so a wheel installed afterwards
    changes the log line and nothing else. Never installs by default; a pip
    install mid-session can replace a working torch, and losing the session to
    that is worse than one slow run.

    Returns a dict for the report, including the `tag` every tok/s number is
    filed under."""
    if install is None:
        install = bool(cfg.get("INSTALL_KERNELS")) or \
            os.environ.get("ARC_INSTALL_KERNELS", "") == "1"
    have, missing = _kernel_state()
    info = {"have": list(have), "missing": list(missing), "installed": False}
    if not missing:
        say(f"  kernels      PRESENT: {', '.join(have)} -- the fast decode path "
            f"is available")
        info["tag"] = kernel_tag()
        return info

    found = find_wheels(cfg)
    why = ", ".join(f"{pip} ({w})" for _m, pip, w in FAST_KERNELS
                    if pip in missing)
    say(f"  kernels      MISSING: {why}")
    if found:
        d, hits = found[0]
        cmd = (f"pip install -q --no-index --find-links={d} "
               f"{' '.join(missing)}")
        say(f"  kernels      wheels found in {d}: {', '.join(hits[:4])}"
            + (f" (+{len(hits) - 4} more)" if len(hits) > 4 else ""))
        say(f"  kernels      install with:  !{cmd}")
        info["wheel_dir"] = d
        info["wheels"] = hits
        info["command"] = cmd
    else:
        cmd = f"pip install -q {' '.join(missing)}"
        say(f"  kernels      no offline wheel found (searched "
            f"{len(WHEEL_DIRS) + 2} paths). Online: !{cmd}")
        info["command"] = cmd
    if not install:
        say(f"  kernels      NOT installing (opt in with --install-kernels or "
            f"ARC_INSTALL_KERNELS=1). Decode runs on the torch fallback; the "
            f"tok/s below is the '{kernel_tag()}' arm of the comparison.")
        info["tag"] = kernel_tag()
        return info

    args = [sys.executable, "-m", "pip", "install", "-q"]
    if found:
        args += ["--no-index", "--find-links", found[0][0]]
    args += list(missing)
    say(f"  kernels      installing: {' '.join(args[3:])} ...")
    t0 = time.time()
    try:
        proc = subprocess.run(args, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=900)
        tail = (proc.stdout or b"").decode("utf-8", "replace").strip()
        info["rc"] = proc.returncode
        if tail:
            for line in tail.splitlines()[-6:]:
                say(f"  kernels      | {line}")
    except Exception as e:
        say(f"  kernels      install raised {type(e).__name__}: {e}")
        info["error"] = f"{type(e).__name__}: {e}"
        info["tag"] = kernel_tag()
        return info
    # The verdict is the IMPORT, not pip's exit code: a wheel built for another
    # torch/CUDA installs cleanly and then fails at `import fla`.
    have2, missing2 = _kernel_state()
    info.update(installed=True, have=list(have2), missing=list(missing2),
                install_secs=round(time.time() - t0, 1))
    if missing2:
        say(f"  kernels      still MISSING after install: {', '.join(missing2)} "
            f"-- installed but not importable (wrong torch/CUDA build?)")
    else:
        say(f"  kernels      now PRESENT: {', '.join(have2)} in "
            f"{_fmt_secs(time.time() - t0)}")
    info["tag"] = kernel_tag()
    return info


# Decode rate MEASURED with both kernels absent (lp85, 2026-08-14: 22 calls,
# 18.1 tok/s, ~90s per answer, 100% of the wall clock). Not a target and not a
# sample -- a recorded prior, so a first run with the kernels still reports a pair.
RECORDED_FALLBACK_TPS = 18.1


def _tps_ledger_path(cfg):
    out = cfg.get("OUT_DIR") or ("/kaggle/working"
                                 if os.path.isdir("/kaggle/working")
                                 else os.getcwd())
    return os.path.join(out, "tok_per_s.json")


def record_tps(cfg, tag, tps, n_out, mnt=0):
    """File this run's decode rate under its kernel tag and print every tag seen
    so far.

    One run can only measure ONE arm -- the kernels are chosen when the model
    loads -- so the before/after pair is assembled ACROSS runs. Making the ledger
    a file rather than a promise is the difference between reporting both numbers
    and remembering one of them."""
    path = _tps_ledger_path(cfg)
    led = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            led = json.load(fh) or {}
    except Exception:
        led = {}
    if not isinstance(led, dict):
        led = {}
    row = led.setdefault(str(tag), {"tok_per_s": [], "n": 0})
    row["tok_per_s"] = (row.get("tok_per_s") or [])[-9:] + [round(float(tps), 2)]
    row["n"] = len(row["tok_per_s"])
    row["last_tokens"] = int(n_out)
    row["when"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(led, fh, indent=1, sort_keys=True)
    except Exception as e:
        say(f"  LLM          (could not write {path}: {e})")

    def _best(r):
        xs = r.get("tok_per_s") or []
        return max(xs) if xs else 0.0
    say(f"  LLM          decode ledger ({os.path.basename(path)}):")
    for k in sorted(led, key=lambda k: -_best(led[k])):
        xs = led[k].get("tok_per_s") or []
        mark = "<- this run" if k == str(tag) else ""
        cost = (f", a {mnt}-tok answer ~{_fmt_secs(mnt / max(_best(led[k]), 1e-6))}"
                if mnt else "")
        say(f"                 kernels={k:<24} best {_best(led[k]):6.1f} tok/s "
            f"(n={len(xs)}){cost} {mark}")
    if len(led) < 2:
        say(f"  LLM          only one kernel arm measured in this ledger -- run "
            f"once more with the other arm for the before/after pair")
        # The torch-fallback arm has already been measured once, on the lp85 run
        # that motivated this: 18.1 tok/s, 22 calls, 100% of the wall clock. It is
        # named here so a first run WITH the kernels still reports a pair, and it
        # is labelled as a prior measurement rather than filed as one of this
        # kernel's own samples.
        if str(tag) != "none" and RECORDED_FALLBACK_TPS:
            say(f"  LLM          vs the recorded torch-fallback arm "
                f"({RECORDED_FALLBACK_TPS} tok/s, lp85 run 2026-08-14): "
                f"{_best(row) / RECORDED_FALLBACK_TPS:.1f}x")
    else:
        arms = sorted(led, key=lambda k: -_best(led[k]))
        hi, lo = _best(led[arms[0]]), _best(led[arms[-1]])
        if lo > 0:
            say(f"  LLM          fast kernels are worth {hi / lo:.1f}x on decode "
                f"({lo:.1f} -> {hi:.1f} tok/s)")
    return led


# The probe asks for the SHAPE each build actually has to parse, because a probe
# that only proves the GPU produces tokens has proved nothing about the pipeline.
PROBE_SYSTEM = "You reply with exactly what is asked for and nothing else."
PROBE_USER_CODE = ("Reply with a single fenced Python block, and nothing outside "
                   "it, containing exactly:\n\n```python\ndef ping():\n    "
                   "return 42\n```")
PROBE_USER_JSON = 'Reply with this exact JSON and nothing else: {"ok": true}'


def warm_llm(MA, cfg):
    """Load the model NOW, in the foreground, with the agent's own prints going
    straight to the console.

    Otherwise the load log (and any traceback) lands in the per-game capture
    buffer, the first minutes of the game run LLM-less, and a load failure looks
    like 'the LLM was never called'. A real generate() follows so three separate
    questions are settled before a single scored action is spent: does it load,
    how fast does it decode (tok/s -- what decides whether 25 games fit in the
    session), and can the agent PARSE what came back."""
    get = getattr(MA, "get_local_llm", None)
    if get is None:
        return {"skipped": "the agent module exports no get_local_llm()"}
    llm = get()
    path = llm_dir(llm)
    if not getattr(llm, "available", False):
        say(f"  LLM          NOT AVAILABLE at {path} "
            f"({getattr(llm, 'load_error', '') or 'disabled'}) "
            f"-- the run continues without a writer.")
        return {"available": False, "path": path,
                "error": str(getattr(llm, "load_error", ""))}
    # BEFORE the load: LLMCoder.load() probes for `fla` / `causal_conv1d` and
    # binds its attention path there. Installing them afterwards would change
    # only the log line, so this call cannot be moved below.
    kern = fast_kernels(cfg)

    say(f"  LLM          loading {path} (foreground; this can take "
        f"several minutes) ...")
    t0 = time.time()
    ok = False
    try:
        ok = bool(llm.ensure_loaded())
    except Exception as e:
        say(f"  LLM          load raised {type(e).__name__}: {e}")
        traceback.print_exc(file=_OUT)
    dt = time.time() - t0
    if not ok:
        say(f"  LLM          LOAD FAILED after {_fmt_secs(dt)}: "
            f"{getattr(llm, 'load_error', '') or 'see the [LLM] lines above'}. "
            f"Everything below runs without a writer.")
        return {"available": True, "loaded": False, "secs": round(dt, 1),
                "path": path, "error": str(getattr(llm, "load_error", "")),
                "kernels": kern}

    # The kernels that decide the decode speed. ABSENT here is not cosmetic: it
    # is the difference between a few seconds and a few minutes per call.
    fla = getattr(llm, "fla_present", None)
    cc1d = getattr(llm, "conv1d_present", None)
    foot = float(getattr(llm, "footprint_gb", 0.0) or 0.0)
    say(f"  LLM          loaded in {_fmt_secs(dt)}"
        + (f", {foot:.1f}GB resident" if foot else "")
        + (f"   fla={'PRESENT' if fla else 'ABSENT (torch fallback: slow)'}"
           if fla is not None else "")
        + (f"  causal_conv1d={'PRESENT' if cc1d else 'ABSENT'}"
           if cc1d is not None else ""))

    wants_code = callable(getattr(MA, "extract_code", None)) or callable(
        getattr(MA, "extract_candidate", None))
    t1 = time.time()
    try:
        probe_txt = llm_generate(
            llm, PROBE_SYSTEM,
            PROBE_USER_CODE if wants_code else PROBE_USER_JSON,
            max_new_tokens=64) or ""
    except Exception as e:
        say(f"  LLM          probe generate raised {type(e).__name__}: {e}")
        traceback.print_exc(file=_OUT)
        return {"available": True, "loaded": True, "secs": round(dt, 1),
                "path": path, "probe_error": f"{type(e).__name__}: {e}"}
    gen_dt = time.time() - t1
    n_out = int(getattr(llm, "last_out_tokens", 0) or 0)
    tps = (n_out / gen_dt) if (n_out and gen_dt > 0) else None
    parsed, reader = llm_parse_probe(MA, probe_txt)

    say(f"  LLM          probe generate {gen_dt:.1f}s"
        + (f" for {n_out} tok = {tps:.1f} tok/s" if tps else "")
        + f" -> {' '.join(probe_txt.split())[:70]!r}")
    if not probe_txt.strip():
        # Nothing came back, so the reader was never exercised. Calling that a
        # reader failure is how a preflight reported a broken reader while the
        # real run parsed 20 of 22 replies: an empty input is not a verdict.
        say(f"  LLM          reader NOT CHECKED: the probe returned no text "
            f"(a spent budget, a refused load, or a broken call -- not the reader)")
    elif parsed is None:
        say(f"  LLM          reader NOT CHECKED ({reader})")
    else:
        say(f"  LLM          reader {'OK' if parsed else 'FAILED'}: {reader}"
            + ("" if parsed else "  -- every budgeted call will be thrown away "
                                "at the reader, not at the model"))
    # Project the per-call cost onto the deliberation budget, so 'this cannot
    # finish' is known now rather than discovered nine hours in.
    mnt = int(getattr(MA, "LLM_MAX_NEW_TOKENS", 0) or 0)
    if tps and mnt:
        say(f"  LLM          at {tps:.1f} tok/s a full {mnt}-token answer costs "
            f"~{_fmt_secs(mnt / tps)}; the per-game LLM budget is "
            f"{_fmt_secs(float(getattr(llm, 'budget_s', 0.0) or 0.0))}")
    # File the rate under the kernel arm that produced it, and print every arm
    # measured so far -- the before/after pair the kernels are judged on.
    if tps:
        record_tps(cfg, kern.get("tag") or kernel_tag(), tps, n_out, mnt)
    return {"available": True, "loaded": True, "secs": round(dt, 1), "path": path,
            "footprint_gb": round(foot, 1), "fla": fla, "causal_conv1d": cc1d,
            "probe_secs": round(gen_dt, 2), "probe_out": probe_txt[:400],
            "probe_out_tokens": n_out, "tok_per_s": round(tps, 2) if tps else None,
            "probe_parse_ok": parsed if probe_txt.strip() else None, "reader": reader,
            "max_new_tokens": mnt or None, "kernels": kern}


FAKE_ANSWER = json.dumps({
    "hypothesis": "fake-llm rehearsal answer",
    "updated_hypothesis": "fake-llm rehearsal answer",
    "goal": None, "not_obstacle_colors": [], "obstacle_colors": [], "decoy_colors": [],
    "clicks": [[10, 10], [20, 20], [30, 30]],
})

# The LLM-as-coder build parses a PROGRAM, so the canned answer has to be one --
# a JSON blob would be discarded at the reader and the rehearsal would only ever
# exercise the failure path. This program is deliberately the simplest thing that
# can pass every check on a static frame: `parse_state` abstracts (CHECK 0 forbids
# a verbatim grid copy AND rejects a summary that collapses every frame to the
# same state, hence per-colour PIXEL COUNTS rather than just the palette --
# a first draft used the palette and CHECK 0 correctly threw it out), `render`
# reproduces the
# grid byte-exactly from the volatile `_grid` key (CHECK 1), `step` is identity so
# a no-op transition replays (CHECK 2), and `is_goal` never fires, so BFS finds no
# plan and the agent falls through to its probe floor. The four names are the
# agent's REQUIRED_DEFS -- the preflight sandbox check below is what keeps this
# copy honest if they ever change. It rehearses the PLUMBING; it models nothing.
FAKE_PROGRAM_CODE = '''import numpy as np

def parse_state(grid):
    g = np.asarray(grid)
    colors, counts = np.unique(g, return_counts=True)
    return {"h": int(g.shape[0]), "w": int(g.shape[1]),
            "counts": {int(c): int(n) for c, n in zip(colors, counts)},
            "_grid": g.tolist()}

def render(state):
    return np.array(state["_grid"], dtype=np.int64)

def step(state, action):
    return dict(state)

def is_goal(state):
    return False
'''

# The same program as a reply: fenced, with the notes block the reader also looks
# for. Kept derived from one source so the preflight sandbox check and the
# --fake-llm rehearsal cannot drift apart.
FAKE_PROGRAM = ("```python\n" + FAKE_PROGRAM_CODE + "```\n\n"
                "NOTES: fake-llm rehearsal program (no modelling content).\n")


def install_fake_llm(MA, delay=0.05):
    """Rehearse the LLM half WITHOUT a model (dev box, no GPU).

    Marks the singleton as loaded and answers every prompt with one canned reply
    in the shape THIS agent parses. Nothing about the model is being tested --
    the point is to prove, before a Kaggle session is spent, that the gates fire,
    the calls are counted, the reply survives the reader, and the report and
    findings render. `_generate_inner` is what we replace (not `generate`), so
    the real path -- budget gate, lock, extraction, certification -- still runs
    and the probe still sees the call."""
    llm = MA.get_local_llm()
    # `available` is a read-only PROPERTY on the LLM-as-coder writer (it is
    # derived from 'is there a config.json' and 'did the load fail'), so it
    # cannot simply be assigned -- and every budget gate reads it, so leaving it
    # False makes generate() return None and the whole rehearsal measure nothing.
    # Set the inputs the property is computed from, and only fall back to
    # assignment for the writer where it IS a plain attribute.
    try:
        llm.available = True
    except AttributeError:
        llm.enabled = True
        llm.load_error = ""
    llm.model = type("FakeModel", (), {"device": "cpu"})()
    # Both spellings: the two builds hold the tokenizer under different names and
    # `load()` short-circuits on `model`, so this only has to be non-None.
    llm.tok = llm.tokenizer = object()
    llm._load_attempted = True
    wants_code = callable(getattr(MA, "extract_code", None)) or callable(
        getattr(MA, "extract_candidate", None))
    answer = FAKE_PROGRAM if wants_code else FAKE_ANSWER

    def fake_inner(*a, **k):
        """Accepts EITHER arity: the single-prompt writer and the
        (system, user, max_new_tokens, max_time) writer both land here."""
        time.sleep(delay)
        return answer
    llm._generate_inner = fake_inner
    say(f"  LLM          FAKE (--fake-llm): canned "
        f"{'Python program' if wants_code else 'JSON'}, no model. Use this to "
        f"rehearse the plumbing only.")
    return {"fake": True, "shape": "code" if wants_code else "json"}


# ============================== the game loop ===============================
def world_changed(agent, prev_grid, new_grid, default=None):
    """Did the WORLD change, or only the HUD?

    The no-op rate is the harness's central diagnostic, and counting a ticking
    timer as 'the world moved' would flatter every agent. Each build knows its
    own mask and they are not the same kind of object:

      * my_agent.py    -- StateGraph learns which cells are volatile from the
                          periodicity of their change-gaps: ask it directly.
      * my_agent_llm.py -- masking is a DECLARATION by the certified model (keys
                          prefixed `_` are volatile), so the comparison runs on
                          parsed, masked states. Only a model that certified is
                          trusted here; before that there is nothing to ask.

    Falls back to raw-pixel change, never to a guess."""
    try:
        return bool(agent.sgraph.changed_masked(prev_grid, new_grid))
    except Exception:
        pass
    try:
        model = agent.delib.model
        if model is not None:
            mod = sys.modules[type(agent).__module__]
            mask, canon = mod.mask_volatile, mod.canon
            return canon(mask(model.parse(prev_grid))) != canon(mask(model.parse(new_grid)))
    except Exception:
        pass
    return default


def run_game(game_file, gid, meta, cfg, out_dir, board=None):
    """Play one game with the exact Kaggle framework loop (choose_action /
    is_done), recording a per-step diagnostic row. Any exception is captured and
    returned as a stall so a single bad game never kills the run.

    `board` is the OfficialScorecard for the whole run. Passing its manager into
    the wrapper is what makes the competition's own bookkeeping run: from here on
    every frame goes through ScorecardManager.update_scorecard()."""
    from arc_agi.local_wrapper import LocalEnvironmentWrapper
    from arcengine import GameState
    from my_agent import MyAgent

    seed = cfg["SEED"]
    max_actions = cfg["MAX_ACTIONS"]
    time_cap = cfg["TIME_CAP_SECS"]
    trace_mode = cfg["TRACE"]
    grid_mode = cfg["TRACE_GRIDS"]
    comp_mode = bool(cfg.get("COMPETITION_MODE", True))

    lg = logging.getLogger("kaggle_test")
    lg.setLevel(logging.ERROR)
    if not lg.handlers:
        lg.addHandler(logging.StreamHandler(sys.stderr))

    info, base_src = make_env_info(game_file, gid, meta)
    if board is not None and board.ok:
        board.register(info)
    wrapper = LocalEnvironmentWrapper(
        info, lg,
        scorecard_id=(board.card_id if board is not None and board.ok else "local"),
        seed=seed,
        scorecard_manager=(board.mgr if board is not None and board.ok else None))
    frame0 = wrapper.observation_space
    if frame0 is None:
        return {"game": gid, "error": "failed to load/reset game"}

    win_levels = int(getattr(frame0, "win_levels", 0) or 0) or meta.get("win_levels", 0)

    os.environ["ARC_AGENT_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    agent = MyAgent(game_id=gid)
    agent.MAX_ACTIONS = max_actions      # so the agent's own is_done matches ours
    frames = [frame0]

    trace_path = None
    tf = None
    if trace_mode != "none":
        tdir = os.path.join(out_dir, "traces")
        os.makedirs(tdir, exist_ok=True)
        trace_path = os.path.join(tdir, f"trace_{gid}_{int(time.time())}.jsonl")
        tf = open(trace_path, "w", encoding="utf-8")
        tf.write(json.dumps({"kind": "header", "game": gid, "seed": seed,
                             "max_actions": max_actions, "win_levels": win_levels,
                             "trace": trace_mode, "grids": grid_mode}) + "\n")

    steps = []                     # one compact dict per scored action
    actions_at_level = {}          # level -> cumulative action_counter when banked
    level_events = LevelEvents()   # every level CHANGE, drops included (leaderboard)
    level_events.observe(getattr(frame0, "levels_completed", 0) or 0, 0)
    best = 0
    stall = None
    game_overs = 0
    swallowed_resets = 0        # RESETs competition mode refused to perform
    t0 = time.time()

    def _grid(frame_obj):
        raw = getattr(frame_obj, "grid", None)
        if raw is None:
            raw = getattr(frame_obj, "frame", None)
        if raw is None:
            return None
        try:
            arr = np.array(raw, dtype=np.int32)
        except Exception:
            return None
        if arr.ndim == 3:                     # FrameData.frame is a LIST of grids
            arr = arr[-1]
        return arr if arr.size and arr.ndim == 2 else None

    try:
        while not agent.is_done(frames, frames[-1]) and agent.action_counter < max_actions:
            if time_cap and time.time() - t0 > time_cap:
                stall = f"per-game time cap {time_cap}s"
                break

            prev_grid = _grid(frames[-1])
            PROBE.ctx = {"game": gid, "i": agent.action_counter, "lvl": best}
            PROBE.step_begin()
            ts = time.time()
            try:
                action = agent.choose_action(frames, frames[-1])
            except Exception as e:
                stall = f"choose_action raised: {type(e).__name__}: {e}"
                traceback.print_exc(file=sys.stderr)
                break
            think = time.time() - ts

            data = {}
            if hasattr(action, "is_complex") and action.is_complex():
                data = {"x": int(getattr(action.action_data, "x", 0)),
                        "y": int(getattr(action.action_data, "y", 0))}
            is_reset = getattr(action, "name", str(action)) == "RESET"
            swallowed = False
            if (comp_mode and is_reset
                    and getattr(getattr(wrapper, "_game", None), "_action_count", -1) == 0):
                # COMPETITION MODE, reproduced from arc_agi/api.py:316-334.
                # A RESET sent when the engine's action count is 0 is the one that
                # would full_reset() and wipe the score -- so the hosted API DOES
                # NOT PERFORM IT. It returns the current frame and books the update
                # anyway: the action is spent, the world does not change, the score
                # survives. Without this branch a local run measures a game the
                # leaderboard will never play (there, the wipe is impossible).
                resp = wrapper.observation_space
                sc = board.raw() if (board is not None and board.ok) else None
                if sc is not None and resp is not None:
                    try:
                        sc.update_scorecard(wrapper._guid, resp, False)
                    except Exception as e:
                        say(f"    [scorecard] refused-RESET update failed: {e}")
                swallowed = True
                swallowed_resets += 1
            else:
                resp = wrapper.step(action, data=data)
            if resp is None:
                stall = "wrapper.step returned None"
                break
            frames.append(resp)
            if len(frames) > 64:                  # the agent only reads the tail
                del frames[:32]
            agent.action_counter += 1

            action_str = getattr(agent, "last_action_str", "") or ""
            src = PROBE.attribute_agent(agent, action_str, is_reset)

            new_grid = _grid(resp)
            raw_changed = None
            masked_changed = None
            if prev_grid is not None and new_grid is not None:
                raw_changed = not np.array_equal(prev_grid, new_grid)
                masked_changed = world_changed(agent, prev_grid, new_grid,
                                               default=raw_changed)

            lv = int(getattr(resp, "levels_completed", 0) or 0)
            leveled = lv > best
            while lv > best:                      # bank each new level exactly once
                best += 1
                actions_at_level[best] = agent.action_counter
            # The competition does NOT use the running max above. `Card.set_levels_
            # completed` appends on any CHANGE, drops included, and the scorer then
            # credits the i-th entry as level i+1 completed -- so a full_reset()
            # score wipe is paid for as a fresh level. `level_events` records what
            # the leaderboard actually sees; `actions_at_level` stays monotonic and
            # feeds the honest score. See eval/official_score.py.
            level_events.observe(lv, agent.action_counter)
            state = getattr(getattr(resp, "state", None), "name", str(getattr(resp, "state", "?")))
            if state == "GAME_OVER":
                game_overs += 1

            # last_action_str is stale on the GAME_OVER / NOT_PLAYED early-return
            # paths, so trust the action object for RESETs.
            act_name = "RESET" if is_reset else (action_str or getattr(action, "name", "?"))
            row = {
                "i": agent.action_counter,
                "lvl": lv,
                "src": src,
                "swal": swallowed,
                "act": act_name,
                "chg": raw_changed,
                "wchg": masked_changed,
                "h": int(hash(new_grid.tobytes())) if new_grid is not None else None,
                "dt": round(think, 4),
                "st": state,
                "up": leveled,
            }
            steps.append(row)

            if tf is not None:
                rec = dict(row)
                if trace_mode == "full":
                    rec["internals"] = _planner_internals(agent)
                want_grid = (grid_mode == "all"
                             or (grid_mode == "keyframes"
                                 and (leveled or state == "GAME_OVER"
                                      or agent.action_counter == 1)))
                if want_grid and new_grid is not None:
                    rec["grid"] = new_grid.tolist()
                tf.write(json.dumps(rec, default=_json_default) + "\n")

            if cfg["PROGRESS_EVERY"] and agent.action_counter % cfg["PROGRESS_EVERY"] == 0:
                say(f"    ... {gid}: {agent.action_counter}/{max_actions} actions, "
                    f"levels {best}/{win_levels}, {_fmt_secs(time.time() - t0)}")
    except KeyboardInterrupt:
        stall = "interrupted by user"
    finally:
        if tf is not None:
            try:
                tf.write(json.dumps({"kind": "footer", "levels": best,
                                     "actions": agent.action_counter}) + "\n")
                tf.close()
            except Exception:
                pass

    final_state = getattr(getattr(frames[-1], "state", None), "name",
                          str(getattr(frames[-1], "state", "?")))
    return {
        "game": gid,
        # The scorecard keys the game by its VERSIONED id, and the baselines it
        # divides by are the ones in EnvironmentInfo -- both are reported so the
        # honest score cannot end up using a different denominator.
        "env_game_id": info.game_id,
        "baselines": list(info.baseline_actions or []),
        "baseline_src": base_src,
        "levels": best,
        "win_levels": win_levels,
        "actions": agent.action_counter,
        "actions_at_level": actions_at_level,
        "level_events": level_events,          # the leaderboard's view (drops included)
        "state": final_state,
        "secs": round(time.time() - t0, 1),
        "stall": stall,
        "game_overs": game_overs,
        "competition_mode": comp_mode,
        "swallowed_resets": swallowed_resets,
        "steps": steps,
        "trace": trace_path,
        "final_internals": _planner_internals(agent),
    }


def _planner_internals(agent):
    """A small, defensive snapshot of what the agent currently believes. Every
    field is optional -- the agent's internals move around between iterations."""
    out = {}

    def grab(path, fn):
        try:
            out[path] = fn()
        except Exception:
            pass

    rp = getattr(agent, "rplanner", None)
    if rp is not None:
        grab("avatar_color", lambda: None if rp.avatar_color is None else int(rp.avatar_color))
        grab("action_disp", lambda: {k: list(v) for k, v in rp.action_disp.items()})
        grab("obstacles", lambda: sorted(int(c) for c in rp.obstacle_colors))
        grab("decoys", lambda: sorted(int(c) for c in rp.non_goal_colors))
        grab("walls", lambda: len(rp.walls))
        grab("cert_green", lambda: bool(rp._cert_green))
        grab("mechanical", lambda: bool(rp.looks_mechanical()))
    cp = getattr(agent, "cplanner", None)
    if cp is not None:
        grab("click_phase", lambda: str(cp._phase))
        grab("click_total", lambda: int(cp._total_clicks))
        grab("click_searching", lambda: bool(cp.searching))
    sg = getattr(agent, "sgraph", None)
    if sg is not None:
        grab("graph_states", lambda: len(sg.adj))
        grab("graph_untested", lambda: sum(len(v) for v in sg.untested.values()))
        grab("hud_masked_cells", lambda: int(sg._mask.sum()) if sg._mask is not None else 0)
    wm = getattr(agent, "world_model", None)
    if wm is not None:
        grab("world_model", lambda: None if wm.active_model is None
             else str(getattr(wm.active_model, "hypothesis", "?"))[:120])
    grab("llm_calls_this_level", lambda: int(agent.llm_calls_this_level))

    # --- the LLM-as-coder build ---
    # What it BELIEVES is one file, so the useful snapshot is: is there a
    # certified model at all, what did the last check say, and what is the
    # per-level cost trend (level 2 costing fewer turns than level 1 is the
    # transfer gate, and it is only visible from these two lists).
    dl = getattr(agent, "delib", None)
    if dl is not None:
        grab("delib", lambda: dict(dl.stats()))
        grab("model_digest", lambda: None if dl.model is None else dl.model.digest[:12])
        grab("model_notes", lambda: (dl.notes or "")[:160])
        grab("code_lines", lambda: len((dl.code or "").splitlines()))
        grab("last_check", lambda: None if dl.last_report is None
             else str(dl.last_report.summary())[:160])
    grab("mispredictions", lambda: int(agent.mispredictions))
    grab("plan_aborts", lambda: int(agent.plan_aborts))
    grab("actions_per_level", lambda: [int(x) for x in agent.level_costs])
    grab("llm_turns_per_level", lambda: [int(x) for x in agent.level_turns])
    grab("llm", lambda: dict(agent.llm.stats()))

    # Both builds keep a Timeline, but its container differs.
    grab("timeline_len", lambda: len(getattr(agent.timeline, "entries", agent.timeline)))
    return out


def llm_effect(llm, result, window=30):
    """Did the LLM half of the pipeline change anything? For each budgeted call,
    look at the next `window` scored actions and ask whether a level was banked
    and whether the world started changing again. This is the only honest way to
    tell 'the LLM ran' from 'the LLM helped'."""
    log = llm.get("log") or []
    steps = result.get("steps") or []
    llm["ups_after_call"] = 0
    llm["calls_followed_by_progress"] = 0
    if not log or not steps:
        return
    by_i = {s["i"]: s for s in steps}
    for rec in log:
        nxt = [by_i[j] for j in range(rec["at"] + 1, rec["at"] + 1 + window) if j in by_i]
        if not nxt:
            continue
        if any(s["up"] for s in nxt):
            llm["ups_after_call"] += 1
        ch = [s for s in nxt if s["wchg"] is not None]
        if ch and sum(1 for s in ch if s["wchg"]) / len(ch) > 0.5:
            llm["calls_followed_by_progress"] += 1


# ============================== metrics =====================================
def rhae(result, baseline_steps):
    """Both competition numbers for one game, via eval/official_score.py.

    Returns (honest_rhae_as_a_fraction, per_level, scores) so every existing
    caller that wanted "the RHAE number" keeps getting a capability number, while
    `scores` carries the rest:

        scores["official"]  0..100. As returned here this is the RECONSTRUCTION
                            (one play, level-CHANGE events, drops included);
                            merge_official() then REPLACES it with the number the
                            real Scorecard produced and keeps this one under
                            `official_reconstructed`. Read that pair, not this
                            value alone.
        scores["honest"]    0..100, each level counted once
        scores["wipes"]     score wipes seen in the frame stream
        scores["inflation"] reconstruction / honest

    The old body of this function divided by sum(1..win_levels) with no
    per-environment cap and monotonic level accounting. Two of those three were
    wrong. Never transcribe the formula here again.
    """
    ev = result.get("level_events")
    if ev is None:                       # crash rows carry no events
        ev = LevelEvents()
        prev = 0
        for k in sorted(result.get("actions_at_level") or {}):
            ev.observe(k, result["actions_at_level"][k])
            prev = k
    sc = score_run(ev, result.get("actions") or 0, list(baseline_steps or []),
                   result.get("win_levels") or 0)
    per_level = []
    lvl_scores = sc.get("honest_level_scores") or []
    for i, ai in enumerate(sc.get("honest_level_actions") or [], start=1):
        at = (result.get("actions_at_level") or {}).get(i)
        human = baseline_steps[i - 1] if i - 1 < len(baseline_steps) else None
        # The per-level score comes from the scorer, not from a local copy of
        # the formula. `official_score` publishes it for exactly this reason.
        per_level.append({"level": i, "ai": ai, "human": human,
                          "score": (lvl_scores[i - 1] / 100.0
                                    if i - 1 < len(lvl_scores) else 0.0),
                          "completed": at is not None})
    return sc["honest_rhae"], per_level, sc


def merge_official(scores, off):
    """Put the REAL scorecard's number in `official` and demote ours to a check.

    They are computed from different models of the same run: `official` now comes
    from Scorecard/EnvironmentScorecard (plays split on full_reset, score = max
    over plays), while the reconstruction treats the run as one play with the
    drops appended. When they disagree the report says so rather than quietly
    preferring one -- a disagreement is a modelling error worth seeing.
    """
    scores = dict(scores)
    scores["official_reconstructed"] = scores.get("official", 0.0)
    if not off:
        scores["official_source"] = "reconstructed (no scorecard)"
        return scores
    scores["official"] = off["score"]
    scores["official_source"] = "arc_agi.scorecard"
    scores["official_card"] = off
    d = abs(off["score"] - scores["official_reconstructed"])
    if d > 0.01:
        scores["official_mismatch"] = round(d, 4)
    return scores


def _p(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


def analyse(result, meta):
    """Turn the per-step rows into the numbers that explain the score."""
    steps = result.get("steps") or []
    n = len(steps)
    d = {"n": n}
    if not n:
        return d

    scored = [s for s in steps if s["act"] != "RESET"]
    wchg = [s for s in scored if s["wchg"] is not None]
    d["noop_rate"] = (1.0 - sum(1 for s in wchg if s["wchg"]) / len(wchg)) if wchg else None
    raw = [s for s in scored if s["chg"] is not None]
    d["raw_noop_rate"] = (1.0 - sum(1 for s in raw if s["chg"]) / len(raw)) if raw else None

    hashes = [s["h"] for s in steps if s["h"] is not None]
    hc = Counter(hashes)
    d["unique_states"] = len(hc)
    d["novel_rate"] = len(hc) / len(hashes) if hashes else None
    # Livelock: share of actions taken from a frame already seen >=5 times.
    d["revisit_rate"] = (sum(c for c in hc.values() if c >= 5) / len(hashes)) if hashes else None
    d["max_state_visits"] = max(hc.values()) if hc else 0

    # Count RESETs by ACTION, not by ledger row: the ClickPlanner's
    # env-as-simulator replay issues RESETs deliberately and they are charged
    # to 'click_search', but they still cost a scored action.
    d["resets"] = sum(1 for s in steps if s["act"] == "RESET")
    d["resets_after_death"] = sum(1 for s in steps if s["src"] == "reset")
    d["game_overs"] = result.get("game_overs", 0)

    dts = [s["dt"] for s in steps]
    d["think_total"] = round(sum(dts), 1)
    d["think_p50"] = round(_p(dts, 0.50), 4)
    d["think_p95"] = round(_p(dts, 0.95), 4)
    d["think_max"] = round(max(dts), 3)

    # Budget ledger: per subsystem, how many actions, its no-op rate, and how
    # many level-ups it produced. The row that is big and scores nothing is the
    # thing to fix -- RHAE squares the action ratio.
    ledger = {}
    for s in steps:
        e = ledger.setdefault(s["src"], {"n": 0, "chg": 0, "chg_n": 0, "ups": 0, "dt": 0.0})
        e["n"] += 1
        e["dt"] += s["dt"]
        if s["wchg"] is not None:
            e["chg_n"] += 1
            e["chg"] += 1 if s["wchg"] else 0
        if s["up"]:
            e["ups"] += 1
    for e in ledger.values():
        e["share"] = e["n"] / n
        e["noop_rate"] = (1.0 - e["chg"] / e["chg_n"]) if e["chg_n"] else None
        e["dt"] = round(e["dt"], 1)
    d["ledger"] = ledger

    d["actions_by_type"] = Counter(s["act"].split("_")[0] for s in steps)
    clicks = [s["act"] for s in steps if s["act"].startswith("ACTION6_")]
    if clicks:
        d["clicks"] = len(clicks)
        d["click_unique_cells"] = len(set(clicks))
        d["click_repeat_rate"] = 1.0 - len(set(clicks)) / len(clicks)

    # Per-level segments, including the unfinished tail, with their own no-op rate.
    segs, prev = [], 0
    bases = list(result["actions_at_level"].items())
    for lvl, at in bases:
        segs.append({"level": lvl, "from": prev, "to": at, "done": True})
        prev = at
    if prev < n:
        segs.append({"level": len(bases) + 1, "from": prev, "to": n, "done": False})
    for seg in segs:
        rows = steps[seg["from"]:seg["to"]]
        seg["actions"] = len(rows)
        ch = [r for r in rows if r["wchg"] is not None]
        seg["noop_rate"] = (1.0 - sum(1 for r in ch if r["wchg"]) / len(ch)) if ch else None
        seg["deaths"] = sum(1 for r in rows if r["st"] == "GAME_OVER")
        seg["top_src"] = Counter(r["src"] for r in rows).most_common(1)[0][0] if rows else None
    d["segments"] = segs

    base = meta.get("baseline_steps") or []
    if base and result["levels"] >= 1 and 1 in result["actions_at_level"]:
        d["l1_ratio"] = result["actions_at_level"][1] / base[0]
    elif base:
        # Never banked L1: how many human-lifetimes of actions did we spend?
        d["l1_ratio"] = n / base[0] if base[0] else None
    return d


def diagnose(result, ana, meta, llm, cfg):
    """Ranked, plain-language findings. Each is (severity 0-3, code, text);
    3 = this is why the score is what it is."""
    out = []
    gid = result["game"]
    n = ana.get("n", 0)
    if not n:
        return out
    lv, win = result["levels"], result["win_levels"]
    base = meta.get("baseline_steps") or []
    led = ana.get("ledger", {})

    def top_src():
        if not led:
            return None, 0.0
        k = max(led, key=lambda s: led[s]["n"])
        return k, led[k]["share"]

    if lv == 0:
        src, share = top_src()
        out.append((3, "no-level", (
            f"0/{win} levels in {n} actions. {_pct(share)} of the budget went to "
            f"'{src}' ({SRC_HELP.get(src, '?')}) and it never banked a level.")))
    elif lv < win:
        weight_done = sum(range(1, lv + 1)) / sum(range(1, win + 1))
        out.append((2, "partial", (
            f"stopped at level {lv}/{win}: only {_pct(weight_done)} of this game's "
            f"RHAE weight was even reachable ({n} actions spent).")))

    if ana.get("noop_rate") is not None and ana["noop_rate"] > 0.5:
        out.append((3, "noop", (
            f"{_pct(ana['noop_rate'])} of scored actions left the world unchanged "
            f"(HUD-masked). Those are pure RHAE loss -- they buy no information "
            f"the agent could not have imagined in an unscored model.")))
    elif ana.get("noop_rate") is not None and ana["noop_rate"] > 0.3:
        out.append((2, "noop", f"{_pct(ana['noop_rate'])} of scored actions were "
                               f"world-no-ops."))

    if ana.get("revisit_rate") is not None and ana["revisit_rate"] > 0.5:
        out.append((3, "livelock", (
            f"{_pct(ana['revisit_rate'])} of actions were taken from frames already "
            f"seen 5+ times (one frame hit {ana['max_state_visits']}x) -- the agent "
            f"is cycling, not exploring.")))

    if result.get("swallowed_resets", 0) >= 3:
        out.append((2 if result["swallowed_resets"] < 20 else 3, "reset-refused", (
            f"{result['swallowed_resets']} RESETs were REFUSED by competition mode "
            f"(sent when the engine's action count was 0, i.e. right after a level "
            f"change or another reset). Each still cost a scored action and changed "
            f"nothing, and the agent went on believing it had reset.")))

    if ana.get("game_overs", 0) >= 5:
        out.append((2, "deaths", (
            f"{ana['game_overs']} GAME_OVERs and {ana['resets']} RESETs; each reset "
            f"is a scored action and the replay back to the death point is paid for "
            f"again.")))

    if base and ana.get("l1_ratio"):
        r = ana["l1_ratio"]
        if lv >= 1 and r > 3:
            out.append((2, "l1-cost", (
                f"level 1 cost {result['actions_at_level'][1]} actions vs human "
                f"{base[0]} ({r:.1f}x) -> level_score "
                f"{level_score(base[0], result['actions_at_level'][1]) / 100.0:.3f}. "
                f"RHAE squares the ratio, so this alone caps the game.")))
        elif lv == 0:
            out.append((3, "l1-cost", (
                f"spent {r:.0f}x the human budget for level 1 ({n} vs {base[0]}) "
                f"without banking it.")))

    if ana.get("click_repeat_rate") is not None and ana["click_repeat_rate"] > 0.6:
        out.append((2, "click-repeat", (
            f"{_pct(ana['click_repeat_rate'])} of clicks re-hit a cell already "
            f"clicked ({ana['click_unique_cells']} unique of {ana['clicks']}).")))

    for src, e in sorted(led.items(), key=lambda kv: -kv[1]["n"]):
        # Bootstrap rows are EXPECTED to bank nothing -- they buy the first
        # transitions -- so flagging them as idle would bury the real finding.
        if src in ("reset", "bootstrap", "probe_bootstrap", "init"):
            continue
        if e["share"] > 0.35 and e["ups"] == 0 and lv >= 0:
            out.append((2, f"idle-{src}", (
                f"'{src}' took {_pct(e['share'])} of the budget ({e['n']} actions, "
                f"no-op {_pct(e['noop_rate'])}) and produced 0 level-ups.")))

    if ana.get("think_p50", 0) > 0.15:
        per_game = ana["think_total"] / max(n, 1) * cfg["MAX_ACTIONS"]
        out.append((2 if ana["think_p50"] > 0.5 else 1, "slow", (
            f"{ana['think_p50']:.2f}s/action median (p95 {ana['think_p95']:.2f}s). "
            f"At this rate one full-budget game is ~{_fmt_secs(per_game)}; 25 games "
            f"~{_fmt_secs(per_game * 25)} against a 9h session.")))

    # ---- LLM-specific findings: this is what a Kaggle run is FOR ----
    if llm.get("configured"):
        if llm["calls"] == 0 and not llm["load_attempted"]:
            out.append((2, "llm-idle", (
                "the LLM was never even asked to load -- the writer reported "
                "available=False (no config.json under LLM_MODEL_PATH), so the "
                "model dataset is not attached or the path is wrong.")))
        elif not llm["loaded"] and llm["calls"] == 0:
            out.append((3, "llm-load", (
                "the LLM load FAILED. See the [LLM] lines above for the exception; "
                "everything below ran without a writer.")))
        elif llm["calls"] == 0:
            out.append((2, "llm-unused", (
                "the LLM loaded but was never called on this game -- a gate before "
                "the writer never opened. Check the bootstrap threshold (one press "
                "per button before the first model is authored) and the "
                "deliberation interval against the ledger rows above.")))
        else:
            gs = llm["gen_secs"]
            avg = sum(gs) / len(gs)
            share = sum(gs) / max(result["secs"], 1e-6)
            sev = 2 if share > 0.35 else 1
            out.append((sev, "llm-cost", (
                f"{llm['calls']} LLM calls, {avg:.1f}s each ({_fmt_secs(sum(gs))} "
                f"total = {_pct(share)} of this game's wall time). Empty returns: "
                f"{llm['empty']}.")))
            if llm["empty"] >= max(2, 0.5 * llm["calls"]):
                out.append((3, "llm-empty", (
                    f"{llm['empty']} of {llm['calls']} LLM calls returned NOTHING "
                    f"usable. Either generate() is erroring (see [LLM] lines) or the "
                    f"answer is not the JSON the caller parses -- the exchanges "
                    f"printed above show which.")))
            if llm["calls"] >= 3 and llm.get("ups_after_call", 0) == 0:
                out.append((2, "llm-noeffect", (
                    f"{llm['calls']} LLM calls and not one was followed by a level-up "
                    f"within 30 actions ({llm.get('calls_followed_by_progress', 0)} "
                    f"were followed by sustained world-change). The LLM is running "
                    f"but not steering: check the ledger above for a row that its "
                    f"output actually produced ('plan' / 'experiment_*'), not just "
                    f"probe rows.")))
            if llm["theory_proposed"] and llm["theory_adopted"] == 0:
                out.append((2, "llm-rejected", (
                    f"all {llm['theory_proposed']} theorizer proposals were rejected "
                    f"by the full-timeline backtest -- the proposal schema is too "
                    f"narrow to express what this game actually needs.")))
            elif llm["theory_adopted"]:
                out.append((0, "llm-adopted", (
                    f"{llm['theory_adopted']}/{llm['theory_proposed']} theorizer "
                    f"proposals passed the backtest gate.")))

            # ---- the LLM-as-coder gates, in the order they must open ----
            # Reply -> code -> certified -> plan. Each finding names the FIRST
            # gate that closed, because a later zero is a consequence, not a
            # separate problem.
            if llm.get("backtests") or llm.get("code_replies") or llm.get("no_code"):
                if llm["code_replies"] == 0:
                    out.append((3, "coder-nocode", (
                        f"{llm['calls']} replies and NOT ONE contained usable code "
                        f"({llm['no_code']} unusable). This is a reader/format "
                        f"failure, not a modelling failure -- read the raw "
                        f"exchanges above before changing any prompt.")))
                elif llm["models_certified"] == 0:
                    by = llm.get("reject_by_check") or {}
                    worst = max(by, key=by.get) if by else "?"
                    out.append((3, "coder-nocert", (
                        f"{llm['code_replies']} programs were written and NONE "
                        f"certified in {llm['backtests']} backtests; the check that "
                        f"killed most of them was {worst} "
                        f"({dict(sorted(by.items(), key=lambda kv: -kv[1]))}). "
                        f"Fix that check's prompt guidance first.")))
                elif llm.get("plans_nonempty", 0) == 0:
                    out.append((2, "coder-noplan", (
                        f"{llm['models_certified']} model(s) certified but BFS found "
                        f"no winning path in {llm.get('plans', 0)} searches -- the "
                        f"model explains the past and predicts no future win, which "
                        f"is what the optimism trigger exists to catch.")))
                else:
                    out.append((0, "coder-ok", (
                        f"the full chain ran: {llm['code_replies']} programs, "
                        f"{llm['models_certified']} certified, "
                        f"{llm.get('plans_nonempty', 0)}/{llm.get('plans', 0)} "
                        f"searches produced a plan "
                        f"({llm.get('plan_actions', 0)} planned actions).")))
            if llm.get("out_tokens") and llm["gen_secs"]:
                tps = llm["out_tokens"] / max(sum(llm["gen_secs"]), 1e-6)
                if tps < 8:
                    out.append((3, "llm-slow-decode", (
                        f"{tps:.1f} tok/s. At this rate one full answer costs "
                        f"minutes and multi-turn deliberation is unaffordable -- "
                        f"check the fla / causal_conv1d wheels in the preflight "
                        f"before drawing any conclusion about reasoning quality.")))
                else:
                    out.append((0, "llm-decode", f"{tps:.1f} tok/s decode."))
    return sorted(out, key=lambda t: -t[0])


# ============================== reporting ===================================
def print_game_report(result, ana, per_level, gs, meta, llm, findings, capture, cfg,
                      scores=None):
    gid = result["game"]
    say("")
    say("=" * 78)
    say(f"  {gid.upper()}   levels {result['levels']}/{result['win_levels']}   "
        f"actions {result['actions']}   RHAE {gs:.4f}   "
        f"state {result['state']}   {_fmt_secs(result['secs'])}"
        + (f"   [{result['stall']}]" if result.get("stall") else ""))
    say("=" * 78)

    # What the competition's own scorecard made of this game. Everything on this
    # block is read back from EnvironmentScorecard -- none of it is our arithmetic.
    oc = (scores or {}).get("official_card")
    if oc:
        say(f"  OFFICIAL SCORECARD   score {oc['score']:.4f}/100   "
            f"levels {oc['levels_completed']}   actions {oc['actions']}   "
            f"resets {oc['resets']}   plays {oc['plays']}"
            + (f"   per-play {oc['per_play']}" if oc["plays"] > 1 else ""))
        if oc.get("message"):
            say(f"      !! scorer says: {oc['message']}  -> this game scores 0")
        if oc.get("level_scores"):
            say("      per level  " + "  ".join(
                f"L{i + 1}:{s:.1f}({a}a/{b}h)"
                for i, (s, a, b) in enumerate(zip(oc["level_scores"],
                                                  oc["level_actions"],
                                                  oc["baselines"]))))
        if (scores or {}).get("official_mismatch"):
            say(f"      note: our reconstruction says "
                f"{scores['official_reconstructed']:.4f} "
                f"(delta {scores['official_mismatch']}); the scorecard above is "
                f"the authority.")
    say(f"  baselines            {result.get('baseline_src', '?')}: "
        f"{result.get('baselines') or meta.get('baseline_steps')}")

    if per_level:
        say("  per-level RHAE")
        for p in per_level:
            hum = p["human"] if p["human"] is not None else "?"
            ratio = f"{p['ai'] / p['human']:.1f}x" if p["human"] else "?"
            say(f"    L{p['level']}  ai={p['ai']:<6} human={str(hum):<6} {ratio:<7} "
                f"level_score={p['score']:.4f}")
    for seg in ana.get("segments", []):
        if not seg["done"]:
            say(f"    L{seg['level']}  ai={seg['actions']:<6} UNFINISHED   "
                f"no-op {_pct(seg['noop_rate'])}  deaths {seg['deaths']}  "
                f"mostly '{seg['top_src']}'")

    led = ana.get("ledger", {})
    if led:
        say("")
        say(f"  budget ledger ({ana['n']} scored actions)")
        say(f"    {'subsystem':<13} {'actions':>8} {'share':>7} {'no-op':>7} "
            f"{'level-ups':>10}  {'think':>7}")
        order = [s for s in SRC_ORDER if s in led] + [s for s in led if s not in SRC_ORDER]
        for src in order:
            e = led[src]
            say(f"    {src:<13} {e['n']:>8} {_pct(e['share']):>7} "
                f"{_pct(e['noop_rate']):>7} {e['ups']:>10}  {e['dt']:>6.0f}s"
                f"   {SRC_HELP.get(src, '')}")

    say("")
    say("  action economy")
    say(f"    world-no-op rate   {_pct(ana.get('noop_rate'))}"
        f"   (raw-pixel {_pct(ana.get('raw_noop_rate'))})")
    say(f"    unique frames      {ana.get('unique_states')} "
        f"({_pct(ana.get('novel_rate'))} of actions were novel)")
    say(f"    revisit/livelock   {_pct(ana.get('revisit_rate'))} of actions from "
        f"frames seen 5+ times (worst frame {ana.get('max_state_visits')}x)")
    say(f"    deaths / resets    {ana.get('game_overs')} GAME_OVER, "
        f"{ana.get('resets')} RESET actions spent "
        f"({ana.get('resets_after_death', 0)} of them death-recovery, the rest "
        f"deliberate search replay)")
    if result.get("competition_mode"):
        say(f"    refused RESETs     {result.get('swallowed_resets', 0)} of those "
            f"would have wiped the score; competition mode refuses them -- the "
            f"action is spent, the world does not change")
    say(f"    think time         p50 {ana.get('think_p50')}s  p95 "
        f"{ana.get('think_p95')}s  max {ana.get('think_max')}s  "
        f"total {_fmt_secs(ana.get('think_total', 0))}")
    if "clicks" in ana:
        say(f"    clicks             {ana['clicks']} "
            f"({ana['click_unique_cells']} unique cells, repeat "
            f"{_pct(ana['click_repeat_rate'])})")

    fin = result.get("final_internals") or {}
    if fin:
        say("")
        say("  agent state at the end")
        for k in ("avatar_color", "action_disp", "obstacles", "decoys", "walls",
                  "cert_green", "mechanical", "click_phase", "click_total",
                  "graph_states", "graph_untested", "hud_masked_cells",
                  "world_model", "timeline_len",
                  # the LLM-as-coder build
                  "delib", "model_digest", "code_lines", "last_check",
                  "mispredictions", "plan_aborts",
                  "actions_per_level", "llm_turns_per_level"):
            if k in fin:
                say(f"    {k:<18} {fin[k]}")
        if fin.get("model_notes"):
            say(f"    {'model_notes':<18} {' '.join(fin['model_notes'].split())}")
        # The transfer gate, spelled out: level 2 must cost FEWER LLM turns than
        # level 1, or the agent is re-deriving the game every level and the RHAE
        # denominator (which spans all levels) can never be paid off.
        turns = fin.get("llm_turns_per_level") or []
        if len(turns) >= 2:
            say(f"    {'transfer':<18} LLM turns per level {turns} -> "
                + ("EDITING (later levels cost less)" if turns[-1] < turns[0]
                   else "NO TRANSFER (later levels cost the same or more)"))

    if llm.get("configured"):
        say("")
        say("  LLM")
        # Report on CALLS, not on the load flag: a model that was already warm
        # never re-enters the load path, and the calls are what matter.
        if llm["loaded"] or llm["calls"]:
            gs_list = llm["gen_secs"]
            say(f"    loaded             yes"
                + (f" in {llm['load_secs']}s" if llm.get("load_secs") else " (warm)")
                + (f", {llm['vram_gb']:.1f}GB VRAM used" if llm.get("vram_gb") else ""))
            say(f"    calls              {llm['calls']} "
                f"({dict(llm['by_kind'])}) empty={llm['empty']} errors={llm['errors']}")
            if gs_list:
                say(f"    latency            p50 {_p(gs_list, 0.5):.1f}s  "
                    f"p95 {_p(gs_list, 0.95):.1f}s  total {_fmt_secs(sum(gs_list))}")
            if llm.get("out_tokens"):
                tot = sum(gs_list) or 1e-6
                say(f"    decode             {llm['out_tokens']} tok out / "
                    f"{llm.get('in_tokens', 0)} in  ->  "
                    f"{llm['out_tokens'] / tot:.1f} tok/s")
            # The gate chain: a zero here localises the failure to one step.
            if llm.get("backtests") or llm.get("code_replies") or llm.get("no_code"):
                say(f"    code out of replies {llm['code_replies']} usable / "
                    f"{llm['no_code']} unusable")
                say(f"    certified          {llm['models_certified']} of "
                    f"{llm['backtests']} backtests"
                    + (f"   rejected by {dict(sorted((llm.get('reject_by_check') or {}).items(), key=lambda kv: -kv[1]))}"
                       if llm.get("reject_by_check") else ""))
                say(f"    bfs plans          {llm.get('plans_nonempty', 0)} of "
                    f"{llm.get('plans', 0)} searches found a path "
                    f"({llm.get('plan_actions', 0)} actions planned)")
            if llm["theory_proposed"]:
                say(f"    theorizer          {llm['theory_adopted']} adopted / "
                    f"{llm['theory_rejected']} rejected of {llm['theory_proposed']}")
            if llm["click_plans"]:
                say(f"    click plans        {llm['click_plans']} "
                    f"({llm['click_plan_targets']} targets)")
            if llm["calls"]:
                say(f"    did it help        {llm.get('ups_after_call', 0)} calls were "
                    f"followed by a level-up within 30 actions; "
                    f"{llm.get('calls_followed_by_progress', 0)} by a >50% "
                    f"world-change rate")
            n_show = cfg.get("SHOW_LLM_CALLS", 0)
            for rec in (llm.get("log") or [])[-n_show:] if n_show else []:
                head = (f"    call @a{rec['at']} L{rec['lvl']} {rec['kind']} "
                        f"{rec['secs']}s  prompt {rec['prompt_chars']}ch -> "
                        f"{rec['out_chars']}ch")
                say(head + (f"  ERROR {rec['error']}" if rec["error"] else ""))
                body = " ".join((rec["out"] or "").split())
                say(f"      {(body[:300] + '...') if len(body) > 300 else (body or '<empty>')}")
        else:
            say(f"    loaded             NO"
                f"{'  (load attempted and failed)' if llm['load_attempted'] else '  (never attempted)'}")
            say(f"    model path         {llm.get('model_path')}")

    if capture.markers:
        say("")
        say("  agent log markers   " + "  ".join(f"[{k}]x{v}" for k, v in
                                                 capture.markers.most_common(10)))
    notable = capture.notable[-6:]
    for ln in notable:
        say(f"    {ln[:110]}")
    seen = set(notable)
    tail = [ln for ln in capture.tail() if ln not in seen]
    if tail and cfg["QUIET_AGENT"]:
        say("  last agent lines")
        for ln in tail:
            say(f"    {ln[:110]}")

    if findings:
        say("")
        say("  FINDINGS")
        for sev, code, text in findings:
            flag = {3: "!!!", 2: " !!", 1: "  !", 0: "  +"}[sev]
            say(f"   {flag} [{code}] {text}")
    if result.get("trace"):
        say(f"\n  trace: {result['trace']}")


def print_summary(rows, cfg, started, board=None):
    say("")
    say("#" * 78)
    say("  SUMMARY")
    say("#" * 78)
    say(f"  {'game':<7} {'levels':>8} {'actions':>8} {'official':>9} {'honest':>8} "
        f"{'wipe':>5} {'no-op':>7} {'deaths':>7} {'s/act':>7}  {'state'}")
    for r in rows:
        a = r["analysis"]
        sc = r.get("scores") or {}
        say(f"  {r['game']:<7} {str(r['levels']) + '/' + str(r['win_levels']):>8} "
            f"{r['actions']:>8} {sc.get('official', 0.0):>9.4f} "
            f"{sc.get('honest', 0.0):>8.4f} {sc.get('wipes', 0):>5} "
            f"{_pct(a.get('noop_rate')):>7} {a.get('game_overs', 0):>7} "
            f"{a.get('think_p50', 0):>7.3f}  {r['state']}"
            + (f"  [{r['stall']}]" if r.get("stall") else ""))
    if rows:
        mean = sum(r["rhae"] for r in rows) / len(rows)
        mean_off = sum((r.get("scores") or {}).get("official", 0.0) for r in rows) / len(rows)
        mean_hon = sum((r.get("scores") or {}).get("honest", 0.0) for r in rows) / len(rows)
        wipes = sum((r.get("scores") or {}).get("wipes", 0) for r in rows)
        banked = sum(r["levels"] for r in rows)
        possible = sum(r["win_levels"] for r in rows)
        say("")
        # The headline number is read back from EnvironmentScorecard -- the same
        # object the leaderboard scores. `honest` is beside it because the official
        # one takes the MAX over plays, so a lucky early play can outrank the run
        # you actually care about.
        ov = board.overall() if (board is not None and board.ok) else None
        if ov:
            say(f"  OFFICIAL SCORECARD         : {ov['score']:.4f}   /100  "
                f"(arc_agi.scorecard, mean over {ov['environments']} environments)")
            say(f"    levels credited          : {ov['levels_completed']} of "
                f"{ov['levels']} scored levels, {ov['actions']} actions "
                f"(RESETs included)")
            say(f"    per game                 : "
                + "  ".join(f"{k.split('-')[0]}:{v:.2f}" for k, v in
                            sorted(ov["per_game"].items())))
        else:
            say(f"  OFFICIAL SCORECARD         : unavailable "
                f"(arc_agi.scorecard did not load); the number below is ours")
            say(f"  MEAN OFFICIAL (reconstructed): {mean_off:.4f}   /100")
        say(f"  MEAN HONEST   (capability) : {mean_hon:.4f}   /100  "
            f"(= {mean:.6f} as a fraction)")
        mism = [r["game"] for r in rows if (r.get("scores") or {}).get("official_mismatch")]
        if mism:
            say(f"  !! our reconstruction disagrees with the scorecard on "
                f"{' '.join(mism)} (mean recon {mean_off:.4f}). The scorecard wins; "
                f"the gap is a bug in the reconstruction, not in the agent.")
        if wipes:
            say(f"  note: {wipes} score-wipe event(s) seen. Under the real scorecard a "
                f"full_reset starts a NEW PLAY and the game keeps the MAX over "
                f"plays -- churn cannot inflate the official number, it only "
                f"throws away the play in progress.")
        say(f"  levels banked              : {banked}/{possible}")
        total_think = sum(r["analysis"].get("think_total", 0) for r in rows)
        wall = time.time() - started
        play = sum(r.get("secs", 0) for r in rows)
        say(f"  wall time                  : {_fmt_secs(wall)} "
            f"({_fmt_secs(play)} playing, {_fmt_secs(total_think)} of that inside "
            f"choose_action; the rest is import + model load)")
        # Project from PLAY time only -- the ~40s torch/my_agent import and the
        # one-off LLM load are paid once per session, not once per game.
        per_game = play / len(rows)
        say(f"  projected 25-game run      : {_fmt_secs(per_game * 25)} of play "
            f"at {cfg['MAX_ACTIONS']} actions/game (+ one-off startup)")
        say("  (competition overall = mean over all 25 games; RHAE squares the "
            "human/ai action ratio)")

    # The cross-game view: which subsystem is eating the budget overall.
    agg = defaultdict(lambda: {"n": 0, "ups": 0, "chg": 0, "chg_n": 0})
    for r in rows:
        for src, e in r["analysis"].get("ledger", {}).items():
            agg[src]["n"] += e["n"]
            agg[src]["ups"] += e["ups"]
            agg[src]["chg"] += e["chg"]
            agg[src]["chg_n"] += e["chg_n"]
    if agg:
        tot = sum(e["n"] for e in agg.values()) or 1
        say("")
        say("  where the whole budget went")
        for src, e in sorted(agg.items(), key=lambda kv: -kv[1]["n"]):
            nr = (1.0 - e["chg"] / e["chg_n"]) if e["chg_n"] else None
            say(f"    {src:<13} {e['n']:>7} actions  {_pct(e['n'] / tot):>6}  "
                f"no-op {_pct(nr):>5}  {e['ups']} level-ups   "
                f"{SRC_HELP.get(src, '')}")

    print_next_steps(rows, agg)
    return agg


# Each finding code maps to the concrete thing to change. Kept next to the
# diagnoses so a new code without an action is obvious.
NEXT_STEP = {
    "llm-load": "Fix the model load FIRST -- nothing else about the LLM half is "
                "measurable until it loads. Re-run with --preflight to see the "
                "checkpoint's model_type/quantization against the installed "
                "transformers.",
    "llm-idle": "Attach the model dataset / point $ARC_LLM_PATH at the directory "
                "that holds config.json. Until then this is a no-LLM measurement.",
    "llm-empty": "The model answers but the caller cannot parse it. Make the "
                 "reader tolerate the wrapper it is actually emitting (fenced "
                 "code, prose preamble, literal newlines inside a string) rather "
                 "than tightening the prompt -- answers are lost at the reader.",
    "coder-nocode": "Nothing usable came out of any reply. Read the raw exchanges "
                    "printed above: if the program is there but unfenced or "
                    "wrapped, that is the reader's job to accept, not the "
                    "prompt's job to prevent.",
    "coder-nocert": "One check is killing every program. CHECK 1 means the "
                    "representation cannot reproduce the grid (grounding); "
                    "CHECK 2 means the mechanism is wrong; REJECTED/TIMEOUT means "
                    "the sandbox, not the model. Give the guidance for THAT check.",
    "coder-noplan": "Models certify but admit no win. This is the optimism case: "
                    "a model that explains the past and forbids winning is wrong "
                    "about the goal -- feed the failed search back as the bug.",
    "llm-slow-decode": "Decode speed, not reasoning, is the binding constraint. "
                       "Install the fla / causal_conv1d wheels (see the "
                       "`kernels` lines: re-run with --install-kernels, or "
                       "ARC_INSTALL_KERNELS=1 in a notebook) before tuning "
                       "anything about the prompts. tok_per_s.json holds the "
                       "before/after pair.",
    "llm-unused": "The budget gates never fired. Either lower them for the game "
                  "types that stall early, or trigger on a stall signal (no-op "
                  "streak / livelock) instead of a fixed action count.",
    "llm-noeffect": "The LLM's output is not reaching an action. Make its plan a "
                    "first-class producer (a real 'llm_click'/'llm_plan' ledger "
                    "row), not just a hint another planner may ignore.",
    "llm-rejected": "The proposal schema cannot express what these games need. "
                    "Widen what a theory may assert before spending more calls.",
    "llm-cost": "The LLM is eating wall time. Cap calls per level or shrink "
                "max_new_tokens; on a 9h session this bounds how many games run.",
    "noop": "Most scored actions change nothing. Predict the no-op in an unscored "
            "model and prune it from the frontier before spending the action.",
    "livelock": "The agent cycles through frames it has already seen. Add a "
                "revisit penalty / forced-novelty escape keyed on the frame hash.",
    "no-level": "The dominant subsystem never banks a level: that subsystem's goal "
                "model is wrong for this game type, not merely slow.",
    "l1-cost": "Level 1 alone caps the game score. RHAE squares the ratio, so "
               "halving level-1 actions is worth ~4x on this game.",
    "click-repeat": "Clicks re-hit known cells. Track clicked cells per level and "
                    "make coverage sample unvisited cells first.",
    "deaths": "Deaths are expensive twice (the RESET plus the replay). Learn the "
              "lethal transition and treat it as a wall in the planner.",
    "reset-refused": "Never send RESET when the engine's action count is 0 (right "
                     "after a level change, or twice in a row). On the leaderboard "
                     "it is refused and charged; locally it wipes the score. Track "
                     "'has an action been taken since the last reset' and gate on it.",
    "slow": "Per-action think time bounds how many of the 25 games a session can "
            "reach. Profile the per-step planner work before adding anything.",
    "partial": "Levels were left on the table; the denominator spans ALL levels, "
               "so unfinished levels cap the score even when early ones are cheap.",
}


def print_next_steps(rows, agg):
    """Roll the per-game findings up into an ordered work list. Ranked by
    (how severe, how many games show it) -- what to fix next, not a data dump."""
    if not rows:
        return
    tally = defaultdict(lambda: {"sev": 0, "games": [], "text": ""})
    for r in rows:
        for sev, code, text in r.get("findings") or []:
            if sev == 0:
                continue
            t = tally[code]
            t["sev"] = max(t["sev"], sev)
            t["games"].append(r["game"])
            t["text"] = t["text"] or text
    if not tally:
        return
    say("")
    say("  NEXT STEPS  (ranked: severity, then how many games show it)")
    ranked = sorted(tally.items(), key=lambda kv: (-kv[1]["sev"], -len(kv[1]["games"])))
    for i, (code, t) in enumerate(ranked[:8], 1):
        say(f"   {i}. [{code}] {len(t['games'])}/{len(rows)} games "
            f"({' '.join(sorted(set(t['games'])))})")
        action = NEXT_STEP.get(code) or (
            NEXT_STEP.get(code.split("-")[0])
            if code.split("-")[0] in NEXT_STEP else None)
        if code.startswith("idle-"):
            action = (f"'{code[5:]}' burns budget for no level-ups: either give it a "
                      f"goal model that can finish, or stop letting it own the "
                      f"fallback slot for this game type.")
        say(f"      -> {action or t['text']}")


# ================================= main =====================================
def main(config=None):
    started = time.time()
    cfg = dict(CONFIG)
    if config:
        cfg.update(config)
    _prepare_process(cfg)

    if not _in_notebook():
        import argparse
        ap = argparse.ArgumentParser(description="Diagnostic eval for my_agent.py")
        ap.add_argument("--games", nargs="*", default=None,
                        help="ids, file paths, or a preset (quick5/all/click/movement/mixed)")
        ap.add_argument("--seed", type=int, default=cfg["SEED"])
        ap.add_argument("--max-actions", type=int, default=cfg["MAX_ACTIONS"])
        ap.add_argument("--time-cap", type=int, default=cfg["TIME_CAP_SECS"])
        ap.add_argument("--total-budget", type=int, default=cfg["TOTAL_TIME_BUDGET_SECS"])
        ap.add_argument("--out", default=cfg["OUT_DIR"])
        ap.add_argument("--games-dir", default=cfg["GAMES_DIR"])
        ap.add_argument("--agent-dir", default=cfg["AGENT_DIR"])
        ap.add_argument("--trace", default=cfg["TRACE"], choices=["none", "steps", "full"])
        ap.add_argument("--trace-grids", default=cfg["TRACE_GRIDS"],
                        choices=["none", "keyframes", "all"])
        ap.add_argument("--verbose-agent", action="store_true",
                        help="mirror the agent's own prints live")
        ap.add_argument("--preflight", action="store_true",
                        help="run the environment/LLM checks and stop")
        ap.add_argument("--no-llm-warmup", action="store_true",
                        help="do not foreground-load the LLM before the games")
        ap.add_argument("--force", action="store_true",
                        help="run even if a preflight check FAILed")
        ap.add_argument("--show-llm-calls", type=int, default=cfg["SHOW_LLM_CALLS"],
                        help="print this many raw LLM exchanges per game")
        ap.add_argument("--fake-llm", action="store_true",
                        help="rehearse the LLM plumbing with canned answers (no model)")
        ap.add_argument("--no-competition-mode", action="store_true",
                        help="do NOT refuse score-wiping RESETs (raw local rules)")
        ap.add_argument("--env", nargs="*", default=[], metavar="K=V",
                        help="env vars for the agent, e.g. --env ARC_NO_THEORIZE=1")
        ap.add_argument("--install-kernels", action="store_true",
                        help="pip install the fla / causal_conv1d wheels before "
                             "the model loads (off by default; reported either way)")
        ap.add_argument("--wheels", default=cfg["WHEELS_DIR"], metavar="DIR",
                        help="directory of offline .whl files for --install-kernels")
        args, _ = ap.parse_known_args()
        if args.games:
            cfg["GAMES"] = args.games[0] if len(args.games) == 1 else args.games
        cfg.update(SEED=args.seed, MAX_ACTIONS=args.max_actions,
                   TIME_CAP_SECS=args.time_cap, TOTAL_TIME_BUDGET_SECS=args.total_budget,
                   OUT_DIR=args.out, GAMES_DIR=args.games_dir, AGENT_DIR=args.agent_dir,
                   TRACE=args.trace, TRACE_GRIDS=args.trace_grids)
        if args.verbose_agent:
            cfg["QUIET_AGENT"] = False
        cfg.update(PREFLIGHT_ONLY=args.preflight, FORCE=args.force,
                   SHOW_LLM_CALLS=args.show_llm_calls, FAKE_LLM=args.fake_llm,
                   COMPETITION_MODE=not args.no_competition_mode,
                   INSTALL_KERNELS=args.install_kernels, WHEELS_DIR=args.wheels)
        if args.no_llm_warmup:
            cfg["WARM_LLM"] = False
        env = dict(cfg.get("ENV") or {})
        for kv in args.env:
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k] = v
        cfg["ENV"] = env

    if _in_notebook() and os.environ.get("PYTHONHASHSEED") != "0":
        say("NOTE: PYTHONHASHSEED is not pinned in this kernel -- results are "
            "reproducible within this session but may shift after a restart.\n"
            "      Use the `!python Kaggle_test.py` route for exact reproducibility.\n")

    for k, v in (cfg.get("ENV") or {}).items():         # must precede the import
        os.environ[str(k)] = str(v)

    try:
        from arc_agi.local_wrapper import LocalEnvironmentWrapper  # noqa: F401
    except ImportError:
        say("arc_agi is not installed in this kernel. Install it first:\n"
            "    !pip install -q arc-agi python-dotenv\n"
            "  (or run the offline-wheels install cell of the submission notebook).")
        return None

    try:
        agent_dir = find_agent_dir(cfg)
        games_root = find_games_root(cfg)
    except FileNotFoundError as e:
        say(f"SETUP ERROR: {e}")
        return None

    out_dir = cfg["OUT_DIR"] or ("/kaggle/working" if os.path.isdir("/kaggle/working")
                                 else os.getcwd())
    os.makedirs(out_dir, exist_ok=True)

    index, index_src = load_index(cfg, games_root)
    games = resolve_games(cfg["GAMES"], index)

    # Import once: the LLM singleton then loads once and is reused across games.
    t_imp = time.time()
    import my_agent as MA
    PROBE.install(MA)

    # ONE scorecard for the whole run -- that is what a submission is. Every
    # wrapper below gets its manager, so the competition's own code does the
    # recording and `board.compute()` reads the score straight back out.
    board = OfficialScorecard(competition_mode=cfg.get("COMPETITION_MODE", True))

    llm_path = getattr(MA, "LLM_MODEL_PATH", "?")
    # The budget is spelled differently by the two builds -- calls per level in
    # the symbolic one, wall-clock seconds per game in the LLM-as-coder one.
    # Report whichever exists; never invent the other.
    llm_conf = {"configured": True, "model_path": llm_path,
                "embed_path": getattr(MA, "EMBED_MODEL_PATH", "") or "(none)",
                "budget_per_level": getattr(MA, "LLM_CALL_BUDGET_PER_LEVEL", None),
                "budget_secs_per_game": getattr(MA, "LLM_BUDGET_S", None),
                "max_new_tokens": getattr(MA, "LLM_MAX_NEW_TOKENS", None)}

    say("=" * 78)
    say("  ARC-AGI-3 AGENT DIAGNOSTIC EVAL")
    say("=" * 78)
    say(f"  agent        {os.path.join(agent_dir, 'my_agent.py')} "
        f"(imported in {time.time() - t_imp:.1f}s)")
    say(f"  attribution  {PROBE.attribution_note()}")
    say(f"  games root   {games_root}")
    say(f"  baselines    {index_src}  (per game, metadata.json wins)")
    say(f"  scoring      "
        + (f"arc_agi.scorecard, card {board.card_id[:8]}" if board.ok
           else "NO OFFICIAL SCORER -- arc_agi.scorecard failed to import")
        + f"   |  honest via {SCORER_SRC}")
    say(f"  reset rules  "
        + ("competition_mode: a RESET that would wipe the score is REFUSED "
           "(it still costs an action)" if cfg.get("COMPETITION_MODE", True)
           else "RAW LOCAL (--no-competition-mode): a second RESET wipes every "
                "banked level"))
    say(f"  out dir      {out_dir}")
    say(f"  games        {len(games)}: {' '.join(games)}")
    say(f"  seed {cfg['SEED']}   max_actions {cfg['MAX_ACTIONS']}"
        + (f"   per-game cap {cfg['TIME_CAP_SECS']}s" if cfg["TIME_CAP_SECS"] else "")
        + (f"   total budget {cfg['TOTAL_TIME_BUDGET_SECS']}s"
           if cfg["TOTAL_TIME_BUDGET_SECS"] else ""))
    say(f"  LLM path     {llm_path}"
        + (f"   budget {llm_conf['budget_secs_per_game']:.0f}s/game"
           if llm_conf.get("budget_secs_per_game") else "")
        + (f"   budget {llm_conf['budget_per_level']} calls/level"
           if llm_conf.get("budget_per_level") else "")
        + (f"   max_new_tokens {llm_conf['max_new_tokens']}"
           if llm_conf.get("max_new_tokens") else ""))
    say(f"  embed path   {llm_conf['embed_path']}")
    if cfg.get("ENV"):
        say(f"  agent env    {cfg['ENV']}")

    pre, worst = preflight(MA, agent_dir, games_root, index_src, games, cfg)
    if cfg.get("PREFLIGHT_ONLY"):
        say("\n  (--preflight: stopping here, no actions spent)")
        return {"preflight": pre}
    if worst >= 2:
        say("\n  One or more preflight checks FAILED. Fix those first -- a run "
            "started now would measure the wrong thing.\n  (Pass --force to run "
            "anyway; --preflight to stop here.)")
        if not cfg.get("FORCE"):
            return {"preflight": pre, "aborted": True}

    if cfg.get("FAKE_LLM"):
        warm = install_fake_llm(MA)
    elif cfg.get("WARM_LLM", True):
        warm = warm_llm(MA, cfg)
    else:
        warm = {"skipped": "WARM_LLM=False"}
    say("")

    rows, report_games = [], []
    report_path = os.path.join(out_dir, "kaggle_test_report.json")

    def write_report(final=False):
        """Written after EVERY game, not just at the end: a Kaggle session that
        is killed mid-run still leaves everything measured so far on disk."""
        rep = {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "complete": final,
            "config": {k: v for k, v in cfg.items()},
            "agent_dir": agent_dir, "games_root": games_root,
            "baselines": index_src, "python": sys.version.split()[0],
            "preflight": pre, "llm_warmup": warm,
            # mean_rhae is the HONEST number as a fraction, kept under its old name
            # so older reports stay comparable. mean_official is what the
            # leaderboard pays; the two diverge exactly when the agent churns
            # full_resets. See eval/official_score.py.
            # The authoritative block: what arc_agi.scorecard says about this run.
            "scorer": {"official": "arc_agi.scorecard" if board.ok else None,
                       "honest": SCORER_SRC},
            "official_scorecard": board.overall() if board.ok else None,
            "mean_rhae": (sum(r["rhae"] for r in rows) / len(rows)) if rows else 0.0,
            "mean_official": (sum((r.get("scores") or {}).get("official", 0.0)
                                  for r in rows) / len(rows)) if rows else 0.0,
            "mean_honest": (sum((r.get("scores") or {}).get("honest", 0.0)
                                for r in rows) / len(rows)) if rows else 0.0,
            "total_wipes": sum((r.get("scores") or {}).get("wipes", 0) for r in rows),
            "wall_secs": round(time.time() - started, 1),
            "games": report_games,
        }
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=1, default=_json_default)
        except Exception as e:
            say(f"  could not write the report: {e}")
        return rep

    for raw in games:
        if cfg["TOTAL_TIME_BUDGET_SECS"] and time.time() - started > cfg["TOTAL_TIME_BUDGET_SECS"]:
            say(f"  [total time budget {cfg['TOTAL_TIME_BUDGET_SECS']}s reached; "
                f"skipping the rest]")
            break
        if os.path.isfile(str(raw)):
            gf, gid = raw, os.path.splitext(os.path.basename(raw))[0]
        else:
            gid = str(raw)
            gf = find_game_file(games_root, gid)
        if gf is None:
            say(f"  {gid:<6} MISSING (no game file under {games_root})")
            continue

        meta = index.get(gid, {"win_levels": 0, "baseline_steps": [], "available_actions": []})
        say(f"  running {gid} ...")

        # Per-game counters only; 'loaded'/'load_secs' persist (one model per process).
        PROBE.reset_game_counters()

        capture = LogCapture(cfg["QUIET_AGENT"], cfg["AGENT_LOG_TAIL"])
        real_stdout = sys.stdout
        sys.stdout = capture
        try:
            result = run_game(gf, gid, meta, cfg, out_dir, board=board)
        except KeyboardInterrupt:
            sys.stdout = real_stdout
            say("  interrupted -- reporting what has been collected so far")
            break
        except Exception as e:
            sys.stdout = real_stdout
            say(f"  {gid:<6} CRASHED: {type(e).__name__}: {e}")
            traceback.print_exc(file=_OUT)
            result = {"game": gid, "error": f"{type(e).__name__}: {e}", "levels": 0,
                      "win_levels": meta.get("win_levels", 0), "actions": 0,
                      "actions_at_level": {}, "state": "CRASH", "secs": 0,
                      "stall": "crashed", "steps": [], "game_overs": 0}
        finally:
            sys.stdout = real_stdout

        if result.get("error") and not result.get("steps"):
            say(f"  {gid:<6} ERROR: {result['error']}")
            report_games.append({"game": gid, "error": result["error"]})
            continue

        ana = analyse(result, meta)
        # Same denominator for both numbers: whatever EnvironmentInfo carries.
        baselines = result.get("baselines") or meta.get("baseline_steps") or []
        gs, per_level, scores = rhae(result, baselines)
        scores = merge_official(
            scores,
            board.game(result.get("env_game_id") or gid) if board.ok else None)
        llm = PROBE.snapshot_llm()
        llm.update(llm_conf)
        llm_effect(llm, result)
        try:
            import torch
            if torch.cuda.is_available() and llm["loaded"]:
                llm["vram_gb"] = torch.cuda.memory_allocated() / 1e9
        except Exception:
            pass
        findings = diagnose(result, ana, meta, llm, cfg)

        print_game_report(result, ana, per_level, gs, meta, llm, findings, capture,
                          cfg, scores)

        rows.append({"game": gid, "levels": result["levels"],
                     "win_levels": result["win_levels"], "actions": result["actions"],
                     "rhae": gs, "scores": scores,
                     "state": result["state"], "stall": result.get("stall"),
                     "secs": result.get("secs", 0), "analysis": ana,
                     "llm": llm, "findings": findings})
        slim = {k: v for k, v in result.items() if k not in ("steps", "level_events")}
        slim["actions_at_level"] = {str(k): v for k, v in result["actions_at_level"].items()}
        slim["level_events"] = result["level_events"].to_json()
        report_games.append({
            "result": slim, "rhae": gs, "scores": scores, "per_level": per_level,
            "analysis": {k: v for k, v in ana.items()},
            "llm": llm, "findings": [{"severity": s, "code": c, "text": t}
                                     for s, c, t in findings],
            "log_markers": dict(capture.markers),
            "log_notable": capture.notable[-40:],
            "log_tail": capture.tail(),
            "meta": meta,
        })
        write_report()

    print_summary(rows, cfg, started, board)

    report = write_report(final=True)
    say(f"\n  full report: {report_path}")
    return report


if __name__ == "__main__":
    main()
