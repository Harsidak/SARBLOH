"""Every knob of a Prime Agent run. The notebook passes overrides; the merged config is printed and hashed."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

DEFAULT: dict[str, Any] = {
    "experiment": "sarbloh_experimentation",   # the ledger label; the notebook sets the experiment directory name
    "games": None,                     # None = every environment; else ids or id prefixes (offline only)
    "concurrency": 8,                  # games played at once (each is a root session with its own kernel)
    "notebook_budget_s": 3600.0,       # whole notebook wall clock, measured from its first cell
    "teardown_reserve_s": 300.0,       # games are stopped this long before the budget ends
    "game_wall_s": 1800.0,             # cap per game
    "max_actions_per_game": 400,       # hard action budget per game (host-enforced)
    "stop_after_levels": None,         # end each game after this many levels (local tests); None = play them all
    # ARC SDK recording (Arcade.make(save_recording=True)): one JSONL line per env step, with frames and the agent's
    # step ref (turn, tool call, and on an act's first action its thought and arguments). Off on the competition rerun.
    "record": True,
    # The trace for analysis and LoRA SFT: each game's transcript.jsonl (with a snapshot of the context after every
    # rewrite, the reasoning effort and a digest of every request), the pictures sent (games/<id>/frames/), and after
    # the run harness/trace.py (reports + sft_levels.jsonl). Logging only: the agent's context and requests are the same
    # either way. Off: none of it is written. Always off on the competition rerun (its outputs are never seen).
    "tracing": True,
    # Priority scheduler (harness/scheduler.py). Off: `concurrency` games at a time, each for game_wall_s.
    # On: every game starts at once and only `slots` of them hold the GPU; a game keeps its slot for `quantum_calls`
    # LLM calls, then the waiting game with the best (next-level weight x hope) gets it. Set game_wall_s to the budget.
    "scheduler": {"enabled": False, "slots": 6, "quantum_calls": 4, "token_scale": 80000.0},
    # --- agent: tools ipython (read-only `observe()`), act and recall (harness/agent/tools.py) -------------
    "agent": {
        "act_max_actions": 5,          # actions per act call
        "allow_fenced_code": True,     # run ```python blocks as ipython when no native tool call came back
        "max_tokens_per_turn": 16384,
        # The Qwen3.8 chat template's reasoning effort: "xhigh" (its default, a "think carefully" line at the top of
        # the system prompt), "medium" (no line) or "low"; there is no "high". "early" is used on levels 1 and 2
        # (prompts.REASON_EARLY_LEVELS), "later" after. None sends nothing (the template's default).
        "reasoning_effort": {"early": "xhigh", "later": "medium"},
        "request_timeout_s": 900.0,
        "cell_timeout_s": 300.0,
        "tool_output_chars": 6000,     # upstream: 65536 per stream; ours is cut to fit small context windows
        "context_window": 131072,      # served context; run.py sets it from the server profile / local server
        # upstream DEFAULT_COMPACTION_SETTINGS: compact when context > window - reserve; keep the newest 20k tokens.
        # Ours: also compact above trigger_tokens and keep 16k, because upstream's defaults are tuned for
        # 200k-context frontier models. trigger_tokens must be >= 2x keep_recent_tokens; None = upstream only.
        # After a compaction the first prompt's size is the floor: no new compaction until the context passes
        # floor + floor_gap_tokens. The summary is at most summary_tokens long and replaces the previous one.
        "compaction": {"reserve_tokens": 16384, "keep_recent_tokens": 16000, "trigger_tokens": 120000,
                       "floor_gap_tokens": 8192, "summary_tokens": 2048},
        # The prompt only grows between rewrites, so the server's prefix cache keeps it (harness/agent/context.py):
        # every thinking and every picture stay, and the memory is a stored message, in full after a rewrite and
        # every memory_full_every states, else only its changed sections. At the compaction trigger a drain (drop the
        # thinking before the current chain, the old pictures and the memory messages; no model call) runs first; a
        # summary is made only when the drain would leave more than drain_max_fraction of the trigger.
        # False: the old builder (thinking only in the current chain, newest picture, memory moved to every state).
        "append_only": True,
        "memory_full_every": 10,
        "drain_max_fraction": 0.7,
        # The thinking sent back in the current tool-call chain, newest reply first, up to this many tokens (chars/4);
        # older replies keep their tool calls and outputs but lose their thinking. Each turn still thinks freely.
        "chain_reasoning_tokens": 16000,
        "max_consecutive_llm_failures": 6,
        "limits": {"max_turns": 400, "max_output_tokens": 3_000_000},
        "vision": False,               # set by run.py: True when the served profile passed the image smoke test
        # harness/agent/perception.py: the full view (briefing, objects, whole board, picture) at a level start,
        # after a compaction and on `observe()`; after an act the short view (changed region) and the picture.
        "perception": {
            "image": True,             # the picture with every observation (only if "vision" is True)
            "cell": 10,                # picture pixels per grid cell: 64x64 -> 640x640 map, plus the HUD panel
            "ascii": True,             # letter board: whole in the full view, changed region after an act
            "segmentation": True,      # object list with ids, hashes, containment, adjacency
            "briefing": True,          # briefing: MEASURED facts and GUESSES, in the full view
            "briefing_lines": 30,      # most briefing lines shown (the rest: `observe().briefing` in ipython)
            "crop_margin": 3,          # cells around the changed region
            "max_objects": 40,         # object rows shown per observation (the rest: `observe().objects`)
            "observation_tokens": 800,  # cap on the short view's text (chars / 4)
        },
        "memory": {
            "lessons": True,           # the cross-game lessons graph
            "curator": True,           # hidden curator: open questions, lessons, skills (2 to 5 at every level-up)
            "curator_every_turns": 25,  # also at every level-up (before its memory is cleared) and after compactions
            "curator_max_tokens": 4096,
            "recall_tokens": 1500,
            # context blocks, in tokens (chars / 4); a block over its cap is trimmed oldest first
            "caps": {"goal": 150, "plan": 200, "skills": 600, "hypotheses": 400, "findings": 300, "lessons": 400,
                     "questions": 100},
        },
    },
    # --- model ---------------------------------------------------------------------------------------------
    # "gemma" | "qwen" | "qwen_flash_next". Picks harness/llm/<name>.py: weights, server profiles, parsers, sampling and
    # thinking policy all come from there (build_config merges them under the overrides below).
    "model": "qwen",
    "llm": {
        "base_url": "http://127.0.0.1:8000/v1",
        "request_timeout_s": 900.0,    # whole budget per call, retries and waiting for a restart included
        "retries": 4,
    },
    # --- model server on Kaggle ----------------------------------------------------------------------------
    "server": {
        "wheelhouse_dataset": "banwait13/sarbloh-vllm-wheelhouse",  # kaggle/wheels.ipynb, vLLM 0.19.0
        "port": 8000,
        "profile_chain": None,         # None = the model spec's chain
        "startup_timeout_s": 1500.0,
        "max_inflight": 32,            # requests admitted at once; ramps 2 -> 32 after a watchdog restart
        "bench": True,
        "watchdog": True,
        "watchdog_cfg": {},            # overrides of harness.llm.server.WATCHDOG
    },
}


def merge(base: dict[str, Any], over: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def build_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """DEFAULT, then the chosen model's spec defaults (served name, sampling, thinking), then ``overrides``."""
    from harness.llm.spec import get_spec

    if "vllm" in (overrides or {}):  # renamed 2026-10-02; merge() would keep it as a dead key and drop its profile_chain
        raise ValueError('config key "vllm" was renamed to "server"')
    spec = get_spec((overrides or {}).get("model", DEFAULT["model"]))
    base = merge(DEFAULT, {"llm": {"model": spec.served_model_name, **spec.llm}})
    return merge(base, overrides)


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
