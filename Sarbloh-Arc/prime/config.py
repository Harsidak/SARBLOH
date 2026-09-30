"""Every knob of a Prime Agent run. The notebook passes overrides; the merged config is printed and hashed."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

DEFAULT: dict[str, Any] = {
    "experiment": "E006_dedicated_tools",
    "games": None,                     # None = every environment; else ids or id prefixes (offline only)
    "concurrency": 8,                  # games played at once (each is a root session with its own kernel)
    "notebook_budget_s": 3600.0,       # whole notebook wall clock, measured from its first cell
    "teardown_reserve_s": 300.0,       # games are stopped this long before the budget ends
    "game_wall_s": 1800.0,             # cap per game
    "max_actions_per_game": 400,       # hard action budget per game (host-enforced)
    "max_actions_per_cell": None,      # E004: host refuses arc.step beyond this many in one cell (None = off)
    "stop_after_levels": None,         # end each game after this many levels (local tests); None = play them all
    # ARC SDK recording (Arcade.make(save_recording=True)): one JSONL line per env step, with frames and the agent's
    # step ref (turn, tool call, and on a cell's first action its thought and code). Off on the competition rerun.
    "record": True,
    # --- agent ---------------------------------------------------------------------------------------------
    "agent": {
        "tool_mode": "native",         # native (ipython tool) | fenced (```python blocks); vLLM start decides
        # E006: "dedicated" = ipython + plan/act/reset_level/remember/recall/delegate/message (prime.agent.tools);
        # "ipython" = upstream's single REPL tool (E003-E005). Fenced mode always uses "ipython".
        "toolset": "dedicated",
        "act_max_actions": 5,          # actions per act call
        "act_halt_on_mispredict": True,  # stop an act batch at the first step the registered world model gets wrong
        "replan_every_actions": 15,    # planner nudge after this many actions without a plan update
        "allow_fenced_code": True,     # also execute fenced code when no native call came back
        "max_tokens_per_turn": 16384,
        "request_timeout_s": 900.0,
        "cell_timeout_s": 300.0,
        "tool_output_chars": 6000,     # upstream: 65536 per stream; ours is cut to fit small context windows
        "context_window": 131072,      # served context; run.py sets it from the vLLM profile / local server
        # upstream DEFAULT_COMPACTION_SETTINGS: compact when context > window - reserve; keep the newest 20k tokens.
        # Ours (E005): also compact above trigger_tokens and keep 16k, because upstream's defaults are tuned for
        # 200k-context frontier models. trigger_tokens must be >= 2x keep_recent_tokens; None = upstream only.
        "compaction": {"reserve_tokens": 16384, "keep_recent_tokens": 16000, "trigger_tokens": 40000},
        "max_depth": 1,                # root may spawn children; children may not
        "max_running_children": 2,
        "subagent_keepalive_s": 300.0,
        "max_consecutive_llm_failures": 6,
        "reflect_every_actions": None,  # E004: forced harness write every N actions / level-up / GAME_OVER (None = off)
        "reflect_max_reasks": 2,
        # host-driven Continual Harness refinement; upstream defaults: on, every 25 turns + after compaction, 20 min
        # cooldown
        "auto_refine": {"enabled": True, "turn_interval": 25, "compact": True, "cooldown_s": 1200.0,
                        "max_tokens": 4096},
        "limits": {"max_turns": 400, "max_output_tokens": 3_000_000},
        "child_limits": {"max_turns": 60, "max_output_tokens": 400_000, "wall_s": 900.0},
    },
    # --- model ---------------------------------------------------------------------------------------------
    # "gemma" | "qwen". Picks prime/llm/gemma.py or prime/llm/qwen.py: weights, vLLM profiles, parsers, sampling and
    # thinking policy all come from there (build_config merges them under the overrides below).
    "model": "gemma",
    "llm": {
        "base_url": "http://127.0.0.1:8000/v1",
        "request_timeout_s": 900.0,    # whole budget per call, retries and waiting for a restart included (E007)
        "retries": 4,
    },
    # --- vLLM on Kaggle ------------------------------------------------------------------------------------
    "vllm": {
        "wheelhouse_dataset": "driessmit1/arc3-vllm-h100-wheelhouse-v3",  # vLLM 0.19.0, torch 2.10
        "port": 8000,
        "profile_chain": None,         # None = the model spec's chain
        "startup_timeout_s": 1500.0,
        "max_inflight": 32,            # requests admitted at once; ramps 2 -> 32 after a watchdog restart
        "bench": True,
        "watchdog": True,
        "watchdog_cfg": {},            # overrides of prime.llm.vllm.WATCHDOG
    },
}


def merge(base: dict[str, Any], over: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def build_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """DEFAULT, then the chosen model's spec defaults (served name, sampling, thinking), then ``overrides``."""
    from prime.llm.spec import get_spec

    spec = get_spec((overrides or {}).get("model", DEFAULT["model"]))
    base = merge(DEFAULT, {"llm": {"model": spec.served_model_name, **spec.llm}})
    return merge(base, overrides)


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
