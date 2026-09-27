"""Every knob of a Prime Agent run. The notebook passes overrides; the merged config is printed and hashed."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

DEFAULT: dict[str, Any] = {
    "experiment": "E003_prime_harness_smoke",
    "games": None,                     # None = every environment; else ids or id prefixes (offline only)
    "concurrency": 8,                  # games played at once (each is a root session with its own kernel)
    "notebook_budget_s": 3600.0,       # whole notebook wall clock, measured from its first cell
    "teardown_reserve_s": 300.0,       # games are stopped this long before the budget ends
    "game_wall_s": 1800.0,             # cap per game
    "max_actions_per_game": 400,       # hard action budget per game (host-enforced)
    "max_actions_per_cell": None,      # E004: host refuses arc.step beyond this many in one cell (None = off)
    # --- agent ---------------------------------------------------------------------------------------------
    "agent": {
        "tool_mode": "native",         # native (ipython tool) | fenced (```python blocks); vLLM start decides
        "allow_fenced_code": True,     # also execute fenced code when no native call came back
        "max_tokens_per_turn": 16384,
        "request_timeout_s": 900.0,
        "cell_timeout_s": 300.0,
        "tool_output_chars": 6000,     # upstream: 65536 per stream; ours is cut to fit small context windows
        "context_window": 131072,      # served context; run.py sets it from the vLLM profile / local server
        # upstream DEFAULT_COMPACTION_SETTINGS: compact when context > window - reserve; keep the newest 20k tokens
        "compaction": {"reserve_tokens": 16384, "keep_recent_tokens": 20000},
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
    "llm": {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "gemma-4-31b-it",
        "temperature": 1.0,            # Gemma 4 model card defaults: temperature 1.0, top_p 0.95, top_k 64
        "top_p": 0.95,
        "top_k": 64,
        "chat_template_kwargs": {"enable_thinking": True},
        "request_timeout_s": 900.0,
        "retries": 4,
    },
    # --- vLLM on Kaggle ------------------------------------------------------------------------------------
    "vllm": {
        "model_dataset": "banwait13/sarblohmodels",                  # Gemma-4-31B-IT-NVFP4, flat safetensors
        "wheelhouse_dataset": "driessmit1/arc3-vllm-h100-wheelhouse-v3",  # vLLM 0.19.0, torch 2.10
        "overlay_dataset": "banwait13/sarbloh-wheels-gemma4",         # transformers 5.5.0 overlay
        "served_model_name": "gemma-4-31b-it",
        "port": 8000,
        "profile_chain": ["gemma_fast", "gemma_safe", "gemma_min"],
        "startup_timeout_s": 1500.0,
        "bench": True,
        "watchdog": True,
    },
}


def merge(base: dict[str, Any], over: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
