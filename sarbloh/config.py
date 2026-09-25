"""One CONFIG for a run: agent, LLM client, vLLM server, and harness knobs.

The Kaggle notebook and the local runner both start from ``default_config()`` and override keys. Every value
that changes behaviour lives here, so a run is fully described by its config dict (hashed into the ledger).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any

# The Duck's exact client settings, from its Kaggle setup command (bundle 2026-06-12).
_DUCK_LLM = {
    "base_url": "http://127.0.0.1:1234/v1",
    "model_id": "vrfai/Qwen3.6-27B-FP8",  # the served model name vLLM advertises
    "provider": "vllm",
    "context_window": 32768,
    "max_output": 0,  # 0 = no cap
    "tool_steps": 0,  # 0 = unlimited tool calls per analysis step
    "tool_timeout": 30,
    "tool_output_tokens": 1024,
    "yield_seconds": 60,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "thinking": True,
    "multimodal_context": "current_grid",
    "multimodal_upscale": 4,
}

# vLLM server profiles. "duck" is Tufa's exact server; "fast" adds the DuckQwen profile knobs
# (kv5-bf16-mtp3-c8-cg32). Whether "fast" helps THIS model is UNCONFIRMED until a run measures it.
VLLM_PROFILES: dict[str, dict[str, Any]] = {
    "duck": {
        "max_model_len": 65536,
        "tensor_parallel_size": 1,
        "enable_prefix_caching": True,
        "extra_args": [],
        "env": {},
    },
    "fast": {
        "max_model_len": 65536,
        "tensor_parallel_size": 1,
        "enable_prefix_caching": False,
        "speculative_config": {"method": "mtp", "num_speculative_tokens": 3},
        "max_num_batched_tokens": 8192,
        "max_cudagraph_capture_size": 32,
        # DuckQwen also set max_num_seqs=8 and a 5 GiB KV cache for a 4-bit model; left off until measured.
        "extra_args": [],
        "env": {"OMP_NUM_THREADS": "1"},
    },
}


def default_config() -> dict[str, Any]:
    return {
        "label": "sarbloh",
        "agent": "duck",  # which agent plays: "duck" (B000). New agents are added in sarbloh.agent.
        # --- harness / budget (the Duck's Kaggle settings) ---
        "concurrency": 28,
        "max_runtime_s_per_game": 7920.0,
        "analyzer_timeout": 900.0,
        "max_actions_per_game": None,
        "notebook_budget_s": 32400.0,  # 9 h Kaggle run; the teardown reserve comes off this
        "teardown_reserve_s": 600.0,
        "games": None,  # None = every game the Arcade offers; else a list of game ids or id prefixes
        "save_request_logs": False,
        # --- LLM client ---
        "llm": copy.deepcopy(_DUCK_LLM),
        # --- vLLM server (Kaggle only) ---
        "vllm": {
            "profile": "fast",
            "fallback_profile": "duck",  # restart with this if the profile fails to come up
            "port": 1234,
            "served_model_name": _DUCK_LLM["model_id"],
            "model_dataset": "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
            "wheelhouse_dataset": "driessmit1/arc3-vllm-h100-wheelhouse-v3",
            "wheelhouse_stamp": "vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6\n",
            "tool_call_parser": "qwen3_coder",
            "reasoning_parser": "qwen3",
            "startup_timeout_s": 1500,
            "watchdog": True,
        },
    }


def merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Deep-merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def apply_llm_env(config: dict[str, Any]) -> None:
    """Export the LLM settings as the ``LOCAL_ANALYZER_*`` variables the Duck reads at import time."""
    llm = config["llm"]
    env = {
        "LOCAL_ANALYZER_BASE_URL": llm["base_url"],
        "OPENAI_BASE_URL": llm["base_url"],
        "LOCAL_ANALYZER_PROVIDER": llm["provider"],
        "OPENAI_PROVIDER": llm["provider"],
        "LOCAL_ANALYZER_MODEL_ID": llm["model_id"],
        "INFERENCE_ANALYZER_MODEL": llm["model_id"],
        "LOCAL_ANALYZER_APP_NAME": "SARBLOH",
        "LOCAL_ANALYZER_CONTEXT_WINDOW": str(llm["context_window"]),
        "LOCAL_ANALYZER_MAX_OUTPUT": str(llm["max_output"]),
        "LOCAL_ANALYZER_TOOL_STEPS": str(llm["tool_steps"]),
        "LOCAL_ANALYZER_TOOL_TIMEOUT": str(llm["tool_timeout"]),
        "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": str(llm["tool_output_tokens"]),
        "LOCAL_ANALYZER_YIELD_SECONDS": str(llm["yield_seconds"]),
        "LOCAL_ANALYZER_TEMPERATURE": str(llm["temperature"]),
        "LOCAL_ANALYZER_TOP_P": str(llm["top_p"]),
        "LOCAL_ANALYZER_TOP_K": str(llm["top_k"]),
        "LOCAL_ANALYZER_ENABLE_THINKING": "true" if llm["thinking"] else "false",
        "MULTIMODAL_CONTEXT": llm["multimodal_context"],
        "MULTIMODAL_UPSCALE": str(llm["multimodal_upscale"]),
    }
    os.environ.update(env)
