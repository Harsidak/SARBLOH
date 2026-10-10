"""Every knob of a Prime Agent run. The notebook passes overrides; the merged config is printed and hashed."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

# Local trace collection for LoRA SFT: `python -m harness.run --local` plays the games with the DeepSeek API
# (harness/llm/deepseek.py) instead of a local server, and saves each run under runs/sft_deepseek/<time>/.
# Never used on Kaggle (no internet there): main() ignores it.
local_SFT_RUN = True

DEFAULT: dict[str, Any] = {
    "experiment": "sarbloh_experimentation",   # the ledger label; the notebook sets the experiment directory name
    "games": None,                     # None = every environment; else ids or id prefixes (offline only)
    "concurrency": 8,                  # games played at once (each is a root session with its own kernel)
    "notebook_budget_s": 3600.0,       # whole notebook wall clock, measured from its first cell
    "teardown_reserve_s": 300.0,       # games are stopped this long before the budget ends
    "game_wall_s": 1800.0,             # cap per game
    "max_actions_per_game": 2000,       # hard action budget per game (host-enforced)
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
    "scheduler": {"enabled": True, "slots": 10, "quantum_calls": 10, "token_scale": 40000.0},
    # --- agent: tools ipython (`observe()` reads, `await act([...])` moves) and recall (harness/agent/tools.py)
    "agent": {
        "act_max_actions": 100,        # moves per act() call: a whole searched path in one call (max_actions caps the game)
        "allow_fenced_code": True,     # run ```python blocks as ipython when no native tool call came back
        # The most a reply may write, thinking and code together: "early" on levels 1 and 2, "later" after, or one
        # int for every level. Room for a world model of several hundred lines; the prompt asks for short thinking.
        "max_tokens_per_turn": 8192,
        # The Qwen3.8 chat template's reasoning effort: "xhigh" (its default, a "think carefully" line at the top of
        # the system prompt), "medium" (no line) or "low"; there is no "high". "early" is used on levels 1 and 2
        # (prompts.EARLY_LEVELS), "later" after. None sends nothing (the template's default).
        "reasoning_effort": {"early": "xhigh", "later": "xhigh"},
        "request_timeout_s": 900.0,
        "cell_timeout_s": 300.0,
        "tool_output_chars": 5000,     # upstream: 65536 per stream; ours is cut to fit small context windows
        "context_window": 131072,      # served context; run.py sets it from the server profile / local server
        # No summary: the context is kept under cap_tokens by draining old messages in code (no model call). Above
        # trigger_tokens (or window - reserve, if lower) a drain keeps the newest keep_recent_tokens word for word and
        # shrinks everything before them; if that leaves more than drain_target_tokens, the word-for-word tail is
        # halved step by step (down to 2000 tokens), then the history is trimmed back to the newest full state.
        # trigger_tokens sits 20k below cap_tokens: one turn (8k reply, cell output, new board, picture) fits in that.
        "compaction": {"reserve_tokens": 8192, "cap_tokens": 120000, "trigger_tokens": 100000,
                       "keep_recent_tokens": 30000, "drain_target_tokens": 60000},
        # The prompt only grows between rewrites, so the server's prefix cache keeps it (harness/agent/context.py):
        # every thinking and every picture stay, and the memory is a stored message, in full after a rewrite and
        # every memory_full_every states, else only its changed sections. A drain: before the word-for-word tail
        # every board becomes a one-line stub, every ipython output is cut to drain_output_chars, every act result
        # becomes a short move log (one cut line per step), and the thinking, the pictures and the memory messages
        # go (a full memory goes back).
        "append_only": True,
        "memory_full_every": 10,
        "drain_output_chars": 400,
        # The thinking sent back in the current tool-call chain, newest reply first, up to this many tokens (chars/4);
        # older replies keep their tool calls and outputs but lose their thinking. Each turn still thinks freely.
        "chain_reasoning_tokens": 4000,
        # Experiment: only the newest ipython call's code is sent in full. When a new reply comes, the code of each
        # older call longer than stub_code_min_lines becomes a short note naming what it defined (the kernel still
        # holds it; `show_source("name")` prints it), and the system prompt says so. The transcript keeps the full code.
        # Costs one turn of re-prefill per request (the reply before the newest one changes). False = as before.
        "stub_old_code": True,
        "stub_code_min_lines": 20,
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
            "briefing_lines": 20,      # most briefing lines shown (the rest: `observe().briefing` in ipython)
            "crop_margin": 3,          # cells around the changed region
            "max_objects": 35,         # object rows shown per observation (the rest: `observe().objects`)
            "observation_tokens": 600,  # cap on the short view's text (chars / 4)
        },
        "memory": {
            "lessons": True,           # the cross-game lessons graph
            "curator": True,           # hidden curator: open questions, lessons, skills (2 to 5 at every level-up)
            "curator_every_turns": 35,  # also at every level-up (before its memory is cleared)
            "curator_max_tokens": 1536,
            "recall_tokens": 1000,
            # context blocks, in tokens (chars / 4); a block over its cap is trimmed oldest first
            "caps": {"goal": 150, "plan": 200, "skills": 800, "hypotheses": 400, "findings": 300, "lessons": 400,
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
    # --- DeepSeek API (only when local_SFT_RUN is on; key from $DEEPSEEK_API_KEY or the repo .env) -------------------
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-reasoner", "send_reasoning": True},
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
