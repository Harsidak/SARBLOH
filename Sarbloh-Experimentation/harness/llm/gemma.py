"""Gemma-4-31B-IT-NVFP4: everything Gemma-specific for the shared serving layer (see spec.py).

- The driessmit wheelhouse ships vLLM 0.19.0 with transformers 4.57.6, which does not know ``model_type: gemma4``.
  The ``sarbloh-wheels-gemma4`` overlay (transformers 5.5.0 + huggingface_hub 1.x) goes on top with --no-deps
  (vllm-project/vllm#39216). Verified by E003-E006: vLLM 0.19.0 runs with it.
- The gemma4 tool parser sometimes wraps a string argument in an extra pair of quotes (E006: ``"\\"explore\\""``),
  which the agent then sees as the literal text ``"explore"``. ``unquote`` repairs it.
"""

from __future__ import annotations

import json
from typing import Any

from harness.llm.spec import ModelSpec

WEIGHTS = "banwait13/sarblohmodels"  # Gemma-4-31B-IT-NVFP4, flat safetensors
BASE = {"--quantization": "modelopt", "--enable-prefix-caching": None, "--generation-config": "vllm"}
TOOLS = {"--enable-auto-tool-choice": None, "--tool-call-parser": "gemma4", "--reasoning-parser": "gemma4"}
TEXT_ONLY = {"--limit-mm-per-prompt": '{"image": 0, "audio": 0, "video": 0}'}
# Long context, fp8 KV (the checkpoint's hf_quant_config sets kv_cache_quant_algo FP8), text only. E006's profile.
FAST = {**BASE, **TOOLS, **TEXT_ONLY, "--max-model-len": "131072", "--kv-cache-dtype": "fp8", "--max-num-seqs": "32",
        "--max-num-batched-tokens": "8192", "--gpu-memory-utilization": "0.92"}

PROFILES: dict[str, dict[str, Any]] = {
    # E007 speed arm: prompt-lookup (n-gram) speculative decoding, since the agent copies grids and code from its own
    # context, plus bigger prefill chunks for the ~40k-token prompts. UNCONFIRMED until the E007 bench.
    "gemma_turbo": {"model_dataset": WEIGHTS, "env": {"OMP_NUM_THREADS": "1"}, "flags": {
        **FAST, "--max-num-batched-tokens": "16384",
        "--speculative-config": '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4, '
                                '"prompt_lookup_min": 2}'}},
    "gemma_fast": {"model_dataset": WEIGHTS, "env": {}, "flags": dict(FAST)},
    # Fewer knobs: default KV dtype, 64k context.
    "gemma_safe": {"model_dataset": WEIGHTS, "env": {}, "flags": {
        **BASE, **TOOLS, "--max-model-len": "65536", "--gpu-memory-utilization": "0.90"}},
    # No parsers at all: fenced-code mode, eager mode, short context. The last resort.
    "gemma_min": {"model_dataset": WEIGHTS, "env": {}, "flags": {
        "--quantization": "modelopt", "--max-model-len": "32768", "--enforce-eager": None,
        "--gpu-memory-utilization": "0.90"}},
}


def unquote(value: Any) -> Any:
    """``'"explore"'`` -> ``'explore'``, recursively through lists and dicts. Only a string that is itself exactly one
    JSON string literal is unwrapped, so real quotes inside text stay."""
    if isinstance(value, dict):
        return {k: unquote(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unquote(v) for v in value]
    if isinstance(value, str) and len(value) >= 2 and value[0] == value[-1] == '"':
        try:
            inner = json.loads(value)
        except ValueError:
            return value
        return inner if isinstance(inner, str) else value
    return value


SPEC = ModelSpec(
    name="gemma",
    served_model_name="gemma-4-31b-it",
    profiles=PROFILES,
    profile_chain=["gemma_fast", "gemma_safe", "gemma_min"],
    # Gemma 4 model card defaults: temperature 1.0, top_p 0.95, top_k 64; thinking on.
    llm={"temperature": 1.0, "top_p": 0.95, "top_k": 64, "chat_template_kwargs": {"enable_thinking": True}},
    overlay_dataset="banwait13/sarbloh-wheels-gemma4",
    fix_tool_arguments=unquote,
)
