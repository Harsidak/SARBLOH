"""Qwen3.8-27B: everything Qwen-specific for the shared serving layer (see spec.py).

Flags and sampling follow the Duck (Tufa Labs, Milestone #1 winner, Qwen3.6-27B-FP8 on this same vLLM 0.19.0 wheelhouse):
qwen3_coder tool parser, qwen3 reasoning parser, preserve_thinking, thinking on, temperature 0.6 / top_p 0.95 / top_k 20.
Qwen3.8 is the same ``Qwen3_5ForConditionalGeneration`` architecture (hybrid: 3 of 4 layers are linear attention),
so no transformers overlay is needed. UNCONFIRMED until the E007 bench: that vLLM 0.19.0 loads both checkpoints below.

Weights (E007 hypothesis.md has the full reasoning):
- NVFP4: Kaggle model ``overseer66/qwen3-8-27b-nvfp4`` (community ModelOpt quant, bf16 MTP head in model_mtp.safetensors).
- FP8: ``fumiyauchiyama/qwen3-8-27b-fp8-hf-snapshot``, the unmodified official Qwen/Qwen3.8-27B-FP8 (rev 017b9c7a),
  MTP head in mtp.safetensors.
Quantisation is read from each checkpoint's config (no --quantization flag), so one flag set serves both.

DuckQwen's measured winner turned prefix caching off when MTP was on; ``qwen_nvfp4_mtp_pc`` tests whether both work
together on this build, because our agent's prompts are long and append-only.
"""

from __future__ import annotations

from typing import Any

from prime.llm.spec import ModelSpec

NVFP4 = "overseer66/qwen3-8-27b-nvfp4"
FP8 = "fumiyauchiyama/qwen3-8-27b-fp8-hf-snapshot"
BASE = {"--generation-config": "vllm", "--default-chat-template-kwargs": '{"preserve_thinking": true}',
        "--limit-mm-per-prompt": '{"image": 0, "video": 0}'}
TOOLS = {"--enable-auto-tool-choice": None, "--tool-call-parser": "qwen3_coder", "--reasoning-parser": "qwen3"}
BATCH = {"--max-model-len": "131072", "--max-num-seqs": "32", "--max-num-batched-tokens": "16384",
         "--gpu-memory-utilization": "0.92"}
MTP = {"--speculative-config": '{"method": "mtp", "num_speculative_tokens": 3}'}
ENV = {"OMP_NUM_THREADS": "1"}

PROFILES: dict[str, dict[str, Any]] = {
    "qwen_nvfp4_mtp_pc": {"model_dataset": NVFP4, "env": ENV, "flags": {
        **BASE, **TOOLS, **BATCH, **MTP, "--enable-prefix-caching": None}},
    "qwen_nvfp4_mtp": {"model_dataset": NVFP4, "env": ENV, "flags": {
        **BASE, **TOOLS, **BATCH, **MTP, "--no-enable-prefix-caching": None}},
    "qwen_fp8_mtp": {"model_dataset": FP8, "env": ENV, "flags": {
        **BASE, **TOOLS, **BATCH, **MTP, "--no-enable-prefix-caching": None}},
    # The Duck's own flags (65k, prefix caching, no speculation) on the official FP8 weights: the reference.
    "qwen_fp8": {"model_dataset": FP8, "env": ENV, "flags": {
        **BASE, **TOOLS, "--enable-prefix-caching": None, "--max-model-len": "65536", "--max-num-seqs": "32",
        "--max-num-batched-tokens": "8192"}},
    # No parsers: fenced-code mode, eager, short context. The last resort.
    "qwen_min": {"model_dataset": FP8, "env": ENV, "flags": {
        "--generation-config": "vllm", "--max-model-len": "32768", "--enforce-eager": None,
        "--gpu-memory-utilization": "0.90"}},
}

SPEC = ModelSpec(
    name="qwen",
    served_model_name="qwen3.8-27b",
    profiles=PROFILES,
    profile_chain=["qwen_nvfp4_mtp", "qwen_fp8_mtp", "qwen_fp8", "qwen_min"],
    llm={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": True}},
    smoke_template_kwargs={"enable_thinking": False},  # the Duck's smoke test: thinking could exhaust max_tokens
)
