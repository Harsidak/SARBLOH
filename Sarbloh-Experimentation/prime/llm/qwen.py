"""Qwen 3.8: everything Qwen-specific for the shared serving layer (see spec.py).

E008 (owner decision, 2026-10-01): the model is Qwen3.8-27B, not Flash-Next. The chain is
``qwen_fp8_vision`` -> ``qwen_fp8``: the official ``Qwen/Qwen3.8-27B-FP8`` (Kaggle snapshot ``FP8`` below) on the shared
vLLM 0.19 wheelhouse with prefix caching and a 65k window, first with one image per prompt (the agent is shown a PNG of
the game after every act), then text only if the image start or the image smoke test fails. The vision tower is in the
checkpoint's config (checked 2026-10-01); image serving on vLLM 0.19 FP8 is UNCONFIRMED until the first Kaggle start.

Flags: qwen3_coder tool parser, qwen3 reasoning parser. Sampling follows the Duck: thinking on, temperature 0.6 /
top_p 0.95 / top_k 20. E007 v3 measured ``qwen_fp8`` at 388 tok/s over 10 streams with a 0.63 prefix hit rate.

The Flash-Next profiles (Keith Tyser's pinned runtime, ``qwen4_exp`` MoE, E007) stay defined but are not in the chain:
E007 v4 killed ``qwen_flash_pc_64k`` (KV cache does not fit) and ``qwen_flash_pc`` (8.7% prefix hits), and the Flash-Next
stress test failed. The bundle licence is MIT in its SOURCE_IDENTITY and "unknown" on Kaggle: UNCONFIRMED.
The 27B NVFP4 (overseer66, an unsloth mixed quant) is gone: vLLM 0.19 has no ``lm_head.weight_scale`` for its FP8
lm_head (E007 v3). ``nvidia/Qwen3.8-27B-NVFP4`` is E008 arm D, once it is on Kaggle and a newer vLLM starts it.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable

from prime.llm.spec import ModelSpec

FLASH = "keithtyser/qwen3-8-flash-next-nvfp4"  # Kaggle model, PyTorch/radixark-modelopt-fp4/1
FLASH_BUNDLE = "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1"  # serving_setup.py + vllm-patches/
FLASH_RUNTIME = "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"  # the image layers
FP8 = "fumiyauchiyama/qwen3-8-27b-fp8-hf-snapshot"  # official Qwen/Qwen3.8-27B-FP8 rev 017b9c7a


def flash_next_runtime(working_dir: Path, find_input: Callable[[str], Path]) -> dict[str, Any]:
    """Unpack and patch the pinned runtime with the bundle's own checks (its "fast start" mode: identity files and
    sizes, not a re-hash of 8 GB of layers), and return the server environment it builds."""
    bundle = find_input(FLASH_BUNDLE)
    setup_dir = working_dir / "flash-next-setup"
    setup_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update({  # read at import by serving_setup.py
        "TAAF_KAGGLE_BUNDLE_DIR": str(bundle), "TAAF_KAGGLE_WORKING_DIR": str(setup_dir),
        "TAAF_KAGGLE_SETUP_ENV": str(setup_dir / "analyzer.env"),
        "TAAF_KAGGLE_INPUT_PATHS": json.dumps({FLASH_RUNTIME: str(find_input(FLASH_RUNTIME))})})
    loader = importlib.util.spec_from_file_location("flash_next_serving_setup", bundle / "serving_setup.py")
    ss = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(ss)
    ss.source_identity()  # serving_setup.py's own hash must match SOURCE_IDENTITY.json
    storage = ss.validate_runtime_storage()
    runtime = ss.verify_and_extract_runtime(ss.resolve_runtime_dir(), full_layer_hashes=False,
                                            scan_extracted_caches=False)
    patch = ss.patch_ple_layer()
    env, check = ss.runtime_environment(deep_preload_validation=False, tuning=ss.resolve_vllm_tuning())
    return {"env": env, "serve": ["-m", "vllm.entrypoints.cli.main", "serve"],
            "info": {"vllm": ss.VLLM_VERSION, "image": ss.VLLM_IMAGE, "extract_s": round(runtime["extract_seconds"]),
                     "patched_sha256": patch["patched_sha256"], "free_bytes": storage["roots"][0]["free_bytes"],
                     "python": check.get("python"), "tools": check.get("required_tools")}}


# The Duck/Keith command, flag for flag (serving_setup.py server_command with default tuning).
FLASH_FLAGS = {
    "--load-format": "safetensors", "--dtype": "bfloat16", "--quantization": "modelopt_fp4",
    "--tensor-parallel-size": "1", "--distributed-executor-backend": "mp", "--gpu-memory-utilization": "0.92",
    "--max-model-len": "32768", "--max-num-seqs": "28", "--max-num-batched-tokens": "8192",
    "--async-scheduling": None, "--enable-chunked-prefill": None, "--no-enable-prefix-caching": None,
    "--enable-auto-tool-choice": None, "--tool-call-parser": "qwen3_coder", "--reasoning-parser": "qwen3",
    "--chat-template": "{model_dir}/chat_template.jinja",
    "--speculative-config": '{"method":"mtp","num_speculative_tokens":3}',
    "--no-enable-log-requests": None, "--disable-uvicorn-access-log": None}
FLASH_PC = {**{k: v for k, v in FLASH_FLAGS.items() if k != "--no-enable-prefix-caching"},
            "--enable-prefix-caching": None}

BASE = {"--generation-config": "vllm", "--default-chat-template-kwargs": '{"preserve_thinking": true}',
        "--limit-mm-per-prompt": '{"image": 0, "video": 0}'}
TOOLS = {"--enable-auto-tool-choice": None, "--tool-call-parser": "qwen3_coder", "--reasoning-parser": "qwen3"}
ENV = {"OMP_NUM_THREADS": "1"}

PROFILES: dict[str, dict[str, Any]] = {
    # Prefix caching plus the 65k window our agent's compaction needs (trigger 40k, reserve 16k).
    "qwen_flash_pc_64k": {"model_dataset": FLASH, "runtime": "flash_next", "env": {},
                          "flags": {**FLASH_PC, "--max-model-len": "65536"}},
    # Prefix caching: the agent's prompts are long and append-only (E007 v3: 2.0 s vs 8.1 s per later turn).
    "qwen_flash_pc": {"model_dataset": FLASH, "runtime": "flash_next", "env": {}, "flags": FLASH_PC},
    # The Duck's exact serving recipe: the reference.
    "qwen_flash": {"model_dataset": FLASH, "runtime": "flash_next", "env": {}, "flags": FLASH_FLAGS},
    # Qwen3.8-27B FP8 on the shared vLLM 0.19 wheelhouse: measured fallback, not in the chain.
    "qwen_fp8_mtp": {"model_dataset": FP8, "env": ENV, "flags": {
        **BASE, **TOOLS, "--max-model-len": "131072", "--max-num-seqs": "32", "--max-num-batched-tokens": "16384",
        "--gpu-memory-utilization": "0.92", "--speculative-config": '{"method": "mtp", "num_speculative_tokens": 3}',
        "--no-enable-prefix-caching": None}},
    "qwen_fp8": {"model_dataset": FP8, "env": ENV, "flags": {
        **BASE, **TOOLS, "--enable-prefix-caching": None, "--max-model-len": "65536", "--max-num-seqs": "32",
        "--max-num-batched-tokens": "8192"}},
}
# E008: the same profile with one image per prompt (the agent keeps only the newest game image in its context).
PROFILES["qwen_fp8_vision"] = {**PROFILES["qwen_fp8"], "flags": {
    **PROFILES["qwen_fp8"]["flags"], "--limit-mm-per-prompt": '{"image": 1, "video": 0}'}}

SPEC = ModelSpec(
    name="qwen",
    served_model_name="qwen3.8",
    profiles=PROFILES,
    # E008: Qwen3.8-27B-FP8 with images, then the same without (a failed image start or image smoke test).
    profile_chain=["qwen_fp8_vision", "qwen_fp8"],
    llm={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": True}},
    smoke_template_kwargs={"enable_thinking": False},  # the Duck's smoke test: thinking could exhaust max_tokens
    runtimes={"flash_next": flash_next_runtime},
)
