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
lm_head (E007 v3). ``nvidia/Qwen3.8-27B-NVFP4`` runs on SGLang instead (E112, ``SGLANG_CHAIN`` below).
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable

from prime.llm import sglang
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

# E112 (option B, owner go 2026-10-02): nvidia/Qwen3.8-27B-NVFP4 on SGLang 0.5.19 "Pennyroyal" (the milestone-2
# reference's RTX Pro 6000 build). The checkpoint is ModelOpt MIXED_PRECISION: MLP + lm_head NVFP4 (group 16),
# self-attention and Gated-DeltaNet projections FP8, the MTP layer (mtp.*) unquantized, so NEXTN speculation runs on
# the model's own MTP head. SGLang detects the format from config.json (quant_method "modelopt" -> "modelopt_mixed"),
# so no --quantization flag. Kaggle: banwait13/qwencoder = the HF snapshot (shard sizes equal our SHA256-verified local
# download of 2026-10-01). Wheels: banwait13/sglangwheels (flat copy of the Pennyroyal v2.5.3 wheelhouse).
# Flags follow the reference launcher (CFG in kaggle/reference/arc-agi-3-milestone-2-solution.ipynb, cell 12) minus
# its Flash-Next-only ones (AutoRound, external draft, PLE offload, MoE runner). E112 measured 1162 tok/s with MTP5
# (10 streams, thinking on). FR-Spec (--speculative-token-map) is not used: it crashes with the NVFP4 lm_head.
NVFP4 = "banwait13/qwencoder"
SGLANG_WHEELS = "banwait13/sglangwheels"
SGL_NV = {
    "--load-format": "safetensors", "--model-loader-extra-config": '{"enable_multithread_load":false}',
    "--weight-loader-prefetch-checkpoints": None,
    "--tensor-parallel-size": "1", "--dtype": "bfloat16", "--trust-remote-code": None,
    "--kv-cache-dtype": "fp8_e4m3", "--mem-fraction-static": "0.92", "--context-length": "65536", "--page-size": "64",
    "--max-running-requests": "10", "--chunked-prefill-size": "8192", "--max-prefill-tokens": "16384",
    "--cuda-graph-max-bs-decode": "10", "--cuda-graph-bs-decode": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    "--mamba-ssm-dtype": "bfloat16", "--max-mamba-cache-size": "60", "--mamba-radix-cache-strategy": "extra_buffer",
    "--mamba-track-interval": "64", "--mamba-backend": "flashinfer", "--linear-attn-decode-backend": "flashinfer",
    "--linear-attn-prefill-backend": "flashinfer",
    "--mm-feature-transport": "cpu", "--image-processor-backend": "pil",
    "--reasoning-parser": "qwen3", "--tool-call-parser": "qwen3_coder",
    "--chat-template": "{model_dir}/chat_template.jinja",
    "--default-chat-template-kwargs": '{"preserve_thinking": true}',
    "--watchdog-timeout": "1800", "--schedule-policy": "lpm", "--enable-cache-report": None, "--enable-metrics": None}
# NEXTN on the built-in MTP layer: linear draft chain (topk 1, required by --gdn-mtp-cache-mode none), strict accept.
SGL_NV_MTP = {"--speculative-algorithm": "NEXTN", "--speculative-num-steps": "3", "--speculative-eagle-topk": "1",
              "--speculative-num-draft-tokens": "4", "--gdn-mtp-cache-mode": "none",
              "--speculative-accept-threshold-single": "1.0", "--speculative-accept-threshold-acc": "1.0"}
# Five draft steps instead of three: the E112 bench winner.
SGL_NV_MTP5 = {**SGL_NV_MTP, "--speculative-num-steps": "5", "--speculative-num-draft-tokens": "6"}


def _nv(vision: bool, *extra: dict[str, Any]) -> dict[str, Any]:
    flags: dict[str, Any] = dict(SGL_NV)
    for e in extra:
        flags.update(e)
    return {"model_dataset": NVFP4, "runtime": "sglang", "vision": vision, "env": {}, "flags": flags}


PROFILES.update({
    "sglang_nvfp4_mtp5_vision": _nv(True, SGL_NV_MTP5),
    "sglang_nvfp4_mtp_vision": _nv(True, SGL_NV_MTP),
    "sglang_nvfp4_vision": _nv(True),
    "sglang_nvfp4": _nv(False),
})
# E112 chain: NVFP4 + MTP5 + image, then MTP3, then no MTP, then text only, then the measured vLLM FP8 fallback.
SGLANG_CHAIN = ["sglang_nvfp4_mtp5_vision", "sglang_nvfp4_mtp_vision", "sglang_nvfp4_vision", "sglang_nvfp4",
                "qwen_fp8_vision", "qwen_fp8"]

SPEC = ModelSpec(
    name="qwen",
    served_model_name="qwen3.8",
    profiles=PROFILES,
    # E008: Qwen3.8-27B-FP8 with images, then the same without (a failed image start or image smoke test).
    profile_chain=["qwen_fp8_vision", "qwen_fp8"],
    llm={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": True}},
    smoke_template_kwargs={"enable_thinking": False},  # the Duck's smoke test: thinking could exhaust max_tokens
    runtimes={"flash_next": flash_next_runtime,
              "sglang": sglang.runtime(SGLANG_WHEELS, precache_datasets=(NVFP4,))},
)
