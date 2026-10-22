"""Qwen3.8-Flash-Next on SGLang with MTP speculation (config ``model: "qwen_flash_next"``).

A copy of qwen.py (owner decision 2026-10-04: qwen.py stays as it is for the 27B) with the Flash-Next SGLang
profiles added at the end and the chain starting on them; the 27B NVFP4 chain is the fallback. Everything above the
"Flash-Next on SGLang" section is qwen.py unchanged.

qwen.py's notes, as of 2026-10-01: the model is Qwen3.8-27B, not Flash-Next. The chain is
``qwen_fp8_vision`` -> ``qwen_fp8``: the official ``Qwen/Qwen3.8-27B-FP8`` (Kaggle snapshot ``FP8`` below) on the shared
vLLM 0.19 wheelhouse with prefix caching and a 65k window, first with one image per prompt (the agent is shown a PNG of
the game after every act), then text only if the image start or the image smoke test fails. The vision tower is in the
checkpoint's config (checked 2026-10-01); image serving on vLLM 0.19 FP8 is UNCONFIRMED until the first Kaggle start.

Flags: qwen3_coder tool parser, qwen3 reasoning parser. Sampling follows the Duck: thinking on, temperature 0.6 /
top_p 0.95 / top_k 20. The serving bench (v3) measured ``qwen_fp8`` at 388 tok/s over 10 streams with a 0.63 prefix hit rate.

The Flash-Next profiles (Keith Tyser's pinned runtime, ``qwen4_exp`` MoE) stay defined but are not in the chain:
The serving bench (v4) killed ``qwen_flash_pc_64k`` (KV cache does not fit) and ``qwen_flash_pc`` (8.7% prefix hits), and the Flash-Next
stress test failed. The bundle licence is MIT in its SOURCE_IDENTITY and "unknown" on Kaggle: UNCONFIRMED.
The 27B NVFP4 (overseer66, an unsloth mixed quant) is gone: vLLM 0.19 has no ``lm_head.weight_scale`` for its FP8
lm_head (bench v3). ``nvidia/Qwen3.8-27B-NVFP4`` runs on SGLang instead (``SGLANG_CHAIN`` below).
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Callable

from harness.llm import sglang
from harness.llm.server import find_model_dir
from harness.llm.spec import ModelSpec

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
    # Prefix caching plus the 65k window our agent's compaction needs (reserve 16k; the 60k trigger is capped at
    # window - reserve, so 49k here).
    "qwen_flash_pc_64k": {"model_dataset": FLASH, "runtime": "flash_next", "env": {},
                          "flags": {**FLASH_PC, "--max-model-len": "65536"}},
    # Prefix caching: the agent's prompts are long and append-only (bench v3: 2.0 s vs 8.1 s per later turn).
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
# The same profile with one image per prompt (the agent keeps only the newest game image in its context).
PROFILES["qwen_fp8_vision"] = {**PROFILES["qwen_fp8"], "flags": {
    **PROFILES["qwen_fp8"]["flags"], "--limit-mm-per-prompt": '{"image": 1, "video": 0}'}}

# Option B (owner go 2026-10-02): nvidia/Qwen3.8-27B-NVFP4 on SGLang 0.5.19 "Pennyroyal" (the milestone-2
# reference's RTX Pro 6000 build). The checkpoint is ModelOpt MIXED_PRECISION: MLP + lm_head NVFP4 (group 16),
# self-attention and Gated-DeltaNet projections FP8, the MTP layer (mtp.*) unquantized, so NEXTN speculation runs on
# the model's own MTP head. SGLang detects the format from config.json (quant_method "modelopt" -> "modelopt_mixed"),
# so no --quantization flag. Kaggle: banwait13/qwencoder = the HF snapshot (shard sizes equal our SHA256-verified local
# download of 2026-10-01). Wheels: banwait13/sglangwheels (flat copy of the Pennyroyal v2.5.3 wheelhouse).
# Flags follow the reference launcher (CFG in kaggle/reference/arc-agi-3-milestone-2-solution.ipynb, cell 12) minus
# its Flash-Next-only ones (AutoRound, external draft, PLE offload, MoE runner). The NVFP4 bench measured 1162 tok/s with MTP5
# (10 streams, thinking on). FR-Spec (--speculative-token-map) is not used: it crashes with the NVFP4 lm_head.
NVFP4 = "banwait13/qwencoder"
SGLANG_WHEELS = "banwait13/sglangwheels"
SGL_NV = {
    "--load-format": "safetensors", "--model-loader-extra-config": '{"enable_multithread_load":false}',
    "--weight-loader-prefetch-checkpoints": None,
    "--tensor-parallel-size": "1", "--dtype": "bfloat16", "--trust-remote-code": None,
    "--kv-cache-dtype": "fp8_e4m3", "--mem-fraction-static": "0.92", "--context-length": "131072", "--page-size": "64",
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
# Five draft steps instead of three: the NVFP4 bench winner.
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
# NVFP4 chain: NVFP4 + MTP5 + image, then MTP3, then no MTP, then text only, then the measured vLLM FP8 fallback.
SGLANG_CHAIN = ["sglang_nvfp4_mtp5_vision", "sglang_nvfp4_mtp_vision", "sglang_nvfp4_vision", "sglang_nvfp4",
                "qwen_fp8_vision", "qwen_fp8"]

# --- Flash-Next on SGLang -----------------------------------------------------------------------------------------
# Qwen3.8-Flash-Next (qwen4_exp: 125B MoE, 6B active, Gated DeltaNet + sparse attention, a 51B n-gram (PLE) embedding)
# served the way the milestone-2 reference serves it (kaggle/reference/arc-agi-3-milestone-2-solution.ipynb, cell 12):
# Intel's AutoRound W4A16 weights on the Pennyroyal SGLang build (our banwait13/sglangwheels is the same wheelhouse and
# carries its FR-Spec map hot_tokens_64k.pt), the PLE table in pinned host RAM (--ple-offload-embedding, ~102 GB of
# BF16 shards), NEXTN speculation with 3 steps on albucino's INT4 MTP drafter plus FR-Spec. Flags are the reference's,
# window included: 139264, so a 120k prompt (the agent's compaction trigger) still has room for a full 16k reply.
# Licences (qwen-community-1.0 for the model, dfranzen's uploads) are UNCONFIRMED for prize eligibility; throughput on
# our agent is UNCONFIRMED until a run.
FLASH_SGL = "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1"
FLASH_DRAFT = "dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1"
# Built by the runtimes below, before launch: the drafter's SGLang view and the checked FR-Spec map.
FLASH_DRAFT_VIEW = sglang.PREFIX / "flash-draft-view"
FLASH_TOKEN_MAP = sglang.PREFIX / "flash-token-map" / "hot_tokens_64k.pt"
# The reference pins both: the map only fits the tokenizer it was built from.
TOKEN_MAP_SHA = "becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25"
TOKENIZER_SHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
EXPERT_TARGET = r"re:^mtp\.layers\.0\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$"
DENSE_IGNORE = r"re:^(?!mtp\.layers\.0\.mlp\.experts(?:\.|$)).*"

SGL_FLASH = {
    "--load-format": "safetensors", "--model-loader-extra-config": '{"enable_multithread_load":false}',
    "--weight-loader-prefetch-checkpoints": None, "--weight-loader-drop-cache-after-load": None,
    "--tensor-parallel-size": "1", "--dtype": "bfloat16", "--quantization": "auto-round", "--trust-remote-code": None,
    "--kv-cache-dtype": "fp8_e4m3", "--mem-fraction-static": "0.96", "--context-length": "139264", "--page-size": "64",
    "--max-running-requests": "10", "--chunked-prefill-size": "8192", "--max-prefill-tokens": "16384",
    # a graph for every batch size up to 10: a missing size (the reference skipped 3, 5 and 6) pads to the next one
    "--cuda-graph-max-bs-decode": "10", "--cuda-graph-bs-decode": [str(b) for b in range(1, 11)],
    "--mamba-ssm-dtype": "bfloat16", "--max-mamba-cache-size": "60", "--mamba-radix-cache-strategy": "extra_buffer",
    "--mamba-track-interval": "64", "--mamba-backend": "flashinfer", "--linear-attn-decode-backend": "flashinfer",
    "--linear-attn-prefill-backend": "flashinfer", "--moe-runner-backend": "auto", "--ple-offload-embedding": None,
    "--mm-feature-transport": "cpu", "--image-processor-backend": "pil",
    "--reasoning-parser": "qwen3", "--tool-call-parser": "qwen3_coder",
    "--chat-template": "{model_dir}/chat_template.jinja",
    "--default-chat-template-kwargs": '{"preserve_thinking":true}',
    "--watchdog-timeout": "1800", "--schedule-policy": "lpm", "--warmups": "structured_output",
    "--enable-cache-report": None, "--enable-metrics": None, "--enable-request-time-stats-logging": None,
    "--gdn-mtp-cache-mode": "none"}
# NEXTN on the external INT4 MTP drafter: linear chain (topk 1), strict accept.
SGL_FLASH_MTP = {"--speculative-algorithm": "NEXTN", "--speculative-num-steps": "3", "--speculative-eagle-topk": "1",
                 "--speculative-num-draft-tokens": "4", "--speculative-draft-model-path": str(FLASH_DRAFT_VIEW),
                 "--speculative-draft-model-quantization": "compressed-tensors",
                 "--speculative-moe-runner-backend": "auto", "--speculative-draft-kv-cache-dtype": "fp8_e4m3",
                 "--speculative-accept-threshold-single": "1.0", "--speculative-accept-threshold-acc": "1.0"}
SGL_FLASH_FRSPEC = {**SGL_FLASH_MTP, "--speculative-token-map": str(FLASH_TOKEN_MAP)}

_PRECACHED: set[str] = set()
_sglang_wheels = sglang.runtime(SGLANG_WHEELS)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _precache_once(find_input: Callable[[str], Path], *refs: str) -> None:
    """Warm the page cache with the weights in the background, once per input. Three threads, as the reference: the
    target plus the PLE table is ~180 GB against 176 GB of RAM, so this only helps the first shards load."""
    for ref in refs:
        if ref in _PRECACHED:
            continue
        try:
            root = find_input(ref)
        except Exception as exc:  # noqa: BLE001 - a missing optional input only loses the warm cache
            sglang._log(f"precache {ref} skipped: {exc!r}")
            continue
        _PRECACHED.add(ref)
        threading.Thread(target=sglang.precache, args=([root],), kwargs={"threads": 3}, name=f"precache-{ref}",
                         daemon=True).start()


def flash_sglang_runtime(working_dir: Path, find_input: Callable[[str], Path]) -> dict[str, Any]:
    """The SGLang wheelhouse, plus a check that the target is the AutoRound INT4 checkpoint the flags expect."""
    _precache_once(find_input, FLASH_SGL, FLASH_DRAFT)
    rt = _sglang_wheels(working_dir, find_input)
    target = find_model_dir(find_input(FLASH_SGL))
    q = json.loads((target / "config.json").read_text(encoding="utf-8")).get("quantization_config", {})
    if q.get("quant_method") != "auto-round" or q.get("bits") != 4:
        raise RuntimeError(f"{FLASH_SGL} is not the AutoRound INT4 checkpoint: {q.get('quant_method')} {q.get('bits')}")
    return {**rt, "info": {**rt["info"], "target": str(target)}}


def prepare_draft_view(target: Path, source_root: Path, view: Path) -> Path:
    """The drafter as SGLang loads it, without copying weights: its config with the expert target and ignore rules
    the reference uses, symlinks to its shards, and the target's tokenizer and template (the reference's
    prepare_draft_view, rebuilt in a fixed directory)."""
    configs = [source_root / "config.json"] if (source_root / "config.json").is_file() else []
    configs += sorted(p for p in source_root.rglob("config.json") if p not in configs)
    found = []
    for path in configs:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        q = cfg.get("quantization_config", {})
        if q.get("quant_method") == "compressed-tensors" and "mtp_routed_experts" in q.get("config_groups", {}):
            found.append((path.parent, cfg))
    if len(found) != 1:
        raise RuntimeError(f"expected one INT4 g32 MTP checkpoint under {source_root}, found {len(found)}")
    source, cfg = found[0]
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    if not any(k.startswith("mtp.layers.0.mlp.experts.") for k in index["weight_map"]):
        raise RuntimeError("the drafter's index has no MTP experts")
    missing = [s for s in shards if not (source / s).is_file()]
    if missing:
        raise RuntimeError(f"drafter shards missing: {missing}")
    cfg = copy.deepcopy(cfg)
    q = cfg["quantization_config"]
    group = q["config_groups"]["mtp_routed_experts"]
    w = group["weights"]
    if (w["num_bits"], w["group_size"], w["symmetric"]) != (4, 32, True):
        raise RuntimeError("expected symmetric INT4 group-32 draft experts")
    if group["targets"] not in (["RoutedExperts"], [EXPERT_TARGET]) or q.get("ignore", []) not in ([], [DENSE_IGNORE]):
        raise RuntimeError(f"unexpected draft target/ignore rules: {group['targets']} {q.get('ignore')}")
    group["targets"], q["ignore"] = [EXPERT_TARGET], [DENSE_IGNORE]
    shutil.rmtree(view, ignore_errors=True)
    view.mkdir(parents=True)
    (view / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    links = {name: source / name for name in shards}
    links["model.safetensors.index.json"] = source / "model.safetensors.index.json"
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json",
                 "generation_config.json"):
        if (target / name).is_file():
            links[name] = target / name
    for name, src in links.items():
        (view / name).parent.mkdir(parents=True, exist_ok=True)
        (view / name).symlink_to(src.resolve())
    sglang._log(f"draft view {view}: {len(shards)} shards from {source}")
    return view


def flash_mtp_runtime(working_dir: Path, find_input: Callable[[str], Path]) -> dict[str, Any]:
    rt = flash_sglang_runtime(working_dir, find_input)
    view = prepare_draft_view(Path(rt["info"]["target"]), find_input(FLASH_DRAFT), FLASH_DRAFT_VIEW)
    return {**rt, "info": {**rt["info"], "draft_view": str(view)}}


def flash_frspec_runtime(working_dir: Path, find_input: Callable[[str], Path]) -> dict[str, Any]:
    """MTP plus the FR-Spec map from the wheelhouse, both hashes checked as in the reference."""
    rt = flash_mtp_runtime(working_dir, find_input)
    hits = sorted(find_input(SGLANG_WHEELS).rglob("hot_tokens_64k.pt"))
    if not hits:
        raise FileNotFoundError(f"hot_tokens_64k.pt not in {SGLANG_WHEELS}")
    if _sha256(hits[0]) != TOKEN_MAP_SHA:
        raise RuntimeError(f"FR-Spec map {hits[0]} is not the pinned one")
    if _sha256(Path(rt["info"]["target"]) / "tokenizer.json") != TOKENIZER_SHA:
        raise RuntimeError("the target's tokenizer differs from the one the FR-Spec map was built for")
    FLASH_TOKEN_MAP.parent.mkdir(parents=True, exist_ok=True)
    if FLASH_TOKEN_MAP.exists() or FLASH_TOKEN_MAP.is_symlink():
        FLASH_TOKEN_MAP.unlink()
    FLASH_TOKEN_MAP.symlink_to(hits[0].resolve())
    return {**rt, "info": {**rt["info"], "token_map": str(hits[0])}}


def _flash(runtime: str, vision: bool, *extra: dict[str, Any]) -> dict[str, Any]:
    flags: dict[str, Any] = dict(SGL_FLASH)
    for e in extra:
        flags.update(e)
    return {"model_dataset": FLASH_SGL, "runtime": runtime, "vision": vision, "env": {}, "flags": flags}


PROFILES.update({
    "sglang_flash_frspec_vision": _flash("flash_frspec", True, SGL_FLASH_FRSPEC),  # the reference's recipe
    "sglang_flash_mtp_vision": _flash("flash_mtp", True, SGL_FLASH_MTP),
    "sglang_flash_vision": _flash("flash_sglang", True),
    "sglang_flash": _flash("flash_sglang", False),
})
# Flash-Next with MTP + FR-Spec, then without FR-Spec, then without speculation, then text only; then the 27B NVFP4 chain.
FLASH_CHAIN = ["sglang_flash_frspec_vision", "sglang_flash_mtp_vision", "sglang_flash_vision", "sglang_flash",
               *SGLANG_CHAIN]

SPEC = ModelSpec(
    name="qwen_flash_next",
    served_model_name="qwen3.8",
    profiles=PROFILES,
    profile_chain=FLASH_CHAIN,
    # The reference's sampling (the model card says temperature 1.0 for thinking; 0.7 is what scored in the reference).
    llm={"temperature": 0.7, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": True}},
    smoke_template_kwargs={"enable_thinking": False},  # the Duck's smoke test: thinking could exhaust max_tokens
    runtimes={"flash_next": flash_next_runtime,
              "sglang": sglang.runtime(SGLANG_WHEELS, precache_datasets=(NVFP4,)),
              "flash_sglang": flash_sglang_runtime, "flash_mtp": flash_mtp_runtime,
              "flash_frspec": flash_frspec_runtime},
)
