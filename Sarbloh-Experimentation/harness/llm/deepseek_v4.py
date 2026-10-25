"""DeepSeek-V4-Flash-0731 on SGLang (config ``model: "deepseek_v4"``). Text only.

The model (``DeepseekV4ForCausalLM``, native in SGLang): 43 layers, 256 routed experts (6 per token, one shared), 13B
active of 284B, hybrid compressed attention (c4 / c128 layers plus a 128-token sliding window), hyper-connections.
The routed experts ship in MXFP4, everything else FP8. Read in the 0.5.20 wheel (2026-10-25), not yet run:
- SGLang detects the FP4 experts itself (``SGLANG_DSV4_FP4_EXPERTS``) and on SM120 runs them with flashinfer's
  cutlass MXFP8 x MXFP4 MoE (``--moe-runner-backend flashinfer_mxfp4``, its default for this layout). NVIDIA's NVFP4
  builds get ``flashinfer_trtllm_routed`` instead, a B200 (SM100) kernel: the MXFP4 originals are the right files here.
- It forces the ``dsv4`` attention backend (FlashMLA has an SM120 path), page size 256 and ``swa_full_tokens_ratio``
  0.1. Radix (prefix) caching stays on.
- Prompts come from SGLang's own port of DeepSeek's ``encoding_dsv4`` (the model has no chat template). Thinking is
  switched by ``chat_template_kwargs.thinking`` (SGLang's default is off), the effort by ``reasoning_effort``
  (low / high / max under the "official" profile; without ``encoding/`` in the weights SGLang falls back to the
  "preview" profile, so the profile is set in the config override). When the request has tools, every earlier turn's
  thinking is rendered back, so each prompt is a prefix of the next one, as with Flash-Next's preserve_thinking.
- Images: the DSV4 path flattens message parts to text, so images are dropped. Neither wheelhouse can serve
  DeepSeek's vision model's vision tower; every profile here is text only.

Weights (both already on Kaggle, MIT):
- ``REAP``: DeepSeek-V4-Flash-0731 pruned to 132 experts a layer (puwaer's REAP, 79 GiB; its card: GSM8K / MATH-500 /
  HumanEval+ / MBPP+ within 2 points of the full model). It fits the 96 GB card with room for the KV cache, so it
  runs batched like Flash-Next. No MTP or DSpark weights: no speculative decoding.
- ``VISION_EXP``: the full DeepSeek-V4-Flash-Vision-Exp (168 GB), served text only with the weights that do not fit
  offloaded to CPU RAM (``--cpu-offload-gb``): every decode step then streams them over PCIe, so it is a measurement,
  not in the chain. SGLang's loader would stop on its vision tensors (``vision.*.mlp.w1`` maps to a parameter the
  text model lacks), so the runtime serves a view of the weights without them.
"""

from __future__ import annotations

import json
import shutil
import struct
import time
from pathlib import Path
from typing import Any, Callable

from harness.llm import sglang
from harness.llm.server import find_model_dir
from harness.llm.spec import ModelSpec

REAP = "cosmicbe/dsv4-flash-0731-reap-150b/pyTorch/default/1"  # puwaer/DeepSeek-V4-Flash-0731-reap-150b mirror
VISION_EXP = "trnmtin/deepseek-v4-flash-vision-exp/pyTorch/hf-original/1"  # deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
WHEELS_0520 = "banwait13/sglang0520wheels"
WHEELS_PENNYROYAL = "banwait13/sglangwheels"  # SGLang 0.5.19, the build Flash-Next runs on

SGL = {
    "--load-format": "safetensors", "--model-loader-extra-config": '{"enable_multithread_load":false}',
    "--weight-loader-prefetch-checkpoints": None,
    "--tensor-parallel-size": "1", "--dtype": "bfloat16", "--trust-remote-code": None,
    "--kv-cache-dtype": "fp8_e4m3", "--mem-fraction-static": "0.90", "--context-length": "139264", "--page-size": "256",
    "--max-running-requests": "10", "--chunked-prefill-size": "8192", "--max-prefill-tokens": "16384",
    "--cuda-graph-max-bs-decode": "10", "--cuda-graph-bs-decode": [str(b) for b in range(1, 11)],
    "--moe-runner-backend": "flashinfer_mxfp4",
    "--reasoning-parser": "deepseek-v4", "--tool-call-parser": "deepseekv4",
    "--json-model-override-args": '{"dsv4_reasoning_effort_profile": "official"}',
    "--watchdog-timeout": "1800", "--schedule-policy": "lpm", "--enable-cache-report": None, "--enable-metrics": None}
MARLIN = {"--moe-runner-backend": "marlin"}  # SGLang's other SM120 MXFP4 MoE path
OFFLOAD = {"--cpu-offload-gb": "80"}  # 157 GiB of text weights against ~78 GiB left for them on the GPU
DSPARK = {"--speculative-algorithm": "DSPARK"}  # DeepSeek's draft heads, inside the Vision-Exp checkpoint
ENV = {"SGLANG_DEFAULT_THINKING": "1"}  # a request without chat_template_kwargs still thinks


def _safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n


def _text_only(name: str) -> bool:
    return not (name.startswith(("vision.", "aligner.", "image_")))


def prepare_text_view(root: Path, view: Path) -> Path:
    """The Vision-Exp weights as the text model loads them, without copying the big shards: the shards that hold
    vision tensors (all in the first, checked 2026-10-25) are rewritten without them, every other shard is a symlink,
    and the index drops the vision names. The Kaggle copy keeps its configs and tokenizer in a sub-folder apart from
    the shards; the view puts them side by side."""
    t0 = time.time()
    meta = find_model_dir(root)
    index_path = meta / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_dirs = {p.parent for p in root.rglob("model-*-of-*.safetensors")}
    if len(shard_dirs) != 1:
        raise RuntimeError(f"expected the shards in one folder under {root}, found {sorted(map(str, shard_dirs))}")
    shards = shard_dirs.pop()
    files = sorted(set(index["weight_map"].values()))
    missing = [f for f in files if not (shards / f).is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} shards missing from {shards}: {missing[:3]}")
    drop = {n for n in index["weight_map"] if not _text_only(n)}
    rewrite = sorted({index["weight_map"][n] for n in drop})
    shutil.rmtree(view, ignore_errors=True)
    view.mkdir(parents=True)
    for name in rewrite:
        header, base = _safetensors_header(shards / name)
        keep = sorted((k for k in header if k != "__metadata__" and _text_only(k)),
                      key=lambda k: header[k]["data_offsets"][0])
        out_header: dict[str, Any] = {"__metadata__": header.get("__metadata__", {"format": "pt"})}
        size = 0
        for k in keep:
            start, end = header[k]["data_offsets"]
            out_header[k] = {**header[k], "data_offsets": [size, size + end - start]}
            size += end - start
        blob = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
        blob += b" " * (-len(blob) % 8)
        with (shards / name).open("rb") as src, (view / name).open("wb") as out:
            out.write(struct.pack("<Q", len(blob)) + blob)
            for k in keep:
                start, end = header[k]["data_offsets"]
                src.seek(base + start)
                left = end - start
                while left:
                    block = src.read(min(left, 64 << 20))
                    if not block:
                        raise RuntimeError(f"{name} ends inside {k}")
                    out.write(block)
                    left -= len(block)
    for name in files:
        if name not in rewrite:
            (view / name).symlink_to((shards / name).resolve())
    for item in meta.iterdir():
        if item.name != index_path.name and not (view / item.name).exists():
            (view / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
    index["weight_map"] = {k: v for k, v in index["weight_map"].items() if k not in drop}
    (view / index_path.name).write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    sglang._log(f"text view {view}: {len(drop)} vision tensors dropped from {rewrite}, {len(files)} shards, "
                f"{time.time() - t0:.0f} s")
    return view


VISION_EXP_VIEW = sglang.PREFIX / "dsv4-vision-exp-text"


def _vision_exp_runtime(wheels: str) -> Callable[[Path, Callable[[str], Path]], dict[str, Any]]:
    base = sglang.runtime(wheels)  # no precache: 168 GB does not fit the page cache next to the offloaded weights

    def prepare(working_dir: Path, find_input: Callable[[str], Path]) -> dict[str, Any]:
        rt = base(working_dir, find_input)
        view = prepare_text_view(find_input(VISION_EXP), VISION_EXP_VIEW)
        return {**rt, "model_dir": str(view), "info": {**rt["info"], "text_view": str(view)}}

    return prepare


def _profile(model: str, runtime: str, *extra: dict[str, Any]) -> dict[str, Any]:
    flags: dict[str, Any] = dict(SGL)
    for e in extra:
        flags.update(e)
    return {"model_dataset": model, "runtime": runtime, "vision": False, "env": dict(ENV), "flags": flags}


PROFILES: dict[str, dict[str, Any]] = {
    "dsv4_reap150": _profile(REAP, "sglang_0520"),
    "dsv4_reap150_marlin": _profile(REAP, "sglang_0520", MARLIN),
    "dsv4_reap150_pennyroyal": _profile(REAP, "sglang_pennyroyal"),
    # measurements, not in the chain (see the module note)
    "dsv4_vision_exp_offload_dspark": _profile(VISION_EXP, "vision_exp_0520", OFFLOAD, DSPARK),
    "dsv4_vision_exp_offload": _profile(VISION_EXP, "vision_exp_0520", OFFLOAD),
}
# REAP on SGLang 0.5.20 with flashinfer's MoE, then Marlin's, then the 0.5.19 Pennyroyal build.
CHAIN = ["dsv4_reap150", "dsv4_reap150_marlin", "dsv4_reap150_pennyroyal"]

# The harness's effort names (Qwen3.8's) in DeepSeek's: xhigh, the top one, is DeepSeek's max (the setting of its own
# agent benchmarks).
EFFORT = {"xhigh": "max", "medium": "high", "low": "low"}


def template_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The harness's per-call template arguments in SGLang's DSV4 terms: ``enable_thinking`` is ``thinking`` and the
    effort goes in ``reasoning_effort`` (SGLang moves it from the template arguments onto the request)."""
    out = {k: v for k, v in kwargs.items() if k not in ("enable_thinking", "reasoning_effort")}
    if "enable_thinking" in kwargs:
        out["thinking"] = bool(kwargs["enable_thinking"])
    if kwargs.get("reasoning_effort") in EFFORT:
        out["reasoning_effort"] = EFFORT[kwargs["reasoning_effort"]]
    return out


SPEC = ModelSpec(
    name="deepseek_v4",
    served_model_name="deepseek-v4-flash",
    profiles=PROFILES,
    profile_chain=CHAIN,
    # DeepSeek's sampling for its agent benchmarks: temperature 1.0, top_p 0.95. Thinking on.
    llm={"temperature": 1.0, "top_p": 0.95, "top_k": None, "chat_template_kwargs": {"thinking": True}},
    smoke_template_kwargs={"thinking": False},  # thinking could exhaust the smoke test's max_tokens
    template_kwargs=template_kwargs,
    runtimes={"sglang_0520": sglang.runtime(WHEELS_0520, precache_datasets=(REAP,)),
              "sglang_pennyroyal": sglang.runtime(WHEELS_PENNYROYAL, precache_datasets=(REAP,)),
              "vision_exp_0520": _vision_exp_runtime(WHEELS_0520)},
)
