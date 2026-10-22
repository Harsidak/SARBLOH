"""NVIDIA Nemotron 3.5 Super VL (120B MoE, 12B active) NVFP4 on SGLang (config ``model: "nemotron"``).

The model (``NemotronH_Omni_Reasoning_V3``, trust_remote_code): 88 layers, about 40 Mamba-2, about 40 MoE (512 experts,
22 per token, one shared) and 8 attention layers with 2 KV heads, one MTP layer, a C-RADIO ViT-H vision tower. The
checkpoint is ModelOpt mixed precision: experts NVFP4 (group 16), Mamba projections, attention and lm_head FP8. SGLang
reads the format from config.json, so no --quantization flag. 1M native context; we serve 131072, the harness's window.

Flags follow the model card's SGLang recipe (lmsysorg/sglang:v0.5.20, one GB200): flashinfer Mamba backend, float16 SSM
state with stochastic rounding (5 Philox rounds), EAGLE on the built-in MTP head with 5 steps and topk 1, the
qwen3_coder tool parser and the nemotron_3 reasoning parser. Added from our Qwen SGLang profiles: prefix caching of the
Mamba state (extra_buffer, a checkpoint every 64 tokens, page size 64), fp8 KV, 10 running requests with a CUDA graph
for every batch size, lpm scheduling and the cache report the prefix-hit metrics read.

Memory (UNCONFIRMED until the first start): about 80 GB of weights. KV is 4 KB a token in fp8 (8 attention layers x 2
KV heads x 128 x K and V), so 10 x 131k is about 5.4 GB; the Mamba state is about 84 MB a sequence (40 layers x 128
heads x 64 x 128 x 2 bytes), and --max-mamba-cache-size 40 holds about 3.4 GB of it, for live slots and cached prefixes.

Chat template: ``truncate_history_thinking`` defaults to True and strips the thinking of every assistant message before
the last user message. Our context adds a user message almost every turn (observations, ipython output), so the
default would drop each reply's thinking on the next request: the model loses its earlier reasoning and the reply just
generated (thinking included) misses the prefix cache. It is set False, as Qwen's preserve_thinking, both in the
client defaults and on the server (checked by rendering the template: with False every turn's prompt is a prefix of
the next and keeps all thinking). The template has no "xhigh"/"low" effort, only ``medium_effort`` (a
"{reasoning effort: efficient}" line after the last user message); ``template_kwargs`` maps the harness's
reasoning_effort onto it.

Wheels: the card asks for SGLang 0.5.20; our Pennyroyal wheelhouse is 0.5.19. The chain tries a 0.5.20 wheelhouse
first, then Pennyroyal (whether 0.5.19 knows this architecture is UNCONFIRMED). Licence OpenMDW-1.1: prize eligibility
UNCONFIRMED.
"""

from __future__ import annotations

from typing import Any

from harness.llm import sglang
from harness.llm.spec import ModelSpec

WEIGHTS = "banwait13/nemotron-3-5-super-vl-nvfp4"  # nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 snapshot
WHEELS_0520 = "banwait13/sglang0520wheels"  # SGLang 0.5.20 wheelhouse, same layout as sglangwheels
WHEELS_PENNYROYAL = "banwait13/sglangwheels"  # Pennyroyal v2.5.3, SGLang 0.5.19

SGL = {
    "--load-format": "safetensors", "--model-loader-extra-config": '{"enable_multithread_load":false}',
    "--weight-loader-prefetch-checkpoints": None,
    "--tensor-parallel-size": "1", "--dtype": "bfloat16", "--trust-remote-code": None,
    "--kv-cache-dtype": "fp8_e4m3", "--mem-fraction-static": "0.94", "--context-length": "131072", "--page-size": "64",
    "--max-running-requests": "10", "--chunked-prefill-size": "8192", "--max-prefill-tokens": "16384",
    "--cuda-graph-max-bs-decode": "10", "--cuda-graph-bs-decode": [str(b) for b in range(1, 11)],
    # the card's Mamba settings
    "--mamba-backend": "flashinfer", "--mamba-ssm-dtype": "float16", "--enable-mamba-cache-stochastic-rounding": None,
    "--mamba-cache-philox-rounds": "5",
    # prefix caching of the Mamba state, as in our Qwen profiles
    "--max-mamba-cache-size": "40", "--mamba-radix-cache-strategy": "extra_buffer", "--mamba-track-interval": "64",
    "--mm-feature-transport": "cpu",
    "--reasoning-parser": "nemotron_3", "--tool-call-parser": "qwen3_coder",
    "--chat-template": "{model_dir}/chat_template.jinja",
    "--default-chat-template-kwargs": '{"truncate_history_thinking": false}',
    "--watchdog-timeout": "1800", "--schedule-policy": "lpm", "--enable-cache-report": None, "--enable-metrics": None}
# The card's MTP recipe: EAGLE on the built-in MTP layer, 5 steps, linear chain. Draft tokens = steps + 1 (topk 1).
MTP5 = {"--speculative-algorithm": "EAGLE", "--speculative-num-steps": "5", "--speculative-eagle-topk": "1",
        "--speculative-num-draft-tokens": "6"}
MTP3 = {**MTP5, "--speculative-num-steps": "3", "--speculative-num-draft-tokens": "4"}


# The card's 0.5.20 Mamba-cache rounding flags: left out on 0.5.19, where they may not exist (UNCONFIRMED).
NEW_IN_0520 = ("--enable-mamba-cache-stochastic-rounding", "--mamba-cache-philox-rounds")


def _profile(runtime: str, vision: bool, *extra: dict[str, Any]) -> dict[str, Any]:
    flags: dict[str, Any] = dict(SGL)
    for e in extra:
        flags.update(e)
    if runtime == "sglang_pennyroyal":
        flags = {k: v for k, v in flags.items() if k not in NEW_IN_0520}
    return {"model_dataset": WEIGHTS, "runtime": runtime, "vision": vision, "env": {}, "flags": flags}


PROFILES: dict[str, dict[str, Any]] = {
    "nemotron_mtp5_vision": _profile("sglang_0520", True, MTP5),   # the card's recipe plus our prefix cache
    "nemotron_mtp3_vision": _profile("sglang_0520", True, MTP3),
    "nemotron_vision": _profile("sglang_0520", True),
    "nemotron": _profile("sglang_0520", False),
    "nemotron_pennyroyal_mtp5_vision": _profile("sglang_pennyroyal", True, MTP5),
    "nemotron_pennyroyal": _profile("sglang_pennyroyal", False),
}
# SGLang 0.5.20: MTP5 + image, MTP3, no MTP, text only; then the same on the 0.5.19 Pennyroyal wheels.
CHAIN = list(PROFILES)

EFFORT = {"xhigh": {}, "medium": {}, "low": {"medium_effort": True}}


def template_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The harness's ``reasoning_effort`` (xhigh | medium | low, Qwen3.8's names) in this template's terms: full
    reasoning is its default, so xhigh and medium send nothing and low turns on ``medium_effort``."""
    out = {k: v for k, v in kwargs.items() if k != "reasoning_effort"}
    return {**EFFORT.get(kwargs.get("reasoning_effort"), {}), **out}


SPEC = ModelSpec(
    name="nemotron",
    served_model_name="nemotron-3.5-super-vl",
    profiles=PROFILES,
    profile_chain=CHAIN,
    # The card's text sampling: temperature 1.0, top_p 0.95, no top_k (vision tasks: 0.6; our images are an aid to a
    # text-and-code task, so the text setting). Thinking on, history thinking kept (see the module note).
    llm={"temperature": 1.0, "top_p": 0.95, "top_k": None,
         "chat_template_kwargs": {"enable_thinking": True, "truncate_history_thinking": False}},
    smoke_template_kwargs={"enable_thinking": False},  # thinking could exhaust the smoke test's max_tokens
    template_kwargs=template_kwargs,
    runtimes={"sglang_0520": sglang.runtime(WHEELS_0520, precache_datasets=(WEIGHTS,)),
              "sglang_pennyroyal": sglang.runtime(WHEELS_PENNYROYAL, precache_datasets=(WEIGHTS,))},
)
