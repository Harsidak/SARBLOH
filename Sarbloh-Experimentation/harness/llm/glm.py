"""GLM-5.3-Flash on llama.cpp, experts split between the GPU and CPU RAM (config ``model: "glm"``). Images on.

The model (``Glm5NextForConditionalGeneration``, zai-org, MIT): 320B MoE, 18B active, 45 layers (42 with 288 routed
experts, 8 per token, plus one shared), linear attention (KDA) on 34 layers and DeepSeek sparse attention on 11,
hyper-connections, one MTP layer, a 24-layer vision tower. Natively multimodal.

Why llama.cpp and not SGLang: at 4 bits the experts alone are about 150 GB, more than the 96 GB card. SGLang can only
stream offloaded weights over PCIe on every step (``--cpu-offload-gb``); llama.cpp keeps the experts that do not fit in
CPU RAM and runs them on the CPU, the rest on the GPU (``--fit``, on by default, picks how many layers' experts stay
on the CPU so the GPU is full). GLM-5.3-Flash is in llama.cpp since PR 27773 (merged 2026-09-30), its chat template
needs the ``unique`` filter (PR 30819, 2026-10-19), so the build must be newer than both.

Inputs (read 2026-10-25, not yet run):
- ``GGUF``: kemake's Kaggle copy of unsloth/GLM-5.3-Flash-GGUF UD-IQ4_XS (revision 621d456, 158 GB in 5 shards) with
  ``mmproj-BF16.gguf`` (the vision tower, 1.2 GB). Its description: "Pinned offline ARC-AGI-3 diagnostic artifact".
- ``LLAMA``: our dataset, one flat folder: llama.cpp release b11878's Linux CUDA 12.8 build (llama-server and its
  libraries), its CUDA runtime, and glibc 2.39 + libstdc++ from Ubuntu 24.04 (the release needs them; Kaggle has
  glibc 2.35), every symlink stored as a real file. CUDA 12.8 runs on the Kaggle driver (580, CUDA 13.0).

Serving: 10 slots (the scheduler's 10), one KV pool shared by the slots with each slot capped at 131072 tokens, the
GLM tool-call and reasoning parsers that llama.cpp picks from the chat template (``--jinja``), and a host-RAM prompt
cache (``--cache-ram``) so a game that gets its slot back does not prefill from scratch. llama.cpp keeps a recurrent
model's state at checkpoints, so an append-only prompt reuses the slot's prefix. No MTP: the GGUF has no MTP layer.
Throughput on 10 games and the prefix hit rate are UNCONFIRMED until the first run.

The chat template always thinks (it opens ``<think>``); there is no switch. ``reasoning_effort`` takes ``low`` and
``high``; anything else is ``max``, the template's default.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from harness.llm.spec import ModelSpec

GGUF = "kemake/glm53-flash-unsloth-public/gguf/ud-iq4-xs/1"
LLAMA = "banwait13/llamacpp-cuda-b11878"
PREFIX = Path("/tmp/sarbloh-llamacpp")
# Fixed paths the flags name: the runtime links the vision tower here before launch.
MMPROJ = PREFIX / "mmproj.gguf"
WINDOW = 131072  # per slot

# The harness's efforts (xhigh / medium / low) in the template's terms. Thinking cannot be turned off, so a request
# without thinking (compaction summaries, the bench's plain decode) gets the least of it.
EFFORT = {"xhigh": "max", "medium": "high", "low": "low"}


def template_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in kwargs.items() if k not in ("enable_thinking", "reasoning_effort")}
    effort = "low" if kwargs.get("enable_thinking") is False else kwargs.get("reasoning_effort")
    if effort:
        out["reasoning_effort"] = EFFORT.get(effort, effort)
    return out


def _log(msg: str) -> None:
    print(f"[llamacpp {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stage(root: Path, dest: Path) -> Path:
    """Copies the dataset's flat folder (llama-server, its libraries, the CUDA runtime, and glibc 2.39 + libstdc++
    from Ubuntu 24.04 with their loader; every symlink stored as a real file, since Kaggle drops symlinks) into
    ``dest`` once, executable, and returns ``dest``."""
    servers = sorted(p for p in root.rglob("llama-server") if p.is_file())
    if not servers:
        raise FileNotFoundError(f"no llama-server in {root}")
    src = servers[0].parent
    want = "\n".join(sorted(f"{p.name}:{p.stat().st_size}" for p in src.iterdir() if p.is_file()))
    stamp = dest / ".staged"
    if not (stamp.exists() and stamp.read_text(encoding="utf-8") == want):
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(src, dest)
        for p in dest.iterdir():
            p.chmod(0o755)
        stamp.write_text(want, encoding="utf-8")
    return dest


def launcher(dest: Path, env: dict[str, str]) -> tuple[str, list[str], str]:
    """(program, args before the model, how) that runs llama-server here. The release is built on Ubuntu 24.04
    (glibc 2.38, GLIBCXX_3.4.32); Kaggle's image has glibc 2.35, so when the plain start fails, llama-server runs
    under the bundled glibc's own loader. Its backends (libggml-cuda.so, libggml-cpu-*.so) sit next to the loader,
    where ggml looks for them."""
    host = env.get("LD_LIBRARY_PATH", "")
    tries = [("direct", str(dest / "llama-server"), []),
             ("bundled glibc", str(dest / "ld-linux-x86-64.so.2"),
              ["--library-path", os.pathsep.join(p for p in (str(dest), host) if p), str(dest / "llama-server")])]
    errors = []
    for how, program, pre in tries:
        if not Path(program).exists():
            continue
        out = subprocess.run([program, *pre, "--version"], env=env, capture_output=True, text=True, timeout=120)
        if out.returncode == 0:
            return program, pre, f"{how}: {(out.stdout + out.stderr).strip()[-300:]}"
        errors.append(f"{how}: {(out.stderr or out.stdout)[-1000:]}")
    raise RuntimeError("llama-server does not run: " + " | ".join(errors))


def first_shard(root: Path) -> Path:
    """The GGUF llama.cpp is pointed at: the first of a split model (it finds the others), else the only model file."""
    models = sorted(p for p in root.rglob("*.gguf") if not p.name.startswith(("mmproj", "mtp")))
    firsts = [p for p in models if "-00001-of-" in p.name]
    if len(firsts) == 1:
        return firsts[0]
    if len(models) == 1:
        return models[0]
    raise FileNotFoundError(f"expected one model in {root}, found {[p.name for p in models]}")


def llamacpp_runtime(working_dir: Path, find_input: Callable[[str], Path]) -> dict[str, Any]:
    t = time.time()
    dest = stage(find_input(LLAMA), PREFIX / "runtime")
    weights = find_input(GGUF)
    model = first_shard(weights)
    mmproj = sorted(weights.rglob("mmproj*.gguf"))
    if MMPROJ.is_symlink() or MMPROJ.exists():
        MMPROJ.unlink()
    if mmproj:
        MMPROJ.symlink_to(mmproj[0].resolve())
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in (str(dest), env.get("LD_LIBRARY_PATH", "")) if p)
    program, pre, how = launcher(dest, env)
    _log(f"runtime ready in {time.time() - t:.0f} s ({how}), model {model.name}, mmproj {bool(mmproj)}")
    return {"env": env, "python": program, "serve": [*pre, "--model"], "model_dir": str(model), "backend": "llamacpp",
            "name_flag": "--alias",
            "info": {"launch": how, "model": str(model), "mmproj": str(mmproj[0]) if mmproj else None}}


FLAGS: dict[str, Any] = {
    "--parallel": "10", "--kv-unified": None, "--kv-unified-per-slot": str(WINDOW),
    # --fit (llama.cpp's default) sets the GPU layers and how many layers' experts stay in CPU RAM; the margin covers
    # the vision tower and the prompt-cache copies.
    "--fit": "on", "--fit-target": "3072",
    "--batch-size": "4096", "--ubatch-size": "2048", "--flash-attn": "auto",
    "--jinja": None, "--reasoning-format": "deepseek",
    "--cache-ram": "16384", "--ctx-checkpoints": "32",
    # the model card's sampling is temperature 1.0 / top_p 0.95 with nothing else: llama.cpp's own top_k 40 and
    # min_p 0.05 would apply to every request that does not set them
    "--top-k": "0", "--min-p": "0.0",
    "--metrics": None, "--timeout": "7200", "--no-warmup": None}

PROFILES: dict[str, dict[str, Any]] = {
    "glm_llamacpp_vision": {"model_dataset": GGUF, "runtime": "llamacpp", "vision": True, "tools": True,
                            "context": WINDOW, "env": {}, "flags": {**FLAGS, "--mmproj": str(MMPROJ)}},
    "glm_llamacpp": {"model_dataset": GGUF, "runtime": "llamacpp", "vision": False, "tools": True,
                     "context": WINDOW, "env": {}, "flags": {**FLAGS, "--no-mmproj": None}},
}

SPEC = ModelSpec(
    name="glm",
    served_model_name="glm-5.3-flash",
    profiles=PROFILES,
    profile_chain=["glm_llamacpp_vision", "glm_llamacpp"],
    llm={"temperature": 1.0, "top_p": 0.95, "top_k": 0, "chat_template_kwargs": {}},
    smoke_template_kwargs={"reasoning_effort": "low"},  # thinking could use up the smoke test's max_tokens
    runtimes={"llamacpp": llamacpp_runtime},
    template_kwargs=template_kwargs,
)
