"""TPU backend of lora_sft.py: LoRA SFT with Tunix's PeftTrainer on a MaxText model (``backend: "tunix"``).

    from lora_sft import run; card = run({"backend": "tunix", "model": "Qwen/Qwen3.8-27B", ...})

Same data path as the GPU backend (lora_sft.py renders and masks the conversations, so the tokens and loss masks are
identical), same result card. What differs is the model side:

  1. resolve the HF checkpoint (download with internet on, or a local directory);
  2. convert it once to a MaxText (Orbax) checkpoint on CPU, in a subprocess: ``maxtext.checkpoint_conversion.to_maxtext``;
  3. load it sharded over the TPU chips (``model_creation_utils.from_pretrained``), attach LoRA with qwix
     (``lora_utils.apply_lora_to_model``), train with Tunix (MaxText's ``MaxTextPeftTrainer`` + its loss);
  4. export the LoRA as a PEFT adapter (``to_huggingface``, CPU subprocess) that vLLM/SGLang load on the GPU.

Plain Tunix only has Gemma/Llama/Qwen2/Qwen3 models; the qwen3_5 architecture (Qwen3.5/3.8: Gated DeltaNet + gated
attention) is only in MaxText. MaxText ships ``qwen3.5-9b`` (dense) but no 27B, so the 27B runs as ``qwen3.5-9b`` with the
checkpoint's own dimensions as overrides, and the conversion tables get the checkpoint's config.json (``_patch``).
Gemma-4-31B is a native MaxText model (``gemma4-31b``) and needs neither.

LoRA defaults to the MLPs only on qwen3_5 (``wi_0``/``wi_1``/``wo`` = ``gate_proj``/``up_proj``/``down_proj``, one to one in
the export); MaxText fuses the DeltaNet projections (``in_proj_qkvz``), which has no 1:1 PEFT name.

UNCONFIRMED until a run: the 27B overrides, the export of scanned qwen3_5 LoRA weights, memory at the default length,
the Kaggle TPU VM's disk (the HF download and the converted copy are ~2 bytes/param each).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import lora_sft as base  # noqa: E402

TUNIX_DEFAULTS: dict[str, Any] = {
    "maxtext_model": "auto",          # auto: from config.json (qwen3_5 -> qwen3.5-9b + the checkpoint's dims; gemma4 with
                                      # 60 layers -> gemma4-31b) or any MaxText model name
    "work_dir": "auto",               # HF download + converted weights; auto = the writable dir with most free space
                                      # (never /kaggle/working: its 20 GB is the run's output)
    "keep_converted": True,           # reuse a finished conversion in work_dir
    "fsdp": -1,                       # ICI mesh: -1 = every chip not used by the other axes
    "tensor": 1,
    "per_device_batch": 1,            # global batch = this x chips (8 rows per step on a v5e-8)
    "remat_policy": "full",
    "vocab_tiling": 4,                # num_vocab_tiling: logits in chunks (248k vocab x 8k tokens is 8 GB in fp32)
    "lora_module_path": "auto",       # regex over MaxText module paths; auto: MLP only on qwen3_5, else MaxText's table
    "export_hf": True,                # PEFT adapter for vLLM/SGLang (adapter_model.safetensors + adapter_config.json)
    "resume_from": None,              # a previous run's checkpoint dir (e.g. the attached last output), copied in first
    "maxtext_overrides": {},          # any other MaxText config key, e.g. {"attention": "flash"}
}

# HF text_config -> MaxText keys for the qwen3_5 decoder (the 27B runs as qwen3.5-9b with these overridden)
QWEN3_5_KEYS = {
    "hidden_size": "base_emb_dim", "num_hidden_layers": "base_num_decoder_layers",
    "num_attention_heads": "base_num_query_heads", "num_key_value_heads": "base_num_kv_heads",
    "intermediate_size": "base_mlp_dim", "head_dim": "head_dim", "vocab_size": "vocab_size",
    "linear_num_key_heads": "gdn_num_key_heads", "linear_num_value_heads": "gdn_num_value_heads",
    "linear_key_head_dim": "gdn_key_head_dim", "linear_value_head_dim": "gdn_value_head_dim",
    "linear_conv_kernel_dim": "gdn_conv_kernel_dim", "full_attention_interval": "inhomogeneous_layer_cycle_interval",
    "rms_norm_eps": "normalization_layer_epsilon",
}
QWEN3_5_MLP_LORA = r"decoder/layers/.*mlp/(wi_0|wi_1|wo)"


def tunix_config(cfg: dict[str, Any]) -> dict[str, Any]:
    t = cfg.get("tunix") or {}
    unknown = sorted(set(t) - set(TUNIX_DEFAULTS))
    if unknown:
        raise KeyError(f"unknown tunix keys: {unknown} (known: {sorted(TUNIX_DEFAULTS)})")
    return {**copy.deepcopy(TUNIX_DEFAULTS), **copy.deepcopy(t)}


# --- MaxText model and arguments ----------------------------------------------------------------------------------
def maxtext_model(model_path: str, tcfg: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    """(MaxText model name, config overrides, HF config.json to patch into the conversion tables or None)."""
    hf = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    text = hf.get("text_config", hf)
    if tcfg["maxtext_model"] != "auto":
        return tcfg["maxtext_model"], {}, None
    if hf.get("model_type") == "qwen3_5":
        over = {mt: text[k] for k, mt in QWEN3_5_KEYS.items() if k in text}
        return "qwen3.5-9b", over, hf
    if hf.get("model_type") == "gemma4" and text.get("num_hidden_layers") == 60:
        return "gemma4-31b", {}, None
    raise ValueError(f"{model_path}: no MaxText mapping for model_type {hf.get('model_type')!r}; "
                     f"set tunix.maxtext_model to a MaxText model name")


def _fmt(v: Any) -> str:
    return str(v) if not isinstance(v, (list, dict)) else json.dumps(v)


def mt_args(name: str, overrides: dict[str, Any], extra: dict[str, Any]) -> list[str]:
    a: dict[str, Any] = {"model_name": name, "use_multimodal": False, "scan_layers": True,
                         "skip_jax_distributed_system": True}
    if overrides:
        a["override_model_config"] = True
        a.update(overrides)
    a.update(extra)
    return [f"{k}={_fmt(v)}" for k, v in a.items()]


def _patch(name: str, hf_config: dict[str, Any] | None) -> None:
    """Point MaxText's conversion table for ``name`` at this checkpoint's config.json (the 27B under qwen3.5-9b)."""
    if hf_config is None:
        return
    import transformers
    from maxtext.checkpoint_conversion.utils import hf_model_configs
    cls = {"qwen3_5": "Qwen3_5Config", "gemma4": "Gemma4Config"}.get(hf_config.get("model_type"))
    hf_model_configs.HF_MODEL_CONFIGS[name] = getattr(transformers, cls)(**hf_config)


def lora_module_path(name: str, tcfg: dict[str, Any]) -> str:
    if tcfg["lora_module_path"] != "auto":
        return tcfg["lora_module_path"]
    return QWEN3_5_MLP_LORA if name.startswith("qwen3.5") else ""   # "" = MaxText's lora_module_path.yml


# --- work dir, conversion, export (CPU subprocesses) --------------------------------------------------------------
def pick_work_dir(spec: str) -> Path:
    if spec != "auto":
        Path(spec).mkdir(parents=True, exist_ok=True)
        return Path(spec)
    best, free = None, -1
    for c in ("/kaggle/tmp", "/tmp", str(Path.home())):
        p = Path(c)
        try:
            p.mkdir(parents=True, exist_ok=True)
            f = shutil.disk_usage(p).free
        except OSError:
            continue
        if f > free:
            best, free = p, f
    return best / "sft_tunix"


def disk_report(paths: list[Path]) -> dict[str, float]:
    out = {}
    for p in paths:
        try:
            out[str(p)] = round(shutil.disk_usage(p).free / 2**30, 1)
        except OSError:
            pass
    return out


def _subprocess(mode: str, spec: dict[str, Any], log) -> None:
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=16"}
    cmd = [sys.executable, str(Path(__file__).resolve()), mode, json.dumps(spec)]
    log(f"{mode}: {' '.join(cmd[:3])} ...")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: list[str] = []
    for line in proc.stdout:   # stream: the conversion takes many minutes and must be visible in the Kaggle log
        print(line, end="", flush=True)
        tail = (tail + [line])[-60:]
    if proc.wait() != 0:
        raise RuntimeError(f"{mode} failed (exit {proc.returncode}):\n{''.join(tail)}")


def convert(model_path: str, name: str, overrides: dict[str, Any], hf_config: dict | None, dest: Path,
            keep: bool, log) -> Path:
    """HF safetensors -> MaxText Orbax checkpoint at dest/0/items (once; a marker records what was converted)."""
    marker = dest / "CONVERTED.json"
    key = hashlib.sha256(json.dumps([model_path, name, overrides], sort_keys=True).encode()).hexdigest()[:16]
    if keep and marker.exists() and json.loads(marker.read_text()).get("key") == key:
        log(f"converted checkpoint reused: {dest}")
        return dest / "0" / "items"
    shutil.rmtree(dest, ignore_errors=True)
    t = time.time()
    _subprocess("convert", {"name": name, "hf_config": hf_config, "hf_model_path": model_path,
                            "args": mt_args(name, overrides, {"base_output_directory": str(dest)})}, log)
    if not (dest / "0" / "items").exists():
        raise RuntimeError(f"conversion wrote no {dest / '0' / 'items'}")
    marker.write_text(json.dumps({"key": key, "model": model_path, "name": name, "seconds": round(time.time() - t)}))
    log(f"converted in {time.time() - t:.0f}s -> {dest}")
    return dest / "0" / "items"


def latest_lora_checkpoint(root: Path) -> Path | None:
    steps = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit() and (p / "model_params").exists()] \
        if root.exists() else []
    return max(steps, key=lambda p: int(p.name)) / "model_params" if steps else None


def export_adapter(cfg: dict[str, Any], model_path: str, name: str, overrides: dict, hf_config: dict | None,
                   lora_ckpt: Path, dest: Path, log) -> dict[str, Any]:
    shutil.rmtree(dest, ignore_errors=True)
    extra = {"base_output_directory": str(dest), "lora.lora_restore_path": str(lora_ckpt),
             "lora.lora_rank": cfg["lora_r"], "lora.lora_alpha": float(cfg["lora_alpha"])}
    _subprocess("export", {"name": name, "hf_config": hf_config, "hf_model_path": model_path,
                           "args": mt_args(name, overrides, extra)}, log)
    acfg_path = dest / "adapter_config.json"
    acfg = json.loads(acfg_path.read_text(encoding="utf-8"))
    acfg["base_model_name_or_path"] = cfg["model"]   # the export writes MaxText's HF id for the model name
    acfg_path.write_text(json.dumps(acfg, indent=2), encoding="utf-8")
    from safetensors.numpy import load_file
    w = load_file(str(dest / "adapter_model.safetensors"))
    ranks = {v.shape[0] for k, v in w.items() if k.endswith("lora_A.weight")} | \
            {v.shape[1] for k, v in w.items() if k.endswith("lora_B.weight")}
    return {"dir": str(dest), "tensors": len(w), "target_modules": acfg.get("target_modules"), "ranks": sorted(ranks),
            "mb": round((dest / "adapter_model.safetensors").stat().st_size / 2**20, 2)}


def _cpu_main(mode: str, spec: dict[str, Any]) -> None:
    import jax
    jax.config.update("jax_platforms", "cpu")
    _patch(spec["name"], spec["hf_config"])
    from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
    argv = [mode, os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml"), *spec["args"]]
    if mode == "convert":
        from maxtext.checkpoint_conversion import to_maxtext
        to_maxtext.main(args=argv, lazy_load_tensors=True, hf_model_path=spec["hf_model_path"],
                        simulated_cpu_devices_count=16)
    else:
        from absl import app
        from maxtext.checkpoint_conversion import to_huggingface
        app.run(to_huggingface.main, argv=[*argv, f"--hf_model_path={spec['hf_model_path']}"])


# --- batches ------------------------------------------------------------------------------------------------------
def make_batch(rows: list[base.Example], size: int, length: int, pad: int):
    """MaxText's SFT batch: targets are the inputs shifted by one; targets_segmentation is the loss mask (assistant
    tokens only), inputs_segmentation the attention segment (0 = padding). Fixed shape, so one compilation."""
    import numpy as np
    b = {k: np.zeros((size, length), np.int32) for k in
         ("inputs", "inputs_position", "inputs_segmentation", "targets", "targets_position", "targets_segmentation")}
    b["inputs"][:] = pad
    b["targets"][:] = pad
    for r, e in enumerate(rows):
        n = len(e.input_ids) - 1
        b["inputs"][r, :n] = e.input_ids[:-1]
        b["targets"][r, :n] = e.input_ids[1:]
        b["inputs_position"][r, :n] = np.arange(n)
        b["targets_position"][r, :n] = np.arange(n)
        b["inputs_segmentation"][r, :n] = 1
        b["targets_segmentation"][r, :n] = [int(l != base.IGNORE) for l in e.labels[1:]]
    return b


class Batches:
    """Epochs of shuffled fixed-shape batches; stops at the deadline (the trainer then saves and exits cleanly). On a
    resume Tunix skips the batches it already trained on, so the order must be the same every run (seeded)."""

    def __init__(self, examples, size: int, length: int, pad: int, epochs: int, seed: int, deadline: float | None,
                 max_steps: int | None):
        self.examples, self.size, self.length, self.pad = examples, size, length, pad
        self.epochs, self.seed, self.deadline, self.max_steps = epochs, seed, deadline, max_steps
        self.hit_deadline = False

    def steps(self) -> int:
        per_epoch = math.ceil(len(self.examples) / self.size)
        return self.max_steps or per_epoch * self.epochs

    def __iter__(self):
        rng, n = random.Random(self.seed), 0
        for _ in range(self.epochs if not self.max_steps else 10**9):
            order = list(range(len(self.examples)))
            rng.shuffle(order)
            for i in range(0, len(order), self.size):
                if n >= self.steps():
                    return
                if self.deadline and time.time() > self.deadline:
                    self.hit_deadline = True
                    return
                n += 1
                yield make_batch([self.examples[j] for j in order[i:i + self.size]], self.size, self.length, self.pad)


# --- TPU side -----------------------------------------------------------------------------------------------------
def environment() -> dict[str, Any]:
    import importlib.metadata as md
    import jax
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "jax": jax.__version__,
           "backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
           "device_kind": jax.devices()[0].device_kind, "device_count": jax.device_count()}
    for pkg in ("maxtext", "google-tunix", "qwix", "flax", "transformers"):
        try:
            env[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            env[pkg] = None
    return env


def evaluate(model, mt_config, mesh, examples, size: int, length: int, pad: int) -> dict[str, Any]:
    """Mean cross-entropy over the target tokens (MaxText's own loss, eval mode)."""
    import jax
    from flax import nnx
    from flax.core.spmd import logical_axis_rules
    from maxtext.trainers.pre_train.train import loss_fn
    from tunix.sft import sharding_utils

    @nnx.jit
    def step(m, data):
        _, aux = loss_fn(m, mt_config, data, None, None, is_train=False)
        return aux["xent_sum"], aux["total_weights"]

    xent, tokens = 0.0, 0.0
    with jax.set_mesh(mesh), logical_axis_rules(mt_config.logical_axis_rules):
        for i in range(0, len(examples), size):
            data = sharding_utils.shard_input(make_batch(examples[i:i + size], size, length, pad),
                                              tuple(mt_config.data_sharding))
            x, w = step(model, data)
            xent, tokens = xent + float(x), tokens + float(w)
    return {"loss": xent / max(tokens, 1.0), "tokens": int(tokens)}


def lora_stats(model) -> dict[str, Any]:
    import jax
    import numpy as np
    from flax import nnx
    state = nnx.state(model, nnx.LoRAParam)
    leaves = jax.tree_util.tree_leaves_with_path(state)
    params = sum(int(np.prod(v.shape)) for _, v in leaves)
    b_max = max((float(jax.numpy.abs(v).max()) for p, v in leaves if "lora_b" in jax.tree_util.keystr(p)),
                default=0.0)
    return {"trainable_params": params, "lora_tensors": len(leaves), "max_abs_lora_b": b_max}


def loss_hooks(log, every: int):
    """Tunix training hooks keeping the loss history and step times for the card and the plot (built here: tunix
    is imported only on the TPU side)."""
    from tunix.sft.hooks import TrainingHooks

    class LossHooks(TrainingHooks):
        def __init__(self):
            self.every, self.history, self.t = max(int(every), 1), [], time.time()

        def on_train_start(self, train_ctx):
            self.t = time.time()

        def on_train_end(self, train_ctx):
            pass

        def on_train_step_start(self, train_ctx):
            pass

        def on_train_step_end(self, train_ctx, train_step, train_loss, step_time=0.0):
            now = time.time()
            h = {"step": int(train_step), "loss": float(train_loss), "seconds": round(now - self.t, 2)}
            self.t = now
            self.history.append(h)
            if h["step"] % self.every == 0:
                log(f"step {h['step']} loss {h['loss']:.4f} ({h['seconds']}s)")

        def on_eval_step_start(self, train_ctx):
            pass

        def on_eval_step_end(self, train_ctx, eval_loss):
            pass

    return LossHooks()


def _run(cfg: dict[str, Any], card: dict[str, Any], out: Path, log) -> None:
    tcfg = tunix_config(cfg)
    work = pick_work_dir(tcfg["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(work / "hf"))   # the 27B must not land in /kaggle/working
    import jax
    card["environment"] = env = environment()
    card["disk_free_gb"] = disk_report([work, out])
    log(f"environment {json.dumps(env)}; work dir {work}; free GB {card['disk_free_gb']}")
    if env["backend"] != "tpu":
        log(f"WARNING: jax backend is {env['backend']}, not tpu")

    t = time.time()
    path = base.resolve_model(cfg["model"])
    card["model_path"] = path
    log(f"model files at {path} ({time.time() - t:.0f}s)")
    tok = base.load_tokenizer(path, cfg)
    name, overrides, hf_config = maxtext_model(path, tcfg)
    card["maxtext"] = {"model_name": name, "overrides": overrides, "lora_module_path": lora_module_path(name, tcfg)}
    log(f"MaxText model {name} overrides {overrides}")

    # data: the GPU backend's own rendering and loss masks
    stats = base.Stats()
    train_convs = base.load_conversations(cfg["data_path"], cfg, stats, cfg["max_samples"])
    eval_convs = base.load_conversations(cfg["eval_path"], cfg, base.Stats(), None) if cfg["eval_path"] else None
    cached = None
    if cfg["examples_cache"]:
        key = base.cache_key(cfg, tok)
        cached = base.load_examples(Path(cfg["examples_cache"]), key)
    if cached is not None:
        train_ex, eval_ex, stats = cached
        log(f"examples from cache {cfg['examples_cache']} ({len(train_ex)} train)")
    else:
        train_ex = [e for c in train_convs for e in base.build_examples(tok, c, cfg, stats)]
        eval_ex = [e for c in eval_convs for e in base.build_examples(tok, c, cfg, base.Stats())] if eval_convs else None
        if cfg["examples_cache"]:
            base.save_examples(Path(cfg["examples_cache"]), key, train_ex, eval_ex, stats)
    eval_ex = eval_ex if eval_ex is not None else train_ex
    if cfg["eval_max_examples"] is not None:
        eval_ex = eval_ex[: cfg["eval_max_examples"]]
    card["data"] = asdict(stats)
    card["data"]["eval_examples"] = len(eval_ex)
    log(f"data {json.dumps({k: v for k, v in asdict(stats).items() if k != 'errors'})}")
    base.check(card, "data", bool(train_ex) and stats.skipped_conversations == 0,
               f"{len(train_ex)} examples from {stats.conversations} conversations, "
               f"{stats.skipped_conversations} rejected")
    base.check(card, "loss_mask", stats.boundary_straddles == 0 and all(e.n_target > 0 for e in train_ex),
               f"{stats.target_tokens} target of {stats.total_tokens} tokens; straddles {stats.boundary_straddles}")
    if not train_ex:
        raise RuntimeError("no training examples")

    # MaxText config first: a bad key fails here, before the long conversion
    from maxtext.configs import pyconfig
    from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
    conv_dir = work / f"maxtext_{name}_{hashlib.sha256(path.encode()).hexdigest()[:8]}"
    size = int(tcfg["per_device_batch"] * jax.device_count())
    steps = Batches(train_ex, size, cfg["max_seq_len"], tok.pad_token_id, cfg["epochs"], cfg["seed"], None,
                    cfg["max_steps"]).steps()
    extra = {
        "run_name": "sft", "base_output_directory": str(work / "maxtext_run"),
        "load_parameters_path": str(conv_dir / "0" / "items"),
        "tokenizer_path": path, "tokenizer_type": "huggingface",
        "per_device_batch_size": tcfg["per_device_batch"], "max_target_length": cfg["max_seq_len"],
        "steps": steps, "learning_rate": cfg["lr"], "warmup_steps_fraction": cfg["warmup_ratio"],
        "adam_weight_decay": cfg["weight_decay"], "gradient_clipping_threshold": cfg["max_grad_norm"] or 0.0,
        "gradient_accumulation_steps": cfg["grad_accum"], "use_tunix_gradient_accumulation": True,
        "weight_dtype": "bfloat16", "dtype": "bfloat16", "remat_policy": tcfg["remat_policy"],
        "num_vocab_tiling": tcfg["vocab_tiling"], "ici_fsdp_parallelism": tcfg["fsdp"],
        "ici_tensor_parallelism": tcfg["tensor"], "packing": False, "eval_interval": -1,
        "enable_checkpointing": False, "enable_goodput_recording": False,
        "lora.enable_lora": True, "lora.lora_rank": cfg["lora_r"], "lora.lora_alpha": float(cfg["lora_alpha"]),
        **tcfg["maxtext_overrides"],
    }
    if lora_module_path(name, tcfg):   # left out = MaxText's table ("key=" would parse as None and fail validation)
        extra["lora.lora_module_path"] = lora_module_path(name, tcfg)
    if hf_config is not None:
        _patch(name, hf_config)
    mt_config = pyconfig.initialize(["sft", os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "sft.yml"),
                                     *mt_args(name, overrides, extra)])
    card["maxtext"]["global_batch"] = size
    card["maxtext"]["steps"] = steps

    convert(path, name, overrides, hf_config, conv_dir, tcfg["keep_converted"], log)
    card["disk_free_gb_after_convert"] = disk_report([work])

    import optax
    from flax import nnx
    from flax.core.spmd import logical_axis_rules
    from orbax import checkpoint as ocp
    from maxtext.optimizers import optimizers
    from maxtext.trainers.post_train.sft import train_sft
    from maxtext.utils import lora_utils, maxtext_utils, model_creation_utils
    from tunix.sft import peft_trainer

    t = time.time()
    model, mesh = model_creation_utils.from_pretrained(mt_config)
    card["load"] = {"seconds": round(time.time() - t), "mesh": dict(mesh.shape)}
    log(f"loaded {name} on mesh {dict(mesh.shape)} in {time.time() - t:.0f}s")
    base.check(card, "weights", True, f"MaxText restore of {conv_dir.name}")
    with jax.set_mesh(mesh), logical_axis_rules(mt_config.logical_axis_rules):
        model = lora_utils.apply_lora_to_model(model, mesh, mt_config)
    lo = lora_stats(model)
    card["lora"] = lo
    log(f"LoRA: {lo['lora_tensors']} tensors, {lo['trainable_params']:,} trainable")
    base.check(card, "trainable", lo["trainable_params"] > 0, f"{lo['trainable_params']:,} params")

    pad = tok.pad_token_id
    card["eval_before"] = evaluate(model, mt_config, mesh, eval_ex, size, cfg["max_seq_len"], pad)
    log(f"eval before: loss {card['eval_before']['loss']:.4f} over {card['eval_before']['tokens']} tokens")

    ckpt_root = out / "checkpoint"
    if tcfg["resume_from"] and not ckpt_root.exists():
        shutil.copytree(tcfg["resume_from"], ckpt_root)
        log(f"resuming from a copy of {tcfg['resume_from']}")
    with jax.set_mesh(mesh), logical_axis_rules(mt_config.logical_axis_rules):
        schedule = maxtext_utils.create_learning_rate_schedule(mt_config)
        optimizer = optimizers.get_optimizer(mt_config, schedule, model)
        if mt_config.gradient_clipping_threshold > 0:
            optimizer = optax.chain(optax.clip_by_global_norm(max_norm=mt_config.gradient_clipping_threshold), optimizer)
        tconf = peft_trainer.TrainingConfig(
            eval_every_n_steps=10**9, max_steps=steps,
            gradient_accumulation_steps=cfg["grad_accum"] if cfg["grad_accum"] > 1 else None,
            checkpoint_root_directory=str(ckpt_root),
            checkpointing_options=ocp.CheckpointManagerOptions(
                save_interval_steps=cfg["save_every"] or steps, max_to_keep=2, enable_async_checkpointing=False),
            data_sharding_axis=tuple(mt_config.data_sharding))
        nnx.pop(model, nnx.Intermediate)
        trainer = train_sft.MaxTextPeftTrainer(model, optimizer, tconf)
        hooks = loss_hooks(log, cfg["log_every"])
        trainer.with_training_hooks(hooks)
        trainer = train_sft.use_maxtext_loss_function(trainer, mt_config)
        start_step = trainer.train_steps
        deadline = time.time() + cfg["max_train_hours"] * 3600 if cfg["max_train_hours"] else None
        batches = Batches(train_ex, size, cfg["max_seq_len"], pad, cfg["epochs"], cfg["seed"], deadline,
                          cfg["max_steps"])
        log(f"training {steps} steps of {size} rows x {cfg['max_seq_len']} tokens (from step {start_step})")
        t = time.time()
        trainer.train(batches, None, cache_nnx_graph=True)
        trainer.close()   # writes the last checkpoint
        secs = time.time() - t
    hist = hooks.history
    card["training"] = {"steps": len(hist), "start_step": start_step, "total_steps": steps, "seconds": round(secs),
                        "hit_deadline": batches.hit_deadline, "history": hist,
                        "rows_per_s": round(len(hist) * size / max(secs, 1e-9), 3)}
    log(f"trained {len(hist)} steps in {secs:.0f}s{' (stopped at the time limit)' if batches.hit_deadline else ''}")
    card["eval_after"] = evaluate(model, mt_config, mesh, eval_ex, size, cfg["max_seq_len"], pad)
    log(f"eval after: loss {card['eval_after']['loss']:.4f}")
    lo_after = lora_stats(model)
    base.check(card, "finite", all(math.isfinite(h["loss"]) for h in hist), "")
    base.check(card, "adapter_updated", lo_after["max_abs_lora_b"] > 0, f"max |lora_b| = {lo_after['max_abs_lora_b']:.3g}")
    b, a = card["eval_before"]["loss"], card["eval_after"]["loss"]
    base.check(card, "loss_dropped", a <= b * (1 - cfg["min_loss_drop"]) and a < b, f"{b:.4f} -> {a:.4f}")
    card["plot"] = "loss.png" if base.write_plot(hist, out / "loss.png") else None

    lora_ckpt = latest_lora_checkpoint(ckpt_root)
    card["lora_checkpoint"] = str(lora_ckpt) if lora_ckpt else None
    base.check(card, "checkpoint_saved", lora_ckpt is not None, str(lora_ckpt))
    if tcfg["export_hf"] and lora_ckpt is not None:
        del model, trainer
        card["adapter"] = ad = export_adapter(cfg, path, name, overrides, hf_config, lora_ckpt, out / "adapter", log)
        tok.save_pretrained(out / "adapter")
        (out / "adapter" / "sft_config.json").write_text(json.dumps(cfg, indent=1, default=str), encoding="utf-8")
        base.check(card, "adapter_saved", ad["tensors"] > 0 and ad["ranks"] == [cfg["lora_r"]],
                   f"{ad['tensors']} tensors, ranks {ad['ranks']}, modules {ad['target_modules']}, {ad['mb']} MB")
        log(f"adapter exported: {ad}")


def render_card(card: dict[str, Any]) -> str:
    c, e = card["config"], card.get("environment", {})
    lines = [f"# LoRA SFT (TPU, Tunix + MaxText) result card — {card['verdict']}", "",
             f"run `{card['run_id']}` · model `{c['model']}` · {e.get('device_kind', '?')} x{e.get('device_count', '?')}"
             f" · {card.get('seconds_total')}s total", "", "## Checks", "", "| check | result | detail |", "|---|---|---|"]
    lines += [f"| {k} | {'PASS' if v['pass'] else 'FAIL'} | {v.get('detail', '')} |" for k, v in card["checks"].items()]
    if card.get("error"):
        lines += ["", "## Error", "", "```", card["error"][-3000:], "```"]
    ev0, ev1 = card.get("eval_before"), card.get("eval_after")
    if ev0 and ev1:
        lines += ["", "## Evaluation (target tokens only)", "", "| | loss | tokens |", "|---|---|---|",
                  f"| before | {ev0['loss']:.4f} | {ev0['tokens']} |", f"| after | {ev1['loss']:.4f} | {ev1['tokens']} |"]
    t = card.get("training")
    if t and t["history"]:
        lines += ["", "## Training", "", f"{t['steps']} steps (from {t['start_step']}, of {t['total_steps']}), "
                  f"{t['seconds']}s, {t['rows_per_s']} rows/s; first loss {t['history'][0]['loss']:.4f}, last "
                  f"{t['history'][-1]['loss']:.4f}{'; stopped at the time limit' if t['hit_deadline'] else ''}."]
        if card.get("plot"):
            lines += ["", f"![loss]({card['plot']})"]
    for k in ("maxtext", "lora", "adapter", "data", "disk_free_gb", "disk_free_gb_after_convert"):
        if card.get(k):
            lines += ["", f"## {k}", "", "```json", json.dumps(card[k], indent=1, default=str)[:4000], "```"]
    lines += ["", "## Environment", "", "```json", json.dumps(e, indent=1), "```", "",
              "## Config", "", "```json", json.dumps(c, indent=1, default=str), "```", ""]
    return "\n".join(lines)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    tunix_config(cfg)   # unknown tunix keys fail before anything runs
    return base.run_with_card(cfg, _run, render_card)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in ("convert", "export"):
        _cpu_main(sys.argv[1], json.loads(sys.argv[2]))
    else:
        sys.exit("usage: lora_sft_tunix.py convert|export <json spec>  (training: lora_sft.run({'backend': 'tunix'}))")
