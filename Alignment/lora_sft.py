"""LoRA SFT on chat / tool-call trajectories. One file; needs only torch, transformers and peft (no TRL: fewer moving
parts to break offline on Kaggle).

    python lora_sft.py --config smoke.json [--set key=value ...]
    from lora_sft import run; card = run({...})

Pipeline: resolve model -> load + validate data -> render with the model's own chat template and mask the loss to
assistant tokens -> load the declared architecture class (refusing silently missing weights) -> attach LoRA to the
language model's linear layers only -> evaluate (loss, token accuracy, greedy generation) -> train -> evaluate again ->
save the adapter -> reload it into a fresh base and re-evaluate -> write ``result_card.json`` / ``result_card.md`` /
``loss.png`` with a PASS/FAIL verdict built from explicit checks.

Data: JSONL, one conversation per line. Two shapes are accepted:
  * ``{"messages": [...], "tools": [...]}`` in OpenAI chat format, optionally with a system message first.
  * Prime's ``sft_levels.jsonl`` rows (``Sarbloh/harness/trace.py``): ``system`` and ``tools`` beside ``messages``.
Assistant messages may carry ``reasoning_content``, ``tool_calls`` (arguments as dict or JSON string) and
``"train": false`` to keep a turn as context but out of the loss. Image parts are replaced by a text placeholder
(``images: "placeholder"``) or rejected (``"error"``); vision training is out of scope.

Masking: every assistant turn is a target. ``mask_mode: "auto"`` renders the whole conversation once when the
template is prefix-stable (each assistant turn renders identically inside the full conversation) and otherwise falls
back to one example per assistant turn (``"per_turn"``), which is exact for any template (Qwen3.x drops earlier turns'
``<think>`` blocks, so its renders are not prefix-stable across user turns). The loss covers what the model would
generate after the template's generation prompt, including the end-of-turn token.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import difflib
import gc
import json
import math
import os
import platform
import random
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

IGNORE = -100

DEFAULTS: dict[str, Any] = {
    # model
    "model": "Qwen/Qwen3.5-2B",       # HF repo id (downloaded, needs internet) or a local directory (offline)
    "model_class": "auto",            # "auto" = config.architectures[0]; or a transformers class name
    "dtype": "auto",                  # auto | bf16 | fp16 | fp32. auto: bf16 on sm80+, else fp32 (fp16 overflows Qwen)
    "load_in_4bit": False,            # QLoRA via bitsandbytes
    "trust_remote_code": False,
    "allow_missing_weights": [r"(^|\.)mtp\.", r"lm_head"],  # regexes; any other missing weight is a hard FAIL
    "attn_implementation": None,      # None = transformers default
    # data
    "data_path": "data/smoke3.jsonl",
    "eval_path": None,                # None = evaluate on the training examples (the smoke test's overfit check)
    "max_samples": None,              # conversations taken from data_path (in file order)
    "filter_cleared": False,          # Prime rows: keep level_cleared only
    "images": "placeholder",          # placeholder | error
    "max_seq_len": 4096,              # longer examples lose context from the left; targets are never cut
    "mask_mode": "auto",              # auto | whole | per_turn
    "turns": "all",                   # all | last (only the last assistant turn of each conversation is a target)
    "template_kwargs": {},            # passed to apply_chat_template, e.g. {"enable_thinking": true}
    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                     "in_proj_qkvz", "in_proj_ba", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                     "out_proj"],     # leaf names; only layers outside the vision tower are adapted. "all-linear" ok
    "lora_exclude": [r"visual", r"vision", r"(^|\.)mtp\.", r"lm_head", r"embed"],
    # optimisation
    "epochs": 3,
    "max_steps": None,                # overrides epochs
    "lr": 2e-4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.0,
    "batch_size": 1,
    "grad_accum": 1,
    "max_grad_norm": 1.0,
    "gradient_checkpointing": True,
    "seed": 0,
    "log_every": 1,
    # evaluation and checks
    "gen_samples": 3,                 # greedy generations before/after on the eval set (0 = off)
    "gen_max_new_tokens": 256,
    "reload_check": True,             # save -> reload into a fresh base -> eval loss must match
    "reload_tol": 0.02,               # relative tolerance on the reloaded eval loss
    "min_loss_drop": 0.0,             # PASS needs eval loss after <= before * (1 - min_loss_drop)
    "min_gen_similarity": None,       # PASS needs mean after-training similarity >= this (None = report only)
    # output
    "output_dir": "sft_out",
    "save_merged": False,             # also write base+LoRA merged weights (large)
}


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]
    source: str                       # conversation id and turn(s)
    n_target: int = 0


@dataclass
class Stats:
    conversations: int = 0
    skipped_conversations: int = 0
    examples: int = 0
    target_tokens: int = 0
    total_tokens: int = 0
    max_len: int = 0
    whole_renders: int = 0
    per_turn_renders: int = 0
    left_truncated: int = 0
    dropped_too_long: int = 0
    dropped_no_target: int = 0
    images_replaced: int = 0
    header_in_loss: int = 0           # turns where no generation prompt matched: the role header is trained too
    boundary_straddles: int = 0       # tokens spanning prompt/target: excluded from the loss, should be 0
    tokenization_mismatches: int = 0  # prompt tokenised alone != prefix of the full tokenisation
    errors: list[str] = field(default_factory=list)


class DataError(ValueError):
    pass


# --- config -------------------------------------------------------------------------------------------------------
def make_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    unknown = sorted(set(overrides or {}) - set(cfg))
    if unknown:
        raise KeyError(f"unknown config keys: {unknown} (known: {sorted(cfg)})")
    cfg.update(copy.deepcopy(overrides or {}))
    if cfg["mask_mode"] not in ("auto", "whole", "per_turn"):
        raise ValueError(f"mask_mode {cfg['mask_mode']!r}")
    if cfg["turns"] not in ("all", "last"):
        raise ValueError(f"turns {cfg['turns']!r}")
    if cfg["images"] not in ("placeholder", "error"):
        raise ValueError(f"images {cfg['images']!r}")
    return cfg


def _parse_set(items: list[str]) -> dict[str, Any]:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects key=value, got {item!r}")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


# --- data ---------------------------------------------------------------------------------------------------------
def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise DataError(f"{path}:{n}: not JSON ({exc})") from exc
    return rows


def _text_of(content: Any, cfg: dict[str, Any], stats: Stats, where: str) -> str | None:
    """Content as a string (None stays None). Text parts are joined; image parts follow cfg['images']."""
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise DataError(f"{where}: content must be a string, null or a list of parts")
    out = []
    for p in content:
        kind = p.get("type") if isinstance(p, dict) else None
        if kind == "text":
            out.append(str(p.get("text", "")))
        elif kind in ("image_url", "image") or (isinstance(p, dict) and ("image" in p or "image_url" in p)):
            if cfg["images"] == "error":
                raise DataError(f"{where}: image part (set images='placeholder' to train text-only)")
            stats.images_replaced += 1
            out.append("[image]")
        else:
            raise DataError(f"{where}: unknown content part {p!r:.80}")
    return "".join(out)


def normalize_record(row: dict[str, Any], idx: int, cfg: dict[str, Any], stats: Stats) -> dict[str, Any]:
    """Validate one conversation and return {"id", "tools", "messages"} in the shape chat templates expect."""
    rid = str(row.get("id") or (f"{row['game_id']}/L{row['level']}" if "game_id" in row else f"row{idx}"))
    if not isinstance(row.get("messages"), list) or not row["messages"]:
        raise DataError(f"{rid}: no messages")
    tools = row.get("tools") or None
    tool_names: set[str] = set()
    required: dict[str, list[str]] = {}
    if tools is not None:
        if not isinstance(tools, list):
            raise DataError(f"{rid}: tools must be a list")
        for t in tools:
            fn = t.get("function", t) if isinstance(t, dict) else None
            if not isinstance(fn, dict) or not fn.get("name"):
                raise DataError(f"{rid}: malformed tool schema {t!r:.80}")
            tool_names.add(fn["name"])
            required[fn["name"]] = list((fn.get("parameters") or {}).get("required") or [])
    msgs: list[dict[str, Any]] = []
    if row.get("system") and row["messages"][0].get("role") != "system":
        msgs.append({"role": "system", "content": _text_of(row["system"], cfg, stats, f"{rid} system")})
    open_calls = 0
    for i, m in enumerate(row["messages"]):
        where = f"{rid} message {i}"
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise DataError(f"{where}: role {role!r}")
        if role == "system" and (i > 0 or msgs):
            raise DataError(f"{where}: system message not first")
        out: dict[str, Any] = {"role": role, "content": _text_of(m.get("content"), cfg, stats, where) or ""}
        if role == "assistant":
            if m.get("reasoning_content"):
                out["reasoning_content"] = str(m["reasoning_content"])
            calls = []
            for c in m.get("tool_calls") or []:
                fn = c.get("function", c)
                name, args = fn.get("name"), fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError as exc:
                        raise DataError(f"{where}: tool call {name!r} arguments are not JSON") from exc
                if not isinstance(args, dict):
                    raise DataError(f"{where}: tool call {name!r} arguments must be an object")
                if tools is not None and name not in tool_names:
                    raise DataError(f"{where}: tool {name!r} not in tools {sorted(tool_names)}")
                missing = [k for k in required.get(name, []) if k not in args]
                if missing:
                    raise DataError(f"{where}: tool call {name!r} misses required {missing}")
                call = {"type": "function", "function": {"name": name, "arguments": args}}
                if c.get("id"):
                    call["id"] = c["id"]
                calls.append(call)
            if calls:
                out["tool_calls"] = calls
            if not out["content"] and not calls and not out.get("reasoning_content"):
                raise DataError(f"{where}: empty assistant message")
            out["train"] = bool(m.get("train", True))
            open_calls = len(calls)
        elif role == "tool":
            prev = msgs[-1]["role"] if msgs else None
            if prev not in ("assistant", "tool") or open_calls <= 0:
                raise DataError(f"{where}: tool result without a pending tool call")
            open_calls -= 1
            if m.get("tool_call_id"):
                out["tool_call_id"] = m["tool_call_id"]
            if m.get("name"):
                out["name"] = m["name"]
        msgs.append(out)
    first_asst = next((i for i, m in enumerate(msgs) if m["role"] == "assistant"), None)
    if first_asst is None:
        raise DataError(f"{rid}: no assistant message")
    if not any(m["role"] == "user" for m in msgs[:first_asst]):
        raise DataError(f"{rid}: no user message before the first assistant message")
    return {"id": rid, "tools": tools, "messages": msgs}


def load_conversations(path: str | Path, cfg: dict[str, Any], stats: Stats, limit: int | None) -> list[dict]:
    rows = load_jsonl(path)
    if cfg["filter_cleared"]:
        rows = [r for r in rows if r.get("level_cleared", True)]
    convs = []
    for i, row in enumerate(rows):
        if limit is not None and len(convs) >= limit:
            break
        try:
            convs.append(normalize_record(row, i, cfg, stats))
        except DataError as exc:
            stats.skipped_conversations += 1
            stats.errors.append(str(exc))
    stats.conversations += len(convs)
    return convs


# --- rendering and masking ----------------------------------------------------------------------------------------
def _for_template(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in m.items() if k != "train"} for m in msgs]


def render(tok, msgs, tools, gen: bool, kwargs: dict[str, Any]) -> str:
    return tok.apply_chat_template(_for_template(msgs), tools=tools, tokenize=False, add_generation_prompt=gen,
                                   **kwargs)


def _gen_candidates(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    base = dict(cfg["template_kwargs"])
    if "enable_thinking" in base:
        return [base]
    return [base, {**base, "enable_thinking": True}, {**base, "enable_thinking": False}]


def target_span(tok, msgs, tools, i: int, cfg: dict[str, Any], stats: Stats) -> tuple[str, int, dict]:
    """Render msgs[:i+1] (msgs[i] is an assistant turn). Returns (text, target_start_char, gen_kwargs): the target is
    text[start:], i.e. what the model emits after the longest generation prompt that is a prefix of the render."""
    full = render(tok, msgs[: i + 1], tools, False, cfg["template_kwargs"])
    best, best_kw = None, None
    for kw in _gen_candidates(cfg):
        try:
            p = render(tok, msgs[:i], tools, True, kw)
        except Exception:  # noqa: BLE001 - a template that rejects a kwarg: try the next candidate
            continue
        if full.startswith(p) and (best is None or len(p) > len(best)):
            best, best_kw = p, kw
    if best is None:
        p = render(tok, msgs[:i], tools, False, cfg["template_kwargs"])
        if not full.startswith(p):
            raise DataError(f"turn {i}: the template renders the history differently once the turn is added")
        stats.header_in_loss += 1
        best, best_kw = p, dict(cfg["template_kwargs"])
    return full, len(best), best_kw


def tokenize_spans(tok, text: str, spans: list[tuple[int, int]], stats: Stats) -> tuple[list[int], list[int]]:
    """Tokenise text once; label the tokens that start inside a target span [a, b)."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    labels = [IGNORE] * len(ids)
    for a, b in spans:
        for t, (s, e) in enumerate(offs):
            if a <= s < b:
                labels[t] = ids[t]
            elif s < a < e:
                stats.boundary_straddles += 1
        prompt_ids = tok(text[:a], add_special_tokens=False)["input_ids"]
        if ids[: len(prompt_ids)] != prompt_ids:
            stats.tokenization_mismatches += 1
    return ids, labels


def _fit(ids: list[int], labels: list[int], max_len: int, stats: Stats) -> tuple[list[int], list[int]] | None:
    if len(ids) <= max_len:
        return ids, labels
    first = next((t for t, l in enumerate(labels) if l != IGNORE), len(ids))
    if len(ids) - first > max_len:
        stats.dropped_too_long += 1
        return None
    stats.left_truncated += 1
    return ids[-max_len:], labels[-max_len:]


def build_examples(tok, conv: dict[str, Any], cfg: dict[str, Any], stats: Stats) -> list[Example]:
    msgs, tools, cid = conv["messages"], conv["tools"], conv["id"]
    turns = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and m.get("train", True)]
    if cfg["turns"] == "last":
        turns = turns[-1:]
    if not turns:
        stats.dropped_no_target += 1
        return []
    raw: list[tuple[list[int], list[int], str]] = []
    whole_ok = False
    if cfg["mask_mode"] in ("auto", "whole"):
        full = render(tok, msgs, tools, False, cfg["template_kwargs"])
        spans, whole_ok = [], True
        for i in turns:
            text, start, _ = target_span(tok, msgs, tools, i, cfg, stats)
            if not full.startswith(text):
                whole_ok = False
                break
            spans.append((start, len(text)))
        if whole_ok:
            ids, labels = tokenize_spans(tok, full, spans, stats)
            raw.append((ids, labels, f"{cid}#whole"))
            stats.whole_renders += 1
        elif cfg["mask_mode"] == "whole":
            raise DataError(f"{cid}: mask_mode 'whole' but the template is not prefix-stable here")
    if not whole_ok:
        for i in turns:
            text, start, _ = target_span(tok, msgs, tools, i, cfg, stats)
            ids, labels = tokenize_spans(tok, text, [(start, len(text))], stats)
            raw.append((ids, labels, f"{cid}#t{i}"))
            stats.per_turn_renders += 1
    out = []
    for ids, labels, src in raw:
        fitted = _fit(ids, labels, cfg["max_seq_len"], stats)
        if fitted is None:
            continue
        ids, labels = fitted
        n = sum(1 for l in labels if l != IGNORE)
        if n == 0:
            stats.dropped_no_target += 1
            continue
        out.append(Example(ids, labels, src, n))
        stats.examples += 1
        stats.target_tokens += n
        stats.total_tokens += len(ids)
        stats.max_len = max(stats.max_len, len(ids))
    return out


def generation_prompt(tok, conv: dict[str, Any], cfg: dict[str, Any], stats: Stats) -> tuple[str, str] | None:
    """(prompt, reference) for the conversation's last trained assistant turn."""
    msgs = conv["messages"]
    turns = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and m.get("train", True)]
    if not turns:
        return None
    text, start, _ = target_span(tok, msgs, conv["tools"], turns[-1], cfg, stats)
    return text[:start], text[start:]   # text[:start] is exactly the generation prompt the target was masked after


# --- environment and model ----------------------------------------------------------------------------------------
def environment() -> dict[str, Any]:
    import torch
    import transformers
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
           "transformers": transformers.__version__, "cuda": torch.cuda.is_available()}
    try:
        import peft
        env["peft"] = peft.__version__
    except ImportError:
        env["peft"] = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update(gpu=p.name, gpu_count=torch.cuda.device_count(), vram_gb=round(p.total_memory / 2**30, 1),
                   capability=f"{p.major}.{p.minor}", torch_cuda=torch.version.cuda)
    return env


def pick_dtype(name: str):
    import torch
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
            return torch.bfloat16
        return torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def resolve_model(name: str) -> str:
    if Path(name).is_dir():
        return str(Path(name))
    from huggingface_hub import snapshot_download
    return snapshot_download(name, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model",
                                                   "*.tiktoken", "tokenizer*"])


def load_tokenizer(path: str, cfg: dict[str, Any]):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=cfg["trust_remote_code"])
    if not getattr(tok, "chat_template", None):
        try:  # multimodal checkpoints sometimes keep the template on the processor
            from transformers import AutoProcessor
            tok.chat_template = AutoProcessor.from_pretrained(path).chat_template
        except Exception:  # noqa: BLE001
            pass
    if not getattr(tok, "chat_template", None):
        raise RuntimeError(f"{path}: tokenizer has no chat template")
    if not getattr(tok, "is_fast", False):
        raise RuntimeError(f"{path}: a fast tokenizer is required (offset mapping for loss masks)")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(path: str, cfg: dict[str, Any], dtype) -> tuple[Any, dict[str, Any]]:
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM
    mcfg = AutoConfig.from_pretrained(path, trust_remote_code=cfg["trust_remote_code"])
    arch = (getattr(mcfg, "architectures", None) or [None])[0]
    name = arch if cfg["model_class"] == "auto" else cfg["model_class"]
    cls = getattr(transformers, name, None) if name else None
    if cls is None:
        cls = AutoModelForCausalLM
    kw: dict[str, Any] = {"trust_remote_code": cfg["trust_remote_code"], "output_loading_info": True,
                          "low_cpu_mem_usage": True}
    # transformers >= 4.56 takes `dtype`; older versions take `torch_dtype` (and would ignore `dtype` silently)
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    kw["dtype" if (major, minor) >= (4, 56) else "torch_dtype"] = dtype
    if cfg["attn_implementation"]:
        kw["attn_implementation"] = cfg["attn_implementation"]
    if cfg["load_in_4bit"]:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
    model, info = cls.from_pretrained(path, **kw)
    missing = list(info.get("missing_keys") or [])
    allowed = [re.compile(p) for p in cfg["allow_missing_weights"]]
    bad = [k for k in missing if not any(p.search(k) for p in allowed)]
    if not cfg["load_in_4bit"] and torch.cuda.is_available():
        model.to("cuda")
    model.config.use_cache = False
    load = {"class": type(model).__name__, "architecture": arch, "missing_keys": missing[:50],
            "unexpected_keys": list(info.get("unexpected_keys") or [])[:50], "missing_not_allowed": bad[:50],
            "params": sum(p.numel() for p in model.parameters()),
            "param_dtypes": sorted({str(p.dtype) for p in model.parameters()})}
    return model, load


def lora_target_regex(model, cfg: dict[str, Any]) -> tuple[str, list[str]]:
    import torch.nn as nn
    exclude = [re.compile(p) for p in cfg["lora_exclude"]]
    leaves = cfg["lora_targets"]
    names = []
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear) and type(m).__name__ not in ("Linear4bit", "Linear8bitLt"):
            continue
        if any(p.search(n) for p in exclude):
            continue
        if leaves != "all-linear" and n.rsplit(".", 1)[-1] not in leaves:
            continue
        names.append(n)
    if not names:
        raise RuntimeError("no LoRA target layers matched lora_targets / lora_exclude")
    return "^(" + "|".join(re.escape(n) for n in names) + ")$", names


def guard_peft_torchao(log=print) -> None:
    """peft raises ImportError (instead of returning False) when an old torchao is installed, which Kaggle's image
    has (0.10). Nothing here uses torchao, so an incompatible one is treated as absent."""
    import importlib
    import peft.import_utils as iu
    try:
        iu.is_torchao_available()
        return
    except ImportError as e:
        log(f"torchao treated as unavailable: {e}")
    for mod in ("peft.import_utils", "peft.tuners.lora.torchao", "peft.utils.quantization_utils"):
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        if hasattr(m, "is_torchao_available"):
            m.is_torchao_available = lambda: False


def attach_lora(model, cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    from peft import LoraConfig, get_peft_model
    guard_peft_torchao()
    if cfg["load_in_4bit"]:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg["gradient_checkpointing"])
    regex, names = lora_target_regex(model, cfg)
    lcfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      target_modules=regex, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    leaf_counts: dict[str, int] = {}
    for n in names:
        leaf = n.rsplit(".", 1)[-1]
        leaf_counts[leaf] = leaf_counts.get(leaf, 0) + 1
    info = {"target_layers": len(names), "target_leaf_counts": leaf_counts,
            "trainable_params": sum(p.numel() for _, p in trainable), "total_params": total,
            "trainable_non_lora": [n for n, _ in trainable if "lora_" not in n][:20],
            "trainable_dtypes": sorted({str(p.dtype) for _, p in trainable})}
    return model, info


# --- loss, training, evaluation -----------------------------------------------------------------------------------
def collate(batch: list[Example], pad_id: int, device) -> dict[str, Any]:
    import torch
    n = max(len(e.input_ids) for e in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for r, e in enumerate(batch):
        ids[r, : len(e.input_ids)] = torch.tensor(e.input_ids)
        labels[r, : len(e.labels)] = torch.tensor(e.labels)
        mask[r, : len(e.input_ids)] = 1
    return {"input_ids": ids.to(device), "labels": labels.to(device), "attention_mask": mask.to(device)}


class LossFn:
    """Summed next-token cross-entropy over target tokens, in fp32. With batch size 1 it asks the model for logits at
    target positions only (``logits_to_keep`` as an index tensor), which saves ~vocab x seq_len x 4 bytes; the first
    call checks that the model honours it and otherwise falls back to full logits."""

    def __init__(self) -> None:
        self.keep_ok: bool | None = None

    def __call__(self, model, b: dict[str, Any], autocast) -> tuple[Any, int, int]:
        import torch
        import torch.nn.functional as F
        tgt = b["labels"][:, 1:]
        sel = tgt != IGNORE
        n = int(sel.sum())
        if b["input_ids"].shape[0] == 1 and self.keep_ok is not False:
            idx = sel[0].nonzero().squeeze(-1)
            try:
                with autocast():
                    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], logits_to_keep=idx)
                if out.logits.shape[1] == idx.numel():
                    self.keep_ok = True
                    logits = out.logits[0].float()
                    y = tgt[0][idx]
                    loss = F.cross_entropy(logits, y, reduction="sum")
                    return loss, n, int((logits.argmax(-1) == y).sum())
                self.keep_ok = False
            except TypeError:
                self.keep_ok = False
        with autocast():
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        logits = out.logits[:, :-1][sel].float()
        y = tgt[sel]
        loss = F.cross_entropy(logits, y, reduction="sum")
        return loss, n, int((logits.argmax(-1) == y).sum())


def make_autocast(dtype):
    import contextlib
    import torch
    if dtype == torch.float32 or not torch.cuda.is_available():
        return contextlib.nullcontext
    return lambda: torch.autocast(device_type="cuda", dtype=dtype)


def evaluate(model, examples: list[Example], pad_id: int, device, autocast, loss_fn: LossFn) -> dict[str, Any]:
    import torch
    model.eval()
    tot, n, hit, per = 0.0, 0, 0, []
    with torch.no_grad():
        for e in examples:
            loss, k, h = loss_fn(model, collate([e], pad_id, device), autocast)
            tot, n, hit = tot + float(loss), n + k, hit + h
            per.append({"source": e.source, "loss": round(float(loss) / max(k, 1), 4), "tokens": k})
    return {"loss": tot / max(n, 1), "token_accuracy": hit / max(n, 1), "tokens": n, "per_example": per}


def tool_names_in(text: str) -> list[str]:
    """Tool names called in generated text: Qwen3.5 XML (<function=name>) or Hermes JSON ({"name": ...})."""
    names = re.findall(r"<function=([\w.\-]+)>", text)
    for blob in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.S):
        try:
            names.append(json.loads(blob)["name"])
        except Exception:  # noqa: BLE001
            pass
    return names


def stop_token_ids(model, tok) -> list[int]:
    """End-of-turn ids for generation. A tokenizer's eos is often <|endoftext|>, not the chat template's <|im_end|>,
    and without the latter greedy decoding runs on into an invented next turn."""
    ids = set()
    gc_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    for i in (gc_eos if isinstance(gc_eos, (list, tuple)) else [gc_eos]) + [tok.eos_token_id]:
        if isinstance(i, int):
            ids.add(i)
    for t in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<turn|>", "<|end|>"):
        i = tok.convert_tokens_to_ids(t) if t in tok.get_vocab() else None
        if isinstance(i, int) and i != tok.unk_token_id:
            ids.add(i)
    return sorted(ids)


@contextlib.contextmanager
def generation_mode(model):
    """Eval mode with the KV cache on. Gradient checkpointing turns the cache off, which makes generation quadratic,
    so it is suspended while generating and restored afterwards (with the train/eval mode)."""
    was_training = model.training
    checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    model.eval()
    model.config.use_cache = True
    if checkpointing:
        model.gradient_checkpointing_disable()
    try:
        yield
    finally:
        model.config.use_cache = False
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train(was_training)


def cut_at_stop(new: list[int], stops: list[int]) -> tuple[list[int], bool]:
    """Generated ids up to and including the first stop token, and whether one was found."""
    for j, t in enumerate(new):
        if t in stops:
            return new[: j + 1], True
    return new, False


def generate(model, tok, prompts: list[tuple[str, str]], cfg: dict[str, Any], device, autocast) -> list[dict]:
    import torch
    stops = stop_token_ids(model, tok)
    out = []
    with generation_mode(model):
        for prompt, ref in prompts:
            ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(device)
            t0 = time.time()
            with torch.no_grad(), autocast():
                gen = model.generate(**ids, max_new_tokens=cfg["gen_max_new_tokens"], do_sample=False,
                                     pad_token_id=tok.pad_token_id, eos_token_id=stops)
            new, stopped = cut_at_stop(gen[0, ids["input_ids"].shape[1]:].tolist(), stops)
            text = tok.decode(new, skip_special_tokens=False)
            ref_cut = ref[: len(text) + 1]
            out.append({"generated": text, "reference": ref,
                        "similarity": round(difflib.SequenceMatcher(None, text, ref).ratio(), 4),
                        "prefix_similarity": round(difflib.SequenceMatcher(None, text, ref_cut).ratio(), 4),
                        "exact": text.strip() == ref.strip(),
                        "tool_calls_generated": tool_names_in(text), "tool_calls_reference": tool_names_in(ref),
                        "tool_names_match": tool_names_in(text) == tool_names_in(ref),
                        "stopped": stopped, "new_tokens": len(new), "seconds": round(time.time() - t0, 2)})
    return out


def lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return f


def train(model, examples: list[Example], cfg: dict[str, Any], pad_id: int, device, dtype, autocast,
          loss_fn: LossFn, log) -> dict[str, Any]:
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    per_epoch = math.ceil(len(examples) / (cfg["batch_size"] * cfg["grad_accum"]))
    total = cfg["max_steps"] or per_epoch * cfg["epochs"]
    warmup = int(total * cfg["warmup_ratio"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(warmup, total))
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 and torch.cuda.is_available() else None
    rng = random.Random(cfg["seed"])
    history, nonfinite, step, t0, tokens_seen = [], 0, 0, time.time(), 0
    order: list[int] = []
    model.train()
    while step < total:
        window = []
        for _ in range(cfg["grad_accum"]):
            batch = []
            for _ in range(cfg["batch_size"]):
                if not order:
                    order = list(range(len(examples)))
                    rng.shuffle(order)
                batch.append(examples[order.pop()])
            window.append(batch)
        n_window = sum(e.n_target for b in window for e in b)
        loss_sum = 0.0
        ok = True
        for batch in window:
            loss, n, _ = loss_fn(model, collate(batch, pad_id, device), autocast)
            if not torch.isfinite(loss):
                ok = False
                break
            scaled = loss / n_window
            (scaler.scale(scaled) if scaler else scaled).backward()
            loss_sum += float(loss.detach())
            tokens_seen += sum(len(e.input_ids) for e in batch)
        if not ok:
            nonfinite += 1
            opt.zero_grad(set_to_none=True)
            log(f"step {step}: non-finite loss, step skipped ({nonfinite})")
            if nonfinite > 3:
                raise RuntimeError("more than 3 non-finite losses: aborting (try dtype fp32 or a lower lr)")
            step += 1
            continue
        if scaler:
            scaler.unscale_(opt)
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"]))
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        row = {"step": step + 1, "loss": loss_sum / max(n_window, 1), "lr": sched.get_last_lr()[0],
               "grad_norm": gnorm, "elapsed_s": round(time.time() - t0, 2)}
        history.append(row)
        if (step + 1) % cfg["log_every"] == 0 or step + 1 == total:
            log(f"step {row['step']}/{total} loss {row['loss']:.4f} lr {row['lr']:.2e} gnorm {gnorm:.3f} "
                f"{row['elapsed_s']}s")
        step += 1
    secs = time.time() - t0
    return {"steps": total, "warmup_steps": warmup, "history": history, "nonfinite_steps": nonfinite,
            "seconds": round(secs, 1), "tokens_per_s": round(tokens_seen / max(secs, 1e-9), 1)}


def lora_moved(model) -> float:
    """Max |lora_B| over the adapter: LoRA starts with B = 0, so > 0 proves the optimiser touched the adapter."""
    best = 0.0
    for n, p in model.named_parameters():
        if "lora_B" in n:
            best = max(best, float(p.detach().abs().max()))
    return best


# --- result card --------------------------------------------------------------------------------------------------
def write_plot(history: list[dict[str, Any]], path: Path,
               series: tuple[tuple[str, str], ...] = (("loss", "train loss (target tokens)"),)) -> bool:
    """One line per (key, label) in ``series``; a second series goes on a right-hand axis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if not history:
        return False
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=120)
    axes = [ax] + [ax.twinx() for _ in series[1:2]]
    for (key, label), a, colour in zip(series, axes, ("tab:blue", "tab:orange")):
        a.plot([h["step"] for h in history], [h.get(key) for h in history], marker="o", ms=3, lw=1.5, color=colour)
        a.set_ylabel(label, color=colour)
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def render_card(card: dict[str, Any]) -> str:
    c, e, d = card["config"], card.get("environment", {}), card.get("data", {})
    lines = [f"# LoRA SFT result card — {card['verdict']}", "",
             f"run `{card['run_id']}` · model `{c['model']}` · {e.get('gpu', 'cpu')} · dtype `{card.get('dtype')}` · "
             f"{card.get('seconds_total')}s total", ""]
    lines += ["## Checks", "", "| check | result | detail |", "|---|---|---|"]
    for k, v in card["checks"].items():
        lines.append(f"| {k} | {'PASS' if v['pass'] else 'FAIL'} | {v.get('detail', '')} |")
    if card.get("error"):
        lines += ["", "## Error", "", "```", card["error"][-3000:], "```"]
    ev0, ev1, ev2 = card.get("eval_before"), card.get("eval_after"), card.get("eval_reloaded")
    if ev0 and ev1:
        lines += ["", "## Evaluation (target tokens only)", "", "| | loss | token accuracy | tokens |", "|---|---|---|---|",
                  f"| before | {ev0['loss']:.4f} | {ev0['token_accuracy']:.3f} | {ev0['tokens']} |",
                  f"| after | {ev1['loss']:.4f} | {ev1['token_accuracy']:.3f} | {ev1['tokens']} |"]
        if ev2:
            lines.append(f"| reloaded adapter | {ev2['loss']:.4f} | {ev2['token_accuracy']:.3f} | {ev2['tokens']} |")
    t = card.get("training")
    if t:
        lines += ["", "## Training", "", f"{t['steps']} steps ({t['warmup_steps']} warmup), {t['seconds']}s, "
                  f"{t['tokens_per_s']} tok/s, non-finite steps {t['nonfinite_steps']}. "
                  f"First loss {t['history'][0]['loss']:.4f}, last {t['history'][-1]['loss']:.4f}." if t["history"]
                  else "no steps"]
        if card.get("plot"):
            lines += ["", f"![loss]({card['plot']})"]
    lines += ["", "## Data", "", "| stat | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in d.items() if k != "errors"]
    if d.get("errors"):
        lines += ["", "Rejected conversations:", ""] + [f"- {x}" for x in d["errors"][:20]]
    if card.get("lora"):
        lo = card["lora"]
        lines += ["", "## LoRA", "", f"r={c['lora_r']} alpha={c['lora_alpha']} dropout={c['lora_dropout']}; "
                  f"{lo['target_layers']} layers {lo['target_leaf_counts']}; trainable {lo['trainable_params']:,} of "
                  f"{lo['total_params']:,} ({100 * lo['trainable_params'] / max(lo['total_params'], 1):.3f}%)"]
    for tag in ("generation_before", "generation_after"):
        if card.get(tag):
            lines += ["", f"## {tag.replace('_', ' ').capitalize()}", ""]
            for g in card[tag]:
                lines += [f"- similarity {g['similarity']} · exact {g['exact']} · tools {g['tool_calls_generated']} "
                          f"(ref {g['tool_calls_reference']}) · {g['new_tokens']} tokens"
                          f"{'' if g.get('stopped', True) else ' (hit max_new_tokens)'}", "",
                          "  ```", "  " + g["generated"][:600].replace("\n", "\n  "), "  ```"]
    lines += ["", "## Environment", "", "```json", json.dumps(e, indent=1), "```", "",
              "## Config", "", "```json", json.dumps(c, indent=1, default=str), "```", ""]
    return "\n".join(lines)


def check(card: dict[str, Any], name: str, ok: bool, detail: str = "") -> None:
    card["checks"][name] = {"pass": bool(ok), "detail": detail}


# --- driver -------------------------------------------------------------------------------------------------------
def run(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = make_config(overrides)
    return run_with_card(cfg, _run, render_card)


def run_with_card(cfg: dict[str, Any], body, render) -> dict[str, Any]:
    """Run ``body(cfg, card, out, log)`` and always write result_card.{json,md} and train.log, even on an exception.
    The verdict is PASS only if every check recorded in card["checks"] passes."""
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    log_fh = open(out / "train.log", "w", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    t_start = time.time()
    card: dict[str, Any] = {"run_id": time.strftime("%Y%m%d_%H%M%S"), "config": cfg, "checks": {},
                            "verdict": "FAIL", "error": None}
    try:
        body(cfg, card, out, log)
    except Exception:  # noqa: BLE001 - every failure still produces a card
        card["error"] = traceback.format_exc()
        log(card["error"])
        check(card, "completed", False, "exception: see Error")
    else:
        check(card, "completed", True)
    card["seconds_total"] = round(time.time() - t_start, 1)
    card["verdict"] = "PASS" if all(v["pass"] for v in card["checks"].values()) else "FAIL"
    (out / "result_card.json").write_text(json.dumps(card, indent=1, default=str), encoding="utf-8")
    (out / "result_card.md").write_text(render(card), encoding="utf-8")
    log(f"VERDICT {card['verdict']}: " + ", ".join(f"{k}={'ok' if v['pass'] else 'FAIL'}"
                                                   for k, v in card["checks"].items()))
    log_fh.close()
    return card


def _run(cfg: dict[str, Any], card: dict[str, Any], out: Path, log) -> None:
    import torch
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    card["environment"] = env = environment()
    log(f"environment {json.dumps(env)}")
    if env["peft"] is None:
        raise RuntimeError("peft is not installed")
    dtype = pick_dtype(cfg["dtype"])
    card["dtype"] = str(dtype).replace("torch.", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t = time.time()
    path = resolve_model(cfg["model"])
    card["model_path"] = path
    log(f"model files at {path} ({time.time() - t:.0f}s)")
    tok = load_tokenizer(path, cfg)

    stats = Stats()
    train_convs = load_conversations(cfg["data_path"], cfg, stats, cfg["max_samples"])
    eval_convs = load_conversations(cfg["eval_path"], cfg, Stats(), None) if cfg["eval_path"] else train_convs
    train_ex = [e for c in train_convs for e in build_examples(tok, c, cfg, stats)]
    eval_stats = Stats()
    eval_ex = ([e for c in eval_convs for e in build_examples(tok, c, cfg, eval_stats)]
               if cfg["eval_path"] else train_ex)
    card["data"] = asdict(stats)
    card["data"]["eval_examples"] = len(eval_ex)
    card["data"]["sources"] = [e.source for e in train_ex][:50]
    log(f"data {json.dumps({k: v for k, v in asdict(stats).items() if k != 'errors'})}")
    for err in stats.errors[:10]:
        log(f"rejected: {err}")
    check(card, "data", bool(train_ex) and stats.skipped_conversations == 0,
          f"{len(train_ex)} examples from {stats.conversations} conversations, {stats.skipped_conversations} rejected")
    check(card, "loss_mask", stats.boundary_straddles == 0 and all(e.n_target > 0 for e in train_ex),
          f"{stats.target_tokens} target of {stats.total_tokens} tokens; straddles {stats.boundary_straddles}; "
          f"tokenization mismatches {stats.tokenization_mismatches}; header in loss {stats.header_in_loss}")
    if not train_ex:
        raise RuntimeError("no training examples")
    prompts = [p for c in eval_convs[: cfg["gen_samples"]] if (p := generation_prompt(tok, c, cfg, Stats()))]

    t = time.time()
    model, load = load_model(path, cfg, dtype)
    card["load"] = load
    log(f"loaded {load['class']} ({load['params']:,} params, {load['param_dtypes']}) in {time.time() - t:.0f}s")
    check(card, "weights", not load["missing_not_allowed"],
          f"{len(load['missing_keys'])} missing ({len(load['missing_not_allowed'])} not allowed), "
          f"{len(load['unexpected_keys'])} unexpected")
    if load["missing_not_allowed"]:
        raise RuntimeError(f"weights missing from the checkpoint: {load['missing_not_allowed'][:10]}")
    model, lora = attach_lora(model, cfg)
    card["lora"] = lora
    log(f"LoRA on {lora['target_layers']} layers {lora['target_leaf_counts']}: "
        f"{lora['trainable_params']:,} trainable")
    check(card, "trainable", lora["trainable_params"] > 0 and not lora["trainable_non_lora"],
          f"{lora['trainable_params']:,} params; non-LoRA trainable {lora['trainable_non_lora']}")

    autocast = make_autocast(dtype)
    loss_fn = LossFn()
    pad = tok.pad_token_id
    card["eval_before"] = evaluate(model, eval_ex, pad, device, autocast, loss_fn)
    card["logits_to_keep"] = loss_fn.keep_ok
    log(f"eval before: loss {card['eval_before']['loss']:.4f} acc {card['eval_before']['token_accuracy']:.3f}")
    if prompts:
        card["generation_before"] = generate(model, tok, prompts, cfg, device, autocast)

    card["training"] = train(model, train_ex, cfg, pad, device, dtype, autocast, loss_fn, log)
    if torch.cuda.is_available():
        card["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    card["eval_after"] = evaluate(model, eval_ex, pad, device, autocast, loss_fn)
    log(f"eval after: loss {card['eval_after']['loss']:.4f} acc {card['eval_after']['token_accuracy']:.3f}")
    if prompts:
        card["generation_after"] = generate(model, tok, prompts, cfg, device, autocast)
    moved = lora_moved(model)
    tr = card["training"]
    check(card, "finite", tr["nonfinite_steps"] == 0 and all(math.isfinite(h["loss"]) for h in tr["history"]),
          f"{tr['nonfinite_steps']} non-finite steps")
    check(card, "adapter_updated", moved > 0, f"max |lora_B| = {moved:.3g}")
    b, a = card["eval_before"]["loss"], card["eval_after"]["loss"]
    check(card, "loss_dropped", a <= b * (1 - cfg["min_loss_drop"]) and a < b,
          f"{b:.4f} -> {a:.4f} ({100 * (b - a) / max(b, 1e-9):.1f}% drop, need >= {100 * cfg['min_loss_drop']:.0f}%)")
    if cfg["min_gen_similarity"] is not None and card.get("generation_after"):
        sim = sum(g["similarity"] for g in card["generation_after"]) / len(card["generation_after"])
        check(card, "generation", sim >= cfg["min_gen_similarity"],
              f"mean similarity {sim:.3f} (need >= {cfg['min_gen_similarity']})")

    adapter_dir = out / "adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    (adapter_dir / "sft_config.json").write_text(json.dumps(cfg, indent=1, default=str), encoding="utf-8")
    files = sorted(p.name for p in adapter_dir.iterdir())
    size = sum(p.stat().st_size for p in adapter_dir.iterdir() if p.is_file())
    card["adapter"] = {"dir": str(adapter_dir), "files": files, "mb": round(size / 2**20, 2)}
    check(card, "adapter_saved", "adapter_config.json" in files and any(f.startswith("adapter_model") for f in files),
          f"{card['adapter']['mb']} MB")
    log(f"adapter saved: {files}")

    if cfg["save_merged"]:
        merged = model.merge_and_unload()
        merged.save_pretrained(out / "merged")
        tok.save_pretrained(out / "merged")
        del merged
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cfg["reload_check"]:
        from peft import PeftModel
        base, _ = load_model(path, cfg, dtype)
        re_model = PeftModel.from_pretrained(base, adapter_dir)
        card["eval_reloaded"] = evaluate(re_model, eval_ex, pad, device, autocast, LossFn())
        r = card["eval_reloaded"]["loss"]
        log(f"eval reloaded: loss {r:.4f}")
        check(card, "reload_matches", abs(r - a) <= cfg["reload_tol"] * max(abs(a), 1e-3) + 1e-4,
              f"after {a:.4f} vs reloaded {r:.4f}")
        del re_model, base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()   # the notebook runs the RL pipeline next in the same process

    card["plot"] = "loss.png" if write_plot(tr["history"], out / "loss.png") else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LoRA SFT with a result card")
    ap.add_argument("--config", help="JSON file of overrides")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (value parsed as JSON when possible)")
    args = ap.parse_args(argv)
    overrides = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    overrides.update(_parse_set(args.set))
    card = run(overrides)
    print(render_card(card))
    return 0 if card["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
