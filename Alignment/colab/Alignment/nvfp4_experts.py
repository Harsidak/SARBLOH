"""Train LoRA on the served Flash-Next NVFP4 checkpoint directly (RadixArk rev 7b71922 = Kaggle keithtyser model).

Why: the bf16 Flash-Next is 360 GB and needs 8 x H100. The NVFP4 checkpoint is ~135 GB: routed experts in NVFP4
(68 GB), the n-gram table in fp8 (51 GB), everything else bf16 (16 GB). bitsandbytes cannot shrink the experts (they
are fused 3D parameters, not nn.Linear), so this file keeps them in their stored 4-bit form and unpacks one expert at a
time inside the forward and backward pass. The experts are frozen, so no weight gradient is ever needed: the backward
pass unpacks again instead of keeping unpacked weights alive (memory stays at the 4-bit size).

The LoRA targets (attention and DeltaNet projections) are bf16 in this checkpoint, so training sees exactly the
weights Kaggle serves and ``merge_lora.py --touched-only`` writes the adapter into the same 4 bf16 shards.

Not modelled: at serving time NVFP4 also rounds the expert *inputs* to 4 bits (``input_scale``); here the inputs stay
bf16. That is a small difference on the frozen experts only.

    model = nvfp4_experts.load(path, dtype)   # CPU; lora_sft's "split" placement then moves it to the GPUs
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.nn as nn

# E2M1 (fp4) code -> value; bit 3 is the sign. Two codes per byte, the even element in the low nibble (modelopt).
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
_LUT: dict[torch.device, torch.Tensor] = {}
BLOCK = 16


def _lut(device) -> torch.Tensor:
    if device not in _LUT:
        _LUT[device] = torch.tensor(_E2M1, dtype=torch.float32, device=device)
    return _LUT[device]


def dequant(packed: torch.Tensor, scale: torch.Tensor, scale_2: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """packed uint8 [out, in/2], scale fp8_e4m3 [out, in/16], scale_2 fp32 scalar -> [out, in] in ``dtype``."""
    lo, hi = packed & 0x0F, packed >> 4
    codes = torch.stack((lo, hi), dim=-1).flatten(-2).long()
    w = _lut(packed.device)[codes]
    out, n = w.shape
    w = w.view(out, n // BLOCK, BLOCK) * (scale.float() * scale_2.float())[..., None]
    return w.view(out, n).to(dtype)


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The inverse, for tests: [out, in] -> (packed uint8, fp8 block scales, fp32 global scale), modelopt layout."""
    w = w.float()
    out, n = w.shape
    blocks = w.view(out, n // BLOCK, BLOCK)
    scale_2 = (w.abs().max() / (6.0 * 448.0)).clamp_min(1e-12)
    scale = (blocks.abs().amax(-1) / 6.0 / scale_2).clamp(max=448.0).to(torch.float8_e4m3fn)
    denom = (scale.float() * scale_2)[..., None]
    x = torch.where(denom > 0, blocks / denom.clamp_min(1e-30), torch.zeros_like(blocks)).view(out, n)
    mags = torch.tensor(_E2M1[:8])
    code = (x.abs()[..., None] - mags).abs().argmin(-1) + (x < 0).long() * 8
    code = torch.where(code == 8, torch.zeros_like(code), code)   # no negative zero
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
    return packed, scale, scale_2.reshape(())


class _FrozenFP4Linear(torch.autograd.Function):
    """y = x @ W.T with W unpacked on the fly; backward unpacks again (no weight grad, nothing large saved)."""

    @staticmethod
    def forward(ctx, x, packed, scale, scale_2):
        ctx.save_for_backward(packed, scale, scale_2)
        return x @ dequant(packed, scale, scale_2, x.dtype).t()

    @staticmethod
    def backward(ctx, g):
        packed, scale, scale_2 = ctx.saved_tensors
        return g @ dequant(packed, scale, scale_2, g.dtype), None, None, None


def _frozen(shape, dtype) -> nn.Parameter:
    return nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)


class NVFP4Experts(nn.Module):
    """Drop-in for Qwen4ExpTextExperts (same forward signature) holding the routed experts in NVFP4."""

    def __init__(self, num_experts: int, hidden: int, inter: int, act_fn):
        super().__init__()
        self.num_experts, self.hidden_dim, self.intermediate_dim = num_experts, hidden, inter
        self.act_fn = act_fn
        for proj, (o, i) in (("gate_proj", (inter, hidden)), ("up_proj", (inter, hidden)),
                             ("down_proj", (hidden, inter))):
            setattr(self, f"{proj}_w", _frozen((num_experts, o, i // 2), torch.uint8))
            setattr(self, f"{proj}_s", _frozen((num_experts, o, i // BLOCK), torch.float8_e4m3fn))
            setattr(self, f"{proj}_s2", _frozen((num_experts,), torch.float32))

    # offload: True = these experts stay in (pinned) CPU memory and are copied to the GPU for each forward, one layer
    # at a time (place_split skips them). Under gradient checkpointing only one layer's copy is alive at once.
    offload = False

    def _tensors(self, dev) -> dict[str, torch.Tensor]:
        out = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suf in ("_w", "_s", "_s2"):
                t = getattr(self, proj + suf)
                out[proj + suf] = t if t.device == dev else t.to(dev, non_blocking=True)
        return out

    def _lin(self, x, proj: str, e: int, ts=None):
        ts = ts if ts is not None else self._tensors(x.device)
        return _FrozenFP4Linear.apply(x, ts[f"{proj}_w"][e], ts[f"{proj}_s"][e], ts[f"{proj}_s2"][e])

    def weight(self, proj: str, e: int, dtype=torch.float32) -> torch.Tensor:
        return dequant(getattr(self, f"{proj}_w")[e], getattr(self, f"{proj}_s")[e], getattr(self, f"{proj}_s2")[e],
                       dtype)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        ts = self._tensors(hidden_states.device)
        with torch.no_grad():
            mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit[:, 0].tolist():
            pos, tok = torch.where(mask[e])
            x = hidden_states[tok]
            h = self.act_fn(self._lin(x, "gate_proj", e, ts)) * self._lin(x, "up_proj", e, ts)
            h = self._lin(h, "down_proj", e, ts) * top_k_weights[tok, pos, None]
            final.index_add_(0, tok, h.to(final.dtype))
        return final


def mark_offload(model, n_layers: int) -> list[str]:
    """Keep the routed experts of the last ``n_layers`` MoE layers in pinned CPU memory, streamed to the GPU per
    forward (~1.4 GB per Flash-Next layer). Call after load(), before placement. Returns the module names."""
    mods = [(n, m) for n, m in model.named_modules() if isinstance(m, NVFP4Experts)]
    if n_layers <= 0:
        return []
    chosen = mods[-n_layers:]
    for _, m in chosen:
        m.offload = True
        if torch.cuda.is_available():
            for p in m.parameters():
                if p.device.type == "cpu":
                    p.data = p.data.pin_memory()
    return [n for n, _ in chosen]


class FP8Embedding(nn.Module):
    """The n-gram table as stored (fp8_e4m3, no scale); looked-up rows are cast to ``dtype``. Frozen, lives on CPU."""

    def __init__(self, rows: int, dim: int, dtype=torch.bfloat16):
        super().__init__()
        self.weight = _frozen((rows, dim), torch.float8_e4m3fn)
        self.out_dtype = dtype

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = self.weight.view(torch.uint8).index_select(0, ids.reshape(-1).to(self.weight.device))
        return flat.view(torch.float8_e4m3fn).to(self.out_dtype).view(*ids.shape, self.weight.shape[1])


_EXPERT = re.compile(r"^(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale|weight_scale_2"
                     r"|input_scale)$")
_NGRAM = re.compile(r"^(.*\.ngram_embedding)\.shard_(\d+)\.weight$")
_SKIP = re.compile(r"^(model\.)?mtp\.")


def convert(model, dtype=torch.bfloat16) -> dict[str, int]:
    """Swap every routed-experts module and every n-gram table for the NVFP4 / fp8 holders (on whatever device the
    model is; call it on a meta model before loading)."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m
    n = {"experts": 0, "ngram": 0}
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, m.Qwen4ExpTextExperts):
                setattr(mod, cname, NVFP4Experts(child.num_experts, child.hidden_dim, child.intermediate_dim,
                                                 child.act_fn))
                n["experts"] += 1
            elif cname == "ngram_embedding" and isinstance(child, nn.Embedding):
                setattr(mod, cname, FP8Embedding(child.num_embeddings, child.embedding_dim, dtype))
                n["ngram"] += 1
    return n


def load(path: str | Path, dtype=torch.bfloat16, model_class=None, skip_mtp: bool = True):
    """Build the model with empty weights, convert it, then fill it shard by shard from the NVFP4 checkpoint.
    Returns (model, info); info["missing"] lists anything left unloaded (only mtp/lm_head may be)."""
    import transformers
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig
    path = Path(path)
    cfg = AutoConfig.from_pretrained(path)
    if hasattr(cfg, "quantization_config"):
        del cfg.quantization_config          # the experts are handled here, not by a transformers quantizer
    cls = model_class or getattr(transformers, (cfg.architectures or ["Qwen4ExpForConditionalGeneration"])[0])
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with init_empty_weights(include_buffers=False):
            model = cls(cfg)
    finally:
        torch.set_default_dtype(old)
    counts = convert(model, dtype)
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    filled: set[str] = set()

    def put(name: str, t: torch.Tensor) -> nn.Parameter:
        """Replace a (meta) parameter by a real one holding ``t`` (meta tensors cannot take .data from CPU)."""
        mod_name, leaf = name.rsplit(".", 1)
        old = params[name]
        new = nn.Parameter(t, requires_grad=old.requires_grad and t.is_floating_point())
        setattr(model.get_submodule(mod_name), leaf, new)
        params[name] = new
        return new

    def target(name: str) -> nn.Parameter:
        p = params[name]
        if p.device.type == "meta":
            p = put(name, torch.empty(p.shape, dtype=p.dtype, device="cpu"))
        return p

    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    by_file: dict[str, list[str]] = {}
    for k, f in index.items():
        by_file.setdefault(f, []).append(k)
    unexpected = []
    for f in sorted(by_file):
        with safe_open(str(path / f), framework="pt") as sf:
            for k in by_file[f]:
                if skip_mtp and _SKIP.match(k):
                    continue
                if (mt := _EXPERT.match(k)):
                    base, e, proj, kind = mt.group(1), int(mt.group(2)), mt.group(3), mt.group(4)
                    if kind == "input_scale":
                        continue
                    suffix = {"weight": "_w", "weight_scale": "_s", "weight_scale_2": "_s2"}[kind]
                    name = f"{base}.{proj}{suffix}"
                    target(name).data[e].copy_(sf.get_tensor(k).view(params[name].dtype) if kind != "weight_scale_2"
                                               else sf.get_tensor(k))
                    filled.add(name)
                elif (mt := _NGRAM.match(k)):
                    name = mt.group(1) + ".weight"
                    t = sf.get_tensor(k)
                    rows = t.shape[0]
                    i = int(mt.group(2))
                    target(name).data[i * rows:(i + 1) * rows].copy_(t.view(params[name].dtype))
                    filled.add(name)
                elif k in params:
                    t = sf.get_tensor(k)
                    put(k, t.to(params[k].dtype) if t.dtype != params[k].dtype else t)
                    filled.add(k)
                elif k in buffers:                 # computed from the config at init: kept, shape checked
                    t = sf.get_tensor(k)
                    if t.shape != buffers[k].shape:
                        raise ValueError(f"{k}: checkpoint {tuple(t.shape)} vs model {tuple(buffers[k].shape)}")
                else:
                    unexpected.append(k)
    missing = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    model.tie_weights() if getattr(cfg, "tie_word_embeddings", False) else None
    info = {"converted": counts, "missing": missing[:50], "n_missing": len(missing), "unexpected": unexpected[:50],
            "n_unexpected": len(unexpected)}
    return model, info
