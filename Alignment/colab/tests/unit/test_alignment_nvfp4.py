# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_ROOT, _os.path.join(_ROOT, "Alignment")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""LoRA SFT on the NVFP4 Flash-Next checkpoint (Alignment/nvfp4_experts.py + lora_sft weights_format).

A tiny random qwen4_exp model is written as a fake NVFP4 checkpoint in the RadixArk layout (per-expert NVFP4 tensors,
n-gram table as fp8 shards, the rest bf16, an mtp tensor to skip). Loading it must give the same logits and LoRA
gradients as the plain model holding the same (dequantised) weights; then lora_sft trains it end to end, the adapter
merges into the bf16 shard, and (with a GPU) the split placement runs with the n-gram table on CPU.

    uv run --with torch --with "transformers>=5.19" --with peft --with accelerate --with matplotlib \\
        python tests/unit/test_alignment_nvfp4.py
"""

import json
import shutil
import tempfile
from pathlib import Path

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


try:
    import torch
    import peft  # noqa: F401
    import accelerate  # noqa: F401
    from transformers import Qwen4ExpForConditionalGeneration
except ImportError as exc:
    print(f"SKIP: {exc}")
    print("CHECKS: 0")
    print("HARD FAILS: 0 -> []")
    raise SystemExit(0)

import lora_sft as L  # noqa: E402
import nvfp4_experts as N  # noqa: E402
import qwen4_exp_long as Q  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

work = Path(tempfile.mkdtemp(prefix="nvfp4_test_"))

# --- 1. the format: dequantise(quantise(w)) is close, codes round-trip, packing order -------------------------------
torch.manual_seed(0)
w = torch.randn(64, 128)
pk, sc, s2 = N.quantize(w)
d = N.dequant(pk, sc, s2, torch.float32)
rel = float((d - w).norm() / w.norm())
ok("fp4 round trip error is fp4-sized (< 0.15)", rel < 0.15, str(rel))
ok("packed shapes follow modelopt", pk.shape == (64, 64) and sc.shape == (64, 8) and pk.dtype == torch.uint8
   and sc.dtype == torch.float8_e4m3fn, str((pk.shape, sc.shape)))
exact = torch.tensor([[0.5, -1.0, 6.0, -6.0] + [0.0] * 12])
pk1, sc1, s21 = N.quantize(exact)
ok("exact fp4 values survive (low nibble = even element)",
   torch.equal(N.dequant(pk1, sc1, s21, torch.float32), exact), str(N.dequant(pk1, sc1, s21, torch.float32)))
ok("quantise is stable (requantising the dequantised weight gives the same codes)",
   torch.equal(N.quantize(d)[0], pk))

# frozen linear: same output and input grad as a plain matmul with the dequantised weight, nothing large saved
x = torch.randn(5, 128, requires_grad=True)
y = N._FrozenFP4Linear.apply(x, pk, sc, s2)
y.sum().backward()
x2 = x.detach().clone().requires_grad_(True)
(x2 @ d.t()).sum().backward()
ok("frozen fp4 linear forward/backward equal the dequantised matmul",
   torch.allclose(y, x2 @ d.t(), atol=1e-5) and torch.allclose(x.grad, x2.grad, atol=1e-5))
saved = []
with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
    N._FrozenFP4Linear.apply(torch.randn(5, 128, requires_grad=True), pk, sc, s2)
ok("frozen fp4 linear saves only the packed tensors (no unpacked weight)",
   all(t.dtype in (torch.uint8, torch.float8_e4m3fn, torch.float32) and t.numel() <= pk.numel() for t in saved)
   and not any(t.dtype == torch.float32 and t.numel() == d.numel() for t in saved),
   str([(t.dtype, tuple(t.shape)) for t in saved]))

# --- 2. a fake NVFP4 checkpoint of a tiny model ---------------------------------------------------------------------
torch.manual_seed(1)
cfg = Q.tiny_config()
ref = Qwen4ExpForConditionalGeneration(cfg).float()
with torch.no_grad():
    for n, p in ref.named_parameters():
        if p.dim() == 1 and "norm" in n:
            p.normal_(0, 0.1)
        p.copy_(p.to(torch.bfloat16).float())      # the checkpoint stores bf16: start from bf16-exact values
sd = ref.state_dict()
ckpt: dict[str, torch.Tensor] = {}
n_shards = 4
for k, v in sd.items():
    if k.endswith("mlp.experts.gate_up_proj"):
        base = k[: -len(".gate_up_proj")]
        inter = v.shape[1] // 2
        for e in range(v.shape[0]):
            for proj, m in (("gate_proj", v[e, :inter]), ("up_proj", v[e, inter:])):
                a, b, c = N.quantize(m)
                ckpt.update({f"{base}.{e}.{proj}.weight": a, f"{base}.{e}.{proj}.weight_scale": b,
                             f"{base}.{e}.{proj}.weight_scale_2": c, f"{base}.{e}.{proj}.input_scale": torch.ones(())})
    elif k.endswith("mlp.experts.down_proj"):
        base = k[: -len(".down_proj")]
        for e in range(v.shape[0]):
            a, b, c = N.quantize(v[e])
            ckpt.update({f"{base}.{e}.down_proj.weight": a, f"{base}.{e}.down_proj.weight_scale": b,
                         f"{base}.{e}.down_proj.weight_scale_2": c, f"{base}.{e}.down_proj.input_scale": torch.ones(())})
    elif k.endswith("ngram_embedding.weight"):
        f8 = v.to(torch.float8_e4m3fn)
        rows = v.shape[0] // n_shards
        assert rows * n_shards == v.shape[0]
        base = k[: -len(".weight")]
        for i in range(n_shards):
            ckpt[f"{base}.shard_{i}.weight"] = f8[i * rows:(i + 1) * rows].contiguous()
    else:
        ckpt[k] = (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous()
ckpt["mtp.layers.0.fake.weight"] = torch.zeros(2, 2, dtype=torch.bfloat16)

# the reference model now holds exactly what the checkpoint means
with torch.no_grad():
    for name, mod in ref.named_modules():
        if name.endswith("mlp.experts"):
            for e in range(mod.num_experts):
                g = N.dequant(ckpt[f"{name}.{e}.gate_proj.weight"], ckpt[f"{name}.{e}.gate_proj.weight_scale"],
                              ckpt[f"{name}.{e}.gate_proj.weight_scale_2"], torch.float32)
                u = N.dequant(ckpt[f"{name}.{e}.up_proj.weight"], ckpt[f"{name}.{e}.up_proj.weight_scale"],
                              ckpt[f"{name}.{e}.up_proj.weight_scale_2"], torch.float32)
                dn = N.dequant(ckpt[f"{name}.{e}.down_proj.weight"], ckpt[f"{name}.{e}.down_proj.weight_scale"],
                               ckpt[f"{name}.{e}.down_proj.weight_scale_2"], torch.float32)
                mod.gate_up_proj[e].copy_(torch.cat([g, u]))
                mod.down_proj[e].copy_(dn)
        if name.endswith("ngram_embedding"):
            mod.weight.copy_(mod.weight.to(torch.float8_e4m3fn).float())

mdir = work / "tiny-nvfp4"
mdir.mkdir()
keys = sorted(ckpt)
shard_of = {}
groups = {"model-bf16-00001.safetensors": [], "layer-experts.safetensors": [], "model-plefp8-00000.safetensors": []}
for k in keys:
    f = ("layer-experts.safetensors" if ".mlp.experts." in k else
         "model-plefp8-00000.safetensors" if "ngram_embedding" in k else "model-bf16-00001.safetensors")
    groups[f].append(k)
    shard_of[k] = f
for f, ks in groups.items():
    save_file({k: ckpt[k] for k in ks}, str(mdir / f), metadata={"format": "pt"})
(mdir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": shard_of}))
cfg.save_pretrained(mdir)
c = json.loads((mdir / "config.json").read_text())
c["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["*.self_attn.*"]}
(mdir / "config.json").write_text(json.dumps(c))

# --- 3. load: nothing missing, the right modules swapped, same logits and LoRA grads ------------------------------
model, info = N.load(mdir, torch.float32)
ok("loader swapped every experts module and the n-gram table",
   info["converted"]["experts"] == cfg.text_config.num_hidden_layers and info["converted"]["ngram"] == 1,
   json.dumps(info["converted"]))
ok("loader left no weight unloaded and skipped mtp only", info["n_missing"] == 0 and info["n_unexpected"] == 0,
   json.dumps(info))
ok("experts stay packed (uint8 / fp8 parameters, frozen)",
   all(p.dtype in (torch.uint8, torch.float8_e4m3fn, torch.float32) and not p.requires_grad
       for n, p in model.named_parameters() if ".mlp.experts." in n or "ngram" in n))
from transformers import AutoConfig  # noqa: E402
ok("weights_format auto detects the modelopt NVFP4 config",
   L.is_nvfp4(AutoConfig.from_pretrained(mdir), L.make_config({})) and
   not L.is_nvfp4(cfg, L.make_config({})))

ids = torch.randint(0, cfg.text_config.vocab_size, (1, 200))
labels = ids.clone()
labels[0, :50] = L.IGNORE
batch = {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}
ref.eval()
model.eval()
with torch.no_grad():
    a = ref(input_ids=ids, use_cache=False).logits
    b = model(input_ids=ids, use_cache=False).logits
dl = float((a - b).abs().max())
ok("NVFP4 model logits equal the reference holding the same weights", dl <= 1e-4 * max(float(a.abs().max()), 1.0),
   f"max diff {dl}")


def lora_grads(m, seed):
    torch.manual_seed(seed)
    m, _ = L.attach_lora(m, L.make_config({"lora_r": 4, "lora_alpha": 8, "gradient_checkpointing": True,
                                           "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv",
                                                            "in_proj_z", "out_proj"],
                                           "lora_exclude": ["visual", "mtp", "lm_head", "embed", "indexer", "ple",
                                                            "hyper_connection", "shared_expert"]}))
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)
    m.train()
    loss, _, _ = L.LossFn(64)(m, batch, L.make_autocast(torch.float32))
    loss.backward()
    return float(loss), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


Q.enable(query_chunk=32, moe_chunk=64)
la, ga = lora_grads(ref, 7)
lb, gb = lora_grads(model, 7)
Q.disable()
ok("same loss on the long-context path", abs(la - lb) <= 1e-4 * abs(la), f"{la} vs {lb}")
worst = max(float((ga[k] - gb[k]).abs().max() / (ga[k].abs().max() + 1e-12)) for k in ga) if set(ga) == set(gb) else 9
ok("same LoRA gradients (through the frozen fp4 experts, with checkpointing)", set(ga) == set(gb) and len(ga) > 0
   and worst < 1e-3, f"{len(ga)} vs {len(gb)} grads, worst rel {worst}")
ok("only LoRA parameters get gradients", all("lora_" in k for k in gb), str([k for k in gb if "lora_" not in k][:5]))
del ref, model

# --- 4. lora_sft end to end on the NVFP4 checkpoint, then merge into the bf16 shard -------------------------------
from huggingface_hub import snapshot_download  # noqa: E402
TOKDIR = snapshot_download("Qwen/Qwen3.8-Flash-Next", allow_patterns=["*.json", "*.jinja", "merges.txt", "vocab.json",
                                                                     "tokenizer*"],
                           ignore_patterns=["model.safetensors*"])
tok = L.load_tokenizer(TOKDIR, L.make_config({}))
# a tokenizer-sized copy of the checkpoint (vocab must match the real tokenizer)
torch.manual_seed(2)
big_cfg = Q.tiny_config(vocab_size=len(tok), eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
                        bos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
src = Qwen4ExpForConditionalGeneration(big_cfg).to(torch.bfloat16)
ck2: dict[str, torch.Tensor] = {}
for k, v in src.state_dict().items():
    if k.endswith("mlp.experts.gate_up_proj") or k.endswith("mlp.experts.down_proj"):
        base, which = k.rsplit(".", 1)
        for e in range(v.shape[0]):
            parts = ({"gate_proj": v[e, :v.shape[1] // 2], "up_proj": v[e, v.shape[1] // 2:]}
                     if which == "gate_up_proj" else {"down_proj": v[e]})
            for proj, mtx in parts.items():
                a_, b_, c_ = N.quantize(mtx.float())
                ck2.update({f"{base}.{e}.{proj}.weight": a_, f"{base}.{e}.{proj}.weight_scale": b_,
                            f"{base}.{e}.{proj}.weight_scale_2": c_})
    elif k.endswith("ngram_embedding.weight"):
        ck2[k[: -len(".weight")] + ".shard_0.weight"] = v.to(torch.float8_e4m3fn).contiguous()
    else:
        ck2[k] = v.contiguous()
edir = work / "tiny-nvfp4-tok"
edir.mkdir()
bf = {k: t for k, t in ck2.items() if ".mlp.experts." not in k and "ngram" not in k}
save_file(bf, str(edir / "model-bf16-00001.safetensors"), metadata={"format": "pt"})
save_file({k: t for k, t in ck2.items() if k not in bf}, str(edir / "layer-experts.safetensors"),
          metadata={"format": "pt"})
(edir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": {
    k: ("model-bf16-00001.safetensors" if k in bf else "layer-experts.safetensors") for k in ck2}}))
big_cfg.save_pretrained(edir)
c = json.loads((edir / "config.json").read_text())
c["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
(edir / "config.json").write_text(json.dumps(c))
for f in Path(TOKDIR).iterdir():
    if f.is_file() and f.name != "config.json" and not f.name.endswith(".safetensors") and "index" not in f.name:
        shutil.copy(f, edir / f.name)
del src

SMOKE = Path(_ROOT) / "Alignment" / "fixtures" / "smoke3.jsonl"
card = L.run({"model": str(edir), "data_path": str(SMOKE), "lora_r": 8, "lora_alpha": 16, "lr": 5e-3,
              "warmup_ratio": 0.0, "dtype": "fp32", "loss_chunk": 64, "overlong": "drop", "max_seq_len": 32768,
              "long_query_chunk": 64, "long_moe_chunk": 256, "output_dir": str(work / "e2e"), "max_steps": 24,
              "gen_samples": 0, "min_loss_drop": 0.01,
              "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj"],
              "lora_exclude": ["visual", "mtp", "lm_head", "embed", "indexer", "ple", "hyper_connection",
                               "shared_expert"]})
ok("e2e on the NVFP4 checkpoint: verdict PASS (loss drops, reload matches)", card["verdict"] == "PASS",
   json.dumps({k: v for k, v in card.get("checks", {}).items() if not v["pass"]}) + (card.get("error") or "")[-1500:])
ok("e2e loaded through the nvfp4 path with nothing missing",
   not card.get("load", {}).get("missing_keys") and not card.get("load", {}).get("missing_not_allowed"),
   str(card.get("load"))[:600])

import merge_lora  # noqa: E402
rep = merge_lora.merge(edir, work / "e2e" / "adapter", work / "overlay", touched_only=True)
ok("adapter merges into the bf16 shard only (experts untouched)",
   rep["merged"] == rep["targets"] > 0 and rep["touched_shards"] == ["model-bf16-00001.safetensors"], json.dumps(rep))

# --- 5. GPU: split placement with the n-gram table on CPU ---------------------------------------------------------
if torch.cuda.is_available():
    m2, _ = L.load_model(str(mdir), L.make_config({"device_map": "split", "split_devices": ["cuda:0"],
                                                   "cpu_modules": ["ngram_embedding"]}), torch.bfloat16)
    ng = next(mod for n, mod in m2.named_modules() if n.endswith("ngram_embedding"))
    ex = next(mod for n, mod in m2.named_modules() if n.endswith("mlp.experts"))
    ok("split: n-gram table on CPU, packed experts on the GPU",
       ng.weight.device.type == "cpu" and ex.gate_proj_w.device.type == "cuda")
    Q.enable(query_chunk=32, moe_chunk=64)
    with torch.no_grad():
        out = m2(input_ids=ids.cuda(), use_cache=False).logits.float().cpu()
    Q.disable()
    rel = float((out - b).norm() / b.norm())
    ok("split bf16 GPU logits close to the fp32 CPU result", rel < 0.05, str(rel))

    # streamed experts: the last MoE layer's experts stay in pinned CPU memory, compute on the GPU, same result
    m3, ld3 = L.load_model(str(mdir), L.make_config({"device_map": "split", "split_devices": ["cuda:0"],
                                                     "cpu_modules": ["ngram_embedding"], "offload_expert_layers": 1}),
                           torch.bfloat16)
    exs = [mod for n, mod in m3.named_modules() if isinstance(mod, N.NVFP4Experts)]
    ok("offload: last experts pinned on CPU, the others on the GPU",
       exs[-1].offload and exs[-1].gate_proj_w.device.type == "cpu" and exs[-1].gate_proj_w.is_pinned()
       and all(e.gate_proj_w.device.type == "cuda" for e in exs[:-1]) and ld3["offloaded_expert_layers"] == 1,
       str([e.gate_proj_w.device for e in exs]))
    Q.enable(query_chunk=32, moe_chunk=64)
    with torch.no_grad():
        out3 = m3(input_ids=ids.cuda(), use_cache=False).logits.float().cpu()
    rel3 = float((out3 - out).norm() / out.norm())
    ok("offload: logits match the all-GPU model", rel3 < 1e-3, str(rel3))
    m3.train()
    loss3 = m3(input_ids=ids.cuda(), use_cache=False).logits.float().pow(2).mean()
    loss3.backward()
    gs = [p.grad for p in m3.parameters() if p.grad is not None]
    Q.disable()
    ok("offload: backward runs, grads finite", len(gs) > 0 and all(torch.isfinite(g).all() for g in gs))
    del m2, m3
    torch.cuda.empty_cache()
else:
    print("INFO no GPU: split placement check skipped")

shutil.rmtree(work, ignore_errors=True)
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
