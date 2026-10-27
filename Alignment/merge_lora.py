"""Merge a LoRA adapter into a safetensors checkpoint, one shard at a time (the 360 GB Flash-Next never sits in RAM).

    python merge_lora.py --base <model dir> --adapter <sft_out/adapter> --out <merged dir>

For every LoRA target W (adapter key ``base_model.model.<name>.lora_A.weight``): W += scale * B @ A in fp32, written
back in W's dtype; scale = lora_alpha / r (alpha / sqrt(r) with rslora). Every other tensor is copied byte for byte, as
are config, tokenizer and chat template files. Refuses an adapter it cannot place completely (a target missing from
the checkpoint), DoRA, and adapters with modules_to_save. Writes ``merge_report.json`` (targets merged, max |delta|).
Works on a quantised checkpoint too when the targets stay unquantised: in NVIDIA's Qwen3.8-Flash-Next-NVFP4 only the
routed experts are NVFP4, and every attention / DeltaNet projection (the LoRA targets) is bf16, so the adapter merges
straight into the served checkpoint with no re-quantisation. A quantised target (fp8/fp4 dtype or a *scale* sibling)
is refused.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path


def adapter_deltas(adapter_dir: Path) -> tuple[dict[str, tuple], dict]:
    from safetensors.torch import load_file
    acfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if acfg.get("use_dora"):
        raise SystemExit("DoRA adapters are not supported by this merger")
    if acfg.get("modules_to_save"):
        raise SystemExit(f"modules_to_save {acfg['modules_to_save']} is not supported by this merger")
    r, alpha = acfg["r"], acfg["lora_alpha"]
    scale = alpha / math.sqrt(r) if acfg.get("use_rslora") else alpha / r
    w = load_file(str(adapter_dir / "adapter_model.safetensors"))
    pairs: dict[str, tuple] = {}
    for k, v in w.items():
        if ".lora_A." not in k:
            continue
        name = k.split(".lora_A.")[0]
        for prefix in ("base_model.model.",):
            if name.startswith(prefix):
                name = name[len(prefix):]
        b = w[k.replace(".lora_A.", ".lora_B.")]
        pairs[name + ".weight"] = (v, b)
    return pairs, {"r": r, "alpha": alpha, "scale": scale, "targets": len(pairs)}


def merge(base: Path, adapter: Path, out: Path, touched_only: bool = False) -> dict:
    """touched_only: write just the shards that hold LoRA targets (the other shards need not even be on disk), for
    laying the changed files over an existing copy of the checkpoint (e.g. the Kaggle NVFP4 model)."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    pairs, meta = adapter_deltas(adapter)
    index_path = base / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(index["weight_map"].values()))
        located = {k: index["weight_map"].get(k) for k in pairs}
    else:
        shards = ["model.safetensors"]
        with safe_open(str(base / shards[0]), "pt") as fh:
            names = set(fh.keys())
        located = {k: shards[0] if k in names else None for k in pairs}
    missing = sorted(k for k, v in located.items() if v is None)
    if missing:
        raise SystemExit(f"{len(missing)} adapter targets are not in the checkpoint, e.g. {missing[:5]}")
    out.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy2(f, out / f.name)
    todo_by_shard: dict[str, list[str]] = {}
    for k, s in located.items():
        todo_by_shard.setdefault(s, []).append(k)
    report = {**meta, "touched_shards": sorted(todo_by_shard), "touched_only": touched_only, "merged": 0, "max_abs_delta": 0.0, "shards": len(shards), "seconds": 0.0}
    t0 = time.time()
    for s in shards:
        src = base / s
        if s not in todo_by_shard:
            if not touched_only:
                shutil.copy2(src, out / s)
            continue
        tensors, meta_s = {}, None
        with safe_open(str(src), "pt") as fh:
            meta_s = fh.metadata()
            for k in fh.keys():
                tensors[k] = fh.get_tensor(k)
        for k in todo_by_shard[s]:
            a, b = pairs[k]
            w = tensors[k]
            stem = k[: -len(".weight")]
            scales = [n for n in tensors if n.startswith(stem + ".") and n != k and "scale" in n]
            if not w.is_floating_point() or w.element_size() < 2 or scales:
                raise SystemExit(f"{k}: quantised ({w.dtype}, {scales[:3]}); LoRA merges only into bf16/fp16/fp32 weights")
            delta = (b.float() @ a.float()) * meta["scale"]
            if delta.shape != w.shape:
                raise SystemExit(f"{k}: delta {tuple(delta.shape)} != weight {tuple(w.shape)}")
            tensors[k] = (w.float() + delta).to(w.dtype).contiguous()
            report["merged"] += 1
            report["max_abs_delta"] = max(report["max_abs_delta"], float(delta.abs().max()))
        save_file(tensors, str(out / s), metadata=meta_s or {"format": "pt"})
        del tensors
        print(f"merged {len(todo_by_shard[s])} tensors into {s}", flush=True)
    report["seconds"] = round(time.time() - t0, 1)
    (out / "merge_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--touched-only", action="store_true", help="write only the shards that hold LoRA targets")
    a = ap.parse_args(argv)
    rep = merge(Path(a.base), Path(a.adapter), Path(a.out), a.touched_only)
    print(json.dumps(rep))
    return 0 if rep["merged"] == rep["targets"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
