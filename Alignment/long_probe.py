"""Go/no-go probe for long-row LoRA SFT: can this machine train this model on rows of 8k ... 131k tokens, and how fast?

    python long_probe.py --config azure/flash_next.json [--lengths 8192,32768,65536,131072] [--steps 2]

Loads the model and LoRA exactly as lora_sft.py does (same config file), then for each length runs ``--steps``
training steps (forward + backward + optimizer) on a random-token row whose last quarter is the target, as in real
rows. Prints and writes ``probe.json`` in output_dir: per length the peak memory of every GPU, seconds per step,
tokens/s, and the first failure (out of memory stops the probe). A length passes when it trains with
every GPU under ``--headroom`` of its memory (default 0.92), leaving room for real rows' variation.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import lora_sft as L


def probe(cfg: dict, lengths: list[int], steps: int, headroom: float, log=print) -> dict:
    import torch
    dtype = L.pick_dtype(cfg["dtype"])
    path = L.resolve_model(cfg["model"])
    t = time.time()
    model, load = L.load_model(path, cfg, dtype)
    log(f"loaded {load['class']} in {time.time() - t:.0f}s; placement {load['placement_params']}; "
        f"long_context {load['long_context']}")
    model, lora = L.attach_lora(model, cfg)
    log(f"LoRA on {lora['target_layers']} layers, {lora['trainable_params']:,} trainable")
    device = L.input_device(model)
    autocast = L.make_autocast(dtype)
    loss_fn = L.LossFn(cfg["loss_chunk"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-6)
    vocab = model.get_input_embeddings().weight.shape[0]
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    totals = [torch.cuda.get_device_properties(i).total_memory for i in range(n_gpu)]
    out = {"model": cfg["model"], "load": load, "lora": {k: lora[k] for k in ("target_layers", "trainable_params")},
           "gpus": [torch.cuda.get_device_name(i) for i in range(n_gpu)],
           "gpu_total_gb": [round(x / 2 ** 30, 1) for x in totals], "results": []}
    model.train()
    for n in lengths:
        row = {"tokens": n}
        try:
            for i in range(n_gpu):
                torch.cuda.reset_peak_memory_stats(i)
            ids = torch.randint(0, vocab, (1, n))
            labels = torch.full_like(ids, L.IGNORE)
            labels[0, -n // 4:] = ids[0, -n // 4:]
            b = {"input_ids": ids.to(device), "labels": labels.to(device), "attention_mask": torch.ones_like(ids).to(device)}
            times = []
            for _ in range(steps):
                if n_gpu:
                    torch.cuda.synchronize()
                t = time.time()
                loss, k, _ = loss_fn(model, b, autocast)
                (loss / k).backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                if n_gpu:
                    torch.cuda.synchronize()
                times.append(time.time() - t)
            peaks = [torch.cuda.max_memory_allocated(i) for i in range(n_gpu)]
            sec = min(times)
            row.update(ok=all(p <= headroom * tot for p, tot in zip(peaks, totals)),
                       peak_gb=[round(p / 2 ** 30, 2) for p in peaks],
                       peak_frac=[round(p / tot, 3) for p, tot in zip(peaks, totals)],
                       sec_per_step=round(sec, 2), tokens_per_s=round(n / sec, 1), loss=round(float(loss) / k, 4),
                       chunked_loss_ok=loss_fn.chunk_ok)
        except torch.cuda.OutOfMemoryError as exc:
            row.update(ok=False, error=f"OOM: {str(exc)[:300]}")
        except Exception:  # noqa: BLE001
            row.update(ok=False, error=traceback.format_exc()[-1500:])
        out["results"].append(row)
        log(json.dumps(row))
        opt.zero_grad(set_to_none=True)
        if n_gpu:
            torch.cuda.empty_cache()
        if not row.get("ok"):
            break
    passed = [r["tokens"] for r in out["results"] if r.get("ok")]
    out["max_ok_tokens"] = max(passed) if passed else 0
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="the lora_sft JSON config of the real run")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--lengths", default="8192,32768,65536,131072")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--headroom", type=float, default=0.92)
    a = ap.parse_args(argv)
    over = json.loads(Path(a.config).read_text(encoding="utf-8"))
    over.update(L._parse_set(a.set))
    cfg = L.make_config(over)
    res = probe(cfg, [int(x) for x in a.lengths.split(",")], a.steps, a.headroom)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(f"PROBE max_ok_tokens={res['max_ok_tokens']} -> {out / 'probe.json'}")
    return 0 if res["max_ok_tokens"] >= max(int(x) for x in a.lengths.split(",")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
