"""Before the Azure VM exists: gather the DeepSeek runs, tokenise them for Flash-Next, and size the job.

    python azure/prepare_data.py [--runs ../runs/sft_deepseek] [--config azure/flash_next.json] [--gpu-tok-s 2500]

1. Copies every ``runs/sft_deepseek/<run>/sft_levels.jsonl`` to ``Alignment/data/deepseek/<run>/sft_levels.jsonl``
   (the trace export already makes one row per compaction segment: exactly what the model was sent).
2. Builds the training examples exactly as lora_sft.py will (same config: best run per level, cleared levels only,
   Flash-Next chat template, per-row reasoning effort, nothing cut) and writes the examples cache named in the config,
   so the GPU machine skips tokenisation.
3. Prints the counts the budget depends on: rows, tokens, target tokens, the length histogram, rows dropped as
   longer than max_seq_len, and GPU hours at an assumed training speed (``--gpu-tok-s``: tokens/s of the whole
   machine, forward + backward; replace it with the probe's measured number).
Needs the Flash-Next tokenizer files only (no weights).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import lora_sft as L  # noqa: E402

TOKENIZER = "Qwen/Qwen3.8-Flash-Next"


def gather(runs: Path, dest: Path) -> list[Path]:
    got = []
    for f in sorted(runs.glob("*/sft_levels.jsonl")):
        out = dest / f.parent.name / "sft_levels.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists() or out.stat().st_size != f.stat().st_size:
            shutil.copy2(f, out)
        got.append(out)
    return got


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=str(HERE.parent.parent / "runs" / "sft_deepseek"))
    ap.add_argument("--config", default=str(HERE / "flash_next.json"))
    ap.add_argument("--gpu-tok-s", type=float, default=2500.0, help="assumed whole-machine training tokens/s")
    ap.add_argument("--usd-per-hour", type=float, default=98.0, help="VM price (ND H100 v5 list price, UNCONFIRMED)")
    ap.add_argument("--dest", default="data/deepseek", help="where the run exports go (relative to Alignment/)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value (e.g. for a dry run)")
    a = ap.parse_args(argv)
    import os
    os.chdir(HERE.parent)   # config paths are relative to Alignment/
    files = gather(Path(a.runs), Path(a.dest))
    print(f"{len(files)} run exports in {a.dest}")
    if not files:
        return 1
    over = json.loads(Path(a.config).read_text(encoding="utf-8"))
    over.update(L._parse_set(a.set))
    cfg = L.make_config(over)
    from huggingface_hub import snapshot_download
    tok_dir = snapshot_download(TOKENIZER, allow_patterns=["*.json", "*.jinja", "merges.txt", "vocab.json",
                                                           "tokenizer*"], ignore_patterns=["model.safetensors*"])
    tok = L.load_tokenizer(tok_dir, cfg)
    stats = L.Stats()
    t = time.time()
    convs = L.load_conversations(cfg["data_path"], cfg, stats, cfg["max_samples"])
    ex = [e for c in convs for e in L.build_examples(tok, c, cfg, stats)]
    L.save_examples(Path(cfg["examples_cache"]), L.cache_key(cfg, tok), ex, None, stats)
    lens = sorted(len(e.input_ids) for e in ex)
    hist = {f"<={b // 1024}k": sum(1 for n in lens if n <= b) for b in (16384, 32768, 65536, 98304, 131072)}
    tokens = sum(lens)
    epochs = cfg["epochs"]
    hours = tokens * epochs / a.gpu_tok_s / 3600
    report = {"rows": len(convs), "examples": len(ex), "tokens": tokens, "target_tokens": stats.target_tokens,
              "max_len": max(lens) if lens else 0, "length_cdf": hist, "dropped_too_long": stats.dropped_too_long,
              "dropped_not_best_run": stats.dropped_not_best_run, "rejected": stats.skipped_conversations,
              "tokenization_mismatches": stats.tokenization_mismatches, "build_s": round(time.time() - t),
              "epochs": epochs, "assumed_tok_s": a.gpu_tok_s, "train_hours": round(hours, 2),
              "train_usd": round(hours * a.usd_per_hour), "cache": cfg["examples_cache"]}
    print(json.dumps(report, indent=1))
    for err in stats.errors[:10]:
        print("rejected:", err)
    (Path(a.dest) / "prepare_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
