"""Writes kaggle/experimental/sft_smoke.ipynb, the Kaggle launcher for sft/lora_sft.py.

The notebook is self-contained: ``scripts/kaggle_push.py sft`` replaces the ``# @@SFT_SOURCES@@`` marker with the
current sft/lora_sft.py and sft/data/*.jsonl, so a run tests the working tree with no dataset upload. Edit CONFIG in the
generated notebook (or here, then regenerate) to train on real data: point ``data_path`` at a /kaggle/input file.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "kaggle" / "experimental" / "sft_smoke.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


CELLS = [
    md("""
# SARBLOH — LoRA SFT smoke

`sft/lora_sft.py` end to end on Kaggle: a small model, three Prime-style tool trajectories, a result card.
The run passes only if every check passes: data validated, loss masked to assistant tokens, no missing weights, only
LoRA trainable, finite loss, adapter updated, eval loss down by `min_loss_drop`, adapter saved, and the saved adapter
reloaded into a fresh base reproducing the eval loss. Outputs: `/kaggle/working/sft_out/` (`result_card.md`,
`result_card.json`, `loss.png`, `train.log`, `adapter/`).

To train for real: set `data_path` to a `/kaggle/input/...` JSONL (Prime `sft_levels.jsonl` rows work as is), `model`
to a local weights directory for offline runs, and raise `max_steps` / lower `lr`.
"""),
    code("""
# Environment, and dependencies checked in a subprocess so an upgrade is picked up before transformers is imported.
import json, subprocess, sys, time
T0 = time.time()
subprocess.run(["nvidia-smi"], check=False)
PROBE = (
    "import json, importlib\\n"
    "out = {}\\n"
    "try:\\n"
    "    import transformers; from transformers import CONFIG_MAPPING\\n"
    "    out['transformers'] = transformers.__version__; out['qwen3_5'] = 'qwen3_5' in CONFIG_MAPPING\\n"
    "except Exception as e: out['transformers'] = None; out['qwen3_5'] = False\\n"
    "try:\\n"
    "    import peft; out['peft'] = peft.__version__\\n"
    "except Exception: out['peft'] = None\\n"
    "import torch; out['torch'] = torch.__version__; out['cuda'] = torch.cuda.is_available()\\n"
    "print(json.dumps(out))\\n")

def probe():
    return json.loads(subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True,
                                     check=True).stdout.strip().splitlines()[-1])

deps = probe()
print("before:", deps)
need = []
if not deps["qwen3_5"]:
    need.append("transformers>=5.0")
if not deps["peft"]:
    need.append("peft")
if need:  # needs internet: the smoke kernel runs with internet on
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *need], check=True)
    deps = probe()
    print("after: ", deps)
assert deps["qwen3_5"] and deps["peft"] and deps["cuda"], deps
"""),
    code("""
# Every knob. Keys are lora_sft.DEFAULTS; anything not set here takes its default.
CONFIG = {
    "model": "Qwen/Qwen3.5-2B",          # same qwen3_5 family and chat template as Qwen3.8-27B
    "data_path": "sft_src/data/smoke3.jsonl",
    "max_samples": 3,
    "output_dir": "/kaggle/working/sft_out",
    "dtype": "auto",                     # bf16 on sm80+, fp32 on T4 / P100
    "max_seq_len": 4096,
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.0,
    "lr": 5e-4, "max_steps": 40, "warmup_ratio": 0.1, "batch_size": 1, "grad_accum": 1,
    "gradient_checkpointing": True,
    "gen_samples": 3, "gen_max_new_tokens": 200,
    "min_loss_drop": 0.5,                # smoke bar: three samples must be largely memorised
    "reload_check": True,
    "seed": 0,
}
print(json.dumps(CONFIG, indent=1))
"""),
    code("""
# Sources embedded at push time (scripts/kaggle_push.py sft): the run tests the working tree.
from pathlib import Path
# @@SFT_SOURCES@@
SRC = Path("sft_src")
for rel, text in SOURCES.items():
    p = SRC / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
sys.path.insert(0, str(SRC.resolve()))
print(SOURCES_META)
"""),
    code("""
import lora_sft
card = lora_sft.run(CONFIG)
from IPython.display import Markdown, display
display(Markdown(Path(CONFIG["output_dir"], "result_card.md").read_text(encoding="utf-8")))
"""),
    code("""
summary = {"verdict": card["verdict"], "checks": {k: v["pass"] for k, v in card["checks"].items()},
           "eval_loss_before": (card.get("eval_before") or {}).get("loss"),
           "eval_loss_after": (card.get("eval_after") or {}).get("loss"),
           "eval_loss_reloaded": (card.get("eval_reloaded") or {}).get("loss"),
           "train_seconds": (card.get("training") or {}).get("seconds"), "peak_vram_gb": card.get("peak_vram_gb"),
           "gpu": card.get("environment", {}).get("gpu"), "dtype": card.get("dtype"), "source": SOURCES_META,
           "wall_s": round(time.time() - T0, 1)}
print("SFT_SUMMARY " + json.dumps(summary))
Path("/kaggle/working/sft_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
assert card["verdict"] == "PASS", "result card verdict FAIL: see the card above"
"""),
]

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}

if __name__ == "__main__":
    OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
