#!/usr/bin/env bash
# Training-day steps on the RunPod pod, from Alignment/ with the venv active (see setup.sh):
#   bash runpod/run.sh probe    go/no-go: train steps at 8k/16k/24k tokens, peak memory, tokens/s (~10 min)
#   bash runpod/run.sh train    LoRA SFT on data/*.jsonl (resumes from the last checkpoint if restarted)
#   bash runpod/run.sh merge    adapter -> the 4 bf16 shards of the same checkpoint (nvfp4_overlay/, ~15 GB)
set -euo pipefail
CFG=${CFG:-runpod/flash_next_nvfp4.json}
OUT=$(python -c "import json;print(json.load(open('$CFG'))['output_dir'])")
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
case "${1:-}" in
  probe)
    python -u long_probe.py --config "$CFG" --lengths 8192,16384,24576 --steps 2 2>&1 | tee "$OUT/probe.log" ;;
  train)
    nohup python -u lora_sft.py --config "$CFG" > "$OUT/train.log" 2>&1 &
    echo "training in the background (pid $!): tail -f $OUT/train.log" ;;
  merge)
    M=$(python -c "import json;print(json.load(open('$CFG'))['model'])")
    python -u merge_lora.py --base "$M" --adapter "$OUT/adapter" --out "$OUT/nvfp4_overlay" --touched-only \
      2>&1 | tee "$OUT/merge.log"
    ls -la "$OUT/nvfp4_overlay" ;;
  *) echo "usage: bash runpod/run.sh probe|train|merge"; exit 2 ;;
esac
