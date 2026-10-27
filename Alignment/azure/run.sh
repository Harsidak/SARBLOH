#!/usr/bin/env bash
# Training-day steps on the Azure VM, from the Alignment/ directory with the venv active (see setup.sh):
#   bash azure/run.sh probe    go/no-go: train steps on 8k..131k-token rows, peak memory per GPU, tokens/s (~20 min)
#   bash azure/run.sh train    LoRA SFT (resumes from the last checkpoint after a spot eviction; just run it again)
#   bash azure/run.sh merge    write the adapter into the 4 bf16 shards of the served NVFP4 checkpoint (~15 GB)
# Everything runs under nohup with logs in $OUT; follow with: tail -f $OUT/<step>.log
set -euo pipefail
CFG=${CFG:-azure/flash_next.json}
OUT=$(python -c "import json;print(json.load(open('$CFG'))['output_dir'])")
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # long rows fragment memory otherwise
export TOKENIZERS_PARALLELISM=false

case "${1:-}" in
  probe)
    python -u long_probe.py --config "$CFG" --lengths 8192,32768,65536,98304,131072 --steps 2 2>&1 | tee "$OUT/probe.log"
    ;;
  train)
    # 8 GPUs hold the layers in turn (device_map auto): one row at a time flows through all of them.
    nohup python -u lora_sft.py --config "$CFG" > "$OUT/train.log" 2>&1 &
    echo "training in the background (pid $!): tail -f $OUT/train.log"
    ;;
  merge)
    # The served model: RadixArk NVFP4 rev 7b71922 (= Kaggle keithtyser/qwen3-8-flash-next-nvfp4). Only its 4 bf16
    # shards that hold attention/DeltaNet weights are needed; routed experts (NVFP4) are untouched.
    NV=/mnt/nvme/models/flash-next-nvfp4-bf16-shards
    hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 --revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594 \
      --include "model.safetensors.index.json" "model-bf16-00001.safetensors" "model-bf16-00010.safetensors" \
      "model-bf16-00011.safetensors" "model-bf16-00012.safetensors" "*.json" "*.jinja" --local-dir "$NV"
    python -u merge_lora.py --base "$NV" --adapter "$OUT/adapter" --out "$OUT/nvfp4_overlay" --touched-only \
      2>&1 | tee "$OUT/merge.log"
    ls -la "$OUT/nvfp4_overlay"
    ;;
  *)
    echo "usage: bash azure/run.sh probe|train|merge"; exit 2;;
esac
