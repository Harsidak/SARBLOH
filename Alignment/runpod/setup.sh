#!/usr/bin/env bash
# One-time setup of a RunPod GPU pod (1 x H200 141 GB, or 2 x H100 80 GB; >= 160 GB RAM; >= 200 GB volume at
# /workspace). Trains on the served NVFP4 checkpoint (~135 GB download), not the 360 GB bf16 one. Run from the copied
# Alignment/ directory:   bash runpod/setup.sh
set -euo pipefail
WS=/workspace
MODEL_DIR=$WS/models/flash-next-nvfp4
REV=7b719225242aacd3dbd3f9407468c2ee9a9d2594
nvidia-smi --query-gpu=name,memory.total --format=csv
free -g | head -2
df -h "$WS"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d "$WS/venv" ] || uv venv --python 3.12 "$WS/venv"
source "$WS/venv/bin/activate"
uv pip install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.11.0" "transformers==5.19.0" "peft==0.21.2" "accelerate==1.15.0" "safetensors==0.8.0" \
  "huggingface_hub[hf_xet]==2.2.0" matplotlib jinja2
uv pip install "flash-linear-attention==0.5.2"
uv pip install --no-build-isolation "causal-conv1d==1.7.0" || echo "WARN causal-conv1d build failed: torch fallback"

# weights first in the background-friendly order: resumable, skips files already there
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=1
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 --revision "$REV" --local-dir "$MODEL_DIR"
du -sh "$MODEL_DIR"

# the NVFP4 path and the long-context path on this GPU
python ../tests/unit/test_alignment_nvfp4.py | grep -E "^(FAIL|CHECKS|HARD|INFO)"
echo "setup done: source $WS/venv/bin/activate && bash runpod/run.sh probe"
