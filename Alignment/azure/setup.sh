#!/usr/bin/env bash
# One-time setup of the Azure training VM (ND H100 v5 or ND H200 v5, 8 GPUs; Ubuntu 22.04 HPC image, which ships the
# NVIDIA driver). Run from the copied Alignment/ directory:   bash azure/setup.sh
# Steps: local NVMe as one RAID0 volume at /mnt/nvme -> Python env with the exact versions tested locally (2026-10-10)
# plus the fast DeltaNet kernels -> Flash-Next bf16 weights (360 GB) -> GPU tests of the long-context path.
set -euo pipefail
NVME=/mnt/nvme
MODEL_DIR=$NVME/models/Qwen3.8-Flash-Next

# 1. NVMe scratch (the OS disk is too small for 360 GB of weights). Skipped if already mounted.
if ! mountpoint -q "$NVME"; then
  mapfile -t DISKS < <(lsblk -dpno NAME,TYPE,MOUNTPOINT | awk '$1 ~ /nvme/ && $2 == "disk" && $3 == "" {print $1}')
  if [ "${#DISKS[@]}" -eq 0 ]; then echo "no free NVMe disks found"; exit 1; fi
  sudo mdadm --create /dev/md0 --level=0 --raid-devices="${#DISKS[@]}" "${DISKS[@]}" --force --run
  sudo mkfs.xfs -f /dev/md0
  sudo mkdir -p "$NVME" && sudo mount /dev/md0 "$NVME" && sudo chown "$USER" "$NVME"
fi
df -h "$NVME"

# 2. Python env (uv). Versions = the local test env; the kernels are what makes DeltaNet fast on H100/H200.
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 "$NVME/venv"
source "$NVME/venv/bin/activate"
uv pip install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.11.0" "transformers==5.19.0" "peft==0.21.2" "accelerate==1.15.0" "safetensors==0.8.0" \
  "huggingface_hub[hf_xet]==2.2.0" matplotlib
uv pip install "flash-linear-attention==0.5.2"
# causal-conv1d compiles against the installed torch (several minutes); the model falls back to torch if it fails.
uv pip install --no-build-isolation "causal-conv1d==1.7.0" || echo "WARN causal-conv1d build failed: torch fallback"
python - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("transformers", transformers.__version__)
import fla  # noqa: F401
print("flash-linear-attention OK")
EOF

# 3. Weights (all 131 shards; the MTP head is in them and is ignored for training). Resumable.
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ] || [ "$(ls "$MODEL_DIR"/model-*.safetensors 2>/dev/null | wc -l)" -lt 131 ]; then
  hf download Qwen/Qwen3.8-Flash-Next --local-dir "$MODEL_DIR"
fi
du -sh "$MODEL_DIR"

# 4. The long-context path on this GPU: equivalence to the reference, chunked loss, resume, merge, memory scaling.
if [ -f ../tests/unit/test_alignment_long.py ]; then
  python ../tests/unit/test_alignment_long.py | grep -E "^(FAIL|CHECKS|HARD|INFO)"
fi
echo "setup done: source $NVME/venv/bin/activate && bash azure/run.sh probe"
