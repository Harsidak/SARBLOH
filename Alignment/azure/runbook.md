# Flash-Next LoRA SFT on Azure: runbook

Goal: LoRA SFT of Qwen3.8-Flash-Next on our DeepSeek traces, on full rows of up to 131,072 tokens (the harness's
context), so training sees exactly what the model sees on Kaggle. Nothing is cut.

## What is ready (tested on the RTX 4050, 2026-10-10)

| Piece | File | Tested |
|---|---|---|
| Sparse attention for 131k training (no Python loop, no 131k x 131k mask) | `qwen4_exp_long.py` | equals the transformers reference to 1e-7 (logits) / 1e-11 (grads); memory grows linearly (8k: 252 MB, 32k: 945 MB, tiny model) |
| LM head in chunks over target tokens only | `lora_sft.py` `loss_chunk` | equals the full loss and grads |
| Layers split over 8 GPUs, 95 GB n-gram table kept on CPU | `lora_sft.py` `device_map: "split"` | same loss/grads as one device (GPU + CPU split) |
| Several runs, best run per level, per-row reasoning effort, rows never cut | `lora_sft.py` `select`, `overlong: "drop"` | yes |
| Tokenised-example cache (build here, reuse on Azure) | `examples_cache` | yes |
| Resume after a spot eviction; stop after N hours | `save_every`, `resume`, `max_train_hours` | resumed adapter equals uninterrupted |
| Go/no-go probe: train steps at 8k..131k, memory per GPU, tokens/s | `long_probe.py` | Qwen3.5-0.8B locally |
| Merge into the served NVFP4 checkpoint (only 4 bf16 shards, 15 GB) | `merge_lora.py --touched-only` | merged logits equal base + adapter |
| End to end on a real model under 1B | `azure/local_0p8b.json` | Qwen3.5-0.8B on real wire rows: PASS |

Tests: `python tests/unit/test_alignment_long.py` (40 checks), `test_alignment_sft.py` (42), `test_alignment_rl.py` (30).

## Model facts (model card + config, checked 2026-10-10)

- 125B MoE (6B active) + 51B n-gram embedding + 4B MTP; bf16 checkpoint 360 GB, 131 shards; transformers >= 5.19
  (`qwen4_exp`).
- 48 layers: 36 Gated DeltaNet + 12 sparse attention (QSA: each token reads its best 2,048 earlier tokens).
- Residual stream is 4 x 2560 wide, so one layer's saved activations at 131k are 2.7 GB (48 layers: 129 GB in total,
  about 16 GB per GPU on 8 GPUs).
- LoRA targets: `q/k/v/o_proj` (attention) and `in_proj_qkv`, `in_proj_z`, `out_proj` (DeltaNet). That is 156 matrices
  and about 23M parameters. Experts, router, indexer and n-gram table stay frozen.
- On Kaggle we serve RadixArk's NVFP4 (`keithtyser/qwen3-8-flash-next-nvfp4` = RadixArk rev 7b71922). Only its routed
  experts are NVFP4, and all 156 LoRA targets are bf16, in `model-bf16-0000{1,10,11,12}.safetensors`. So the trained
  adapter is merged into those 4 files, and nothing is re-quantised.

## Before the credits arrive (this PC)

1. Collect DeepSeek runs (config `local_SFT_RUN = True`): from `Sarbloh-Experimentation/`, run
   `uv run python -m harness.run --games <ids> --runs 3`. That gives 3 runs per game in `runs/sft_deepseek/` and
   needs `DEEPSEEK_API_KEY` in `.env`.
2. `python azure/prepare_data.py` (from `Alignment/`). It copies the runs to `data/deepseek/`, builds the tokenised
   cache, and prints rows, tokens, the length spread, and hours and dollars at an assumed speed. One ls20 run alone
   was 3.2M tokens (27 of its 32 rows are over 64k), so the total depends on how many runs we keep.
3. Request quota in the Azure portal now (it can take days): the **ND H100 v5** family (Standard_ND96isr_H100_v5,
   8 x H100 80 GB, 1.9 TB RAM). Ask for both spot and regular. If available, also ask for **ND H200 v5**
   (8 x H200 141 GB), which has more room for activations.

## Day one on Azure

1. Create the VM: Ubuntu 22.04 **HPC** image (drivers included), 256 GB OS disk, SSH key.
2. Copy the code, data and cache, keeping the repo layout:
   `scp -r Alignment tests azureuser@<ip>:~/sarbloh/` (with `Alignment/data/deepseek/` inside).
3. `cd ~/sarbloh/Alignment && bash azure/setup.sh`. This sets up the NVMe RAID, the Python env, the 360 GB download
   and the GPU tests (about 40-60 min, mostly the download).
4. `source /mnt/nvme/venv/bin/activate && bash azure/run.sh probe`. **Go/no-go, about 20 min.**
   - Go: the 131072 row trains with every GPU under 92% of its memory. Note the tokens/s; for the real cost, rerun
     `prepare_data.py --gpu-tok-s <that number>`.
   - No-go on memory: lower `long_query_chunk` (512) and `long_moe_chunk` (8192) and probe again; if it still fails,
     switch to ND H200 v5.
   - No-go on an error in the patched path: the probe log names the module. Stop the VM (billing stops) and fix it
     here first.
5. `bash azure/run.sh train`, then `tail -f /mnt/nvme/sft_out/flash_next/train.log`. A checkpoint is saved every
   20 steps. After a spot eviction, start the VM and run `bash azure/run.sh train` again: it resumes. Training
   stops by itself after `max_train_hours` (10) and saves.
6. `bash azure/run.sh merge`. This writes `nvfp4_overlay/` (4 shards, about 15 GB). Copy it and `adapter/` back
   (`scp -r`), then **delete the VM** (a stopped VM still bills for its disks).

## After training (not built yet)

- Upload `nvfp4_overlay/` as a Kaggle dataset. At startup, the harness serves a directory that links Keith's model
  files and swaps in these 4 shards.
- A/B test on Kaggle on the same games: base versus SFT (ledger rows).

## Costs (UNCONFIRMED until the probe measures tokens/s)

ND H100 v5 list price is about $98/h (spot is lower). Hours ≈ tokens x epochs / tokens per second. At an assumed
2,500 tok/s, 10M tokens x 2 epochs is about 2.2 h, or about $220. Add about 1.5 h of setup and probing.

## Known limits

- Only one GPU computes at a time (layers in sequence). That is simple and robust, but 7/8 of the compute idles. If
  the data turns out large: on H200, run 2 replicas of 4 GPUs and all-reduce the LoRA gradients.
- MTP (speculative decoding) is not trained. Its acceptance rate may drop slightly after SFT; measure it on Kaggle.
- Vision is off: pictures in rows become a text placeholder, as in the DeepSeek runs.
