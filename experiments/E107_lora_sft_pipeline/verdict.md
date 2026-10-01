# E107 — verdict: KEPT (2026-10-01)

The pipeline works end to end on Kaggle. Kernel `banwait13/sarbloh-lora-sft-smoke` v3 (Qwen/Qwen3.5-2B on a T4
in fp32) produced a **PASS** card with all 10 checks green. Outputs are in
`runs/kaggle_sarbloh-lora-sft-smoke_20261001_193531/`; the numbers are in `results.json`.

## Predictions vs result

| prediction | result | met |
|---|---|---|
| Verdict PASS, every check green | PASS, 10/10 | yes |
| Eval loss drops by 50% or more | 1.1494 → 0.0028 (99.8%); token accuracy 0.78 → 1.00 | yes |
| Reloaded adapter matches within 2% | 0.0028 vs 0.0028 (identical) | yes |
| Generations match the reference tool names | 3/3 match, and 3/3 are exact after training (before: 1/3 match, 0/3 exact) | yes |
| Under 30 min on a T4 | 354 s wall clock, 150 s of training, 9.6 GB peak VRAM | yes |

No kill criterion fired: the checkpoint loaded with 0 missing and 0 unexpected weights, and the reload matched.

## What it took

- **v1: FAIL.** On Kaggle's image, `get_peft_model` crashed because peft's `is_torchao_available()` raises
  ImportError instead of returning False when torchao 0.10.0 is installed. The fix is `guard_peft_torchao()`, which
  treats an incompatible torchao as absent; the unit suite simulates the old torchao. The FAIL card was still
  written, which confirms the card is produced even when the run crashes.
- **v2: PASS, with one defect.** Greedy generation ran past `<|im_end|>`, because the tokenizer's eos is
  `<|endoftext|>`, and invented the next turn. That made the similarity and tool-name metrics unreliable. The fix is
  `stop_token_ids()`, which stops generation at the template's end-of-turn token and records `stopped` per sample.
- **v3: PASS, clean.**

## What this does and does not show

It shows that the plumbing is right: the data and masks, the LoRA placement (186 language-model linears including
the qwen3_5 linear-attention projections, none in the vision tower), the gradient path, saving and reloading.
Three samples memorised in 40 steps says **nothing** about generalisation or about RHAE. The next step is the same
pipeline on our own verified trajectories, with a held-out eval file (`eval_path`) and an actual agent run.

## Using it for real training

*(Updated 2026-10-01: `sft/` was renamed `Alignment/`, and the generated `sft_smoke.ipynb` was replaced by the
shared, hand-edited `Alignment/alignment.ipynb`, which runs SFT and RL (E108).)*

In `Alignment/alignment.ipynb`, edit:
- `MODEL`;
- `DATA`: a `/kaggle/input/...` JSONL file. Attach its dataset under `alignment.datasets` in
  `scripts/kaggle_settings.json`;
- `MAX_SAMPLES`: `None` for all rows;
- `SFT_CONFIG`: steps, lr, `eval_path`, `min_loss_drop`.

Then run `uv run python scripts/kaggle_push.py alignment`, followed by `alignment-status` / `alignment-output`. A
27B base on the RTX Pro 6000 needs bf16 (selected automatically on sm80+) or `load_in_4bit`. Neither has been
measured yet (UNCONFIRMED).
