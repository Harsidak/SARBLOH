# E108 — verdict: KEPT (2026-10-01)

Both pipelines run from the single shared notebook `Alignment/alignment.ipynb`. Kernel `banwait13/sarbloh-alignment`
v1 (Qwen/Qwen3.5-2B, T4, fp32) passed first time: **SFT PASS (10/10)** and **RL PASS (11/11)**, 1775 s wall clock in
total. Outputs are in `runs/kaggle_sarbloh-alignment_20261001_204325/` (with the exact notebook that was pushed, as
`kaggle_notebook_v1.ipynb`); the numbers are in `results.json`.

## Predictions vs result

| prediction | result | met |
|---|---|---|
| RL verdict PASS, every check green | PASS, 11/11 | yes |
| Greedy `tool_match` reward rises by 0.1 or more | 0.300 → 0.589 (+0.289); sampled 0.270 → 0.546 | yes |
| SFT in the shared notebook within ±10% of E107 | eval loss 1.1494 → 0.00284, identical to E107 v3; 265 s vs 354 s | yes |
| Under 45 min on a T4 | 29.6 min (SFT 265 s, RL 1450 s); peak VRAM 9.6 GB (SFT), 9.4 GB (RL) | yes |

No kill criterion fired: the references score 1.0, 37 of 45 groups had reward variance, every step updated, there were
no non-finite steps, the reloaded adapter matched exactly (-0.2417 vs -0.2417), and nothing ran out of memory.

## What the reward gain actually is

Read the generations before concluding anything. The gain is almost entirely **format**:
- the policy learned to close its reasoning, emit a well-formed `act` call and stop (stopped fraction 0.33 → 1.0;
  completion length 143 → 115 tokens);
- `smoke-ls20-push` went from the wrong tool to `act` (0.20 → 0.87);
- `smoke-vc33-nudge` went from a ramble cut off at max tokens (0.0) to stopping with an `act` call (0.2). The
  reference there is `remember`, so the name is still wrong;
- the arguments are not better. On `ft09/L0` the after-training call is `["1", "yes"]`, which is nonsense, and the
  argument similarity is unchanged at 0.10.

That is the expected behaviour for a 15-step, 3-prompt smoke test. The reward partly rewards format (stop + valid call =
0.2) and the tool name is worth 0.4, so the easy gradient is to always call `act`. It shows the GRPO machinery works:
group advantages, KL to the adapter-disabled reference (0.06 at step 2, peaking at 0.46, with beta 0.02), sampling with
stop tokens, logits_to_keep, saving and reloading. It does **not** show that RL improves the agent.

## Consequences

- The RL arm of CLAUDE.md §0 is now runnable, but it is not adopted. Adoption still needs a measured win over LoRA
  SFT inside Kaggle's budget, on held-out prompts and then in an agent run.
- Before any real RL run, `tool_match` needs held-out prompts (`eval_path`) and an argument term that cannot be
  satisfied by a constant call. Otherwise the cheapest policy (always `act`) collects 0.6 for free on act-heavy data.
- Cost: about 7.7 s per sampled completion (180 completions in 1381 s) at 160 tokens on a T4 in fp32. Real Prime
  traces have prompts of up to 3k tokens and longer completions, so a real RL run belongs on the RTX Pro 6000 in bf16,
  and only once SFT has a baseline to beat.
- Cosmetic: the RL card's "Data" table reprints the SFT `Stats` fields, which are zero for RL (`examples`,
  `target_tokens`, ...). They are harmless; trim them if the card is reused.

## Using it

The notebook flags are `RUN_SFT` / `RUN_RL`. Each pipeline has a config cell (`SFT_CONFIG`, `RL_CONFIG`).
`RL_CONFIG["init_adapter"]` = the SFT adapter path gives SFT → RL. `reward` takes `"module:function"` or a callable
`f(completion, item, info) -> float | dict`. Push with `uv run python scripts/kaggle_push.py alignment`.
