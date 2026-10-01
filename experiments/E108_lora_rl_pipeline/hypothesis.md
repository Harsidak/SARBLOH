# E108 — LoRA RL (GRPO) pipeline smoke, in the shared Alignment notebook (Kaggle)

- **Date opened:** 2026-10-01
- **Axis varied:** adaptation (training infrastructure). This is the RL arm next to E107's SFT. CLAUDE.md §0 says
  RL is adopted only if it beats SFT within Kaggle's budget, and this pipeline is what makes that comparison
  possible.
- **Baseline:** none. This is plumbing, not a score. E107 (SFT) is the sibling pipeline.
- **Split:** none. The data is the same three hand-written Prime-style trajectories (`Alignment/data/smoke3.jsonl`).
  No game is played.
- **Owner request (2026-10-01):** rename `sft/` to `Alignment/`. Run two pipelines, SFT and RL, from one shared
  notebook (`Alignment/alignment.ipynb`), which is edited directly and pushed to Kaggle as is.

## Mechanism

`Alignment/lora_rl.py` is plain torch + transformers + peft, with no TRL. It reuses `lora_sft.py` for data
validation, chat-template rendering, model loading and LoRA placement. The steps:

1. **Prompts.** Each trained assistant turn becomes a prompt: the template's generation prompt for that turn. The
   reference completion is the turn as the template renders it.
2. **Reward.** It is verifiable and computed offline from the reference tool calls (`tool_match`, range 0 to 1):
   - 0.1 if the completion stops at the end-of-turn token;
   - 0.1 if it makes a well-formed call to a known tool with every required argument;
   - 0.4 for matching the reference tool names;
   - 0.4 for argument similarity.

   Reasoning text is deliberately not rewarded: RL scores the action, while SFT imitates the text. The reward can be
   swapped via `reward: "module:function"`, which is the hook for a later environment-replay reward.
3. **GRPO.** For each prompt, sample G completions. The advantage is `(r - mean) / (std + eps)` within the group.
   Groups with zero variance are skipped. Training is on-policy with a single update per batch, so the importance
   ratio is 1 and the loss is `-A * exp(logp - logp.detach())` plus `beta * KL_k3(policy || reference)`, normalised
   per token. The reference is the same model with its adapter disabled (`init_adapter`, if given, is merged into
   the base first, so RL on top of SFT keeps SFT as the reference).
4. **Evaluate.** Greedy reward before and after training.
5. **Save and reload.** Save the adapter, reload it into a fresh base, and require the same log-probabilities on the
   after-training completions.
6. **Report.** Result card with PASS/FAIL checks, like E107.

## Why it should work

On these prompts the base Qwen3.5-2B already varies under sampling: it called `ipython` where the reference calls
`act`, called `act` correctly once, and ran past its token budget once (E107 v3 "generation before"). That gives
the group-relative advantage nonzero variance. A KL-regularised policy gradient on three prompts should move the
greedy policy toward the rewarded tool calls within a few hundred samples.

## Prediction

Committed before the first Kaggle run (T4, fp32, G=4, 3 prompts per step, 15 steps, lr 2e-4, beta 0.02):
- Verdict PASS. The checks are: data, weights, trainable, reward_sane (each reference scores ≥ 0.99 under its own
  reward), finite, learning_signal, adapter_updated, reward_improved, adapter_saved, reload_matches, completed.
- Mean greedy reward rises by ≥ 0.1 (from roughly 0.3).
- The SFT smoke in the same notebook still passes (E107 v3 numbers ±10%).
- Wall clock for the whole notebook is under 45 minutes on a T4.

## Kill criterion

Kill and fix before any real RL if any of these happens:
- `reward_sane` fails (the reward cannot recognise its own references).
- The verdict is FAIL for a reason other than reward_improved.
- The reload mismatches.
- RL OOMs on a T4 at these settings.

A FAIL on reward_improved alone is a tuning result (lr, G, steps, beta), not a pipeline bug. It is recorded and
tuned once; if it fails a second time, the reward or the advantages are suspect.

## Measurement

Read `rl_out/result_card.json` and `sft_out/result_card.json` from the kernel output
(`scripts/kaggle_push.py alignment-output`). Record:
- the checks;
- greedy reward before and after;
- the reward curve, KL and completion length per step;
- the fraction of skipped groups;
- seconds and peak VRAM.
