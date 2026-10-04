# E120 — GLM-4.6V as the agent model (image-led perception)

- **Date opened:** 2026-10-04
- **Axis varied:** model (backbone), and where the picture enters the conversation
- **Baseline:** current Sarbloh-Experimentation agent on Qwen3.8-Flash-Next (SGLang NVFP4), the run of 2026-10-04:
  8 games at concurrency 8, 69 of 183 levels, mean score ~20 (UNCONFIRMED until the run is in the ledger)
- **Split:** stage 1 on the dev games' level-1 frames; stage 2 on the regression games ls20, tr87, lp85
- **Status:** written 2026-10-04 at the owner's order. Not run. No Kaggle dataset of GLM-4.6V exists yet (see Weights).

## Why

Owner's Gemini experiment (2026-09-29, unlogged until now; recorded in `areas/arc-agi` memory and here):

| Input given to Gemini | Actions to clear the level |
| --- | --- |
| raw grid + action space | 200-300, game lost |
| + image of the board | 12-24 |
| + explanation of the board (bars, walls, colour rotations), no goal | 12 |
| + "reason with code" | 6 |

The largest single jump came from adding the image. SARBLOH's perception (picture + letters + segmentation +
briefing) was built on that result. GLM-4.6V is a vision-first model with native image tool calling (images as
tool inputs and results).

## Hypothesis

GLM-4.6V reads SARBLOH's picture + briefing more correctly than Qwen3.8-Flash-Next, and that reading converts into
more levels per hour, enough to outweigh its slower decode (12B active vs 6B) and shorter context (128k vs 262k).

Two parts, tested separately, because "having an image matters" (shown) is not "a better image model matters more
than everything else" (not shown):
1. **Vision:** GLM reads the frames better (stage 1).
2. **Conversion:** that reading wins levels inside the time budget (stage 2).

## Mechanism

**Stage 1 — perception-only probe (no play, ~1 GPU hour per model).**
For each dev game, level 1: the first frame exactly as the agent sees it (picture + letters + objects + briefing).
Five questions per frame, answers written by the owner as the key:
1. which object is the player (or what is controlled);
2. which object is the goal;
3. which strip is a timer / moves bar;
4. which objects are walls / blocking terrain;
5. which pairs of objects have the same shape (any rotation, colour or size).
Score each model against the key. Same prompt, same frames, thinking on, same max tokens.
Second pass: the same five questions on the first frame of every level the 2026-10-04 run failed, to see whether
those failures start with misreading the frame.

**Stage 2 — A/B in the harness (only if stage 1 passes).**
Same harness, same 3 regression games, same wall clock, same concurrency, both models:
- Flash-Next (baseline config) vs GLM-4.6V;
- in **both** arms the picture is returned inside the `act` tool result instead of a separate user message
  (`agent._tool_act`, `perception.image_message`), so GLM's image tool calling is actually used and the
  comparison stays fair;
- GLM: `--context-length` 131072 is above its 128k window: set 128k, trigger 60k unchanged; tool parser = the
  GLM one in the SGLang build (UNCONFIRMED name, check `--tool-call-parser` list), thinking per GLM's template.

## Prediction

Committed before the run:
- Stage 1: GLM answers at least 10% more of the 5 x N questions correctly than Flash-Next.
- Stage 2: GLM solves more levels than Flash-Next on the 3 games in the same wall clock, with fewer turns per
  solved level.

## Kill criterion

Binding.
- Stage 1: GLM's score is not at least 10% above Flash-Next's -> stop, no stage 2. Vision is not where
  Flash-Next is short.
- Stage 2: GLM does not beat Flash-Next on RHAE over the 3 games -> drop GLM, even if its perception errors are
  lower (better reading that does not convert means perception was not the bottleneck; record that).
- Infrastructure: if no 4-bit GLM-4.6V serves on one RTX PRO 6000 with images on and native tool calls passing
  the server smoke test, stop and record why.

## Measurement

- Stage 1: correct answers per question type per model; tokens and seconds per frame.
- Stage 2: levels solved, RHAE (`eval/official_score.py`), turns per solved level, seconds per turn, output tok/s
  per game at concurrency, compactions, LLM failures, and the rate of wrong click coordinates / object references
  (from `transcript.jsonl` tool errors and refused acts).

## Weights (checked 2026-10-04)

No Kaggle dataset of GLM-4.6V in any 4-bit format was found. Hugging Face has:
- `vivek51/GLM-4.6V-NVFP4`: NVFP4, 64 GB on disk, vision encoder kept in original precision, served with vLLM,
  78 tok/s single GPU on an RTX PRO 6000 Blackwell (model card); MMLU -2.45% vs BF16. SGLang support UNCONFIRMED.
  Card notes multi-Blackwell issues needing a vLLM fork.
- `cyankiwi/GLM-4.6V-AWQ-4bit`: AWQ int4, vLLM >= 0.12.0 and SGLang >= 0.5.6.post1 listed, vision kept.
- Licence of the base model: MIT.
To use either on Kaggle (internet off), it must be uploaded as a private dataset first (the same way as the
wheelhouse: a notebook with internet on, or the CLI from a local download).

## Order of work

1. Pick the checkpoint: the one SGLang 0.5.19 (Pennyroyal wheels) serves with images + tool calls; AWQ is the
   safer first try on SGLang, NVFP4 on vLLM.
2. Upload as a private Kaggle dataset; add a `glm` spec + profile to `harness/llm/` (new spec file like
   `qwen.py`).
3. Server smoke test (tool call + one image) -> stage 1 -> stage 2.
