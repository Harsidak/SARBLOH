# E112: Qwen3.8-27B NVFP4 + built-in MTP on SGLang (Pennyroyal 0.5.19), serving bench

Owner go: 2026-10-02 ("start implementing B ... don't stop until you got 500 toks"). It replaces E109's FP8 arm,
which was killed: the untuned SGLang fallback measured 45.3 tok/s single stream and 342 tok/s at 8 streams, and MTP
was never tested because `--mamba-ssm-dtype bfloat16` was missing.

## Mechanism
At 10 concurrent requests, decoding is bound by memory bandwidth: every step reads the whole dense 27B once.
`nvidia/Qwen3.8-27B-NVFP4` is ModelOpt MIXED_PRECISION:
- MLP and lm_head are NVFP4 (group 16).
- Attention and Gated-DeltaNet projections are FP8.
- MTP is unquantized.

That is about 18 GB read per step instead of about 29 GB for FP8, roughly 1.6x fewer bytes. FP4 GEMMs also run
natively on SM120.

NEXTN speculation on the model's own MTP layer turns each weight read into up to 4 verified tokens per stream. The
flags are the milestone-2 reference's tuning for this card:
- FP8 KV cache, 64-token pages;
- decode CUDA graphs for batch sizes 1-10;
- mamba radix cache (extra_buffer);
- flashinfer GDN kernels with a bf16 SSM state;
- `--gdn-mtp-cache-mode none`.

## Prediction
`decode_10` (forced 512 tokens per stream) and `think_10` (thinking-on puzzle, Duck sampling) are each predicted at
**>= 500 tok/s aggregate** for `sglang_nvfp4_mtp_vision`. Without MTP, `sglang_nvfp4_vision` is predicted at about
350-450 tok/s. MTP mean acceptance length is predicted at 2.0 or more.

## Arms (one Kaggle run, bench mode, no games)
`kaggle/experimental/002_prime.ipynb`, `SERVE_BENCH`; code in `Sarbloh-Arc/prime/llm/{qwen,sglang,vllm,bench}.py`.
1. `sglang_nvfp4_mtp_vision`: NEXTN with 3 steps.
2. `sglang_nvfp4_mtp_frspec_vision`: arm 1 plus the reference's FR-Spec 64k hot-token map. The map was built for
   the Flash-Next tokenizer, and our tokenizer hash differs, so it is UNCONFIRMED. A bad map only costs acceptance.
3. `sglang_nvfp4_mtp5_vision`: NEXTN with 5 steps.
4. `sglang_nvfp4_vision`: no speculation, as the control.

## Kill criterion
If no arm reaches 500 tok/s on `think_10` after two tuning iterations, the NVFP4 27B path is killed as the route to
500 tok/s, and option C (the Flash-Next MoE, as in the reference) is next.

A profile that fails to start or fails the tool smoke test is recorded and does not count as a measurement.

Quality on ARC is NOT measured here; a games run on the winning profile follows.
