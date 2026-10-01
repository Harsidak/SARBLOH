# E007: one vLLM server layer for Qwen3.8-27B and Gemma-4-31B, robust and faster

- **Date opened:** 2026-09-30
- **Axis varied:** substrate (serving layer); no agent or prompt change
- **Baseline:** E006 Kaggle run (Gemma-4-31B-IT-NVFP4, profile `gemma_fast`): 233 tok/s throughput probe, 3 APIServer
  freezes, `max_restarts=2` exhausted, then 900 s x 4 client retries blocked every game for about 30 minutes
- **Split:** none. This is a serving benchmark with no games played, so there is no RHAE claim.

## Mechanism

`Sarbloh-Arc/prime/llm/` is split into a model-free architecture and one file per model:
- `spec.py` holds `ModelSpec` and a registry; config `model: "gemma" | "qwen"` picks one.
- `gemma.py` and `qwen.py` hold everything model-specific:
  - weights datasets;
  - profile chain and flags;
  - parsers;
  - sampling;
  - thinking policy;
  - tool-argument repair.
- `vllm.py` is generic. Robustness changes:
  1. Freeze detection. `/metrics` token counters must advance while requests are running, `/health` must answer, and
     the process must be alive.
  2. `PYTHONFAULTHANDLER=1`, and SIGABRT before SIGKILL, so every freeze leaves Python stacks in the server log.
  3. A restart budget based on remaining time, not a fixed count of 2. After 2 freezes on one profile, the server
     moves to the next (safer) profile. Before relaunching it waits for the GPU memory to be released.
  4. `ServerGate` (`client.py`):
     - the client waits for "ready" instead of burning retries;
     - killing the server closes in-flight sockets, so blocked requests fail fast and retry after the restart;
     - requests are capped by the game deadline.
  5. The same gate limits requests in flight and ramps back up after a restart (2 → full over about 1 minute).
  6. `server_events.jsonl`: one line per start, ready, freeze, dump, restart, fallback and stop.
  7. vLLM site-packages move to `/tmp` (out of `/kaggle/working`).
- Speed levers:
  - **Qwen:** NVFP4 weights with native MTP speculative decoding (3 tokens).
  - **Gemma:** n-gram prompt-lookup speculative decoding.
  - **Both:** larger prefill chunks (`max-num-batched-tokens` 16384) and the vision encoder off.
- `bench.py` measures each profile the same way:
  - decode speed, single stream;
  - aggregate speed at 10 streams;
  - an agent-shaped multi-turn workload (prefix-cache hit rate, time per turn);
  - spec-decode acceptance;
  - a tool-call check.

  `stress()` then runs the winner under agent-shaped load, freezes the engine on purpose (SIGSTOP), and checks that
  the watchdog detects it, dumps stacks, restarts, and the client recovers with no manual step.

## Why it should work

- E006 lost about 30 minutes per game because a frozen server looked alive and the client retried blindly. A progress check plus fail-fast
  sockets turns a freeze into a restart of about 3-5 minutes.
- Decode on a 27-31B model is bound by weight bandwidth, so NVFP4 (about 4.5 bits) and MTP (several tokens per forward pass) both raise tokens per second.
- The agent's prompts are long and grow by appending, so larger prefill chunks and prefix caching cut time per turn.

## Prediction

Committed before the run.
- **Robustness:** the SIGSTOP freeze is detected within 150 s and the server is ready again within 10 min. The stress
  client finishes with 0 unrecovered requests. The log contains a faulthandler stack dump.
- **Speed:** the best Qwen profile reaches at least 1.3x E006's 233 tok/s aggregate (about 300 tok/s or more at 10 streams).
  The best Gemma profile reaches at least 233 tok/s at 10 streams (no regression).
- The agent-shaped workload reaches a prefix-cache hit rate of at least 50% on turns 2 onward.

## Kill criterion

Binding.
- **Kill the watchdog redesign** if the injected freeze is not recovered without manual action in 15 min on either model.
- **Kill a speed profile** (it leaves the chain) if it fails to start, fails the tool-call check, or is slower at 10
  streams than the plain profile below it.
- **Kill "Qwen3.8 as the next agent model"** only if no Qwen profile starts on vLLM 0.19.0. Slower-than-Gemma alone does not kill it.
  The agent score decides that, in a later experiment.

## Measurement

- `vllm_bench.json` (one row per profile) and `server_events.jsonl`, from the Kaggle notebook
  `kaggle/experimental/007_vllm_bench.ipynb`, pushed with `scripts/kaggle_push.py vllm-bench`.
- Local: `tests/unit/test_llm_server.py`. A fake OpenAI server that freezes checks the gate, the fail-fast retry, the
  freeze detector, the limiter ramp and the Gemma tool-argument repair.
- Seeds: 1. It is a throughput benchmark, so numbers are reported per profile together with the flags that produced them.

## Weights chosen (UNCONFIRMED on vLLM 0.19 until this run)

| Profile | Weights | Kaggle source | Why |
| --- | --- | --- | --- |
| `qwen_nvfp4_mtp` | Qwen3.8-27B NVFP4 + bf16 MTP head | model `overseer66/qwen3-8-27b-nvfp4` (PyTorch/default/1) | Only NVFP4 Qwen3.8 on Kaggle with MTP weights. Community quant: provenance UNCONFIRMED, and 46.9 GB listed although its weight files total about 23.4 GB |
| `qwen_fp8_mtp`, `qwen_fp8` | Official `Qwen/Qwen3.8-27B-FP8` rev 017b9c7a, with `mtp.safetensors` | dataset `fumiyauchiyama/qwen3-8-27b-fp8-hf-snapshot` | Unmodified official snapshot, flat layout, the Duck's quantisation type |

`nvidia/Qwen3.8-27B-NVFP4` (official, 22 GB, MTP included) is not on Kaggle yet. If the community NVFP4 fails, mirroring the official one
is the next step. That is the owner's upload.

## Addendum, v4 (2026-09-30, before code): Qwen3.8-Flash-Next NVFP4 replaces Qwen3.8-27B

**Owner decision:** the Qwen 3.8 model is `RadixArk/Qwen3.8-Flash-Next-NVFP4` rev 7b719225, the model the owner verified
the Duck harness uses. It is the only Qwen 3.8 we run; otherwise Gemma. The model is a `qwen4_exp` MoE: 512 experts,
10 active, 48 layers, 1 full-attention layer in 4. It has about 120B parameters in 135 GB, and a 20M-row FP8
n-gram embedding (PLE) that is offloaded to CPU.

**v3 result behind the switch:** both 27B NVFP4 profiles failed to start, which hits the "fails to start" kill
criterion, so they leave the chain. The cause, from `vllm-server.log`: `There is no module or parameter named
'lm_head.weight_scale' in Qwen3_5ForCausalLM`. The unsloth quant's FP8 lm_head does not load on vLLM 0.19.

**Mechanism:** stock vLLM 0.19 cannot serve `qwen4_exp`. The profile therefore runs Keith Tyser's pinned runtime and
does not use our wheelhouse. Its pieces:
- the Docker image layers `vllm/vllm-openai:qwen38-flash-next` (vLLM `0.1.dev20073+g8e685d198`, torch 2.13 cu130),
  from dataset `keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1`;
- unpacked to `/tmp` and patched for the PLE loader by that bundle's own `serving_setup.py` functions, taken from
  dataset `keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1` (MIT per its SOURCE_IDENTITY, "unknown" on Kaggle:
  UNCONFIRMED);
- run under the host's Python 3.12.

`ModelSpec.runtimes` carries this. `vllm.py` stays model-free, and the watchdog, gate and stress test are unchanged.

**Profiles:**

| Profile | What it is |
| --- | --- |
| `qwen_flash` | The Duck/Keith recipe, flag for flag: 32k context, 28 seqs, 8192 batched tokens, MTP 3, prefix caching off, async scheduling |
| `qwen_flash_pc` | Adds prefix caching |
| `qwen_flash_pc_64k` | Adds prefix caching and a 65k context. Our agent's compaction (trigger 40k, reserve 16k) cannot work in a 32k window |

**Prediction:**
- `qwen_flash` starts, and its tool-call check passes natively.
- At 10 streams it decodes at least as fast as v3's best Qwen (`qwen_fp8_mtp`, 648 tok/s), because about 10 of 512 experts are active per token.
- `qwen_flash_pc` brings the agent_like later-turn time down to 4 s or less (v3: `qwen_fp8` with prefix caching 2.0 s, `qwen_fp8_mtp` without it 8.1 s).
- The stress test recovers with 0 failed requests.

**Kill criteria (binding):**
- A Flash profile that fails to start or fails the tool check leaves the chain.
- If `qwen_flash` itself fails to start, Flash-Next is killed as the Qwen model until the cause is fixed. The FP8 27B
  profiles stay defined as the measured fallback.
- `qwen_flash_pc` / `qwen_flash_pc_64k` are kept only if they are not slower than `qwen_flash` at 10 streams, and are faster on agent_like.
