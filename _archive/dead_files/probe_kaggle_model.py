# Load test for Qwen3.6-27B-NVFP4 -- paste straight into a Jupyter cell.
# NVFP4 needs vLLM >= 0.19.0 and a Blackwell GPU (sm_100+), per the unsloth card.
# Runs fully offline / in-process: no network call.
#
# NOT PART OF THE LOCAL TEST SUITE. Despite the `test_` prefix this is a
# KAGGLE-ONLY environment probe -- it needs vLLM and a Blackwell card, neither of
# which exists on the dev box, so it exits early there instead of crashing and
# making the suite look broken. The component tests are:
#   test_metrics.py  test_eyes.py  test_clickplanner.py  test_stategraph.py
#   test_griddsl.py  test_synth.py  test_livelock.py     test_goalmodel.py

import os, time, json, sys

MODEL_PATH = "/kaggle/input/datasets/banwait13/models/Qwen3.6-27B-NVFP4"
PROMPT = 'Reply with JSON only: {"ok": true, "n": 7}'

# --- what is actually on disk / on the GPU -----------------------------------
print("path   :", MODEL_PATH)
print("exists :", os.path.isdir(MODEL_PATH))

if os.path.isdir(MODEL_PATH):
    files = sorted(os.listdir(MODEL_PATH))
    weights = [f for f in files if f.endswith((".safetensors", ".bin", ".gguf"))]
    print("files  :", len(files), files[:12])
    print("weights:", len(weights), "shard(s)")
    if not weights:
        print("  !! no weight shards -- the download is incomplete")
    if os.path.isfile(MODEL_PATH + "/config.json"):
        cfg = json.load(open(MODEL_PATH + "/config.json", encoding="utf-8"))
        qc = cfg.get("quantization_config", {})
        print("type   :", cfg.get("model_type"), "| quant:", qc.get("quant_method"), qc.get("format"))
else:
    print("  !! directory missing -- attach the dataset and check the mounted name")

import torch
print("torch  :", torch.__version__, "cuda =", torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print("gpu    :", torch.cuda.get_device_name(0), f"sm_{cap[0]}{cap[1]}",
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB")
    if cap[0] < 10:
        print("  !! NVFP4 needs sm_100+ (Blackwell); this card can't run the 4-bit kernels")

# --- load --------------------------------------------------------------------
try:
    import vllm
except ImportError:
    print("\nSKIP: vLLM is not installed -- this probe only runs on Kaggle.")
    sys.exit(0)
from vllm import LLM, SamplingParams
print("vllm   :", vllm.__version__)

t0 = time.time()
llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=4096,        # small KV cache; the agent sends short prompts
    gpu_memory_utilization=0.85,
    enforce_eager=True,        # skip CUDA-graph capture -> much faster startup
)
print(f"loaded in {time.time() - t0:.0f}s")

# --- generate ----------------------------------------------------------------
tok = llm.get_tokenizer()
text = tok.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,     # thinking is ON by default on Qwen3.6 and eats the budget
)

t0 = time.time()
out = llm.generate([text], SamplingParams(temperature=0.2, max_tokens=128))
print(f"generated in {time.time() - t0:.1f}s")
print("-" * 60)
print(out[0].outputs[0].text.strip())
print("-" * 60)
