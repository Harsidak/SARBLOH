# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_ROOT, _os.path.join(_ROOT, "Sarbloh-Experimentation")):  # deepseek_v4.py lives only there
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""DeepSeek-V4-Flash on SGLang (Sarbloh-Experimentation/harness/llm/deepseek_v4.py), without a GPU: the spec is
registered, the flags are the ones read in the 0.5.20 wheel, the thinking switch and effort reach the server in DSV4's
terms (and Qwen's requests are unchanged), the text view drops Vision-Exp's vision tensors byte for byte, and the
bench helpers read the prefix-cache count from a reply's usage."""
import json
import struct
import sys
import tempfile
from pathlib import Path

from harness.config import build_config
from harness.llm import bench, deepseek_v4 as ds
from harness.llm.client import LLM
from harness.llm.spec import get_spec

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


spec = ds.SPEC
REAP = ["dsv4_reap150", "dsv4_reap150_marlin", "dsv4_reap150_pennyroyal"]
OFF = ["dsv4_vision_exp_offload_dspark", "dsv4_vision_exp_offload"]
# --- 1. spec and config ----------------------------------------------------------------------------------------
ok("registered", get_spec("deepseek_v4") is spec and spec.name == "deepseek_v4")
cfg = build_config({"model": "deepseek_v4"})
ok("config_picks_it", cfg["llm"]["model"] == "deepseek-v4-flash" and cfg["llm"]["temperature"] == 1.0
   and cfg["llm"]["top_p"] == 0.95 and cfg["llm"]["top_k"] is None
   and cfg["llm"]["chat_template_kwargs"] == {"thinking": True}, cfg["llm"])
ok("chain_is_reap", spec.profile_chain == REAP and all(p in spec.profiles for p in REAP + OFF))
ok("offload_not_in_chain", not set(OFF) & set(spec.profile_chain))
ok("api_adapter_untouched", __import__("harness.llm.deepseek", fromlist=["DeepSeekLLM"]).DeepSeekLLM is not None)

# --- 2. flags -------------------------------------------------------------------------------------------------
for p in REAP + OFF:
    prof, f = spec.profiles[p], spec.profiles[p]["flags"]
    ok(f"{p}_weights", prof["model_dataset"] == (ds.REAP if p in REAP else ds.VISION_EXP))
    ok(f"{p}_text_only", spec.has_vision(p) is False)
    ok(f"{p}_window", spec.max_model_len(p) == 139264)
    ok(f"{p}_dsv4_pages", f["--page-size"] == "256" and f["--kv-cache-dtype"] == "fp8_e4m3")
    ok(f"{p}_ten_requests", f["--max-running-requests"] == "10" and f["--cuda-graph-max-bs-decode"] == "10"
       and [int(b) for b in f["--cuda-graph-bs-decode"]] == list(range(1, 11)))
    ok(f"{p}_parsers", spec.has_tool_parser(p) and f["--tool-call-parser"] == "deepseekv4"
       and f["--reasoning-parser"] == "deepseek-v4")
    ok(f"{p}_effort_profile", json.loads(f["--json-model-override-args"]) == {"dsv4_reasoning_effort_profile": "official"})
    ok(f"{p}_cache_report", "--enable-cache-report" in f and "--enable-metrics" in f and f["--schedule-policy"] == "lpm")
    ok(f"{p}_thinks_by_default", prof["env"] == {"SGLANG_DEFAULT_THINKING": "1"})
    ok(f"{p}_runtime", prof["runtime"] in spec.runtimes and callable(spec.runtimes[prof["runtime"]]))
    ok(f"{p}_flag_values_str", all(v is None or isinstance(v, (str, list)) for v in f.values()))
ok("reap_no_speculation", all("--speculative-algorithm" not in spec.profiles[p]["flags"] for p in REAP))
ok("reap_moe_runners", [spec.profiles[p]["flags"]["--moe-runner-backend"] for p in REAP]
   == ["flashinfer_mxfp4", "marlin", "flashinfer_mxfp4"])
ok("reap_wheels", [spec.profiles[p]["runtime"] for p in REAP] == ["sglang_0520", "sglang_0520", "sglang_pennyroyal"])
ok("offload_flags", all(spec.profiles[p]["flags"]["--cpu-offload-gb"] == "80" for p in OFF)
   and spec.profiles[OFF[0]]["flags"]["--speculative-algorithm"] == "DSPARK"
   and "--speculative-algorithm" not in spec.profiles[OFF[1]]["flags"])
ok("profiles_do_not_share_flags", spec.profiles[REAP[0]]["flags"] is not spec.profiles[REAP[1]]["flags"]
   and "--cpu-offload-gb" not in ds.SGL)

# --- 3. thinking and effort in DSV4's terms; Qwen's requests unchanged -------------------------------------------
tk = spec.template_kwargs
ok("effort_map", tk({"reasoning_effort": "xhigh"}) == {"reasoning_effort": "max"}
   and tk({"reasoning_effort": "medium"}) == {"reasoning_effort": "high"}
   and tk({"reasoning_effort": "low"}) == {"reasoning_effort": "low"})
ok("thinking_switch_renamed", tk({"enable_thinking": False}) == {"thinking": False}
   and tk({"enable_thinking": True, "reasoning_effort": "xhigh"}) == {"thinking": True, "reasoning_effort": "max"})
ok("unknown_effort_dropped", tk({"reasoning_effort": "none"}) == {})


class Capture(LLM):
    def _post(self, path, body, timeout):
        self.body = body
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def body_of(model, **kw):
    s = get_spec(model)
    c = build_config({"model": model})["llm"]
    llm = Capture({**c, "base_url": "http://x/v1"}, spec=s)
    llm.chat([{"role": "user", "content": "hi"}], **kw)
    return llm.body


b = body_of("deepseek_v4", template_kwargs={"reasoning_effort": "xhigh"})
ok("client_ds_default_thinking", b["chat_template_kwargs"] == {"thinking": True, "reasoning_effort": "max"},
   b["chat_template_kwargs"])
b = body_of("deepseek_v4", thinking=False)
ok("client_ds_thinking_off", b["chat_template_kwargs"] == {"thinking": False}, b["chat_template_kwargs"])
ok("client_ds_no_top_k", "top_k" not in b and b["temperature"] == 1.0)
b = body_of("qwen_flash_next", thinking=False, template_kwargs={"reasoning_effort": "low"})
ok("client_qwen_unchanged", b["chat_template_kwargs"] == {"enable_thinking": False, "reasoning_effort": "low"},
   b["chat_template_kwargs"])
b = body_of("nemotron", thinking=False, template_kwargs={"reasoning_effort": "low"})
ok("client_nemotron_unchanged", b["chat_template_kwargs"] == {"enable_thinking": False, "truncate_history_thinking": False,
                                                              "medium_effort": True}, b["chat_template_kwargs"])


# --- 4. bench helpers ----------------------------------------------------------------------------------------
class FakeServer:
    def __init__(self, s):
        self.spec = s


ok("bench_ctk_ds", bench._ctk(FakeServer(spec), True) == {"thinking": True})
ok("bench_ctk_qwen", bench._ctk(FakeServer(get_spec("qwen_flash_next")), False) == {"enable_thinking": False})
ok("bench_cached", bench._cached({"usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 94}}})
   == (100, 94))
ok("bench_cached_unreported", bench._cached({"usage": {"prompt_tokens": 100}}) == (100, -1))
ok("bench_tools_workload", "agent_like_tools" in bench.profile_bench.__code__.co_consts
   or any("agent_like_tools" in str(c) for c in bench.profile_bench.__code__.co_consts))

# --- 5. the Vision-Exp text view ------------------------------------------------------------------------------
def write_st(path, tensors):  # a safetensors file from {name: (dtype, shape, raw bytes)}
    header, data = {"__metadata__": {"format": "pt"}}, b""
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + data)


def read_st(path):
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    return {k: (v["dtype"], v["shape"], raw[8 + n + v["data_offsets"][0]:8 + n + v["data_offsets"][1]])
            for k, v in header.items() if k != "__metadata__"}


work = Path(tempfile.mkdtemp(prefix="dsv4-test-"))
root = work / "models" / "trnmtin" / "deepseek-v4-flash-vision-exp" / "pytorch" / "hf-original" / "1"
meta = root / "hf-metadata-be724c5975d4"
(meta / "encoding").mkdir(parents=True)
(meta / "inference").mkdir()
(meta / "encoding" / "encoding_dsv4.py").write_text("DEFAULT_REASONING_EFFORT = 'low'\n", encoding="utf-8")
(meta / "inference" / "config.json").write_text("{}", encoding="utf-8")  # deeper: must not be taken as the model dir
(meta / "config.json").write_text(json.dumps({"architectures": ["DeepseekV4ForCausalLM"]}), encoding="utf-8")
(meta / "tokenizer.json").write_text("{}", encoding="utf-8")
S1, S2 = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
T1 = {"embed.weight": ("BF16", [2, 2], b"E" * 8), "vision.blocks.0.mlp.w1.weight": ("BF16", [1, 2], b"V" * 4),
      "layers.0.attn.wq_a.weight": ("F8_E4M3", [4], b"Q" * 4), "aligner.w1.bias": ("BF16", [2], b"A" * 4),
      "image_pad": ("BF16", [2], b"I" * 4), "layers.0.ffn.experts.0.w1.weight": ("U8", [3], b"X" * 3)}
T2 = {"layers.1.attn.wq_a.weight": ("F8_E4M3", [4], b"R" * 4), "head.weight": ("BF16", [2], b"H" * 4)}
write_st(root / S1, T1)
write_st(root / S2, T2)
wm = {**{k: S1 for k in T1}, **{k: S2 for k in T2}}
(meta / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}), encoding="utf-8")
(root / "_hf_mirror_manifest.json").write_text("{}", encoding="utf-8")

links = {}
_real_symlink = Path.symlink_to
Path.symlink_to = lambda self, target, *a, **k: (links.__setitem__(self.name, Path(target)),
                                                  self.write_text(str(target), encoding="utf-8"))
try:
    view = ds.prepare_text_view(root, work / "view")
    kept = read_st(view / S1)
    TEXT1 = {k: v for k, v in T1.items() if not k.startswith(("vision.", "aligner.", "image_"))}
    ok("view_drops_vision", sorted(kept) == sorted(TEXT1), sorted(kept))
    ok("view_text_bytes", all(kept[k] == TEXT1[k] for k in TEXT1))
    idx = json.loads((view / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    ok("view_index", sorted(idx) == sorted([*TEXT1, *T2]) and idx["head.weight"] == S2)
    ok("view_links_untouched_shard", links.get(S2) == (root / S2).resolve())
    ok("view_links_configs", links.get("config.json") == (meta / "config.json").resolve()
       and links.get("encoding") == (meta / "encoding").resolve() and "tokenizer.json" in links)
    (root / S2).unlink()
    try:
        ds.prepare_text_view(root, work / "view2")
        ok("view_rejects_missing_shard", False)
    except RuntimeError as exc:
        ok("view_rejects_missing_shard", "missing" in str(exc), exc)
finally:
    Path.symlink_to = _real_symlink

print(f"\n{CHECKS[0] - len(FAILS)}/{CHECKS[0]} passed")
if FAILS:
    print("FAILED:", FAILS)
    sys.exit(1)
