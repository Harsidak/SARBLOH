# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")   # E109 exists only in the copy so far
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""E109 SGLang serving without a GPU (experiments/E109_sglang_serving): the Qwen SGLang profiles and chain, the
spec's window and vision for SGLang profiles, the launch command a runtime with its own python builds, the
SGLang metric names in the watchdog (and the health fallback), and the offline installer's steps (fake wheelhouse,
fake uv). The real start is the Kaggle run."""
import io
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from harness.llm import qwen, sglang, vllm
from harness.llm.vllm import VllmServer, parse_metrics, token_counters

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


spec = qwen.SPEC
# --- 1. profiles and chain ---------------------------------------------------------------------------------------
ok("default_chain_unchanged", spec.profile_chain == ["qwen_fp8_vision", "qwen_fp8"])
ok("sglang_chain_known", all(p in spec.profiles for p in qwen.SGLANG_CHAIN) and qwen.SGLANG_CHAIN[-2:] ==
   ["qwen_fp8_vision", "qwen_fp8"])
for p in ("sglang_fp8_mtp_vision", "sglang_fp8_vision", "sglang_fp8"):
    prof = spec.profiles[p]
    ok(f"{p}_runtime", prof["runtime"] == "pennyroyal" and prof["model_dataset"] == qwen.FP8)
    ok(f"{p}_window", spec.max_model_len(p) == 65536)
    ok(f"{p}_tools", spec.has_tool_parser(p) and prof["flags"]["--tool-call-parser"] == "qwen3_coder")
    ok(f"{p}_metrics_on", "--enable-metrics" in prof["flags"])
    ok(f"{p}_no_vllm_flags", not any(k in prof["flags"] for k in ("--max-model-len", "--limit-mm-per-prompt",
                                                                  "--enable-auto-tool-choice")))
ok("vision_flags", spec.has_vision("sglang_fp8_mtp_vision") and spec.has_vision("sglang_fp8_vision")
   and not spec.has_vision("sglang_fp8"))
ok("vllm_vision_unchanged", spec.has_vision("qwen_fp8_vision") and not spec.has_vision("qwen_fp8")
   and spec.max_model_len("qwen_fp8") == 65536)
mtp = spec.profiles["sglang_fp8_mtp_vision"]["flags"]
ok("mtp_flags", mtp["--speculative-algorithm"] == "NEXTN" and mtp["--speculative-num-draft-tokens"] == "4"
   and "--speculative-draft-model-path" not in mtp)
ok("no_flash_next_flags", not any(k in f for f in (p["flags"] for n, p in spec.profiles.items() if n.startswith("sglang"))
                                  for k in ("--quantization", "--ple-offload-embedding", "--speculative-token-map")))
ok("runtime_registered", callable(spec.runtimes["pennyroyal"]))

# --- 2. launch with a runtime that brings its own python ----------------------------------------------------------
work = Path(tempfile.mkdtemp(prefix="e109_"))
model_dir = work / "model"
model_dir.mkdir()
(model_dir / "config.json").write_text("{}", encoding="utf-8")
cfg = {"port": 8123, "profile_chain": qwen.SGLANG_CHAIN, "startup_timeout_s": 1, "max_inflight": 8}
srv = VllmServer(spec, cfg, work)
fake_rt = {"env": {"PATH": "/x"}, "python": "/tmp/sarbloh-sglang/venv/bin/python", "serve": list(sglang.SERVE),
           "backend": "sglang", "info": {}}
vllm._RUNTIMES["pennyroyal"] = fake_rt
_real_find, _real_popen = vllm.find_kaggle_input, subprocess.Popen
captured = {}


class FakeProc:
    pid = 1

    def poll(self):
        return None


def fake_popen(cmd, **kw):
    captured["cmd"], captured["env"] = cmd, kw.get("env")
    return FakeProc()


vllm.find_kaggle_input = lambda ref: model_dir
vllm.subprocess.Popen = fake_popen
try:
    srv.launch("sglang_fp8_mtp_vision")
finally:
    vllm.find_kaggle_input, vllm.subprocess.Popen = _real_find, _real_popen
cmd = captured["cmd"]
ok("launch_uses_runtime_python", cmd[0] == fake_rt["python"], cmd[:4])
ok("launch_sglang_entry", cmd[1:5] == ["-m", "sglang.launch_server", "--model-path", str(model_dir)], cmd[:6])
ok("launch_common_args", cmd[cmd.index("--served-model-name") + 1] == "qwen3.8" and "8123" in cmd)
ok("launch_flags", cmd[cmd.index("--context-length") + 1] == "65536" and "--enable-metrics" in cmd
   and cmd[cmd.index("--speculative-algorithm") + 1] == "NEXTN")
ok("backend_set", srv.backend == "sglang" and srv.max_model_len == 65536 and srv.vision)
ok("runtime_env_used", captured["env"]["PATH"] == "/x" and captured["env"]["HF_HUB_OFFLINE"] == "1")

# --- 3. metrics: SGLang names, health fallback --------------------------------------------------------------------
text = ("sglang:prompt_tokens_total{model_name=\"q\"} 1200\nsglang:generation_tokens_total{model_name=\"q\"} 300\n"
        "sglang:num_running_reqs{model_name=\"q\"} 2\n")
ok("sglang_counters", token_counters(parse_metrics(text)) == (1500.0, 2.0))
# E109 ls20: one 12k-token thinking request; the *_total counters only move when it finishes, realtime moves per step
mid = [("sglang:prompt_tokens_total 1200\nsglang:generation_tokens_total 300\nsglang:num_running_reqs 1\n"
        f"sglang:realtime_tokens_total{{mode=\"decode\"}} {n}\nsglang:realtime_tokens_total{{mode=\"prefill_compute\"}} 6658\n")
       for n in (4000, 9000)]
a, b = (token_counters(parse_metrics(t)) for t in mid)
ok("sglang_long_request_moves", a[0] != b[0] and a[1] == b[1] == 1.0, (a, b))
ok("vllm_counters",token_counters(parse_metrics("vllm:prompt_tokens_total 5\nvllm:generation_tokens_total 7\n"
                                                 "vllm:num_requests_running 1\n")) == (12.0, 1.0))
srv.process = FakeProc()
srv.metrics = lambda timeout=10: None
srv.healthy = lambda timeout=5: True
st = {}
ok("sglang_no_metrics_but_healthy_is_fine", all(srv.check(st) is None for _ in range(6)) and st["failures"] == 0)
srv.healthy = lambda timeout=5: False
st = {}
res = [srv.check(st) for _ in range(4)]
ok("sglang_dead_is_unresponsive", res[-1] == "unresponsive", res)
srv.backend = "vllm"
srv.healthy = lambda timeout=5: True
st = {}
ok("vllm_keeps_metrics_rule", [srv.check(st) for _ in range(4)][-1] == "unresponsive")
srv.backend = "sglang"
seq = iter([parse_metrics(text)] * 20)
srv.metrics = lambda timeout=10: next(seq)
srv.wd["freeze_after_s"] = 0.0
st = {}
srv.check(st)
ok("sglang_freeze_detected", srv.check(st) == "frozen")

# --- 4. installer: wheelhouse layout, uv from its wheel, install steps --------------------------------------------
ds = work / "pennyroyal"
(ds / "wheels").mkdir(parents=True)
(ds / "requirements.lock").write_text("sglang==0.5.19\n", encoding="utf-8")
(ds / "wheels" / "sglang-0.5.19+gd00d88efc8d6-cp312-cp312-linux_x86_64.whl").write_bytes(b"x")
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    z.writestr("uv-0.12.20.data/scripts/uv", b"\x7fELFfake")
(ds / "wheels" / "uv-0.12.20-py3-none-manylinux_2_17_x86_64.whl").write_bytes(buf.getvalue())
ok("wheelhouse_found", sglang.wheelhouse(ds) == (ds / "wheels", ds / "requirements.lock"))
nested = work / "nested"
(nested / "inner").mkdir(parents=True)
(ds / "wheels").rename(nested / "inner" / "wheels")
(ds / "requirements.lock").rename(nested / "inner" / "requirements.lock")
ok("wheelhouse_one_level_down", sglang.wheelhouse(nested)[0] == nested / "inner" / "wheels")
uvb = sglang.uv_binary(nested / "inner" / "wheels", work / "bin" / "uv")
ok("uv_extracted", uvb.read_bytes().startswith(b"\x7fELF"))
runs = []
_real_run, _real_prefix = sglang._run, sglang.PREFIX
sglang._run = lambda cmd, env=None, check=True: runs.append([str(c) for c in cmd]) or subprocess.CompletedProcess(cmd, 0, "", "")
sglang.PREFIX = work / "sgl"
try:
    py = sglang.install("dfranzen/pennyroyal-v253", lambda ref: nested)
    ok("install_steps", len(runs) == 2 and runs[0][1] == "venv" and runs[1][1:3] == ["pip", "install"]
       and "--no-index" in runs[1] and runs[1][-1].endswith("requirements.lock"), runs)
    ok("install_python_path", py == work / "sgl" / "venv" / "bin" / "python")
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("", encoding="utf-8")
    runs.clear()
    sglang.install("dfranzen/pennyroyal-v253", lambda ref: nested)
    ok("install_cached_by_stamp", runs == [], runs)
finally:
    sglang._run, sglang.PREFIX = _real_run, _real_prefix

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys.exit(1 if FAILS else 0)
