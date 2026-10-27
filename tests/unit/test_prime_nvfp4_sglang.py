# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh")   # E112 lives in Sarbloh (the sarbloh-prime notebook)
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""E112 Qwen3.8-27B NVFP4 + MTP on SGLang, without a GPU (experiments/E112_nvfp4_sglang_serving): the NVFP4 profiles
and the chain, the launch command (list-valued flags, ``{model_dir}``), the flat wheelhouse layout of
banwait13/sglangwheels, the install steps and the page-cache precache. With ``SGLANG_SRC`` set to an unpacked sglang
wheel (``.../site-packages/sglang``), every profile flag is also checked against that build's ``ServerArgs`` fields."""
import io
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from harness.llm import qwen, server, sglang
from harness.llm.server import LlmServer

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


spec = qwen.SPEC
NV = [n for n, p in spec.profiles.items() if p.get("model_dataset") == qwen.NVFP4]
# --- 1. profiles and chain ---------------------------------------------------------------------------------------
ok("nv_profiles", set(NV) >= {"sglang_nvfp4_mtp_vision", "sglang_nvfp4_vision", "sglang_nvfp4"}, NV)
ok("chain_known_and_ends_on_vllm", all(p in spec.profiles for p in qwen.SGLANG_CHAIN)
   and qwen.SGLANG_CHAIN[0] == "sglang_nvfp4_mtp5_vision" and qwen.SGLANG_CHAIN[-2:] == ["qwen_fp8_vision", "qwen_fp8"])
for p in NV:
    f = spec.profiles[p]["flags"]
    ok(f"{p}_runtime", spec.profiles[p]["runtime"] == "sglang")
    # Both copies serve 131072 since Sarbloh mirrored the experimentation copy (2026-10-10).
    ok(f"{p}_window", spec.max_model_len(p) == 131072)
    ok(f"{p}_tools", spec.has_tool_parser(p) and f["--tool-call-parser"] == "qwen3_coder")
    ok(f"{p}_bf16_ssm_for_flashinfer", f["--mamba-ssm-dtype"] == "bfloat16"
       and f["--linear-attn-decode-backend"] == "flashinfer")
    ok(f"{p}_quant_autodetected", "--quantization" not in f)  # modelopt -> modelopt_mixed from config.json
    ok(f"{p}_graphs_cover_requests", int(f["--cuda-graph-max-bs-decode"]) >= int(f["--max-running-requests"])
       and max(map(int, f["--cuda-graph-bs-decode"])) == int(f["--cuda-graph-max-bs-decode"]))
    if "--speculative-algorithm" in f:
        ok(f"{p}_mtp_linear_chain", f["--speculative-algorithm"] == "NEXTN" and f["--speculative-eagle-topk"] == "1"
           and int(f["--speculative-num-draft-tokens"]) == int(f["--speculative-num-steps"]) + 1
           and "--speculative-draft-model-path" not in f and f["--gdn-mtp-cache-mode"] == "none")
ok("vision_flags", spec.has_vision("sglang_nvfp4_mtp_vision") and not spec.has_vision("sglang_nvfp4"))
ok("runtime_registered", callable(spec.runtimes["sglang"]))

# --- 2. launch: list flags and {model_dir} ---------------------------------------------------------------------------
work = Path(tempfile.mkdtemp(prefix="e112_"))
model_dir = work / "model"
model_dir.mkdir()
(model_dir / "config.json").write_text("{}", encoding="utf-8")
srv = LlmServer(spec, {"port": 8124, "profile_chain": qwen.SGLANG_CHAIN, "startup_timeout_s": 1, "max_inflight": 8},
                 work)
fake_rt = {"env": {"PATH": "/x"}, "python": "/tmp/sarbloh-sglang/venv/bin/python", "serve": list(sglang.SERVE),
           "backend": "sglang", "info": {}}
server._RUNTIMES["sglang"] = fake_rt
captured = {}


class FakeProc:
    pid = 1

    def poll(self):
        return None


_real_find, _real_popen = server.find_kaggle_input, subprocess.Popen
server.find_kaggle_input = lambda ref: model_dir
server.subprocess.Popen = lambda cmd, **kw: captured.update(cmd=cmd, env=kw.get("env")) or FakeProc()
try:
    srv.launch("sglang_nvfp4_mtp5_vision")
finally:
    server.find_kaggle_input, server.subprocess.Popen = _real_find, _real_popen
cmd = captured["cmd"]
i = cmd.index("--cuda-graph-bs-decode")
ok("list_flag_expanded", cmd[i + 1:i + 11] == [str(n) for n in range(1, 11)] and cmd[i + 11].startswith("--"), cmd[i:i + 12])
ok("model_dir_substituted", cmd[cmd.index("--chat-template") + 1] == f"{model_dir}/chat_template.jinja")
ok("bool_flag_alone", cmd[cmd.index("--weight-loader-prefetch-checkpoints") + 1].startswith("--"))
ok("runtime_python", cmd[0] == fake_rt["python"] and cmd[1:4] == ["-m", "sglang.launch_server", "--model-path"])

# --- 3. flat wheelhouse (banwait13/sglangwheels), install steps, precache -----------------------------------------
flat = work / "sglangwheels"
flat.mkdir()
# Kaggle strips "+" from uploaded names: the lock says torch==2.13.0+cu130, the file is torch-2.13.0cu130-...whl
(flat / "requirements.lock").write_text("sglang==0.5.19+gd00d88efc8d6\ntorch==2.13.0+cu130\n"
                                        "flashinfer-jit-cache==0.6.17+cu130\nsgl-deep-gemm==0.1.5.post3+cu130\n"
                                        "triton==3.7.1\n", encoding="utf-8")
for n in ("sglang-0.5.19gd00d88efc8d6-cp312-cp312-linux_x86_64.whl", "torch-2.13.0cu130-cp312-cp312-manylinux_2_28_x86_64.whl",
          "flashinfer_jit_cache-0.6.17cu130-cp39-abi3-manylinux_2_28_x86_64.whl",
          "sgl_deep_gemm-0.1.5.post3cu130-py3-none-manylinux2014_x86_64.whl", "triton-3.7.1-cp312-cp312-manylinux_2_28_x86_64.whl"):
    (flat / n).write_bytes(b"x")
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    z.writestr("uv-0.12.20.data/scripts/uv", b"\x7fELFfake")
(flat / "uv-0.12.20-py3-none-manylinux_2_17_x86_64.whl").write_bytes(buf.getvalue())
(flat / "uvicorn-0.54.0-py3-none-any.whl").write_bytes(b"x")  # must not be taken for uv
ok("flat_wheelhouse", sglang.wheelhouse(flat) == (flat, flat / "requirements.lock"))
_real_symlink = Path.symlink_to
Path.symlink_to = lambda self, target, *a: self.write_text(str(target), encoding="utf-8")  # Windows: no symlink rights
links = sglang.relink(flat, flat / "requirements.lock", work / "links")
got = sorted(p.name for p in links.iterdir())
ok("relink_restores_plus", {"sglang-0.5.19+gd00d88efc8d6-cp312-cp312-linux_x86_64.whl",
                            "torch-2.13.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl",
                            "flashinfer_jit_cache-0.6.17+cu130-cp39-abi3-manylinux_2_28_x86_64.whl",
                            "sgl_deep_gemm-0.1.5.post3+cu130-py3-none-manylinux2014_x86_64.whl",
                            "triton-3.7.1-cp312-cp312-manylinux_2_28_x86_64.whl"} <= set(got)
   and not any("cu130-" in g and "+" not in g for g in got), got)
if "copy" in sglang.relink.__code__.co_varnames:   # the experimentation copy stages the wheels on local disk
    copied = sglang.relink(flat, flat / "requirements.lock", work / "copied", copy=True)
    ok("relink_copy_is_local", all(not p.is_symlink() and p.read_bytes() == b"x" for p in copied.glob("torch-*.whl"))
       and (copied / "torch-2.13.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl").is_file())
ok("relink_points_at_dataset", (links / "torch-2.13.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl").read_text(
    encoding="utf-8") == str(flat / "torch-2.13.0cu130-cp312-cp312-manylinux_2_28_x86_64.whl"))
runs = []
_real_run, _real_prefix = sglang._run, sglang.PREFIX
sglang._run = lambda cmd, env=None, check=True, **kw: runs.append([str(c) for c in cmd]) or subprocess.CompletedProcess(cmd, 0, "", "")
sglang.PREFIX = work / "sgl"
try:
    py = sglang.install("banwait13/sglangwheels", lambda ref: flat)
    ok("install_steps", len(runs) == 3 and runs[0][1] == "venv" and runs[1][-1].endswith("requirements.lock")
       and "--reinstall" in runs[2] and "--no-deps" in runs[2] and runs[2][-1].endswith(".whl"), runs)
    ok("install_uses_relinked", runs[1][runs[1].index("--find-links") + 1].endswith("links")
       and "+gd00d88efc8d6" in runs[2][-1], runs)
    ok("uv_from_wheel", Path(runs[0][0]).read_bytes().startswith(b"\x7fELF"))
finally:
    sglang._run, sglang.PREFIX = _real_run, _real_prefix
    Path.symlink_to = _real_symlink
weights = work / "weights"
weights.mkdir()
(weights / "a.safetensors").write_bytes(os.urandom(3000))
(weights / "b.json").write_bytes(b"{}")
info = sglang.precache([weights], threads=2, chunk=1024)
ok("precache_reads_all", info["files"] == 2 and info["gb"] == round(3002 / 1e9, 1))

# --- 4. optional: every flag exists in the real wheel's ServerArgs ------------------------------------------------
src = os.environ.get("SGLANG_SRC")
if src:
    text = (Path(src) / "srt" / "server_args.py").read_text(encoding="utf-8")
    fields = set(re.findall(r"^    ([a-z_0-9]+): A\[", text, re.M)) | {"tensor_parallel_size"}  # tp_size alias
    for p in NV:
        unknown = [f for f in spec.profiles[p]["flags"] if f[2:].replace("-", "_") not in fields]
        ok(f"{p}_flags_in_wheel", not unknown, unknown)

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys.exit(1 if FAILS else 0)
