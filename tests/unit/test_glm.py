# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_ROOT, _os.path.join(_ROOT, "Sarbloh-Experimentation")):  # glm.py lives only there
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""GLM-5.3-Flash on llama.cpp (Sarbloh-Experimentation/harness/llm/glm.py), without a GPU: the spec is registered,
the profiles carry their window, tools and images, the effort and the thinking switch reach the server in the
template's terms, the runtime unpacks the release tarballs (library symlinks kept) and finds the first GGUF shard,
and the server is launched as llama-server with --alias."""
import tempfile
from pathlib import Path

from harness.config import build_config
from harness.llm import glm, server as srv
from harness.llm.client import LLM
from harness.llm.spec import get_spec

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


spec = glm.SPEC
CHAIN = ["glm_llamacpp_vision", "glm_llamacpp"]
# --- 1. spec and config ----------------------------------------------------------------------------------------
ok("registered", get_spec("glm") is spec and spec.name == "glm")
cfg = build_config({"model": "glm"})
ok("config_picks_it", cfg["llm"]["model"] == "glm-5.3-flash" and cfg["llm"]["temperature"] == 1.0
   and cfg["llm"]["top_p"] == 0.95 and cfg["llm"]["top_k"] == 0, cfg["llm"])
ok("chain", spec.profile_chain == CHAIN and all(p in spec.profiles for p in CHAIN))
for p in CHAIN:
    f = spec.profiles[p]["flags"]
    ok(f"{p}_window", spec.max_model_len(p) == 131072 and f["--kv-unified-per-slot"] == "131072")
    ok(f"{p}_tools", spec.has_tool_parser(p) and "--jinja" in f and f["--reasoning-format"] == "deepseek")
    ok(f"{p}_ten_slots", f["--parallel"] == "10" and "--kv-unified" in f)
    ok(f"{p}_fit", f["--fit"] == "on" and "--n-cpu-moe" not in f and "--n-gpu-layers" not in f)
    ok(f"{p}_sampling_defaults", f["--top-k"] == "0" and f["--min-p"] == "0.0")
    ok(f"{p}_runtime", spec.profiles[p]["runtime"] == "llamacpp" and callable(spec.runtimes["llamacpp"]))
ok("vision_profile", spec.has_vision("glm_llamacpp_vision") and spec.profiles["glm_llamacpp_vision"]["flags"]["--mmproj"]
   == str(glm.MMPROJ))
ok("text_profile", not spec.has_vision("glm_llamacpp") and "--no-mmproj" in spec.profiles["glm_llamacpp"]["flags"])
# other models keep reading their window and parser from the flags
qwen = get_spec("qwen_flash_next")
ok("qwen_unchanged", qwen.max_model_len("sglang_flash_vision") == 139264 and qwen.has_tool_parser("sglang_flash_vision"))

# --- 2. effort and thinking in the template's terms ------------------------------------------------------------
tk = spec.template_kwargs
ok("xhigh_is_max", tk({"reasoning_effort": "xhigh"}) == {"reasoning_effort": "max"})
ok("medium_is_high", tk({"reasoning_effort": "medium"}) == {"reasoning_effort": "high"})
ok("low_is_low", tk({"reasoning_effort": "low"}) == {"reasoning_effort": "low"})
ok("no_thinking_is_low", tk({"enable_thinking": False, "reasoning_effort": "xhigh"}) == {"reasoning_effort": "low"})
ok("thinking_on_is_default", tk({"enable_thinking": True}) == {})
ok("other_kwargs_kept", tk({"clear_thinking": False, "reasoning_effort": "xhigh"})
   == {"clear_thinking": False, "reasoning_effort": "max"})

sent = []
llm = LLM({**cfg["llm"], "base_url": "http://x/v1"}, spec=spec)
llm._post = lambda path, body, timeout: sent.append(body) or {
    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}
llm.chat([{"role": "user", "content": "hi"}], template_kwargs={"reasoning_effort": "xhigh"})
llm.chat([{"role": "user", "content": "hi"}], thinking=False)
ok("request_effort", sent[0]["chat_template_kwargs"] == {"reasoning_effort": "max"} and sent[0]["top_k"] == 0,
   sent[0])
ok("request_no_thinking", sent[1]["chat_template_kwargs"] == {"reasoning_effort": "low"}, sent[1])

# --- 3. runtime: tarballs, first shard ------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    data = tmp / "dataset" / "flat"  # Kaggle mounts the folder one level down
    data.mkdir(parents=True)
    for name, body in (("llama-server", b"#!/bin/sh\n"), ("libllama.so.0", b"lib"), ("libcudart.so.12", b"rt"),
                       ("ld-linux-x86-64.so.2", b"ld")):
        (data / name).write_bytes(body)
    dest = tmp / "rt"
    glm.stage(tmp / "dataset", dest)
    ok("stage_files", sorted(p.name for p in dest.iterdir() if p.name != ".staged")
       == sorted(["llama-server", "libllama.so.0", "libcudart.so.12", "ld-linux-x86-64.so.2"]))
    (dest / "marker").write_text("x")
    glm.stage(tmp / "dataset", dest)
    ok("stage_once", (dest / "marker").exists())
    (data / "libggml-cuda.so").write_bytes(b"new")
    glm.stage(tmp / "dataset", dest)
    ok("stage_again_on_change", not (dest / "marker").exists() and (dest / "libggml-cuda.so").exists())
    # the plain start fails (glibc 2.35), the bundled loader works: the launcher picks the loader
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        bad = cmd[0].endswith("llama-server")
        return type("R", (), {"returncode": 1 if bad else 0, "stdout": "" if bad else "version: 11878",
                              "stderr": "GLIBC_2.38 not found" if bad else ""})()

    real_run = glm.subprocess.run
    glm.subprocess.run = fake_run
    try:
        program, pre, how = glm.launcher(dest, {"LD_LIBRARY_PATH": "/usr/local/nvidia/lib64"})
    finally:
        glm.subprocess.run = real_run
    ok("launcher_falls_back", program.endswith("ld-linux-x86-64.so.2") and how.startswith("bundled glibc")
       and pre[0] == "--library-path" and pre[1].endswith("/usr/local/nvidia/lib64")
       and pre[2].endswith("llama-server") and len(calls) == 2, (program, pre, how))

    w = tmp / "weights" / "gguf"
    w.mkdir(parents=True)
    for n in ("GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf", "GLM-5.3-Flash-UD-IQ4_XS-00002-of-00005.gguf",
              "mmproj-BF16.gguf"):
        (w / n).write_bytes(b"")
    ok("first_shard", glm.first_shard(tmp / "weights").name == "GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf")

# --- 4. launch: llama-server, the model file, --alias --------------------------------------------------------
captured = {}


class FakeProc:
    pid = 0

    def poll(self):
        return None


def fake_popen(cmd, **kw):
    captured["cmd"], captured["env"] = cmd, kw.get("env")
    return FakeProc()


with tempfile.TemporaryDirectory() as tmp:
    srv._RUNTIMES["llamacpp"] = {"env": {"LD_LIBRARY_PATH": "/x"}, "python": "/rt/llama-server", "serve": ["--model"],
                                 "model_dir": "/w/m-00001-of-00005.gguf", "backend": "llamacpp", "name_flag": "--alias"}
    real = srv.subprocess.Popen
    srv.subprocess.Popen = fake_popen
    try:
        s = srv.LlmServer(spec, {"port": 8000, "startup_timeout_s": 1}, Path(tmp))
        s.launch("glm_llamacpp_vision")
    finally:
        srv.subprocess.Popen = real
        srv._RUNTIMES.pop("llamacpp", None)
    cmd = captured["cmd"]
    ok("launch_binary", cmd[:3] == ["/rt/llama-server", "--model", str(Path("/w/m-00001-of-00005.gguf"))], cmd[:3])
    ok("launch_alias", cmd[3:5] == ["--alias", "glm-5.3-flash"] and "--served-model-name" not in cmd, cmd[3:6])
    ok("launch_flags", cmd[cmd.index("--mmproj") + 1] == str(glm.MMPROJ) and "--jinja" in cmd)
    ok("launch_env", captured["env"]["LD_LIBRARY_PATH"] == "/x" and captured["env"]["HF_HUB_OFFLINE"] == "1")
    ok("backend", s.backend == "llamacpp" and s.vision and s.max_model_len == 131072)

print(f"\n{CHECKS[0] - len(FAILS)}/{CHECKS[0]} passed")
_sys.exit(1 if FAILS else 0)
