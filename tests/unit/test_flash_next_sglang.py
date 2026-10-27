# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
for _p in (_ROOT, _os.path.join(_ROOT, "Sarbloh-Experimentation")):  # qwen_flash_next.py lives only there
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""Qwen3.8-Flash-Next on SGLang with MTP (Sarbloh-Experimentation/harness/llm/qwen_flash_next.py), without a GPU:
the spec is registered, the flags match the milestone-2 reference launch, the chain falls back to the 27B NVFP4
chain, the launch command, the drafter view built from a fake drafter, and the FR-Spec hash checks."""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from harness.config import build_config
from harness.llm import qwen, qwen_flash_next as qfn, server
from harness.llm.server import LlmServer
from harness.llm.spec import get_spec

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


spec = qfn.SPEC
FL = ["sglang_flash_frspec_vision", "sglang_flash_mtp_vision", "sglang_flash_vision", "sglang_flash"]
# --- 1. spec, chain, the 27B left alone ---------------------------------------------------------------------------
ok("registered", get_spec("qwen_flash_next") is spec and spec.name == "qwen_flash_next")
ok("config_picks_it", build_config({"model": "qwen_flash_next"})["llm"]["temperature"] == 0.7)
# NVFP4 sits between Flash-Next and the 27B FP8 since 10-22; a profile whose input is not attached fails at once.
NV = ["sglang_nvfp4_mtp5_vision", "sglang_nvfp4_mtp_vision", "sglang_nvfp4_vision", "sglang_nvfp4"]
ok("chain_flash_nvfp4_then_27b_fp8", spec.profile_chain == FL + NV + ["qwen_fp8_vision", "qwen_fp8"]
   and all(p in spec.profiles for p in spec.profile_chain))
ok("chain_nvfp4_profiles_use_nvfp4", all(spec.profiles[p]["model_dataset"] == qfn.NVFP4 for p in NV))
ok("qwen_py_untouched", qwen.SPEC.name == "qwen" and not any(p in qwen.SPEC.profiles for p in FL))
ok("27b_profiles_copied", all(spec.profiles[p] == qwen.SPEC.profiles[p] for p in qwen.SPEC.profiles))
ok("runtimes", all(callable(spec.runtimes[r]) for r in ("flash_sglang", "flash_mtp", "flash_frspec", "sglang")))

# --- 2. flags: the reference's launch ------------------------------------------------------------------------------
for p in FL:
    prof, f = spec.profiles[p], spec.profiles[p]["flags"]
    ok(f"{p}_target", prof["model_dataset"] == qfn.FLASH_SGL)
    ok(f"{p}_window", spec.max_model_len(p) == 139264)
    ok(f"{p}_autoround_ple", f["--quantization"] == "auto-round" and "--ple-offload-embedding" in f
       and f["--moe-runner-backend"] == "auto")
    ok(f"{p}_memory", f["--mem-fraction-static"] == "0.96" and f["--kv-cache-dtype"] == "fp8_e4m3"
       and f["--mamba-ssm-dtype"] == "bfloat16" and f["--mamba-radix-cache-strategy"] == "extra_buffer")
    ok(f"{p}_ten_requests", f["--max-running-requests"] == "10" and f["--cuda-graph-max-bs-decode"] == "10"
       and max(map(int, f["--cuda-graph-bs-decode"])) == 10)
    ok(f"{p}_tools", spec.has_tool_parser(p) and f["--tool-call-parser"] == "qwen3_coder"
       and f["--reasoning-parser"] == "qwen3")
    if "--speculative-algorithm" in f:
        ok(f"{p}_mtp", f["--speculative-algorithm"] == "NEXTN" and f["--speculative-eagle-topk"] == "1"
           and int(f["--speculative-num-draft-tokens"]) == int(f["--speculative-num-steps"]) + 1 == 4
           and f["--speculative-draft-model-path"] == str(qfn.FLASH_DRAFT_VIEW)
           and f["--speculative-draft-model-quantization"] == "compressed-tensors" and f["--gdn-mtp-cache-mode"] == "none")
ok("frspec_only_first", spec.profiles[FL[0]]["flags"]["--speculative-token-map"] == str(qfn.FLASH_TOKEN_MAP)
   and not any("--speculative-token-map" in spec.profiles[p]["flags"] for p in FL[1:]))
ok("no_spec_last_two", not any("--speculative-algorithm" in spec.profiles[p]["flags"] for p in FL[2:]))
ok("runtime_names", [spec.profiles[p]["runtime"] for p in FL] == ["flash_frspec", "flash_mtp", "flash_sglang", "flash_sglang"])
ok("vision", [spec.has_vision(p) for p in FL] == [True, True, True, False])

# --- 3. launch command ---------------------------------------------------------------------------------------------
work = Path(tempfile.mkdtemp(prefix="qfn_"))
model_dir = work / "target"
model_dir.mkdir()
(model_dir / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "auto-round", "bits": 4}}),
                                       encoding="utf-8")
(model_dir / "tokenizer.json").write_text('{"tok": 1}', encoding="utf-8")
srv = LlmServer(spec, {"port": 8125, "profile_chain": spec.profile_chain, "startup_timeout_s": 1, "max_inflight": 8}, work)
server._RUNTIMES["flash_frspec"] = {"env": {"PATH": "/x"}, "python": "/venv/bin/python",
                                    "serve": ["-m", "sglang.launch_server", "--model-path"], "backend": "sglang", "info": {}}
captured = {}


class FakeProc:
    pid = 1

    def poll(self):
        return None


_real_find, _real_popen = server.find_kaggle_input, subprocess.Popen
server.find_kaggle_input = lambda ref: model_dir
server.subprocess.Popen = lambda cmd, **kw: captured.update(cmd=cmd) or FakeProc()
try:
    srv.launch(FL[0])
finally:
    server.find_kaggle_input, server.subprocess.Popen = _real_find, _real_popen
    server._RUNTIMES.pop("flash_frspec", None)
cmd = captured["cmd"]
i = cmd.index("--cuda-graph-bs-decode")
ok("graph_list_expanded", cmd[i + 1:i + 11] == [str(b) for b in range(1, 11)] and cmd[i + 11].startswith("--"))
ok("bool_flags_alone", all(cmd[cmd.index(b) + 1].startswith("--") for b in ("--ple-offload-embedding",
                                                                             "--weight-loader-drop-cache-after-load")))
ok("template_from_target", cmd[cmd.index("--chat-template") + 1] == f"{model_dir}/chat_template.jinja")

# --- 4. drafter view from a fake drafter ---------------------------------------------------------------------------
draft = work / "drafter"
draft.mkdir()
group = {"targets": ["RoutedExperts"], "weights": {"num_bits": 4, "group_size": 32, "symmetric": True}}
(draft / "config.json").write_text(json.dumps({"quantization_config": {
    "quant_method": "compressed-tensors", "config_groups": {"mtp_routed_experts": group}, "ignore": []}}), encoding="utf-8")
(draft / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
    "mtp.layers.0.mlp.experts.0.gate_proj.weight": "mtp-routed-experts-int4.safetensors",
    "mtp.fc.weight": "mtp-dense.safetensors"}}), encoding="utf-8")
for n in ("mtp-routed-experts-int4.safetensors", "mtp-dense.safetensors"):
    (draft / n).write_bytes(b"x" * 16)
_real_symlink = Path.symlink_to
Path.symlink_to = lambda self, target, *a: self.write_text(str(target), encoding="utf-8")  # Windows: no symlink rights
try:
    view = qfn.prepare_draft_view(model_dir, draft, work / "view")
    cfg = json.loads((view / "config.json").read_text(encoding="utf-8"))["quantization_config"]
    ok("view_rules", cfg["config_groups"]["mtp_routed_experts"]["targets"] == [qfn.EXPERT_TARGET]
       and cfg["ignore"] == [qfn.DENSE_IGNORE])
    ok("view_links", {p.name for p in view.iterdir()} >= {"mtp-dense.safetensors", "mtp-routed-experts-int4.safetensors",
                                                          "model.safetensors.index.json", "tokenizer.json"}
       and (view / "tokenizer.json").read_text(encoding="utf-8") == str((model_dir / "tokenizer.json").resolve()))
    ok("view_rebuilds", qfn.prepare_draft_view(model_dir, draft, work / "view") == view)
    bad = json.loads((draft / "config.json").read_text(encoding="utf-8"))
    bad["quantization_config"]["config_groups"]["mtp_routed_experts"]["weights"]["group_size"] = 128
    (draft / "config.json").write_text(json.dumps(bad), encoding="utf-8")
    try:
        qfn.prepare_draft_view(model_dir, draft, work / "view")
        ok("view_rejects_g128", False)
    except RuntimeError:
        ok("view_rejects_g128", True)
finally:
    Path.symlink_to = _real_symlink

# --- 5. runtimes: target check and FR-Spec hashes (the wheel install faked) ----------------------------------------
wheels = work / "wheels"
wheels.mkdir()
(wheels / "hot_tokens_64k.pt").write_bytes(b"map")
inputs = {qfn.FLASH_SGL: model_dir, qfn.FLASH_DRAFT: draft, qfn.SGLANG_WHEELS: wheels}


def find(ref):  # find_kaggle_input over ``inputs``
    if ref not in inputs:
        raise FileNotFoundError(ref)
    return inputs[ref]


_real_wheels, _real_view = qfn._sglang_wheels, qfn.prepare_draft_view
qfn._sglang_wheels = lambda wd, fi: {"env": {}, "python": "py", "serve": [], "backend": "sglang", "info": {}}
qfn.prepare_draft_view = lambda target, src, view: view
qfn._PRECACHED.update(inputs)  # no background reads in the test
try:
    rt = qfn.flash_sglang_runtime(work, find)
    ok("runtime_target", rt["info"]["target"] == str(model_dir))
    ok("no_adapter_base_model", "model_dir" not in rt and "lora_view" not in rt["info"])
    try:
        qfn.flash_frspec_runtime(work, find)
        ok("frspec_rejects_unpinned_map", False)
    except RuntimeError as exc:
        ok("frspec_rejects_unpinned_map", "not the pinned one" in str(exc), exc)
    qfn.TOKEN_MAP_SHA = hashlib.sha256(b"map").hexdigest()
    try:
        qfn.flash_frspec_runtime(work, find)
        ok("frspec_rejects_other_tokenizer", False)
    except RuntimeError as exc:
        ok("frspec_rejects_other_tokenizer", "tokenizer" in str(exc), exc)
    (model_dir / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "modelopt"}}),
                                           encoding="utf-8")
    try:
        qfn.flash_sglang_runtime(work, find)
        ok("runtime_rejects_other_quant", False)
    except RuntimeError:
        ok("runtime_rejects_other_quant", True)
finally:
    qfn._sglang_wheels, qfn.prepare_draft_view = _real_wheels, _real_view

# --- 5b. the LoRA view: the adapter's merged projections in one extra file that the index points at ---------------
import struct


def write_st(path, tensors):  # a safetensors file from {name: (dtype, shape, raw bytes)}
    header, data = {}, b""
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


L = "model.language_model.layers."
Q, IN, Z, A = (L + "3.self_attn.q_proj.weight", L + "0.linear_attn.in_proj_qkv.weight",
               L + "0.linear_attn.in_proj_z.weight", L + "0.linear_attn.in_proj_a.weight")
EXP, MTPQ = L + "0.mlp.experts.0.gate_proj.qweight", "mtp.layers.0.self_attn.q_proj.weight"
S1, S2 = "model-00001-of-00017.safetensors", "model-00017-of-00017.safetensors"
base = work / "intel"
base.mkdir()
(base / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "auto-round", "bits": 4}}),
                                  encoding="utf-8")
write_st(base / S1, {IN: ("BF16", [2, 2], b"a" * 8), Z: ("BF16", [1, 2], b"b" * 4), A: ("BF16", [1, 2], b"c" * 4)})
write_st(base / S2, {Q: ("BF16", [2, 1], b"d" * 4), EXP: ("I32", [1], b"e" * 4), MTPQ: ("BF16", [1, 1], b"f" * 2)})
base_map = {IN: S1, Z: S1, A: S1, Q: S2, EXP: S2, MTPQ: S2}
(base / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": base_map}),
                                                   encoding="utf-8")
adapter = work / "lora"
adapter.mkdir()
ADAPTER_1 = {IN: ("BF16", [2, 2], b"A" * 8), L + "0.mlp.experts.0.gate_proj.weight": ("U8", [2], b"xx")}
ADAPTER_2 = {Z: ("BF16", [1, 2], b"B" * 4), Q: ("BF16", [2, 1], b"D" * 4), MTPQ: ("BF16", [1, 1], b"F" * 2)}
write_st(adapter / "model-bf16-00001.safetensors", ADAPTER_1)
write_st(adapter / "model-bf16-00010.safetensors", ADAPTER_2)
(adapter / "merge_report.json").write_text(json.dumps({"merged": 3}), encoding="utf-8")
lview = work / "lora-view"


def rejects(name, needle):
    try:
        qfn.prepare_lora_view(base, adapter, lview)
        ok(name, False, "not rejected")
    except RuntimeError as exc:
        ok(name, needle in str(exc), exc)


Path.symlink_to = lambda self, target, *a, **k: self.write_text(str(target), encoding="utf-8")
try:
    v = qfn.prepare_lora_view(base, adapter, lview)
    merged = read_st(v / qfn.LORA_FILE)
    ok("lora_targets_only", sorted(merged) == sorted([IN, Z, Q]), sorted(merged))
    ok("lora_adapter_bytes", merged[IN] == ("BF16", [2, 2], b"A" * 8) and merged[Z][2] == b"B" * 4
       and merged[Q][2] == b"D" * 4)
    vmap = json.loads((v / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    ok("lora_index", all(vmap[k] == qfn.LORA_FILE for k in (IN, Z, Q))
       and all(vmap[k] == base_map[k] for k in (A, EXP, MTPQ)), vmap)
    ok("lora_file_loads_last", sorted([S1, S2, qfn.LORA_FILE])[-1] == qfn.LORA_FILE)
    ok("lora_links_rest", {p.name for p in v.iterdir()} == {"config.json", S1, S2, qfn.LORA_FILE,
                                                            "model.safetensors.index.json"}
       and (v / S2).read_text(encoding="utf-8") == str((base / S2).resolve()))
    ok("lora_rebuilds", qfn.prepare_lora_view(base, adapter, lview) == lview)
    (adapter / "merge_report.json").write_text(json.dumps({"merged": 4}), encoding="utf-8")
    rejects("lora_rejects_report_mismatch", "merged a different number")
    (adapter / "merge_report.json").write_text(json.dumps({"merged": 3}), encoding="utf-8")
    write_st(adapter / "model-bf16-00010.safetensors", {**ADAPTER_2, Q: ("BF16", [1, 2], b"D" * 4)})
    rejects("lora_rejects_shape", "vs target")
    write_st(adapter / "model-bf16-00010.safetensors", {Z: ADAPTER_2[Z]})
    rejects("lora_rejects_missing", "not in")
    write_st(adapter / "model-bf16-00010.safetensors", ADAPTER_2)

    # the runtime serves the view when the adapter is attached
    inputs.update({qfn.FLASH_SGL: base, qfn.LORA: adapter})
    _real_lview, qfn.FLASH_LORA_VIEW = qfn.FLASH_LORA_VIEW, lview
    qfn._sglang_wheels = lambda wd, fi: {"env": {}, "python": "py", "serve": [], "backend": "sglang", "info": {}}
    try:
        rt = qfn.flash_sglang_runtime(work, find)
        ok("runtime_serves_lora_view", rt.get("model_dir") == str(lview) and rt["info"]["target"] == str(base)
           and rt["info"]["lora_view"] == str(lview), rt)
    finally:
        qfn.FLASH_LORA_VIEW, qfn._sglang_wheels = _real_lview, _real_wheels
finally:
    Path.symlink_to = _real_symlink

# the launch takes the runtime's model_dir over the profile's dataset
server._RUNTIMES["flash_sglang"] = {"env": {}, "python": "py", "serve": ["-m", "sglang.launch_server", "--model-path"],
                                    "backend": "sglang", "model_dir": str(lview), "info": {}}
captured.clear()
server.find_kaggle_input = lambda ref: model_dir
server.subprocess.Popen = lambda cmd, **kw: captured.update(cmd=cmd) or FakeProc()
try:
    srv.launch("sglang_flash")
finally:
    server.find_kaggle_input, server.subprocess.Popen = _real_find, _real_popen
    server._RUNTIMES.pop("flash_sglang", None)
cmd = captured["cmd"]
ok("launch_uses_runtime_model_dir", cmd[cmd.index("--model-path") + 1] == str(lview)
   and cmd[cmd.index("--chat-template") + 1] == f"{lview}/chat_template.jinja", cmd[:6])

# --- 6. start wait: ready when healthy, false at the timeout or when the process exits ---------------------------
ok("config_wait_knobs", build_config({})["server"]["startup_timeout_s"] == 5400.0)


class FakeClock:
    """server.time with a fake monotonic clock: each sleep moves it on."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, s):
        self.now += s

    def __getattr__(self, name):
        return getattr(_real_time, name)


_real_time = server.time
wsrv = LlmServer(spec, {"port": 8126, "startup_timeout_s": 1, "max_inflight": 8}, work)
wsrv.process, wsrv.healthy = FakeProc(), lambda timeout=5: False

try:
    server.time = FakeClock()
    ok("slow_start_waits_to_timeout", wsrv.wait_ready(5400) is False and server.time.now >= 5400, server.time.now)
    calls = [0]
    wsrv.healthy = lambda timeout=5: (calls.__setitem__(0, calls[0] + 1) or calls[0] >= 3)
    server.time = FakeClock()
    ok("ready_returns_true", wsrv.wait_ready(5400) is True and server.time.now < 60, server.time.now)

    class DeadProc(FakeProc):
        def poll(self):
            return 1

    wsrv.process = DeadProc()
    server.time = FakeClock()
    ok("exit_returns_false_at_once", wsrv.wait_ready(5400) is False and server.time.now == 0, server.time.now)
finally:
    server.time = _real_time

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys.exit(1 if FAILS else 0)
