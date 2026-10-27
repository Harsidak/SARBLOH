# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")   # these changes are in the experimentation copy only
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""The model fallback and the patched weights folder (Sarbloh-Experimentation):
- ``run._start_server`` starts the configured model; when its whole chain fails it rebuilds the config for the next
  entry of ``model_fallback`` (that model's served name and sampling, the given chain) and starts that; the last
  failure is raised; a rerun keeps tracing off after the rebuild;
- the Nemotron chain is the three SGLang 0.5.20 profiles (MTP5 + image, image, text only) on the native config (no
  config_overrides), every profile sets the flashinfer_cutlass MoE backend and holds enough Mamba cache for its running
  requests; the 0.5.19 Pennyroyal profiles keep the Nano Omni overrides;
- ``server.patched_model_dir`` links every file but config.json, which gets the overrides merged in.

Run: uv run python tests/unit/test_model_fallback.py
"""
import json
import tempfile
from pathlib import Path

from harness import run
from harness.config import build_config
from harness.llm import nemotron, server as S
from harness.llm.spec import get_spec

FAILS, WARNS, CHECKS = [], [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


class FakeServer:
    started: list = []
    broken: set = set()

    def __init__(self, spec, cfg, working_dir, deadline=None):
        self.spec, self.cfg, self.stopped = spec, cfg, False

    def start(self):
        FakeServer.started.append((self.spec.name, self.cfg.get("profile_chain")))
        if self.spec.name in FakeServer.broken:
            raise RuntimeError(f"the model server failed on every profile: {self.spec.name}")

    def stop(self):
        self.stopped = True


real = S.LlmServer
S.LlmServer = FakeServer
try:
    over = {"model": "nemotron", "tracing": True,
            "model_fallback": [{"model": "qwen_flash_next", "profile_chain": ["sglang_flash_vision", "sglang_flash"]}]}
    FakeServer.started, FakeServer.broken = [], {"nemotron"}
    cfg, spec, srv = run._start_server(build_config(over), over, Path(tempfile.mkdtemp()), 0.0, rerun=True)
    ok("falls_back", spec.name == "qwen_flash_next" and cfg["model"] == "qwen_flash_next", spec.name)
    ok("tried_in_order", FakeServer.started == [("nemotron", None),
                                                ("qwen_flash_next", ["sglang_flash_vision", "sglang_flash"])],
       str(FakeServer.started))
    ok("fallback_sampling", cfg["llm"]["model"] == get_spec("qwen_flash_next").served_model_name, cfg["llm"]["model"])
    ok("rerun_tracing_off", cfg["tracing"] is False)
    FakeServer.started, FakeServer.broken = [], set()
    cfg, spec, srv = run._start_server(build_config(over), over, Path(tempfile.mkdtemp()), 0.0, rerun=False)
    ok("first_model_kept", spec.name == "nemotron" and FakeServer.started == [("nemotron", None)]
       and cfg["llm"]["model"] == "nemotron-3.5-super-vl", str(FakeServer.started))
    FakeServer.started, FakeServer.broken = [], {"nemotron", "qwen_flash_next"}
    try:
        run._start_server(build_config(over), over, Path(tempfile.mkdtemp()), 0.0, rerun=False)
        ok("last_failure_raised", False, "no error")
    except RuntimeError as exc:
        ok("last_failure_raised", "qwen_flash_next" in str(exc) and len(FakeServer.started) == 2, str(exc))
finally:
    S.LlmServer = real

spec = get_spec("nemotron")
ok("nemotron_chain", spec.profile_chain == ["nemotron_mtp5_vision", "nemotron_vision", "nemotron"],
   str(spec.profile_chain))
ok("chain_native_config", all("config_overrides" not in spec.profiles[p] for p in spec.profile_chain))
ok("chain_on_0520", all(spec.profiles[p]["runtime"] == "sglang_0520" for p in spec.profile_chain)
   and nemotron.WHEELS_0520 == "banwait13/sglang0520wheels")
ok("pennyroyal_patched", all(v.get("config_overrides") == nemotron.NANO_OMNI
                             for k, v in spec.profiles.items() if v["runtime"] == "sglang_pennyroyal"))
ok("moe_backend_cutlass", all(v["flags"].get("--moe-runner-backend") == "flashinfer_cutlass"
                              for v in spec.profiles.values()))
MAMBA_SLOTS_PER_REQUEST = 5   # extra_buffer, measured on the 0.5.19 start: a cache of 40 capped running requests at 8
ok("mamba_cache_fits_requests", all(int(v["flags"]["--max-mamba-cache-size"]) // MAMBA_SLOTS_PER_REQUEST
                                    >= int(v["flags"]["--max-running-requests"]) for v in spec.profiles.values()))
ok("chain_keeps_card_rounding", all("--enable-mamba-cache-stochastic-rounding" in spec.profiles[p]["flags"]
                                    for p in spec.profile_chain))
MAMBA_CHUNK = 128   # the checkpoint's mamba_chunk_size; SGLang asserts track interval >= it
ok("mamba_interval_ok", all(int(spec.profiles[p]["flags"]["--mamba-track-interval"]) %
                            int(spec.profiles[p]["flags"]["--page-size"]) == 0
                            and int(spec.profiles[p]["flags"]["--mamba-track-interval"]) >= MAMBA_CHUNK
                            for p in spec.profiles))

src, dest = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp()) / "patched"
(src / "config.json").write_text(json.dumps({"architectures": ["X"], "model_type": "x", "keep": 1}), encoding="utf-8")
(src / "model-00001.safetensors").write_bytes(b"w")
(src / "sub").mkdir()
try:
    out = S.patched_model_dir(src, {"architectures": ["Y"], "model_type": "y"}, dest)
    got = json.loads((out / "config.json").read_text(encoding="utf-8"))
    ok("config_patched", got == {"architectures": ["Y"], "model_type": "y", "keep": 1}, str(got))
    ok("weights_linked", (out / "model-00001.safetensors").is_symlink() and (out / "sub").is_symlink()
       and (out / "model-00001.safetensors").read_bytes() == b"w")
    ok("source_untouched", json.loads((src / "config.json").read_text(encoding="utf-8"))["model_type"] == "x")
except OSError as exc:   # Windows without symlink rights; Kaggle is Linux
    WARNS.append(f"symlinks unavailable here: {exc!r}")

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
print(f"WARNINGS  : {len(WARNS)} -> {WARNS}")
raise SystemExit(1 if FAILS else 0)
