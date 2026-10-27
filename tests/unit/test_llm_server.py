# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh")   # PRIME_DIR=Sarbloh-Experimentation tests the copy
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""E007 serving layer without a GPU: model specs and the config switch, the Gemma tool-argument repair, the
ServerGate (ready, ramp, fail), the client against a fake OpenAI server that drops connections while "restarting",
the watchdog's freeze detector and restart policy, and model-name hygiene of the shared files.
"""
import json
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from harness.config import build_config
from harness.llm import gemma, qwen
from harness.llm.client import LLM, ServerDown, ServerGate
from harness.llm.spec import get_spec
from harness.llm.server import LlmServer, parse_metrics

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# --- 1. specs and config -------------------------------------------------------------------------------------
g, q = get_spec("gemma"), get_spec("qwen")
ok("spec_names", g.name == "gemma" and q.name == "qwen")
try:
    get_spec("llama")
    ok("spec_unknown_rejected", False)
except ValueError:
    ok("spec_unknown_rejected", True)
for spec in (g, q):
    ok(f"{spec.name}_chain_in_profiles", all(p in spec.profiles for p in spec.profile_chain))
    ok(f"{spec.name}_profiles_have_len_and_weights",
       all(spec.max_model_len(n) > 0 and p["model_dataset"] for n, p in spec.profiles.items()))  # E112: SGLang uses --context-length
ok("no_cross_parsers", not any("qwen" in json.dumps(p["flags"]) for p in g.profiles.values())
   and not any("gemma" in json.dumps(p["flags"]) for p in q.profiles.values()))
cq, cg = build_config({"model": "qwen"}), build_config({"model": "gemma"})
# Sarbloh selects the E008 toolset by config; Sarbloh-Experimentation (E118) has no toolset setting, only those tools.
ok("config_default_is_qwen_e008", build_config()["model"] == "qwen"
   and build_config()["agent"].get("toolset", "e008") == "e008")
ok("config_qwen_sampling", cq["llm"]["model"] == q.served_model_name and cq["llm"]["temperature"] == 1.0
   and cq["llm"]["top_k"] == 20 and cq["llm"]["chat_template_kwargs"]["enable_thinking"] is True)
ok("config_gemma_default", cg["model"] == "gemma" and cg["llm"]["model"] == g.served_model_name
   and cg["llm"]["top_k"] == 64)
ok("config_override_wins", build_config({"model": "qwen", "llm": {"temperature": 0.2}})["llm"]["temperature"] == 0.2)

# --- 2. tool-argument repair ---------------------------------------------------------------------------------
ok("unquote_basic", gemma.unquote({"phase": '"explore"', "n": 3}) == {"phase": "explore", "n": 3})
ok("unquote_nested", gemma.unquote({"a": ['"1"', "2"], "b": {"c": '"x"'}}) == {"a": ["1", "2"], "b": {"c": "x"}})
ok("unquote_keeps_inner_quotes", gemma.unquote({"t": 'say "hi" now'}) == {"t": 'say "hi" now'}
   and gemma.unquote({"t": '"a" and "b"'}) == {"t": '"a" and "b"'})
calls = [{"id": "1", "type": "function", "function": {"name": "plan", "arguments": '{"phase": "\\"explore\\""}'}},
         {"id": "2", "type": "function", "function": {"name": "plan", "arguments": "{broken"}}]
fixed = g.fix_tool_calls(calls)
ok("fix_tool_calls_gemma", json.loads(fixed[0]["function"]["arguments"]) == {"phase": "explore"}
   and fixed[1]["function"]["arguments"] == "{broken")
ok("fix_tool_calls_qwen_noop", q.fix_tool_calls(calls) == calls)

# --- 3. gate -------------------------------------------------------------------------------------------------
gate = ServerGate(max_inflight=6, ramp_start=2, ramp_step=2, ramp_every_s=0.3)
ok("gate_initially_ready", gate.ready and gate.capacity() == 6)
gate.down()
t = time.monotonic()
try:
    with gate.slot(time.monotonic() + 0.3):
        pass
    ok("gate_down_blocks", False)
except TimeoutError:
    ok("gate_down_blocks", 0.25 <= time.monotonic() - t < 2)
gate.up(ramp=True)
ok("gate_ramp_starts_small", gate.capacity() == 2)
time.sleep(0.65)
ok("gate_ramp_grows", gate.capacity() == 6, str(gate.capacity()))
gate.fail("gone")
try:
    with gate.slot(time.monotonic() + 5):
        pass
    ok("gate_fail_raises_now", False)
except ServerDown:
    ok("gate_fail_raises_now", True)
ok("gate_wait_ready_false_when_dead", gate.wait_ready(0.1) is False)


# --- 4. client vs a fake server that "restarts" ---------------------------------------------------------------
class Fake:
    mode = "ok"  # ok | drop | 500
    hits = 0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        Fake.hits += 1
        Fake.last = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if Fake.mode == "drop":  # a killed server: the socket closes with no response
            self.close_connection = True
            self.connection.shutdown(2)
            return
        if Fake.mode == "500":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        body = json.dumps({"choices": [{"finish_reason": "tool_calls", "message": {
            "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "plan", "arguments": json.dumps({"phase": '"explore"'})}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{httpd.server_address[1]}/v1"

gate = ServerGate(max_inflight=4, ramp_every_s=0.2)
llm = LLM({"base_url": base, "model": "m", "retries": 2}, spec=g, gate=gate)
r = llm.chat([{"role": "user", "content": "hi"}], timeout_s=10)
ok("client_ok_and_repaired", json.loads(r.tool_calls[0]["function"]["arguments"]) == {"phase": "explore"})

# The watchdog takes the gate down and kills the server (connections drop), then brings it back 1.5 s later.
Fake.mode, Fake.hits = "drop", 0
gate.down()


def restart():
    time.sleep(1.5)
    Fake.mode = "ok"
    gate.up(ramp=True)


threading.Thread(target=restart, daemon=True).start()
t = time.monotonic()
r = llm.chat([{"role": "user", "content": "hi"}], timeout_s=20)
ok("client_waits_for_restart", r.finish_reason == "tool_calls" and 1.3 <= time.monotonic() - t < 8
   and llm.usage.failures == 0, f"{time.monotonic() - t:.1f}s failures={llm.usage.failures}")

# A request already in flight when the server dies: fails on the dropped socket, waits, succeeds, spends no retry.
Fake.mode = "drop"


def die_then_restart():
    time.sleep(0.3)
    gate.down()
    time.sleep(1.0)
    Fake.mode = "ok"
    gate.up(ramp=True)


gate2_failures = llm.usage.failures
threading.Thread(target=die_then_restart, daemon=True).start()
time.sleep(0.05)
try:
    r = llm.chat([{"role": "user", "content": "hi"}], timeout_s=20)
    ok("client_inflight_survives_restart", r.finish_reason == "tool_calls")
except Exception as exc:  # the first attempt may hit "drop" before the gate goes down: that one costs a retry
    ok("client_inflight_survives_restart", False, repr(exc))
ok("client_restart_costs_at_most_one_retry", llm.usage.failures - gate2_failures <= 1)

# The whole call is bounded by timeout_s (E006: 900 s x 4 retries).
Fake.mode = "500"
llm3 = LLM({"base_url": base, "model": "m", "retries": 10}, gate=ServerGate())
t = time.monotonic()
try:
    llm3.chat([{"role": "user", "content": "hi"}], timeout_s=3)
    ok("client_total_budget", False)
except RuntimeError:
    ok("client_total_budget", time.monotonic() - t < 5, f"{time.monotonic() - t:.1f}s")
Fake.mode = "ok"
llm4 = LLM({"base_url": base, "model": "m"}, gate=ServerGate())
llm4.gate.fail("gave up")
t = time.monotonic()
try:
    llm4.chat([{"role": "user", "content": "hi"}], timeout_s=30)
    ok("client_dead_server_fails_fast", False)
except RuntimeError:
    ok("client_dead_server_fails_fast", time.monotonic() - t < 1)
httpd.shutdown()

# --- 5. watchdog: metrics parsing, freeze detection, restart policy -------------------------------------------
text = """# HELP vllm:num_requests_running x
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:prompt_tokens_total{engine="0",model_name="m"} 1200.0
vllm:generation_tokens_total{engine="0",model_name="m"} 300.0
vllm:generation_tokens_created{engine="0",model_name="m"} 1.7e9
vllm:prefix_cache_hits_total{engine="0"} 50
vllm:prefix_cache_hits_total{engine="1"} 25
"""
m = parse_metrics(text)
ok("parse_metrics", m["vllm:num_requests_running"] == 3 and m["vllm:prefix_cache_hits_total"] == 75
   and "vllm:generation_tokens_created" not in m)

tmp = Path(tempfile.mkdtemp(prefix="e007_"))
# A chain naming another model's profiles fails at construction, not as one KeyError per profile at launch (the
# 2026-10-01 Kaggle run: a stale E006 notebook forced the gemma chain on model "qwen").
try:
    LlmServer(q, {"port": 1, "site_dir": str(tmp / "bad"), "profile_chain": ["gemma_fast", "qwen_fp8"]}, tmp)
    ok("chain_unknown_profile_fails_fast", False)
except ValueError as exc:
    ok("chain_unknown_profile_fails_fast", "gemma_fast" in str(exc) and "qwen_fp8_vision" in str(exc), str(exc))
srv = LlmServer(q, {"port": 1, "site_dir": str(tmp / "site"),
                     "watchdog_cfg": {"freeze_after_s": 0.3, "failures_to_restart": 2}}, tmp, deadline=None)


class Proc:
    rc = None

    def poll(self):
        return self.rc


srv.chain = ["qwen_flash_pc_64k", "qwen_flash_pc", "qwen_flash"]  # a 3-deep chain for the fallback checks
srv.process, srv.profile = Proc(), "qwen_flash_pc_64k"
feed = {"m": {"vllm:num_requests_running": 2, "vllm:prompt_tokens_total": 100, "vllm:generation_tokens_total": 5}}
srv.metrics = lambda timeout=10: feed["m"]
state = {}
ok("check_first_tick_fine", srv.check(state) is None)
ok("check_stall_below_threshold", srv.check(state) is None)
time.sleep(0.35)
ok("check_frozen", srv.check(state) == "frozen")
feed["m"] = {**feed["m"], "vllm:generation_tokens_total": 6}
ok("check_progress_resets", srv.check(state) is None)
feed["m"] = {**feed["m"], "vllm:num_requests_running": 0}
time.sleep(0.35)
ok("check_idle_is_not_frozen", srv.check(state) is None)
feed["m"] = None
ok("check_one_miss_tolerated", srv.check(state) is None)
ok("check_unresponsive", srv.check(state) == "unresponsive")
srv.process.rc = -9
ok("check_exited", srv.check(state) == "exited")

srv.tool_mode = "native"
ok("fallback_same_profile_first", srv._fallbacks()[0] == "qwen_flash_pc_64k")
srv.freezes["qwen_flash_pc_64k"] = 2
fb = srv._fallbacks()
ok("fallback_moves_down_chain", fb == ["qwen_flash_pc", "qwen_flash"], str(fb))
srv.deadline = time.time() + 300
srv.startup_s = 240
ok("budget_blocks_late_restart", srv._budget_allows_restart()[0] is False)
srv.deadline = time.time() + 3600
ok("budget_allows_early_restart", srv._budget_allows_restart()[0] is True)
srv.restarts = 8
ok("budget_caps_restarts", srv._budget_allows_restart()[0] is False)
srv.event("unit", x=1)
ok("events_jsonl", json.loads(srv.events_path.read_text(encoding="utf-8").splitlines()[-1])["event"] == "unit")

# --- 5b. prebuilt runtime profiles (ModelSpec.runtimes) ------------------------------------------------------
import harness.llm.server as server_mod
from harness.llm.spec import ModelSpec

calls = []


def fake_runtime(working_dir, find_input):
    calls.append(working_dir)
    return {"env": {"PATH": "/rt/bin", "PYTHONPATH": "/rt/site"}, "serve": ["-m", "fake.cli", "serve"], "info": {}}


rspec = ModelSpec(name="fake", served_model_name="fake-model", profile_chain=["rt", "wh"], llm={},
                  profiles={"rt": {"model_dataset": "o/m", "runtime": "fake_rt", "env": {"X": "1"},
                                   "flags": {"--max-model-len": "8", "--chat-template": "{model_dir}/t.jinja"}},
                            "wh": {"model_dataset": "o/m", "flags": {"--max-model-len": "8"}}},
                  runtimes={"fake_rt": fake_runtime})
rsrv = LlmServer(rspec, {"port": 1, "site_dir": str(tmp / "site2")}, tmp, deadline=None)
rsrv.install = lambda: calls.append("install")
renv = rsrv.env("rt")
ok("runtime_env_owns_paths", renv["PYTHONPATH"] == "/rt/site" and renv["X"] == "1"
   and renv["PYTHONFAULTHANDLER"] == "1" and "site2" not in json.dumps(renv))
rsrv.prepare("rt")
rsrv.prepare("rt")
LlmServer(rspec, {"port": 1, "site_dir": str(tmp / "site2")}, tmp, deadline=None).prepare("rt")
ok("runtime_prepared_once", calls.count(tmp) == 1 and "install" not in calls, str(calls))
rsrv.prepare("wh")
ok("wheelhouse_profile_installs", calls[-1] == "install")
launched = {}


class FakePopen:
    def __init__(self, cmd, env=None, **kw):
        launched["cmd"], launched["env"] = cmd, env
        self.pid = 0


real_popen, real_find = server_mod.subprocess.Popen, server_mod.find_kaggle_input
server_mod.subprocess.Popen, server_mod.find_kaggle_input = FakePopen, lambda ref: tmp / "weights"
(tmp / "weights").mkdir(exist_ok=True)
(tmp / "weights" / "config.json").write_text("{}", encoding="utf-8")
try:
    rsrv.launch("rt")
    cmd = launched["cmd"]
    ok("runtime_launch_serve_prefix", cmd[1:5] == ["-m", "fake.cli", "serve", str(tmp / "weights")], str(cmd[:5]))
    ok("model_dir_substituted", cmd[cmd.index("--chat-template") + 1] == f"{tmp / 'weights'}/t.jinja")
    rsrv.launch("wh")
    ok("wheelhouse_launch_default", launched["cmd"][1:5] == ["-m", "vllm.entrypoints.openai.api_server", "--model",
                                                              str(tmp / "weights")])
finally:
    server_mod.subprocess.Popen, server_mod.find_kaggle_input = real_popen, real_find
    rsrv.process = None
broken = ModelSpec(name="b", served_model_name="b", profile_chain=["rt"], llm={},
                   profiles={"rt": {"model_dataset": "o/m", "runtime": "boom", "flags": {"--max-model-len": "8"}}},
                   runtimes={"boom": lambda w, f: (_ for _ in ()).throw(FileNotFoundError("not attached"))})
bsrv = LlmServer(broken, {"port": 1, "site_dir": str(tmp / "site3"), "startup_timeout_s": 1}, tmp, deadline=None)
ok("runtime_failure_fails_profile_only", bsrv._start_profile("rt") is False
   and "not attached" in bsrv.attempts[-1]["error"])
# E008: Qwen3.8-27B-FP8 with one image per prompt, then the same profile text-only.
ok("qwen_chain_is_fp8_vision_then_text", q.profile_chain == ["qwen_fp8_vision", "qwen_fp8"]
   and all(q.profiles[p]["model_dataset"] == qwen.FP8 and not q.profiles[p].get("runtime") for p in q.profile_chain))
vf, tf = q.profiles["qwen_fp8_vision"]["flags"], q.profiles["qwen_fp8"]["flags"]
ok("qwen_vision_flag_only_difference", json.loads(vf["--limit-mm-per-prompt"]) == {"image": 1, "video": 0}
   and {k: v for k, v in vf.items() if k != "--limit-mm-per-prompt"}
   == {k: v for k, v in tf.items() if k != "--limit-mm-per-prompt"} and "--enable-prefix-caching" in vf)
ok("has_vision", q.has_vision("qwen_fp8_vision") and not q.has_vision("qwen_fp8")
   and not any(g.has_vision(p) for p in g.profiles))
vsrv = LlmServer(q, {"port": 1, "site_dir": str(tmp / "site4")}, tmp, deadline=None)
vsrv.profile, vsrv.tool_mode = "qwen_fp8_vision", "native"
vsrv.freezes["qwen_fp8_vision"] = 5
ok("fallback_keeps_image_support", vsrv._fallbacks() == ["qwen_fp8_vision"] and vsrv.vision, str(vsrv._fallbacks()))
httpd2 = ThreadingHTTPServer(("127.0.0.1", 0), Handler)   # the first fake server is shut down above
threading.Thread(target=httpd2.serve_forever, daemon=True).start()
vsrv.base_url, Fake.mode = f"http://127.0.0.1:{httpd2.server_address[1]}/v1", "ok"
vsrv.image_smoke()
httpd2.shutdown()
sent = json.loads(Fake.last)
parts = sent["messages"][0]["content"]
ok("image_smoke_sends_png", parts[1]["type"] == "image_url"
   and parts[1]["image_url"]["url"].startswith("data:image/png;base64,") and vsrv.image_probe is not None)
vsrv.profile = "qwen_fp8"
ok("text_profile_has_no_vision", not vsrv.vision)

# --- 5b. wheelhouse layout: our kaggle/wheels.ipynb output keeps a wheelhouse/ folder; driessmit's is flat ------
from harness.llm.server import wheelhouse_dir
wh = tmp / "wh"
(wh / "flat").mkdir(parents=True)
(wh / "flat" / "requirements.lock").write_text("vllm==0.19.0\n")
(wh / "nested" / "wheelhouse").mkdir(parents=True)
(wh / "nested" / "wheelhouse" / "requirements.lock").write_text("vllm==0.19.0\n")
(wh / "empty").mkdir()
ok("wheelhouse_flat", wheelhouse_dir(wh / "flat") == wh / "flat")
ok("wheelhouse_nested", wheelhouse_dir(wh / "nested") == wh / "nested" / "wheelhouse")
try:
    wheelhouse_dir(wh / "empty")
    ok("wheelhouse_missing_fails", False)
except FileNotFoundError as exc:
    ok("wheelhouse_missing_fails", "requirements.lock" in str(exc), str(exc))

# --- 6. hygiene: the shared files name no model --------------------------------------------------------------
llm_dir = Path(_ROOT) / _PRIME / "harness" / "llm"
for f in ("client.py", "server.py", "bench.py"):  # spec.py is the registry: it lists the model modules by name
    src = (llm_dir / f).read_text(encoding="utf-8").lower().replace("gemma.py", "").replace("qwen.py", "")
    ok(f"hygiene_{f}", "qwen" not in src and "gemma" not in src and "nemotron" not in src)

# --- 7. Nemotron: history thinking kept (prefix cache), effort mapped onto medium_effort, card parsers ----------
if (llm_dir / "nemotron.py").is_file():
    nm = get_spec("nemotron")
    cn = build_config({"model": "nemotron"})
    ok("nemotron_keeps_history_thinking", cn["llm"]["chat_template_kwargs"].get("truncate_history_thinking") is False
       and all('"truncate_history_thinking": false' in p["flags"]["--default-chat-template-kwargs"]
               for p in nm.profiles.values()))
    ok("nemotron_sampling", (cn["llm"]["temperature"], cn["llm"]["top_p"], cn["llm"].get("top_k")) == (1.0, 0.95, None))
    ok("nemotron_effort_map", nm.template_kwargs({"reasoning_effort": "low"}) == {"medium_effort": True}
       and nm.template_kwargs({"reasoning_effort": "medium"}) == {} and nm.template_kwargs({"reasoning_effort": "xhigh"}) == {})
    ok("nemotron_parsers", all(p["flags"]["--reasoning-parser"] == "nemotron_3"
                               and p["flags"]["--tool-call-parser"] == "qwen3_coder" for p in nm.profiles.values()))
    ok("nemotron_chain_vision_first", nm.has_vision(nm.profile_chain[0]) and nm.max_model_len(nm.profile_chain[0]) == 131072)
    ok("qwen_effort_passthrough", q.template_kwargs({"reasoning_effort": "low"}) == {"reasoning_effort": "low"})

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
_sys.exit(1 if FAILS else 0)
