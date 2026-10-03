"""The model server on Kaggle (vLLM or SGLang), model-free: offline install, a profile chain, a tool-call smoke test and a watchdog. Everything
model-specific (weights, flags, parsers, sampling) comes from a ``ModelSpec`` (gemma.py, qwen.py).

Robustness, from a past run (three APIServer freezes; the fixed 2-restart budget ran out, then 900 s x 4 client retries
blocked every game for about 30 minutes):
- A freeze counts as a failure even when the process is alive: /metrics must answer, and its token counters must move
  while requests are running.
- Before a kill, SIGABRT goes to the whole process group with PYTHONFAULTHANDLER=1, so every freeze leaves the Python
  stacks of the API server and the engine in llm-server.log.
- ``ServerGate`` goes down before the kill, so in-flight requests fail fast on the closed socket and wait for the
  restart without spending retries. After the restart they are admitted a few at a time.
- Restarts are budgeted by remaining wall clock, not a fixed count. After ``fallback_after`` freezes on one profile,
  the watchdog moves to the next profile in the chain with the same tool mode.
- Every server event is one line in server_events.jsonl.
- Site-packages live in /tmp, not /kaggle/working (10+ GB of wheels once made the output download hang).

A prebuilt runtime may bring its own interpreter (``python``) and server (``backend: "sglang"``, see sglang.py).
The watchdog then reads ``sglang:*`` counters, and a /metrics that does not answer counts
as a failure only when the health endpoint does not answer either.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from harness.llm.client import ServerGate
from harness.llm.spec import ModelSpec

WATCHDOG = {"interval_s": 15.0, "failures_to_restart": 4, "freeze_after_s": 120.0, "fallback_after": 2,
            "max_restarts": 8, "min_useful_s": 600.0}
# Prepared prebuilt runtimes (ModelSpec.runtimes), by name: unpacking one takes minutes, so every server in the
# process (bench profiles, restarts) shares it.
_RUNTIMES: dict[str, dict[str, Any]] = {}
DEFAULT_SERVE = ["-m", "vllm.entrypoints.openai.api_server", "--model"]


def find_kaggle_input(ref: str) -> Path:
    """A dataset (owner/slug) or model (owner/slug[/framework/variation/version]) attached to the notebook."""
    parts = ref.split("/")
    owner, slug = parts[0], parts[1]
    for p in (Path("/kaggle/input/datasets") / owner / slug, Path("/kaggle/input/models") / owner / slug,
              Path("/kaggle/input") / slug):
        if p.exists():
            return p
    for p in Path("/kaggle/input").glob(f"**/{slug}"):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Kaggle input {ref!r} is not attached")


def wheelhouse_dir(root: Path) -> Path:
    """The folder holding requirements.lock: the input root (driessmit's flat layout) or one level down (a dataset
    made from kaggle/wheels.ipynb's output keeps its wheelhouse/ folder)."""
    if (root / "requirements.lock").exists():
        return root
    for lock in sorted(root.glob("*/requirements.lock")):
        return lock.parent
    raise FileNotFoundError(f"no requirements.lock in {root} or one level below")


def find_model_dir(root: Path) -> Path:
    if (root / "config.json").exists():
        return root
    hits = sorted((p for p in root.glob("**/config.json") if ".git" not in p.parts), key=lambda p: len(p.parts))
    if not hits:
        raise FileNotFoundError(f"no config.json under {root}")
    return hits[0].parent


def _request(url: str, payload: dict | None = None, timeout: float = 30) -> bytes:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def request_json(url: str, payload: dict | None = None, timeout: float = 30) -> dict:
    return json.loads(_request(url, payload, timeout).decode("utf-8"))


_METRIC = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)")


def parse_metrics(text: str) -> dict[str, float]:
    """Prometheus text -> {name: value summed over label sets}. ``_created`` timestamps are dropped."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _METRIC.match(line)
        if m and not m.group(1).endswith("_created"):
            try:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
            except ValueError:
                pass
    return out


def token_counters(m: dict[str, float]) -> tuple[float, float]:
    """(prompt + generation tokens so far, requests running) from vLLM or SGLang metrics; (0, 0) when absent.
    SGLang's ``prompt/generation_tokens_total`` only move when a request finishes, so one long thinking request
    looked frozen and was killed (an ls20 run); ``sglang:realtime_tokens_total`` moves every decode step."""
    tokens = sum(m.get(f"{p}:{k}", 0.0) for p in ("vllm", "sglang")
                 for k in ("prompt_tokens_total", "generation_tokens_total", "realtime_tokens_total"))
    running = m.get("vllm:num_requests_running", 0.0) + m.get("sglang:num_running_reqs", 0.0)
    return tokens, running


def log(msg: str) -> None:
    print(f"[server {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class LlmServer:
    def __init__(self, spec: ModelSpec, cfg: dict[str, Any], working_dir: Path, gate: ServerGate | None = None,
                 deadline: float | None = None) -> None:
        self.spec = spec
        self.cfg = cfg
        self.working_dir = working_dir
        self.site = self._site_dir(cfg, working_dir)
        self.log_path = working_dir / "llm-server.log"
        self.events_path = working_dir / "server_events.jsonl"
        self.root_url = f"http://127.0.0.1:{cfg['port']}"
        self.base_url = f"{self.root_url}/v1"
        self.chain = list(cfg.get("profile_chain") or spec.profile_chain)
        unknown = [p for p in self.chain if p not in spec.profiles]
        if unknown:  # e.g. a stale notebook forcing another model's chain: say so once, before any launch
            raise ValueError(f"profile_chain {unknown} not in model {spec.name!r}; known: {sorted(spec.profiles)}")
        self.gate = gate or ServerGate(max_inflight=cfg.get("max_inflight"))
        self.deadline = deadline  # time.time() by which the run ends; None = no wall-clock budget
        self.wd = {**WATCHDOG, **(cfg.get("watchdog_cfg") or {})}
        self.process: subprocess.Popen | None = None
        self.profile: str | None = None
        self.tool_mode = "native"
        self.attempts: list[dict[str, Any]] = []
        self.bench: dict[str, Any] = {}
        self.restarts = 0
        self.freezes: dict[str, int] = {}
        self.startup_s: float | None = None
        self.image_probe: str | None = None  # the model's answer to the image smoke test (vision profiles)
        self.backend = "vllm"                # "sglang" when the running profile's runtime says so
        self._stop = threading.Event()
        self._lock = threading.RLock()

    @staticmethod
    def _site_dir(cfg: dict[str, Any], working_dir: Path) -> Path:
        if cfg.get("site_dir"):
            return Path(cfg["site_dir"])
        tmp = Path("/tmp/sarbloh-vllm")
        try:
            tmp.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(tmp).free > 30e9:
                return tmp / "site-packages"
        except OSError:
            pass
        return working_dir / "vllm-site-packages"

    @property
    def max_model_len(self) -> int:
        return self.spec.max_model_len(self.profile) if self.profile else 0

    @property
    def vision(self) -> bool:
        """The running profile accepts an image per prompt (and passed the image smoke test)."""
        return bool(self.profile) and self.spec.has_vision(self.profile)

    def event(self, name: str, **detail: Any) -> None:
        row = {"t": round(time.time(), 1), "event": name, "profile": self.profile, **detail}
        try:
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        except OSError:
            pass
        log(f"{name} {json.dumps(detail, default=str)[:600]}")

    # --- install -----------------------------------------------------------------------------------------
    def libcuda_link_dir(self) -> Path | None:
        """FlashInfer JIT-links its sm120 NVFP4 GEMM with ``-lcuda``. The Kaggle image ships only the driver's
        ``libcuda.so.1`` (no unversioned ``libcuda.so``, no CUDA stub), so ld fails (seen in the first run: "cannot find -lcuda").
        Give ld a ``libcuda.so`` symlink via LIBRARY_PATH."""
        if Path("/usr/local/cuda/lib64/stubs/libcuda.so").exists():
            return None
        candidates: list[Path] = []
        try:
            out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout
            candidates += [Path(line.rsplit("=>", 1)[1].strip()) for line in out.splitlines()
                           if "libcuda.so" in line and "=>" in line and "x86-64" in line]
        except Exception:  # noqa: BLE001
            pass
        for d in ("/usr/lib/x86_64-linux-gnu", "/usr/local/nvidia/lib64", "/usr/lib64", "/usr/local/cuda/compat"):
            candidates += sorted(Path(d).glob("libcuda.so*"))
        target = next((p for p in candidates if p.exists()), None)
        if target is None:
            log("WARNING: no libcuda.so* found; FlashInfer JIT link may fail")
            return None
        link_dir = self.site.parent / "libcuda-link"
        link_dir.mkdir(parents=True, exist_ok=True)
        link = link_dir / "libcuda.so"
        if not link.exists():
            link.symlink_to(target.resolve())
        return link_dir

    def runtime(self, profile: str) -> dict[str, Any] | None:
        """The prepared prebuilt runtime this profile runs on, or None for the shared wheelhouse."""
        name = self.spec.profiles[profile].get("runtime")
        if not name:
            return None
        if name not in _RUNTIMES:
            t = time.time()
            _RUNTIMES[name] = self.spec.runtimes[name](self.working_dir, find_kaggle_input)
            self.event("runtime_ready", runtime=name, seconds=round(time.time() - t),
                       info=_RUNTIMES[name].get("info"))
        return _RUNTIMES[name]

    def prepare(self, profile: str) -> None:
        if self.runtime(profile) is None:
            self.install()

    def env(self, profile: str | None = None) -> dict[str, str]:
        rt = self.runtime(profile) if profile else None
        if rt is not None:  # a prebuilt runtime owns its whole environment: none of the wheelhouse paths
            env = dict(rt["env"])
            env.update({"VLLM_NO_USAGE_STATS": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "PYTHONFAULTHANDLER": "1"})
            env.update(self.spec.profiles[profile].get("env") or {})
            return env
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(self.site), env.get("PYTHONPATH", "")) if p)
        link_dir = self.libcuda_link_dir()
        if link_dir is not None:
            env["LIBRARY_PATH"] = os.pathsep.join(p for p in (str(link_dir), env.get("LIBRARY_PATH", "")) if p)
        env.update({"USE_TF": "0", "TRANSFORMERS_NO_TF": "1", "TRANSFORMERS_NO_TORCHVISION": "1",
                    "VLLM_NO_USAGE_STATS": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "PYTHONFAULTHANDLER": "1"})
        if profile:
            # One compile cache per profile: the text-only fallback reused the image profile's compiled graph and
            # crashed in its first forward ("'NoneType' object has no attribute 'size'").
            env["VLLM_CACHE_ROOT"] = str(self.site.parent / "vllm-cache" / profile)
            env.update(self.spec.profiles[profile].get("env") or {})
        return env

    def install(self) -> None:
        stamp = self.site / ".sarbloh-stamp"
        want = json.dumps([self.cfg["wheelhouse_dataset"], self.spec.overlay_dataset])
        if stamp.exists() and stamp.read_text(encoding="utf-8") == want:
            log(f"install cached at {self.site}")
            return
        wheelhouse = wheelhouse_dir(find_kaggle_input(self.cfg["wheelhouse_dataset"]))
        shutil.rmtree(self.site, ignore_errors=True)
        self.site.mkdir(parents=True)
        t = time.time()
        pip = [sys.executable, "-m", "pip", "install", "--no-index", "--target", str(self.site), "--upgrade",
               "--only-binary", ":all:", "--no-compile", "--disable-pip-version-check", "--no-warn-conflicts",
               "--quiet"]
        subprocess.run([*pip, "--find-links", str(wheelhouse), "--ignore-installed",
                        "--requirement", str(wheelhouse / "requirements.lock")], check=True)
        if self.spec.overlay_dataset:
            overlay_wheels = sorted(find_kaggle_input(self.spec.overlay_dataset).glob("*.whl"))
            for whl in overlay_wheels:  # --target --upgrade can leave the old dist-info next to the new one
                pkg = whl.name.split("-")[0]
                for old in [*self.site.glob(f"{pkg}-*.dist-info"), self.site / pkg]:
                    shutil.rmtree(old, ignore_errors=True)
            subprocess.run([*pip, "--no-deps", *map(str, overlay_wheels)], check=True)
        check = subprocess.run(
            [sys.executable, "-c", "import vllm, torch, transformers, huggingface_hub as h; "
                                   "print(vllm.__version__, torch.__version__, transformers.__version__, h.__version__)"],
            env=self.env(), capture_output=True, text=True)
        if check.returncode != 0:
            raise RuntimeError(f"vLLM import failed after install:\n{check.stderr[-3000:]}")
        stamp.write_text(want, encoding="utf-8")
        self.event("installed", seconds=round(time.time() - t), site=str(self.site),
                   versions=check.stdout.strip())

    # --- probes ------------------------------------------------------------------------------------------
    def healthy(self, timeout: float = 5) -> bool:
        try:
            request_json(f"{self.base_url}/models", timeout=timeout)
            return True
        except Exception:  # noqa: BLE001
            return False

    def metrics(self, timeout: float = 10) -> dict[str, float] | None:
        try:
            return parse_metrics(_request(f"{self.root_url}/metrics", timeout=timeout).decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001
            return None

    def log_tail(self, lines: int = 60) -> str:
        if not self.log_path.exists():
            return ""
        return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])

    def failure_excerpt(self, lines: int = 40) -> str:
        """The engine's own error lines from the last profile section. The plain tail is only the APIServer
        traceback ("See root cause above"), which hid the real cause in the first run."""
        if not self.log_path.exists():
            return ""
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        section = text.rsplit("\n===== profile=", 1)[-1].splitlines()
        keys = ("ERROR", "Error", "error:", "FAILED", "No such file", "out of memory")
        hits = [l[:600] for l in section if "(APIServer" not in l and any(k in l for k in keys)]
        return "\n".join(hits[-lines:]) or self.log_tail(lines)

    def stack_dump(self, max_lines: int = 80) -> str:
        """faulthandler output written by the SIGABRT in ``kill``: the Python stack of every thread of every
        process in the group, at the moment of the freeze."""
        if not self.log_path.exists():
            return ""
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        section = text.rsplit("\n===== profile=", 1)[-1]
        i = section.find("Fatal Python error")
        return "\n".join(section[i:].splitlines()[:max_lines]) if i >= 0 else ""

    # --- process -----------------------------------------------------------------------------------------
    def launch(self, profile: str) -> None:
        model_dir = find_model_dir(find_kaggle_input(self.spec.profiles[profile]["model_dataset"]))
        rt = self.runtime(profile) or {}
        python = rt.get("python") or sys.executable
        serve = rt.get("serve", DEFAULT_SERVE)
        self.backend = rt.get("backend") or "vllm"
        cmd = [python, *serve, str(model_dir), "--served-model-name", self.spec.served_model_name,
               "--host", "127.0.0.1", "--port", str(self.cfg["port"])]

        def fill(v: str) -> str:
            return v.replace("{model_dir}", str(model_dir))

        for flag, value in self.spec.profiles[profile]["flags"].items():
            if value is None:
                cmd += [flag]
            elif isinstance(value, (list, tuple)):  # nargs flags, e.g. SGLang --cuda-graph-bs-decode 1 2 4 8
                cmd += [flag, *map(fill, map(str, value))]
            else:
                cmd += [flag, fill(value)]
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n===== profile={profile} {time.ctime()} =====\n{' '.join(cmd)}\n")
        self.profile = profile
        self.event("launch", model_dir=str(model_dir))
        self.process = subprocess.Popen(cmd, env=self.env(profile), stdout=self.log_path.open("a", encoding="utf-8"),
                                        stderr=subprocess.STDOUT, text=True, start_new_session=True)

    def wait_ready(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not self._stop.is_set():
            if self.process is None or self.process.poll() is not None:
                return False
            if self.healthy():
                return True
            time.sleep(5)
        return False

    def kill(self, reason: str, dump: bool = False) -> None:
        """Gate down first (in-flight requests fail fast on the closed socket), optional SIGABRT stack dump, then
        SIGKILL, then wait for the GPU memory to come back so the next launch does not OOM."""
        self.gate.down()
        proc, self.process = self.process, None
        if proc is not None and proc.poll() is None:
            if dump:
                try:
                    os.killpg(proc.pid, signal.SIGABRT)
                    os.killpg(proc.pid, signal.SIGCONT)  # a stopped process only handles the SIGABRT once resumed
                    proc.wait(timeout=20)
                except Exception:  # noqa: BLE001
                    pass
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=30)
                except Exception:  # noqa: BLE001
                    pass
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
            try:
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                pass
        freed = self._wait_gpu_free()
        self.event("killed", reason=reason, dump=dump, gpu_freed=freed)

    @staticmethod
    def _wait_gpu_free(timeout_s: float = 120.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=20).stdout.strip()
            except Exception:  # noqa: BLE001 - no nvidia-smi (local tests)
                return True
            if not out:
                return True
            time.sleep(3)
        return False

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            if self.process is not None:
                self.kill("stop")
            self.gate.fail("server stopped")
        self.event("stopped", restarts=self.restarts, freezes=self.freezes)

    # --- start -------------------------------------------------------------------------------------------
    def smoke_test(self) -> str:
        """Returns the tool mode this server supports: "native" if a tool call is parsed, else "fenced"."""
        name = self.spec.served_model_name
        tools = [{"type": "function", "function": {
            "name": "ipython", "description": "Run python code.",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
        body: dict[str, Any] = {
            "model": name, "temperature": 0.0, "max_tokens": 1024, "tools": tools,
            "messages": [{"role": "user", "content": "Use the ipython tool to compute 17*23. Call the tool."}]}
        if self.spec.smoke_template_kwargs:
            body["chat_template_kwargs"] = dict(self.spec.smoke_template_kwargs)
        t = time.time()
        out = request_json(f"{self.base_url}/chat/completions", body, timeout=600)
        msg = out["choices"][0]["message"]
        calls = self.spec.fix_tool_calls(list(msg.get("tool_calls") or []))
        if calls and calls[0]["function"]["name"] == "ipython":
            json.loads(calls[0]["function"]["arguments"])
            log(f"smoke ok (native tools) in {time.time() - t:.1f}s: {calls[0]['function']['arguments'][:120]!r}")
            # Thinking as the agent will use it: check the reasoning parser splits thoughts out of the content.
            probe: dict[str, Any] = {"model": name, "temperature": 0.0, "max_tokens": 2048,
                                     "messages": [{"role": "user", "content": "Is 391 prime? Answer yes or no."}]}
            if self.spec.llm.get("chat_template_kwargs"):
                probe["chat_template_kwargs"] = dict(self.spec.llm["chat_template_kwargs"])
            m = request_json(f"{self.base_url}/chat/completions", probe, timeout=600)["choices"][0]["message"]
            log(f"thinking probe: reasoning_chars={len(m.get('reasoning_content') or m.get('reasoning') or '')} "
                f"content={str(m.get('content'))[:200]!r}")
            return "native"
        log(f"smoke: no native tool call parsed; using fenced mode. message={json.dumps(msg)[:800]}")
        return "fenced"

    def image_smoke(self) -> str:
        """One request with a small PNG (a red square): the server must accept the image part. The answer is logged,
        not judged: the check is that the multimodal path works, not the model's eyesight."""
        from harness.agent.perception import data_url, render

        body: dict[str, Any] = {"model": self.spec.served_model_name, "temperature": 0.0, "max_tokens": 256,
                                "messages": [{"role": "user", "content": [
                                    {"type": "text", "text": "What colour fills this image? Answer in one word."},
                                    {"type": "image_url", "image_url": {"url": data_url(render([[8] * 16] * 16, 4))}}]}]}
        if self.spec.smoke_template_kwargs:
            body["chat_template_kwargs"] = dict(self.spec.smoke_template_kwargs)
        t = time.time()
        out = request_json(f"{self.base_url}/chat/completions", body, timeout=600)
        msg = out["choices"][0]["message"]
        answer = str(msg.get("content") or msg.get("reasoning_content") or "")[:200]
        log(f"image smoke ok in {time.time() - t:.1f}s: prompt_tokens={(out.get('usage') or {}).get('prompt_tokens')} "
            f"answer={answer!r}")
        self.image_probe = answer
        return answer

    def _start_profile(self, profile: str, smoke: bool = True) -> bool:
        t = time.time()
        try:  # a missing input or a failed runtime unpack fails this profile, not the chain
            self.prepare(profile)
            self.launch(profile)
        except Exception as exc:  # noqa: BLE001
            self.profile = profile
            self.attempts.append({"profile": profile, "ok": False, "seconds": round(time.time() - t),
                                  "tool_mode": None, "error": f"prepare/launch: {exc!r}"[:500]})
            self.event("start_failed", seconds=round(time.time() - t), error=f"prepare/launch: {exc!r}"[:3000])
            return False
        ok, err = self.wait_ready(self.cfg["startup_timeout_s"]), ""
        if ok and smoke:
            try:
                self.tool_mode = self.smoke_test() if self.spec.has_tool_parser(profile) else "fenced"
            except Exception as exc:  # noqa: BLE001 - the server is up; fall back to fenced code
                err = repr(exc)
                self.tool_mode = "fenced"
                log(f"smoke test error on {profile}: {err[:800]}")
        elif not ok:
            err = "did not become ready"
        if ok and smoke and self.spec.has_vision(profile):
            try:  # a vision profile that cannot take an image fails; the chain moves on to a text-only profile
                self.image_smoke()
            except Exception as exc:  # noqa: BLE001
                ok, err = False, f"image smoke failed: {exc!r}"
                log(f"{err[:800]}")
        seconds = round(time.time() - t)
        self.attempts.append({"profile": profile, "ok": ok, "seconds": seconds,
                              "tool_mode": self.tool_mode if ok else None, "error": err[:500]})
        if ok:
            self.startup_s = seconds
            self.event("ready", seconds=seconds, tool_mode=self.tool_mode, max_model_len=self.max_model_len)
        else:
            self.event("start_failed", seconds=seconds, error=err, engine_errors=self.failure_excerpt()[-3000:])
        return ok

    def start(self, chain: list[str] | None = None) -> None:
        self.gate.down()  # nothing is served until a profile passes
        for profile in chain or self.chain:  # each profile installs or unpacks what it runs on (_start_profile)
            if self._start_profile(profile):
                self.gate.up(ramp=False)
                return
            self.kill("start_failed")
        raise RuntimeError(f"the model server failed on every profile: {self.attempts}")

    # --- watchdog ----------------------------------------------------------------------------------------
    def _fallbacks(self) -> list[str]:
        """Profiles to try on a restart: the current one unless it froze ``fallback_after`` times, then the rest of
        the chain. Only profiles with the current tool mode and image support: the agents are already configured for
        them."""
        native = self.tool_mode == "native"
        rest = self.chain[self.chain.index(self.profile) + 1:] if self.profile in self.chain else []
        rest = [p for p in rest if self.spec.has_tool_parser(p) == native
                and self.spec.has_vision(p) == self.spec.has_vision(self.profile)]
        if self.freezes.get(self.profile, 0) < self.wd["fallback_after"]:
            return [self.profile, *rest]
        return rest or [self.profile]

    def _budget_allows_restart(self) -> tuple[bool, str]:
        if self.restarts >= self.wd["max_restarts"]:
            return False, f"max_restarts={self.wd['max_restarts']}"
        if self.deadline is None:
            return True, ""
        left = self.deadline - time.time()
        need = 1.5 * (self.startup_s or 600.0) + self.wd["min_useful_s"]
        return (left > need), f"left={left:.0f}s need={need:.0f}s"

    def recover(self, reason: str) -> bool:
        with self._lock:
            if self._stop.is_set():
                return False
            profile = self.profile
            self.freezes[profile] = self.freezes.get(profile, 0) + 1
            self.event("freeze_detected", reason=reason, count=self.freezes[profile])
            self.kill(reason, dump=True)
            self.event("freeze", reason=reason, count=self.freezes[profile], stacks=self.stack_dump())
            for candidate in self._fallbacks():
                ok, why = self._budget_allows_restart()
                if not ok:
                    self.event("give_up", why=why)
                    self.gate.fail(f"the model server gave up after {reason}: {why}")
                    return False
                self.restarts += 1
                if candidate != profile:
                    self.event("fallback", to=candidate)
                if self._start_profile(candidate, smoke=False):
                    self.gate.up(ramp=True)
                    return True
                self.kill("restart_failed")
            self.event("give_up", why="no profile restarted")
            self.gate.fail(f"the model server could not restart after {reason}")
            return False

    def check(self, state: dict[str, Any]) -> str | None:
        """One watchdog tick. Returns a failure reason, or None if the server is fine. ``state`` carries counters
        between ticks: failures, last token total, and when the token total last moved."""
        if self.process is None or self.process.poll() is not None:
            return "exited"
        m = self.metrics()
        if m is None and self.backend != "vllm" and self.healthy(timeout=10):
            m = {}  # a server without (or with a slow) /metrics is alive if it answers /v1/models
        if m is None:
            state["failures"] = state.get("failures", 0) + 1
            return "unresponsive" if state["failures"] >= self.wd["failures_to_restart"] else None
        state["failures"] = 0
        tokens, running = token_counters(m)
        now = time.monotonic()
        if running <= 0 or tokens != state.get("tokens"):
            state["tokens"], state["moved_at"] = tokens, now
            return None
        return "frozen" if now - state.get("moved_at", now) >= self.wd["freeze_after_s"] else None

    def start_watchdog(self) -> None:
        def loop() -> None:
            state: dict[str, Any] = {}
            logged = False
            while not self._stop.wait(self.wd["interval_s"]):
                if not logged:  # which counters the freeze detector actually sees on this build
                    m = self.metrics()
                    if m is not None:
                        logged = True
                        self.event("metrics_keys", keys=sorted(k for k in m if k.startswith(("vllm:", "sglang:")))[:80])
                reason = self.check(state)
                if reason and not self._stop.is_set():
                    if not self.recover(reason):
                        return
                    state = {}

        self.event("watchdog_on", **self.wd)
        threading.Thread(target=loop, name="server-watchdog", daemon=True).start()
