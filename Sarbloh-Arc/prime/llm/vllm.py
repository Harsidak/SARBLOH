"""vLLM on Kaggle for Gemma 4: offline install, a profile chain, a tool-call smoke test, a throughput probe, a watchdog.

Adapted from kaggle/main.ipynb (the Duck launcher). Differences:
- The driessmit wheelhouse ships vLLM 0.19.0 with transformers 4.57.6, which does not know ``model_type: gemma4``.
  The ``sarbloh-wheels-gemma4`` overlay (transformers 5.5.0 + huggingface_hub 1.x) is installed on top with
  --no-deps (vllm-project/vllm#39216). UNCONFIRMED until the first run: that vLLM 0.19.0 runs with it.
- The smoke test decides the agent's tool mode: "native" when vLLM parses a Gemma 4 tool call, else "fenced"
  (the agent then executes ```python blocks from the reply text). A server that is up is never wasted.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

GEMMA_BASE = {
    "--quantization": "modelopt",
    "--enable-prefix-caching": None,
    "--generation-config": "vllm",
}
TOOL_FLAGS = {"--enable-auto-tool-choice": None, "--tool-call-parser": "gemma4", "--reasoning-parser": "gemma4"}

# Tried in order. The first that starts wins; the smoke test then picks the tool mode.
PROFILES: dict[str, dict[str, Any]] = {
    # Long context, fp8 KV (the checkpoint's hf_quant_config sets kv_cache_quant_algo FP8), text only.
    "gemma_fast": {**GEMMA_BASE, **TOOL_FLAGS, "--max-model-len": "131072", "--kv-cache-dtype": "fp8",
                   "--max-num-seqs": "32", "--max-num-batched-tokens": "8192", "--gpu-memory-utilization": "0.92",
                   "--limit-mm-per-prompt": '{"image": 0, "audio": 0, "video": 0}'},
    # Fewer knobs: default KV dtype, 64k context.
    "gemma_safe": {**GEMMA_BASE, **TOOL_FLAGS, "--max-model-len": "65536", "--gpu-memory-utilization": "0.90"},
    # No parsers at all: fenced-code mode, eager mode, short context. The last resort.
    "gemma_min": {"--quantization": "modelopt", "--max-model-len": "32768", "--enforce-eager": None,
                  "--gpu-memory-utilization": "0.90"},
}


def find_kaggle_input(ref: str) -> Path:
    owner, slug = ref.split("/", 1)
    for p in (Path("/kaggle/input/datasets") / owner / slug, Path("/kaggle/input") / slug):
        if p.exists():
            return p
    for p in Path("/kaggle/input").glob(f"**/{slug}"):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Kaggle input {ref!r} is not attached")


def find_model_dir(root: Path) -> Path:
    if (root / "config.json").exists():
        return root
    hits = sorted(root.glob("**/config.json"), key=lambda p: len(p.parts))
    if not hits:
        raise FileNotFoundError(f"no config.json under {root}")
    return hits[0].parent


def _request_json(url: str, payload: dict | None = None, timeout: float = 30) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def log(msg: str) -> None:
    print(f"[vllm {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class VllmServer:
    def __init__(self, cfg: dict[str, Any], working_dir: Path) -> None:
        self.cfg = cfg
        self.site = working_dir / "vllm-site-packages"
        self.log_path = working_dir / "vllm-server.log"
        self.base_url = f"http://127.0.0.1:{cfg['port']}/v1"
        self.process: subprocess.Popen | None = None
        self.profile: str | None = None
        self.tool_mode = "native"
        self.attempts: list[dict[str, Any]] = []
        self.bench: dict[str, Any] = {}
        self.restarts = 0
        self._stop = threading.Event()

    # --- install -----------------------------------------------------------------------------------------
    def libcuda_link_dir(self) -> Path | None:
        """FlashInfer JIT-links its sm120 NVFP4 GEMM with ``-lcuda``. The Kaggle image ships only the driver's
        ``libcuda.so.1`` (no unversioned ``libcuda.so``, no CUDA stub), so ld fails (E003 v1: "cannot find -lcuda").
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

    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(self.site), env.get("PYTHONPATH", "")) if p)
        link_dir = self.libcuda_link_dir()
        if link_dir is not None:
            env["LIBRARY_PATH"] = os.pathsep.join(p for p in (str(link_dir), env.get("LIBRARY_PATH", "")) if p)
        env.update({"USE_TF": "0", "TRANSFORMERS_NO_TF": "1", "TRANSFORMERS_NO_TORCHVISION": "1",
                    "VLLM_NO_USAGE_STATS": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        return env

    def install(self) -> None:
        stamp = self.site / ".sarbloh-stamp"
        want = json.dumps([self.cfg["wheelhouse_dataset"], self.cfg["overlay_dataset"]])
        if stamp.exists() and stamp.read_text(encoding="utf-8") == want:
            log("install cached")
            return
        wheelhouse = find_kaggle_input(self.cfg["wheelhouse_dataset"])
        overlay = find_kaggle_input(self.cfg["overlay_dataset"])
        shutil.rmtree(self.site, ignore_errors=True)
        self.site.mkdir(parents=True)
        t = time.time()
        pip = [sys.executable, "-m", "pip", "install", "--no-index", "--target", str(self.site), "--upgrade",
               "--only-binary", ":all:", "--no-compile", "--disable-pip-version-check", "--no-warn-conflicts",
               "--quiet"]
        subprocess.run([*pip, "--find-links", str(wheelhouse), "--ignore-installed",
                        "--requirement", str(wheelhouse / "requirements.lock")], check=True)
        overlay_wheels = sorted(overlay.glob("*.whl"))
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
        log(f"installed in {time.time() - t:.0f}s: vllm/torch/transformers/hub = {check.stdout.strip()}")

    # --- serve -------------------------------------------------------------------------------------------
    def healthy(self, timeout: float = 5) -> bool:
        try:
            _request_json(f"{self.base_url}/models", timeout=timeout)
            return True
        except Exception:  # noqa: BLE001
            return False

    def log_tail(self, lines: int = 60) -> str:
        if not self.log_path.exists():
            return ""
        return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])

    def failure_excerpt(self, lines: int = 40) -> str:
        """The engine's own error lines from the last profile section. The plain tail is only the APIServer
        traceback ("See root cause above"), which hid the real cause in E003 v1."""
        if not self.log_path.exists():
            return ""
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        section = text.rsplit("\n===== profile=", 1)[-1].splitlines()
        keys = ("ERROR", "Error", "error:", "FAILED", "No such file", "out of memory")
        hits = [l[:600] for l in section if "(APIServer" not in l and any(k in l for k in keys)]
        return "\n".join(hits[-lines:]) or self.log_tail(lines)

    def launch(self, profile: str) -> None:
        model_dir = find_model_dir(find_kaggle_input(self.cfg["model_dataset"]))
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model_dir),
               "--served-model-name", self.cfg["served_model_name"], "--host", "127.0.0.1",
               "--port", str(self.cfg["port"])]
        for flag, value in PROFILES[profile].items():
            cmd += [flag] if value is None else [flag, value]
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n===== profile={profile} {time.ctime()} =====\n{' '.join(cmd)}\n")
        env = self.env()
        log(f"starting profile={profile} LIBRARY_PATH={env.get('LIBRARY_PATH', '')!r}")
        self.process = subprocess.Popen(cmd, env=env, stdout=self.log_path.open("a", encoding="utf-8"),
                                        stderr=subprocess.STDOUT, text=True, start_new_session=True)
        self.profile = profile

    def wait_ready(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process is None or self.process.poll() is not None:
                return False
            if self.healthy():
                return True
            time.sleep(5)
        return False

    def stop(self) -> None:
        self._stop.set()
        proc, self.process = self.process, None
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:  # noqa: BLE001
            proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                proc.kill()
        for _ in range(30):
            if not self.healthy(timeout=2):
                break
            time.sleep(2)

    def smoke_test(self) -> str:
        """Returns the tool mode this server supports: "native" if a Gemma 4 tool call is parsed, else "fenced"."""
        name = self.cfg["served_model_name"]
        tools = [{"type": "function", "function": {
            "name": "ipython", "description": "Run python code.",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
        t = time.time()
        out = _request_json(f"{self.base_url}/chat/completions", {
            "model": name, "temperature": 0.0, "max_tokens": 1024, "tools": tools,
            "messages": [{"role": "user", "content": "Use the ipython tool to compute 17*23. Call the tool."}]},
            timeout=600)
        msg = out["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if calls and calls[0]["function"]["name"] == "ipython":
            json.loads(calls[0]["function"]["arguments"])
            log(f"smoke ok (native tools) in {time.time() - t:.1f}s: {calls[0]['function']['arguments'][:120]!r}")
            # Thinking on: check the reasoning parser splits thoughts out of the content.
            out = _request_json(f"{self.base_url}/chat/completions", {
                "model": name, "temperature": 0.0, "max_tokens": 2048,
                "chat_template_kwargs": {"enable_thinking": True},
                "messages": [{"role": "user", "content": "Is 391 prime? Answer yes or no."}]}, timeout=600)
            m = out["choices"][0]["message"]
            log(f"thinking probe: reasoning_chars={len(m.get('reasoning_content') or m.get('reasoning') or '')} "
                f"content={str(m.get('content'))[:200]!r}")
            return "native"
        log(f"smoke: no native tool call parsed; using fenced mode. message={json.dumps(msg)[:800]}")
        return "fenced"

    def start(self) -> None:
        self.install()
        chain = self.cfg["profile_chain"]
        for profile in chain:
            t = time.time()
            self.launch(profile)
            ok, err = self.wait_ready(self.cfg["startup_timeout_s"]), ""
            if ok:
                try:
                    self.tool_mode = self.smoke_test() if "--tool-call-parser" in PROFILES[profile] else "fenced"
                except Exception as exc:  # noqa: BLE001 - the server is up; fall back to fenced code
                    err = repr(exc)
                    self.tool_mode = "fenced"
                    log(f"smoke test error on {profile}: {err[:800]}")
            else:
                err = "did not become ready"
            self.attempts.append({"profile": profile, "ok": ok, "seconds": round(time.time() - t),
                                  "tool_mode": self.tool_mode if ok else None, "error": err[:500]})
            if ok:
                log(f"READY profile={profile} tool_mode={self.tool_mode} after {time.time() - t:.0f}s")
                return
            log(f"profile={profile} FAILED ({err}); engine errors:\n{self.failure_excerpt()}")
            self.stop()
            self._stop.clear()
        raise RuntimeError(f"vLLM failed on every profile: {self.attempts}")

    def throughput_probe(self, concurrency: int = 8, max_tokens: int = 512) -> dict[str, Any]:
        payload = {"model": self.cfg["served_model_name"], "temperature": 0.7, "max_tokens": max_tokens,
                   "ignore_eos": True, "messages": [{"role": "user", "content": "Describe a grid puzzle game."}]}

        def one(_: int) -> int:
            out = _request_json(f"{self.base_url}/chat/completions", payload, timeout=900)
            return (out.get("usage") or {}).get("completion_tokens", 0)

        t = time.time()
        single = one(0)
        single_tps = single / max(1e-6, time.time() - t)
        t = time.time()
        with ThreadPoolExecutor(concurrency) as pool:
            total = sum(pool.map(one, range(concurrency)))
        self.bench = {"single_stream_tok_s": round(single_tps, 1),
                      "aggregate_tok_s": round(total / max(1e-6, time.time() - t), 1), "concurrency": concurrency}
        log(f"throughput: {self.bench}")
        return self.bench

    def start_watchdog(self, interval_s: float = 15.0, failures_to_restart: int = 4, max_restarts: int = 2) -> None:
        def loop() -> None:
            failures = 0
            while not self._stop.wait(interval_s):
                dead = self.process is None or self.process.poll() is not None
                if not dead and self.healthy():
                    failures = 0
                    continue
                failures += 1
                if (dead or failures >= failures_to_restart) and self.restarts < max_restarts and self.profile:
                    self.restarts += 1
                    log(f"[watchdog] restart #{self.restarts} dead={dead}\n{self.log_tail(30)}")
                    try:
                        proc, self.process = self.process, None
                        if proc is not None and proc.poll() is None:
                            os.killpg(proc.pid, signal.SIGKILL)
                        self.launch(self.profile)
                        log(f"[watchdog] ready={self.wait_ready(self.cfg['startup_timeout_s'])}")
                    except Exception as exc:  # noqa: BLE001
                        log(f"[watchdog] restart failed: {exc!r}")
                    failures = 0

        threading.Thread(target=loop, name="vllm-watchdog", daemon=True).start()
