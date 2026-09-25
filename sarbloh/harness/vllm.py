"""vLLM server lifecycle on Kaggle: install the offline wheelhouse, start, health-check, watchdog, stop.

Ported from the Duck's setup/teardown commands and extended with server profiles (sarbloh.config.VLLM_PROFILES)
and a restart watchdog (the DuckQwen notebook's idea). If the chosen profile does not come up, the server is
restarted with ``fallback_profile`` so a bad speed experiment never costs a whole run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from sarbloh.config import VLLM_PROFILES


def find_kaggle_input(ref: str) -> Path:
    """Where Kaggle mounted a dataset or model: /kaggle/input/<slug>, .../datasets/<owner>/<slug>, or a search."""
    owner, slug = ref.split("/", 1)
    for p in (Path("/kaggle/input") / slug, Path("/kaggle/input/datasets") / owner / slug):
        if p.exists():
            return p
    for p in Path("/kaggle/input").glob(f"**/{slug}"):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Kaggle input {ref!r} is not attached")


def find_model_dir(root: Path) -> Path:
    """The directory holding config.json (model snapshots are sometimes nested)."""
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


class VllmServer:
    def __init__(self, config: dict[str, Any], working_dir: Path) -> None:
        self.cfg = config["vllm"]
        self.working_dir = working_dir
        self.site_packages = working_dir / "vllm-site-packages"
        self.log_path = working_dir / "vllm-server.log"
        self.base_url = f"http://127.0.0.1:{self.cfg['port']}/v1"
        self.process: subprocess.Popen | None = None
        self.active_profile: str | None = None
        self._watchdog: threading.Thread | None = None
        self._watchdog_stop = threading.Event()
        self.restarts = 0

    # --- install -----------------------------------------------------------------------------------------
    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(self.site_packages), env.get("PYTHONPATH", "")) if p)
        env.update({"USE_TF": "0", "TRANSFORMERS_NO_TF": "1", "TRANSFORMERS_NO_TORCHVISION": "1", "VLLM_NO_USAGE_STATS": "1"})
        return env

    def install(self) -> None:
        stamp = self.site_packages / ".sarbloh-stamp"
        if stamp.exists() and stamp.read_text(encoding="utf-8") == self.cfg["wheelhouse_stamp"]:
            print(f"[vllm] using cached install at {self.site_packages}", flush=True)
            return
        wheelhouse = find_kaggle_input(self.cfg["wheelhouse_dataset"])
        lock = wheelhouse / "requirements.lock"
        if not lock.exists():
            raise FileNotFoundError(f"missing {lock}")
        shutil.rmtree(self.site_packages, ignore_errors=True)
        self.site_packages.mkdir(parents=True)
        t = time.time()
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(wheelhouse), "--requirement",
             str(lock), "--target", str(self.site_packages), "--upgrade", "--ignore-installed", "--only-binary",
             ":all:", "--no-compile", "--disable-pip-version-check", "--no-warn-conflicts", "--quiet"],
            check=True,
        )
        stamp.write_text(self.cfg["wheelhouse_stamp"], encoding="utf-8")
        print(f"[vllm] wheelhouse installed in {time.time() - t:.0f}s", flush=True)

    # --- start / stop ------------------------------------------------------------------------------------
    def command(self, profile_name: str) -> tuple[list[str], dict[str, str]]:
        prof = VLLM_PROFILES[profile_name]
        model_dir = find_model_dir(find_kaggle_input(self.cfg["model_dataset"]))
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(model_dir),
            "--served-model-name", self.cfg["served_model_name"],
            "--host", "127.0.0.1", "--port", str(self.cfg["port"]),
            "--tensor-parallel-size", str(prof["tensor_parallel_size"]),
            "--max-model-len", str(prof["max_model_len"]),
            "--enable-auto-tool-choice", "--tool-call-parser", self.cfg["tool_call_parser"],
            "--reasoning-parser", self.cfg["reasoning_parser"],
            "--generation-config", "vllm",
            "--default-chat-template-kwargs", json.dumps({"preserve_thinking": True}),
        ]
        cmd.append("--enable-prefix-caching" if prof.get("enable_prefix_caching") else "--no-enable-prefix-caching")
        if prof.get("speculative_config"):
            cmd += ["--speculative-config", json.dumps(prof["speculative_config"])]
        for key, flag in (
            ("max_num_batched_tokens", "--max-num-batched-tokens"),
            ("max_num_seqs", "--max-num-seqs"),
            ("max_cudagraph_capture_size", "--max-cudagraph-capture-size"),
            ("kv_cache_memory_bytes", "--kv-cache-memory-bytes"),
            ("gpu_memory_utilization", "--gpu-memory-utilization"),
            ("kv_cache_dtype", "--kv-cache-dtype"),
        ):
            if prof.get(key) is not None:
                cmd += [flag, str(prof[key])]
        cmd += list(prof.get("extra_args", []))
        env = self.env()
        env.update(prof.get("env", {}))
        return cmd, env

    def _launch(self, profile_name: str) -> None:
        cmd, env = self.command(profile_name)
        print(f"[vllm] starting profile={profile_name}: {' '.join(cmd)}", flush=True)
        log = self.log_path.open("a", encoding="utf-8")
        log.write(f"\n===== profile={profile_name} {time.ctime()} =====\n")
        log.flush()
        self.process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        self.active_profile = profile_name

    def _wait_ready(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                return False
            if self.healthy(timeout=5):
                return True
            time.sleep(5)
        return False

    def healthy(self, timeout: float = 5) -> bool:
        try:
            _request_json(f"{self.base_url}/models", timeout=timeout)
            return True
        except Exception:
            return False

    def start(self) -> None:
        self.install()
        t = time.time()
        order = [self.cfg["profile"]]
        if self.cfg.get("fallback_profile") and self.cfg["fallback_profile"] not in order:
            order.append(self.cfg["fallback_profile"])
        for profile in order:
            self._launch(profile)
            if self._wait_ready(self.cfg["startup_timeout_s"]):
                print(f"[vllm] ready with profile={profile} after {time.time() - t:.0f}s", flush=True)
                self.smoke_test()
                return
            print(f"[vllm] profile={profile} failed to come up; log tail:\n{self.tail()}", flush=True)
            self.kill()
        raise RuntimeError("vLLM did not start with any profile")

    def smoke_test(self) -> None:
        payload = {
            "model": self.cfg["served_model_name"],
            "messages": [{"role": "user", "content": "Answer in one short sentence: what is 2 + 2?"}],
            "temperature": 0.0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t = time.time()
        out = _request_json(f"{self.base_url}/chat/completions", payload=payload, timeout=300)
        n = (out.get("usage") or {}).get("completion_tokens", 0)
        text = (out["choices"][0]["message"].get("content") or "").strip()
        print(f"[vllm] smoke: {text!r} ({n} tok, {n / max(1e-6, time.time() - t):.1f} tok/s single-stream)", flush=True)

    def tail(self, lines: int = 60) -> str:
        if not self.log_path.exists():
            return ""
        return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])

    def kill(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.process = None

    # --- watchdog ----------------------------------------------------------------------------------------
    def start_watchdog(self, interval_s: float = 15.0, failures_to_restart: int = 4, max_restarts: int = 2) -> None:
        def loop() -> None:
            failures = 0
            while not self._watchdog_stop.wait(interval_s):
                dead = self.process is None or self.process.poll() is not None
                if not dead and self.healthy(timeout=5):
                    failures = 0
                    continue
                failures += 1
                if (dead or failures >= failures_to_restart) and self.restarts < max_restarts:
                    self.restarts += 1
                    print(f"[vllm-watchdog] restarting (dead={dead}, failures={failures}, n={self.restarts})", flush=True)
                    self.kill()
                    self._launch(self.active_profile or self.cfg["fallback_profile"])
                    self._wait_ready(self.cfg["startup_timeout_s"])
                    failures = 0

        self._watchdog = threading.Thread(target=loop, name="vllm-watchdog", daemon=True)
        self._watchdog.start()

    def stop(self) -> None:
        self._watchdog_stop.set()
        self.kill()
