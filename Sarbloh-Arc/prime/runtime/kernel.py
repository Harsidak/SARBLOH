"""Host side of the Prime Agent kernel: drives the upstream ``python -m rlm.repl`` runtime over NDJSON.

This is a Python port of ``packages/coding-agent/src/core/kernel/repl-manager.ts`` (upstream commit 2d24ad4),
reduced to what one session needs: start, execute with timeout + interrupt, typed host requests, shutdown.
The wire protocol is specified in ``rlm/repl.md`` (protocol 3).

Host requests (``await rlm.host_request(...)`` in a cell) are answered on a worker thread so a slow handler
(``rlm.collect`` with a timeout) never stops output from being drained.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PRIME_ROOT = Path(__file__).resolve().parents[2]  # holds rlm/ (upstream runtime) and prime/
KERNEL_SKILLS = Path(__file__).resolve().parent / "skills"
PROTOCOL_VERSION = 3

HostHandler = Callable[[dict[str, Any]], dict[str, Any]]

BOOT_CODE = """
import os, sys, json, math, re, itertools, collections, functools, random
from pathlib import Path
from rlm import rlm, bash
import rlm as _rlm_module
import agent_message
try:
    import numpy as np
except Exception:
    np = None
try:
    import arc
except Exception as _arc_exc:
    arc = None
try:
    import worldmodel as wm
except Exception as _wm_exc:
    wm = None
"""


@dataclass
class ExecResult:
    status: str = "ok"            # ok | error | timeout | dead
    stdout: str = ""
    result: str | None = None
    error: str | None = None
    duration_s: float = 0.0

    def render(self, limit: int) -> str:
        parts: list[str] = []
        if self.stdout:
            parts.append(self.stdout.rstrip("\n"))
        if self.result is not None:
            parts.append(self.result)
        if self.error:
            parts.append(self.error.rstrip("\n"))
        if self.status == "timeout":
            parts.append("[cell interrupted: execution timeout]")
        elif self.status == "dead":
            parts.append("[kernel died; it was restarted and Python state was lost]")
        text = "\n".join(parts) if parts else "[no output]"
        if len(text) > limit:
            head = limit * 2 // 3
            tail = limit - head
            text = (f"{text[:head]}\n... [{len(text) - limit} characters truncated; keep large values in variables "
                    f"and print slices] ...\n{text[-tail:]}")
        return text


@dataclass
class _Pending:
    events: queue.Queue[dict[str, Any]] = field(default_factory=queue.Queue)


class Kernel:
    """One persistent Python REPL process (L2 in the paper's state hierarchy)."""

    def __init__(self, session_dir: Path, host_handler: HostHandler, env: dict[str, str] | None = None,
                 python: str = sys.executable) -> None:
        self.session_dir = session_dir
        self.host_handler = host_handler
        self.extra_env = env or {}
        self.python = python
        self.proc: subprocess.Popen | None = None
        self._write_lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}
        self._orphan_output: deque[str] = deque(maxlen=200)  # id:null writes (raw fd / threads)
        self._stderr_tail: deque[str] = deque(maxlen=200)
        self._ready = threading.Event()
        self.restarts = 0

    # --- lifecycle ---------------------------------------------------------------------------------------
    def start(self, timeout_s: float = 60.0) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(PRIME_ROOT), str(KERNEL_SKILLS), env.get("PYTHONPATH", "")) if p)
        env["RLM_SESSION_DIR"] = str(self.session_dir)
        env["PRIME_AGENT_KERNEL_OWNER_PID"] = str(os.getpid())
        env["PYTHONIOENCODING"] = "utf-8"
        env["MPLBACKEND"] = "Agg"
        if os.name == "nt" and "PRIME_AGENT_BASH_SHELL" not in env:  # rlm.bash needs a POSIX shell on Windows
            git_bash = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe"
            if git_bash.exists():
                env["PRIME_AGENT_BASH_SHELL"] = str(git_bash)
        env.update(self.extra_env)
        self._ready.clear()
        self.proc = subprocess.Popen(
            [self.python, "-m", "rlm.repl"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(self.session_dir), env=env, bufsize=0,
        )
        threading.Thread(target=self._read_events, args=(self.proc,), daemon=True, name="kernel-events").start()
        threading.Thread(target=self._read_stderr, args=(self.proc,), daemon=True, name="kernel-stderr").start()
        if not self._ready.wait(timeout_s):
            raise RuntimeError(f"kernel did not become ready: {self.stderr_tail()}")
        boot = self.execute(BOOT_CODE, timeout_s=timeout_s)
        if boot.status != "ok":
            raise RuntimeError(f"kernel boot failed: {boot.render(4000)}")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def close(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            self._send(proc, {"type": "shutdown"})
            proc.stdin.close()
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        if proc.poll() is None:
            proc.kill()
        for pending in list(self._pending.values()):
            pending.events.put({"event": "done", "status": "dead"})

    def restart(self) -> None:
        self.close()
        self.restarts += 1
        self.start()

    def stderr_tail(self, n: int = 40) -> str:
        return "".join(list(self._stderr_tail)[-n:])

    # --- execution ---------------------------------------------------------------------------------------
    def execute(self, code: str, timeout_s: float = 600.0) -> ExecResult:
        if not self.alive():
            self.restart()
        proc = self.proc
        rid = uuid.uuid4().hex
        pending = _Pending()
        self._pending[rid] = pending
        out: list[str] = []
        res = ExecResult()
        t0 = time.monotonic()
        interrupted_at: float | None = None
        try:
            self._send(proc, {"type": "execute", "id": rid, "code": code})
            while True:
                now = time.monotonic()
                if interrupted_at is None and now - t0 > timeout_s:
                    self._send(proc, {"type": "interrupt", "id": rid})
                    interrupted_at = now
                    res.status = "timeout"
                if interrupted_at is not None and now - interrupted_at > 15:
                    # Blocked in native code and deaf to the interrupt: restart, state is lost.
                    self.restart()
                    res.status = "dead"
                    break
                try:
                    ev = pending.events.get(timeout=0.5)
                except queue.Empty:
                    if not self.alive():
                        res.status = "dead"
                        self.restart()
                        break
                    continue
                kind = ev.get("event")
                if kind in ("stdout", "stderr"):
                    out.append(ev.get("text", ""))
                elif kind == "result":
                    res.result = ev.get("text")
                elif kind == "display":
                    out.append(f"[display: {', '.join(ev.get('data', {}).keys())}]\n")
                elif kind == "error":
                    tb = "".join(ev.get("traceback") or [])
                    res.error = tb or f"{ev.get('ename')}: {ev.get('evalue')}"
                elif kind == "done":
                    if res.status == "ok":
                        res.status = "ok" if ev.get("status") == "ok" else ("dead" if ev.get("status") == "dead" else "error")
                    break
        finally:
            self._pending.pop(rid, None)
        while self._orphan_output:
            out.append(self._orphan_output.popleft())
        res.stdout = "".join(out)
        res.duration_s = time.monotonic() - t0
        return res

    # --- protocol plumbing -------------------------------------------------------------------------------
    def _send(self, proc: subprocess.Popen, obj: dict[str, Any]) -> None:
        line = (json.dumps(obj) + "\n").encode("utf-8")
        with self._write_lock:
            proc.stdin.write(line)
            proc.stdin.flush()

    def _read_events(self, proc: subprocess.Popen) -> None:
        for raw in proc.stdout:
            try:
                ev = json.loads(raw.decode("utf-8"))
            except ValueError:
                continue
            kind = ev.get("event")
            if kind == "ready":
                if ev.get("protocol") != PROTOCOL_VERSION:
                    self._stderr_tail.append(f"protocol mismatch: {ev}\n")
                self._ready.set()
                continue
            if kind == "host_request":
                threading.Thread(target=self._answer_host, args=(proc, ev), daemon=True).start()
                continue
            rid = ev.get("id")
            target = self._pending.get(rid) if rid else None
            if target is not None:
                target.events.put(ev)
            elif kind in ("stdout", "stderr"):
                self._orphan_output.append(ev.get("text", ""))

    def _answer_host(self, proc: subprocess.Popen, ev: dict[str, Any]) -> None:
        data = ev.get("data") or {}
        try:
            reply = {"status": "ok", "result": self.host_handler(data)}
        except Exception as exc:  # noqa: BLE001 - errors travel back into the cell as RuntimeError
            reply = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        try:
            self._send(proc, {"type": "host_reply", "id": ev.get("id"), "data": reply})
        except Exception:  # noqa: BLE001
            pass

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        for raw in proc.stderr:
            self._stderr_tail.append(raw.decode("utf-8", errors="replace"))
