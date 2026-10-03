"""OpenAI-compatible chat client (vLLM or SGLang on Kaggle, llama.cpp locally). Standard library only: Kaggle is offline.

Model-free: sampling and the thinking policy arrive in ``cfg`` (from the model's spec, see spec.py), and parser quirks
are repaired by ``spec.fix_tool_calls``. ``ServerGate`` is shared with the server watchdog: while the server restarts,
requests wait for it instead of burning retries, and after a restart they ramp back up a few at a time (a past run: the third
freeze came right after a restart, on ten cold 40k-token prefills arriving at once).
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    failures: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, prompt: int, completion: int) -> None:
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.calls += 1

    def to_json(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "calls": self.calls, "failures": self.failures}


@dataclass
class Reply:
    content: str
    reasoning: str
    tool_calls: list[dict[str, Any]]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


class ServerGate:
    """Ready flag plus an in-flight limit, shared by the watchdog (``down``/``up``) and every client (``slot``).

    ``epoch`` counts ``down`` calls: a request that fails while the epoch moved failed because of a restart, not
    because of the request, so it does not use up a retry."""

    def __init__(self, max_inflight: int | None = None, ramp_start: int = 2, ramp_step: int = 2,
                 ramp_every_s: float = 15.0) -> None:
        self._cond = threading.Condition()
        self.ready = True
        self.dead: str | None = None  # set by fail(): the server will not come back, so requests stop waiting
        self.epoch = 0
        self.inflight = 0
        self.max_inflight = max_inflight
        self.ramp_start, self.ramp_step, self.ramp_every_s = ramp_start, ramp_step, ramp_every_s
        self._ramp_t0: float | None = None

    def down(self) -> None:
        with self._cond:
            self.ready = False
            self.epoch += 1
            self._cond.notify_all()

    def fail(self, reason: str) -> None:
        with self._cond:
            self.ready = False
            self.dead = reason
            self.epoch += 1
            self._cond.notify_all()

    def up(self, ramp: bool = True) -> None:
        with self._cond:
            self.ready = True
            self._ramp_t0 = time.monotonic() if ramp else None
            self._cond.notify_all()

    def capacity(self) -> float:
        cap = float("inf") if self.max_inflight is None else float(self.max_inflight)
        if self._ramp_t0 is not None:
            steps = int((time.monotonic() - self._ramp_t0) / self.ramp_every_s)
            ramped = self.ramp_start + self.ramp_step * steps
            if ramped >= cap:
                self._ramp_t0 = None
            else:
                cap = float(ramped)
        return cap

    def wait_ready(self, timeout_s: float) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: self.ready or self.dead, timeout=max(0.0, timeout_s)) and self.ready

    @contextlib.contextmanager
    def slot(self, deadline: float) -> Iterator[int]:
        """Blocks until the server is ready and under its in-flight limit; yields the epoch at admission."""
        with self._cond:
            while not (self.ready and self.inflight < self.capacity()):
                if self.dead:
                    raise ServerDown(self.dead)
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("LLM server not ready before the request deadline")
                self._cond.wait(timeout=min(1.0, left))  # capacity grows with time during a ramp: poll
            self.inflight += 1
            epoch = self.epoch
        try:
            yield epoch
        finally:
            with self._cond:
                self.inflight -= 1
                self._cond.notify_all()


class LLM:
    def __init__(self, cfg: dict[str, Any], spec: Any = None, gate: ServerGate | None = None) -> None:
        self.base_url = cfg["base_url"].rstrip("/")
        self.model = cfg["model"]
        self.api_key = cfg.get("api_key", "sk-local")
        self.cfg = cfg
        self.spec = spec
        self.gate = gate or ServerGate()
        self.usage = Usage()

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             max_tokens: int | None = None, timeout_s: float | None = None, thinking: bool | None = None) -> Reply:
        """``thinking=False`` turns the chat template's thinking off for this call (compaction summaries).

        ``timeout_s`` is the whole budget for this call, retries and waiting for a restarting server included."""
        cfg = self.cfg
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens or cfg.get("max_tokens", 8192)),
            "temperature": cfg.get("temperature", 1.0),
            "top_p": cfg.get("top_p", 0.95),
        }
        if cfg.get("top_k") is not None:
            body["top_k"] = cfg["top_k"]
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if cfg.get("chat_template_kwargs"):
            body["chat_template_kwargs"] = dict(cfg["chat_template_kwargs"])
        if thinking is not None:
            body["chat_template_kwargs"] = {**body.get("chat_template_kwargs", {}), "enable_thinking": thinking}
        deadline = time.monotonic() + float(timeout_s or cfg.get("request_timeout_s", 900))
        retries = int(cfg.get("retries", 4))
        attempts, last = 0, None
        while True:
            epoch = self.gate.epoch
            try:
                with self.gate.slot(deadline) as epoch:
                    data = self._post("/chat/completions", body, max(1.0, deadline - time.monotonic()))
                return self._parse(data)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                last = RuntimeError(f"HTTP {exc.code}: {detail}")
                low = detail.lower()  # vLLM: "maximum context length"; llama.cpp: "exceeds the available context size"
                if exc.code == 400 and ("maximum context length" in low or "exceed_context_size" in low
                                        or "exceeds the available context" in low):
                    raise ContextOverflow(detail) from exc
                if exc.code < 500 and exc.code != 429:
                    break
            except (TimeoutError, ServerDown) as exc:  # the gate stayed closed until the deadline, or for good
                last = exc
                break
            except Exception as exc:  # noqa: BLE001 - connection reset, timeout, server restart
                last = exc
            if deadline - time.monotonic() <= 1:
                break
            if not self.gate.ready or self.gate.epoch != epoch:
                continue  # the server went down under this request: wait in slot(), do not spend a retry
            attempts += 1
            self.usage.failures += 1
            if attempts >= retries:
                break
            time.sleep(min(60.0, 5.0 * 2 ** (attempts - 1), max(0.0, deadline - time.monotonic() - 1)))
        raise RuntimeError(f"LLM request failed: {last!r}")

    def _post(self, path: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        req = urllib.request.Request(
            self.base_url + path, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _parse(self, data: dict[str, Any]) -> Reply:
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        p, c = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        self.usage.add(p, c)
        calls = list(msg.get("tool_calls") or [])
        if self.spec is not None and calls:
            calls = self.spec.fix_tool_calls(calls)
        return Reply(
            content=msg.get("content") or "",
            reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
            tool_calls=calls,
            finish_reason=choice.get("finish_reason") or "",
            prompt_tokens=p,
            completion_tokens=c,
        )


class ServerDown(RuntimeError):
    """The watchdog gave up on the server: no request will be served again."""


class ContextOverflow(RuntimeError):
    """The prompt no longer fits the served context window: compact and retry."""
