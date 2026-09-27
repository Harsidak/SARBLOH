"""OpenAI-compatible chat client (vLLM on Kaggle, llama.cpp locally). Standard library only: Kaggle is offline."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


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


class LLM:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.base_url = cfg["base_url"].rstrip("/")
        self.model = cfg["model"]
        self.api_key = cfg.get("api_key", "sk-local")
        self.cfg = cfg
        self.usage = Usage()

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             max_tokens: int | None = None, timeout_s: float | None = None) -> Reply:
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
            body["chat_template_kwargs"] = cfg["chat_template_kwargs"]
        timeout = timeout_s or cfg.get("request_timeout_s", 900)
        last: Exception | None = None
        for attempt in range(int(cfg.get("retries", 4))):
            try:
                data = self._post("/chat/completions", body, timeout)
                return self._parse(data)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                last = RuntimeError(f"HTTP {exc.code}: {detail}")
                if exc.code == 400 and "maximum context length" in detail.lower():
                    raise ContextOverflow(detail) from exc
                if exc.code < 500 and exc.code != 429:
                    break
            except Exception as exc:  # noqa: BLE001 - connection reset, timeout, server restart
                last = exc
            self.usage.failures += 1
            time.sleep(min(60.0, 5.0 * 2 ** attempt))
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
        return Reply(
            content=msg.get("content") or "",
            reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
            tool_calls=list(msg.get("tool_calls") or []),
            finish_reason=choice.get("finish_reason") or "",
            prompt_tokens=p,
            completion_tokens=c,
        )


class ContextOverflow(RuntimeError):
    """The prompt no longer fits the served context window: compact and retry."""
