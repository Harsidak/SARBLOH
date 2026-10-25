"""DeepSeek API adapter for local trace collection (config ``local_SFT_RUN``). Not used on Kaggle: it needs internet.

The agent's requests are built for our own vLLM/SGLang servers; this turns them into requests the DeepSeek API accepts:
- ``chat_template_kwargs`` and ``top_k`` are dropped (the API has no chat template knobs).
- Pictures are replaced by a short text note (the DeepSeek chat models take text only; the ASCII board stays).
- ``reasoning`` is dropped; ``reasoning_content`` (the thinking sent back in a tool-call chain) is kept unless
  ``send_reasoning`` is off. If the API rejects it, it is turned off for the rest of the run.
The reply's thinking comes back in ``reasoning_content``, which the shared client already reads, so the transcript and
the SFT export get it like any other run.

The key is read from $DEEPSEEK_API_KEY, else from the repo's gitignored .env.
"""

from __future__ import annotations

import io
import os
import urllib.error
from pathlib import Path
from typing import Any

from harness.llm.client import LLM

DEFAULTS: dict[str, Any] = {
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-reasoner",   # thinking model; "deepseek-chat" for no thinking
    "send_reasoning": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "request_timeout_s": 900.0,
    "retries": 6,
}


def api_key(repo_root: Path) -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    env = repo_root / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "DEEPSEEK_API_KEY":
                key = value.strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set (environment or repo .env)")
    return key


def _text_only(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    parts = [p.get("text", "") if p.get("type") == "text" else "[picture not sent: text-only model]"
             for p in content if isinstance(p, dict)]
    return "\n".join(p for p in parts if p)


class DeepSeekLLM(LLM):
    def __init__(self, cfg: dict[str, Any], repo_root: Path) -> None:
        super().__init__({**DEFAULTS, **cfg, "api_key": api_key(repo_root)})
        self.send_reasoning = bool(self.cfg.get("send_reasoning", True))

    def _post(self, path: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        try:
            return super()._post(path, self._adapt(body), timeout)
        except urllib.error.HTTPError as exc:
            if exc.code != 400 or not self.send_reasoning:
                raise
            detail = exc.read().decode("utf-8", errors="replace")
            if "reasoning_content" not in detail:
                raise urllib.error.HTTPError(exc.url, exc.code, exc.msg, exc.headers,
                                             io.BytesIO(detail.encode("utf-8"))) from exc  # body already read
            print("[deepseek] API rejected reasoning_content in the input; no longer sending it", flush=True)
            self.send_reasoning = False
            return super()._post(path, self._adapt(body), timeout)

    def _adapt(self, body: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in body.items() if k not in ("chat_template_kwargs", "top_k")}
        msgs = []
        for m in body["messages"]:
            m = {k: v for k, v in m.items() if k != "reasoning"}
            if not self.send_reasoning or m.get("role") != "assistant":
                m.pop("reasoning_content", None)
            m["content"] = _text_only(m.get("content"))
            msgs.append(m)
        out["messages"] = msgs
        return out
