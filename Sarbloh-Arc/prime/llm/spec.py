"""What a model brings to the shared serving layer. ``gemma.py`` and ``qwen.py`` each define one ``SPEC``; config key
``model`` ("gemma" | "qwen") picks it. Nothing outside those two files may name a model, parser or sampling value."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable


def _no_fix(args: dict[str, Any]) -> dict[str, Any]:
    return args


@dataclass(frozen=True)
class ModelSpec:
    name: str
    served_model_name: str
    # Profiles are tried in order until one starts and passes the tool-call smoke test. Each is
    # {"model_dataset": "<owner/slug>", "flags": {"--flag": "value" | None}, "env": {...}, "runtime": "<name>"?}.
    # "{model_dir}" in a flag value becomes the resolved weights directory.
    profiles: dict[str, dict[str, Any]]
    profile_chain: list[str]
    # Client defaults: sampling and chat_template_kwargs (the thinking policy). Config "llm" overrides win.
    llm: dict[str, Any]
    # Extra --no-deps wheels installed over the shared wheelhouse (a Kaggle dataset ref), or None.
    overlay_dataset: str | None = None
    # Repairs parser quirks in tool-call arguments before the agent sees them.
    fix_tool_arguments: Callable[[dict[str, Any]], dict[str, Any]] = field(default=_no_fix)
    # Sent with the tool-call smoke test (thinking can exhaust max_tokens before the call).
    smoke_template_kwargs: dict[str, Any] = field(default_factory=dict)
    # Prebuilt vLLM runtimes, for models the shared wheelhouse cannot serve. A profile with "runtime": "<name>" skips the
    # wheelhouse install; runtimes[name](working_dir, find_input) prepares the runtime once per process and returns
    # {"env": {...} (the server's whole environment), "serve": [python args before the model dir], "info": {...}}.
    runtimes: dict[str, Callable[..., dict[str, Any]]] = field(default_factory=dict)

    def max_model_len(self, profile: str) -> int:
        return int(self.profiles[profile]["flags"]["--max-model-len"])

    def has_tool_parser(self, profile: str) -> bool:
        return "--tool-call-parser" in self.profiles[profile]["flags"]

    def fix_tool_calls(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Applies ``fix_tool_arguments`` to each call's JSON arguments; arguments that do not parse are left alone
        (the agent reports those to the model)."""
        out = []
        for tc in calls:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            try:
                parsed = json.loads(args) if isinstance(args, str) else args
            except ValueError:
                out.append(tc)
                continue
            if isinstance(parsed, dict):
                fixed = self.fix_tool_arguments(parsed)
                if fixed != parsed:
                    tc = {**tc, "function": {**fn, "arguments": json.dumps(fixed)}}
            out.append(tc)
        return out


MODELS = ("gemma", "qwen")


def get_spec(name: str) -> ModelSpec:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose one of {MODELS}")
    return importlib.import_module(f"prime.llm.{name}").SPEC
