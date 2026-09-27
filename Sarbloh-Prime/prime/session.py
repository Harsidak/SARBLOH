"""AgentSession: the Prime Agent loop, ported to Python from upstream ``core/agent-session.ts`` (commit 2d24ad4).

What is kept from the paper (arXiv 2608.23552, section 2) and how it maps here:
- L1 active context  -> ``self.messages``; compaction replaces the prefix with a model-written summary.
- L2 REPL/subagents  -> one persistent ``rlm.repl`` kernel per session (``Kernel``); ``rlm.spawn`` children.
- L3 disk state      -> ``transcript.jsonl`` (full history, survives compaction) and the Continual Harness files
                        (local per game, global per run), injected into every turn's system prompt.
- Autonomous mode    -> when the model stops calling tools before the game ends, a continuation prompt is sent,
                        bounded by turn, token and wall-clock budgets. The end-condition test is "game won".
- Accounting         -> tokens, turns, tool calls and child usage are recorded per session.

Not ported (no use offline on one GPU): the daemon/worker/TUI split, heartbeats/cron, create_session, MCP, model
switching, session recovery after a crash.
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prime import prompts
from prime.arc_host import ArcHost
from prime.kernel import Kernel
from prime.llm import LLM, ContextOverflow

IPYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "ipython",
        "description": "Execute Python code in the persistent IPython REPL. State persists across calls. "
                       "Top-level await works. Returns stdout, the repr of the last expression, and errors.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to execute."}},
            "required": ["code"],
        },
    },
}
_FENCED = re.compile(r"```(?:python|py|ipython|repl)?[ \t]*\n(.*?)```", re.DOTALL)


def _merge_users(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join consecutive user messages: some chat templates (Gemma's among them) require alternating roles."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if out and m["role"] == "user" and out[-1]["role"] == "user":
            out[-1] = {"role": "user", "content": f"{out[-1]['content']}\n\n{m['content']}"}
        else:
            out.append(m)
    return out


@dataclass
class Child:
    child_id: str
    name: str
    session: AgentSession
    thread: threading.Thread
    started: float = field(default_factory=time.time)
    status: str = "running"          # running | completed | error | cancelled
    answer: str | None = None
    error: str | None = None
    progress: str | None = None
    replied: bool = False
    ended: float | None = None

    def row(self) -> dict[str, Any]:
        s = self.session
        return {
            "rlm_child_id": self.child_id, "active_session_id": self.child_id, "session_id": self.child_id,
            "session_name": self.name, "session_dir": str(s.session_dir),
            "status": "running" if self.status == "running" else ("completed" if self.status == "completed" else "error"),
            "tool_use_count": s.stats["tool_calls"],
            "duration_ms": int(((self.ended or time.time()) - self.started) * 1000),
            "answer_preview": (self.answer or "")[:500] or None,
            "replied_since_task": self.replied, "progress_note": self.progress,
        }

    def result(self) -> dict[str, Any]:
        status = {"running": "running", "completed": "done", "error": "error", "cancelled": "cancelled"}[self.status]
        return {
            "rlm_child_id": self.child_id, "session_name": self.name, "session_dir": str(self.session.session_dir),
            "status": status, "settled": self.status != "running",
            "answer_preview": (self.answer or "")[:2000] or None, "error": self.error,
            "duration_ms": int(((self.ended or time.time()) - self.started) * 1000),
            "tool_use_count": self.session.stats["tool_calls"], "replied_since_task": self.replied,
        }


class AgentSession:
    def __init__(self, *, cfg: dict[str, Any], llm: LLM, name: str, session_dir: Path, task: str,
                 arc: ArcHost | None, deadline: float, stop_event: threading.Event, global_harness_dir: Path,
                 depth: int = 0, parent: AgentSession | None = None) -> None:
        self.cfg = cfg
        self.llm = llm
        self.name = name
        self.session_dir = session_dir
        self.task = task
        self.arc = arc
        self.deadline = deadline
        self.stop_event = stop_event
        self.global_harness_dir = global_harness_dir
        self.local_harness_dir = session_dir / "harness"
        self.depth = depth
        self.parent = parent
        self.inbox: queue.Queue[str] = queue.Queue()
        self.children: dict[str, Child] = {}
        self.messages: list[dict[str, Any]] = []
        self.transcript = session_dir / "transcript.jsonl"
        self.final_answer: str | None = None
        self.end_reason = ""
        self.stats = {"turns": 0, "tool_calls": 0, "output_tokens": 0, "prompt_tokens_last": 0, "compactions": 0,
                      "continuations": 0, "llm_failures": 0, "cell_errors": 0, "native_calls": 0,
                      "fenced_calls": 0, "children": 0}
        self._harness_cache: tuple[tuple, str] | None = None
        self.kernel = Kernel(session_dir / "work", self._host, env={
            "RLM_HARNESS_STATE_DIR": str(self.local_harness_dir),
            "RLM_GLOBAL_HARNESS_STATE_DIR": str(global_harness_dir),
        })

    # --- budget ------------------------------------------------------------------------------------------
    def tokens_spent(self) -> int:
        return self.stats["output_tokens"] + sum(c.session.tokens_spent() for c in self.children.values())

    def should_stop(self) -> str | None:
        if self.stop_event.is_set():
            return "stopped"
        if time.time() >= self.deadline:
            return "wall_clock"
        if self.depth == 0 and self.arc is not None and self.arc.finished:
            return "game_finished"
        if self.depth == 0 and self.arc is not None and self.arc.budget_left == 0:
            return "action_budget"
        lim = self.cfg["limits"] if self.depth == 0 else self.cfg["child_limits"]
        if self.stats["turns"] >= lim["max_turns"]:
            return "max_turns"
        if self.stats["output_tokens"] >= lim["max_output_tokens"]:
            return "max_output_tokens"
        if self.stats["llm_failures"] >= self.cfg["max_consecutive_llm_failures"]:
            return "llm_failures"
        return None

    # --- main loop ---------------------------------------------------------------------------------------
    def run(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.local_harness_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.kernel.start()
            self._append({"role": "user", "content": self.task})
            self._loop()
        except Exception as exc:  # noqa: BLE001 - a session crash must not take the run down
            self.end_reason = f"crash: {type(exc).__name__}: {exc}"
            self._log_event({"event": "crash", "traceback": traceback.format_exc()})
        finally:
            for child in list(self.children.values()):
                child.session.stop_event.set()
            for child in list(self.children.values()):
                child.thread.join(timeout=30)
            self.kernel.close()
            self._log_event({"event": "end", "reason": self.end_reason, "stats": self.stats})

    def _loop(self) -> None:
        tools = [IPYTHON_TOOL] if self.cfg["tool_mode"] == "native" else None
        while True:
            reason = self.should_stop()
            if reason:
                self.end_reason = reason
                return
            self._drain_inbox()
            if self.stats["prompt_tokens_last"] >= self.cfg["compact_at_tokens"]:
                self._compact()
            msgs = [{"role": "system", "content": self._system_prompt()}, *_merge_users(self.messages)]
            try:
                reply = self.llm.chat(msgs, tools=tools, max_tokens=self.cfg["max_tokens_per_turn"],
                                      timeout_s=max(60.0, min(self.cfg["request_timeout_s"], self.deadline - time.time())))
            except ContextOverflow:
                self._compact(force=True)
                continue
            except Exception as exc:  # noqa: BLE001
                self.stats["llm_failures"] += 1
                self._log_event({"event": "llm_error", "error": repr(exc)})
                time.sleep(10)
                continue
            self.stats["llm_failures"] = 0
            self.stats["turns"] += 1
            self.stats["output_tokens"] += reply.completion_tokens
            self.stats["prompt_tokens_last"] = reply.prompt_tokens
            calls = self._calls(reply)
            assistant: dict[str, Any] = {"role": "assistant", "content": reply.content or ""}
            if calls and calls[0]["native"]:
                assistant["tool_calls"] = [c["raw"] for c in calls]
            self._append(assistant, reasoning=reply.reasoning, usage=(reply.prompt_tokens, reply.completion_tokens),
                         finish=reply.finish_reason)
            if calls:
                for call in calls:
                    self._run_call(call)
                continue
            if reply.finish_reason == "length":
                self._append({"role": "user", "content": "Your reply was cut off by the output limit. Be shorter: "
                                                         "put the work in one `ipython` call."})
                continue
            # No tool call: the model ended its turn.
            if self.depth > 0:
                self.final_answer = reply.content.strip()
                self.end_reason = "answered"
                return
            if self.arc is None or self.arc.finished:
                self.end_reason = "answered"
                return
            if self._running_children():
                # Hold the timer-driven continuation while children work; their messages are the wake-up.
                try:
                    msg = self.inbox.get(timeout=self.cfg["subagent_keepalive_s"])
                    self._append({"role": "user", "content": msg})
                    continue
                except queue.Empty:
                    pass
            self.stats["continuations"] += 1
            self._append({"role": "user", "content": prompts.CONTINUATION.format(status=self._status())})

    # --- tool calls --------------------------------------------------------------------------------------
    def _calls(self, reply: Any) -> list[dict[str, Any]]:
        calls = []
        for tc in reply.tool_calls:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            code, err = None, None
            if fn.get("name") != "ipython":
                err = f"Unknown tool {fn.get('name')!r}. The only tool is `ipython`."
            else:
                try:
                    parsed = json.loads(args) if isinstance(args, str) else (args or {})
                    code = parsed.get("code")
                    if not isinstance(code, str):
                        err = "ipython needs a string argument `code`."
                except ValueError as exc:
                    err = f"Could not parse tool arguments as JSON: {exc}"
            raw = {"id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                   "function": {"name": fn.get("name") or "ipython",
                                "arguments": args if isinstance(args, str) else json.dumps(args or {})}}
            calls.append({"native": True, "raw": raw, "code": code, "error": err})
        if calls:
            self.stats["native_calls"] += len(calls)
            return calls
        # Fallback: code in fenced blocks (models or servers without working tool-call parsing).
        blocks = [b for b in _FENCED.findall(reply.content or "") if b.strip()]
        if blocks and self.cfg["allow_fenced_code"]:
            self.stats["fenced_calls"] += 1
            return [{"native": False, "raw": None, "code": "\n\n".join(blocks), "error": None}]
        return []

    def _run_call(self, call: dict[str, Any]) -> None:
        self.stats["tool_calls"] += 1
        if call["error"]:
            text = call["error"]
        else:
            timeout = max(10.0, min(self.cfg["cell_timeout_s"], self.deadline - time.time()))
            res = self.kernel.execute(call["code"], timeout_s=timeout)
            if res.status != "ok":
                self.stats["cell_errors"] += 1
            text = res.render(self.cfg["tool_output_chars"])
            self._log_event({"event": "cell", "status": res.status, "duration_s": round(res.duration_s, 2)})
        if self.depth == 0 and self.arc is not None:
            text += f"\n[game: {self.arc.status_line()} | {self._time_left()}]"
        if call["native"]:
            self._append({"role": "tool", "tool_call_id": call["raw"]["id"], "content": text})
        else:
            self._append({"role": "user", "content": f"[ipython output]\n{text}"})

    # --- context: system prompt, harness digest, compaction ----------------------------------------------
    def _system_prompt(self) -> str:
        base = prompts.base_prompt(
            cwd=str(self.kernel.session_dir), transcript=str(self.transcript), depth=self.depth,
            parent=self.parent.name if self.parent else None,
            allow_recursion=self.depth < self.cfg["max_depth"])
        digest = self._harness_digest()
        return f"{base}\n\n# Continual harness\n\n{digest}" if digest else base

    def _harness_digest(self) -> str:
        from rlm.harness import _DEFAULT_FILE_NAME, HarnessState

        files = [(scope, d / _DEFAULT_FILE_NAME) for scope, d in
                 (("global", self.global_harness_dir), ("local", self.local_harness_dir))]
        key = tuple((str(p), p.stat().st_mtime_ns if p.exists() else 0) for _, p in files)
        if self._harness_cache and self._harness_cache[0] == key:
            return self._harness_cache[1]
        parts = []
        for scope, path in files:
            if not path.exists():
                continue
            try:
                st = HarnessState(path, scope=scope)
                notes = st.list("prompt")
                if notes:
                    parts.append(f"## Prompt notes ({scope})")
                    parts += [f"- {n.title}: {n.content.strip()}" for n in notes[:30]]
                parts.append(st.overview(max_entries_per_kind=30).split("\n", 2)[-1])
            except Exception as exc:  # noqa: BLE001
                parts.append(f"({scope} harness state unreadable: {exc})")
        text = "\n".join(parts)
        cap = self.cfg["harness_digest_chars"]
        if len(text) > cap:
            text = text[:cap] + "\n... (truncated; call rlm.harness.overview() or .search(...) for the rest)"
        self._harness_cache = (key, text)
        return text

    def _compact(self, force: bool = False) -> None:
        self.stats["compactions"] += 1
        msgs = [{"role": "system", "content": self._system_prompt()},
                *_merge_users([*self.messages, {"role": "user", "content": prompts.COMPACTION}])]
        summary = None
        for _ in range(3):
            try:
                reply = self.llm.chat(msgs, tools=None, max_tokens=self.cfg["compaction_max_tokens"])
                summary = reply.content.strip()
                if summary:
                    break
            except ContextOverflow:
                # Drop the oldest half of the middle and try again.
                middle = msgs[2:-1]
                msgs = [msgs[0], msgs[1], *middle[len(middle) // 2:], msgs[-1]]
            except Exception as exc:  # noqa: BLE001
                self._log_event({"event": "compaction_error", "error": repr(exc)})
                break
        if not summary:
            summary = "(summary unavailable) Rebuild your understanding from the REPL variables and the transcript."
        handoff = (f"[compaction #{self.stats['compactions']}] Earlier messages were replaced by this summary. The "
                   f"Python REPL state is intact. The full history is in {self.transcript}.\n\n{summary}")
        if self.depth == 0 and self.arc is not None:
            handoff += f"\n\n[game: {self.arc.status_line()} | {self._time_left()}]"
        self.messages = [self.messages[0]]
        self._append({"role": "user", "content": handoff})
        self.stats["prompt_tokens_last"] = 0
        self._log_event({"event": "compaction", "forced": force})

    # --- subagents and messages (host requests) ----------------------------------------------------------
    def _host(self, req: dict[str, Any]) -> dict[str, Any]:
        kind = str(req.get("type", ""))
        if kind.startswith("arc."):
            if self.arc is None:
                raise RuntimeError("no game is attached to this session")
            return self.arc.handle(req, self.depth)
        if kind == "rlm.run":
            return self._spawn(req)
        if kind == "rlm.list_subagents":
            return {"subagents": [c.row() for c in self.children.values()]}
        if kind == "rlm.collect":
            return self._collect(req)
        if kind == "rlm.progress.note":
            if self.parent is not None:
                for c in self.parent.children.values():
                    if c.session is self:
                        c.progress = str(req.get("message", ""))[:512]
            return {"accepted": True}
        if kind == "rlm.delete_subagent":
            child = self._find_child(str(req.get("target", "")))
            child.session.stop_event.set()
            row = child.row()
            self.children.pop(child.name, None)
            return {"subagent": row}
        if kind == "agent_message.send":
            return self._send_message(req)
        raise RuntimeError(f"host request {kind!r} is not supported by this harness")

    def _spawn(self, req: dict[str, Any]) -> dict[str, Any]:
        if self.depth >= self.cfg["max_depth"]:
            raise RuntimeError("maximum recursion depth reached; do the work yourself")
        prompt = str(req.get("prompt", ""))
        name = str((req.get("kwargs") or {}).get("name", "")).strip()
        if not prompt.strip() or not re.fullmatch(r"[A-Za-z0-9_.-]{1,48}", name or ""):
            raise ValueError("rlm.spawn needs a prompt and a name of letters, digits, '_', '-' or '.'")
        if name in self.children:
            raise ValueError(f"a child named {name!r} already exists; pick another name or delete it")
        if len(self._running_children()) >= self.cfg["max_running_children"]:
            raise RuntimeError(f"at most {self.cfg['max_running_children']} children may run at once")
        child_id = f"{self.name}.{name}"
        lim = self.cfg["child_limits"]
        session = AgentSession(
            cfg=self.cfg, llm=self.llm, name=child_id, session_dir=self.session_dir / "children" / name,
            task=f"[task from parent]\n{prompt}", arc=self.arc,
            deadline=min(self.deadline, time.time() + lim["wall_s"]), stop_event=threading.Event(),
            global_harness_dir=self.global_harness_dir, depth=self.depth + 1, parent=self)
        thread = threading.Thread(target=self._run_child, args=(name,), daemon=True, name=f"child-{child_id}")
        self.children[name] = Child(child_id=child_id, name=name, session=session, thread=thread)
        self.stats["children"] += 1
        thread.start()
        self._log_event({"event": "spawn", "child": name, "prompt": prompt[:2000]})
        return {"rlm_child_id": child_id, "name": name, "session_dir": str(session.session_dir),
                "model": self.llm.model}

    def _run_child(self, name: str) -> None:
        child = self.children[name]
        if self.stop_event.is_set():
            child.session.stop_event.set()
        child.session.run()
        child.ended = time.time()
        child.answer = child.session.final_answer
        if child.session.stop_event.is_set():
            child.status = "cancelled"
        elif child.session.end_reason.startswith("crash"):
            child.status, child.error = "error", child.session.end_reason
        else:
            child.status = "completed"
        if name in self.children:
            self.inbox.put(f"[message from child {name}] finished ({child.session.end_reason}). Final answer:\n"
                           f"{child.answer or '(none)'}")

    def _collect(self, req: dict[str, Any]) -> dict[str, Any]:
        targets = [self._find_child(t) for t in (req.get("targets") or [])] or list(self.children.values())
        deadline = time.time() + max(0, int(req.get("timeout_ms", 0))) / 1000
        while time.time() < deadline and any(c.status == "running" for c in targets):
            time.sleep(0.5)
        return {"results": [c.result() for c in targets]}

    def _send_message(self, req: dict[str, Any]) -> dict[str, Any]:
        message = str(req.get("message", ""))
        if req.get("receiver_role") == "parent":
            if self.parent is None:
                raise RuntimeError("this is the root agent; it has no parent")
            for c in self.parent.children.values():
                if c.session is self:
                    c.replied = True
            self.parent.inbox.put(f"[message from child {self.name.rsplit('.', 1)[-1]}]\n{message}")
            return {"delivered": True}
        child = self._find_child(str(req.get("receiver_name", "")))
        child.session.inbox.put(f"[message from parent]\n{message}")
        return {"delivered": True}

    def _find_child(self, selector: str) -> Child:
        for c in self.children.values():
            if selector in (c.name, c.child_id):
                return c
        raise KeyError(f"no child {selector!r}; children: {sorted(self.children)}")

    def _running_children(self) -> list[Child]:
        return [c for c in self.children.values() if c.status == "running"]

    def _drain_inbox(self) -> None:
        while True:
            try:
                msg = self.inbox.get_nowait()
            except queue.Empty:
                return
            self._append({"role": "user", "content": msg})

    # --- bookkeeping -------------------------------------------------------------------------------------
    def _time_left(self) -> str:
        return f"{max(0, int(self.deadline - time.time())) // 60} min left"

    def _status(self) -> str:
        return f"{self.arc.status_line()}, {self._time_left()}" if self.arc else self._time_left()

    def _append(self, msg: dict[str, Any], **extra: Any) -> None:
        self.messages.append(msg)
        self._log_event({"event": "message", **msg, **{k: v for k, v in extra.items() if v}})

    def _log_event(self, obj: dict[str, Any]) -> None:
        obj = {"t": round(time.time(), 2), **obj}
        try:
            with self.transcript.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(obj, default=str) + "\n")
        except Exception:  # noqa: BLE001
            pass
