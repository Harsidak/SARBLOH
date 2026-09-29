"""AgentSession: the Prime Agent loop, ported to Python from upstream ``core/agent-session.ts`` (commit 2d24ad4).

What is kept from the paper (arXiv 2608.23552, section 2) and how it maps here:
- L1 active context  -> ``self.messages``. Compaction (``prime.agent.compaction``) replaces the older prefix with a
                        summary and keeps the newest messages verbatim.
- L2 REPL/subagents  -> one persistent ``rlm.repl`` kernel per session (``Kernel``); ``rlm.spawn`` children.
- L3 disk state      -> ``transcript.jsonl`` (full history, survives compaction) and the Continual Harness files
                        (local per game, global per run). Their digest is delivered as a ``[harness-digest]``
                        message on the first turn and on every compaction head; auto /refine edits are announced
                        with an ``[auto-refinement]`` message (``prime.agent.refine``).
- Autonomous mode    -> when the root stops calling tools before the game ends, an ``[autonomous-continuation]``
                        message is sent, bounded by turn, token and wall-clock budgets; the end-condition test is
                        "game won". A threshold compaction is followed by a continuation too, like upstream.
- Accounting         -> tokens, turns, tool calls and child usage are recorded per session.

The system prompt is fixed for the whole session (upstream keeps it stable for the prefix cache): the base prompt
plus the ARC section (``prompts``). Not ported (no use offline on one GPU): the daemon/worker/TUI split, goals,
heartbeats/cron, create_session, MCP, model switching, session recovery after a crash, ``bash()`` completion
follow-ups, the agent-callable ``refine.run()``.
Ours, not upstream: the game status line after each tool result, the "reply was cut off" nudge, the E004
host-forced reflection checkpoint (off by default), and scaling upstream's chars/4 token estimate by the measured
prompt-token ratio (digit-heavy grid text is ~4 tokens per 4 chars, so chars/4 alone never triggered compaction).
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

from prime.agent import compaction, prompts, refine
from prime.game.arc_host import ArcHost
from prime.runtime.kernel import Kernel
from prime.llm.client import LLM, ContextOverflow
from prime.agent.tools import Py

_FENCED = re.compile(r"```(?:python|py|ipython|repl)?[ \t]*\n(.*?)```", re.DOTALL)


def _wire(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Messages as sent to the server: private ``_`` keys dropped, consecutive user messages joined (some chat
    templates, Gemma's among them, require alternating roles)."""
    out: list[dict[str, Any]] = []
    for m in messages:
        m = {k: v for k, v in m.items() if not k.startswith("_")}
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
        self.task = task
        self.arc = arc
        self.deadline = deadline
        self.stop_event = stop_event
        self.session_dir = session_dir = session_dir.resolve()  # the kernel runs in work/: relative paths break
        self.global_harness_dir = global_harness_dir = global_harness_dir.resolve()
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
                      "compaction_failures": 0, "continuations": 0, "llm_failures": 0, "cell_errors": 0,
                      "native_calls": 0, "fenced_calls": 0, "children": 0, "reflections": 0,
                      "reflection_skipped": 0, "refine_reviews": 0, "refines": 0, "refine_edits_applied": 0,
                      "refine_errors": 0}
        self._summary: str | None = None     # the latest compaction summary (updated, not re-summarized, next time)
        self._usage_tokens: int | None = None  # prompt + completion tokens of the last call (upstream usage)
        self._usage_at = 0                   # messages after this index are estimated at chars/4
        # Measured tokens per chars/4 estimate. Upstream assumes chars/4; ARC grids are digit text and these
        # tokenizers give every digit its own token, so a printed grid is ~4x the estimate. Found in E005 smoke.
        self._token_scale = 1.0
        self._compact_failed_at: int | None = None
        self._overflow_retry = False
        self._system: str | None = None
        self._reflect_mark = 0             # action_count at the last reflection checkpoint
        self._reflect_level = 0
        self._reflect_state = ""
        self._reflect_sig: tuple = ()
        self._reflect_asks = 0
        self._turns_since_refine = 0
        self._last_refine_at = 0.0
        self._compact_refine_pending = False
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
            self._log_event({"event": "system_prompt", "content": self._system_prompt()})
            self._append({"role": "user", "_kind": compaction.DIGEST_KIND,
                          "content": refine.digest_block(self._digest())})  # upstream: first-turn digest
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
            if self.arc is not None:
                self.stats["cell_cap_hits"] = self.arc.cell_cap_hits
            self.stats["harness"] = self._harness_counts()
            self._log_event({"event": "end", "reason": self.end_reason, "stats": self.stats})

    def _loop(self) -> None:
        tools = [IPYTHON_TOOL] if self.cfg["tool_mode"] == "native" else None
        while True:
            reason = self.should_stop()
            if reason:
                self.end_reason = reason
                return
            self._drain_inbox() # adding messages from subagents
            window, reserve, _ = self._context_limits()
            tokens = self._context_tokens()
            if compaction.should_compact(tokens, window, reserve, self.cfg["compaction"].get("trigger_tokens")) and (
                    self._compact_failed_at is None or tokens > self._compact_failed_at + 1024):
                self._compact("threshold")
            msgs = [{"role": "system", "content": self._system_prompt()}, *_wire(self.messages)]
            try:
                reply = self.llm.chat(msgs, tools=tools, max_tokens=self.cfg["max_tokens_per_turn"],
                                      timeout_s=max(60.0, min(self.cfg["request_timeout_s"], self.deadline - time.time())))
            except ContextOverflow:
                if self._overflow_retry:  # upstream: one compact-and-retry per overflow
                    self.end_reason = "context_overflow"
                    return
                self._overflow_retry = True
                self._compact("overflow")
                continue
            except Exception as exc:  # noqa: BLE001
                self.stats["llm_failures"] += 1
                self._log_event({"event": "llm_error", "error": repr(exc)})
                time.sleep(10)
                continue
            self._overflow_retry = False
            self.stats["llm_failures"] = 0
            self.stats["turns"] += 1
            self.stats["output_tokens"] += reply.completion_tokens
            self.stats["prompt_tokens_last"] = reply.prompt_tokens
            self._turns_since_refine += 1
            calls = self._calls(reply)
            assistant: dict[str, Any] = {"role": "assistant", "content": reply.content or ""}
            if calls and calls[0]["native"]:
                assistant["tool_calls"] = [c["raw"] for c in calls]
            estimate = (len(msgs[0]["content"]) + 3) // 4 + sum(compaction.estimate_tokens(m) for m in self.messages)
            if reply.prompt_tokens > 0 and estimate > 0:
                self._token_scale = min(4.0, max(0.5, reply.prompt_tokens / estimate))
            self._append(assistant, reasoning=reply.reasoning, usage=(reply.prompt_tokens, reply.completion_tokens),
                         finish=reply.finish_reason, turn=self.stats["turns"])
            self._usage_tokens = reply.prompt_tokens + reply.completion_tokens
            self._usage_at = len(self.messages)
            if calls:
                for call in calls:
                    self._run_call(call, reply)
                self._reflection_tick()
                self._maybe_auto_refine()
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
            self._maybe_auto_refine()
            if self._running_children():
                # Hold the timer-driven continuation while children work; their messages are the wake-up.
                try:
                    msg = self.inbox.get(timeout=self.cfg["subagent_keepalive_s"])
                    self._append({"role": "user", "content": msg})
                    continue
                except queue.Empty:
                    self.stats["continuations"] += 1
                    minutes = max(1, int(self.cfg["subagent_keepalive_s"] // 60))
                    self._append({"role": "user", "content": (
                        "[autonomous-continuation: subagent-keep-alive]\n\nSubagents have been running for at least "
                        f"{minutes} minute{'s' if minutes != 1 else ''} without a reply or exit being delivered. Check "
                        "their status (for example rlm.list_subagents) and cancel or unblock any that are hung; then "
                        "continue working.")})
                    continue
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
            if err and isinstance(args, str):
                # Keep history valid JSON: servers (llama.cpp) re-parse old tool calls and 500 on a broken one.
                args = {"code": "", "unparsed_arguments": args[:2000]}
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

    def _run_call(self, call: dict[str, Any], reply: Any = None) -> None:
        self.stats["tool_calls"] += 1
        failed = bool(call["error"])
        turn = self.stats["turns"]
        call_id = call["raw"]["id"] if call["native"] else f"fenced-t{turn}"
        if call["error"]:
            text = call["error"]
        else:
            # Only the root spends actions: it owns the cell counter and the step ref stamped on each action.
            arc = self.arc if self.depth == 0 else None
            start = arc.game.action_count if arc is not None else 0
            ref = {"session": self.name, "turn": turn, "call": call_id}
            if arc is not None:
                arc.cell_actions = 0
                arc.step_ref = {**ref, "code": call["code"], "say": getattr(reply, "content", None),
                                "thought": getattr(reply, "reasoning", None)}
            timeout = max(10.0, min(self.cfg["cell_timeout_s"], self.deadline - time.time()))
            res = self.kernel.execute(call["code"], timeout_s=timeout)
            if res.status != "ok":
                self.stats["cell_errors"] += 1
                failed = True
            text = res.render(self.cfg["tool_output_chars"])
            spent = arc.game.action_count - start if arc is not None else 0
            if arc is not None:
                arc.step_ref = {**ref, "after_cell": True}  # actions a background task spends after the cell
            self._log_event({"event": "cell", "turn": turn, "call": call_id, "status": res.status,
                             "duration_s": round(res.duration_s, 2),
                             **({"actions": [start, start + spent]} if spent else {})})
            if spent:
                print(f"[{self.name} t{turn}] +{spent} {arc.actions_line(start)} | {arc.status_line()}", flush=True)
        if self.depth == 0 and self.arc is not None:
            text += f"\n[game: {self.arc.status_line()} | {self._time_left()}]"
        if call["native"]:
            msg = {"role": "tool", "tool_call_id": call["raw"]["id"], "content": text}
        else:
            msg = {"role": "user", "content": f"[ipython output]\n{text}"}
        if failed:
            msg["_error"] = True
        self._append(msg)

    # --- host-driven auto /refine (upstream serialized-refine checkpoint between turns) ----------------------
    def _maybe_auto_refine(self) -> None:
        ar = self.cfg.get("auto_refine") or {}
        if self.depth > 0 or not ar.get("enabled"):
            return
        if self._compact_refine_pending and ar.get("compact", True):
            reason = "compact"
        elif self._turns_since_refine >= ar.get("turn_interval", 25):
            reason = "turn_interval"
        else:
            return
        if self._last_refine_at and time.time() - self._last_refine_at < ar.get("cooldown_s", 1200):
            return  # a pending compact trigger is kept for a later boundary, like upstream
        from rlm.harness import _DEFAULT_FILE_NAME

        self._compact_refine_pending = False
        turns, self._turns_since_refine, self._last_refine_at = self._turns_since_refine, 0, time.time()
        self.stats["refine_reviews"] += 1
        t0 = time.time()
        try:
            rec = refine.auto_refine(llm=self.llm, global_file=self.global_harness_dir / _DEFAULT_FILE_NAME,
                                     local_file=self.local_harness_dir / _DEFAULT_FILE_NAME, messages=self.messages,
                                     reason=reason, turns_since=turns, max_tokens=ar.get("max_tokens", 4096),
                                     overflow=ContextOverflow)
        except Exception as exc:  # noqa: BLE001 - refinement must never kill the session
            self.stats["refine_errors"] += 1
            self._log_event({"event": "auto_refine", "reason": reason, "error": f"{type(exc).__name__}: {exc}"[:500]})
            return
        applied = sum(1 for e in rec.get("edits", []) if e.get("applied"))
        self.stats["refines"] += int("edits" in rec)
        self.stats["refine_edits_applied"] += applied
        self._log_event({"event": "auto_refine", "duration_s": round(time.time() - t0, 1), **rec})
        if rec.get("notice"):
            self._append({"role": "user", "content": rec["notice"]})

    # --- forced reflection checkpoints (E004, off by default, not upstream) --------------------------------
    def _harness_sig(self) -> tuple:
        from rlm.harness import _DEFAULT_FILE_NAME

        paths = (self.local_harness_dir / _DEFAULT_FILE_NAME, self.global_harness_dir / _DEFAULT_FILE_NAME)
        return tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in paths)

    def _reflection_tick(self) -> None:
        """Host-driven L3: at checkpoints, refuse arc.step until the model writes to the Continual Harness."""
        every = self.cfg.get("reflect_every_actions")
        if self.depth > 0 or self.arc is None or not every:
            return
        arc = self.arc
        if arc.reflection_due:
            if self._harness_sig() != self._reflect_sig:
                self.stats["reflections"] += 1
                self._log_event({"event": "reflection_done", "reason": arc.reflection_due,
                                 "harness": self._harness_counts()})
                arc.reflection_due = None
            elif self._reflect_asks >= self.cfg.get("reflect_max_reasks", 2):
                self.stats["reflection_skipped"] += 1
                self._log_event({"event": "reflection_skipped", "reason": arc.reflection_due})
                arc.reflection_due = None
            else:
                self._reflect_asks += 1
                self._append({"role": "user", "content": prompts.REFLECT_AGAIN})
            return
        st = arc.game.state
        state = st.engine_state.name
        reason = None
        if st.levels_completed > self._reflect_level:
            reason = f"level {st.levels_completed} completed"
        elif state == "GAME_OVER" and self._reflect_state != "GAME_OVER":
            reason = "game over"
        elif arc.game.action_count - self._reflect_mark >= every:
            reason = f"{arc.game.action_count - self._reflect_mark} actions since the last checkpoint"
        self._reflect_level, self._reflect_state = st.levels_completed, state
        if reason:
            self._reflect_mark = arc.game.action_count
            self._reflect_sig = self._harness_sig()
            self._reflect_asks = 0
            arc.reflection_due = reason
            self._log_event({"event": "reflection_due", "reason": reason})
            self._append({"role": "user", "content": prompts.REFLECT.format(reason=reason, status=self._status())})

    def _harness_counts(self) -> dict[str, Any]:
        from rlm.harness import _DEFAULT_FILE_NAME, HarnessState

        out: dict[str, Any] = {}
        for scope, d in (("local", self.local_harness_dir), ("global", self.global_harness_dir)):
            path = d / _DEFAULT_FILE_NAME
            if not path.exists():
                continue
            try:
                st = HarnessState(path, scope=scope)
                out[scope] = {k: [f"{e.title} v{e.version}" for e in st.list(k)] for k in ("memory", "skill", "prompt", "subagent")}
            except Exception as exc:  # noqa: BLE001
                out[scope] = f"unreadable: {exc}"
        return out

    # --- context: system prompt, harness digest, compaction ----------------------------------------------
    def _system_prompt(self) -> str:
        if self._system is None:
            base = prompts.base_prompt(
                cwd=str(self.kernel.session_dir), transcript=str(self.transcript), depth=self.depth,
                parent=self.parent.name if self.parent else None,
                allow_recursion=self.depth < self.cfg["max_depth"])
            if self.arc is not None:  # upstream appendSystemPrompt
                base += "\n\n" + prompts.arc_section(
                    game_id=self.arc.game.game_id, win_levels=self.arc.game.number_of_levels, depth=self.depth,
                    cell_cap=self.arc.max_actions_per_cell, output_chars=self.cfg["tool_output_chars"])
            self._system = base
        return self._system

    def _digest(self) -> str:
        from rlm.harness import _DEFAULT_FILE_NAME

        try:
            merged, refinements = refine.load_merged(self.global_harness_dir / _DEFAULT_FILE_NAME,
                                                     self.local_harness_dir / _DEFAULT_FILE_NAME)
            return refine.format_digest(merged, refinements, refine.build_query_terms(self.messages))
        except Exception as exc:  # noqa: BLE001
            return f"(harness state unreadable: {type(exc).__name__}: {exc})"

    def _context_limits(self) -> tuple[int, int, int]:
        c = self.cfg["compaction"]
        return int(self.cfg["context_window"]), int(c["reserve_tokens"]), int(c["keep_recent_tokens"])

    def _context_tokens(self) -> int:
        """Upstream estimateContextTokens: last usage + estimate of the messages after it (chars/4, scaled)."""
        est = lambda ms: int(self._token_scale * sum(compaction.estimate_tokens(m) for m in ms))
        if self._usage_tokens is None:
            return est(self.messages)
        return self._usage_tokens + est(self.messages[self._usage_at:])

    def _compact(self, reason: str) -> None:
        _, reserve, keep = self._context_limits()
        tokens_before = self._context_tokens()
        # keep_recent_tokens is in real tokens; the cut-point walk counts chars/4 estimates.
        prep = compaction.prepare(self.messages, int(keep / self._token_scale), self._summary, tokens_before)
        if prep is None:
            self._log_event({"event": "compaction_skipped", "reason": reason, "tokens": tokens_before})
            return
        t0 = time.time()
        try:
            summary = compaction.summarize(self.llm, prep, reserve, ContextOverflow)
        except Exception as exc:  # noqa: BLE001
            self.stats["compaction_failures"] += 1
            self._log_event({"event": "compaction_error", "reason": reason, "error": f"{type(exc).__name__}: {exc}"})
            if reason != "overflow":
                self._compact_failed_at = tokens_before
                return
            summary = "(summary unavailable) Rebuild your understanding from the REPL variables, " \
                      f"`await arc.transitions()` and the conversation log {self.transcript}."
        self._compact_failed_at = None
        self._summary = summary
        kept = [m for m in self.messages[prep.first_kept:] if m.get("_kind") != compaction.DIGEST_KIND]
        head = compaction.head_message(summary, refine.digest_block(self._digest()) + "\n\n")
        self.messages = [head, *kept]
        self._usage_tokens = None
        self.stats["compactions"] += 1
        self._compact_refine_pending = True
        self._log_event({"event": "compaction", "reason": reason, "tokens_before": tokens_before,
                         "token_scale": round(self._token_scale, 2),
                         "summarized_messages": len(prep.to_summarize), "turn_prefix_messages": len(prep.turn_prefix),
                         "kept_messages": len(kept), "split_turn": prep.is_split,
                         "duration_s": round(time.time() - t0, 1)})
        self._log_event({"event": "message", **head})
        if self.depth == 0 and self.arc is not None and (not kept or kept[-1]["role"] != "user"):
            # Upstream queues the autonomous continuation for a threshold compaction in autonomous mode.
            self.stats["continuations"] += 1
            self._append({"role": "user", "content": prompts.CONTINUATION.format(status=self._status())})

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
            task=f"[task from parent]\n\n{prompt}", arc=self.arc,
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
            notice = f"[child-exited: cancelled child:{name}]"
        elif child.session.end_reason.startswith("crash"):
            child.status, child.error = "error", child.session.end_reason
            notice = f"[child-failed child:{name}]\n\n{child.error}"
        else:
            child.status = "completed"
            # Upstream sends an exit notice only when the child never replied; a reply was already delivered.
            notice = None if child.replied else (f"[child-exited: no-reply child:{name}]"
                                                 + (f"\n\nLast assistant text: {child.answer[:2000]}"
                                                    if child.answer else ""))
        if notice and name in self.children:
            self.inbox.put(notice)

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
            self.parent.inbox.put(f"[agent-message from child:{self.name.rsplit('.', 1)[-1]}]\n\n{message}")
            return {"delivered": True}
        child = self._find_child(str(req.get("receiver_name", "")))
        child.session.inbox.put(f"[agent-message from parent:{self.name}]\n\n{message}")
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
