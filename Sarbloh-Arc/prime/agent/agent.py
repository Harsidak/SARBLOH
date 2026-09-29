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

E006 ``toolset: "dedicated"`` (``prime.agent.tools``): besides ``ipython`` the model gets ``plan``, ``act``,
``reset_level``, ``remember``, ``recall``, ``delegate`` and ``message``, handled in ``_run_tool``. The REPL can no
longer spend actions. Each act step is checked against the agent's registered world model (``wm`` in the REPL) and
every tool call is logged as a structured transcript event for ``prime.trace``.
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
from prime.agent.Planner import Planner
from prime.agent.tools import toolset
from prime.game.arc_host import ArcHost
from prime.runtime.kernel import Kernel
from prime.llm.client import LLM, ContextOverflow

_FENCED = re.compile(r"```(?:python|py|ipython|repl)?[ \t]*\n(.*?)```", re.DOTALL)
# Host hooks run in the agent's REPL (E006): read the registered world model and check one act step against it.
# The marker separates the hook's JSON line from any background output the kernel flushes with it.
_WM_MARK = "@@prime-wm@@"
_WM_SYNC = f"print('{_WM_MARK}' + __import__('json').dumps(__import__('worldmodel')._host_sync()))"
_WM_CHECK = f"print('{_WM_MARK}' + __import__('json').dumps(await __import__('worldmodel')._host_check_at({{i}})))"


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
                      "refine_errors": 0, "tool_counts": {}, "tool_errors": 0, "act_actions": 0, "wm_checks": 0,
                      "wm_mispredictions": 0}
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
        # E006: fenced mode has no native tools, so it keeps the one-tool REPL interface.
        self.toolset = cfg.get("toolset", "ipython") if cfg["tool_mode"] == "native" else "ipython"
        self.tools = toolset(self.toolset, depth=depth, max_depth=cfg["max_depth"],
                             act_max=int(cfg.get("act_max_actions", 5))) if cfg["tool_mode"] == "native" else None
        self.planner = Planner(int(cfg.get("replan_every_actions", 15)))
        self._wm: dict[str, Any] = {"model": None, "certified": False}   # the REPL world model, as last synced
        self._acted_in_reply = False
        # The game's memories: `remember` writes the root's harness, and children `recall` from it too.
        self.game_harness_dir = parent.game_harness_dir if parent is not None else self.local_harness_dir
        if depth == 0 and arc is not None:
            arc.on_step = lambda ev: self._log_event(ev)
            if self.toolset == "dedicated":
                arc.repl_actions = False

    # --- budget ------------------------------------------------------------------------------------------
    def tokens_spent(self) -> int:
        # calculates the total cumulative output tokens generated across
        # this agent session and all of its descendants (subagents) recursively.
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
            self._log_event({"event": "system_prompt", "content": self._system_prompt(), "tools": self.tools,
                             "toolset": self.toolset})
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
            if self.depth == 0:
                self.stats["plan_versions"] = len(self.planner.versions)
            self._log_event({"event": "end", "reason": self.end_reason, "stats": self.stats})

    def _loop(self) -> None:
        tools = self.tools
        while True:
            # 1) any reason to stop?
            reason = self.should_stop()
            if reason:
                self.end_reason = reason
                return

            # 2) get the messages from the sub agents
            self._drain_inbox() # adding messages from subagents

            # 3) Compaction (if needed)
            window, reserve, _ = self._context_limits()
            tokens = self._context_tokens()
            if compaction.should_compact(tokens, window, reserve, self.cfg["compaction"].get("trigger_tokens")) and (
                    self._compact_failed_at is None or tokens > self._compact_failed_at + 1024):
                self._compact("threshold")
            msgs = [{"role": "system", "content": self._system_prompt()}, *_wire(self.messages)]

            # 4) send the current message to LLM and handles other errors like emergency compaction, llm failures time out
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

            # 5) Parse reply for tool calls
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

            # 6) If tool calls: execute each, run reflection tick, maybe auto-refine
            if calls:
                self._acted_in_reply = False   # one act/reset per reply: the model must read each result first
                for call in calls:
                    self._run_call(call, reply)
                self._reflection_tick()
                self._maybe_auto_refine()
                continue

                # in case the reply curs off
            if reply.finish_reason == "length":
                self._append({"role": "user", "content": "Your reply was cut off by the output limit. Be shorter: "
                                                         "put the work in one `ipython` call."})
                continue
            # 7) if No tool call: the model ended its turn.
            if self.depth > 0:
                self.final_answer = reply.content.strip()
                self.end_reason = "answered"
                return
            if self.arc is None or self.arc.finished:
                self.end_reason = "answered"
                return
            self._maybe_auto_refine()
            if self._running_children():
                # Hold the timer-driven continuation while children work; their messages are the wake-up. Polled, so
                # a child that replied and then exited (no exit notice follows a reply) does not hold the root for
                # the whole keep-alive.
                msg, until = None, time.time() + self.cfg["subagent_keepalive_s"]
                while msg is None and time.time() < until and self._running_children() and not self.should_stop():
                    try:
                        msg = self.inbox.get(timeout=1.0)
                    except queue.Empty:
                        pass
                if msg is not None:
                    self._append({"role": "user", "content": msg})
                    continue
                if self._running_children() and not self.should_stop():
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
        names = [t["function"]["name"] for t in (self.tools or [])] or ["ipython"]
        for tc in reply.tool_calls:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            name = fn.get("name") or "ipython"
            code, err, parsed = None, None, {}
            if name not in names:
                err = (f"Unknown tool {name!r}. The only tool is `ipython`." if names == ["ipython"]
                       else f"Unknown tool {name!r}. Tools: {', '.join(names)}.")
            else:
                try:
                    parsed = json.loads(args) if isinstance(args, str) else (args or {})
                    if not isinstance(parsed, dict):
                        raise ValueError("arguments must be a JSON object")
                    if name == "ipython":
                        code = parsed.get("code")
                        if not isinstance(code, str):
                            err = "ipython needs a string argument `code`."
                except ValueError as exc:
                    err = f"Could not parse tool arguments as JSON: {exc}"
            if err and isinstance(args, str):
                # Keep history valid JSON: servers (llama.cpp) re-parse old tool calls and 500 on a broken one.
                try:
                    keep = isinstance(json.loads(args), dict)
                except ValueError:
                    keep = False
                if not keep:
                    args = {"code": "", "unparsed_arguments": args[:2000]}
            raw = {"id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                   "function": {"name": name,
                                "arguments": args if isinstance(args, str) else json.dumps(args or {})}}
            calls.append({"native": True, "raw": raw, "name": name, "args": parsed, "code": code, "error": err})
        if calls:
            self.stats["native_calls"] += len(calls)
            return calls
        # Fallback: code in fenced blocks (models or servers without working tool-call parsing).
        blocks = [b for b in _FENCED.findall(reply.content or "") if b.strip()]
        if blocks and self.cfg["allow_fenced_code"]:
            self.stats["fenced_calls"] += 1
            return [{"native": False, "raw": None, "name": "ipython", "args": {}, "code": "\n\n".join(blocks),
                     "error": None}]
        return []

    def _run_call(self, call: dict[str, Any], reply: Any = None) -> None:
        self.stats["tool_calls"] += 1
        counts = self.stats["tool_counts"]
        counts[call["name"]] = counts.get(call["name"], 0) + 1
        failed = bool(call["error"])
        turn = self.stats["turns"]
        call_id = call["raw"]["id"] if call["native"] else f"fenced-t{turn}"
        if call["error"]:
            text = call["error"]
            self.stats["tool_errors"] += 1
            self._log_event({"event": "tool_error", "turn": turn, "call": call_id, "tool": call["name"],
                             "error": call["error"][:500]})
        elif call["name"] in ("act", "reset_level") and self._acted_in_reply:
            failed = True
            self.stats["tool_errors"] += 1
            text = (f"{call['name']} refused: one act or reset_level per reply. Nothing was spent. Read the result of "
                    "the previous one, then act in your next reply.")
            self._log_event({"event": "tool_error", "turn": turn, "call": call_id, "tool": call["name"],
                             "error": "second act/reset in one reply", "args": call["args"]})
        elif call["name"] != "ipython":
            before = self.arc.game.action_count if self.arc is not None else 0
            try:
                if call["name"] in ("act", "reset_level"):
                    self._acted_in_reply = True
                text = self._run_tool(call["name"], call["args"], call_id, turn, reply)
            except Exception as exc:  # noqa: BLE001 - a bad argument is the model's to fix, not a crash
                failed = True
                self.stats["tool_errors"] += 1
                text = f"{call['name']} refused: {exc}"
                self._log_event({"event": "tool_error", "turn": turn, "call": call_id, "tool": call["name"],
                                 "error": str(exc)[:500], "args": call["args"]})
                if self.arc is None or self.arc.game.action_count == before:
                    self._acted_in_reply = False   # a refused act spent nothing: a corrected retry may follow
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
                             "duration_s": round(res.duration_s, 2), "output_chars": len(text),
                             **({"actions": [start, start + spent]} if spent else {})})
            if spent:
                print(f"[{self.name} t{turn}] +{spent} {arc.actions_line(start)} | {arc.status_line()}", flush=True)
            if self.depth == 0 and self.toolset == "dedicated" and ("wm" in call["code"] or "worldmodel" in call["code"]):
                self._wm_sync()
        if self.depth == 0 and self.arc is not None:
            text += f"\n[game: {self.arc.status_line()} | {self._status_extra()}{self._time_left()}]"
        if call["native"]:
            msg = {"role": "tool", "tool_call_id": call["raw"]["id"], "content": text}
        else:
            msg = {"role": "user", "content": f"[ipython output]\n{text}"}
        if failed:
            msg["_error"] = True
        self._append(msg)

    # --- E006 dedicated tools -----------------------------------------------------------------------------
    def _run_tool(self, name: str, args: dict[str, Any], call_id: str, turn: int, reply: Any) -> str:
        if name in ("act", "reset_level", "plan") and (self.depth > 0 or self.arc is None):
            raise PermissionError("only the root agent plays the game")
        if name == "act":
            return self._tool_act(args, call_id, turn, reply)
        if name == "reset_level":
            return self._tool_reset(args, call_id, turn, reply)
        if name == "plan":
            return self._tool_plan(args, call_id, turn)
        if name == "remember":
            return self._tool_remember(args, call_id, turn)
        if name == "recall":
            return self._tool_recall(args, call_id, turn)
        if name == "delegate":
            return self._tool_delegate(args, call_id, turn)
        if name == "message":
            return self._tool_message(args, call_id, turn)
        raise ValueError(f"unknown tool {name!r}")

    @staticmethod
    def _parse_action(item: Any) -> tuple[int, int | None, int | None]:
        """"1", 1, "A1", "ACTION1" -> (1, None, None); "6 12 40", "A6(12,40)", {"action": 6, "x": 12, "y": 40}."""
        if isinstance(item, dict):
            aid = item.get("action", item.get("id"))
            nums = [aid, item.get("x"), item.get("y")]
            nums = [int(n) for n in nums if n is not None]
        elif isinstance(item, (list, tuple)):
            nums = [int(n) for n in item]
        else:
            nums = [int(n) for n in re.findall(r"-?\d+", str(item))]
        if not nums:
            raise ValueError(f"cannot read action {item!r}: use an id like \"1\", or \"6 x y\" for a click")
        if nums[0] == 6:
            if len(nums) != 3:
                raise ValueError(f"action {item!r}: a click needs x and y, as \"6 x y\" (x = column, y = row)")
            return 6, nums[1], nums[2]
        if len(nums) != 1:
            raise ValueError(f"action {item!r}: one action per list item (only action 6 takes x y)")
        if nums[0] == 0:
            raise ValueError("0 is RESET: use the reset_level tool with a reason")
        return nums[0], None, None

    def _wm_run(self, code: str) -> dict[str, Any] | None:
        """Run a host hook in the REPL and read its JSON line; None if it fails (the agent's REPL state wins)."""
        try:
            res = self.kernel.execute(code, timeout_s=20)
        except Exception:  # noqa: BLE001
            return None
        for line in reversed((res.stdout or "").split("\n")):
            if line.startswith(_WM_MARK):
                try:
                    return json.loads(line[len(_WM_MARK):])
                except ValueError:
                    break
        self._log_event({"event": "wm_hook_error", "status": res.status, "output": (res.stdout or "")[-500:],
                         "error": (res.error or "")[-500:]})
        return None

    def _wm_sync(self) -> None:
        got = self._wm_run(_WM_SYNC)
        if got is None:
            return
        self._wm = {"model": got.get("model"), "certified": bool(got.get("certified"))}
        for ev in got.get("events") or []:
            self._log_event({"event": "wm", **ev, "action_count": self.arc.game.action_count if self.arc else None,
                             "turn": self.stats["turns"]})

    def _status_extra(self) -> str:
        if self.depth > 0 or self.toolset != "dedicated":
            return ""
        wm = (f"wm {self._wm['model']}{' certified' if self._wm['certified'] else ' not certified'} | "
              if self._wm["model"] else "")
        return f"{self.planner.status()} | {wm}"

    def _step_ref(self, call_id: str, turn: int, reply: Any, args: dict[str, Any]) -> None:
        self.arc.cell_actions = 0
        self.arc.step_ref = {"session": self.name, "turn": turn, "call": call_id, "code": json.dumps(args),
                             "say": getattr(reply, "content", None), "thought": getattr(reply, "reasoning", None)}

    def _tool_act(self, args: dict[str, Any], call_id: str, turn: int, reply: Any) -> str:
        arc = self.arc
        raw = args.get("actions")
        if isinstance(raw, (str, int)):
            raw = [raw] if not isinstance(raw, str) or not raw.strip().startswith("[") else json.loads(raw)
        if not isinstance(raw, list) or not raw:
            raise ValueError('actions must be a non-empty list, e.g. ["1", "4"] or ["6 12 40"]')
        cap = int(self.cfg.get("act_max_actions", 5))
        if len(raw) > cap:
            raise ValueError(f"at most {cap} actions per act call, you sent {len(raw)}. Send the first ones, read "
                             "the result, then continue.")
        expect = str(args.get("expect") or "").strip()
        if not expect:
            raise ValueError("expect is required: one sentence with what you predict these actions will do")
        parsed = [self._parse_action(a) for a in raw]   # all or nothing: nothing runs if one item is bad
        self._wm_sync()
        model = self._wm["model"]
        start_actions, start_level = arc.game.action_count, arc.game.state.levels_completed
        self._step_ref(call_id, turn, reply, args)
        lines, steps, stop = [], [], None
        halt = bool(self.cfg.get("act_halt_on_mispredict", True))
        for n, (a, x, y) in enumerate(parsed):
            label = f"A{a}" + (f"({x},{y})" if x is not None else "")
            try:
                obs = arc.handle({"type": "arc.step", "action": a, **({"x": x, "y": y} if x is not None else {})},
                                 0, source="tool")
            except Exception as exc:  # noqa: BLE001
                stop = f"action {n + 1} ({label}) refused: {exc}"
                break
            st = dict(arc.last_step or {})
            line = f"#{st.get('i')} {label}: {st.get('change')}"
            if st.get("repeat_of") is not None:
                line += f" [repeat: same state and action as #{st['repeat_of']}]"
            check = None
            if model:
                check = self._wm_run(_WM_CHECK.format(i=int(st["i"])))
                if check and check.get("match") is not None:
                    self.stats["wm_checks"] += 1
                    if check["match"]:
                        line += " | model: correct"
                    else:
                        self.stats["wm_mispredictions"] += 1
                        self._wm["certified"] = False   # the hook also voids the REPL's last check
                        line += f" | model WRONG: {check.get('detail')}"
            lines.append(line)
            steps.append({k: st.get(k) for k in ("i", "action", "x", "y", "level", "level_after", "state", "level_up",
                                                 "changed", "change", "repeat_of", "state_key")} | {"wm": check})
            if obs.get("level_up"):
                stop = "level up"
                break
            if obs.get("state") == "GAME_OVER":
                stop = "GAME_OVER"
                break
            if arc.finished:
                stop = "game finished"
                break
            if check and check.get("match") is False and halt and n + 1 < len(parsed):
                stop = f"your world model {model} predicted #{st.get('i')} wrong; fix it before acting on it"
                break
        arc.step_ref = {"session": self.name, "turn": turn, "call": call_id, "after_cell": True}
        done = len(steps)
        self.stats["act_actions"] += done
        s = arc.game.state
        head = f"act: {done} of {len(parsed)} actions done" + (f"; stopped: {stop}" if stop else "") + \
               f". You expected: {expect[:200]}"
        out = [head, *lines]
        if s.levels_completed > start_level:
            out.append(f"LEVEL UP: {s.levels_completed} of {arc.game.number_of_levels} levels done. The layout "
                       "changed: look at the new grid in ipython before acting. Save what you learned (remember).")
        if s.engine_state.name == "GAME_OVER":
            out.append("GAME_OVER: the level is lost. Find out why in the transitions, then call reset_level.")
        legal = [a for a in s.available_actions if a != 0]
        out.append(f"now: level {s.levels_completed}/{arc.game.number_of_levels}, {s.engine_state.name}, legal {legal}, "
                   f"{arc.game.action_count} actions spent, {arc.budget_left if arc.max_actions else 'unlimited'} left."
                   + (f" Grids: `ts = await arc.transitions({steps[0]['i']})`." if steps else ""))
        note = self.planner.stale_note(action_count=arc.game.action_count, level=s.levels_completed,
                                       game_over=s.engine_state.name == "GAME_OVER") if self.toolset == "dedicated" \
            else None
        if note:
            out.append(note)
        self._log_event({"event": "act", "turn": turn, "call": call_id, "expect": expect,
                         "was_right": args.get("was_right"), "requested": len(parsed), "done": done, "stop": stop,
                         "steps": steps, "actions": [start_actions, arc.game.action_count], "wm_model": model,
                         "wm_certified": self._wm["certified"], "planner_note": note})
        if done == 0 and stop:
            raise RuntimeError(stop)
        return "\n".join(out)

    def _tool_reset(self, args: dict[str, Any], call_id: str, turn: int, reply: Any) -> str:
        reason = str(args.get("reason") or "").strip()
        if not reason:
            raise ValueError("reason is required: say why restarting beats continuing from here")
        arc = self.arc
        was = arc.game.state.engine_state.name
        before = arc.game.action_count
        self._step_ref(call_id, turn, reply, args)
        try:
            arc.handle({"type": "arc.reset"}, 0, source="tool")
        finally:
            arc.step_ref = {"session": self.name, "turn": turn, "call": call_id, "after_cell": True}
        self.stats["act_actions"] += 1
        s = arc.game.state
        self._log_event({"event": "reset", "turn": turn, "call": call_id, "reason": reason, "state_before": was,
                         "level": s.levels_completed, "i": (arc.last_step or {}).get("i"),
                         "action_count": arc.game.action_count, "level_actions_lost": before - arc._level_start})
        return (f"level {s.levels_completed} restarted (transition #{(arc.last_step or {}).get('i')}); state "
                f"{s.engine_state.name}; {arc.game.action_count} actions spent. Your REPL state, memories and plan are "
                "kept.")

    def _tool_plan(self, args: dict[str, Any], call_id: str, turn: int) -> str:
        self._wm_sync()
        text, event = self.planner.update(args, turn=turn, action_count=self.arc.game.action_count,
                                          level=self.arc.game.state.levels_completed,
                                          certified=self._wm["certified"])
        self._log_event({**event, "call": call_id})
        return text

    def _harness(self, all_games: bool = False) -> Any:
        from rlm.harness import _DEFAULT_FILE_NAME, HarnessState

        d = self.global_harness_dir if all_games else self.game_harness_dir
        d.mkdir(parents=True, exist_ok=True)
        return HarnessState(d / _DEFAULT_FILE_NAME, scope="global" if all_games else "local")

    _KINDS = ("fact", "hypothesis", "refuted", "goal", "procedure")

    def _tool_remember(self, args: dict[str, Any], call_id: str, turn: int) -> str:
        from rlm.harness import _slug

        kind = str(args.get("kind") or "").strip().lower()
        if kind not in self._KINDS:
            raise ValueError(f"kind must be one of {list(self._KINDS)}")
        title, content = str(args.get("title") or "").strip(), str(args.get("content") or "").strip()
        if not title or not content:
            raise ValueError("title and content are required")
        n_trans = len(self.arc.transitions) if self.arc is not None else 0
        ev = args.get("evidence") or []
        if not isinstance(ev, list):
            ev = [ev]
        evidence = sorted({int(e) for e in ev})
        bad = [e for e in evidence if not 0 <= e < n_trans]
        if bad:
            raise ValueError(f"evidence {bad} out of range: transitions are #0..#{n_trans - 1}")
        if kind in ("fact", "refuted") and not evidence:
            raise ValueError(f"a {kind} needs evidence: the transition indices that show it (from act results). "
                             "Without evidence, save it as a hypothesis.")
        all_games = bool(args.get("all_games"))
        st = self._harness(all_games)
        mid = _slug(title, "memory")
        old = st.get("memory", mid)
        # update() edits the entry in place: keep what it was before for the log and the reply
        prev_kind, prev_content, prev_version = (old.path, old.content, old.version) if old else (None, None, None)
        level = self.arc.game.state.levels_completed if self.arc is not None else None
        acount = self.arc.game.action_count if self.arc is not None else None
        meta = {"kind": kind, "evidence": evidence, "turn": turn, "action_count": acount, "level": level,
                "session": self.name}
        if old is None:
            entry = st.create("memory", title, content, id=mid, path=kind, metadata=meta)
        else:
            entry = st.update("memory", mid, title, content, path=kind, metadata={**(old.metadata or {}), **meta})
        self._log_event({"event": "remember", "turn": turn, "call": call_id, "kind": kind, "title": title,
                         "id": entry.id, "version": entry.version, "created": old is None,
                         "prev_kind": prev_kind, "prev_content": (prev_content or "")[:500] or None,
                         "content": content[:1000],
                         "evidence": evidence, "action_count": acount, "level": level,
                         "scope": "global" if all_games else "local"})
        verb = "saved" if old is None else f"updated (was {prev_kind} v{prev_version})"
        return f"memory '{entry.id}' {verb} as {kind} v{entry.version}" + \
            (f", evidence {', '.join(f'#{e}' for e in evidence)}" if evidence else "")

    def _tool_recall(self, args: dict[str, Any], call_id: str, turn: int) -> str:
        query = str(args.get("query") or "").strip()
        local, glob = self._harness(False), self._harness(True)
        if not query or query == "*":
            hits = local.list("memory")
            head = f"{len(hits)} memories in this game" + (f" (+{len(glob.list('memory'))} general lessons: search "
                                                           "for them by words)" if glob.list("memory") else "")
        else:
            hits = local.search(query, kind="memory", limit=8) + glob.search(query, kind="memory", limit=4)
            if not hits:
                # The harness search drops words under 3 letters, but the model asks for "A1", "#12", "6".
                words = [w for w in re.split(r"[\s,;]+", query.lower()) if w]
                hits = [e for e in local.list("memory") + glob.list("memory")
                        if any(w in f"{e.title} {e.content}".lower() for w in words)][:12]
            head = f"{len(hits)} memories match {query!r}"
        lines = [head]
        for e in hits[:20]:
            ev = (e.metadata or {}).get("evidence") or []
            lines.append(f"- [{e.path} v{e.version}{' general' if e.scope == 'global' else ''}] {e.title}: "
                         f"{e.content[:400]}" + (f" (evidence {', '.join(f'#{i}' for i in ev[:8])})" if ev else ""))
        self._log_event({"event": "recall", "turn": turn, "call": call_id, "query": query,
                         "hits": [e.id for e in hits[:20]]})
        return "\n".join(lines)

    def _tool_delegate(self, args: dict[str, Any], call_id: str, turn: int) -> str:
        name, task = str(args.get("name") or "").strip(), str(args.get("task") or "").strip()
        if not task:
            raise ValueError("task is required")
        prompt = (f"{task}\n\nYou have your own REPL with read-only game access: `await arc.observe()`, "
                  "`await arc.transitions()`, `arc.show`, `arc.diff`, `wm`. You cannot spend actions. When you are "
                  "done, send your answer with the message tool (to=\"parent\"), then stop.")
        res = self._spawn({"prompt": prompt, "kwargs": {"name": name}})
        self._log_event({"event": "delegate", "turn": turn, "call": call_id, "child": name, "task": task[:2000]})
        return (f"child '{res['name']}' started. Its answer arrives later as a message; continue your own work. "
                f"Running children: {len(self._running_children())}.")

    def _tool_message(self, args: dict[str, Any], call_id: str, turn: int) -> str:
        to, text = str(args.get("to") or "").strip(), str(args.get("text") or "").strip()
        if not to or not text:
            raise ValueError("to and text are required")
        req = ({"receiver_role": "parent", "message": text} if to == "parent"
               else {"receiver_role": "child", "receiver_name": to, "message": text})
        self._send_message(req)
        self._log_event({"event": "message_sent", "turn": turn, "call": call_id, "to": to, "text": text[:2000]})
        return f"message delivered to {to}"

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
                allow_recursion=self.depth < self.cfg["max_depth"], toolset=self.toolset)
            if self.arc is not None:  # upstream appendSystemPrompt
                base += "\n\n" + prompts.arc_section(
                    game_id=self.arc.game.game_id, win_levels=self.arc.game.number_of_levels, depth=self.depth,
                    cell_cap=self.arc.max_actions_per_cell, output_chars=self.cfg["tool_output_chars"],
                    toolset=self.toolset, act_max=int(self.cfg.get("act_max_actions", 5)))
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
        plan = self.planner.render() if self.depth == 0 else ""
        head = compaction.head_message(summary, refine.digest_block(self._digest()) + "\n\n"
                                       + (plan + "\n\n" if plan else ""))
        self.messages = [head, *kept]
        self._usage_tokens = None
        self.stats["compactions"] += 1
        self._compact_refine_pending = True
        self._log_event({"event": "compaction", "reason": reason, "tokens_before": tokens_before,
                         "token_scale": round(self._token_scale, 2),
                         "summarized_messages": len(prep.to_summarize), "turn_prefix_messages": len(prep.turn_prefix),
                         "kept_messages": len(kept), "split_turn": prep.is_split, "summary_chars": len(summary),
                         "action_count": self.arc.game.action_count if self.arc is not None else None,
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
