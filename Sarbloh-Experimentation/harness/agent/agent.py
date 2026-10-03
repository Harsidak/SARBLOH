"""AgentSession: the Prime Agent loop, ported to Python from upstream ``core/agent-session.ts`` (commit 2d24ad4).

What is kept from the paper (arXiv 2608.23552, section 2) and how it maps here:
- L1 active context  -> ``self.messages``. Compaction (``harness.agent.compaction``) replaces the older prefix with a
                        summary and keeps the newest messages verbatim.
- L2 REPL            -> one persistent ``rlm.repl`` kernel per session (``Kernel``).
- L3 disk state      -> ``transcript.jsonl`` (full history, survives compaction) and the Continual Harness files
                        (local per game, global per run). Their digest is delivered as a ``[harness-digest]``
                        message on the first turn and on every compaction head; auto /refine edits are announced
                        with an ``[auto-refinement]`` message (``harness.agent.refine``).
- Autonomous mode    -> when the root stops calling tools before the game ends, an ``[autonomous-continuation]``
                        message is sent, bounded by turn, token and wall-clock budgets; the end-condition test is
                        "game won". A threshold compaction is followed by a continuation too, like upstream.
- Accounting         -> tokens, turns and tool calls are recorded per session.

The system prompt is fixed for the whole session (upstream keeps it stable for the prefix cache): the base prompt
plus the ARC section (``prompts``). Not ported (no use offline on one GPU): the daemon/worker/TUI split, goals,
heartbeats/cron, create_session, MCP, model switching, session recovery after a crash, ``bash()`` completion
follow-ups, the agent-callable ``refine.run()``, ``rlm.spawn`` subagents and agent messages (E008 never enabled them;
git history before this change has them).
Ours, not upstream: the game status line after each tool result, the "reply was cut off" nudge, the E004
host-forced reflection checkpoint (off by default), and scaling upstream's chars/4 token estimate by the measured
prompt-token ratio (digit-heavy grid text is ~4 tokens per 4 chars, so chars/4 alone never triggered compaction).

E008 ``toolset: "e008"`` (``harness.agent.tools``): ``ipython`` (think and compute; reads the state with ``observe()``,
cannot act or fetch the game), ``act`` (1 to N actions plus the agent's plan, hypotheses, findings and goal) and
``recall`` (search memory and skills). After every act the host pushes the new state as the next user message: a
change line per action in the act result, then the short view (changed region) and the X8 picture
(``harness.agent.perception``, E116). The full view (X8 briefing, every object, the whole board, the picture) is pushed
at a level start, after a compaction, and when the agent calls ``observe()`` in ipython (once per state). The agent's
memory (``harness.memory``) is rendered into one pinned message before the recent
turns on every turn (``harness.agent.context``); the hidden curator (``harness.agent.curator``) replaces auto-refine. The
host records what the agent writes and judges nothing. Every tool call is logged as a structured transcript event for
``harness.trace``. E006's ``dedicated`` toolset (plan, reset_level, remember, delegate, message, world-model checks)
was removed for E008 (git history: commit f117ca4).
"""

from __future__ import annotations

import json
import re
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from harness.agent import compaction, context, curator, perception, prompts, refine
from harness.agent.tools import toolset
from harness.game.arc_host import ArcHost
from harness.llm.client import LLM, ContextOverflow
from harness.memory.lifecycle import GameMemory
from harness.runtime.kernel import Kernel
from harness.runtime.skills import observation

_FENCED = re.compile(r"```(?:python|py|ipython|repl)?[ \t]*\n(.*?)```", re.DOTALL)
_ACTION_NAMES = {0: "RESET", 1: "A1(up)", 2: "A2(down)", 3: "A3(left)", 4: "A4(right)", 5: "A5(space)",
                 7: "A7(undo)"}


def _wire(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Messages as sent to the server: private ``_`` keys dropped, consecutive user messages joined (some chat
    templates, Gemma's among them, require alternating roles)."""
    return context.merge_users(messages)


class AgentSession:
    def __init__(self, *, cfg: dict[str, Any], llm: LLM, name: str, session_dir: Path, task: str,
                 arc: ArcHost | None, deadline: float, stop_event: threading.Event, global_harness_dir: Path,
                 memory_root: Path | None = None) -> None:
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
        self.messages: list[dict[str, Any]] = []
        self.transcript = session_dir / "transcript.jsonl"
        self.end_reason = ""
        self.stats = {"turns": 0, "tool_calls": 0, "output_tokens": 0, "prompt_tokens_last": 0, "compactions": 0,
                      "compaction_failures": 0, "continuations": 0, "llm_failures": 0, "cell_errors": 0,
                      "native_calls": 0, "fenced_calls": 0, "reflections": 0,
                      "reflection_skipped": 0, "refine_reviews": 0, "refines": 0, "refine_edits_applied": 0,
                      "refine_errors": 0, "tool_counts": {}, "tool_errors": 0, "act_actions": 0, "act_calls": 0,
                      "act_arg_errors": 0, "recalls": 0, "hypothesis_events": 0, "promotions": 0,
                      "curator_runs": 0, "curator_errors": 0, "observations": 0, "images_sent": 0,
                      "observation_chars_max": 0, "pinned_chars_max": 0, "observe_calls": 0, "observe_pushes": 0,
                      "perception_errors": 0, "perception_s_total": 0.0, "perception_s_max": 0.0}
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
        # Fenced mode has no native tools, so it keeps the one-tool REPL interface; E008 needs a game.
        self.toolset = cfg.get("toolset", "ipython") if cfg["tool_mode"] == "native" else "ipython"
        if self.toolset == "e008" and arc is None:
            self.toolset = "ipython"
        self.e008 = self.toolset == "e008"
        self.kernel = Kernel(session_dir / "work", self._host, env={
            "RLM_HARNESS_STATE_DIR": str(self.local_harness_dir),
            "RLM_GLOBAL_HARNESS_STATE_DIR": str(global_harness_dir),
            "PRIME_TOOLSET": self.toolset,
        })
        self.tools = toolset(self.toolset, act_max=int(cfg.get("act_max_actions", 5))) if cfg["tool_mode"] == "native" else None
        self._acted_in_reply = False
        # E008: perception state, the game's memory, and what is pushed after the current reply's tool results.
        self.perception = dict(cfg.get("perception") or {})
        self.vision = bool(cfg.get("vision")) and bool(self.perception.get("image", True))
        self.tracker = perception.Tracker()
        self._objects: list[Any] = []            # objects of the newest frame, ids stable within the level
        self._obs_rows: list[list[int]] | None = None   # the grid of the last observation pushed (crop baseline)
        self._level_start = True
        self._last_step: int | None = None
        self._last_change: str | None = None     # the newest change line, "#12 A1(up): ..." (shown and in observe())
        self._brief: tuple[Any, Any] = (None, None)   # (state key, Briefing): the X8 analysis of the newest frame
        self._pending_obs: dict[str, Any] | None = None
        self._curate_pending: str | None = None
        self._pinned_chars = 0
        self.memory: GameMemory | None = None
        if self.e008:
            self.memory = GameMemory(memory_root or (session_dir / "memory"), arc.game.game_id,
                                     cfg.get("memory") or {})
        if arc is not None:
            arc.on_step = lambda ev: self._log_event(ev)
            if self.e008:
                arc.repl_actions = False

    # --- budget ------------------------------------------------------------------------------------------
    def tokens_spent(self) -> int:
        return self.stats["output_tokens"]

    def should_stop(self) -> str | None:
        if self.stop_event.is_set():
            return "stopped"
        if time.time() >= self.deadline:
            return "wall_clock"
        if self.arc is not None and self.arc.finished:
            return "game_finished"
        if self.arc is not None and self.arc.budget_left == 0:
            return "action_budget"
        lim = self.cfg["limits"]
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
                             "toolset": self.toolset, "vision": self.vision})
            if self.e008:
                # E008: no harness digest (its memory is the pinned block); the first state comes with the task.
                self._log_event({"event": "memory_init", **self.memory.on_new_game()})
                self._append({"role": "user", "content": self.task})
                self._append(self._observation())
            else:
                self._append({"role": "user", "_kind": compaction.DIGEST_KIND,
                              "content": refine.digest_block(self._digest())})  # upstream: first-turn digest
                self._append({"role": "user", "content": self.task})
            self._loop()
        except Exception as exc:  # noqa: BLE001 - a session crash must not take the run down
            self.end_reason = f"crash: {type(exc).__name__}: {exc}"
            self._log_event({"event": "crash", "traceback": traceback.format_exc()})
        finally:
            self.kernel.close()
            if self.arc is not None:
                self.stats["cell_cap_hits"] = self.arc.cell_cap_hits
            self.stats["harness"] = self._harness_counts()
            if self.memory is not None:
                self.memory.save()
                self.stats["lessons"] = self.memory.lessons.counts() if self.memory.lessons is not None else None
                self.stats["skills_loaded"] = self.memory.skills.loaded_for(self.memory.game)
            self._log_event({"event": "end", "reason": self.end_reason, "stats": self.stats})

    def _loop(self) -> None:
        tools = self.tools
        while True:
            # 1) any reason to stop?
            reason = self.should_stop()
            if reason:
                self.end_reason = reason
                return

            # 2) Compaction (if needed)
            window, reserve, _ = self._context_limits()
            tokens = self._context_tokens()
            if compaction.should_compact(tokens, window, reserve, self.cfg["compaction"].get("trigger_tokens")) and (
                    self._compact_failed_at is None or tokens > self._compact_failed_at + 1024):
                self._compact("threshold")
            if self.e008:
                pinned = context.pinned_message(self.memory.blocks(self._objects), self._game_status())
                self._pinned_chars = len(pinned["content"])
                self.stats["pinned_chars_max"] = max(self.stats["pinned_chars_max"], self._pinned_chars)
                msgs = context.build(self._system_prompt(), self.messages, pinned, vision=self.vision)
            else:
                msgs = [{"role": "system", "content": self._system_prompt()}, *_wire(self.messages)]

            # 3) send the current message to LLM and handles other errors like emergency compaction, llm failures time out
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

            # 4) Parse reply for tool calls
            calls = self._calls(reply)
            assistant: dict[str, Any] = {"role": "assistant", "content": reply.content or ""}
            if calls and calls[0]["native"]:
                assistant["tool_calls"] = [c["raw"] for c in calls]
            estimate = (len(msgs[0]["content"]) + self._pinned_chars + 3) // 4 + \
                sum(compaction.estimate_tokens(m) for m in self.messages)
            if reply.prompt_tokens > 0 and estimate > 0:
                self._token_scale = min(4.0, max(0.5, reply.prompt_tokens / estimate))
            self._append(assistant, reasoning=reply.reasoning, usage=(reply.prompt_tokens, reply.completion_tokens),
                         finish=reply.finish_reason, turn=self.stats["turns"])
            self._usage_tokens = reply.prompt_tokens + reply.completion_tokens
            self._usage_at = len(self.messages)

            # 5) If tool calls: execute each, run reflection tick, maybe auto-refine
            if calls:
                self._acted_in_reply = False   # one act per reply: the model must read each result first
                for call in calls:
                    self._run_call(call, reply)
                if self._pending_obs is not None:
                    # E008: the new state goes in right after this reply's tool results, before the next turn.
                    self._append(self._pending_obs)
                    self._pending_obs = None
                self._reflection_tick()
                self._maybe_auto_refine()
                continue

            # 6) the reply was cut off
            if reply.finish_reason == "length":
                self._append({"role": "user", "content": prompts.event_message("cut_off", self._toolset)})
                continue
            # 7) if No tool call: the model ended its turn.
            if self.arc is None or self.arc.finished:
                self.end_reason = "answered"
                return
            self._maybe_auto_refine()
            self.stats["continuations"] += 1
            self._append({"role": "user", "content": self._continuation()})

    def _continuation(self) -> str:
        return prompts.continuation(status=self._status(), toolset=self._toolset, **self._left())

    @property
    def _toolset(self) -> str:
        return "game" if self.e008 else "ipython"

    def _left(self) -> dict[str, int | None]:
        """Moves and minutes left, for the low-budget and low-time notes."""
        return {"moves_left": self.arc.budget_left if self.arc is not None else None,
                "minutes_left": max(0, int(self.deadline - time.time())) // 60}

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
            if call["name"] == "act":
                self.stats["act_arg_errors"] += 1
            self._log_event({"event": "tool_error", "turn": turn, "call": call_id, "tool": call["name"],
                             "error": call["error"][:500]})
        elif call["name"] == "act" and self._acted_in_reply:
            failed = True
            self.stats["tool_errors"] += 1
            text = "act refused: " + prompts.event_message("act_twice")
            self._log_event({"event": "tool_error", "turn": turn, "call": call_id, "tool": call["name"],
                             "error": "second act in one reply", "args": call["args"]})
        elif call["name"] != "ipython":
            before = self.arc.game.action_count if self.arc is not None else 0
            try:
                if call["name"] == "act":
                    self._acted_in_reply = True
                text = self._run_tool(call["name"], call["args"], call_id, turn, reply)
            except Exception as exc:  # noqa: BLE001 - a bad argument is the model's to fix, not a crash
                failed = True
                self.stats["tool_errors"] += 1
                if call["name"] == "act" and isinstance(exc, (ValueError, TypeError)):
                    self.stats["act_arg_errors"] += 1
                text = f"{call['name']} refused: {exc}"
                self._log_event({"event": "tool_error", "turn": turn, "call": call_id, "tool": call["name"],
                                 "error": str(exc)[:500], "args": call["args"]})
                if self.arc is None or self.arc.game.action_count == before:
                    self._acted_in_reply = False   # a refused act spent nothing: a corrected retry may follow
        else:
            arc = self.arc
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
            if self.e008 and any(observation.MIME in d for d in res.displays):
                text += "\n" + self._observe_called()
            spent = arc.game.action_count - start if arc is not None else 0
            if arc is not None:
                arc.step_ref = {**ref, "after_cell": True}  # actions a background task spends after the cell
            self._log_event({"event": "cell", "turn": turn, "call": call_id, "status": res.status,
                             "duration_s": round(res.duration_s, 2), "output_chars": len(text),
                             **({"actions": [start, start + spent]} if spent else {})})
            if spent:
                print(f"[{self.name} t{turn}] +{spent} {arc.actions_line(start)} | {arc.status_line()}", flush=True)
        if self.arc is not None:
            text += f"\n[game: {self.arc.status_line()} | {self._time_left()}]"
        if call["native"]:
            msg = {"role": "tool", "tool_call_id": call["raw"]["id"], "content": text}
        else:
            msg = {"role": "user", "content": f"[ipython output]\n{text}"}
        if failed:
            msg["_error"] = True
        self._append(msg)

    # --- E008 tools: act and recall ------------------------------------------------------------------------
    def _run_tool(self, name: str, args: dict[str, Any], call_id: str, turn: int, reply: Any) -> str:
        if not self.e008:
            raise ValueError(f"unknown tool {name!r}")
        if name == "act":
            return self._tool_act(args, call_id, turn, reply)
        if name == "recall":
            return self._tool_recall(args, call_id, turn)
        raise ValueError(f"unknown tool {name!r}")

    @staticmethod
    def _parse_action(item: Any) -> tuple[int, int | None, int | None]:
        """(action id, row, column). "1", 1, "A1", "ACTION1" -> (1, None, None); "6 12 40", "A6(12,40)",
        {"action": 6, "r": 12, "c": 40} -> (6, 12, 40) (row 12, column 40); "reset" -> (0, None, None)."""
        if isinstance(item, dict):
            aid = item.get("action", item.get("id"))
            if str(aid).strip().lower() == "reset":
                return 0, None, None
            r, c = item.get("r", item.get("row")), item.get("c", item.get("col", item.get("column")))
            nums = [int(n) for n in (aid, r, c) if n is not None]
        elif isinstance(item, (list, tuple)):
            nums = [int(n) for n in item]
        else:
            text = str(item).strip()
            if text.lower() in ("reset", "r", "restart", "action0", "a0"):
                return 0, None, None
            nums = [int(n) for n in re.findall(r"-?\d+", text)]
        if not nums:
            raise ValueError(f"cannot read action {item!r}: use \"1\" to \"7\", \"6 r c\" for a click, or \"reset\"")
        if nums[0] == 6:
            if len(nums) != 3:
                raise ValueError(f"action {item!r}: a click needs a row and a column, as \"6 r c\"")
            return 6, nums[1], nums[2]
        if len(nums) != 1:
            raise ValueError(f"action {item!r}: one action per list item (only action 6 takes r c)")
        if nums[0] == 0:
            return 0, None, None
        if not 1 <= nums[0] <= 7:
            raise ValueError(f"action {item!r}: ids are 1 to 7, or \"reset\"")
        return nums[0], None, None

    @staticmethod
    def _label(a: int, r: int | None, c: int | None) -> str:
        return f"A6(r{r},c{c})" if a == 6 else _ACTION_NAMES.get(a, f"A{a}")

    def _step_ref(self, call_id: str, turn: int, reply: Any, args: dict[str, Any]) -> None:
        self.arc.cell_actions = 0
        self.arc.step_ref = {"session": self.name, "turn": turn, "call": call_id, "code": json.dumps(args),
                             "say": getattr(reply, "content", None), "thought": getattr(reply, "reasoning", None)}

    def _frame(self) -> list[list[int]]:
        f = self.arc.game.state.raw.frame[-1]
        return perception.rows_of(f)

    def _game_status(self) -> str:
        arc = self.arc
        s = arc.game.state
        legal = " ".join(str(a) for a in s.available_actions if a != 0)
        left = arc.budget_left if arc.max_actions else "unlimited"
        return (f"game {arc.game.game_id} | level {s.levels_completed + 1} of {arc.game.number_of_levels} "
                f"({s.levels_completed} done) | {s.engine_state.name} | legal actions {legal} | "
                f"{arc.game.action_count} actions spent ({arc.game.action_count - arc._level_start} this level), "
                f"{left} left | {self._time_left()}")

    def _observation(self, full: bool = False) -> dict[str, Any]:
        """The newest state as a user message, with the X8 picture when vision is on. The full view (X8 briefing,
        every object, the whole board) at a level start or when ``full``; else the short view (the changed region).
        The full view and the data are also published for ``observe()`` in the REPL."""
        t0 = time.time()
        rows = self._frame()
        if not self._objects:   # the first look at the game: ids start at 1 (act re-tracks at every level-up)
            self.tracker.reset()
            self._objects, _ = self.tracker.observe(rows)
        full = full or self._level_start or self._obs_rows is None
        p = self.perception
        b = self._briefing(rows)
        sc = perception.Scene(self._last_step, rows, self._objects, change=self._last_change, prev=self._obs_rows,
                              status=self._game_status(), briefing=b, briefing_lines=int(p.get("briefing_lines", 30)))
        opts = {"max_objects": int(p.get("max_objects", 40)), "show_ascii": bool(p.get("ascii", True)),
                "segmentation": bool(p.get("segmentation", True))}
        full_text = sc.text(full=True, **opts)
        text = full_text if full else sc.text(budget_tokens=int(p.get("observation_tokens", 800)),
                                              margin=int(p.get("crop_margin", 3)), **opts)
        try:
            observation.write(self.kernel.session_dir, {**sc.to_json(), "text": full_text})
        except Exception as exc:  # noqa: BLE001 - the REPL view must never fail an act
            self._log_event({"event": "observation_write_error", "error": repr(exc)[:300]})
        self._obs_rows, self._level_start = rows, False
        msg = {"role": "user", "_kind": "observation", "content": text}
        if self.vision:
            png = self._picture(rows, b)
            msg = perception.image_message(text, png, self._last_step)
            self.stats["images_sent"] += 1
        msg.update({"_full": full, "_step": self._last_step})
        dt = time.time() - t0
        self.stats["observations"] += 1
        self.stats["observation_chars_max"] = max(self.stats["observation_chars_max"], len(text))
        self.stats["perception_s_total"] = round(self.stats["perception_s_total"] + dt, 3)
        self.stats["perception_s_max"] = round(max(self.stats["perception_s_max"], dt), 3)
        return msg

    def _briefing(self, rows: list[list[int]]) -> perception.Briefing | None:
        """The X8 analysis of the newest frame, once per state. None when switched off or when it fails: the agent
        then gets the text views without it, because perception must never fail an act."""
        if not self.perception.get("briefing", True):
            return None
        key = (self._last_step, self.arc.game.state.levels_completed)
        if self._brief[0] != key:
            legal = [a for a in self.arc.game.state.available_actions if a != 0]
            prev = self._brief[1] if self._brief[0] is not None and self._brief[0][1] == key[1] else None   # same level
            try:
                self._brief = (key, perception.brief(rows, self._objects, legal, prev))
            except Exception as exc:  # noqa: BLE001
                self.stats["perception_errors"] += 1
                self._log_event({"event": "perception_error", "where": "brief", "error": repr(exc)[:300],
                                 "traceback": traceback.format_exc()[-2000:]})
                self._brief = (key, None)
        return self._brief[1]

    def _picture(self, rows: list[list[int]], b: perception.Briefing | None) -> bytes:
        """The X8 picture; the plain frame when there is no briefing or drawing it fails."""
        cell = int(self.perception.get("cell", 10))
        if b is not None:
            try:
                return perception.picture(b, cell)
            except Exception as exc:  # noqa: BLE001
                self.stats["perception_errors"] += 1
                self._log_event({"event": "perception_error", "where": "picture", "error": repr(exc)[:300],
                                 "traceback": traceback.format_exc()[-2000:]})
        return perception.render(rows, cell)

    def _observe_called(self) -> str:
        """``observe()`` ran in ipython: the full view goes in after this reply's tool results, once per state. The
        returned note is added to the cell's output."""
        self.stats["observe_calls"] += 1
        where = f"step #{self._last_step}" if self._last_step is not None else "the game start"
        if self._pending_obs is not None and self._pending_obs.get("_full"):
            return f"[observe: the full state at {where} follows in the next message]"
        newest = self._pending_obs or next((m for m in reversed(self.messages) if m.get("_kind") == "observation"),
                                           None)
        if newest is not None and newest.get("_full") and newest.get("_step") == self._last_step:
            return f"[observe: the full state at {where} is already in the newest board message above]"
        self._pending_obs = self._observation(full=True)
        self.stats["observe_pushes"] += 1
        return f"[observe: the full state at {where} follows in the next message]"

    def _tool_act(self, args: dict[str, Any], call_id: str, turn: int, reply: Any) -> str:
        arc, mem = self.arc, self.memory
        raw = args.get("actions")
        if isinstance(raw, str):
            raw = json.loads(raw) if raw.strip().startswith("[") else [raw]
        elif isinstance(raw, int):
            raw = [raw]
        if not isinstance(raw, list) or not raw:
            raise ValueError(prompts.event_message("act_bad_args", problem="`actions` must be a list of 1 or more moves"))
        cap = int(self.cfg.get("act_max_actions", 5))
        if len(raw) > cap:
            raise ValueError(prompts.event_message("act_too_many", sent=len(raw), act_max=cap))
        try:
            parsed = [self._parse_action(a) for a in raw]   # all or nothing: nothing runs if one item is bad
        except ValueError as exc:
            raise ValueError(prompts.event_message("act_bad_args", problem=str(exc))) from None
        legal = [a for a in arc.game.state.available_actions if a != 0]
        bad = [self._label(a, r, c) for a, r, c in parsed if a != 0 and a not in legal]
        if bad:
            raise ValueError(prompts.event_message("act_illegal", bad=", ".join(bad), legal=legal))
        level0 = arc.game.state.levels_completed
        try:
            wrote = mem.write_act(args, turn=turn, level=level0)   # raises on a bad hypothesis before any action
        except ValueError as exc:
            raise ValueError(prompts.event_message("act_bad_args", problem=str(exc))) from None
        self.stats["act_calls"] += 1
        self._log_memory(wrote, call_id, turn)
        start = arc.game.action_count
        self._step_ref(call_id, turn, reply, args)
        lines, steps, stop, level_up = [], [], None, False
        for n, (a, r, c) in enumerate(parsed):
            label = self._label(a, r, c)
            try:
                if a == 0:
                    obs = arc.handle({"type": "arc.reset"}, source="tool")
                else:
                    obs = arc.handle({"type": "arc.step", "action": a, **({"x": c, "y": r} if a == 6 else {})},
                                     source="tool")
            except Exception as exc:  # noqa: BLE001
                stop = f"action {n + 1} ({label}) refused: {exc}"
                break
            st = dict(arc.last_step or {})
            before_objs = self._objects
            if obs.get("level_up"):
                self.tracker.reset()
                self._objects, _ = self.tracker.observe(self._frame())
                change = "LEVEL UP: a new level begins (shown in full below)"
            else:
                self._objects, change = self.tracker.observe(self._frame())
                change = change or "no change"
            self._last_step = st.get("i")
            line = f"#{st.get('i')} {label}: {change}"
            if st.get("repeat_of") is not None:
                line += f" [repeat: same state and action as #{st['repeat_of']}]"
            if st.get("state") == "GAME_OVER":
                line += " [GAME_OVER]"
            lines.append(line)
            row = {"i": st.get("i"), "level": st.get("level"), "action": label, "change": change,
                   "state": st.get("state"), "level_up": bool(obs.get("level_up")), "repeat_of": st.get("repeat_of"),
                   "turn": turn}
            mem.record_step(row)
            steps.append(row)
            if obs.get("level_up"):
                level_up = True
                promo = mem.on_level_up(level_won=arc.game.state.levels_completed, step=st.get("i"),
                                        objects=before_objs)
                self.stats["promotions"] += len(promo["promoted"])
                self._log_event({"event": "level_up", "turn": turn, "level": arc.game.state.levels_completed,
                                 "i": st.get("i"), **promo})
                self._level_start = True
                self._curate_pending = "level_up"
                stop = "level up"
                break
            if obs.get("state") == "GAME_OVER":
                stop = "GAME_OVER"
                break
            if arc.finished:
                stop = "game finished"
                break
        arc.step_ref = {"session": self.name, "turn": turn, "call": call_id, "after_cell": True}
        self._publish_history(steps)
        done = len(steps)
        self.stats["act_actions"] += done
        s = arc.game.state
        out = [f"act: {done} of {len(parsed)} actions done" + (f"; stopped: {stop}" if stop else ""), *lines]
        if s.engine_state.name == "WIN":
            out.append(prompts.event_message("win"))
        elif level_up:
            out.append(prompts.event_message("level_up", levels_done=s.levels_completed,
                                             win_levels=arc.game.number_of_levels, **self._left()))
        if s.engine_state.name == "GAME_OVER":
            out.append(prompts.event_message("game_over", **self._left()))
        mem.record_act({"turn": turn, "level": level0, "steps": [x["i"] for x in steps], "plan": wrote["plan"],
                        "stop": stop})
        self._log_event({"event": "act", "turn": turn, "call": call_id, "plan": wrote["plan"],
                         "goal": wrote["goal"], "hypotheses": wrote["hypotheses"], "findings": wrote["findings"],
                         "requested": len(parsed), "done": done, "stop": stop, "steps": steps,
                         "actions": [start, arc.game.action_count]})
        if done == 0 and stop:
            raise RuntimeError(stop + " (your plan and memory writes were saved)")
        if lines:
            self._last_change = lines[-1]
        self._pending_obs = self._observation()
        return "\n".join(out)

    def _publish_history(self, steps: list[dict[str, Any]]) -> None:
        """One line per step for ``obs.history`` in the REPL."""
        try:
            observation.append_history(self.kernel.session_dir, steps)
        except Exception as exc:  # noqa: BLE001 - the REPL view must never fail an act
            self._log_event({"event": "observation_write_error", "error": repr(exc)[:300]})

    def _log_memory(self, wrote: dict[str, Any], call_id: str, turn: int) -> None:
        level = self.arc.game.state.levels_completed
        acount = self.arc.game.action_count
        for h in wrote["hypotheses"]:
            self.stats["hypothesis_events"] += 1
            self._log_event({"event": "hypothesis", "turn": turn, "call": call_id, "level": level,
                             "action_count": acount, **h})
        for f in wrote["findings"]:
            self._log_event({"event": "finding", "turn": turn, "call": call_id, "level": level,
                             "action_count": acount, "text": f["text"], "evidence": f["evidence"]})
        if wrote["goal"]:
            self._log_event({"event": "goal", "turn": turn, "call": call_id, "level": level, "action_count": acount,
                             "goal": wrote["goal"]})

    def _tool_recall(self, args: dict[str, Any], call_id: str, turn: int) -> str:
        query = str(args.get("query") or "").strip()
        scope = str(args.get("scope") or "all").strip().lower()
        cap = int((self.cfg.get("memory") or {}).get("recall_tokens", 1500))
        text, hits = self.memory.recall(query, scope, cap)
        self.stats["recalls"] += 1
        self._log_event({"event": "recall", "turn": turn, "call": call_id, "query": query, "scope": scope,
                         "hits": hits, "chars": len(text)})
        return text

    # --- host-driven auto /refine (upstream serialized-refine checkpoint between turns) ----------------------
    def _maybe_auto_refine(self) -> None:
        if self.e008:
            self._maybe_curate()
            return
        ar = self.cfg.get("auto_refine") or {}
        if not ar.get("enabled"):
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

    def _maybe_curate(self) -> None:
        """E008: the hidden curator at a level-up, after a compaction, and every ``curator_every_turns`` turns."""
        mc = self.cfg.get("memory") or {}
        if not mc.get("curator", True) or self.memory is None:
            return
        reason = self._curate_pending
        if reason is None and self._turns_since_refine >= int(mc.get("curator_every_turns", 25)):
            reason = "turn_interval"
        if reason is None:
            return
        self._curate_pending, self._turns_since_refine = None, 0
        t0 = time.time()
        try:
            rec = curator.curate(llm=self.llm, memory=self.memory, objects=self._objects, reason=reason,
                                 max_tokens=int(mc.get("curator_max_tokens", 4096)), overflow=ContextOverflow)
        except Exception as exc:  # noqa: BLE001 - the curator must never kill the session
            self.stats["curator_errors"] += 1
            self._log_event({"event": "curator", "reason": reason, "error": f"{type(exc).__name__}: {exc}"[:500]})
            return
        self.stats["curator_runs"] += 1
        self._log_event({"event": "curator", "duration_s": round(time.time() - t0, 1),
                         "action_count": self.arc.game.action_count, **rec})

    # --- forced reflection checkpoints (E004, off by default, not upstream) --------------------------------
    def _harness_sig(self) -> tuple:
        from rlm.harness import _DEFAULT_FILE_NAME

        paths = (self.local_harness_dir / _DEFAULT_FILE_NAME, self.global_harness_dir / _DEFAULT_FILE_NAME)
        return tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in paths)

    def _reflection_tick(self) -> None:
        """Host-driven L3: at checkpoints, refuse arc.step until the model writes to the Continual Harness."""
        every = self.cfg.get("reflect_every_actions")
        if self.arc is None or not every:
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
                out[scope] = {k: [f"{e.title} v{e.version}" for e in st.list(k)] for k in ("memory", "skill", "prompt")}
            except Exception as exc:  # noqa: BLE001
                out[scope] = f"unreadable: {exc}"
        return out

    # --- context: system prompt, harness digest, compaction ----------------------------------------------
    def _system_prompt(self) -> str:
        if self._system is None and self.e008:
            self._system = prompts.game_system(game_id=self.arc.game.game_id, win_levels=self.arc.game.number_of_levels,
                                               act_max=int(self.cfg.get("act_max_actions", 5)),
                                               cwd=str(self.kernel.session_dir), vision=self.vision)
        if self._system is None:
            base = prompts.base_prompt(cwd=str(self.kernel.session_dir), transcript=str(self.transcript))
            if self.arc is not None:  # upstream appendSystemPrompt
                base += "\n\n" + prompts.arc_section(
                    game_id=self.arc.game.game_id, win_levels=self.arc.game.number_of_levels,
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
            summary = ("(summary unavailable) Your memory block above is intact; use `recall` for earlier steps."
                       if self.e008 else "(summary unavailable) Rebuild your understanding from the REPL variables, "
                       f"`await arc.transitions()` and the conversation log {self.transcript}.")
        self._compact_failed_at = None
        self._summary = summary
        kept = [m for m in self.messages[prep.first_kept:] if m.get("_kind") != compaction.DIGEST_KIND]
        # E008: the memory is pinned outside the conversation, so the head carries the summary only.
        head = compaction.head_message(summary, prompts.event_message("compacted", **self._left()) + "\n\n"
                                       if self.e008 else refine.digest_block(self._digest()) + "\n\n")
        if self.e008 and not any(context._has_image(m) for m in kept):
            kept.append(self._observation_refresh())   # the newest state must survive a compaction
        self.messages = [head, *kept]
        self._usage_tokens = None
        self.stats["compactions"] += 1
        self._compact_refine_pending = True
        if self.e008:
            self._curate_pending = self._curate_pending or "compact"
        self._log_event({"event": "compaction", "reason": reason, "tokens_before": tokens_before,
                         "token_scale": round(self._token_scale, 2),
                         "summarized_messages": len(prep.to_summarize), "turn_prefix_messages": len(prep.turn_prefix),
                         "kept_messages": len(kept), "split_turn": prep.is_split, "summary_chars": len(summary),
                         "action_count": self.arc.game.action_count if self.arc is not None else None,
                         "duration_s": round(time.time() - t0, 1)})
        self._log_event({"event": "message", **head})
        if self.arc is not None and (not kept or kept[-1]["role"] != "user"):
            # Upstream queues the autonomous continuation for a threshold compaction in autonomous mode.
            self.stats["continuations"] += 1
            self._append({"role": "user", "content": self._continuation()})

    def _observation_refresh(self) -> dict[str, Any]:
        """The current state in full (briefing, objects, board, picture), e.g. after a compaction cut the last one
        away."""
        msg = self._observation(full=True)
        self._log_event({"event": "message", **context.loggable(msg)})
        return msg

    # --- host requests from the REPL ----------------------------------------------------------------------
    def _host(self, req: dict[str, Any]) -> dict[str, Any]:
        kind = str(req.get("type", ""))
        if kind.startswith("arc."):
            if self.arc is None:
                raise RuntimeError("no game is attached to this session")
            if self.e008:
                raise PermissionError("in this harness the game state is pushed to you after every act; read it "
                                      "with `observe()` and act with the act tool")
            return self.arc.handle(req)
        raise RuntimeError(f"host request {kind!r} is not supported by this harness")

    # --- bookkeeping -------------------------------------------------------------------------------------
    def _time_left(self) -> str:
        return f"{max(0, int(self.deadline - time.time())) // 60} min left"

    def _status(self) -> str:
        return f"{self.arc.status_line()}, {self._time_left()}" if self.arc else self._time_left()

    def _append(self, msg: dict[str, Any], **extra: Any) -> None:
        self.messages.append(msg)
        self._log_event({"event": "message", **context.loggable(msg), **{k: v for k, v in extra.items() if v}})

    def _log_event(self, obj: dict[str, Any]) -> None:
        obj = {"t": round(time.time(), 2), **obj}
        try:
            with self.transcript.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(obj, default=str) + "\n")
        except Exception:  # noqa: BLE001
            pass
