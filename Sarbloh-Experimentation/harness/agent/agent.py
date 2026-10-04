"""AgentSession: the Prime Agent loop, ported to Python from upstream ``core/agent-session.ts`` (commit 2d24ad4).

What is kept from the paper (arXiv 2608.23552, section 2) and how it maps here:
- L1 active context  -> ``self.messages``. Compaction (``harness.agent.compaction``) replaces the older prefix with a
                        short summary and keeps the newest messages verbatim; the kernel's names are listed in code.
                        After a compaction the first prompt's size is the floor, and the next compaction waits until
                        the context passes it by ``floor_gap_tokens``. An overflow never ends the game: the first one
                        compacts, the second trims in code (back to the newest full state, thinking only on the newest
                        reply, old tool outputs as one-line placeholders), a third resets to the newest state. Every
                        level-up resets the context in code: the level-up message and the new level's full state.
- L2 REPL            -> one persistent ``rlm.repl`` kernel per session (``Kernel``).
- L3 disk state      -> ``transcript.jsonl`` (full history, survives compaction) and the game's memory
                        (``harness.memory``). Append-only (the default): a stored message, in full after every
                        rewrite, else its changed sections before each new state. Else: rendered once per user
                        message and joined to the newest one.
- Autonomous mode    -> when the root stops calling tools before the game ends, a continuation message is sent,
                        bounded by turn, token and wall-clock budgets; the end-condition test is "game won". A
                        compaction is followed by a continuation only when the kept messages end on a reply with
                        no tool call; after a tool result the agent goes on with its chain.
- Accounting         -> tokens, turns and tool calls are recorded per session.

The system prompt is fixed for the whole session (upstream keeps it stable for the prefix cache). Not ported (no use
offline on one GPU): the daemon/worker/TUI split, goals, heartbeats/cron, create_session, MCP, model switching,
session recovery after a crash, ``bash()`` completion follow-ups, the Continual Harness digest and auto /refine,
``rlm.spawn`` subagents and agent messages (git history has them).
Ours, not upstream: the game status line after each tool result, the "reply was cut off" nudge, and scaling
upstream's chars/4 token estimate by the measured prompt-token ratio (digit-heavy grid text is ~4 tokens per 4 chars,
so chars/4 alone never triggered compaction).

Tools (``harness.agent.tools``): ``ipython`` (think and compute; reads the state with ``observe()``, cannot act or fetch
the game), ``act`` (1 to N actions plus the agent's plan, hypotheses, findings and goal) and ``recall`` (search memory
and skills). After every act the host pushes the new state as the next user message: a change line per action in the
act result, then the short view (changed region) and the picture (``harness.agent.perception``). The full view
(briefing, every object, the whole board, the picture) is pushed at a level start, after a compaction, and when the
agent calls ``observe()`` in ipython (once per state). Append-only (config ``append_only``): nothing already sent is
changed until a rewrite, so the server's prefix cache keeps the whole prompt; at the compaction trigger a drain in code
(old thinking, old pictures and memory messages out) comes before a summary (``harness.agent.context``). The hidden
curator (``harness.agent.curator``) reviews the memory at every level-up (before the level's memory is cleared, writing
skills from it), after a compaction and every ``curator_every_turns`` turns. The host records what the agent writes and judges nothing. Every tool call is logged as
a structured transcript event for ``harness.trace``.
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

from harness.agent import compaction, context, curator, perception, prompts
from harness.agent.tools import game_tools
from harness.game.arc_host import ArcHost
from harness.llm.client import LLM, ContextOverflow
from harness.memory.lifecycle import GameMemory
from harness.runtime.kernel import Kernel
from harness.runtime.skills import observation

_FENCED = re.compile(r"```(?:python|py|ipython|repl)?[ \t]*\n(.*?)```", re.DOTALL)
# The append-only estimate of one picture: the 640-pixel map plus its panel at 32 pixels per token (patch 16, merge 2).
IMAGE_TOKENS = 500
_ACTION_NAMES = {0: "RESET", 1: "A1(up)", 2: "A2(down)", 3: "A3(left)", 4: "A4(right)", 5: "A5(space)",
                 7: "A7(undo)"}


def _wire(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Messages as sent to the server: private ``_`` keys dropped, consecutive user messages joined (some chat
    templates, Gemma's among them, require alternating roles)."""
    return context.merge_users(messages)


class AgentSession:
    def __init__(self, *, cfg: dict[str, Any], llm: LLM, name: str, session_dir: Path, task: str,
                 arc: ArcHost, deadline: float, stop_event: threading.Event, memory_root: Path | None = None) -> None:
        self.cfg = cfg
        self.llm = llm
        self.name = name
        self.task = task
        self.arc = arc
        self.deadline = deadline
        self.stop_event = stop_event
        self.session_dir = session_dir = session_dir.resolve()  # the kernel runs in work/: relative paths break
        self.messages: list[dict[str, Any]] = []
        self.transcript = session_dir / "transcript.jsonl"
        self.end_reason = ""
        self.stats = {"turns": 0, "tool_calls": 0, "output_tokens": 0, "prompt_tokens_last": 0, "compactions": 0,
                      "compaction_failures": 0, "continuations": 0, "llm_failures": 0, "cell_errors": 0,
                      "native_calls": 0, "fenced_calls": 0, "tool_counts": {}, "tool_errors": 0, "act_actions": 0,
                      "act_calls": 0, "act_arg_errors": 0, "recalls": 0, "hypothesis_events": 0, "promotions": 0,
                      "curator_runs": 0, "curator_errors": 0, "skills_written": 0, "observations": 0, "images_sent": 0,
                      "observation_chars_max": 0, "pinned_chars_max": 0, "observe_calls": 0, "observe_pushes": 0,
                      "perception_errors": 0, "perception_s_total": 0.0, "perception_s_max": 0.0,
                      "context_overflows": 0, "context_trims": 0, "context_resets": 0, "drains": 0,
                      "memory_full": 0, "memory_updates": 0}
        self._summary: str | None = None     # the latest compaction summary (updated, not re-summarized, next time)
        self._usage_tokens: int | None = None  # prompt tokens of the last call
        self._usage_at = 0                   # messages after this index are estimated at chars/4
        # Measured tokens per chars/4 estimate. Upstream assumes chars/4; ARC grids are digit text and these
        # tokenizers give every digit its own token, so a printed grid is ~4x the estimate (measured in a smoke run).
        self._token_scale = 1.0
        self._compact_failed_at: int | None = None
        # The prompt size right after the last compaction; no new compaction until the context passes it by
        # compaction.floor_gap_tokens. Pending until the first request after the compaction measures it.
        self._compact_floor: int | None = None
        self._floor_pending = False
        self._overflows = 0                  # context overflows in a row: compact, then trim in code, then reset
        self._level_up_text: str | None = None   # set by a level-up act: the context is reset after its reply
        self._system: str | None = None
        self._turns_since_curate = 0
        self._effort_sent: str | None = None    # the reasoning effort of the last request (logged when it changes)
        self.kernel = Kernel(session_dir / "work", self._host)
        self.tools = game_tools(act_max=int(cfg.get("act_max_actions", 5)))
        self._acted_in_reply = False
        # Perception state, the game's memory, and what is pushed after the current reply's tool results.
        self.perception = dict(cfg.get("perception") or {})
        self.vision = bool(cfg.get("vision")) and bool(self.perception.get("image", True))
        self.tracker = perception.Tracker()
        self._objects: list[Any] = []            # objects of the newest frame, ids stable within the level
        self._obs_rows: list[list[int]] | None = None   # the grid of the last observation pushed (crop baseline)
        self._level_start = True
        self._last_step: int | None = None
        self._last_change: str | None = None     # the newest change line, "#12 A1(up): ..." (shown and in observe())
        self._brief: tuple[Any, Any] = (None, None)   # (state key, Briefing): the analysis of the newest frame
        self._pending_obs: dict[str, Any] | None = None
        self._curate_pending: str | None = None
        self._pinned_chars = 0
        self._pinned: dict[str, Any] | None = None   # the memory message, kept until a new user message arrives
        self._pinned_for: dict[str, Any] | None = None   # the user message it was built for
        # Append-only context: the memory is a stored message; these track the sections as last shown.
        self.append_only = bool(cfg.get("append_only", True))
        self._shown: dict[str, str] | None = None
        self._states_since_full = 0
        self.memory = GameMemory(memory_root or (session_dir / "memory"), arc.game.game_id, cfg.get("memory") or {})
        arc.on_step = lambda ev: self._log_event(ev)

    # --- budget ------------------------------------------------------------------------------------------
    def tokens_spent(self) -> int:
        return self.stats["output_tokens"]

    def should_stop(self) -> str | None:
        if self.stop_event.is_set():
            return "stopped"
        if time.time() >= self.deadline:
            return "wall_clock"
        if self.arc.finished:
            return "game_finished"
        if self.arc.budget_left == 0:
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
        try:
            self.kernel.start()
            self._log_event({"event": "system_prompt", "content": self._system_prompt(), "tools": self.tools,
                             "vision": self.vision})
            # The memory is the pinned block; the first state comes with the task.
            self._log_event({"event": "memory_init", **self.memory.on_new_game()})
            self._append({"role": "user", "content": self.task})
            first = self._observation()
            if self.append_only:
                self._append(self._memory_full())
            self._append(first)
            self._loop()
        except Exception as exc:  # noqa: BLE001 - a session crash must not take the run down
            self.end_reason = f"crash: {type(exc).__name__}: {exc}"
            self._log_event({"event": "crash", "traceback": traceback.format_exc()})
        finally:
            self.kernel.close()
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
            tokens = self._context_tokens()
            if self._wants_compaction(tokens):
                if not (self.append_only and self._drain(tokens)):
                    self._compact("threshold")
            pinned = None if self.append_only else self._memory_message()
            msgs = context.build(self._system_prompt(), self.messages, pinned, vision=self.vision,
                                 reasoning_budget=self._reasoning_budget(), append_only=self.append_only)

            # 3) send the current message to LLM and handles other errors like emergency compaction, llm failures time out
            try:
                reply = self.llm.chat(msgs, tools=tools, max_tokens=self.cfg["max_tokens_per_turn"],
                                      timeout_s=max(60.0, min(self.cfg["request_timeout_s"], self.deadline - time.time())),
                                      template_kwargs=self._effort())
            except ContextOverflow as exc:
                self._on_overflow(exc, tokens)
                continue
            except Exception as exc:  # noqa: BLE001
                self.stats["llm_failures"] += 1
                self._log_event({"event": "llm_error", "error": repr(exc)})
                time.sleep(10)
                continue
            self._overflows = 0
            self._record_floor(reply.prompt_tokens)
            self.stats["llm_failures"] = 0
            self.stats["turns"] += 1
            self.stats["output_tokens"] += reply.completion_tokens
            self.stats["prompt_tokens_last"] = reply.prompt_tokens
            self._turns_since_curate += 1

            # 4) Parse reply for tool calls
            calls = self._calls(reply)
            assistant: dict[str, Any] = {"role": "assistant", "content": reply.content or ""}
            native = bool(calls and calls[0]["native"])
            if native:
                assistant["tool_calls"] = [c["raw"] for c in calls]
            if reply.reasoning and (native or self.append_only):
                assistant["_reasoning"] = reply.reasoning   # old builder: sent back until the next user message
            estimate = (len(msgs[0]["content"]) + self._pinned_chars + 3) // 4 + \
                sum(self._estimate(m) for m in self.messages) + self._chain_reasoning_tokens(0)
            if reply.prompt_tokens > 0 and estimate > 0:
                self._token_scale = min(4.0, max(0.5, reply.prompt_tokens / estimate))
            self._append(assistant, reasoning=reply.reasoning, usage=(reply.prompt_tokens, reply.completion_tokens),
                         finish=reply.finish_reason, turn=self.stats["turns"])
            # Anchor on the prompt and estimate the stored reply like any newer message: its thinking counts only
            # while it is still sent (until the next user message), so completion_tokens would overcount.
            self._usage_tokens = reply.prompt_tokens
            self._usage_at = len(self.messages) - 1

            # 5) If tool calls: execute each, then maybe run the curator
            if calls:
                self._acted_in_reply = False   # one act per reply: the model must read each result first
                for call in calls:
                    self._run_call(call, reply)
                if self._pending_obs is not None:
                    # The new state goes in right after this reply's tool results, before the next turn.
                    self._push(self._pending_obs)
                    self._pending_obs = None
                self._reset_if_level_up()
                self._maybe_curate()
                continue

            # 6) the reply was cut off
            if reply.finish_reason == "length":
                self._push({"role": "user", "content": prompts.event_message("cut_off")})
                continue
            # 7) if No tool call: the model ended its turn.
            if self.arc.finished:
                self.end_reason = "answered"
                return
            self._maybe_curate()
            self.stats["continuations"] += 1
            self._push({"role": "user", "content": self._continuation()})

    def _push(self, msg: dict[str, Any]) -> None:
        """A new user message. Append-only: the memory sections that changed go in just before it."""
        if self.append_only:
            note = self._memory_note()
            if note is not None:
                self._append(note)
        self._append(msg)

    def _memory_full(self) -> dict[str, Any]:
        """The whole memory as a stored message (append-only), at the start, after a rewrite and every
        ``memory_full_every`` states. What it shows becomes the baseline for the next updates."""
        blocks = self.memory.blocks(self._objects)
        self._shown, self._states_since_full = dict(blocks), 0
        msg = context.pinned_message(blocks, self._game_status())
        self.stats["memory_full"] += 1
        self.stats["pinned_chars_max"] = max(self.stats["pinned_chars_max"], len(msg["content"]))
        return msg

    def _memory_note(self) -> dict[str, Any] | None:
        """Before a new user message (append-only): the full memory every ``memory_full_every`` states, else only the
        sections that changed since they were last shown, or nothing."""
        self._states_since_full += 1
        every = int(self.cfg.get("memory_full_every") or 0)
        if self._shown is None or (every and self._states_since_full >= every):
            return self._memory_full()
        blocks = self.memory.blocks(self._objects)
        keys = context.changed_sections(blocks, self._shown)
        if not keys:
            return None
        self._shown = dict(blocks)
        self.stats["memory_updates"] += 1
        return context.memory_update(blocks, keys)

    def _insert_memory(self) -> None:
        """After a rewrite (append-only): the full memory just before the newest user message, where the old builder
        put it."""
        msg = self._memory_full()
        self.messages.insert(max(context.chain_start(self.messages) - 1, 0), msg)
        self._log_event({"event": "message", **msg})

    def _cold(self) -> None:
        """The prompt was just rewritten, so its next request is prefilled in full wherever it runs: a free moment to
        give the scheduler slot to a game whose cache is still warm (``ScheduledLLM.mark_cold``)."""
        mark = getattr(self.llm, "mark_cold", None)
        if callable(mark):
            mark()

    def _memory_message(self) -> dict[str, Any]:
        """The memory, built once per user message (a new state, a continuation, a compaction) and then kept byte for
        byte through the tool-call chain that follows, so the prompt stays a prefix of the next one. Writes made in
        between (an ipython turn changes nothing; the curator may) show with the next state."""
        newest = self.messages[context.chain_start(self.messages) - 1] if self.messages else None
        if self._pinned is None or self._pinned_for is not newest:
            self._pinned = context.pinned_message(self.memory.blocks(self._objects), self._game_status())
            self._pinned_for = newest
            self._pinned_chars = len(self._pinned["content"])
            self.stats["pinned_chars_max"] = max(self.stats["pinned_chars_max"], self._pinned_chars)
        return self._pinned

    def _effort(self) -> dict[str, str] | None:
        """The chat template's reasoning effort for the level the agent is on (config ``reasoning_effort``). A change
        alters the first line of the system prompt, so the whole prompt is prefilled again once."""
        eff = self.cfg.get("reasoning_effort") or {}
        value = eff.get("early") if self._level() <= prompts.REASON_EARLY_LEVELS else eff.get("later")
        if value not in (None, "xhigh", "medium", "low"):
            raise ValueError(f"reasoning_effort {value!r}: the chat template takes xhigh, medium or low")
        if value != self._effort_sent:
            self._effort_sent = value
            self._log_event({"event": "reasoning_effort", "level": self._level(), "value": value})
        return {"reasoning_effort": value} if value else None

    def _continuation(self) -> str:
        return prompts.continuation(**self._left())

    def _left(self) -> dict[str, int | None]:
        """Moves and minutes left, for the low-budget and low-time notes."""
        return {"moves_left": self.arc.budget_left, "minutes_left": max(0, int(self.deadline - time.time())) // 60}

    def _level(self) -> int:
        """The level the agent is on, 1-based."""
        return self.arc.game.state.levels_completed + 1

    # --- tool calls --------------------------------------------------------------------------------------
    def _calls(self, reply: Any) -> list[dict[str, Any]]:
        calls = []
        names = [t["function"]["name"] for t in self.tools]
        for tc in reply.tool_calls:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            name = fn.get("name") or "ipython"
            code, err, parsed = None, None, {}
            if name not in names:
                err = f"Unknown tool {name!r}. Tools: {', '.join(names)}."
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
            before = self.arc.game.action_count
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
                if self.arc.game.action_count == before:
                    self._acted_in_reply = False   # a refused act spent nothing: a corrected retry may follow
        else:
            timeout = max(10.0, min(self.cfg["cell_timeout_s"], self.deadline - time.time()))
            res = self.kernel.execute(call["code"], timeout_s=timeout)
            if res.status != "ok":
                self.stats["cell_errors"] += 1
                failed = True
            text = res.render(self.cfg["tool_output_chars"])
            if any(observation.MIME in d for d in res.displays):
                text += "\n" + self._observe_called()
            self._log_event({"event": "cell", "turn": turn, "call": call_id, "status": res.status,
                             "duration_s": round(res.duration_s, 2), "output_chars": len(text)})
        text += f"\n[game: {self.arc.status_line()} | {self._time_left()}]"
        if call["native"]:
            msg = {"role": "tool", "tool_call_id": call["raw"]["id"], "content": text}
        else:
            msg = {"role": "user", "content": f"[ipython output]\n{text}"}
        if failed:
            msg["_error"] = True
        self._append(msg)

    # --- tools: act and recall ----------------------------------------------------------------------------
    def _run_tool(self, name: str, args: dict[str, Any], call_id: str, turn: int, reply: Any) -> str:
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
        """The newest state as a user message, with the picture when vision is on. The full view (briefing,
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
        """The briefing analysis of the newest frame, once per state. None when switched off or when it fails: the agent
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
        """The picture; the plain frame when there is no briefing or drawing it fails."""
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
        lines, steps, stop, level_up, new_skills = [], [], None, False, []
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
                # Skills are written from the level's memory before the level-up clears it; the open questions the
                # curator wrote are put back after the clear.
                cur = self._curate("level_up", before_objs)
                promo = mem.on_level_up(level_won=arc.game.state.levels_completed, step=st.get("i"),
                                        objects=before_objs)
                if cur is not None:
                    new_skills = cur["skills"]
                    if cur["questions"]:
                        mem.working.set_questions(cur["questions"])
                        mem.save()
                self.stats["promotions"] += len(promo["promoted"])
                self._log_event({"event": "level_up", "turn": turn, "level": arc.game.state.levels_completed,
                                 "i": st.get("i"), **promo})
                self._level_start = True
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
                                             win_levels=arc.game.number_of_levels, level=self._level(),
                                             new_skills=new_skills, **self._left()))
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
        if level_up:
            self._level_up_text = "\n".join(out)
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

    # --- the hidden curator ------------------------------------------------------------------------------
    def _maybe_curate(self) -> None:
        """The curator after a compaction and every ``curator_every_turns`` turns (the level-up pass runs inside
        ``act``, before the level's memory is cleared)."""
        reason = self._curate_pending
        if reason is None and self._turns_since_curate >= int((self.cfg.get("memory") or {}).get("curator_every_turns",
                                                                                                   25)):
            reason = "turn_interval"
        if reason is None:
            return
        self._curate(reason, self._objects)

    def _curate(self, reason: str, objects: list[Any]) -> dict[str, Any] | None:
        """One curator pass. Returns its record (questions, lessons, skills written, loaded), or None when it is off
        or fails."""
        mc = self.cfg.get("memory") or {}
        self._curate_pending, self._turns_since_curate = None, 0
        if not mc.get("curator", True):
            return None
        t0 = time.time()
        try:
            rec = curator.curate(llm=self.llm, memory=self.memory, objects=objects, reason=reason,
                                 max_tokens=int(mc.get("curator_max_tokens", 4096)), overflow=ContextOverflow)
        except Exception as exc:  # noqa: BLE001 - the curator must never kill the session
            self.stats["curator_errors"] += 1
            self._log_event({"event": "curator", "reason": reason, "error": f"{type(exc).__name__}: {exc}"[:500]})
            return None
        self.stats["curator_runs"] += 1
        self.stats["skills_written"] += len(rec["skills"])
        self._log_event({"event": "curator", "duration_s": round(time.time() - t0, 1),
                         "action_count": self.arc.game.action_count, **rec})
        return rec

    # --- context: system prompt, compaction ---------------------------------------------------------------
    def _system_prompt(self) -> str:
        if self._system is None:
            self._system = prompts.game_system(game_id=self.arc.game.game_id, win_levels=self.arc.game.number_of_levels,
                                               act_max=int(self.cfg.get("act_max_actions", 5)),
                                               cwd=str(self.kernel.session_dir), vision=self.vision)
        return self._system

    def _context_limits(self) -> tuple[int, int, int]:
        c = self.cfg["compaction"]
        return int(self.cfg["context_window"]), int(c["reserve_tokens"]), int(c["keep_recent_tokens"])

    def _context_tokens(self) -> int:
        """Upstream estimateContextTokens: last usage + estimate of the messages after it (chars/4, scaled)."""
        est = lambda ms: int(self._token_scale * sum(self._estimate(m) for m in ms))
        if self._usage_tokens is None:
            return est(self.messages) + int(self._token_scale * self._chain_reasoning_tokens(0))
        return self._usage_tokens + est(self.messages[self._usage_at:]) + \
            int(self._token_scale * self._chain_reasoning_tokens(self._usage_at))

    def _wants_compaction(self, tokens: int) -> bool:
        """Above the trigger (``compaction.should_compact``), not right after a failed try, and well above the size the
        last compaction left: the floor plus ``floor_gap_tokens``, once the first request after it has measured it."""
        window, reserve, _ = self._context_limits()
        c = self.cfg["compaction"]
        if not compaction.should_compact(tokens, window, reserve, c.get("trigger_tokens")):
            return False
        if self._compact_failed_at is not None and tokens <= self._compact_failed_at + 1024:
            return False
        if self._floor_pending:
            return False
        return self._compact_floor is None or tokens > self._compact_floor + int(c.get("floor_gap_tokens", 8192))

    def _record_floor(self, prompt_tokens: int) -> None:
        """The first request after a compaction measures the floor."""
        if self._floor_pending:
            self._compact_floor = prompt_tokens or self._context_tokens()
            self._floor_pending = False
            self._log_event({"event": "compaction_floor", "tokens": self._compact_floor})

    def _on_overflow(self, exc: Exception, tokens: int) -> None:
        """An overflow never ends the game: the first in a row compacts, the second trims in code, any later one resets
        the context to the newest state. Past the third, each also counts as an LLM failure, so a context that cannot
        fit even then ends at ``max_consecutive_llm_failures`` instead of looping for ever."""
        self._overflows += 1
        self.stats["context_overflows"] += 1
        self._log_event({"event": "context_overflow", "count": self._overflows, "tokens": tokens,
                         "error": str(exc)[:300]})
        if self._overflows == 1:
            self._compact("overflow")
        elif self._overflows == 2:
            self._trim_context()
        else:
            if self._overflows > 3:
                self.stats["llm_failures"] += 1
            self._reset_context("overflow", prompts.event_message("compacted", level=self._level(), **self._left()))

    def _reset_if_level_up(self) -> None:
        """After a reply whose act won a level: the new level starts from a clean context."""
        if self._level_up_text is None:
            return
        text, self._level_up_text = self._level_up_text, None
        if not self.arc.finished:
            self._reset_context("level_up", text)

    def _reasoning_budget(self) -> int:
        return int(self.cfg.get("chain_reasoning_tokens", context.REASONING_BUDGET_TOKENS))

    def _estimate(self, m: dict[str, Any]) -> int:
        """chars/4 of a message without its thinking; append-only counts its pictures too (every one is sent)."""
        return compaction.estimate_tokens(m, reasoning=False, image_tokens=IMAGE_TOKENS if self.append_only else 0)

    def _chain_reasoning_tokens(self, start: int) -> int:
        """chars/4 of the thinking that is still sent, in the messages from ``start`` on: the same rule as the request
        (append-only: every reply's; else ``context.reasoning_sent``: the current chain, newest first, within the
        thinking budget)."""
        if self.append_only:
            return sum(context.reasoning_tokens(m) for m in self.messages[start:])
        return sum(context.reasoning_tokens(self.messages[i])
                   for i in context.reasoning_sent(self.messages, self._reasoning_budget()) if i >= start)

    def _drain(self, tokens_before: int) -> bool:
        """Append-only, at the compaction trigger: rewrite the history as the old builder sent it, with no model call
        (``context.drain``: thinking before the current chain, old pictures and memory messages out, a full memory
        back in). Returns False, changing nothing, when the drained context would still be above
        ``drain_max_fraction`` of the trigger: then a compaction is due."""
        window, reserve, _ = self._context_limits()
        c = self.cfg["compaction"]
        limit = window - reserve
        if c.get("trigger_tokens"):
            limit = min(limit, int(c["trigger_tokens"]))
        drained = context.drain(self.messages)
        mem = context.pinned_message(self.memory.blocks(self._objects), self._game_status())
        est = (len(self._system_prompt()) + len(mem["content"]) + 3) // 4 + sum(self._estimate(m) for m in drained) + \
            sum(context.reasoning_tokens(m) for m in drained)
        after = int(self._token_scale * est)
        if after > float(self.cfg.get("drain_max_fraction", 0.7)) * limit:
            self._log_event({"event": "drain_skipped", "tokens_before": tokens_before, "tokens_after_est": after,
                             "limit": limit})
            return False
        self.messages = drained
        self._insert_memory()
        self._pinned, self._usage_tokens, self._usage_at = None, None, 0
        self._floor_pending, self._compact_floor = True, None
        self.stats["drains"] += 1
        self._log_event({"event": "drain", "tokens_before": tokens_before, "tokens_after_est": after,
                         "messages": len(self.messages), "action_count": self.arc.game.action_count})
        self._cold()
        return True

    def _compact(self, reason: str) -> None:
        _, reserve, keep = self._context_limits()
        tokens_before = self._context_tokens()
        # The memory messages are not conversation: they stay out of the summary, and a full one goes back in.
        msgs = [m for m in self.messages if not context.is_memory(m)] if self.append_only else self.messages
        # keep_recent_tokens is in real tokens; the cut-point walk counts chars/4 estimates.
        prep = compaction.prepare(msgs, int(keep / self._token_scale), self._summary, tokens_before)
        if prep is None:
            self._log_event({"event": "compaction_skipped", "reason": reason, "tokens": tokens_before})
            return
        t0 = time.time()
        try:
            summary = compaction.summarize(self.llm, prep, int(self.cfg["compaction"].get("summary_tokens", 2048)),
                                           ContextOverflow)
        except Exception as exc:  # noqa: BLE001
            self.stats["compaction_failures"] += 1
            self._log_event({"event": "compaction_error", "reason": reason, "error": f"{type(exc).__name__}: {exc}"})
            if reason != "overflow":
                self._compact_failed_at = tokens_before
                return
            summary = "(summary unavailable) Your memory block is intact; use `recall` for earlier steps."
        self._compact_failed_at = None
        self._summary = summary
        kept = msgs[prep.first_kept:]
        if self.append_only:
            kept = context.drain(kept)
        # The memory is not part of the conversation, so the head carries the summary only.
        names = self._kernel_names()
        head = compaction.head_message(summary, prompts.event_message("compacted", level=self._level(),
                                                                      **self._left()) + "\n\n" +
                                       (names + "\n\n" if names else ""))
        self._refresh_after_compaction(kept)
        self.messages = [head, *kept]
        if self.append_only:
            self._insert_memory()
        self._pinned = None   # the memory is built again for the new prompt
        self._usage_tokens = None
        self._floor_pending, self._compact_floor = True, None
        self.stats["compactions"] += 1
        self._curate_pending = self._curate_pending or "compact"
        self._log_event({"event": "compaction", "reason": reason, "tokens_before": tokens_before,
                         "token_scale": round(self._token_scale, 2),
                         "summarized_messages": len(prep.to_summarize), "turn_prefix_messages": len(prep.turn_prefix),
                         "kept_messages": len(kept), "split_turn": prep.is_split, "summary_chars": len(summary),
                         "action_count": self.arc.game.action_count, "duration_s": round(time.time() - t0, 1)})
        self._log_event({"event": "message", **head})
        if kept[-1]["role"] == "assistant":
            # Upstream queues the autonomous continuation for a threshold compaction in autonomous mode. After a tool
            # result nothing is added: the agent goes on with its tool-call chain and keeps its thinking.
            self.stats["continuations"] += 1
            self._push({"role": "user", "content": self._continuation()})
        self._cold()

    def _refresh_after_compaction(self, kept: list[dict[str, Any]]) -> None:
        """The newest state in full must survive a compaction. Nothing is added when the newest kept observation is
        already the full view of the current step. Otherwise the full view goes where the newest user message is: it
        replaces that message when it is the short view of the current step, else it goes just after it. Either way it
        stays before the current tool-call chain, so the chain keeps its thinking."""
        start = context.chain_start(kept)
        newest = next((m for m in reversed(kept) if m.get("_kind") == "observation"), None)
        if newest is not None and newest.get("_full") and newest.get("_step") == self._last_step:
            return
        last_user = kept[start - 1] if start else None
        if last_user is not None and last_user.get("_kind") == "observation" and \
                last_user.get("_step") == self._last_step:
            kept[start - 1] = self._observation_refresh()
        else:
            kept.insert(start, self._observation_refresh())

    def _observation_refresh(self) -> dict[str, Any]:
        """The current state in full (briefing, objects, board, picture), e.g. after a compaction cut the last one
        away."""
        msg = self._observation(full=True)
        self._log_event({"event": "message", **context.loggable(msg)})
        return msg

    def _trim_context(self) -> None:
        """The second overflow in a row: trim in code, with no model call. The history goes back to the newest full
        state (the compaction head stays), every reply but the newest loses its thinking, and every tool output but the
        newest reply's becomes a one-line placeholder. The full view of the current step is put back if it is missing."""
        before = self._context_tokens()
        msgs = self.messages
        head = [msgs[0]] if msgs and msgs[0].get("_kind") == compaction.COMPACTION_KIND else []
        full = next((i for i in range(len(msgs) - 1, len(head) - 1, -1)
                     if msgs[i].get("_kind") == "observation" and msgs[i].get("_full")), None)
        kept = list(msgs[full:] if full is not None else msgs[max(context.chain_start(msgs) - 1, len(head)):])
        last = max((i for i, m in enumerate(kept) if m["role"] == "assistant"), default=len(kept))
        for i, m in enumerate(kept):
            if m["role"] == "assistant" and i != last and m.get("_reasoning"):
                kept[i] = {k: v for k, v in m.items() if k != "_reasoning"}
            elif m["role"] == "tool" and i < last:
                first = next((ln.strip() for ln in compaction._text(m).splitlines() if ln.strip()), "")
                kept[i] = {**m, "content": f"[output removed to fit the context; it began: {first[:120]}]"}
        self._refresh_after_compaction(kept)
        self.messages = head + kept
        if self.append_only:
            self.messages = [m for m in self.messages if not context.is_memory(m)]
            self._insert_memory()
        self._pinned, self._usage_tokens, self._usage_at = None, None, 0
        self._floor_pending, self._compact_floor = True, None
        self.stats["context_trims"] += 1
        self._cold()
        self._log_event({"event": "context_trim", "tokens_before": before, "tokens_after": self._context_tokens(),
                         "messages_before": len(msgs), "messages_after": len(self.messages),
                         "action_count": self.arc.game.action_count})

    def _reset_context(self, reason: str, text: str) -> None:
        """A clean context in code, with no model call: ``text`` (the level-up message, or the compaction note when even
        a trimmed context overflows), the kernel's names, and the newest full state. The system prompt and the memory
        go with every request anyway. Used at every level-up: a new level starts from its own context."""
        obs = next((m for m in reversed(self.messages) if m.get("_kind") == "observation"), None)
        if obs is None or not obs.get("_full") or obs.get("_step") != self._last_step:
            obs = self._observation_refresh()
        names = self._kernel_names()
        dropped = len(self.messages)
        self.messages = []
        self._append({"role": "user", "_kind": "reset", "content": text + (f"\n\n{names}" if names else "")})
        if self.append_only:
            self._append(self._memory_full())
        self.messages.append(obs)   # logged when it was made
        self._summary, self._pinned, self._usage_tokens, self._usage_at = None, None, None, 0
        self._compact_floor, self._floor_pending, self._compact_failed_at = None, False, None
        self.stats["context_resets"] += 1
        self._cold()
        self._log_event({"event": "context_reset", "reason": reason, "messages_dropped": dropped,
                         "tokens_after": self._context_tokens(), "action_count": self.arc.game.action_count})

    def _kernel_names(self) -> str:
        """One line naming what the agent defined in its kernel, read from the REPL globals (not from a summarizer)."""
        try:
            names = self.kernel.user_names()
        except Exception as exc:  # noqa: BLE001 - the list is a help, never a failure
            self._log_event({"event": "kernel_names_error", "error": repr(exc)[:300]})
            return ""
        parts = [f"{key}: {', '.join(names[key])}" for key in ("functions", "classes", "values", "modules")
                 if names and names.get(key)]
        if not parts:
            return ""
        text = "Your Python kernel still holds what you defined there; reuse it instead of writing it again. " + \
            "; ".join(parts)
        return text if len(text) <= 2000 else text[:1990] + " ..."

    # --- host requests from the REPL ----------------------------------------------------------------------
    def _host(self, req: dict[str, Any]) -> dict[str, Any]:
        kind = str(req.get("type", ""))
        if kind.startswith("arc."):
            raise PermissionError("in this harness the game state is pushed to you after every act; read it "
                                  "with `observe()` and act with the act tool")
        raise RuntimeError(f"host request {kind!r} is not supported by this harness")

    # --- bookkeeping -------------------------------------------------------------------------------------
    def _time_left(self) -> str:
        return f"{max(0, int(self.deadline - time.time())) // 60} min left"

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
