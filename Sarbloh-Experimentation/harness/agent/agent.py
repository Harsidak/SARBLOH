"""The agent's main loop: one AgentSession plays one game.

How a game runs: ``AgentSession._loop`` is the outer loop, one pass per turn. A turn is up to max_inner_requests
model requests (``_request``) sent back to back; it ends at the first request whose cells moved, at a cap
(max_inner_requests, turn_yield_tokens) or at a stop, and ``guardrails.end_turn`` then gives the scheduler slot back.
One request:
1. stop?            ``guardrails.should_stop`` (time, actions, turns, tokens, game over)
2. room?            ``_drain`` drops the oldest whole turns when the context is past the trigger
3. ask the model    ``self.llm.chat``; an overflow goes to ``_on_overflow`` (drain, then trim, then reset)
4. store the reply  its tool calls are read by ``guardrails.read_calls``
5. run the tools    ``_run_call``: ipython runs in the kernel; each move of ``await play(...)`` comes back to
                    ``_host`` -> ``_tool_act``, which steps the game and builds the new view (``_observation``);
                    the output then gets the memory receipt and the code-graph change line (``_graph_delta``);
                    then the level-up note
No tool call: a continuation message (or, if the reply was cut off, the "call the tool now" note) and the turn goes on.
End of a turn: stuck note, curator (``_maybe_curate``, off by default: it runs at level-ups), scheduler slot
(``guardrails.end_turn``).
The guardrails (refused moves, notes that push the model to act, tool-call checks, hard stops) are in guardrails.py.
There is one tool, ipython: ``recall()`` and ``graph()`` are kernel functions, and the memory is five kernel variables
(GOAL, PLAN, HYPOTHESES, FINDINGS, SKILLS) copied after every call (``_read_memory``).
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from typing import Any

# pyrefly: ignore [missing-import]
from harness.agent import compaction, context, curator, guardrails, perception, prompts
# pyrefly: ignore [missing-import]
from harness.agent.tools import game_tools
# pyrefly: ignore [missing-import]
from harness.game.arc_host import ArcHost
# pyrefly: ignore [missing-import]
from harness.llm.client import LLM, ContextOverflow
# pyrefly: ignore [missing-import]
from harness.memory.lifecycle import GameMemory
# pyrefly: ignore [missing-import]
from harness.runtime.kernel import Kernel
# pyrefly: ignore [missing-import]
from harness.runtime.skills import codegraph, observation, recalling

# The append-only estimate of one picture: the 640-pixel map plus its panel at 32 pixels per token (patch 16, merge 2).
IMAGE_TOKENS = 500


def _wire(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Messages as sent to the server: private ``_`` keys dropped, consecutive user messages joined (some chat
    templates, Gemma's among them, require alternating roles)."""
    return context.merge_users(messages)


def system_prompt(cfg: dict[str, Any]) -> str:
    """The system prompt for the agent config ``cfg``: the same for every game, so the run can pre-warm it."""
    p = cfg.get("perception") or {}
    return prompts.game_system(act_max=int(cfg.get("act_max_actions", 5)),
                               vision=bool(cfg.get("vision")) and bool(p.get("image", True)),
                               picture_last=not bool(p.get("old_images", False)),
                               code_graph=bool(cfg.get("code_graph", True)),
                               reset_move=bool(cfg.get("reset_move", False)),
                               revisit_stop=bool(cfg.get("revisit_stop", False)),
                               code_kept=bool(cfg.get("keep_code", False)),
                               reasoning_kept=bool(cfg.get("keep_reasoning", True)),
                               board_text=bool(p.get("ascii", True)),
                               short_objects=bool(p.get("short_objects", True)) and bool(p.get("segmentation", True)),
                               segmentation=bool(p.get("segmentation", True)),
                               briefing=bool(p.get("briefing", True)),
                               observe_view=bool(p.get("observe_view", True)),
                               memory_shown=bool(cfg.get("append_only", True))
                               and bool((cfg.get("memory") or {}).get("in_context", True)))


def effort_kwargs(cfg: dict[str, Any], level: int) -> dict[str, str] | None:
    """The chat template's reasoning effort on ``level`` (1-based; config ``reasoning_effort``)."""
    eff = cfg.get("reasoning_effort") or {}
    value = eff.get("early") if level <= prompts.EARLY_LEVELS else eff.get("later")
    if value not in (None, "xhigh", "medium", "low"):
        raise ValueError(f"reasoning_effort {value!r}: the chat template takes xhigh, medium or low")
    return {"reasoning_effort": value} if value else None


def prewarm(cfg: dict[str, Any], llm: Any) -> int:
    """One 1-token request per reasoning effort the run uses, with the shared system prompt and tools, so the server's
    prefix cache already holds them when the games start. Returns the requests sent; a failure only skips it."""
    system, tools, sent, seen = system_prompt(cfg), game_tools(), 0, []
    for level in (1, prompts.EARLY_LEVELS + 1):   # the early and the later effort
        kwargs = effort_kwargs(cfg, level)
        if kwargs in seen:
            continue
        seen.append(kwargs)
        try:
            llm.chat([{"role": "system", "content": system}, {"role": "user", "content": "."}], tools=tools,
                     max_tokens=1, timeout_s=120.0, template_kwargs=kwargs)
            sent += 1
        except Exception as exc:  # noqa: BLE001 - only a cache warm-up
            print(f"[prewarm] failed: {exc!r}", flush=True)
    return sent


class AgentSession:
    tracing = True
    _frames = 0
    code_graph = True
    _graph: dict[str, Any] = {}   # replaced, never changed in place
    picture_last = False

    def __init__(self, *, cfg: dict[str, Any], llm: LLM, name: str, session_dir: Path, task: str,
                 arc: ArcHost, deadline: float, stop_event: threading.Event, memory_root: Path | None = None,
                 tracing: bool = True) -> None:
        self.cfg = cfg
        # Config ``tracing``: the transcript, with what harness/trace.py needs to rebuild every request exactly (a
        # snapshot of the context after each rewrite, the pictures as files, a digest of each request). Off: nothing
        # is written. Logging only: the messages and the requests are the same either way.
        self.tracing = tracing
        self._frames = 0                     # pictures saved to frames/
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
        self.stats = {"turns": 0, "tool_calls": 0, "output_tokens": 0, "prompt_tokens_last": 0, "continuations": 0, "llm_failures": 0, "cell_errors": 0,
                      "native_calls": 0, "fenced_calls": 0, "tool_counts": {}, "tool_errors": 0, "act_actions": 0,
                      "act_calls": 0, "act_arg_errors": 0, "recalls": 0, "hypothesis_events": 0, "promotions": 0,
                      "curator_runs": 0, "curator_errors": 0, "skills_written": 0, "observations": 0, "images_sent": 0,
                      "observation_chars_max": 0, "pinned_chars_max": 0, "observe_calls": 0, "observe_pushes": 0,
                      "perception_errors": 0, "perception_s_total": 0.0, "perception_s_max": 0.0,
                      "context_overflows": 0, "context_trims": 0, "context_resets": 0, "drains": 0,
                      "graph_deltas": 0, "graph_snapshots": 0,
                      "memory_full": 0, "memory_updates": 0, "turn_yields": 0, "outer_turns": 0,
                      "turn_requests_max": 0,
                      "surprises": 0, "surprise_turns": 0, "stuck_notes": 0, "wall_hits": 0, "auto_resets": 0,
                      "model_resets": 0, "revisits": 0}
        self._usage_tokens: int | None = None  # prompt tokens of the last call
        self._usage_at = 0                   # messages after this index are estimated at chars/4
        # Measured tokens per chars/4 estimate. Upstream assumes chars/4; ARC grids are digit text and these
        # tokenizers give every digit its own token, so a printed grid is ~4x the estimate (measured in a smoke run).
        self._token_scale = 1.0
        self._overflows = 0                  # context overflows in a row: drain, then trim in code, then reset
        self._level_up_text: str | None = None   # set by a level-up act: the context is reset after its reply
        self._system: str | None = None
        self._turns_since_curate = 0
        self._effort_sent: str | None = None    # the reasoning effort of the last request (logged when it changes)
        # recall() in the kernel reads the game's memory files itself: it needs where they are.
        memory_root = Path(memory_root or (session_dir / "memory")).resolve()
        self.kernel = Kernel(session_dir / "work", self._host, env={
            "SARBLOH_MEMORY_DIR": str(memory_root), "SARBLOH_GAME": arc.game.game_id,
            "SARBLOH_FILE_GUARD": "1" if cfg.get("file_guard", True) else "0",
            "SARBLOH_RECALL_TOKENS": str(int((cfg.get("memory") or {}).get("recall_tokens", 1000)))})
        self.tools = game_tools()
        # Config code_graph: the map of the agent's live functions (codegraph.snapshot) after its last cell, a change
        # line at the end of each ipython output, and the whole map back in the context after a rewrite.
        self.code_graph = bool(cfg.get("code_graph", True))
        self._graph: dict[str, Any] = {}
        self._call_graphs: dict[str, str] = {}   # call id -> graph lines of what that call defined (_store_reply)
        self._act_info: dict[str, Any] | None = None
        self._life_boards: set[bytes] | None = None   # the revisit leash: the boards of this try (guardrails.new_life)
        self._resets_used: dict[int, int] = {}        # level (levels completed) -> resets the agent sent on it
        self._cell: dict[str, Any] | None = None   # the ipython cell running now: its call, and the acts it made
        # Perception state, the game's memory, and what is pushed after the current reply's tool results.
        self.perception = dict(cfg.get("perception") or {})
        self.vision = bool(cfg.get("vision")) and bool(self.perception.get("image", True))
        self.old_images = bool(self.perception.get("old_images", False))   # off: only the newest picture is sent
        # With old_images off (append-only), the views stay text and the picture of the board is its own stored
        # message, added once per new state right after the output that changed it. It is never moved or removed
        # before a rewrite: on this hybrid model the server reuses a prefix only up to the end of an earlier request,
        # so a picture sent last and then dropped made every request prefill from the system prompt again (Qwen run of
        # 2026-10-19: 4416 cached tokens on every one of 960 requests, 10% of the prompt).
        self.picture_last = self.vision and not self.old_images and bool(cfg.get("append_only", True))
        self.keep_pictures = self.old_images or self.picture_last   # stored pictures stay until a rewrite
        self.tracker = perception.Tracker()
        self._objects: list[Any] = []            # objects of the newest frame, ids stable within the level
        self._obs_rows: list[list[int]] | None = None   # the grid of the last observation pushed (crop baseline)
        self._level_start = True
        self._last_step: int | None = None
        self._last_change: str | None = None     # the newest change line, "#12 A1(up): ..." (shown and in observe())
        self._brief: tuple[Any, Any] = (None, None)   # (state key, Briefing): the analysis of the newest frame
        self._pending_obs: dict[str, Any] | None = None
        self._full_shown_step: int | None = None   # the step whose full view an observe() cell output last showed
        self._act_via: str | None = None
        self._act_predicted: bool | None = None
        self._act_play_id: int | None = None
        self._pinned_chars = 0
        self._pinned: dict[str, Any] | None = None   # the memory message, kept until a new user message arrives
        self._pinned_for: dict[str, Any] | None = None   # the user message it was built for
        # Append-only context: the memory is a stored message; these track the sections as last shown.
        self.append_only = bool(cfg.get("append_only", True))
        # Config memory.in_context: off, no memory message goes into the prompt (the agent reads it with `recall`).
        self.memory_in_context = bool((cfg.get("memory") or {}).get("in_context", True))
        # Config turn_yield_tokens: reply tokens written since the last move (the turn's thinking budget).
        self.turn_yield_tokens = int(cfg.get("turn_yield_tokens") or 0)
        self._turn_tokens = 0
        # Config surprise_yield_tokens: the smaller thinking budget of the turn after play() stopped at a surprise (one
        # rule was wrong: a small fix, not a rethink). 0 = the normal budget.
        self.surprise_yield_tokens = int(cfg.get("surprise_yield_tokens") or 0)
        self._turn_budget = self.turn_yield_tokens   # the budget of the current turn
        self._surprised = False                      # set by play() (host request arc.surprise) during this reply
        # Config stuck_note_tokens: every time the current level has eaten this many more reply tokens without being
        # won, a note says to question the goal and the least tested rule. 0 = off.
        self.stuck_note_tokens = int(cfg.get("stuck_note_tokens") or 0)
        self._stuck_level, self._level_tokens, self._stuck_notes = 0, 0, 0
        self._shown: dict[str, str] | None = None
        self._states_since_full = 0
        self.memory = GameMemory(memory_root, arc.game.game_id, cfg.get("memory") or {})
        arc.on_step = lambda ev: self._log_event(ev)

    # --- budget ------------------------------------------------------------------------------------------
    def tokens_spent(self) -> int:
        return self.stats["output_tokens"]

    # --- main loop ---------------------------------------------------------------------------------------
    def run(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.kernel.start()
            self._log_event({"event": "system_prompt", "content": self._system_prompt(), "tools": self.tools,
                             "vision": self.vision, "append_only": self.append_only, "old_images": self.old_images,
                             "picture_last": self.picture_last, "keep_pictures": self.keep_pictures,
                             "reasoning_budget": self._reasoning_budget()})
            # The memory is the pinned block; the first state comes with the task.
            self._log_event({"event": "memory_init", **self.memory.on_new_game()})
            self._append({"role": "user", "content": self.task})
            first = self._observation()
            if self.append_only and self.memory_in_context:
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
            self.stats["skills_loaded"] = list(self.memory.working.skills)
            self._log_event({"event": "end", "reason": self.end_reason, "stats": self.stats})

    def _loop(self) -> None:
        """The outer loop: one turn after another until a stop. A turn is up to max_inner_requests model requests sent
        back to back: while the model only runs Python (to look, compute, check its own output) the turn goes on, with
        no note and no change of scheduler slot. The turn ends at the first request whose cells moved (``await
        play(...)``), at a cap (max_inner_requests, or turn_yield_tokens of thinking since the last move), or at a stop;
        then ``guardrails.end_turn`` hands the slot to the scheduler."""
        cap = max(1, int(self.cfg.get("max_inner_requests", 12)))
        while True:
            reason, requests = "request_cap", 0
            while requests < cap:
                # 1) any reason to stop?
                stop = guardrails.should_stop(self)
                if stop:
                    self.end_reason, reason = stop, "stop"
                    break
                requests += 1
                outcome = self._request()
                if outcome in ("moved", "answered"):
                    reason = outcome
                    break
                if guardrails.over_turn_budget(self):
                    reason = "token_cap"
                    break
            guardrails.end_turn(self, reason, requests)
            if self.end_reason:
                return

    def _request(self) -> str:
        """One model request of a turn and its tool calls. Returns "moved" (a cell sent moves or the level changed),
        "looked" (code ran without moving, or no tool call), "retry" (overflow or LLM failure: nothing was stored) or
        "answered" (no tool call and the game is over). Called from _loop."""
        tools = self.tools
        # 2) Drain the old messages (no model call) when the context passes the trigger
        tokens = self._context_tokens()
        if self._wants_drain(tokens):
            self._drain(tokens, "threshold")
        pinned = None
        if not self.append_only and self.memory_in_context:
            # Old builder only: the memory block sent with each request, rebuilt when a new user message arrives.
            newest = self.messages[context.chain_start(self.messages) - 1] if self.messages else None
            if self._pinned is None or self._pinned_for is not newest:
                self._pinned = context.pinned_message(self.memory.blocks(self._objects), self._game_status())
                self._pinned_for = newest
                self._pinned_chars = len(self._pinned["content"])
                self._log_event({"event": "pinned", **self._pinned})   # not a stored message: the trace needs it
                self.stats["pinned_chars_max"] = max(self.stats["pinned_chars_max"], self._pinned_chars)
            pinned = self._pinned
        self._add_picture()
        msgs = context.build(self._system_prompt(), self.messages, pinned, vision=self.vision,
                             reasoning_budget=self._reasoning_budget(), append_only=self.append_only,
                             old_images=self.keep_pictures)
        effort = self._effort()

        # 3) send the current message to LLM and handles a context overflow (drain, trim, reset), llm failures and time outs
        try:
            reply = self.llm.chat(msgs, tools=tools, max_tokens=self._max_tokens(),
                                  timeout_s=max(60.0, min(self.cfg["request_timeout_s"], self.deadline - time.time())),
                                  template_kwargs=effort)
        except ContextOverflow as exc:
            self._on_overflow(exc, tokens)
            return "retry"
        except Exception as exc:  # noqa: BLE001
            self.stats["llm_failures"] += 1
            self._log_event({"event": "llm_error", "error": repr(exc)})
            time.sleep(10)
            return "retry"
        self._overflows = 0
        self.stats["llm_failures"] = 0
        self.stats["turns"] += 1
        self.stats["output_tokens"] += reply.completion_tokens
        self.stats["prompt_tokens_last"] = reply.prompt_tokens
        self._turns_since_curate += 1
        self._turn_tokens += reply.completion_tokens
        if self.arc.game.state.levels_completed != self._stuck_level:   # a new level: its own token count
            self._stuck_level, self._level_tokens, self._stuck_notes = self.arc.game.state.levels_completed, 0, 0
        self._level_tokens += reply.completion_tokens

        # 4) Parse reply for tool calls
        calls = guardrails.read_calls(self, reply)
        assistant: dict[str, Any] = {"role": "assistant", "content": reply.content or ""}
        native = bool(calls and calls[0]["native"])
        if native:
            assistant["tool_calls"] = [c["raw"] for c in calls]
        # Config keep_reasoning off: the reply is stored without its thinking, so it is never sent back (the transcript
        # still logs it, marked reasoning_unsent for the trace).
        if reply.reasoning and (native or self.append_only) and self.cfg.get("keep_reasoning", True):
            assistant["_reasoning"] = reply.reasoning   # old builder: sent back until the next user message
        # Config keep_code off: the reply is stored without its code; once its cells ran, the code-graph lines of what
        # each cell defined take the code's place (step 5). The transcript keeps the code as written (``code_full``)
        # for the trace and the SFT export.
        full = None
        self._call_graphs = {}
        if calls and not self.cfg.get("keep_code", False):
            full = {k: assistant[k] for k in ("content", "tool_calls") if k in assistant}
            assistant = context.without_code(assistant)
            if assistant == {**assistant, **full}:
                full = None
        estimate = (len(msgs[0]["content"]) + self._pinned_chars + 3) // 4 + \
            sum(self._estimate(m) for m in self.messages) + self._chain_reasoning_tokens(0)
        if reply.prompt_tokens > 0 and estimate > 0:
            self._token_scale = min(4.0, max(0.5, reply.prompt_tokens / estimate))
        # For the trace: the request's digest (harness/trace.py rebuilds it from the transcript and checks it), the
        # template's reasoning effort, and whether the thinking stays in the stored reply (it always does except
        # in the old builder after a fenced-code reply).
        trace = {"wire_sha": context.digest(msgs), "effort": (effort or {}).get("reasoning_effort"),
                 "reasoning_unsent": bool(reply.reasoning) and "_reasoning" not in assistant} if self.tracing else {}
        self._append(assistant, reasoning=reply.reasoning, usage=(reply.prompt_tokens, reply.completion_tokens),
                     finish=reply.finish_reason, turn=self.stats["turns"], code_full=full, **trace)
        # Anchor on the prompt and estimate the stored reply like any newer message: its thinking counts only
        # while it is still sent (until the next user message), so completion_tokens would overcount.
        self._usage_tokens = reply.prompt_tokens
        self._usage_at = reply_at = len(self.messages) - 1

        # 5) The tool calls (each ipython call saves the memory variables of the kernel). Code that only looks keeps
        # the turn going: the model reads its output in the next request, right away.
        if calls:
            moves, level0 = self.stats["act_actions"], self.arc.game.state.levels_completed
            self._surprised = False
            for call in calls:
                self._run_call(call, reply)
            if full is not None:
                self._store_reply(reply_at, full)
            leveled = self.arc.game.state.levels_completed != level0
            if leveled and not self.arc.finished:
                guardrails.level_start_note(self)
            if self._pending_obs is not None:
                # The new state goes in right after this reply's tool results, before the next request.
                self._push(self._with_picture(self._pending_obs))
                self._pending_obs = None
            self._reset_if_level_up()
            return "moved" if self.stats["act_actions"] > moves or leveled else "looked"

        # 6) the reply was cut off before it called a tool
        if reply.finish_reason == "length":
            guardrails.cut_off_note(self)
            return "looked"
        # 7) if No tool call: the model ended its reply.
        if self.arc.finished:
            self.end_reason = "answered"
            return "answered"
        self.stats["continuations"] += 1
        self._push({"role": "user", "content": prompts.continuation(**self._left())})
        return "looked"


    def _push(self, msg: dict[str, Any]) -> None:
        """Adds a new user message, with the memory just before it: in full now and then (memory_full_every), else only
        the sections that changed. Called from _request and the notes in guardrails.py."""
        if self.append_only and self.memory_in_context:
            self._states_since_full += 1
            every = int(self.cfg.get("memory_full_every") or 0)
            if self._shown is None or (every and self._states_since_full >= every):
                self._append(self._memory_full())
            else:
                blocks = self.memory.blocks(self._objects)
                keys = context.changed_sections(blocks, self._shown)
                if keys:
                    self._shown = dict(blocks)
                    self.stats["memory_updates"] += 1
                    self._append(context.memory_update(blocks, keys))
        self._append(msg)

    def _kernel_names(self, full: bool = True) -> str:
        """What the agent defined in its kernel. full=False: the functions and classes only, for the level-start note
        (guardrails.level_start_note); full=True: one line with everything, for _reset_context."""
        try:
            names = self.kernel.user_names()
        except Exception as exc:  # noqa: BLE001 - the list is a help, never a failure
            self._log_event({"event": "kernel_names_error", "error": repr(exc)[:300]})
            return ""
        names = names or {}
        if not full:
            found = [n.split("(")[0] for n in names.get("functions", [])] + list(names.get("classes", []))
            text = ", ".join(found)
            return text if len(text) <= 600 else text[:590] + " ..."
        keys = ("values", "modules") if self.code_graph else ("functions", "classes", "values", "modules")
        parts = [f"{key}: {', '.join(names[key])}" for key in keys if names.get(key)]   # the graph lists the rest
        if not parts:
            return ""
        text = "Your Python kernel still holds what you defined there; reuse it instead of writing it again. " + \
            "; ".join(parts)
        return text if len(text) <= 2000 else text[:1990] + " ..."

    def _graph_delta(self, call_id: str = "") -> str:
        """After an ipython cell: the new code graph of the kernel and the one line of what changed since the last
        cell ("" when nothing did, or config code_graph is off). Diffed on the live namespace, so a deleted or
        rebound function shows as removed. The graph lines of what the cell defined are kept for the stored reply
        (``_call_graphs``, see _store_reply). Called from _run_call."""
        if not self.code_graph:
            return ""
        try:
            nodes = self.kernel.code_graph()
        except Exception as exc:  # noqa: BLE001 - the graph is a help, never a failure
            self._log_event({"event": "code_graph_error", "error": repr(exc)[:300]})
            return ""
        if nodes is None:
            return ""
        line = codegraph.delta(self._graph, nodes)
        self._call_graphs[call_id] = codegraph.cell_map(self._graph, nodes)
        self._graph = nodes
        if line:
            self.stats["graph_deltas"] += 1
            self._log_event({"event": "code_graph", "turn": self.stats["turns"], "delta": line, "nodes": len(nodes)})
        return line

    def _store_reply(self, at: int, full: dict[str, Any]) -> None:
        """After a reply's cells ran (config keep_code off): the stored reply at ``at`` gets, in place of each call's
        code, the code-graph lines of what that call defined. Decided once, before the next request, so the stored
        messages stay a prefix. Logged as a ``reply_stored`` event (trace.sft_wire applies it). Called from _request
        step 5."""
        stored = context.without_code({**self.messages[at], **full}, self._call_graphs)
        if stored == self.messages[at]:
            return
        self.messages[at] = stored
        self._log_event({"event": "reply_stored", "turn": self.stats["turns"],
                         **{k: stored[k] for k in ("content", "tool_calls") if k in stored}})

    def _insert_graph(self) -> None:
        """After a drain, trim or reset: the whole code graph once, before the newest user message (the change lines
        that built it up are gone by then)."""
        if not self.code_graph or not self._graph:
            return
        msg = context.graph_message(codegraph.render(self._graph))
        self.messages.insert(max(context.chain_start(self.messages) - 1, 0), msg)
        self.stats["graph_snapshots"] += 1
        self._log_event({"event": "message", **msg})

    def _memory_full(self) -> dict[str, Any]:
        """The whole memory as one message. Called at the start (run), from _push, and after a drain, trim or reset."""
        blocks = self.memory.blocks(self._objects)
        self._shown, self._states_since_full = dict(blocks), 0
        msg = context.pinned_message(blocks, self._game_status())
        self.stats["memory_full"] += 1
        self.stats["pinned_chars_max"] = max(self.stats["pinned_chars_max"], len(msg["content"]))
        return msg

    def _insert_memory(self) -> None:
        """After a drain or trim: puts the full memory back before the newest user message."""
        if not self.memory_in_context:
            return
        msg = self._memory_full()
        self.messages.insert(max(context.chain_start(self.messages) - 1, 0), msg)
        self._log_event({"event": "message", **msg})

    def _effort(self) -> dict[str, str] | None:
        """The reasoning effort for the current level (config reasoning_effort). Called from _request step 2."""
        kwargs = effort_kwargs(self.cfg, self._level())
        value = (kwargs or {}).get("reasoning_effort")
        if value != self._effort_sent:
            self._effort_sent = value
            self._log_event({"event": "reasoning_effort", "level": self._level(), "value": value})
        return kwargs

    def _max_tokens(self) -> int:
        """The most one reply may write (config max_tokens_per_turn). Called from _request step 3."""
        cap = self.cfg["max_tokens_per_turn"]
        if isinstance(cap, dict):
            cap = cap.get("early") if self._level() <= prompts.EARLY_LEVELS else cap.get("later")
        return int(cap)

    def _left(self) -> dict[str, int | None]:
        """Moves and minutes left, for the notes and the status lines."""
        return {"moves_left": self.arc.budget_left, "minutes_left": max(0, int(self.deadline - time.time())) // 60}

    def _level(self) -> int:
        """The level the agent is on, 1-based."""
        return self.arc.game.state.levels_completed + 1

    # --- tool calls --------------------------------------------------------------------------------------
    def _run_call(self, call: dict[str, Any], reply: Any = None) -> None:
        """Runs one ipython call in the kernel and stores its output as a tool message, built from pieces a drain can
        mask one by one (``context.cell_output``): what the code printed, its error, the state after its moves, the
        memory receipt, the code-graph change line and the game status. Called from _request step 5."""
        self.stats["tool_calls"] += 1
        counts = self.stats["tool_counts"]
        counts[call["name"]] = counts.get(call["name"], 0) + 1
        failed = bool(call["error"])
        turn = self.stats["turns"]
        call_id = call["raw"]["id"] if call["native"] else f"fenced-t{turn}"
        cell: dict[str, Any] | None = None
        if call["error"]:   # an unknown tool or arguments that do not parse (guardrails.read_calls)
            pieces = [("error", call["error"])]
            self.stats["tool_errors"] += 1
            self._log_event({"event": "tool_error", "turn": turn, "call": call_id, "tool": call["name"],
                             "error": call["error"][:500]})
        else:
            timeout = max(10.0, min(self.cfg["cell_timeout_s"], self.deadline - time.time()))
            self._cell = {"call_id": call_id, "turn": turn, "reply": reply, "acts": [], "closed": None, "memory": [],
                          "rows": self._obs_rows}
            try:
                res = self.kernel.execute(call["code"], timeout_s=timeout)
            finally:
                cell, self._cell = self._cell, None
            self._read_memory(cell)
            if res.status != "ok":
                self.stats["cell_errors"] += 1
                failed = True
            out, err = res.split(self.cfg["tool_output_chars"])
            pieces = [("out", out), ("error", err)]
            for d in res.displays:   # recall() in the cell: what it searched, for the trace
                if recalling.MIME in d:
                    self.stats["recalls"] += 1
                    self._log_event({"event": "recall", "turn": turn, "call": call_id, **(d[recalling.MIME] or {})})
            if any(observation.MIME in d for d in res.displays):
                self.stats["observe_calls"] += 1
                if self.perception.get("observe_view", True):   # off: observe() is read in code, nothing is shown
                    pieces.append(("observe", self._observe_called()))
            if self._pending_obs is not None and self._level_up_text is None:
                # The new state after this cell's moves goes in the cell's own output, not in a new user message,
                # so the turn goes on: the agent reads it and plays again in the same turn.
                pieces.append(("state", compaction._text(self._pending_obs)))
                self._pending_obs = None
            # The "[memory saved: ...]" receipt of what this cell's memory variables changed.
            saved = [x for done, _ in cell["memory"] for x in done]
            errors = list(dict.fromkeys(e for _, errs in cell["memory"] for e in errs))
            receipt = []
            if saved:
                receipt.append(f"[memory saved: {', '.join(saved)}]")
            if errors:
                receipt.append(f"[memory not saved: {'; '.join(errors)}. That variable is back to its saved value.]")
            pieces += [("memory", "\n".join(receipt)), ("graph", self._graph_delta(call_id))]
        text, spans = context.cell_output(pieces)
        text = text or "(no output)"
        if cell is not None:
            self._log_event({"event": "cell", "turn": turn, "call": call_id, "status": res.status,
                             "duration_s": round(res.duration_s, 2), "output_chars": len(text),
                             "acts": len(cell["acts"])})
        if call["native"]:
            msg = {"role": "tool", "tool_call_id": call["raw"]["id"], "content": text, "_spans": spans}
        else:
            msg = {"role": "user", "content": f"[ipython output]\n{text}"}
        if failed:
            msg["_error"] = True
        if cell is not None and cell["acts"]:
            msg["_act_log"] = "\n".join(cell["acts"])   # a drain keeps the moves of this cell as a move log
        self._append(msg)

    # --- the game view ------------------------------------------------------------------------------------
    _parse_action = staticmethod(guardrails.parse_action)   # kept for the older tests

    def _frame(self) -> list[list[int]]:
        f = self.arc.game.state.raw.frame[-1]
        return perception.rows_of(f)

    @staticmethod
    def _letters(grid: Any) -> list[str]:
        return ["".join(perception.letter(v) for v in r) for r in perception.rows_of(grid)]

    def _board_before(self) -> list[str] | None:
        """The board before the newest move, for obs.before in the kernel. Called from _observation."""
        t = self.arc.transitions
        return self._letters(t[-1]["before"]) if t else None

    def _move_rows(self, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Each step with its move as a play() path writes it and the boards before and after it, for moves() in the
        kernel. Called from _tool_act."""
        out = []
        for row in steps:
            t = self.arc.transitions[row["i"]]
            a = t["action"]
            move = "reset" if a == 0 else f"6 {t['y']} {t['x']}" if a == 6 else str(a)
            out.append({**row, "move": move, "before": self._letters(t["before"]),
                        "after": self._letters(t["frames"][-1])})
        return out

    def _game_status(self) -> str:
        arc = self.arc
        s = arc.game.state
        legal = " ".join(str(a) for a in s.available_actions if a != 0)
        left = arc.budget_left if arc.max_actions else "unlimited"
        return (f"level {s.levels_completed + 1} of {arc.game.number_of_levels} "
                f"({s.levels_completed} done) | {s.engine_state.name} | legal actions {legal} | "
                f"{arc.game.action_count} actions spent ({arc.game.action_count - arc._level_start} this level), "
                f"{left} left")

    def _observation(self, full: bool = False, picture: bool = True) -> dict[str, Any]:
        """Builds the view of the current state as a user message: the full view at a level start or when full=True, else the short view (what changed). Also writes the board to disk for observe()/play(). Called from _tool_act (after moves), run (first state), _observe_called and _observation_refresh."""
        t0 = time.time()
        rows = self._frame()
        if not self._objects:   # the first look at the game: ids start at 1 (act re-tracks at every level-up)
            self.tracker.reset()
            self._objects, _ = self.tracker.observe(rows)
        full = full or self._level_start or self._obs_rows is None
        p = self.perception
        b = self._briefing(rows)
        # The short view shows what changed since the start of the running cell, not only its last move.
        prev = self._cell["rows"] if self._cell is not None and self._cell.get("rows") is not None else self._obs_rows
        sc = perception.Scene(self._last_step, rows, self._objects, change=self._last_change, prev=prev,
                              status=self._game_status(), briefing=b, briefing_lines=int(p.get("briefing_lines", 30)))
        opts = {"max_objects": int(p.get("max_objects", 40)), "show_ascii": bool(p.get("ascii", True)),
                "segmentation": bool(p.get("segmentation", True))}
        full_text = sc.text(full=True, **opts)
        text = full_text if full else sc.text(budget_tokens=int(p.get("observation_tokens", 800)),
                                              margin=int(p.get("crop_margin", 3)),
                                              short_objects=bool(p.get("short_objects", True)), **opts)
        if not full:   # the state after moves: the same short direction every turn
            text = f"{text}\n\n{prompts.TURN_NOTE}"
        try:
            observation.write(self.kernel.session_dir, {**sc.to_json(), "text": full_text,
                                                        "before": self._board_before()})
        except Exception as exc:  # noqa: BLE001 - the REPL view must never fail an act
            self._log_event({"event": "observation_write_error", "error": repr(exc)[:300]})
        self._obs_rows, self._level_start = rows, False
        msg = {"role": "user", "_kind": "observation", "content": text}
        msg.update({"_full": full, "_step": self._last_step})
        if picture:
            msg = self._with_picture(msg)
        dt = time.time() - t0
        self.stats["observations"] += 1
        self.stats["observation_chars_max"] = max(self.stats["observation_chars_max"], len(text))
        self.stats["perception_s_total"] = round(self.stats["perception_s_total"] + dt, 3)
        self.stats["perception_s_max"] = round(max(self.stats["perception_s_max"], dt), 3)
        return msg

    def _briefing(self, rows: list[list[int]]) -> perception.Briefing | None:
        """The briefing (measured facts and guesses) of the current frame, once per state. Called from _observation and _with_picture."""
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

    def _with_picture(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Adds the picture to a view message, only for messages really sent: the briefing picture (the plain frame if
        the briefing failed), always saved under frames/ for analysis. With ``picture_last`` the message stays text: the
        picture is its own stored message instead (``_add_picture``). Called from
        _request step 5 and _observation."""
        if not self.vision or self.picture_last or isinstance(msg.get("content"), list):
            return msg
        png, name = self._picture()
        out = perception.image_message(str(msg.get("content") or ""), png, msg.get("_step"))
        out.update({"_full": msg.get("_full"), "_step": msg.get("_step")})
        if name:
            out["_image_file"] = name
        return out

    def _picture(self) -> tuple[bytes, str | None]:
        """The picture of the current frame (the briefing picture, or the plain frame if the briefing failed) and the
        file it was saved to under frames/ (None if saving failed)."""
        rows = self._frame()
        b = self._briefing(rows)
        cell = int(self.perception.get("cell", 10))
        png = None
        if b is not None:
            try:
                png = perception.picture(b, cell)
            except Exception as exc:  # noqa: BLE001
                self.stats["perception_errors"] += 1
                self._log_event({"event": "perception_error", "where": "picture", "error": repr(exc)[:300],
                                 "traceback": traceback.format_exc()[-2000:]})
        if png is None:
            png = perception.render(rows, cell)
        self.stats["images_sent"] += 1
        self._frames += 1
        name: str | None = f"frames/img_{self._frames:05d}.png"
        try:
            path = self.session_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(png)
        except Exception as exc:  # noqa: BLE001 - the trace must never fail an act
            self._log_event({"event": "frame_write_error", "error": repr(exc)[:300]})
            name = None
        return png, name

    def _add_picture(self) -> None:
        """With ``picture_last``, stores the picture of the board now as a user message at the end of the history, once
        per state (a move, a level): the request after a move ends with it, and it stays where it is until a rewrite.
        Drawn again after a rewrite that stubbed or dropped it. Called from _request step 3."""
        if not self.picture_last:
            return
        key = [self.arc.game.action_count, self.arc.game.state.levels_completed, self._last_step]
        newest = next((m for m in reversed(self.messages) if m.get("_kind") == context.PICTURE_KIND), None)
        if newest is not None and newest.get("_picture_key") == key and context.has_image(newest):
            return
        png, name = self._picture()
        where = f"after step #{self._last_step}" if self._last_step is not None else "at the start"
        msg = perception.image_message(f"[picture of current state(board), {where}]", png, self._last_step)
        msg.update({"_kind": context.PICTURE_KIND, "_picture_key": key})
        if name:
            msg["_image_file"] = name
        self._append(msg)

    def _observe_called(self) -> str:
        """The agent called observe() in a cell: adds the full view to that cell's output, once per state. Called from _run_call."""
        where = f"step #{self._last_step}" if self._last_step is not None else "the game start"
        newest = next((m for m in reversed(self.messages) if m.get("_kind") == "observation"), None)
        shown = self._full_shown_step == self._last_step or (
            newest is not None and newest.get("_full") and newest.get("_step") == self._last_step)
        if shown and self._pending_obs is None:
            return f"[observe: the full state at {where} is already shown above]"
        self._pending_obs = None
        self._full_shown_step = self._last_step
        self.stats["observe_pushes"] += 1
        return f"[observe: the full state at {where}]\n" + compaction._text(self._observation(full=True, picture=False))

    def _tool_act(self, args: dict[str, Any], call_id: str, turn: int, reply: Any) -> str:
        """Steps the game for the given moves, tracks objects, handles a level-up (curator, memory), and sets _pending_obs to the new view. Called from _host (a move from play())."""
        arc, mem = self.arc, self.memory
        via, self._act_via = self._act_via, None
        predicted, self._act_predicted = self._act_predicted, None
        play_id, self._act_play_id = self._act_play_id, None
        parsed = guardrails.read_moves(self, args)   # all or nothing: nothing runs if one item is bad
        level0 = arc.game.state.levels_completed
        self.stats["act_calls"] += 1
        start = arc.game.action_count
        arc.cell_actions = 0
        arc.step_ref = {"session": self.name, "turn": turn, "call": call_id, "code": json.dumps(args),
                        "say": getattr(reply, "content", None), "thought": getattr(reply, "reasoning", None)}
        lines, steps, stop, level_up, new_skills, load, revisit = [], [], None, False, [], {}, False
        if self._life_boards is None:
            guardrails.new_life(self)
        for n, (a, r, c) in enumerate(parsed):
            label = guardrails.label(a, r, c)
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
                guardrails.new_life(self)
            else:
                self._objects, change = self.tracker.observe(self._frame())
                change = change or "no change"
                if a == 0:
                    self._resets_used[level0] = self._resets_used.get(level0, 0) + 1
                    self.stats["model_resets"] += 1
                    guardrails.new_life(self)
                elif obs.get("state") != "GAME_OVER":
                    revisit = guardrails.seen_before(self)
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
                # Skills are written from the level's memory before the level-up clears it; the skills the curator
                # picks go back to play() with this move, which loads them into SKILLS in the kernel.
                cur = self._curate("level_up", before_objs)
                promo = mem.on_level_up(level_won=arc.game.state.levels_completed, step=st.get("i"),
                                        objects=before_objs)
                if cur is not None:
                    new_skills = cur["skills"]
                    load = {n: mem.skills.text(n) or "" for n in cur["loaded"]}
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
        done = len(steps)
        lost = stop == "GAME_OVER"
        restarted = lost and self._auto_reset(turn, lines, steps)
        arc.step_ref = {"session": self.name, "turn": turn, "call": call_id, "after_cell": True}
        try:   # one line per step for obs.history and moves() in the kernel
            observation.append_history(self.kernel.session_dir, steps)
            observation.append_moves(self.kernel.session_dir, self._move_rows(steps))
        except Exception as exc:  # noqa: BLE001 - the REPL view must never fail an act
            self._log_event({"event": "observation_write_error", "error": repr(exc)[:300]})
        self.stats["act_actions"] += done
        s = arc.game.state
        out = [f"act: {done} of {len(parsed)} actions done" + (f"; stopped: {stop}" if stop else ""), *lines]
        if s.engine_state.name == "WIN":
            out.append(prompts.event_message("win"))
        elif level_up:
            out.append(prompts.event_message("level_up", levels_done=s.levels_completed,
                                             win_levels=arc.game.number_of_levels, level=self._level(),
                                             new_skills=new_skills, loaded_skills=list(load), **self._left()))
        if lost:
            out.append(prompts.event_message("game_over" if restarted else "game_over_reset", **self._left()))
        plan = mem.working.plan_for(level0) or None
        mem.record_act({"turn": turn, "level": level0, "steps": [x["i"] for x in steps], "plan": plan, "stop": stop})
        self._log_event({"event": "act", "turn": turn, "call": call_id, "plan": plan,
                         "requested": len(parsed), "done": done, "stop": stop, "steps": steps,
                         "actions": [start, arc.game.action_count], "via": via, "predicted": predicted,
                         "play_id": play_id})
        self._act_info = {"done": done, "stopped": stop, "level_up": level_up,
                          "state": "GAME_OVER" if lost else s.engine_state.name, "revisit": revisit}
        if load:
            self._act_info["load_skills"] = load
        if level_up:   # the host's copy of GOAL goes back to the kernel by play()
            self._act_info["goal"] = mem.working.kernel_values()["GOAL"]
        if done == 0 and stop:
            raise RuntimeError(stop)
        if lines:
            self._last_change = lines[-1]
        self._pending_obs = self._observation(picture=False)   # drawn only if it is pushed as a message (level up)
        if level_up:
            self._level_up_text = "\n".join(out)
        return "\n".join(out)

    def _auto_reset(self, turn: int, lines: list[str], steps: list[dict[str, Any]]) -> bool:
        """After a GAME_OVER the host restarts the level itself with one RESET, as the Duck harness does, and adds its
        change line and step row; it is not one of the agent's config reset_move resets. False when the reset was refused
        (budget or time spent). Called from _tool_act."""
        try:
            self.arc.handle({"type": "arc.reset"}, source="auto")
        except Exception as exc:  # noqa: BLE001 - out of moves or time: the game ends here
            self._log_event({"event": "auto_reset_error", "turn": turn, "error": str(exc)[:300]})
            return False
        st = dict(self.arc.last_step or {})
        self._objects, _ = self.tracker.observe(self._frame())
        guardrails.new_life(self)
        self._last_step = st.get("i")
        change = "the level starts again from its first state"
        lines.append(f"#{st.get('i')} RESET (automatic after GAME_OVER): {change}")
        row = {"i": st.get("i"), "level": st.get("level"), "action": "RESET", "change": change,
               "state": st.get("state"), "level_up": False, "repeat_of": st.get("repeat_of"), "turn": turn}
        self.memory.record_step(row)
        steps.append(row)
        self.stats["auto_resets"] += 1
        self._log_event({"event": "auto_reset", "turn": turn, "i": st.get("i")})
        return True

    def _read_memory(self, cell: dict[str, Any]) -> None:
        """After an ipython cell: copies GOAL/PLAN/HYPOTHESES/FINDINGS from the kernel into memory. Called from _run_call."""
        try:
            values = self.kernel.memory_values()
        except Exception as exc:  # noqa: BLE001 - reading the memory must never fail a call
            values = None
            self._log_event({"event": "memory_read_error", "error": repr(exc)[:300]})
        if values is None:
            return
        wrote = self._sync_memory(values, cell)
        # A deleted variable and one that was not saved (wrong shape) get the saved value back.
        lost = {k: v for k, v in self.memory.working.kernel_values().items()
                if k not in values or k in wrote["rejected"]}
        if lost:
            self.kernel.set_memory(lost)
            self._log_event({"event": "memory_restored", "names": sorted(lost)})

    def _sync_memory(self, values: Any, cell: dict[str, Any]) -> dict[str, Any]:
        """Saves the memory variables, logs what changed and notes it for the cell's receipt; returns what
        ``GameMemory.sync`` wrote. Called from _read_memory and _host (the memory sent with a move)."""
        call_id, turn = cell["call_id"], cell["turn"]
        wrote = self.memory.sync(values if isinstance(values, dict) else {}, turn=turn,
                                 level=self.arc.game.state.levels_completed)
        level, acount = self.arc.game.state.levels_completed, self.arc.game.action_count
        for h in wrote["hypotheses"]:
            self.stats["hypothesis_events"] += 1
            self._log_event({"event": "hypothesis", "turn": turn, "call": call_id, "level": level,
                             "action_count": acount, **h})
        for f in wrote["findings"]:
            self._log_event({"event": "finding", "turn": turn, "call": call_id, "level": level,
                             "action_count": acount, "text": f["text"], "evidence": f["evidence"]})
        for g in wrote["goal"]:
            self._log_event({"event": "goal", "turn": turn, "call": call_id, "level": g["level"],
                             "action_count": acount, "goal": g["goal"], "status": g["status"],
                             "prev_status": g["prev_status"]})
        saved = [f"plan {k}" for k in wrote["plan"]]
        saved += ["goal added" for _ in wrote["goal"]]
        saved += [f"{h['id']} added" for h in wrote["hypotheses"]]
        if wrote["findings"]:
            saved.append(f"{len(wrote['findings'])} finding{'s' if len(wrote['findings']) > 1 else ''} added")
        if wrote["removed"]:
            saved.append(f"{len(wrote['removed'])} removed")
            self._log_event({"event": "memory_removed", "turn": turn, "call": call_id, "texts": wrote["removed"]})
        if wrote["errors"]:
            self._log_event({"event": "memory_error", "turn": turn, "call": call_id, "errors": wrote["errors"]})
        cell["memory"].append((saved, wrote["errors"]))
        return wrote

    # --- the hidden curator ------------------------------------------------------------------------------
    def _maybe_curate(self) -> None:
        """Runs the curator once curator_every_turns requests have gone by (0: never here, only at level-ups); the
        skills it picks go into SKILLS in the kernel (no cell runs at the end of a turn). Called from
        guardrails.end_turn."""
        every = int((self.cfg.get("memory") or {}).get("curator_every_turns", 0))
        if every <= 0 or self._turns_since_curate < every:
            return
        cur = self._curate("turn_interval", self._objects)
        if cur is not None and cur["loaded"]:
            try:
                self.kernel.load_skills({n: self.memory.skills.text(n) or "" for n in cur["loaded"]})
            except Exception as exc:  # noqa: BLE001 - skills not loaded are still found by recall()
                self._log_event({"event": "load_skills_error", "error": repr(exc)[:300]})

    def _curate(self, reason: str, objects: list[Any]) -> dict[str, Any] | None:
        """One curator pass (writes lessons and skills, picks the skills to load). Called from _maybe_curate and
        _tool_act at a level-up."""
        mc = self.cfg.get("memory") or {}
        self._turns_since_curate = 0
        if not mc.get("curator", True):
            return None
        t0 = time.time()
        try:
            rec = curator.curate(llm=self.llm, memory=self.memory, objects=objects, reason=reason,
                                 max_tokens=int(mc.get("curator_max_tokens", 4096)), overflow=ContextOverflow,
                                 code_graph=codegraph.render(self._graph) if self._graph else "")
        except Exception as exc:  # noqa: BLE001 - the curator must never kill the session
            self.stats["curator_errors"] += 1
            self._log_event({"event": "curator", "reason": reason, "error": f"{type(exc).__name__}: {exc}"[:500]})
            return None
        self.stats["curator_runs"] += 1
        self.stats["skills_written"] += len(rec["skills"])
        self._log_event({"event": "curator", "duration_s": round(time.time() - t0, 1),
                         "action_count": self.arc.game.action_count, **rec})
        return rec

    # --- context: system prompt, drain ----------------------------------------------------------------------
    def _system_prompt(self) -> str:
        if self._system is None:
            self._system = system_prompt(self.cfg)
        return self._system

    def _context_tokens(self) -> int:
        """Estimated prompt size: last measured prompt tokens plus an estimate of the newer messages. Called from _request step 2."""
        est = lambda ms: int(self._token_scale * sum(self._estimate(m) for m in ms))
        if self._usage_tokens is None:
            return est(self.messages) + int(self._token_scale * self._chain_reasoning_tokens(0))
        return self._usage_tokens + est(self.messages[self._usage_at:]) + \
            int(self._token_scale * self._chain_reasoning_tokens(self._usage_at))

    def _wants_drain(self, tokens: int) -> bool:
        """True when the context is past the drain trigger. Called from _request step 2."""
        c = self.cfg["compaction"]
        return compaction.should_compact(tokens, int(self.cfg["context_window"]), int(c["reserve_tokens"]),
                                         c.get("trigger_tokens"))

    def _on_overflow(self, exc: Exception, tokens: int) -> None:
        """The server said the prompt is too long: 1st time drain, 2nd trim, then reset. Called from _request step 3."""
        self._overflows += 1
        self.stats["context_overflows"] += 1
        self._log_event({"event": "context_overflow", "count": self._overflows, "tokens": tokens,
                         "error": str(exc)[:300]})
        if self._overflows == 1:
            self._drain(tokens, "overflow")
        elif self._overflows == 2:
            self._trim_context()
        else:
            if self._overflows > 3:
                self.stats["llm_failures"] += 1
            self._reset_context("overflow", prompts.event_message("compacted", level=self._level(), **self._left()))

    def _reset_if_level_up(self) -> None:
        """Config level_up_reset: after a level-up reply, starts the new level from a clean context. Called from _request step 5."""
        if self._level_up_text is None:
            return
        text, self._level_up_text = self._level_up_text, None
        if not self.arc.finished and self.cfg.get("level_up_reset", True):
            self._reset_context("level_up", text)

    def _reasoning_budget(self) -> int:
        return int(self.cfg.get("chain_reasoning_tokens", context.REASONING_BUDGET_TOKENS))

    def _estimate(self, m: dict[str, Any]) -> int:
        """chars/4 of one message (pictures counted)."""
        return compaction.estimate_tokens(m, reasoning=False,
                                          image_tokens=IMAGE_TOKENS if self.append_only and self.keep_pictures else 0)

    def _chain_reasoning_tokens(self, start: int) -> int:
        """chars/4 of the thinking still sent, from message start on. Used by _context_tokens."""
        if self.append_only:
            return sum(context.reasoning_tokens(m) for m in self.messages[start:])
        return sum(context.reasoning_tokens(self.messages[i])
                   for i in context.reasoning_sent(self.messages, self._reasoning_budget()) if i >= start)

    def _drain(self, tokens_before: int, reason: str) -> None:
        """Drops the oldest whole turns in code (no model call), as the Duck harness does, until the context is under
        drain_target_tokens; what is kept stays word for word. Trims if the newest turn alone is over the target.
        Called from _request step 2 and _on_overflow."""
        target = int(self.cfg["compaction"].get("drain_target_tokens", 70000))
        mem = context.pinned_message(self.memory.blocks(self._objects), self._game_status())             if self.append_only and self.memory_in_context else None
        graph = context.graph_message(codegraph.render(self._graph)) if self.code_graph and self._graph else None
        size = lambda m: self._estimate(m) + context.reasoning_tokens(m)
        fixed = (len(self._system_prompt()) + (len(mem["content"]) if mem else 0) +
                 (len(graph["content"]) if graph else 0) + 3) // 4
        # The tail kept word for word: the target less the system prompt, the memory, the code graph and the first message.
        keep = max(int(target / self._token_scale) - fixed - (size(self.messages[0]) if self.messages else 0), 0)
        # The oldest user or assistant message from which the rest fits in keep (never a tool result: it belongs to
        # the reply before it); at least the newest turn stays.
        turns = [i for i in range(1, len(self.messages)) if self.messages[i]["role"] in ("user", "assistant")]
        cut_at, acc = (turns[-1] if turns else len(self.messages)), 0
        for i in range(len(self.messages) - 1, 0, -1):
            acc += size(self.messages[i])
            if acc > keep:
                break
            if self.messages[i]["role"] in ("user", "assistant"):
                cut_at = i
        drained = context.drop_turns(self.messages, cut_at)
        after = int(self._token_scale * (fixed + sum(size(m) for m in drained)))
        dropped = len(self.messages) - len(drained)
        self.messages = drained
        if self.append_only:
            self._insert_memory()
        self._insert_graph()
        self._after_rewrite("drain")
        self.stats["drains"] += 1
        self._log_event({"event": "drain", "reason": reason, "tokens_before": tokens_before, "tokens_after_est": after,
                         "keep_tokens": keep, "messages_dropped": dropped, "messages": len(self.messages),
                         "action_count": self.arc.game.action_count})
        if after > target:
            self._trim_context()

    def _refresh_after_compaction(self, kept: list[dict[str, Any]]) -> None:
        """Makes sure the full view of the current state survives a trim. Called from _trim_context."""
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
        """The current state in full, as a logged message. Called from _refresh_after_compaction and _reset_context."""
        msg = self._observation(full=True)
        self._log_event({"event": "message", **context.loggable(msg)})
        return msg

    def _trim_context(self) -> None:
        """Cuts the history back to the newest full view and blanks old tool outputs. Called from _on_overflow and _drain."""
        before = self._context_tokens()
        msgs = self.messages
        full = next((i for i in range(len(msgs) - 1, -1, -1)
                     if msgs[i].get("_kind") == "observation" and msgs[i].get("_full")), None)
        kept = list(msgs[full:] if full is not None else msgs[max(context.chain_start(msgs) - 1, 0):])
        last = max((i for i, m in enumerate(kept) if m["role"] == "assistant"), default=len(kept))
        for i, m in enumerate(kept):
            if m["role"] == "assistant" and i != last and m.get("_reasoning"):
                kept[i] = {k: v for k, v in m.items() if k != "_reasoning"}
            elif m["role"] == "tool" and i < last:
                first = next((ln.strip() for ln in compaction._text(m).splitlines() if ln.strip()), "")
                kept[i] = {**{k: v for k, v in m.items() if k != "_spans"},
                           "content": f"[output removed to fit the context; it began: {first[:120]}]"}
        self._refresh_after_compaction(kept)
        self.messages = [m for m in kept if m.get("_kind") != context.GRAPH_KIND]
        if self.append_only:
            self.messages = [m for m in self.messages if not context.is_memory(m)]
            self._insert_memory()
        self._insert_graph()
        self._after_rewrite("trim")
        self.stats["context_trims"] += 1
        self._log_event({"event": "context_trim", "tokens_before": before, "tokens_after": self._context_tokens(),
                         "messages_before": len(msgs), "messages_after": len(self.messages),
                         "action_count": self.arc.game.action_count})

    def _reset_context(self, reason: str, text: str) -> None:
        """A clean context: one message (level-up or overflow note), the memory, and the current full view. Called from _on_overflow and _reset_if_level_up."""
        obs = next((m for m in reversed(self.messages) if m.get("_kind") == "observation"), None)
        if obs is None or not obs.get("_full") or obs.get("_step") != self._last_step:
            obs = self._observation_refresh()
        names = self._kernel_names()
        dropped = len(self.messages)
        self.messages = []
        self._append({"role": "user", "_kind": "reset", "content": text + (f"\n\n{names}" if names else "")})
        if self.append_only and self.memory_in_context:
            self._append(self._memory_full())
        if self.code_graph and self._graph:
            self.stats["graph_snapshots"] += 1
            self._append(context.graph_message(codegraph.render(self._graph)))
        self.messages.append(obs)   # logged when it was made
        self._after_rewrite(reason)
        self.stats["context_resets"] += 1
        self._log_event({"event": "context_reset", "reason": reason, "messages_dropped": dropped,
                         "tokens_after": self._context_tokens(), "action_count": self.arc.game.action_count})

    def _after_rewrite(self, reason: str) -> None:
        """After the context was rewritten: logs it whole (tracing), drops the measured prompt size and tells the
        scheduler this game's cache is cold. Called from _drain, _trim_context and _reset_context."""
        if self.tracing:
            self._log_event({"event": "context", "reason": reason,
                             "messages": [context.snapshot(m) for m in self.messages]})
        self._pinned, self._usage_tokens, self._usage_at = None, None, 0
        self._full_shown_step = None   # old cell outputs may be cut: the next observe() shows the full view again
        mark = getattr(self.llm, "mark_cold", None)
        if callable(mark):
            mark()

    # --- host requests from the REPL ----------------------------------------------------------------------
    def _host(self, req: dict[str, Any]) -> dict[str, Any]:
        """Requests from the kernel (runs on its host thread): arc.act is one move sent by play() inside a cell, checked
        by the guardrails and run by _tool_act; arc.surprise means play() saw a wrong prediction. Called from _run_call."""
        kind = str(req.get("type", ""))
        if kind == "arc.surprise":   # play() stopped at a move the agent's model got wrong
            self._surprised = True
            self.stats["surprises"] += 1
            self._log_event({"event": "surprise", **(req.get("args") or {})})
            return {}
        if kind != "arc.act":
            guardrails.refuse_host(kind)
        cell = self._cell
        args = dict(req.get("args") or {})
        if cell is not None:
            self._act_via = args.pop("via", None)   # the kernel helper that sent it ("play"), for the trace
            self._act_predicted = args.pop("predicted", None)   # whether the agent predicted this move
            self._act_play_id = args.pop("play_id", None)       # which play() call of the game sent it
        guardrails.check_play(self, cell, args.pop("path_len", None))
        if "memory" in args:   # the memory variables, saved before the moves run
            self._sync_memory(args.pop("memory"), cell)
        guardrails.check_cell_open(cell)
        self._act_info = None
        before = self._frame()
        try:
            text = self._tool_act(args, cell["call_id"], cell["turn"], cell["reply"])
        except Exception as exc:
            self.stats["tool_errors"] += 1
            if isinstance(exc, (ValueError, TypeError)):
                self.stats["act_arg_errors"] += 1
            self._log_event({"event": "tool_error", "turn": cell["turn"], "call": cell["call_id"], "tool": "act",
                             "error": str(exc)[:500], "args": args})
            raise
        info = self._act_info or {}
        moves = args.get("actions")
        stop = guardrails.batch_stop(self, cell, moves[0] if isinstance(moves, list) and len(moves) == 1 else moves,
                                     before, self._frame(), info)
        cell["acts"].append(text)
        if info.get("level_up") or self.arc.finished or info.get("state") == "WIN":
            cell["closed"] = "the level changed" if info.get("level_up") else "the game ended"
        return {"text": text, **info, "batch_stop": stop}

    # --- bookkeeping -------------------------------------------------------------------------------------
    def _append(self, msg: dict[str, Any], **extra: Any) -> None:
        self.messages.append(msg)
        self._log_event({"event": "message", **context.loggable(msg), **{k: v for k, v in extra.items() if v}})

    def _log_event(self, obj: dict[str, Any]) -> None:
        if not self.tracing:
            return
        obj = {"t": round(time.time(), 2), **obj}
        try:
            with self.transcript.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(obj, default=str) + "\n")
        except Exception:  # noqa: BLE001
            pass
