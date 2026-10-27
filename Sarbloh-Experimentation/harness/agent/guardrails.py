"""The guardrails of the agent loop: what refuses a move, the notes that push the model to act, the tool-call checks
and the hard stops. Each one takes the running AgentSession ``s`` (harness/agent/agent.py).

Where they are called from in ``AgentSession._loop`` and ``_request``:
1. stop?            ``should_stop`` (before every request of a turn)
4. store the reply  ``read_calls`` (unknown tool, broken JSON)
5. run the tools    ``check_play`` and ``read_moves`` for each move ``play()`` sends, ``seen_before`` (the revisit
                    leash) and ``batch_stop`` after it;
                    ``refuse_host`` for other requests; ``level_start_note`` after a level-up
Cut off, no tool:   ``cut_off_note``
Turn caps:          ``over_turn_budget`` (thinking tokens since the last move)
End of a turn:      ``end_turn`` (budget restart, ``stuck_note``, curator, the scheduler slot)
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from typing import Any

# pyrefly: ignore [missing-import]
from harness.agent import prompts

_FENCED = re.compile(r"```(?:python|py|ipython|repl)?[ \t]*\n(.*?)```", re.DOTALL)
_ACTION_NAMES = {0: "RESET", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "FIRE",
                 7: "UNDO"}
EDGE_BAND = 3                         # rows and columns at each edge the revisit leash does not compare
REVISIT_STOP = "board seen before"    # the stop reason play() reports
GRID_MAX = 63                         # the largest row and column of a click (the grid is at most 64x64)


# --- hard stops ---------------------------------------------------------------------------------------------------
def should_stop(s: Any) -> str | None:
    """Why the game must stop now (time, actions, turns, tokens, game over, LLM failures), or None. Called from _loop before each request."""
    if s.stop_event.is_set():
        return "stopped"
    if time.time() >= s.deadline:
        return "wall_clock"
    if s.arc.finished:
        return "game_finished"
    if s.arc.budget_left == 0:
        return "action_budget"
    lim = s.cfg["limits"]
    if s.stats["turns"] >= lim["max_turns"]:
        return "max_turns"
    if s.stats["output_tokens"] >= lim["max_output_tokens"]:
        return "max_output_tokens"
    if s.stats["llm_failures"] >= s.cfg["max_consecutive_llm_failures"]:
        return "llm_failures"
    return None


# --- tool-call checks ---------------------------------------------------------------------------------------------
def read_calls(s: Any, reply: Any) -> list[dict[str, Any]]:
    """Reads the tool calls out of a reply (or ```python blocks as a fallback); an unknown tool or broken JSON becomes
    the call's error, sent back to the model. Called from _request step 4."""
    calls = []
    names = [t["function"]["name"] for t in s.tools]
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
        s.stats["native_calls"] += len(calls)
        return calls
    # Fallbacks for models or servers without working tool-call parsing: a <tool_call> written in the reply's text,
    # then code in fenced blocks, then the last <tool_call> in the thinking (earlier ones there are drafts).
    recover = s.cfg.get("recover_tool_calls", False)
    codes = text_tool_calls(reply.content or "") if recover else []
    if codes:
        return _text_calls(s, codes)
    blocks = [b for b in _FENCED.findall(reply.content or "") if b.strip()]
    if blocks and s.cfg["allow_fenced_code"]:
        s.stats["fenced_calls"] += 1
        return [{"native": False, "raw": None, "name": "ipython", "args": {}, "code": "\n\n".join(blocks),
                 "error": None}]
    codes = text_tool_calls(getattr(reply, "reasoning", None) or "")[-1:] if recover else []
    return _text_calls(s, codes) if codes else []


_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_XML_CODE = re.compile(r"<function=ipython>\s*<parameter=code>\n?(.*?)\n?</parameter>", re.DOTALL)


def text_tool_calls(text: str) -> list[str]:
    """The ipython code of each <tool_call> written as text: {"name": "ipython", "arguments": {"code": ...}} (the
    arguments may be a JSON string) or <function=ipython><parameter=code>...</parameter></function>. A call to another
    tool, or one that cannot be read, is skipped. Used by read_calls."""
    codes = []
    for body in _TOOL_CALL.findall(text):
        xml = _XML_CODE.search(body)
        if xml:
            code = xml.group(1)
        else:
            try:
                obj = json.loads(body.strip())
                args = obj.get("arguments", obj.get("parameters")) if obj.get("name") == "ipython" else None
                args = json.loads(args) if isinstance(args, str) else args
                code = args.get("code") if isinstance(args, dict) else None
            except (ValueError, AttributeError):
                code = None
        if isinstance(code, str) and code.strip():
            codes.append(code)
    return codes


def _text_calls(s: Any, codes: list[str]) -> list[dict[str, Any]]:
    s.stats["recovered_calls"] = s.stats.get("recovered_calls", 0) + 1
    return [{"native": False, "raw": None, "name": "ipython", "args": {}, "code": "\n\n".join(codes), "error": None}]


# --- what refuses a move ------------------------------------------------------------------------------------------
def refuse_host(kind: str) -> None:
    """Refuses every kernel request other than a move or a surprise. Called from AgentSession._host (step 5)."""
    if kind.startswith("arc."):
        raise PermissionError("in this harness the state is read with `observe()` and moves are made with "
                              "`await play([...])`")
    raise RuntimeError(f"host request {kind!r} is not supported by this harness")


def check_play(s: Any, cell: dict[str, Any] | None, path_len: Any) -> None:
    """Before a move from play(): it must come from an ipython cell, and its whole path must fit act_max_actions
    (refused whole, nothing spent). Called from AgentSession._host (step 5)."""
    if cell is None:
        raise RuntimeError("play() works only inside an ipython call")
    cap = int(s.cfg.get("act_max_actions", 5))
    if path_len is not None and int(path_len) > cap:
        s.stats["tool_errors"] += 1
        s.stats["act_arg_errors"] += 1
        raise RuntimeError(prompts.event_message("act_too_many", sent=int(path_len), act_max=cap))


def new_life(s: Any) -> None:
    """A new try: the level started or was reset, and the board now is its first one. The revisit leash forgets its
    boards and the replay stop its moves. Called from AgentSession._tool_act (a level up, a reset) and _auto_reset."""
    key = _board_key(s._frame())
    s._life_boards = {key}
    s._life_start, s._life_moves = key, []


def record_loss(s: Any) -> None:
    """The try ended in GAME_OVER: its first board and its moves (the losing one last) are kept for the replay stop.
    Called from AgentSession._tool_act at a GAME_OVER, before the automatic reset."""
    s._lost_tries.add((s._life_start, tuple(s._life_moves)))


def check_replay(s: Any, moves: list[tuple[int, int | None, int | None]], path: Any = None) -> None:
    """The replay stop: refuses, before anything is sent, moves that would repeat from the start of this try the exact
    moves of a try that ended in GAME_OVER from the same first board. Same board and same moves give the same result,
    so the move that lost would lose again. ``path`` is the whole play() path (sent with its first move): then the
    whole path is checked and refused; else only ``moves``. A reset in the path ends the check (the next try starts
    there). Off with config replay_stop off. Called from AgentSession._tool_act."""
    if not s.cfg.get("replay_stop", False) or not s._lost_tries:
        return
    if path:
        read = []
        for m in path:
            try:
                read.append(parse_action(m))
            except ValueError:
                break   # read_moves refuses the unreadable move when it comes; check the moves before it
        moves = read or moves
    series = list(s._life_moves)
    for n, move in enumerate(moves):
        if move[0] == 0:
            return
        series.append(move)
        if (s._life_start, tuple(series)) in s._lost_tries:
            s.stats["replay_stops"] = s.stats.get("replay_stops", 0) + 1
            raise ValueError(prompts.event_message("act_replay", at=n + 1, label=label(*move), total=len(series)))


def seen_before(s: Any) -> bool:
    """The revisit leash: records the board after a move and says whether it was already seen in this try (since the
    level started or was last reset). In a game with fixed rules the shortest way to a goal never passes the same
    board twice, so a repeat is a loop, or a move that changed nothing. The edge band is not compared: a move bar or a
    counter there changes on every move. Always False with config revisit_stop off. Called from AgentSession._tool_act."""
    if not s.cfg.get("revisit_stop", False):
        return False
    key = _board_key(s._frame())
    if key in s._life_boards:
        return True
    s._life_boards.add(key)
    return False


def _inner(rows: list[list[int]], band: int) -> list[list[int]]:
    """The board without ``band`` rows and columns at each edge (the whole board if it is too small or band is 0)."""
    if band > 0 and len(rows) > 2 * band and all(len(r) > 2 * band for r in rows):
        return [r[band:-band] for r in rows[band:-band]]
    return rows


def _board_key(rows: list[list[int]]) -> bytes:
    return hashlib.md5(json.dumps(_inner(rows, EDGE_BAND)).encode()).digest()


def batch_stop(s: Any, cell: dict[str, Any], move: Any, before: Any, after: Any, info: dict[str, Any]) -> str | None:
    """Why the rest of a play() path must not be sent, or None. Only four things stop it: a level up, the game's
    end (GAME_OVER or WIN), a board seen before in this try (config revisit_stop, see seen_before) and a wall hit
    twice (the same move twice in a row with nothing changed outside the config wall_edge_band rows and columns at each
    edge, where move bars and counters sit; 0 compares the whole frame). A timer bar or an animation changing never
    stops it. Called from AgentSession._host after each move."""
    if info.get("level_up"):
        return "level up"
    if info.get("state") in ("GAME_OVER", "WIN") or s.arc.finished:
        return "GAME_OVER" if info.get("state") == "GAME_OVER" else "game finished"
    if info.get("revisit"):
        s.stats["revisits"] += 1
        return REVISIT_STOP
    band = int(s.cfg.get("wall_edge_band") or 0)
    if _inner(before, band) != _inner(after, band):
        cell["noop_move"] = None
        return None
    if cell.get("noop_move") == move:
        s.stats["wall_hits"] += 1
        return "wall hit twice"
    cell["noop_move"] = move
    return None


def check_cell_open(cell: dict[str, Any]) -> None:
    """After a level-up or the game's end in a cell, no more moves run in that cell. Called from AgentSession._host."""
    if cell["closed"]:
        raise RuntimeError(f"not run, nothing spent: {cell['closed']} in this cell already. Read the new state "
                           "in your next reply before you move again.")


def read_moves(s: Any, args: dict[str, Any]) -> list[tuple[int, int | None, int | None]]:
    """The moves of one act as (action, row, column), all or nothing: refused whole if the list is empty or over
    act_max_actions, an item cannot be read, it holds a reset while config reset_move is 0 or more resets than this
    level has left, or an action is not legal now. Called from AgentSession._tool_act."""
    raw = args.get("actions")
    if isinstance(raw, str):
        raw = json.loads(raw) if raw.strip().startswith("[") else [raw]
    elif isinstance(raw, int):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise ValueError(prompts.event_message("act_bad_args", problem="the path is empty or is not a list of moves", act_max=int(s.cfg.get("act_max_actions", 5))))
    cap = int(s.cfg.get("act_max_actions", 5))
    if len(raw) > cap:
        raise ValueError(prompts.event_message("act_too_many", sent=len(raw), act_max=cap))
    try:
        parsed = [parse_action(a) for a in raw]
    except ValueError as exc:
        raise ValueError(prompts.event_message("act_bad_args", problem=str(exc), act_max=int(s.cfg.get("act_max_actions", 5)))) from None
    resets = sum(1 for a, _, _ in parsed if a == 0)
    if resets:
        allowed = int(s.cfg.get("reset_move") or 0)
        if not allowed:
            raise ValueError(prompts.event_message("act_no_reset"))
        used = s._resets_used.get(s.arc.game.state.levels_completed, 0)
        if used + resets > allowed:
            raise ValueError(prompts.event_message("act_no_resets_left", allowed=allowed))
    legal = [a for a in s.arc.game.state.available_actions if a != 0]
    bad = [label(a, r, c) for a, r, c in parsed if a != 0 and a not in legal]
    if bad:
        raise ValueError(prompts.event_message("act_illegal", bad=", ".join(bad), legal=legal))
    return parsed


def parse_action(item: Any) -> tuple[int, int | None, int | None]:
    """(action id, row, column). "1", 1, "A1", "ACTION1" -> (1, None, None); "6 12 40", "A6(12,40)",
    {"action": 6, "r": 12, "c": 40} -> (6, 12, 40) (row 12, column 40); "reset" -> (0, None, None). Used by read_moves."""
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
        if not (0 <= nums[1] <= GRID_MAX and 0 <= nums[2] <= GRID_MAX):
            raise ValueError(f"action {item!r}: a click's row and column are 0 to {GRID_MAX}")
        return 6, nums[1], nums[2]
    if len(nums) != 1:
        raise ValueError(f"action {item!r}: one action per list item (only action 6 takes r c)")
    if nums[0] == 0:
        return 0, None, None
    if not 1 <= nums[0] <= 7:
        raise ValueError(f"action {item!r}: ids are 1 to 7, or \"reset\"")
    return nums[0], None, None


def label(a: int, r: int | None, c: int | None) -> str:
    """A move's name in the act log: "UP", "A6(r12,c40)". Used by read_moves and AgentSession._tool_act."""
    return f"A6(r{r},c{c})" if a == 6 else _ACTION_NAMES.get(a, f"A{a}")


# --- notes that push the model to act -----------------------------------------------------------------------------
def over_turn_budget(s: Any) -> bool:
    """True when the replies since the last move used up the turn's thinking budget, a cap that ends the turn. Called
    from _loop."""
    return bool(s._turn_budget) and s._turn_tokens >= s._turn_budget


def cut_off_note(s: Any) -> None:
    """The reply was cut at the length limit before it called a tool: the "call the tool now" note. The turn goes on and
    the scheduler slot is kept. Called from _request."""
    s.stats["cut_off_notes"] = s.stats.get("cut_off_notes", 0) + 1
    s._log_event({"event": "cut_off_note", "tokens": s._turn_tokens})
    s._push({"role": "user", "content": prompts.event_message("call_tool_now", **s._left())})


def end_turn(s: Any, reason: str, requests: int) -> None:
    """The end of one turn of the outer loop (``reason``: "moved", a cap, or the game stopped): restarts the thinking
    budget (the short one after a surprise), adds the stuck note, runs the stuck review and the curator when they are due, and gives the GPU
    slot to the best waiting game (this game queues again and may get it straight back). No note says to move: a turn
    that ends at a cap just goes on in the next one. Called from _loop."""
    s.stats["outer_turns"] += 1
    s.stats["turn_requests_max"] = max(s.stats["turn_requests_max"], requests)
    if reason.endswith("_cap"):   # the turn ended without a move
        s.stats["turn_yields"] += 1
    s._log_event({"event": "turn_end", "reason": reason, "requests": requests, "tokens": s._turn_tokens})
    s._turn_tokens, s._turn_budget = 0, s.turn_yield_tokens
    if reason == "moved" and s._surprised and s.surprise_yield_tokens:
        s._turn_budget = s.surprise_yield_tokens
        s.stats["surprise_turns"] += 1
    if s.end_reason:
        return
    stuck_note(s)
    s._maybe_review()
    s._maybe_curate()
    give = getattr(s.llm, "yield_turn", None)
    if callable(give):
        give()


def stuck_note(s: Any) -> None:
    """Every stuck_note_tokens spent on one level, adds a "question your goal" note. Called from end_turn."""
    if not s.stuck_note_tokens or s.arc.finished:
        return
    if s._level_tokens < s.stuck_note_tokens * (s._stuck_notes + 1):
        return
    s._stuck_notes += 1
    s.stats["stuck_notes"] += 1
    s._log_event({"event": "stuck_note", "level": s._level(), "tokens": s._level_tokens})
    s._push({"role": "user", "content": prompts.event_message("stuck", tokens=s._level_tokens // 1000, **s._left())})


def level_start_note(s: Any) -> None:
    """After a level-up, names the functions and classes already in the kernel so the agent reuses them. Called from
    _request step 5."""
    names = s._kernel_names(full=False)
    if names:
        s._push({"role": "user", "content": prompts.event_message("reuse_code", names=names)})
