"""Context builder: what is sent to the server on every turn.

Append-only (config ``append_only``, the default): the prompt only grows between rewrites, so each request starts with
the whole previous one and the server's prefix cache keeps it. On this hybrid model (Gated DeltaNet + attention) a
change anywhere in the history makes the server prefill everything again from about that point: in the 2026-10-04
run, every request after a new board reused ~0% of its prompt (3283 of 5599 requests) and requests inside a tool-call
chain reused 90%+. So nothing already sent is changed until a rewrite (a drain, a compaction, a level-up reset):
- every reply keeps its thinking (with config ``old_images`` on, every picture also stays where it was sent);
- the memory is a stored message: in full at a level start and after every rewrite (``pinned_message``), then after
  each new board only the sections that changed (``memory_update``), and in full again every few boards;
- the picture of the board (config ``old_images`` off) is its own stored user message, added once per new state
  right after the output that changed it, and kept where it is until a rewrite. It is never sent last and then taken
  away: the server keeps the hybrid model's recurrent state only at the end of an earlier request, so a request that
  drops anything the previous one ended with is prefilled from the system prompt again (2026-10-19, picture at the
  tail: 4416 cached tokens on every request, 10% of the prompt);
- each ipython output ends with a one-line change note of the code graph (the agent's live functions and what they
  call, ``harness/runtime/skills/codegraph.py``); the old lines stay as they were, so the prompt keeps its prefix;
- at the compaction trigger the agent first drains (``drain``): the newest messages stay word for word, thinking
  included, so the agent carries on mid-thought; before them the thinking goes and the outputs are masked
  (``short_cell_output``): the code itself and every error traceback stay, a printed board becomes a one-line stub, the
  state shown after moves becomes a one-line stub, the rest of an output is cut short, act results become a short move
  log (one cut line per step), the code-graph change lines go, and the memory and code-graph messages go; a full
  memory and the whole code graph go back in once. If that is not enough, the history is trimmed.

The old builder (``append_only`` off): the memory is built once per user message and joined to the front of the newest
one; only the newest image is kept (older image parts become a one-line text stub); thinking is sent back only within
the current tool-call chain, newest reply first and up to about 16k tokens in all.

Either way the observation (change lines, state text, picture) is a user message right after each act result, so the
newest state is the last thing the agent reads, and consecutive user messages are joined (some chat templates need
alternating roles).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

PINNED_KIND = "pinned"
MEMORY_UPDATE_KIND = "memory_update"
GRAPH_KIND = "code_graph"          # the whole code graph, put back once after a drain, trim or reset
TAIL_KIND = "tail"                 # the old picture sent last and never stored; kept so older transcripts rebuild
PICTURE_KIND = "picture"           # the picture of the board after a new state, a stored message
STUB_KIND = "board_stub"           # an old board after a drain: one line, no picture
ACT_LINE_CHARS = 200               # an old act result's change line after a drain
ERROR_CHARS = 1500                 # an old error traceback after a drain keeps this many characters (its end)
BOARD_RUN = 4                      # this many board-like lines in a row are a printed board
# What a stored reply holds in place of its code (config keep_code off) until its cells ran; then the code graph of
# what the code defined (code_note).
CODE_STUB = "# code not kept in the messages"
_FENCED = re.compile(r"```(?:python|py|ipython|repl)?[ \t]*\n(.*?)```", re.DOTALL)   # = guardrails._FENCED
REASONING_BUDGET_TOKENS = 16000   # the thinking re-sent in the current chain (config chain_reasoning_tokens)
ORDER = ("goal", "plan", "skills", "hypotheses", "findings", "review")
TITLES = {
    "goal": "Goals by level (GOAL)",
    "plan": "Plans by level (PLAN)",
    "skills": "Skills you loaded (SKILLS; print(SKILLS[name]) shows one)",
    "hypotheses": "Hypotheses by level (HYPOTHESES)",
    "findings": "Findings by level (FINDINGS)",
    "review": "The reviewer's advice on this level (kept until you win it)",
}
EMPTY = {"goal": "(none yet: write one in GOAL as soon as you have a guess)",
         "plan": "(none yet)", "hypotheses": "(none yet)"}


def _section(blocks: dict[str, str], key: str) -> str:
    return (blocks.get(key) or "").strip() or EMPTY.get(key, "")


def pinned_message(blocks: dict[str, str], status: str) -> dict[str, Any]:
    parts = ["[memory] Your memory, don't reply to this message."]
    for key in ORDER:
        text = _section(blocks, key)
        if text:
            parts.append(f"## {TITLES[key]}\n{text}")
    return {"role": "user", "_kind": PINNED_KIND, "content": "\n\n".join(parts)}


def changed_sections(blocks: dict[str, str], shown: dict[str, str]) -> list[str]:
    """The memory sections whose text differs from the version last shown, in memory order."""
    return [key for key in ORDER if _section(blocks, key) != _section(shown, key)]


def memory_update(blocks: dict[str, str], keys: list[str]) -> dict[str, Any]:
    """Only the sections in ``keys``, as they are now; the others are as last shown."""
    parts = ["[memory update] These sections of your memory changed."]
    parts += [f"## {TITLES[key]}\n{_section(blocks, key) or '(empty)'}" for key in keys]
    return {"role": "user", "_kind": MEMORY_UPDATE_KIND, "content": "\n\n".join(parts)}


def is_memory(m: dict[str, Any]) -> bool:
    return m.get("_kind") in (PINNED_KIND, MEMORY_UPDATE_KIND)


def graph_message(text: str) -> dict[str, Any]:
    """The whole code graph as one message, put back after a drain, trim or reset (the change lines before it are
    gone by then)."""
    return {"role": "user", "_kind": GRAPH_KIND,
            "content": f"[code graph] Your functions and classes as they are in your kernel now. Later outputs end with "
                       f"a line of what changed; graph() prints the current map.\n{text}"}


def _parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [dict(p) for p in content]
    return [{"type": "text", "text": str(content or "")}]


def _has_image(m: dict[str, Any]) -> bool:
    return isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"])


has_image = _has_image


def _stub(m: dict[str, Any], why: str) -> dict[str, Any]:
    parts = []
    for p in m["content"]:
        if p.get("type") == "image_url":
            parts.append({"type": "text", "text": f"[image of step {m.get('_image_step')} {why}]"})
        else:
            parts.append(p)
    return {**m, "content": parts}


def images(messages: list[dict[str, Any]], vision: bool) -> list[dict[str, Any]]:
    """Every image but the newest becomes a stub; with ``vision`` off, every image does."""
    last = max((i for i, m in enumerate(messages) if _has_image(m)), default=None) if vision else None
    return [(_stub(m, "dropped: only the newest image is kept" if vision else "not shown: images are off")
             if _has_image(m) and i != last else m) for i, m in enumerate(messages)]


def merge_users(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Private ``_`` keys dropped and consecutive user messages joined; text stays a string unless a part is an
    image, then the joined content is a list of parts."""
    out: list[dict[str, Any]] = []
    for m in messages:
        m = {k: v for k, v in m.items() if not k.startswith("_")}
        if out and m["role"] == "user" and out[-1]["role"] == "user":
            prev = out[-1]["content"]
            if isinstance(prev, str) and isinstance(m["content"], str):
                out[-1] = {"role": "user", "content": f"{prev}\n\n{m['content']}"}
            else:
                a, b = _parts(prev), _parts(m["content"])
                if a and b and a[-1].get("type") == "text" and b[0].get("type") == "text":
                    a[-1] = {"type": "text", "text": f"{a[-1]['text']}\n\n{b[0]['text']}"}
                    b = b[1:]
                out[-1] = {"role": "user", "content": a + b}
        else:
            out.append(m)
    return out


def chain_start(messages: list[dict[str, Any]]) -> int:
    """Index just after the newest user message: the replies from here on are the current tool-call chain."""
    return next((i + 1 for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "user"), 0)


def reasoning_tokens(m: dict[str, Any]) -> int:
    """chars/4 of a reply's thinking."""
    return -(-len(m.get("_reasoning") or "") // 4)


def reasoning_sent(messages: list[dict[str, Any]], budget_tokens: int = REASONING_BUDGET_TOKENS) -> list[int]:
    """Indices of the replies whose thinking is sent: replies with tool calls in the current chain, newest first, while
    their thinking adds up to at most ``budget_tokens`` (chars/4). The newest reply's thinking always goes. Older
    replies keep their tool calls and outputs but lose their thinking. The agent's token estimate uses this same rule."""
    out, total = [], 0
    for i in range(len(messages) - 1, chain_start(messages) - 1, -1):
        m = messages[i]
        if not (m.get("_reasoning") and m.get("tool_calls")):
            continue
        t = reasoning_tokens(m)
        if out and total + t > budget_tokens:
            break
        out.append(i)
        total += t
    return out


def reasoning(messages: list[dict[str, Any]], budget_tokens: int = REASONING_BUDGET_TOKENS) -> list[dict[str, Any]]:
    """The agent's thinking goes back with its tool calls in the current chain (what the chat templates expect), up to
    the thinking budget (``reasoning_sent``). Once a new user message arrives (an observation after an act), the older
    thinking is no longer sent."""
    sent = set(reasoning_sent(messages, budget_tokens))
    return [({**m, "reasoning": m["_reasoning"], "reasoning_content": m["_reasoning"]} if i in sent else m)
            for i, m in enumerate(messages)]


def all_reasoning(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every stored reply goes back with its thinking (the append-only builder)."""
    return [({**m, "reasoning": m["_reasoning"], "reasoning_content": m["_reasoning"]} if m.get("_reasoning") else m)
            for m in messages]


def is_act_result(m: dict[str, Any]) -> bool:
    """A tool result of `act`: its change lines, one per step, are the move log (a drain shortens it, never drops
    it)."""
    return m.get("role") == "tool" and str(m.get("content") or "").startswith("act")


def board_stub(m: dict[str, Any]) -> dict[str, Any]:
    """An old board as one line: its header ("[state after step #12]") and a note that the text is gone."""
    first = next((ln.strip() for ln in _text_of(m).splitlines() if ln.strip()), "")
    head = first[1:first.index("]")] if first.startswith("[") and "]" in first else "board"
    return {"role": "user", "_kind": STUB_KIND, "_step": m.get("_step"),
            "content": f"[{head}: the board is removed to save space; the change lines are in the act output]"}


def cut_output(m: dict[str, Any], chars: int) -> dict[str, Any]:
    """An ipython output cut to its first ``chars`` characters, its last line (the game status) kept."""
    text = _text_of(m)
    if len(text) <= chars:
        return m
    lines = text.rstrip().splitlines()
    tail = lines[-1] if lines and lines[-1].startswith("[game:") else ""
    return {**m, "content": f"{text[:chars].rstrip()}\n[... {len(text) - chars} more characters removed to save "
                            f"space]" + (f"\n{tail}" if tail else "")}


_MARKS = re.compile(r"(?: \[repeat: [^\]]*\]| \[GAME_OVER\])+$")


def short_act_result(m: dict[str, Any], line_chars: int = ACT_LINE_CHARS) -> dict[str, Any]:
    """An old act result as a move log: its first line, each change line cut to ``line_chars`` with its repeat and
    GAME_OVER marks kept, and its status line; the event notes after them (what to do after a GAME_OVER, ...) go."""
    out = []
    for ln in _text_of(m).splitlines():
        if ln.startswith("#") and len(ln) > line_chars:
            marks = _MARKS.search(ln)
            tail = marks.group(0) if marks else ""
            ln = ln[:max(line_chars - len(tail), 40)].rstrip() + " ..." + tail
        if ln.startswith(("act", "#", "[game:")):
            out.append(ln)
    return {**m, "content": "\n".join(out)}


_BOARD_LINE = re.compile(r"^\s*\d{0,3}\s?[A-Za-z0-9.#|]{8,70}\s*$")          # a row of letters, maybe labelled
_NUMPY_LINE = re.compile(r"^\s*\[*\s*(?:-?\d+[\s,]+){7,}-?\d*\s*\]*,?\s*$")    # a row of a printed array
_LONG_BOARD = re.compile(r"^[\s\[\]'\",A-Za-z0-9.]{400,}$")                    # a whole board printed as one list


def mask_boards(text: str) -> str:
    """``text`` with every printed board replaced by one line: ``BOARD_RUN`` or more board-like lines in a row (rows
    of colour letters, labelled or not, or rows of a printed array), or one very long line that is only a list of
    them."""
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        j = i
        while j < len(lines) and (_BOARD_LINE.match(lines[j]) or _NUMPY_LINE.match(lines[j])):
            j += 1
        if j - i >= BOARD_RUN:
            out.append(f"[a printed board of {j - i} rows, removed to save space]")
            i = j
        elif _LONG_BOARD.match(lines[i]) and sum(ch.isalpha() for ch in lines[i]) > len(lines[i]) // 2:
            out.append(f"[a printed board of {len(lines[i])} characters, removed to save space]")
            i += 1
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def cell_output(pieces: list[tuple[str, str]]) -> tuple[str, list[list[Any]]]:
    """The text of an ipython output from its pieces (kind, text), joined by newlines, and where each piece is in it:
    [[kind, start, end], ...]. The kinds: "out" (what the code printed), "error" (its traceback), "observe" (the full
    state an `observe()` call shows), "state" (the state after its moves), "memory" (the memory receipt), "graph" (the
    code-graph change line). A drain masks each kind its own way."""
    text, spans = "", []
    for kind, piece in pieces:
        if not piece:
            continue
        if text:
            text += "\n"
        spans.append([kind, len(text), len(text) + len(piece)])
        text += piece
    return text, spans


def _stub_line(text: str, what: str) -> str:
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    head = first[1:first.index("]")] if first.startswith("[") and "]" in first else what
    return f"[{head}: removed to save space]"


def short_cell_output(m: dict[str, Any], chars: int) -> dict[str, Any]:
    """An old ipython output as a drain leaves it: what the code printed has its printed boards masked
    (``mask_boards``) and is then cut to ``chars``; the error traceback stays (its last ``ERROR_CHARS``); the states
    shown become one-line stubs; the code-graph change line goes (the whole graph goes back after the drain); the
    memory receipt and the status line stay. Its moves are added as a move log when it made any."""
    text = _text_of(m)
    out = []
    for kind, a, b in m["_spans"]:
        piece = text[a:b]
        if kind == "out":
            piece = mask_boards(piece)
            if len(piece) > chars:
                piece = f"{piece[:chars].rstrip()}\n[... {len(piece) - chars} more characters removed to save space]"
        elif kind == "error":
            piece = piece if len(piece) <= ERROR_CHARS else f"[... start of the traceback removed]\n{piece[-ERROR_CHARS:]}"
        elif kind in ("observe", "state"):
            piece = _stub_line(piece, "state")
        elif kind == "graph":
            continue
        out.append(piece)
    if m.get("_act_log"):
        out.append("[moves made by this call]\n" + short_act_result({"content": m["_act_log"]})["content"])
    content = "\n".join(out)
    return {k: v for k, v in m.items() if k != "_spans"} | {"content": content}


def drain(messages: list[dict[str, Any]], keep_from: int | None = None,
          output_chars: int | None = None) -> list[dict[str, Any]]:
    """A smaller history with no model call. The messages from ``keep_from`` on (default: the current tool-call
    chain) stay word for word, thinking included. Before them: every reply loses its thinking (its code stays), every
    board but the newest becomes a one-line stub, every ipython output is masked (``short_cell_output``: printed boards
    and shown states become stubs, errors stay, the rest is cut to ``output_chars``) and every act result becomes a
    short move log (``short_act_result``); None keeps boards and outputs whole. Every picture but the newest becomes a
    stub, and the memory and code-graph messages go (the caller puts a full one of each back)."""
    start = chain_start(messages) if keep_from is None else keep_from
    newest = max((i for i, m in enumerate(messages) if m.get("_kind") == "observation"), default=None)
    out = []
    for i, m in enumerate(messages[:start]):
        if is_memory(m) or m.get("_kind") == GRAPH_KIND:
            continue
        if m["role"] == "assistant":
            m = {k: v for k, v in m.items() if k != "_reasoning"}
        elif output_chars is not None and m.get("_kind") == "observation" and i != newest:
            m = board_stub(m)
        elif output_chars is not None and m["role"] == "tool":
            if m.get("_spans"):
                m = short_cell_output(m, output_chars)
            elif m.get("_act_log"):   # an ipython cell that moved: its output cut short, then its move log whole
                log = short_act_result({"content": m["_act_log"]})["content"]
                cut = _text_of(cut_output(m, output_chars))
                m = {**m, "content": f"{cut}\n[moves made by this call]\n{log}"}
            elif is_act_result(m):
                m = short_act_result(m)
            else:
                m = cut_output(m, output_chars)
        out.append(m)
    out += [m for m in messages[start:] if not is_memory(m) and m.get("_kind") != GRAPH_KIND]
    return images(out, True)


def drop_turns(messages: list[dict[str, Any]], keep_from: int) -> list[dict[str, Any]]:
    """A smaller history with no model call, as the Duck harness makes it: the first message (the task, or the note of
    the last reset) and every message from ``keep_from`` on, word for word; the turns in between are dropped whole.
    The memory and code-graph messages go (the caller puts a full one of each back), and every picture but the newest
    becomes a stub."""
    head = messages[:1] if keep_from > 0 else []
    out = [m for m in head + messages[max(keep_from, 0):] if not is_memory(m) and m.get("_kind") != GRAPH_KIND]
    return images(out, True)


def _text_of(m: dict[str, Any]) -> str:
    c = m.get("content") or ""
    return c if isinstance(c, str) else "".join(p.get("text", "") for p in c if isinstance(p, dict))


def build(system: str, messages: list[dict[str, Any]], pinned: dict[str, Any] | None = None,
          vision: bool = True, reasoning_budget: int = REASONING_BUDGET_TOKENS,
          append_only: bool = False, old_images: bool = True,
          tail: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The request. Append-only: the system prompt and the stored conversation as it is, every thinking included (the
    memory is already in it), and every stored picture, or with ``old_images`` off only the newest one (the older ones
    become a one-line stub); then ``tail``, the picture of the board now, as the last message, so a new picture never
    changes the history before it. Else: the memory joined to the front of the newest user message, so everything
    before that message is the same as in the last request."""
    if append_only:
        msgs = all_reasoning(messages)
        msgs = msgs if vision and old_images else images(msgs, vision)
        return merge_users([{"role": "system", "content": system}] + msgs + ([tail] if tail and vision else []))
    msgs = images(reasoning(messages, reasoning_budget), vision)
    if pinned:
        at = max(chain_start(msgs) - 1, 0)
        msgs = msgs[:at] + [pinned] + msgs[at:]
    return merge_users([{"role": "system", "content": system}] + msgs)


def code_note(graph: str | None) -> str:
    """What a stored reply holds in place of one call's code: the code-graph lines of what that code defined, changed
    or removed (``codegraph.cell_map``), or ``CODE_STUB`` when the graph is not known (the call did not run, or config
    code_graph is off)."""
    if graph is None:
        return CODE_STUB
    if not graph:
        return "# code not kept in the messages; it defined no functions or classes"
    return "# code not kept in the messages; the code graph of what it defined:\n" + \
        "\n".join(f"# {line}" for line in graph.splitlines())


def without_code(msg: dict[str, Any], graphs: dict[str, str] | None = None) -> dict[str, Any]:
    """A reply as it is stored with config keep_code off: the code of each ipython call (and the ```python blocks of a
    fenced reply, which run as one call) becomes ``code_note`` of that call's entry in ``graphs`` (call id -> graph
    lines); the thinking, the text and the call ids stay."""
    graphs = graphs or {}
    out = dict(msg)
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        if fn.get("name") == "ipython":
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (ValueError, TypeError):
                args = {}
            args = {**(args if isinstance(args, dict) else {}), "code": code_note(graphs.get(tc.get("id")))}
            tc = {**tc, "function": {**fn, "arguments": json.dumps(args)}}
        calls.append(tc)
    if calls:
        out["tool_calls"] = calls
    elif isinstance(msg.get("content"), str) and _FENCED.search(msg["content"]):
        note = code_note(next(iter(graphs.values()), None))
        blocks = iter(range(10 ** 6))   # the first block holds the note, the others go
        out["content"] = _FENCED.sub(lambda m: f"```python\n{note}\n```" if next(blocks) == 0 else "", msg["content"])
    return out


def loggable(msg: dict[str, Any]) -> dict[str, Any]:
    """A message for the transcript: image data replaced by a marker (the recording keeps every frame); the thinking
    is logged once, as the event's ``reasoning``."""
    msg = {k: v for k, v in msg.items() if k != "_reasoning"}
    if not _has_image(msg):
        return msg
    return {**msg, "content": [p if p.get("type") != "image_url" else
                               {"type": "image_url", "image_url": {"url": f"[png of step {msg.get('_image_step')}]"}}
                               for p in msg["content"]]}


def snapshot(msg: dict[str, Any]) -> dict[str, Any]:
    """A stored message as ``loggable`` but with its thinking kept (``_reasoning``): one entry of the trace's context
    snapshot, taken after every rewrite."""
    out = loggable(msg)
    if msg.get("_reasoning"):
        out["_reasoning"] = msg["_reasoning"]
    return out


def digest(request: list[dict[str, Any]]) -> str:
    """A short hash of a request with every picture reduced to its part type: the transcript keeps the pictures as
    files, not bytes, so the trace can rebuild each request from the transcript and check it against this."""
    canon = [{**m, "content": [{"type": "image_url"} if p.get("type") == "image_url" else p for p in m["content"]]}
             if isinstance(m.get("content"), list) else m for m in request]
    return hashlib.sha256(json.dumps(canon, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]
