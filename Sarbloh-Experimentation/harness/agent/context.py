"""Context builder: what is sent to the server on every turn.

Append-only (config ``append_only``, the default): the prompt only grows between rewrites, so each request starts with
the whole previous one and the server's prefix cache keeps it. On this hybrid model (Gated DeltaNet + attention) a
change anywhere in the history makes the server prefill everything again from about that point: in the 2026-10-04
run, every request after a new board reused ~0% of its prompt (3283 of 5599 requests) and requests inside a tool-call
chain reused 90%+. So nothing already sent is changed until a rewrite (a drain, a compaction, a level-up reset):
- every reply keeps its thinking and every picture stays;
- the memory is a stored message: in full at a level start and after every rewrite (``pinned_message``), then after
  each new board only the sections that changed (``memory_update``), and in full again every few boards;
- at the compaction trigger the agent first drains (``drain``): the newest messages stay word for word, thinking
  included, so the agent carries on mid-thought; before them the old boards become one-line stubs, the ipython outputs
  are cut short, act results become a short move log (one cut line per step), and the thinking, the old pictures and
  the memory messages go; a full memory goes back in. A summary is made only when a drain would not free enough.

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
STUB_KIND = "board_stub"           # an old board after a drain: one line, no picture
ACT_LINE_CHARS = 200               # an old act result's change line after a drain
REASONING_BUDGET_TOKENS = 16000   # the thinking re-sent in the current chain (config chain_reasoning_tokens)
ORDER = ("goal", "plan", "skills", "hypotheses", "findings", "lessons", "questions")
TITLES = {
    "goal": "Goal (you write it; kept across levels)",
    "plan": "Plan (from your last act)",
    "skills": "Skills (background knowledge)",
    "hypotheses": "Hypotheses (status set by you)",
    "findings": "Findings this level",
    "lessons": "Lessons from earlier levels and games",
    "questions": "Open questions (from a review of your last steps)",
}
EMPTY = {"goal": "(none yet: write one with act's `goal` as soon as you have a guess)",
         "plan": "(none yet)", "hypotheses": "(none yet)"}


def _section(blocks: dict[str, str], key: str) -> str:
    return (blocks.get(key) or "").strip() or EMPTY.get(key, "")


def pinned_message(blocks: dict[str, str], status: str) -> dict[str, Any]:
    parts = ["[memory] Your memory, from what you wrote with `act`. When a section changes, the new version is shown "
             "with the next state. It is not a message to answer.", f"## Game\n{status}"]
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
    parts = ["[memory update] These sections of your memory changed; the others are as shown before."]
    parts += [f"## {TITLES[key]}\n{_section(blocks, key) or '(empty)'}" for key in keys]
    return {"role": "user", "_kind": MEMORY_UPDATE_KIND, "content": "\n\n".join(parts)}


def is_memory(m: dict[str, Any]) -> bool:
    return m.get("_kind") in (PINNED_KIND, MEMORY_UPDATE_KIND)


def _parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [dict(p) for p in content]
    return [{"type": "text", "text": str(content or "")}]


def _has_image(m: dict[str, Any]) -> bool:
    return isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"])


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
            "content": f"[{head}: the board is removed to save space; the change lines are in the act result]"}


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


def drain(messages: list[dict[str, Any]], keep_from: int | None = None,
          output_chars: int | None = None) -> list[dict[str, Any]]:
    """A smaller history with no model call. The messages from ``keep_from`` on (default: the current tool-call
    chain) stay word for word, thinking included. Before them: every reply loses its thinking (its tool calls stay),
    every board but the newest becomes a one-line stub, every ipython output is cut to ``output_chars`` and every act
    result becomes a short move log (``short_act_result``); None keeps boards and outputs whole. Every picture but the
    newest becomes a stub, and the memory messages go (the caller puts a full one back)."""
    start = chain_start(messages) if keep_from is None else keep_from
    newest = max((i for i, m in enumerate(messages) if m.get("_kind") == "observation"), default=None)
    out = []
    for i, m in enumerate(messages[:start]):
        if is_memory(m):
            continue
        if m["role"] == "assistant":
            m = {k: v for k, v in m.items() if k != "_reasoning"}
        elif output_chars is not None and m.get("_kind") == "observation" and i != newest:
            m = board_stub(m)
        elif output_chars is not None and m["role"] == "tool":
            m = short_act_result(m) if is_act_result(m) else cut_output(m, output_chars)
        out.append(m)
    out += [m for m in messages[start:] if not is_memory(m)]
    return images(out, True)


def _text_of(m: dict[str, Any]) -> str:
    c = m.get("content") or ""
    return c if isinstance(c, str) else "".join(p.get("text", "") for p in c if isinstance(p, dict))


def build(system: str, messages: list[dict[str, Any]], pinned: dict[str, Any] | None = None,
          vision: bool = True, reasoning_budget: int = REASONING_BUDGET_TOKENS,
          append_only: bool = False) -> list[dict[str, Any]]:
    """The request. Append-only: the system prompt and the stored conversation as it is, every thinking and every
    picture included (the memory is already in it). Else: the memory joined to the front of the newest user message,
    so everything before that message is the same as in the last request."""
    if append_only:
        msgs = all_reasoning(messages)
        return merge_users([{"role": "system", "content": system}] + (msgs if vision else images(msgs, False)))
    msgs = images(reasoning(messages, reasoning_budget), vision)
    if pinned:
        at = max(chain_start(msgs) - 1, 0)
        msgs = msgs[:at] + [pinned] + msgs[at:]
    return merge_users([{"role": "system", "content": system}] + msgs)


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
