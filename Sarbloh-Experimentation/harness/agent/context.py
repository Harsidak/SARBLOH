"""Context builder: what is sent to the server on every turn.

    [system] -> older turns -> [memory + newest user message (the newest state)] -> the current tool-call chain

The memory (game status, goal, plan, skills, hypotheses, findings, lessons, open questions) is built from
``harness.memory`` once per user message and goes in just before the newest one. It is never stored in the
transcript or summarised by compaction: it is the agent's memory, not its conversation. It sits near the end, not
after the system prompt, so the server's prefix cache keeps everything before it: a memory block at the top changed
on almost every request and forced the whole conversation to be prefilled again. Inside the newest user message (not
after the chain) it keeps the chat template sending the chain's thinking back. The observation (change lines, state
text, picture) is not part of the memory: it is pushed as a user message right after each act result, so the newest
state is always the last thing the agent reads. Only the newest image is kept; older image parts become a one-line
text stub (the server allows one image per prompt). Consecutive user messages are joined (some chat templates need
alternating roles).
The agent's thinking is sent back only within the current tool-call chain, up to the next user message.
"""

from __future__ import annotations

from typing import Any

PINNED_KIND = "pinned"
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


def pinned_message(blocks: dict[str, str], status: str) -> dict[str, Any]:
    parts = ["[memory] Your memory, from what you wrote with `act`; it is shown again with every new state. It is "
             "not a message to answer.", f"## Game\n{status}"]
    for key in ORDER:
        text = (blocks.get(key) or "").strip() or EMPTY.get(key, "")
        if text:
            parts.append(f"## {TITLES[key]}\n{text}")
    return {"role": "user", "_kind": PINNED_KIND, "content": "\n\n".join(parts)}


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


def reasoning(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The agent's thinking goes back with its tool calls in the current chain (what Gemma's chat template expects).
    Once a new user message arrives (an observation after an act), the older thinking is no longer sent."""
    start = chain_start(messages)
    return [({**m, "reasoning": m["_reasoning"], "reasoning_content": m["_reasoning"]}
             if i >= start and m.get("_reasoning") and m.get("tool_calls") else m) for i, m in enumerate(messages)]


def build(system: str, messages: list[dict[str, Any]], pinned: dict[str, Any] | None = None,
          vision: bool = True) -> list[dict[str, Any]]:
    """The request: the system prompt, the conversation, and the memory joined to the front of the newest user
    message, so everything before that message is the same as in the last request."""
    msgs = images(reasoning(messages), vision)
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
