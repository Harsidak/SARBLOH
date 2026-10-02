"""E008 context builder: what is sent to the server on every turn.

    [system] -> [memory: game status, goal, plan, skills, hypotheses, findings, lessons, open questions] -> recent turns

The memory message is rebuilt before every turn from ``prime.memory`` and is never stored in the transcript or
summarised by compaction: it is the agent's memory, not its conversation. The observation (change lines, scene text,
image) is not pinned here: it is pushed as a user message right after each act result, so the newest state is always
the last thing the agent reads. Only the newest image is kept; older image parts become a one-line text stub (the
server allows one image per prompt). Consecutive user messages are joined (some chat templates need alternating roles).
"""

from __future__ import annotations

from typing import Any

PINNED_KIND = "pinned"
ORDER = ("goal", "levels", "plan", "skills", "hypotheses", "findings", "lessons", "wrong", "questions")
TITLES = {
    "goal": "Goal (you write it; kept across levels)",
    "levels": "Level reviews (written after each level you won)",   # E022; empty with the switch off
    "plan": "Plan (from your last act)",
    "skills": "Skills (background knowledge)",
    "hypotheses": "Hypotheses (status set by you)",
    "findings": "Findings this level",
    "lessons": "Lessons from earlier levels and games",
    "wrong": "Wrong rulebook (rules shown false: do not propose them again)",   # E018; empty with the switch off
    "questions": "Open questions (from a review of your last steps)",
}
EMPTY = {"goal": "(none yet: write one with act's `goal` as soon as you have a guess)",
         "plan": "(none yet)", "hypotheses": "(none yet)"}


def pinned_message(blocks: dict[str, str], status: str) -> dict[str, Any]:
    parts = ["[memory] Rebuilt before every turn from what you wrote with `act`. It is your memory, not a message "
             "to answer.", f"## Game\n{status}"]
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


def build(system: str, messages: list[dict[str, Any]], pinned: dict[str, Any] | None = None,
          vision: bool = True) -> list[dict[str, Any]]:
    head = [{"role": "system", "content": system}] + ([pinned] if pinned else [])
    return merge_users(head + images(messages, vision))


def loggable(msg: dict[str, Any]) -> dict[str, Any]:
    """A message for the transcript: image data replaced by a marker (the recording keeps every frame)."""
    if not _has_image(msg):
        return msg
    return {**msg, "content": [p if p.get("type") != "image_url" else
                               {"type": "image_url", "image_url": {"url": f"[png of step {msg.get('_image_step')}]"}}
                               for p in msg["content"]]}
