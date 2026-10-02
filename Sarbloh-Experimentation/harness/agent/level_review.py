"""E022 level review (switch ``memory.level_review``): after every level-up, one hidden JSON call (thinking off, like
the curator) reads the whole level just won and writes what it shows into the memories.

Input: every step of the level (timeline rows, resets and GAME_OVERs included; clipped to ``steps_tokens`` by keeping
the first and the last steps), every plan the agent wrote in it, what its memory held when the level ended (goal,
hypotheses, findings) and the earlier reviews of this game. Output: how the level was won, the win condition, rules
(verified or refuted), wasted actions and up to three notes for the next level. ``GameMemory.add_level_review`` writes
them into the working memory (block "levels"), the lessons graph and ``levels.jsonl``. The review never edits the goal.
A reply that does not parse twice raises; the caller logs it and carries on.
"""

from __future__ import annotations

import json
from typing import Any

from harness.agent.refine import extract_json
from harness.memory.store import tokens
from harness.memory.timeline import render_row

REVIEW_SYSTEM = """You review one level of a turn-based grid game that an agent has just won. You never act. You return JSON only.

You get every step of the level (action and what changed), the agent's plans, and what the agent believed when the level ended. Find out what really happened:
1. summary: how the level was won, in one or two sentences. Name the actions that mattered (by step number).
2. win_condition: what made the level end in a win, as one general sentence (name objects by colour and shape).
3. rules: at most 5 rules the steps show. "verified" if a step shows it, "refuted" if a step shows it false. Never a guess.
4. mistakes: at most 3 kinds of action that cost steps without need (repeats, dead ends, GAME_OVERs), and why.
5. carry_over: at most 3 short notes for the next level: what to try first, what to avoid.

Return exactly:
{"summary": "...", "win_condition": "...", "rules": [{"text": "...", "status": "verified|refuted"}], "mistakes": ["..."], "carry_over": ["..."]}"""


def clip_steps(lines: list[str], cap_tokens: int) -> list[str]:
    """All lines if they fit ``cap_tokens``; else the first third and the rest from the end, with a note between."""
    if tokens("\n".join(lines)) <= cap_tokens:
        return lines
    budget = cap_tokens * 4 - 60
    head, tail = [], []
    for line in lines:
        if sum(len(x) + 1 for x in head) + len(line) + 1 > budget // 3:
            break
        head.append(line)
    for line in reversed(lines[len(head):]):
        if sum(len(x) + 1 for x in head + tail) + len(line) + 1 > budget:
            break
        tail.append(line)
    tail.reverse()
    return head + [f"(... {len(lines) - len(head) - len(tail)} steps omitted ...)"] + tail


def review_input(memory: Any, level: int, steps_tokens: int) -> str:
    """``level`` is 0-based (the level just won is ``level + 1`` for the agent)."""
    rec = memory.level_record(level)
    w = memory.working
    ended = rec["ended"]
    steps = clip_steps([render_row(r) for r in rec["steps"]], steps_tokens)
    plans = [f"turn {a.get('turn')} steps {a.get('steps')}: {a.get('plan')}" for a in rec["acts"]][-20:]
    hyps = [w.hyp_line(h) for h in ended.get("hypotheses", [])]
    finds = [f"- {f['text']}" for f in ended.get("findings", [])]
    goals = [w.goal_line(g) for g in ended.get("goals", [])] or ([ended["goal"]] if ended.get("goal") else [])
    earlier = [w.review_line(r) for r in w.reviews]
    return "\n\n".join([
        f"<level>{level + 1} won after {len(rec['steps'])} steps</level>",
        "<steps>\n" + ("\n".join(steps) or "(none)") + "\n</steps>",
        "<plans>\n" + ("\n".join(plans) or "(none)") + "\n</plans>",
        "<goal_at_end>\n" + ("\n".join(goals) or "(none)") + f"\n{ended.get('confirmation', '')}\n</goal_at_end>",
        "<hypotheses_at_end>\n" + ("\n".join(hyps) or "(none)") + "\n</hypotheses_at_end>",
        "<findings_at_end>\n" + ("\n".join(finds) or "(none)") + "\n</findings_at_end>",
        "<earlier_levels>\n" + ("\n".join(earlier) or "(none)") + "\n</earlier_levels>",
    ])


def _ask(llm: Any, user: str, max_tokens: int, overflow: type[Exception]) -> dict[str, Any]:
    last: Exception | None = None
    for _ in range(2):
        try:
            reply = llm.chat([{"role": "system", "content": REVIEW_SYSTEM}, {"role": "user", "content": user}],
                             tools=None, max_tokens=max_tokens, thinking=False)
        except overflow:
            user = user[len(user) // 2:]
            continue
        try:
            return extract_json(reply.content)
        except (ValueError, json.JSONDecodeError) as exc:
            last = exc
    raise ValueError(f"level review reply unparseable: {last}")


def review(*, llm: Any, memory: Any, level: int, max_tokens: int, steps_tokens: int,
           overflow: type[Exception]) -> dict[str, Any]:
    """One review of level ``level`` (0-based), written into the memories. Raises on a reply that never parses."""
    out = _ask(llm, review_input(memory, level, steps_tokens), max_tokens, overflow)
    return memory.add_level_review(level, out)
