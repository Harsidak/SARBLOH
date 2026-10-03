"""The curator: the hidden reviewer that turns the agent's trajectory into memory. It runs at every level-up (inside
``act``, before the level-up clears the level's plan, hypotheses and findings, so it can write skills from them),
after every compaction and every ``curator_every_turns`` turns.

One JSON call, thinking off. From the last steps of the timeline, the agent's working memory and the lessons graph it
writes: up to three open questions, lessons the agent has already settled (verified or refuted, never proposed ones),
skills, and which three skills to load. At a level-up it writes 2 to 5 specific skills from what the level taught;
otherwise at most 2. The agent never sees the curator's turns; it sees their effect in its pinned memory block (and
the level-up message names the new skills). A reply that does not parse, or an item that does not validate, is
dropped and logged; the rest is applied.
"""

from __future__ import annotations

import json
import re
from typing import Any

from harness.memory.lessons import EDGE_TYPES, shape_id
from harness.memory.timeline import render_row

_INTRO = """You review the memory of an agent that is learning an unknown turn-based grid game by acting in it. You never act. You return JSON only.
Do not think or reason much: write your answer straight from the memory below.

The agent writes its own plan, hypotheses (proposed / verified / refuted), findings and goal. Your job:
1. open_questions: at most 3 short questions that the last steps raise and that the agent has not answered. Prefer questions whose answer changes what the agent should do next. Do not repeat answered ones.
2. lessons: facts to keep for later levels and other games. Only rules the agent marked verified or refuted, findings, or a goal that won a level. Never a proposed hypothesis. Each lesson is one short general sentence (name objects by colour and shape, not by id). Link it to the shapes it is about with their key, e.g. "R#a1b2c3"."""

_SKILLS = """
3. skills: at most 2 short notes (under 120 words each) that turn several lessons into general knowledge for playing such games. Give a snake_case name. Update an existing curated skill by reusing its name. Skip this when the lessons do not support a new note."""

_SKILLS_LEVEL_UP = """
3. skills: the agent has just won a level, and its memory of that level is below, before it is cleared. Write 2 to 5 skills from what this level taught: one skill per mechanic, control, goal pattern or procedure that worked. Make each skill specific: name the objects by colour and shape, say what the moves do and in what order, and say when the skill applies, so the agent can use it on the next level without testing it again. Base them on the verified hypotheses, the findings, the goal that won and the steps. Under 120 words each, with a snake_case name. Update an existing curated skill by reusing its name."""

_OUTRO = """
4. load_skills: the names of at most 3 skills (base or curated) most useful for the current situation.

Return exactly:
{"open_questions": ["..."], "lessons": [{"type": "rule|refuted|goal", "text": "...", "shapes": ["R#a1b2c3"], "edge": "about|moves|blocks|wins-when|contradicts"}], "skills": [{"name": "...", "content": "..."}], "load_skills": ["..."]}"""

CURATOR_SYSTEM = _INTRO + _SKILLS + _OUTRO
CURATOR_SYSTEM_LEVEL_UP = _INTRO + _SKILLS_LEVEL_UP + _OUTRO

MAX_SKILLS = 2             # skills applied per pass
MAX_SKILLS_LEVEL_UP = 5    # skills applied at a level-up
STEPS = 25                 # timeline steps shown per pass
STEPS_LEVEL_UP = 60        # at a level-up: enough to cover most of the level

_SHAPE_KEY = re.compile(r"^([A-Za-z])#([0-9a-f]{6})$")


def extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("reply JSON is not an object")
    return value


def _ask(llm: Any, system: str, user: str, max_tokens: int, overflow: type[Exception]) -> dict[str, Any]:
    last: Exception | None = None
    for _ in range(2):
        try:
            reply = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                             tools=None, max_tokens=max_tokens, thinking=False)
        except overflow:
            user = user[len(user) // 2:]
            continue
        try:
            return extract_json(reply.content)
        except (ValueError, json.JSONDecodeError) as exc:
            last = exc
    raise ValueError(f"curator reply unparseable: {last}")


def curator_input(memory: Any, objects: list[Any], reason: str, steps: int = STEPS) -> str:
    w = memory.working
    b = w.blocks()
    shapes = sorted({f"{o.label()[0]}#{o.hash} ({o.label()}, obj {o.id})" for o in objects or []})[:40]
    skills = []
    for name in memory.skills.names():
        text = memory.skills.text(name) or ""
        skills.append(f"- {name}{' (base)' if name in memory.skills.base else ''}: {text[:100]}")
    lessons = memory.lessons.subgraph([shape_id(o.label()[0], o.hash) for o in objects or []], 600, mark_used=False) \
        if memory.lessons is not None else "(lessons graph off)"
    return "\n\n".join([
        f"<trigger>{reason}</trigger>",
        "<last_steps>\n" + ("\n".join(render_row(r) for r in memory.timeline.recent_steps(steps)) or "(none)")
        + "\n</last_steps>",
        f"<goal>{w.goal or '(none)'}</goal>\n<plan>{w.plan or '(none)'}</plan>",
        f"<hypotheses>\n{b['hypotheses'] or '(none)'}\n</hypotheses>\n<findings>\n{b['findings'] or '(none)'}\n</findings>",
        f"<open_questions_now>\n{b['questions'] or '(none)'}\n</open_questions_now>",
        "<shapes_on_screen>\n" + "\n".join(shapes) + "\n</shapes_on_screen>",
        f"<lessons_graph>\n{lessons or '(empty)'}\n</lessons_graph>",
        "<skills>\n" + "\n".join(skills) + "\n</skills>",
    ])


def apply(memory: Any, out: dict[str, Any], objects: list[Any], max_skills: int = MAX_SKILLS) -> dict[str, Any]:
    rec: dict[str, Any] = {"questions": [], "lessons": [], "skills": [], "loaded": None, "errors": []}
    labels = {(o.label()[0], o.hash): o.label() for o in objects or []}
    qs = out.get("open_questions")
    if isinstance(qs, list):
        memory.working.set_questions([str(q)[:200] for q in qs])
        rec["questions"] = memory.working.questions
    for item in out.get("lessons") or []:
        try:
            if memory.lessons is None:
                break
            kind, text = str(item.get("type") or "rule"), str(item.get("text") or "")
            nid = memory.lessons.add_node(kind, text, game=memory.game, source="curator")
            edge = str(item.get("edge") or "about")
            edge = edge if edge in EDGE_TYPES else "about"
            linked = []
            for key in item.get("shapes") or []:
                m = _SHAPE_KEY.match(str(key).strip())
                if not m:
                    continue
                letter, h = m.group(1), m.group(2)
                sid = memory.lessons.add_shape(letter, h, labels.get((letter, h), letter), game=memory.game)
                if memory.lessons.add_edge(nid, sid, edge):
                    linked.append(sid)
            rec["lessons"].append({"node": nid, "shapes": linked})
        except Exception as exc:  # noqa: BLE001 - one bad item does not void the others
            rec["errors"].append(f"lesson: {exc}"[:200])
    for item in (out.get("skills") or [])[:max_skills]:
        try:
            rec["skills"].append(memory.skills.upsert(item.get("name"), item.get("content"), game=memory.game))
        except Exception as exc:  # noqa: BLE001
            rec["errors"].append(f"skill: {exc}"[:200])
    load = out.get("load_skills")
    if isinstance(load, list) and load:
        rec["loaded"] = memory.skills.set_loaded(memory.game, [str(x) for x in load])
    memory.save()
    return rec


def curate(*, llm: Any, memory: Any, objects: list[Any], reason: str, max_tokens: int,
           overflow: type[Exception]) -> dict[str, Any]:
    """One curator pass. ``reason`` "level_up" asks for 2 to 5 skills from the level just won. Raises when the reply
    never parses (the caller logs it and carries on)."""
    level_up = reason == "level_up"
    system = CURATOR_SYSTEM_LEVEL_UP if level_up else CURATOR_SYSTEM
    user = curator_input(memory, objects, reason, STEPS_LEVEL_UP if level_up else STEPS)
    out = _ask(llm, system, user, max_tokens, overflow)
    return {"reason": reason, **apply(memory, out, objects, MAX_SKILLS_LEVEL_UP if level_up else MAX_SKILLS)}
