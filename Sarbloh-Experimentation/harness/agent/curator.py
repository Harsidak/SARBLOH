"""The curator: the hidden reviewer that turns the agent's trajectory into memory. It runs at every level-up (inside
``play()``, before the level-up clears the level's plan, hypotheses and findings, so it can write skills from them),
and, when config ``curator_every_turns`` is above 0, at the end of a turn once that many model requests have gone by
since its last pass.

One JSON call, thinking off, with its own prompt (it never touches the agent's cached prefix). It reads every kind of
memory: the last steps of the timeline, the agent's goal, plan, hypotheses and findings, the skills it has loaded
(SKILLS), the map of its code (the code graph) and the lessons graph. It writes lessons (at a level-up it judges which of
the agent's hypotheses the level proved or refuted, from what the agent tried and what worked), skills (2 to 5 at a
level-up, otherwise at most 2) and the names of up to five skills to load, which go into the agent's SKILLS. The agent
never sees the curator's turns or the lessons: it sees the level-up message (it names the new skills) and SKILLS. A
reply that does not parse, or an item that does not validate, is dropped and logged; the rest is applied.
"""

from __future__ import annotations

import json
import re
from typing import Any

from harness.memory.lessons import EDGE_TYPES, shape_id
from harness.memory.lifecycle import LOADED_SKILLS
from harness.memory.timeline import render_row

_INTRO = """Your job is to review the memory of the agent that is playing a turn based game you need to return in json only.
IMPORTANT: Don't think much and reason answer straight from the memory below.

The agent writes its own plan, hypotheses, findings and goal, and it keeps the skills it is using in SKILLS. The code graph below lists the functions it wrote and what each one calls. The agent does not mark its hypotheses true or false: you do. Your job:
1. lessons: facts to keep for later levels and other games: findings, a goal that won a level, and hypotheses you judge from the last steps. Read each hypothesis next to the moves the agent made and what they did. A hypothesis is a "rule" when the moves show it held, above all when following it won the level. It is "refuted" when a move did something else. When the steps show neither, leave it out. Each lesson is one short general sentence (name objects by colour and shape, not by id). Link it to the shapes it is about with their key, e.g. "R#a1b2c3"."""

_SKILLS = """
2. skills: at most 2 short notes (under 120 words each) that turn several lessons into general knowledge for playing such games. Give a snake_case name. Update an existing curated skill by reusing its name. Skip this when the lessons do not support a new note."""

_SKILLS_LEVEL_UP = """
2. skills: the agent has just won a level, and its memory of that level is below. Write skills that can assist the agent to further solve the levels
likewise making sure the skills are useful for the agent to understand the game mechanics, rules, and how to detail with different situations.
From what this level taught you can write skills on repeated patterns or procedure that worked. The code graph shows how the agent's code was put together, so a skill can describe a procedure that worked, such as which model and which search it used and in what order. Make each skill specific not to game related objects but likewise what worked for the model that it implemented.
say what the moves do and in what order, and say when the skill applies. Under 120 words each, with a snake_case name. Update an existing curated skill by reusing its name."""

_OUTRO = """
3. load_skills: the names of at most 5 skills (base or curated, new ones included) most useful for the agent now. They are loaded into its SKILLS.

Return exactly:
{"lessons": [{"type": "rule|refuted|goal", "text": "...", "shapes": ["R#a1b2c3"], "edge": "about|moves|blocks|wins-when|contradicts"}], "skills": [{"name": "...", "content": "..."}], "load_skills": ["..."]}"""

CURATOR_SYSTEM = _INTRO + _SKILLS + _OUTRO
CURATOR_SYSTEM_LEVEL_UP = _INTRO + _SKILLS_LEVEL_UP + _OUTRO

MAX_SKILLS = 2             # skills applied per pass
MAX_SKILLS_LEVEL_UP = 5    # skills applied at a level-up
STEPS = 25                 # timeline steps shown per pass
STEPS_LEVEL_UP = 60        # at a level-up: enough to cover most of the level
GRAPH_CHARS = 2000         # the code graph shown

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


def curator_input(memory: Any, objects: list[Any], reason: str, steps: int = STEPS, code_graph: str = "") -> str:
    w = memory.working
    b = w.blocks()
    shapes = sorted({f"{o.label()[0]}#{o.hash} ({o.label()}, obj {o.id})" for o in objects or []})[:40]
    skills = []
    for name in memory.skills.names():
        text = memory.skills.text(name) or ""
        skills.append(f"- {name}{' (base)' if name in memory.skills.base else ''}: {text[:100]}")
    lessons = memory.lessons.subgraph([shape_id(o.label()[0], o.hash) for o in objects or []], 600, mark_used=False) \
        if memory.lessons is not None else "(lessons graph off)"
    graph = code_graph if len(code_graph) <= GRAPH_CHARS else code_graph[:GRAPH_CHARS].rsplit("\n", 1)[0] + "\n..."
    return "\n\n".join([
        f"<trigger>{reason}</trigger>",
        "<last_steps>\n" + ("\n".join(render_row(r) for r in memory.timeline.recent_steps(steps)) or "(none)")
        + "\n</last_steps>",
        f"<goal>{w.goal or '(none)'}</goal>\n<plan>{w.plan or '(none)'}</plan>",
        f"<hypotheses>\n{b['hypotheses'] or '(none)'}\n</hypotheses>\n<findings>\n{b['findings'] or '(none)'}\n</findings>",
        f"<loaded_skills>\n{', '.join(w.skills) or '(none)'}\n</loaded_skills>",
        f"<code_graph>\n{graph or '(no functions yet)'}\n</code_graph>",
        "<shapes_on_screen>\n" + "\n".join(shapes) + "\n</shapes_on_screen>",
        f"<lessons_graph>\n{lessons or '(empty)'}\n</lessons_graph>",
        "<skills>\n" + "\n".join(skills) + "\n</skills>",
    ])


def apply(memory: Any, out: dict[str, Any], objects: list[Any], max_skills: int = MAX_SKILLS) -> dict[str, Any]:
    """Applies a reply: lessons and skills are written; ``loaded`` is the names to put into the agent's SKILLS (known
    skills only, at most LOADED_SKILLS), which the caller loads into the kernel."""
    rec: dict[str, Any] = {"lessons": [], "skills": [], "loaded": [], "errors": []}
    labels = {(o.label()[0], o.hash): o.label() for o in objects or []}
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
    if isinstance(load, list):
        rec["loaded"] = [n for n in dict.fromkeys(str(x) for x in load)
                         if memory.skills.text(n) is not None][:LOADED_SKILLS]
    memory.save()
    return rec


def curate(*, llm: Any, memory: Any, objects: list[Any], reason: str, max_tokens: int,
           overflow: type[Exception], code_graph: str = "") -> dict[str, Any]:
    """One curator pass. ``reason`` "level_up" asks for 2 to 5 skills from the level just won; ``code_graph`` is the
    rendered map of the agent's code. Raises when the reply never parses (the caller logs it and carries on)."""
    level_up = reason == "level_up"
    system = CURATOR_SYSTEM_LEVEL_UP if level_up else CURATOR_SYSTEM
    user = curator_input(memory, objects, reason, STEPS_LEVEL_UP if level_up else STEPS, code_graph)
    out = _ask(llm, system, user, max_tokens, overflow)
    return {"reason": reason, **apply(memory, out, objects, MAX_SKILLS_LEVEL_UP if level_up else MAX_SKILLS)}
