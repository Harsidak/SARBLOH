"""E008 curator: the hidden reviewer that turns the agent's trajectory into memory. It replaces upstream auto-refine
for the "e008" toolset (``prime.agent.refine`` stays for the "ipython" control arm) and runs at every level-up, after
every compaction and every ``curator_every_turns`` turns.

One JSON call, thinking off. From the last steps of the timeline, the agent's working memory and the lessons graph it
writes: up to three open questions (#5), lessons the agent has already settled (verified or refuted, never proposed
ones), short skills distilled from lessons, and which three skills to load. The agent never sees the curator's turns;
it sees their effect in its pinned memory block. A reply that does not parse, or an item that does not validate, is
dropped and logged; the rest is applied.

E018: with ``memory.wrong_rulebook`` the curator sees the wrong rulebook and its ``refuted`` lessons go there instead
of the lessons graph; with ``memory.skill_names: "use_case"`` it names skills by their use case (at most 10 words).
"""

from __future__ import annotations

import json
import re
from typing import Any

from prime.agent.refine import extract_json
from prime.memory.lessons import EDGE_TYPES, shape_id
from prime.memory.timeline import render_row

CURATOR_SYSTEM = """You review the memory of an agent that is learning an unknown turn-based grid game by acting in it. You never act. You return JSON only.

The agent writes its own plan, hypotheses (proposed / verified / refuted), findings and goal. Your job:
1. open_questions: at most 3 short questions that the last steps raise and that the agent has not answered. Prefer questions whose answer changes what the agent should do next. Do not repeat answered ones.
2. lessons: facts to keep for later levels and other games. Only rules the agent marked verified or refuted, findings, or a goal that won a level. Never a proposed hypothesis. Each lesson is one short general sentence (name objects by colour and shape, not by id). Link it to the shapes it is about with their key, e.g. "R#a1b2c3".
3. skills: at most 2 short notes (under 120 words each) that turn several lessons into general knowledge for playing such games. Give a snake_case name. Update an existing curated skill by reusing its name. Skip this when the lessons do not support a new note.
4. load_skills: the names of at most 3 skills (base or curated) most useful for the current situation.

Return exactly:
{"open_questions": ["..."], "lessons": [{"type": "rule|refuted|goal", "text": "...", "shapes": ["R#a1b2c3"], "edge": "about|moves|blocks|wins-when|contradicts"}], "skills": [{"name": "...", "content": "..."}], "load_skills": ["..."]}"""

# E018 edits of CURATOR_SYSTEM, (old, new); each old occurs once.
E018_WRONG = ("Never a proposed hypothesis.",
              ("Never a proposed hypothesis. A refuted rule goes to the agent's wrong rulebook (in the input): do not "
               "repeat one that is already there."))
E018_USE_CASE = ("Give a snake_case name.",
                 ('Name it by its use case, when to use it, in at most 10 words (e.g. "when blocks must be pushed onto '
                  'matching targets").'))

_SHAPE_KEY = re.compile(r"^([A-Za-z])#([0-9a-f]{6})$")


def system_prompt(memory: Any) -> str:
    """CURATOR_SYSTEM with the E018 edits of the switches that are on."""
    text = CURATOR_SYSTEM
    edits = ([E018_WRONG] if memory.wrong is not None else []) + ([E018_USE_CASE] if memory.skills.use_case_names else [])
    for old, new in edits:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    return text


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


def curator_input(memory: Any, objects: list[Any], reason: str, steps: int = 25) -> str:
    w = memory.working
    b = w.blocks()
    shapes = sorted({f"{o.label()[0]}#{o.hash} ({o.label()}, obj {o.id})" for o in objects or []})[:40]
    skills = []
    for name in memory.skills.names():
        text = memory.skills.text(name) or ""
        skills.append(f"- {name}{' (base)' if name in memory.skills.base else ''}: {text[:100]}")
    lessons = memory.lessons.subgraph([shape_id(o.label()[0], o.hash) for o in objects or []], 600, mark_used=False) \
        if memory.lessons is not None else "(lessons graph off)"
    wrong = [] if memory.wrong is None else \
        [f"<wrong_rulebook>\n{memory.wrong.render(memory.game, 300) or '(empty)'}\n</wrong_rulebook>"]   # E018
    return "\n\n".join([
        f"<trigger>{reason}</trigger>",
        "<last_steps>\n" + ("\n".join(render_row(r) for r in memory.timeline.recent_steps(steps)) or "(none)")
        + "\n</last_steps>",
        f"<goal>{(b['goal'] if w.goal_versioning else w.goal) or '(none)'}</goal>\n<plan>{w.plan or '(none)'}</plan>",
        f"<hypotheses>\n{b['hypotheses'] or '(none)'}\n</hypotheses>\n<findings>\n{b['findings'] or '(none)'}\n</findings>",
        f"<open_questions_now>\n{b['questions'] or '(none)'}\n</open_questions_now>",
        "<shapes_on_screen>\n" + "\n".join(shapes) + "\n</shapes_on_screen>",
        f"<lessons_graph>\n{lessons or '(empty)'}\n</lessons_graph>",
        *wrong,
        "<skills>\n" + "\n".join(skills) + "\n</skills>",
    ])


def apply(memory: Any, out: dict[str, Any], objects: list[Any]) -> dict[str, Any]:
    rec: dict[str, Any] = {"questions": [], "lessons": [], "skills": [], "loaded": None, "errors": []}
    labels = {(o.label()[0], o.hash): o.label() for o in objects or []}
    qs = out.get("open_questions")
    if isinstance(qs, list):
        memory.working.set_questions([str(q)[:200] for q in qs])
        rec["questions"] = memory.working.questions
    if memory.wrong is not None:
        rec["wrong"] = []
    for item in out.get("lessons") or []:
        try:
            kind, text = str(item.get("type") or "rule"), str(item.get("text") or "")
            if kind == "refuted" and memory.wrong is not None:                          # E018
                rec["wrong"].append(memory.wrong.add(text, game=memory.game, source="curator"))
                continue
            if memory.lessons is None:
                continue
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
    for item in (out.get("skills") or [])[:2]:
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
    """One curator pass. Raises when the reply never parses (the caller logs it and carries on)."""
    out = _ask(llm, system_prompt(memory), curator_input(memory, objects, reason), max_tokens, overflow)
    return {"reason": reason, **apply(memory, out, objects)}
