"""The curator: the hidden reviewer that turns the agent's trajectory into memory. It runs at every level-up (inside
``play()``, before the level-up marks the level's goals "won"; nothing of the agent's memory is erased),
and, when config ``curator_every_turns`` is above 0, at the end of a turn once that many model requests have gone by
since its last pass.

One JSON call, thinking off, with its own prompt (it never touches the agent's cached prefix). It reads every kind of
memory: the last steps of the timeline, the agent's goal, plan, hypotheses and findings, the skills it has loaded
(SKILLS), the map of its code (the code graph) and the lessons graph. It writes lessons (at a level-up it judges which of
the agent's hypotheses the level proved or refuted, from what the agent tried and what worked), skills (2 to 5 at a
level-up, otherwise at most 2) and the names of up to five skills to load, which go into the agent's SKILLS. The agent
never sees the curator's turns or the lessons: it sees the level-up message (it names the new skills) and SKILLS. A
reply that does not parse, or an item that does not validate, is dropped and logged; the rest is applied.

Its second job is the stuck review (``review``). Once per game, at the first turn end after config ``review_at_s`` of
run time (4 h), the agent's turn waits while the curator reads the whole level the agent is on: every move and what it
changed (also the moves drained from the agent's context), the goals and hypotheses the agent dropped, the ones it holds
now, and a few facts the host counts itself (moves never used, objects never clicked, moves that led back to a board
already seen). With thinking on, it checks each belief against the moves, names the mistakes, lists what was never
tried, proposes other goals and one next experiment. Unlike the lessons, the agent sees this: it comes back as a
message (``review_message``). It writes nothing into the agent's memory; the agent decides what to change.
"""

from __future__ import annotations

import json
import re
from typing import Any

from harness.agent import guardrails
from harness.memory.lessons import EDGE_TYPES, shape_id
from harness.memory.lifecycle import LOADED_SKILLS
from harness.memory.timeline import render_row

_INTRO = """Your job is to review the memory of the agent that is playing a turn based game you need to return in json only.
IMPORTANT: Don't think much and reason answer straight from the memory below.

The agent writes its own plan, hypotheses, findings and goal, and it keeps the skills it is using in SKILLS. The code graph below lists the functions it wrote and what each one calls. Its notes are kept per level ("L1", "L2", ...), and a goal marked [won] won its level; judge the hypotheses yourself from the steps. Your job:
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
        f"<goal>\n{b['goal'] or '(none)'}\n</goal>\n<plan>\n{b['plan'] or '(none)'}\n</plan>",
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


# =====================================================================================================================
# The stuck review
# =====================================================================================================================

REVIEW_SYSTEM = """You are reviewing the play of an agent that is stuck on one level of a turn-based game. It has spent a long time on this level without winning it. You did not play, so you are not tied to its ideas, and you can see every move it made on this level, also the early ones it has forgotten. Your job is to find where its thinking went wrong and to point it to a better way forward.

Every level of these games can be won. If the agent has come to believe that the level cannot be won, or that the game is broken, that belief is the mistake to fix.

Trust the moves, not the agent's words. Its goal, plan, hypotheses and findings are only what it believes. The moves, and what each move changed on the board, are what really happened. Check every belief against them. Do not repeat the agent's own conclusions back to it: if you agree with its goal, say which moves show it is right, and if no move shows it, say that it is untested.

Look for these common mistakes: a goal that was never tested; an idea kept after a move contradicted it; the same approach tried again and again with the same result; a move, an object or a part of the board that was never tried; a change on the board (a counter, a bar, a colour, the edge of the screen) that the agent never explained; a GAME_OVER whose cause it never found; a goal it dropped too early, or dropped for a reason the moves do not support.

Write for the agent, briefly and plainly. Name objects by colour, shape and position, not by id. Cite moves by their number, such as #12.

Reply with one JSON object and nothing else:
{"beliefs": [{"belief": "the agent's goal or hypothesis, in short", "status": "one word: supported, contradicted or untested", "evidence": "the moves that show it, such as #12 and #40, and what they did"}],
 "mistakes": ["one mistake in how the agent played, with the moves that show it"],
 "untried": ["a move, object or part of the board the agent has not tried, and why it is worth trying"],
 "goals": ["a goal that fits everything seen so far, most likely first; two or three of them, not ones the agent already dropped unless the moves never really tested them"],
 "next_experiment": "a few moves that would tell these goals apart, and what each result would mean"}"""

REVIEW_STEPS = 200          # move lines shown; above it the first REVIEW_HEAD and the last ones
REVIEW_HEAD = 40
REVIEW_LINE_CHARS = 180
REVIEW_OBJECTS = 30         # objects on screen listed
REVIEW_NEVER_CLICKED = 15   # objects never clicked listed
REVIEW_CAPS = {"beliefs": 12, "mistakes": 5, "untried": 6, "goals": 3}
REVIEW_ITEM_CHARS = 400
_CLICK = re.compile(r"A6\(r(\d+),c(\d+)\)")


def _level_steps(memory: Any, level: int) -> list[dict[str, Any]]:
    return [r for r in memory.timeline.steps() if r.get("level") == level]


def _move_name(action: Any) -> str:
    """A move's name as the timeline writes it (guardrails.label: UP, FIRE, ...), every click under one name; an int
    is an action id."""
    if isinstance(action, int):
        return "A6 (click)" if action == 6 else guardrails.label(action, None, None)
    return "A6 (click)" if str(action).startswith("A6") else str(action)


def _clip(text: Any, chars: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= chars else text[:chars - 3] + "..."


def _step_lines(steps: list[dict[str, Any]], max_steps: int) -> list[str]:
    """One line per move; a run of the same move with the same change becomes one line; above ``max_steps`` lines the
    first REVIEW_HEAD and the last ones are kept."""
    runs: list[list[dict[str, Any]]] = []
    for r in steps:
        prev = runs[-1][-1] if runs else None
        if prev is not None and (r.get("action"), r.get("change")) == (prev.get("action"), prev.get("change")) \
                and "GAME_OVER" not in (r.get("state"), prev.get("state")) and not prev.get("level_up"):
            runs[-1].append(r)
        else:
            runs.append([r])
    lines = []
    for run in runs:
        line = render_row({**run[0], "kind": "step"}) if len(run) == 1 else \
            f"#{run[0].get('i')}-#{run[-1].get('i')} {run[0].get('action')} x{len(run)}: {run[0].get('change')}"
        lines.append(_clip(line, REVIEW_LINE_CHARS))
    if len(lines) > max_steps:
        head = min(REVIEW_HEAD, max_steps // 4)
        cut = len(lines) - max_steps
        lines = lines[:head] + [f"... {cut} lines left out here (the host counts cover them) ..."] \
            + lines[head + cut:]
    return lines


def host_facts(steps: list[dict[str, Any]], legal: list[int], objects: list[Any]) -> list[str]:
    """What the host counts from the level's moves: moves per action and how many changed nothing, legal moves never
    used, GAME_OVERs, moves repeated from the same board, and the objects on screen no click has landed on."""
    facts = []
    by: dict[str, list[int]] = {}
    for r in steps:
        n = by.setdefault(_move_name(r.get("action")), [0, 0])
        n[0] += 1
        n[1] += str(r.get("change", "")).startswith("no change")
    if by:
        facts.append("moves used: " + ", ".join(f"{k} {v[0]} times ({v[1]} changed nothing)" for k, v in by.items()))
    never = [_move_name(a) for a in legal if _move_name(a) not in by]
    if never:
        facts.append("legal moves never used on this level: " + ", ".join(never))
    overs = sum(1 for r in steps if r.get("state") == "GAME_OVER")
    if overs:
        facts.append(f"GAME_OVER {overs} times on this level")
    back = sum(1 for r in steps if r.get("repeat_of") is not None)
    if back:
        facts.append(f"{back} of {len(steps)} moves repeated a move already made from the same board")
    if 6 in legal:
        clicks = {(int(m.group(1)), int(m.group(2))) for r in steps
                  for m in [_CLICK.search(str(r.get("action", "")))] if m}
        missed = [o for o in objects or [] if not o.hud and not (o.cells & clicks)]
        if missed:
            facts.append(f"objects on screen now that no click on this level has landed on ({len(missed)}): "
                         + "; ".join(f"{o.label()} at {o.at()}" for o in missed[:REVIEW_NEVER_CLICKED])
                         + (" ..." if len(missed) > REVIEW_NEVER_CLICKED else ""))
    return facts


def review_input(memory: Any, *, level: int, legal: list[int], objects: list[Any], tokens: int,
                 moves_left: int | None, max_steps: int = REVIEW_STEPS, code_graph: str = "") -> str:
    """The stuck review's input for ``level`` (0-based, as levels_completed counts)."""
    w = memory.working
    steps = _level_steps(memory, level)
    mine = lambda items: [x for x in items if x["level"] == level]
    goals = [g["text"] for g in mine(w.goals)]
    hyps = [h["text"] for h in mine(w.hypotheses)]
    finds = [f["text"] for f in mine(w.findings)]
    earlier = [f"L{f['level'] + 1}: {f['text']}" for f in w.findings if f["level"] < level][-15:]
    dropped = [f"{d['kind']} (dropped at turn {d['turn']}): {d['text']}" for d in mine(w.dropped)]
    bullet = lambda xs: "\n".join(f"- {x}" for x in xs) or "(none)"
    shapes = [f"{o.label()} at {o.at()}{' (edge of screen)' if o.hud else ''}" for o in (objects or [])]
    graph = code_graph if len(code_graph) <= GRAPH_CHARS else code_graph[:GRAPH_CHARS].rsplit("\n", 1)[0] + "\n..."
    left = f", {moves_left} moves left in the game" if moves_left is not None else ""
    return "\n\n".join([
        f"<level>\nLevel {level + 1}. The agent has made {len(steps)} moves on it and thought for about "
        f"{tokens // 1000}k tokens without winning it{left}. Legal moves: "
        f"{', '.join(_move_name(a) for a in legal) or '(unknown)'}.\n</level>",
        "<earlier_levels>\n" + bullet(w.goal_log) + "\n</earlier_levels>",
        "<what_the_agent_believes_now>\nGOAL:\n" + bullet(goals) + "\nPLAN:\n" + (w.plan_for(level) or "(none)")
        + "\nHYPOTHESES:\n" + bullet(hyps) + "\nFINDINGS on this level:\n" + bullet(finds)
        + "\nFINDINGS from earlier levels:\n" + bullet(earlier) + "\n</what_the_agent_believes_now>",
        "<what_the_agent_dropped_on_this_level>\n" + bullet(dropped) + "\n</what_the_agent_dropped_on_this_level>",
        "<counted_by_the_host>\n" + bullet(host_facts(steps, legal, objects)) + "\n</counted_by_the_host>",
        "<moves_on_this_level>\n" + ("\n".join(_step_lines(steps, max_steps)) or "(none)")
        + "\n</moves_on_this_level>",
        "<objects_on_screen_now>\n" + ("\n".join(shapes[:REVIEW_OBJECTS]) or "(none)")
        + ("\n..." if len(shapes) > REVIEW_OBJECTS else "") + "\n</objects_on_screen_now>",
        f"<code_graph>\n{graph or '(no functions yet)'}\n</code_graph>",
    ])


def _clean_review(out: dict[str, Any]) -> dict[str, Any]:
    """The reply cut to its caps; items that are not text (or a belief without its text) are dropped. Raises when
    nothing usable is left."""
    texts = lambda v: [_clip(x, REVIEW_ITEM_CHARS) for x in (v if isinstance(v, list) else [])
                       if isinstance(x, (str, int, float)) and str(x).strip()]
    beliefs = [{k: _clip(b.get(k) or "", REVIEW_ITEM_CHARS) for k in ("belief", "status", "evidence")}
               for b in (out.get("beliefs") if isinstance(out.get("beliefs"), list) else [])
               if isinstance(b, dict) and str(b.get("belief") or "").strip()]
    rev: dict[str, Any] = {"beliefs": beliefs[:REVIEW_CAPS["beliefs"]]}
    for k in ("mistakes", "untried", "goals"):
        rev[k] = texts(out.get(k))[:REVIEW_CAPS[k]]
    nxt = out.get("next_experiment")
    rev["next_experiment"] = _clip(nxt, REVIEW_ITEM_CHARS) if isinstance(nxt, (str, int, float)) else ""
    if not any(rev.values()):
        raise ValueError("review reply has nothing usable")
    return rev


def review(*, llm: Any, memory: Any, level: int, legal: list[int], objects: list[Any], tokens: int,
           moves_left: int | None, max_tokens: int, overflow: type[Exception], timeout_s: float | None = None,
           template_kwargs: dict[str, Any] | None = None, code_graph: str = "") -> dict[str, Any]:
    """One stuck review, thinking on (one call, no retry: it costs the game's clock). Returns the cleaned review plus
    the input size and reply tokens. A context overflow halves the move lines once; a reply with no usable JSON raises
    (the caller logs it and the agent goes on)."""
    steps = REVIEW_STEPS
    for _ in range(2):
        user = review_input(memory, level=level, legal=legal, objects=objects, tokens=tokens, moves_left=moves_left,
                            max_steps=steps, code_graph=code_graph)
        try:
            reply = llm.chat([{"role": "system", "content": REVIEW_SYSTEM}, {"role": "user", "content": user}],
                             tools=None, max_tokens=max_tokens, timeout_s=timeout_s, thinking=True,
                             template_kwargs=template_kwargs)
        except overflow:
            steps //= 2
            continue
        rev = _clean_review(extract_json(reply.content))
        return {**rev, "input_chars": len(user), "reply_tokens": getattr(reply, "completion_tokens", None)}
    raise ValueError("review input too long even with half the move lines")


def review_message(rev: dict[str, Any], *, moves: int) -> str:
    """The review as the agent reads it (a user message)."""
    parts = [f"A reviewer has read everything you did on this level: all {moves} moves, also the early ones that are "
             f"no longer in your context, and has checked your beliefs against what the moves did. Read it, decide "
             f"what you agree with, and update GOAL, HYPOTHESES and PLAN before your next move. A move it cites, such "
             f"as #12, can be read with `recall(\"#12\")`."]
    if rev.get("beliefs"):
        parts.append("Your beliefs, checked against the moves:\n" + "\n".join(
            f"- {b['belief']}: {b['status'] or 'not judged'}" + (f" ({b['evidence']})" if b["evidence"] else "")
            for b in rev["beliefs"]))
    for key, title in (("mistakes", "Mistakes so far:"), ("untried", "Not tried yet:")):
        if rev.get(key):
            parts.append(title + "\n" + "\n".join(f"- {x}" for x in rev[key]))
    if rev.get("goals"):
        parts.append("Goals that fit what you have seen, most likely first:\n"
                     + "\n".join(f"{i}. {g}" for i, g in enumerate(rev["goals"], 1)))
    if rev.get("next_experiment"):
        parts.append("Next experiment: " + rev["next_experiment"])
    return "\n\n".join(parts)
