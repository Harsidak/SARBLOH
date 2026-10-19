"""``GameMemory``: all memories of one game, and what happens to them at a level-up and at a new game.

Files: ``<root>/<game>/timeline.jsonl`` (#1), ``<root>/<game>/working.json`` (the five kernel variables GOAL, PLAN,
HYPOTHESES, FINDINGS and SKILLS), ``<root>/<game>/levels.jsonl`` (what each level-up erased), and, shared by all
games, ``<root>/lessons.json`` (#8) and ``<root>/skills.json``.

Level-up: the findings become ``rule`` nodes of the lessons graph and the goal a ``goal`` node, each linked to the
shapes it names as "obj N" (the curator judges the hypotheses); the goal gets "confirmed: won level N"; then plan,
hypotheses and findings are erased (SKILLS is kept). New game: the timeline is archived and everything except the
lessons graph and the skills is erased.

The agent reads two of them with ``recall()`` in its kernel (``recall_text``): the timeline and the skills. The lessons
graph is read only by the curator.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from harness.memory.lessons import LessonsGraph, SkillBook
from harness.memory.store import append_jsonl, clip
from harness.memory.timeline import Timeline, render_row
from harness.memory.working import WorkingMemory

SCOPES = ("all", "timeline", "skills")   # what `recall()` searches; memory and lessons are not in it
LOADED_SKILLS = 5   # SKILLS in the kernel holds at most this many skills, the oldest dropped first
TEXT_MATCHES = 2    # a recall whose words name no skill loads this many of the skills whose text matches
_OBJ = re.compile(r"\bobj(?:ect)?s?\s*#?(\d+)", re.IGNORECASE)


def _shapes(objects: list[Any]) -> dict[int, Any]:
    return {o.id: o for o in objects or []}


class GameMemory:
    def __init__(self, root: Path, game: str, cfg: dict[str, Any]) -> None:
        self.root = Path(root)
        self.game = game
        self.dir = self.root / game
        self.cfg = cfg
        self.caps = cfg["caps"]
        self.timeline = Timeline(self.dir / "timeline.jsonl")
        self.working = WorkingMemory(self.dir / "working.json", self.caps)
        self.lessons = LessonsGraph.shared(self.root / "lessons.json") if cfg.get("lessons", True) else None
        self.skills = SkillBook.shared(self.root / "skills.json")
        self.promoted = 0

    def save(self) -> None:
        self.working.save()
        if self.lessons is not None:
            self.lessons.save()
        self.skills.save()

    # --- lifecycle -----------------------------------------------------------------------------------------
    def on_new_game(self) -> dict[str, Any]:
        archived = self.timeline.archive()
        self.working.clear_game()
        self.working.save()
        return {"archived": str(archived) if archived else None}

    def on_level_up(self, *, level_won: int, step: int | None, objects: list[Any]) -> dict[str, Any]:
        """``level_won`` = levels completed after the step; ``objects`` = the level's objects before the new layout,
        so "obj N" in the agent's text resolves to the ids it saw."""
        w = self.working
        line = w.confirm_goal(level_won)
        promoted: list[dict[str, Any]] = []
        if self.lessons is not None:
            shapes = _shapes(objects)
            # Hypotheses are not promoted here: the curator's level-up pass judges which ones the level proved.
            items = [("rule", f["text"]) for f in w.findings]
            if w.goal:
                items.append(("goal", f"won a level with goal: {w.goal}"))
            for kind, text in items:
                nid = self.lessons.add_node(kind, text, game=self.game, level=level_won - 1, source="level_up")
                linked = []
                for m in _OBJ.finditer(text):
                    o = shapes.get(int(m.group(1)))
                    if o is not None:
                        sid = self.lessons.add_shape(o.label()[0], o.hash, o.label(), game=self.game,
                                                     level=level_won - 1)
                        self.lessons.add_edge(nid, sid, "about")
                        linked.append(sid)
                promoted.append({"node": nid, "type": kind, "text": clip(text, 60), "shapes": linked})
            self.promoted += len(promoted)
        gone = w.clear_level()
        append_jsonl(self.dir / "levels.jsonl", {"level": level_won - 1, "step": step, "goal": w.goal,
                                                 "confirmation": line, **gone})
        self.timeline.append({"kind": "level_up", "level": level_won, "i": step, "goal": w.goal})
        self.save()
        return {"confirmation": line, "promoted": promoted, "erased": {k: len(v) if isinstance(v, list) else bool(v)
                                                                      for k, v in gone.items()}}

    # --- writes ---------------------------------------------------------------------------------
    def write_act(self, args: dict[str, Any], *, turn: int, level: int) -> dict[str, Any]:
        """plan, hypotheses, findings, goal, all optional. Validates everything before writing anything. A missing plan
        keeps the old one; the returned ``plan`` is the one written in this call, or None."""
        plan = str(args.get("plan") or "").strip() or None
        w = self.working
        snapshot = (w.plan, [dict(h, history=list(h["history"])) for h in w.hypotheses], list(w.findings), w.goal,
                    w.next_h)
        try:
            hyp = w.update_hypotheses(args.get("hypotheses"), turn=turn, level=level)
            found = w.add_findings(args.get("findings"), turn=turn, level=level)
        except Exception:
            w.plan, w.hypotheses, w.findings, w.goal, w.next_h = snapshot
            raise
        if plan:
            w.set_plan(plan)
        goal_changed = w.set_goal(args.get("goal") or "")
        w.save()
        return {"plan": plan, "hypotheses": hyp, "findings": found, "goal": w.goal if goal_changed else None}

    def sync(self, values: dict[str, Any], *, turn: int, level: int) -> dict[str, Any]:
        """The kernel's memory variables (see ``WorkingMemory.sync``), saved when anything changed. Never raises."""
        out = self.working.sync(values, turn=turn, level=level)
        if any(out[k] for k in ("hypotheses", "findings", "removed")) or out["plan"] is not None or                 out["goal"] is not None:
            self.working.save()
        return out

    def record_step(self, row: dict[str, Any]) -> None:
        self.timeline.append({"kind": "step", **row})

    def record_act(self, row: dict[str, Any]) -> None:
        self.timeline.append({"kind": "act", **row})

    # --- reads ---------------------------------------------------------------------------------------------
    def blocks(self, objects: list[Any] | None = None) -> dict[str, str]:
        """The agent's memory block (config memory.in_context), each section under its cap, in context order: the five
        kernel variables, SKILLS by name and purpose. The lessons are not in it: only the curator reads them."""
        b = self.working.blocks()
        return {"goal": b["goal"], "plan": b["plan"], "skills": b["skills"], "hypotheses": b["hypotheses"],
                "findings": b["findings"]}

    def recall(self, query: str, scope: str = "all", cap_tokens: int = 1500) -> tuple[str, int, list[str]]:
        """``recall_text`` over this game's timeline and the skills."""
        return recall_text(self.timeline, self.skills, query, scope, cap_tokens)


def recall_text(timeline: Timeline, skills: SkillBook, query: str, scope: str = "all",
                cap_tokens: int = 1500) -> tuple[str, int, list[str]]:
    """What ``recall()`` in the kernel prints (at most ``cap_tokens``), its number of hits, and the names of the skills it
    found (the kernel loads them into ``SKILLS``). Two places to search: "timeline" (every step and what it changed:
    "#12", "12-20", "level 1" or words) and "skills" (words; "" lists every skill by name, one line each, and loads
    none); "all" is both."""
    scope = (scope or "all").lower()
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {list(SCOPES)}")
    q = (query or "").strip()
    sections: list[tuple[str, list[str]]] = []
    found: list[str] = []
    if scope in ("all", "timeline"):
        rows = timeline.search(q) if q else timeline.rows[-30:]
        sections.append(("timeline", [render_row(r) for r in rows]))
    if scope in ("all", "skills"):
        if q:
            # The skills a query word names are loaded; if none is named, the best TEXT_MATCHES text matches.
            matches = skills.find(q)
            words = q.lower().split()
            named = [n for n in matches if any(w in n.lower() for w in words)]
            found = (named or matches[:TEXT_MATCHES])[:LOADED_SKILLS]
            lines = [f"### {n}\n{skills.text(n)}" for n in found]
            rest = [n for n in matches if n not in found]
            if rest:
                lines.append(f"(also matching, not loaded: {', '.join(rest)})")
            sections.append(("skills", lines))
        else:
            sections.append(("skills", [f"- {n}: {skills.purpose(n)}" for n in skills.names()]))
    hits = sum(len(v) for _, v in sections)
    out: list[str] = [f"recall {q!r} in {scope}: {hits} hit{'s' if hits != 1 else ''}"]
    room = cap_tokens * 4 - 60   # chars, with space for one "(+ more ...)" note
    used = len(out[0])
    cut = False
    for name, lines in sections:
        if not lines or cut:
            continue
        block = [f"## {name} ({len(lines)})"]
        for line in lines[-60:]:
            if used + sum(len(x) + 1 for x in block) + len(line) + 1 > room:
                cut = True
                break
            block.append(line)
        if len(block) > 1:
            out += block
            used += sum(len(x) + 1 for x in block)
        if cut:
            out.append(f"(more hits in {name} and later scopes: narrow the query or the scope)")
    return "\n".join(out), hits, found
