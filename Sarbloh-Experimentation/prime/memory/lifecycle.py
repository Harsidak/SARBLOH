"""``GameMemory``: all memories of one game, and what happens to them at a level-up and at a new game.

Files: ``<root>/<game>/timeline.jsonl`` (#1), ``<root>/<game>/working.json`` (#2 #3 #5 #6 #7),
``<root>/<game>/levels.jsonl`` (what each level-up erased, still searchable by ``recall``), and, shared by all games,
``<root>/lessons.json`` (#8) and ``<root>/skills.json``.

Level-up: hypotheses the agent marked ``verified`` and its findings become ``rule`` nodes of the lessons graph
(``refuted`` hypotheses become ``refuted`` nodes, the goal a ``goal`` node), each linked to the shapes it names as
"obj N"; the goal gets "confirmed: won level N"; then plan, hypotheses, findings and open questions are erased.
New game: the timeline is archived and everything except the lessons graph and the skills is erased.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from prime.memory.lessons import LessonsGraph, SkillBook, shape_id
from prime.memory.store import append_jsonl, clip, read_jsonl
from prime.memory.timeline import Timeline, render_row
from prime.memory.working import WorkingMemory

SCOPES = ("all", "timeline", "hypotheses", "findings", "goal", "lessons", "skills")
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
        self.working = WorkingMemory(self.dir / "working.json", self.caps,
                                     goal_versioning=bool(cfg.get("goal_versioning", False)))
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
        goals = copy.deepcopy(w.goals)               # E021: the goals before the rivals are cleared
        line = w.confirm_goal(level_won)
        promoted: list[dict[str, Any]] = []
        if self.lessons is not None:
            shapes = _shapes(objects)
            items = [("rule" if h["status"] == "verified" else "refuted", h["text"]) for h in w.hypotheses
                     if h["status"] in ("verified", "refuted")]
            items += [("rule", f["text"]) for f in w.findings]
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
        extra = {"goals": goals} if w.goal_versioning else {}
        append_jsonl(self.dir / "levels.jsonl", {"level": level_won - 1, "step": step, "goal": w.goal,
                                                 "confirmation": line, **extra, **gone})
        self.timeline.append({"kind": "level_up", "level": level_won, "i": step, "goal": w.goal})
        self.save()
        return {"confirmation": line, "promoted": promoted, "erased": {k: len(v) if isinstance(v, list) else bool(v)
                                                                      for k, v in gone.items()}}

    # --- writes from the act tool --------------------------------------------------------------------------
    def write_act(self, args: dict[str, Any], *, turn: int, level: int) -> dict[str, Any]:
        """plan (required), hypotheses, findings, goal. Validates everything before writing anything. With
        ``goal_versioning`` (E021) ``goal`` may be a list and the result carries ``goals``: one event per change."""
        plan = str(args.get("plan") or "").strip()
        if not plan:
            raise ValueError("plan is required: your next steps and why, in one or two sentences")
        w = self.working
        snapshot = (w.plan, [dict(h, history=list(h["history"])) for h in w.hypotheses], list(w.findings), w.goal,
                    w.next_h)
        try:
            hyp = w.update_hypotheses(args.get("hypotheses"), turn=turn, level=level)
            found = w.add_findings(args.get("findings"), turn=turn, level=level)
            goals = w.update_goals(args.get("goal"), turn=turn, level=level) if w.goal_versioning else []
        except Exception:
            w.plan, w.hypotheses, w.findings, w.goal, w.next_h = snapshot
            raise
        w.set_plan(plan)
        if w.goal_versioning:
            goal_changed = bool(goals)
        else:
            goal_changed = w.set_goal(args.get("goal") or "")
        w.save()
        return {"plan": plan, "hypotheses": hyp, "findings": found, "goal": w.goal if goal_changed else None,
                "goals": goals}

    def record_step(self, row: dict[str, Any]) -> None:
        self.timeline.append({"kind": "step", **row})

    def record_act(self, row: dict[str, Any]) -> None:
        self.timeline.append({"kind": "act", **row})

    # --- reads ---------------------------------------------------------------------------------------------
    def blocks(self, objects: list[Any]) -> dict[str, str]:
        """The pinned context blocks, each under its cap, in context order."""
        b = self.working.blocks()
        lessons = ""
        if self.lessons is not None:
            on_screen = [shape_id(o.label()[0], o.hash) for o in objects or []]
            lessons = self.lessons.subgraph(on_screen, self.caps["lessons"])
        return {"goal": b["goal"], "plan": b["plan"], "skills": self.skills.render(self.game, self.caps["skills"]),
                "hypotheses": b["hypotheses"], "findings": b["findings"], "lessons": lessons,
                "questions": b["questions"]}

    def recall(self, query: str, scope: str = "all", cap_tokens: int = 1500) -> tuple[str, int]:
        """Text for the agent (at most ``cap_tokens``) and the number of hits."""
        scope = (scope or "all").lower()
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {list(SCOPES)}")
        q = (query or "").strip()
        words = [x for x in q.lower().split() if x]
        match = (lambda s: all(x in s.lower() for x in words)) if words else (lambda s: True)
        sections: list[tuple[str, list[str]]] = []
        if scope in ("all", "timeline"):
            rows = self.timeline.search(q) if q else self.timeline.rows[-30:]
            sections.append(("timeline", [render_row(r) for r in rows]))
        if scope in ("all", "hypotheses"):
            past = [h for lv in read_jsonl(self.dir / "levels.jsonl") for h in lv.get("hypotheses", [])]
            lines = [f"(level {h.get('level')}) {self.working.hyp_line(h)}" for h in past + self.working.hypotheses
                     if match(h["text"] + " " + h["status"] + " " + h["id"])]
            sections.append(("hypotheses", lines))
        if scope in ("all", "findings"):
            past = [f for lv in read_jsonl(self.dir / "levels.jsonl") for f in lv.get("findings", [])]
            sections.append(("findings", [f"(level {f.get('level')}) {f['text']}" for f in past + self.working.findings
                                          if match(f["text"])]))
        if scope in ("all", "goal"):
            lines = ([*self.working.goal_history(), *self.working.goal_log] if self.working.goal_versioning else
                     [self.working.goal, *self.working.goal_log])
            sections.append(("goal", [x for x in lines if x and match(x)]))
        if scope in ("all", "lessons") and self.lessons is not None:
            sections.append(("lessons", self.lessons.search(q)))
        if scope in ("all", "skills"):
            sections.append(("skills", self.skills.search(q)))
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
        return "\n".join(out), hits
