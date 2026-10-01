"""``GameMemory``: all memories of one game, and what happens to them at a level-up and at a new game.

Files: ``<root>/<game>/timeline.jsonl`` (#1), ``<root>/<game>/working.json`` (#2 #3 #5 #6 #7),
``<root>/<game>/levels.jsonl`` (what each level-up erased, still searchable by ``recall``), and, shared by all games,
``<root>/lessons.json`` (#8) and ``<root>/skills.json``.

Level-up: hypotheses the agent marked ``verified`` and its findings become ``rule`` nodes of the lessons graph
(``refuted`` hypotheses become ``refuted`` nodes, the goal a ``goal`` node), each linked to the shapes it names as
"obj N"; the goal gets "confirmed: won level N"; then plan, hypotheses, findings and open questions are erased.
New game: the timeline is archived and everything except the lessons graph and the skills is erased.

E022 (switch ``level_review``): after a level-up the agent runs one hidden review of the whole level
(``prime.agent.level_review``); ``add_level_review`` writes it into the working memory (block "levels"), the lessons
graph (source ``level_review``) and ``levels.jsonl`` (a ``review`` row that ``recall`` returns).
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
                                     goal_versioning=bool(cfg.get("goal_versioning", False)),
                                     lock_after=int(cfg.get("goal_lock_after_level", 0) or 0),
                                     history_shown=int(cfg.get("goal_history_shown", 5)))
        self.lessons = LessonsGraph.shared(self.root / "lessons.json") if cfg.get("lessons", True) else None
        self.skills = SkillBook.shared(self.root / "skills.json")
        self.promoted = 0
        self.level_objects: dict[int, list[Any]] = {}   # E022: the objects of each level won, for "obj N" links

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
        self.level_objects = {level_won - 1: list(objects or [])}
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
                promoted.append(self._promote(kind, text, level_won - 1, shapes, "level_up"))
            self.promoted += len(promoted)
        gone = w.clear_level()
        extra = {"goals": goals} if w.goal_versioning else {}
        append_jsonl(self.dir / "levels.jsonl", {"level": level_won - 1, "step": step, "goal": w.goal,
                                                 "confirmation": line, **extra, **gone})
        self.timeline.append({"kind": "level_up", "level": level_won, "i": step, "goal": w.goal})
        self.save()
        return {"confirmation": line, "promoted": promoted, "erased": {k: len(v) if isinstance(v, list) else bool(v)
                                                                      for k, v in gone.items()}}

    def _promote(self, kind: str, text: str, level: int, shapes: dict[int, Any], source: str) -> dict[str, Any]:
        """One lessons-graph node, linked to the shapes its text names as "obj N"."""
        nid = self.lessons.add_node(kind, text, game=self.game, level=level, source=source)
        linked = []
        for m in _OBJ.finditer(text):
            o = shapes.get(int(m.group(1)))
            if o is not None:
                sid = self.lessons.add_shape(o.label()[0], o.hash, o.label(), game=self.game, level=level)
                self.lessons.add_edge(nid, sid, "about")
                linked.append(sid)
        return {"node": nid, "type": kind, "text": clip(text, 60), "shapes": linked}

    def level_record(self, level: int) -> dict[str, Any]:
        """E022: what the review reads about level ``level`` (0-based): its steps and acts from the timeline, and what
        the working memory held when it ended (the ``levels.jsonl`` row)."""
        steps = [r for r in self.timeline.rows if r.get("kind") == "step" and r.get("level") == level]
        acts = [r for r in self.timeline.rows if r.get("kind") == "act" and r.get("level") == level]
        rows = [r for r in read_jsonl(self.dir / "levels.jsonl") if r.get("kind") != "review"]
        ended = next((r for r in reversed(rows) if r.get("level") == level), rows[-1] if rows else {})
        return {"level": level, "steps": steps, "acts": acts, "ended": ended}

    def add_level_review(self, level: int, review: dict[str, Any]) -> dict[str, Any]:
        """E022: writes a parsed review of level ``level`` (0-based) into the memories. Returns what was written."""
        w = self.working
        clean = lambda xs, n: [clip(" ".join(str(x).split()), 40) for x in (xs or []) if str(x).strip()][:n]  # noqa: E731
        rules = []
        for r in review.get("rules") or []:
            text = str((r.get("text") if isinstance(r, dict) else r) or "").strip()
            status = str(r.get("status") or "verified").lower() if isinstance(r, dict) else "verified"
            if text and status in ("verified", "refuted"):
                rules.append({"text": clip(text, 40), "status": status})
        rec = {"level": level + 1, "steps": len(self.level_record(level)["steps"]),
               "summary": clip(str(review.get("summary") or ""), 80),
               "win_condition": clip(str(review.get("win_condition") or ""), 40),
               "rules": rules[:5], "mistakes": clean(review.get("mistakes"), 3),
               "carry_over": clean(review.get("carry_over"), 3)}
        if not (rec["summary"] or rec["win_condition"] or rec["rules"] or rec["carry_over"]):
            raise ValueError("the review has no summary, win condition, rules or carry-over")
        w.reviews.append(rec)
        promoted: list[dict[str, Any]] = []
        if self.lessons is not None:
            shapes = _shapes(self.level_objects.get(level, []))
            items = [("rule" if r["status"] == "verified" else "refuted", r["text"]) for r in rec["rules"]]
            if rec["win_condition"]:
                items.append(("goal", f"won a level when: {rec['win_condition']}"))
            promoted = [self._promote(kind, text, level, shapes, "level_review") for kind, text in items]
        append_jsonl(self.dir / "levels.jsonl", {"kind": "review", **rec, "level": level})
        self.save()
        return {"review": rec, "promoted": promoted}

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
        raw_goal = args.get("goal")
        # E021 amendment: after level ``lock_after`` the goal argument is ignored (the act still runs).
        locked = bool(w.goal_versioning and w.goal_locked and raw_goal not in (None, "", []))
        try:
            hyp = w.update_hypotheses(args.get("hypotheses"), turn=turn, level=level)
            found = w.add_findings(args.get("findings"), turn=turn, level=level)
            goals = w.update_goals(raw_goal, turn=turn, level=level) if w.goal_versioning and not locked else []
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
                "goals": goals, "goal_locked": locked}

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
        return {"goal": b["goal"], "levels": b["levels"], "plan": b["plan"], "skills": self.skills.render(self.game, self.caps["skills"]),
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
            rows = read_jsonl(self.dir / "levels.jsonl")
            past = [f for lv in rows for f in lv.get("findings", [])]
            reviews = [f"(level {r['level'] + 1} review) {self.working.review_line(r | {'level': r['level'] + 1})}"
                       + (f" Summary: {r['summary']}" if r.get("summary") else "")
                       for r in rows if r.get("kind") == "review"]
            sections.append(("findings", [f"(level {f.get('level')}) {f['text']}" for f in past + self.working.findings
                                          if match(f["text"])] + [x for x in reviews if match(x)]))
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
