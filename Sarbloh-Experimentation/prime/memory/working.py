"""The agent's working memory: #2 plan, #3 hypotheses, #5 open questions, #6 findings, #7 goal.

The agent writes plan, hypotheses, findings and goal through the ``act`` tool; the curator writes the open questions.
The host checks the shape of what is written and judges nothing: a hypothesis is "verified" because the agent says so.
Every block is rendered under its token cap (chars / 4); a block over its cap loses its oldest lines, which stay in
the file and in ``recall``.

E021 (switch ``memory.goal_versioning``, off = E008): the goal becomes up to 3 open goals g1..g3 (one ``active``, the
rest ``candidate``) plus the ``refuted`` ones, each with a version, evidence, why and a history. ``goal`` mirrors the
active goal's text, so the level-up and the lessons graph read it as before.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from prime.memory.store import clip, fit_lines, read_json, write_json

STATUSES = ("proposed", "verified", "refuted")
_HYP_STR = re.compile(r"^\s*(?:(h\d+)\b[\s:.-]*)?(?:\[?(proposed|verified|refuted)\]?\b[\s:.-]*)?(.*)$", re.IGNORECASE | re.DOTALL)
# E021: versioned goals (switch ``memory.goal_versioning``). The agent sets these; "won" is added by the host.
GOAL_STATUSES = ("active", "candidate", "refuted")
MAX_OPEN_GOALS = 3
_GOAL_STR = re.compile(r"^\s*(?:(g\d+)\b[\s:.-]*)?(?:\[?(active|candidate|refuted)\]?\b[\s:.-]*)?(.*)$",
                       re.IGNORECASE | re.DOTALL)


def _evidence(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, (int, str)):
        raw = [raw]
    out = []
    for x in raw if isinstance(raw, list) else []:
        for n in re.findall(r"\d+", str(x)):
            out.append(int(n))
    return sorted(set(out))


def parse_hypothesis(item: Any) -> dict[str, Any]:
    """{"id"?, "text", "status", "evidence"} from a dict or a string such as "h2 verified: A1 moves the block up"."""
    if isinstance(item, dict):
        text = str(item.get("text") or item.get("hypothesis") or item.get("content") or "").strip()
        status = str(item.get("status") or "proposed").strip().lower()
        hid = str(item.get("id") or "").strip() or None
        ev = _evidence(item.get("evidence"))
    else:
        m = _HYP_STR.match(str(item))
        hid, status, text = (m.group(1), (m.group(2) or "proposed").lower(), m.group(3).strip()) if m else \
            (None, "proposed", str(item).strip())
        ev = []
    if status not in STATUSES:
        raise ValueError(f"hypothesis status must be one of {list(STATUSES)}, got {status!r}")
    if not text and not hid:
        raise ValueError("a hypothesis needs text (or the id of an existing one, to change its status)")
    return {"id": hid.lower() if hid else None, "text": text, "status": status, "evidence": ev}


def parse_goal(item: Any) -> dict[str, Any]:
    """E021: {"id"?, "text", "status"?, "evidence", "why"} from a dict or a string such as "g2 refuted: touch obj 4".
    ``status`` is None when not given (the caller picks the default)."""
    if isinstance(item, dict):
        text = str(item.get("text") or item.get("goal") or "").strip()
        status = str(item.get("status") or "").strip().lower() or None
        gid = str(item.get("id") or "").strip() or None
        ev = _evidence(item.get("evidence"))
        why = str(item.get("why") or "").strip()
    else:
        m = _GOAL_STR.match(str(item))
        gid, status, text = (m.group(1), (m.group(2) or "").lower() or None, m.group(3).strip()) if m else \
            (None, None, str(item).strip())
        ev, why = [], ""
    if status is not None and status not in GOAL_STATUSES:
        raise ValueError(f"goal status must be one of {list(GOAL_STATUSES)}, got {status!r}")
    if not text and not gid:
        raise ValueError("a goal needs text (or the id of an existing one, to change its status)")
    return {"id": gid.lower() if gid else None, "text": text, "status": status, "evidence": ev, "why": why}


class WorkingMemory:
    def __init__(self, path: Path, caps: dict[str, int], goal_versioning: bool = False) -> None:
        self.path = Path(path)
        self.caps = caps
        self.goal_versioning = bool(goal_versioning)
        d = read_json(self.path, {})
        self.plan: str = d.get("plan", "")
        self.hypotheses: list[dict[str, Any]] = d.get("hypotheses", [])
        self.findings: list[dict[str, Any]] = d.get("findings", [])
        self.goal: str = d.get("goal", "")
        self.goal_log: list[str] = d.get("goal_log", [])
        self.questions: list[str] = d.get("questions", [])
        self.next_h: int = d.get("next_h", 1)
        self.goals: list[dict[str, Any]] = d.get("goals", [])     # E021
        self.next_g: int = d.get("next_g", 1)

    def save(self) -> None:
        write_json(self.path, {"plan": self.plan, "hypotheses": self.hypotheses, "findings": self.findings,
                               "goal": self.goal, "goal_log": self.goal_log, "questions": self.questions,
                               "next_h": self.next_h, "goals": self.goals, "next_g": self.next_g})

    # --- writes --------------------------------------------------------------------------------------------
    def set_plan(self, text: str) -> None:
        self.plan = str(text or "").strip()

    def update_hypotheses(self, items: Any, *, turn: int, level: int) -> list[dict[str, Any]]:
        """Adds new hypotheses and changes existing ones (matched by id, else by text). Returns one event per change:
        {"id", "text", "status", "prev_status" (None when new), "evidence"}."""
        if items is None:
            return []
        if not isinstance(items, list):
            items = [items]
        parsed = [parse_hypothesis(x) for x in items]       # all or nothing: a bad item raises before any write
        events = []
        for h in parsed:
            old = None
            if h["id"]:
                old = next((x for x in self.hypotheses if x["id"] == h["id"]), None)
                if old is None and not h["text"]:
                    raise ValueError(f"no hypothesis {h['id']!r}; ids: {[x['id'] for x in self.hypotheses]}")
            if old is None and h["text"]:
                old = next((x for x in self.hypotheses if x["text"].lower() == h["text"].lower()), None)
            if old is None:
                new = {"id": f"h{self.next_h}", "text": h["text"], "status": h["status"], "evidence": h["evidence"],
                       "level": level, "turn": turn, "history": [[turn, h["status"]]]}
                self.next_h += 1
                self.hypotheses.append(new)
                events.append({**{k: new[k] for k in ("id", "text", "status", "evidence")}, "prev_status": None})
                continue
            prev = old["status"]
            changed = prev != h["status"] or (h["text"] and h["text"] != old["text"]) or \
                (h["evidence"] and h["evidence"] != old["evidence"])
            if h["text"]:
                old["text"] = h["text"]
            old["evidence"] = sorted(set(old["evidence"]) | set(h["evidence"]))
            if prev != h["status"]:
                old["status"] = h["status"]
                old["history"].append([turn, h["status"]])
            if changed:
                events.append({**{k: old[k] for k in ("id", "text", "status", "evidence")}, "prev_status": prev})
        return events

    def add_findings(self, items: Any, *, turn: int, level: int) -> list[dict[str, Any]]:
        if items is None:
            return []
        if not isinstance(items, list):
            items = [items]
        added = []
        for x in items:
            text = str(x.get("text") if isinstance(x, dict) else x).strip()
            if not text or any(f["text"].lower() == text.lower() for f in self.findings):
                continue
            f = {"text": text, "evidence": _evidence(x.get("evidence")) if isinstance(x, dict) else [],
                 "level": level, "turn": turn}
            self.findings.append(f)
            added.append(f)
        return added

    def set_goal(self, text: str) -> bool:
        text = str(text or "").strip()
        if not text or text == self.goal:
            return False
        self.goal = text
        return True

    def update_goals(self, raw: Any, *, turn: int, level: int) -> list[dict[str, Any]]:
        """E021. A string is the active goal's new text (E008's call shape); a list adds rivals, switches the active
        goal or refutes one (matched by id, else by text). All or nothing: a bad item raises before any write. At
        most one goal is active and at most ``MAX_OPEN_GOALS`` are open (active + candidate). A refuted goal set back
        to active or candidate is allowed and flagged ``reproposed``. Returns one event per changed goal:
        {"id", "version", "text", "status", "prev_status", "kind", "why", "evidence", "reproposed"}."""
        if raw is None or (isinstance(raw, (str, list)) and not raw):
            return []
        if isinstance(raw, str):
            if not raw.strip():
                return []
            items = [{"id": None, "text": raw.strip(), "status": "active", "evidence": [], "why": "", "plain": True}]
        else:
            items = [parse_goal(x) for x in (raw if isinstance(raw, list) else [raw])]
        goals = copy.deepcopy(self.goals)
        next_g = self.next_g
        touched: dict[str, dict[str, Any]] = {}       # id -> {"prev_status", "prev_version", "new"}
        activated: list[str] = []
        for it in items:
            old = None
            if it["id"]:
                old = next((g for g in goals if g["id"] == it["id"]), None)
                if old is None and not it["text"]:
                    raise ValueError(f"no goal {it['id']!r}; ids: {[g['id'] for g in goals]}")
            if old is None and it["text"]:
                old = next((g for g in goals if g["text"].lower() == it["text"].lower()), None)
            if old is None and it.get("plain"):
                old = next((g for g in goals if g["status"] == "active"), None)   # a new version of the active goal
            if old is None:
                status = it["status"] or ("candidate" if any(g["status"] == "active" for g in goals) else "active")
                old = {"id": f"g{next_g}", "text": it["text"], "status": status, "version": 1,
                       "evidence": it["evidence"], "why": it["why"], "level": level, "turn": turn,
                       "history": [[turn, 1, status, it["text"]]]}
                next_g += 1
                goals.append(old)
                touched[old["id"]] = {"prev_status": None, "prev_version": 0, "new": True}
            else:
                mark = touched.setdefault(old["id"], {"prev_status": old["status"], "prev_version": old["version"],
                                                      "prev_evidence": list(old["evidence"]),
                                                      "prev_why": old.get("why", ""), "new": False})
                status = it["status"] or old["status"]
                if it["text"] and it["text"].lower() != old["text"].lower():     # a case change is not a version
                    old["text"] = it["text"]
                    old["version"] += 1
                old["evidence"] = sorted(set(old["evidence"]) | set(it["evidence"]))
                if it["why"]:
                    old["why"] = it["why"]
                old["status"] = status
                if (old["status"], old["version"]) != (mark["prev_status"], mark["prev_version"]):
                    old["history"].append([turn, old["version"], old["status"], old["text"]])
            if old["status"] == "active" and old["id"] not in activated:
                activated.append(old["id"])
        if len(activated) > 1:
            raise ValueError(f"only one goal can be active, got {activated}: make the others candidate")
        if activated:
            for g in goals:
                if g["status"] == "active" and g["id"] != activated[0]:
                    touched.setdefault(g["id"], {"prev_status": "active", "prev_version": g["version"],
                                                 "prev_evidence": list(g["evidence"]), "prev_why": g.get("why", ""),
                                                 "new": False})
                    g["status"] = "candidate"
                    g["history"].append([turn, g["version"], "candidate", g["text"]])
        open_ = [g["id"] for g in goals if g["status"] in ("active", "candidate")]
        if len(open_) > MAX_OPEN_GOALS:
            raise ValueError(f"at most {MAX_OPEN_GOALS} open goals (active + candidate), got {open_}: refute one first")
        events = []
        for g in goals:
            mark = touched.get(g["id"])
            if mark is None:
                continue
            prev = mark["prev_status"]
            if mark["new"]:
                kind = "new"
            elif g["status"] == "refuted" and prev != "refuted":
                kind = "refuted"
            elif g["status"] == "active" and prev != "active":
                kind = "switch"
            elif g["version"] != mark["prev_version"]:
                kind = "version"
            elif g["status"] != prev:
                kind = "status"
            elif g["evidence"] != mark["prev_evidence"] or g.get("why", "") != mark["prev_why"]:
                kind = "evidence"
            else:
                continue
            events.append({"id": g["id"], "version": g["version"], "text": g["text"], "status": g["status"],
                           "prev_status": prev, "kind": kind, "why": g.get("why", ""), "evidence": g["evidence"],
                           "reproposed": prev == "refuted" and g["status"] != "refuted"})
        self.goals, self.next_g = goals, next_g
        active = next((g for g in goals if g["status"] == "active"), None)
        self.goal = active["text"] if active else ""
        return events

    def set_questions(self, questions: list[str]) -> None:
        self.questions = [str(q).strip() for q in questions if str(q).strip()][:3]

    def confirm_goal(self, level: int) -> str:
        if self.goal_versioning:
            # E021: the active goal is marked won and stays active; rivals are cleared; refuted goals stay.
            active = next((g for g in self.goals if g["status"] == "active"), None)
            if active is not None:
                active.setdefault("won", []).append(level)
            self.goals = [g for g in self.goals if g["status"] != "candidate"]
            line = f"confirmed: won level {level}" + (f" with goal {active['id']} v{active['version']}: "
                                                      f"{active['text']}" if active else " (no active goal)")
            self.goal_log.append(line)
            return line
        line = f"confirmed: won level {level}" + (f" with goal: {self.goal}" if self.goal else "")
        self.goal_log.append(line)
        return line

    def clear_level(self) -> dict[str, Any]:
        """Erases #2, #3, #5 and #6 (the goal stays). Returns what was erased, for promotion and the log."""
        gone = {"plan": self.plan, "hypotheses": self.hypotheses, "findings": self.findings,
                "questions": self.questions}
        self.plan, self.hypotheses, self.findings, self.questions = "", [], [], []
        return gone

    def clear_game(self) -> None:
        self.clear_level()
        self.goal, self.goal_log, self.next_h = "", [], 1
        self.goals, self.next_g = [], 1

    # --- reads ---------------------------------------------------------------------------------------------
    def hyp_line(self, h: dict[str, Any]) -> str:
        ev = f" (evidence {', '.join('#' + str(e) for e in h['evidence'][:8])})" if h.get("evidence") else ""
        return f"{h['id']} [{h['status']}] {h['text']}{ev}"

    def goal_line(self, g: dict[str, Any]) -> str:
        won = f" (won level {', '.join(str(x) for x in g['won'])})" if g.get("won") else ""
        ev = f" (evidence {', '.join('#' + str(e) for e in g['evidence'][:6])})" if g.get("evidence") else ""
        why = f" why: {g['why']}" if g.get("why") else ""
        return f"{g['id']} v{g['version']} [{g['status']}] {g['text']}{won}{ev}{why}"

    def goal_history(self) -> list[str]:
        """E021: every version and status change of every goal, oldest first, for ``recall`` scope goal."""
        return [f"{g['id']} turn {t} v{v} [{s}] {text}" for g in self.goals for t, v, s, text in g["history"]]

    def _goal_block(self) -> str:
        if not self.goals and not self.goal_log:
            return ""
        per_line = max(30, self.caps.get("goal_versioned", 250) // 6)
        active = [g for g in self.goals if g["status"] == "active"]
        rivals = [g for g in self.goals if g["status"] == "candidate"]
        refuted = sorted((g for g in self.goals if g["status"] == "refuted"), key=lambda g: g["history"][-1][0])
        lines = [clip(self.goal_line(g), per_line) for g in active + rivals + refuted[-2:]]
        if not active and self.goals:
            lines.insert(0, "(no active goal: make one active)")
        if len(refuted) > 2:
            lines.append(f"(+{len(refuted) - 2} older refuted: recall scope goal)")
        if self.goal_log:
            lines.append(clip(self.goal_log[-1], per_line))
        return "\n".join(lines)

    def blocks(self) -> dict[str, str]:
        """Rendered blocks under their caps; empty blocks are ""."""
        c = self.caps
        goal = ""
        if self.goal_versioning:
            goal = self._goal_block()
        elif self.goal or self.goal_log:
            goal = clip(self.goal or "(not stated yet)", c["goal"] - 20)
            if self.goal_log:
                goal += "\n" + self.goal_log[-1]
        hyps = fit_lines([self.hyp_line(h) for h in self.hypotheses], c["hypotheses"])
        finds = fit_lines([f"- {f['text']}" + (f" (evidence {', '.join('#' + str(e) for e in f['evidence'][:6])})"
                                               if f.get("evidence") else "") for f in self.findings], c["findings"])
        qs = fit_lines([f"- {q}" for q in self.questions[:3]], c["questions"], older_note="")
        return {"goal": goal, "plan": clip(self.plan, c["plan"]), "hypotheses": "\n".join(hyps),
                "findings": "\n".join(finds), "questions": "\n".join(x for x in qs if x)}
