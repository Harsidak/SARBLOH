"""The agent's working memory: #2 plan, #3 hypotheses, #5 open questions, #6 findings, #7 goal.

The agent writes plan, hypotheses, findings and goal through the ``act`` tool; the curator writes the open questions.
The host checks the shape of what is written and judges nothing: a hypothesis is "verified" because the agent says so.
Every block is rendered under its token cap (chars / 4); a block over its cap loses its oldest lines, which stay in
the file and in ``recall``.

E021 (switch ``memory.goal_versioning``, off = E008): the goal becomes up to 3 open goals g1..g3 (one ``active``, the
rest ``candidate``) plus the ``refuted`` ones, each with a version, evidence, why and a history. ``goal`` mirrors the
active goal's text, so the level-up and the lessons graph read it as before. Each history entry keeps the reason
(``why``) for its version; the block shows the last ``history_shown`` changes. Amendment 1 (owner): once level
``lock_after`` (1) is won with an active goal, the goal is locked for the rest of the game (``goal_locked``).

E022 (switch ``memory.level_review``): ``reviews`` holds the host's review of each level won; the block "levels" shows
the last 2. Kept for the game, erased at a new game.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from harness.memory.store import clip, fit_lines, read_json, write_json

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
    def __init__(self, path: Path, caps: dict[str, int], goal_versioning: bool = False, lock_after: int = 0,
                 history_shown: int = 5) -> None:
        self.path = Path(path)
        self.caps = caps
        self.goal_versioning = bool(goal_versioning)
        self.lock_after = int(lock_after or 0)          # E021 amendment: 0 = never lock
        self.history_shown = int(history_shown)
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
        self.levels_won: int = d.get("levels_won", 0)
        self.reviews: list[dict[str, Any]] = d.get("reviews", [])   # E022
        self.goal_past: list[list[Any]] = d.get("goal_past", [])   # E021: [goal id, history entry] of cleared rivals

    def save(self) -> None:
        write_json(self.path, {"plan": self.plan, "hypotheses": self.hypotheses, "findings": self.findings,
                               "goal": self.goal, "goal_log": self.goal_log, "questions": self.questions,
                               "next_h": self.next_h, "goals": self.goals, "next_g": self.next_g,
                               "levels_won": self.levels_won, "reviews": self.reviews,
                               "goal_past": self.goal_past})

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
        if self.goal_locked:
            raise ValueError(f"the goal is locked after level {self.lock_after}")
        if isinstance(raw, str):
            if not raw.strip():
                return []
            items = [{"id": None, "text": raw.strip(), "status": "active", "evidence": [], "why": "", "plain": True}]
        else:
            items = [parse_goal(x) for x in (raw if isinstance(raw, list) else [raw])]
        goals = copy.deepcopy(self.goals)
        next_g = self.next_g
        seq = [self._last_seq()]

        def entry(*fields: Any) -> list[Any]:      # [turn, version, status, text, why, seq]: seq orders one turn
            seq[0] += 1
            return [*fields, seq[0]]
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
                       "history": [entry(turn, 1, status, it["text"], it["why"])]}
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
                    old["history"].append(entry(turn, old["version"], old["status"], old["text"], it["why"]))
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
                    g["history"].append(entry(turn, g["version"], "candidate", g["text"],
                                              f"{activated[0]} became active"))
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
            self.levels_won = max(self.levels_won, level)
            turn = max((e[0] for _, e in self.history_entries()), default=0)
            seq = self._last_seq()
            for g in self.goals:          # cleared rivals keep their history (and a last line saying why they went)
                if g["status"] == "candidate":
                    seq += 1
                    self.goal_past += [[g["id"], e] for e in g["history"]]
                    self.goal_past.append([g["id"], [turn, g["version"], "cleared", g["text"],
                                                     f"level {level} was won with another goal", seq]])
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
        self.levels_won, self.reviews, self.goal_past = 0, [], []

    @property
    def goal_locked(self) -> bool:
        """E021 amendment: level ``lock_after`` is won and a goal is active. If it was won with no active goal, the
        agent may set one, which then locks."""
        return (self.goal_versioning and self.lock_after > 0 and self.levels_won >= self.lock_after
                and any(g["status"] == "active" for g in self.goals))

    # --- reads ---------------------------------------------------------------------------------------------
    def hyp_line(self, h: dict[str, Any]) -> str:
        ev = f" (evidence {', '.join('#' + str(e) for e in h['evidence'][:8])})" if h.get("evidence") else ""
        return f"{h['id']} [{h['status']}] {h['text']}{ev}"

    def goal_line(self, g: dict[str, Any]) -> str:
        won = f" (won level {', '.join(str(x) for x in g['won'])})" if g.get("won") else ""
        ev = f" (evidence {', '.join('#' + str(e) for e in g['evidence'][:6])})" if g.get("evidence") else ""
        why = f" why: {g['why']}" if g.get("why") else ""
        return f"{g['id']} v{g['version']} [{g['status']}] {g['text']}{won}{ev}{why}"

    @staticmethod
    def _history_line(gid: str, entry: list[Any]) -> str:
        t, v, s, text = entry[:4]
        why = entry[4] if len(entry) > 4 else ""
        return f"{gid} turn {t} v{v} [{s}] {text}" + (f" (why: {why})" if why else "")

    def _last_seq(self) -> int:
        return max((e[5] for _, e in self.history_entries() if len(e) > 5), default=0)

    def history_entries(self) -> list[tuple[str, list[Any]]]:
        """Every (goal id, history entry), oldest first (by turn, then by write order)."""
        live = [(g["id"], e) for g in self.goals for e in g["history"]]
        return sorted([(gid, e) for gid, e in self.goal_past] + live,
                      key=lambda x: (x[1][0], x[1][5] if len(x[1]) > 5 else 0))

    def goal_history(self) -> list[str]:
        """E021: every version and status change of every goal with its reason, oldest first, for ``recall``."""
        return [self._history_line(gid, e) for gid, e in self.history_entries()]

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
        if self.goal_locked:
            lines.append(f"(locked: level {self.lock_after} is won, so this goal is kept for the rest of the game; "
                         "act's goal is ignored)")
        hist = self.history_entries()
        if self.history_shown > 0 and len(hist) > 1:
            shown = hist[-self.history_shown:]
            older = len(hist) - len(shown)
            lines.append(f"Goal changes, last {len(shown)}" + (f" (+{older} older: recall scope goal)" if older else "")
                         + ":")
            lines += [clip("- " + self._history_line(gid, e), per_line) for gid, e in shown]
        return "\n".join(lines)

    def review_line(self, r: dict[str, Any]) -> str:
        """E022: one level review, compact."""
        parts = [f"Level {r['level']} won in {r.get('steps', '?')} steps."]
        if r.get("win_condition"):
            parts.append(f"Won by: {r['win_condition']}")
        if r.get("carry_over"):
            parts.append("Next: " + "; ".join(r["carry_over"]))
        if r.get("mistakes"):
            parts.append("Wasted: " + "; ".join(r["mistakes"]))
        return " ".join(parts)

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
        levels = fit_lines([clip(self.review_line(r), c.get("levels", 250) // 2) for r in self.reviews[-2:]],
                           c.get("levels", 250), older_note="")
        return {"goal": goal, "levels": "\n".join(x for x in levels if x), "plan": clip(self.plan, c["plan"]),
                "hypotheses": "\n".join(hyps),
                "findings": "\n".join(finds), "questions": "\n".join(x for x in qs if x)}
