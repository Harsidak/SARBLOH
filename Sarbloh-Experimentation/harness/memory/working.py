"""The agent's working memory: #2 plan, #3 hypotheses, #5 open questions, #6 findings, #7 goal.

The agent writes plan, hypotheses, findings and goal through ``act()`` in the REPL; the curator writes the open questions.
The host checks the shape of what is written and judges nothing: a hypothesis is "verified" because the agent says so.
Every block is rendered under its token cap (chars / 4); a block over its cap loses its oldest lines, which stay in
the file and in ``recall``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from harness.memory.store import clip, fit_lines, read_json, write_json

STATUSES = ("proposed", "verified", "refuted")
_HYP_STR = re.compile(r"^\s*(?:(h\d+)\b[\s:.-]*)?(?:\[?(proposed|verified|refuted)\]?\b[\s:.-]*)?(.*)$", re.IGNORECASE | re.DOTALL)


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


_SECTION = re.compile(r"^\s*[#*_]*\s*(plan|hypothes[ie]s|findings?|goal)\s*[*_]*\s*:\s*[*_]*\s*(.*)$", re.IGNORECASE)
_FENCE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_NEW = re.compile(r"^\s*new\s*[:.-]\s*", re.IGNORECASE)
_EVIDENCE = re.compile(r"\s*[(\[]\s*(#\s*\d+(?:\s*,\s*#?\s*\d+)*)\s*[)\]]\s*$")


def parse_memory_text(text: str) -> dict[str, Any]:
    """The memory sections the agent writes in its reply before the code: "Plan:", "Hypotheses:", "Goal:" and
    "Findings:", each on its own line, with one list item per line under Hypotheses and Findings. A blank line or a
    code block ends a section; text outside the sections is ignored. Returns only the sections that were written:
    {"plan": str, "goal": str, "hypotheses": [str], "findings": [str]}."""
    found: dict[str, list[str]] = {}
    key = None
    for line in _FENCE.sub("\n\n", text or "").splitlines():
        m = _SECTION.match(line)
        if m:
            key = m.group(1).lower()
            key = "hypotheses" if key.startswith("hypothes") else "findings" if key.startswith("finding") else key
            found.setdefault(key, [])
            line = m.group(2)
        elif key is None:
            continue
        elif not line.strip():
            key = None
            continue
        item = _BULLET.sub("", line).strip()
        if item and item.lower().strip(".") not in ("none", "(none)", "no change", "unchanged", "same"):
            found[key].append(item)
    out: dict[str, Any] = {}
    for k, items in found.items():
        if k in ("plan", "goal"):
            if items:
                out[k] = " ".join(items)
        elif items:
            out[k] = items
    return out


def hypothesis_from_line(line: str) -> Any:
    """A hypothesis line from the reply as an item for ``update_hypotheses``: "new: text", "h2 refuted: text" or
    "h2 verified", with optional step numbers at the end, as "(#12, #14)"."""
    line = _NEW.sub("", line)
    m = _EVIDENCE.search(line)
    if not m:
        return line
    ev = _evidence(m.group(1))
    item = parse_hypothesis(line[:m.start()])
    return {**item, "evidence": ev}


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


class WorkingMemory:
    def __init__(self, path: Path, caps: dict[str, int]) -> None:
        self.path = Path(path)
        self.caps = caps
        d = read_json(self.path, {})
        self.plan: str = d.get("plan", "")
        self.hypotheses: list[dict[str, Any]] = d.get("hypotheses", [])
        self.findings: list[dict[str, Any]] = d.get("findings", [])
        self.goal: str = d.get("goal", "")
        self.goal_log: list[str] = d.get("goal_log", [])
        self.questions: list[str] = d.get("questions", [])
        self.next_h: int = d.get("next_h", 1)

    def save(self) -> None:
        write_json(self.path, {"plan": self.plan, "hypotheses": self.hypotheses, "findings": self.findings,
                               "goal": self.goal, "goal_log": self.goal_log, "questions": self.questions,
                               "next_h": self.next_h})

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

    def set_questions(self, questions: list[str]) -> None:
        self.questions = [str(q).strip() for q in questions if str(q).strip()][:3]

    def confirm_goal(self, level: int) -> str:
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

    # --- reads ---------------------------------------------------------------------------------------------
    def hyp_line(self, h: dict[str, Any]) -> str:
        ev = f" (evidence {', '.join('#' + str(e) for e in h['evidence'][:8])})" if h.get("evidence") else ""
        return f"{h['id']} [{h['status']}] {h['text']}{ev}"

    def blocks(self) -> dict[str, str]:
        """Rendered blocks under their caps; empty blocks are ""."""
        c = self.caps
        goal = ""
        if self.goal or self.goal_log:
            goal = clip(self.goal or "(not stated yet)", c["goal"] - 20)
            if self.goal_log:
                goal += "\n" + self.goal_log[-1]
        hyps = fit_lines([self.hyp_line(h) for h in self.hypotheses], c["hypotheses"])
        finds = fit_lines([f"- {f['text']}" + (f" (evidence {', '.join('#' + str(e) for e in f['evidence'][:6])})"
                                               if f.get("evidence") else "") for f in self.findings], c["findings"])
        qs = fit_lines([f"- {q}" for q in self.questions[:3]], c["questions"], older_note="")
        return {"goal": goal, "plan": clip(self.plan, c["plan"]), "hypotheses": "\n".join(hyps),
                "findings": "\n".join(finds), "questions": "\n".join(x for x in qs if x)}
