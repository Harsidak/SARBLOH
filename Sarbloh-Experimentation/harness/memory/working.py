"""The agent's working memory: #2 plan, #3 hypotheses, #5 open questions, #6 findings, #7 goal.

The agent writes plan, hypotheses, findings and goal as four variables in its kernel (``KERNEL_VARS``); the host
copies them here after every ipython call and before every move (``sync``). The curator writes the open questions.
The host checks the shape of what is written and judges nothing. The agent only writes hypotheses, never their
status: every one stays "proposed", and the curator decides at a level up which ones the level proved or refuted.
Every block is rendered under its token cap (chars / 4); a block over its cap loses its oldest lines, which stay in
the file and in ``recall``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from harness.memory.store import clip, fit_lines, read_json, write_json

STATUSES = ("proposed", "verified", "refuted")
KERNEL_VARS = ("GOAL", "PLAN", "HYPOTHESES", "FINDINGS")   # the memory variables in the agent's kernel
# A hypothesis status as the agent writes it in HYPOTHESES -> as stored ("open" is shown for "proposed").
_KERNEL_STATUS = {"open": "proposed", "proposed": "proposed", "verified": "verified", "refuted": "refuted"}
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

    def sync(self, values: dict[str, Any], *, turn: int, level: int) -> dict[str, Any]:
        """Copies the kernel's memory variables (``KERNEL_VARS``; a missing one is left alone): GOAL and PLAN are text,
        HYPOTHESES a list of text (a dict counts by its keys, its values ignored), FINDINGS a list of
        text. Memory mirrors them: what the agent deleted there is deleted here. A value of the wrong type is not
        copied and is named in ``errors``. Returns what changed: ``plan`` and ``goal`` (the new text, or None),
        ``hypotheses`` (one event per new or changed one), ``findings`` (added), ``removed`` (texts), ``errors``."""
        out: dict[str, Any] = {"plan": None, "goal": None, "hypotheses": [], "findings": [], "removed": [],
                               "errors": []}
        for name, attr in (("PLAN", "plan"), ("GOAL", "goal")):
            if name not in values:
                continue
            v = "" if values[name] is None else values[name]
            if not isinstance(v, str):
                out["errors"].append(f"{name} must be text, not {type(v).__name__}")
            elif v.strip() != getattr(self, attr):
                setattr(self, attr, v.strip())
                out[attr] = v.strip()
        if "HYPOTHESES" in values:
            v = values["HYPOTHESES"]
            if isinstance(v, str):
                v = [v]
            if isinstance(v, dict):   # an old habit: the keys are the hypotheses, a status written as a value is ignored
                v = list(v)
            if not isinstance(v, list):
                out["errors"].append(f"HYPOTHESES must be a list of text, not {type(v).__name__}")
            else:
                self._sync_hypotheses(v, out, turn=turn, level=level)
        if "FINDINGS" in values:
            v = values["FINDINGS"]
            if isinstance(v, str):
                v = [v]
            if not isinstance(v, list):
                out["errors"].append(f"FINDINGS must be a list of text, not {type(v).__name__}")
            else:
                old = {f["text"].lower(): f for f in self.findings}
                kept: list[dict[str, Any]] = []
                for x in v:
                    text = str(x).strip()
                    if not text or any(f["text"].lower() == text.lower() for f in kept):
                        continue
                    f = old.get(text.lower())
                    if f is None:
                        f = {"text": text, "evidence": [], "level": level, "turn": turn}
                        out["findings"].append(f)
                    kept.append(f)
                out["removed"] += [f["text"] for f in self.findings if f not in kept]
                self.findings = kept
        return out

    def _sync_hypotheses(self, rules: list[Any], out: dict[str, Any], *, turn: int, level: int) -> None:
        old = {h["text"].lower(): h for h in self.hypotheses}
        kept: list[dict[str, Any]] = []
        for text in rules:
            text = str(text).strip()
            if not text or any(h["text"].lower() == text.lower() for h in kept):
                continue
            h = old.get(text.lower())
            if h is None:
                h = {"id": f"h{self.next_h}", "text": text, "status": "proposed", "evidence": [], "level": level,
                     "turn": turn, "history": [[turn, "proposed"]]}
                self.next_h += 1
                out["hypotheses"].append({**{k: h[k] for k in ("id", "text", "status", "evidence")},
                                          "prev_status": None})
            kept.append(h)
        out["removed"] += [h["text"] for h in self.hypotheses if h not in kept]
        self.hypotheses = kept

    def kernel_values(self) -> dict[str, Any]:
        """The memory as the kernel variables hold it (to put them back after a kernel restart)."""
        return {"GOAL": self.goal, "PLAN": self.plan,
                "HYPOTHESES": [h["text"] for h in self.hypotheses],
                "FINDINGS": [f["text"] for f in self.findings]}

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
        return f"{h['id']} {h['text']}{ev}"

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
