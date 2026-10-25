"""The agent's working memory: #2 plan, #3 hypotheses, #6 findings, #7 goal, and the skills it has loaded.

They are five variables in the agent's kernel (``KERNEL_VARS``). GOAL, PLAN, HYPOTHESES and FINDINGS the agent writes,
each a dict keyed by level ("L1", "L2", ...), so what it wrote on an earlier level stays and can be used again:

    GOAL = {"L4": {"light all three windows": "kept", "line the plunger up with the plug": "refuted"}}
    PLAN = {"L4": "click id21 twice, then check window B"}
    HYPOTHESES = {"L4": {"A6 on obj 18 grows R1 by 3 rows": "kept"}}
    FINDINGS = {"L4": ["the pink bar loses one cell per move"]}

A goal or a hypothesis is "kept" or "refuted"; a level-up marks the kept goals of the level just won "won" (only a win
does). SKILLS, a dict of skill name -> text, is filled by ``recall()`` (at most five, the oldest dropped first) and by
the curator at a level-up. The host copies them here after every ipython call and before every move (``sync``); an
edit the agent makes to SKILLS stays in this game's memory and never reaches the shared skill book. The host checks
the shape of what is written, and that a goal does not say the level cannot be won (every level can be); it judges
nothing else. Nothing is erased at a level-up, only at a new game. Every block is rendered under its token cap
(chars / 4); a block over its cap loses its oldest lines, which stay in the file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from harness.memory.store import fit_lines, read_json, write_json

KERNEL_VARS = ("GOAL", "PLAN", "HYPOTHESES", "FINDINGS", "SKILLS")   # the memory variables in the agent's kernel
GOAL_STATUSES = ("kept", "refuted", "won")   # "won" only from a level win
HYP_STATUSES = ("kept", "refuted")
SKILL_TEXT_CHARS = 1200   # a SKILLS entry is kept up to this many characters
# A goal says what wins the level; one that says the level cannot be won is not saved.
NO_WIN = re.compile(r"\b(unsolvable|unwinnable|not (?:be )?(?:solvable|winnable)|(?:impossible|no way) to "
                    r"(?:win|solve|clear|beat|complete)|(?:cannot|can't|can never) be (?:won|solved|cleared|beaten))\b",
                    re.IGNORECASE)
_KEY = re.compile(r"^\s*L?\s*(\d+)\s*$", re.IGNORECASE)
_EXAMPLE = {"GOAL": 'GOAL["L2"] = {"push the block onto the flag": "kept"}',
            "PLAN": 'PLAN["L2"] = "push the block right"',
            "HYPOTHESES": 'HYPOTHESES["L2"] = {"1 moves obj 4 up by 5 cells": "kept"}',
            "FINDINGS": 'FINDINGS["L2"] = ["the bar loses one block per move"]'}


def level_key(level: int) -> str:
    """The key of level ``level`` (0-based, as levels_completed counts) in the memory dicts: "L1" for the first."""
    return f"L{int(level) + 1}"


def _level(key: Any) -> int | None:
    """0-based level of a key such as "L3", "3" or 3; None when it is not one."""
    m = _KEY.match(str(key))
    return int(m.group(1)) - 1 if m and int(m.group(1)) >= 1 else None


def _texts(v: Any) -> list[str]:
    """A list of text from a list, a single text or None."""
    if v is None:
        return []
    items = v if isinstance(v, (list, tuple)) else [v]
    return [t for t in (str(x).strip() for x in items) if t]


class WorkingMemory:
    def __init__(self, path: Path, caps: dict[str, int]) -> None:
        self.path = Path(path)
        self.caps = caps
        d = read_json(self.path, {})
        self.goals: list[dict[str, Any]] = d.get("goals", [])          # {"level", "text", "status", "turn"}
        self.plans: dict[str, str] = d.get("plans", {})                 # level key -> text
        self.hypotheses: list[dict[str, Any]] = d.get("hypotheses", [])  # {"id", "text", "status", "level", "turn", "history"}
        self.findings: list[dict[str, Any]] = d.get("findings", [])      # {"text", "evidence", "level", "turn"}
        self.goal_log: list[str] = d.get("goal_log", [])
        self.skills: dict[str, str] = d.get("skills", {})
        self.next_h: int = d.get("next_h", 1)

    def save(self) -> None:
        write_json(self.path, {"goals": self.goals, "plans": self.plans, "hypotheses": self.hypotheses,
                               "findings": self.findings, "goal_log": self.goal_log, "skills": self.skills,
                               "next_h": self.next_h})

    # --- writes --------------------------------------------------------------------------------------------
    def sync(self, values: dict[str, Any], *, turn: int, level: int) -> dict[str, Any]:
        """Copies the kernel's memory variables (``KERNEL_VARS``; a missing one is left alone). Memory mirrors them: what
        the agent deleted there is deleted here. A variable with a wrong shape is not copied at all: it is named in
        ``rejected`` and the reason is in ``errors``. Returns what changed: ``plan`` (the level keys whose plan changed),
        ``goal`` (one event per new or changed goal: level, goal, status, prev_status), ``hypotheses`` (one event per
        new or changed one), ``findings`` (added), ``removed`` (texts), ``skills`` (the names, when a skill was loaded,
        dropped or changed, else None), ``errors`` and ``rejected``."""
        out: dict[str, Any] = {"plan": [], "goal": [], "hypotheses": [], "findings": [], "removed": [],
                               "skills": None, "errors": [], "rejected": []}
        for name, parse, apply in (("GOAL", self._parse_goals, self._apply_goals),
                                   ("PLAN", self._parse_plans, self._apply_plans),
                                   ("HYPOTHESES", self._parse_hypotheses, self._apply_hypotheses),
                                   ("FINDINGS", self._parse_findings, self._apply_findings)):
            if name not in values:
                continue
            v = {} if values[name] is None else values[name]
            if not isinstance(v, dict):
                out["errors"].append(f"{name} must be a dict of level -> notes, not {type(v).__name__}, "
                                     f"for example {_EXAMPLE[name]}")
                out["rejected"].append(name)
                continue
            errors: list[str] = []
            parsed: dict[int, Any] = {}
            for key, item in v.items():
                lv = _level(key)
                if lv is None:
                    errors.append(f"{name} key {key!r} is not a level: use \"L1\", \"L2\", ...")
                    continue
                parsed[lv] = parse(item, lv, errors)
            if errors:
                out["errors"] += errors
                out["rejected"].append(name)
                continue
            apply(parsed, out, turn=turn)
        if "SKILLS" in values:
            v = values["SKILLS"]
            if not isinstance(v, dict):
                out["errors"].append(f"SKILLS must be a dict of skill name -> text, not {type(v).__name__}")
                out["rejected"].append("SKILLS")
            else:
                new = {str(k): str(t)[:SKILL_TEXT_CHARS] for k, t in v.items()}
                if new != self.skills:
                    self.skills = new
                    out["skills"] = list(new)
        return out

    # GOAL ---------------------------------------------------------------------------------------------------
    def _parse_goals(self, item: Any, lv: int, errors: list[str]) -> dict[str, str]:
        """{goal text: status} of one level; a text or a list of texts means kept goals."""
        if not isinstance(item, dict):
            item = {t: "kept" for t in _texts(item)}
        won = {g["text"].lower() for g in self.goals if g["level"] == lv and g["status"] == "won"}
        goals: dict[str, str] = {}
        for text, status in item.items():
            text, status = str(text).strip(), str(status or "kept").strip().lower()
            if not text:
                continue
            if status not in GOAL_STATUSES:
                errors.append(f"GOAL {level_key(lv)} {text!r}: the status must be \"kept\" or \"refuted\", not {status!r}")
            elif status == "won" and text.lower() not in won:
                errors.append(f"GOAL {level_key(lv)} {text!r}: only winning the level marks a goal \"won\"")
            elif NO_WIN.search(text):
                errors.append(f"GOAL {level_key(lv)} {text!r}: a goal says what wins the level, and every level can be "
                              f"won. Mark a goal you ruled out \"refuted\" and write the next goal to test")
            else:
                goals[text] = status
        return goals

    def _apply_goals(self, parsed: dict[int, dict[str, str]], out: dict[str, Any], *, turn: int) -> None:
        old = {(g["level"], g["text"].lower()): g for g in self.goals}
        kept: list[dict[str, Any]] = []
        for lv in sorted(parsed):
            for text, status in parsed[lv].items():
                g = old.get((lv, text.lower()))
                if g is None:
                    g = {"level": lv, "text": text, "status": status, "turn": turn}
                    out["goal"].append({"level": lv, "goal": text, "status": status, "prev_status": None})
                elif g["status"] != status:
                    out["goal"].append({"level": lv, "goal": text, "status": status, "prev_status": g["status"]})
                    g = dict(g, status=status)
                kept.append(g)
        keys = {(g["level"], g["text"].lower()) for g in kept}
        out["removed"] += [g["text"] for g in self.goals if (g["level"], g["text"].lower()) not in keys]
        self.goals = kept

    # PLAN ---------------------------------------------------------------------------------------------------
    def _parse_plans(self, item: Any, lv: int, errors: list[str]) -> str:
        if item is not None and not isinstance(item, str):
            errors.append(f"PLAN {level_key(lv)} must be text, not {type(item).__name__}")
            return ""
        return (item or "").strip()

    def _apply_plans(self, parsed: dict[int, str], out: dict[str, Any], *, turn: int) -> None:
        new = {level_key(lv): parsed[lv] for lv in sorted(parsed) if parsed[lv]}
        out["plan"] += [k for k in new if self.plans.get(k) != new[k]]
        out["removed"] += [t for k, t in self.plans.items() if k not in new]
        self.plans = new

    # HYPOTHESES ---------------------------------------------------------------------------------------------
    def _parse_hypotheses(self, item: Any, lv: int, errors: list[str]) -> dict[str, str]:
        """{rule: status} of one level; a text or a list of texts means kept ones."""
        if not isinstance(item, dict):
            item = {t: "kept" for t in _texts(item)}
        hyps: dict[str, str] = {}
        for text, status in item.items():
            text, status = str(text).strip(), str(status or "kept").strip().lower()
            if not text:
                continue
            if status not in HYP_STATUSES:
                errors.append(f"HYPOTHESES {level_key(lv)} {text!r}: the status must be \"kept\" or \"refuted\", "
                              f"not {status!r}")
            else:
                hyps[text] = status
        return hyps

    def _apply_hypotheses(self, parsed: dict[int, dict[str, str]], out: dict[str, Any], *, turn: int) -> None:
        old = {(h["level"], h["text"].lower()): h for h in self.hypotheses}
        kept: list[dict[str, Any]] = []
        for lv in sorted(parsed):
            for text, status in parsed[lv].items():
                h = old.get((lv, text.lower()))
                if h is None:
                    h = {"id": f"h{self.next_h}", "text": text, "status": status, "evidence": [], "level": lv,
                         "turn": turn, "history": [[turn, status]]}
                    self.next_h += 1
                    out["hypotheses"].append({**{k: h[k] for k in ("id", "text", "status", "evidence")},
                                              "prev_status": None})
                elif h["status"] != status:
                    out["hypotheses"].append({"id": h["id"], "text": h["text"], "status": status,
                                              "evidence": h["evidence"], "prev_status": h["status"]})
                    h = dict(h, status=status, history=[*h["history"], [turn, status]])
                kept.append(h)
        keys = {(h["level"], h["text"].lower()) for h in kept}
        out["removed"] += [h["text"] for h in self.hypotheses if (h["level"], h["text"].lower()) not in keys]
        self.hypotheses = kept

    # FINDINGS -----------------------------------------------------------------------------------------------
    def _parse_findings(self, item: Any, lv: int, errors: list[str]) -> list[str]:
        if isinstance(item, dict):
            errors.append(f"FINDINGS {level_key(lv)} must be a list of text, not a dict")
            return []
        return _texts(item)

    def _apply_findings(self, parsed: dict[int, list[str]], out: dict[str, Any], *, turn: int) -> None:
        old = {(f["level"], f["text"].lower()): f for f in self.findings}
        kept: list[dict[str, Any]] = []
        for lv in sorted(parsed):
            for text in parsed[lv]:
                if any(f["level"] == lv and f["text"].lower() == text.lower() for f in kept):
                    continue
                f = old.get((lv, text.lower()))
                if f is None:
                    f = {"text": text, "evidence": [], "level": lv, "turn": turn}
                    out["findings"].append(f)
                kept.append(f)
        keys = {(f["level"], f["text"].lower()) for f in kept}
        out["removed"] += [f["text"] for f in self.findings if (f["level"], f["text"].lower()) not in keys]
        self.findings = kept

    # --- level-up and new game -------------------------------------------------------------------------------
    def win_level(self, level: int) -> tuple[str, list[str]]:
        """Level ``level`` (0-based) was won: its kept goals become "won". Returns the confirmation line and the goals
        it marked. Nothing is erased."""
        won = []
        for i, g in enumerate(self.goals):
            if g["level"] == level and g["status"] == "kept":
                self.goals[i] = dict(g, status="won")
                won.append(g["text"])
        line = f"confirmed: won level {level + 1}" + (f" with goal: {'; '.join(won)}" if won else "")
        self.goal_log.append(line)
        return line, won

    def clear_game(self) -> None:
        self.goals, self.plans, self.hypotheses, self.findings = [], {}, [], []
        self.goal_log, self.next_h, self.skills = [], 1, {}

    # --- reads -----------------------------------------------------------------------------------------------
    def plan_for(self, level: int) -> str:
        return self.plans.get(level_key(level), "")

    def kernel_values(self) -> dict[str, Any]:
        """The memory as the kernel variables hold it (to put them back after a kernel restart, a refused write or a
        level-up)."""
        goal: dict[str, dict[str, str]] = {}
        for g in sorted(self.goals, key=lambda g: g["level"]):
            goal.setdefault(level_key(g["level"]), {})[g["text"]] = g["status"]
        hyps: dict[str, dict[str, str]] = {}
        for h in sorted(self.hypotheses, key=lambda h: h["level"]):
            hyps.setdefault(level_key(h["level"]), {})[h["text"]] = h["status"]
        finds: dict[str, list[str]] = {}
        for f in sorted(self.findings, key=lambda f: f["level"]):
            finds.setdefault(level_key(f["level"]), []).append(f["text"])
        return {"GOAL": goal, "PLAN": dict(self.plans), "HYPOTHESES": hyps, "FINDINGS": finds,
                "SKILLS": dict(self.skills)}

    def blocks(self) -> dict[str, str]:
        """Rendered blocks under their caps, oldest level first; empty blocks are ""."""
        c = self.caps
        order = lambda items: sorted(items, key=lambda x: x["level"])
        goals = fit_lines([f"{level_key(g['level'])} [{g['status']}] {g['text']}" for g in order(self.goals)], c["goal"])
        plans = fit_lines([f"{k}: {t}" for k, t in sorted(self.plans.items(), key=lambda kv: _level(kv[0]) or 0)],
                          c["plan"])
        hyps = fit_lines([f"{level_key(h['level'])} [{h['status']}] {h['id']} {h['text']}"
                          for h in order(self.hypotheses)], c["hypotheses"])
        finds = fit_lines([f"{level_key(f['level'])} - {f['text']}" for f in order(self.findings)], c["findings"])
        skills = fit_lines([f"- {n}: {first_sentence(t)}" for n, t in self.skills.items()], c["skills"])
        return {"goal": "\n".join(goals), "plan": "\n".join(plans), "hypotheses": "\n".join(hyps),
                "findings": "\n".join(finds), "skills": "\n".join(skills)}


def first_sentence(text: str, chars: int = 120) -> str:
    """A skill's purpose: its first sentence (without a leading "What it is:"), cut to ``chars``."""
    text = re.sub(r"^What it is:\s*", "", " ".join(str(text or "").split()))
    first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    return first if len(first) <= chars else first[:chars - 3] + "..."
