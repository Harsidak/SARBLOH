"""E018 (switch ``memory.wrong_rulebook``): the wrong rulebook, rules the agent showed false in one game, kept across
its levels in ``<root>/<game>/wrong_rules.json``. Each game has its own: a rule wrong in one game can hold in another.

Rules arrive from the agent's own act (a hypothesis marked ``refuted``, written at once), from the curator and from the
E022 level review (their ``refuted`` rules); with the switch on they no longer become ``refuted`` nodes of the lessons
graph. The agent never sees the rulebook: it is a store. A new or changed hypothesis that matches a wrong rule is
written as usual and only counted (a re-proposal, the waste event of CLAUDE.md). E039 reads it after every level won
to correct the lessons graph (``harness.agent.lessons_reconcile``).
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any

from harness.memory.lessons import _shared
from harness.memory.store import clip, lock_for, read_json, tokens, write_json

TEXT_TOKENS = 80          # same cap as a lessons-graph node (about 320 characters)
MATCH_JACCARD = 0.75      # word-set overlap at which a hypothesis counts as a re-proposal (UNCONFIRMED threshold)
_WORD = re.compile(r"[a-z0-9#]+")


def words(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(str(text or "").lower()))


def similar(a: str, b: str) -> bool:
    wa, wb = words(a), words(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= MATCH_JACCARD


class WrongRulebook:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lock = lock_for(self.path)
        self.rules: dict[str, dict[str, Any]] = read_json(self.path, {}).get("rules", {})

    @classmethod
    def shared(cls, path: Path) -> WrongRulebook:
        return _shared(cls, path)

    def save(self) -> None:
        with self.lock:
            write_json(self.path, {"rules": self.rules})

    def add(self, text: str, *, level: int | None = None, source: str = "agent") -> str:
        """One wrong rule; the same text again (any case, any spacing) adds the level and source to the same entry."""
        text = " ".join(str(text or "").split())
        if not text:
            raise ValueError("a wrong rule needs text")
        wid = f"wrong:{hashlib.blake2b(text.lower().encode(), digest_size=4).hexdigest()}"
        with self.lock:
            r = self.rules.setdefault(wid, {"id": wid, "text": clip(text, TEXT_TOKENS), "levels": [], "sources": [],
                                            "reproposed": 0, "created": round(time.time())})
            for key, value in (("levels", level), ("sources", source)):
                if value is not None and value not in r[key]:
                    r[key].append(value)
        return wid

    def match(self, text: str) -> dict[str, Any] | None:
        """The wrong rule that ``text`` restates, if any: the most re-proposed of the matches."""
        with self.lock:
            hits = [r for r in self.rules.values() if similar(text, r["text"])]
        return max(hits, key=lambda r: (r["reproposed"], r["created"])) if hits else None

    def mark_reproposed(self, wid: str) -> None:
        with self.lock:
            if wid in self.rules:
                self.rules[wid]["reproposed"] += 1

    @staticmethod
    def line(r: dict[str, Any]) -> str:
        lv = sorted({int(x) + 1 for x in r["levels"]})
        return f"- {r['text']}" + (f" (level {', '.join(map(str, lv))})" if lv else "")

    def render(self, cap_tokens: int) -> str:
        """The rules, newest first, under ``cap_tokens`` (for the E039 reconcile, never for the agent)."""
        with self.lock:
            rules = sorted(self.rules.values(), key=lambda r: -r["created"])
        lines: list[str] = []
        for r in rules:
            line = self.line(r)
            if tokens("\n".join(lines + [line])) > cap_tokens:
                lines.append(f"(+{len(rules) - len(lines)} more)")
                break
            lines.append(line)
        return "\n".join(lines)

    def counts(self) -> dict[str, int]:
        with self.lock:
            return {"rules": len(self.rules), "reproposed": sum(r["reproposed"] for r in self.rules.values())}
