"""Memory #8, the lessons graph, and the skills: both kept across levels and games, never shown whole.

Lessons graph. Nodes: ``shape`` (a colour + shape hash seen on screen), ``rule`` (a verified mechanic), ``refuted``
(a rule shown false, so it is not proposed again), ``goal`` (a goal that won a level). Edges: ``moves``, ``blocks``,
``wins-when``, ``appears-in-level``, ``contradicts``, ``about`` (rule -> shape). Node ids are content hashes, so the
same lesson written twice, by two games at once, is one node. The graph reaches the context as a subgraph: the nodes
linked to the shapes on screen now, then the most used goal patterns and rules, under a token cap.

Skills. Short knowledge notes (<= 200 tokens): three base notes shipped in ``harness/agent/skills/*.md`` and the notes
the curator writes from lessons. At most three are loaded into the context; the curator picks them.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from pathlib import Path
from typing import Any

from harness.memory.store import clip, lock_for, read_json, tokens, write_json

NODE_TYPES = ("shape", "rule", "refuted", "goal")
EDGE_TYPES = ("moves", "blocks", "wins-when", "appears-in-level", "contradicts", "about")
BASE_SKILLS_DIR = Path(__file__).resolve().parents[1] / "agent" / "skills"
SKILL_TOKENS = 200

_SHARED: dict[str, Any] = {}
_SHARED_LOCK = threading.Lock()


def _shared(cls: type, path: Path, *args: Any) -> Any:
    """One instance per file per process: the games of a run share it (they run in threads)."""
    key = f"{cls.__name__}:{Path(path).resolve()}"
    with _SHARED_LOCK:
        if key not in _SHARED:
            _SHARED[key] = cls(path, *args)
        return _SHARED[key]


def shape_id(colour_letter: str, shape_hash: str) -> str:
    return f"shape:{colour_letter}#{shape_hash}"


class LessonsGraph:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lock = lock_for(self.path)
        d = read_json(self.path, {})
        self.nodes: dict[str, dict[str, Any]] = d.get("nodes", {})
        self.edges: list[dict[str, Any]] = d.get("edges", [])

    @classmethod
    def shared(cls, path: Path) -> LessonsGraph:
        return _shared(cls, path)

    def save(self) -> None:
        with self.lock:
            write_json(self.path, {"nodes": self.nodes, "edges": self.edges})

    def add_node(self, kind: str, text: str, *, game: str | None = None, level: int | None = None,
                 source: str = "host", node_id: str | None = None) -> str:
        if kind not in NODE_TYPES:
            raise ValueError(f"node type must be one of {NODE_TYPES}")
        text = " ".join(str(text or "").split())
        if not text:
            raise ValueError("a lesson needs text")
        nid = node_id or f"{kind}:{hashlib.blake2b(text.lower().encode(), digest_size=4).hexdigest()}"
        with self.lock:
            n = self.nodes.setdefault(nid, {"id": nid, "type": kind, "text": clip(text, 80), "games": [],
                                            "levels": [], "uses": 0, "created": round(time.time()),
                                            "source": source})
            if game and game not in n["games"]:
                n["games"].append(game)
            if level is not None and level not in n["levels"]:
                n["levels"].append(level)
        return nid

    def add_edge(self, src: str, dst: str, kind: str) -> bool:
        if kind not in EDGE_TYPES or src not in self.nodes or dst not in self.nodes or src == dst:
            return False
        with self.lock:
            if any(e["src"] == src and e["dst"] == dst and e["type"] == kind for e in self.edges):
                return False
            self.edges.append({"src": src, "dst": dst, "type": kind})
        return True

    def add_shape(self, colour_letter: str, shape_hash: str, label: str, *, game: str | None = None,
                  level: int | None = None) -> str:
        return self.add_node("shape", f"{label} #{shape_hash}", game=game, level=level,
                             node_id=shape_id(colour_letter, shape_hash))

    def _line(self, n: dict[str, Any]) -> str:
        where = []
        if n["games"]:
            where.append(f"{len(n['games'])} game{'s' if len(n['games']) > 1 else ''}")
        rel = [f"{e['type']} {self.nodes[e['dst']]['text']}" for e in self.edges
               if e["src"] == n["id"] and e["dst"] in self.nodes][:3]
        return (f"- [{n['type']}] {n['text']}" + (f" -> {'; '.join(rel)}" if rel else "")
                + (f" ({', '.join(where)})" if where else ""))

    def subgraph(self, on_screen: list[str], cap_tokens: int, mark_used: bool = True) -> str:
        """Lessons about the shapes now on screen (``on_screen``: shape node ids), then goal patterns and rules by use,
        under ``cap_tokens``. Shape nodes are shown only through the lessons that point at them."""
        with self.lock:
            screen = set(on_screen)
            about = [self.nodes[e["src"]] for e in self.edges if e["dst"] in screen and e["src"] in self.nodes]
            rest = sorted((n for n in self.nodes.values() if n["type"] in ("goal", "rule", "refuted")),
                          key=lambda n: (-n["uses"], -len(n["games"]), -n["created"]))
            picked: list[dict[str, Any]] = []
            for n in about + rest:
                if n["type"] != "shape" and n not in picked:
                    picked.append(n)
            lines: list[str] = []
            for n in picked:
                line = self._line(n)
                if tokens("\n".join(lines + [line])) > cap_tokens:
                    break
                lines.append(line)
                if mark_used:
                    n["uses"] += 1
            return "\n".join(lines)

    def search(self, query: str, limit: int = 20) -> list[str]:
        words = [w for w in re.split(r"\s+", (query or "").lower()) if w]
        with self.lock:
            hits = [n for n in self.nodes.values() if n["type"] != "shape"
                    and all(w in (n["text"] + " " + n["type"]).lower() for w in words)]
            return [self._line(n) for n in hits[:limit]]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for n in self.nodes.values():
            out[n["type"]] = out.get(n["type"], 0) + 1
        out["edges"] = len(self.edges)
        return out


class SkillBook:
    """Base notes (read-only markdown files) plus curated notes (JSON, shared across games). ``loaded[game]`` is the
    curator's pick of at most three; before its first pick a game loads the three base notes."""

    def __init__(self, path: Path, base_dir: Path = BASE_SKILLS_DIR) -> None:
        self.path = Path(path)
        self.lock = lock_for(self.path)
        self.base: dict[str, str] = {p.stem: p.read_text(encoding="utf-8").strip()
                                     for p in sorted(Path(base_dir).glob("*.md"))}
        d = read_json(self.path, {})
        self.curated: dict[str, dict[str, Any]] = d.get("curated", {})
        self.loaded: dict[str, list[str]] = d.get("loaded", {})

    @classmethod
    def shared(cls, path: Path) -> SkillBook:
        return _shared(cls, path)

    def save(self) -> None:
        with self.lock:
            write_json(self.path, {"curated": self.curated, "loaded": self.loaded})

    def names(self) -> list[str]:
        return list(self.base) + [n for n in self.curated if n not in self.base]

    def text(self, name: str) -> str | None:
        if name in self.curated:
            return self.curated[name]["content"]
        return self.base.get(name)

    def upsert(self, name: str, content: str, *, game: str | None = None) -> str:
        name = re.sub(r"[^a-z0-9_]+", "_", str(name or "").lower()).strip("_")[:40]
        if not name or name in self.base:
            raise ValueError("a curated skill needs a new snake_case name (base skills are read-only)")
        content = clip(str(content or ""), SKILL_TOKENS)
        if not content:
            raise ValueError("a skill needs content")
        with self.lock:
            s = self.curated.setdefault(name, {"content": content, "version": 0, "games": []})
            s["content"], s["version"] = content, s["version"] + 1
            if game and game not in s["games"]:
                s["games"].append(game)
        return name

    def set_loaded(self, game: str, names: list[str]) -> list[str]:
        known = [n for n in dict.fromkeys(names) if self.text(n) is not None][:3]
        with self.lock:
            self.loaded[game] = known
        return known

    def loaded_for(self, game: str) -> list[str]:
        return self.loaded.get(game) or list(self.base)[:3]

    def render(self, game: str, cap_tokens: int) -> str:
        out: list[str] = []
        for name in self.loaded_for(game):
            block = f"### {name}\n{self.text(name)}"
            if tokens("\n\n".join(out + [block])) > cap_tokens:
                break
            out.append(block)
        return "\n\n".join(out)

    def search(self, query: str) -> list[str]:
        words = [w for w in re.split(r"\s+", (query or "").lower()) if w]
        out = []
        for name in self.names():
            text = self.text(name) or ""
            if all(w in f"{name} {text}".lower() for w in words):
                out.append(f"### {name}\n{text}")
        return out
