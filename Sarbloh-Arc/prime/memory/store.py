"""Files behind the memories: JSON written atomically, JSONL appended, one lock per path (games run in threads of one
process and share the lessons graph and the skills)."""

from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Any

_LOCKS: dict[str, threading.RLock] = {}
_GUARD = threading.Lock()


def lock_for(path: Path) -> threading.RLock:
    key = str(Path(path).resolve())
    with _GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_for(path), path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return rows


def tokens(text: str) -> int:
    """The chars/4 estimate every cap uses."""
    return math.ceil(len(text or "") / 4)


def fit_lines(lines: list[str], cap_tokens: int, older_note: str = "(+{n} older: recall)") -> list[str]:
    """Keep the newest lines (the end of the list) that fit ``cap_tokens``; a note says how many were dropped."""
    kept: list[str] = []
    budget = cap_tokens * 4
    for line in reversed(lines):
        if sum(len(x) + 1 for x in kept) + len(line) + 1 > budget - len(older_note) - 4:
            break
        kept.append(line)
    kept.reverse()
    dropped = len(lines) - len(kept)
    return ([older_note.format(n=dropped)] if dropped else []) + kept


def clip(text: str, cap_tokens: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= cap_tokens * 4 else text[:cap_tokens * 4 - 4].rstrip() + " ..."
