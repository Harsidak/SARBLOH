"""The current game state for the E008 REPL, read-only. Pre-imported as ``scene`` when the toolset is "e008".

The host writes ``$RLM_SESSION_DIR/.prime/scene.json`` after every act (and ``history.jsonl``, one line per step);
this module reads them when you touch it. There is no raw numeric grid: colours are letters (legend in
``scene.legend``), coordinates are (row, column).

    scene.objects                  # list of dicts: id, letter, name, size, bbox [r0, c0, r1, c1], hash, corners,
                                   #   parent, children, adjacent, hud, cells [(r, c), ...] (None above 256 cells)
    scene.letters                  # the board, one string of letters per row
    print(scene.ascii(r0, c0, r1, c1))   # a window with row and column labels
    scene.find(letter="R")         # objects by letter, hash, id or size
    scene.history(10)              # the last 10 steps: {"i", "level", "action", "change", "state"}
    scene.step, scene.status, scene.change, scene.background, scene.legend
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

LETTERS = "WwgGcBMPRbSYOrNp"
NAMES = ("white", "light grey", "grey", "dark grey", "charcoal", "black", "magenta", "pink", "red", "blue", "sky blue",
         "yellow", "orange", "dark red", "green", "purple")
_DIR = Path(os.environ.get("RLM_SESSION_DIR", ".")) / ".prime"
_cache: dict[str, Any] = {"mtime": None, "data": {}}


def _data() -> dict[str, Any]:
    path = _DIR / "scene.json"
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return {}
    if mtime != _cache["mtime"]:
        _cache["data"] = json.loads(path.read_text(encoding="utf-8"))
        _cache["mtime"] = mtime
    return _cache["data"]


def ascii(r0: int = 0, c0: int = 0, r1: int | None = None, c1: int | None = None) -> str:
    """The window rows r0..r1, columns c0..c1 (inclusive) as letters, with labels."""
    rows = _data().get("letters") or []
    if not rows:
        return "(no scene yet)"
    r1 = len(rows) - 1 if r1 is None else min(r1, len(rows) - 1)
    c1 = len(rows[0]) - 1 if c1 is None else min(c1, len(rows[0]) - 1)
    r0, c0 = max(0, r0), max(0, c0)
    cols = range(c0, c1 + 1)
    out = ["    " + "".join(str(c // 10 % 10) for c in cols), "    " + "".join(str(c % 10) for c in cols)]
    out += [f"{r:>3} " + rows[r][c0:c1 + 1] for r in range(r0, r1 + 1)]
    return "\n".join(out)


def find(letter: str | None = None, hash: str | None = None, id: int | None = None,
         size: int | None = None) -> list[dict[str, Any]]:
    out = []
    for o in _data().get("objects") or []:
        if letter is not None and o["letter"] != letter:
            continue
        if hash is not None and o["hash"] != hash.lstrip("#"):
            continue
        if id is not None and o["id"] != id:
            continue
        if size is not None and o["size"] != size:
            continue
        out.append(o)
    return out


def history(n: int = 10) -> list[dict[str, Any]]:
    path = _DIR / "history.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [json.loads(x) for x in lines[-int(n):]] if n else []


def __getattr__(name: str) -> Any:
    d = _data()
    if name == "objects":
        return d.get("objects") or []
    if name == "letters":
        return d.get("letters") or []
    if name in ("step", "status", "change", "background"):
        return d.get(name)
    if name == "legend":
        used = sorted({ch for row in d.get("letters") or [] for ch in row})
        return " ".join(f"{ch}={NAMES[LETTERS.index(ch)]}({LETTERS.index(ch)})" for ch in used if ch in LETTERS)
    raise AttributeError(f"scene has no {name!r}; use scene.objects, scene.letters, scene.ascii(...), scene.find(...), "
                         "scene.history(n), scene.step, scene.status, scene.change, scene.legend")


def write(session_dir: Path, data: dict[str, Any], step_row: dict[str, Any] | None = None) -> None:
    """Host side: publish the scene (and append one history row) for the REPL."""
    d = Path(session_dir) / ".prime"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "scene.json.tmp"
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, d / "scene.json")
    if step_row is not None:
        with (d / "history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(step_row) + "\n")
