"""Step map of a Prime run: for every game, what the agent thought, ran and spent, turn by turn.

    uv run python Sarbloh-Prime/prime/trace.py <run_dir>     # e.g. runs/kaggle_.../prime_run; run.py calls build_run

Sources in <run_dir>: ``results.json`` and, per game, ``games/<game_id>/transcript.jsonl`` (every message of the
root session with the model's reasoning, and every cell and host event), plus the ARC SDK recording
``recordings/<scorecard>/<game_id>-<guid>.jsonl``: one line per environment step with state, levels, frames and the
agent's ref (session, turn, call, k) in ``action_input.reasoning``. Without a recording the actions come from
``run.history``. Writes ``steps.jsonl`` (one row per action) and ``steps.md`` (the turn-by-turn map) next to each
transcript, and ``trace.md`` (one row per game) in <run_dir>.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def _cut(text: Any, n: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= n else f"{text[:n // 2]}\n[... {len(text) - n} chars cut ...]\n{text[-n // 2:]}"


def action_rows(recording: Path | None, run: dict[str, Any] | None) -> list[dict[str, Any]]:
    """One row per agent action: the SDK recording if there is one, else ``run.history``."""
    rows: list[dict[str, Any]] = []
    if recording is not None and recording.exists():
        for n, ev in enumerate(_jsonl(recording)):
            d = ev.get("data") or {}
            ai = d.get("action_input") or {}
            ref = ai.get("reasoning") if isinstance(ai.get("reasoning"), dict) else {}
            if n == 0 and not ref:
                continue  # the SDK records its own reset at Arcade.make(); not an agent action
            xy = ai.get("data") or {}
            rows.append({"i": len(rows), "action": ai.get("id"), "x": xy.get("x"), "y": xy.get("y"),
                         "levels": d.get("levels_completed"), "state": d.get("state"),
                         "full_reset": bool(d.get("full_reset")), "turn": ref.get("turn"), "call": ref.get("call"),
                         "k": ref.get("k"), "after_cell": bool(ref.get("after_cell")), "ts": ev.get("timestamp")})
        return rows
    hist = (run or {}).get("history") or []
    for i, h in enumerate(hist):
        ref = h.get("ref") or {}
        after = hist[i + 1]["lvl"] if i + 1 < len(hist) else (run or {}).get("levels_completed")
        rows.append({"i": i, "action": h.get("a"), "x": (h.get("d") or {}).get("x"), "y": (h.get("d") or {}).get("y"),
                     "levels": after, "state": None, "full_reset": False, "turn": ref.get("turn"),
                     "call": ref.get("call"), "k": ref.get("k"), "after_cell": bool(ref.get("after_cell"))})
    return rows


def _name(r: dict[str, Any]) -> str:
    return str(r["action"]).replace("ACTION", "A") + (f"({r['x']},{r['y']})" if r.get("x") is not None else "")


def render_actions(rows: list[dict[str, Any]], levels_before: int | None) -> str:
    """Runs of the same action, broken at every event: ``#12-#14 A1x3, #15 A6(3,4) -> LEVEL UP 1``."""
    parts: list[str] = []
    run_start, prev = None, levels_before
    for j, r in enumerate(rows):
        marks = []
        if prev is not None and r.get("levels") is not None and r["levels"] > prev:
            marks.append(f"LEVEL UP {r['levels']}")
        if r.get("state") in ("GAME_OVER", "WIN"):
            marks.append(r["state"])
        if r.get("full_reset"):
            marks.append("FULL RESET (score wiped)")
        prev = r["levels"] if r.get("levels") is not None else prev
        if run_start is None:
            run_start = j
        nxt = rows[j + 1] if j + 1 < len(rows) else None
        if marks or nxt is None or _name(nxt) != _name(r):
            a, n = rows[run_start]["i"], j - run_start + 1
            parts.append((f"#{a}" if n == 1 else f"#{a}-#{r['i']}") + f" {_name(r)}" + (f"x{n}" if n > 1 else "")
                         + (" -> " + ", ".join(marks) if marks else ""))
            run_start = None
    return ", ".join(parts)


def game_map(game_dir: Path, rows: list[dict[str, Any]], run: dict[str, Any], session: dict[str, Any]) -> str:
    by_call: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if not r["after_cell"] and r.get("call"):
            by_call[r["call"]].append(r)
    shown: set[int] = set()
    game_id = run.get("game_id", game_dir.name)
    out = [f"# {game_id}: step map", "",
           f"level {run.get('levels_completed')}/{run.get('number_of_levels')} | {len(rows)} actions | "
           f"score {run.get('final_score')} | end {session.get('end_reason')} | turns {session.get('turns')} | "
           f"compactions {session.get('compactions')} | children {session.get('children')}",
           f"actions per level {run.get('actions_per_level')} vs baseline {run.get('baseline_actions')}", ""]
    events = _jsonl(game_dir / "transcript.jsonl") if (game_dir / "transcript.jsonl").exists() else []
    t0 = events[0]["t"] if events else 0.0
    levels = 0
    for ev in events:
        kind, t = ev.get("event"), f"+{ev.get('t', t0) - t0:.0f}s"
        if kind == "message" and ev.get("role") == "assistant":
            usage = ev.get("usage") or ["?", "?"]
            out += ["", f"## Turn {ev.get('turn', '?')} ({t}, prompt {usage[0]} / out {usage[1]} tokens"
                        f"{', CUT OFF' if ev.get('finish') == 'length' else ''})"]
            if ev.get("reasoning"):
                out += [f"**Thought** ({len(ev['reasoning'])} chars):", "", "> " + _cut(ev["reasoning"], 1600)
                        .replace("\n", "\n> "), ""]
            if (ev.get("content") or "").strip():
                out += ["**Said:** " + _cut(ev["content"], 1200), ""]
            for tc in ev.get("tool_calls") or []:
                try:
                    code = json.loads(tc["function"]["arguments"]).get("code") or ""
                except (ValueError, KeyError, TypeError, AttributeError):
                    code = str((tc.get("function") or {}).get("arguments"))
                out += [f"**Code** (`{tc.get('id')}`):", "````python", _cut(code, 4000), "````"]
        elif kind == "message" and ev.get("role") == "tool":
            out += ["**Output:**", "````", _cut(ev.get("content"), 1500), "````"]
        elif kind == "message" and ev.get("role") == "user":
            first = str(ev.get("content") or "").strip().splitlines()[:1]
            label = ev.get("_kind") or "host"
            out += ["", f"> **{label} -> agent** ({t}): {_cut(first[0] if first else '', 300)}"]
        elif kind == "cell":
            acts = by_call.get(str(ev.get("call")), [])
            shown.update(r["i"] for r in acts)
            status = "" if ev.get("status") == "ok" else f" (cell {ev.get('status')})"
            if acts:
                out += [f"**Actions** ({len(acts)}){status}: " + render_actions(acts, levels), ""]
                levels = acts[-1].get("levels") if acts[-1].get("levels") is not None else levels
            elif status:
                out += [f"**No actions**{status}", ""]
        elif kind in ("compaction", "compaction_error", "compaction_skipped", "auto_refine", "spawn", "crash",
                      "llm_error", "reflection_due", "reflection_done", "reflection_skipped", "end"):
            detail = {k: v for k, v in ev.items() if k not in ("t", "event", "stats", "traceback")}
            out += ["", f"- **[{kind}]** {t} " + _cut(json.dumps(detail, default=str), 600)]
            if kind == "crash":
                out += ["````", _cut(ev.get("traceback"), 3000), "````"]
    rest = [r for r in rows if r["i"] not in shown]
    if rest:
        out += ["", "## Actions not matched to a cell (background tasks or no transcript)", "",
                render_actions(rest, None)]
    return "\n".join(out) + "\n"


def build_run(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    runs = {r["game_id"]: r for r in results.get("runs", [])}
    sessions = results.get("sessions", {})
    table = ["# Trace index", "", "| game | levels | actions | recorded | with ref | turns | resets | GAME_OVERs | "
             "full resets | compactions | end | map |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for game_id, run in sorted(runs.items()):
        game_dir = run_dir / "games" / game_id
        session = sessions.get(game_id, {})
        rec_name = Path(session["recording"]).name if session.get("recording") else None
        hits = sorted((run_dir / "recordings").rglob(rec_name)) if rec_name else []
        recording = hits[0] if hits else None
        rows = action_rows(recording, run)
        game_dir.mkdir(parents=True, exist_ok=True)
        with (game_dir / "steps.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        (game_dir / "steps.md").write_text(game_map(game_dir, rows, run, session), encoding="utf-8")
        table.append(
            f"| {game_id} | {run.get('levels_completed')}/{run.get('number_of_levels')} | {run.get('actions')} | "
            f"{len(rows) if recording else 'no'} | {sum(1 for r in rows if r.get('call'))} | {session.get('turns')} | "
            f"{sum(1 for r in rows if r['action'] == 'RESET')} | "
            f"{sum(1 for r in rows if r.get('state') == 'GAME_OVER')} | "
            f"{sum(1 for r in rows if r['full_reset'])} | {session.get('compactions')} | {session.get('end_reason')} | "
            f"[steps.md](games/{game_id}/steps.md) |")
    out = run_dir / "trace.md"
    out.write_text("\n".join(table) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build_run(Path(sys.argv[1] if len(sys.argv) > 1 else ".")))
