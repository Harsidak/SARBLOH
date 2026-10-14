"""Speed of a Prime run: where each turn's time went, how many levels were cleared how fast, and what was wasted.

    uv run python Sarbloh-Experimentation/harness/speed.py <run_dir> [--vs <other_run_dir>] [--window 45]

<run_dir> is the folder with ``results.json`` (``.../prime_run``) or the folder above it. Sources:
- ``games/<game_id>/transcript.jsonl``: one row per turn. A turn starts at the agent's last event before the request
  and ends when the reply is logged, so its wall time includes the wait for a scheduler slot. The reply's usage gives
  prompt and output tokens; the cells, acts, tool errors and level-ups logged with the same turn number give the rest.
- ``llm-server.log`` (SGLang, next to ``prime_run`` or inside it), optional: its ``ReqTimeStats`` lines give each
  request's cached prompt tokens, queue, prefill and decode time. A turn is matched to the line with the same prompt
  length, an output at most 3 tokens longer (the stop tokens), and an entry time inside the turn. Lines no turn matches are other calls (curator, perception).
- ``results.json``: levels, actions per level and the human baseline per level.

Writes ``speed.md`` (the report), ``speed.json`` (headline numbers, per game, per level) and ``speed_turns.jsonl`` (one
row per turn) in <run_dir>. ``--vs`` adds a column with another run's headline numbers and the change.

Games run in parallel, so per-game times add up to more than the run's wall clock. The headline measure is levels
cleared within the first ``--window`` minutes of the run (all games together), plus levels per hour over the run.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


def _med(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _r(x: float, n: int = 1) -> float:
    return round(float(x), n)


# --- the server log -------------------------------------------------------------------------------------------
_REQ = re.compile(r"ReqTimeStats\((.*?)\)")
_KV = re.compile(r"(\w+)=([\d.]+)")


def server_requests(log: Path | None) -> list[dict[str, float]]:
    """One dict per finished request in an SGLang log: input_len, cached_input_len, output_len, queue_duration,
    initial_prefill_elapsed, post_prefill_elapsed, forward_duration (ms) and entry_time (epoch s)."""
    if log is None or not log.exists():
        return []
    rows = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "ReqTimeStats" not in line:
            continue
        tail = line.split("ReqTimeStats", 1)[1]
        row = {k: float(v) for k, v in _KV.findall(tail)}
        if "input_len" in row and "output_len" in row:
            rows.append(row)
    return rows


class _ServerIndex:
    def __init__(self, reqs: list[dict[str, float]]) -> None:
        self.by_len: dict[int, list[dict[str, float]]] = defaultdict(list)
        for r in reqs:
            self.by_len[int(r["input_len"])].append(r)
        self.used: set[int] = set()

    def match(self, prompt: int, output: int, t0: float, t1: float) -> dict[str, float] | None:
        best, gap = None, None
        for r in self.by_len.get(prompt, []):
            # the server counts the stop token(s) too, so its output can be a few tokens longer than the usage
            if id(r) in self.used or not 0 <= r["output_len"] - output <= 3:
                continue
            entry = r.get("entry_time", 0.0)
            if not t0 - 5 <= entry <= t1 + 5:
                continue
            d = abs(entry - t0)
            if gap is None or d < gap:
                best, gap = r, d
        if best is not None:
            self.used.add(id(best))
        return best


# --- turns ----------------------------------------------------------------------------------------------------
def game_turns(events: list[dict[str, Any]], game_id: str, cap: int | None,
               server: _ServerIndex | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per turn: timing, tokens, what it ran and whether it was wasted. Also the game's start time, level-up times
    and curator time."""
    by_turn: dict[int, dict[str, Any]] = defaultdict(lambda: {"cell_s": 0.0, "cells": 0, "cell_errors": 0,
                                                              "tool_errors": 0, "actions": 0, "levels_up": 0})
    level_ups, curator_s, curator_runs, start = [], 0.0, 0, None
    for e in events:
        ev, turn = e.get("event"), e.get("turn")
        if ev == "system_prompt" and start is None:
            start = e.get("t")
        elif ev == "cell" and turn is not None:
            b = by_turn[turn]
            b["cells"] += 1
            b["cell_s"] += float(e.get("duration_s") or 0)
            b["cell_errors"] += e.get("status") != "ok"
        elif ev == "tool_error" and turn is not None:
            by_turn[turn]["tool_errors"] += 1
        elif ev == "act" and turn is not None:
            by_turn[turn]["actions"] += int(e.get("done") or 0)
        elif ev == "level_up":
            level_ups.append({"t": e.get("t"), "level": e.get("level"), "turn": turn})
            if turn is not None:
                by_turn[turn]["levels_up"] += 1
        elif ev == "curator" and not e.get("error"):
            curator_s += float(e.get("duration_s") or 0)
            curator_runs += 1
    if start is None and events:
        start = events[0].get("t")

    turns, last_t, level = [], start, 0
    ups = sorted(level_ups, key=lambda u: u["t"] or 0)
    for e in events:
        t = e.get("t")
        if e.get("event") == "message" and e.get("role") == "assistant" and e.get("usage"):
            prompt, output = (list(e["usage"]) + [0, 0])[:2]
            t0 = last_t if last_t is not None else t
            n = e.get("turn")
            b = by_turn.get(n, {}) if n is not None else {}
            reasoning = len(e.get("reasoning") or "")
            content = len(str(e.get("content") or "")) + len(json.dumps(e.get("tool_calls") or []))
            finish = e.get("finish") or ""
            row = {"game": game_id, "turn": n, "level": level, "t0": t0, "t1": t,
                   "wall_s": _r(max(0.0, (t or 0) - (t0 or 0)), 2), "prompt": int(prompt), "output": int(output),
                   "reasoning_share": _r(reasoning / max(1, reasoning + content), 2), "finish": finish,
                   "effort": e.get("effort"), "tool_calls": len(e.get("tool_calls") or []),
                   "cell_s": _r(b.get("cell_s", 0.0), 2), "cell_errors": b.get("cell_errors", 0),
                   "tool_errors": b.get("tool_errors", 0), "actions": b.get("actions", 0),
                   "levels_up": b.get("levels_up", 0)}
            row["cut_off"] = finish == "length" or bool(cap and output >= cap)
            row["no_tool"] = not row["tool_calls"] and not row["cut_off"]
            row["wasted"] = row["cut_off"] or row["no_tool"] or bool(row["tool_errors"] or row["cell_errors"])
            row["idle"] = row["actions"] == 0
            m = server.match(row["prompt"], row["output"], t0 or 0, t or 0) if server else None
            if m:
                row.update({"cached": int(m.get("cached_input_len", 0)),
                            "queue_s": _r(m.get("queue_duration", 0) / 1000, 2),
                            "prefill_s": _r(m.get("initial_prefill_elapsed", 0) / 1000, 2),
                            "decode_s": _r(m.get("post_prefill_elapsed", 0) / 1000, 2)})
                row["wait_s"] = _r(max(0.0, row["wall_s"] - m.get("forward_duration", 0) / 1000
                                       - m.get("queue_duration", 0) / 1000), 2)
            turns.append(row)
            level += row["levels_up"]
        if t is not None:
            last_t = t
    info = {"start": start, "level_ups": ups, "curator_s": curator_s, "curator_runs": curator_runs,
            "end": events[-1].get("t") if events else start}
    return turns, info


# --- the report -----------------------------------------------------------------------------------------------
def _cap(results: dict[str, Any]) -> int | None:
    cfg = results.get("config") or {}
    for key in ("max_tokens_per_turn", "max_tokens"):
        if cfg.get(key):
            return int(cfg[key])
        if (cfg.get("agent") or {}).get(key):
            return int(cfg["agent"][key])
    return None


def _find(run_dir: Path) -> tuple[Path, Path | None]:
    run_dir = Path(run_dir)
    if not (run_dir / "results.json").exists() and (run_dir / "prime_run" / "results.json").exists():
        run_dir = run_dir / "prime_run"
    log = next((p for p in (run_dir / "llm-server.log", run_dir.parent / "llm-server.log") if p.exists()), None)
    return run_dir, log


def analyze(run_dir: Path, window_min: float = 45.0) -> dict[str, Any]:
    run_dir, log = _find(run_dir)
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    runs = {r["game_id"]: r for r in results.get("runs", [])}
    cap = _cap(results)
    reqs = server_requests(log)
    server = _ServerIndex(reqs) if reqs else None

    turns: list[dict[str, Any]] = []
    infos: dict[str, dict[str, Any]] = {}
    for game_id in sorted(runs):
        path = run_dir / "games" / game_id / "transcript.jsonl"
        if not path.exists():
            continue
        got, info = game_turns(_jsonl(path), game_id, cap, server)
        turns += got
        infos[game_id] = info
    starts = [i["start"] for i in infos.values() if i["start"]]
    ends = [i["end"] for i in infos.values() if i["end"]]
    t_start = min(starts) if starts else 0.0
    wall_min = (max(ends) - t_start) / 60 if ends else 0.0

    # levels: one row per cleared level, timed from the game's start or the previous level-up
    levels = []
    for game_id, info in infos.items():
        run = runs[game_id]
        base, spent = run.get("baseline_actions") or [], run.get("actions_per_level") or []
        prev = info["start"]
        gturns = [t for t in turns if t["game"] == game_id]
        for k, up in enumerate(info["level_ups"]):
            lt = [t for t in gturns if t["level"] == k]
            human = base[k] if k < len(base) else None
            acts = spent[k] if k < len(spent) else sum(t["actions"] for t in lt)
            levels.append({"game": game_id, "level": k + 1, "cleared_at_min": _r((up["t"] - t_start) / 60, 1),
                           "minutes": _r((up["t"] - prev) / 60, 1), "turns": len(lt),
                           "wasted_turns": sum(t["wasted"] for t in lt), "actions": acts, "human": human,
                           "rhae": _r(min((human / acts) ** 2, 1.15), 3) if human and acts else None,
                           "output_tokens": sum(t["output"] for t in lt)})
            prev = up["t"]
    cleared_t = sorted(lv["cleared_at_min"] for lv in levels)

    def _in(minutes: float) -> int:
        return sum(1 for m in cleared_t if m <= minutes)

    n = max(1, len(turns))
    walls = [t["wall_s"] for t in turns]
    outs = [t["output"] for t in turns]
    matched = [t for t in turns if "decode_s" in t]
    decode = sum(t["decode_s"] for t in matched)
    prefill = sum(t["prefill_s"] for t in matched)
    queue = sum(t["queue_s"] for t in matched)
    wait = sum(t["wait_s"] for t in matched)
    cells = sum(t["cell_s"] for t in turns)
    curator = sum(i["curator_s"] for i in infos.values())
    turn_wall = sum(walls)
    busy = turn_wall + cells + curator
    prompt_m = sum(t["prompt"] for t in matched)
    other = [r for r in reqs if id(r) not in server.used] if server else []
    concurrency = (turn_wall / 60) / wall_min if wall_min else 0.0
    head = {
        "games": len(infos), "run_minutes": _r(wall_min), "levels_cleared": len(levels),
        f"levels_in_{int(window_min)}_min": _in(window_min), "levels_in_15_min": _in(15), "levels_in_30_min": _in(30),
        "levels_in_60_min": _in(60), "levels_per_hour": _r(len(levels) / (wall_min / 60), 2) if wall_min else 0.0,
        "minutes_per_level_median": _r(_med([lv["minutes"] for lv in levels])),
        "turns": len(turns), "turns_per_level_median": _r(_med([lv["turns"] for lv in levels])),
        "turn_s_median": _r(_med(walls)), "turn_s_p90": _r(_pct(walls, 0.9)),
        "turns_per_minute_all_games": _r(len(turns) / wall_min, 2) if wall_min else 0.0,
        "games_in_flight_mean": _r(concurrency, 1),
        "output_tokens_per_minute_all_games": int(sum(outs) / wall_min) if wall_min else 0,
        "output_tokens_median": int(_med(outs)), "output_tokens_p90": int(_pct(outs, 0.9)),
        "output_tokens_max": max(outs or [0]), "reasoning_share_mean": _r(sum(t["reasoning_share"] for t in turns) / n, 2),
        "prompt_tokens_median": int(_med([t["prompt"] for t in turns])),
        "actions_per_turn": _r(sum(t["actions"] for t in turns) / n, 2),
        "share_cut_off": _r(sum(t["cut_off"] for t in turns) / n, 3),
        "share_no_tool": _r(sum(t["no_tool"] for t in turns) / n, 3),
        "share_errors": _r(sum(bool(t["tool_errors"] or t["cell_errors"]) for t in turns) / n, 3),
        "share_wasted": _r(sum(t["wasted"] for t in turns) / n, 3),
        "share_idle_turns": _r(sum(t["idle"] for t in turns) / n, 3),
        "time_in_wasted_turns": _r(sum(t["wall_s"] for t in turns if t["wasted"]) / max(1e-9, turn_wall), 3),
        "server_matched_turns": len(matched),
        "cache_hit": _r(sum(t["cached"] for t in matched) / prompt_m, 3) if prompt_m else None,
        "decode_tok_s_per_request": _r(sum(t["output"] for t in matched) / decode, 1) if decode else None,
        "share_decode": _r(decode / busy, 3) if matched and busy else None,
        "share_prefill": _r(prefill / busy, 3) if matched and busy else None,
        "share_server_queue": _r(queue / busy, 3) if matched and busy else None,
        "share_wait_for_slot": _r(wait / busy, 3) if matched and busy else None,
        "share_cells": _r(cells / busy, 3) if busy else None,
        "share_curator": _r(curator / busy, 3) if busy else None,
        "other_requests": len(other),
        "other_requests_s": _r(sum(r.get("forward_duration", 0) for r in other) / 1000),
        "rhae_median_cleared": _r(_med([lv["rhae"] for lv in levels if lv["rhae"] is not None]), 3),
        "actions_vs_human_median": _r(_med([lv["actions"] / lv["human"] for lv in levels if lv["human"]]), 2),
    }
    games = []
    for game_id, info in infos.items():
        gt = [t for t in turns if t["game"] == game_id]
        run = runs[game_id]
        games.append({"game": game_id, "levels": f"{run.get('levels_completed')}/{run.get('number_of_levels')}",
                      "minutes": _r(((info["end"] or 0) - (info["start"] or 0)) / 60), "turns": len(gt),
                      "turn_s_median": _r(_med([t["wall_s"] for t in gt])),
                      "output_median": int(_med([t["output"] for t in gt])),
                      "cut_off": sum(t["cut_off"] for t in gt), "wasted": sum(t["wasted"] for t in gt),
                      "actions": run.get("actions"), "curator_s": _r(info["curator_s"]),
                      "first_level_min": _r((info["level_ups"][0]["t"] - info["start"]) / 60)
                      if info["level_ups"] else None})
    return {"run_dir": str(run_dir), "server_log": str(log) if log else None, "cap": cap, "window_min": window_min,
            "headline": head, "games": games, "levels": levels, "turns": turns}


_LABELS = [
    ("levels_in_{w}_min", "levels cleared in the first {w} min (all games)"),
    ("levels_in_15_min", "levels cleared in the first 15 min"), ("levels_in_30_min", "... in the first 30 min"),
    ("levels_in_60_min", "... in the first 60 min"), ("levels_cleared", "levels cleared, whole run"),
    ("levels_per_hour", "levels per hour, whole run"), ("run_minutes", "run wall clock (min)"),
    ("minutes_per_level_median", "minutes per cleared level (median)"),
    ("turns_per_level_median", "turns per cleared level (median)"),
    ("turn_s_median", "turn wall time, median (s)"), ("turn_s_p90", "turn wall time, p90 (s)"),
    ("turns_per_minute_all_games", "turns per minute, all games"), ("games_in_flight_mean", "games inside a turn at once, mean"),
    ("output_tokens_per_minute_all_games", "reply tokens per minute, all games"),
    ("output_tokens_median", "reply tokens, median"), ("output_tokens_p90", "reply tokens, p90"),
    ("output_tokens_max", "reply tokens, max"), ("reasoning_share_mean", "share of the reply that is thinking (chars)"),
    ("prompt_tokens_median", "prompt tokens, median"), ("cache_hit", "prompt tokens served from cache"),
    ("decode_tok_s_per_request", "decode speed per request (tok/s)"),
    ("share_decode", "busy time: decoding"), ("share_prefill", "busy time: prefill"),
    ("share_server_queue", "busy time: server queue"), ("share_wait_for_slot", "busy time: waiting for a scheduler slot (others use the GPU)"),
    ("share_cells", "busy time: running cells"), ("share_curator", "busy time: curator"),
    ("actions_per_turn", "actions per turn"), ("share_idle_turns", "turns with no action"),
    ("share_cut_off", "turns cut off at the cap"), ("share_no_tool", "turns with no tool call"),
    ("share_errors", "turns with a tool or cell error"), ("share_wasted", "wasted turns (any of the three)"),
    ("time_in_wasted_turns", "share of turn time in wasted turns"),
    ("other_requests", "other model calls (curator etc.)"), ("other_requests_s", "their server time (s)"),
    ("actions_vs_human_median", "actions / human baseline per cleared level (median)"),
    ("rhae_median_cleared", "RHAE per cleared level (median)"),
]


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and 0 < abs(v) < 1:
        return f"{v:.3f}".rstrip("0")
    return str(v)


def render(a: dict[str, Any], vs: dict[str, Any] | None = None) -> str:
    w = int(a["window_min"])
    h, o = a["headline"], (vs or {}).get("headline")
    lines = ["# Speed report", "", f"run `{a['run_dir']}` | server log `{a['server_log']}` | reply cap {a['cap']}"]
    if vs:
        lines.append(f"compared with `{vs['run_dir']}`")
    lines += ["", "Busy time = turn wall time + cell time + curator time, summed over games (they run in parallel).",
              "Turn wall time runs from the agent's last event to the logged reply, so it includes the wait for a "
              "scheduler slot.", "", "## Headline", ""]
    lines += ["| measure | this run |" + (" other | change |" if o else ""),
              "|---|---|" + ("---|---|" if o else "")]
    for key, label in _LABELS:
        key, label = key.format(w=w), label.format(w=w)
        cur = h.get(key)
        row = f"| {label} | {_fmt(cur)} |"
        if o:
            old = o.get(key)
            diff = _fmt(_r(cur - old, 3)) if isinstance(cur, (int, float)) and isinstance(old, (int, float)) else "-"
            row += f" {_fmt(old)} | {diff} |"
        lines.append(row)
    lines += ["", "## Per game", "",
              "| game | levels | minutes | first level (min) | turns | turn s (median) | reply tokens (median) | "
              "cut off | wasted | actions | curator s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in sorted(a["games"], key=lambda g: g["game"]):
        lines.append(f"| {g['game']} | {g['levels']} | {g['minutes']} | {_fmt(g['first_level_min'])} | {g['turns']} | "
                     f"{g['turn_s_median']} | {g['output_median']} | {g['cut_off']} | {g['wasted']} | {g['actions']} | "
                     f"{g['curator_s']} |")
    lines += ["", "## Cleared levels, in the order they were cleared", "",
              "| cleared at (min) | game | level | minutes | turns | wasted turns | actions | human | RHAE | "
              "reply tokens |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lv in sorted(a["levels"], key=lambda lv: lv["cleared_at_min"]):
        lines.append(f"| {lv['cleared_at_min']} | {lv['game']} | {lv['level']} | {lv['minutes']} | {lv['turns']} | "
                     f"{lv['wasted_turns']} | {lv['actions']} | {_fmt(lv['human'])} | {_fmt(lv['rhae'])} | "
                     f"{lv['output_tokens']} |")
    slow = sorted(a["turns"], key=lambda t: -t["wall_s"])[:15]
    lines += ["", "## Slowest turns", "",
              "| game | turn | level | wall s | decode s | prefill s | wait s | prompt | cached | reply | finish | "
              "actions |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in slow:
        lines.append(f"| {t['game']} | {t['turn']} | {t['level']} | {t['wall_s']} | {_fmt(t.get('decode_s'))} | "
                     f"{_fmt(t.get('prefill_s'))} | {_fmt(t.get('wait_s'))} | {t['prompt']} | {_fmt(t.get('cached'))} "
                     f"| {t['output']} | {t['finish']} | {t['actions']} |")
    return "\n".join(lines) + "\n"


def build(run_dir: Path, vs_dir: Path | None = None, window_min: float = 45.0) -> Path:
    a = analyze(run_dir, window_min)
    vs = analyze(vs_dir, window_min) if vs_dir else None
    out_dir = Path(a["run_dir"])
    with (out_dir / "speed_turns.jsonl").open("w", encoding="utf-8") as fh:
        for t in a["turns"]:
            fh.write(json.dumps(t) + "\n")
    (out_dir / "speed.json").write_text(json.dumps({k: v for k, v in a.items() if k != "turns"}, indent=1,
                                                   default=str), encoding="utf-8")
    out = out_dir / "speed.md"
    out.write_text(render(a, vs), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--vs", type=Path, default=None, help="another run to compare with")
    ap.add_argument("--window", type=float, default=45.0, help="minutes for the levels-cleared headline")
    args = ap.parse_args(argv)
    out = build(args.run_dir, args.vs, args.window)
    print(out)
    h = json.loads((out.parent / "speed.json").read_text(encoding="utf-8"))["headline"]
    print(json.dumps(h, indent=1))


if __name__ == "__main__":
    sys.exit(main())
