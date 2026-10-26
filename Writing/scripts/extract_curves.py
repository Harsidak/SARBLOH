"""Levels cleared over wall-clock time: our best 9 h run vs the TAAF/Duck harness on the same server.

Writes Writing/data/levels_vs_time.json. Ours: the [status] lines of the Kaggle log (t = seconds since the games
started). TAAF: benchmark.json of the 100 runs (25 games x 4 passes); a level counts once per game, at the first pass
that clears it ("distinct levels"), and also summed over passes ("sum over passes", what TAAF's own plots show)."""

import ast
import json
import re
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OURS = REPO / "runs/kaggle_sarbloh-experimentation_20261023_115157/sarbloh-experimentation.log"
TAAF = REPO / "runs/TAAF results/benchmark.json"
OUT = REPO / "Writing/data/levels_vs_time.json"


def ours() -> list[tuple[float, int]]:
    pts = [(0.0, 0)]
    for m in re.finditer(r"\[status\] t=(\d+)s playing=\d+ levels=(\d+)", OURS.read_text(encoding="utf-8")):
        pts.append((int(m.group(1)) / 3600, int(m.group(2))))
    return pts


def taaf() -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
    b = json.loads(TAAF.read_text(encoding="utf-8"))
    t0 = datetime.fromisoformat(b["start_time"])
    events = []   # (hours, game, pass, level index)
    for run in b["game_runs"]:
        hist = run["history"] if isinstance(run["history"], list) else ast.literal_eval(run["history"])
        per = run["actions_per_level"] if isinstance(run["actions_per_level"], list) else ast.literal_eval(run["actions_per_level"])
        start = (datetime.fromisoformat(run["started_at"]) - t0).total_seconds()
        walls = [h["wallclock_seconds"] for h in hist]
        mono = all(b >= a for a, b in zip(walls, walls[1:]))
        clock = walls if mono else [sum(walls[:i + 1]) for i in range(len(walls))]
        done = int(run["levels_completed"])
        k = 0
        for lvl in range(done):
            k += per[lvl]
            if 0 < k <= len(clock):
                events.append(((start + clock[k - 1]) / 3600, run["game_id"], lvl))
    events.sort()
    distinct, seen, total = [(0.0, 0)], set(), [(0.0, 0)]
    for i, (h, g, lvl) in enumerate(events):
        total.append((h, i + 1))
        if (g, lvl) not in seen:
            seen.add((g, lvl))
            distinct.append((h, len(seen)))
    return distinct, total


if __name__ == "__main__":
    d, s = taaf()
    o = ours()
    at = lambda pts, h: max([n for t, n in pts if t <= h] or [0])
    for h in (1, 2, 3, 4, 8.2):
        print(f"{h} h: ours {at(o, h)}  taaf distinct {at(d, h)}  taaf sum {at(s, h)}")
    OUT.write_text(json.dumps({"ours": o, "taaf_distinct": d, "taaf_sum": s, "levels_total": 183}), encoding="utf-8")
    print("wrote", OUT)
