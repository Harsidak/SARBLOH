"""Copy an authored run's SFT rows into Alignment/data/ after checking them.

    uv run python Alignment/authoring/export.py runs/authored/ls20_a [--name ls20_a]

Every row must be a cleared level whose requests matched the agent's logged digest (``wire_verified``) and whose
assistant turns carry no flags (tool error, cell error, repeat, game over, cut off). Each row gets an ``id``
("<name>/<game>/L<level>") so rows of several files in one folder stay distinct in ``lora_sft.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def check(rows: list[dict]) -> list[str]:
    problems = []
    for r in rows:
        where = f"level {r.get('level')}"
        if not r.get("level_cleared"):
            problems.append(f"{where}: not cleared")
        if not r.get("wire_verified"):
            problems.append(f"{where}: not wire_verified")
        for m in r.get("messages", []):
            if m.get("role") == "assistant" and (m.get("flags") or not m.get("train", True)):
                problems.append(f"{where}: turn {m.get('turn')} flags {m.get('flags')} train {m.get('train')}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    name = a.name or run.name
    rows = [json.loads(line) for line in (run / "sft_levels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    problems = check(rows)
    if problems:
        raise SystemExit("not exported:\n" + "\n".join(problems))
    out = REPO / "Alignment" / "data" / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"id": f"{name}/{r['game_id']}/L{r['level']}", **r}, ensure_ascii=False) + "\n")
    print(json.dumps({"file": str(out.relative_to(REPO)), "rows": len(rows),
                      "levels": [r["level"] for r in rows], "actions": [r["actions"] for r in rows],
                      "baseline": [r["baseline"] for r in rows], "rhae": [round(r["rhae"], 3) for r in rows]}))


if __name__ == "__main__":
    main()
