"""The error budget and the action-ratio distribution of the best offline run, from its own speed report.

Reads runs/kaggle_sarbloh-experimentation_20261023_115157/prime_run/{speed.md,summary.json}. Per game the official
score is min(sum_l l*S_l / sum_l l, 100 * sum_{cleared} l / sum_l l) with S_l = min(100 (h_l/a_l)^2, 115)
(eval/official_score.py). The points lost from 100 split into three parts that add up exactly:
  never reached  = 100 * weight of the levels after the first uncleared one / total weight
  attempted      = 100 * weight of the first uncleared level / total weight
  overhead       = 100 * cleared weight / total weight - score  (action overhead on cleared levels, net of the cap)
Writes Writing/data/error_budget.json. Run: uv run python Writing/scripts/error_budget.py
"""

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/kaggle_sarbloh-experimentation_20261023_115157/prime_run"


def cleared_levels() -> list[dict]:
    lines = (RUN / "speed.md").read_text(encoding="utf-8").splitlines()
    start = lines.index("## Cleared levels, in the order they were cleared")
    rows = []
    for ln in lines[start + 1:]:
        if not ln.strip() or ln.startswith("| cleared") or ln.startswith("|---"):
            continue
        if not ln.startswith("|"):
            break
        c = [x.strip() for x in ln.strip("|").split("|")]
        rows.append({"game": c[1].split("-")[0], "level": int(c[2]), "actions": int(c[6]), "human": int(c[9]),
                     "rhae": float(c[10])})
    return rows


def main() -> None:
    per_game = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))["per_game"]
    rows = cleared_levels()
    budget = {"never_reached": 0.0, "attempted": 0.0, "overhead": 0.0}
    check = []
    for gid, v in per_game.items():
        g, n = gid.split("-")[0], v["of"]
        mine = {r["level"]: r for r in rows if r["game"] == g}
        total = n * (n + 1) / 2
        s = sum(l * min(100 * (mine[l]["human"] / mine[l]["actions"]) ** 2, 115) for l in mine) / total
        cap = 100 * sum(mine) / total
        score = min(s, cap)
        check.append(abs(score - v["score"]))
        first_open = next((l for l in range(1, n + 1) if l not in mine), None)
        if first_open is not None:
            budget["attempted"] += 100 * first_open / total
            budget["never_reached"] += 100 * sum(l for l in range(first_open + 1, n + 1) if l not in mine) / total
        budget["overhead"] += cap - score
    k = len(per_game)
    budget = {key: val / k for key, val in budget.items()}
    ratios = [r["actions"] / r["human"] for r in rows]
    out = {"games": k, "cleared": len(rows), "mean_score": sum(v["score"] for v in per_game.values()) / k,
           "max_score_mismatch": max(check), "budget": budget,
           "ratio_median": statistics.median(ratios), "ratios": ratios,
           "share_below_1": sum(x <= 1 for x in ratios) / len(ratios),
           "share_capped": sum(r["rhae"] >= 1.15 for r in rows) / len(rows)}
    (REPO / "Writing/data/error_budget.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print({key: val for key, val in out.items() if key != "ratios"})


if __name__ == "__main__":
    main()
