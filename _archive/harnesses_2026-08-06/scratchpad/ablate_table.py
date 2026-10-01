"""Read the ablation sweep results and print the with/without/verdict table.

A component's verdict comes from what removing it does to the tier-1 aggregate,
not from what the code looks like. Time is reported alongside because a
component that costs nothing in score but 40% of the wall clock is still a bill.
"""
import os
import json
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ablate")
BASE = os.path.join(HERE, "final_tier1.json")


def agg(path):
    rows = json.load(open(path, encoding="utf-8"))
    n = len(rows)
    return {
        "official": sum(r["official"] for r in rows) / n,
        "honest": sum(r["honest"] for r in rows) / n,
        "levels": sum(r["real_levels"] for r in rows),
        "secs": sum(r["secs"] for r in rows),
        "per_game": {r["game"]: (r["real_levels"], round(r["honest"], 4), r["secs"])
                     for r in rows},
    }


def main():
    base = agg(BASE)
    print(f"{'ablation':<18}{'levels':>7}{'official':>10}{'honest':>9}{'secs':>9}"
          f"   verdict")
    print(f"{'BASELINE (all on)':<18}{base['levels']:>7}{base['official']:>10.4f}"
          f"{base['honest']:>9.4f}{base['secs']:>9.0f}")
    print("-" * 78)
    for path in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        tag = os.path.basename(path)[:-5]
        try:
            a = agg(path)
        except Exception as exc:
            print(f"{tag:<18}  unreadable: {exc}")
            continue
        d_lv = a["levels"] - base["levels"]
        d_sc = a["honest"] - base["honest"]
        d_t = (a["secs"] - base["secs"]) / max(base["secs"], 1e-9)
        if d_lv > 0 or d_sc > 1e-6:
            v = "HARMFUL - removing it helps"
        elif d_lv < 0 or d_sc < -1e-6:
            v = "LOAD-BEARING"
        elif d_t < -0.15:
            v = f"DEAD WEIGHT - and costs {-d_t:.0%} of wall clock"
        else:
            v = "DEAD WEIGHT - score-neutral"
        print(f"{tag:<18}{a['levels']:>7}{a['official']:>10.4f}{a['honest']:>9.4f}"
              f"{a['secs']:>9.0f}   {v}")
        for g, (lv, hs, sc) in a["per_game"].items():
            blv, bhs, bsc = base["per_game"][g]
            if lv != blv or abs(hs - bhs) > 1e-6 or abs(sc - bsc) / max(bsc, 1) > 0.3:
                print(f"    {g}: {blv}->{lv} lv, {bhs:.4f}->{hs:.4f}, "
                      f"{bsc:.0f}s->{sc:.0f}s")


if __name__ == "__main__":
    main()
