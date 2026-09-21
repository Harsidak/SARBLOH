"""Delta table for two diag_eval runs (baseline vs changed).

Usage: python scratchpad/diag_compare.py PREFIX_A PREFIX_B
       e.g.  ... scratchpad/diag/base scratchpad/diag/brk

Prints per game, then the three headline numbers the goal asks for:
  (a) games ending with NO promoted world model
  (b) games dying at ~100% masked no-op / never moving the world
  (c) level-ups
plus the livelock-specific ones: masked-no-op share, longest inert streak,
share of scored actions re-spent on an already-observed (state, action) pair.

HELD-OUT games are listed separately and excluded from the tuning totals: the
split exists so that a number used to justify a change is never a number the
change was fitted to.
"""
import os
import sys
import glob
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

with open(os.path.join(ROOT, "heldout_split.json"), "r", encoding="utf-8") as f:
    _split = json.load(f)
HELDOUT = set(_split.get("heldout") or _split.get("HELDOUT") or [])


def load(prefix):
    out = {}
    for p in sorted(glob.glob(prefix + "_*.json")):
        with open(p, "r", encoding="utf-8") as f:
            for r in json.load(f):
                out[r["game"]] = r
    return out


def fmt_ttfc(t):
    if not t:
        return "-"
    return ",".join("x" if v is None else str(v) for v in t[:4])


def main():
    a = load(sys.argv[1])
    b = load(sys.argv[2])
    games = sorted(set(a) | set(b))
    hdr = (f"{'game':<6} {'lvl A':>6} {'lvl B':>6} | {'noop% A':>8} {'noop% B':>8} |"
           f" {'maxrun A':>9} {'maxrun B':>9} | {'rpt% A':>7} {'rpt% B':>7} |"
           f" {'cells A':>8} {'cells B':>8} | {'prom A':>7} {'prom B':>7}")
    print(hdr)
    print("-" * len(hdr))
    tot = {"dev": [0, 0, 0, 0, 0, 0], "held": [0, 0, 0, 0, 0, 0]}
    nd = {"dev": 0, "held": 0}
    for g in games:
        ra, rb = a.get(g, {}), b.get(g, {})
        if not ra or not rb:
            print(f"{g:<6} MISSING in {'A' if not ra else 'B'}")
            continue
        tag = "held" if g in HELDOUT else "dev"
        mark = "*" if g in HELDOUT else " "
        print(f"{g:<5}{mark} {ra['levels_completed']:>3}/{ra['win_levels']:<2}"
              f" {rb['levels_completed']:>3}/{rb['win_levels']:<2} |"
              f" {ra['noop_pct']:>8} {rb['noop_pct']:>8} |"
              f" {ra['maxrun']:>9} {rb['maxrun']:>9} |"
              f" {ra['repeat_pct']:>7} {rb['repeat_pct']:>7} |"
              f" {ra['cells']:>4}/{ra['clicks']:<3} {rb['cells']:>4}/{rb['clicks']:<3} |"
              f" {ra['prom']:>3}/{ra['dem']:<3} {rb['prom']:>3}/{rb['dem']:<3}")
        t = tot[tag]
        nd[tag] += 1
        t[0] += ra["levels_completed"]; t[1] += rb["levels_completed"]
        t[2] += ra["noop_pct"];         t[3] += rb["noop_pct"]
        t[4] += ra["repeat_pct"];       t[5] += rb["repeat_pct"]

    print("\n(* = held-out, excluded from the tuning totals below)")
    for tag in ("dev", "held"):
        n = max(1, nd[tag])
        t = tot[tag]
        print(f"\n== {tag.upper()} ({nd[tag]} games) ==")
        print(f"  levels total     {t[0]:>6} -> {t[1]:<6}  ({t[1]-t[0]:+d})")
        print(f"  mean masked-noop {t[2]/n:>6.1f}% -> {t[3]/n:<6.1f}% ({(t[3]-t[2])/n:+.1f})")
        print(f"  mean re-spend    {t[4]/n:>6.1f}% -> {t[5]/n:<6.1f}% ({(t[5]-t[4])/n:+.1f})")
        gs = [g for g in games if (g in HELDOUT) == (tag == "held")
              and g in a and g in b]
        no_wm_a = sum(1 for g in gs if a[g]["prom"] == 0)
        no_wm_b = sum(1 for g in gs if b[g]["prom"] == 0)
        print(f"  games with NO world model  {no_wm_a} -> {no_wm_b}")
        dead_a = sum(1 for g in gs if a[g]["maxrun"] >= 100)
        dead_b = sum(1 for g in gs if b[g]["maxrun"] >= 100)
        print(f"  games with a >=100-action inert streak  {dead_a} -> {dead_b}")
        lv_a = sum(1 for g in gs if a[g]["levels_completed"] > 0)
        lv_b = sum(1 for g in gs if b[g]["levels_completed"] > 0)
        print(f"  games completing >=1 level  {lv_a} -> {lv_b}")
        reg = [g for g in gs if b[g]["levels_completed"] < a[g]["levels_completed"]]
        gain = [g for g in gs if b[g]["levels_completed"] > a[g]["levels_completed"]]
        print(f"  REGRESSED: {reg or 'none'}")
        print(f"  GAINED   : {gain or 'none'}")


if __name__ == "__main__":
    main()
