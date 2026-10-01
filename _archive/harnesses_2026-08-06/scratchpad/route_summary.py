"""Roll the route_*.json shards into the dev-set verdict.

Two questions, in order of authority:
  1) did LEVELS regress?  (the only thing that scores)
  2) where do the actions go, and what do they buy?

Everything else in the table is diagnosis, not evidence. `score_wipes` is here
because a wipe silently deletes a completed level and shows up nowhere else --
the run just looks like it never won.

Usage:  python scratchpad/route_summary.py [glob]
"""
import os
import sys
import glob
import json

HERE = os.path.dirname(os.path.abspath(__file__))

# Held out on purpose: no routing change may be derived from these.
HELDOUT = {"ar25", "cd82", "ft09", "g50t", "m0r0", "s5i5", "sk48", "vc33"}


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "diag", "route_*.json")
    rows = []
    for f in sorted(glob.glob(pat)):
        with open(f, encoding="utf-8") as fh:
            rows.extend(json.load(fh))
    if not rows:
        print(f"no shards matched {pat}")
        return

    rows.sort(key=lambda r: r.get("game_id", ""))
    dev = [r for r in rows if r.get("game_id") not in HELDOUT]
    held = [r for r in rows if r.get("game_id") in HELDOUT]

    print(f"{'game':<7}{'lvl':>8}{'act':>7}{'secs':>8}{'think':>8}"
          f"{'wipes':>7}{'dbl':>5}  top branch (share / chg% / new%)")
    for r in rows:
        gid = r.get("game_id", "?")
        br = r.get("routes") or []
        think = round(sum(x.get("secs", 0.0) for x in br), 1)
        top = br[0] if br else {}
        tag = "(dev)" if gid not in HELDOUT else "(held)"
        print(f"{gid:<7}{r.get('levels_completed',0):>4}/"
              f"{r.get('win_levels',0):<3}{r.get('actions',0):>7}"
              f"{r.get('secs',0):>8.0f}{think:>8.1f}"
              f"{r.get('score_wipes',0):>7}{r.get('dbl_reset',0):>5}  "
              f"{top.get('route','-'):<22}"
              f"{top.get('share',0):>5.1f}% {top.get('chg_pct',0):>5.1f}%"
              f" {top.get('new_pct',0):>5.1f}%  {tag}")

    def block(name, rs):
        if not rs:
            return
        lv = sum(r.get("levels_completed", 0) for r in rs)
        wp = sum(r.get("score_wipes", 0) for r in rs)
        db = sum(r.get("dbl_reset", 0) for r in rs)
        th = sum(sum(x.get("secs", 0.0) for x in (r.get("routes") or [])) for r in rs)
        wall = sum(r.get("secs", 0) for r in rs)
        print(f"\n{name} ({len(rs)} games): levels={lv}  score_wipes={wp}  "
              f"dbl_reset={db}  think={th:.0f}s of {wall:.0f}s wall "
              f"({100.0*th/max(1.0,wall):.1f}%)")

    block("DEV ", dev)
    block("HELD", held)

    # Per-branch totals across DEV only -- the routing verdict.
    agg = {}
    for r in dev:
        for x in r.get("routes") or []:
            a = agg.setdefault(x["route"], {"n": 0, "chg": 0.0, "new": 0.0,
                                            "rew": 0, "secs": 0.0})
            a["n"] += x["n"]
            a["chg"] += x["chg_pct"] * x["n"] / 100.0
            a["new"] += x["new_pct"] * x["n"] / 100.0
            a["rew"] += x.get("rew", 0)
            a["secs"] += x.get("secs", 0.0)
    tot = sum(a["n"] for a in agg.values()) or 1
    print(f"\nDEV per-branch totals ({tot} actions)")
    print(f"  {'route':<24}{'n':>7}{'share':>8}{'chg%':>8}{'new%':>8}"
          f"{'rew':>5}{'secs':>8}")
    for route, a in sorted(agg.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {route:<24}{a['n']:>7}{100.0*a['n']/tot:>7.1f}%"
              f"{100.0*a['chg']/a['n']:>7.1f}%{100.0*a['new']/a['n']:>7.1f}%"
              f"{a['rew']:>5}{a['secs']:>8.1f}")


if __name__ == "__main__":
    main()
