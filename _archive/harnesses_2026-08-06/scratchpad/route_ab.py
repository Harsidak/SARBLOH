"""Paired A/B comparator for two route_diag arms.

Levels first -- it is the only thing that scores. Everything else is diagnosis.
A wall-clock/think-time win that costs a level is not a win.

Usage:  python scratchpad/route_ab.py mon moff
        python scratchpad/route_ab.py con coff
"""
import os
import sys
import glob
import json

HERE = os.path.dirname(os.path.abspath(__file__))
HELDOUT = {"ar25", "cd82", "ft09", "g50t", "m0r0", "s5i5", "sk48", "vc33"}


def load(tag):
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, "diag", f"{tag}_*.json"))):
        with open(f, encoding="utf-8") as fh:
            for r in json.load(fh):
                out[r.get("game_id", "?")] = r
    return out


def think(r):
    return sum(x.get("secs", 0.0) for x in (r.get("routes") or []))


def main():
    a_tag, b_tag = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("mon", "moff")
    A, B = load(a_tag), load(b_tag)
    games = sorted(set(A) & set(B))
    if not games:
        print(f"no paired games for {a_tag}/{b_tag} "
              f"(A={sorted(A)} B={sorted(B)})")
        return
    held = [g for g in games if g in HELDOUT]
    if held:
        print(f"WARNING: held-out games present, excluded from the verdict: {held}")
    games = [g for g in games if g not in HELDOUT]

    print(f"{'game':<7}{'levels':>14}{'actions':>16}{'wall s':>15}{'think s':>16}")
    print(f"{'':<7}{a_tag+'->'+b_tag:>14}{a_tag+'->'+b_tag:>16}"
          f"{a_tag+'->'+b_tag:>15}{a_tag+'->'+b_tag:>16}")
    reg, gain = [], []
    for g in games:
        a, b = A[g], B[g]
        la = a.get("levels_completed", 0)
        lb = b.get("levels_completed", 0)
        if lb < la:
            reg.append(g)
        if lb > la:
            gain.append(g)
        flag = "  <== REGRESSED" if lb < la else ("  <== GAINED" if lb > la else "")
        print(f"{g:<7}{la:>6} ->{lb:>4}"
              f"{a.get('actions',0):>9} ->{b.get('actions',0):>5}"
              f"{a.get('secs',0):>9.0f} ->{b.get('secs',0):>4.0f}"
              f"{think(a):>10.1f} ->{think(b):>5.1f}{flag}")

    la = sum(A[g].get("levels_completed", 0) for g in games)
    lb = sum(B[g].get("levels_completed", 0) for g in games)
    wa = sum(A[g].get("secs", 0) for g in games)
    wb = sum(B[g].get("secs", 0) for g in games)
    ta = sum(think(A[g]) for g in games)
    tb = sum(think(B[g]) for g in games)
    wpa = sum(A[g].get("score_wipes", 0) for g in games)
    wpb = sum(B[g].get("score_wipes", 0) for g in games)
    print(f"\nTOTAL ({len(games)} games)")
    print(f"  levels      {la} -> {lb}")
    print(f"  wall clock  {wa:.0f}s -> {wb:.0f}s   ({wb-wa:+.0f}s)")
    print(f"  think time  {ta:.1f}s -> {tb:.1f}s   ({tb-ta:+.1f}s)")
    print(f"  score wipes {wpa} -> {wpb}")
    print(f"  REGRESSED: {reg or 'none'}")
    print(f"  GAINED   : {gain or 'none'}")


if __name__ == "__main__":
    main()
