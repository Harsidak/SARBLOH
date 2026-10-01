"""Routing diagnostic: WHICH planner spends the actions, at WHAT cost, for WHAT yield.

The premise under test is the one the routing work started from -- "the planners
are basically responsible for the actions". That is checkable, and until now it
was not checked: choose_action is a single function and the identity of the
branch that produced an action died with the call. `MyAgent._route` / `_route_ms`
(pure assignments, no control flow) name the branch and time it; this script
aggregates them.

Per branch it reports:
  n, share     -- how many scored actions that branch spent
  ms           -- mean / p95 wall-clock to DECIDE (RHAE ignores thinking time, but
                  the 9h Kaggle budget does not: slow deciding caps games reached)
  chg%         -- share of that branch's actions that changed the masked world.
                  This is DECISION QUALITY at its most basic: an action that moves
                  nothing bought nothing.
  new%         -- share that reached a state the graph had never seen. Stricter
                  than chg%: re-treading a known state changes pixels and teaches
                  nothing.
  rew          -- score events credited to the branch.

The last two are the point. A branch can be fast, busy and still worthless; only
new% and rew separate a planner that is solving from one that is fidgeting.

Usage:
  PYTHONHASHSEED=0 python scratchpad/route_diag.py [ids...] --seed 0 [--timeout 390]
                                                   [--out FILE]
"""
import os
import sys
import json
import time
import glob
import random
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

import local_eval as LE                                   # noqa: E402
from my_agent import MyAgent, masked_diff                 # noqa: E402


class RouteAgent(MyAgent):
    """MyAgent + per-branch accounting. Overrides nothing that decides."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.R = {}                 # route -> dict of counters
        self._pending = None        # (route, ms) awaiting its outcome next step
        # THE RESET/SCORE INVARIANT, checked empirically on every emitter rather
        # than argued per call site. Two RESETs with no action between them are a
        # full_reset in arcengine (base_game L313), which zeroes levels_completed.
        # `wipes` counts the times the score actually fell -- the observable
        # consequence -- and `dbl` the times the pattern that causes it occurred.
        self._dbl_reset = 0
        self._wipes = 0
        self._last_emitted = ""

    def _slot(self, route):
        if route not in self.R:
            self.R[route] = {"n": 0, "ms": [], "chg": 0, "new": 0, "rew": 0}
        return self.R[route]

    def choose_action(self, frames, latest_frame):
        # Settle the PREVIOUS decision first: its outcome is the frame we are
        # being handed right now. Attribution has to be deferred like this --
        # at decision time the consequence does not exist yet.
        prev_grid, prev_score = self.last_grid, self.last_score
        act = super().choose_action(frames, latest_frame)

        if self._pending is not None and prev_grid is not None:
            route, ms = self._pending
            s = self._slot(route)
            s["n"] += 1
            s["ms"].append(ms)
            nxt = self.last_grid
            if nxt is not None:
                try:
                    diff = masked_diff(prev_grid, nxt, self.sgraph._mask)
                    if diff is None or bool(diff.any()):
                        s["chg"] += 1
                except Exception:
                    pass
                try:
                    if self.sgraph._hash(nxt) not in self._seen_hashes:
                        s["new"] += 1
                        self._seen_hashes.add(self.sgraph._hash(nxt))
                except Exception:
                    pass
            if self.last_score > prev_score:
                s["rew"] += 1

        # Score regressions and the double-RESET that causes them.
        if self.last_score < prev_score:
            self._wipes += 1
        cur = self.last_action_str or ""
        if cur == "RESET" and self._last_emitted == "RESET":
            self._dbl_reset += 1
        self._last_emitted = cur

        self._pending = (self._route or "?", self._route_ms)
        return act

    _seen_hashes: set = None

    def report(self) -> dict:
        rows = []
        tot = sum(v["n"] for v in self.R.values()) or 1
        for route, v in sorted(self.R.items(), key=lambda kv: -kv[1]["n"]):
            ms = sorted(v["ms"])
            if not ms:
                continue
            p95 = ms[min(len(ms) - 1, int(0.95 * len(ms)))]
            rows.append({
                "route": route, "n": v["n"],
                "share": round(100.0 * v["n"] / tot, 1),
                "ms": round(sum(ms) / len(ms), 2), "p95": round(p95, 2),
                "chg_pct": round(100.0 * v["chg"] / v["n"], 1),
                "new_pct": round(100.0 * v["new"] / v["n"], 1),
                "rew": v["rew"],
                # Total seconds this branch spent THINKING across the game. The
                # wall-clock budget is what decides how many games are reachable.
                "secs": round(sum(ms) / 1000.0, 1),
            })
        return {"routes": rows, "dbl_reset": self._dbl_reset,
                "score_wipes": self._wipes}


_LAST = {}


def factory(game_id: str):
    ag = RouteAgent(game_id=game_id)
    ag._seen_hashes = set()
    _LAST["agent"] = ag
    return ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-actions", type=int, default=1500)
    ap.add_argument("--timeout", type=float, default=0.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    files = [f for f in LE._discover_real_games()
             if not os.path.basename(f).startswith("_")]
    if args.games:
        want = set(args.games)
        files = [f for f in files
                 if os.path.splitext(os.path.basename(f))[0] in want]
    if not files:
        print("No games found.")
        return

    out = []
    for gf in files:
        gid = os.path.splitext(os.path.basename(gf))[0]
        stop = {"t0": time.time()}
        if args.timeout:
            import threading

            def _watch(a=args, s=stop):
                while not s.get("done"):
                    time.sleep(2.0)
                    ag = _LAST.get("agent")
                    if ag is not None and time.time() - s["t0"] > a.timeout:
                        ag.MAX_ACTIONS = 0
                        return
            threading.Thread(target=_watch, daemon=True).start()

        r = LE.run_game(gf, factory, seed=args.seed,
                        max_actions=args.max_actions, verbose=False)
        stop["done"] = True
        ag = _LAST.get("agent")
        r["game_id"] = gid                     # diag_eval omits this; do not repeat it
        r.update(ag.report() if ag is not None else {})
        out.append(r)

        print(f"\n=== {gid}  lvl {r.get('levels_completed',0)}/{r.get('win_levels',0)}"
              f"  act={r.get('actions',0)}  {r.get('secs',0)}s"
              f"  dbl_reset={r.get('dbl_reset',0)}"
              f"  score_wipes={r.get('score_wipes',0)} ===", flush=True)
        print(f"  {'route':<22}{'n':>5}{'share':>7}{'ms':>8}{'p95':>8}"
              f"{'chg%':>7}{'new%':>7}{'rew':>5}{'secs':>7}", flush=True)
        for x in r.get("routes", []):
            print(f"  {x['route']:<22}{x['n']:>5}{x['share']:>6.1f}%"
                  f"{x['ms']:>8.2f}{x['p95']:>8.2f}"
                  f"{x['chg_pct']:>6.1f}%{x['new_pct']:>6.1f}%"
                  f"{x['rew']:>5}{x['secs']:>7.1f}", flush=True)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
