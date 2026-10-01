"""Diagnostic eval: the numbers the livelock / world-model work is judged on.

Runs the SAME loop as eval/local_eval.py (it reuses run_game's sibling logic)
but wraps MyAgent so every step is measured. Nothing here changes the agent --
DiagAgent only observes, so a diagnostic run is byte-identical in behaviour to a
plain local_eval run with the same seed.

Per game it reports:
  levels/win_levels, actions, secs                      -- the score
  noop%                                                 -- share of steps with NO
                                                           HUD-masked world change
  maxrun                                                -- longest consecutive
                                                           masked-no-op streak
  ttfc                                                  -- actions To First
                                                           (masked) Change, per level
  prom/dem                                              -- world-model promotions
                                                           and demotions
  verify                                                -- verify score of the last
                                                           promoted model
  repeat%                                               -- share of steps that re-spent
                                                           an ALREADY-OBSERVED
                                                           (state, action) pair
  cells                                                 -- distinct click cells / clicks

Usage:
  PYTHONHASHSEED=0 python scratchpad/diag_eval.py --out FILE [ids...]
                                    [--seed 0] [--max-actions 1500] [--timeout 900]
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

import local_eval as LE            # noqa: E402
from my_agent import MyAgent, base_action, masked_diff   # noqa: E402


class DiagAgent(MyAgent):
    """MyAgent + read-only instrumentation. Overrides nothing that decides."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.d = {
            "steps": 0, "noop": 0, "maxrun": 0, "_run": 0,
            "ttfc": [], "_level_steps": 0, "_level_changed": False,
            "prom": 0, "dem": 0, "verify": None, "hyp": None,
            "acts": {}, "clicks": 0, "cells": set(),
            "repeat": 0, "_seen": set(),
        }
        self._d_model = None
        self._d_score = 0

    def choose_action(self, frames, latest_frame):
        prev = self.last_grid
        prev_act = self.last_action_str
        act = super().choose_action(frames, latest_frame)
        d = self.d
        nxt = self.last_grid

        # --- world-model promotions / demotions (identity of active_model) ---
        m = self.world_model.active_model
        if m is not self._d_model:
            if m is not None:
                d["prom"] += 1
                d["verify"] = round(float(m.accuracy), 3)
                d["hyp"] = str(m.hypothesis)[:120]
            elif self._d_model is not None:
                d["dem"] += 1
            self._d_model = m

        # --- level boundary: restart the time-to-first-change clock ---
        if self.last_score != self._d_score:
            if self._d_score < self.last_score:
                d["ttfc"].append(d["_level_steps"] if d["_level_changed"] else None)
                d["_level_steps"] = 0
                d["_level_changed"] = False
            self._d_score = self.last_score

        # --- per-step masked-change accounting ---
        if prev is not None and nxt is not None and prev_act:
            d["steps"] += 1
            d["_level_steps"] += 1
            diff = masked_diff(prev, nxt, self.sgraph._mask)
            changed = True if diff is None else bool(diff.any())
            if changed:
                if not d["_level_changed"]:
                    d["_level_changed"] = True
                    d["ttfc"].append(d["_level_steps"])
                    d["_level_steps"] = 0
                d["_run"] = 0
            else:
                d["noop"] += 1
                d["_run"] += 1
                d["maxrun"] = max(d["maxrun"], d["_run"])
            # (state, action) re-spend: the pair whose outcome was ALREADY on record
            try:
                key = (self.sgraph._hash(prev), prev_act)
                if key in d["_seen"]:
                    d["repeat"] += 1
                else:
                    d["_seen"].add(key)
            except Exception:
                pass

        a = str(act if isinstance(act, str) else self.last_action_str)
        b = base_action(self.last_action_str or a)
        d["acts"][b] = d["acts"].get(b, 0) + 1
        if b == "ACTION6":
            d["clicks"] += 1
            d["cells"].add(self.last_action_str)
        return act

    def report(self) -> dict:
        d = self.d
        n = max(1, d["steps"])
        ttfc = list(d["ttfc"])
        if d["_level_steps"] and not d["_level_changed"]:
            ttfc.append(None)       # current level never moved the world
        return {
            "steps": d["steps"],
            "noop": d["noop"],
            "noop_pct": round(100.0 * d["noop"] / n, 1),
            "maxrun": d["maxrun"],
            "ttfc": ttfc[:8],
            "prom": d["prom"], "dem": d["dem"],
            "verify": d["verify"], "hyp": d["hyp"],
            "repeat_pct": round(100.0 * d["repeat"] / n, 1),
            "clicks": d["clicks"], "cells": len(d["cells"]),
            "acts": dict(sorted(d["acts"].items())),
            # COMPONENT 5.12: what the goal model actually believes at the end.
            # A goal model that cannot say what it believes cannot be debugged,
            # and "re-spend fell" is not evidence on its own -- this is how a run
            # says WHICH hypothesis moved the number.
            "goal": self._goal_report(),
        }

    def _goal_report(self) -> dict:
        gm = getattr(self, "goals", None)
        if gm is None:
            return {}
        try:
            live = [f for f in gm._feat.values() if f.direction and f.channel]
            return {"pool": len(gm._feat), "live": len(live),
                    "retired": len(gm._retired),
                    "A": sum(1 for f in live if f.channel == "A"),
                    "ranked": bool(gm.frontier_ranker() is not None),
                    # Did the pursuit loop actually CLOSE? armed>0 with everything
                    # else 0 means predicates are being armed and never resolved;
                    # directed>0 is the only proof the re-ranker changed a real
                    # decision, and it is the precondition for any alpha at all.
                    "tally": dict(getattr(gm, "_tally", {})),
                    "steering": bool(gm.steering()),
                    "explain": gm.explain()[:160]}
        except Exception as e:
            return {"error": repr(e)[:80]}


_LAST = {}


def factory(game_id: str):
    ag = DiagAgent(game_id=game_id)
    _LAST["agent"] = ag
    return ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-actions", type=int, default=1500)
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="per-game wall-clock cap (0 = none)")
    ap.add_argument("--out", default="")
    ap.add_argument("--toys", action="store_true", help="run eval/games/ instead")
    args = ap.parse_args()

    files = (sorted(glob.glob(os.path.join(LE.GAMES_DIR, "*.py"))) if args.toys
             else LE._discover_real_games())
    files = [f for f in files if not os.path.basename(f).startswith("_")]
    if args.games:
        want = set(args.games)
        files = [f for f in files
                 if os.path.splitext(os.path.basename(f))[0] in want]
    if not files:
        print("No games found."); return

    # Per-game wall-clock cap: MyAgent.is_done only knows the action budget, so a
    # slow game is stopped by shrinking its budget from a watchdog thread instead.
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
                        ag.MAX_ACTIONS = 0        # is_done() -> True next call
                        return
            threading.Thread(target=_watch, daemon=True).start()
        r = LE.run_game(gf, factory, seed=args.seed,
                        max_actions=args.max_actions, verbose=False)
        stop["done"] = True
        ag = _LAST.get("agent")
        r.update(ag.report() if ag is not None else {})
        r["timeout"] = bool(args.timeout and r["secs"] >= args.timeout - 5)
        out.append(r)
        print(f"  {gid:<6} lvl {r.get('levels_completed',0)}/{r.get('win_levels',0)}"
              f" act={r.get('actions',0):<5} noop={r.get('noop_pct',0):>5.1f}%"
              f" maxrun={r.get('maxrun',0):<5} ttfc={r.get('ttfc')}"
              f" prom/dem={r.get('prom',0)}/{r.get('dem',0)}"
              f" verify={r.get('verify')} repeat={r.get('repeat_pct',0):>5.1f}%"
              f" cells={r.get('cells',0)}/{r.get('clicks',0)}"
              f" {r.get('secs',0)}s{' TIMEOUT' if r['timeout'] else ''}", flush=True)
        _g = r.get("goal") or {}
        if _g:
            print(f"         goal: live={_g.get('live')}/{_g.get('pool')} "
                  f"A={_g.get('A')} ranked={_g.get('ranked')} "
                  f"retired={_g.get('retired')} | {_g.get('explain')}", flush=True)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
