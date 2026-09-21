"""The scored local harness. Runs the framework loop and reports BOTH scores.

Scoring lives in eval/official_score.py and nowhere else -- this file must never
transcribe the RHAE formula again. See that module's docstring for why.

    official  what the leaderboard pays, 0..100. Driven by level-CHANGE events,
              so a full_reset() that wipes the score is credited as another
              completed level. Reset churn inflates this without solving
              anything (measured: lp85 scored 50.5 off one real level).
    honest    the same formula with each level counted once, 0..100. This is the
              capability number. Steer on it.
    infl      official / honest. >1 means the leaderboard number is being
              carried by churn rather than by progress.

Both are printed because either one alone misleads: `honest` alone cannot
explain the leaderboard, and `official` alone cannot tell you whether the agent
learned anything.

INTENT REPORTING. A score with no attribution cannot steer anything, so every
run also reports which routing branch spent the actions, which branch was on the
clock when a level was banked, and which branch was on the clock when a level
was wiped. "graph burned 93% of the budget and banked nothing" is a finding;
"RHAE 0.0001" is not.

Usage:
    python eval/rhae_eval.py                 # all real games
    python eval/rhae_eval.py lp85 ft09       # subset
    python eval/rhae_eval.py --seed 0 --timeout 300 --out run.json

Seeding: MyAgent seeds its RNG from ARC_AGENT_SEED + hash(game_id), so pin
PYTHONHASHSEED=0 or runs are not reproducible.
"""
import os
import sys
import glob
import json
import time
import random
import logging
import argparse
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.dirname(HERE)
sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, HERE)
REAL_GAMES_DIR = os.path.join(HERE, "real_games")

from arc_agi.local_wrapper import LocalEnvironmentWrapper      # noqa: E402
from arc_agi.models import EnvironmentInfo                     # noqa: E402
from official_score import LevelEvents, score_run              # noqa: E402


def _quiet_logger():
    lg = logging.getLogger("rhae_eval")
    lg.setLevel(logging.ERROR)
    if not lg.handlers:
        lg.addHandler(logging.StreamHandler(sys.stderr))
    return lg


def _class_name_from_file(path):
    import importlib.util
    from arcengine import ARCBaseGame
    spec = importlib.util.spec_from_file_location("_probe_" + os.path.basename(path), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, obj in vars(mod).items():
        if isinstance(obj, type) and issubclass(obj, ARCBaseGame) and obj is not ARCBaseGame:
            return name
    raise RuntimeError(f"No ARCBaseGame subclass found in {path}")


def _games_index():
    idx = os.path.join(REAL_GAMES_DIR, "games_index.json")
    with open(idx, "r", encoding="utf-8") as f:
        return {g["id"]: g for g in json.load(f)}


def _discover_real_games():
    files = []
    for gdir in sorted(glob.glob(os.path.join(REAL_GAMES_DIR, "*"))):
        if not os.path.isdir(gdir):
            continue
        gid = os.path.basename(gdir)
        hits = glob.glob(os.path.join(gdir, "*", f"{gid}.py"))
        if hits:
            files.append(sorted(hits)[0])
    return files


_AGENT_FILE = ""          # set by --agent-file; "" => the normal `my_agent` module
_AGENT_CACHE = {}


def _load_agent_class():
    """MyAgent from `my_agent.py`, or from an arbitrary file via --agent-file.

    The alternate builds live under names Python cannot import (`my_agent
    original.py` has a space), so they are loaded by PATH. Cached: re-executing
    the module per game would reset its module-level state and cost seconds.
    """
    if not _AGENT_FILE:
        from my_agent import MyAgent
        return MyAgent
    if _AGENT_FILE not in _AGENT_CACHE:
        from agent_loader import load_agent_module
        _AGENT_CACHE[_AGENT_FILE] = load_agent_module(_AGENT_FILE).MyAgent
    return _AGENT_CACHE[_AGENT_FILE]


def run_game(game_file, seed=0, max_actions=1500, per_game_timeout=None):
    """Play one game, recording every level EVENT and who caused it."""
    MyAgent = _load_agent_class()

    game_id = os.path.splitext(os.path.basename(game_file))[0]
    class_name = _class_name_from_file(game_file)
    info = EnvironmentInfo(game_id=game_id, class_name=class_name,
                           local_dir=os.path.dirname(os.path.abspath(game_file)))
    wrapper = LocalEnvironmentWrapper(info, _quiet_logger(),
                                      scorecard_id="local", seed=seed)
    frame0 = wrapper.observation_space
    if frame0 is None:
        return {"game": game_id, "error": "failed to load/reset game"}

    win_levels = int(getattr(frame0, "win_levels", 0) or 0)
    os.environ["ARC_AGENT_SEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    agent = MyAgent(game_id=game_id)
    frames = [frame0]

    ev = LevelEvents()
    ev.observe(getattr(frame0, "levels_completed", 0) or 0, 0)

    # --- intent ledger: who spends, who scores, who loses ------------------
    spend = Counter()           # route -> actions
    banked = Counter()          # route -> levels first-reached on its watch
    wiped = Counter()           # route -> level wipes on its watch
    deaths = Counter()          # route -> GAME_OVER frames on its watch
    # Spend alone cannot distinguish "working" from "fidgeting": a branch that owns
    # 87% of the budget and never moves the board is doing nothing expensively.
    changed = Counter()         # route -> actions that altered the frame at all
    novel = Counter()           # route -> actions that produced a NEVER-SEEN frame
    seen_frames: set = set()
    resets = 0
    game_overs = 0
    route_ms = defaultdict(float)

    def _fhash(fr):
        g = getattr(fr, "frame", None)
        try:
            return hash(np.asarray(g, dtype=np.int16).tobytes())
        except Exception:
            return hash(repr(g))

    t0 = time.time()
    stall = None
    while not agent.is_done(frames, frames[-1]) and agent.action_counter <= max_actions:
        if per_game_timeout and (time.time() - t0) > per_game_timeout:
            stall = f"timeout>{per_game_timeout}s"
            break
        tick = time.perf_counter()
        try:
            action = agent.choose_action(frames, frames[-1])
        except Exception as e:
            stall = f"choose_action raised: {e}"
            break
        think = time.perf_counter() - tick

        route = str(getattr(agent, "_route", "") or "?")
        spend[route] += 1
        route_ms[route] += think * 1000.0

        data = {}
        if hasattr(action, "is_complex") and action.is_complex():
            data = {"x": int(action.action_data.x), "y": int(action.action_data.y)}
        prev_h = _fhash(frames[-1])
        resp = wrapper.step(action, data=data)
        if resp is None:
            stall = "wrapper.step returned None"
            break
        cur_h = _fhash(resp)
        if cur_h != prev_h:
            changed[route] += 1
        if cur_h not in seen_frames:
            novel[route] += 1
            seen_frames.add(cur_h)
        frames.append(resp)
        agent.action_counter += 1

        act_name = str(getattr(agent, "last_action_str", "") or "")
        if act_name.startswith("RESET"):
            resets += 1
        st = getattr(getattr(resp, "state", None), "name", "")
        if st == "GAME_OVER":
            game_overs += 1
            deaths[route] += 1

        before_peak, before_wipes = ev.real_levels, ev.wipes
        if ev.observe(getattr(resp, "levels_completed", 0) or 0, agent.action_counter):
            if ev.real_levels > before_peak:
                banked[route] += ev.real_levels - before_peak
            if ev.wipes > before_wipes:
                wiped[route] += ev.wipes - before_wipes

    last = frames[-1]
    state_name = getattr(getattr(last, "state", None), "name", str(getattr(last, "state", None)))
    return {
        "game": game_id,
        "win_levels": win_levels,
        "actions": agent.action_counter,
        "events": ev,
        "state": state_name,
        "won": state_name == "WIN",
        "secs": round(time.time() - t0, 1),
        "stall": stall,
        "resets": resets,
        "game_overs": game_overs,
        "spend": dict(spend),
        "banked": dict(banked),
        "wiped": dict(wiped),
        "deaths": dict(deaths),
        "changed": dict(changed),
        "novel": dict(novel),
        "route_ms": {k: round(v, 1) for k, v in route_ms.items()},
    }


def _fmt_routes(spend, banked, wiped, total, top=4):
    """The one line that says where the budget went and what it bought."""
    if not spend:
        return ""
    parts = []
    for r, n in sorted(spend.items(), key=lambda kv: -kv[1])[:top]:
        tag = f"{r} {100.0 * n / max(1, total):.0f}%"
        if banked.get(r):
            tag += f" +{banked[r]}L"
        if wiped.get(r):
            tag += f" -{wiped[r]}W"
        parts.append(tag)
    return "  ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*", help="game ids (default: all real games)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-actions", type=int, default=1500)
    ap.add_argument("--timeout", type=float, default=None, help="per-game wall cap (s)")
    ap.add_argument("--out", default="", help="write the full result JSON here")
    ap.add_argument("--agent-file", default="",
                    help="evaluate an ALTERNATE agent build by path (e.g. "
                         "'my_agent original.py'). Default: the my_agent module.")
    args = ap.parse_args()

    global _AGENT_FILE
    _AGENT_FILE = args.agent_file
    if _AGENT_FILE:
        print(f"[agent] {_AGENT_FILE}")

    idx = _games_index()
    all_files = _discover_real_games()
    if args.games:
        wanted = set(args.games)
        all_files = [f for f in all_files
                     if os.path.splitext(os.path.basename(f))[0] in wanted]
    if not all_files:
        print("No games found."); return

    print(f"RHAE eval - {len(all_files)} game(s), seed={args.seed}, "
          f"cap={args.max_actions} actions")
    print("official = leaderboard units (level-CHANGE events, churn counts)")
    print("honest   = same formula, each level counted once (capability)\n")
    print(f"{'game':<7}{'real':>5}{'cred':>5}{'official':>10}{'honest':>9}"
          f"{'infl':>7}{'wipe':>6}{'acts':>6}{'secs':>7}  where the budget went")

    rows, dump = [], []
    for gf in all_files:
        gid = os.path.splitext(os.path.basename(gf))[0]
        meta = idx.get(gid, {})
        res = run_game(gf, seed=args.seed, max_actions=args.max_actions,
                       per_game_timeout=args.timeout)
        if "error" in res:
            print(f"{gid:<7}  ERROR: {res['error']}")
            continue
        sc = score_run(res["events"], res["actions"],
                       meta.get("baseline_steps") or [], res["win_levels"])
        rows.append((gid, sc, res))
        stall = f"  [{res['stall']}]" if res.get("stall") else ""
        warn = f"  !{sc['warn']}" if sc.get("warn") else ""
        print(f"{gid:<7}{sc['real_levels']:>3}/{res['win_levels']:<2}"
              f"{sc['credited_levels']:>5}{sc['official']:>10.4f}{sc['honest']:>9.4f}"
              f"{sc['inflation']:>7.1f}{sc['wipes']:>6}{res['actions']:>6}"
              f"{res['secs']:>6}s  "
              f"{_fmt_routes(res['spend'], res['banked'], res['wiped'], res['actions'])}"
              f"{stall}{warn}", flush=True)
        d = dict(sc)
        d.update({k: v for k, v in res.items() if k != "events"})
        d["level_events"] = res["events"].to_json()
        dump.append(d)

    if not rows:
        return
    off = sum(r[1]["official"] for r in rows) / len(rows)
    hon = sum(r[1]["honest"] for r in rows) / len(rows)
    print(f"\nOVERALL over {len(rows)} games:")
    print(f"  official (leaderboard) : {off:.4f}   = {off / 100:.6f} as a fraction")
    print(f"  honest   (capability)  : {hon:.4f}   = {hon / 100:.6f} as a fraction")
    if hon > 0:
        print(f"  inflation              : {off / hon:.1f}x")
    total_wipes = sum(r[1]["wipes"] for r in rows)
    real = sum(r[1]["real_levels"] for r in rows)
    poss = sum(r[2]["win_levels"] for r in rows)
    print(f"  real levels banked     : {real}/{poss}   wipe events: {total_wipes}")
    if total_wipes:
        print(f"  NOTE: {total_wipes} score wipes inflate `official`. A rise there "
              f"with no rise in `honest` is churn, not progress.")

    # Where the whole budget went, pooled -- the steering view.
    pooled_spend, pooled_bank, pooled_wipe = Counter(), Counter(), Counter()
    pooled_chg, pooled_new = Counter(), Counter()
    for _g, _s, r in rows:
        pooled_spend.update(r["spend"]); pooled_bank.update(r["banked"])
        pooled_wipe.update(r["wiped"])
        pooled_chg.update(r.get("changed", {})); pooled_new.update(r.get("novel", {}))
    tot = sum(pooled_spend.values()) or 1
    print(f"\n  {'route':<16}{'actions':>9}{'share':>8}{'chg%':>7}{'new%':>7}"
          f"{'banked':>8}{'wiped':>7}{'per-1k':>9}")
    for r, n in pooled_spend.most_common(10):
        per_k = 1000.0 * pooled_bank.get(r, 0) / n if n else 0.0
        chg = 100.0 * pooled_chg.get(r, 0) / n if n else 0.0
        new = 100.0 * pooled_new.get(r, 0) / n if n else 0.0
        print(f"  {r:<16}{n:>9}{100.0 * n / tot:>7.1f}%{chg:>6.1f}%{new:>6.1f}%"
              f"{pooled_bank.get(r, 0):>8}{pooled_wipe.get(r, 0):>7}{per_k:>9.2f}")

    worst = sorted(rows, key=lambda r: r[1]["honest"])[:8]
    print("\n  Lowest honest score:")
    for gid, sc, res in worst:
        print(f"    {gid:<7} honest={sc['honest']:.4f}  "
              f"levels={sc['real_levels']}/{res['win_levels']}  {res['actions']} acts")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(dump, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
