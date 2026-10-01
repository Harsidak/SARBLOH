"""Local evaluation harness for MyAgent against ARCBaseGame environments.

Runs the SAME loop the Kaggle framework uses (choose_action / is_done /
action_counter <= MAX_ACTIONS), but against local games loaded by
arc_agi.LocalEnvironmentWrapper. No network, no submission — iterate here
thousands of times instead of once / 17h.

Usage:
    python eval/local_eval.py                 # run all games in eval/games/
    python eval/local_eval.py reachgoal       # run one game
    python eval/local_eval.py --verbose reachgoal
    python eval/local_eval.py --real          # run the REAL 25 competition games
    python eval/local_eval.py --real ls20     # one real game

Each game dir entry is a `<class_name>.py` file (lowercased) defining an
ARCBaseGame subclass. The competition's real games drop in the same way if
their sources become available.
"""
import os
import sys
import glob
import time
import random
import logging
import argparse

import numpy as np

# Make `my_agent` importable and silence its optional-dep chatter.
HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.dirname(HERE)
sys.path.insert(0, AGENT_DIR)
GAMES_DIR = os.path.join(HERE, "games")
REAL_GAMES_DIR = os.path.join(HERE, "real_games")

from arc_agi.local_wrapper import LocalEnvironmentWrapper
from arc_agi.models import EnvironmentInfo


def _quiet_logger() -> logging.Logger:
    lg = logging.getLogger("local_eval")
    lg.setLevel(logging.ERROR)
    if not lg.handlers:
        lg.addHandler(logging.StreamHandler(sys.stderr))
    return lg


def _class_name_from_file(path: str) -> str:
    """eval/games/reachgoal.py -> the class the wrapper will look for.

    The wrapper tries `<class_name>.lower().py` then `<class_name>.py`, so we can
    hand it the Capitalized name and it finds the lowercase file. We recover the
    class name by importing the module and taking the single ARCBaseGame subclass.
    """
    import importlib.util
    from arcengine import ARCBaseGame

    spec = importlib.util.spec_from_file_location("_probe_" + os.path.basename(path), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, obj in vars(mod).items():
        if isinstance(obj, type) and issubclass(obj, ARCBaseGame) and obj is not ARCBaseGame:
            return name
    raise RuntimeError(f"No ARCBaseGame subclass found in {path}")


def run_game(game_file: str, agent_factory, seed: int = 0,
             max_actions: int = 1500, verbose: bool = False) -> dict:
    """Play one game to completion (or budget/stall) and return a result dict."""
    game_id = os.path.splitext(os.path.basename(game_file))[0]
    class_name = _class_name_from_file(game_file)
    info = EnvironmentInfo(game_id=game_id, class_name=class_name,
                           local_dir=os.path.dirname(os.path.abspath(game_file)))
    logger = _quiet_logger()

    wrapper = LocalEnvironmentWrapper(info, logger, scorecard_id="local", seed=seed)
    frame0 = wrapper.observation_space
    if frame0 is None:
        return {"game": game_id, "error": "failed to load/reset game"}

    win_levels = int(getattr(frame0, "win_levels", 0) or 0)
    # Pin the AGENT's exploration RNG so a run is reproducible (the wrapper seed above
    # only controls the game, not the agent's clicks). MyAgent reads ARC_AGENT_SEED.
    os.environ["ARC_AGENT_SEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    agent = agent_factory(game_id)
    frames = [frame0]

    t0 = time.time()
    stall = None
    while not agent.is_done(frames, frames[-1]) and agent.action_counter <= max_actions:
        latest = frames[-1]
        try:
            action = agent.choose_action(frames, latest)
        except Exception as e:
            stall = f"choose_action raised: {e}"
            break
        data = {}
        if hasattr(action, "is_complex") and action.is_complex():
            data = {"x": int(action.action_data.x), "y": int(action.action_data.y)}
        resp = wrapper.step(action, data=data)
        if resp is None:
            stall = "wrapper.step returned None"
            break
        frames.append(resp)
        agent.action_counter += 1
        if verbose and agent.action_counter % 100 == 0:
            print(f"    [{game_id}] {agent.action_counter} actions, "
                  f"levels={getattr(resp, 'levels_completed', 0)}")

    last = frames[-1]
    # Competition semantics: the scorecard takes max(levels_completed) over runs
    # (arc_agi.scorecard EnvironmentScoreList), so a level banked before a
    # GAME_OVER->RESET still counts even though the final frame reads 0.
    levels = max(int(getattr(f, "levels_completed", 0) or 0) for f in frames)
    state = getattr(last, "state", None)
    state_name = getattr(state, "name", str(state))
    return {
        "game": game_id,
        "levels_completed": levels,
        "win_levels": win_levels,
        "frac": (levels / win_levels) if win_levels else 0.0,
        "actions": agent.action_counter,
        "state": state_name,
        "won": state_name == "WIN",
        "secs": round(time.time() - t0, 1),
        "stall": stall,
    }


def default_agent_factory(game_id: str):
    from my_agent import MyAgent
    return MyAgent(game_id=game_id)


def _discover_real_games():
    """real_games/<id>/<hash>/<id>.py — official competition game sources."""
    files = []
    for gdir in sorted(glob.glob(os.path.join(REAL_GAMES_DIR, "*"))):
        if not os.path.isdir(gdir):
            continue
        gid = os.path.basename(gdir)
        hits = glob.glob(os.path.join(gdir, "*", f"{gid}.py"))
        if hits:
            files.append(sorted(hits)[0])
    return files


def _human_baselines():
    """game_id -> total human baseline actions, from games_index.json."""
    import json
    idx = os.path.join(REAL_GAMES_DIR, "games_index.json")
    if not os.path.exists(idx):
        return {}
    with open(idx, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {g["id"]: sum(g.get("baseline_steps", [])) for g in data}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*", help="game ids to run (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-actions", type=int, default=1500)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--real", action="store_true",
                    help="run the real competition games from eval/real_games/")
    args = ap.parse_args()

    if args.real:
        all_files = _discover_real_games()
    else:
        all_files = sorted(glob.glob(os.path.join(GAMES_DIR, "*.py")))
    all_files = [f for f in all_files if not os.path.basename(f).startswith("_")]
    if args.games:
        wanted = set(args.games)
        all_files = [f for f in all_files
                     if os.path.splitext(os.path.basename(f))[0] in wanted]
    if not all_files:
        print("No games found."); return

    baselines = _human_baselines() if args.real else {}
    print(f"Running {len(all_files)} game(s) through MyAgent...\n")
    results = []
    for gf in all_files:
        r = run_game(gf, default_agent_factory, seed=args.seed,
                     max_actions=args.max_actions, verbose=args.verbose)
        results.append(r)
        if "error" in r:
            print(f"  {r['game']:<14} ERROR: {r['error']}")
        else:
            tag = "WON " if r["won"] else "----"
            stall = f"  ({r['stall']})" if r.get("stall") else ""
            hb = baselines.get(r["game"])
            eff = f"  human={hb}" if hb else ""
            print(f"  {tag} {r['game']:<14} levels {r['levels_completed']}/{r['win_levels']}"
                  f"  frac={r['frac']:.2f}  actions={r['actions']}{eff}  {r['secs']}s{stall}",
                  flush=True)

    scored = [r for r in results if "error" not in r]
    if scored:
        avg = sum(r["frac"] for r in scored) / len(scored)
        wins = sum(1 for r in scored if r["won"])
        print(f"\n  AVG level-completion fraction: {avg:.3f}   "
              f"({wins}/{len(scored)} games fully won)")
        print("  NOTE: level-completion fraction is NOT the competition metric and"
              " never was.\n        It ignores action efficiency, and it ignores the"
              " score wipes that\n        drive the leaderboard number. For a score,"
              " run eval/rhae_eval.py.")


if __name__ == "__main__":
    main()
