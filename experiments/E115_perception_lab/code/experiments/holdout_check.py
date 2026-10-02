"""
Held-out generalisation check for X7 (roles) and X5 (keyboard foresight) on the 23 official games that
were NOT used to design the experiments. Ground truth comes from the engine only:
  YOU      : scene objects whose cells change under ACTION1..4 (minus HUD)       -> is X7's YOU among them?
  CLICK    : object 'responds' if clicking its centre changes any non-HUD cell    -> precision of X7 HANDLE
  GAUGE    : does the bar X7 calls MOVES LEFT change after one action?
  FORESIGHT: X5 predicted self position vs engine for each arrow action
Screenshots are synthesised from the reset frame (7 px cells) and go through the normal loader.
"""
from __future__ import annotations

import io
import logging
import os
import sys
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "X3_routines", "X5_foresight", "X7_ledger"):
    sys.path.insert(0, os.path.join(HERE, sub))
logging.disable(logging.INFO)
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402
from scene import build_scene, load_grid, to_image  # noqa: E402
from common_out import is_hud  # noqa: E402
import x7_ledger, x5_foresight  # noqa: E402

GAMES = os.environ.get("ARC_GAMES", "/home/claude/repos/arc-prize-2026-arc-agi-3/arc_games")
OUT = os.path.join(HERE, "..", "results", "holdout")
os.makedirs(OUT, exist_ok=True)
arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES)


def act(a, data=None):
    return getattr(GameAction, f"ACTION{a}")


rows = []
for e in arc.get_environments():
    gid = e.game_id.split("-")[0]
    if gid in ("ls20", "r11l"):
        continue
    env = arc.make(gid); o = env.step(GameAction.RESET)
    f0 = np.array(o.frame)[-1]
    acts = [a.value if hasattr(a, "value") else int(a) for a in o.available_actions]
    png = os.path.join(OUT, f"{gid}.png")
    to_image(f0, cell=7).save(png)
    g, notes = load_grid(png)
    grid_ok = float((g == f0).mean())
    sc = build_scene(g, notes)
    with redirect_stdout(io.StringIO()):
        x7_ledger.run(png, os.path.join(OUT, f"{gid}_x7.png"), os.path.join(OUT, f"{gid}_x7.txt"), tuple(acts))
    ledger_txt = open(os.path.join(OUT, f"{gid}_x7.txt")).read()
    # rebuild X7 decisions from its text (top role per object line)
    tops = {}
    for line in ledger_txt.splitlines():
        if line.startswith("* "):
            loc = line.split(" @ rows ")[1].split(":")[0]  # "r0-r1, cols c0-c1"
            r0 = int(loc.split("-")[0]); c0 = int(loc.split("cols ")[1].split("-")[0])
            role = line.split(": ", 1)[1].split(" ")[0]
            if role == "MOVES": role = "MOVES LEFT"
            if role == "HANDLE": role = "HANDLE"
            if role == "CARRIED": role = "CARRIED"
            tops[(r0, c0)] = role
    objmap = {(ob.r0, ob.c0): ob for ob in sc.objects}
    rec = {"game": gid, "actions": acts, "grid_acc": grid_ok}
    hud_cells = {c for ob in sc.objects if is_hud(ob, sc) for c in ob.cells}
    # YOU
    keys = [a for a in acts if a in (1, 2, 3, 4)]
    if keys:
        moved = np.zeros(f0.shape, bool)
        for a in keys:
            env = arc.make(gid); env.step(GameAction.RESET)
            f1 = np.array(env.step(act(a)).frame)[-1]
            moved |= (f1 != f0)
        movers = {ob.id for ob in sc.objects if not is_hud(ob, sc) and any(moved[y, x] for y, x in ob.cells)}
        you = [objmap[k] for k, r in tops.items() if r == "YOU" and k in objmap]
        rec["you_pred"] = you[0].name if you else None
        rec["you_hit"] = (you[0].id in movers) if you else False
        rec["n_movers"] = len(movers)
        # X5 foresight
        with redirect_stdout(io.StringIO()):
            try:
                x5_foresight.run(png, os.path.join(OUT, f"{gid}_x5.png"), os.path.join(OUT, f"{gid}_x5.txt"), tuple(keys), gid)
                v = open(os.path.join(OUT, f"{gid}_x5_verify.txt")).read()
                rec["x5"] = v.splitlines()[-1]
            except Exception as ex:  # noqa: BLE001
                rec["x5"] = f"error {type(ex).__name__}"
    # CLICK
    if 6 in acts:
        resp, labelled = 0, 0
        responsive_ids = set()
        for ob in sc.objects:
            if is_hud(ob, sc) or ob.kind == "line":
                continue
            env = arc.make(gid); env.step(GameAction.RESET)
            fr = np.array(env.step(GameAction.ACTION6, data={"x": int(round(ob.cx)), "y": int(round(ob.cy))}).frame)[-1]
            ch = (fr != f0)
            if any(ch[y, x] for y in range(64) for x in range(64) if (y, x) not in hud_cells):
                responsive_ids.add(ob.id)
        handles = [objmap[k] for k, r in tops.items() if r in ("HANDLE",) and k in objmap]
        rec["click_resp"] = f"{len(responsive_ids)}/{sum(1 for ob in sc.objects if not is_hud(ob, sc) and ob.kind != 'line')} objects respond"
        rec["handle_prec"] = (f"{sum(h.id in responsive_ids for h in handles)}/{len(handles)}" if handles else "no HANDLE labels")
    # GAUGE
    gauges = [objmap[k] for k, r in tops.items() if r == "MOVES LEFT" and k in objmap]
    if gauges:
        env = arc.make(gid); env.step(GameAction.RESET)
        a0 = keys[0] if keys else None
        if a0:
            f1 = np.array(env.step(act(a0)).frame)[-1]
        else:
            f1 = np.array(env.step(GameAction.ACTION6, data={"x": 32, "y": 32}).frame)[-1]
        gb = gauges[0]
        rec["gauge_hit"] = bool(any(f1[y, x] != f0[y, x] for y, x in gb.cells))
    rows.append(rec)
    print(rec, flush=True)

import json  # noqa: E402
json.dump(rows, open(os.path.join(OUT, "holdout_results.json"), "w"), indent=1)
