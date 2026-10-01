"""Probe lp85's reactive structure: how many distinct button cells exist, what
clicking each does, and how many distinct frame-states are reachable. Drives the
env directly (no agent). Run with anaconda python + PYTHONIOENCODING=utf-8."""
import os, sys, glob, hashlib
import numpy as np
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base); sys.path.insert(0, os.path.join(base, "eval"))
import local_eval as le
from arc_agi.local_wrapper import LocalEnvironmentWrapper
from arc_agi.models import EnvironmentInfo
from my_agent import GameAction, GameState

gid = "lp85"
gf = glob.glob(os.path.join(base, "eval", "real_games", gid, "*", gid + ".py"))[0]
info = EnvironmentInfo(game_id=gid, class_name=le._class_name_from_file(gf),
                       local_dir=os.path.dirname(gf))
wrap = LocalEnvironmentWrapper(info, le._quiet_logger(), scorecard_id="local", seed=0)

def grid_of(f):
    g = f.frame
    return np.array(g[-1] if g else [[0]], dtype=np.int32)

def h(g):
    return hashlib.md5(g.tobytes()).hexdigest()[:8]

def click(r, c):
    a = GameAction.ACTION6; a.set_data({"x": c, "y": r})
    return wrap.step(a, data={"x": c, "y": r})

def reset():
    return wrap.step(GameAction.RESET, data={})

fr = reset()
if getattr(fr.state, "name", "") == "NOT_PLAYED":
    fr = reset()
g0 = grid_of(fr)
print("initial:", g0.shape, "hash", h(g0), "state", fr.state.name,
      "levels", getattr(fr, "levels_completed", 0), "colors", sorted(set(g0.flatten().tolist())))

# Sweep a lattice, note exact cells whose click changes the frame (from initial each time).
reactive = []
for r in range(2, 64, 4):
    for c in range(2, 64, 4):
        reset()
        before = grid_of(wrap.observation_space)
        fr = click(r, c)
        after = grid_of(fr)
        st = getattr(fr.state, "name", "")
        if st == "GAME_OVER":
            reset()
            continue
        if not np.array_equal(before, after):
            reactive.append((r, c, h(after)))
print(f"\nreactive cells (of {16*16} lattice): {len(reactive)}")
for r, c, hh in reactive:
    print(f"  ({r:2d},{c:2d}) -> {hh}")

# Group reactive cells by the state they produce (distinct buttons = distinct effects).
by_state = {}
for r, c, hh in reactive:
    by_state.setdefault(hh, []).append((r, c))
print(f"\ndistinct 1-click states: {len(by_state)}")
for hh, cells in by_state.items():
    print(f"  {hh}: {len(cells)} cells e.g. {cells[:3]}")

# From initial, pick one cell per distinct button and see repeated-click behaviour (slide vs toggle).
print("\nrepeated-click behaviour per button:")
reps = {}
for hh, cells in by_state.items():
    r, c = cells[0]
    reset()
    seq = [h(grid_of(wrap.observation_space))]
    for _ in range(8):
        fr = click(r, c)
        if getattr(fr.state, "name", "") == "GAME_OVER":
            seq.append("DEAD"); break
        seq.append(h(grid_of(fr)))
    print(f"  button@({r},{c}): {' '.join(seq)}")
