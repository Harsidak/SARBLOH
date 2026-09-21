"""Diagnostic: run MyAgent on one real game, print who drives + graph health."""
import os, sys, glob, random, logging, time
import numpy as np

HERE = r"C:\Users\Banwa\Desktop\IWT\CODING\PROJECTS\Kaggle\ARC_AGI EXPERMENTATIONS"
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "eval"))
from arc_agi.local_wrapper import LocalEnvironmentWrapper
from arc_agi.models import EnvironmentInfo
from arcengine import ARCBaseGame
import importlib.util

gid = sys.argv[1] if len(sys.argv) > 1 else "tr87"
budget = int(sys.argv[2]) if len(sys.argv) > 2 else 400

gf = glob.glob(os.path.join(HERE, "eval", "real_games", gid, "*", f"{gid}.py"))[0]
spec = importlib.util.spec_from_file_location("_g_" + gid, gf)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
cls = [o for n, o in vars(mod).items() if isinstance(o, type)
       and issubclass(o, ARCBaseGame) and o is not ARCBaseGame][0].__name__

seed = 0
os.environ["ARC_AGENT_SEED"] = str(seed)
random.seed(seed); np.random.seed(seed)
lg = logging.getLogger("dbg"); lg.setLevel(logging.ERROR)
info = EnvironmentInfo(game_id=gid, class_name=cls, local_dir=os.path.dirname(gf))
wrapper = LocalEnvironmentWrapper(info, lg, scorecard_id="local", seed=seed)
frame0 = wrapper.observation_space

import my_agent as M
agent = M.MyAgent(game_id=gid)

# instrument: count which source produced each action
stats = {"graph": 0, "graph_none": 0}
orig_ge = M.MyAgent._graph_explore
def ge(self, grid, valid):
    r = orig_ge(self, grid, valid)
    stats["graph" if r is not None else "graph_none"] += 1
    return r
M.MyAgent._graph_explore = ge

frames = [frame0]
for step in range(budget):
    if agent.is_done(frames, frames[-1]):
        break
    action = agent.choose_action(frames, frames[-1])
    data = {}
    if hasattr(action, "is_complex") and action.is_complex():
        data = {"x": int(action.action_data.x), "y": int(action.action_data.y)}
    resp = wrapper.step(action, data=data)
    if resp is None:
        print("wrapper returned None"); break
    frames.append(resp)
    agent.action_counter = getattr(agent, "action_counter", 0) + 1
    if (step + 1) % 100 == 0:
        rp = agent.rplanner
        sgr = agent.sgraph
        n_edges = sum(len(v) for v in sgr.adj.values())
        n_unt = sum(len(v) for v in sgr.untested.values())
        mask_n = int(sgr._mask.sum()) if sgr._mask is not None else 0
        print(f"step {step+1:4d} lvl={agent.last_score} "
              f"ready={rp.ready(['ACTION1','ACTION2','ACTION3','ACTION4'])} "
              f"mech={rp.looks_mechanical()} expl={rp.explained_moves} "
              f"unex={rp.unexplained_changes} disp={dict(rp.action_disp)} "
              f"| states={len(sgr.untested)} edges={n_edges} untested={n_unt} "
              f"mask={mask_n}px | graph_hits={stats['graph']} misses={stats['graph_none']}")
print("final level:", agent.last_score)
sgr = agent.sgraph
n = max(1, sgr._n)
print(f"\n_n={sgr._n}  overall row rates >0.3:",
      {i: round(float(r / n), 2) for i, r in enumerate(sgr._rowchg) if r / n > 0.3})
print("overall col rates >0.3:",
      {i: round(float(r / n), 2) for i, r in enumerate(sgr._colchg) if r / n > 0.3})
print("overall max cell rate:", round(float(sgr._change.max() / n), 2))
for a, (cc, rc, colc, na) in sorted(sgr._act.items()):
    rows = {i: round(float(r / na), 2) for i, r in enumerate(rc) if r / na > 0.3}
    print(f"  {a}: n={na} maxcell={round(float(cc.max()/na),2)} rows>{0.3}: {rows}")
