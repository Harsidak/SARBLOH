"""Probe: is the masked state graph deterministic? Post-hoc conflict count over the
raw transition log using the FINAL mask (avoids moving-mask noise)."""
import os, sys, glob, random, logging
import numpy as np

HERE = r"C:\Users\Banwa\Desktop\IWT\CODING\PROJECTS\Kaggle\ARC_AGI EXPERMENTATIONS"
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "eval"))
from arc_agi.local_wrapper import LocalEnvironmentWrapper
from arc_agi.models import EnvironmentInfo
from arcengine import ARCBaseGame
import importlib.util

gid = sys.argv[1] if len(sys.argv) > 1 else "tu93"
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
        break
    frames.append(resp)
    agent.action_counter = getattr(agent, "action_counter", 0) + 1

sgr = agent.sgraph
mask = sgr._mask


def h(g):
    if mask is not None:
        g = g.copy(); g[mask] = -1
    return hash(g.tobytes())


seen = {}     # (prev_hash, action) -> {next_hash: count}
for prev, a, nxt in sgr._trans:
    seen.setdefault((h(prev), a), {}).setdefault(h(nxt), 0)
    seen[(h(prev), a)][h(nxt)] += 1
confl = {k: v for k, v in seen.items() if len(v) > 1}
print(f"{gid}: {len(confl)} nondeterministic (state,action) pairs "
      f"of {len(seen)} recorded; log={len(sgr._trans)} lvl={agent.last_score}")
for (hp, a), outs in list(confl.items())[:8]:
    print(f"  action={a} outcomes={sorted(outs.values(), reverse=True)}")
