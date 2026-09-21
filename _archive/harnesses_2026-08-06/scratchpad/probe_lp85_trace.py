import os, sys, random, numpy as np, glob
base=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base)
sys.path.insert(0, os.path.join(base,"eval"))
import local_eval as le
from arc_agi.local_wrapper import LocalEnvironmentWrapper
from arc_agi.models import EnvironmentInfo
import my_agent as MA
gf=glob.glob(os.path.join(base,"eval","real_games","lp85","*","lp85.py"))[0]
seed=0; random.seed(seed); np.random.seed(seed)
info=EnvironmentInfo(game_id="lp85", class_name=le._class_name_from_file(gf), local_dir=os.path.dirname(gf))
wrap=LocalEnvironmentWrapper(info, le._quiet_logger(), scorecard_id="local", seed=seed)
f0=wrap.observation_space
ag=MA.MyAgent("lp85")
frames=[f0]; prev=None
while not ag.is_done(frames,frames[-1]) and ag.action_counter<=1500:
    a=ag.choose_action(frames,frames[-1])
    data={}
    if hasattr(a,"is_complex") and a.is_complex(): data={"x":int(a.action_data.x),"y":int(a.action_data.y)}
    resp=wrap.step(a,data=data)
    if resp is None: break
    frames.append(resp); ag.action_counter+=1
    lc=getattr(resp,"levels_completed",0); st=getattr(getattr(resp,"state",None),"name","?"); fr=getattr(resp,"full_reset",False)
    cur=(lc,st,fr)
    if cur!=prev:
        print(f"  a={ag.action_counter:4d} levels={lc} state={st} full_reset={fr} last={ag.last_action_str} searching={ag.cplanner.searching} nbtn={len(ag.cplanner._buttons)}")
        prev=cur
print("FINAL levels=",getattr(frames[-1],"levels_completed",0),"actions=",ag.action_counter, "searching=",ag.cplanner.searching)
