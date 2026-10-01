"""Instrument looks_mechanical() on keydoor: which condition trips, and when.

Run from ARC_AGI EXPERMENTATIONS/:
    PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 python -u scratchpad/dbg_mech_keydoor.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from local_eval import run_game, GAMES_DIR, default_agent_factory  # noqa: E402
import my_agent as ma  # noqa: E402


def factory(game_id):
    agent = default_agent_factory(game_id)
    rp = agent.rplanner
    orig_update = rp.update
    state = {"step": 0, "mech": False}

    def update(prev, action, nxt, encoder, reward=0.0):
        em0, uc0, dc0 = rp.explained_moves, rp.unexplained_changes, rp.disp_contradictions
        orig_update(prev, action, nxt, encoder, reward)
        state["step"] += 1
        if rp.disp_contradictions > dc0:
            print(f"[{state['step']:4d}] DCON+1 -> {rp.disp_contradictions} "
                  f"action={action} disp_now={rp.action_disp.get(ma.base_action(action))}")
        if rp.unexplained_changes > uc0:
            print(f"[{state['step']:4d}] UNEXPL+1 -> {rp.unexplained_changes} action={action}")
        mech = rp.looks_mechanical()
        if mech != state["mech"]:
            total = rp.explained_moves + rp.unexplained_changes
            print(f"[{state['step']:4d}] MECH FLIP -> {mech}  "
                  f"(em={rp.explained_moves} uc={rp.unexplained_changes} "
                  f"dcon={rp.disp_contradictions} total={total})")
            state["mech"] = mech

    rp.update = update
    return agent


if __name__ == "__main__":
    gid = sys.argv[1] if len(sys.argv) > 1 else "keydoor"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    gf = os.path.join(GAMES_DIR, gid + ".py")
    if not os.path.exists(gf):
        import glob as _g
        hits = _g.glob(os.path.join(ROOT, "eval", "real_games", gid, "*", gid + ".py"))
        gf = hits[0]
    r = run_game(gf, factory, seed=0, max_actions=steps)
    print(r)
