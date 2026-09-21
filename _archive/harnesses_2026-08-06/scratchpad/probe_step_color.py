"""Evidence probe for the step_color DSL op (added 2026-07-29).

The claim under test is NOT "step_color helps game X". It is a claim about the
DSL's expressiveness:

  EnumerativeSynthesizer.synthesize() drops every UNCHANGED transition before
  fitting. A blocked keypress produces exactly that -- an unchanged transition.
  So translate_color gets certified on the presses that moved something, and is
  then trusted during unscored lookahead in states where the real game refuses
  to move. step_color fits the same moving samples AND predicts the blocked
  press correctly.

This probe measures that on REAL transitions, scoring each hypothesis over ALL
transitions of an action -- including the unchanged ones the synthesizer never
sees. Env stepping only, no agent thinking, so it costs seconds.

Run:  PYTHONIOENCODING=utf-8 python scratchpad/probe_step_color.py [n_steps]
"""
import os
import sys
import random
import logging

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.dirname(HERE)
sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, os.path.join(AGENT_DIR, "eval"))

from arc_agi.local_wrapper import LocalEnvironmentWrapper
from arc_agi.models import EnvironmentInfo
from local_eval import _class_name_from_file, _discover_real_games
from my_agent import StateEncoder, MyAgent, GridDSL as D

E = StateEncoder()
_nums = [a for a in sys.argv[1:] if a.isdigit()]
N_STEPS = int(_nums[0]) if _nums else 150
DIRS = ("down", "up", "left", "right")
STEP_OF = {"down": (1, 0), "up": (-1, 0), "left": (0, -1), "right": (0, 1)}


def collect(game_file, n_steps, seed=0):
    """Return (game_id, [(prev_grid, action_name, next_grid), ...])."""
    from arcengine import GameAction
    gid = os.path.splitext(os.path.basename(game_file))[0]
    info = EnvironmentInfo(game_id=gid, class_name=_class_name_from_file(game_file),
                           local_dir=os.path.dirname(os.path.abspath(game_file)))
    lg = logging.getLogger("probe")
    lg.setLevel(logging.ERROR)
    w = LocalEnvironmentWrapper(info, lg, scorecard_id="probe", seed=seed)
    frame = w.observation_space
    if frame is None:
        return gid, []
    rng = random.Random(seed)
    avail = list(getattr(frame, "available_actions", None) or [])
    out = []
    for _ in range(n_steps):
        prev = MyAgent._grid_from(frame)
        act = rng.choice(avail) if avail else GameAction.ACTION1
        name = getattr(act, "name", str(act))
        data = {}
        if name in ("ACTION6", "ACTION7"):
            data = {"x": rng.randrange(0, 64), "y": rng.randrange(0, 64)}
        try:
            resp = w.step(act, data=data)
        except Exception:
            break
        if resp is None:
            break
        nxt = MyAgent._grid_from(resp)
        if prev is not None and nxt is not None and prev.shape == nxt.shape:
            out.append((prev, name, nxt))
        frame = resp
    return gid, out


def score(fn, args, samples):
    """Fraction of transitions this hypothesis predicts EXACTLY."""
    hit = 0
    for prev, nxt in samples:
        bg = E.detect_background(prev)
        try:
            pred = fn(prev.copy(), bg, **args)
        except Exception:
            continue
        if isinstance(pred, np.ndarray) and pred.shape == nxt.shape and np.array_equal(pred, nxt):
            hit += 1
    return hit / len(samples) if samples else 0.0


def mismatch(fn, args, samples):
    """Per-sample count of pixels where the prediction differs from reality."""
    out = []
    for prev, nxt in samples:
        bg = E.detect_background(prev)
        try:
            pred = fn(prev.copy(), bg, **args)
        except Exception:
            continue
        if isinstance(pred, np.ndarray) and pred.shape == nxt.shape:
            out.append(int((pred != nxt).sum()))
    return out


def near_miss_report():
    """The follow-up question to a flat 0.00: is the DSL WRONG, or merely
    outvoted by a handful of always-changing HUD pixels? Exact whole-grid
    equality cannot tell those apart, but the mismatch SIZE can. A best-case
    miss of a few pixels means the fitting criterion is the bug; a miss of
    hundreds means the vocabulary genuinely lacks the game's physics."""
    print(f"\n{'game':6} {'act':4} {'n':>4} {'moved_px':>9} {'best_miss_px':>13} {'<=8px':>6}  hypothesis")
    print("-" * 78)
    for gf in _discover_real_games():
        gid, trans = collect(gf, 60)
        if not trans:
            continue
        by_act = {}
        for prev, act, nxt in trans:
            by_act.setdefault(act, []).append((prev, nxt))
        for act, samples in sorted(by_act.items()):
            changed = [(p, n) for p, n in samples if not np.array_equal(p, n)]
            if len(changed) < 3:
                continue
            cols = set()
            for p, n in changed:
                d = p != n
                cols.update(int(x) for x in np.unique(p[d]))
                cols.update(int(x) for x in np.unique(n[d]))
            bg0 = E.detect_background(changed[0][0])
            cols.discard(bg0)
            if not cols or len(cols) > 12:
                continue
            moved = float(np.median([int((p != n).sum()) for p, n in changed]))
            best = (1e9, None)
            for col in sorted(cols):
                for d in DIRS:
                    for nm, fn, args in (("step", D.step_color, {"color": col, "direction": d}),
                                         ("trans", D.translate_color,
                                          {"color": col, "dr": STEP_OF[d][0], "dc": STEP_OF[d][1]})):
                        ms = mismatch(fn, args, changed)
                        if ms:
                            med = float(np.median(ms))
                            if med < best[0]:
                                best = (med, f"{nm}({col},{d})", ms)
            if best[1] is None:
                continue
            tight = sum(1 for m in best[2] if m <= 8)
            print(f"{gid:6} {act[-1]:4} {len(changed):4d} {moved:9.0f} {best[0]:13.0f} "
                  f"{tight:3d}/{len(best[2]):<3d} {best[1]}")


def main():
    print(f"{'game':6} {'action':9} {'n':>4} {'blocked':>8} "
          f"{'step_color':>11} {'translate':>10}  verdict")
    print("-" * 68)
    tot_step = tot_tr = tot_n = 0
    for gf in _discover_real_games():
        gid, trans = collect(gf, N_STEPS)
        if not trans:
            continue
        by_act = {}
        for prev, act, nxt in trans:
            by_act.setdefault(act, []).append((prev, nxt))
        for act, samples in sorted(by_act.items()):
            changed = [(p, n) for p, n in samples if not np.array_equal(p, n)]
            # Diagnostic, not a filter: how much of this action's evidence is a
            # no-op, and does ANY object-scoped hypothesis fit the moving part
            # exactly? If nothing ever fits, the DSL is not the binding
            # constraint on the world model and step_color cannot show up here.
            if len(changed) < 3:
                continue
            # Candidate colors: the ones observed changing (what the synthesizer uses).
            cols = set()
            for p, n in changed:
                d = p != n
                cols.update(int(x) for x in np.unique(p[d]))
                cols.update(int(x) for x in np.unique(n[d]))
            bg0 = E.detect_background(changed[0][0])
            cols.discard(bg0)
            if not cols or len(cols) > 12:
                continue
            best_step = best_tr = 0.0
            for col in cols:
                for d in DIRS:
                    # Certify on CHANGED samples only (as synthesize() does), then
                    # score the certified rule over ALL samples.
                    if score(D.step_color, {"color": col, "direction": d}, changed) == 1.0:
                        best_step = max(best_step, score(D.step_color, {"color": col, "direction": d}, samples))
                    dr, dc = STEP_OF[d]
                    if score(D.translate_color, {"color": col, "dr": dr, "dc": dc}, changed) == 1.0:
                        best_tr = max(best_tr, score(D.translate_color, {"color": col, "dr": dr, "dc": dc}, samples))
            blocked = len(samples) - len(changed)
            if best_step == 0.0 and best_tr == 0.0:
                verdict = "NO object rule fits at all"
            else:
                verdict = ("step_color better" if best_step > best_tr + 1e-9 else
                           "translate better" if best_tr > best_step + 1e-9 else "tie")
            print(f"{gid:6} {act:9} {len(samples):4d} {blocked:8d} "
                  f"{best_step:11.2f} {best_tr:10.2f}  {verdict}")
            tot_step += best_step * len(samples)
            tot_tr += best_tr * len(samples)
            tot_n += len(samples)
    if tot_n:
        print("-" * 68)
        print(f"weighted accuracy over {tot_n} real transitions: "
              f"step_color {tot_step / tot_n:.3f}   translate_color {tot_tr / tot_n:.3f}")
    else:
        print("no action had both moving and blocked evidence in this sample")


def stride_report():
    """Is the synthesizer's hardcoded displacement set the binding constraint?

    _candidate_ops enumerates translate_color over dr,dc in {+-1, +-2} pixels
    only. If these games draw objects as k x k sprites on a lattice, one move is
    k pixels, and NO candidate it can propose is even the right shape. Here the
    axis-aligned displacement is searched out to +-16 px: if the miss collapses
    at some stride k > 2, the vocabulary was never the problem -- the candidate
    set was."""
    print(f"\n{'game':6} {'act':4} {'n':>4} {'miss@<=2px':>11} {'miss@best':>10} "
          f"{'best_shift':>11}  {'verdict'}")
    print("-" * 76)
    shifts = [(d, 0) for d in range(-16, 17) if d] + [(0, d) for d in range(-16, 17) if d]
    for gf in _discover_real_games():
        gid, trans = collect(gf, 60)
        if not trans:
            continue
        by_act = {}
        for prev, act, nxt in trans:
            by_act.setdefault(act, []).append((prev, nxt))
        for act, samples in sorted(by_act.items()):
            changed = [(p, n) for p, n in samples if not np.array_equal(p, n)]
            if len(changed) < 3:
                continue
            cols = set()
            for p, n in changed:
                d = p != n
                cols.update(int(x) for x in np.unique(p[d]))
                cols.update(int(x) for x in np.unique(n[d]))
            bg0 = E.detect_background(changed[0][0])
            cols.discard(bg0)
            if not cols or len(cols) > 12:
                continue
            small = (1e9, None)     # best miss using ONLY the shifts the synthesizer can propose
            wide = (1e9, None)      # best miss over the full axis-aligned range
            for col in sorted(cols):
                for dr, dc in shifts:
                    ms = mismatch(D.translate_color, {"color": col, "dr": dr, "dc": dc}, changed)
                    if not ms:
                        continue
                    med = float(np.median(ms))
                    if max(abs(dr), abs(dc)) <= 2 and med < small[0]:
                        small = (med, f"({col},{dr},{dc})")
                    if med < wide[0]:
                        wide = (med, f"({col},{dr},{dc})")
            if wide[1] is None:
                continue
            gain = small[0] - wide[0]
            verdict = "WIDER SHIFT WINS" if gain > 0.5 else "no gain from stride"
            print(f"{gid:6} {act[-1]:4} {len(changed):4d} {small[0]:11.0f} {wide[0]:10.0f} "
                  f"{wide[1]:>11}  {verdict}")


def synth_report(n_steps=120):
    """End-to-end: does EnumerativeSynthesizer actually recover a model on real
    games, and how accurate is it on HELD-OUT transitions? This is the number
    the DSL work exists to move -- before the data-derived stride it was zero
    models on all 25 games."""
    import time as _time
    from my_agent import EnumerativeSynthesizer, Transition
    syn = EnumerativeSynthesizer(E)
    print(f"\n{'game':6} {'trans':>6} {'model':>6} {'train_acc':>10} {'heldout_acc':>12}  rule")
    print("-" * 84)
    got = 0
    tot = 0
    for gf in _discover_real_games():
        gid, trans = collect(gf, n_steps)
        if len(trans) < 12:
            continue
        tot += 1
        cut = int(len(trans) * 0.6)
        train = [Transition(p, a, n, 0.0, "", _time.time()) for p, a, n in trans[:cut]]
        held = [(p, a, n) for p, a, n in trans[cut:]]
        try:
            wm = syn.synthesize(train)
        except Exception as e:
            print(f"{gid:6} {len(trans):6d}  ERROR {type(e).__name__}: {e}")
            continue
        if wm is None:
            print(f"{gid:6} {len(trans):6d} {'-':>6}")
            continue
        got += 1

        def acc(rows):
            if not rows:
                return float("nan")
            hit = sum(1 for p, a, n in rows
                      if np.array_equal(wm.fn(p.copy(), a), n))
            return hit / len(rows)

        tr_acc = acc([(p, a, n) for p, a, n in trans[:cut]])
        rule = wm.hypothesis.replace("Enumerated: ", "")
        print(f"{gid:6} {len(trans):6d} {'yes':>6} {tr_acc:10.2f} {acc(held):12.2f}  {rule[:40]}")
    print("-" * 84)
    print(f"models recovered: {got}/{tot} games")


def residual_report(n_steps=60):
    """With the right stride the best rule still misses ~10-14 px. Are those
    pixels CONCENTRATED in a few rows (a HUD/counter that changes every step,
    which exact-equality fitting can never tolerate) or SPREAD across the play
    field (physics the DSL genuinely does not model)? The answer decides whether
    the remaining work belongs to the vocabulary or to the fitting criterion."""
    print(f"\n{'game':6} {'act':4} {'miss':>5} {'rows_hit':>9} {'row_span':>9} "
          f"{'top_rows_share':>15}  where")
    print("-" * 74)
    shifts = [(d, 0) for d in range(-8, 9) if d] + [(0, d) for d in range(-8, 9) if d]
    for gf in _discover_real_games():
        gid, trans = collect(gf, n_steps)
        if not trans:
            continue
        by_act = {}
        for prev, act, nxt in trans:
            by_act.setdefault(act, []).append((prev, nxt))
        for act, samples in sorted(by_act.items()):
            changed = [(p, n) for p, n in samples if not np.array_equal(p, n)]
            if len(changed) < 3:
                continue
            cols = set()
            for p, n in changed:
                d = p != n
                cols.update(int(x) for x in np.unique(p[d]))
                cols.update(int(x) for x in np.unique(n[d]))
            cols.discard(E.detect_background(changed[0][0]))
            if not cols or len(cols) > 12:
                continue
            best = (1e9, None, None)
            for col in sorted(cols):
                for dr, dc in shifts:
                    ms = mismatch(D.translate_color, {"color": col, "dr": dr, "dc": dc}, changed)
                    if ms and float(np.median(ms)) < best[0]:
                        best = (float(np.median(ms)), col, (dr, dc))
            if best[1] is None or best[0] == 0:
                continue
            col, (dr, dc) = best[1], best[2]
            acc_rows = np.zeros(changed[0][0].shape[0], dtype=int)
            for p, n in changed:
                bg = E.detect_background(p)
                pred = D.translate_color(p.copy(), bg, color=col, dr=dr, dc=dc)
                if pred.shape != n.shape:
                    continue
                rr = np.nonzero((pred != n).any(axis=1))[0]
                acc_rows[rr] += 1
            hit = np.nonzero(acc_rows)[0]
            if len(hit) == 0:
                continue
            span = int(hit.max() - hit.min() + 1)
            order = np.argsort(acc_rows)[::-1]
            top3 = float(acc_rows[order[:3]].sum()) / max(1, acc_rows.sum())
            where = "HUD-like (few rows)" if len(hit) <= 4 else (
                "concentrated" if top3 >= 0.6 else "spread across field")
            print(f"{gid:6} {act[-1]:4} {best[0]:5.0f} {len(hit):9d} {span:9d} "
                  f"{top3:15.2f}  {where}  rows={list(hit[:6])}")


if __name__ == "__main__":
    if "--residual" in sys.argv:
        residual_report()
    elif "--synth" in sys.argv:
        synth_report()
    else:
        main()
        near_miss_report()
        stride_report()
