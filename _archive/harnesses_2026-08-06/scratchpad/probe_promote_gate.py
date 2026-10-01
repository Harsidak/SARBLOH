"""Why does the world model certify NOTHING on the real games, now that the
vocabulary gap is closed?

Context. `probe_residual_anatomy.py` measured that with a scalar `vacate` colour
the DSL reproduces 8 of 46 dev (game, action) pairs EXACTLY, against 0 for the
pre-2026-07-30 vocabulary. `vacate` is now implemented in GridDSL and emitted by
_candidate_ops. Yet a full tier-1 run plus tr87/tu93 printed
`[world-model] PROMOTED` exactly 0 times. Something downstream of the vocabulary
is refusing every hypothesis, and guessing which gate it is from reading the code
is not evidence.

So instrument the REAL objects -- no reimplementation. Feed collected real-game
transitions to the actual EnumerativeSynthesizer and the actual
WorldModelManager.verify, and report, per game, the three quantities that
separate the candidate explanations:

  fits/acts : how many actions got a rule at all (i.e. _fits held on that
              action's last-5 window). 0 here means the vocabulary or the
              all-or-nothing exactness of _fits is still the binding constraint.
  cover     : share of the verify SAMPLE (last 5 changed transitions, mixed
              actions) whose action the program actually has a rule for.
              compile_program returns the grid UNCHANGED for an uncovered
              action, so every uncovered transition in the sample is scored
              wrong no matter how good the covered rules are.
  verify    : the real score, against VERIFY_PROMOTE_THRESHOLD.

Reading: `fits>0 and verify < threshold and cover < 1` is a GATE-SHAPE defect --
a per-action program judged on a mixed-action sample, which no improvement to
GridDSL can fix. `fits==0` instead says the fitting criterion (component 2b) is
the wall and the DSL still cannot express these transitions in a last-5 window.

Held-out games (component_suite.json tier 3) are excluded by default, same rule
as the other probes: this is a tuning instrument.

Run:  PYTHONIOENCODING=utf-8 python scratchpad/probe_promote_gate.py [n_steps]
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.dirname(HERE)
sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, os.path.join(AGENT_DIR, "eval"))

from local_eval import _discover_real_games                      # noqa: E402
from probe_step_color import collect                             # noqa: E402
import my_agent as M                                            # noqa: E402
from my_agent import (Transition, StateEncoder, EnumerativeSynthesizer,   # noqa: E402
                      WorldModelManager, base_action, compile_program)

HELDOUT = {"ar25", "cd82", "ft09", "g50t", "m0r0", "s5i5", "sk48", "vc33"}


def as_transitions(trans):
    """Wrap (prev, act, next) triples in the real Transition record."""
    return [Transition(prev=p, action=a, next=n, reward=0.0,
                       diff_encoding="", timestamp=float(i))
            for i, (p, a, n) in enumerate(trans)]


def graph_mask(trans):
    """The HUD mask the AGENT would have, derived the way the agent derives it:
    feed the real transitions to a real StateGraph and read what it learned. The
    world model has no mask of its own -- it reads the graph's through mask_fn."""
    sg = M.StateGraph()
    for p, a, n in trans:
        sg.record(p, a, n)
    sg._update_mask()          # the agent refreshes every MASK_CHECK_EVERY steps
    return sg.hud_mask()


def main(n_steps=60, use_mask=True):
    enc = StateEncoder()
    thr = M.VERIFY_PROMOTE_THRESHOLD

    print(f"\nVERIFY_PROMOTE_THRESHOLD = {thr}   MIN_CHANGED_FOR_VERIFY = "
          f"{M.MIN_CHANGED_FOR_VERIFY}   MIN_SAMPLES_PER_RULE = {M.MIN_SAMPLES_PER_RULE}"
          f"   HUD mask: {'ON' if use_mask else 'OFF'}")
    print(f"\n{'game':6} {'chg':>4} {'acts':>5} {'mask':>5} {'fits':>5} {'cover':>6} "
          f"{'verify':>7} {'promo':>6}  program")
    print("-" * 100)

    n_games = n_fit = n_promo = 0
    cover_when_fit = []
    for gf in _discover_real_games():
        if os.path.basename(gf)[:4] in HELDOUT:
            continue
        gid, trans = collect(gf, n_steps)
        if not trans or gid in HELDOUT:
            continue
        ts = as_transitions(trans)
        mask = graph_mask(trans) if use_mask else None
        n_masked = int(mask.sum()) if mask is not None else 0
        syn = EnumerativeSynthesizer(enc, mask_fn=lambda m=mask: m)
        mgr = WorldModelManager(None, enc, mask_fn=lambda m=mask: m)

        changed = [t for t in ts if t.changed and t.prev.shape == t.next.shape]
        # How many distinct actions have enough changed samples to get a rule.
        per_act = {}
        for t in changed:
            per_act.setdefault(base_action(t.action), []).append(t)
        acts = sum(1 for v in per_act.values() if len(v) >= M.MIN_SAMPLES_PER_RULE)

        wm = syn.synthesize(ts, set())
        n_games += 1
        if wm is None or wm.fn is None:
            print(f"{gid:6} {len(changed):4d} {acts:5d} {n_masked:5d} {0:5d} {'-':>6} "
                  f"{'-':>7} {'no':>6}  (synthesize returned None)")
            continue
        n_fit += 1
        fits = len(wm.spec) if isinstance(wm.spec, list) else 0
        covered = {r["action"] for r in wm.spec} if isinstance(wm.spec, list) else set()

        # Reproduce verify's own sample so `cover` describes the scored set.
        sample = changed[-5:]
        cover = (sum(1 for t in sample if base_action(t.action) in covered) / len(sample)
                 if sample else float("nan"))
        score = mgr.verify(wm.fn, ts)
        cover_when_fit.append(cover)
        promo = "YES" if score >= thr else "no"
        if score >= thr:
            n_promo += 1
        desc = "; ".join(f"{r['action']}->{r['op']}{r['args']}" for r in wm.spec)
        print(f"{gid:6} {len(changed):4d} {acts:5d} {n_masked:5d} {fits:5d} {cover:6.2f} "
              f"{score:7.2f} {promo:>6}  {desc[:120]}")

    print("-" * 100)
    print(f"games {n_games}   synthesize produced a program: {n_fit}   "
          f"passed the promote gate: {n_promo}")
    if cover_when_fit:
        cw = np.asarray(cover_when_fit, dtype=float)
        print(f"coverage of the verify sample when a program existed: "
              f"mean {cw.mean():.2f}, =1.00 in {int((cw >= 1.0).sum())}/{len(cw)} games")
    print("\nreading: fits>0 with cover<1 and verify<threshold => the promote gate "
          "scores a per-action\nprogram on a mixed-action sample; fits==0 => the "
          "fitting criterion is still the wall.")


def detail(n_steps=60):
    """Per (game, action): does the candidate list CONTAIN a rule that fits?

    `main` established that synthesize returns None on every dev game, so _fits
    never holds. Two very different causes:

      (i)  no candidate is even close -- the transition is not one object moving
           (a second object moved, a counter ticked), so one rule per action can
           never fit and the payoff is in the fitting criterion, not the DSL.
      (ii) an exactly-fitting rule EXISTS in GridDSL's expressive range but is
           never PROPOSED, because _observed_shift / _observed_vacate derive one
           (shift, vacate) pair from the data and it differs from the one an
           exhaustive search would pick.

    So report, for each action's real last-5 window: the best candidate's fit
    count and median residual (over the proposed set), and then the best over an
    EXHAUSTIVE sweep of (colour x axis shift x vacate) -- the same space
    probe_residual_anatomy searched. `best_prop < best_swept` is case (ii) and
    names the candidate generator; both at 0 fits is case (i).
    """
    enc = StateEncoder()
    syn = EnumerativeSynthesizer(enc)
    shifts = [(d, 0) for d in range(-8, 9) if d] + [(0, d) for d in range(-8, 9) if d]

    print(f"\n{'game':6} {'act':10} {'n':>3} {'cands':>6} {'prop_fit':>9} "
          f"{'prop_res':>9} {'swept_fit':>10} {'swept_rule':>34}")
    print("-" * 100)
    gap_rows = exact_swept = exact_prop = rows = 0
    for gf in _discover_real_games():
        if os.path.basename(gf)[:4] in HELDOUT:
            continue
        gid, trans = collect(gf, n_steps)
        if not trans or gid in HELDOUT:
            continue
        by_action = {}
        for p, a, n in trans:
            if np.array_equal(p, n) or p.shape != n.shape:
                continue
            by_action.setdefault(base_action(a), []).append((p, n))

        for act, samples in sorted(by_action.items()):
            use = samples[-5:]
            if len(use) < M.MIN_SAMPLES_PER_RULE:
                continue
            bgs = [enc.detect_background(p) for p, _ in use]
            cands = syn._candidate_ops(use, sorted({int(x) for p, n in use
                                                    for x in np.unique(p)}), bgs[0])
            # Best over the rules actually PROPOSED.
            best_prop, best_res = 0, None
            for op, args in cands:
                hits, res = 0, []
                for (p, n), bg in zip(use, bgs):
                    try:
                        pred = M.GridDSL.OPS[op](p.copy(), bg, **args)
                    except Exception:
                        continue
                    if pred.shape != n.shape:
                        continue
                    d = int((pred != n).sum())
                    res.append(d)
                    if d == 0:
                        hits += 1
                if hits > best_prop or (hits == best_prop and res and
                                        (best_res is None or float(np.median(res)) < best_res)):
                    best_prop = max(best_prop, hits)
                    if res:
                        best_res = min(best_res, float(np.median(res))) if best_res is not None \
                            else float(np.median(res))
            # Best over an EXHAUSTIVE colour x shift x vacate sweep.
            pal = sorted({int(x) for p, n in use for x in np.unique(p)} |
                         {int(x) for p, n in use for x in np.unique(n)})
            best_swept, best_rule = 0, "-"
            for col in pal:
                for dr, dc in shifts:
                    for vac in [None] + pal:
                        hits = 0
                        for (p, n), bg in zip(use, bgs):
                            pred = M.GridDSL.translate_color(p.copy(), bg, color=col,
                                                             dr=dr, dc=dc, vacate=vac)
                            if pred.shape == n.shape and np.array_equal(pred, n):
                                hits += 1
                        if hits > best_swept:
                            best_swept = hits
                            best_rule = f"col={col} d=({dr},{dc}) vac={vac}"
                        if best_swept == len(use):
                            break
                    if best_swept == len(use):
                        break
                if best_swept == len(use):
                    break
            rows += 1
            if best_swept == len(use):
                exact_swept += 1
            if best_prop == len(use):
                exact_prop += 1
            if best_swept > best_prop:
                gap_rows += 1
            print(f"{gid:6} {act:10} {len(use):3d} {len(cands):6d} {best_prop:9d} "
                  f"{('%.0f' % best_res) if best_res is not None else '-':>9} "
                  f"{best_swept:10d} {best_rule:>34}")
    print("-" * 100)
    print(f"rows {rows}   fully-fit by a PROPOSED rule: {exact_prop}   "
          f"fully-fit by a SWEPT rule: {exact_swept}   sweep beats proposal: {gap_rows}")
    print("reading: swept > proposed => the candidate GENERATOR is the wall (GridDSL can "
          "express it,\n_candidate_ops never offers it). Both ~0 => one-rule-per-action is "
          "the wrong model class.")


if __name__ == "__main__":
    t0 = time.time()
    if "--detail" in sys.argv:
        detail(60)
    else:
        steps = next((int(a) for a in sys.argv[1:] if a.isdigit()), 60)
        main(steps, use_mask="--nomask" not in sys.argv)
    print(f"\n[{time.time() - t0:.1f}s]")
