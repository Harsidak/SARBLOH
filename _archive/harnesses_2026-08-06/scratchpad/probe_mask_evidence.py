"""What does the HUD mask do to the EVIDENCE the synthesizer gets?

`probe_promote_gate.py` measured that switching the world model onto the
graph's broad mask (per-cell verdict + clock-periodic row/col strips) makes two
dev games much richer -- dc22 and sp80 go from one rule at verify 0.80/1.00 to
FOUR rules at verify 1.00 -- while tu93 loses its only rule and drops out
entirely. With the `precise=True` mask (per-cell only, no strips) nothing moves
at all: on 15 of 16 dev games that component is empty, so the strips carry the
whole signal. The choice is therefore broad-or-nothing, and tu93 is the price.

Masking can only make a comparison MORE permissive on a fixed sample, so a rule
cannot fit fewer of the SAME samples. Any loss has to come from the sample set
changing. Two candidate mechanisms:

  (a) starvation -- HUD-only transitions are dropped, leaving an action with
      fewer than MIN_SAMPLES_PER_RULE samples, so it gets no rule at all.
  (b) window slide -- synthesize fits `samples[-5:]`. Dropping recent HUD-only
      transitions pulls OLDER world-changes into the window, and the rule that
      fit the recent five does not fit those.

(a) says the mask is eating real evidence and should be narrowed; (b) says the
five most recent genuine world-changes simply are not explained by the rule --
the old score was a favourable window, not a better model.

So report, per (game, action): raw changed count, live (masked) changed count,
and how many of each window the game's best raw rule reproduces.

Run: PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 python scratchpad/probe_mask_evidence.py [ids...]
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
from my_agent import (StateEncoder, EnumerativeSynthesizer,      # noqa: E402
                      base_action, masked_equal)

HELDOUT = {"ar25", "cd82", "ft09", "g50t", "m0r0", "s5i5", "sk48", "vc33"}


def masks(trans):
    sg = M.StateGraph()
    for p, a, n in trans:
        sg.record(p, a, n)
    sg._update_mask()
    return sg.hud_mask(), sg.hud_mask(precise=True)


def best_rule(syn, enc, use, mask=None):
    """The best-scoring proposed rule on `use`, compared under `mask`."""
    if not use:
        return None, 0
    bgs = [enc.detect_background(p) for p, _ in use]
    pal = sorted({int(x) for p, n in use for x in np.unique(p)})
    best, hits = None, 0
    for op, args in syn._candidate_ops(use, pal, bgs[0]):
        s = syn._score(op, args, use, bgs, mask=mask)
        if s > hits:
            best, hits = (op, args), s
    return best, hits


def fit(syn, enc, rule, use, mask=None):
    if rule is None:
        return 0
    bgs = [enc.detect_background(p) for p, _ in use]
    return syn._score(rule[0], rule[1], use, bgs, mask=mask)


def main(ids):
    enc = StateEncoder()
    syn = EnumerativeSynthesizer(enc)
    # rawbest = best rule on the raw last-5, scored raw. livebest = best rule on
    # the window masking hands over, scored masked. Comparing the raw-best rule
    # against the live window would understate masking: the live window usually
    # has a DIFFERENT best rule, and that rule is the one the agent would get.
    print(f"\n{'game':6} {'act':9} {'raw_n':>6} {'live_n':>7} {'broad':>6} {'prec':>5} "
          f"{'rawbest':>9} {'livebest':>9}  verdict")
    print("-" * 104)
    for gf in _discover_real_games():
        gid0 = os.path.basename(gf)[:4]
        if gid0 in HELDOUT or (ids and gid0 not in ids):
            continue
        gid, trans = collect(gf, 60)
        if not trans:
            continue
        broad, prec = masks(trans)
        nb = int(broad.sum()) if broad is not None else 0
        npz = int(prec.sum()) if prec is not None else 0

        raw, live = {}, {}
        for p, a, n in trans:
            if p.shape != n.shape or np.array_equal(p, n):
                continue
            raw.setdefault(base_action(a), []).append((p, n))
            if not masked_equal(p, n, broad):
                live.setdefault(base_action(a), []).append((p, n))

        for act in sorted(raw):
            rw, lw = raw[act][-5:], live.get(act, [])[-5:]
            if len(rw) < M.MIN_SAMPLES_PER_RULE:
                continue
            _, rhits = best_rule(syn, enc, rw)
            _, lhits = best_rule(syn, enc, lw, mask=broad)
            need_r = max(M.MIN_SAMPLES_PER_RULE, int(np.ceil(0.8 * len(rw))))
            need_l = max(M.MIN_SAMPLES_PER_RULE, int(np.ceil(0.8 * len(lw)))) if lw else 0
            got_r, got_l = rhits >= need_r, bool(lw) and lhits >= need_l
            if len(lw) < M.MIN_SAMPLES_PER_RULE:
                verdict = ("(a) STARVED: mask ate a rule that FIT" if got_r
                           else "(a) starved, but there was no rule anyway")
            elif got_l and not got_r:
                verdict = "GAIN: mask makes this action fittable"
            elif got_r and not got_l:
                verdict = "LOSS: fittable raw, not under the mask"
            else:
                verdict = "same verdict either way"
            print(f"{gid:6} {act:9} {len(raw[act]):6d} {len(live.get(act, [])):7d} "
                  f"{nb:6d} {npz:5d} {rhits:6d}/{len(rw):<2d} {lhits:6d}/{len(lw):<2d}  {verdict}")
    print("-" * 104)
    print("reading: (a) => the mask is eating real evidence (narrow it). (b) with the rule "
          "failing\n=> the recent five genuine world-changes are simply not one moving object.")


if __name__ == "__main__":
    t0 = time.time()
    main({a for a in sys.argv[1:] if not a.startswith("-")})
    print(f"\n[{time.time() - t0:.1f}s]")
