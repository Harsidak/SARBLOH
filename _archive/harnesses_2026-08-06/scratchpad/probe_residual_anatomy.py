"""Where exactly do the residual pixels live? -- the test that decides whether
GridDSL (component 2a) is DONE or still missing physics.

`probe_step_color.py --residual` established that after fixing the stride, the
best single-colour rule still misses ~10-14 px per transition and those pixels
are "spread across the field" rather than confined to a HUD strip. That verdict
is necessary but not sufficient: "spread" has two causes with opposite
implications.

  (A) The moving object is modelled fine, but the DSL's notion of what happens
      at the cells it LEAVES and the cells it ENTERS is wrong. translate_color /
      step_color vacate to `bg` and overwrite the destination -- so a game where
      the avatar walks over a floor tile, a key, or a goal marker is mispredicted
      at exactly those cells, forever, and the error compounds over a rollout.
      That is a VOCABULARY gap: it belongs to GridDSL.

  (B) The residual sits away from the moving object entirely, because a SECOND
      object moved in the same action (enemy, counter, door opening). No
      one-rule-per-action program can ever fit that, no matter how rich a single
      rule is. That is a FITTING-CRITERION gap: it belongs to
      EnumerativeSynthesizer (component 2b).

So: take the best single-colour axis-aligned rule per action, and split its
residual pixels into three disjoint buckets --
    src  : cells the rule vacated (predicted bg, reality says otherwise)
    dst  : cells the rule painted (predicted colour, reality says otherwise)
    away : everything else -- untouched by the rule, so another object moved
-- and report the share of each. A src/dst-dominated residual says (A) and the
fix is a GridDSL op. An away-dominated residual says (B) and further DSL work is
measurably not the payoff.

The `passable` column is the direct test of the (A) fix: re-score the same rule
under "the mover paints over what it enters and RESTORES what it left" (i.e. the
grid is two layers -- a static floor and a mover on top). If that collapses the
src/dst residual, the missing op is a layered move, not a richer shift.

Run:  PYTHONIOENCODING=utf-8 python scratchpad/probe_residual_anatomy.py [n_steps]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.dirname(HERE)
sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, os.path.join(AGENT_DIR, "eval"))

from local_eval import _discover_real_games          # noqa: E402
from my_agent import StateEncoder, GridDSL as D      # noqa: E402
from probe_step_color import collect                 # noqa: E402

E = StateEncoder()
SHIFTS = [(d, 0) for d in range(-8, 9) if d] + [(0, d) for d in range(-8, 9) if d]

# The held-out split (component_suite.json tier 3) is excluded by default: a
# probe is a tuning instrument, and reading these rows while choosing a DSL op
# is exactly the inspection the split exists to prevent. `--all` is for the
# generalization report only, once a design is already frozen.
HELDOUT = {"ar25", "cd82", "ft09", "g50t", "m0r0", "s5i5", "sk48", "vc33"}


def const_vacate_move(g, floor_color, color, dr, dc):
    """Layered move with a SCALAR floor: the mover leaves one fixed colour
    behind instead of a whole restored layer.

    This is the version GridDSL can actually express -- `floor_color` is an
    ordinary op argument the synthesizer can enumerate over the palette,
    whereas layered_move's `floor` is a grid that would have to be carried as
    hidden model state. If this scores as well as layered_move, the fix is one
    optional argument on translate_color/step_color; if it does not, the world
    model genuinely needs a static layer."""
    out = g.copy()
    src = (g == color)
    out[src] = floor_color
    h, w = g.shape
    rs, cs = np.nonzero(src)
    nr, nc = rs + dr, cs + dc
    keep = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
    out[nr[keep], nc[keep]] = color
    return out


def layered_move(g, bg, floor, color, dr, dc):
    """The (A) hypothesis: the grid is a static floor plus a mover on top.

    The mover's cells are lifted off, the floor beneath them is restored, and
    the mover is painted at the shifted position -- so walking over a tile
    neither erases it nor is blocked by it. `floor` is the per-action estimate
    of the static layer (see floor_estimate)."""
    out = g.copy()
    src = (g == color)
    out[src] = floor[src]
    h, w = g.shape
    rs, cs = np.nonzero(src)
    nr, nc = rs + dr, cs + dc
    keep = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
    out[nr[keep], nc[keep]] = color
    return out


def floor_estimate(samples, color):
    """Per-pixel modal colour across the observed frames, with the mover's own
    colour excluded -- i.e. what the board looks like underneath the mover.

    This is an ESTIMATE from data, not an oracle: it uses only frames the agent
    has already seen, so anything it shows is available to the real synthesizer
    too."""
    frames = [p for p, _ in samples] + [n for _, n in samples]
    stack = np.stack(frames)
    h, w = stack.shape[1:]
    out = np.zeros((h, w), dtype=stack.dtype)
    for r in range(h):
        for c in range(w):
            col = stack[:, r, c]
            col = col[col != color]
            if len(col) == 0:
                out[r, c] = color
            else:
                vals, cnts = np.unique(col, return_counts=True)
                out[r, c] = vals[int(np.argmax(cnts))]
    return out


def main(n_steps=60, include_heldout=False):
    print(f"\n{'game':6} {'act':4} {'n':>4} {'miss':>5} {'src%':>6} {'dst%':>6} "
          f"{'away%':>6} {'layered':>8} {'vac':>5} {'vcol':>5}  verdict")
    print("-" * 88)
    tot = {"src": 0.0, "dst": 0.0, "away": 0.0}
    rows = 0
    improved = 0
    vac_exact = lay_exact = 0
    for gf in _discover_real_games():
        if not include_heldout and os.path.basename(gf)[:4] in HELDOUT:
            continue
        gid, trans = collect(gf, n_steps)
        if not trans:
            continue
        if not include_heldout and gid in HELDOUT:
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
            cols.discard(int(E.detect_background(changed[0][0])))
            if not cols or len(cols) > 12:
                continue

            # Best single-colour axis-aligned rule, exactly as --residual picks it.
            best = (1e9, None, None)
            for col in sorted(cols):
                for dr, dc in SHIFTS:
                    ms = []
                    for p, n in changed:
                        bg = E.detect_background(p)
                        pred = D.translate_color(p.copy(), bg, color=col, dr=dr, dc=dc)
                        if pred.shape == n.shape:
                            ms.append(int((pred != n).sum()))
                    if ms and float(np.median(ms)) < best[0]:
                        best = (float(np.median(ms)), col, (dr, dc))
            if best[1] is None or best[0] == 0:
                continue
            col, (dr, dc) = best[1], best[2]

            buckets = {"src": 0, "dst": 0, "away": 0}
            for p, n in changed:
                bg = E.detect_background(p)
                pred = D.translate_color(p.copy(), bg, color=col, dr=dr, dc=dc)
                if pred.shape != n.shape:
                    continue
                wrong = (pred != n)
                src = (p == col)
                dst = np.zeros_like(src)
                rs, cs = np.nonzero(src)
                nr, nc = rs + dr, cs + dc
                keep = (nr >= 0) & (nr < n.shape[0]) & (nc >= 0) & (nc < n.shape[1])
                dst[nr[keep], nc[keep]] = True
                # src and dst overlap for small shifts; charge the overlap to dst,
                # since that is the cell the rule actually wrote last.
                src = src & ~dst
                buckets["src"] += int((wrong & src).sum())
                buckets["dst"] += int((wrong & dst).sum())
                buckets["away"] += int((wrong & ~src & ~dst).sum())
            total = sum(buckets.values())
            if total == 0:
                continue

            floor = floor_estimate(changed, col)
            lm = []
            for p, n in changed:
                bg = E.detect_background(p)
                pred = layered_move(p.copy(), bg, floor, col, dr, dc)
                if pred.shape == n.shape:
                    lm.append(int((pred != n).sum()))
            lay = float(np.median(lm)) if lm else float("nan")

            # Best SCALAR vacate colour, searched over the same palette the
            # synthesizer would enumerate -- no oracle, no per-pixel layer.
            pal = sorted(set(int(x) for p, _ in changed for x in np.unique(p)))
            vac, vcol = float("inf"), None
            for fc in pal:
                vm = []
                for p, n in changed:
                    pred = const_vacate_move(p.copy(), fc, col, dr, dc)
                    if pred.shape == n.shape:
                        vm.append(int((pred != n).sum()))
                if vm and float(np.median(vm)) < vac:
                    vac, vcol = float(np.median(vm)), fc

            sh = {k: 100.0 * v / total for k, v in buckets.items()}
            for k in tot:
                tot[k] += buckets[k]
            rows += 1
            if lay < best[0] - 0.5:
                improved += 1
            if lay == 0:
                lay_exact += 1
            if vac == 0:
                vac_exact += 1
            verdict = ("DSL gap (mover cells)" if sh["src"] + sh["dst"] >= 60
                       else "2b gap (other objects moved)")
            print(f"{gid:6} {act[-1]:4} {len(changed):4d} {best[0]:5.0f} "
                  f"{sh['src']:6.1f} {sh['dst']:6.1f} {sh['away']:6.1f} "
                  f"{lay:8.0f} {vac:5.0f} {str(vcol):>5}  {verdict}")
    print("-" * 88)
    g = sum(tot.values())
    if g:
        print(f"pooled residual over {rows} (game, action) pairs: "
              f"src {100 * tot['src'] / g:.1f}%  dst {100 * tot['dst'] / g:.1f}%  "
              f"away {100 * tot['away'] / g:.1f}%")
        print(f"layered move beat the flat rule on {improved}/{rows} pairs")
        print(f"EXACT (zero-residual) rules: layered {lay_exact}/{rows}, "
              f"scalar-vacate {vac_exact}/{rows}  "
              f"(flat rule: 0/{rows} by construction)")
    else:
        print("no residual to analyse")


if __name__ == "__main__":
    _n = [a for a in sys.argv[1:] if a.isdigit()]
    main(int(_n[0]) if _n else 60, include_heldout="--all" in sys.argv)
