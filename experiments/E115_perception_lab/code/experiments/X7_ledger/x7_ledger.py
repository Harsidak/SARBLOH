"""
X7 -- Role ledger (perception as a falsifiable theory).

Basis: EMPA / theory-based RL (arXiv 2107.12544): people infer object TYPES and ROLES from very little
data using strong priors (agents, goals, obstacles, items); contingency-awareness (arXiv 1811.01483).

Mechanism: every object gets scored against a fixed set of roles from design-grammar priors plus the
outputs of the other routines (lattice, tethers/midpoint, look-alikes, broken symmetry, enclosed space).
Scores -> probabilities. Each top hypothesis comes with the cheapest action that would confirm/kill it.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
from PIL import ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("..", "../X2_surprise", "../X3_routines", "../X4_threads"):
    sys.path.insert(0, os.path.join(HERE, sub))
from scene import N, NAMES, PAL, build_scene, load_grid, to_image, font, lattice_unit, fill_holes  # noqa: E402
from common_out import is_hud, text_block, vstack, titled  # noqa: E402
from x2_surprise import symmetry_break  # noqa: E402
from x3_routines import self_candidates  # noqa: E402
from x4_threads import relate  # noqa: E402

CELL = 10
ROLE_COL = {"YOU": (0, 230, 0), "GOAL": (255, 60, 60), "MODIFIER": (255, 160, 0), "INVENTORY": (0, 200, 255),
            "MOVES LEFT": (255, 255, 0), "LIVES": (255, 120, 200), "HANDLE (click)": (0, 255, 160),
            "CARRIED THING": (200, 120, 255), "SCENERY": (160, 160, 160)}
TESTS = {"YOU": "press one arrow key: it should move one tile and the gauge should drop",
         "GOAL": "get the right thing into it and see if the level ends",
         "MODIFIER": "step onto it, then check whether the inventory icon changed",
         "INVENTORY": "watch it after touching the modifier",
         "MOVES LEFT": "take any action and check it shrinks by one unit",
         "LIVES": "watch it when the gauge runs out or you fail",
         "HANDLE (click)": "click an empty spot once: the selected handle should jump there",
         "CARRIED THING": "move one handle: it should slide to the new midpoint",
         "SCENERY": "ignore unless something else fails"}


def run(path, out_png, out_txt, actions=(1, 2, 3, 4)):
    g, notes = load_grid(path)
    sc = build_scene(g, notes)
    objs = [o for o in sc.objects if o.kind != "line"]
    lines_ = [o for o in sc.objects if o.kind == "line"]
    score = {o.id: defaultdict(float) for o in objs}
    ev = {o.id: defaultdict(list) for o in objs}

    def add(o, role, pts, why):
        score[o.id][role] += pts; ev[o.id][role].append(why)

    keyboard = any(a in (1, 2, 3, 4) for a in actions)
    clicking = 6 in actions
    lu = lattice_unit(sc)
    cands = self_candidates(sc)
    hud_copies = defaultdict(list)
    for o in objs:
        if is_hud(o, sc):
            hud_copies[(o.name, o.h, o.w)].append(o)
    # correspondences
    links = []
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            rels, big, small = relate(a, b)
            if rels:
                links.append((rels[0][0], a, b, rels[0][1]))
    for o in objs:
        hud = is_hud(o, sc)
        if o.kind == "bar" and hud:
            add(o, "MOVES LEFT", 3, "long bar at the screen edge = a gauge")
        if hud and len(hud_copies[(o.name, o.h, o.w)]) >= 2:
            add(o, "LIVES", 3, f"{len(hud_copies[(o.name, o.h, o.w)])} identical copies at the screen edge = a count")
        if hud and o.kind in ("icon", "object") and len(hud_copies[(o.name, o.h, o.w)]) == 1 and o.kind != "bar":
            add(o, "INVENTORY", 2, "single icon in a status box")
        if not hud and keyboard and cands and o is cands[0]:
            add(o, "YOU", 3, "most self-like: compact, multi-coloured, unique, standing on the floor")
            if lu and lu[0] == o.h:
                add(o, "YOU", 1, f"its size ({o.h}) equals the floor tile size -> it walks tile by tile")
        sb = symmetry_break(o, g)
        if sb and not hud and o.size <= 12 and (not cands or o is not cands[0]):
            add(o, "MODIFIER", 2, "small symbol on the floor with deliberately lopsided colours")
        if o.kind == "dotted" or (fill_holes(o.mask()).sum() > o.size and not hud):
            inner = [q for q in objs if q is not o and q.h == o.h - 2 and q.w == o.w - 2]
            if inner:
                add(o, "GOAL", 3, f"hollow outline whose empty inside fits a {inner[0].h}x{inner[0].w} object exactly")
        if o.panel is not None and not hud:
            add(o, "GOAL", 1, "displayed inside a special box in the play area")
    for kind, a, b, txt in links:
        if kind in ("TURNED", "NEAR"):
            for x, y in ((a, b), (b, a)):
                if is_hud(x, sc) and not is_hud(y, sc):
                    add(x, "INVENTORY", 2, f"HUD icon is the same shape as the {y.name} (differently oriented/sized)")
                    add(y, "GOAL", 2, f"its shape is repeated by the HUD icon -> it asks for that item, in this orientation")
        if kind == "STATE":
            for x in (a, b):
                add(x, "HANDLE (click)" if clicking else "SCENERY", 2, "one of a same-shape pair that differs only in colour (on/off state)")
    if clicking and lines_:
        ends = defaultdict(int)
        for ln in lines_:
            s = set(ln.cells)
            for e in [c for c in ln.cells if sum((c[0] + dy, c[1] + dx) in s for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx) <= 1]:
                best = min(objs, key=lambda o: max(o.r0 - e[0], 0, e[0] - o.r1) + max(o.c0 - e[1], 0, e[1] - o.c1))
                ends[best.id] += 1
        for oid, k in ends.items():
            o = sc.objects[oid]
            if k >= 2:
                add(o, "CARRIED THING", 3, "both tethers end on it (it hangs between the anchors)")
            else:
                add(o, "HANDLE (click)", 2, "a tether starts here -> an anchor you can grab")
        cols = defaultdict(list)
        for o in objs:
            for c in o.colors: cols[c].append(o)
        for o in objs:
            if score[o.id].get("GOAL") and score[o.id].get("GOAL") >= 3:
                body = [q for q in objs if score[q.id].get("CARRIED THING")]
                if body and set(o.colors) & set(body[0].colors):
                    add(o, "GOAL", 1, f"same colour as the carried {body[0].name}")
    # probabilities
    ledger = []
    for o in objs:
        sc_ = dict(score[o.id])
        tot = sum(sc_.values()) + 1.0
        probs = sorted(((r, v / tot) for r, v in sc_.items()), key=lambda t: -t[1])
        if not probs:
            probs = [("SCENERY", 0.5)]
        ledger.append((o, probs))
    ledger.sort(key=lambda t: -t[1][0][1])
    img = to_image(g, cell=CELL, dim=0.3)
    d = ImageDraw.Draw(img)
    out = [f"ROLE LEDGER ({'arrow keys' if keyboard else 'clicks' if clicking else 'unknown controls'} available)"]
    for o, probs in ledger:
        r, p = probs[0]
        col = ROLE_COL.get(r, (200, 200, 200))
        d.rectangle([o.c0 * CELL - 2, o.r0 * CELL - 2, (o.c1 + 1) * CELL + 1, (o.r1 + 1) * CELL + 1], outline=col, width=3)
        lab = f"{r} {p:.0%}"
        tx = min(o.c0 * CELL, N * CELL - 9 * len(lab)); ty = max(0, o.r0 * CELL - 18) if o.r0 > 3 else (o.r1 + 1) * CELL + 2
        d.text((tx, ty), lab, fill=col, font=font(14, True), stroke_width=3, stroke_fill=(0, 0, 0))
        alt = ", ".join(f"{rr} {pp:.0%}" for rr, pp in probs[1:3])
        out.append(f"* {o.name} @ rows {o.r0}-{o.r1}, cols {o.c0}-{o.c1}: {r} {p:.0%}" + (f" (else {alt})" if alt else ""))
        for why in ev[o.id][r][:3]:
            out.append(f"     because: {why}")
        out.append(f"     test: {TESTS.get(r, '')}")
    vstack([titled(img, "Role hypotheses (colour = role, % = confidence)"), text_block(out, img.width, size=13)]).save(out_png)
    txt = "\n".join(out); open(out_txt, "w").write(txt)
    return txt


if __name__ == "__main__":
    acts = tuple(int(a) for a in sys.argv[4].split(",")) if len(sys.argv) > 4 else (1, 2, 3, 4)
    print(run(sys.argv[1], sys.argv[2], sys.argv[3], acts))
