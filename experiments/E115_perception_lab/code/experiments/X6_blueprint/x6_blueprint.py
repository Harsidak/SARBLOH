"""
X6 -- Blueprint redraw (vision as inverse graphics, Marr; recognition-by-components, Biederman).

Mechanism: re-draw the scene as a clean schematic instead of pixels:
  - floor = white, walls/void = hatched dark, the game's tile lattice drawn faintly (if any)
  - every object outlined and labelled with a shape word (plus, disc, hollow diamond, dotted ring, bar...)
  - status items (bars, counters, icons at the screen edge) moved OUT of the map into a labelled side panel
Goal: remove colour noise, give every part a name the model can reason with, separate world from UI.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "X3_routines"))
from scene import N, NAMES, PAL, build_scene, load_grid, font, lattice_unit  # noqa: E402
from common_out import is_hud, text_block, hstack, vstack, titled  # noqa: E402
from x3_routines import self_candidates  # noqa: E402

CELL = 10


def run(path, out_png, out_txt, actions=None):
    g, notes = load_grid(path)
    sc = build_scene(g, notes)
    cands = self_candidates(sc)
    world = [o for o in sc.objects if not is_hud(o, sc)]
    hud = [o for o in sc.objects if is_hud(o, sc)]
    floor = cands[0].on if cands else max(sc.terrain, key=lambda t: sum(o.on == t for o in world))
    hud_cells = set()
    for o in hud:
        if o.panel is not None:
            p = sc.panels[o.panel]
            hud_cells |= {(y, x) for y in range(p.r0, p.r1 + 1) for x in range(p.c0, p.c1 + 1)}
        hud_cells |= set(o.cells)
    img = Image.new("RGB", (N * CELL, N * CELL), (255, 255, 255))
    d = ImageDraw.Draw(img)
    lu = lattice_unit(sc)
    for y in range(N):
        for x in range(N):
            X, Y = x * CELL, y * CELL
            if (y, x) in hud_cells:
                d.rectangle([X, Y, X + CELL - 1, Y + CELL - 1], fill=(225, 225, 235))
            elif g[y, x] == floor or not (g[y, x] in sc.terrain):
                d.rectangle([X, Y, X + CELL - 1, Y + CELL - 1], fill=(236, 231, 216))
            elif any(p.r0 <= y <= p.r1 and p.c0 <= x <= p.c1 for p in sc.panels if not p.border):
                d.rectangle([X, Y, X + CELL - 1, Y + CELL - 1], fill=(255, 244, 200))
            else:
                d.rectangle([X, Y, X + CELL - 1, Y + CELL - 1], fill=(70, 72, 80))
                d.line([X, Y + CELL - 1, X + CELL - 1, Y], fill=(100, 102, 112))
    if lu:
        u, ro, co = lu
        for r in range(ro, N, u):
            d.line([(0, r * CELL), (N * CELL, r * CELL)], fill=(215, 215, 215))
        for c in range(co, N, u):
            d.line([(c * CELL, 0), (c * CELL, N * CELL)], fill=(215, 215, 215))
    for p in sc.panels:
        if not p.border and any(o.panel == p.id for o in world):
            d.rectangle([p.c0 * CELL, p.r0 * CELL, (p.c1 + 1) * CELL - 1, (p.r1 + 1) * CELL - 1], outline=(200, 150, 0), width=3)
    f = font(12, True)
    labels = []
    for k, o in enumerate(world):
        for y, x in o.cells:
            v = g[y, x]
            d.rectangle([x * CELL, y * CELL, (x + 1) * CELL - 1, (y + 1) * CELL - 1], fill=tuple(PAL[v]), outline=(60, 60, 60))
        if o.kind != "line":
            d.rectangle([o.c0 * CELL - 1, o.r0 * CELL - 1, (o.c1 + 1) * CELL, (o.r1 + 1) * CELL], outline=(0, 0, 0), width=1)
        letter = chr(65 + k)
        tx, ty = (o.c1 + 1) * CELL + 3, o.r0 * CELL
        d.text((tx, ty), letter, fill=(200, 0, 0), font=font(16, True), stroke_width=2, stroke_fill=(255, 255, 255))
        labels.append(f"{letter} = {o.name} ({o.h}x{o.w})" + (f", inside the yellow-outlined box" if o.panel is not None else ""))
    # side panel: HUD
    side_items = []
    for o in hud:
        cell = max(4, min(14, 160 // max(o.h, o.w)))
        im = Image.new("RGB", (max(200, o.w * cell + 8), o.h * cell + 8), (60, 60, 70))
        dd = ImageDraw.Draw(im)
        cg = o.colgrid(g)
        for y in range(o.h):
            for x in range(o.w):
                if cg[y, x] >= 0:
                    dd.rectangle([4 + x * cell, 4 + y * cell, 4 + (x + 1) * cell - 2, 4 + (y + 1) * cell - 2], fill=tuple(PAL[cg[y, x]]))
        if o.kind == "bar":
            n = max(o.h, o.w) // 1
            cap = f"{NAMES[o.colors.most_common(1)[0][0]]} bar, length {max(o.h, o.w)} ({'vertical' if o.h > o.w else 'horizontal'}) -- a gauge"
        else:
            cap = f"{o.name}"
        side_items.append(titled(im, cap, size=12))
    # merge identical HUD copies in caption list
    hud_names = [o.name for o in hud]
    hud_lines = []
    for nm in dict.fromkeys(hud_names):
        k = hud_names.count(nm)
        hud_lines.append(f"{nm}" + (f" x{k}" if k > 1 else ""))
    side = vstack([text_block(["STATUS / HUD (moved out of the map)"], 260)] + side_items) if side_items else text_block(["no HUD found"], 260)
    walls = [NAMES[t] for t in sc.terrain if t != floor]
    lines = [f"BLUEPRINT: light beige = open floor ({NAMES[floor]} in the original), hatched = wall/void ({', '.join(walls)}), "
             + ("yellow = special box. " if any(not p.border for p in sc.panels) else "")
             + (f"Faint grid = the game's {lu[0]}x{lu[0]} tile lattice." if lu else "No tile lattice: free placement."),
             "Objects on the map:"] + labels + ["Status panel: " + "; ".join(hud_lines) if hud_lines else "Status panel: none"]
    out = hstack([titled(img, "Blueprint (world only)"), side])
    vstack([out, text_block(lines, out.width, size=13)]).save(out_png)
    txt = "\n".join(lines); open(out_txt, "w").write(txt)
    return txt


if __name__ == "__main__":
    print(run(sys.argv[1], sys.argv[2], sys.argv[3]))
