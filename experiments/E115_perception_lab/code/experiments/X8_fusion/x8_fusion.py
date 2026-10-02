"""
X8 -- Fusion briefing card.

Built AFTER grading X1-X7 and the held-out check:
  - X6 blueprint is the clearest canvas (world vs walls vs HUD, tile lattice)       -> base layer
  - X3 routines and X4 look-alikes produce FACTS that are true by construction    -> 'MEASURED' tier
  - X7 roles scored best on the two screenshots but generalised poorly on 23 unseen games (YOU 6/14)
    -> shown only as 'GUESSES' with a confidence and the one action that settles each
  - X5 dynamics priors failed out of sample (1/16 games) -> dropped; only its 'test' idea is kept
Output: one image (map + role tags + threads + HUD side panel) and <= ~200 words in two tiers.
"""
from __future__ import annotations

import io
import os
import re
import sys
from contextlib import redirect_stdout

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("..", "../X2_surprise", "../X3_routines", "../X4_threads", "../X7_ledger"):
    sys.path.insert(0, os.path.join(HERE, sub))
from scene import N, NAMES, PAL, build_scene, load_grid, font, lattice_unit, shape_word  # noqa: E402
from common_out import is_hud, text_block, hstack, vstack, titled  # noqa: E402
import x3_routines, x4_threads, x7_ledger  # noqa: E402
from x2_surprise import symmetry_break  # noqa: E402
from x3_routines import self_candidates  # noqa: E402

CELL = 10


def short(o):
    cols = [c for c, _ in o.colors.most_common(2)]
    return f"{'/'.join(NAMES[c] for c in cols)} {shape_word(o.mask(), o.kind)}"


def run(path, out_png, out_txt, actions=(1, 2, 3, 4)):
    g, notes = load_grid(path)
    sc = build_scene(g, notes)
    import shutil, tempfile
    tmpdir = tempfile.mkdtemp(prefix="x8_")
    tmp = os.path.join(tmpdir, "part")
    with redirect_stdout(io.StringIO()):
        t3 = x3_routines.run(path, tmp + "3.png", tmp + "3.txt")
        t4 = x4_threads.run(path, tmp + "4.png", tmp + "4.txt")
        t7 = x7_ledger.run(path, tmp + "7.png", tmp + "7.txt", actions)
    shutil.rmtree(tmpdir, ignore_errors=True)
    # letters
    items = [o for o in sc.objects if o.kind != "line"]
    letter = {o.id: chr(65 + i) for i, o in enumerate(items)}
    by_name = {}
    for o in items:
        by_name.setdefault(o.name, []).append(o)
    label_of = {nm: f"{'/'.join(letter[o.id] for o in os_)} ({short(os_[0])})" for nm, os_ in by_name.items()}
    pat = re.compile("(?:the )?(" + "|".join(re.escape(n) for n in sorted(by_name, key=len, reverse=True)) + ")")
    def rename(txt):
        return pat.sub(lambda m: label_of[m.group(1)], txt)
    # ---------------- image: blueprint base
    cands = self_candidates(sc)
    world = [o for o in sc.objects if not is_hud(o, sc)]
    hud = [o for o in sc.objects if is_hud(o, sc)]
    floor = cands[0].on if cands else max(sc.terrain, key=lambda t: sum(o.on == t for o in world))
    hud_cells = set()
    for o in hud:
        if o.panel is not None:
            p = sc.panels[o.panel]; hud_cells |= {(y, x) for y in range(p.r0, p.r1 + 1) for x in range(p.c0, p.c1 + 1)}
        hud_cells |= set(o.cells)
    img = Image.new("RGB", (N * CELL, N * CELL), (236, 231, 216)); d = ImageDraw.Draw(img)
    for y in range(N):
        for x in range(N):
            X, Y = x * CELL, y * CELL
            if (y, x) in hud_cells:
                d.rectangle([X, Y, X + CELL - 1, Y + CELL - 1], fill=(205, 205, 215))
            elif g[y, x] in sc.terrain and g[y, x] != floor and not any(p.r0 <= y <= p.r1 and p.c0 <= x <= p.c1 for p in sc.panels if not p.border):
                d.rectangle([X, Y, X + CELL - 1, Y + CELL - 1], fill=(70, 72, 80)); d.line([X, Y + CELL - 1, X + CELL - 1, Y], fill=(100, 102, 112))
            elif any(p.r0 <= y <= p.r1 and p.c0 <= x <= p.c1 for p in sc.panels if not p.border):
                d.rectangle([X, Y, X + CELL - 1, Y + CELL - 1], fill=(255, 244, 200))
    lu = lattice_unit(sc)
    if lu:
        u, ro, co = lu
        for r in range(ro, N, u): d.line([(0, r * CELL), (N * CELL, r * CELL)], fill=(210, 205, 190))
        for c in range(co, N, u): d.line([(c * CELL, 0), (c * CELL, N * CELL)], fill=(210, 205, 190))
    for o in sc.objects:
        for y, x in o.cells:
            d.rectangle([x * CELL, y * CELL, (x + 1) * CELL - 1, (y + 1) * CELL - 1], fill=tuple(PAL[g[y, x]]), outline=(60, 60, 60))
    # threads from X4 (TURNED / STATE)
    for o_a in items:
        for o_b in items:
            if o_a.id >= o_b.id: continue
            rels, big, small = x4_threads.relate(o_a, o_b)
            if rels and rels[0][0] in ("TURNED", "STATE"):
                col = (255, 0, 255) if rels[0][0] == "TURNED" else (0, 170, 255)
                pts = [(o_a.cx * CELL + 5, o_a.cy * CELL + 5), (o_b.cx * CELL + 5, o_b.cy * CELL + 5)]
                for t in np.linspace(0, 1, 40)[::2]:
                    x0 = pts[0][0] + (pts[1][0] - pts[0][0]) * t; y0 = pts[0][1] + (pts[1][1] - pts[0][1]) * t
                    x1 = pts[0][0] + (pts[1][0] - pts[0][0]) * (t + 0.025); y1 = pts[0][1] + (pts[1][1] - pts[0][1]) * (t + 0.025)
                    d.line([(x0, y0), (x1, y1)], fill=col, width=3)
    # role tags from X7
    roles = {}
    for line in t7.splitlines():
        m = re.match(r"\* (.+) @ rows (\d+)-(\d+), cols (\d+)-(\d+): ([A-Z ()a-z]+?) (\d+)%", line)
        if m:
            roles[(int(m.group(2)), int(m.group(4)))] = (m.group(6).strip(), int(m.group(7)))
    for o in items:
        L = letter[o.id]
        role = roles.get((o.r0, o.c0))
        col = x7_ledger.ROLE_COL.get(role[0], (220, 220, 220)) if role else (220, 220, 220)
        d.rectangle([o.c0 * CELL - 2, o.r0 * CELL - 2, (o.c1 + 1) * CELL + 1, (o.r1 + 1) * CELL + 1], outline=col, width=3)
        lab = f"{L}: {role[0]}?" if role else L
        tx = min(max(0, o.c0 * CELL), N * CELL - 8 * len(lab)); ty = o.r0 * CELL - 18 if o.r0 > 2 else (o.r1 + 1) * CELL + 2
        d.text((tx, ty), lab, fill=col, font=font(14, True), stroke_width=3, stroke_fill=(0, 0, 0))
    # midpoint marker
    if "MIDPOINT" in t3:
        for o in items:
            if o.name in t3.split("MIDPOINT")[0].split("R3 TRACE: the")[-1]:
                d.ellipse([o.cx * CELL - 4, o.cy * CELL - 4, o.cx * CELL + 14, o.cy * CELL + 14], outline=(255, 255, 0), width=3)
    # ---------------- text: two tiers
    facts = []
    for ln in t3.splitlines():
        if ln.startswith("R1 REACH: the most self-like"):
            m = re.search(r"floor is built from (\d+)x\d+ tiles", ln)
            facts.append(f"Floor is a {m.group(1)}x{m.group(1)} tile grid." if m else "")
        elif ln.strip().startswith("- ") and ("reachable" in ln or "steps" in ln or "rock" in ln or "open space" in ln):
            facts.append(ln.strip()[2:])
        elif ln.startswith("R1 REACH: no tile grid"):
            facts.append(ln[len("R1 REACH: "):])
        elif "encloses" in ln or "MIDPOINT" in ln or ("R4 ALIGN" in ln and "no two" not in ln):
            facts.append(ln.split(": ", 1)[1])
        elif "R3 TRACE: a " in ln:
            facts.append(ln.split(": ", 1)[1])
    for ln in t4.splitlines():
        if ln.startswith("#") and ("TURNED" in ln or "STATE" in ln):
            facts.append(ln.split(": ", 1)[1].split("  [")[0])
        elif ln.startswith("COPIES"):
            facts.append(ln.split(": ", 1)[1])
    for o in items:
        sb = symmetry_break(o, g)
        if sb and not is_hud(o, sc):
            facts.append(f"{letter[o.id]} has a symmetric shape but lopsided colours (marks a direction/state)")
    guesses = []
    for o in items:
        role = roles.get((o.r0, o.c0))
        if role:
            guesses.append(f"{letter[o.id]} ({short(o)}) = {role[0]} ({role[1]}%) - test: {x7_ledger.TESTS.get(role[0], '')}")
    facts = [rename(f) for f in facts if f]
    ctl = "arrow keys" if any(a in (1, 2, 3, 4) for a in actions) else "clicks" if 6 in actions else "?"
    lines = [f"BRIEFING ({ctl})", "MEASURED (certain):"] + [f"- {f}" for f in facts] + ["GUESSES (check with one action):"] + [f"- {gx}" for gx in guesses]
    # HUD side
    side_imgs = []
    for o in hud:
        cell = max(3, min(12, 150 // max(o.h, o.w)))
        im = Image.new("RGB", (190, o.h * cell + 8), (60, 60, 70)); dd = ImageDraw.Draw(im); cg = o.colgrid(g)
        for y in range(o.h):
            for x in range(o.w):
                if cg[y, x] >= 0:
                    dd.rectangle([4 + x * cell, 4 + y * cell, 4 + (x + 1) * cell - 2, 4 + (y + 1) * cell - 2], fill=tuple(PAL[cg[y, x]]))
        role = roles.get((o.r0, o.c0))
        side_imgs.append(titled(im, f"{letter.get(o.id, '')}: {role[0] if role else short(o)}", size=12))
    side = vstack([text_block(["HUD"], 190)] + side_imgs)
    card = hstack([titled(img, "Briefing map: tags = guessed roles, dashed = same shape, yellow ring = held at midpoint"), side])
    vstack([card, text_block(lines, card.width, size=13)]).save(out_png)
    txt = "\n".join(lines); open(out_txt, "w").write(txt)
    return txt


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="X8 briefing card: screenshot -> briefing image + text")
    ap.add_argument("screenshot", help="screenshot of an ARC-AGI-3 game frame (png/jpg)")
    ap.add_argument("--actions", default="1,2,3,4",
                    help="available actions, comma-separated: 1,2,3,4 = arrow keys, 6 = click (default 1,2,3,4)")
    ap.add_argument("--out", default=None, help="output folder (default: next to the screenshot)")
    a = ap.parse_args()
    base = os.path.splitext(os.path.basename(a.screenshot))[0]
    out_dir = os.path.abspath(a.out or os.path.dirname(os.path.abspath(a.screenshot)))
    os.makedirs(out_dir, exist_ok=True)
    png, txt = os.path.join(out_dir, base + "_x8.png"), os.path.join(out_dir, base + "_x8.txt")
    print(run(a.screenshot, png, txt, tuple(int(x) for x in a.actions.split(","))))
    print(f"\nsaved: {png}\n       {txt}")
