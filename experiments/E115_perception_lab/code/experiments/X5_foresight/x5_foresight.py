"""
X5 -- Foresight strip (imagined next frames from one screenshot).

Basis: visual foresight / mental simulation (humans imagine outcomes before acting; EMPA arXiv 2107.12544
plans by simulating its theory). Core-knowledge priors only:
  P1 an agent moves by its own size along the floor grid and stops at walls
  P2 the longest HUD bar is a budget that drops one unit per action
  P3 (click games) of two same-shaped objects, the brighter one is the selected one; clicking empty space
     sends the selected one there; clicking the other one selects it
  P4 a thing held at the midpoint of two anchors stays at their midpoint (constraint found by X3 R3)
Output: strip of imagined frames + predicted consequences. The engine is used ONLY afterwards to score
the predictions (written to a separate *_verify.txt that a reader would not see).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "X3_routines"))
from scene import N, NAMES, build_scene, load_grid, to_image, font, lattice_unit  # noqa: E402
from common_out import is_hud, text_block, hstack, vstack, titled  # noqa: E402
from x3_routines import self_candidates  # noqa: E402

CELL = 5
DIRS = {1: ("UP", -1, 0), 2: ("DOWN", 1, 0), 3: ("LEFT", 0, -1), 4: ("RIGHT", 0, 1)}  # ARC-AGI-3 key convention


def budget_bar(sc):
    bars = [o for o in sc.objects if o.kind == "bar"]
    return max(bars, key=lambda o: o.size) if bars else None


def shrink_bar(g, bar, units=1):
    g = g.copy()
    if bar is None:
        return g
    vertical = bar.h > bar.w
    cells = sorted(bar.cells, key=lambda c: c[0] if vertical else -c[1])  # vertical drains from top, horizontal from right
    thick = bar.w if vertical else bar.h
    under = 5
    for y, x in cells[: thick * units]:
        g[y, x] = under if vertical else 5
    return g


def keyboard(sc, g, lines, panels, actions):
    me = self_candidates(sc)[0]
    lu = lattice_unit(sc)
    u = lu[0] if lu else me.h
    floor = me.on
    occ = np.zeros(g.shape, bool)
    for o in sc.objects:
        if not is_hud(o, sc):
            for y, x in o.cells: occ[y, x] = True
    bar = budget_bar(sc)
    preds = {}
    lines.append(f"FORESIGHT: you are probably the {me.name}; it should move {u} cells per key press (its own size).")
    for a in actions:
        if a not in DIRS: continue
        nm, dy, dx = DIRS[a]
        r, c = me.r0 + dy * u, me.c0 + dx * u
        ok = 0 <= r <= N - me.h and 0 <= c <= N - me.w and all(
            (g[y, x] == floor or occ[y, x]) for y in range(r, r + me.h) for x in range(c, c + me.w))
        ng = g.copy()
        touched = []
        if ok:
            for y, x in me.cells: ng[y, x] = floor
            for y, x in me.cells: ng[y + dy * u, x + dx * u] = g[y, x]
            touched = [o.name for o in sc.objects if o is not me and not is_hud(o, sc)
                       and any(r <= y < r + me.h and c <= x < c + me.w for y, x in o.cells)]
        ng = shrink_bar(ng, bar)
        preds[a] = {"pos": (r, c) if ok else (me.r0, me.c0), "moved": ok, "bar_cells": None if bar is None else bar.size - (bar.h if bar.w > bar.h else bar.w)}
        msg = f"  ACTION{a} ({nm}): " + (f"moves to rows {r}-{r + me.h - 1}, cols {c}-{c + me.w - 1}" if ok else "BLOCKED by wall, stays put")
        if touched: msg += f"; lands ON the {', '.join(touched)} -> expect something to happen"
        if bar is not None: msg += f"; {NAMES[bar.colors.most_common(1)[0][0]]} bar shrinks by one unit"
        lines.append(msg)
        img = to_image(ng, cell=CELL)
        d = ImageDraw.Draw(img)
        d.rectangle([c * CELL, r * CELL, (c + me.w) * CELL - 1, (r + me.h) * CELL - 1], outline=(0, 255, 0) if ok else (255, 0, 0), width=2)
        panels.append(titled(img, f"if ACTION{a} {nm}" + ("" if ok else " (blocked)")))
    return preds, me


def bresenham(r0, c0, r1, c1):
    pts = []; dr, dc = abs(r1 - r0), abs(c1 - c0); sr, scn = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
    err = dc - dr; r, c = r0, c0
    while True:
        pts.append((r, c))
        if (r, c) == (r1, c1): break
        e2 = 2 * err
        if e2 > -dr: err -= dr; c += scn
        if e2 < dc: err += dc; r += sr
    return pts


def click(sc, g, lines, panels):
    sys.path.insert(0, os.path.join(HERE, "..", "X4_threads"))
    from x4_threads import relate
    objs = [o for o in sc.objects if o.kind in ("object", "icon") and not is_hud(o, sc)]
    pair = None
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            rels, _, _ = relate(a, b)
            if rels and rels[0][0] == "STATE":
                pair = (a, b)
    lines.append("FORESIGHT (click game): imagined effects of clicks under simple UI priors.")
    if not pair:
        lines.append("  no selectable pair found; prior: clicking a distinct object probably selects/toggles it."); return {}, None
    lum = lambda o: max(sum(int(v) for v in __import__('scene').PAL[c]) for c in o.colors)
    sel, other = sorted(pair, key=lum, reverse=True)
    lines.append(f"  the {sel.name} looks SELECTED (brighter twin of the {other.name}).")
    # held object at midpoint
    held = [o for o in objs if o not in pair and np.hypot(o.cy - (sel.cy + other.cy) / 2, o.cx - (sel.cx + other.cx) / 2) <= 1.0]
    target = [o for o in sc.objects if o.kind == "dotted"]
    if not held or not target:
        lines.append("  no held object / target found."); return {}, None
    held, target = held[0], target[0]
    tc = ((target.r0 + target.r1) / 2, (target.c0 + target.c1) / 2)
    floor = held.on
    rock = set(sc.terrain) - {floor}
    free = lambda y, x: 0 <= y < N and 0 <= x < N and g[y, x] not in rock
    best = None
    for vr in range(-20, 21):
        for vc in range(-20, 21):
            A = (tc[0] + vr, tc[1] + vc); B = (tc[0] - vr, tc[1] - vc)
            if not all(2 <= p[0] <= N - 3 and 3 <= p[1] <= N - 3 for p in (A, B)): continue
            if not all(free(int(p[0]) + dy, int(p[1]) + dx) for p in (A, B) for dy in (-2, -1, 0, 1, 2) for dx in (-2, -1, 0, 1, 2)): continue
            mid1 = ((A[0] + other.cy) / 2, (A[1] + other.cx) / 2)
            path = [(held.cy + (mid1[0] - held.cy) * t, held.cx + (mid1[1] - held.cx) * t) for t in np.linspace(0, 1, 30)] + \
                   [(mid1[0] + (tc[0] - mid1[0]) * t, mid1[1] + (tc[1] - mid1[1]) * t) for t in np.linspace(0, 1, 30)]
            if not all(free(int(round(y)) + dy, int(round(x)) + dx) for y, x in path for dy in (-2, 0, 2) for dx in (-2, 0, 2)):
                continue
            cost = np.hypot(A[0] - sel.cy, A[1] - sel.cx) + np.hypot(B[0] - other.cy, B[1] - other.cx)
            if best is None or cost < best[0]:
                best = (cost, (int(A[0]), int(A[1])), (int(B[0]), int(B[1])), mid1)
    if not best:
        lines.append("  no clear 3-click plan found."); return {}, None
    _, A, B, mid1 = best
    plan = [("click empty spot", A, "selected handle moves there; held disc slides to the new midpoint"),
            ("click the other handle", (int(other.cy), int(other.cx)), "it becomes the selected one"),
            ("click empty spot", B, "it moves there; disc lands on the midpoint = centre of the dotted ring")]
    lines.append(f"  the {held.name} sits at the midpoint of the two handles, so: imagined 3-click plan:")
    pos = {"sel": (sel.cy, sel.cx), "oth": (other.cy, other.cx)}
    preds = {}
    for k, (what, (r, c), eff) in enumerate(plan):
        if k == 0: pos["sel"] = (r, c)
        if k == 2: pos["oth"] = (r, c)
        disc = ((pos["sel"][0] + pos["oth"][0]) / 2, (pos["sel"][1] + pos["oth"][1]) / 2)
        preds[k] = {"click": (r, c), "disc": disc}
        lines.append(f"   {k + 1}. {what} at row {r}, col {c} -> {eff}; disc centre predicted at ({disc[0]:.0f},{disc[1]:.0f})")
        # imagined frame
        ng = g.copy()
        for o in (sel, other, held):
            for y, x in o.cells: ng[y, x] = floor
        for ln in [o for o in sc.objects if o.kind == "line"]:
            for y, x in ln.cells: ng[y, x] = floor
        def stamp(o, ctr):
            for y, x in o.cells:
                yy, xx = int(round(y - o.cy + ctr[0])), int(round(x - o.cx + ctr[1]))
                if 0 <= yy < N and 0 <= xx < N: ng[yy, xx] = g[y, x]
        for p in (pos["sel"], pos["oth"]):
            for y, x in bresenham(int(p[0]), int(p[1]), int(disc[0]), int(disc[1])):
                if ng[y, x] == floor: ng[y, x] = 1
        stamp(sel, pos["sel"]); stamp(other, pos["oth"]); stamp(held, disc)
        img = to_image(ng, cell=CELL)
        dd = ImageDraw.Draw(img)
        dd.ellipse([c * CELL - 6, r * CELL - 6, c * CELL + 10, r * CELL + 10], outline=(255, 255, 0), width=2)
        panels.append(titled(img, f"imagined after click {k + 1}"))
    return preds, held


def run(path, out_png, out_txt, actions=(1, 2, 3, 4), verify_game=None):
    g, notes = load_grid(path)
    sc = build_scene(g, notes)
    lines, panels = [], []
    if 6 in actions and not any(a in DIRS for a in actions):
        preds, held = click(sc, g, lines, panels); mode = "click"
    else:
        preds, me = keyboard(sc, g, lines, panels, actions); mode = "key"
    strip = hstack(panels[:4]) if panels else to_image(g)
    vstack([strip, text_block(lines, max(strip.width, 700), size=13)]).save(out_png)
    txt = "\n".join(lines); open(out_txt, "w").write(txt)
    if verify_game:
        v = verify(verify_game, mode, preds, sc)
        open(out_txt.replace(".txt", "_verify.txt"), "w").write(v)
        txt += "\n\n[VERIFY vs engine]\n" + v
    return txt


def verify(game, mode, preds, sc):
    import logging; logging.disable(logging.INFO)
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=os.environ.get("ARC_GAMES", "/home/claude/repos/arc-prize-2026-arc-agi-3/arc_games"))
    out = []
    if mode == "key":
        ok = 0
        for a, p in preds.items():
            env = arc.make(game); env.step(GameAction.RESET)
            f = np.array(env.step(getattr(GameAction, f'ACTION{a}')).frame)[-1]
            me = self_candidates(sc)[0]
            col = me.colors.most_common(1)[0][0]
            # locate the self by its unique top colour
            top = [c for c in me.colors if c != col] or [col]
            ys, xs = np.nonzero(f == top[0])
            actual = (int(ys.min()), int(xs.min())) if len(ys) else None
            predicted = p["pos"]
            hit = actual == predicted
            bar = budget_bar(sc)
            bar_ok = None
            if bar is not None:
                bc = bar.colors.most_common(1)[0][0]
                bar_ok = int((f == bc).sum()) == p["bar_cells"]
            ok += hit
            out.append(f"ACTION{a}: predicted self top-left {predicted}, engine {actual} -> {'HIT' if hit else 'MISS'}; bar prediction {'HIT' if bar_ok else 'MISS'}")
        out.append(f"position accuracy {ok}/{len(preds)}")
    else:
        env = arc.make(game); env.step(GameAction.RESET)
        for k in sorted(preds):
            r, c = preds[k]["click"]
            o = env.step(GameAction.ACTION6, data={"x": int(c), "y": int(r)})
            f = np.array(o.frame)[-1]
            ys, xs = np.nonzero(f == 6)
            actual = (float(ys.mean()), float(xs.mean())) if len(ys) else None
            pr = preds[k]["disc"]
            err = None if actual is None else float(np.hypot(actual[0] - pr[0], actual[1] - pr[1]))
            out.append(f"click {k + 1} at ({r},{c}): predicted disc ({pr[0]:.1f},{pr[1]:.1f}), engine {actual} -> error {err if err is None else round(err, 1)} cells; levels_completed={o.levels_completed}")
    return "\n".join(out)


if __name__ == "__main__":
    acts = tuple(int(a) for a in sys.argv[4].split(",")) if len(sys.argv) > 4 else (1, 2, 3, 4)
    print(run(sys.argv[1], sys.argv[2], sys.argv[3], acts, sys.argv[5] if len(sys.argv) > 5 else None))
