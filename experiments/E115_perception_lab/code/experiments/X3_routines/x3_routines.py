"""
X3 -- Visual routines (Ullman 1984).

Relations such as reachability, inside/outside, "what is this line attached to", alignment are not
pre-attentive: humans compute them with serial routines (bounded activation = flood fill, curve tracing,
indexing, marking). VLMs fail exactly here (BlindTest). So we run the routines in code and show results.

Routines
 R1 REACH  bounded activation from the most self-like object. If the game has a tile unit, the fill
           steps by that unit (an object moves by its own size); otherwise plain open-space fill +
           straight-path (line-of-sight) test between candidates.
 R2 FIT    inside/outside: what sits inside which box; hollow figures and the size of their empty inside;
           which objects would fit exactly.
 R3 TRACE  curve tracing of thin lines: endpoints -> attached objects; geometric invariants among linked
           objects (midpoint, collinearity).
 R4 ALIGN  indexing: pairs of things sharing rows/columns, and whether the straight path between is clear.
"""
from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np
from PIL import ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scene import N, NAMES, build_scene, load_grid, to_image, font, lattice_unit, fill_holes  # noqa: E402
from common_out import is_hud, text_block, hstack, vstack, titled, zone  # noqa: E402

CELL = 6


def self_candidates(sc):
    c = [o for o in sc.objects if o.kind == "object" and not is_hud(o, sc) and len(o.colors) >= 2 and o.size >= 9]
    return sorted(c, key=lambda o: (-(o.h == o.w), -o.size))


def r1_reach(sc, lines, img):
    g = sc.g; d = ImageDraw.Draw(img)
    cands = self_candidates(sc)
    lu = lattice_unit(sc)
    if not cands:
        lines.append("R1 REACH: no self-like object found."); return None
    me = cands[0]
    floor = me.on
    occ = np.zeros(g.shape, bool)
    for o in sc.objects:
        if not is_hud(o, sc):
            for y, x in o.cells:
                occ[y, x] = True
    walk = (g == floor) | occ
    if lu:
        u, ro, co = lu
        start = (me.r0, me.c0)
        def ok(p):
            r, c = p
            return 0 <= r <= N - u and 0 <= c <= N - u and walk[r:r + u, c:c + u].all()
        dist = {start: 0}; q = deque([start])
        while q:
            p = q.popleft()
            for dr, dc in ((-u, 0), (u, 0), (0, -u), (0, u)):
                nb = (p[0] + dr, p[1] + dc)
                if nb not in dist and ok(nb):
                    dist[nb] = dist[p] + 1; q.append(nb)
        for (r, c), k in dist.items():
            d.rectangle([c * CELL, r * CELL, (c + u) * CELL - 1, (r + u) * CELL - 1], outline=(0, 200, 0), width=2)
            d.text((c * CELL + 6, r * CELL + 6), str(k), fill=(0, 120, 0), font=font(12, True))
        lines.append(f"R1 REACH: the most self-like thing is the {me.name} at rows {me.r0}-{me.r1}, cols {me.c0}-{me.c1}. "
                     f"It is {u}x{u} and the floor is built from {u}x{u} tiles, so it most likely moves one tile ({u} cells) per step. "
                     f"Flood-filling the {NAMES[floor]} floor in tile steps reaches {len(dist)} tiles (green numbers = steps away).")
        for o in sc.objects:
            if o is me or o.kind in ("line",):
                continue
            hit = [k for (r, c), k in dist.items() if any(r <= y < r + u and c <= x < c + u for y, x in o.cells)]
            if hit:
                lines.append(f"   - {o.name}: reachable in {min(hit)} steps")
            elif not is_hud(o, sc):
                # adjacent? (box entrance)
                adj = [(k, (r, c)) for (r, c), k in dist.items()
                       if (o.panel is not None and _touches(sc.panels[o.panel], r, c, u))]
                if adj:
                    k, (r, c) = min(adj)
                    side = _side(sc.panels[o.panel], r, c, u)
                    lines.append(f"   - {o.name}: sits inside a {NAMES[sc.panels[o.panel].color]} box that is NOT plain floor; "
                                 f"the floor reaches the box's {side} side in {k} steps (the only way in)")
                else:
                    lines.append(f"   - {o.name}: NOT reachable on the floor")
        return me
    # no lattice: open-space connectivity + straight-path test
    lines.append(f"R1 REACH: no tile grid -> free movement. Open space = {NAMES[floor]} ({(g == floor).mean():.0%} of screen).")
    sal = [o for o in sc.objects if not is_hud(o, sc) and o.kind != "line" and o.size >= 9]
    blocked = []
    for i, a in enumerate(sal):
        for b in sal[i + 1:]:
            clear, hits = _line_clear(g, a, b, floor, occ)
            col = (0, 200, 0) if clear else (220, 0, 0)
            d.line([(a.cx * CELL + CELL / 2, a.cy * CELL + CELL / 2), (b.cx * CELL + CELL / 2, b.cy * CELL + CELL / 2)], fill=col, width=2)
            if not clear:
                blocked.append(f"{a.name} -> {b.name} ({hits} rock cells)")
    lines.append(f"   - all {len(sal)} main objects share one open space; straight paths between them are "
                 + ("ALL CLEAR of rock" if not blocked else "clear except: " + "; ".join(blocked)))
    rock = [t for t in sc.terrain if t != floor]
    if rock:
        lines.append(f"   - the {', '.join(NAMES[t] for t in rock)} areas are solid rock/walls")
    return me


def _touches(p, r, c, u):
    """tile (r,c,u) is within one tile of the box and overlaps it in the other axis"""
    gap_r = max(p.r0 - (r + u - 1), r - p.r1, 0)
    gap_c = max(p.c0 - (c + u - 1), c - p.c1, 0)
    return (gap_r <= u and gap_c == 0) or (gap_c <= u and gap_r == 0)


def _side(p, r, c, u):
    if r >= p.r1: return "bottom"
    if r + u <= p.r0: return "top"
    return "left" if c + u <= p.c0 else "right"


def _line_clear(g, a, b, floor, occ):
    """sweep the smaller object's footprint along the straight segment; count rock cells hit."""
    hh, ww = min(a.h, b.h), min(a.w, b.w)
    hits = set()
    for t in np.linspace(0, 1, 60):
        cy = a.cy + (b.cy - a.cy) * t; cx = a.cx + (b.cx - a.cx) * t
        r0, c0 = int(round(cy - hh / 2 + 0.5)), int(round(cx - ww / 2 + 0.5))
        for y in range(r0, r0 + hh):
            for x in range(c0, c0 + ww):
                if 0 <= y < N and 0 <= x < N and g[y, x] != floor and not occ[y, x] and g[y, x] >= 0:
                    hits.add((y, x))
    return len(hits) == 0, len(hits)


def r2_fit(sc, lines, img):
    g = sc.g; d = ImageDraw.Draw(img)
    any_ = False
    for p in sc.panels:
        inside = [o for o in sc.objects if o.panel == p.id]
        if inside:
            any_ = True
            d.rectangle([p.c0 * CELL, p.r0 * CELL, (p.c1 + 1) * CELL - 1, (p.r1 + 1) * CELL - 1], outline=(255, 140, 0), width=2)
            where = "at the screen edge (HUD)" if (p.r0 <= 2 or p.c0 <= 2 or p.r1 >= N - 3 or p.c1 >= N - 3) else "inside the play area"
            lines.append(f"R2 FIT: a {NAMES[p.color]} box rows {p.r0}-{p.r1}, cols {p.c0}-{p.c1} {where} contains: "
                         + "; ".join(o.name for o in inside))
    for o in sc.objects:
        m = o.mask()
        if o.kind == "dotted" or (fill_holes(m).sum() > m.sum()):
            inner_h, inner_w = o.h - 2, o.w - 2
            others = {c for q in sc.objects if q is not o for c in q.cells}
            empty = not any((y, x) in others for y in range(o.r0 + 1, o.r1) for x in range(o.c0 + 1, o.c1))
            fits = [q for q in sc.objects if q is not o and q.h == inner_h and q.w == inner_w and q.kind == "object"]
            if inner_h >= 2:
                any_ = True
                d.rectangle([(o.c0 + 1) * CELL, (o.r0 + 1) * CELL, o.c1 * CELL - 1, o.r1 * CELL - 1], outline=(255, 0, 255), width=2)
                lines.append(f"R2 FIT: the {o.name} encloses a {'EMPTY ' if empty else ''}{inner_h}x{inner_w} space"
                             + (f" -- exactly the size of: {', '.join(q.name for q in fits)}" if fits else ""))
    if not any_:
        lines.append("R2 FIT: nothing encloses anything.")


def r3_trace(sc, lines, img):
    d = ImageDraw.Draw(img)
    links = []
    for ln in [o for o in sc.objects if o.kind == "line"]:
        s = set(ln.cells)
        ends = [c for c in ln.cells if sum((c[0] + dy, c[1] + dx) in s for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx) <= 1]
        att = []
        for e in ends[:2]:
            best = min((o for o in sc.objects if o.kind != "line" and o is not ln),
                       key=lambda o: max(o.r0 - e[0], 0, e[0] - o.r1) + max(o.c0 - e[1], 0, e[1] - o.c1))
            att.append(best)
        if len(att) == 2:
            links.append((att[0], att[1]))
            lines.append(f"R3 TRACE: a {ln.name} connects the {att[0].name} to the {att[1].name}")
            d.line([(att[0].cx * CELL + 3, att[0].cy * CELL + 3), (att[1].cx * CELL + 3, att[1].cy * CELL + 3)], fill=(0, 255, 255), width=3)
    if not links:
        lines.append("R3 TRACE: no connecting lines on screen."); return
    # invariants: an object linked to exactly two others
    from collections import defaultdict
    deg = defaultdict(list)
    for a, b in links:
        deg[a.id].append(b); deg[b.id].append(a)
    for oid, nbs in deg.items():
        if len(nbs) == 2:
            o = sc.objects[oid]; a, b = nbs
            my, mx = (a.cy + b.cy) / 2, (a.cx + b.cx) / 2
            err = float(np.hypot(o.cy - my, o.cx - mx))
            if err <= 1.0:
                lines.append(f"R3 TRACE: the {o.name} sits at the MIDPOINT of the {a.name} and the {b.name} "
                             f"(error {err:.1f} cells) -> it is held between them; moving either end should move it to the new midpoint")
                d.ellipse([mx * CELL - 6, my * CELL - 6, mx * CELL + 12, my * CELL + 12], outline=(255, 255, 0), width=3)


def r4_align(sc, lines, img):
    d = ImageDraw.Draw(img)
    sal = [o for o in sc.objects if not is_hud(o, sc) and o.kind not in ("line",)]
    g = sc.g
    out = 0
    for i, a in enumerate(sal):
        for b in sal[i + 1:]:
            colov = min(a.c1, b.c1) - max(a.c0, b.c0) + 1
            rowov = min(a.r1, b.r1) - max(a.r0, b.r0) + 1
            if colov >= 0.6 * min(a.w, b.w):
                rel = "directly above" if a.cy < b.cy else "directly below"
                lines.append(f"R4 ALIGN: the {a.name} is {rel} the {b.name} (shared columns {max(a.c0, b.c0)}-{min(a.c1, b.c1)})")
                d.line([(max(a.c0, b.c0) * CELL, a.cy * CELL), (max(a.c0, b.c0) * CELL, b.cy * CELL)], fill=(255, 255, 0), width=2); out += 1
            elif rowov >= 0.6 * min(a.h, b.h):
                rel = "left of" if a.cx < b.cx else "right of"
                lines.append(f"R4 ALIGN: the {a.name} is level with and {rel} the {b.name} (shared rows {max(a.r0, b.r0)}-{min(a.r1, b.r1)})")
                d.line([(a.cx * CELL, max(a.r0, b.r0) * CELL), (b.cx * CELL, max(a.r0, b.r0) * CELL)], fill=(255, 255, 0), width=2); out += 1
    if not out:
        lines.append("R4 ALIGN: no two things line up.")


def run(path, out_png, out_txt, actions=None):
    g, notes = load_grid(path)
    sc = build_scene(g, notes)
    panels = []
    for title, fn in [("R1 reach (where can the self go?)", r1_reach), ("R2 fit (what is inside what?)", r2_fit),
                      ("R3 trace (what is connected?)", r3_trace), ("R4 align (what lines up?)", r4_align)]:
        img = to_image(g, cell=CELL, dim=0.35)
        ls: list[str] = []
        fn(sc, ls, img)
        panels.append((titled(img, title), ls))
    allines = ["VISUAL ROUTINES"] + [l for _, ls in panels for l in ls]
    grid = vstack([hstack([panels[0][0], panels[1][0]]), hstack([panels[2][0], panels[3][0]])])
    vstack([grid, text_block(allines, grid.width, size=13)]).save(out_png)
    txt = "\n".join(allines); open(out_txt, "w").write(txt)
    return txt


if __name__ == "__main__":
    print(run(sys.argv[1], sys.argv[2], sys.argv[3]))
