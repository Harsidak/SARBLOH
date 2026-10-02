"""
X4 -- Correspondence threads (analogy finder).

Basis: ARC games state goals as "make A match B"; Core Knowledge geometry (rotation, symmetry, scale);
Visual Sketchpad (arXiv 2406.09403): auxiliary lines drawn between things carry reasoning.

Mechanism: for every pair of objects test equality of binary shape under the 8 rotations/flips and
integer scale 1..3 (block-downsampling). Report:
  EXACT  same shape as drawn (copies)            -> counters, repeated tiles, same type
  TURNED same shape after a rotation/flip/scale  -> "A turned 90 deg clockwise becomes B" (goal hint)
  NEAR   same size, a few cells differ           -> "what must change"
  STATE  same shape, different colour            -> same kind of thing in a different state
  COLOUR a distinctive colour shared by several objects -> they belong together
Threads are drawn between the matched objects with the relation written on them.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
from PIL import ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scene import NAMES, build_scene, load_grid, to_image, font, d4  # noqa: E402
from common_out import is_hud, text_block, vstack, titled  # noqa: E402

CELL = 10
T_WORDS = {"same": "as is", "rot90ccw": "turned 90 deg anticlockwise", "rot180": "turned 180 deg",
           "rot90cw": "turned 90 deg clockwise", "flipLR": "mirrored left-right",
           "rot90ccw+flipLR": "turned 90 deg anticlockwise then mirrored", "rot180+flipLR": "mirrored top-bottom",
           "rot90cw+flipLR": "turned 90 deg clockwise then mirrored"}


def downsample(m: np.ndarray, s: int):
    h, w = m.shape
    if h % s or w % s:
        return None
    b = m.reshape(h // s, s, w // s, s)
    if not ((b.all(axis=(1, 3))) | (~b.any(axis=(1, 3)))).all():
        return None
    return b.all(axis=(1, 3))


def relate(a, b):
    """returns list of (kind, text, transform) describing how a relates to b"""
    out = []
    ma, mb = a.mask(), b.mask()
    big, small, flip = (a, b, False) if ma.size >= mb.size else (b, a, True)
    mbig, msmall = big.mask(), small.mask()
    for s in (1, 2, 3):
        dm = downsample(mbig, s) if s > 1 else mbig
        if dm is None or dm.shape not in (msmall.shape, msmall.shape[::-1]):
            continue
        matches = [nm for nm, t in d4(dm) if t.shape == msmall.shape and np.array_equal(t, msmall)]
        scale_txt = "" if s == 1 else f" (it is drawn {s}x bigger)"
        if matches:
            same_cols = set(a.colors) == set(b.colors)
            if "same" in matches and s == 1:
                kind = "EXACT" if same_cols else "STATE"
                out.append((kind, "same shape" + ("" if same_cols else f", different colour ({'/'.join(NAMES[c] for c in a.colors)} vs {'/'.join(NAMES[c] for c in b.colors)})"), "same"))
            else:
                ts = [m for m in matches if m != "same"] if "same" not in matches else ["same"]
                out.append(("TURNED", f"the {big.name}{scale_txt}, {' OR '.join(T_WORDS[t] for t in ts)}, becomes exactly the {small.name}", ts[0]))
            if s > 1 or "same" not in matches:
                if dm.shape == msmall.shape:
                    diff = int((dm != msmall).sum())
                    if diff:
                        out.append(("NEAR", f"as currently drawn they differ in {diff} of {msmall.size} cells", None))
            return out, big, small
    return out, big, small


def run(path, out_png, out_txt, actions=None):
    g, notes = load_grid(path)
    sc = build_scene(g, notes)
    objs = [o for o in sc.objects if o.kind in ("object", "icon", "dotted") and o.size >= 2]
    img = to_image(g, cell=CELL, dim=0.25)
    d = ImageDraw.Draw(img)
    lines = ["LOOK-ALIKE THREADS (things that are the same up to move / turn / mirror / resize / recolour)"]
    palette_cols = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 128, 0), (255, 80, 80)]
    k = 0
    copies = defaultdict(list)
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            rels, big, small = relate(a, b)
            if not rels:
                continue
            if rels[0][0] == "EXACT":
                copies[(a.name, a.h, a.w)].extend([a, b]); continue
            col = palette_cols[k % len(palette_cols)]; k += 1
            d.line([(a.cx * CELL + 5, a.cy * CELL + 5), (b.cx * CELL + 5, b.cy * CELL + 5)], fill=col, width=3)
            for o in (a, b):
                d.rectangle([o.c0 * CELL - 2, o.r0 * CELL - 2, (o.c1 + 1) * CELL + 1, (o.r1 + 1) * CELL + 1], outline=col, width=2)
            mx, my = (a.cx + b.cx) / 2 * CELL, (a.cy + b.cy) / 2 * CELL
            d.text((mx + 6, my), f"#{k}", fill=col, font=font(18, True), stroke_width=2, stroke_fill=(0, 0, 0))
            where = lambda o: "HUD" if is_hud(o, sc) else f"rows {o.r0}-{o.r1}, cols {o.c0}-{o.c1}"
            lines.append(f"#{k} {rels[0][0]}: {rels[0][1]}  [{big.name} @ {where(big)}  <->  {small.name} @ {where(small)}]")
            for r in rels[1:]:
                lines.append(f"     {r[0]}: {r[1]}")
    for (nm, h, w), lst in copies.items():
        uniq = {id(o): o for o in lst}.values()
        lines.append(f"COPIES: {len(uniq)} identical {nm}s ({h}x{w}) -> a count of something (lives, items, tiles)")
    # colour links
    col_owners = defaultdict(list)
    for o in objs:
        if is_hud(o, sc) and o.kind == "bar":
            continue
        for c in o.colors:
            col_owners[c].append(o)
    for c, lst in col_owners.items():
        if len(lst) >= 3 and len({o.colgrid(g).tobytes() for o in lst}) > 1:
            lines.append(f"COLOUR LINK: {NAMES[c]} appears in {len(lst)} different things: " + "; ".join(o.name for o in lst)
                         + " -> they likely belong together")
    if len(lines) == 1:
        lines.append("no correspondences found")
    vstack([titled(img, "Threads join things that are the same shape (see list)"), text_block(lines, img.width, size=14)]).save(out_png)
    txt = "\n".join(lines); open(out_txt, "w").write(txt)
    return txt


if __name__ == "__main__":
    print(run(sys.argv[1], sys.argv[2], sys.argv[3]))
