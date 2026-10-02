"""
X2 -- Surprise map (perception as "what can't be predicted").

Basis: Bruce & Tsotsos AIM (NeurIPS 2005): saliency = self-information -log p(local feature);
Treisman pop-out; MDL/compression (CompressARC arXiv 2512.06104): structure = what compresses,
meaning lives in the residual.

Mechanism:
 1. every cell gets its 3x3 neighbourhood pattern; p(pattern) is its frequency in this frame;
    surprise(cell) = -log2 p. Repeated texture (floor, walls, tiles) scores near 0 bits.
 2. objects are ranked by total surprise (bits) -> "look here first" order.
 3. symmetry-break test: object whose SHAPE is symmetric but whose COLOURING is not -> flagged,
    because designers use broken symmetry to encode direction/state.
Output: faded scene where only the unpredictable residue keeps full colour, numbered by rank.
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scene import N, NAMES, build_scene, load_grid, to_image, font, d4  # noqa: E402
from common_out import zone, is_hud, text_block, hstack, titled  # noqa: E402

CELL = 9


def surprise_map(g: np.ndarray) -> np.ndarray:
    pad = np.pad(g, 1, constant_values=-2)
    pats = np.empty(g.shape, dtype=object)
    for r in range(N):
        for c in range(N):
            pats[r, c] = pad[r:r + 3, c:c + 3].tobytes()
    cnt = Counter(pats.ravel().tolist())
    tot = g.size
    return np.vectorize(lambda p: -math.log2(cnt[p] / tot))(pats).astype(float)


def symmetry_break(o, g):
    m = o.mask(); cg = o.colgrid(g)
    if len(o.colors) < 2:
        return None
    ops = {"mirror left-right": lambda a: np.fliplr(a), "mirror top-bottom": lambda a: np.flipud(a)}
    if m.shape[0] == m.shape[1]:
        ops["quarter turn"] = lambda a: np.rot90(a)
    broken = [nm for nm, f in ops.items() if np.array_equal(f(m), m) and not np.array_equal(f(cg), cg)]
    if not broken:
        return None
    minority = min(o.colors, key=o.colors.get)
    ys, xs = np.nonzero(cg == minority)
    cy, cx = (m.shape[0] - 1) / 2, (m.shape[1] - 1) / 2
    vy, vx = ys.mean() - cy, xs.mean() - cx
    side = []
    if vy < -0.2: side.append("top")
    if vy > 0.2: side.append("bottom")
    if vx < -0.2: side.append("left")
    if vx > 0.2: side.append("right")
    return (f"shape is symmetric ({', '.join(broken)}) but colours are not: the odd colour "
            f"({NAMES[minority]}) sits on the {'-'.join(side) or 'centre'} side -> it encodes a direction or state")


def run(path, out_png, out_txt, actions=None):
    g, notes = load_grid(path)
    sc = build_scene(g, notes)
    S = surprise_map(g)
    predictable = (S < math.log2(g.size / 8)).mean()   # pattern seen >= 8 times = predictable texture
    ranked = []
    for o in sc.objects:
        bits = sum(S[y, x] for y, x in o.cells)
        ranked.append((bits, o))
    ranked.sort(key=lambda t: -t[0])
    tot_bits = S.sum()
    # image: fade everything, restore colour where surprise is high or on ranked objects
    fade = np.clip(1 - (S - S.min()) / (np.percentile(S, 97) - S.min() + 1e-9), 0, 1) * 0.85
    for _, o in ranked[:8]:
        for y, x in o.cells:
            fade[y, x] = 0.0
    img = to_image(g, cell=CELL, dim=fade)
    d = ImageDraw.Draw(img)
    for k, (b, o) in enumerate(ranked[:8]):
        x, y = (o.c1 + 1) * CELL + 2, max(0, o.r0 * CELL - 4)
        d.rectangle([o.c0 * CELL - 2, o.r0 * CELL - 2, (o.c1 + 1) * CELL + 1, (o.r1 + 1) * CELL + 1], outline=(255, 40, 40), width=2)
        d.text((x, y), str(k + 1), fill=(255, 255, 0), font=font(16, True), stroke_width=2, stroke_fill=(0, 0, 0))
    heat = (np.clip(S / S.max(), 0, 1) * 255).astype(np.uint8)
    heat_img = Image.fromarray(np.stack([heat, (heat * 0.3).astype(np.uint8), 255 - heat], -1)).resize((N * 4, N * 4), Image.NEAREST)
    lines = [f"SURPRISE MAP - {predictable:.0%} of the screen is predictable texture; the rest, ranked by information:"]
    for k, (b, o) in enumerate(ranked[:8]):
        tag = "HUD" if is_hud(o, sc) else zone(o.cy, o.cx)
        lines.append(f"{k + 1}. {o.name} ({o.h}x{o.w}) at rows {o.r0}-{o.r1}, cols {o.c0}-{o.c1} [{tag}] - {b:.0f} bits ({b / tot_bits:.0%} of all surprise)")
    sb = [(o, symmetry_break(o, g)) for _, o in ranked]
    sb = [(o, s) for o, s in sb if s]
    if sb:
        lines.append("Broken symmetries:")
        for o, s in sb:
            lines.append(f"- {o.name}: {s}")
    panel = text_block(lines, 560, size=14)
    out = hstack([titled(img, "Only the surprising parts keep their colour (numbers = look-here-first order)"),
                  titled(heat_img, "raw surprise heat (red = rare)")])
    from common_out import vstack
    vstack([out, panel]).save(out_png)
    txt = "\n".join(lines); open(out_txt, "w").write(txt)
    return txt


if __name__ == "__main__":
    print(run(sys.argv[1], sys.argv[2], sys.argv[3]))
