"""
X1 -- Exploded inventory.

Basis: "VLMs are blind" (arXiv 2407.06581): VLMs fail when shapes are small/close, succeed when
shapes are separated; the vision encoder has the information, the decoder does not verbalise it.
V* (arXiv 2312.14135): small details need dedicated enlarged views.

Mechanism: pull every object out of the scene, enlarge it to a common readable size, space them apart,
stack identical copies as "xN", and give each a thumbnail showing where it lives.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scene import N, NAMES, PAL, build_scene, load_grid, to_image, font  # noqa: E402
from common_out import zone, is_hud, text_block, vstack  # noqa: E402

TILE = 150


def obj_tile(o, g, letter, count, sc) -> Image.Image:
    cg = o.colgrid(g)
    cell = max(6, min(26, 110 // max(o.h, o.w)))
    W = max(TILE, o.w * cell + 20)
    img = Image.new("RGB", (W, 245), (250, 250, 247))
    d = ImageDraw.Draw(img)
    ox, oy = (W - o.w * cell) // 2, 28
    for y in range(o.h):
        for x in range(o.w):
            if cg[y, x] >= 0:
                d.rectangle([ox + x * cell, oy + y * cell, ox + (x + 1) * cell - 1, oy + (y + 1) * cell - 1],
                            fill=tuple(PAL[cg[y, x]]), outline=(90, 90, 90))
            else:  # empty part of bounding box: light checker so holes read as holes
                c = (232, 232, 228) if (x + y) % 2 else (242, 242, 238)
                d.rectangle([ox + x * cell, oy + y * cell, ox + (x + 1) * cell - 1, oy + (y + 1) * cell - 1], fill=c)
    d.text((6, 4), f"{letter}" + (f"  x{count}" if count > 1 else ""), fill=(180, 20, 20), font=font(18, True))
    # location thumbnail
    th = to_image(g, cell=1).resize((64, 64), Image.NEAREST)
    th = Image.eval(th, lambda v: int(v * 0.45 + 120))
    td = ImageDraw.Draw(th)
    td.rectangle([o.c0 - 1, o.r0 - 1, o.c1 + 1, o.r1 + 1], outline=(255, 0, 0))
    th = th.resize((96, 96), Image.NEAREST)
    img.paste(th, ((W - 96) // 2, 140))
    return img


def run(path: str, out_png: str, out_txt: str, actions=None):
    g, notes = load_grid(path)
    sc = build_scene(g, notes)
    # dedupe identical objects (same colour grid)
    types = []
    for o in sc.objects:
        if o.kind == "line":
            continue
        key = o.colgrid(g).tobytes() + bytes(str(o.colgrid(g).shape), "ascii")
        for t in types:
            if t["key"] == key:
                t["members"].append(o); break
        else:
            types.append({"key": key, "members": [o]})
    tiles, lines = [], []
    for k, t in enumerate(types):
        o = t["members"][0]; letter = chr(65 + k); n = len(t["members"])
        tiles.append(obj_tile(o, g, letter, n, sc))
        where = "; ".join(f"rows {m.r0}-{m.r1}, cols {m.c0}-{m.c1}" for m in t["members"][:4])
        place = "screen edge / HUD" if is_hud(o, sc) else (f"inside a {NAMES[sc.panels[o.panel].color]} box" if o.panel is not None else f"on {NAMES[o.on]} ground")
        lines.append(f"{letter}{' x'+str(n) if n > 1 else ''}: {o.name} ({o.h}x{o.w} cells) -- {zone(o.cy, o.cx)}, {place}; {where}")
    tethers = [o for o in sc.objects if o.kind == "line"]
    for o in tethers:
        lines.append(f"thin {o.name} from ({o.r0},{o.c0}) to ({o.r1},{o.c1})")
    terrain = ", ".join(f"{NAMES[t]} ({(g == t).mean():.0%})" for t in sc.terrain)
    head = [f"EXPLODED INVENTORY - {len(types)} kinds of thing on screen", f"Background/terrain: {terrain}"]
    # layout tiles into rows
    rows, cur, curw = [], [], 0
    for t in tiles:
        if curw + t.width > 900 and cur:
            rows.append(cur); cur, curw = [], 0
        cur.append(t); curw += t.width + 16
    rows.append(cur)
    row_imgs = []
    for r in rows:
        W = sum(t.width for t in r) + 16 * (len(r) - 1)
        im = Image.new("RGB", (W, 245), (250, 250, 247)); x = 0
        for t in r:
            im.paste(t, (x, 0)); x += t.width + 16
        row_imgs.append(im)
    sheet = vstack([text_block(head, 900)] + row_imgs + [text_block(["Legend"] + lines, 900, size=14)])
    sheet.save(out_png)
    txt = "\n".join(head + lines)
    open(out_txt, "w").write(txt)
    return txt


if __name__ == "__main__":
    print(run(sys.argv[1], sys.argv[2], sys.argv[3]))
