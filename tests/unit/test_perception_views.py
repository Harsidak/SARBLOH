# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh")   # PRIME_DIR=Sarbloh-Experimentation tests the copy
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""E008 perception on hand-made grids: the PNG (size, decodes, palette), the letter grid (labels, colours 10-15 keep
columns aligned), segmentation (ids, shape hash, corners, containment, adjacency, HUD band), the tracker's change line
(moved, recoloured, vanished, appeared, stable ids) and the scene text (crop margin, budget, level start).
"""
import importlib.util
import struct
import sys
import zlib

if importlib.util.find_spec("harness.agent.intuition") is None:
    print(f"SKIP: {_PRIME} replaced intuition.py and vision.py with perception.py (E116): "
          "tests/unit/test_e116_perception.py covers it")
    print("\n================ SUMMARY ================\nCHECKS: 0\nHARD FAILS: 0 -> []")
    sys.exit(0)

from harness.agent import intuition as I
from harness.agent import vision as V

FAILS, CHECKS = [], [0]


def ok(name, cond, detail=""):
    CHECKS[0] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def blank(h=16, w=16, v=0):
    return [[v] * w for _ in range(h)]


def fill(g, r0, c0, r1, c1, v):
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            g[r][c] = v
    return g


def png_pixels(png):
    """Decode our own PNG (RGB8, filter 0) back to rows of RGB tuples."""
    w, h = V.decode_png_size(png)
    pos, data = 8, b""
    while pos < len(png):
        n = struct.unpack(">I", png[pos:pos + 4])[0]
        kind = png[pos + 4:pos + 8]
        if kind == b"IDAT":
            data += png[pos + 8:pos + 8 + n]
        pos += 12 + n
    raw = zlib.decompress(data)
    stride = 1 + 3 * w
    return [[tuple(raw[y * stride + 1 + 3 * x: y * stride + 4 + 3 * x]) for x in range(w)] for y in range(h)]


# --- 1. image ------------------------------------------------------------------------------------------------
g = fill(blank(64, 64), 10, 20, 12, 22, 8)
png = V.render_png(g, upscale=4)
ok("png_256x256", V.decode_png_size(png) == (256, 256))
px = png_pixels(png)
ok("png_decodes_palette", px[0][0] == V.PALETTE[0] and px[10 * 4][20 * 4] == V.PALETTE[8]
   and px[12 * 4 + 3][22 * 4 + 3] == V.PALETTE[8] and px[13 * 4][20 * 4] == V.PALETTE[0])
ok("palette_16", len(V.PALETTE) == 16 and len(set(V.PALETTE)) == 16)
msg = V.image_message(g, "hello", step=3)
ok("image_message_shape", msg["role"] == "user" and msg["_image_step"] == 3
   and msg["content"][0] == {"type": "text", "text": "hello"}
   and msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))

# --- 2. letters ----------------------------------------------------------------------------------------------
g = blank(3, 12)
for c in range(12):
    g[1][c] = c + 4          # colours 4..15, including the two-digit ones
txt = I.ascii(g).splitlines()
ok("ascii_header_and_rows", len(txt) == 5 and txt[1].strip() == "012345678901" and txt[3].startswith("  1 "))
ok("ascii_colours_10_15_aligned", len({len(x) for x in txt[2:]}) == 1 and txt[3][4:] == I.LETTERS[4:16])
ok("ascii_box", I.ascii(g, (1, 2, 1, 4)).splitlines()[2] == "  1 " + I.LETTERS[6:9])
ok("legend_present_colours", I.legend([8, 0, 8]) == "W=white(0) R=red(8)")

# --- 3. segmentation -----------------------------------------------------------------------------------------
g = blank(16, 16)
fill(g, 2, 2, 9, 9, 3)       # frame 8x8 of colour 3 ...
fill(g, 3, 3, 8, 8, 0)       # ... hollow
fill(g, 5, 5, 6, 6, 8)       # red 2x2 inside the frame
fill(g, 2, 10, 4, 10, 9)     # blue bar touching the frame on its right
for c in range(0, 16):
    g[15][c] = 11            # yellow HUD bar on the bottom edge
objs, bg = I.segment(g)
by_colour = {o.colour: o for o in objs}
frame, red, blue, hud = by_colour[3], by_colour[8], by_colour[9], by_colour[11]
ok("segment_background", bg == 0 and len(objs) == 4, str([(o.colour, o.size) for o in objs]))
ok("segment_ids_reading_order", [o.id for o in objs] == [1, 2, 3, 4] and objs[0].colour == 3)
ok("segment_containment", red.parent == frame.id and red.id in frame.children and blue.parent is None)
ok("segment_adjacency", blue.id in frame.adjacent and frame.id in blue.adjacent and red.id not in frame.adjacent)
ok("segment_corners", red.corners == 4 and frame.corners == 8, f"red {red.corners} frame {frame.corners}")
ok("segment_hud_band", hud.hud and not red.hud and not frame.hud)
g2 = [row[:] for row in g]
fill(g2, 5, 5, 6, 6, 0)
fill(g2, 11, 12, 12, 13, 8)  # the same red 2x2 elsewhere
objs2, _ = I.segment(g2)
red2 = next(o for o in objs2 if o.colour == 8)
ok("shape_hash_translation_invariant", red2.hash == red.hash and red2.hash != blue.hash)
L = blank(6, 6)
fill(L, 1, 1, 3, 1, 5)
fill(L, 3, 1, 3, 3, 5)
ok("corners_L_shape", I.segment(L)[0][0].corners == 6)

# --- 4. tracker: stable ids and change lines ---------------------------------------------------------------
base = blank(16, 16)
fill(base, 5, 5, 6, 6, 8)    # red player
fill(base, 1, 12, 1, 12, 11)  # yellow dot
fill(base, 10, 2, 11, 3, 9)  # blue block
tr = I.Tracker()
o0, ch0 = tr.observe(base)
red_id = next(o.id for o in o0 if o.colour == 8)
blue_id = next(o.id for o in o0 if o.colour == 9)
ok("tracker_first_frame_no_change", ch0 is None and len(o0) == 3)
s1 = [r[:] for r in base]
fill(s1, 5, 5, 6, 6, 0)
fill(s1, 2, 5, 3, 5 + 1, 8)  # red moved up 3
fill(s1, 1, 12, 1, 12, 0)    # yellow dot vanished
fill(s1, 10, 2, 11, 3, 14)   # blue recoloured green
fill(s1, 14, 14, 14, 14, 6)  # a magenta dot appeared
o1, ch1 = tr.observe(s1)
ok("change_moved_dr", f"obj {red_id} R 2x2 moved up 3" in ch1, ch1)
ok("change_recoloured", f"obj {blue_id} b 2x2 at r10-11 c2-3 recoloured b->N" in ch1, ch1)
ok("change_vanished_appeared", "vanished from r1 c12" in ch1 and "appeared at r14 c14" in ch1, ch1)
ok("change_counts_cells", ch1.endswith("cells changed)"), ch1)
new_red = next(o for o in o1 if o.colour == 8)
new_mag = next(o for o in o1 if o.colour == 6)
ok("tracker_ids_stable", new_red.id == red_id and next(o.id for o in o1 if o.colour == 14) == blue_id
   and new_mag.id == 4, str([(o.id, o.colour) for o in o1]))
o2, ch2 = tr.observe(s1)
ok("tracker_no_change", ch2 == "no change")
s3 = [r[:] for r in s1]
fill(s3, 2, 5, 3, 6, 0)
fill(s3, 2, 9, 3, 10, 8)
_, ch3 = tr.observe(s3)
ok("change_moved_right", f"obj {red_id} R 2x2 moved right 4 -> r2-3 c9-10" in ch3, ch3)
tr.reset()
ok("tracker_reset_restarts_ids", tr.observe(base)[1] is None and tr.next_id == 4)

# --- 5. crops and scene text ------------------------------------------------------------------------------
a = blank(64, 64)
b = [r[:] for r in a]
b[30][30] = 8
boxes = I.change_boxes(a, b, margin=3)
ok("crop_margin_3", boxes == [(27, 27, 33, 33)], str(boxes))
b[62][2] = 11
ok("crop_two_regions", len(I.change_boxes(a, b, margin=3)) == 2)
ok("crop_clipped_at_edge", (59, 0, 63, 5) in I.change_boxes(a, b, margin=3))
tr = I.Tracker()
tr.observe(a)
objs_b, ch = tr.observe(b)
sc = I.Scene(7, b, objs_b, ch, prev=a, status="level 0/3 | NOT_FINISHED")
text = sc.text(budget_tokens=800)
ok("scene_has_change_and_crop", "last step:" in text and "changed region r27-33 c27-33:" in text
   and "R=red(8)" in text, text)
ok("scene_under_budget", len(text) <= 800 * 4, str(len(text)))
big = [[(r * 7 + c * 3) % 16 for c in range(64)] for r in range(64)]
tr = I.Tracker()
tr.observe(a)
objs_big, ch_big = tr.observe(big)
text_big = I.Scene(8, big, objs_big, ch_big, prev=a).text(budget_tokens=300)
ok("scene_budget_enforced", len(text_big) <= 300 * 4 and "too large to show" in text_big, str(len(text_big)))
start = I.Scene(0, base, I.Tracker().observe(base)[0], level_start=True, status="level 0/3").text()
ok("scene_level_start_full_board", "board:" in start and start.count("\n") > 16 and "objects (3," in start)
js = sc.to_json()
ok("scene_json_no_numeric_grid", js["letters"][30][30] == "R" and "rows" not in js and js["objects"][0]["bbox"])

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys.exit(1 if FAILS else 0)
