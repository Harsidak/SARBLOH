# --- path shim: these suites live in tests/ but import from the repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PRIME = _os.environ.get("PRIME_DIR", "Sarbloh-Experimentation")   # perception.py exists only in the copy for now
for _p in (_ROOT, _os.path.join(_ROOT, _PRIME)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end shim ---
"""E116 perception (harness/agent/perception.py) and the REPL's observe() (harness/runtime/skills/observation.py) on
hand-made grids: letters, segmentation and change lines (kept from intuition.py); the X8 briefing (things named by
object id, measured facts, role guesses, the line cap); the picture (size, map drawn from the frame, no text panel,
same bytes from parallel threads); the full and short views; observe() reading what the host wrote. Busy and empty
frames must not fail. The end-to-end path (observe() in a real kernel -> the full state pushed once per state) is in
test_prime_tools.py.
"""
import importlib.util
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

if importlib.util.find_spec("harness.agent.perception") is None:
    print(f"SKIP: {_PRIME} has no harness/agent/perception.py (E116 lives in Sarbloh-Experimentation)")
    print("\n================ SUMMARY ================\nCHECKS: 0\nHARD FAILS: 0 -> []")
    sys.exit(0)

from PIL import Image

from harness.agent import perception as P
from harness.runtime.skills import observation as O

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


# --- 1. letters, objects, change lines (the intuition.py part) ------------------------------------------------
g = blank(3, 12)
for c in range(12):
    g[1][c] = c + 4
txt = P.ascii(g).splitlines()
ok("ascii_header_and_rows", len(txt) == 5 and txt[1].strip() == "012345678901" and txt[3][4:] == P.LETTERS[4:16])
ok("legend", P.legend([8, 0, 8]) == "W=white(0) R=red(8)")

g = blank(16, 16)
fill(g, 2, 2, 9, 9, 3)
fill(g, 3, 3, 8, 8, 0)
fill(g, 5, 5, 6, 6, 8)
fill(g, 2, 10, 4, 10, 9)
fill(g, 15, 0, 15, 15, 11)
objs, bg = P.segment(g)
by = {o.colour: o for o in objs}
ok("segment", bg == 0 and [o.id for o in objs] == [1, 2, 3, 4] and by[8].parent == by[3].id
   and by[9].id in by[3].adjacent and by[11].hud and by[8].corners == 4 and by[3].corners == 8)

base = blank(16, 16)
fill(base, 5, 5, 6, 6, 8)
fill(base, 10, 2, 11, 3, 9)
tr = P.Tracker()
o0, ch0 = tr.observe(base)
red_id = next(o.id for o in o0 if o.colour == 8)
s1 = [r[:] for r in base]
fill(s1, 5, 5, 6, 6, 0)
fill(s1, 2, 5, 3, 6, 8)
o1, ch1 = tr.observe(s1)
ok("tracker_change_line", ch0 is None and ch1 == "8 cells changed at r2-6 c5-6"
   and next(o.id for o in o1 if o.colour == 8) == red_id, ch1)
ok("tracker_no_change", tr.observe(s1)[1] == "no change")

# --- 2. the X8 briefing on a made-up level -------------------------------------------------------------------
# light-grey floor; a charcoal wall band across the middle; a blue-top/orange 3x3 (the likely self) above it; a green
# 5x5 ring below it (empty inside, exactly the self's size); a yellow bar on the bottom edge; three red 2x2 copies
# at the top right (a count).


def level(gap=False):
    g = blank(64, 64, 1)
    fill(g, 30, 0, 37, 63, 4)
    if gap:
        fill(g, 30, 28, 37, 37, 1)
    fill(g, 10, 10, 10, 12, 9)
    fill(g, 11, 10, 12, 12, 12)
    fill(g, 48, 48, 52, 52, 14)
    fill(g, 49, 49, 51, 51, 1)
    fill(g, 63, 0, 63, 40, 11)
    for c in (50, 54, 58):
        fill(g, 1, c, 2, c + 1, 8)
    return g


g = level()
objs, _ = P.Tracker().observe(g)
oid = {o.colour: o.id for o in objs}                 # one object per colour, except the three red copies
reds = sorted(o.id for o in objs if o.colour == 8)
t0 = time.time()
b = P.brief(g, objs, actions=[1, 2, 3, 4])
dt = time.time() - t0
text = "\n".join(b.lines())
ok("briefing_controls", b.controls == "arrow keys" and text.startswith("BRIEFING (arrow keys)"))
ok("things_named_by_object_id", f"obj {oid[12]} (orange/blue square) = YOU" in text, text)
ok("goal_guess_ring", any(f"obj {oid[14]} (" in x and "= GOAL" in x for x in b.guesses), str(b.guesses))
ok("lives_guess_copies", any("= LIVES" in x and all(str(r) in x for r in reds) for x in b.guesses), str(b.guesses))
ok("moves_left_guess_bar", any(f"obj {oid[11]} (" in x and "= MOVES LEFT" in x for x in b.guesses), str(b.guesses))
ok("guess_has_test", all(" - test: " in x for x in b.guesses))
ok("fact_enclosure_fits", any(f"encloses an EMPTY 3x3 space -- exactly the size of: obj {oid[12]}" in f
                              for f in b.facts["inside"]), str(b.facts["inside"]))
ok("fact_copies", any("3 identical red square (2x2)" in f for f in b.facts["copies"]), str(b.facts["copies"]))
ok("fact_walls", any("charcoal areas are not open floor" in f for f in b.facts["reach"]), str(b.facts["reach"]))
ok("fact_two_areas_flood_fill", any("2 main objects are in 2 separate open areas" in f for f in b.facts["reach"]),
   str(b.facts["reach"]))
b_gap = P.brief(level(gap=True), P.Tracker().observe(level(gap=True))[0], actions=[1, 2, 3, 4])
ok("fact_one_area_with_gap", any("share one open space" in f for f in b_gap.facts["reach"]), str(b_gap.facts["reach"]))
ok("no_letter_names", not re.search(r"thing \d+ \(|\([A-Z]\) ", text), text)
b_click = P.brief(g, objs, actions=[6])
ok("clicks_only_no_you", b_click.controls == "clicks" and not any("= YOU" in x for x in b_click.guesses))
ok("brief_fast", dt < 2.0, f"{dt:.2f}s")

# the tile grid: wall tiles on a 5-cell grid and a 5x5 self. A gauge in a box on the screen edge must not hide it,
# and it is kept (from the previous frame) while the self touches another thing and stops looking like a tile.


def tiles(gauge=False, touching=False, aligned=True):
    g = blank(64, 64, 1)
    step = 10 if aligned else 7
    for r in range(0, 56, step):
        for c in range(0, 56, step):
            fill(g, r, c, r + 4, c + 4, 4)
    fill(g, 5, 5, 6, 9, 9)
    fill(g, 7, 5, 9, 9, 12)
    if touching:
        fill(g, 5, 10, 9, 11, 8)
    if gauge:   # a wall-coloured status box: a chequered floor-coloured gauge (lots of off-grid edges) and lives
        fill(g, 59, 0, 63, 63, 4)
        for r in range(60, 63):
            for c in range(1, 63):
                if (r + c) % 2:
                    g[r][c] = 1
        fill(g, 60, 55, 61, 56, 8)
    return g


def lattice_of(g, prev=None):
    return P.brief(g, P.segment(g)[0], [1, 2, 3, 4], prev).lattice


L = lattice_of(tiles())
ok("lattice_found", L == (5, 0, 0, 1), str(L))
ok("lattice_ignores_edge_box_gauge", lattice_of(tiles(gauge=True)) == L)
b_tiles = P.brief(tiles(), P.segment(tiles())[0], [1, 2, 3, 4])
ok("lattice_kept_while_touching", lattice_of(tiles(touching=True)) is None
   and lattice_of(tiles(touching=True), b_tiles) == L)
ok("lattice_hint_must_still_fit", lattice_of(tiles(aligned=False), b_tiles) != L)
ok("lattice_in_reach_facts", any(f == "Floor is a 5x5 tile grid." for f in b_tiles.facts["reach"]))

# the line cap: at most `limit` fact and guess lines, the rest counted and left in `observe().briefing`
many = P.Briefing("arrow keys", {"align": [f"a{i}" for i in range(40)], "copies": ["c1", "c2"], "reach": ["r1"]},
                  [f"g{i}" for i in range(9)], b.layout, 1, None, {}, [], [])
capped = many.lines(10)
body = [x for x in capped if x.startswith("- ") and "more" not in x]
ok("cap_respected", len(body) == 10, str(capped))
ok("cap_every_kind_shown", all(f"- {x}" in capped for x in ("r1", "a0", "c1")), str(capped))
ok("cap_guesses_third", sum(x.startswith("- g") for x in capped) == 3, str(capped))
ok("cap_overflow_counted", "- (+36 more facts: `observe().briefing` in ipython)" in capped
   and "- (+6 more guesses: `observe().briefing` in ipython)" in capped, str(capped))
ok("uncapped_has_all", len(many.lines()) == 3 + 43 + 9)

# a grid of many things (tr87, bp35) gives hundreds of 'directly above' pairs: none is dropped from the briefing,
# and the shown text still gives every other fact kind its lines
grid_many = blank(64, 64)
for n, (r, c) in enumerate((r, c) for r in range(2, 58, 12) for c in range(2, 58, 12)):   # 25 things, 100 pairs
    fill(grid_many, r, c, r + 4, c + 4, (9, 8, 14, 11)[n % 4])
b_many = P.brief(grid_many, P.segment(grid_many)[0], [1, 2, 3, 4])
shown_many = b_many.lines(30)
ok("align_all_kept", len(b_many.facts["align"]) > 30, str(len(b_many.facts["align"])))
ok("align_no_crowding", all(any(f"- {f}" in shown_many for f in fs) for k, fs in b_many.facts.items() if fs),
   str({k: len(fs) for k, fs in b_many.facts.items()}))
ok("align_overflow_counted", any("more facts" in x for x in shown_many))

# --- 3. the picture ------------------------------------------------------------------------------------------
png = P.picture(b, 10)
img = Image.open(io.BytesIO(png)).convert("RGB")
ok("picture_size", img.size == (64 * 10 + 12 + 190, 64 * 10 + 28), str(img.size))
ok("picture_map_is_frame", img.getpixel((11 * 10 + 5, 28 + 11 * 10 + 5)) == P.PALETTE[12]
   and img.getpixel((10 * 10 + 5, 28 + 10 * 10 + 5)) == P.PALETTE[9])
ok("picture_small_cell", Image.open(io.BytesIO(P.picture(b, 4))).size == (64 * 4 + 12 + 190, 64 * 4 + 28))
out = [None] * 6


def draw(i):
    out[i] = P.picture(b, 10)


threads = [threading.Thread(target=draw, args=(i,)) for i in range(6)]
[t.start() for t in threads]
[t.join() for t in threads]
ok("picture_thread_safe", all(x == png for x in out))
ok("render_plain", Image.open(io.BytesIO(P.render(g, 4))).size == (256, 256))
msg = P.image_message("hello", png, 3)
ok("image_message", msg["_image_step"] == 3 and msg["content"][0] == {"type": "text", "text": "hello"}
   and msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))

# --- 4. hard frames do not fail ---------------------------------------------------------------------------------
busy = [[(r * 7 + c * 3) % 16 for c in range(64)] for r in range(64)]
noise = [[(r * 31 + c * 17 + r * c) % 16 for c in range(64)] for r in range(64)]
for name, grid in (("busy", busy), ("noise", noise), ("one_colour", blank(64, 64, 5)), ("tiny", blank(2, 3, 0))):
    try:
        t0 = time.time()
        bb = P.brief(grid, P.segment(grid)[0], actions=[1, 6])
        P.picture(bb, 10)
        P.Scene(1, grid, P.segment(grid)[0], briefing=bb).text(full=True)
        ok(f"hard_frame_{name}", time.time() - t0 < 10, f"{time.time() - t0:.1f}s")
    except Exception as e:  # noqa: BLE001
        ok(f"hard_frame_{name}", False, repr(e))

# --- 5. the views ------------------------------------------------------------------------------------------------
tr = P.Tracker()
objs0, _ = tr.observe(g)
g2 = [r[:] for r in g]
fill(g2, 10, 10, 12, 12, 1)
fill(g2, 7, 10, 7, 12, 9)
fill(g2, 8, 10, 9, 12, 12)
objs2, ch = tr.observe(g2)
b2 = P.brief(g2, objs2, [1, 2, 3, 4])
sc = P.Scene(5, g2, objs2, change=f"#5 A1(up): {ch}", prev=g, status="game x | level 1 of 3", briefing=b2,
             briefing_lines=6)
full = sc.text(full=True)
short = sc.text(budget_tokens=800)
ok("full_view_parts", full.startswith("[state after step #5] game x | level 1 of 3\nlast step: #5 A1(up): ")
   and "BRIEFING (arrow keys)" in full and "MEASURED (certain):" in full and "GUESSES" in full
   and "\nobjects (" in full and "\nboard:\n" in full and "\nlegend: " in full)
brief_part = full.split("\nobjects (")[0]
ok("full_view_briefing_capped", sum(x.startswith("- ") and "more" not in x for x in brief_part.splitlines()) <= 6)
ok("short_view_parts", "last step: #5 A1(up)" in short and "changed region" in short and "BRIEFING" not in short
   and "board:" not in short and len(short) <= 800 * 4, short)
ok("full_view_without_briefing", "BRIEFING" not in P.Scene(5, g2, objs2, briefing=None).text(full=True))
js = sc.to_json()
ok("json_for_observe", js["change"].startswith("#5 A1(up)") and js["letters"][8][10] == "O"
   and js["briefing"] == b2.lines() and "rows" not in js and json.dumps(js))

# --- 6. observe() in the REPL reads what the host wrote ------------------------------------------------------------
d = Path(tempfile.mkdtemp(prefix="e116_obs_"))
os.environ["RLM_SESSION_DIR"] = str(d)
empty = O.observe()
ok("observe_before_any_state", empty.step is None and empty.board == [] and empty.ascii() == "(no state yet)")
O.write(d, {**js, "text": full})
O.append_history(d, [{"i": 4, "level": 0, "action": "A1(up)", "change": "x", "state": "NOT_FINISHED"},
                     {"i": 5, "level": 0, "action": "A1(up)", "change": "y", "state": "NOT_FINISHED"}])
with (d / ".prime" / "history.jsonl").open("a", encoding="utf-8") as fh:
    fh.write('{"i": 6, "lev')                  # a line the host is still writing
obs = O.observe()
ok("observe_fields", obs.step == 5 and obs.change == js["change"] and obs.board[8][10] == "O"
   and obs.objects == json.loads(json.dumps(js["objects"])) and obs.briefing == b2.lines() and obs.text == full)
ok("observe_history_skips_partial_line", [h["i"] for h in obs.history] == [4, 5])
ok("observe_ascii_window", obs.ascii(8, 10, 9, 12).splitlines()[2] == "  8 OOO")
try:
    obs.letters
    ok("observe_old_name_hint", False)
except AttributeError as e:
    ok("observe_old_name_hint", "obs.board" in str(e), str(e))
ok("observe_repr", repr(obs).startswith("Observation(step 5, ") and "64x64" in repr(obs))

print("\n================ SUMMARY ================")
print(f"CHECKS: {CHECKS[0]}")
print(f"HARD FAILS: {len(FAILS)} -> {FAILS}")
sys.exit(1 if FAILS else 0)
