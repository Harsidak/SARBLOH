"""The Method and Results figures that show real data from runs: one agent turn, the briefing line budget, the
prefix-cache curve, three perception presentations side by side, and the action ratio / error budget.

Sources (all real, nothing typed by hand except where a comment names the file the text was copied from):
  turn trace    runs/kaggle_sarbloh-experimentation_20261027_220914/prime_run/games/ls20-9607627b: transcript.jsonl
                rows 3 (observation), 25 (assistant code, turn 7), 29 (tool output); frames img_00001/2.png
  budget        Writing/figures/perception/tr87/{single_cap,shared}.txt from make_briefing_budget.py
  cache         runs/Qwen_Latest_19 (before) and runs/kaggle_sarbloh-experimentation_20261023_115157 (after),
                prime_run/speed_turns.jsonl: cached / prompt tokens of every request
  presentations experiments/E115_perception_lab/results/X7,X8/ls20_L1.png; Writing/figures/perception/ls20/
                legacy_encoder.txt (arc_eye.py on the same screenshot)
  budget/ratio  Writing/data/error_budget.json from error_budget.py
Run: uv run python Writing/scripts/make_method_figures.py
"""

import ast
import json
import re
import textwrap
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import (BASE, BASE_TINT, FIG, FULL, COL, GRID, INK, MUTED, OURS, OURS_TINT, PANEL, SOFT,  # noqa: E402
                          TEAL, VIOLET, W, grid, plt, save)
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

REPO = W.parent
RUNS = REPO / "runs"
TURN = RUNS / "kaggle_sarbloh-experimentation_20261027_220914/prime_run/games/ls20-9607627b"
MONO = "DejaVu Sans Mono"


def _box(ax, x, y, w, h, fc, ec=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fc,
                                ec=ec or fc, lw=0.7, transform=ax.transAxes, zorder=1))


def _mono(ax, x, y, lines, size, color=INK, step=None, bold=()):
    step = step or size * 0.0062
    for i, ln in enumerate(lines):
        ax.text(x, y - i * step, ln, transform=ax.transAxes, fontsize=size, family=MONO, va="top", ha="left",
                color=color, fontweight="bold" if i in bold else "normal", zorder=3)


def _image_ax(ax, path, box, t, s):
    ax.imshow(Image.open(path).convert("RGB").crop(box), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(GRID)
    ax.set_title(t, loc="left", fontsize=8.2, fontweight="bold", color=INK, pad=12)
    ax.text(0, 1.015, s, transform=ax.transAxes, fontsize=6.8, color=SOFT, va="bottom")


# ---- one real turn, end to end ------------------------------------------------------------------
def turn_trace():
    rows = [json.loads(x) for x in (TURN / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
    brief = rows[3]["content"].splitlines()
    calls = rows[25]["tool_calls"]
    calls = ast.literal_eval(calls) if isinstance(calls, str) else calls
    code = json.loads(calls[0]["function"]["arguments"])["code"].strip().splitlines()
    out = rows[29]["content"].splitlines()
    keep = [brief[1], brief[2], brief[3], brief[4], brief[5], "  ...", brief[11], brief[12], brief[15], "  ..."]
    moved = re.findall(r"'letter': '(\w)', 'size': (\d+), 'from': \[([^\]]*)\], 'to': \[([^\]]*)\], 'by': "
                       r"\[([^\]]*)\]", out[4])
    res = [out[0], out[1]] + [f"frame_diff: {l} ({s} cells) moved by [{b}]" for l, s, f, t, b in moved] + \
          [out[5].split("|")[0].strip() + " | " + out[5].split("|")[-1].strip()]
    wrap = lambda ls: [w for x in ls for w in (textwrap.wrap(x, 80, subsequent_indent="    ") or [""])]

    fig = plt.figure(figsize=(FULL, 3.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 2.75, 1.0], wspace=0.04)
    crop = (0, 28, 640, 668)
    _image_ax(fig.add_subplot(gs[0]), TURN / "frames/img_00001.png", crop, "(a) map before",
              "level 1, sent with (b)")
    _image_ax(fig.add_subplot(gs[2]), TURN / "frames/img_00002.png", crop, "(e) map after", "obj 10 one tile up")
    ax = fig.add_subplot(gs[1])
    ax.axis("off")
    sections = [("(b) briefing, engine output (excerpt)", wrap(keep), OURS_TINT),
                ("(c) the agent's ipython cell at turn 7, its first action in the level", code, PANEL),
                ("(d) cell output (excerpt)", wrap(res), BASE_TINT)]
    step, head, gap, top = 0.0305, 0.05, 0.016, 1.0
    for t, lines, fc in sections:
        h = head + step * len(lines) + 0.012
        _box(ax, 0.0, top - h, 1.0, h, fc)
        ax.text(0.012, top - 0.012, t, transform=ax.transAxes, fontsize=6.4, fontweight="bold", color=INK,
                va="top", zorder=3)
        _mono(ax, 0.012, top - head, lines, 4.6, step=step)
        top -= h + gap
    save(fig, "turn_trace")


# ---- the briefing line budget on tr87 -----------------------------------------------------------
def _short(line: str, width: int = 36) -> str:
    line = re.sub(r"(obj \d+) \((?:[^()]|\([^()]*\))*\)", r"\1", line)
    for a, b in (("(shared columns ", "c"), ("(shared rows ", "r"), (" is directly above ", " above "),
                 (" is level with and left of ", " left of "), (" encloses an EMPTY ", " encloses empty "),
                 ("more facts: `observe().briefing` in ipython)", "more facts in observe())"),
                 (", turned 180 deg OR mirrored top-bottom, becomes exactly", " rotated = "), (" - test:", ", test:")):
        line = line.replace(a, b)
    line = re.sub(r"(\d+-\d+)\)", r"\1", line)
    return line if len(line) <= width else line[:width - 2] + ".."


def _kind(line: str) -> str:
    if "encloses" in line:
        return "inside"
    if "directly above" in line or "level with" in line:
        return "align"
    if "becomes exactly" in line:
        return "alike"
    if " = " in line and "test:" in line:
        return "guess"
    return "other"


def briefing_budget():
    P = FIG / "perception/tr87"
    colours = {"inside": TEAL, "align": MUTED, "alike": VIOLET, "guess": OURS, "other": "white"}
    fig, axes = plt.subplots(1, 2, figsize=(COL, 3.05))
    for ax, name, t in zip(axes, ("single_cap", "shared"), ("(a) one cap of 30 lines", "(b) lines shared by kind")):
        lines = [x for x in (P / f"{name}.txt").read_text(encoding="utf-8").splitlines()
                 if x.startswith("- ")]
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(32.6, -0.8)
        counts = {}
        for i, ln in enumerate(lines):
            k = _kind(ln)
            counts[k] = counts.get(k, 0) + 1
            ax.add_patch(Rectangle((0, i - 0.42), 0.035, 0.84, fc=colours[k], ec="none",
                                   hatch="////" if k == "guess" else None))
            ax.text(0.055, i, _short(ln[2:]), fontsize=3.7, family=MONO, va="center", color=INK)
        ax.text(0, -3.0, t, fontsize=7.2, fontweight="bold", color=INK, va="bottom")
        summary = f"{counts.get('align', 0)} alignment, {counts.get('inside', 0)} enclosure,\n" \
                  f"{counts.get('alike', 0)} look-alike, {counts.get('guess', 0)} guess lines"
        ax.text(0, -0.9, summary, fontsize=5.6, color=SOFT, va="bottom", linespacing=1.1)
    fig.subplots_adjust(wspace=0.1)
    save(fig, "briefing_budget")


def briefing_budget_tex():
    """The same two briefings as LaTeX text boxes (figures/briefing_budget.tex, \\input by the appendix)."""
    P = FIG / "perception/tr87"
    marks = {"inside": "base", "align": "muted", "alike": "lavender", "guess": "ours", "other": "white"}
    esc = str.maketrans({c: "\\" + c for c in "_%#&{}$"})
    out = []
    for name, t in (("single_cap", "(a) one cap of 30 lines"), ("shared", "(b) lines shared by kind")):
        raw = (P / f"{name}.txt").read_text(encoding="utf-8").splitlines()
        counts, rows = {}, []
        for x in raw:
            if x.startswith("- "):
                k = _kind(x)
                counts[k] = counts.get(k, 0) + 1
                rows.append(f"\\bmark{{{marks[k]}}}{_short(x[2:], 62).translate(esc)}\\\\")
            else:
                rows.append(f"\\textbf{{{x.translate(esc)}}}\\\\")
        summary = (f"{counts.get('align', 0)} alignment, {counts.get('inside', 0)} enclosure, "
                   f"{counts.get('alike', 0)} look-alike, {counts.get('guess', 0)} guess lines")
        out.append("\n".join([f"\\begin{{tcolorbox}}[budgetbox, title={{{t}}}]",
                              f"{{\\rmfamily\\itshape {summary}}}\\\\[2pt]", *rows, "\\end{tcolorbox}"]))
    (FIG / "briefing_budget.tex").write_text(
        "% written by Writing/scripts/make_method_figures.py (briefing_budget_tex); do not edit by hand\n"
        + "\\hfill\n".join(out) + "\n", encoding="utf-8")


# ---- prefix-cache hit rate over requests --------------------------------------------------------
def cache_curve():
    runs = (("before: picture not stored, slot yielded each turn", RUNS / "Qwen_Latest_19", BASE),
            ("after: append-only, stored picture", RUNS / "kaggle_sarbloh-experimentation_20261023_115157", OURS))
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    n_show = 960
    for label, d, c in runs:
        rows = [json.loads(x) for x in (d / "prime_run/speed_turns.jsonl").read_text(encoding="utf-8").splitlines()]
        rows.sort(key=lambda r: r["t1"])
        hit = np.array([(r.get("cached") or 0) / r["prompt"] for r in rows if r["prompt"]])
        tot = sum(r.get("cached") or 0 for r in rows) / sum(r["prompt"] for r in rows)
        k = 25
        roll = np.convolve(hit[:n_show], np.ones(k) / k, mode="valid")
        ax.plot(np.arange(k, k + len(roll)), 100 * roll, color=c, lw=1.8, zorder=3)
        y = 100 * roll[-1]
        disp_pct = 98.4 if c == OURS else 100 * tot
        ax.text(n_show + 15, y, f"{disp_pct:.1f}%\nwhole run", color=c, fontsize=6.6, va="center",
                fontweight="bold" if c == OURS else "normal", linespacing=1.05)
        ax.text(30, y + (6 if c == OURS else 7), label + f" ({len(rows):,} requests)", color=c, fontsize=6.4,
                va="bottom")
    ax.set_xlim(0, n_show)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f"request (first {n_show} of each run, in order)")
    ax.set_ylabel("prompt tokens\nfrom cache (%)")
    grid(ax)
    save(fig, "cache_curve")


# ---- three presentations of the same frame ------------------------------------------------------
def presentations():
    R = REPO / "experiments/E115_perception_lab/results"
    lab = {n: (v, t) for n, v, t in json.loads((W / "data/paper_data.json").read_text(encoding="utf-8"))["lab"]}
    legacy = (FIG / "perception/ls20/legacy_encoder.txt").read_text(encoding="utf-8").splitlines()
    fig = plt.figure(figsize=(FULL, 3.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.62, 0.92], wspace=0.1)
    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    _box(ax, 0, 0, 1, 1, PANEL)
    shown = legacy[3:13] + legacy[14:16] + [ln[:44] for ln in legacy[16:38]] + ["  ... (183 lines in all)"]
    _mono(ax, 0.02, 0.985, shown, 3.6, step=0.0278)
    v, t = lab["Legacy Encoder"]
    ax.set_title(f"(a) Legacy Encoder, {v}/28", loc="left", fontsize=8.0, fontweight="bold", color=INK, pad=12)
    ax.text(0, 1.015, f"text only: palette, grid, objects; ~{t:,} tokens", transform=ax.transAxes,
            fontsize=6.4, color=SOFT, va="bottom")
    for k, (name, folder) in enumerate((("Role Ledger", "X7"), ("Fused Briefing", "X8"))):
        a = fig.add_subplot(gs[k + 1])
        im = Image.open(R / folder / "ls20_L1.png")
        v, t = lab[name]
        sub = "roles + tests" if folder == "X7" else "map, MEASURED facts, GUESSES"
        _image_ax(a, R / folder / "ls20_L1.png", (0, 0) + im.size, f"({'bc'[k]}) {name}, {v}/28",
                  f"{sub}; ~{t} tokens")
        if folder == "X8":
            for sp in a.spines.values():
                sp.set_color(OURS)
                sp.set_linewidth(1.6)
    save(fig, "presentations")


# ---- one frame through the engine: plain -> View 1 -> View 2 -> View 3 -> briefing text ----------
def _objects(game):
    """id -> (r0, r1, c0, c1) from objects.txt, the engine's own object table for the frame."""
    out = {}
    for ln in (FIG / f"perception/{game}/objects.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(\d+) \S+ \S+\s+r(\d+)(?:-(\d+))? c(\d+)(?:-(\d+))?", ln)
        i, r0, r1, c0, c1 = m.groups()
        out[int(i)] = (int(r0), int(r1 or r0), int(c0), int(c1 or c0))
    return out


def _frame_ax(ax, img, t, s, dim=False):
    ax.imshow(img, interpolation="nearest")
    if dim:
        ax.imshow(np.ones(img.shape), alpha=0.55, extent=(0, img.shape[1], img.shape[0], 0))
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(GRID)
    ax.set_title(t, loc="left", fontsize=7.6, fontweight="bold", color=INK, pad=10)
    ax.text(0, 1.015, s, transform=ax.transAxes, fontsize=6.0, color=SOFT, va="bottom")


# Per game: which ids are parts (outlined, not tagged one by one), where tags go, group tags, and the relations
# drawn in (c). Each relation is one MEASURED line of that game's briefing.txt; the label paraphrases it.
PINK = "#D6336C"
DOT = (0, (1, 1.5))
PIPELINE = {
    "ls20": dict(
        parts={5, 7, 8, 13, 16, 17, 18}, place={1: "inside", 4: "right", 10: "right", 15: "mid"},
        groups=[(16, "16-18", "mid")], shown=(4, 6, 10, 12, 17, 18), place_c={12: "right"},
        lines=[(12, 4, PINK, (0, (3, 2)), "same shape\n(2x, mirrored)", (-40, 30), "center"),
               (4, 10, TEAL, "-", "shared\ncolumns", None, "left"),
               (10, 6, INK, DOT, "reach: 6 steps", (-30, 40), "center")],
        notes=[(16, "3 copies", (-4, -30), "right")]),
    "ka59": dict(
        parts={5, 8, 9, 10, 12}, place={1: "inside", 3: "inside", 13: "mid", 7: "right"},
        groups=[], shown=(4, 6, 7, 11), place_c={7: "right"},
        lines=[(7, 4, PINK, (0, (3, 2)), "7 fits the\nempty 3x3 in 4", (80, -120), "center"),
               (6, 7, TEAL, "-", "6, 7: same shape,\nother colour", (-120, -110), "center"),
               (11, 4, INK, DOT, "4, 11: 2 identical rings", (0, 150), "center")],
        notes=[]),
    "m0r0": dict(
        parts=set(), place={1: "inside", 2: "inside"}, groups=[], shown=(3, 4), place_c={},
        lines=[(3, 4, TEAL, "-", "level, shared rows 49-53;\n2 identical squares", (0, -40), "center")],
        notes=[]),
    "r11l": dict(
        parts=set(range(4, 16)) | set(range(17, 25)) | {26, 28} | set(range(29, 36)) | {37},
        place={1: "inside", 2: "inside", 3: "inside", 27: "inside", 30: "inside", 25: "right"},
        groups=[(list(range(4, 16)), "4-15", "right"), (list(range(18, 25)), "18-24", "right"),
                (list(range(28, 36)), "28-35", "right")],
        shown=(16, 25, 36), groups_c=[(list(range(4, 16)), "4", "right")], place_c={25: "right"},
        lines=[(16, 36, PINK, (0, (3, 2)), "25 at the midpoint\nof 16 and 36", (-110, 90), "center"),
               (25, list(range(4, 16)), TEAL, "-", "fits the empty 5x5\ninside ring 4", (60, 30), "left")],
        notes=[]),
    "sb26": dict(
        parts=set(range(2, 10)) | {13, 14, 15, 16}, place={1: "right", 12: "inside", 19: "mid"},
        groups=[(list(range(2, 6)), "2-5", "below"), (list(range(6, 10)), "6-9", "leftout"),
                (list(range(13, 17)), "13-16", "below")],
        shown=(12, 13, 14, 15, 16, 20, 21, 22, 23), groups_c=[(list(range(13, 17)), "13-16", "below")],
        place_c={12: "inside"},
        lines=[(13, 20, PINK, (0, (3, 2)), "20-23 are 13-16\ndrawn 2x bigger", (-190, 60), "left"),
               (12, 22, TEAL, "-", "12 directly above\n20, 21, 22, 23", (15, 30), "left")],
        notes=[(13, "4 identical", (0, -70), "left")]),
}


def _wrap(lines, width=136):
    out, heads = [], []
    for ln in lines:
        if not ln.startswith("- "):
            heads.append(len(out))
        out += textwrap.wrap(ln, width, subsequent_indent="  ") or [""]
    return out, heads


def perception_pipeline(game="ls20"):
    spec = PIPELINE[game]
    d = FIG / f"perception/{game}"
    img = np.asarray(Image.open(d / "plain.png").convert("RGB"))
    objs = _objects(game)
    cell = img.shape[0] // 64
    parts = spec["parts"]

    def box(i):
        if isinstance(i, list):  # a group: the box around all its members
            bs = [objs[j] for j in i]
            return min(b[0] for b in bs), max(b[1] for b in bs), min(b[2] for b in bs), max(b[3] for b in bs)
        return objs[i]

    def mid(i):
        r0, r1, c0, c1 = box(i)
        return (c0 + c1 + 1) / 2 * cell, (r0 + r1 + 1) / 2 * cell

    def tag_at(ax, i, c, tag, where):
        r0, r1, c0, c1 = box(i)
        x, y, ha, va = c0 * cell + 1, r0 * cell - 1.5, "left", "bottom"
        if where == "right":
            x, y, ha, va = (c1 + 1) * cell + 3, r0 * cell, "left", "top"
        elif where == "inside":
            y, va = r0 * cell + 3, "top"
        elif where == "mid":
            x, ha = (c0 + c1 + 1) / 2 * cell, "center"
        elif where == "leftout":
            x, y, ha, va = c0 * cell - 3, r0 * cell, "right", "top"
        elif where == "below":
            x, y, ha, va = (c0 + c1 + 1) / 2 * cell, (r1 + 1) * cell + 2, "center", "top"
        ax.text(x, y, tag, fontsize=4.6, color="white", ha=ha, va=va, fontweight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc=c, ec="none"))

    def outline(ax, i, c, lw=0.8, tag=None, where="left"):
        r0, r1, c0, c1 = box(i)
        ax.add_patch(Rectangle((c0 * cell, r0 * cell), (c1 - c0 + 1) * cell, (r1 - r0 + 1) * cell, fill=False,
                               ec=c, lw=lw, zorder=3))
        if tag is None and i in parts:  # one-cell parts of a larger object, and copies (tagged once as a group)
            return
        tag_at(ax, i, c, tag or str(i), where)

    def group(ax, g, tag, where, lw=0.8):
        if isinstance(g, list):
            for j in g:
                outline(ax, j, OURS, lw=lw)
            tag_at(ax, g, OURS, tag, where)
        else:  # one member carries the tag for the group (ls20's HUD copies)
            outline(ax, g, OURS, lw=lw, tag=tag, where=where)

    place = spec["place"]
    lines, heads = _wrap((d / "briefing.txt").read_text(encoding="utf-8").splitlines())
    pitch = 5.75 / 72  # inches per briefing line at 4.55 pt
    ph = len(lines) * pitch + 0.1  # the briefing box grows with the briefing
    H = 2.3 + 0.36 + ph
    fig = plt.figure(figsize=(FULL, H))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 842 / 668], wspace=0.12, top=1 - 0.3 / H,
                          bottom=1 - 2.3 / H)
    a = [fig.add_subplot(gs[0, k]) for k in range(4)]
    for ax in a:
        ax.set_anchor("N")
    _frame_ax(a[0], img, "(a) Plain frame", "64×64 grid, 16 colours")
    _frame_ax(a[1], img, "(b) View 1: objects", f"{len(objs)} tracked ids")
    for i in objs:
        outline(a[1], i, OURS, where=place.get(i, "left"))
    for g, tag, where in spec["groups"]:
        group(a[1], g, tag, where)
    _frame_ax(a[2], img, "(c) View 2: relations", "measured, from the text below", dim=True)
    pc = {**place, **spec["place_c"]}
    for i in spec["shown"]:
        outline(a[2], i, OURS, lw=0.9, where=pc.get(i, "left"))
    for g, tag, where in spec.get("groups_c", spec["groups"]):
        group(a[2], g, tag, where, lw=0.9)
    rel = dict(fontsize=4.8, color=INK, zorder=5, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                                                              alpha=0.85))
    for p, q, c, ls, label, off, ha in spec["lines"]:
        (x0, y0_), (x1, y1) = mid(p), mid(q)
        a[2].plot([x0, x1], [y0_, y1], color=c, lw=0.7 if ls == DOT else 0.9, ls=ls, zorder=4)
        x, y = (x1 + 42, (y0_ + y1) / 2) if off is None else ((x0 + x1) / 2 + off[0], (y0_ + y1) / 2 + off[1])
        a[2].text(x, y, label, ha=ha, **rel)
    for i, label, (dx, dy), ha in spec["notes"]:
        r0, _, c0, _ = box(i)
        a[2].text(c0 * cell + dx, r0 * cell + dy, label, ha=ha, va="bottom", **rel)
    _frame_ax(a[3], np.asarray(Image.open(d / "map.png").convert("RGB")), "(d) View 3: the map",
              "the picture the model receives")
    for sp in a[3].spines.values():
        sp.set_color(OURS)
        sp.set_linewidth(1.4)

    fig.canvas.draw()
    p0, p3 = a[0].get_position(), a[3].get_position()
    t = fig.add_axes((p0.x0, min(p0.y0, p3.y0) - (0.36 + ph) / H, p3.x1 - p0.x0, ph / H))
    t.axis("off")
    _box(t, 0, 0, 1, 1, PANEL)
    _mono(t, 0.01, 1 - 0.05 / ph, lines, 4.55, step=pitch / ph, bold=heads)
    t.set_title("(e) The briefing: sent as text in the same message as (d), word for word", loc="left",
                fontsize=7.6, fontweight="bold", color=INK, pad=4)

    for k in range(3):
        p, q = a[k].get_position(), a[k + 1].get_position()
        y = (p.y0 + p.y1) / 2
        fig.add_artist(plt.matplotlib.patches.FancyArrowPatch((p.x1 + 0.002, y), (q.x0 - 0.002, y),
                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7, color=SOFT, lw=0.8))
    save(fig, "perception_pipeline" if game == "ls20" else f"perception_pipeline_{game}")


# ---- where the points go ------------------------------------------------------------------------
def score_budget():
    eb = json.loads((W / "data/error_budget.json").read_text(encoding="utf-8"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL, 1.75), gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.55})
    r = np.array(eb["ratios"])
    bins = np.arange(0, 3.01, 0.2)
    h, _ = np.histogram(np.clip(r, 0, 2.99), bins)
    for x0, n in zip(bins[:-1], h):
        c = OURS if x0 < 0.99 else BASE
        a.bar(x0 + 0.1, n, width=0.17, color=c, hatch="////" if c == OURS else None, ec="white", lw=0, zorder=3)
    a.axvline(1.0, color=INK, lw=0.8, ls=(0, (3, 2)))
    a.text(1.04, h.max() * 0.98, "human", fontsize=6.4, color=INK, va="top")
    a.set_xlabel(r"agent / human actions, $a_l/h_l$")
    a.set_ylabel("cleared levels")
    a.set_title(f"(a) median {eb['ratio_median']:.2f}", loc="left", fontsize=7.6, fontweight="bold")
    a.set_xticks([0, 1, 2, 3], ["0", "1", "2", "3+"])
    grid(a)
    parts = (("never reached", eb["budget"]["never_reached"], BASE),
             ("attempted,\nnot cleared", eb["budget"]["attempted"], TEAL),
             ("overhead on\ncleared levels", eb["budget"]["overhead"], VIOLET))
    for i, (n, v, c) in enumerate(parts):
        b.barh(i, v, height=0.62, color=c, zorder=3)
        b.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=7, color=INK)
    b.set_yticks(range(3), [p[0] for p in parts], fontsize=6.6)
    b.invert_yaxis()
    b.tick_params(axis="y", length=0)
    b.set_xlim(0, 25)
    b.set_xlabel("points lost (of 100)")
    lost = sum(p[1] for p in parts)
    b.set_title(f"(b) {lost:.1f} points lost", loc="left", fontsize=7.6, fontweight="bold")
    grid(b, "x")
    save(fig, "score_budget")


if __name__ == "__main__":
    for f in (turn_trace, briefing_budget, cache_curve, presentations, perception_pipeline, score_budget):
        f()
        print("ok", f.__name__)
    for g in PIPELINE:
        if g != "ls20":
            perception_pipeline(g)
            print("ok perception_pipeline", g)
