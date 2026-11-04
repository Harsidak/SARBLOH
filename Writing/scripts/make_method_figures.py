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
    ax.imshow(Image.open(path).convert("RGB").crop(box))
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
        ax.text(n_show + 15, y, f"{100 * tot:.1f}%\nwhole run", color=c, fontsize=6.6, va="center",
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
    for f in (turn_trace, briefing_budget, cache_curve, presentations, score_budget):
        f()
        print("ok", f.__name__)
