"""Every data figure of the paper, from Writing/data/*.json and Writing/figures/perception/.
Writes vector PDFs into Writing/figures/ and PNGs into Writing/figures/png/.

    uv run python Writing/scripts/plot_results.py

One palette for all figures (checked with the dataviz validator, light surface):
hatched indigo = SARBLOH, grey = the Duck baseline, periwinkle / lavender = extra series, slate = text.
Same stripe palette as Table 1. Grey is deliberately neutral (fails only the chroma floor); the pastels are
always direct-labelled because they sit under 3:1 contrast on white.
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.dates import DateFormatter, date2num  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from PIL import Image  # noqa: E402

W = Path(__file__).resolve().parents[1]
DATA = json.loads((W / "data/paper_data.json").read_text(encoding="utf-8"))
CURVES = json.loads((W / "data/levels_vs_time.json").read_text(encoding="utf-8"))
FIG = W / "figures"

# ---- palette -----------------------------------------------------------------------------------
OURS = "#5B50E6"      # series 1: SARBLOH (hatched indigo)
BASE = "#8F8F99"      # series 2: Duck baseline / reference (grey)
TEAL = "#7F8BFF"      # series 3: periwinkle
VIOLET = "#A090FF"    # series 4: lavender
INK = "#0F172A"       # text
SOFT = "#475569"      # secondary text
MUTED = "#C4C4CC"     # de-emphasised marks (light grey)
GRID = "#E6E6EA"      # gridlines
PANEL = "#F4F4F7"     # background fills
OURS_TINT = "#ECEEFF"
BASE_TINT = "#EEEEF1"
plt.rcParams["hatch.linewidth"] = 0.9
COL, FULL = 3.4, 7.1  # AAAI column and text width, inches

_have = {f.name for f in font_manager.fontManager.ttflist}
SANS = next((f for f in ("Inter", "Segoe UI", "Helvetica Neue", "Arial") if f in _have), "DejaVu Sans")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [SANS, "DejaVu Sans"], "mathtext.fontset": "dejavusans",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": SOFT,
    "xtick.color": SOFT, "ytick.color": SOFT, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3,
    "ytick.major.size": 3, "figure.facecolor": "white", "axes.facecolor": "white", "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})


def save(fig, name):
    (FIG / "png").mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / "png" / f"{name}.png", dpi=260)
    plt.close(fig)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def rbar(ax, x, h, w, color, base=0.0, z=3, horizontal=False, r=0.0, hatch=None):
    """A bar with a slightly rounded data end (the end away from the baseline)."""
    if horizontal:
        p = FancyBboxPatch((base, x - w / 2), h, w, boxstyle=f"round,pad=0,rounding_size={r}", lw=0,
                           fc=color, zorder=z, mutation_aspect=1, hatch=hatch, ec="white" if hatch else color)
    else:
        p = FancyBboxPatch((x - w / 2, base), w, h, boxstyle=f"round,pad=0,rounding_size={r}", lw=0,
                           fc=color, zorder=z, mutation_aspect=1, hatch=hatch, ec="white" if hatch else color)
    ax.add_patch(p)


def title(ax, text, sub=None):
    ax.text(0, 1.14 if sub else 1.07, text, transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=INK,
            va="bottom")
    if sub:
        ax.text(0, 1.04, sub, transform=ax.transAxes, fontsize=7.8, color=SOFT, va="bottom")


# ---- 1. leaderboard trajectory -----------------------------------------------------------------
def leaderboard():
    rows = [r for r in DATA["submissions"] if r[0] >= "2026-09-27"]
    failed = {"2026-10-15", "2026-10-16", "2026-10-22"}
    D = lambda d: date2num(datetime.fromisoformat(d))
    fig, ax = plt.subplots(figsize=(FULL, 3.0))
    xs = [D(r[0]) for r in rows]
    best, bx, by = 0.0, [], []
    for x, r in zip(xs, rows):
        best = max(best, r[1])
        bx.append(x)
        by.append(best)
    ax.fill_between(bx, 0, by, step="post", color=OURS_TINT, zorder=1)
    ax.step(bx, by, where="post", color=OURS, lw=2.2, zorder=3, label="best score so far")
    for x, r in zip(xs, rows):
        bad = r[0] in failed
        ax.scatter(x, r[1], s=34, zorder=4, color="white" if bad else OURS, edgecolor=MUTED if bad else OURS,
                   lw=1.3)
    ax.scatter([], [], s=34, color="white", edgecolor=MUTED, lw=1.3, label="regression or failed experiment")
    ax.axhline(1.21, color=BASE, lw=1.1, ls=(0, (5, 3)), zorder=2)
    ax.text(D("2026-10-06"), 1.7, "Duck, Milestone #1: 1.21", color=BASE, fontsize=8, va="bottom")
    notes = ((D("2026-09-28"), 2.4, "Prime Agent port\n0.00 to 0.27", "left"),
             (D("2026-10-05"), 6.6, "5.14  Flash-Next on SGLang,\nprefix-cache prompts", "left"),
             (D("2026-10-10"), 9.2, "5.65, 6.06  one tool,\ndrain-only context", "left"),
             (D("2026-10-20"), 12.2, "10.98  code graph,\nlevel memory", "right"),
             (D("2026-10-26"), 17.6, "16.53", "center"))
    for x, y, t, ha in notes:
        ax.text(x, y, t, fontsize=8, color=INK, ha=ha, va="bottom", linespacing=1.25,
                fontweight="bold" if t == "16.53" else "normal")
    ax.xaxis.set_major_formatter(DateFormatter("%d %b"))
    ax.set_xlim(xs[0] - 0.8, xs[-1] + 2.2)
    ax.set_ylim(0, 20)
    ax.set_ylabel("Kaggle public score")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.98))
    grid(ax)
    save(fig, "leaderboard")


# ---- 2. levels over wall clock -----------------------------------------------------------------
def curves():
    fig, ax = plt.subplots(figsize=(COL, 2.9))
    spec = (("taaf_sum", "Duck harness, summed over 4 passes", MUTED, (0, (1.5, 2)), 1.4),
            ("taaf_distinct", "Duck harness, distinct levels", BASE, "-", 1.9),
            ("ours", "SARBLOH, one pass", OURS, "-", 2.4))
    for key, lab, col, ls, lw in spec:
        pts = CURVES[key]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.step(xs, ys, where="post", color=col, ls=ls, lw=lw, label=lab, zorder=3)
        if key == "ours":
            ax.fill_between(xs, 0, ys, step="post", color=OURS_TINT, zorder=1)
    ax.axhline(183, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(0.15, 186, "183 levels in the 25 public games", fontsize=7.8, color=SOFT, va="bottom")
    for h, n in ((2, 97), (4, 136), (8.2, 143)):
        ax.scatter(h, n, s=22, color=OURS, zorder=5, edgecolor="white", lw=0.8)
        ax.annotate(str(n), (h, n), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8,
                    color=OURS, fontweight="bold")
    for h, n in ((2, 63), (4, 103), (8.2, 129)):
        ax.scatter(h, n, s=22, color=BASE, zorder=5, edgecolor="white", lw=0.8)
        ax.annotate(str(n), (h, n), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=8,
                    color=BASE, fontweight="bold")
    ax.set_xlim(0, 8.7)
    ax.set_ylim(0, 205)
    ax.set_xlabel("wall clock (hours), one RTX PRO 6000")
    ax.set_ylabel("levels cleared")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, handlelength=1.8, labelspacing=0.4)
    grid(ax)
    save(fig, "levels_vs_time")


# ---- 3. ablation ladder (motivation) -----------------------------------------------------------
def gemini():
    rows = DATA["gemini"]
    names = ["Raw integer grid", "+ rendered image", "+ board explained", "+ reasoning in code"]
    vals = [r[1] for r in rows]
    notes = ["200-300 actions, game lost", "12-24 actions", "12 actions", "6 actions"]
    cols = [MUTED, TEAL, VIOLET, OURS]
    fig, ax = plt.subplots(figsize=(COL, 2.35))
    n = len(rows)
    for i, (v, c) in enumerate(zip(vals, cols)):
        y = n - 1 - i
        ax.barh(y, v, height=0.58, color=c, zorder=3, lw=0, hatch="////" if c == OURS else None, ec="white")
        ax.text(v * 1.18, y, notes[i], va="center", fontsize=8, color=INK,
                fontweight="bold" if c == OURS else "normal")
    ax.set_yticks(range(n), names[::-1])
    ax.set_xscale("log")
    ax.set_xlim(3, 3000)
    ax.set_xlabel("actions to clear ls20 level 1 (log scale)")
    ax.tick_params(axis="y", length=0, labelsize=8.2)
    ax.spines["left"].set_visible(False)
    grid(ax, "x")
    save(fig, "gemini_ladder")


# ---- 4. perception lab -------------------------------------------------------------------------
def lab():
    rows = DATA["lab"]
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    toks = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(COL, 2.7))
    cols = [OURS if n.startswith("X8") else (MUTED if n.startswith("E0") else BASE) for n in names]
    for i, (v, c) in enumerate(zip(vals, cols)):
        ax.bar(i, v, width=0.7, color=c, zorder=3, lw=0, hatch="////" if c == OURS else None, ec="white")
        ax.text(i, v + 0.6, str(v), ha="center", fontsize=8.2, color=INK,
                fontweight="bold" if c == OURS else "normal")
    ax.set_xticks(range(len(rows)), [n.split(" ", 1)[0] for n in names], fontsize=8)
    ax.set_ylim(0, 30)
    ax.set_ylabel("rubric score (of 28)")
    ax.axhline(28, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(len(rows) - 0.5, 28.4, "maximum 28", ha="right", fontsize=7.6, color=SOFT)
    # tokens per frame, as a second row of small labels under the x tick labels
    for i, t in enumerate(toks):
        ax.text(i, -0.08, f"{t:,}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7,
                color=SOFT)
    ax.text(-0.04, -0.08, "tokens\nper frame", transform=ax.transAxes, ha="right", va="top", fontsize=6.6,
            color=SOFT, linespacing=1.1)
    grid(ax)
    save(fig, "perception_lab")


# ---- 5. per game vs frontier models (teaser, from data/Score.csv) ------------------------------
FRONTIER = (("openai-gpt6.1-sol", "GPT-6.1 Sol", INK, "o"),
            ("anthropic-claude-opus-5", "Claude Opus 5", SOFT, "s"),
            ("google-gemini-3.8-flash", "Gemini 3.8 Flash", TEAL, "^"),
            ("xai-grok-4.6-xhigh", "Grok 4.6 xhigh", VIOLET, "D"),
            ("openai-gpt5.6-terra", "GPT-5.6 Terra", BASE, "v"))


def per_game():
    import csv
    rows = list(csv.reader((W / "data/Score.csv").read_text(encoding="utf-8").splitlines()))
    games = rows[0][1:]
    sc = {r[0]: [float(v.strip("%")) for v in r[1:]] for r in rows[1:]}
    ours = next(k for k in sc if k.startswith("Sarbloh"))
    mean = lambda k: sum(sc[k]) / len(sc[k])
    order = sorted(range(len(games)), key=lambda i: (sc[ours][i], sum(sc[k][i] for k, *_ in FRONTIER)))
    fig, ax = plt.subplots(figsize=(FULL, 4.3))
    xs = np.arange(len(order))
    series = [(ours, OURS)] + [(k, c) for k, _, c, _ in FRONTIER]
    w = 0.82 / len(series)
    for j, (k, c) in enumerate(series):
        ax.bar(xs - 0.41 + w * (j + 0.5), [sc[k][i] for i in order], w * 0.92, color=c, lw=0, zorder=3)
    ax.set_xticks(xs, [games[i] for i in order], fontsize=7.4, rotation=90)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0, 104)
    ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel("score on the game (%)")
    ax.set_xlabel("games, sorted by the SARBLOH score")
    hs = [Patch(fc=c, lw=0) for _, c in series]
    names = [f"SARBLOH, mean {mean(ours):.1f}"] + [f"{n}, {mean(k):.1f}" for k, n, _, _ in FRONTIER]
    fig.legend(hs, names, loc="upper left", bbox_to_anchor=(0.06, 1.02), ncol=6, frameon=False, handlelength=1.6,
               columnspacing=1.2, fontsize=7.2)
    grid(ax)
    save(fig, "per_game")


# ---- 6. perception views panel -----------------------------------------------------------------
def perception_panels():
    P = FIG / "perception"
    raw = [r.split() for r in (P / "ls20" / "raw.txt").read_text(encoding="utf-8").splitlines()]
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.75), gridspec_kw={"width_ratios": [1.0, 1.0, 1.3, 1.3]})
    a = axes[0]
    r0, c0, n = 42, 30, 14
    crop = [[int(v) for v in row[c0:c0 + n]] for row in raw[r0:r0 + n]]
    a.imshow([[0.98] * n for _ in range(n)], cmap="gray", vmin=0, vmax=1, extent=(0, n, n, 0))
    for y, row in enumerate(crop):
        for x, v in enumerate(row):
            hot = v in (9, 12)
            a.text(x + 0.5, y + 0.55, str(v), ha="center", va="center", fontsize=4.8, family="monospace",
                   fontweight="bold" if hot else "normal", color=OURS if hot else (INK if v else MUTED))
    a.set_xlim(0, n)
    a.set_ylim(n, 0)
    panels = [(a, "(a) raw integers", "14x14 crop of the grid"),
              (axes[1], "(b) plain frame", "what a vision model gets"),
              (axes[2], "(c) map: ls20", "keys, tiles, walls"),
              (axes[3], "(d) map: r11l", "clicks, objects")]
    for ax, img in zip(axes[1:], (P / "ls20" / "plain.png", P / "ls20" / "map.png", P / "r11l" / "map.png")):
        ax.imshow(Image.open(img))
    for ax, t, s in panels:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(GRID)
            sp.set_linewidth(1.0)
        ax.set_title(t, loc="left", fontsize=8.6, fontweight="bold", color=INK, pad=14)
        ax.text(0, 1.02, s, transform=ax.transAxes, fontsize=7.2, color=SOFT, va="bottom")
    fig.subplots_adjust(wspace=0.12)
    save(fig, "perception_views")


if __name__ == "__main__":
    for f in (leaderboard, curves, gemini, lab, per_game, perception_panels):
        f()
        print("ok", f.__name__)
