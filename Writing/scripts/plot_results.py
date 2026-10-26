"""Every data figure of the paper, from Writing/data/*.json and Writing/figures/perception/ (run
collect_data.py, extract_curves.py and make_perception_figure.py first). Writes PDFs into Writing/figures/.

    uv run python Writing/scripts/plot_results.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.dates import DateFormatter, date2num  # noqa: E402
from PIL import Image  # noqa: E402
from datetime import datetime  # noqa: E402

W = Path(__file__).resolve().parents[1]
DATA = json.loads((W / "data/paper_data.json").read_text(encoding="utf-8"))
CURVES = json.loads((W / "data/levels_vs_time.json").read_text(encoding="utf-8"))
FIG = W / "figures"

INK = "#1f2430"
OURS = "#c2410c"      # burnt orange: our system everywhere
BASE = "#2563eb"      # blue: the Duck/TAAF baseline
MUTED = "#9ca3af"
GRID = "#e5e7eb"
COL, FULL = 3.33, 7.0  # AAAI column and text widths, inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    """The PDF for the paper and a PNG for the repository README."""
    fig.savefig(FIG / f"{name}.pdf", dpi=300)
    fig.savefig(FIG / "png" / f"{name}.png", dpi=200)
    plt.close(fig)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def leaderboard():
    rows = [r for r in DATA["submissions"] if r[0] >= "2026-09-27"]   # the legacy agent (Aug, <= 0.15) is in the text
    ok = {"2026-10-15", "2026-10-16", "2026-10-22"}   # failed experiments, drawn grey
    fig, ax = plt.subplots(figsize=(FULL, 2.05))
    xs = [date2num(datetime.fromisoformat(d)) for d, *_ in rows]
    ys = [s for _, s, *_ in rows]
    best, bx, by = 0.0, [], []
    for x, y in zip(xs, ys):
        best = max(best, y)
        bx.append(x)
        by.append(best)
    ax.step(bx, by, where="post", color=OURS, lw=1.6, zorder=3, label="best score so far")
    for (d, s, kernel, note), x in zip(rows, xs):
        bad = d in ok
        ax.scatter(x, s, s=16, zorder=4, color="white" if bad else OURS, edgecolor=MUTED if bad else OURS, lw=0.9)
    D = lambda d: date2num(datetime.fromisoformat(d))
    for x, y, txt in ((D("2026-09-27T12"), 2.2, "Prime Agent port: 0.00 to 0.27"),
                      (D("2026-09-30T12"), 9.6, "5.14: Flash-Next on SGLang, prefix-cache\nprompts, 10 games at once"),
                      (D("2026-10-11T12"), 7.4, "5.65, 6.06: one tool, drain-only context"),
                      (D("2026-10-12T12"), 12.4, "10.98: code graph, level memory"),
                      (D("2026-10-24T06"), 16.0, "16.53")):
        ax.text(x, y, txt, fontsize=6.6, color=INK, va="bottom")
    ax.axhline(1.21, color=BASE, lw=0.8, ls=(0, (4, 2)))
    ax.text(D("2026-10-05T18"), 1.55, "Milestone #1 winner, June (Duck): 1.21", color=BASE, fontsize=6.6)
    ax.scatter([], [], s=16, color="white", edgecolor=MUTED, label="failed experiment")
    ax.xaxis.set_major_formatter(DateFormatter("%d %b"))
    ax.set_xlim(xs[0] - 1, xs[-1] + 3)
    ax.set_ylim(-0.6, 19)
    ax.set_ylabel("Kaggle public score")
    ax.legend(loc="upper center", frameon=False)
    grid(ax)
    save(fig, "leaderboard")


def curves():
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    for key, lab, col, ls in (("ours", "SARBLOH (1 pass)", OURS, "-"),
                              ("taaf_distinct", "Duck harness, distinct levels (4 passes)", BASE, "-"),
                              ("taaf_sum", "Duck harness, summed over passes", BASE, ":")):
        pts = CURVES[key]
        ax.step([p[0] for p in pts], [p[1] for p in pts], where="post", color=col, ls=ls, lw=1.4, label=lab)
    ax.axhline(183, color=MUTED, lw=0.6, ls="--")
    ax.text(0.1, 186, "183 levels in the 25 games", fontsize=6.5, color=MUTED)
    for h, n in ((2, 97), (4, 136), (8.2, 143)):
        ax.annotate(str(n), (h, n), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=6.5, color=OURS)
    for h, n in ((2, 63), (4, 103), (8.2, 129)):
        ax.annotate(str(n), (h, n), xytext=(0, -9), textcoords="offset points", ha="center", fontsize=6.5, color=BASE)
    ax.set_xlim(0, 8.3)
    ax.set_ylim(0, 200)
    ax.set_xlabel("wall clock (hours), one RTX PRO 6000")
    ax.set_ylabel("levels cleared")
    ax.legend(loc="lower right", frameon=False, handlelength=1.6)
    grid(ax)
    save(fig, "levels_vs_time")


def gemini():
    rows = DATA["gemini"]
    fig, ax = plt.subplots(figsize=(COL, 1.45))
    names = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    cols = [OURS if n != "raw grid" else MUTED for n in names]
    ax.barh(names, vals, color=cols, height=0.62, zorder=3)
    for i, (n, v) in enumerate(zip(names, vals)):
        txt = dict((r[0], r[2]) for r in rows)[n]
        ax.text(v * 1.12, i, txt + (" actions" if n != "raw grid" else ""), va="center", fontsize=6.6, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(3, 2500)
    ax.set_xlabel("actions to clear ls20 level 1 (log scale)")
    grid(ax, "x")
    save(fig, "gemini_ladder")


def lab():
    rows = DATA["lab"]
    fig, ax = plt.subplots(figsize=(COL, 1.85))
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    toks = [r[2] for r in rows]
    cols = [OURS if n.startswith("X8") else (MUTED if n.startswith("E0") else "#f4a261") for n in names]
    ax.bar(range(len(rows)), vals, color=cols, width=0.68, zorder=3)
    for i, (v, t) in enumerate(zip(vals, toks)):
        ax.text(i, v + 0.6, f"{v}", ha="center", fontsize=6.6, color=INK)
    ax.set_xticks(range(len(rows)), [n.split(" ", 1)[0] + "\n" + f"{t:,}" for n, t in zip(names, toks)])
    ax.set_xlabel("arm, and its text tokens per frame", fontsize=7)
    ax.set_ylim(0, 30)
    ax.set_ylabel("rubric score (of 28)")
    grid(ax)
    save(fig, "perception_lab")


def per_game():
    pg = DATA["per_game"]
    games = sorted(pg, key=lambda g: (pg[g]["ours"] / pg[g]["of"], pg[g]["ours"]))
    fig, ax = plt.subplots(figsize=(FULL, 1.75))
    x = range(len(games))
    ax.bar(list(x), [pg[g]["of"] for g in games], width=0.82, color="#f3f4f6", zorder=1)
    ax.bar([i - 0.17 for i in x], [pg[g]["taaf_best_of_4"] for g in games], width=0.34, color=BASE, zorder=3,
           label="Duck harness, best of 4 passes")
    ax.bar([i + 0.17 for i in x], [pg[g]["ours"] for g in games], width=0.34, color=OURS, zorder=3,
           label="SARBLOH, 1 pass")
    ax.set_xticks(list(x), games, rotation=0, fontsize=6.4)
    ax.set_ylabel("levels cleared")
    ax.set_xlim(-0.7, len(games) - 0.3)
    from matplotlib.patches import Patch
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h + [Patch(color="#f3f4f6")], lab + ["levels in the game"], loc="lower left", frameon=False, ncol=3,
              bbox_to_anchor=(0, 1.0))
    ax.set_ylim(0, 10)
    grid(ax)
    save(fig, "per_game")


def perception_panels():
    """Four panels: raw integers (a crop), the plain frame, the ls20 map, the r11l map."""
    P = FIG / "perception"
    raw = [r.split() for r in (P / "ls20" / "raw.txt").read_text(encoding="utf-8").splitlines()]
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.9), gridspec_kw={"width_ratios": [0.82, 1, 1.26, 1.26]})
    a = axes[0]
    r0, c0, n = 42, 30, 14                             # the player block and the corridor around it
    crop = [[int(v) for v in row[c0:c0 + n]] for row in raw[r0:r0 + n]]
    a.imshow([[0.97] * n for _ in range(n)], cmap="gray", vmin=0, vmax=1, extent=(0, n, n, 0))
    for y, row in enumerate(crop):
        for x, v in enumerate(row):
            a.text(x + 0.5, y + 0.55, str(v), ha="center", va="center", fontsize=3.4, family="monospace",
                   color=OURS if v in (9, 12) else INK)
    a.set_xlim(0, n)
    a.set_ylim(n, 0)
    a.set_axis_off()
    a.set_title(f"(a) raw integers, {n}x{n} crop", loc="left", fontsize=7)
    for ax, img, title in ((axes[1], P / "ls20" / "plain.png", "(b) plain frame"),
                           (axes[2], P / "ls20" / "map.png", "(c) map, ls20 (keys)"),
                           (axes[3], P / "r11l" / "map.png", "(d) map, r11l (clicks)")):
        ax.imshow(Image.open(img))
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=7)
    fig.subplots_adjust(wspace=0.12)
    save(fig, "perception_views")


if __name__ == "__main__":
    for f in (leaderboard, curves, gemini, lab, per_game, perception_panels):
        f()
        print("ok", f.__name__)
