"""Every data figure of the paper, from Writing/data/*.json and Writing/figures/perception/.
Writes publication-quality PDFs into Writing/figures/ and PNGs into Writing/figures/png/.

Run:
    uv run python Writing/scripts/plot_results.py
"""

import json
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, date2num
from matplotlib.patches import Patch
from PIL import Image

W = Path(__file__).resolve().parents[1]
DATA = json.loads((W / 'data/paper_data.json').read_text(encoding='utf-8'))
CURVES = json.loads((W / 'data/levels_vs_time.json').read_text(encoding='utf-8'))
FIG = W / 'figures'

# Refined Publication Color Palette
INK = '#0F172A'         # Slate 900: primary text and axes
OURS = '#C2410C'        # Deep Terracotta / Rust: SARBLOH
OURS_LIGHT = '#EA580C'  # Warm Amber-Orange
OURS_TINT = '#FFF7ED'   # Soft warm background tint
BASE = '#2563EB'        # Slate Blue: Duck / baseline
BASE_LIGHT = '#3B82F6'  # Medium blue
BASE_TINT = '#EFF6FF'   # Soft blue tint
MUTED = '#64748B'       # Slate 500: neutral muted elements
MUTED_LIGHT = '#94A3B8' # Slate 400
GRID = '#E2E8F0'        # Slate 200: subtle gridlines
BG_BAR = '#F1F5F9'      # Slate 100: bar backgrounds / frames
COL, FULL = 3.33, 7.0   # AAAI column and text widths in inches

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.titlesize': 8.5,
    'axes.labelsize': 8,
    'xtick.labelsize': 7.2,
    'ytick.labelsize': 7.2,
    'legend.fontsize': 7.2,
    'axes.edgecolor': INK,
    'axes.labelcolor': INK,
    'xtick.color': INK,
    'ytick.color': INK,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.65,
    'xtick.major.width': 0.65,
    'ytick.major.width': 0.65,
    'pdf.fonttype': 42,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
})


def save(fig, name):
    """Save publication-grade vector PDF and high-res raster PNG."""
    fig.savefig(FIG / f'{name}.pdf', dpi=300)
    fig.savefig(FIG / 'png' / f'{name}.png', dpi=250)
    plt.close(fig)


def grid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, lw=0.6, ls='-', zorder=0)
    ax.set_axisbelow(True)


def gemini():
    """Figure 2: Aesthetic ablation ladder showing effect of structured input views."""
    rows = DATA['gemini']
    fig, ax = plt.subplots(figsize=(COL, 1.85))

    formal_names = [
        'Raw Grid (Tokens)',
        '+ Visual Frame',
        '+ Geometric Briefing',
        '+ Code-as-Action (REPL)'
    ][::-1]
    vals = [r[1] for r in rows][::-1]
    descs = [
        '200-300 moves (Failed)',
        '18 moves (14x speedup)',
        '12 moves (21x speedup)',
        '6 moves (Super-human)'
    ][::-1]

    palette = [OURS, OURS_LIGHT, '#F97316', MUTED]
    bars = ax.barh(formal_names, vals, color=palette, height=0.55, zorder=3,
                   edgecolor=[INK if p == OURS else 'none' for p in palette], lw=0.6)

    # Human baseline reference line
    ax.axvline(10, color=BASE, ls='--', lw=0.85, zorder=2)
    ax.text(10.5, 3.32, 'Human Baseline (~10 actions)', fontsize=6.3, color=BASE,
            fontweight='bold', va='center')

    for i, (v, desc, col) in enumerate(zip(vals, descs, palette)):
        ax.annotate(desc, (v, i), xytext=(8, 0), textcoords='offset points',
                    va='center', fontsize=6.6, color=INK,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFFFFF',
                              edgecolor=col if col != MUTED else GRID, lw=0.7))

    ax.set_xscale('log')
    ax.set_xlim(3, 2400)
    ax.set_xlabel('Actions to Complete ls20 Level 1 (Log Scale)')
    grid(ax, 'x')
    save(fig, 'gemini_ladder')


def perception_panels():
    """Figure 3: Enlarged, aesthetic four-panel perception views comparison."""
    P = FIG / 'perception'
    raw = [r.split() for r in (P / 'ls20' / 'raw.txt').read_text(encoding='utf-8').splitlines()]
    
    # Substantially enlarged vertical height (2.75 inches vs original 1.9 inches)
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.75), gridspec_kw={'width_ratios': [0.95, 1.0, 1.35, 1.35]})
    
    # (a) Integer crop
    a = axes[0]
    r0, c0, n = 42, 30, 14
    crop = [[int(v) for v in row[c0:c0 + n]] for row in raw[r0:r0 + n]]
    a.imshow([[0.98] * n for _ in range(n)], cmap='gray', vmin=0, vmax=1, extent=(0, n, n, 0))
    for y, row in enumerate(crop):
        for x, v in enumerate(row):
            is_agent = v in (9, 12)
            a.text(x + 0.5, y + 0.55, str(v), ha='center', va='center',
                   fontsize=4.6, family='monospace', fontweight='bold' if is_agent else 'normal',
                   color=OURS if is_agent else (INK if v != 0 else MUTED_LIGHT))
    a.set_xlim(0, n)
    a.set_ylim(n, 0)
    a.set_xticks([])
    a.set_yticks([])
    for spine in a.spines.values():
        spine.set_visible(True)
        spine.set_color(GRID)
        spine.set_linewidth(0.8)
    a.set_title('(a) Raw Grid (14x14 crop)', loc='left', fontsize=7.5, pad=5, fontweight='bold', color=INK)

    # (b, c, d) Image panels
    panels = [
        (axes[1], P / 'ls20' / 'plain.png', '(b) Visual Frame (64x64)'),
        (axes[2], P / 'ls20' / 'map.png', '(c) Perception Map: ls20 (Keys)'),
        (axes[3], P / 'r11l' / 'map.png', '(d) Perception Map: r11l (Clicks)')
    ]
    for ax, img, title in panels:
        ax.imshow(Image.open(img))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(GRID)
            spine.set_linewidth(0.8)
        ax.set_title(title, loc='left', fontsize=7.5, pad=5, fontweight='bold', color=INK)

    fig.subplots_adjust(wspace=0.15)
    save(fig, 'perception_views')


def lab():
    """Figure 4: Aesthetic bar chart of perception lab ablation arms."""
    rows = DATA['lab']
    fig, ax = plt.subplots(figsize=(COL, 2.05))
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    toks = [r[2] for r in rows]

    colors = []
    for n in names:
        if n.startswith('X8'):
            colors.append(OURS)
        elif n.startswith('E0'):
            colors.append(MUTED)
        else:
            colors.append('#F59E0B')

    bars = ax.bar(range(len(rows)), vals, color=colors, width=0.64, zorder=3,
                  edgecolor=[INK if c == OURS else 'none' for c in colors], lw=0.6)

    for i, (v, t, c) in enumerate(zip(vals, toks, colors)):
        ax.text(i, v + 0.7, f'{v}/28', ha='center', fontsize=6.4, color=INK,
                fontweight='bold' if c == OURS else 'normal')

    ax.set_xticks(range(len(rows)), [n.split(' ', 1)[0] + '\n' + f'{t:,} t' for n, t in zip(names, toks)])
    ax.set_xlabel('Perception Arm & Text Tokens per Frame', fontsize=7.4)
    ax.set_ylim(0, 31)
    ax.set_ylabel('Visual Rubric Score (Max 28)')
    
    # Highlight badge for X8
    ax.annotate('X8 (Ours): 25/28\n(89.4% token reduction)', xy=(7, 25), xytext=(2.2, 27.5),
                arrowprops=dict(arrowstyle='->', color=OURS, lw=0.8),
                fontsize=6.5, color=OURS, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor=OURS_TINT, edgecolor=OURS, lw=0.8))

    grid(ax)
    save(fig, 'perception_lab')


def curves():
    """Figure 6: Levels cleared over wall-clock time comparing SARBLOH with baseline."""
    fig, ax = plt.subplots(figsize=(COL, 2.2))

    for key, lab, col, ls, lw in (
        ('ours', 'SARBLOH (1 pass, scheduler)', OURS, '-', 1.8),
        ('taaf_distinct', 'Duck Harness, distinct (4 passes)', BASE, '-', 1.4),
        ('taaf_sum', 'Duck Harness, total across 4 passes', MUTED, ':', 1.1)
    ):
        pts = CURVES[key]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.step(xs, ys, where='post', color=col, ls=ls, lw=lw, label=lab, zorder=3)
        if key == 'ours':
            ax.fill_between(xs, 0, ys, step='post', color=OURS, alpha=0.06, zorder=2)

    ax.axhline(183, color=MUTED_LIGHT, lw=0.75, ls='--')
    ax.text(0.15, 186, 'Benchmark Total (183 Levels across 25 Games)', fontsize=6.3, color=MUTED)

    # Annotated checkpoints
    for h, n in ((2, 97), (4, 136), (8.2, 143)):
        ax.annotate(str(n), (h, n), xytext=(0, 4), textcoords='offset points',
                    ha='center', fontsize=6.8, color=OURS, fontweight='bold')
    for h, n in ((2, 63), (4, 103), (8.2, 129)):
        ax.annotate(str(n), (h, n), xytext=(0, -10), textcoords='offset points',
                    ha='center', fontsize=6.8, color=BASE, fontweight='bold')

    ax.set_xlim(0, 8.4)
    ax.set_ylim(0, 205)
    ax.set_xlabel('Wall-Clock Execution Time (Hours, 1x RTX PRO 6000)')
    ax.set_ylabel('Total ARC-AGI-3 Levels Cleared')
    ax.legend(loc='lower right', frameon=True, facecolor='#FFFFFF', edgecolor=GRID, framealpha=0.9, handlelength=1.6)
    grid(ax)
    save(fig, 'levels_vs_time')


def per_game():
    """Figure 7 / Teaser Figure: Comprehensive per-game comparative breakdown."""
    pg = DATA['per_game']
    games = sorted(pg, key=lambda g: (pg[g]['ours'] / pg[g]['of'], pg[g]['ours']))
    fig, ax = plt.subplots(figsize=(FULL, 2.35))
    x = list(range(len(games)))

    # Total levels background bar
    ax.bar(x, [pg[g]['of'] for g in games], width=0.82, color=BG_BAR, zorder=1, edgecolor=GRID, lw=0.5)

    # Grouped comparative bars
    w = 0.33
    ax.bar([i - 0.17 for i in x], [pg[g]['taaf_best_of_4'] for g in games], width=w, color=BASE, zorder=3,
           label='Duck Harness (Best of 4 Concurrent Passes)', edgecolor='none')
    ax.bar([i + 0.17 for i in x], [pg[g]['ours'] for g in games], width=w, color=OURS, zorder=3,
           label='SARBLOH (Single Pass, Priority-Scheduled)', edgecolor=INK, lw=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=0, fontsize=6.6, color=INK)
    ax.set_ylabel('Levels Solved (Count)', fontsize=8)
    ax.set_xlabel('ARC-AGI-3 Public Game Environment (Sorted by SARBLOH Completion Rate)', fontsize=8)
    ax.set_xlim(-0.7, len(games) - 0.3)
    ax.set_ylim(0, 10.5)

    h, lab = ax.get_legend_handles_labels()
    ax.legend(h + [Patch(facecolor=BG_BAR, edgecolor=GRID, lw=0.6)],
              lab + ['Total Available Levels'],
              loc='lower left', frameon=True, facecolor='#FFFFFF', edgecolor=GRID, framealpha=0.95,
              ncol=3, bbox_to_anchor=(0.0, 1.02), fontsize=7.2)

    grid(ax)
    save(fig, 'per_game')


if __name__ == '__main__':
    for f in (curves, gemini, lab, per_game, perception_panels):
        f()
        print('Successfully plotted:', f.__name__)
