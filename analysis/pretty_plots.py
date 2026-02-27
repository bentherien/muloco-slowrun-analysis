#!/usr/bin/env python3
"""Pretty publication-quality plots for MuLoCo slowrun analysis."""

import re
import os
import glob
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '-',
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.dpi': 200,
})

PALETTE = [
    '#1f77b4', '#e74c3c', '#2ecc71', '#f39c12',
    '#9b59b6', '#00bcd4', '#e67e22', '#34495e',
]
TARGET = 3.402
OUTDIR = '/mnt/raid0/claude/slowrun/analysis'


# ─── Parsing ──────────────────────────────────────────────────────────────────
def parse_log(filepath):
    train_steps, train_losses = [], []
    val_steps, val_losses, val_epochs = [], [], []

    with open(filepath) as f:
        for line in f:
            m = re.match(r'step\s+(\d+)\s+\([\d.]+%\)\s+\|\s+loss:\s+([\d.]+)', line)
            if m:
                train_steps.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
            m = re.match(
                r'Step\s+(\d+)\s+\|\s+Epoch\s+(\d+)\s+\|\s+Val BPB:\s+[\d.]+\s+\|\s+Val Loss:\s+([\d.]+)',
                line,
            )
            if m:
                val_steps.append(int(m.group(1)))
                val_epochs.append(int(m.group(2)))
                val_losses.append(float(m.group(3)))

    return dict(
        train_steps=np.array(train_steps),
        train_losses=np.array(train_losses),
        val_steps=np.array(val_steps),
        val_losses=np.array(val_losses),
        val_epochs=np.array(val_epochs),
        filename=os.path.basename(filepath),
    )


def smooth(y, w=50):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode='valid')


def load_all():
    runs = []
    for tag, d in [('fir', 'fir_logs'), ('tamia', 'tamia_logs')]:
        base = os.path.join(OUTDIR, 'logs', d)
        for f in sorted(glob.glob(os.path.join(base, '*.out'))):
            r = parse_log(f)
            if len(r['val_losses']) >= 2:
                r['cluster'] = tag
                runs.append(r)
    return runs


def label(fn):
    n = fn.replace('.out', '')
    n = re.sub(r'_\d{6,}$', '', n)
    return n


# ─── Helpers ──────────────────────────────────────────────────────────────────
def add_target_line(ax, ymin=None, ymax=None):
    ax.axhline(y=TARGET, color='#333333', ls='--', lw=1.4, zorder=1)
    xlim = ax.get_xlim()
    ax.text(xlim[1] * 0.98, TARGET + 0.008, 'baseline 3.402',
            ha='right', va='bottom', fontsize=9, color='#555555', fontstyle='italic')


# ─── PLOT 1  — Train + Val loss for top‑4 (overfitting diagnostic) ───────────
def plot_train_val_top4(runs):
    best = sorted(runs, key=lambda r: r['val_losses'][-1])
    best = [r for r in best if len(r['train_losses']) > 200][:4]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, run in enumerate(best):
        ax = axes[idx // 2][idx % 2]
        n = label(run['filename'])
        # pretty name
        pretty = n
        for old, new in [('p5_e', ''), ('p4_e', ''), ('muloco_', ''),
                         ('_omom0.5', ''), ('_omom0.6', ''),
                         ('_lrm0.25', ''), ('_wd1.6', ''),
                         ('_flr0.0', ''), ('_do0.1', '')]:
            pretty = pretty.replace(old, new)

        # raw train (faint)
        ax.plot(run['train_steps'], run['train_losses'],
                color=PALETTE[0], alpha=0.10, lw=0.4, rasterized=True)
        # smoothed train
        if len(run['train_losses']) > 60:
            sm = smooth(run['train_losses'], 60)
            ax.plot(run['train_steps'][59:], sm,
                    color=PALETTE[0], lw=1.6, label='Train (smoothed)')
        # val
        ax.plot(run['val_steps'], run['val_losses'], 'o-',
                color=PALETTE[1], ms=5, lw=2, label='Validation', zorder=5)

        add_target_line(ax)
        ax.set_xlabel('Optimizer Step')
        ax.set_ylabel('Loss')
        title = f"[{run['cluster']}]  {pretty}"
        final = run['val_losses'][-1]
        beat = "  ✓ BEAT" if final < TARGET else ""
        ax.set_title(f'{title}\nFinal val = {final:.4f}{beat}', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.9)

    fig.suptitle('MuLoCo K=1 — Train vs Validation Loss  (Top 4 Runs)',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(OUTDIR, 'pretty_train_val_top4.png')
    fig.savefig(path)
    plt.close()
    print(f'Saved {path}')


# ─── PLOT 2  — Val loss curves best‑8 ────────────────────────────────────────
def plot_val_best8(runs):
    best = sorted(runs, key=lambda r: r['val_losses'][-1])
    best = [r for r in best if len(r['val_epochs']) >= 6][:8]

    fig, ax = plt.subplots(figsize=(13, 7))
    for i, run in enumerate(best):
        n = label(run['filename'])
        short = n
        for old, new in [('muloco_olr', 'olr'), ('_omom0.5', ''), ('_omom0.6', ''),
                         ('_lrm0.25', ''), ('_wd1.6', ''), ('_do0.1', ''),
                         ('_flr0.0', '')]:
            short = short.replace(old, new)
        final = run['val_losses'][-1]
        marker = '★' if final < TARGET else ''
        ax.plot(run['val_epochs'], run['val_losses'], 'o-',
                color=PALETTE[i % len(PALETTE)], ms=4, lw=1.8,
                label=f'[{run["cluster"]}] {short}  →  {final:.3f} {marker}')

    add_target_line(ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('MuLoCo K=1 — Best Runs: Validation Loss vs Epoch')
    ax.set_ylim(3.30, 4.15)
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.95, ncol=1)
    fig.tight_layout()
    path = os.path.join(OUTDIR, 'pretty_val_best8.png')
    fig.savefig(path)
    plt.close()
    print(f'Saved {path}')


# ─── PLOT 3  — Generalization gap (overfitting diagnostic) ───────────────────
def plot_gen_gap(runs):
    best = sorted(runs, key=lambda r: r['val_losses'][-1])
    best = [r for r in best if len(r['val_losses']) >= 8 and r['val_losses'][-1] < 3.45][:6]

    fig, ax = plt.subplots(figsize=(13, 7))
    for i, run in enumerate(best):
        n = label(run['filename'])
        short = n
        for old, new in [('muloco_olr', 'olr'), ('_omom0.5', ''), ('_omom0.6', ''),
                         ('_lrm0.25', ''), ('_wd1.6', ''), ('_do0.1', ''),
                         ('_flr0.0', '')]:
            short = short.replace(old, new)

        gaps = []
        for vs, vl in zip(run['val_steps'], run['val_losses']):
            mask = (run['train_steps'] >= vs - 60) & (run['train_steps'] <= vs)
            if mask.sum() > 10:
                gaps.append(vl - run['train_losses'][mask].mean())
            else:
                gaps.append(np.nan)

        ax.plot(run['val_epochs'], gaps, 'o-',
                color=PALETTE[i % len(PALETTE)], ms=4, lw=1.6,
                label=f'[{run["cluster"]}] {short}')

    ax.axhline(y=0, color='#999', ls='-', lw=0.6)

    # Annotate the overfitting zone
    ax.axhspan(0.3, 0.8, color='#ffcccc', alpha=0.25, zorder=0)
    ax.text(ax.get_xlim()[1] * 0.95, 0.55, 'large gap\n(mild overfit)',
            ha='right', va='center', fontsize=9, color='#cc3333', fontstyle='italic')
    ax.axhspan(0, 0.15, color='#ccffcc', alpha=0.25, zorder=0)
    ax.text(ax.get_xlim()[1] * 0.95, 0.07, 'healthy gap',
            ha='right', va='center', fontsize=9, color='#339933', fontstyle='italic')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss − Train Loss  (gap)')
    ax.set_title('MuLoCo K=1 — Generalization Gap Over Training')
    ax.legend(fontsize=8.5, loc='upper left', framealpha=0.95)
    fig.tight_layout()
    path = os.path.join(OUTDIR, 'pretty_gen_gap.png')
    fig.savefig(path)
    plt.close()
    print(f'Saved {path}')


# ─── PLOT 4  — Sync‑interval comparison (12 ep, olr=0.5) ────────────────────
def plot_sync_interval(runs):
    targets = {
        'H5':  [r for r in runs if 'muloco_olr0.5_omom0.5_H5' in r['filename'] and r['cluster'] == 'fir'],
        'H10': [r for r in runs if 'muloco_olr0.5_omom0.5_H10' in r['filename'] and r['cluster'] == 'fir'],
        'H30': [r for r in runs if 'muloco_olr0.5_omom0.5_H30' in r['filename'] and r['cluster'] == 'fir'],
        'H50': [r for r in runs if 'muloco_olr0.5_omom0.5_H50' in r['filename'] and r['cluster'] == 'fir'],
    }

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (name, rlist) in enumerate(targets.items()):
        if not rlist:
            continue
        run = rlist[0]
        if len(run['val_epochs']) < 4:
            continue
        final = run['val_losses'][-1]
        ax.plot(run['val_epochs'], run['val_losses'], 'o-',
                color=PALETTE[i], ms=6, lw=2.2,
                label=f'{name.replace("H", "H=")}   →  {final:.3f}')

    add_target_line(ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('MuLoCo K=1 — Effect of Sync Interval H  (olr=0.5, 12 epochs, fir)')
    ax.set_ylim(3.30, 4.65)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95, title='Sync Interval')
    fig.tight_layout()
    path = os.path.join(OUTDIR, 'pretty_sync_interval.png')
    fig.savefig(path)
    plt.close()
    print(f'Saved {path}')


# ─── PLOT 5  — Epoch count / warmdown comparison ─────────────────────────────
def plot_epoch_warmdown(runs):
    picks = {
        '12ep  H=5  (wd_ratio=0.50)': ('muloco_olr0.5_omom0.5_H5', 'fir'),
        '16ep  H=5  (wd_ratio=0.375)': ('p5_e16_olr0.5_omom0.5_H5_lrm0.25_wd1.6_wdr0.375', 'fir'),
        '18ep  H=10 (wd_ratio=0.333)': ('p5_e18_olr0.5_omom0.5_H10_lrm0.25_wd1.6_wdr0.333', 'fir'),
        '20ep  H=10 (wd_ratio=0.30)': ('p5_e20_olr0.5_omom0.5_H10_lrm0.25_wd1.6_wdr0.3', 'fir'),
        '20ep  H=5  (wd_ratio=0.50)': ('p3_e20_olr0.5_omom0.5_H5_lrm0.25_wd1.6', 'fir'),
        '25ep  H=5  (wd_ratio=0.50)': ('p3_e25_olr0.5_omom0.5_H5_lrm0.25_wd1.6', 'fir'),
    }

    fig, ax = plt.subplots(figsize=(13, 7.5))
    for i, (name, (pattern, cluster)) in enumerate(picks.items()):
        match = [r for r in runs if pattern in r['filename'] and r['cluster'] == cluster]
        if not match or len(match[0]['val_epochs']) < 4:
            continue
        run = match[0]
        final = run['val_losses'][-1]
        star = ' ★' if final < TARGET else ''
        ax.plot(run['val_epochs'], run['val_losses'], 'o-',
                color=PALETTE[i % len(PALETTE)], ms=4, lw=2,
                label=f'{name}  →  {final:.3f}{star}')

    add_target_line(ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('MuLoCo K=1 — Epoch Count & Warmdown Schedule Comparison')
    ax.set_ylim(3.25, 4.65)
    ax.legend(fontsize=9.5, loc='upper right', framealpha=0.95, title='Configuration')
    fig.tight_layout()
    path = os.path.join(OUTDIR, 'pretty_epoch_warmdown.png')
    fig.savefig(path)
    plt.close()
    print(f'Saved {path}')


# ─── PLOT 6 — Summary bar chart of final val losses ─────────────────────────
def plot_summary_bar(runs):
    best = sorted(runs, key=lambda r: r['val_losses'][-1])
    best = [r for r in best if len(r['val_losses']) >= 6][:15]

    fig, ax = plt.subplots(figsize=(14, 7))
    names = []
    vals = []
    clrs = []
    for run in reversed(best):
        n = label(run['filename'])
        for old, new in [('muloco_olr', 'olr'), ('_omom0.5', ''), ('_omom0.6', ''),
                         ('_lrm0.25', ''), ('_wd1.6', ''), ('_do0.1', ''),
                         ('_flr0.0', '')]:
            n = n.replace(old, new)
        final = run['val_losses'][-1]
        names.append(f'[{run["cluster"]}] {n}')
        vals.append(final)
        clrs.append('#2ecc71' if final < TARGET else '#e74c3c' if final > 3.42 else '#f39c12')

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, vals, color=clrs, edgecolor='white', height=0.7)
    ax.axvline(x=TARGET, color='#333', ls='--', lw=1.5)
    ax.text(TARGET - 0.002, len(names) + 0.3, f'target {TARGET}',
            ha='right', va='bottom', fontsize=10, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Final Validation Loss')
    ax.set_xlim(3.38, max(vals) + 0.01)
    ax.set_title('MuLoCo K=1 — Top 15 Runs by Final Validation Loss')
    ax.invert_yaxis()

    # Add value labels
    for bar, v in zip(bars, vals):
        beat = ' ✓' if v < TARGET else ''
        ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{v:.4f}{beat}', va='center', fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label=f'Beat target (< {TARGET})'),
        Patch(facecolor='#f39c12', label=f'Near target (< 3.42)'),
        Patch(facecolor='#e74c3c', label=f'Above 3.42'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUTDIR, 'pretty_summary_bar.png')
    fig.savefig(path)
    plt.close()
    print(f'Saved {path}')


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    runs = load_all()
    print(f'Loaded {len(runs)} runs')

    plot_train_val_top4(runs)
    plot_val_best8(runs)
    plot_gen_gap(runs)
    plot_sync_interval(runs)
    plot_epoch_warmdown(runs)
    plot_summary_bar(runs)

    print('\nAll plots saved.')
