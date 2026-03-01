#!/usr/bin/env python3
"""Sweep3 (shuffling + H=3) analysis plots."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import re

OUT = Path('/mnt/raid0/claude/slowrun/analysis/sweep3_plots')
OUT.mkdir(exist_ok=True)

with open('/mnt/raid0/claude/slowrun/analysis/sweep3_wandb_data.json') as f:
    data = json.load(f)

# Parse configs from run names
def parse_config(name):
    lrm = re.search(r'lrm([\d.]+)', name)
    wd = re.search(r'wd([\d.]+)', name)
    h = re.search(r'_h(\d+)', name)
    return {
        'lrm': float(lrm.group(1)) if lrm else None,
        'wd': float(wd.group(1)) if wd else None,
        'h': int(h.group(1)) if h else 3,  # default H=3 for fir runs
    }

# Get finished runs with valid final loss
finished = {}
for name, rd in data.items():
    if rd['state'] == 'finished' and rd['history']:
        vals = [p['val/loss'] for p in rd['history'] if p.get('val/loss') is not None]
        if vals and isinstance(vals[-1], (int, float)) and not np.isnan(float(vals[-1])):
            finished[name] = {**rd, 'final_val': vals[-1], **parse_config(name)}

running = {}
for name, rd in data.items():
    if rd['state'] == 'running' and rd['history']:
        vals = [p['val/loss'] for p in rd['history'] if p.get('val/loss') is not None]
        if vals:
            running[name] = {**rd, 'current_val': vals[-1], **parse_config(name)}

print(f"Finished: {len(finished)}, Running: {len(running)}")

# ===== Plot 1: Top 10 validation curves =====
fig, ax = plt.subplots(figsize=(12, 7))
top10 = sorted(finished.items(), key=lambda x: x[1]['final_val'])[:10]
colors = plt.cm.viridis(np.linspace(0, 0.9, 10))
for i, (name, rd) in enumerate(top10):
    steps = [p['_step'] for p in rd['history'] if p.get('val/loss') is not None]
    vals = [p['val/loss'] for p in rd['history'] if p.get('val/loss') is not None]
    ax.plot(steps, vals, color=colors[i], linewidth=1.5,
            label=f"{name}: {rd['final_val']:.4f}")
ax.axhline(y=3.402, color='red', linestyle='--', alpha=0.7, label='Target: 3.402')
ax.axhline(y=3.4004, color='orange', linestyle='--', alpha=0.7, label='Sweep2 best: 3.4004 (no shuffle)')
ax.set_xlabel('Step')
ax.set_ylabel('Validation Loss')
ax.set_title('Sweep3 (Shuffling + H=3): Top 10 Val Curves')
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(3.3, 4.0)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'val_curves_top10.png', dpi=150)
plt.close()
print("Saved val_curves_top10.png")

# ===== Plot 2: LR x WD heatmap (H=3 runs only) =====
h3_runs = {n: r for n, r in finished.items() if r.get('h') == 3 or r.get('h') is None}
lrms = sorted(set(r['lrm'] for r in h3_runs.values() if r['lrm']))
wds = sorted(set(r['wd'] for r in h3_runs.values() if r['wd']))

if lrms and wds:
    grid = np.full((len(lrms), len(wds)), np.nan)
    for name, rd in h3_runs.items():
        if rd['lrm'] in lrms and rd['wd'] in wds:
            i = lrms.index(rd['lrm'])
            j = wds.index(rd['wd'])
            grid[i, j] = rd['final_val']

    fig, ax = plt.subplots(figsize=(10, 7))
    valid_vals = grid[~np.isnan(grid)]
    if len(valid_vals) > 0:
        vmin, vmax = valid_vals.min(), min(valid_vals.max(), 3.5)
        im = ax.imshow(grid, cmap='RdYlGn_r', aspect='auto', vmin=vmin, vmax=vmax)
        for i in range(len(lrms)):
            for j in range(len(wds)):
                if not np.isnan(grid[i, j]):
                    color = 'white' if grid[i, j] > (vmin + vmax) / 2 else 'black'
                    ax.text(j, i, f'{grid[i, j]:.4f}', ha='center', va='center',
                            fontsize=8, fontweight='bold', color=color)
                else:
                    ax.text(j, i, 'N/A', ha='center', va='center', fontsize=7, color='gray')
        ax.set_xticks(range(len(wds)))
        ax.set_xticklabels([str(w) for w in wds])
        ax.set_yticks(range(len(lrms)))
        ax.set_yticklabels([str(l) for l in lrms])
        ax.set_xlabel('Weight Decay')
        ax.set_ylabel('LR Multiplier')
        ax.set_title('Sweep3: LR x WD Heatmap (olr=0.5, omom=0.5, H=3, with shuffling)')
        plt.colorbar(im, label='Val Loss')
        fig.tight_layout()
        fig.savefig(OUT / 'lr_wd_heatmap.png', dpi=150)
        plt.close()
        print("Saved lr_wd_heatmap.png")

# ===== Plot 3: Shuffling comparison (sweep2 vs sweep3 same configs) =====
# Compare key configs between sweep2 (no shuffle) and sweep3 (shuffle)
sweep2_results = {
    'olr0.5_omom0.5_H3': 3.4004,
    'olr0.6_omom0.4_H5': 3.4034,
    'olr0.5_omom0.5_H2': 3.4054,
    'olr0.5_omom0.5_H10': 3.4065,
    'baseline_H5': 3.4028,
}
sweep3_matches = {
    'olr0.5_omom0.5_H3': finished.get('s3_lrm0.25_wd1.6_h3', {}).get('final_val'),
    'olr0.6_omom0.4_H5': finished.get('s3_lrm0.25_wd1.6_h5_o64', {}).get('final_val'),
    'olr0.5_omom0.5_H2': finished.get('s3_lrm0.25_wd1.6_h2', {}).get('final_val'),
}

fig, ax = plt.subplots(figsize=(10, 6))
labels = []
s2_vals = []
s3_vals = []
for key in sweep2_results:
    if key in sweep3_matches and sweep3_matches[key] is not None:
        labels.append(key)
        s2_vals.append(sweep2_results[key])
        s3_vals.append(sweep3_matches[key])

x = np.arange(len(labels))
width = 0.35
bars1 = ax.bar(x - width/2, s2_vals, width, label='Sweep2 (no shuffle)', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, s3_vals, width, label='Sweep3 (with shuffle)', color='#2ecc71', alpha=0.8)
for bar, val in zip(bars1, s2_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f'{val:.4f}',
            ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars2, s3_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f'{val:.4f}',
            ha='center', va='bottom', fontsize=8)
ax.axhline(y=3.402, color='red', linestyle='--', alpha=0.5, label='Target: 3.402')
ax.set_ylabel('Val Loss')
ax.set_title('Impact of Epoch Shuffling: Sweep2 vs Sweep3')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)
ax.legend()
ax.set_ylim(3.35, 3.42)
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(OUT / 'shuffle_comparison.png', dpi=150)
plt.close()
print("Saved shuffle_comparison.png")

# ===== Plot 4: Ranked bar chart of all finished runs =====
fig, ax = plt.subplots(figsize=(14, 8))
sorted_runs = sorted(finished.items(), key=lambda x: x[1]['final_val'])
names = [n for n, _ in sorted_runs]
vals = [r['final_val'] for _, r in sorted_runs]
colors_bar = ['#2ecc71' if v < 3.402 else '#e74c3c' for v in vals]
bars = ax.barh(range(len(names)), vals, color=colors_bar, alpha=0.8)
ax.axvline(x=3.402, color='red', linestyle='--', linewidth=2, label='Target: 3.402')
ax.axvline(x=3.4004, color='orange', linestyle='--', linewidth=1.5, label='Sweep2 best: 3.4004')
for i, (name, val) in enumerate(zip(names, vals)):
    ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=7)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel('Val Loss')
ax.set_title('Sweep3 (Shuffling): All Finished Runs Ranked')
ax.legend(loc='lower right')
ax.set_xlim(3.35, max(3.5, max(v for v in vals if not np.isnan(v)) + 0.02))
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(OUT / 'summary_bar.png', dpi=150)
plt.close()
print("Saved summary_bar.png")

# ===== Plot 5: WD comparison at each LR =====
lrm_groups = {}
for name, rd in finished.items():
    lrm = rd.get('lrm')
    if lrm and rd.get('h') in [3, None]:
        lrm_groups.setdefault(lrm, []).append((name, rd))

lrms_plot = sorted([l for l in lrm_groups if len(lrm_groups[l]) >= 3])[:6]
if lrms_plot:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, lrm in enumerate(lrms_plot):
        ax = axes[idx]
        runs_sorted = sorted(lrm_groups[lrm], key=lambda x: x[1]['wd'])
        wds_plot = [r['wd'] for _, r in runs_sorted]
        vals_plot = [r['final_val'] for _, r in runs_sorted]
        ax.bar(range(len(wds_plot)), vals_plot,
               color=['#2ecc71' if v < 3.402 else '#3498db' for v in vals_plot], alpha=0.8)
        ax.set_xticks(range(len(wds_plot)))
        ax.set_xticklabels([str(w) for w in wds_plot], fontsize=8)
        ax.set_title(f'LRM={lrm}', fontweight='bold')
        ax.set_ylabel('Val Loss')
        ax.set_xlabel('Weight Decay')
        ax.axhline(y=3.402, color='red', linestyle='--', alpha=0.5)
        for i, v in enumerate(vals_plot):
            ax.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=7)
        ax.set_ylim(3.35, max(vals_plot) + 0.02 if max(vals_plot) < 3.6 else 3.55)
    for idx in range(len(lrms_plot), 6):
        axes[idx].set_visible(False)
    fig.suptitle('Sweep3: Weight Decay Comparison at Each LR', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'wd_comparison.png', dpi=150)
    plt.close()
    print("Saved wd_comparison.png")

# ===== Plot 6: Train vs Val (overfitting analysis) for top 4 =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
top4 = sorted(finished.items(), key=lambda x: x[1]['final_val'])[:4]
for idx, (name, rd) in enumerate(top4):
    ax = axes[idx // 2][idx % 2]
    steps_v = [p['_step'] for p in rd['history'] if p.get('val/loss') is not None]
    vals_v = [p['val/loss'] for p in rd['history'] if p.get('val/loss') is not None]
    steps_t = [p['_step'] for p in rd['history'] if p.get('train/loss') is not None]
    vals_t = [p['train/loss'] for p in rd['history'] if p.get('train/loss') is not None]
    ax.plot(steps_v, vals_v, 'b-', alpha=0.8, label='Val', linewidth=1.5)
    ax.plot(steps_t, vals_t, 'r-', alpha=0.5, label='Train', linewidth=1)
    ax.set_title(f'{name}\nFinal val: {rd["final_val"]:.4f}', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
fig.suptitle('Sweep3 Top 4: Train vs Val (Overfitting Check)', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'train_val_top4.png', dpi=150)
plt.close()
print("Saved train_val_top4.png")

print(f"\nAll plots saved to {OUT}/")
print(f"\nBest result: {top10[0][0]} = {top10[0][1]['final_val']:.4f}")
