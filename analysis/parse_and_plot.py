#!/usr/bin/env python3
"""Parse MuLoCo slowrun training logs and analyze overfitting + plot curves."""

import re
import os
import glob
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_log(filepath):
    """Parse a single log file to extract train loss, val loss, and metadata."""
    train_steps = []
    train_losses = []
    val_steps = []
    val_losses = []
    val_epochs = []
    val_bpbs = []
    config = {}

    with open(filepath) as f:
        for line in f:
            # Parse config line
            m = re.match(r'Config:\s*(.*)', line)
            if m:
                for pair in re.findall(r'(\w+)=([\d.]+)', m.group(1)):
                    config[pair[0]] = pair[1]

            # Parse training step: step 00123 (5.38%) | loss: 3.912102 | ...
            m = re.match(r'step\s+(\d+)\s+\([\d.]+%\)\s+\|\s+loss:\s+([\d.]+)', line)
            if m:
                train_steps.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))

            # Parse val loss: Step 00191 | Epoch 1 | Val BPB: 1.457034 | Val Loss: 4.483177
            m = re.match(r'Step\s+(\d+)\s+\|\s+Epoch\s+(\d+)\s+\|\s+Val BPB:\s+([\d.]+)\s+\|\s+Val Loss:\s+([\d.]+)', line)
            if m:
                val_steps.append(int(m.group(1)))
                val_epochs.append(int(m.group(2)))
                val_bpbs.append(float(m.group(3)))
                val_losses.append(float(m.group(4)))

    return {
        'train_steps': np.array(train_steps),
        'train_losses': np.array(train_losses),
        'val_steps': np.array(val_steps),
        'val_losses': np.array(val_losses),
        'val_epochs': np.array(val_epochs),
        'val_bpbs': np.array(val_bpbs),
        'config': config,
        'filename': os.path.basename(filepath),
    }


def smooth(y, window=50):
    """Simple moving average smoothing."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')


def get_run_label(filename):
    """Generate a readable label from filename."""
    name = filename.replace('.out', '')
    # Remove job ID suffix
    name = re.sub(r'_\d{6,}$', '', name)
    return name


# ==================== MAIN ====================

# Gather all logs
log_dirs = [
    '/mnt/raid0/claude/slowrun/analysis/logs/fir_logs',
    '/mnt/raid0/claude/slowrun/analysis/logs/tamia_logs',
]

all_runs = []
for log_dir in log_dirs:
    cluster = 'fir' if 'fir' in log_dir else 'tamia'
    for f in sorted(glob.glob(os.path.join(log_dir, '*.out'))):
        data = parse_log(f)
        if len(data['val_losses']) > 0:  # Only runs with val data
            data['cluster'] = cluster
            all_runs.append(data)

print(f"Parsed {len(all_runs)} runs total")

# ==================== OVERFITTING ANALYSIS ====================
print("\n" + "="*80)
print("OVERFITTING ANALYSIS")
print("="*80)

# For each run: compare train loss (smoothed) near each val eval with val loss
print(f"\n{'Run':<65} | {'Final Train':>11} | {'Final Val':>10} | {'Gap':>7} | {'Trend':>12}")
print("-"*120)

best_runs = []
for run in sorted(all_runs, key=lambda r: r['val_losses'][-1] if len(r['val_losses']) > 0 else 999):
    if len(run['val_losses']) < 2 or len(run['train_losses']) < 100:
        continue

    label = f"[{run['cluster']}] {get_run_label(run['filename'])}"
    final_val = run['val_losses'][-1]

    # Compute smoothed train loss around the final validation step
    final_val_step = run['val_steps'][-1]
    mask = (run['train_steps'] >= final_val_step - 100) & (run['train_steps'] <= final_val_step)
    if mask.sum() > 0:
        final_train = run['train_losses'][mask].mean()
    else:
        final_train = run['train_losses'][-100:].mean()

    gap = final_val - final_train

    # Check if val loss is increasing while train loss decreasing (overfitting)
    if len(run['val_losses']) >= 4:
        last_4_val = run['val_losses'][-4:]
        val_trend = last_4_val[-1] - last_4_val[0]
        if val_trend > 0:
            trend = "OVERFITTING!"
        elif val_trend > -0.01:
            trend = "plateauing"
        else:
            trend = "improving"
    else:
        trend = "too few"

    print(f"{label:<65} | {final_train:>11.4f} | {final_val:>10.4f} | {gap:>+7.3f} | {trend:>12}")
    best_runs.append(run)

# ==================== DETAILED OVERFITTING CHECK ====================
print("\n" + "="*80)
print("TRAIN vs VAL GAP OVER TIME (best runs)")
print("="*80)

# Focus on the top runs
top_runs = [r for r in all_runs if len(r['val_losses']) >= 6 and r['val_losses'][-1] < 3.50]
top_runs.sort(key=lambda r: r['val_losses'][-1])

for run in top_runs[:10]:
    label = f"[{run['cluster']}] {get_run_label(run['filename'])}"
    print(f"\n{label}")
    print(f"  {'Epoch':>6} | {'Val Loss':>9} | {'~Train Loss':>11} | {'Gap':>7} | {'Val delta':>9}")

    prev_val = None
    for i, (vstep, vepoch, vloss) in enumerate(zip(run['val_steps'], run['val_epochs'], run['val_losses'])):
        # Get average train loss near this val step
        mask = (run['train_steps'] >= vstep - 50) & (run['train_steps'] <= vstep)
        if mask.sum() > 0:
            avg_train = run['train_losses'][mask].mean()
        else:
            avg_train = float('nan')

        gap = vloss - avg_train
        delta = f"{vloss - prev_val:+.4f}" if prev_val is not None else "    -"
        prev_val = vloss

        print(f"  {vepoch:>6} | {vloss:>9.4f} | {avg_train:>11.4f} | {gap:>+7.3f} | {delta:>9}")

# ==================== PLOTTING ====================
outdir = '/mnt/raid0/claude/slowrun/analysis'

# Select the best runs for plotting (val loss < 3.50 and enough data)
plot_runs = [r for r in all_runs if len(r['val_losses']) >= 6 and r['val_losses'][-1] < 3.50]
plot_runs.sort(key=lambda r: r['val_losses'][-1])
plot_runs = plot_runs[:8]  # Top 8

# Color palette
colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4', '#795548', '#E91E63']

# ---- Plot 1: Validation Loss vs Epoch (best runs) ----
fig, ax = plt.subplots(figsize=(12, 7))
for i, run in enumerate(plot_runs):
    label = f"[{run['cluster']}] {get_run_label(run['filename'])}"
    # Shorten label
    short = label.replace('muloco_', '').replace('_lrm0.25_wd1.6', '').replace('_do0.1', '')
    ax.plot(run['val_epochs'], run['val_losses'], 'o-', color=colors[i % len(colors)],
            label=f"{short} ({run['val_losses'][-1]:.3f})", markersize=4, linewidth=1.8)

ax.axhline(y=3.402, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (3.402)')
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Validation Loss', fontsize=13)
ax.set_title('MuLoCo K=1 — Best Runs: Validation Loss vs Epoch', fontsize=15, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=3.30, top=4.10)
fig.tight_layout()
fig.savefig(os.path.join(outdir, 'val_loss_best_runs.png'), dpi=150)
print(f"\nSaved: {outdir}/val_loss_best_runs.png")
plt.close()

# ---- Plot 2: Train + Val Loss for top 4 runs (overfitting diagnostic) ----
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for idx, run in enumerate(plot_runs[:4]):
    ax = axes[idx // 2][idx % 2]
    label = get_run_label(run['filename'])

    # Smoothed train loss
    if len(run['train_steps']) > 50:
        sm_train = smooth(run['train_losses'], window=50)
        sm_steps = run['train_steps'][49:]  # align with smoothed output
        ax.plot(sm_steps, sm_train, color=colors[0], alpha=0.8, linewidth=1.2, label='Train Loss (smoothed)')

    # Raw train loss (light)
    ax.plot(run['train_steps'], run['train_losses'], color=colors[0], alpha=0.1, linewidth=0.3)

    # Val loss
    ax.plot(run['val_steps'], run['val_losses'], 'o-', color=colors[1], markersize=6,
            linewidth=2, label='Val Loss', zorder=5)

    ax.axhline(y=3.402, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Target')
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title(f'[{run["cluster"]}] {label}\nFinal Val: {run["val_losses"][-1]:.4f}',
                fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('MuLoCo K=1 — Train vs Val Loss (Top 4 Runs)', fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(outdir, 'train_val_loss_top4.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {outdir}/train_val_loss_top4.png")
plt.close()

# ---- Plot 3: Train-Val Gap Over Epochs (overfitting diagnostic) ----
fig, ax = plt.subplots(figsize=(12, 7))
for i, run in enumerate(plot_runs[:6]):
    label = get_run_label(run['filename'])
    short = label.replace('muloco_', '').replace('_lrm0.25_wd1.6', '').replace('_do0.1', '')

    gaps = []
    for vstep, vloss in zip(run['val_steps'], run['val_losses']):
        mask = (run['train_steps'] >= vstep - 50) & (run['train_steps'] <= vstep)
        if mask.sum() > 0:
            avg_train = run['train_losses'][mask].mean()
            gaps.append(vloss - avg_train)
        else:
            gaps.append(float('nan'))

    ax.plot(run['val_epochs'], gaps, 'o-', color=colors[i % len(colors)],
            label=f"[{run['cluster']}] {short}", markersize=4, linewidth=1.5)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Val Loss - Train Loss (Gap)', fontsize=13)
ax.set_title('MuLoCo K=1 — Generalization Gap Over Training', fontsize=15, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(outdir, 'generalization_gap.png'), dpi=150)
print(f"Saved: {outdir}/generalization_gap.png")
plt.close()

# ---- Plot 4: Comparison of all H values at 12 epochs (olr=0.5) ----
h_runs = [r for r in all_runs
          if 'muloco_olr0.5' in r['filename'] and r['cluster'] == 'fir'
          and len(r['val_losses']) >= 10]
h_runs.sort(key=lambda r: r['val_losses'][-1])

if h_runs:
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, run in enumerate(h_runs):
        label = get_run_label(run['filename']).replace('muloco_olr0.5_omom0.5_', 'H=').split('_')[0]
        ax.plot(run['val_epochs'], run['val_losses'], 'o-', color=colors[i % len(colors)],
                label=f"{label} (final: {run['val_losses'][-1]:.3f})", markersize=5, linewidth=2)

    ax.axhline(y=3.402, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (3.402)')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Validation Loss', fontsize=13)
    ax.set_title('MuLoCo K=1 — Sync Interval Comparison (olr=0.5, 12 epochs)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=3.30, top=4.60)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'sync_interval_comparison.png'), dpi=150)
    print(f"Saved: {outdir}/sync_interval_comparison.png")
    plt.close()

# ---- Plot 5: Phase comparison (12ep vs 15ep vs 20ep vs 25ep) ----
phase_runs = {
    '12ep H=5': next((r for r in all_runs if 'muloco_olr0.5_omom0.5_H5' in r['filename'] and r['cluster'] == 'fir'), None),
    '15ep H=20': next((r for r in all_runs if 'p2_e15_olr0.5_omom0.5_H20' in r['filename'] and r['cluster'] == 'fir'), None),
    '20ep H=5': next((r for r in all_runs if 'p3_e20_olr0.5_omom0.5_H5_lrm0.25' in r['filename'] and r['cluster'] == 'fir'), None),
    '25ep H=5': next((r for r in all_runs if 'p3_e25_olr0.5_omom0.5_H5' in r['filename'] and r['cluster'] == 'fir'), None),
    '20ep H=10': next((r for r in all_runs if 'p3_e20_olr0.5_omom0.5_H10_lrm0.25' in r['filename'] and r['cluster'] == 'fir'), None),
}

fig, ax = plt.subplots(figsize=(12, 7))
for i, (name, run) in enumerate(phase_runs.items()):
    if run is None or len(run['val_losses']) < 3:
        continue
    ax.plot(run['val_epochs'], run['val_losses'], 'o-', color=colors[i % len(colors)],
            label=f"{name} (final: {run['val_losses'][-1]:.3f})", markersize=4, linewidth=2)

ax.axhline(y=3.402, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (3.402)')
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Validation Loss', fontsize=13)
ax.set_title('MuLoCo K=1 — Epoch Count Comparison (olr=0.5)', fontsize=15, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=3.20, top=4.60)
fig.tight_layout()
fig.savefig(os.path.join(outdir, 'epoch_comparison.png'), dpi=150)
print(f"Saved: {outdir}/epoch_comparison.png")
plt.close()

print("\n" + "="*80)
print("SUMMARY OF KEY FINDINGS")
print("="*80)

# Print top 15 runs by final val loss
print(f"\n{'Rank':>4} | {'Run':<65} | {'Final Val':>10} | {'Epochs':>6}")
print("-"*100)
sorted_runs = sorted(all_runs, key=lambda r: r['val_losses'][-1] if len(r['val_losses']) > 0 else 999)
for i, run in enumerate(sorted_runs[:20]):
    if len(run['val_losses']) == 0:
        continue
    label = f"[{run['cluster']}] {get_run_label(run['filename'])}"
    epochs = int(run['val_epochs'][-1]) if len(run['val_epochs']) > 0 else 0
    print(f"{i+1:>4} | {label:<65} | {run['val_losses'][-1]:>10.4f} | {epochs:>6}")
