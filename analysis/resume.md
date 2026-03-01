# Slowrun Sweep Resume State (2026-03-01 ~07:30 EST)

## What You're Doing
Sweep3 (broad grid) mostly complete. Sweep4 (fine-grained) nearly done on tamia, still running on mila. Fir jobs stuck pending. Monitoring remaining runs.

## KEY RESULT: Val Loss = 3.3666
**Best: LRM=0.23, WD=1.3, olr=0.5, omom=0.5, H=3, 12 epochs, shuffling ON**
- Target: 3.402 → beat by **0.0354** (1.04%)
- Previous best (no shuffle): 3.4004 → improved by **0.0338**

## Optimum Region
The optimum is very flat: LRM=0.22-0.27 × WD=1.2-1.4 all within 0.003 of each other.

### WD=1.3 Results (by LRM)
| LRM | Val Loss |
|-----|----------|
| 0.177 | 3.3834 |
| 0.20 | 3.3777 |
| 0.21 | 3.3711 |
| 0.22 | 3.3696 |
| **0.23** | **3.3666** |
| 0.24 | 3.3696 |
| 0.26 | 3.3679 |
| 0.27 | 3.3680 |

### LRM=0.23 Results (by WD)
| WD | Val Loss |
|----|----------|
| 1.0 | 3.3859 |
| 1.1 | 3.3796 |
| 1.2 | 3.3728 |
| **1.3** | **3.3666** |
| 1.4 | 3.3686 |
| 1.5 | 3.3716 |
| 1.6 | 3.3747 |
| 2.0 | 3.3889 |

## Cluster Status (07:30 EST)
- **Fir**: 12 sweep3 jobs still PENDING (LRM=0.25×6, LRM=0.30×6) — stuck for >6 hours
- **Tamia**: 2 sweep4 running (s4_lrm0.24_wd1.0, s4_lrm0.24_wd1.6), 0 pending
- **Mila**: 4 sweep4 running (s4_lrm0.19_wd1.3, s4_lrm0.23_wd1.15, s4_lrm0.23_wd1.25, s4_lrm0.25_wd1.3)

## Completed Runs Summary
- **Sweep3**: 33 finished + 1 running (mila preempted)
- **Sweep4**: 14 finished, 6 running
- **Total**: 47 completed configs across sweep3+sweep4

## Analysis Files
- `/mnt/raid0/claude/slowrun/analysis/sweep3_plots/` — Updated plots (6 files)
- `/mnt/raid0/claude/slowrun/analysis/sweep3_wandb_data.json` — 58 runs
- `/mnt/raid0/claude/slowrun/analysis/sweep3_plots.py` — Plot generation

## Key Observations
1. **Epoch shuffling provides ~0.034 loss improvement** (3.4004 → 3.3666)
2. **WD=1.3 is optimal** (was 1.6 without shuffling)
3. **LRM=0.23 is optimal** but the landscape is very flat (0.22-0.27 all within 0.003)
4. **Lower LRM (0.177) prefers higher WD (1.6-2.0)** — the optimal WD shifts with LRM
5. **WD > 2.5 always worse**, WD < 1.0 also suboptimal
6. **H=3 sync interval confirmed** as optimal from sweep2
