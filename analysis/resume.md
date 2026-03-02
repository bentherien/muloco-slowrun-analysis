# Slowrun Sweep Resume State (2026-03-01 ~18:30 EST)

## What You're Doing
Sweep3+4+5 largely complete. 57+ finished, 5 still running on mila (repeatedly preempted). All tamia sweep5 done. Fir cancelled. Monitoring mila stragglers.

## KEY RESULT: Val Loss = 3.3660
**Best: LRM=0.23, WD=1.3, olr=0.5, omom=0.5, H=2, 12 epochs, shuffling ON**
- Target: 3.402 → beat by **0.036** (1.06%)
- Previous best (H=3): 3.3666 → improved by **0.0006** with H=2

## Top 10 Results
| Rank | Config | Val Loss |
|------|--------|----------|
| 1 | LRM=0.23 WD=1.3 **H=2** olr=0.5 | **3.3660** |
| 2 | LRM=0.23 WD=1.3 H=3 olr=0.5 | 3.3666 |
| 3 | LRM=0.23 WD=1.3 H=3 **olr=0.7** | 3.3669 |
| 4 | LRM=0.23 WD=1.3 H=3 **olr=0.6** | 3.3678 |
| 5 | LRM=0.26 WD=1.3 H=3 olr=0.5 | 3.3679 |
| 6 | LRM=0.27 WD=1.3 H=3 olr=0.5 | 3.3680 |
| 7 | LRM=0.23 WD=1.4 H=3 olr=0.5 | 3.3686 |
| 8 | LRM=0.23 WD=1.3 **H=5** olr=0.5 | 3.3688 |
| 9 | LRM=0.23 WD=1.25 H=3 olr=0.5 | 3.3693 |
| 10 | LRM=0.25 WD=1.3 H=3 olr=0.5 | 3.3693 |

## Sweep5 Exploratory Results
### H Comparison (LRM=0.23, WD=1.3, olr=0.5)
| H | Val Loss |
|---|----------|
| **2** | **3.3660** |
| 3 | 3.3666 |
| 5 | 3.3688 |

### Outer LR Comparison (LRM=0.23, WD=1.3, H=3)
| olr | omom | Val Loss |
|-----|------|----------|
| 0.4 | 0.6 | 3.3717 |
| **0.5** | **0.5** | **3.3666** |
| 0.6 | 0.4 | 3.3678 |
| 0.7 | 0.3 | 3.3669 |

### Extended Training
| Epochs | Val Loss | Notes |
|--------|----------|-------|
| **12** | **3.3666** | Optimal |
| 16 | 3.3819 | Worse — overfitting on 100M tokens |

### LRM=0.25 Gap Fill
| WD | Val Loss |
|----|----------|
| 1.0 | 3.3814 |
| 1.3 | 3.3693 (mila) |
| 1.6 | 3.3728 |

## Cluster Status (18:30 EST)
- **Tamia**: All sweep5 done (8/8 complete)
- **Mila**: 3 running (s4_lrm0.23_wd1.15 3.5h, s4_lrm0.19_wd1.3 3.5h, s3_lrm0.22_wd1.6 1h — all preempted multiple times, restarting)
- **Fir**: Cancelled (were stuck 24h+)

## Completed Runs Summary
- **Sweep3**: 34 finished (s3_lrm0.22_wd1.6 still running on mila)
- **Sweep4**: 18 finished (s4_lrm0.23_wd1.25 + s4_lrm0.25_wd1.3 just completed), 2 running (mila)
- **Sweep5**: 8/8 finished on tamia
- **Total**: ~57 completed, 5 still on mila

## Key Observations
1. **H=2 marginally beats H=3** (3.3660 vs 3.3666) — more frequent sync helps
2. **olr=0.5 optimal** but olr=0.7 nearly as good (3.3669 vs 3.3666)
3. **16 epochs worse than 12** (3.3819 vs 3.3666) — overfitting on 100M token dataset
4. **LRM=0.22-0.27 × WD=1.2-1.4** all within 0.003 of best — very flat optimum
5. **38+ out of 57+ runs beat 3.402 target**
6. **Epoch shuffling provides ~0.034 improvement** (3.4004 → 3.3666)
7. **All improvements are marginal** — further HP tuning unlikely to yield much more

## Analysis Files
- `/mnt/raid0/claude/slowrun/analysis/sweep3_plots/` — 6 plot files
- `/mnt/raid0/claude/slowrun/analysis/sweep3_wandb_data.json` — All sweep data
- `/mnt/raid0/claude/slowrun/analysis/sweep3_plots.py` — Plot generation
