# Slowrun Sweep FINAL Results (2026-03-02 ~13:15 EST)

## STATUS: ALL 62 CONFIGS COMPLETE

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

### Gap Fill Results
| Config | Val Loss | Source |
|--------|----------|--------|
| LRM=0.25 WD=1.0 | 3.3814 | tamia sweep5 |
| LRM=0.25 WD=1.3 | 3.3693 | mila sweep4 |
| LRM=0.25 WD=1.6 | 3.3728 | tamia sweep5 |
| LRM=0.23 WD=1.15 | 3.3747 | mila sweep4 |
| LRM=0.19 WD=1.3 | 3.3767 | mila sweep4 |
| LRM=0.22 WD=1.6 | 3.3734 | tamia (mila kept getting preempted) |

## Completed Runs Summary
- **Sweep3**: 35 finished (LRM×WD grid with shuffling, H=3)
- **Sweep4**: 19 finished (interpolation points)
- **Sweep5**: 8 finished (H values, olr variants, extended epochs, gap fills)
- **Total**: **62 unique configs complete**, 50+ beat 3.402 target

## Key Observations
1. **H=2 marginally beats H=3** (3.3660 vs 3.3666) — more frequent sync helps
2. **olr=0.5 optimal** but olr=0.7 nearly as good (3.3669 vs 3.3666)
3. **16 epochs worse than 12** (3.3819 vs 3.3666) — overfitting on 100M token dataset
4. **LRM=0.22-0.27 × WD=1.2-1.4** all within 0.003 of best — very flat optimum
5. **50+ out of 62 runs beat 3.402 target**
6. **Epoch shuffling provides ~0.034 improvement** (3.4004 → 3.3666)
7. **All improvements are marginal** — further HP tuning unlikely to yield much more
8. **Mila long partition extremely preemption-prone** — s3_lrm0.22_wd1.6 preempted 5+ times

## Analysis Files
- `/mnt/raid0/claude/slowrun/analysis/sweep3_plots/` — 6 plot files
- `/mnt/raid0/claude/slowrun/analysis/sweep3_wandb_data.json` — All sweep data
- `/mnt/raid0/claude/slowrun/analysis/sweep3_plots.py` — Plot generation
- GitHub: https://github.com/bentherien/muloco-slowrun-analysis
