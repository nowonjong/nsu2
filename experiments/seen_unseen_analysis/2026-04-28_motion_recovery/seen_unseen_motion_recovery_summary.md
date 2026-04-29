# Seen/Unseen Motion and Recovery Analysis

- Compared models: Vision-only vs Feature Redesigned Fusion
- Practical datasets: final practical 10-shot sets
- Recovery counts are accumulated over 5 seeds and aligned per sample.
- Motion variability is computed once per practical capture from stored `.npy` sequences.

## Vision Error Recovery

| Condition | Total sample-seed cases | Vision wrong | Fusion recovered | Recovery / Vision errors | Vision correct -> Fusion wrong |
|---|---:|---:|---:|---:|---:|
| Seen | 1800 | 348 | 276 | 0.7931 | 82 |
| Unseen | 1800 | 719 | 443 | 0.6161 | 109 |

## Motion Variability

| Condition | Samples | Vision motion | Vision path length | Flex range | Flex std | IMU range | IMU std |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seen | 360 | 1.0203 | 59.1862 | 0.4012 | 0.1479 | 87.7346 | 22.6797 |
| Unseen | 360 | 1.0096 | 58.6788 | 0.4534 | 0.1638 | 83.7820 | 22.0121 |

## Interpretation Note

This analysis supports a cautious interpretation: subject-dependent motion and sensor-pattern differences are present in the practical captures, and Feature Redesigned Fusion recovers a substantial portion of Vision-only errors, especially in the Unseen condition.

## Files

- JSON: `C:\SignProject\experiments\seen_unseen_analysis\2026-04-28_motion_recovery\seen_unseen_motion_recovery_summary.json`
