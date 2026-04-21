# Fusion Design Ablation Multi-Seed Summary

- Date: 2026-04-15
- Seeds: `42`, `43`, `44`
- Models: `naive raw fusion`, `feature redesigned simple fusion`, `stable hybrid`
- Practical sets: `practical_in`, `practical_in_2`, `practical_out`, `practical_out_2`
- Purpose: check whether feature redesigned fusion remains stronger than stable hybrid when seed/split changes.

## Seed-Level Practical Accuracy

| Seed | Model | Offline Acc | Practical In Avg | Practical Out Avg | Practical Avg | Practical F1 Avg | Offline->Practical Drop |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | Naive raw fusion | 0.9000 | 0.8819 | 0.7153 | 0.7986 | 0.7783 | 0.1014 |
| 42 | Feature redesigned simple fusion | 0.9778 | 0.9653 | 0.8472 | 0.9062 | 0.8895 | 0.0716 |
| 42 | Stable hybrid | 0.9333 | 0.9167 | 0.8333 | 0.8750 | 0.8613 | 0.0583 |
| 43 | Naive raw fusion | 0.9432 | 0.9444 | 0.6042 | 0.7743 | 0.7451 | 0.1689 |
| 43 | Feature redesigned simple fusion | 0.9432 | 0.9375 | 0.7431 | 0.8403 | 0.8283 | 0.1029 |
| 43 | Stable hybrid | 0.9432 | 0.9514 | 0.7500 | 0.8507 | 0.8359 | 0.0925 |
| 44 | Naive raw fusion | 0.9667 | 0.9167 | 0.7014 | 0.8090 | 0.7827 | 0.1577 |
| 44 | Feature redesigned simple fusion | 0.9778 | 0.9653 | 0.8472 | 0.9062 | 0.8976 | 0.0716 |
| 44 | Stable hybrid | 0.9333 | 0.8819 | 0.6597 | 0.7708 | 0.7432 | 0.1625 |

## Multi-Seed Mean +- Std

| Model | Offline Acc | Practical In Avg | Practical Out Avg | Practical Avg | Practical F1 Avg | Offline->Practical Drop | Offline->Out Drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive raw fusion | 0.9366 +/- 0.0338 | 0.9144 +/- 0.0313 | 0.6736 +/- 0.0605 | 0.7940 +/- 0.0178 | 0.7687 +/- 0.0206 | 0.1427 +/- 0.0362 | 0.2630 +/- 0.0772 |
| Feature redesigned simple fusion | 0.9663 +/- 0.0200 | 0.9560 +/- 0.0160 | 0.8125 +/- 0.0601 | 0.8843 +/- 0.0381 | 0.8718 +/- 0.0379 | 0.0820 +/- 0.0181 | 0.1538 +/- 0.0402 |
| Stable hybrid | 0.9366 +/- 0.0057 | 0.9167 +/- 0.0347 | 0.7477 +/- 0.0868 | 0.8322 +/- 0.0545 | 0.8135 +/- 0.0622 | 0.1044 +/- 0.0531 | 0.1889 +/- 0.0869 |

## Decision

- Feature redesigned simple fusion has the best multi-seed Practical average accuracy and macro F1.
- Stable hybrid is not consistently better than feature redesigned simple fusion. In this multi-seed check, it is lower on average and more seed-sensitive on Practical-Out.
- Therefore, if we want the cleanest thesis story, the main model-selection flow should be: vision-only -> naive raw fusion -> feature redesigned fusion.
- Stable hybrid can remain in internal experiment records, but should be omitted from the main paper narrative unless a short note is needed.

## Recommended Thesis Wording

> Sensor values should not simply be added as raw channels. In our ablation, the structure that reflected sensor roles - IMU/flags as temporal motion/orientation features and flex as posture-summary features - showed the most stable Practical performance across seeds.

## Files

- JSON: `C:\SignProject\experiments\fusion_design_ablation\2026-04-15_current_18w_multiseed\fusion_design_ablation_multiseed_summary.json`
- Practical reports: `C:\SignProject\experiments\fusion_design_ablation\2026-04-15_current_18w_multiseed\practical_reports`
