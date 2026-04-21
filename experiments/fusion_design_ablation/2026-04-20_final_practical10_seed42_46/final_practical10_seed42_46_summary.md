# Final Practical 10-Shot Multi-Seed Summary

- Seeds: `42`, `43`, `44`, `45`, `46`
- Practical datasets: final 10-shot sets, each `18 words x 10 samples`
- Models: vision-only, naive raw fusion, feature redesigned simple fusion, stable hybrid
- Note: stable hybrid is evaluated as a candidate, not assumed as the final model.
- Add6 samples remain separable via each dataset manifest and can be excluded later if needed.

## Seed-Level Results

| Seed | Model | Offline Acc | In Avg | Out Avg | Practical Avg | Practical F1 Avg | Drop |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | Glove vision-only | 0.8889 | 0.8389 | 0.6167 | 0.7278 | 0.6956 | 0.1611 |
| 42 | Naive raw fusion | 0.9000 | 0.8694 | 0.7444 | 0.8069 | 0.7873 | 0.0931 |
| 42 | Feature redesigned simple fusion | 0.9444 | 0.9389 | 0.8056 | 0.8722 | 0.8597 | 0.0722 |
| 42 | Stable hybrid | 0.9778 | 0.8861 | 0.7861 | 0.8361 | 0.8306 | 0.1417 |
| 43 | Glove vision-only | 0.9318 | 0.8000 | 0.5500 | 0.6750 | 0.6317 | 0.2568 |
| 43 | Naive raw fusion | 0.9318 | 0.9056 | 0.6694 | 0.7875 | 0.7687 | 0.1443 |
| 43 | Feature redesigned simple fusion | 0.9659 | 0.9222 | 0.7389 | 0.8306 | 0.8176 | 0.1353 |
| 43 | Stable hybrid | 0.9545 | 0.9306 | 0.7611 | 0.8458 | 0.8312 | 0.1087 |
| 44 | Glove vision-only | 0.9333 | 0.8000 | 0.6028 | 0.7014 | 0.6616 | 0.2319 |
| 44 | Naive raw fusion | 0.9667 | 0.8806 | 0.7056 | 0.7931 | 0.7775 | 0.1736 |
| 44 | Feature redesigned simple fusion | 0.9444 | 0.9000 | 0.7833 | 0.8417 | 0.8206 | 0.1027 |
| 44 | Stable hybrid | 0.9333 | 0.8417 | 0.6667 | 0.7542 | 0.7303 | 0.1791 |
| 45 | Glove vision-only | 0.8523 | 0.7861 | 0.6000 | 0.6931 | 0.6531 | 0.1592 |
| 45 | Naive raw fusion | 0.9545 | 0.9083 | 0.8111 | 0.8597 | 0.8559 | 0.0948 |
| 45 | Feature redesigned simple fusion | 0.9545 | 0.9472 | 0.8000 | 0.8736 | 0.8648 | 0.0809 |
| 45 | Stable hybrid | 0.9545 | 0.9417 | 0.7667 | 0.8542 | 0.8420 | 0.1003 |
| 46 | Glove vision-only | 0.9889 | 0.8083 | 0.6333 | 0.7208 | 0.6955 | 0.2681 |
| 46 | Naive raw fusion | 0.9889 | 0.9306 | 0.7417 | 0.8361 | 0.8191 | 0.1528 |
| 46 | Feature redesigned simple fusion | 0.9889 | 0.8639 | 0.8028 | 0.8333 | 0.8181 | 0.1556 |
| 46 | Stable hybrid | 0.9889 | 0.9472 | 0.7889 | 0.8681 | 0.8598 | 0.1208 |

## Mean +- Std

| Model | Offline Acc | In Avg | Out Avg | Practical Avg | Practical F1 Avg | Drop |
|---|---:|---:|---:|---:|---:|---:|
| Glove vision-only | 0.9190 +- 0.0515 | 0.8067 +- 0.0197 | 0.6006 +- 0.0312 | 0.7036 +- 0.0213 | 0.6675 +- 0.0278 | 0.2154 +- 0.0521 |
| Naive raw fusion | 0.9484 +- 0.0340 | 0.8989 +- 0.0242 | 0.7344 +- 0.0527 | 0.8167 +- 0.0306 | 0.8017 +- 0.0358 | 0.1317 +- 0.0361 |
| Feature redesigned simple fusion | 0.9596 +- 0.0186 | 0.9144 +- 0.0335 | 0.7861 +- 0.0278 | 0.8503 +- 0.0211 | 0.8362 +- 0.0239 | 0.1093 +- 0.0355 |
| Stable hybrid | 0.9618 +- 0.0218 | 0.9094 +- 0.0449 | 0.7539 +- 0.0502 | 0.8317 +- 0.0449 | 0.8188 +- 0.0509 | 0.1301 +- 0.0315 |

## Files

- JSON: `C:\SignProject\experiments\fusion_design_ablation\2026-04-20_final_practical10_seed42_46\final_practical10_seed42_46_summary.json`
- TSV: `C:\SignProject\experiments\fusion_design_ablation\2026-04-20_final_practical10_seed42_46\final_practical10_seed42_46_summary.tsv`
- Per-dataset reports: `C:\SignProject\experiments\fusion_design_ablation\2026-04-20_final_practical10_seed42_46\practical_reports`
