# Sensor Contribution with Flags Summary

- Seeds: `42, 43, 44, 45, 46`
- Practical datasets: final 10-shot sets, each `18 words x 10 samples`
- Vision-only and Vision + IMU + Flex + flags rows reuse the existing final fusion-method multi-seed results.
- IMU-only and Flex-only rows were newly trained as simple input-only fusion models with frame/state flags included.

| Model | Offline Acc | In Avg | Out Avg | Practical Avg | Practical F1 Avg | Drop |
|---|---:|---:|---:|---:|---:|---:|
| Vision-only | 0.9190 ± 0.0515 | 0.8067 ± 0.0197 | 0.6006 ± 0.0312 | 0.7036 ± 0.0213 | 0.6675 ± 0.0278 | 0.2154 ± 0.0521 |
| Vision + IMU + flags | 0.9261 ± 0.0437 | 0.8483 ± 0.0250 | 0.6294 ± 0.0368 | 0.7389 ± 0.0248 | 0.7053 ± 0.0310 | 0.1872 ± 0.0363 |
| Vision + Flex + flags | 0.9573 ± 0.0267 | 0.8900 ± 0.0215 | 0.7539 ± 0.0604 | 0.8219 ± 0.0397 | 0.8091 ± 0.0412 | 0.1354 ± 0.0505 |
| Vision + IMU + Flex + flags | 0.9484 ± 0.0340 | 0.8989 ± 0.0242 | 0.7344 ± 0.0527 | 0.8167 ± 0.0306 | 0.8017 ± 0.0358 | 0.1317 ± 0.0361 |

## Files

- JSON: `C:\SignProject\experiments\sensor_contribution_flags\2026-04-28_seed42_46\sensor_contribution_flags_seed42_46_summary.json`
- New per-dataset reports: `C:\SignProject\experiments\sensor_contribution_flags\2026-04-28_seed42_46\practical_reports`
- Existing baseline source: `C:\SignProject\experiments\fusion_design_ablation\2026-04-20_final_practical10_seed42_46\final_practical10_seed42_46_summary.json`

