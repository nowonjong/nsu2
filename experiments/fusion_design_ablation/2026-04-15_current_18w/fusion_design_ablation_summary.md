# Fusion Design Ablation Summary (Current 18 Words)

- Date: 2026-04-15
- Training data: `dataset_fusion_hold_v4_flags`
- Seed: `42`
- Practical datasets: `practical_in`, `practical_in_2`, `practical_out`, `practical_out_2`
- Purpose: justify why the fusion design should use sensor-role redesign instead of simply dumping raw sensor values.

## Model Definitions

- `01_glove_vision_only`: Glove vision-only - Vision-only baseline using glove-collected visual landmarks only.
  - config: input_format=vision_only_lstm, model_variant=None, sensor_mode=None, sensor_raw_dim=None, use_flex_posture=None, flex_posture_dim=None
- `02_naive_raw_fusion`: Naive raw fusion - Vision + all 26 raw sensor channels, no flex posture summary.
  - config: input_format=dual_input_no_flex_lstm, model_variant=input_only, sensor_mode=all, sensor_raw_dim=26, use_flex_posture=False, flex_posture_dim=0
- `03_feature_redesigned_simple_fusion`: Feature redesigned simple fusion - Vision + IMU/flags sequence + flex posture summary, simple concat fusion.
  - config: input_format=dual_input_lstm, model_variant=input_only, sensor_mode=imu_flags, sensor_raw_dim=18, use_flex_posture=True, flex_posture_dim=44
- `04_stable_hybrid`: Stable hybrid - Vision-guided sensor gate + IMU/flags sequence + flex posture summary.
  - config: input_format=vision_guided_sensor_gate_flex_lstm, model_variant=stable_hybrid, sensor_mode=imu_flags, sensor_raw_dim=18, use_flex_posture=True, flex_posture_dim=44

## Main Table

| Model | Offline Acc | Offline F1 | Practical In Avg Acc | Practical Out Avg Acc | Practical Avg Acc | Practical Avg F1 | Offline->Practical Avg Drop | Offline->Out Drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Glove vision-only | 0.9333 | 0.9147 | 0.8819 | 0.5903 | 0.7361 | 0.6940 | 0.1972 | 0.3430 |
| Naive raw fusion | 0.9000 | 0.8796 | 0.8819 | 0.7153 | 0.7986 | 0.7783 | 0.1014 | 0.1847 |
| Feature redesigned simple fusion | 0.9778 | 0.9769 | 0.9653 | 0.8472 | 0.9062 | 0.8895 | 0.0716 | 0.1306 |
| Stable hybrid | 0.9333 | 0.9147 | 0.9167 | 0.8333 | 0.8750 | 0.8613 | 0.0583 | 0.1000 |

## Practical Detail

| Model | In Acc | In-2 Acc | Out Acc | Out-2 Acc | In F1 | In-2 F1 | Out F1 | Out-2 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Glove vision-only | 0.8194 | 0.9444 | 0.5972 | 0.5833 | 0.8073 | 0.9422 | 0.5395 | 0.4870 |
| Naive raw fusion | 0.8611 | 0.9028 | 0.6389 | 0.7917 | 0.8510 | 0.8838 | 0.6077 | 0.7707 |
| Feature redesigned simple fusion | 0.9444 | 0.9861 | 0.8750 | 0.8194 | 0.9374 | 0.9859 | 0.8383 | 0.7965 |
| Stable hybrid | 0.8889 | 0.9444 | 0.8333 | 0.8333 | 0.8699 | 0.9438 | 0.8241 | 0.8076 |

## Interpretation

- The result supports the claim that adding sensor values naively is not enough. Naive raw fusion improves over vision-only, but still drops strongly on Out-user Practical data.
- The largest gain in this ablation comes from feature redesign: IMU/flags remain sequence features while flex is converted into posture-summary features.
- In this seed/current-data run, feature redesigned simple fusion is stronger than the stable hybrid gate in Practical average. Therefore, stable hybrid should not be described as the final confirmed model solely from this experiment.
- Safe thesis wording: raw sensor dumping is not a reliable design principle; sensor-role redesign is empirically supported. Avoid overclaiming that flex raw is universally harmful.

## Files

- JSON summary: `C:\SignProject\experiments\fusion_design_ablation\2026-04-15_current_18w\fusion_design_ablation_summary.json`
- Practical reports: `C:\SignProject\experiments\fusion_design_ablation\2026-04-15_current_18w\practical_reports`
