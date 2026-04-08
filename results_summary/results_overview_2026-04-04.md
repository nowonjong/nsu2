# Results Overview (2026-04-04)

## Purpose
- This folder collects the main quantitative results produced so far for the `dataset_fusion_hold_v4_flags` experiments.
- Focus is on comparing `vision-only`, `input-only structured fusion`, `friend gate fusion`, and our final `stable hybrid` model.

## Dataset / Evaluation Context
- Dataset: `dataset_fusion_hold_v4_flags`
- Classes: 8
- Main split strategy: `group_session_stratified`
- Typical split used by the main experiments: `Train 248 / Val 16 / Test 56`
- Main metrics tracked: `Test Accuracy`, `Test Macro F1`

## Single-Run Reference Results

| Model | Path | Test Accuracy | Macro F1 | Notes |
| --- | --- | ---: | ---: | --- |
| Vision-only latest baseline | `vision_results_from_fusion_hold_v4_flags` | 0.9286 | 0.9280 | Latest pure vision baseline |
| Input-structured fusion v2 | `fusion_train_output_hold_v4_flags_imu_flags_flexposture_v2` | 0.9286 | 0.9260 | `imu_flags + flex posture`, no gate |
| Friend vision-only | `gksdydtjr/vision_only_v1` | 0.8214 | 0.8177 | From friend's rewritten train pipeline |
| Friend fusion_v2 | `gksdydtjr/fusion_v2` | 0.9107 | 0.9064 | Friend both-mode fusion run |
| Friend gate raw fusion | `gksdydtjr/fusion_train_output_hold_v4_flags_gate_v2` | 0.9643 | 0.9625 | Raw sensor + flags gate fusion |
| Stable hybrid (current main candidate) | `fusion_train_output_hold_v4_flags_gate_flexposture_stable` | 0.9464 | 0.9446 | Vision main + imu_flags gate + flex posture branch |

## Repeated-Seed Comparisons

### Stable Hybrid vs Friend Alpha 0.3 Gate (10 seeds: 42-51)

| Model | Mean Accuracy | Std Accuracy | Mean Macro F1 | Std Macro F1 | Min Acc | Max Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Friend raw gate + alpha 0.3 | 0.9022 | 0.0494 | 0.8925 | 0.0581 | 0.8000 | 0.9643 |
| Stable hybrid | 0.9225 | 0.0362 | 0.9170 | 0.0396 | 0.8500 | 0.9821 |

Interpretation:
- `friend alpha 0.3` had strong peak runs, but its average and stability were worse than `stable hybrid`.
- `stable hybrid` became the preferred final candidate because it showed both higher mean performance and lower spread over 10 seeds.

### Earlier 3-Seed Comparison (42, 43, 44)

| Model | Mean Accuracy | Std Accuracy | Mean Macro F1 | Std Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Friend raw gate (alpha 1.0) | 0.8958 | 0.0629 | 0.8799 | 0.0800 |
| Friend raw gate (alpha 0.5) | 0.9149 | 0.0242 | 0.9076 | 0.0273 |
| Friend raw gate (alpha 0.3) | 0.9191 | 0.0321 | 0.9147 | 0.0340 |
| Stable hybrid | 0.9024 | 0.0398 | 0.8949 | 0.0442 |

Note:
- The 3-seed result made `friend alpha 0.3` look strongest.
- After expanding to 10 seeds, `stable hybrid` overtook it on mean and stability.

## Input-Only Structured Version (3 runs)

Model idea:
- Uses our cleaned input structure (`imu_flags + flex posture`) but without gate-based vision correction.

Runs:
- `fusion_cmp_seed42_input_only`: `0.9643 / 0.9625`
- `fusion_cmp_seed43_input_only`: `0.7875 / 0.7661`
- `fusion_cmp_seed44_input_only`: `0.8393 / 0.8097`

Summary:
- Mean Accuracy: `0.8637`
- Std Accuracy: `0.0742`
- Mean Macro F1: `0.8461`
- Std Macro F1: `0.0842`

Important caution:
- `seed43` input-only run used a different split size (`Train 184 / Val 56 / Test 80`) than the usual `248 / 16 / 56`.
- So the input-only 3-run average is useful as a rough reference, but not as clean as the 10-seed stable-hybrid vs friend-alpha comparison.

## Current Practical Conclusion
- Best single run among the major compared models: `friend gate raw fusion` at `0.9643 / 0.9625`
- Best repeated-seed average and stability: `stable hybrid`
- Current recommended main model for thesis/demo: `fusion_train_output_hold_v4_flags_gate_flexposture_stable`

## Stable Hybrid Structure
- Vision branch is the main branch.
- `imu_flags` sensor branch produces a gate-based correction for vision.
- `flex raw` is not used as a full time-series branch.
- `flex posture v2` is used as a separate auxiliary feature branch.
- Final classifier uses original vision, adapted vision, sensor summary, and flex posture summary together.

## Suggested Use in Thesis
- Report both `single-run reference performance` and `multi-seed average/std`.
- Use `stable hybrid` as the main proposed model.
- Use `friend raw gate` and `friend alpha 0.3` as comparative baselines showing that high peak performance alone is not enough without stability.
