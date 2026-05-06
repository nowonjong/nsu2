# Final Run Model Bundle

This branch includes the two single-model checkpoints selected for the final runtime comparison.

## Feature Redesigned Fusion

- Model: `models/final_run/feature_redesigned_seed45/fusion_lstm_best.keras`
- Scaler: `models/final_run/feature_redesigned_seed45/fusion_scaler.npz`
- Labels: `models/final_run/feature_redesigned_seed45/labels.json`
- Config: `models/final_run/feature_redesigned_seed45/config.json`
- Selection criterion: best practical average among seeds 42-46
- Practical Avg: `0.8736`
- Practical F1 Avg: `0.8648`

Run:

```bat
run_fusion_with_arduino.bat --skip-upload
```

The fusion runtime can also be launched directly:

```bat
python nsu_run_fusion_v6_overlap_hold.py
```

## Vision-only

- Model: `models/final_run/vision_only_seed43/best_lstm.keras`
- Scaler: `models/final_run/vision_only_seed43/vision_scaler.npz`
- Labels: `models/final_run/vision_only_seed43/class_names.json`
- Config: `models/final_run/vision_only_seed43/config.json`
- Selection criterion: lowest practical average among seeds 42-46
- Practical Avg: `0.6750`
- Practical F1 Avg: `0.6317`

Run:

```bat
python gpt_run.py
```

## Environment Overrides

The default paths point to the bundled files above. To test another model without editing code, set:

- Fusion: `NSU_RUN_MODEL_DIR`
- Vision-only: `NSU_VISION_MODEL_DIR`

