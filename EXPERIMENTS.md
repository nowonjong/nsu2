## Experiment Layout

All trained models and live-run artifacts should live under `experiments/`.

Recommended structure:

- `experiments/glove_fusion/`
- `experiments/glove_vision/`
- `experiments/bare_vision/`
- `experiments/live_eval/`

Each training run gets its own folder, for example:

- `experiments/glove_fusion/2026-04-07_stable_hybrid_18w_v1/`

Expected files inside one experiment folder:

- `fusion_lstm_best.keras`
- `fusion_scaler.npz`
- `labels.json`
- `config.json`
- `train_summary.txt`
- `confusion_matrix.npy`
- `confusion_matrix.png`
- `accuracy_curve.png`
- `loss_curve.png`
- `split_indices.npz`
- `sample_sources.json`

Recommended naming:

- date
- model family
- vocabulary size
- optional note/version

Examples:

- `2026-04-07_stable_hybrid_18w_v1`
- `2026-04-07_imu_flags_gate_18w_v1`
- `2026-04-07_vision_only_18w_v1`

Live run debug captures should be separated per model under:

- `experiments/live_eval/<experiment_name>/run_debug_captures/`

Use the helper scripts:

- `train_nsu_experiment.ps1`
- `run_nsu_experiment.ps1`
