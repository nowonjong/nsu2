# Group Confusion Matrix Summary

- Unit: word-level predictions remapped to 7 predefined similar-sign groups
- Datasets: final practical 10-shot sets
- Seeds: 42-46 accumulated
- Note: diagonal values indicate predictions that stayed within the same predefined group, not exact word accuracy.

## Vision-only

### Row-normalized matrix

| Actual \ Predicted | Direction | Wrist angle | Bend/path | Additional motion | Cross-hand | Instant handshape | Handshape |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direction | 0.982 | 0.000 | 0.018 | 0.000 | 0.000 | 0.000 | 0.000 |
| Wrist angle | 0.000 | 0.973 | 0.000 | 0.010 | 0.007 | 0.000 | 0.010 |
| Bend/path | 0.000 | 0.005 | 0.838 | 0.000 | 0.153 | 0.003 | 0.000 |
| Additional motion | 0.010 | 0.220 | 0.068 | 0.667 | 0.018 | 0.000 | 0.018 |
| Cross-hand | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| Instant handshape | 0.075 | 0.000 | 0.007 | 0.020 | 0.000 | 0.897 | 0.000 |
| Handshape | 0.000 | 0.018 | 0.000 | 0.004 | 0.010 | 0.000 | 0.969 |

### Top cross-group confusions

| Actual group | Predicted group | Count | Row rate |
|---|---|---:|---:|
| Bend/path | Cross-hand | 92 | 0.153 |
| Additional motion | Wrist angle | 88 | 0.220 |
| Instant handshape | Direction | 30 | 0.075 |
| Additional motion | Bend/path | 27 | 0.068 |
| Handshape | Wrist angle | 14 | 0.018 |
| Direction | Bend/path | 11 | 0.018 |
| Instant handshape | Additional motion | 8 | 0.020 |
| Handshape | Cross-hand | 8 | 0.010 |

## Feature Redesigned Fusion

### Row-normalized matrix

| Actual \ Predicted | Direction | Wrist angle | Bend/path | Additional motion | Cross-hand | Instant handshape | Handshape |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direction | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Wrist angle | 0.000 | 0.920 | 0.000 | 0.060 | 0.003 | 0.000 | 0.018 |
| Bend/path | 0.002 | 0.002 | 0.970 | 0.002 | 0.025 | 0.000 | 0.000 |
| Additional motion | 0.000 | 0.212 | 0.005 | 0.762 | 0.015 | 0.000 | 0.005 |
| Cross-hand | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| Instant handshape | 0.003 | 0.003 | 0.000 | 0.000 | 0.000 | 0.995 | 0.000 |
| Handshape | 0.000 | 0.014 | 0.000 | 0.001 | 0.003 | 0.000 | 0.983 |

### Top cross-group confusions

| Actual group | Predicted group | Count | Row rate |
|---|---|---:|---:|
| Additional motion | Wrist angle | 85 | 0.212 |
| Wrist angle | Additional motion | 24 | 0.060 |
| Bend/path | Cross-hand | 15 | 0.025 |
| Handshape | Wrist angle | 11 | 0.014 |
| Wrist angle | Handshape | 7 | 0.018 |
| Additional motion | Cross-hand | 6 | 0.015 |
| Additional motion | Bend/path | 2 | 0.005 |
| Additional motion | Handshape | 2 | 0.005 |

## Files

- JSON: `C:\SignProject\experiments\group_confusion\2026-04-28_final10_seed42_46\group_confusion_matrix_summary.json`
- PNG: `C:\SignProject\Paper\figures\fig_3_group_confusion_matrix.png`
- SVG: `C:\SignProject\Paper\figures\fig_3_group_confusion_matrix.svg`
