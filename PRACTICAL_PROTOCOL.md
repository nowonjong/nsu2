# Practical Protocol

## Goal
- Use only two official comparison axes:
  - Offline
  - Practical
- Collect practical data only with `nsu_run_fusion_v6_overlap_hold.py`.
- Reuse the same captured sequences to evaluate:
  - glove vision-only
  - glove fusion
  - optional past candidate models

## Practical Types
- `Practical-In`: performed by a data collection participant.
- `Practical-In-2`: performed by an additional data collection participant under the same protocol.
- `Practical-Out`: performed by a non-participant.

## Capture Tool
- Script: `nsu_run_fusion_v6_overlap_hold.py`
- Reason:
  - stores both vision and sensor streams
  - the same real execution can be replayed for both vision-only and fusion models

## Official Word Set
1. `more`
2. `naeil`
3. `eoje`
4. `teukbyeol`
5. `byeollo`
6. `jamkkan`
7. `oraenman`
8. `gakkapda`
9. `jalhada`
10. `annyeonghaseyo`
11. `mannada`
12. `byeongyeonghada`
13. `banggeum`
14. `billida`
15. `gandanhada`
16. `sada`
17. `gamsahamnida`
18. `joesonghada`

## Repetitions
- 4 repetitions per word
- 18 words x 4 = 72 captures per person

## Recommended Order
1. `more`
2. `teukbyeol`
3. `jamkkan`
4. `jalhada`
5. `mannada`
6. `gandanhada`
7. `naeil`
8. `byeollo`
9. `oraenman`
10. `annyeonghaseyo`
11. `byeongyeonghada`
12. `sada`
13. `eoje`
14. `gakkapda`
15. `banggeum`
16. `gamsahamnida`
17. `billida`
18. `joesonghada`

Repeat the full 18-word order for 4 rounds.

## Run Controls
- `N`: next GT
- `B`: previous GT
- `G`: clear GT
- `S`: start capture
- `Q`: quit

## Performance Style
- Perform naturally, like actual usage.
- Do not over-correct into an overly clean collection style.
- Keep most trials, even if slightly awkward.
- Only note clearly wrong-GT or completely broken trials.

## Storage Paths
- `Practical-In` captures:
  - `C:\SignProject\experiments\live_eval\practical_in\captures`
- `Practical-In` reports:
  - `C:\SignProject\experiments\live_eval\practical_in\reports`
- `Practical-In-2` captures:
  - `C:\SignProject\experiments\live_eval\practical_in_2\captures`
- `Practical-In-2` reports:
  - `C:\SignProject\experiments\live_eval\practical_in_2\reports`
- `Practical-Out` captures:
  - `C:\SignProject\experiments\live_eval\practical_out\captures`
- `Practical-Out` reports:
  - `C:\SignProject\experiments\live_eval\practical_out\reports`

## Environment Variables
Before running, point the capture directory to the proper group.

### Practical-In
```powershell
$env:NSU_RUN_DEBUG_CAPTURE_DIR='C:\SignProject\experiments\live_eval\practical_in\captures'
```

### Practical-In-2
```powershell
$env:NSU_RUN_DEBUG_CAPTURE_DIR='C:\SignProject\experiments\live_eval\practical_in_2\captures'
```

### Practical-Out
```powershell
$env:NSU_RUN_DEBUG_CAPTURE_DIR='C:\SignProject\experiments\live_eval\practical_out\captures'
```

## Official Metrics
### Offline
- Accuracy
- Macro F1

### Practical
- Accuracy
- Macro F1

Optional support metrics:
- coverage
- confusion matrix

## Evaluation Rule
For practical scoring, always use the `nsu_run` captures as the common benchmark.
- Vision-only model: use the vision portion only.
- Fusion model: use the full captured input.
