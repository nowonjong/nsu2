# Final Practical 10-Shot Dataset Summary

Created merged practical evaluation datasets from the original 4-shot captures plus the later add6 captures.

Important policy:
- Source folders were not modified or deleted.
- Add6 samples are copied with the `add6__` prefix and recorded separately in each `manifest.json`.
- If we later decide that the newly recorded 6 samples should be excluded, use only `subset=original4` entries from the manifests or go back to the original `experiments/live_eval/practical_*` folders.
- The merged 10-shot folders are convenience evaluation datasets, not replacements for the original data.

## Dataset Paths

- `final_practical_in_10`: `C:\SignProject\experiments\live_eval_dataset\final_practical_in_10\captures`
- `final_practical_in_2_10`: `C:\SignProject\experiments\live_eval_dataset\final_practical_in_2_10\captures`
- `final_practical_out_10`: `C:\SignProject\experiments\live_eval_dataset\final_practical_out_10\captures`
- `final_practical_out_2_10`: `C:\SignProject\experiments\live_eval_dataset\final_practical_out_2_10\captures`

## Counts

| Dataset | JSON | NPY | Per-word total | Original4 per word | Add6 per word | Problems |
|---|---:|---:|---:|---:|---:|---:|
| `final_practical_in_10` | 180 | 180 | 10 OK | 4 OK | 6 OK | 1 |
| `final_practical_in_2_10` | 180 | 180 | 10 OK | 4 OK | 6 OK | 0 |
| `final_practical_out_10` | 180 | 180 | 10 OK | 4 OK | 6 OK | 0 |
| `final_practical_out_2_10` | 180 | 180 | 10 OK | 4 OK | 6 OK | 0 |

## Sensor/Shape Notes

### `final_practical_in_10`
- Sensor warnings/bad rows:
  - `original4__20260410_151628_734810_gamsahamnida.npy` gt=`gamsahamnida` nonzero_rows=60 flex_abs=0.0 imu_abs=17199.3359375

The warning above is from an original4 sample, not the newly recorded add6 set. It is kept for now so that evaluation history remains transparent.

## Next Step

Evaluate candidate models on these 10-shot practical datasets, while keeping the ability to rerun the same evaluation on original4-only data if add6 should be excluded later.
