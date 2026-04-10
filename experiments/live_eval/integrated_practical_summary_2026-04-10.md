# Integrated Practical Summary (2026-04-10)

## Main Message
- Offline에서는 vision-only가 더 높지만, Practical에서는 fusion baseline이 더 강하다.
- 특히 Practical-Out에서 fusion baseline이 가장 robust했다.
- Practical-In 계열에서는 일부 후보 모델이 더 높게 나왔지만, Out까지 포함하면 baseline fusion이 최종 구조 기준선으로 가장 안전하다.

## Baseline Comparison

| Split | Vision-only Acc | Vision-only F1 | Fusion Acc | Fusion F1 | Note |
|---|---:|---:|---:|---:|---|
| Offline | 0.9333 | 0.9239 | 0.9000 | 0.8796 | vision-only higher |
| Practical-In | 0.7361 | 0.7073 | 0.8750 | 0.8490 | fusion higher |
| Practical-In-2 | 0.9028 | 0.8810 | 0.9028 | 0.8990 | same acc, fusion higher F1 |
| Practical-Out | 0.5694 | 0.5055 | 0.8889 | 0.8725 | fusion much higher |

## Candidate Models on Practical Sets

| Split | Model | Acc | Macro F1 |
|---|---|---:|---:|
| Practical-In | sensor_correction | 0.9861 | 0.9859 |
| Practical-In | sensor_correction_ambiguous | 0.9444 | 0.9422 |
| Practical-In | modality_aware_correction | 0.9583 | 0.9563 |
| Practical-In | modality_aware_rerank | 0.9444 | 0.9443 |
| Practical-In-2 | sensor_correction | 0.9583 | 0.9577 |
| Practical-In-2 | sensor_correction_ambiguous | 0.9861 | 0.9859 |
| Practical-In-2 | modality_aware_correction | 0.9583 | 0.9589 |
| Practical-In-2 | modality_aware_rerank | 0.9167 | 0.9088 |
| Practical-Out | sensor_correction | 0.8056 | 0.7658 |
| Practical-Out | sensor_correction_ambiguous | 0.8472 | 0.8099 |
| Practical-Out | modality_aware_correction | 0.7222 | 0.6642 |
| Practical-Out | modality_aware_rerank | 0.8333 | 0.8033 |

## Interpretation
- 후보 모델은 수집 참여자 practical에서는 baseline보다 더 좋아 보일 수 있다.
- 하지만 Practical-Out에서는 baseline fusion이 가장 안정적이다.
- 따라서 최종 구조 선택 기준을 robustness까지 포함하면 baseline fusion 유지가 맞다.

## Related Files
- Practical-In group analysis: `C:\SignProject\experiments\live_eval\practical_group_analysis_2026-04-10.md`
- Candidate model practical summary: `C:\SignProject\experiments\live_eval\candidate_model_practical_summary_2026-04-10.json`
- This integrated summary JSON: `C:\SignProject\experiments\live_eval\integrated_practical_summary_2026-04-10.json`

