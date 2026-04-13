# Final Practical Group Analysis (2026-04-13)

## Scope
- Sets: Practical-In, Practical-In-2, Practical-Out, Practical-Out-2
- Models: glove vision-only baseline vs glove fusion baseline
- Metric: group accuracy by predefined similar-sign groups

## Overall Across All Practical Sets

| Group | Actions | Vision Acc | Fusion Acc | Fusion - Vision | Count |
|---|---|---:|---:|---:|---:|
| 방향 차이형 | more, naeil, eoje | 0.6875 | 1.0000 | +0.3125 | 48 |
| 순간 손모양 차이형 | banggeum, billida | 0.5938 | 0.9062 | +0.3125 | 32 |
| 추가동작 차이형 | jalhada, annyeonghaseyo | 0.7500 | 1.0000 | +0.2500 | 32 |
| 굽힘/경로 차이형 | jamkkan, oraenman, gakkapda | 0.4792 | 0.6667 | +0.1875 | 48 |
| 손목각도 차이형 | teukbyeol, byeollo | 0.8125 | 0.9062 | +0.0938 | 32 |
| 양손교차 차이형 | mannada, byeongyeonghada | 0.9062 | 0.9375 | +0.0312 | 32 |
| 손모양 차이형 | gandanhada, sada, gamsahamnida, joesonghada | 0.8594 | 0.8906 | +0.0312 | 64 |

## Practical-In

- Vision-only: Acc 0.7361, Macro F1 0.7073
- Fusion baseline: Acc 0.8750, Macro F1 0.8490

| Group | Vision Acc | Fusion Acc | Fusion - Vision | Count |
|---|---:|---:|---:|---:|
| 순간 손모양 차이형 | 0.2500 | 1.0000 | +0.7500 | 8 |
| 방향 차이형 | 0.5833 | 1.0000 | +0.4167 | 12 |
| 추가동작 차이형 | 0.7500 | 1.0000 | +0.2500 | 8 |
| 손목각도 차이형 | 0.8750 | 1.0000 | +0.1250 | 8 |
| 양손교차 차이형 | 1.0000 | 1.0000 | +0.0000 | 8 |
| 굽힘/경로 차이형 | 0.6667 | 0.5833 | -0.0833 | 12 |
| 손모양 차이형 | 0.9375 | 0.7500 | -0.1875 | 16 |

## Practical-In-2

- Vision-only: Acc 0.8889, Macro F1 0.8845
- Fusion baseline: Acc 0.9306, Macro F1 0.9270

| Group | Vision Acc | Fusion Acc | Fusion - Vision | Count |
|---|---:|---:|---:|---:|
| 방향 차이형 | 0.8333 | 1.0000 | +0.1667 | 12 |
| 손목각도 차이형 | 0.7500 | 0.8750 | +0.1250 | 8 |
| 굽힘/경로 차이형 | 0.8333 | 0.9167 | +0.0833 | 12 |
| 추가동작 차이형 | 1.0000 | 1.0000 | +0.0000 | 8 |
| 양손교차 차이형 | 1.0000 | 1.0000 | +0.0000 | 8 |
| 순간 손모양 차이형 | 1.0000 | 1.0000 | +0.0000 | 8 |
| 손모양 차이형 | 0.8750 | 0.8125 | -0.0625 | 16 |

## Practical-Out

- Vision-only: Acc 0.5694, Macro F1 0.5055
- Fusion baseline: Acc 0.8889, Macro F1 0.8725

| Group | Vision Acc | Fusion Acc | Fusion - Vision | Count |
|---|---:|---:|---:|---:|
| 추가동작 차이형 | 0.3750 | 1.0000 | +0.6250 | 8 |
| 순간 손모양 차이형 | 0.5000 | 1.0000 | +0.5000 | 8 |
| 굽힘/경로 차이형 | 0.0833 | 0.5000 | +0.4167 | 12 |
| 방향 차이형 | 0.6667 | 1.0000 | +0.3333 | 12 |
| 양손교차 차이형 | 0.7500 | 1.0000 | +0.2500 | 8 |
| 손목각도 차이형 | 0.6250 | 0.7500 | +0.1250 | 8 |
| 손모양 차이형 | 0.8750 | 1.0000 | +0.1250 | 16 |

## Practical-Out-2

- Vision-only: Acc 0.7083, Macro F1 0.6517
- Fusion baseline: Acc 0.8750, Macro F1 0.8621

| Group | Vision Acc | Fusion Acc | Fusion - Vision | Count |
|---|---:|---:|---:|---:|
| 방향 차이형 | 0.6667 | 1.0000 | +0.3333 | 12 |
| 굽힘/경로 차이형 | 0.3333 | 0.6667 | +0.3333 | 12 |
| 손모양 차이형 | 0.7500 | 1.0000 | +0.2500 | 16 |
| 추가동작 차이형 | 0.8750 | 1.0000 | +0.1250 | 8 |
| 손목각도 차이형 | 1.0000 | 1.0000 | +0.0000 | 8 |
| 순간 손모양 차이형 | 0.6250 | 0.6250 | +0.0000 | 8 |
| 양손교차 차이형 | 0.8750 | 0.7500 | -0.1250 | 8 |

## Interpretation
- ?? practical 4?? ?? ???? ?? ???? ???? fusion baseline? vision-only?? ??? ?? ????.
- ?? ? ??? ?? ???, ?? ??? ???, ???? ???, ??/?? ????? ????.
- ?? ?? ??? ?? ?? ???? ??? ??? ?? ?? ??? ??? ??? ?? ?direction_difference direction_difference.