# Realistic Offline Evaluation

- Model: `C:\SignProject\experiments\glove_fusion\2026-04-08_modality_aware_rerank_18w_v1`
- Input format: `modality_aware_rerank_flex_lstm`
- Test Accuracy: `0.9778`
- Test Macro F1: `0.9769`
- Hard-case Accuracy: `0.9259`
- Hard-case Macro F1: `0.9167`
- Low-margin rate < 0.10: `0.0000`
- Low-margin rate < 0.20: `0.0000`
- Worst-group Accuracy: `0.8667`

## Similar Groups
- 방향 차이형 (`direction_difference`): acc=1.0000, macro_f1=1.0000, intra_conf=0.0000, n=15
- 손목각도 차이형 (`wrist_angle_difference`): acc=1.0000, macro_f1=1.0000, intra_conf=0.0000, n=10
- 굽힘/경로 차이형 (`bend_path_difference`): acc=0.8667, macro_f1=0.8611, intra_conf=0.1333, n=15
- 추가동작 차이형 (`additional_motion_difference`): acc=1.0000, macro_f1=1.0000, intra_conf=0.0000, n=10
- 양손교차 차이형 (`cross_hand_difference`): acc=1.0000, macro_f1=1.0000, intra_conf=0.0000, n=10
- 순간 손모양 차이형 (`instant_handshape_difference`): acc=1.0000, macro_f1=1.0000, intra_conf=0.0000, n=10
- 손모양 차이형 (`handshape_difference`): acc=1.0000, macro_f1=1.0000, intra_conf=0.0000, n=20
