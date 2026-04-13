# FPS / Latency Protocol

이 문서는 25FPS 고정 run 조건과 runtime profiling 결과 확인 방법을 정리한다.

## 목적

- 센서 fusion을 추가해도 실시간성이 크게 훼손되지 않는지 확인한다.
- FPS가 20~30 사이로 흔들리는 원인이 MediaPipe, model inference, sensor read, render 중 어디에 가까운지 분리해서 본다.
- 공식 Practical 평가와 발표 시연에서 동일한 target FPS 조건을 유지한다.

## 현재 기본값

`nsu_run_fusion_v6_overlap_hold.py`와 `gpt_run.py` 모두 기본 target FPS는 `25`이다.

- Fusion run env: `NSU_RUN_TARGET_FPS`
- Fusion camera env: `NSU_RUN_CAMERA_WIDTH`, `NSU_RUN_CAMERA_HEIGHT`
- Fusion stability env: `NSU_RUN_MIN_STABLE_FPS`
- Vision run env: `NSU_VISION_TARGET_FPS`
- Vision camera env: `NSU_VISION_CAMERA_WIDTH`, `NSU_VISION_CAMERA_HEIGHT`
- Vision stability env: `NSU_VISION_MIN_STABLE_FPS`
- 기본 profile 저장 폴더: `experiments\runtime_profile`

0 또는 음수로 지정하면 FPS cap을 끌 수 있다.
기본 안정 하한은 `20FPS`이다.

카메라는 MJPG fourcc, target FPS, buffer size 1을 요청한다. 이는 카메라 입력 지연과 불필요한 버퍼 누적을 줄이기 위한 안정화 옵션이다.

## Fusion 측정

```powershell
$env:NSU_RUN_TARGET_FPS='25'
$env:NSU_RUN_MIN_STABLE_FPS='20'
$env:NSU_RUN_PROFILE='1'
$env:NSU_RUN_PROFILE_DIR='C:\SignProject\experiments\runtime_profile'
& C:\SignProject\myenv\Scripts\python.exe C:\SignProject\nsu_run_fusion_v6_overlap_hold.py
```

실행 후 30초 정도 대기하거나 단어 몇 개를 `s` 키로 테스트한 뒤 `q`로 종료한다.
종료 시 `experiments\runtime_profile` 아래에 `*_nsu_run_fusion_v6_overlap_hold_runtime_profile.json` 파일이 저장된다.

## Vision-Only 측정

```powershell
$env:NSU_VISION_TARGET_FPS='25'
$env:NSU_VISION_MIN_STABLE_FPS='20'
$env:NSU_VISION_PROFILE='1'
$env:NSU_VISION_PROFILE_DIR='C:\SignProject\experiments\runtime_profile'
& C:\SignProject\myenv\Scripts\python.exe C:\SignProject\gpt_run.py
```

종료 시 `*_gpt_run_vision_runtime_profile.json` 파일이 저장된다.

## 결과 요약

```powershell
& C:\SignProject\myenv\Scripts\python.exe C:\SignProject\summarize_runtime_profiles.py --dir C:\SignProject\experiments\runtime_profile --latest 10
```

주요 컬럼:

- `achieved_fps`: 실제 평균 loop 기준 FPS
- `loop_mean_ms`: 전체 1프레임 loop 평균 시간
- `loop_p95_ms`: 느린 프레임 기준 지연 확인용
- `mediapipe_mean_ms`: MediaPipe 손 검출/랜드마크 처리 시간
- `inference_mean_ms`: LSTM 모델 추론 시간
- `sensor_mean_ms`: 센서 패킷 정렬/조회 시간
- `render_mean_ms`: OpenCV UI 그리기/표시 시간
- `sleep_mean_ms`: 25FPS cap을 맞추기 위해 쉬는 시간

## 해석 기준

- 25FPS 기준 1프레임 budget은 약 `40ms`이다.
- `loop_mean_ms`가 40ms 근처이고 `achieved_fps`가 24~25FPS면 정상이다.
- `sleep_mean_ms`가 충분히 남아 있으면 시스템 여유가 있다는 뜻이다.
- `sleep_mean_ms`가 거의 0이고 `loop_p95_ms`가 40ms를 자주 넘으면 병목을 찾아야 한다.
- 보통 병목 후보는 MediaPipe, rendering, 카메라 read, 모델 inference 순서로 확인한다.
- `status=STABLE`이면 평균 FPS와 p95 loop time이 20FPS 안정 기준을 만족한다.
- `status=WARNING`이면 평균 FPS는 20 이상이지만 순간 드랍이 있다.
- `status=UNSTABLE`이면 평균 FPS가 20 미만이다.

## 20FPS 아래로 떨어질 때의 저부하 설정

먼저 기본 `640x480 / 25FPS`로 측정한다. 만약 `status=WARNING` 또는 `UNSTABLE`이 나오면 해상도를 낮춰서 다시 측정한다.

Fusion 저부하 예시:

```powershell
$env:NSU_RUN_TARGET_FPS='25'
$env:NSU_RUN_MIN_STABLE_FPS='20'
$env:NSU_RUN_CAMERA_WIDTH='512'
$env:NSU_RUN_CAMERA_HEIGHT='384'
$env:NSU_RUN_PROFILE='1'
$env:NSU_RUN_PROFILE_DIR='C:\SignProject\experiments\runtime_profile'
& C:\SignProject\myenv\Scripts\python.exe C:\SignProject\nsu_run_fusion_v6_overlap_hold.py
```

Vision-only 저부하 예시:

```powershell
$env:NSU_VISION_TARGET_FPS='25'
$env:NSU_VISION_MIN_STABLE_FPS='20'
$env:NSU_VISION_CAMERA_WIDTH='512'
$env:NSU_VISION_CAMERA_HEIGHT='384'
$env:NSU_VISION_PROFILE='1'
$env:NSU_VISION_PROFILE_DIR='C:\SignProject\experiments\runtime_profile'
& C:\SignProject\myenv\Scripts\python.exe C:\SignProject\gpt_run.py
```

주의:

- RECORDING 중에는 프레임을 건너뛰지 않는다. 60프레임 시퀀스 자체가 모델 입력이기 때문이다.
- 처리빈도 조정은 공식 정확도 평가보다는 대기/시연 UI 최적화에서 사용하는 것이 안전하다.
- 공식 Practical 비교는 같은 해상도와 같은 target FPS에서 수행해야 한다.
- 해상도를 낮춘 경우, 그 조건을 결과표 또는 실험 설정에 명시한다.

## 보고서용 안전한 표현

본 시스템은 실사용 환경의 입력 안정성을 위해 카메라 처리 루프를 25FPS 목표로 제한하였다.
이는 순간적인 과부하와 프레임 드랍을 줄이고, 60프레임 입력 시퀀스의 시간 길이를 일정하게 유지하기 위한 설정이다.
모든 practical 비교는 동일한 target FPS 조건에서 수행한다.
