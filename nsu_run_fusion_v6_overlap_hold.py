import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import json
import time
from datetime import datetime
import numpy as np
import mediapipe as mp
import tensorflow as tf
import serial
from collections import deque
import threading
from PIL import Image, ImageDraw, ImageFont
from runtime_profile_utils import FpsLimiter, RuntimeProfiler, env_flag, env_float, env_int

# =========================
# 경로 설정
# =========================
DEFAULT_MODEL_DIR = os.getenv(
    "NSU_RUN_MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "models", "final_run", "feature_redesigned_seed45"),
)
MODEL_PATH = os.getenv("NSU_RUN_MODEL_PATH", os.path.join(DEFAULT_MODEL_DIR, "fusion_lstm_best.keras"))
SCALER_PATH = os.getenv("NSU_RUN_SCALER_PATH", os.path.join(DEFAULT_MODEL_DIR, "fusion_scaler.npz"))
CLASS_NAMES_PATH = os.getenv("NSU_RUN_LABELS_PATH", os.path.join(DEFAULT_MODEL_DIR, "labels.json"))
CONFIG_PATH = os.getenv("NSU_RUN_CONFIG_PATH", os.path.join(DEFAULT_MODEL_DIR, "config.json"))
DEBUG_CAPTURE_DIR = os.getenv("NSU_RUN_DEBUG_CAPTURE_DIR", "run_debug_captures")

# =========================
# 기본 설정
# =========================
SEQ_LEN = 60
VISION_DIM = 126
RAW_SENSOR_DIM = 26
SENSOR_FLAG_DIM = 6
SENSOR_DIM = RAW_SENSOR_DIM + SENSOR_FLAG_DIM
FEATURE_DIM = VISION_DIM + SENSOR_DIM
FLEX_SENSOR_DIM = 8
DEFAULT_FLEX_BASELINE_FRAMES = 5
DEFAULT_FRAME_FLAG_NAMES = [
    'left_detected',
    'right_detected',
    'left_held',
    'right_held',
    'overlap_flag',
    'ambiguous_flag',
]
MODEL_SENSOR_RAW_DIM = RAW_SENSOR_DIM
MODEL_SENSOR_FLAG_DIM = SENSOR_FLAG_DIM
MODEL_SENSOR_DIM = SENSOR_DIM
MODEL_FEATURE_DIM = FEATURE_DIM
ACTIVE_RAW_SENSOR_INDICES = list(range(RAW_SENSOR_DIM))
ACTIVE_FLAG_INDICES = list(range(SENSOR_FLAG_DIM))
USE_FLEX_POSTURE = False
FLEX_POSTURE_DIM = 0
FLEX_BASELINE_FRAMES = DEFAULT_FLEX_BASELINE_FRAMES

CAM_WIDTH = env_int("NSU_RUN_CAMERA_WIDTH", 640)
CAM_HEIGHT = env_int("NSU_RUN_CAMERA_HEIGHT", 480)

DRAW_LANDMARKS = False
MODEL_COMPLEXITY = 0
MAX_NUM_HANDS = 2
TRIGGER_DELAY_SEC = 2.0

# =========================
# 시리얼 설정
# =========================
SERIAL_PORT = 'COM9'
SERIAL_BAUD = 115200

# =========================
# 시작 캘리브레이션 안내 시간
# 현재 Arduino 코드 기준
# 1) Keep still        : 약 4.5초
# 2) Fingers stretched : 약 4.0초
# 3) Fingers bent      : 약 4.0초
# =========================
STARTUP_GUIDE_STAGES = [
    ("장갑을 움직이지 말고 유지하세요", "IMU 안정화 중", 4.5, (40, 40, 40)),
    ("검지부터 새끼손가락까지 쭉 펴주세요", "Flex 펴짐 보정 중", 4.0, (60, 70, 40)),
    ("검지부터 새끼손가락까지 구부려주세요", "Flex 굽힘 보정 중", 4.0, (70, 40, 40)),
]

# =========================
# 인식 안정화 파라미터
# =========================
HANDEDNESS_SCORE_TH = 0.55
OVERLAP_IOU_THRESHOLD = 0.24
OVERLAP_CENTER_DIST_PX = 60.0
SLOT_AMBIGUOUS_MARGIN = 35.0
OVERLAP_APPROACH_IOU_THRESHOLD = 0.05
OVERLAP_APPROACH_CENTER_DIST_PX = 140.0
POST_OVERLAP_HOLD_FRAMES = 3
MISSING_HOLD_FRAMES = 6
SHORT_OVERLAP_FREEZE_FRAMES = 4
SHORT_MISSING_FREEZE_FRAMES = 3
STATIC_SUPPORT_MOTION_PX = 18.0
NO_HAND_RESET_FRAMES = 18
MIN_HAND_FRAMES_FOR_WORD = 20
SHORT_MOTION_WINDOW = 8
WORD_MOTION_THRESHOLD = 0.040
LOW_CONF_THRESHOLD = 0.55
LOW_MARGIN_THRESHOLD = 0.08
LOW_QUALITY_ZERO_BOTH_THRESHOLD = 2
LOW_QUALITY_OVERLAP_THRESHOLD = 3
LOW_QUALITY_MISSING_THRESHOLD = 6

# UI / 디버그
PRINT_DEBUG = False
SHOW_DEBUG_OVERLAY = False
TARGET_FPS = env_float("NSU_RUN_TARGET_FPS", 25.0)
MIN_STABLE_FPS = env_float("NSU_RUN_MIN_STABLE_FPS", 20.0)
PROFILE_ENABLED = env_flag("NSU_RUN_PROFILE", True)
PROFILE_DIR = os.getenv("NSU_RUN_PROFILE_DIR", r"experiments\runtime_profile")
DASHBOARD_WIDTH = env_int("NSU_RUN_DASHBOARD_WIDTH", 1180)
DASHBOARD_HEIGHT = env_int("NSU_RUN_DASHBOARD_HEIGHT", 700)
PLOT_HISTORY_LEN = env_int("NSU_RUN_PLOT_HISTORY", 120)

# =========================
# 표시명 사전
# =========================
KOR_MAP = {
    'none': '대기',
    'more': '모레',
    'naeil': '내일',
    'eoje': '어제',
    'teukbyeol': '특별',
    'byeollo': '별로',
    'jamkkan': '잠깐',
    'oraenman': '오랜만',
    'gakkapda': '가깝다',
    'jalhada': '잘하다',
    'annyeonghaseyo': '안녕하세요',
    'gandanhada': '간단하다',
    'sada': '사다',
    'gamsahamnida': '감사합니다',
    'joesonghada': '죄송하다',
    'banggeum': '방금',
    'billida': '빌리다',
    'mannada': '만나다',
    'byeongyeonghada': '변경하다',
}

UI_FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
_FONT_CACHE = {}

# =========================
# 센서 키 순서
# =========================
SENSOR_KEYS = [
    'lf0_norm', 'lf1_norm', 'lf2_norm', 'lf3_norm',
    'rf0_norm', 'rf1_norm', 'rf2_norm', 'rf3_norm',

    'l_ax_g', 'l_ay_g', 'l_az_g',
    'l_gx_dps', 'l_gy_dps', 'l_gz_dps',
    'l_roll_deg', 'l_pitch_deg', 'l_yaw_deg',

    'r_ax_g', 'r_ay_g', 'r_az_g',
    'r_gx_dps', 'r_gy_dps', 'r_gz_dps',
    'r_roll_deg', 'r_pitch_deg', 'r_yaw_deg',
]


# =========================
# TensorFlow GPU 설정
# =========================
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("Detected GPUs:", gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except Exception as e:
        print("GPU memory growth setting failed:", e)
else:
    print("GPU를 찾지 못해 CPU로 동작합니다.")

# =========================
# 모델 / 스케일러 / 클래스 로드
# =========================
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        runtime_config = json.load(f)
    SEQ_LEN = int(runtime_config.get('seq_length', SEQ_LEN))
    VISION_DIM = int(runtime_config.get('vision_dim', VISION_DIM))
else:
    runtime_config = {}

MODEL_SENSOR_RAW_DIM = int(runtime_config.get('sensor_raw_dim', runtime_config.get('full_raw_sensor_dim', RAW_SENSOR_DIM)))
MODEL_SENSOR_FLAG_DIM = int(runtime_config.get('sensor_flag_dim', runtime_config.get('full_sensor_flag_dim', SENSOR_FLAG_DIM)))
ACTIVE_RAW_SENSOR_INDICES = list(runtime_config.get('sensor_raw_indices', list(range(MODEL_SENSOR_RAW_DIM))))
ACTIVE_FLAG_INDICES = list(runtime_config.get('sensor_flag_indices', list(range(MODEL_SENSOR_FLAG_DIM))))
MODEL_SENSOR_DIM = int(
    runtime_config.get(
        'sensor_dim',
        len(ACTIVE_RAW_SENSOR_INDICES) + len(ACTIVE_FLAG_INDICES),
    )
)
MODEL_FEATURE_DIM = int(runtime_config.get('feature_dim', VISION_DIM + MODEL_SENSOR_DIM))
USE_FLEX_POSTURE = bool(runtime_config.get('use_flex_posture', False))
FLEX_POSTURE_DIM = int(runtime_config.get('flex_posture_dim', 0)) if USE_FLEX_POSTURE else 0
FLEX_BASELINE_FRAMES = int(runtime_config.get('flex_baseline_frames', DEFAULT_FLEX_BASELINE_FRAMES))

model = tf.keras.models.load_model(MODEL_PATH)
MODEL_INPUT_COUNT = len(model.inputs)

with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
    ACTIONS = json.load(f)

scaler_data = np.load(SCALER_PATH)
vision_scaler_mean = scaler_data['vision_mean'].astype(np.float32)
vision_scaler_scale = scaler_data['vision_scale'].astype(np.float32)
sensor_scaler_mean = scaler_data['sensor_mean'].astype(np.float32)
sensor_scaler_scale = scaler_data['sensor_scale'].astype(np.float32)
flex_posture_scaler_mean = scaler_data['flex_posture_mean'].astype(np.float32) if 'flex_posture_mean' in scaler_data.files else np.zeros((0,), dtype=np.float32)
flex_posture_scaler_scale = scaler_data['flex_posture_scale'].astype(np.float32) if 'flex_posture_scale' in scaler_data.files else np.ones((0,), dtype=np.float32)
vision_scaler_scale = np.where(
    np.abs(vision_scaler_scale) < 1e-8, 1.0, vision_scaler_scale
).astype(np.float32)
sensor_scaler_scale = np.where(
    np.abs(sensor_scaler_scale) < 1e-8, 1.0, sensor_scaler_scale
).astype(np.float32)
flex_posture_scaler_scale = np.where(
    np.abs(flex_posture_scaler_scale) < 1e-8, 1.0, flex_posture_scaler_scale
).astype(np.float32)

FRAME_FLAG_NAMES = runtime_config.get(
    'sensor_flag_names',
    DEFAULT_FRAME_FLAG_NAMES,
)

print(
    f"[ModelConfig] seq_len={SEQ_LEN}, vision_dim={VISION_DIM}, "
    f"stream_raw_dim={RAW_SENSOR_DIM}, stream_flag_dim={SENSOR_FLAG_DIM}, "
    f"model_sensor_raw_dim={MODEL_SENSOR_RAW_DIM}, model_sensor_flag_dim={MODEL_SENSOR_FLAG_DIM}, "
    f"model_sensor_dim={MODEL_SENSOR_DIM}, model_feature_dim={MODEL_FEATURE_DIM}, "
    f"use_flex_posture={USE_FLEX_POSTURE}, flex_posture_dim={FLEX_POSTURE_DIM}, "
    f"model_inputs={MODEL_INPUT_COUNT}"
)
print(f"[ModelConfig] active_raw_indices={ACTIVE_RAW_SENSOR_INDICES}")
print(f"[ModelConfig] active_flag_indices={ACTIVE_FLAG_INDICES}")
print("로드된 클래스:", ACTIONS)

if 'none' not in ACTIONS:
    print("[경고] labels.json에 'none' 클래스가 없습니다.")

# =========================
# 시리얼 리더
# =========================
class SerialReader(threading.Thread):
    def __init__(self, port, baud):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud

        self.ser = None
        self.stop_flag = False
        self.lock = threading.Lock()

        self.latest_sensor_time_ms = 0
        self.latest_sensor_vector = np.zeros(RAW_SENSOR_DIM, dtype=np.float32)
        self.latest_host_time_ms = 0.0
        self.packet_count = 0
        self.packet_buffer = deque(maxlen=256)

        self.connected = False
        self.connection_error = None

    def connect(self):
        self.ser = serial.Serial(self.port, self.baud, timeout=0.05)
        time.sleep(2.0)
        self.connected = True
        print(f"[Serial] Connected: {self.port} @ {self.baud}")

    def close(self):
        self.stop_flag = True
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
        except:
            pass

    def parse_csv_line(self, line):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 27:
            return None

        try:
            values = [float(x) for x in parts]
        except ValueError:
            return None

        sensor_time_ms = int(values[0])
        sensor_vec = np.array(values[1:], dtype=np.float32)

        if sensor_vec.shape[0] != RAW_SENSOR_DIM:
            return None

        return sensor_time_ms, sensor_vec

    def run(self):
        try:
            self.connect()
        except Exception as e:
            self.connection_error = e
            print(f"[Serial] Connection failed: {e}")
            return

        while not self.stop_flag:
            try:
                raw = self.ser.readline()
                if not raw:
                    continue

                line = raw.decode('utf-8', errors='ignore').strip()
                if not line:
                    continue

                parsed = self.parse_csv_line(line)
                if parsed is None:
                    continue

                sensor_time_ms, sensor_vec = parsed
                host_time_ms = time.perf_counter() * 1000.0

                with self.lock:
                    self.latest_sensor_time_ms = sensor_time_ms
                    self.latest_sensor_vector = sensor_vec
                    self.latest_host_time_ms = host_time_ms
                    self.packet_count += 1
                    self.packet_buffer.append(
                        (host_time_ms, sensor_time_ms, sensor_vec.copy(), self.packet_count)
                    )

            except Exception as e:
                print(f"[Serial] Read error: {e}")
                time.sleep(0.1)

    def get_latest(self):
        with self.lock:
            return (
                self.latest_sensor_time_ms,
                self.latest_sensor_vector.copy(),
                self.packet_count,
                self.latest_host_time_ms,
            )

    def get_aligned_packet(self, frame_host_time_ms):
        with self.lock:
            if not self.packet_buffer:
                return (
                    self.latest_sensor_time_ms,
                    self.latest_sensor_vector.copy(),
                    self.packet_count,
                    self.latest_host_time_ms,
                )

            eligible = [item for item in self.packet_buffer if item[0] <= frame_host_time_ms]
            if eligible:
                host_time_ms, sensor_time_ms, sensor_vec, packet_count = eligible[-1]
                return sensor_time_ms, sensor_vec.copy(), packet_count, host_time_ms

            host_time_ms, sensor_time_ms, sensor_vec, packet_count = self.packet_buffer[0]
            return sensor_time_ms, sensor_vec.copy(), packet_count, host_time_ms


def wait_for_serial_connected(serial_reader, timeout_sec=5):
    start = time.time()
    while time.time() - start < timeout_sec:
        if serial_reader.connected:
            return True
        if serial_reader.connection_error is not None:
            return False
        time.sleep(0.05)
    return False


def wait_for_sensor_ready(serial_reader, timeout_sec=12):
    start = time.time()
    while time.time() - start < timeout_sec:
        if serial_reader.connection_error is not None:
            return False

        _, _, packet_count, _ = serial_reader.get_latest()
        if packet_count > 0:
            print("[Sensor] Ready")
            return True

        time.sleep(0.05)

    print("[Sensor] Timeout: 센서 패킷이 들어오지 않습니다.")
    return False


# =========================
# MediaPipe Hands
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# 카메라 열기
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# =========================
# 유틸 함수
# =========================
def get_hand_data(res_hand):
    wrist = res_hand.landmark[0]
    joint = np.zeros((21, 3), dtype=np.float32)

    for j, lm in enumerate(res_hand.landmark):
        if j == 0:
            joint[j] = [0.0, 0.0, 0.0]
        else:
            joint[j] = [lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]

    scale = np.linalg.norm(joint[9])
    if scale < 1e-6:
        scale = 1e-6

    joint = joint / scale
    return joint.flatten()

def get_hand_center_px(res_hand, frame_w, frame_h):
    xs = [lm.x for lm in res_hand.landmark]
    ys = [lm.y for lm in res_hand.landmark]
    cx = float(np.mean(xs) * frame_w)
    cy = float(np.mean(ys) * frame_h)
    return np.array([cx, cy], dtype=np.float32)


def get_hand_bbox_px(res_hand, frame_w, frame_h, pad_px=12.0):
    xs = [lm.x * frame_w for lm in res_hand.landmark]
    ys = [lm.y * frame_h for lm in res_hand.landmark]
    x1 = max(0.0, min(xs) - pad_px)
    y1 = max(0.0, min(ys) - pad_px)
    x2 = min(float(frame_w - 1), max(xs) + pad_px)
    y2 = min(float(frame_h - 1), max(ys) + pad_px)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def bbox_iou(box1, box2):
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area1 = max(1.0, float(box1[2] - box1[0]) * float(box1[3] - box1[1]))
    area2 = max(1.0, float(box2[2] - box2[0]) * float(box2[3] - box2[1]))
    union = area1 + area2 - inter
    if union <= 1e-6:
        return 0.0
    return float(inter / union)


def hands_are_overlapping(candidates):
    if len(candidates) < 2:
        return False

    c0, c1 = candidates[:2]
    iou = bbox_iou(c0['bbox'], c1['bbox'])
    center_dist = float(np.linalg.norm(c0['center'] - c1['center']))
    return (iou >= OVERLAP_IOU_THRESHOLD) or (center_dist <= OVERLAP_CENTER_DIST_PX)


def hands_are_approaching_overlap(candidates):
    if len(candidates) < 2:
        return False

    c0, c1 = candidates[:2]
    iou = bbox_iou(c0['bbox'], c1['bbox'])
    center_dist = float(np.linalg.norm(c0['center'] - c1['center']))
    return (iou >= OVERLAP_APPROACH_IOU_THRESHOLD) or (center_dist <= OVERLAP_APPROACH_CENTER_DIST_PX)


def mp_label_to_person_side(label, frame_flipped=True):
    if label not in ('Left', 'Right'):
        return None

    # ?꾩옱 肄붾뱶??frame??癒쇱? 醫뚯슦諛섏쟾????MediaPipe???ｊ퀬 ?덉쑝誘濡?    # selfie ?낅젰 湲곗? 洹몃?濡??щ엺 湲곗? 醫??곕줈 ?ъ슜?⑸땲??
    if frame_flipped:
        return label.lower()

    # 諛섏쟾?섏? ?딆? ?먮낯 ?꾨젅?꾩쓣 ?ｋ뒗 寃쎌슦?먮뒗 ?꾩슂 ???ш린??swap ?섏꽭??
    return 'right' if label == 'Left' else 'left'


def make_hand_candidates(results, frame_w, frame_h):
    candidates = []

    if not (results.multi_hand_landmarks and results.multi_handedness):
        return candidates

    for res_hand, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = handedness.classification[0].label
        score = handedness.classification[0].score
        side_hint = None
        if score >= HANDEDNESS_SCORE_TH:
            side_hint = mp_label_to_person_side(label, frame_flipped=True)

        candidates.append({
            'data': get_hand_data(res_hand),
            'center': get_hand_center_px(res_hand, frame_w, frame_h),
            'bbox': get_hand_bbox_px(res_hand, frame_w, frame_h),
            'mp_label': label,
            'mp_score': float(score),
            'person_side_hint': side_hint,
            'res_hand': res_hand,
        })

    return candidates


def slot_cost(candidate, slot_name, prev_center):
    cost = 0.0

    hint = candidate['person_side_hint']
    if hint is not None and hint != slot_name:
        cost += 120.0

    if prev_center is not None:
        cost += float(np.linalg.norm(candidate['center'] - prev_center))
    else:
        cost += 20.0

    return cost


def choose_slot_for_one_hand(candidate, prev_left_center, prev_right_center):
    hint = candidate['person_side_hint']

    if prev_left_center is None and prev_right_center is None and hint in ('left', 'right'):
        return hint

    left_cost = slot_cost(candidate, 'left', prev_left_center)
    right_cost = slot_cost(candidate, 'right', prev_right_center)
    if abs(left_cost - right_cost) < SLOT_AMBIGUOUS_MARGIN:
        return None
    return 'left' if left_cost <= right_cost else 'right'


def assign_hand_slots(candidates, prev_left_center, prev_right_center):
    assigned = {'left': None, 'right': None}
    frame_ambiguous = False

    if len(candidates) == 0:
        return assigned, frame_ambiguous

    if len(candidates) == 1:
        slot = choose_slot_for_one_hand(candidates[0], prev_left_center, prev_right_center)
        if slot is None:
            return assigned, True
        assigned[slot] = candidates[0]
        return assigned, frame_ambiguous

    # 理쒕? 2???ъ슜 議곌굔?대?濡??먯닔 ?믪? ??媛쒕쭔 ?ъ슜
    cands = candidates[:2]

    case1 = slot_cost(cands[0], 'left', prev_left_center) + slot_cost(cands[1], 'right', prev_right_center)
    case2 = slot_cost(cands[1], 'left', prev_left_center) + slot_cost(cands[0], 'right', prev_right_center)

    if abs(case1 - case2) < SLOT_AMBIGUOUS_MARGIN:
        return assigned, True

    if case1 <= case2:
        assigned['left'] = cands[0]
        assigned['right'] = cands[1]
    else:
        assigned['left'] = cands[1]
        assigned['right'] = cands[0]

    return assigned, frame_ambiguous


def draw_custom_landmarks(img, hand_landmarks, color, label):
    global DRAW_LANDMARKS
    if not DRAW_LANDMARKS:
        return

    h, w, _ = img.shape
    wrist = hand_landmarks.landmark[0]

    for lm in hand_landmarks.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 4, color, cv2.FILLED)

    cv2.putText(
        img,
        label,
        (int(wrist.x * w) - 15, int(wrist.y * h) - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )


def get_ui_font(size):
    key = int(size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(UI_FONT_PATH, key)
    return _FONT_CACHE[key]


def draw_text_unicode(img, text, org, font_size=28, text_color=(0, 0, 0)):
    if not text:
        return

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(org, text, font=get_ui_font(font_size), fill=(text_color[2], text_color[1], text_color[0]))
    img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_text_box(img, top_left, bottom_right, fill_color=(255, 255, 255), alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, top_left, bottom_right, fill_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_panel_card(img, x1, y1, x2, y2, title="", accent=(90, 130, 210)):
    draw_text_box(img, (x1, y1), (x2, y2), fill_color=(248, 250, 252), alpha=0.92)
    cv2.rectangle(img, (x1, y1), (x2, y2), (214, 220, 228), 1)
    cv2.rectangle(img, (x1, y1), (x2, y1 + 6), accent, -1)
    if title:
        draw_text_unicode(img, title, (x1 + 14, y1 + 14), font_size=24, text_color=(28, 32, 38))


def draw_progress_bar(img, x1, y1, width, height, value, fg_color, bg_color=(210, 214, 220), label=""):
    value = max(0.0, min(1.0, float(value)))
    cv2.rectangle(img, (x1, y1), (x1 + width, y1 + height), bg_color, -1)
    cv2.rectangle(img, (x1, y1), (x1 + int(width * value), y1 + height), fg_color, -1)
    cv2.rectangle(img, (x1, y1), (x1 + width, y1 + height), (160, 165, 170), 1)
    if label:
        draw_text_unicode(img, label, (x1, y1 - 24), font_size=20, text_color=(45, 45, 45))


def draw_led(img, center, radius, color, active=True):
    c = tuple(int(v) for v in color)
    if active:
        cv2.circle(img, center, radius + 5, tuple(min(255, int(v * 0.35 + 120)) for v in c), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, center, radius, c, -1, lineType=cv2.LINE_AA)
    else:
        cv2.circle(img, center, radius, (145, 145, 145), -1, lineType=cv2.LINE_AA)


def draw_line_chart(img, x1, y1, x2, y2, series_map, y_min=None, y_max=None):
    cv2.rectangle(img, (x1, y1), (x2, y2), (232, 236, 240), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (205, 210, 216), 1)

    width = max(1, x2 - x1 - 12)
    height = max(1, y2 - y1 - 12)
    chart_x1 = x1 + 6
    chart_y1 = y1 + 6
    chart_x2 = chart_x1 + width
    chart_y2 = chart_y1 + height

    all_vals = []
    for values, _color in series_map.values():
        all_vals.extend(list(values))
    if not all_vals:
        return

    if y_min is None:
        y_min = float(min(all_vals))
    if y_max is None:
        y_max = float(max(all_vals))
    if abs(y_max - y_min) < 1e-6:
        y_max = y_min + 1.0

    for ratio in [0.25, 0.5, 0.75]:
        yy = int(chart_y2 - ratio * height)
        cv2.line(img, (chart_x1, yy), (chart_x2, yy), (220, 224, 228), 1)

    for values, color in series_map.values():
        vals = list(values)
        if len(vals) < 2:
            continue
        pts = []
        for i, val in enumerate(vals):
            px = chart_x1 + int((i / max(1, len(vals) - 1)) * width)
            py = chart_y2 - int(((float(val) - y_min) / (y_max - y_min)) * height)
            pts.append((px, py))
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, 2, lineType=cv2.LINE_AA)

    draw_text_unicode(img, f"{y_max:.2f}", (x2 - 70, y1 + 2), font_size=16, text_color=(90, 90, 90))
    draw_text_unicode(img, f"{y_min:.2f}", (x2 - 70, y2 - 24), font_size=16, text_color=(90, 90, 90))


def fit_frame_to_box(frame, box_w, box_h):
    h, w = frame.shape[:2]
    scale = min(box_w / max(1, w), box_h / max(1, h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h))


def standardize_modalities(seq):
    seq = np.asarray(seq, dtype=np.float32)
    seq_vision = seq[:, :VISION_DIM].astype(np.float32)
    seq_sensor_full = seq[:, VISION_DIM:].astype(np.float32)

    seq_sensor_raw_full = seq_sensor_full[:, :RAW_SENSOR_DIM]
    seq_sensor_flags_full = seq_sensor_full[:, RAW_SENSOR_DIM:RAW_SENSOR_DIM + SENSOR_FLAG_DIM]

    selected_sensor_parts = []
    if ACTIVE_RAW_SENSOR_INDICES:
        selected_sensor_parts.append(seq_sensor_raw_full[:, ACTIVE_RAW_SENSOR_INDICES])
    if ACTIVE_FLAG_INDICES:
        selected_sensor_parts.append(seq_sensor_flags_full[:, ACTIVE_FLAG_INDICES])

    if selected_sensor_parts:
        seq_sensor = np.concatenate(selected_sensor_parts, axis=1).astype(np.float32)
    else:
        seq_sensor = np.zeros((seq.shape[0], 0), dtype=np.float32)

    seq_vision_scaled = np.zeros_like(seq_vision, dtype=np.float32)
    nonzero_mask = np.any(np.abs(seq_vision) > 1e-6, axis=1)
    if np.any(nonzero_mask):
        seq_vision_scaled[nonzero_mask] = (
            (seq_vision[nonzero_mask] - vision_scaler_mean) / vision_scaler_scale
        ).astype(np.float32)

    if seq_sensor.shape[1] > 0:
        seq_sensor_scaled = ((seq_sensor - sensor_scaler_mean) / sensor_scaler_scale).astype(np.float32)
    else:
        seq_sensor_scaled = seq_sensor

    return seq_vision_scaled, seq_sensor_scaled, seq_sensor_raw_full


def extract_flex_posture_features_from_sequence(seq_sensor_raw_full):
    if not USE_FLEX_POSTURE or FLEX_POSTURE_DIM <= 0:
        return np.zeros((0,), dtype=np.float32)

    if seq_sensor_raw_full.shape[1] < FLEX_SENSOR_DIM:
        raise ValueError(
            f"Expected at least {FLEX_SENSOR_DIM} flex channels, got {seq_sensor_raw_full.shape[1]}"
        )

    flex_seq = seq_sensor_raw_full[:, :FLEX_SENSOR_DIM].astype(np.float32)
    baseline_frames = max(1, min(FLEX_BASELINE_FRAMES, flex_seq.shape[0]))
    base = np.mean(flex_seq[:baseline_frames, :], axis=0, keepdims=True)
    rel = flex_seq - base

    mean_rel = np.mean(rel, axis=0)
    std_rel = np.std(rel, axis=0)
    range_rel = np.max(rel, axis=0) - np.min(rel, axis=0)
    end_rel = np.mean(rel[-baseline_frames:, :], axis=0)
    peak_abs_rel = np.max(np.abs(rel), axis=0)
    lr_pair_diff = end_rel[: (FLEX_SENSOR_DIM // 2)] - end_rel[(FLEX_SENSOR_DIM // 2):]

    features = np.concatenate(
        [mean_rel, std_rel, range_rel, end_rel, peak_abs_rel, lr_pair_diff],
        axis=0,
    ).astype(np.float32)

    if features.shape[0] != FLEX_POSTURE_DIM:
        raise ValueError(
            f"Expected flex posture dim {FLEX_POSTURE_DIM}, got {features.shape[0]}"
        )

    return ((features - flex_posture_scaler_mean) / flex_posture_scaler_scale).astype(np.float32)

def get_motion_score(seq_vision):
    if len(seq_vision) < 2:
        return 0.0
    diffs = np.diff(seq_vision, axis=0)
    motion = np.mean(np.linalg.norm(diffs, axis=1))
    return float(motion)

def get_flex_stats(sensor_vec):
    left_mean = float(np.mean(sensor_vec[0:4]))
    right_mean = float(np.mean(sensor_vec[4:8]))
    both_mean = (left_mean + right_mean) / 2.0
    return left_mean, right_mean, both_mean


def build_sensor_model_vec(sensor_vec, left_seen, right_seen, left_held, right_held, overlap_now, frame_ambiguous):
    base_sensor = np.asarray(sensor_vec, dtype=np.float32)
    if base_sensor.shape[0] != RAW_SENSOR_DIM:
        raise ValueError(f"Expected raw sensor dim {RAW_SENSOR_DIM}, got {base_sensor.shape[0]}")

    if SENSOR_FLAG_DIM <= 0:
        return base_sensor

    full_flag_vec = np.array(
        [
            1.0 if left_seen else 0.0,
            1.0 if right_seen else 0.0,
            1.0 if left_held else 0.0,
            1.0 if right_held else 0.0,
            1.0 if overlap_now else 0.0,
            1.0 if frame_ambiguous else 0.0,
        ],
        dtype=np.float32,
    )
    flag_vec = full_flag_vec[:SENSOR_FLAG_DIM]
    return np.concatenate([base_sensor, flag_vec]).astype(np.float32)


def classify_sequence_once(seq_array):
    classify_start = time.perf_counter()
    seq_vision = seq_array[:, :VISION_DIM]
    long_motion = get_motion_score(seq_vision)
    hand_count = int(np.sum(np.any(np.abs(seq_vision) > 1e-6, axis=1)))

    result = {
        'label': 'none',
        'conf': 1.0,
        'raw_label': 'none',
        'raw_conf': 1.0,
        'margin': 0.0,
        'long_motion': long_motion,
        'hand_count': hand_count,
    }

    if hand_count < MIN_HAND_FRAMES_FOR_WORD or long_motion < WORD_MOTION_THRESHOLD:
        runtime_profiler.add_duration("classify_skip_gate", classify_start)
        return result

    prep_start = time.perf_counter()
    seq_vision_scaled, seq_sensor_scaled, seq_sensor_raw_full = standardize_modalities(seq_array)
    flex_posture_scaled = extract_flex_posture_features_from_sequence(seq_sensor_raw_full)
    vision_input = tf.convert_to_tensor(
        np.expand_dims(seq_vision_scaled, axis=0), dtype=tf.float32
    )
    sensor_input = tf.convert_to_tensor(
        np.expand_dims(seq_sensor_scaled, axis=0), dtype=tf.float32
    )
    if MODEL_INPUT_COUNT >= 3:
        flex_input = tf.convert_to_tensor(
            np.expand_dims(flex_posture_scaled, axis=0), dtype=tf.float32
        )
        runtime_profiler.add_duration("classify_preprocess", prep_start)
        infer_start = time.perf_counter()
        y_prob = infer(vision_input, sensor_input, flex_input).numpy()[0]
        runtime_profiler.add_duration("model_inference", infer_start)
    else:
        runtime_profiler.add_duration("classify_preprocess", prep_start)
        infer_start = time.perf_counter()
        y_prob = infer(vision_input, sensor_input).numpy()[0]
        runtime_profiler.add_duration("model_inference", infer_start)

    sorted_idx = np.argsort(y_prob)
    top1_idx = int(sorted_idx[-1])
    top2_idx = int(sorted_idx[-2])

    top1_label = ACTIONS[top1_idx]
    top1_conf = float(y_prob[top1_idx])
    top2_conf = float(y_prob[top2_idx])
    margin = top1_conf - top2_conf

    result['raw_label'] = top1_label
    result['raw_conf'] = top1_conf
    result['margin'] = margin
    result['label'] = top1_label
    result['conf'] = top1_conf
    top_k = min(3, len(ACTIONS))
    result['top_candidates'] = [
        {
            'label': ACTIONS[int(idx)],
            'conf': float(y_prob[int(idx)]),
        }
        for idx in sorted_idx[-top_k:][::-1]
    ]
    runtime_profiler.add_duration("classify_total", classify_start)

    return result


def should_hold_as_none(result, zero_both_count, zero_reason_counts):
    raw_conf = float(result.get('raw_conf', 0.0))
    margin = float(result.get('margin', 0.0))
    low_prediction_quality = (raw_conf < LOW_CONF_THRESHOLD) and (margin < LOW_MARGIN_THRESHOLD)
    poor_input_quality = (
        int(zero_both_count) >= LOW_QUALITY_ZERO_BOTH_THRESHOLD
        or int(zero_reason_counts.get('overlap', 0)) >= LOW_QUALITY_OVERLAP_THRESHOLD
        or (
            int(zero_reason_counts.get('missing_left', 0))
            + int(zero_reason_counts.get('missing_right', 0))
        ) >= LOW_QUALITY_MISSING_THRESHOLD
    )
    return low_prediction_quality and poor_input_quality



def draw_status_panel(img, state_text, sub_text, result_text, fps_text, progress_ratio=0.0, color=(40, 40, 40)):
    x1, y1, x2, y2 = 10, 10, 430, 120
    draw_text_box(img, (x1, y1), (x2, y2), fill_color=(255, 255, 255), alpha=0.58)
    draw_text_unicode(img, state_text, (22, 18), font_size=28, text_color=(0, 0, 0))
    draw_text_unicode(img, sub_text, (22, 50), font_size=22, text_color=(25, 25, 25))
    draw_text_unicode(img, result_text, (22, 78), font_size=24, text_color=(0, 120, 0))
    draw_text_unicode(img, fps_text, (300, 82), font_size=20, text_color=(30, 30, 30))

    if progress_ratio > 0:
        bar_w = x2 - x1 - 24
        fill = int(bar_w * max(0.0, min(1.0, progress_ratio)))
        cv2.rectangle(img, (22, 98), (22 + bar_w, 104), (70, 70, 70), -1)
        cv2.rectangle(img, (22, 98), (22 + fill, 104), (0, 255, 0), -1)


def draw_debug_panel(img, lines):
    if not SHOW_DEBUG_OVERLAY:
        return

    x1, y1 = 10, 120
    width = 360
    line_h = 22
    height = 16 + line_h * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + width, y1 + height), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.60, img, 0.40, 0, img)

    for i, line in enumerate(lines):
        cv2.putText(img, line, (x1 + 12, y1 + 24 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1)


def render_dashboard(
    camera_img,
    fps_smooth,
    mode,
    sequence_len,
    current_pred_label,
    current_pred_conf,
    stable_label,
    stable_conf,
    gt_kor,
    long_motion_score,
    short_motion_score,
    last_margin,
    sensor_vec,
    sensor_packet_count,
    last_sensor_time_ms,
    last_sensor_host_delta_ms,
    left_seen,
    right_seen,
    overlap_now,
    frame_ambiguous,
    top_candidates,
    flex_left_hist,
    flex_right_hist,
    imu_left_hist,
    imu_right_hist,
):
    dashboard = np.full((DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3), (240, 243, 247), dtype=np.uint8)
    pad = 18
    sidebar_w = min(420, max(380, int(DASHBOARD_WIDTH * 0.35)))
    camera_w = DASHBOARD_WIDTH - pad * 3 - sidebar_w
    camera_h = DASHBOARD_HEIGHT - pad * 2
    cam_x1, cam_y1 = pad, pad
    cam_x2, cam_y2 = cam_x1 + camera_w, cam_y1 + camera_h
    side_x1, side_y1 = cam_x2 + pad, pad
    side_x2, side_y2 = DASHBOARD_WIDTH - pad, DASHBOARD_HEIGHT - pad

    cv2.rectangle(dashboard, (cam_x1, cam_y1), (cam_x2, cam_y2), (16, 20, 26), -1)
    fitted = fit_frame_to_box(camera_img, camera_w, camera_h)
    fh, fw = fitted.shape[:2]
    off_x = cam_x1 + (camera_w - fw) // 2
    off_y = cam_y1 + (camera_h - fh) // 2
    dashboard[off_y:off_y + fh, off_x:off_x + fw] = fitted
    cv2.rectangle(dashboard, (off_x, off_y), (off_x + fw, off_y + fh), (255, 255, 255), 1)

    # Minimal Korean overlay on camera
    stable_kor = KOR_MAP.get(stable_label, stable_label)
    is_correct = (stable_kor == gt_kor) and stable_label != 'none' and gt_kor != "미지정"
    result_color = (0, 150, 50) if is_correct else (200, 50, 50) if stable_label != 'none' else (40, 40, 40)
    draw_text_box(dashboard, (off_x + 12, off_y + 12), (off_x + 360, off_y + 84), fill_color=(255, 255, 255), alpha=0.82)
    draw_text_unicode(dashboard, f"목표 단어 (GT): {gt_kor}", (off_x + 20, off_y + 18), font_size=19, text_color=(90, 90, 90))
    draw_text_unicode(dashboard, f"현재 번역: {stable_kor}", (off_x + 20, off_y + 44), font_size=28, text_color=result_color)

    # sidebar base
    cv2.rectangle(dashboard, (side_x1, side_y1), (side_x2, side_y2), (249, 250, 252), -1)
    cv2.rectangle(dashboard, (side_x1, side_y1), (side_x2, side_y2), (208, 214, 222), 1)
    cv2.rectangle(dashboard, (side_x1, side_y1), (side_x2, side_y1 + 8), (40, 40, 40), -1)

    def put(line, y, x_offset=14, scale=0.6, color=(30, 34, 40), thick=1):
        cv2.putText(dashboard, line, (side_x1 + x_offset, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    mode_txt = {"WAIT": "WAITING", "COUNTDOWN": "READY...", "RECORDING": "ANALYZING"}.get(mode, mode)
    mode_color = (60, 190, 90) if mode == "WAIT" else (0, 140, 255) if mode == "RECORDING" else (225, 150, 60)
    draw_led(dashboard, (side_x1 + 30, side_y1 + 45), 10, mode_color, active=True)
    put(mode_txt, side_y1 + 52, x_offset=50, scale=0.78, thick=2, color=mode_color)
    put(f"FPS: {fps_smooth:.1f}", side_y1 + 50, x_offset=sidebar_w - 110, scale=0.58, color=(100, 100, 100))

    put("SIGN DASHBOARD", side_y1 + 30, scale=0.72, thick=2)
    # System status
    status_y = side_y1 + 92
    put("SYSTEM STATUS", status_y, scale=0.55, thick=2)
    draw_led(dashboard, (side_x1 + 25, status_y + 25), 6, (60, 190, 90), active=left_seen)
    put("VIS_L", status_y + 30, x_offset=38, scale=0.45, color=(70, 70, 70))
    draw_led(dashboard, (side_x1 + 105, status_y + 25), 6, (60, 190, 90), active=right_seen)
    put("VIS_R", status_y + 30, x_offset=118, scale=0.45, color=(70, 70, 70))
    draw_led(dashboard, (side_x1 + 185, status_y + 25), 6, (60, 190, 90), active=(sensor_packet_count > 0))
    put("GLOVE", status_y + 30, x_offset=198, scale=0.45, color=(70, 70, 70))
    if overlap_now:
        draw_led(dashboard, (side_x1 + 280, status_y + 25), 6, (0, 165, 255), active=True)
        put("OVERLAP", status_y + 30, x_offset=293, scale=0.45, color=(0, 120, 200))

    if overlap_now:
        status_msg = "VISION LIMITED  |  OVERLAP DETECTED"
        status_msg_color = (0, 120, 200)
    elif not (left_seen and right_seen):
        status_msg = "VISION TRACKING PARTIAL"
        status_msg_color = (120, 120, 120)
    elif sensor_packet_count <= 0:
        status_msg = "GLOVE STREAM CHECK"
        status_msg_color = (180, 80, 80)
    else:
        status_msg = "VISION + GLOVE ACTIVE"
        status_msg_color = (60, 140, 90)
    put(status_msg, status_y + 58, x_offset=14, scale=0.46, color=status_msg_color, thick=2)

    result_y = status_y + 88
    put("FINAL RESULT", result_y, scale=0.55, thick=2)
    result_box_y1 = result_y + 10
    result_box_y2 = result_box_y1 + 52
    cv2.rectangle(dashboard, (side_x1 + 14, result_box_y1), (side_x2 - 14, result_box_y2), (255, 255, 255), -1)
    cv2.rectangle(dashboard, (side_x1 + 14, result_box_y1), (side_x2 - 14, result_box_y2), (220, 224, 230), 1)
    result_text_color = result_color if stable_label != 'none' else (70, 74, 82)
    draw_text_unicode(dashboard, stable_kor, (side_x1 + 24, result_box_y1 + 10), font_size=28, text_color=result_text_color)

    # Final confidence only
    conf_y = result_y + 86
    put("AI CONFIDENCE", conf_y, scale=0.55, thick=2)
    if stable_conf >= 0.70:
        bar_color = (60, 190, 90)
    elif stable_conf >= 0.40:
        bar_color = (0, 165, 255)
    else:
        bar_color = (60, 60, 220)
    put(f"{stable_conf * 100:.1f}%", conf_y + 25, x_offset=sidebar_w - 70, scale=0.60, thick=2, color=bar_color)
    draw_progress_bar(dashboard, side_x1 + 14, conf_y + 12, sidebar_w - 90, 16, stable_conf, bar_color)

    # Top-3 with emphasized top1
    cand_y = conf_y + 64
    put("TOP PREDICTIONS", cand_y, scale=0.55, thick=2)
    for idx, cand in enumerate(top_candidates[:3]):
        c_label = cand.get('label', '-')
        c_kor = KOR_MAP.get(c_label, c_label)
        c_conf = cand.get('conf', 0.0)
        if idx == 0 and c_conf > 0.40:
            draw_text_unicode(dashboard, f"1. {c_kor}", (side_x1 + 14, cand_y + 15), font_size=24, text_color=(0, 0, 0))
            put(f"{c_conf:.2f}", cand_y + 35, x_offset=sidebar_w - 60, scale=0.60, thick=2, color=(0, 0, 0))
        else:
            draw_text_unicode(dashboard, f"{idx + 1}. {c_kor}", (side_x1 + 14, cand_y + 20 + idx * 30), font_size=18, text_color=(100, 100, 100))
            put(f"{c_conf:.2f}", cand_y + 35 + idx * 30, x_offset=sidebar_w - 60, scale=0.50, color=(100, 100, 100))

    # Main sensor graph
    sens_y = cand_y + 126
    put("LIVE SENSOR STREAM (FLEX)", sens_y, scale=0.55, thick=2)
    left_flex = float(np.mean(sensor_vec[0:4]))
    right_flex = float(np.mean(sensor_vec[4:8]))
    put(f"L: {left_flex:.2f} | R: {right_flex:.2f}", sens_y, x_offset=sidebar_w - 130, scale=0.45, color=(120, 120, 120))
    draw_line_chart(
        dashboard,
        side_x1 + 14,
        sens_y + 15,
        side_x2 - 14,
        sens_y + 195,
        {
            "flex_l": (flex_left_hist, (193, 113, 56)),
            "flex_r": (flex_right_hist, (89, 89, 211)),
        },
    )

    # Bottom lightweight footer
    seq_y = sens_y + 226
    put(f"BUFFER: {sequence_len}/{SEQ_LEN}", seq_y, scale=0.45, color=(100, 100, 100))
    draw_progress_bar(dashboard, side_x1 + 14, seq_y + 10, sidebar_w - 28, 6, sequence_len / max(1, SEQ_LEN), (100, 100, 100))
    put("KEYS  S/B/N/G/Q", seq_y + 38, scale=0.48, color=(70, 74, 82), thick=2)

    return dashboard

def reset_runtime_state():
    sequence.clear()
    hand_presence_history.clear()
    recent_frame_motion.clear()

def draw_center_text(img, title, subtitle="", remain_text="", bar_color=(40, 40, 40)):
    h, w, _ = img.shape
    draw_text_box(img, (0, 0), (w, 92), fill_color=(255, 255, 255), alpha=0.58)
    draw_text_unicode(img, "수집 준비 안내", (16, 18), font_size=30, text_color=(0, 0, 0))
    draw_text_unicode(img, title, (40, 200), font_size=34, text_color=(255, 255, 255))

    if subtitle:
        draw_text_unicode(img, subtitle, (40, 246), font_size=28, text_color=(235, 235, 235))

    if remain_text:
        draw_text_unicode(img, remain_text, (220, 332), font_size=40, text_color=(0, 255, 255))

    draw_text_box(img, (10, h - 46), (132, h - 8), fill_color=(255, 255, 255), alpha=0.48)
    draw_text_unicode(img, "Q : 종료", (18, h - 40), font_size=24, text_color=(0, 0, 0))

def show_startup_calibration_guide():
    if not wait_for_serial_connected(serial_reader, timeout_sec=5):
        return False

    for title, subtitle, duration_sec, bar_color in STARTUP_GUIDE_STAGES:
        stage_start = time.time()

        while True:
            elapsed = time.time() - stage_start
            remain = duration_sec - elapsed
            if remain <= 0:
                break

            ret, img = cap.read()
            if not ret:
                continue

            img = cv2.flip(img, 1)
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            remain_text = f"{int(np.ceil(remain))} sec"
            draw_center_text(img, title, subtitle, remain_text, bar_color=bar_color)

            if results.multi_hand_landmarks and results.multi_handedness and DRAW_LANDMARKS:
                for res_hand, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    if label == 'Left':
                        draw_custom_landmarks(img, res_hand, (255, 0, 0), "L")
                    else:
                        draw_custom_landmarks(img, res_hand, (0, 0, 255), "R")

            cv2.imshow('Fusion Real-time Sign Recognition v5', img)
            key = cv2.waitKey(1)

            if key in [ord('q'), ord('Q')]:
                return None

    return True

def show_wait_sensor_screen():
    start = time.time()

    while True:
        if serial_reader.connection_error is not None:
            return False

        _, _, packet_count, _ = serial_reader.get_latest()
        if packet_count > 0:
            print("[Sensor] Ready")
            return True

        if time.time() - start > 12:
            print("[Sensor] Timeout: 센서 패킷이 들어오지 않습니다.")
            return False

        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        draw_center_text(
            img,
            "센서 패킷을 기다리는 중입니다",
            "잠시만 기다려주세요",
            "",
            bar_color=(40, 40, 40)
        )

        cv2.imshow('Fusion Real-time Sign Recognition v5', img)
        key = cv2.waitKey(1)
        if key in [ord('q'), ord('Q')]:
            return None

@tf.function
def infer(vision_input_tensor, sensor_input_tensor, flex_input_tensor=None):
    model_inputs = [vision_input_tensor, sensor_input_tensor]
    if MODEL_INPUT_COUNT >= 3:
        if flex_input_tensor is None:
            flex_input_tensor = tf.zeros((tf.shape(vision_input_tensor)[0], FLEX_POSTURE_DIM), dtype=tf.float32)
        model_inputs.append(flex_input_tensor)
    return model(model_inputs, training=False)

dummy_vision = np.zeros((1, SEQ_LEN, VISION_DIM), dtype=np.float32)
dummy_sensor = np.zeros((1, SEQ_LEN, MODEL_SENSOR_DIM), dtype=np.float32)
if MODEL_INPUT_COUNT >= 3:
    dummy_flex = np.zeros((1, FLEX_POSTURE_DIM), dtype=np.float32)
    _ = infer(
        tf.convert_to_tensor(dummy_vision),
        tf.convert_to_tensor(dummy_sensor),
        tf.convert_to_tensor(dummy_flex),
    )
else:
    _ = infer(
        tf.convert_to_tensor(dummy_vision),
        tf.convert_to_tensor(dummy_sensor)
    )

# =========================
# 센서 시작
# =========================
serial_reader = SerialReader(SERIAL_PORT, SERIAL_BAUD)
serial_reader.start()

guide_result = show_startup_calibration_guide()
if guide_result is None:
    serial_reader.close()
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit
if guide_result is False:
    serial_reader.close()
    cap.release()
    cv2.destroyAllWindows()
    raise RuntimeError("시리얼 연결 실패: COM 포트 확인 또는 시리얼 모니터 종료 필요")

ready_result = show_wait_sensor_screen()
if ready_result is None:
    serial_reader.close()
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit
if ready_result is False:
    serial_reader.close()
    cap.release()
    cv2.destroyAllWindows()
    raise RuntimeError("센서 준비 실패: COM 포트 또는 Arduino 출력 상태 확인 필요")

# =========================
# 상태 변수
# =========================
sequence = deque(maxlen=SEQ_LEN)
hand_presence_history = deque(maxlen=SEQ_LEN)
recent_frame_motion = deque(maxlen=SHORT_MOTION_WINDOW)

prev_left = np.zeros(63, dtype=np.float32)
prev_right = np.zeros(63, dtype=np.float32)
prev_left_center = None
prev_right_center = None
prev_vision_vec = np.zeros(VISION_DIM, dtype=np.float32)

left_missing_count = 0
right_missing_count = 0
no_hand_run = 0
post_overlap_hold_count = 0
overlap_freeze_count = 0
left_recent_motion_px = 0.0
right_recent_motion_px = 0.0

mode = "WAIT"
countdown_start_time = None

stable_label = 'none'
stable_conf = 0.0
current_pred_label = ''
current_pred_conf = 0.0
current_top_candidates = []
long_motion_score = 0.0
short_motion_score = 0.0
last_margin = 0.0
gt_label_index = -1

last_sensor_time_ms = 0
last_sensor_host_delta_ms = 0.0
sensor_packet_count = 0
zero_any_frame_indices = []
zero_both_frame_indices = []
zero_reason_counts = {}
flex_left_history = deque(maxlen=PLOT_HISTORY_LEN)
flex_right_history = deque(maxlen=PLOT_HISTORY_LEN)
imu_left_history = deque(maxlen=PLOT_HISTORY_LEN)
imu_right_history = deque(maxlen=PLOT_HISTORY_LEN)

fps_prev_time = time.time()
fps_smooth = 0.0
fps_limiter = FpsLimiter(TARGET_FPS)
runtime_profiler = RuntimeProfiler(enabled=PROFILE_ENABLED, report_dir=PROFILE_DIR)

print(f"[Runtime] target_fps={TARGET_FPS:.1f} profile_enabled={PROFILE_ENABLED} profile_dir={PROFILE_DIR}")
print(f"[Runtime] camera={CAM_WIDTH}x{CAM_HEIGHT} min_stable_fps={MIN_STABLE_FPS:.1f}")

os.makedirs(DEBUG_CAPTURE_DIR, exist_ok=True)


def get_current_gt_label():
    if 0 <= gt_label_index < len(ACTIONS):
        return ACTIONS[gt_label_index]
    return None


def save_debug_capture(seq_array, result, frame_count, sensor_time_ms, sensor_host_delta_ms):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    label = str(result.get('raw_label', 'unknown'))
    stem = f"{timestamp}_{label}"
    npy_path = os.path.join(DEBUG_CAPTURE_DIR, f"{stem}.npy")
    json_path = os.path.join(DEBUG_CAPTURE_DIR, f"{stem}.json")
    gt_label = get_current_gt_label()
    pred_label = result.get("raw_label")

    np.save(npy_path, seq_array.astype(np.float32))

    meta = {
        "timestamp": timestamp,
        "model_dir": DEFAULT_MODEL_DIR,
        "model_path": MODEL_PATH,
        "seq_len": int(seq_array.shape[0]),
        "feature_dim": int(seq_array.shape[1]),
        "captured_frames": int(frame_count),
        "raw_label": result.get("raw_label"),
        "raw_conf": float(result.get("raw_conf", 0.0)),
        "label": result.get("label"),
        "conf": float(result.get("conf", 0.0)),
        "margin": float(result.get("margin", 0.0)),
        "long_motion": float(result.get("long_motion", 0.0)),
        "hand_count": int(result.get("hand_count", 0)),
        "sensor_time_ms": int(sensor_time_ms),
        "sensor_host_delta_ms": float(sensor_host_delta_ms),
        "zero_any_count": int(len(zero_any_frame_indices)),
        "zero_both_count": int(len(zero_both_frame_indices)),
        "zero_any_indices": list(zero_any_frame_indices),
        "zero_both_indices": list(zero_both_frame_indices),
        "zero_reason_counts": dict(zero_reason_counts),
        "gt_label": gt_label,
        "gt_matches_prediction": (gt_label == pred_label) if gt_label is not None else None,
        "actions": ACTIONS,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DebugCapture] saved: {npy_path}")
    print(f"[DebugCaptureMeta] saved: {json_path}")


def reset_recording_buffers():
    global left_missing_count, right_missing_count, no_hand_run
    global current_pred_label, current_pred_conf, current_top_candidates
    global long_motion_score, short_motion_score, last_margin
    global prev_left_center, prev_right_center
    global post_overlap_hold_count, overlap_freeze_count
    global left_recent_motion_px, right_recent_motion_px
    global zero_any_frame_indices, zero_both_frame_indices, zero_reason_counts

    sequence.clear()
    hand_presence_history.clear()
    recent_frame_motion.clear()

    prev_left[:] = 0
    prev_right[:] = 0
    prev_vision_vec[:] = 0
    prev_left_center = None
    prev_right_center = None

    left_missing_count = 0
    right_missing_count = 0
    no_hand_run = 0
    post_overlap_hold_count = 0
    overlap_freeze_count = 0
    left_recent_motion_px = 0.0
    right_recent_motion_px = 0.0
    zero_any_frame_indices = []
    zero_both_frame_indices = []
    zero_reason_counts = {
        'overlap': 0,
        'ambiguous_slot': 0,
        'missing_left': 0,
        'missing_right': 0,
        'both_missing': 0,
    }

    current_pred_label = ''
    current_pred_conf = 0.0
    current_top_candidates = []
    long_motion_score = 0.0
    short_motion_score = 0.0
    last_margin = 0.0


def reset_all_state():
    global mode, countdown_start_time, stable_label, stable_conf, current_top_candidates

    reset_recording_buffers()
    mode = "WAIT"
    countdown_start_time = None
    stable_label = 'none'
    stable_conf = 0.0
    current_top_candidates = []


# =========================
# Main loop
# =========================
try:
    while True:
        loop_start_perf = time.perf_counter()
        cap_read_start = time.perf_counter()
        ret, frame = cap.read()
        runtime_profiler.add_duration("camera_read", cap_read_start)
        if not ret:
            continue

        preprocess_start = time.perf_counter()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
        display_img = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        runtime_profiler.add_duration("frame_preprocess", preprocess_start)
        mediapipe_start = time.perf_counter()
        results = hands.process(rgb)
        runtime_profiler.add_duration("mediapipe_process", mediapipe_start)
        rgb.flags.writeable = True

        frame_host_time_ms = time.perf_counter() * 1000.0
        sensor_align_start = time.perf_counter()
        sensor_time_ms, sensor_vec, sensor_packet_count, sensor_host_time_ms = serial_reader.get_aligned_packet(frame_host_time_ms)
        runtime_profiler.add_duration("sensor_align", sensor_align_start)
        last_sensor_time_ms = sensor_time_ms
        last_sensor_host_delta_ms = frame_host_time_ms - sensor_host_time_ms
        flex_left_history.append(float(np.mean(sensor_vec[0:4])))
        flex_right_history.append(float(np.mean(sensor_vec[4:8])))
        imu_left_history.append(float(np.linalg.norm(sensor_vec[8:17])))
        imu_right_history.append(float(np.linalg.norm(sensor_vec[17:26])))

        left_data = np.zeros(63, dtype=np.float32)
        right_data = np.zeros(63, dtype=np.float32)
        left_seen = False
        right_seen = False
        zero_vision_reason = None

        candidates = make_hand_candidates(results, CAM_WIDTH, CAM_HEIGHT)
        overlap_now = hands_are_overlapping(candidates)
        frame_ambiguous = False

        overlap_freeze_applied = False
        left_missing_freeze = False
        right_missing_freeze = False

        if mode == "RECORDING" and overlap_now:
            if (
                overlap_freeze_count < SHORT_OVERLAP_FREEZE_FRAMES
                and prev_left_center is not None
                and prev_right_center is not None
                and np.any(np.abs(prev_left) > 1e-6)
                and np.any(np.abs(prev_right) > 1e-6)
            ):
                overlap_freeze_applied = True
                overlap_freeze_count += 1
                assigned = {'left': None, 'right': None}
            else:
                assigned = {'left': None, 'right': None}
                frame_ambiguous = True
                zero_vision_reason = "overlap"
        else:
            overlap_freeze_count = 0
            assigned, frame_ambiguous = assign_hand_slots(candidates, prev_left_center, prev_right_center)
            if frame_ambiguous:
                zero_vision_reason = "ambiguous_slot"

        left_candidate = assigned['left']
        right_candidate = assigned['right']

        if frame_ambiguous:
            warn_text = 'VISION AMBIGUOUS - ZERO VISION FRAME'
            if zero_vision_reason == "overlap":
                warn_text = 'HAND OVERLAP - ZERO VISION FRAME'
            cv2.putText(display_img, warn_text, (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        elif overlap_freeze_applied:
            cv2.putText(display_img, 'SHORT OVERLAP - HOLD PREV VISION', (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        if left_candidate is not None:
            left_data = left_candidate['data']
            left_seen = True
            if left_candidate.get('res_hand') is not None:
                draw_custom_landmarks(display_img, left_candidate['res_hand'], (255, 0, 0), 'L')
        elif overlap_freeze_applied:
            left_data = prev_left.copy()

        if right_candidate is not None:
            right_data = right_candidate['data']
            right_seen = True
            if right_candidate.get('res_hand') is not None:
                draw_custom_landmarks(display_img, right_candidate['res_hand'], (0, 0, 255), 'R')
        elif overlap_freeze_applied:
            right_data = prev_right.copy()

        any_hand_seen = left_seen or right_seen

        if mode == "RECORDING":
            if left_candidate is not None:
                if prev_left_center is not None:
                    left_recent_motion_px = float(
                        0.7 * left_recent_motion_px
                        + 0.3 * np.linalg.norm(left_candidate['center'] - prev_left_center)
                    )
                else:
                    left_recent_motion_px = 0.0
                prev_left = left_data.copy()
                prev_left_center = left_candidate['center'].copy()
                left_missing_count = 0
            else:
                left_missing_count += 1
                left_is_static_support = (
                    prev_left_center is not None
                    and np.any(np.abs(prev_left) > 1e-6)
                    and left_recent_motion_px <= STATIC_SUPPORT_MOTION_PX
                )
                if (
                    left_missing_count <= (
                        MISSING_HOLD_FRAMES if left_is_static_support else SHORT_MISSING_FREEZE_FRAMES
                    )
                    and prev_left_center is not None
                    and np.any(np.abs(prev_left) > 1e-6)
                ):
                    left_data = prev_left.copy()
                    left_missing_freeze = True
                elif left_missing_count > MISSING_HOLD_FRAMES:
                    prev_left = np.zeros(63, dtype=np.float32)
                    left_data = prev_left.copy()
                    prev_left_center = None
                    left_recent_motion_px = 0.0

            if right_candidate is not None:
                if prev_right_center is not None:
                    right_recent_motion_px = float(
                        0.7 * right_recent_motion_px
                        + 0.3 * np.linalg.norm(right_candidate['center'] - prev_right_center)
                    )
                else:
                    right_recent_motion_px = 0.0
                prev_right = right_data.copy()
                prev_right_center = right_candidate['center'].copy()
                right_missing_count = 0
            else:
                right_missing_count += 1
                right_is_static_support = (
                    prev_right_center is not None
                    and np.any(np.abs(prev_right) > 1e-6)
                    and right_recent_motion_px <= STATIC_SUPPORT_MOTION_PX
                )
                if (
                    right_missing_count <= (
                        MISSING_HOLD_FRAMES if right_is_static_support else SHORT_MISSING_FREEZE_FRAMES
                    )
                    and prev_right_center is not None
                    and np.any(np.abs(prev_right) > 1e-6)
                ):
                    right_data = prev_right.copy()
                    right_missing_freeze = True
                elif right_missing_count > MISSING_HOLD_FRAMES:
                    prev_right = np.zeros(63, dtype=np.float32)
                    right_data = prev_right.copy()
                    prev_right_center = None
                    right_recent_motion_px = 0.0

            if any_hand_seen:
                no_hand_run = 0
                hand_presence_history.append(1)
            else:
                no_hand_run += 1
                hand_presence_history.append(0)

            vision_vec = np.concatenate([left_data, right_data]).astype(np.float32)
            frame_idx_1based = len(sequence) + 1
            left_zero_now = not np.any(np.abs(left_data) > 1e-6)
            right_zero_now = not np.any(np.abs(right_data) > 1e-6)
            if left_zero_now or right_zero_now:
                zero_any_frame_indices.append(frame_idx_1based)
            if left_zero_now and right_zero_now:
                zero_both_frame_indices.append(frame_idx_1based)
            if zero_vision_reason in zero_reason_counts:
                zero_reason_counts[zero_vision_reason] += 1
            elif overlap_freeze_applied:
                pass
            elif left_zero_now and right_zero_now:
                zero_reason_counts['both_missing'] += 1
            elif left_zero_now:
                zero_reason_counts['missing_left'] += 1
            elif right_zero_now:
                zero_reason_counts['missing_right'] += 1

            sensor_model_vec = build_sensor_model_vec(
                sensor_vec=sensor_vec,
                left_seen=left_seen,
                right_seen=right_seen,
                left_held=bool(overlap_freeze_applied),
                right_held=bool(overlap_freeze_applied),
                overlap_now=overlap_now,
                frame_ambiguous=frame_ambiguous,
            )

            frame_motion = float(np.linalg.norm(vision_vec - prev_vision_vec))
            recent_frame_motion.append(frame_motion)
            prev_vision_vec = vision_vec.copy()
            short_motion_score = float(np.mean(recent_frame_motion)) if len(recent_frame_motion) > 0 else 0.0

            full_joint = np.concatenate([vision_vec, sensor_model_vec]).astype(np.float32)
            sequence.append(full_joint)

            if no_hand_run >= NO_HAND_RESET_FRAMES:
                reset_all_state()

            elif len(sequence) >= SEQ_LEN:
                seq_array = np.array(sequence, dtype=np.float32)
                result = classify_sequence_once(seq_array)
                if should_hold_as_none(result, len(zero_both_frame_indices), zero_reason_counts):
                    result['label'] = 'none'
                    result['conf'] = result['raw_conf']
                save_debug_capture(
                    seq_array,
                    result,
                    len(sequence),
                    last_sensor_time_ms,
                    last_sensor_host_delta_ms,
                )

                current_pred_label = result['raw_label']
                current_pred_conf = result['raw_conf']
                stable_label = result['label']
                stable_conf = result['conf']
                current_top_candidates = result.get('top_candidates', [])
                long_motion_score = result['long_motion']
                last_margin = result['margin']

                kor_text = KOR_MAP.get(stable_label, stable_label)
                print(
                    f"[Result] stable={stable_label} ({kor_text}) | conf={stable_conf:.3f} | "
                    f"margin={last_margin:.3f} | motion={long_motion_score:.3f}"
                )
                first_zero_any = zero_any_frame_indices[0] if zero_any_frame_indices else "-"
                first_zero_both = zero_both_frame_indices[0] if zero_both_frame_indices else "-"
                print(
                    f"[VisionZero] any_zero={len(zero_any_frame_indices)}/{SEQ_LEN} "
                    f"first_any={first_zero_any} indices={zero_any_frame_indices}"
                )
                print(
                    f"[VisionZeroBoth] both_zero={len(zero_both_frame_indices)}/{SEQ_LEN} "
                    f"first_both={first_zero_both} indices={zero_both_frame_indices}"
                )
                print(f"[VisionZeroReason] {zero_reason_counts}")

                mode = "WAIT"
                countdown_start_time = None

        elif mode == "COUNTDOWN":
            elapsed = time.time() - countdown_start_time
            remain = TRIGGER_DELAY_SEC - elapsed
            if remain <= 0:
                reset_recording_buffers()
                mode = "RECORDING"

        now = time.time()
        instant_fps = 1.0 / max(now - fps_prev_time, 1e-6)
        fps_prev_time = now
        fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps if fps_smooth > 0 else instant_fps

        render_start = time.perf_counter()
        gt_label = get_current_gt_label()
        gt_kor = KOR_MAP.get(gt_label, gt_label) if gt_label else "미지정"
        if mode == "COUNTDOWN":
            remain = max(0.0, TRIGGER_DELAY_SEC - (time.time() - countdown_start_time))
            remain_int = int(np.ceil(remain))
            cv2.putText(display_img, str(remain_int), (display_img.shape[1] // 2 - 30, display_img.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 6)
        elif mode == "RECORDING":
            prog_w = int((len(sequence) / SEQ_LEN) * display_img.shape[1])
            cv2.rectangle(display_img, (0, display_img.shape[0] - 10), (prog_w, display_img.shape[0]), (0, 255, 0), -1)

        debug_lines = [
            f"Hands L={left_seen} R={right_seen}  overlap={overlap_now}",
            f"Raw={current_pred_label} conf={current_pred_conf:.3f} stable={stable_label} conf={stable_conf:.3f}",
            f"Margin={last_margin:.3f}  motion={long_motion_score:.3f} / {short_motion_score:.3f}",
            f"Sensor delta={last_sensor_host_delta_ms:.1f}ms  pkt={sensor_packet_count}",
        ]
        draw_debug_panel(display_img, debug_lines)

        dashboard_img = render_dashboard(
            camera_img=display_img,
            fps_smooth=fps_smooth,
            mode=mode,
            sequence_len=len(sequence),
            current_pred_label=current_pred_label,
            current_pred_conf=current_pred_conf,
            stable_label=stable_label,
            stable_conf=stable_conf,
            gt_kor=gt_kor,
            long_motion_score=long_motion_score,
            short_motion_score=short_motion_score,
            last_margin=last_margin,
            sensor_vec=sensor_vec,
            sensor_packet_count=sensor_packet_count,
            last_sensor_time_ms=last_sensor_time_ms,
            last_sensor_host_delta_ms=last_sensor_host_delta_ms,
            left_seen=left_seen,
            right_seen=right_seen,
            overlap_now=overlap_now,
            frame_ambiguous=frame_ambiguous,
            top_candidates=current_top_candidates,
            flex_left_hist=flex_left_history,
            flex_right_hist=flex_right_history,
            imu_left_hist=imu_left_history,
            imu_right_hist=imu_right_history,
        )

        cv2.imshow('Fusion Real-time Sign Recognition v5', dashboard_img)
        runtime_profiler.add_duration("render_and_imshow", render_start)

        waitkey_start = time.perf_counter()
        key = cv2.waitKey(1) & 0xFF
        runtime_profiler.add_duration("cv2_waitkey", waitkey_start)
        if key == ord('q'):
            break
        elif key == ord('s'):
            if mode == "WAIT":
                reset_recording_buffers()
                countdown_start_time = time.time()
                mode = "COUNTDOWN"
                print("시작 입력 감지. 2초 뒤 녹화를 시작합니다.")
        elif key == ord('c'):
            reset_all_state()
            print("상태를 초기화했습니다.")
        elif key == ord('d'):
            DRAW_LANDMARKS = not DRAW_LANDMARKS
            print(f'DRAW_LANDMARKS = {DRAW_LANDMARKS}')
        elif key == ord('p'):
            PRINT_DEBUG = not PRINT_DEBUG
            print(f'PRINT_DEBUG = {PRINT_DEBUG}')
        elif key == ord('i'):
            SHOW_DEBUG_OVERLAY = not SHOW_DEBUG_OVERLAY
            print(f'SHOW_DEBUG_OVERLAY = {SHOW_DEBUG_OVERLAY}')
        elif key == ord('n'):
            if ACTIONS:
                gt_label_index = (gt_label_index + 1) % len(ACTIONS)
                print(f"[GT] {get_current_gt_label()}")
        elif key == ord('b'):
            if ACTIONS:
                gt_label_index = len(ACTIONS) - 1 if gt_label_index < 0 else (gt_label_index - 1) % len(ACTIONS)
                print(f"[GT] {get_current_gt_label()}")
        elif key == ord('g'):
            gt_label_index = -1
            print("[GT] cleared")

        sleep_ms = fps_limiter.sleep_remaining(loop_start_perf)
        runtime_profiler.add_ms("fps_cap_sleep", sleep_ms)
        runtime_profiler.add_duration("loop_total", loop_start_perf)

finally:
    profile_path = runtime_profiler.save(
        "nsu_run_fusion_v6_overlap_hold",
        TARGET_FPS,
        extra={
            "model_path": MODEL_PATH,
            "debug_capture_dir": DEBUG_CAPTURE_DIR,
            "serial_port": SERIAL_PORT,
            "serial_baud": SERIAL_BAUD,
            "camera_width": CAM_WIDTH,
            "camera_height": CAM_HEIGHT,
            "min_stable_fps": MIN_STABLE_FPS,
        },
    )
    if profile_path:
        print(f"[RuntimeProfile] saved: {profile_path}")
    serial_reader.close()
    cap.release()
    cv2.destroyAllWindows()
