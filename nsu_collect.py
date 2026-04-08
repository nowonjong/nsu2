import os
import sys
import csv
import time
import json
import threading
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import serial
import serial.tools.list_ports
from PIL import Image, ImageDraw, ImageFont


# =========================
# 설정
# =========================
ACTIONS = [
    'more', 'naeil', 'eoje', 'teukbyeol',
    'byeollo', 'jamkkan', 'oraenman', 'gakkapda',
    'jalhada', 'annyeonghaseyo', 'mannada', 'byeongyeonghada',
    'banggeum', 'billida', 'gandanhada', 'sada',
    'gamsahamnida', 'joesonghada',
]

ACTION_DISPLAY_NAMES = {
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
    'mannada': '만나다',
    'byeongyeonghada': '변경하다',
    'banggeum': '방금',
    'billida': '빌리다',
    'gandanhada': '간단하다',
    'sada': '사다',
    'gamsahamnida': '감사합니다',
    'joesonghada': '죄송하다',
}

SEQ_LENGTH = 60
SAMPLES_PER_ACTION = 999999
DATA_PATH = 'dataset_fusion_hold_v4_flags'

SERIAL_PORT = 'COM9'          # 실제 포트 번호로 수정
SERIAL_BAUD = 115200
CAM_INDEX = 0

SWAP_HAND_LABELS = False      # 좌우가 뒤집히면 True
HANDEDNESS_SCORE_TH = 0.55
OVERLAP_IOU_THRESHOLD = 0.24
OVERLAP_CENTER_DIST_PX = 60.0
SLOT_AMBIGUOUS_MARGIN = 35.0
MISSING_HOLD_FRAMES = 6
SHORT_OVERLAP_FREEZE_FRAMES = 4
SHORT_MISSING_FREEZE_FRAMES = 3
STATIC_SUPPORT_MOTION_PX = 18.0

WINDOW_NAME = 'Collect Fusion Data'
COLLECT_SESSION_ID = time.strftime('%Y%m%d_%H%M%S')
ALIGN_WARN_DELTA_MS = 33.0
ALIGN_BAD_DELTA_MS = 50.0
# 현재 Arduino 코드 기준 대략적인 캘리브레이션 안내 시간
# setup: delay(1000) + "잠시 가만히" delay(3500) ≈ 4.5초
# straight: delay(3000) + average(100*10ms=1초) ≈ 4.0초
# bent: delay(3000) + average(100*10ms=1초) ≈ 4.0초
STARTUP_GUIDE_STAGES = [
    ("장갑을 움직이지 말고 유지하세요", "IMU 안정화 중", 4.5, (40, 40, 40)),
    ("검지부터 새끼손가락까지 쭉 펴주세요", "Flex 펴짐 보정 중", 4.0, (60, 70, 40)),
    ("검지부터 새끼손가락까지 구부려주세요", "Flex 굽힘 보정 중", 4.0, (70, 40, 40)),
]

UI_FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
_FONT_CACHE = {}
WINDOW_INITIALIZED = False

# =========================
# 센서 순서 (26차원)
# Arduino CSV 출력 순서와 동일
# =========================
RAW_SENSOR_KEYS = [
    'lf0_norm', 'lf1_norm', 'lf2_norm', 'lf3_norm',
    'rf0_norm', 'rf1_norm', 'rf2_norm', 'rf3_norm',

    'l_ax_g', 'l_ay_g', 'l_az_g',
    'l_gx_dps', 'l_gy_dps', 'l_gz_dps',
    'l_roll_deg', 'l_pitch_deg', 'l_yaw_deg',

    'r_ax_g', 'r_ay_g', 'r_az_g',
    'r_gx_dps', 'r_gy_dps', 'r_gz_dps',
    'r_roll_deg', 'r_pitch_deg', 'r_yaw_deg',
]

FRAME_FLAG_KEYS = [
    'left_detected',
    'right_detected',
    'left_held',
    'right_held',
    'overlap_flag',
    'ambiguous_flag',
]

SENSOR_KEYS = RAW_SENSOR_KEYS + FRAME_FLAG_KEYS


# =========================
# 폴더 준비
# =========================
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

for action in ACTIONS:
    action_dir = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_dir):
        os.makedirs(action_dir)


# =========================
# MediaPipe Hands
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    sys.exit()


# =========================
# 시리얼 유틸
# =========================
def list_serial_ports():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        print(f"{p.device} - {p.description}")


class SerialReader(threading.Thread):
    def __init__(self, port, baud):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud

        self.ser = None
        self.stop_flag = False
        self.lock = threading.Lock()

        self.latest_sensor_time_ms = 0
        self.latest_sensor_vector = np.zeros(len(RAW_SENSOR_KEYS), dtype=np.float32)
        self.latest_host_time_ms = 0.0
        self.packet_count = 0
        self.packet_buffer = deque(maxlen=256)

        self.connected = False
        self.connection_error = None

    def connect(self):
        self.ser = serial.Serial(self.port, self.baud, timeout=0.05)
        time.sleep(2.0)  # 포트 오픈 후 아두이노 재시작 대기
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

        # timestamp 1개 + sensor 26개 = 총 27개
        if len(parts) != 27:
            return None

        try:
            values = [float(x) for x in parts]
        except ValueError:
            return None

        sensor_time_ms = int(values[0])
        sensor_vec = np.array(values[1:], dtype=np.float32)
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
                    # calibration 메시지 / header 문장 등 무시
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
                self.latest_host_time_ms
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


def wait_for_sensor_ready(serial_reader, timeout_sec=10):
    start = time.time()
    while time.time() - start < timeout_sec:
        if serial_reader.connection_error is not None:
            return False

        _, _, packet_count, _ = serial_reader.get_latest()
        if packet_count > 0:
            print("[Sensor] Ready")
            return True

        time.sleep(0.05)

    print("[Sensor] Timeout: 센서 패킷이 안 들어왔습니다.")
    return False


# =========================
# 파일 / 카운트 유틸
# =========================
def get_action_dir(action):
    return os.path.join(DATA_PATH, action)


def get_action_display_name(action):
    return ACTION_DISPLAY_NAMES.get(action, action)


def get_npy_files(action):
    action_dir = get_action_dir(action)
    files = []
    for f in os.listdir(action_dir):
        if f.endswith('.npy'):
            files.append(f)
    return files


def get_sample_count(action):
    return len(get_npy_files(action))


def get_next_sample_idx(action):
    files = get_npy_files(action)
    if not files:
        return 0

    max_idx = -1
    for f in files:
        name = os.path.splitext(f)[0]   # hello_0003
        if '_' in name:
            try:
                idx = int(name.split('_')[-1])
                max_idx = max(max_idx, idx)
            except ValueError:
                pass
    return max_idx + 1


# =========================
# CSV 헤더
# =========================
def make_csv_header():
    header = [
        'collect_session_id',
        'pc_time_ms',
        'frame_host_time_ms',
        'sensor_host_time_ms',
        'sensor_host_delta_ms',
        'sensor_time_ms',
        'action',
        'sample_idx',
        'frame_idx'
    ]

    for hand_name in ['left', 'right']:
        for i in range(21):
            header.append(f'vis_{hand_name}_{i}_x')
            header.append(f'vis_{hand_name}_{i}_y')
            header.append(f'vis_{hand_name}_{i}_z')

    header.extend(SENSOR_KEYS)
    return header


CSV_PATH = os.path.join(DATA_PATH, 'fusion_frame_log.csv')
csv_exists = os.path.exists(CSV_PATH)
csv_file = open(CSV_PATH, 'a', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)

if not csv_exists:
    csv_writer.writerow(make_csv_header())
    csv_file.flush()


# =========================
# 비전 유틸
# =========================
def get_hand_data(res_hand):
    """
    21개 랜드마크를 손목 기준 상대좌표로 변환한 뒤,
    손 크기 차이를 줄이기 위해 스케일 정규화하여 63차원으로 반환
    """
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
    return joint.flatten()  # (63,)


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


def mp_label_to_person_side(label, frame_flipped=True):
    if label not in ('Left', 'Right'):
        return None
    if frame_flipped:
        return label.lower()
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

        if SWAP_HAND_LABELS:
            label = 'Right' if label == 'Left' else 'Left'

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


def compute_alignment_quality(frame_records):
    deltas = np.array(
        [abs(item['sensor_host_delta_ms']) for item in frame_records],
        dtype=np.float64
    )

    if len(deltas) == 0:
        return {
            'frame_count': 0,
            'delta_mean_ms': 0.0,
            'delta_std_ms': 0.0,
            'delta_max_ms': 0.0,
            'delta_p95_ms': 0.0,
            'warn_over_33ms_frames': 0,
            'bad_over_50ms_frames': 0,
            'warn_over_33ms_ratio': 0.0,
            'bad_over_50ms_ratio': 0.0,
        }

    warn_count = int(np.sum(deltas > ALIGN_WARN_DELTA_MS))
    bad_count = int(np.sum(deltas > ALIGN_BAD_DELTA_MS))
    return {
        'frame_count': int(len(deltas)),
        'delta_mean_ms': float(np.mean(deltas)),
        'delta_std_ms': float(np.std(deltas)),
        'delta_max_ms': float(np.max(deltas)),
        'delta_p95_ms': float(np.percentile(deltas, 95)),
        'warn_over_33ms_frames': warn_count,
        'bad_over_50ms_frames': bad_count,
        'warn_over_33ms_ratio': float(warn_count / len(deltas)),
        'bad_over_50ms_ratio': float(bad_count / len(deltas)),
    }


def compute_zero_frame_quality(frame_records):
    if not frame_records:
        return {
            'any_zero_frames': 0,
            'both_zero_frames': 0,
            'first_any_zero_frame': None,
            'first_both_zero_frame': None,
            'any_zero_ratio': 0.0,
            'both_zero_ratio': 0.0,
            'reason_counts': {},
        }

    any_zero = [item['frame_idx'] for item in frame_records if item['left_zero'] or item['right_zero']]
    both_zero = [item['frame_idx'] for item in frame_records if item['left_zero'] and item['right_zero']]
    reason_counts = {}
    for item in frame_records:
        reason = item.get('zero_reason')
        if reason is None:
            continue
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    frame_count = len(frame_records)
    return {
        'any_zero_frames': int(len(any_zero)),
        'both_zero_frames': int(len(both_zero)),
        'first_any_zero_frame': int(any_zero[0]) if any_zero else None,
        'first_both_zero_frame': int(both_zero[0]) if both_zero else None,
        'any_zero_ratio': float(len(any_zero) / frame_count),
        'both_zero_ratio': float(len(both_zero) / frame_count),
        'reason_counts': reason_counts,
    }


def draw_custom_landmarks(img, hand_landmarks, color, label):
    h, w, _ = img.shape
    wrist = hand_landmarks.landmark[0]

    for lm in hand_landmarks.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

    cv2.putText(
        img,
        label,
        (int(wrist.x * w) - 20, int(wrist.y * h) - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3
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
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def ensure_window_ready():
    global WINDOW_INITIALIZED
    if WINDOW_INITIALIZED:
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    WINDOW_INITIALIZED = True


ensure_window_ready()


def draw_top_bar(img, action, sample_idx, total_samples, message, bar_color=(40, 40, 40)):
    h, w, _ = img.shape
    draw_text_box(img, (0, 0), (w, 85), fill_color=(255, 255, 255), alpha=0.58)
    action_text = get_action_display_name(action)

    draw_text_unicode(img, f"Action: {action_text} ({sample_idx}/{total_samples})", (10, 8), font_size=28, text_color=(0, 0, 0))
    draw_text_unicode(img, message, (10, 42), font_size=24, text_color=(0, 0, 0))


def draw_progress_bar(img, current, total):
    h, w, _ = img.shape
    progress = int((current / total) * w)
    cv2.rectangle(img, (0, h - 12), (progress, h), (0, 255, 0), -1)


def draw_center_text(img, title, subtitle="", remain_text="", bar_color=(40, 40, 40)):
    h, w, _ = img.shape

    draw_text_box(img, (0, 0), (w, 95), fill_color=(255, 255, 255), alpha=0.58)
    draw_text_unicode(img, "CALIBRATION GUIDE", (10, 8), font_size=30, text_color=(0, 0, 0))
    draw_text_unicode(img, title, (40, 180), font_size=34, text_color=(0, 0, 0))

    if subtitle:
        draw_text_unicode(img, subtitle, (40, 235), font_size=28, text_color=(0, 0, 0))

    if remain_text:
        draw_text_unicode(img, remain_text, (220, 320), font_size=44, text_color=(0, 0, 0))

    draw_text_unicode(img, "Q : Quit", (10, h - 36), font_size=24, text_color=(0, 0, 0))


def show_countdown(action, sample_num, total_samples, seconds=3):
    start_time = time.time()

    while True:
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        elapsed = time.time() - start_time
        remain = seconds - int(elapsed)

        if remain <= 0:
            break

        draw_top_bar(
            img,
            action,
            sample_num,
            total_samples,
            "준비 자세를 유지하세요",
            bar_color=(40, 40, 40)
        )

        cv2.putText(
            img,
            str(remain),
            (280, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            (0, 255, 255),
            6
        )

        if results.multi_hand_landmarks and results.multi_handedness:
            for res_hand, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                if SWAP_HAND_LABELS:
                    label = 'Right' if label == 'Left' else 'Left'

                if label == 'Left':
                    draw_custom_landmarks(img, res_hand, (255, 0, 0), "L")
                else:
                    draw_custom_landmarks(img, res_hand, (0, 0, 255), "R")

        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(1)

        if key in [ord('q'), ord('Q')]:
            shutdown_and_exit()


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

            if results.multi_hand_landmarks and results.multi_handedness:
                for res_hand, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    if SWAP_HAND_LABELS:
                        label = 'Right' if label == 'Left' else 'Left'

                    if label == 'Left':
                        draw_custom_landmarks(img, res_hand, (255, 0, 0), "L")
                    else:
                        draw_custom_landmarks(img, res_hand, (0, 0, 255), "R")

            cv2.imshow(WINDOW_NAME, img)
            key = cv2.waitKey(1)

            if key in [ord('q'), ord('Q')]:
                shutdown_and_exit()

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

        elapsed = time.time() - start
        if elapsed > 10:
            print("[Sensor] Timeout: 센서 패킷이 안 들어왔습니다.")
            return False

        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        draw_center_text(
            img,
            "센서 준비 확인 중",
            "잠시만 기다려주세요",
            "",
            bar_color=(40, 40, 40)
        )
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key in [ord('q'), ord('Q')]:
            shutdown_and_exit()


def draw_review_screen(action, sample_num, total_samples):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)
    action_text = get_action_display_name(action)

    draw_text_unicode(img, "Save(Y) / Retry(R) / Discard(N) / Quit(Q)", (40, 150), font_size=30, text_color=(255, 255, 255))
    draw_text_unicode(img, f"{action_text} ({sample_num}/{total_samples})", (150, 225), font_size=34, text_color=(255, 255, 255))
    draw_text_unicode(img, "A/D action move is available only in standby mode", (55, 300), font_size=24, text_color=(220, 220, 220))

    cv2.imshow(WINDOW_NAME, img)


# =========================
# 종료 함수
# =========================
def shutdown_and_exit():
    try:
        serial_reader.close()
    except:
        pass

    try:
        csv_file.close()
    except:
        pass

    cap.release()
    cv2.destroyAllWindows()
    sys.exit()


# =========================
# 시리얼 시작
# =========================
print("현재 감지된 COM 포트:")
list_serial_ports()

serial_reader = SerialReader(SERIAL_PORT, SERIAL_BAUD)
serial_reader.start()

if serial_reader.connection_error is not None:
    print("")
    print("시리얼 연결 실패 가능성:")
    print("1) Arduino IDE 시리얼 모니터/플로터가 열려 있음")
    print("2) 다른 파이썬 코드가 같은 COM 포트를 사용 중")
    print("3) SERIAL_PORT 번호가 실제 포트와 다름")
    shutdown_and_exit()

# 시작 캘리브레이션 안내 화면
if not show_startup_calibration_guide():
    if serial_reader.connection_error is not None:
        print("")
        print("시리얼 연결 실패 가능성:")
        print("1) Arduino IDE 시리얼 모니터/플로터가 열려 있음")
        print("2) 다른 파이썬 코드가 같은 COM 포트를 사용 중")
        print("3) SERIAL_PORT 번호가 실제 포트와 다름")
    shutdown_and_exit()

# 실제 센서 패킷 준비 확인
if not show_wait_sensor_screen():
    if serial_reader.connection_error is not None:
        print("")
        print("시리얼 연결 실패 가능성:")
        print("1) Arduino IDE 시리얼 모니터/플로터가 열려 있음")
        print("2) 다른 파이썬 코드가 같은 COM 포트를 사용 중")
        print("3) SERIAL_PORT 번호가 실제 포트와 다름")
    shutdown_and_exit()


# =========================
# 메인 수집 루프
# =========================
current_action_idx = 0

while True:
    current_action = ACTIONS[current_action_idx]
    current_count = get_sample_count(current_action)

    display_sample_num = current_count + 1

    # -------------------------
    # 대기 모드
    # -------------------------
    while True:
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        msg = "A/D Move  |  S Start  |  Q Quit"

        draw_top_bar(
            img,
            current_action,
            display_sample_num,
            display_sample_num,
            msg,
            bar_color=(40, 40, 40)
        )

        if results.multi_hand_landmarks and results.multi_handedness:
            for res_hand, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                if SWAP_HAND_LABELS:
                    label = 'Right' if label == 'Left' else 'Left'

                if label == 'Left':
                    draw_custom_landmarks(img, res_hand, (255, 0, 0), "L")
                else:
                    draw_custom_landmarks(img, res_hand, (0, 0, 255), "R")

        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(1)

        if key in [ord('a'), ord('A')]:
            current_action_idx = (current_action_idx - 1) % len(ACTIONS)
            break

        elif key in [ord('d'), ord('D')]:
            current_action_idx = (current_action_idx + 1) % len(ACTIONS)
            break

        elif key in [ord('s'), ord('S')]:
            if current_count >= SAMPLES_PER_ACTION:
                print(
                    f"[Info] {get_action_display_name(current_action)} 은(는) "
                    f"목표 개수 {SAMPLES_PER_ACTION}개를 이미 채웠습니다."
                )
                continue

            _, _, packet_count, _ = serial_reader.get_latest()
            if packet_count == 0:
                print("[Warn] 센서 패킷이 아직 안 들어옵니다.")
                continue

            record_action = current_action
            record_sample_idx = get_next_sample_idx(record_action)
            break

        elif key in [ord('q'), ord('Q')]:
            shutdown_and_exit()

    if current_action != ACTIONS[current_action_idx]:
        continue

    # -------------------------
    # 카운트다운
    # -------------------------
    show_countdown(record_action, record_sample_idx + 1, record_sample_idx + 1, seconds=3)

    # -------------------------
    # 녹화 모드
    # -------------------------
    print(
        f"Recording {get_action_display_name(record_action)} "
        f"({record_sample_idx + 1})..."
    )

    frame_records = []

    prev_left = np.zeros(63, dtype=np.float32)
    prev_right = np.zeros(63, dtype=np.float32)
    prev_left_center = None
    prev_right_center = None
    left_missing_count = 0
    right_missing_count = 0
    overlap_freeze_count = 0
    left_recent_motion_px = 0.0
    right_recent_motion_px = 0.0

    while len(frame_records) < SEQ_LENGTH:
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        pc_time_ms = int(time.time() * 1000)
        frame_host_time_ms = time.perf_counter() * 1000.0
        sensor_time_ms, sensor_vec, packet_count, sensor_host_time_ms = serial_reader.get_aligned_packet(frame_host_time_ms)
        sensor_host_delta_ms = frame_host_time_ms - sensor_host_time_ms

        left_data = np.zeros(63, dtype=np.float32)
        right_data = np.zeros(63, dtype=np.float32)
        left_seen = False
        right_seen = False
        zero_vision_reason = None

        draw_top_bar(
            img,
            record_action,
            record_sample_idx + 1,
            record_sample_idx + 1,
            f"RECORDING... {len(frame_records) + 1}/{SEQ_LENGTH}",
            bar_color=(0, 0, 150)
        )

        candidates = make_hand_candidates(results, img.shape[1], img.shape[0])
        overlap_now = hands_are_overlapping(candidates)
        frame_ambiguous = False

        overlap_freeze_applied = False
        left_missing_freeze = False
        right_missing_freeze = False
        if overlap_now:
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
                zero_vision_reason = 'overlap'
        else:
            overlap_freeze_count = 0
            assigned, frame_ambiguous = assign_hand_slots(candidates, prev_left_center, prev_right_center)
            if frame_ambiguous:
                zero_vision_reason = 'ambiguous_slot'

        left_candidate = assigned['left']
        right_candidate = assigned['right']

        if frame_ambiguous:
            warn_text = 'VISION AMBIGUOUS - ZERO FRAME'
            if zero_vision_reason == 'overlap':
                warn_text = 'HAND OVERLAP - ZERO FRAME'
            cv2.putText(img, warn_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        elif overlap_freeze_applied:
            cv2.putText(img, 'SHORT OVERLAP - HOLD PREV VISION', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        if left_candidate is not None:
            left_data = left_candidate['data']
            left_seen = True
            draw_custom_landmarks(img, left_candidate['res_hand'], (255, 0, 0), "L")
        elif overlap_freeze_applied:
            left_data = prev_left.copy()

        if right_candidate is not None:
            right_data = right_candidate['data']
            right_seen = True
            draw_custom_landmarks(img, right_candidate['res_hand'], (0, 0, 255), "R")
        elif overlap_freeze_applied:
            right_data = prev_right.copy()

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
            elif left_missing_count > MISSING_HOLD_FRAMES:
                prev_left = np.zeros(63, dtype=np.float32)
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
            elif right_missing_count > MISSING_HOLD_FRAMES:
                prev_right = np.zeros(63, dtype=np.float32)
                prev_right_center = None
                right_recent_motion_px = 0.0

        vision_vec = np.concatenate([left_data, right_data]).astype(np.float32)
        left_zero = not np.any(np.abs(left_data) > 1e-6)
        right_zero = not np.any(np.abs(right_data) > 1e-6)
        if zero_vision_reason is None:
            if left_zero and right_zero:
                zero_vision_reason = 'both_missing'
            elif left_zero:
                zero_vision_reason = 'missing_left'
            elif right_zero:
                zero_vision_reason = 'missing_right'

        left_held = bool(overlap_freeze_applied or left_missing_freeze)
        right_held = bool(overlap_freeze_applied or right_missing_freeze)
        flag_vec = np.array(
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
        sensor_model_vec = np.concatenate([sensor_vec, flag_vec]).astype(np.float32)
        fused_vec = np.concatenate([vision_vec, sensor_model_vec]).astype(np.float32)

        frame_records.append({
            'pc_time_ms': pc_time_ms,
            'frame_host_time_ms': float(frame_host_time_ms),
            'sensor_host_time_ms': float(sensor_host_time_ms),
            'sensor_host_delta_ms': float(sensor_host_delta_ms),
            'sensor_time_ms': int(sensor_time_ms),
            'frame_idx': len(frame_records),
            'fused_vec': fused_vec,
            'left_zero': bool(left_zero),
            'right_zero': bool(right_zero),
            'flag_vec': flag_vec,
            'zero_reason': zero_vision_reason,
        })

        draw_progress_bar(img, len(frame_records), SEQ_LENGTH)
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key in [ord('q'), ord('Q')]:
            shutdown_and_exit()

    # -------------------------
    # 검수 모드
    # -------------------------
    while True:
        draw_review_screen(record_action, record_sample_idx + 1, record_sample_idx + 1)
        key = cv2.waitKey(1)

        if key in [ord('y'), ord('Y')]:
            seq_array = np.array(
                [item['fused_vec'] for item in frame_records],
                dtype=np.float32
            )   # (60, 152)
            quality_summary = compute_alignment_quality(frame_records)
            zero_quality = compute_zero_frame_quality(frame_records)

            pc_times = np.array(
                [item['pc_time_ms'] for item in frame_records],
                dtype=np.int64
            )
            frame_host_times = np.array(
                [item['frame_host_time_ms'] for item in frame_records],
                dtype=np.float64
            )
            sensor_host_times = np.array(
                [item['sensor_host_time_ms'] for item in frame_records],
                dtype=np.float64
            )
            sensor_host_deltas = np.array(
                [item['sensor_host_delta_ms'] for item in frame_records],
                dtype=np.float64
            )

            sensor_times = np.array(
                [item['sensor_time_ms'] for item in frame_records],
                dtype=np.int64
            )

            frame_indices = np.array(
                [item['frame_idx'] for item in frame_records],
                dtype=np.int32
            )

            action_dir = get_action_dir(record_action)

            npy_path = os.path.join(action_dir, f"{record_action}_{record_sample_idx:04d}.npy")
            meta_path = os.path.join(action_dir, f"{record_action}_{record_sample_idx:04d}_meta.npz")
            quality_path = os.path.join(action_dir, f"{record_action}_{record_sample_idx:04d}_quality.json")

            np.save(npy_path, seq_array)

            np.savez(
                meta_path,
                pc_time_ms=pc_times,
                frame_host_time_ms=frame_host_times,
                sensor_host_time_ms=sensor_host_times,
                sensor_host_delta_ms=sensor_host_deltas,
                sensor_time_ms=sensor_times,
                frame_idx=frame_indices,
                collect_session_id=np.array(COLLECT_SESSION_ID),
                align_delta_mean_ms=np.array(quality_summary['delta_mean_ms'], dtype=np.float32),
                align_delta_std_ms=np.array(quality_summary['delta_std_ms'], dtype=np.float32),
                align_delta_max_ms=np.array(quality_summary['delta_max_ms'], dtype=np.float32),
                align_delta_p95_ms=np.array(quality_summary['delta_p95_ms'], dtype=np.float32),
                align_warn_over_33ms_frames=np.array(quality_summary['warn_over_33ms_frames'], dtype=np.int32),
                align_bad_over_50ms_frames=np.array(quality_summary['bad_over_50ms_frames'], dtype=np.int32),
                vision_any_zero_frames=np.array(zero_quality['any_zero_frames'], dtype=np.int32),
                vision_both_zero_frames=np.array(zero_quality['both_zero_frames'], dtype=np.int32),
                frame_flag_names=np.array(FRAME_FLAG_KEYS, dtype='<U32'),
            )

            quality_payload = {
                'collect_session_id': COLLECT_SESSION_ID,
                'action': record_action,
                'sample_idx': int(record_sample_idx),
                'sequence_length': int(SEQ_LENGTH),
                'alignment_quality': quality_summary,
                'vision_zero_quality': zero_quality,
                'frame_flag_names': FRAME_FLAG_KEYS,
            }
            with open(quality_path, 'w', encoding='utf-8') as f:
                json.dump(quality_payload, f, ensure_ascii=False, indent=2)

            for item in frame_records:
                row = [
                    COLLECT_SESSION_ID,
                    item['pc_time_ms'],
                    item['frame_host_time_ms'],
                    item['sensor_host_time_ms'],
                    item['sensor_host_delta_ms'],
                    item['sensor_time_ms'],
                    record_action,
                    record_sample_idx,
                    item['frame_idx']
                ]
                row.extend(item['fused_vec'].tolist())
                csv_writer.writerow(row)

            csv_file.flush()

            print(f"✅ 저장 완료: {npy_path}")
            print(f"✅ 메타 저장 완료: {meta_path}")
            print(f"[Quality] {quality_path}")
            print(
                f"[VisionZero] any={zero_quality['any_zero_frames']}/{SEQ_LENGTH} "
                f"both={zero_quality['both_zero_frames']}/{SEQ_LENGTH} "
                f"first_any={zero_quality['first_any_zero_frame']} "
                f"first_both={zero_quality['first_both_zero_frame']} "
                f"reasons={zero_quality['reason_counts']}"
            )
            break

        elif key in [ord('r'), ord('R')]:
            print("🔄 현재 샘플을 다시 촬영합니다.")
            break

        elif key in [ord('n'), ord('N')]:
            print("🗑 현재 샘플을 폐기하고 대기 모드로 돌아갑니다.")
            break

        elif key in [ord('q'), ord('Q')]:
            shutdown_and_exit()

    if key in [ord('r'), ord('R')]:
        continue
