import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import json
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont


MODEL_DIR = os.getenv(
    "NSU_VISION_MODEL_DIR",
    r"C:\SignProject\experiments\glove_vision\2026-04-07_glove_vision_18w_clean_v1",
)
MODEL_PATH = os.path.join(MODEL_DIR, "best_lstm.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "vision_scaler.npz")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
DEBUG_CAPTURE_DIR = os.getenv("NSU_VISION_DEBUG_CAPTURE_DIR", r"experiments\live_eval\vision_run_debug_captures")

SEQ_LEN = 60
FEATURE_DIM = 126
CAM_WIDTH = 640
CAM_HEIGHT = 480

DRAW_LANDMARKS = False
MODEL_COMPLEXITY = 0
MAX_NUM_HANDS = 2
TRIGGER_DELAY_SEC = 2.0

HAND_SCORE_THRESHOLD = 0.55
OVERLAP_IOU_THRESHOLD = 0.24
OVERLAP_CENTER_DIST_PX = 60.0
SLOT_AMBIGUOUS_MARGIN = 35.0
MISSING_HOLD_FRAMES = 6
NO_HAND_RESET_FRAMES = 18
MIN_HAND_FRAMES_FOR_WORD = 20
MOTION_THRESHOLD = 0.040
LOW_CONF_THRESHOLD = 0.55
LOW_MARGIN_THRESHOLD = 0.08

STARTUP_GUIDE_STAGES = [
    ("손이 화면 안에 잘 보이게 준비하세요", "시작 자세를 맞춰주세요", 2.0, (40, 40, 40)),
    ("검지부터 새끼손가락까지 쭉 펴주세요", "손모양을 확인합니다", 2.0, (60, 70, 40)),
    ("검지부터 새끼손가락까지 구부려주세요", "손모양 변화를 확인합니다", 2.0, (70, 40, 40)),
]

KOR_MAP = {
    "none": "대기",
    "more": "모레",
    "naeil": "내일",
    "eoje": "어제",
    "teukbyeol": "특별",
    "byeollo": "별로",
    "jamkkan": "잠깐",
    "oraenman": "오랜만",
    "gakkapda": "가깝다",
    "jalhada": "잘하다",
    "annyeonghaseyo": "안녕하세요",
    "mannada": "만나다",
    "byeongyeonghada": "변경하다",
    "banggeum": "방금",
    "billida": "빌리다",
    "gandanhada": "간단하다",
    "sada": "사다",
    "gamsahamnida": "감사합니다",
    "joesonghada": "죄송하다",
}

UI_FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
_FONT_CACHE = {}

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("Detected GPUs:", gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except Exception as e:
        print("GPU memory growth setting failed:", e)
else:
    print("GPU not found. Running on CPU.")

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        runtime_config = json.load(f)
    SEQ_LEN = int(runtime_config.get("seq_length", SEQ_LEN))
    FEATURE_DIM = int(runtime_config.get("feature_dim", FEATURE_DIM))

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    ACTIONS = json.load(f)
scaler_data = np.load(SCALER_PATH)
scaler_mean = scaler_data["mean"].astype(np.float32)
scaler_scale = scaler_data["scale"].astype(np.float32)
scaler_scale = np.where(np.abs(scaler_scale) < 1e-8, 1.0, scaler_scale)

print("로드된 클래스:", ACTIONS)
if "none" not in ACTIONS:
    print("[경고] class_names.json에 'none' 클래스가 없습니다.")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


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


def draw_center_text(img, title, subtitle="", remain_text="", bar_color=(255, 255, 255)):
    h, w, _ = img.shape
    draw_text_box(img, (0, 0), (w, 92), fill_color=bar_color, alpha=0.58)
    draw_text_unicode(img, "수집 준비 안내", (16, 18), font_size=30, text_color=(0, 0, 0))
    draw_text_unicode(img, title, (40, 200), font_size=34, text_color=(255, 255, 255))
    if subtitle:
        draw_text_unicode(img, subtitle, (40, 246), font_size=28, text_color=(235, 235, 235))
    if remain_text:
        draw_text_unicode(img, remain_text, (220, 332), font_size=40, text_color=(0, 255, 255))
    draw_text_box(img, (10, h - 46), (132, h - 8), fill_color=(255, 255, 255), alpha=0.48)
    draw_text_unicode(img, "Q : 종료", (18, h - 40), font_size=24, text_color=(0, 0, 0))


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
    return np.array([float(np.mean(xs) * frame_w), float(np.mean(ys) * frame_h)], dtype=np.float32)


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
    iou = bbox_iou(c0["bbox"], c1["bbox"])
    center_dist = float(np.linalg.norm(c0["center"] - c1["center"]))
    return (iou >= OVERLAP_IOU_THRESHOLD) or (center_dist <= OVERLAP_CENTER_DIST_PX)


def mp_label_to_person_side(label, frame_flipped=True):
    if label not in ("Left", "Right"):
        return None
    if frame_flipped:
        return label.lower()
    return "right" if label == "Left" else "left"


def make_hand_candidates(results, frame_w, frame_h):
    candidates = []
    if not (results.multi_hand_landmarks and results.multi_handedness):
        return candidates
    for res_hand, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = handedness.classification[0].label
        score = handedness.classification[0].score
        if score < HAND_SCORE_THRESHOLD:
            continue
        candidates.append({
            "data": get_hand_data(res_hand),
            "center": get_hand_center_px(res_hand, frame_w, frame_h),
            "bbox": get_hand_bbox_px(res_hand, frame_w, frame_h),
            "person_side_hint": mp_label_to_person_side(label, frame_flipped=True),
            "res_hand": res_hand,
        })
    return candidates


def slot_cost(candidate, slot_name, prev_center):
    cost = 0.0
    hint = candidate["person_side_hint"]
    if hint is not None and hint != slot_name:
        cost += 120.0
    if prev_center is not None:
        cost += float(np.linalg.norm(candidate["center"] - prev_center))
    else:
        cost += 20.0
    return cost


def choose_slot_for_one_hand(candidate, prev_left_center, prev_right_center):
    left_cost = slot_cost(candidate, "left", prev_left_center)
    right_cost = slot_cost(candidate, "right", prev_right_center)
    if abs(left_cost - right_cost) < SLOT_AMBIGUOUS_MARGIN:
        return None, True
    if left_cost <= right_cost:
        return "left", False
    return "right", False


def assign_hand_slots(candidates, prev_left_center, prev_right_center):
    assigned = {"left": None, "right": None}
    if len(candidates) == 0:
        return assigned, False
    if len(candidates) == 1:
        slot_name, ambiguous = choose_slot_for_one_hand(candidates[0], prev_left_center, prev_right_center)
        if slot_name is None:
            return assigned, ambiguous
        assigned[slot_name] = candidates[0]
        return assigned, False

    cands = candidates[:2]
    case1 = slot_cost(cands[0], "left", prev_left_center) + slot_cost(cands[1], "right", prev_right_center)
    case2 = slot_cost(cands[1], "left", prev_left_center) + slot_cost(cands[0], "right", prev_right_center)
    if abs(case1 - case2) < SLOT_AMBIGUOUS_MARGIN:
        return assigned, True
    if case1 <= case2:
        assigned["left"] = cands[0]
        assigned["right"] = cands[1]
    else:
        assigned["left"] = cands[1]
        assigned["right"] = cands[0]
    return assigned, False


def draw_custom_landmarks(img, hand_landmarks, color, label):
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
        2,
    )


def standardize_sequence(seq):
    return ((seq - scaler_mean) / scaler_scale).astype(np.float32)


def get_motion_score(seq):
    if len(seq) < 2:
        return 0.0
    diffs = np.diff(seq, axis=0)
    return float(np.mean(np.linalg.norm(diffs, axis=1)))


@tf.function
def infer(input_tensor):
    return model(input_tensor, training=False)


_ = infer(tf.convert_to_tensor(np.zeros((1, SEQ_LEN, FEATURE_DIM), dtype=np.float32)))

sequence = deque(maxlen=SEQ_LEN)
hand_presence_history = deque(maxlen=SEQ_LEN)
prev_left = np.zeros(63, dtype=np.float32)
prev_right = np.zeros(63, dtype=np.float32)
prev_left_center = None
prev_right_center = None
left_missing_count = 0
right_missing_count = 0
no_hand_run = 0

stable_label = "none"
stable_conf = 0.0
current_pred_label = ""
current_pred_conf = 0.0
motion_score = 0.0
last_margin = 0.0
gt_label_index = -1

mode = "WAIT"
countdown_start = None
fps_prev_time = time.time()
fps_smooth = 0.0

os.makedirs(DEBUG_CAPTURE_DIR, exist_ok=True)


def get_current_gt_label():
    if 0 <= gt_label_index < len(ACTIONS):
        return ACTIONS[gt_label_index]
    return None


def save_debug_capture(seq_array, raw_label, raw_conf, final_label, final_conf, top2_label, top2_conf, margin, motion):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"{timestamp}_{raw_label}"
    npy_path = os.path.join(DEBUG_CAPTURE_DIR, f"{stem}.npy")
    json_path = os.path.join(DEBUG_CAPTURE_DIR, f"{stem}.json")
    gt_label = get_current_gt_label()

    np.save(npy_path, seq_array.astype(np.float32))

    meta = {
        "timestamp": timestamp,
        "model_dir": MODEL_DIR,
        "model_path": MODEL_PATH,
        "seq_len": int(seq_array.shape[0]),
        "feature_dim": int(seq_array.shape[1]),
        "raw_label": raw_label,
        "raw_conf": float(raw_conf),
        "label": final_label,
        "conf": float(final_conf),
        "top2_label": top2_label,
        "top2_conf": float(top2_conf),
        "margin": float(margin),
        "motion": float(motion),
        "gt_label": gt_label,
        "gt_matches_prediction": (gt_label == final_label) if gt_label is not None else None,
        "actions": ACTIONS,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DebugCapture] saved: {npy_path}")
    print(f"[DebugCaptureMeta] saved: {json_path}")


def reset_recording_buffers():
    global left_missing_count, right_missing_count, no_hand_run
    global current_pred_label, current_pred_conf, motion_score, last_margin
    global prev_left_center, prev_right_center

    sequence.clear()
    hand_presence_history.clear()
    prev_left[:] = 0
    prev_right[:] = 0
    prev_left_center = None
    prev_right_center = None
    left_missing_count = 0
    right_missing_count = 0
    no_hand_run = 0
    current_pred_label = ""
    current_pred_conf = 0.0
    motion_score = 0.0
    last_margin = 0.0


def reset_all_state():
    global mode, countdown_start, stable_label, stable_conf
    reset_recording_buffers()
    mode = "WAIT"
    countdown_start = None
    stable_label = "none"
    stable_conf = 0.0


def run_one_shot_prediction():
    global stable_label, stable_conf, current_pred_label, current_pred_conf, motion_score, last_margin

    if len(sequence) < SEQ_LEN:
        stable_label = "none"
        stable_conf = 0.0
        current_pred_label = ""
        current_pred_conf = 0.0
        motion_score = 0.0
        last_margin = 0.0
        return

    seq_array = np.array(sequence, dtype=np.float32)
    motion_score = get_motion_score(seq_array)
    hand_count = int(np.sum(np.any(np.abs(seq_array) > 1e-6, axis=1)))

    if hand_count < MIN_HAND_FRAMES_FOR_WORD or motion_score < MOTION_THRESHOLD:
        stable_label = "none"
        stable_conf = 1.0
        current_pred_label = "none"
        current_pred_conf = 1.0
        last_margin = 0.0
        save_debug_capture(seq_array, "none", 1.0, "none", 1.0, "none", 1.0, 0.0, motion_score)
        print(
            f"[Result] stable=none ({KOR_MAP.get('none', 'none')}) | "
            f"hand_count={hand_count} | motion={motion_score:.3f}"
        )
        return

    x = np.expand_dims(standardize_sequence(seq_array), axis=0)
    y_prob = infer(tf.convert_to_tensor(x, dtype=tf.float32)).numpy()[0]
    sorted_idx = np.argsort(y_prob)
    top1_idx = int(sorted_idx[-1])
    top2_idx = int(sorted_idx[-2])
    top1_label = ACTIONS[top1_idx]
    top1_conf = float(y_prob[top1_idx])
    top2_label = ACTIONS[top2_idx]
    top2_conf = float(y_prob[top2_idx])
    margin = top1_conf - top2_conf

    current_pred_label = top1_label
    current_pred_conf = top1_conf

    final_label = top1_label
    final_conf = top1_conf
    if top1_conf < LOW_CONF_THRESHOLD and margin < LOW_MARGIN_THRESHOLD:
        final_label = "none"
        final_conf = top1_conf

    stable_label = final_label
    stable_conf = final_conf
    last_margin = margin

    save_debug_capture(seq_array, top1_label, top1_conf, final_label, final_conf, top2_label, top2_conf, margin, motion_score)
    print(
        f"[Result] stable={stable_label} ({KOR_MAP.get(stable_label, stable_label)}) | "
        f"conf={stable_conf:.3f} | top2={top2_label} ({top2_conf:.3f}) | "
        f"margin={margin:.3f} | motion={motion_score:.3f}"
    )


def show_startup_guide():
    for title, subtitle, duration_sec, bar_color in STARTUP_GUIDE_STAGES:
        stage_start = time.time()
        while True:
            ret, img = cap.read()
            if not ret:
                continue
            img = cv2.flip(img, 1)
            remain = duration_sec - (time.time() - stage_start)
            if remain <= 0:
                break
            draw_center_text(img, title, subtitle, f"{int(np.ceil(remain))}초", bar_color=bar_color)
            cv2.imshow("Vision Real-time Sign Recognition", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
    return True


if not show_startup_guide():
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
    display_img = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    left_data = np.zeros(63, dtype=np.float32)
    right_data = np.zeros(63, dtype=np.float32)
    left_seen = False
    right_seen = False

    candidates = make_hand_candidates(results, CAM_WIDTH, CAM_HEIGHT)
    overlap_now = hands_are_overlapping(candidates)
    frame_ambiguous = False
    if overlap_now:
        assigned = {"left": None, "right": None}
        frame_ambiguous = True
    else:
        assigned, frame_ambiguous = assign_hand_slots(candidates, prev_left_center, prev_right_center)

    left_candidate = assigned["left"]
    right_candidate = assigned["right"]

    if left_candidate is not None:
        left_data = left_candidate["data"]
        left_seen = True
        draw_custom_landmarks(display_img, left_candidate["res_hand"], (255, 0, 0), "L")
    if right_candidate is not None:
        right_data = right_candidate["data"]
        right_seen = True
        draw_custom_landmarks(display_img, right_candidate["res_hand"], (0, 0, 255), "R")

    any_hand_seen = left_seen or right_seen

    if mode == "RECORDING":
        if left_seen:
            prev_left = left_data.copy()
            prev_left_center = left_candidate["center"].copy()
            left_missing_count = 0
        else:
            left_missing_count += 1
            if left_missing_count > MISSING_HOLD_FRAMES:
                prev_left = np.zeros(63, dtype=np.float32)
                prev_left_center = None

        if right_seen:
            prev_right = right_data.copy()
            prev_right_center = right_candidate["center"].copy()
            right_missing_count = 0
        else:
            right_missing_count += 1
            if right_missing_count > MISSING_HOLD_FRAMES:
                prev_right = np.zeros(63, dtype=np.float32)
                prev_right_center = None

        if any_hand_seen:
            no_hand_run = 0
            hand_presence_history.append(1)
        else:
            no_hand_run += 1
            hand_presence_history.append(0)

        sequence.append(np.concatenate([left_data, right_data]).astype(np.float32))

        if no_hand_run >= NO_HAND_RESET_FRAMES:
            reset_all_state()
        elif len(sequence) >= SEQ_LEN:
            run_one_shot_prediction()
            mode = "WAIT"
            countdown_start = None

    elif mode == "COUNTDOWN":
        remain = TRIGGER_DELAY_SEC - (time.time() - countdown_start)
        if remain <= 0:
            reset_recording_buffers()
            mode = "RECORDING"

    now = time.time()
    instant_fps = 1.0 / max(now - fps_prev_time, 1e-6)
    fps_prev_time = now
    fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps if fps_smooth > 0 else instant_fps

    h, w, _ = display_img.shape
    draw_text_box(display_img, (0, 0), (w, 138), fill_color=(255, 255, 255), alpha=0.58)
    mode_kor = {"WAIT": "대기", "COUNTDOWN": "카운트다운", "RECORDING": "녹화중"}.get(mode, mode)
    top1_kor = KOR_MAP.get(current_pred_label, current_pred_label) if current_pred_label else ""
    stable_kor = KOR_MAP.get(stable_label, stable_label)
    gt_label = get_current_gt_label()
    gt_kor = KOR_MAP.get(gt_label, gt_label) if gt_label else "미지정"

    draw_text_unicode(display_img, f"FPS: {fps_smooth:.1f}", (10, 12), font_size=28, text_color=(0, 0, 0))
    draw_text_unicode(display_img, f"상태: {mode_kor}", (165, 12), font_size=28, text_color=(0, 0, 0))
    draw_text_unicode(display_img, f"시퀀스: {len(sequence)}/{SEQ_LEN}", (380, 12), font_size=28, text_color=(0, 0, 0))
    draw_text_unicode(display_img, f"동작량: {motion_score:.3f}", (10, 48), font_size=24, text_color=(20, 20, 20))
    draw_text_unicode(display_img, f"GT: {gt_kor}", (10, 82), font_size=24, text_color=(120, 60, 0))
    if current_pred_label:
        draw_text_unicode(
            display_img,
            f"예측 1순위: {top1_kor} ({current_pred_conf:.2f})",
            (230, 48),
            font_size=24,
            text_color=(0, 120, 160),
        )

    result_color = (0, 120, 0) if stable_label != "none" else (90, 90, 90)
    draw_text_unicode(display_img, f"결과: {stable_kor}", (260, 82), font_size=30, text_color=result_color)

    if mode == "WAIT":
        draw_text_unicode(display_img, "S 시작 | B 이전 GT | N 다음 GT | G GT해제 | Q 종료", (10, 112), font_size=22, text_color=(25, 25, 25))
    elif mode == "COUNTDOWN":
        remain_int = int(np.ceil(max(0.0, TRIGGER_DELAY_SEC - (time.time() - countdown_start))))
        draw_text_unicode(display_img, "준비 자세를 유지하세요", (10, 112), font_size=26, text_color=(0, 120, 160))
        cv2.putText(display_img, str(remain_int), (w // 2 - 30, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 6)
    elif mode == "RECORDING":
        draw_text_unicode(display_img, f"녹화 중... {len(sequence)}/{SEQ_LEN}", (10, 112), font_size=26, text_color=(0, 120, 160))

    progress = int((len(sequence) / SEQ_LEN) * w) if mode == "RECORDING" else 0
    cv2.rectangle(display_img, (0, h - 8), (progress, h), (0, 255, 0), -1)

    cv2.imshow("Vision Real-time Sign Recognition", display_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s") and mode == "WAIT":
        reset_recording_buffers()
        countdown_start = time.time()
        mode = "COUNTDOWN"
        print("시작 입력 감지. 2초 뒤 녹화를 시작합니다.")
    elif key == ord("c"):
        reset_all_state()
        print("상태를 초기화했습니다.")
    elif key == ord("d"):
        DRAW_LANDMARKS = not DRAW_LANDMARKS
        print(f"DRAW_LANDMARKS = {DRAW_LANDMARKS}")
    elif key == ord("n"):
        if ACTIONS:
            gt_label_index = (gt_label_index + 1) % len(ACTIONS)
            print(f"[GT] {get_current_gt_label()}")
    elif key == ord("b"):
        if ACTIONS:
            gt_label_index = len(ACTIONS) - 1 if gt_label_index < 0 else (gt_label_index - 1) % len(ACTIONS)
            print(f"[GT] {get_current_gt_label()}")
    elif key == ord("g"):
        gt_label_index = -1
        print("[GT] cleared")

cap.release()
cv2.destroyAllWindows()
