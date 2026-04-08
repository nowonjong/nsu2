import os
import sys
import json
import time

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont


ACTIONS = [
    "gandanhada",
    "sada",
    "gamsahamnida",
    "joesonghada",
    "banggeum",
    "billida",
    "mannada",
    "byeongyeong",
    "jamkkan",
    "oraenman",
    "gakkapda",
    "more",
    "naeil",
    "eoje",
    "teukbyeol",
    "byeollo",
    "jalhada",
    "annyeonghaseyo",
]

ACTION_DISPLAY_NAMES = {
    "gandanhada": "간단하다",
    "sada": "사다",
    "gamsahamnida": "감사합니다",
    "joesonghada": "죄송하다",
    "banggeum": "방금",
    "billida": "빌리다",
    "mannada": "만나다",
    "byeongyeong": "변경",
}

ACTION_DISPLAY_NAMES.update({
    "jamkkan": "잠깐",
    "oraenman": "오랜만",
    "gakkapda": "가깝다",
    "more": "모레",
    "naeil": "내일",
    "eoje": "어제",
    "teukbyeol": "특별",
    "byeollo": "별로",
    "jalhada": "잘하다",
    "annyeonghaseyo": "안녕하세요",
})

SEQ_LENGTH = 60
DATA_PATH = os.getenv("NSU_VISION_COLLECT_DIR", "bare_hand_vision_18words_v1")
COLLECT_SESSION_ID = time.strftime("%Y%m%d_%H%M%S")

SWAP_HAND_LABELS = False
HANDEDNESS_SCORE_TH = 0.55
OVERLAP_IOU_THRESHOLD = 0.24
OVERLAP_CENTER_DIST_PX = 60.0
SLOT_AMBIGUOUS_MARGIN = 35.0
MISSING_HOLD_FRAMES = 6

WINDOW_NAME = "Collect Vision Data"
UI_FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
_FONT_CACHE = {}

STARTUP_GUIDE_STAGES = [
    ("촬영 위치를 맞추고 손을 준비하세요", "손이 화면 안에 잘 보이게 해주세요", 2.0, (40, 40, 40)),
    ("검지부터 새끼손가락까지 쭉 펴주세요", "시작 손모양을 확인합니다", 2.0, (60, 70, 40)),
    ("검지부터 새끼손가락까지 구부려주세요", "손모양 변화를 확인합니다", 2.0, (70, 40, 40)),
]

os.makedirs(DATA_PATH, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    sys.exit()


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


def draw_top_bar(img, action_display, sample_idx, total_samples, message, color=(255, 255, 255)):
    h, w, _ = img.shape
    draw_text_box(img, (0, 0), (w, 86), fill_color=color, alpha=0.58)
    draw_text_unicode(img, f"단어: {action_display} ({sample_idx}/{total_samples})", (10, 12), font_size=28, text_color=(0, 0, 0))
    draw_text_unicode(img, message, (10, 46), font_size=24, text_color=(20, 20, 20))


def draw_progress_bar(img, current, total):
    h, w, _ = img.shape
    progress = int((current / total) * w)
    cv2.rectangle(img, (0, h - 10), (progress, h), (0, 255, 0), -1)


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
        side_hint = None
        if score >= HANDEDNESS_SCORE_TH:
            side_hint = mp_label_to_person_side(label, frame_flipped=True)
        if SWAP_HAND_LABELS:
            label = "Right" if label == "Left" else "Left"
        candidates.append({
            "data": get_hand_data(res_hand),
            "center": get_hand_center_px(res_hand, frame_w, frame_h),
            "bbox": get_hand_bbox_px(res_hand, frame_w, frame_h),
            "mp_label": label,
            "mp_score": float(score),
            "person_side_hint": side_hint,
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
    hint = candidate["person_side_hint"]
    if prev_left_center is None and prev_right_center is None and hint in ("left", "right"):
        return hint, False
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
    h, w, _ = img.shape
    wrist = hand_landmarks.landmark[0]
    for lm in hand_landmarks.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 4, color, cv2.FILLED)
    cv2.putText(img, label, (int(wrist.x * w) - 15, int(wrist.y * h) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def compute_zero_frame_quality(frame_records):
    any_zero = [item["frame_idx"] for item in frame_records if item["left_zero"] or item["right_zero"]]
    both_zero = [item["frame_idx"] for item in frame_records if item["left_zero"] and item["right_zero"]]
    reason_counts = {}
    for item in frame_records:
        reason = item.get("zero_reason")
        if reason is None:
            continue
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    frame_count = len(frame_records)
    return {
        "any_zero_frames": int(len(any_zero)),
        "both_zero_frames": int(len(both_zero)),
        "first_any_zero_frame": int(any_zero[0]) if any_zero else None,
        "first_both_zero_frame": int(both_zero[0]) if both_zero else None,
        "any_zero_ratio": float(len(any_zero) / frame_count) if frame_count else 0.0,
        "both_zero_ratio": float(len(both_zero) / frame_count) if frame_count else 0.0,
        "reason_counts": reason_counts,
    }


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
            cv2.imshow(WINDOW_NAME, img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
    return True


def show_countdown(action_display, sample_num, total_samples, seconds=3):
    start_time = time.time()
    while True:
        ret, img = cap.read()
        if not ret:
            continue
        img = cv2.flip(img, 1)
        remain = seconds - int(time.time() - start_time)
        if remain <= 0:
            break
        draw_top_bar(img, action_display, sample_num, total_samples, "준비 자세를 유지하세요")
        cv2.putText(img, str(remain), (280, 240), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 6)
        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()


def draw_review_screen(action_display, sample_num, total_samples):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (55, 55, 55)
    draw_text_unicode(img, "저장(Y) / 다시촬영(R) / 종료(Q)", (40, 150), font_size=32, text_color=(255, 255, 255))
    draw_text_unicode(img, f"{action_display} ({sample_num}/{total_samples})", (180, 240), font_size=34, text_color=(255, 255, 255))
    return img


if not show_startup_guide():
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit

sample_counts = {}
for action in ACTIONS:
    existing = len([f for f in os.listdir(DATA_PATH) if f.startswith(f"{action}_") and f.endswith(".npy")])
    sample_counts[action] = existing

action_index = 0

while True:
    action = ACTIONS[action_index]
    action_display = ACTION_DISPLAY_NAMES.get(action, action)
    sample = sample_counts[action]

    prev_left = np.zeros(63, dtype=np.float32)
    prev_right = np.zeros(63, dtype=np.float32)
    prev_left_center = None
    prev_right_center = None
    left_missing_count = 0
    right_missing_count = 0
    data = []
    frame_records = []

    while True:
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw_top_bar(img, action_display, sample + 1, sample + 1, "S 시작 | A 이전 | D 다음 | Q 종료")

        candidates = make_hand_candidates(results, img.shape[1], img.shape[0])
        assigned, frame_ambiguous = assign_hand_slots(candidates, prev_left_center, prev_right_center)
        left_candidate = assigned["left"]
        right_candidate = assigned["right"]
        if left_candidate is not None:
            draw_custom_landmarks(img, left_candidate["res_hand"], (255, 0, 0), "L")
        if right_candidate is not None:
            draw_custom_landmarks(img, right_candidate["res_hand"], (0, 0, 255), "R")

        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            break
        if key == ord("a"):
            action_index = (action_index - 1) % len(ACTIONS)
            data = None
            break
        if key == ord("d"):
            action_index = (action_index + 1) % len(ACTIONS)
            data = None
            break
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

    if data is None:
        continue

    show_countdown(action_display, sample + 1, sample + 1, seconds=3)
    print(f"Recording {action_display} ({sample + 1})...")

    while len(data) < SEQ_LENGTH:
            ret, img = cap.read()
            if not ret:
                continue

            img = cv2.flip(img, 1)
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            left_data = np.zeros(63, dtype=np.float32)
            right_data = np.zeros(63, dtype=np.float32)
            left_seen = False
            right_seen = False
            zero_reason = None

            candidates = make_hand_candidates(results, img.shape[1], img.shape[0])
            overlap_now = hands_are_overlapping(candidates)
            frame_ambiguous = False

            if overlap_now:
                assigned = {"left": None, "right": None}
                frame_ambiguous = True
                zero_reason = "overlap"
            else:
                assigned, frame_ambiguous = assign_hand_slots(candidates, prev_left_center, prev_right_center)
                if frame_ambiguous:
                    zero_reason = "ambiguous_slot"

            left_candidate = assigned["left"]
            right_candidate = assigned["right"]

            if frame_ambiguous:
                cv2.putText(img, "VISION AMBIGUOUS - ZERO FRAME", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            if left_candidate is not None:
                left_data = left_candidate["data"]
                left_seen = True
                draw_custom_landmarks(img, left_candidate["res_hand"], (255, 0, 0), "L")
            else:
                left_seen = False

            if right_candidate is not None:
                right_data = right_candidate["data"]
                right_seen = True
                draw_custom_landmarks(img, right_candidate["res_hand"], (0, 0, 255), "R")
            else:
                right_seen = False

            if left_candidate is not None:
                prev_left = left_data.copy()
                prev_left_center = left_candidate["center"].copy()
                left_missing_count = 0
            else:
                left_missing_count += 1
                if left_missing_count > MISSING_HOLD_FRAMES:
                    prev_left = np.zeros(63, dtype=np.float32)
                    prev_left_center = None

            if right_candidate is not None:
                prev_right = right_data.copy()
                prev_right_center = right_candidate["center"].copy()
                right_missing_count = 0
            else:
                right_missing_count += 1
                if right_missing_count > MISSING_HOLD_FRAMES:
                    prev_right = np.zeros(63, dtype=np.float32)
                    prev_right_center = None

            full_joint = np.concatenate([left_data, right_data]).astype(np.float32)
            data.append(full_joint)

            frame_idx = len(data)
            left_zero = not np.any(np.abs(left_data) > 1e-6)
            right_zero = not np.any(np.abs(right_data) > 1e-6)
            if zero_reason is None:
                if left_zero and right_zero:
                    zero_reason = "both_missing"
                elif left_zero:
                    zero_reason = "missing_left"
                elif right_zero:
                    zero_reason = "missing_right"

            frame_records.append({
                "frame_idx": frame_idx,
                "left_zero": left_zero,
                "right_zero": right_zero,
                "zero_reason": zero_reason,
            })

            draw_top_bar(img, action_display, sample + 1, sample + 1, f"녹화 중... {len(data)}/{SEQ_LENGTH}", color=(255, 255, 255))
            draw_progress_bar(img, len(data), SEQ_LENGTH)
            cv2.imshow(WINDOW_NAME, img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                sys.exit()

    while True:
        img = draw_review_screen(action_display, sample + 1, sample + 1)
        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("y"):
            save_file = os.path.join(DATA_PATH, f"{action}_{sample:04d}.npy")
            meta_path = os.path.join(DATA_PATH, f"{action}_{sample:04d}_meta.npz")
            quality_path = os.path.join(DATA_PATH, f"{action}_{sample:04d}_quality.json")

            np.save(save_file, np.array(data, dtype=np.float32))
            np.savez(meta_path, collect_session_id=np.array(COLLECT_SESSION_ID))
            with open(quality_path, "w", encoding="utf-8") as f:
                json.dump({"collect_session_id": COLLECT_SESSION_ID, "vision_zero_quality": compute_zero_frame_quality(frame_records)}, f, ensure_ascii=False, indent=2)
            print(f"저장 완료: {save_file}")
            sample_counts[action] += 1
            break
        if key == ord("r"):
            print("현재 샘플을 버리고 다시 촬영합니다.")
            break
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

cap.release()
cv2.destroyAllWindows()
