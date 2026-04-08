import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import tensorflow as tf


SEQ_LEN = 60
VISION_DIM = 126
RAW_SENSOR_DIM = 26
SENSOR_FLAG_DIM = 6
FLEX_SENSOR_DIM = 8


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_labels_path(model_dir: Path) -> Path:
    for name in ["labels.json", "class_names.json"]:
        p = model_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"labels file not found under {model_dir}")


def find_model_path(model_dir: Path) -> Path:
    for name in ["fusion_lstm_best.keras", "best_lstm.keras", "lstm_final.keras"]:
        p = model_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"model file not found under {model_dir}")


def find_scaler_path(model_dir: Path) -> Path:
    for name in ["fusion_scaler.npz", "vision_scaler.npz"]:
        p = model_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"scaler file not found under {model_dir}")


def extract_flex_posture_features(seq_sensor_raw_full, flex_baseline_frames, flex_posture_dim):
    if flex_posture_dim <= 0:
        return np.zeros((0,), dtype=np.float32)

    flex_seq = seq_sensor_raw_full[:, :FLEX_SENSOR_DIM].astype(np.float32)
    baseline_frames = max(1, min(flex_baseline_frames, flex_seq.shape[0]))
    base = np.mean(flex_seq[:baseline_frames, :], axis=0, keepdims=True)
    rel = flex_seq - base

    mean_rel = np.mean(rel, axis=0)
    std_rel = np.std(rel, axis=0)
    range_rel = np.max(rel, axis=0) - np.min(rel, axis=0)
    end_rel = np.mean(rel[-baseline_frames:, :], axis=0)
    peak_abs_rel = np.max(np.abs(rel), axis=0)
    lr_pair_diff = end_rel[: (FLEX_SENSOR_DIM // 2)] - end_rel[(FLEX_SENSOR_DIM // 2):]
    return np.concatenate([mean_rel, std_rel, range_rel, end_rel, peak_abs_rel, lr_pair_diff], axis=0).astype(np.float32)


def prepare_fusion_inputs(X_full, config, scaler_data):
    active_raw_indices = list(config.get("sensor_raw_indices", list(range(config.get("sensor_raw_dim", RAW_SENSOR_DIM)))))
    active_flag_indices = list(config.get("sensor_flag_indices", list(range(config.get("sensor_flag_dim", SENSOR_FLAG_DIM)))))
    use_flex = bool(config.get("use_flex_posture", False))
    flex_posture_dim = int(config.get("flex_posture_dim", 0)) if use_flex else 0
    flex_baseline_frames = int(config.get("flex_baseline_frames", 5))

    vision_mean = scaler_data["vision_mean"].astype(np.float32)
    vision_scale = np.where(np.abs(scaler_data["vision_scale"].astype(np.float32)) < 1e-8, 1.0, scaler_data["vision_scale"].astype(np.float32))
    sensor_mean = scaler_data["sensor_mean"].astype(np.float32)
    sensor_scale = np.where(np.abs(scaler_data["sensor_scale"].astype(np.float32)) < 1e-8, 1.0, scaler_data["sensor_scale"].astype(np.float32))
    flex_mean = scaler_data["flex_posture_mean"].astype(np.float32) if "flex_posture_mean" in scaler_data.files else np.zeros((0,), dtype=np.float32)
    flex_scale = scaler_data["flex_posture_scale"].astype(np.float32) if "flex_posture_scale" in scaler_data.files else np.ones((0,), dtype=np.float32)
    flex_scale = np.where(np.abs(flex_scale) < 1e-8, 1.0, flex_scale)

    X_vision = X_full[:, :, :VISION_DIM].astype(np.float32)
    X_sensor_full = X_full[:, :, VISION_DIM:].astype(np.float32)
    X_sensor_raw_full = X_sensor_full[:, :, :RAW_SENSOR_DIM]
    X_sensor_flags_full = X_sensor_full[:, :, RAW_SENSOR_DIM:RAW_SENSOR_DIM + SENSOR_FLAG_DIM]

    parts = []
    if active_raw_indices:
        parts.append(X_sensor_raw_full[:, :, active_raw_indices])
    if active_flag_indices:
        parts.append(X_sensor_flags_full[:, :, active_flag_indices])
    X_sensor = np.concatenate(parts, axis=2).astype(np.float32) if parts else np.zeros((len(X_full), SEQ_LEN, 0), dtype=np.float32)

    X_vision_scaled = np.zeros_like(X_vision, dtype=np.float32)
    for i in range(len(X_vision)):
        nonzero_mask = np.any(np.abs(X_vision[i]) > 1e-6, axis=1)
        if np.any(nonzero_mask):
            X_vision_scaled[i, nonzero_mask] = ((X_vision[i, nonzero_mask] - vision_mean) / vision_scale).astype(np.float32)

    X_sensor_scaled = ((X_sensor - sensor_mean) / sensor_scale).astype(np.float32) if X_sensor.shape[2] > 0 else X_sensor

    if flex_posture_dim > 0:
        X_flex = []
        for i in range(len(X_sensor_raw_full)):
            feat = extract_flex_posture_features(X_sensor_raw_full[i], flex_baseline_frames, flex_posture_dim)
            feat = ((feat - flex_mean) / flex_scale).astype(np.float32)
            X_flex.append(feat)
        return [X_vision_scaled, X_sensor_scaled, np.stack(X_flex, axis=0).astype(np.float32)]

    return [X_vision_scaled, X_sensor_scaled]


def prepare_vision_inputs(X_full, scaler_data):
    X_vision = X_full[:, :, :VISION_DIM].astype(np.float32)
    mean = scaler_data["mean"].astype(np.float32)
    scale = np.where(np.abs(scaler_data["scale"].astype(np.float32)) < 1e-8, 1.0, scaler_data["scale"].astype(np.float32))
    X_2d = X_vision.reshape(-1, VISION_DIM)
    X_scaled = ((X_2d - mean) / scale).reshape(-1, SEQ_LEN, VISION_DIM).astype(np.float32)
    return X_scaled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-dir", default="live_eval_dataset/all")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    dataset_dir = Path(args.dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    manifest = load_json(manifest_path)

    config = load_json(model_dir / "config.json")
    labels = load_json(find_labels_path(model_dir))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    X = []
    y = []
    rows = []
    for item in manifest:
        gt_label = item["gt_label"]
        if gt_label not in label_to_idx:
            continue
        arr = np.load(item["target_npy"] if Path(item["target_npy"]).exists() else item["source_npy"]).astype(np.float32)
        if arr.shape[0] != SEQ_LEN:
            continue
        X.append(arr)
        y.append(label_to_idx[gt_label])
        rows.append(item)

    X = np.stack(X, axis=0).astype(np.float32)
    y = np.array(y, dtype=np.int32)

    scaler_data = np.load(find_scaler_path(model_dir))
    model = tf.keras.models.load_model(find_model_path(model_dir))

    if config.get("input_format") == "vision_only_lstm":
        model_inputs = prepare_vision_inputs(X, scaler_data)
        probs = model.predict(model_inputs, verbose=0)
    else:
        model_inputs = prepare_fusion_inputs(X, config, scaler_data)
        probs = model.predict(model_inputs, verbose=0)

    preds = np.argmax(probs, axis=1).astype(np.int32)
    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    acc = float(accuracy_score(y, preds))
    macro_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
    cm = confusion_matrix(y, preds, labels=list(range(len(labels))))

    report_rows = []
    for item, true_idx, pred_idx, margin, prob in zip(rows, y, preds, margins, probs):
        report_rows.append({
            "gt_label": labels[true_idx],
            "pred_label": labels[pred_idx],
            "correct": bool(true_idx == pred_idx),
            "margin": float(margin),
            "pred_conf": float(prob[pred_idx]),
            "source_model_name": item["source_model_name"],
            "source_json": item["source_json"],
        })

    out = {
        "model_dir": str(model_dir),
        "dataset_dir": str(dataset_dir),
        "sample_count": int(len(y)),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "low_margin_rate_lt_0_10": float(np.mean(margins < 0.10)),
        "low_margin_rate_lt_0_20": float(np.mean(margins < 0.20)),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "per_sample": report_rows,
    }

    out_path = model_dir / "live_eval_dataset_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "model_dir": str(model_dir),
        "dataset_dir": str(dataset_dir),
        "sample_count": int(len(y)),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "low_margin_rate_lt_0_10": float(np.mean(margins < 0.10)),
        "low_margin_rate_lt_0_20": float(np.mean(margins < 0.20)),
        "report": str(out_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
