import argparse
import json
import os
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
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"labels file not found under {model_dir}")


def find_model_path(model_dir: Path) -> Path:
    for name in ["fusion_lstm_best.keras", "best_lstm.keras", "lstm_final.keras"]:
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"model file not found under {model_dir}")


def find_scaler_path(model_dir: Path) -> Path:
    for name in ["fusion_scaler.npz", "vision_scaler.npz"]:
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"scaler file not found under {model_dir}")


def load_groups(path: Path):
    raw = load_json(path)
    action_to_group = {}
    for group_key, payload in raw["groups"].items():
        for action in payload["actions"]:
            action_to_group[action] = {
                "group_key": group_key,
                "display_name": payload["display_name"],
            }
    return raw, action_to_group


def build_quality_path(sample_path: str) -> Path:
    p = Path(sample_path)
    return p.with_name(f"{p.stem}_quality.json")


def read_quality(sample_path: str):
    quality_path = build_quality_path(sample_path)
    if not quality_path.exists():
        return {
            "zero_any": 0,
            "zero_both": 0,
            "overlap": 0,
            "ambiguous": 0,
            "is_hard": False,
        }

    payload = load_json(quality_path)
    vision = payload.get("vision_zero_quality", {})
    reasons = vision.get("reason_counts", {}) or {}
    zero_any = int(vision.get("any_zero_frames", 0))
    zero_both = int(vision.get("both_zero_frames", 0))
    overlap = int(reasons.get("overlap", 0))
    ambiguous = int(reasons.get("ambiguous_slot", 0))
    is_hard = zero_any >= 3 or zero_both >= 1 or overlap > 0 or ambiguous > 0
    return {
        "zero_any": zero_any,
        "zero_both": zero_both,
        "overlap": overlap,
        "ambiguous": ambiguous,
        "is_hard": is_hard,
    }


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

    features = np.concatenate(
        [mean_rel, std_rel, range_rel, end_rel, peak_abs_rel, lr_pair_diff],
        axis=0,
    ).astype(np.float32)

    if features.shape[0] != flex_posture_dim:
        raise ValueError(f"Expected flex posture dim {flex_posture_dim}, got {features.shape[0]}")
    return features


def prepare_fusion_inputs(X_full, config, scaler_data):
    active_raw_indices = list(config.get("sensor_raw_indices", list(range(config.get("sensor_raw_dim", RAW_SENSOR_DIM)))))
    active_flag_indices = list(config.get("sensor_flag_indices", list(range(config.get("sensor_flag_dim", SENSOR_FLAG_DIM)))))
    flex_posture_dim = int(config.get("flex_posture_dim", 0)) if bool(config.get("use_flex_posture", False)) else 0
    flex_baseline_frames = int(config.get("flex_baseline_frames", 5))

    vision_mean = scaler_data["vision_mean"].astype(np.float32)
    vision_scale = scaler_data["vision_scale"].astype(np.float32)
    sensor_mean = scaler_data["sensor_mean"].astype(np.float32)
    sensor_scale = scaler_data["sensor_scale"].astype(np.float32)
    vision_scale = np.where(np.abs(vision_scale) < 1e-8, 1.0, vision_scale).astype(np.float32)
    sensor_scale = np.where(np.abs(sensor_scale) < 1e-8, 1.0, sensor_scale).astype(np.float32)

    flex_mean = scaler_data["flex_posture_mean"].astype(np.float32) if "flex_posture_mean" in scaler_data.files else np.zeros((0,), dtype=np.float32)
    flex_scale = scaler_data["flex_posture_scale"].astype(np.float32) if "flex_posture_scale" in scaler_data.files else np.ones((0,), dtype=np.float32)
    flex_scale = np.where(np.abs(flex_scale) < 1e-8, 1.0, flex_scale).astype(np.float32)

    X_vision = X_full[:, :, :VISION_DIM].astype(np.float32)
    X_sensor_full = X_full[:, :, VISION_DIM:].astype(np.float32)
    X_sensor_raw_full = X_sensor_full[:, :, :RAW_SENSOR_DIM]
    X_sensor_flags_full = X_sensor_full[:, :, RAW_SENSOR_DIM:RAW_SENSOR_DIM + SENSOR_FLAG_DIM]

    selected_parts = []
    if active_raw_indices:
        selected_parts.append(X_sensor_raw_full[:, :, active_raw_indices])
    if active_flag_indices:
        selected_parts.append(X_sensor_flags_full[:, :, active_flag_indices])
    X_sensor = np.concatenate(selected_parts, axis=2).astype(np.float32) if selected_parts else np.zeros((len(X_full), SEQ_LEN, 0), dtype=np.float32)

    X_vision_scaled = np.zeros_like(X_vision, dtype=np.float32)
    for i in range(len(X_vision)):
        nonzero_mask = np.any(np.abs(X_vision[i]) > 1e-6, axis=1)
        if np.any(nonzero_mask):
            X_vision_scaled[i, nonzero_mask] = ((X_vision[i, nonzero_mask] - vision_mean) / vision_scale).astype(np.float32)

    X_sensor_scaled = ((X_sensor - sensor_mean) / sensor_scale).astype(np.float32) if X_sensor.shape[2] > 0 else X_sensor

    if flex_posture_dim > 0:
        flex_features = []
        for i in range(len(X_sensor_raw_full)):
            feat = extract_flex_posture_features(X_sensor_raw_full[i], flex_baseline_frames, flex_posture_dim)
            feat = ((feat - flex_mean) / flex_scale).astype(np.float32)
            flex_features.append(feat)
        X_flex = np.stack(flex_features, axis=0).astype(np.float32)
        return [X_vision_scaled, X_sensor_scaled, X_flex]

    return [X_vision_scaled, X_sensor_scaled]


def prepare_vision_inputs(X_full, scaler_data):
    X_vision = X_full[:, :, :VISION_DIM].astype(np.float32)
    mean = scaler_data["mean"].astype(np.float32)
    scale = scaler_data["scale"].astype(np.float32)
    scale = np.where(np.abs(scale) < 1e-8, 1.0, scale).astype(np.float32)
    X_2d = X_vision.reshape(-1, VISION_DIM)
    X_scaled = ((X_2d - mean) / scale).reshape(-1, SEQ_LEN, VISION_DIM).astype(np.float32)
    return X_scaled


def compute_group_metrics(y_true, y_pred, labels, action_to_group):
    group_to_indices = {}
    for idx, action in enumerate(labels):
        group_key = action_to_group[action]["group_key"]
        group_to_indices.setdefault(group_key, []).append(idx)

    results = {}
    worst_group_accuracy = None
    for group_key, label_indices in group_to_indices.items():
        sample_mask = np.isin(y_true, np.array(label_indices, dtype=np.int32))
        y_true_group = y_true[sample_mask]
        y_pred_group = y_pred[sample_mask]
        if len(y_true_group) == 0:
            continue
        group_accuracy = float(accuracy_score(y_true_group, y_pred_group))
        group_macro_f1 = float(f1_score(y_true_group, y_pred_group, labels=label_indices, average="macro", zero_division=0))
        intra_group_confusions = 0
        for true_idx, pred_idx in zip(y_true_group, y_pred_group):
            if true_idx != pred_idx and pred_idx in label_indices:
                intra_group_confusions += 1
        intra_group_confusion_rate = float(intra_group_confusions / len(y_true_group))
        display_name = action_to_group[labels[label_indices[0]]]["display_name"]
        results[group_key] = {
            "display_name": display_name,
            "sample_count": int(len(y_true_group)),
            "accuracy": group_accuracy,
            "macro_f1": group_macro_f1,
            "intra_group_confusion_rate": intra_group_confusion_rate,
        }
        if worst_group_accuracy is None or group_accuracy < worst_group_accuracy:
            worst_group_accuracy = group_accuracy

    return results, float(worst_group_accuracy if worst_group_accuracy is not None else 0.0)


def summarize_model(model_dir: Path, group_map_path: Path):
    config = load_json(model_dir / "config.json")
    labels = load_json(find_labels_path(model_dir))
    sample_sources = load_json(model_dir / "sample_sources.json")
    split_indices = np.load(model_dir / "split_indices.npz")
    test_indices = split_indices["test"].astype(np.int32)

    action_group_payload, action_to_group = load_groups(group_map_path)

    X_test_full = []
    y_test = []
    quality_rows = []
    for idx in test_indices:
        item = sample_sources[int(idx)]
        arr = np.load(item["path"]).astype(np.float32)
        X_test_full.append(arr)
        y_test.append(labels.index(item["action"]))
        quality_rows.append(read_quality(item["path"]))

    X_test_full = np.stack(X_test_full, axis=0).astype(np.float32)
    y_test = np.array(y_test, dtype=np.int32)

    scaler_data = np.load(find_scaler_path(model_dir))
    model = tf.keras.models.load_model(find_model_path(model_dir))

    input_format = config.get("input_format", "")
    if input_format == "vision_only_lstm":
        model_inputs = prepare_vision_inputs(X_test_full, scaler_data)
        probs = model.predict(model_inputs, verbose=0)
    else:
        model_inputs = prepare_fusion_inputs(X_test_full, config, scaler_data)
        probs = model.predict(model_inputs, verbose=0)

    preds = np.argmax(probs, axis=1).astype(np.int32)
    sorted_probs = np.sort(probs, axis=1)
    margins = (sorted_probs[:, -1] - sorted_probs[:, -2]).astype(np.float32)

    overall_accuracy = float(accuracy_score(y_test, preds))
    overall_macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))

    group_metrics, worst_group_accuracy = compute_group_metrics(y_test, preds, labels, action_to_group)

    hard_mask = np.array([row["is_hard"] for row in quality_rows], dtype=bool)
    hard_accuracy = None
    hard_macro_f1 = None
    if np.any(hard_mask):
        hard_accuracy = float(accuracy_score(y_test[hard_mask], preds[hard_mask]))
        hard_macro_f1 = float(f1_score(y_test[hard_mask], preds[hard_mask], average="macro", zero_division=0))

    summary = {
        "model_dir": str(model_dir),
        "input_format": input_format,
        "overall": {
            "test_sample_count": int(len(y_test)),
            "accuracy": overall_accuracy,
            "macro_f1": overall_macro_f1,
            "low_margin_rate_lt_0_10": float(np.mean(margins < 0.10)),
            "low_margin_rate_lt_0_20": float(np.mean(margins < 0.20)),
            "mean_margin": float(np.mean(margins)),
        },
        "hard_case": {
            "sample_count": int(np.sum(hard_mask)),
            "accuracy": hard_accuracy,
            "macro_f1": hard_macro_f1,
            "criteria": {
                "zero_any_gte": 3,
                "zero_both_gte": 1,
                "overlap_gt": 0,
                "ambiguous_gt": 0,
            },
        },
        "similar_groups": group_metrics,
        "worst_group_accuracy": worst_group_accuracy,
        "split_metadata": {
            "strategy": config.get("split_strategy"),
            "group_count": config.get("split_group_count"),
            "session_group_count": config.get("split_session_group_count"),
        },
        "group_definition": action_group_payload["groups"],
    }

    out_json = model_dir / "realistic_offline_eval.json"
    out_md = model_dir / "realistic_offline_eval.md"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("# Realistic Offline Evaluation")
    lines.append("")
    lines.append(f"- Model: `{model_dir}`")
    lines.append(f"- Input format: `{input_format}`")
    lines.append(f"- Test Accuracy: `{overall_accuracy:.4f}`")
    lines.append(f"- Test Macro F1: `{overall_macro_f1:.4f}`")
    lines.append(f"- Hard-case Accuracy: `{hard_accuracy:.4f}`" if hard_accuracy is not None else "- Hard-case Accuracy: `N/A`")
    lines.append(f"- Hard-case Macro F1: `{hard_macro_f1:.4f}`" if hard_macro_f1 is not None else "- Hard-case Macro F1: `N/A`")
    lines.append(f"- Low-margin rate < 0.10: `{summary['overall']['low_margin_rate_lt_0_10']:.4f}`")
    lines.append(f"- Low-margin rate < 0.20: `{summary['overall']['low_margin_rate_lt_0_20']:.4f}`")
    lines.append(f"- Worst-group Accuracy: `{worst_group_accuracy:.4f}`")
    lines.append("")
    lines.append("## Similar Groups")
    for group_key, payload in group_metrics.items():
        lines.append(
            f"- {payload['display_name']} (`{group_key}`): "
            f"acc={payload['accuracy']:.4f}, macro_f1={payload['macro_f1']:.4f}, "
            f"intra_conf={payload['intra_group_confusion_rate']:.4f}, n={payload['sample_count']}"
        )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--group-map", default="similar_sign_groups.json")
    args = parser.parse_args()

    summary = summarize_model(Path(args.model_dir), Path(args.group_map))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
