import os
import json
import random
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


ACTIONS = [
    "more",
    "naeil",
    "eoje",
    "teukbyeol",
    "byeollo",
    "jamkkan",
    "oraenman",
    "gakkapda",
    "jalhada",
    "annyeonghaseyo",
    "mannada",
    "byeongyeonghada",
    "banggeum",
    "billida",
    "gandanhada",
    "sada",
    "gamsahamnida",
    "joesonghada",
]

DATA_PATH = os.getenv("NSU_DATA_PATH", "dataset_fusion_hold_v4_flags")
OUTPUT_DIR = os.getenv("NSU_OUTPUT_DIR", "fusion_train_output_hold_v4_flags")

SEQ_LENGTH = 60
VISION_DIM = 126
RAW_SENSOR_DIM = 26
SENSOR_FLAG_DIM = 6
FLEX_SENSOR_DIM = 8
IMU_SENSOR_DIM = RAW_SENSOR_DIM - FLEX_SENSOR_DIM
SENSOR_MODE = os.getenv("NSU_SENSOR_MODE", "imu_flags").strip().lower()

if SENSOR_MODE == "all":
    ACTIVE_RAW_SENSOR_INDICES = list(range(RAW_SENSOR_DIM))
    ACTIVE_FLAG_INDICES = list(range(SENSOR_FLAG_DIM))
elif SENSOR_MODE == "imu_flags":
    ACTIVE_RAW_SENSOR_INDICES = list(range(FLEX_SENSOR_DIM, RAW_SENSOR_DIM))
    ACTIVE_FLAG_INDICES = list(range(SENSOR_FLAG_DIM))
elif SENSOR_MODE == "imu_only":
    ACTIVE_RAW_SENSOR_INDICES = list(range(FLEX_SENSOR_DIM, RAW_SENSOR_DIM))
    ACTIVE_FLAG_INDICES = []
elif SENSOR_MODE == "flex_flags":
    ACTIVE_RAW_SENSOR_INDICES = list(range(FLEX_SENSOR_DIM))
    ACTIVE_FLAG_INDICES = list(range(SENSOR_FLAG_DIM))
elif SENSOR_MODE == "flex_only":
    ACTIVE_RAW_SENSOR_INDICES = list(range(FLEX_SENSOR_DIM))
    ACTIVE_FLAG_INDICES = []
else:
    raise ValueError(
        "Unsupported NSU_SENSOR_MODE. "
        "Use one of: all, imu_flags, imu_only, flex_flags, flex_only"
    )

ACTIVE_RAW_SENSOR_DIM = len(ACTIVE_RAW_SENSOR_INDICES)
ACTIVE_SENSOR_FLAG_DIM = len(ACTIVE_FLAG_INDICES)
SENSOR_DIM = ACTIVE_RAW_SENSOR_DIM + ACTIVE_SENSOR_FLAG_DIM
FEATURE_DIM = VISION_DIM + SENSOR_DIM
RANDOM_SEED = int(os.environ.get("NSU_RANDOM_SEED", "42"))

TEST_RATIO = 0.15
VAL_RATIO = 0.15
MIN_SAMPLES_PER_CLASS = 5
LEGACY_GROUP_BLOCK_SIZE = 5

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
SENSOR_FLAG_NAMES = [
    "left_detected",
    "right_detected",
    "left_held",
    "right_held",
    "overlap_flag",
    "ambiguous_flag",
]
USE_FLEX_POSTURE = os.getenv("NSU_USE_FLEX_POSTURE", "1").strip().lower() not in {"0", "false", "no"}
FLEX_BASELINE_FRAMES = 5
FLEX_POSTURE_DIM = (FLEX_SENSOR_DIM * 5 + (FLEX_SENSOR_DIM // 2)) if USE_FLEX_POSTURE else 0
MODEL_VARIANT = os.getenv("NSU_MODEL_VARIANT", "stable_hybrid").strip().lower()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_min_required_samples_per_class():
    # Conservative guard so stratified train/val/test split has room to allocate each class.
    return max(MIN_SAMPLES_PER_CLASS, 3)


def infer_legacy_group_key(action, file_name):
    stem = os.path.splitext(file_name)[0]
    match = re.search(r"_(\d+)$", stem)
    if match:
        sample_idx = int(match.group(1))
        block_idx = sample_idx // LEGACY_GROUP_BLOCK_SIZE
        return f"legacy_block::{action}::{block_idx:04d}", "legacy_index_block"
    return f"legacy_file::{action}::{stem}", "legacy_file"


def read_collect_session_id(meta_path):
    if not os.path.exists(meta_path):
        return None

    try:
        with np.load(meta_path, allow_pickle=True) as meta:
            if "collect_session_id" not in meta:
                return None
            value = meta["collect_session_id"]
            if np.ndim(value) == 0:
                return str(value.item())
            flat = np.ravel(value)
            if flat.size == 0:
                return None
            return str(flat[0])
    except Exception:
        return None


def build_sample_group_info(file_path, action, file_name):
    stem = os.path.splitext(file_path)[0]
    meta_path = f"{stem}_meta.npz"
    collect_session_id = read_collect_session_id(meta_path)

    if collect_session_id:
        return {
            "meta_path": meta_path.replace("\\", "/"),
            "collect_session_id": collect_session_id,
            "group_key": f"session::{collect_session_id}",
            "group_source": "collect_session_id",
        }

    group_key, group_source = infer_legacy_group_key(action, file_name)
    return {
        "meta_path": meta_path.replace("\\", "/") if os.path.exists(meta_path) else None,
        "collect_session_id": None,
        "group_key": group_key,
        "group_source": group_source,
    }


def load_dataset():
    X = []
    y = []
    sample_sources = []
    class_counts = {}
    active_actions = []
    skipped_actions = {}

    for action in ACTIONS:
        action_dir = os.path.join(DATA_PATH, action)

        if not os.path.exists(action_dir):
            raise FileNotFoundError(f"Directory not found: {action_dir}")

        files = sorted(
            f for f in os.listdir(action_dir)
            if f.endswith(".npy")
        )
        class_counts[action] = len(files)

        if len(files) < get_min_required_samples_per_class():
            skipped_actions[action] = (
                f"too_few_samples({len(files)} < {get_min_required_samples_per_class()})"
            )
            continue

        label_idx = len(active_actions)
        active_actions.append(action)
        for file_name in files:
            file_path = os.path.join(action_dir, file_name)
            arr = np.load(file_path)
            group_info = build_sample_group_info(file_path, action, file_name)

            if arr.shape == (SEQ_LENGTH, VISION_DIM + RAW_SENSOR_DIM + SENSOR_FLAG_DIM):
                final_arr = arr
            elif arr.shape == (SEQ_LENGTH, VISION_DIM + RAW_SENSOR_DIM):
                # Backward compatibility for older fusion data without frame flags.
                flag_pad = np.zeros((SEQ_LENGTH, SENSOR_FLAG_DIM), dtype=np.float32)
                final_arr = np.concatenate([arr.astype(np.float32), flag_pad], axis=1)
            else:
                raise ValueError(
                    f"{file_path} shape must be {(SEQ_LENGTH, VISION_DIM + RAW_SENSOR_DIM + SENSOR_FLAG_DIM)} "
                    f"or {(SEQ_LENGTH, VISION_DIM + RAW_SENSOR_DIM)}, got {arr.shape}"
                )

            X.append(final_arr.astype(np.float32))
            y.append(label_idx)
            sample_sources.append(
                {
                    "action": action,
                    "file": file_name,
                    "path": file_path.replace("\\", "/"),
                    "meta_path": group_info["meta_path"],
                    "collect_session_id": group_info["collect_session_id"],
                    "group_key": group_info["group_key"],
                    "group_source": group_info["group_source"],
                }
            )

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    if len(active_actions) < 2:
        raise ValueError(
            "Not enough trainable classes after filtering. "
            f"Need at least 2 classes with >= {get_min_required_samples_per_class()} samples."
        )

    return X, y, class_counts, active_actions, skipped_actions, sample_sources


def split_dataset_random_stratified(X, y):
    indices = np.arange(len(X), dtype=np.int32)
    X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
        X,
        y,
        indices,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    val_relative_ratio = VAL_RATIO / (1.0 - TEST_RATIO)

    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train_val,
        y_train_val,
        idx_train_val,
        test_size=val_relative_ratio,
        random_state=RANDOM_SEED,
        stratify=y_train_val,
    )

    split_indices = {
        "train": idx_train.astype(np.int32),
        "val": idx_val.astype(np.int32),
        "test": idx_test.astype(np.int32),
    }
    split_meta = {
        "strategy": "random_stratified",
        "warning": "Random split may overestimate real-time performance when samples are temporally correlated.",
        "group_count": 0,
        "session_group_count": 0,
        "legacy_group_count": 0,
    }
    return X_train, X_val, X_test, y_train, y_val, y_test, split_indices, split_meta


def choose_best_group_fold(splits, y, target_size):
    best_pair = None
    best_score = None

    for train_idx, test_idx in splits:
        score = (
            abs(len(test_idx) - target_size),
            -len(np.unique(y[test_idx])),
            -len(np.unique(y[train_idx])),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_pair = (train_idx, test_idx)

    return best_pair


def collect_group_class_sets(y, sample_sources):
    group_to_classes = {}
    for label_idx, item in zip(y, sample_sources):
        group_key = item["group_key"]
        if group_key not in group_to_classes:
            group_to_classes[group_key] = set()
        group_to_classes[group_key].add(int(label_idx))
    return group_to_classes


def find_complete_eval_groups(y, sample_sources):
    required_classes = set(int(v) for v in np.unique(y))
    group_to_classes = collect_group_class_sets(y, sample_sources)
    complete_groups = sorted(
        group_key for group_key, class_set in group_to_classes.items()
        if class_set >= required_classes
    )
    incomplete_groups = sorted(
        group_key for group_key, class_set in group_to_classes.items()
        if class_set < required_classes
    )
    return required_classes, complete_groups, incomplete_groups


def split_dataset_grouped(X, y, sample_sources):
    if StratifiedGroupKFold is None:
        raise ValueError("StratifiedGroupKFold is unavailable in this scikit-learn version.")

    groups = np.array([item["group_key"] for item in sample_sources], dtype=object)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError(f"Need at least 3 unique groups, got {len(unique_groups)}.")

    required_classes, complete_eval_groups, incomplete_groups = find_complete_eval_groups(y, sample_sources)

    if len(complete_eval_groups) < 3:
        raise ValueError(
            "Need at least 3 complete session groups containing every class "
            f"for grouped val/test split, got {len(complete_eval_groups)} complete groups."
        )

    eval_mask = np.isin(groups, np.array(complete_eval_groups, dtype=object))
    forced_train_mask = ~eval_mask
    eval_indices = np.where(eval_mask)[0].astype(np.int32)
    forced_train_indices = np.where(forced_train_mask)[0].astype(np.int32)

    X_eval = X[eval_indices]
    y_eval = y[eval_indices]
    groups_eval = groups[eval_indices]

    outer_splits = min(5, len(np.unique(groups_eval)))
    if outer_splits < 2:
        raise ValueError("Need at least 2 group folds for test split.")

    dummy_X = np.zeros((len(y_eval), 1), dtype=np.float32)
    outer_cv = StratifiedGroupKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    outer_candidates = list(outer_cv.split(dummy_X, y_eval, groups_eval))
    eval_train_val_rel, eval_test_rel = choose_best_group_fold(
        outer_candidates,
        y_eval,
        target_size=max(1, int(round(len(y_eval) * TEST_RATIO))),
    )

    idx_train_val_eval = eval_indices[eval_train_val_rel].astype(np.int32)
    idx_test = eval_indices[eval_test_rel].astype(np.int32)

    y_train_val = y[idx_train_val_eval]
    groups_train_val = groups[idx_train_val_eval]
    unique_train_val_groups = np.unique(groups_train_val)
    inner_splits = min(5, len(unique_train_val_groups))
    if inner_splits < 2:
        raise ValueError("Need at least 2 remaining group folds for val split.")

    inner_cv = StratifiedGroupKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=RANDOM_SEED + 1,
    )
    inner_candidates = list(inner_cv.split(
        np.zeros((len(idx_train_val_eval), 1), dtype=np.float32),
        y_train_val,
        groups_train_val,
    ))
    train_rel, val_rel = choose_best_group_fold(
        inner_candidates,
        y_train_val,
        target_size=max(1, int(round(len(idx_train_val_eval) * (VAL_RATIO / (1.0 - TEST_RATIO))))),
    )

    idx_train = idx_train_val_eval[train_rel].astype(np.int32)
    idx_val = idx_train_val_eval[val_rel].astype(np.int32)
    idx_test = idx_test.astype(np.int32)

    if forced_train_indices.size > 0:
        idx_train = np.concatenate([idx_train, forced_train_indices]).astype(np.int32)
        idx_train = np.unique(idx_train).astype(np.int32)

    split_indices = {
        "train": idx_train,
        "val": idx_val,
        "test": idx_test,
    }
    session_groups = {
        item["group_key"] for item in sample_sources
        if item["group_source"] == "collect_session_id"
    }
    legacy_groups = {
        item["group_key"] for item in sample_sources
        if item["group_source"] != "collect_session_id"
    }
    split_meta = {
        "strategy": "group_session_stratified",
        "warning": (
            "Group/session split used to reduce leakage across temporally related samples. "
            "Legacy samples without session metadata fall back to coarse index blocks. "
            "Incomplete session groups are forced into train and excluded from val/test."
        ),
        "group_count": int(len(unique_groups)),
        "session_group_count": int(len(session_groups)),
        "legacy_group_count": int(len(legacy_groups)),
        "complete_eval_group_count": int(len(complete_eval_groups)),
        "incomplete_group_count": int(len(incomplete_groups)),
        "train_group_count": int(len(np.unique(groups[idx_train]))),
        "val_group_count": int(len(np.unique(groups[idx_val]))),
        "test_group_count": int(len(np.unique(groups[idx_test]))),
    }

    return (
        X[idx_train],
        X[idx_val],
        X[idx_test],
        y[idx_train],
        y[idx_val],
        y[idx_test],
        split_indices,
        split_meta,
    )


def split_dataset(X, y, sample_sources):
    try:
        return split_dataset_grouped(X, y, sample_sources)
    except ValueError as group_error:
        try:
            result = split_dataset_random_stratified(X, y)
            result[-1]["warning"] += f" Group split fallback reason: {group_error}"
            return result
        except ValueError as e:
            raise ValueError(
                "Dataset split failed. "
                "Check per-class sample counts, split ratios, and session grouping. "
                f"Group split error: {group_error}. "
                f"Random split error: {e}"
            ) from e


def split_modalities(X):
    X_vision = X[:, :, :VISION_DIM]
    X_sensor_full = X[:, :, VISION_DIM:]

    raw_sensor = X_sensor_full[:, :, :RAW_SENSOR_DIM]
    flag_sensor = X_sensor_full[:, :, RAW_SENSOR_DIM:]

    selected_parts = []
    if ACTIVE_RAW_SENSOR_INDICES:
        selected_parts.append(raw_sensor[:, :, ACTIVE_RAW_SENSOR_INDICES])
    if ACTIVE_FLAG_INDICES:
        selected_parts.append(flag_sensor[:, :, ACTIVE_FLAG_INDICES])

    if selected_parts:
        X_sensor = np.concatenate(selected_parts, axis=2)
    else:
        X_sensor = np.zeros((X.shape[0], X.shape[1], 0), dtype=np.float32)

    return X_vision.astype(np.float32), X_sensor.astype(np.float32)


def fit_scaler(X_train, feature_dim):
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, feature_dim)
    scaler.fit(X_train_2d)
    return scaler


def transform_with_scaler(X, scaler, feature_dim):
    orig_shape = X.shape
    X_2d = X.reshape(-1, feature_dim)
    X_scaled = scaler.transform(X_2d)
    return X_scaled.reshape(orig_shape).astype(np.float32)


def fit_vision_scaler_preserve_zero(X_train_vision):
    vision_scaler = StandardScaler()
    X_train_2d = X_train_vision.reshape(-1, VISION_DIM)
    nonzero_mask = np.any(np.abs(X_train_2d) > 1e-6, axis=1)

    if np.any(nonzero_mask):
        vision_scaler.fit(X_train_2d[nonzero_mask])
    else:
        vision_scaler.fit(X_train_2d)

    return vision_scaler


def transform_vision_with_preserved_zero(X_vision, vision_scaler):
    orig_shape = X_vision.shape
    X_vision_2d = X_vision.reshape(-1, VISION_DIM).astype(np.float32)
    nonzero_mask = np.any(np.abs(X_vision_2d) > 1e-6, axis=1)

    X_vision_scaled = np.zeros_like(X_vision_2d, dtype=np.float32)
    if np.any(nonzero_mask):
        X_vision_scaled[nonzero_mask] = vision_scaler.transform(
            X_vision_2d[nonzero_mask]
        ).astype(np.float32)

    return X_vision_scaled.reshape(orig_shape).astype(np.float32)


def fit_sensor_scaler_preserve_flags(X_train_sensor):
    raw_scaler = StandardScaler()

    if ACTIVE_RAW_SENSOR_DIM > 0:
        X_train_raw_2d = X_train_sensor[:, :, :ACTIVE_RAW_SENSOR_DIM].reshape(-1, ACTIVE_RAW_SENSOR_DIM)
        raw_scaler.fit(X_train_raw_2d)
        raw_mean = raw_scaler.mean_.astype(np.float32)
        raw_scale = raw_scaler.scale_.astype(np.float32)
    else:
        raw_mean = np.zeros((0,), dtype=np.float32)
        raw_scale = np.ones((0,), dtype=np.float32)

    sensor_mean = np.concatenate(
        [
            raw_mean,
            np.zeros(ACTIVE_SENSOR_FLAG_DIM, dtype=np.float32),
        ]
    )
    sensor_scale = np.concatenate(
        [
            raw_scale,
            np.ones(ACTIVE_SENSOR_FLAG_DIM, dtype=np.float32),
        ]
    )
    sensor_scale = np.where(np.abs(sensor_scale) < 1e-8, 1.0, sensor_scale).astype(np.float32)

    return {
        "raw_scaler": raw_scaler,
        "mean_": sensor_mean,
        "scale_": sensor_scale,
    }


def transform_sensor_with_preserved_flags(X_sensor, sensor_scaler):
    orig_shape = X_sensor.shape
    X_sensor_2d = X_sensor.reshape(-1, SENSOR_DIM).astype(np.float32)

    X_sensor_raw = X_sensor_2d[:, :ACTIVE_RAW_SENSOR_DIM]
    X_sensor_flags = X_sensor_2d[:, ACTIVE_RAW_SENSOR_DIM:]

    if ACTIVE_RAW_SENSOR_DIM > 0:
        X_sensor_raw_scaled = sensor_scaler["raw_scaler"].transform(X_sensor_raw).astype(np.float32)
    else:
        X_sensor_raw_scaled = np.zeros((X_sensor_2d.shape[0], 0), dtype=np.float32)

    X_sensor_scaled = np.concatenate([X_sensor_raw_scaled, X_sensor_flags], axis=1).astype(np.float32)
    return X_sensor_scaled.reshape(orig_shape).astype(np.float32)


def extract_flex_posture_features(X_full):
    if FLEX_POSTURE_DIM <= 0:
        return np.zeros((X_full.shape[0], 0), dtype=np.float32)

    flex_seq = X_full[:, :, VISION_DIM:VISION_DIM + FLEX_SENSOR_DIM].astype(np.float32)
    base = np.mean(flex_seq[:, :FLEX_BASELINE_FRAMES, :], axis=1, keepdims=True)
    rel = flex_seq - base

    mean_rel = np.mean(rel, axis=1)
    std_rel = np.std(rel, axis=1)
    range_rel = np.max(rel, axis=1) - np.min(rel, axis=1)
    end_rel = np.mean(rel[:, -FLEX_BASELINE_FRAMES:, :], axis=1)
    peak_abs_rel = np.max(np.abs(rel), axis=1)
    lr_pair_diff = end_rel[:, : (FLEX_SENSOR_DIM // 2)] - end_rel[:, (FLEX_SENSOR_DIM // 2) :]

    return np.concatenate(
        [mean_rel, std_rel, range_rel, end_rel, peak_abs_rel, lr_pair_diff],
        axis=1,
    ).astype(np.float32)


def fit_flex_posture_scaler(X_train_flex):
    scaler = StandardScaler()
    scaler.fit(X_train_flex.astype(np.float32))
    return scaler


def transform_flex_posture_features(X_flex, scaler):
    if X_flex.shape[1] == 0:
        return X_flex.astype(np.float32)
    return scaler.transform(X_flex.astype(np.float32)).astype(np.float32)


def build_model(num_classes):
    vision_input = layers.Input(
        shape=(SEQ_LENGTH, VISION_DIM),
        name="vision_input",
    )
    sensor_input = layers.Input(
        shape=(SEQ_LENGTH, SENSOR_DIM),
        name="sensor_input",
    )
    inputs = [vision_input, sensor_input]

    vision_branch = layers.LSTM(128, return_sequences=True)(vision_input)
    vision_branch = layers.Dropout(0.3)(vision_branch)
    vision_branch = layers.LSTM(64)(vision_branch)
    vision_branch = layers.Dropout(0.3)(vision_branch)

    sensor_branch = layers.LSTM(64, return_sequences=True)(sensor_input)
    sensor_branch = layers.Dropout(0.3)(sensor_branch)
    sensor_branch = layers.LSTM(32)(sensor_branch)
    sensor_branch = layers.Dropout(0.3)(sensor_branch)

    if MODEL_VARIANT == "input_only":
        fusion_parts = [vision_branch, sensor_branch]
        if FLEX_POSTURE_DIM > 0:
            flex_input = layers.Input(shape=(FLEX_POSTURE_DIM,), name="flex_posture_input")
            inputs.append(flex_input)
            flex_branch = layers.Dense(32, activation="relu", name="flex_posture_dense_1")(flex_input)
            flex_branch = layers.Dropout(0.2, name="flex_posture_dropout")(flex_branch)
            flex_branch = layers.Dense(32, activation="relu", name="flex_posture_dense_2")(flex_branch)
            fusion_parts.append(flex_branch)

        fused = layers.Concatenate(name="fusion_concat")(fusion_parts)
        fused = layers.Dense(96, activation="relu")(fused)
        fused = layers.Dropout(0.3)(fused)
        output = layers.Dense(num_classes, activation="softmax")(fused)

        model = models.Model(
            inputs=inputs,
            outputs=output,
            name="dual_input_lstm" if FLEX_POSTURE_DIM > 0 else "dual_input_no_flex_lstm",
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    if MODEL_VARIANT in {"modality_aware_correction", "modality_aware_rerank"}:
        sensor_proj = layers.Dense(64, activation="relu", name="sensor_project")(sensor_branch)
        fusion_parts = [vision_branch, sensor_proj]
        correction_modalities = [sensor_proj]

        if FLEX_POSTURE_DIM > 0:
            flex_input = layers.Input(shape=(FLEX_POSTURE_DIM,), name="flex_posture_input")
            inputs.append(flex_input)
            flex_branch = layers.Dense(32, activation="relu", name="flex_posture_dense_1")(flex_input)
            flex_branch = layers.Dropout(0.2, name="flex_posture_dropout")(flex_branch)
            flex_branch = layers.Dense(32, activation="relu", name="flex_posture_dense_2")(flex_branch)
            flex_proj = layers.Dense(64, activation="relu", name="flex_modality_project")(flex_branch)
            fusion_parts.append(flex_branch)
            correction_modalities.append(flex_proj)

        fused = layers.Concatenate(name="fusion_concat")(fusion_parts)
        fused = layers.Dense(96, activation="relu", name="fusion_hidden")(fused)
        fused = layers.Dropout(0.3, name="fusion_hidden_dropout")(fused)
        base_logits = layers.Dense(num_classes, name="base_logits")(fused)
        base_probs = layers.Activation("softmax", name="base_probs")(base_logits)

        modality_context = layers.Concatenate(
            name="modality_context",
        )([vision_branch] + correction_modalities)
        gate_logits = layers.Dense(
            len(correction_modalities),
            activation=None,
            bias_initializer=tf.keras.initializers.Constant(0.0),
            name="modality_gate_logits",
        )(modality_context)
        modality_weights = layers.Activation("softmax", name="modality_weights")(gate_logits)

        correction_terms = []
        sensor_delta_hidden = layers.Dense(64, activation="relu", name="sensor_delta_hidden")(sensor_proj)
        sensor_delta = layers.Dense(num_classes, activation="tanh", name="sensor_delta")(sensor_delta_hidden)
        sensor_weight = layers.Lambda(lambda w: w[:, 0:1], name="sensor_weight")(modality_weights)
        correction_terms.append(layers.Multiply(name="sensor_correction_term")([sensor_delta, sensor_weight]))

        if FLEX_POSTURE_DIM > 0:
            flex_delta_hidden = layers.Dense(64, activation="relu", name="flex_delta_hidden")(correction_modalities[1])
            flex_delta = layers.Dense(num_classes, activation="tanh", name="flex_delta")(flex_delta_hidden)
            flex_weight = layers.Lambda(lambda w: w[:, 1:2], name="flex_weight")(modality_weights)
            correction_terms.append(layers.Multiply(name="flex_correction_term")([flex_delta, flex_weight]))

        if len(correction_terms) == 1:
            modality_correction = correction_terms[0]
        else:
            modality_correction = layers.Add(name="modality_correction_sum")(correction_terms)

        ambiguity_score = layers.Lambda(
            lambda p: 1.0 - tf.reduce_max(p, axis=1, keepdims=True),
            name="ambiguity_score",
        )(base_probs)

        if MODEL_VARIANT == "modality_aware_rerank":
            top_k = min(3, num_classes)
            topk_mask = layers.Lambda(
                lambda p: tf.reduce_max(
                    tf.one_hot(tf.math.top_k(p, k=top_k).indices, depth=num_classes),
                    axis=1,
                ),
                name="topk_mask",
            )(base_probs)
            rerank_logits = layers.Multiply(name="rerank_logits")([modality_correction, topk_mask])
            gated_correction = layers.Multiply(name="gated_correction")([rerank_logits, ambiguity_score])
        else:
            gated_correction = layers.Multiply(name="gated_correction")([modality_correction, ambiguity_score])

        corrected_logits = layers.Add(name="corrected_logits")([base_logits, gated_correction])
        output = layers.Activation("softmax", name="class_probs")(corrected_logits)

        model = models.Model(
            inputs=inputs,
            outputs=output,
            name=(
                "modality_aware_rerank_flex_lstm"
                if MODEL_VARIANT == "modality_aware_rerank" and FLEX_POSTURE_DIM > 0
                else "modality_aware_rerank_lstm"
                if MODEL_VARIANT == "modality_aware_rerank"
                else "modality_aware_correction_flex_lstm"
                if FLEX_POSTURE_DIM > 0
                else "modality_aware_correction_lstm"
            ),
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    if MODEL_VARIANT in {"sensor_correction", "sensor_correction_ambiguous"}:
        sensor_proj = layers.Dense(64, activation="relu", name="sensor_project")(sensor_branch)
        fusion_parts = [vision_branch, sensor_proj]

        correction_context_parts = [vision_branch, sensor_proj]
        if FLEX_POSTURE_DIM > 0:
            flex_input = layers.Input(shape=(FLEX_POSTURE_DIM,), name="flex_posture_input")
            inputs.append(flex_input)
            flex_branch = layers.Dense(32, activation="relu", name="flex_posture_dense_1")(flex_input)
            flex_branch = layers.Dropout(0.2, name="flex_posture_dropout")(flex_branch)
            flex_branch = layers.Dense(32, activation="relu", name="flex_posture_dense_2")(flex_branch)
            fusion_parts.append(flex_branch)
            correction_context_parts.append(flex_branch)

        fused = layers.Concatenate(name="fusion_concat")(fusion_parts)
        fused = layers.Dense(96, activation="relu", name="fusion_hidden")(fused)
        fused = layers.Dropout(0.3, name="fusion_hidden_dropout")(fused)

        base_logits = layers.Dense(num_classes, name="base_logits")(fused)
        base_probs = layers.Activation("softmax", name="base_probs")(base_logits)

        correction_context = layers.Concatenate(name="correction_context")(correction_context_parts)
        correction_gate = layers.Dense(
            32,
            activation="relu",
            name="correction_gate_hidden",
        )(correction_context)
        correction_gate = layers.Dense(
            num_classes,
            activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(-2.0),
            name="correction_gate",
        )(correction_gate)

        correction_delta = layers.Dense(64, activation="relu", name="correction_delta_hidden")(sensor_proj)
        correction_delta = layers.Dense(
            num_classes,
            activation="tanh",
            name="correction_delta",
        )(correction_delta)

        gated_correction = layers.Multiply(name="gated_correction_raw")([correction_delta, correction_gate])

        if MODEL_VARIANT == "sensor_correction_ambiguous":
            ambiguity_score = layers.Lambda(
                lambda p: 1.0 - tf.reduce_max(p, axis=1, keepdims=True),
                name="ambiguity_score",
            )(base_probs)
            gated_correction = layers.Multiply(name="gated_correction")([gated_correction, ambiguity_score])
        else:
            gated_correction = layers.Lambda(lambda x: x, name="gated_correction")(gated_correction)

        corrected_logits = layers.Add(name="corrected_logits")([base_logits, gated_correction])
        output = layers.Activation("softmax", name="class_probs")(corrected_logits)

        model = models.Model(
            inputs=inputs,
            outputs=output,
            name=(
                "sensor_correction_ambiguous_flex_lstm"
                if MODEL_VARIANT == "sensor_correction_ambiguous" and FLEX_POSTURE_DIM > 0
                else "sensor_correction_ambiguous_lstm"
                if MODEL_VARIANT == "sensor_correction_ambiguous"
                else "sensor_correction_flex_lstm"
                if FLEX_POSTURE_DIM > 0
                else "sensor_correction_lstm"
            ),
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    sensor_proj = layers.Dense(64, activation="relu", name="sensor_project")(sensor_branch)
    fusion_context = layers.Concatenate(name="fusion_context")([vision_branch, sensor_proj])
    sensor_gate = layers.Dense(
        64,
        activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(-1.5),
        name="sensor_gate",
    )(fusion_context)
    sensor_delta = layers.Dense(64, activation="tanh", name="sensor_delta")(sensor_proj)
    gated_sensor = layers.Multiply(name="gated_sensor")([sensor_delta, sensor_gate])
    adapted_vision = layers.Add(name="adapted_vision")([vision_branch, gated_sensor])
    fusion_parts = [vision_branch, adapted_vision, sensor_proj]

    if FLEX_POSTURE_DIM > 0:
        flex_input = layers.Input(shape=(FLEX_POSTURE_DIM,), name="flex_posture_input")
        inputs.append(flex_input)
        flex_branch = layers.Dense(32, activation="relu", name="flex_posture_dense_1")(flex_input)
        flex_branch = layers.Dropout(0.2, name="flex_posture_dropout")(flex_branch)
        flex_branch = layers.Dense(32, activation="relu", name="flex_posture_dense_2")(flex_branch)
        fusion_parts.append(flex_branch)

    fused = layers.Concatenate(name="fusion_concat")(fusion_parts)
    fused = layers.Dense(96, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)
    output = layers.Dense(num_classes, activation="softmax")(fused)

    model = models.Model(
        inputs=inputs,
        outputs=output,
        name="vision_guided_sensor_gate_flex_lstm" if FLEX_POSTURE_DIM > 0 else "vision_guided_sensor_gate_lstm",
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.close()


def plot_confusion_matrix(cm, labels, save_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_split(model, X_inputs, y, split_name):
    loss, acc = model.evaluate(X_inputs, y, verbose=0)
    probs = model.predict(X_inputs, verbose=0)
    preds = np.argmax(probs, axis=1)

    macro_f1 = f1_score(y, preds, average="macro")
    weighted_f1 = f1_score(y, preds, average="weighted")

    print(f"\n===== {split_name} =====")
    print(f"Loss        : {loss:.4f}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")

    return {
        "loss": float(loss),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "preds": preds,
    }


def prepare_model_inputs(X_vision, X_sensor, X_flex):
    if FLEX_POSTURE_DIM > 0:
        return [X_vision, X_sensor, X_flex]
    return [X_vision, X_sensor]


def main():
    set_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("===== Load dataset =====")
    X, y, class_counts, active_actions, skipped_actions, sample_sources = load_dataset()

    print(f"Full X shape: {X.shape}")
    print(f"Full y shape: {y.shape}")
    print("\n===== Samples per class =====")
    for action in ACTIONS:
        print(f"{action:10s} : {class_counts[action]}")
    print("\n===== Trainable classes =====")
    print(", ".join(active_actions))
    if skipped_actions:
        print("\n===== Skipped classes =====")
        for action, reason in skipped_actions.items():
            print(f"{action:10s} : {reason}")

    X_train, X_val, X_test, y_train, y_val, y_test, split_indices, split_meta = split_dataset(X, y, sample_sources)

    print("\n===== Split result =====")
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Val  : {X_val.shape}, {y_val.shape}")
    print(f"Test : {X_test.shape}, {y_test.shape}")
    print("\n===== Sensor mode =====")
    print(f"Mode                 : {SENSOR_MODE}")
    print(f"Active raw indices   : {ACTIVE_RAW_SENSOR_INDICES}")
    print(f"Active flag indices  : {ACTIVE_FLAG_INDICES}")
    print(f"Active sensor dim    : {SENSOR_DIM}")
    print(f"Use flex posture     : {USE_FLEX_POSTURE}")
    print(f"Flex posture dim     : {FLEX_POSTURE_DIM}")
    print(f"Model variant        : {MODEL_VARIANT}")

    X_train_vision, X_train_sensor = split_modalities(X_train)
    X_val_vision, X_val_sensor = split_modalities(X_val)
    X_test_vision, X_test_sensor = split_modalities(X_test)
    X_train_flex = extract_flex_posture_features(X_train)
    X_val_flex = extract_flex_posture_features(X_val)
    X_test_flex = extract_flex_posture_features(X_test)

    vision_scaler = fit_vision_scaler_preserve_zero(X_train_vision)
    sensor_scaler = fit_sensor_scaler_preserve_flags(X_train_sensor)
    flex_scaler = fit_flex_posture_scaler(X_train_flex) if FLEX_POSTURE_DIM > 0 else None

    X_train_vision = transform_vision_with_preserved_zero(X_train_vision, vision_scaler)
    X_val_vision = transform_vision_with_preserved_zero(X_val_vision, vision_scaler)
    X_test_vision = transform_vision_with_preserved_zero(X_test_vision, vision_scaler)

    X_train_sensor = transform_sensor_with_preserved_flags(X_train_sensor, sensor_scaler)
    X_val_sensor = transform_sensor_with_preserved_flags(X_val_sensor, sensor_scaler)
    X_test_sensor = transform_sensor_with_preserved_flags(X_test_sensor, sensor_scaler)
    X_train_flex = transform_flex_posture_features(X_train_flex, flex_scaler) if FLEX_POSTURE_DIM > 0 else X_train_flex
    X_val_flex = transform_flex_posture_features(X_val_flex, flex_scaler) if FLEX_POSTURE_DIM > 0 else X_val_flex
    X_test_flex = transform_flex_posture_features(X_test_flex, flex_scaler) if FLEX_POSTURE_DIM > 0 else X_test_flex

    combined_mean = np.concatenate(
        [vision_scaler.mean_, sensor_scaler["mean_"]]
    ).astype(np.float32)
    combined_scale = np.concatenate(
        [vision_scaler.scale_, sensor_scaler["scale_"]]
    ).astype(np.float32)

    np.savez(
        os.path.join(OUTPUT_DIR, "fusion_scaler.npz"),
        mean=combined_mean,
        scale=combined_scale,
        vision_mean=vision_scaler.mean_.astype(np.float32),
        vision_scale=vision_scaler.scale_.astype(np.float32),
        sensor_mean=sensor_scaler["mean_"].astype(np.float32),
        sensor_scale=sensor_scaler["scale_"].astype(np.float32),
        flex_posture_mean=flex_scaler.mean_.astype(np.float32) if flex_scaler is not None else np.zeros((0,), dtype=np.float32),
        flex_posture_scale=flex_scaler.scale_.astype(np.float32) if flex_scaler is not None else np.ones((0,), dtype=np.float32),
    )

    with open(os.path.join(OUTPUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(active_actions, f, ensure_ascii=False, indent=2)

    if MODEL_VARIANT == "input_only":
        model_type = "dual_input_lstm" if FLEX_POSTURE_DIM > 0 else "dual_input_no_flex_lstm"
    elif MODEL_VARIANT == "sensor_correction":
        model_type = "sensor_correction_flex_lstm" if FLEX_POSTURE_DIM > 0 else "sensor_correction_lstm"
    elif MODEL_VARIANT == "sensor_correction_ambiguous":
        model_type = (
            "sensor_correction_ambiguous_flex_lstm"
            if FLEX_POSTURE_DIM > 0
            else "sensor_correction_ambiguous_lstm"
        )
    elif MODEL_VARIANT == "modality_aware_correction":
        model_type = (
            "modality_aware_correction_flex_lstm"
            if FLEX_POSTURE_DIM > 0
            else "modality_aware_correction_lstm"
        )
    elif MODEL_VARIANT == "modality_aware_rerank":
        model_type = (
            "modality_aware_rerank_flex_lstm"
            if FLEX_POSTURE_DIM > 0
            else "modality_aware_rerank_lstm"
        )
    else:
        model_type = "vision_guided_sensor_gate_flex_lstm" if FLEX_POSTURE_DIM > 0 else "vision_guided_sensor_gate_lstm"

    config = {
        "seq_length": SEQ_LENGTH,
        "vision_dim": VISION_DIM,
        "sensor_dim": SENSOR_DIM,
        "sensor_mode": SENSOR_MODE,
        "use_flex_posture": USE_FLEX_POSTURE,
        "flex_posture_dim": FLEX_POSTURE_DIM,
        "flex_baseline_frames": FLEX_BASELINE_FRAMES,
        "sensor_raw_dim": ACTIVE_RAW_SENSOR_DIM,
        "sensor_flag_dim": ACTIVE_SENSOR_FLAG_DIM,
        "sensor_raw_indices": ACTIVE_RAW_SENSOR_INDICES,
        "sensor_flag_indices": ACTIVE_FLAG_INDICES,
        "full_raw_sensor_dim": RAW_SENSOR_DIM,
        "full_sensor_flag_dim": SENSOR_FLAG_DIM,
        "sensor_flag_names": [SENSOR_FLAG_NAMES[i] for i in ACTIVE_FLAG_INDICES],
        "feature_dim": FEATURE_DIM,
        "input_format": model_type,
        "model_type": model_type,
        "model_variant": MODEL_VARIANT,
        "num_classes": len(active_actions),
        "actions": active_actions,
        "split_strategy": split_meta["strategy"],
        "split_warning": split_meta["warning"],
        "split_group_count": split_meta["group_count"],
        "split_session_group_count": split_meta["session_group_count"],
        "split_legacy_group_count": split_meta["legacy_group_count"],
        "min_samples_per_class": get_min_required_samples_per_class(),
        "skipped_actions": skipped_actions,
        "split_index_file": "split_indices.npz",
        "sample_source_file": "sample_sources.json",
    }
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    np.savez(
        os.path.join(OUTPUT_DIR, "split_indices.npz"),
        train=split_indices["train"],
        val=split_indices["val"],
        test=split_indices["test"],
    )
    with open(os.path.join(OUTPUT_DIR, "sample_sources.json"), "w", encoding="utf-8") as f:
        json.dump(sample_sources, f, ensure_ascii=False, indent=2)

    model = build_model(num_classes=len(active_actions))
    model.summary()

    cb_list = [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(OUTPUT_DIR, "fusion_lstm_best.keras"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\n===== Train =====")
    history = model.fit(
        prepare_model_inputs(X_train_vision, X_train_sensor, X_train_flex),
        y_train,
        validation_data=(prepare_model_inputs(X_val_vision, X_val_sensor, X_val_flex), y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_list,
        verbose=1,
    )

    plot_history(history, OUTPUT_DIR)

    best_model_path = os.path.join(OUTPUT_DIR, "fusion_lstm_best.keras")
    model = tf.keras.models.load_model(best_model_path)

    train_result = evaluate_split(
        model,
        prepare_model_inputs(X_train_vision, X_train_sensor, X_train_flex),
        y_train,
        "TRAIN",
    )
    val_result = evaluate_split(
        model,
        prepare_model_inputs(X_val_vision, X_val_sensor, X_val_flex),
        y_val,
        "VAL",
    )
    test_result = evaluate_split(
        model,
        prepare_model_inputs(X_test_vision, X_test_sensor, X_test_flex),
        y_test,
        "TEST",
    )

    test_preds = test_result["preds"]
    label_indices = list(range(len(active_actions)))
    report = classification_report(
        y_test,
        test_preds,
        labels=label_indices,
        target_names=active_actions,
        digits=4,
        zero_division=0,
    )
    print("\n===== TEST Classification Report =====")
    print(report)

    cm = confusion_matrix(y_test, test_preds, labels=label_indices)
    np.save(os.path.join(OUTPUT_DIR, "confusion_matrix.npy"), cm)
    plot_confusion_matrix(
        cm,
        active_actions,
        os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    )

    with open(os.path.join(OUTPUT_DIR, "train_summary.txt"), "w", encoding="utf-8") as f:
        f.write("===== Samples per class =====\n")
        for action in ACTIONS:
            f.write(f"{action:10s} : {class_counts[action]}\n")

        f.write("\n===== Trainable classes =====\n")
        f.write(", ".join(active_actions) + "\n")

        if skipped_actions:
            f.write("\n===== Skipped classes =====\n")
            for action, reason in skipped_actions.items():
                f.write(f"{action:10s} : {reason}\n")

        f.write("\n===== Split result =====\n")
        f.write(f"Train: {X_train.shape}, {y_train.shape}\n")
        f.write(f"Val  : {X_val.shape}, {y_val.shape}\n")
        f.write(f"Test : {X_test.shape}, {y_test.shape}\n")

        f.write("\n===== Split metadata =====\n")
        f.write(f"Strategy         : {split_meta['strategy']}\n")
        f.write(f"Warning          : {split_meta['warning']}\n")
        f.write(f"Group count      : {split_meta['group_count']}\n")
        f.write(f"Session groups   : {split_meta['session_group_count']}\n")
        f.write(f"Legacy groups    : {split_meta['legacy_group_count']}\n")
        f.write(f"Min/class used   : {get_min_required_samples_per_class()}\n")
        f.write(f"Split index file : {os.path.join(OUTPUT_DIR, 'split_indices.npz')}\n")
        f.write(f"Sample source file: {os.path.join(OUTPUT_DIR, 'sample_sources.json')}\n")
        f.write(f"Train count      : {len(split_indices['train'])}\n")
        f.write(f"Val count        : {len(split_indices['val'])}\n")
        f.write(f"Test count       : {len(split_indices['test'])}\n")
        if "train_group_count" in split_meta:
            f.write(f"Train groups     : {split_meta['train_group_count']}\n")
            f.write(f"Val groups       : {split_meta['val_group_count']}\n")
            f.write(f"Test groups      : {split_meta['test_group_count']}\n")

        f.write("\n===== Modal Split =====\n")
        f.write(f"Vision Train: {X_train_vision.shape}\n")
        f.write(f"Sensor Train: {X_train_sensor.shape}\n")
        f.write(f"Flex Train  : {X_train_flex.shape}\n")
        f.write(f"Vision Val  : {X_val_vision.shape}\n")
        f.write(f"Sensor Val  : {X_val_sensor.shape}\n")
        f.write(f"Flex Val    : {X_val_flex.shape}\n")
        f.write(f"Vision Test : {X_test_vision.shape}\n")
        f.write(f"Sensor Test : {X_test_sensor.shape}\n")
        f.write(f"Flex Test   : {X_test_flex.shape}\n")

        f.write("\n===== Final metrics =====\n")
        f.write(f"TRAIN Loss        : {train_result['loss']:.4f}\n")
        f.write(f"TRAIN Accuracy    : {train_result['accuracy']:.4f}\n")
        f.write(f"TRAIN Macro F1    : {train_result['macro_f1']:.4f}\n")
        f.write(f"TRAIN Weighted F1 : {train_result['weighted_f1']:.4f}\n\n")

        f.write(f"VAL Loss          : {val_result['loss']:.4f}\n")
        f.write(f"VAL Accuracy      : {val_result['accuracy']:.4f}\n")
        f.write(f"VAL Macro F1      : {val_result['macro_f1']:.4f}\n")
        f.write(f"VAL Weighted F1   : {val_result['weighted_f1']:.4f}\n\n")

        f.write(f"TEST Loss         : {test_result['loss']:.4f}\n")
        f.write(f"TEST Accuracy     : {test_result['accuracy']:.4f}\n")
        f.write(f"TEST Macro F1     : {test_result['macro_f1']:.4f}\n")
        f.write(f"TEST Weighted F1  : {test_result['weighted_f1']:.4f}\n\n")

        f.write("===== TEST Classification Report =====\n")
        f.write(report)

    print("\n===== Training complete =====")
    print(f"Best model saved : {best_model_path}")
    print(f"Loss plot saved  : {os.path.join(OUTPUT_DIR, 'loss_curve.png')}")
    print(f"Acc plot saved   : {os.path.join(OUTPUT_DIR, 'accuracy_curve.png')}")
    print(f"CM saved         : {os.path.join(OUTPUT_DIR, 'confusion_matrix.png')}")
    print(f"Summary saved    : {os.path.join(OUTPUT_DIR, 'train_summary.txt')}")
    print(f"Scaler saved     : {os.path.join(OUTPUT_DIR, 'fusion_scaler.npz')}")
    print(f"Labels saved     : {os.path.join(OUTPUT_DIR, 'labels.json')}")
    print(f"Split saved      : {os.path.join(OUTPUT_DIR, 'split_indices.npz')}")
    print(f"Sources saved    : {os.path.join(OUTPUT_DIR, 'sample_sources.json')}")


if __name__ == "__main__":
    main()
