import os
import glob
import json
import time
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
    ConfusionMatrixDisplay,
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

DATA_DIR = os.getenv("NSU_VISION_DATA_DIR", "bare_hand_vision_8words_v1")
RESULT_DIR = os.getenv("NSU_VISION_RESULT_DIR", "bare_hand_vision_8words_v1_results")
SEQ_LEN = 60
FEATURE_DIM = 126
TEST_SIZE = 0.15
VAL_SIZE = 0.15
MIN_SAMPLES_PER_CLASS = 5
LEGACY_GROUP_BLOCK_SIZE = 5
BATCH_SIZE = 16
EPOCHS = 100
RANDOM_STATE = int(os.getenv("NSU_RANDOM_SEED", "42"))
LEARNING_RATE = 1e-3


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_min_required_samples_per_class():
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


def build_sample_group_info(file_path, action):
    file_name = os.path.basename(file_path)
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


def load_data(data_dir, actions, seq_len=60, feature_dim=126):
    X = []
    y = []
    sample_sources = []
    class_counts = {}
    active_actions = []
    skipped_actions = {}

    for action in actions:
        action_dir = os.path.join(data_dir, action)
        if os.path.isdir(action_dir):
            files = sorted(glob.glob(os.path.join(action_dir, "*.npy")))
        else:
            pattern = os.path.join(data_dir, f"{action}_*.npy")
            files = sorted(glob.glob(pattern))
        class_counts[action] = len(files)

        if len(files) < get_min_required_samples_per_class():
            skipped_actions[action] = (
                f"too_few_samples({len(files)} < {get_min_required_samples_per_class()})"
            )
            continue

        label_idx = len(active_actions)
        active_actions.append(action)

        for file_path in files:
            arr = np.load(file_path)
            group_info = build_sample_group_info(file_path, action)
            if arr.shape == (seq_len, feature_dim):
                vision_arr = arr
            elif arr.shape[0] == seq_len and arr.shape[1] >= feature_dim:
                # Allow training a vision-only baseline from fusion-collected data
                # by slicing the vision portion (first 126 dims) from 152-d samples.
                vision_arr = arr[:, :feature_dim]
            else:
                print(f"[skip] shape mismatch: {file_path}, got={arr.shape}")
                continue

            X.append(vision_arr.astype(np.float32))
            y.append(label_idx)
            sample_sources.append(
                {
                    "action": action,
                    "file": os.path.basename(file_path),
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
    X_trainval, X_test, y_trainval, y_test, idx_trainval, idx_test = train_test_split(
        X,
        y,
        indices,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    val_ratio_in_trainval = VAL_SIZE / (1.0 - TEST_SIZE)

    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_trainval,
        y_trainval,
        idx_trainval,
        test_size=val_ratio_in_trainval,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
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

    _, complete_eval_groups, incomplete_groups = find_complete_eval_groups(y, sample_sources)
    if len(complete_eval_groups) < 3:
        raise ValueError(
            "Need at least 3 complete session groups containing every class "
            f"for grouped val/test split, got {len(complete_eval_groups)} complete groups."
        )

    eval_mask = np.isin(groups, np.array(complete_eval_groups, dtype=object))
    forced_train_mask = ~eval_mask
    eval_indices = np.where(eval_mask)[0].astype(np.int32)
    forced_train_indices = np.where(forced_train_mask)[0].astype(np.int32)
    y_eval = y[eval_indices]
    groups_eval = groups[eval_indices]

    outer_splits = min(5, len(np.unique(groups_eval)))
    if outer_splits < 2:
        raise ValueError("Need at least 2 group folds for test split.")

    outer_cv = StratifiedGroupKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    outer_candidates = list(outer_cv.split(np.zeros((len(y_eval), 1), dtype=np.float32), y_eval, groups_eval))
    trainval_rel, test_rel = choose_best_group_fold(
        outer_candidates,
        y_eval,
        target_size=max(1, int(round(len(y_eval) * TEST_SIZE))),
    )

    idx_trainval = eval_indices[trainval_rel].astype(np.int32)
    idx_test = eval_indices[test_rel].astype(np.int32)
    y_trainval = y[idx_trainval]
    groups_trainval = groups[idx_trainval]
    unique_trainval_groups = np.unique(groups_trainval)
    inner_splits = min(5, len(unique_trainval_groups))
    if inner_splits < 2:
        raise ValueError("Need at least 2 remaining group folds for val split.")

    inner_cv = StratifiedGroupKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=RANDOM_STATE + 1,
    )
    inner_candidates = list(inner_cv.split(
        np.zeros((len(idx_trainval), 1), dtype=np.float32),
        y_trainval,
        groups_trainval,
    ))
    train_rel, val_rel = choose_best_group_fold(
        inner_candidates,
        y_trainval,
        target_size=max(1, int(round(len(idx_trainval) * (VAL_SIZE / (1.0 - TEST_SIZE))))),
    )

    idx_train = idx_trainval[train_rel].astype(np.int32)
    idx_val = idx_trainval[val_rel].astype(np.int32)
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


def fit_and_scale(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, FEATURE_DIM)
    X_val_2d = X_val.reshape(-1, FEATURE_DIM)
    X_test_2d = X_test.reshape(-1, FEATURE_DIM)

    scaler.fit(X_train_2d)

    X_train_scaled = scaler.transform(X_train_2d).reshape(-1, SEQ_LEN, FEATURE_DIM).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_2d).reshape(-1, SEQ_LEN, FEATURE_DIM).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_2d).reshape(-1, SEQ_LEN, FEATURE_DIM).astype(np.float32)
    return scaler, X_train_scaled, X_val_scaled, X_test_scaled


def build_lstm_model(input_shape, num_classes):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Masking(mask_value=0.0),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, result_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("lstm Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "lstm_loss.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.title("lstm Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "lstm_accuracy.png"))
    plt.close()


def measure_latency_ms(model, x_test, warmup=10):
    for i in range(min(warmup, len(x_test))):
        _ = model(x_test[i:i + 1], training=False).numpy()

    times = []
    for i in range(len(x_test)):
        start = time.perf_counter()
        _ = model(x_test[i:i + 1], training=False).numpy()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    return float(np.mean(times)), float(np.std(times))


def evaluate_split(model, x_input, y_true, split_name):
    loss, acc = model.evaluate(x_input, y_true, verbose=0)
    probs = model.predict(x_input, batch_size=BATCH_SIZE, verbose=0)
    preds = np.argmax(probs, axis=1)

    macro_f1 = f1_score(y_true, preds, average="macro")
    weighted_f1 = f1_score(y_true, preds, average="weighted")

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
        "probs": probs,
    }


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    set_seed(RANDOM_STATE)

    print("===== Load data =====")
    X, y, class_counts, active_actions, skipped_actions, sample_sources = load_data(
        DATA_DIR, ACTIONS, seq_len=SEQ_LEN, feature_dim=FEATURE_DIM
    )

    print("Full X shape:", X.shape)
    print("Full y shape:", y.shape)
    print("\n===== Samples per class =====")
    for action in ACTIONS:
        print(f"{action:15s} : {class_counts[action]}")
    print("\n===== Trainable classes =====")
    print(", ".join(active_actions))
    if skipped_actions:
        print("\n===== Skipped classes =====")
        for action, reason in skipped_actions.items():
            print(f"{action:15s} : {reason}")

    X_train, X_val, X_test, y_train, y_val, y_test, split_indices, split_meta = split_dataset(X, y, sample_sources)

    print("\n===== Split result =====")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val  :", X_val.shape, "y_val  :", y_val.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

    scaler, X_train_scaled, X_val_scaled, X_test_scaled = fit_and_scale(X_train, X_val, X_test)

    np.savez(
        os.path.join(RESULT_DIR, "vision_scaler.npz"),
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
    )

    with open(os.path.join(RESULT_DIR, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(active_actions, f, ensure_ascii=False, indent=2)

    config = {
        "seq_length": SEQ_LEN,
        "feature_dim": FEATURE_DIM,
        "input_format": "vision_only_lstm",
        "model_type": "lstm",
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
    with open(os.path.join(RESULT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    np.savez(
        os.path.join(RESULT_DIR, "split_indices.npz"),
        train=split_indices["train"],
        val=split_indices["val"],
        test=split_indices["test"],
    )
    with open(os.path.join(RESULT_DIR, "sample_sources.json"), "w", encoding="utf-8") as f:
        json.dump(sample_sources, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("[Train start] lstm")
    print("=" * 60)

    model = build_lstm_model((SEQ_LEN, FEATURE_DIM), len(active_actions))
    model.summary()

    checkpoint_path = os.path.join(RESULT_DIR, "best_lstm.keras")
    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=1,
    )

    plot_history(history, RESULT_DIR)

    best_model = tf.keras.models.load_model(checkpoint_path)

    train_result = evaluate_split(best_model, X_train_scaled, y_train, "TRAIN")
    val_result = evaluate_split(best_model, X_val_scaled, y_val, "VAL")
    test_result = evaluate_split(best_model, X_test_scaled, y_test, "TEST")

    y_pred = test_result["preds"]
    label_indices = list(range(len(active_actions)))
    report_text = classification_report(
        y_test,
        y_pred,
        labels=label_indices,
        target_names=active_actions,
        digits=4,
        zero_division=0,
    )
    report_json = classification_report(
        y_test,
        y_pred,
        labels=label_indices,
        target_names=active_actions,
        output_dict=True,
        digits=4,
        zero_division=0,
    )
    print("\n===== TEST Classification Report =====")
    print(report_text)

    cm = confusion_matrix(y_test, y_pred, labels=label_indices)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=active_actions)
    disp.plot(cmap="Blues", xticks_rotation=45, values_format="d")
    plt.title("lstm Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "lstm_confusion_matrix.png"))
    plt.close()

    cm_norm = confusion_matrix(y_test, y_pred, labels=label_indices, normalize="true")
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=active_actions)
    disp.plot(cmap="Blues", xticks_rotation=45, values_format=".2f")
    plt.title("lstm Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "lstm_confusion_matrix_norm.png"))
    plt.close()

    mean_latency_ms, std_latency_ms = measure_latency_ms(best_model, X_test_scaled)
    final_model_path = os.path.join(RESULT_DIR, "lstm_final.keras")
    best_model.save(final_model_path)
    model_size_mb = os.path.getsize(final_model_path) / (1024 * 1024)

    result = {
        "model_name": "lstm",
        "test_loss": float(test_result["loss"]),
        "test_accuracy": float(test_result["accuracy"]),
        "macro_f1": float(test_result["macro_f1"]),
        "weighted_f1": float(test_result["weighted_f1"]),
        "mean_latency_ms_per_sample": float(mean_latency_ms),
        "std_latency_ms_per_sample": float(std_latency_ms),
        "approx_fps_if_single_window": float(1000.0 / mean_latency_ms) if mean_latency_ms > 0 else None,
        "num_params": int(best_model.count_params()),
        "model_size_mb": float(model_size_mb),
        "classification_report": report_json,
    }

    with open(os.path.join(RESULT_DIR, "lstm_report.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    summary = [
        {
            "model_name": "lstm",
            "test_accuracy": result["test_accuracy"],
            "macro_f1": result["macro_f1"],
            "weighted_f1": result["weighted_f1"],
            "mean_latency_ms_per_sample": result["mean_latency_ms_per_sample"],
            "approx_fps_if_single_window": result["approx_fps_if_single_window"],
            "num_params": result["num_params"],
            "model_size_mb": result["model_size_mb"],
        }
    ]
    with open(os.path.join(RESULT_DIR, "comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(RESULT_DIR, "train_summary.txt"), "w", encoding="utf-8") as f:
        f.write("===== Samples per class =====\n")
        for action in ACTIONS:
            f.write(f"{action:15s} : {class_counts[action]}\n")

        f.write("\n===== Trainable classes =====\n")
        f.write(", ".join(active_actions) + "\n")

        if skipped_actions:
            f.write("\n===== Skipped classes =====\n")
            for action, reason in skipped_actions.items():
                f.write(f"{action:15s} : {reason}\n")

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
        f.write(f"Split index file : {os.path.join(RESULT_DIR, 'split_indices.npz')}\n")
        f.write(f"Sample source file: {os.path.join(RESULT_DIR, 'sample_sources.json')}\n")
        f.write(f"Train count      : {len(split_indices['train'])}\n")
        f.write(f"Val count        : {len(split_indices['val'])}\n")
        f.write(f"Test count       : {len(split_indices['test'])}\n")
        if "train_group_count" in split_meta:
            f.write(f"Train groups     : {split_meta['train_group_count']}\n")
            f.write(f"Val groups       : {split_meta['val_group_count']}\n")
            f.write(f"Test groups      : {split_meta['test_group_count']}\n")

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
        f.write(f"Mean latency(ms)  : {mean_latency_ms:.2f}\n")
        f.write(f"Std latency(ms)   : {std_latency_ms:.2f}\n")
        f.write(f"Params            : {best_model.count_params():,}\n")
        f.write(f"Model size(MB)    : {model_size_mb:.2f}\n\n")
        f.write("===== TEST Classification Report =====\n")
        f.write(report_text)

    print("\n===== Training complete =====")
    print(f"Best model saved : {checkpoint_path}")
    print(f"Final model saved: {final_model_path}")
    print(f"Loss plot saved  : {os.path.join(RESULT_DIR, 'lstm_loss.png')}")
    print(f"Acc plot saved   : {os.path.join(RESULT_DIR, 'lstm_accuracy.png')}")
    print(f"CM saved         : {os.path.join(RESULT_DIR, 'lstm_confusion_matrix.png')}")
    print(f"Summary saved    : {os.path.join(RESULT_DIR, 'train_summary.txt')}")
    print(f"Scaler saved     : {os.path.join(RESULT_DIR, 'vision_scaler.npz')}")
    print(f"Labels saved     : {os.path.join(RESULT_DIR, 'class_names.json')}")
    print(f"Split saved      : {os.path.join(RESULT_DIR, 'split_indices.npz')}")
    print(f"Sources saved    : {os.path.join(RESULT_DIR, 'sample_sources.json')}")


if __name__ == "__main__":
    main()
