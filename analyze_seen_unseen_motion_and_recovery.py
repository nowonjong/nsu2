import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "experiments" / "fusion_design_ablation" / "2026-04-20_final_practical10_seed42_46" / "practical_reports"
OUT_DIR = ROOT / "experiments" / "seen_unseen_analysis" / "2026-04-28_motion_recovery"

DATASETS = {
    "in_1": {
        "condition": "Seen",
        "path": ROOT / "experiments" / "live_eval_dataset" / "final_practical_in_10",
    },
    "in_2": {
        "condition": "Seen",
        "path": ROOT / "experiments" / "live_eval_dataset" / "final_practical_in_2_10",
    },
    "out_1": {
        "condition": "Unseen",
        "path": ROOT / "experiments" / "live_eval_dataset" / "final_practical_out_10",
    },
    "out_2": {
        "condition": "Unseen",
        "path": ROOT / "experiments" / "live_eval_dataset" / "final_practical_out_2_10",
    },
}

SEEDS = [42, 43, 44, 45, 46]
VISION_KEY = "01_glove_vision_only"
FUSION_KEY = "03_feature_redesigned_simple_fusion"

VISION_DIM = 126
RAW_SENSOR_DIM = 26
FLEX_DIM = 8


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_sample_path(item, dataset_dir):
    for key in ["target_npy", "copied_npy", "source_npy"]:
        value = item.get(key)
        if not value:
            continue
        path = Path(value)
        if path.exists():
            return path
        for candidate in [dataset_dir / value, dataset_dir / "captures" / value]:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Could not resolve sample npy path: {item}")


def sample_id_from_report_row(row):
    return Path(row["source_json"]).name


def load_report(seed, model_key, dataset_key):
    path = REPORT_DIR / f"seed{seed}__{model_key}__{dataset_key}.json"
    report = load_json(path)
    return {sample_id_from_report_row(row): row for row in report["per_sample"]}


def compute_recovery_counts():
    counts = {
        "Seen": {
            "total": 0,
            "both_correct": 0,
            "vision_wrong_fusion_correct": 0,
            "vision_correct_fusion_wrong": 0,
            "both_wrong": 0,
            "vision_wrong": 0,
            "fusion_correct": 0,
        },
        "Unseen": {
            "total": 0,
            "both_correct": 0,
            "vision_wrong_fusion_correct": 0,
            "vision_correct_fusion_wrong": 0,
            "both_wrong": 0,
            "vision_wrong": 0,
            "fusion_correct": 0,
        },
    }

    for seed in SEEDS:
        for dataset_key, info in DATASETS.items():
            condition = info["condition"]
            vision_rows = load_report(seed, VISION_KEY, dataset_key)
            fusion_rows = load_report(seed, FUSION_KEY, dataset_key)
            common_ids = sorted(set(vision_rows) & set(fusion_rows))
            for sample_id in common_ids:
                v_correct = bool(vision_rows[sample_id]["correct"])
                f_correct = bool(fusion_rows[sample_id]["correct"])
                bucket = counts[condition]
                bucket["total"] += 1
                bucket["vision_wrong"] += int(not v_correct)
                bucket["fusion_correct"] += int(f_correct)
                if v_correct and f_correct:
                    bucket["both_correct"] += 1
                elif (not v_correct) and f_correct:
                    bucket["vision_wrong_fusion_correct"] += 1
                elif v_correct and (not f_correct):
                    bucket["vision_correct_fusion_wrong"] += 1
                else:
                    bucket["both_wrong"] += 1

    for condition, bucket in counts.items():
        total = bucket["total"]
        vision_wrong = bucket["vision_wrong"]
        bucket["vision_error_rate"] = bucket["vision_wrong"] / total if total else 0.0
        bucket["fusion_accuracy"] = bucket["fusion_correct"] / total if total else 0.0
        bucket["recovery_rate_among_vision_errors"] = (
            bucket["vision_wrong_fusion_correct"] / vision_wrong if vision_wrong else 0.0
        )
        bucket["recovery_rate_among_all_samples"] = (
            bucket["vision_wrong_fusion_correct"] / total if total else 0.0
        )
        bucket["regression_rate_among_all_samples"] = (
            bucket["vision_correct_fusion_wrong"] / total if total else 0.0
        )
    return counts


def mean_nonzero_frame_diff(vision):
    diffs = []
    for i in range(1, len(vision)):
        prev_nonzero = np.any(np.abs(vision[i - 1]) > 1e-6)
        cur_nonzero = np.any(np.abs(vision[i]) > 1e-6)
        if prev_nonzero and cur_nonzero:
            diffs.append(float(np.linalg.norm(vision[i] - vision[i - 1])))
    return float(np.mean(diffs)) if diffs else 0.0


def path_length_nonzero(vision):
    total = 0.0
    count = 0
    for i in range(1, len(vision)):
        prev_nonzero = np.any(np.abs(vision[i - 1]) > 1e-6)
        cur_nonzero = np.any(np.abs(vision[i]) > 1e-6)
        if prev_nonzero and cur_nonzero:
            total += float(np.linalg.norm(vision[i] - vision[i - 1]))
            count += 1
    return total, count


def sample_motion_metrics(npy_path):
    arr = np.load(npy_path).astype(np.float32)
    vision = arr[:, :VISION_DIM]
    sensor_raw = arr[:, VISION_DIM : VISION_DIM + RAW_SENSOR_DIM]
    flex = sensor_raw[:, :FLEX_DIM]
    imu = sensor_raw[:, FLEX_DIM:RAW_SENSOR_DIM]

    path_len, valid_steps = path_length_nonzero(vision)
    vision_motion = mean_nonzero_frame_diff(vision)
    flex_range = float(np.mean(np.max(flex, axis=0) - np.min(flex, axis=0)))
    flex_std = float(np.mean(np.std(flex, axis=0)))
    imu_range = float(np.mean(np.max(imu, axis=0) - np.min(imu, axis=0)))
    imu_std = float(np.mean(np.std(imu, axis=0)))

    return {
        "vision_motion": vision_motion,
        "vision_path_length": float(path_len),
        "vision_valid_steps": valid_steps,
        "flex_range": flex_range,
        "flex_std": flex_std,
        "imu_range": imu_range,
        "imu_std": imu_std,
    }


def summarize_metric(values):
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
    }


def compute_motion_summary():
    per_condition = {"Seen": [], "Unseen": []}
    for dataset_key, info in DATASETS.items():
        dataset_dir = info["path"]
        condition = info["condition"]
        manifest = load_json(dataset_dir / "manifest.json")
        samples = manifest["samples"] if isinstance(manifest, dict) else manifest
        for item in samples:
            npy_path = resolve_sample_path(item, dataset_dir)
            row = sample_motion_metrics(npy_path)
            row["dataset"] = dataset_key
            row["condition"] = condition
            row["gt_label"] = item.get("gt_label", item.get("label"))
            row["sample"] = npy_path.name
            per_condition[condition].append(row)

    summary = {}
    for condition, rows in per_condition.items():
        summary[condition] = {"sample_count": len(rows)}
        for metric in [
            "vision_motion",
            "vision_path_length",
            "flex_range",
            "flex_std",
            "imu_range",
            "imu_std",
        ]:
            summary[condition][metric] = summarize_metric([row[metric] for row in rows])
    return summary, per_condition


def fmt(value):
    return f"{value:.4f}"


def write_outputs(recovery, motion_summary, motion_rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "seen_unseen_motion_recovery_summary.json"
    md_path = OUT_DIR / "seen_unseen_motion_recovery_summary.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seeds": SEEDS,
                "recovery_counts": recovery,
                "motion_summary": motion_summary,
                "motion_rows": motion_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    lines = [
        "# Seen/Unseen Motion and Recovery Analysis",
        "",
        "- Compared models: Vision-only vs Feature Redesigned Fusion",
        "- Practical datasets: final practical 10-shot sets",
        "- Recovery counts are accumulated over 5 seeds and aligned per sample.",
        "- Motion variability is computed once per practical capture from stored `.npy` sequences.",
        "",
        "## Vision Error Recovery",
        "",
        "| Condition | Total sample-seed cases | Vision wrong | Fusion recovered | Recovery / Vision errors | Vision correct -> Fusion wrong |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in ["Seen", "Unseen"]:
        row = recovery[condition]
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    str(row["total"]),
                    str(row["vision_wrong"]),
                    str(row["vision_wrong_fusion_correct"]),
                    fmt(row["recovery_rate_among_vision_errors"]),
                    str(row["vision_correct_fusion_wrong"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Motion Variability",
            "",
            "| Condition | Samples | Vision motion | Vision path length | Flex range | Flex std | IMU range | IMU std |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in ["Seen", "Unseen"]:
        row = motion_summary[condition]
        lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    str(row["sample_count"]),
                    fmt(row["vision_motion"]["mean"]),
                    fmt(row["vision_path_length"]["mean"]),
                    fmt(row["flex_range"]["mean"]),
                    fmt(row["flex_std"]["mean"]),
                    fmt(row["imu_range"]["mean"]),
                    fmt(row["imu_std"]["mean"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Note",
            "",
            "This analysis supports a cautious interpretation: subject-dependent motion and sensor-pattern differences are present in the practical captures, and Feature Redesigned Fusion recovers a substantial portion of Vision-only errors, especially in the Unseen condition.",
            "",
            "## Files",
            "",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)


def main():
    recovery = compute_recovery_counts()
    motion_summary, motion_rows = compute_motion_summary()
    write_outputs(recovery, motion_summary, motion_rows)


if __name__ == "__main__":
    main()
