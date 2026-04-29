import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "myenv" / "Scripts" / "python.exe"
DATA_PATH = ROOT / "dataset_fusion_hold_v4_flags"
OUT_ROOT = ROOT / "experiments" / "sensor_contribution_flags" / "2026-04-28_seed42_46"
BASE_SUMMARY_PATH = (
    ROOT
    / "experiments"
    / "fusion_design_ablation"
    / "2026-04-20_final_practical10_seed42_46"
    / "final_practical10_seed42_46_summary.json"
)

PRACTICAL_DATASETS = {
    "in_1": ROOT / "experiments" / "live_eval_dataset" / "final_practical_in_10",
    "in_2": ROOT / "experiments" / "live_eval_dataset" / "final_practical_in_2_10",
    "out_1": ROOT / "experiments" / "live_eval_dataset" / "final_practical_out_10",
    "out_2": ROOT / "experiments" / "live_eval_dataset" / "final_practical_out_2_10",
}

SEEDS = [42, 43, 44, 45, 46]

MODELS = [
    {
        "key": "02_vision_imu_flags",
        "label": "Vision + IMU + flags",
        "script": "nsu_train.py",
        "env": {
            "NSU_DATA_PATH": str(DATA_PATH),
            "NSU_SENSOR_MODE": "imu_flags",
            "NSU_USE_FLEX_POSTURE": "0",
            "NSU_MODEL_VARIANT": "input_only",
        },
    },
    {
        "key": "03_vision_flex_flags",
        "label": "Vision + Flex + flags",
        "script": "nsu_train.py",
        "env": {
            "NSU_DATA_PATH": str(DATA_PATH),
            "NSU_SENSOR_MODE": "flex_flags",
            "NSU_USE_FLEX_POSTURE": "0",
            "NSU_MODEL_VARIANT": "input_only",
        },
    },
]

BASELINE_KEYS = [
    ("01_glove_vision_only", "Vision-only"),
    ("02_naive_raw_fusion", "Vision + IMU + Flex + flags"),
]


def run(cmd, env=None):
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update({k: str(v) for k, v in env.items()})
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(ROOT),
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit={proc.returncode}: {' '.join(str(x) for x in cmd)}")
    return proc.stdout


def model_done(model_dir):
    return (
        (model_dir / "config.json").exists()
        and (model_dir / "labels.json").exists()
        and (model_dir / "fusion_lstm_best.keras").exists()
        and (model_dir / "fusion_scaler.npz").exists()
    )


def train_model(seed, spec, model_dir):
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_done(model_dir):
        print(f"[skip train] already exists: {model_dir}", flush=True)
        return

    env = dict(spec["env"])
    env["NSU_RANDOM_SEED"] = str(seed)
    env["NSU_OUTPUT_DIR"] = str(model_dir)
    run([PYTHON, ROOT / spec["script"]], env=env)


def evaluate_model(seed, spec, model_dir, reports_dir):
    reports_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for dataset_name, dataset_dir in PRACTICAL_DATASETS.items():
        run([PYTHON, ROOT / "evaluate_live_eval_dataset.py", "--model-dir", model_dir, "--dataset-dir", dataset_dir])
        src = model_dir / "live_eval_dataset_report.json"
        dst = reports_dir / f"seed{seed}__{spec['key']}__{dataset_name}.json"
        shutil.copy2(src, dst)
        with open(dst, "r", encoding="utf-8") as f:
            results[dataset_name] = json.load(f)
    return results


def parse_train_summary(model_dir):
    path = model_dir / "train_summary.txt"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    out = {}
    for key, pattern in {
        "test_accuracy": r"TEST Accuracy\s*:\s*([0-9.]+)",
        "test_macro_f1": r"TEST Macro F1\s*:\s*([0-9.]+)",
    }.items():
        match = re.search(pattern, text)
        if match:
            out[key] = float(match.group(1))
    return out


def summarize_rows(rows):
    out = []
    by_model = {}
    for row in rows:
        by_model.setdefault((row["model_key"], row["model_label"]), []).append(row)

    for (model_key, model_label), model_rows in by_model.items():
        summary = {
            "model_key": model_key,
            "model_label": model_label,
            "seeds": [row["seed"] for row in model_rows],
        }
        for key in [
            "offline_test_acc",
            "offline_test_macro_f1",
            "practical_in_avg",
            "practical_out_avg",
            "practical_avg",
            "practical_macro_f1_avg",
            "offline_to_practical_drop",
        ]:
            values = [row[key] for row in model_rows if row.get(key) is not None]
            if values:
                arr = np.array(values, dtype=np.float64)
                summary[f"{key}_mean"] = float(arr.mean())
                summary[f"{key}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        out.append(summary)
    return out


def load_baseline_summaries():
    with open(BASE_SUMMARY_PATH, "r", encoding="utf-8") as f:
        base = json.load(f)["rows"]
    selected = []
    for base_key, label in BASELINE_KEYS:
        rows = []
        for row in base:
            if row["model_key"] != base_key:
                continue
            copied = dict(row)
            copied["model_label"] = label
            rows.append(copied)
        selected.extend(summarize_rows(rows))
    return selected


def fmt_mean_std(row, key):
    mean = row.get(f"{key}_mean")
    std = row.get(f"{key}_std")
    if mean is None:
        return ""
    return f"{mean:.4f} +- {std:.4f}"


def write_outputs(rows):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize_rows(rows)
    baseline_rows = load_baseline_summaries()
    combined_rows = baseline_rows[:1] + summary_rows + baseline_rows[1:]

    json_path = OUT_ROOT / "sensor_contribution_flags_seed42_46_summary.json"
    tsv_path = OUT_ROOT / "sensor_contribution_flags_seed42_46_summary.tsv"
    md_path = OUT_ROOT / "sensor_contribution_flags_seed42_46_summary.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "seeds": SEEDS,
                "note": "Compares existing Vision-only and Naive raw fusion averages with newly trained IMU+flags and Flex+flags simple fusion models.",
                "baseline_summary_source": str(BASE_SUMMARY_PATH),
                "new_seed_rows": rows,
                "combined_summary_rows": combined_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    cols = [
        "model_label",
        "offline_test_acc",
        "practical_in_avg",
        "practical_out_avg",
        "practical_avg",
        "practical_macro_f1_avg",
        "offline_to_practical_drop",
    ]
    tsv_cols = ["model_label"]
    for col in cols[1:]:
        tsv_cols.extend([f"{col}_mean", f"{col}_std"])
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\t".join(tsv_cols) + "\n")
        for row in combined_rows:
            values = [row["model_label"]]
            for col in cols[1:]:
                values.append(str(row.get(f"{col}_mean", "")))
                values.append(str(row.get(f"{col}_std", "")))
            f.write("\t".join(values) + "\n")

    lines = [
        "# Sensor Contribution with Flags Summary",
        "",
        f"- Seeds: `{', '.join(str(seed) for seed in SEEDS)}`",
        "- Practical datasets: final 10-shot sets, each `18 words x 10 samples`",
        "- Vision-only and Vision + IMU + Flex + flags rows reuse the existing final fusion-method multi-seed results.",
        "- IMU-only and Flex-only rows were newly trained as simple input-only fusion models with frame/state flags included.",
        "",
        "| Model | Offline Acc | In Avg | Out Avg | Practical Avg | Practical F1 Avg | Drop |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model_label"],
                    fmt_mean_std(row, "offline_test_acc"),
                    fmt_mean_std(row, "practical_in_avg"),
                    fmt_mean_std(row, "practical_out_avg"),
                    fmt_mean_std(row, "practical_avg"),
                    fmt_mean_std(row, "practical_macro_f1_avg"),
                    fmt_mean_std(row, "offline_to_practical_drop"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- JSON: `{json_path}`",
            f"- New per-dataset reports: `{OUT_ROOT / 'practical_reports'}`",
            f"- Existing baseline source: `{BASE_SUMMARY_PATH}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[summary] {md_path}", flush=True)


def main():
    rows = []
    reports_dir = OUT_ROOT / "practical_reports"
    for seed in SEEDS:
        for spec in MODELS:
            model_dir = OUT_ROOT / f"seed{seed}" / spec["key"]
            train_model(seed, spec, model_dir)
            practical = evaluate_model(seed, spec, model_dir, reports_dir)
            train_metrics = parse_train_summary(model_dir)

            in_values = [practical["in_1"]["accuracy"], practical["in_2"]["accuracy"]]
            out_values = [practical["out_1"]["accuracy"], practical["out_2"]["accuracy"]]
            all_values = in_values + out_values
            all_f1 = [practical[name]["macro_f1"] for name in ["in_1", "in_2", "out_1", "out_2"]]
            practical_avg = float(np.mean(all_values))
            offline_acc = train_metrics.get("test_accuracy")

            rows.append(
                {
                    "seed": seed,
                    "model_key": spec["key"],
                    "model_label": spec["label"],
                    "offline_test_acc": offline_acc,
                    "offline_test_macro_f1": train_metrics.get("test_macro_f1"),
                    "practical_in_1_acc": practical["in_1"]["accuracy"],
                    "practical_in_2_acc": practical["in_2"]["accuracy"],
                    "practical_out_1_acc": practical["out_1"]["accuracy"],
                    "practical_out_2_acc": practical["out_2"]["accuracy"],
                    "practical_in_avg": float(np.mean(in_values)),
                    "practical_out_avg": float(np.mean(out_values)),
                    "practical_avg": practical_avg,
                    "practical_macro_f1_avg": float(np.mean(all_f1)),
                    "offline_to_practical_drop": None if offline_acc is None else float(offline_acc - practical_avg),
                }
            )

    write_outputs(rows)


if __name__ == "__main__":
    main()
