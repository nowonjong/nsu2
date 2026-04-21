import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "myenv" / "Scripts" / "python.exe"
DATA_PATH = ROOT / "dataset_fusion_hold_v4_flags"
OUT_ROOT = ROOT / "experiments" / "fusion_design_ablation" / "2026-04-20_final_practical10_seed42_46"
PRACTICAL_DATASETS = {
    "in_1": ROOT / "experiments" / "live_eval_dataset" / "final_practical_in_10",
    "in_2": ROOT / "experiments" / "live_eval_dataset" / "final_practical_in_2_10",
    "out_1": ROOT / "experiments" / "live_eval_dataset" / "final_practical_out_10",
    "out_2": ROOT / "experiments" / "live_eval_dataset" / "final_practical_out_2_10",
}
SEEDS = [42, 43, 44, 45, 46]


MODELS = [
    {
        "key": "01_glove_vision_only",
        "label": "Glove vision-only",
        "script": "gpt_train.py",
        "env": {
            "NSU_VISION_DATA_DIR": str(DATA_PATH),
        },
    },
    {
        "key": "02_naive_raw_fusion",
        "label": "Naive raw fusion",
        "script": "nsu_train.py",
        "env": {
            "NSU_DATA_PATH": str(DATA_PATH),
            "NSU_SENSOR_MODE": "all",
            "NSU_USE_FLEX_POSTURE": "0",
            "NSU_MODEL_VARIANT": "input_only",
        },
    },
    {
        "key": "03_feature_redesigned_simple_fusion",
        "label": "Feature redesigned simple fusion",
        "script": "nsu_train.py",
        "env": {
            "NSU_DATA_PATH": str(DATA_PATH),
            "NSU_SENSOR_MODE": "imu_flags",
            "NSU_USE_FLEX_POSTURE": "1",
            "NSU_MODEL_VARIANT": "input_only",
        },
    },
    {
        "key": "04_stable_hybrid",
        "label": "Stable hybrid",
        "script": "nsu_train.py",
        "env": {
            "NSU_DATA_PATH": str(DATA_PATH),
            "NSU_SENSOR_MODE": "imu_flags",
            "NSU_USE_FLEX_POSTURE": "1",
            "NSU_MODEL_VARIANT": "stable_hybrid",
        },
    },
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


def model_done(model_dir: Path):
    return (
        (model_dir / "config.json").exists()
        and (model_dir / "labels.json").exists()
        and ((model_dir / "fusion_lstm_best.keras").exists() or (model_dir / "best_lstm.keras").exists())
        and ((model_dir / "fusion_scaler.npz").exists() or (model_dir / "vision_scaler.npz").exists())
    )


def train_model(seed, spec, model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_done(model_dir):
        print(f"[skip train] already exists: {model_dir}", flush=True)
        return

    env = dict(spec["env"])
    env["NSU_RANDOM_SEED"] = str(seed)
    if spec["script"] == "gpt_train.py":
        env["NSU_VISION_RESULT_DIR"] = str(model_dir)
    else:
        env["NSU_OUTPUT_DIR"] = str(model_dir)

    run([PYTHON, ROOT / spec["script"]], env=env)


def evaluate_model(seed, spec, model_dir: Path, reports_dir: Path):
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


def parse_train_summary(model_dir: Path):
    path = model_dir / "train_summary.txt"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    out = {}
    for key, pattern in {
        "test_accuracy": r"TEST Accuracy\s*:\s*([0-9.]+)",
        "test_macro_f1": r"TEST Macro F1\s*:\s*([0-9.]+)",
        "val_accuracy": r"VAL Accuracy\s*:\s*([0-9.]+)",
        "val_macro_f1": r"VAL Macro F1\s*:\s*([0-9.]+)",
    }.items():
        m = re.search(pattern, text)
        if m:
            out[key] = float(m.group(1))
    return out


def mean_std(values):
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def write_summaries(rows):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUT_ROOT / "final_practical10_seed42_46_summary.json"
    tsv_path = OUT_ROOT / "final_practical10_seed42_46_summary.tsv"
    md_path = OUT_ROOT / "final_practical10_seed42_46_summary.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"created_at": datetime.now().isoformat(timespec="seconds"), "rows": rows}, f, ensure_ascii=False, indent=2)

    cols = [
        "seed",
        "model_key",
        "model_label",
        "offline_test_acc",
        "offline_test_macro_f1",
        "practical_in_1_acc",
        "practical_in_2_acc",
        "practical_out_1_acc",
        "practical_out_2_acc",
        "practical_in_avg",
        "practical_out_avg",
        "practical_avg",
        "practical_macro_f1_avg",
        "offline_to_practical_drop",
    ]
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")

    grouped = {}
    for row in rows:
        grouped.setdefault(row["model_key"], []).append(row)

    lines = []
    lines.append("# Final Practical 10-Shot Multi-Seed Summary")
    lines.append("")
    lines.append("- Seeds: `42`, `43`, `44`, `45`, `46`")
    lines.append("- Practical datasets: final 10-shot sets, each `18 words x 10 samples`")
    lines.append("- Models: vision-only, naive raw fusion, feature redesigned simple fusion, stable hybrid")
    lines.append("- Note: stable hybrid is evaluated as a candidate, not assumed as the final model.")
    lines.append("- Add6 samples remain separable via each dataset manifest and can be excluded later if needed.")
    lines.append("")
    lines.append("## Seed-Level Results")
    lines.append("")
    lines.append("| Seed | Model | Offline Acc | In Avg | Out Avg | Practical Avg | Practical F1 Avg | Drop |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['model_label']} | "
            f"{row.get('offline_test_acc', float('nan')):.4f} | "
            f"{row['practical_in_avg']:.4f} | {row['practical_out_avg']:.4f} | "
            f"{row['practical_avg']:.4f} | {row['practical_macro_f1_avg']:.4f} | "
            f"{row['offline_to_practical_drop']:.4f} |"
        )
    lines.append("")
    lines.append("## Mean +- Std")
    lines.append("")
    lines.append("| Model | Offline Acc | In Avg | Out Avg | Practical Avg | Practical F1 Avg | Drop |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key, items in grouped.items():
        label = items[0]["model_label"]
        metrics = {}
        for field in ["offline_test_acc", "practical_in_avg", "practical_out_avg", "practical_avg", "practical_macro_f1_avg", "offline_to_practical_drop"]:
            metrics[field] = mean_std([x[field] for x in items if field in x])
        lines.append(
            f"| {label} | "
            f"{metrics['offline_test_acc'][0]:.4f} +- {metrics['offline_test_acc'][1]:.4f} | "
            f"{metrics['practical_in_avg'][0]:.4f} +- {metrics['practical_in_avg'][1]:.4f} | "
            f"{metrics['practical_out_avg'][0]:.4f} +- {metrics['practical_out_avg'][1]:.4f} | "
            f"{metrics['practical_avg'][0]:.4f} +- {metrics['practical_avg'][1]:.4f} | "
            f"{metrics['practical_macro_f1_avg'][0]:.4f} +- {metrics['practical_macro_f1_avg'][1]:.4f} | "
            f"{metrics['offline_to_practical_drop'][0]:.4f} +- {metrics['offline_to_practical_drop'][1]:.4f} |"
        )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- JSON: `{json_path}`")
    lines.append(f"- TSV: `{tsv_path}`")
    lines.append(f"- Per-dataset reports: `{OUT_ROOT / 'practical_reports'}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[summary] {md_path}", flush=True)


def main():
    if not PYTHON.exists():
        raise FileNotFoundError(PYTHON)
    for path in [DATA_PATH, *PRACTICAL_DATASETS.values()]:
        if not path.exists():
            raise FileNotFoundError(path)

    rows = []
    reports_dir = OUT_ROOT / "practical_reports"
    for seed in SEEDS:
        for spec in MODELS:
            model_dir = OUT_ROOT / f"seed{seed}" / spec["key"]
            print(f"\n=== seed={seed} model={spec['label']} ===", flush=True)
            train_model(seed, spec, model_dir)
            evals = evaluate_model(seed, spec, model_dir, reports_dir)
            train_metrics = parse_train_summary(model_dir)
            in_1_acc = evals["in_1"]["accuracy"]
            in_2_acc = evals["in_2"]["accuracy"]
            out_1_acc = evals["out_1"]["accuracy"]
            out_2_acc = evals["out_2"]["accuracy"]
            practical_f1s = [evals[k]["macro_f1"] for k in ["in_1", "in_2", "out_1", "out_2"]]
            practical_avg = float(np.mean([in_1_acc, in_2_acc, out_1_acc, out_2_acc]))
            offline_acc = train_metrics.get("test_accuracy", float("nan"))
            row = {
                "seed": seed,
                "model_key": spec["key"],
                "model_label": spec["label"],
                "model_dir": str(model_dir),
                "offline_test_acc": offline_acc,
                "offline_test_macro_f1": train_metrics.get("test_macro_f1", float("nan")),
                "practical_in_1_acc": in_1_acc,
                "practical_in_2_acc": in_2_acc,
                "practical_out_1_acc": out_1_acc,
                "practical_out_2_acc": out_2_acc,
                "practical_in_avg": float(np.mean([in_1_acc, in_2_acc])),
                "practical_out_avg": float(np.mean([out_1_acc, out_2_acc])),
                "practical_avg": practical_avg,
                "practical_macro_f1_avg": float(np.mean(practical_f1s)),
                "offline_to_practical_drop": float(offline_acc - practical_avg) if not np.isnan(offline_acc) else float("nan"),
            }
            rows.append(row)
            write_summaries(rows)

    write_summaries(rows)


if __name__ == "__main__":
    main()
