import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import statistics


PROJECT_ROOT = Path(__file__).resolve().parent


SUMMARY_PATTERNS = {
    "test_accuracy": re.compile(r"TEST Accuracy\s*:\s*([0-9.]+)"),
    "test_macro_f1": re.compile(r"TEST Macro F1\s*:\s*([0-9.]+)"),
    "test_weighted_f1": re.compile(r"TEST Weighted F1\s*:\s*([0-9.]+)"),
    "val_accuracy": re.compile(r"VAL Accuracy\s*:\s*([0-9.]+)"),
    "split_strategy": re.compile(r"Strategy\s*:\s*(.+)"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeated-seed training for vision/fusion comparisons and collect summaries."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use. Default: current Python.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(42, 52)),
        help="Seeds to run. Default: 42 43 44 45 46 47 48 49 50 51",
    )
    parser.add_argument(
        "--run-glove-vision",
        action="store_true",
        help="Run glove + vision-only repeated seeds.",
    )
    parser.add_argument(
        "--run-glove-fusion",
        action="store_true",
        help="Run glove + fusion repeated seeds.",
    )
    parser.add_argument(
        "--run-bare-vision",
        action="store_true",
        help="Run bare-hand + vision-only repeated seeds. Requires --bare-data-dir.",
    )
    parser.add_argument(
        "--glove-vision-data-dir",
        default="dataset_fusion_hold_v4_flags",
        help="Dataset for glove + vision-only. Default: dataset_fusion_hold_v4_flags",
    )
    parser.add_argument(
        "--glove-vision-output-prefix",
        default="cmp_seed_glove_vision",
        help="Output prefix for glove + vision-only runs.",
    )
    parser.add_argument(
        "--glove-fusion-data-dir",
        default="dataset_fusion_hold_v4_flags",
        help="Dataset for glove + fusion. Default: dataset_fusion_hold_v4_flags",
    )
    parser.add_argument(
        "--glove-fusion-output-prefix",
        default="cmp_seed_glove_fusion_stable",
        help="Output prefix for glove + fusion runs.",
    )
    parser.add_argument(
        "--bare-data-dir",
        default=None,
        help="Dataset for bare-hand + vision-only repeated runs.",
    )
    parser.add_argument(
        "--bare-output-prefix",
        default="cmp_seed_bare_vision",
        help="Output prefix for bare-hand + vision-only runs.",
    )
    parser.add_argument(
        "--summary-dir",
        default="results_summary",
        help="Directory where combined CSV/JSON summaries will be written.",
    )
    parser.add_argument(
        "--only-summary",
        action="store_true",
        help="Skip training and only aggregate existing output folders.",
    )
    return parser.parse_args()


def ensure_one_target_selected(args):
    if args.run_glove_vision or args.run_glove_fusion or args.run_bare_vision:
        return
    args.run_glove_vision = True
    args.run_glove_fusion = True


def run_command(command, env):
    print(f"[RUN] {' '.join(command)}")
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def parse_train_summary(summary_path):
    if not summary_path.exists():
        raise FileNotFoundError(f"train_summary.txt not found: {summary_path}")

    text = summary_path.read_text(encoding="utf-8")
    parsed = {}
    for key, pattern in SUMMARY_PATTERNS.items():
        match = pattern.search(text)
        parsed[key] = match.group(1).strip() if match else None

    for key in ("test_accuracy", "test_macro_f1", "test_weighted_f1", "val_accuracy"):
        if parsed[key] is not None:
            parsed[key] = float(parsed[key])

    return parsed


def build_glove_vision_env(args, seed, output_dir):
    env = os.environ.copy()
    env["NSU_RANDOM_SEED"] = str(seed)
    env["NSU_VISION_DATA_DIR"] = args.glove_vision_data_dir
    env["NSU_VISION_RESULT_DIR"] = str(output_dir)
    return env


def build_glove_fusion_env(args, seed, output_dir):
    env = os.environ.copy()
    env["NSU_RANDOM_SEED"] = str(seed)
    env["NSU_DATA_PATH"] = args.glove_fusion_data_dir
    env["NSU_OUTPUT_DIR"] = str(output_dir)
    env["NSU_MODEL_VARIANT"] = "stable_hybrid"
    env["NSU_SENSOR_MODE"] = "imu_flags"
    env["NSU_USE_FLEX_POSTURE"] = "1"
    return env


def build_bare_vision_env(args, seed, output_dir):
    env = os.environ.copy()
    env["NSU_RANDOM_SEED"] = str(seed)
    env["NSU_VISION_DATA_DIR"] = args.bare_data_dir
    env["NSU_VISION_RESULT_DIR"] = str(output_dir)
    return env


def run_target(label, script_name, seeds, output_prefix, env_builder, args, summary_rows):
    for seed in seeds:
        output_dir = PROJECT_ROOT / f"{output_prefix}{seed}"
        summary_path = output_dir / "train_summary.txt"

        if not args.only_summary:
            env = env_builder(args, seed, output_dir)
            run_command([args.python, script_name], env=env)

        metrics = parse_train_summary(summary_path)
        summary_rows.append(
            {
                "label": label,
                "seed": seed,
                "output_dir": output_dir.name,
                "test_accuracy": metrics["test_accuracy"],
                "test_macro_f1": metrics["test_macro_f1"],
                "test_weighted_f1": metrics["test_weighted_f1"],
                "val_accuracy": metrics["val_accuracy"],
                "split_strategy": metrics["split_strategy"],
            }
        )


def compute_group_summary(rows):
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    summary = []
    for label, items in sorted(by_label.items()):
        accs = [item["test_accuracy"] for item in items]
        f1s = [item["test_macro_f1"] for item in items]
        summary.append(
            {
                "label": label,
                "count": len(items),
                "mean_accuracy": sum(accs) / len(accs),
                "std_accuracy": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
                "min_accuracy": min(accs),
                "max_accuracy": max(accs),
                "mean_macro_f1": sum(f1s) / len(f1s),
                "std_macro_f1": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
                "min_macro_f1": min(f1s),
                "max_macro_f1": max(f1s),
            }
        )
    return summary


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    ensure_one_target_selected(args)

    if args.run_bare_vision and not args.bare_data_dir:
        raise ValueError("--run-bare-vision requires --bare-data-dir")

    summary_dir = PROJECT_ROOT / args.summary_dir
    summary_dir.mkdir(parents=True, exist_ok=True)

    per_run_rows = []

    if args.run_glove_vision:
        run_target(
            label="glove_vision_only",
            script_name="gpt_train.py",
            seeds=args.seeds,
            output_prefix=args.glove_vision_output_prefix,
            env_builder=build_glove_vision_env,
            args=args,
            summary_rows=per_run_rows,
        )

    if args.run_glove_fusion:
        run_target(
            label="glove_fusion_stable_hybrid",
            script_name="nsu_train.py",
            seeds=args.seeds,
            output_prefix=args.glove_fusion_output_prefix,
            env_builder=build_glove_fusion_env,
            args=args,
            summary_rows=per_run_rows,
        )

    if args.run_bare_vision:
        run_target(
            label="bare_vision_only",
            script_name="gpt_train.py",
            seeds=args.seeds,
            output_prefix=args.bare_output_prefix,
            env_builder=build_bare_vision_env,
            args=args,
            summary_rows=per_run_rows,
        )

    summary_rows = compute_group_summary(per_run_rows)

    per_run_csv = summary_dir / "multiseed_per_run.csv"
    group_csv = summary_dir / "multiseed_group_summary.csv"
    group_json = summary_dir / "multiseed_group_summary.json"

    write_csv(
        per_run_csv,
        per_run_rows,
        [
            "label",
            "seed",
            "output_dir",
            "test_accuracy",
            "test_macro_f1",
            "test_weighted_f1",
            "val_accuracy",
            "split_strategy",
        ],
    )
    write_csv(
        group_csv,
        summary_rows,
        [
            "label",
            "count",
            "mean_accuracy",
            "std_accuracy",
            "min_accuracy",
            "max_accuracy",
            "mean_macro_f1",
            "std_macro_f1",
            "min_macro_f1",
            "max_macro_f1",
        ],
    )
    group_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Per-run summary: {per_run_csv}")
    print(f"[DONE] Group summary CSV: {group_csv}")
    print(f"[DONE] Group summary JSON: {group_json}")


if __name__ == "__main__":
    main()
