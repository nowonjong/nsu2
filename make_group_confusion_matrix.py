import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "experiments" / "fusion_design_ablation" / "2026-04-20_final_practical10_seed42_46" / "practical_reports"
OUT_DIR = ROOT / "experiments" / "group_confusion" / "2026-04-28_final10_seed42_46"
FIG_DIR = ROOT / "Paper" / "figures"

SEEDS = [42, 43, 44, 45, 46]
DATASETS = ["in_1", "in_2", "out_1", "out_2"]

MODELS = [
    ("01_glove_vision_only", "Vision-only", "vision_only"),
    ("03_feature_redesigned_simple_fusion", "Feature Redesigned Fusion", "feature_redesigned"),
]

GROUPS = [
    ("direction_difference", "Direction", ["more", "naeil", "eoje"]),
    ("wrist_angle_difference", "Wrist angle", ["teukbyeol", "byeollo"]),
    ("bend_path_difference", "Bend/path", ["jamkkan", "oraenman", "gakkapda"]),
    ("additional_motion_difference", "Additional motion", ["jalhada", "annyeonghaseyo"]),
    ("cross_hand_difference", "Cross-hand", ["mannada", "byeongyeonghada"]),
    ("instant_handshape_difference", "Instant handshape", ["banggeum", "billida"]),
    ("handshape_difference", "Handshape", ["gandanhada", "sada", "gamsahamnida", "joesonghada"]),
]

ACTION_TO_GROUP = {}
for group_idx, (_key, _name, actions) in enumerate(GROUPS):
    for action in actions:
        ACTION_TO_GROUP[action] = group_idx


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_matrix(model_key):
    n = len(GROUPS)
    cm = np.zeros((n, n), dtype=np.int64)
    skipped = []
    for seed in SEEDS:
        for dataset in DATASETS:
            path = REPORT_DIR / f"seed{seed}__{model_key}__{dataset}.json"
            report = load_json(path)
            for row in report["per_sample"]:
                gt = row["gt_label"]
                pred = row["pred_label"]
                if gt not in ACTION_TO_GROUP or pred not in ACTION_TO_GROUP:
                    skipped.append({"seed": seed, "dataset": dataset, "gt": gt, "pred": pred})
                    continue
                cm[ACTION_TO_GROUP[gt], ACTION_TO_GROUP[pred]] += 1
    return cm, skipped


def row_normalize(cm):
    totals = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, totals, out=np.zeros_like(cm, dtype=np.float64), where=totals != 0)


def write_csv(path, cm, norm):
    labels = [name for _key, name, _actions in GROUPS]
    with open(path, "w", encoding="utf-8") as f:
        f.write("actual_group,predicted_group,count,row_rate\n")
        for i, actual in enumerate(labels):
            for j, pred in enumerate(labels):
                f.write(f"{actual},{pred},{int(cm[i, j])},{norm[i, j]:.6f}\n")


def plot_heatmaps(results):
    labels = [name for _key, name, _actions in GROUPS]
    short_labels = ["Direction", "Wrist", "Bend/path", "Add. motion", "Cross-hand", "Instant", "Handshape"]

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), dpi=300, constrained_layout=True)
    for ax, (_model_key, title, slug) in zip(axes, MODELS):
        norm = results[slug]["row_normalized"]
        im = ax.imshow(norm, cmap="Greys", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=15, pad=10)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(short_labels, fontsize=9)
        ax.set_xlabel("Predicted group", fontsize=12)
        ax.set_ylabel("Actual group", fontsize=12)

        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                value = norm[i, j]
                text_color = "white" if value > 0.55 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    cbar = fig.colorbar(im, ax=axes, shrink=0.86, location="right")
    cbar.set_label("Row-normalized rate", fontsize=11)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIG_DIR / "fig_3_group_confusion_matrix.png"
    svg_path = FIG_DIR / "fig_3_group_confusion_matrix.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    return png_path, svg_path


def top_cross_group_confusions(cm, limit=8):
    labels = [name for _key, name, _actions in GROUPS]
    rows = []
    for i, actual in enumerate(labels):
        row_total = cm[i].sum()
        for j, pred in enumerate(labels):
            if i == j or cm[i, j] == 0:
                continue
            rows.append({
                "actual_group": actual,
                "predicted_group": pred,
                "count": int(cm[i, j]),
                "row_rate": float(cm[i, j] / row_total) if row_total else 0.0,
            })
    rows.sort(key=lambda x: x["count"], reverse=True)
    return rows[:limit]


def write_summary(results, fig_paths):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "group_confusion_matrix_summary.json"
    md_path = OUT_DIR / "group_confusion_matrix_summary.md"

    serializable = {}
    for slug, data in results.items():
        serializable[slug] = {
            "model_label": data["model_label"],
            "count_matrix": data["count_matrix"].tolist(),
            "row_normalized": data["row_normalized"].tolist(),
            "top_cross_group_confusions": data["top_cross_group_confusions"],
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seeds": SEEDS,
                "datasets": DATASETS,
                "groups": [
                    {"key": key, "label": label, "actions": actions}
                    for key, label, actions in GROUPS
                ],
                "results": serializable,
                "figures": [str(path) for path in fig_paths],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    labels = [name for _key, name, _actions in GROUPS]
    lines = [
        "# Group Confusion Matrix Summary",
        "",
        "- Unit: word-level predictions remapped to 7 predefined similar-sign groups",
        "- Datasets: final practical 10-shot sets",
        "- Seeds: 42-46 accumulated",
        "- Note: diagonal values indicate predictions that stayed within the same predefined group, not exact word accuracy.",
        "",
    ]
    for _model_key, title, slug in MODELS:
        data = results[slug]
        lines.extend([
            f"## {title}",
            "",
            "### Row-normalized matrix",
            "",
            "| Actual \\ Predicted | " + " | ".join(labels) + " |",
            "|---" + "|---:" * len(labels) + "|",
        ])
        norm = data["row_normalized"]
        for i, actual in enumerate(labels):
            lines.append(
                "| "
                + " | ".join([actual] + [f"{norm[i, j]:.3f}" for j in range(len(labels))])
                + " |"
            )
        lines.extend([
            "",
            "### Top cross-group confusions",
            "",
            "| Actual group | Predicted group | Count | Row rate |",
            "|---|---|---:|---:|",
        ])
        for row in data["top_cross_group_confusions"]:
            lines.append(
                f"| {row['actual_group']} | {row['predicted_group']} | {row['count']} | {row['row_rate']:.3f} |"
            )
        lines.append("")

    lines.extend([
        "## Files",
        "",
        f"- JSON: `{json_path}`",
        f"- PNG: `{fig_paths[0]}`",
        f"- SVG: `{fig_paths[1]}`",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    print(fig_paths[0])
    print(fig_paths[1])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for model_key, model_label, slug in MODELS:
        cm, skipped = compute_matrix(model_key)
        norm = row_normalize(cm)
        write_csv(OUT_DIR / f"{slug}_group_confusion_matrix.csv", cm, norm)
        results[slug] = {
            "model_label": model_label,
            "count_matrix": cm,
            "row_normalized": norm,
            "skipped": skipped,
            "top_cross_group_confusions": top_cross_group_confusions(cm),
        }
    fig_paths = plot_heatmaps(results)
    write_summary(results, fig_paths)


if __name__ == "__main__":
    main()
