import argparse
import json
import shutil
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="experiments/live_eval")
    parser.add_argument("--output-dir", default="live_eval_dataset/all")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of writing manifest only")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for json_path in sorted(source_root.rglob("*.json")):
        meta = load_json(json_path)
        gt_label = meta.get("gt_label")
        if not gt_label:
            continue
        npy_path = json_path.with_suffix(".npy")
        if not npy_path.exists():
            continue

        source_model_dir = meta.get("model_dir", "")
        source_model_name = Path(source_model_dir).name if source_model_dir else "unknown_model"
        target_dir = output_dir / gt_label
        target_dir.mkdir(parents=True, exist_ok=True)

        stem = json_path.stem
        target_stem = f"{stem}__src_{source_model_name}"
        target_npy = target_dir / f"{target_stem}.npy"
        target_json = target_dir / f"{target_stem}.json"

        entry = {
            "gt_label": gt_label,
            "source_json": str(json_path.resolve()),
            "source_npy": str(npy_path.resolve()),
            "source_model_dir": source_model_dir,
            "source_model_name": source_model_name,
            "target_npy": str(target_npy.resolve()),
            "target_json": str(target_json.resolve()),
        }
        manifest.append(entry)

        if args.copy:
            shutil.copy2(npy_path, target_npy)
            shutil.copy2(json_path, target_json)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"entries={len(manifest)}")
    print(f"manifest={manifest_path}")
    if args.copy:
        print(f"copied_to={output_dir}")


if __name__ == "__main__":
    main()
