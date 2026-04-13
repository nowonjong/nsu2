import argparse
import glob
import json
import os


def fmt(value, digits=2):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def get_stat(report, name, key):
    item = report.get("stats", {}).get(name)
    if not item:
        return None
    return item.get(key)


def main():
    parser = argparse.ArgumentParser(description="Summarize runtime profile JSON files.")
    parser.add_argument(
        "--dir",
        default=os.path.join("experiments", "runtime_profile"),
        help="Directory containing *_runtime_profile.json files.",
    )
    parser.add_argument("--latest", type=int, default=10, help="Number of latest reports to show.")
    args = parser.parse_args()

    pattern = os.path.join(args.dir, "*_runtime_profile.json")
    paths = sorted(glob.glob(pattern), key=os.path.getmtime)[-args.latest :]
    if not paths:
        print(f"No runtime profile files found in: {args.dir}")
        return

    headers = [
        "file",
        "script",
        "target_fps",
        "achieved_fps",
        "loop_mean_ms",
        "loop_p95_ms",
        "mediapipe_mean_ms",
        "inference_mean_ms",
        "sensor_mean_ms",
        "render_mean_ms",
        "sleep_mean_ms",
        "status",
    ]
    print("\t".join(headers))

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)

        loop_mean = get_stat(report, "loop_total", "mean_ms")
        achieved_fps = report.get("achieved_fps_from_loop_mean")
        if achieved_fps is None and loop_mean:
            achieved_fps = 1000.0 / loop_mean

        row = [
            os.path.basename(path),
            report.get("script", "-"),
            fmt(report.get("target_fps"), 1),
            fmt(achieved_fps, 2),
            fmt(loop_mean, 2),
            fmt(get_stat(report, "loop_total", "p95_ms"), 2),
            fmt(get_stat(report, "mediapipe_process", "mean_ms"), 2),
            fmt(get_stat(report, "model_inference", "mean_ms"), 2),
            fmt(get_stat(report, "sensor_align", "mean_ms"), 2),
            fmt(get_stat(report, "render_and_imshow", "mean_ms"), 2),
            fmt(get_stat(report, "fps_cap_sleep", "mean_ms"), 2),
            report.get("runtime_status", "-"),
        ]
        print("\t".join(row))


if __name__ == "__main__":
    main()
