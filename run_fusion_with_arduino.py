import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SKETCH = PROJECT_ROOT / "sensor_send" / "sensor_send.ino"
DEFAULT_FUSION_SCRIPT = PROJECT_ROOT / "nsu_run_fusion_v6_overlap_hold.py"
ARDUINO_CONFIG = PROJECT_ROOT / "arduino-cli.yaml"
ARDUINO_BUILD_ROOT = PROJECT_ROOT / "arduino_build"


def find_arduino_cli():
    candidates = [
        shutil.which("arduino-cli"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Arduino15" / "arduino-cli.exe",
        Path("C:/Program Files/Arduino CLI/arduino-cli.exe"),
        Path("C:/Program Files (x86)/Arduino CLI/arduino-cli.exe"),
    ]

    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return str(candidate_path)
    return None


def with_arduino_config(command):
    if not ARDUINO_CONFIG.exists():
        return command
    return [command[0], "--config-file", str(ARDUINO_CONFIG), *command[1:]]


def run_command(command, cwd):
    print(f"[RUN] {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd)
    return result.returncode == 0


def upload_sketch(arduino_cli, sketch_path, board_fqbn, port):
    sketch_dir = str(sketch_path.parent)
    build_dir = ARDUINO_BUILD_ROOT / sketch_path.stem
    build_dir.mkdir(parents=True, exist_ok=True)

    compile_cmd = [
        arduino_cli,
        "compile",
        "--build-path",
        str(build_dir),
        "--fqbn",
        board_fqbn,
        sketch_dir,
    ]
    compile_cmd = with_arduino_config(compile_cmd)
    if not run_command(compile_cmd, cwd=PROJECT_ROOT):
        return False

    upload_cmd = [
        arduino_cli,
        "upload",
        "-p",
        port,
        "--input-dir",
        str(build_dir),
        "--fqbn",
        board_fqbn,
        sketch_dir,
    ]
    upload_cmd = with_arduino_config(upload_cmd)
    return run_command(upload_cmd, cwd=PROJECT_ROOT)


def run_fusion(python_exe, fusion_script):
    cmd = [python_exe, str(fusion_script)]
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload the Arduino sketch if possible, then run fusion recognition."
    )
    parser.add_argument(
        "--port",
        default="COM9",
        help="Arduino serial/upload port. Default: COM9",
    )
    parser.add_argument(
        "--board-fqbn",
        default="arduino:avr:mega",
        help="Arduino board FQBN. Default: arduino:avr:mega",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for the fusion script. Default: current Python",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Run fusion only without trying to upload the Arduino sketch.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not DEFAULT_FUSION_SCRIPT.exists():
        print(f"[ERROR] Fusion script not found: {DEFAULT_FUSION_SCRIPT}")
        return 1

    if not DEFAULT_SKETCH.exists():
        print(f"[ERROR] Arduino sketch not found: {DEFAULT_SKETCH}")
        return 1

    if args.skip_upload:
        print("[INFO] Upload step skipped by option.")
    else:
        arduino_cli = find_arduino_cli()
        if arduino_cli is None:
            print("[WARN] arduino-cli was not found.")
            print("[WARN] The fusion app will still run, but the board must already have sensor_send uploaded.")
            print("[HINT] Install arduino-cli, then run this launcher again to make upload automatic.")
        else:
            print(f"[INFO] Found arduino-cli: {arduino_cli}")
            ok = upload_sketch(
                arduino_cli=arduino_cli,
                sketch_path=DEFAULT_SKETCH,
                board_fqbn=args.board_fqbn,
                port=args.port,
            )
            if not ok:
                print("[WARN] Arduino upload failed. Fusion will still start.")
                print("[HINT] Check board model, COM port, and whether another program is using the port.")

    return run_fusion(args.python, DEFAULT_FUSION_SCRIPT)


if __name__ == "__main__":
    raise SystemExit(main())
