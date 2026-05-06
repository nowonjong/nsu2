import json
import os
import time
from collections import defaultdict
from datetime import datetime

import numpy as np


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def env_int(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


class RuntimeProfiler:
    def __init__(self, enabled=True, report_dir=None, max_samples=20000):
        self.enabled = bool(enabled)
        self.report_dir = report_dir or os.path.join("experiments", "runtime_profile")
        self.max_samples = int(max_samples)
        self.samples = defaultdict(list)
        self.counters = defaultdict(int)
        self.started_at = datetime.now()

    def add_ms(self, name, value_ms):
        if not self.enabled:
            return
        values = self.samples[name]
        if len(values) < self.max_samples:
            values.append(float(value_ms))

    def add_duration(self, name, start_perf, end_perf=None):
        if not self.enabled:
            return
        end_perf = time.perf_counter() if end_perf is None else end_perf
        self.add_ms(name, (end_perf - start_perf) * 1000.0)

    def inc(self, name, amount=1):
        if self.enabled:
            self.counters[name] += int(amount)

    @staticmethod
    def _stats(values):
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return None
        return {
            "count": int(arr.size),
            "mean_ms": float(np.mean(arr)),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
        }

    def build_report(self, script_name, target_fps, extra=None):
        stats = {}
        for name, values in sorted(self.samples.items()):
            item = self._stats(values)
            if item is not None:
                stats[name] = item

        loop_stats = stats.get("loop_total")
        achieved_fps = None
        if loop_stats and loop_stats["mean_ms"] > 0:
            achieved_fps = 1000.0 / loop_stats["mean_ms"]

        min_stable_fps = float((extra or {}).get("min_stable_fps", 20.0))
        max_stable_frame_ms = 1000.0 / min_stable_fps if min_stable_fps > 0 else None
        loop_p95 = loop_stats.get("p95_ms") if loop_stats else None
        if achieved_fps is None:
            runtime_status = "UNKNOWN"
            runtime_reason = "No loop timing samples were recorded."
        elif achieved_fps < min_stable_fps:
            runtime_status = "UNSTABLE"
            runtime_reason = f"Average FPS {achieved_fps:.2f} is below the {min_stable_fps:.1f}FPS minimum."
        elif max_stable_frame_ms is not None and loop_p95 is not None and loop_p95 > max_stable_frame_ms:
            runtime_status = "WARNING"
            runtime_reason = f"Average FPS is acceptable, but p95 loop time {loop_p95:.2f}ms exceeds {max_stable_frame_ms:.2f}ms."
        else:
            runtime_status = "STABLE"
            runtime_reason = f"Average FPS and p95 loop time satisfy the {min_stable_fps:.1f}FPS stability criterion."

        return {
            "script": script_name,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "target_fps": float(target_fps),
            "min_stable_fps": min_stable_fps,
            "achieved_fps_from_loop_mean": achieved_fps,
            "runtime_status": runtime_status,
            "runtime_reason": runtime_reason,
            "counters": dict(sorted(self.counters.items())),
            "stats": stats,
            "extra": extra or {},
        }

    def save(self, script_name, target_fps, extra=None):
        if not self.enabled:
            return None
        os.makedirs(self.report_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.report_dir, f"{timestamp}_{script_name}_runtime_profile.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.build_report(script_name, target_fps, extra), f, ensure_ascii=False, indent=2)
        return path


class FpsLimiter:
    def __init__(self, target_fps):
        self.target_fps = float(target_fps)
        self.period_sec = 1.0 / self.target_fps if self.target_fps > 0 else 0.0

    def sleep_remaining(self, loop_start_perf):
        if self.period_sec <= 0:
            return 0.0
        elapsed = time.perf_counter() - loop_start_perf
        remaining = self.period_sec - elapsed
        if remaining > 0:
            time.sleep(remaining)
            return remaining * 1000.0
        return 0.0
