#!/usr/bin/env python3
"""Measure usable webcam modes for CameraDrive AI without starting the HUD."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from camera_imaging import (
    LowLightEnhancer,
    actual_backend_name,
    apply_camera_controls,
    compute_light_metrics,
)


@dataclass(frozen=True)
class Result:
    requested_width: int
    requested_height: int
    requested_fps: int
    actual_width: int
    actual_height: int
    reported_fps: float
    measured_fps: float
    successful_frames: int
    backend: str
    median_luma: float
    p10_luma: float
    dark_fraction: float
    sharpness: float
    corner_count: float
    unique_ratio: float
    p95_interval_ms: float


def backend_candidates(name: str) -> list[Optional[int]]:
    if sys.platform != "win32":
        return [None]
    mapping = {
        "dshow": getattr(cv2, "CAP_DSHOW", None),
        "msmf": getattr(cv2, "CAP_MSMF", None),
        "any": None,
    }
    preferred = mapping.get(name.strip().lower(), mapping["dshow"])
    output: list[Optional[int]] = []
    for value in (
        preferred,
        getattr(cv2, "CAP_DSHOW", None),
        getattr(cv2, "CAP_MSMF", None),
        None,
    ):
        if value not in output:
            output.append(value)
    return output


def open_camera(index: int, backend_name: str) -> cv2.VideoCapture:
    for backend in backend_candidates(backend_name):
        capture = (
            cv2.VideoCapture(index)
            if backend is None
            else cv2.VideoCapture(index, int(backend))
        )
        if capture.isOpened():
            return capture
        capture.release()
    raise RuntimeError(f"Could not open camera index {index}")


def _mean(values: list[float], fallback: float = 0.0) -> float:
    return fallback if not values else float(sum(values) / len(values))


def rotate_for_metrics(frame: np.ndarray, degrees: int) -> np.ndarray:
    normalized = int(degrees) % 360
    if normalized == 0:
        return frame
    if normalized == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if normalized == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if normalized == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("camera_rotation_degrees must be 0, 90, 180, or 270")


def test_mode(
    config: dict,
    width: int,
    height: int,
    fps: int,
    warmup_seconds: float = 0.35,
    measure_seconds: float = 1.75,
) -> Result:
    capture = open_camera(
        int(config.get("camera_index", 0)),
        str(config.get("camera_backend", "dshow")),
    )
    try:
        if bool(config.get("camera_use_mjpg", True)):
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        mode_config = copy.deepcopy(config)
        # The benchmark controls its short post-control settling interval.
        mode_config["camera_warmup_seconds"] = min(
            1.5, float(config.get("camera_warmup_seconds", 1.0))
        )
        apply_camera_controls(capture, mode_config)
        warmup_deadline = time.perf_counter() + max(0.0, warmup_seconds)
        while time.perf_counter() < warmup_deadline:
            capture.read()

        frames = 0
        samples: list[np.ndarray] = []
        signatures: list[bytes] = []
        intervals: list[float] = []
        started = time.perf_counter()
        previous_at: Optional[float] = None
        next_sample_at = started
        sample_interval = max(0.08, measure_seconds / 10.0)
        deadline = started + measure_seconds
        while time.perf_counter() < deadline:
            ok, frame = capture.read()
            captured_at = time.perf_counter()
            if not ok or frame is None:
                continue
            frames += 1
            if previous_at is not None:
                intervals.append(captured_at - previous_at)
            previous_at = captured_at
            gray_small = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                (64, 48),
                interpolation=cv2.INTER_AREA,
            )
            signatures.append(gray_small.tobytes())
            if captured_at >= next_sample_at:
                samples.append(frame.copy())
                next_sample_at += sample_interval

        elapsed = max(1e-6, time.perf_counter() - started)
        enhancer = LowLightEnhancer(config)
        rotated_samples = [
            rotate_for_metrics(
                frame,
                int(config.get("camera_rotation_degrees", 0)),
            )
            for frame in samples
        ]
        raw_metrics = [compute_light_metrics(frame) for frame in rotated_samples]
        enhanced_metrics = [
            compute_light_metrics(enhancer.apply(frame)) for frame in rotated_samples
        ]
        p95_interval = (
            float(np.percentile(np.asarray(intervals), 95.0) * 1000.0)
            if intervals
            else 0.0
        )
        unique_ratio = (
            len(set(signatures)) / max(1.0, float(len(signatures)))
            if signatures
            else 0.0
        )
        return Result(
            requested_width=width,
            requested_height=height,
            requested_fps=fps,
            actual_width=int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)),
            actual_height=int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)),
            reported_fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            measured_fps=frames / elapsed,
            successful_frames=frames,
            backend=actual_backend_name(capture),
            median_luma=_mean([item.median_luma for item in raw_metrics]),
            p10_luma=_mean([item.p10_luma for item in raw_metrics]),
            dark_fraction=_mean([item.dark_fraction for item in raw_metrics], 1.0),
            sharpness=_mean([item.sharpness for item in enhanced_metrics]),
            corner_count=_mean(
                [float(item.corner_count) for item in enhanced_metrics]
            ),
            unique_ratio=unique_ratio,
            p95_interval_ms=p95_interval,
        )
    finally:
        capture.release()
        time.sleep(0.15)


def result_score(result: Result, maximum_fps: float, maximum_sharpness: float) -> float:
    fps_score = min(1.0, result.measured_fps / max(1.0, maximum_fps))
    light_score = min(1.0, max(0.0, (result.median_luma - 10.0) / 70.0))
    light_score *= 1.0 - 0.65 * min(1.0, result.dark_fraction)
    corner_score = min(1.0, result.corner_count / 45.0)
    sharpness_score = min(1.0, result.sharpness / max(1.0, maximum_sharpness))
    expected_interval_ms = 1000.0 / max(1.0, result.measured_fps)
    cadence_score = min(
        1.0,
        expected_interval_ms / max(expected_interval_ms, result.p95_interval_ms),
    )
    return (
        0.32 * fps_score
        + 0.25 * light_score
        + 0.20 * corner_score
        + 0.15 * sharpness_score
        + 0.04 * cadence_score
        + 0.04 * min(1.0, result.unique_ratio)
    )


def select_best_result(results: list[Result]) -> Result:
    if not results:
        raise ValueError("No camera results were supplied")
    maximum_fps = max(item.measured_fps for item in results)
    maximum_sharpness = max(item.sharpness for item in results)
    viable = [item for item in results if item.successful_frames >= 8]
    candidates = viable or results
    return max(
        candidates,
        key=lambda item: (
            result_score(item, maximum_fps, maximum_sharpness),
            item.measured_fps,
            min(item.actual_width, 1280) * min(item.actual_height, 720),
        ),
    )


def selected_fps(result: Result) -> int:
    reported = float(result.reported_fps)
    requested = int(result.requested_fps)
    if reported > 1.0 and abs(reported - requested) > max(5.0, requested * 0.15):
        return max(1, int(round(reported)))
    return requested


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure usable webcam modes for CameraDrive AI."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Save the best usable mode to config.json after making a backup",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    base = Path(__file__).resolve().parent
    config_path = base / "config.json"
    if not config_path.exists():
        print("config.json was not found.")
        return 2
    config = json.loads(config_path.read_text(encoding="utf-8"))

    modes = [
        (640, 360, 240),
        (640, 480, 240),
        (640, 480, 144),
        (640, 480, 120),
        (960, 540, 120),
        (1280, 720, 120),
        (640, 480, 100),
        (640, 480, 90),
        (640, 480, 60),
        (960, 540, 60),
        (1280, 720, 60),
        (640, 480, 50),
        (960, 540, 50),
        (1280, 720, 50),
        (640, 480, 30),
    ]
    print("CameraDrive camera quality/FPS test")
    print("Keep both legs and feet visible under normal playing light.\n")

    results: list[Result] = []
    for width, height, fps in modes:
        print(f"Testing {width}x{height} at requested {fps} FPS ...", flush=True)
        try:
            result = test_mode(config, width, height, fps)
        except Exception as exc:
            print(f"  failed: {exc}")
            continue
        results.append(result)
        print(
            f"  actual {result.actual_width}x{result.actual_height}; "
            f"reported {result.reported_fps:0.1f}; measured {result.measured_fps:0.1f} FPS; "
            f"light {result.median_luma:0.0f}; dark {result.dark_fraction * 100:0.0f}%; "
            f"corners {result.corner_count:0.0f}"
        )

    if not results:
        print("\nNo camera mode produced frames.")
        return 1

    best = select_best_result(results)
    max_fps = max(item.measured_fps for item in results)
    max_sharpness = max(item.sharpness for item in results)
    recommended_fps = selected_fps(best)
    report_lines = [
        "CameraDrive AI camera quality benchmark",
        "",
        *[
            (
                f"requested {r.requested_width}x{r.requested_height}@{r.requested_fps}: "
                f"actual {r.actual_width}x{r.actual_height}, backend {r.backend}, "
                f"reported {r.reported_fps:0.1f}, measured {r.measured_fps:0.1f} FPS, "
                f"p95 interval {r.p95_interval_ms:0.1f} ms, luma {r.median_luma:0.1f}, "
                f"dark {r.dark_fraction * 100:0.1f}%, sharpness {r.sharpness:0.1f}, "
                f"corners {r.corner_count:0.1f}, unique {r.unique_ratio * 100:0.1f}%, "
                f"score {result_score(r, max_fps, max_sharpness):0.3f}"
            )
            for r in results
        ],
        "",
        "Recommended usable mode:",
        f'  "capture_width": {best.actual_width},',
        f'  "capture_height": {best.actual_height},',
        f'  "capture_fps": {recommended_fps}',
        "",
        "Selection balances delivered FPS with light, texture, sharpness, and frame uniqueness.",
    ]
    if best.median_luma < 25.0 or best.dark_fraction > 0.55:
        report_lines.append(
            "Warning: every useful mode is still very dark; add diffuse light or reduce FPS."
        )
    report = "\n".join(report_lines) + "\n"
    (base / "camera_fps_results.txt").write_text(report, encoding="utf-8")
    print("\n" + report)

    if args.apply:
        selected_width = best.actual_width or best.requested_width
        selected_height = best.actual_height or best.requested_height
        backup_path = base / "config.before-camera-benchmark.json"
        if not backup_path.exists():
            shutil.copy2(config_path, backup_path)
        config["capture_width"] = int(selected_width)
        config["capture_height"] = int(selected_height)
        config["capture_fps"] = int(recommended_fps)
        temporary_path = base / "config.json.camera-test.tmp"
        temporary_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(config_path)
        print(
            "Applied the best usable mode to config.json: "
            f"{selected_width}x{selected_height} at {recommended_fps} FPS"
        )
        print("The previous configuration is in config.before-camera-benchmark.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
