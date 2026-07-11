#!/usr/bin/env python3
"""Measure practical webcam modes for CameraDrive AI without starting the HUD."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2


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


def test_mode(
    index: int,
    backend_name: str,
    width: int,
    height: int,
    fps: int,
    use_mjpg: bool,
    warmup_seconds: float = 0.55,
    measure_seconds: float = 1.35,
) -> Result:
    capture = open_camera(index, backend_name)
    try:
        if use_mjpg:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        warmup_deadline = time.perf_counter() + warmup_seconds
        while time.perf_counter() < warmup_deadline:
            capture.read()

        frames = 0
        started = time.perf_counter()
        deadline = started + measure_seconds
        while time.perf_counter() < deadline:
            ok, frame = capture.read()
            if ok and frame is not None:
                frames += 1
        elapsed = max(1e-6, time.perf_counter() - started)
        return Result(
            requested_width=width,
            requested_height=height,
            requested_fps=fps,
            actual_width=int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)),
            actual_height=int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)),
            reported_fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            measured_fps=frames / elapsed,
            successful_frames=frames,
        )
    finally:
        capture.release()
        time.sleep(0.20)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure webcam modes for CameraDrive AI."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Save the fastest measured mode to config.json after making a backup",
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
    camera_index = int(config.get("camera_index", 0))
    backend_name = str(config.get("camera_backend", "dshow"))
    use_mjpg = bool(config.get("camera_use_mjpg", True))

    # Probe high-rate MJPEG modes first. Unsupported requests usually fall back
    # to a lower mode; selection is based on measured delivery, not the request.
    modes = [
        (640, 360, 240),
        (640, 480, 240),
        (640, 480, 144),
        (640, 480, 120),
        (960, 540, 120),
        (1280, 720, 120),
        (640, 480, 90),
        (640, 480, 60),
        (960, 540, 60),
        (1280, 720, 60),
        (640, 480, 30),
    ]
    print("CameraDrive camera FPS test")
    print("Close CameraDrive, browsers, chat apps, and other camera software first.\n")

    results: list[Result] = []
    for width, height, fps in modes:
        print(f"Testing {width}x{height} at requested {fps} FPS ...", flush=True)
        try:
            result = test_mode(
                camera_index,
                backend_name,
                width,
                height,
                fps,
                use_mjpg,
            )
        except Exception as exc:
            print(f"  failed: {exc}")
            continue
        results.append(result)
        print(
            "  actual "
            f"{result.actual_width}x{result.actual_height}; "
            f"driver reports {result.reported_fps:0.1f}; "
            f"measured {result.measured_fps:0.1f} FPS"
        )

    if not results:
        print("\nNo camera mode produced frames.")
        return 1

    # Prefer measured frame rate, then useful foot-model resolution.
    best = max(
        results,
        key=lambda item: (
            item.measured_fps,
            min(item.actual_width, 1280) * min(item.actual_height, 720),
        ),
    )
    report_lines = [
        "CameraDrive AI camera benchmark",
        "",
        *[
            (
                f"requested {r.requested_width}x{r.requested_height}@{r.requested_fps}: "
                f"actual {r.actual_width}x{r.actual_height}, "
                f"reported {r.reported_fps:0.1f}, measured {r.measured_fps:0.1f} FPS"
            )
            for r in results
        ],
        "",
        "Recommended measured mode:",
        f'  "capture_width": {best.actual_width},',
        f'  "capture_height": {best.actual_height},',
        f'  "capture_fps": {best.requested_fps}',
        "",
        "The measured result is more trustworthy than the driver-reported value.",
    ]
    report = "\n".join(report_lines) + "\n"
    (base / "camera_fps_results.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print("Results were saved to camera_fps_results.txt.")

    if args.apply:
        selected_width = best.actual_width if best.actual_width > 0 else best.requested_width
        selected_height = best.actual_height if best.actual_height > 0 else best.requested_height
        backup_path = base / "config.before-camera-benchmark.json"
        if not backup_path.exists():
            shutil.copy2(config_path, backup_path)
        config["capture_width"] = int(selected_width)
        config["capture_height"] = int(selected_height)
        config["capture_fps"] = int(best.requested_fps)
        temporary_path = base / "config.json.camera-test.tmp"
        temporary_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(config_path)
        print(
            "Applied the fastest measured mode to config.json: "
            f"{selected_width}x{selected_height} at requested {best.requested_fps} FPS"
        )
        print("The previous configuration is in config.before-camera-benchmark.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
