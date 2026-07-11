#!/usr/bin/env python3
"""Camera controls and deterministic low-light imaging for CameraDrive AI."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class LightMetrics:
    median_luma: float
    p10_luma: float
    p90_luma: float
    dark_fraction: float
    sharpness: float
    corner_count: int

    @property
    def usable(self) -> bool:
        return self.median_luma >= 18.0 and self.corner_count >= 4


@dataclass(frozen=True)
class CameraControlReport:
    backend: str
    exposure_mode: str
    exposure: Optional[float]
    gain: Optional[float]
    brightness: Optional[float]
    messages: tuple[str, ...]


def _to_bgr_uint8(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim != 3:
        raise ValueError("Camera frame must be grayscale, BGR, or BGRA")
    if array.shape[2] == 3:
        return np.ascontiguousarray(array)
    if array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    raise ValueError("Camera frame must have one, three, or four channels")


class LowLightEnhancer:
    """Apply one stable photometric transform to AI and optical-flow frames.

    Gamma is fixed and CLAHE uses fixed limits. CLAHE still responds to each
    frame's local histogram, so the important invariant is that enhancement is
    applied exactly once and the identical result feeds AI and optical flow.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("low_light_enhancement_enabled", True))
        self.gamma = float(config.get("low_light_gamma", 0.72))
        self.clahe_clip_limit = float(
            config.get("low_light_clahe_clip_limit", 1.6)
        )
        self.clahe_grid_size = int(config.get("low_light_clahe_grid_size", 8))
        self.gamma_lut: Optional[np.ndarray] = None
        self.clahe: Optional[Any] = None

        if self.enabled and abs(self.gamma - 1.0) > 1e-6:
            values = np.arange(256, dtype=np.float64) / 255.0
            self.gamma_lut = np.clip(
                np.rint(255.0 * np.power(values, self.gamma)), 0, 255
            ).astype(np.uint8)
        if self.enabled and self.clahe_clip_limit > 0.0:
            grid = max(2, self.clahe_grid_size)
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=(grid, grid),
            )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        source = _to_bgr_uint8(frame)
        if not self.enabled or (self.gamma_lut is None and self.clahe is None):
            return source

        output = cv2.LUT(source, self.gamma_lut) if self.gamma_lut is not None else source.copy()
        if self.clahe is not None:
            ycrcb = cv2.cvtColor(output, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = self.clahe.apply(ycrcb[:, :, 0])
            output = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return np.ascontiguousarray(output)


def compute_light_metrics(frame: np.ndarray) -> LightMetrics:
    bgr = _to_bgr_uint8(frame)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # CameraDrive supports overhead, side-on, and rotated bed cameras, so there
    # is no reliable image-space "lower" region. Sample the full frame instead
    # of accidentally grading bedding while the legs are elsewhere.
    roi = gray
    target_width = min(192, max(32, roi.shape[1]))
    if roi.shape[1] != target_width:
        scale = target_width / max(1, roi.shape[1])
        roi = cv2.resize(
            roi,
            (target_width, max(16, int(round(roi.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    median = float(np.median(roi))
    p10, p90 = (float(value) for value in np.percentile(roi, (10.0, 90.0)))
    dark_fraction = float(np.mean(roi < 24))
    softened = cv2.GaussianBlur(roi, (3, 3), 0.65)
    sharpness = float(cv2.Laplacian(softened, cv2.CV_64F).var())
    corners = cv2.goodFeaturesToTrack(
        softened,
        maxCorners=80,
        qualityLevel=0.012,
        minDistance=4.0,
        blockSize=5,
        useHarrisDetector=False,
    )
    corner_count = 0 if corners is None else int(len(corners))
    return LightMetrics(
        median_luma=median,
        p10_luma=p10,
        p90_luma=p90,
        dark_fraction=dark_fraction,
        sharpness=sharpness,
        corner_count=corner_count,
    )


def actual_backend_name(capture: Any) -> str:
    try:
        name = str(capture.getBackendName()).strip()
        return name or "unknown"
    except Exception:
        return "unknown"


def _finite_capture_value(capture: Any, prop: int) -> Optional[float]:
    try:
        value = float(capture.get(prop))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _set_capture_value(
    capture: Any,
    prop: int,
    value: float,
    label: str,
    messages: list[str],
) -> bool:
    try:
        accepted = bool(capture.set(prop, float(value)))
    except Exception as exc:
        messages.append(f"{label} failed: {exc}")
        return False
    effective = _finite_capture_value(capture, prop)
    suffix = "unknown readback" if effective is None else f"readback {effective:g}"
    messages.append(
        f"{label} requested {float(value):g}; {suffix}; "
        + ("accepted" if accepted else "driver rejected")
    )
    return accepted


def _warm_camera(
    capture: Any,
    seconds: float,
    stop_event: Optional[threading.Event],
) -> None:
    deadline = time.monotonic() + max(0.0, float(seconds))
    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            return
        ok, frame = capture.read()
        if not ok or frame is None:
            time.sleep(0.005)


def apply_camera_controls(
    capture: Any,
    config: dict[str, Any],
    stop_event: Optional[threading.Event] = None,
) -> CameraControlReport:
    """Best-effort camera tuning; unsupported properties are never fatal."""

    backend = actual_backend_name(capture)
    backend_lower = backend.lower()
    requested_mode = str(
        config.get("camera_exposure_mode", "auto_lock")
    ).strip().lower()
    effective_mode = "unchanged"
    warmup = max(0.0, float(config.get("camera_warmup_seconds", 1.0)))
    messages: list[str] = [f"backend {backend}"]

    # Apply controls that change the brightness/focus operating point before
    # auto exposure is measured and locked. Many drivers ignore manual focus
    # until autofocus has first been disabled.
    for key, prop, label in (
        ("camera_gain", cv2.CAP_PROP_GAIN, "gain"),
        ("camera_brightness", cv2.CAP_PROP_BRIGHTNESS, "brightness"),
    ):
        value = config.get(key)
        if value is not None:
            _set_capture_value(capture, prop, float(value), label, messages)

    autofocus = config.get("camera_autofocus")
    focus = config.get("camera_focus")
    if autofocus is not None:
        _set_capture_value(
            capture,
            cv2.CAP_PROP_AUTOFOCUS,
            1.0 if bool(autofocus) else 0.0,
            "autofocus",
            messages,
        )
    elif focus is not None:
        _set_capture_value(
            capture,
            cv2.CAP_PROP_AUTOFOCUS,
            0.0,
            "autofocus for manual focus",
            messages,
        )
    if focus is not None:
        if autofocus is True:
            messages.append("manual focus skipped because autofocus is enabled")
        else:
            _set_capture_value(
                capture, cv2.CAP_PROP_FOCUS, float(focus), "focus", messages
            )

    # DirectShow exposes its auto/manual switch as 0.75/0.25. Other backends
    # differ, so never assume those values when the opened backend is not DSHOW.
    is_dshow = "dshow" in backend_lower
    if requested_mode == "auto":
        if is_dshow:
            _set_capture_value(
                capture, cv2.CAP_PROP_AUTO_EXPOSURE, 0.75, "auto exposure", messages
            )
            effective_mode = "auto"
        _warm_camera(capture, warmup, stop_event)
    elif requested_mode == "auto_lock":
        if is_dshow:
            auto_ok = _set_capture_value(
                capture, cv2.CAP_PROP_AUTO_EXPOSURE, 0.75, "auto exposure", messages
            )
            _warm_camera(capture, warmup, stop_event)
            auto_readback = _finite_capture_value(
                capture, cv2.CAP_PROP_AUTO_EXPOSURE
            )
            exposure = _finite_capture_value(capture, cv2.CAP_PROP_EXPOSURE)
            auto_verified = (
                auto_ok
                and auto_readback is not None
                and 0.50 <= auto_readback <= 1.05
            )
            exposure_credible = exposure is not None and abs(exposure) > 1e-6
            if auto_verified and exposure_credible:
                manual_ok = _set_capture_value(
                    capture,
                    cv2.CAP_PROP_AUTO_EXPOSURE,
                    0.25,
                    "manual exposure lock",
                    messages,
                )
                manual_readback = _finite_capture_value(
                    capture, cv2.CAP_PROP_AUTO_EXPOSURE
                )
                manual_verified = (
                    manual_ok
                    and manual_readback is not None
                    and -0.05 <= manual_readback <= 0.50
                )
                exposure_set = False
                if manual_verified:
                    exposure_set = _set_capture_value(
                        capture,
                        cv2.CAP_PROP_EXPOSURE,
                        exposure,
                        "locked exposure",
                        messages,
                    )
                if manual_verified and exposure_set:
                    effective_mode = "locked"
                else:
                    _set_capture_value(
                        capture,
                        cv2.CAP_PROP_AUTO_EXPOSURE,
                        0.75,
                        "auto exposure fallback",
                        messages,
                    )
                    effective_mode = "auto"
            else:
                messages.append(
                    "auto exposure lock unsupported or exposure readback unsafe; "
                    "leaving automatic exposure enabled"
                )
                effective_mode = "auto"
        else:
            _warm_camera(capture, warmup, stop_event)
            messages.append("auto-lock skipped because backend is not DirectShow")
    elif requested_mode == "manual":
        requested = float(config.get("camera_manual_exposure", -7.0))
        if is_dshow:
            _set_capture_value(
                capture,
                cv2.CAP_PROP_AUTO_EXPOSURE,
                0.25,
                "manual exposure mode",
                messages,
            )
        exposure_set = _set_capture_value(
            capture, cv2.CAP_PROP_EXPOSURE, requested, "manual exposure", messages
        )
        effective_mode = "manual" if exposure_set else "unchanged"
        _warm_camera(capture, min(warmup, 0.35), stop_event)
    else:  # unchanged
        _warm_camera(capture, warmup, stop_event)

    return CameraControlReport(
        backend=backend,
        exposure_mode=effective_mode,
        exposure=_finite_capture_value(capture, cv2.CAP_PROP_EXPOSURE),
        gain=_finite_capture_value(capture, cv2.CAP_PROP_GAIN),
        brightness=_finite_capture_value(capture, cv2.CAP_PROP_BRIGHTNESS),
        messages=tuple(messages),
    )
