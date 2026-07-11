#!/usr/bin/env python3
"""CameraDrive AI

A Windows accessibility/game-control prototype that maps webcam pose landmarks to
an Xbox 360 virtual controller. The default low-latency mode uses only the feet:

* right foot movement -> accelerator (right trigger)
* left foot movement  -> brake (left trigger)
* steering remains centred and can come from another controller or keyboard

An optional full-controls mode restores the two-hand steering gesture. The
application never injects code into the game. It presents an ordinary virtual
XInput controller and a click-through transparent HUD.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import json
import math
import os
import sys
import threading
import time
import traceback
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from camera_imaging import (
    CameraControlReport,
    LightMetrics,
    LowLightEnhancer,
    apply_camera_controls,
    compute_light_metrics,
)


APP_NAME = "CameraDrive AI"
APP_VERSION = "0.7.0"

MODEL_URLS = {
    "lite": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
    "full": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/1/pose_landmarker_full.task"
    ),
    "heavy": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    ),
}

# MediaPipe Pose landmark indices.
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# Manual hard-lock order. The right accelerator foot is placed first so the
# primary control is usable as quickly as possible.
MANUAL_ANCHOR_SEQUENCE: tuple[tuple[int, str], ...] = (
    (RIGHT_HEEL, "RIGHT HEEL — accelerator pivot"),
    (RIGHT_ANKLE, "RIGHT ANKLE — triangle reference"),
    (RIGHT_FOOT_INDEX, "RIGHT TOE / FOREFOOT"),
    (LEFT_HEEL, "LEFT HEEL — brake pivot"),
    (LEFT_ANKLE, "LEFT ANKLE — triangle reference"),
    (LEFT_FOOT_INDEX, "LEFT TOE / FOREFOOT"),
)
MANUAL_FOOT_TRIANGLES: dict[str, tuple[int, int, int]] = {
    "left": (LEFT_HEEL, LEFT_ANKLE, LEFT_FOOT_INDEX),
    "right": (RIGHT_HEEL, RIGHT_ANKLE, RIGHT_FOOT_INDEX),
}
MANUAL_FOOT_ANCHOR_INDICES = frozenset(
    index for indices in MANUAL_FOOT_TRIANGLES.values() for index in indices
)

POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL),
    (LEFT_HEEL, LEFT_FOOT_INDEX),
    (LEFT_ANKLE, LEFT_FOOT_INDEX),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (RIGHT_ANKLE, RIGHT_HEEL),
    (RIGHT_HEEL, RIGHT_FOOT_INDEX),
    (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
)


DEFAULT_CONFIG: dict[str, Any] = {
    "config_version": 7,
    # Feet-only is the default accessibility mode. The virtual left stick stays
    # centred so steering can come from a keyboard, wheel, or physical gamepad.
    "control_mode": "pedals_only",
    "camera_index": 0,
    # Request a common high-speed UVC mode. The driver may negotiate 120, 90,
    # 60, or 30 FPS; the HUD reports both camera and processed tracking rates.
    "capture_width": 640,
    "capture_height": 480,
    "capture_fps": 120,
    "camera_use_mjpg": True,
    "camera_backend": "dshow",
    # Camera controls are best-effort because property ranges depend on the
    # webcam and backend. DirectShow auto_lock lets exposure settle, then locks
    # it so frame-to-frame brightness changes do not confuse optical flow.
    "camera_exposure_mode": "auto_lock",
    "camera_manual_exposure": -7.0,
    "camera_gain": None,
    "camera_brightness": None,
    "camera_autofocus": None,
    "camera_focus": None,
    "camera_warmup_seconds": 1.0,
    # A single deterministic transform is shared by MediaPipe and optical flow.
    # Gamma below 1 brightens shadows; CLAHE raises local lower-leg contrast.
    "low_light_enhancement_enabled": True,
    "low_light_gamma": 0.72,
    "low_light_clahe_clip_limit": 1.6,
    "low_light_clahe_grid_size": 8,
    # Optical flow runs at tracking_width on every available camera frame. The
    # Full pose model runs independently at inference_width and only re-anchors
    # the semantic knee/ankle/heel/toe identities.
    "tracking_width": 640,
    "inference_width": 512,
    "ai_max_fps": 30.0,
    "camera_rotation_degrees": 0,
    "mirror_preview": True,
    "monitor_index": 0,
    "show_camera_preview": True,
    "preview_max_fps": 15,
    "auto_hide_preview_when_active": True,
    "model": "full",
    "auto_calibrate": True,
    "swap_pedals": False,
    "invert_steering": False,
    "pose_detection_confidence": 0.26,
    "pose_presence_confidence": 0.24,
    "pose_tracking_confidence": 0.24,
    "landmark_confidence": 0.21,
    "ai_anchor_max_age_seconds": 0.42,
    # Keep MediaPipe's left/right labels stable when a fresh result briefly
    # swaps two similar-looking feet.
    "foot_identity_lock": True,
    "foot_identity_swap_margin_pixels": 22.0,
    # The coherent leg lock tracks each complete hip-to-toe chain. MediaPipe's
    # person mask is used only as a soft foreground cue, so a weak mask cannot
    # make a valid foot disappear.
    "enable_segmentation_mask": True,
    "segmentation_threshold": 0.16,
    "segmentation_dilate_pixels": 13,
    "leg_lock_enabled": True,
    "leg_lock_feature_count": 34,
    "leg_lock_feature_quality": 0.006,
    "leg_lock_feature_min_distance_pixels": 5.0,
    "leg_lock_line_thickness_pixels": 24,
    "leg_lock_roi_padding_pixels": 22,
    "leg_lock_min_inliers": 6,
    "leg_lock_ransac_threshold_pixels": 2.4,
    "leg_lock_max_scale_change": 0.12,
    "leg_lock_max_rotation_degrees": 14.0,
    "leg_lock_max_translation_pixels": 30.0,
    "leg_lock_landmark_blend": 0.24,
    "leg_lock_outlier_pixels": 12.0,
    "leg_lock_anchor_jump_pixels": 26.0,
    "leg_lock_bone_length_tolerance": 0.34,
    "leg_lock_reacquire_anchors": 3,
    # Camera-rate, forward/backward-validated patch-cloud optical flow.
    "enable_optical_flow": True,
    "optical_flow_hold_seconds": 0.34,
    "optical_flow_anchor_blend": 0.20,
    "optical_flow_anchor_deadband_pixels": 1.1,
    "optical_flow_max_fb_error_pixels": 1.45,
    "optical_flow_anchor_max_distance_pixels": 48.0,
    "optical_flow_patch_radius_pixels": 4.0,
    "optical_flow_patch_grid_size": 3,
    "optical_flow_min_patch_votes": 4,
    "optical_flow_window_pixels": 15,
    "optical_flow_max_level": 1,
    "optical_flow_validation_interval": 6,
    "optical_flow_use_roi": True,
    "optical_flow_roi_padding_pixels": 48,
    "feature_hold_seconds": 0.12,
    # World-coordinate cues arrive only at AI rate, so the micro-pedal profile
    # excludes them from trigger projection. AI still supplies semantic labels.
    "pedal_world_feature_weight": 0.0,
    # Minimal filtering: stable calibration noise suppression plus an almost
    # direct camera-rate path for deliberate micro-movement.
    "feature_filter_min_cutoff_hz": 24.0,
    "feature_filter_beta": 3.0,
    "feature_filter_derivative_cutoff_hz": 4.0,
    "pedal_raw_feature_blend": 0.90,
    # Legacy common values remain for compatibility with third-party configs.
    "pedal_sensitivity": 1.0,
    "pedal_deadzone": 0.004,
    "pedal_curve_exponent": 1.0,
    # Calibration defines the smallest comfortable action as full travel. The
    # endpoint-safe boost makes initial response visible without collapsing the
    # useful trigger range into the first few percent of foot movement.
    "throttle_sensitivity": 1.0,
    "brake_sensitivity": 1.0,
    "throttle_deadzone": 0.004,
    "brake_deadzone": 0.005,
    "throttle_curve_exponent": 1.0,
    "brake_curve_exponent": 1.0,
    "throttle_response_boost": 0.55,
    "brake_response_boost": 0.35,
    "throttle_initial_response": 0.0,
    "brake_initial_response": 0.0,
    # Accept a press that follows the calibrated articulation at a different
    # comfortable foot angle, while rejecting unrelated sideways motion.
    "pedal_min_feature_coverage": 0.35,
    "pedal_direction_tolerance_degrees": 55.0,
    "pedal_magnitude_blend": 0.25,
    "foot_core_min_confidence": 0.24,
    # The fixed-heel workflow requires six user-selected points every session.
    # Manual points are followed only by strict local optical flow; MediaPipe
    # may identify the person but can never overwrite a confirmed point.
    "manual_anchor_required": True,
    "manual_anchor_min_separation_pixels": 6.0,
    "manual_anchor_patch_grid_size": 5,
    "manual_anchor_min_patch_votes": 9,
    "manual_anchor_template_size_pixels": 17,
    "manual_anchor_template_search_pixels": 7,
    "manual_anchor_template_min_score": 0.38,
    "manual_anchor_template_rotation_degrees": 14.0,
    "manual_triangle_max_edge_change_ratio": 0.22,
    "calibration_min_heel_tilt_degrees": 2.0,
    "heel_extension_weight": 0.15,
    # A small, bounded rising-edge prediction offsets one camera frame of delay;
    # release is never predicted and still snaps to zero.
    "pedal_lookahead_seconds": 0.012,
    "pedal_prediction_min_delta": 0.0008,
    "pedal_prediction_max_advance": 0.035,
    "pedal_noise_floor": 0.0010,
    "pedal_noise_multiplier": 1.20,
    "pedal_noise_deadzone_factor": 0.020,
    "calibration_min_signal_to_noise": 1.8,
    "calibration_min_absolute_motion": 0.00045,
    "steering_deadzone": 0.045,
    "steering_curve_exponent": 0.95,
    "steering_sensitivity": 1.20,
    "calibration_min_steering_degrees": 4.0,
    "smoothing_tau_seconds": 0.035,
    "steering_smoothing_tau_seconds": 0.035,
    # Pedal output has no attack smoothing. Release is almost immediate and a
    # true neutral value snaps exactly to zero.
    "pedal_attack_tau_seconds": 0.0,
    "pedal_release_tau_seconds": 0.003,
    "pedal_release_snap_to_zero": True,
    "lost_tracking_timeout_seconds": 0.18,
    "camera_start_timeout_seconds": 8.0,
    "ai_start_timeout_seconds": 15.0,
    "camera_frame_timeout_seconds": 1.0,
    "calibration_prepare_seconds": 0.50,
    "calibration_capture_seconds": 0.90,
    "calibration_min_samples": 20,
    "preview_width": 384,
    "preview_height": 216,
    "opencv_threads": 2,
    "prefer_low_latency_thread_priorities": True,
}


# Runtime profiles only override performance-related values. Camera index,
# rotation, monitor, pedal swapping, and user layout choices remain untouched.
PERFORMANCE_PROFILES: dict[str, dict[str, Any]] = {
    # Recommended hybrid: Full AI labels at 512 px, 640 px camera-rate
    # optical flow inside a lower-body ROI. This protects foot identity while
    # keeping the pedal loop independent from inference latency.
    "micro-pedals": {
        "model": "full",
        "tracking_width": 640,
        "inference_width": 576,
        "ai_max_fps": 30.0,
        "preview_max_fps": 15,
        "auto_hide_preview_when_active": True,
        "optical_flow_patch_radius_pixels": 4.0,
        "optical_flow_min_patch_votes": 4,
        "optical_flow_window_pixels": 15,
        "optical_flow_max_level": 1,
        "optical_flow_validation_interval": 6,
        "optical_flow_use_roi": True,
        "optical_flow_roi_padding_pixels": 48,
        "leg_lock_enabled": True,
        "leg_lock_feature_count": 34,
        "enable_segmentation_mask": True,
        "opencv_threads": 2,
    },
    "balanced": {
        "model": "full",
        "tracking_width": 512,
        "inference_width": 512,
        "ai_max_fps": 30.0,
        "preview_max_fps": 12,
        "auto_hide_preview_when_active": True,
        "optical_flow_patch_radius_pixels": 3.5,
        "optical_flow_min_patch_votes": 4,
        "optical_flow_window_pixels": 13,
        "optical_flow_max_level": 1,
        "optical_flow_validation_interval": 6,
        "optical_flow_use_roi": True,
        "leg_lock_enabled": True,
        "leg_lock_feature_count": 28,
        "enable_segmentation_mask": True,
        "opencv_threads": 2,
    },
    # Designed for cameras that genuinely deliver 90/120 FPS. Semantic AI
    # anchors use the Lite model, while micro-motion remains camera-rate.
    "max-fps": {
        "model": "lite",
        "tracking_width": 448,
        "inference_width": 448,
        "ai_max_fps": 36.0,
        "preview_max_fps": 10,
        "auto_hide_preview_when_active": True,
        "optical_flow_patch_radius_pixels": 3.0,
        "optical_flow_min_patch_votes": 4,
        "optical_flow_window_pixels": 11,
        "optical_flow_max_level": 0,
        "optical_flow_validation_interval": 8,
        "optical_flow_use_roi": True,
        "leg_lock_enabled": True,
        "leg_lock_feature_count": 22,
        "enable_segmentation_mask": False,
        "opencv_threads": 2,
    },
    # Uses a larger Full-model image and a wider LK search. This is useful when
    # feet are small in frame or semantic anchors occasionally land incorrectly.
    "foot-accuracy": {
        "model": "full",
        "tracking_width": 640,
        "inference_width": 640,
        "ai_max_fps": 24.0,
        "preview_max_fps": 12,
        "auto_hide_preview_when_active": True,
        "optical_flow_patch_radius_pixels": 4.0,
        "optical_flow_min_patch_votes": 4,
        "optical_flow_window_pixels": 15,
        "optical_flow_max_level": 1,
        "optical_flow_validation_interval": 5,
        "optical_flow_use_roi": True,
        "leg_lock_enabled": True,
        "leg_lock_feature_count": 40,
        "leg_lock_line_thickness_pixels": 28,
        "enable_segmentation_mask": True,
        "opencv_threads": 2,
    },
    # Strongest lower-body identity and geometry lock.  It keeps the 640-pixel
    # camera-rate path, gives the Full model more detail, and uses more
    # foreground features per leg.  This is the recommended diagnostic profile
    # when a knee/ankle/toe point occasionally jumps onto bedding.
    "leg-lock": {
        "model": "full",
        "tracking_width": 640,
        "inference_width": 704,
        "ai_max_fps": 24.0,
        "preview_max_fps": 12,
        "auto_hide_preview_when_active": True,
        "optical_flow_patch_radius_pixels": 4.0,
        "optical_flow_min_patch_votes": 4,
        "optical_flow_window_pixels": 17,
        "optical_flow_max_level": 2,
        "optical_flow_validation_interval": 4,
        "optical_flow_use_roi": True,
        "optical_flow_roi_padding_pixels": 56,
        "leg_lock_enabled": True,
        "leg_lock_feature_count": 48,
        "leg_lock_feature_quality": 0.004,
        "leg_lock_feature_min_distance_pixels": 4.0,
        "leg_lock_line_thickness_pixels": 30,
        "leg_lock_roi_padding_pixels": 26,
        "enable_segmentation_mask": True,
        "opencv_threads": 2,
    },
}

@dataclass
class FootFeatureObservation:
    """Foot geometry plus per-dimension validity in a shin-local frame."""

    values: np.ndarray
    validity: np.ndarray
    confidence: float

    def copy(self) -> "FootFeatureObservation":
        return FootFeatureObservation(
            values=self.values.copy(),
            validity=self.validity.copy(),
            confidence=float(self.confidence),
        )


@dataclass
class PoseFeatures:
    left_foot: Optional[FootFeatureObservation] = None
    right_foot: Optional[FootFeatureObservation] = None
    steering_angle: Optional[float] = None
    left_foot_ok: bool = False
    right_foot_ok: bool = False
    steering_ok: bool = False
    left_foot_fresh: bool = False
    right_foot_fresh: bool = False
    steering_fresh: bool = False
    pose_detected: bool = False
    landmarks: list[tuple[float, float, float]] = field(default_factory=list)
    # Refined 2-D keypoints. The third value is 1.0 for a current AI anchor and
    # 0.65 for a point temporarily carried by verified optical flow.
    tracked_landmarks: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    hard_locked_indices: frozenset[int] = field(default_factory=frozenset)

    @property
    def pedals_ok(self) -> bool:
        return self.left_foot_ok and self.right_foot_ok

    @property
    def pedals_fresh(self) -> bool:
        return self.left_foot_fresh and self.right_foot_fresh

    @property
    def all_controls_ok(self) -> bool:
        return self.pedals_ok and self.steering_ok

    @property
    def all_controls_fresh(self) -> bool:
        return self.pedals_fresh and self.steering_fresh


@dataclass
class CalibrationData:
    left_foot_neutral: np.ndarray
    left_foot_pressed: np.ndarray
    left_foot_noise: np.ndarray
    left_foot_reliability: np.ndarray
    left_foot_signal_to_noise: float
    right_foot_neutral: np.ndarray
    right_foot_pressed: np.ndarray
    right_foot_noise: np.ndarray
    right_foot_reliability: np.ndarray
    right_foot_signal_to_noise: float
    steering_neutral: float
    steering_left: float
    steering_right: float


@dataclass
class UiSnapshot:
    mode: str = "STARTING"
    pedals_only: bool = True
    headline: str = "Starting camera and AI tracking"
    detail: str = ""
    prompt: str = ""
    countdown: float = 0.0
    calibration_phase_index: int = 0
    calibration_phase_count: int = 5
    calibration_phase_key: str = "neutral"
    calibration_stage: str = "GET READY"
    calibration_progress: float = 0.0
    calibration_samples: int = 0
    left_foot_ok: bool = False
    right_foot_ok: bool = False
    steering_ok: bool = False
    tracking_hint: str = "Waiting for the first camera frame"
    tracking_ok: bool = False
    calibrated: bool = False
    active: bool = False
    controller_available: bool = False
    controller_message: str = "Virtual controller is starting"
    gas: float = 0.0
    brake: float = 0.0
    steering: float = 0.0
    fps: float = 0.0
    camera_fps: float = 0.0
    ai_fps: float = 0.0
    light_luma: float = 0.0
    light_dark_fraction: float = 0.0
    camera_mode: str = "Camera is starting"
    preview: Optional[QImage] = None
    show_preview: bool = True
    preview_mirrored: bool = True
    anchor_setup_active: bool = False
    anchor_setup_step: int = 0
    anchor_setup_count: int = len(MANUAL_ANCHOR_SEQUENCE)
    anchor_setup_label: str = ""
    anchor_setup_points: dict[int, tuple[float, float]] = field(default_factory=dict)
    anchor_input_available: bool = False
    fatal_error: str = ""
    exit_requested: bool = False


class SharedState:
    def __init__(self, show_preview: bool) -> None:
        self._lock = threading.RLock()
        self._snapshot = UiSnapshot(show_preview=show_preview)
        self._anchor_clicks: list[tuple[float, float]] = []

    def update(self, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                if not hasattr(self._snapshot, key):
                    raise AttributeError(f"Unknown UI state field: {key}")
                setattr(self._snapshot, key, value)

    def snapshot(self) -> UiSnapshot:
        with self._lock:
            return copy.copy(self._snapshot)

    def submit_anchor_click(self, x: float, y: float) -> None:
        with self._lock:
            self._anchor_clicks.append((float(x), float(y)))

    def take_anchor_clicks(self) -> list[tuple[float, float]]:
        with self._lock:
            clicks = self._anchor_clicks
            self._anchor_clicks = []
            return clicks

    def clear_anchor_clicks(self) -> None:
        with self._lock:
            self._anchor_clicks = []


@dataclass(frozen=True)
class FramePacket:
    sequence: int
    captured_at: float
    frame: np.ndarray
    alignment_gray: Optional[np.ndarray] = None


@dataclass(frozen=True)
class PoseAnchorPacket:
    sequence: int
    captured_at: float
    completed_at: float
    gray: np.ndarray
    landmarks: tuple[tuple[float, float, float], ...]
    world_points: dict[int, np.ndarray]
    person_mask: Optional[np.ndarray] = None

    @property
    def pose_detected(self) -> bool:
        return len(self.landmarks) >= 33


class LatestFrameBuffer:
    """A one-slot frame queue: consumers always receive the newest frame."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._packet: Optional[FramePacket] = None
        self._closed = False

    def publish(self, packet: FramePacket) -> None:
        with self._condition:
            if self._closed:
                return
            self._packet = packet
            self._condition.notify_all()

    def wait_for_new(
        self,
        last_sequence: int,
        timeout: float,
    ) -> Optional[FramePacket]:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while not self._closed:
                if self._packet is not None and self._packet.sequence > last_sequence:
                    return self._packet
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(remaining)
            if self._packet is not None and self._packet.sequence > last_sequence:
                return self._packet
            return None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class LatestPoseBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packet: Optional[PoseAnchorPacket] = None

    def publish(self, packet: PoseAnchorPacket) -> None:
        with self._lock:
            self._packet = packet

    def snapshot(self) -> Optional[PoseAnchorPacket]:
        with self._lock:
            return self._packet


class ExponentialSmoother:
    def __init__(self, tau_seconds: float, initial: float = 0.0) -> None:
        self.tau_seconds = max(0.001, float(tau_seconds))
        self.value = float(initial)
        self.initialized = False

    def reset(self, value: float = 0.0) -> None:
        self.value = float(value)
        self.initialized = False

    def update(self, target: float, dt: float) -> float:
        target = float(target)
        if not self.initialized:
            self.value = target
            self.initialized = True
            return self.value
        dt = max(0.0, min(float(dt), 0.25))
        alpha = 1.0 - math.exp(-dt / self.tau_seconds)
        self.value += alpha * (target - self.value)
        return self.value


class AsymmetricPedalSmoother:
    """Near-instant pedal attack/release with an immediate zero safety snap."""

    def __init__(
        self,
        attack_tau_seconds: float,
        release_tau_seconds: float,
        snap_zero: bool = True,
        initial: float = 0.0,
    ) -> None:
        self.attack_tau_seconds = max(0.0, float(attack_tau_seconds))
        self.release_tau_seconds = max(0.0, float(release_tau_seconds))
        self.snap_zero = bool(snap_zero)
        self.value = float(initial)
        self.initialized = False

    def reset(self, value: float = 0.0) -> None:
        self.value = clip(float(value), 0.0, 1.0)
        self.initialized = False

    def update(self, target: float, dt: float) -> float:
        target = clip(float(target), 0.0, 1.0)
        if self.snap_zero and target <= 0.0:
            self.value = 0.0
            self.initialized = True
            return 0.0
        if not self.initialized:
            self.value = target
            self.initialized = True
            return self.value
        tau = self.attack_tau_seconds if target > self.value else self.release_tau_seconds
        if tau <= 1e-5:
            self.value = target
            return self.value
        dt = clip(float(dt), 0.0, 0.25)
        alpha = 1.0 - math.exp(-dt / tau)
        self.value += alpha * (target - self.value)
        self.value = clip(self.value, 0.0, 1.0)
        return self.value


class OneEuroFilter:
    """Adaptive low-pass filter: steady points are smooth, moving points react fast."""

    def __init__(
        self,
        min_cutoff_hz: float,
        beta: float,
        derivative_cutoff_hz: float,
    ) -> None:
        self.min_cutoff_hz = max(0.01, float(min_cutoff_hz))
        self.beta = max(0.0, float(beta))
        self.derivative_cutoff_hz = max(0.01, float(derivative_cutoff_hz))
        self._raw_previous: Optional[np.ndarray] = None
        self._filtered_previous: Optional[np.ndarray] = None
        self._derivative_previous: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._raw_previous = None
        self._filtered_previous = None
        self._derivative_previous = None

    @staticmethod
    def _alpha(cutoff_hz: np.ndarray | float, dt: float) -> np.ndarray:
        cutoff = np.maximum(np.asarray(cutoff_hz, dtype=np.float64), 0.01)
        dt = clip(dt, 1e-4, 0.25)
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, value: np.ndarray, dt: float) -> np.ndarray:
        current = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(current)):
            raise ValueError("OneEuroFilter received a non-finite value")
        if self._raw_previous is None:
            self._raw_previous = current.copy()
            self._filtered_previous = current.copy()
            self._derivative_previous = np.zeros_like(current)
            return current.copy()

        assert self._filtered_previous is not None
        assert self._derivative_previous is not None
        dt = clip(dt, 1e-4, 0.25)
        derivative = (current - self._raw_previous) / dt
        derivative_alpha = self._alpha(self.derivative_cutoff_hz, dt)
        filtered_derivative = (
            derivative_alpha * derivative
            + (1.0 - derivative_alpha) * self._derivative_previous
        )
        cutoff = self.min_cutoff_hz + self.beta * np.abs(filtered_derivative)
        value_alpha = self._alpha(cutoff, dt)
        filtered = (
            value_alpha * current
            + (1.0 - value_alpha) * self._filtered_previous
        )
        self._raw_previous = current.copy()
        self._filtered_previous = filtered.copy()
        self._derivative_previous = filtered_derivative.copy()
        return filtered


class CircularOneEuroFilter:
    """One Euro filtering for an angle while preserving wraparound continuity."""

    def __init__(
        self,
        min_cutoff_hz: float,
        beta: float,
        derivative_cutoff_hz: float,
    ) -> None:
        self._filter = OneEuroFilter(min_cutoff_hz, beta, derivative_cutoff_hz)
        self._last_unwrapped: Optional[float] = None

    def reset(self) -> None:
        self._filter.reset()
        self._last_unwrapped = None

    def update(self, angle: float, dt: float) -> float:
        angle = float(angle)
        if self._last_unwrapped is None:
            unwrapped = angle
        else:
            unwrapped = self._last_unwrapped + wrap_angle(angle - self._last_unwrapped)
        self._last_unwrapped = unwrapped
        filtered = self._filter.update(np.array([unwrapped], dtype=np.float64), dt)
        return wrap_angle(float(filtered[0]))


def set_current_thread_priority(level: str) -> None:
    """Best-effort Windows thread priority without changing process priority."""
    if sys.platform != "win32" or not hasattr(ctypes, "windll"):
        return
    priorities = {
        "below_normal": -1,
        "normal": 0,
        "above_normal": 1,
    }
    value = priorities.get(str(level).strip().lower())
    if value is None:
        return
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), int(value))
    except Exception:
        pass


class CameraCaptureThread(threading.Thread):
    """Continuously drain the webcam and publish only its newest frame."""

    def __init__(
        self,
        config: dict[str, Any],
        output: LatestFrameBuffer,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="CameraCapture", daemon=True)
        self.config = config
        self.output = output
        self.stop_event = stop_event
        self.ready = threading.Event()
        self.error: Optional[BaseException] = None
        self.actual_width = 0
        self.actual_height = 0
        self.reported_fps = 0.0
        self.measured_fps = 0.0
        self.control_report: Optional[CameraControlReport] = None
        self._fps_smoother = ExponentialSmoother(0.45)
        self.last_frame_at = 0.0

    @property
    def mode_description(self) -> str:
        width = self.actual_width or int(self.config["capture_width"])
        height = self.actual_height or int(self.config["capture_height"])
        requested = int(self.config["capture_fps"])
        reported = self.reported_fps
        if reported > 0.0:
            description = (
                f"{width}x{height} • camera reports {reported:0.0f} FPS "
                f"• requested {requested}"
            )
        else:
            description = f"{width}x{height} • requested {requested} FPS"
        if self.control_report is not None:
            description += f" • {self.control_report.backend}"
            if self.control_report.exposure is not None:
                description += f" • EXP {self.control_report.exposure:g}"
        return description

    def run(self) -> None:
        capture: Optional[cv2.VideoCapture] = None
        if bool(self.config.get("prefer_low_latency_thread_priorities", True)):
            set_current_thread_priority("above_normal")
        try:
            capture = open_camera(self.config)
            self.control_report = apply_camera_controls(
                capture,
                self.config,
                stop_event=self.stop_event,
            )
            for message in self.control_report.messages:
                print(f"[Camera] {message}", flush=True)
            self.actual_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0))
            self.actual_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0))
            self.reported_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            self.ready.set()

            sequence = 0
            previous_time: Optional[float] = None
            failures = 0
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                captured_at = time.monotonic()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 90:
                        raise RuntimeError("The camera stopped returning frames")
                    time.sleep(0.005)
                    continue
                failures = 0
                sequence += 1
                self.last_frame_at = captured_at
                if previous_time is not None:
                    dt = max(1e-6, captured_at - previous_time)
                    self.measured_fps = self._fps_smoother.update(1.0 / dt, dt)
                previous_time = captured_at
                self.output.publish(
                    FramePacket(
                        sequence=sequence,
                        captured_at=captured_at,
                        frame=frame,
                    )
                )
        except BaseException as exc:
            self.error = exc
            self.ready.set()
        finally:
            if capture is not None:
                capture.release()
            self.output.close()


class PoseInferenceThread(threading.Thread):
    """Run semantic pose inference separately from high-rate foot tracking."""

    def __init__(
        self,
        model_path: Path,
        config: dict[str, Any],
        input_frames: LatestFrameBuffer,
        output: LatestPoseBuffer,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="PoseInference", daemon=True)
        self.model_path = model_path
        self.config = config
        self.input_frames = input_frames
        self.output = output
        self.stop_event = stop_event
        self.ready = threading.Event()
        self.error: Optional[BaseException] = None
        self.measured_fps = 0.0
        self._fps_smoother = ExponentialSmoother(0.55)

    def run(self) -> None:
        landmarker: Any = None
        if bool(self.config.get("prefer_low_latency_thread_priorities", True)):
            set_current_thread_priority("below_normal")
        try:
            landmarker = create_pose_landmarker(self.model_path, self.config)
            self.ready.set()
            last_sequence = -1
            timestamp_origin: Optional[float] = None
            last_timestamp_ms = -1
            previous_completion: Optional[float] = None

            while not self.stop_event.is_set():
                packet = self.input_frames.wait_for_new(last_sequence, 0.10)
                if packet is None:
                    continue
                last_sequence = packet.sequence
                if timestamp_origin is None:
                    timestamp_origin = packet.captured_at
                timestamp_ms = int((packet.captured_at - timestamp_origin) * 1000.0)
                timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                processing_started = time.monotonic()
                gray = (
                    packet.alignment_gray
                    if packet.alignment_gray is not None
                    else cv2.cvtColor(packet.frame, cv2.COLOR_BGR2GRAY)
                )
                inference_frame = resize_to_width(
                    packet.frame,
                    int(self.config["inference_width"]),
                )
                rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                completed_at = time.monotonic()
                if previous_completion is not None:
                    dt = max(1e-6, completed_at - previous_completion)
                    self.measured_fps = self._fps_smoother.update(1.0 / dt, dt)
                previous_completion = completed_at
                self.output.publish(
                    pose_anchor_from_result(
                        result=result,
                        gray=gray,
                        sequence=packet.sequence,
                        captured_at=packet.captured_at,
                        completed_at=completed_at,
                    )
                )
                maximum_fps = max(0.0, float(self.config.get("ai_max_fps", 0.0)))
                if maximum_fps > 0.0:
                    minimum_interval = 1.0 / maximum_fps
                    remaining = minimum_interval - (time.monotonic() - processing_started)
                    if remaining > 0.0:
                        self.stop_event.wait(remaining)
        except BaseException as exc:
            self.error = exc
            self.ready.set()
        finally:
            if landmarker is not None:
                try:
                    landmarker.close()
                except Exception:
                    pass


class HotkeyEdges:
    """Poll global Windows keys without stealing focus from the game."""

    VK_ESCAPE = 0x1B
    VK_F7 = 0x76
    VK_F8 = 0x77
    VK_F9 = 0x78
    VK_F10 = 0x79

    def __init__(self) -> None:
        self._previous: dict[int, bool] = {}
        self._available = sys.platform == "win32" and hasattr(ctypes, "windll")

    def pressed(self, virtual_key: int) -> bool:
        if not self._available:
            return False
        down = bool(ctypes.windll.user32.GetAsyncKeyState(virtual_key) & 0x8000)
        was_down = self._previous.get(virtual_key, False)
        self._previous[virtual_key] = down
        return down and not was_down


class VirtualXboxController:
    def __init__(self, preview_only: bool = False) -> None:
        self.available = False
        self.message = "Preview-only mode"
        self._gamepad: Any = None
        if preview_only:
            return
        try:
            import vgamepad as vg  # Imported here so preview-only mode still works.

            self._gamepad = vg.VX360Gamepad()
            self.available = True
            self.message = "Virtual Xbox 360 controller connected"
            self.neutral()
        except Exception as exc:  # Driver/library problems should not kill the HUD.
            self.available = False
            self.message = f"Controller unavailable: {exc}"
            self._gamepad = None

    def send(self, steering: float, gas: float, brake: float) -> None:
        if not self.available or self._gamepad is None:
            return
        steering = clip(steering, -1.0, 1.0)
        gas = clip(gas, 0.0, 1.0)
        brake = clip(brake, 0.0, 1.0)
        self._gamepad.left_joystick_float(x_value_float=steering, y_value_float=0.0)
        self._gamepad.right_trigger_float(value_float=gas)
        self._gamepad.left_trigger_float(value_float=brake)
        self._gamepad.update()

    def neutral(self) -> None:
        if not self.available or self._gamepad is None:
            return
        try:
            self._gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            self._gamepad.right_trigger_float(value_float=0.0)
            self._gamepad.left_trigger_float(value_float=0.0)
            self._gamepad.update()
        except Exception:
            pass

    def close(self) -> None:
        self.neutral()
        self._gamepad = None
        self.available = False


class CalibrationManager:
    """Noise-aware three- or five-pose calibration.

    Pedal-only mode intentionally never asks for wrists. Capture time advances
    only while the landmarks required for the current pose are fresh.
    """

    FULL_PHASES: tuple[tuple[str, str], ...] = (
        (
            "neutral",
            "RELAX: both feet neutral, hands in your straight-ahead steering pose",
        ),
        (
            "gas",
            "ACCELERATOR: make a SMALL comfortable RIGHT-foot movement and hold it",
        ),
        (
            "brake",
            "BRAKE: make a SMALL comfortable LEFT-foot movement and hold it",
        ),
        (
            "left",
            "STEER LEFT: turn your two-hand wheel gesture fully left",
        ),
        (
            "right",
            "STEER RIGHT: turn your two-hand wheel gesture fully right",
        ),
    )
    PEDAL_PHASES: tuple[tuple[str, str], ...] = (
        (
            "neutral",
            "RELAX: keep BOTH feet in their comfortable neutral positions",
        ),
        (
            "gas",
            "ACCELERATOR: keep RIGHT heel planted, tilt the toe like a pedal, and hold",
        ),
        (
            "brake",
            "BRAKE: keep LEFT heel planted, tilt the toe like a pedal, and hold",
        ),
    )
    # Retained as a class-level compatibility alias; each instance selects its
    # own tuple in __init__.
    PHASES = FULL_PHASES

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.pedals_only = str(config.get("control_mode", "pedals_only")) == "pedals_only"
        self.PHASES = self.PEDAL_PHASES if self.pedals_only else self.FULL_PHASES
        self.prepare_seconds = max(0.1, float(config["calibration_prepare_seconds"]))
        self.capture_seconds = max(0.25, float(config["calibration_capture_seconds"]))
        self.min_samples = max(3, int(config["calibration_min_samples"]))
        self.active = False
        self.complete = False
        self.failed_reason = ""
        self.phase_index = 0
        self.samples: list[Any] = []
        self.captured: dict[str, Any] = {}
        self.data: Optional[CalibrationData] = None
        self.prepare_elapsed = 0.0
        self.capture_elapsed = 0.0
        self.last_update = 0.0
        self.current_tracking_ok = False
        self.tracking_hint = "Waiting for the first camera frame"

    def start(self, now: Optional[float] = None) -> None:
        started = time.monotonic() if now is None else float(now)
        self.active = True
        self.complete = False
        self.failed_reason = ""
        self.phase_index = 0
        self.samples = []
        self.captured = {}
        self.data = None
        self._reset_phase(started)

    def _reset_phase(self, now: float) -> None:
        self.samples = []
        self.prepare_elapsed = 0.0
        self.capture_elapsed = 0.0
        self.last_update = float(now)
        self.current_tracking_ok = False
        self.tracking_hint = "Move into the shown pose"

    @property
    def phase_name(self) -> str:
        if self.phase_index >= len(self.PHASES):
            return "done"
        return self.PHASES[self.phase_index][0]

    @property
    def prompt(self) -> str:
        if self.failed_reason:
            return f"Calibration failed: {self.failed_reason}. Press F9 to try again."
        if self.phase_index >= len(self.PHASES):
            return "Calibration complete"
        return self.PHASES[self.phase_index][1]

    @property
    def required_groups(self) -> tuple[str, ...]:
        phase = self.phase_name
        if phase == "neutral":
            return ("left foot", "right foot") if self.pedals_only else (
                "left foot",
                "right foot",
                "hands",
            )
        if phase == "gas":
            return ("right foot",)
        if phase == "brake":
            return ("left foot",)
        if phase in {"left", "right"}:
            return ("hands",)
        return ()

    def progress(self) -> tuple[float, bool, float, str]:
        if self.prepare_elapsed < self.prepare_seconds:
            fraction = self.prepare_elapsed / max(
                self.prepare_seconds + self.capture_seconds, 1e-6
            )
            return (
                max(0.0, self.prepare_seconds - self.prepare_elapsed),
                False,
                clip(fraction, 0.0, 1.0),
                "GET READY",
            )

        time_fraction = clip(self.capture_elapsed / self.capture_seconds, 0.0, 1.0)
        sample_fraction = clip(len(self.samples) / self.min_samples, 0.0, 1.0)
        capture_fraction = min(time_fraction, sample_fraction)
        fraction = (
            self.prepare_seconds + capture_fraction * self.capture_seconds
        ) / max(self.prepare_seconds + self.capture_seconds, 1e-6)
        stage = "CAPTURING" if self.current_tracking_ok else "WAITING FOR TRACKING"
        return (
            max(0.0, self.capture_seconds - self.capture_elapsed),
            self.current_tracking_ok,
            clip(fraction, 0.0, 1.0),
            stage,
        )

    def update(self, features: PoseFeatures, now: float) -> Optional[CalibrationData]:
        if not self.active or self.complete or self.failed_reason:
            return self.data

        now = float(now)
        dt = clip(now - self.last_update, 0.0, 0.25)
        self.last_update = now
        sample = self._sample_for_phase(self.phase_name, features)
        self.current_tracking_ok = sample is not None
        self.tracking_hint = self._tracking_hint(features)

        if self.prepare_elapsed < self.prepare_seconds:
            self.prepare_elapsed = min(self.prepare_seconds, self.prepare_elapsed + dt)
            return None
        if sample is None:
            return None

        self.samples.append(sample)
        self.capture_elapsed += dt
        if self.capture_elapsed < self.capture_seconds or len(self.samples) < self.min_samples:
            return None

        self.captured[self.phase_name] = self._reduce_samples(self.phase_name, self.samples)
        self.phase_index += 1
        if self.phase_index < len(self.PHASES):
            self._reset_phase(now)
            return None

        try:
            self.data = self._build_and_validate()
        except ValueError as exc:
            self.failed_reason = str(exc)
            self.active = False
            self.complete = False
            self.data = None
            self.current_tracking_ok = False
            self.tracking_hint = (
                "Press F9, hold neutral more steadily, then use a tiny distinct movement"
            )
            return None

        self.active = False
        self.complete = True
        self.current_tracking_ok = True
        self.tracking_hint = "Calibration complete"
        return self.data

    def _tracking_hint(self, features: PoseFeatures) -> str:
        if not features.pose_detected:
            return "No body detected — include the knees and feet, then check camera rotation"

        missing: list[str] = []
        requirements = self.required_groups
        if "left foot" in requirements and (
            not features.left_foot_ok or not features.left_foot_fresh
        ):
            missing.append("LEFT HEEL TRIANGLE (heel/ankle/toe hard lock)")
        if "right foot" in requirements and (
            not features.right_foot_ok or not features.right_foot_fresh
        ):
            missing.append("RIGHT HEEL TRIANGLE (heel/ankle/toe hard lock)")
        if "hands" in requirements and (
            not features.steering_ok or not features.steering_fresh
        ):
            missing.append("BOTH WRISTS (fresh tracking)")

        if missing:
            return "Missing: " + ", ".join(missing) + " — keep them uncovered and inside the frame"
        if self.prepare_elapsed < self.prepare_seconds:
            return "Tracking ready — settle into the shown pose"
        return "Tracking locked — hold the tiny pose steadily while camera noise is measured"

    def _sample_for_phase(self, phase: str, features: PoseFeatures) -> Optional[Any]:
        if phase == "neutral":
            if not features.pedals_ok or not features.pedals_fresh:
                return None
            assert features.left_foot is not None
            assert features.right_foot is not None
            if self.pedals_only:
                return (features.left_foot.copy(), features.right_foot.copy())
            if not features.steering_ok or not features.steering_fresh:
                return None
            assert features.steering_angle is not None
            return (
                features.left_foot.copy(),
                features.right_foot.copy(),
                float(features.steering_angle),
            )
        if phase == "gas":
            return (
                features.right_foot.copy()
                if features.right_foot_ok
                and features.right_foot_fresh
                and features.right_foot is not None
                else None
            )
        if phase == "brake":
            return (
                features.left_foot.copy()
                if features.left_foot_ok
                and features.left_foot_fresh
                and features.left_foot is not None
                else None
            )
        if phase in {"left", "right"}:
            return (
                float(features.steering_angle)
                if features.steering_ok
                and features.steering_fresh
                and features.steering_angle is not None
                else None
            )
        return None

    def _reduce_samples(self, phase: str, samples: Sequence[Any]) -> Any:
        if phase == "neutral":
            left, left_noise, left_reliability = robust_observation_stats(
                [sample[0] for sample in samples]
            )
            right, right_noise, right_reliability = robust_observation_stats(
                [sample[1] for sample in samples]
            )
            steer = 0.0 if self.pedals_only else circular_mean(
                [sample[2] for sample in samples]
            )
            return (
                left,
                left_noise,
                left_reliability,
                right,
                right_noise,
                right_reliability,
                steer,
            )
        if phase in {"gas", "brake"}:
            return robust_observation_stats(samples)
        if phase in {"left", "right"}:
            return circular_mean(samples)
        raise ValueError(f"Unknown calibration phase: {phase}")

    def _build_and_validate(self) -> CalibrationData:
        (
            neutral_left,
            neutral_left_noise,
            neutral_left_reliability,
            neutral_right,
            neutral_right_noise,
            neutral_right_reliability,
            neutral_steer,
        ) = self.captured["neutral"]
        gas_right, gas_right_noise, gas_right_reliability = self.captured["gas"]
        brake_left, brake_left_noise, brake_left_reliability = self.captured["brake"]

        left_noise = np.maximum(neutral_left_noise, brake_left_noise)
        right_noise = np.maximum(neutral_right_noise, gas_right_noise)
        left_reliability = np.minimum(
            neutral_left_reliability, brake_left_reliability
        )
        right_reliability = np.minimum(
            neutral_right_reliability, gas_right_reliability
        )
        noise_floor = float(self.config["pedal_noise_floor"])
        left_motion = float(
            np.sqrt(np.sum(left_reliability * np.square(brake_left - neutral_left)))
        )
        right_motion = float(
            np.sqrt(np.sum(right_reliability * np.square(gas_right - neutral_right)))
        )
        left_snr = feature_signal_to_noise(
            neutral_left,
            brake_left,
            left_noise,
            noise_floor,
            weights=left_reliability,
        )
        right_snr = feature_signal_to_noise(
            neutral_right,
            gas_right,
            right_noise,
            noise_floor,
            weights=right_reliability,
        )
        minimum_motion = max(1e-5, float(self.config["calibration_min_absolute_motion"]))
        minimum_snr = max(1.0, float(self.config["calibration_min_signal_to_noise"]))
        minimum_heel_tilt = math.radians(
            max(0.1, float(self.config.get("calibration_min_heel_tilt_degrees", 2.0)))
        )
        right_heel_tilt = abs(
            wrap_angle(
                math.atan2(float(gas_right[0]), float(gas_right[1]))
                - math.atan2(float(neutral_right[0]), float(neutral_right[1]))
            )
        )
        left_heel_tilt = abs(
            wrap_angle(
                math.atan2(float(brake_left[0]), float(brake_left[1]))
                - math.atan2(float(neutral_left[0]), float(neutral_left[1]))
            )
        )

        if right_heel_tilt < minimum_heel_tilt:
            raise ValueError(
                "right heel-to-toe tilt was too small; keep the heel planted and "
                "make a tiny visible toe rotation"
            )
        if left_heel_tilt < minimum_heel_tilt:
            raise ValueError(
                "left heel-to-toe tilt was too small; keep the heel planted and "
                "make a tiny visible toe rotation"
            )

        if right_motion < minimum_motion or right_snr < minimum_snr:
            raise ValueError(
                "right-foot movement was too close to tracking noise; keep the foot "
                "comfortable and move the camera closer or improve the lighting"
            )
        if left_motion < minimum_motion or left_snr < minimum_snr:
            raise ValueError(
                "left-foot movement was too close to tracking noise; keep the foot "
                "comfortable and move the camera closer or improve the lighting"
            )

        if self.pedals_only:
            steer_left = 0.0
            steer_right = 0.0
            neutral_steer = 0.0
        else:
            steer_left = float(self.captured["left"])
            steer_right = float(self.captured["right"])
            left_delta = wrap_angle(steer_left - neutral_steer)
            right_delta = wrap_angle(steer_right - neutral_steer)
            minimum_steer = math.radians(
                max(1.0, float(self.config["calibration_min_steering_degrees"]))
            )
            if abs(left_delta) < minimum_steer:
                raise ValueError("left steering movement was too small")
            if abs(right_delta) < minimum_steer:
                raise ValueError("right steering movement was too small")
            if left_delta * right_delta >= 0.0:
                raise ValueError(
                    "left and right steering poses were not on opposite sides of centre"
                )

        return CalibrationData(
            left_foot_neutral=neutral_left,
            left_foot_pressed=brake_left,
            left_foot_noise=left_noise,
            left_foot_reliability=left_reliability,
            left_foot_signal_to_noise=left_snr,
            right_foot_neutral=neutral_right,
            right_foot_pressed=gas_right,
            right_foot_noise=right_noise,
            right_foot_reliability=right_reliability,
            right_foot_signal_to_noise=right_snr,
            steering_neutral=float(neutral_steer),
            steering_left=float(steer_left),
            steering_right=float(steer_right),
        )


class ControlMapper:
    def __init__(self, calibration: CalibrationData, config: dict[str, Any]) -> None:
        self.calibration = calibration
        self.config = config
        self.pedals_only = str(config.get("control_mode", "pedals_only")) == "pedals_only"
        self._last_projection: dict[str, Optional[float]] = {
            "gas": None,
            "brake": None,
        }

    def reset(self) -> None:
        self._last_projection = {"gas": None, "brake": None}

    def _project_foot(
        self,
        side: str,
        feature: FootFeatureObservation,
    ) -> Optional[tuple[float, float]]:
        if side == "left":
            neutral = self.calibration.left_foot_neutral
            pressed = self.calibration.left_foot_pressed
            snr = self.calibration.left_foot_signal_to_noise
        elif side == "right":
            neutral = self.calibration.right_foot_neutral
            pressed = self.calibration.right_foot_pressed
            snr = self.calibration.right_foot_signal_to_noise
        else:
            raise ValueError("side must be 'left' or 'right'")
        value = project_heel_hinge(
            feature,
            neutral,
            pressed,
            minimum_tilt_degrees=float(
                self.config.get("calibration_min_heel_tilt_degrees", 2.0)
            ),
            extension_weight=float(self.config.get("heel_extension_weight", 0.15)),
        )
        if value is None:
            return None
        return value, float(snr)

    def _predict_rising_projection(
        self,
        label: str,
        raw: float,
        dt: Optional[float],
    ) -> float:
        raw = clip(raw, 0.0, 1.35)
        previous = self._last_projection[label]
        self._last_projection[label] = raw
        if previous is None or dt is None:
            return raw
        dt = float(dt)
        if dt <= 1e-4 or dt > 0.060:
            return raw
        rise = raw - previous
        minimum_delta = max(0.0, float(self.config.get("pedal_prediction_min_delta", 0.0)))
        if rise <= minimum_delta:
            return raw
        lookahead = max(0.0, float(self.config.get("pedal_lookahead_seconds", 0.0)))
        if lookahead <= 0.0:
            return raw
        maximum_advance = max(
            0.0,
            float(self.config.get("pedal_prediction_max_advance", 0.0)),
        )
        advance = rise * min(2.0, lookahead / dt)
        advance = min(advance, maximum_advance)
        return clip(raw + advance, 0.0, 1.35)

    def map_pedal_features(
        self,
        left_foot: Optional[FootFeatureObservation],
        right_foot: Optional[FootFeatureObservation],
        dt: Optional[float] = None,
    ) -> tuple[Optional[float], Optional[float]]:
        left_projection = (
            self._project_foot("left", left_foot) if left_foot is not None else None
        )
        right_projection = (
            self._project_foot("right", right_foot) if right_foot is not None else None
        )

        if bool(self.config["swap_pedals"]):
            gas_projection, brake_projection = left_projection, right_projection
        else:
            gas_projection, brake_projection = right_projection, left_projection

        noise_factor = max(0.0, float(self.config["pedal_noise_deadzone_factor"]))
        gas: Optional[float] = None
        brake: Optional[float] = None
        if gas_projection is not None:
            raw, snr = gas_projection
            raw = self._predict_rising_projection("gas", raw, dt)
            deadzone = adaptive_pedal_deadzone(
                float(self.config.get("throttle_deadzone", self.config["pedal_deadzone"])),
                snr,
                noise_factor,
            )
            gas = shape_unipolar(
                raw
                * max(
                    0.1,
                    float(
                        self.config.get(
                            "throttle_sensitivity", self.config["pedal_sensitivity"]
                        )
                    ),
                ),
                deadzone=deadzone,
                exponent=float(
                    self.config.get(
                        "throttle_curve_exponent", self.config["pedal_curve_exponent"]
                    )
                ),
                response_floor=float(self.config.get("throttle_initial_response", 0.0)),
                response_boost=float(self.config.get("throttle_response_boost", 0.0)),
            )
        else:
            self._last_projection["gas"] = None

        if brake_projection is not None:
            raw, snr = brake_projection
            raw = self._predict_rising_projection("brake", raw, dt)
            deadzone = adaptive_pedal_deadzone(
                float(self.config.get("brake_deadzone", self.config["pedal_deadzone"])),
                snr,
                noise_factor,
            )
            brake = shape_unipolar(
                raw
                * max(
                    0.1,
                    float(
                        self.config.get(
                            "brake_sensitivity", self.config["pedal_sensitivity"]
                        )
                    ),
                ),
                deadzone=deadzone,
                exponent=float(
                    self.config.get(
                        "brake_curve_exponent", self.config["pedal_curve_exponent"]
                    )
                ),
                response_floor=float(self.config.get("brake_initial_response", 0.0)),
                response_boost=float(self.config.get("brake_response_boost", 0.0)),
            )
        else:
            self._last_projection["brake"] = None
        return gas, brake

    def map(
        self,
        features: PoseFeatures,
        dt: Optional[float] = None,
    ) -> tuple[float, float, float]:
        if not features.pedals_ok:
            raise ValueError("Foot tracking is incomplete")
        if not self.pedals_only and not features.steering_ok:
            raise ValueError("Hand tracking is incomplete")
        assert features.left_foot is not None
        assert features.right_foot is not None
        gas, brake = self.map_pedal_features(
            features.left_foot,
            features.right_foot,
            dt=dt,
        )
        # Optional heel/world dimensions can drop out after calibration even
        # while the core landmarks remain present. A failed coverage gate must
        # neutralize only that input, never crash the control thread.
        gas_value = 0.0 if gas is None else gas
        brake_value = 0.0 if brake is None else brake
        steering = (
            0.0
            if self.pedals_only
            else self._map_steering(float(features.steering_angle))
        )
        return steering, gas_value, brake_value

    def _map_steering(self, angle: float) -> float:
        neutral = self.calibration.steering_neutral
        current_delta = wrap_angle(angle - neutral)
        left_delta = wrap_angle(self.calibration.steering_left - neutral)
        right_delta = wrap_angle(self.calibration.steering_right - neutral)

        if current_delta * left_delta > 0.0:
            raw = -abs(current_delta / left_delta)
        elif current_delta * right_delta > 0.0:
            raw = abs(current_delta / right_delta)
        else:
            raw = 0.0

        raw *= float(self.config["steering_sensitivity"])
        if bool(self.config["invert_steering"]):
            raw = -raw
        return shape_bipolar(
            raw,
            deadzone=float(self.config["steering_deadzone"]),
            exponent=float(self.config["steering_curve_exponent"]),
        )


class CameraDriveWorker(threading.Thread):
    def __init__(
        self,
        config: dict[str, Any],
        shared: SharedState,
        stop_event: threading.Event,
        preview_only: bool = False,
    ) -> None:
        super().__init__(name="CameraDriveWorker", daemon=True)
        self.config = config
        self.shared = shared
        self.stop_event = stop_event
        self.preview_only = preview_only
        self.pedals_only = str(config.get("control_mode", "pedals_only")) == "pedals_only"
        self.hotkeys = HotkeyEdges()
        self.controller: Optional[VirtualXboxController] = None
        self.calibration = CalibrationManager(config)
        self.mapper: Optional[ControlMapper] = None
        self.low_light_enhancer = LowLightEnhancer(config)
        self.light_metrics: Optional[LightMetrics] = None
        self.pose_tracker = PoseFeatureTracker(config)
        steering_tau = float(
            config.get("steering_smoothing_tau_seconds", config["smoothing_tau_seconds"])
        )
        self.steer_smoother = ExponentialSmoother(steering_tau)
        self.gas_smoother = AsymmetricPedalSmoother(
            float(config["pedal_attack_tau_seconds"]),
            float(config["pedal_release_tau_seconds"]),
            bool(config["pedal_release_snap_to_zero"]),
        )
        self.brake_smoother = AsymmetricPedalSmoother(
            float(config["pedal_attack_tau_seconds"]),
            float(config["pedal_release_tau_seconds"]),
            bool(config["pedal_release_snap_to_zero"]),
        )
        self.active = False
        self.last_valid_tracking = 0.0
        self.last_left_tracking = 0.0
        self.last_right_tracking = 0.0
        self.last_loop_time = time.monotonic()
        self.fps_smoother = ExponentialSmoother(0.5)
        self._preview_enabled = bool(config["show_camera_preview"])
        self._last_calibration_console_state: Optional[tuple[int, str, str]] = None
        self.anchor_setup_active = False
        self.anchor_setup_points: dict[int, np.ndarray] = {}
        self.anchor_setup_frame: Optional[np.ndarray] = None
        self.anchor_setup_preview: Optional[QImage] = None
        self.anchor_setup_requested = bool(
            config.get("_anchor_setup_requested", False)
            or config.get("manual_anchor_required", True)
        )
        self.windowed_hud = bool(config.get("_windowed_hud", False))

    def run(self) -> None:
        if bool(self.config.get("prefer_low_latency_thread_priorities", True)):
            set_current_thread_priority("above_normal")
        capture_frames = LatestFrameBuffer()
        inference_frames = LatestFrameBuffer()
        pose_results = LatestPoseBuffer()
        capture_thread: Optional[CameraCaptureThread] = None
        inference_thread: Optional[PoseInferenceThread] = None
        try:
            model_path = ensure_pose_model(str(self.config["model"]), self.shared)
            self.controller = VirtualXboxController(preview_only=self.preview_only)
            self.shared.update(
                controller_available=self.controller.available,
                controller_message=self.controller.message,
            )

            capture_thread = CameraCaptureThread(
                config=self.config,
                output=capture_frames,
                stop_event=self.stop_event,
            )
            capture_thread.start()
            if not capture_thread.ready.wait(
                timeout=float(self.config["camera_start_timeout_seconds"])
            ):
                raise RuntimeError("Camera startup timed out")
            if capture_thread.error is not None:
                raise RuntimeError(f"Camera could not start: {capture_thread.error}")

            inference_thread = PoseInferenceThread(
                model_path=model_path,
                config=self.config,
                input_frames=inference_frames,
                output=pose_results,
                stop_event=self.stop_event,
            )
            inference_thread.start()
            if not inference_thread.ready.wait(
                timeout=float(self.config["ai_start_timeout_seconds"])
            ):
                raise RuntimeError("AI pose model startup timed out")
            if inference_thread.error is not None:
                raise RuntimeError(f"AI pose model could not start: {inference_thread.error}")

            self.shared.update(
                mode="PAUSED",
                pedals_only=self.pedals_only,
                headline=(
                    "Camera-rate feet-only tracking is ready"
                    if self.pedals_only
                    else "Camera-rate feet-and-hands tracking is ready"
                ),
                detail=(
                    "AI identifies each foot • patch flow follows tiny movement every camera frame"
                    if self.pedals_only
                    else "AI anchors + camera-rate optical flow • F9 calibrates • F8 starts/pauses"
                ),
                camera_mode=(
                    f"{capture_thread.mode_description} • "
                    f"{self.config.get('_runtime_profile', 'custom/balanced')} profile"
                ),
            )

            if self.anchor_setup_requested:
                self._start_anchor_setup()
            elif bool(self.config["auto_calibrate"]):
                self._start_calibration(time.monotonic())
            else:
                self.shared.update(
                    prompt="Press F9 to calibrate before driving",
                    mode="NEEDS CALIBRATION",
                )

            last_sequence = -1
            last_pose_sequence = -1
            previous_capture_time: Optional[float] = None
            last_frame_received_at = time.monotonic()
            last_preview_at = 0.0
            last_light_metrics_at = 0.0
            preview_interval = 1.0 / max(
                1.0, float(self.config.get("preview_max_fps", 15.0))
            )

            while not self.stop_event.is_set():
                self._handle_hotkeys()
                if self.stop_event.is_set():
                    break

                packet = capture_frames.wait_for_new(last_sequence, 0.05)
                now = time.monotonic()
                if packet is None:
                    if capture_thread.error is not None:
                        raise RuntimeError(f"Camera failed: {capture_thread.error}")
                    if inference_thread.error is not None:
                        raise RuntimeError(f"AI tracking failed: {inference_thread.error}")
                    if (
                        now - last_frame_received_at
                        >= float(self.config["camera_frame_timeout_seconds"])
                    ):
                        self._neutral_due_to_tracking_loss(now, force=True)
                        self.shared.update(
                            mode="CAMERA LOST",
                            headline="No fresh camera frame",
                            detail="Close other camera apps and reconnect the webcam",
                            tracking_ok=False,
                        )
                    continue

                last_sequence = packet.sequence
                last_frame_received_at = now
                if previous_capture_time is None:
                    dt = 1.0 / max(1.0, float(self.config["capture_fps"]))
                else:
                    dt = clip(packet.captured_at - previous_capture_time, 0.0005, 0.25)
                previous_capture_time = packet.captured_at

                frame = rotate_frame(
                    packet.frame,
                    int(self.config["camera_rotation_degrees"]),
                )
                if now - last_light_metrics_at >= 0.50:
                    self.light_metrics = compute_light_metrics(frame)
                    last_light_metrics_at = now
                enhanced_frame, tracking_frame = prepare_processing_frames(
                    frame,
                    self.config,
                    self.low_light_enhancer,
                )
                alignment_gray = cv2.cvtColor(tracking_frame, cv2.COLOR_BGR2GRAY)
                # This one-slot queue drops any unprocessed image, so AI always
                # works on the newest full capture instead of adding latency.
                # The attached gray image matches the high-rate tracking size,
                # so delayed semantic anchors can be aligned without upscaling.
                inference_frames.publish(
                    FramePacket(
                        sequence=packet.sequence,
                        captured_at=packet.captured_at,
                        frame=enhanced_frame,
                        alignment_gray=alignment_gray,
                    )
                )

                latest_pose = pose_results.snapshot()
                anchor: Optional[PoseAnchorPacket] = None
                if latest_pose is not None and latest_pose.sequence > last_pose_sequence:
                    anchor = latest_pose
                    last_pose_sequence = latest_pose.sequence

                self._ensure_anchor_setup_frame(tracking_frame)
                self._consume_anchor_clicks(tracking_frame, now)

                features = self.pose_tracker.update(
                    frame=tracking_frame,
                    now=now,
                    dt=dt,
                    anchor=anchor,
                    sequence=packet.sequence,
                )

                if self.anchor_setup_active:
                    self._process_anchor_setup(features)
                elif self.calibration.active:
                    self._process_calibration(features, now)
                else:
                    self._process_controls(features, now, dt)

                instantaneous_fps = 1.0 / max(dt, 1e-6)
                tracking_fps = self.fps_smoother.update(instantaneous_fps, dt)
                changes: dict[str, Any] = {
                    "fps": tracking_fps,
                    "camera_fps": float(capture_thread.measured_fps),
                    "ai_fps": float(inference_thread.measured_fps),
                    "camera_mode": (
                        f"{capture_thread.mode_description} • "
                        f"{self.config.get('_runtime_profile', 'custom/balanced')} profile"
                    ),
                }
                if self.light_metrics is not None:
                    changes["light_luma"] = self.light_metrics.median_luma
                    changes["light_dark_fraction"] = self.light_metrics.dark_fraction
                if self.anchor_setup_active and self.anchor_setup_preview is not None:
                    # Every setup click must refer to one immutable image. A
                    # live preview would let early clicks age before click six.
                    changes["preview"] = self.anchor_setup_preview
                elif self._preview_enabled and now - last_preview_at >= preview_interval:
                    changes["preview"] = make_preview_qimage(
                        enhanced_frame, features, self.config
                    )
                    last_preview_at = now
                self.shared.update(**changes)

                if capture_thread.error is not None:
                    raise RuntimeError(f"Camera failed: {capture_thread.error}")
                if inference_thread.error is not None:
                    raise RuntimeError(f"AI tracking failed: {inference_thread.error}")

        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            self.active = False
            if self.controller is not None:
                self.controller.neutral()
            self.shared.update(
                mode="ERROR",
                headline="CameraDrive stopped",
                detail=error_text,
                fatal_error=error_text,
                tracking_ok=False,
                active=False,
                gas=0.0,
                brake=0.0,
                steering=0.0,
            )
            show_native_error(f"{APP_NAME}\n\n{error_text}")
            self.stop_event.set()
            self.shared.update(exit_requested=True)
        finally:
            inference_frames.close()
            capture_frames.close()
            if capture_thread is not None and capture_thread.is_alive():
                capture_thread.join(timeout=2.0)
            if inference_thread is not None and inference_thread.is_alive():
                inference_thread.join(timeout=2.0)
            if self.controller is not None:
                self.controller.close()
            self.shared.update(
                active=False,
                tracking_ok=False,
                gas=0.0,
                brake=0.0,
                steering=0.0,
            )

    def _handle_hotkeys(self) -> None:
        if self.hotkeys.pressed(HotkeyEdges.VK_ESCAPE):
            self.stop_event.set()
            self.shared.update(exit_requested=True)
            return

        if self.hotkeys.pressed(HotkeyEdges.VK_F10):
            self._preview_enabled = not self._preview_enabled
            self.shared.update(show_preview=self._preview_enabled, preview=None if not self._preview_enabled else self.shared.snapshot().preview)

        if self.hotkeys.pressed(HotkeyEdges.VK_F7):
            self._start_anchor_setup()

        if self.hotkeys.pressed(HotkeyEdges.VK_F9):
            if bool(self.config.get("manual_anchor_required", True)) and not self.pose_tracker.manual_anchors_complete:
                self._start_anchor_setup()
            else:
                self._start_calibration(time.monotonic())

        if self.hotkeys.pressed(HotkeyEdges.VK_F8):
            if self.anchor_setup_active:
                self.shared.update(
                    headline="Finish the six anchor clicks before enabling controls",
                    prompt=self._anchor_prompt(),
                )
                return
            if self.calibration.complete and self.mapper is not None:
                self.active = not self.active
                if not self.active:
                    self._reset_controls()
                label = "ACTIVE" if self.active else "PAUSED"
                suffix = "" if (self.controller and self.controller.available) else " (preview only)"
                hide_preview = self.active and bool(
                    self.config.get("auto_hide_preview_when_active", False)
                )
                if hide_preview:
                    self._preview_enabled = False
                self.shared.update(
                    mode=label + suffix,
                    headline=(
                        (
                            "Foot tracking is driving RT and LT; steering is ignored"
                            if self.pedals_only
                            else "Camera controls are driving the virtual controller"
                        )
                        if self.active
                        else "Controls are paused and neutral"
                    ),
                    active=self.active,
                    prompt="",
                    show_preview=self._preview_enabled,
                    preview=None if hide_preview else self.shared.snapshot().preview,
                )
            else:
                self.shared.update(
                    mode="NEEDS CALIBRATION",
                    headline="Calibrate before enabling controls",
                    prompt=(
                        "Press F9 and follow the three foot poses"
                        if self.pedals_only
                        else "Press F9 and follow the five on-screen poses"
                    ),
                    active=False,
                )

    def _anchor_prompt(self) -> str:
        step = len(self.anchor_setup_points)
        if step >= len(MANUAL_ANCHOR_SEQUENCE):
            return "Validating the fixed heel triangles"
        return f"CLICK {step + 1}/6: {MANUAL_ANCHOR_SEQUENCE[step][1]}"

    def _start_anchor_setup(self) -> None:
        self.active = False
        self.mapper = None
        self.calibration.active = False
        self.calibration.complete = False
        self.anchor_setup_points = {}
        self.anchor_setup_frame = None
        self.anchor_setup_preview = None
        self.shared.clear_anchor_clicks()
        self.pose_tracker.clear_manual_foot_anchors()
        self._preview_enabled = True
        self._reset_controls()
        if not self.windowed_hud:
            self.anchor_setup_active = False
            self.shared.update(
                mode="ANCHOR SETUP REQUIRED",
                headline="Fixed heel anchors need a clickable camera window",
                detail="Run run_fixed_heel_pedals_windowed.bat",
                prompt="Open the fixed-heel windowed launcher, then click the six points",
                anchor_setup_active=False,
                anchor_input_available=False,
                calibrated=False,
                active=False,
                tracking_ok=False,
                gas=0.0,
                brake=0.0,
                steering=0.0,
                show_preview=True,
            )
            return
        self.anchor_setup_active = True
        self.shared.update(
            mode="SET FOOT ANCHORS",
            headline="Hard-lock the heel-pivot triangles",
            detail="Click the exact points on the live image • F7 restarts",
            prompt=self._anchor_prompt(),
            anchor_setup_active=True,
            anchor_setup_step=0,
            anchor_setup_count=len(MANUAL_ANCHOR_SEQUENCE),
            anchor_setup_label=MANUAL_ANCHOR_SEQUENCE[0][1],
            anchor_setup_points={},
            anchor_input_available=False,
            preview_mirrored=bool(self.config.get("mirror_preview", True)),
            calibrated=False,
            active=False,
            tracking_ok=False,
            gas=0.0,
            brake=0.0,
            steering=0.0,
            show_preview=True,
        )

    def _ensure_anchor_setup_frame(self, tracking_frame: np.ndarray) -> None:
        if not self.anchor_setup_active or self.anchor_setup_frame is not None:
            return
        self.anchor_setup_frame = np.ascontiguousarray(tracking_frame.copy())
        self.anchor_setup_preview = make_preview_qimage(
            self.anchor_setup_frame,
            PoseFeatures(),
            self.config,
        )
        self.shared.update(
            preview=self.anchor_setup_preview,
            anchor_input_available=True,
            headline="Hard-lock the heel-pivot triangles",
            detail="Camera image frozen for all six clicks • keep both feet still",
            prompt=self._anchor_prompt(),
        )

    def _consume_anchor_clicks(
        self,
        tracking_frame: np.ndarray,
        now: float,
    ) -> None:
        clicks = self.shared.take_anchor_clicks()
        if not self.anchor_setup_active or self.anchor_setup_frame is None:
            return
        frame_shape = tracking_frame.shape
        for x, y in clicks:
            step = len(self.anchor_setup_points)
            if step >= len(MANUAL_ANCHOR_SEQUENCE):
                break
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                continue
            index, _label = MANUAL_ANCHOR_SEQUENCE[step]
            self.anchor_setup_points[index] = np.array([x, y], dtype=np.float64)
            if len(self.anchor_setup_points) == len(MANUAL_ANCHOR_SEQUENCE):
                error = validate_manual_anchor_points(
                    self.anchor_setup_points,
                    frame_shape,
                    float(self.config.get("manual_anchor_min_separation_pixels", 6.0)),
                )
                if error:
                    self.anchor_setup_points = {}
                    self.shared.update(
                        headline=f"Anchor setup restarted: {error}",
                        prompt=self._anchor_prompt(),
                        anchor_setup_step=0,
                        anchor_setup_label=MANUAL_ANCHOR_SEQUENCE[0][1],
                        anchor_setup_points={},
                    )
                    continue
                reference_gray = cv2.cvtColor(
                    self.anchor_setup_frame,
                    cv2.COLOR_BGR2GRAY,
                )
                current_gray = cv2.cvtColor(tracking_frame, cv2.COLOR_BGR2GRAY)
                aligned = self.pose_tracker.align_manual_anchor_selection(
                    reference_gray,
                    current_gray,
                    self.anchor_setup_points,
                )
                if set(aligned) != set(MANUAL_FOOT_ANCHOR_INDICES):
                    self.anchor_setup_points = {}
                    self.anchor_setup_frame = None
                    self.anchor_setup_preview = None
                    self.shared.update(
                        headline="Anchor setup restarted: a selected texture moved or was lost",
                        detail="Keep both feet still while selecting all six points",
                        prompt="Freezing a new setup frame",
                        anchor_setup_step=0,
                        anchor_setup_label=MANUAL_ANCHOR_SEQUENCE[0][1],
                        anchor_setup_points={},
                        anchor_input_available=False,
                    )
                    continue
                aligned_error = validate_manual_anchor_points(
                    aligned,
                    current_gray.shape,
                    float(self.config.get("manual_anchor_min_separation_pixels", 6.0)),
                )
                if aligned_error:
                    self.anchor_setup_points = {}
                    self.anchor_setup_frame = None
                    self.anchor_setup_preview = None
                    self.shared.update(
                        headline=f"Anchor setup restarted: {aligned_error}",
                        detail="Keep both feet still while selecting all six points",
                        prompt="Freezing a new setup frame",
                        anchor_setup_step=0,
                        anchor_setup_label=MANUAL_ANCHOR_SEQUENCE[0][1],
                        anchor_setup_points={},
                        anchor_input_available=False,
                    )
                    continue
                try:
                    self.pose_tracker.set_manual_foot_anchors(
                        aligned,
                        reference_gray=reference_gray,
                        reference_points=self.anchor_setup_points,
                    )
                except ValueError as exc:
                    self.anchor_setup_points = {}
                    self.anchor_setup_frame = None
                    self.anchor_setup_preview = None
                    self.shared.update(
                        headline=f"Anchor setup restarted: {exc}",
                        detail="Choose visible textured skin, sock, or shoe points",
                        prompt="Freezing a new setup frame",
                        anchor_setup_step=0,
                        anchor_setup_label=MANUAL_ANCHOR_SEQUENCE[0][1],
                        anchor_setup_points={},
                        anchor_input_available=False,
                    )
                    continue
                self.anchor_setup_active = False
                self.anchor_setup_frame = None
                self.anchor_setup_preview = None
                self.shared.update(
                    anchor_setup_active=False,
                    anchor_input_available=False,
                    anchor_setup_points={
                        key: (float(value[0]), float(value[1]))
                        for key, value in self.anchor_setup_points.items()
                    },
                )
                self._start_calibration(now)
                return
            next_step = len(self.anchor_setup_points)
            self.shared.update(
                prompt=self._anchor_prompt(),
                anchor_setup_step=next_step,
                anchor_setup_label=MANUAL_ANCHOR_SEQUENCE[next_step][1],
                anchor_setup_points={
                    key: (float(value[0]), float(value[1]))
                    for key, value in self.anchor_setup_points.items()
                },
            )

    def _process_anchor_setup(self, features: PoseFeatures) -> None:
        self._reset_controls(update_ui=False)
        self.shared.update(
            mode="SET FOOT ANCHORS",
            prompt=self._anchor_prompt(),
            left_foot_ok=False,
            right_foot_ok=False,
            steering_ok=False,
            tracking_hint="Clicks are fixed to local foot texture; AI cannot overwrite them",
            tracking_ok=False,
            gas=0.0,
            brake=0.0,
            steering=0.0,
        )

    def _start_calibration(self, now: float) -> None:
        if bool(self.config.get("manual_anchor_required", True)) and not self.pose_tracker.manual_anchors_complete:
            self._start_anchor_setup()
            return
        self.active = False
        self.mapper = None
        self.calibration.start(now)
        self.pose_tracker.reset()
        self.last_valid_tracking = 0.0
        self.last_left_tracking = 0.0
        self.last_right_tracking = 0.0
        self._last_calibration_console_state = None
        print(
            f"[Calibration] Started {len(self.calibration.PHASES)}-pose calibration",
            flush=True,
        )
        # Calibration is much easier when the skeleton view cannot be
        # accidentally left hidden from an earlier F10 press.
        self._preview_enabled = True
        self._reset_controls()
        self.shared.update(
            mode="CALIBRATING",
            headline="Copy the pose diagram",
            detail=f"Pose 1 of {len(self.calibration.PHASES)} • GET READY",
            prompt=self.calibration.prompt,
            countdown=self.calibration.prepare_seconds,
            calibration_phase_index=0,
            calibration_phase_count=len(self.calibration.PHASES),
            calibration_phase_key=self.calibration.phase_name,
            calibration_stage="GET READY",
            calibration_progress=0.0,
            calibration_samples=0,
            left_foot_ok=False,
            right_foot_ok=False,
            steering_ok=False,
            pedals_only=self.pedals_only,
            tracking_hint="Waiting for the first camera frame",
            calibrated=False,
            active=False,
            tracking_ok=False,
            show_preview=True,
            anchor_setup_active=False,
            anchor_input_available=False,
        )

    def _process_calibration(self, features: PoseFeatures, now: float) -> None:
        self._reset_controls(update_ui=False)
        calibration_data = self.calibration.update(features, now)
        remaining, _capturing, progress, stage = self.calibration.progress()
        if self.calibration.failed_reason:
            stage = "FAILED"
        headline_by_stage = {
            "GET READY": "Copy the pose diagram",
            "CAPTURING": "Hold still — capturing this pose",
            "WAITING FOR TRACKING": "Capture paused — restore the missing landmarks",
            "FAILED": "Calibration needs another attempt",
        }
        phase_number = min(self.calibration.phase_index + 1, len(self.calibration.PHASES))
        console_state = (phase_number, stage, self.calibration.tracking_hint)
        if console_state != self._last_calibration_console_state:
            print(
                f"[Calibration] Pose {phase_number}/{len(self.calibration.PHASES)} "
                f"{stage}: {self.calibration.tracking_hint}",
                flush=True,
            )
            self._last_calibration_console_state = console_state
        self.shared.update(
            mode="CALIBRATING" if not self.calibration.failed_reason else "CALIBRATION FAILED",
            headline=headline_by_stage.get(stage, "Calibration"),
            detail=(
                f"Pose {phase_number} of {len(self.calibration.PHASES)} • {stage}"
                if not self.calibration.failed_reason
                else "No controls will be sent until calibration succeeds"
            ),
            prompt=self.calibration.prompt,
            countdown=remaining,
            calibration_phase_index=max(0, min(self.calibration.phase_index, len(self.calibration.PHASES) - 1)),
            calibration_phase_count=len(self.calibration.PHASES),
            calibration_phase_key=self.calibration.phase_name,
            calibration_stage=stage,
            calibration_progress=progress,
            calibration_samples=len(self.calibration.samples),
            left_foot_ok=features.left_foot_ok,
            right_foot_ok=features.right_foot_ok,
            steering_ok=features.steering_ok,
            tracking_hint=self.calibration.tracking_hint,
            tracking_ok=self.calibration.current_tracking_ok,
            gas=0.0,
            brake=0.0,
            steering=0.0,
        )

        if self.calibration.failed_reason:
            self.active = False
            self.mapper = None
            print(f"[Calibration] FAILED: {self.calibration.failed_reason}", flush=True)
            self.shared.update(calibrated=False, active=False)
            return

        if calibration_data is not None:
            self.mapper = ControlMapper(calibration_data, self.config)
            self.active = False
            print("[Calibration] COMPLETE. Controls remain paused until F8.", flush=True)
            self.shared.update(
                mode="PAUSED",
                headline="Calibration complete — controls remain neutral",
                detail=(
                    "Press F8 to enable pedals • steering stays centred • F9 recalibrates"
                    if self.pedals_only
                    else "Press F8 to enable driving • F9 recalibrates"
                ),
                prompt="",
                countdown=0.0,
                calibration_progress=1.0,
                calibration_stage="COMPLETE",
                tracking_hint="Calibration complete",
                calibrated=True,
                active=False,
                tracking_ok=(features.pedals_ok if self.pedals_only else features.all_controls_ok),
            )

    def _process_controls(self, features: PoseFeatures, now: float, dt: float) -> None:
        if not self.active or self.mapper is None:
            self._reset_controls(update_ui=False)
            tracking_ready = features.pedals_ok if self.pedals_only else features.all_controls_ok
            self.shared.update(
                tracking_ok=tracking_ready,
                left_foot_ok=features.left_foot_ok,
                right_foot_ok=features.right_foot_ok,
                steering_ok=(False if self.pedals_only else features.steering_ok),
                active=False,
                gas=0.0,
                brake=0.0,
                steering=0.0,
            )
            return

        if self.pedals_only:
            self._process_pedal_only_controls(features, now, dt)
            return

        if features.all_controls_ok:
            if features.all_controls_fresh:
                self.last_valid_tracking = now
            else:
                timeout = float(self.config["lost_tracking_timeout_seconds"])
                expired = (
                    self.last_valid_tracking <= 0.0
                    or (now - self.last_valid_tracking) >= timeout
                )
                if expired:
                    self._neutral_due_to_tracking_loss(now, force=True)
                    return

            steering_raw, gas_raw, brake_raw = self.mapper.map(features, dt=dt)
            steering = self.steer_smoother.update(steering_raw, dt)
            gas = self.gas_smoother.update(gas_raw, dt)
            brake = self.brake_smoother.update(brake_raw, dt)
            if self.controller is not None:
                self.controller.send(steering=steering, gas=gas, brake=brake)
            base_mode = (
                "ACTIVE" if (self.controller and self.controller.available)
                else "ACTIVE (preview only)"
            )
            bridging = not features.all_controls_fresh
            mode = base_mode + (" • BRIEF HOLD" if bridging else "")
            self.shared.update(
                mode=mode,
                headline=(
                    "Brief landmark interruption — holding the last verified pose"
                    if bridging
                    else "Tracking feet and hands with micro-movement refinement"
                ),
                detail="F8 pauses instantly • prolonged tracking loss sends neutral controls",
                tracking_ok=features.all_controls_fresh,
                left_foot_ok=features.left_foot_ok,
                right_foot_ok=features.right_foot_ok,
                steering_ok=features.steering_ok,
                active=True,
                gas=gas,
                brake=brake,
                steering=steering,
            )
            return

        self._neutral_due_to_tracking_loss(now)

    def _usable_foot_feature(
        self,
        features: PoseFeatures,
        side: str,
        now: float,
    ) -> tuple[Optional[FootFeatureObservation], bool]:
        timeout = float(self.config["lost_tracking_timeout_seconds"])
        if side == "left":
            value = features.left_foot
            ok = features.left_foot_ok
            fresh = features.left_foot_fresh
            last = self.last_left_tracking
        elif side == "right":
            value = features.right_foot
            ok = features.right_foot_ok
            fresh = features.right_foot_fresh
            last = self.last_right_tracking
        else:
            raise ValueError("side must be 'left' or 'right'")

        if ok and value is not None and fresh:
            if side == "left":
                self.last_left_tracking = now
            else:
                self.last_right_tracking = now
            return value, True
        if ok and value is not None and last > 0.0 and now - last < timeout:
            return value, False
        return None, False

    def _process_pedal_only_controls(
        self,
        features: PoseFeatures,
        now: float,
        dt: float,
    ) -> None:
        assert self.mapper is not None
        left_feature, left_fresh = self._usable_foot_feature(features, "left", now)
        right_feature, right_fresh = self._usable_foot_feature(features, "right", now)
        gas_raw, brake_raw = self.mapper.map_pedal_features(
            left_feature, right_feature, dt=dt
        )
        gas = self.gas_smoother.update(0.0 if gas_raw is None else gas_raw, dt)
        brake = self.brake_smoother.update(0.0 if brake_raw is None else brake_raw, dt)
        steering = 0.0
        if self.controller is not None:
            self.controller.send(steering=steering, gas=gas, brake=brake)

        left_ok = left_feature is not None
        right_ok = right_feature is not None
        both_ok = left_ok and right_ok
        both_fresh = left_fresh and right_fresh
        base_mode = (
            "ACTIVE • PEDALS ONLY"
            if (self.controller and self.controller.available)
            else "ACTIVE • PEDALS ONLY (preview)"
        )
        held = (left_ok and not left_fresh) or (right_ok and not right_fresh)
        if not both_ok:
            mode = base_mode + " • PARTIAL TRACKING"
            missing = []
            if not left_ok:
                missing.append("left foot: brake neutral")
            if not right_ok:
                missing.append("right foot: throttle neutral")
            headline = " • ".join(missing)
        elif held:
            mode = base_mode + " • BRIEF HOLD"
            headline = "Brief foot-landmark interruption — holding only the verified pedal"
        else:
            mode = base_mode
            headline = "Camera-rate micro-foot tracking is driving the pedals"

        self.shared.update(
            mode=mode,
            headline=headline,
            detail="RT throttle • LT brake • preview auto-hides for maximum tracking rate • F8 pauses",
            tracking_ok=both_fresh,
            left_foot_ok=left_ok,
            right_foot_ok=right_ok,
            steering_ok=False,
            active=True,
            gas=gas,
            brake=brake,
            steering=0.0,
        )

    def _neutral_due_to_tracking_loss(self, now: float, force: bool = False) -> None:
        timeout = float(self.config["lost_tracking_timeout_seconds"])
        expired = force or self.last_valid_tracking <= 0.0 or (now - self.last_valid_tracking) >= timeout
        if expired:
            self._reset_controls(update_ui=False)
            headline = "Tracking lost — controls forced to neutral"
        else:
            headline = "Tracking interrupted — neutral will engage immediately if it continues"
        self.shared.update(
            mode="TRACKING LOST",
            headline=headline,
            detail=(
                "Keep both hard-locked heel, ankle, and toe triangles visible"
                if self.pedals_only
                else "Keep both wrists and both feet inside the camera view"
            ),
            tracking_ok=False,
            active=self.active,
            gas=0.0 if expired else self.shared.snapshot().gas,
            brake=0.0 if expired else self.shared.snapshot().brake,
            steering=0.0 if expired else self.shared.snapshot().steering,
        )

    def _reset_controls(self, update_ui: bool = True) -> None:
        self.steer_smoother.reset(0.0)
        self.gas_smoother.reset(0.0)
        self.brake_smoother.reset(0.0)
        if self.mapper is not None:
            self.mapper.reset()
        if self.controller is not None:
            self.controller.neutral()
        if update_ui:
            self.shared.update(gas=0.0, brake=0.0, steering=0.0)


class OverlayWindow(QWidget):
    def __init__(
        self,
        shared: SharedState,
        stop_event: threading.Event,
        monitor_index: int,
        windowed_hud: bool = False,
    ) -> None:
        super().__init__()
        self.shared = shared
        self.stop_event = stop_event
        self.windowed_hud = bool(windowed_hud)
        self._last_snapshot = shared.snapshot()
        self._anchor_image_rect = QRectF()
        self.setWindowTitle(APP_NAME)

        screens = QGuiApplication.screens()
        if not screens:
            raise RuntimeError("No display screen was found")
        index = max(0, min(int(monitor_index), len(screens) - 1))
        screen_geometry = screens[index].availableGeometry()

        if self.windowed_hud:
            # A normal opaque window is intentionally used by preview-only mode.
            # It is easier to diagnose than a click-through desktop overlay and
            # cannot disappear behind an exclusive-fullscreen game.
            self.setWindowFlags(Qt.WindowType.Window)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
            self.setMinimumSize(900, 560)
            target_width = min(1280, max(900, screen_geometry.width() - 80))
            target_height = min(760, max(560, screen_geometry.height() - 100))
            self.resize(target_width, target_height)
            self.move(
                screen_geometry.x() + (screen_geometry.width() - target_width) // 2,
                screen_geometry.y() + (screen_geometry.height() - target_height) // 2,
            )
        else:
            flags = (
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
                | Qt.WindowType.NoDropShadowWindowHint
            )
            if hasattr(Qt.WindowType, "WindowTransparentForInput"):
                flags |= Qt.WindowType.WindowTransparentForInput
            self.setWindowFlags(flags)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.setGeometry(screens[index].geometry())

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        if not self.windowed_hud:
            self._apply_windows_click_through()

    def closeEvent(self, event: Any) -> None:
        self.stop_event.set()
        super().closeEvent(event)

    def mousePressEvent(self, event: Any) -> None:
        snapshot = self._last_snapshot
        if (
            self.windowed_hud
            and snapshot.anchor_setup_active
            and snapshot.anchor_input_available
            and event.button() == Qt.MouseButton.LeftButton
        ):
            position = event.position()
            point = preview_click_to_normalized(
                float(position.x()),
                float(position.y()),
                (
                    float(self._anchor_image_rect.left()),
                    float(self._anchor_image_rect.top()),
                    float(self._anchor_image_rect.width()),
                    float(self._anchor_image_rect.height()),
                ),
                snapshot.preview_mirrored,
            )
            if point is not None:
                self.shared.submit_anchor_click(*point)
                event.accept()
                return
        super().mousePressEvent(event)

    def _tick(self) -> None:
        self._last_snapshot = self.shared.snapshot()
        if self._last_snapshot.exit_requested or self.stop_event.is_set():
            QApplication.instance().quit()
            return
        self.update()

    def _apply_windows_click_through(self) -> None:
        if sys.platform != "win32" or not hasattr(ctypes, "windll"):
            return
        try:
            hwnd = int(self.winId())
            GWL_EXSTYLE = -20
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_LAYERED = 0x00080000
            WS_EX_NOACTIVATE = 0x08000000
            HWND_TOPMOST = -1
            SWP_NOSIZE = 0x0001
            SWP_NOMOVE = 0x0002
            SWP_NOACTIVATE = 0x0010
            user32 = ctypes.windll.user32
            get_style = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
            set_style = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
            style = get_style(hwnd, GWL_EXSTYLE)
            set_style(
                hwnd,
                GWL_EXSTYLE,
                style | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_NOACTIVATE,
            )
            user32.SetWindowPos(
                hwnd,
                HWND_TOPMOST,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
            )
        except Exception:
            # Qt flags already provide a useful fallback.
            pass

    def paintEvent(self, event: Any) -> None:
        snapshot = self._last_snapshot
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        width = float(self.width())
        height = float(self.height())
        if self.windowed_hud:
            painter.fillRect(self.rect(), QColor(13, 18, 28))

        top_width = min(780.0, max(520.0, width - 48.0))
        top_panel = QRectF(width / 2.0 - top_width / 2.0, 20.0, top_width, 108.0)
        self._panel(painter, top_panel, opacity=205 if self.windowed_hud else 185)

        status_color = QColor(85, 220, 140) if snapshot.tracking_ok else QColor(255, 190, 75)
        if snapshot.mode in {"ERROR", "CALIBRATION FAILED", "TRACKING LOST", "CAMERA LOST"}:
            status_color = QColor(255, 105, 105)

        painter.setPen(status_color)
        painter.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        painter.drawText(
            QRectF(top_panel.left() + 22.0, top_panel.top() + 10.0, top_panel.width() - 44.0, 30.0),
            Qt.AlignmentFlag.AlignCenter,
            snapshot.mode,
        )
        painter.setPen(QColor(245, 248, 252))
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(top_panel.left() + 20.0, top_panel.top() + 43.0, top_panel.width() - 40.0, 25.0),
            Qt.AlignmentFlag.AlignCenter,
            snapshot.headline,
        )
        painter.setPen(QColor(200, 208, 220))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(
            QRectF(top_panel.left() + 18.0, top_panel.top() + 70.0, top_panel.width() - 36.0, 27.0),
            Qt.AlignmentFlag.AlignCenter,
            snapshot.detail,
        )

        calibration_visible = snapshot.mode in {"CALIBRATING", "CALIBRATION FAILED"}
        if snapshot.anchor_setup_active:
            self._draw_anchor_setup(painter, snapshot, width, height)
        elif calibration_visible:
            self._anchor_image_rect = QRectF()
            self._draw_calibration_panel(painter, snapshot, width, height)
        else:
            self._anchor_image_rect = QRectF()
            if snapshot.pedals_only:
                self._draw_pedal_only_status(painter, width, height)
            else:
                self._draw_steering(painter, snapshot, width, height)
            self._draw_pedal(
                painter,
                QRectF(38.0, height - 330.0, 92.0, 260.0),
                snapshot.brake,
                "BRAKE",
                QColor(255, 95, 95),
            )
            self._draw_pedal(
                painter,
                QRectF(width - 130.0, height - 330.0, 92.0, 260.0),
                snapshot.gas,
                "GAS",
                QColor(80, 225, 135),
            )
            if snapshot.prompt:
                self._draw_generic_prompt(painter, snapshot, width, height)
            if snapshot.show_preview and snapshot.preview is not None:
                self._draw_preview(painter, snapshot, width, height)

        # Small diagnostic strip.
        diag_width = min(760.0, max(560.0, width - 48.0))
        diag_rect = QRectF(width / 2.0 - diag_width / 2.0, height - 50.0, diag_width, 32.0)
        self._panel(painter, diag_rect, opacity=165 if self.windowed_hud else 145)
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(205, 214, 226))
        painter.drawText(
            diag_rect.adjusted(12.0, 0.0, -12.0, 0.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            snapshot.controller_message,
        )
        painter.drawText(
            diag_rect.adjusted(12.0, 0.0, -12.0, 0.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            (
                f"TRACK {snapshot.fps:0.0f} • "
                f"AI {snapshot.ai_fps:0.0f} • "
                f"CAM {snapshot.camera_fps:0.0f} • "
                f"LIGHT {snapshot.light_luma:0.0f} • "
                f"DARK {snapshot.light_dark_fraction * 100.0:0.0f}%"
            ),
        )

    def _draw_anchor_setup(
        self,
        painter: QPainter,
        snapshot: UiSnapshot,
        width: float,
        height: float,
    ) -> None:
        panel = QRectF(24.0, 142.0, max(420.0, width - 48.0), max(300.0, height - 210.0))
        panel.setWidth(min(panel.width(), width - 48.0))
        panel.setHeight(min(panel.height(), height - panel.top() - 62.0))
        self._panel(painter, panel, opacity=235)
        inner = panel.adjusted(22.0, 18.0, -22.0, -20.0)
        instruction_height = 76.0
        image_area = QRectF(
            inner.left(),
            inner.top() + instruction_height,
            inner.width(),
            max(120.0, inner.height() - instruction_height),
        )
        image = snapshot.preview
        if image is None or image.width() <= 0 or image.height() <= 0:
            self._anchor_image_rect = QRectF()
            painter.setPen(QColor(245, 248, 252))
            painter.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
            painter.drawText(image_area, Qt.AlignmentFlag.AlignCenter, "Waiting for camera frame")
        else:
            aspect = float(image.width()) / max(1.0, float(image.height()))
            target_width = min(image_area.width(), image_area.height() * aspect)
            target_height = target_width / aspect
            if target_height > image_area.height():
                target_height = image_area.height()
                target_width = target_height * aspect
            target = QRectF(
                image_area.center().x() - target_width / 2.0,
                image_area.center().y() - target_height / 2.0,
                target_width,
                target_height,
            )
            self._anchor_image_rect = target
            painter.setPen(QPen(QColor(125, 155, 190), 2.0))
            painter.setBrush(QColor(20, 28, 40))
            painter.drawRoundedRect(target.adjusted(-3.0, -3.0, 3.0, 3.0), 9.0, 9.0)
            painter.drawImage(target, image)

            def display_point(index: int) -> Optional[QPointF]:
                value = snapshot.anchor_setup_points.get(index)
                if value is None:
                    return None
                x, y = value
                if snapshot.preview_mirrored:
                    x = 1.0 - x
                return QPointF(target.left() + x * target.width(), target.top() + y * target.height())

            for side, indices in MANUAL_FOOT_TRIANGLES.items():
                color = QColor(80, 225, 135) if side == "right" else QColor(255, 105, 125)
                heel, ankle, toe = indices
                triangle = [display_point(heel), display_point(ankle), display_point(toe)]
                painter.setPen(QPen(color, 3.0))
                if triangle[0] is not None and triangle[1] is not None:
                    painter.drawLine(triangle[0], triangle[1])
                if triangle[1] is not None and triangle[2] is not None:
                    painter.drawLine(triangle[1], triangle[2])
                if triangle[2] is not None and triangle[0] is not None:
                    painter.drawLine(triangle[2], triangle[0])
                for point_index, point in zip(indices, triangle):
                    if point is None:
                        continue
                    painter.setBrush(QColor(color.red(), color.green(), color.blue(), 80))
                    painter.drawEllipse(point, 10.0, 10.0)
                    if point_index in {LEFT_HEEL, RIGHT_HEEL}:
                        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                        painter.drawText(
                            QRectF(point.x() - 38.0, point.y() - 31.0, 76.0, 20.0),
                            Qt.AlignmentFlag.AlignCenter,
                            "PIVOT",
                        )

        painter.setPen(QColor(105, 215, 255))
        painter.setFont(QFont("Segoe UI", 17, QFont.Weight.Bold))
        painter.drawText(
            QRectF(inner.left(), inner.top(), inner.width(), 34.0),
            Qt.AlignmentFlag.AlignCenter,
            snapshot.prompt,
        )
        painter.setPen(QColor(215, 225, 238))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(inner.left(), inner.top() + 37.0, inner.width(), 30.0),
            Qt.AlignmentFlag.AlignCenter,
            "Use the exact skin/shoe texture point • F7 clears all six points and restarts",
        )

    def _draw_generic_prompt(
        self,
        painter: QPainter,
        snapshot: UiSnapshot,
        width: float,
        height: float,
    ) -> None:
        prompt_width = min(860.0, width - 48.0)
        prompt_rect = QRectF(width / 2.0 - prompt_width / 2.0, height / 2.0 - 100.0, prompt_width, 200.0)
        self._panel(painter, prompt_rect, opacity=225)
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        painter.drawText(
            QRectF(prompt_rect.left() + 35.0, prompt_rect.top() + 32.0, prompt_rect.width() - 70.0, 92.0),
            Qt.AlignmentFlag.AlignCenter.value | Qt.TextFlag.TextWordWrap.value,
            snapshot.prompt,
        )
        if snapshot.countdown > 0.0:
            painter.setPen(QColor(110, 215, 255))
            painter.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
            painter.drawText(
                QRectF(prompt_rect.left(), prompt_rect.bottom() - 68.0, prompt_rect.width(), 50.0),
                Qt.AlignmentFlag.AlignCenter,
                f"{snapshot.countdown:0.1f}",
            )

    def _draw_calibration_panel(
        self,
        painter: QPainter,
        snapshot: UiSnapshot,
        width: float,
        height: float,
    ) -> None:
        panel = QRectF(24.0, 142.0, max(700.0, width - 48.0), max(340.0, height - 210.0))
        # Keep the panel inside unusually small displays rather than allowing a
        # fixed 860-pixel prompt to disappear off-screen.
        panel.setWidth(min(panel.width(), width - 48.0))
        panel.setHeight(min(panel.height(), height - panel.top() - 62.0))
        self._panel(painter, panel, opacity=232 if self.windowed_hud else 218)
        inner = panel.adjusted(18.0, 18.0, -18.0, -18.0)
        gap = 14.0
        camera_width = min(390.0, inner.width() * 0.35)
        pose_width = min(235.0, inner.width() * 0.21)
        text_width = inner.width() - camera_width - pose_width - gap * 2.0
        if text_width < 310.0:
            shortage = 310.0 - text_width
            camera_width = max(250.0, camera_width - shortage * 0.65)
            pose_width = max(155.0, pose_width - shortage * 0.35)
            text_width = inner.width() - camera_width - pose_width - gap * 2.0

        camera_rect = QRectF(inner.left(), inner.top(), camera_width, inner.height())
        pose_rect = QRectF(camera_rect.right() + gap, inner.top(), pose_width, inner.height())
        text_rect = QRectF(pose_rect.right() + gap, inner.top(), text_width, inner.height())

        self._draw_calibration_camera(painter, snapshot, camera_rect)
        self._draw_calibration_pose(
            painter,
            snapshot.calibration_phase_key,
            pose_rect,
            snapshot.pedals_only,
        )
        self._draw_calibration_instructions(painter, snapshot, text_rect)

    def _draw_calibration_camera(
        self,
        painter: QPainter,
        snapshot: UiSnapshot,
        rect: QRectF,
    ) -> None:
        self._panel(painter, rect, opacity=180)
        painter.setPen(QColor(235, 242, 250))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(rect.left(), rect.top() + 7.0, rect.width(), 22.0),
            Qt.AlignmentFlag.AlignCenter,
            "YOUR CAMERA + AI LANDMARKS",
        )

        available = rect.adjusted(10.0, 34.0, -10.0, -82.0)
        image_height = min(available.height(), available.width() * 9.0 / 16.0)
        image_rect = QRectF(
            available.left(),
            available.top() + max(0.0, (available.height() - image_height) / 2.0),
            available.width(),
            image_height,
        )
        painter.setPen(QPen(QColor(120, 145, 175, 130), 1.0))
        painter.setBrush(QColor(24, 31, 44, 230))
        painter.drawRoundedRect(image_rect, 8.0, 8.0)
        if snapshot.show_preview and snapshot.preview is not None:
            painter.drawImage(image_rect.adjusted(2.0, 2.0, -2.0, -2.0), snapshot.preview)
        else:
            painter.setPen(QColor(185, 196, 210))
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
            message = "Waiting for camera frame" if snapshot.show_preview else "Camera view hidden — press F10"
            painter.drawText(
                image_rect.adjusted(14.0, 14.0, -14.0, -14.0),
                Qt.AlignmentFlag.AlignCenter.value | Qt.TextFlag.TextWordWrap.value,
                message,
            )

        badge_top = rect.bottom() - 62.0
        badge_gap = 6.0
        badges = [
            ("LEFT FOOT", snapshot.left_foot_ok),
            ("RIGHT FOOT", snapshot.right_foot_ok),
        ]
        if not snapshot.pedals_only:
            badges.append(("HANDS", snapshot.steering_ok))
        badge_width = (
            rect.width() - 20.0 - badge_gap * max(0, len(badges) - 1)
        ) / max(1, len(badges))
        for index, (label, ok) in enumerate(badges):
            badge = QRectF(
                rect.left() + 10.0 + index * (badge_width + badge_gap),
                badge_top,
                badge_width,
                38.0,
            )
            self._draw_tracking_badge(painter, badge, label, ok)

    def _draw_tracking_badge(
        self,
        painter: QPainter,
        rect: QRectF,
        label: str,
        ok: bool,
    ) -> None:
        color = QColor(80, 225, 135) if ok else QColor(255, 112, 112)
        painter.setPen(QPen(color, 1.5))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 35))
        painter.drawRoundedRect(rect, 8.0, 8.0)
        painter.setPen(color)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, ("OK  " if ok else "MISS  ") + label)

    def _draw_calibration_pose(
        self,
        painter: QPainter,
        phase: str,
        rect: QRectF,
        pedals_only: bool,
    ) -> None:
        self._panel(painter, rect, opacity=180)
        painter.setPen(QColor(235, 242, 250))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(rect.left(), rect.top() + 7.0, rect.width(), 22.0),
            Qt.AlignmentFlag.AlignCenter,
            "POSE TO COPY",
        )

        body = rect.adjusted(16.0, 38.0, -16.0, -50.0)
        cx = body.center().x()
        top = body.top()
        w = body.width()
        h = body.height()
        base = QColor(175, 190, 210)
        accent = QColor(105, 215, 255)
        gas_color = QColor(80, 225, 135)
        brake_color = QColor(255, 105, 105)

        head = QPointF(cx, top + h * 0.09)
        shoulder_y = top + h * 0.24
        hip_y = top + h * 0.52
        knee_y = top + h * 0.70
        ankle_y = top + h * 0.86
        shoulder_left = QPointF(cx - w * 0.20, shoulder_y)
        shoulder_right = QPointF(cx + w * 0.20, shoulder_y)
        hip_left = QPointF(cx - w * 0.11, hip_y)
        hip_right = QPointF(cx + w * 0.11, hip_y)
        knee_left = QPointF(cx - w * 0.15, knee_y)
        knee_right = QPointF(cx + w * 0.15, knee_y)
        ankle_left = QPointF(cx - w * 0.18, ankle_y)
        ankle_right = QPointF(cx + w * 0.18, ankle_y)

        hand_y = top + h * 0.38
        turn_angle = 0.0
        if phase == "left":
            turn_angle = math.radians(-24.0)
        elif phase == "right":
            turn_angle = math.radians(24.0)
        half_hand = w * 0.36
        dx = half_hand * math.cos(turn_angle)
        dy = half_hand * math.sin(turn_angle)
        hand_left = QPointF(cx - dx, hand_y - dy)
        hand_right = QPointF(cx + dx, hand_y + dy)

        painter.setPen(QPen(base, 5.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(head, max(8.0, w * 0.075), max(8.0, w * 0.075))
        painter.drawLine(QPointF(cx, top + h * 0.16), QPointF(cx, hip_y))
        painter.drawLine(shoulder_left, shoulder_right)
        painter.drawLine(QPointF(cx, hip_y), hip_left)
        painter.drawLine(QPointF(cx, hip_y), hip_right)
        painter.drawLine(hip_left, knee_left)
        painter.drawLine(knee_left, ankle_left)
        painter.drawLine(hip_right, knee_right)
        painter.drawLine(knee_right, ankle_right)

        if not pedals_only:
            hand_color = accent if phase in {"neutral", "left", "right"} else base
            painter.setPen(QPen(hand_color, 7.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(hand_left, hand_right)
            painter.setPen(QPen(base, 4.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(shoulder_left, hand_left)
            painter.drawLine(shoulder_right, hand_right)

        left_foot_end = QPointF(ankle_left.x() - w * 0.08, ankle_left.y() + h * 0.08)
        right_foot_end = QPointF(ankle_right.x() + w * 0.08, ankle_right.y() + h * 0.08)
        left_color = brake_color if phase == "brake" else (accent if phase == "neutral" else base)
        right_color = gas_color if phase == "gas" else (accent if phase == "neutral" else base)
        painter.setPen(QPen(left_color, 8.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(ankle_left, left_foot_end)
        painter.setPen(QPen(right_color, 8.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(ankle_right, right_foot_end)

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(left_color)
        painter.drawText(QRectF(left_foot_end.x() - 20.0, left_foot_end.y() + 4.0, 40.0, 20.0), Qt.AlignmentFlag.AlignCenter, "L")
        painter.setPen(right_color)
        painter.drawText(QRectF(right_foot_end.x() - 20.0, right_foot_end.y() + 4.0, 40.0, 20.0), Qt.AlignmentFlag.AlignCenter, "R")

        phase_labels = {
            "neutral": "RELAX BOTH FEET" if pedals_only else "RELAX + HANDS CENTRED",
            "gas": "PRESS RIGHT FOOT",
            "brake": "PRESS LEFT FOOT",
            "left": "TURN HANDS LEFT",
            "right": "TURN HANDS RIGHT",
        }
        painter.setPen(QColor(245, 248, 252))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(
            QRectF(rect.left() + 8.0, rect.bottom() - 38.0, rect.width() - 16.0, 28.0),
            Qt.AlignmentFlag.AlignCenter.value | Qt.TextFlag.TextWordWrap.value,
            phase_labels.get(phase, "FOLLOW THE PROMPT"),
        )

    def _draw_calibration_instructions(
        self,
        painter: QPainter,
        snapshot: UiSnapshot,
        rect: QRectF,
    ) -> None:
        self._panel(painter, rect, opacity=180)
        inner = rect.adjusted(18.0, 14.0, -18.0, -14.0)
        count = max(1, snapshot.calibration_phase_count)
        active_index = max(0, min(snapshot.calibration_phase_index, count - 1))

        step_gap = 5.0
        step_width = (inner.width() - step_gap * (count - 1)) / count
        for index in range(count):
            step_rect = QRectF(
                inner.left() + index * (step_width + step_gap),
                inner.top(),
                step_width,
                32.0,
            )
            if index < active_index:
                color = QColor(80, 225, 135)
            elif index == active_index:
                color = QColor(105, 215, 255)
            else:
                color = QColor(105, 116, 134)
            painter.setPen(QPen(color, 1.5))
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 42))
            painter.drawRoundedRect(step_rect, 7.0, 7.0)
            painter.setPen(color)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.drawText(step_rect, Qt.AlignmentFlag.AlignCenter, str(index + 1))

        stage_rect = QRectF(inner.left(), inner.top() + 43.0, inner.width(), 34.0)
        stage_color = QColor(105, 215, 255)
        if snapshot.calibration_stage == "WAITING FOR TRACKING":
            stage_color = QColor(255, 190, 75)
        elif snapshot.calibration_stage == "FAILED":
            stage_color = QColor(255, 105, 105)
        elif snapshot.calibration_stage == "CAPTURING":
            stage_color = QColor(80, 225, 135)
        painter.setPen(QPen(stage_color, 1.5))
        painter.setBrush(QColor(stage_color.red(), stage_color.green(), stage_color.blue(), 35))
        painter.drawRoundedRect(stage_rect, 8.0, 8.0)
        painter.setPen(stage_color)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(stage_rect, Qt.AlignmentFlag.AlignCenter, snapshot.calibration_stage)

        prompt_top = stage_rect.bottom() + 12.0
        prompt_height = max(82.0, inner.height() * 0.29)
        painter.setPen(QColor(250, 252, 255))
        painter.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        painter.drawText(
            QRectF(inner.left(), prompt_top, inner.width(), prompt_height),
            Qt.AlignmentFlag.AlignCenter.value | Qt.TextFlag.TextWordWrap.value,
            snapshot.prompt,
        )

        hint_top = prompt_top + prompt_height + 4.0
        hint_height = max(54.0, inner.height() * 0.18)
        hint_color = QColor(200, 212, 228) if snapshot.tracking_ok else QColor(255, 188, 105)
        painter.setPen(hint_color)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(inner.left(), hint_top, inner.width(), hint_height),
            Qt.AlignmentFlag.AlignCenter.value | Qt.TextFlag.TextWordWrap.value,
            snapshot.tracking_hint,
        )

        bar_height = 18.0
        bar_rect = QRectF(inner.left(), inner.bottom() - 58.0, inner.width(), bar_height)
        painter.setPen(QPen(QColor(135, 150, 174, 130), 1.0))
        painter.setBrush(QColor(38, 47, 63, 235))
        painter.drawRoundedRect(bar_rect, bar_height / 2.0, bar_height / 2.0)
        fill_width = bar_rect.width() * clip(snapshot.calibration_progress, 0.0, 1.0)
        if fill_width > 0.5:
            fill_rect = QRectF(bar_rect.left(), bar_rect.top(), fill_width, bar_rect.height())
            fill_color = QColor(80, 225, 135) if snapshot.calibration_stage == "CAPTURING" else QColor(105, 215, 255)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill_color)
            painter.drawRoundedRect(fill_rect, bar_height / 2.0, bar_height / 2.0)

        if snapshot.calibration_stage == "GET READY":
            footer = f"Capture starts in {snapshot.countdown:0.1f} s"
        elif snapshot.calibration_stage == "CAPTURING":
            footer = f"Hold for {snapshot.countdown:0.1f} s  •  {snapshot.calibration_samples} valid frames"
        elif snapshot.calibration_stage == "WAITING FOR TRACKING":
            footer = "Progress is paused until the required landmarks turn green"
        elif snapshot.calibration_stage == "FAILED":
            footer = "Press F9 to restart calibration"
        else:
            footer = ""
        painter.setPen(QColor(215, 225, 238))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(inner.left(), bar_rect.bottom() + 7.0, inner.width(), 24.0),
            Qt.AlignmentFlag.AlignCenter,
            footer,
        )

    def _panel(self, painter: QPainter, rect: QRectF, opacity: int) -> None:
        painter.setPen(QPen(QColor(150, 170, 200, 100), 1.0))
        painter.setBrush(QColor(10, 15, 24, opacity))
        painter.drawRoundedRect(rect, 14.0, 14.0)

    def _draw_pedal(
        self,
        painter: QPainter,
        rect: QRectF,
        value: float,
        label: str,
        color: QColor,
    ) -> None:
        self._panel(painter, rect, opacity=175)
        inner = rect.adjusted(18.0, 46.0, -18.0, -22.0)
        painter.setPen(QPen(QColor(180, 190, 205, 150), 1.0))
        painter.setBrush(QColor(40, 48, 62, 210))
        painter.drawRoundedRect(inner, 8.0, 8.0)
        fill_height = inner.height() * clip(value, 0.0, 1.0)
        fill = QRectF(inner.left(), inner.bottom() - fill_height, inner.width(), fill_height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(fill, 7.0, 7.0)
        painter.setPen(QColor(245, 248, 252))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(QRectF(rect.left(), rect.top() + 12.0, rect.width(), 24.0), Qt.AlignmentFlag.AlignCenter, label)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(rect.left(), rect.bottom() - 36.0, rect.width(), 24.0),
            Qt.AlignmentFlag.AlignCenter,
            f"{value * 100:0.0f}%",
        )

    def _draw_pedal_only_status(self, painter: QPainter, width: float, height: float) -> None:
        rect = QRectF(width / 2.0 - 260.0, height - 140.0, 520.0, 62.0)
        self._panel(painter, rect, opacity=175)
        painter.setPen(QColor(105, 215, 255))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(
            QRectF(rect.left() + 12.0, rect.top() + 8.0, rect.width() - 24.0, 22.0),
            Qt.AlignmentFlag.AlignCenter,
            "PEDAL-ONLY MODE",
        )
        painter.setPen(QColor(215, 225, 238))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(rect.left() + 12.0, rect.top() + 31.0, rect.width() - 24.0, 20.0),
            Qt.AlignmentFlag.AlignCenter,
            "Steering output is centred — use another input device for steering",
        )

    def _draw_steering(self, painter: QPainter, snapshot: UiSnapshot, width: float, height: float) -> None:
        rect = QRectF(width / 2.0 - 300.0, height - 140.0, 600.0, 62.0)
        self._panel(painter, rect, opacity=175)
        track = QRectF(rect.left() + 38.0, rect.top() + 29.0, rect.width() - 76.0, 12.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(50, 60, 76, 230))
        painter.drawRoundedRect(track, 6.0, 6.0)
        centre_x = track.center().x()
        painter.setPen(QPen(QColor(215, 225, 238, 170), 2.0))
        painter.drawLine(QPointF(centre_x, track.top() - 8.0), QPointF(centre_x, track.bottom() + 8.0))
        marker_x = centre_x + clip(snapshot.steering, -1.0, 1.0) * track.width() / 2.0
        painter.setPen(QPen(QColor(105, 215, 255), 4.0))
        painter.setBrush(QColor(105, 215, 255))
        painter.drawEllipse(QPointF(marker_x, track.center().y()), 10.0, 10.0)
        painter.setPen(QColor(245, 248, 252))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(QRectF(rect.left(), rect.top() + 4.0, rect.width(), 20.0), Qt.AlignmentFlag.AlignCenter, "STEERING")
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(QRectF(rect.left() + 12.0, rect.top() + 25.0, 40.0, 20.0), Qt.AlignmentFlag.AlignCenter, "L")
        painter.drawText(QRectF(rect.right() - 52.0, rect.top() + 25.0, 40.0, 20.0), Qt.AlignmentFlag.AlignCenter, "R")

    def _draw_preview(self, painter: QPainter, snapshot: UiSnapshot, width: float, height: float) -> None:
        image = snapshot.preview
        if image is None:
            return
        target = QRectF(width - image.width() - 38.0, 158.0, float(image.width()), float(image.height()))
        frame = target.adjusted(-8.0, -32.0, 8.0, 8.0)
        self._panel(painter, frame, opacity=185)
        painter.drawImage(target, image)
        painter.setPen(QColor(235, 242, 250))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(frame.left(), frame.top() + 5.0, frame.width(), 22.0),
            Qt.AlignmentFlag.AlignCenter,
            "LOCAL AI CAMERA • F10 HIDE",
        )


def preview_click_to_normalized(
    x: float,
    y: float,
    image_rect: tuple[float, float, float, float],
    mirrored: bool,
) -> Optional[tuple[float, float]]:
    left, top, width, height = (float(value) for value in image_rect)
    if width <= 0.0 or height <= 0.0:
        return None
    u = (float(x) - left) / width
    v = (float(y) - top) / height
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return None
    if mirrored:
        u = 1.0 - u
    return float(u), float(v)


def validate_manual_anchor_points(
    points: dict[int, np.ndarray],
    frame_shape: tuple[int, ...],
    minimum_separation_pixels: float,
) -> Optional[str]:
    if set(points) != set(MANUAL_FOOT_ANCHOR_INDICES):
        return "all six heel, ankle, and toe points are required"
    if len(frame_shape) < 2:
        return "camera frame size is unavailable"
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if width <= 1 or height <= 1:
        return "camera frame size is invalid"
    scale = np.array([width, height], dtype=np.float64)
    minimum = max(2.0, float(minimum_separation_pixels))
    for side, indices in MANUAL_FOOT_TRIANGLES.items():
        values: list[np.ndarray] = []
        for index in indices:
            value = np.asarray(points.get(index), dtype=np.float64)
            if value.shape != (2,) or not np.all(np.isfinite(value)):
                return f"{side} anchor coordinates are invalid"
            if not (0.0 <= value[0] <= 1.0 and 0.0 <= value[1] <= 1.0):
                return f"{side} anchors must be inside the camera image"
            values.append(value * scale)
        for start, end in ((0, 1), (1, 2), (2, 0)):
            if float(np.linalg.norm(values[end] - values[start])) < minimum:
                return f"{side} anchor points are too close together"
        twice_area = abs(
            float(
                (values[1][0] - values[0][0]) * (values[2][1] - values[0][1])
                - (values[1][1] - values[0][1]) * (values[2][0] - values[0][0])
            )
        )
        if twice_area < minimum * minimum:
            return f"{side} heel, ankle, and toe must form a clear triangle"
    left_centroid = np.mean(
        [np.asarray(points[index], dtype=np.float64) * scale for index in MANUAL_FOOT_TRIANGLES["left"]],
        axis=0,
    )
    right_centroid = np.mean(
        [np.asarray(points[index], dtype=np.float64) * scale for index in MANUAL_FOOT_TRIANGLES["right"]],
        axis=0,
    )
    if float(np.linalg.norm(left_centroid - right_centroid)) < minimum * 2.0:
        return "left and right foot anchors overlap"
    return None


def clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def circular_mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        raise ValueError("Cannot average an empty angle list")
    sin_mean = sum(math.sin(v) for v in values_list) / len(values_list)
    cos_mean = sum(math.cos(v) for v in values_list) / len(values_list)
    return math.atan2(sin_mean, cos_mean)


def median_vector(vectors: Sequence[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("Cannot average an empty vector list")
    stacked = np.stack([np.asarray(v, dtype=np.float64) for v in vectors], axis=0)
    if not np.all(np.isfinite(stacked)):
        raise ValueError("Pose samples contained non-finite values")
    return np.median(stacked, axis=0)


def normalized_median(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Backward-compatible helper retained for external scripts/tests."""
    median = median_vector(vectors)
    norm = float(np.linalg.norm(median))
    if norm < 1e-8:
        raise ValueError("Pose feature vector collapsed to zero")
    return median / norm


def robust_center_and_noise(
    vectors: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Median centre and dimension-wise robust sigma (scaled MAD)."""
    if not vectors:
        raise ValueError("Cannot analyse an empty vector list")
    stacked = np.stack([np.asarray(v, dtype=np.float64) for v in vectors], axis=0)
    if not np.all(np.isfinite(stacked)):
        raise ValueError("Pose samples contained non-finite values")
    centre = np.median(stacked, axis=0)
    mad = np.median(np.abs(stacked - centre), axis=0)
    noise = mad * 1.4826
    return centre, noise


def robust_observation_stats(
    observations: Sequence[FootFeatureObservation],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Masked median, robust noise, and reliability for foot observations."""

    if not observations:
        raise ValueError("Cannot analyse an empty observation list")
    values = np.stack(
        [np.asarray(item.values, dtype=np.float64) for item in observations], axis=0
    )
    validity = np.stack(
        [np.asarray(item.validity, dtype=np.float64) for item in observations], axis=0
    )
    if values.shape != validity.shape or not np.all(np.isfinite(values)):
        raise ValueError("Foot observations contained invalid values")
    dimension_count = values.shape[1]
    centre = np.zeros(dimension_count, dtype=np.float64)
    noise = np.full(dimension_count, 1e6, dtype=np.float64)
    reliability = np.zeros(dimension_count, dtype=np.float64)
    required_samples = max(3, int(math.ceil(len(observations) * 0.45)))
    for index in range(dimension_count):
        valid = np.isfinite(validity[:, index]) & (validity[:, index] >= 0.05)
        count = int(valid.sum())
        if count < required_samples:
            continue
        selected = values[valid, index]
        median = float(np.median(selected))
        centre[index] = median
        noise[index] = float(np.median(np.abs(selected - median)) * 1.4826)
        coverage = count / max(1.0, float(len(observations)))
        reliability[index] = clip(
            coverage * float(np.median(validity[valid, index])), 0.0, 1.0
        )
    return centre, noise, reliability


def unit_vector(vector: np.ndarray) -> Optional[np.ndarray]:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        return None
    return vector / norm


def feature_signal_to_noise(
    neutral: np.ndarray,
    pressed: np.ndarray,
    noise: np.ndarray,
    noise_floor: float,
    weights: Optional[np.ndarray] = None,
) -> float:
    axis = np.asarray(pressed, dtype=np.float64) - np.asarray(neutral, dtype=np.float64)
    scale = np.maximum(np.abs(np.asarray(noise, dtype=np.float64)), max(1e-6, float(noise_floor)))
    normalized = axis / scale
    if weights is None:
        return float(np.linalg.norm(normalized))
    weight_array = np.clip(np.asarray(weights, dtype=np.float64), 0.0, 1.0)
    return float(np.sqrt(np.sum(weight_array * np.square(normalized))))


def project_heel_hinge(
    current: FootFeatureObservation,
    neutral: np.ndarray,
    pressed: np.ndarray,
    minimum_tilt_degrees: float = 2.0,
    extension_weight: float = 0.15,
) -> Optional[float]:
    """Map only the calibrated heel-to-toe tilt and optional radial extension."""

    values = np.asarray(current.values, dtype=np.float64)
    neutral = np.asarray(neutral, dtype=np.float64)
    pressed = np.asarray(pressed, dtype=np.float64)
    validity = np.asarray(current.validity, dtype=np.float64)
    if (
        values.shape != (FOOT_FEATURE_DIMENSION,)
        or neutral.shape != values.shape
        or pressed.shape != values.shape
        or validity.shape != values.shape
        or not np.all(np.isfinite(values))
        or float(np.min(validity)) < 0.05
    ):
        return None

    neutral_angle = math.atan2(float(neutral[0]), float(neutral[1]))
    pressed_angle = math.atan2(float(pressed[0]), float(pressed[1]))
    current_angle = math.atan2(float(values[0]), float(values[1]))
    calibrated_tilt = wrap_angle(pressed_angle - neutral_angle)
    minimum_tilt = math.radians(max(0.1, float(minimum_tilt_degrees)))
    if abs(calibrated_tilt) < minimum_tilt:
        return None

    current_tilt = wrap_angle(current_angle - neutral_angle)
    if current_tilt * calibrated_tilt <= 0.0:
        return 0.0
    tilt_fraction = max(0.0, current_tilt / calibrated_tilt)

    weight = clip(extension_weight, 0.0, 0.45)
    calibrated_extension = float(pressed[2] - neutral[2])
    current_extension = float(values[2] - neutral[2])
    if (
        abs(calibrated_extension) < 0.002
        or current_extension * calibrated_extension <= 0.0
    ):
        weight = 0.0
        extension_fraction = tilt_fraction
    else:
        extension_fraction = max(0.0, current_extension / calibrated_extension)
    value = (1.0 - weight) * tilt_fraction + weight * extension_fraction
    return clip(value, 0.0, 1.0)


def project_pedal(
    current: FootFeatureObservation,
    neutral: np.ndarray,
    pressed: np.ndarray,
    noise: Optional[np.ndarray] = None,
    calibration_weights: Optional[np.ndarray] = None,
    noise_floor: float = 0.003,
    noise_multiplier: float = 1.0,
    minimum_coverage: float = 0.35,
    direction_tolerance_degrees: float = 55.0,
    magnitude_blend: float = 0.25,
) -> Optional[float]:
    """Project valid shin-local foot geometry onto its calibrated action.

    Missing dimensions are excluded instead of becoming numeric zeros. A small
    magnitude contribution accepts a comfortable press that bends along a nearby
    direction, while the direction gate rejects unrelated sideways motion.
    """
    values = np.asarray(current.values, dtype=np.float64)
    validity = np.clip(np.asarray(current.validity, dtype=np.float64), 0.0, 1.0)
    neutral = np.asarray(neutral, dtype=np.float64)
    pressed = np.asarray(pressed, dtype=np.float64)
    if values.shape != neutral.shape or validity.shape != neutral.shape:
        return None
    axis = pressed - neutral
    if noise is None:
        scale = np.ones_like(axis)
    else:
        scale = np.maximum(
            np.abs(np.asarray(noise, dtype=np.float64)) * max(0.1, float(noise_multiplier)),
            max(1e-6, float(noise_floor)),
        )
    scaled_axis = axis / scale
    if calibration_weights is None:
        reference_weights = np.ones_like(scaled_axis)
    else:
        reference_weights = np.clip(
            np.asarray(calibration_weights, dtype=np.float64), 0.0, 1.0
        )
    if reference_weights.shape != scaled_axis.shape:
        return None
    active_weights = reference_weights * validity
    reference_energy = float(np.sum(reference_weights * np.square(scaled_axis)))
    available_energy = float(np.sum(active_weights * np.square(scaled_axis)))
    if reference_energy < 1e-8:
        return None
    coverage = available_energy / reference_energy
    if coverage < clip(minimum_coverage, 0.05, 1.0) or available_energy < 1e-8:
        return None

    scaled_delta = (values - neutral) / scale
    numerator = float(np.sum(active_weights * scaled_delta * scaled_axis))
    parallel = numerator / available_energy
    delta_energy = float(np.sum(active_weights * np.square(scaled_delta)))
    direction_gate = 0.0
    magnitude = 0.0
    if delta_energy > 1e-10 and numerator > 0.0:
        cosine = clip(
            numerator / math.sqrt(max(1e-12, delta_energy * available_energy)),
            -1.0,
            1.0,
        )
        tolerance = math.cos(
            math.radians(clip(direction_tolerance_degrees, 0.0, 89.0))
        )
        normalized_direction = clip(
            (cosine - tolerance) / max(1e-6, 1.0 - tolerance),
            0.0,
            1.0,
        )
        # Keep near-calibrated articulation almost linear, while still taking
        # unrelated motion completely to zero outside the angular tolerance.
        direction_gate = normalized_direction ** 0.25
        magnitude = math.sqrt(delta_energy / available_energy)
    blend = clip(magnitude_blend, 0.0, 1.0)
    value = direction_gate * (
        (1.0 - blend) * max(0.0, parallel) + blend * magnitude
    )
    return clip(value, 0.0, 1.35)


def adaptive_pedal_deadzone(base: float, signal_to_noise: float, factor: float) -> float:
    base = clip(base, 0.0, 0.4)
    signal_to_noise = max(1.0, float(signal_to_noise))
    adaptive = max(0.0, float(factor)) / signal_to_noise
    return clip(max(base, adaptive), 0.0, 0.18)


def shape_unipolar(
    value: float,
    deadzone: float,
    exponent: float,
    response_floor: float = 0.0,
    response_boost: float = 0.0,
) -> float:
    value = clip(value, 0.0, 1.0)
    deadzone = clip(deadzone, 0.0, 0.95)
    if value <= deadzone:
        return 0.0
    remapped = (value - deadzone) / (1.0 - deadzone)
    shaped = clip(remapped ** max(0.1, exponent), 0.0, 1.0)
    boost = clip(response_boost, 0.0, 2.0)
    shaped = clip(shaped + boost * shaped * (1.0 - shaped), 0.0, 1.0)
    floor = clip(response_floor, 0.0, 0.50)
    return clip(floor + (1.0 - floor) * shaped, 0.0, 1.0)


def shape_bipolar(value: float, deadzone: float, exponent: float) -> float:
    value = clip(value, -1.25, 1.25)
    magnitude = abs(value)
    deadzone = clip(deadzone, 0.0, 0.95)
    if magnitude <= deadzone:
        return 0.0
    remapped = (magnitude - deadzone) / (1.0 - deadzone)
    shaped = clip(remapped, 0.0, 1.0) ** max(0.1, exponent)
    return math.copysign(shaped, value)


def landmark_confidence(landmark: Any) -> float:
    visibility = float(getattr(landmark, "visibility", 1.0) or 0.0)
    presence = float(getattr(landmark, "presence", 1.0) or 0.0)
    return min(visibility, presence)


def landmark_xyz(landmark: Any) -> np.ndarray:
    return np.array(
        [float(landmark.x), float(landmark.y), float(landmark.z)],
        dtype=np.float64,
    )


def pose_anchor_from_result(
    result: Any,
    gray: np.ndarray,
    sequence: int,
    captured_at: float,
    completed_at: float,
) -> PoseAnchorPacket:
    pose_lists = getattr(result, "pose_landmarks", None)
    normalized = pose_lists[0] if pose_lists and len(pose_lists[0]) >= 33 else None
    if normalized is None:
        landmarks: tuple[tuple[float, float, float], ...] = ()
    else:
        landmarks = tuple(
            (float(lm.x), float(lm.y), landmark_confidence(lm))
            for lm in normalized
        )

    world_points: dict[int, np.ndarray] = {}
    world_lists = getattr(result, "pose_world_landmarks", None)
    world = world_lists[0] if world_lists and len(world_lists[0]) >= 33 else None
    if world is not None:
        for index, landmark in enumerate(world):
            value = landmark_xyz(landmark)
            if np.all(np.isfinite(value)):
                world_points[index] = value

    person_mask: Optional[np.ndarray] = None
    mask_images = getattr(result, "segmentation_masks", None)
    if mask_images:
        try:
            mask = np.asarray(mask_images[0].numpy_view(), dtype=np.float32)
            mask = np.squeeze(mask)
            if mask.ndim == 2 and mask.size > 0 and np.all(np.isfinite(mask)):
                if mask.shape != gray.shape[:2]:
                    mask = cv2.resize(
                        mask,
                        (gray.shape[1], gray.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                person_mask = np.ascontiguousarray(np.clip(mask, 0.0, 1.0))
        except Exception:
            person_mask = None

    return PoseAnchorPacket(
        sequence=int(sequence),
        captured_at=float(captured_at),
        completed_at=float(completed_at),
        gray=np.ascontiguousarray(gray),
        landmarks=landmarks,
        world_points=world_points,
        person_mask=person_mask,
    )


PEDAL_TRACKED_POINT_INDICES: tuple[int, ...] = (
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    LEFT_HEEL,
    RIGHT_HEEL,
    LEFT_FOOT_INDEX,
    RIGHT_FOOT_INDEX,
)

TRACKED_POINT_INDICES: tuple[int, ...] = (
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    LEFT_HEEL,
    RIGHT_HEEL,
    LEFT_FOOT_INDEX,
    RIGHT_FOOT_INDEX,
)

FOOT_LANDMARK_PAIRS: tuple[tuple[int, int], ...] = (
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_KNEE, RIGHT_KNEE),
    (LEFT_ANKLE, RIGHT_ANKLE),
    (LEFT_HEEL, RIGHT_HEEL),
    (LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX),
)
LEFT_FOOT_TRACK_INDICES: tuple[int, ...] = tuple(pair[0] for pair in FOOT_LANDMARK_PAIRS)
RIGHT_FOOT_TRACK_INDICES: tuple[int, ...] = tuple(pair[1] for pair in FOOT_LANDMARK_PAIRS)

LEG_TRACK_INDICES: dict[str, tuple[int, ...]] = {
    "left": (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX),
    "right": (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX),
}

LEG_BONE_EDGES: dict[str, tuple[tuple[int, int], ...]] = {
    "left": (
        (LEFT_HIP, LEFT_KNEE),
        (LEFT_KNEE, LEFT_ANKLE),
        (LEFT_ANKLE, LEFT_HEEL),
        (LEFT_ANKLE, LEFT_FOOT_INDEX),
        (LEFT_HEEL, LEFT_FOOT_INDEX),
    ),
    "right": (
        (RIGHT_HIP, RIGHT_KNEE),
        (RIGHT_KNEE, RIGHT_ANKLE),
        (RIGHT_ANKLE, RIGHT_HEEL),
        (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
        (RIGHT_HEEL, RIGHT_FOOT_INDEX),
    ),
}


class PoseFeatureTracker:
    """Semantic, coherent-leg, and camera-rate micro-motion tracker.

    MediaPipe periodically identifies each hip-to-toe chain and supplies a soft
    person mask. Dense foreground features estimate a robust transform for each
    leg, while local landmark patches preserve tiny ankle and foot articulation.
    This keeps the pedal signal responsive without requiring pose inference to
    finish at the camera's full frame rate.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.pedals_only = str(config.get("control_mode", "pedals_only")) == "pedals_only"
        self.tracked_point_indices = (
            PEDAL_TRACKED_POINT_INDICES if self.pedals_only else TRACKED_POINT_INDICES
        )
        self.confidence_threshold = clip(float(config["landmark_confidence"]), 0.05, 0.95)
        self.soft_confidence_threshold = max(0.08, self.confidence_threshold * 0.58)
        self.ai_anchor_max_age = max(0.05, float(config["ai_anchor_max_age_seconds"]))
        self.foot_identity_lock = bool(config.get("foot_identity_lock", True))
        self.foot_identity_swap_margin_pixels = max(
            0.0, float(config.get("foot_identity_swap_margin_pixels", 12.0))
        )
        self.segmentation_enabled = bool(config.get("enable_segmentation_mask", True))
        self.segmentation_threshold = clip(
            float(config.get("segmentation_threshold", 0.16)), 0.01, 0.95
        )
        self.segmentation_dilate_pixels = max(
            0, int(config.get("segmentation_dilate_pixels", 13))
        )
        self.leg_lock_enabled = bool(config.get("leg_lock_enabled", True))
        self.leg_feature_count = max(8, int(config.get("leg_lock_feature_count", 34)))
        self.leg_feature_quality = clip(
            float(config.get("leg_lock_feature_quality", 0.006)), 0.0001, 0.25
        )
        self.leg_feature_min_distance = max(
            2.0, float(config.get("leg_lock_feature_min_distance_pixels", 5.0))
        )
        self.leg_line_thickness = max(
            8, int(config.get("leg_lock_line_thickness_pixels", 24))
        )
        self.leg_roi_padding = max(
            8, int(config.get("leg_lock_roi_padding_pixels", 22))
        )
        self.leg_min_inliers = max(3, int(config.get("leg_lock_min_inliers", 6)))
        self.leg_ransac_threshold = max(
            0.5, float(config.get("leg_lock_ransac_threshold_pixels", 2.4))
        )
        self.leg_max_scale_change = clip(
            float(config.get("leg_lock_max_scale_change", 0.12)), 0.01, 0.50
        )
        self.leg_max_rotation_radians = math.radians(
            clip(float(config.get("leg_lock_max_rotation_degrees", 14.0)), 1.0, 45.0)
        )
        self.leg_max_translation = max(
            4.0, float(config.get("leg_lock_max_translation_pixels", 30.0))
        )
        self.leg_landmark_blend = clip(
            float(config.get("leg_lock_landmark_blend", 0.24)), 0.0, 1.0
        )
        self.leg_outlier_pixels = max(
            3.0, float(config.get("leg_lock_outlier_pixels", 12.0))
        )
        self.leg_anchor_jump_pixels = max(
            6.0, float(config.get("leg_lock_anchor_jump_pixels", 26.0))
        )
        self.leg_bone_length_tolerance = clip(
            float(config.get("leg_lock_bone_length_tolerance", 0.34)), 0.10, 0.80
        )
        self.manual_triangle_max_edge_change = clip(
            float(config.get("manual_triangle_max_edge_change_ratio", 0.22)),
            0.03,
            0.80,
        )
        self.leg_reacquire_anchors = max(
            1, int(config.get("leg_lock_reacquire_anchors", 3))
        )
        self.world_feature_weight = clip(
            float(config.get("pedal_world_feature_weight", 0.0)), 0.0, 1.0
        )
        self.enable_flow = bool(config["enable_optical_flow"])
        self.flow_hold_seconds = max(0.0, float(config["optical_flow_hold_seconds"]))
        self.feature_hold_seconds = max(0.0, float(config["feature_hold_seconds"]))
        self.anchor_blend = clip(float(config["optical_flow_anchor_blend"]), 0.0, 1.0)
        self.anchor_deadband = max(0.0, float(config["optical_flow_anchor_deadband_pixels"]))
        self.max_fb_error = max(0.20, float(config["optical_flow_max_fb_error_pixels"]))
        self.max_anchor_distance = max(
            2.0, float(config["optical_flow_anchor_max_distance_pixels"])
        )
        self.patch_radius = max(1.0, float(config["optical_flow_patch_radius_pixels"]))
        patch_grid = max(1, int(config.get("optical_flow_patch_grid_size", 3)))
        self.patch_grid_size = patch_grid if patch_grid % 2 == 1 else patch_grid + 1
        self.min_patch_votes = min(
            self.patch_grid_size * self.patch_grid_size,
            max(1, int(config["optical_flow_min_patch_votes"])),
        )
        manual_patch_grid = max(3, int(config.get("manual_anchor_patch_grid_size", 5)))
        self.manual_patch_grid_size = (
            manual_patch_grid if manual_patch_grid % 2 == 1 else manual_patch_grid + 1
        )
        self.manual_min_patch_votes = min(
            self.manual_patch_grid_size * self.manual_patch_grid_size,
            max(3, int(config.get("manual_anchor_min_patch_votes", 9))),
        )
        window = max(9, int(config["optical_flow_window_pixels"]))
        self.flow_window = window if window % 2 == 1 else window + 1
        self.flow_max_level = max(0, int(config["optical_flow_max_level"]))
        self.flow_validation_interval = max(1, int(config["optical_flow_validation_interval"]))
        self.flow_use_roi = bool(config.get("optical_flow_use_roi", True))
        self.flow_roi_padding = max(
            16, int(config.get("optical_flow_roi_padding_pixels", 48))
        )
        self.flow_frame_counter = 0

        min_cutoff = float(config["feature_filter_min_cutoff_hz"])
        beta = float(config["feature_filter_beta"])
        derivative_cutoff = float(config["feature_filter_derivative_cutoff_hz"])
        self.left_filter = OneEuroFilter(min_cutoff, beta, derivative_cutoff)
        self.right_filter = OneEuroFilter(min_cutoff, beta, derivative_cutoff)
        self.steering_filter = CircularOneEuroFilter(min_cutoff, beta, derivative_cutoff)

        self.prev_gray: Optional[np.ndarray] = None
        self.points: dict[int, np.ndarray] = {}
        self.manual_anchor_points: dict[int, np.ndarray] = {}
        self.manual_anchor_indices: set[int] = set()
        self.manual_anchor_tracking_lost = False
        self.manual_reference_points: dict[int, np.ndarray] = {}
        self.manual_anchor_templates: dict[int, tuple[np.ndarray, ...]] = {}
        self.manual_match_quality: dict[int, float] = {}
        template_size = max(
            9,
            int(config.get("manual_anchor_template_size_pixels", 17)),
        )
        self.manual_template_size = template_size if template_size % 2 == 1 else template_size + 1
        self.manual_template_search = max(
            2,
            int(config.get("manual_anchor_template_search_pixels", 7)),
        )
        self.manual_template_min_score = clip(
            float(config.get("manual_anchor_template_min_score", 0.38)),
            0.10,
            0.95,
        )
        self.manual_template_rotation = max(
            0.0,
            float(config.get("manual_anchor_template_rotation_degrees", 14.0)),
        )
        self.point_quality: dict[int, float] = {}
        self.last_ai_anchor: dict[int, float] = {}
        self.last_ai_quality: dict[int, float] = {}
        self.world_points: dict[int, np.ndarray] = {}
        self.last_world_anchor: dict[int, float] = {}
        self.last_landmarks: tuple[tuple[float, float, float], ...] = ()
        self.last_anchor_sequence = -1
        self._anchor_foot_swapped = False
        self._anchor_point_support: dict[int, float] = {}
        self.anchor_mismatch_counts: dict[int, int] = {}
        self.person_mask: Optional[np.ndarray] = None
        self.person_mask_updated_at = 0.0
        self.last_leg_affine_quality: dict[str, float] = {"left": 0.0, "right": 0.0}
        self.last_left_feature: Optional[FootFeatureObservation] = None
        self.last_right_feature: Optional[FootFeatureObservation] = None
        self.last_steering: Optional[float] = None
        self.last_left_time = 0.0
        self.last_right_time = 0.0
        self.last_steering_time = 0.0

    def reset(self) -> None:
        preserved_manual = {
            index: np.asarray(self.points.get(index, point), dtype=np.float64).copy()
            for index, point in self.manual_anchor_points.items()
        }
        self.prev_gray = None
        self.points.clear()
        self.points.update(preserved_manual)
        self.manual_anchor_points = {
            index: point.copy() for index, point in preserved_manual.items()
        }
        self.point_quality.clear()
        self.last_ai_anchor.clear()
        self.last_ai_quality.clear()
        self.world_points.clear()
        self.last_world_anchor.clear()
        self.last_landmarks = ()
        self.last_anchor_sequence = -1
        self._anchor_foot_swapped = False
        self._anchor_point_support.clear()
        self.anchor_mismatch_counts.clear()
        self.manual_match_quality.clear()
        self.person_mask = None
        self.person_mask_updated_at = 0.0
        self.last_leg_affine_quality = {"left": 0.0, "right": 0.0}
        self.flow_frame_counter = 0
        self.left_filter.reset()
        self.right_filter.reset()
        self.steering_filter.reset()
        self.last_left_feature = None
        self.last_right_feature = None
        self.last_steering = None
        self.last_left_time = 0.0
        self.last_right_time = 0.0
        self.last_steering_time = 0.0

    @property
    def manual_anchors_complete(self) -> bool:
        return (
            self.manual_anchor_indices == set(MANUAL_FOOT_ANCHOR_INDICES)
            and not self.manual_anchor_tracking_lost
        )

    def set_manual_foot_anchors(
        self,
        points: dict[int, np.ndarray],
        *,
        reference_gray: Optional[np.ndarray] = None,
        reference_points: Optional[dict[int, np.ndarray]] = None,
    ) -> None:
        if set(points) != set(MANUAL_FOOT_ANCHOR_INDICES):
            raise ValueError("Manual foot anchors must contain heel, ankle, and toe for both feet")
        normalized: dict[int, np.ndarray] = {}
        for index, point in points.items():
            value = np.asarray(point, dtype=np.float64)
            if value.shape != (2,) or not np.all(np.isfinite(value)):
                raise ValueError("Manual foot anchor coordinates must be finite 2-D points")
            if not (-0.02 <= value[0] <= 1.02 and -0.02 <= value[1] <= 1.02):
                raise ValueError("Manual foot anchor coordinates must lie inside the camera frame")
            normalized[index] = value.copy()
        reference_source = reference_points if reference_points is not None else normalized
        reference_normalized: dict[int, np.ndarray] = {}
        for index in MANUAL_FOOT_ANCHOR_INDICES:
            value = np.asarray(reference_source[index], dtype=np.float64)
            if value.shape != (2,) or not np.all(np.isfinite(value)):
                raise ValueError("Manual reference coordinates must be finite 2-D points")
            reference_normalized[index] = value.copy()
        templates: dict[int, tuple[np.ndarray, ...]] = {}
        if reference_gray is not None:
            templates = self._build_manual_anchor_templates(
                reference_gray,
                reference_normalized,
            )
        self.manual_anchor_points = normalized
        self.manual_anchor_indices = set(normalized)
        self.manual_anchor_tracking_lost = False
        self.manual_reference_points = reference_normalized
        self.manual_anchor_templates = templates
        self.manual_match_quality = {}
        for index in MANUAL_FOOT_ANCHOR_INDICES:
            self.points.pop(index, None)
        self.points.update({index: point.copy() for index, point in normalized.items()})
        self.prev_gray = None
        self.left_filter.reset()
        self.right_filter.reset()
        self.last_left_feature = None
        self.last_right_feature = None

    def clear_manual_foot_anchors(self) -> None:
        for index in self.manual_anchor_indices:
            self.points.pop(index, None)
        self.manual_anchor_points = {}
        self.manual_anchor_indices = set()
        self.manual_anchor_tracking_lost = False
        self.manual_reference_points = {}
        self.manual_anchor_templates = {}
        self.manual_match_quality = {}
        self.prev_gray = None
        self.last_left_feature = None
        self.last_right_feature = None

    @staticmethod
    def _centered_gray_patch(
        gray: np.ndarray,
        centre_pixels: np.ndarray,
        size: int,
    ) -> np.ndarray:
        size = max(3, int(size))
        if size % 2 == 0:
            size += 1
        if gray.ndim != 2 or gray.size == 0:
            raise ValueError("anchor reference image is invalid")
        padding = size // 2 + 2
        padded = cv2.copyMakeBorder(
            np.ascontiguousarray(gray),
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_REFLECT_101,
        )
        centre = (
            float(centre_pixels[0]) + padding,
            float(centre_pixels[1]) + padding,
        )
        return np.ascontiguousarray(cv2.getRectSubPix(padded, (size, size), centre))

    def _build_manual_anchor_templates(
        self,
        reference_gray: np.ndarray,
        reference_points: dict[int, np.ndarray],
    ) -> dict[int, tuple[np.ndarray, ...]]:
        gray = np.ascontiguousarray(reference_gray)
        if gray.ndim != 2 or min(gray.shape[:2]) <= 1:
            raise ValueError("the frozen anchor image is invalid")
        height, width = gray.shape[:2]
        scale = np.array([width, height], dtype=np.float64)
        labels = dict(MANUAL_ANCHOR_SEQUENCE)
        output: dict[int, tuple[np.ndarray, ...]] = {}
        angles = [0.0]
        if self.manual_template_rotation >= 0.5:
            angles = [
                0.0,
                -self.manual_template_rotation,
                self.manual_template_rotation,
            ]
        for index in MANUAL_FOOT_ANCHOR_INDICES:
            point = np.asarray(reference_points[index], dtype=np.float64)
            base = self._centered_gray_patch(
                gray,
                point * scale,
                self.manual_template_size,
            )
            if float(np.std(base.astype(np.float32))) < 2.0:
                label = labels.get(index, "selected point").lower()
                raise ValueError(f"{label} has too little visible texture")
            centre = (self.manual_template_size - 1) / 2.0
            variants: list[np.ndarray] = []
            for angle in angles:
                if abs(angle) < 1e-6:
                    variant = base.copy()
                else:
                    matrix = cv2.getRotationMatrix2D((centre, centre), angle, 1.0)
                    variant = cv2.warpAffine(
                        base,
                        matrix,
                        (self.manual_template_size, self.manual_template_size),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101,
                    )
                variants.append(np.ascontiguousarray(variant))
            output[index] = tuple(variants)
        return output

    def _validate_manual_template_flow(
        self,
        gray: np.ndarray,
        candidates: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        """Reattach LK predictions to the original clicked texture every frame."""

        self.manual_match_quality = {}
        if not candidates or not self.manual_anchor_templates:
            self.manual_match_quality = {index: 1.0 for index in candidates}
            return candidates
        height, width = gray.shape[:2]
        scale = np.array([width, height], dtype=np.float64)
        output: dict[int, np.ndarray] = {}
        search = self.manual_template_search
        search_size = self.manual_template_size + 2 * search
        for index, candidate in candidates.items():
            templates = self.manual_anchor_templates.get(index)
            if not templates:
                output[index] = np.asarray(candidate, dtype=np.float64).copy()
                self.manual_match_quality[index] = 1.0
                continue
            candidate_pixels = np.asarray(candidate, dtype=np.float64) * scale
            try:
                search_patch = self._centered_gray_patch(
                    gray,
                    candidate_pixels,
                    search_size,
                )
            except (ValueError, cv2.error):
                continue
            best_score = -1.0
            best_location = (search, search)
            for variant_index, template in enumerate(templates):
                try:
                    response = cv2.matchTemplate(
                        search_patch,
                        template,
                        cv2.TM_CCOEFF_NORMED,
                    )
                except cv2.error:
                    continue
                _minimum, maximum, _min_location, max_location = cv2.minMaxLoc(response)
                if math.isfinite(maximum) and maximum > best_score:
                    best_score = float(maximum)
                    best_location = max_location
                # The unrotated canonical template is checked first and is the
                # common steady-foot case. Only pay for rotated variants when
                # tilt has lowered its match below the acceptance threshold.
                if (
                    variant_index == 0
                    and best_score >= min(0.95, self.manual_template_min_score + 0.08)
                ):
                    break
            if best_score < self.manual_template_min_score:
                continue
            corrected_pixels = candidate_pixels + np.array(
                [best_location[0] - search, best_location[1] - search],
                dtype=np.float64,
            )
            corrected = corrected_pixels / scale
            if not (-0.08 <= corrected[0] <= 1.08 and -0.08 <= corrected[1] <= 1.08):
                continue
            output[index] = corrected
            self.manual_match_quality[index] = clip(best_score, 0.0, 1.0)
        return output

    def align_manual_anchor_selection(
        self,
        reference_gray: np.ndarray,
        current_gray: np.ndarray,
        points: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        """Carry all six clicks from the frozen setup frame to the live frame."""

        if set(points) != set(MANUAL_FOOT_ANCHOR_INDICES):
            return {}
        return self._track_point_clouds(
            np.ascontiguousarray(reference_gray),
            np.ascontiguousarray(current_gray),
            points,
            validate_backward=True,
            long_range=True,
            patch_grid_size=self.manual_patch_grid_size,
            minimum_votes=self.manual_min_patch_votes,
        )

    def update(
        self,
        frame: np.ndarray,
        now: float,
        dt: float,
        anchor: Optional[PoseAnchorPacket] = None,
        sequence: int = -1,
    ) -> PoseFeatures:
        features = PoseFeatures()
        features.hard_locked_indices = frozenset(self.manual_anchor_indices)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow_points = self._calculate_optical_flow(gray)

        ai_points: dict[int, np.ndarray] = {}
        ai_confidences: dict[int, float] = {}
        anchor_is_new = anchor is not None and anchor.sequence > self.last_anchor_sequence
        if anchor_is_new and anchor is not None:
            self.last_anchor_sequence = anchor.sequence
            if anchor.pose_detected:
                self.last_landmarks = anchor.landmarks
                age = max(0.0, float(now) - anchor.captured_at)
                if age <= self.ai_anchor_max_age:
                    ai_points = self._align_anchor_points(anchor, gray, sequence)
                    for index in ai_points:
                        source_index = self._paired_source_index(
                            index, self._anchor_foot_swapped
                        )
                        if source_index < len(anchor.landmarks):
                            ai_confidences[index] = float(
                                anchor.landmarks[source_index][2]
                            )
                    self._update_world_points_from_anchor(
                        anchor,
                        float(now),
                        swap_feet=self._anchor_foot_swapped,
                    )

        features.landmarks = list(self.last_landmarks)
        merged, sources, point_quality = self._merge_points(
            ai_points=ai_points,
            ai_confidences=ai_confidences,
            flow_points=flow_points,
            frame_shape=gray.shape,
            now=float(now),
        )
        self.points = merged
        self.point_quality = point_quality
        if (
            self.prev_gray is not None
            and self.manual_anchor_indices
            and not self.manual_anchor_indices.issubset(merged)
        ):
            self.manual_anchor_tracking_lost = True
        features.tracked_landmarks = {
            index: (float(point[0]), float(point[1]), float(sources[index]))
            for index, point in merged.items()
        }
        features.pose_detected = bool(self.last_landmarks) or bool(merged)

        current_world = self._current_world_points(float(now))
        frame_aspect_ratio = float(gray.shape[1]) / max(1.0, float(gray.shape[0]))
        left_raw = build_foot_feature(
            merged,
            current_world,
            side="left",
            qualities=point_quality,
            frame_aspect_ratio=frame_aspect_ratio,
        )
        right_raw = build_foot_feature(
            merged,
            current_world,
            side="right",
            qualities=point_quality,
            frame_aspect_ratio=frame_aspect_ratio,
        )
        if left_raw is not None:
            left_raw.validity[FOOT_WORLD_FEATURE_START:] *= self.world_feature_weight
        if right_raw is not None:
            right_raw.validity[FOOT_WORLD_FEATURE_START:] *= self.world_feature_weight
        (
            features.left_foot,
            features.left_foot_ok,
            features.left_foot_fresh,
        ) = self._stabilize_foot("left", left_raw, float(now), float(dt))
        (
            features.right_foot,
            features.right_foot_ok,
            features.right_foot_fresh,
        ) = self._stabilize_foot("right", right_raw, float(now), float(dt))

        steering_raw: Optional[float] = None
        if not self.pedals_only:
            left_wrist = merged.get(LEFT_WRIST)
            right_wrist = merged.get(RIGHT_WRIST)
            if left_wrist is not None and right_wrist is not None:
                delta = right_wrist - left_wrist
                if float(np.linalg.norm(delta)) >= 0.025:
                    steering_raw = math.atan2(float(delta[1]), float(delta[0]))
        if steering_raw is not None:
            steering = self.steering_filter.update(steering_raw, dt)
            self.last_steering = steering
            self.last_steering_time = float(now)
            features.steering_angle = steering
            features.steering_ok = True
            features.steering_fresh = True
        elif (
            self.last_steering is not None
            and float(now) - self.last_steering_time <= self.feature_hold_seconds
        ):
            features.steering_angle = self.last_steering
            features.steering_ok = True
            features.steering_fresh = False
        else:
            self.steering_filter.reset()
            self.last_steering = None
            features.steering_angle = None
            features.steering_ok = False
            features.steering_fresh = False

        self.prev_gray = gray
        return features

    def _patch_offsets(self, grid_size: Optional[int] = None) -> np.ndarray:
        radius = self.patch_radius
        size = self.patch_grid_size if grid_size is None else max(1, int(grid_size))
        if size <= 1:
            return np.zeros((1, 2), dtype=np.float32)
        coordinates = np.linspace(
            -radius, radius, size, dtype=np.float32
        )
        return np.array(
            [(x, y) for y in coordinates for x in coordinates], dtype=np.float32
        )

    def _track_point_clouds(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        centres: dict[int, np.ndarray],
        *,
        validate_backward: bool = False,
        long_range: bool = False,
        patch_grid_size: Optional[int] = None,
        minimum_votes: Optional[int] = None,
    ) -> dict[int, np.ndarray]:
        """Track a small texture grid around every semantic landmark.

        The OpenCV call and all vote aggregation are batched. This avoids a
        Python loop and median allocation for every landmark, which is important
        when the camera genuinely delivers 90 or 120 frames per second.
        """
        if not centres:
            return {}
        previous_height, previous_width = previous_gray.shape[:2]
        current_height, current_width = current_gray.shape[:2]
        if min(previous_height, previous_width, current_height, current_width) <= 1:
            return {}

        keys = list(centres)
        centre_normalized = np.stack(
            [np.asarray(centres[index], dtype=np.float64) for index in keys],
            axis=0,
        )
        if not np.all(np.isfinite(centre_normalized)):
            return {}
        centre_pixels = centre_normalized * np.array(
            [previous_width, previous_height], dtype=np.float64
        )
        offsets = self._patch_offsets(patch_grid_size).astype(np.float64, copy=False)
        required_votes = self.min_patch_votes if minimum_votes is None else max(
            1, int(minimum_votes)
        )
        required_votes = min(required_votes, len(offsets))
        window = max(self.flow_window, 19) if long_range else self.flow_window
        max_level = max(self.flow_max_level, 2) if long_range else self.flow_max_level

        roi_x0 = 0
        roi_y0 = 0
        roi_x1 = previous_width
        roi_y1 = previous_height
        can_crop = (
            self.flow_use_roi
            and previous_width == current_width
            and previous_height == current_height
        )
        if can_crop:
            pyramid_margin = int(
                math.ceil(window * (2 ** max_level) * 0.65 + self.patch_radius)
            )
            padding = max(self.flow_roi_padding, pyramid_margin)
            roi_x0 = max(0, int(math.floor(float(centre_pixels[:, 0].min()))) - padding)
            roi_y0 = max(0, int(math.floor(float(centre_pixels[:, 1].min()))) - padding)
            roi_x1 = min(
                previous_width,
                int(math.ceil(float(centre_pixels[:, 0].max()))) + padding + 1,
            )
            roi_y1 = min(
                previous_height,
                int(math.ceil(float(centre_pixels[:, 1].max()))) + padding + 1,
            )
            minimum_side = max(64, window * (2 ** max_level) + 16)
            if roi_x1 - roi_x0 < minimum_side:
                needed = minimum_side - (roi_x1 - roi_x0)
                roi_x0 = max(0, roi_x0 - needed // 2)
                roi_x1 = min(previous_width, roi_x1 + needed - needed // 2)
            if roi_y1 - roi_y0 < minimum_side:
                needed = minimum_side - (roi_y1 - roi_y0)
                roi_y0 = max(0, roi_y0 - needed // 2)
                roi_y1 = min(previous_height, roi_y1 + needed - needed // 2)

        previous_roi = previous_gray[roi_y0:roi_y1, roi_x0:roi_x1]
        current_roi = current_gray[roi_y0:roi_y1, roi_x0:roi_x1]
        if previous_roi.size == 0 or current_roi.size == 0:
            return {}
        previous_roi = np.ascontiguousarray(previous_roi)
        current_roi = np.ascontiguousarray(current_roi)
        roi_origin = np.array([roi_x0, roi_y0], dtype=np.float64)
        roi_height, roi_width = previous_roi.shape[:2]

        starts_full = centre_pixels[:, None, :] + offsets[None, :, :]
        starts_full[..., 0] = np.clip(
            starts_full[..., 0], float(roi_x0), float(max(roi_x0, roi_x1 - 1))
        )
        starts_full[..., 1] = np.clip(
            starts_full[..., 1], float(roi_y0), float(max(roi_y0, roi_y1 - 1))
        )
        starts_local = starts_full - roi_origin[None, None, :]
        starts_local[..., 0] = np.clip(starts_local[..., 0], 0.0, roi_width - 1.0)
        starts_local[..., 1] = np.clip(starts_local[..., 1], 0.0, roi_height - 1.0)
        previous_pixels = np.ascontiguousarray(
            starts_local.astype(np.float32).reshape(-1, 1, 2)
        )

        try:
            current_pixels, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
                previous_roi,
                current_roi,
                previous_pixels,
                None,
                winSize=(window, window),
                maxLevel=max_level,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    12 if not long_range else 16,
                    0.01,
                ),
                minEigThreshold=1e-5,
            )
            if current_pixels is None or forward_status is None:
                return {}
            backward_pixels: Optional[np.ndarray] = None
            backward_status: Optional[np.ndarray] = None
            if validate_backward:
                backward_pixels, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                    current_roi,
                    previous_roi,
                    current_pixels,
                    None,
                    winSize=(window, window),
                    maxLevel=max_level,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        12,
                        0.01,
                    ),
                    minEigThreshold=1e-5,
                )
                if backward_pixels is None or backward_status is None:
                    return {}
        except cv2.error:
            return {}

        point_count = len(keys)
        cloud_size = len(offsets)
        starts = previous_pixels.reshape(point_count, cloud_size, 2).astype(np.float64)
        ends = current_pixels.reshape(point_count, cloud_size, 2).astype(np.float64)
        displacements = ends - starts
        valid = forward_status.reshape(point_count, cloud_size).astype(bool)
        if forward_error is not None:
            errors = forward_error.reshape(point_count, cloud_size)
            valid &= np.isfinite(errors) & (errors <= 55.0)
        if validate_backward:
            assert backward_pixels is not None and backward_status is not None
            valid &= backward_status.reshape(point_count, cloud_size).astype(bool)
            returns = backward_pixels.reshape(point_count, cloud_size, 2).astype(np.float64)
            fb_error = np.linalg.norm(returns - starts, axis=2)
            valid &= np.isfinite(fb_error) & (fb_error <= self.max_fb_error)

        motion_limit = 80.0 if long_range else 18.0
        magnitudes = np.linalg.norm(displacements, axis=2)
        valid &= np.all(np.isfinite(displacements), axis=2)
        valid &= np.isfinite(magnitudes) & (magnitudes <= motion_limit)

        weights = valid.astype(np.float64)
        counts = weights.sum(axis=1)
        safe_counts = np.maximum(counts, 1.0)
        motion = (displacements * weights[..., None]).sum(axis=1) / safe_counts[:, None]

        # One consensus-refinement pass rejects a bedding edge or low-texture
        # sample that moves differently from the rest of the foot patch.
        residuals = np.linalg.norm(displacements - motion[:, None, :], axis=2)
        residual_limit = 6.0 if long_range else 2.5
        refined_valid = valid & (residuals <= residual_limit)
        refined_weights = refined_valid.astype(np.float64)
        refined_counts = refined_weights.sum(axis=1)
        refined_safe = np.maximum(refined_counts, 1.0)
        refined_motion = (
            displacements * refined_weights[..., None]
        ).sum(axis=1) / refined_safe[:, None]
        use_refined = refined_counts >= required_votes
        motion[use_refined] = refined_motion[use_refined]
        counts[use_refined] = refined_counts[use_refined]

        new_pixels = centre_pixels + motion
        normalized = new_pixels / np.array(
            [max(1, current_width), max(1, current_height)], dtype=np.float64
        )
        output: dict[int, np.ndarray] = {}
        for position, index in enumerate(keys):
            if counts[position] < required_votes:
                continue
            x, y = normalized[position]
            if -0.08 <= x <= 1.08 and -0.08 <= y <= 1.08:
                output[index] = np.array([x, y], dtype=np.float64)
        return output

    def _calculate_optical_flow(self, gray: np.ndarray) -> dict[int, np.ndarray]:
        if not self.enable_flow or self.prev_gray is None or not self.points:
            return {}
        self.flow_frame_counter += 1
        validate = self.flow_frame_counter % self.flow_validation_interval == 0
        automatic_centres = {
            index: point
            for index, point in self.points.items()
            if index not in self.manual_anchor_indices
        }
        individual = self._track_point_clouds(
            self.prev_gray,
            gray,
            automatic_centres,
            validate_backward=validate,
            long_range=False,
        )
        manual_centres = {
            index: self.points[index]
            for index in self.manual_anchor_indices
            if index in self.points
        }
        manual_flow = self._track_point_clouds(
            self.prev_gray,
            gray,
            manual_centres,
            validate_backward=True,
            long_range=False,
            patch_grid_size=self.manual_patch_grid_size,
            minimum_votes=self.manual_min_patch_votes,
        )
        manual_flow = self._validate_manual_template_flow(gray, manual_flow)
        manual_flow = self._validate_manual_triangle_flow(manual_flow, gray.shape)
        individual.update(manual_flow)
        if not self.leg_lock_enabled:
            return individual

        coherent: dict[int, np.ndarray] = {}
        qualities: dict[str, float] = {"left": 0.0, "right": 0.0}
        for side in ("left", "right"):
            predicted, quality = self._track_leg_affine(
                self.prev_gray,
                gray,
                side,
            )
            coherent.update(predicted)
            qualities[side] = quality
        self.last_leg_affine_quality = qualities
        combined = self._combine_leg_flow(
            individual=individual,
            coherent=coherent,
            frame_shape=gray.shape,
        )
        # Confirmed manual points are owned solely by strict local patch flow.
        # Coherent leg motion and MediaPipe may never flatten or replace them.
        for index in self.manual_anchor_indices:
            if index in manual_flow:
                combined[index] = manual_flow[index].copy()
            else:
                combined.pop(index, None)
        return combined

    def _validate_manual_triangle_flow(
        self,
        candidates: dict[int, np.ndarray],
        frame_shape: tuple[int, ...],
    ) -> dict[int, np.ndarray]:
        if not self.manual_anchor_indices:
            return candidates
        height, width = frame_shape[:2]
        scale = np.array([width, height], dtype=np.float64)
        output = candidates.copy()
        for indices in MANUAL_FOOT_TRIANGLES.values():
            if not all(index in self.manual_anchor_indices for index in indices):
                continue
            # Losing one vertex makes the complete pedal triangle unavailable;
            # never infer or freeze the missing point.
            if not all(index in candidates and index in self.points for index in indices):
                for index in indices:
                    output.pop(index, None)
                continue
            previous = [self.points[index] * scale for index in indices]
            reference_source = (
                self.manual_reference_points
                if all(index in self.manual_reference_points for index in indices)
                else self.points
            )
            reference = [reference_source[index] * scale for index in indices]
            current = [candidates[index] * scale for index in indices]
            edge_pairs = ((0, 1), (1, 2), (2, 0))
            coherent = True
            for start, end in edge_pairs:
                old_length = float(np.linalg.norm(previous[end] - previous[start]))
                new_length = float(np.linalg.norm(current[end] - current[start]))
                if old_length < 2.0 or new_length < 2.0:
                    coherent = False
                    break
                if abs(new_length / old_length - 1.0) > self.manual_triangle_max_edge_change:
                    coherent = False
                    break
                reference_length = float(
                    np.linalg.norm(reference[end] - reference[start])
                )
                if (
                    reference_length < 2.0
                    or abs(new_length / reference_length - 1.0)
                    > self.manual_triangle_max_edge_change
                ):
                    coherent = False
                    break
            previous_cross = float(
                (previous[1][0] - previous[0][0])
                * (previous[2][1] - previous[0][1])
                - (previous[1][1] - previous[0][1])
                * (previous[2][0] - previous[0][0])
            )
            current_cross = float(
                (current[1][0] - current[0][0])
                * (current[2][1] - current[0][1])
                - (current[1][1] - current[0][1])
                * (current[2][0] - current[0][0])
            )
            reference_cross = float(
                (reference[1][0] - reference[0][0])
                * (reference[2][1] - reference[0][1])
                - (reference[1][1] - reference[0][1])
                * (reference[2][0] - reference[0][0])
            )
            # Reflection preserves all three edge lengths, so the signed area
            # is an independent guard against a toe/ankle swap or a patch that
            # jumps across the heel-pivot axis.
            if (
                abs(previous_cross) < 1.0
                or abs(current_cross) < 1.0
                or abs(reference_cross) < 1.0
                or previous_cross * current_cross <= 0.0
                or reference_cross * current_cross <= 0.0
            ):
                coherent = False
            if not coherent:
                for index in indices:
                    output.pop(index, None)
        return output

    def _leg_geometry_mask(
        self,
        frame_shape: tuple[int, int],
        side: str,
    ) -> tuple[Optional[np.ndarray], Optional[tuple[int, int, int, int]]]:
        """Build a tight foreground search area around one hip-to-toe chain."""
        height, width = frame_shape[:2]
        indices = LEG_TRACK_INDICES[side]
        available = [self.points[index] for index in indices if index in self.points]
        if len(available) < 3:
            return None, None
        pixels = {
            index: np.asarray(self.points[index], dtype=np.float64)
            * np.array([width, height], dtype=np.float64)
            for index in indices
            if index in self.points
        }
        mask = np.zeros((height, width), dtype=np.uint8)
        thickness = self.leg_line_thickness
        for start, end in LEG_BONE_EDGES[side]:
            if start not in pixels or end not in pixels:
                continue
            p0 = tuple(np.round(pixels[start]).astype(int))
            p1 = tuple(np.round(pixels[end]).astype(int))
            cv2.line(mask, p0, p1, 255, thickness, cv2.LINE_AA)
        joint_radius = max(6, thickness // 2)
        for point in pixels.values():
            cv2.circle(
                mask,
                tuple(np.round(point).astype(int)),
                joint_radius,
                255,
                -1,
                cv2.LINE_AA,
            )

        # The segmentation mask is a soft foreground gate.  If its overlap is
        # weak (common at toes or under imperfect lighting), keep the geometric
        # leg mask rather than dropping the leg.
        if (
            self.segmentation_enabled
            and self.person_mask is not None
            and self.person_mask.shape[:2] == (height, width)
            and time.monotonic() - self.person_mask_updated_at <= 0.75
        ):
            person = (self.person_mask >= self.segmentation_threshold).astype(np.uint8) * 255
            if self.segmentation_dilate_pixels > 0:
                size = self.segmentation_dilate_pixels
                if size % 2 == 0:
                    size += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
                person = cv2.dilate(person, kernel, iterations=1)
            gated = cv2.bitwise_and(mask, person)
            if int(cv2.countNonZero(gated)) >= 80:
                mask = gated

        all_pixels = np.stack(list(pixels.values()), axis=0)
        x0 = max(0, int(math.floor(float(all_pixels[:, 0].min()))) - self.leg_roi_padding)
        y0 = max(0, int(math.floor(float(all_pixels[:, 1].min()))) - self.leg_roi_padding)
        x1 = min(width, int(math.ceil(float(all_pixels[:, 0].max()))) + self.leg_roi_padding + 1)
        y1 = min(height, int(math.ceil(float(all_pixels[:, 1].max()))) + self.leg_roi_padding + 1)
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None, None
        return mask, (x0, y0, x1, y1)

    def _track_leg_affine(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        side: str,
    ) -> tuple[dict[int, np.ndarray], float]:
        """Estimate one robust camera-rate transform from many pixels on a leg."""
        if previous_gray.shape != current_gray.shape:
            return {}, 0.0
        mask, bounds = self._leg_geometry_mask(previous_gray.shape, side)
        if mask is None or bounds is None:
            return {}, 0.0
        x0, y0, x1, y1 = bounds
        previous_roi = np.ascontiguousarray(previous_gray[y0:y1, x0:x1])
        current_roi = np.ascontiguousarray(current_gray[y0:y1, x0:x1])
        search_mask = np.ascontiguousarray(mask[y0:y1, x0:x1])
        if previous_roi.size == 0 or int(cv2.countNonZero(search_mask)) < 40:
            return {}, 0.0
        try:
            corners = cv2.goodFeaturesToTrack(
                previous_roi,
                maxCorners=self.leg_feature_count,
                qualityLevel=self.leg_feature_quality,
                minDistance=self.leg_feature_min_distance,
                mask=search_mask,
                blockSize=5,
                useHarrisDetector=False,
            )
        except cv2.error:
            return {}, 0.0
        if corners is None or len(corners) < self.leg_min_inliers:
            return {}, 0.0

        window = max(15, self.flow_window)
        max_level = max(1, self.flow_max_level)
        try:
            current, status, error = cv2.calcOpticalFlowPyrLK(
                previous_roi,
                current_roi,
                corners,
                None,
                winSize=(window, window),
                maxLevel=max_level,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    15,
                    0.01,
                ),
                minEigThreshold=1e-5,
            )
            if current is None or status is None:
                return {}, 0.0
            returned, back_status, _ = cv2.calcOpticalFlowPyrLK(
                current_roi,
                previous_roi,
                current,
                None,
                winSize=(window, window),
                maxLevel=max_level,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    12,
                    0.01,
                ),
                minEigThreshold=1e-5,
            )
            if returned is None or back_status is None:
                return {}, 0.0
        except cv2.error:
            return {}, 0.0

        starts = corners.reshape(-1, 2).astype(np.float64)
        ends = current.reshape(-1, 2).astype(np.float64)
        returns = returned.reshape(-1, 2).astype(np.float64)
        valid = status.reshape(-1).astype(bool) & back_status.reshape(-1).astype(bool)
        valid &= np.all(np.isfinite(ends), axis=1)
        valid &= np.linalg.norm(returns - starts, axis=1) <= self.max_fb_error * 1.25
        if error is not None:
            forward_error = error.reshape(-1)
            valid &= np.isfinite(forward_error) & (forward_error <= 60.0)
        displacement = np.linalg.norm(ends - starts, axis=1)
        valid &= np.isfinite(displacement) & (displacement <= self.leg_max_translation * 1.4)
        if int(valid.sum()) < self.leg_min_inliers:
            return {}, 0.0

        starts_valid = starts[valid]
        ends_valid = ends[valid]
        try:
            matrix, inliers = cv2.estimateAffinePartial2D(
                starts_valid,
                ends_valid,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.leg_ransac_threshold,
                maxIters=120,
                confidence=0.98,
                refineIters=5,
            )
        except cv2.error:
            return {}, 0.0
        if matrix is None or inliers is None or not np.all(np.isfinite(matrix)):
            return {}, 0.0
        inlier_count = int(inliers.reshape(-1).astype(bool).sum())
        if inlier_count < self.leg_min_inliers:
            return {}, 0.0

        a = float(matrix[0, 0])
        b = float(matrix[1, 0])
        scale = math.hypot(a, b)
        rotation = abs(math.atan2(b, a))
        translation = math.hypot(float(matrix[0, 2]), float(matrix[1, 2]))
        if (
            abs(scale - 1.0) > self.leg_max_scale_change
            or rotation > self.leg_max_rotation_radians
            or translation > self.leg_max_translation
        ):
            return {}, 0.0

        height, width = previous_gray.shape[:2]
        roi_origin = np.array([x0, y0], dtype=np.float64)
        output: dict[int, np.ndarray] = {}
        for index in LEG_TRACK_INDICES[side]:
            point = self.points.get(index)
            if point is None:
                continue
            local = point * np.array([width, height], dtype=np.float64) - roi_origin
            transformed = matrix[:, :2] @ local + matrix[:, 2] + roi_origin
            normalized = transformed / np.array([width, height], dtype=np.float64)
            if np.all(np.isfinite(normalized)) and -0.08 <= normalized[0] <= 1.08 and -0.08 <= normalized[1] <= 1.08:
                output[index] = normalized.astype(np.float64)

        valid_count = max(1, int(valid.sum()))
        quality = (inlier_count / valid_count) * min(
            1.0,
            inlier_count / max(1.0, float(self.leg_min_inliers * 2)),
        )
        return output, clip(float(quality), 0.0, 1.0)

    def _combine_leg_flow(
        self,
        individual: dict[int, np.ndarray],
        coherent: dict[int, np.ndarray],
        frame_shape: tuple[int, int],
    ) -> dict[int, np.ndarray]:
        """Keep local foot motion, but reject landmarks leaving their leg."""
        height, width = frame_shape[:2]
        scale = np.array([width, height], dtype=np.float64)
        combined: dict[int, np.ndarray] = {}
        all_indices = set(individual) | set(coherent)
        for index in all_indices:
            local = individual.get(index)
            rigid = coherent.get(index)
            if local is not None and rigid is not None:
                distance = float(np.linalg.norm((local - rigid) * scale))
                if distance <= self.leg_outlier_pixels:
                    # Most of the signal remains the landmark's own patch, so
                    # ankle/toe micro-motion is not flattened by the leg model.
                    combined[index] = (
                        (1.0 - self.leg_landmark_blend) * local
                        + self.leg_landmark_blend * rigid
                    )
                else:
                    combined[index] = rigid.copy()
            elif local is not None:
                combined[index] = local.copy()
            elif rigid is not None:
                combined[index] = rigid.copy()

        for side in ("left", "right"):
            indices = LEG_TRACK_INDICES[side]
            matched = [
                index for index in indices if index in combined and index in self.points
            ]
            if not matched:
                continue
            displacements = np.stack(
                [(combined[index] - self.points[index]) * scale for index in matched],
                axis=0,
            )
            median_motion = np.median(displacements, axis=0)
            predicted = {
                index: self.points[index] + median_motion / scale
                for index in indices
                if index in self.points
            }
            # Bone lengths are stable even while the leg rotates.  Replace only
            # the distal endpoint of an implausible segment, never the complete
            # chain, so a single bedding edge cannot pull the whole leg away.
            for _ in range(2):
                for start, end in LEG_BONE_EDGES[side]:
                    if (
                        start not in self.points
                        or end not in self.points
                        or start not in combined
                        or end not in combined
                    ):
                        continue
                    previous_length = float(
                        np.linalg.norm((self.points[end] - self.points[start]) * scale)
                    )
                    current_length = float(
                        np.linalg.norm((combined[end] - combined[start]) * scale)
                    )
                    if previous_length < 3.0:
                        continue
                    ratio_error = abs(current_length / previous_length - 1.0)
                    if ratio_error <= self.leg_bone_length_tolerance:
                        continue
                    fallback = coherent.get(end, predicted.get(end))
                    if fallback is not None:
                        combined[end] = np.asarray(fallback, dtype=np.float64).copy()
        return combined

    @staticmethod
    def _paired_source_index(index: int, swap_feet: bool) -> int:
        if not swap_feet:
            return index
        for left_index, right_index in FOOT_LANDMARK_PAIRS:
            if index == left_index:
                return right_index
            if index == right_index:
                return left_index
        return index

    @staticmethod
    def _available_centroid(
        points: dict[int, np.ndarray],
        indices: Sequence[int],
    ) -> Optional[np.ndarray]:
        values = [points[index] for index in indices if index in points]
        if len(values) < 2:
            return None
        return np.median(np.stack(values, axis=0), axis=0)

    def _lock_foot_identity(
        self,
        candidates: dict[int, np.ndarray],
        frame_shape: tuple[int, int],
    ) -> tuple[dict[int, np.ndarray], bool]:
        if not self.foot_identity_lock or not self.points:
            return candidates, False
        height, width = frame_shape[:2]
        scale = np.array([width, height], dtype=np.float64)
        landmark_weights = {
            LEFT_HIP: 1.9,
            RIGHT_HIP: 1.9,
            LEFT_KNEE: 1.7,
            RIGHT_KNEE: 1.7,
            LEFT_ANKLE: 1.45,
            RIGHT_ANKLE: 1.45,
            LEFT_HEEL: 1.0,
            RIGHT_HEEL: 1.0,
            LEFT_FOOT_INDEX: 1.1,
            RIGHT_FOOT_INDEX: 1.1,
        }

        def assignment_cost(swapped: bool) -> tuple[float, int]:
            cost = 0.0
            total_weight = 0.0
            matches = 0
            for left_index, right_index in FOOT_LANDMARK_PAIRS:
                candidate_for_left = right_index if swapped else left_index
                candidate_for_right = left_index if swapped else right_index
                for previous_index, candidate_index in (
                    (left_index, candidate_for_left),
                    (right_index, candidate_for_right),
                ):
                    previous = self.points.get(previous_index)
                    candidate = candidates.get(candidate_index)
                    if previous is None or candidate is None:
                        continue
                    weight = landmark_weights.get(previous_index, 1.0)
                    cost += weight * float(np.linalg.norm((candidate - previous) * scale))
                    total_weight += weight
                    matches += 1
            if total_weight <= 0.0:
                return float("inf"), matches
            cost /= total_weight

            # Preserve the direction between the two leg centroids.  This
            # makes identity robust even when feet cross in image space.
            previous_left = self._available_centroid(self.points, LEFT_FOOT_TRACK_INDICES)
            previous_right = self._available_centroid(self.points, RIGHT_FOOT_TRACK_INDICES)
            candidate_left_indices = (
                RIGHT_FOOT_TRACK_INDICES if swapped else LEFT_FOOT_TRACK_INDICES
            )
            candidate_right_indices = (
                LEFT_FOOT_TRACK_INDICES if swapped else RIGHT_FOOT_TRACK_INDICES
            )
            candidate_left = self._available_centroid(candidates, candidate_left_indices)
            candidate_right = self._available_centroid(candidates, candidate_right_indices)
            if (
                previous_left is not None
                and previous_right is not None
                and candidate_left is not None
                and candidate_right is not None
            ):
                old_axis = (previous_right - previous_left) * scale
                new_axis = (candidate_right - candidate_left) * scale
                old_norm = float(np.linalg.norm(old_axis))
                new_norm = float(np.linalg.norm(new_axis))
                if old_norm > 2.0 and new_norm > 2.0:
                    cosine = float(np.dot(old_axis, new_axis) / (old_norm * new_norm))
                    cost += max(0.0, -cosine) * 28.0
            return cost, matches

        direct_cost, direct_matches = assignment_cost(False)
        swapped_cost, swapped_matches = assignment_cost(True)
        if min(direct_matches, swapped_matches) < 4:
            return candidates, False
        if swapped_cost + self.foot_identity_swap_margin_pixels >= direct_cost:
            return candidates, False

        corrected = candidates.copy()
        for left_index, right_index in FOOT_LANDMARK_PAIRS:
            left_value = candidates.get(left_index)
            right_value = candidates.get(right_index)
            if right_value is None:
                corrected.pop(left_index, None)
            else:
                corrected[left_index] = right_value.copy()
            if left_value is None:
                corrected.pop(right_index, None)
            else:
                corrected[right_index] = left_value.copy()
        return corrected, True

    @staticmethod
    def _mask_support_at(mask: Optional[np.ndarray], point: np.ndarray) -> float:
        if mask is None or mask.ndim != 2 or mask.size == 0:
            return 1.0
        height, width = mask.shape[:2]
        x = int(round(float(point[0]) * width))
        y = int(round(float(point[1]) * height))
        radius = 4
        x0 = max(0, x - radius)
        y0 = max(0, y - radius)
        x1 = min(width, x + radius + 1)
        y1 = min(height, y + radius + 1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        patch = mask[y0:y1, x0:x1]
        if patch.size == 0:
            return 0.0
        # A high percentile is more tolerant than a single pixel at the foot's
        # silhouette boundary while still rejecting bedding far from the body.
        return clip(float(np.percentile(patch, 80.0)), 0.0, 1.0)

    def _refresh_person_mask(
        self,
        anchor: PoseAnchorPacket,
        raw: dict[int, np.ndarray],
        aligned: dict[int, np.ndarray],
        current_shape: tuple[int, int],
    ) -> None:
        if not self.segmentation_enabled or anchor.person_mask is None:
            return
        height, width = current_shape[:2]
        mask = np.asarray(anchor.person_mask, dtype=np.float32)
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        shifts = [
            (aligned[index] - raw[index]) * np.array([width, height], dtype=np.float64)
            for index in aligned
            if index in raw
        ]
        if shifts:
            shift = np.median(np.stack(shifts, axis=0), axis=0)
            transform = np.array(
                [[1.0, 0.0, float(shift[0])], [0.0, 1.0, float(shift[1])]],
                dtype=np.float32,
            )
            mask = cv2.warpAffine(
                mask,
                transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
        self.person_mask = np.ascontiguousarray(np.clip(mask, 0.0, 1.0))
        self.person_mask_updated_at = time.monotonic()

    def _validate_anchor_chain(
        self,
        candidates: dict[int, np.ndarray],
        frame_shape: tuple[int, int],
    ) -> dict[int, np.ndarray]:
        """Reject an isolated AI point that breaks an otherwise stable leg."""
        if not self.leg_lock_enabled or not self.points:
            return candidates
        height, width = frame_shape[:2]
        scale = np.array([width, height], dtype=np.float64)
        validated = {index: value.copy() for index, value in candidates.items()}
        for side in ("left", "right"):
            indices = LEG_TRACK_INDICES[side]
            matched = [
                index for index in indices if index in validated and index in self.points
            ]
            if len(matched) >= 3:
                movement = np.stack(
                    [(validated[index] - self.points[index]) * scale for index in matched],
                    axis=0,
                )
                median_movement = np.median(movement, axis=0)
                for index, displacement in zip(matched, movement):
                    residual = float(np.linalg.norm(displacement - median_movement))
                    if residual <= self.leg_anchor_jump_pixels:
                        continue
                    support = self._anchor_point_support.get(index, 1.0)
                    self._anchor_point_support[index] = support * 0.20
                    # A lone low-support jump is discarded.  High-support
                    # points remain available for controlled re-acquisition.
                    if support < 0.72:
                        validated.pop(index, None)

            for start, end in LEG_BONE_EDGES[side]:
                if (
                    start not in self.points
                    or end not in self.points
                    or start not in validated
                    or end not in validated
                ):
                    continue
                old_length = float(
                    np.linalg.norm((self.points[end] - self.points[start]) * scale)
                )
                new_length = float(
                    np.linalg.norm((validated[end] - validated[start]) * scale)
                )
                if old_length < 3.0:
                    continue
                if abs(new_length / old_length - 1.0) <= self.leg_bone_length_tolerance:
                    continue
                start_support = self._anchor_point_support.get(start, 1.0)
                end_support = self._anchor_point_support.get(end, 1.0)
                reject = end if end_support <= start_support else start
                self._anchor_point_support[reject] = (
                    self._anchor_point_support.get(reject, 1.0) * 0.15
                )
                validated.pop(reject, None)
        return validated

    def _align_anchor_points(
        self,
        anchor: PoseAnchorPacket,
        current_gray: np.ndarray,
        current_sequence: int,
    ) -> dict[int, np.ndarray]:
        self._anchor_foot_swapped = False
        self._anchor_point_support = {}
        raw: dict[int, np.ndarray] = {}
        for index in self.tracked_point_indices:
            if index >= len(anchor.landmarks):
                continue
            x, y, confidence = anchor.landmarks[index]
            if confidence < self.soft_confidence_threshold:
                continue
            candidate = np.array([x, y], dtype=np.float64)
            if (
                np.all(np.isfinite(candidate))
                and -0.10 <= candidate[0] <= 1.10
                and -0.10 <= candidate[1] <= 1.10
            ):
                mask_support = self._mask_support_at(anchor.person_mask, candidate)
                # Only reject the weakest joint estimates when both MediaPipe
                # and its person mask disagree.  This keeps toes at a silhouette
                # edge usable while stopping low-confidence bedding points.
                if (
                    self.segmentation_enabled
                    and mask_support < self.segmentation_threshold * 0.55
                    and confidence < self.confidence_threshold
                ):
                    continue
                raw[index] = candidate
                self._anchor_point_support[index] = max(0.10, mask_support)
        if not raw:
            return {}

        if current_sequence == anchor.sequence or anchor.gray is current_gray:
            aligned = raw
        else:
            aligned = self._track_point_clouds(
                anchor.gray,
                current_gray,
                raw,
                validate_backward=False,
                long_range=True,
            )
            # Very recent anchors may still be useful when an individual patch
            # lacks texture. Older unaligned points are rejected rather than
            # snapping the pedal backwards.
            age = max(0.0, time.monotonic() - anchor.captured_at)
            if age <= 0.045:
                for index, point in raw.items():
                    aligned.setdefault(index, point)

        self._refresh_person_mask(anchor, raw, aligned, current_gray.shape)
        aligned, swapped = self._lock_foot_identity(aligned, current_gray.shape)
        self._anchor_foot_swapped = swapped
        if swapped:
            original_support = self._anchor_point_support.copy()
            self._anchor_point_support = {
                index: original_support.get(self._paired_source_index(index, True), 1.0)
                for index in aligned
            }
        else:
            self._anchor_point_support = {
                index: self._anchor_point_support.get(index, 1.0)
                for index in aligned
            }
        return self._validate_anchor_chain(aligned, current_gray.shape)

    def _merge_points(
        self,
        ai_points: dict[int, np.ndarray],
        ai_confidences: dict[int, float],
        flow_points: dict[int, np.ndarray],
        frame_shape: tuple[int, int],
        now: float,
    ) -> tuple[dict[int, np.ndarray], dict[int, float], dict[int, float]]:
        height, width = frame_shape[:2]
        merged: dict[int, np.ndarray] = {}
        sources: dict[int, float] = {}
        quality: dict[int, float] = {}
        scale = np.array([width, height], dtype=np.float64)
        for index in self.tracked_point_indices:
            ai_point = ai_points.get(index)
            confidence = float(ai_confidences.get(index, 0.0))
            support = clip(float(self._anchor_point_support.get(index, 1.0)), 0.0, 1.0)
            support_for_quality = support if self.segmentation_enabled else 1.0
            ai_quality = math.sqrt(
                clip(confidence, 0.0, 1.0) * clip(support_for_quality, 0.0, 1.0)
            )
            flow_point = flow_points.get(index)
            if index in self.manual_anchor_indices:
                if flow_point is not None:
                    manual_quality = clip(
                        float(self.manual_match_quality.get(index, 1.0)),
                        0.0,
                        1.0,
                    )
                    merged[index] = np.asarray(flow_point, dtype=np.float64).copy()
                    sources[index] = manual_quality
                    quality[index] = manual_quality
                elif self.prev_gray is None and index in self.points:
                    # Seed the selected point onto the first frame after setup
                    # or calibration reset. Every later frame requires verified
                    # forward/backward local flow.
                    merged[index] = np.asarray(self.points[index], dtype=np.float64).copy()
                    sources[index] = 1.0
                    quality[index] = 1.0
                continue
            ai_is_strong = (
                ai_point is not None
                and confidence >= self.confidence_threshold
                and support >= 0.16
            )
            ai_is_soft = (
                ai_point is not None
                and confidence >= self.soft_confidence_threshold
                and support >= 0.10
            )

            if ai_is_strong and ai_point is not None:
                if flow_point is not None:
                    pixel_distance = float(np.linalg.norm((ai_point - flow_point) * scale))
                    if pixel_distance <= self.anchor_deadband:
                        point = flow_point
                        self.anchor_mismatch_counts[index] = 0
                    elif pixel_distance <= self.max_anchor_distance:
                        correction_scale = clip(
                            (pixel_distance - self.anchor_deadband)
                            / max(1e-6, self.max_anchor_distance - self.anchor_deadband),
                            0.0,
                            1.0,
                        )
                        correction = self.anchor_blend * correction_scale * (0.55 + 0.45 * support)
                        point = (1.0 - correction) * flow_point + correction * ai_point
                        self.anchor_mismatch_counts[index] = 0
                    else:
                        # Never let one new AI frame yank a stable toe/knee onto
                        # bedding.  Require the same well-supported location on
                        # several semantic anchors before controlled reacquire.
                        mismatch_count = self.anchor_mismatch_counts.get(index, 0) + 1
                        self.anchor_mismatch_counts[index] = mismatch_count
                        if (
                            mismatch_count >= self.leg_reacquire_anchors
                            and support >= 0.72
                            and confidence >= max(self.confidence_threshold, 0.35)
                        ):
                            point = ai_point
                            self.anchor_mismatch_counts[index] = 0
                        else:
                            merged[index] = flow_point
                            sources[index] = 0.80
                            anchor_age = now - self.last_ai_anchor.get(index, -1e9)
                            decay = 1.0 - 0.03 * clip(
                                max(0.0, anchor_age)
                                / max(1e-6, self.flow_hold_seconds),
                                0.0,
                                1.0,
                            )
                            quality[index] = clip(
                                self.last_ai_quality.get(index, 0.0) * decay,
                                0.0,
                                1.0,
                            )
                            continue
                else:
                    point = ai_point
                    self.anchor_mismatch_counts[index] = 0
                merged[index] = point
                sources[index] = 1.0
                quality[index] = ai_quality
                self.last_ai_anchor[index] = now
                self.last_ai_quality[index] = ai_quality
                continue

            anchor_age = now - self.last_ai_anchor.get(index, -1e9)
            if flow_point is not None and anchor_age <= self.flow_hold_seconds:
                if ai_is_soft and ai_point is not None:
                    pixel_distance = float(np.linalg.norm((ai_point - flow_point) * scale))
                    if self.anchor_deadband < pixel_distance <= self.max_anchor_distance:
                        correction_scale = clip(
                            (pixel_distance - self.anchor_deadband)
                            / max(1e-6, self.max_anchor_distance - self.anchor_deadband),
                            0.0,
                            1.0,
                        )
                        correction = min(
                            0.12,
                            self.anchor_blend * correction_scale * (0.45 + 0.55 * support),
                        )
                        flow_point = (1.0 - correction) * flow_point + correction * ai_point
                merged[index] = flow_point
                sources[index] = 0.72
                decay = 1.0 - 0.03 * clip(
                    max(0.0, anchor_age)
                    / max(1e-6, self.flow_hold_seconds),
                    0.0,
                    1.0,
                )
                flow_quality = self.last_ai_quality.get(index, 0.0) * decay
                if ai_is_soft:
                    flow_quality = max(flow_quality, ai_quality * 0.60)
                quality[index] = clip(flow_quality, 0.0, 1.0)
        return merged, sources, quality

    def _update_world_points_from_anchor(
        self,
        anchor: PoseAnchorPacket,
        now: float,
        swap_feet: bool = False,
    ) -> None:
        if not anchor.pose_detected:
            return
        for index in self.tracked_point_indices:
            source_index = self._paired_source_index(index, swap_feet)
            if source_index >= len(anchor.landmarks):
                continue
            if anchor.landmarks[source_index][2] < self.confidence_threshold:
                continue
            value = anchor.world_points.get(source_index)
            if value is not None and np.all(np.isfinite(value)):
                self.world_points[index] = np.asarray(value, dtype=np.float64).copy()
                self.last_world_anchor[index] = now

    def _current_world_points(self, now: float) -> dict[int, np.ndarray]:
        maximum_age = max(self.flow_hold_seconds, self.feature_hold_seconds)
        return {
            index: value
            for index, value in self.world_points.items()
            if now - self.last_world_anchor.get(index, -1e9) <= maximum_age
        }

    def _stabilize_foot(
        self,
        side: str,
        raw: Optional[FootFeatureObservation],
        now: float,
        dt: float,
    ) -> tuple[Optional[FootFeatureObservation], bool, bool]:
        if side == "left":
            filter_object = self.left_filter
            last_feature = self.last_left_feature
            last_time = self.last_left_time
        else:
            filter_object = self.right_filter
            last_feature = self.last_right_feature
            last_time = self.last_right_time

        manual_indices = set(MANUAL_FOOT_TRIANGLES[side])
        manual_side_is_locked = manual_indices.issubset(self.manual_anchor_indices)
        if manual_side_is_locked and raw is None:
            # A hard-locked triangle is atomic. Never bridge a missing heel,
            # ankle, toe, or invalid triangle with the generic short feature
            # hold: that would keep sending stale pressure after the user's
            # selected geometry was lost.
            filter_object.reset()
            if side == "left":
                self.last_left_feature = None
            else:
                self.last_right_feature = None
            return None, False, False

        core_threshold = clip(
            float(self.config.get("foot_core_min_confidence", 0.24)), 0.05, 1.0
        )
        if raw is not None and raw.confidence < core_threshold:
            raw = None

        if raw is not None:
            filter_input = raw.values.copy()
            if last_feature is not None:
                missing = raw.validity < 0.05
                filter_input[missing] = last_feature.values[missing]
            filtered = filter_object.update(filter_input, dt)
            output_values = filtered
            if self.pedals_only:
                raw_blend = clip(float(self.config.get("pedal_raw_feature_blend", 0.0)), 0.0, 1.0)
                output_values = raw_blend * filter_input + (1.0 - raw_blend) * filtered
            output = FootFeatureObservation(
                values=np.asarray(output_values, dtype=np.float64),
                validity=raw.validity.copy(),
                confidence=float(raw.confidence),
            )
            if side == "left":
                self.last_left_feature = output.copy()
                self.last_left_time = now
            else:
                self.last_right_feature = output.copy()
                self.last_right_time = now
            return output, True, True

        if last_feature is not None and now - last_time <= self.feature_hold_seconds:
            return last_feature.copy(), True, False

        filter_object.reset()
        if side == "left":
            self.last_left_feature = None
        else:
            self.last_right_feature = None
        return None, False, False


FOOT_FEATURE_DIMENSION = 3
FOOT_WORLD_FEATURE_START = FOOT_FEATURE_DIMENSION


def shin_local_components(
    vector: np.ndarray,
    shin_unit: np.ndarray,
    shin_length: float,
) -> np.ndarray:
    """Express a vector in a rotation/scale-invariant shin coordinate frame."""

    vector = np.asarray(vector, dtype=np.float64)
    shin_unit = np.asarray(shin_unit, dtype=np.float64)
    across = np.array([-shin_unit[1], shin_unit[0]], dtype=np.float64)
    scale = max(1e-8, float(shin_length))
    return np.array(
        [float(np.dot(vector, shin_unit)) / scale, float(np.dot(vector, across)) / scale],
        dtype=np.float64,
    )


def build_foot_feature(
    points: dict[int, np.ndarray],
    world_points: dict[int, np.ndarray],
    side: str,
    qualities: Optional[dict[int, float]] = None,
    frame_aspect_ratio: float = 1.0,
) -> Optional[FootFeatureObservation]:
    """Build the strict heel-pivot triangle used for pedal output.

    Only heel, ankle, and toe/forefoot enter the feature. Hip, knee, bedding,
    MediaPipe world points, and the remainder of the leg cannot change pedal
    demand. The heel is the translation-cancelling pivot; the ankle supplies a
    scale reference; and the heel-to-toe ray supplies pedal tilt.
    """
    if side == "left":
        ankle_index, heel_index, toe_index = LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX
    elif side == "right":
        ankle_index, heel_index, toe_index = RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX
    else:
        raise ValueError("side must be 'left' or 'right'")

    aspect = max(1e-6, float(frame_aspect_ratio))

    def metric_point(index: int) -> Optional[np.ndarray]:
        point = points.get(index)
        if point is None:
            return None
        value = np.asarray(point, dtype=np.float64).copy()
        if value.size < 2:
            return None
        value[0] *= aspect
        return value[:2]

    ankle = metric_point(ankle_index)
    heel = metric_point(heel_index)
    toe = metric_point(toe_index)
    if heel is None or ankle is None or toe is None:
        return None

    def quality(*indices: int) -> float:
        if qualities is None:
            return 1.0
        return clip(
            min(float(qualities.get(index, 0.0)) for index in indices),
            0.0,
            1.0,
        )

    pivot_to_toe = np.asarray(toe, dtype=np.float64) - np.asarray(heel, dtype=np.float64)
    pivot_to_ankle = np.asarray(ankle, dtype=np.float64) - np.asarray(heel, dtype=np.float64)
    ankle_to_toe = np.asarray(toe, dtype=np.float64) - np.asarray(ankle, dtype=np.float64)
    sole_length = float(np.linalg.norm(pivot_to_toe))
    reference_length = float(np.linalg.norm(pivot_to_ankle))
    third_edge = float(np.linalg.norm(ankle_to_toe))
    if min(sole_length, reference_length, third_edge) < 0.008:
        return None
    ratio = sole_length / reference_length
    if not 0.05 <= ratio <= 8.0:
        return None
    angle = math.atan2(float(pivot_to_toe[1]), float(pivot_to_toe[0]))
    values = np.array(
        [math.sin(angle), math.cos(angle), clip(math.log(ratio), -3.0, 3.0)],
        dtype=np.float64,
    )
    core_quality = quality(heel_index, ankle_index, toe_index)
    validity = np.full(FOOT_FEATURE_DIMENSION, core_quality, dtype=np.float64)

    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(validity)):
        return None
    _ = world_points  # Kept in the signature for third-party compatibility.
    return FootFeatureObservation(values=values, validity=validity, confidence=core_quality)


def extract_pose_features(
    result: Any,
    confidence_threshold: float,
    frame_aspect_ratio: float = 1.0,
) -> PoseFeatures:
    """Compatibility wrapper without optical flow for third-party callers/tests."""
    features = PoseFeatures()
    pose_lists = getattr(result, "pose_landmarks", None)
    if not pose_lists or len(pose_lists[0]) < 33:
        return features
    normalized = pose_lists[0]
    world_lists = getattr(result, "pose_world_landmarks", None)
    world = world_lists[0] if world_lists and len(world_lists[0]) >= 33 else None
    features.pose_detected = True
    features.landmarks = [
        (float(lm.x), float(lm.y), landmark_confidence(lm)) for lm in normalized
    ]
    points: dict[int, np.ndarray] = {}
    world_points: dict[int, np.ndarray] = {}
    for index in TRACKED_POINT_INDICES:
        if landmark_confidence(normalized[index]) >= confidence_threshold:
            points[index] = np.array(
                [float(normalized[index].x), float(normalized[index].y)],
                dtype=np.float64,
            )
            if world is not None:
                world_points[index] = landmark_xyz(world[index])
    features.tracked_landmarks = {
        index: (float(point[0]), float(point[1]), 1.0)
        for index, point in points.items()
    }
    features.left_foot = build_foot_feature(
        points,
        world_points,
        "left",
        frame_aspect_ratio=frame_aspect_ratio,
    )
    features.right_foot = build_foot_feature(
        points,
        world_points,
        "right",
        frame_aspect_ratio=frame_aspect_ratio,
    )
    features.left_foot_ok = features.left_foot is not None
    features.right_foot_ok = features.right_foot is not None
    features.left_foot_fresh = features.left_foot_ok
    features.right_foot_fresh = features.right_foot_ok
    left_wrist = points.get(LEFT_WRIST)
    right_wrist = points.get(RIGHT_WRIST)
    if left_wrist is not None and right_wrist is not None:
        delta = right_wrist - left_wrist
        if float(np.linalg.norm(delta)) >= 0.025:
            features.steering_angle = math.atan2(float(delta[1]), float(delta[0]))
            features.steering_ok = True
            features.steering_fresh = True
    return features

def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    normalized = degrees % 360
    if normalized == 0:
        return frame
    if normalized == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if normalized == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if normalized == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("camera_rotation_degrees must be 0, 90, 180, or 270")


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    width = max(160, int(width))
    height = int(round(frame.shape[0] * width / frame.shape[1]))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def prepare_processing_frames(
    frame: np.ndarray,
    config: dict[str, Any],
    enhancer: LowLightEnhancer,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize once, enhance once, then feed one image basis to AI and flow."""

    tracking_width = int(
        config.get("tracking_width", config.get("inference_width", 512))
    )
    processing_width = max(
        tracking_width,
        int(config.get("inference_width", tracking_width)),
        int(config.get("preview_width", tracking_width)),
    )
    processing_frame = resize_to_width(frame, processing_width)
    enhanced_frame = enhancer.apply(processing_frame)
    tracking_frame = (
        enhanced_frame
        if enhanced_frame.shape[1] == tracking_width
        else resize_to_width(enhanced_frame, tracking_width)
    )
    return enhanced_frame, tracking_frame


def make_preview_qimage(frame: np.ndarray, features: PoseFeatures, config: dict[str, Any]) -> QImage:
    preview_width = int(config["preview_width"])
    preview_height = int(config["preview_height"])
    preview = cv2.resize(frame, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
    threshold = float(config["landmark_confidence"])
    pedals_only = str(config.get("control_mode", "pedals_only")) == "pedals_only"
    visible_indices = (
        set(PEDAL_TRACKED_POINT_INDICES)
        if pedals_only
        else set(TRACKED_POINT_INDICES)
    )

    for start, end in POSE_CONNECTIONS:
        if pedals_only and (start not in visible_indices or end not in visible_indices):
            continue
        if start >= len(features.landmarks) or end >= len(features.landmarks):
            continue
        x1, y1, c1 = features.landmarks[start]
        x2, y2, c2 = features.landmarks[end]
        if c1 < threshold or c2 < threshold:
            continue
        p1 = (int(x1 * preview_width), int(y1 * preview_height))
        p2 = (int(x2 * preview_width), int(y2 * preview_height))
        cv2.line(preview, p1, p2, (90, 220, 255), 2, cv2.LINE_AA)

    # Required landmarks are always marked: green means usable and red means
    # currently below the confidence threshold. This makes a stalled
    # calibration diagnosable without reading a console window.
    required_preview_indices = (
        PEDAL_TRACKED_POINT_INDICES
        if pedals_only
        else (
            LEFT_WRIST,
            RIGHT_WRIST,
            LEFT_KNEE,
            RIGHT_KNEE,
            LEFT_ANKLE,
            RIGHT_ANKLE,
            LEFT_HEEL,
            RIGHT_HEEL,
            LEFT_FOOT_INDEX,
            RIGHT_FOOT_INDEX,
        )
    )
    for index in required_preview_indices:
        if index >= len(features.landmarks):
            continue
        x, y, confidence = features.landmarks[index]
        if not (-0.2 <= x <= 1.2 and -0.2 <= y <= 1.2):
            continue
        color = (80, 245, 155) if confidence >= threshold else (90, 90, 255)
        cv2.circle(
            preview,
            (int(x * preview_width), int(y * preview_height)),
            4,
            color,
            -1,
            cv2.LINE_AA,
        )

    # Cyan rings are the fused points actually used for controls. A thinner ring
    # means the point is being carried briefly by verified optical flow rather
    # than by a fresh high-confidence AI anchor.
    for index, (x, y, source_strength) in features.tracked_landmarks.items():
        if index not in visible_indices:
            continue
        if not (-0.2 <= x <= 1.2 and -0.2 <= y <= 1.2):
            continue
        centre = (int(x * preview_width), int(y * preview_height))
        thickness = 2 if source_strength >= 0.9 else 1
        cv2.circle(preview, centre, 7, (255, 210, 80), thickness, cv2.LINE_AA)

    # Confirmed manual anchors are deliberately visually distinct from AI
    # landmarks. These are the only points allowed to drive pedal output.
    hard_locked = set(features.hard_locked_indices)
    for side, indices in MANUAL_FOOT_TRIANGLES.items():
        if not set(indices).issubset(hard_locked):
            continue
        points: list[tuple[int, int]] = []
        for index in indices:
            tracked = features.tracked_landmarks.get(index)
            if tracked is None:
                points = []
                break
            points.append(
                (int(tracked[0] * preview_width), int(tracked[1] * preview_height))
            )
        if len(points) != 3:
            continue
        color = (90, 255, 130) if side == "right" else (120, 90, 255)
        cv2.line(preview, points[0], points[1], color, 3, cv2.LINE_AA)
        cv2.line(preview, points[1], points[2], color, 3, cv2.LINE_AA)
        cv2.line(preview, points[2], points[0], color, 3, cv2.LINE_AA)
        for point in points:
            cv2.circle(preview, point, 9, color, 3, cv2.LINE_AA)
        cv2.putText(
            preview,
            "PIVOT",
            (points[0][0] + 8, points[0][1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    if bool(config["mirror_preview"]):
        preview = cv2.flip(preview, 1)

    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    return QImage(
        rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    ).copy()


def ensure_pose_model(model_name: str, shared: SharedState) -> Path:
    model_name = model_name.lower().strip()
    if model_name not in MODEL_URLS:
        raise ValueError(f"Unknown model '{model_name}'. Use lite, full, or heavy.")
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / f"pose_landmarker_{model_name}.task"
    if destination.exists() and destination.stat().st_size > 1_000_000:
        return destination

    shared.update(
        mode="DOWNLOADING MODEL",
        headline=f"Downloading the MediaPipe {model_name} pose model",
        detail="This happens once; camera frames stay local on this computer",
    )
    temporary = destination.with_suffix(".task.part")
    try:
        with urllib.request.urlopen(MODEL_URLS[model_name], timeout=45) as response, temporary.open("wb") as output:
            expected = int(response.headers.get("Content-Length", "0") or "0")
            downloaded = 0
            while True:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if expected > 0:
                    percentage = downloaded / expected * 100.0
                    shared.update(detail=f"Pose model download: {percentage:0.0f}%")
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded model file is unexpectedly small")
        temporary.replace(destination)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    return destination


def create_pose_landmarker(model_path: Path, config: dict[str, Any]) -> Any:
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=clip(
            float(config["pose_detection_confidence"]), 0.05, 0.95
        ),
        min_pose_presence_confidence=clip(
            float(config["pose_presence_confidence"]), 0.05, 0.95
        ),
        min_tracking_confidence=clip(
            float(config["pose_tracking_confidence"]), 0.05, 0.95
        ),
        output_segmentation_masks=bool(config.get("enable_segmentation_mask", True)),
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def open_camera(config: dict[str, Any]) -> cv2.VideoCapture:
    index = int(config["camera_index"])
    backends: list[Optional[int]] = [None]
    if sys.platform == "win32":
        preferred = str(config.get("camera_backend", "dshow")).strip().lower()
        backend_map = {
            "dshow": getattr(cv2, "CAP_DSHOW", None),
            "msmf": getattr(cv2, "CAP_MSMF", None),
            "any": None,
        }
        preferred_backend = backend_map.get(preferred, backend_map["dshow"])
        candidates = [
            preferred_backend,
            getattr(cv2, "CAP_DSHOW", None),
            getattr(cv2, "CAP_MSMF", None),
            None,
        ]
        backends = []
        for candidate in candidates:
            if candidate not in backends:
                backends.append(candidate)

    capture: Optional[cv2.VideoCapture] = None
    for backend in backends:
        candidate = (
            cv2.VideoCapture(index)
            if backend is None
            else cv2.VideoCapture(index, int(backend))
        )
        if candidate.isOpened():
            capture = candidate
            break
        candidate.release()
    if capture is None or not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")

    if bool(config.get("camera_use_mjpg", False)):
        try:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(config["capture_width"]))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config["capture_height"]))
    capture.set(cv2.CAP_PROP_FPS, int(config["capture_fps"]))
    try:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return capture


def show_native_error(message: str) -> None:
    if sys.platform == "win32" and hasattr(ctypes, "windll"):
        try:
            ctypes.windll.user32.MessageBoxW(0, message, APP_NAME, 0x10)
            return
        except Exception:
            pass
    print(message, file=sys.stderr)


def load_config(path: Path) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError("config.json must contain a JSON object")
        unknown = sorted(set(loaded) - set(DEFAULT_CONFIG))
        if unknown:
            raise ValueError(f"Unknown config key(s): {', '.join(unknown)}")

        loaded_version = int(loaded.get("config_version", 1))
        if loaded_version > int(DEFAULT_CONFIG["config_version"]):
            raise ValueError(
                f"config_version {loaded_version} is newer than this application supports"
            )
        if loaded_version < 4:
            # v0.1-v0.3 predate the camera-rate coherent-leg pipeline. Keep
            # only physical layout choices and use the complete v0.6 profile.
            preserved_keys = {
                "camera_index",
                "camera_rotation_degrees",
                "mirror_preview",
                "monitor_index",
                "show_camera_preview",
                "auto_calibrate",
                "swap_pedals",
                "invert_steering",
            }
            config.update(
                {key: value for key, value in loaded.items() if key in preserved_keys}
            )
        else:
            config.update(loaded)
            if loaded_version == 4:
                # Preserve v0.4 camera/pedal tuning, but apply the v0.5
                # coherent-leg values before the v0.6 migration below.
                config.update(
                    {
                        "ai_anchor_max_age_seconds": 0.42,
                        "foot_identity_swap_margin_pixels": 22.0,
                        "enable_segmentation_mask": True,
                        "segmentation_threshold": 0.16,
                        "segmentation_dilate_pixels": 13,
                        "leg_lock_enabled": True,
                        "leg_lock_feature_count": 34,
                        "leg_lock_feature_quality": 0.006,
                        "leg_lock_feature_min_distance_pixels": 5.0,
                        "leg_lock_line_thickness_pixels": 24,
                        "leg_lock_roi_padding_pixels": 22,
                        "leg_lock_min_inliers": 6,
                        "leg_lock_ransac_threshold_pixels": 2.4,
                        "leg_lock_max_scale_change": 0.12,
                        "leg_lock_max_rotation_degrees": 14.0,
                        "leg_lock_max_translation_pixels": 30.0,
                        "leg_lock_landmark_blend": 0.24,
                        "leg_lock_outlier_pixels": 12.0,
                        "leg_lock_anchor_jump_pixels": 26.0,
                        "leg_lock_bone_length_tolerance": 0.34,
                        "leg_lock_reacquire_anchors": 3,
                        "optical_flow_hold_seconds": 0.34,
                    }
                )

        if loaded_version < 7:
            if loaded_version < 6:
                # Replace only values still exactly equal to the v0.5 defaults.
                # Deliberate tuning survives the v0.6 response migration.
                v5_to_v6_defaults: dict[str, tuple[Any, Any]] = {
                    "optical_flow_min_patch_votes": (3, 4),
                    "pedal_raw_feature_blend": (0.97, 0.90),
                    "pedal_sensitivity": (2.20, 1.0),
                    "pedal_curve_exponent": (0.44, 1.0),
                    "throttle_sensitivity": (3.40, 1.0),
                    "brake_sensitivity": (2.25, 1.0),
                    "throttle_deadzone": (0.0015, 0.004),
                    "throttle_curve_exponent": (0.27, 1.0),
                    "brake_curve_exponent": (0.44, 1.0),
                    "throttle_initial_response": (0.05, 0.0),
                    "pedal_prediction_max_advance": (0.10, 0.035),
                }
                if loaded_version >= 4:
                    for key, (old_default, new_default) in v5_to_v6_defaults.items():
                        current = config.get(key, old_default)
                        same = (
                            math.isclose(
                                float(current),
                                float(old_default),
                                rel_tol=0.0,
                                abs_tol=1e-9,
                            )
                            if isinstance(old_default, (int, float))
                            else current == old_default
                        )
                        if same:
                            config[key] = new_default
            config["config_version"] = 7
            old_label = {
                1: "0.1",
                2: "0.2",
                3: "0.3",
                4: "0.4",
                5: "0.5",
                6: "0.6",
            }.get(loaded_version, str(loaded_version))
            backup_path = path.with_name(f"config.v{old_label}-backup.json")
            temporary_path = path.with_name(path.name + ".v7.tmp")
            migration_saved = False
            try:
                if not backup_path.exists():
                    backup_path.write_text(
                        json.dumps(loaded, indent=2) + "\n",
                        encoding="utf-8",
                    )
                temporary_path.write_text(
                    json.dumps(config, indent=2) + "\n",
                    encoding="utf-8",
                )
                temporary_path.replace(path)
                migration_saved = True
            except OSError as exc:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
                print(
                    f"[Config] Could not save the v7 profile ({exc}); "
                    "using it in memory for this run.",
                    flush=True,
                )
            status = f" and saved {backup_path.name}" if migration_saved else ""
            print(
                "[Config] Applied the v0.7 hard-locked heel-pivot profile"
                f"{status}.",
                flush=True,
            )

    mode = str(config.get("control_mode", "pedals_only")).strip().lower()
    if mode not in {"pedals_only", "feet_and_hands"}:
        raise ValueError("control_mode must be 'pedals_only' or 'feet_and_hands'")
    config["control_mode"] = mode

    exposure_mode = str(config.get("camera_exposure_mode", "auto_lock")).strip().lower()
    if exposure_mode not in {"unchanged", "auto", "auto_lock", "manual"}:
        raise ValueError(
            "camera_exposure_mode must be unchanged, auto, auto_lock, or manual"
        )
    config["camera_exposure_mode"] = exposure_mode

    def require_range(key: str, minimum: float, maximum: float) -> None:
        value = float(config[key])
        if not minimum <= value <= maximum:
            raise ValueError(f"{key} must be between {minimum:g} and {maximum:g}")

    require_range("camera_warmup_seconds", 0.0, 5.0)
    require_range("low_light_gamma", 0.2, 3.0)
    require_range("low_light_clahe_clip_limit", 0.0, 8.0)
    grid = int(config["low_light_clahe_grid_size"])
    if not 2 <= grid <= 32:
        raise ValueError("low_light_clahe_grid_size must be between 2 and 32")
    patch_grid = int(config["optical_flow_patch_grid_size"])
    if patch_grid not in {1, 3, 5}:
        raise ValueError("optical_flow_patch_grid_size must be 1, 3, or 5")
    manual_grid = int(config["manual_anchor_patch_grid_size"])
    if manual_grid not in {3, 5, 7}:
        raise ValueError("manual_anchor_patch_grid_size must be 3, 5, or 7")
    manual_votes = int(config["manual_anchor_min_patch_votes"])
    if not 3 <= manual_votes <= manual_grid * manual_grid:
        raise ValueError(
            "manual_anchor_min_patch_votes must be between 3 and the manual grid area"
        )
    manual_template_size = int(config["manual_anchor_template_size_pixels"])
    if manual_template_size < 9 or manual_template_size > 41 or manual_template_size % 2 == 0:
        raise ValueError(
            "manual_anchor_template_size_pixels must be an odd number from 9 to 41"
        )
    manual_template_search = int(config["manual_anchor_template_search_pixels"])
    if not 2 <= manual_template_search <= 24:
        raise ValueError("manual_anchor_template_search_pixels must be between 2 and 24")
    require_range("pedal_min_feature_coverage", 0.05, 1.0)
    require_range("pedal_direction_tolerance_degrees", 0.0, 89.0)
    require_range("pedal_magnitude_blend", 0.0, 1.0)
    require_range("foot_core_min_confidence", 0.05, 1.0)
    require_range("manual_anchor_min_separation_pixels", 2.0, 40.0)
    require_range("manual_anchor_template_min_score", 0.10, 0.95)
    require_range("manual_anchor_template_rotation_degrees", 0.0, 35.0)
    require_range("manual_triangle_max_edge_change_ratio", 0.03, 0.80)
    require_range("calibration_min_heel_tilt_degrees", 0.1, 30.0)
    require_range("heel_extension_weight", 0.0, 0.45)
    require_range("throttle_response_boost", 0.0, 2.0)
    require_range("brake_response_boost", 0.0, 2.0)
    return config



def apply_performance_profile(
    config: dict[str, Any],
    profile_name: Optional[str],
) -> dict[str, Any]:
    """Apply a named runtime profile without altering camera/layout choices."""
    if profile_name is None:
        config["_runtime_profile"] = "micro-pedals"
        return config
    normalized = str(profile_name).strip().lower()
    overrides = PERFORMANCE_PROFILES.get(normalized)
    if overrides is None:
        choices = ", ".join(sorted(PERFORMANCE_PROFILES))
        raise ValueError(f"Unknown performance profile '{profile_name}'. Use {choices}.")
    config.update(copy.deepcopy(overrides))
    config["_runtime_profile"] = normalized
    return config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.json",
        help="Path to JSON configuration file",
    )
    parser.add_argument("--camera", type=int, help="Override camera index")
    parser.add_argument("--model", choices=sorted(MODEL_URLS), help="Override pose model")
    parser.add_argument(
        "--profile",
        choices=sorted(PERFORMANCE_PROFILES),
        help=(
            "Runtime tracking profile: micro-pedals, max-fps, balanced, or foot-accuracy; "
            "camera/layout settings are preserved"
        ),
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Run tracking and the HUD without creating a virtual controller",
    )
    parser.add_argument(
        "--windowed-hud",
        action="store_true",
        help="Show a normal opaque diagnostic window instead of a desktop overlay",
    )
    parser.add_argument(
        "--anchor-setup",
        action="store_true",
        help="Start the clickable six-point heel/ankle/toe hard-lock setup",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--pedals-only",
        action="store_true",
        help="Track only the feet, send RT/LT, and keep steering centred",
    )
    mode_group.add_argument(
        "--full-controls",
        action="store_true",
        help="Restore feet plus two-hand steering and five-pose calibration",
    )
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {APP_VERSION}")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if sys.platform != "win32":
        print(
            f"{APP_NAME} currently targets Windows 10/11 because its game output is XInput.",
            file=sys.stderr,
        )
        return 2

    try:
        config = load_config(args.config)
        apply_performance_profile(config, args.profile)
        if args.camera is not None:
            config["camera_index"] = args.camera
        if args.model is not None:
            config["model"] = args.model
        if args.pedals_only:
            config["control_mode"] = "pedals_only"
        elif args.full_controls:
            config["control_mode"] = "feet_and_hands"
        config["_anchor_setup_requested"] = bool(args.anchor_setup)
        config["_windowed_hud"] = bool(args.windowed_hud)
    except Exception as exc:
        show_native_error(f"Could not load configuration:\n\n{exc}")
        return 2

    try:
        cv2.setUseOptimized(True)
        cv2.setNumThreads(max(1, int(config.get("opencv_threads", 2))))
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setQuitOnLastWindowClosed(True)

    stop_event = threading.Event()
    shared = SharedState(show_preview=bool(config["show_camera_preview"]))
    shared.update(
        pedals_only=config["control_mode"] == "pedals_only",
        preview_mirrored=bool(config.get("mirror_preview", True)),
    )
    overlay = OverlayWindow(
        shared=shared,
        stop_event=stop_event,
        monitor_index=int(config["monitor_index"]),
        windowed_hud=bool(args.windowed_hud),
    )
    overlay.show()

    worker = CameraDriveWorker(
        config=config,
        shared=shared,
        stop_event=stop_event,
        preview_only=bool(args.preview_only),
    )
    worker.start()

    exit_code = app.exec()
    stop_event.set()
    if worker.is_alive():
        worker.join(timeout=3.0)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
