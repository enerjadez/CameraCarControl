from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

import camera_drive as cd
from camera_fps_test import Result, result_score, select_best_result, selected_fps
from camera_imaging import LowLightEnhancer, apply_camera_controls, compute_light_metrics


class FakeCapture:
    def __init__(self, backend: str = "DSHOW") -> None:
        self.backend = backend
        self.values: dict[int, float] = {
            cd.cv2.CAP_PROP_EXPOSURE: -6.5,
            cd.cv2.CAP_PROP_GAIN: 2.0,
            cd.cv2.CAP_PROP_BRIGHTNESS: 0.0,
        }
        self.calls: list[tuple[int, float]] = []
        self.read_count = 0

    def getBackendName(self) -> str:
        return self.backend

    def set(self, prop: int, value: float) -> bool:
        self.calls.append((prop, float(value)))
        self.values[prop] = float(value)
        return True

    def get(self, prop: int) -> float:
        return self.values.get(prop, 0.0)

    def read(self):
        self.read_count += 1
        return True, np.full((12, 16, 3), 30, dtype=np.uint8)


def canonical_points(press: float = 0.0) -> dict[int, np.ndarray]:
    angle = math.radians(-20.0 * float(press))
    heel = np.array([0.0, 1.0], dtype=np.float64)
    return {
        cd.LEFT_HIP: np.array([0.0, -1.0]),
        cd.LEFT_KNEE: np.array([0.0, 0.0]),
        cd.LEFT_ANKLE: np.array([0.0, 0.60]),
        cd.LEFT_HEEL: heel,
        cd.LEFT_FOOT_INDEX: heel
        + 0.65 * np.array([math.cos(angle), math.sin(angle)], dtype=np.float64),
    }


def transform_points(
    points: dict[int, np.ndarray],
    degrees: float,
    scale: float,
    translation: tuple[float, float],
) -> dict[int, np.ndarray]:
    angle = math.radians(degrees)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    shift = np.asarray(translation, dtype=np.float64)
    return {key: scale * (rotation @ value) + shift for key, value in points.items()}


def pixel_rotated_normalized_points(
    points: dict[int, np.ndarray],
    degrees: float,
    width: int = 640,
    height: int = 480,
) -> dict[int, np.ndarray]:
    angle = math.radians(degrees)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    centre = np.array([width * 0.52, height * 0.48])
    return {
        key: (110.0 * (rotation @ value) + centre) / np.array([width, height])
        for key, value in points.items()
    }


class LowLightTests(unittest.TestCase):
    def test_disabled_is_exact_bypass(self) -> None:
        frame = np.arange(18 * 24 * 3, dtype=np.uint8).reshape(18, 24, 3)
        enhancer = LowLightEnhancer(
            {
                "low_light_enhancement_enabled": False,
                "low_light_gamma": 0.72,
                "low_light_clahe_clip_limit": 1.6,
                "low_light_clahe_grid_size": 8,
            }
        )
        np.testing.assert_array_equal(enhancer.apply(frame), frame)

    def test_dark_frame_is_brightened_deterministically(self) -> None:
        yy, xx = np.indices((96, 128))
        gray = (10 + xx // 6 + ((xx + yy) % 2) * 4).astype(np.uint8)
        frame = np.repeat(gray[:, :, None], 3, axis=2)
        enhancer = LowLightEnhancer(cd.DEFAULT_CONFIG)
        first = enhancer.apply(frame)
        second = enhancer.apply(frame)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(float(np.median(first)), float(np.median(frame)))
        self.assertGreater(int(enhancer.gamma_lut[32]), 32)
        self.assertEqual(int(enhancer.gamma_lut[0]), 0)
        self.assertEqual(int(enhancer.gamma_lut[255]), 255)

    def test_light_metrics_distinguish_texture(self) -> None:
        flat = np.full((96, 128, 3), 30, dtype=np.uint8)
        checker = flat.copy()
        checker[::4, :, :] = 90
        checker[:, ::4, :] = 90
        self.assertGreater(
            compute_light_metrics(checker).corner_count,
            compute_light_metrics(flat).corner_count,
        )

    def test_nearby_dark_frames_do_not_flicker_wildly(self) -> None:
        yy, xx = np.indices((120, 160))
        base = (14 + xx // 8 + yy // 16 + ((xx + yy) % 3)).astype(np.uint8)
        first = np.repeat(base[:, :, None], 3, axis=2)
        second = np.clip(first.astype(np.int16) + 1, 0, 255).astype(np.uint8)
        enhancer = LowLightEnhancer(cd.DEFAULT_CONFIG)
        delta = np.abs(
            enhancer.apply(second).astype(np.int16)
            - enhancer.apply(first).astype(np.int16)
        )
        self.assertLess(float(np.mean(delta)), 5.0)
        self.assertLess(float(np.percentile(delta, 99.0)), 20.0)

    def test_processing_pipeline_enhances_once_before_ai_and_flow_resize(self) -> None:
        class SpyEnhancer:
            def __init__(self) -> None:
                self.calls = 0

            def apply(self, frame: np.ndarray) -> np.ndarray:
                self.calls += 1
                return np.clip(frame.astype(np.int16) + 7, 0, 255).astype(np.uint8)

        frame = np.full((720, 1280, 3), 10, dtype=np.uint8)
        config = dict(cd.DEFAULT_CONFIG)
        config.update({"tracking_width": 640, "inference_width": 512})
        spy = SpyEnhancer()
        inference_frame, tracking_frame = cd.prepare_processing_frames(
            frame, config, spy
        )
        self.assertEqual(spy.calls, 1)
        self.assertEqual(inference_frame.shape[1], 640)
        self.assertEqual(tracking_frame.shape[1], 640)
        self.assertTrue(np.all(inference_frame == 17))
        self.assertTrue(np.all(tracking_frame == 17))


class CameraControlTests(unittest.TestCase):
    def test_dshow_auto_lock_sequence(self) -> None:
        capture = FakeCapture("DSHOW")
        config = dict(cd.DEFAULT_CONFIG)
        config.update(
            {
                "camera_exposure_mode": "auto_lock",
                "camera_warmup_seconds": 0.01,
                "camera_gain": 3.0,
            }
        )
        report = apply_camera_controls(capture, config)
        auto_calls = [
            value
            for prop, value in capture.calls
            if prop == cd.cv2.CAP_PROP_AUTO_EXPOSURE
        ]
        self.assertEqual(auto_calls[:2], [0.75, 0.25])
        self.assertTrue(
            any(prop == cd.cv2.CAP_PROP_EXPOSURE for prop, _ in capture.calls)
        )
        self.assertTrue(any(prop == cd.cv2.CAP_PROP_GAIN for prop, _ in capture.calls))
        self.assertGreater(capture.read_count, 0)
        self.assertEqual(report.exposure_mode, "locked")
        gain_index = next(
            index
            for index, (prop, _value) in enumerate(capture.calls)
            if prop == cd.cv2.CAP_PROP_GAIN
        )
        auto_index = next(
            index
            for index, (prop, value) in enumerate(capture.calls)
            if prop == cd.cv2.CAP_PROP_AUTO_EXPOSURE and value == 0.75
        )
        self.assertLess(gain_index, auto_index)

    def test_zero_exposure_readback_never_locks(self) -> None:
        capture = FakeCapture("DSHOW")
        capture.values[cd.cv2.CAP_PROP_EXPOSURE] = 0.0
        config = dict(cd.DEFAULT_CONFIG)
        config["camera_warmup_seconds"] = 0.0
        report = apply_camera_controls(capture, config)
        auto_calls = [
            value
            for prop, value in capture.calls
            if prop == cd.cv2.CAP_PROP_AUTO_EXPOSURE
        ]
        self.assertNotIn(0.25, auto_calls)
        self.assertEqual(report.exposure_mode, "auto")

    def test_warmup_honours_stop_event(self) -> None:
        capture = FakeCapture("DSHOW")
        config = dict(cd.DEFAULT_CONFIG)
        config["camera_warmup_seconds"] = 5.0
        stopped = threading.Event()
        stopped.set()
        started = cd.time.monotonic()
        apply_camera_controls(capture, config, stopped)
        self.assertLess(cd.time.monotonic() - started, 0.15)


class FootGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.neutral = cd.build_foot_feature(canonical_points(0.0), {}, "left")
        self.pressed = cd.build_foot_feature(canonical_points(1.0), {}, "left")
        assert self.neutral is not None and self.pressed is not None

    @staticmethod
    def project(
        observation: cd.FootFeatureObservation,
        neutral: cd.FootFeatureObservation,
        pressed: cd.FootFeatureObservation,
    ):
        return cd.project_heel_hinge(
            observation,
            neutral.values,
            pressed.values,
            minimum_tilt_degrees=2.0,
            extension_weight=0.0,
        )

    def test_similarity_transform_invariance(self) -> None:
        for press in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            canonical = cd.build_foot_feature(canonical_points(press), {}, "left")
            assert canonical is not None
            canonical_value = self.project(canonical, self.neutral, self.pressed)
            for angle in (-75.0, -35.0, 0.0, 40.0, 75.0):
                for scale in (0.6, 1.0, 1.8):
                    transformed_neutral = cd.build_foot_feature(
                        transform_points(
                            canonical_points(0.0), angle, scale, (2.3, -0.8)
                        ),
                        {},
                        "left",
                    )
                    transformed_pressed = cd.build_foot_feature(
                        transform_points(
                            canonical_points(1.0), angle, scale, (2.3, -0.8)
                        ),
                        {},
                        "left",
                    )
                    transformed_current = cd.build_foot_feature(
                        transform_points(
                            canonical_points(press), angle, scale, (2.3, -0.8)
                        ),
                        {},
                        "left",
                    )
                    assert transformed_neutral is not None
                    assert transformed_pressed is not None
                    assert transformed_current is not None
                    self.assertAlmostEqual(
                        float(
                            self.project(
                                transformed_current,
                                transformed_neutral,
                                transformed_pressed,
                            )
                        ),
                        float(canonical_value),
                        places=7,
                    )

    def test_non_square_camera_rotation_invariance(self) -> None:
        aspect = 640.0 / 480.0
        for press in (0.0, 0.25, 0.5, 1.0):
            for angle in (-75.0, -30.0, 30.0, 75.0, 90.0):
                rotated_neutral = cd.build_foot_feature(
                    pixel_rotated_normalized_points(canonical_points(0.0), angle),
                    {},
                    "left",
                    frame_aspect_ratio=aspect,
                )
                rotated_pressed = cd.build_foot_feature(
                    pixel_rotated_normalized_points(canonical_points(1.0), angle),
                    {},
                    "left",
                    frame_aspect_ratio=aspect,
                )
                rotated_current = cd.build_foot_feature(
                    pixel_rotated_normalized_points(canonical_points(press), angle),
                    {},
                    "left",
                    frame_aspect_ratio=aspect,
                )
                assert rotated_neutral is not None
                assert rotated_pressed is not None
                assert rotated_current is not None
                self.assertAlmostEqual(
                    float(
                        self.project(
                            rotated_current,
                            rotated_neutral,
                            rotated_pressed,
                        )
                    ),
                    press,
                    places=7,
                )

    def test_projection_tracks_press_fraction(self) -> None:
        values = []
        for press in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            observation = cd.build_foot_feature(canonical_points(press), {}, "left")
            assert observation is not None
            values.append(float(self.project(observation, self.neutral, self.pressed)))
        expected = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
        np.testing.assert_allclose(values, expected, atol=0.015)

    def test_missing_triangle_vertex_is_unavailable(self) -> None:
        for missing in (cd.LEFT_HEEL, cd.LEFT_ANKLE, cd.LEFT_FOOT_INDEX):
            points = canonical_points(0.5)
            points.pop(missing)
            self.assertIsNone(cd.build_foot_feature(points, {}, "left"))

    def test_degenerate_heel_invalidates_the_triangle(self) -> None:
        points = canonical_points(0.5)
        points[cd.LEFT_HEEL] = points[cd.LEFT_ANKLE].copy()
        self.assertIsNone(cd.build_foot_feature(points, {}, "left"))

    def test_reverse_heel_tilt_is_rejected(self) -> None:
        reverse = cd.build_foot_feature(canonical_points(-0.4), {}, "left")
        assert reverse is not None
        value = self.project(reverse, self.neutral, self.pressed)
        self.assertEqual(value, 0.0)

    def test_only_the_three_triangle_points_affect_the_feature(self) -> None:
        baseline = cd.build_foot_feature(canonical_points(0.5), {}, "left")
        assert baseline is not None
        changed_points = canonical_points(0.5)
        changed_points[cd.LEFT_HIP] = np.array([800.0, -600.0])
        changed_points[cd.LEFT_KNEE] = np.array([-700.0, 500.0])
        changed = cd.build_foot_feature(
            changed_points,
            {
                cd.LEFT_HEEL: np.array([100.0, 200.0, 300.0]),
                cd.LEFT_ANKLE: np.array([-90.0, 80.0, -70.0]),
                cd.LEFT_FOOT_INDEX: np.array([60.0, -50.0, 40.0]),
            },
            "left",
        )
        assert changed is not None
        np.testing.assert_allclose(changed.values, baseline.values, atol=1e-12)

    def test_live_point_quality_preserves_real_confidence(self) -> None:
        config = dict(cd.DEFAULT_CONFIG)
        tracker = cd.PoseFeatureTracker(config)
        points = canonical_points(0.0)
        core = {
            index: points[index]
            for index in (cd.LEFT_HEEL, cd.LEFT_ANKLE, cd.LEFT_FOOT_INDEX)
        }
        tracker._anchor_point_support = {index: 0.16 for index in core}
        merged, _sources, qualities = tracker._merge_points(
            ai_points=core,
            ai_confidences={index: 0.21 for index in core},
            flow_points={},
            frame_shape=(480, 640),
            now=1.0,
        )
        observation = cd.build_foot_feature(
            merged, {}, "left", qualities=qualities
        )
        assert observation is not None
        self.assertLess(observation.confidence, config["foot_core_min_confidence"])

    def test_verified_flow_retains_anchor_quality_until_hold_expires(self) -> None:
        config = dict(cd.DEFAULT_CONFIG)
        tracker = cd.PoseFeatureTracker(config)
        points = canonical_points(0.0)
        core = {
            index: points[index]
            for index in (cd.LEFT_HEEL, cd.LEFT_ANKLE, cd.LEFT_FOOT_INDEX)
        }
        tracker._anchor_point_support = {index: 0.40 for index in core}
        merged, _sources, anchored_quality = tracker._merge_points(
            ai_points=core,
            ai_confidences={index: 0.40 for index in core},
            flow_points={},
            frame_shape=(480, 640),
            now=1.0,
        )
        tracker.points = merged
        tracker.point_quality = anchored_quality
        _merged, _sources, flow_quality = tracker._merge_points(
            ai_points={},
            ai_confidences={},
            flow_points=core,
            frame_shape=(480, 640),
            now=1.0 + config["optical_flow_hold_seconds"] * 0.90,
        )
        for index in core:
            self.assertGreaterEqual(
                flow_quality[index], anchored_quality[index] * 0.97
            )


class TrackerConfigurationTests(unittest.TestCase):
    def test_one_point_patch_grid_clamps_vote_requirement(self) -> None:
        config = dict(cd.DEFAULT_CONFIG)
        config.update(
            {
                "optical_flow_patch_grid_size": 1,
                "optical_flow_min_patch_votes": 4,
            }
        )
        tracker = cd.PoseFeatureTracker(config)
        self.assertEqual(tracker.patch_grid_size, 1)
        self.assertEqual(tracker.min_patch_votes, 1)


class MappingCurveTests(unittest.TestCase):
    def test_fine_curve_keeps_resolution_and_endpoints(self) -> None:
        inputs = (0.0, 0.02, 0.10, 0.25, 0.50, 0.75, 1.0)
        outputs = [
            cd.shape_unipolar(value, 0.0, 1.0, response_boost=0.55)
            for value in inputs
        ]
        self.assertEqual(outputs[0], 0.0)
        self.assertEqual(outputs[-1], 1.0)
        self.assertTrue(all(a < b for a, b in zip(outputs, outputs[1:])))
        self.assertLess(outputs[1], 0.15)
        self.assertGreater(outputs[3] - outputs[2], 0.08)
        self.assertLess(outputs[-2], 0.95)

    def test_full_controls_neutralize_invalid_triangle_observation(self) -> None:
        neutral_observation = cd.build_foot_feature(canonical_points(0.0), {}, "left")
        pressed_observation = cd.build_foot_feature(canonical_points(1.0), {}, "left")
        assert neutral_observation is not None and pressed_observation is not None
        neutral = neutral_observation.values
        pressed = pressed_observation.values
        noise = np.full(cd.FOOT_FEATURE_DIMENSION, 0.003)
        reliability = np.ones(cd.FOOT_FEATURE_DIMENSION)
        calibration = cd.CalibrationData(
            left_foot_neutral=neutral,
            left_foot_pressed=pressed,
            left_foot_noise=noise,
            left_foot_reliability=reliability,
            left_foot_signal_to_noise=10.0,
            right_foot_neutral=neutral,
            right_foot_pressed=pressed,
            right_foot_noise=noise,
            right_foot_reliability=reliability,
            right_foot_signal_to_noise=10.0,
            steering_neutral=0.0,
            steering_left=-0.2,
            steering_right=0.2,
        )
        config = dict(cd.DEFAULT_CONFIG)
        config["control_mode"] = "feet_and_hands"
        mapper = cd.ControlMapper(calibration, config)
        invalid_triangle = cd.FootFeatureObservation(
            values=neutral.copy(),
            validity=np.zeros(cd.FOOT_FEATURE_DIMENSION),
            confidence=1.0,
        )
        features = cd.PoseFeatures(
            left_foot=invalid_triangle,
            right_foot=invalid_triangle,
            steering_angle=0.0,
            left_foot_ok=True,
            right_foot_ok=True,
            steering_ok=True,
        )
        steering, gas, brake = mapper.map(features)
        self.assertEqual((steering, gas, brake), (0.0, 0.0, 0.0))


class ConfigMigrationTests(unittest.TestCase):
    def test_v5_migrates_and_preserves_custom_camera_values(self) -> None:
        fixture = Path(__file__).with_name("config_v05_fixture.json")
        v5 = json.loads(fixture.read_text(encoding="utf-8"))
        v5.update(
            {
                "config_version": 5,
                "camera_index": 3,
                "capture_width": 960,
                "capture_height": 540,
                "capture_fps": 60,
                # Non-default tuning must survive the migration.
                "brake_sensitivity": 2.77,
                "brake_curve_exponent": 0.70,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(v5), encoding="utf-8")
            migrated = cd.load_config(path)
            self.assertEqual(migrated["config_version"], 7)
            self.assertEqual(migrated["camera_index"], 3)
            self.assertEqual(migrated["capture_width"], 960)
            self.assertEqual(migrated["capture_fps"], 60)
            self.assertEqual(migrated["throttle_sensitivity"], 1.0)
            self.assertEqual(migrated["throttle_curve_exponent"], 1.0)
            self.assertEqual(migrated["brake_sensitivity"], 2.77)
            self.assertEqual(migrated["brake_curve_exponent"], 0.70)
            backup = Path(directory) / "config.v0.5-backup.json"
            self.assertTrue(backup.exists())
            backup_contents = backup.read_text(encoding="utf-8")
            saved_contents = path.read_text(encoding="utf-8")
            second = cd.load_config(path)
            self.assertEqual(second["config_version"], 7)
            self.assertEqual(path.read_text(encoding="utf-8"), saved_contents)
            self.assertEqual(backup.read_text(encoding="utf-8"), backup_contents)
            self.assertFalse((Path(directory) / "config.json.v7.tmp").exists())

    def test_pre_v4_keeps_layout_but_not_obsolete_tracking_tuning(self) -> None:
        legacy = {
            "config_version": 3,
            "camera_index": 2,
            "camera_rotation_degrees": 90,
            "swap_pedals": True,
            "throttle_sensitivity": 9.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            migrated = cd.load_config(path)
            self.assertEqual(migrated["config_version"], 7)
            self.assertEqual(migrated["camera_index"], 2)
            self.assertEqual(migrated["camera_rotation_degrees"], 90)
            self.assertTrue(migrated["swap_pedals"])
            self.assertEqual(migrated["throttle_sensitivity"], 1.0)
            self.assertTrue((Path(directory) / "config.v0.3-backup.json").exists())


class BenchmarkSelectionTests(unittest.TestCase):
    @staticmethod
    def result(
        requested_fps: int,
        measured_fps: float,
        luma: float,
        dark: float,
        sharpness: float,
        corners: float,
        reported_fps: float | None = None,
    ) -> Result:
        return Result(
            requested_width=640,
            requested_height=480,
            requested_fps=requested_fps,
            actual_width=640,
            actual_height=480,
            reported_fps=float(reported_fps or requested_fps),
            measured_fps=measured_fps,
            successful_frames=60,
            backend="DSHOW",
            median_luma=luma,
            p10_luma=max(0.0, luma - 15.0),
            dark_fraction=dark,
            sharpness=sharpness,
            corner_count=corners,
            unique_ratio=1.0,
            p95_interval_ms=1000.0 / max(1.0, measured_fps),
        )

    def test_clean_60_beats_dark_noisy_120(self) -> None:
        dark_120 = self.result(120, 118.0, 8.0, 0.92, 6.0, 4.0)
        clean_60 = self.result(60, 59.0, 72.0, 0.08, 55.0, 48.0)
        self.assertIs(select_best_result([dark_120, clean_60]), clean_60)

    def test_negotiated_fps_is_saved(self) -> None:
        fallback = self.result(240, 30.0, 60.0, 0.1, 40.0, 35.0, reported_fps=30.0)
        self.assertEqual(selected_fps(fallback), 30)

    def test_unstable_frame_cadence_is_penalized(self) -> None:
        stable = self.result(60, 60.0, 60.0, 0.1, 40.0, 35.0)
        unstable = self.result(60, 60.0, 60.0, 0.1, 40.0, 35.0)
        unstable = Result(**{**unstable.__dict__, "p95_interval_ms": 80.0})
        self.assertGreater(result_score(stable, 60.0, 40.0), result_score(unstable, 60.0, 40.0))


if __name__ == "__main__":
    unittest.main()
