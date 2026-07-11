from __future__ import annotations

import math
import os
import threading
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import cv2

import camera_drive as cd


def triangle_points(
    side: str,
    foot_angle_degrees: float,
    *,
    scale: float = 1.0,
    rotation_degrees: float = 0.0,
    translation: tuple[float, float] = (0.0, 0.0),
) -> dict[int, np.ndarray]:
    """Create a heel-pivot triangle, then apply one global similarity transform."""

    if side == "left":
        heel_index, ankle_index, toe_index = (
            cd.LEFT_HEEL,
            cd.LEFT_ANKLE,
            cd.LEFT_FOOT_INDEX,
        )
    elif side == "right":
        heel_index, ankle_index, toe_index = (
            cd.RIGHT_HEEL,
            cd.RIGHT_ANKLE,
            cd.RIGHT_FOOT_INDEX,
        )
    else:
        raise ValueError("side must be left or right")

    foot_angle = math.radians(foot_angle_degrees)
    points = {
        heel_index: np.array([0.0, 0.0], dtype=np.float64),
        ankle_index: np.array([0.0, -1.0], dtype=np.float64),
        toe_index: np.array(
            [2.0 * math.cos(foot_angle), 2.0 * math.sin(foot_angle)],
            dtype=np.float64,
        ),
    }
    rotation_angle = math.radians(rotation_degrees)
    rotation = np.array(
        [
            [math.cos(rotation_angle), -math.sin(rotation_angle)],
            [math.sin(rotation_angle), math.cos(rotation_angle)],
        ],
        dtype=np.float64,
    )
    shift = np.asarray(translation, dtype=np.float64)
    return {
        index: float(scale) * (rotation @ point) + shift
        for index, point in points.items()
    }


def feature(
    side: str,
    foot_angle_degrees: float,
    **transform: object,
) -> cd.FootFeatureObservation:
    observation = cd.build_foot_feature(
        triangle_points(side, foot_angle_degrees, **transform),
        {},
        side,
    )
    if observation is None:
        raise AssertionError("valid heel-pivot geometry was rejected")
    return observation


def project(
    current: cd.FootFeatureObservation,
    neutral: cd.FootFeatureObservation,
    pressed: cd.FootFeatureObservation,
    minimum_tilt_degrees: float = 2.0,
) -> float | None:
    return cd.project_heel_hinge(
        current,
        neutral.values,
        pressed.values,
        minimum_tilt_degrees,
    )


def six_manual_anchors() -> dict[int, np.ndarray]:
    return {
        cd.LEFT_HEEL: np.array([0.24, 0.72], dtype=np.float64),
        cd.LEFT_ANKLE: np.array([0.25, 0.56], dtype=np.float64),
        cd.LEFT_FOOT_INDEX: np.array([0.42, 0.69], dtype=np.float64),
        cd.RIGHT_HEEL: np.array([0.62, 0.72], dtype=np.float64),
        cd.RIGHT_ANKLE: np.array([0.64, 0.56], dtype=np.float64),
        cd.RIGHT_FOOT_INDEX: np.array([0.82, 0.69], dtype=np.float64),
    }


class HeelTriangleGeometryTests(unittest.TestCase):
    def test_feature_is_only_angle_and_scale_normalized_triangle(self) -> None:
        observation = feature("left", 30.0)
        self.assertEqual(observation.values.shape, (3,))
        self.assertEqual(observation.validity.shape, (3,))
        self.assertAlmostEqual(float(observation.values[0]), 0.5, places=7)
        self.assertAlmostEqual(
            float(observation.values[1]), math.sqrt(3.0) / 2.0, places=7
        )
        self.assertAlmostEqual(float(observation.values[2]), math.log(2.0), places=7)
        self.assertAlmostEqual(
            float(observation.values[0] ** 2 + observation.values[1] ** 2),
            1.0,
            places=7,
        )

    def test_press_fraction_is_monotonic_and_fine_grained(self) -> None:
        neutral = feature("left", 8.0)
        pressed = feature("left", 28.0)
        fractions = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
        outputs = []
        for fraction in fractions:
            current = feature("left", 8.0 + 20.0 * fraction)
            value = project(current, neutral, pressed)
            self.assertIsNotNone(value)
            outputs.append(float(value))
        np.testing.assert_allclose(outputs, fractions, atol=0.015)
        self.assertTrue(all(a < b for a, b in zip(outputs, outputs[1:])))

    def test_calibration_view_similarity_transform_is_invariant(self) -> None:
        baseline = project(feature("left", 17.0), feature("left", 2.0), feature("left", 32.0))
        self.assertIsNotNone(baseline)
        for rotation in (-179.0, -75.0, 0.0, 90.0, 179.0):
            for scale in (0.5, 1.0, 2.0):
                transform = {
                    "rotation_degrees": rotation,
                    "scale": scale,
                    "translation": (2.3, -1.7),
                }
                value = project(
                    feature("left", 17.0, **transform),
                    feature("left", 2.0, **transform),
                    feature("left", 32.0, **transform),
                )
                self.assertIsNotNone(value)
                self.assertAlmostEqual(float(value), float(baseline), places=7)

    def test_angle_wrap_uses_short_direction(self) -> None:
        neutral = feature("right", 175.0)
        pressed = feature("right", -175.0)
        halfway = feature("right", 180.0)
        value = project(halfway, neutral, pressed)
        self.assertIsNotNone(value)
        self.assertAlmostEqual(float(value), 0.5, delta=0.015)

    def test_reverse_tilt_is_zero_and_overshoot_clamps(self) -> None:
        neutral = feature("right", 0.0)
        pressed = feature("right", 20.0)
        reverse = project(feature("right", -8.0), neutral, pressed)
        overshoot = project(feature("right", 35.0), neutral, pressed)
        self.assertEqual(reverse, 0.0)
        self.assertEqual(overshoot, 1.0)

    def test_too_small_calibration_tilt_is_unavailable(self) -> None:
        value = project(
            feature("left", 0.5),
            feature("left", 0.0),
            feature("left", 1.0),
            minimum_tilt_degrees=2.0,
        )
        self.assertIsNone(value)

    def test_heel_ankle_and_toe_are_all_required(self) -> None:
        for side, indices in (
            ("left", (cd.LEFT_HEEL, cd.LEFT_ANKLE, cd.LEFT_FOOT_INDEX)),
            ("right", (cd.RIGHT_HEEL, cd.RIGHT_ANKLE, cd.RIGHT_FOOT_INDEX)),
        ):
            for missing in indices:
                points = triangle_points(side, 10.0)
                points.pop(missing)
                self.assertIsNone(
                    cd.build_foot_feature(points, {}, side),
                    msg=f"{side} feature survived without landmark {missing}",
                )

    def test_unrelated_body_and_world_points_cannot_drive_pedal(self) -> None:
        points = triangle_points("left", 12.0)
        baseline = cd.build_foot_feature(points, {}, "left")
        self.assertIsNotNone(baseline)
        points.update(
            {
                cd.LEFT_HIP: np.array([900.0, -700.0]),
                cd.LEFT_KNEE: np.array([-500.0, 600.0]),
                cd.RIGHT_FOOT_INDEX: np.array([123.0, 456.0]),
            }
        )
        world = {
            cd.LEFT_HEEL: np.array([100.0, 200.0, 300.0]),
            cd.LEFT_ANKLE: np.array([-90.0, 80.0, -70.0]),
            cd.LEFT_FOOT_INDEX: np.array([60.0, -50.0, 40.0]),
        }
        changed = cd.build_foot_feature(points, world, "left")
        self.assertIsNotNone(changed)
        np.testing.assert_allclose(changed.values, baseline.values, atol=1e-12)
        np.testing.assert_allclose(changed.validity, baseline.validity, atol=1e-12)


class ManualAnchorUtilityTests(unittest.TestCase):
    def test_preview_click_conversion_handles_offset_and_mirroring(self) -> None:
        rectangle = (100.0, 50.0, 200.0, 100.0)
        normal = cd.preview_click_to_normalized(150.0, 100.0, rectangle, False)
        mirrored = cd.preview_click_to_normalized(150.0, 100.0, rectangle, True)
        self.assertIsNotNone(normal)
        self.assertIsNotNone(mirrored)
        np.testing.assert_allclose(normal, (0.25, 0.50), atol=1e-12)
        np.testing.assert_allclose(mirrored, (0.75, 0.50), atol=1e-12)

    def test_manual_anchor_validation_requires_six_finite_separated_points(self) -> None:
        anchors = six_manual_anchors()
        self.assertIsNone(cd.validate_manual_anchor_points(anchors, (480, 640), 8.0))

        missing = {index: point.copy() for index, point in anchors.items()}
        missing.pop(cd.RIGHT_FOOT_INDEX)
        self.assertTrue(cd.validate_manual_anchor_points(missing, (480, 640), 8.0))

        too_close = {index: point.copy() for index, point in anchors.items()}
        too_close[cd.LEFT_FOOT_INDEX] = too_close[cd.LEFT_HEEL] + np.array(
            [1.0 / 640.0, 1.0 / 480.0]
        )
        self.assertTrue(cd.validate_manual_anchor_points(too_close, (480, 640), 8.0))

        nonfinite = {index: point.copy() for index, point in anchors.items()}
        nonfinite[cd.LEFT_ANKLE] = np.array([math.nan, 0.5])
        self.assertTrue(cd.validate_manual_anchor_points(nonfinite, (480, 640), 8.0))

    def test_worker_uses_one_frozen_frame_for_all_six_clicks(self) -> None:
        config = dict(cd.DEFAULT_CONFIG)
        config["_windowed_hud"] = True
        config["_anchor_setup_requested"] = True
        shared = cd.SharedState(show_preview=True)
        worker = cd.CameraDriveWorker(
            config,
            shared,
            threading.Event(),
            preview_only=True,
        )
        worker._start_anchor_setup()
        rng = np.random.default_rng(7300)
        gray = rng.integers(0, 256, size=(480, 640), dtype=np.uint8)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        worker._ensure_anchor_setup_frame(frame)
        frozen = worker.anchor_setup_frame
        self.assertIsNotNone(frozen)
        self.assertTrue(shared.snapshot().anchor_input_available)
        anchors = six_manual_anchors()
        for index, _label in cd.MANUAL_ANCHOR_SEQUENCE:
            shared.submit_anchor_click(*anchors[index])
        worker._consume_anchor_clicks(frame.copy(), now=1.0)
        self.assertFalse(worker.anchor_setup_active)
        self.assertIsNone(worker.anchor_setup_frame)
        self.assertTrue(worker.pose_tracker.manual_anchors_complete)
        self.assertEqual(
            set(worker.pose_tracker.manual_anchor_templates),
            set(cd.MANUAL_FOOT_ANCHOR_INDICES),
        )
        self.assertTrue(worker.calibration.active)
        self.assertFalse(shared.snapshot().anchor_input_available)


class ManualHardLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = dict(cd.DEFAULT_CONFIG)
        self.tracker = cd.PoseFeatureTracker(self.config)
        self.anchors = six_manual_anchors()
        self.tracker.set_manual_foot_anchors(self.anchors)
        self.frame_shape = (480, 640)
        self.conflicting_ai = {
            index: point + np.array([0.20, -0.15], dtype=np.float64)
            for index, point in self.anchors.items()
        }

    def _assert_manual_points(self, merged: dict[int, np.ndarray]) -> None:
        for index, expected in self.anchors.items():
            self.assertIn(index, merged)
            np.testing.assert_allclose(merged[index], expected, atol=1e-12)

    def _run_conflict_frames(self, count: int) -> None:
        for frame in range(count):
            self.tracker._anchor_point_support = {
                index: 1.0 for index in self.conflicting_ai
            }
            merged, _sources, quality = self.tracker._merge_points(
                ai_points=self.conflicting_ai,
                ai_confidences={index: 1.0 for index in self.conflicting_ai},
                flow_points={index: point.copy() for index, point in self.anchors.items()},
                frame_shape=self.frame_shape,
                now=1.0 + frame / 60.0,
            )
            self._assert_manual_points(merged)
            self.tracker.points = {index: point.copy() for index, point in merged.items()}
            self.tracker.point_quality = quality

    def test_conflicting_ai_never_overrides_confirmed_manual_points(self) -> None:
        self._run_conflict_frames(
            max(6, int(self.config.get("leg_lock_reacquire_anchors", 3)) + 3)
        )

    def test_flow_loss_rejects_conflicting_ai_instead_of_snapping(self) -> None:
        # The first frame after selection is allowed to seed the clicked points.
        # Simulate an established tracker so this specifically tests later loss.
        self.tracker.prev_gray = np.zeros(self.frame_shape, dtype=np.uint8)
        self.tracker._anchor_point_support = {
            index: 1.0 for index in self.conflicting_ai
        }
        merged, _sources, _quality = self.tracker._merge_points(
            ai_points=self.conflicting_ai,
            ai_confidences={index: 1.0 for index in self.conflicting_ai},
            flow_points={},
            frame_shape=self.frame_shape,
            now=2.0,
        )
        for index in self.anchors:
            self.assertNotIn(index, merged)

    def test_reset_preserves_manual_hard_lock_definitions(self) -> None:
        self.tracker.reset()
        self._run_conflict_frames(
            max(6, int(self.config.get("leg_lock_reacquire_anchors", 3)) + 3)
        )

    def test_losing_one_vertex_drops_only_that_complete_foot_triangle(self) -> None:
        candidates = {
            index: point.copy()
            for index, point in self.anchors.items()
            if index != cd.LEFT_FOOT_INDEX
        }
        validated = self.tracker._validate_manual_triangle_flow(
            candidates,
            self.frame_shape,
        )
        for index in cd.MANUAL_FOOT_TRIANGLES["left"]:
            self.assertNotIn(index, validated)
        for index in cd.MANUAL_FOOT_TRIANGLES["right"]:
            self.assertIn(index, validated)

    def test_triangle_orientation_flip_is_rejected_even_when_edges_match(self) -> None:
        candidates = {index: point.copy() for index, point in self.anchors.items()}
        heel = self.anchors[cd.LEFT_HEEL]
        ankle = self.anchors[cd.LEFT_ANKLE]
        toe = self.anchors[cd.LEFT_FOOT_INDEX]
        axis = ankle - heel
        axis /= np.linalg.norm(axis)
        relative_toe = toe - heel
        # Reflect the toe across the heel-to-ankle axis. This preserves all
        # triangle edge lengths but reverses its handedness and hinge direction.
        candidates[cd.LEFT_FOOT_INDEX] = (
            heel + 2.0 * float(np.dot(relative_toe, axis)) * axis - relative_toe
        )
        validated = self.tracker._validate_manual_triangle_flow(
            candidates,
            self.frame_shape,
        )
        for index in cd.MANUAL_FOOT_TRIANGLES["left"]:
            self.assertNotIn(index, validated)
        for index in cd.MANUAL_FOOT_TRIANGLES["right"]:
            self.assertIn(index, validated)

    def test_triangle_is_bounded_to_confirmed_geometry_not_only_previous_frame(self) -> None:
        left_indices = cd.MANUAL_FOOT_TRIANGLES["left"]
        reference_centroid = np.mean(
            [self.anchors[index] for index in left_indices],
            axis=0,
        )
        # Pretend gradual drift already enlarged the tracked triangle by 23%;
        # the next frame changes by only 0.5%, so a previous-frame-only check
        # would accept it even though it has left the confirmed geometry.
        self.tracker.points = {
            index: (
                reference_centroid + 1.225 * (point - reference_centroid)
                if index in left_indices
                else point.copy()
            )
            for index, point in self.anchors.items()
        }
        candidates = {
            index: (
                reference_centroid + 1.230 * (point - reference_centroid)
                if index in left_indices
                else point.copy()
            )
            for index, point in self.anchors.items()
        }
        validated = self.tracker._validate_manual_triangle_flow(
            candidates,
            self.frame_shape,
        )
        for index in left_indices:
            self.assertNotIn(index, validated)
        for index in cd.MANUAL_FOOT_TRIANGLES["right"]:
            self.assertIn(index, validated)

    def test_manual_triangle_loss_never_holds_a_stale_pressed_feature(self) -> None:
        pressed = feature("left", 20.0)
        established, available, fresh = self.tracker._stabilize_foot(
            "left",
            pressed,
            now=1.0,
            dt=1.0 / 60.0,
        )
        self.assertIsNotNone(established)
        self.assertTrue(available)
        self.assertTrue(fresh)

        held, available, fresh = self.tracker._stabilize_foot(
            "left",
            None,
            now=1.01,
            dt=0.01,
        )
        self.assertIsNone(held)
        self.assertFalse(available)
        self.assertFalse(fresh)

    def test_canonical_texture_relocks_a_slightly_imprecise_flow_prediction(self) -> None:
        height, width = self.frame_shape
        rng = np.random.default_rng(7301)
        reference = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
        shift = np.array([4.0, -3.0], dtype=np.float64)
        current = cv2.warpAffine(
            reference,
            np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]]),
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        shift_normalized = shift / np.array([width, height], dtype=np.float64)
        aligned = {
            index: point + shift_normalized
            for index, point in self.anchors.items()
        }
        self.tracker.set_manual_foot_anchors(
            aligned,
            reference_gray=reference,
            reference_points=self.anchors,
        )
        flow_error = np.array([2.0 / width, 1.0 / height], dtype=np.float64)
        candidates = {
            index: point + flow_error
            for index, point in aligned.items()
        }
        verified = self.tracker._validate_manual_template_flow(current, candidates)
        self.assertEqual(set(verified), set(self.anchors))
        for index, expected in aligned.items():
            np.testing.assert_allclose(verified[index], expected, atol=0.6 / min(width, height))
            self.assertGreaterEqual(
                self.tracker.manual_match_quality[index],
                self.tracker.manual_template_min_score,
            )

    def test_camera_rate_flow_and_canonical_lock_follow_two_pixel_motion(self) -> None:
        height, width = self.frame_shape
        rng = np.random.default_rng(7303)
        reference = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
        shift = np.array([2.0, 1.0], dtype=np.float64)
        current = cv2.warpAffine(
            reference,
            np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]]),
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        self.tracker.leg_lock_enabled = False
        self.tracker.set_manual_foot_anchors(
            self.anchors,
            reference_gray=reference,
            reference_points=self.anchors,
        )
        self.tracker.prev_gray = reference
        flow = self.tracker._calculate_optical_flow(current)
        expected_shift = shift / np.array([width, height], dtype=np.float64)
        self.assertEqual(set(flow), set(self.anchors))
        for index, point in self.anchors.items():
            np.testing.assert_allclose(
                flow[index],
                point + expected_shift,
                atol=0.8 / min(width, height),
            )

    def test_canonical_texture_rejects_slow_coherent_drift_to_other_pixels(self) -> None:
        height, width = self.frame_shape
        rng = np.random.default_rng(7302)
        reference = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
        self.tracker.set_manual_foot_anchors(
            self.anchors,
            reference_gray=reference,
            reference_points=self.anchors,
        )
        drift = np.array([0.12, -0.10], dtype=np.float64)
        candidates = {
            index: point + drift
            for index, point in self.anchors.items()
        }
        verified = self.tracker._validate_manual_template_flow(reference, candidates)
        self.assertEqual(verified, {})


if __name__ == "__main__":
    unittest.main()
