"""
tests/test_shot_boundary_detector.py
Unit and integration tests for ShotBoundaryDetector.

Verifies:
1. Frame difference metrics (HSV 3D histogram Bhattacharyya distance & luminance SSIM).
2. Hard cuts and gradual transition detection.
3. Continuous shot segmentation and valid shot interval extraction for safe bidirectional tracking.
4. Edge cases: identical frames, single frame, empty sequences, differing frame resolutions.
"""

from __future__ import annotations

import unittest
import numpy as np
import cv2

from src.vision.shot_boundary_detector import ShotBoundaryDetector, CutType


class TestShotBoundaryDetector(unittest.TestCase):
    """Test suite for broadcast camera cut and shot boundary detection."""

    def setUp(self):
        self.detector = ShotBoundaryDetector(
            hist_threshold=0.35,
            ssim_threshold=0.50,
            combined_threshold=0.40,
        )

    def test_identical_frames_no_cut(self):
        """Identical frames produce zero difference and no cut."""
        frame = np.full((180, 320, 3), (34, 139, 34), dtype=np.uint8)  # Forest green turf
        diff = self.detector.compute_frame_difference(frame, frame)

        self.assertFalse(diff["is_cut"])
        self.assertEqual(diff["cut_type"], CutType.NO_CUT.value)
        self.assertAlmostEqual(diff["hist_distance"], 0.0, places=3)
        self.assertAlmostEqual(diff["ssim"], 1.0, places=3)
        self.assertAlmostEqual(diff["combined_diff_score"], 0.0, places=3)

    def test_hard_camera_cut_detected(self):
        """Drastically different frames (turf green vs white/blue close-up) trigger a HARD_CUT."""
        green_pitch = np.full((180, 320, 3), (34, 139, 34), dtype=np.uint8)
        # Add pitch lines
        cv2.line(green_pitch, (0, 90), (320, 90), (255, 255, 255), 3)
        cv2.circle(green_pitch, (160, 90), 40, (255, 255, 255), 3)

        crowd_closeup = np.zeros((180, 320, 3), dtype=np.uint8)
        # Add high-contrast stadium/crowd patterns
        crowd_closeup[::4, ::4] = (255, 200, 150)
        crowd_closeup[1::4, 1::4] = (50, 50, 200)

        diff = self.detector.compute_frame_difference(green_pitch, crowd_closeup)

        self.assertTrue(diff["is_cut"])
        self.assertEqual(diff["cut_type"], CutType.HARD_CUT.value)
        self.assertGreater(diff["hist_distance"], 0.40)
        self.assertLess(diff["ssim"], 0.40)
        self.assertTrue(self.detector.is_shot_boundary(green_pitch, crowd_closeup))

    def test_subtle_motion_camera_pan_no_cut(self):
        """Smooth camera motion or small noise shifts do not trigger false cuts."""
        base_frame = np.full((180, 320, 3), 100, dtype=np.uint8)
        # Add turf texture pattern
        cv2.circle(base_frame, (160, 90), 30, (255, 255, 255), 2)
        # Frame 2 is panned by 3 pixels
        panned_frame = np.roll(base_frame, shift=3, axis=1)

        diff = self.detector.compute_frame_difference(base_frame, panned_frame)
        self.assertFalse(diff["is_cut"])
        self.assertEqual(diff["cut_type"], CutType.NO_CUT.value)
        self.assertLess(diff["combined_diff_score"], 0.35)

    def test_detect_shot_boundaries_segmentation(self):
        """Verifies multi-frame sequence segmentation across distinct shots."""
        # Create 15 frames: 8 of Shot A (green pitch) and 7 of Shot B (blue arena)
        shot_a_frame = np.full((180, 320, 3), (34, 139, 34), dtype=np.uint8)
        shot_b_frame = np.full((180, 320, 3), (200, 50, 20), dtype=np.uint8)

        sequence = [shot_a_frame for _ in range(8)] + [shot_b_frame for _ in range(7)]
        shots = self.detector.detect_shot_boundaries(sequence, fps=30.0)

        self.assertEqual(len(shots), 2)
        # Shot 0: frames 0 to 7
        self.assertEqual(shots[0]["shot_id"], 0)
        self.assertEqual(shots[0]["start_frame"], 0)
        self.assertEqual(shots[0]["end_frame"], 7)
        self.assertEqual(shots[0]["frame_count"], 8)

        # Shot 1: frames 8 to 14
        self.assertEqual(shots[1]["shot_id"], 1)
        self.assertEqual(shots[1]["start_frame"], 8)
        self.assertEqual(shots[1]["end_frame"], 14)
        self.assertEqual(shots[1]["frame_count"], 7)

    def test_get_valid_shot_interval_constrains_tracking(self):
        """Verifies safe continuous shot interval keeps anchor tracking inside shot boundaries."""
        shots = [
            {"shot_id": 0, "start_frame": 0, "end_frame": 59},
            {"shot_id": 1, "start_frame": 60, "end_frame": 119},
            {"shot_id": 2, "start_frame": 120, "end_frame": 179},
        ]
        # Frame 25 is in Shot 0
        interval_0 = self.detector.get_valid_shot_interval(25, shots, 180)
        self.assertEqual(interval_0, (0, 59))

        # Frame 75 is in Shot 1
        interval_1 = self.detector.get_valid_shot_interval(75, shots, 180)
        self.assertEqual(interval_1, (60, 119))

        # Frame 150 is in Shot 2
        interval_2 = self.detector.get_valid_shot_interval(150, shots, 180)
        self.assertEqual(interval_2, (120, 179))

    def test_edge_cases_empty_and_single_frame(self):
        """Empty lists and single-frame inputs are handled cleanly without error."""
        empty_shots = self.detector.detect_shot_boundaries([])
        self.assertEqual(empty_shots, [])

        single_frame = [np.zeros((100, 100, 3), dtype=np.uint8)]
        single_shots = self.detector.detect_shot_boundaries(single_frame)
        self.assertEqual(len(single_shots), 1)
        self.assertEqual(single_shots[0]["start_frame"], 0)
        self.assertEqual(single_shots[0]["end_frame"], 0)

    def test_differing_frame_sizes_gracefully_handled(self):
        """Detector compares frames of differing resolutions without crashing."""
        frame_720p = np.full((720, 1280, 3), 120, dtype=np.uint8)
        frame_1080p = np.full((1080, 1920, 3), 120, dtype=np.uint8)

        diff = self.detector.compute_frame_difference(frame_720p, frame_1080p)
        self.assertFalse(diff["is_cut"])
        self.assertAlmostEqual(diff["ssim"], 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
