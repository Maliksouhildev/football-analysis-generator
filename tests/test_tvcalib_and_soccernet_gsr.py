"""
tests/test_tvcalib_and_soccernet_gsr.py
Unit and integration tests for TVCalib Camera Calibration & SoccernetGSR Tracking:
1. TVCalib frame filter: Rejection of zoomed-in action shots vs acceptance of wide broadcast angles.
2. Homography matrix H in R^(3x3) mathematical condition and calibration.
3. SoccernetGSR: StrongSORT Kalman tracking across occlusions (zero ID-swapping).
4. K-Means kit color clustering on torso pixels.
5. Deterministic Ground Truth Lineup matching without VLM hallucinations.
"""

import math
import os
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.scraper.live_ground_truth import (
    GroundTruthDictionary,
    PlayerGroundTruth,
    register_ground_truth,
    clear_registered_ground_truth,
)
from src.vision.tvcalib_frame_filter import TVCalibCameraFilter, CameraFramingResult
from src.vision.soccernet_gsr_tracker import (
    SoccernetGSRTracker,
    GSRDetection,
    GSRTrackedPlayer,
)


class TestTVCalibAndSoccernetGSR(unittest.TestCase):
    """Verifies camera calibration filtering and SoccernetGSR tracking."""

    @classmethod
    def setUpClass(cls):
        cls.filter = TVCalibCameraFilter()
        cls.tracker = SoccernetGSRTracker(enable_ocr=False)

    def tearDown(self):
        clear_registered_ground_truth()

    # -------------------------------------------------------------------------
    # 1. TVCalib Camera Calibration & Frame Filtering (The Zoom-in Issue)
    # -------------------------------------------------------------------------

    def test_tvcalib_wide_broadcast_angle_accepted(self):
        """
        Verifies that a wide broadcast camera angle with high turf ratio and visible field lines
        is accepted, computes a valid homography H in R^(3x3), and is NOT skipped.
        """
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # 85% green pitch turf in BGR (H ~ 50, S ~ 180, V ~ 160)
        frame[150:, :] = (40, 160, 40)

        # Draw structural pitch lines (touchlines + transverse lines)
        # Near touchline (longitudinal)
        cv2.line(frame, (100, 920), (1820, 920), (255, 255, 255), 4)
        # Far touchline (longitudinal)
        cv2.line(frame, (250, 320), (1670, 320), (255, 255, 255), 3)
        # Halfway line (transverse)
        cv2.line(frame, (960, 320), (960, 920), (255, 255, 255), 4)
        # Penalty box line (transverse)
        cv2.line(frame, (420, 400), (420, 880), (255, 255, 255), 3)

        result: CameraFramingResult = self.filter.evaluate_frame(frame)

        self.assertTrue(result.is_wide_angle, f"Must be classified as wide angle: {result.reasons}")
        self.assertFalse(result.is_zoomed_in)
        self.assertFalse(result.skip_tactical_analysis)
        self.assertEqual(result.shot_type, "wide_broadcast_angle")
        self.assertIsNotNone(result.homography)
        self.assertEqual(result.homography.shape, (3, 3))
        self.assertGreaterEqual(result.detected_lines_count, 3)
        self.assertGreaterEqual(result.turf_ratio, 0.35)

    def test_tvcalib_zoomed_in_action_shot_rejected(self):
        """
        Verifies that a zoomed-in action play (close-up on player tackle, low turf ratio,
        lack of structural field lines) is classified as 'zoomed_in_shot' and skipped.
        """
        # Tight camera crop on player with non-turf background (dugout/stands)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Stands / crowd backdrop
        frame[:, :] = (30, 20, 50)
        # Single large player occupying 65% of frame height
        cv2.rectangle(frame, (500, 150), (1400, 850), (220, 220, 220), -1)

        result: CameraFramingResult = self.filter.evaluate_frame(frame)

        self.assertTrue(result.is_zoomed_in)
        self.assertFalse(result.is_wide_angle)
        self.assertTrue(result.skip_tactical_analysis, "Zoomed-in shot must be autonomously skipped")
        self.assertEqual(result.shot_type, "zoomed_in_shot")
        self.assertIsNone(result.homography)
        self.assertGreater(len(result.reasons), 0)

    def test_tvcalib_rejects_tight_framing_with_oversized_player(self):
        """
        Verifies that even if green turf exists, an oversized player (> 20% of frame height)
        triggers the close-up zoom rejection rule.
        """
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :] = (40, 160, 40)  # Green turf
        # Oversized player occupying 45% of frame height (h = 486 px)
        cv2.rectangle(frame, (800, 300), (1050, 786), (200, 30, 30), -1)

        result: CameraFramingResult = self.filter.evaluate_frame(frame)

        self.assertTrue(result.is_zoomed_in)
        self.assertTrue(result.skip_tactical_analysis)
        self.assertGreater(result.max_player_height_ratio, 0.20)

    # -------------------------------------------------------------------------
    # 2. SoccernetGSR: StrongSORT Multi-Object Tracking & Occlusion Handling
    # -------------------------------------------------------------------------

    def test_strongsort_persistent_tracking_across_occlusions(self):
        """
        Verifies StrongSORT tracking across sequential frames:
        Maintains persistent track ID across an occlusion gap without ID-swapping.
        """
        tracker = SoccernetGSRTracker(enable_ocr=False)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Frame 1: Initial detection of player A
        d1 = [GSRDetection(bbox=(200, 300, 260, 440), confidence=0.92, foot_pos=(230.0, 440.0))]
        t1 = tracker.update_tracks(frame, d1)
        self.assertEqual(len(t1), 1)
        initial_id = t1[0].track_id

        # Frame 2: Player A moves right
        d2 = [GSRDetection(bbox=(215, 300, 275, 440), confidence=0.91, foot_pos=(245.0, 440.0))]
        t2 = tracker.update_tracks(frame, d2)
        self.assertEqual(len(t2), 1)
        self.assertEqual(t2[0].track_id, initial_id)

        # Frame 3: Occlusion! Player A is temporarily hidden behind another player (0 detections)
        d3 = []
        t3 = tracker.update_tracks(frame, d3)
        self.assertEqual(len(t3), 1, "Track must be coasted during temporary occlusion")
        self.assertEqual(t3[0].track_id, initial_id)
        self.assertTrue(t3[0].is_occluded)

        # Frame 4: Player A re-emerges
        d4 = [GSRDetection(bbox=(245, 300, 305, 440), confidence=0.88, foot_pos=(275.0, 440.0))]
        t4 = tracker.update_tracks(frame, d4)
        self.assertEqual(len(t4), 1)
        self.assertEqual(t4[0].track_id, initial_id, "Track ID must remain identical after re-emergence (Zero ID-swapping)")
        self.assertFalse(t4[0].is_occluded)

    # -------------------------------------------------------------------------
    # 3. K-Means Kit Color Clustering
    # -------------------------------------------------------------------------

    def test_kmeans_kit_clustering(self):
        """Verifies that K-Means clustering extracts dominant torso kit color and distinguishes teams."""
        frame = np.full((600, 600, 3), (35, 140, 45), dtype=np.uint8)  # Green turf background

        # Away white kit player (245, 245, 245) in bbox (100, 100, 200, 320)
        frame[130:230, 120:180] = (245, 245, 245)
        rgb_white, team_hint_w = SoccernetGSRTracker.extract_torso_kit_color(frame, (100, 100, 200, 320))
        self.assertGreater(rgb_white[0], 200)
        self.assertEqual(team_hint_w, "away")

        # Home crimson kit player (BGR: 20, 20, 180 -> RGB: 180, 20, 20) in bbox (300, 100, 400, 320)
        frame[130:230, 320:380] = (20, 20, 180)
        rgb_red, team_hint_r = SoccernetGSRTracker.extract_torso_kit_color(frame, (300, 100, 400, 320))
        self.assertGreater(rgb_red[0], rgb_red[2])  # R > B
        self.assertEqual(team_hint_r, "home")

    # -------------------------------------------------------------------------
    # 4. Live Ground Truth Lineup Matching (Zero VLM Guessing)
    # -------------------------------------------------------------------------

    def test_live_lineup_matching_without_vlm_hallucination(self):
        """
        Verifies matching OCR-detected jersey number + kit color to active live API lineup
        to lock in player identity deterministically without VLM guessing.
        """
        gt = GroundTruthDictionary(
            provider="sportmonks",
            fixture_id="ucl_final_2026",
            home_team_id=101,
            home_team="Real Madrid",
            away_team_id=102,
            away_team="Barcelona",
            starting_xi_home=[
                PlayerGroundTruth(
                    player_id=231747,
                    name="Kylian Mbappé",
                    short_name="Mbappé",
                    number=9,
                    position="CF",
                    is_gk=False,
                    team_id=101,
                    team_side="home",
                    team_name="Real Madrid",
                    is_starter=True,
                ),
                PlayerGroundTruth(
                    player_id=252371,
                    name="Jude Bellingham",
                    short_name="Bellingham",
                    number=5,
                    position="MF",
                    is_gk=False,
                    team_id=101,
                    team_side="home",
                    team_name="Real Madrid",
                    is_starter=True,
                ),
            ],
            starting_xi_away=[
                PlayerGroundTruth(
                    player_id=277643,
                    name="Lamine Yamal",
                    short_name="L. Yamal",
                    number=19,
                    position="RW",
                    is_gk=False,
                    team_id=102,
                    team_side="away",
                    team_name="Barcelona",
                    is_starter=True,
                ),
            ],
            substitutes_home=[],
            substitutes_away=[],
        )
        register_ground_truth(gt)

        tracker = SoccernetGSRTracker(enable_ocr=False)

        # Player with detected jersey number 19
        p19 = GSRTrackedPlayer(
            track_id=10,
            bbox=(200, 300, 260, 440),
            foot_pos=(230.0, 440.0),
            consensus_jersey_number=19,
        )
        matched19 = tracker.match_live_ground_truth(p19, match_minute=30)
        self.assertTrue(matched19)
        self.assertEqual(p19.player_name, "Lamine Yamal")
        self.assertEqual(p19.player_id, 277643)
        self.assertEqual(p19.position, "RW")
        self.assertTrue(p19.is_identity_locked)

        # Player with detected jersey number 9
        p9 = GSRTrackedPlayer(
            track_id=11,
            bbox=(400, 300, 460, 440),
            foot_pos=(430.0, 440.0),
            consensus_jersey_number=9,
        )
        matched9 = tracker.match_live_ground_truth(p9, match_minute=30)
        self.assertTrue(matched9)
        self.assertEqual(p9.player_name, "Kylian Mbappé")
        self.assertEqual(p9.player_id, 231747)
        self.assertTrue(p9.is_identity_locked)


if __name__ == "__main__":
    unittest.main()
