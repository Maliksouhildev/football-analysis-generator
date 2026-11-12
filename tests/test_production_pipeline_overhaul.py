"""
tests/test_production_pipeline_overhaul.py
Comprehensive Verification Suite for the Production Football Pipeline Overhaul:
1. Flaw A: Event Feed Illusion & Clock Drift (EventClockSynchronizer & Event Feed Parsers)
2. Flaw B: MOT Bounding Box Identity Swaps (ProbabilisticBidirectionalTracker)
3. Flaw C: Single-Frame Blurring & VLM Hallucination (TemporalCropAggregator)
4. Flaw D: Relative Formation Graph Matching (GlobalFormationMatcher & Hungarian Solver)
5. Flaw E: Goalkeeper Penalty Box Overload Bug (Kit Color Firewall)
"""

from __future__ import annotations

import unittest
import numpy as np

from src.vision.event_clock_sync import EventClockSynchronizer
from src.vision.probabilistic_bidi_tracker import (
    ProbabilisticBidirectionalTracker,
    TrackStatus,
    compute_box_iou,
)
from src.vision.temporal_crop_aggregator import TemporalCropAggregator
from src.vision.global_formation_matcher import (
    GlobalFormationMatcher,
    CANONICAL_FORMATION_TEMPLATES,
)
from src.scraper.event_feed_parsers import (
    WhoScoredMatchCenterParser,
    StatsBombOpenDataParser,
)
from src.scraper.event_scraper import CanonicalEventType


class TestProductionPipelineOverhaul(unittest.TestCase):
    """Test suite covering all 5 fatal flaw fixes and pipeline integrations."""

    # =========================================================================
    # Flaw A: EventClockSynchronizer & Feed Parsers
    # =========================================================================

    def test_clock_sync_peak_estimation(self):
        """Verifies cross-correlation histogram peak detection on known offset."""
        sync = EventClockSynchronizer(search_window_sec=4.0, bin_width_sec=0.2)
        # Shift video kicks by exactly -2.1s relative to feed
        detected_kicks = [10.0, 25.0, 45.0, 60.0]
        feed_events = [12.1, 27.1, 47.1, 62.1]
        offset = sync.estimate_clock_offset(detected_kicks, feed_events)
        self.assertAlmostEqual(offset, 2.1, places=1)

    def test_clock_sync_edge_cases(self):
        """Empty lists or no correlations within search window return 0.0 offset."""
        sync = EventClockSynchronizer(search_window_sec=3.0)
        self.assertEqual(sync.estimate_clock_offset([], []), 0.0)
        self.assertEqual(sync.estimate_clock_offset([10.0], []), 0.0)
        # Delta 10.0s is well outside 3.0s window
        self.assertEqual(sync.estimate_clock_offset([10.0], [20.0]), 0.0)

    def test_clock_sync_velocity_spikes(self):
        """Verifies candidate kick extraction from ball velocity trajectory."""
        # 1-second trajectory: stationary ball then sudden 20 m/s displacement
        traj = [
            (50.0, 30.0, 0.0),
            (50.0, 30.0, 0.04),
            (51.0, 30.0, 0.08),  # 25 m/s spike!
            (52.0, 30.0, 0.12),
        ]
        spikes = EventClockSynchronizer.detect_velocity_spikes(traj, velocity_threshold_mps=15.0)
        self.assertGreaterEqual(len(spikes), 1)
        self.assertAlmostEqual(spikes[0], 0.08, places=2)

    def test_whoscored_feed_parser(self):
        """Verifies WhoScored JSON event extraction and 0-100 pitch normalization."""
        ws_payload = {
            "events": [
                {
                    "id": 555,
                    "minute": 10,
                    "second": 42,
                    "x": 75.4,
                    "y": 32.1,
                    "type": {"displayName": "Pass"},
                    "playerId": 8,
                    "teamId": 100,
                }
            ],
            "playerIdNameDictionary": {8: "Pedri"},
            "home": {"teamId": 100, "name": "Barcelona"},
        }
        events = WhoScoredMatchCenterParser.parse_events(ws_payload)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.primary_player.name, "Pedri")
        self.assertEqual(ev.pitch_x, 75.4)
        self.assertEqual(ev.pitch_y, 32.1)
        self.assertEqual(ev.event_type, CanonicalEventType.PASS)

    def test_statsbomb_feed_parser(self):
        """Verifies StatsBomb 120x80 coordinate transformation to 0-100 canonical space."""
        sb_payload = [
            {
                "id": "sb_shot_1",
                "minute": 23,
                "second": 15,
                "type": {"name": "Shot"},
                "shot": {"statsbomb_xg": 0.45, "outcome": {"name": "Goal"}},
                "location": [108.0, 40.0],  # 108/120 = 90.0%, 40/80 = 50.0%
                "player": {"name": "Kylian Mbappé"},
                "team": {"name": "Real Madrid"},
            }
        ]
        events = StatsBombOpenDataParser.parse_events(sb_payload)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.event_type, CanonicalEventType.GOAL)
        self.assertEqual(ev.pitch_x, 90.0)
        self.assertEqual(ev.pitch_y, 50.0)
        self.assertTrue(ev.is_big_chance)

    # =========================================================================
    # Flaw B: ProbabilisticBidirectionalTracker
    # =========================================================================

    def test_box_iou_computation(self):
        """Verifies correct IoU calculations including edge cases."""
        box1 = [0, 0, 10, 10]
        box2 = [0, 0, 10, 10]
        self.assertAlmostEqual(compute_box_iou(box1, box2), 1.0)

        # 50% overlap horizontally
        box3 = [5, 0, 15, 10]
        # Inter = 5 * 10 = 50, Union = 100 + 100 - 50 = 150 -> IoU = 1/3
        self.assertAlmostEqual(compute_box_iou(box1, box3), 1.0 / 3.0, places=4)

        # Non-overlapping
        box4 = [20, 20, 30, 30]
        self.assertAlmostEqual(compute_box_iou(box1, box4), 0.0)

    def test_tracker_overlap_demotion(self):
        """
        When two teammates intersect with IoU > 0.35 for > 0.4s, track must be
        demoted from ANCHORED_GROUND_TRUTH to PROBABILISTIC_PRIOR (conf <= 0.55).
        """
        tracker = ProbabilisticBidirectionalTracker(
            iou_threshold=0.35,
            decay_rate=0.60,
            overlap_time_threshold_sec=0.40,
            fps=10.0,  # 4 frames = 0.4s
        )
        # 6 frames of continuous overlap between teammate tracks 1 & 2
        tracks = {}
        for f in range(6):
            tracks[f] = [
                {"track_id": 1, "team": "home", "box": [100, 100, 200, 200]},
                {"track_id": 2, "team": "home", "box": [120, 100, 220, 200]},  # IoU > 0.45
            ]

        res = tracker.backward_propagate(
            {"track_id": 1, "team": "home", "box": [100, 100, 200, 200]},
            None,
            tracks,
        )

        # Early frames in backward order have sustained overlap > 0.4s
        self.assertIn(0, res)
        self.assertEqual(res[0]["status"], TrackStatus.PROBABILISTIC_PRIOR.value)
        self.assertLessEqual(res[0]["confidence"], 0.55)
        self.assertFalse(res[0]["is_ground_truth"])

    def test_tracker_occlusion_demotion(self):
        """Missing tracklet for > 0.5s must be demoted to PROBABILISTIC_PRIOR."""
        tracker = ProbabilisticBidirectionalTracker(
            occlusion_time_threshold_sec=0.50,
            fps=10.0,  # 5 frames = 0.5s
        )
        tracks = {
            f: [] if f <= 5 else [{"track_id": 7, "team": "away", "box": [50, 50, 80, 120]}]
            for f in range(10)
        }
        res = tracker.backward_propagate({"track_id": 7, "team": "away"}, None, tracks)
        self.assertEqual(res[0]["status"], TrackStatus.PROBABILISTIC_PRIOR.value)
        self.assertTrue(res[0]["occluded"])

    # =========================================================================
    # Flaw C: TemporalCropAggregator
    # =========================================================================

    def test_temporal_crop_sharpness_and_diversity(self):
        """Verifies sharpness ranking and horizontal stitching into target height."""
        agg = TemporalCropAggregator(num_samples=3, target_height=100, min_frame_spacing=2)
        frames = {}
        boxes = {}
        for f in range(10):
            # Sharpness varies by injecting high-frequency patterns
            img = np.zeros((300, 300, 3), dtype=np.uint8)
            freq = (f + 1) * 2
            img[50:150, 50:150, :] = (np.indices((100, 100))[0] * freq % 255).astype(np.uint8)[:, :, None]
            frames[f] = img
            boxes[f] = [50, 50, 150, 150]

        strip = agg.extract_sharpest_strip(track_id=1, sequence_frames=frames, track_boxes=boxes)
        self.assertIsNotNone(strip)
        self.assertEqual(strip.shape[0], 100)
        self.assertGreater(strip.shape[1], 150)

    # =========================================================================
    # Flaw D: GlobalFormationMatcher Hungarian Solver
    # =========================================================================

    def test_hungarian_formation_matching(self):
        """Verifies Hungarian bipartite assignment solves 10 outfield roles without duplicates."""
        matcher = GlobalFormationMatcher()
        coords = np.array([
            [25.0, 10.0],  # LB
            [20.0, 24.0],  # LCB
            [20.0, 44.0],  # RCB
            [25.0, 58.0],  # RB
            [38.0, 34.0],  # DM
            [52.0, 22.0],  # LCM
            [52.0, 46.0],  # RCM
            [75.0, 12.0],  # LW
            [82.0, 34.0],  # ST
            [75.0, 56.0],  # RW
        ])
        roster = [
            {"name": "Player_LB", "role": "LB"},
            {"name": "Player_LCB", "role": "LCB"},
            {"name": "Player_RCB", "role": "RCB"},
            {"name": "Player_RB", "role": "RB"},
            {"name": "Player_DM", "role": "DM"},
            {"name": "Player_LCM", "role": "LCM"},
            {"name": "Player_RCM", "role": "RCM"},
            {"name": "Player_LW", "role": "LW"},
            {"name": "Player_ST", "role": "ST"},
            {"name": "Player_RW", "role": "RW"},
        ]
        assignments = matcher.solve_formation_assignment(coords, "4-3-3", roster)
        self.assertEqual(len(assignments), 10)
        assigned_names = [a["profile"]["name"] for a in assignments.values()]
        self.assertEqual(len(set(assigned_names)), 10, "Duplicate player assignment detected!")

    # =========================================================================
    # Flaw E: Goalkeeper Kit Color Firewall
    # =========================================================================

    def test_goalkeeper_kit_color_firewall(self):
        """
        Outfield player in 18-yard box MUST NOT be marked as GK.
        Only candidate matching registered GK kit HSV distance qualifies.
        """
        matcher = GlobalFormationMatcher(gk_color_threshold=38.0)
        gk_kit = [35.0, 200.0, 220.0]  # Neon Green

        candidates = [
            # Striker in 6-yard box wearing white outfield kit
            {"id": "opp_striker", "pitch_x": 4.0, "pitch_y": 34.0, "mean_hsv": [0.0, 10.0, 240.0]},
            # Defender in penalty box wearing home kit
            {"id": "cb_defender", "pitch_x": 10.0, "pitch_y": 25.0, "mean_hsv": [10.0, 180.0, 150.0]},
            # Real Goalkeeper wearing neon kit
            {"id": "real_gk", "pitch_x": 2.0, "pitch_y": 34.0, "mean_hsv": [36.0, 195.0, 215.0]},
        ]

        best_gk, outfield = matcher.filter_goalkeepers(candidates, gk_kit)
        self.assertIsNotNone(best_gk)
        self.assertEqual(best_gk["id"], "real_gk")
        self.assertEqual(len(outfield), 2)
        outfield_ids = [o["id"] for o in outfield]
        self.assertIn("opp_striker", outfield_ids)
        self.assertIn("cb_defender", outfield_ids)


if __name__ == "__main__":
    unittest.main()
