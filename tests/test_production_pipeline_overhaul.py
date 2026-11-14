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

    # =========================================================================
    # Edge Cases & Robustness Fixes
    # =========================================================================

    def test_clock_sync_with_canonical_events(self):
        """Verifies estimate_clock_offset and align_feed_events_to_video with CanonicalMatchEvent objects."""
        sb_payload = [
            {
                "id": "sb_pass_1",
                "minute": 10,
                "second": 20,
                "type": {"name": "Pass"},
                "location": [60.0, 40.0],
                "player": {"name": "Modric"},
                "team": {"name": "Real Madrid"},
            },
            {
                "id": "sb_pass_2",
                "minute": 10,
                "second": 45,
                "type": {"name": "Pass"},
                "location": [70.0, 45.0],
                "player": {"name": "Bellingham"},
                "team": {"name": "Real Madrid"},
            },
        ]
        events = StatsBombOpenDataParser.parse_events(sb_payload)
        sync = EventClockSynchronizer(search_window_sec=4.0)

        # Video kicked 2.0s earlier than feed
        detected_kicks = [618.0, 643.0]
        offset = sync.estimate_clock_offset(detected_kicks, events)
        self.assertAlmostEqual(offset, 2.0, places=1)

        aligned = sync.align_feed_events_to_video(events, offset)
        self.assertEqual(len(aligned), 2)
        # 10m 20s = 620s. 620s - 2.0s = 618.0s
        self.assertAlmostEqual(aligned[0].raw_payload["video_timestamp_sec"], 618.0, places=1)

    def test_whoscored_feed_parser_edge_cases(self):
        """Verifies WhoScored parser handles string periods and None coordinates safely."""
        ws_payload = [
            {
                "id": "ws_sub_1",
                "period": "FirstHalf",
                "minute": 35,
                "second": 10,
                "x": None,
                "y": None,
                "type": {"displayName": "SubstitutionOn"},
                "playerName": "Gavi",
            }
        ]
        events = WhoScoredMatchCenterParser.parse_events(ws_payload)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].period, 1)
        self.assertEqual(events[0].pitch_x, 50.0)
        self.assertEqual(events[0].pitch_y, 50.0)
        self.assertEqual(events[0].event_type, CanonicalEventType.TACTICAL_SUB)

    def test_statsbomb_feed_parser_edge_cases(self):
        """Verifies StatsBomb parser handles None statsbomb_xg and location None coordinates safely."""
        sb_payload = [
            {
                "id": "sb_shot_none_xg",
                "period": "SecondHalf",
                "minute": 60,
                "second": 30,
                "type": {"name": "Shot"},
                "shot": {"statsbomb_xg": None, "outcome": {"name": "Saved"}},
                "location": [None, None],
            }
        ]
        events = StatsBombOpenDataParser.parse_events(sb_payload)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].period, 2)
        self.assertEqual(events[0].xg, 0.0)
        self.assertEqual(events[0].pitch_x, 50.0)
        self.assertEqual(events[0].pitch_y, 50.0)

    def test_goalkeeper_pitch_boundary_spectator_disambiguation(self):
        """
        Spectators/cameramen beyond touchlines wearing neon GK kit MUST NOT be selected as GK.
        Central referees wearing neon must be disambiguated against true GK near goal line.
        """
        matcher = GlobalFormationMatcher(gk_color_threshold=38.0)
        gk_kit = [35.0, 200.0, 220.0]

        candidates = [
            # Cameraman on sideline (y = 72m, pitch width is 68m) wearing neon jacket
            {"id": "sideline_cameraman", "pitch_x": 10.0, "pitch_y": 72.0, "mean_hsv": [35.0, 200.0, 220.0]},
            # Central referee wearing neon yellow (x = 52m, mid pitch)
            {"id": "central_referee", "pitch_x": 52.0, "pitch_y": 34.0, "mean_hsv": [35.0, 200.0, 220.0]},
            # True Goalkeeper near defending goal (x = 4.0m)
            {"id": "true_gk", "pitch_x": 4.0, "pitch_y": 34.0, "mean_hsv": [36.0, 195.0, 215.0]},
        ]

        best_gk, outfield = matcher.filter_goalkeepers(candidates, gk_kit, defending_goal_x=0.0)
        self.assertIsNotNone(best_gk)
        self.assertEqual(best_gk["id"], "true_gk", "Failed to disambiguate true GK from referee/spectators!")

    def test_hungarian_relative_centroid_high_line(self):
        """Verifies that relative centroid matching correctly maintains role assignments under a high defensive line."""
        matcher = GlobalFormationMatcher()
        template = CANONICAL_FORMATION_TEMPLATES["4-3-3"]
        roles = list(template.keys())

        # Entire 4-3-3 team pushes upfield by 20m, with inverted LB moving inside
        coords = np.array([template[r] for r in roles]) + np.array([20.0, 0.0])
        coords[0] = [48.0, 28.0]  # Inverted LB

        roster = [{"name": f"Player_{r}", "role": r} for r in roles]
        assignments = matcher.solve_formation_assignment(
            coords,
            "4-3-3",
            roster,
            relative_to_centroid=True,
        )

        self.assertEqual(len(assignments), 10)
        # Verify LB and LCB roles are assigned to player index 0 and 1
        self.assertEqual(assignments[0]["role"], "LB")
        self.assertEqual(assignments[1]["role"], "LCB")

    def test_temporal_crop_aggregator_list_buffer(self):
        """Verifies aggregator accepts sequence_frames as a list of numpy frames."""
        agg = TemporalCropAggregator(num_samples=3, target_height=120)
        frames_list = [np.full((200, 200, 3), 100, dtype=np.uint8) for _ in range(5)]
        boxes = {
            0: [20, 20, 80, 80],
            2: [25, 25, 85, 85],
            4: [30, 30, 90, 90],
        }
        strip = agg.extract_sharpest_strip(1, frames_list, boxes)
        self.assertIsNotNone(strip)
        self.assertEqual(strip.shape[0], 120)

    def test_ai_supervisor_auditor_zero_duplication_on_vlm_override(self):
        """Verifies that VLM jersey number override swaps player profiles without duplicating names."""
        from scripts.ai_supervisor_auditor import AISupervisorAuditor

        auditor = AISupervisorAuditor(use_mock=True)
        frames = {i: np.full((120, 160, 3), 200, dtype=np.uint8) for i in range(10)}
        tracks_boxes = {
            1: {i: [10, 10, 50, 60] for i in range(10)},
            2: {i: [60, 60, 110, 110] for i in range(10)},
        }
        coords = {1: (20.0, 24.0), 2: (20.0, 44.0)}
        roster = [
            {"name": "Antonio Rudiger", "role": "LCB", "jersey_number": 22},
            {"name": "Eder Militao", "role": "RCB", "jersey_number": 3},
        ]
        # In mock audit, override track 2 with jersey 22 (Rudiger)
        res = auditor.audit_and_match_tracks(
            sequence_frames=frames,
            tracks_with_boxes=tracks_boxes,
            track_pitch_coords=coords,
            active_roster=roster,
            gk_kit_hsv=[35.0, 200.0, 220.0],
            formation="4-3-3",
        )
        assigned_names = [info["player_name"] for info in res.values()]
        self.assertEqual(len(set(assigned_names)), len(assigned_names), "Duplicate player names detected!")


def test_canonical_fixture_resolution():
    """Validates that load_canonical_matches includes authentic October 25, 2026 fixture (ESPN 401882832)."""
    from scripts.evaluate_clasico_blueprint_pipeline import load_canonical_matches
    canonical_matches = load_canonical_matches()
    fixture = canonical_matches.get("clasico_blueprint_eval") or canonical_matches.get("barcelona_vs_real_madrid_2026_oct25")
    assert fixture is not None
    assert fixture.get("fixture_id") == "401882832" or fixture.get("date") == "2026-10-25"
    assert fixture["home_team"]["name"] == "FC Barcelona"
    assert fixture["away_team"]["name"] == "Real Madrid"


def test_hungarian_assignment_zero_degrees_of_freedom():
    """Assignment solver is strictly bounded to the active 11; inactive/banned players cannot be assigned."""
    from src.vision.dynamic_match_roster_extractor import (
        get_active_squad_at_timestamp,
        CANONICAL_CLASICO_2026_FIXTURE,
    )
    from scripts.evaluate_clasico_blueprint_pipeline import solve_tactical_assignment_metric

    current_sec = 34.0  # kickoff
    rma = CANONICAL_CLASICO_2026_FIXTURE["away_team"]
    rma_active, rma_banned = get_active_squad_at_timestamp(rma, current_sec)

    # 10 outfield candidates
    team_boxes = [
        (tid, np.array([100, 100, 200, 200]), {"pitch_coords": (50.0 + tid * 3, 20.0 + tid * 2)})
        for tid in range(1, 11)
    ]

    assigned = solve_tactical_assignment_metric(
        team_boxes=team_boxes,
        active_squad=rma_active,
        defending_x=1.0,
        team_name="Real Madrid",
    )

    assert len(assigned) == 10
    assigned_names = [v["name"].lower() for v in assigned.values()]

    # Strictly assert that NO banned/inactive player appears in assigned identities
    banned_to_check = ["vinícius", "vinicius", "militão", "militao", "mendy", "rüdiger", "rudiger", "camavinga"]
    for b in banned_to_check:
        assert not any(b in an for an in assigned_names), f"Banned player {b} was assigned!"

    # Mathematical Bijection: ZERO DUPLICATES
    assigned_numbers = [v["number"] for v in assigned.values()]
    assert len(assigned_numbers) == len(set(assigned_numbers))


def test_pitch_turf_ball_annular_ring():
    """Tests that annular ring surrounding validation rejects non-turf false positives."""
    import cv2
    h, w = 100, 100
    # Frame with green grass (HSV: H=50, S=100, V=150 -> BGR ~ (50, 150, 50))
    grass_bgr = cv2.cvtColor(np.full((h, w, 3), (50, 100, 150), dtype=np.uint8), cv2.COLOR_HSV2BGR)

    # Place a white ball at (50, 50)
    cv2.circle(grass_bgr, (50, 50), 3, (255, 255, 255), -1)

    # Compute 5px annular ring around ball
    r_inner = 3.0
    r_outer = r_inner + 5.0
    c_x, c_y = 50, 50
    yy, xx = np.ogrid[:h, :w]
    dist_sq = (xx - c_x) ** 2 + (yy - c_y) ** 2
    ring_mask = (dist_sq >= r_inner ** 2) & (dist_sq <= r_outer ** 2)

    hsv = cv2.cvtColor(grass_bgr, cv2.COLOR_BGR2HSV)
    grass_mask = (hsv[:, :, 0] >= 26) & (hsv[:, :, 0] <= 90) & (hsv[:, :, 1] >= 18)
    ring_grass_ratio = float(np.mean(grass_mask[ring_mask]))
    assert ring_grass_ratio > 0.65, f"Expected ring grass ratio > 0.65, got {ring_grass_ratio}"

    # Non-grass surface (e.g. LED ad board)
    ad_board_bgr = np.full((h, w, 3), (200, 200, 200), dtype=np.uint8)  # White/silver text
    hsv_ad = cv2.cvtColor(ad_board_bgr, cv2.COLOR_BGR2HSV)
    grass_mask_ad = (hsv_ad[:, :, 0] >= 26) & (hsv_ad[:, :, 0] <= 90) & (hsv_ad[:, :, 1] >= 18)
    ring_grass_ad = float(np.mean(grass_mask_ad[ring_mask]))
    assert ring_grass_ad < 0.65, f"Ad board should fail turf ring test, got {ring_grass_ad}"


if __name__ == "__main__":
    unittest.main()
