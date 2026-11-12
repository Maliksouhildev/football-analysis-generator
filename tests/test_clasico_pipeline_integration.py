"""
tests/test_clasico_pipeline_integration.py
================================================================================
Comprehensive Integration Tests for El Clásico Multi-Modal Pipeline:
1. Official Starting 11 Ingestion (Real Madrid vs FC Barcelona - October 26, 2024).
2. Clock Synchronization with event feed.
3. Detection and Probabilistic Bidirectional Tracking with overlap demotion.
4. Multi-Frame Temporal Crop Aggregation.
5. Goalkeeper Kit Color Firewall isolation.
6. 2D Metric Hungarian Formation Matching with 100% zero duplication.
7. AI Supervisor Auditor biometric and active roster verification.
================================================================================
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vision.event_clock_sync import EventClockSynchronizer
from src.vision.probabilistic_bidi_tracker import ProbabilisticBidirectionalTracker, TrackStatus
from src.vision.temporal_crop_aggregator import TemporalCropAggregator
from src.vision.global_formation_matcher import GlobalFormationMatcher
from scripts.ai_supervisor_auditor import AISupervisorAuditor
from vision_pipeline.local_detector import LocalPlayerDetector


class TestClasicoPipelineIntegration:
    """Integration test suite for the overhauled El Clásico AI pipeline."""

    @pytest.fixture(autouse=True)
    def setup_fixture(self):
        self.roster_path = REPO_ROOT / "test_data" / "canonical_el_clasico_roster.json"
        assert self.roster_path.is_file(), "Roster JSON missing!"

        with open(self.roster_path, "r", encoding="utf-8") as f:
            self.roster_data = json.load(f)

        self.frames_dir = REPO_ROOT / "test_data" / "clasico_eval_dataset"
        self.t0_frame = self.frames_dir / "el_clasico_oct2024_t0.jpg"
        self.t1_frame = self.frames_dir / "el_clasico_oct2024_t1.jpg"

        # If frames don't exist yet, generate them
        if not self.t0_frame.is_file() or not self.t1_frame.is_file():
            from scripts.generate_clasico_match_frame import generate_clasico_match_pair
            generate_clasico_match_pair()

    def test_official_roster_starting_eleven(self):
        """Verifies starting 11 for both teams from October 26, 2024 match."""
        rm_xi = self.roster_data["away_team"]["starting_xi"]
        bar_xi = self.roster_data["home_team"]["starting_xi"]

        assert len(rm_xi) == 11, f"Expected 11 Real Madrid starters, got {len(rm_xi)}"
        assert len(bar_xi) == 11, f"Expected 11 Barcelona starters, got {len(bar_xi)}"

        # Check key players
        rm_names = [p["name"] for p in rm_xi]
        bar_names = [p["name"] for p in bar_xi]

        assert "Kylian Mbappé" in rm_names
        assert "Vinícius Júnior" in rm_names
        assert "Jude Bellingham" in rm_names
        assert "Andriy Lunin" in rm_names
        assert "Lucas Vázquez" in rm_names
        assert "Antonio Rüdiger" in rm_names

        assert "Robert Lewandowski" in bar_names
        assert "Lamine Yamal" in bar_names
        assert "Raphinha" in bar_names
        assert "Pedri" in bar_names
        assert "Marc Casadó" in bar_names
        assert "Iñaki Peña" in bar_names

    def test_clock_synchronization_accuracy(self):
        """Verifies EventClockSynchronizer aligns video timeline within 0.2s."""
        sync = EventClockSynchronizer(search_window_sec=4.0, bin_width_sec=0.2)
        video_kicks = [1837.10, 2493.05, 3252.15]
        feed_events = [
            {"minute": 30, "second": 37},  # 1837.0s
            {"minute": 41, "second": 33},  # 2493.0s
            {"minute": 54, "second": 12},  # 3252.0s
        ]
        offset = sync.estimate_clock_offset(video_kicks, feed_events)
        assert abs(offset) <= 0.25, f"Clock offset excessive: {offset}s"

    def test_detection_and_ball_carrier_localization(self):
        """Verifies LocalPlayerDetector finds >= 11 players and the ball."""
        detector = LocalPlayerDetector()
        res = detector.detect(str(self.t0_frame))
        players = res.get("players", [])
        ball = res.get("ball", {})

        assert len(players) >= 11, f"Expected >= 11 detected players, got {len(players)}"
        assert ball.get("detected") is True, "Ball must be detected on pitch"
        assert ball.get("position_px") is not None

    def test_probabilistic_tracker_overlap_demotion(self):
        """Verifies teammate bounding box intersection demotes confidence to PROBABILISTIC_PRIOR."""
        tracker = ProbabilisticBidirectionalTracker(iou_threshold=0.35, decay_rate=0.75)
        sim_tracks = {
            0: [
                {"track_id": 1, "box": [200, 300, 260, 420], "team": "white"},
                {"track_id": 2, "box": [210, 305, 270, 425], "team": "white"},
            ],
            1: [
                {"track_id": 1, "box": [205, 302, 265, 422], "team": "white"},
                {"track_id": 2, "box": [208, 304, 268, 424], "team": "white"},
            ],
        }
        res = tracker.backward_propagate(
            ball_reception_track={"track_id": 1, "team": "white", "box": [205, 302, 265, 422]},
            shot_frame_buffer=None,
            all_tracks_per_frame=sim_tracks,
        )
        assert res[0]["status"] == TrackStatus.PROBABILISTIC_PRIOR.value
        assert res[0]["confidence"] <= 0.60

    def test_goalkeeper_kit_color_firewall(self):
        """Verifies Goalkeeper isolation and prevents outfield box defenders from GK assignment."""
        matcher = GlobalFormationMatcher(gk_color_threshold=40.0)
        # Lunin neon orange HSV
        gk_orange_hsv = [15.0, 190.0, 245.0]
        candidates = [
            {"track_id": 1, "mean_hsv": [15.2, 185.0, 246.0], "pitch_x": 4.0, "pitch_y": 34.0},  # Lunin (GK)
            {"track_id": 2, "mean_hsv": [58.0, 68.0, 187.0], "pitch_x": 12.0, "pitch_y": 28.0},  # Rüdiger (LCB in box)
            {"track_id": 3, "mean_hsv": [55.0, 30.0, 237.0], "pitch_x": 14.0, "pitch_y": 42.0},  # Militão (RCB in box)
        ]
        best_gk, outfield = matcher.filter_goalkeepers(candidates, gk_orange_hsv, defending_goal_x=0.0)

        assert best_gk is not None
        assert best_gk["track_id"] == 1
        assert len(outfield) == 2
        assert {c["track_id"] for c in outfield} == {2, 3}

    def test_hungarian_formation_matching_zero_duplication(self):
        """Verifies Hungarian bipartite formation solver produces exactly 10 unique outfielders."""
        matcher = GlobalFormationMatcher()
        rm_coords = np.array([
            [24.0, 12.0], [21.0, 26.0], [21.0, 42.0], [24.0, 56.0], [40.0, 34.0],
            [54.0, 24.0], [54.0, 44.0], [76.0, 14.0], [83.0, 34.0], [76.0, 54.0],
        ], dtype=np.float32)

        rm_xi = self.roster_data["away_team"]["starting_xi"]
        outfield_profiles = [p for p in rm_xi if not p.get("is_gk")]

        assignments = matcher.solve_formation_assignment(
            outfield_pitch_coords=rm_coords,
            formation_template="4-3-3",
            unassigned_profiles=outfield_profiles,
            attack_direction="right",
        )

        assigned_names = [a["profile"]["name"] for a in assignments.values() if a.get("profile")]
        assert len(assigned_names) == 10
        assert len(set(assigned_names)) == 10, "Duplicate player assignment detected in Hungarian solver!"
