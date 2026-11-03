"""
Unit Tests for Tracking Subsystem:
1. Shot Boundary Detector (Hist-diff + Turf %)
2. BoT-SORT Tracker (with_reid=False, sparseOptFlow CMC, Stride=2 interpolation)
3. Team Color Clusterer (K-Means k=2 outfield + Outlier GK/Ref isolation)
4. Smart Collage Batcher (3-angle strips, context frame, 2D radar, payload building)
"""

import numpy as np
import pytest

from src.detection_engine.tracking.shot_detector import ShotBoundaryDetector
from src.detection_engine.tracking.bot_sort_tracker import BoTSORTTracker, TrackletLifecycle
from src.detection_engine.tracking.team_color_clusterer import TeamColorClusterer, TeamClusterResult
from src.detection_engine.tracking.smart_collage_batcher import SmartCollageBatcher


class TestShotDetector:
    def test_shot_boundary_continuity_and_cut(self):
        detector = ShotBoundaryDetector(hist_thresh=0.40, turf_diff_thresh=0.25, min_shot_duration_frames=5)

        # Green turf frame
        green_frame = np.full((180, 320, 3), (35, 120, 45), dtype=np.uint8)
        # Non-green (crowd/close-up) frame
        blue_frame = np.full((180, 320, 3), (180, 50, 40), dtype=np.uint8)

        # Frame 0: Init
        is_cut, _, turf_pct = detector.process_frame(green_frame, 0)
        assert not is_cut
        assert turf_pct > 0.50

        # Frame 1 to 5: identical green frame (no cut)
        for i in range(1, 6):
            is_cut, dist, turf_pct = detector.process_frame(green_frame, i)
            assert not is_cut
            assert dist < 0.10

        # Frame 6: sudden switch to blue (camera cut)
        is_cut, dist, turf_pct = detector.process_frame(blue_frame, 6)
        assert is_cut
        assert dist > 0.40


class TestBoTSORTTrackerConstraints:
    def test_bot_sort_configuration(self):
        tracker = BoTSORTTracker(stride=2)

        # Verify Mandatory Constraint 1: with_reid=False
        assert tracker.tracker.encoder is None, "BoT-SORT must have encoder=None (with_reid=False) to avoid CPU choke"
        assert tracker.tracker.gmc.method == "sparseOptFlow", "BoT-SORT must use sparseOptFlow for Camera Motion Compensation"
        assert tracker.stride == 2

    def test_lifecycle_3_angle_crops(self):
        lc = TrackletLifecycle(
            track_id=1,
            start_frame=0,
            end_frame=30,
            class_id=0,
            frame_indices=list(range(0, 31, 2)),
            bboxes=[[100, 100, 160, 220]] * 16,
            confidences=[0.9] * 16,
            is_interpolated=[False] * 16,
        )

        # Create dummy frame cache
        frame_cache = {
            i: np.full((720, 1280, 3), 120, dtype=np.uint8)
            for i in range(0, 31, 2)
        }
        # Add texture so Laplacian has variance
        for i in frame_cache:
            frame_cache[i][100:220, 100:160] = np.random.randint(0, 255, (120, 60, 3), dtype=np.uint8)

        crops = lc.extract_3_angle_crops(frame_cache, min_box_dim=15)
        assert len(crops) == 3, "Tracklet must yield exactly 3 multi-angle crops (Early, Peak, Late)"
        assert crops[0].label == "early"
        assert crops[1].label == "peak"
        assert crops[2].label == "late"


class TestTeamColorClustererOutlierLogic:
    def test_kmeans_k2_with_outlier_gk_ref(self):
        clusterer = TeamColorClusterer(outlier_std_factor=1.2)

        # Create mock 1080p frame
        frame = np.full((1080, 1920, 3), (35, 100, 40), dtype=np.uint8)  # Grass background

        # Place 10 White outfield players (Home)
        for i in range(10):
            x = 200 + i * 50
            frame[300:400, x:x+40] = (245, 245, 245)

        # Place 10 Dark/Red outfield players (Away)
        for i in range(10):
            x = 800 + i * 50
            frame[300:400, x:x+40] = (40, 40, 180)

        # Place 1 Orange Goalkeeper near left goal line (x = 80)
        frame[300:400, 80:120] = (0, 140, 255)

        # Place 1 Dark/Black Referee near center (x = 700)
        frame[300:400, 700:740] = (15, 15, 15)

        actors = []
        for i in range(10):
            x = 200 + i * 50
            actors.append((i + 1, [x, 300, x + 40, 400]))
        for i in range(10):
            x = 800 + i * 50
            actors.append((i + 11, [x, 300, x + 40, 400]))
        # GK (ID 99)
        actors.append((99, [80, 300, 120, 400]))
        # Ref (ID 88)
        actors.append((88, [700, 300, 740, 400]))

        res = clusterer.cluster_actors(frame, actors, frame_width=1920)

        # Outfield home players must be "home"
        assert res[1].team == "home"
        assert not res[1].is_outlier

        # Outfield away players must be "away"
        assert res[11].team == "away"
        assert not res[11].is_outlier

        # GK must be identified as outlier and labeled as GK
        assert res[99].is_outlier
        assert "gk" in res[99].team

        # Ref must be identified as outlier and labeled as referee
        assert res[88].is_outlier
        assert res[88].team == "referee"


class TestSmartCollageBatcher:
    def test_batch_payload_creation(self):
        batcher = SmartCollageBatcher(min_tracklet_duration_sec=0.1)

        # Mock frame cache
        frame = np.full((720, 1280, 3), 80, dtype=np.uint8)
        for y in range(200, 350):
            for x in range(300, 450):
                frame[y, x] = (x % 255, y % 255, (x + y) % 255)
        frame_cache = {0: frame, 10: frame, 20: frame}

        # Mock lifecycles
        lc1 = TrackletLifecycle(
            track_id=1,
            start_frame=0,
            end_frame=20,
            frame_indices=[0, 10, 20],
            bboxes=[[300, 200, 350, 320], [305, 200, 355, 320], [310, 200, 360, 320]],
            confidences=[0.9, 0.9, 0.9],
            is_interpolated=[False, False, False],
        )
        lc2 = TrackletLifecycle(
            track_id=2,
            start_frame=0,
            end_frame=20,
            frame_indices=[0, 10, 20],
            bboxes=[[400, 200, 450, 320], [405, 200, 455, 320], [410, 200, 460, 320]],
            confidences=[0.88, 0.88, 0.88],
            is_interpolated=[False, False, False],
        )

        lifecycles = {1: lc1, 2: lc2}
        team_assignments = {
            1: TeamClusterResult(1, "home", 0.95, False),
            2: TeamClusterResult(2, "away", 0.92, False),
        }
        active_roster = {
            "home_team": "Real Madrid",
            "away_team": "Barcelona",
            "home_active_xi": [{"name": "Vinícius Júnior", "number": 7}],
            "away_active_xi": [{"name": "Robert Lewandowski", "number": 9}],
        }

        payload = batcher.build_batch_payload(
            shot_id=1,
            lifecycles=lifecycles,
            team_assignments=team_assignments,
            frame_cache=frame_cache,
            active_roster=active_roster,
            fps=30.0,
        )

        assert payload is not None
        assert payload.total_tracklets == 2
        assert payload.home_visible == 1
        assert payload.away_visible == 1
        assert payload.montage_img.shape[0] > 0
        assert payload.context_frame_img.shape[0] > 0
        assert payload.radar_minimap_img.shape[0] > 0
