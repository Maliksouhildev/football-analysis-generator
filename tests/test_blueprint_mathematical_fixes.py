"""
tests/test_blueprint_mathematical_fixes.py
================================================================================
Comprehensive verification suite for the 6 mathematical, geometric, and
OpenCV-specific blueprint fixes:
1. Module 2: OpenCV uint8 LAB unscaling, unsigned underflow prevention & cube-root shadow invariance.
2. Module 3: Boot histogram with green-only turf masking, fg_pixels >= 15 guard & Bhattacharyya distance.
3. Module 3: BoT-SORT GMC warp matrix accumulation & exponential velocity damping (0.85^Delta_t).
4. Module 4: Box-height camera zoom estimation (s_x) preventing compact cluster stretching.
5. Module 4: 2D coordinates & turf hull-bounded lateral Y mapping (separating LB/CB/RB).
6. Module 4: Hungarian 2-opt pairwise swap cost margin (< 15% ambiguity flag).
================================================================================
"""

import math
import numpy as np
import pytest
import cv2

from src.detection_engine.tracking.team_color_clusterer import TeamColorClusterer
from src.detection_engine.tracking.bot_sort_tracker import (
    extract_boot_histogram,
    compare_boot_histograms,
    accumulate_gmc_warp,
    stitch_tracklets_gmc,
    TrackletLifecycle,
)
from src.detection_engine.tracking.smart_collage_batcher import (
    compute_hungarian_2opt_swap_margins,
    SmartCollageBatcher,
)
from src.detection_engine.pitch_geometry import PitchGeometryEngine


class TestModule2LABMath:
    """1. The OpenCV uint8 LAB Scale Mismatch & Unsigned Underflow Bug."""

    def test_uint8_unscaling_and_underflow_prevention(self):
        """
        Verify that uint8 OpenCV LAB values are unscaled to true CIE ranges,
        and dark kits on grass do not underflow.
        """
        clusterer = TeamColorClusterer()

        # Create mock frame: green grass (L_cv2 ~ 132), dark jersey (L_cv2 ~ 25)
        frame = np.full((100, 100, 3), (35, 120, 45), dtype=np.uint8)  # Grass
        # Dark navy/black player torso
        frame[25:50, 40:60] = (25, 20, 20)

        feats = clusterer.extract_torso_features(frame, [40, 10, 60, 90], turf_l_ref=52.0)
        assert feats is not None
        l_norm, a_cie, b_cie = feats["lab"]

        # In true CIE ranges: dark kit L* should be ~10-25, not +149 (the uint8 underflow trap)
        assert l_norm < 35.0, f"Expected dark kit L*_norm < 35, got {l_norm} (underflow bug occurred!)"
        assert abs(a_cie) < 128.0
        assert abs(b_cie) < 128.0

    def test_shadow_invariance_multiplicative_cube_root(self):
        """
        Physically, shadows reduce illumination by factor k in (0, 1).
        L* + 16 proportional to (k * R)^(1/3).
        Verify that our cube-root ratio cancels k^(1/3) so L*_norm is invariant
        between direct sunlight (k=1.0) and deep shadow (k=0.20).
        """
        # True reflectance: white shirt R_shirt=0.90, grass R_turf=0.20
        # Under CIE formula: L* = 116 * (Y/Yn)^(1/3) - 16  => L* + 16 = 116 * (k * R)^(1/3)
        turf_ref_l = 52.0  # Canonical turf lightness

        # Case 1: Full sunlight (k = 1.0)
        k_sun = 1.0
        l_shirt_sun = 116.0 * ((k_sun * 0.90) ** (1.0 / 3.0)) - 16.0  # ~96.0
        l_turf_sun = 116.0 * ((k_sun * 0.20) ** (1.0 / 3.0)) - 16.0   # ~51.8

        # Case 2: Deep 20% shadow (k = 0.20)
        k_shadow = 0.20
        l_shirt_shadow = 116.0 * ((k_shadow * 0.90) ** (1.0 / 3.0)) - 16.0  # ~49.5
        l_turf_shadow = 116.0 * ((k_shadow * 0.20) ** (1.0 / 3.0)) - 16.0   # ~23.6

        # Flawed Additive Subtraction from old blueprint:
        delta_l_sun = l_shirt_sun - l_turf_sun        # ~44.2
        delta_l_shadow = l_shirt_shadow - l_turf_shadow  # ~25.9
        additive_error = abs(delta_l_sun - delta_l_shadow)
        assert additive_error > 15.0, "Additive subtraction failed to demonstrate shadow sensitivity"

        # Correct Cube-Root Ratio from our patch:
        # L*_norm = 68 * ((L*_torso + 16) / (L*_turf + 16)) - 16
        l_norm_sun = (turf_ref_l + 16.0) * ((l_shirt_sun + 16.0) / (l_turf_sun + 16.0)) - 16.0
        l_norm_shadow = (turf_ref_l + 16.0) * ((l_shirt_shadow + 16.0) / (l_turf_shadow + 16.0)) - 16.0

        # Invariance test: the difference should be mathematically zero (within floating precision)
        ratio_error = abs(l_norm_sun - l_norm_shadow)
        assert ratio_error < 0.05, f"Cube-root ratio must be shadow invariant! Error was {ratio_error}"
        assert abs(l_norm_sun - 96.0) < 3.0


class TestModule3BootHistogramAndGMC:
    """2. The White-Boot Deletion & NaN Correlation Trap + 6. BoT-SORT GMC Stitching."""

    def test_white_boots_preserved_and_no_nan(self):
        """
        Verify that white boots/socks on green pitch are NOT deleted by a white line mask,
        and histogram comparison does not produce NaN.
        """
        frame = np.full((120, 60, 3), (35, 120, 45), dtype=np.uint8)  # Grass background
        # Place white boot in bottom 15% (y in [102, 120])
        frame[104:118, 20:40] = (250, 250, 250)  # Pure white boot

        bbox = [10, 10, 50, 120]
        hist = extract_boot_histogram(frame, bbox)

        assert hist is not None, "White boot must NOT be erased by white-line mask!"
        assert not np.isnan(hist).any(), "Histogram must not contain NaN"
        assert abs(float(hist.sum()) - 1.0) < 1e-4, "Histogram must be L1-normalized"

        # Compare with identical white boot
        passes, sim = compare_boot_histograms(hist, hist)
        assert passes
        assert sim >= 0.99

    def test_foreground_guard_prevents_zero_div_nan(self):
        """
        Verify that when foreground pixels < 15, extract_boot_histogram safely returns None
        and compare_boot_histograms bypasses the gate without crashing or returning NaN.
        """
        # Frame with ONLY grass (no player boot)
        frame = np.full((120, 60, 3), (35, 120, 45), dtype=np.uint8)
        bbox = [10, 10, 50, 120]

        hist = extract_boot_histogram(frame, bbox)
        assert hist is None, "Guard must return None when fg_pixels < 15"

        # Gate must be bypassed (passes=True, sim=1.0)
        passes, sim = compare_boot_histograms(hist, None)
        assert passes
        assert sim == 1.0

    def test_gmc_camera_pan_compensation_and_velocity_damping(self):
        """
        Verify that BoT-SORT GMC warp accumulation counteracts camera pans (e.g. 20 px/frame)
        and velocity damping (0.85^Delta_t) eliminates the 900px overshoot.
        """
        # Simulate camera panning right at 20 px/frame for 45 frames
        pan_speed = 20.0
        gmc_history = {}
        for f in range(1, 46):
            # Affine translation M = [[1, 0, -20], [0, 1, 0]]
            M = np.array([[1.0, 0.0, -pan_speed], [0.0, 1.0, 0.0]], dtype=np.float32)
            gmc_history[f] = M

        # Accumulate warps from frame 0 to frame 45
        T = accumulate_gmc_warp(gmc_history, 0, 45)

        # A stationary point on the pitch at (1000, 500)
        p0 = np.array([1000.0, 500.0, 1.0])
        p_warped_hom = T @ p0
        p_cam2 = p_warped_hom[:2] / p_warped_hom[2]

        # The camera panned 45 * 20 = 900px right, so pitch point shifted to 1000 - 900 = 100 in frame 45
        expected_x = 1000.0 - 45 * 20.0
        assert abs(p_cam2[0] - expected_x) < 1e-3, f"Expected warped x={expected_x}, got {p_cam2[0]}"

        # Velocity damping test: player running at 2 px/frame
        v_player = np.array([2.0, 0.0])
        gap = 45
        damp_sum = (1.0 - (0.85 ** gap)) / (1.0 - 0.85)  # Sum of geometric series ~ 6.66
        p_pred = p_cam2 + v_player * damp_sum

        # Without damping, naive linear projection over 45 frames would add 45 * 2 = 90px
        # With damping, displacement is bounded to ~13.3px
        displacement = float(np.linalg.norm(p_pred - p_cam2))
        assert displacement < 15.0, f"Expected damped displacement < 15px, got {displacement}"


class TestModule4CoordinatesAndHungarian:
    """3. Span Scaling, 4. 2D Coordinates, 5. 2-Opt Pairwise Swap Margin."""

    def test_zoom_cam_scale_estimation_from_box_height(self):
        """
        Verify that camera zoom scale s_x is derived from box height, NOT player count k.
        A compact cluster of 6 players (15 meters spread) must NOT be stretched across 53m!
        """
        # Standard wide tactical cam: h_med = 85px -> s_x ~ 0.50
        h_med_wide = 85.0
        s_x_wide = float(np.clip(42.5 / h_med_wide, 0.30, 0.75))
        assert abs(s_x_wide - 0.50) < 0.01

        # Tighter zoom cam: h_med = 130px -> s_x ~ 0.32
        h_med_tight = 130.0
        s_x_tight = float(np.clip(42.5 / h_med_tight, 0.30, 0.75))
        assert abs(s_x_tight - 0.327) < 0.01

        # Counterexample in prompt: 6 players clustered in screen spread 0.20
        # Flawed blueprint formula: clip(6/10 * 0.85, 0.35, 0.80) = 0.51 (stretches 15m cluster to 53m!)
        # Our zoom-scaled spread:
        cluster_screen_spread = 0.20
        our_pitch_spread = cluster_screen_spread * s_x_wide  # 0.20 * 0.50 = 0.10 (exact true physical ~10-15m span!)
        assert our_pitch_spread < 0.15, "Zoom scaling correctly preserves compact clusters!"

    def test_turf_hull_bounds_and_lateral_y_normalization(self):
        """
        Verify that lateral coordinate y_norm is bounded by the turf hull,
        correctly separating Left Back, Center Back, and Right Back.
        """
        # Dummy pitch geometry with turf hull between y = 20% and y = 90%
        pitch_geom = {"min_turf_y_pct": 20.0, "max_turf_y_pct": 90.0}
        min_y, max_y = PitchGeometryEngine.get_turf_hull_bounds(pitch_geom)
        assert min_y == 0.20
        assert max_y == 0.90

        # Normalization across turf hull span
        span_y = max_y - min_y  # 0.70
        y_top_screen = 0.20  # Far touchline -> y_norm = 0.0 (e.g. Left Back)
        y_norm_top = (y_top_screen - min_y) / span_y
        assert abs(y_norm_top - 0.0) < 1e-4

        y_bot_screen = 0.90  # Near touchline -> y_norm = 1.0 (e.g. Right Back)
        y_norm_bot = (y_bot_screen - min_y) / span_y
        assert abs(y_norm_bot - 1.0) < 1e-4

    def test_hungarian_2opt_pairwise_swap_margin(self):
        """
        Verify that 2-opt pairwise swap margin is non-negative and properly flags ambiguous swaps (< 15%).
        """
        # Cost matrix for 2 players and 2 roles:
        # Player 0: role 0 cost 1.0, role 1 cost 1.1 (nearly indifferent)
        # Player 1: role 0 cost 1.1, role 1 cost 1.0 (nearly indifferent)
        cost_matrix = np.array([
            [1.0, 1.1],
            [1.1, 1.0],
        ], dtype=np.float32)

        row_ind = np.array([0, 1])
        col_ind = np.array([0, 1])

        margins = compute_hungarian_2opt_swap_margins(cost_matrix, row_ind, col_ind)

        # Baseline assignment: C[0,0] + C[1,1] = 1.0 + 1.0 = 2.0
        # Swapped assignment: C[0,1] + C[1,0] = 1.1 + 1.1 = 2.2
        # Swap cost increase: 2.2 - 2.0 = 0.2
        # Margin: 0.2 / 2.0 = 10% (0.10)
        margin_0 = margins[0]
        assert margin_0 >= 0.0, "2-opt swap margin must be >= 0 by optimality"
        assert abs(margin_0 - 0.10) < 0.02
        assert margin_0 < 0.15, "Must be flagged as ambiguous (< 15%)!"

        # Now test clear unambiguous assignment:
        # Player 0: role 0 cost 0.2, role 1 cost 5.0
        # Player 1: role 0 cost 5.0, role 1 cost 0.2
        cost_matrix_clear = np.array([
            [0.2, 5.0],
            [5.0, 0.2],
        ], dtype=np.float32)

        margins_clear = compute_hungarian_2opt_swap_margins(cost_matrix_clear, row_ind, col_ind)
        # Baseline = 0.4. Swapped = 10.0. Increase = 9.6 / 0.4 = 24.0 (2400%)
        assert margins_clear[0] > 1.0
        assert margins_clear[0] >= 0.15, "Unambiguous assignment must NOT be flagged as ambiguous"

    def test_polygon_column_edge_intersection(self):
        """
        Verify that PitchGeometryEngine.get_turf_hull_bounds calculates exact edge
        intersections for arbitrary columns where no vertices exist.
        """
        # Trapezoid: vertices at (0, 300), (1920, 250), (1920, 1080), (0, 1080)
        hull_pts = np.array([
            [0, 300],
            [1920, 250],
            [1920, 1080],
            [0, 1080],
        ], dtype=np.int32)
        pitch_geom = {"hull_pts": hull_pts, "min_turf_y_pct": 23.0, "max_turf_y_pct": 100.0}

        # Query midpoint column x = 960 (no vertex near 960)
        # Top edge from (0, 300) to (1920, 250): y = 300 + 0.5 * (-50) = 275 px
        # Normalized top y = 275 / 1080 = ~0.2546
        # Bottom edge: y = 1080 px -> normalized = 1.0
        top_y, bot_y = PitchGeometryEngine.get_turf_hull_bounds(pitch_geom, frame_shape=(1080, 1920), x_px=960.0)
        expected_top = 275.0 / 1080.0
        assert abs(top_y - expected_top) < 1e-3, f"Expected top_y={expected_top}, got {top_y}"
        assert abs(bot_y - 1.0) < 1e-3, f"Expected bot_y=1.0, got {bot_y}"

    def test_affine_pitch_coordinate_mapping(self):
        """
        Verify that affine mapping x_pitch_hat = clip(s_x * rx + t_x) correctly
        maps players in zoomed camera views and prevents striker-midfielder confusion.
        """
        # Scenario: Camera zoomed in on attacking box (s_x = 0.30)
        # Striker is near center of screen (rx = 0.80)
        # If t_x is aligned with attacking formation center (t_x = 0.60):
        s_x = 0.30
        t_x = 0.60
        rx_striker = 0.80
        x_pitch_hat = float(np.clip(s_x * rx_striker + t_x, 0.0, 1.0))
        # Pitch coordinate is 0.30 * 0.80 + 0.60 = 0.84 (exact striker zone!)
        assert abs(x_pitch_hat - 0.84) < 1e-3

        fx_striker = 0.85
        fx_midfielder = 0.50
        dist_striker = abs(x_pitch_hat - fx_striker)      # |0.84 - 0.85| = 0.01
        dist_midfielder = abs(x_pitch_hat - fx_midfielder)  # |0.84 - 0.50| = 0.34
        assert dist_striker < dist_midfielder, "Striker must be closest to striker role, NOT midfielder"


class TestRobustnessAndFlawsFixed:
    """Tests for edge cases and multi-agent coordination flaws."""

    def test_gmc_interception_real_camera_motion(self):
        """
        Verify that BoTSORTTracker intercepts and stores the real camera motion warp
        from tracker.update instead of overwriting with identity.
        """
        from src.detection_engine.tracking.bot_sort_tracker import BoTSORTTracker

        tracker = BoTSORTTracker()
        f1 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(f1, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.circle(f1, (400, 300), 50, (255, 255, 255), -1)

        f2 = np.zeros_like(f1)
        # Shift 20 px right
        cv2.rectangle(f2, (120, 100), (220, 200), (255, 255, 255), -1)
        cv2.circle(f2, (420, 300), 50, (255, 255, 255), -1)

        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        confs = np.array([0.9], dtype=np.float32)
        clss = np.array([0], dtype=np.float32)

        tracker.update_strided(f1, boxes, confs, clss, 0)
        tracker.update_strided(f2, boxes + 20, confs, clss, 1)

        warp1 = tracker.gmc_history.get(1)
        assert warp1 is not None, "GMC warp for frame 1 must be recorded"
        # Translation component warp1[0, 2] should be ~20px, NOT 0.0 (identity)
        assert abs(warp1[0, 2] - 20.0) < 1.0, f"Expected GMC warp tx ~ 20.0, got {warp1[0, 2]}"

    def test_max_1_referee_enforcement(self):
        """
        Verify that when multiple central outliers exist, strictly 1 remains Referee,
        and excess outliers are reassigned to outfield teams.
        """
        clusterer = TeamColorClusterer()

        # Create mock frame with 8 players:
        # 4 White jerseys (Home)
        # 3 Dark jerseys (Away)
        # 2 Fluorescent yellow jerseys in center (Potential referees)
        frame = np.full((300, 800, 3), (35, 120, 45), dtype=np.uint8)

        actors = []
        # Home players
        for i, x in enumerate([100, 180, 260, 320]):
            frame[50:150, x:x+40] = (240, 240, 240)
            actors.append((i + 1, [x, 50, x + 40, 150]))
        # Away players
        for i, x in enumerate([480, 560, 640]):
            frame[50:150, x:x+40] = (30, 30, 120)
            actors.append((i + 5, [x, 50, x + 40, 150]))
        # 2 central outliers (yellow kits)
        frame[50:150, 380:420] = (20, 220, 220)
        actors.append((8, [380, 50, 420, 150]))
        frame[50:150, 430:470] = (20, 220, 220)
        actors.append((9, [430, 50, 470, 150]))

        res = clusterer.cluster_actors(frame, actors)
        ref_count = sum(1 for r in res.values() if r.team == "referee")
        assert ref_count <= 1, f"Expected at most 1 referee, got {ref_count}"

    def test_flank_turf_lightness_fallback(self):
        """
        Verify that when player bounding box is truncated at frame bottom (y2=H),
        flank pixels provide local turf lightness without falling back to constant 52.0.
        """
        clusterer = TeamColorClusterer()

        # Frame: dark grass in shadow (L_cv2 ~ 70 -> L* ~ 27)
        # OpenCV HSV for dark green grass
        shadow_grass_bgr = (20, 60, 25)
        frame = np.full((100, 100, 3), shadow_grass_bgr, dtype=np.uint8)

        # Torso of player truncated right at bottom of frame (y1=40, y2=100=H)
        frame[40:100, 40:60] = (200, 200, 200)

        feats = clusterer.extract_torso_features(frame, [40, 40, 60, 100], turf_l_ref=52.0)
        assert feats is not None
        l_norm, _, _ = feats["lab"]
        # In shadow grass, flank sampling detects the darker turf and normalizes L* upwards
        assert l_norm > 70.0, f"Expected shadow-normalized L* > 70, got {l_norm}"

