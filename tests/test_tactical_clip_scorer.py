"""
tests/test_tactical_clip_scorer.py
Unit verification tests for TacticalBuildup & Climax Hook Scorer.
"""

import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.ingestion.clip_extractor import ClipExtractor
from src.vision.pitch_homography_role_mapper import (
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    PitchHomographyProjector,
)
from src.vision.tactical_clip_scorer import (
    CELL_HEIGHT_M,
    CELL_WIDTH_M,
    KARUN_SINGH_XT_12x8,
    XT_GRID_COLS,
    XT_GRID_ROWS,
    AudioPeak,
    BuildupCandidate,
    TacticalClipScorer,
    compute_audio_excitement,
    compute_bounded_voronoi_areas,
    detect_audio_excitement_peaks,
    find_continuous_tactical_buildup,
    get_xt_value,
    normalize_feature,
    rank_buildup_candidates,
)


class TestTacticalClipScorer(unittest.TestCase):
    """Test suite for TacticalClipScorer and its mathematical sub-components."""

    def test_karun_singh_xt_grid_properties(self):
        """Verifies 12x8 xT grid dimensions, symmetry, and metric lookup."""
        self.assertEqual(KARUN_SINGH_XT_12x8.shape, (8, 12))
        self.assertEqual(XT_GRID_COLS, 12)
        self.assertEqual(XT_GRID_ROWS, 8)
        self.assertAlmostEqual(CELL_WIDTH_M, 105.0 / 12.0, places=4)
        self.assertAlmostEqual(CELL_HEIGHT_M, 68.0 / 8.0, places=4)

        # Check vertical pitch symmetry (rows 0..3 mirror rows 7..4)
        for r in range(4):
            np.testing.assert_allclose(
                KARUN_SINGH_XT_12x8[r],
                KARUN_SINGH_XT_12x8[7 - r],
                rtol=1e-5,
                err_msg=f"Row {r} should match mirrored row {7 - r}",
            )

        # Check low threat at own goal line vs peak threat in opponent penalty box
        xt_own_goal = get_xt_value(5.0, 34.0, attack_direction="right")
        xt_box = get_xt_value(95.0, 34.0, attack_direction="right")
        self.assertLess(xt_own_goal, 0.02)
        self.assertGreater(xt_box, 0.10)
        self.assertGreater(xt_box, xt_own_goal)

        # Check inverted attack direction ('left')
        xt_left_att_high = get_xt_value(5.0, 34.0, attack_direction="left")
        xt_left_att_low = get_xt_value(95.0, 34.0, attack_direction="left")
        self.assertGreater(xt_left_att_high, 0.10)
        self.assertLess(xt_left_att_low, 0.02)

    def test_multi_modal_audio_excitement_peak_detection(self):
        """
        Synthesizes a 16kHz audio signal with background noise and a 1.5s
        climax burst (broadband RMS crowd roar + 1000Hz commentator excitement).
        Verifies local maxima peak detection where excitement exceeds mean + 1.6 * std.
        """
        sr = 16000
        duration = 12.0
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Baseline noise (std = 0.03)
        np.random.seed(42)
        audio = np.random.normal(0.0, 0.03, n_samples).astype(np.float32)

        # Inject excitement burst centered at 6.0s [5.2s .. 6.8s]
        burst_mask = (t >= 5.2) & (t <= 6.8)
        # Broadband crowd roar burst (high RMS)
        audio += (burst_mask * np.random.normal(0.0, 0.35, n_samples)).astype(np.float32)
        # Commentator voice pitch in 450Hz - 2400Hz band (1000Hz sine wave)
        audio += (burst_mask * 0.40 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)

        times, rms_arr, band_arr, combined = compute_audio_excitement(
            audio, sr=sr, window_sec=1.5, hop_sec=0.25
        )
        self.assertGreater(len(times), 0)
        self.assertEqual(len(times), len(combined))

        peaks = detect_audio_excitement_peaks(
            audio,
            sr=sr,
            window_sec=1.5,
            hop_sec=0.25,
            threshold_factor=1.6,
            min_peak_distance_sec=5.0,
        )

        self.assertGreaterEqual(len(peaks), 1, "Must detect at least 1 excitement peak")
        best_peak = max(peaks, key=lambda p: p.audio_peak)
        # Peak should be identified around 6.0s (center of burst)
        self.assertAlmostEqual(best_peak.t_peak, 6.0, delta=0.85)

        # Verify exact mathematical threshold: combined >= mean + 1.6 * std
        comb_mean = float(np.mean(combined))
        comb_std = float(np.std(combined))
        expected_threshold = comb_mean + 1.6 * comb_std
        self.assertGreaterEqual(best_peak.audio_peak, expected_threshold)

    def test_buildup_lookback_and_climax_hook_slicing(self):
        """Verifies buildup scan window [t_peak - 22s, t_peak + 3s] and climax hook window [t_peak - 1.5s, t_peak + 2.0s]."""
        t_peak = 35.0
        # Simulated run on synthetic source
        res = find_continuous_tactical_buildup(
            media_path="synthetic_match.mp4",
            t_peak=t_peak,
            lookback_sec=22.0,
            post_peak_sec=3.0,
        )

        self.assertEqual(res["t_peak"], 35.0)
        self.assertLessEqual(res["buildup_start_sec"], 35.0)
        self.assertGreaterEqual(res["buildup_start_sec"], 35.0 - 22.0)
        self.assertLessEqual(res["buildup_end_sec"], 35.0 + 3.0)

        hook = res["climax_hook_window"]
        self.assertAlmostEqual(hook["start_sec"], 35.0 - 1.5, places=2)
        self.assertAlmostEqual(hook["end_sec"], 35.0 + 2.0, places=2)
        self.assertAlmostEqual(hook["duration_sec"], 3.5, places=2)
        self.assertEqual(hook["t_peak"], 35.0)

    def test_voronoi_bounded_open_space_areas(self):
        """Tests Voronoi open space computation for attacking receivers."""
        # 4 attacking players well spread out in attacking third
        attackers = np.array([
            [68.0, 20.0],
            [80.0, 34.0],
            [72.0, 50.0],
            [92.0, 34.0],
        ])
        areas = compute_bounded_voronoi_areas(
            attackers, x_min=63.0, x_max=105.0, y_min=0.0, y_max=68.0
        )
        self.assertEqual(len(areas), 4)
        for a in areas:
            self.assertGreater(a, 30.0)
            self.assertLessEqual(a, 600.0)

    def test_candidate_scoring_and_ranking_formula(self):
        """
        Verifies exact formula:
        TotalScore = 0.35 * norm(delta_xT) + 0.30 * norm(defenders_bypassed) +
                     0.20 * norm(audio_peak) + 0.15 * norm(max_unmarked_space_m2)
        """
        # Candidate 1: High xT progression + many defenders bypassed (Great buildup)
        c1 = BuildupCandidate(
            t_peak=45.0,
            buildup_start_sec=25.0,
            buildup_end_sec=46.0,
            buildup_duration_sec=21.0,
            climax_hook_window={"start_sec": 43.5, "end_sec": 47.0, "duration_sec": 3.5, "t_peak": 45.0},
            delta_xT=0.18,
            defenders_bypassed=5,
            audio_peak=3.8,
            max_unmarked_space_m2=220.0,
        )

        # Candidate 2: Low progression, fewer bypassed (Unexciting possession)
        c2 = BuildupCandidate(
            t_peak=15.0,
            buildup_start_sec=5.0,
            buildup_end_sec=16.0,
            buildup_duration_sec=11.0,
            climax_hook_window={"start_sec": 13.5, "end_sec": 17.0, "duration_sec": 3.5, "t_peak": 15.0},
            delta_xT=0.02,
            defenders_bypassed=1,
            audio_peak=1.9,
            max_unmarked_space_m2=60.0,
        )

        ranked = rank_buildup_candidates([c2, c1])
        self.assertEqual(len(ranked), 2)
        # c1 must rank #1
        self.assertEqual(ranked[0].t_peak, 45.0)
        self.assertEqual(ranked[0].rank, 1)
        self.assertEqual(ranked[1].rank, 2)
        self.assertGreater(ranked[0].total_score, ranked[1].total_score)

        # In a 2-candidate min-max setup, candidate 1 has top values in all 4 dimensions,
        # so norm = 1.0 for all, TotalScore should equal 0.35 + 0.30 + 0.20 + 0.15 = 1.0000
        self.assertAlmostEqual(ranked[0].total_score, 1.0, places=3)
        self.assertAlmostEqual(ranked[1].total_score, 0.0, places=3)

    def test_clip_extractor_buildup_integration(self):
        """Verifies ClipExtractor initializes TacticalClipScorer and provides score_and_extract_buildup_clips."""
        from src.ingestion.time_indexer import TimeIndexer
        indexer = TimeIndexer()
        extractor = ClipExtractor(
            time_indexer=indexer,
            media_source="synthetic",
            output_dir="data/test_clips",
            frames_dir="data/test_frames",
        )

        scorer = extractor.get_tactical_scorer()
        self.assertIsInstance(scorer, TacticalClipScorer)

        # Test score_and_extract_buildup_clips interface without cutting disk videos
        extracted = extractor.score_and_extract_buildup_clips(
            media_source="synthetic",
            top_k=2,
            candidate_peaks=[25.0, 50.0],
            cut_video_clips=False,
        )
        self.assertEqual(len(extracted), 2)
        for clip in extracted:
            self.assertIn("total_score", clip)
            self.assertIn("climax_hook_window", clip)
            self.assertIn("delta_xT", clip)
            self.assertIn("defenders_bypassed", clip)
            self.assertIn("audio_peak", clip)
            self.assertIn("max_unmarked_space_m2", clip)
            self.assertIn("buildup_start_sec", clip)
            self.assertIn("buildup_end_sec", clip)

        # Verify extract_event_assets backwards compatibility and climax_hook_window presence
        assets = extractor.extract_event_assets(match_minute=10)
        self.assertIn("clip_path", assets)
        self.assertIn("freeze_frame_path", assets)
        self.assertIn("climax_hook_window", assets)
        self.assertEqual(assets["climax_hook_window"]["duration_sec"], 3.5)


if __name__ == "__main__":
    unittest.main()
