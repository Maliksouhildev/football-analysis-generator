"""
tests/test_tactical_visualizations.py
Comprehensive Verification Suite for:
1. Programmatic Tactical Visualizations via kloppy & mplsoccer
2. Full-Time Match Status Detection ('FT' Trigger from Sportmonks / API-Football)
3. Pitch Coordinate Standardization (Opta / Normalized to Metric FIFA 105m x 68m)
4. Kernel Density Estimation (KDE) Heatmaps with 100% Transparent Backgrounds
5. Studio-Grade Team Pass Networks with Weighted Edges and Squad Numbers
6. Integration into MatchStatsProvider Broadcast Telemetry Payloads
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

from src.rendering.tactical_visualization_engine import (
    TacticalVisualizationEngine,
    VisualizationResult,
    PITCH_LENGTH_METERS,
    PITCH_WIDTH_METERS,
)
from src.rendering.match_stats_provider import MatchStatsProvider


class TestTacticalVisualizations(unittest.TestCase):
    """Unit tests for kloppy coordinate standardization and mplsoccer rendering."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="tactical_viz_test_"))
        self.engine = TacticalVisualizationEngine(output_dir=self.temp_dir)
        self.stats_provider = MatchStatsProvider()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. Full-Time ('FT') Trigger Tests
    # -------------------------------------------------------------------------

    def test_full_time_status_recognition(self):
        """Ensures the engine triggers only upon official Full Time ('FT') status."""
        # String statuses
        self.assertTrue(self.engine.is_match_completed("FT"))
        self.assertTrue(self.engine.is_match_completed("FULL_TIME"))
        self.assertTrue(self.engine.is_match_completed("FINISHED"))
        self.assertTrue(self.engine.is_match_completed("AET"))
        self.assertTrue(self.engine.is_match_completed("FT_PEN"))

        # In-play or pre-match should NOT trigger
        self.assertFalse(self.engine.is_match_completed("HT"))
        self.assertFalse(self.engine.is_match_completed("INPLAY_1ST_HALF"))
        self.assertFalse(self.engine.is_match_completed("LIVE"))
        self.assertFalse(self.engine.is_match_completed("NS"))

        # Dictionary fixtures from Sportmonks / API-Football
        sportmonks_ft = {"state": {"short_name": "FT"}}
        api_football_ft = {"status": {"short": "FT"}}
        api_football_live = {"status": {"short": "2H"}}

        self.assertTrue(self.engine.is_match_completed(sportmonks_ft))
        self.assertTrue(self.engine.is_match_completed(api_football_ft))
        self.assertFalse(self.engine.is_match_completed(api_football_live))

    def test_process_full_time_pipeline_trigger(self):
        """Validates that automated post-match pipeline runs when FT and skips when LIVE."""
        key_players = [
            {
                "name": "Lamine Yamal",
                "number": 19,
                "points": [{"x": 80, "y": 70}, {"x": 85, "y": 75}, {"x": 78, "y": 68}],
            }
        ]

        # In-progress match should abort cleanly without generating assets
        in_progress_res = self.engine.process_full_time_visualizations(
            fixture_or_status="1H",
            key_players=key_players,
            team_name="Barcelona",
        )
        self.assertEqual(in_progress_res["status"], "in_progress")
        self.assertEqual(len(in_progress_res["generated_heatmaps"]), 0)

        # Full-time match triggers generation
        ft_res = self.engine.process_full_time_visualizations(
            fixture_or_status="FT",
            key_players=key_players,
            team_name="Barcelona",
        )
        self.assertEqual(ft_res["status"], "completed")
        self.assertTrue(ft_res["is_full_time"])
        self.assertEqual(len(ft_res["generated_heatmaps"]), 1)
        self.assertTrue(Path(ft_res["generated_heatmaps"][0]).is_file())
        self.assertTrue(Path(ft_res["pass_network"]).is_file())

    # -------------------------------------------------------------------------
    # 2. Coordinate Standardization Tests (kloppy)
    # -------------------------------------------------------------------------

    def test_standardize_touch_coordinates_opta(self):
        """Tests conversion from Opta [0, 100] coordinates to metric FIFA [105m, 68m]."""
        raw_opta_points = [
            {"x": 0.0, "y": 0.0},        # Top-left corner
            {"x": 50.0, "y": 50.0},      # Center spot
            {"x": 100.0, "y": 100.0},    # Bottom-right corner
        ]

        df = self.engine.standardize_touch_coordinates(
            raw_opta_points,
            src_system="opta",
            player_name="Pedri",
            player_id=8,
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn("x", df.columns)
        self.assertIn("y", df.columns)

        # Verify metric bounds
        self.assertAlmostEqual(df.iloc[0]["x"], 0.0, places=1)
        self.assertAlmostEqual(df.iloc[0]["y"], 0.0, places=1)
        self.assertAlmostEqual(df.iloc[1]["x"], 52.5, places=1)  # 50% of 105m
        self.assertAlmostEqual(df.iloc[1]["y"], 34.0, places=1)  # 50% of 68m
        self.assertAlmostEqual(df.iloc[2]["x"], 105.0, places=1)
        self.assertAlmostEqual(df.iloc[2]["y"], 68.0, places=1)

    def test_standardize_touch_coordinates_normalized(self):
        """Tests conversion from normalized [0, 1] coordinates to metric meters."""
        raw_norm_points = [
            {"x": 0.25, "y": 0.50},
            {"x": 0.75, "y": 0.50},
        ]
        df = self.engine.standardize_touch_coordinates(
            raw_norm_points,
            src_system="normalized",
            player_name="Gavi",
        )
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(df.iloc[0]["x"], 26.25, places=1)
        self.assertAlmostEqual(df.iloc[0]["y"], 34.0, places=1)

    def test_standardize_pass_network(self):
        """Tests pass event standardization into a clean DataFrame."""
        raw_passes = [
            {
                "passer_id": 8, "passer_name": "Pedri", "passer_number": 8,
                "receiver_id": 19, "receiver_name": "Lamine Yamal", "receiver_number": 19,
                "start_x": 50.0, "start_y": 50.0, "end_x": 80.0, "end_y": 70.0,
                "success": True,
            }
        ]
        df_passes = self.engine.standardize_pass_network(raw_passes, src_system="opta")
        self.assertIsInstance(df_passes, pd.DataFrame)
        self.assertEqual(len(df_passes), 1)
        self.assertEqual(df_passes.iloc[0]["passer_name"], "Pedri")
        self.assertEqual(df_passes.iloc[0]["receiver_name"], "Lamine Yamal")
        self.assertAlmostEqual(df_passes.iloc[0]["x"], 52.5, places=1)

    # -------------------------------------------------------------------------
    # 3. Transparent KDE Heatmap Rendering Tests (mplsoccer)
    # -------------------------------------------------------------------------

    def test_generate_player_heatmap_transparent_png(self):
        """
        Validates that mplsoccer generates studio-quality KDE heatmaps
        with an authentic transparent alpha channel.
        """
        # Create cluster of touches in the attacking right wing
        np.random.seed(42)
        xs = np.random.normal(loc=80.0, scale=8.0, size=40)
        ys = np.random.normal(loc=55.0, scale=5.0, size=40)
        df_coords = pd.DataFrame({"player_name": "Lamine Yamal", "x": xs, "y": ys})

        out_path = self.temp_dir / "heatmap_yamal_test.png"
        res = self.engine.generate_player_heatmap(
            df_coords=df_coords,
            player_name="Lamine Yamal",
            jersey_number=19,
            club_name="Barcelona",
            output_path=out_path,
        )

        self.assertTrue(res.success)
        self.assertTrue(out_path.is_file())
        self.assertGreater(out_path.stat().st_size, 5000)

        # Inspect RGBA pixel buffer for true transparency
        with Image.open(out_path) as img:
            self.assertEqual(img.mode, "RGBA")
            self.assertGreater(img.width, 800)
            self.assertGreater(img.height, 500)

            arr = np.array(img)
            alpha_channel = arr[:, :, 3]

            # 1. Outer regions of transparent canvas must have alpha == 0
            transparent_count = np.sum(alpha_channel == 0)
            self.assertGreater(transparent_count, 100000, "Image must have significant transparent pixels")

            # 2. Foreground pitch markings, text, and heat contours must have alpha > 200
            opaque_count = np.sum(alpha_channel > 200)
            self.assertGreater(opaque_count, 5000, "Image must have crisp non-transparent foreground artwork")

    # -------------------------------------------------------------------------
    # 4. Transparent Team Pass Network Tests (mplsoccer)
    # -------------------------------------------------------------------------

    def test_generate_pass_network_transparent_png(self):
        """
        Validates that mplsoccer generates a studio-grade team Pass Network
        with weighted lines, squad circles, and a transparent canvas.
        """
        df_passes = self.engine._generate_fallback_team_passes("Real Madrid")
        out_path = self.temp_dir / "pass_network_real_madrid_test.png"

        res = self.engine.generate_pass_network(
            df_passes=df_passes,
            team_name="Real Madrid",
            output_path=out_path,
        )

        self.assertTrue(res.success)
        self.assertTrue(out_path.is_file())
        self.assertGreater(out_path.stat().st_size, 5000)

        # Inspect RGBA pixel buffer
        with Image.open(out_path) as img:
            self.assertEqual(img.mode, "RGBA")
            arr = np.array(img)
            alpha_channel = arr[:, :, 3]

            transparent_pixels = np.sum(alpha_channel == 0)
            self.assertGreater(transparent_pixels, 100000)

            opaque_pixels = np.sum(alpha_channel > 200)
            self.assertGreater(opaque_pixels, 3000)

    # -------------------------------------------------------------------------
    # 5. Integration with MatchStatsProvider
    # -------------------------------------------------------------------------

    def test_match_stats_provider_heatmap_enrichment(self):
        """Validates that MatchStatsProvider automatically produces heatmap paths in payload."""
        stats = self.stats_provider.get_player_stats("Kylian Mbappe")
        self.assertIsNotNone(stats)
        self.assertIn("heatmap_png_path", stats)
        self.assertIn("heatmap_url", stats)

        heatmap_path = Path(stats["heatmap_png_path"])
        self.assertTrue(heatmap_path.is_file())
        self.assertTrue(stats["heatmap_url"].endswith(".png"))

    def test_convenience_tactical_methods(self):
        """Validates MatchStatsProvider convenience generators for heatmap & pass network."""
        hm_path = self.stats_provider.generate_player_tactical_heatmap("Vinicius Junior")
        self.assertIsNotNone(hm_path)
        self.assertTrue(Path(hm_path).is_file())

        pn_path = self.stats_provider.generate_team_pass_network("Real Madrid")
        self.assertIsNotNone(pn_path)
        self.assertTrue(Path(pn_path).is_file())


if __name__ == "__main__":
    unittest.main()
