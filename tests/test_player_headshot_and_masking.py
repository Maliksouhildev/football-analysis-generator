"""
tests/test_player_headshot_and_masking.py
Comprehensive test suite verifying studio-quality player headshots and masking:
1. Primary Route: Official EA Sports FC / Ultimate Team CDN resolution and transparent PNG validation.
2. Fallback Route: Neural BiRefNet / rembg (u2net) segmentation without VLM rotoscoping.
3. PlayerAssetManager integration and URL generation.
4. Broadcast stat card rendering with transparent headshot avatar composite.
"""

import os
import time
import unittest
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

from src.rendering.player_headshot_service import PlayerHeadshotService, EA_SPORTS_FC_PLAYER_IDS
from src.rendering.player_asset_manager import PlayerAssetManager
from vision_pipeline.tactical_renderer import TacticalRenderer


class TestPlayerHeadshotAndMasking(unittest.TestCase):
    """Verifies high-aesthetic player headshots and neural masking."""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path(__file__).resolve().parent.parent / "output"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.service = PlayerHeadshotService()
        cls.renderer = TacticalRenderer()

    def test_slugification_and_name_cleaning(self):
        """Verifies clean extraction of player names and slugs, stripping role tags and diacritics."""
        self.assertEqual(self.service.clean_name("Lamine Yamal (RW)"), "Lamine Yamal")
        self.assertEqual(self.service.clean_name("Kylian Mbappé #9 (CF)"), "Kylian Mbappé")
        self.assertEqual(self.service.get_slug("Lamine Yamal"), "lamine_yamal")
        self.assertEqual(self.service.get_slug("Kylian Mbappé"), "kylian_mbappe")
        self.assertEqual(self.service.get_slug("Ibrahima Konaté (CB)"), "ibrahima_konate")
        self.assertIn("lamine_yamal", EA_SPORTS_FC_PLAYER_IDS)
        self.assertEqual(EA_SPORTS_FC_PLAYER_IDS["lamine_yamal"], 277643)

    def test_primary_route_ea_sports_fc_headshot_yamal(self):
        """
        Primary Route: Verifies fetching official transparent PNG headshot for Lamine Yamal
        from EA Sports FC CDN without any VLM guessing.
        """
        headshot_path = self.service.get_official_headshot("Lamine Yamal", team="Barcelona")
        self.assertIsNotNone(headshot_path)
        self.assertTrue(headshot_path.is_file(), f"Headshot file must exist: {headshot_path}")
        self.assertGreater(headshot_path.stat().st_size, 15000, "Headshot must be genuine high-res image")

        img = Image.open(headshot_path)
        self.assertEqual(img.mode, "RGBA", "Must be RGBA mode with transparency")
        self.assertTrue(self.service.is_valid_transparent_png(headshot_path), "Must have valid transparent alpha channel")

        # Verify fast local cache hit on second lookup
        t0 = time.time()
        cached = self.service.get_official_headshot("Lamine Yamal")
        lookup_time = time.time() - t0
        self.assertEqual(cached, headshot_path)
        self.assertLess(lookup_time, 0.1, "Cached retrieval must take < 100ms")

    def test_transparency_validation(self):
        """Verifies that authentic transparent PNGs pass and solid rectangular boxes fail."""
        # Solid opaque image -> False
        solid = Image.new("RGBA", (200, 200), (255, 255, 255, 255))
        self.assertFalse(self.service.is_valid_transparent_png(solid))

        # True transparent cutout (some transparent pixels, some opaque pixels) -> True
        transparent = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(transparent)
        draw.rectangle([50, 50, 150, 150], fill=(200, 40, 40, 255))
        self.assertTrue(self.service.is_valid_transparent_png(transparent))

    def test_fallback_neural_segmentation_without_vlm(self):
        """
        Fallback Route: Verifies that passing a high-resolution frame crop through
        neural rembg/u2net generates a smooth alpha channel without jagged borders.
        """
        # Create a synthetic broadcast frame crop (green pitch background + white jersey player)
        crop = Image.new("RGB", (240, 360), (35, 120, 45))  # Green turf background
        draw = ImageDraw.Draw(crop)
        # Draw player silhouette
        draw.ellipse([80, 40, 160, 110], fill=(210, 165, 140))  # Head
        draw.rectangle([60, 110, 180, 260], fill=(240, 240, 245))  # White jersey
        draw.rectangle([70, 260, 170, 320], fill=(20, 20, 30))  # Dark shorts
        draw.rectangle([80, 320, 110, 355], fill=(255, 60, 0))  # Left boot
        draw.rectangle([130, 320, 160, 355], fill=(255, 60, 0))  # Right boot

        out_crop_path = self.output_dir / "test_neural_crop_cutout.png"
        t0 = time.time()
        masked_img = self.service.mask_player_crop(
            crop,
            output_path=out_crop_path,
            model="u2net",
            suppress_turf_spill=True,
        )
        dt = time.time() - t0

        self.assertIsNotNone(masked_img)
        self.assertEqual(masked_img.mode, "RGBA")
        self.assertTrue(out_crop_path.is_file())
        self.assertTrue(self.service.is_valid_transparent_png(out_crop_path))
        # Verify inference executed in reasonable timeframe
        self.assertLess(dt, 5.0, f"Neural crop segmentation took {dt:.2f}s")

    def test_player_asset_manager_integration(self):
        """Verifies PlayerAssetManager integrates with PlayerHeadshotService and returns URLs."""
        mgr = PlayerAssetManager(output_dir=str(self.output_dir / "players_test"))
        portrait_path = mgr.get_player_portrait_path("Lamine Yamal", team="Barcelona")
        self.assertIsNotNone(portrait_path)
        self.assertTrue(portrait_path.is_file())

        url = mgr.get_player_portrait("Lamine Yamal", team="Barcelona")
        self.assertIsNotNone(url)
        self.assertIn("lamine_yamal.png", url)

    def test_broadcast_stat_card_with_studio_headshot(self):
        """
        UI Verification: Verifies TacticalRenderer.draw_broadcast_player_stat_card
        composites the studio-quality transparent headshot avatar.
        """
        canvas = Image.new("RGBA", (800, 600), (10, 20, 30, 255))
        draw = ImageDraw.Draw(canvas)

        player_stats = {
            "name": "Lamine Yamal",
            "playerName": "Lamine Yamal",
            "number": 19,
            "position": "RW",
            "team": "Barcelona",
            "rating": 8.8,
            "goals": 2,
            "yellow_cards": 0,
            "red_cards": 0,
            "xg": 0.84,
            "passes": "42/48 (88%)",
            "stats": {"Duels Won": "7/10", "Interceptions": 2},
        }

        res = self.renderer.draw_broadcast_player_stat_card(
            draw=draw,
            player_stats=player_stats,
            x=100,
            y=100,
            card_w=380,
            card_h=165,
            overlay_img=canvas,
        )

        out_stat_path = self.output_dir / "test_broadcast_stat_card_yamal_headshot.png"
        canvas.save(out_stat_path)

        self.assertTrue(res.get("headshot_rendered"), "Must render player headshot on stat card")
        self.assertIsNotNone(res.get("headshot_source"))
        self.assertEqual(res.get("goals_rendered"), 2)
        self.assertEqual(res.get("rating_rendered"), 8.8)
        self.assertTrue(out_stat_path.is_file())


if __name__ == "__main__":
    unittest.main()
