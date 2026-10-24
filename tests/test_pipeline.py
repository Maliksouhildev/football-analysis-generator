"""
Unit tests for the Football Vision AI Pipeline components:
- RosterManager multi-cue Bayesian resolution
- TacticalRenderer geometry, anti-collision, and text sanitization
- Reverse-timeline spatial accuracy evaluation
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vision_pipeline.roster_manager import RosterManager
from vision_pipeline.tactical_renderer import TacticalRenderer, sanitize_text
from vision_pipeline.vision_engine import VisionEngine

class TestRosterManager(unittest.TestCase):
    """Tests dynamic player identification without hardcoding."""

    def setUp(self):
        self.rm = RosterManager()
        self.fixture_gal_bar = self.rm.get_fixture_data("Galatasaray vs Barcelona")
        self.fixture_rma_mci = self.rm.get_fixture_data("Real Madrid vs Manchester City")

    def test_fixture_loading(self):
        self.assertIn("home_team", self.fixture_gal_bar)
        self.assertIn("away_team", self.fixture_gal_bar)
        self.assertEqual(len(self.fixture_gal_bar["away_team"]["lineup"]), 11)
        self.assertEqual(len(self.fixture_gal_bar["home_team"]["lineup"]), 11)

    def test_resolve_exact_number_and_team(self):
        # Pedri #16 on Barcelona
        detection = {
            "team": "Barcelona",
            "jersey_number": 16,
            "tactical_role": "CM",
            "physical_traits": "dark brown short, slender agile"
        }
        res = self.rm.resolve_player(detection, self.fixture_gal_bar)
        self.assertEqual(res["name"], "Pedri")
        self.assertEqual(res["number"], 16)
        self.assertEqual(res["position"], "CM")
        self.assertGreaterEqual(res["confidence"], 90.0)

    def test_resolve_obscured_number_by_traits_and_position(self):
        # Frenkie de Jong #21 with obscured number (None), blond hair, CM position
        detection = {
            "team": "Barcelona",
            "jersey_number": None,
            "tactical_role": "CM",
            "physical_traits": "blonde floppy hair, tall elegant"
        }
        res = self.rm.resolve_player(detection, self.fixture_gal_bar)
        self.assertEqual(res["name"], "Frenkie de Jong")
        self.assertEqual(res["number"], 21)
        self.assertGreaterEqual(res["confidence"], 80.0)

    def test_resolve_goalkeeper(self):
        detection = {
            "team": "Galatasaray",
            "jersey_number": 26,
            "tactical_role": "GK",
            "kit_color": "Green"
        }
        res = self.rm.resolve_player(detection, self.fixture_gal_bar)
        self.assertEqual(res["name"], "Iñaki Peña")
        self.assertEqual(res["position"], "GK")

    def test_resolve_real_madrid_player(self):
        detection = {
            "team": "Real Madrid",
            "jersey_number": 7,
            "tactical_role": "LW",
            "physical_traits": "black locs, agile explosive"
        }
        res = self.rm.resolve_player(detection, self.fixture_rma_mci)
        self.assertEqual(res["name"], "Vinícius Júnior")
        self.assertEqual(res["number"], 7)

class TestTacticalRenderer(unittest.TestCase):
    """Tests visual graphics calculations, anti-collision, and SVG generation."""

    def setUp(self):
        self.renderer = TacticalRenderer()

    def test_sanitize_text(self):
        raw = "Kerem Aktürkoğlu & Taylan Antalyalı & Marcão"
        clean = sanitize_text(raw)
        self.assertEqual(clean, "Kerem Akturkoglu & Taylan Antalyali & Marcao")

    def test_badge_collision_avoidance(self):
        # Two badges overlapping in same area
        badges = [
            {"player": {"id": "P1"}, "team_color": {}, "x": 500.0, "y": 300.0, "is_carrier": True},
            {"player": {"id": "P2"}, "team_color": {}, "x": 520.0, "y": 305.0, "is_carrier": False}
        ]
        resolved = self.renderer._resolve_badge_collisions(badges)
        # Verify vertical distance between badges is now at least 20px
        dy = abs(resolved[0]["y"] - resolved[1]["y"])
        self.assertGreaterEqual(dy, 20.0)

    def test_animated_svg_generation(self):
        analysis = {
            "players": [
                {
                    "id": "P1",
                    "bbox_norm": [300, 400, 360, 440],
                    "foot_pos_norm": [420, 360],
                    "team": "Barcelona",
                    "is_ball_carrier": True,
                    "resolved": {"display_name": "Pedri", "number": 16}
                }
            ],
            "pass_prediction": {
                "pass_origin_norm": [420, 360],
                "pass_destination_norm": [700, 250],
                "pass_type": "through ball"
            }
        }
        svg = self.renderer.generate_animated_svg(1280, 720, analysis, is_t0=True)
        self.assertIn("<svg", svg)
        self.assertIn("pass-vector", svg)
        self.assertIn("#16 Pedri", svg)
        self.assertIn("spotlightGlow", svg)

class TestVisionEngineMetrics(unittest.TestCase):
    """Tests reverse timeline metric calculations."""

    def test_evaluate_reverse_timeline_accuracy(self):
        engine = VisionEngine(api_key="TEST_MOCK_KEY")
        t0 = {
            "pass_prediction": {
                "pass_destination_norm": [800, 300],
                "pass_type": "through ball"
            },
            "players": [{"resolved": {"confidence": 95.0}}]
        }
        t1 = {
            "ball": {"position_norm": [800, 300]}, # Exact match
            "players": [{"resolved": {"confidence": 90.0}}]
        }
        metrics = engine._evaluate_reverse_timeline_accuracy(t0, t1)
        self.assertEqual(metrics["spatial_pass_accuracy_percent"], 100.0)
        self.assertEqual(metrics["euclidean_distance_error_norm"], 0.0)
        self.assertEqual(metrics["average_player_id_confidence"], 92.5)

if __name__ == "__main__":
    unittest.main()
