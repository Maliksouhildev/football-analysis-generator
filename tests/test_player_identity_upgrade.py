"""
tests/test_player_identity_upgrade.py
Unit tests verifying the 4 key requirements:
1. Strict 11-player Active Squad Filtering (no bench player dumping).
2. Barcelona Right-to-Left Geometry (Depth inverted, Flanks inverted).
3. Safety Valve Overrides (--overrides '{"T18": 9, "T12": 11}').
4. Hard Team Firewall & Deduplication.
"""

import unittest
import numpy as np
from src.vision.match_roster_engine import ActiveXIResolver
from src.vision.player_identity_resolver import PlayerIdentityResolver, GEMINI_PLAYER_IDENTIFICATION_SYSTEM_PROMPT


class TestPlayerIdentityUpgrade(unittest.TestCase):

    def test_strict_active_11_filtering(self):
        """Active squad must strictly contain 11 players on-pitch, no future subs or unentered bench players."""
        resolver = ActiveXIResolver(home_team="Real Madrid", away_team="Barcelona")

        # Minute 46: Early 2nd half, no subs have occurred yet
        state_46 = resolver.get_active_players(match_minute=46)
        self.assertEqual(len(state_46.home_active), 11, f"Expected 11 home players at min 46, got {len(state_46.home_active)}")
        self.assertEqual(len(state_46.away_active), 11, f"Expected 11 away players at min 46, got {len(state_46.away_active)}")

        # Verify starters are on pitch
        home_names_46 = [p["name"] for p in state_46.home_active]
        away_names_46 = [p["name"] for p in state_46.away_active]
        self.assertIn("Robert Lewandowski", away_names_46)
        self.assertIn("Vinícius Júnior", home_names_46)

        # Future substitutes must be in benched_list, NOT in active squad
        benched_names = [p["name"] for p in state_46.benched_players]
        self.assertIn("Frenkie de Jong", benched_names)
        self.assertNotIn("Frenkie de Jong", away_names_46)

        # Minute 80: Substitutions have occurred (Casadó -> de Jong at 65', Olmo -> Fermín at 75')
        state_80 = resolver.get_active_players(match_minute=80)
        self.assertEqual(len(state_80.home_active), 11, f"Expected strictly 11 home players at min 80, got {len(state_80.home_active)}")
        self.assertEqual(len(state_80.away_active), 11, f"Expected strictly 11 away players at min 80, got {len(state_80.away_active)}")

        away_names_80 = [p["name"] for p in state_80.away_active]
        self.assertIn("Frenkie de Jong", away_names_80)
        self.assertIn("Fermín López", away_names_80)
        self.assertNotIn("Marc Casadó", away_names_80)
        self.assertNotIn("Dani Olmo", away_names_80)

        # Subbed-off players must be in subbed_off_players
        subbed_off_names = [p["name"] for p in state_80.subbed_off_players]
        self.assertIn("Marc Casadó", subbed_off_names)
        self.assertIn("Dani Olmo", subbed_off_names)

    def test_barcelona_right_to_left_geometry(self):
        """
        Barcelona attacks right-to-left (-X):
        Smaller X = Forwards/Wingers (near opponent's box)
        Larger X = Defenders/GK (near midfield/own goal)
        Top of screen (Y -> 0) = Right Flank (RB, RW)
        Bottom of screen (Y -> 1) = Left Flank (LB, LW)
        """
        # Mock track near top touchline and small X (e.g. Lamine Yamal at RW on Barcelona)
        # x_pct: 25.0 (near opponent box), y_pct: 20.0 (top of screen)
        yamal_track = [{"track_id": 19, "x_pct": 25.0, "y_pct": 20.0, "team": "away"}]
        # Mock track near bottom touchline and small X (e.g. Balde/Raphinha at LW/LB on Barcelona)
        # x_pct: 25.0 (near opponent box), y_pct: 80.0 (bottom of screen)
        balde_track = [{"track_id": 3, "x_pct": 25.0, "y_pct": 80.0, "team": "away"}]

        # Test project_tracks_to_topology with left attack
        coords_yamal = PlayerIdentityResolver._project_tracks_to_topology(yamal_track, attack_direction="left")
        coords_balde = PlayerIdentityResolver._project_tracks_to_topology(balde_track, attack_direction="left")

        # In single track normalization, min=max so norm=0.
        # Let's test with a squad of tracks across the pitch:
        squad_tracks = [
            {"track_id": 1, "x_pct": 15.0, "y_pct": 25.0},  # Front attacker top flank (RW)
            {"track_id": 2, "x_pct": 15.0, "y_pct": 75.0},  # Front attacker bottom flank (LW)
            {"track_id": 3, "x_pct": 85.0, "y_pct": 25.0},  # Back defender top flank (RB)
            {"track_id": 4, "x_pct": 85.0, "y_pct": 75.0},  # Back defender bottom flank (LB)
        ]
        coords = PlayerIdentityResolver._project_tracks_to_topology(squad_tracks, attack_direction="left")

        # Track 1 (x=15, y=25): should have high attacking depth (0.85) and high width (Right Flank: 0.85)
        self.assertAlmostEqual(coords[0][0], 0.85, delta=0.05)
        self.assertAlmostEqual(coords[0][1], 0.85, delta=0.05)

        # Track 2 (x=15, y=75): should have high attacking depth (0.85) and low width (Left Flank: 0.15)
        self.assertAlmostEqual(coords[1][0], 0.85, delta=0.05)
        self.assertAlmostEqual(coords[1][1], 0.15, delta=0.05)

        # Track 3 (x=85, y=25): back defender near midfield/own goal (depth: 0.20), top flank (width: 0.85)
        self.assertAlmostEqual(coords[2][0], 0.20, delta=0.05)
        self.assertAlmostEqual(coords[2][1], 0.85, delta=0.05)

        # Track 4 (x=85, y=75): back defender (depth: 0.20), bottom flank (width: 0.15)
        self.assertAlmostEqual(coords[3][0], 0.20, delta=0.05)
        self.assertAlmostEqual(coords[3][1], 0.15, delta=0.05)

    def test_safety_valve_overrides(self):
        """Overrides (--overrides '{"T18": 9, "T12": 11}') lock identities with 100% confidence and prevent duplicates."""
        resolver = ActiveXIResolver(home_team="Real Madrid", away_team="Barcelona")
        state = resolver.get_active_players(match_minute=46)
        lineup_info = {
            "home_active_xi": state.home_active,
            "away_active_xi": state.away_active,
        }

        # Suppose track 18 was ambiguous or wrongly identified
        assignments = {
            18: {"name": "Wrong Player", "short_name": "Wrong (CF)", "role": "CF", "number": 99, "team": "away"},
            12: {"name": "Another Player", "short_name": "Another (LW)", "role": "LW", "number": 98, "team": "away"},
            5: {"name": "Robert Lewandowski", "short_name": "Lewandowski (CF)", "role": "CF", "number": 9, "team": "away"},
        }

        # Human override: T18 is #9 Lewandowski, T12 is #11 Raphinha
        overrides = {"T18": 9, "T12": 11}
        locked = PlayerIdentityResolver.apply_overrides(assignments, overrides, lineup_info)

        # Check T18 is locked to Robert Lewandowski
        self.assertEqual(locked[18]["name"], "Robert Lewandowski")
        self.assertEqual(locked[18]["number"], 9)
        self.assertEqual(locked[18]["confidence"], 1.0)
        self.assertTrue(locked[18]["verified_by_override"])

        # Check T12 is locked to Raphinha
        self.assertEqual(locked[12]["name"], "Raphinha")
        self.assertEqual(locked[12]["number"], 11)
        self.assertEqual(locked[12]["confidence"], 1.0)

        # Check duplicate elimination: T5 no longer holds Lewandowski
        self.assertNotIn(5, locked, "Old track holding Lewandowski should have been stripped to prevent duplicate")

    def test_system_prompt_present(self):
        """Verify the exact user system prompt is embedded."""
        self.assertIn("CRITICAL RULES TO PREVENT HALLUCINATIONS", GEMINI_PLAYER_IDENTIFICATION_SYSTEM_PROMPT)
        self.assertIn("HARD TEAM FIREWALL", GEMINI_PLAYER_IDENTIFICATION_SYSTEM_PROMPT)
        self.assertIn("ATTACKING DIRECTION & DEPTH GEOMETRY", GEMINI_PLAYER_IDENTIFICATION_SYSTEM_PROMPT)
        self.assertIn("NO DUPLICATES", GEMINI_PLAYER_IDENTIFICATION_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
