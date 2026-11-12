"""
tests/test_tactical_subgraph_selector.py
Unit and integration tests for TacticalSubgraphSelector.

Verifies:
1. Identification and tagging of the Ball Carrier.
2. Immediate pressers inside dynamic pressing radius (R <= 4.5m) with pressure intensity.
3. Clear line-of-sight passing options vs obstructed passing lanes.
4. Identification of Lane Blockers intersecting passing corridors.
5. Forward off-the-ball runners penetrating space.
6. Zero player duplication across tactical actors in the proximity subgraph.
7. Edge cases: solo carrier, no opponents, distant teammates.
"""

from __future__ import annotations

import unittest
from src.vision.tactical_subgraph_selector import (
    TacticalSubgraphSelector,
    TacticalRole,
    PassingLane,
    TacticalSubgraph,
)


class TestTacticalSubgraphSelector(unittest.TestCase):
    """Test suite for TacticalSubgraphSelector."""

    def setUp(self):
        self.selector = TacticalSubgraphSelector(
            pressing_radius_m=4.5,
            lane_corridor_width_m=2.0,
            min_pass_distance_m=3.0,
            max_pass_distance_m=40.0,
            runner_forward_gain_m=4.0,
        )

    def test_ball_carrier_identification(self):
        """Ball carrier is correctly placed into subgraph with BALL_CARRIER role."""
        carrier = {
            "track_id": 8,
            "player_name": "Pedri",
            "team": "Barcelona",
            "pitch_x": 50.0,
            "pitch_y": 34.0,
        }
        subgraph = self.selector.extract_subgraph(carrier, [carrier])

        self.assertIsInstance(subgraph, TacticalSubgraph)
        self.assertEqual(subgraph.carrier["player_name"], "Pedri")
        self.assertEqual(subgraph.carrier["tactical_role"], TacticalRole.BALL_CARRIER.value)
        self.assertEqual(len(subgraph.tactical_actors), 1)

    def test_immediate_pressers_within_pressing_radius(self):
        """Opponents within 4.5m are immediate pressers; opponents beyond 4.5m are not."""
        carrier = {"track_id": 8, "player_name": "Pedri", "team": "Barcelona", "pitch_x": 50.0, "pitch_y": 34.0}

        # Opponent 1: 2.0m away (Inside pressing radius)
        opp_close = {"track_id": 15, "player_name": "Valverde", "team": "Real Madrid", "pitch_x": 51.5, "pitch_y": 35.0}

        # Opponent 2: 8.0m away (Outside pressing radius)
        opp_far = {"track_id": 14, "player_name": "Tchouameni", "team": "Real Madrid", "pitch_x": 58.0, "pitch_y": 34.0}

        subgraph = self.selector.extract_subgraph(carrier, [carrier, opp_close, opp_far])

        self.assertEqual(len(subgraph.pressers), 1)
        presser = subgraph.pressers[0]
        self.assertEqual(presser["track_id"], 15)
        self.assertEqual(presser["player_name"], "Valverde")
        self.assertEqual(presser["tactical_role"], TacticalRole.IMMEDIATE_PRESSER.value)
        self.assertLessEqual(presser["dist_to_carrier_m"], 4.5)
        self.assertGreater(presser["pressing_intensity"], 0.0)

    def test_passing_lanes_open_vs_blocked(self):
        """Line-of-sight analysis detects open passing options vs obstructed lanes."""
        carrier = {"track_id": 8, "player_name": "Pedri", "team": "Barcelona", "pitch_x": 50.0, "pitch_y": 34.0}

        # Teammate A (Open on right flank): (60, 55), corridor clear
        mate_open = {"track_id": 10, "player_name": "Lamine Yamal", "team": "Barcelona", "pitch_x": 60.0, "pitch_y": 55.0}

        # Teammate B (Central forward): (70, 34)
        mate_blocked = {"track_id": 9, "player_name": "Lewandowski", "team": "Barcelona", "pitch_x": 70.0, "pitch_y": 34.0}

        # Opponent blocking lane to Teammate B: standing at (60, 34) directly on the line
        opp_blocker = {"track_id": 22, "player_name": "Rudiger", "team": "Real Madrid", "pitch_x": 60.0, "pitch_y": 34.0}

        subgraph = self.selector.extract_subgraph(carrier, [carrier, mate_open, mate_blocked, opp_blocker])

        # Verify passing lanes
        lanes = {l.receiver_name: l for l in subgraph.passing_lanes}
        self.assertTrue(lanes["Lamine Yamal"].is_open)
        self.assertEqual(lanes["Lamine Yamal"].openness_score, 1.0)

        self.assertFalse(lanes["Lewandowski"].is_open)
        self.assertIn(22, lanes["Lewandowski"].blockers)
        self.assertLess(lanes["Lewandowski"].openness_score, 1.0)

        # Verify Lane Blocker role
        self.assertEqual(len(subgraph.lane_blockers), 1)
        self.assertEqual(subgraph.lane_blockers[0]["track_id"], 22)
        self.assertEqual(subgraph.lane_blockers[0]["tactical_role"], TacticalRole.LANE_BLOCKER.value)

    def test_off_the_ball_runners(self):
        """Teammates advancing ahead of carrier in attacking direction are off-the-ball runners."""
        carrier = {"track_id": 8, "player_name": "Pedri", "team": "Barcelona", "pitch_x": 50.0, "pitch_y": 34.0}

        # Runner making a deep forward run: X = 65m (+15m forward)
        runner = {"track_id": 11, "player_name": "Raphinha", "team": "Barcelona", "pitch_x": 65.0, "pitch_y": 20.0}

        # Drop-back passing option: X = 45m (-5m backward)
        anchor_cb = {"track_id": 2, "player_name": "Cubarsi", "team": "Barcelona", "pitch_x": 42.0, "pitch_y": 34.0}

        subgraph = self.selector.extract_subgraph(carrier, [carrier, runner, anchor_cb], attack_direction="right")

        runner_names = [r["player_name"] for r in subgraph.runners]
        self.assertIn("Raphinha", runner_names)
        self.assertNotIn("Cubarsi", runner_names)

    def test_tactical_actors_uniqueness_guarantee(self):
        """All selected tactical actors have unique track IDs."""
        carrier = {"track_id": 8, "player_name": "Pedri", "team": "Barcelona", "pitch_x": 50.0, "pitch_y": 34.0}
        p1 = {"track_id": 1, "player_name": "Mate1", "team": "Barcelona", "pitch_x": 55.0, "pitch_y": 40.0}
        p2 = {"track_id": 2, "player_name": "Mate2", "team": "Barcelona", "pitch_x": 60.0, "pitch_y": 30.0}
        o1 = {"track_id": 3, "player_name": "Opp1", "team": "Real Madrid", "pitch_x": 52.0, "pitch_y": 34.0}
        o2 = {"track_id": 4, "player_name": "Opp2", "team": "Real Madrid", "pitch_x": 54.0, "pitch_y": 37.0}

        subgraph = self.selector.extract_subgraph(carrier, [carrier, p1, p2, o1, o2])

        actor_tids = [a["track_id"] for a in subgraph.tactical_actors]
        self.assertEqual(len(actor_tids), len(set(actor_tids)), "Duplicate actor track ID detected in subgraph!")

    def test_referee_excluded_from_tactical_subgraph(self):
        """Referees are never selected as teammates, opponents, or passing options."""
        carrier = {"track_id": 8, "player_name": "Pedri", "team": "Barcelona", "pitch_x": 50.0, "pitch_y": 34.0}
        referee = {"track_id": 99, "role": "referee", "is_referee": True, "pitch_x": 52.0, "pitch_y": 33.0}

        subgraph = self.selector.extract_subgraph(carrier, [carrier, referee])

        tids = [a["track_id"] for a in subgraph.tactical_actors]
        self.assertNotIn(99, tids)

    def test_none_pitch_coords_gracefully_handled(self):
        """Candidates with pitch_coords=None or missing coords do not cause TypeError."""
        carrier = {"track_id": 1, "player_name": "Solo", "team": "Home", "pitch_coords": None, "pitch_x": None, "pitch_y": None}
        p1 = {"track_id": 2, "player_name": "Teammate", "team": "Home", "pitch_coords": None, "pitch_x": 65.0, "pitch_y": 34.0}

        subgraph = self.selector.extract_subgraph(carrier, [carrier, p1])
        self.assertIsInstance(subgraph, TacticalSubgraph)
        self.assertEqual(len(subgraph.tactical_actors), 2)

    def test_solo_carrier_no_other_players(self):
        """Carrier alone on pitch extracts valid subgraph with 0 pressers and 0 options."""
        carrier = {"track_id": 1, "player_name": "Solo", "team": "Home", "pitch_x": 50.0, "pitch_y": 34.0}
        subgraph = self.selector.extract_subgraph(carrier, [carrier])
        self.assertEqual(len(subgraph.pressers), 0)
        self.assertEqual(len(subgraph.passing_options), 0)
        self.assertEqual(len(subgraph.tactical_actors), 1)


if __name__ == "__main__":
    unittest.main()
