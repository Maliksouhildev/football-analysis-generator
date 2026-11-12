"""
tests/test_event_anchor_resolver.py
Unit and integration tests for EventAnchorResolver.

Verifies:
1. Ground truth anchoring of ball carriers fusing synchronized StatsBomb events.
2. Prevention of the "Nearest Player Fallback Trap" (False Carrier Attribution in tackles/duels).
3. Carrier-ball velocity directional alignment.
4. Consecutive event-chain linking (Passer -> Receiver).
5. Edge cases: empty candidates, missing ball detections, low-confidence cutoffs.
"""

from __future__ import annotations

import unittest
from src.vision.event_anchor_resolver import EventAnchorResolver, GroundTruthAnchorNode
from src.scraper.event_scraper import CanonicalMatchEvent, CanonicalEventType, PlayerInfo, TeamInfo


class TestEventAnchorResolver(unittest.TestCase):
    """Test suite for EventAnchorResolver."""

    def setUp(self):
        self.resolver = EventAnchorResolver(
            pitch_length_m=105.0,
            pitch_width_m=68.0,
            max_carrier_ball_dist_m=4.0,
            w_dist=0.35,
            w_kit=0.35,
            w_pitch=0.20,
            w_vel=0.10,
            min_possession_threshold=0.40,
        )

    def test_anchor_ball_carrier_positive(self):
        """Attacking player in ball possession matching event is anchored as Ground Truth."""
        candidates = [
            {
                "track_id": 10,
                "team": "Barcelona",
                "pitch_x": 52.0,
                "pitch_y": 34.0,
                "box": [500, 300, 560, 420],
            },
            {
                "track_id": 11,
                "team": "Barcelona",
                "pitch_x": 75.0,
                "pitch_y": 15.0,
                "box": [800, 150, 840, 250],
            },
        ]
        ball = {"pitch_x": 52.2, "pitch_y": 34.1}
        event = {
            "player": {"name": "Pedri"},
            "team": {"name": "Barcelona"},
            "type": {"name": "pass"},
            "pitch_x": 50.0,
            "pitch_y": 34.0,
            "timestamp_sec": 14.5,
        }

        anchor = self.resolver.resolve_anchor(candidates, ball, event)
        self.assertIsNotNone(anchor)
        self.assertIsInstance(anchor, GroundTruthAnchorNode)
        self.assertEqual(anchor.track_id, 10)
        self.assertEqual(anchor.player_name, "Pedri")
        self.assertEqual(anchor.team, "Barcelona")
        self.assertEqual(anchor.role, "BALL_CARRIER")
        self.assertTrue(anchor.is_ground_truth)
        self.assertEqual(anchor.status, "ANCHORED_GROUND_TRUTH")
        self.assertGreater(anchor.possession_score, 0.70)

    def test_false_carrier_attribution_tackle_firewall(self):
        """
        Critical Test: In a tackle/duel, defending tackler is closer to the ball,
        but the offensive pass belongs to Barcelona. Kit firewall prevents the defender
        from stealing the ball-carrier anchor identity!
        """
        # Slide-tackling defender is 0.2m from the ball
        cand_defender = {
            "track_id": 99,
            "team": "Real Madrid",
            "pitch_x": 50.2,
            "pitch_y": 34.0,
            "box": [480, 310, 540, 410],
        }
        # Attacking ball-carrier is 0.7m from the ball
        cand_carrier = {
            "track_id": 8,
            "team": "Barcelona",
            "pitch_x": 50.7,
            "pitch_y": 34.0,
            "box": [490, 300, 550, 420],
        }
        ball = {"pitch_x": 50.0, "pitch_y": 34.0}
        event = {
            "player": {"name": "Pedri"},
            "team": {"name": "Barcelona"},
            "type": {"name": "pass"},
            "pitch_x": 50.0,
            "pitch_y": 34.0,
        }

        # Pass defender first in candidate list to test sorting & kit weighting
        anchor = self.resolver.resolve_anchor([cand_defender, cand_carrier], ball, event)

        self.assertIsNotNone(anchor)
        # MUST anchor the Barcelona carrier (Track 8), NEVER the Real Madrid defender (Track 99)
        self.assertEqual(anchor.track_id, 8)
        self.assertEqual(anchor.player_name, "Pedri")
        self.assertEqual(anchor.team, "Barcelona")

    def test_carrier_velocity_alignment(self):
        """Carrier running with ball direction scores higher than player running against ball."""
        ball = {
            "pitch_x": 50.0,
            "pitch_y": 34.0,
            "velocity": (5.0, 0.0),  # Ball moving along +X
        }
        event = {
            "player": {"name": "Lamine Yamal"},
            "team": {"name": "Barcelona"},
            "type": {"name": "carry"},
            "pitch_x": 50.0,
            "pitch_y": 34.0,
        }
        # Player A moving forward with ball
        cand_aligned = {
            "track_id": 1,
            "team": "Barcelona",
            "pitch_x": 50.5,
            "pitch_y": 34.0,
            "velocity": (4.8, 0.0),
            "box": [100, 100, 150, 200],
        }
        # Player B moving backward away from ball
        cand_opposed = {
            "track_id": 2,
            "team": "Barcelona",
            "pitch_x": 50.5,
            "pitch_y": 34.0,
            "velocity": (-4.8, 0.0),
            "box": [100, 100, 150, 200],
        }

        scored = self.resolver.evaluate_candidates([cand_aligned, cand_opposed], ball, event)
        self.assertGreater(scored[0]["s_vel"], scored[1]["s_vel"])
        self.assertGreater(scored[0]["possession_score"], scored[1]["possession_score"])
        self.assertEqual(scored[0]["track_id"], 1)

    def test_canonical_match_event_support(self):
        """Resolver accepts CanonicalMatchEvent objects directly."""
        canonical_ev = CanonicalMatchEvent(
            id="sb_12345",
            match_id="clasico_match",
            period=1,
            minute=15,
            second=22,
            event_type=CanonicalEventType.PASS,
            team=TeamInfo(name="Barcelona", id="77"),
            primary_player=PlayerInfo(name="Pedri", id="8"),
            pitch_x=60.0,
            pitch_y=40.0,
            provider="statsbomb",
        )
        candidates = [
            {"track_id": 5, "team": "Barcelona", "pitch_x": 63.0, "pitch_y": 27.2, "box": [10, 10, 40, 80]},
        ]
        ball = {"pitch_x": 63.2, "pitch_y": 27.1}

        anchor = self.resolver.resolve_anchor(candidates, ball, canonical_ev)
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor.player_name, "Pedri")
        self.assertEqual(anchor.team, "Barcelona")
        self.assertEqual(anchor.event_type, "pass")

    def test_event_chain_linking(self):
        """Resolves consecutive events (Passer at t1 -> Receiver at t2)."""
        ev1 = {"player": {"name": "Passer"}, "team": {"name": "Home"}, "type": "pass", "timestamp_sec": 1.0, "pitch_x": 30.0, "pitch_y": 30.0}
        ev2 = {"player": {"name": "Receiver"}, "team": {"name": "Home"}, "type": "pass", "timestamp_sec": 3.0, "pitch_x": 60.0, "pitch_y": 30.0}

        frames = {
            30: {  # t = 1.0s (30 fps)
                "timestamp_sec": 1.0,
                "players": [{"track_id": 1, "team": "Home", "pitch_x": 30.0, "pitch_y": 30.0, "box": [10, 10, 50, 90]}],
                "ball": {"pitch_x": 30.2, "pitch_y": 30.0},
            },
            90: {  # t = 3.0s (30 fps)
                "timestamp_sec": 3.0,
                "players": [{"track_id": 2, "team": "Home", "pitch_x": 60.0, "pitch_y": 30.0, "box": [10, 10, 50, 90]}],
                "ball": {"pitch_x": 60.1, "pitch_y": 30.0},
            },
        }

        chain = self.resolver.resolve_event_chain([ev1, ev2], frames, fps=30.0)
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[30].player_name, "Passer")
        self.assertEqual(chain[30].track_id, 1)
        self.assertEqual(chain[90].player_name, "Receiver")
        self.assertEqual(chain[90].track_id, 2)

    def test_edge_cases_empty_or_subthreshold(self):
        """Handles empty candidate lists and sub-threshold possession cleanly."""
        self.assertIsNone(self.resolver.resolve_anchor([], None, {}))

        # Distant candidate far outside possession range
        distant_cand = [{"track_id": 1, "team": "Barcelona", "pitch_x": 10.0, "pitch_y": 10.0, "box": [0, 0, 10, 10]}]
        ball = {"pitch_x": 90.0, "pitch_y": 60.0}
        ev = {"player": {"name": "Pedri"}, "team": {"name": "Barcelona"}, "type": "pass", "pitch_x": 90.0, "pitch_y": 60.0}

        # Player is 90 meters away from ball and event location -> should score below threshold
        anchor = self.resolver.resolve_anchor(distant_cand, ball, ev)
        self.assertIsNone(anchor)


if __name__ == "__main__":
    unittest.main()
