"""
tests/test_live_ground_truth.py
Comprehensive Verification Suite for Prompt 1:
- Sportmonks API Live Scores (/v3/football/livescores/inplay?include=scores;participants;events;lineups)
- API-Football Live Events Fallback
- Dynamic Ground Truth Dictionary & Events Timeline Parser
- Zero Hardcoded Lineups Enforcement
- Anti-Hallucination IdentityGate Mathematical Verification
"""

import unittest
from typing import Any, Dict

from src.scraper.live_ground_truth import (
    ApiFootballClient,
    GroundTruthDictionary,
    GroundTruthEvent,
    IdentityGate,
    MatchInstant,
    PlayerGroundTruth,
    SportmonksClient,
    clear_registered_ground_truth,
    fetch_live_ground_truth,
    get_registered_ground_truth,
    register_ground_truth,
)
from vision_pipeline.roster_manager import RosterManager
from src.vision.match_roster_engine import ActiveXIResolver


class TestLiveGroundTruth(unittest.TestCase):
    """Test suite for live API ground truth dictionary and hallucination gate."""

    def setUp(self):
        clear_registered_ground_truth()

        # Deterministic Sportmonks live scores payload (include=scores;participants;events;lineups)
        self.sportmonks_inplay_payload: Dict[str, Any] = {
            "data": [
                {
                    "id": 19482011,
                    "name": "Arsenal vs Chelsea",
                    "result_info": "2nd Half Inplay",
                    "minute": 65,
                    "scores": [
                        {"id": 1, "score": {"goals": 2, "participant": "home"}, "description": "CURRENT"},
                        {"id": 2, "score": {"goals": 1, "participant": "away"}, "description": "CURRENT"},
                    ],
                    "participants": [
                        {"id": 101, "name": "Arsenal", "short_code": "ARS", "meta": {"location": "home"}},
                        {"id": 102, "name": "Chelsea", "short_code": "CHE", "meta": {"location": "away"}},
                    ],
                    "lineups": [
                        # Arsenal Starting XI (type_id=11)
                        {"player_id": 1, "team_id": 101, "type_id": 11, "jersey_number": 22, "player_name": "David Raya", "position": {"code": "GK"}},
                        {"player_id": 2, "team_id": 101, "type_id": 11, "jersey_number": 4, "player_name": "Ben White", "position": {"code": "RB"}},
                        {"player_id": 3, "team_id": 101, "type_id": 11, "jersey_number": 2, "player_name": "William Saliba", "position": {"code": "CB"}},
                        {"player_id": 4, "team_id": 101, "type_id": 11, "jersey_number": 6, "player_name": "Gabriel Magalhães", "position": {"code": "CB"}},
                        {"player_id": 5, "team_id": 101, "type_id": 11, "jersey_number": 12, "player_name": "Jurriën Timber", "position": {"code": "LB"}},
                        {"player_id": 6, "team_id": 101, "type_id": 11, "jersey_number": 5, "player_name": "Thomas Partey", "position": {"code": "DM"}},
                        {"player_id": 7, "team_id": 101, "type_id": 11, "jersey_number": 41, "player_name": "Declan Rice", "position": {"code": "CM"}},
                        {"player_id": 8, "team_id": 101, "type_id": 11, "jersey_number": 8, "player_name": "Martin Ødegaard", "position": {"code": "AM"}},
                        {"player_id": 9, "team_id": 101, "type_id": 11, "jersey_number": 7, "player_name": "Bukayo Saka", "position": {"code": "RW"}},
                        {"player_id": 10, "team_id": 101, "type_id": 11, "jersey_number": 29, "player_name": "Kai Havertz", "position": {"code": "CF"}},
                        {"player_id": 11, "team_id": 101, "type_id": 11, "jersey_number": 11, "player_name": "Gabriel Martinelli", "position": {"code": "LW"}},
                        # Arsenal Bench (type_id=12)
                        {"player_id": 12, "team_id": 101, "type_id": 12, "jersey_number": 19, "player_name": "Leandro Trossard", "position": {"code": "LW"}},
                        {"player_id": 13, "team_id": 101, "type_id": 12, "jersey_number": 9, "player_name": "Gabriel Jesus", "position": {"code": "CF"}},

                        # Chelsea Starting XI (type_id=11)
                        {"player_id": 20, "team_id": 102, "type_id": 11, "jersey_number": 1, "player_name": "Robert Sánchez", "position": {"code": "GK"}},
                        {"player_id": 21, "team_id": 102, "type_id": 11, "jersey_number": 27, "player_name": "Malo Gusto", "position": {"code": "RB"}},
                        {"player_id": 22, "team_id": 102, "type_id": 11, "jersey_number": 29, "player_name": "Wesley Fofana", "position": {"code": "CB"}},
                        {"player_id": 23, "team_id": 102, "type_id": 11, "jersey_number": 6, "player_name": "Levi Colwill", "position": {"code": "CB"}},
                        {"player_id": 24, "team_id": 102, "type_id": 11, "jersey_number": 3, "player_name": "Marc Cucurella", "position": {"code": "LB"}},
                        {"player_id": 25, "team_id": 102, "type_id": 11, "jersey_number": 25, "player_name": "Moisés Caicedo", "position": {"code": "DM"}},
                        {"player_id": 26, "team_id": 102, "type_id": 11, "jersey_number": 8, "player_name": "Enzo Fernández", "position": {"code": "CM"}},
                        {"player_id": 27, "team_id": 102, "type_id": 11, "jersey_number": 20, "player_name": "Cole Palmer", "position": {"code": "AM"}},
                        {"player_id": 28, "team_id": 102, "type_id": 11, "jersey_number": 11, "player_name": "Noni Madueke", "position": {"code": "RW"}},
                        {"player_id": 29, "team_id": 102, "type_id": 11, "jersey_number": 15, "player_name": "Nicolas Jackson", "position": {"code": "CF"}},
                        {"player_id": 30, "team_id": 102, "type_id": 11, "jersey_number": 7, "player_name": "Pedro Neto", "position": {"code": "LW"}},
                        # Chelsea Bench (type_id=12)
                        {"player_id": 31, "team_id": 102, "type_id": 12, "jersey_number": 18, "player_name": "Christopher Nkunku", "position": {"code": "AM"}},
                    ],
                    "events": [
                        # Substitution at minute 58: Trossard replaces Martinelli for Arsenal
                        {
                            "id": 501,
                            "type_id": 18,
                            "participant_id": 101,
                            "period_id": 2,
                            "minute": 58,
                            "player_id": 11,
                            "player_name": "Gabriel Martinelli",
                            "related_player_id": 12,
                            "related_player_name": "Leandro Trossard",
                            "type": {"name": "Substitution", "code": "sub"},
                        },
                        # Red Card at minute 62: Wesley Fofana sent off for Chelsea
                        {
                            "id": 502,
                            "type_id": 14,
                            "participant_id": 102,
                            "period_id": 2,
                            "minute": 62,
                            "player_id": 22,
                            "player_name": "Wesley Fofana",
                            "type": {"name": "Red Card", "code": "redcard"},
                        },
                    ],
                }
            ]
        }

    def tearDown(self):
        clear_registered_ground_truth()

    def test_sportmonks_parser_generates_ground_truth_dictionary(self):
        """Verifies Sportmonks client parses participants, starting XI, subs, and red cards."""
        gt = SportmonksClient.parse_payload_to_ground_truth(self.sportmonks_inplay_payload)

        self.assertEqual(gt.provider, "sportmonks")
        self.assertEqual(gt.home_team, "Arsenal")
        self.assertEqual(gt.away_team, "Chelsea")
        self.assertEqual(gt.scores, {"home": 2, "away": 1})
        self.assertEqual(len(gt.starting_xi_home), 11)
        self.assertEqual(len(gt.starting_xi_away), 11)
        self.assertEqual(len(gt.substitutes_home), 2)
        self.assertEqual(len(gt.substitutes_away), 1)
        self.assertEqual(len(gt.events_timeline), 2)

    def test_mathematical_on_pitch_tracking_before_and_after_substitution(self):
        """
        Mathematical verification:
        At minute 30 (before sub at 58'): Martinelli is ON pitch, Trossard is OFF pitch (bench).
        At minute 60 (after sub at 58'): Martinelli is OFF pitch (subbed), Trossard is ON pitch.
        """
        gt = SportmonksClient.parse_payload_to_ground_truth(self.sportmonks_inplay_payload)

        # 1. Minute 30 (Period 1)
        t30 = MatchInstant.from_clock(minute=30, second=0, period=1)
        self.assertTrue(gt.is_on_pitch("Gabriel Martinelli", t30))
        self.assertFalse(gt.is_on_pitch("Leandro Trossard", t30))

        home_on_30 = gt.on_pitch(t30, "home")
        self.assertEqual(len(home_on_30), 11)
        self.assertTrue(any(p.name == "Gabriel Martinelli" for p in home_on_30))
        self.assertFalse(any(p.name == "Leandro Trossard" for p in home_on_30))

        # 2. Minute 60 (Period 2)
        t60 = MatchInstant.from_clock(minute=60, second=0, period=2)
        self.assertFalse(gt.is_on_pitch("Gabriel Martinelli", t60))
        self.assertTrue(gt.is_on_pitch("Leandro Trossard", t60))

        home_on_60 = gt.on_pitch(t60, "home")
        self.assertEqual(len(home_on_60), 11)
        self.assertFalse(any(p.name == "Gabriel Martinelli" for p in home_on_60))
        self.assertTrue(any(p.name == "Leandro Trossard" for p in home_on_60))

    def test_mathematical_squad_size_reduction_on_red_card(self):
        """
        Mathematical verification:
        At minute 60 (before red card at 62'): Chelsea has 11 men, Fofana is ON pitch.
        At minute 65 (after red card at 62'): Chelsea has exactly 10 men, Fofana is OFF pitch.
        """
        gt = SportmonksClient.parse_payload_to_ground_truth(self.sportmonks_inplay_payload)

        # Before red card
        t60 = MatchInstant.from_clock(minute=60, period=2)
        away_on_60 = gt.on_pitch(t60, "away")
        self.assertEqual(len(away_on_60), 11)
        self.assertTrue(gt.is_on_pitch("Wesley Fofana", t60))

        # After red card
        t65 = MatchInstant.from_clock(minute=65, period=2)
        away_on_65 = gt.on_pitch(t65, "away")
        self.assertEqual(len(away_on_65), 10)
        self.assertFalse(gt.is_on_pitch("Wesley Fofana", t65))

    def test_identity_gate_strictly_rejects_hallucinations(self):
        """
        IdentityGate must:
        - Verify active legal player.
        - REJECT player subbed off (Gabriel Martinelli at 65').
        - REJECT player sent off (Wesley Fofana at 65').
        - REJECT player benched who never entered (Gabriel Jesus).
        - REJECT injured / transferred / retired hallucinations (e.g. Thierry Henry, Granit Xhaka, Eden Hazard).
        """
        gt = SportmonksClient.parse_payload_to_ground_truth(self.sportmonks_inplay_payload)
        register_ground_truth(gt)

        t65 = MatchInstant.from_clock(minute=65, period=2)

        # 1. Active legal player -> VERIFIED
        v1, p1, r1 = IdentityGate.verify_identity("Bukayo Saka", t65)
        self.assertTrue(v1)
        self.assertIsNotNone(p1)
        self.assertEqual(p1.name, "Bukayo Saka")

        # 2. Subbed-off player -> REJECTED
        v2, p2, r2 = IdentityGate.verify_identity("Gabriel Martinelli", t65)
        self.assertFalse(v2)
        self.assertIsNone(p2)
        self.assertIn("REJECTED_HALLUCINATION", r2)

        # 3. Sent-off player -> REJECTED
        v3, p3, r3 = IdentityGate.verify_identity("Wesley Fofana", t65)
        self.assertFalse(v3)
        self.assertIsNone(p3)
        self.assertIn("REJECTED_HALLUCINATION", r3)

        # 4. Benched player who hasn't played -> REJECTED
        v4, p4, r4 = IdentityGate.verify_identity("Gabriel Jesus", t65)
        self.assertFalse(v4)
        self.assertIsNone(p4)

        # 5. Hallucinated transferred / injured player -> REJECTED
        v5, p5, r5 = IdentityGate.verify_identity("Thierry Henry", t65)
        self.assertFalse(v5)
        self.assertIsNone(p5)

        v6, p6, r6 = IdentityGate.verify_identity("Eden Hazard", t65)
        self.assertFalse(v6)
        self.assertIsNone(p6)

    def test_api_football_fallback_parser(self):
        """Verifies API-Football response parser operates correctly as fallback."""
        af_payload = {
            "response": [
                {
                    "fixture": {"id": 883921, "status": {"elapsed": 70, "long": "Match Finished"}},
                    "teams": {
                        "home": {"id": 50, "name": "Manchester City"},
                        "away": {"id": 40, "name": "Liverpool"},
                    },
                    "goals": {"home": 1, "away": 1},
                    "lineups": [
                        {
                            "team": {"id": 50, "name": "Manchester City"},
                            "startXI": [
                                {"player": {"id": 617, "name": "Ederson", "number": 31, "pos": "G"}},
                                {"player": {"id": 620, "name": "Erling Haaland", "number": 9, "pos": "F"}},
                            ],
                            "substitutes": [
                                {"player": {"id": 625, "name": "Phil Foden", "number": 47, "pos": "M"}},
                            ],
                        },
                        {
                            "team": {"id": 40, "name": "Liverpool"},
                            "startXI": [
                                {"player": {"id": 280, "name": "Alisson", "number": 1, "pos": "G"}},
                                {"player": {"id": 306, "name": "Mohamed Salah", "number": 11, "pos": "F"}},
                            ],
                            "substitutes": [],
                        },
                    ],
                    "events": [
                        {
                            "time": {"elapsed": 60, "extra": None},
                            "team": {"id": 50, "name": "Manchester City"},
                            "type": "subst",
                            "player": {"id": 620, "name": "Erling Haaland"},
                            "assist": {"id": 625, "name": "Phil Foden"},
                        }
                    ],
                }
            ]
        }

        gt = ApiFootballClient.parse_payload_to_ground_truth(af_payload)
        self.assertEqual(gt.provider, "api_football")
        self.assertEqual(gt.home_team, "Manchester City")
        self.assertEqual(gt.away_team, "Liverpool")

        # Before sub (t=45): Haaland on pitch, Foden on bench
        t45 = MatchInstant.from_clock(45)
        self.assertTrue(gt.is_on_pitch("Erling Haaland", t45))
        self.assertFalse(gt.is_on_pitch("Phil Foden", t45))

        # After sub (t=65): Haaland off pitch, Foden on pitch
        t65 = MatchInstant.from_clock(65)
        self.assertFalse(gt.is_on_pitch("Erling Haaland", t65))
        self.assertTrue(gt.is_on_pitch("Phil Foden", t65))

    def test_roster_manager_resolves_only_verified_on_pitch_players(self):
        """Verifies RosterManager rejects hallucinations using IdentityGate."""
        gt = SportmonksClient.parse_payload_to_ground_truth(self.sportmonks_inplay_payload)
        register_ground_truth(gt)

        rm = RosterManager()
        fixture = rm.get_fixture_data("Arsenal vs Chelsea")

        # 1. Resolve active starter Bukayo Saka at minute 30 -> Resolved
        det_saka = {
            "team": "Arsenal",
            "jersey_number": 7,
            "tactical_role": "RW",
            "physical_traits": "dark short athletic",
            "match_minute": 30,
        }
        res_saka = rm.resolve_player(det_saka, fixture, match_minute=30)
        self.assertEqual(res_saka["name"], "Bukayo Saka")
        self.assertEqual(res_saka["number"], 7)
        self.assertNotIn("hallucination_blocked", res_saka)

        # 2. Resolve subbed-off player (Martinelli at minute 65) -> Hallucination Blocked!
        det_martinelli_subbed = {
            "team": "Arsenal",
            "jersey_number": 11,
            "tactical_role": "LW",
            "physical_traits": "dark short athletic",
            "match_minute": 65,
        }
        res_blocked = rm.resolve_player(det_martinelli_subbed, fixture, match_minute=65)
        self.assertTrue(res_blocked.get("hallucination_blocked", False))
        self.assertEqual(res_blocked["resolved_by"], "unverified_hallucination_blocked")

    def test_active_xi_resolver_binds_live_ground_truth(self):
        """Verifies ActiveXIResolver seamlessly adopts registered GroundTruthDictionary."""
        gt = SportmonksClient.parse_payload_to_ground_truth(self.sportmonks_inplay_payload)
        register_ground_truth(gt)

        resolver = ActiveXIResolver()
        self.assertIsNotNone(resolver.ground_truth)
        self.assertEqual(resolver.match_data["home_team"]["name"], "Arsenal")
        self.assertEqual(len(resolver.match_data["home_team"]["starting_xi"]), 11)

        # At minute 65:
        # Arsenal has 11 players (Trossard in for Martinelli)
        # Chelsea has 10 players (Fofana red carded)
        roster_state = resolver.get_active_players(match_minute=65)
        self.assertEqual(roster_state.home_count, 11)
        self.assertEqual(roster_state.away_count, 10)
        self.assertIn("Leandro Trossard", roster_state.active_on_pitch_players)
        self.assertNotIn("Gabriel Martinelli", roster_state.active_on_pitch_players)
        self.assertNotIn("Wesley Fofana", roster_state.active_on_pitch_players)


if __name__ == "__main__":
    unittest.main()
