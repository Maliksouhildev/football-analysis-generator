"""
src/scraper/test_standalone.py
Standalone Test Suite for Free Crowd Consensus Scraper & Cause-to-Effect Evidence Matcher.

Verifies:
1. MatchLineupResolver dynamic parsing & chronological substitutions (zero hardcoded rosters).
2. CrowdConsensusScraper VADER sentiment, tone routing, and KMeans(5) centroid extraction (<300 tokens).
3. EvidenceMatcher 10s-18s backward cause tracing and strict TACTICAL_CONTRACT.md EventWindowSchema validation.
"""

import json
import re
import unittest
from typing import Any, Dict, List

from src.scraper.lineup_scraper import MatchLineupResolver
from src.scraper.crowd_consensus import CrowdConsensusScraper, CrowdConsensusResult
from src.scraper.evidence_matcher import EvidenceMatcher, GeminiFailoverPool, TacticalThesis


class TestTacticalContractValidation(unittest.TestCase):
    """Verifies strict adherence to TACTICAL_CONTRACT.md EventWindowSchema."""

    @classmethod
    def setUpClass(cls):
        # Sample matchday JSON adhering to SofaScore/FotMob schemas
        cls.sample_matchday_json = {
            "event": {
                "homeTeam": {"name": "AS Roma", "teamColors": {"primary": "#8a1529"}},
                "awayTeam": {"name": "Real Madrid", "teamColors": {"primary": "#ffffff"}},
            },
            "lineups": {
                "home": {
                    "players": [
                        {"player": {"name": "Svilar", "jerseyNumber": 99}, "position": "GK", "substitute": False},
                        {"player": {"name": "Karsdorp", "jerseyNumber": 2}, "position": "RB", "substitute": False},
                        {"player": {"name": "Mancini", "jerseyNumber": 23}, "position": "CB", "substitute": False},
                        {"player": {"name": "Ndicka", "jerseyNumber": 5}, "position": "CB", "substitute": False},
                        {"player": {"name": "Angelino", "jerseyNumber": 3}, "position": "LB", "substitute": False},
                        {"player": {"name": "Cristante", "jerseyNumber": 4}, "position": "DM", "substitute": False},
                        {"player": {"name": "Kone", "jerseyNumber": 17}, "position": "CM", "substitute": False},
                        {"player": {"name": "Pellegrini", "jerseyNumber": 7}, "position": "AM", "substitute": False},
                        {"player": {"name": "Soule", "jerseyNumber": 18}, "position": "RW", "substitute": False},
                        {"player": {"name": "El Shaarawy", "jerseyNumber": 92}, "position": "LW", "substitute": False},
                        {"player": {"name": "Dovbyk", "jerseyNumber": 11}, "position": "CF", "substitute": False},
                    ]
                },
                "away": {
                    "players": [
                        {"player": {"name": "Courtois", "jerseyNumber": 1}, "position": "GK", "substitute": False},
                        {"player": {"name": "Vazquez", "jerseyNumber": 17}, "position": "RB", "substitute": False},
                        {"player": {"name": "Rudiger", "jerseyNumber": 22}, "position": "CB", "substitute": False},
                        {"player": {"name": "Tchouameni", "jerseyNumber": 14}, "position": "CB", "substitute": False},
                        {"player": {"name": "Mendy", "jerseyNumber": 23}, "position": "LB", "substitute": False},
                        {"player": {"name": "Valverde", "jerseyNumber": 8}, "position": "CM", "substitute": False},
                        {"player": {"name": "Camavinga", "jerseyNumber": 6}, "position": "DM", "substitute": False},
                        {"player": {"name": "Bellingham", "jerseyNumber": 5}, "position": "AM", "substitute": False},
                        {"player": {"name": "Rodrygo", "jerseyNumber": 11}, "position": "RW", "substitute": False},
                        {"player": {"name": "Vinicius Jr", "jerseyNumber": 7}, "position": "LW", "substitute": False},
                        {"player": {"name": "Mbappe", "jerseyNumber": 9}, "position": "CF", "substitute": False},
                    ]
                },
            },
            "incidents": [
                {
                    "incidentType": "substitution",
                    "time": 62,
                    "isHome": True,
                    "playerIn": {"name": "Baldanzi", "jerseyNumber": 35, "position": "RW"},
                    "playerOut": {"name": "Soule", "jerseyNumber": 18},
                },
                {
                    "incidentType": "substitution",
                    "time": 75,
                    "isHome": False,
                    "playerIn": {"name": "Ceballos", "jerseyNumber": 19, "position": "CM"},
                    "playerOut": {"name": "Camavinga", "jerseyNumber": 6},
                },
            ],
        }

        # Play-by-play events with discrete precursor causes
        cls.sample_events = [
            # Event 1: Minute 24: Shot preceded by broken press
            {"period": 1, "minute": 23, "second": 48, "type": "broken_press", "pitch_x": 42.0, "pitch_y": 50.0, "player": "Valverde"},
            {"period": 1, "minute": 24, "second": 4, "type": "shot", "pitch_x": 88.0, "pitch_y": 55.0, "player": "Mbappe"},
            # Event 2: Minute 64: Turnover in defensive third leading to big chance
            {"period": 2, "minute": 63, "second": 46, "type": "turnover", "pitch_x": 28.0, "pitch_y": 32.0, "player": "Kone"},
            {"period": 2, "minute": 64, "second": 2, "type": "shot", "pitch_x": 89.0, "pitch_y": 48.0, "player": "Rodrygo"},
            # Event 3: Minute 84: Goal preceded by flank overload 16 seconds prior (contract example: 83:56 -> 84:12)
            {"period": 2, "minute": 83, "second": 56, "type": "pass", "pitch_x": 72.0, "pitch_y": 88.0, "player": "Mendy"},
            {"period": 2, "minute": 84, "second": 12, "type": "goal", "pitch_x": 92.0, "pitch_y": 54.0, "player": "Vinicius Jr"},
        ]

    # -------------------------------------------------------------------------
    # 1. MatchLineupResolver Tests
    # -------------------------------------------------------------------------

    def test_lineup_resolver_has_no_hardcoded_players(self):
        """Verify FIXTURE_DATABASE and hardcoded player blocks are deleted."""
        import src.scraper.lineup_scraper as ls_module
        self.assertFalse(hasattr(ls_module, "FIXTURE_DATABASE"), "FIXTURE_DATABASE must be completely removed")
        with open(ls_module.__file__, "r", encoding="utf-8") as f:
            code = f.read()
        self.assertNotIn("Militão", code)
        self.assertNotIn("De Roon", code)
        self.assertNotIn("galatasaray_vs_barcelona", code)

    def test_lineup_resolver_chronological_substitutions(self):
        """Verify starters, jersey numbers, and substitutions at (period, minute)."""
        # Pre-substitution at minute 50
        lineup_min50 = MatchLineupResolver.get_active_lineup(
            matchday_data=self.sample_matchday_json,
            period=2,
            match_minute=50,
        )
        home_players_min50 = [p["name"] for p in lineup_min50["home_active_xi"]]
        self.assertIn("Soule", home_players_min50)
        self.assertNotIn("Baldanzi", home_players_min50)

        # Post-substitution at minute 65
        lineup_min65 = MatchLineupResolver.get_active_lineup(
            matchday_data=self.sample_matchday_json,
            period=2,
            match_minute=65,
        )
        home_players_min65 = [p["name"] for p in lineup_min65["home_active_xi"]]
        self.assertIn("Baldanzi", home_players_min65)
        self.assertNotIn("Soule", home_players_min65)

        # Post-substitution at minute 80 for away team
        lineup_min80 = MatchLineupResolver.get_active_lineup(
            matchday_data=self.sample_matchday_json,
            period=2,
            match_minute=80,
        )
        away_players_min80 = [p["name"] for p in lineup_min80["away_active_xi"]]
        self.assertIn("Ceballos", away_players_min80)
        self.assertNotIn("Camavinga", away_players_min80)

        # Check active roster format
        roster = MatchLineupResolver.resolve_active_roster(
            matchday_data=self.sample_matchday_json,
            period=2,
            minute=84,
        )
        self.assertIn("home", roster)
        self.assertIn("away", roster)
        for p in roster["home"] + roster["away"]:
            self.assertIn("num", p)
            self.assertIn("name", p)
            self.assertIn("role", p)
            self.assertIsInstance(p["num"], int)

    # -------------------------------------------------------------------------
    # 2. Crowd Consensus Scraper Tests (Free, Zero LLM Tokens)
    # -------------------------------------------------------------------------

    def test_crowd_consensus_sentiment_and_tone(self):
        """Verify VADER sentiment thresholds (<= -0.25 -> RUTHLESS_AUTOPSY, >= 0.25 -> TACTICAL_MASTERCLASS)."""
        negative_comments = [
            "Horrific defensive setup tonight, completely exposed the right back in constant two versus one overloads.",
            "Midfield was utterly dismantled, unable to make three consecutive progressive passes without turning it over constantly.",
            "Winger completely gave up tracking back, walking while their full back created endless dangerous crosses into the box.",
            "Terrible tactical substitutions in the second half destroyed all team shape and conceded easy goals inside our box.",
            "Zero defensive awareness, pathetic press resistance under pressure and we thoroughly deserved this absolute humiliation completely.",
        ]
        result_neg = CrowdConsensusScraper.analyze(negative_comments)
        self.assertEqual(result_neg.narrative_tone, "RUTHLESS_AUTOPSY")
        self.assertLessEqual(result_neg.team_sentiment_score, -0.25)
        self.assertEqual(len(result_neg.centroid_comments), 5)

        positive_comments = [
            "Magnificent tactical display, the high pressing structure completely suffocated their build-up play beautifully.",
            "Sensational defensive discipline and glorious transitions on the counter attack created wonderful chances tonight.",
            "Absolute masterclass from our manager, neutralizing their key playmaker with perfection across ninety minutes.",
            "Brilliant combination play in the attacking third with continuous overloads breaking their defensive block down.",
            "Flawless game management, excellent progressive passing and a thoroughly deserved triumphant victory.",
        ]
        result_pos = CrowdConsensusScraper.analyze(positive_comments)
        self.assertEqual(result_pos.narrative_tone, "TACTICAL_MASTERCLASS")
        self.assertGreaterEqual(result_pos.team_sentiment_score, 0.25)
        self.assertEqual(len(result_pos.centroid_comments), 5)

    def test_crowd_consensus_word_filter_and_token_budget(self):
        """Verify >= 15 words filter and < 300 token budget for centroids."""
        mixed_comments = [
            "Bad game.",  # 2 words -> filtered out
            "LOL what was that",  # 4 words -> filtered out
            "Nice goal!",  # 2 words -> filtered out
            "The defensive block was repeatedly stretched laterally under relentless pressure because wide recovery runs were completely abandoned.",  # 19 words -> kept
            "Midfield coordination broke down completely in the second half leading to repeated dangerous turnovers right in front of the box.",  # 20 words -> kept
            "Excellent pressing trap in Zone 14 forced their defenders into rushed clearances and gave us dominant possession throughout.",  # 17 words -> kept
            "Defensive structure held solid while our rapid counter attacks through the wide channels tore their back four apart continuously.",  # 18 words -> kept
            "Superb tactical flexibility switching from a back four to a back five during defensive transition phases under intense pressure.",  # 18 words -> kept
        ]
        result = CrowdConsensusScraper.analyze(mixed_comments, min_words=15, n_clusters=5)
        self.assertEqual(result.filtered_comments_count, 5)
        self.assertEqual(len(result.centroid_comments), 5)

        # Verify token budget is strictly < 300 tokens (< 250 words total)
        total_words = sum(len(c.split()) for c in result.centroid_comments)
        self.assertLess(total_words, 250, "Centroid comments should easily fit in <300 token budget")

    # -------------------------------------------------------------------------
    # 3. Evidence Matcher & TACTICAL_CONTRACT.md Validation
    # -------------------------------------------------------------------------

    def test_evidence_matcher_schema_and_cause_window(self):
        """
        Verify exact EventWindowSchema defined in TACTICAL_CONTRACT.md:
        - clip_id: str
        - period: int
        - t_cause_clock: str ("MM:SS")
        - t_effect_clock: str ("MM:SS")
        - narrative_tone: "RUTHLESS_AUTOPSY" | "TACTICAL_MASTERCLASS"
        - thesis: str
        - crowd_evidence: str
        - active_roster: {"home": [{"num": int, "name": str, "role": str}], "away": [...]}
        - Duration (t_effect - t_cause) in [10, 18] seconds!
        """
        negative_comments = [
            "Defensive block was repeatedly stretched laterally due to aggressive flank overload.",
            "Continuous turnovers right in the defensive third gifted the opposition multiple big scoring opportunities.",
            "Midfield pocket in Zone 14 was left completely vacant allowing progressive line breaking vertical passes.",
            "Defensive transitions were non existent, allowing rapid counter attacks into our penalty area continuously.",
            "Winger walked back while opponent full back overlapped freely creating uncontested crosses into the box.",
        ]
        crowd_res = CrowdConsensusScraper.analyze(negative_comments)

        matcher = EvidenceMatcher()
        windows = matcher.generate_event_windows(
            crowd_result=crowd_res,
            events=self.sample_events,
            matchday_data=self.sample_matchday_json,
        )

        self.assertGreaterEqual(len(windows), 4, "Must output between 4 and 6 Tactical Theses windows")
        self.assertLessEqual(len(windows), 6)

        clock_regex = re.compile(r"^\d{2}:\d{2}$")

        for w in windows:
            # 1. Schema key presence
            self.assertIn("clip_id", w)
            self.assertIn("period", w)
            self.assertIn("t_cause_clock", w)
            self.assertIn("t_effect_clock", w)
            self.assertIn("narrative_tone", w)
            self.assertIn("thesis", w)
            self.assertIn("crowd_evidence", w)
            self.assertIn("active_roster", w)

            # 2. Types & values
            self.assertIsInstance(w["clip_id"], str)
            self.assertIn(w["period"], (1, 2))
            self.assertTrue(clock_regex.match(w["t_cause_clock"]), f"Invalid clock format: {w['t_cause_clock']}")
            self.assertTrue(clock_regex.match(w["t_effect_clock"]), f"Invalid clock format: {w['t_effect_clock']}")
            self.assertIn(w["narrative_tone"], ("RUTHLESS_AUTOPSY", "TACTICAL_MASTERCLASS"))
            self.assertTrue(len(w["thesis"]) > 5)
            self.assertTrue(len(w["crowd_evidence"]) > 5)

            # 3. Cause-to-Effect backward duration (10s to 18s window constraint)
            cause_m, cause_s = map(int, w["t_cause_clock"].split(":"))
            effect_m, effect_s = map(int, w["t_effect_clock"].split(":"))
            cause_total_sec = cause_m * 60 + cause_s
            effect_total_sec = effect_m * 60 + effect_s
            delta_sec = effect_total_sec - cause_total_sec

            self.assertGreaterEqual(
                delta_sec, 10,
                f"Window duration {delta_sec}s is less than 10s ({w['t_cause_clock']} -> {w['t_effect_clock']})"
            )
            self.assertLessEqual(
                delta_sec, 18,
                f"Window duration {delta_sec}s exceeds 18s ({w['t_cause_clock']} -> {w['t_effect_clock']})"
            )

            # 4. Active Roster schema strictly matching TACTICAL_CONTRACT.md
            roster = w["active_roster"]
            self.assertIsInstance(roster, dict)
            self.assertIn("home", roster)
            self.assertIn("away", roster)
            self.assertIsInstance(roster["home"], list)
            self.assertIsInstance(roster["away"], list)

            for p in roster["home"] + roster["away"]:
                self.assertIn("num", p)
                self.assertIn("name", p)
                self.assertIn("role", p)
                self.assertIsInstance(p["num"], int)
                self.assertIsInstance(p["name"], str)
                self.assertIsInstance(p["role"], str)

        # Test JSON serialization
        serialized = json.dumps(windows, indent=2)
        parsed = json.loads(serialized)
        self.assertEqual(len(parsed), len(windows))


if __name__ == "__main__":
    unittest.main()
