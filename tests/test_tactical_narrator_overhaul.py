"""
tests/test_tactical_narrator_overhaul.py
Comprehensive verification test for TacticalNarrator commentary overhaul:
- Zero 'Backed by...' openers
- Zero cross-team passing (Real Madrid to Barcelona or vice versa)
- Zero phantom players (Pisilli, Malen, Dumfries, Konaté, Ndicka)
- Jargon capping (max 1 occurrence per clip)
- Zero robotic clichés ('Unfreezing into motion', 'using precise telemetry')
- Zero static video-game ratings in speech
"""

import unittest
from src.rendering.tactical_narrator import TacticalNarrator, _normalize_name


class TestTacticalNarratorOverhaul(unittest.TestCase):

    def setUp(self):
        self.narrator = TacticalNarrator(api_key="")

    def test_roster_resolution(self):
        roster = TacticalNarrator._resolve_match_roster("Barcelona", "Real Madrid")
        self.assertEqual(roster["home_club"], "FC Barcelona")
        self.assertEqual(roster["away_club"], "Real Madrid")

        lookup = roster["all_players_by_team"]
        self.assertEqual(lookup.get("tchouameni"), "Real Madrid")
        self.assertEqual(lookup.get("lewandowski"), "FC Barcelona")
        self.assertEqual(lookup.get("pedri"), "FC Barcelona")
        self.assertEqual(lookup.get("mbappe"), "Real Madrid")
        self.assertEqual(lookup.get("lunin"), "Real Madrid")
        self.assertEqual(lookup.get("courtois"), "Real Madrid")
        self.assertEqual(lookup.get("rudiger"), "Real Madrid")
        self.assertEqual(lookup.get("cubarsi"), "FC Barcelona")

    def test_banned_backed_by_elimination(self):
        beats = {
            "beat_1_setup": "Opening whistle sounds at Camp Nou.",
            "beat_2_cause": "Backed by ninety-three percent pass accuracy, Tchouaméni steps forward into space.",
            "beat_3_effect": "Play resets smoothly as structure holds firm.",
        }
        processed = TacticalNarrator._post_process_beats(beats, home_team="Barcelona", away_team="Real Madrid")
        b2 = processed["beat_2_cause"]
        self.assertNotIn("Backed by", b2)
        self.assertNotIn("backed by", b2)
        self.assertIn("Tchouaméni", b2)

    def test_cross_team_pass_elimination(self):
        beats = {
            "beat_1_setup": "Zone fourteen breached before halftime whistle.",
            "beat_2_cause": "Tchouaméni exploits vacant space, sliding a lethal ball through for Lewandowski.",
            "beat_3_effect": "Clinical finish buries the chance past helpless goalkeeper.",
        }
        processed = TacticalNarrator._post_process_beats(beats, home_team="Barcelona", away_team="Real Madrid")
        b2 = processed["beat_2_cause"]
        b3 = processed["beat_3_effect"]

        # Tchouaméni (Real Madrid) must NOT pass to Lewandowski (Barcelona)
        self.assertNotIn("Lewandowski", b2)
        self.assertIn("Mbappé", b2)
        # Cliché finish must be replaced
        self.assertNotIn("helpless goalkeeper", b3)

    def test_phantom_player_eradication(self):
        beats = {
            "beat_1_setup": "Watch the midfield transition unfold.",
            "beat_2_cause": "Pisilli works with Malen, while Dumfries tracks back on the right.",
            "beat_3_effect": "Konaté and Ndicka clear the danger.",
        }
        processed = TacticalNarrator._post_process_beats(beats, home_team="Barcelona", away_team="Real Madrid")
        all_text = f"{processed['beat_1_setup']} {processed['beat_2_cause']} {processed['beat_3_effect']}"

        self.assertNotIn("Pisilli", all_text)
        self.assertNotIn("Malen", all_text)
        self.assertNotIn("Dumfries", all_text)
        self.assertNotIn("Konaté", all_text)
        self.assertNotIn("Ndicka", all_text)

    def test_robotic_cliche_and_static_rating_removal(self):
        beats = {
            "beat_1_setup": "Balde replaces injured Fort on left.",
            "beat_2_cause": "Lewandowski triggers the shift, backed by seventy-five rating and three key passes, using precise telemetry.",
            "beat_3_effect": "Unfreezing into motion, Barcelona immediately reorganizes.",
        }
        processed = TacticalNarrator._post_process_beats(beats, home_team="Barcelona", away_team="Real Madrid")
        all_text = f"{processed['beat_1_setup']} {processed['beat_2_cause']} {processed['beat_3_effect']}"

        self.assertNotIn("seventy-five rating", all_text)
        self.assertNotIn("using precise telemetry", all_text)
        self.assertNotIn("Unfreezing into motion", all_text)

    def test_jargon_budget_cap(self):
        beats = {
            "beat_1_setup": "Watch the flank corridor open here.",
            "beat_2_cause": "The carrier drives into the flank corridor, while the defensive block splits to open another corridor.",
            "beat_3_effect": "The defensive block is beaten by the corridor pass.",
        }
        processed = TacticalNarrator._post_process_beats(beats, home_team="Barcelona", away_team="Real Madrid")
        all_text = f"{processed['beat_1_setup']} {processed['beat_2_cause']} {processed['beat_3_effect']}".lower()

        # Each buzzword should occur AT MOST 1 time
        self.assertLessEqual(all_text.count("flank corridor"), 1)
        self.assertLessEqual(all_text.count("defensive block"), 1)

    def test_deterministic_fallback_generation(self):
        event_window = {
            "clip_id": "clip_03_goal",
            "thesis": "Central half-space penetration",
            "narrative_tone": "TACTICAL_MASTERCLASS",
            "tactical_zone": "zone_14",
            "t_cause_clock": "44:12",
            "home_team": "Barcelona",
            "away_team": "Real Madrid",
        }
        cause_tel = {
            "clip_id": "clip_03_goal",
            "catalyst_player": {"name": "Fermín López", "team": "home"},
            "runner_player": {"name": "Robert Lewandowski", "team": "home"},
        }
        script = self.narrator.generate_3beat_script(
            event_window=event_window,
            cause_telestration=cause_tel,
        )
        self.assertIn("beat_1_setup", script)
        self.assertIn("beat_2_cause", script)
        self.assertIn("beat_3_effect", script)

        b2 = script["beat_2_cause"]
        self.assertNotIn("Backed by", b2)
        self.assertNotIn("backed by", b2)
        self.assertNotIn("Unfreezing", script["beat_3_effect"])


if __name__ == "__main__":
    unittest.main()
