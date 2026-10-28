import unittest
import json
from pathlib import Path

from src.rendering.match_stats_provider import MatchStatsProvider
from src.rendering.tactical_narrator import TacticalNarrator
from src.rendering.audio_engine import AudioEngine


class TestVisualCueTagsAndHUD(unittest.TestCase):
    def setUp(self):
        self.stats_provider = MatchStatsProvider()

    def test_extract_cue_tags_basic(self):
        raw_text = "Watch [highlight:Mbappé] as he attacks the channel [candidate_passes] with lethal speed [stat_card:Mbappé]."
        clean_text, tags = TacticalNarrator.extract_cue_tags(raw_text)

        # 1. Spoken text should be clean of all bracket tags
        self.assertEqual(clean_text, "Watch Mbappé as he attacks the channel with lethal speed.")
        self.assertNotIn("[", clean_text)
        self.assertNotIn("]", clean_text)

        # 2. Tag count and types
        self.assertEqual(len(tags), 3)
        self.assertEqual(tags[0]["type"], "highlight")
        self.assertEqual(tags[0]["target"], "Mbappé")
        self.assertEqual(tags[1]["type"], "candidate_passes")
        self.assertEqual(tags[2]["type"], "stat_card")
        self.assertEqual(tags[2]["target"], "Mbappé")

    def test_extract_cue_tags_run_vectors_and_pause(self):
        raw_text = "Notice the pressing trap [run_vectors] closing down [highlight:Dybala] immediately [pause]."
        clean_text, tags = TacticalNarrator.extract_cue_tags(raw_text)

        self.assertEqual(clean_text, "Notice the pressing trap closing down Dybala immediately.")
        self.assertEqual(len(tags), 3)
        self.assertEqual(tags[0]["type"], "run_vectors")
        self.assertEqual(tags[1]["type"], "highlight")
        self.assertEqual(tags[1]["target"], "Dybala")
        self.assertEqual(tags[2]["type"], "pause")

    def test_clean_commentary_text_audio_engine(self):
        engine = AudioEngine()
        raw = "Notice the underlapping run [run_vectors] by [highlight:Valverde] targeting the half-space [stat_card:Valverde]."
        cleaned = engine.clean_commentary_text(raw)
        self.assertNotIn("[", cleaned)
        self.assertNotIn("]", cleaned)
        # Valverde is phonetically overridden to Val-vair-day for Kokoro
        self.assertIn("Val-vair-day", cleaned)

    def test_match_stats_provider_known_players(self):
        # Test Real Madrid player
        stats_mbappe = self.stats_provider.get_player_stats("Mbappé")
        self.assertIsNotNone(stats_mbappe)
        self.assertEqual(stats_mbappe["playerName"], "Kylian Mbappé")
        self.assertEqual(stats_mbappe["number"], 9)
        self.assertEqual(stats_mbappe["team"], "away")
        self.assertEqual(stats_mbappe["clubName"], "Real Madrid")
        self.assertIn("rating", stats_mbappe)
        self.assertIn("heatmap", stats_mbappe)
        self.assertIn("avg_x", stats_mbappe["heatmap"])
        self.assertIn("avg_y", stats_mbappe["heatmap"])

        # Test AS Roma player
        stats_dybala = self.stats_provider.get_player_stats("Paulo Dybala")
        self.assertIsNotNone(stats_dybala)
        self.assertEqual(stats_dybala["playerName"], "Paulo Dybala")
        self.assertEqual(stats_dybala["number"], 21)
        self.assertEqual(stats_dybala["team"], "home")
        self.assertEqual(stats_dybala["clubName"], "AS Roma")

        # Test Konate (Clip 04)
        stats_konate = self.stats_provider.get_player_stats("Konaté")
        self.assertIsNotNone(stats_konate)
        self.assertEqual(stats_konate["playerName"], "Ibrahima Konaté")
        self.assertEqual(stats_konate["team"], "away")

    def test_match_stats_provider_unknown_player_fallback(self):
        stats_unknown = self.stats_provider.get_player_stats("Unknown Striker")
        self.assertIsNotNone(stats_unknown)
        self.assertEqual(stats_unknown["playerName"], "Unknown Striker")
        self.assertEqual(stats_unknown["number"], 10)
        self.assertEqual(stats_unknown["role"], "CM")
        self.assertIn("rating", stats_unknown)
        self.assertIn("stats", stats_unknown)

    def test_fallback_scripts_contain_visual_cue_tags(self):
        narrator = TacticalNarrator()
        for cid in ["clip_01", "clip_02", "clip_03", "clip_04", "clip_05"]:
            script = narrator.generate_3beat_script(
                event_window={"clip_id": cid, "duration_sec": 12.0},
                cause_telestration={},
            )
            self.assertIn("beat_1_setup", script)
            self.assertIn("beat_2_cause", script)
            self.assertIn("beat_3_effect", script)

            # Combined text across beats should contain visual cues
            all_text = " ".join([script[b] if isinstance(script[b], str) else script[b].get("text", "") for b in ["beat_1_setup", "beat_2_cause", "beat_3_effect"]])
            self.assertTrue(
                "[" in all_text and "]" in all_text,
                f"Fallback script for {cid} missing visual cue tags: {all_text}"
            )

    def test_json_serializability(self):
        narrator = TacticalNarrator()
        script = narrator.generate_3beat_script(
            event_window={"clip_id": "clip_04", "duration_sec": 11.5},
            cause_telestration={},
        )
        b2_text = script["beat_2_cause"] if isinstance(script["beat_2_cause"], str) else script["beat_2_cause"].get("text", "")
        _, tags = narrator.extract_cue_tags(b2_text)
        for tag in tags:
            if tag.get("target"):
                tag["player_stats"] = self.stats_provider.get_player_stats(tag["target"])

        serialized = json.dumps(tags)
        deserialized = json.loads(serialized)
        self.assertEqual(len(deserialized), len(tags))


    def test_extract_cue_tags_no_duplicate_name_at_end(self):
        raw = "Watch Tchouaméni as he steps up [highlight:Tchouaméni]."
        clean, tags = TacticalNarrator.extract_cue_tags(raw)
        self.assertEqual(clean, "Watch Tchouaméni as he steps up.")
        self.assertEqual(clean.count("Tchouaméni"), 1)
        self.assertEqual(len(tags), 1)
        self.assertEqual(tags[0]["type"], "highlight")

    def test_extract_cue_tags_no_duplicate_name_at_start(self):
        raw = "[highlight:Mbappé] Mbappé attacks the channel with lethal speed."
        clean, tags = TacticalNarrator.extract_cue_tags(raw)
        self.assertEqual(clean, "Mbappé attacks the channel with lethal speed.")
        self.assertEqual(clean.count("Mbappé"), 1)
        self.assertEqual(len(tags), 1)

    def test_extract_cue_tags_noun_preservation(self):
        raw = "Watch [highlight:Konaté] as he turns in the pocket."
        clean, tags = TacticalNarrator.extract_cue_tags(raw)
        self.assertEqual(clean, "Watch Konaté as he turns in the pocket.")
        self.assertEqual(len(tags), 1)
        self.assertEqual(tags[0]["target"], "Konaté")

    def test_extract_cue_tags_number_and_role_suppression(self):
        raw = "Watch the progression [highlight:#16] into midfield."
        clean, tags = TacticalNarrator.extract_cue_tags(raw)
        self.assertNotIn("#16", clean)
        self.assertNotIn("[", clean)
        self.assertEqual(len(tags), 1)
        self.assertEqual(tags[0]["target"], "#16")

    def test_match_stats_provider_role_resolution(self):
        cdm_stats = self.stats_provider.get_player_stats("cdm", team_hint="Real Madrid")
        self.assertIsNotNone(cdm_stats)
        self.assertEqual(cdm_stats["playerName"], "Aurélien Tchouaméni")

        gk_stats = self.stats_provider.get_player_stats("goalkeeper", team_hint="Real Madrid")
        self.assertIsNotNone(gk_stats)
        self.assertEqual(gk_stats["playerName"], "Thibaut Courtois")

    def test_match_stats_provider_online_fetch(self):
        modric_stats = self.stats_provider.get_player_stats("Luka Modrić")
        self.assertIsNotNone(modric_stats)
        self.assertIn("Modri", modric_stats["playerName"])
        self.assertEqual(modric_stats["position"], "MID")
        self.assertIn("heatmap", modric_stats)

    def test_ensure_clip_visual_cue_tags_populates_all_cues(self):
        from render_final import ensure_clip_visual_cue_tags
        mock_sc = {
            "clip_id": "clip_01_left_wing",
            "cause_frame_idx": 80,
            "cause_telestration": {
                "catalyst_player": {"name": "Ibrahima Konaté"},
                "runner_player": {"name": "Thibaut Courtois"},
                "vectors": [{"type": "PASS_OPTION_DASHED"}],
            },
            "beats": {
                "beat_1_setup": {"text": "Moving buildup"},
                "beat_2_cause": {"text": "Composed pass", "freeze_frames": 150},
                "beat_3_effect": {"text": "Play restarts"},
            },
        }
        tags = ensure_clip_visual_cue_tags(mock_sc, mock_sc, "clip_01_left_wing")
        self.assertTrue(len(tags) >= 2)
        types = [t["type"] for t in tags]
        self.assertIn("highlight", types)
        self.assertIn("stat_card", types)
        # Verify player stats are attached
        stat_tag = next(t for t in tags if t["type"] == "stat_card")
        self.assertIsNotNone(stat_tag.get("player_stats"))
        self.assertEqual(stat_tag["player_stats"]["playerName"], "Ibrahima Konaté")


if __name__ == "__main__":
    unittest.main()

