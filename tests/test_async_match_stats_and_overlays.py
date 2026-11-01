"""
tests/test_async_match_stats_and_overlays.py
Comprehensive Verification Suite for:
1. Asynchronous Match Statistics Scraper (ScraperFC / Sofascore / Fotmob DataFrames)
2. Accurate Algorithmic Match Ratings (No AI Hallucinations)
3. Direct Mapping of Discrete Integer Events (Goals, Cards, Assists, xG)
4. Dynamic UI Graphic Overlays (Automatic Football PNG icons on player card)
"""

import asyncio
import os
import unittest
from PIL import Image, ImageDraw

import pandas as pd

from src.scraper.async_match_stats import AsyncMatchStatsScraper, PlayerOfficialStats
from src.rendering.match_stats_provider import MatchStatsProvider
from vision_pipeline.tactical_renderer import TacticalRenderer


class TestAsyncMatchStatsAndOverlays(unittest.TestCase):
    """Test suite for async stats scraping and dynamic graphic generator overlays."""

    def setUp(self):
        self.scraper = AsyncMatchStatsScraper()
        self.stats_provider = MatchStatsProvider()
        self.renderer = TacticalRenderer()

        # Deterministic Sofascore event lineups payload with official ratings & discrete events
        self.sofascore_lineup_payload = {
            "home": {
                "team": {"id": 2817, "name": "Barcelona"},
                "players": [
                    {
                        "player": {"id": 990011, "name": "Pedri", "shortName": "Pedri", "position": "M", "jerseyNumber": 8},
                        "statistics": {
                            "rating": 8.4,
                            "goals": 1,
                            "goalAssist": 1,
                            "yellowCards": 0,
                            "redCards": 0,
                            "expectedGoals": 0.42,
                            "expectedAssists": 0.68,
                            "totalPass": 72,
                            "accuratePass": 67,
                            "interceptionWon": 3,
                            "duelWon": 7,
                            "duelLost": 2,
                            "minutesPlayed": 90,
                        },
                    },
                    {
                        "player": {"id": 990012, "name": "Robert Lewandowski", "shortName": "Lewandowski", "position": "F", "jerseyNumber": 9},
                        "statistics": {
                            "rating": 8.9,
                            "goals": 2,  # Exactly 2 discrete goals
                            "goalAssist": 0,
                            "yellowCards": 1,
                            "redCards": 0,
                            "expectedGoals": 1.54,
                            "expectedAssists": 0.12,
                            "totalPass": 22,
                            "accuratePass": 18,
                            "interceptionWon": 0,
                            "duelWon": 4,
                            "duelLost": 3,
                            "minutesPlayed": 82,
                        },
                    },
                ],
            },
            "away": {
                "team": {"id": 2829, "name": "Real Madrid"},
                "players": [
                    {
                        "player": {"id": 880011, "name": "Kylian Mbappé", "shortName": "Mbappé", "position": "F", "jerseyNumber": 9},
                        "statistics": {
                            "rating": 8.5,
                            "goals": 1,
                            "goalAssist": 0,
                            "yellowCards": 0,
                            "redCards": 0,
                            "expectedGoals": 0.88,
                            "expectedAssists": 0.20,
                            "totalPass": 30,
                            "accuratePass": 26,
                            "interceptionWon": 1,
                            "duelWon": 5,
                            "duelLost": 4,
                            "minutesPlayed": 90,
                        },
                    },
                    {
                        "player": {"id": 880012, "name": "Antonio Rüdiger", "shortName": "Rüdiger", "position": "D", "jerseyNumber": 22},
                        "statistics": {
                            "rating": 7.4,
                            "goals": 0,
                            "goalAssist": 0,
                            "yellowCards": 1,
                            "redCards": 0,
                            "expectedGoals": 0.04,
                            "expectedAssists": 0.00,
                            "totalPass": 58,
                            "accuratePass": 53,
                            "interceptionWon": 4,
                            "duelWon": 8,
                            "duelLost": 1,
                            "minutesPlayed": 90,
                        },
                    },
                ],
            },
        }

    def test_parse_sofascore_lineup_to_dataframe(self):
        """Verifies Sofascore payload parses into detailed DataFrames without hallucinated ratings."""
        df_players, df_events = self.scraper.parse_sofascore_payload(self.sofascore_lineup_payload)

        self.assertIsInstance(df_players, pd.DataFrame)
        self.assertFalse(df_players.empty)
        self.assertEqual(len(df_players), 4)

        # 1. Official Algorithmic Match Rating for Pedri (must be exactly 8.4, not random guess)
        pedri_row = df_players[df_players["playerName"] == "Pedri"].iloc[0]
        self.assertEqual(pedri_row["rating"], 8.4)
        self.assertEqual(pedri_row["ratingColor"], "#22c55e")
        self.assertEqual(pedri_row["goals"], 1)
        self.assertEqual(pedri_row["assists"], 1)
        self.assertAlmostEqual(pedri_row["xg"], 0.42, places=2)

        # 2. Official Discrete Events for Lewandowski (must be exactly 2 goals, 1 yellow card)
        lewy_row = df_players[df_players["playerName"] == "Robert Lewandowski"].iloc[0]
        self.assertEqual(lewy_row["rating"], 8.9)
        self.assertEqual(lewy_row["goals"], 2)
        self.assertEqual(lewy_row["yellow_cards"], 1)
        self.assertEqual(lewy_row["cards"], 1)
        self.assertAlmostEqual(lewy_row["xg"], 1.54, places=2)

        # Event badges list for UI
        lewy_badges = lewy_row["event_badges"]
        goal_badge = next((b for b in lewy_badges if b["type"] == "goal"), None)
        self.assertIsNotNone(goal_badge)
        self.assertEqual(goal_badge["count"], 2)
        self.assertEqual(goal_badge["icon"], "football.png")

    def test_async_background_scrape_execution(self):
        """Verifies scraping runs asynchronously in background alongside processing."""
        future = self.scraper.start_background_scrape("el_clasico")
        # Ensure future returns without blocking
        df_players, df_events = future.result(timeout=5.0)
        self.assertIsInstance(df_players, pd.DataFrame)
        self.assertGreaterEqual(len(df_players), 6)

        # Test asyncio coroutine
        async def _test_coroutine():
            return await self.scraper.fetch_match_stats_async("el_clasico")

        df_async, _ = asyncio.run(_test_coroutine())
        self.assertFalse(df_async.empty)

    def test_match_stats_provider_retrieves_official_ratings(self):
        """Verifies MatchStatsProvider maps official DataFrame stats to player queries."""
        # Query Pedri
        pedri = self.stats_provider.get_player_stats("Pedri")
        self.assertIsNotNone(pedri)
        self.assertEqual(pedri["rating"], 8.4)
        self.assertEqual(pedri["ratingColor"], "#22c55e")
        self.assertEqual(pedri["goals"], 1)
        self.assertEqual(pedri["assists"], 0)

        # Query Lewandowski (2 goals)
        lewy = self.stats_provider.get_player_stats("Robert Lewandowski")
        self.assertIsNotNone(lewy)
        self.assertEqual(lewy["rating"], 8.8)
        self.assertEqual(lewy["goals"], 2)
        self.assertEqual(lewy["cards"], 0)

    def test_dynamic_ui_overlay_draws_two_football_icons_for_two_goals(self):
        """
        Critical Prompt Requirement:
        If the JSON says 2 goals, the UI generator must automatically overlay 2 football PNG icons
        on that specific player's card.
        """
        # Create blank image to render on
        img = Image.new("RGBA", (800, 600), (20, 25, 35, 255))
        draw = ImageDraw.Draw(img)

        # Player with exactly 2 goals from official API data
        lewy_stats = {
            "name": "Robert Lewandowski",
            "playerName": "Robert Lewandowski",
            "number": 9,
            "position": "CF",
            "role": "CF",
            "clubName": "Barcelona",
            "team": "home",
            "rating": 8.8,
            "goals": 2,  # 2 discrete goals from API
            "yellow_cards": 1,
            "red_cards": 0,
            "xg": 1.48,
            "passes": "16/21 (76%)",
            "stats": {"Duels Won": "3/6", "Interceptions": "0"},
        }

        # 1. Render HUD badge with 2 goals
        b_info = {
            "x": 400,
            "y": 250,
            "head_x": 400,
            "head_y": 270,
            "player": lewy_stats,
            "team_color": {"primary": (0, 77, 152), "accent": (237, 187, 0), "text": (255, 255, 255)},
        }
        self.renderer._draw_player_hud_badge(draw, b_info, overlay_img=img)

        # 2. Render Broadcast Stat Card modal overlay with 2 goals
        card_res = self.renderer.draw_broadcast_player_stat_card(
            draw=draw,
            player_stats=lewy_stats,
            x=200,
            y=320,
            card_w=360,
            card_h=160,
            overlay_img=img,
        )

        self.assertEqual(card_res["goals_rendered"], 2)
        self.assertEqual(card_res["yellow_cards_rendered"], 1)
        self.assertEqual(card_res["rating_rendered"], 8.8)

        # Save output verification image
        out_file = os.path.join(self.renderer.output_dir, "test_dynamic_stat_card_overlay.png")
        img.convert("RGB").save(out_file)
        self.assertTrue(os.path.exists(out_file))

    def test_single_goal_overlays_one_football_icon(self):
        """Verifies 1 goal renders exactly 1 football icon, while 0 goals renders 0 football icons."""
        img = Image.new("RGBA", (500, 300), (20, 25, 35, 255))
        draw = ImageDraw.Draw(img)

        # Mbappé with 1 goal
        mbappe_stats = {
            "playerName": "Kylian Mbappé",
            "number": 9,
            "position": "CF",
            "role": "CF",
            "rating": 8.5,
            "goals": 1,
            "yellow_cards": 0,
            "red_cards": 0,
        }
        card_1 = self.renderer.draw_broadcast_player_stat_card(
            draw=draw, player_stats=mbappe_stats, x=50, y=50, card_w=320, card_h=150, overlay_img=img
        )
        self.assertEqual(card_1["goals_rendered"], 1)

        # Courtois with 0 goals
        courtois_stats = {
            "playerName": "Thibaut Courtois",
            "number": 1,
            "position": "GK",
            "role": "GK",
            "rating": 8.2,
            "goals": 0,
            "yellow_cards": 0,
            "red_cards": 0,
        }
        card_0 = self.renderer.draw_broadcast_player_stat_card(
            draw=draw, player_stats=courtois_stats, x=50, y=50, card_w=320, card_h=150, overlay_img=img
        )
        self.assertEqual(card_0["goals_rendered"], 0)


if __name__ == "__main__":
    unittest.main()
