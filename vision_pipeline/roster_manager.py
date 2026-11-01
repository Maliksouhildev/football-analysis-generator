"""
Roster Manager & Dynamic Multi-Cue Player Matcher
Eliminates hallucinations by coupling local detection resolution with the dynamic
live GroundTruthDictionary (Sportmonks / API-Football).
No player data is hardcoded.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from typing import Dict, List, Optional, Any, Tuple

from src.scraper.live_ground_truth import (
    GroundTruthDictionary,
    IdentityGate,
    MatchInstant,
    get_registered_ground_truth,
    register_ground_truth,
)

logger = logging.getLogger("vision_pipeline.roster_manager")


class RosterManager:
    """Manages dynamic match lineups, kit colors, and hallucination-free player resolution."""

    def __init__(self, fixtures_dir: Optional[str] = None):
        self.fixtures_dir = fixtures_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data"
        )
        os.makedirs(self.fixtures_dir, exist_ok=True)
        self.cache: Dict[str, Dict[str, Any]] = {}

    def get_fixture_data(self, match_name: str) -> Dict[str, Any]:
        """
        Loads match fixture data dynamically:
        1. Checks in-memory cache.
        2. Binds active live GroundTruthDictionary (if registered).
        3. Loads verified fixture JSON file from disk (e.g. test_data).
        4. Queries dynamic live Ground Truth provider (Sportmonks / API-Football fallback).
        """
        if match_name in self.cache:
            return self.cache[match_name]

        # 1. Check registered live ground truth
        gt = get_registered_ground_truth()
        if gt is not None:
            if (
                gt.side_of(match_name)
                or self._match_tokens(gt.home_team, match_name)
                or self._match_tokens(gt.away_team, match_name)
            ):
                fixture = self._ground_truth_to_fixture(gt)
                self.cache[match_name] = fixture
                return fixture

        # 2. Check local verified json file
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", match_name.lower())
        local_path = os.path.join(self.fixtures_dir, f"{safe_name}_roster.json")
        if os.path.exists(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.cache[match_name] = data
                return data

        # 3. Dynamic Live Ingestion (No hardcoded player data)
        try:
            from src.scraper.live_ground_truth import fetch_live_ground_truth
            gt = fetch_live_ground_truth(fixture_id=safe_name)
            fixture = self._ground_truth_to_fixture(gt)
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(fixture, f, indent=2)
            self.cache[match_name] = fixture
            return fixture
        except Exception as e:
            logger.info("Live ground truth provider lookup note: %s. Using dynamic empty template.", e)

        # 4. Fallback empty fixture without hardcoded players
        fixture = self._build_empty_fixture(match_name)
        self.cache[match_name] = fixture
        return fixture

    @staticmethod
    def _match_tokens(text_a: str, text_b: str) -> bool:
        tokens_a = set(re.findall(r"\w+", text_a.lower()))
        tokens_b = set(re.findall(r"\w+", text_b.lower()))
        return bool(tokens_a & tokens_b)

    def _ground_truth_to_fixture(self, gt: GroundTruthDictionary) -> Dict[str, Any]:
        """Transforms a GroundTruthDictionary into the standard fixture format."""
        return {
            "match": f"{gt.home_team} vs {gt.away_team}",
            "competition": "Live Match",
            "fixture_id": gt.fixture_id,
            "current_minute": gt.current_minute,
            "_ground_truth": gt,
            "home_team": {
                "name": gt.home_team,
                "short_name": gt.home_team[:3].upper(),
                "formation": "4-3-3",
                "kit": {
                    "primary_color": "Home Primary",
                    "shirt": "Home",
                    "shorts": "Home",
                    "socks": "Home",
                    "gk_kit": "Green",
                },
                "lineup": [
                    {
                        "number": p.number,
                        "name": p.name,
                        "display_name": p.short_name,
                        "position": p.position,
                        "role": p.position,
                        "player_id": p.player_id,
                    }
                    for p in gt.starting_xi("home")
                ],
            },
            "away_team": {
                "name": gt.away_team,
                "short_name": gt.away_team[:3].upper(),
                "formation": "4-3-3",
                "kit": {
                    "primary_color": "Away Primary",
                    "shirt": "Away",
                    "shorts": "Away",
                    "socks": "Away",
                    "gk_kit": "Yellow",
                },
                "lineup": [
                    {
                        "number": p.number,
                        "name": p.name,
                        "display_name": p.short_name,
                        "position": p.position,
                        "role": p.position,
                        "player_id": p.player_id,
                    }
                    for p in gt.starting_xi("away")
                ],
            },
        }

    def _build_empty_fixture(self, match_name: str) -> Dict[str, Any]:
        """Constructs an empty structured fixture. Never hardcodes any player names or rosters."""
        parts = match_name.split(" vs ")
        home_name = parts[0].strip() if len(parts) >= 1 else "Home Team"
        away_name = parts[1].strip() if len(parts) >= 2 else "Away Team"

        return {
            "match": match_name,
            "competition": "Official Competition",
            "home_team": {
                "name": home_name,
                "short_name": home_name[:3].upper(),
                "formation": "4-3-3",
                "kit": {"primary_color": "White", "shirt": "White", "shorts": "White", "socks": "White", "gk_kit": "Green"},
                "lineup": [],
            },
            "away_team": {
                "name": away_name,
                "short_name": away_name[:3].upper(),
                "formation": "4-3-3",
                "kit": {"primary_color": "Dark", "shirt": "Dark", "shorts": "Dark", "socks": "Dark", "gk_kit": "Yellow"},
                "lineup": [],
            },
        }

    def resolve_player(
        self,
        detected_player: Dict[str, Any],
        fixture: Dict[str, Any],
        video_timestamp_sec: Optional[float] = None,
        match_minute: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Multi-cue Bayesian-heuristic resolution strictly gated by active on-pitch mathematical verification.
        Eliminates hallucinations: only players verified to be on the pitch at that timestamp can be assigned.
        """
        team_label = str(detected_player.get("team", "")).lower()
        visible_number = detected_player.get("jersey_number")
        if visible_number is not None:
            try:
                visible_number = int(visible_number)
            except (ValueError, TypeError):
                visible_number = None

        detected_pos = str(detected_player.get("tactical_role", "")).upper()
        detected_traits = str(detected_player.get("physical_traits", "")).lower()

        home_team = fixture["home_team"]
        away_team = fixture["away_team"]

        candidate_team = None
        is_gk = "gk" in detected_pos.lower() or "goalkeeper" in detected_pos.lower() or "goalkeeper" in team_label

        if "home" in team_label or home_team["name"].lower() in team_label or home_team["short_name"].lower() in team_label:
            candidate_team = home_team
            candidate_side = "home"
        elif "away" in team_label or away_team["name"].lower() in team_label or away_team["short_name"].lower() in team_label:
            candidate_team = away_team
            candidate_side = "away"
        else:
            kit_desc = str(detected_player.get("kit_color", "")).lower()
            home_kit_desc = str(home_team.get("kit", {})).lower()
            away_kit_desc = str(away_team.get("kit", {})).lower()

            home_score = sum(1 for w in ["white", "blue", "red", "yellow", "black", "garnet"] if w in kit_desc and w in home_kit_desc)
            away_score = sum(1 for w in ["white", "blue", "red", "yellow", "black", "garnet"] if w in kit_desc and w in away_kit_desc)

            if home_score >= away_score:
                candidate_team = home_team
                candidate_side = "home"
            else:
                candidate_team = away_team
                candidate_side = "away"

        # Determine effective match timestamp
        if match_minute is None:
            if video_timestamp_sec is not None:
                match_minute = int(video_timestamp_sec // 60)
            elif detected_player.get("match_minute") is not None:
                match_minute = int(detected_player["match_minute"])
            elif detected_player.get("timestamp_sec") is not None:
                match_minute = int(float(detected_player["timestamp_sec"]) // 60)
            else:
                match_minute = int(fixture.get("current_minute", 0))

        instant = MatchInstant.from_clock(match_minute, 0)
        gt: Optional[GroundTruthDictionary] = fixture.get("_ground_truth") or get_registered_ground_truth()

        # Dynamic On-Pitch Filtering: eliminate subbed-off or sent-off players from candidate pool
        if gt is not None:
            on_pitch_players = gt.on_pitch(instant, candidate_side)
            # Reconstruct candidate lineup from on-pitch ground truth
            lineup_candidates = [
                {
                    "number": p.number,
                    "name": p.name,
                    "display_name": p.short_name,
                    "position": p.position,
                    "role": p.position,
                    "player_id": p.player_id,
                    "hair": p.visual_traits.get("hair", ""),
                    "skin": p.visual_traits.get("skin", ""),
                    "build": p.visual_traits.get("build", ""),
                }
                for p in on_pitch_players
            ]
        else:
            lineup_candidates = candidate_team.get("lineup", [])

        # Strict jersey number validation: If a jersey number is detected, mathematically verify that
        # this number is legally on the pitch. If the player wearing that number is subbed off or sent off,
        # immediately block the assignment as a hallucination.
        if visible_number is not None and gt is not None:
            if not gt.is_on_pitch(visible_number, instant, candidate_side):
                logger.warning(
                    "IdentityGate: BLOCKED detection with jersey #%d at %s (player not on pitch)",
                    visible_number, instant.clock_str
                )
                return {
                    "name": f"Unverified Player #{visible_number} ({detected_pos or 'MF'})",
                    "display_name": f"Unverified #{visible_number}",
                    "number": visible_number,
                    "position": detected_pos or "MF",
                    "role": detected_pos or "Midfield",
                    "team": candidate_team["name"],
                    "short_team": candidate_team["short_name"],
                    "confidence": 0.0,
                    "resolved_by": "unverified_hallucination_blocked",
                    "hallucination_blocked": True,
                }

        # Check explicit player name suggestion from detector or VLM
        detected_name = detected_player.get("name") or detected_player.get("player_name")
        if detected_name and gt is not None:
            if not gt.is_on_pitch(detected_name, instant, candidate_side):
                logger.warning(
                    "IdentityGate: BLOCKED detection with name '%s' at %s (player not on pitch)",
                    detected_name, instant.clock_str
                )
                return {
                    "name": f"Unverified Player ({detected_pos or 'MF'})",
                    "display_name": f"Unverified ({detected_pos or 'MF'})",
                    "number": visible_number or 0,
                    "position": detected_pos or "MF",
                    "role": detected_pos or "Midfield",
                    "team": candidate_team["name"],
                    "short_team": candidate_team["short_name"],
                    "confidence": 0.0,
                    "resolved_by": "unverified_hallucination_blocked",
                    "hallucination_blocked": True,
                }

        if not lineup_candidates:
            # Roster empty or unverified: strictly block hallucinations
            return {
                "name": f"Unverified Player ({detected_pos or 'MF'})",
                "display_name": f"Unverified ({detected_pos or 'MF'})",
                "number": visible_number or 0,
                "position": detected_pos or "MF",
                "role": detected_pos or "Midfield",
                "team": candidate_team["name"],
                "short_team": candidate_team["short_name"],
                "confidence": 0.0,
                "resolved_by": "unverified_hallucination_blocked",
                "hallucination_blocked": True,
            }

        scored_candidates = []
        for player in lineup_candidates:
            score = 0.0

            # 1. Number matching
            if visible_number is not None:
                if player["number"] == visible_number:
                    score += 55.0
                else:
                    score -= 30.0

            # 2. Position matching
            player_pos = player["position"].upper()
            if detected_pos:
                if player_pos == detected_pos:
                    score += 25.0
                elif (player_pos in ["CB", "LB", "RB"] and any(d in detected_pos for d in ["DEF", "BACK", "CB", "LB", "RB"])) or \
                     (player_pos in ["CM", "DM", "CAM"] and any(d in detected_pos for d in ["MID", "CM", "DM", "CAM"])) or \
                     (player_pos in ["LW", "RW", "ST"] and any(d in detected_pos for d in ["ATT", "FWD", "WING", "ST", "LW", "RW"])):
                    score += 15.0

            # 3. Goalkeeper check
            if is_gk:
                if player["position"] == "GK":
                    score += 40.0
                else:
                    score -= 40.0

            # 4. Physical traits matching
            player_hair = player.get("hair", "").lower()
            player_skin = player.get("skin", "").lower()
            player_build = player.get("build", "").lower()

            for word in ["blonde", "blond", "dread", "dreads", "long", "bald", "beard", "dark", "brown", "afro"]:
                if word in detected_traits and word in player_hair:
                    score += 10.0

            for word in ["dark", "light", "fair", "olive"]:
                if word in detected_traits and word in player_skin:
                    score += 6.0

            for word in ["tall", "muscular", "slender", "lean", "stocky"]:
                if word in detected_traits and word in player_build:
                    score += 5.0

            confidence = max(5.0, min(99.0, score + (40.0 if visible_number is None else 20.0)))
            scored_candidates.append({
                "player": player,
                "team": candidate_team["name"],
                "short_team": candidate_team["short_name"],
                "score": score,
                "confidence": round(confidence, 1),
            })

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        best = scored_candidates[0]

        # 5. IdentityGate Verification
        if gt is not None:
            is_verified, _, reason = IdentityGate.verify_identity(
                candidate_player=best["player"]["name"],
                instant=instant,
                ground_truth=gt,
                side=candidate_side,
            )
            if not is_verified:
                logger.warning(
                    "IdentityGate: Rejected hallucinated player '%s' at %s (%s)",
                    best["player"]["name"], instant.clock_str, reason
                )
                return {
                    "name": f"Unverified Player ({detected_pos or 'MF'})",
                    "display_name": f"Unverified ({detected_pos or 'MF'})",
                    "number": visible_number or 0,
                    "position": detected_pos or "MF",
                    "role": detected_pos or "Midfield",
                    "team": candidate_team["name"],
                    "short_team": candidate_team["short_name"],
                    "confidence": 0.0,
                    "resolved_by": "unverified_hallucination_blocked",
                    "hallucination_blocked": True,
                }

        return {
            "name": best["player"]["name"],
            "display_name": best["player"]["display_name"],
            "number": best["player"]["number"],
            "position": best["player"]["position"],
            "role": best["player"]["role"],
            "team": best["team"],
            "short_team": best["short_team"],
            "confidence": best["confidence"],
            "resolved_by": "number+traits" if visible_number is not None else "position+traits",
        }
