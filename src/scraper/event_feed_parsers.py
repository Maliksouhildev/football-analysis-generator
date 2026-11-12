"""
src/scraper/event_feed_parsers.py
High-Precision Structured (X, Y, t) Event Feed Parsers (Fixing Flaw A: Event Feed Illusion).

Eliminates reliance on coarse text commentary feeds by targeting structured (X,Y,t) event feeds:
Option 1: WhoScored Match Center internal JSON endpoint parser (extracts pass-by-pass (X,Y),
          second-accurate timestamps, and player IDs).
Option 2: StatsBomb Open Data parser (100% free, ground-truth (X,Y) event streams and
          360 freeze frames for top-tier matches).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from src.scraper.event_scraper import (
    BaseProviderClient,
    CanonicalEventType,
    CanonicalMatchEvent,
    PlayerInfo,
    TeamInfo,
)

logger = logging.getLogger("scraper.event_feed_parsers")


class WhoScoredMatchCenterParser:
    """
    Parses WhoScored internal match center event streams.
    WhoScored events contain exact normalized coordinates:
        x: float in [0.0..100.0] (0 = defending goal, 100 = attacking goal)
        y: float in [0.0..100.0] (0 = left touchline, 100 = right touchline)
        minute: int
        second: int
        playerId: int
        teamId: int
        type: dict {'displayName': 'Pass', 'value': 1}
        outcomeType: dict {'displayName': 'Successful'}
    """

    EVENT_TYPE_MAP: Dict[str, CanonicalEventType] = {
        "pass": CanonicalEventType.PASS,
        "shot": CanonicalEventType.SHOT,
        "goal": CanonicalEventType.GOAL,
        "tackle": CanonicalEventType.KEY_RECOVERY,
        "interception": CanonicalEventType.KEY_RECOVERY,
        "clearance": CanonicalEventType.CLEARANCE,
        "aerial": CanonicalEventType.PASS,
        "takeon": CanonicalEventType.PROGRESSIVE_CARRY,
        "foul": CanonicalEventType.FOUL,
        "card": CanonicalEventType.YELLOW_CARD,
        "substitutionon": CanonicalEventType.TACTICAL_SUB,
        "substitutionoff": CanonicalEventType.TACTICAL_SUB,
        "cornerawarded": CanonicalEventType.CORNER_KICK,
    }

    @classmethod
    def parse_events(
        cls,
        payload: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        fixture_id: str = "whoscored_match",
    ) -> List[CanonicalMatchEvent]:
        if isinstance(payload, str):
            data = json.loads(payload)
        else:
            data = payload

        raw_events: List[Dict[str, Any]] = []
        player_map: Dict[int, str] = {}
        team_map: Dict[int, str] = {}

        if isinstance(data, dict):
            # WhoScored match center bundle contains 'events', 'playerIdNameDictionary', 'matchCentreData'
            if "matchCentreData" in data:
                data = data["matchCentreData"]

            raw_events = data.get("events", [])
            player_map = {
                int(k): str(v)
                for k, v in data.get("playerIdNameDictionary", {}).items()
            }
            if "home" in data and "away" in data:
                home_id = data["home"].get("teamId")
                away_id = data["away"].get("teamId")
                if home_id is not None:
                    team_map[int(home_id)] = data["home"].get("name", "Home")
                if away_id is not None:
                    team_map[int(away_id)] = data["away"].get("name", "Away")
        elif isinstance(data, list):
            raw_events = data

        canonical: List[CanonicalMatchEvent] = []

        for idx, ev in enumerate(raw_events):
            ev_id = str(ev.get("id") or ev.get("eventId") or f"ws_{fixture_id}_{idx}")
            minute = int(ev.get("minute") or 0)
            second = int(ev.get("second") or 0)
            period = int(ev.get("period", {}).get("value", 1) if isinstance(ev.get("period"), dict) else (ev.get("period") or 1))

            raw_type = ev.get("type", {})
            type_name = str(raw_type.get("displayName", "") if isinstance(raw_type, dict) else raw_type).lower()

            canonical_type = cls.EVENT_TYPE_MAP.get(type_name, CanonicalEventType.UNKNOWN)
            # Check for goal flag
            is_goal = False
            for q in ev.get("qualifiers", []):
                q_type = q.get("type", {}).get("displayName", "").lower()
                if "goal" in q_type:
                    canonical_type = CanonicalEventType.GOAL
                    is_goal = True
                    break

            # Coordinates are in 0-100 space
            px = float(ev.get("x", 50.0))
            py = float(ev.get("y", 50.0))
            end_x = float(ev.get("endX", px))
            end_y = float(ev.get("endY", py))

            player_id = ev.get("playerId")
            player_name = player_map.get(int(player_id), "") if player_id is not None else ""
            if not player_name and "playerName" in ev:
                player_name = str(ev["playerName"])

            team_id = ev.get("teamId")
            team_name = team_map.get(int(team_id), "Team") if team_id is not None else "Team"

            evt = CanonicalMatchEvent(
                id=ev_id,
                match_id=fixture_id,
                period=period,
                minute=minute,
                second=second,
                event_type=canonical_type,
                team=TeamInfo(name=team_name, id=str(team_id) if team_id else None),
                primary_player=PlayerInfo(name=player_name, id=str(player_id) if player_id else None) if player_name else None,
                pitch_x=px,
                pitch_y=py,
                is_big_chance=is_goal or (canonical_type == CanonicalEventType.SHOT and px >= 85.0),
                provider="whoscored",
                raw_payload={"endX": end_x, "endY": end_y, "outcome": ev.get("outcomeType")},
            )
            canonical.append(evt)

        return canonical


class StatsBombOpenDataParser:
    """
    Parses StatsBomb Open Data event format (used in free public StatsBomb repositories).
    StatsBomb pitch coordinates:
        X in [0.0..120.0] meters (defending line = 0, attacking goal = 120)
        Y in [0.0..80.0] meters (top touchline = 0, bottom touchline = 80)
    Normalized to CanonicalMatchEvent:
        pitch_x = (X / 120.0) * 100.0
        pitch_y = (Y / 80.0) * 100.0
    """

    EVENT_TYPE_MAP: Dict[str, CanonicalEventType] = {
        "pass": CanonicalEventType.PASS,
        "shot": CanonicalEventType.SHOT,
        "carry": CanonicalEventType.PROGRESSIVE_CARRY,
        "ball receipt*": CanonicalEventType.PASS,
        "interception": CanonicalEventType.KEY_RECOVERY,
        "clearance": CanonicalEventType.CLEARANCE,
        "duel": CanonicalEventType.KEY_RECOVERY,
        "foul committed": CanonicalEventType.FOUL,
        "foul won": CanonicalEventType.FOUL,
        "bad behaviour": CanonicalEventType.YELLOW_CARD,
        "substitution": CanonicalEventType.TACTICAL_SUB,
    }

    @classmethod
    def parse_events(
        cls,
        payload: Union[str, List[Dict[str, Any]]],
        fixture_id: str = "statsbomb_match",
    ) -> List[CanonicalMatchEvent]:
        if isinstance(payload, str):
            raw_events = json.loads(payload)
        else:
            raw_events = payload

        canonical: List[CanonicalMatchEvent] = []

        for idx, ev in enumerate(raw_events):
            ev_id = str(ev.get("id") or f"sb_{fixture_id}_{idx}")
            minute = int(ev.get("minute") or 0)
            second = int(ev.get("second") or 0)
            period = int(ev.get("period") or 1)

            type_dict = ev.get("type", {})
            type_name = str(type_dict.get("name", "") if isinstance(type_dict, dict) else type_dict).lower()

            canonical_type = cls.EVENT_TYPE_MAP.get(type_name, CanonicalEventType.UNKNOWN)

            # Check if shot was a goal
            shot_dict = ev.get("shot", {})
            outcome_dict = shot_dict.get("outcome", {})
            outcome_name = outcome_dict.get("name", "").lower() if isinstance(outcome_dict, dict) else ""
            is_goal = outcome_name == "goal"
            if is_goal:
                canonical_type = CanonicalEventType.GOAL

            xg = float(shot_dict.get("statsbomb_xg", 0.0))

            # Location: [x, y] on 120x80 pitch
            location = ev.get("location", [60.0, 40.0])
            if isinstance(location, (list, tuple)) and len(location) >= 2:
                raw_x = float(location[0])
                raw_y = float(location[1])
                # Normalize to 0-100 canonical space
                px = min(100.0, max(0.0, (raw_x / 120.0) * 100.0))
                py = min(100.0, max(0.0, (raw_y / 80.0) * 100.0))
            else:
                px, py = 50.0, 50.0

            player_dict = ev.get("player", {})
            player_name = player_dict.get("name", "") if isinstance(player_dict, dict) else str(player_dict)
            player_id = player_dict.get("id") if isinstance(player_dict, dict) else None

            team_dict = ev.get("team", {})
            team_name = team_dict.get("name", "Team") if isinstance(team_dict, dict) else str(team_dict)
            team_id = team_dict.get("id") if isinstance(team_dict, dict) else None

            # Freeze frame extraction (e.g. 360 player coordinates if present)
            freeze_frame = shot_dict.get("freeze_frame") or ev.get("shot_360") or ev.get("freeze_frame")

            evt = CanonicalMatchEvent(
                id=ev_id,
                match_id=fixture_id,
                period=period,
                minute=minute,
                second=second,
                event_type=canonical_type,
                team=TeamInfo(name=team_name, id=str(team_id) if team_id else None),
                primary_player=PlayerInfo(name=player_name, id=str(player_id) if player_id else None) if player_name else None,
                pitch_x=round(px, 2),
                pitch_y=round(py, 2),
                xg=xg,
                is_big_chance=is_goal or xg >= 0.25,
                provider="statsbomb",
                raw_payload={"freeze_frame": freeze_frame} if freeze_frame else None,
            )
            canonical.append(evt)

        return canonical
