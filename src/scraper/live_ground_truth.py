"""
src/scraper/live_ground_truth.py
Live Ground Truth Roster & Identity Gate Engine.

Implements autonomous, dynamic player tracking from:
1. Sportmonks API (/v3/football/livescores/inplay?include=scores;participants;events;lineups)
2. API-Football (/fixtures?live=all / /fixtures/lineups / /fixtures/events) [Fallback]

Eliminates player identity hallucinations by mathematically verifying that any player
assigned an identity is actively and legally on the pitch at that exact video timestamp/minute.
NO PLAYER DATA IS EVER HARDCODED.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("scraper.live_ground_truth")

# Global process-wide ground truth registry
_REGISTERED_GROUND_TRUTH: Optional[GroundTruthDictionary] = None


# =============================================================================
# 1. Match Timestamp & Clock Models
# =============================================================================

@dataclass
class MatchInstant:
    """Exact timestamp representation in match timeline."""
    minute: int
    second: int = 0
    period: int = 1
    extra_minute: int = 0

    @classmethod
    def from_clock(cls, minute: int, second: int = 0, period: Optional[int] = None, extra_minute: int = 0) -> MatchInstant:
        m = int(minute)
        if period is None:
            if m > 105:
                p = 4
            elif m > 90:
                p = 3
            elif m > 45:
                p = 2
            else:
                p = 1
        else:
            p = int(period)
        return cls(minute=m, second=int(second), period=p, extra_minute=int(extra_minute))

    @classmethod
    def from_seconds(cls, total_seconds: float, period: int = 1) -> MatchInstant:
        sec = max(0.0, float(total_seconds))
        m = int(sec // 60)
        s = int(sec % 60)
        return cls(minute=m, second=s, period=int(period))

    @property
    def total_seconds(self) -> float:
        return float(self.minute * 60 + self.second)

    @property
    def clock_str(self) -> str:
        if self.extra_minute > 0:
            return f"{self.minute}+{self.extra_minute}:{self.second:02d}"
        return f"{self.minute:02d}:{self.second:02d}"

    def __le__(self, other: MatchInstant) -> bool:
        if self.period != other.period:
            return self.period < other.period
        return self.total_seconds <= other.total_seconds

    def __lt__(self, other: MatchInstant) -> bool:
        if self.period != other.period:
            return self.period < other.period
        return self.total_seconds < other.total_seconds

    def __ge__(self, other: MatchInstant) -> bool:
        return not (self < other)

    def __gt__(self, other: MatchInstant) -> bool:
        return not (self <= other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MatchInstant):
            return False
        return self.period == other.period and self.minute == other.minute and self.second == other.second


# =============================================================================
# 2. Player Ground Truth & Event Models
# =============================================================================

@dataclass
class PlayerGroundTruth:
    """Canonical ground truth record of a player from official API lineups."""
    player_id: Union[int, str]
    name: str
    short_name: str
    number: int
    position: str  # GK, DF, MF, FW, etc.
    is_gk: bool
    team_id: Union[int, str]
    team_side: str  # "home" | "away"
    team_name: str
    is_starter: bool = True
    subbed_in_minute: Optional[int] = None
    subbed_out_minute: Optional[int] = None
    red_card_minute: Optional[int] = None
    visual_traits: Dict[str, Any] = field(default_factory=dict)
    tactical_flags: List[str] = field(default_factory=list)
    is_captain: bool = False

    def __post_init__(self):
        if not self.short_name:
            last = self.name.split()[-1] if self.name else "Player"
            self.short_name = f"{last} ({self.position or 'MF'})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "short_name": self.short_name,
            "number": self.number,
            "position": self.position,
            "role": self.position,
            "is_gk": self.is_gk,
            "team_id": self.team_id,
            "team_side": self.team_side,
            "team": self.team_side,
            "team_name": self.team_name,
            "is_starter": self.is_starter,
            "subbed_in_minute": self.subbed_in_minute,
            "subbed_out_minute": self.subbed_out_minute,
            "red_card_minute": self.red_card_minute,
            "visual_traits": self.visual_traits,
            "tactical_flags": self.tactical_flags,
            "is_captain": self.is_captain,
        }


@dataclass
class GroundTruthEvent:
    """Timeline event (substitution, red card, goal) with absolute timestamps."""
    event_id: Union[int, str]
    event_type: str  # "substitution" | "red_card" | "yellow_red" | "goal"
    minute: int
    extra_minute: Optional[int] = None
    second: int = 0
    period: int = 1
    team_side: str = "home"  # "home" | "away"
    team_id: Optional[Union[int, str]] = None
    player_id: Optional[Union[int, str]] = None
    player_name: Optional[str] = None
    related_player_id: Optional[Union[int, str]] = None  # for sub: player entering pitch
    related_player_name: Optional[str] = None  # for sub: player entering pitch name
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "minute": self.minute,
            "extra_minute": self.extra_minute,
            "second": self.second,
            "period": self.period,
            "team_side": self.team_side,
            "team_id": self.team_id,
            "player_id": self.player_id,
            "player_name": self.player_name,
            "related_player_id": self.related_player_id,
            "related_player_name": self.related_player_name,
        }


# =============================================================================
# 3. Ground Truth Dictionary
# =============================================================================

class GroundTruthDictionary:
    """
    Dynamic backend Ground Truth repository for match lineups and active on-pitch players.
    Maps out the starting XI and parses the events timeline to track substitutions and red
    cards with absolute timestamps, ensuring zero hallucinations.
    """

    def __init__(
        self,
        fixture_id: Union[int, str],
        provider: str,
        home_team: str,
        away_team: str,
        home_team_id: Optional[Union[int, str]] = None,
        away_team_id: Optional[Union[int, str]] = None,
        match_status: str = "INPLAY",
        current_minute: int = 0,
        scores: Optional[Dict[str, int]] = None,
        starting_xi_home: Optional[List[PlayerGroundTruth]] = None,
        starting_xi_away: Optional[List[PlayerGroundTruth]] = None,
        substitutes_home: Optional[List[PlayerGroundTruth]] = None,
        substitutes_away: Optional[List[PlayerGroundTruth]] = None,
        events_timeline: Optional[List[GroundTruthEvent]] = None,
        raw_payload: Optional[Dict[str, Any]] = None,
    ):
        self.fixture_id = fixture_id
        self.provider = provider
        self.home_team = home_team
        self.away_team = away_team
        self.home_team_id = home_team_id
        self.away_team_id = away_team_id
        self.match_status = match_status
        self.current_minute = current_minute
        self.scores = scores or {"home": 0, "away": 0}
        self.starting_xi_home = starting_xi_home or []
        self.starting_xi_away = starting_xi_away or []
        self.substitutes_home = substitutes_home or []
        self.substitutes_away = substitutes_away or []
        self.events_timeline = events_timeline or []
        self.raw_payload = raw_payload or {}

        # Sort timeline chronologically
        self.events_timeline.sort(key=lambda e: (e.period, e.minute, e.extra_minute or 0, e.second))

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Strips accents, punctuation, and extra whitespace for robust fuzzy matching."""
        if not name:
            return ""
        n = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("utf-8").lower().strip()
        n = re.sub(r"\s*\([A-Za-z0-9_/ -]+\)$", "", n).strip()
        return re.sub(r"[^a-z0-9\s]", "", n)

    def side_of(self, team_identifier: Union[str, int]) -> str:
        """Determines if a team name or ID corresponds to 'home' or 'away'."""
        if team_identifier is None:
            return ""
        if str(team_identifier) == str(self.home_team_id):
            return "home"
        if str(team_identifier) == str(self.away_team_id):
            return "away"

        t_norm = self._normalize_name(str(team_identifier))
        h_norm = self._normalize_name(self.home_team)
        a_norm = self._normalize_name(self.away_team)

        if t_norm in h_norm or h_norm in t_norm or "home" in t_norm:
            return "home"
        if t_norm in a_norm or a_norm in t_norm or "away" in t_norm:
            return "away"
        return ""

    def starting_xi(self, side: str = "home") -> List[PlayerGroundTruth]:
        """Returns the official Starting XI for the requested side."""
        norm_side = side.lower().strip()
        if norm_side == "home":
            return list(self.starting_xi_home)
        elif norm_side == "away":
            return list(self.starting_xi_away)
        return list(self.starting_xi_home + self.starting_xi_away)

    def substitutes(self, side: str = "home") -> List[PlayerGroundTruth]:
        """Returns legal bench substitutes for the requested side."""
        norm_side = side.lower().strip()
        if norm_side == "home":
            return list(self.substitutes_home)
        elif norm_side == "away":
            return list(self.substitutes_away)
        return list(self.substitutes_home + self.substitutes_away)

    def all_squad_players(self, side: Optional[str] = None) -> List[PlayerGroundTruth]:
        """Returns all registered matchday players (starters + bench) for a side."""
        if side:
            return self.starting_xi(side) + self.substitutes(side)
        return self.starting_xi("home") + self.substitutes("home") + self.starting_xi("away") + self.substitutes("away")

    def on_pitch(self, instant: MatchInstant, side: Optional[str] = None) -> List[PlayerGroundTruth]:
        """
        Mathematically resolves the exact on-pitch active players at instant:
        1. Starts with official Starting XI.
        2. Applies substitutions where event.minute <= instant.minute.
        3. Removes red-carded players where event.minute <= instant.minute.
        """
        target_sides = ["home", "away"] if not side else [self.side_of(side) or side.lower()]

        result: List[PlayerGroundTruth] = []
        for s in target_sides:
            active = [copy.deepcopy(p) for p in self.starting_xi(s)]
            bench = [copy.deepcopy(p) for p in self.substitutes(s)]

            # Relevant events up to instant
            relevant_events = [
                e for e in self.events_timeline
                if e.team_side == s and (
                    e.minute < instant.minute or (
                        e.minute == instant.minute and (e.extra_minute or 0) <= instant.extra_minute
                    )
                )
            ]

            for ev in relevant_events:
                if ev.event_type == "substitution":
                    # Player out leaves pitch
                    out_idx = None
                    for idx, p in enumerate(active):
                        if (ev.player_id and str(p.player_id) == str(ev.player_id)) or \
                           (ev.player_name and self._match_names(p.name, ev.player_name)):
                            out_idx = idx
                            break

                    # Player in enters pitch
                    in_player = None
                    for p in bench:
                        if (ev.related_player_id and str(p.player_id) == str(ev.related_player_id)) or \
                           (ev.related_player_name and self._match_names(p.name, ev.related_player_name)):
                            in_player = p
                            break

                    if in_player is None and ev.related_player_name:
                        # Construct from event metadata if bench object missing
                        in_player = PlayerGroundTruth(
                            player_id=ev.related_player_id or f"sub_{ev.minute}",
                            name=ev.related_player_name,
                            short_name=f"{ev.related_player_name.split()[-1]} (SUB)",
                            number=ev.raw_data.get("number", 12),
                            position=ev.raw_data.get("position", "MF"),
                            is_gk=False,
                            team_id=ev.team_id or (self.home_team_id if s == "home" else self.away_team_id) or 0,
                            team_side=s,
                            team_name=self.home_team if s == "home" else self.away_team,
                            is_starter=False,
                            subbed_in_minute=ev.minute,
                        )

                    if in_player is not None:
                        in_player.subbed_in_minute = ev.minute
                        if out_idx is not None:
                            active[out_idx].subbed_out_minute = ev.minute
                            active[out_idx] = in_player
                        else:
                            active.append(in_player)

                elif ev.event_type in ("red_card", "yellow_red"):
                    # Player is sent off; reduce team count (11 -> 10)
                    card_idx = None
                    for idx, p in enumerate(active):
                        if (ev.player_id and str(p.player_id) == str(ev.player_id)) or \
                           (ev.player_name and self._match_names(p.name, ev.player_name)):
                            card_idx = idx
                            break
                    if card_idx is not None:
                        active[card_idx].red_card_minute = ev.minute
                        active.pop(card_idx)

            result.extend(active)
        return result

    def _match_names(self, name_a: str, name_b: str) -> bool:
        """Compares two player names tolerant of accents, nicknames, and role suffixes."""
        na = self._normalize_name(name_a)
        nb = self._normalize_name(name_b)
        if not na or not nb:
            return False
        if na == nb:
            return True
        if len(na) >= 4 and len(nb) >= 4 and (na in nb or nb in na):
            return True
        tokens_a = [t for t in na.split() if len(t) > 2]
        tokens_b = [t for t in nb.split() if len(t) > 2]
        if tokens_a and tokens_b and tokens_a[-1] == tokens_b[-1]:
            return True
        return False

    def is_on_pitch(
        self,
        player_identifier: Union[str, int, PlayerGroundTruth],
        instant: MatchInstant,
        side: Optional[str] = None
    ) -> bool:
        """
        Mathematically verifies if a player is on the pitch at instant.
        Rejects injured, transferred, benched, subbed-off, or red-carded players.
        """
        active_list = self.on_pitch(instant, side)

        if isinstance(player_identifier, PlayerGroundTruth):
            return any(str(p.player_id) == str(player_identifier.player_id) for p in active_list)

        # Check by jersey number
        if isinstance(player_identifier, int) or (isinstance(player_identifier, str) and player_identifier.isdigit()):
            num = int(player_identifier)
            return any(p.number == num for p in active_list)

        # Check by name or ID
        ident_str = str(player_identifier).strip()
        for p in active_list:
            if str(p.player_id) == ident_str:
                return True
            if self._match_names(p.name, ident_str) or self._match_names(p.short_name, ident_str):
                return True
        return False

    def get_verified_player(
        self,
        player_identifier: Union[str, int, PlayerGroundTruth],
        instant: MatchInstant,
        side: Optional[str] = None
    ) -> Optional[PlayerGroundTruth]:
        """Returns the verified PlayerGroundTruth object if on pitch, else None."""
        active_list = self.on_pitch(instant, side)

        if isinstance(player_identifier, PlayerGroundTruth):
            for p in active_list:
                if str(p.player_id) == str(player_identifier.player_id):
                    return p
            return None

        if isinstance(player_identifier, int) or (isinstance(player_identifier, str) and player_identifier.isdigit()):
            num = int(player_identifier)
            for p in active_list:
                if p.number == num:
                    return p
            return None

        ident_str = str(player_identifier).strip()
        for p in active_list:
            if str(p.player_id) == ident_str or self._match_names(p.name, ident_str) or self._match_names(p.short_name, ident_str):
                return p
        return None

    def active_whitelist_enum(self, instant: MatchInstant, side: Optional[str] = None) -> List[str]:
        """Returns strict enum whitelist of legally active player names for VLM prompts."""
        active_players = self.on_pitch(instant, side)
        names = []
        for p in active_players:
            names.append(p.name)
            if p.short_name and p.short_name != p.name:
                names.append(p.short_name)
        return sorted(list(set(names)))

    def banned_identities(self, instant: MatchInstant, side: Optional[str] = None) -> Set[str]:
        """Returns names of players strictly forbidden from being identified at this instant."""
        all_squad = self.all_squad_players(side)
        active_players = self.on_pitch(instant, side)
        active_ids = {str(p.player_id) for p in active_players}
        active_names = {self._normalize_name(p.name) for p in active_players}

        banned = set()
        for p in all_squad:
            if str(p.player_id) not in active_ids and self._normalize_name(p.name) not in active_names:
                banned.add(p.name)
        return banned

    def to_dict(self) -> Dict[str, Any]:
        """Serializes GroundTruthDictionary for API transmission or caching."""
        return {
            "fixture_id": self.fixture_id,
            "provider": self.provider,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "match_status": self.match_status,
            "current_minute": self.current_minute,
            "scores": self.scores,
            "starting_xi_home": [p.to_dict() for p in self.starting_xi_home],
            "starting_xi_away": [p.to_dict() for p in self.starting_xi_away],
            "substitutes_home": [p.to_dict() for p in self.substitutes_home],
            "substitutes_away": [p.to_dict() for p in self.substitutes_away],
            "events_timeline": [e.to_dict() for e in self.events_timeline],
        }


# =============================================================================
# 4. Identity Gate (Mathematical Verification Engine)
# =============================================================================

class IdentityGate:
    """
    Identity Verification Gate for AI Vision Modules.
    Only allows an identity to be assigned to a detected player if that player is
    mathematically verified to be legally on the pitch at that specific video timestamp.
    """

    @classmethod
    def verify_identity(
        cls,
        candidate_player: Union[str, int, Dict[str, Any], PlayerGroundTruth],
        instant: MatchInstant,
        ground_truth: Optional[GroundTruthDictionary] = None,
        side: Optional[str] = None,
    ) -> Tuple[bool, Optional[PlayerGroundTruth], str]:
        """
        Mathematical verification check:
        Returns (is_verified: bool, verified_player: Optional[PlayerGroundTruth], reason: str).
        """
        gt = ground_truth or get_registered_ground_truth()
        if gt is None:
            return (
                False,
                None,
                "REJECTED_NO_GROUND_TRUTH: No legal live ground truth registered. Hallucinations strictly blocked."
            )

        identifier: Union[str, int, PlayerGroundTruth]
        if isinstance(candidate_player, dict):
            identifier = candidate_player.get("name") or candidate_player.get("player_name") or candidate_player.get("number") or ""
        else:
            identifier = candidate_player

        if not identifier:
            return (False, None, "REJECTED_EMPTY_IDENTIFIER: Blank or invalid player identifier.")

        verified_player = gt.get_verified_player(identifier, instant, side)
        if verified_player is not None:
            return (
                True,
                verified_player,
                f"VERIFIED_ON_PITCH: {verified_player.name} (#{verified_player.number}) verified on pitch at {instant.clock_str}"
            )

        return (
            False,
            None,
            f"REJECTED_HALLUCINATION: Player '{identifier}' is mathematically verified NOT to be on pitch at {instant.clock_str}."
        )


# =============================================================================
# 5. Sportmonks API Client (/v3/football/livescores/inplay)
# =============================================================================

class SportmonksClient:
    """
    Primary Provider: Sportmonks API v3.
    Target endpoint: /v3/football/livescores/inplay (or /v3/football/fixtures/{fixture_id})
    Target query param: include=scores;participants;events;lineups
    """
    BASE_URL = "https://api.sportmonks.com/v3/football"

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.environ.get("SPORTMONKS_API_TOKEN") or os.environ.get("SPORTMONKS_API_KEY")

    def fetch_live_fixture(
        self,
        fixture_id: Optional[Union[int, str]] = None,
        timeout: int = 10
    ) -> Dict[str, Any]:
        """Queries Sportmonks inplay livescores or specific fixture with standard include parameters."""
        if not self.api_token:
            raise ValueError("SPORTMONKS_API_TOKEN is required to execute Sportmonks live scores request.")

        if fixture_id:
            endpoint = f"{self.BASE_URL}/fixtures/{fixture_id}"
        else:
            endpoint = f"{self.BASE_URL}/livescores/inplay"

        params = {
            "api_token": self.api_token,
            "include": "scores;participants;events;lineups",
        }
        url = f"{endpoint}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "FootballAutomation/1.0", "Accept": "application/json"})

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data

    @classmethod
    def parse_payload_to_ground_truth(
        cls,
        payload: Dict[str, Any],
        fixture_id: Optional[Union[str, int]] = None
    ) -> GroundTruthDictionary:
        """Parses Sportmonks API v3 JSON response into GroundTruthDictionary."""
        data_block = payload.get("data")
        if isinstance(data_block, list):
            if not data_block:
                raise ValueError("Sportmonks response contains empty inplay data array.")
            # Select target fixture or default first inplay fixture
            if fixture_id is not None:
                match_data = next((m for m in data_block if str(m.get("id")) == str(fixture_id)), data_block[0])
            else:
                match_data = data_block[0]
        elif isinstance(data_block, dict):
            match_data = data_block
        else:
            match_data = payload

        fid = match_data.get("id") or fixture_id or "sportmonks_live"

        # 1. Participants
        participants = match_data.get("participants", [])
        home_part = next((p for p in participants if p.get("meta", {}).get("location") == "home"), None)
        away_part = next((p for p in participants if p.get("meta", {}).get("location") == "away"), None)

        if not home_part and len(participants) >= 1:
            home_part = participants[0]
        if not away_part and len(participants) >= 2:
            away_part = participants[1]

        home_name = (home_part or {}).get("name", "Home Team")
        away_name = (away_part or {}).get("name", "Away Team")
        home_id = (home_part or {}).get("id")
        away_id = (away_part or {}).get("id")

        # 2. Lineups
        lineups = match_data.get("lineups", [])
        starting_xi_home: List[PlayerGroundTruth] = []
        starting_xi_away: List[PlayerGroundTruth] = []
        substitutes_home: List[PlayerGroundTruth] = []
        substitutes_away: List[PlayerGroundTruth] = []

        for item in lineups:
            team_id = item.get("team_id")
            side = "home" if str(team_id) == str(home_id) else ("away" if str(team_id) == str(away_id) else "home")
            tname = home_name if side == "home" else away_name

            player_info = item.get("player") or {}
            pname = item.get("player_name") or player_info.get("display_name") or player_info.get("common_name") or "Player"
            pnum = int(item.get("jersey_number") or player_info.get("jersey_number") or 0)
            pid = item.get("player_id") or player_info.get("id") or f"{pname}_{pnum}"

            pos_info = item.get("position") or player_info.get("position") or {}
            pos_code = pos_info.get("code") or pos_info.get("name") or "MF"
            is_gk = (pos_code == "GK") or (item.get("position_id") == 24)

            # Sportmonks type_id: 11 = starting XI, 12 = bench/substitute
            type_id = item.get("type_id")
            lineup_type = str(item.get("lineup_type", "")).lower()
            is_starter = (type_id == 11) or ("lineup" in lineup_type) or ("start" in lineup_type)

            player_obj = PlayerGroundTruth(
                player_id=pid,
                name=pname,
                short_name=f"{pname.split()[-1]} ({pos_code})",
                number=pnum,
                position=pos_code,
                is_gk=is_gk,
                team_id=team_id or (home_id if side == "home" else away_id) or 0,
                team_side=side,
                team_name=tname,
                is_starter=is_starter,
                is_captain=bool(item.get("captain", False)),
            )

            if side == "home":
                if is_starter:
                    starting_xi_home.append(player_obj)
                else:
                    substitutes_home.append(player_obj)
            else:
                if is_starter:
                    starting_xi_away.append(player_obj)
                else:
                    substitutes_away.append(player_obj)

        # 3. Events Timeline
        events = match_data.get("events", [])
        events_timeline: List[GroundTruthEvent] = []

        for ev in events:
            ev_type_info = ev.get("type") or {}
            type_name = str(ev_type_info.get("name") or ev_type_info.get("code") or "").lower()
            type_id = ev.get("type_id")

            is_sub = (type_id == 18) or ("sub" in type_name)
            is_red = (type_id == 14) or ("red" in type_name)

            part_id = ev.get("participant_id")
            ev_side = "home" if str(part_id) == str(home_id) else ("away" if str(part_id) == str(away_id) else "home")

            minute = int(ev.get("minute") or 0)
            extra_min = ev.get("extra_minute")
            extra_minute = int(extra_min) if extra_min is not None else None

            if is_sub:
                events_timeline.append(GroundTruthEvent(
                    event_id=ev.get("id") or f"sub_{minute}",
                    event_type="substitution",
                    minute=minute,
                    extra_minute=extra_minute,
                    period=int(ev.get("period_id") or (1 if minute <= 45 else 2)),
                    team_side=ev_side,
                    team_id=part_id,
                    player_id=ev.get("player_id"),
                    player_name=ev.get("player_name"),
                    related_player_id=ev.get("related_player_id"),
                    related_player_name=ev.get("related_player_name"),
                    raw_data=ev,
                ))
            elif is_red:
                events_timeline.append(GroundTruthEvent(
                    event_id=ev.get("id") or f"red_{minute}",
                    event_type="red_card",
                    minute=minute,
                    extra_minute=extra_minute,
                    period=int(ev.get("period_id") or (1 if minute <= 45 else 2)),
                    team_side=ev_side,
                    team_id=part_id,
                    player_id=ev.get("player_id"),
                    player_name=ev.get("player_name"),
                    raw_data=ev,
                ))

        # 4. Scores
        scores = {"home": 0, "away": 0}
        for sc in match_data.get("scores", []):
            if sc.get("description") == "CURRENT":
                score_info = sc.get("score", {})
                loc = score_info.get("participant")
                if loc in ("home", "away"):
                    scores[loc] = int(score_info.get("goals", 0))

        return GroundTruthDictionary(
            fixture_id=fid,
            provider="sportmonks",
            home_team=home_name,
            away_team=away_name,
            home_team_id=home_id,
            away_team_id=away_id,
            match_status=str(match_data.get("result_info") or "INPLAY"),
            current_minute=int(match_data.get("minute") or 0),
            scores=scores,
            starting_xi_home=starting_xi_home,
            starting_xi_away=starting_xi_away,
            substitutes_home=substitutes_home,
            substitutes_away=substitutes_away,
            events_timeline=events_timeline,
            raw_payload=payload,
        )


# =============================================================================
# 6. API-Football Client (Fallback Provider)
# =============================================================================

class ApiFootballClient:
    """
    Fallback Provider: API-Football (v3.football.api-sports.io).
    Fetches live fixture lineups and events if Sportmonks is unavailable.
    """
    BASE_URL = "https://v3.football.api-sports.io"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("API_FOOTBALL_KEY") or os.environ.get("APISPORTS_KEY")

    def fetch_live_fixture(
        self,
        fixture_id: Optional[Union[int, str]] = None,
        timeout: int = 10
    ) -> Dict[str, Any]:
        """Queries API-Football for live fixtures or specific match lineups + events."""
        if not self.api_key:
            raise ValueError("API_FOOTBALL_KEY is required to execute API-Football live scores request.")

        if fixture_id:
            url = f"{self.BASE_URL}/fixtures?id={fixture_id}"
        else:
            url = f"{self.BASE_URL}/fixtures?live=all"

        req = urllib.request.Request(
            url,
            headers={
                "x-apisports-key": self.api_key,
                "User-Agent": "FootballAutomation/1.0",
                "Accept": "application/json",
            }
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data

    @classmethod
    def parse_payload_to_ground_truth(
        cls,
        payload: Dict[str, Any],
        fixture_id: Optional[Union[str, int]] = None
    ) -> GroundTruthDictionary:
        """Parses API-Football response into canonical GroundTruthDictionary."""
        responses = payload.get("response", [])
        if not responses:
            raise ValueError("API-Football response contains empty fixture list.")

        match = responses[0]
        fixture_meta = match.get("fixture", {})
        fid = fixture_meta.get("id") or fixture_id or "apifootball_live"

        teams = match.get("teams", {})
        home_name = teams.get("home", {}).get("name", "Home Team")
        away_name = teams.get("away", {}).get("name", "Away Team")
        home_id = teams.get("home", {}).get("id")
        away_id = teams.get("away", {}).get("id")

        starting_xi_home: List[PlayerGroundTruth] = []
        starting_xi_away: List[PlayerGroundTruth] = []
        substitutes_home: List[PlayerGroundTruth] = []
        substitutes_away: List[PlayerGroundTruth] = []

        # Parse lineups
        for l_item in match.get("lineups", []):
            t_info = l_item.get("team", {})
            t_id = t_info.get("id")
            side = "home" if str(t_id) == str(home_id) else ("away" if str(t_id) == str(away_id) else "home")
            tname = home_name if side == "home" else away_name

            for p_entry in l_item.get("startXI", []):
                p = p_entry.get("player", {})
                pos = p.get("pos") or "MF"
                p_obj = PlayerGroundTruth(
                    player_id=p.get("id") or p.get("name"),
                    name=p.get("name", "Player"),
                    short_name=f"{p.get('name', 'Player').split()[-1]} ({pos})",
                    number=int(p.get("number") or 0),
                    position=pos,
                    is_gk=bool(pos == "G" or pos == "GK"),
                    team_id=t_id or 0,
                    team_side=side,
                    team_name=tname,
                    is_starter=True,
                )
                if side == "home":
                    starting_xi_home.append(p_obj)
                else:
                    starting_xi_away.append(p_obj)

            for p_entry in l_item.get("substitutes", []):
                p = p_entry.get("player", {})
                pos = p.get("pos") or "MF"
                p_obj = PlayerGroundTruth(
                    player_id=p.get("id") or p.get("name"),
                    name=p.get("name", "Player"),
                    short_name=f"{p.get('name', 'Player').split()[-1]} ({pos})",
                    number=int(p.get("number") or 0),
                    position=pos,
                    is_gk=bool(pos == "G" or pos == "GK"),
                    team_id=t_id or 0,
                    team_side=side,
                    team_name=tname,
                    is_starter=False,
                )
                if side == "home":
                    substitutes_home.append(p_obj)
                else:
                    substitutes_away.append(p_obj)

        # Parse events
        events_timeline: List[GroundTruthEvent] = []
        for ev in match.get("events", []):
            ev_type = str(ev.get("type", "")).lower()
            detail = str(ev.get("detail", "")).lower()
            ev_team_id = ev.get("team", {}).get("id")
            side = "home" if str(ev_team_id) == str(home_id) else ("away" if str(ev_team_id) == str(away_id) else "home")

            t_info = ev.get("time", {})
            minute = int(t_info.get("elapsed") or 0)
            extra = t_info.get("extra")
            extra_minute = int(extra) if extra is not None else None

            p_main = ev.get("player", {})
            p_assist = ev.get("assist", {})

            if "subst" in ev_type:
                events_timeline.append(GroundTruthEvent(
                    event_id=f"sub_{minute}_{p_main.get('id')}",
                    event_type="substitution",
                    minute=minute,
                    extra_minute=extra_minute,
                    period=1 if minute <= 45 else 2,
                    team_side=side,
                    team_id=ev_team_id,
                    player_id=p_main.get("id"),
                    player_name=p_main.get("name"),
                    related_player_id=p_assist.get("id"),
                    related_player_name=p_assist.get("name"),
                    raw_data=ev,
                ))
            elif "card" in ev_type and ("red" in detail or "second yellow" in detail):
                events_timeline.append(GroundTruthEvent(
                    event_id=f"red_{minute}_{p_main.get('id')}",
                    event_type="red_card",
                    minute=minute,
                    extra_minute=extra_minute,
                    period=1 if minute <= 45 else 2,
                    team_side=side,
                    team_id=ev_team_id,
                    player_id=p_main.get("id"),
                    player_name=p_main.get("name"),
                    raw_data=ev,
                ))

        goals = match.get("goals", {})
        scores = {
            "home": int(goals.get("home") or 0),
            "away": int(goals.get("away") or 0),
        }

        return GroundTruthDictionary(
            fixture_id=fid,
            provider="api_football",
            home_team=home_name,
            away_team=away_name,
            home_team_id=home_id,
            away_team_id=away_id,
            match_status=str(fixture_meta.get("status", {}).get("long", "INPLAY")),
            current_minute=int(fixture_meta.get("status", {}).get("elapsed") or 0),
            scores=scores,
            starting_xi_home=starting_xi_home,
            starting_xi_away=starting_xi_away,
            substitutes_home=substitutes_home,
            substitutes_away=substitutes_away,
            events_timeline=events_timeline,
            raw_payload=payload,
        )


# =============================================================================
# 7. Global Registry & Fetch Functions
# =============================================================================

def register_ground_truth(ground_truth: GroundTruthDictionary) -> None:
    """Registers the active live GroundTruthDictionary in process memory."""
    global _REGISTERED_GROUND_TRUTH
    _REGISTERED_GROUND_TRUTH = ground_truth
    logger.info(
        "Registered live GroundTruthDictionary: %s vs %s (provider: %s, fixture_id: %s, starters: %d/%d)",
        ground_truth.home_team,
        ground_truth.away_team,
        ground_truth.provider,
        ground_truth.fixture_id,
        len(ground_truth.starting_xi_home),
        len(ground_truth.starting_xi_away),
    )


def get_registered_ground_truth() -> Optional[GroundTruthDictionary]:
    """Retrieves the active process-wide live GroundTruthDictionary."""
    return _REGISTERED_GROUND_TRUTH


def clear_registered_ground_truth() -> None:
    """Clears registered ground truth (resets to unverified strict mode)."""
    global _REGISTERED_GROUND_TRUTH
    _REGISTERED_GROUND_TRUTH = None


def fetch_live_ground_truth(
    fixture_id: Optional[Union[str, int]] = None,
    sportmonks_token: Optional[str] = None,
    api_football_key: Optional[str] = None,
    offline_fallback_payload: Optional[Dict[str, Any]] = None,
) -> GroundTruthDictionary:
    """
    Primary ingestion gateway:
    1. Makes GET request to Sportmonks API /v3/football/livescores/inplay?include=scores;participants;events;lineups.
    2. If unavailable / fails, makes fallback request to API-Football.
    3. If running in offline test environment without network/API keys, parses provided offline payload.
    4. Automatically registers the resulting GroundTruthDictionary.
    """
    # 1. Sportmonks
    try:
        sm_client = SportmonksClient(api_token=sportmonks_token)
        payload = sm_client.fetch_live_fixture(fixture_id=fixture_id)
        gt = SportmonksClient.parse_payload_to_ground_truth(payload, fixture_id=fixture_id)
        register_ground_truth(gt)
        return gt
    except Exception as e:
        logger.info("Sportmonks live scores acquisition notice: %s. Initiating API-Football fallback.", e)

    # 2. API-Football Fallback
    try:
        af_client = ApiFootballClient(api_key=api_football_key)
        payload = af_client.fetch_live_fixture(fixture_id=fixture_id)
        gt = ApiFootballClient.parse_payload_to_ground_truth(payload, fixture_id=fixture_id)
        register_ground_truth(gt)
        return gt
    except Exception as e:
        logger.info("API-Football fallback acquisition notice: %s.", e)

    # 3. Offline / Synthetic Fallback for Unit Tests
    if offline_fallback_payload:
        if "data" in offline_fallback_payload:
            gt = SportmonksClient.parse_payload_to_ground_truth(offline_fallback_payload, fixture_id=fixture_id)
        else:
            gt = ApiFootballClient.parse_payload_to_ground_truth(offline_fallback_payload, fixture_id=fixture_id)
        register_ground_truth(gt)
        return gt

    raise RuntimeError(
        "Could not establish live Ground Truth from Sportmonks or API-Football. "
        "Provide valid SPORTMONKS_API_TOKEN or API_FOOTBALL_KEY or test payload."
    )
