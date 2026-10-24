"""
src/vision/gemini_client.py
Production Gemini 2.5 Flash Vision REST API Client & Prompt Engineering Subsystem.

Implements dual authentication (api_key param / GEMINI_API_KEY env var), boot-anchored
system prompt engineering, structured JSON schema validation, resilient retry loops,
and seamless zero-crash failover to the OfflineTacticalExtractor.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.vision.tactical_parser import (
    TacticalFormationParser,
    TacticalTelestrationProps,
    TelestrationPlayer,
    TelestrationLine,
    TelestrationPolygon,
)

logger = logging.getLogger("vision.gemini_client")

DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent"

CANONICAL_SYSTEM_INSTRUCTION = (
    "You are a World-Class Football Tactical Telestrator and Video Analyst for UEFA Champions League "
    "and Premier League broadcasts. Your task is to analyze match freeze-frame keyframes and output "
    "precise 2D tactical telestration vectors for broadcast rendering.\n\n"
    "CRITICAL RULE — BOOT ANCHORING:\n"
    "For every player detected, you MUST extract the ground contact coordinate of their boots/pitch surface "
    "on the turf as normalized percentage coordinates: x (0 to 100 from left to right) and y (0 to 100 from top to bottom). "
    "Do NOT anchor at the torso, head, chest, or bounding box center. The spotlight ring is a ground ellipse that lies "
    "flat on the turf under their studs.\n\n"
    "RULES FOR DETECTING PLAYERS:\n"
    "1. Spotlights: Identify 3 to 8 key players involved in the tactical moment (defenders in the backline, "
    "pressing attackers, ball-carrier, passing options).\n"
    "2. Anchor coordinates at boots/pitch surface [0..100].\n"
    "3. Tactical Roles: GK, CB, LB, RB, LWB, RWB, DM, CM, AM, LW, RW, CF.\n"
    "4. Colors:\n"
    "   - Yellow ('#ffff00'): Ball carrier or focal player.\n"
    "   - Emerald ('#00e676'): Passing options, progressive runners, or defensive backline.\n"
    "   - Red ('#ef4444'): Opposition pressing players or defensive pressure.\n"
    "   - Purple ('#a855f7'): Midfield overload orchestrator.\n"
    "5. Spotlight Elements: showRing=True for all detected players. showBeam=True for the focal ball carrier (#ffff00), "
    "showBadge=True when name or number is identifiable.\n\n"
    "RULES FOR TACTICAL LINES:\n"
    "1. Defensive Backlines: Draw unbroken line (dashed: false, color: '#00e676', width: 4) connecting the 3, 4, or 5 defenders "
    "in lateral spatial order from touchline to touchline.\n"
    "2. Pressing Lines: Draw dashed lines (dashed: true, color: '#ef4444', width: 3) between pressing attackers.\n"
    "3. Every line must connect at least 2 valid player IDs.\n\n"
    "RULES FOR TACTICAL POLYGONS:\n"
    "1. Passing Triangles: Connect 3 or 4 players creating a passing triangle or diamond. Set pattern to 'stripes' "
    "(diagonal hatched pattern), fillColor to 'rgba(0, 230, 118, 0.25)', and borderColor to '#00e676'. Label as 'PASSING TRIANGLE'.\n"
    "2. Pressing Traps: Connect 3 or 4 pressing players forming a trap. Set pattern to 'stripes', fillColor to 'rgba(239, 68, 68, 0.25)', "
    "borderColor to '#ef4444'. Label as 'PRESS TRAP'.\n"
    "3. Every polygon must connect at least 3 valid player IDs.\n\n"
    "CRITICAL GROUND-TRUTH TRAJECTORY EXTRACTION (FORBIDDEN GUESSING):\n"
    "You are provided a Multi-Frame Temporal Window [t - 2.0s to t + 4.5s] with reference frames:\n"
    "- FRAME PREV (t - 2.0s): Approaching play, ball carrier origin, and prior movements.\n"
    "- FRAME FREEZE (t = 0.0s): Tactical freeze-frame where graphics and arrows are anchored.\n"
    "- FRAME NEXT 1 (t + 1.5s): Initial play unfolding and pass release.\n"
    "- FRAME NEXT 2 (t + 3.0s): Pass travel, goalkeeper kick flight, corner trajectory.\n"
    "- FRAME NEXT 3 (t + 4.5s): Ground-truth play outcome.\n"
    "DO NOT GUESS OR INVENT TRAJECTORIES FROM FRAME FREEZE ALONE. In past runs, single-frame models hallucinated fake "
    "goalkeeper long-ball trajectories ('long ballon') or guessed corner destinations. You MUST observe future frames:\n"
    "1. Goalkeeper / Buildup: Check if goalkeeper actually plays short on the turf to an open defender (GROUND_PASS_SOLID) "
    "or boots long downfield (AERIAL_BALL_ARC). NEVER invent a fake 'long ballon' if the goalkeeper plays short!\n"
    "2. Corner Kicks: Observe the true kicker, delivery arc, and contested landing spot in the box.\n"
    "3. Pass Recipient & Player Runs: Observe the true receiver and run destinations, mapping coordinates back to FRAME FREEZE (t=0.0s)."
)


# =============================================================================
# 1. Vision Prompt Engineering Builder
# =============================================================================

class VisionPromptBuilder:
    """Constructs and validates Gemini 2.5 Flash Vision structured requests."""

    @staticmethod
    def build_request(
        base64_image: str,
        match_minute: int,
        context_desc: str,
        system_instruction_text: Optional[str] = None,
        active_roster_state: Optional[Any] = None,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Constructs the canonical wire JSON payload for gemini-2.5-flash:generateContent."""
        sys_text = system_instruction_text or CANONICAL_SYSTEM_INSTRUCTION
        user_prompt = f"Analyze this Minute {match_minute}' tactical freeze frame: {context_desc}"

        if active_roster_state is not None:
            from src.vision.match_roster_engine import ActiveXIResolver, ActiveRosterState
            resolver = ActiveXIResolver()
            if isinstance(active_roster_state, ActiveRosterState):
                user_prompt += f"\n\n{resolver.build_prompt_directive(active_roster_state)}"
                if response_schema is None:
                    response_schema = resolver.build_vision_schema(active_roster_state)
            elif isinstance(active_roster_state, dict) and "whitelist_enum" in active_roster_state:
                home_strs = [f"#{p.get('number', '')} {p.get('name', '')}" for p in active_roster_state.get('home_active', [])]
                away_strs = [f"#{p.get('number', '')} {p.get('name', '')}" for p in active_roster_state.get('away_active', [])]
                user_prompt += (
                    f"\n\nSTRICT ACTIVE ROSTER ENUM WHITELIST ENFORCEMENT:\n"
                    f"Match Minute: {active_roster_state.get('clock_str', match_minute)}\n"
                    f"Active players on pitch:\n"
                    f"- {active_roster_state.get('home_team', 'Home')}: {', '.join(home_strs)}\n"
                    f"- {active_roster_state.get('away_team', 'Away')}: {', '.join(away_strs)}\n"
                    f"FORBIDDEN PLAYERS (0% chance of appearing): Retired, benched, or subbed-off players.\n"
                )

        gen_config: Dict[str, Any] = {
            "temperature": 0.1,
            "response_mime_type": "application/json",
        }
        if response_schema is not None:
            gen_config["response_schema"] = response_schema

        return {
            "system_instruction": {
                "parts": [{"text": sys_text}]
            },
            "contents": [{
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_image,
                        }
                    },
                    {
                        "text": user_prompt
                    }
                ]
            }],
            "generationConfig": gen_config
        }

    @staticmethod
    def build_temporal_burst_request(
        burst_images: Dict[str, str],
        match_minute: int,
        context_desc: str,
        system_instruction_text: Optional[str] = None,
        active_roster_state: Optional[Any] = None,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Builds Gemini multimodal request incorporating the 5-frame temporal burst window [t-2.0s to t+4.5s]:
        - frame_prev (t - 2.0s): Approaching play, ball carrier origin, and prior movements.
        - frame_freeze (t = 0.0s): Tactical freeze-frame where graphics and arrows are anchored.
        - frame_next_1 (t + 1.5s): Initial play unfolding.
        - frame_next_2 (t + 3.0s): Pass travel / goalkeeper kick flight / corner trajectory.
        - frame_next_3 (t + 4.5s): Ground-truth outcome of the play.
        """
        sys_text = system_instruction_text or CANONICAL_SYSTEM_INSTRUCTION
        parts: List[Dict[str, Any]] = []

        labels = {
            "frame_prev": "IMAGE 1: FRAME PREV (t - 2.0s) — Approaching play, ball carrier origin, and prior movements:",
            "frame_freeze": "IMAGE 2: FRAME FREEZE (t = 0.0s) [Pristine] — Broadcast freeze moment for player & ball detection:",
            "frame_freeze_annotated": "IMAGE 3: FRAME FREEZE (t = 0.0s) [Annotated] — Candidate player [ID] tags and 10% coordinate ruler:",
            "frame_next_1": "IMAGE 4: FRAME NEXT 1 (t + 1.5s) — Initial play unfolding and pass release:",
            "frame_next_2": "IMAGE 5: FRAME NEXT 2 (t + 3.0s) — Pass travel, goalkeeper kick flight, or corner trajectory:",
            "frame_next_3": "IMAGE 6: FRAME NEXT 3 (t + 4.5s) — Ground-truth outcome of the play:",
        }

        for key, lbl in labels.items():
            b64_img = burst_images.get(key)
            if b64_img:
                parts.append({"text": lbl})
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": b64_img,
                    }
                })

        user_prompt = (
            f"Analyze this Minute {match_minute}' Multi-Frame Temporal Window [t - 2.0s to t + 4.5s]: {context_desc}\n\n"
            "GROUND-TRUTH TRAJECTORY INSTRUCTIONS (FORBIDDEN GUESSING):\n"
            "Extract the TRUE ground-truth trajectory by observing the play across the future reference frames "
            "(FRAME NEXT 1, FRAME NEXT 2, and FRAME NEXT 3).\n"
            "1. If the goalkeeper or deep defender plays short on the grass to a center-back or fullback, "
            "output GROUND_PASS_SOLID to that receiving teammate. DO NOT invent a fake 'long ballon' or aerial kick!\n"
            "2. If an aerial kick or corner cross is taken, identify the true kicker and delivery zone.\n"
            "3. Map all trajectory coordinates (to_pct) into the coordinate space of FRAME FREEZE (t = 0.0s)."
        )

        if active_roster_state is not None:
            from src.vision.match_roster_engine import ActiveXIResolver, ActiveRosterState
            resolver = ActiveXIResolver()
            if isinstance(active_roster_state, ActiveRosterState):
                user_prompt += f"\n\n{resolver.build_prompt_directive(active_roster_state)}"
                if response_schema is None:
                    response_schema = resolver.build_vision_schema(active_roster_state)
            elif isinstance(active_roster_state, dict) and "whitelist_enum" in active_roster_state:
                home_strs = [f"#{p.get('number', '')} {p.get('name', '')}" for p in active_roster_state.get('home_active', [])]
                away_strs = [f"#{p.get('number', '')} {p.get('name', '')}" for p in active_roster_state.get('away_active', [])]
                user_prompt += (
                    f"\n\nSTRICT ACTIVE ROSTER ENUM WHITELIST ENFORCEMENT:\n"
                    f"Match Minute: {active_roster_state.get('clock_str', match_minute)}\n"
                    f"Active players on pitch:\n"
                    f"- {active_roster_state.get('home_team', 'Home')}: {', '.join(home_strs)}\n"
                    f"- {active_roster_state.get('away_team', 'Away')}: {', '.join(away_strs)}\n"
                    f"FORBIDDEN PLAYERS (0% chance of appearing): Retired, benched, or subbed-off players.\n"
                )

        parts.append({"text": user_prompt})

        gen_config: Dict[str, Any] = {
            "temperature": 0.1,
            "response_mime_type": "application/json",
        }
        if response_schema is not None:
            gen_config["response_schema"] = response_schema

        return {
            "system_instruction": {
                "parts": [{"text": sys_text}]
            },
            "contents": [{
                "role": "user",
                "parts": parts,
            }],
            "generationConfig": gen_config
        }


    @staticmethod
    def encode_image_file(image_path: Union[str, Path]) -> Tuple[str, str]:
        """Reads image file from disk and encodes into Base64 string and MIME type."""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Freeze frame image file not found: {path}")

        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/jpeg"

        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return encoded, mime_type

    @staticmethod
    def encode_image_bytes(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        """Encodes raw binary image bytes into Base64 ASCII string."""
        return base64.b64encode(image_bytes).decode("utf-8")

    @staticmethod
    def format_user_prompt(
        match_minute: int,
        stoppage_minute: int = 0,
        event_type: str = "",
        description: str = "",
        team: str = "",
    ) -> str:
        clock = f"{match_minute}'+{stoppage_minute}'" if stoppage_minute > 0 else f"{match_minute}'"
        ev_str = f": {event_type.upper()}" if event_type else ""
        tm_str = f" by {team}" if team else ""
        ctx_str = f" Context: {description}." if description else ""
        return (
            f"Analyze this Minute {clock} tactical freeze frame{ev_str}{tm_str}.{ctx_str} "
            f"Extract key players (anchored at boots), defensive backline, and tactical geometries."
        )

    @staticmethod
    def sanitize_json_response(raw_text: str) -> str:
        """Strips markdown code fences (```json ... ```) and whitespace from model output."""
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-len("```")].strip()
        return cleaned


# =============================================================================
# 2. Resilient Offline Tactical Extractor (Event-Specific Fallback)
# =============================================================================

class OfflineTacticalExtractor:
    """
    Intelligent synthetic tactical geometry generator.
    Produces high-fidelity broadcast coordinates for all event categories when
    running offline, in CI/CD, or during API failover.
    """

    def extract(self, event_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generates realistic tactical coordinates based on event type and metadata."""
        ctx = event_context or {}
        event_type = str(ctx.get("event_type", "turnover")).lower()

        if "goal" in event_type:
            return self._generate_goal_telestration(ctx)
        elif "turnover" in event_type or "recovery" in event_type:
            return self._generate_turnover_telestration(ctx)
        elif "press" in event_type:
            return self._generate_broken_press_telestration(ctx)
        elif "chance" in event_type or "shot" in event_type:
            return self._generate_big_chance_telestration(ctx)
        elif "sub" in event_type:
            return self._generate_sub_telestration(ctx)
        else:
            return self._generate_default_telestration(ctx)

    def extract_props(
        self,
        event_context: Optional[Dict[str, Any]] = None,
        image: Optional[Union[str, Path, bytes]] = None,
    ) -> TacticalTelestrationProps:
        """Returns extracted geometry as typed TacticalTelestrationProps using real CV pitch detection."""
        ctx = event_context or {}
        if image:
            try:
                from src.vision.player_detector import CVPlayerDetector
                res = CVPlayerDetector.generate_roster_telestration(image, ctx)
                if res and res.players and len(res.players) >= 2:
                    return res
            except Exception as e:
                logger.warning("CVPlayerDetector fallback to rule-based extractor: %s", e)
        raw = self.extract(ctx)
        return TacticalFormationParser.parse_formation(raw)

    def _resolve_names(self, ctx: Dict[str, Any], count: int, default_names: List[str]) -> List[str]:
        raw_inv = ctx.get("players_involved", [])
        names = []
        for p in raw_inv:
            if isinstance(p, dict) and p.get("name"):
                names.append(p["name"])
            elif isinstance(p, str) and p.strip():
                names.append(p.strip())
        while len(names) < count:
            names.append(default_names[len(names) % len(default_names)])
        return names[:count]

    def _generate_goal_telestration(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        minute = ctx.get("match_minute", 24)
        team = ctx.get("team", "FC Barcelona")
        names = self._resolve_names(ctx, 4, ["Striker", "Winger", "Playmaker", "Defender"])

        return {
            "players": [
                {
                    "id": "p_scorer",
                    "name": names[0],
                    "number": 9,
                    "role": "CF",
                    "x": 74.5,
                    "y": 48.0,
                    "color": "#ffff00",
                    "showRing": True,
                    "showBeam": True,
                    "showBadge": True,
                },
                {
                    "id": "p_assister",
                    "name": names[1],
                    "number": 11,
                    "role": "LW",
                    "x": 68.0,
                    "y": 22.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": True,
                },
                {
                    "id": "p_mid",
                    "name": names[2],
                    "number": 8,
                    "role": "CM",
                    "x": 55.0,
                    "y": 38.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": True,
                },
                {
                    "id": "p_d1",
                    "name": "Def 1",
                    "number": 3,
                    "role": "LB",
                    "x": 60.0,
                    "y": 20.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": False,
                },
                {
                    "id": "p_d2",
                    "name": "Def 2",
                    "number": 4,
                    "role": "CB",
                    "x": 63.0,
                    "y": 44.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": False,
                },
                {
                    "id": "p_d3",
                    "name": "Def 3",
                    "number": 5,
                    "role": "CB",
                    "x": 65.0,
                    "y": 62.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": False,
                },
                {
                    "id": "p_d4",
                    "name": "Def 4",
                    "number": 2,
                    "role": "RB",
                    "x": 68.0,
                    "y": 80.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": False,
                },
            ],
            "lines": [
                {
                    "id": "defensive_backline",
                    "playerIds": ["p_d1", "p_d2", "p_d3", "p_d4"],
                    "color": "#00e676",
                    "width": 4,
                    "dashed": False,
                }
            ],
            "polygons": [
                {
                    "id": "goal_attack_triangle",
                    "playerIds": ["p_scorer", "p_assister", "p_mid"],
                    "fillColor": "rgba(0, 230, 118, 0.25)",
                    "borderColor": "#00e676",
                    "pattern": "stripes",
                    "label": "PENETRATION TRIANGLE",
                }
            ],
            "title": f"MINUTE {minute}' // GOAL SEQUENCE",
            "subtitle": f"{team.upper()} ATTACKING MASTERY",
        }

    def _generate_turnover_telestration(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        minute = ctx.get("match_minute", 34)
        names = self._resolve_names(ctx, 3, ["Interception Player", "Pressing Forward", "Support Pivot"])

        return {
            "players": [
                {
                    "id": "p_winner",
                    "name": names[0],
                    "number": 6,
                    "role": "DM",
                    "x": 32.5,
                    "y": 52.0,
                    "color": "#ffff00",
                    "showRing": True,
                    "showBeam": True,
                    "showBadge": True,
                },
                {
                    "id": "p_presser",
                    "name": names[1],
                    "number": 10,
                    "role": "AM",
                    "x": 42.0,
                    "y": 46.0,
                    "color": "#ef4444",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": True,
                },
                {
                    "id": "p_support",
                    "name": names[2],
                    "number": 8,
                    "role": "CM",
                    "x": 38.0,
                    "y": 70.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": True,
                },
                {
                    "id": "p_d1",
                    "name": "Defender 1",
                    "number": 3,
                    "role": "LB",
                    "x": 22.0,
                    "y": 18.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": False,
                },
                {
                    "id": "p_d2",
                    "name": "Defender 2",
                    "number": 4,
                    "role": "CB",
                    "x": 20.0,
                    "y": 42.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": False,
                },
                {
                    "id": "p_d3",
                    "name": "Defender 3",
                    "number": 5,
                    "role": "CB",
                    "x": 21.0,
                    "y": 62.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": False,
                },
                {
                    "id": "p_d4",
                    "name": "Defender 4",
                    "number": 2,
                    "role": "RB",
                    "x": 23.0,
                    "y": 82.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": False,
                },
            ],
            "lines": [
                {
                    "id": "defensive_backline",
                    "playerIds": ["p_d1", "p_d2", "p_d3", "p_d4"],
                    "color": "#00e676",
                    "width": 4,
                    "dashed": False,
                },
                {
                    "id": "pressing_corridor",
                    "playerIds": ["p_presser", "p_winner"],
                    "color": "#ef4444",
                    "width": 3,
                    "dashed": True,
                },
            ],
            "polygons": [
                {
                    "id": "turnover_trap_zone",
                    "playerIds": ["p_winner", "p_support", "p_presser"],
                    "fillColor": "rgba(239, 68, 68, 0.25)",
                    "borderColor": "#ef4444",
                    "pattern": "stripes",
                    "label": "PRESS TRAP",
                }
            ],
            "title": f"MINUTE {minute}' // DEFENSIVE THIRD TURNOVER",
            "subtitle": "COMPACT BLOCK RECOVERY",
        }

    def _generate_broken_press_telestration(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        minute = ctx.get("match_minute", 42)
        names = self._resolve_names(ctx, 3, ["Pedri", "Yamal", "Lewandowski"])

        return {
            "players": [
                {
                    "id": "p_pedri",
                    "name": names[0],
                    "number": 8,
                    "role": "CM",
                    "x": 45.2,
                    "y": 62.4,
                    "color": "#ffff00",
                    "showRing": True,
                    "showBeam": True,
                    "showBadge": True,
                },
                {
                    "id": "p_yamal",
                    "name": names[1],
                    "number": 19,
                    "role": "RW",
                    "x": 78.5,
                    "y": 35.1,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": True,
                },
                {
                    "id": "p_lewa",
                    "name": names[2],
                    "number": 9,
                    "role": "CF",
                    "x": 62.0,
                    "y": 48.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": True,
                },
                {
                    "id": "p_presser",
                    "name": "Presser",
                    "number": 10,
                    "role": "CM",
                    "x": 52.0,
                    "y": 58.0,
                    "color": "#ef4444",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": True,
                },
                {
                    "id": "p_balde",
                    "name": "Balde",
                    "number": 3,
                    "role": "LB",
                    "x": 32.0,
                    "y": 18.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": False,
                },
                {
                    "id": "p_martinez",
                    "name": "Martinez",
                    "number": 5,
                    "role": "CB",
                    "x": 26.0,
                    "y": 42.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": False,
                },
                {
                    "id": "p_cubarsi",
                    "name": "Cubarsi",
                    "number": 2,
                    "role": "CB",
                    "x": 28.0,
                    "y": 65.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": False,
                },
                {
                    "id": "p_kounde",
                    "name": "Kounde",
                    "number": 23,
                    "role": "RB",
                    "x": 38.0,
                    "y": 82.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": False,
                },
            ],
            "lines": [
                {
                    "id": "defensive_backline",
                    "playerIds": ["p_balde", "p_martinez", "p_cubarsi", "p_kounde"],
                    "color": "#00e676",
                    "width": 4,
                    "dashed": False,
                },
                {
                    "id": "pressing_corridor",
                    "playerIds": ["p_presser", "p_pedri"],
                    "color": "#ef4444",
                    "width": 3,
                    "dashed": True,
                },
            ],
            "polygons": [
                {
                    "id": "passing_triangle_midfield",
                    "playerIds": ["p_pedri", "p_yamal", "p_lewa"],
                    "fillColor": "rgba(0, 230, 118, 0.25)",
                    "borderColor": "#00e676",
                    "pattern": "stripes",
                    "label": "PASSING TRIANGLE",
                }
            ],
            "title": f"MINUTE {minute}' // PRESS BREAKOUT",
            "subtitle": "PROGRESSIVE MIDFIELD OVERLOAD",
        }

    def _generate_big_chance_telestration(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        minute = ctx.get("match_minute", 68)
        names = self._resolve_names(ctx, 3, ["Striker", "Playmaker", "Winger"])

        return {
            "players": [
                {
                    "id": "p_target",
                    "name": names[0],
                    "number": 9,
                    "role": "CF",
                    "x": 82.0,
                    "y": 50.0,
                    "color": "#ffff00",
                    "showRing": True,
                    "showBeam": True,
                    "showBadge": True,
                },
                {
                    "id": "p_creator",
                    "name": names[1],
                    "number": 10,
                    "role": "AM",
                    "x": 70.0,
                    "y": 38.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": True,
                },
                {
                    "id": "p_runner",
                    "name": names[2],
                    "number": 7,
                    "role": "RW",
                    "x": 76.0,
                    "y": 72.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": True,
                },
                {
                    "id": "p_cb1",
                    "name": "CB 1",
                    "number": 4,
                    "role": "CB",
                    "x": 80.0,
                    "y": 42.0,
                    "color": "#ef4444",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": False,
                },
                {
                    "id": "p_cb2",
                    "name": "CB 2",
                    "number": 5,
                    "role": "CB",
                    "x": 81.0,
                    "y": 58.0,
                    "color": "#ef4444",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": False,
                },
            ],
            "lines": [
                {
                    "id": "pressing_line",
                    "playerIds": ["p_cb1", "p_target"],
                    "color": "#ef4444",
                    "width": 3,
                    "dashed": True,
                }
            ],
            "polygons": [
                {
                    "id": "big_chance_triangle",
                    "playerIds": ["p_target", "p_creator", "p_runner"],
                    "fillColor": "rgba(0, 230, 118, 0.25)",
                    "borderColor": "#00e676",
                    "pattern": "stripes",
                    "label": "ZONE 14 PENETRATION",
                }
            ],
            "title": f"MINUTE {minute}' // BIG CHANCE CREATION",
            "subtitle": "HALF-SPACE CUTBACK ANGLE",
        }

    def _generate_sub_telestration(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        minute = ctx.get("match_minute", 60)
        names = self._resolve_names(ctx, 1, ["Sub Player"])

        return {
            "players": [
                {
                    "id": "p_sub",
                    "name": names[0],
                    "number": 14,
                    "role": "CM",
                    "x": 50.0,
                    "y": 50.0,
                    "color": "#ffff00",
                    "showRing": True,
                    "showBeam": True,
                    "showBadge": True,
                },
                {
                    "id": "p_d1",
                    "name": "LB",
                    "number": 3,
                    "role": "LB",
                    "x": 30.0,
                    "y": 20.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": False,
                },
                {
                    "id": "p_d2",
                    "name": "CB",
                    "number": 4,
                    "role": "CB",
                    "x": 28.0,
                    "y": 50.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": False,
                },
                {
                    "id": "p_d3",
                    "name": "RB",
                    "number": 2,
                    "role": "RB",
                    "x": 30.0,
                    "y": 80.0,
                    "color": "#00e676",
                    "showRing": True,
                    "showBeam": False,
                    "showBadge": False,
                },
            ],
            "lines": [
                {
                    "id": "defensive_backline",
                    "playerIds": ["p_d1", "p_d2", "p_d3"],
                    "color": "#00e676",
                    "width": 4,
                    "dashed": False,
                }
            ],
            "polygons": [],
            "title": f"MINUTE {minute}' // TACTICAL SUBSTITUTION",
            "subtitle": "MIDFIELD STRUCTURE REORGANIZATION",
        }

    def _generate_default_telestration(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return self._generate_broken_press_telestration(ctx)


# =============================================================================
# 3. Production Gemini 2.5 Flash Vision Client
# =============================================================================

class GeminiVisionClient:
    """
    Robust REST client for Google Gemini 2.5 Flash Vision.
    Directly interfaces with /models/gemini-2.5-flash:generateContent with dual auth,
    automatic retries, circuit breaking, and seamless offline failover.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        timeout: float = 12.0,
        offline_mode: Optional[bool] = None,
        max_retries: int = 3,
    ) -> None:
        self.endpoint_url = endpoint_url or DEFAULT_ENDPOINT
        self.timeout = float(timeout)
        self.max_retries = max(1, int(max_retries))
        self.offline_extractor = OfflineTacticalExtractor()

        # Dual authentication resolution
        resolved_key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        self.api_key: Optional[str] = resolved_key if resolved_key else None

        # Mode determination
        if offline_mode is not None:
            self._offline_mode = bool(offline_mode)
        else:
            self._offline_mode = self.api_key is None

        # Circuit breaker metrics
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_opened_at = 0.0
        self._circuit_cooldown_seconds = 60.0

    @property
    def is_offline(self) -> bool:
        """Indicates if the client is currently routing requests to offline extractor."""
        if self._offline_mode:
            return True
        if self._circuit_open:
            if time.time() - self._circuit_opened_at > self._circuit_cooldown_seconds:
                # Cooldown expired, half-open
                self._circuit_open = False
                self._consecutive_failures = 0
                return False
            return True
        return False

    def extract_telestration(
        self,
        image: Union[str, Path, bytes],
        match_minute: int = 0,
        context_desc: str = "",
        event_metadata: Optional[Dict[str, Any]] = None,
    ) -> TacticalTelestrationProps:
        """
        Extracts structured tactical telestrations from an image keyframe.
        Automatically falls back to offline extractor if offline mode is active or API fails.
        """
        ctx = dict(event_metadata or {})
        if "match_minute" not in ctx:
            ctx["match_minute"] = match_minute

        # Use Next-Gen TacticalVisionEngine with dynamic failover and lineup awareness
        try:
            from src.vision.tactical_vision_engine import TacticalVisionEngine
            engine = TacticalVisionEngine(api_key=self.api_key)
            if isinstance(image, (str, Path)) and Path(image).is_file():
                return engine.analyze_frame(image, ctx)
        except Exception as e:
            logger.warning("TacticalVisionEngine execution warning (%s). Falling back to standard pipeline.", e)

        # If offline or circuit open, bypass network call immediately
        if self.is_offline:
            logger.info("GeminiVisionClient running in offline/cooldown mode. Generating CV tactical geometry on turf.")
            return self.offline_extractor.extract_props(ctx, image=image)

        # Encode image to Base64
        try:
            if isinstance(image, bytes):
                base64_data = VisionPromptBuilder.encode_image_bytes(image)
            else:
                base64_data, _ = VisionPromptBuilder.encode_image_file(image)
        except Exception as e:
            logger.warning("Failed to encode freeze frame image (%s). Falling back to offline extractor.", e)
            return self.offline_extractor.extract_props(ctx, image=image)

        # Build wire request payload
        payload = VisionPromptBuilder.build_request(
            base64_image=base64_data,
            match_minute=match_minute,
            context_desc=context_desc or "Extract key players and tactical formations.",
            system_instruction_text=CANONICAL_SYSTEM_INSTRUCTION,
        )

        # Execute API call with retries and failover
        try:
            response_json = self._call_api_with_retry(payload)
            return self._parse_and_validate(response_json, ctx, image=image)
        except Exception as e:
            logger.warning("Gemini Vision API execution failed (%s). Activating graceful offline failover.", e)
            self._record_failure()
            return self.offline_extractor.extract_props(ctx, image=image)

    def analyze_tactical_event(
        self,
        image: Union[str, Path, bytes],
        tactical_event: Any,
    ) -> TacticalTelestrationProps:
        """Convenience method accepting a TacticalEvent domain model from Milestone 2."""
        minute = getattr(tactical_event, "match_minute", 0)
        hook = getattr(tactical_event, "narrative_hook", "")
        ev_type = getattr(tactical_event, "event_type", "turnover")
        team = getattr(tactical_event, "team", "")
        players = getattr(tactical_event, "players_involved", [])

        ctx: Dict[str, Any] = {
            "match_minute": minute,
            "event_type": ev_type,
            "team": team,
            "players_involved": players,
            "narrative_hook": hook,
        }
        desc = f"{ev_type.upper()} by {team}. {hook}".strip()
        return self.extract_telestration(image, match_minute=minute, context_desc=desc, event_metadata=ctx)

    def _call_api_with_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Calls Gemini API with exponential backoff on transient errors."""
        url = self.endpoint_url
        if self.api_key and "key=" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}key={urllib.parse.quote(self.api_key)}"

        req_data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
        }
        if self.api_key:
            headers["x-goog-api-key"] = self.api_key

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    resp_bytes = resp.read()
                    data = json.loads(resp_bytes.decode("utf-8"))
                    self._consecutive_failures = 0
                    return data
            except urllib.error.HTTPError as e:
                last_error = e
                status = e.code
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8", errors="ignore")
                except Exception:
                    pass
                logger.warning("Gemini Vision HTTP %s on attempt %s/%s: %s", status, attempt + 1, self.max_retries, err_body[:200])

                if status in (400, 401, 403):
                    self._circuit_open = True
                    self._circuit_opened_at = time.time()
                    break
                # Retry on 429 / 500 / 503
                backoff = 0.5 * (2 ** attempt)
                time.sleep(backoff)
            except Exception as e:
                last_error = e
                logger.warning("Gemini Vision transport error on attempt %s/%s: %s", attempt + 1, self.max_retries, e)
                backoff = 0.5 * (2 ** attempt)
                time.sleep(backoff)

        raise RuntimeError(f"Gemini Vision API request failed after {self.max_retries} attempts: {last_error}")

    def _parse_and_validate(
        self,
        response_json: Dict[str, Any],
        ctx: Dict[str, Any],
        image: Optional[Union[str, Path, bytes]] = None,
    ) -> TacticalTelestrationProps:
        """Extracts text candidate from API response, strips fences, and parses into props."""
        try:
            candidates = response_json.get("candidates", [])
            if not candidates:
                raise ValueError("No candidates returned from Gemini Vision API.")

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise ValueError("No content parts in first candidate.")

            raw_text = parts[0].get("text", "")
            clean_text = VisionPromptBuilder.sanitize_json_response(raw_text)
            parsed_dict = json.loads(clean_text)

            return TacticalFormationParser.parse_formation(parsed_dict)
        except Exception as e:
            logger.warning("Failed to parse Gemini Vision structured response (%s). Falling back to offline extractor.", e)
            return self.offline_extractor.extract_props(ctx, image=image)

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= 3:
            self._circuit_open = True
            self._circuit_opened_at = time.time()
            self._circuit_cooldown_seconds = 600.0
            logger.warning("GeminiVisionClient circuit breaker tripped OPEN for %ss (fast offline failover active).", self._circuit_cooldown_seconds)
