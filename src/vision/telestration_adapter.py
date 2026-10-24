"""
src/vision/telestration_adapter.py
Remotion Telestration Adapter & Synthetic Tactical Geometry Generator.

Transforms raw coordinates and match event telemetry into broadcast-grade Remotion
TacticalTelestrationProps schemas matching TacticalTelestrationOverlay.tsx. Provides
a comprehensive coordinate normalization engine, degenerate entity pruning, and
deterministic offline tactical scenarios.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.vision.tactical_parser import (
    TacticalFormationParser,
    TacticalGeometryExtractor,
    TacticalTelestrationProps,
    TelestrationLine,
    TelestrationPlayer,
    TelestrationPolygon,
)
from src.vision.gemini_client import (
    GeminiVisionClient,
    OfflineTacticalExtractor,
    VisionPromptBuilder,
)

logger = logging.getLogger("vision.telestration_adapter")


# =============================================================================
# 1. High-Precision Coordinate Normalization Engine
# =============================================================================

class CoordinateNormalizer:
    """
    Coordinates normalization and boundary clamping engine.
    Supports 0..100 percentages, 1920x1080 pixels, and unit ratio floats (0.0..1.0).
    """

    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamps a numeric scalar strictly within [min_val, max_val]."""
        try:
            val = float(value)
        except (ValueError, TypeError):
            return min_val
        return max(min_val, min(max_val, val))

    @classmethod
    def normalize_coordinate_pair(
        cls,
        x: float,
        y: float,
        frame_w: int = 1920,
        frame_h: int = 1080,
        target_mode: str = "auto",  # 'auto', 'percentage', 'pixels'
    ) -> Tuple[float, float]:
        """
        Normalizes (x, y) coordinates with automatic ratio detection and boundary clamping.
        """
        try:
            x_val = float(x)
            y_val = float(y)
        except (ValueError, TypeError):
            x_val, y_val = 0.0, 0.0

        # 1. Detect unit ratio interval (0.0 < max <= 1.0)
        is_ratio = (0.0 < max(abs(x_val), abs(y_val)) <= 1.0)
        if is_ratio and target_mode != "pixels":
            x_val *= 100.0
            y_val *= 100.0

        # 2. Determine coordinate mode
        if target_mode == "percentage":
            is_percentage = True
        elif target_mode == "pixels":
            is_percentage = False
        else:
            # Auto mode: if within percentage magnitude, treat as percentage
            is_percentage = max(abs(x_val), abs(y_val)) <= 150.0

        if is_percentage:
            norm_x = cls.clamp(x_val, 0.0, 100.0)
            norm_y = cls.clamp(y_val, 0.0, 100.0)
        else:
            norm_x = cls.clamp(x_val, 0.0, float(frame_w))
            norm_y = cls.clamp(y_val, 0.0, float(frame_h))

        return round(norm_x, 2), round(norm_y, 2)

    @classmethod
    def to_pixels(
        cls,
        x: float,
        y: float,
        frame_w: int = 1920,
        frame_h: int = 1080,
    ) -> Tuple[float, float]:
        """Converts percentage coordinates (0..100) to absolute screen pixels."""
        norm_x, norm_y = cls.normalize_coordinate_pair(x, y, frame_w, frame_h, target_mode="percentage")
        px_x = cls.clamp((norm_x / 100.0) * frame_w, 0.0, float(frame_w))
        px_y = cls.clamp((norm_y / 100.0) * frame_h, 0.0, float(frame_h))
        return round(px_x, 2), round(px_y, 2)

    @classmethod
    def to_percentage(
        cls,
        x: float,
        y: float,
        frame_w: int = 1920,
        frame_h: int = 1080,
    ) -> Tuple[float, float]:
        """Converts absolute screen pixels (0..frame_w, 0..frame_h) to percentages (0..100)."""
        x_val = cls.clamp(float(x), 0.0, float(frame_w))
        y_val = cls.clamp(float(y), 0.0, float(frame_h))
        pct_x = (x_val / float(frame_w)) * 100.0 if frame_w > 0 else 0.0
        pct_y = (y_val / float(frame_h)) * 100.0 if frame_h > 0 else 0.0
        return round(cls.clamp(pct_x, 0.0, 100.0), 2), round(cls.clamp(pct_y, 0.0, 100.0), 2)


# =============================================================================
# 2. Canonical Tactical Scenarios & Deterministic Synthetic Generator
# =============================================================================

CANONICAL_TACTICAL_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "high_press_breakout": {
        "match_minute": 34,
        "title": "MINUTE 34' // PRESS BREAKOUT",
        "subtitle": "PROGRESSIVE MIDFIELD OVERLOAD",
        "players": [
            {"id": "p_pedri", "name": "Pedri", "number": 8, "role": "CM", "x": 45.2, "y": 62.4, "color": "#ffff00", "showRing": True, "showBeam": True, "showBadge": True},
            {"id": "p_yamal", "name": "L. Yamal", "number": 19, "role": "RW", "x": 78.5, "y": 35.1, "color": "#00e676", "showRing": True, "showBeam": False, "showBadge": True},
            {"id": "p_lewa", "name": "R. Lewandowski", "number": 9, "role": "CF", "x": 62.0, "y": 48.0, "color": "#00e676", "showRing": True, "showBeam": False, "showBadge": True},
            {"id": "p_balde", "name": "A. Balde", "number": 3, "role": "LB", "x": 32.0, "y": 18.0, "color": "#00e676", "showRing": True, "showBeam": False, "showBadge": False},
            {"id": "p_martinez", "name": "I. Martinez", "number": 5, "role": "CB", "x": 26.0, "y": 42.0, "color": "#00e676", "showRing": True, "showBeam": False, "showBadge": False},
            {"id": "p_cubarsi", "name": "P. Cubarsi", "number": 2, "role": "CB", "x": 28.0, "y": 65.0, "color": "#00e676", "showRing": True, "showBeam": False, "showBadge": False},
            {"id": "p_kounde", "name": "J. Kounde", "number": 23, "role": "RB", "x": 38.0, "y": 82.0, "color": "#00e676", "showRing": True, "showBeam": False, "showBadge": False},
            {"id": "p_presser", "name": "Pressing Midfielder", "number": 10, "role": "CM", "x": 52.0, "y": 58.0, "color": "#ef4444", "showRing": True, "showBeam": False, "showBadge": True},
        ],
        "lines": [
            {"id": "defensive_backline", "playerIds": ["p_balde", "p_martinez", "p_cubarsi", "p_kounde"], "color": "#00e676", "width": 4, "dashed": False},
            {"id": "pressing_corridor", "playerIds": ["p_presser", "p_pedri"], "color": "#ef4444", "width": 3, "dashed": True},
        ],
        "polygons": [
            {"id": "passing_triangle_midfield", "playerIds": ["p_pedri", "p_yamal", "p_lewa"], "fillColor": "rgba(0, 230, 118, 0.25)", "borderColor": "#00e676", "pattern": "stripes", "label": "PASSING TRIANGLE"},
        ],
    },
    "defensive_third_turnover": {
        "match_minute": 24,
        "title": "MINUTE 24' // DEFENSIVE THIRD TURNOVER",
        "subtitle": "COUNTER-PRESSING TRANSITION",
        "players": [
            {"id": "p_carrier", "name": "Dispossessed Defender", "number": 4, "role": "CB", "x": 22.0, "y": 48.0, "color": "#ffff00", "showRing": True, "showBeam": True, "showBadge": True},
            {"id": "p_press1", "name": "Pressing Striker", "number": 9, "role": "CF", "x": 28.0, "y": 42.0, "color": "#ef4444", "showRing": True, "showBeam": False, "showBadge": True},
            {"id": "p_press2", "name": "Pressing Winger", "number": 11, "role": "LW", "x": 25.0, "y": 62.0, "color": "#ef4444", "showRing": True, "showBeam": False, "showBadge": True},
            {"id": "p_press3", "name": "Attacking Mid", "number": 10, "role": "AM", "x": 35.0, "y": 50.0, "color": "#ef4444", "showRing": True, "showBeam": False, "showBadge": True},
        ],
        "lines": [
            {"id": "pressing_corridor", "playerIds": ["p_press1", "p_carrier"], "color": "#ef4444", "width": 3, "dashed": True},
        ],
        "polygons": [
            {"id": "pressing_trap", "playerIds": ["p_press1", "p_press2", "p_press3"], "fillColor": "rgba(239, 68, 68, 0.25)", "borderColor": "#ef4444", "pattern": "stripes", "label": "PRESS TRAP"},
        ],
    },
    "midfield_overload_triangle": {
        "match_minute": 15,
        "title": "MINUTE 15' // MIDFIELD OVERLOAD",
        "subtitle": "POSITIONAL SUPERIORITY IN ZONE 14",
        "players": [
            {"id": "p_pivot", "name": "Deep Pivot", "number": 6, "role": "DM", "x": 48.0, "y": 50.0, "color": "#ffff00", "showRing": True, "showBeam": True, "showBadge": True},
            {"id": "p_am", "name": "Attacking Mid", "number": 10, "role": "AM", "x": 62.0, "y": 42.0, "color": "#00e676", "showRing": True, "showBeam": False, "showBadge": True},
            {"id": "p_winger", "name": "Inverted Winger", "number": 7, "role": "RW", "x": 58.0, "y": 68.0, "color": "#00e676", "showRing": True, "showBeam": False, "showBadge": True},
        ],
        "lines": [],
        "polygons": [
            {"id": "overload_triangle", "playerIds": ["p_pivot", "p_am", "p_winger"], "fillColor": "rgba(0, 230, 118, 0.25)", "borderColor": "#00e676", "pattern": "stripes", "label": "OVERLOAD TRIANGLE"},
        ],
    },
    "counter_attack_transition": {
        "match_minute": 68,
        "title": "MINUTE 68' // COUNTER-ATTACK BREAKOUT",
        "subtitle": "VERTICAL PROGRESSION",
        "players": [
            {"id": "p_carrier", "name": "Ball Carrier", "number": 8, "role": "CM", "x": 42.0, "y": 50.0, "color": "#ffff00", "showRing": True, "showBeam": True, "showBadge": True},
            {"id": "p_winger_left", "name": "Left Winger", "number": 11, "role": "LW", "x": 65.0, "y": 25.0, "color": "#00e676", "showRing": True, "showBeam": False, "showBadge": True},
            {"id": "p_striker", "name": "Striker", "number": 9, "role": "CF", "x": 70.0, "y": 52.0, "color": "#00e676", "showRing": True, "showBeam": False, "showBadge": True},
            {"id": "p_cb1", "name": "Retreating CB", "number": 4, "role": "CB", "x": 62.0, "y": 45.0, "color": "#ef4444", "showRing": True, "showBeam": False, "showBadge": False},
            {"id": "p_cb2", "name": "Retreating CB", "number": 5, "role": "CB", "x": 64.0, "y": 60.0, "color": "#ef4444", "showRing": True, "showBeam": False, "showBadge": False},
        ],
        "lines": [
            {"id": "defensive_delay", "playerIds": ["p_cb1", "p_cb2"], "color": "#ef4444", "width": 4, "dashed": False},
        ],
        "polygons": [
            {"id": "transition_triangle", "playerIds": ["p_carrier", "p_winger_left", "p_striker"], "fillColor": "rgba(0, 230, 118, 0.25)", "borderColor": "#00e676", "pattern": "stripes", "label": "TRANSITION VECTOR"},
        ],
    },
    "deep_block_compactness": {
        "match_minute": 82,
        "title": "MINUTE 82' // LOW BLOCK COMPACTNESS",
        "subtitle": "5-MAN DEFENSIVE CHAIN",
        "players": [
            {"id": "p_lb", "name": "LB", "number": 3, "role": "LB", "x": 18.0, "y": 15.0, "color": "#00e676", "showRing": True, "showBeam": False, "showBadge": False},
            {"id": "p_lcb", "name": "LCB", "number": 4, "role": "CB", "x": 16.0, "y": 32.0, "color": "#00e676", "showRing": True, "showBeam": False, "showBadge": False},
            {"id": "p_cb", "name": "CB", "number": 5, "role": "CB", "x": 15.0, "y": 50.0, "color": "#ffff00", "showRing": True, "showBeam": True, "showBadge": True},
            {"id": "p_rcb", "name": "RCB", "number": 6, "role": "CB", "x": 16.0, "y": 68.0, "color": "#00e676", "showRing": True, "showBeam": False, "showBadge": False},
            {"id": "p_rb", "name": "RB", "number": 2, "role": "RB", "x": 18.0, "y": 85.0, "color": "#00e676", "showRing": True, "showBeam": False, "showBadge": False},
        ],
        "lines": [
            {"id": "defensive_backline", "playerIds": ["p_lb", "p_lcb", "p_cb", "p_rcb", "p_rb"], "color": "#00e676", "width": 4, "dashed": False},
        ],
        "polygons": [],
    },
}


class SyntheticTacticalGenerator:
    """
    Deterministic offline tactical fixture generator.
    Produces broadcast-grade coordinates for test execution and offline fallback.
    """

    @classmethod
    def get_scenario(cls, scenario_id: str) -> TacticalTelestrationProps:
        """Returns a canonical scenario by key ID."""
        raw = CANONICAL_TACTICAL_SCENARIOS.get(scenario_id)
        if not raw:
            raw = CANONICAL_TACTICAL_SCENARIOS["high_press_breakout"]
        return TacticalFormationParser.parse_formation(raw)

    @classmethod
    def generate_from_context(cls, event_context: Optional[Dict[str, Any]] = None) -> TacticalTelestrationProps:
        """Procedurally synthesizes tactical telestrations adapted to a specific event context."""
        ctx = event_context or {}
        event_type = str(ctx.get("event_type", "turnover")).lower()

        if "goal" in event_type:
            scenario = "counter_attack_transition"
        elif "turnover" in event_type or "recovery" in event_type:
            scenario = "defensive_third_turnover"
        elif "block" in event_type or "sub" in event_type:
            scenario = "deep_block_compactness"
        elif "overload" in event_type or "chance" in event_type:
            scenario = "midfield_overload_triangle"
        else:
            scenario = "high_press_breakout"

        base_data = dict(CANONICAL_TACTICAL_SCENARIOS[scenario])

        # Adapt minute, title, and team
        minute = ctx.get("match_minute", base_data.get("match_minute", 34))
        team = ctx.get("team", "TEAM")
        title = f"MINUTE {minute}' // {event_type.upper().replace('_', ' ')}"
        subtitle = f"{team.upper()} TACTICAL SETUP"

        raw_copy = {
            "players": [dict(p) for p in base_data.get("players", [])],
            "lines": [dict(l) for l in base_data.get("lines", [])],
            "polygons": [dict(poly) for poly in base_data.get("polygons", [])],
            "title": title,
            "subtitle": subtitle,
        }

        # Inject known player names if provided
        inv = ctx.get("players_involved", [])
        if inv and raw_copy["players"]:
            for i, p_info in enumerate(inv):
                if i >= len(raw_copy["players"]):
                    break
                p_name = p_info if isinstance(p_info, str) else p_info.get("name", "")
                if p_name:
                    raw_copy["players"][i]["name"] = p_name
                    raw_copy["players"][i]["showBadge"] = True

        return TacticalFormationParser.parse_formation(raw_copy)


# =============================================================================
# 3. Remotion Telestration Adapter
# =============================================================================

class RemotionTelestrationAdapter:
    """
    Formats tactical geometries strictly complying with the TypeScript
    TacticalTelestrationProps interface required by TacticalTelestrationOverlay.tsx.
    """

    @classmethod
    def format_props(cls, props: TacticalTelestrationProps) -> Dict[str, Any]:
        """Converts TacticalTelestrationProps into JSON-serializable dictionary."""
        return props.to_dict()

    @classmethod
    def to_remotion_segment(
        cls,
        segment_id: str,
        clip_or_frame_url: str,
        start_time: float,
        duration: float,
        telestration_props: TacticalTelestrationProps,
        segment_type: str = "freeze_frame",
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        match_minute: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Packages tactical telestration into a full TacticalSegment object for TacticalMasterclass."""
        return {
            "id": segment_id,
            "type": segment_type,
            "url": clip_or_frame_url,
            "startTime": round(start_time, 2),
            "duration": round(duration, 2),
            "title": title or telestration_props.title,
            "subtitle": subtitle or telestration_props.subtitle,
            "matchMinute": match_minute,
            "telestration": cls.format_props(telestration_props),
            "lowerThird": {
                "line1": title or telestration_props.title or "TACTICAL BREAKDOWN",
                "line2": subtitle or telestration_props.subtitle or "",
                "tag": f"MIN {match_minute}'" if match_minute is not None else "ANALYSIS",
            },
        }


# =============================================================================
# 4. Master Unified Pipeline Entry Point
# =============================================================================

def extract_tactical_telestration(
    freeze_frame_path: str,
    event_context: Optional[Dict[str, Any]] = None,
    *,
    client: Optional[GeminiVisionClient] = None,
    offline_mode: Optional[bool] = None,
    frame_w: int = 1920,
    frame_h: int = 1080,
) -> TacticalTelestrationProps:
    """
    Extracts structured Remotion tactical telestrations from a freeze-frame image.

    Args:
        freeze_frame_path: Absolute or relative path to 1080p JPEG/PNG freeze frame.
        event_context: Optional dictionary containing match event metadata.
        client: Optional custom GeminiVisionClient instance.
        offline_mode: If True, forces synthetic generator. If None, auto-detects.
        frame_w: Target video width (default 1920).
        frame_h: Target video height (default 1080).

    Returns:
        TacticalTelestrationProps: Fully sanitized, boundary-clamped, typed object
                                   matching Remotion TacticalTelestrationOverlay schema.
    """
    ctx = event_context or {}
    path = Path(freeze_frame_path) if freeze_frame_path else None
    has_image = bool(path and path.is_file())

    # If no image file on disk, generate roster-aware tactical geometry with rule-based fallback
    if not has_image:
        try:
            from src.vision.player_detector import CVPlayerDetector
            res = CVPlayerDetector.generate_roster_telestration(None, ctx)
            if res and res.players and len(res.players) >= 2:
                return res
        except Exception:
            pass
        extractor = OfflineTacticalExtractor()
        return extractor.extract_props(ctx)

    vision_client = client or GeminiVisionClient(offline_mode=offline_mode)

    # Execute vision extraction via GeminiVisionClient (with automatic CVPlayerDetector fallback)
    minute = ctx.get("match_minute", 0)
    ev_type = ctx.get("event_type", "")
    team = ctx.get("team", "")
    desc = ctx.get("narrative_hook", f"{ev_type.upper()} by {team}")

    return vision_client.extract_telestration(
        image=str(path),
        match_minute=minute,
        context_desc=desc,
        event_metadata=ctx,
    )
