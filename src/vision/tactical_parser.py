"""
src/vision/tactical_parser.py
Tactical Geometry & Formation Parser for Broadcast Football Telestration.

Parses raw vision entity coordinates, detects tactical structures (defensive backlines,
pressing lines, passing triangles), validates geometric properties, applies neon
color palettes and HUD badges, and outputs schemas conforming to TacticalTelestrationProps.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from src.vision.tactical_geometry_optimizer import (
    LineSegment,
    Point2D,
    ShortestLinkGeometryEngine,
    TacticalLineStyle,
    TacticalLineType,
)

logger = logging.getLogger("vision.tactical_parser")


# =============================================================================
# 1. Tactical Roles & Taxonomy
# =============================================================================

class TacticalRole(str, Enum):
    """Standardized football tactical player roles."""
    GK = "GK"
    CB = "CB"
    LB = "LB"
    RB = "RB"
    LWB = "LWB"
    RWB = "RWB"
    SW = "SW"
    DM = "DM"
    CM = "CM"
    AM = "AM"
    LW = "LW"
    RW = "RW"
    CF = "CF"
    ST = "CF"
    PLAYER = "PLAYER"


ROLE_NORMALIZATION_MAP: Dict[str, str] = {
    "GK": "GK", "GOALKEEPER": "GK", "GOALIE": "GK", "PORTERO": "GK",
    "CB": "CB", "CENTRE_BACK": "CB", "CENTER_BACK": "CB", "LCB": "CB", "RCB": "CB", "DEF": "CB",
    "LB": "LB", "LEFT_BACK": "LB", "LWB": "LWB", "LEFT_WING_BACK": "LWB",
    "RB": "RB", "RIGHT_BACK": "RB", "RWB": "RWB", "RIGHT_WING_BACK": "RWB",
    "SW": "SW", "SWEEPER": "SW", "LIBERO": "SW",
    "DM": "DM", "DEFENSIVE_MIDFIELDER": "DM", "CDM": "DM", "PIVOT": "DM", "PIVOTE": "DM",
    "CM": "CM", "CENTRAL_MIDFIELDER": "CM", "LCM": "CM", "RCM": "CM", "MID": "CM",
    "AM": "AM", "ATTACKING_MIDFIELDER": "AM", "CAM": "AM", "TREQUARTISTA": "AM", "ENGANCHE": "AM",
    "LW": "LW", "LEFT_WINGER": "LW", "LEFT_MIDFIELDER": "LW", "LM": "LW",
    "RW": "RW", "RIGHT_WINGER": "RW", "RIGHT_MIDFIELDER": "RW", "RM": "RW",
    "CF": "CF", "CENTRE_FORWARD": "CF", "ST": "CF", "STRIKER": "CF", "FWD": "CF", "DELANTERO": "CF",
    "PLAYER": "PLAYER",
}

BACKLINE_ROLES: Set[str] = {"CB", "LB", "RB", "LWB", "RWB", "SW"}
PRESSING_ROLES: Set[str] = {"CF", "LW", "RW", "AM", "CM", "DM"}


# =============================================================================
# 2. Broadcast Neon Color Palettes
# =============================================================================

CLUB_NEON_PALETTES: Dict[str, Dict[str, str]] = {
    "BARCELONA": {
        "carrier": "#ffff00",
        "teammate": "#00e5ff",
        "backline": "#00e676",
        "opponent": "#ef4444",
        "overload": "#a855f7",
        "fill": "rgba(0, 229, 255, 0.25)",
    },
    "REAL_MADRID": {
        "carrier": "#ffff00",
        "teammate": "#ffffff",
        "backline": "#00e676",
        "opponent": "#ef4444",
        "overload": "#febe10",
        "fill": "rgba(254, 190, 16, 0.25)",
    },
    "MANCHESTER_CITY": {
        "carrier": "#ffff00",
        "teammate": "#00f0ff",
        "backline": "#00e676",
        "opponent": "#ef4444",
        "overload": "#a855f7",
        "fill": "rgba(0, 240, 255, 0.25)",
    },
    "ARSENAL": {
        "carrier": "#ffff00",
        "teammate": "#ff1744",
        "backline": "#00e676",
        "opponent": "#ef4444",
        "overload": "#9c824a",
        "fill": "rgba(255, 23, 68, 0.25)",
    },
    "LIVERPOOL": {
        "carrier": "#ffff00",
        "teammate": "#ff2a4b",
        "backline": "#00e676",
        "opponent": "#ef4444",
        "overload": "#00e676",
        "fill": "rgba(255, 42, 75, 0.25)",
    },
    "BAYERN": {
        "carrier": "#ffff00",
        "teammate": "#ff1744",
        "backline": "#00e676",
        "opponent": "#ef4444",
        "overload": "#ffab00",
        "fill": "rgba(255, 23, 68, 0.25)",
    },
    "PSG": {
        "carrier": "#ffff00",
        "teammate": "#a855f7",
        "backline": "#00e676",
        "opponent": "#ef4444",
        "overload": "#00f0ff",
        "fill": "rgba(168, 85, 247, 0.25)",
    },
    "INTER": {
        "carrier": "#ffff00",
        "teammate": "#00529f",
        "backline": "#00e676",
        "opponent": "#ef4444",
        "overload": "#00f0ff",
        "fill": "rgba(0, 240, 255, 0.25)",
    },
    "DEFAULT": {
        "carrier": "#ffff00",
        "teammate": "#00e676",
        "backline": "#00e676",
        "opponent": "#ef4444",
        "overload": "#a855f7",
        "fill": "rgba(0, 230, 118, 0.25)",
        "fill_overload": "rgba(168, 85, 247, 0.25)",
        "fill_trap": "rgba(239, 68, 68, 0.25)",
    },
}


class NeonColorManager:
    """Resolves functional and club-specific neon palettes."""

    @classmethod
    def resolve_player_color(
        cls,
        role: str,
        is_carrier: bool = False,
        is_opponent: bool = False,
        is_overload: bool = False,
        club: Optional[str] = None,
    ) -> str:
        palette = CLUB_NEON_PALETTES.get(str(club).upper(), CLUB_NEON_PALETTES["DEFAULT"])
        if is_carrier:
            return palette.get("carrier", "#ffff00")
        if is_opponent:
            return palette.get("opponent", "#ef4444")
        if is_overload:
            return palette.get("overload", "#a855f7")
        if role in BACKLINE_ROLES:
            return palette.get("backline", "#00e676")
        return palette.get("teammate", "#00e676")


# =============================================================================
# 3. Telestration Domain Entities
# =============================================================================

@dataclass
class TelestrationPlayer:
    """Player spotlight entity anchored at boots on the pitch surface."""
    id: str
    x: float
    y: float
    name: Optional[str] = None
    number: Optional[Union[int, str]] = None
    role: str = "PLAYER"
    color: str = "#ffff00"
    showRing: bool = True
    showBeam: bool = False
    showBadge: bool = True
    team: Optional[str] = None

    # Pythonic snake_case property aliases
    @property
    def show_ring(self) -> bool:
        return self.showRing

    @property
    def show_beam(self) -> bool:
        return self.showBeam

    @property
    def show_badge(self) -> bool:
        return self.showBadge

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "name": self.name or "",
            "number": self.number,
            "role": self.role,
            "x": round(float(self.x), 2),
            "y": round(float(self.y), 2),
            "color": self.color,
            "showRing": self.showRing,
            "showBeam": self.showBeam,
            "showBadge": self.showBadge,
        }
        return d


@dataclass
class TelestrationLine:
    """Connecting tactical line (defensive chain, dashed pressing line, curved pass, or dotted run)."""
    id: str
    playerIds: List[str]
    color: str = "#00e676"
    width: int = 4
    dashed: bool = False
    style: str = "solid"  # "solid" | "dashed" | "dotted" | "bezier"
    line_type: str = "tactical_structure"  # "tactical_structure" | "ball_pass" | "player_run"
    halo_padding_px: float = 22.0
    bezier_control: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.dashed and self.style == "solid":
            self.style = "dashed"

    @property
    def player_ids(self) -> List[str]:
        return self.playerIds

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "playerIds": list(self.playerIds),
            "color": self.color,
            "width": self.width,
            "dashed": self.dashed or (self.style == "dashed"),
            "style": self.style,
            "lineType": self.line_type,
            "haloPaddingPx": self.halo_padding_px,
        }
        if self.bezier_control:
            d["bezierControl"] = dict(self.bezier_control)
        return d


@dataclass
class TelestrationPolygon:
    """Tactical shaded polygon (passing triangle, pressing trap, overload zone, vacated empty space)."""
    id: str
    playerIds: List[str] = field(default_factory=list)
    points: Optional[List[Dict[str, float]]] = None
    fillColor: str = "rgba(0, 230, 118, 0.25)"
    borderColor: str = "#00e676"
    pattern: str = "stripes"
    label: str = "PASSING TRIANGLE"
    isVacatedZone: bool = False

    @property
    def player_ids(self) -> List[str]:
        return self.playerIds

    @property
    def fill_color(self) -> str:
        return self.fillColor

    @property
    def border_color(self) -> str:
        return self.borderColor

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "playerIds": list(self.playerIds),
            "fillColor": self.fillColor,
            "borderColor": self.borderColor,
            "pattern": self.pattern,
            "label": self.label,
        }
        if self.points:
            d["points"] = [{"x": round(float(pt["x"]), 2), "y": round(float(pt["y"]), 2)} for pt in self.points]
        if self.isVacatedZone:
            d["isVacatedZone"] = True
        return d


@dataclass
class TacticalTelestrationProps:
    """Root props object matching Remotion TacticalTelestrationOverlay schema."""
    players: List[TelestrationPlayer] = field(default_factory=list)
    lines: List[TelestrationLine] = field(default_factory=list)
    polygons: List[TelestrationPolygon] = field(default_factory=list)
    title: Optional[str] = None
    subtitle: Optional[str] = None
    delay: int = 0

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "players": [p.to_dict() for p in self.players],
            "lines": [l.to_dict() for l in self.lines],
            "polygons": [poly.to_dict() for poly in self.polygons],
        }
        if self.title:
            res["title"] = self.title
        if self.subtitle:
            res["subtitle"] = self.subtitle
        if self.delay > 0:
            res["delay"] = self.delay
        return res

    def __getitem__(self, item: str) -> Any:
        return self.to_dict()[item]

    def get(self, item: str, default: Any = None) -> Any:
        return self.to_dict().get(item, default)

    def __contains__(self, item: str) -> bool:
        return item in self.to_dict()


# =============================================================================
# 4. Geometry Builders
# =============================================================================

class TacticalBacklineBuilder:
    """
    Builds unbroken solid lines connecting 3-5 defenders across pitch width.
    Projects backline player points onto their 1st Principal Component (PCA eigenvector)
    and sorts monotonically along that axis instead of raw screen y.
    """

    @staticmethod
    def _sort_by_pca_projection(
        player_ids: List[str],
        valid_map: Dict[str, TelestrationPlayer],
    ) -> List[str]:
        """
        Computes the shortest non-intersecting open chain connecting defenders using
        ShortestLinkGeometryEngine (Euclidean distance matrix + 2-opt uncrossing).
        """
        if len(player_ids) <= 1:
            return player_ids
        if len(player_ids) == 2:
            return sorted(player_ids, key=lambda pid: (valid_map[pid].y, valid_map[pid].x))

        pts = [(valid_map[pid].x, valid_map[pid].y) for pid in player_ids]
        chain_indices = ShortestLinkGeometryEngine.compute_shortest_open_chain(pts)
        return [player_ids[idx] for idx in chain_indices]

    @staticmethod
    def build_backline(
        players: Sequence[TelestrationPlayer],
        raw_line: Optional[Dict[str, Any]] = None,
        color: str = "#00e676",
        width: int = 4,
    ) -> Optional[TelestrationLine]:
        valid_map = {p.id: p for p in players}

        # 1. If explicit line provided, validate and sort along shortest path without crisscross
        if raw_line and "playerIds" in raw_line:
            ids = [pid for pid in raw_line["playerIds"] if pid in valid_map]
            if len(ids) < 2:
                return None
            sorted_ids = TacticalBacklineBuilder._sort_by_pca_projection(ids, valid_map)
            return TelestrationLine(
                id=str(raw_line.get("id", "defensive_backline")),
                playerIds=sorted_ids,
                color=raw_line.get("color", color),
                width=int(raw_line.get("width", width)),
                dashed=False,
                style=str(raw_line.get("style", "solid")),
                line_type=str(raw_line.get("line_type", "tactical_structure")),
                halo_padding_px=float(raw_line.get("halo_padding_px", 22.0)),
            )

        # 2. Auto-detect backline from player roles & positions
        defenders = [p for p in players if p.role in BACKLINE_ROLES]

        # If fewer than 3 defenders found by role, look for deepest cluster in defensive third
        if len(defenders) < 3:
            sorted_by_depth = sorted(players, key=lambda p: p.x)
            candidates = [
                p for p in sorted_by_depth
                if p.role not in PRESSING_ROLES and p.role != "GK" and p.x <= 50.0
            ][:5]
            if len(candidates) >= 3:
                defenders = candidates

        if len(defenders) < 2:
            return None

        # Clamp to max 5 defenders
        if len(defenders) > 5:
            defenders = sorted(defenders, key=lambda p: p.x)[:5]

        # Sort monotonically along shortest non-crossing chain
        ordered_ids = TacticalBacklineBuilder._sort_by_pca_projection([d.id for d in defenders], valid_map)

        return TelestrationLine(
            id="defensive_backline",
            playerIds=ordered_ids,
            color=color,
            width=width,
            dashed=False,
            style="solid",
            line_type="tactical_structure",
            halo_padding_px=22.0,
        )


class PressingLineBuilder:
    """
    Builds dashed high-alert lines connecting pressing players and pressure corridors.
    """

    @staticmethod
    def build_pressing_lines(
        players: Sequence[TelestrationPlayer],
        raw_lines: Optional[List[Dict[str, Any]]] = None,
        color: str = "#ef4444",
        width: int = 3,
    ) -> List[TelestrationLine]:
        valid_map = {p.id: p for p in players}
        results: List[TelestrationLine] = []

        # 1. Process explicit dashed lines
        if raw_lines:
            for rl in raw_lines:
                if not rl.get("dashed", False):
                    continue
                ids = [pid for pid in rl.get("playerIds", []) if pid in valid_map]
                if len(ids) < 2:
                    continue
                sorted_press_ids = ids
                if len(ids) >= 3:
                    press_pts = [(valid_map[pid].x, valid_map[pid].y) for pid in ids]
                    chain_idx = ShortestLinkGeometryEngine.compute_shortest_open_chain(press_pts)
                    sorted_press_ids = [ids[idx] for idx in chain_idx]

                results.append(TelestrationLine(
                    id=str(rl.get("id", f"pressing_line_{len(results)}")),
                    playerIds=sorted_press_ids,
                    color=rl.get("color", color),
                    width=int(rl.get("width", width)),
                    dashed=True,
                    style="dashed",
                    line_type="tactical_structure",
                    halo_padding_px=float(rl.get("halo_padding_px", 22.0)),
                ))

        # 2. Auto-detect pressing duels (opponent close to ball carrier) if no explicit dashed line
        if not results:
            carrier = next((p for p in players if p.color == "#ffff00" or p.showBeam), None)
            if carrier:
                for p in players:
                    if p.id == carrier.id:
                        continue
                    if p.color == "#ef4444" or p.role in PRESSING_ROLES:
                        dist = math.hypot(p.x - carrier.x, p.y - carrier.y)
                        if dist <= 25.0:  # Within 25% pitch distance
                            results.append(TelestrationLine(
                                id=f"pressing_corridor_{p.id}_{carrier.id}",
                                playerIds=[p.id, carrier.id],
                                color=color,
                                width=width,
                                dashed=True,
                                style="dashed",
                                line_type="tactical_structure",
                                halo_padding_px=22.0,
                            ))
                            break

        return results


class PassingTriangleBuilder:
    """
    Constructs non-self-intersecting passing polygons with broadcast hatched stripes.
    Uses polar centroid angular sorting to avoid bowtie / hourglass self-intersections.
    """

    @staticmethod
    def clamp_coordinate(coord: float, max_bound: float = 100.0) -> float:
        try:
            val = float(coord)
        except (ValueError, TypeError):
            return 0.0
        return max(0.0, min(max_bound, val))

    @staticmethod
    def _calculate_polygon_area(points: List[Tuple[float, float]]) -> float:
        """Calculates area using the Shoelace formula."""
        n = len(points)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        return 0.5 * abs(area)

    @classmethod
    def calculate_effective_area(cls, points: List[Tuple[float, float]]) -> float:
        """
        Calculates Shoelace area normalized by screen depth:
        effective_area = raw_area / ((y_centroid / 100.0) ** 1.5)
        Compensates for camera perspective foreshortening so far-side triangles
        aren't falsely rejected.
        """
        raw_area = cls._calculate_polygon_area(points)
        if not points or raw_area <= 0.0:
            return 0.0
        y_centroid = sum(p[1] for p in points) / len(points)
        depth_factor = max(0.01, (max(y_centroid, 1.0) / 100.0)) ** 1.5
        return raw_area / depth_factor

    @classmethod
    def _sort_counter_clockwise(
        cls,
        player_ids: List[str],
        valid_map: Dict[str, TelestrationPlayer],
    ) -> List[str]:
        """Orders polygon vertices counter-clockwise around centroid and uncrosses any self-intersections."""
        pts = [(valid_map[pid].x, valid_map[pid].y, pid) for pid in player_ids]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)

        # Sort by polar angle theta = atan2(y - cy, x - cx)
        sorted_pts = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        ordered_ids = [p[2] for p in sorted_pts]

        # Verify segments don't cross using ShortestLinkGeometryEngine
        poly_pts = [(valid_map[pid].x, valid_map[pid].y) for pid in ordered_ids]
        uncrossed_indices = ShortestLinkGeometryEngine.uncross_chain_2opt(list(range(len(poly_pts))), poly_pts)
        return [ordered_ids[idx] for idx in uncrossed_indices]

    @classmethod
    def build_polygons(
        cls,
        players: Sequence[TelestrationPlayer],
        raw_polygons: Optional[List[Dict[str, Any]]] = None,
        default_fill: str = "rgba(0, 230, 118, 0.25)",
        default_border: str = "#00e676",
        min_area: float = 10.0,
    ) -> List[TelestrationPolygon]:
        valid_map = {p.id: p for p in players}
        results: List[TelestrationPolygon] = []

        if raw_polygons:
            for rp in raw_polygons:
                raw_pts = rp.get("points")
                is_vacated = bool(rp.get("isVacatedZone")) or bool(raw_pts and len(raw_pts) >= 3) or (
                    bool(rp.get("label")) and bool(re.search(r"vacat|empty|exploit|vacuum|pocket|space|zone", str(rp.get("label")), re.I))
                )

                # Case A: Arbitrary coordinate polygon (e.g. Empty Vacated Zone / Space)
                if raw_pts and len(raw_pts) >= 3:
                    clean_pts = []
                    for pt in raw_pts:
                        try:
                            clean_pts.append({
                                "x": cls.clamp_coordinate(pt.get("x", 50.0)),
                                "y": cls.clamp_coordinate(pt.get("y", 50.0)),
                            })
                        except Exception:
                            pass
                    if len(clean_pts) >= 3:
                        poly_pts = [(p["x"], p["y"]) for p in clean_pts]
                        effective_area = cls.calculate_effective_area(poly_pts)
                        if effective_area >= min_area:
                            pattern = rp.get("pattern", "stripes")
                            results.append(TelestrationPolygon(
                                id=str(rp.get("id", f"vacated_zone_{len(results)}")),
                                playerIds=[],
                                points=clean_pts,
                                fillColor=rp.get("fillColor", "rgba(245, 158, 11, 0.22)" if is_vacated else default_fill),
                                borderColor=rp.get("borderColor", "#f59e0b" if is_vacated else default_border),
                                pattern=pattern if pattern in ("stripes", "solid", "dots") else "stripes",
                                label=rp.get("label", "VACATED DEFENSIVE ZONE" if is_vacated else "TACTICAL SPACE"),
                                isVacatedZone=is_vacated,
                            ))
                            continue

                # Case B: Player-anchored polygon (Passing Triangle / Press Trap)
                ids = [pid for pid in rp.get("playerIds", []) if pid in valid_map]
                if len(ids) < 3:
                    continue

                # Ensure vertices are ordered counter-clockwise around centroid
                ordered_ids = cls._sort_counter_clockwise(ids, valid_map)

                # Check non-collinear area normalized by screen depth
                pts = [(valid_map[pid].x, valid_map[pid].y) for pid in ordered_ids]
                effective_area = cls.calculate_effective_area(pts)
                if effective_area < min_area:
                    logger.debug("Polygon %s rejected: effective area %s < %s", rp.get("id"), effective_area, min_area)
                    continue

                pattern = rp.get("pattern", "stripes")
                results.append(TelestrationPolygon(
                    id=str(rp.get("id", f"poly_{len(results)}")),
                    playerIds=ordered_ids,
                    points=[{"x": valid_map[pid].x, "y": valid_map[pid].y} for pid in ordered_ids],
                    fillColor=rp.get("fillColor", default_fill),
                    borderColor=rp.get("borderColor", default_border),
                    pattern=pattern if pattern in ("stripes", "solid", "dots") else "stripes",
                    label=rp.get("label", "PASSING TRIANGLE"),
                    isVacatedZone=is_vacated,
                ))

        return results


# =============================================================================
# 5. Master Formation Parser & Backwards-Compatible Extractor
# =============================================================================

class TacticalFormationParser:
    """Master orchestration engine for tactical entity geometry parsing."""

    @staticmethod
    def clamp_coordinate(coord: float, max_bound: float = 100.0) -> float:
        try:
            val = float(coord)
        except (ValueError, TypeError):
            return 0.0
        return max(0.0, min(max_bound, val))

    @classmethod
    def parse_players(
        cls,
        raw_players: List[Dict[str, Any]],
        frame_w: int = 1920,
        frame_h: int = 1080,
    ) -> List[TelestrationPlayer]:
        players: List[TelestrationPlayer] = []
        for p in raw_players:
            raw_x = p.get("x", 0.0)
            raw_y = p.get("y", 0.0)

            try:
                x_val = float(raw_x)
                y_val = float(raw_y)
            except (ValueError, TypeError):
                x_val, y_val = 0.0, 0.0

            is_pct = p.get("is_percentage", False) or (max(abs(x_val), abs(y_val)) <= 150.0)
            norm_x = cls.clamp_coordinate(x_val, 100.0 if is_pct else float(frame_w))
            norm_y = cls.clamp_coordinate(y_val, 100.0 if is_pct else float(frame_h))

            name = str(p.get("name", "")).strip() if p.get("name") is not None else ""
            num = p.get("number")
            raw_role = str(p.get("role", "PLAYER")).upper()
            role = ROLE_NORMALIZATION_MAP.get(raw_role, "PLAYER")

            # Suppress badge if both name and number are absent
            has_badge_info = bool(name or num is not None)
            explicit_badge = p.get("showBadge")
            show_badge = bool(explicit_badge if explicit_badge is not None else has_badge_info)
            if not has_badge_info:
                show_badge = False

            # Focal players receive vertical beam
            color = p.get("color", "#ffff00" if role == "CF" else "#00e676")
            is_carrier = (color == "#ffff00")
            show_beam = bool(p.get("showBeam", is_carrier))

            players.append(TelestrationPlayer(
                id=str(p.get("id", f"p_{len(players)}")),
                name=name,
                number=num,
                role=role,
                x=norm_x,
                y=norm_y,
                color=color,
                showRing=bool(p.get("showRing", True)),
                showBeam=show_beam,
                showBadge=show_badge,
                team=p.get("team"),
            ))
        return players

    @classmethod
    def parse_formation(
        cls,
        raw_data: Dict[str, Any],
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
    ) -> TacticalTelestrationProps:
        raw_players = raw_data.get("players", [])
        players = cls.parse_players(raw_players)

        raw_lines = raw_data.get("lines", [])
        raw_polygons = raw_data.get("polygons", [])

        # Build backline
        backline_line = next((l for l in raw_lines if not l.get("dashed", False)), None)
        backline = TacticalBacklineBuilder.build_backline(players, backline_line)

        # Build pressing lines
        pressing_lines = PressingLineBuilder.build_pressing_lines(players, raw_lines)

        lines: List[TelestrationLine] = []
        if backline:
            lines.append(backline)
        lines.extend(pressing_lines)

        # Build passing polygons
        polygons = PassingTriangleBuilder.build_polygons(players, raw_polygons)

        return TacticalTelestrationProps(
            players=players,
            lines=lines,
            polygons=polygons,
            title=title or raw_data.get("title"),
            subtitle=subtitle or raw_data.get("subtitle"),
            delay=int(raw_data.get("delay", 0)),
        )


class TacticalGeometryExtractor:
    """
    Backwards-compatible drop-in adapter matching tests/e2e/harnesses/vision_harness.py.
    Guarantees 100% compatibility with Tier 1-4 tests and Remotion schema.
    """

    @staticmethod
    def normalize_coordinate(coord: float, max_bound: float = 100.0) -> float:
        """Clamps coordinate within bounds."""
        return max(0.0, min(max_bound, float(coord)))

    @classmethod
    def process_players(
        cls,
        raw_players: List[Dict[str, Any]],
        frame_w: int = 1920,
        frame_h: int = 1080,
    ) -> List[Dict[str, Any]]:
        parsed = TacticalFormationParser.parse_players(raw_players, frame_w, frame_h)
        return [p.to_dict() for p in parsed]

    @staticmethod
    def process_lines(
        raw_lines: List[Dict[str, Any]],
        valid_player_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Filters degenerate lines (< 2 players) or unknown IDs."""
        processed = []
        for line in raw_lines:
            ids = [pid for pid in line.get("playerIds", []) if pid in valid_player_ids]
            # Must have at least 2 valid connected players
            if len(ids) < 2:
                continue

            is_dashed = bool(line.get("dashed", False))
            processed.append({
                "id": str(line.get("id", f"line_{len(processed)}")),
                "playerIds": ids,
                "color": line.get("color", "#ef4444" if is_dashed else "#00e676"),
                "width": line.get("width", 3 if is_dashed else 4),
                "dashed": is_dashed,
            })
        return processed

    @staticmethod
    def process_polygons(
        raw_polygons: List[Dict[str, Any]],
        valid_player_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Filters degenerate polygons (< 3 players) and applies broadcast stripe pattern."""
        processed = []
        for poly in raw_polygons:
            ids = [pid for pid in poly.get("playerIds", []) if pid in valid_player_ids]
            # Must have at least 3 valid connected players
            if len(ids) < 3:
                continue

            processed.append({
                "id": str(poly.get("id", f"poly_{len(processed)}")),
                "playerIds": ids,
                "fillColor": poly.get("fillColor", "rgba(0, 230, 118, 0.25)"),
                "borderColor": poly.get("borderColor", "#00e676"),
                "pattern": poly.get("pattern", "stripes"),
                "label": poly.get("label", "PASSING TRIANGLE"),
            })
        return processed
