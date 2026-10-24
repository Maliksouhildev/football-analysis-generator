"""
src/vision/player_detector.py
OpenCV Computer Vision Pitch Player Detector & Roster-Aware Telestrator.

Detects player silhouettes directly on the green turf ROI, extracts true ground
boot contact coordinates (x, y % of frame), classifies team kits, and anchors
spotlights, passing corridors, and tactical polygons to actual detected positions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.vision.tactical_parser import (
    TacticalFormationParser,
    TacticalTelestrationProps,
    TelestrationPlayer,
    TelestrationLine,
    TelestrationPolygon,
)

logger = logging.getLogger("vision.player_detector")

# Authoritative squad rosters with shirt numbers, names, and tactical roles
TEAM_ROSTERS: Dict[str, List[Dict[str, Any]]] = {
    "as roma": [
        {"name": "M. Svilar", "number": 99, "role": "GK", "is_gk": True},
        {"name": "G. Mancini", "number": 23, "role": "CB"},
        {"name": "E. Ndicka", "number": 5, "role": "CB"},
        {"name": "M. Hermoso", "number": 22, "role": "CB"},
        {"name": "Angeliño", "number": 3, "role": "LB"},
        {"name": "Z. Çelik", "number": 19, "role": "RB"},
        {"name": "M. Koné", "number": 17, "role": "CM"},
        {"name": "B. Cristante", "number": 4, "role": "DM"},
        {"name": "L. Pellegrini", "number": 7, "role": "AM"},
        {"name": "L. Pisilli", "number": 21, "role": "CM"},
        {"name": "A. Dovbyk", "number": 11, "role": "CF"},
        {"name": "M. Soulé", "number": 18, "role": "RW"},
        {"name": "E. Shomurodov", "number": 14, "role": "CF"},
        {"name": "N. Zalewski", "number": 59, "role": "LM"},
        {"name": "T. Baldanzi", "number": 35, "role": "AM"},
    ],
    "roma": [
        {"name": "M. Svilar", "number": 99, "role": "GK", "is_gk": True},
        {"name": "G. Mancini", "number": 23, "role": "CB"},
        {"name": "E. Ndicka", "number": 5, "role": "CB"},
        {"name": "M. Hermoso", "number": 22, "role": "CB"},
        {"name": "Angeliño", "number": 3, "role": "LB"},
        {"name": "Z. Çelik", "number": 19, "role": "RB"},
        {"name": "M. Koné", "number": 17, "role": "CM"},
        {"name": "B. Cristante", "number": 4, "role": "DM"},
        {"name": "L. Pellegrini", "number": 7, "role": "AM"},
        {"name": "L. Pisilli", "number": 21, "role": "CM"},
        {"name": "A. Dovbyk", "number": 11, "role": "CF"},
        {"name": "M. Soulé", "number": 18, "role": "RW"},
        {"name": "E. Shomurodov", "number": 14, "role": "CF"},
        {"name": "N. Zalewski", "number": 59, "role": "LM"},
        {"name": "T. Baldanzi", "number": 35, "role": "AM"},
    ],
    "real madrid": [
        {"name": "T. Courtois", "number": 1, "role": "GK", "is_gk": True},
        {"name": "D. Carvajal", "number": 2, "role": "RB"},
        {"name": "L. Vázquez", "number": 17, "role": "RB"},
        {"name": "A. Tchouaméni", "number": 14, "role": "CB"},
        {"name": "A. Rüdiger", "number": 22, "role": "CB"},
        {"name": "R. Asencio", "number": 35, "role": "CB"},
        {"name": "F. Mendy", "number": 23, "role": "LB"},
        {"name": "F. Valverde", "number": 8, "role": "CM"},
        {"name": "E. Camavinga", "number": 6, "role": "DM"},
        {"name": "J. Bellingham", "number": 5, "role": "AM"},
        {"name": "Rodrygo", "number": 11, "role": "RW"},
        {"name": "Vinícius Jr", "number": 7, "role": "LW"},
        {"name": "K. Mbappé", "number": 9, "role": "CF"},
        {"name": "Endrick", "number": 16, "role": "CF"},
        {"name": "L. Modrić", "number": 10, "role": "CM"},
        {"name": "B. Díaz", "number": 21, "role": "AM"},
        {"name": "A. Güler", "number": 15, "role": "AM"},
    ],
    "barcelona": [
        {"name": "M. ter Stegen", "number": 1, "role": "GK", "is_gk": True},
        {"name": "J. Koundé", "number": 23, "role": "RB"},
        {"name": "P. Cubarsí", "number": 2, "role": "CB"},
        {"name": "I. Martínez", "number": 5, "role": "CB"},
        {"name": "A. Balde", "number": 3, "role": "LB"},
        {"name": "M. Casadó", "number": 17, "role": "DM"},
        {"name": "Pedri", "number": 8, "role": "CM"},
        {"name": "D. Olmo", "number": 20, "role": "AM"},
        {"name": "L. Yamal", "number": 19, "role": "RW"},
        {"name": "Raphinha", "number": 11, "role": "LW"},
        {"name": "R. Lewandowski", "number": 9, "role": "CF"},
        {"name": "Gavi", "number": 6, "role": "CM"},
        {"name": "F. Torres", "number": 7, "role": "LW"},
        {"name": "Fermín", "number": 16, "role": "AM"},
        {"name": "J. Cancelo", "number": 2, "role": "RB"},
        {"name": "K. Adeyemi", "number": 14, "role": "CF"},
    ],
    "galatasaray": [
        {"name": "F. Muslera", "number": 1, "role": "GK", "is_gk": True},
        {"name": "K. Ayhan", "number": 23, "role": "RB"},
        {"name": "D. Sánchez", "number": 6, "role": "CB"},
        {"name": "A. Bardakcı", "number": 42, "role": "CB"},
        {"name": "I. Jakobs", "number": 4, "role": "LB"},
        {"name": "L. Torreira", "number": 34, "role": "DM"},
        {"name": "G. Sara", "number": 18, "role": "CM"},
        {"name": "D. Mertens", "number": 10, "role": "AM"},
        {"name": "A. Batrakov", "number": 83, "role": "AM"},
        {"name": "B. Yılmaz", "number": 53, "role": "RW"},
        {"name": "Y. Akgün", "number": 11, "role": "LW"},
        {"name": "V. Osimhen", "number": 9, "role": "CF"},
        {"name": "M. Icardi", "number": 99, "role": "CF"},
        {"name": "M. Batshuayi", "number": 44, "role": "CF"},
        {"name": "R. Sallai", "number": 7, "role": "RW"},
    ],
}


@dataclass
class DetectedPitchPlayer:
    boot_x: float  # Normalized 0..100
    boot_y: float  # Normalized 0..100
    w_px: int
    h_px: int
    area_px: float
    team_hint: str  # "home", "away", "gk_or_ref"


def get_pitch_convex_hull(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Thresholds HSV grass green, applies cv2.MORPH_CLOSE (15x15 kernel),
    finds the largest contour, and returns cv2.convexHull(contour).
    Returns None if no pitch green contour is detected.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_grass = cv2.inRange(hsv, np.array([28, 30, 30]), np.array([88, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(mask_grass, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    return cv2.convexHull(largest_contour)


class CVPlayerDetector:
    """
    OpenCV-based player detection engine on broadcast turf.
    """

    @staticmethod
    def detect_players_on_pitch(
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> List[DetectedPitchPlayer]:
        # Load image into BGR NumPy array
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        elif isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            img = image
        else:
            return []

        if img is None or img.size == 0:
            return []

        h, w, _ = img.shape

        # 1. Detect turf region via HSV green range
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_grass = cv2.inRange(hsv, np.array([28, 30, 30]), np.array([88, 255, 255]))

        # Morphological clean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        pitch_mask = cv2.morphologyEx(mask_grass, cv2.MORPH_CLOSE, kernel)

        # 2. Extract OpenCV Convex Hull Pitch Mask (replaces static ROI clamps)
        hull = get_pitch_convex_hull(img)
        hull_mask = np.zeros((h, w), dtype=np.uint8)
        if hull is not None:
            cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)
            valid_pitch = cv2.bitwise_and(pitch_mask, hull_mask)
        else:
            valid_pitch = pitch_mask

        # 3. Non-grass foreground objects on the pitch (players, lines, ball)
        fg_mask = cv2.bitwise_and(cv2.bitwise_not(mask_grass), valid_pitch)
        fg_clean = cv2.morphologyEx(
            fg_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        )

        contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[DetectedPitchPlayer] = []

        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            aspect = float(ch) / max(1, cw)
            area = cv2.contourArea(c)

            by_px, bx_px = int(y + ch), int(x + cw / 2.0)

            # Discard any detection whose boot point is outside the pitch convex hull
            if hull is not None and cv2.pointPolygonTest(hull, (float(bx_px), float(by_px)), False) < 0:
                continue

            # Player geometry: aspect ratio [1.05..5.0], height [16..160] px, area [25..6500] px
            if 1.05 <= aspect <= 5.0 and 16 <= ch <= 160 and 25 <= area <= 6500:
                boot_x = (x + cw / 2.0) / w * 100.0
                boot_y = (y + ch) / h * 100.0

                # Strict pitch boundary: reject touchline photographers, dugout benches, crowd
                if boot_y > 81.5 or boot_y < 34.0:
                    continue

                # Reject any candidate not standing in verified green turf
                below = mask_grass[by_px:min(h, by_px + 8), max(0, bx_px - 8):min(w, bx_px + 8)]
                if below.size > 0 and (below.mean() / 255.0) < 0.35:
                    continue

                # Analyze upper shirt region (y + 15%..55% of height) for kit classification
                shirt = img[y + int(ch * 0.15):y + int(ch * 0.55), x:x + cw]
                team_hint = "away"
                if shirt.size > 0:
                    mean_bgr = np.mean(shirt, axis=(0, 1))
                    b, g, r = mean_bgr

                    # 1. Referee detection (Cyan/Sky blue, neon yellow/pink, or all-black)
                    is_cyan_ref = (b > 65 and g > 65 and r < b - 5 and r < g - 5) or (b > 85 and g > 85 and r < b - 10 and r < g - 10)
                    is_neon_yellow_ref = (r > 160 and g > 160 and b < 100)
                    is_black_ref = (r < 55 and g < 55 and b < 55)

                    # 2. Goalkeeper neon kit
                    is_gk_neon = (g > 175 and r < 140 and b < 140) or (r > 185 and g > 120 and b < 60)

                    # 3. Outfield kits
                    is_white = (
                        ((r > 135 and g > 135 and b > 135) or (r + g + b) / 3 > 135)
                        and abs(r - b) < 30
                        and abs(g - b) < 30
                    )
                    is_dark_red = (r > b + 15 and r > g + 10)

                    if is_cyan_ref or is_neon_yellow_ref or is_black_ref:
                        team_hint = "referee"
                    elif is_gk_neon:
                        team_hint = "goalkeeper"
                    elif is_dark_red:
                        team_hint = "home"
                    elif is_white:
                        team_hint = "away"
                    else:
                        team_hint = "home" if r > b else "away"

                candidates.append(
                    DetectedPitchPlayer(
                        boot_x=round(boot_x, 1),
                        boot_y=round(boot_y, 1),
                        w_px=cw,
                        h_px=ch,
                        area_px=float(area),
                        team_hint=team_hint,
                    )
                )

        # Reject horizontal text rows (e.g. broadcast graphic banners/tickers)
        # Any cluster of 3+ objects with nearly identical y (> 75% height) is graphic text
        y_buckets: Dict[int, List[DetectedPitchPlayer]] = {}
        for cand in candidates:
            y_int = int(round(cand.boot_y))
            y_buckets.setdefault(y_int, []).append(cand)

        filtered_candidates = []
        for cand in candidates:
            y_int = int(round(cand.boot_y))
            if len(y_buckets.get(y_int, [])) >= 3 and cand.boot_y > 75.0:
                continue
            filtered_candidates.append(cand)
        candidates = filtered_candidates

        # Sort candidates horizontally from left to right
        candidates.sort(key=lambda p: p.boot_x)
        return candidates

    @classmethod
    def generate_player_cutout(
        cls,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Optional[str]:
        """
        Generates an Augmented Reality transparent PNG containing solely the detected players'
        silhouettes and boot contacts, allowing Remotion to render turf cones/rings/lines
        BEHIND the players while keeping players standing in 3D in the foreground.
        """
        img_p = Path(image_path)
        if not img_p.is_file():
            return None

        img = cv2.imread(str(img_p))
        if img is None or img.size == 0:
            return None

        h, w, _ = img.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_grass = cv2.inRange(hsv, np.array([28, 30, 30]), np.array([88, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        pitch_mask = cv2.morphologyEx(mask_grass, cv2.MORPH_CLOSE, kernel)

        hull = get_pitch_convex_hull(img)
        hull_mask = np.zeros((h, w), dtype=np.uint8)
        if hull is not None:
            cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)
            valid_pitch = cv2.bitwise_and(pitch_mask, hull_mask)
        else:
            valid_pitch = pitch_mask

        fg_mask = cv2.bitwise_and(cv2.bitwise_not(mask_grass), valid_pitch)
        fg_clean = cv2.morphologyEx(
            fg_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        )

        contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        alpha = np.zeros((h, w), dtype=np.uint8)

        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            aspect = float(ch) / max(1, cw)
            area = cv2.contourArea(c)
            bx_px = int(x + cw / 2.0)
            by_px = int(y + ch)
            if hull is not None and cv2.pointPolygonTest(hull, (float(bx_px), float(by_px)), False) < 0:
                continue
            if 1.05 <= aspect <= 5.0 and 16 <= ch <= 160 and 25 <= area <= 6500:
                y1 = max(0, y - 4)
                y2 = min(h, y + ch + 3)
                x1 = max(0, x - 4)
                x2 = min(w, x + cw + 4)
                box_grass = mask_grass[y1:y2, x1:x2]
                box_player = cv2.bitwise_not(box_grass)
                box_player_filled = cv2.morphologyEx(
                    box_player,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                )
                box_smooth = cv2.GaussianBlur(box_player_filled, (3, 3), 0)
                alpha[y1:y2, x1:x2] = np.maximum(alpha[y1:y2, x1:x2], box_smooth)

        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), bgra)
        return str(out_p)

    @classmethod
    def generate_roster_telestration(
        cls,
        image: Union[str, Path, np.ndarray, Image.Image],
        event_ctx: Dict[str, Any],
    ) -> TacticalTelestrationProps:
        """
        Builds a high-precision broadcast telestration overlay using verified boot coordinates,
        active XI match lineups with substitution awareness, and strict referee kit exclusion.
        Suppresses telestration entirely on close-ups or unpopulated scenes.
        """
        from src.vision.shot_classifier import CameraShotClassifier, ShotType
        from src.scraper.lineup_scraper import MatchLineupResolver

        # 1. Close-up Suppression (Complete overlay blackout on close-ups/tackle replays)
        if image is not None:
            shot_type, _ = CameraShotClassifier.classify(image)
            if shot_type == ShotType.CLOSE_UP:
                logger.info("Shot classified as CLOSE_UP. Suppressing all pitch telestration overlays.")
                return TacticalFormationParser.parse_formation({
                    "players": [],
                    "lines": [],
                    "polygons": [],
                    "title": None,
                    "subtitle": None,
                })

        minute = int(event_ctx.get("match_minute", 0))
        home_team_name = str(event_ctx.get("home_team", "AS Roma")).lower()
        away_team_name = str(event_ctx.get("away_team", "Real Madrid")).lower()
        acting_team = str(event_ctx.get("team", home_team_name)).lower()
        focal_player_name = str(event_ctx.get("players_involved", [""])[0] if event_ctx.get("players_involved") else "").strip()
        ev_type = str(event_ctx.get("event_type", "transition")).lower()

        # 2. Resolve Active XI and substitutions for exact minute
        lineup_info = MatchLineupResolver.get_active_lineup(home_team_name, away_team_name, match_minute=minute)
        home_outfield = lineup_info["home_outfield"]
        away_outfield = lineup_info["away_outfield"]

        # 3. Detect true players on turf
        detected = cls.detect_players_on_pitch(image)

        # 4. Strict Exclusion: Remove referees and goalkeepers from candidate pool
        outfield_detected = [p for p in detected if p.team_hint not in ("referee", "goalkeeper")]

        # If fewer than 2 verified outfield players detected: ZERO PHANTOM TAGS
        if len(outfield_detected) < 2:
            logger.info("Fewer than 2 outfield players confirmed on pitch. Zero phantom tags emitted.")
            return TacticalFormationParser.parse_formation({
                "players": [],
                "lines": [],
                "polygons": [],
                "title": None,
                "subtitle": None,
            })

        # 5. Determine which squad is acting
        is_home_acting = None
        focal_entry = None

        if focal_player_name:
            for p in home_outfield:
                if any(part.lower() in p["name"].lower() for part in focal_player_name.split() if len(part) > 2):
                    focal_entry = p
                    is_home_acting = True
                    break
            if not focal_entry:
                for p in away_outfield:
                    if any(part.lower() in p["name"].lower() for part in focal_player_name.split() if len(part) > 2):
                        focal_entry = p
                        is_home_acting = False
                        break

        if is_home_acting is None:
            if home_team_name in acting_team or any(k in acting_team for k in home_team_name.split() if len(k) > 3):
                is_home_acting = True
            elif away_team_name in acting_team or any(k in acting_team for k in away_team_name.split() if len(k) > 3):
                is_home_acting = False
            else:
                is_home_acting = True

        active_roster = home_outfield if is_home_acting else away_outfield
        defending_roster = away_outfield if is_home_acting else home_outfield
        acting_hint = "home" if is_home_acting else "away"
        defending_hint = "away" if is_home_acting else "home"

        if not focal_entry:
            focal_entry = next((p for p in active_roster if p["role"] in ("CF", "RW", "LW", "AM")), active_roster[0])

        # 6. Locate the Active Contested Zone (Action Cluster)
        # Find opposing players in close proximity (< 28% frame distance)
        acting_candidates = [p for p in outfield_detected if p.team_hint == acting_hint]
        defending_candidates = [p for p in outfield_detected if p.team_hint == defending_hint]

        if not acting_candidates:
            acting_candidates = outfield_detected[:1]
        if not defending_candidates:
            defending_candidates = [p for p in outfield_detected if p not in acting_candidates]

        # Score candidates by proximity to opposing players (contested action)
        def min_opponent_dist(p: DetectedPitchPlayer, opps: List[DetectedPitchPlayer]) -> float:
            if not opps:
                return 100.0
            return min(((p.boot_x - o.boot_x)**2 + (p.boot_y - o.boot_y)**2)**0.5 for o in opps)

        acting_candidates.sort(key=lambda p: min_opponent_dist(p, defending_candidates))
        target_focal_player = acting_candidates[0]

        # Supporting teammates strictly within 32% distance of focal player
        teammate_detected = [
            p for p in acting_candidates[1:]
            if ((p.boot_x - target_focal_player.boot_x)**2 + (p.boot_y - target_focal_player.boot_y)**2)**0.5 < 32.0
        ][:2]

        # Defending opponents strictly within 28% distance of focal player
        defender_detected = [
            p for p in defending_candidates
            if ((p.boot_x - target_focal_player.boot_x)**2 + (p.boot_y - target_focal_player.boot_y)**2)**0.5 < 28.0
        ][:2]

        # Action cluster limit: at most 4 players total (1 focal + up to 3 context players)
        if len(teammate_detected) + len(defender_detected) > 3:
            if len(teammate_detected) >= 2:
                defender_detected = defender_detected[:1]
            else:
                teammate_detected = teammate_detected[:1]

        players_list: List[Dict[str, Any]] = []

        # 1. Focal Player (Yellow Spotlight Ring + Vertical Beam + Badge)
        focal_id = "p_focal"
        players_list.append({
            "id": focal_id,
            "name": focal_entry["name"],
            "number": focal_entry["number"],
            "role": focal_entry["role"],
            "x": target_focal_player.boot_x,
            "y": target_focal_player.boot_y,
            "color": "#ffff00",
            "showRing": True,
            "showBeam": True,
            "showBadge": True,
        })

        # 2. Supporting Teammates (Emerald Spotlight Rings)
        teammate_pool = [p for p in active_roster if p["name"] != focal_entry["name"]]
        support_ids: List[str] = []

        for idx, d_p in enumerate(teammate_detected):
            tm = teammate_pool[idx % len(teammate_pool)]
            p_id = f"p_support_{idx}"
            support_ids.append(p_id)
            players_list.append({
                "id": p_id,
                "name": tm["name"],
                "number": tm["number"],
                "role": tm["role"],
                "x": d_p.boot_x,
                "y": d_p.boot_y,
                "color": "#00e676",
                "showRing": True,
                "showBeam": False,
                "showBadge": True,
            })

        # 3. Opposition Defenders / Pressers (Red Spotlight Rings)
        defender_pool = [p for p in defending_roster if p["role"] in ("CB", "LB", "RB", "DM")] or defending_roster
        defending_ids: List[str] = []

        for idx, d_p in enumerate(defender_detected):
            df = defender_pool[idx % len(defender_pool)]
            p_id = f"p_def_{idx}"
            defending_ids.append(p_id)
            players_list.append({
                "id": p_id,
                "name": df["name"],
                "number": df["number"],
                "role": df["role"],
                "x": d_p.boot_x,
                "y": d_p.boot_y,
                "color": "#ef4444",
                "showRing": True,
                "showBeam": False,
                "showBadge": True,
            })

        # Lines and Polygons
        lines_list: List[Dict[str, Any]] = []
        polygons_list: List[Dict[str, Any]] = []

        # Passing corridors connecting focal player and teammates (clean open lines, no closed triangles)
        if len(support_ids) >= 2:
            lines_list.append({
                "id": "passing_line_1",
                "playerIds": [focal_id, support_ids[0]],
                "color": "#ffff00",
                "width": 3,
                "dashed": False,
            })
            lines_list.append({
                "id": "passing_line_2",
                "playerIds": [focal_id, support_ids[1]],
                "color": "#00e676",
                "width": 3,
                "dashed": True,
            })
        elif len(support_ids) == 1:
            lines_list.append({
                "id": "passing_corridor",
                "playerIds": [focal_id, support_ids[0]],
                "color": "#ffff00",
                "width": 3,
                "dashed": False,
            })

        # Defensive backline or pressing pressure
        if len(defending_ids) >= 2:
            lines_list.append({
                "id": "defensive_containment_line",
                "playerIds": defending_ids,
                "color": "#ef4444",
                "width": 4,
                "dashed": False,
            })
        elif len(defending_ids) == 1:
            lines_list.append({
                "id": "pressing_arrow",
                "playerIds": [defending_ids[0], focal_id],
                "color": "#ef4444",
                "width": 3,
                "dashed": True,
            })

        title_action = "GOAL SCORING ACTION" if "goal" in ev_type else "PROGRESSIVE BREAKOUT"
        return TacticalFormationParser.parse_formation({
            "players": players_list,
            "lines": lines_list,
            "polygons": polygons_list,
            "title": f"MINUTE {minute}' // {title_action}",
            "subtitle": f"{focal_entry['name'].upper()} // TACTICAL SHAPE BREAKDOWN",
        })

    @classmethod
    def _get_roster(cls, team_name: str) -> List[Dict[str, Any]]:
        clean = team_name.strip().lower()
        for k, v in TEAM_ROSTERS.items():
            if k in clean or clean in k:
                return v
        # Generic fallback roster
        return [
            {"name": "Goalkeeper", "number": 1, "role": "GK", "is_gk": True},
            {"name": "Right Back", "number": 2, "role": "RB"},
            {"name": "Centre Back 1", "number": 4, "role": "CB"},
            {"name": "Centre Back 2", "number": 5, "role": "CB"},
            {"name": "Left Back", "number": 3, "role": "LB"},
            {"name": "Defensive Mid", "number": 6, "role": "DM"},
            {"name": "Central Mid", "number": 8, "role": "CM"},
            {"name": "Playmaker", "number": 10, "role": "AM"},
            {"name": "Right Winger", "number": 7, "role": "RW"},
            {"name": "Striker", "number": 9, "role": "CF"},
            {"name": "Left Winger", "number": 11, "role": "LW"},
        ]
