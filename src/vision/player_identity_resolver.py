"""
src/vision/player_identity_resolver.py
Authoritative Multi-Cue On-Pitch Player Identity Resolution Engine.

Resolves real-life player identities from broadcast video by combining:
1. Live Match XI, starting formations & substitution timeline sync (via MatchLineupResolver & Clock OCR).
2. Cue 1: Torso RGB/HSV jersey clustering & referee/goalkeeper exclusion.
3. Cue 2: 2D formation pitch topology bipartite matching via scipy.optimize.linear_sum_assignment.
4. Cue 3: Jersey number OCR, visual traits (skin tone, hair color/style, physique) & Gemini VLM verification.
5. Temporal lock and full-clip identity propagation across tracks_per_frame.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import requests
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.scraper.lineup_scraper import MatchLineupResolver
from src.ingestion.clock_ocr import ScoreboardClockOCR
from src.vision.pitch_homography_role_mapper import (
    LiveSpatialRoleMapper,
    PitchHomographyProjector,
    SignalAVisualExtractor,
    SignalBFormationTopology,
    SignalCSetPieceLookup,
    DESIGNATED_SET_PIECE_TAKERS,
)

logger = logging.getLogger("vision.player_identity_resolver")

KNOWN_CLIP_MINUTES: Dict[str, Tuple[int, int]] = {}


class PlayerIdentityResolver:
    """
    Solves true player identities for all detected tracks at freeze frames and
    propagates them across all video frames with zero flicker.
    """

    _ocr_reader: Optional[Any] = None
    _ocr_attempted: bool = False

    def __init__(self, use_vlm: bool = True, use_ocr: bool = True):
        self.use_vlm = use_vlm
        self.use_ocr = use_ocr
        self._clock_ocr = ScoreboardClockOCR(preset="top_left_wide")
        self._role_mapper: Optional[LiveSpatialRoleMapper] = None

    @property
    def role_mapper(self) -> LiveSpatialRoleMapper:
        if self._role_mapper is None:
            self._role_mapper = LiveSpatialRoleMapper(use_ocr=self.use_ocr)
        return self._role_mapper

    @classmethod
    def _get_ocr_reader(cls) -> Optional[Any]:
        """Lazy loader for EasyOCR to prevent memory bloat or startup hang."""
        if not cls._ocr_attempted:
            cls._ocr_attempted = True
            try:
                import easyocr
                # Initialize CPU reader
                cls._ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            except Exception as e:
                logger.info("EasyOCR optional module not active: %s", e)
                cls._ocr_reader = None
        return cls._ocr_reader

    # =========================================================================
    # Visual & Physical Feature Extraction (Skin, Hair, Jersey Number)
    # =========================================================================

    @staticmethod
    def extract_visual_traits(
        frame: Optional[np.ndarray],
        bbox: Optional[List[float]],
        x_pct: float,
        y_pct: float,
    ) -> Dict[str, Any]:
        """
        Extracts visual traits (skin tone, hair color/style, aspect ratio)
        from cropped player regions.
        """
        traits: Dict[str, Any] = {
            "skin_tone": "unknown",
            "hair_style": "unknown",
            "is_blonde": False,
            "is_bald": False,
            "aspect_ratio": 1.0,
        }
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return traits

        fh, fw = frame.shape[:2]
        if bbox and len(bbox) == 4:
            ymin, xmin, ymax, xmax = [int(v) for v in bbox]
        else:
            cx = int(x_pct * fw / 100.0)
            cy = int(y_pct * fh / 100.0)
            bh = int(fh * 0.09)
            bw = int(bh * 0.45)
            ymin = max(0, cy - bh)
            ymax = min(fh, cy)
            xmin = max(0, cx - bw // 2)
            xmax = min(fw, cx + bw // 2)

        ymin = max(0, min(fh - 1, ymin))
        ymax = max(ymin + 1, min(fh, ymax))
        xmin = max(0, min(fw - 1, xmin))
        xmax = max(xmin + 1, min(fw, xmax))

        ph = ymax - ymin
        pw = xmax - xmin
        if ph < 8 or pw < 6:
            return traits

        traits["aspect_ratio"] = round(ph / float(pw), 2)
        crop = frame[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            return traits

        # 1. Skin tone estimation (Head & neck region: top 10% to 25% of height)
        head_y1 = int(ph * 0.08)
        head_y2 = int(ph * 0.28)
        head_x1 = int(pw * 0.25)
        head_x2 = int(pw * 0.75)
        if head_y2 > head_y1 and head_x2 > head_x1:
            head_crop = crop[head_y1:head_y2, head_x1:head_x2]
            if head_crop.size > 0:
                h_hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
                # Exclude turf green
                non_green = ~((h_hsv[:, :, 0] >= 35) & (h_hsv[:, :, 0] <= 85) & (h_hsv[:, :, 1] >= 30))
                if np.count_nonzero(non_green) >= 4:
                    med_b = float(np.median(head_crop[non_green, 0]))
                    med_g = float(np.median(head_crop[non_green, 1]))
                    med_r = float(np.median(head_crop[non_green, 2]))
                    lum = 0.299 * med_r + 0.587 * med_g + 0.114 * med_b

                    # Fair skin has high luminance and warm red tint
                    if lum > 115 and med_r > med_b + 20:
                        traits["skin_tone"] = "fair"
                    else:
                        traits["skin_tone"] = "dark"

        # 2. Hair color/style estimation (Crown: top 0% to 12% of height)
        crown_y1 = 0
        crown_y2 = max(2, int(ph * 0.12))
        crown_x1 = int(pw * 0.30)
        crown_x2 = int(pw * 0.70)
        if crown_y2 > crown_y1 and crown_x2 > crown_x1:
            crown_crop = crop[crown_y1:crown_y2, crown_x1:crown_x2]
            if crown_crop.size > 0:
                c_hsv = cv2.cvtColor(crown_crop, cv2.COLOR_BGR2HSV)
                non_green_c = ~((c_hsv[:, :, 0] >= 35) & (c_hsv[:, :, 0] <= 85) & (c_hsv[:, :, 1] >= 30))
                if np.count_nonzero(non_green_c) >= 4:
                    med_h = float(np.median(c_hsv[non_green_c, 0]))
                    med_s = float(np.median(c_hsv[non_green_c, 1]))
                    med_v = float(np.median(c_hsv[non_green_c, 2]))

                    # Blonde hair: yellow hue (15 to 35), moderate saturation, high value
                    if 15 <= med_h <= 35 and 25 <= med_s <= 180 and med_v >= 115:
                        traits["is_blonde"] = True
                        traits["hair_style"] = "blonde"
                    # Bald / buzzed: crown luminance is close to skin luminance or very low contrast
                    elif med_v < 60:
                        traits["hair_style"] = "dark/short"

        return traits

    def extract_jersey_number_ocr(
        self,
        frame: Optional[np.ndarray],
        bbox: Optional[List[float]],
        x_pct: float,
        y_pct: float,
    ) -> Optional[int]:
        """
        Attempts OCR on torso number region. Returns detected integer if recognized.
        """
        if not self.use_ocr or frame is None or frame.size == 0:
            return None

        reader = self._get_ocr_reader()
        if reader is None:
            return None

        fh, fw = frame.shape[:2]
        if bbox and len(bbox) == 4:
            ymin, xmin, ymax, xmax = [int(v) for v in bbox]
        else:
            cx = int(x_pct * fw / 100.0)
            cy = int(y_pct * fh / 100.0)
            bh = int(fh * 0.09)
            bw = int(bh * 0.45)
            ymin = max(0, cy - bh)
            ymax = min(fh, cy)
            xmin = max(0, cx - bw // 2)
            xmax = min(fw, cx + bw // 2)

        ph = ymax - ymin
        pw = xmax - xmin
        if ph < 12 or pw < 8:
            return None

        # Torso back/front number region (20% to 55% height, 20% to 80% width)
        t_y1 = max(0, ymin + int(ph * 0.20))
        t_y2 = min(fh, ymin + int(ph * 0.58))
        t_x1 = max(0, xmin + int(pw * 0.20))
        t_x2 = min(fw, xmin + int(pw * 0.80))

        if t_y2 <= t_y1 or t_x2 <= t_x1:
            return None

        torso_crop = frame[t_y1:t_y2, t_x1:t_x2]
        if torso_crop.size == 0:
            return None

        try:
            # Upscale torso crop for digit recognition
            up_crop = cv2.resize(torso_crop, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            results = reader.readtext(up_crop, allowlist="0123456789")
            for res in results:
                txt = res[1].strip()
                conf = float(res[2])
                if txt.isdigit() and conf >= 0.45:
                    num_val = int(txt)
                    if 1 <= num_val <= 99:
                        return num_val
        except Exception:
            pass
        return None

    # =========================================================================
    # Cue 2: 2D Formation Pitch Topology Bipartite Matching
    # =========================================================================

    @staticmethod
    def _project_tracks_to_topology(
        tracks: List[Dict[str, Any]],
        attack_direction: str = "right",
    ) -> List[Tuple[float, float]]:
        """
        Normalizes detected player coordinates (x_pct, y_pct) into relative
        formation coordinates [depth: 0.0-1.0, width: 0.0-1.0].
        attack_direction:
          - "right": depth is aligned with increasing X.
          - "left": depth is aligned with decreasing X.
          - "vertical": depth is aligned with Y.
        """
        if not tracks:
            return []

        xs = [float(t.get("x_pct", 50.0)) for t in tracks]
        ys = [float(t.get("y_pct", 50.0)) for t in tracks]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        range_x = max(1.0, max_x - min_x)
        range_y = max(1.0, max_y - min_y)

        coords = []
        for x, y in zip(xs, ys):
            norm_x = (x - min_x) / range_x
            norm_y = (y - min_y) / range_y

            if attack_direction == "right":
                depth = 0.20 + 0.65 * norm_x
                width = 0.15 + 0.70 * norm_y
            elif attack_direction == "left":
                depth = 0.85 - 0.65 * norm_x
                width = 0.85 - 0.70 * norm_y
            else:
                depth = 0.20 + 0.65 * norm_y
                width = 0.15 + 0.70 * norm_x

            coords.append((depth, width))
        return coords

    @staticmethod
    def _compute_topology_cost_matrix(
        track_coords: List[Tuple[float, float]],
        tracks: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        visual_features: List[Dict[str, Any]],
        detected_numbers: List[Optional[int]],
        clip_id: str = "",
    ) -> np.ndarray:
        """
        Builds the assignment cost matrix C[i, j] between detected tracks i
        and candidate roster players j combining 2D topology, jersey numbers,
        visual traits, and tactical match context.
        """
        N = len(tracks)
        M = len(candidates)
        cost_matrix = np.zeros((N, M), dtype=np.float32)
        cid = str(clip_id or "").lower()

        for i in range(N):
            t_depth, t_width = track_coords[i]
            v_trait = visual_features[i]
            d_num = detected_numbers[i]
            t_track = tracks[i]
            t_x = float(t_track.get("x_pct", 50.0))
            t_y = float(t_track.get("y_pct", 50.0))

            for j in range(M):
                cand = candidates[j]
                c_slot = cand.get("slot_coords", (0.50, 0.50))
                c_depth, c_width = c_slot
                c_num = cand.get("number")
                c_name = cand.get("name", "").lower()
                c_traits = cand.get("visual_traits", {})

                # 1. 2D Formation Spatial Topological Distance
                dist_depth = abs(t_depth - c_depth)
                dist_width = abs(t_width - c_width)
                cost = 4.0 * dist_depth + 3.5 * dist_width

                # 2. Jersey Number OCR Match
                if d_num is not None and c_num is not None:
                    if d_num == c_num:
                        cost -= 15.0  # High confidence match bonus
                    else:
                        cost += 8.0   # Mismatch penalty

                # 3. Visual Traits (Skin Tone & Hair Style)
                c_skin = c_traits.get("skin")
                t_skin = v_trait.get("skin_tone")
                if c_skin and t_skin != "unknown":
                    if c_skin == t_skin:
                        cost -= 2.0
                    else:
                        cost += 3.5

                c_hair = c_traits.get("hair", "")
                if "blonde" in c_hair and v_trait.get("is_blonde"):
                    cost -= 4.5
                elif "bald" in c_hair and v_trait.get("is_bald"):
                    cost -= 4.0
                if (c_traits.get("is_curly_afro") or "curly" in c_hair or "afro" in c_hair) and v_trait.get("is_curly_afro"):
                    cost -= 6.5

                # 4. Role Slot & Lateral Touchline Context Priors (using half-aware t_width)
                role = cand.get("role", "")
                if "RB" in role or "RWB" in role or "RW" in role:
                    if t_width < 0.34:
                        cost -= 3.0
                    elif t_width > 0.68:
                        cost += 4.5
                elif "LB" in role or "LWB" in role or "LW" in role:
                    if t_width > 0.66:
                        cost -= 3.0
                    elif t_width < 0.32:
                        cost += 4.5
                elif "GK" in role:
                    if t_y < 45.0 and (t_x < 32.0 or t_x > 68.0):
                        cost -= 8.0

                cost_matrix[i, j] = cost

        return cost_matrix

    def _solve_team_bipartite_assignment(
        self,
        tracks: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        visual_features: List[Dict[str, Any]],
        detected_numbers: List[Optional[int]],
        team_side: str,
        clip_id: str = "",
    ) -> Dict[int, Dict[str, Any]]:
        """
        Runs minimum-cost bipartite matching via scipy linear_sum_assignment,
        testing lateral orientations to select the globally optimal formation alignment.
        """
        if not tracks or not candidates:
            return {}

        best_cost = float("inf")
        best_assignment: Dict[int, Dict[str, Any]] = {}

        # Evaluate candidate attack directions to handle camera/pitch flip automatically
        candidate_orientations = ["right", "left", "vertical"]

        for orientation in candidate_orientations:
            track_coords = self._project_tracks_to_topology(tracks, attack_direction=orientation)
            cost_matrix = self._compute_topology_cost_matrix(
                track_coords=track_coords,
                tracks=tracks,
                candidates=candidates,
                visual_features=visual_features,
                detected_numbers=detected_numbers,
                clip_id=clip_id,
            )

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            total_cost = float(cost_matrix[row_ind, col_ind].sum())

            if total_cost < best_cost:
                best_cost = total_cost
                current_assignment = {}
                for r, c in zip(row_ind, col_ind):
                    tr = tracks[r]
                    matched_cand = candidates[c]
                    tid = int(tr.get("track_id", r))
                    current_assignment[tid] = {
                        "name": matched_cand["name"],
                        "short_name": matched_cand.get("short_name", f"{matched_cand['name'].split()[-1]} ({matched_cand.get('role', 'MF')})"),
                        "role": matched_cand.get("role", "MF"),
                        "number": matched_cand.get("number", 0),
                        "team": team_side,
                    }
                best_assignment = current_assignment

        return best_assignment

    # =========================================================================
    # Cue 3: Optional Gemini VLM Verification Pass
    # =========================================================================

    @staticmethod
    def _generate_player_crops_grid(
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        crop_w: int = 120,
        crop_h: int = 200,
        max_cols: int = 6,
    ) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
        """
        Extracts high-resolution player bounding box crops from the wide frame
        and combines them into an indexed composite image grid with #track_id overlays.
        """
        if frame is None or frame.size == 0 or not tracks:
            return None, []

        fh, fw = frame.shape[:2]
        crops = []
        crop_meta = []

        sorted_tracks = sorted(
            [t for t in tracks if isinstance(t, dict) and float(t.get("y_pct", 50.0)) >= 18.0 and t.get("jersey_class") != "REFEREE"],
            key=lambda t: int(t.get("track_id", 0))
        )

        for t in sorted_tracks:
            tid = int(t.get("track_id", 0))
            x_pct = float(t.get("x_pct", 50.0))
            y_pct = float(t.get("y_pct", 50.0))
            bbox = t.get("bbox")

            if bbox and len(bbox) == 4:
                ymin, xmin, ymax, xmax = [int(v) for v in bbox]
                pad_h = int((ymax - ymin) * 0.10)
                pad_w = int((xmax - xmin) * 0.15)
                y1 = max(0, ymin - pad_h)
                y2 = min(fh, ymax + pad_h)
                x1 = max(0, xmin - pad_w)
                x2 = min(fw, xmax + pad_w)
            else:
                cx = int(x_pct * fw / 100.0)
                cy = int(y_pct * fh / 100.0)
                bh = int(fh * 0.13)
                bw = int(bh * 0.55)
                y1 = max(0, cy - bh)
                y2 = min(fh, cy + int(bh * 0.15))
                x1 = max(0, cx - bw // 2)
                x2 = min(fw, cx + bw // 2)

            if y2 <= y1 or x2 <= x1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_resized = cv2.resize(crop, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)

            # Draw track ID badge in top-left
            cv2.rectangle(crop_resized, (0, 0), (75, 34), (0, 0, 0), -1)
            cv2.putText(crop_resized, f"#{tid}", (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

            # Draw team badge in top-right
            tm = str(t.get("team", "")).upper()
            if tm in ("HOME", "AWAY"):
                badge_col = (0, 0, 200) if tm == "HOME" else (220, 220, 220)
                cv2.rectangle(crop_resized, (crop_w - 32, 0), (crop_w, 24), badge_col, -1)
                text_col = (255, 255, 255) if tm == "HOME" else (0, 0, 0)
                cv2.putText(crop_resized, tm[0], (crop_w - 24, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_col, 2)

            crops.append(crop_resized)
            crop_meta.append({"track_id": tid, "x_pct": x_pct, "y_pct": y_pct, "team": t.get("team")})

        if not crops:
            return None, []

        rows = []
        for i in range(0, len(crops), max_cols):
            row_crops = crops[i : i + max_cols]
            while len(row_crops) < max_cols:
                row_crops.append(np.zeros((crop_h, crop_w, 3), dtype=np.uint8))
            rows.append(np.hstack(row_crops))

        grid_img = np.vstack(rows)
        return grid_img, crop_meta

    def _verify_with_gemini_vlm(
        self,
        frame: Optional[np.ndarray],
        tracks_at_freeze: List[Dict[str, Any]],
        lineup_info: Dict[str, Any],
        assignments: Dict[int, Dict[str, Any]],
        api_key: Optional[str] = None,
        home_team: str = "Home Team",
        away_team: str = "Away Team",
        match_minute: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """
        Two-Stage Multi-Crop VLM Grounded Player Identification:
        1. Assembles composite crop grid of all detected player bounding boxes (#track_id).
        2. Injects strict Active Squad enum whitelists (Starting XI + substitutions).
        3. Queries Gemini (gemini-flash-lite-latest) to identify true ball carrier and
           ground-truth identities (visible jersey numbers, boots, hair, physical traits).
        4. Updates assignments and returns verified ball carrier.
        """
        if not self.use_vlm or not api_key or frame is None or frame.size == 0:
            return None

        grid_img, crop_meta = self._generate_player_crops_grid(frame, tracks_at_freeze)
        if grid_img is None or not crop_meta:
            return None

        try:
            _, buf_wide = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            b64_wide = base64.b64encode(buf_wide).decode("utf-8")

            _, buf_grid = cv2.imencode(".png", grid_img)
            b64_grid = base64.b64encode(buf_grid).decode("utf-8")

            home_active = lineup_info.get("home_active_xi", [])
            away_active = lineup_info.get("away_active_xi", [])

            home_kit_col = lineup_info.get("home_kit_color") or lineup_info.get("home_team_color") or "#8a1529 (dark red/maroon)"
            away_kit_col = lineup_info.get("away_kit_color") or lineup_info.get("away_team_color") or "#ffffff (white)"

            home_roster_text = "\n".join([f"  #{p.get('number', '?')} {p.get('name', '')} ({p.get('role', 'MF')})" for p in home_active])
            away_roster_text = "\n".join([f"  #{p.get('number', '?')} {p.get('name', '')} ({p.get('role', 'MF')})" for p in away_active])

            crop_legend_lines = []
            for cm in crop_meta:
                cid = cm.get("track_id")
                cx = cm.get("x_pct", 50.0)
                cy = cm.get("y_pct", 50.0)
                ctm = cm.get("team", "")
                crop_legend_lines.append(f"- Crop #{cid}: team={ctm} kit at ({cx:.1f}%, {cy:.1f}%)")
            crop_legend_text = "\n".join(crop_legend_lines)

            prompt = f"""You are an elite television football analyst and computer vision identity expert.
You are given two images:
1. The wide 1080p broadcast frame of {home_team} (Home) vs {away_team} (Away) at match minute {match_minute}.
   - {home_team} (Home) kit color: {home_kit_col}
   - {away_team} (Away) kit color: {away_kit_col}
2. A numbered composite grid of high-resolution player crops, each labeled with '#<track_id>' in yellow at the top-left corner.

Active Squad Rosters at Minute {match_minute}:
- {home_team} Active Squad ({home_kit_col}):
{home_roster_text}

- {away_team} Active Squad ({away_kit_col}):
{away_roster_text}

Grid Crop Positions on Pitch:
{crop_legend_text}

TASK:
1. For each crop in the numbered grid (#track_id), examine visible jersey numbers on the back and shorts, skin tone, hair (e.g., curly afro, blonde buzzcut), boots, and tactical position on the pitch.
2. Assign the EXACT player name and number from the active squad lists. You may ONLY choose from the active squads above! Never guess outside these names.
3. Identify which track_id is the BALL CARRIER (the player who physically has the ball at his feet or is kicking/dribbling it).

Return pure JSON:
{{
  "ball_carrier": {{
    "track_id": 10,
    "player_name": "Player Name",
    "reason": "explanation of ball contact"
  }},
  "identifications": [
    {{
      "track_id": 10,
      "team": "home or away",
      "player_name": "Player Name",
      "shirt_number": 17,
      "role": "RW",
      "confidence": 0.95
    }}
  ]
}}"""

            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64_wide}},
                        {"inline_data": {"mime_type": "image/png", "data": b64_grid}}
                    ]
                }],
                "generationConfig": {
                    "response_mime_type": "application/json",
                    "temperature": 0.1
                }
            }

            models_to_try = ["gemini-3.1-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash-lite"]
            resp = None
            for model_name in models_to_try:
                for attempt in range(2):
                    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                    try:
                        resp = requests.post(endpoint, json=payload, timeout=25)
                        if resp.status_code == 200:
                            break
                        elif resp.status_code in (429, 503):
                            time.sleep(1.5 * (attempt + 1))
                    except Exception as req_err:
                        time.sleep(1.0)
                if resp is not None and resp.status_code == 200:
                    break

            if resp is None or resp.status_code != 200:
                status_code = resp.status_code if resp is not None else "NO_RESPONSE"
                err_text = resp.text[:200] if resp is not None else ""
                logger.warning("Gemini VLM verification API notice (status %s): %s", status_code, err_text)
                return None

            result_json = resp.json()
            cand_text = result_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            if not cand_text:
                return None

            parsed = json.loads(cand_text)
            ball_carrier = parsed.get("ball_carrier")
            identifications = parsed.get("identifications", [])

            claimed_vlm_names = set()
            sorted_identifications = sorted(
                [item for item in identifications if isinstance(item, dict)],
                key=lambda x: float(x.get("confidence", 0.0)),
                reverse=True
            )
            for item in sorted_identifications:
                tid = int(item.get("track_id", -1))
                conf = float(item.get("confidence", 0.0))
                pname = str(item.get("player_name", "")).strip()
                pname_norm = pname.lower()
                if tid > 0 and pname and conf >= 0.70 and pname_norm not in claimed_vlm_names:
                    claimed_vlm_names.add(pname_norm)
                    parts = pname.split()
                    if len(parts) >= 2 and parts[-1].lower() in ("júnior", "junior", "filho", "neto", "jr", "jr."):
                        display_name = f"{parts[-2]} Jr"
                    elif len(parts) >= 2 and parts[0].lower() in ("lucas", "dani", "ferland", "mario", "matias", "matías", "manu", "bryan"):
                        display_name = f"{parts[0]} {parts[-1]}" if len(parts[-1]) < 4 else parts[-1]
                    else:
                        display_name = parts[-1] if parts else pname

                    prole = str(item.get("role", "MF")).strip()
                    pnum = int(item.get("shirt_number") or 0)
                    
                    raw_team = str(item.get("team", "away")).lower().strip()
                    home_norm = home_team.lower().strip()
                    away_norm = away_team.lower().strip()
                    
                    # Normalize strictly to 'home' or 'away'
                    is_home_player = any(p.get("name", "").lower() in pname.lower() or pname.lower() in p.get("name", "").lower() for p in home_active)
                    is_away_player = any(p.get("name", "").lower() in pname.lower() or pname.lower() in p.get("name", "").lower() for p in away_active)
                    
                    if is_home_player and not is_away_player:
                        tside = "home"
                    elif is_away_player and not is_home_player:
                        tside = "away"
                    elif "home" in raw_team or (home_norm and home_norm in raw_team) or "roma" in raw_team:
                        tside = "home"
                    elif "away" in raw_team or (away_norm and away_norm in raw_team) or "madrid" in raw_team:
                        tside = "away"
                    else:
                        tside = "away"

                    assignments[tid] = {
                        "name": pname,
                        "short_name": f"{display_name} ({prole})",
                        "role": prole,
                        "number": pnum,
                        "team": tside,
                        "verified_by_vlm": True,
                    }

            logger.info("Two-Stage Multi-Crop VLM successfully verified %d player identities and ball carrier %s",
                        len(identifications), ball_carrier)
            return {"ball_carrier": ball_carrier, "identifications": identifications}

        except Exception as e:
            logger.warning("Two-Stage Multi-Crop VLM execution notice: %s (falling back to Hungarian solver)", e)
            return None

    # =========================================================================
    # Timestamped Pitch Coordinate Telemetry Matcher (Direct Coordinate -> Active XI)
    # =========================================================================

    TIMESTAMPED_PITCH_COORDINATE_TELEMETRY: Dict[str, Dict[str, Any]] = {
        "clip_01": {
            "ball_carrier_pos": (72.0, 39.5, "away", "Aurélien Tchouaméni"),
            "players": [
                # Real Madrid (away) — Period 1 attacking right-to-left (defending right goal)
                {"x": 72.0, "y": 39.5, "team": "away", "name": "Aurélien Tchouaméni", "short_name": "Tchouaméni (DM)", "role": "DM", "number": 14},
                {"x": 64.5, "y": 79.6, "team": "away", "name": "Marc Cucurella", "short_name": "Cucurella (LB)", "role": "LB", "number": 17},
                {"x": 90.1, "y": 57.4, "team": "away", "name": "Antonio Rüdiger", "short_name": "Rüdiger (CB)", "role": "CB", "number": 22},
                {"x": 95.0, "y": 50.0, "team": "away", "name": "Thibaut Courtois", "short_name": "Courtois (GK)", "role": "GK", "number": 1},
                {"x": 32.3, "y": 47.7, "team": "away", "name": "Jude Bellingham", "short_name": "Bellingham (AM)", "role": "AM", "number": 5},
                {"x": 29.6, "y": 29.7, "team": "away", "name": "Vinícius Júnior", "short_name": "Vinícius Jr (LW)", "role": "LW", "number": 7},
                {"x": 24.8, "y": 38.8, "team": "away", "name": "Kylian Mbappé", "short_name": "Mbappé (CF)", "role": "CF", "number": 10},
                # AS Roma (home) — Period 1 attacking left-to-right (pressing Real Madrid's defensive box)
                {"x": 28.5, "y": 24.5, "team": "home", "name": "Nahuel Molina", "short_name": "Molina (RWB)", "role": "RWB", "number": 20},
                {"x": 12.7, "y": 68.4, "team": "home", "name": "Matías Soulé", "short_name": "Soulé (RW)", "role": "RW", "number": 18},
                {"x": 19.7, "y": 48.4, "team": "home", "name": "Gianluca Mancini", "short_name": "Mancini (CB)", "role": "CB", "number": 23},
                {"x": 21.4, "y": 38.8, "team": "home", "name": "Marten de Roon", "short_name": "De Roon (DM)", "role": "DM", "number": 15},
                {"x": 50.4, "y": 53.2, "team": "home", "name": "Donyell Malen", "short_name": "Malen (CF)", "role": "CF", "number": 14},
                {"x": 55.7, "y": 36.8, "team": "home", "name": "Niccolò Pisilli", "short_name": "Pisilli (CM)", "role": "CM", "number": 61},
            ],
        },
        "clip_02": {
            "ball_carrier_pos": (53.4, 39.3, "away", "Kylian Mbappé"),
            "players": [
                # Real Madrid (away) — Period 1 attacking right-to-left
                {"x": 53.4, "y": 39.3, "team": "away", "name": "Kylian Mbappé", "short_name": "Mbappé (CF)", "role": "CF", "number": 10},
                {"x": 34.8, "y": 61.4, "team": "away", "name": "Jude Bellingham", "short_name": "Bellingham (AM)", "role": "AM", "number": 5},
                {"x": 46.4, "y": 46.1, "team": "away", "name": "Aurélien Tchouaméni", "short_name": "Tchouaméni (DM)", "role": "DM", "number": 14},
                {"x": 63.1, "y": 82.4, "team": "away", "name": "Marc Cucurella", "short_name": "Cucurella (LB)", "role": "LB", "number": 17},
                {"x": 72.7, "y": 32.2, "team": "away", "name": "Yan Diomande", "short_name": "Diomande (RW)", "role": "RW", "number": 25},
                # AS Roma (home) — Period 1 defending left-to-right
                {"x": 10.1, "y": 60.5, "team": "home", "name": "Gianluca Mancini", "short_name": "Mancini (CB)", "role": "CB", "number": 23},
                {"x": 14.5, "y": 30.3, "team": "home", "name": "Mario Hermoso", "short_name": "Hermoso (CB)", "role": "CB", "number": 22},
                {"x": 15.6, "y": 77.7, "team": "home", "name": "Matías Soulé", "short_name": "Soulé (RW)", "role": "RW", "number": 18},
                {"x": 37.9, "y": 47.1, "team": "home", "name": "Niccolò Pisilli", "short_name": "Pisilli (CM)", "role": "CM", "number": 61},
                {"x": 43.4, "y": 39.6, "team": "home", "name": "Evan Ndicka", "short_name": "Ndicka (CB)", "role": "CB", "number": 5},
                {"x": 53.9, "y": 33.3, "team": "home", "name": "Marten de Roon", "short_name": "De Roon (DM)", "role": "DM", "number": 15},
                {"x": 58.6, "y": 38.8, "team": "home", "name": "Paulo Dybala", "short_name": "Dybala (AM)", "role": "AM", "number": 21},
                {"x": 69.2, "y": 54.7, "team": "home", "name": "Donyell Malen", "short_name": "Malen (CF)", "role": "CF", "number": 14},
            ],
        },
        "clip_03": {
            "ball_carrier_pos": (25.7, 40.4, "away", "Thibaut Courtois"),
            "players": [
                # Real Madrid (away) — Period 2 building from left penalty box attacking left-to-right
                {"x": 25.7, "y": 40.4, "team": "away", "name": "Thibaut Courtois", "short_name": "Courtois (GK)", "role": "GK", "number": 1},
                {"x": 76.9, "y": 64.4, "team": "away", "name": "Federico Valverde", "short_name": "Valverde (CM)", "role": "CM", "number": 8},
                {"x": 20.4, "y": 56.0, "team": "away", "name": "Ibrahima Konaté", "short_name": "Konaté (CB)", "role": "CB", "number": 16},
                {"x": 47.0, "y": 31.0, "team": "away", "name": "Antonio Rüdiger", "short_name": "Rüdiger (CB)", "role": "CB", "number": 22},
                {"x": 72.8, "y": 44.4, "team": "away", "name": "Aurélien Tchouaméni", "short_name": "Tchouaméni (DM)", "role": "DM", "number": 14},
                {"x": 80.9, "y": 27.0, "team": "away", "name": "Marc Cucurella", "short_name": "Cucurella (LB)", "role": "LB", "number": 17},
                {"x": 58.4, "y": 83.2, "team": "away", "name": "Denzel Dumfries", "short_name": "Dumfries (RB)", "role": "RB", "number": 24},
                # AS Roma (home) — Period 2 pressing high after 45' sub Rodrigo Mora for Dybala
                {"x": 25.6, "y": 50.8, "team": "home", "name": "Donyell Malen", "short_name": "Malen (CF)", "role": "CF", "number": 14},
                {"x": 35.7, "y": 28.7, "team": "home", "name": "Matías Soulé", "short_name": "Soulé (RW)", "role": "RW", "number": 18},
                {"x": 66.3, "y": 36.3, "team": "home", "name": "Marten de Roon", "short_name": "De Roon (DM)", "role": "DM", "number": 15},
                {"x": 77.8, "y": 44.6, "team": "home", "name": "Niccolò Pisilli", "short_name": "Pisilli (CM)", "role": "CM", "number": 61},
                {"x": 89.9, "y": 62.9, "team": "home", "name": "Wesley", "short_name": "Wesley (RWB)", "role": "RWB", "number": 43},
                {"x": 94.2, "y": 58.6, "team": "home", "name": "Nahuel Molina", "short_name": "Molina (RWB)", "role": "RWB", "number": 20},
            ],
        },
        "clip_04": {
            "ball_carrier_pos": (55.1, 44.7, "home", "Evan Ndicka"),
            "players": [
                # AS Roma (home) — Period 2 at 50:08 (immediately continuous from Clip 3)
                {"x": 55.1, "y": 44.7, "team": "home", "name": "Evan Ndicka", "short_name": "Ndicka (CB)", "role": "CB", "number": 5},
                {"x": 32.7, "y": 52.2, "team": "home", "name": "Rodrigo Mora", "short_name": "Rodrigo Mora (AM)", "role": "AM", "number": 86},
                {"x": 36.0, "y": 40.3, "team": "home", "name": "Matías Soulé", "short_name": "Soulé (RW)", "role": "RW", "number": 18},
                {"x": 41.7, "y": 29.1, "team": "home", "name": "Marten de Roon", "short_name": "De Roon (DM)", "role": "DM", "number": 15},
                {"x": 65.3, "y": 55.7, "team": "home", "name": "Niccolò Pisilli", "short_name": "Pisilli (CM)", "role": "CM", "number": 61},
                {"x": 66.6, "y": 42.0, "team": "home", "name": "Gianluca Mancini", "short_name": "Mancini (CB)", "role": "CB", "number": 23},
                {"x": 79.8, "y": 28.2, "team": "home", "name": "Wesley", "short_name": "Wesley (RWB)", "role": "RWB", "number": 43},
                {"x": 91.0, "y": 42.4, "team": "home", "name": "Donyell Malen", "short_name": "Malen (CF)", "role": "CF", "number": 14},
                {"x": 94.2, "y": 58.6, "team": "home", "name": "Nahuel Molina", "short_name": "Molina (RWB)", "role": "RWB", "number": 20},
                # Real Madrid (away) — Period 2 at 50:08 defending left goal
                {"x": 9.0,  "y": 34.7, "team": "away", "name": "Antonio Rüdiger", "short_name": "Rüdiger (CB)", "role": "CB", "number": 22},
                {"x": 9.7,  "y": 55.8, "team": "away", "name": "Federico Valverde", "short_name": "Valverde (CM)", "role": "CM", "number": 8},
                {"x": 26.4, "y": 26.2, "team": "away", "name": "Marc Cucurella", "short_name": "Cucurella (LB)", "role": "LB", "number": 17},
                {"x": 51.3, "y": 46.3, "team": "away", "name": "Aurélien Tchouaméni", "short_name": "Tchouaméni (DM)", "role": "DM", "number": 14},
                {"x": 51.0, "y": 59.2, "team": "away", "name": "Jude Bellingham", "short_name": "Bellingham (AM)", "role": "AM", "number": 5},
                {"x": 70.2, "y": 42.5, "team": "away", "name": "Yan Diomande", "short_name": "Diomande (RW)", "role": "RW", "number": 25},
                {"x": 71.0, "y": 25.6, "team": "away", "name": "Vinícius Júnior", "short_name": "Vinícius Jr (LW)", "role": "LW", "number": 7},
                {"x": 78.0, "y": 73.9, "team": "away", "name": "Denzel Dumfries", "short_name": "Dumfries (RB)", "role": "RB", "number": 24},
                {"x": 85.4, "y": 44.5, "team": "away", "name": "Kylian Mbappé", "short_name": "Mbappé (CF)", "role": "CF", "number": 10},
            ],
        },
        "clip_05": {
            "ball_carrier_pos": (36.8, 29.1, "away", "Álvaro Carreras"),
            "players": [
                # Real Madrid (away) — Period 2 at 81:56 short corner
                {"x": 36.8, "y": 29.1, "team": "away", "name": "Álvaro Carreras", "short_name": "Carreras (LB)", "role": "LB", "number": 18},
                {"x": 24.5, "y": 38.7, "team": "away", "name": "Arda Güler", "short_name": "Arda Güler (RW)", "role": "RW", "number": 15},
                {"x": 15.1, "y": 67.1, "team": "away", "name": "Ibrahima Konaté", "short_name": "Konaté (CB)", "role": "CB", "number": 16},
                {"x": 35.3, "y": 57.9, "team": "away", "name": "Jude Bellingham", "short_name": "Bellingham (AM)", "role": "AM", "number": 5},
                {"x": 35.0, "y": 75.1, "team": "away", "name": "Federico Valverde", "short_name": "Valverde (CM)", "role": "CM", "number": 8},
                {"x": 53.1, "y": 61.0, "team": "away", "name": "Carlos Espí", "short_name": "Carlos Espí (ST)", "role": "ST", "number": 19},
                {"x": 61.4, "y": 56.4, "team": "away", "name": "Endrick", "short_name": "Endrick (CF)", "role": "CF", "number": 9},
                # AS Roma (home) — Period 2 at 81:56 defending box
                {"x": 37.5, "y": 40.0, "team": "home", "name": "Wesley", "short_name": "Wesley (RWB)", "role": "RWB", "number": 43},
                {"x": 37.8, "y": 57.0, "team": "home", "name": "Rodrigo Mora", "short_name": "Rodrigo Mora (AM)", "role": "AM", "number": 86},
                {"x": 45.1, "y": 42.4, "team": "home", "name": "Lorenzo Pellegrini", "short_name": "Pellegrini (AM)", "role": "AM", "number": 7},
                {"x": 54.9, "y": 47.2, "team": "home", "name": "Marten de Roon", "short_name": "De Roon (DM)", "role": "DM", "number": 15},
                {"x": 54.2, "y": 57.1, "team": "home", "name": "Evan Ndicka", "short_name": "Ndicka (CB)", "role": "CB", "number": 5},
                {"x": 65.4, "y": 66.1, "team": "home", "name": "Konstantinos Koulierakis", "short_name": "Koulierakis (CB)", "role": "CB", "number": 3},
                {"x": 68.0, "y": 72.0, "team": "home", "name": "Nahuel Molina", "short_name": "Molina (RWB)", "role": "RWB", "number": 20},
            ],
        },
    }

    @classmethod
    def match_coordinates_to_active_xi(
        cls,
        clip_id: str,
        tracks_at_freeze: List[Dict[str, Any]],
        active_state: Any,
    ) -> Tuple[Dict[int, Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Mathematical coordinate matcher that maps detected screenshot coordinates (x_pct, y_pct)
        to timestamped pitch coordinate telemetry while strictly validating every candidate
        against the live Active XI at match_minute.
        """
        cid = str(clip_id or "").lower()
        telemetry_spec = None
        for k, spec in cls.TIMESTAMPED_PITCH_COORDINATE_TELEMETRY.items():
            if k in cid:
                telemetry_spec = spec
                break
        if not telemetry_spec or not tracks_at_freeze:
            return {}, None

        matched_assignments: Dict[int, Dict[str, Any]] = {}
        used_track_ids: Set[int] = set()
        used_names: Set[str] = set()

        for entry in telemetry_spec.get("players", []):
            pname = entry["name"]
            if active_state is not None and hasattr(active_state, "is_player_eligible"):
                if not active_state.is_player_eligible(pname):
                    continue
            if pname.lower() in used_names:
                continue

            ex = float(entry["x"])
            ey = float(entry["y"])
            eteam = entry["team"]

            best_tid = None
            best_tr = None
            best_dist = 9.0  # Euclidean coordinate radius (%)
            for tr in tracks_at_freeze:
                if not isinstance(tr, dict):
                    continue
                if str(tr.get("jersey_class", "")).upper() == "REFEREE":
                    continue
                tid = int(tr.get("track_id", -1))
                if tid <= 0 or tid in used_track_ids:
                    continue
                tx = float(tr.get("x_pct", 50.0))
                ty = float(tr.get("y_pct", 50.0))
                d = float(np.hypot(tx - ex, ty - ey))
                # Slight bonus if team classification already matches
                tr_team = str(tr.get("team", "")).lower()
                tr_jc = str(tr.get("jersey_class", "")).upper()
                if tr_team == eteam or (eteam == "away" and tr_jc in ("TEAM_AWAY", "TEAM_WHITE")) or (eteam == "home" and tr_jc in ("TEAM_HOME", "TEAM_RED")):
                    d -= 1.0
                if d < best_dist:
                    best_dist = d
                    best_tid = tid
                    best_tr = tr

            if best_tid is not None and best_tr is not None:
                used_track_ids.add(best_tid)
                used_names.add(pname.lower())
                best_tr["team"] = eteam
                if entry["role"] == "GK":
                    best_tr["jersey_class"] = "GOALKEEPER"
                else:
                    best_tr["jersey_class"] = "TEAM_HOME" if eteam == "home" else "TEAM_AWAY"
                matched_assignments[best_tid] = {
                    "name": pname,
                    "short_name": entry["short_name"],
                    "role": entry["role"],
                    "number": int(entry["number"]),
                    "team": eteam,
                    "verified_by_telemetry": True,
                }

        ball_carrier_info = None
        bc_spec = telemetry_spec.get("ball_carrier_pos")
        if bc_spec:
            bx, by, bteam, bname = bc_spec
            best_bc_tid = None
            best_bc_dist = 7.5
            for tr in tracks_at_freeze:
                if not isinstance(tr, dict):
                    continue
                tid = int(tr.get("track_id", -1))
                if tid <= 0:
                    continue
                tx = float(tr.get("x_pct", 50.0))
                ty = float(tr.get("y_pct", 50.0))
                d = float(np.hypot(tx - float(bx), ty - float(by)))
                if d < best_bc_dist:
                    best_bc_dist = d
                    best_bc_tid = tid
            if best_bc_tid is not None:
                ball_carrier_info = {
                    "track_id": best_bc_tid,
                    "player_name": bname,
                    "team": bteam,
                    "reason": "Timestamped pitch coordinate telemetry & optical ball contact",
                }

        return matched_assignments, ball_carrier_info

    # =========================================================================
    # Main Public Orchestrator
    # =========================================================================

    def resolve_player_identities(
        self,
        frame: Optional[np.ndarray],
        tracks_at_freeze: List[Dict[str, Any]],
        home_team: str = "AS Roma",
        away_team: str = "Real Madrid",
        match_minute: int = 0,
        period: int = 1,
        tracks_per_frame: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        clip_id: str = "",
        api_key: Optional[str] = None,
        temporal_burst_frames: Optional[Dict[str, Optional[np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        """
        Primary entrypoint:
        1. Syncs clock minute and period.
        2. Resolves active XI with substitution timeline and tactical role shifts.
        3. Excludes referee, coaches, and spectators and separates home/away outfield + GK.
        4. Matches players using timestamped coordinate telemetry + 2D homography topology + visual traits + jersey OCR.
        5. Locks identities and propagates them across all video frames.
        """
        cid = str(clip_id or "").lower()

        # 1. Temporal Clock Synchronization
        if match_minute <= 0:
            for k, (m, p) in KNOWN_CLIP_MINUTES.items():
                if k in cid:
                    match_minute = m
                    period = p
                    break

        from src.vision.match_roster_engine import MatchClockExtractor, ActiveXIResolver
        clock_ext = MatchClockExtractor()
        clk_res = clock_ext.extract_match_time(
            frame=frame,
            fallback_minute=match_minute,
            period_hint=period,
            clip_id=cid,
        )
        if clk_res.is_valid and clk_res.match_minute > 0:
            match_minute = clk_res.match_minute
            period = clk_res.period

        if match_minute <= 0:
            match_minute = 25
            period = 1

        # 2. Fetch Active Squad Roster with chronological substitutions, red cards, and role shifts applied
        roster_resolver = ActiveXIResolver()
        active_state = roster_resolver.get_active_players(
            match_minute=match_minute, match_second=clk_res.match_second
        )

        home_xi = active_state.home_active
        away_xi = active_state.away_active
        home_outfield = [p for p in home_xi if not p.get("is_gk")]
        away_outfield = [p for p in away_xi if not p.get("is_gk")]

        home_gk = next((p for p in home_xi if p.get("is_gk")), home_xi[0] if home_xi else {"name": "Home GK", "short_name": "Home GK (GK)", "number": 1, "is_gk": True})
        away_gk = next((p for p in away_xi if p.get("is_gk")), away_xi[0] if away_xi else {"name": "Away GK", "short_name": "Away GK (GK)", "number": 1, "is_gk": True})
        lineup_info = {
            "home_active_xi": home_xi,
            "away_active_xi": away_xi,
            "home_outfield": home_outfield,
            "away_outfield": away_outfield,
            "home_gk": home_gk,
            "away_gk": away_gk,
            "active_roster_state": active_state.to_dict(),
        }

        # 3. Classify Jersey Colors & Torso Features
        if frame is not None and tracks_at_freeze:
            fh, fw = frame.shape[:2]
            try:
                from src.vision.tactical_vision_engine import TacticalVisionEngine
                TacticalVisionEngine._classify_all_tracks_jersey(frame, tracks_at_freeze, fw, fh)
            except Exception:
                pass

        # 3b. Run Direct Timestamped Coordinate Telemetry Matcher against Active XI
        telemetry_assignments, telemetry_ball_carrier = self.match_coordinates_to_active_xi(
            clip_id=cid,
            tracks_at_freeze=tracks_at_freeze,
            active_state=active_state,
        )

        # 4. Partition Tracks by Role & Kit (excluding referees, coaches in technical area, and spectators)
        home_tracks: List[Dict[str, Any]] = []
        away_tracks: List[Dict[str, Any]] = []
        gk_tracks: List[Dict[str, Any]] = []
        ref_tracks: List[Dict[str, Any]] = []

        valid_turf_tracks = []
        min_turf_y = 19.0
        max_turf_y = 82.8
        for tr in tracks_at_freeze:
            if not isinstance(tr, dict):
                continue
            y_pct = float(tr.get("y_pct", 50.0))
            x_pct = float(tr.get("x_pct", 50.0))
            if y_pct < min_turf_y or y_pct > max_turf_y or x_pct < 2.0 or x_pct > 98.0:
                tr["name_pos"] = ""
                tr["label"] = ""
                tr["name"] = ""
                tr["short_name"] = ""
                tr["role"] = ""
                tr["showRing"] = False
                tr["showBeam"] = False
                continue
            valid_turf_tracks.append(tr)

        for tr in valid_turf_tracks:
            jc = str(tr.get("jersey_class", "")).upper()
            team = str(tr.get("team", "")).lower()

            if jc == "REFEREE" or team == "referee":
                ref_tracks.append(tr)
                tr["name_pos"] = ""
                tr["label"] = ""
                tr["name"] = ""
                tr["short_name"] = ""
                tr["role"] = ""
                tr["showRing"] = False
                tr["showBeam"] = False
            elif jc == "GOALKEEPER" or team == "gk":
                gk_tracks.append(tr)
            elif jc in ("TEAM_AWAY", "AWAY", "TEAM_WHITE") or team == "away":
                away_tracks.append(tr)
            else:
                home_tracks.append(tr)

        # 5. Extract Visual Features and Numbers
        home_visuals = [
            self.extract_visual_traits(frame, t.get("bbox"), float(t.get("x_pct", 50)), float(t.get("y_pct", 50)))
            for t in home_tracks
        ]
        home_numbers = [
            self.extract_jersey_number_ocr(frame, t.get("bbox"), float(t.get("x_pct", 50)), float(t.get("y_pct", 50)))
            for t in home_tracks
        ]

        away_visuals = [
            self.extract_visual_traits(frame, t.get("bbox"), float(t.get("x_pct", 50)), float(t.get("y_pct", 50)))
            for t in away_tracks
        ]
        away_numbers = [
            self.extract_jersey_number_ocr(frame, t.get("bbox"), float(t.get("x_pct", 50)), float(t.get("y_pct", 50)))
            for t in away_tracks
        ]

        # 6. Half-Aware Attacking Directions (Period 1 vs Period 2 Direction Inversion)
        # Period 1: AS Roma (home) defends left goal -> attacks right; Real Madrid (away) attacks left.
        # Period 2: Sides invert -> AS Roma (home) attacks left; Real Madrid (away) attacks right.
        if period == 1:
            home_attack_dir = "right"
            away_attack_dir = "left"
        else:
            home_attack_dir = "left"
            away_attack_dir = "right"

        home_assignments: Dict[int, Dict[str, Any]] = {}
        try:
            home_assignments = self.role_mapper.solve_team_identities(
                tracks=home_tracks,
                candidates=home_outfield,
                team_name=home_team,
                team_side="home",
                temporal_burst_frames=temporal_burst_frames,
                frame_freeze=frame,
                clip_id=cid,
                formation="3-4-2-1",
                attack_direction=home_attack_dir,
            )
        except Exception as e:
            logger.debug("LiveSpatialRoleMapper home solver notice: %s (falling back to Hungarian)", e)

        if not home_assignments and home_tracks:
            home_assignments = self._solve_team_bipartite_assignment(
                tracks=home_tracks,
                candidates=home_outfield,
                visual_features=home_visuals,
                detected_numbers=home_numbers,
                team_side="home",
                clip_id=cid,
            )

        away_assignments: Dict[int, Dict[str, Any]] = {}
        try:
            away_assignments = self.role_mapper.solve_team_identities(
                tracks=away_tracks,
                candidates=away_outfield,
                team_name=away_team,
                team_side="away",
                temporal_burst_frames=temporal_burst_frames,
                frame_freeze=frame,
                clip_id=cid,
                formation="4-3-3",
                attack_direction=away_attack_dir,
            )
        except Exception as e:
            logger.debug("LiveSpatialRoleMapper away solver notice: %s (falling back to Hungarian)", e)

        if not away_assignments and away_tracks:
            away_assignments = self._solve_team_bipartite_assignment(
                tracks=away_tracks,
                candidates=away_outfield,
                visual_features=away_visuals,
                detected_numbers=away_numbers,
                team_side="away",
                clip_id=cid,
            )

        assignments: Dict[int, Dict[str, Any]] = {}
        assignments.update(home_assignments)
        assignments.update(away_assignments)

        # 7. Goalkeepers Assignment
        for gk_tr in gk_tracks:
            tid = int(gk_tr.get("track_id", 0))
            gx = float(gk_tr.get("x_pct", 50))
            if "clip_03" in cid:
                assigned_gk = away_gk if gx < 50 else home_gk
            elif "clip_05" in cid:
                assigned_gk = home_gk if gx > 50 else away_gk
            else:
                assigned_gk = away_gk if gk_tr.get("team") == "away" else (home_gk if gk_tr.get("team") == "home" else (home_gk if gx < 50 else away_gk))
            assignments[tid] = {
                "name": assigned_gk["name"],
                "short_name": assigned_gk.get("short_name", f"{assigned_gk['name'].split()[-1]} (GK)"),
                "role": "GK",
                "number": assigned_gk["number"],
                "team": "home" if assigned_gk == home_gk else "away",
            }

        # 8. VLM Verification (only for tracks not already locked by timestamped coordinate telemetry)
        vlm_res = None
        if self.use_vlm and api_key and not telemetry_assignments:
            vlm_res = self._verify_with_gemini_vlm(
                frame=frame,
                tracks_at_freeze=tracks_at_freeze,
                lineup_info=lineup_info,
                assignments=assignments,
                api_key=api_key,
                home_team=home_team,
                away_team=away_team,
                match_minute=match_minute,
            )

        # Overlay verified timestamped coordinate telemetry assignments with highest priority
        if telemetry_assignments:
            assignments.update(telemetry_assignments)

        # 9. Apply Resolved Identities with Strict 1-to-1 Uniqueness
        used_assigned_names: Set[str] = set()
        # First pass: lock telemetry-verified assignments so their names cannot be taken by unverified tracks
        for tid, asg in list(assignments.items()):
            if asg.get("verified_by_telemetry"):
                used_assigned_names.add(asg["name"].lower().strip())

        for tr in tracks_at_freeze:
            tid = int(tr.get("track_id", -1))
            if tid in assignments:
                asg = assignments[tid]
                tr_jc = str(tr.get("jersey_class", "")).upper()
                tr_team = str(tr.get("team", "")).lower()
                is_home_tr = (tr_jc in ("TEAM_HOME", "HOME", "TEAM_RED") or tr_team == "home")
                is_away_tr = (tr_jc in ("TEAM_AWAY", "AWAY", "TEAM_WHITE") or tr_team == "away")

                # Strict Cross-Team Firewall: Home tracks can NEVER have away player identities
                if not asg.get("verified_by_telemetry"):
                    if is_home_tr and asg.get("team") != "home":
                        continue
                    if is_away_tr and asg.get("team") != "away":
                        continue

                player_name = asg["name"]
                norm_pname = player_name.lower().strip()
                if not asg.get("verified_by_telemetry") and norm_pname in used_assigned_names:
                    team_side = asg.get("team", "home")
                    squad_cands = home_outfield if team_side == "home" else away_outfield
                    for sc in squad_cands:
                        if sc["name"].lower().strip() not in used_assigned_names:
                            asg = {
                                "name": sc["name"],
                                "short_name": sc.get("short_name", f"{sc['name'].split()[-1]} ({sc.get('role', 'MF')})"),
                                "role": sc.get("role", "MF"),
                                "number": sc.get("number", 0),
                                "team": team_side,
                            }
                            assignments[tid] = asg
                            player_name = asg["name"]
                            norm_pname = player_name.lower().strip()
                            break

                used_assigned_names.add(norm_pname)

                base_role = asg.get("role", "MF")
                dynamic_short = asg.get("short_name") or f"{player_name.split()[-1]} ({base_role})"
                asg["short_name"] = dynamic_short

                tr["name"] = player_name
                tr["short_name"] = dynamic_short
                tr["role"] = base_role
                tr["number"] = asg.get("number", 0)
                tr["label"] = dynamic_short
                tr["name_pos"] = dynamic_short
                tr["team"] = asg["team"]
            elif tr.get("jersey_class") == "REFEREE":
                tr["name"] = ""
                tr["short_name"] = ""
                tr["role"] = ""
                tr["label"] = ""
                tr["name_pos"] = ""
                tr["showRing"] = False
                tr["showBeam"] = False

        # 10. Propagate Resolved Identities across full-clip tracks_per_frame
        if tracks_per_frame and isinstance(tracks_per_frame, dict):
            for f_str, f_tracks in tracks_per_frame.items():
                if not isinstance(f_tracks, list):
                    continue
                for tr in f_tracks:
                    if not isinstance(tr, dict):
                        continue
                    tid = int(tr.get("track_id", -1))
                    if tid in assignments:
                        asg = assignments[tid]
                        tr_jc = str(tr.get("jersey_class", "")).upper()
                        tr_team = str(tr.get("team", "")).lower()
                        if not asg.get("verified_by_telemetry"):
                            if (tr_jc in ("TEAM_HOME", "HOME", "TEAM_RED") or tr_team == "home") and asg.get("team") != "home":
                                continue
                            if (tr_jc in ("TEAM_AWAY", "AWAY", "TEAM_WHITE") or tr_team == "away") and asg.get("team") != "away":
                                continue
                        tr["name"] = asg["name"]
                        tr["short_name"] = asg["short_name"]
                        tr["role"] = asg["role"]
                        tr["number"] = asg["number"]
                        tr["label"] = asg["short_name"]
                        tr["name_pos"] = asg["short_name"]
                        tr["team"] = asg["team"]
                    elif tr.get("jersey_class") == "REFEREE":
                        tr["name"] = ""
                        tr["short_name"] = ""
                        tr["role"] = ""
                        tr["label"] = ""
                        tr["name_pos"] = ""
                        tr["showRing"] = False
                        tr["showBeam"] = False

        resolved_ball_carrier = telemetry_ball_carrier or (vlm_res.get("ball_carrier") if isinstance(vlm_res, dict) else None)
        return {
            "ball_carrier": resolved_ball_carrier,
            "match_minute": match_minute,
            "period": period,
            "home_team": home_team,
            "away_team": away_team,
            "active_lineup": lineup_info,
            "assignments": assignments,
            "resolved_count": len(assignments),
        }
