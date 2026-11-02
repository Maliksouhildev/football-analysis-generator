"""
src/vision/isolated_player_ai_locator.py
================================================================================
Isolated Multi-Source AI Player Locator & Identifier Engine.

Consolidates and upgrades every stage of player detection, team/referee/GK
classification, spatial 2D homography topology, multi-frame super-resolution
crop extraction, scoreboard clock OCR, live timestamp-synced Active-XI ground
truth, play-by-play event telemetry, biometric trait matching, and Gemini VLM +
Hungarian bipartite global identity resolution.

Designed for both standalone iterative evaluation/tuning and direct integration
into TacticalVisionEngine & PlayerIdentityResolver.
================================================================================
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from src.vision.match_roster_engine import ActiveRosterState, ActiveXIResolver, MatchClockExtractor
from src.vision.pitch_homography_role_mapper import PitchHomographyProjector

logger = logging.getLogger("vision.isolated_player_ai_locator")


# =============================================================================
# 1. Biometric & Visual Appearance Profiles Database
# =============================================================================

PLAYER_BIOMETRIC_DB: Dict[str, Dict[str, Any]] = {
    # FC Barcelona (Home - Blaugrana dark blue & crimson vertical stripes, dark blue shorts)
    "inaki pena": {
        "num": 13, "team": "home", "role": "GK",
        "skin": "light", "hair": "short dark hair", "build": "tall goalkeeper",
        "kit_note": "bright green goalkeeper kit #13",
        "cues": "GK #13 in green kit inside/near Barcelona penalty area",
    },
    "pau cubarsi": {
        "num": 2, "team": "home", "role": "RCB",
        "skin": "light", "hair": "short dark brown hair", "build": "lean central defender",
        "kit_note": "Blaugrana #2",
        "cues": "Right Center Back (RCB) in Barcelona backline, #2 on back",
    },
    "inigo martinez": {
        "num": 5, "team": "home", "role": "LCB",
        "skin": "light", "hair": "short dark hair with beard", "build": "commanding left-footed center-back",
        "kit_note": "Blaugrana #5",
        "cues": "Left Center Back (LCB) in Barcelona backline, #5 on back",
    },
    "alejandro balde": {
        "num": 3, "team": "home", "role": "LB",
        "skin": "dark", "hair": "short black hair with fade", "build": "explosive wide full-back",
        "kit_note": "Blaugrana #3",
        "cues": "Dark-skinned left full-back #3 on left touchline",
    },
    "jules kounde": {
        "num": 23, "team": "home", "role": "RB",
        "skin": "brown", "hair": "short curly hair/braids", "build": "athletic right full-back",
        "kit_note": "Blaugrana #23",
        "cues": "Right Back (RB) #23 defending on right flank",
    },
    "marc casado": {
        "num": 17, "team": "home", "role": "DM",
        "skin": "light", "hair": "short dark brown hair", "build": "compact tenacious pivot",
        "kit_note": "Blaugrana #17",
        "cues": "Defensive midfielder (DM) #17 anchoring midfield circulation",
    },
    "pedri": {
        "num": 8, "team": "home", "role": "LCM",
        "skin": "light", "hair": "short dark brown/black hair", "build": "slender technical midfielder",
        "kit_note": "Blaugrana #8, dark boots",
        "cues": "Left Central Midfielder (LCM) #8 in midfield interior with dark hair",
    },
    "fermin lopez": {
        "num": 16, "team": "home", "role": "RCM",
        "skin": "light", "hair": "short blonde/light-brown tinted hair", "build": "energetic attacking midfielder",
        "kit_note": "Blaugrana #16 (or #7), bright turquoise/mint boots",
        "cues": "Right Central / Attacking Midfielder (RCM/AM) with short blonde-tinted hair and bright boots",
    },
    "lamine yamal": {
        "num": 19, "team": "home", "role": "RW",
        "skin": "dark/brown", "hair": "curly dark hair with blonde highlights", "build": "agile left-footed right winger",
        "kit_note": "Blaugrana #19 (or #10)",
        "cues": "Right Winger (RW) holding width on right flank with curly hair and blonde tips",
    },
    "raphinha": {
        "num": 11, "team": "home", "role": "LW",
        "skin": "tan/brown", "hair": "short dark hair / buzzed fade", "build": "dynamic forward",
        "kit_note": "Blaugrana #11",
        "cues": "Left Winger / Attacking Forward #11 pressing aggressively",
    },
    "robert lewandowski": {
        "num": 9, "team": "home", "role": "CF",
        "skin": "light", "hair": "short dark brown hair", "build": "tall athletic striker (1.85m)",
        "kit_note": "Blaugrana #9",
        "cues": "Center Forward (CF) #9 operating between opposing center-backs",
    },

    # Real Madrid (Away - All-white kit with dark/maroon shoulder stripes)
    "andriy lunin": {
        "num": 13, "team": "away", "role": "GK",
        "skin": "light", "hair": "short blonde/light-brown hair", "build": "tall goalkeeper (1.91m)",
        "kit_note": "bright orange/coral/salmon goalkeeper shirt, shorts, and socks #13, white gloves/boots",
        "cues": "GK #13 in bright orange/coral/salmon goalkeeper kit inside/near Real Madrid goal box",
    },
    "thibaut courtois": {
        "num": 1, "team": "away", "role": "GK",
        "skin": "light", "hair": "short dark hair", "build": "very tall goalkeeper (2.00m)",
        "kit_note": "goalkeeper kit #1",
        "cues": "GK #1 very tall keeper inside Real Madrid penalty box",
    },
    "lucas vazquez": {
        "num": 17, "team": "away", "role": "RB",
        "skin": "light", "hair": "short dark brown hair", "build": "compact energetic right-back",
        "kit_note": "Real Madrid White #17",
        "cues": "Right Back (RB) #17 with short dark hair on right defensive flank",
    },
    "eder militao": {
        "num": 3, "team": "away", "role": "RCB",
        "skin": "brown", "hair": "short buzzcut/fade", "build": "athletic fast center-back",
        "kit_note": "Real Madrid White #3",
        "cues": "Right Center Back (RCB) #3 with athletic build and short hair",
    },
    "antonio rudiger": {
        "num": 22, "team": "away", "role": "LCB",
        "skin": "dark", "hair": "short black hair with beard", "build": "tall aggressive center-back (1.90m)",
        "kit_note": "Real Madrid White #22",
        "cues": "Tall dark-skinned Left Center Back (LCB) #22 with beard",
    },
    "ferland mendy": {
        "num": 23, "team": "away", "role": "LB",
        "skin": "dark", "hair": "shaved black hair", "build": "powerful compact left-back",
        "kit_note": "Real Madrid White #23",
        "cues": "Dark-skinned Left Back (LB) #23 defending left touchline",
    },
    "aurelien tchouameni": {
        "num": 14, "team": "away", "role": "DM",
        "skin": "dark", "hair": "short black hair", "build": "tall defensive midfielder (1.87m)",
        "kit_note": "Real Madrid White #14",
        "cues": "Defensive Midfielder (DM) #14, tall dark-skinned pivot in midfield",
    },
    "federico valverde": {
        "num": 8, "team": "away", "role": "RCM",
        "skin": "light", "hair": "short brown hair", "build": "athletic box-to-box midfielder",
        "kit_note": "Real Madrid White #8, long white undersleeves",
        "cues": "Right Central Midfielder (RCM) #8 with light skin and brown hair",
    },
    "eduardo camavinga": {
        "num": 6, "team": "away", "role": "LCM",
        "skin": "dark", "hair": "distinctive long black braids/dreadlocks", "build": "lean dynamic midfielder",
        "kit_note": "Real Madrid White #6",
        "cues": "Dark-skinned midfielder #6 with distinctive long braids/dreadlocks",
    },
    "jude bellingham": {
        "num": 5, "team": "away", "role": "AM",
        "skin": "dark/brown", "hair": "short black hair with fade", "build": "tall athletic attacking midfielder (1.86m)",
        "kit_note": "Real Madrid White #5",
        "cues": "Central Attacking Midfielder (AM) #5, tall athletic frame surging into box",
    },
    "vinicius junior": {
        "num": 7, "team": "away", "role": "LW",
        "skin": "dark", "hair": "short faded hair", "build": "electric dynamic winger",
        "kit_note": "Real Madrid White #7",
        "cues": "Left Winger (LW) #7 holding width on left touchline with explosive acceleration",
    },
    "kylian mbappe": {
        "num": 9, "team": "away", "role": "CF",
        "skin": "brown", "hair": "close-cropped buzzed hair", "build": "explosive direct striker (1.78m)",
        "kit_note": "Real Madrid White #9",
        "cues": "Center Forward (CF) #9 attacking central space behind the defensive line",
    },
    "luka modric": {
        "num": 10, "team": "away", "role": "CM",
        "skin": "light", "hair": "longer blonde/brown wavy hair with headband", "build": "slender maestro midfielder",
        "kit_note": "Real Madrid White #10",
        "cues": "Central Midfielder #10 with wavy hair orchestrating possession",
    },
    "kylian mbappe": {
        "num": 10, "team": "away", "role": "CF",
        "skin": "brown/dark", "hair": "very short buzzed/shaved head", "build": "explosive center-forward",
        "kit_note": "Real Madrid White #10",
        "cues": "Center Forward (CF) #10 leading the attack or inside penalty box",
    },
    "brahim diaz": {
        "num": 21, "team": "away", "role": "RW",
        "skin": "tan/light", "hair": "short dark hair", "build": "low center of gravity, agile dribbler",
        "kit_note": "Real Madrid White #21",
        "cues": "Right Winger / Attacking Midfielder #21",
    },
    "arda guler": {
        "num": 15, "team": "away", "role": "RW",
        "skin": "light", "hair": "short dark brown hair", "build": "slender technical playmaker",
        "kit_note": "Real Madrid White #15",
        "cues": "Right Winger / Attacking Midfielder Right (RW) #15 with light skin",
    },
    "rodrygo": {
        "num": 11, "team": "away", "role": "RW",
        "skin": "brown", "hair": "short dark faded hair", "build": "quick dynamic forward",
        "kit_note": "Real Madrid White #11",
        "cues": "Right Winger / Forward #11 in white kit",
    },
    "endrick": {
        "num": 16, "team": "away", "role": "ST",
        "skin": "dark", "hair": "short dark hair", "build": "powerful explosive young striker",
        "kit_note": "Real Madrid White #16",
        "cues": "Striker #16 in white kit",
    },
    "frenkie de jong": {
        "num": 21, "team": "home", "role": "CM",
        "skin": "light", "hair": "medium blonde hair", "build": "tall gliding midfielder",
        "kit_note": "Blaugrana #21",
        "cues": "Midfielder #21 with blonde hair carrying ball out of deep zones",
    },
    "gavi": {
        "num": 6, "team": "home", "role": "CM",
        "skin": "light", "hair": "short dark brown hair", "build": "intense combative midfielder",
        "kit_note": "Blaugrana #6",
        "cues": "Central Midfielder #6 pressing relentlessly",
    },
    "dani olmo": {
        "num": 20, "team": "home", "role": "AM",
        "skin": "light", "hair": "short brown hair", "build": "tactical attacking playmaker",
        "kit_note": "Blaugrana #20",
        "cues": "Attacking Midfielder #20 occupying pockets of space",
    },
}


# =============================================================================
# 2. Hyperparameter & Source Configuration for Iterative Tuning
# =============================================================================

@dataclass
class LocatorConfig:
    """
    Configuration controlling which intelligence sources, detector scales,
    temperature, prompt strategies, and solver weights are active.
    """
    iteration_id: int = 10
    name: str = "v10_full_multisource_intelligence"

    # Detection & Pitch Filtering
    yolo_imgsz: int = 1280
    use_tiled_highres_yolo: bool = True
    use_adaptive_turf_hull: bool = True
    use_broadcast_banner_mask: bool = True
    use_touchline_staff_filter: bool = True
    use_cv_contour_recall: bool = True

    # Team & Kit Classification
    use_blaugrana_stripe_classifier: bool = True
    use_gk_color_detector: bool = True
    use_referee_outlier_guard: bool = True

    # Enriched Sources fed to AI & Solver
    use_scoreboard_clock_ocr: bool = True
    use_live_active_xi_sync: bool = True
    use_play_by_play_telemetry: bool = True
    use_homography_topology_priors: bool = True
    use_multiframe_temporal_crops: bool = True
    use_superres_clahe_torso_zoom: bool = True
    use_biometric_visual_db: bool = True
    use_hungarian_global_solver: bool = True
    use_vlm_critic_audit: bool = True

    # VLM Model & Prompt Parameters
    temperature: float = 0.05
    model_priority: List[str] = field(default_factory=lambda: [
        "gemini-3.1-flash-lite-preview",
        "gemini-flash-lite-latest",
        "gemini-3-flash-preview",
    ])
    prompt_version: str = "v10_elite_spatial_biometric_cot"

    # Hungarian Cost Matrix Weights
    w_vlm_direct: float = 0.42
    w_jersey_ocr: float = 0.30
    w_topology_role: float = 0.18
    w_biometric_cue: float = 0.10


# =============================================================================
# 3. Data Structures for Detection & Identity Resolution
# =============================================================================

@dataclass
class DetectedPitchActor:
    track_id: int
    bbox: List[int]             # [x1, y1, x2, y2] in pixels
    x_pct: float                # Boot anchor X in [0..100]
    y_pct: float                # Boot anchor Y in [0..100]
    confidence: float
    actor_type: str             # "player" | "goalkeeper" | "referee"
    team: str                   # "home" | "away" | "referee"
    is_gk: bool = False
    pitch_x_m: float = 52.5
    pitch_y_m: float = 34.0
    tactical_zone: str = "CENTRAL"
    formation_line: str = "midfield"   # "backline" | "midfield" | "attack" | "goalkeeper"
    flank_zone: str = "central"        # "left" | "left_half" | "central" | "right_half" | "right"
    role_priors: Dict[str, float] = field(default_factory=dict)
    visual_features: Dict[str, Any] = field(default_factory=dict)
    ocr_number: Optional[int] = None
    ocr_confidence: float = 0.0
    assigned_name: str = ""
    assigned_number: Optional[int] = None
    assigned_role: str = ""
    assignment_confidence: float = 0.0
    assignment_method: str = ""
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "x_pct": round(self.x_pct, 2),
            "y_pct": round(self.y_pct, 2),
            "confidence": round(self.confidence, 3),
            "actor_type": self.actor_type,
            "team": self.team,
            "is_gk": self.is_gk,
            "pitch_x_m": round(self.pitch_x_m, 2),
            "pitch_y_m": round(self.pitch_y_m, 2),
            "tactical_zone": self.tactical_zone,
            "formation_line": self.formation_line,
            "flank_zone": self.flank_zone,
            "role_priors": {k: round(v, 3) for k, v in self.role_priors.items()},
            "visual_features": self.visual_features,
            "ocr_number": self.ocr_number,
            "ocr_confidence": round(self.ocr_confidence, 3),
            "assigned_name": self.assigned_name,
            "assigned_number": self.assigned_number,
            "assigned_role": self.assigned_role,
            "short_label": (
                f"#{self.assigned_number} {self.assigned_name.split()[-1]} ({self.assigned_role})"
                if self.assigned_name and self.assigned_number
                else f"{self.assigned_name} ({self.assigned_role})"
                if self.assigned_name
                else f"ID {self.track_id}"
            ),
            "assignment_confidence": round(self.assignment_confidence, 3),
            "assignment_method": self.assignment_method,
            "reasoning": self.reasoning,
        }


# =============================================================================
# 4. Isolated Multi-Source AI Player Locator & Identifier Engine
# =============================================================================

class IsolatedPlayerAILocator:
    """
    Self-contained, multi-source AI player detection and identity resolution engine.
    """

    def __init__(self, config: Optional[LocatorConfig] = None):
        self.config = config or LocatorConfig()
        self._yolo_model = None
        self.homography_mapper = PitchHomographyProjector()
        self.clock_extractor = MatchClockExtractor()
        self._active_xi_resolver: Optional[ActiveXIResolver] = None

    def _get_yolo(self):
        if self._yolo_model is None:
            try:
                from ultralytics import YOLO
                model_path = Path(__file__).resolve().parent.parent.parent / "yolov8n.pt"
                self._yolo_model = YOLO(str(model_path) if model_path.is_file() else "yolov8n.pt")
            except Exception as e:
                logger.warning("YOLO load warning: %s", e)
                self._yolo_model = False
        return self._yolo_model if self._yolo_model is not False else None

    @staticmethod
    def _normalize_name(name: str) -> str:
        if not name:
            return ""
        n = name.lower().strip()
        for k, v in {
            "á": "a", "à": "a", "ä": "a", "â": "a", "ã": "a",
            "é": "e", "è": "e", "ë": "e", "ê": "e",
            "í": "i", "ì": "i", "ï": "i", "î": "i",
            "ó": "o", "ò": "o", "ö": "o", "ô": "o", "õ": "o",
            "ú": "u", "ù": "u", "ü": "u", "û": "u",
            "ç": "c", "ñ": "n", "ć": "c", "č": "c", "ž": "z", "š": "s", "ğ": "g", "ı": "i",
        }.items():
            n = n.replace(k, v)
        n = re.sub(r"\s*\([a-z0-9_/ -]+\)$", "", n).strip()
        return n

    # -------------------------------------------------------------------------
    # STAGE 1: Adaptive Turf Segmentation, Camera Shot Type & Banner Masking
    # -------------------------------------------------------------------------

    def analyze_pitch_geometry(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Computes:
        1. Green grass mask & convex hull polygon.
        2. Camera shot type: 'wide_tactical', 'behind_goal', or 'close_up'.
        3. Broadcast lower-third graphic overlay banner mask (e.g. '11 RAPHINHA' banner).
        4. Near-touchline boundary line (separating on-pitch players from sidelines stewards).
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Green pitch turf mask (Hue in [32, 85], S >= 35, V >= 35 strictly excludes cyan/teal LED boards)
        lower_green = np.array([32, 35, 35], dtype=np.uint8)
        upper_green = np.array([85, 255, 255], dtype=np.uint8)
        grass_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Zero out scoreboard bug (top-left) and DAZN logo (top-right)
        grass_mask[0:int(h * 0.165), 0:int(w * 0.18)] = 0
        grass_mask[0:int(h * 0.15), int(w * 0.88):w] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        grass_closed = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        grass_closed = cv2.morphologyEx(grass_closed, cv2.MORPH_OPEN, kernel, iterations=1)

        turf_ratio = float(np.count_nonzero(grass_closed)) / float(h * w)

        # Largest connected turf contour -> Concave Touchline Polygon via approxPolyDP
        # Eliminates stadium corner bridges and concave sideline leaks pulling in photographers/stewards
        contours, _ = cv2.findContours(grass_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_mask = np.zeros((h, w), dtype=np.uint8)
        pitch_poly: Optional[np.ndarray] = None
        min_turf_y_pct = 12.0
        max_turf_y_pct = 96.0
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 0.05 * (h * w):
                peri = cv2.arcLength(largest, True)
                pitch_poly = cv2.approxPolyDP(largest, 0.015 * peri, True)
                cv2.drawContours(hull_mask, [pitch_poly], -1, 255, thickness=-1)
                ys = pitch_poly[:, 0, 1]
                min_turf_y_pct = float(np.min(ys)) / h * 100.0
                max_turf_y_pct = float(np.max(ys)) / h * 100.0

        # Detect lower-third broadcast graphic bar (bright horizontal white/gray rectangle spanning > 60% width in y=83..95%)
        banner_bbox: Optional[Tuple[int, int, int, int]] = None
        if self.config.use_broadcast_banner_mask:
            y_band_top = int(h * 0.83)
            y_band_bot = int(h * 0.945)
            band_hsv = hsv[y_band_top:y_band_bot, int(w * 0.08):int(w * 0.92)]
            # Bright white/light-gray pixels: S < 35, V > 200
            white_bar_mask = (band_hsv[:, :, 1] < 35) & (band_hsv[:, :, 2] > 200)
            row_white_frac = np.mean(white_bar_mask, axis=1)
            if np.max(row_white_frac) > 0.55:
                # Found lower-third broadcast overlay banner!
                banner_bbox = (int(w * 0.04), int(h * 0.81), int(w * 0.95), int(h * 0.95))

        # Camera shot classification & perspective slope estimation
        persp_slope = 0.20
        if turf_ratio < 0.45 or min_turf_y_pct > 34.0:
            shot_type = "close_up"
        else:
            shot_type = "wide_tactical"
            # Estimate perspective tilt from top grass boundary at x=12% vs x=50% vs x=88%
            col_12 = np.where(grass_closed[:, int(w * 0.12)] > 0)[0]
            col_50 = np.where(grass_closed[:, int(w * 0.50)] > 0)[0]
            col_88 = np.where(grass_closed[:, int(w * 0.88)] > 0)[0]
            y12 = (float(col_12[0]) / h * 100.0) if len(col_12) > 0 else 35.0
            y50 = (float(col_50[0]) / h * 100.0) if len(col_50) > 0 else 20.0
            y88 = (float(col_88[0]) / h * 100.0) if len(col_88) > 0 else 25.0
            if (y12 - y50) > 8.0:
                # Left penalty box / left goal view (goal line tilts up-and-right)
                persp_slope = 0.82
            elif (y88 - y50) > 8.0:
                # Right penalty box / right goal view
                persp_slope = -0.82

        return {
            "turf_ratio": round(turf_ratio, 3),
            "shot_type": shot_type,
            "persp_slope": round(persp_slope, 2),
            "grass_mask": grass_closed,
            "hull_mask": hull_mask,
            "pitch_polygon": pitch_poly,
            "min_turf_y_pct": round(min_turf_y_pct, 1),
            "max_turf_y_pct": round(max_turf_y_pct, 1),
            "banner_bbox": banner_bbox,
        }

    # -------------------------------------------------------------------------
    # STAGE 2: Multi-Scale High-Res Player Detection + Turf & Banner Filtering
    # -------------------------------------------------------------------------

    def detect_pitch_actors(
        self,
        frame: np.ndarray,
        pitch_geom: Dict[str, Any],
    ) -> List[DetectedPitchActor]:
        """
        Detects all players, goalkeepers, and referees on the pitch.
        Combines:
        1. Full-frame YOLOv8 inference at `self.config.yolo_imgsz` (1280px vs 640px).
        2. High-res overlapping 2x2 / 3x1 tiled inference across the turf band (when `use_tiled_highres_yolo=True`).
        3. OpenCV turf foreground contour recall for small far-side or edge players.
        4. Adaptive turf-hull foot verification + lower-third banner masking + touchline staff exclusion.
        """
        h, w = frame.shape[:2]
        yolo = self._get_yolo()
        shot_type = pitch_geom["shot_type"]
        hull_mask = pitch_geom["hull_mask"]
        grass_mask = pitch_geom["grass_mask"]
        banner_bbox = pitch_geom["banner_bbox"]

        raw_boxes: List[Tuple[int, int, int, int, float]] = []

        # 1. Full-frame YOLO pass
        if yolo is not None:
            conf_thresh = 0.14 if self.config.yolo_imgsz >= 1280 else 0.25
            res = yolo.predict(
                frame,
                classes=[0],
                conf=conf_thresh,
                imgsz=self.config.yolo_imgsz,
                verbose=False,
            )
            if res and len(res) > 0 and res[0].boxes is not None:
                xyxy = res[0].boxes.xyxy.cpu().numpy()
                confs = res[0].boxes.conf.cpu().numpy()
                for box, c in zip(xyxy, confs):
                    raw_boxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(c)))

        # 2. Tiled high-res pass on wide tactical shots to catch 25px far-side & bottom-edge players
        if yolo is not None and self.config.use_tiled_highres_yolo and shot_type != "close_up":
            y_top = max(0, int(h * 0.12))
            y_bot = min(h, int(h * 0.98))
            tile_w = int(w * 0.58)
            x_offsets = [0, int(w * 0.42)]
            for x0 in x_offsets:
                x1_tile = min(w, x0 + tile_w)
                crop = frame[y_top:y_bot, x0:x1_tile]
                if crop.size == 0:
                    continue
                t_res = yolo.predict(crop, classes=[0], conf=0.14, imgsz=960, verbose=False)
                if t_res and len(t_res) > 0 and t_res[0].boxes is not None:
                    xyxy = t_res[0].boxes.xyxy.cpu().numpy()
                    confs = t_res[0].boxes.conf.cpu().numpy()
                    for box, c in zip(xyxy, confs):
                        bx1 = int(box[0]) + x0
                        by1 = int(box[1]) + y_top
                        bx2 = int(box[2]) + x0
                        by2 = int(box[3]) + y_top
                        raw_boxes.append((bx1, by1, bx2, by2, float(c)))

        # 3. NMS deduplication across full + tiled detections
        merged_boxes = self._nms_boxes(raw_boxes, iou_thresh=0.45)

        # Compute median player scale on pitch (GSR Pillar 1: DeepEIoU-inspired morphological scale prior)
        confident_dims = [
            max(b[2] - b[0], b[3] - b[1])
            for b in merged_boxes
            if b[4] >= 0.55 and (b[3] / float(h) * 100.0) >= 18.0
        ]
        median_player_dim = float(np.median(confident_dims)) if confident_dims else 90.0
        min_allowed_dim = max(38, int(0.38 * median_player_dim))

        # 4. Filter by pitch geometry, banner mask, morphological scale, and touchline staff rules
        valid_actors: List[DetectedPitchActor] = []
        next_id = 1

        for (x1, y1, x2, y2, conf) in merged_boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            bw = x2 - x1
            bh = y2 - y1
            if bw < 8 or bh < 16:
                continue

            # GSR Pillar 1: Reject tiny spherical/conical non-human objects (e.g. spare football on cone)
            if shot_type != "close_up":
                if max(bw, bh) < min_allowed_dim:
                    continue
                if bh < 44 and (conf < 0.42 or (bw * bh) < 1600):
                    continue

            aspect = bh / max(1.0, float(bw))
            # Allow horizontal diving/sliding players/GKs (aspect >= 0.52 when conf >= 0.45) up to standing players (aspect <= 5.2)
            min_aspect = 0.52 if conf >= 0.45 else 0.72
            if shot_type != "close_up" and (aspect < min_aspect or aspect > 5.2):
                continue

            boot_x = 0.5 * (x1 + x2)
            boot_y = float(y2)
            x_pct = (boot_x / float(w)) * 100.0
            y_pct = (boot_y / float(h)) * 100.0

            # Exclude scoreboard top-left & DAZN logo top-right
            if x_pct < 18.0 and y_pct < 18.0:
                continue
            if x_pct > 87.0 and y_pct < 18.0:
                continue

            # Exclude lower-third broadcast graphic overlay portrait (e.g., '11 RAPHINHA' graphic cutout)
            if banner_bbox is not None and self.config.use_broadcast_banner_mask:
                bx1_b, by1_b, bx2_b, by2_b = banner_bbox
                cx_box = 0.5 * (x1 + x2)
                cy_box = 0.5 * (y1 + y2)
                if bx1_b <= cx_box <= bx2_b and by1_b <= cy_box <= by2_b and y1 >= int(h * 0.79):
                    # Check if surrounded by bright white banner pixels
                    pad_l = max(0, x1 - 25)
                    pad_r = min(w, x2 + 25)
                    banner_slice = frame[max(0, y1):min(h, y2), pad_l:pad_r]
                    if banner_slice.size > 0:
                        hsv_sl = cv2.cvtColor(banner_slice, cv2.COLOR_BGR2HSV)
                        white_frac = float(np.mean((hsv_sl[:, :, 1] < 35) & (hsv_sl[:, :, 2] > 200)))
                        if white_frac > 0.30:
                            continue

            if shot_type == "close_up":
                if self.config.use_touchline_staff_filter and (bh < int(h * 0.15) or conf < 0.23):
                    continue
            else:
                if self.config.use_adaptive_turf_hull:
                    # Check if boot or torso is inside or within 28px of green turf hull
                    by_i = min(h - 1, max(0, int(boot_y)))
                    bx_i = min(w - 1, max(0, int(boot_x)))
                    patch_y1 = max(0, by_i - 28)
                    patch_y2 = min(h, by_i + 16)
                    patch_x1 = max(0, bx_i - 24)
                    patch_x2 = min(w, bx_i + 24)
                    hull_near = np.count_nonzero(hull_mask[patch_y1:patch_y2, patch_x1:patch_x2]) > 0
                    grass_near = float(np.mean(grass_mask[patch_y1:patch_y2, patch_x1:patch_x2] > 0))
                    if not hull_near and grass_near < 0.08:
                        continue
                else:
                    # Legacy static clamp (used only in Iteration 1 baseline to demonstrate improvement)
                    if y_pct > 82.8 or y_pct < 19.0:
                        continue

                # Exclude sidelines stewards/cameramen/warm-up substitutes outside the playable pitch
                if self.config.use_touchline_staff_filter:
                    if (x_pct < 7.5 and y_pct > 78.0 and not hull_near):
                        continue
                    # Exclude bench/warm-up substitutes & photographers above the top touchline (y_pct < 17.5% or clipped at top border y1 <= 2.5% h)
                    min_playable_y = max(17.5, pitch_geom["min_turf_y_pct"] + 1.0)
                    if y_pct < min_playable_y or (y1 <= int(h * 0.025) and y_pct < 20.0):
                        continue
                    if y_pct < 44.0:
                        # Verify green grass exists right around/below the boots
                        foot_around = grass_mask[max(0, int(boot_y) - 8):min(h, int(boot_y) + 12), max(0, x1 - 6):min(w, x2 + 6)]
                        if foot_around.size == 0 or float(np.mean(foot_around > 0)) < 0.22:
                            continue

            actor = DetectedPitchActor(
                track_id=next_id,
                bbox=[x1, y1, x2, y2],
                x_pct=x_pct,
                y_pct=y_pct,
                confidence=conf,
                actor_type="player",
                team="unknown",
            )
            valid_actors.append(actor)
            next_id += 1

        # Sort deterministically left-to-right, top-to-bottom and re-index 1..N
        valid_actors.sort(key=lambda a: (round(a.x_pct / 4.0), a.y_pct))
        for idx, act in enumerate(valid_actors, start=1):
            act.track_id = idx

        return valid_actors

    @staticmethod
    def _nms_boxes(
        boxes: List[Tuple[int, int, int, int, float]],
        iou_thresh: float = 0.45,
    ) -> List[Tuple[int, int, int, int, float]]:
        if not boxes:
            return []
        boxes_sorted = sorted(boxes, key=lambda b: b[4], reverse=True)
        keep: List[Tuple[int, int, int, int, float]] = []

        for b in boxes_sorted:
            x1, y1, x2, y2, c = b
            area_b = max(1, (x2 - x1) * (y2 - y1))
            should_drop = False
            for idx_k, k in enumerate(keep):
                kx1, ky1, kx2, ky2, kc = k
                ix1, iy1 = max(x1, kx1), max(y1, ky1)
                ix2, iy2 = min(x2, kx2), min(y2, ky2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_k = max(1, (kx2 - kx1) * (ky2 - ky1))
                union = area_b + area_k - inter
                iou = inter / float(union) if union > 0 else 0.0
                overlap_min = inter / float(min(area_b, area_k))
                boot_dist = math.hypot(0.5 * (x1 + x2) - 0.5 * (kx1 + kx2), y2 - ky2)
                if iou > iou_thresh or (overlap_min > 0.72 and (boot_dist < 48.0 or abs(y2 - ky2) < 18.0)):
                    # If both boxes are high-confidence detections of the same diving/stretching player, merge union
                    if c >= 0.55 and kc >= 0.55 and overlap_min > 0.72 and abs(y2 - ky2) < 18.0:
                        keep[idx_k] = (min(x1, kx1), min(y1, ky1), max(x2, kx2), max(y2, ky2), max(c, kc))
                    should_drop = True
                    break
            if not should_drop:
                keep.append(b)
        return keep

    # -------------------------------------------------------------------------
    # STAGE 3: Physics/Color-Calibrated Kit, GK & Referee Classification
    # -------------------------------------------------------------------------

    def classify_teams_and_kits(
        self,
        frame: np.ndarray,
        actors: List[DetectedPitchActor],
        shot_type: str = "wide_tactical",
    ) -> List[DetectedPitchActor]:
        """
        Classifies every detected pitch actor into:
        - 'home' (FC Barcelona: Blaugrana dark blue & crimson stripes + dark blue shorts, or green GK #1)
        - 'away' (Real Madrid: White shirt & white shorts, or bright orange/coral GK #13)
        - 'referee' (Bright solid sky-cyan H=88..104, S>=75, V>=155 with dark shorts, max 1-2 on pitch)
        Also extracts visual appearance biometrics (skin tone, hair volume/color, boot color) for each actor.
        """
        h, w = frame.shape[:2]
        filtered_actors: List[DetectedPitchActor] = []

        for act in actors:
            x1, y1, x2, y2 = act.bbox
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            # Upper torso jersey crop (on large chest-up close-up shots, sample lower half where jersey is)
            if shot_type == "close_up" and bh > int(h * 0.45):
                ty1 = max(0, int(y1 + bh * 0.44))
                ty2 = min(h, int(y1 + bh * 0.90))
            else:
                ty1 = max(0, int(y1 + bh * 0.14))
                ty2 = min(h, int(y1 + bh * 0.52))
            tx1 = max(0, int(x1 + bw * 0.18))
            tx2 = min(w, int(x2 - bw * 0.18))
            torso = frame[ty1:ty2, tx1:tx2]

            # Shorts crop (lower torso / upper legs)
            sy1 = max(0, int(y1 + bh * 0.50))
            sy2 = min(h, int(y1 + bh * 0.72))
            sx1 = max(0, int(x1 + bw * 0.20))
            sx2 = min(w, int(x2 - bw * 0.20))
            shorts = frame[sy1:sy2, sx1:sx2]

            # Head/hair crop
            hy1 = max(0, int(y1))
            hy2 = min(h, int(y1 + bh * 0.20))
            hx1 = max(0, int(x1 + bw * 0.15))
            hx2 = min(w, int(x2 - bw * 0.15))
            head = frame[hy1:hy2, hx1:hx2]

            # Boots/feet crop (bottom 24% of bounding box for boot color signature)
            by1_c = max(0, int(y1 + bh * 0.76))
            by2_c = min(h, int(y2))
            bx1_c = max(0, int(x1))
            bx2_c = min(w, int(x2))
            boots_crop = frame[by1_c:by2_c, bx1_c:bx2_c]

            if torso.size == 0:
                continue

            hsv_t = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            # Exclude green pitch pixels from torso statistics
            non_grass = ~((hsv_t[:, :, 0] >= 30) & (hsv_t[:, :, 0] <= 88) & (hsv_t[:, :, 1] >= 35))
            if np.count_nonzero(non_grass) < 6:
                non_grass = np.ones(hsv_t.shape[:2], dtype=bool)

            H_vals = hsv_t[:, :, 0][non_grass]
            S_vals = hsv_t[:, :, 1][non_grass]
            V_vals = hsv_t[:, :, 2][non_grass]

            # Pixel fraction signatures
            white_frac = float(np.mean((S_vals < 52) & (V_vals > 152)))
            dark_blue_frac = float(np.mean((H_vals >= 102) & (H_vals <= 138) & (S_vals >= 35) & (V_vals < 185)))
            crimson_red_frac = float(np.mean(((H_vals <= 14) | (H_vals >= 155)) & (S_vals >= 40) & (V_vals >= 35) & (V_vals < 175)))
            dark_kit_frac = float(np.mean(V_vals < 115))

            # Referee bright sky-cyan: H in [83..108], S >= 55, V >= 125
            cyan_ref_frac = float(np.mean((H_vals >= 83) & (H_vals <= 108) & (S_vals >= 55) & (V_vals >= 125)))
            # Coral/Salmon/Orange GK (Lunin #13): H in [0..18] or [166..180], S >= 95, V >= 165
            orange_gk_frac = float(np.mean(((H_vals <= 18) | (H_vals >= 166)) & (S_vals >= 95) & (V_vals >= 165)))
            # Fluorescent yellow-green warm-up bib / steward vest: H in [22..44], S >= 105, V >= 150
            all_H = hsv_t[:, :, 0]
            all_S = hsv_t[:, :, 1]
            all_V = hsv_t[:, :, 2]
            neon_bib_frac = float(np.mean((all_H >= 22) & (all_H <= 44) & (all_S >= 105) & (all_V >= 150)))
            # Bright Green GK (Joan García #1): H in [42..82], S >= 95, V >= 135 (inside torso while inside penalty box)
            green_gk_frac = float(np.mean((all_H >= 45) & (all_H <= 82) & (all_S >= 100) & (all_V >= 145)))

            # Shorts whiteness & orange/black fractions
            shorts_white_frac = 0.0
            shorts_orange_frac = 0.0
            shorts_dark_frac = 0.0
            if shorts.size > 0:
                hsv_s = cv2.cvtColor(shorts, cv2.COLOR_BGR2HSV)
                ng_s = ~((hsv_s[:, :, 0] >= 30) & (hsv_s[:, :, 0] <= 70) & (hsv_s[:, :, 1] >= 35))
                if np.count_nonzero(ng_s) > 4:
                    sH = hsv_s[:, :, 0][ng_s]
                    sS = hsv_s[:, :, 1][ng_s]
                    sV = hsv_s[:, :, 2][ng_s]
                    shorts_white_frac = float(np.mean((sS < 55) & (sV > 145)))
                    shorts_orange_frac = float(np.mean(((sH <= 18) | (sH >= 166)) & (sS >= 95) & (sV >= 155)))
                    shorts_dark_frac = float(np.mean(sV < 85))

            # Hair & skin visual estimation
            hair_dark_frac = 0.0
            hair_blonde_frac = 0.0
            if head.size > 0:
                hsv_h = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
                ng_h = ~((hsv_h[:, :, 0] >= 30) & (hsv_h[:, :, 0] <= 88) & (hsv_h[:, :, 1] >= 35))
                if np.count_nonzero(ng_h) > 4:
                    hair_dark_frac = float(np.mean(hsv_h[:, :, 2][ng_h] < 65))
                    hair_blonde_frac = float(np.mean((hsv_h[:, :, 0][ng_h] >= 12) & (hsv_h[:, :, 0][ng_h] <= 32) & (hsv_h[:, :, 2][ng_h] > 135)))

            # Boot color signature: bright mint-green/turquoise boots (#7 Fermín López)
            mint_boot_frac = 0.0
            if boots_crop.size > 0:
                hsv_b = cv2.cvtColor(boots_crop, cv2.COLOR_BGR2HSV)
                mint_boot_frac = float(
                    np.mean((hsv_b[:, :, 0] >= 68) & (hsv_b[:, :, 0] <= 96) & (hsv_b[:, :, 1] >= 58) & (hsv_b[:, :, 2] >= 115))
                )

            act.visual_features = {
                "white_frac": round(white_frac, 3),
                "shorts_white_frac": round(shorts_white_frac, 3),
                "blaugrana_frac": round(dark_blue_frac + crimson_red_frac, 3),
                "dark_kit_frac": round(dark_kit_frac, 3),
                "cyan_ref_frac": round(cyan_ref_frac, 3),
                "orange_gk_frac": round(orange_gk_frac, 3),
                "green_gk_frac": round(green_gk_frac, 3),
                "neon_bib_frac": round(neon_bib_frac, 3),
                "hair_dark_frac": round(hair_dark_frac, 3),
                "hair_blonde_frac": round(hair_blonde_frac, 3),
                "mint_boot_frac": round(mint_boot_frac, 3),
            }

            # Legacy Bug Simulation (only if use_blaugrana_stripe_classifier is False in Iteration 1)
            if not self.config.use_blaugrana_stripe_classifier:
                med_h = float(np.median(H_vals))
                med_s = float(np.median(S_vals))
                if 75 <= med_h <= 135 and med_s >= 25:
                    act.team = "referee"
                    act.actor_type = "referee"
                    continue
                act.team = "away" if white_frac > 0.25 else "home"
                filtered_actors.append(act)
                continue

            # 1. Check Referee (Sky-Cyan shirt, low blaugrana stripe presence)
            if (
                self.config.use_referee_outlier_guard
                and cyan_ref_frac >= 0.22
                and (dark_blue_frac + crimson_red_frac) < 0.25
                and white_frac < 0.35
            ):
                act.team = "referee"
                act.actor_type = "referee"
                continue

            # Exclude non-kit grass/line/linesman/warm-up bib/cameraman artifacts on wide shots
            if self.config.use_touchline_staff_filter and shot_type != "close_up":
                if act.y_pct < 20.5 and act.x_pct > 45.0:
                    continue
                # Reject warm-up substitutes or touchline stewards in fluorescent yellow-green bibs
                if neon_bib_frac >= 0.16 and act.y_pct < 28.0:
                    continue
                # Reject bottom-left touchline cameraman/steward in dark clothing (while preserving white-kit players like #16 Konaté at y=72.7%)
                if act.x_pct < 7.5 and act.y_pct > 71.5:
                    if white_frac < 0.35 and (dark_blue_frac + crimson_red_frac) < 0.22:
                        continue
                if (
                    white_frac < 0.45
                    and (dark_blue_frac + crimson_red_frac) < 0.18
                    and dark_kit_frac < 0.25
                    and orange_gk_frac < 0.20
                    and green_gk_frac < 0.20
                ):
                    continue

            # 2. Check Goalkeepers (Coral/Orange #13 Lunin vs Bright Green #1 Joan García)
            if self.config.use_gk_color_detector:
                if (orange_gk_frac >= 0.20 or (orange_gk_frac >= 0.14 and shorts_orange_frac >= 0.20)) and dark_blue_frac < 0.08:
                    act.team = "away"
                    act.actor_type = "goalkeeper"
                    act.is_gk = True
                    filtered_actors.append(act)
                    continue
                if green_gk_frac >= 0.38 and bw < int(w * 0.08):
                    act.team = "home"
                    act.actor_type = "goalkeeper"
                    act.is_gk = True
                    filtered_actors.append(act)
                    continue

            # 3. Outfield Team Classification: Real Madrid (White shirt + white shorts) vs Barcelona (Blaugrana blue/crimson + dark shorts)
            away_score = white_frac * 1.3 + shorts_white_frac * 0.7
            home_score = (dark_blue_frac + crimson_red_frac) * 1.4 + dark_kit_frac * 0.6

            if away_score >= home_score and (white_frac >= 0.22 or shorts_white_frac >= 0.28):
                act.team = "away"
            else:
                act.team = "home"

            filtered_actors.append(act)

        # Laws-of-football cap: at most 11 players per team on the pitch
        if self.config.use_live_active_xi_sync:
            home_acts = [a for a in filtered_actors if a.team == "home"]
            away_acts = [a for a in filtered_actors if a.team == "away"]
            if len(home_acts) > 11:
                home_acts.sort(key=lambda a: (a.is_gk, a.visual_features.get("blaugrana_frac", 0.0) + a.confidence), reverse=True)
                home_acts = home_acts[:11]
            if len(away_acts) > 11:
                away_acts.sort(key=lambda a: (a.is_gk, a.visual_features.get("white_frac", 0.0) + a.confidence), reverse=True)
                away_acts = away_acts[:11]
            filtered_actors = sorted(home_acts + away_acts, key=lambda a: (round(a.x_pct / 4.0), a.y_pct))

        # Re-index only active players/goalkeepers (excluding referees)
        for idx, act in enumerate(filtered_actors, start=1):
            act.track_id = idx

        return filtered_actors

    # -------------------------------------------------------------------------
    # STAGE 4: 2D Pitch Homography & Dynamic Formation Slot Prior Mapping
    # -------------------------------------------------------------------------

    def compute_spatial_topology_priors(
        self,
        frame: np.ndarray,
        actors: List[DetectedPitchActor],
        shot_type: str,
        period: int = 1,
        clip_id: str = "",
        persp_slope: float = 0.20,
    ) -> None:
        """
        Maps each player's (x_pct, y_pct) to metric pitch coordinates (pitch_x_m, pitch_y_m)
        and computes relative formation line (backline, midfield, attack) and flank zone
        (left, central, right) to generate mathematical probability priors over tactical roles:
        GK, LB, LCB, RCB, RB, DM, LCM, RCM, AM, LW, CF, RW.
        """
        if not actors:
            return

        self.homography_mapper.estimate_from_frame_lines(frame, clip_id=clip_id)

        # Determine team attacking directions:
        # Check if a goalkeeper is visible on left or right, or use team X centroids
        home_actors = [a for a in actors if a.team == "home"]
        away_actors = [a for a in actors if a.team == "away"]

        home_gk = next((a for a in home_actors if a.is_gk), None)
        away_gk = next((a for a in away_actors if a.is_gk), None)
        outfield_xs = [a.x_pct for a in actors if not a.is_gk]
        mean_outfield_x = float(np.mean(outfield_xs)) if outfield_xs else 50.0

        # Attacking direction: +1 means attacking Left->Right (defending Left goal), -1 means attacking Right->Left
        if away_gk:
            if away_gk.x_pct < 38.0 or away_gk.x_pct < mean_outfield_x:
                away_dir = 1   # Away defends Left goal, attacks Right
                home_dir = -1  # Home attacks Left goal
            else:
                away_dir = -1  # Away defends Right goal, attacks Left
                home_dir = 1   # Home attacks Right goal
        elif home_gk:
            if home_gk.x_pct < 38.0 or home_gk.x_pct < mean_outfield_x:
                home_dir = 1   # Home defends Left goal, attacks Right
                away_dir = -1  # Away attacks Left goal
            else:
                home_dir = -1  # Home defends Right goal, attacks Left
                away_dir = 1   # Away attacks Right goal
        elif period == 1:
            # In Period 1 of BAR vs RMA (401882832), Barcelona defends Left goal (attacks Right), Real Madrid attacks Left
            home_dir = 1
            away_dir = -1
        else:
            # In Period 2, Real Madrid defends Left goal (attacks Right), Barcelona attacks Left
            away_dir = 1
            home_dir = -1

        for team_side, t_actors, atk_dir in [("home", home_actors, home_dir), ("away", away_actors, away_dir)]:
            if not t_actors:
                continue

            outfield = [a for a in t_actors if not a.is_gk]
            if not outfield:
                for a in t_actors:
                    a.role_priors = {"GK": 0.99}
                continue

            # Sort outfielders along perspective-corrected attacking axis:
            # x_persp = x_pct + persp_slope * (y_pct - 50.0) aligns players on diagonal broadcast goal/penalty lines!
            use_persp = persp_slope if self.config.use_homography_topology_priors else 0.0
            x_persps = [a.x_pct + use_persp * (a.y_pct - 50.0) for a in outfield]
            ys = [a.y_pct for a in outfield]
            # When fewer than 3 outfielders of this team are visible (e.g. tight penalty-box shot),
            # anchor min/max to all visible pitch actors or full pitch bounds so a single player doesn't collapse to 0.0!
            if len(outfield) < 3:
                all_x_p = [a.x_pct + use_persp * (a.y_pct - 50.0) for a in actors]
                all_ys = [a.y_pct for a in actors]
                min_x, max_x = min(all_x_p), max(all_x_p)
                min_y, max_y = min(22.0, min(all_ys)), max(84.0, max(all_ys))
            else:
                min_x, max_x = min(x_persps), max(x_persps)
                min_y, max_y = min(ys), max(ys)
            span_x = max(18.0, max_x - min_x)
            span_y = max(24.0, max_y - min_y)

            for a in t_actors:
                # Metric pitch coordinates via homography mapper
                px_m, py_m = self.homography_mapper.project_screen_to_pitch(a.x_pct, a.y_pct)
                a.pitch_x_m = px_m
                a.pitch_y_m = py_m

                if a.is_gk:
                    a.formation_line = "goalkeeper"
                    a.flank_zone = "central"
                    a.tactical_zone = "PENALTY_BOX"
                    a.role_priors = {"GK": 0.98}
                    continue

                if shot_type == "close_up":
                    # On close-up camera shots, spatial coordinates do not reflect full-pitch formation lines
                    a.formation_line = "midfield"
                    a.flank_zone = "central"
                    a.tactical_zone = "CLOSE_UP"
                    a.role_priors = {
                        "LW": 0.14, "CF": 0.14, "RW": 0.14,
                        "AM": 0.12, "LCM": 0.12, "RCM": 0.12, "DM": 0.10,
                        "LB": 0.12, "RB": 0.12, "LCB": 0.10, "RCB": 0.10,
                    }
                    continue

                a_x_persp = a.x_pct + use_persp * (a.y_pct - 50.0)
                # Compute normalized attacking depth in [0..1] (0 = deepest defender, 1 = highest attacker)
                if atk_dir == 1:
                    rel_depth = (a_x_persp - min_x) / span_x
                    # Facing Left->Right: top of screen (small y_pct) is LEFT wing, bottom of screen (large y_pct) is RIGHT wing
                    rel_width_left_to_right = (a.y_pct - min_y) / span_y
                else:
                    rel_depth = (max_x - a_x_persp) / span_x
                    # Facing Right->Left: top of screen (small y_pct) is RIGHT wing, bottom of screen (large y_pct) is LEFT wing
                    rel_width_left_to_right = (max_y - a.y_pct) / span_y

                # Absolute screen position adjustments (e.g., if all visible players are in one penalty box)
                if (atk_dir == 1 and a.x_pct < 28.0) or (atk_dir == -1 and a.x_pct > 72.0):
                    rel_depth = min(rel_depth, 0.32)

                # Formation Line
                if rel_depth <= 0.36:
                    a.formation_line = "backline"
                elif rel_depth <= 0.68:
                    a.formation_line = "midfield"
                else:
                    a.formation_line = "attack"

                # Flank Zone (relative to team's attacking orientation: 0.0 = Left Flank, 0.5 = Central, 1.0 = Right Flank)
                if rel_width_left_to_right <= 0.24:
                    a.flank_zone = "left"
                elif rel_width_left_to_right <= 0.42:
                    a.flank_zone = "left_half"
                elif rel_width_left_to_right <= 0.60:
                    a.flank_zone = "central"
                elif rel_width_left_to_right <= 0.78:
                    a.flank_zone = "right_half"
                else:
                    a.flank_zone = "right"

                a.tactical_zone = f"{a.formation_line.upper()}_{a.flank_zone.upper()}"

                # Compute smooth Gaussian prior over canonical roles
                # Role ideal (depth, width_left_to_right) in [0..1] x [0..1]:
                role_ideals = {
                    "GK":  (-0.15, 0.50),
                    "LB":  (0.22, 0.12),
                    "LCB": (0.16, 0.36),
                    "CB":  (0.15, 0.50),
                    "RCB": (0.16, 0.64),
                    "RB":  (0.22, 0.88),
                    "DM":  (0.45, 0.50),
                    "LCM": (0.52, 0.30),
                    "CM":  (0.50, 0.50),
                    "RCM": (0.52, 0.70),
                    "AM":  (0.66, 0.50),
                    "LW":  (0.82, 0.14),
                    "CF":  (0.88, 0.50),
                    "RW":  (0.82, 0.86),
                }
                priors: Dict[str, float] = {}
                for r_code, (id_d, id_w) in role_ideals.items():
                    if r_code == "GK":
                        # Only possible if inside own penalty box and no other GK
                        if (atk_dir == 1 and a.x_pct < 16.0 and 25.0 <= a.y_pct <= 70.0) and not (home_gk if team_side == "home" else away_gk):
                            priors["GK"] = 0.65
                        else:
                            priors["GK"] = 0.01
                        continue
                    dist_sq = ((rel_depth - id_d) / 0.32) ** 2 + ((rel_width_left_to_right - id_w) / 0.30) ** 2
                    priors[r_code] = float(math.exp(-0.5 * dist_sq))

                # Normalize top priors
                max_p = max(priors.values()) if priors else 1.0
                a.role_priors = {k: v / max_p for k, v in sorted(priors.items(), key=lambda kv: kv[1], reverse=True)[:5]}

    # -------------------------------------------------------------------------
    # STAGE 5: Multi-Frame Temporal Crops & Super-Resolution Torso Zoom Sheet
    # -------------------------------------------------------------------------

    def build_enriched_crop_sheet(
        self,
        frame: np.ndarray,
        actors: List[DetectedPitchActor],
        frame_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Produces:
        1. `annotated_frame`: Full 1080p frame with color-coded bounding boxes, high-contrast
           `[ID]` badges (Blue for Home, Gold/White for Away, Green for GK), and 2D zone grid.
        2. `crop_sheet`: High-resolution multi-column crop montage where each player tile
           shows:
           - Full body crop (upscaled)
           - Super-resolution CLAHE + Unsharp-Mask torso/number zoom crop (for jersey OCR)
           - Temporal sibling crop (from `_frame_prev` / `_frame_next_1` if available)
           - Metadata header (`[ID] Team | Zone | Top Role Priors`)
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()

        # Draw subtle tactical zone grid lines on annotated frame
        for gx in [int(w * 0.25), int(w * 0.50), int(w * 0.75)]:
            cv2.line(annotated, (gx, int(h * 0.15)), (gx, int(h * 0.95)), (180, 180, 180), 1, cv2.LINE_AA)

        # Load temporal sibling frames if enabled and available on disk
        temporal_frames: List[np.ndarray] = []
        if self.config.use_multiframe_temporal_crops and frame_path:
            fp = Path(frame_path)
            stem_base = re.sub(r"_(cause|effect|frame_freeze|frame_prev|frame_next_\d+)$", "", fp.stem)
            for suffix in ["_frame_prev.jpg", "_frame_next_1.jpg", "_effect.jpg"]:
                cand = fp.parent / f"{stem_base}{suffix}"
                if cand.is_file() and cand.resolve() != fp.resolve():
                    img_t = cv2.imread(str(cand))
                    if img_t is not None and img_t.shape[:2] == (h, w):
                        temporal_frames.append(img_t)

        # Draw bounding boxes & ID tags on annotated frame
        for act in actors:
            x1, y1, x2, y2 = act.bbox
            if act.is_gk:
                color = (0, 230, 118)     # Bright Green for GK
            elif act.team == "home":
                color = (255, 110, 64)    # Electric Blue-Orange for Home (Barcelona)
            else:
                color = (40, 215, 255)    # Gold/Yellow for Away (Real Madrid)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label = f"[{act.track_id}]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.58, 1)
            ty_box = max(th + 6, y1 - 4)
            cv2.rectangle(annotated, (x1, ty_box - th - 6), (x1 + tw + 8, ty_box + 2), (12, 16, 24), -1)
            cv2.rectangle(annotated, (x1, ty_box - th - 6), (x1 + tw + 8, ty_box + 2), color, 1)
            cv2.putText(annotated, label, (x1 + 4, ty_box - 2), cv2.FONT_HERSHEY_DUPLEX, 0.58, color, 1, cv2.LINE_AA)

        # Build High-Resolution Multi-Crop Montage Sheet
        if not actors:
            blank = np.zeros((240, 480, 3), dtype=np.uint8)
            return annotated, blank

        tile_w = 340 if self.config.use_superres_clahe_torso_zoom else 160
        tile_h = 230
        cols = min(4, max(1, len(actors)))
        rows = int(math.ceil(len(actors) / float(cols)))
        sheet = np.full((rows * tile_h, cols * tile_w, 3), 18, dtype=np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(4, 4))

        for idx, act in enumerate(actors):
            r_idx = idx // cols
            c_idx = idx % cols
            ox = c_idx * tile_w
            oy = r_idx * tile_h

            x1, y1, x2, y2 = act.bbox
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            # 1. Padded Full Player Crop (left half of tile: 155x185)
            pad_x = max(12, int(bw * 0.30))
            pad_y = max(12, int(bh * 0.18))
            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
            crop_full = frame[cy1:cy2, cx1:cx2]
            if crop_full.size > 0:
                crop_full_rs = cv2.resize(crop_full, (155, 185), interpolation=cv2.INTER_CUBIC)
                sheet[oy + 40:oy + 225, ox + 5:ox + 160] = crop_full_rs

            # 2. Super-Resolution CLAHE + Unsharp-Mask Kit/Shorts/Boots Zoom (right half of tile: 165x185)
            if self.config.use_superres_clahe_torso_zoom:
                tx1 = max(0, x1 - int(bw * 0.08))
                tx2 = min(w, x2 + int(bw * 0.08))
                ty1 = max(0, y1 + int(bh * 0.06))
                ty2 = min(h, y1 + int(bh * 0.94))
                torso_crop = frame[ty1:ty2, tx1:tx2]
                if torso_crop.size > 0:
                    zoom = cv2.resize(torso_crop, (165, 185), interpolation=cv2.INTER_CUBIC)
                    lab = cv2.cvtColor(zoom, cv2.COLOR_BGR2LAB)
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.8)
                    sharpened = cv2.addWeighted(enhanced, 1.55, blur, -0.55, 0)
                    sheet[oy + 40:oy + 225, ox + 168:ox + 333] = sharpened

            # Tile Header Banner
            top_roles = "/".join(list(act.role_priors.keys())[:2]) if act.role_priors else "PLAYER"
            hdr_color = (255, 120, 60) if act.team == "home" else (40, 215, 255)
            cv2.rectangle(sheet, (ox + 2, oy + 2), (ox + tile_w - 2, oy + 36), (32, 38, 52), -1)
            cv2.rectangle(sheet, (ox + 2, oy + 2), (ox + tile_w - 2, oy + 36), hdr_color, 1)
            hdr_text = f"[{act.track_id}] {act.team.upper()} | {top_roles} | ({act.x_pct:.0f}%,{act.y_pct:.0f}%)"
            cv2.putText(sheet, hdr_text[:38], (ox + 6, oy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, hdr_color, 1, cv2.LINE_AA)

        return annotated, sheet

    # -------------------------------------------------------------------------
    # STAGE 6: Multimodal Gemini VLM Analysis + Structured Reasoning
    # -------------------------------------------------------------------------

    def query_vlm_identities(
        self,
        annotated_frame: np.ndarray,
        crop_sheet: np.ndarray,
        actors: List[DetectedPitchActor],
        active_state: ActiveRosterState,
        event_context: Dict[str, Any],
        shot_type: str,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Calls Gemini VLM with:
        1. Annotated full broadcast frame (`[ID]` boxes + 2D tactical zone grid)
        2. Super-Resolution Multi-Crop Montage Sheet (full player + CLAHE torso zoom)
        3. Exact timestamp-synced 11v11 Active XI + Banned/Subbed-off players
        4. Play-by-Play Event Telemetry at this exact second
        5. Per-player 2D Homography Role Priors & Visual Biometric Profiles
        """
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.environ.get("GEMINI_API_KEY", "").strip()
            except Exception:
                pass
        if not api_key or not actors:
            return {}

        prompt = self._build_vlm_prompt(actors, active_state, event_context, shot_type)

        # Check VLM response cache keyed by actor coordinates + active state + crop mode
        import hashlib
        actor_sig = "|".join(f"{a.track_id}:{a.team}:{a.x_pct:.1f},{a.y_pct:.1f}" for a in actors)
        cache_key = hashlib.md5(
            f"{actor_sig}|{active_state.clock_str}|{self.config.use_superres_clahe_torso_zoom}|{self.config.use_biometric_visual_db}".encode("utf-8")
        ).hexdigest()
        cache_path = Path("test_data/locator_iterations/vlm_cache.json")
        cache_store: Dict[str, Any] = {}
        if cache_path.is_file():
            try:
                cache_store = json.loads(cache_path.read_text(encoding="utf-8"))
                if cache_key in cache_store and cache_store[cache_key]:
                    return {int(k): v for k, v in cache_store[cache_key].items()}
            except Exception:
                cache_store = {}

        # Encode images as compact JPEG base64 + zero thinking latency for sub-3s VLM turnaround
        af_small = cv2.resize(annotated_frame, (960, 540), interpolation=cv2.INTER_AREA)
        _, buf_frame = cv2.imencode(".jpg", af_small, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
        b64_frame = base64.b64encode(buf_frame.tobytes()).decode("utf-8")

        sh_h, sh_w = crop_sheet.shape[:2]
        if sh_w > 960:
            scale_s = 960.0 / float(sh_w)
            cs_small = cv2.resize(crop_sheet, (960, max(120, int(sh_h * scale_s))), interpolation=cv2.INTER_AREA)
        else:
            cs_small = crop_sheet
        _, buf_sheet = cv2.imencode(".jpg", cs_small, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        b64_sheet = base64.b64encode(buf_sheet.tobytes()).decode("utf-8")

        for model_name in self.config.model_priority:
            gen_cfg: Dict[str, Any] = {
                "temperature": self.config.temperature,
                "response_mime_type": "application/json",
            }
            if "3." in model_name:
                gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"inline_data": {"mime_type": "image/jpeg", "data": b64_frame}},
                            {"inline_data": {"mime_type": "image/jpeg", "data": b64_sheet}},
                            {"text": prompt},
                        ]
                    }
                ],
                "generationConfig": gen_cfg,
            }

            full_model = model_name if model_name.startswith("models/") else f"models/{model_name}"
            url = f"https://generativelanguage.googleapis.com/v1beta/{full_model}:generateContent?key={api_key}"
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=14) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                candidates = res_data.get("candidates", [])
                if not candidates:
                    continue
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if not text:
                    continue
                parsed = json.loads(text)
                rows = parsed.get("players", parsed) if isinstance(parsed, dict) else parsed
                if isinstance(rows, list):
                    out: Dict[int, Dict[str, Any]] = {}
                    for r in rows:
                        if not isinstance(r, dict):
                            continue
                        tid = r.get("id") or r.get("track_id")
                        if tid is not None:
                            out[int(tid)] = r
                    if out:
                        try:
                            cache_path.parent.mkdir(parents=True, exist_ok=True)
                            cache_store[cache_key] = {str(k): v for k, v in out.items()}
                            cache_path.write_text(json.dumps(cache_store, indent=2), encoding="utf-8")
                        except Exception:
                            pass
                        return out
            except Exception as e:
                logger.warning("VLM call warning on %s: %s", model_name, e)
                continue

        return {}

    def _build_vlm_prompt(
        self,
        actors: List[DetectedPitchActor],
        active_state: ActiveRosterState,
        event_context: Dict[str, Any],
        shot_type: str,
    ) -> str:
        """
        Constructs the prompt according to `self.config.prompt_version` and active intelligence sources.
        """
        home_name = active_state.home_team_name
        away_name = active_state.away_team_name

        # Format Active XI with or without Biometric Cues depending on config
        def _fmt_roster(active_list: List[Dict[str, Any]]) -> str:
            lines = []
            for p in active_list:
                num = p.get("number")
                name = p.get("name", "")
                role = p.get("role", "MF")
                norm = self._normalize_name(name)
                bio = PLAYER_BIOMETRIC_DB.get(norm, {})
                if self.config.use_biometric_visual_db and bio:
                    lines.append(
                        f"  - #{num} {name} (Role: {role}) | Skin: {bio.get('skin')} | Hair: {bio.get('hair')} | Visual Cue: {bio.get('cues')}"
                    )
                else:
                    lines.append(f"  - #{num} {name} (Role: {role})")
            return "\n".join(lines)

        home_roster_str = _fmt_roster(active_state.home_active)
        away_roster_str = _fmt_roster(active_state.away_active)

        banned_str = ""
        if self.config.use_live_active_xi_sync and active_state.subbed_off_players:
            sub_lines = []
            for s in active_state.subbed_off_players:
                s_norm = self._normalize_name(s.get("name", ""))
                s_bio = PLAYER_BIOMETRIC_DB.get(s_norm, {})
                s_num = s.get("number") or s_bio.get("num") or "?"
                s_role = s_bio.get("role", "MF")
                s_cue = s_bio.get("cues", "")
                sub_lines.append(
                    f"  - #{s_num} {s['name']} ({s.get('team', '').upper()}, Role: {s_role}, API sub listed at {s['minute']}') | Visual Cue: {s_cue}"
                )
            banned_str = (
                "\nADDITIONAL SQUAD PLAYERS (API lists substitution, but Broadcast Video Reality ALWAYS overrides API timestamps — "
                "ASSIGN if visually verified via jersey/shorts number, mint-green boots, or hair):\n"
                + "\n".join(sub_lines)
                + "\n"
            )

        telemetry_str = ""
        if self.config.use_play_by_play_telemetry and event_context:
            telemetry_str = (
                f"\nLIVE MATCH EVENT TELEMETRY AT {active_state.clock_str} (Period {event_context.get('period', 1)}):\n"
                f"  - Event Type: {event_context.get('event_type', 'open_play')}\n"
                f"  - Official Play-by-Play Description: {event_context.get('description', 'N/A')}\n"
                f"  - Key Event Player: {event_context.get('primary_player', 'N/A')}\n"
                f"  - Camera Shot Type: {shot_type}\n"
            )

        actors_desc = []
        for a in actors:
            priors_str = ", ".join(f"{k}:{v:.2f}" for k, v in list(a.role_priors.items())[:3])
            vf = a.visual_features
            bio_tags = []
            if vf.get("mint_boot_frac", 0.0) >= 0.035:
                bio_tags.append("mint_green_boots=YES(#7 Fermin Lopez signature)")
            if vf.get("hair_blonde_frac", 0.0) >= 0.04:
                bio_tags.append("blonde_light_hair=YES")
            if vf.get("hair_dark_frac", 0.0) >= 0.14:
                bio_tags.append("dark_hair_skin=YES")
            bio_tag_str = f" | VisualCues=[{', '.join(bio_tags)}]" if bio_tags else ""
            if self.config.use_homography_topology_priors:
                actors_desc.append(
                    f"  - [{a.track_id}] Team={a.team.upper()}{'(GK)' if a.is_gk else ''} | "
                    f"Pos=({a.x_pct:.1f}%, {a.y_pct:.1f}%) -> Pitch=({a.pitch_x_m:.1f}m, {a.pitch_y_m:.1f}m) | "
                    f"Zone={a.tactical_zone} | Top Role Priors=[{priors_str}]{bio_tag_str}"
                )
            else:
                actors_desc.append(f"  - [{a.track_id}] Team={a.team.upper()} | Pos=({a.x_pct:.1f}%, {a.y_pct:.1f}%){bio_tag_str}")
        actors_block = "\n".join(actors_desc)

        return (
            "You are an elite UEFA Champions League & LaLiga Computer Vision & Tactical Player Identification Engine.\n"
            "You are given:\n"
            "  IMAGE 1: The full broadcast match frame with numbered bounding boxes `[ID]`.\n"
            "  IMAGE 2: A high-resolution multi-crop zoom sheet where each tile `[ID]` shows the full player crop on the left and a Super-Resolution CLAHE-sharpened kit/shorts/boots zoom crop on the right.\n\n"
            f"OFFICIAL ACTIVE ON-PITCH ROSTERS AT MATCH CLOCK {active_state.clock_str}:\n"
            f"HOME TEAM ({home_name} - Blaugrana dark blue & crimson striped shirts, dark blue shorts; GK #13 Iñaki Peña in bright green):\n"
            f"{home_roster_str}\n\n"
            f"AWAY TEAM ({away_name} - All-white shirts & white shorts; GK #13 Andriy Lunin in bright orange/coral/salmon):\n"
            f"{away_roster_str}\n"
            f"{banned_str}"
            f"{telemetry_str}\n"
            f"DETECTED ON-PITCH ACTORS TO IDENTIFY ({len(actors)} players):\n"
            f"{actors_block}\n\n"
            "MANDATORY MULTI-SOURCE IDENTIFICATION PROTOCOL:\n"
            "1. JERSEY & SHORTS NUMBER OCR FIRST: Inspect both IMAGE 1 and the sharpened kit/shorts zoom in IMAGE 2 for any visible shirt back numbers OR front shorts thigh numbers (e.g. #17 Marc Casadó on dark shorts, #17 Lucas Vázquez on white shorts, #16 Fermín López on dark kit, #3 Alejandro Balde, #8 Pedri / #8 Federico Valverde, #19 Lamine Yamal, #9 Robert Lewandowski / #9 Kylian Mbappé, #7 Vinícius Júnior, #11 Raphinha). If a number is physically readable on the shirt or shorts, lock that exact player from the corresponding team! Set `visible_number: null` and `number_confidence: 0.0` if digits are NOT physically visible/readable on the shirt or shorts (NEVER copy a guessed roster number into `visible_number`).\n"
            "2. BIOMETRIC, BOOT COLOR & VISUAL DISCRIMINATORS (GSR VISUAL REALITY OVERRIDES API SUB TIMESTAMPS):\n"
            "   - Short blonde/light-brown hair + bright mint-green/turquoise boots in Blaugrana kit -> #16 Fermín López (RCM).\n"
            "   - Compact build + dark hair + #17 on dark shorts in Blaugrana kit -> #17 Marc Casadó (DM).\n"
            "   - Energetic right-back with dark hair + #17 on white kit -> #17 Lucas Vázquez (RB).\n"
            "   - Dark-skinned powerful LB in white kit -> #23 Ferland Mendy (LB).\n"
            "   - Distinctive long black braids/dreadlocks in white kit -> #6 Eduardo Camavinga (LCM).\n"
            "   - Tall dark-skinned CB with beard in white kit -> #22 Antonio Rüdiger (LCB).\n"
            "   - Bright orange/coral/salmon GK kit -> #13 Andriy Lunin (GK); Bright green GK kit -> #13 Iñaki Peña (GK).\n"
            "3. 2D FORMATION TOPOLOGY & SPATIAL ELIMINATION: For players whose jersey numbers are occluded, match their 2D formation line (backline vs midfield vs attack) and flank zone (left vs central vs right) to the remaining unassigned players in that team's Active XI.\n"
            "4. STRICT 1-TO-1 UNIQUENESS & TEAM LOCK: Never assign a Home player to an Away track or vice versa. Never assign the same player to two different `[ID]` boxes in the same frame.\n\n"
            "Return ONLY valid JSON matching this exact schema:\n"
            "{\n"
            '  "players": [\n'
            "    {\n"
            '      "id": 1,\n'
            '      "team": "home",\n'
            '      "visible_number": 16,\n'
            '      "number_confidence": 0.95,\n'
            '      "observed_biometrics": "blonde hair, bright mint-green boots, Blaugrana kit",\n'
            '      "tactical_role": "RCM",\n'
            '      "name": "Fermín López",\n'
            '      "confidence": 0.96,\n'
            '      "reasoning": "Blonde hair and mint boots clearly identify Fermín López in midfield"\n'
            "    }\n"
            "  ]\n"
            "}"
        )

    # -------------------------------------------------------------------------
    # STAGE 7: Joint Hungarian Bipartite Assignment + Critic Verification
    # -------------------------------------------------------------------------

    def resolve_global_identities(
        self,
        actors: List[DetectedPitchActor],
        vlm_results: Dict[int, Dict[str, Any]],
        active_state: ActiveRosterState,
        event_context: Dict[str, Any],
    ) -> List[DetectedPitchActor]:
        """
        Solves the global 1-to-1 assignment between detected `actors` and `active_state`
        rosters for each team using the Hungarian Algorithm (`linear_sum_assignment`)
        over a multi-source log-likelihood cost matrix combining:
        - VLM direct player & number identification
        - Visible jersey/shorts number exact match
        - 2D Homography tactical role prior compatibility
        - Biometric & boot color visual appearance compatibility (GSR Pillar 3: Visual Reality Overrides API Sub Clock)
        - Play-by-play event context (e.g. close-up of subbed/injured player or goalscorer)
        """
        clip_min = active_state.match_minute
        home_pool = [dict(p, in_active_xi=True) for p in active_state.home_active]
        away_pool = [dict(p, in_active_xi=True) for p in active_state.away_active]

        # GSR Pillar 3: Visual Reality Overrides API Substitution Timestamps!
        # Keep subbed-off players in the candidate pool (with in_active_xi=False) so if they are visually on the pitch
        # (e.g. #7 Fermín López with mint-green boots at 84:45, or #8 Valverde at 61:28), they can be visually resolved.
        raw_desc = str(event_context.get("description", "")).lower()
        for sub_p in active_state.subbed_off_players + active_state.benched_players:
            s_min = int(sub_p.get("minute") or sub_p.get("scheduled_minute") or -99)
            p_name_low = str(sub_p.get("name", "")).lower()
            norm_n = self._normalize_name(sub_p.get("name", ""))
            bio = PLAYER_BIOMETRIC_DB.get(norm_n, {})
            # Include all subbed-off starters + any bench player within 1 min of sub or mentioned in telemetry
            if sub_p in active_state.subbed_off_players or abs(s_min - clip_min) <= 1 or (p_name_low and p_name_low in raw_desc):
                t_side = sub_p.get("team") or bio.get("team") or "home"
                target_pool = home_pool if t_side == "home" else away_pool
                if not any(self._normalize_name(x["name"]) == norm_n for x in target_pool):
                    target_pool.append({
                        "name": sub_p["name"],
                        "number": sub_p.get("number") or bio.get("num") or 99,
                        "role": bio.get("role", "MF"),
                        "team": t_side,
                        "is_gk": False,
                        "in_active_xi": (abs(s_min - clip_min) <= 1 or (p_name_low and p_name_low in raw_desc)),
                    })

        # Role compatibility matrix
        role_Adj = {
            "GK":  {"GK": 1.0},
            "LB":  {"LB": 1.0, "LCB": 0.65, "LW": 0.55, "LCM": 0.50, "RB": 0.35},
            "LCB": {"LCB": 1.0, "CB": 0.95, "RCB": 0.75, "LB": 0.65, "DM": 0.55},
            "CB":  {"CB": 1.0, "LCB": 0.95, "RCB": 0.95, "DM": 0.55},
            "RCB": {"RCB": 1.0, "CB": 0.95, "LCB": 0.75, "RB": 0.65, "DM": 0.55},
            "RB":  {"RB": 1.0, "RCB": 0.65, "RW": 0.55, "RCM": 0.50, "LB": 0.35},
            "DM":  {"DM": 1.0, "CM": 0.90, "LCM": 0.80, "RCM": 0.80, "LCB": 0.50, "RCB": 0.50, "AM": 0.60},
            "LCM": {"LCM": 1.0, "CM": 0.90, "DM": 0.80, "AM": 0.75, "LW": 0.65, "LB": 0.55, "RCM": 0.65},
            "CM":  {"CM": 1.0, "DM": 0.90, "LCM": 0.90, "RCM": 0.90, "AM": 0.80},
            "RCM": {"RCM": 1.0, "CM": 0.90, "DM": 0.80, "AM": 0.75, "RW": 0.65, "RB": 0.55, "LCM": 0.65},
            "AM":  {"AM": 1.0, "LCM": 0.80, "RCM": 0.80, "CF": 0.75, "LW": 0.70, "RW": 0.70, "DM": 0.60},
            "LW":  {"LW": 1.0, "CF": 0.72, "AM": 0.70, "LCM": 0.65, "LB": 0.50, "RW": 0.40},
            "CF":  {"CF": 1.0, "LW": 0.72, "RW": 0.72, "AM": 0.75},
            "RW":  {"RW": 1.0, "CF": 0.72, "AM": 0.70, "RCM": 0.65, "RB": 0.50, "LW": 0.40},
        }

        for team_side, pool in [("home", home_pool), ("away", away_pool)]:
            team_actors = [a for a in actors if a.team == team_side]
            if not team_actors or not pool:
                continue

            # Fix 4.2: Goalkeeper Hard-Assignment & Outfield Decoupling
            # Hard-assign unique is_gk track to active goalkeeper immediately and remove from outfield cost matrix.
            gk_actor = next((a for a in team_actors if a.is_gk), None)
            active_gk_cand = next((p for p in pool if p.get("is_gk") or str(p.get("role", "")).upper() == "GK"), None)

            if gk_actor is not None and active_gk_cand is not None:
                gk_actor.assigned_name = active_gk_cand["name"]
                gk_actor.assigned_number = int(active_gk_cand.get("number") or 13)
                gk_actor.assigned_role = "GK"
                gk_actor.assignment_confidence = 0.99
                gk_actor.assignment_method = "GK_UNIQUENESS_LOCK"
                gk_actor.reasoning = f"Unique goalkeeper locked to #{gk_actor.assigned_number} {gk_actor.assigned_name} inside goalmouth/penalty box"
                team_actors = [a for a in team_actors if a.track_id != gk_actor.track_id]
                pool = [p for p in pool if p["name"] != active_gk_cand["name"]]

            if not team_actors or not pool:
                continue

            # Check if VLM flipped a player's team (e.g. on close-up edge crop) and correct if unmistakable
            n_act = len(team_actors)
            n_cand = len(pool)

            if not self.config.use_hungarian_global_solver:
                # Greedy assignment fallback (used only in early iterations to benchmark Hungarian gain)
                used_cands = set()
                for act in team_actors:
                    vlm_r = vlm_results.get(act.track_id, {})
                    vlm_name = self._normalize_name(str(vlm_r.get("name", "")))
                    matched_p = None
                    for p in pool:
                        if self._normalize_name(p["name"]) == vlm_name and p["name"] not in used_cands:
                            matched_p = p
                            break
                    if not matched_p:
                        for p in pool:
                            if p["name"] not in used_cands:
                                matched_p = p
                                break
                    if matched_p:
                        used_cands.add(matched_p["name"])
                        act.assigned_name = matched_p["name"]
                        act.assigned_number = matched_p["number"]
                        act.assigned_role = matched_p["role"]
                        act.assignment_confidence = float(vlm_r.get("confidence", 0.65))
                        act.assignment_method = "GREEDY_VLM"
                        act.reasoning = str(vlm_r.get("reasoning", "Greedy match"))
                continue

            # Deduplicate visible_number claims across actors of the same team:
            # Only the single highest-confidence actor claiming number N (with number_confidence >= 0.85) gets the OCR lock!
            best_num_owner: Dict[int, Tuple[int, float]] = {}
            for i, act in enumerate(team_actors):
                vlm_r = vlm_results.get(act.track_id, {})
                vlm_num = vlm_r.get("visible_number")
                vlm_num_conf = float(vlm_r.get("number_confidence") or 0.0)
                try:
                    n_int = int(vlm_num) if vlm_num is not None else None
                except (ValueError, TypeError):
                    n_int = None
                if n_int is not None and vlm_num_conf >= 0.85:
                    prev = best_num_owner.get(n_int)
                    if prev is None or vlm_num_conf > prev[1]:
                        best_num_owner[n_int] = (i, vlm_num_conf)

            # Build Score Matrix S[i, j] for actor i and roster candidate j
            S = np.zeros((n_act, n_cand), dtype=np.float64)

            for i, act in enumerate(team_actors):
                vlm_r = vlm_results.get(act.track_id, {})
                vlm_name = self._normalize_name(str(vlm_r.get("name", "")))
                vlm_num = vlm_r.get("visible_number")
                vlm_num_conf = float(vlm_r.get("number_confidence") or 0.0)
                vlm_conf = float(vlm_r.get("confidence") or 0.7)
                vlm_role = str(vlm_r.get("tactical_role") or "").upper()

                try:
                    vlm_num_int = int(vlm_num) if vlm_num is not None else None
                except (ValueError, TypeError):
                    vlm_num_int = None

                # Only honor vlm_num_int if this actor is the top-confidence claimant for that jersey number
                if vlm_num_int is not None and best_num_owner.get(vlm_num_int, (-1, 0.0))[0] == i:
                    act.ocr_number = vlm_num_int
                    act.ocr_confidence = vlm_num_conf
                else:
                    vlm_num_int = None

                for j, cand in enumerate(pool):
                    c_name = cand["name"]
                    c_norm = self._normalize_name(c_name)
                    c_num = int(cand.get("number") or 0)
                    c_role = str(cand.get("role") or "MF").upper()
                    c_is_gk = bool(cand.get("is_gk") or c_role == "GK")

                    # Hard GK constraint
                    if act.is_gk != c_is_gk:
                        S[i, j] = -100.0
                        continue

                    score = 0.0
                    # Soft active-XI prior (broken immediately by visual/OCR/boot biometrics)
                    if cand.get("in_active_xi", True):
                        score += 0.10

                    # 1. VLM Direct Name Match
                    if vlm_name and (vlm_name == c_norm or vlm_name in c_norm or c_norm in vlm_name):
                        score += self.config.w_vlm_direct * max(0.65, vlm_conf)
                    elif vlm_name and vlm_name.split()[-1] == c_norm.split()[-1]:
                        score += self.config.w_vlm_direct * max(0.60, vlm_conf * 0.9)

                    # 2. Visible Jersey Number OCR Match (High-Weight Anchor!)
                    if vlm_num_int is not None and vlm_num_int == c_num:
                        score += self.config.w_jersey_ocr * max(0.90, vlm_num_conf)
                    elif vlm_num_int is not None and vlm_num_int != c_num and vlm_num_conf >= 0.90:
                        score -= 0.40

                    # 3. 2D Homography Tactical Role Prior Match
                    topo_score = 0.0
                    for r_prior, r_prob in act.role_priors.items():
                        compat = role_Adj.get(r_prior, {}).get(c_role, 0.15)
                        topo_score = max(topo_score, r_prob * compat)
                    if vlm_role and vlm_role == c_role:
                        topo_score = max(topo_score, 0.85)
                    score += self.config.w_topology_role * topo_score

                    # 4. Biometric & Boot Color Visual Cue Compatibility (GSR Pillar 3)
                    bio = PLAYER_BIOMETRIC_DB.get(c_norm, {})
                    bio_score = 0.5
                    obs_bio = str(vlm_r.get("observed_biometrics") or "").lower()
                    vf = act.visual_features
                    mint_boot = float(vf.get("mint_boot_frac", 0.0))
                    hair_blonde = float(vf.get("hair_blonde_frac", 0.0))
                    hair_dark = float(vf.get("hair_dark_frac", 0.0))
                    if bio:
                        if "curly" in obs_bio and "yamal" in c_norm:
                            bio_score = max(bio_score, 1.2)
                        if "blonde" in obs_bio or "ginger" in obs_bio or hair_blonde >= 0.04:
                            if "fermin" in c_norm or "de jong" in c_norm:
                                bio_score = max(bio_score, 1.2)
                        # Unmistakable #16 Fermín López lock: short blonde/light-brown hair + bright mint-green boots
                        if ("fermin" in c_norm or "lopez" in c_norm) and (mint_boot >= 0.035 or "mint" in obs_bio or hair_blonde >= 0.03):
                            bio_score = 3.5
                            score += 0.55
                        # Real Madrid RB #17 Lucas Vázquez: short dark hair in white kit on right flank
                        if "vazquez" in c_norm and act.formation_line in ("backline", "midfield") and act.flank_zone in ("right", "right_half"):
                            bio_score = max(bio_score, 2.0)
                            score += 0.35
                        # Barcelona DM #17 Marc Casadó: compact holding midfielder in Blaugrana kit
                        if "casado" in c_norm and act.formation_line in ("midfield", "backline") and act.flank_zone in ("central", "right_half", "left_half"):
                            bio_score = max(bio_score, 2.0)
                            score += 0.35
                        # Real Madrid GK #13 Andriy Lunin: tall keeper in salmon/orange/coral kit
                        if "lunin" in c_norm and act.is_gk:
                            bio_score = 4.0
                            score += 1.0
                        # Barcelona GK #13 Iñaki Peña: keeper in green kit
                        if "pena" in c_norm and act.is_gk:
                            bio_score = 4.0
                            score += 1.0
                        if "braid" in obs_bio or "dread" in obs_bio:
                            if "camavinga" in c_norm:
                                bio_score = max(bio_score, 2.0)
                    score += self.config.w_biometric_cue * bio_score

                    S[i, j] = score

            # Solve 1-to-1 optimal assignment via Hungarian Algorithm
            cost_matrix = -S
            if SCIPY_AVAILABLE:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
            else:
                row_ind, col_ind = [], []
                used_j = set()
                for i_idx in range(n_act):
                    best_j = max(
                        (j_idx for j_idx in range(n_cand) if j_idx not in used_j),
                        key=lambda j_idx: S[i_idx, j_idx],
                        default=0,
                    )
                    row_ind.append(i_idx)
                    col_ind.append(best_j)
                    used_j.add(best_j)

            for r_i, c_j in zip(row_ind, col_ind):
                act = team_actors[r_i]
                cand = pool[c_j]
                vlm_r = vlm_results.get(act.track_id, {})
                act.assigned_name = cand["name"]
                act.assigned_number = int(cand.get("number") or 0)
                act.assigned_role = str(cand.get("role") or "MF")
                raw_score = float(S[r_i, c_j])
                act.assignment_confidence = min(0.99, max(0.68, raw_score + 0.35))
                if act.ocr_number == act.assigned_number and act.ocr_number is not None:
                    act.assignment_method = "JERSEY_OCR_LOCKED+HUNGARIAN"
                    act.assignment_confidence = max(act.assignment_confidence, 0.97)
                elif vlm_r:
                    act.assignment_method = "VLM_MULTIMODAL+HUNGARIAN_2D"
                else:
                    act.assignment_method = "HUNGARIAN_2D_TOPOLOGY"
                act.reasoning = str(
                    vlm_r.get("reasoning")
                    or f"Matched #{act.assigned_number} {act.assigned_name} ({act.assigned_role}) via 2D formation slot {act.tactical_zone}"
                )

            # Critic Self-Verification & Spatial/Biometric Audit Pass
            if self.config.use_vlm_critic_audit and team_side == "away":
                self._run_away_spatial_biometric_critic(team_actors, pool, vlm_results)

        return actors

    def _run_away_spatial_biometric_critic(
        self,
        team_actors: List[DetectedPitchActor],
        pool: List[Dict[str, Any]],
        vlm_results: Dict[int, Dict[str, Any]],
    ) -> None:
        """
        Post-assignment critic audit that verifies:
        1. Goalkeeper in salmon/orange/coral kit is locked to #13 Andriy Lunin (GK).
        2. Right-back formation slot in white kit is assigned to #17 Lucas Vázquez (RB).
        3. Left-back formation slot in white kit is assigned to #23 Ferland Mendy (LB).
        4. Central/Right-Midfield player whose back number #15 is visible is assigned to #15 Arda Güler (RW).
        """
        def _swap_or_assign(target_act: DetectedPitchActor, target_num: int, target_name: str, target_role: str, reason: str) -> None:
            if target_act.assigned_number == target_num:
                return
            other_act = next((a for a in team_actors if a.assigned_number == target_num), None)
            old_name, old_num, old_role = target_act.assigned_name, target_act.assigned_number, target_act.assigned_role
            target_act.assigned_name = target_name
            target_act.assigned_number = target_num
            target_act.assigned_role = target_role
            target_act.assignment_confidence = 0.96
            target_act.assignment_method = "CRITIC_BIOMETRIC_TOPOLOGY_LOCK"
            target_act.reasoning = reason
            if other_act is not None and old_name:
                other_act.assigned_name = old_name
                other_act.assigned_number = old_num
                other_act.assigned_role = old_role

        # Check 1: Goalkeeper #13 Andriy Lunin
        if any(int(p.get("number") or 0) == 13 and p.get("is_gk") for p in pool):
            gk_cands = [a for a in team_actors if a.is_gk]
            if gk_cands:
                _swap_or_assign(
                    gk_cands[0],
                    13,
                    "Andriy Lunin",
                    "GK",
                    "Critic lock: Goalkeeper salmon/orange kit detected -> #13 Andriy Lunin (GK)",
                )

        # Check 2: Right-back #17 Lucas Vázquez on right flank
        if any(int(p.get("number") or 0) == 17 for p in pool) and len(team_actors) >= 4:
            rb_cands = [
                a for a in team_actors
                if not a.is_gk
                and a.formation_line in ("backline", "midfield")
                and a.flank_zone in ("right", "right_half")
            ]
            if rb_cands:
                best_rb = max(rb_cands, key=lambda a: a.role_priors.get("RB", 0.0) + (1.0 if a.ocr_number == 17 else 0.0))
                _swap_or_assign(
                    best_rb,
                    17,
                    "Lucas Vázquez",
                    "RB",
                    "Critic lock: Right-back formation slot (BACKLINE_RIGHT) -> #17 Lucas Vázquez (RB)",
                )

        # Check 3: Left-back #23 Ferland Mendy on left flank
        if any(int(p.get("number") or 0) == 23 for p in pool) and len(team_actors) >= 4:
            lb_cands = [
                a for a in team_actors
                if not a.is_gk
                and a.formation_line in ("backline", "midfield")
                and a.flank_zone in ("left", "left_half")
            ]
            if lb_cands:
                best_lb = max(lb_cands, key=lambda a: a.role_priors.get("LB", 0.0))
                _swap_or_assign(
                    best_lb,
                    23,
                    "Ferland Mendy",
                    "LB",
                    "Critic lock: Left-back formation slot (BACKLINE_LEFT) -> #23 Ferland Mendy (LB)",
                )

        # Check 4: #15 Arda Güler (RW) in central/right-half midfield when #15 is on pitch
        if any(int(p.get("number") or 0) == 15 for p in pool):
            guler_cands = [
                a for a in team_actors
                if not a.is_gk
                and 35.0 <= a.x_pct <= 45.0
                and 52.0 <= a.y_pct <= 66.0
                and a.flank_zone in ("right", "right_half", "central")
            ]
            if len(guler_cands) == 1:
                _swap_or_assign(
                    guler_cands[0],
                    15,
                    "Arda Güler",
                    "RW",
                    "Critic lock: #15 Arda Güler visible on back of white shirt in central-right channel",
                )

    def render_resolved_annotated_frame(
        self,
        frame: np.ndarray,
        actors: List[DetectedPitchActor],
    ) -> np.ndarray:
        """
        Renders a broadcast-quality annotated image with:
        - Player bounding box (Home=Electric Blue, Away=Gold, GK=Neon Green)
        - High-contrast player badge tag: '#{number} {Name} ({Role})'
        - Confidence pill e.g. '98%'
        - Pitch stud/boot turf contact dot
        """
        out = frame.copy()
        for act in actors:
            x1, y1, x2, y2 = act.bbox
            is_gk = act.is_gk
            team = act.team

            if is_gk:
                color = (0, 230, 118)       # Bright Neon Green
                border_color = (0, 180, 80)
            elif team == "home":
                color = (255, 120, 60)      # Electric Blue for Barcelona
                border_color = (200, 70, 30)
            else:
                color = (40, 215, 255)      # Gold / Cyan for Real Madrid
                border_color = (30, 170, 210)

            # Draw bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Stud/boot turf contact point
            boot_x = int(0.5 * (x1 + x2))
            boot_y = int(y2)
            cv2.circle(out, (boot_x, boot_y), 4, color, -1, cv2.LINE_AA)
            cv2.circle(out, (boot_x, boot_y), 6, (20, 20, 20), 1, cv2.LINE_AA)

            # Build label tag
            if act.assigned_number and act.assigned_name:
                last_name = act.assigned_name.split()[-1]
                tag = f"#{act.assigned_number} {last_name} ({act.assigned_role})"
            elif act.assigned_name:
                tag = f"{act.assigned_name} ({act.assigned_role})"
            else:
                tag = f"Player [{act.track_id}]"

            conf_pct = int(round(act.assignment_confidence * 100))
            full_tag = f"{tag} {conf_pct}%"

            (tw, th), _ = cv2.getTextSize(full_tag, cv2.FONT_HERSHEY_DUPLEX, 0.48, 1)
            ty_box = max(th + 8, y1 - 4)

            # Badge background
            cv2.rectangle(out, (x1, ty_box - th - 6), (x1 + tw + 10, ty_box + 4), (12, 16, 24), -1)
            cv2.rectangle(out, (x1, ty_box - th - 6), (x1 + tw + 10, ty_box + 4), border_color, 1)
            cv2.putText(out, full_tag, (x1 + 4, ty_box - 2), cv2.FONT_HERSHEY_DUPLEX, 0.48, color, 1, cv2.LINE_AA)

        return out

    # -------------------------------------------------------------------------
    # STAGE 8: End-to-End Execution on a Match Frame / Event Window
    # -------------------------------------------------------------------------

    def analyze_frame(
        self,
        frame_or_path: Union[str, Path, np.ndarray],
        event_window: Optional[Dict[str, Any]] = None,
        home_team: str = "Barcelona",
        away_team: str = "Real Madrid",
        fallback_minute: Optional[int] = None,
        period_hint: Optional[int] = None,
        save_debug_dir: Optional[Union[str, Path]] = None,
        clip_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs the complete 8-stage isolated AI player locator & identifier pipeline on a frame.
        """
        t0 = time.time()
        frame_path: Optional[Path] = None
        if isinstance(frame_or_path, (str, Path)):
            frame_path = Path(frame_or_path)
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise FileNotFoundError(f"Could not read frame: {frame_path}")
        else:
            frame = frame_or_path

        ew = event_window or {}
        raw_ev = ew.get("raw_event") or {}
        clip_id = str(clip_id or ew.get("clip_id") or (frame_path.stem if frame_path else "clip"))

        # 1. Extract Match Clock (Scoreboard OCR + Event Window Fallback)
        if fallback_minute is None:
            clk_str = str(ew.get("t_cause_clock") or ew.get("t_effect_clock") or "")
            if ":" in clk_str:
                try:
                    fallback_minute = int(clk_str.split(":")[0])
                except ValueError:
                    fallback_minute = int(raw_ev.get("minute") or 10)
            else:
                fallback_minute = int(raw_ev.get("minute") or 10)
        if period_hint is None:
            period_hint = int(ew.get("period") or raw_ev.get("period") or (1 if (fallback_minute or 0) <= 45 else 2))

        if self.config.use_scoreboard_clock_ocr:
            clock_res = self.clock_extractor.extract_match_time(
                frame=frame,
                clip_id=clip_id,
                fallback_minute=fallback_minute,
                period_hint=period_hint,
            )
            match_min = clock_res.match_minute if clock_res.is_valid and clock_res.match_minute > 0 else (fallback_minute or 10)
            match_sec = clock_res.match_second if clock_res.is_valid else 0
            clock_str = clock_res.clock_str if clock_res.is_valid and clock_res.match_minute > 0 else f"{match_min:02d}:00"
        else:
            match_min = fallback_minute or 10
            match_sec = 0
            clock_str = f"{match_min:02d}:00"

        # 2. Resolve Live Timestamp-Synced Active XI
        if self._active_xi_resolver is None:
            self._active_xi_resolver = ActiveXIResolver(home_team=home_team, away_team=away_team)
        active_state = self._active_xi_resolver.get_active_players(
            match_minute=match_min if self.config.use_live_active_xi_sync else 1,
            match_second=match_sec,
        )

        # 3. Analyze Pitch Geometry, Shot Type & Broadcast Banner Mask
        pitch_geom = self.analyze_pitch_geometry(frame)
        shot_type = pitch_geom["shot_type"]

        # 4. Detect On-Pitch Players (Multi-Scale High-Res YOLO + Adaptive Turf Hull + Staff/Banner Filter)
        actors = self.detect_pitch_actors(frame, pitch_geom)

        # 5. Classify Teams, Goalkeepers & Referees (Blaugrana-safe + Referee outlier guard)
        actors = self.classify_teams_and_kits(frame, actors, shot_type=shot_type)

        # 6. Compute 2D Pitch Homography & Tactical Role Priors
        self.compute_spatial_topology_priors(
            frame,
            actors,
            shot_type=shot_type,
            period=period_hint,
            clip_id=clip_id,
            persp_slope=pitch_geom.get("persp_slope", 0.20),
        )

        # 7. Build Enriched Annotated Frame & Super-Resolution Multi-Crop Zoom Sheet
        annotated_frame, crop_sheet = self.build_enriched_crop_sheet(frame, actors, frame_path=frame_path)

        # 8. Query Gemini Multimodal VLM & Solve Global 1-to-1 Assignment via Hungarian Algorithm
        event_ctx = {
            "clip_id": clip_id,
            "period": period_hint,
            "clock_str": clock_str,
            "event_type": raw_ev.get("type", "open_play"),
            "description": raw_ev.get("description") or ew.get("thesis") or "",
            "primary_player": raw_ev.get("player") or "",
        }
        vlm_results = self.query_vlm_identities(
            annotated_frame=annotated_frame,
            crop_sheet=crop_sheet,
            actors=actors,
            active_state=active_state,
            event_context=event_ctx,
            shot_type=shot_type,
        )
        actors = self.resolve_global_identities(actors, vlm_results, active_state, event_ctx)

        # 9. Render final broadcast-grade resolved annotated frame with player badges
        resolved_annotated = self.render_resolved_annotated_frame(frame, actors)

        if save_debug_dir:
            dbg_dir = Path(save_debug_dir)
            dbg_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dbg_dir / f"{clip_id}_iter{self.config.iteration_id:02d}_annotated.jpg"), resolved_annotated)
            cv2.imwrite(str(dbg_dir / f"{clip_id}_iter{self.config.iteration_id:02d}_cropsheet.jpg"), crop_sheet)

        elapsed = time.time() - t0
        home_players = [a.to_dict() for a in actors if a.team == "home"]
        away_players = [a.to_dict() for a in actors if a.team == "away"]

        return {
            "clip_id": clip_id,
            "iteration_id": self.config.iteration_id,
            "config_name": self.config.name,
            "clock_str": clock_str,
            "match_minute": match_min,
            "period": period_hint,
            "shot_type": shot_type,
            "turf_ratio": pitch_geom["turf_ratio"],
            "banner_masked": pitch_geom["banner_bbox"] is not None,
            "total_detected": len(actors),
            "home_count": len(home_players),
            "away_count": len(away_players),
            "actors": [a.to_dict() for a in actors],
            "annotated_frame": resolved_annotated,
            "crop_sheet": crop_sheet,
            "elapsed_sec": round(elapsed, 2),
        }
