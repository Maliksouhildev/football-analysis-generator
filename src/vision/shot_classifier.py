"""
src/vision/shot_classifier.py
Broadcast Camera Framing & Shot Classifier for Football Footage.

Differentiates between:
1. WIDE_TACTICAL: High-angle broadcast pitch view showing multiple players, tactical shape,
   and broad turf coverage (>= 40% green grass). Suitable for AR pitch telestration.
2. CLOSE_UP: Tight player zoom, tackle replay, referee conference, bench shot, or goal-mouth
   reaction (< 40% turf or single player occupies > 22% of frame height). Requires complete
   overlay blackout for 100% clean broadcast footage.
"""

from __future__ import annotations

import enum
import logging
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("vision.shot_classifier")


class ShotType(str, enum.Enum):
    WIDE_TACTICAL = "wide_tactical"
    CLOSE_UP = "close_up"


class CameraShotClassifier:
    """
    Evaluates pitch turf dominance, player scale, and framing geometry
    to determine whether tactical pitch overlays should be applied or suppressed.
    """

    @classmethod
    def classify(
        cls,
        image: Union[str, Path, np.ndarray, Image.Image],
        turf_threshold: float = 0.40,
        max_player_height_threshold: float = 0.22,
    ) -> Tuple[ShotType, dict]:
        """
        Classifies an input broadcast frame into WIDE_TACTICAL or CLOSE_UP.

        Returns:
            (ShotType, metrics_dict)
        """
        # Load into BGR numpy array
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        elif isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            img = image
        else:
            return ShotType.WIDE_TACTICAL, {"error": "Invalid image input"}

        if img is None or img.size == 0:
            return ShotType.WIDE_TACTICAL, {"error": "Empty image"}

        h, w, _ = img.shape

        # 1. Compute Green Turf Ratio (Global and Zonal)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_grass = cv2.inRange(hsv, np.array([28, 30, 30]), np.array([88, 255, 255]))
        turf_pixels = int(cv2.countNonZero(mask_grass))
        total_pixels = h * w
        turf_ratio = turf_pixels / max(1, total_pixels)

        upper_turf = float(mask_grass[:int(0.35 * h), :].mean() / 255.0)
        mid_turf = float(mask_grass[int(0.35 * h):int(0.75 * h), :].mean() / 255.0)

        # 2. Compute Max Player Scale on Pitch
        # Morphological close to bridge player gaps on turf
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        pitch_closed = cv2.morphologyEx(mask_grass, cv2.MORPH_CLOSE, kernel)

        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[int(0.12 * h):int(0.82 * h), int(0.04 * w):int(0.96 * w)] = 255
        valid_pitch = cv2.bitwise_and(pitch_closed, roi_mask)

        # Non-grass silhouettes inside the pitch region
        fg_mask = cv2.bitwise_and(cv2.bitwise_not(mask_grass), valid_pitch)
        fg_clean = cv2.morphologyEx(
            fg_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        )
        contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_player_h = 0
        valid_players = 0
        for c in contours:
            _, _, cw, ch = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            aspect = float(ch) / max(1, cw)
            if ch > max_player_h:
                max_player_h = ch
            # Standard wide-angle player silhouette criteria
            if 1.1 <= aspect <= 4.5 and 20 <= ch <= 120 and 80 <= area <= 4500:
                valid_players += 1

        max_player_ratio = max_player_h / float(h)

        metrics = {
            "turf_ratio": round(turf_ratio, 3),
            "upper_turf": round(upper_turf, 3),
            "mid_turf": round(mid_turf, 3),
            "max_player_ratio": round(max_player_ratio, 3),
            "valid_players": valid_players,
            "turf_threshold": turf_threshold,
            "max_player_threshold": max_player_height_threshold,
        }

        # Decision Logic:
        # A wide tactical master framing requires broad turf coverage (>= 48%),
        # upper pitch visibility (upper >= 12% to reject sideline low-tilted duels),
        # middle pitch grass dominance (mid >= 68%), and multiple outfield players (>= 4),
        # without any single foreground giant element (> 22% of frame).
        is_close_up = (
            turf_ratio < turf_threshold
            or upper_turf < 0.12
            or mid_turf < 0.68
            or max_player_ratio > max_player_height_threshold
            or valid_players < 4
        )

        if is_close_up:
            logger.info(
                f"Camera shot classified as CLOSE_UP (turf={turf_ratio:.2f}, upper={upper_turf:.2f}, mid={mid_turf:.2f}, players={valid_players}, scale={max_player_ratio:.2f})"
            )
            return ShotType.CLOSE_UP, metrics

        logger.info(
            f"Camera shot classified as WIDE_TACTICAL (turf={turf_ratio:.2f}, upper={upper_turf:.2f}, mid={mid_turf:.2f}, players={valid_players}, scale={max_player_ratio:.2f})"
        )
        return ShotType.WIDE_TACTICAL, metrics
