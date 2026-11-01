"""
src/vision/shot_classifier.py
Broadcast Camera Framing & Autonomous Tactical Frame Intelligence Gate.

Differentiates between:
1. WIDE_TACTICAL: High-angle broadcast pitch view showing multiple players from both teams,
   wide spatial formation structure, clear upper/mid pitch visibility (no giant ad-board tarps
   filling the top half), and zero broadcast substitution/lower-third graphic overlays.
   Suitable for freeze-frame tactical breakdown and 2D/3D AR pitch telestration.
2. CLOSE_UP: Tight player zoom, pitch-level mid-shot, tackle replay, referee conference,
   technical area / substitution bench shot, or tight 1v1 touchline zoom dominated by ad boards.
   Requires complete freeze-frame & overlay suppression for 100% clean live broadcast flow.
"""

from __future__ import annotations

import enum
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("vision.shot_classifier")


class ShotType(str, enum.Enum):
    WIDE_TACTICAL = "wide_tactical"
    CLOSE_UP = "close_up"


class CameraShotClassifier:
    """
    Autonomous Tactical Frame Intelligence Gate:
    Evaluates true playable pitch grass (rejecting olive/grey stadium tarps),
    upper/mid pitch boundary geometry, broadcast graphic overlays (substitutions/cards),
    real YOLO person bounding-box scales & spatial distribution, and tactical interest scoring.
    """

    _yolo_model: Optional[Any] = None
    _yolo_failed: bool = False

    # True playable grass HSV bounds (excludes low-saturation olive/grey stadium tarps where S < 48)
    GRASS_HSV_LOW = np.array([32, 48, 42], dtype=np.uint8)
    GRASS_HSV_HIGH = np.array([84, 255, 255], dtype=np.uint8)

    @classmethod
    def _get_yolo(cls) -> Optional[Any]:
        if cls._yolo_model is None and not cls._yolo_failed:
            try:
                from ultralytics import YOLO
                cls._yolo_model = YOLO("yolov8n.pt")
            except Exception as e:
                logger.warning("CameraShotClassifier could not load YOLOv8n (%s); using contour fallback", e)
                cls._yolo_failed = True
        return cls._yolo_model

    @classmethod
    def _detect_broadcast_overlay_banner(cls, img: np.ndarray, hsv: np.ndarray, turf_ratio: float) -> bool:
        """
        Detects tall lower-third broadcast substitution / player card graphic boxes
        (e.g. 'OUT 8 VALVERDE / IN 24' tall white box at bottom-left) when combined
        with non-wide framing. Ignores thin horizontal lower-third straps on wide 20-player shots.
        """
        if turf_ratio >= 0.58:
            return False
        h, w = img.shape[:2]
        y1, y2 = int(0.75 * h), int(0.99 * h)
        x1, x2 = int(0.03 * w), int(0.42 * w)
        roi_hsv = hsv[y1:y2, x1:x2]
        if roi_hsv.size == 0:
            return False

        white_mask = ((roi_hsv[:, :, 1] < 22) & (roi_hsv[:, :, 2] > 215)).astype(np.uint8) * 255
        white_frac = float(np.count_nonzero(white_mask)) / float(white_mask.size)
        if white_frac > 0.24:
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                _, _, cw, ch = cv2.boundingRect(c)
                if cw >= int(0.07 * w) and ch >= int(0.14 * h):
                    return True
        return False

    @classmethod
    def classify(
        cls,
        image: Union[str, Path, np.ndarray, Image.Image],
        turf_threshold: float = 0.45,
        max_player_height_threshold: float = 0.23,
        min_players: int = 5,
    ) -> Tuple[ShotType, Dict[str, Any]]:
        """
        Classifies an input broadcast frame into WIDE_TACTICAL or CLOSE_UP,
        and computes a multi-factor `tactical_interest_score` (0..100).

        Returns:
            (ShotType, metrics_dict)
        """
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        elif isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            img = image
        else:
            return ShotType.CLOSE_UP, {"error": "Invalid image input", "is_tactical": False}

        if img is None or img.size == 0:
            return ShotType.CLOSE_UP, {"error": "Empty image", "is_tactical": False}

        h, w = img.shape[:2]

        # 1. Playable Green Pitch Segmentation (excludes olive/grey ad-board tarps)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        raw_grass = cv2.inRange(hsv, cls.GRASS_HSV_LOW, cls.GRASS_HSV_HIGH)
        raw_grass[0:int(0.16 * h), 0:int(0.20 * w)] = 0
        raw_grass[0:int(0.15 * h), int(0.86 * w):w] = 0

        # Adaptive saturation threshold: separates vibrant turf from desaturated olive/grey stadium tarps
        grass_s_vals = hsv[:, :, 1][raw_grass > 0]
        s_floor = 48
        if grass_s_vals.size > 500:
            s_ref = float(np.percentile(grass_s_vals, 75))
            s_floor = max(48, min(86, int(0.68 * s_ref)))
        adaptive_low = np.array([32, s_floor, 42], dtype=np.uint8)
        mask_grass = cv2.inRange(hsv, adaptive_low, cls.GRASS_HSV_HIGH)

        # Mask out top-left scoreboard bug and top-right broadcaster logo
        mask_grass[0:int(0.16 * h), 0:int(0.20 * w)] = 0
        mask_grass[0:int(0.15 * h), int(0.86 * w):w] = 0

        # Clean small noise specks in crowd/banners
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_grass = cv2.morphologyEx(mask_grass, cv2.MORPH_OPEN, kernel_clean)
        mask_grass = cv2.morphologyEx(mask_grass, cv2.MORPH_CLOSE, kernel_clean)

        total_pixels = max(1, h * w)
        turf_ratio = float(cv2.countNonZero(mask_grass)) / float(total_pixels)

        # Zonal playable grass ratios
        upper_turf = float(mask_grass[int(0.16 * h):int(0.40 * h), :].mean() / 255.0)
        mid_turf = float(mask_grass[int(0.40 * h):int(0.78 * h), :].mean() / 255.0)
        lower_turf = float(mask_grass[int(0.78 * h):int(0.96 * h), :].mean() / 255.0)

        # Find highest playable pitch line (top touchline Y%)
        row_grass_frac = (mask_grass > 0).mean(axis=1)
        valid_rows = np.where(row_grass_frac >= 0.40)[0]
        pitch_top_y_pct = float(valid_rows[0]) / float(h) * 100.0 if len(valid_rows) > 0 else 100.0

        # 2. Check for lower-third substitution / replay graphic overlay
        has_sub_graphic = cls._detect_broadcast_overlay_banner(img, hsv, turf_ratio)

        # 3. Detect Players & Evaluate Foreground Scale + Spatial Pitch Distribution
        yolo = cls._get_yolo()
        max_player_h = 0
        max_player_w = 0
        max_player_area_ratio = 0.0
        on_pitch_players: List[Dict[str, float]] = []
        all_person_heights: List[float] = []
        home_like_count = 0
        away_like_count = 0

        if yolo is not None:
            try:
                res = yolo.predict(img, classes=[0], conf=0.20, imgsz=640, verbose=False)[0]
                boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
                confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else []
                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    bw = max(1, x2 - x1)
                    bh = max(1, y2 - y1)
                    area_r = float(bw * bh) / float(total_pixels)
                    x_foot = int(0.5 * (x1 + x2))
                    y_foot = int(y2)
                    x_pct = (x_foot / float(w)) * 100.0
                    y_pct = (y_foot / float(h)) * 100.0

                    # Ignore tiny cropped edge stewards/spectators on far left/right border (< 5% or > 95%) unless huge
                    is_central_or_large = (5.5 <= x_pct <= 94.5 and y_pct >= 18.0) or area_r > 0.035
                    if float(conf) >= 0.25 and is_central_or_large:
                        if bh > max_player_h:
                            max_player_h = bh
                        if bw > max_player_w:
                            max_player_w = bw
                        if area_r > max_player_area_ratio:
                            max_player_area_ratio = area_r
                        all_person_heights.append(float(bh) / float(h))

                    if y_pct < max(13.0, pitch_top_y_pct - 2.0) or y_pct > 93.0:
                        continue
                    if x_pct < 2.5 or x_pct > 97.5:
                        continue

                    # Foot grass check (20x16px window around boots)
                    fx1, fx2 = max(0, x_foot - 10), min(w, x_foot + 10)
                    fy1, fy2 = max(0, y_foot - 8), min(h, y_foot + 8)
                    foot_patch = mask_grass[fy1:fy2, fx1:fx2]
                    foot_grass = float(np.count_nonzero(foot_patch)) / float(max(1, foot_patch.size))
                    if foot_grass < 0.22:
                        continue

                    # Aspect & scale sanity for wide-angle outfield player
                    aspect = float(bh) / float(bw)
                    if aspect < 0.75 or bh < int(0.03 * h) or bh > int(0.24 * h):
                        continue

                    # Quick torso kit luminance/saturation check to verify multi-team presence
                    ty1 = max(0, y1 + int(0.15 * bh))
                    ty2 = min(h, y1 + int(0.52 * bh))
                    tx1 = max(0, x1 + int(0.18 * bw))
                    tx2 = min(w, x2 - int(0.18 * bw))
                    if ty2 > ty1 and tx2 > tx1:
                        torso_hsv = hsv[ty1:ty2, tx1:tx2]
                        med_s = float(np.median(torso_hsv[:, :, 1]))
                        med_v = float(np.median(torso_hsv[:, :, 2]))
                        if med_s < 42 and med_v > 145:
                            away_like_count += 1
                        else:
                            home_like_count += 1

                    on_pitch_players.append({
                        "x_pct": x_pct,
                        "y_pct": y_pct,
                        "bh_ratio": float(bh) / float(h),
                        "conf": float(conf),
                    })
            except Exception as e:
                logger.debug("YOLO check inside CameraShotClassifier failed: %s", e)

        # Fallback/supplemental contour check if YOLO didn't run
        if not on_pitch_players and yolo is None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            pitch_closed = cv2.morphologyEx(mask_grass, cv2.MORPH_CLOSE, kernel)
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            roi_mask[int(0.14 * h):int(0.86 * h), int(0.04 * w):int(0.96 * w)] = 255
            valid_pitch = cv2.bitwise_and(pitch_closed, roi_mask)
            fg_mask = cv2.bitwise_and(cv2.bitwise_not(mask_grass), valid_pitch)
            fg_clean = cv2.morphologyEx(
                fg_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            )
            contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                cx, cy, cw, ch = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                aspect = float(ch) / max(1, cw)
                if ch > max_player_h:
                    max_player_h = ch
                if 1.1 <= aspect <= 4.5 and 24 <= ch <= int(0.22 * h) and 90 <= area <= 5500:
                    on_pitch_players.append({
                        "x_pct": ((cx + 0.5 * cw) / float(w)) * 100.0,
                        "y_pct": ((cy + ch) / float(h)) * 100.0,
                        "bh_ratio": float(ch) / float(h),
                        "conf": 0.5,
                    })

        valid_players = len(on_pitch_players)
        max_player_ratio = float(max_player_h) / float(h)
        median_player_ratio = float(np.median(all_person_heights)) if all_person_heights else max_player_ratio

        # Spatial spread of on-pitch players
        if valid_players >= 2:
            xs = [p["x_pct"] for p in on_pitch_players]
            ys = [p["y_pct"] for p in on_pitch_players]
            x_span_pct = float(max(xs) - min(xs))
            y_span_pct = float(max(ys) - min(ys))
        else:
            x_span_pct = 0.0
            y_span_pct = 0.0

        # 4. Multi-Factor Tactical Interest Score (0..100)
        # Rewards:
        # - Rich player count across both teams (up to 35 pts)
        # - Wide horizontal & vertical formation depth (up to 30 pts)
        # - High playable turf dominance in mid & upper pitch (up to 20 pts)
        # - Clean compact player scale typical of master tactical cam (up to 15 pts)
        player_score = min(35.0, float(valid_players) * 3.5)
        if home_like_count >= 1 and away_like_count >= 1:
            player_score = min(35.0, player_score + 6.0)
        spread_score = min(18.0, x_span_pct * 0.28) + min(12.0, y_span_pct * 0.35)
        turf_score = min(20.0, (turf_ratio * 12.0) + (upper_turf * 8.0) + (mid_turf * 6.0))
        scale_score = 15.0 if (0.04 <= max_player_ratio <= 0.17) else (8.0 if max_player_ratio <= 0.22 else 0.0)
        tactical_interest_score = round(player_score + spread_score + turf_score + scale_score, 2)

        # 5. Strict Rejection Reasons (Why a frame is NEVER allowed to freeze)
        rejection_reasons: List[str] = []
        if has_sub_graphic:
            rejection_reasons.append("broadcast_substitution_or_lower_third_graphic")
        if turf_ratio < max(0.45, turf_threshold):
            rejection_reasons.append(f"low_global_turf({turf_ratio:.2f}<{max(0.45, turf_threshold):.2f})")
        if upper_turf < 0.16:
            rejection_reasons.append(f"low_upper_pitch_turf({upper_turf:.2f}<0.16_ad_board_or_crowd_top)")
        if mid_turf < 0.62:
            rejection_reasons.append(f"low_mid_pitch_turf({mid_turf:.2f}<0.62)")
        if pitch_top_y_pct > 42.0:
            rejection_reasons.append(f"pitch_top_too_low({pitch_top_y_pct:.1f}%>42%_tight_touchline_angle)")
        if max_player_ratio > max_player_height_threshold or max_player_area_ratio > 0.045:
            rejection_reasons.append(f"giant_foreground_player_closeup(h={max_player_ratio:.2f},area={max_player_area_ratio:.3f})")
        if median_player_ratio > 0.165:
            rejection_reasons.append(f"camera_zoom_too_tight(median_h={median_player_ratio:.2f}>0.165)")
        if valid_players < min_players:
            rejection_reasons.append(f"insufficient_on_pitch_players({valid_players}<{min_players})")
        if x_span_pct < 22.0 or y_span_pct < 12.0:
            rejection_reasons.append(f"narrow_tactical_spread(x_span={x_span_pct:.1f}%,y_span={y_span_pct:.1f}%)")

        is_close_up = len(rejection_reasons) > 0
        if is_close_up:
            tactical_interest_score = 0.0

        metrics: Dict[str, Any] = {
            "is_tactical": not is_close_up,
            "turf_ratio": round(turf_ratio, 3),
            "upper_turf": round(upper_turf, 3),
            "mid_turf": round(mid_turf, 3),
            "lower_turf": round(lower_turf, 3),
            "pitch_top_y_pct": round(pitch_top_y_pct, 1),
            "max_player_ratio": round(max_player_ratio, 3),
            "median_player_ratio": round(median_player_ratio, 3),
            "valid_players": valid_players,
            "home_like_players": home_like_count,
            "away_like_players": away_like_count,
            "x_span_pct": round(x_span_pct, 1),
            "y_span_pct": round(y_span_pct, 1),
            "has_sub_graphic": has_sub_graphic,
            "tactical_interest_score": tactical_interest_score,
            "rejection_reasons": rejection_reasons,
            "turf_threshold": turf_threshold,
            "max_player_threshold": max_player_height_threshold,
        }

        if is_close_up:
            logger.info(
                "Camera shot classified as CLOSE_UP / NON-TACTICAL (reasons=%s, turf=%.2f, upper=%.2f, players=%d, max_h=%.2f)",
                ",".join(rejection_reasons),
                turf_ratio,
                upper_turf,
                valid_players,
                max_player_ratio,
            )
            return ShotType.CLOSE_UP, metrics

        logger.info(
            "Camera shot classified as WIDE_TACTICAL (interest=%.1f, turf=%.2f, upper=%.2f, players=%d, span=%.1fx%.1f%%)",
            tactical_interest_score,
            turf_ratio,
            upper_turf,
            valid_players,
            x_span_pct,
            y_span_pct,
        )
        return ShotType.WIDE_TACTICAL, metrics
