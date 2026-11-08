"""
src/detection_engine/pitch_geometry.py
================================================================================
Pitch boundary segmentation, turf convex hull extraction, physical pitch line
detection via Hough Transform, lower-third broadcast banner masking, and visual
turf polygon debug rendering.
================================================================================
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detection_engine.types import CandidateBox, DetectedPitchActor, PipelineConfig

logger = logging.getLogger("detection_engine.pitch_geometry")


class PitchGeometryEngine:
    """Handles pitch boundary segmentation, Hough line extraction, and turf debug rendering."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Segments green turf, fits convex hull, detects physical pitch lines via Hough lines,
        detects broadcast overlay banners, and classifies camera perspective / shot type.
        """
        return self._analyze_impl(frame)

    analyze_frame = analyze

    def _analyze_impl(self, frame: np.ndarray) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. Green pitch turf mask (Hue in [32, 85], S >= 35, V >= 35 strictly excludes cyan/teal LED boards)
        lower_green = np.array([32, 35, 35], dtype=np.uint8)
        upper_green = np.array([85, 255, 255], dtype=np.uint8)
        grass_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Zero out scoreboard bug (top-left) and broadcast bug (top-right)
        grass_mask[0:int(h * 0.165), 0:int(w * 0.18)] = 0
        grass_mask[0:int(h * 0.15), int(w * 0.88):w] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        grass_closed = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        grass_closed = cv2.morphologyEx(grass_closed, cv2.MORPH_OPEN, kernel, iterations=1)

        turf_ratio = float(np.count_nonzero(grass_closed)) / float(h * w)

        # 2. Largest connected turf contour -> Concave Touchline Polygon via approxPolyDP
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
                # Tight boundary hugging actual pitch grass (epsilon = 0.015 * arcLength)
                pitch_poly = cv2.approxPolyDP(largest, 0.015 * peri, True)
                cv2.drawContours(hull_mask, [pitch_poly], -1, 255, thickness=-1)
                ys = pitch_poly[:, 0, 1]
                min_turf_y_pct = float(np.min(ys)) / h * 100.0
                max_turf_y_pct = float(np.max(ys)) / h * 100.0

        # 3. Detect physical pitch lines (white markings) inside turf mask via Probabilistic Hough Lines
        # White pitch lines inside grass area: low saturation (S < 65) and high value (V > 165)
        white_line_mask = (hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 165) & (hull_mask > 0)
        white_line_u8 = white_line_mask.astype(np.uint8) * 255

        edges = cv2.Canny(white_line_u8, 50, 150, apertureSize=3)
        hough_lines_raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=int(w * 0.04),
            maxLineGap=int(w * 0.02),
        )

        hough_lines: List[Tuple[int, int, int, int]] = []
        if hough_lines_raw is not None:
            for l_entry in hough_lines_raw:
                pts = l_entry.flatten()
                if len(pts) >= 4:
                    lx1, ly1, lx2, ly2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
                    hough_lines.append((lx1, ly1, lx2, ly2))

        # 4. Detect lower-third broadcast graphic bar (bright horizontal rectangle in y=81..95%)
        banner_bbox: Optional[Tuple[int, int, int, int]] = None
        if self.config.use_broadcast_banner_mask:
            y_band_top = int(h * 0.81)
            y_band_bot = int(h * 0.95)
            band_hsv = hsv[y_band_top:y_band_bot, int(w * 0.08):int(w * 0.92)]
            white_bar_mask = (band_hsv[:, :, 1] < 35) & (band_hsv[:, :, 2] > 200)
            row_white_frac = np.mean(white_bar_mask, axis=1)
            if np.max(row_white_frac) > 0.52:
                banner_bbox = (int(w * 0.04), int(h * 0.80), int(w * 0.95), int(h * 0.95))

        # 5. Camera shot classification & perspective slope
        persp_slope = 0.20
        if turf_ratio < 0.45 or min_turf_y_pct > 34.0:
            shot_type = "close_up"
        else:
            shot_type = "wide_tactical"
            col_12 = np.where(grass_closed[:, int(w * 0.12)] > 0)[0]
            col_50 = np.where(grass_closed[:, int(w * 0.50)] > 0)[0]
            col_88 = np.where(grass_closed[:, int(w * 0.88)] > 0)[0]
            y12 = (float(col_12[0]) / h * 100.0) if len(col_12) > 0 else 35.0
            y50 = (float(col_50[0]) / h * 100.0) if len(col_50) > 0 else 20.0
            y88 = (float(col_88[0]) / h * 100.0) if len(col_88) > 0 else 25.0
            if (y12 - y50) > 8.0:
                persp_slope = 0.82
            elif (y88 - y50) > 8.0:
                persp_slope = -0.82

        return {
            "turf_ratio": round(turf_ratio, 3),
            "shot_type": shot_type,
            "persp_slope": round(persp_slope, 2),
            "grass_mask": grass_closed,
            "hull_mask": hull_mask,
            "hull_pts": pitch_poly,
            "pitch_polygon": pitch_poly,
            "hough_lines": hough_lines,
            "min_turf_y_pct": round(min_turf_y_pct, 1),
            "max_turf_y_pct": round(max_turf_y_pct, 1),
            "banner_bbox": banner_bbox,
        }

    @staticmethod
    def get_turf_hull_bounds(
        pitch_geom: Optional[Dict[str, Any]] = None,
        frame_shape: Tuple[int, int] = (1080, 1920),
        x_px: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Returns the top and bottom vertical bounds (in normalized [0.0, 1.0] coordinates)
        of the visible turf convex hull for perspective lateral Y-axis normalization.
        If x_px is provided and hull_pts exists, computes column-specific vertical bounds.
        """
        h = frame_shape[0]
        if pitch_geom is None:
            return 0.15, 0.95

        hull_pts = pitch_geom.get("hull_pts")
        if hull_pts is not None and len(hull_pts) >= 3 and x_px is not None:
            pts = hull_pts.reshape(-1, 2).astype(np.float64)
            n_pts = len(pts)
            y_intersections = []

            for i in range(n_pts):
                p1 = pts[i]
                p2 = pts[(i + 1) % n_pts]
                x1, y1 = float(p1[0]), float(p1[1])
                x2, y2 = float(p2[0]), float(p2[1])

                # Check if vertical line x = x_px crosses this edge segment
                if (x1 <= x_px <= x2) or (x2 <= x_px <= x1):
                    dx = x2 - x1
                    if abs(dx) > 1e-4:
                        t = (x_px - x1) / dx
                        y_val = y1 + t * (y2 - y1)
                        y_intersections.append(y_val)
                    else:
                        y_intersections.extend([y1, y2])

            if len(y_intersections) >= 2:
                min_y = float(min(y_intersections)) / float(h)
                max_y = float(max(y_intersections)) / float(h)
                return float(np.clip(min_y, 0.0, 1.0)), float(np.clip(max_y, 0.0, 1.0))

        min_pct = float(pitch_geom.get("min_turf_y_pct", 15.0)) / 100.0
        max_pct = float(pitch_geom.get("max_turf_y_pct", 95.0)) / 100.0
        return max(0.0, min_pct), min(1.0, max_pct)

    def render_turf_poly_debug(
        self,
        frame: np.ndarray,
        pitch_geom: Dict[str, Any],
        candidate_boxes: List[CandidateBox],
        accepted_actors: List[DetectedPitchActor],
    ) -> np.ndarray:
        """
        Renders `turf_poly_debug.png`:
        - Translucent green fill over the turf convex hull
        - Bright cyan lines for detected Hough pitch markings
        - Accepted player boxes in green with Track ID
        - Rejected candidate boxes in red with exact rejection reason
        - Lower-third banner box (if detected)
        - Master HUD header with summary metrics
        """
        h, w = frame.shape[:2]
        debug_img = frame.copy()

        # 1. Draw translucent green turf polygon
        hull_pts = pitch_geom.get("hull_pts")
        if hull_pts is not None and len(hull_pts) > 2:
            overlay = debug_img.copy()
            cv2.drawContours(overlay, [hull_pts], -1, (0, 200, 50), -1)
            cv2.addWeighted(overlay, 0.22, debug_img, 0.78, 0, debug_img)
            cv2.drawContours(debug_img, [hull_pts], -1, (0, 255, 100), 2, cv2.LINE_AA)

        # 2. Draw detected Hough lines (pitch markings)
        for lx1, ly1, lx2, ly2 in pitch_geom.get("hough_lines", []):
            cv2.line(debug_img, (lx1, ly1), (lx2, ly2), (255, 240, 40), 2, cv2.LINE_AA)

        # 3. Draw lower-third broadcast graphic bar if present
        banner_bbox = pitch_geom.get("banner_bbox")
        if banner_bbox is not None:
            bx1, by1, bx2, by2 = banner_bbox
            cv2.rectangle(debug_img, (bx1, by1), (bx2, by2), (220, 50, 220), 2, cv2.LINE_AA)
            cv2.putText(
                debug_img,
                "LOWER-THIRD BANNER MASK",
                (bx1 + 10, by1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 50, 220),
                2,
                cv2.LINE_AA,
            )

        # 4. Draw Rejected Candidate Boxes in Red
        rejected = [c for c in candidate_boxes if c.status == "rejected"]
        for c in rejected:
            x1, y1, x2, y2 = c.bbox
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 30, 240), 2, cv2.LINE_AA)
            reason_txt = f"REJECT: {c.rejection_reason}"
            (tw, th), _ = cv2.getTextSize(reason_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(debug_img, (x1, max(0, y1 - th - 4)), (x1 + tw + 6, y1), (0, 0, 160), -1)
            cv2.putText(
                debug_img,
                reason_txt,
                (x1 + 3, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 5. Draw Accepted Player Boxes in Green
        for act in accepted_actors:
            x1, y1, x2, y2 = act.bbox
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 240, 80), 2, cv2.LINE_AA)
            label = f"T{act.track_id} ({act.team.upper()})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(debug_img, (x1, max(0, y1 - th - 5)), (x1 + tw + 6, y1), (15, 60, 20), -1)
            cv2.putText(
                debug_img,
                label,
                (x1 + 3, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 120),
                1,
                cv2.LINE_AA,
            )

        # 6. Master Diagnostic Header HUD
        hud_h = 32
        cv2.rectangle(debug_img, (0, 0), (w, hud_h), (16, 20, 28), -1)
        hud_txt = (
            f"TURF POLYGON & GEOMETRY AUDIT | Shot: {pitch_geom.get('shot_type')} | "
            f"Turf Ratio: {pitch_geom.get('turf_ratio'):.1%} | "
            f"Pitch Lines: {len(pitch_geom.get('hough_lines', []))} | "
            f"Accepted: {len(accepted_actors)} | "
            f"Rejected: {len(rejected)}"
        )
        cv2.putText(debug_img, hud_txt, (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 245, 250), 1, cv2.LINE_AA)

        return debug_img
