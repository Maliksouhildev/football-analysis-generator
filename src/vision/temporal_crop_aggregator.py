"""
src/vision/temporal_crop_aggregator.py
Temporal Multi-Frame Crop Aggregator (Fixing Flaw C: Single-Frame 35x80 Blurring & VLM Hallucination).

Solves the low-resolution 35x80 single-frame limitation by evaluating all bounding box
occurrences of a player tracklet across a 10-30s sequence.
Ranks crops by Laplacian variance (sharpness) and bounding box pixel area (proximity to camera),
enforces temporal spacing across distinct viewpoints (front, back/jersey numbers, side profiles),
and stitches the top 5 sharpest frames into a composite inspection strip (e.g. 1120x224px).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("vision.temporal_crop_aggregator")


class TemporalCropAggregator:
    """
    Extracts, ranks, and stitches the sharpest temporally-diverse crops for each
    tracked player across a video sequence for high-precision OCR and VLM biometric recognition.
    """

    def __init__(
        self,
        num_samples: int = 5,
        target_height: int = 224,
        min_crop_h: int = 30,
        min_crop_w: int = 15,
        min_frame_spacing: int = 6,
    ):
        """
        Args:
            num_samples: Number of distinct temporal crops to include in the stitched strip.
            target_height: Uniform height in pixels of the output composite strip.
            min_crop_h: Minimum raw crop height in pixels to qualify.
            min_crop_w: Minimum raw crop width in pixels to qualify.
            min_frame_spacing: Minimum frame index gap between selected crops to guarantee multi-angle diversity.
        """
        self.num_samples = max(1, int(num_samples))
        self.target_height = max(32, int(target_height))
        self.min_crop_h = max(10, int(min_crop_h))
        self.min_crop_w = max(5, int(min_crop_w))
        self.min_frame_spacing = max(1, int(min_frame_spacing))

    def extract_sharpest_crops(
        self,
        track_id: int,
        sequence_frames: Dict[int, np.ndarray],
        track_boxes: Dict[int, Sequence[float]],
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Extracts and scores candidate crops across all available frames.

        Returns:
            List of (frame_idx, score, crop_bgr) sorted by quality with temporal diversity.
        """
        scored_crops: List[Tuple[float, int, np.ndarray]] = []

        for f_idx, box in track_boxes.items():
            frame = sequence_frames.get(f_idx)
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                continue

            fh, fw = frame.shape[:2]
            if len(box) < 4:
                continue

            # Support normalized or pixel coordinates
            b = [float(v) for v in box[:4]]
            if max(b) <= 1.05 and fw > 10 and fh > 10:
                x1 = int(round(b[0] * fw))
                y1 = int(round(b[1] * fh))
                x2 = int(round(b[2] * fw))
                y2 = int(round(b[3] * fh))
            else:
                x1, y1, x2, y2 = [int(round(v)) for v in b]

            # Boundary clamps
            x1 = max(0, min(fw - 1, x1))
            x2 = max(0, min(fw, x2))
            y1 = max(0, min(fh - 1, y1))
            y2 = max(0, min(fh, y2))

            crop_w = x2 - x1
            crop_h = y2 - y1

            if crop_h < self.min_crop_h or crop_w < self.min_crop_w:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Compute sharpness via discrete Laplacian variance
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            area = float(crop_h * crop_w)

            # Combined visual metric: sharpness scaled by proximity / area
            score = float(sharpness * np.sqrt(area))
            scored_crops.append((score, f_idx, crop))

        if not scored_crops:
            return []

        # Sort candidate pool descending by score
        scored_crops.sort(key=lambda x: x[0], reverse=True)

        # Enforce temporal diversity: do not pick 5 identical adjacent frames
        selected: List[Tuple[int, float, np.ndarray]] = []
        selected_frames: List[int] = []

        for score, f_idx, crop in scored_crops:
            if len(selected) >= self.num_samples:
                break

            # Check minimum frame separation from all currently selected frames
            if any(abs(f_idx - sf) < self.min_frame_spacing for sf in selected_frames):
                continue

            selected.append((f_idx, score, crop))
            selected_frames.append(f_idx)

        # If strict spacing left us with fewer than num_samples, backfill with remaining best
        if len(selected) < self.num_samples:
            for score, f_idx, crop in scored_crops:
                if len(selected) >= self.num_samples:
                    break
                if f_idx not in selected_frames:
                    selected.append((f_idx, score, crop))
                    selected_frames.append(f_idx)

        # Return sorted chronologically by frame index for intuitive viewing
        selected.sort(key=lambda x: x[0])
        return selected

    def extract_sharpest_strip(
        self,
        track_id: int,
        sequence_frames: Dict[int, np.ndarray],
        track_boxes: Dict[int, Sequence[float]],
    ) -> Optional[np.ndarray]:
        """
        Extracts the top distinct temporal crops and stitches them horizontally
        into a uniform high-resolution inspection strip.

        Returns:
            Composite BGR image strip of shape (target_height, total_width, 3),
            or None if no valid crops were found.
        """
        selected = self.extract_sharpest_crops(track_id, sequence_frames, track_boxes)
        if not selected:
            return None

        resized: List[np.ndarray] = []
        for _, _, crop in selected:
            h, w = crop.shape[:2]
            scale = self.target_height / max(1, h)
            rw = max(1, int(round(w * scale)))
            # Cap extreme widths to preserve reasonable aspect ratio
            rw = min(rw, self.target_height * 3)

            r_crop = cv2.resize(
                crop,
                (rw, self.target_height),
                interpolation=cv2.INTER_LANCZOS4,
            )
            resized.append(r_crop)

        if not resized:
            return None

        # Horizontal stitch
        strip = np.hstack(resized)
        return strip
