"""
src/vision/shot_boundary_detector.py
Broadcast Camera Cut and Shot Boundary Detector.

Ensures bidirectional tracking operates safely within continuous camera shots
without jumping across perspective cuts or replay transitions.

Combines:
1. HSV Color Histogram Bhattacharyya distance (detects color palette shifts).
2. Structural Similarity Index (SSIM) on luminance (detects spatial geometry/framing breaks).
3. Cut type classification (HARD_CUT, TRANSITION/DISSOLVE, NO_CUT).
4. Continuous shot boundary interval resolver for safe bidirectional tracklet propagation.
"""

from __future__ import annotations

import enum
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("vision.shot_boundary_detector")


class CutType(str, enum.Enum):
    NO_CUT = "NO_CUT"
    HARD_CUT = "HARD_CUT"
    TRANSITION = "TRANSITION"


class ShotBoundaryDetector:
    """
    Detects camera cuts, dissolves, and broadcast transitions between video frames.
    Provides continuous shot intervals to constrain bidirectional tracking.
    """

    def __init__(
        self,
        hist_threshold: float = 0.35,
        ssim_threshold: float = 0.50,
        combined_threshold: float = 0.40,
        hist_weight: float = 0.55,
        ssim_weight: float = 0.45,
        h_bins: int = 16,
        s_bins: int = 8,
        v_bins: int = 8,
        downsample_size: Tuple[int, int] = (320, 180),
    ):
        """
        Args:
            hist_threshold: Bhattacharyya distance threshold above which a cut is detected (range 0..1).
            ssim_threshold: SSIM similarity threshold below which a cut is detected (range 0..1).
            combined_threshold: Weighted combination threshold for cut classification.
            hist_weight: Weight assigned to histogram distance in combined score.
            ssim_weight: Weight assigned to structural difference in combined score.
            h_bins: Number of hue bins for HSV histogram.
            s_bins: Number of saturation bins for HSV histogram.
            v_bins: Number of value/brightness bins for HSV histogram.
            downsample_size: Width, Height for accelerated metric computation.
        """
        self.hist_threshold = float(hist_threshold)
        self.ssim_threshold = float(ssim_threshold)
        self.combined_threshold = float(combined_threshold)
        self.hist_weight = float(hist_weight)
        self.ssim_weight = float(ssim_weight)
        self.h_bins = int(h_bins)
        self.s_bins = int(s_bins)
        self.v_bins = int(v_bins)
        self.downsample_size = downsample_size

    def compute_hsv_histogram(self, frame: np.ndarray) -> np.ndarray:
        """
        Computes normalized 3D HSV (Hue-Saturation-Value) color histogram.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return np.zeros((self.h_bins, self.s_bins, self.v_bins), dtype=np.float32)

        # Downsample for efficiency
        if self.downsample_size and (frame.shape[1] > self.downsample_size[0] or frame.shape[0] > self.downsample_size[1]):
            img = cv2.resize(frame, self.downsample_size, interpolation=cv2.INTER_AREA)
        else:
            img = frame

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [self.h_bins, self.s_bins, self.v_bins],
            [0, 180, 0, 256, 0, 256],
        )
        # Normalize histogram so sum is 1.0
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        return hist.astype(np.float32)

    def compute_bhattacharyya_distance(self, hist_a: np.ndarray, hist_b: np.ndarray) -> float:
        """
        Computes Bhattacharyya distance between two normalized histograms.
        Returns a float in [0.0, 1.0], where 0.0 means identical distribution.
        """
        if hist_a is None or hist_b is None or hist_a.size == 0 or hist_b.size == 0:
            return 1.0

        dist = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA)
        if math.isnan(dist) or math.isinf(dist):
            return 1.0
        return float(max(0.0, min(1.0, dist)))

    def compute_ssim(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """
        Computes Structural Similarity Index (SSIM) on grayscale luminance.
        Returns a float in [-1.0, 1.0], where 1.0 means identical image structure.
        """
        if frame_a is None or frame_b is None or frame_a.size == 0 or frame_b.size == 0:
            return 0.0

        # Match dimensions if differing
        h_a, w_a = frame_a.shape[:2]
        h_b, w_b = frame_b.shape[:2]

        # Target downsampled size for fast computation
        target_w, target_h = self.downsample_size
        img_a = cv2.resize(frame_a, (target_w, target_h), interpolation=cv2.INTER_AREA)
        img_b = cv2.resize(frame_b, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Grayscale conversion
        if len(img_a.shape) == 3:
            gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        else:
            gray_a = img_a

        if len(img_b.shape) == 3:
            gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        else:
            gray_b = img_b

        # Exact pixel match check
        if np.array_equal(gray_a, gray_b):
            return 1.0

        # SSIM calculation constants
        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2

        f_a = gray_a.astype(np.float64)
        f_b = gray_b.astype(np.float64)

        # 11x11 Gaussian window
        kernel_size = 11
        sigma = 1.5
        mu_a = cv2.GaussianBlur(f_a, (kernel_size, kernel_size), sigma)
        mu_b = cv2.GaussianBlur(f_b, (kernel_size, kernel_size), sigma)

        mu_a_sq = mu_a * mu_a
        mu_b_sq = mu_b * mu_b
        mu_ab = mu_a * mu_b

        sigma_a_sq = cv2.GaussianBlur(f_a * f_a, (kernel_size, kernel_size), sigma) - mu_a_sq
        sigma_b_sq = cv2.GaussianBlur(f_b * f_b, (kernel_size, kernel_size), sigma) - mu_b_sq
        sigma_ab = cv2.GaussianBlur(f_a * f_b, (kernel_size, kernel_size), sigma) - mu_ab

        num = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
        den = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
        den = np.where(den == 0.0, 1e-7, den)

        ssim_map = num / den
        mean_ssim = float(np.mean(ssim_map))
        return float(max(-1.0, min(1.0, mean_ssim)))

    def compute_frame_difference(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluates visual discontinuity metrics between two frames.

        Returns:
            Dict containing:
                'hist_distance': float in [0..1] (Bhattacharyya)
                'ssim': float in [-1..1]
                'structural_diff': float in [0..1] (1 - ssim clamped)
                'combined_diff_score': float in [0..1]
                'is_cut': bool
                'cut_type': str ("NO_CUT", "HARD_CUT", "TRANSITION")
        """
        if frame_a is None or frame_b is None or frame_a.size == 0 or frame_b.size == 0:
            return {
                "hist_distance": 1.0,
                "ssim": 0.0,
                "structural_diff": 1.0,
                "combined_diff_score": 1.0,
                "is_cut": True,
                "cut_type": CutType.HARD_CUT.value,
            }

        hist_a = self.compute_hsv_histogram(frame_a)
        hist_b = self.compute_hsv_histogram(frame_b)
        hist_dist = self.compute_bhattacharyya_distance(hist_a, hist_b)

        ssim_val = self.compute_ssim(frame_a, frame_b)
        structural_diff = max(0.0, min(1.0, 1.0 - max(0.0, ssim_val)))

        # Weighted composite score
        combined_score = (
            self.hist_weight * hist_dist + self.ssim_weight * structural_diff
        )

        # High structural similarity (SSIM >= 0.85) indicates continuous framing within same shot
        if ssim_val >= 0.85:
            is_cut = False
        else:
            is_cut = (
                combined_score >= self.combined_threshold
                or (hist_dist >= self.hist_threshold and ssim_val <= 0.70)
                or ssim_val <= self.ssim_threshold
            )

        if not is_cut:
            cut_type = CutType.NO_CUT
        elif hist_dist >= 0.50 or ssim_val <= 0.35:
            cut_type = CutType.HARD_CUT
        else:
            cut_type = CutType.TRANSITION

        return {
            "hist_distance": round(hist_dist, 4),
            "ssim": round(ssim_val, 4),
            "structural_diff": round(structural_diff, 4),
            "combined_diff_score": round(combined_score, 4),
            "is_cut": is_cut,
            "cut_type": cut_type.value,
        }

    def is_shot_boundary(self, frame_a: np.ndarray, frame_b: np.ndarray) -> bool:
        """Returns True if the transition from frame_a to frame_b represents a camera cut."""
        res = self.compute_frame_difference(frame_a, frame_b)
        return bool(res["is_cut"])

    def detect_shot_boundaries(
        self,
        frames: Union[Sequence[np.ndarray], Dict[int, np.ndarray]],
        fps: float = 30.0,
        detect_gradual_transitions: bool = True,
        transition_window: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Analyzes a sequence of frames and segments into continuous shot intervals.
        Detects both instantaneous hard cuts and gradual dissolves/transitions.

        Args:
            frames: Sequence of frames or dict mapping frame_idx -> image array.
            fps: Video frames per second.
            detect_gradual_transitions: If True, evaluates multi-frame rolling window for dissolves/wipes.
            transition_window: Span of frames over which to detect gradual transitions.

        Returns:
            List of shot segment dicts:
            [
                {
                    "shot_id": int,
                    "start_frame": int,
                    "end_frame": int,
                    "start_time_sec": float,
                    "end_time_sec": float,
                    "frame_count": int,
                    "duration_sec": float,
                }, ...
            ]
        """
        if isinstance(frames, dict):
            sorted_indices = sorted(frames.keys())
            frame_list = [(idx, frames[idx]) for idx in sorted_indices]
        else:
            frame_list = [(i, f) for i, f in enumerate(frames)]

        if not frame_list:
            return []

        if len(frame_list) == 1:
            idx = frame_list[0][0]
            t = idx / fps
            return [
                {
                    "shot_id": 0,
                    "start_frame": idx,
                    "end_frame": idx,
                    "start_time_sec": round(t, 3),
                    "end_time_sec": round(t, 3),
                    "frame_count": 1,
                    "duration_sec": 0.0,
                }
            ]

        shots: List[Dict[str, Any]] = []
        shot_start_idx = frame_list[0][0]
        shot_ref_frame = frame_list[0][1]

        for i in range(len(frame_list) - 1):
            curr_idx, curr_frame = frame_list[i]
            next_idx, next_frame = frame_list[i + 1]

            diff = self.compute_frame_difference(curr_frame, next_frame)
            is_cut = diff["is_cut"]

            # Gradual dissolve / transition sting detection across rolling window
            if not is_cut and detect_gradual_transitions and i >= 2:
                win_back_idx, win_back_frame = frame_list[max(0, i + 1 - transition_window)]
                win_diff = self.compute_frame_difference(win_back_frame, next_frame)
                ref_diff = self.compute_frame_difference(shot_ref_frame, next_frame)

                if (win_diff["hist_distance"] >= self.hist_threshold * 1.05 and win_diff["structural_diff"] >= 0.40) or \
                   (ref_diff["hist_distance"] >= self.hist_threshold * 1.15 and win_diff["combined_diff_score"] >= 0.38):
                    is_cut = True

            if is_cut:
                # End current shot at curr_idx
                start_t = shot_start_idx / fps
                end_t = curr_idx / fps
                count = curr_idx - shot_start_idx + 1
                shots.append(
                    {
                        "shot_id": len(shots),
                        "start_frame": shot_start_idx,
                        "end_frame": curr_idx,
                        "start_time_sec": round(start_t, 3),
                        "end_time_sec": round(end_t, 3),
                        "frame_count": count,
                        "duration_sec": round(end_t - start_t, 3),
                    }
                )
                shot_start_idx = next_idx
                shot_ref_frame = next_frame

        # Append final segment
        last_idx = frame_list[-1][0]
        start_t = shot_start_idx / fps
        end_t = last_idx / fps
        count = last_idx - shot_start_idx + 1
        shots.append(
            {
                "shot_id": len(shots),
                "start_frame": shot_start_idx,
                "end_frame": last_idx,
                "start_time_sec": round(start_t, 3),
                "end_time_sec": round(end_t, 3),
                "frame_count": count,
                "duration_sec": round(end_t - start_t, 3),
            }
        )

        return shots

    @staticmethod
    def get_valid_shot_interval(
        anchor_frame: int,
        shot_segments: Sequence[Dict[str, Any]],
        total_frames: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Determines the safe continuous shot interval [start_frame, end_frame] containing anchor_frame.
        Guarantees bidirectional tracklets do not propagate across shot boundaries.

        Args:
            anchor_frame: The reference event frame index.
            shot_segments: List of shot segments produced by detect_shot_boundaries.
            total_frames: Total sequence frames fallback if segment not found.

        Returns:
            Tuple of (start_frame, end_frame) defining the boundary of continuous shot.
        """
        for shot in shot_segments:
            if shot["start_frame"] <= anchor_frame <= shot["end_frame"]:
                return (int(shot["start_frame"]), int(shot["end_frame"]))

        # Fallback if no matching segment or empty segments list
        if total_frames is not None and total_frames > 0:
            return (0, total_frames - 1)
        return (anchor_frame, anchor_frame)
