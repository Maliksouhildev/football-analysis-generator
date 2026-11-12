"""
src/vision/probabilistic_bidi_tracker.py
Probabilistic Bidirectional Tracker (Fixing Flaw B: MOT Bounding Box Identity Swaps).

Replaces naive Confidence = 1.0 backward tracking with a probabilistic prior that decays
dynamically when player bounding boxes intersect (IoU > 0.35) or suffer occlusions.
When two teammates wearing identical kits intersect with IoU > 0.35 for > 0.4s, the system
downgrades the track status from ANCHORED_GROUND_TRUTH to PROBABILISTIC_PRIOR (confidence <= 0.55),
forcing Hungarian topology and visual biometric solvers to independently re-verify identity.
"""

from __future__ import annotations

import copy
import enum
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("vision.probabilistic_bidi_tracker")


class TrackStatus(str, enum.Enum):
    ANCHORED_GROUND_TRUTH = "ANCHORED_GROUND_TRUTH"
    LOCKED_FACT = "LOCKED_FACT"
    PROBABILISTIC_PRIOR = "PROBABILISTIC_PRIOR"
    UNVERIFIED = "UNVERIFIED"


def compute_box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """
    Computes Intersection-over-Union (IoU) between two boxes [x1, y1, x2, y2].
    Handles zero area, empty lists, and non-overlapping geometries safely.
    """
    if box_a is None or box_b is None or len(box_a) < 4 or len(box_b) < 4:
        return 0.0

    ax1, ay1, ax2, ay2 = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
    bx1, by1, bx2, by2 = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])

    # Standardize so x1 <= x2, y1 <= y2
    if ax1 > ax2:
        ax1, ax2 = ax2, ax1
    if ay1 > ay2:
        ay1, ay2 = ay2, ay1
    if bx1 > bx2:
        bx1, bx2 = bx2, bx1
    if by1 > by2:
        by1, by2 = by2, by1

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 1e-9:
        return 0.0

    return float(inter_area / union_area)


class ProbabilisticBidirectionalTracker:
    """
    Tracks player bounding boxes backward (or forward) from an anchor frame
    while penalizing identity ambiguity during teammate overlaps and occlusions.
    """

    def __init__(
        self,
        iou_threshold: float = 0.35,
        decay_rate: float = 0.60,
        overlap_time_threshold_sec: float = 0.40,
        occlusion_time_threshold_sec: float = 0.50,
        fps: float = 30.0,
        gt_confidence_threshold: float = 0.85,
    ):
        self.iou_threshold = float(iou_threshold)
        self.decay_rate = float(decay_rate)
        self.fps = max(1.0, float(fps))
        self.overlap_frame_threshold = int(math.ceil(self.fps * overlap_time_threshold_sec))
        self.occlusion_frame_threshold = int(math.ceil(self.fps * occlusion_time_threshold_sec))
        self.gt_confidence_threshold = float(gt_confidence_threshold)

    def backward_propagate(
        self,
        ball_reception_track: Dict[str, Any],
        shot_frame_buffer: Optional[List[np.ndarray]],
        all_tracks_per_frame: Dict[int, List[Dict[str, Any]]],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Propagates track backward from reception / freeze frame.

        Args:
            ball_reception_track: Dict containing at least 'track_id', and optionally 'team', 'box'.
            shot_frame_buffer: Optional list of frame images (for visual re-ID).
            all_tracks_per_frame: Dict mapping frame_idx -> list of track dicts.

        Returns:
            Dict mapping frame_idx -> {
                'box': list[float],
                'confidence': float,
                'is_ground_truth': bool,
                'status': str,
                'overlap_duration_sec': float,
                'occluded': bool
            }
        """
        return self._propagate_sequence(
            target_track=ball_reception_track,
            all_tracks_per_frame=all_tracks_per_frame,
            reverse=True,
        )

    def forward_propagate(
        self,
        anchor_track: Dict[str, Any],
        all_tracks_per_frame: Dict[int, List[Dict[str, Any]]],
    ) -> Dict[int, Dict[str, Any]]:
        """Propagates track forward from anchor frame."""
        return self._propagate_sequence(
            target_track=anchor_track,
            all_tracks_per_frame=all_tracks_per_frame,
            reverse=False,
        )

    def _propagate_sequence(
        self,
        target_track: Dict[str, Any],
        all_tracks_per_frame: Dict[int, List[Dict[str, Any]]],
        reverse: bool = True,
    ) -> Dict[int, Dict[str, Any]]:
        propagated: Dict[int, Dict[str, Any]] = {}
        curr_confidence = 1.0
        target_tid = target_track.get("track_id")
        target_team = target_track.get("team")

        consecutive_overlaps = 0
        consecutive_missing = 0
        last_known_box: Optional[List[float]] = target_track.get("box")

        frame_indices = sorted(all_tracks_per_frame.keys(), reverse=reverse)
        anchor_frame = target_track.get("frame_idx")
        if anchor_frame is not None:
            if reverse:
                frame_indices = [f for f in frame_indices if f <= anchor_frame]
            else:
                frame_indices = [f for f in frame_indices if f >= anchor_frame]

        for frame_idx in frame_indices:
            frame_tracks = all_tracks_per_frame[frame_idx]
            match = next((t for t in frame_tracks if t.get("track_id") == target_tid), None)

            if match is None:
                consecutive_missing += 1
                occluded = True
                box = last_known_box if last_known_box is not None else [0.0, 0.0, 0.0, 0.0]

                # Check if missing for longer than occlusion threshold (>0.5s)
                if consecutive_missing >= self.occlusion_frame_threshold:
                    curr_confidence = min(curr_confidence * 0.85, 0.50)
                else:
                    curr_confidence *= 0.95
            else:
                consecutive_missing = 0
                occluded = False
                box = list(match.get("box", [0.0, 0.0, 0.0, 0.0]))
                last_known_box = box

                match_team = match.get("team", target_team)

                # Check overlap with same-team bounding boxes
                had_same_team_overlap = False
                max_iou = 0.0
                for other in frame_tracks:
                    if other.get("track_id") != target_tid:
                        other_team = other.get("team")
                        # Overlap penalty triggers on teammates or ambiguous team kits
                        if other_team == match_team or other_team is None or match_team is None:
                            iou = compute_box_iou(box, other.get("box", [0, 0, 0, 0]))
                            if iou > max_iou:
                                max_iou = iou
                            if iou > self.iou_threshold:
                                had_same_team_overlap = True

                # Apply decay at most once per frame when overlap occurs
                if had_same_team_overlap:
                    consecutive_overlaps += 1
                    curr_confidence *= self.decay_rate
                else:
                    consecutive_overlaps = max(0, consecutive_overlaps - 1)

                # Overlap penalty threshold: > 0.4s sustained intersection
                if consecutive_overlaps >= self.overlap_frame_threshold:
                    # Downgrade to PROBABILISTIC_PRIOR (confidence <= 0.55)
                    curr_confidence = min(curr_confidence, 0.55)

            curr_confidence = max(0.10, min(1.0, curr_confidence))
            overlap_duration = consecutive_overlaps / self.fps

            is_gt = (
                curr_confidence >= self.gt_confidence_threshold
                and consecutive_overlaps < self.overlap_frame_threshold
                and not occluded
            )

            status = (
                TrackStatus.ANCHORED_GROUND_TRUTH.value
                if is_gt
                else TrackStatus.PROBABILISTIC_PRIOR.value
            )

            propagated[frame_idx] = {
                "box": box,
                "confidence": round(curr_confidence, 4),
                "is_ground_truth": is_gt,
                "status": status,
                "overlap_duration_sec": round(overlap_duration, 3),
                "occluded": occluded,
            }

        return propagated
