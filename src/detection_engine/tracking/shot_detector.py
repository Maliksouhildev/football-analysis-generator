"""
Shot Boundary Detector.
Detects camera cuts (switches between tactical wide camera, tight close-ups, and replays)
using downscaled HSV color histogram Bhattacharyya distance and green pitch turf percentage.
Runs in < 0.5 ms per frame on CPU.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class ShotSegment:
    shot_id: int
    start_frame: int
    end_frame: int
    fps: float
    shot_type: str  # "tactical_wide" or "close_up_or_replay"
    avg_turf_pct: float
    keyframe_index: int
    total_frames: int = field(init=False)

    def __post_init__(self):
        self.total_frames = max(1, self.end_frame - self.start_frame + 1)

    @property
    def duration_sec(self) -> float:
        return self.total_frames / max(1.0, self.fps)


class ShotBoundaryDetector:
    """
    Real-time, lightweight Shot Boundary Detector.
    Uses downscaled (320x180) 2D HSV color histograms and green turf segmentation
    to flag broadcast camera cuts and categorize shots.
    """

    def __init__(
        self,
        hist_thresh: float = 0.42,
        turf_diff_thresh: float = 0.28,
        min_tactical_turf_pct: float = 0.35,
        min_shot_duration_frames: int = 15,
    ):
        self.hist_thresh = hist_thresh
        self.turf_diff_thresh = turf_diff_thresh
        self.min_tactical_turf_pct = min_tactical_turf_pct
        self.min_shot_duration_frames = min_shot_duration_frames

        self._prev_hist: Optional[np.ndarray] = None
        self._prev_turf_pct: Optional[float] = None
        self._curr_shot_id: int = 0
        self._shot_start_frame: int = 0
        self._shot_turf_samples: List[float] = []

    def compute_frame_metrics(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Downscale frame and compute 2D HS histogram + green turf percentage."""
        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # 2D Histogram on Hue (30 bins) and Saturation (32 bins)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Green turf mask: Hue in [32, 85], Sat >= 35, Val >= 35
        turf_mask = cv2.inRange(hsv, (32, 35, 35), (85, 255, 255))
        turf_pct = float(np.count_nonzero(turf_mask)) / (320.0 * 180.0)

        return hist, turf_pct

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[bool, float, float]:
        """
        Evaluate a single frame against the previous frame.
        Returns (is_cut, hist_distance, turf_pct).
        """
        hist, turf_pct = self.compute_frame_metrics(frame)

        if self._prev_hist is None:
            self._prev_hist = hist
            self._prev_turf_pct = turf_pct
            self._shot_start_frame = frame_idx
            self._shot_turf_samples = [turf_pct]
            return False, 0.0, turf_pct

        # Bhattacharyya distance in [0, 1] (0 = identical, 1 = completely different)
        hist_dist = float(cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        turf_diff = abs(turf_pct - (self._prev_turf_pct or 0.0))

        # Check cut condition
        is_cut = (hist_dist > self.hist_thresh) or (turf_diff > self.turf_diff_thresh)

        # Enforce minimum shot duration to avoid flickering on rapid lighting changes
        frames_in_shot = frame_idx - self._shot_start_frame
        if is_cut and frames_in_shot < self.min_shot_duration_frames:
            is_cut = False

        if is_cut:
            self._curr_shot_id += 1
            self._shot_start_frame = frame_idx
            self._shot_turf_samples = [turf_pct]
        else:
            self._shot_turf_samples.append(turf_pct)

        self._prev_hist = hist
        self._prev_turf_pct = turf_pct
        return is_cut, hist_dist, turf_pct

    def segment_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        stride: int = 1,
    ) -> List[ShotSegment]:
        """
        Scan a video and return all segmented shots with classified types.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_video_frames = min(total_video_frames, max_frames)

        shots: List[ShotSegment] = []
        shot_id = 0
        current_start = 0
        turf_accumulator: List[float] = []

        self._prev_hist = None
        self._prev_turf_pct = None

        frame_idx = 0
        while cap.isOpened() and (max_frames is None or frame_idx < max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            if stride > 1 and (frame_idx % stride != 0) and frame_idx > 0:
                frame_idx += 1
                continue

            hist, turf_pct = self.compute_frame_metrics(frame)

            is_cut = False
            if self._prev_hist is not None:
                hist_dist = float(cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                turf_diff = abs(turf_pct - (self._prev_turf_pct or 0.0))
                if (hist_dist > self.hist_thresh or turf_diff > self.turf_diff_thresh):
                    if (frame_idx - current_start) >= self.min_shot_duration_frames:
                        is_cut = True

            if is_cut and frame_idx > current_start:
                avg_turf = float(np.mean(turf_accumulator)) if turf_accumulator else 0.0
                shot_type = "tactical_wide" if avg_turf >= self.min_tactical_turf_pct else "close_up_or_replay"
                keyframe = current_start + (frame_idx - current_start) // 2

                shots.append(ShotSegment(
                    shot_id=shot_id,
                    start_frame=current_start,
                    end_frame=frame_idx - 1,
                    fps=fps,
                    shot_type=shot_type,
                    avg_turf_pct=round(avg_turf, 3),
                    keyframe_index=keyframe,
                ))

                shot_id += 1
                current_start = frame_idx
                turf_accumulator = [turf_pct]
            else:
                turf_accumulator.append(turf_pct)

            self._prev_hist = hist
            self._prev_turf_pct = turf_pct
            frame_idx += 1

        # Final segment
        if frame_idx > current_start:
            avg_turf = float(np.mean(turf_accumulator)) if turf_accumulator else 0.0
            shot_type = "tactical_wide" if avg_turf >= self.min_tactical_turf_pct else "close_up_or_replay"
            keyframe = current_start + (frame_idx - current_start) // 2

            shots.append(ShotSegment(
                shot_id=shot_id,
                start_frame=current_start,
                end_frame=frame_idx - 1,
                fps=fps,
                shot_type=shot_type,
                avg_turf_pct=round(avg_turf, 3),
                keyframe_index=keyframe,
            ))

        cap.release()
        return shots
