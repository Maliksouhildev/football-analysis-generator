"""
BoT-SORT Tracker Wrapper with Camera Motion Compensation (GMC).
Runs at 15 FPS with stride=2 and linearly interpolates to 30 FPS for smooth rendering.
Explicitly sets with_reid=False to eliminate heavy secondary neural network overhead on CPU.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from ultralytics.trackers.bot_sort import BOTSORT


@dataclass
class TrackedObject:
    """Bounding box detection state on a specific frame."""
    frame_idx: int
    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    is_interpolated: bool = False


@dataclass
class TrackletCrop:
    """A crop extracted from a specific moment in a tracklet's life."""
    crop_img: np.ndarray
    frame_idx: int
    position_pct: float  # e.g., 0.2, 0.5, 0.8
    sharpness: float
    bbox: List[float]
    label: str  # "early", "peak", "late"


@dataclass
class TrackletLifecycle:
    """Complete temporal history and multi-angle biometric crops of a tracked entity."""
    track_id: int
    start_frame: int
    end_frame: int
    class_id: int = 0
    frame_indices: List[int] = field(default_factory=list)
    bboxes: List[List[float]] = field(default_factory=list)  # [x1, y1, x2, y2]
    confidences: List[float] = field(default_factory=list)
    is_interpolated: List[bool] = field(default_factory=list)

    # Identity and team attributes assigned downstream
    team: str = "unknown"  # "home", "away", "gk", "referee"
    assigned_name: Optional[str] = None
    assigned_number: Optional[int] = None
    assigned_confidence: float = 0.0

    @property
    def total_frames(self) -> int:
        return len(self.frame_indices)

    def duration_seconds(self, fps: float = 30.0) -> float:
        return self.total_frames / max(1.0, fps)

    def get_average_center(self) -> Tuple[float, float]:
        """Compute average center coordinates (cx, cy) over tracklet lifetime."""
        if not self.bboxes:
            return 0.0, 0.0
        cxs = [(b[0] + b[2]) / 2.0 for b in self.bboxes]
        cys = [(b[1] + b[3]) / 2.0 for b in self.bboxes]
        return float(np.mean(cxs)), float(np.mean(cys))

    def extract_3_angle_crops(
        self,
        frame_cache: Dict[int, np.ndarray],
        min_box_dim: int = 24,
    ) -> List[TrackletCrop]:
        """
        Extract 3 distinct temporal angle crops:
        1) Early (~20% lifespan)
        2) Peak sharpness and area (optimal front/back visibility)
        3) Late (~80% lifespan, spaced >= 10 frames from peak)
        """
        non_interp_entries = [
            (f_idx, bbox)
            for f_idx, bbox, interp in zip(self.frame_indices, self.bboxes, self.is_interpolated)
            if not interp and f_idx in frame_cache
        ]

        if not non_interp_entries:
            # Fallback to any cached frames
            non_interp_entries = [
                (f_idx, bbox)
                for f_idx, bbox in zip(self.frame_indices, self.bboxes)
                if f_idx in frame_cache
            ]

        if not non_interp_entries:
            return []

        # Score each entry by sharpness * area
        scored_entries = []
        for f_idx, bbox in non_interp_entries:
            frame = frame_cache[f_idx]
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop_w, crop_h = x2 - x1, y2 - y1

            if crop_w < min_box_dim or crop_h < min_box_dim:
                continue

            crop = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            area = crop_w * crop_h
            score = sharpness * (area ** 0.5)

            scored_entries.append((f_idx, bbox, crop, sharpness, score))

        if not scored_entries:
            return []

        # Find Peak sharpness/size
        peak_idx = int(np.argmax([s[4] for s in scored_entries]))
        peak_entry = scored_entries[peak_idx]

        # Early candidate: near the first 25% of timeline
        early_target = len(scored_entries) // 4
        early_candidates = scored_entries[:max(1, early_target + 1)]
        early_entry = max(early_candidates, key=lambda s: s[3])

        # Late candidate: near the last 25% of timeline, spaced at least 5 indices away
        late_target = (len(scored_entries) * 3) // 4
        late_candidates = scored_entries[min(len(scored_entries) - 1, late_target):]
        if not late_candidates:
            late_candidates = scored_entries[-1:]
        late_entry = max(late_candidates, key=lambda s: s[3])

        total_span = max(1, len(self.frame_indices))

        crops = [
            TrackletCrop(
                crop_img=early_entry[2],
                frame_idx=early_entry[0],
                position_pct=round((early_entry[0] - self.start_frame) / total_span, 2),
                sharpness=round(early_entry[3], 1),
                bbox=early_entry[1],
                label="early",
            ),
            TrackletCrop(
                crop_img=peak_entry[2],
                frame_idx=peak_entry[0],
                position_pct=round((peak_entry[0] - self.start_frame) / total_span, 2),
                sharpness=round(peak_entry[3], 1),
                bbox=peak_entry[1],
                label="peak",
            ),
            TrackletCrop(
                crop_img=late_entry[2],
                frame_idx=late_entry[0],
                position_pct=round((late_entry[0] - self.start_frame) / total_span, 2),
                sharpness=round(late_entry[3], 1),
                bbox=late_entry[1],
                label="late",
            ),
        ]
        return crops


class _DetectionsWrapper:
    """NumPy detection structure compatible with Ultralytics tracker update."""
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy.astype(np.float32)
        if len(xyxy) > 0:
            x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
            self.xywh = np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], axis=1)
        else:
            self.xywh = np.empty((0, 4), dtype=np.float32)
        self.conf = conf.astype(np.float32)
        self.cls = cls.astype(np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx):
        return _DetectionsWrapper(self.xyxy[idx], self.conf[idx], self.cls[idx])


class BoTSORTTracker:
    """
    Production BoT-SORT Tracker.
    Features:
    - with_reid=False: Zero secondary neural network overhead on CPU.
    - gmc_method="sparseOptFlow": Camera Motion Compensation via sparse optical flow.
    - Stride=2 (15 FPS inference) with linear bounding box interpolation to 30 FPS.
    - Tracklet lifecycle tracking across the duration of a camera shot.
    """

    def __init__(
        self,
        stride: int = 2,
        track_high_thresh: float = 0.45,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.50,
        track_buffer: int = 25,
        match_thresh: float = 0.85,
    ):
        self.stride = stride
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh

        self.tracker = self._build_tracker()
        self.lifecycles: Dict[int, TrackletLifecycle] = {}
        self._last_strided_tracks: Dict[int, List[float]] = {}
        self._last_strided_frame_idx: Optional[int] = None

    def _build_tracker(self) -> BOTSORT:
        args = SimpleNamespace(
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh,
            new_track_thresh=self.new_track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            fuse_score=True,
            gmc_method="sparseOptFlow",  # Sparse Optical Flow CMC
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            with_reid=False,  # CRITICAL: Disable heavy Re-ID neural net on CPU
            model="auto",
        )
        return BOTSORT(args)

    def reset(self):
        """Reset tracker state upon camera cut."""
        self.tracker = self._build_tracker()
        self.lifecycles.clear()
        self._last_strided_tracks.clear()
        self._last_strided_frame_idx = None

    def update_strided(
        self,
        frame: np.ndarray,
        raw_boxes_xyxy: np.ndarray,
        confidences: np.ndarray,
        classes: np.ndarray,
        frame_idx: int,
    ) -> List[TrackedObject]:
        """
        Execute BoT-SORT update on an active strided frame.
        """
        if len(raw_boxes_xyxy) == 0:
            dets = _DetectionsWrapper(np.empty((0, 4)), np.empty(0), np.empty(0))
        else:
            dets = _DetectionsWrapper(raw_boxes_xyxy, confidences, classes)

        # BoT-SORT update with sparse optical flow on current frame
        tracks = self.tracker.update(dets, img=frame)
        current_tracked_objects: List[TrackedObject] = []
        current_frame_boxes: Dict[int, List[float]] = {}

        if tracks is not None and len(tracks) > 0:
            for row in tracks:
                # row format: [x1, y1, x2, y2, track_id, score, cls, idx]
                x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                track_id = int(row[4])
                score = float(row[5])
                cls_id = int(row[6])
                bbox = [x1, y1, x2, y2]

                current_frame_boxes[track_id] = bbox
                obj = TrackedObject(
                    frame_idx=frame_idx,
                    track_id=track_id,
                    bbox=bbox,
                    confidence=score,
                    class_id=cls_id,
                    is_interpolated=False,
                )
                current_tracked_objects.append(obj)

                # Update lifecycle
                if track_id not in self.lifecycles:
                    self.lifecycles[track_id] = TrackletLifecycle(
                        track_id=track_id,
                        start_frame=frame_idx,
                        end_frame=frame_idx,
                        class_id=cls_id,
                        frame_indices=[frame_idx],
                        bboxes=[bbox],
                        confidences=[score],
                        is_interpolated=[False],
                    )
                else:
                    lc = self.lifecycles[track_id]
                    lc.end_frame = frame_idx
                    lc.frame_indices.append(frame_idx)
                    lc.bboxes.append(bbox)
                    lc.confidences.append(score)
                    lc.is_interpolated.append(False)

        self._last_strided_tracks = current_frame_boxes
        self._last_strided_frame_idx = frame_idx
        return current_tracked_objects

    def interpolate_intermediate(
        self,
        frame_idx: int,
        next_strided_idx: int,
        next_boxes: Dict[int, List[float]],
    ) -> List[TrackedObject]:
        """
        Linearly interpolate bounding boxes between previous and next strided frame.
        """
        if self._last_strided_frame_idx is None or self._last_strided_frame_idx >= next_strided_idx:
            return []

        t0 = self._last_strided_frame_idx
        t1 = next_strided_idx
        alpha = float(frame_idx - t0) / float(t1 - t0)

        interpolated_objects: List[TrackedObject] = []
        for track_id, box1 in next_boxes.items():
            if track_id in self._last_strided_tracks:
                box0 = self._last_strided_tracks[track_id]
                # Linear blend of (x1, y1, x2, y2)
                interp_box = [
                    float(box0[i] + alpha * (box1[i] - box0[i]))
                    for i in range(4)
                ]

                obj = TrackedObject(
                    frame_idx=frame_idx,
                    track_id=track_id,
                    bbox=interp_box,
                    confidence=0.85,
                    class_id=0,
                    is_interpolated=True,
                )
                interpolated_objects.append(obj)

                # Add to lifecycle
                if track_id in self.lifecycles:
                    lc = self.lifecycles[track_id]
                    lc.frame_indices.append(frame_idx)
                    lc.bboxes.append(interp_box)
                    lc.confidences.append(0.85)
                    lc.is_interpolated.append(True)

        return interpolated_objects
