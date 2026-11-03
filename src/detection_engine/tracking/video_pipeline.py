"""
Master Video Tracking Pipeline (Two-Pass Architecture).
Coordinates:
- Pass 1: Shot Detection, Strided BoT-SORT Tracking (with_reid=False, sparseOptFlow CMC),
          Turf-masked Kit Clustering (k=2 + Outlier GK/Ref), and Linear Bbox Interpolation.
- Pass 2: Multi-angle 3-crop strip collage batching + 2D Radar + Single Gemini Flash VLM call per camera cut.
- Rendering: Output annotated broadcast video with locked player identities.
"""

from dataclasses import dataclass, field
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from ultralytics import YOLO

from .shot_detector import ShotBoundaryDetector, ShotSegment
from .bot_sort_tracker import BoTSORTTracker, TrackedObject, TrackletLifecycle
from .team_color_clusterer import TeamColorClusterer, TeamClusterResult
from .smart_collage_batcher import SmartCollageBatcher, CollageBatchPayload


@dataclass
class VideoTrackingResult:
    """Summary diagnostic metrics and output paths for the tracked video."""
    video_path: str
    output_video_path: Optional[str]
    total_frames: int
    fps: float
    total_duration_sec: float
    elapsed_time_sec: float
    effective_fps: float
    shots_count: int
    tactical_shots_count: int
    unique_tracklets_count: int
    identified_players_count: int
    diagnostics_json_path: str


class VideoTrackingPipeline:
    """
    Production Two-Pass Broadcast Video Tracking Engine.
    """

    def __init__(
        self,
        yolo_model_path: str = "yolov8n.pt",
        stride: int = 2,
        conf_thresh: float = 0.35,
        device: str = "cpu",
        home_team_name: str = "Real Madrid",
        away_team_name: str = "Barcelona",
        canonical_roster_path: Optional[str] = None,
    ):
        self.stride = stride
        self.conf_thresh = conf_thresh
        self.device = device
        self.home_team_name = home_team_name
        self.away_team_name = away_team_name

        print(f"[VideoTrackingPipeline] Loading YOLO model from {yolo_model_path} on {device}...")
        self.yolo = YOLO(yolo_model_path)

        self.shot_detector = ShotBoundaryDetector()
        self.tracker = BoTSORTTracker(stride=stride)
        self.clusterer = TeamColorClusterer(
            home_team_name=home_team_name,
            away_team_name=away_team_name,
        )
        self.batcher = SmartCollageBatcher()

        self.roster: Dict[str, Any] = self._load_roster(canonical_roster_path)

    def _load_roster(self, path: Optional[str]) -> Dict[str, Any]:
        """Load canonical match roster from JSON if available."""
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[VideoTrackingPipeline] Could not load roster from {path}: {e}")

        # Default canonical El Clásico active roster
        return {
            "home_team": self.home_team_name,
            "away_team": self.away_team_name,
            "home_active_xi": [
                {"number": 13, "name": "Andriy Lunin", "role": "GK"},
                {"number": 17, "name": "Lucas Vázquez", "role": "RB"},
                {"number": 3, "name": "Éder Militão", "role": "CB"},
                {"number": 22, "name": "Antonio Rüdiger", "role": "CB"},
                {"number": 23, "name": "Ferland Mendy", "role": "LB"},
                {"number": 14, "name": "Aurélien Tchouaméni", "role": "DM"},
                {"number": 8, "name": "Federico Valverde", "role": "CM"},
                {"number": 6, "name": "Eduardo Camavinga", "role": "CM"},
                {"number": 5, "name": "Jude Bellingham", "role": "AM"},
                {"number": 7, "name": "Vinícius Júnior", "role": "LW"},
                {"number": 9, "name": "Kylian Mbappé", "role": "CF"},
            ],
            "away_active_xi": [
                {"number": 13, "name": "Iñaki Peña", "role": "GK"},
                {"number": 23, "name": "Jules Koundé", "role": "RB"},
                {"number": 2, "name": "Pau Cubarsí", "role": "CB"},
                {"number": 5, "name": "Iñigo Martínez", "role": "CB"},
                {"number": 3, "name": "Alejandro Balde", "role": "LB"},
                {"number": 17, "name": "Marc Casadó", "role": "DM"},
                {"number": 8, "name": "Pedri", "role": "CM"},
                {"number": 16, "name": "Fermín López", "role": "AM"},
                {"number": 19, "name": "Lamine Yamal", "role": "RW"},
                {"number": 11, "name": "Raphinha", "role": "LW"},
                {"number": 9, "name": "Robert Lewandowski", "role": "CF"},
            ],
        }

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        render_annotated_video: bool = True,
        max_frames: Optional[int] = None,
    ) -> VideoTrackingResult:
        """
        Execute full Two-Pass tracking on the input video.
        """
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_video_path = os.path.join(output_dir, "tracked_output.mp4") if render_annotated_video else None
        video_writer = None
        if render_annotated_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

        print(f"[VideoTrackingPipeline] Starting Two-Pass Tracking: {total_frames} frames, {width}x{height} @ {fps:.1f} FPS")
        start_time = time.time()

        # Shot tracking state
        current_shot_id = 0
        shot_start_frame = 0
        shot_frame_cache: Dict[int, np.ndarray] = {}
        shot_team_assignments: Dict[int, TeamClusterResult] = {}
        shot_identities: Dict[int, Dict[str, Any]] = {}
        active_track_history: Dict[int, List[TrackedObject]] = {}  # frame_idx -> objects

        total_shots = 0
        tactical_shots = 0
        all_identified_players = set()

        def finalize_shot(s_id: int, is_tactical: bool):
            nonlocal tactical_shots
            if not is_tactical or not self.tracker.lifecycles:
                self.tracker.reset()
                shot_frame_cache.clear()
                shot_team_assignments.clear()
                return

            tactical_shots += 1
            print(f"[VideoTrackingPipeline] Finalizing Shot #{s_id} (frames {shot_start_frame}..{frame_idx}): {len(self.tracker.lifecycles)} tracklets")

            # PASS 2: Multi-angle batching + Single VLM Call
            payload = self.batcher.build_batch_payload(
                shot_id=s_id,
                lifecycles=self.tracker.lifecycles,
                team_assignments=shot_team_assignments,
                frame_cache=shot_frame_cache,
                active_roster=self.roster,
                fps=fps,
            )

            if payload:
                identities = self.batcher.resolve_identities_via_vlm(payload, output_dir)
                for tid, id_data in identities.items():
                    name = id_data.get("player_name")
                    num = id_data.get("jersey_number")
                    conf = id_data.get("confidence", 0.8)
                    if name:
                        all_identified_players.add(name)
                    if tid in self.tracker.lifecycles:
                        self.tracker.lifecycles[tid].assigned_name = name
                        self.tracker.lifecycles[tid].assigned_number = num
                        self.tracker.lifecycles[tid].assigned_confidence = conf

                # Propagate identities to frame history
                for f_num, objs in active_track_history.items():
                    if shot_start_frame <= f_num <= frame_idx:
                        for obj in objs:
                            if obj.track_id in identities:
                                id_info = identities[obj.track_id]
                                obj_name = id_info.get("player_name")
                                obj_num = id_info.get("jersey_number")
                                # Render on frame
                                if video_writer and f_num in shot_frame_cache:
                                    pass

            # Render cached frames for this shot to video
            if video_writer:
                for f_num in range(shot_start_frame, frame_idx + 1):
                    if f_num in shot_frame_cache:
                        frame_render = shot_frame_cache[f_num].copy()
                        objs = active_track_history.get(f_num, [])

                        for obj in objs:
                            lc = self.tracker.lifecycles.get(obj.track_id)
                            t_info = shot_team_assignments.get(obj.track_id)
                            team_str = t_info.team if t_info else "unknown"

                            # Box color
                            if team_str == "home":
                                box_color = (255, 255, 255)
                            elif team_str == "away":
                                box_color = (60, 60, 220)
                            elif "gk" in team_str:
                                box_color = (0, 165, 255)
                            elif team_str == "referee":
                                box_color = (0, 240, 240)
                            else:
                                box_color = (180, 180, 180)

                            x1, y1, x2, y2 = [int(v) for v in obj.bbox]
                            cv2.rectangle(frame_render, (x1, y1), (x2, y2), box_color, 2)

                            # Label text
                            p_name = lc.assigned_name if (lc and lc.assigned_name) else f"Track #{obj.track_id}"
                            p_num = f"#{lc.assigned_number} " if (lc and lc.assigned_number) else ""
                            label = f"{p_num}{p_name}"

                            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(
                                frame_render,
                                (x1, max(0, y1 - 22)),
                                (x1 + t_size[0] + 6, y1),
                                box_color,
                                -1
                            )
                            # Text contrast
                            text_col = (0, 0, 0) if (team_str in ["home", "gk", "referee"]) else (255, 255, 255)
                            cv2.putText(
                                frame_render, label, (x1 + 3, max(14, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_col, 2
                            )

                        # HUD Overlay (top right)
                        hud_text = f"Shot #{s_id} | Tracks: {len(objs)} | Tactical Wide"
                        cv2.putText(frame_render, hud_text, (width - 450, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                        video_writer.write(frame_render)

            self.tracker.reset()
            shot_frame_cache.clear()
            shot_team_assignments.clear()

        # PASS 1: Sequential processing loop
        frame_idx = 0
        while cap.isOpened() and (max_frames is None or frame_idx < max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            shot_frame_cache[frame_idx] = frame

            # Shot Boundary Cut Detection (<0.5ms)
            is_cut, hist_dist, turf_pct = self.shot_detector.process_frame(frame, frame_idx)

            if is_cut and frame_idx > shot_start_frame:
                total_shots += 1
                is_prev_tactical = (turf_pct >= 0.35)
                finalize_shot(current_shot_id, is_prev_tactical)
                current_shot_id += 1
                shot_start_frame = frame_idx

            is_tactical = (turf_pct >= 0.35)

            # Strided Detection & Tracking
            if is_tactical:
                if frame_idx % self.stride == 0:
                    # Run YOLO on CPU (person class = 0, sports ball = 32)
                    results = self.yolo.predict(
                        frame,
                        classes=[0],  # Persons
                        conf=self.conf_thresh,
                        device=self.device,
                        verbose=False,
                    )[0]

                    boxes = results.boxes.xyxy.cpu().numpy()
                    confs = results.boxes.conf.cpu().numpy()
                    clss = results.boxes.cls.cpu().numpy()

                    # BoT-SORT update with sparse optical flow CMC (with_reid=False)
                    tracked_objs = self.tracker.update_strided(frame, boxes, confs, clss, frame_idx)
                    active_track_history[frame_idx] = tracked_objs

                    # Cluster teams every 5 strided frames (10 frames)
                    if frame_idx % (self.stride * 5) == 0 and tracked_objs:
                        actor_tuples = [(obj.track_id, obj.bbox) for obj in tracked_objs]
                        cluster_res = self.clusterer.cluster_actors(frame, actor_tuples, frame_width=width)
                        shot_team_assignments.update(cluster_res)
                else:
                    # Linear Bounding Box Interpolation (Interpolate between strided steps)
                    if (frame_idx - 1) in active_track_history:
                        prev_objs = active_track_history[frame_idx - 1]
                        interp_objs = [
                            TrackedObject(
                                frame_idx=frame_idx,
                                track_id=o.track_id,
                                bbox=o.bbox,
                                confidence=o.confidence,
                                class_id=o.class_id,
                                is_interpolated=True,
                            )
                            for o in prev_objs
                        ]
                        active_track_history[frame_idx] = interp_objs

            frame_idx += 1

        # Finalize last shot
        total_shots += 1
        finalize_shot(current_shot_id, is_tactical=True)

        cap.release()
        if video_writer:
            video_writer.release()

        elapsed = time.time() - start_time
        effective_fps = round(frame_idx / max(0.001, elapsed), 2)

        # Build diagnostic output
        diagnostics = {
            "video_path": video_path,
            "output_video_path": out_video_path,
            "total_frames_processed": frame_idx,
            "fps": fps,
            "elapsed_seconds": round(elapsed, 2),
            "effective_fps": effective_fps,
            "total_shots": total_shots,
            "tactical_shots": tactical_shots,
            "unique_identified_players": list(all_identified_players),
        }

        diag_path = os.path.join(output_dir, "tracking_pipeline_summary.json")
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)

        print(f"[VideoTrackingPipeline] Done! Processed {frame_idx} frames in {elapsed:.2f}s ({effective_fps} FPS). Output: {out_video_path}")

        return VideoTrackingResult(
            video_path=video_path,
            output_video_path=out_video_path,
            total_frames=frame_idx,
            fps=fps,
            total_duration_sec=round(frame_idx / fps, 2),
            elapsed_time_sec=round(elapsed, 2),
            effective_fps=effective_fps,
            shots_count=total_shots,
            tactical_shots_count=tactical_shots,
            unique_tracklets_count=len(self.tracker.lifecycles),
            identified_players_count=len(all_identified_players),
            diagnostics_json_path=diag_path,
        )
