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
from .smart_collage_batcher import SmartCollageBatcher, TeamGridPayload


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
        home_def_x: Optional[float] = None,
        away_def_x: Optional[float] = None,
        period: Optional[int] = None,
        match_minute: int = 46,
    ):
        self.stride = stride
        self.conf_thresh = conf_thresh
        self.device = device
        self.home_team_name = home_team_name
        self.away_team_name = away_team_name
        self.match_minute = match_minute
        self.period = period if period is not None else (2 if match_minute > 45 else 1)
        # Halftime orientation: Period 1 (RMA defends right 1.0), Period 2 (RMA defends left 0.0)
        self.home_def_x = home_def_x if home_def_x is not None else (0.0 if self.period == 2 else 1.0)
        self.away_def_x = away_def_x if away_def_x is not None else (1.0 if self.period == 2 else 0.0)
        self.locked_track_identities: Dict[int, Dict[str, Any]] = {}

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

    def filter_detections_by_pitch(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        confs: np.ndarray,
        clss: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter out TV overlays, scoreboards, and off-pitch personnel (cameramen/stewards).
        Preserves real players (e.g. Balde on the bottom-left grass) while killing flat 2D TV graphics.
        """
        if len(boxes) == 0:
            return boxes, confs, clss

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        turf_mask = cv2.inRange(hsv, (32, 35, 35), (85, 255, 255))
        contours, _ = cv2.findContours(turf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hull = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 0.15 * (w * h):
                hull = cv2.convexHull(largest)

        keep_indices = []
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in b]
            xc = (x1 + x2) // 2
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            hw_ratio = float(box_h) / float(box_w)

            # Scoreboard deadzone (top left)
            if x1 < 0.20 * w and y1 < 0.18 * h:
                continue

            # Check foot turf fraction
            fy1 = max(0, y2 - 8)
            fy2 = min(h, y2 + 4)
            fx1 = max(0, x1)
            fx2 = min(w, x2)
            foot_patch = turf_mask[fy1:fy2, fx1:fx2]
            foot_turf = float((foot_patch > 0).mean()) if foot_patch.size > 0 else 0.0

            # Check overall box turf fraction
            box_patch = turf_mask[y1:y2, x1:x2]
            box_turf = float((box_patch > 0).mean()) if box_patch.size > 0 else 0.0

            # Lower-third broadcast graphic banner check (y2 > 0.83 * H)
            # Real players (like Balde at bottom-left) have hw_ratio >= 1.60 and green turf under their feet.
            # 2D TV graphic portraits (like Raphinha banner) have hw_ratio ~ 1.15 and low box/foot turf.
            if y2 > 0.83 * h:
                if hw_ratio < 1.50 or box_turf < 0.28 or foot_turf < 0.30:
                    continue

            # Playable pitch hull check (foot/ankle contact point)
            if hull is not None:
                pt_dist = cv2.pointPolygonTest(hull, (float(xc), float(y2)), False)
                if pt_dist < 0:
                    continue

            keep_indices.append(i)

        if not keep_indices:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )

        return boxes[keep_indices], confs[keep_indices], clss[keep_indices]

    def _load_roster(self, path: Optional[str]) -> Dict[str, Any]:
        """Load canonical match roster from JSON if available, combining starters + substitutes."""
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)

                # Schema 1: Match metadata JSON with home_team/away_team objects
                if isinstance(raw.get("home_team"), dict) and "starting_xi" in raw["home_team"]:
                    h_dict = raw["home_team"]
                    a_dict = raw.get("away_team", {})

                    if "madrid" in h_dict.get("name", "").lower():
                        madrid_dict = h_dict
                        barca_dict = a_dict
                    else:
                        madrid_dict = a_dict
                        barca_dict = h_dict

                    # Combine starting XI + substitutes for full matchday squad
                    m_starters = madrid_dict.get("starting_xi", [])
                    m_subs = madrid_dict.get("substitutes", [])
                    m_seen = {p["number"] for p in m_starters}
                    home_squad = list(m_starters) + [p for p in m_subs if p.get("number") not in m_seen]

                    b_starters = barca_dict.get("starting_xi", [])
                    b_subs = barca_dict.get("substitutes", [])
                    b_seen = {p["number"] for p in b_starters}
                    away_squad = list(b_starters) + [p for p in b_subs if p.get("number") not in b_seen]

                    return {
                        "home_team": self.home_team_name,
                        "away_team": self.away_team_name,
                        "home_active_xi": home_squad,
                        "away_active_xi": away_squad,
                    }

                # Schema 2: Direct active squad lists
                if "home_active_xi" in raw:
                    return raw
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

            # Module 3: Camera Motion Compensated Tracklet Stitching
            stitched = self.tracker.stitch_lifecycles(shot_frame_cache)
            if stitched > 0:
                print(f"[VideoTrackingPipeline] Stitched {stitched} fragmented tracklets via BoT-SORT GMC + Boot Appearance Gate")

            # PASS 2: Team-Split 2D Grid Collages + Strict 1-to-1 Bijection
            home_payload, away_payload, context_frame = self.batcher.build_team_grid_payloads(
                shot_id=s_id,
                lifecycles=self.tracker.lifecycles,
                team_assignments=shot_team_assignments,
                frame_cache=shot_frame_cache,
                active_roster=self.roster,
                fps=fps,
                home_def_x=self.home_def_x,
                away_def_x=self.away_def_x,
                period=self.period,
            )

            identities: Dict[int, Dict[str, Any]] = {}
            if home_payload:
                # Track-level cache check: reuse high-confidence assignments
                if all(tid in self.locked_track_identities for tid in home_payload.track_ids):
                    h_ids = {tid: dict(self.locked_track_identities[tid]) for tid in home_payload.track_ids}
                    print(f"[VideoTrackingPipeline] Track-level Cache HIT: Reused {len(h_ids)} {self.home_team_name} identities")
                else:
                    h_ids = self.batcher.resolve_team_identities(home_payload, output_dir)
                    for tid, id_data in h_ids.items():
                        if id_data.get("confidence", 0.0) >= 0.80:
                            self.locked_track_identities[tid] = id_data
                identities.update(h_ids)

            if away_payload:
                if all(tid in self.locked_track_identities for tid in away_payload.track_ids):
                    a_ids = {tid: dict(self.locked_track_identities[tid]) for tid in away_payload.track_ids}
                    print(f"[VideoTrackingPipeline] Track-level Cache HIT: Reused {len(a_ids)} {self.away_team_name} identities")
                else:
                    a_ids = self.batcher.resolve_team_identities(away_payload, output_dir)
                    for tid, id_data in a_ids.items():
                        if id_data.get("confidence", 0.0) >= 0.80:
                            self.locked_track_identities[tid] = id_data
                identities.update(a_ids)

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

                            # Label text: Strictly "T{track_id}" - NO '#' prefix for Track IDs
                            if lc and lc.assigned_name:
                                p_num = f"#{lc.assigned_number} " if lc.assigned_number else ""
                                label = f"T{obj.track_id}: {p_num}{lc.assigned_name}"
                            else:
                                label = f"T{obj.track_id}"

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

                    # Filter out TV overlays, scoreboards, and off-pitch personnel
                    boxes, confs, clss = self.filter_detections_by_pitch(frame, boxes, confs, clss)

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
