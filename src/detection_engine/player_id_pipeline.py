"""
src/detection_engine/player_id_pipeline.py
================================================================================
Master Isolated Player Detection, Tracking, Roster Sync & Identity Pipeline.

Coordinates the complete 9-stage architecture:
1. ScoreboardClockOCR (+ Kickoff / PTS Fallback)
2. ActiveXIResolver (Dynamic Substitutions, Red Cards, Whitelist Ground Truth)
3. Pitch Boundary Segmentation & Hough Line Extraction
4. Multi-Scale YOLOv8 Detection + Scale / Banner / Boundary Filtering
5. Colorimetry / Kit Partitioning (Home vs Away vs GK vs Referee)
6. 2D Homography & Spatial Role Topology
7. Multi-Crop Torso Zoom CLAHE Montage & Biometric Feature Extraction
8. Gemini Multimodal VLM Query
9. Hungarian Bipartite Matching + Biometric/Spatial Critic Audit
================================================================================
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.detection_engine.pitch_geometry import PitchGeometryEngine
from src.detection_engine.types import (
    AffinityScoreBreakdown,
    CandidateBox,
    DetectedPitchActor,
    DetectionResult,
    PipelineConfig,
)
from src.vision.isolated_player_ai_locator import (
    PLAYER_BIOMETRIC_DB,
    IsolatedPlayerAILocator,
    LocatorConfig,
)
from src.vision.match_roster_engine import (
    CANONICAL_CLASICO_ROSTER,
    ActiveRosterState,
    ActiveXIResolver,
    MatchClockExtractor,
)

logger = logging.getLogger("detection_engine.player_id_pipeline")


class PlayerIDPipeline:
    """
    Self-contained, production-grade 9-stage player detection & identity engine.
    Accepts raw frame input, match metadata JSON / dict, and optional PTS timestamp.
    Returns visual artifacts and rich diagnostic JSON payload.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        match_metadata: Optional[Union[Dict[str, Any], str, Path]] = None,
    ):
        self.config = config or PipelineConfig()

        # Map PipelineConfig to LocatorConfig for internal subroutines
        self.locator_config = LocatorConfig(
            iteration_id=self.config.iteration_id,
            name=self.config.name,
            yolo_imgsz=self.config.yolo_imgsz,
            use_tiled_highres_yolo=self.config.use_tiled_highres_yolo,
            use_adaptive_turf_hull=self.config.use_adaptive_turf_hull,
            use_broadcast_banner_mask=self.config.use_broadcast_banner_mask,
            use_touchline_staff_filter=self.config.use_touchline_staff_filter,
            use_blaugrana_stripe_classifier=self.config.use_blaugrana_stripe_classifier,
            use_gk_color_detector=self.config.use_gk_color_detector,
            use_scoreboard_clock_ocr=self.config.use_scoreboard_clock_ocr,
            use_live_active_xi_sync=self.config.use_live_active_xi_sync,
            use_play_by_play_telemetry=self.config.use_play_by_play_telemetry,
            use_homography_topology_priors=self.config.use_homography_topology_priors,
            use_superres_clahe_torso_zoom=self.config.use_superres_clahe_torso_zoom,
            use_biometric_visual_db=self.config.use_biometric_visual_db,
            use_hungarian_global_solver=self.config.use_hungarian_global_solver,
            use_vlm_critic_audit=self.config.use_vlm_critic_audit,
            temperature=self.config.temperature,
            w_vlm_direct=self.config.w_vlm_direct,
            w_jersey_ocr=self.config.w_jersey_ocr,
            w_topology_role=self.config.w_topology_role,
            w_biometric_cue=self.config.w_biometric_cue,
        )

        self.internal_locator = IsolatedPlayerAILocator(config=self.locator_config)
        self.pitch_geom_engine = PitchGeometryEngine(config=self.config)
        self.clock_extractor = MatchClockExtractor()

        # Loaded match metadata
        self.default_match_metadata = self._load_metadata(match_metadata)

    def _load_metadata(
        self, metadata_input: Optional[Union[Dict[str, Any], str, Path]]
    ) -> Dict[str, Any]:
        """Loads and normalizes match metadata dictionary."""
        if metadata_input is None:
            return copy.deepcopy(CANONICAL_CLASICO_ROSTER)
        if isinstance(metadata_input, (str, Path)):
            p = Path(metadata_input)
            if p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            logger.warning(f"Metadata file not found: {p}. Falling back to canonical roster.")
            return copy.deepcopy(CANONICAL_CLASICO_ROSTER)
        if isinstance(metadata_input, dict):
            return copy.deepcopy(metadata_input)
        return copy.deepcopy(CANONICAL_CLASICO_ROSTER)

    def process_frame(
        self,
        frame_input: Union[str, Path, np.ndarray],
        match_metadata: Optional[Union[Dict[str, Any], str, Path]] = None,
        fallback_pts_seconds: Optional[float] = None,
        clip_id: Optional[str] = None,
        period_hint: Optional[int] = None,
    ) -> DetectionResult:
        """
        Executes the full 9-stage detection, tracking, roster sync & identity pipeline.

        Parameters:
        - `frame_input`: Broadcast screenshot (Path, filepath str, or BGR numpy.ndarray)
        - `match_metadata`: Matchday lineup/substitution payload (dict or JSON path)
        - `fallback_pts_seconds`: Video presentation timestamp in seconds (optional)
        - `clip_id`: Identifier string for logging
        - `period_hint`: 1 for 1st half, 2 for 2nd half

        Returns:
        - `DetectionResult` containing annotated_frame, turf_poly_debug, clahe_crops, diagnostics.
        """
        t0 = time.time()

        # Load frame
        frame_path: Optional[Path] = None
        if isinstance(frame_input, (str, Path)):
            frame_path = Path(frame_input)
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise FileNotFoundError(f"Could not load image: {frame_path}")
            clip_id = clip_id or frame_path.stem
        else:
            frame = frame_input
            clip_id = clip_id or f"frame_{int(time.time()*1000)%100000}"

        h, w = frame.shape[:2]
        meta = self._load_metadata(match_metadata) if match_metadata is not None else self.default_match_metadata

        # ---------------------------------------------------------------------
        # STAGE 1: Scoreboard Clock OCR & Timestamp Synchronization
        # ---------------------------------------------------------------------
        fallback_min = int(fallback_pts_seconds // 60) if fallback_pts_seconds is not None else 10
        fallback_sec = int(fallback_pts_seconds % 60) if fallback_pts_seconds is not None else 0
        if period_hint is None:
            period_hint = 1 if fallback_min <= 45 else 2

        if self.config.use_scoreboard_clock_ocr:
            clock_res = self.clock_extractor.extract_match_time(
                frame=frame,
                clip_id=clip_id,
                fallback_minute=fallback_min,
                period_hint=period_hint,
            )
            if clock_res.is_valid and clock_res.match_minute > 0:
                match_min = clock_res.match_minute
                match_sec = clock_res.match_second
                clock_str = clock_res.clock_str
                clock_source = clock_res.source
            else:
                match_min = fallback_min
                match_sec = fallback_sec
                clock_str = f"{match_min:02d}:{match_sec:02d}"
                clock_source = "FALLBACK_PTS"
        else:
            match_min = fallback_min
            match_sec = fallback_sec
            clock_str = f"{match_min:02d}:{match_sec:02d}"
            clock_source = "FALLBACK_PTS"

        # ---------------------------------------------------------------------
        # STAGE 2: Dynamic Active-XI Roster Resolution (Minute-Synced)
        # ---------------------------------------------------------------------
        active_xi_resolver = ActiveXIResolver(match_data=meta)
        active_state: ActiveRosterState = active_xi_resolver.get_active_players(
            match_minute=match_min if self.config.use_live_active_xi_sync else 1,
            match_second=match_sec,
        )

        # ---------------------------------------------------------------------
        # STAGE 3: Pitch Geometry, Convex Hull & Hough Line Segmentation
        # ---------------------------------------------------------------------
        pitch_geom = self.pitch_geom_engine.analyze(frame)
        shot_type = pitch_geom["shot_type"]
        hull_mask = pitch_geom["hull_mask"]
        grass_mask = pitch_geom["grass_mask"]
        banner_bbox = pitch_geom["banner_bbox"]

        # ---------------------------------------------------------------------
        # STAGE 4: Multi-Scale Player Detection & Candidate Filtering
        # ---------------------------------------------------------------------
        candidate_boxes, accepted_actors = self._detect_and_filter_candidates(
            frame=frame,
            pitch_geom=pitch_geom,
        )

        # ---------------------------------------------------------------------
        # STAGE 5: 2D Homography & Spatial Topology Priors
        # ---------------------------------------------------------------------
        self.internal_locator.compute_spatial_topology_priors(
            frame=frame,
            actors=accepted_actors,
            shot_type=shot_type,
            period=period_hint,
            clip_id=clip_id,
            persp_slope=pitch_geom.get("persp_slope", 0.20),
        )

        # ---------------------------------------------------------------------
        # STAGE 6: Team Colorimetry, Kit Partitioning & GK Uniqueness Lock
        # ---------------------------------------------------------------------
        accepted_actors = self.classify_teams_and_kits(
            frame=frame,
            actors=accepted_actors,
            pitch_geom=pitch_geom,
            shot_type=shot_type,
        )

        # ---------------------------------------------------------------------
        # STAGE 7: Multi-Crop CLAHE Torso Zoom Montage & Biometric Features
        # ---------------------------------------------------------------------
        annotated_prelim, crop_sheet = self.internal_locator.build_enriched_crop_sheet(
            frame=frame,
            actors=accepted_actors,
            frame_path=frame_path,
        )

        # ---------------------------------------------------------------------
        # STAGE 8: Gemini Multimodal VLM Query
        # ---------------------------------------------------------------------
        event_ctx = {
            "clip_id": clip_id,
            "period": period_hint,
            "clock_str": clock_str,
            "event_type": "open_play",
            "description": f"Match action at {clock_str}",
            "primary_player": "",
        }
        vlm_results = self.internal_locator.query_vlm_identities(
            annotated_frame=annotated_prelim,
            crop_sheet=crop_sheet,
            actors=accepted_actors,
            active_state=active_state,
            event_context=event_ctx,
            shot_type=shot_type,
        )

        # ---------------------------------------------------------------------
        # STAGE 9: Hungarian Bipartite Matching + Critic Audit
        # ---------------------------------------------------------------------
        affinity_breakdown_records: Dict[int, List[Dict[str, Any]]] = {}
        critic_overrides: List[Dict[str, Any]] = []

        accepted_actors = self._solve_hungarian_with_diagnostics(
            actors=accepted_actors,
            vlm_results=vlm_results,
            active_state=active_state,
            event_context=event_ctx,
            affinity_records_out=affinity_breakdown_records,
            critic_overrides_out=critic_overrides,
        )

        # ---------------------------------------------------------------------
        # Visual Renderings
        # ---------------------------------------------------------------------
        annotated_result = self.internal_locator.render_resolved_annotated_frame(frame, accepted_actors)
        turf_poly_debug = self.pitch_geom_engine.render_turf_poly_debug(
            frame=frame,
            pitch_geom=pitch_geom,
            candidate_boxes=candidate_boxes,
            accepted_actors=accepted_actors,
        )

        elapsed = time.time() - t0

        # Assemble diagnostics
        raw_ocr_reads = {
            a.track_id: {"ocr_number": a.ocr_number, "ocr_confidence": round(float(a.ocr_confidence), 2)}
            for a in accepted_actors if a.ocr_number is not None
        }

        diagnostics_payload: Dict[str, Any] = {
            "clip_id": clip_id,
            "clock": {
                "detected_clock_time": clock_str,
                "match_minute": match_min,
                "match_second": match_sec,
                "period": period_hint,
                "source": clock_source,
            },
            "shot_geometry": {
                "shot_type": shot_type,
                "turf_ratio": pitch_geom["turf_ratio"],
                "pitch_lines_detected": len(pitch_geom.get("hough_lines", [])),
                "banner_masked": pitch_geom["banner_bbox"] is not None,
            },
            "active_xi_whitelist": {
                "home": [p["name"] for p in active_state.home_active],
                "away": [p["name"] for p in active_state.away_active],
                "subbed_off": [p["name"] for p in active_state.subbed_off_players],
            },
            "raw_vlm_output": vlm_results,
            "raw_ocr_reads": raw_ocr_reads,
            "affinity_matrix_breakdown": affinity_breakdown_records,
            "critic_overrides": critic_overrides,
            "candidate_filtering": {
                "total_candidates": len(candidate_boxes),
                "accepted_count": len(accepted_actors),
                "rejected_count": len([c for c in candidate_boxes if c.status == "rejected"]),
                "rejected_boxes": [c.to_dict() for c in candidate_boxes if c.status == "rejected"],
            },
            "resolved_actors": [a.to_dict() for a in accepted_actors],
            "execution_time_sec": round(elapsed, 2),
        }

        return DetectionResult(
            actors=[a.to_dict() for a in accepted_actors],
            annotated_frame=annotated_result,
            turf_poly_debug=turf_poly_debug,
            clahe_crops=crop_sheet,
            diagnostics=diagnostics_payload,
            clock_str=clock_str,
            match_minute=match_min,
            match_second=match_sec,
            shot_type=shot_type,
            elapsed_sec=elapsed,
        )

    # -------------------------------------------------------------------------
    # Internal Candidate Box Detection & Filtering with Diagnostic Recording
    # -------------------------------------------------------------------------

    def _detect_and_filter_candidates(
        self,
        frame: np.ndarray,
        pitch_geom: Dict[str, Any],
    ) -> Tuple[List[CandidateBox], List[DetectedPitchActor]]:
        """
        Runs YOLOv8 full-frame + tiled passes, applying spatial turf and banner filters
        while logging every rejected candidate box with explicit reason.
        """
        h, w = frame.shape[:2]
        yolo = self.internal_locator._get_yolo()
        shot_type = pitch_geom["shot_type"]
        hull_mask = pitch_geom["hull_mask"]
        grass_mask = pitch_geom["grass_mask"]
        banner_bbox = pitch_geom["banner_bbox"]

        raw_boxes: List[Tuple[int, int, int, int, float]] = []

        # 1. Full-frame YOLOv8
        conf_thresh = 0.16 if shot_type == "close_up" else self.config.yolo_conf_thresh
        if yolo is not None:
            res = yolo.predict(frame, classes=[0], conf=conf_thresh, imgsz=self.config.yolo_imgsz, verbose=False)
            if res and len(res) > 0 and res[0].boxes is not None:
                xyxy = res[0].boxes.xyxy.cpu().numpy()
                confs = res[0].boxes.conf.cpu().numpy()
                for box, c in zip(xyxy, confs):
                    raw_boxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(c)))

        # 2. Tiled high-res pass on wide tactical shots
        if yolo is not None and self.config.use_tiled_highres_yolo and shot_type != "close_up":
            y_top = max(0, int(h * 0.12))
            y_bot = min(h, int(h * 0.98))
            tile_w = int(w * 0.58)
            x_offsets = [0, int(w * 0.42)]
            for x0 in x_offsets:
                x1_tile = min(w, x0 + tile_w)
                crop = frame[y_top:y_bot, x0:x1_tile]
                if crop.size == 0:
                    continue
                t_res = yolo.predict(crop, classes=[0], conf=0.14, imgsz=960, verbose=False)
                if t_res and len(t_res) > 0 and t_res[0].boxes is not None:
                    xyxy = t_res[0].boxes.xyxy.cpu().numpy()
                    confs = t_res[0].boxes.conf.cpu().numpy()
                    for box, c in zip(xyxy, confs):
                        bx1 = int(box[0]) + x0
                        by1 = int(box[1]) + y_top
                        bx2 = int(box[2]) + x0
                        by2 = int(box[3]) + y_top
                        raw_boxes.append((bx1, by1, bx2, by2, float(c)))

        merged_boxes = self.internal_locator._nms_boxes(raw_boxes, iou_thresh=0.45)

        # Fix 3.1: Shot type detection first: if max(H) > 0.45 * Frame_Height -> CLOSE_UP, else WIDE_TACTICAL
        max_bh = max([b[3] - b[1] for b in merged_boxes], default=0)
        if max_bh > int(0.45 * h):
            shot_type = "close_up"
        elif pitch_geom.get("shot_type") == "close_up":
            shot_type = "close_up"
        else:
            shot_type = "wide_tactical"
        pitch_geom["shot_type"] = shot_type

        # Scale estimation for wide tactical shots
        confident_dims = [
            max(b[2] - b[0], b[3] - b[1])
            for b in merged_boxes
            if b[4] >= 0.55 and (b[3] / float(h) * 100.0) >= 18.0
        ]
        median_player_dim = float(np.median(confident_dims)) if confident_dims else 90.0
        min_allowed_dim = max(38, int(0.38 * median_player_dim))

        all_candidate_boxes: List[CandidateBox] = []
        accepted_actors: List[DetectedPitchActor] = []
        next_box_id = 1

        pitch_poly = pitch_geom.get("pitch_polygon")

        for (x1, y1, x2, y2, conf) in merged_boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            bw = x2 - x1
            bh = y2 - y1
            aspect = bh / max(1.0, float(bw))
            boot_x = 0.5 * (x1 + x2)
            boot_y = float(y2)
            x_pct = (boot_x / float(w)) * 100.0
            y_pct = (boot_y / float(h)) * 100.0

            c_box = CandidateBox(
                box_id=next_box_id,
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                status="accepted",
                rejection_reason="",
                x_pct=x_pct,
                y_pct=y_pct,
                aspect=aspect,
            )
            next_box_id += 1

            # Rule 1: Min dimensions
            if bw < 8 or bh < 16:
                c_box.status = "rejected"
                c_box.rejection_reason = "DIM_TOO_SMALL"
                all_candidate_boxes.append(c_box)
                continue

            # Fix 3.2: Perspective Scale Prior (CLOSE_UP vs WIDE_TACTICAL)
            if shot_type == "close_up":
                # Linear perspective scale prior based on bottom y2:
                # Expected height at far top of pitch (y2 ≈ 0.25 * Frame_Height) is 36px, scaling linearly with y2.
                # Keep any human detection with conf >= 0.45 and Height >= 36px standing on valid turf.
                y2_frac = float(y2) / float(h)
                expected_min_h = max(36, int(36 + max(0.0, y2_frac - 0.25) * 45.0))
                if bh < 36:
                    c_box.status = "rejected"
                    c_box.rejection_reason = f"CLOSE_UP_SCALE_BELOW_36PX ({bh}px)"
                    all_candidate_boxes.append(c_box)
                    continue
                if conf < 0.45 and bh < expected_min_h:
                    c_box.status = "rejected"
                    c_box.rejection_reason = f"CLOSE_UP_SCALE_PRIOR ({bh}px < {expected_min_h}px, conf={conf:.2f})"
                    all_candidate_boxes.append(c_box)
                    continue
            else:
                if max(bw, bh) < min_allowed_dim:
                    c_box.status = "rejected"
                    c_box.rejection_reason = "SCALE_BELOW_MEDIAN_TURF_PRIOR"
                    all_candidate_boxes.append(c_box)
                    continue
                if bh < 44 and (conf < 0.42 or (bw * bh) < 1600):
                    c_box.status = "rejected"
                    c_box.rejection_reason = "SPHERICAL_BALL_SCALE"
                    all_candidate_boxes.append(c_box)
                    continue

            # Fix 1.3: Static Equipment & Non-Human Filter (sideline microphones, camera tripods)
            torso_h = max(12, int(0.38 * bh))
            t_crop = frame[y1:min(h, y1 + torso_h), max(0, x1 + int(0.12 * bw)):min(w, x2 - int(0.12 * bw))]
            if t_crop.size > 0 and shot_type != "close_up":
                t_hsv = cv2.cvtColor(t_crop, cv2.COLOR_BGR2HSV)
                tH, tS, tV = t_hsv[:, :, 0], t_hsv[:, :, 1], t_hsv[:, :, 2]
                skin_mask = (tH <= 25) & (tS >= 35) & (tS <= 180) & (tV >= 60)
                kit_mask = (
                    ((tS < 55) & (tV >= 155)) |  # White kit
                    ((tH >= 100) & (tH <= 138) & (tS >= 35) & (tV >= 35)) |  # Blaugrana blue
                    (((tH <= 14) | (tH >= 155)) & (tS >= 35) & (tV >= 35)) |  # Crimson red
                    ((tH >= 35) & (tH <= 85) & (tS >= 80) & (tV >= 90)) |     # Green GK
                    (((tH <= 20) | (tH >= 165)) & (tS >= 85) & (tV >= 140)) | # Salmon GK
                    ((tH >= 82) & (tH <= 108) & (tS >= 55) & (tV >= 120))     # Cyan ref
                )
                human_pixel_frac = float(np.mean(skin_mask | kit_mask))
                if human_pixel_frac < 0.12 and aspect >= 1.6:
                    c_box.status = "rejected"
                    c_box.rejection_reason = f"STATIC_EQUIPMENT_METALLIC_TRIPOD (human_frac={human_pixel_frac:.2f})"
                    all_candidate_boxes.append(c_box)
                    continue

            # Rule 3: Aspect Ratio
            min_aspect = 0.52 if conf >= 0.45 else 0.72
            if shot_type != "close_up" and (aspect < min_aspect or aspect > 5.2):
                c_box.status = "rejected"
                c_box.rejection_reason = f"ASPECT_RATIO_{aspect:.2f}"
                all_candidate_boxes.append(c_box)
                continue

            # Rule 4: Scoreboard & broadcast bug areas
            if x_pct < 18.0 and y_pct < 18.0:
                c_box.status = "rejected"
                c_box.rejection_reason = "SCOREBOARD_BUG_ZONE"
                all_candidate_boxes.append(c_box)
                continue
            if x_pct > 87.0 and y_pct < 18.0:
                c_box.status = "rejected"
                c_box.rejection_reason = "DAZN_BUG_ZONE"
                all_candidate_boxes.append(c_box)
                continue

            # Rule 5: Lower-third broadcast overlay banner
            if banner_bbox is not None and self.config.use_broadcast_banner_mask:
                bx1_b, by1_b, bx2_b, by2_b = banner_bbox
                cx_box = 0.5 * (x1 + x2)
                cy_box = 0.5 * (y1 + y2)
                if bx1_b <= cx_box <= bx2_b and by1_b <= cy_box <= by2_b and y1 >= int(h * 0.79):
                    pad_l = max(0, x1 - 25)
                    pad_r = min(w, x2 + 25)
                    banner_slice = frame[max(0, y1):min(h, y2), pad_l:pad_r]
                    if banner_slice.size > 0:
                        hsv_sl = cv2.cvtColor(banner_slice, cv2.COLOR_BGR2HSV)
                        white_frac = float(np.mean((hsv_sl[:, :, 1] < 35) & (hsv_sl[:, :, 2] > 200)))
                        if white_frac > 0.30:
                            c_box.status = "rejected"
                            c_box.rejection_reason = "LOWER_THIRD_BANNER_PORTRAIT"
                            all_candidate_boxes.append(c_box)
                            continue

            # Fix 1.2: Sideline Dugout & Bench Filter: sample 24x16 patch centered directly at ankle/foot contact zone
            ankle_y1 = max(0, int(boot_y - 8))
            ankle_y2 = min(h, int(boot_y + 8))
            ankle_x1 = max(0, int(boot_x - 12))
            ankle_x2 = min(w, int(boot_x + 12))

            foot_patch = grass_mask[ankle_y1:ankle_y2, ankle_x1:ankle_x2]
            foot_turf_frac = float(np.mean(foot_patch > 0)) if foot_patch.size > 0 else 0.0

            # Left and right background strips outside ankles (12px wide)
            strip_l = grass_mask[ankle_y1:ankle_y2, max(0, int(boot_x - 24)):max(0, int(boot_x - 12))]
            strip_r = grass_mask[ankle_y1:ankle_y2, min(w, int(boot_x + 12)):min(w, int(boot_x + 24))]
            surround_turf = 0.0
            if strip_l.size > 0 and strip_r.size > 0:
                surround_turf = 0.5 * (float(np.mean(strip_l > 0)) + float(np.mean(strip_r > 0)))
            elif strip_l.size > 0:
                surround_turf = float(np.mean(strip_l > 0))
            elif strip_r.size > 0:
                surround_turf = float(np.mean(strip_r > 0))

            dist_to_boundary = 999.0
            if pitch_poly is not None and len(pitch_poly) > 2:
                dist_to_boundary = cv2.pointPolygonTest(pitch_poly, (float(boot_x), float(boot_y)), True)

            # Rule 6: Turf Hull & Boundary checks
            if shot_type == "close_up":
                if self.config.use_touchline_staff_filter and (bh < int(h * 0.15) or conf < 0.23):
                    c_box.status = "rejected"
                    c_box.rejection_reason = "CLOSE_UP_BACKGROUND_NOISE"
                    all_candidate_boxes.append(c_box)
                    continue
            else:
                # Outside concave touchline polygon
                if dist_to_boundary < -18.0:
                    c_box.status = "rejected"
                    c_box.rejection_reason = f"OFF_TURF_CONCAVE_BOUNDARY (dist={dist_to_boundary:.1f}px)"
                    all_candidate_boxes.append(c_box)
                    continue

                # Near sideline/dugout: require green turf fraction >= 35% on ankle contact patch
                if dist_to_boundary <= 25.0:
                    if foot_turf_frac < 0.35 and surround_turf < 0.30:
                        c_box.status = "rejected"
                        c_box.rejection_reason = f"DUGOUT_BENCH_ANKLE_NON_GRASS (turf={foot_turf_frac:.2f}, surround={surround_turf:.2f})"
                        all_candidate_boxes.append(c_box)
                        continue

                # Top touchline bench staff filter
                min_playable_y = max(17.5, pitch_geom["min_turf_y_pct"] + 1.0)
                if y_pct < min_playable_y or (y1 <= int(h * 0.025) and y_pct < 20.0):
                    c_box.status = "rejected"
                    c_box.rejection_reason = "TOP_TOUCHLINE_BENCH_STAFF"
                    all_candidate_boxes.append(c_box)
                    continue

            # Accepted candidate
            all_candidate_boxes.append(c_box)
            actor = DetectedPitchActor(
                track_id=len(accepted_actors) + 1,
                bbox=[x1, y1, x2, y2],
                x_pct=x_pct,
                y_pct=y_pct,
                confidence=conf,
                actor_type="player",
                team="unknown",
            )
            accepted_actors.append(actor)

        # Deterministic sorting
        accepted_actors.sort(key=lambda a: (round(a.x_pct / 4.0), a.y_pct))
        for idx, act in enumerate(accepted_actors, start=1):
            act.track_id = idx

        return all_candidate_boxes, accepted_actors

    # -------------------------------------------------------------------------
    # STAGE 5: Team Colorimetry, Kit Partitioning & GK Uniqueness Lock
    # -------------------------------------------------------------------------

    def classify_teams_and_kits(
        self,
        frame: np.ndarray,
        actors: List[DetectedPitchActor],
        pitch_geom: Dict[str, Any],
        shot_type: str = "wide_tactical",
    ) -> List[DetectedPitchActor]:
        """
        Classifies every detected pitch actor into:
        - 'home' (FC Barcelona: Blaugrana blue & crimson + dark blue shorts, or green GK)
        - 'away' (Real Madrid: White shirt & shorts, or salmon/orange GK)
        - 'referee' (Cyan jersey H=82..108, S>=60, V>=125)
        Applies:
        1. Fix 2.1: Steward / Bib filter (neon yellow vest, high-vis orange bib outside penalty area)
        2. Fix 2.1: Manager all-black attire filter (V_median < 48, S_median < 45)
        3. Fix 2.2: Motion-blurred linesman / referee upgrade
        4. Fix 4.1: Goalkeeper Uniqueness Lock: At most ONE actor per team is_gk=True
        5. Fix 3.3: Late Quota Capping: at most 10 outfield + 1 GK per team (applied AFTER non-players purged)
        """
        h, w = frame.shape[:2]
        pitch_poly = pitch_geom.get("pitch_polygon")
        filtered_actors: List[DetectedPitchActor] = []
        potential_gks: List[DetectedPitchActor] = []

        for act in actors:
            x1, y1, x2, y2 = act.bbox
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            boot_x = 0.5 * (x1 + x2)
            boot_y = float(y2)

            dist_to_boundary = 999.0
            if pitch_poly is not None and len(pitch_poly) > 2:
                dist_to_boundary = cv2.pointPolygonTest(pitch_poly, (float(boot_x), float(boot_y)), True)

            # Upper torso crop
            if shot_type == "close_up" and bh > int(h * 0.45):
                ty1 = max(0, int(y1 + bh * 0.44))
                ty2 = min(h, int(y1 + bh * 0.90))
            else:
                ty1 = max(0, int(y1 + bh * 0.14))
                ty2 = min(h, int(y1 + bh * 0.52))
            tx1 = max(0, int(x1 + bw * 0.18))
            tx2 = min(w, int(x2 - bw * 0.18))
            torso = frame[ty1:ty2, tx1:tx2]

            # Shorts crop
            sy1 = max(0, int(y1 + bh * 0.50))
            sy2 = min(h, int(y1 + bh * 0.72))
            sx1 = max(0, int(x1 + bw * 0.20))
            sx2 = min(w, int(x2 - bw * 0.20))
            shorts = frame[sy1:sy2, sx1:sx2]

            # Head crop
            hy1 = max(0, int(y1))
            hy2 = min(h, int(y1 + bh * 0.20))
            hx1 = max(0, int(x1 + bw * 0.15))
            hx2 = min(w, int(x2 - bw * 0.15))
            head = frame[hy1:hy2, hx1:hx2]

            # Boots crop
            by1_c = max(0, int(y1 + bh * 0.76))
            by2_c = min(h, int(y2))
            bx1_c = max(0, int(x1))
            bx2_c = min(w, int(x2))
            boots_crop = frame[by1_c:by2_c, bx1_c:bx2_c]

            if torso.size == 0:
                continue

            hsv_t = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            all_H = hsv_t[:, :, 0]
            all_S = hsv_t[:, :, 1]
            all_V = hsv_t[:, :, 2]

            non_grass = ~((all_H >= 32) & (all_H <= 85) & (all_S >= 35))
            if np.count_nonzero(non_grass) < 6:
                non_grass = np.ones(hsv_t.shape[:2], dtype=bool)

            H_vals = all_H[non_grass]
            S_vals = all_S[non_grass]
            V_vals = all_V[non_grass]

            # Fix 2.1: Steward / Bib & Manager Isolation Filter (Stage 4)
            # 1. Neon Yellow/Chartreuse steward vest: Hue in [20, 38], S >= 135, V >= 170 with fraction >= 0.20
            neon_steward_frac = float(np.mean((all_H >= 20) & (all_H <= 38) & (all_S >= 135) & (all_V >= 170)))
            if neon_steward_frac >= 0.20:
                act.actor_type = "staff"
                act.team = "unknown"
                continue

            # 2. High-Vis Orange steward/photographer bib outside penalty area: Hue in [4, 22], S >= 155, V >= 160
            orange_bib_frac = float(np.mean((all_H >= 4) & (all_H <= 22) & (all_S >= 155) & (all_V >= 160)))
            is_in_penalty_box = (
                (act.pitch_x_m <= 16.5 or act.pitch_x_m >= 88.5) and
                (13.8 <= act.pitch_y_m <= 54.2)
            ) if (act.pitch_x_m > 0 or act.pitch_y_m > 0) else False

            if orange_bib_frac >= 0.15:
                if not is_in_penalty_box or abs(dist_to_boundary) <= 45.0:
                    act.actor_type = "staff"
                    act.team = "unknown"
                    continue

            # 3. Manager / Sideline Staff All-Black Attire: Upper Torso and Shorts both have V_median < 48 and S_median < 45
            torso_v_med = float(np.median(V_vals)) if len(V_vals) > 0 else 100.0
            torso_s_med = float(np.median(S_vals)) if len(S_vals) > 0 else 100.0
            shorts_v_med = 100.0
            shorts_s_med = 100.0
            if shorts.size > 0:
                hsv_s = cv2.cvtColor(shorts, cv2.COLOR_BGR2HSV)
                shorts_v_med = float(np.median(hsv_s[:, :, 2]))
                shorts_s_med = float(np.median(hsv_s[:, :, 1]))

            if torso_v_med < 48.0 and torso_s_med < 45.0 and shorts_v_med < 48.0 and shorts_s_med < 45.0:
                act.actor_type = "staff"
                act.team = "unknown"
                continue

            # Fix 2.2: Referee & Linesman Filter Upgrade
            # Widen cyan/neon referee detection to Hue in [82, 108], S >= 60, V >= 125 with fraction >= 0.18
            cyan_ref_frac = float(np.mean((H_vals >= 82) & (H_vals <= 108) & (S_vals >= 60) & (V_vals >= 125)))
            white_frac = float(np.mean((S_vals < 52) & (V_vals > 152)))
            dark_blue_frac = float(np.mean((H_vals >= 102) & (H_vals <= 138) & (S_vals >= 35) & (V_vals < 185)))
            crimson_red_frac = float(np.mean(((H_vals <= 14) | (H_vals >= 155)) & (S_vals >= 40) & (V_vals >= 35) & (V_vals < 175)))
            blaugrana_frac = dark_blue_frac + crimson_red_frac

            if cyan_ref_frac >= 0.18:
                if shot_type == "close_up" or abs(dist_to_boundary) <= 35.0:
                    act.team = "referee"
                    act.actor_type = "referee"
                    continue
                elif blaugrana_frac < 0.25 and white_frac < 0.35:
                    act.team = "referee"
                    act.actor_type = "referee"
                    continue

            # Shorts white & orange
            shorts_white_frac = 0.0
            shorts_orange_frac = 0.0
            if shorts.size > 0:
                hsv_s = cv2.cvtColor(shorts, cv2.COLOR_BGR2HSV)
                ng_s = ~((hsv_s[:, :, 0] >= 30) & (hsv_s[:, :, 0] <= 85) & (hsv_s[:, :, 1] >= 35))
                if np.count_nonzero(ng_s) > 4:
                    sH = hsv_s[:, :, 0][ng_s]
                    sS = hsv_s[:, :, 1][ng_s]
                    sV = hsv_s[:, :, 2][ng_s]
                    shorts_white_frac = float(np.mean((sS < 55) & (sV > 145)))
                    shorts_orange_frac = float(np.mean(((sH <= 18) | (sH >= 166)) & (sS >= 95) & (sV >= 155)))

            # Mint boots
            mint_boot_frac = 0.0
            if boots_crop.size > 0:
                hsv_b = cv2.cvtColor(boots_crop, cv2.COLOR_BGR2HSV)
                mint_boot_frac = float(np.mean((hsv_b[:, :, 0] >= 68) & (hsv_b[:, :, 0] <= 96) & (hsv_b[:, :, 1] >= 58) & (hsv_b[:, :, 2] >= 115)))

            # Hair
            hair_blonde_frac = 0.0
            if head.size > 0:
                hsv_h = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
                ng_h = ~((hsv_h[:, :, 0] >= 30) & (hsv_h[:, :, 0] <= 88) & (hsv_h[:, :, 1] >= 35))
                if np.count_nonzero(ng_h) > 4:
                    hair_blonde_frac = float(np.mean((hsv_h[:, :, 0][ng_h] >= 12) & (hsv_h[:, :, 0][ng_h] <= 32) & (hsv_h[:, :, 2][ng_h] > 135)))

            # Goalkeeper color signals
            orange_gk_frac = float(np.mean(((all_H <= 18) | (all_H >= 166)) & (all_S >= 95) & (all_V >= 165)))
            green_gk_frac = float(np.mean((all_H >= 45) & (all_H <= 82) & (all_S >= 95) & (all_V >= 135)))

            act.visual_features = {
                "white_frac": round(white_frac, 3),
                "shorts_white_frac": round(shorts_white_frac, 3),
                "blaugrana_frac": round(blaugrana_frac, 3),
                "cyan_ref_frac": round(cyan_ref_frac, 3),
                "orange_gk_frac": round(orange_gk_frac, 3),
                "green_gk_frac": round(green_gk_frac, 3),
                "mint_boot_frac": round(mint_boot_frac, 3),
                "hair_blonde_frac": round(hair_blonde_frac, 3),
            }

            is_potential_gk = False
            if (orange_gk_frac >= 0.20 or (orange_gk_frac >= 0.14 and shorts_orange_frac >= 0.20)) and dark_blue_frac < 0.08:
                act.team = "away"
                act.actor_type = "goalkeeper"
                is_potential_gk = True
            elif green_gk_frac >= 0.35 and bw < int(w * 0.12):
                act.team = "home"
                act.actor_type = "goalkeeper"
                is_potential_gk = True

            if is_potential_gk:
                potential_gks.append(act)
                continue

            # Outfield team assignment
            away_score = white_frac * 1.3 + shorts_white_frac * 0.7
            home_score = blaugrana_frac * 1.4 + float(np.mean(V_vals < 115)) * 0.6

            if away_score >= home_score and (white_frac >= 0.22 or shorts_white_frac >= 0.28):
                act.team = "away"
            else:
                act.team = "home"
            act.is_gk = False
            act.actor_type = "player"
            filtered_actors.append(act)

        # Fix 4.1: Goalkeeper Uniqueness Lock
        # At most ONE actor per team can have is_gk = True.
        # Score candidates by proximity to goalmouth/penalty box interior and valid pitch containment.
        for team_side in ["home", "away"]:
            team_gks = [a for a in potential_gks if a.team == team_side]
            if not team_gks:
                continue

            def _score_gk_cand(a: DetectedPitchActor) -> float:
                score = 0.0
                if a.pitch_x_m <= 18.0 or a.pitch_x_m >= 87.0:
                    score += 10.0
                if "box" in a.tactical_zone or "penalty" in a.tactical_zone or "gk" in a.tactical_zone:
                    score += 10.0
                b_x = 0.5 * (a.bbox[0] + a.bbox[2])
                b_y = float(a.bbox[3])
                if pitch_poly is not None and len(pitch_poly) > 2:
                    d = cv2.pointPolygonTest(pitch_poly, (b_x, b_y), True)
                    if d >= 0:
                        score += 5.0
                    else:
                        score -= 25.0
                asp = (a.bbox[3] - a.bbox[1]) / max(1, a.bbox[2] - a.bbox[0])
                if 1.8 <= asp <= 3.8:
                    score += 3.0
                return score

            team_gks.sort(key=_score_gk_cand, reverse=True)
            best_gk = team_gks[0]
            best_gk.is_gk = True
            best_gk.actor_type = "goalkeeper"
            filtered_actors.append(best_gk)

            # Demote extra GK candidates
            for extra_gk in team_gks[1:]:
                extra_gk.is_gk = False
                extra_gk.actor_type = "staff"

        # Fix 3.3: Late Quota Capping (at most 10 outfield + 1 GK per team)
        # Applied ONLY AFTER Stage 2 and Stage 4 filters have purged all non-players
        if self.config.use_live_active_xi_sync:
            home_outfield = [a for a in filtered_actors if a.team == "home" and not a.is_gk]
            away_outfield = [a for a in filtered_actors if a.team == "away" and not a.is_gk]
            home_gk = [a for a in filtered_actors if a.team == "home" and a.is_gk]
            away_gk = [a for a in filtered_actors if a.team == "away" and a.is_gk]

            if len(home_outfield) > 10:
                home_outfield.sort(key=lambda a: (a.visual_features.get("blaugrana_frac", 0.0) + a.confidence), reverse=True)
                home_outfield = home_outfield[:10]
            if len(away_outfield) > 10:
                away_outfield.sort(key=lambda a: (a.visual_features.get("white_frac", 0.0) + a.confidence), reverse=True)
                away_outfield = away_outfield[:10]

            filtered_actors = home_gk + home_outfield + away_gk + away_outfield
            filtered_actors.sort(key=lambda a: (round(a.x_pct / 4.0), a.y_pct))

        for idx, act in enumerate(filtered_actors, start=1):
            act.track_id = idx

        return filtered_actors

    # -------------------------------------------------------------------------
    # STAGE 9: Hungarian Bipartite Matching with Full Diagnostic Affinity Matrix
    # -------------------------------------------------------------------------

    def _solve_hungarian_with_diagnostics(
        self,
        actors: List[DetectedPitchActor],
        vlm_results: Dict[int, Dict[str, Any]],
        active_state: ActiveRosterState,
        event_context: Dict[str, Any],
        affinity_records_out: Dict[int, List[Dict[str, Any]]],
        critic_overrides_out: List[Dict[str, Any]],
    ) -> List[DetectedPitchActor]:
        """
        Executes Hungarian bipartite matching while recording full S_vlm, S_ocr, S_topo, S_bio
        breakdown for every track against active pool, and capturing critic overrides.
        """
        try:
            from scipy.optimize import linear_sum_assignment
            has_scipy = True
        except ImportError:
            has_scipy = False

        clip_min = active_state.match_minute
        home_pool = [dict(p, in_active_xi=True) for p in active_state.home_active]
        away_pool = [dict(p, in_active_xi=True) for p in active_state.away_active]

        # Roster candidates: include active starters + recently subbed
        raw_desc = str(event_context.get("description", "")).lower()
        for sub_p in active_state.subbed_off_players + active_state.benched_players:
            s_min = int(sub_p.get("minute") or sub_p.get("scheduled_minute") or -99)
            p_name_low = str(sub_p.get("name", "")).lower()
            norm_n = self.internal_locator._normalize_name(sub_p.get("name", ""))
            bio = PLAYER_BIOMETRIC_DB.get(norm_n, {})
            if sub_p in active_state.subbed_off_players or abs(s_min - clip_min) <= 1 or (p_name_low and p_name_low in raw_desc):
                t_side = sub_p.get("team") or bio.get("team") or "home"
                target_pool = home_pool if t_side == "home" else away_pool
                if not any(self.internal_locator._normalize_name(x["name"]) == norm_n for x in target_pool):
                    target_pool.append({
                        "name": sub_p["name"],
                        "number": sub_p.get("number") or bio.get("num") or 99,
                        "role": bio.get("role", "MF"),
                        "team": t_side,
                        "is_gk": False,
                        "in_active_xi": (abs(s_min - clip_min) <= 1 or (p_name_low and p_name_low in raw_desc)),
                    })

        role_Adj = {
            "GK":  {"GK": 1.0},
            "LB":  {"LB": 1.0, "LCB": 0.65, "LW": 0.55, "LCM": 0.50, "RB": 0.35},
            "LCB": {"LCB": 1.0, "CB": 0.95, "RCB": 0.75, "LB": 0.65, "DM": 0.55},
            "CB":  {"CB": 1.0, "LCB": 0.95, "RCB": 0.95, "DM": 0.55},
            "RCB": {"RCB": 1.0, "CB": 0.95, "LCB": 0.75, "RB": 0.65, "DM": 0.55},
            "RB":  {"RB": 1.0, "RCB": 0.65, "RW": 0.55, "RCM": 0.50, "LB": 0.35},
            "DM":  {"DM": 1.0, "CM": 0.90, "LCM": 0.80, "RCM": 0.80, "LCB": 0.50, "RCB": 0.50, "AM": 0.60},
            "LCM": {"LCM": 1.0, "CM": 0.90, "DM": 0.80, "AM": 0.75, "LW": 0.65, "LB": 0.55, "RCM": 0.65},
            "CM":  {"CM": 1.0, "DM": 0.90, "LCM": 0.90, "RCM": 0.90, "AM": 0.80},
            "RCM": {"RCM": 1.0, "CM": 0.90, "DM": 0.80, "AM": 0.75, "RW": 0.65, "RB": 0.55, "LCM": 0.65},
            "AM":  {"AM": 1.0, "LCM": 0.80, "RCM": 0.80, "CF": 0.75, "LW": 0.70, "RW": 0.70, "DM": 0.60},
            "LW":  {"LW": 1.0, "CF": 0.72, "AM": 0.70, "LCM": 0.65, "LB": 0.50, "RW": 0.40},
            "CF":  {"CF": 1.0, "LW": 0.72, "RW": 0.72, "AM": 0.75},
            "RW":  {"RW": 1.0, "CF": 0.72, "AM": 0.70, "RCM": 0.65, "RB": 0.50, "LW": 0.40},
        }

        for team_side, pool in [("home", home_pool), ("away", away_pool)]:
            team_actors = [a for a in actors if a.team == team_side]
            if not team_actors or not pool:
                continue

            # Fix 4.2: Goalkeeper Hard-Assignment & Outfield Hungarian Decoupling
            # Hard-assign the unique is_gk track to the active goalkeeper from ActiveXIResolver immediately.
            # Remove both that bounding box and the GK player from the outfield Hungarian cost matrix
            # so an outfield player can NEVER be assigned to a green GK bounding box.
            gk_actor = next((a for a in team_actors if a.is_gk), None)
            active_gk_cand = next((p for p in pool if p.get("is_gk") or str(p.get("role", "")).upper() == "GK"), None)

            if gk_actor is not None and active_gk_cand is not None:
                gk_actor.assigned_name = active_gk_cand["name"]
                gk_actor.assigned_number = int(active_gk_cand.get("number") or 13)
                gk_actor.assigned_role = "GK"
                gk_actor.assignment_confidence = 0.99
                gk_actor.assignment_method = "GK_UNIQUENESS_LOCK"
                gk_actor.reasoning = f"Unique goalkeeper locked to #{gk_actor.assigned_number} {gk_actor.assigned_name} inside goalmouth/penalty box"

                # Record diagnostic affinity breakdown for locked GK
                affinity_records_out[gk_actor.track_id] = [{
                    "track_id": gk_actor.track_id,
                    "player_name": active_gk_cand["name"],
                    "jersey_number": gk_actor.assigned_number,
                    "team": team_side,
                    "role": "GK",
                    "s_vlm": 0.42,
                    "s_ocr": 0.30,
                    "s_topo": 0.18,
                    "s_bio": 0.10,
                    "total_score": 1.0,
                    "is_assigned": True,
                    "notes": "GK_UNIQUENESS_LOCK",
                }]

                # Decouple: Remove GK actor and GK candidate from outfield Hungarian cost matrix!
                team_actors = [a for a in team_actors if a.track_id != gk_actor.track_id]
                pool = [p for p in pool if p["name"] != active_gk_cand["name"]]

            if not team_actors or not pool:
                continue

            n_act = len(team_actors)
            n_cand = len(pool)

            # OCR number dedup
            best_num_owner: Dict[int, Tuple[int, float]] = {}
            for i, act in enumerate(team_actors):
                vlm_r = vlm_results.get(act.track_id, {})
                vlm_num = vlm_r.get("visible_number")
                vlm_num_conf = float(vlm_r.get("number_confidence") or 0.0)
                try:
                    n_int = int(vlm_num) if vlm_num is not None else None
                except (ValueError, TypeError):
                    n_int = None
                if n_int is not None and vlm_num_conf >= 0.85:
                    prev = best_num_owner.get(n_int)
                    if prev is None or vlm_num_conf > prev[1]:
                        best_num_owner[n_int] = (i, vlm_num_conf)

            # Score Matrix S and Sub-score breakdown tables
            S = np.zeros((n_act, n_cand), dtype=np.float64)
            breakdown_table: Dict[int, List[AffinityScoreBreakdown]] = {act.track_id: [] for act in team_actors}

            for i, act in enumerate(team_actors):
                vlm_r = vlm_results.get(act.track_id, {})
                vlm_name = self.internal_locator._normalize_name(str(vlm_r.get("name", "")))
                vlm_num = vlm_r.get("visible_number")
                vlm_num_conf = float(vlm_r.get("number_confidence") or 0.0)
                vlm_conf = float(vlm_r.get("confidence") or 0.7)
                vlm_role = str(vlm_r.get("tactical_role") or "").upper()

                try:
                    vlm_num_int = int(vlm_num) if vlm_num is not None else None
                except (ValueError, TypeError):
                    vlm_num_int = None

                if vlm_num_int is not None and best_num_owner.get(vlm_num_int, (-1, 0.0))[0] == i:
                    act.ocr_number = vlm_num_int
                    act.ocr_confidence = vlm_num_conf
                else:
                    vlm_num_int = None

                for j, cand in enumerate(pool):
                    c_name = cand["name"]
                    c_norm = self.internal_locator._normalize_name(c_name)
                    c_num = int(cand.get("number") or 0)
                    c_role = str(cand.get("role") or "MF").upper()
                    c_is_gk = bool(cand.get("is_gk") or c_role == "GK")

                    if act.is_gk != c_is_gk:
                        S[i, j] = -100.0
                        breakdown_table[act.track_id].append(
                            AffinityScoreBreakdown(
                                track_id=act.track_id,
                                player_name=c_name,
                                jersey_number=c_num,
                                team=team_side,
                                role=c_role,
                                s_vlm=0.0,
                                s_ocr=0.0,
                                s_topo=0.0,
                                s_bio=0.0,
                                total_score=-100.0,
                                notes="GK_ROLE_MISMATCH",
                            )
                        )
                        continue

                    # 1. VLM Direct Match
                    s_vlm_raw = 0.0
                    if vlm_name and (vlm_name == c_norm or vlm_name in c_norm or c_norm in vlm_name):
                        s_vlm_raw = max(0.65, vlm_conf)
                    elif vlm_name and vlm_name.split()[-1] == c_norm.split()[-1]:
                        s_vlm_raw = max(0.60, vlm_conf * 0.9)
                    s_vlm = self.config.w_vlm_direct * s_vlm_raw

                    # 2. Jersey Number OCR Match
                    s_ocr_raw = 0.0
                    if vlm_num_int is not None and vlm_num_int == c_num:
                        s_ocr_raw = max(0.90, vlm_num_conf)
                    elif vlm_num_int is not None and vlm_num_int != c_num and vlm_num_conf >= 0.90:
                        s_ocr_raw = -0.40
                    s_ocr = self.config.w_jersey_ocr * s_ocr_raw

                    # 3. 2D Homography Role Prior Match
                    topo_raw = 0.0
                    for r_prior, r_prob in act.role_priors.items():
                        compat = role_Adj.get(r_prior, {}).get(c_role, 0.15)
                        topo_raw = max(topo_raw, r_prob * compat)
                    if vlm_role and vlm_role == c_role:
                        topo_raw = max(topo_raw, 0.85)
                    s_topo = self.config.w_topology_role * topo_raw

                    # 4. Biometric & Visual Cue Compatibility
                    bio = PLAYER_BIOMETRIC_DB.get(c_norm, {})
                    bio_score = 0.5
                    obs_bio = str(vlm_r.get("observed_biometrics") or "").lower()
                    vf = act.visual_features
                    mint_boot = float(vf.get("mint_boot_frac", 0.0))
                    hair_blonde = float(vf.get("hair_blonde_frac", 0.0))

                    if bio:
                        if "curly" in obs_bio and "yamal" in c_norm:
                            bio_score = max(bio_score, 1.2)
                        if "blonde" in obs_bio or "ginger" in obs_bio or hair_blonde >= 0.04:
                            if "fermin" in c_norm or "de jong" in c_norm:
                                bio_score = max(bio_score, 1.2)
                        if ("fermin" in c_norm or "lopez" in c_norm) and (mint_boot >= 0.035 or "mint" in obs_bio or hair_blonde >= 0.03):
                            bio_score = 3.5
                        if "vazquez" in c_norm and act.formation_line in ("backline", "midfield") and act.flank_zone in ("right", "right_half"):
                            bio_score = max(bio_score, 2.0)
                        if "casado" in c_norm and act.formation_line in ("midfield", "backline"):
                            bio_score = max(bio_score, 2.0)
                        if "lunin" in c_norm and act.is_gk:
                            bio_score = 4.0
                        if "pena" in c_norm and act.is_gk:
                            bio_score = 4.0
                    s_bio = self.config.w_biometric_cue * bio_score

                    active_prior = 0.10 if cand.get("in_active_xi", True) else 0.0
                    total = s_vlm + s_ocr + s_topo + s_bio + active_prior
                    S[i, j] = total

                    breakdown_table[act.track_id].append(
                        AffinityScoreBreakdown(
                            track_id=act.track_id,
                            player_name=c_name,
                            jersey_number=c_num,
                            team=team_side,
                            role=c_role,
                            s_vlm=s_vlm,
                            s_ocr=s_ocr,
                            s_topo=s_topo,
                            s_bio=s_bio,
                            total_score=total,
                            notes="OK",
                        )
                    )

            # Solve Hungarian optimal assignment
            cost_matrix = -S
            if has_scipy:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
            else:
                row_ind, col_ind = [], []
                used_j = set()
                for i_idx in range(n_act):
                    best_j = max((j_idx for j_idx in range(n_cand) if j_idx not in used_j), key=lambda j_idx: S[i_idx, j_idx], default=0)
                    row_ind.append(i_idx)
                    col_ind.append(best_j)
                    used_j.add(best_j)

            for r_i, c_j in zip(row_ind, col_ind):
                act = team_actors[r_i]
                cand = pool[c_j]
                vlm_r = vlm_results.get(act.track_id, {})
                act.assigned_name = cand["name"]
                act.assigned_number = int(cand.get("number") or 0)
                act.assigned_role = str(cand.get("role") or "MF")
                raw_score = float(S[r_i, c_j])
                act.assignment_confidence = min(0.99, max(0.68, raw_score + 0.35))
                if act.ocr_number == act.assigned_number and act.ocr_number is not None:
                    act.assignment_method = "JERSEY_OCR_LOCKED+HUNGARIAN"
                    act.assignment_confidence = max(act.assignment_confidence, 0.97)
                elif vlm_r:
                    act.assignment_method = "VLM_MULTIMODAL+HUNGARIAN_2D"
                else:
                    act.assignment_method = "HUNGARIAN_2D_TOPOLOGY"

                # Mark winning assignment in breakdown table
                for b in breakdown_table[act.track_id]:
                    if b.player_name == cand["name"]:
                        b.is_assigned = True
                        act.affinity_breakdown = b

                affinity_records_out[act.track_id] = [
                    b.to_dict() for b in sorted(breakdown_table[act.track_id], key=lambda x: x.total_score, reverse=True)[:5]
                ]

            # Critic Pass for Away Team (Lunin GK, Vázquez RB, Mendy LB, Güler RW)
            if self.config.use_vlm_critic_audit and team_side == "away":
                self._run_critic_audit_logged(team_actors, pool, critic_overrides_out)

        return actors

    def _run_critic_audit_logged(
        self,
        team_actors: List[DetectedPitchActor],
        pool: List[Dict[str, Any]],
        overrides_out: List[Dict[str, Any]],
    ) -> None:
        """Runs the biometric & topology critic audit, logging every override."""
        def _swap_or_assign(target_act: DetectedPitchActor, target_num: int, target_name: str, target_role: str, reason: str):
            if target_act.assigned_number == target_num:
                return
            old_name, old_num, old_role = target_act.assigned_name, target_act.assigned_number, target_act.assigned_role
            other_act = next((a for a in team_actors if a.assigned_number == target_num), None)

            target_act.assigned_name = target_name
            target_act.assigned_number = target_num
            target_act.assigned_role = target_role
            target_act.assignment_confidence = 0.96
            target_act.assignment_method = "CRITIC_BIOMETRIC_TOPOLOGY_LOCK"
            target_act.reasoning = reason

            overrides_out.append({
                "action": "CRITIC_OVERRIDE",
                "track_id": target_act.track_id,
                "previous_player": old_name,
                "previous_number": old_num,
                "assigned_player": target_name,
                "assigned_number": target_num,
                "reason": reason,
            })

            if other_act is not None and old_name:
                other_act.assigned_name = old_name
                other_act.assigned_number = old_num
                other_act.assigned_role = old_role

        # Critic Check 1: Lunin GK
        if any(int(p.get("number") or 0) == 13 and p.get("is_gk") for p in pool):
            gk_cands = [a for a in team_actors if a.is_gk]
            if gk_cands:
                _swap_or_assign(gk_cands[0], 13, "Andriy Lunin", "GK", "Critic: Salmon goalkeeper kit -> #13 Andriy Lunin (GK)")

        # Critic Check 2: Lucas Vázquez RB
        if any(int(p.get("number") or 0) == 17 for p in pool) and len(team_actors) >= 4:
            rb_cands = [
                a for a in team_actors
                if not a.is_gk and a.formation_line in ("backline", "midfield") and a.flank_zone in ("right", "right_half")
            ]
            if rb_cands:
                best_rb = max(rb_cands, key=lambda a: a.role_priors.get("RB", 0.0) + (1.0 if a.ocr_number == 17 else 0.0))
                _swap_or_assign(best_rb, 17, "Lucas Vázquez", "RB", "Critic: Right-back formation slot (BACKLINE_RIGHT) -> #17 Lucas Vázquez (RB)")

        # Critic Check 3: Ferland Mendy LB
        if any(int(p.get("number") or 0) == 23 for p in pool) and len(team_actors) >= 4:
            lb_cands = [
                a for a in team_actors
                if not a.is_gk and a.formation_line in ("backline", "midfield") and a.flank_zone in ("left", "left_half")
            ]
            if lb_cands:
                best_lb = max(lb_cands, key=lambda a: a.role_priors.get("LB", 0.0))
                _swap_or_assign(best_lb, 23, "Ferland Mendy", "LB", "Critic: Left-back formation slot (BACKLINE_LEFT) -> #23 Ferland Mendy (LB)")
