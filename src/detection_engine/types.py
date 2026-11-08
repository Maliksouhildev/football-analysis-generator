"""
src/detection_engine/types.py
================================================================================
Data models, dataclasses, and configurations for the isolated player detection,
tracking, roster sync, and identity assignment engine.
================================================================================
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class CandidateBox:
    """Represents a raw detection candidate box before/after spatial filtering."""
    box_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    status: str = "accepted"         # "accepted" | "rejected"
    rejection_reason: str = ""       # e.g. "OFF_TURF", "ASPECT_RATIO", "DUGOUT_STAFF", "SCALE_TOO_SMALL"
    x_pct: float = 0.0
    y_pct: float = 0.0
    aspect: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box_id": self.box_id,
            "bbox": list(self.bbox),
            "confidence": round(float(self.confidence), 3),
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "x_pct": round(float(self.x_pct), 2),
            "y_pct": round(float(self.y_pct), 2),
            "aspect": round(float(self.aspect), 2),
        }


@dataclass
class AffinityScoreBreakdown:
    """Breakdown of Hungarian bipartite affinity sub-scores for a specific track and roster player."""
    track_id: int
    player_name: str
    jersey_number: int
    team: str
    role: str
    s_vlm: float = 0.0
    s_ocr: float = 0.0
    s_topo: float = 0.0
    s_bio: float = 0.0
    total_score: float = 0.0
    is_assigned: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "player_name": self.player_name,
            "jersey_number": self.jersey_number,
            "team": self.team,
            "role": self.role,
            "s_vlm": round(float(self.s_vlm), 3),
            "s_ocr": round(float(self.s_ocr), 3),
            "s_topo": round(float(self.s_topo), 3),
            "s_bio": round(float(self.s_bio), 3),
            "total_score": round(float(self.total_score), 3),
            "is_assigned": self.is_assigned,
            "notes": self.notes,
        }


@dataclass
class DetectedPitchActor:
    """Internal model for a detected on-pitch actor (player, GK, referee)."""
    track_id: int
    bbox: List[int]  # [x1, y1, x2, y2]
    x_pct: float
    y_pct: float
    confidence: float
    actor_type: str = "player"  # "player" | "goalkeeper" | "referee"
    team: str = "unknown"       # "home" | "away" | "referee"
    kit_color_hsv: Tuple[int, int, int] = (0, 0, 0)
    is_gk: bool = False

    # 2D Homography & Topology priors
    pitch_x_m: float = 0.0
    pitch_y_m: float = 0.0
    tactical_zone: str = "midfield_central"
    formation_line: str = "midfield"
    flank_zone: str = "central"
    role_priors: Dict[str, float] = field(default_factory=dict)

    # Super-res crops & visual cues
    full_crop: Optional[np.ndarray] = None
    torso_crop: Optional[np.ndarray] = None
    visual_features: Dict[str, Any] = field(default_factory=dict)

    # OCR Jersey Number
    ocr_number: Optional[int] = None
    ocr_confidence: float = 0.0

    # Resolved Identity
    assigned_name: str = ""
    assigned_number: Optional[int] = None
    assigned_role: str = ""
    assignment_confidence: float = 0.0
    assignment_method: str = "UNASSIGNED"
    reasoning: str = ""
    affinity_breakdown: Optional[AffinityScoreBreakdown] = None
    is_ambiguous: bool = False
    swap_margin: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "x_pct": round(float(self.x_pct), 2),
            "y_pct": round(float(self.y_pct), 2),
            "actor_type": self.actor_type,
            "team": self.team,
            "is_gk": self.is_gk,
            "pitch_coords_m": (round(float(self.pitch_x_m), 1), round(float(self.pitch_y_m), 1)),
            "tactical_zone": self.tactical_zone,
            "formation_line": self.formation_line,
            "flank_zone": self.flank_zone,
            "role_priors": {k: round(float(v), 2) for k, v in self.role_priors.items()},
            "ocr_number": self.ocr_number,
            "ocr_confidence": round(float(self.ocr_confidence), 2),
            "assigned_name": self.assigned_name,
            "assigned_number": self.assigned_number,
            "assigned_role": self.assigned_role,
            "assignment_confidence": round(float(self.assignment_confidence), 3),
            "assignment_method": self.assignment_method,
            "is_ambiguous": self.is_ambiguous,
            "swap_margin": round(float(self.swap_margin), 3),
            "reasoning": self.reasoning,
            "affinity_breakdown": self.affinity_breakdown.to_dict() if self.affinity_breakdown else None,
        }


@dataclass
class PipelineConfig:
    """Master configuration parameters for the isolated detection & identity engine."""
    iteration_id: int = 12
    name: str = "Isolated Production Detection Engine"

    # Stage 1: Clock OCR
    use_scoreboard_clock_ocr: bool = True

    # Stage 2: Active XI Sync
    use_live_active_xi_sync: bool = True

    # Stage 3: Pitch Boundary & Banner
    use_adaptive_turf_hull: bool = True
    use_broadcast_banner_mask: bool = True
    use_touchline_staff_filter: bool = True

    # Stage 4: Multi-Scale YOLOv8
    yolo_model_name: str = "yolov8x.pt"
    yolo_imgsz: int = 1280
    use_tiled_highres_yolo: bool = True
    yolo_conf_thresh: float = 0.22

    # Stage 5: Colorimetry / Kit Partitioning
    use_blaugrana_stripe_classifier: bool = True
    use_gk_color_detector: bool = True

    # Stage 6: 2D Homography & Role Topology
    use_homography_topology_priors: bool = True

    # Stage 7: CLAHE Super-Resolution Torso Zoom
    use_superres_clahe_torso_zoom: bool = True
    use_biometric_visual_db: bool = True

    # Stage 8: VLM Query
    vlm_model: str = "gemini-2.5-flash"
    temperature: float = 0.01

    # Stage 9: Hungarian Bipartite Solver Weights
    use_hungarian_global_solver: bool = True
    use_play_by_play_telemetry: bool = True
    w_vlm_direct: float = 0.42
    w_jersey_ocr: float = 0.30
    w_topology_role: float = 0.18
    w_biometric_cue: float = 0.10

    # Critic Audit
    use_vlm_critic_audit: bool = True


@dataclass
class DetectionResult:
    """Structured container holding all visual assets and machine-readable diagnostics."""
    actors: List[Dict[str, Any]]
    annotated_frame: np.ndarray
    turf_poly_debug: np.ndarray
    clahe_crops: np.ndarray
    diagnostics: Dict[str, Any]
    clock_str: str = "00:00"
    match_minute: int = 0
    match_second: int = 0
    shot_type: str = "wide_tactical"
    elapsed_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clock_str": self.clock_str,
            "match_minute": self.match_minute,
            "match_second": self.match_second,
            "shot_type": self.shot_type,
            "total_actors": len(self.actors),
            "elapsed_sec": round(self.elapsed_sec, 2),
            "actors": self.actors,
            "diagnostics": self.diagnostics,
        }

    def save_debug_outputs(self, target_dir: Union[str, Path]) -> Dict[str, Path]:
        """Saves annotated_result.png, turf_poly_debug.png, clahe_crops.png, and diagnostics.json."""
        import cv2
        out_dir = Path(target_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        res_map = {
            "annotated_result": out_dir / "annotated_result.png",
            "turf_poly_debug": out_dir / "turf_poly_debug.png",
            "clahe_crops": out_dir / "clahe_crops.png",
            "diagnostics": out_dir / "diagnostics.json",
        }

        if self.annotated_frame is not None and self.annotated_frame.size > 0:
            cv2.imwrite(str(res_map["annotated_result"]), self.annotated_frame)

        if self.turf_poly_debug is not None and self.turf_poly_debug.size > 0:
            cv2.imwrite(str(res_map["turf_poly_debug"]), self.turf_poly_debug)

        if self.clahe_crops is not None and self.clahe_crops.size > 0:
            cv2.imwrite(str(res_map["clahe_crops"]), self.clahe_crops)

        with open(res_map["diagnostics"], "w", encoding="utf-8") as f:
            json.dump(self.diagnostics, f, indent=2, ensure_ascii=False)

        return res_map
