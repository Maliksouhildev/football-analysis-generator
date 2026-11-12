"""
src/vision/event_anchor_resolver.py
Event Anchor Resolver: Fusing Synchronized StatsBomb Match Events with Visual Possession Dynamics.

Fixes the "Nearest Player Fallback Trap" (False Carrier Attribution):
In tight duels, slide tackles, or 50-50 challenges, the player closest to the ball is often
the defender attempting to dispossess the carrier, not the carrier himself!
This module verifies possession dynamics:
1. Foot-to-ball vector in metric pitch coordinates (or pixel coordinates).
2. Kit color alignment against the event attacking team (firewall against dispossessing defenders).
3. Carrier-to-ball velocity directional alignment.
4. StatsBomb pitch coordinate (X, Y) proximity.

Anchors the confirmed player as the Ground Truth Anchor Node for subsequent bidirectional tracking.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("vision.event_anchor_resolver")


@dataclasses.dataclass
class GroundTruthAnchorNode:
    """Represents a verified ground-truth player anchor derived from an authoritative match event."""
    track_id: int
    player_name: str
    team: str
    role: str = "BALL_CARRIER"
    event_type: str = "pass"
    event_id: str = ""
    timestamp_sec: float = 0.0
    pitch_coords: Tuple[float, float] = (50.0, 34.0)
    box: List[float] = dataclasses.field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    foot_pos: Tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.98
    is_ground_truth: bool = True
    status: str = "ANCHORED_GROUND_TRUTH"
    possession_score: float = 1.0
    telemetry: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "player_name": self.player_name,
            "team": self.team,
            "role": self.role,
            "event_type": self.event_type,
            "event_id": self.event_id,
            "timestamp_sec": self.timestamp_sec,
            "pitch_coords": list(self.pitch_coords),
            "pitch_x": self.pitch_coords[0],
            "pitch_y": self.pitch_coords[1],
            "box": list(self.box),
            "foot_pos": list(self.foot_pos),
            "confidence": self.confidence,
            "is_ground_truth": self.is_ground_truth,
            "status": self.status,
            "possession_score": self.possession_score,
            "telemetry": self.telemetry,
        }


class EventAnchorResolver:
    """
    Fuses synchronized StatsBomb match events with visual ball and player detections
    to establish Ground Truth Anchor Nodes without false carrier attribution.
    """

    def __init__(
        self,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
        max_carrier_ball_dist_m: float = 4.0,
        max_event_dist_m: float = 30.0,
        w_dist: float = 0.35,
        w_kit: float = 0.35,
        w_pitch: float = 0.20,
        w_vel: float = 0.10,
        min_possession_threshold: float = 0.40,
    ):
        """
        Args:
            pitch_length_m: Pitch length (meters, default UEFA 105m).
            pitch_width_m: Pitch width (meters, default UEFA 68m).
            max_carrier_ball_dist_m: Maximum realistic distance in meters between ball and carrier foot.
            max_event_dist_m: Maximum distance threshold from StatsBomb event pitch location.
            w_dist: Weight for foot-to-ball proximity.
            w_kit: Weight for kit / team alignment.
            w_pitch: Weight for event pitch location proximity.
            w_vel: Weight for carrier-ball velocity alignment.
            min_possession_threshold: Minimum total score to confirm possession anchor.
        """
        self.pitch_length_m = float(pitch_length_m)
        self.pitch_width_m = float(pitch_width_m)
        self.max_carrier_ball_dist_m = float(max_carrier_ball_dist_m)
        self.max_event_dist_m = float(max_event_dist_m)
        self.w_dist = float(w_dist)
        self.w_kit = float(w_kit)
        self.w_pitch = float(w_pitch)
        self.w_vel = float(w_vel)
        self.min_possession_threshold = float(min_possession_threshold)

    @staticmethod
    def compute_foot_point(box: Sequence[float]) -> Tuple[float, float]:
        """
        Calculates the ground-contact foot point (bottom center of bounding box).
        box format: [x1, y1, x2, y2].
        """
        if not box or len(box) < 4:
            return (0.0, 0.0)
        x1, _, x2, y2 = [float(v) for v in box[:4]]
        return ((x1 + x2) / 2.0, y2)

    @staticmethod
    def _extract_event_info(event: Any) -> Dict[str, Any]:
        """Extracts standardized attributes from CanonicalMatchEvent or dict."""
        if isinstance(event, dict):
            p_dict = event.get("primary_player") or event.get("player") or {}
            p_name = p_dict.get("name") if isinstance(p_dict, dict) else str(p_dict)
            t_dict = event.get("team") or {}
            t_name = t_dict.get("name") if isinstance(t_dict, dict) else str(t_dict)
            ev_type = str(event.get("event_type") or event.get("type") or "pass").lower()
            ev_id = str(event.get("id") or event.get("event_id") or "")
            ts = float(event.get("video_timestamp_sec") or event.get("timestamp_sec") or 0.0)

            px = event.get("pitch_x")
            py = event.get("pitch_y")
            # If coordinates are 0-100 canonical, scale to metric 105x68 if needed
            if px is not None and py is not None:
                px = float(px)
                py = float(py)
                if px <= 100.0 and py <= 100.0:
                    px_m = (px / 100.0) * 105.0
                    py_m = (py / 100.0) * 68.0
                else:
                    px_m, py_m = px, py
            else:
                px_m, py_m = 52.5, 34.0

            return {
                "player_name": p_name or "Event_Player",
                "team_name": t_name or "Attacking_Team",
                "event_type": ev_type,
                "event_id": ev_id,
                "timestamp_sec": ts,
                "pitch_x_m": px_m,
                "pitch_y_m": py_m,
            }

        # CanonicalMatchEvent object
        p_name = ""
        if hasattr(event, "primary_player") and event.primary_player:
            p_name = getattr(event.primary_player, "name", "")
        t_name = ""
        if hasattr(event, "team") and event.team:
            t_name = getattr(event.team, "name", "")
        ev_type = str(getattr(event, "event_type", "pass")).lower()
        if hasattr(event.event_type, "value"):
            ev_type = str(event.event_type.value).lower()
        ev_id = str(getattr(event, "id", ""))
        ts = float(getattr(event, "video_timestamp_sec", 0.0) or getattr(event, "timestamp_sec", 0.0) or 0.0)

        px = getattr(event, "pitch_x", 50.0)
        py = getattr(event, "pitch_y", 50.0)
        if px is not None and py is not None:
            px_m = (float(px) / 100.0) * 105.0
            py_m = (float(py) / 100.0) * 68.0
        else:
            px_m, py_m = 52.5, 34.0

        return {
            "player_name": p_name or "Event_Player",
            "team_name": t_name or "Attacking_Team",
            "event_type": ev_type,
            "event_id": ev_id,
            "timestamp_sec": ts,
            "pitch_x_m": px_m,
            "pitch_y_m": py_m,
        }

    def _compute_kit_match_score(
        self,
        candidate_team: Optional[str],
        event_team: str,
        is_offensive_event: bool = True,
    ) -> float:
        """
        Computes kit/team alignment score.
        In an offensive event (pass, shot, carry), an opponent contesting the ball
        is strictly penalized to prevent False Carrier Attribution.
        """
        if not candidate_team or not event_team:
            return 0.5  # Neutral when team is unidentified

        c_team = candidate_team.strip().lower()
        e_team = event_team.strip().lower()

        # Direct match or substring match
        if c_team == e_team or c_team in e_team or e_team in c_team:
            return 1.0

        # Opponent kit detection
        # If offensive event, opponents cannot be the ball carrier
        if is_offensive_event:
            return 0.0
        return 0.2

    @staticmethod
    def _compute_velocity_alignment(
        player_vel: Optional[Sequence[float]],
        ball_vel: Optional[Sequence[float]],
    ) -> float:
        """
        Computes cosine alignment between player movement vector and ball velocity vector.
        """
        if player_vel is None or ball_vel is None or len(player_vel) < 2 or len(ball_vel) < 2:
            return 1.0  # Neutral if velocity is not available

        pv = np.array([float(player_vel[0]), float(player_vel[1])], dtype=np.float32)
        bv = np.array([float(ball_vel[0]), float(ball_vel[1])], dtype=np.float32)

        norm_p = float(np.linalg.norm(pv))
        norm_b = float(np.linalg.norm(bv))

        # If ball or player is nearly static, alignment is neutral
        if norm_p < 0.2 or norm_b < 0.2:
            return 1.0

        cos_sim = float(np.dot(pv, bv) / (norm_p * norm_b))
        # Positive alignment (moving with the ball) scores up to 1.0, opposing direction scores 0.0
        return float(max(0.0, min(1.0, (cos_sim + 1.0) / 2.0)))

    def evaluate_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
        ball: Optional[Dict[str, Any]],
        event: Any,
    ) -> List[Dict[str, Any]]:
        """
        Evaluates and scores all player candidates against the synchronized event and ball position.

        Returns:
            List of evaluated candidates with detailed possession scoring metrics.
        """
        ev_info = self._extract_event_info(event)
        is_offensive = ev_info["event_type"] in (
            "pass", "shot", "goal", "carry", "progressive_carry", "dribble"
        )

        # Extract ball coordinates
        ball_pitch_x = None
        ball_pitch_y = None
        ball_img_pos = None
        ball_vel = None

        if ball is not None:
            ball_pitch_x = ball.get("pitch_x")
            ball_pitch_y = ball.get("pitch_y")
            ball_vel = ball.get("velocity")
            if "center" in ball:
                ball_img_pos = ball["center"]
            elif "box" in ball:
                bb = ball["box"]
                ball_img_pos = ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)

        # If ball has no pitch coords but has image coords, fallback to candidate foot image distance
        has_pitch_ball = ball_pitch_x is not None and ball_pitch_y is not None

        scored_candidates: List[Dict[str, Any]] = []

        for cand in candidates:
            # Exclude known referees
            if cand.get("is_referee", False) or str(cand.get("role", "")).lower() == "referee":
                continue

            box = cand.get("box", [0, 0, 0, 0])
            foot_pt = self.compute_foot_point(box)

            # 1. Foot to ball distance score (multi-modal: pitch metric + screen pixel distance)
            cand_px = cand.get("pitch_x")
            cand_py = cand.get("pitch_y")

            s_dist_metric = None
            d_ball_m = 999.0
            if has_pitch_ball and cand_px is not None and cand_py is not None:
                d_ball_m = math.hypot(float(cand_px) - float(ball_pitch_x), float(cand_py) - float(ball_pitch_y))
                s_dist_metric = math.exp(- (d_ball_m ** 2) / (2.0 * (3.8 ** 2)))

            s_dist_px = None
            d_norm_px = 999.0
            if ball_img_pos is not None:
                cand_h = max(15.0, float(box[3] - box[1]))
                d_px = math.hypot(float(foot_pt[0]) - float(ball_img_pos[0]), float(foot_pt[1]) - float(ball_img_pos[1]))
                d_norm_px = d_px / cand_h
                s_dist_px = math.exp(- (d_norm_px ** 2) / (2.0 * (1.0 ** 2)))

            if s_dist_metric is not None and s_dist_px is not None:
                s_dist = max(s_dist_metric, s_dist_px)
                d_ball = min(d_ball_m, d_norm_px * 2.0)
            elif s_dist_metric is not None:
                s_dist = s_dist_metric
                d_ball = d_ball_m
            elif s_dist_px is not None:
                s_dist = s_dist_px
                d_ball = d_norm_px * 2.0
            else:
                s_dist = 0.5
                d_ball = 2.0

            # 2. Kit match score (The Defense-Tackle Firewall)
            cand_team = cand.get("team")
            s_kit = self._compute_kit_match_score(
                candidate_team=cand_team,
                event_team=ev_info["team_name"],
                is_offensive_event=is_offensive,
            )

            # 3. Proximity to StatsBomb Event pitch location
            if cand_px is not None and cand_py is not None:
                d_event = math.hypot(float(cand_px) - float(ev_info["pitch_x_m"]), float(cand_py) - float(ev_info["pitch_y_m"]))
                if d_event > self.max_event_dist_m:
                    s_pitch = 0.0
                else:
                    s_pitch = math.exp(- (d_event ** 2) / (2.0 * (14.0 ** 2)))
            else:
                s_pitch = 0.5
                d_event = 10.0

            # 4. Carrier velocity alignment
            cand_vel = cand.get("velocity")
            s_vel = self._compute_velocity_alignment(cand_vel, ball_vel)

            # Composite weighted possession score
            # S = w_dist * S_dist + w_kit * S_kit + w_pitch * S_pitch + w_vel * S_vel
            total_score = (
                self.w_dist * s_dist
                + self.w_kit * s_kit
                + self.w_pitch * s_pitch
                + self.w_vel * s_vel
            )

            # Soft spatial bounds: prevent distant players while preserving carriers during passes/duels
            if d_ball > 16.0:
                total_score *= max(0.4, 1.0 - (d_ball - 16.0) / 25.0)
            if d_event > self.max_event_dist_m:
                total_score *= max(0.0, 1.0 - (d_event - self.max_event_dist_m) / 15.0)

            # Defense Tackle Firewall: strict cap against opposing defenders during offensive events
            if is_offensive and s_kit == 0.0:
                total_score = min(total_score, 0.20)

            cand_eval = dict(cand)
            cand_eval["foot_point"] = foot_pt
            cand_eval["dist_to_ball_m"] = round(d_ball, 2)
            cand_eval["dist_to_event_m"] = round(d_event, 2)
            cand_eval["s_dist"] = round(s_dist, 3)
            cand_eval["s_kit"] = round(s_kit, 3)
            cand_eval["s_pitch"] = round(s_pitch, 3)
            cand_eval["s_vel"] = round(s_vel, 3)
            cand_eval["possession_score"] = round(total_score, 4)
            scored_candidates.append(cand_eval)

        # Sort descending by possession score
        scored_candidates.sort(key=lambda x: x["possession_score"], reverse=True)
        return scored_candidates

    def resolve_anchor(
        self,
        candidates: Sequence[Dict[str, Any]],
        ball: Optional[Dict[str, Any]],
        event: Any,
    ) -> Optional[GroundTruthAnchorNode]:
        """
        Determines the single true ball-carrier and anchors them with StatsBomb ground truth.

        Args:
            candidates: Localized player candidate dictionaries.
            ball: Localized ball detection dictionary (pitch_x, pitch_y, box, center).
            event: StatsBomb CanonicalMatchEvent or dict.

        Returns:
            GroundTruthAnchorNode if possession is confirmed, or None if no confident candidate.
        """
        if not candidates:
            logger.warning("No player candidates provided to resolve_anchor.")
            return None

        ev_info = self._extract_event_info(event)
        scored = self.evaluate_candidates(candidates, ball, event)
        if not scored:
            return None

        best = scored[0]
        if best["possession_score"] < self.min_possession_threshold:
            logger.warning(
                "Best candidate T%s possession score (%.3f) below threshold (%.3f).",
                best.get("track_id"),
                best["possession_score"],
                self.min_possession_threshold,
            )
            return None

        # Build GroundTruthAnchorNode
        px = best.get("pitch_x", ev_info["pitch_x_m"])
        py = best.get("pitch_y", ev_info["pitch_y_m"])

        anchor = GroundTruthAnchorNode(
            track_id=best.get("track_id", 0),
            player_name=ev_info["player_name"],
            team=ev_info["team_name"],
            role="BALL_CARRIER",
            event_type=ev_info["event_type"],
            event_id=ev_info["event_id"],
            timestamp_sec=ev_info["timestamp_sec"],
            pitch_coords=(float(px), float(py)),
            box=list(best.get("box", [0, 0, 0, 0])),
            foot_pos=best.get("foot_point", (0.0, 0.0)),
            confidence=round(min(0.99, max(0.90, best["possession_score"] + 0.10)), 3),
            is_ground_truth=True,
            status="ANCHORED_GROUND_TRUTH",
            possession_score=best["possession_score"],
            telemetry={
                "dist_to_ball_m": best.get("dist_to_ball_m"),
                "dist_to_event_m": best.get("dist_to_event_m"),
                "s_dist": best.get("s_dist"),
                "s_kit": best.get("s_kit"),
                "s_pitch": best.get("s_pitch"),
                "s_vel": best.get("s_vel"),
                "candidate_count": len(candidates),
            },
        )
        logger.info(
            "Anchored Ground Truth: Track T%d -> %s (%s, score=%.3f)",
            anchor.track_id,
            anchor.player_name,
            anchor.team,
            anchor.possession_score,
        )
        return anchor

    def resolve_event_chain(
        self,
        events: Sequence[Any],
        frame_detections: Dict[int, Dict[str, Any]],
        fps: float = 30.0,
    ) -> Dict[int, GroundTruthAnchorNode]:
        """
        Anchors multiple consecutive events along a temporal sequence (e.g. Passer -> Receiver).

        Args:
            events: Sequence of chronological StatsBomb events.
            frame_detections: Dict mapping frame_idx -> {'players': [...], 'ball': {...}, 'timestamp_sec': float}.
            fps: Video frames per second.

        Returns:
            Dict mapping frame_idx -> GroundTruthAnchorNode.
        """
        chain_anchors: Dict[int, GroundTruthAnchorNode] = {}
        for ev in events:
            ev_info = self._extract_event_info(ev)
            ev_t = ev_info["timestamp_sec"]

            # Find closest frame
            best_f_idx = None
            min_dt = float("inf")
            for f_idx, f_data in frame_detections.items():
                f_t = f_data.get("timestamp_sec", f_idx / fps)
                dt = abs(f_t - ev_t)
                if dt < min_dt:
                    min_dt = dt
                    best_f_idx = f_idx

            if best_f_idx is not None and min_dt <= 1.5:
                frame_info = frame_detections[best_f_idx]
                anchor = self.resolve_anchor(
                    candidates=frame_info.get("players", []),
                    ball=frame_info.get("ball"),
                    event=ev,
                )
                if anchor:
                    chain_anchors[best_f_idx] = anchor

        return chain_anchors
