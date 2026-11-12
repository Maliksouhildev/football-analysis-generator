"""
src/vision/tactical_subgraph_selector.py
Action-Relevant Tactical Subgraph Extractor.

Selects the key tactical actors around the ball without requiring 22-player brute-force labeling:
1. Ball Carrier: The verified anchor player in possession.
2. Immediate Pressers / Tacklers: Opponents within a dynamic pressing radius (R <= 4.5m).
3. Passing Options: Teammates in open passing lanes with clear line-of-sight.
4. Lane Blockers: Opponents positioned directly in the corridor between carrier and teammates.
5. Off-the-Ball Runners: Teammates penetrating forward spaces into attacking zones.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("vision.tactical_subgraph_selector")


class TacticalRole(str, enum.Enum):
    BALL_CARRIER = "BALL_CARRIER"
    IMMEDIATE_PRESSER = "IMMEDIATE_PRESSER"
    PASSING_OPTION = "PASSING_OPTION"
    LANE_BLOCKER = "LANE_BLOCKER"
    OFF_BALL_RUNNER = "OFF_BALL_RUNNER"
    SUPPORT_DEFENDER = "SUPPORT_DEFENDER"
    OUTFIELD_UNINVOLVED = "OUTFIELD_UNINVOLVED"


@dataclasses.dataclass
class PassingLane:
    """Represents a passing vector between the ball carrier and a candidate teammate."""
    receiver_track_id: int
    receiver_name: str
    receiver_pos: Tuple[float, float]
    distance_m: float
    angle_deg: float
    is_open: bool
    openness_score: float  # 1.0 = completely clear, 0.0 = fully obstructed
    blockers: List[int] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receiver_track_id": self.receiver_track_id,
            "receiver_name": self.receiver_name,
            "receiver_pos": list(self.receiver_pos),
            "distance_m": round(self.distance_m, 2),
            "angle_deg": round(self.angle_deg, 1),
            "is_open": self.is_open,
            "openness_score": round(self.openness_score, 3),
            "blockers": list(self.blockers),
        }


@dataclasses.dataclass
class TacticalSubgraph:
    """Action-relevant subgraph of key players and passing lanes around the ball."""
    carrier: Dict[str, Any]
    pressers: List[Dict[str, Any]]
    passing_options: List[Dict[str, Any]]
    lane_blockers: List[Dict[str, Any]]
    runners: List[Dict[str, Any]]
    tactical_actors: List[Dict[str, Any]]
    passing_lanes: List[PassingLane]
    telemetry: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "carrier": self.carrier,
            "pressers": self.pressers,
            "passing_options": self.passing_options,
            "lane_blockers": self.lane_blockers,
            "runners": self.runners,
            "tactical_actors": self.tactical_actors,
            "passing_lanes": [l.to_dict() for l in self.passing_lanes],
            "telemetry": self.telemetry,
        }


class TacticalSubgraphSelector:
    """
    Constructs an action-relevant tactical proximity graph around the ball carrier.
    """

    def __init__(
        self,
        pressing_radius_m: float = 4.5,
        lane_corridor_width_m: float = 2.2,
        min_pass_distance_m: float = 2.5,
        max_pass_distance_m: float = 45.0,
        runner_forward_gain_m: float = 4.0,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ):
        """
        Args:
            pressing_radius_m: Radius around carrier defining immediate pressure zone.
            lane_corridor_width_m: Width of the corridor along passing line to detect interception.
            min_pass_distance_m: Minimum viable passing distance.
            max_pass_distance_m: Maximum realistic passing distance in subgraph.
            runner_forward_gain_m: Distance ahead of carrier to qualify as an off-the-ball runner.
            pitch_length_m: UEFA pitch length.
            pitch_width_m: UEFA pitch width.
        """
        self.pressing_radius_m = float(pressing_radius_m)
        self.lane_corridor_width_m = float(lane_corridor_width_m)
        self.min_pass_distance_m = float(min_pass_distance_m)
        self.max_pass_distance_m = float(max_pass_distance_m)
        self.runner_forward_gain_m = float(runner_forward_gain_m)
        self.pitch_length_m = float(pitch_length_m)
        self.pitch_width_m = float(pitch_width_m)

    @staticmethod
    def _get_player_coords(player: Dict[str, Any]) -> Tuple[float, float]:
        """Extracts (pitch_x, pitch_y) from player candidate dictionary."""
        coords = player.get("pitch_coords")
        if coords is not None and isinstance(coords, (list, tuple)) and len(coords) >= 2:
            return (float(coords[0]), float(coords[1]))
        px = player.get("pitch_x")
        py = player.get("pitch_y")
        return (float(px if px is not None else 52.5), float(py if py is not None else 34.0))

    @staticmethod
    def _point_to_line_segment_dist(
        p: Tuple[float, float],
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Calculates perpendicular distance from point p to line segment ab.
        Returns:
            Tuple of (perpendicular_distance, projection_t_parameter).
            t in [0, 1] means the perpendicular projection falls strictly on the segment.
        """
        px, py = p
        ax, ay = a
        bx, by = b

        dx = bx - ax
        dy = by - ay
        l2 = dx * dx + dy * dy

        if l2 == 0.0:
            return (math.hypot(px - ax, py - ay), 0.0)

        # Projection parameter t
        t = ((px - ax) * dx + (py - ay) * dy) / l2
        clamped_t = max(0.0, min(1.0, t))

        proj_x = ax + clamped_t * dx
        proj_y = ay + clamped_t * dy

        dist = math.hypot(px - proj_x, py - proj_y)
        return (float(dist), float(t))

    def evaluate_passing_lane(
        self,
        carrier_pos: Tuple[float, float],
        teammate_pos: Tuple[float, float],
        opponents: Sequence[Dict[str, Any]],
        teammate_track_id: int,
        teammate_name: str,
    ) -> PassingLane:
        """
        Evaluates line-of-sight and obstruction between carrier and a teammate.
        """
        dist = math.hypot(teammate_pos[0] - carrier_pos[0], teammate_pos[1] - carrier_pos[1])
        angle_rad = math.atan2(teammate_pos[1] - carrier_pos[1], teammate_pos[0] - carrier_pos[0])
        angle_deg = math.degrees(angle_rad)

        blockers: List[int] = []
        min_obstruction_dist = float("inf")

        for opp in opponents:
            opp_pos = self._get_player_coords(opp)
            d_perp, t = self._point_to_line_segment_dist(opp_pos, carrier_pos, teammate_pos)

            # Check if opponent is between carrier and teammate (0.05 < t < 0.95)
            # and within half corridor width
            if 0.05 < t < 0.95 and d_perp <= (self.lane_corridor_width_m / 2.0):
                opp_id = opp.get("track_id", 0)
                blockers.append(opp_id)
                if d_perp < min_obstruction_dist:
                    min_obstruction_dist = d_perp

        is_open = len(blockers) == 0

        if is_open:
            openness = 1.0
        else:
            # Score drops with how close the nearest blocker is to the central line
            ratio = min_obstruction_dist / max(0.1, (self.lane_corridor_width_m / 2.0))
            openness = max(0.0, min(0.60, ratio * 0.60))

        return PassingLane(
            receiver_track_id=teammate_track_id,
            receiver_name=teammate_name,
            receiver_pos=teammate_pos,
            distance_m=dist,
            angle_deg=angle_deg,
            is_open=is_open,
            openness_score=openness,
            blockers=blockers,
        )

    def extract_subgraph(
        self,
        carrier: Union[Dict[str, Any], Any],
        all_players: Sequence[Dict[str, Any]],
        attack_direction: str = "right",
    ) -> TacticalSubgraph:
        """
        Builds the tactical subgraph of key actors around the carrier.

        Args:
            carrier: The verified ground truth anchor carrier (GroundTruthAnchorNode or dict).
            all_players: All detected players on the pitch.
            attack_direction: 'right' (X: 0 -> 105) or 'left' (X: 105 -> 0).

        Returns:
            TacticalSubgraph containing carrier, pressers, passing options, lane blockers, and runners.
        """
        # Normalize carrier dict
        if hasattr(carrier, "to_dict"):
            carrier_dict = carrier.to_dict()
        else:
            carrier_dict = dict(carrier)

        carrier_tid = carrier_dict.get("track_id")
        carrier_team = carrier_dict.get("team", "")
        carrier_pos = self._get_player_coords(carrier_dict)

        carrier_actor = copy.deepcopy(carrier_dict)
        carrier_actor["tactical_role"] = TacticalRole.BALL_CARRIER.value

        # Separate teammates and opponents
        teammates: List[Dict[str, Any]] = []
        opponents: List[Dict[str, Any]] = []

        for p in all_players:
            tid = p.get("track_id")
            if tid == carrier_tid:
                continue

            # Skip explicit referees
            if p.get("is_referee", False) or str(p.get("role", "")).lower() == "referee":
                continue

            p_team = p.get("team", "")
            if p_team and carrier_team:
                if p_team.lower() == carrier_team.lower() or p_team.lower() in carrier_team.lower():
                    teammates.append(p)
                else:
                    opponents.append(p)
            else:
                # Fallback if team unassigned: classify based on kit color or default
                teammates.append(p)

        # 1. Identify Immediate Pressers
        pressers: List[Dict[str, Any]] = []
        total_pressure = 0.0

        for opp in opponents:
            opp_pos = self._get_player_coords(opp)
            d = math.hypot(opp_pos[0] - carrier_pos[0], opp_pos[1] - carrier_pos[1])
            if d <= self.pressing_radius_m:
                opp_actor = copy.deepcopy(opp)
                intensity = max(0.0, 1.0 - (d / self.pressing_radius_m))
                opp_actor["tactical_role"] = TacticalRole.IMMEDIATE_PRESSER.value
                opp_actor["dist_to_carrier_m"] = round(d, 2)
                opp_actor["pressing_intensity"] = round(intensity, 3)
                pressers.append(opp_actor)
                total_pressure += intensity

        # Sort pressers by proximity
        pressers.sort(key=lambda x: x["dist_to_carrier_m"])

        # 2. Identify Passing Lanes, Options, and Blockers
        passing_lanes: List[PassingLane] = []
        passing_options: List[Dict[str, Any]] = []
        blocker_tids: set[int] = set()
        runners: List[Dict[str, Any]] = []

        for mate in teammates:
            mate_pos = self._get_player_coords(mate)
            dist = math.hypot(mate_pos[0] - carrier_pos[0], mate_pos[1] - carrier_pos[1])

            # Forward progression check for off-the-ball runner
            dx = mate_pos[0] - carrier_pos[0]
            is_forward = (dx > self.runner_forward_gain_m) if attack_direction == "right" else (dx < -self.runner_forward_gain_m)

            if is_forward and dist <= self.max_pass_distance_m:
                runner_actor = copy.deepcopy(mate)
                runner_actor["tactical_role"] = TacticalRole.OFF_BALL_RUNNER.value
                runner_actor["dist_to_carrier_m"] = round(dist, 2)
                runner_actor["forward_gain_m"] = round(abs(dx), 2)
                runners.append(runner_actor)

            if self.min_pass_distance_m <= dist <= self.max_pass_distance_m:
                mate_tid = mate.get("track_id", 0)
                mate_name = mate.get("player_name", f"Teammate_{mate_tid}")
                lane = self.evaluate_passing_lane(
                    carrier_pos=carrier_pos,
                    teammate_pos=mate_pos,
                    opponents=opponents,
                    teammate_track_id=mate_tid,
                    teammate_name=mate_name,
                )
                passing_lanes.append(lane)

                for b_id in lane.blockers:
                    blocker_tids.add(b_id)

                option_actor = copy.deepcopy(mate)
                option_actor["tactical_role"] = TacticalRole.PASSING_OPTION.value
                option_actor["dist_to_carrier_m"] = round(dist, 2)
                option_actor["lane_openness"] = lane.openness_score
                option_actor["is_lane_open"] = lane.is_open
                passing_options.append(option_actor)

        # Sort passing options by openness and proximity
        passing_options.sort(key=lambda x: (-x["lane_openness"], x["dist_to_carrier_m"]))

        # 3. Identify Lane Blockers
        lane_blockers: List[Dict[str, Any]] = []
        for opp in opponents:
            opp_tid = opp.get("track_id", 0)
            if opp_tid in blocker_tids and opp_tid not in [p["track_id"] for p in pressers]:
                b_actor = copy.deepcopy(opp)
                b_actor["tactical_role"] = TacticalRole.LANE_BLOCKER.value
                lane_blockers.append(b_actor)

        # 4. Assemble Unique Tactical Actors (typically 5 to 9 key players)
        tactical_actors_map: Dict[int, Dict[str, Any]] = {}
        tactical_actors_map[carrier_actor.get("track_id", -1)] = carrier_actor

        for p in pressers:
            tactical_actors_map[p["track_id"]] = p
        for p in passing_options:
            if p["track_id"] not in tactical_actors_map:
                tactical_actors_map[p["track_id"]] = p
        for b in lane_blockers:
            if b["track_id"] not in tactical_actors_map:
                tactical_actors_map[b["track_id"]] = b
        for r in runners:
            if r["track_id"] not in tactical_actors_map:
                tactical_actors_map[r["track_id"]] = r

        all_tactical_actors = list(tactical_actors_map.values())

        telemetry = {
            "carrier_track_id": carrier_tid,
            "carrier_name": carrier_dict.get("player_name"),
            "num_immediate_pressers": len(pressers),
            "pressing_pressure_index": round(total_pressure, 3),
            "num_passing_options": len(passing_options),
            "num_open_passing_lanes": sum(1 for l in passing_lanes if l.is_open),
            "num_blocked_passing_lanes": sum(1 for l in passing_lanes if not l.is_open),
            "num_off_ball_runners": len(runners),
            "total_tactical_actors": len(all_tactical_actors),
        }

        logger.info(
            "Extracted Tactical Subgraph for %s: %d pressers (pressure=%.2f), %d pass options (%d open)",
            carrier_dict.get("player_name", "Carrier"),
            len(pressers),
            total_pressure,
            len(passing_options),
            telemetry["num_open_passing_lanes"],
        )

        return TacticalSubgraph(
            carrier=carrier_actor,
            pressers=pressers,
            passing_options=passing_options,
            lane_blockers=lane_blockers,
            runners=runners,
            tactical_actors=all_tactical_actors,
            passing_lanes=passing_lanes,
            telemetry=telemetry,
        )
