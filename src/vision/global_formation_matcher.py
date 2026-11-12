"""
src/vision/global_formation_matcher.py
Global Formation Matcher & Strict Kit Firewalls
(Fixing Flaw D: Rigid Pitch Corridors & Flaw E: Goalkeeper Penalty Box Overload Bug).

1. Fixes Flaw E: Strictly uses kit color histogram/HSV distance against the registered GK
   kit profile to isolate goalkeepers. Outfield players entering the 18-yard box are
   strictly barred from false GK assignment. Pitch boundaries are used to filter out
   spectators and referees.
2. Fixes Flaw D: Employs Hungarian Bipartite Graph Matching (scipy.optimize.linear_sum_assignment)
   on 2D homography pitch coordinates against tactical formation topology, naturally
   accommodating inverted fullbacks, defensive line compressions, and set-piece overloads.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger("vision.global_formation_matcher")

# Canonical 10-outfielder formations on UEFA standard 105m x 68m pitch
# Defending left goal (0, 34), attacking right goal (105, 34).
CANONICAL_FORMATION_TEMPLATES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "4-3-3": {
        "LB": (25.0, 10.0),
        "LCB": (20.0, 24.0),
        "RCB": (20.0, 44.0),
        "RB": (25.0, 58.0),
        "DM": (38.0, 34.0),
        "LCM": (52.0, 22.0),
        "RCM": (52.0, 46.0),
        "LW": (75.0, 12.0),
        "ST": (82.0, 34.0),
        "RW": (75.0, 56.0),
    },
    "4-2-3-1": {
        "LB": (25.0, 10.0),
        "LCB": (20.0, 24.0),
        "RCB": (20.0, 44.0),
        "RB": (25.0, 58.0),
        "LDM": (38.0, 26.0),
        "RDM": (38.0, 42.0),
        "LAM": (65.0, 14.0),
        "CAM": (68.0, 34.0),
        "RAM": (65.0, 54.0),
        "ST": (84.0, 34.0),
    },
    "3-5-2": {
        "LCB": (22.0, 18.0),
        "CB": (18.0, 34.0),
        "RCB": (22.0, 50.0),
        "LWB": (45.0, 8.0),
        "LDM": (40.0, 25.0),
        "RDM": (40.0, 43.0),
        "RWB": (45.0, 60.0),
        "AM": (62.0, 34.0),
        "LS": (80.0, 26.0),
        "RS": (80.0, 42.0),
    },
    "4-4-2": {
        "LB": (25.0, 10.0),
        "LCB": (20.0, 24.0),
        "RCB": (20.0, 44.0),
        "RB": (25.0, 58.0),
        "LM": (50.0, 10.0),
        "LCM": (50.0, 26.0),
        "RCM": (50.0, 42.0),
        "RM": (50.0, 58.0),
        "LS": (80.0, 26.0),
        "RS": (80.0, 42.0),
    },
}

ROLE_EQUIVALENCE_GROUPS: Dict[str, List[str]] = {
    "GK": ["GK", "GOALKEEPER"],
    "LB": ["LB", "LWB", "DEFENDER", "DF"],
    "RB": ["RB", "RWB", "DEFENDER", "DF"],
    "LCB": ["LCB", "CB", "DEFENDER", "DF"],
    "RCB": ["RCB", "CB", "DEFENDER", "DF"],
    "CB": ["CB", "LCB", "RCB", "DEFENDER", "DF"],
    "DM": ["DM", "CDM", "LDM", "RDM", "CM", "MIDFIELDER", "MF"],
    "LDM": ["LDM", "DM", "CDM", "CM", "MIDFIELDER", "MF"],
    "RDM": ["RDM", "DM", "CDM", "CM", "MIDFIELDER", "MF"],
    "LCM": ["LCM", "CM", "AM", "LM", "MIDFIELDER", "MF"],
    "RCM": ["RCM", "CM", "AM", "RM", "MIDFIELDER", "MF"],
    "CM": ["CM", "LCM", "RCM", "DM", "AM", "MIDFIELDER", "MF"],
    "CAM": ["CAM", "AM", "CM", "SS", "MIDFIELDER", "MF", "FW"],
    "LAM": ["LAM", "LW", "LM", "AM", "MIDFIELDER", "MF", "FW"],
    "RAM": ["RAM", "RW", "RM", "AM", "MIDFIELDER", "MF", "FW"],
    "LW": ["LW", "LM", "LAM", "LWF", "FORWARD", "ST", "FW"],
    "RW": ["RW", "RM", "RAM", "RWF", "FORWARD", "ST", "FW"],
    "ST": ["ST", "CF", "LS", "RS", "FORWARD", "ATTACKER", "FW"],
    "LS": ["LS", "ST", "CF", "FORWARD", "ATTACKER", "FW"],
    "RS": ["RS", "ST", "CF", "FORWARD", "ATTACKER", "FW"],
    "LWB": ["LWB", "LB", "LM", "DEFENDER", "DF"],
    "RWB": ["RWB", "RB", "RM", "DEFENDER", "DF"],
}


class GlobalFormationMatcher:
    """
    Robust Formation Topology Solver with Goalkeeper Kit Firewalls.
    """

    def __init__(
        self,
        gk_color_threshold: float = 38.0,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ):
        """
        Args:
            gk_color_threshold: Maximum color distance threshold to register as Goalkeeper kit.
            pitch_length_m: Pitch length along X (meters).
            pitch_width_m: Pitch width along Y (meters).
        """
        self.gk_color_threshold = float(gk_color_threshold)
        self.pitch_length_m = float(pitch_length_m)
        self.pitch_width_m = float(pitch_width_m)

    @staticmethod
    def compute_kit_color_distance(hsv_a: Sequence[float], hsv_b: Sequence[float]) -> float:
        """
        Computes kit color distance in HSV color space.
        Accounts for OpenCV Hue circularity [0..180].
        """
        ha, sa, va = float(hsv_a[0]), float(hsv_a[1]), float(hsv_a[2])
        hb, sb, vb = float(hsv_b[0]), float(hsv_b[1]), float(hsv_b[2])

        # Angular hue difference in [0..90]
        dh = abs(ha - hb)
        if dh > 90.0:
            dh = 180.0 - dh

        ds = abs(sa - sb)
        dv = abs(va - vb)

        # Standard weighted Euclidean distance
        dist = math.sqrt((dh * 1.2) ** 2 + (ds * 0.8) ** 2 + (dv * 0.5) ** 2)
        return float(dist)

    def filter_goalkeepers(
        self,
        candidates: Sequence[Dict[str, Any]],
        gk_kit_hsv: Sequence[float],
        referee_kit_hsv: Optional[Sequence[float]] = None,
        defending_goal_x: Optional[float] = None,
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Strictly uses kit color histogram/HSV distance to isolate goalkeeper.
        Outfield players in the penalty box are strictly excluded from GK labeling.
        Pitch coordinates and referee color profiles are used to disambiguate against referees/spectators.

        Args:
            candidates: List of player track candidates. Each may contain 'mean_hsv',
                        'pitch_x', 'pitch_y', 'is_referee', etc.
            gk_kit_hsv: Registered HSV color profile of the official goalkeeper kit.
            referee_kit_hsv: Optional HSV color profile of referee.
            defending_goal_x: Optional X coordinate of team's defending goal (e.g. 0.0 or 105.0).

        Returns:
            Tuple of (best_gk_candidate_or_None, list_of_outfield_candidates).
        """
        best_gk: Optional[Dict[str, Any]] = None
        min_score = float("inf")
        outfield_candidates: List[Dict[str, Any]] = []

        target_gk_hsv = np.array(gk_kit_hsv, dtype=np.float32)

        for cand in candidates:
            # Exclude explicit referees or touchline officials
            if cand.get("is_referee", False) or cand.get("role") == "referee":
                continue

            px = cand.get("pitch_x")
            py = cand.get("pitch_y")

            # Pitch coordinate boundary filter: Disambiguate against touchline spectators/staff
            if px is not None and py is not None:
                # Discard spectators/cameramen beyond touchlines or deep behind goal boards
                if py < -1.5 or py > self.pitch_width_m + 1.5 or px < -3.0 or px > self.pitch_length_m + 3.0:
                    continue

            crop_hsv = np.array(cand.get("mean_hsv", [0, 0, 0]), dtype=np.float32)

            # Direct Euclidean norm check (backward compatible with original plan)
            raw_euclidean_dist = float(np.linalg.norm(crop_hsv - target_gk_hsv))
            circular_dist = self.compute_kit_color_distance(crop_hsv, target_gk_hsv)
            dist = min(raw_euclidean_dist, circular_dist)

            # Check if closer to referee kit than GK kit
            if referee_kit_hsv is not None:
                ref_hsv = np.array(referee_kit_hsv, dtype=np.float32)
                ref_dist = min(
                    float(np.linalg.norm(crop_hsv - ref_hsv)),
                    self.compute_kit_color_distance(crop_hsv, ref_hsv),
                )
                if ref_dist < dist:
                    # Referee match, exclude from GK and outfield
                    continue

            # Goalkeeper qualification: strictly kit color below threshold
            if dist < self.gk_color_threshold:
                # Use pitch coordinates to disambiguate against central referees or midfield outfielders
                disambiguation_penalty = 0.0
                if px is not None:
                    # If defending goal is known, measure distance to defending goal line
                    if defending_goal_x is not None:
                        goal_dist = abs(float(px) - float(defending_goal_x))
                    else:
                        # Goalkeepers stay within 35m of either goal line
                        goal_dist = min(abs(float(px) - 0.0), abs(float(px) - self.pitch_length_m))
                    # Central pitch penalty (referees commonly operate around mid-pitch 40-65m)
                    if goal_dist > 35.0:
                        disambiguation_penalty += (goal_dist - 35.0) * 0.8

                score = dist + disambiguation_penalty
                if score < min_score:
                    min_score = score
                    if best_gk is not None:
                        outfield_candidates.append(best_gk)
                    best_gk = cand
                else:
                    outfield_candidates.append(cand)
            else:
                outfield_candidates.append(cand)

        return best_gk, outfield_candidates

    def solve_formation_assignment(
        self,
        outfield_pitch_coords: np.ndarray,
        formation_template: Union[str, Dict[str, Tuple[float, float]]],
        unassigned_profiles: Sequence[Dict[str, Any]],
        attack_direction: str = "right",
        relative_to_centroid: bool = True,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Solves optimal role assignment using the Hungarian Algorithm (linear_sum_assignment)
        on 2D pitch coordinates against canonical formation roles.
        Supports Relative Formation Graph Matching relative to team centroid to naturally
        accommodate high defensive lines, inverted fullbacks, and set-piece overloads.

        Args:
            outfield_pitch_coords: (N, 2) array of detected outfielder pitch coordinates (in meters).
            formation_template: Name (e.g. '4-3-3') or dict mapping role name -> (x_m, y_m).
            unassigned_profiles: Roster profiles available for assignment.
            attack_direction: 'right' (0 -> 105) or 'left' (105 -> 0).
            relative_to_centroid: If True, aligns formation template centroid with the
                                  detected players' centroid to match relative spatial topology.

        Returns:
            Dict mapping player_idx -> {
                'role': assigned_role,
                'target_pos': (x, y),
                'cost': distance_cost,
                'profile': matched_profile_dict_or_None
            }
        """
        if isinstance(formation_template, str):
            template_dict = CANONICAL_FORMATION_TEMPLATES.get(
                formation_template,
                CANONICAL_FORMATION_TEMPLATES["4-3-3"],
            )
        else:
            template_dict = formation_template

        coords = np.asarray(outfield_pitch_coords, dtype=np.float32)
        n_players = len(coords)
        if n_players == 0:
            return {}

        roles = list(template_dict.keys())
        target_positions = np.array([template_dict[r] for r in roles], dtype=np.float32)

        # Invert template positions along X and Y if attacking left
        if attack_direction == "left":
            inverted_targets = np.empty_like(target_positions)
            inverted_targets[:, 0] = self.pitch_length_m - target_positions[:, 0]
            inverted_targets[:, 1] = self.pitch_width_m - target_positions[:, 1]
            target_positions = inverted_targets

        # Relative formation topology: align template centroid with detected team centroid
        if relative_to_centroid and n_players >= 3:
            coords_centroid = np.mean(coords, axis=0)
            target_centroid = np.mean(target_positions, axis=0)
            comparison_targets = target_positions - target_centroid + coords_centroid
        else:
            comparison_targets = target_positions

        # Pairwise Euclidean distance cost matrix: shape (N, M)
        cost_matrix = np.zeros((n_players, len(roles)), dtype=np.float32)
        for i in range(n_players):
            for j in range(len(roles)):
                cost_matrix[i, j] = np.linalg.norm(coords[i] - comparison_targets[j])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments: Dict[int, Dict[str, Any]] = {}
        available_profiles = list(unassigned_profiles)

        for r, c in zip(row_ind, col_ind):
            assigned_role = roles[c]
            target_pos = (float(comparison_targets[c, 0]), float(comparison_targets[c, 1]))
            cost = float(cost_matrix[r, c])

            # Profile matching with role equivalence hierarchy
            matched_profile: Optional[Dict[str, Any]] = None
            equivalent_roles = ROLE_EQUIVALENCE_GROUPS.get(assigned_role, [assigned_role])

            # 1. Exact role match
            for p in available_profiles:
                p_role = str(p.get("role") or p.get("position") or "").upper()
                if p_role == assigned_role:
                    matched_profile = p
                    break

            # 2. Equivalent group match (e.g. LCB -> CB, ST -> CF)
            if matched_profile is None:
                for p in available_profiles:
                    p_role = str(p.get("role") or p.get("position") or "").upper()
                    if p_role in equivalent_roles:
                        matched_profile = p
                        break

            # 3. Fallback to any remaining outfield profile
            if matched_profile is None and available_profiles:
                # Exclude GK profile from outfield assignments
                outfield_p = [
                    p for p in available_profiles
                    if str(p.get("role") or p.get("position") or "").upper() not in ("GK", "GOALKEEPER")
                ]
                if outfield_p:
                    matched_profile = outfield_p[0]
                else:
                    matched_profile = available_profiles[0]

            if matched_profile is not None:
                available_profiles.remove(matched_profile)

            assignments[int(r)] = {
                "role": assigned_role,
                "target_pos": target_pos,
                "cost": round(cost, 2),
                "profile": matched_profile,
            }

        return assignments
