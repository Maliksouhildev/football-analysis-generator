"""
Team and Goalkeeper Color Clusterer.
Solves the K-Means imbalance problem: runs KMeans(k=2) exclusively on outfielders,
isolating Goalkeepers (deepest X near goal) and Referees (central) via outlier distance thresholding.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from sklearn.cluster import KMeans


@dataclass
class TeamClusterResult:
    """Classification result for a single player/actor."""
    actor_id: int  # track_id or detection index
    team: str  # "home", "away", "gk_home", "gk_away", "referee", "unknown"
    confidence: float
    is_outlier: bool
    outlier_reason: Optional[str] = None
    dominant_hsv: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dominant_rgb: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class TeamColorClusterer:
    """
    Robust 2-Step Kit & Role Classifier.
    - Step 1: Extract non-turf upper-torso HSV/RGB color vectors.
    - Step 2: Fit KMeans(n_clusters=2) for outfield teams and flag color outliers.
    - Step 3: Topologically map outliers to Goalkeepers (deep near goal lines) or Referees (central).
    """

    def __init__(
        self,
        home_team_name: str = "Real Madrid",
        away_team_name: str = "Barcelona",
        outlier_std_factor: float = 1.85,
    ):
        self.home_team_name = home_team_name
        self.away_team_name = away_team_name
        self.outlier_std_factor = outlier_std_factor

    def extract_torso_features(
        self,
        frame: np.ndarray,
        bbox: List[float],
        turf_l_ref: float = 52.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract non-turf color representation from the upper torso crop.
        Filters out green pitch grass bleed and implements shadow-invariant CIE LAB:
        - Torso crop: 22% to 48% box height (dropping collar/neck and shorts waistband).
        - Unscales OpenCV uint8 LAB to true CIE ranges (L* in [0, 100], a*, b* in [-128, 127]).
        - Applies multiplicative cube-root reflectance ratio:
          L*_norm = (L*_turf_ref + 16) * ((L*_torso + 16) / (L*_turf + 16)) - 16.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        box_w, box_h = x2 - x1, y2 - y1

        if box_w < 10 or box_h < 20:
            return None

        # Torso crop: y from 22% to 48% (strictly avoids neck collar and shorts waistband)
        ty1 = y1 + int(box_h * 0.22)
        ty2 = y1 + int(box_h * 0.48)
        tx1 = x1 + int(box_w * 0.15)
        tx2 = x2 - int(box_w * 0.15)

        if ty2 <= ty1 or tx2 <= tx1:
            return None

        torso = frame[ty1:ty2, tx1:tx2]
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

        # Green turf mask: Hue in [32, 85], Sat >= 35, Val >= 35
        turf_mask = cv2.inRange(hsv, (32, 35, 35), (85, 255, 255))
        non_turf_mask = cv2.bitwise_not(turf_mask)

        # Non-turf pixels
        valid_hsv = hsv[non_turf_mask > 0]
        valid_bgr = torso[non_turf_mask > 0]

        if len(valid_hsv) < 15:
            valid_hsv = hsv.reshape(-1, 3)
            valid_bgr = torso.reshape(-1, 3)

        med_hsv = np.median(valid_hsv, axis=0)
        med_bgr = np.median(valid_bgr, axis=0)

        # ---------------------------------------------------------------------
        # Local Turf Lightness Extraction (L*_turf)
        # ---------------------------------------------------------------------
        # Sample pitch turf directly beneath feet / adjacent to player
        fy1 = min(h, y2)
        fy2 = min(h, y2 + max(8, int(box_h * 0.25)))
        fx1 = max(0, x1 - 10)
        fx2 = min(w, x2 + 10)
        foot_patch = frame[fy1:fy2, fx1:fx2]
        
        l_turf_val = None
        if foot_patch.size > 0:
            foot_hsv = cv2.cvtColor(foot_patch, cv2.COLOR_BGR2HSV)
            green_foot_mask = cv2.inRange(foot_hsv, (32, 35, 35), (85, 255, 255))
            if cv2.countNonZero(green_foot_mask) >= 12:
                foot_lab = cv2.cvtColor(foot_patch, cv2.COLOR_BGR2Lab)
                green_lab = foot_lab[green_foot_mask > 0].astype(np.float32)
                # Unscale OpenCV uint8 L_cv2 to true CIE L*
                l_turf_val = float(np.median(green_lab[:, 0])) * (100.0 / 255.0)

        # Flank sampling fallback: if player is at bottom frame border, sample turf flanking the box
        if l_turf_val is None:
            flank_left = frame[y1:y2, max(0, x1 - 25):x1]
            flank_right = frame[y1:y2, x2:min(w, x2 + 25)]
            for patch in (flank_left, flank_right):
                if patch.size > 0:
                    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                    patch_mask = cv2.inRange(patch_hsv, (32, 35, 35), (85, 255, 255))
                    if cv2.countNonZero(patch_mask) >= 15:
                        patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2Lab)
                        green_lab = patch_lab[patch_mask > 0].astype(np.float32)
                        l_turf_val = float(np.median(green_lab[:, 0])) * (100.0 / 255.0)
                        break

        if l_turf_val is None:
            l_turf_val = turf_l_ref

        # ---------------------------------------------------------------------
        # Torso CIE LAB Unscaling & Shadow-Invariant Cube-Root Normalization
        # ---------------------------------------------------------------------
        torso_lab = cv2.cvtColor(torso, cv2.COLOR_BGR2Lab)
        valid_lab = torso_lab[non_turf_mask > 0].astype(np.float32)
        if len(valid_lab) < 15:
            valid_lab = torso_lab.reshape(-1, 3).astype(np.float32)

        med_lab_cv2 = np.median(valid_lab, axis=0)
        # Unscale OpenCV uint8 LAB to true CIE ranges:
        # L* in [0, 100], a* in [-128, 127], b* in [-128, 127]
        l_torso_cie = float(med_lab_cv2[0]) * (100.0 / 255.0)
        a_torso_cie = float(med_lab_cv2[1]) - 128.0
        b_torso_cie = float(med_lab_cv2[2]) - 128.0

        # Physical shadow invariance: L* + 16 scales with (k * R)^(1/3)
        # Ratio cancels illumination multiplier k:
        l_norm = (turf_l_ref + 16.0) * ((l_torso_cie + 16.0) / max(1.0, l_turf_val + 16.0)) - 16.0
        l_norm = float(np.clip(l_norm, 0.0, 100.0))

        # Normalized feature vector for clustering:
        # Balanced: 0.5 * (L*_norm - 52.0) on lightness, 1.0 * a*, 1.0 * b* on chrominance
        norm_feat = np.array([
            (0.5 * (l_norm - 52.0)) / 50.0,
            a_torso_cie / 128.0,
            b_torso_cie / 128.0,
            med_hsv[0] / 180.0,
            med_hsv[1] / 255.0,
            med_hsv[2] / 255.0,
            med_bgr[2] / 255.0,  # R
            med_bgr[1] / 255.0,  # G
            med_bgr[0] / 255.0,  # B
        ], dtype=np.float32)

        return {
            "norm_feat": norm_feat,
            "lab": (l_norm, a_torso_cie, b_torso_cie),
            "hsv": (float(med_hsv[0]), float(med_hsv[1]), float(med_hsv[2])),
            "rgb": (float(med_bgr[2]), float(med_bgr[1]), float(med_bgr[0])),
        }

    def cluster_actors(
        self,
        frame: np.ndarray,
        actors: List[Tuple[int, List[float]]],  # List of (actor_id, [x1, y1, x2, y2])
        frame_width: Optional[int] = None,
    ) -> Dict[int, TeamClusterResult]:
        """
        Cluster actors into Home, Away, Goalkeeper, and Referee.
        """
        if not actors:
            return {}

        img_w = frame_width or frame.shape[1]

        # Extract features for all actors
        valid_records = []
        for actor_id, bbox in actors:
            feats = self.extract_torso_features(frame, bbox)
            if feats is not None:
                cx = (bbox[0] + bbox[2]) / 2.0
                valid_records.append((actor_id, bbox, feats, cx))

        if len(valid_records) < 2:
            return {
                actor_id: TeamClusterResult(
                    actor_id=actor_id,
                    team="unknown",
                    confidence=0.5,
                    is_outlier=False,
                )
                for actor_id, _ in actors
            }

        feats_matrix = np.stack([r[2]["norm_feat"] for r in valid_records])

        # Fit KMeans(k=2) to identify the two dominant outfield kit clusters
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(feats_matrix)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        # Count members in each cluster
        count_0 = int(np.count_nonzero(labels == 0))
        count_1 = int(np.count_nonzero(labels == 1))

        # Compute distance to assigned centroid
        distances = np.zeros(len(valid_records), dtype=np.float32)
        for i, feat in enumerate(feats_matrix):
            c_idx = labels[i]
            distances[i] = np.linalg.norm(feat - centroids[c_idx])

        # Robust threshold: distance must exceed 2.2x the median intra-cluster distance
        med_d = float(np.median(distances))
        outlier_thresh = max(0.25, med_d * self.outlier_std_factor)

        inlier_indices = []
        outlier_indices = []

        for i, d in enumerate(distances):
            c_idx = labels[i]
            cluster_count = count_0 if c_idx == 0 else count_1

            # A box is an outlier if:
            # 1. It belongs to a tiny minority cluster (<= 2 players while total >= 8)
            # 2. Its feature distance to its centroid exceeds the outlier threshold
            is_minority = (cluster_count <= 2 and len(valid_records) >= 8)
            is_far_from_both = (
                np.linalg.norm(feats_matrix[i] - centroids[0]) > outlier_thresh and
                np.linalg.norm(feats_matrix[i] - centroids[1]) > outlier_thresh
            )

            if is_minority or is_far_from_both:
                outlier_indices.append(i)
            else:
                inlier_indices.append(i)

        # Refit KMeans(k=2) strictly on inliers if we have enough
        if len(inlier_indices) >= 4:
            inlier_feats = feats_matrix[inlier_indices]
            kmeans_inliers = KMeans(n_clusters=2, random_state=42, n_init=10).fit(inlier_feats)
            c0, c1 = kmeans_inliers.cluster_centers_[0], kmeans_inliers.cluster_centers_[1]
        else:
            c0, c1 = centroids[0], centroids[1]

        # Determine which cluster is Home (White) vs Away (Blaugrana/Dark)
        # Real Madrid (White): High L*_norm, High V, low S. Barcelona: Lower L*, High S, blue/red hues.
        # c0/c1 feature format: [L*_norm, a*, b*, H/180, S/255, V/255, R/255, G/255, B/255]
        l0, s0, v0 = c0[0], c0[4], c0[5]
        l1, s1, v1 = c1[0], c1[4], c1[5]

        # Whiteness score = L*_norm + (V - S)
        white_score_0 = l0 + (v0 - s0)
        white_score_1 = l1 + (v1 - s1)

        if white_score_0 > white_score_1:
            home_cluster_idx = 0
            away_cluster_idx = 1
        else:
            home_cluster_idx = 1
            away_cluster_idx = 0

        # Classify all actors
        results: Dict[int, TeamClusterResult] = {}

        # 1. Classify Outfield inliers
        for i in inlier_indices:
            actor_id, bbox, feat_dict, cx = valid_records[i]
            feat = feat_dict["norm_feat"]
            d0 = np.linalg.norm(feat - c0)
            d1 = np.linalg.norm(feat - c1)

            assigned_cluster = 0 if d0 < d1 else 1
            team = "home" if assigned_cluster == home_cluster_idx else "away"
            conf = 0.85 + 0.10 * max(0.0, 1.0 - (min(d0, d1) / (outlier_thresh + 1e-5)))

            results[actor_id] = TeamClusterResult(
                actor_id=actor_id,
                team=team,
                confidence=round(min(0.98, conf), 2),
                is_outlier=False,
                dominant_hsv=feat_dict["hsv"],
                dominant_rgb=feat_dict["rgb"],
            )

        # 2. Classify Outliers: Goalkeeper vs Referee via spatial topology
        # Find team extremities (leftmost and rightmost X)
        all_cxs = [r[3] for r in valid_records]
        min_x = min(all_cxs) if all_cxs else 0.0
        max_x = max(all_cxs) if all_cxs else float(img_w)
        pitch_span = max(100.0, max_x - min_x)

        for i in outlier_indices:
            actor_id, bbox, feat_dict, cx = valid_records[i]
            h, s, v = feat_dict["hsv"]

            rel_x = (cx - min_x) / pitch_span  # 0.0 to 1.0

            # Spatial check:
            # Real Madrid (White, Home) is defending the left goal (rel_x < 0.35)
            # Barcelona (Away) is defending the right goal (rel_x > 0.65)
            if rel_x < 0.35:
                team = "gk_home"
                reason = f"Goalkeeper near left goal line / Real Madrid box (rel_x={rel_x:.2f})"
            elif rel_x > 0.65:
                team = "gk_away"
                reason = f"Goalkeeper near right goal line / Barcelona box (rel_x={rel_x:.2f})"
            else:
                # Central player with outlier color -> Referee
                team = "referee"
                reason = f"Central player with distinct kit (rel_x={rel_x:.2f}, H={h:.0f}, S={s:.0f}, V={v:.0f})"

            results[actor_id] = TeamClusterResult(
                actor_id=actor_id,
                team=team,
                confidence=0.92,
                is_outlier=True,
                outlier_reason=reason,
                dominant_hsv=feat_dict["hsv"],
                dominant_rgb=feat_dict["rgb"],
            )

        # Enforce strictly Max 1 Match Referee among outliers
        ref_ids = [aid for aid, r in results.items() if r.team == "referee"]
        if len(ref_ids) > 1:
            # Score referees by proximity to pitch center (rel_x=0.50)
            def ref_center_dist(aid):
                cx_val = next(r[3] for r in valid_records if r[0] == aid)
                rel = (cx_val - min_x) / pitch_span
                return abs(rel - 0.50)

            ref_ids.sort(key=ref_center_dist)
            for extra_id in ref_ids[1:]:
                rec = next(r for r in valid_records if r[0] == extra_id)
                feat = rec[2]["norm_feat"]
                d0 = np.linalg.norm(feat - c0)
                d1 = np.linalg.norm(feat - c1)
                assigned_cluster = 0 if d0 < d1 else 1
                fallback_team = "home" if assigned_cluster == home_cluster_idx else "away"

                results[extra_id] = TeamClusterResult(
                    actor_id=extra_id,
                    team=fallback_team,
                    confidence=0.72,
                    is_outlier=False,
                    dominant_hsv=rec[2]["hsv"],
                    dominant_rgb=rec[2]["rgb"],
                )

        # Fill in any missing actors
        for actor_id, _ in actors:
            if actor_id not in results:
                results[actor_id] = TeamClusterResult(
                    actor_id=actor_id,
                    team="unknown",
                    confidence=0.5,
                    is_outlier=False,
                )

        return results
