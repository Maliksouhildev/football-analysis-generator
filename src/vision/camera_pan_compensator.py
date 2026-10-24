"""
src/vision/camera_pan_compensator.py
OpenCV Pitch Homography & Camera Pan Compensator for Broadcast Football Footage.

Compensates for broadcast camera motion (pan, tilt, zoom) between reference frames
in the temporal burst window [t - 2.0s to t + 4.5s] relative to the tactical freeze-frame (t = 0.0s).
Uses pitch line/keypoint matching (SIFT/ORB + cv2.findHomography with RANSAC) and turf-constrained
sparse optical flow on pitch markings to project future ball landing points and player run coordinates
accurately back into the (x, y) pixel/percentage coordinate space of frame_freeze.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("vision.camera_pan_compensator")


class CameraPanCompensator:
    """
    Computes planar homography matrices between temporal frames and projects coordinates
    back to the tactical freeze-frame (t = 0.0s) coordinate space.
    """

    def __init__(
        self,
        frame_w: int = 1920,
        frame_h: int = 1080,
        ransac_thresh: float = 4.0,
        min_inliers: int = 12,
        feature_detector: str = "sift",
    ):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.ransac_thresh = ransac_thresh
        self.min_inliers = min_inliers
        self.feature_detector_name = feature_detector.lower()

        # Initialize detector
        self._sift = None
        self._orb = None
        self._init_detectors()

    def _init_detectors(self):
        try:
            self._sift = cv2.SIFT_create(nfeatures=2500)
        except Exception as e:
            logger.debug("SIFT unavailable (%s), defaulting to ORB", e)
            self._sift = None

        try:
            self._orb = cv2.ORB_create(nfeatures=3000)
        except Exception as e:
            logger.debug("ORB unavailable: %s", e)
            self._orb = None

    @staticmethod
    def extract_pitch_mask(
        frame: np.ndarray,
        mask_scoreboard: bool = True,
        h_min: int = 30,
        h_max: int = 88,
        s_min: int = 25,
        s_max: int = 255,
        v_min: int = 25,
        v_max: int = 255,
    ) -> np.ndarray:
        """
        Extracts pitch grass turf mask in HSV color space.
        Excludes scoreboard/header area (top 12%) and bottom ticker (bottom 2%)
        to isolate the planar pitch turf surface.
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_turf = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper_turf = np.array([h_max, s_max, v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_turf, upper_turf)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        if mask_scoreboard:
            # Zero out top 12% (scoreboard, upper stands, stadium roof)
            top_cut = int(h * 0.12)
            mask[:top_cut, :] = 0
            # Zero out bottom 2% (broadcast ticker)
            bot_cut = int(h * 0.98)
            mask[bot_cut:, :] = 0

        return mask

    @staticmethod
    def extract_pitch_lines_mask(frame: np.ndarray, pitch_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extracts white pitch lines (touchlines, penalty boxes, center circle) within the pitch turf.
        """
        h, w = frame.shape[:2]
        if pitch_mask is None:
            pitch_mask = CameraPanCompensator.extract_pitch_mask(frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # White lines have low saturation and high value
        lower_white = np.array([0, 0, 160], dtype=np.uint8)
        upper_white = np.array([180, 75, 255], dtype=np.uint8)
        line_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Must lie within the pitch grass region
        line_mask = cv2.bitwise_and(line_mask, pitch_mask)

        # Thin morphological opening to filter out white jerseys/socks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel)
        return line_mask

    def estimate_homography(
        self,
        frame_src: np.ndarray,
        frame_dst: np.ndarray,
        mask_src: Optional[np.ndarray] = None,
        mask_dst: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Estimates the 3x3 planar homography matrix H such that:
            p_dst ~ H * p_src
        where frame_src is typically a future frame (e.g. frame_next_2 at t + 3.0s)
        and frame_dst is the tactical freeze-frame (frame_freeze at t = 0.0s).

        Returns:
            (H, metadata_dict) where H is the 3x3 float64 matrix (or identity on failure),
            and metadata_dict contains inlier count, method, and validity flags.
        """
        metadata: Dict[str, Any] = {
            "success": False,
            "inliers": 0,
            "total_matches": 0,
            "method": "none",
            "condition_number": None,
            "determinant": None,
        }

        if frame_src is None or frame_dst is None:
            logger.warning("Invalid input frames for homography estimation")
            return np.eye(3, dtype=np.float64), metadata

        h_s, w_s = frame_src.shape[:2]
        h_d, w_d = frame_dst.shape[:2]

        # Extract pitch masks if not provided
        if mask_src is None:
            mask_src = self.extract_pitch_mask(frame_src)
        if mask_dst is None:
            mask_dst = self.extract_pitch_mask(frame_dst)

        # 1. Primary Strategy: Feature Detection & Matching (SIFT or ORB on pitch turf)
        H, meta = self._match_features_homography(frame_src, frame_dst, mask_src, mask_dst)
        if H is not None and self._validate_homography(H, w_s, h_s, w_d, h_d):
            metadata.update(meta)
            metadata["success"] = True
            metadata["method"] = meta.get("detector", "features")
            return H, metadata

        # 2. Secondary Strategy: Lucas-Kanade Optical Flow on Pitch Markings
        logger.debug("Feature matching homography fell back; attempting Lucas-Kanade pitch optical flow")
        H_of, meta_of = self._optical_flow_homography(frame_src, frame_dst, mask_src, mask_dst)
        if H_of is not None and self._validate_homography(H_of, w_s, h_s, w_d, h_d):
            metadata.update(meta_of)
            metadata["success"] = True
            metadata["method"] = "optical_flow"
            return H_of, metadata

        # 3. Tertiary Strategy: Robust Affine Partial Estimate (translation + rotation + uniform scale)
        logger.debug("Optical flow homography fell back; attempting partial affine estimate")
        H_aff, meta_aff = self._affine_partial_fallback(frame_src, frame_dst, mask_src, mask_dst)
        if H_aff is not None and self._validate_homography(H_aff, w_s, h_s, w_d, h_d):
            metadata.update(meta_aff)
            metadata["success"] = True
            metadata["method"] = "affine_partial"
            return H_aff, metadata

        # Safe fallback: Identity matrix
        logger.warning("All pitch homography estimation strategies failed; defaulting to identity matrix")
        metadata["method"] = "identity_fallback"
        metadata["success"] = False
        return np.eye(3, dtype=np.float64), metadata

    def _match_features_homography(
        self,
        frame_src: np.ndarray,
        frame_dst: np.ndarray,
        mask_src: np.ndarray,
        mask_dst: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        meta: Dict[str, Any] = {"detector": "none", "inliers": 0, "total_matches": 0}

        gray_src = cv2.cvtColor(frame_src, cv2.COLOR_BGR2GRAY)
        gray_dst = cv2.cvtColor(frame_dst, cv2.COLOR_BGR2GRAY)

        detector = self._sift if (self.feature_detector_name == "sift" and self._sift is not None) else self._orb
        det_name = "sift" if detector == self._sift else "orb"
        meta["detector"] = det_name

        if detector is None:
            return None, meta

        kp_src, desc_src = detector.detectAndCompute(gray_src, mask_src)
        kp_dst, desc_dst = detector.detectAndCompute(gray_dst, mask_dst)

        if desc_src is None or desc_dst is None or len(kp_src) < 8 or len(kp_dst) < 8:
            return None, meta

        # Match descriptors
        if det_name == "sift":
            index_params = dict(algorithm=1, trees=5)  # FLANN KDTree
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc_src, desc_dst, k=2)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = matcher.knnMatch(desc_src, desc_dst, k=2)

        # Lowe's ratio test
        good_matches = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        meta["total_matches"] = len(good_matches)
        if len(good_matches) < self.min_inliers:
            return None, meta

        src_pts = np.float32([kp_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)
        num_inliers = int(np.sum(inliers)) if inliers is not None else 0
        meta["inliers"] = num_inliers

        if H is not None and num_inliers >= self.min_inliers:
            return H.astype(np.float64), meta

        return None, meta

    def _optical_flow_homography(
        self,
        frame_src: np.ndarray,
        frame_dst: np.ndarray,
        mask_src: np.ndarray,
        mask_dst: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        meta: Dict[str, Any] = {"detector": "optical_flow", "inliers": 0, "total_matches": 0}
        gray_src = cv2.cvtColor(frame_src, cv2.COLOR_BGR2GRAY)
        gray_dst = cv2.cvtColor(frame_dst, cv2.COLOR_BGR2GRAY)

        # Detect salient corners on pitch grass and pitch lines
        pts_src = cv2.goodFeaturesToTrack(
            gray_src,
            mask=mask_src,
            maxCorners=1200,
            qualityLevel=0.015,
            minDistance=12,
            blockSize=5,
        )
        if pts_src is None or len(pts_src) < self.min_inliers:
            return None, meta

        # Forward optical flow
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        pts_dst, status, err = cv2.calcOpticalFlowPyrLK(gray_src, gray_dst, pts_src, None, **lk_params)

        # Backward optical flow for cycle-consistency verification
        pts_src_back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray_dst, gray_src, pts_dst, None, **lk_params)

        good_src = []
        good_dst = []
        for i in range(len(pts_src)):
            if status[i] == 1 and status_back[i] == 1:
                # Consistency distance < 1.5 pixels
                dist = np.linalg.norm(pts_src[i] - pts_src_back[i])
                if dist < 1.5:
                    pt_d = pts_dst[i][0]
                    # Check if destination point falls inside mask_dst
                    x_int, y_int = int(round(pt_d[0])), int(round(pt_d[1]))
                    if 0 <= y_int < mask_dst.shape[0] and 0 <= x_int < mask_dst.shape[1]:
                        if mask_dst[y_int, x_int] > 0:
                            good_src.append(pts_src[i])
                            good_dst.append(pts_dst[i])

        meta["total_matches"] = len(good_src)
        if len(good_src) < self.min_inliers:
            return None, meta

        src_arr = np.array(good_src, dtype=np.float32)
        dst_arr = np.array(good_dst, dtype=np.float32)

        H, inliers = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, self.ransac_thresh)
        num_inliers = int(np.sum(inliers)) if inliers is not None else 0
        meta["inliers"] = num_inliers

        if H is not None and num_inliers >= self.min_inliers:
            return H.astype(np.float64), meta

        return None, meta

    def _affine_partial_fallback(
        self,
        frame_src: np.ndarray,
        frame_dst: np.ndarray,
        mask_src: np.ndarray,
        mask_dst: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        meta: Dict[str, Any] = {"detector": "affine_partial", "inliers": 0, "total_matches": 0}
        gray_src = cv2.cvtColor(frame_src, cv2.COLOR_BGR2GRAY)
        gray_dst = cv2.cvtColor(frame_dst, cv2.COLOR_BGR2GRAY)

        pts_src = cv2.goodFeaturesToTrack(gray_src, mask=mask_src, maxCorners=800, qualityLevel=0.02, minDistance=15)
        if pts_src is None or len(pts_src) < 10:
            return None, meta

        pts_dst, status, _ = cv2.calcOpticalFlowPyrLK(gray_src, gray_dst, pts_src, None)
        valid_idx = np.where(status.ravel() == 1)[0]
        if len(valid_idx) < 10:
            return None, meta

        s_pts = pts_src[valid_idx]
        d_pts = pts_dst[valid_idx]

        aff_mat, inliers = cv2.estimateAffinePartial2D(s_pts, d_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        num_inliers = int(np.sum(inliers)) if inliers is not None else 0
        meta["inliers"] = num_inliers
        meta["total_matches"] = len(valid_idx)

        if aff_mat is not None and num_inliers >= 8:
            H = np.eye(3, dtype=np.float64)
            H[:2, :] = aff_mat
            return H, meta

        return None, meta

    def _validate_homography(self, H: np.ndarray, w_s: int, h_s: int, w_d: int, h_d: int) -> bool:
        """
        Validates homography matrix stability:
        - Non-zero finite values
        - Determinant within reasonable broadcast scale (0.15 to 6.0)
        - Warped frame corners form a valid non-inverted convex quad
        """
        if H is None or H.shape != (3, 3) or not np.all(np.isfinite(H)):
            return False

        det = float(np.linalg.det(H))
        if det <= 0.15 or det >= 6.0:
            logger.debug("Rejecting H with anomalous determinant: %.3f", det)
            return False

        # Check conditioning
        try:
            cond = float(np.linalg.cond(H))
            if cond > 2e4:
                logger.debug("Rejecting poorly conditioned H (cond=%.1f)", cond)
                return False
        except Exception:
            return False

        # Warp 4 corners of source frame to verify convexity
        corners = np.array([[0, 0], [w_s, 0], [w_s, h_s], [0, h_s]], dtype=np.float32).reshape(-1, 1, 2)
        try:
            warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
            if not cv2.isContourConvex(warped.astype(np.int32)):
                logger.debug("Rejecting H: warped corners do not form convex polygon")
                return False
            # Check area of warped quad
            area = float(cv2.contourArea(warped.astype(np.int32)))
            orig_area = float(w_s * h_s)
            if area < 0.2 * orig_area or area > 4.0 * orig_area:
                logger.debug("Rejecting H: warped area abnormal (%.1f vs %.1f)", area, orig_area)
                return False
        except Exception as e:
            logger.debug("Corner validation exception: %s", e)
            return False

        return True

    def project_point(
        self,
        pt: Tuple[float, float],
        H: np.ndarray,
        from_pct: bool = True,
        to_pct: bool = True,
    ) -> Tuple[float, float, bool]:
        """
        Projects a 2D point (x, y) through homography H:
            [x_dst, y_dst, 1]^T ~ H * [x_src, y_src, 1]^T

        Args:
            pt: (x, y) coordinates
            H: 3x3 homography matrix
            from_pct: If True, input pt is in normalized percentage [0..100]
            to_pct: If True, returns coordinates in normalized percentage [0..100]

        Returns:
            (x_proj, y_proj, is_inside_visible_frame)
        """
        if H is None:
            return pt[0], pt[1], True

        x_in, y_in = float(pt[0]), float(pt[1])
        px = (x_in / 100.0) * self.frame_w if from_pct else x_in
        py = (y_in / 100.0) * self.frame_h if from_pct else y_in

        vec = np.array([px, py, 1.0], dtype=np.float64)
        proj = H @ vec
        if abs(proj[2]) < 1e-7:
            return x_in, y_in, False

        px_out = proj[0] / proj[2]
        py_out = proj[1] / proj[2]

        is_inside = (0.0 <= px_out <= float(self.frame_w)) and (0.0 <= py_out <= float(self.frame_h))

        if to_pct:
            rx = round((px_out / float(self.frame_w)) * 100.0, 1)
            ry = round((py_out / float(self.frame_h)) * 100.0, 1)
            return rx, ry, is_inside
        else:
            return round(px_out, 1), round(py_out, 1), is_inside

    def project_trajectory(
        self,
        points: List[Tuple[float, float]],
        H: np.ndarray,
        from_pct: bool = True,
        to_pct: bool = True,
    ) -> List[Tuple[float, float]]:
        """Projects a series of points along a trajectory through homography H."""
        res: List[Tuple[float, float]] = []
        for p in points:
            x, y, _ = self.project_point(p, H, from_pct=from_pct, to_pct=to_pct)
            res.append((x, y))
        return res

    def compensate_vector(
        self,
        vector: Dict[str, Any],
        H: np.ndarray,
        source_is_future_frame: bool = True,
    ) -> Dict[str, Any]:
        """
        Projects vector target coordinates (to_pct, landing_target_pct) from future frame
        coordinate space back to frame_freeze coordinate space using homography H.
        Preserves original values in debug metadata.
        """
        if not isinstance(vector, dict) or H is None:
            return vector

        vec = dict(vector)
        if "to_pct" in vec and isinstance(vec["to_pct"], (list, tuple)) and len(vec["to_pct"]) == 2:
            orig_to = [float(vec["to_pct"][0]), float(vec["to_pct"][1])]
            proj_x, proj_y, is_in = self.project_point(orig_to, H, from_pct=True, to_pct=True)
            vec["original_to_pct"] = orig_to
            vec["to_pct"] = [proj_x, proj_y]
            vec["target_in_visible_frame"] = is_in

        if "landing_target_pct" in vec and isinstance(vec["landing_target_pct"], (list, tuple)) and len(vec["landing_target_pct"]) == 2:
            orig_land = [float(vec["landing_target_pct"][0]), float(vec["landing_target_pct"][1])]
            proj_lx, proj_ly, _ = self.project_point(orig_land, H, from_pct=True, to_pct=True)
            vec["original_landing_target_pct"] = orig_land
            vec["landing_target_pct"] = [proj_lx, proj_ly]

        vec["homography_compensated"] = True
        return vec


class TemporalBurstAligner:
    """
    Manages multi-frame temporal alignment across all 5 burst frames:
    - frame_prev (t - 2.0s)
    - frame_freeze (t = 0.0s) [REFERENCE TARGET]
    - frame_next_1 (t + 1.5s)
    - frame_next_2 (t + 3.0s)
    - frame_next_3 (t + 4.5s)
    """

    def __init__(self, compensator: Optional[CameraPanCompensator] = None):
        self.compensator = compensator or CameraPanCompensator()
        self.homographies: Dict[str, np.ndarray] = {
            "frame_prev": np.eye(3, dtype=np.float64),
            "frame_freeze": np.eye(3, dtype=np.float64),
            "frame_next_1": np.eye(3, dtype=np.float64),
            "frame_next_2": np.eye(3, dtype=np.float64),
            "frame_next_3": np.eye(3, dtype=np.float64),
        }
        self.alignment_metadata: Dict[str, Dict[str, Any]] = {}

    def align_burst_frames(
        self,
        burst_frames: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Estimates pairwise homographies from each burst frame to frame_freeze (t = 0.0s).
        Returns dictionary of homography matrices keyed by frame name.
        """
        frame_freeze = burst_frames.get("frame_freeze")
        if frame_freeze is None:
            logger.warning("frame_freeze missing in burst_frames; alignment cannot proceed")
            return self.homographies

        mask_freeze = self.compensator.extract_pitch_mask(frame_freeze)

        for name in ["frame_prev", "frame_next_1", "frame_next_2", "frame_next_3"]:
            src = burst_frames.get(name)
            if src is not None:
                mask_src = self.compensator.extract_pitch_mask(src)
                H, meta = self.compensator.estimate_homography(
                    frame_src=src,
                    frame_dst=frame_freeze,
                    mask_src=mask_src,
                    mask_dst=mask_freeze,
                )
                self.homographies[name] = H
                self.alignment_metadata[name] = meta
                logger.debug(
                    "Aligned %s -> frame_freeze: success=%s, inliers=%d, method=%s",
                    name, meta.get("success"), meta.get("inliers", 0), meta.get("method"),
                )
            else:
                self.homographies[name] = np.eye(3, dtype=np.float64)

        return self.homographies

    def project_to_freeze(
        self,
        from_frame_name: str,
        x_pct: float,
        y_pct: float,
    ) -> Tuple[float, float, bool]:
        """Projects coordinate from specified frame into frame_freeze coordinate space."""
        H = self.homographies.get(from_frame_name, np.eye(3, dtype=np.float64))
        return self.compensator.project_point((x_pct, y_pct), H, from_pct=True, to_pct=True)
