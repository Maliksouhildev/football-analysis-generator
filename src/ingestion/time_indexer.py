"""
src/ingestion/time_indexer.py
Bi-directional Match Time Indexer.

Maintains continuous, bi-directional mathematical mapping between raw video
Presentation Timestamps (PTS) and match time (period, minute, second, stoppage).
Resolves stoppage time ambiguities (45+2, 90+4), half-time discontinuities,
piecewise linear interpolation between anchors, and stream reconnection epoch shifts.
"""

from __future__ import annotations

import bisect
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("ingestion.time_indexer")


@dataclass(frozen=True)
class MatchTime:
    """Canonical match time representation with stoppage support."""
    period: int              # 1: 1st Half, 2: 2nd Half, 3: ET1, 4: ET2, 5: Shootout
    minute: int              # Regular minute: 0..45, 45..90, etc.
    second: int              # 0..59
    stoppage_minute: int = 0 # 0 for regular; 1..N for added time (e.g. 2 for 45+2)
    stoppage_second: int = 0 # Seconds into added time
    pts: Optional[float] = None

    @property
    def display(self) -> str:
        """Formatted broadcast display string e.g. '23:45' or '45+2:15'."""
        if self.stoppage_minute > 0 or self.stoppage_second > 0:
            extra_s = self.stoppage_second % 60
            return f"{self.minute}+{self.stoppage_minute}:{extra_s:02d}"
        return f"{self.minute:02d}:{self.second:02d}"

    @property
    def display_short(self) -> str:
        """Short display string e.g. \"24'\" or \"45'+3'\"."""
        if self.stoppage_minute > 0:
            return f"{self.minute}'+{self.stoppage_minute}'"
        return f"{self.minute}'"

    @property
    def period_elapsed_seconds(self) -> float:
        """Strictly monotonic elapsed seconds from kickoff of THIS period."""
        if self.period == 1:
            base_sec = self.minute * 60 + self.second
        elif self.period == 2:
            base_sec = max(0, self.minute - 45) * 60 + self.second
        elif self.period == 3:
            base_sec = max(0, self.minute - 90) * 60 + self.second
        elif self.period == 4:
            base_sec = max(0, self.minute - 105) * 60 + self.second
        else:
            base_sec = self.minute * 60 + self.second

        if self.stoppage_second >= 60 and self.stoppage_second >= self.stoppage_minute * 60:
            extra_sec = float(self.stoppage_second)
        else:
            extra_sec = float(self.stoppage_minute * 60 + self.stoppage_second)
        return float(base_sec + extra_sec)

    @property
    def total_match_seconds(self) -> float:
        """Total nominal match seconds across the game."""
        if self.stoppage_second >= 60 and self.stoppage_second >= self.stoppage_minute * 60:
            extra_sec = float(self.stoppage_second)
        else:
            extra_sec = float(self.stoppage_minute * 60 + self.stoppage_second)
        return float(self.minute * 60 + self.second + extra_sec)

    def __getitem__(self, key: str) -> Any:
        """Dict compatibility interface for test harness and legacy callers."""
        mapping = {
            "period": self.period,
            "minute": self.minute,
            "match_minute": self.minute,
            "second": self.second,
            "match_second": self.second,
            "stoppage_minute": self.stoppage_minute,
            "stoppage_second": self.stoppage_second,
            "total_match_seconds": self.total_match_seconds,
            "display": self.display,
            "display_short": self.display_short,
            "pts": self.pts,
        }
        if key in mapping:
            return mapping[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


@dataclass
class TimeAnchor:
    """A verified synchronization point between video PTS and match time."""
    pts: float
    period: int
    minute: int
    second: int
    stoppage_minute: int = 0
    stoppage_second: int = 0
    confidence: float = 1.0
    wallclock_utc: float = field(default_factory=time.time)
    epoch_id: int = 0

    @property
    def match_time(self) -> MatchTime:
        return MatchTime(
            period=self.period,
            minute=self.minute,
            second=self.second,
            stoppage_minute=self.stoppage_minute,
            stoppage_second=self.stoppage_second,
            pts=self.pts,
        )

    @property
    def tau(self) -> float:
        """Monotonic period elapsed seconds."""
        return self.match_time.period_elapsed_seconds

    @property
    def clock_seconds(self) -> float:
        """Nominal broadcast clock seconds."""
        return float(self.match_time.total_match_seconds)


class MatchTimeIndexer:
    """
    Maintains bi-directional mathematical mapping between video Presentation Timestamps (PTS)
    and canonical MatchTime. Handles stoppage time (45+2, 90+4), half-time gap, and PTS resets.
    Splits indexing into isolated epochs (period=1 and period=2) to resolve halftime clock resets
    and uses RANSAC regression constrained to slope [0.98, 1.02] to filter replays and OCR misreads.
    """

    def __init__(self):
        self._anchors_by_period: Dict[int, List[TimeAnchor]] = {1: [], 2: [], 3: [], 4: []}
        self._inliers_by_period: Dict[int, List[TimeAnchor]] = {1: [], 2: [], 3: [], 4: []}
        self._ransac_models: Dict[int, Any] = {1: None, 2: None, 3: None, 4: None}
        self._slopes: Dict[int, float] = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
        self._intercepts: Dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        self._all_anchors: List[TimeAnchor] = []
        self._lock = threading.RLock()
        self._current_epoch = 0
        self._epoch_offset = 0.0
        self._last_raw_pts = 0.0

    @property
    def pts_to_match(self) -> List[Dict[str, Any]]:
        """List representation of anchors for test harness compatibility."""
        with self._lock:
            return [
                {
                    "pts": a.pts,
                    "period": a.period,
                    "match_minute": a.minute,
                    "match_second": a.second,
                    "stoppage_minute": a.stoppage_minute,
                    "total_match_seconds": a.match_time.total_match_seconds,
                }
                for a in self._all_anchors
            ]

    def fit_epoch_ransac(self, period: int) -> Optional[Any]:
        """
        Applies scikit-learn RANSACRegressor on (media_seconds, clock_seconds) for the isolated period epoch.
        Constrains slope to [0.98, 1.02] so replays and OCR misreads are rejected as outliers.
        """
        with self._lock:
            anchors = self._anchors_by_period.get(period, [])
            if len(anchors) < 2:
                self._inliers_by_period[period] = list(anchors)
                if anchors:
                    self._slopes[period] = 1.0
                    self._intercepts[period] = anchors[0].clock_seconds - anchors[0].pts
                return None

            media_sec = np.array([a.pts for a in anchors], dtype=np.float64).reshape(-1, 1)
            clock_sec = np.array([a.clock_seconds for a in anchors], dtype=np.float64)

            from sklearn.linear_model import LinearRegression, RANSACRegressor

            try:
                ransac = RANSACRegressor(
                    estimator=LinearRegression(),
                    min_samples=min(2, len(anchors)),
                    residual_threshold=3.0,
                    random_state=42,
                )
                ransac.fit(media_sec, clock_sec)
                slope = float(ransac.estimator_.coef_[0])
                intercept = float(ransac.estimator_.intercept_)

                if 0.98 <= slope <= 1.02:
                    inlier_mask = ransac.inlier_mask_
                    self._ransac_models[period] = ransac
                    self._slopes[period] = slope
                    self._intercepts[period] = intercept
                    self._inliers_by_period[period] = [a for i, a in enumerate(anchors) if inlier_mask[i]]
                    return ransac
                else:
                    # Constrain slope to 1.0 and fit intercept with median
                    diffs = clock_sec - media_sec.flatten()
                    med_intercept = float(np.median(diffs))
                    residuals = np.abs(diffs - med_intercept)
                    inlier_mask = residuals <= 3.0
                    self._slopes[period] = 1.0
                    self._intercepts[period] = (
                        float(np.median(diffs[inlier_mask])) if np.any(inlier_mask) else med_intercept
                    )
                    self._inliers_by_period[period] = [a for i, a in enumerate(anchors) if inlier_mask[i]]
                    return None
            except Exception as e:
                logger.warning("RANSAC fitting failed for period %d: %s", period, e)
                self._inliers_by_period[period] = list(anchors)
                return None

    def record_anchor(
        self,
        pts: float,
        match_minute: int,
        match_second: int = 0,
        period: int = 1,
        stoppage_minute: int = 0,
        stoppage_second: int = 0,
        confidence: float = 1.0,
        wallclock_utc: Optional[float] = None,
    ) -> None:
        """Records a verified clock reading anchor (harness-compatible signature)."""
        self.add_anchor(
            pts=pts,
            period=period,
            minute=match_minute,
            second=match_second,
            stoppage_minute=stoppage_minute,
            stoppage_second=stoppage_second,
            confidence=confidence,
            wallclock_utc=wallclock_utc,
        )

    def add_anchor(
        self,
        pts: float,
        period: int,
        minute: int,
        second: int = 0,
        stoppage_minute: int = 0,
        stoppage_second: int = 0,
        confidence: float = 1.0,
        wallclock_utc: Optional[float] = None,
    ) -> None:
        """
        Records a verified synchronization anchor.
        Detects stream reconnects / PTS resets and adjusts virtual epoch offsets.
        Re-fits isolated epoch RANSAC regressor.
        """
        with self._lock:
            utc_now = wallclock_utc if wallclock_utc is not None else time.time()

            # Detect stream reconnect / PTS reset
            if self._all_anchors and pts < (self._last_raw_pts - 2.0):
                self._current_epoch += 1
                self._epoch_offset = self._all_anchors[-1].pts + 2.0
                logger.info(
                    "Stream PTS discontinuity detected: epoch %d, offset %.2fs",
                    self._current_epoch,
                    self._epoch_offset,
                )

            self._last_raw_pts = pts
            effective_pts = pts + self._epoch_offset

            anchor = TimeAnchor(
                pts=effective_pts,
                period=period,
                minute=minute,
                second=second,
                stoppage_minute=stoppage_minute,
                stoppage_second=stoppage_second,
                confidence=confidence,
                wallclock_utc=utc_now,
                epoch_id=self._current_epoch,
            )

            period_list = self._anchors_by_period.setdefault(period, [])
            idx = bisect.bisect_right([a.pts for a in period_list], effective_pts)
            period_list.insert(idx, anchor)
            self._all_anchors.append(anchor)

            # Fit isolated epoch RANSAC regressor
            self.fit_epoch_ransac(period)

    def get_pts_for_match_minute(
        self,
        match_minute: int,
        stoppage_minute: int = 0,
        stoppage_second: int = 0,
        second: int = 0,
        period: Optional[int] = None,
    ) -> float:
        """
        Reverse lookup: converts match minute (+ stoppage) to stream PTS.
        Uses isolated epochs (period=1 or period=2) and RANSAC slope [0.98, 1.02]
        to resolve halftime clock reset collisions (45+3 vs 2nd-half 48:00).
        """
        with self._lock:
            # Auto-infer period if omitted
            if period is None:
                if match_minute <= 45 and stoppage_minute >= 0 and not self._is_period2_only(match_minute):
                    period = 1
                elif match_minute <= 90:
                    period = 2
                elif match_minute <= 105:
                    period = 3
                else:
                    period = 4

            target_mt = MatchTime(
                period=period,
                minute=match_minute,
                second=second,
                stoppage_minute=stoppage_minute,
                stoppage_second=stoppage_second,
            )
            target_tau = target_mt.period_elapsed_seconds

            period_anchors = self._inliers_by_period.get(period) or self._anchors_by_period.get(period, [])
            if not period_anchors:
                return self._fallback_pts_estimate(period, match_minute, second, stoppage_minute, stoppage_second)

            taus = [a.tau for a in period_anchors]
            pts_vals = [a.pts for a in period_anchors]
            slope = self._slopes.get(period, 1.0)

            # Exact match
            if target_tau in taus:
                return pts_vals[taus.index(target_tau)]

            # Before first anchor
            if target_tau < taus[0]:
                return pts_vals[0] - (taus[0] - target_tau) / slope

            # After last anchor (dead reckoning forward)
            if target_tau > taus[-1]:
                return pts_vals[-1] + (target_tau - taus[-1]) / slope

            # Between two anchors: Piecewise Linear Interpolation
            idx = bisect.bisect_right(taus, target_tau)
            t1, t2 = taus[idx - 1], taus[idx]
            p1, p2 = pts_vals[idx - 1], pts_vals[idx]

            frac = (target_tau - t1) / (t2 - t1) if t2 > t1 else 0.0
            return p1 + frac * (p2 - p1)

    def get_pts_for_match_time(
        self,
        match_minute: int,
        stoppage_minute: int = 0,
        period: int = 1,
    ) -> Optional[float]:
        """Test harness compatibility method."""
        with self._lock:
            if not self._all_anchors:
                return None
            target_sec = match_minute * 60 + stoppage_minute * 60
            best_match = None
            min_diff = float("inf")
            for entry in self._all_anchors:
                if entry.period == period:
                    diff = abs(entry.match_time.total_match_seconds - target_sec)
                    if diff < min_diff:
                        min_diff = diff
                        best_match = entry
            return best_match.pts if best_match else self.get_pts_for_match_minute(match_minute, stoppage_minute, period=period)

    def get_match_time_for_pts(self, pts: float) -> MatchTime:
        """
        Forward lookup: converts video presentation timestamp (PTS) to canonical MatchTime
        using isolated period epochs.
        """
        with self._lock:
            if not self._all_anchors:
                return MatchTime(period=1, minute=0, second=0, pts=pts)

            # Isolate epoch: period 1 vs period 2 boundary
            p1_anchors = self._inliers_by_period.get(1) or self._anchors_by_period.get(1, [])
            p2_anchors = self._inliers_by_period.get(2) or self._anchors_by_period.get(2, [])

            if p1_anchors and p2_anchors:
                p1_max = max(a.pts for a in p1_anchors)
                p2_min = min(a.pts for a in p2_anchors)
                halftime_boundary = (p1_max + p2_min) / 2.0
                period = 1 if pts < halftime_boundary else 2
            elif p2_anchors and not p1_anchors:
                period = 2
            elif p1_anchors:
                period = 1
            else:
                all_pts = [a.pts for a in self._all_anchors]
                idx = bisect.bisect_right(all_pts, pts)
                ref_anchor = self._all_anchors[0] if idx == 0 else self._all_anchors[idx - 1]
                period = ref_anchor.period

            period_anchors = self._inliers_by_period.get(period) or self._anchors_by_period.get(period, [])
            if not period_anchors:
                return self._tau_to_match_time(period, pts, pts)

            p_pts = [a.pts for a in period_anchors]
            p_taus = [a.tau for a in period_anchors]
            slope = self._slopes.get(period, 1.0)

            if pts <= p_pts[0]:
                calc_tau = max(0.0, p_taus[0] - (p_pts[0] - pts) * slope)
            elif pts >= p_pts[-1]:
                calc_tau = p_taus[-1] + (pts - p_pts[-1]) * slope
            else:
                p_idx = bisect.bisect_right(p_pts, pts)
                p1, p2 = p_pts[p_idx - 1], p_pts[p_idx]
                t1, t2 = p_taus[p_idx - 1], p_taus[p_idx]
                frac = (pts - p1) / (p2 - p1) if p2 > p1 else 0.0
                calc_tau = t1 + frac * (t2 - t1)

            return self._tau_to_match_time(period, calc_tau, pts)

    def _tau_to_match_time(self, period: int, tau: float, pts: Optional[float] = None) -> MatchTime:
        total_sec = max(0.0, tau)
        if period == 1:
            if total_sec <= 2700.0:  # 45 minutes
                minute = int(total_sec // 60)
                second = int(total_sec % 60)
                return MatchTime(period=1, minute=minute, second=second, pts=pts)
            else:
                extra_sec = total_sec - 2700.0
                extra_min = int(extra_sec // 60)
                extra_s = int(extra_sec % 60)
                return MatchTime(
                    period=1,
                    minute=45,
                    second=0,
                    stoppage_minute=extra_min,
                    stoppage_second=extra_s,
                    pts=pts,
                )
        elif period == 2:
            if total_sec <= 2700.0:
                minute = 45 + int(total_sec // 60)
                second = int(total_sec % 60)
                return MatchTime(period=2, minute=minute, second=second, pts=pts)
            else:
                extra_sec = total_sec - 2700.0
                extra_min = int(extra_sec // 60)
                extra_s = int(extra_sec % 60)
                return MatchTime(
                    period=2,
                    minute=90,
                    second=0,
                    stoppage_minute=extra_min,
                    stoppage_second=extra_s,
                    pts=pts,
                )
        elif period == 3:
            if total_sec <= 900.0:
                minute = 90 + int(total_sec // 60)
                second = int(total_sec % 60)
                return MatchTime(period=3, minute=minute, second=second, pts=pts)
            else:
                extra_sec = total_sec - 900.0
                extra_min = int(extra_sec // 60)
                extra_s = int(extra_sec % 60)
                return MatchTime(
                    period=3,
                    minute=105,
                    second=0,
                    stoppage_minute=extra_min,
                    stoppage_second=extra_s,
                    pts=pts,
                )
        elif period == 4:
            if total_sec <= 900.0:
                minute = 105 + int(total_sec // 60)
                second = int(total_sec % 60)
                return MatchTime(period=4, minute=minute, second=second, pts=pts)
            else:
                extra_sec = total_sec - 900.0
                extra_min = int(extra_sec // 60)
                extra_s = int(extra_sec % 60)
                return MatchTime(
                    period=4,
                    minute=120,
                    second=0,
                    stoppage_minute=extra_min,
                    stoppage_second=extra_s,
                    pts=pts,
                )
        return MatchTime(
            period=period,
            minute=int(total_sec // 60),
            second=int(total_sec % 60),
            pts=pts,
        )

    def _is_period2_only(self, minute: int) -> bool:
        return 45 < minute <= 90

    def _fallback_pts_estimate(
        self,
        period: int,
        minute: int,
        second: int,
        stoppage_minute: int = 0,
        stoppage_second: int = 0,
    ) -> float:
        if stoppage_second >= 60 and stoppage_second >= stoppage_minute * 60:
            extra_sec = float(stoppage_second)
        else:
            extra_sec = float(stoppage_minute * 60 + stoppage_second)
        base_sec = (minute if period == 1 else minute - 45) * 60 + second + extra_sec
        return float(base_sec)

    def save_to_json(self, json_path: str) -> None:
        """Serializes all anchors to JSON."""
        with self._lock:
            data = [asdict(a) for a in self._all_anchors]
            Path(json_path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_from_json(self, json_path: str) -> None:
        """Loads anchors from JSON."""
        with self._lock:
            raw = Path(json_path).read_text(encoding="utf-8")
            data = json.loads(raw)
            self._anchors_by_period = {1: [], 2: [], 3: [], 4: []}
            self._all_anchors = []
            for item in data:
                self.add_anchor(
                    pts=item["pts"],
                    period=item["period"],
                    minute=item["minute"],
                    second=item["second"],
                    stoppage_minute=item.get("stoppage_minute", 0),
                    stoppage_second=item.get("stoppage_second", 0),
                    confidence=item.get("confidence", 1.0),
                    wallclock_utc=item.get("wallclock_utc"),
                )


# Backward-compatible alias
TimeIndexer = MatchTimeIndexer
