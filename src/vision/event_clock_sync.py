"""
src/vision/event_clock_sync.py
Video Ingestion & Clock Synchronization Engine (Fixing Flaw A: Event Feed Illusion & Clock Drift).

Eliminates reliance on coarse text commentary feeds and addresses 1s-3s video clock
drift by performing cross-correlation density matching between detected pitch ball
velocity spikes (passes/shots) and structured event feed timestamps.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("vision.event_clock_sync")


class EventClockSynchronizer:
    """
    Aligns broadcast video clock with match event feed timestamps.
    Computes cross-correlation histogram between detected on-pitch ball velocity
    spikes (kicks, shots, passes) and structured feed event timestamps (e.g. StatsBomb, WhoScored).
    """

    def __init__(
        self,
        search_window_sec: float = 4.0,
        bin_width_sec: float = 0.2,
        min_density_threshold: int = 1,
    ):
        """
        Args:
            search_window_sec: Half-width of search window in seconds (e.g. [-4s, +4s]).
            bin_width_sec: Resolution of the cross-correlation histogram bins.
            min_density_threshold: Minimum delta count required in the peak bin to accept calibration.
        """
        self.search_window = float(abs(search_window_sec))
        self.bin_width = float(max(0.01, bin_width_sec))
        self.min_density_threshold = max(1, min_density_threshold)
        self.num_bins = max(5, int(round((2.0 * self.search_window) / self.bin_width)))

    def estimate_clock_offset(
        self,
        detected_kick_timestamps: Sequence[float],
        feed_event_timestamps: Sequence[float],
    ) -> float:
        """
        Estimates clock offset delta = (t_feed - t_video) in seconds.
        A positive delta means the event feed timestamp is ahead of the video clock:
            t_feed = t_video + offset
            t_video = t_feed - offset

        Args:
            detected_kick_timestamps: List of detected kick/pass timestamps from video (seconds).
            feed_event_timestamps: List of pass/shot timestamps from event feed (seconds).

        Returns:
            Estimated offset in seconds. Returns 0.0 if insufficient data or no correlation peak.
        """
        kicks = [float(t) for t in detected_kick_timestamps if t is not None and not math.isnan(t)]
        feeds = [float(t) for t in feed_event_timestamps if t is not None and not math.isnan(t)]

        if not kicks or not feeds:
            logger.debug("Empty kick or feed timestamps; returning default 0.0s offset.")
            return 0.0

        deltas = [
            t_feed - t_kick
            for t_feed in feeds
            for t_kick in kicks
            if abs(t_feed - t_kick) <= self.search_window
        ]

        if not deltas:
            logger.debug("No deltas found within search window ±%.2fs.", self.search_window)
            return 0.0

        hist, bin_edges = np.histogram(
            deltas,
            bins=self.num_bins,
            range=(-self.search_window, self.search_window),
        )

        best_bin = int(np.argmax(hist))
        peak_count = int(hist[best_bin])

        if peak_count < self.min_density_threshold:
            logger.debug(
                "Peak density (%d) below threshold (%d); fallback to 0.0s.",
                peak_count,
                self.min_density_threshold,
            )
            return 0.0

        # Refined peak: weighted mean of deltas falling within the peak bin
        bin_min = bin_edges[best_bin]
        bin_max = bin_edges[best_bin + 1]
        in_bin = [d for d in deltas if bin_min <= d <= bin_max]

        if in_bin:
            refined_offset = float(np.median(in_bin))
        else:
            refined_offset = float((bin_min + bin_max) / 2.0)

        logger.info(
            "Clock synchronization locked: offset=%.3fs (peak_count=%d deltas)",
            refined_offset,
            peak_count,
        )
        return float(round(refined_offset, 3))

    @staticmethod
    def detect_velocity_spikes(
        ball_trajectory: Sequence[Tuple[float, float, float]],
        velocity_threshold_mps: float = 12.0,
        min_dt_sec: float = 0.02,
        cooldown_sec: float = 0.8,
    ) -> List[float]:
        """
        Extracts candidate kick/pass timestamps from ball trajectory tracking.

        Args:
            ball_trajectory: Sequence of (x_meter, y_meter, timestamp_sec).
            velocity_threshold_mps: Instantaneous speed threshold in meters/sec indicative of kick/pass.
            min_dt_sec: Minimum delta time between frames to avoid division by zero.
            cooldown_sec: Minimum duration between consecutive kick detections.

        Returns:
            Sorted list of kick event timestamps in seconds.
        """
        if len(ball_trajectory) < 2:
            return []

        kick_times: List[float] = []
        last_kick_t = -999.0

        for i in range(1, len(ball_trajectory)):
            x0, y0, t0 = ball_trajectory[i - 1]
            x1, y1, t1 = ball_trajectory[i]
            dt = t1 - t0
            if dt < min_dt_sec:
                continue

            dist = math.hypot(x1 - x0, y1 - y0)
            velocity = dist / dt

            if velocity >= velocity_threshold_mps and (t1 - last_kick_t) >= cooldown_sec:
                kick_times.append(float(round(t1, 3)))
                last_kick_t = t1

        return kick_times

    def align_feed_events_to_video(
        self,
        feed_events: Sequence[Dict[str, Any]],
        clock_offset_sec: float,
    ) -> List[Dict[str, Any]]:
        """
        Applies calibrated clock offset to feed events so their timestamps match video clock.
        t_video = t_feed - clock_offset_sec
        """
        aligned = []
        for ev in feed_events:
            ev_copy = dict(ev)
            raw_sec = ev_copy.get("timestamp_sec")
            if raw_sec is None:
                m = ev_copy.get("minute", 0)
                s = ev_copy.get("second", 0)
                raw_sec = m * 60.0 + s

            ev_copy["video_timestamp_sec"] = max(0.0, float(raw_sec - clock_offset_sec))
            aligned.append(ev_copy)
        return aligned
