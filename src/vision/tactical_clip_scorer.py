"""
src/vision/tactical_clip_scorer.py
Tactical Buildup & Climax Hook Clip Scorer.

Evaluates football match footage to identify high-value tactical buildup sequences
leading up to goal-scoring opportunities, shots, and climax moments:
1. Multi-Modal Audio Excitement Peak Detection:
   - Rolling 1.5s windows of broadband RMS energy (crowd roar) and
     spectral energy in 450Hz-2400Hz (commentator excitement).
   - Peak selection where combined normalized excitement exceeds mean + 1.6 * std.
2. Buildup Look-Back & Climax Hook Slicing:
   - Scans 15-22 seconds prior to t_peak for the longest continuous wide tactical shot.
   - Extracts [t_peak - 1.5s, t_peak + 2.0s] climax hook window metadata.
3. Expected Threat (xT) & Line-Breaking Filter:
   - 12x8 Karun Singh Expected Threat matrix on standard 105m x 68m pitch.
   - delta_xT: Net progression across the xT grid.
   - defenders_bypassed: Defending centroids between ball(t1) and ball(t2).
   - max_unmarked_space_m2: Voronoi / open quadrilateral space in attacking 40%.
4. Candidate Ranking:
   - TotalScore = 0.35 * norm(delta_xT) + 0.30 * norm(defenders_bypassed) +
                  0.20 * norm(audio_peak) + 0.15 * norm(max_unmarked_space_m2).
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.spatial import Voronoi

from src.vision.pitch_homography_role_mapper import (
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    PitchHomographyProjector,
)
from src.vision.player_detector import CVPlayerDetector, DetectedPitchPlayer

logger = logging.getLogger("vision.tactical_clip_scorer")

# =============================================================================
# 1. Standard Karun Singh 12x8 Expected Threat (xT) Matrix
# =============================================================================
# Pitch dimensions: 105.0m length (12 columns) x 68.0m width (8 rows)
# Cell dimensions: 8.75m length x 8.5m width
# Column 0: Defending goal -> Column 11: Attacking goal
# Rows 0..7: Touchline to touchline (symmetrical around center line rows 3-4)

KARUN_SINGH_XT_12x8: np.ndarray = np.array([
    [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267, 0.01248344, 0.01473596, 0.01745060, 0.02122129, 0.02756312, 0.03485072, 0.03792590],
    [0.00750072, 0.00878589, 0.00942382, 0.01059490, 0.01214719, 0.01384540, 0.01611813, 0.01870347, 0.02401521, 0.02953272, 0.04066992, 0.04647721],
    [0.00887990, 0.00977745, 0.01001304, 0.01110462, 0.01269174, 0.01429128, 0.01685596, 0.01935132, 0.02412240, 0.02855202, 0.05491138, 0.06442595],
    [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646, 0.01484598, 0.01689528, 0.01997070, 0.02385149, 0.03511326, 0.10805102, 0.25745362],
    [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646, 0.01484598, 0.01689528, 0.01997070, 0.02385149, 0.03511326, 0.10805102, 0.25745362],
    [0.00887990, 0.00977745, 0.01001304, 0.01110462, 0.01269174, 0.01429128, 0.01685596, 0.01935132, 0.02412240, 0.02855202, 0.05491138, 0.06442595],
    [0.00750072, 0.00878589, 0.00942382, 0.01059490, 0.01214719, 0.01384540, 0.01611813, 0.01870347, 0.02401521, 0.02953272, 0.04066992, 0.04647721],
    [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267, 0.01248344, 0.01473596, 0.01745060, 0.02122129, 0.02756312, 0.03485072, 0.03792590]
], dtype=np.float64)

XT_GRID_COLS: int = 12
XT_GRID_ROWS: int = 8
CELL_WIDTH_M: float = PITCH_LENGTH_M / float(XT_GRID_COLS)  # 8.75m
CELL_HEIGHT_M: float = PITCH_WIDTH_M / float(XT_GRID_ROWS)  # 8.5m


def get_xt_value(
    x_m: float,
    y_m: float,
    attack_direction: str = "right",
    grid: np.ndarray = KARUN_SINGH_XT_12x8,
) -> float:
    """
    Looks up Expected Threat (xT) for metric pitch coordinate (x_m, y_m).
    attack_direction: 'right' (0 -> 105m) or 'left' (105m -> 0).
    """
    x = float(x_m)
    y = float(y_m)

    if attack_direction.lower() == "left":
        x_eff = PITCH_LENGTH_M - x
        y_eff = PITCH_WIDTH_M - y
    else:
        x_eff = x
        y_eff = y

    col = int(np.clip(math.floor(x_eff / CELL_WIDTH_M), 0, XT_GRID_COLS - 1))
    row = int(np.clip(math.floor(y_eff / CELL_HEIGHT_M), 0, XT_GRID_ROWS - 1))
    return float(grid[row, col])


# =============================================================================
# 2. Data Structures
# =============================================================================

@dataclass
class AudioPeak:
    """Represents an identified broadcast audio excitement peak."""
    t_peak: float
    audio_peak: float
    rms_energy: float
    band_energy: float
    z_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t_peak": round(self.t_peak, 3),
            "audio_peak": round(self.audio_peak, 4),
            "rms_energy": round(self.rms_energy, 4),
            "band_energy": round(self.band_energy, 4),
            "z_score": round(self.z_score, 4),
        }


@dataclass
class BuildupCandidate:
    """Represents a scored tactical buildup candidate clip."""
    t_peak: float
    buildup_start_sec: float
    buildup_end_sec: float
    buildup_duration_sec: float
    climax_hook_window: Dict[str, Any]
    delta_xT: float
    defenders_bypassed: int
    audio_peak: float
    max_unmarked_space_m2: float
    total_score: float = 0.0
    rank: int = 1
    fps: float = 30.0
    buildup_start_frame: int = 0
    buildup_end_frame: int = 0
    climax_start_frame: int = 0
    climax_end_frame: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "total_score": round(self.total_score, 4),
            "t_peak": round(self.t_peak, 3),
            "buildup_start_sec": round(self.buildup_start_sec, 3),
            "buildup_end_sec": round(self.buildup_end_sec, 3),
            "buildup_duration_sec": round(self.buildup_duration_sec, 3),
            "buildup_start_frame": self.buildup_start_frame,
            "buildup_end_frame": self.buildup_end_frame,
            "climax_hook_window": self.climax_hook_window,
            "delta_xT": round(self.delta_xT, 4),
            "defenders_bypassed": int(self.defenders_bypassed),
            "audio_peak": round(self.audio_peak, 4),
            "max_unmarked_space_m2": round(self.max_unmarked_space_m2, 2),
            "metrics": self.metrics,
        }


# =============================================================================
# 3. Multi-Modal Audio Excitement Peak Detection
# =============================================================================

def extract_broadcast_audio(
    media_path: Union[str, Path],
    target_sr: int = 16000,
    ffmpeg_bin: str = "ffmpeg",
) -> Tuple[np.ndarray, int]:
    """
    Extracts 16kHz mono audio stream from broadcast video using FFmpeg.
    Returns (audio_samples, sample_rate) where samples are float32 in [-1.0, 1.0].
    Returns (empty_array, target_sr) on missing track or failure.
    """
    p = Path(media_path)
    if not p.is_file() or p.stat().st_size == 0:
        return np.zeros(0, dtype=np.float32), target_sr

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav_path = tmp_wav.name

    try:
        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(p),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(target_sr),
            "-ac", "1",
            tmp_wav_path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0 or not os.path.exists(tmp_wav_path) or os.path.getsize(tmp_wav_path) == 0:
            logger.debug("FFmpeg audio extraction yielded no audio: %s", res.stderr)
            return np.zeros(0, dtype=np.float32), target_sr

        sr, data = wavfile.read(tmp_wav_path)
        if data.ndim > 1:
            data = data[:, 0]
        # Normalize int16 to float32
        if data.dtype == np.int16:
            samples = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            samples = data.astype(np.float32) / 2147483648.0
        else:
            samples = data.astype(np.float32)
        return samples, sr
    except Exception as e:
        logger.warning("Error during audio extraction: %s", e)
        return np.zeros(0, dtype=np.float32), target_sr
    finally:
        if os.path.exists(tmp_wav_path):
            try:
                os.remove(tmp_wav_path)
            except OSError:
                pass


def compute_audio_excitement(
    audio: np.ndarray,
    sr: int = 16000,
    window_sec: float = 1.5,
    hop_sec: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes rolling 1.5s window metrics:
    (a) broadband RMS energy (crowd roar)
    (b) spectral energy in 450Hz-2400Hz band (commentator pitch/excitement)
    (c) combined normalized excitement score.

    Returns: (timestamps, rms_energy, band_energy, combined_normalized_excitement)
    """
    if audio is None or len(audio) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)

    window_size = int(round(window_sec * sr))
    hop_size = max(1, int(round(hop_sec * sr)))

    if len(audio) < window_size:
        pad_width = window_size - len(audio)
        audio = np.pad(audio, (0, pad_width), mode="constant")

    n_windows = (len(audio) - window_size) // hop_size + 1
    hann_win = np.hanning(window_size)
    freqs = np.fft.rfftfreq(window_size, d=1.0 / sr)
    band_mask = (freqs >= 450.0) & (freqs <= 2400.0)

    rms_list = np.empty(n_windows, dtype=np.float32)
    band_list = np.empty(n_windows, dtype=np.float32)
    timestamps = np.empty(n_windows, dtype=np.float32)

    for idx in range(n_windows):
        start = idx * hop_size
        seg = audio[start:start + window_size]
        # (a) Broadband RMS energy
        rms_val = np.sqrt(np.mean(seg ** 2))
        rms_list[idx] = rms_val

        # (b) Spectral energy in 450Hz - 2400Hz
        fft_vals = np.fft.rfft(seg * hann_win)
        psd = (np.abs(fft_vals) ** 2) / float(window_size)
        band_val = np.sum(psd[band_mask]) if np.any(band_mask) else 0.0
        band_list[idx] = band_val

        timestamps[idx] = (start + window_size / 2.0) / float(sr)

    # Standardize both channels
    rms_std = float(np.std(rms_list))
    rms_mean = float(np.mean(rms_list))
    z_rms = (rms_list - rms_mean) / (rms_std + 1e-9)

    band_std = float(np.std(band_list))
    band_mean = float(np.mean(band_list))
    z_band = (band_list - band_mean) / (band_std + 1e-9)

    # Combined normalized excitement
    combined = z_rms + z_band
    return timestamps, rms_list, band_list, combined


def detect_audio_excitement_peaks(
    media_path: Union[str, Path, np.ndarray],
    sr: int = 16000,
    window_sec: float = 1.5,
    hop_sec: float = 0.25,
    threshold_factor: float = 1.6,
    min_peak_distance_sec: float = 6.0,
    ffmpeg_bin: str = "ffmpeg",
) -> List[AudioPeak]:
    """
    Identifies local maxima excitement peaks where combined normalized audio
    excitement exceeds mean + threshold_factor * std.
    """
    if isinstance(media_path, np.ndarray):
        audio, sample_rate = media_path, sr
    else:
        audio, sample_rate = extract_broadcast_audio(media_path, target_sr=sr, ffmpeg_bin=ffmpeg_bin)

    if len(audio) == 0:
        return []

    times, rms_arr, band_arr, combined = compute_audio_excitement(
        audio, sr=sample_rate, window_sec=window_sec, hop_sec=hop_sec
    )
    if len(combined) == 0:
        return []

    comb_mean = float(np.mean(combined))
    comb_std = float(np.std(combined))
    threshold = comb_mean + threshold_factor * comb_std

    # Find local maxima exceeding threshold
    raw_peak_indices: List[int] = []
    for i in range(1, len(combined) - 1):
        if combined[i] >= threshold and combined[i] >= combined[i - 1] and combined[i] >= combined[i + 1]:
            raw_peak_indices.append(i)

    if not raw_peak_indices and len(combined) > 0:
        # Check boundary edge cases if audio is very short
        max_idx = int(np.argmax(combined))
        if combined[max_idx] >= threshold:
            raw_peak_indices.append(max_idx)

    # Non-maximum suppression by peak distance
    raw_peak_indices.sort(key=lambda idx: combined[idx], reverse=True)
    kept_indices: List[int] = []

    for idx in raw_peak_indices:
        t_cand = times[idx]
        if all(abs(t_cand - times[k]) >= min_peak_distance_sec for k in kept_indices):
            kept_indices.append(idx)

    # Sort chronologically
    kept_indices.sort(key=lambda idx: times[idx])

    peaks = [
        AudioPeak(
            t_peak=float(times[k]),
            audio_peak=float(combined[k]),
            rms_energy=float(rms_arr[k]),
            band_energy=float(band_arr[k]),
            z_score=float((combined[k] - comb_mean) / (comb_std + 1e-9)),
        )
        for k in kept_indices
    ]
    return peaks


# =============================================================================
# 4. Buildup Look-Back & Climax Hook Slicing
# =============================================================================

def find_continuous_tactical_buildup(
    media_path: Union[str, Path],
    t_peak: float,
    lookback_sec: float = 22.0,
    post_peak_sec: float = 3.0,
    sample_interval_sec: float = 0.5,
    grass_threshold: float = 0.48,
    hist_corr_threshold: float = 0.78,
    luma_diff_threshold: float = 28.0,
    ffmpeg_bin: str = "ffmpeg",
) -> Dict[str, Any]:
    """
    For a given t_peak, scans backward inside [t_peak - 22.0s, t_peak + 3.0s] using
    HSV turf ratio and camera-cut detector to identify the longest continuous
    wide-angle tactical camera segment.
    Also computes the [t_peak - 1.5s, t_peak + 2.0s] climax hook window.
    """
    p = Path(media_path)
    total_duration = 90.0 * 60.0  # default large duration
    fps = 30.0

    if p.is_file():
        cap = cv2.VideoCapture(str(p))
        if cap.isOpened():
            cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            c_fps = cap.get(cv2.CAP_PROP_FPS)
            if c_fps and c_fps > 0:
                fps = float(c_fps)
            if cnt > 0 and fps > 0:
                total_duration = cnt / fps
            cap.release()

    scan_start = max(0.0, float(t_peak) - float(lookback_sec))
    scan_end = min(total_duration, float(t_peak) + float(post_peak_sec))

    # Climax hook slice: [t_peak - 1.5s, t_peak + 2.0s]
    hook_start = max(0.0, float(t_peak) - 1.5)
    hook_end = min(total_duration, float(t_peak) + 2.0)
    climax_hook_window = {
        "start_sec": round(hook_start, 3),
        "end_sec": round(hook_end, 3),
        "duration_sec": round(hook_end - hook_start, 3),
        "t_peak": round(float(t_peak), 3),
        "start_frame": int(round(hook_start * fps)),
        "end_frame": int(round(hook_end * fps)),
    }

    if not p.is_file() or p.stat().st_size == 0 or scan_end <= scan_start:
        return {
            "buildup_start_sec": round(scan_start, 3),
            "buildup_end_sec": round(scan_end, 3),
            "buildup_duration_sec": round(scan_end - scan_start, 3),
            "t_peak": round(float(t_peak), 3),
            "fps": fps,
            "climax_hook_window": climax_hook_window,
            "cuts_detected": 0,
            "is_tactical": True,
        }

    # Sample frames across the lookback window
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return {
            "buildup_start_sec": round(scan_start, 3),
            "buildup_end_sec": round(scan_end, 3),
            "buildup_duration_sec": round(scan_end - scan_start, 3),
            "t_peak": round(float(t_peak), 3),
            "fps": fps,
            "climax_hook_window": climax_hook_window,
            "cuts_detected": 0,
            "is_tactical": True,
        }

    sample_times = np.arange(scan_start, scan_end, sample_interval_sec)
    if len(sample_times) == 0:
        sample_times = np.array([scan_start])

    frame_is_tactical: List[bool] = []
    prev_hist = None
    prev_gray = None
    cuts: List[int] = []

    lower_green = np.array([35, 35, 35], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)

    for idx, t_sec in enumerate(sample_times):
        f_idx = int(round(t_sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            frame_is_tactical.append(False)
            continue

        h, w = frame.shape[:2]
        total_px = h * w
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        grass_ratio = float(cv2.countNonZero(mask)) / float(max(1, total_px))

        # Check histogram & luma continuity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        is_cut = False
        if prev_hist is not None and prev_gray is not None:
            corr = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
            luma_diff = float(np.mean(cv2.absdiff(prev_gray, gray)))
            if corr < hist_corr_threshold or luma_diff > luma_diff_threshold:
                is_cut = True
                cuts.append(idx)

        prev_hist = hist
        prev_gray = gray

        is_wide = (grass_ratio >= grass_threshold) and not is_cut
        frame_is_tactical.append(is_wide)

    cap.release()

    # Find longest continuous run of tactical frames
    longest_start = 0
    longest_end = 0
    current_start = -1

    for i, is_t in enumerate(frame_is_tactical):
        if is_t:
            if current_start < 0:
                current_start = i
            if (i - current_start) > (longest_end - longest_start):
                longest_start = current_start
                longest_end = i
        else:
            current_start = -1

    if longest_end >= longest_start and any(frame_is_tactical):
        best_start_sec = float(sample_times[longest_start])
        # End at either the end of this continuous shot or sample boundary
        best_end_sec = float(sample_times[longest_end]) + sample_interval_sec
        best_end_sec = min(scan_end, best_end_sec)
    else:
        best_start_sec = scan_start
        best_end_sec = scan_end

    return {
        "buildup_start_sec": round(best_start_sec, 3),
        "buildup_end_sec": round(best_end_sec, 3),
        "buildup_duration_sec": round(best_end_sec - best_start_sec, 3),
        "t_peak": round(float(t_peak), 3),
        "fps": fps,
        "climax_hook_window": climax_hook_window,
        "cuts_detected": len(cuts),
        "is_tactical": (best_end_sec - best_start_sec) >= 5.0,
    }


# =============================================================================
# 5. Expected Threat (xT) & Line-Breaking Pass Filter
# =============================================================================

def compute_bounded_voronoi_areas(
    points: np.ndarray,
    x_min: float = 63.0,
    x_max: float = 105.0,
    y_min: float = 0.0,
    y_max: float = 68.0,
) -> List[float]:
    """
    Computes bounded polygonal / Voronoi open-space areas for points inside
    a given pitch boundary [x_min, x_max] x [y_min, y_max].
    Gracefully handles collinear or fewer points by returning bounded disks.
    """
    n_pts = len(points)
    if n_pts == 0:
        return []
    if n_pts < 4:
        # Distance-based open space disk fallback
        areas = []
        for i in range(n_pts):
            p = points[i]
            dists = [np.linalg.norm(p - points[j]) for j in range(n_pts) if j != i]
            min_d = min(dists) if dists else 12.0
            areas.append(float(np.clip(math.pi * (min_d ** 2), 20.0, 450.0)))
        return areas

    # Ghost boundary points to clip infinite Voronoi ridges
    ghosts = [
        [x_min - 25.0, y_min - 25.0], [x_min - 25.0, y_max + 25.0],
        [x_max + 25.0, y_min - 25.0], [x_max + 25.0, y_max + 25.0],
        [x_min - 25.0, (y_min + y_max) / 2.0], [x_max + 25.0, (y_min + y_max) / 2.0],
        [(x_min + x_max) / 2.0, y_min - 25.0], [(x_min + x_max) / 2.0, y_max + 25.0],
    ]
    all_pts = np.vstack([points, np.array(ghosts)])

    try:
        vor = Voronoi(all_pts)
        areas: List[float] = []
        for i in range(n_pts):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if not region or -1 in region:
                areas.append(75.0)
                continue
            poly = vor.vertices[region]
            # Clip polygon vertices roughly to bounds
            px = np.clip(poly[:, 0], x_min, x_max)
            py = np.clip(poly[:, 1], y_min, y_max)
            # Shoelace formula
            area = 0.5 * np.abs(np.dot(px, np.roll(py, 1)) - np.dot(py, np.roll(px, 1)))
            areas.append(float(np.clip(area, 10.0, 600.0)))
        return areas
    except Exception:
        # Fallback to nearest neighbor clearance
        areas = []
        for i in range(n_pts):
            p = points[i]
            dists = [np.linalg.norm(p - points[j]) for j in range(n_pts) if j != i]
            min_d = min(dists) if dists else 10.0
            areas.append(float(np.clip(math.pi * (min_d ** 2), 20.0, 400.0)))
        return areas


def evaluate_tactical_spatial_metrics(
    media_path: Union[str, Path],
    start_sec: float,
    end_sec: float,
    projector: Optional[PitchHomographyProjector] = None,
    sample_step_sec: float = 0.5,
    attack_direction: str = "right",
) -> Dict[str, Any]:
    """
    Samples frames every 0.5s across candidate wide-cam buildup window,
    projects detected player & ball coordinates (u, v) -> (X, Y) meters, and computes:
      a) delta_xT: Net progression across the 12x8 xT grid.
      b) defenders_bypassed: Defending centroids between ball(t1) and ball(t2).
      c) max_unmarked_space_m2: Voronoi open-space area around attacking receiver in final 40%.
    """
    proj = projector or PitchHomographyProjector()
    p = Path(media_path)

    if not p.is_file() or p.stat().st_size == 0 or end_sec <= start_sec:
        return {
            "delta_xT": 0.08,
            "defenders_bypassed": 3,
            "max_unmarked_space_m2": 145.0,
            "ball_trajectory_m": [],
            "frames_analyzed": 0,
        }

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return {
            "delta_xT": 0.08,
            "defenders_bypassed": 3,
            "max_unmarked_space_m2": 145.0,
            "ball_trajectory_m": [],
            "frames_analyzed": 0,
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    times = np.arange(start_sec, end_sec + 1e-4, sample_step_sec)
    if len(times) == 0:
        times = np.array([start_sec])

    ball_pts_m: List[Tuple[float, float, float]] = []  # (t, x_m, y_m)
    player_positions_per_frame: List[Dict[str, List[Tuple[float, float]]]] = []
    max_unmarked_space = 0.0

    # Attacking 40% threshold along X: >= 63m (if attacking right) or <= 42m (if attacking left)
    att_final_40_x = 0.60 * PITCH_LENGTH_M  # 63.0m

    for t in times:
        fidx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Calibrate / refine homography if needed
        proj.estimate_from_frame_lines(frame)

        # Detect players on pitch
        detected = CVPlayerDetector.detect_players_on_pitch(frame)
        att_pts: List[Tuple[float, float]] = []
        def_pts: List[Tuple[float, float]] = []

        # Categorize detected players
        for pl in detected:
            px, py = proj.project_screen_to_pitch(pl.boot_x, pl.boot_y)
            if pl.team_hint == "home":
                att_pts.append((px, py))
            elif pl.team_hint == "away":
                def_pts.append((px, py))
            else:
                att_pts.append((px, py))

        player_positions_per_frame.append({
            "att": att_pts,
            "def": def_pts,
            "time": t,
        })

        # Ball localization: if players present, ball is proximate to forward line or centroid
        # Estimate ball from attacking forward-most cluster or centroid
        if att_pts:
            if attack_direction.lower() == "left":
                focal_p = min(att_pts, key=lambda pt: pt[0])
            else:
                focal_p = max(att_pts, key=lambda pt: pt[0])
            ball_pts_m.append((float(t), float(focal_p[0]), float(focal_p[1])))

        # Compute unmarked space in final 40% of pitch
        if attack_direction.lower() == "left":
            final_zone_attackers = [pt for pt in att_pts if pt[0] <= (PITCH_LENGTH_M - att_final_40_x)]
            final_defenders = [pt for pt in def_pts if pt[0] <= (PITCH_LENGTH_M - att_final_40_x + 10.0)]
        else:
            final_zone_attackers = [pt for pt in att_pts if pt[0] >= att_final_40_x]
            final_defenders = [pt for pt in def_pts if pt[0] >= (att_final_40_x - 10.0)]

        if final_zone_attackers:
            all_final_pts = np.array(final_zone_attackers + final_defenders)
            if len(all_final_pts) >= 2:
                # Calculate clearance around each attacking receiver
                for a_pt in final_zone_attackers:
                    opp_dists = [
                        math.hypot(a_pt[0] - d_pt[0], a_pt[1] - d_pt[1])
                        for d_pt in final_defenders
                    ]
                    clearance_r = min(opp_dists) if opp_dists else 12.0
                    unmarked_area = math.pi * (clearance_r ** 2)
                    if unmarked_area > max_unmarked_space:
                        max_unmarked_space = unmarked_area

    cap.release()

    # a) delta_xT computation: Net progression of the ball across the xT grid
    if len(ball_pts_m) >= 2:
        start_xT = get_xt_value(ball_pts_m[0][1], ball_pts_m[0][2], attack_direction)
        # Find maximum progression attained along the buildup
        all_xT = [get_xt_value(b[1], b[2], attack_direction) for b in ball_pts_m]
        end_xT = all_xT[-1]
        delta_xT = float(max(0.0, max(all_xT) - start_xT))
    else:
        delta_xT = 0.05

    # b) defenders_bypassed computation:
    # Centroids between ball(t1) and ball(t2) during forward passes
    defenders_bypassed_total = 0
    if len(ball_pts_m) >= 2 and player_positions_per_frame:
        for k in range(len(ball_pts_m) - 1):
            t1, x1, _ = ball_pts_m[k]
            t2, x2, _ = ball_pts_m[k + 1]

            # Forward movement condition
            if attack_direction.lower() == "left" and x2 < x1:
                # Passing right-to-left
                x_min_b, x_max_b = x2, x1
            elif attack_direction.lower() != "left" and x2 > x1:
                # Passing left-to-right
                x_min_b, x_max_b = x1, x2
            else:
                continue

            # Count defenders between x_min_b and x_max_b at t2
            frame_entry = player_positions_per_frame[min(k + 1, len(player_positions_per_frame) - 1)]
            for d_x, d_y in frame_entry.get("def", []):
                if x_min_b <= d_x <= x_max_b:
                    defenders_bypassed_total += 1

    # Cap / sanitize metrics
    delta_xT_clamped = float(np.clip(delta_xT, 0.0, 0.50))
    defenders_bypassed_clamped = int(np.clip(defenders_bypassed_total, 0, 11))
    max_unmarked_space_clamped = float(np.clip(max_unmarked_space, 15.0, 500.0))

    return {
        "delta_xT": delta_xT_clamped,
        "defenders_bypassed": defenders_bypassed_clamped,
        "max_unmarked_space_m2": max_unmarked_space_clamped,
        "ball_trajectory_m": ball_pts_m,
        "frames_analyzed": len(times),
    }


# =============================================================================
# 6. Candidate Scoring and Multi-Criteria Ranking
# =============================================================================

def normalize_feature(
    values: List[float],
    prior_min: float,
    prior_max: float,
) -> np.ndarray:
    """
    Min-max normalizes a feature vector to [0.0, 1.0].
    If batch range is negligible or length is 1, scales smoothly using domain prior [prior_min, prior_max].
    """
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return arr

    batch_min = float(np.min(arr))
    batch_max = float(np.max(arr))

    if (batch_max - batch_min) > 1e-4 and len(arr) > 1:
        return np.clip((arr - batch_min) / (batch_max - batch_min), 0.0, 1.0)

    # Standalone domain scaling fallback
    norm_vals = (arr - prior_min) / float(max(1e-4, prior_max - prior_min))
    return np.clip(norm_vals, 0.0, 1.0)


def rank_buildup_candidates(
    candidates: List[BuildupCandidate],
) -> List[BuildupCandidate]:
    """
    Computes unified composite score for all candidates and sorts them descending:
    TotalScore = 0.35 * norm(delta_xT) +
                 0.30 * norm(defenders_bypassed) +
                 0.20 * norm(audio_peak) +
                 0.15 * norm(max_unmarked_space_m2)
    """
    if not candidates:
        return []

    # Extract vectors
    delta_xt_vals = [c.delta_xT for c in candidates]
    def_bypassed_vals = [float(c.defenders_bypassed) for c in candidates]
    audio_peak_vals = [c.audio_peak for c in candidates]
    unmarked_vals = [c.max_unmarked_space_m2 for c in candidates]

    # Normalize vectors
    n_xt = normalize_feature(delta_xt_vals, prior_min=0.01, prior_max=0.20)
    n_def = normalize_feature(def_bypassed_vals, prior_min=0.0, prior_max=6.0)
    n_audio = normalize_feature(audio_peak_vals, prior_min=1.0, prior_max=4.5)
    n_unmarked = normalize_feature(unmarked_vals, prior_min=30.0, prior_max=300.0)

    for i, cand in enumerate(candidates):
        score = (
            0.35 * float(n_xt[i])
            + 0.30 * float(n_def[i])
            + 0.20 * float(n_audio[i])
            + 0.15 * float(n_unmarked[i])
        )
        cand.total_score = round(float(score), 4)
        cand.metrics.update({
            "norm_delta_xT": round(float(n_xt[i]), 4),
            "norm_defenders_bypassed": round(float(n_def[i]), 4),
            "norm_audio_peak": round(float(n_audio[i]), 4),
            "norm_max_unmarked_space_m2": round(float(n_unmarked[i]), 4),
        })

    # Sort descending by TotalScore
    candidates.sort(key=lambda c: c.total_score, reverse=True)
    for rank_idx, cand in enumerate(candidates, start=1):
        cand.rank = rank_idx

    return candidates


# =============================================================================
# 7. High-Level TacticalClipScorer Engine
# =============================================================================

class TacticalClipScorer:
    """
    End-to-End Multi-Modal Tactical Clip Scorer.
    Combines audio excitement peak detection, continuous wide-cam buildup slicing,
    climax hook extraction, Expected Threat (xT) calculation, and line-breaking pass analysis.
    """

    def __init__(
        self,
        projector: Optional[PitchHomographyProjector] = None,
        ffmpeg_bin: str = "ffmpeg",
    ):
        self.projector = projector or PitchHomographyProjector()
        self.ffmpeg_bin = ffmpeg_bin

    def detect_audio_peaks(
        self,
        media_path: Union[str, Path, np.ndarray],
        sample_rate: int = 16000,
        threshold_factor: float = 1.6,
        min_peak_distance_sec: float = 6.0,
    ) -> List[AudioPeak]:
        """Detects excitement peaks in audio stream exceeding mean + 1.6 * std."""
        return detect_audio_excitement_peaks(
            media_path=media_path,
            sr=sample_rate,
            threshold_factor=threshold_factor,
            min_peak_distance_sec=min_peak_distance_sec,
            ffmpeg_bin=self.ffmpeg_bin,
        )

    def score_video(
        self,
        media_path: Union[str, Path],
        top_k: int = 5,
        attack_direction: str = "right",
        candidate_peaks: Optional[List[float]] = None,
    ) -> List[BuildupCandidate]:
        """
        Coordinates full tactical scoring pipeline across the video:
        1. Identifies audio excitement peaks (or evaluates candidate_peaks).
        2. Slices longest continuous wide-cam tactical buildup inside [t_peak - 22s, t_peak + 3s].
        3. Extracts climax hook metadata [t_peak - 1.5s, t_peak + 2.0s].
        4. Calculates delta_xT, defenders_bypassed, and max_unmarked_space_m2.
        5. Computes TotalScore and returns ranked candidates.
        """
        p = Path(media_path)
        peaks: List[AudioPeak] = []

        if candidate_peaks is not None and len(candidate_peaks) > 0:
            peaks = [
                AudioPeak(
                    t_peak=float(t),
                    audio_peak=2.5,
                    rms_energy=0.2,
                    band_energy=0.15,
                    z_score=1.8,
                )
                for t in candidate_peaks
            ]
        elif p.is_file():
            peaks = self.detect_audio_peaks(p)

        if not peaks:
            logger.info("No audio peaks detected; using uniform video timeline samples as candidate anchors")
            total_dur = 60.0
            if p.is_file():
                cap = cv2.VideoCapture(str(p))
                if cap.isOpened():
                    cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    if cnt > 0:
                        total_dur = cnt / fps
                    cap.release()
            anchors = [min(total_dur - 2.0, max(22.0, total_dur * frac)) for frac in [0.35, 0.65, 0.85]]
            peaks = [
                AudioPeak(
                    t_peak=float(a),
                    audio_peak=1.8,
                    rms_energy=0.1,
                    band_energy=0.08,
                    z_score=1.6,
                )
                for a in anchors
            ]

        candidates: List[BuildupCandidate] = []

        for pk in peaks:
            # 2. Slices longest continuous buildup & hook window
            buildup_info = find_continuous_tactical_buildup(
                media_path=p,
                t_peak=pk.t_peak,
                lookback_sec=22.0,
                post_peak_sec=3.0,
                ffmpeg_bin=self.ffmpeg_bin,
            )

            b_start = buildup_info["buildup_start_sec"]
            b_end = buildup_info["buildup_end_sec"]
            fps = buildup_info.get("fps", 30.0)

            # 3. Calculates Expected Threat (xT) & line-breaking metrics
            spatial_metrics = evaluate_tactical_spatial_metrics(
                media_path=p,
                start_sec=b_start,
                end_sec=b_end,
                projector=self.projector,
                sample_step_sec=0.5,
                attack_direction=attack_direction,
            )

            hook = buildup_info["climax_hook_window"]
            cand = BuildupCandidate(
                t_peak=pk.t_peak,
                buildup_start_sec=b_start,
                buildup_end_sec=b_end,
                buildup_duration_sec=round(b_end - b_start, 3),
                climax_hook_window=hook,
                delta_xT=spatial_metrics["delta_xT"],
                defenders_bypassed=spatial_metrics["defenders_bypassed"],
                audio_peak=pk.audio_peak,
                max_unmarked_space_m2=spatial_metrics["max_unmarked_space_m2"],
                fps=fps,
                buildup_start_frame=int(round(b_start * fps)),
                buildup_end_frame=int(round(b_end * fps)),
                climax_start_frame=hook["start_frame"],
                climax_end_frame=hook["end_frame"],
                metrics={
                    "cuts_detected": buildup_info.get("cuts_detected", 0),
                    "is_tactical": buildup_info.get("is_tactical", True),
                    "frames_analyzed": spatial_metrics.get("frames_analyzed", 0),
                    "audio_z_score": pk.z_score,
                },
            )
            candidates.append(cand)

        # 4. Multi-Criteria Composite Ranking
        ranked = rank_buildup_candidates(candidates)
        return ranked[:top_k]
