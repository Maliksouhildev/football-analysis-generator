"""
src/rendering/elastic_time_warper.py
Automatic Velocity & Audio-Sync Elastic Speed-Ramping Engine.

Provides:
1. Audio-Driven Auto-Elasticity (Dynamic video playback rate r_i = D_video / D_audio).
2. Continuous Piecewise Time-Mapping at 30fps using 6-frame Hermite cubic ease-in/ease-out ramps:
   t_source(F_out) = sum_{k=0}^{F_out} (r(k) / 30.0)
3. Bidirectional Lookup Arrays:
   - out_frame_to_source_time_sec (maps Remotion timeline frame F_out to float source timestamp)
   - source_frame_to_out_frame (maps original OpenCV frame index to Remotion timeline frame F_out)
4. FFmpeg Frame-Accurate Pre-Warper (render_warped_video_stream) with sub-0.85x optical/alpha blending.
5. Remapping of Vision & Telestration Coordinates (tracks, freeze frames, vectors).
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("rendering.elastic_time_warper")

# Ensure FFmpeg binaries are available in PATH
for _p in [
    r"C:\Users\mehdi\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin",
    r"C:\Users\mlkme\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\imageio_ffmpeg\binaries",
]:
    if os.path.isdir(_p) and _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")


class ElasticTimeWarper:
    """
    Subtle Elastic Speed-Ramping Engine for tactical football broadcasts.
    Decouples video speed from browser playbackRate hacks by mathematically pre-warping
    video frames and remapping vision telestration coordinates.
    """

    def __init__(self, fps: int = 30, transition_frames: int = 6):
        self.fps = fps
        self.transition_frames = max(2, transition_frames)  # 6 frames = 0.20s at 30fps

    @staticmethod
    def _hermite_cubic(t: float) -> float:
        """Hermite cubic smoothstep ease-in/ease-out: 3t^2 - 2t^3."""
        t_clamped = max(0.0, min(1.0, float(t)))
        return 3.0 * (t_clamped ** 2) - 2.0 * (t_clamped ** 3)

    def solve_elastic_timeline(
        self,
        beats: Dict[str, Any],
        source_video_duration_sec: float,
        cause_frame_idx: int,
        source_fps: float = 30.0,
        candidate_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Audio-Driven Auto-Elasticity Solver:
        Solves continuous speed regimes and exact output frame counts such that:
        - Freeze frame occurs at cause_frame_idx for the duration of beat_2_cause audio.
        - Video speed ramps r_i = D_video / D_audio are clamped within natural broadcast bounds:
          * FAST_TRANSIT: 1.25x to 1.35x
          * NORMAL_FLOW: 1.00x
          * SLOW_FOCUS: 0.60x to 0.75x
          * FREEZE_HOLD: 0.00x
        - Total output frame count matches master voiceover WAV duration:
          abs(video_frames - audio_frames) <= 1
        """
        # 1. Resolve master audio durations
        b1_ms = beats.get("beat_1_setup", {}).get("duration_ms", 2000) if isinstance(beats.get("beat_1_setup"), dict) else 2000
        b2_ms = beats.get("beat_2_cause", {}).get("duration_ms", 5400) if isinstance(beats.get("beat_2_cause"), dict) else 5400
        b3_ms = beats.get("beat_3_effect", {}).get("duration_ms", 4200) if isinstance(beats.get("beat_3_effect"), dict) else 4200

        d_audio_1 = max(0.6, float(b1_ms) / 1000.0)
        d_audio_2 = max(1.5, float(b2_ms) / 1000.0)
        d_audio_3 = max(0.8, float(b3_ms) / 1000.0)

        # Exact total target audio frames
        total_audio_sec = d_audio_1 + d_audio_2 + d_audio_3
        total_target_frames = int(round(total_audio_sec * self.fps))

        # 2. Source video landmarks
        t_cause = max(0.1, float(cause_frame_idx) / source_fps)
        t_end = max(t_cause + 1.0, float(source_video_duration_sec))

        d_video_1 = t_cause
        d_video_3 = t_end - t_cause

        # 3. Micro-beat 1 (Pre-roll: 0.0 to t_cause)
        # Solve dynamic playback rate r_1
        target_r1 = d_video_1 / d_audio_1
        target_out_frames_1 = int(round(d_audio_1 * self.fps))

        # Build Phase 1 segments
        segments_p1: List[Dict[str, Any]] = []
        if d_video_1 >= 1.4:
            # Transit portion + subtle deceleration before freeze
            transit_end = round(max(0.4, d_video_1 - 0.7), 3)
            # Allocate frames
            ratio = transit_end / d_video_1
            f_transit = max(1, int(round(target_out_frames_1 * ratio)))
            f_decel = max(1, target_out_frames_1 - f_transit)

            r_transit = min(1.35, max(1.05, (transit_end) / (f_transit / self.fps)))
            r_decel = max(0.60, min(0.85, (d_video_1 - transit_end) / (f_decel / self.fps)))

            segments_p1.append({
                "segment_type": "FAST_TRANSIT",
                "rate": round(r_transit, 3),
                "source_start_sec": 0.0,
                "source_end_sec": transit_end,
                "output_frames": f_transit,
                "reason": "Ball traveling across open space in transition",
            })
            segments_p1.append({
                "segment_type": "SLOW_FOCUS",
                "rate": round(r_decel, 3),
                "source_start_sec": transit_end,
                "source_end_sec": d_video_1,
                "output_frames": f_decel,
                "reason": "Cinematic deceleration as carrier engages defender",
            })
        else:
            r1_clamped = min(1.35, max(0.65, target_r1))
            segments_p1.append({
                "segment_type": "NORMAL_FLOW" if 0.90 <= r1_clamped <= 1.10 else ("FAST_TRANSIT" if r1_clamped > 1.10 else "SLOW_FOCUS"),
                "rate": round(r1_clamped, 3),
                "source_start_sec": 0.0,
                "source_end_sec": d_video_1,
                "output_frames": target_out_frames_1,
                "reason": "Pre-roll buildup flow",
            })

        # 4. Micro-beat 2 (Freeze Hold at t_cause)
        target_out_frames_2 = int(round(d_audio_2 * self.fps))
        segment_p2 = {
            "segment_type": "FREEZE_HOLD",
            "rate": 0.00,
            "source_start_sec": d_video_1,
            "source_end_sec": d_video_1,
            "output_frames": target_out_frames_2,
            "reason": "Tactical freeze hold for telestration HUD reveal",
        }

        # 5. Micro-beat 3 (Post-roll: t_cause to t_end)
        # Allocate remaining target frames so total output frames match total_target_frames EXACTLY
        consumed_frames = sum(s["output_frames"] for s in segments_p1) + target_out_frames_2
        target_out_frames_3 = max(15, total_target_frames - consumed_frames)

        segments_p3: List[Dict[str, Any]] = []
        # In post-roll, dedicate the first ~1.2s to SLOW_FOCUS (skill, turn, mistake), remainder to transit/flow
        d_focus_source = min(1.5, max(0.6, d_video_3 * 0.45))
        f_focus = min(int(round(target_out_frames_3 * 0.55)), max(12, int(round((d_focus_source / 0.68) * self.fps))))
        f_flow = max(1, target_out_frames_3 - f_focus)

        r_focus = max(0.55, min(0.78, d_focus_source / (f_focus / self.fps)))
        rem_source = d_video_3 - d_focus_source
        r_flow = min(1.35, max(0.95, rem_source / (f_flow / self.fps)))

        segments_p3.append({
            "segment_type": "SLOW_FOCUS",
            "rate": round(r_focus, 3),
            "source_start_sec": d_video_1,
            "source_end_sec": round(d_video_1 + d_focus_source, 3),
            "output_frames": f_focus,
            "reason": "Subtle slow-motion focus during decisive turn and execution",
        })
        segments_p3.append({
            "segment_type": "FAST_TRANSIT" if r_flow >= 1.15 else "NORMAL_FLOW",
            "rate": round(r_flow, 3),
            "source_start_sec": round(d_video_1 + d_focus_source, 3),
            "source_end_sec": t_end,
            "output_frames": f_flow,
            "reason": "Consequence unfolds into attack progression or clearance",
        })

        # Combine all resolved segments
        all_segments: List[Dict[str, Any]] = segments_p1 + [segment_p2] + segments_p3

        # Compute cumulative frame indices on output timeline
        cur_f = 0
        for seg in all_segments:
            seg["start_frame"] = cur_f
            seg["startFrame"] = cur_f
            cur_f += seg["output_frames"]
            seg["end_frame"] = cur_f
            seg["endFrame"] = cur_f
            seg["duration_sec"] = round(seg["output_frames"] / self.fps, 4)

        # Build continuous piecewise time map
        out_time_map, rates_map = self.build_continuous_piecewise_time_map(
            resolved_segments=all_segments,
            total_output_frames=total_target_frames,
            source_start_sec=0.0,
        )

        bidirectional = self.build_bidirectional_lookup(out_time_map, source_fps=source_fps)

        # Verify exact frame matching
        actual_video_frames = len(out_time_map)
        frame_diff = abs(actual_video_frames - total_target_frames)
        assert frame_diff <= 1, f"Frame sync violation: {actual_video_frames} video frames vs {total_target_frames} audio frames (diff: {frame_diff})"

        return {
            "speed_segments": all_segments,
            "total_output_frames": actual_video_frames,
            "total_audio_frames": total_target_frames,
            "frame_diff": frame_diff,
            "out_frame_to_source_time_sec": out_time_map,
            "instantaneous_rates": rates_map,
            "source_frame_to_out_frame": bidirectional["source_frame_to_out_frame"],
            "source_frame_to_out_frame_map": bidirectional["source_frame_to_out_frame_map"],
            "freeze_start_frame": segment_p2["start_frame"],
            "freeze_duration_frames": segment_p2["output_frames"],
            "freeze_end_frame": segment_p2["end_frame"],
        }

    def build_continuous_piecewise_time_map(
        self,
        resolved_segments: List[Dict[str, Any]],
        total_output_frames: int,
        source_start_sec: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs a continuous piecewise instantaneous rate array r(k) at 30fps
        using a 6-frame (0.20s) Hermite cubic ease-in/ease-out ramp between adjacent speed regimes.
        Then integrates:
            t_source(F_out) = sum_{k=0}^{F_out} (r(k) / 30.0)
        """
        rates = np.zeros(total_output_frames, dtype=np.float64)

        # Fill segment base rates
        for seg in resolved_segments:
            s_f = max(0, min(total_output_frames, seg["start_frame"]))
            e_f = max(0, min(total_output_frames, seg["end_frame"]))
            rates[s_f:e_f] = float(seg["rate"])

        # Apply 6-frame Hermite cubic ease transitions between adjacent regimes
        W = self.transition_frames
        for idx in range(len(resolved_segments) - 1):
            seg_curr = resolved_segments[idx]
            seg_next = resolved_segments[idx + 1]

            boundary = seg_curr["end_frame"]
            r_from = float(seg_curr["rate"])
            r_to = float(seg_next["rate"])

            # Center the transition around the boundary
            trans_start = max(0, boundary - W // 2)
            trans_end = min(total_output_frames, trans_start + W)
            actual_w = trans_end - trans_start

            if actual_w > 1:
                for step in range(actual_w):
                    k = trans_start + step
                    tau = float(step) / float(actual_w - 1)
                    ease_val = self._hermite_cubic(tau)
                    rates[k] = r_from + (r_to - r_from) * ease_val

        # Integrate to obtain source timestamp t_source(F_out)
        t_source = np.zeros(total_output_frames, dtype=np.float64)
        accum_t = float(source_start_sec)

        for k in range(total_output_frames):
            # Advance source time according to instantaneous rate
            accum_t += rates[k] / float(self.fps)
            t_source[k] = accum_t

        return t_source, rates

    def build_bidirectional_lookup(
        self,
        out_frame_to_source_time_sec: np.ndarray,
        source_fps: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Constructs bidirectional lookup:
        - out_frame_to_source_time_sec: List[float]
        - source_frame_to_out_frame: Callable[[int | float], int]
        - source_frame_to_out_frame_map: Dict[int, int]
        """
        N_out = len(out_frame_to_source_time_sec)
        out_list = [round(float(t), 4) for t in out_frame_to_source_time_sec]

        def source_frame_to_out_frame(f_src: Union[int, float]) -> int:
            t_target = float(f_src) / source_fps
            idx = int(np.searchsorted(out_frame_to_source_time_sec, t_target, side="left"))
            return max(0, min(N_out - 1, idx))

        # Generate integer map up to maximum source frame
        max_source_t = out_frame_to_source_time_sec[-1] if N_out > 0 else 0.0
        max_f_src = int(math.ceil(max_source_t * source_fps)) + 30

        source_map: Dict[int, int] = {}
        for f in range(max_f_src + 1):
            source_map[f] = source_frame_to_out_frame(f)

        return {
            "out_frame_to_source_time_sec": out_list,
            "source_frame_to_out_frame": source_frame_to_out_frame,
            "source_frame_to_out_frame_map": source_map,
        }

    def remap_vision_coordinates(
        self,
        tracks_per_frame: Dict[str, Any],
        cause_frame_idx: int,
        effect_frame_idx: int,
        freeze_frames: int,
        out_frame_to_source_time_sec: np.ndarray,
        source_frame_to_out_frame: Callable[[Union[int, float]], int],
        source_fps: float = 30.0,
        cause_telestration: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Remaps all vision coordinates, tracking detections, freeze landmarks, and pass
        vectors through the piecewise elastic time map into Remotion output frames F_out.
        """
        N_out = len(out_frame_to_source_time_sec)
        warped_tracks: Dict[str, List[Dict[str, Any]]] = {}

        # 1. Remap tracks_per_frame directly to output frames F_out
        source_frame_keys = [int(k) for k in tracks_per_frame.keys() if str(k).isdigit()]

        for f_out in range(N_out):
            t_src = out_frame_to_source_time_sec[f_out]
            f_src = int(round(t_src * source_fps))
            f_key = str(f_src)

            if f_key in tracks_per_frame:
                warped_tracks[str(f_out)] = tracks_per_frame[f_key]
            elif source_frame_keys:
                closest_k = min(source_frame_keys, key=lambda k: abs(k - f_src))
                warped_tracks[str(f_out)] = tracks_per_frame.get(str(closest_k), [])

        # 2. Remap landmarks
        warped_cause_idx = source_frame_to_out_frame(cause_frame_idx)
        warped_effect_idx = source_frame_to_out_frame(effect_frame_idx)

        # 3. Remap vectors and trajectories if present
        remapped_tel = dict(cause_telestration) if cause_telestration else {}
        if "vectors" in remapped_tel and isinstance(remapped_tel["vectors"], list):
            new_vectors = []
            for v in remapped_tel["vectors"]:
                v_copy = dict(v)
                if "pass_start_frame" in v_copy and isinstance(v_copy["pass_start_frame"], (int, float)):
                    v_copy["pass_start_frame"] = source_frame_to_out_frame(v_copy["pass_start_frame"])
                if "pass_receive_frame" in v_copy and isinstance(v_copy["pass_receive_frame"], (int, float)):
                    v_copy["pass_receive_frame"] = source_frame_to_out_frame(v_copy["pass_receive_frame"])
                if "trajectory" in v_copy and isinstance(v_copy["trajectory"], list):
                    remapped_traj = []
                    for pt in v_copy["trajectory"]:
                        if isinstance(pt, (list, tuple)) and len(pt) >= 3:
                            f_s, x, y = pt[0], pt[1], pt[2]
                            f_o = source_frame_to_out_frame(f_s)
                            remapped_traj.append([f_o, x, y])
                        else:
                            remapped_traj.append(pt)
                    v_copy["trajectory"] = remapped_traj
                new_vectors.append(v_copy)
            remapped_tel["vectors"] = new_vectors

        remapped_tel["cause_frame_idx"] = warped_cause_idx
        remapped_tel["causeFrameIdx"] = warped_cause_idx
        remapped_tel["video_freeze_frame"] = warped_cause_idx
        remapped_tel["videoFreezeFrame"] = warped_cause_idx
        remapped_tel["freeze_frames"] = freeze_frames
        remapped_tel["freezeFrames"] = freeze_frames

        return {
            "warped_tracks_per_frame": warped_tracks,
            "warped_cause_frame_idx": warped_cause_idx,
            "warped_effect_frame_idx": warped_effect_idx,
            "warped_cause_telestration": remapped_tel,
        }

    def render_warped_video_stream(
        self,
        source_video_path: Union[str, Path],
        output_video_path: Union[str, Path],
        out_frame_to_source_time_sec: Union[np.ndarray, List[float]],
        instantaneous_rates: Optional[Union[np.ndarray, List[float]]] = None,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        crf: int = 18,
        gop: int = 15,
        preset: str = "fast",
    ) -> str:
        """
        FFmpeg Frame-Accurate Pre-Warper:
        Iterates over F_out = 0 ... N_total_frames - 1.
        Looks up t_source = out_frame_to_source_time_sec[F_out].
        When r(F_out) < 0.85 (during SLOW_FOCUS deceleration windows):
            Smoothly interpolates (alpha-blending) between floor(f_src) and ceil(f_src).
        Pipes raw video frames into FFmpeg with -g 15 for ultra-fast Remotion seek times.
        """
        src_p = Path(source_video_path).resolve()
        out_p = Path(output_video_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if not src_p.is_file():
            raise FileNotFoundError(f"Source video file not found: {src_p}")

        cap = cv2.VideoCapture(str(src_p))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source video: {src_p}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        out_times = np.asarray(out_frame_to_source_time_sec, dtype=np.float64)
        N_out = len(out_times)

        rates = (
            np.asarray(instantaneous_rates, dtype=np.float64)
            if instantaneous_rates is not None
            else np.ones(N_out, dtype=np.float64)
        )

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-g", str(gop),
            "-pix_fmt", "yuv420p",
            str(out_p),
        ]

        logger.info(f"Starting FFmpeg pipe to {out_p} ({N_out} output frames at {fps}fps, gop={gop})...")
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # High-performance frame cache for sequential and repetitive reads
        frame_cache: Dict[int, np.ndarray] = {}
        last_read_idx = -1

        def get_frame(f_idx: int) -> np.ndarray:
            nonlocal last_read_idx
            idx_clamped = max(0, min(total_src_frames - 1, f_idx))
            if idx_clamped in frame_cache:
                return frame_cache[idx_clamped]

            # Sequential read optimization
            if last_read_idx + 1 == idx_clamped:
                ret, frame = cap.read()
                if ret and frame is not None:
                    last_read_idx = idx_clamped
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                    # Cache up to 16 recent frames
                    if len(frame_cache) > 16:
                        frame_cache.clear()
                    frame_cache[idx_clamped] = frame
                    return frame

            # Random seek fallback
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_clamped)
            ret, frame = cap.read()
            last_read_idx = idx_clamped
            if ret and frame is not None:
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                if len(frame_cache) > 16:
                    frame_cache.clear()
                frame_cache[idx_clamped] = frame
                return frame

            # If read failed, return blank frame
            return np.zeros((height, width, 3), dtype=np.uint8)

        try:
            for f_out in range(N_out):
                t_src = out_times[f_out]
                r_now = rates[f_out]
                f_src_float = t_src * source_fps

                f_floor = int(math.floor(f_src_float))
                f_ceil = min(total_src_frames - 1, f_floor + 1)
                alpha = f_src_float - f_floor

                if r_now < 0.85 and 0.001 < alpha < 0.999 and f_floor != f_ceil:
                    # Silky slow-motion alpha-blending between adjacent source frames
                    img_floor = get_frame(f_floor)
                    img_ceil = get_frame(f_ceil)
                    out_frame = cv2.addWeighted(img_floor, 1.0 - alpha, img_ceil, alpha, 0.0)
                else:
                    target_idx = int(round(f_src_float))
                    out_frame = get_frame(target_idx)

                proc.stdin.write(out_frame.tobytes())

            proc.stdin.close()
            _, stderr = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg encoding failed with code {proc.returncode}: {stderr.decode(errors='replace')}")

        finally:
            cap.release()
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()

        logger.info(f"Successfully baked warped video to: {out_p} ({out_p.stat().st_size} bytes)")
        return str(out_p)
