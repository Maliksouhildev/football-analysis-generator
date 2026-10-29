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
        - Freeze frame has ZERO drift: t_source(F_out) == cause_frame_idx / source_fps
          for every frame in the freeze window.
        - Source frame cause_frame_idx maps exactly to freeze_start_frame.
        """
        # 1. Resolve master audio durations
        b1_ms = beats.get("beat_1_setup", {}).get("duration_ms", 2000) if isinstance(beats.get("beat_1_setup"), dict) else 2000
        b2_ms = beats.get("beat_2_cause", {}).get("duration_ms", 5400) if isinstance(beats.get("beat_2_cause"), dict) else 5400
        b3_ms = beats.get("beat_3_effect", {}).get("duration_ms", 4200) if isinstance(beats.get("beat_3_effect"), dict) else 4200

        d_audio_1 = max(0.6, float(b1_ms) / 1000.0)
        d_audio_2 = max(1.5, float(b2_ms) / 1000.0)
        d_audio_3 = max(0.8, float(b3_ms) / 1000.0)

        # Exact total target audio frames partitioned per beat
        total_audio_sec = d_audio_1 + d_audio_2 + d_audio_3
        total_target_frames = int(round(total_audio_sec * self.fps))
        f1_frames = int(round(d_audio_1 * self.fps))
        f2_frames = int(round(d_audio_2 * self.fps))
        f3_frames = max(15, total_target_frames - f1_frames - f2_frames)
        total_target_frames = f1_frames + f2_frames + f3_frames

        # 2. Source video landmarks
        t_cause = max(0.1, float(cause_frame_idx) / source_fps)
        t_end = max(t_cause + 1.0, float(source_video_duration_sec))
        d_video_1 = t_cause
        d_video_3 = t_end - t_cause

        W = min(self.transition_frames, max(2, f1_frames // 4))

        # 3. Micro-beat 1 (Pre-roll: 0.0 to t_cause in f1_frames)
        # Target sum of rates: sum(rates[:f1_frames]) / self.fps == t_cause
        target_sum_1 = t_cause * self.fps
        rates = np.zeros(total_target_frames, dtype=np.float64)

        # Ease into freeze over the last W frames of Phase 1: [f1_frames - W, f1_frames]
        tau_ease = np.linspace(0.0, 1.0, W, endpoint=False)
        h_ease = 3.0 * (tau_ease ** 2) - 2.0 * (tau_ease ** 3)
        r_focus_phase1 = 0.70

        # Last W frames ease from r_focus_phase1 down to 0.0
        ease_in_rates = r_focus_phase1 * (1.0 - h_ease)
        sum_ease_in = float(np.sum(ease_in_rates))

        f_rem_p1 = max(1, f1_frames - W)
        sum_rem_p1 = max(0.1, target_sum_1 - sum_ease_in)
        avg_r_rem = sum_rem_p1 / float(f_rem_p1)

        segments_p1: List[Dict[str, Any]] = []

        if avg_r_rem > 1.05 and f_rem_p1 >= 10:
            # Transit acceleration + subtle deceleration before freeze
            f_focus_p1 = max(4, min(14, int(round(f_rem_p1 * 0.22))))
            f_transit_p1 = f_rem_p1 - f_focus_p1

            # Solve r_transit so sum equals sum_rem_p1
            r_transit = (sum_rem_p1 - f_focus_p1 * r_focus_phase1) / float(f_transit_p1)
            # Clamp transit within broadcast limits 1.25x - 1.35x
            if r_transit > 1.35:
                # If required rate exceeds 1.35, reduce f_focus_p1
                r_transit = 1.35
                r_focus_phase1 = max(0.60, (sum_rem_p1 - f_transit_p1 * r_transit) / float(f_focus_p1))
                ease_in_rates = r_focus_phase1 * (1.0 - h_ease)
                # Recalculate remaining to balance exactly
                rem_needed = target_sum_1 - float(np.sum(ease_in_rates)) - f_focus_p1 * r_focus_phase1
                r_transit = rem_needed / float(f_transit_p1)

            rates[:f_transit_p1] = r_transit
            rates[f_transit_p1:f_transit_p1 + f_focus_p1] = r_focus_phase1
            rates[f1_frames - W:f1_frames] = ease_in_rates

            # Apply smooth Hermite transition between transit and focus
            t_w = min(W, f_transit_p1, f_focus_p1)
            if t_w > 1:
                t_tau = np.linspace(0.0, 1.0, t_w)
                t_h = 3.0 * (t_tau ** 2) - 2.0 * (t_tau ** 3)
                b_start = f_transit_p1 - t_w // 2
                rates[b_start:b_start + t_w] = r_transit + (r_focus_phase1 - r_transit) * t_h

            t_transit_sec = round(float(np.sum(rates[:f_transit_p1])) / self.fps, 4)
            segments_p1.append({
                "segment_type": "FAST_TRANSIT",
                "rate": round(float(np.mean(rates[:f_transit_p1])), 3),
                "source_start_sec": 0.0,
                "source_end_sec": t_transit_sec,
                "output_frames": f_transit_p1,
                "reason": "Ball traveling across open space in transition",
            })
            segments_p1.append({
                "segment_type": "SLOW_FOCUS",
                "rate": round(r_focus_phase1, 3),
                "source_start_sec": t_transit_sec,
                "source_end_sec": round(t_cause, 4),
                "output_frames": f_focus_p1 + W,
                "reason": "Cinematic deceleration as carrier engages defender",
            })
        else:
            # Single uniform pre-roll flow with smooth ease into freeze
            r_flat = sum_rem_p1 / float(f_rem_p1)
            rates[:f_rem_p1] = r_flat
            rates[f1_frames - W:f1_frames] = r_flat * (1.0 - h_ease)
            segments_p1.append({
                "segment_type": "NORMAL_FLOW" if 0.90 <= r_flat <= 1.10 else ("FAST_TRANSIT" if r_flat > 1.10 else "SLOW_FOCUS"),
                "rate": round(float(r_flat), 3),
                "source_start_sec": 0.0,
                "source_end_sec": round(t_cause, 4),
                "output_frames": f1_frames,
                "reason": "Pre-roll buildup flow",
            })

        # Ensure Phase 1 sum equals t_cause * fps EXACTLY to machine precision
        scale_p1 = target_sum_1 / max(1e-6, float(np.sum(rates[:f1_frames])))
        rates[:f1_frames] *= scale_p1

        # 4. Micro-beat 2 (Tactical Freeze Hold at t_cause)
        # Rates MUST be strictly 0.00 for the entire freeze duration
        freeze_start_frame = f1_frames
        freeze_end_frame = f1_frames + f2_frames
        rates[freeze_start_frame:freeze_end_frame] = 0.0

        segment_p2 = {
            "segment_type": "FREEZE_HOLD",
            "rate": 0.00,
            "source_start_sec": round(t_cause, 4),
            "source_end_sec": round(t_cause, 4),
            "output_frames": f2_frames,
            "reason": "Tactical freeze hold for telestration HUD reveal",
        }

        # 5. Micro-beat 3 (Post-roll: t_cause to t_end in f3_frames)
        # Dedicated SLOW_FOCUS right as video unfreezes (skill, turn, deflection)
        W_post = min(W, max(2, f3_frames // 4))
        tau_out = np.linspace(0.0, 1.0, W_post)
        h_out = 3.0 * (tau_out ** 2) - 2.0 * (tau_out ** 3)
        r_focus_phase3 = 0.68

        # Ease out of freeze from 0.0 up to r_focus_phase3
        ease_out_rates = r_focus_phase3 * h_out
        f_focus_flat = max(6, min(24, int(round(f3_frames * 0.40)) - W_post))
        f_focus_total = W_post + f_focus_flat
        f_flow_total = max(1, f3_frames - f_focus_total)

        # Remaining source footage to cover in post-roll
        target_rem_source_sec = min(d_video_3, (f_focus_total * (r_focus_phase3 / self.fps)) + (f_flow_total * (1.30 / self.fps)))
        source_covered_focus = (float(np.sum(ease_out_rates)) + f_focus_flat * r_focus_phase3) / self.fps
        source_needed_flow = max(0.2, target_rem_source_sec - source_covered_focus)
        r_flow_phase3 = min(1.35, max(0.90, (source_needed_flow * self.fps) / float(f_flow_total)))

        p3_start = freeze_end_frame
        rates[p3_start:p3_start + W_post] = ease_out_rates
        rates[p3_start + W_post:p3_start + f_focus_total] = r_focus_phase3
        rates[p3_start + f_focus_total:p3_start + f3_frames] = r_flow_phase3

        # Apply smooth Hermite transition between post-roll focus and flow
        t_w3 = min(W_post, f_focus_flat, f_flow_total)
        if t_w3 > 1:
            t_tau3 = np.linspace(0.0, 1.0, t_w3)
            t_h3 = 3.0 * (t_tau3 ** 2) - 2.0 * (t_tau3 ** 3)
            b_start3 = (p3_start + f_focus_total) - t_w3 // 2
            rates[b_start3:b_start3 + t_w3] = r_focus_phase3 + (r_flow_phase3 - r_focus_phase3) * t_h3

        focus_end_sec = round(t_cause + (float(np.sum(rates[p3_start:p3_start + f_focus_total])) / self.fps), 4)
        end_sec_total = round(t_cause + (float(np.sum(rates[p3_start:])) / self.fps), 4)

        segments_p3: List[Dict[str, Any]] = [
            {
                "segment_type": "SLOW_FOCUS",
                "rate": round(r_focus_phase3, 3),
                "source_start_sec": round(t_cause, 4),
                "source_end_sec": focus_end_sec,
                "output_frames": f_focus_total,
                "reason": "Subtle slow-motion focus during decisive turn and execution",
            },
            {
                "segment_type": "FAST_TRANSIT" if r_flow_phase3 >= 1.15 else "NORMAL_FLOW",
                "rate": round(float(r_flow_phase3), 3),
                "source_start_sec": focus_end_sec,
                "source_end_sec": end_sec_total,
                "output_frames": f_flow_total,
                "reason": "Consequence unfolds into attack progression or clearance",
            },
        ]

        # Combine all resolved continuous segments
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
        out_time_map = self.build_continuous_piecewise_time_map(
            rates=rates,
            freeze_start_frame=freeze_start_frame,
            freeze_end_frame=freeze_end_frame,
            t_cause=t_cause,
            source_start_sec=0.0,
        )

        bidirectional = self.build_bidirectional_lookup(
            out_frame_to_source_time_sec=out_time_map,
            cause_frame_idx=cause_frame_idx,
            freeze_start_frame=freeze_start_frame,
            freeze_end_frame=freeze_end_frame,
            source_fps=source_fps,
        )

        actual_video_frames = len(out_time_map)
        frame_diff = abs(actual_video_frames - total_target_frames)
        assert frame_diff <= 1, f"Frame sync violation: {actual_video_frames} video frames vs {total_target_frames} audio frames (diff: {frame_diff})"

        return {
            "speed_segments": all_segments,
            "total_output_frames": actual_video_frames,
            "total_audio_frames": total_target_frames,
            "frame_diff": frame_diff,
            "out_frame_to_source_time_sec": out_time_map,
            "instantaneous_rates": rates,
            "source_frame_to_out_frame": bidirectional["source_frame_to_out_frame"],
            "source_frame_to_out_frame_map": bidirectional["source_frame_to_out_frame_map"],
            "freeze_start_frame": freeze_start_frame,
            "freeze_duration_frames": f2_frames,
            "freeze_end_frame": freeze_end_frame,
        }

    def build_continuous_piecewise_time_map(
        self,
        rates: np.ndarray,
        freeze_start_frame: int,
        freeze_end_frame: int,
        t_cause: float,
        source_start_sec: float = 0.0,
    ) -> np.ndarray:
        """
        Integrates instantaneous rates into source timestamps:
            t_source(F_out) = sum_{k=0}^{F_out} (r(k) / 30.0)
        Guarantees:
        - t_source(F_out) == t_cause for all F_out in [freeze_start_frame, freeze_end_frame - 1].
        - Discrete monotonic continuity without drift.
        """
        N_out = len(rates)
        t_source = np.zeros(N_out, dtype=np.float64)

        # Phase 1 integration
        accum = float(source_start_sec)
        for k in range(freeze_start_frame):
            accum += rates[k] / float(self.fps)
            t_source[k] = min(t_cause, accum)

        # Freeze hold: strictly locked to t_cause (0 drift)
        for k in range(freeze_start_frame, freeze_end_frame):
            t_source[k] = t_cause

        # Phase 3 integration
        accum_post = t_cause
        for k in range(freeze_end_frame, N_out):
            accum_post += rates[k] / float(self.fps)
            t_source[k] = accum_post

        return t_source

    def build_bidirectional_lookup(
        self,
        out_frame_to_source_time_sec: np.ndarray,
        cause_frame_idx: int,
        freeze_start_frame: int,
        freeze_end_frame: int,
        source_fps: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Constructs bidirectional lookup:
        - out_frame_to_source_time_sec: List[float]
        - source_frame_to_out_frame: exact inversion mapping cause_frame_idx directly
          to freeze_start_frame without search ambiguity.
        - source_frame_to_out_frame_map: Dict[int, int]
        """
        N_out = len(out_frame_to_source_time_sec)
        out_list = [round(float(t), 4) for t in out_frame_to_source_time_sec]
        t_cause = float(cause_frame_idx) / source_fps

        def source_frame_to_out_frame(f_src: Union[int, float]) -> int:
            f_val = float(f_src)
            # Exact mapping for cause frame: returns freeze start frame
            if abs(f_val - cause_frame_idx) < 1e-4:
                return freeze_start_frame

            t_target = f_val / source_fps
            if t_target < t_cause:
                # Search in pre-roll
                idx = int(np.searchsorted(out_frame_to_source_time_sec[:freeze_start_frame], t_target, side="left"))
                return max(0, min(freeze_start_frame, idx))
            else:
                # Search in post-roll
                if freeze_end_frame >= N_out:
                    return freeze_end_frame
                idx = freeze_end_frame + int(np.searchsorted(out_frame_to_source_time_sec[freeze_end_frame:], t_target, side="left"))
                return max(freeze_end_frame, min(N_out - 1, idx))

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
        Remaps all vision coordinates, tracking detections, freeze landmarks, pass
        vectors, and ball carrier mappings through the piecewise elastic time map
        into Remotion output frames F_out.
        """
        N_out = len(out_frame_to_source_time_sec)
        warped_tracks: Dict[str, List[Dict[str, Any]]] = {}

        # 1. Remap tracks_per_frame directly to output frames F_out
        source_frame_keys = [int(k) for k in tracks_per_frame.keys() if str(k).isdigit()]
        warped_cause_idx = source_frame_to_out_frame(cause_frame_idx)
        warped_effect_idx = source_frame_to_out_frame(effect_frame_idx)
        freeze_start_frame = warped_cause_idx
        freeze_end_frame = warped_cause_idx + freeze_frames

        for f_out in range(N_out):
            if freeze_start_frame <= f_out < freeze_end_frame:
                # During freeze: lock strictly to cause_frame_idx detections
                f_key = str(cause_frame_idx)
                if f_key in tracks_per_frame:
                    warped_tracks[str(f_out)] = tracks_per_frame[f_key]
                elif source_frame_keys:
                    closest_k = min(source_frame_keys, key=lambda k: abs(k - cause_frame_idx))
                    warped_tracks[str(f_out)] = tracks_per_frame.get(str(closest_k), [])
                else:
                    warped_tracks[str(f_out)] = []
            else:
                t_src = out_frame_to_source_time_sec[f_out]
                f_src = int(round(t_src * source_fps))
                f_key = str(f_src)
                if f_key in tracks_per_frame:
                    warped_tracks[str(f_out)] = tracks_per_frame[f_key]
                elif source_frame_keys:
                    closest_k = min(source_frame_keys, key=lambda k: abs(k - f_src))
                    warped_tracks[str(f_out)] = tracks_per_frame.get(str(closest_k), [])
                else:
                    warped_tracks[str(f_out)] = []

        # 2. Remap landmarks and vectors if present
        remapped_tel = dict(cause_telestration) if cause_telestration else {}
        if "vectors" in remapped_tel and isinstance(remapped_tel["vectors"], list):
            new_vectors = []
            for v in remapped_tel["vectors"]:
                v_copy = dict(v)
                if "pass_start_frame" in v_copy and isinstance(v_copy["pass_start_frame"], (int, float)):
                    v_copy["pass_start_frame"] = source_frame_to_out_frame(v_copy["pass_start_frame"])
                if "pass_receive_frame" in v_copy and isinstance(v_copy["pass_receive_frame"], (int, float)):
                    v_copy["pass_receive_frame"] = source_frame_to_out_frame(v_copy["pass_receive_frame"])
                if "landing_frame" in v_copy and isinstance(v_copy["landing_frame"], (int, float)):
                    v_copy["landing_frame"] = source_frame_to_out_frame(v_copy["landing_frame"])
                if "start_frame" in v_copy and isinstance(v_copy["start_frame"], (int, float)):
                    v_copy["start_frame"] = source_frame_to_out_frame(v_copy["start_frame"])
                if "end_frame" in v_copy and isinstance(v_copy["end_frame"], (int, float)):
                    v_copy["end_frame"] = source_frame_to_out_frame(v_copy["end_frame"])
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

        # 3. Remap ball_carrier_per_frame
        if "ball_carrier_per_frame" in remapped_tel and isinstance(remapped_tel["ball_carrier_per_frame"], dict):
            orig_bc = remapped_tel["ball_carrier_per_frame"]
            bc_keys = [int(k) for k in orig_bc.keys() if str(k).isdigit()]
            warped_bc: Dict[str, Any] = {}
            for f_out in range(N_out):
                if freeze_start_frame <= f_out < freeze_end_frame:
                    f_src = cause_frame_idx
                else:
                    t_src = out_frame_to_source_time_sec[f_out]
                    f_src = int(round(t_src * source_fps))
                f_key = str(f_src)
                if f_key in orig_bc:
                    warped_bc[str(f_out)] = orig_bc[f_key]
                elif bc_keys:
                    closest_k = min(bc_keys, key=lambda k: abs(k - f_src))
                    warped_bc[str(f_out)] = orig_bc.get(str(closest_k))
            remapped_tel["ball_carrier_per_frame"] = warped_bc

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
                    # Keep up to 32 recent frames in LRU order
                    if len(frame_cache) > 32:
                        frame_cache.pop(next(iter(frame_cache)))
                    frame_cache[idx_clamped] = frame
                    return frame

            # Random seek fallback
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_clamped)
            ret, frame = cap.read()
            last_read_idx = idx_clamped
            if ret and frame is not None:
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                if len(frame_cache) > 32:
                    frame_cache.pop(next(iter(frame_cache)))
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

                if 0.05 <= r_now < 0.85 and 0.001 < alpha < 0.999 and f_floor != f_ceil:
                    # Silky slow-motion alpha-blending between adjacent source frames during deceleration
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
