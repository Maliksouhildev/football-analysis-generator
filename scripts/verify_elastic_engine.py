"""
scripts/verify_elastic_engine.py
Comprehensive End-to-End Test Suite for Subtle Elastic Speed-Ramping Engine:
1. Storyboard Schema & Speed Segment Classification.
2. Elastic Solver & Hermite Cubic Ease Continuity.
3. Audio-Sync Frame Accuracy: abs(video_frames - audio_frames) <= 1.
4. Bidirectional Lookup & Vision Coordinate Remapping.
5. FFmpeg Frame-Accurate Pre-Warper on Real Footage.
6. SFX Asset Verification (slow_focus_swell.wav).
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from scipy.io import wavfile

from src.rendering.tactical_narrator import (
    TacticalNarrator,
    TacticalStoryboardSchema,
    SpeedSegment,
)
from src.rendering.elastic_time_warper import ElasticTimeWarper
from src.rendering.sfx_generator import ProceduralSFXGenerator, synthesize_slow_focus_swell


def test_storyboard_schema():
    print("\n--- Test 1: Storyboard Schema & Speed Segments ---")
    narrator = TacticalNarrator(api_key="")
    event_window = {
        "clip_id": "clip_01_left_wing",
        "home_team": "AS Roma",
        "away_team": "Real Madrid",
        "t_cause_clock": "83:56",
        "narrative_tone": "RUTHLESS_AUTOPSY",
        "thesis": "Defensive block stretched laterally due to aggressive flank overload",
    }
    cause_tel = {
        "cause_frame_idx": 80,
        "video_freeze_frame": 80,
        "catalyst_player": {"name": "Vinicius Jr", "role": "LW"},
        "vacated_label": "VACATED FLANK CORRIDOR",
    }

    script = narrator.generate_3beat_script(event_window, cause_telestration=cause_tel)
    assert "speed_segments" in script, "speed_segments missing from generated script!"
    assert len(script["speed_segments"]) >= 4, f"Expected at least 4 speed segments, got {len(script['speed_segments'])}"

    # Validate against Pydantic schema
    storyboard = TacticalStoryboardSchema(**script)
    print(f"Validated TacticalStoryboardSchema: {len(storyboard.speed_segments)} continuous segments.")

    types = [s.segment_type for s in storyboard.speed_segments]
    print(f"Segment regimes: {types}")
    assert "FREEZE_HOLD" in types, "FREEZE_HOLD regime missing!"
    assert any(t in ("SLOW_FOCUS", "FAST_TRANSIT") for t in types), "Dynamic speed regimes missing!"
    print("Test 1 PASSED!")


def test_elastic_solver_and_easing():
    print("\n--- Test 2: Elastic Solver & Hermite Easing Continuity ---")
    warper = ElasticTimeWarper(fps=30, transition_frames=6)
    beats = {
        "beat_1_setup": {"duration_ms": 2200},  # 2.2s -> ~66 frames
        "beat_2_cause": {"duration_ms": 5500},  # 5.5s -> ~165 frames
        "beat_3_effect": {"duration_ms": 3300}, # 3.3s -> ~99 frames
    }
    # Total audio duration = 11.0s -> 330 frames
    sol = warper.solve_elastic_timeline(
        beats=beats,
        source_video_duration_sec=7.5,
        cause_frame_idx=80,
        source_fps=30.0,
    )

    total_out = sol["total_output_frames"]
    total_aud = sol["total_audio_frames"]
    diff = sol["frame_diff"]

    print(f"Total Video Output Frames: {total_out}")
    print(f"Total Audio Frames: {total_aud}")
    print(f"Frame Difference: {diff}")
    assert diff <= 1, f"Frame sync violation: diff={diff} exceeds 1 frame threshold!"

    rates = sol["instantaneous_rates"]
    # Check Hermite easing continuity: maximum single-frame delta
    deltas = np.abs(np.diff(rates))
    max_delta = np.max(deltas)
    print(f"Max single-frame rate delta: {max_delta:.4f}")
    assert max_delta < 0.35, "Abrupt speed cut detected! Transition not smooth."

    # Verify rates are bounded within natural broadcast regimes
    assert np.all(rates >= 0.0), "Negative rate detected!"
    assert np.all(rates <= 1.36), f"Rate {rates.max()} exceeded broadcast limit 1.35x!"

    # Verify freeze frame has ZERO drift
    t_cause_expected = 80 / 30.0
    f_f_start = sol["freeze_start_frame"]
    f_f_end = sol["freeze_end_frame"]
    t_map = sol["out_frame_to_source_time_sec"]

    print(f"Freeze window: output frames [{f_f_start}, {f_f_end})")
    print(f"t_source at freeze start ({f_f_start}): {t_map[f_f_start]:.4f}s (frame: {t_map[f_f_start] * 30.0:.2f})")
    print(f"t_source at freeze end-1 ({f_f_end - 1}): {t_map[f_f_end - 1]:.4f}s (frame: {t_map[f_f_end - 1] * 30.0:.2f})")

    assert abs(t_map[f_f_start] - t_cause_expected) < 1e-4, f"Freeze start mismatch: {t_map[f_f_start]} vs {t_cause_expected}"
    assert abs(t_map[f_f_end - 1] - t_cause_expected) < 1e-4, f"Freeze drift detected: {t_map[f_f_end - 1]} vs {t_cause_expected}"
    print("Zero-Drift Freeze Verification PASSED!")
    print("Test 2 PASSED!")


def test_bidirectional_lookup_and_remapping():
    print("\n--- Test 3: Bidirectional Lookup & Vision Remapping ---")
    warper = ElasticTimeWarper(fps=30)
    beats = {
        "beat_1_setup": {"duration_ms": 2000},  # 2.0s -> 60 frames
        "beat_2_cause": {"duration_ms": 5000},  # 5.0s -> 150 frames
        "beat_3_effect": {"duration_ms": 3000}, # 3.0s -> 90 frames
    }
    sol = warper.solve_elastic_timeline(beats, source_video_duration_sec=8.0, cause_frame_idx=80)

    # Test bidirectional conversion
    src_to_out = sol["source_frame_to_out_frame"]
    out_to_src = sol["out_frame_to_source_time_sec"]

    f_src_test = 80
    f_out = src_to_out(f_src_test)
    print(f"Source frame {f_src_test} maps to Output frame {f_out}")
    assert f_out == sol["freeze_start_frame"], f"Source cause frame 80 must map exactly to freeze start frame {sol['freeze_start_frame']}, got {f_out}!"

    # Remap vision tracks and vector trajectory
    mock_tracks = {
        "0": [{"track_id": 7, "x_pct": 20.0, "y_pct": 40.0}],
        "80": [{"track_id": 7, "x_pct": 55.0, "y_pct": 50.0}],
        "180": [{"track_id": 7, "x_pct": 85.0, "y_pct": 65.0}],
    }
    mock_tel = {
        "ball_carrier_per_frame": {"0": 7, "80": 7, "180": 9},
        "vectors": [
            {
                "from_track_id": 7,
                "pass_start_frame": 80,
                "pass_receive_frame": 180,
                "trajectory": [[80, 55.0, 50.0], [130, 70.0, 58.0], [180, 85.0, 65.0]],
            }
        ]
    }

    remapped = warper.remap_vision_coordinates(
        tracks_per_frame=mock_tracks,
        cause_frame_idx=80,
        effect_frame_idx=180,
        freeze_frames=sol["freeze_duration_frames"],
        out_frame_to_source_time_sec=out_to_src,
        source_frame_to_out_frame=src_to_out,
        cause_telestration=mock_tel,
    )

    warped_tracks = remapped["warped_tracks_per_frame"]
    assert len(warped_tracks) == sol["total_output_frames"], "Tracks not remapped for every output frame!"
    remapped_vectors = remapped["warped_cause_telestration"]["vectors"]
    assert remapped_vectors[0]["pass_start_frame"] == f_out
    assert remapped["warped_cause_frame_idx"] == sol["freeze_start_frame"]

    # Verify ball carrier per frame remapping
    remapped_bc = remapped["warped_cause_telestration"]["ball_carrier_per_frame"]
    assert str(f_out) in remapped_bc, "ball_carrier_per_frame not remapped to output timeline!"
    assert remapped_bc[str(f_out)] == 7, "Carrier at freeze frame must match track 7"

    print(f"Remapped pass vector start frame: {remapped_vectors[0]['pass_start_frame']}")
    print(f"Remapped pass vector receive frame: {remapped_vectors[0]['pass_receive_frame']}")
    print("Test 3 PASSED!")


def test_real_video_prewarper():
    print("\n--- Test 4: FFmpeg Frame-Accurate Pre-Warper on Real Footage ---")
    warper = ElasticTimeWarper(fps=30)
    real_clip = "F:/DockerRenders/assets/clips/clip_01_left_wing.mp4"

    if not os.path.exists(real_clip):
        print(f"Notice: {real_clip} not found, generating temporary test clip...")
        real_clip = "temp_synthetic_source.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(real_clip, fourcc, 30.0, (1920, 1080))
        for i in range(180):
            f = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.putText(f, f"Test {i}", (120, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
            writer.write(f)
        writer.release()

    beats = {
        "beat_1_setup": {"duration_ms": 2000},
        "beat_2_cause": {"duration_ms": 5000},
        "beat_3_effect": {"duration_ms": 3000},
    }
    sol = warper.solve_elastic_timeline(beats, source_video_duration_sec=6.0, cause_frame_idx=80)

    out_clip = "test_warped_output.mp4"
    if os.path.exists(out_clip):
        os.remove(out_clip)

    warper.render_warped_video_stream(
        source_video_path=real_clip,
        output_video_path=out_clip,
        out_frame_to_source_time_sec=sol["out_frame_to_source_time_sec"],
        instantaneous_rates=sol["instantaneous_rates"],
        width=1920,
        height=1080,
        fps=30,
    )

    assert os.path.exists(out_clip), "Output warped video was not created!"
    cap = cv2.VideoCapture(out_clip)
    actual_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Baked video frames: {actual_f} at {w}x{h}")
    expected_f = sol["total_output_frames"]
    diff = abs(actual_f - expected_f)
    print(f"Diff between baked video ({actual_f}) and master voiceover frames ({expected_f}): {diff}")
    assert diff <= 1, f"Frame sync violation: diff={diff} exceeds 1 frame threshold!"

    # Clean up test output
    if os.path.exists(out_clip):
        os.remove(out_clip)
    if os.path.exists("temp_synthetic_source.mp4"):
        os.remove("temp_synthetic_source.mp4")

    print("Test 4 PASSED!")


def test_sfx_assets():
    print("\n--- Test 5: Procedural SFX Asset Verification ---")
    swell_data = synthesize_slow_focus_swell()
    assert swell_data.ndim == 2 and swell_data.shape[1] == 2, "Swell must be stereo PCM16!"
    assert swell_data.dtype == np.int16, "Swell must be int16!"
    assert len(swell_data) > 0, "Swell must have positive samples!"

    # Verify deployed SFX files
    sfx_paths = [
        Path("F:/DockerRenders/assets/audio/sfx/slow_focus_swell.wav"),
        Path("C:/Users/mlkme/Desktop/cursor-mikow/remotion-server/public/assets/audio/sfx/slow_focus_swell.wav"),
    ]
    for p in sfx_paths:
        if p.parent.exists():
            assert p.is_file(), f"Expected SFX file not found at {p}"
            sr, data = wavfile.read(str(p))
            assert sr == 48000, f"Expected 48000Hz, got {sr}Hz"
            print(f"Verified SFX at {p}: {len(data)/sr:.2f}s, {data.shape[1]}ch, {sr}Hz")

    print("Test 5 PASSED!")


def main():
    print("================================================================")
    print("SUBTLE ELASTIC SPEED-RAMPING ENGINE: COMPREHENSIVE VERIFICATION")
    print("================================================================")
    test_storyboard_schema()
    test_elastic_solver_and_easing()
    test_bidirectional_lookup_and_remapping()
    test_real_video_prewarper()
    test_sfx_assets()
    print("\n================================================================")
    print("ALL 5 TESTS PASSED SUCCESSFULLY! 100% VERIFIED.")
    print("================================================================")


if __name__ == "__main__":
    main()
