"""
scripts/verify_tactical_clip_scorer.py
Quick standalone verification script for TacticalClipScorer on a sample video.
"""

import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from scipy.io import wavfile

from src.ingestion.clip_extractor import ClipExtractor
from src.ingestion.time_indexer import TimeIndexer
from src.vision.tactical_clip_scorer import TacticalClipScorer


def create_sample_match_video(output_path: str, duration_sec: float = 25.0, fps: int = 30) -> None:
    """Creates a synthetic football video with moving players, pitch markings, and authentic crowd audio peak."""
    w, h = 1280, 720
    n_frames = int(duration_sec * fps)
    temp_avi = output_path.replace(".mp4", "_raw.avi")
    temp_wav = output_path.replace(".mp4", "_audio.wav")

    # 1. Generate video frames
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(temp_avi, fourcc, fps, (w, h))

    for idx in range(n_frames):
        t = idx / float(fps)
        # Tactical pitch green
        frame = np.full((h, w, 3), (46, 125, 50), dtype=np.uint8)

        # White pitch markings
        cv2.rectangle(frame, (60, 40), (w - 60, h - 40), (240, 240, 240), 3)
        cv2.line(frame, (w // 2, 40), (w // 2, h - 40), (240, 240, 240), 3)
        cv2.circle(frame, (w // 2, h // 2), 80, (240, 240, 240), 3)

        # Moving ball and players progression towards goal (X: 300 -> 1100)
        progression = min(1.0, t / duration_sec)
        ball_x = int(350 + progression * 750)
        ball_y = int(360 + 40 * math.sin(t * 3.0))

        # Attacking players
        cv2.circle(frame, (ball_x, ball_y), 8, (255, 255, 255), -1)  # Football
        cv2.circle(frame, (ball_x - 30, ball_y), 15, (255, 255, 255), -1)  # Ball carrier (White kit)
        cv2.circle(frame, (ball_x - 90, ball_y - 60), 15, (255, 255, 255), -1)
        cv2.circle(frame, (ball_x + 80, ball_y + 80), 15, (255, 255, 255), -1)  # Running winger

        # Defending players (Yellow / Red kit)
        cv2.circle(frame, (600, 320), 15, (0, 0, 220), -1)
        cv2.circle(frame, (750, 400), 15, (0, 0, 220), -1)
        cv2.circle(frame, (900, 360), 15, (0, 0, 220), -1)

        out.write(frame)
    out.release()

    # 2. Generate 16kHz audio with background stadium noise and climax roar around t = 18.0s
    sr = 16000
    n_audio = int(sr * duration_sec)
    t_arr = np.linspace(0, duration_sec, n_audio, endpoint=False)
    np.random.seed(42)
    # Background crowd murmur
    audio = np.random.normal(0.0, 0.04, n_audio).astype(np.float32)

    # Climax crowd roar + commentator excitement at t = [17.0s .. 19.5s]
    peak_mask = (t_arr >= 17.0) & (t_arr <= 19.5)
    audio += (peak_mask * np.random.normal(0.0, 0.45, n_audio)).astype(np.float32)
    # Commentator vocal fundamental/formant frequency in 450Hz-2400Hz (1100Hz tone)
    audio += (peak_mask * 0.40 * np.sin(2.0 * math.pi * 1100.0 * t_arr)).astype(np.float32)

    # Export to WAV
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(temp_wav, sr, audio_int16)

    # 3. Combine with FFmpeg into final MP4
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_avi,
        "-i", temp_wav,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
        "-c:a", "aac",
        output_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Cleanup temporary files
    for tmp in [temp_avi, temp_wav]:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def main():
    print("=== TacticalClipScorer Verification Script ===")
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample_video = os.path.join(tmp_dir, "sample_football_match.mp4")
        print(f"1. Generating 25s sample match video with 18.0s excitement peak at {sample_video}...")
        create_sample_match_video(sample_video, duration_sec=25.0, fps=30)
        print("   Sample video generated successfully.")

        print("2. Running TacticalClipScorer on sample video...")
        scorer = TacticalClipScorer()
        peaks = scorer.detect_audio_peaks(sample_video)
        print(f"   Detected {len(peaks)} audio excitement peak(s):")
        for pk in peaks:
            print(f"   -> t_peak={pk.t_peak:.2f}s, combined_score={pk.audio_peak:.3f}, z_score={pk.z_score:.2f}")

        ranked = scorer.score_video(sample_video, top_k=3, attack_direction="right")
        print(f"\n3. Scored {len(ranked)} buildup candidate(s):")
        for c in ranked:
            print(f"   Rank #{c.rank}: TotalScore={c.total_score:.4f}")
            print(f"     * t_peak: {c.t_peak:.2f}s")
            print(f"     * Buildup Look-Back Window: [{c.buildup_start_sec:.2f}s .. {c.buildup_end_sec:.2f}s] (dur={c.buildup_duration_sec:.2f}s)")
            print(f"     * Climax Hook Window: [{c.climax_hook_window['start_sec']:.2f}s .. {c.climax_hook_window['end_sec']:.2f}s] (dur={c.climax_hook_window['duration_sec']:.2f}s)")
            print(f"     * Expected Threat (delta_xT): +{c.delta_xT:.4f}")
            print(f"     * Defenders Bypassed: {c.defenders_bypassed}")
            print(f"     * Audio Peak: {c.audio_peak:.3f}")
            print(f"     * Max Unmarked Space: {c.max_unmarked_space_m2:.1f} m²")

        print("\n4. Testing ClipExtractor end-to-end integration...")
        extractor = ClipExtractor(
            time_indexer=TimeIndexer(),
            media_source=sample_video,
            output_dir=os.path.join(tmp_dir, "clips"),
            frames_dir=os.path.join(tmp_dir, "frames"),
        )
        extracted = extractor.score_and_extract_buildup_clips(
            media_source=sample_video,
            top_k=1,
            cut_video_clips=True,
        )
        if extracted:
            top_clip = extracted[0]
            print("   Top clip extracted successfully:")
            print(f"     * Buildup Clip: {top_clip['buildup_clip_path']} (exists={os.path.exists(top_clip['buildup_clip_path'])})")
            print(f"     * Climax Hook Clip: {top_clip['climax_hook_path']} (exists={os.path.exists(top_clip['climax_hook_path'])})")
            print(f"     * Freeze Frame: {top_clip['freeze_frame_path']} (exists={os.path.exists(top_clip['freeze_frame_path'])})")

    print("\n=== Verification Completed Successfully! ===")


if __name__ == "__main__":
    main()
