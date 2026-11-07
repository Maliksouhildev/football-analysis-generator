"""
CLI Evaluation Script for Two-Pass Broadcast Video Tracking.
Tests Shot Detection, BoT-SORT with CMC, Kit K-Means (k=2 + Outliers), and Pass-2 Smart Montage Batching.
"""

import argparse
from datetime import datetime
import os
import sys

from src.detection_engine.tracking.video_pipeline import VideoTrackingPipeline


def main():
    parser = argparse.ArgumentParser(description="Test Two-Pass Video Tracking Engine on broadcast football footage.")
    parser.add_argument("--video", type=str, default="data/test_clips/clip_evt_min10_600_0.mp4", help="Path to test MP4 video")
    parser.add_argument("--max-frames", type=int, default=120, help="Maximum number of frames to process")
    parser.add_argument("--stride", type=int, default=2, help="Frame striding factor (default 2 = 15 FPS)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path (yolov8n.pt or yolov8x.pt)")
    parser.add_argument("--roster", type=str, default="test_data/canonical_el_clasico_roster.json", help="Path to matchday roster JSON")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[Error] Video file not found: {args.video}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or f"debug_outputs/tracking_run_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("STARTING TWO-PASS BROADCAST VIDEO TRACKING EVALUATION")
    print(f"Video: {args.video}")
    print(f"Max Frames: {args.max_frames} | Stride: {args.stride} | Model: {args.model}")
    print(f"Output Directory: {out_dir}")
    print("=" * 70)

    pipeline = VideoTrackingPipeline(
        yolo_model_path=args.model,
        stride=args.stride,
        device="cpu",
        canonical_roster_path=args.roster,
    )

    res = pipeline.process_video(
        video_path=args.video,
        output_dir=out_dir,
        render_annotated_video=True,
        max_frames=args.max_frames,
    )

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY RESULTS")
    print(f"Processed Frames:       {res.total_frames}")
    print(f"Elapsed Time:           {res.elapsed_time_sec:.2f} s")
    print(f"Effective Processing:   {res.effective_fps:.2f} FPS (Target >= 15 FPS)")
    print(f"Camera Shots Detected:  {res.shots_count} (Tactical: {res.tactical_shots_count})")
    print(f"Identified Players:     {res.identified_players_count}")
    print(f"Summary JSON:           {res.diagnostics_json_path}")
    if res.output_video_path:
        print(f"Tracked Video:          {res.output_video_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
