"""
test_detection_loop.py
================================================================================
Automated Random Frame Sampling & Visual Evaluation Harness.

Accepts either:
1. A raw match video file (.mp4) -> Stratified/random sampling across 1st & 2nd half
2. A directory of raw match screenshots -> Iterates through test frames

Feeds each frame + matchday roster JSON into the isolated PlayerIDPipeline.
Outputs to debug_outputs/run_<timestamp>/frame_<idx>/:
- annotated_result.png
- turf_poly_debug.png
- clahe_crops.png
- diagnostics.json

Generates master:
- summary_report.json
- summary_grid.png
================================================================================
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dotenv
dotenv.load_dotenv(PROJECT_ROOT / ".env")

from src.detection_engine import (
    CANONICAL_CLASICO_ROSTER,
    DetectionResult,
    PipelineConfig,
    PlayerIDPipeline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_detection_loop")


def parse_args():
    parser = argparse.ArgumentParser(description="Automated Player Detection & Identity Evaluation Harness")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to an input directory of screenshots OR path to a raw match video (.mp4)",
    )
    parser.add_argument(
        "--match-data",
        type=str,
        default="test_data/canonical_el_clasico_roster.json",
        help="Path to canonical matchday roster JSON",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of stratified frames to sample if input is a video file",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=12,
        help="Maximum number of screenshots to process if input is a directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output run directory (defaults to debug_outputs/run_<timestamp>)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLOv8 inference image size (default: 1280)",
    )
    return parser.parse_args()


def sample_frames_from_video(
    video_path: Path, num_samples: int
) -> List[Tuple[str, np.ndarray, float, Optional[int]]]:
    """
    Samples `num_samples` frames stratified across 1st half and 2nd half of match video.
    Returns list of (frame_id, frame_bgr, pts_seconds, period_hint).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    logger.info(f"Loaded video: {video_path.name} | Duration: {duration_sec/60:.1f} mins ({total_frames} frames @ {fps:.1f} fps)")

    # Stratified target timestamps: distribute across available duration
    # If video is full 90m (or short highlight reel), sample proportionally
    sample_pts: List[float] = []
    step = duration_sec / (num_samples + 1)
    for i in range(1, num_samples + 1):
        # Add slight jitter to avoid exact interval boundaries
        jitter = random.uniform(-0.15 * step, 0.15 * step)
        t_target = max(2.0, min(duration_sec - 2.0, (i * step) + jitter))
        sample_pts.append(t_target)

    sampled: List[Tuple[str, np.ndarray, float, Optional[int]]] = []
    for idx, pts in enumerate(sample_pts, start=1):
        target_f = int(pts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning(f"Failed to read frame at {pts:.1f}s")
            continue
        fid = f"vid_sample_{idx:02d}_{int(pts)}s"
        period = 1 if pts < (duration_sec * 0.5) else 2
        sampled.append((fid, frame, pts, period))

    cap.release()
    logger.info(f"Successfully extracted {len(sampled)} stratified frames from video.")
    return sampled


def load_frames_from_directory(
    dir_path: Path, max_frames: int
) -> List[Tuple[str, np.ndarray, Optional[float], Optional[int]]]:
    """
    Loads image files from a directory, sorted and subsampled if needed.
    Returns list of (frame_id, frame_bgr, pts_seconds_or_None, period_hint_or_None).
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    # Filter out cropsheets or already annotated debug images if in test dir
    img_files = [p for p in img_files if not any(k in p.name.lower() for k in ["_annotated", "_cropsheet", "_debug", "_mask"])]
    img_files.sort(key=lambda p: p.name)

    if not img_files:
        raise RuntimeError(f"No valid image files found in: {dir_path}")

    # Subsample if more than max_frames
    if len(img_files) > max_frames:
        indices = np.linspace(0, len(img_files) - 1, max_frames, dtype=int)
        selected_files = [img_files[i] for i in indices]
    else:
        selected_files = img_files

    loaded: List[Tuple[str, np.ndarray, Optional[float], Optional[int]]] = []
    for f in selected_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        # Infer period or minute from filename if possible
        fn = f.stem.lower()
        pts = None
        period = None
        if "min" in fn:
            import re
            m = re.search(r"min(\d+)", fn)
            if m:
                m_int = int(m.group(1))
                pts = float(m_int * 60)
                period = 1 if m_int <= 45 else 2
        elif "62" in fn or "61" in fn:
            pts = 62.0 * 60.0
            period = 2
        loaded.append((f.stem, img, pts, period))

    logger.info(f"Loaded {len(loaded)} test frames from directory: {dir_path}")
    return loaded


def build_summary_grid(
    annotated_images: List[Tuple[str, np.ndarray, Dict[str, Any]]],
    output_path: Path,
) -> None:
    """
    Generates a master summary grid montage (summary_grid.png) displaying all
    annotated frames side-by-side with overlaid diagnostic headers.
    """
    n = len(annotated_images)
    if n == 0:
        return

    # Choose grid dimensions (cols x rows)
    if n <= 2:
        cols, rows = n, 1
    elif n <= 4:
        cols, rows = 2, 2
    elif n <= 6:
        cols, rows = 3, 2
    elif n <= 9:
        cols, rows = 3, 3
    elif n <= 12:
        cols, rows = 4, 3
    else:
        cols = 4
        rows = math.ceil(n / 4)

    target_thumb_w = 640
    target_thumb_h = 360
    header_h = 38

    cell_w = target_thumb_w
    cell_h = target_thumb_h + header_h

    grid_img = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    grid_img[:] = (20, 24, 32)  # Dark slate background

    for idx, (fid, img, meta) in enumerate(annotated_images):
        r = idx // cols
        c = idx % cols
        if r >= rows:
            break

        x_cell = c * cell_w
        y_cell = r * cell_h

        # Header bar
        cv2.rectangle(
            grid_img,
            (x_cell, y_cell),
            (x_cell + cell_w, y_cell + header_h),
            (14, 18, 26),
            -1,
        )
        clock_str = meta.get("clock_str", "00:00")
        actors_cnt = meta.get("actors_count", 0)
        shot = meta.get("shot_type", "tactical")
        hud_txt = f"[{idx+1}/{n}] {fid[:22]} | Clock: {clock_str} | Shot: {shot} | Players: {actors_cnt}"
        cv2.putText(
            grid_img,
            hud_txt,
            (x_cell + 10, y_cell + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (240, 245, 250),
            1,
            cv2.LINE_AA,
        )

        # Thumbnail image
        thumb = cv2.resize(img, (target_thumb_w, target_thumb_h), interpolation=cv2.INTER_AREA)
        grid_img[y_cell + header_h : y_cell + cell_h, x_cell : x_cell + cell_w] = thumb

        # Thin grid separator border
        cv2.rectangle(grid_img, (x_cell, y_cell), (x_cell + cell_w, y_cell + cell_h), (50, 60, 75), 1)

    cv2.imwrite(str(output_path), grid_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    logger.info(f"Master summary grid saved: {output_path}")


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    # Initialize Output Run Directory
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = PROJECT_ROOT / "debug_outputs" / f"run_{timestamp_str}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run Output Directory: {run_dir}")

    # Load Match Data
    match_data_path = Path(args.match_data)
    if not match_data_path.is_file():
        match_data_path = PROJECT_ROOT / args.match_data
    if match_data_path.is_file():
        logger.info(f"Loading Matchday Roster JSON: {match_data_path}")
        with open(match_data_path, "r", encoding="utf-8") as f:
            match_metadata = json.load(f)
    else:
        logger.warning(f"Roster file {match_data_path} not found. Using canonical Clasico roster.")
        match_metadata = CANONICAL_CLASICO_ROSTER

    # Initialize Isolated Detection Pipeline
    config = PipelineConfig(
        iteration_id=12,
        name="Evaluation Harness Pipeline",
        yolo_imgsz=args.imgsz,
        use_adaptive_turf_hull=True,
        use_broadcast_banner_mask=True,
        use_touchline_staff_filter=True,
        use_tiled_highres_yolo=True,
        use_scoreboard_clock_ocr=True,
        use_live_active_xi_sync=True,
        use_homography_topology_priors=True,
        use_superres_clahe_torso_zoom=True,
        use_biometric_visual_db=True,
        use_hungarian_global_solver=True,
        use_vlm_critic_audit=True,
    )
    pipeline = PlayerIDPipeline(config=config, match_metadata=match_metadata)

    # Determine Input Mode: Video or Directory
    if input_path.is_file() and input_path.suffix.lower() in [".mp4", ".mkv", ".avi", ".mov"]:
        frames = sample_frames_from_video(input_path, args.num_samples)
    elif input_path.is_dir():
        frames = load_frames_from_directory(input_path, args.max_frames)
    elif input_path.is_file():
        # Single image file
        img = cv2.imread(str(input_path))
        if img is None:
            logger.error(f"Cannot read image file: {input_path}")
            sys.exit(1)
        frames = [(input_path.stem, img, None, None)]
    else:
        logger.error(f"Unsupported input: {input_path}")
        sys.exit(1)

    logger.info(f"--- Starting Detection Loop Evaluation across {len(frames)} frames ---")

    report_entries = []
    grid_items: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []

    for idx, (fid, frame_bgr, pts_sec, p_hint) in enumerate(frames, start=1):
        frame_dir = run_dir / f"frame_{idx:03d}_{fid}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{idx}/{len(frames)}] Processing frame '{fid}'...")
        t_start = time.time()

        res: DetectionResult = pipeline.process_frame(
            frame_input=frame_bgr,
            match_metadata=match_metadata,
            fallback_pts_seconds=pts_sec,
            clip_id=fid,
            period_hint=p_hint,
        )
        elapsed = time.time() - t_start

        # Save artifacts: annotated_result.png, turf_poly_debug.png, clahe_crops.png, diagnostics.json
        saved_paths = res.save_debug_outputs(frame_dir)

        # Log brief summary
        actors = res.actors
        home_players = [a for a in actors if a.get("team") == "home"]
        away_players = [a for a in actors if a.get("team") == "away"]
        ref_actors = [a for a in actors if a.get("team") == "referee"]

        critics_cnt = len(res.diagnostics.get("critic_overrides", []))
        rejected_cnt = res.diagnostics.get("candidate_filtering", {}).get("rejected_count", 0)

        logger.info(
            f"  Done in {elapsed:.2f}s | Clock: {res.clock_str} | Actors: {len(actors)} "
            f"({len(home_players)} Home, {len(away_players)} Away, {len(ref_actors)} Ref) | "
            f"Critic Overrides: {critics_cnt} | Rejected Boxes: {rejected_cnt}"
        )
        for a in actors[:5]:
            logger.info(
                f"    Track #{a.get('track_id')}: #{a.get('assigned_number')} {a.get('assigned_name')} "
                f"({a.get('assigned_role')}) conf={a.get('assignment_confidence'):.2f} [{a.get('assignment_method')}]"
            )

        frame_meta = {
            "frame_idx": idx,
            "frame_id": fid,
            "pts_seconds": pts_sec,
            "clock_str": res.clock_str,
            "match_minute": res.match_minute,
            "match_second": res.match_second,
            "shot_type": res.shot_type,
            "actors_count": len(actors),
            "home_count": len(home_players),
            "away_count": len(away_players),
            "critic_overrides_count": critics_cnt,
            "rejected_candidates_count": rejected_cnt,
            "elapsed_sec": round(elapsed, 2),
            "files": {k: str(p.relative_to(run_dir)) for k, p in saved_paths.items()},
            "actors": [
                {
                    "track_id": a.get("track_id"),
                    "name": a.get("assigned_name"),
                    "number": a.get("assigned_number"),
                    "role": a.get("assigned_role"),
                    "team": a.get("team"),
                    "confidence": a.get("assignment_confidence"),
                    "method": a.get("assignment_method"),
                    "coords_pct": (a.get("x_pct"), a.get("y_pct")),
                }
                for a in actors
            ],
        }
        report_entries.append(frame_meta)

        if res.annotated_frame is not None and res.annotated_frame.size > 0:
            grid_items.append((fid, res.annotated_frame, frame_meta))

    # Master Summary Report
    summary_report = {
        "run_timestamp": timestamp_str,
        "input_source": str(input_path),
        "total_frames_evaluated": len(frames),
        "total_actors_detected": sum(f["actors_count"] for f in report_entries),
        "average_elapsed_sec": round(sum(f["elapsed_sec"] for f in report_entries) / max(1, len(report_entries)), 2),
        "frames": report_entries,
    }

    report_path = run_dir / "summary_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    logger.info(f"Master summary report saved: {report_path}")

    # Master Summary Grid
    grid_path = run_dir / "summary_grid.png"
    build_summary_grid(grid_items, grid_path)

    logger.info("================================================================================")
    logger.info(f"Evaluation Loop Completed Successfully!")
    logger.info(f"Summary Report: {report_path}")
    logger.info(f"Summary Grid:   {grid_path}")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
