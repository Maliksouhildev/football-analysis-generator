#!/usr/bin/env python3
"""
video_pipeline.py
Production Two-Pass Automated Football Video Player Identification CLI.

Architecture:
- Pass 1 (Local CPU-Optimized Tracking & Clustering):
  Local detection (CVPlayerDetector / BoT-SORT / CMC), frame striding, torso RGB/HSV kit clustering,
  generating labeled tracking bounding boxes and 3-crop temporal strips (Early, Peak, Late).
- Pass 2 (VLM Grounding + Geometry + Human-Lock Safety Valve):
  Injects GEMINI_PLAYER_IDENTIFICATION_SYSTEM_PROMPT with:
  1. Hard Team Firewall (HOME white -> Real Madrid only, AWAY purple/blue -> Barcelona only).
  2. Barcelona Right-to-Left attack depth & flank inversion geometry.
  3. Time-Aware strict 11-player Active Squad whitelists (no bench players).
  4. Instant --overrides safety valve (100% confidence, zero duplicates).

Usage:
  python video_pipeline.py --input match.mp4 --overrides '{"T18": 9, "T12": 11}'
  python video_pipeline.py --input frame.jpg --home "Real Madrid" --away "Barcelona" --minute 46
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.player_identity_resolver import PlayerIdentityResolver
from src.vision.player_detector import CVPlayerDetector
from src.vision.tactical_vision_engine import TacticalVisionEngine
from src.vision.match_roster_engine import ActiveXIResolver
from src.detection_engine.tracking.video_pipeline import VideoTrackingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("video_pipeline")


def load_yolo_model():
    """Loads local YOLO model from disk for robust player bounding box detection."""
    candidates = [
        "F:/DockerRenders/yolov8n.pt",
        "yolov8n.pt",
        Path.home() / ".cache/ultralytics/yolov8n.pt",
    ]
    for c in candidates:
        if Path(c).is_file():
            try:
                from ultralytics import YOLO
                return YOLO(str(c))
            except Exception:
                pass
    try:
        from ultralytics import YOLO
        return YOLO("yolov8n.pt")
    except Exception:
        return None


def run_pass1_local_detection(
    video_or_image_path: Path,
    target_frame_idx: int = 66,
    stride: int = 15,
    force_frame: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Optional[np.ndarray]], int]:
    """
    Pass 1: Runs locally on CPU without lag:
    - Extracts freeze frame, prev frame (-stride), and next frame (+stride).
    - If video has a close-up/celebration cut at target_frame_idx, scans for the wide tactical pitch frame.
    - Uses YOLOv8/CV to detect real human players on the pitch.
    - Clusters jersey colors into HOME, AWAY, GK, REF.
    - NEVER generates fake/synthetic boxes on empty grass.
    """
    logger.info("Running Pass 1 (Local CPU Detection & Jersey Clustering)...")

    temporal_burst: Dict[str, Optional[np.ndarray]] = {"prev": None, "freeze": None, "next_1": None}
    chosen_frame_idx = target_frame_idx
    yolo_model = load_yolo_model()

    # Case A: Video Input
    if video_or_image_path.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov", ".ts"):
        cap = cv2.VideoCapture(str(video_or_image_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_or_image_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        f_idx = min(max(0, target_frame_idx), max(0, total_frames - 1))

        # Check if target frame is a close-up/celebration or has low player count
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame_cand = cap.read()
        probe_count = 0
        if ret and frame_cand is not None:
            fh_c = frame_cand.shape[0]
            if yolo_model:
                res_p = yolo_model.predict(frame_cand, classes=[0], conf=0.30, device="cpu", verbose=False)[0]
                valid_p = [b for b in res_p.boxes if 0.03 <= (b.xyxy[0][3] - b.xyxy[0][1]).item() / fh_c <= 0.28]
                probe_count = len(valid_p)
            else:
                probe_count = len(CVPlayerDetector.detect_players_on_pitch(frame_cand))

        # If probe found < 5 tactical players, scan for the true wide tactical frame
        if probe_count < 5 and total_frames > 30 and not force_frame:
            logger.info("Target frame %d is a close-up/celebration shot (%d players). Scanning video for wide tactical pitch frame...", f_idx, probe_count)
            best_f_idx = f_idx
            best_count = probe_count
            scan_step = max(15, total_frames // 40)
            for cand_idx in range(0, total_frames, scan_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, cand_idx)
                ret_s, f_s = cap.read()
                if not ret_s or f_s is None:
                    continue
                sh = f_s.shape[0]
                s_count = 0
                if yolo_model:
                    res_s = yolo_model.predict(f_s, classes=[0], conf=0.30, device="cpu", verbose=False)[0]
                    v_s = [b for b in res_s.boxes if 0.03 <= (b.xyxy[0][3] - b.xyxy[0][1]).item() / sh <= 0.28]
                    s_count = len(v_s)
                else:
                    s_count = len(CVPlayerDetector.detect_players_on_pitch(f_s))

                if s_count > best_count:
                    best_count = s_count
                    best_f_idx = cand_idx
                if s_count >= 16:
                    break

            logger.info("Selected authentic wide tactical frame %d (%d on-pitch players).", best_f_idx, best_count)
            f_idx = best_f_idx

        chosen_frame_idx = f_idx

        # 1. Read freeze frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame_freeze = cap.read()
        if not ret or frame_freeze is None:
            cap.release()
            raise RuntimeError(f"Could not read frame {f_idx} from video {video_or_image_path}")

        temporal_burst["freeze"] = frame_freeze

        # 2. Read previous frame (-stride)
        prev_idx = max(0, f_idx - stride)
        cap.set(cv2.CAP_PROP_POS_FRAMES, prev_idx)
        ret_prev, frame_prev = cap.read()
        if ret_prev and frame_prev is not None:
            temporal_burst["prev"] = frame_prev

        # 3. Read next frame (+stride)
        next_idx = min(total_frames - 1, f_idx + stride)
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_idx)
        ret_next, frame_next = cap.read()
        if ret_next and frame_next is not None:
            temporal_burst["next_1"] = frame_next

        cap.release()
        frame = frame_freeze

    # Case B: Image Input
    else:
        frame = cv2.imread(str(video_or_image_path))
        if frame is None or frame.size == 0:
            raise FileNotFoundError(f"Could not load image: {video_or_image_path}")
        temporal_burst["freeze"] = frame

    fh, fw = frame.shape[:2]
    tracks: List[Dict[str, Any]] = []

    # Detect real players using YOLO
    if yolo_model and frame is not None:
        res = yolo_model.predict(frame, classes=[0], conf=0.30, device="cpu", verbose=False)[0]
        boxes_list = []
        for b in res.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            bh = y2 - y1
            # Require on-pitch tactical scale (3% to 30% of screen height)
            if 0.03 <= bh / fh <= 0.30:
                boxes_list.append((x1, y1, x2, y2, float(b.conf[0])))

        # Sort left-to-right for consistent track labeling
        boxes_list.sort(key=lambda item: item[0])

        for idx, (x1, y1, x2, y2, conf) in enumerate(boxes_list):
            tid = idx + 1
            x_boot = (x1 + x2) / 2.0
            y_boot = float(y2)
            tracks.append({
                "track_id": tid,
                "bbox": [y1, x1, y2, x2],
                "x_pct": round(x_boot / fw * 100.0, 1),
                "y_pct": round(y_boot / fh * 100.0, 1),
                "team": "unknown",
                "jersey_class": "UNKNOWN",
                "confidence": conf,
            })
    else:
        raw_detections = CVPlayerDetector.detect_players_on_pitch(frame)
        if raw_detections:
            for idx, det in enumerate(raw_detections):
                tid = idx + 1
                w_px = det.w_px
                h_px = det.h_px
                cx = (det.boot_x / 100.0) * fw
                cy = (det.boot_y / 100.0) * fh
                xmin = max(0, int(cx - w_px / 2.0))
                xmax = min(fw, int(cx + w_px / 2.0))
                ymin = max(0, int(cy - h_px))
                ymax = min(fh, int(cy))

                team_str = det.team_hint
                jc = "TEAM_HOME" if team_str == "home" else ("TEAM_AWAY" if team_str == "away" else "REFEREE")
                tracks.append({
                    "track_id": tid,
                    "bbox": [ymin, xmin, ymax, xmax],
                    "x_pct": det.boot_x,
                    "y_pct": det.boot_y,
                    "team": team_str,
                    "jersey_class": jc,
                })

    if not tracks:
        logger.warning("Zero real on-pitch players detected in this frame. No synthetic boxes will be generated.")
        return frame, [], temporal_burst, chosen_frame_idx

    # Classify jersey colors using multi-cue torso clustering
    try:
        TacticalVisionEngine._classify_all_tracks_jersey(frame, tracks, fw, fh)
    except Exception as e:
        logger.debug("Jersey clustering note: %s", e)

    logger.info("Pass 1 Complete: %d real on-pitch tracks localized and clustered into team boundaries.", len(tracks))
    return frame, tracks, temporal_burst, chosen_frame_idx


def main():
    parser = argparse.ArgumentParser(
        description="Two-Pass Automated Football Video Player Identification & Tactical Grounding Pipeline."
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input match video (.mp4/.mkv) or image frame.")
    parser.add_argument("--overrides", "-o", type=str, default=None, help='JSON overrides string e.g. \'{"T18": 9, "T12": 11}\' or \'{"T18": "Lewandowski"}\'.')
    parser.add_argument("--home", type=str, default="Real Madrid", help="Home Team Name (default: Real Madrid).")
    parser.add_argument("--away", type=str, default="Barcelona", help="Away Team Name (default: Barcelona).")
    parser.add_argument("--minute", "-m", type=int, default=46, help="Match Minute (default: 46).")
    parser.add_argument("--period", "-p", type=int, default=None, help="Match Period (1 or 2). Defaults to 2 if minute > 45.")
    parser.add_argument("--frame-idx", type=int, default=66, help="Video frame index to analyze (default: 66).")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for output JSON and images.")
    parser.add_argument("--api-key", type=str, default=None, help="Gemini API Key (optional, defaults to GEMINI_API_KEY env).")
    parser.add_argument("--no-vlm", action="store_true", help="Disable VLM and use local Hungarian solver only.")
    parser.add_argument("--force-frame", action="store_true", help="Force exact frame index without scanning for wide tactical view.")
    parser.add_argument("--continuous", "-c", action="store_true", help="Run full continuous video tracking across all frames using BoT-SORT + SmartCollageBatcher.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process in continuous mode.")
    parser.add_argument("--stride", type=int, default=2, help="Frame stride for detection in continuous mode (default: 2).")
    parser.add_argument("--render-video", action="store_true", default=True, help="Render tracked output MP4 video with tactical HUD badges.")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    period = args.period if args.period is not None else (2 if args.minute > 45 else 1)
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    use_vlm = not args.no_vlm and bool(api_key)

    # -------------------------------------------------------------------------
    # CONTINUOUS VIDEO MODE: BoT-SORT + SmartCollageBatcher + VLM Caching
    # -------------------------------------------------------------------------
    if args.continuous or (input_path.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov", ".ts") and not args.force_frame and args.frame_idx == 66 and not args.overrides):
        logger.info("Executing Continuous Video Tracking Pipeline...")
        yolo_path = "yolov8x.pt" if Path("yolov8x.pt").is_file() else "yolov8n.pt"
        roster_path = str(PROJECT_ROOT / "test_data" / "canonical_el_clasico_roster.json")

        pipeline = VideoTrackingPipeline(
            yolo_model_path=yolo_path,
            stride=args.stride,
            conf_thresh=0.25,
            device="cpu",
            home_team_name=args.home,
            away_team_name=args.away,
            canonical_roster_path=roster_path if Path(roster_path).is_file() else None,
            period=period,
            match_minute=args.minute,
        )

        tracking_res = pipeline.process_video(
            video_path=str(input_path),
            output_dir=str(output_dir),
            render_annotated_video=args.render_video,
            max_frames=args.max_frames,
        )

        print("\n" + "=" * 78)
        print(f" CONTINUOUS TRACKING COMPLETE: {args.home.upper()} vs {args.away.upper()} ({args.minute}')")
        print("=" * 78)
        print(f" Total Frames:     {tracking_res.total_frames}")
        print(f" Shots Detected:   {tracking_res.shots_count} ({tracking_res.tactical_shots_count} tactical)")
        print(f" Unique Tracks:    {tracking_res.unique_tracklets_count}")
        print(f" Output Video:     {tracking_res.output_video_path}")
        print(f" Diagnostics JSON: {tracking_res.diagnostics_json_path}")
        print("=" * 78 + "\n")
        return 0

    # -------------------------------------------------------------------------
    # PASS 1: Local CPU Detection & Team Clustering
    # -------------------------------------------------------------------------
    frame, tracks, temporal_burst, chosen_frame_idx = run_pass1_local_detection(
        video_or_image_path=input_path,
        target_frame_idx=args.frame_idx,
        force_frame=args.force_frame,
    )

    if not tracks:
        print("\n" + "=" * 78)
        print(f" [NOTICE] Zero on-pitch players detected in frame {chosen_frame_idx} (close-up/celebration view).")
        print(" Zero synthetic boxes were generated. Pass a wide tactical frame or video with match play.")
        print("=" * 78 + "\n")
        return 0

    # -------------------------------------------------------------------------
    # PASS 2: VLM Verification + Geometry + Safety Valve Overrides
    # -------------------------------------------------------------------------
    logger.info("Running Pass 2 (Player Identity Resolution & Human-Lock Overrides)...")
    resolver = PlayerIdentityResolver(use_vlm=use_vlm, use_ocr=True)

    result = resolver.resolve_player_identities(
        frame=frame,
        tracks_at_freeze=tracks,
        home_team=args.home,
        away_team=args.away,
        match_minute=args.minute,
        period=period,
        api_key=api_key,
        temporal_burst_frames=temporal_burst,
        overrides=args.overrides,
    )

    assignments = result.get("assignments", {})

    # Generate final annotated context boxes image in output directory
    annotated_path = output_dir / "tracking_pass2_context_boxes.png"
    resolver._generate_annotated_context_frame(frame, tracks, output_path=annotated_path, assignments=assignments)

    # Save output JSON mapping
    json_path = output_dir / "player_identities.json"
    clean_mapping: Dict[str, Any] = {}
    for tid, info in assignments.items():
        clean_mapping[f"T{tid}"] = {
            "player_name": info.get("name"),
            "jersey_number": info.get("number"),
            "role": info.get("role"),
            "team": info.get("team"),
            "confidence": float(info.get("confidence", 0.95)),
            "source": "OVERRIDE" if info.get("verified_by_override") else ("VLM" if info.get("verified_by_vlm") else "TOPOLOGY"),
        }

    output_payload = {
        "match": f"{args.home} vs {args.away}",
        "minute": args.minute,
        "period": period,
        "frame_idx": chosen_frame_idx,
        "identifications": clean_mapping,
        "annotated_context_frame": str(annotated_path),
        "total_resolved": len(clean_mapping),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)


    # -------------------------------------------------------------------------
    # Pretty-Print Final Results to Console
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f" MATCHDAY PLAYER IDENTIFICATION MATRIX: {args.home.upper()} vs {args.away.upper()} ({args.minute}' | Frame {chosen_frame_idx})")
    print("=" * 78)
    print(f" {'TRACK':<8} | {'TEAM':<6} | {'#':<3} | {'ROLE':<5} | {'PLAYER NAME':<24} | {'CONF':<6} | {'STATUS'}")
    print("-" * 78)

    for tid in sorted(assignments.keys()):
        asg = assignments[tid]
        p_name = asg.get("name", "Unknown")
        p_num = asg.get("number", "?")
        p_role = asg.get("role", "MF")
        p_team = str(asg.get("team", "")).upper()
        conf = float(asg.get("confidence", 0.90))
        status = "100% LOCK" if asg.get("verified_by_override") else ("VLM VERIFIED" if asg.get("verified_by_vlm") else "TOPOLOGY")

        print(f" T{tid:<7} | {p_team:<6} | {p_num:<3} | {p_role:<5} | {p_name:<24} | {conf * 100:.0f}%   | {status}")

    print("=" * 78)
    print(f" Artifact Saved: {annotated_path}")
    print(f" JSON Saved:     {json_path}")
    print(f" Total Resolved: {len(assignments)} players\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

