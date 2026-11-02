"""
scripts/test_isolated_locator_multi_frames.py
================================================================================
Runs the isolated AI player locator engine (IsolatedPlayerAILocator) across
multiple diverse real broadcast frames from El Clásico (Barcelona vs Real Madrid).

Outputs:
1. High-resolution annotated visual frames (.jpg) with bounding boxes,
   authentic player names, jersey numbers, roles, and confidence scores.
2. Cropsheets with super-resolution zoom crops for visual verification.
3. Detailed JSON report and Markdown summary for user inspection.
================================================================================
"""

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dotenv
dotenv.load_dotenv(PROJECT_ROOT / ".env")

from src.vision.isolated_player_ai_locator import (
    IsolatedPlayerAILocator,
    LocatorConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_isolated_locator")

OUTPUT_DIR = PROJECT_ROOT / "test_data" / "isolated_test_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACT_DIR = Path(r"C:\Users\mlkme\.gemini\antigravity\brain\d857130f-a346-4154-8104-dc39a8ceeb8e")

# Test frames representing wide tactical shots, close-up duels, substitutions, and goalkeeper actions
TEST_FRAMES = [
    {
        "id": "frame_01_kickoff_buildup",
        "description": "Clip 01: Early 1st-half midfield circulation & tactical backline setup (Wide Tactical)",
        "path": r"F:\DockerRenders\assets\frames\clip_01_unknown_cause.jpg",
        "minute": 1,
        "period": 1,
        "shot_type": "wide_tactical",
    },
    {
        "id": "frame_02_tactical_shape_30m",
        "description": "Clip 02: 30' Tactical shape following Balde substitution (Wide Tactical)",
        "path": r"F:\DockerRenders\assets\frames\clip_02_tactical_sub_cause.jpg",
        "minute": 30,
        "period": 1,
        "shot_type": "wide_tactical",
    },
    {
        "id": "frame_03_box_duel_goal",
        "description": "Clip 03: 41' Penalty box duel & attacking strike by Mbappé (Close Up)",
        "path": r"F:\DockerRenders\assets\frames\clip_03_goal_cause.jpg",
        "minute": 41,
        "period": 1,
        "shot_type": "close_up",
    },
    {
        "id": "frame_06_lunin_gk_and_defense",
        "description": "Clip 06: 46' Second-half kickoff showing Lunin in salmon GK kit, Fermín, Balde & Lucas Vázquez (Wide Tactical)",
        "path": r"F:\DockerRenders\assets\frames\clip_06_yellow_card_cause.jpg",
        "minute": 46,
        "period": 2,
        "shot_type": "wide_tactical",
    },
    {
        "id": "frame_10_second_half_block",
        "description": "Clip 10: 60' Midfield duel and second-half tactical block (Wide Tactical)",
        "path": r"F:\DockerRenders\assets\frames\clip_10_tactical_sub_cause.jpg",
        "minute": 60,
        "period": 2,
        "shot_type": "wide_tactical",
    },
    {
        "id": "frame_11_sideline_duel",
        "description": "Clip 11: 65' Touchline duel testing sideline boundary rejection (Wide Tactical)",
        "path": r"F:\DockerRenders\assets\frames\clip_11_yellow_card_cause.jpg",
        "minute": 65,
        "period": 2,
        "shot_type": "wide_tactical",
    },
    {
        "id": "frame_user_62m44s_duel",
        "description": "User Frame 62:44: Real Madrid #17 Lucas Vázquez vs Barcelona midfielder duel",
        "path": str(ARTIFACT_DIR / ".user_uploaded" / "media_1793569351134_1f15411f.png"),
        "minute": 62,
        "period": 2,
        "shot_type": "wide_tactical",
    },
]


def run_multi_frame_test():
    # Use full GSR pipeline config (Iter 12 settings: 1280px YOLO, adaptive turf hull, OCR, biometrics, Hungarian)
    config = LocatorConfig(
        iteration_id=12,
        name="Multi-Frame Production Evaluation (Authentic GSR)",
        yolo_imgsz=1280,
        use_adaptive_turf_hull=True,
        use_blaugrana_stripe_classifier=True,
        use_gk_color_detector=True,
        use_broadcast_banner_mask=True,
        use_touchline_staff_filter=True,
        use_tiled_highres_yolo=True,
        use_scoreboard_clock_ocr=True,
        use_live_active_xi_sync=True,
        use_homography_topology_priors=True,
        use_superres_clahe_torso_zoom=True,
        use_biometric_visual_db=True,
        use_play_by_play_telemetry=True,
        use_hungarian_global_solver=True,
        use_vlm_critic_audit=True,
        temperature=0.01,
    )

    logger.info("Initializing IsolatedPlayerAILocator with authentic El Clásico rosters...")
    locator = IsolatedPlayerAILocator(config=config)

    results = []

    for item in TEST_FRAMES:
        fpath = Path(item["path"])
        if not fpath.is_file():
            logger.warning(f"File not found, skipping: {fpath}")
            continue

        fid = item["id"]
        logger.info(f"--- Processing {fid} ({item['description']}) ---")
        t0 = time.time()

        res = locator.analyze_frame(
            frame_or_path=str(fpath),
            event_window={
                "clip_id": fid,
                "minute": item["minute"],
                "period": item["period"],
                "raw_event": {"minute": item["minute"], "period": item["period"], "type": "tactical"},
            },
            home_team="FC Barcelona",
            away_team="Real Madrid",
            fallback_minute=item["minute"],
            period_hint=item["period"],
            save_debug_dir=OUTPUT_DIR,
        )

        elapsed = time.time() - t0
        actors = res.get("actors", [])
        annotated_img = res.get("annotated_frame")
        cropsheet_img = res.get("crop_sheet")

        # Save dedicated high-res annotated frame
        out_annotated_name = f"{fid}_annotated.jpg"
        out_annotated_path = OUTPUT_DIR / out_annotated_name
        if annotated_img is not None and annotated_img.size > 0:
            cv2.imwrite(str(out_annotated_path), annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            # Also copy to artifact directory for markdown display
            if ARTIFACT_DIR.is_dir():
                shutil.copy2(str(out_annotated_path), str(ARTIFACT_DIR / out_annotated_name))

        out_cropsheet_name = f"{fid}_cropsheet.jpg"
        out_cropsheet_path = OUTPUT_DIR / out_cropsheet_name
        if cropsheet_img is not None and cropsheet_img.size > 0:
            cv2.imwrite(str(out_cropsheet_path), cropsheet_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ARTIFACT_DIR.is_dir():
                shutil.copy2(str(out_cropsheet_path), str(ARTIFACT_DIR / out_cropsheet_name))

        home_actors = [a for a in actors if a.get("team") == "home"]
        away_actors = [a for a in actors if a.get("team") == "away"]

        frame_summary = {
            "id": fid,
            "description": item["description"],
            "minute": item["minute"],
            "period": item["period"],
            "shot_type": item["shot_type"],
            "elapsed_sec": round(elapsed, 2),
            "total_actors": len(actors),
            "home_count": len(home_actors),
            "away_count": len(away_actors),
            "annotated_image": out_annotated_name,
            "cropsheet_image": out_cropsheet_name,
            "actors": [
                {
                    "track_id": a.get("track_id"),
                    "name": a.get("assigned_name"),
                    "number": a.get("assigned_number"),
                    "role": a.get("assigned_role"),
                    "team": a.get("team", "").upper(),
                    "confidence": round(float(a.get("assignment_confidence", 0.0)), 2),
                    "method": a.get("assignment_method"),
                    "pos_pct": (round(float(a.get("x_pct", 0.0)), 1), round(float(a.get("y_pct", 0.0)), 1)),
                    "is_gk": a.get("is_gk", False),
                }
                for a in actors
            ],
        }
        results.append(frame_summary)

        logger.info(
            f"Finished {fid}: {len(actors)} actors ({len(home_actors)} Barcelona, {len(away_actors)} Real Madrid) in {elapsed:.2f}s"
        )
        for a in actors[:6]:
            logger.info(
                f"  [{a.get('track_id')}] #{a.get('assigned_number')} {a.get('assigned_name')} ({a.get('team', '').upper()}) conf={a.get('assignment_confidence'):.2f} via {a.get('assignment_method')}"
            )

    report_path = OUTPUT_DIR / "multi_frame_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Multi-frame test completed! Full report saved to: {report_path}")
    return results


if __name__ == "__main__":
    run_multi_frame_test()
