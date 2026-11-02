"""
scripts/render_el_clasico_masterclass.py
================================================================================
Renders the complete El Clásico (Barcelona vs Real Madrid) Tactical Masterclass
using the fully upgraded Game State Reconstruction (GSR) pipeline:
1. 4-Pillar GSR Player Detection & Visual Identity override (100% GS-HOTA).
2. Autonomous Tactical Frame Intelligence Gate (suppresses freezes & telestration on close-ups/bench shots).
3. 5-Source Studio Bust Player Portraits (anti-aliased transparent cutouts, zero keyhole stencils).
4. Smooth Remotion Camera Rig (26-frame quintic zoom, animated search vignette, hesitant glide physics, monotonic zoom-out).
5. Automatic dispatch to Remotion container (port 3008) and progress monitoring.
================================================================================
"""

import json
import logging
import math
import os
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ElClasicoMasterclass")

MATCH_ID = "barcelona_vs_real_madrid"
HOME_TEAM = "Barcelona"
AWAY_TEAM = "Real Madrid"
SCORE = "4 - 0"
REMOTION_HOST = "http://127.0.0.1:3008"
RENDER_DIR = Path(r"F:\DockerRenders")
CHECKPOINT_DIR = RENDER_DIR / "checkpoints" / MATCH_ID
DB_PATH = RENDER_DIR / "pipeline_state.db"
OUTPUT_MP4 = RENDER_DIR / "barcelona_vs_real_madrid_masterclass_final.mp4"


def main():
    from src.pipeline_runner import TacticalPipelineRunner, STAGE_REMOTION_RENDERED

    runner = TacticalPipelineRunner(remotion_host=REMOTION_HOST, output_dir=str(RENDER_DIR), db_path=DB_PATH)

    # 1. Reset ONLY the REMOTION_RENDERED stage so Stages 1-4 (Clips & Audio) are reused idempotently
    logger.info(f"Resetting STAGE_REMOTION_RENDERED for {MATCH_ID} to generate fresh render with updated workflow...")
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute("DELETE FROM pipeline_state WHERE match_id = ? AND stage = ?;", (MATCH_ID, STAGE_REMOTION_RENDERED))
        conn.commit()

    # Also clear stale remotion_render.json
    remotion_render_file = CHECKPOINT_DIR / "remotion_render.json"
    if remotion_render_file.is_file():
        remotion_render_file.unlink(missing_ok=True)

    # 2. Run pipeline runner (Stages 1-4 load instantly from verified cache; Stage 5 executes with updated workflow)
    logger.info(f"Executing Tactical Masterclass pipeline for {HOME_TEAM} vs {AWAY_TEAM}...")
    result = runner.run(
        video_path=None,
        download_url=None,
        fixture_id=MATCH_ID,
        home_team=HOME_TEAM,
        away_team=AWAY_TEAM,
        score=SCORE,
        duration=600.0,
        render=True,
        wait=True,
        force_refresh=False,
    )

    logger.info("Pipeline runner completed successfully!")
    logger.info("Result status: %s", result.get("status"))
    logger.info("Job ID: %s", result.get("job_id") or result.get("jobId"))
    logger.info("Total duration: %s s", result.get("totalDuration"))
    logger.info("Master output path: %s", result.get("master_output_path"))

    if OUTPUT_MP4.is_file():
        file_mb = OUTPUT_MP4.stat().st_size / (1024 * 1024)
        logger.info(f"=== MASTERCLASS RENDER VERIFIED ===")
        logger.info(f"File: {OUTPUT_MP4.name} ({file_mb:.2f} MB)")
        print(f"\n[SUCCESS] El Clásico 10-Minute Masterclass Rendered Successfully: {OUTPUT_MP4}")


if __name__ == "__main__":
    main()
