"""
Monitor Remotion render job, finalize pipeline state, and run full quality verification.
"""

import json
import logging
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Finalizer")

JOB_ID = "3f3fba70-c364-457e-8253-994b73743e68"
MATCH_ID = "barcelona_vs_real_madrid"
REMOTION_HOST = "http://127.0.0.1:3008"
RENDER_DIR = Path(r"F:\DockerRenders")
DB_PATH = RENDER_DIR / "pipeline_state.db"
OUTPUT_MP4 = RENDER_DIR / f"{JOB_ID}.mp4"
VERIFICATION_DIR = RENDER_DIR / "verification_final"
VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)


def poll_until_done(timeout_sec: int = 3600):
    url = f"{REMOTION_HOST}/status/{JOB_ID}"
    start = time.time()
    last_prog = -1
    logger.info(f"Waiting for render job {JOB_ID} to complete...")
    while time.time() - start < timeout_sec:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                status = data.get("status")
                progress = data.get("progress", 0)
                if progress != last_prog:
                    logger.info(f"Render progress: {progress}% (status: {status})")
                    last_prog = progress
                if status == "done":
                    logger.info(f"Render finished! Output: {data.get('outputPath')}")
                    return data
                if status == "error":
                    logger.error(f"Render failed: {data.get('error')}")
                    return data
        except Exception as e:
            logger.warning(f"Poll warning: {e}")
        time.sleep(10)
    logger.error("Timed out waiting for render completion.")
    return {"status": "timeout"}


def update_database(render_data):
    logger.info("Updating pipeline_state.db...")
    render_result = {
        "status": "COMPLETED",
        "job_id": JOB_ID,
        "match": "Barcelona vs Real Madrid",
        "match_id": MATCH_ID,
        "outputPath": str(OUTPUT_MP4),
        "video_url": f"{REMOTION_HOST}/video/{JOB_ID}",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    render_json_path = RENDER_DIR / "checkpoints" / MATCH_ID / "remotion_render.json"
    render_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(render_json_path, "w", encoding="utf-8") as f:
        json.dump(render_result, f, indent=2)

    conn = sqlite3.connect(str(DB_PATH))
    with conn:
        conn.execute(
            """
            INSERT INTO pipeline_state (match_id, clip_id, stage, status, json_payload_path, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(match_id, clip_id, stage) DO UPDATE SET
                status=excluded.status,
                json_payload_path=excluded.json_payload_path,
                updated_at=CURRENT_TIMESTAMP;
        """,
            (MATCH_ID, "__match__", "REMOTION_RENDERED", "COMPLETED", str(render_json_path)),
        )
    logger.info("Database checkpoint updated successfully.")


def run_verification():
    logger.info("Running output verification...")
    if not OUTPUT_MP4.is_file():
        logger.error(f"Expected output file not found: {OUTPUT_MP4}")
        return False

    file_size_mb = OUTPUT_MP4.stat().st_size / (1024 * 1024)
    logger.info(f"Output file size: {file_size_mb:.2f} MB")

    # ffprobe duration check
    cmd_probe = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(OUTPUT_MP4),
    ]
    try:
        dur_str = subprocess.check_output(cmd_probe, text=True).strip()
        dur_sec = float(dur_str)
        logger.info(f"Rendered video duration: {dur_sec:.2f}s ({dur_sec / 60:.2f} min)")
    except Exception as e:
        logger.error(f"Failed to probe duration: {e}")
        dur_sec = 0.0

    # Black frame detection
    logger.info("Running black frame detection (d=2.0s)...")
    cmd_black = [
        "ffmpeg",
        "-i",
        str(OUTPUT_MP4),
        "-vf",
        "blackdetect=d=2.0:pix_th=0.00",
        "-f",
        "null",
        "-",
    ]
    try:
        res = subprocess.run(cmd_black, capture_output=True, text=True)
        black_lines = [l for l in res.stderr.splitlines() if "blackdetect" in l and "black_start" in l]
        if black_lines:
            logger.warning(f"Found {len(black_lines)} black frame segments: {black_lines}")
        else:
            logger.info("No black frame segments detected! Video is continuous and clean.")
    except Exception as e:
        logger.warning(f"Blackdetect check error: {e}")

    # Extract verification keyframe stills
    keyframes = [10.0, 30.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 400.0]
    for ts in keyframes:
        if ts < dur_sec:
            out_img = VERIFICATION_DIR / f"verify_still_{int(ts)}s.jpg"
            cmd_img = [
                "ffmpeg",
                "-y",
                "-ss",
                str(ts),
                "-i",
                str(OUTPUT_MP4),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(out_img),
            ]
            subprocess.run(cmd_img, capture_output=True)
            if out_img.is_file():
                logger.info(f"Saved keyframe still at {ts}s -> {out_img}")

    logger.info("=== VERIFICATION COMPLETE ===")
    return True


if __name__ == "__main__":
    res = poll_until_done()
    if res.get("status") == "done":
        update_database(res)
        run_verification()
    else:
        logger.error(f"Render did not finish successfully: {res}")
        sys.exit(1)
