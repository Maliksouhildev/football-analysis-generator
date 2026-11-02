"""
scripts/migrate_audio_locked_checkpoints.py
Migrates and cleans existing *audio_locked.json checkpoint files
using the newly overhauled TacticalNarrator broadcast rules.
"""

import glob
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rendering.tactical_narrator import TacticalNarrator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("migrate_checkpoints")


def migrate():
    ckpt_dir = Path(r"F:\DockerRenders\checkpoints\barcelona_vs_real_madrid")
    if not ckpt_dir.is_dir():
        logger.warning("Checkpoint directory %s does not exist; skipping migration.", ckpt_dir)
        return

    files = sorted(glob.glob(str(ckpt_dir / "*audio_locked.json")))
    logger.info("Found %d audio_locked files to sanitize in %s.", len(files), ckpt_dir)

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        beats = data.get("beats", {})
        if not beats:
            continue

        raw_beats = {
            "beat_1_setup": beats.get("beat_1_setup", {}).get("text", ""),
            "beat_2_cause": beats.get("beat_2_cause", {}).get("text", ""),
            "beat_3_effect": beats.get("beat_3_effect", {}).get("text", ""),
            "beat_2_steps": beats.get("beat_2_steps", {}),
        }

        cleaned = TacticalNarrator._post_process_beats(
            raw_beats,
            home_team="FC Barcelona",
            away_team="Real Madrid",
        )

        # Update texts in data
        if "beat_1_setup" in beats and isinstance(beats["beat_1_setup"], dict):
            beats["beat_1_setup"]["text"] = cleaned["beat_1_setup"]
            beats["beat_1_setup"]["raw_text"] = cleaned["beat_1_setup"]

        if "beat_2_cause" in beats and isinstance(beats["beat_2_cause"], dict):
            beats["beat_2_cause"]["text"] = cleaned["beat_2_cause"]
            beats["beat_2_cause"]["raw_text"] = cleaned["beat_2_cause"]

        if "beat_3_effect" in beats and isinstance(beats["beat_3_effect"], dict):
            beats["beat_3_effect"]["text"] = cleaned["beat_3_effect"]
            beats["beat_3_effect"]["raw_text"] = cleaned["beat_3_effect"]

        if "beat_2_steps" in cleaned and isinstance(cleaned["beat_2_steps"], dict):
            data["beat_2_steps"] = cleaned["beat_2_steps"]
            if "beat_2_cause" in beats and isinstance(beats["beat_2_cause"], dict) and "steps" in beats["beat_2_cause"]:
                beats["beat_2_cause"]["steps"] = cleaned["beat_2_steps"]

        data["beats"] = beats

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        cid = Path(fpath).name
        logger.info("Sanitized %s: b2='%s'", cid, cleaned['beat_2_cause'][:60])


if __name__ == "__main__":
    migrate()
