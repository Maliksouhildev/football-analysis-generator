import os
import sys
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, r"C:\Users\mlkme\.antigravity\workflows\127-0-0-1-5678\football automation")

from src.vision.tactical_vision_engine import TacticalVisionEngine
from src.rendering.audio_engine import AudioEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sanitize_clasico")

CHECKPOINT_DIR = Path(r"F:\DockerRenders\checkpoints\barcelona_vs_real_madrid")
FRAMES_DIR = Path(r"F:\DockerRenders\assets\frames")

NAME_REPLACEMENTS = {
    "Ibrahima Konaté": "Aurélien Tchouaméni",
    "Konaté (CB)": "Tchouaméni (DM)",
    "Konaté": "Tchouaméni",
    "Konate": "Tchouameni",
    "Donyell Malen": "Robert Lewandowski",
    "Malen (CF)": "Lewandowski (CF)",
    "Malen": "Lewandowski",
    "Marten de Roon": "Marc Casadó",
    "De Roon (DM)": "Casadó (DM)",
    "De Roon": "Casadó",
    "de Roon": "Casadó",
    "Roon": "Casadó",
    "Marc Cucurella": "Ferland Mendy",
    "Cucurella (LB)": "Mendy (LB)",
    "Cucurella": "Mendy",
    "Denzel Dumfries": "Lucas Vázquez",
    "Dumfries (RB)": "Lucas Vázquez (RB)",
    "Dumfries": "Lucas Vázquez",
    "Paulo Dybala": "Lamine Yamal",
    "Dybala (AM)": "Yamal (RW)",
    "Dybala": "Yamal",
    "Evan Ndicka": "Pau Cubarsí",
    "Ndicka (CB)": "Cubarsí (CB)",
    "Ndicka": "Cubarsí",
    "Gianluca Mancini": "Jules Koundé",
    "Mancini (CB)": "Koundé (RB)",
    "Mancini": "Koundé",
    "Matías Soulé": "Raphinha",
    "Soulé (RW)": "Raphinha (LW)",
    "Soulé": "Raphinha",
    "Soule": "Raphinha",
    "Niccolò Pisilli": "Pedri",
    "Pisilli (CM)": "Pedri (CM)",
    "Pisilli": "Pedri",
    "Pierluigi Gollini": "Iñaki Peña",
    "Gollini (GK)": "Peña (GK)",
    "Gollini": "Peña",
    "Yan Diomande": "Rodrygo",
    "Diomande (RW)": "Rodrygo (RW)",
    "Diomande": "Rodrygo",
    "Mario Hermoso": "Iñigo Martínez",
    "Hermoso (CB)": "Iñigo Martínez (CB)",
    "Hermoso": "Iñigo Martínez",
    "Carlos Espí": "Endrick",
    "Carlos Espí (ST)": "Endrick (CF)",
    "Espí": "Endrick",
    "Espi": "Endrick",
    "Rodrigo Mora": "Fermín López",
    "Rodrigo Mora (AM)": "Fermín (AM)",
    "Mora": "Fermín",
    "Koulierakis (CB)": "Cubarsí (CB)",
    "Koulierakis": "Cubarsí",
}

def clean_text(text: str) -> str:
    if not text:
        return text
    res = text
    for old_name, new_name in NAME_REPLACEMENTS.items():
        res = res.replace(old_name, new_name)
    return res

def sanitize_clip(cid: str, audio_engine: AudioEngine):
    vis_path = CHECKPOINT_DIR / f"{cid}_vision.json"
    aud_path = CHECKPOINT_DIR / f"{cid}_audio_locked.json"

    if not vis_path.is_file():
        logger.warning(f"Vision file not found for {cid}")
        return

    logger.info(f"--- Processing {cid} ---")
    with open(vis_path, "r", encoding="utf-8") as f:
        v_data = json.load(f)

    ctel = v_data.get("cause_telestration", {})
    t_map = v_data.get("tracks_per_frame", {})
    K = v_data.get("cause_frame_idx", 60)
    tracks_at_freeze = t_map.get(str(K), [])

    # Re-sanitize with Barcelona vs Real Madrid context
    TacticalVisionEngine.sanitize_v3_telestration(
        cause_tel=ctel,
        tracks_at_freeze=tracks_at_freeze,
        clip_id=cid,
        event_context={
            "home_team": "Barcelona",
            "away_team": "Real Madrid",
            "match_minute": ctel.get("match_minute", 25),
            "period": ctel.get("period", 1),
        },
        tracks_per_frame=t_map,
        freeze_frame=K,
    )
    t_map[str(K)] = tracks_at_freeze
    v_data["cause_telestration"] = ctel

    with open(vis_path, "w", encoding="utf-8") as f:
        json.dump(v_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Updated {vis_path.name}")

    if not aud_path.is_file():
        return

    with open(aud_path, "r", encoding="utf-8") as f:
        a_data = json.load(f)

    a_data["cause_telestration"] = ctel
    a_data["tracks_per_frame"] = t_map

    beats = a_data.get("beats", {})
    text_changed = False

    # Check and clean beat texts
    for b_key in ["beat_1_setup", "beat_2_cause", "beat_3_effect"]:
        if b_key in beats and isinstance(beats[b_key], dict):
            orig_txt = beats[b_key].get("text", "")
            cleaned = clean_text(orig_txt)
            if cleaned != orig_txt:
                logger.info(f"Cleaned {b_key} text: '{orig_txt}' -> '{cleaned}'")
                beats[b_key]["text"] = cleaned
                text_changed = True

    # Clean visual cue tags
    if "visual_cue_tags" in beats and isinstance(beats["visual_cue_tags"], list):
        for tag in beats["visual_cue_tags"]:
            if isinstance(tag, dict):
                for k in ["target", "player_name"]:
                    if k in tag and tag[k]:
                        tag[k] = clean_text(tag[k])

    if text_changed:
        logger.info(f"Re-synthesizing voiceover for {cid} with Kokoro TTS...")
        clip_audio_dir = Path(r"F:\DockerRenders\assets\audio") / cid
        clip_audio_dir.mkdir(parents=True, exist_ok=True)
        try:
            for b_num, b_key in [(1, "beat_1_setup"), (2, "beat_2_cause"), (3, "beat_3_effect")]:
                if b_key in beats and isinstance(beats[b_key], dict):
                    txt = beats[b_key].get("text", "")
                    if txt:
                        wav_bytes = audio_engine.synthesize_speech_bytes(txt)
                        out_wav = clip_audio_dir / f"beat_{b_num}.wav"
                        with open(out_wav, "wb") as wf:
                            wf.write(wav_bytes)
                        beats[b_key]["audio_path"] = str(out_wav)
            logger.info(f"Re-synthesized audio for {cid} successfully.")
        except Exception as e:
            logger.warning(f"Kokoro synthesis notice for {cid}: {e}")

    a_data["beats"] = beats
    with open(aud_path, "w", encoding="utf-8") as f:
        json.dump(a_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Updated {aud_path.name}")

def main():
    audio_engine = AudioEngine()
    vis_files = sorted(list(CHECKPOINT_DIR.glob("*_vision.json")))
    logger.info(f"Found {len(vis_files)} vision checkpoints to sanitize.")

    for vf in vis_files:
        cid = vf.name.replace("_vision.json", "")
        try:
            sanitize_clip(cid, audio_engine)
        except Exception as err:
            logger.error(f"Error sanitizing {cid}: {err}", exc_info=True)

    # Clear REMOTION_RENDERED from database
    import sqlite3
    db_path = Path(r"F:\DockerRenders\pipeline_state.db")
    if db_path.is_file():
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute("DELETE FROM pipeline_state WHERE match_id='barcelona_vs_real_madrid' AND stage='REMOTION_RENDERED'")
        conn.commit()
        logger.info("Cleared REMOTION_RENDERED checkpoints from pipeline_state.db")
        conn.close()

if __name__ == "__main__":
    main()
