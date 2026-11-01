import os
import sys
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, r"C:\Users\mlkme\.antigravity\workflows\127-0-0-1-5678\football automation")

from src.rendering.audio_engine import AudioEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("resynthesize_audio")

CHECKPOINT_DIR = Path(r"F:\DockerRenders\checkpoints\barcelona_vs_real_madrid")
AUDIO_DIR = Path(r"F:\DockerRenders\assets\audio")

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

def resync_all_audio():
    audio_engine = AudioEngine()
    aud_files = sorted(list(CHECKPOINT_DIR.glob("*_audio_locked.json")))
    logger.info(f"Checking {len(aud_files)} audio_locked checkpoints for re-synthesis...")

    total_resynthesized = 0
    for af in aud_files:
        cid = af.name.replace("_audio_locked.json", "")
        with open(af, "r", encoding="utf-8") as f:
            a_data = json.load(f)

        beats = a_data.get("beats", {})
        clip_audio_dir = AUDIO_DIR / cid
        clip_audio_dir.mkdir(parents=True, exist_ok=True)

        updated = False
        for b_num, b_key in [(1, "beat_1_setup"), (2, "beat_2_cause"), (3, "beat_3_effect")]:
            if b_key in beats and isinstance(beats[b_key], dict):
                orig_txt = beats[b_key].get("text", "")
                cleaned_txt = clean_text(orig_txt)
                if cleaned_txt != orig_txt:
                    beats[b_key]["text"] = cleaned_txt
                    beats[b_key]["raw_text"] = cleaned_txt
                    updated = True

                # Check if audio file exists or if text changed
                out_wav = clip_audio_dir / f"beat_{b_num}.wav"
                existing_path = beats[b_key].get("wav_path") or beats[b_key].get("audio_path")

                # If text changed or audio path missing/doesn't exist
                should_synth = (
                    cleaned_txt != orig_txt or 
                    not existing_path or 
                    not Path(existing_path).is_file()
                )

                if should_synth and cleaned_txt:
                    logger.info(f"[{cid}] Synthesizing {b_key}: '{cleaned_txt[:50]}...'")
                    try:
                        dur = audio_engine.synthesize_commentary_clip(cleaned_txt, out_wav)
                        dur_ms = int(dur * 1000)
                        beats[b_key]["audio_path"] = str(out_wav)
                        beats[b_key]["wav_path"] = str(out_wav)
                        beats[b_key]["url"] = f"http://127.0.0.1:3000/assets/audio/{cid}/beat_{b_num}.wav"
                        beats[b_key]["duration"] = dur
                        beats[b_key]["duration_sec"] = dur
                        beats[b_key]["duration_ms"] = dur_ms
                        updated = True
                        total_resynthesized += 1
                        logger.info(f"[{cid}] Synthesized {b_key} ({dur:.2f}s) -> {out_wav.name}")
                    except Exception as e:
                        logger.error(f"[{cid}] Failed to synthesize {b_key}: {e}")

        # Also clean visual cue tags in beats
        if "visual_cue_tags" in beats and isinstance(beats["visual_cue_tags"], list):
            for tag in beats["visual_cue_tags"]:
                if isinstance(tag, dict):
                    for k in ["target", "player_name"]:
                        if k in tag and tag[k]:
                            cleaned_t = clean_text(tag[k])
                            if cleaned_t != tag[k]:
                                tag[k] = cleaned_t
                                updated = True

        if updated:
            a_data["beats"] = beats
            with open(af, "w", encoding="utf-8") as f:
                json.dump(a_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved updated {af.name}")

    logger.info(f"Finished audio resync. Total clips re-synthesized: {total_resynthesized}")

if __name__ == "__main__":
    resync_all_audio()
