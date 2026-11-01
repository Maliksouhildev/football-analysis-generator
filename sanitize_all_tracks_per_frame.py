import os
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sanitize_tracks")

CHECKPOINT_DIR = Path(r"F:\DockerRenders\checkpoints\barcelona_vs_real_madrid")

NAME_REPLACEMENTS = [
    ("Ibrahima Konaté", "Aurélien Tchouaméni"),
    ("Konaté (CB)", "Tchouaméni (DM)"),
    ("Konaté", "Tchouaméni"),
    ("Konate (CB)", "Tchouaméni (DM)"),
    ("Konate", "Tchouaméni"),
    ("Donyell Malen", "Robert Lewandowski"),
    ("Malen (CF)", "Lewandowski (CF)"),
    ("Malen", "Lewandowski"),
    ("Marten de Roon", "Marc Casadó"),
    ("De Roon (DM)", "Casadó (DM)"),
    ("de Roon (DM)", "Casadó (DM)"),
    ("De Roon", "Casadó"),
    ("de Roon", "Casadó"),
    ("Roon (DM)", "Casadó (DM)"),
    ("Roon", "Casadó"),
    ("Marc Cucurella", "Ferland Mendy"),
    ("Cucurella (LB)", "Mendy (LB)"),
    ("Cucurella", "Mendy"),
    ("Denzel Dumfries", "Lucas Vázquez"),
    ("Dumfries (RB)", "Lucas Vázquez (RB)"),
    ("Dumfries", "Lucas Vázquez"),
    ("Paulo Dybala", "Lamine Yamal"),
    ("Dybala (AM)", "Yamal (RW)"),
    ("Dybala", "Yamal"),
    ("Evan Ndicka", "Pau Cubarsí"),
    ("Ndicka (CB)", "Cubarsí (CB)"),
    ("Ndicka", "Cubarsí"),
    ("Gianluca Mancini", "Jules Koundé"),
    ("Mancini (CB)", "Koundé (RB)"),
    ("Mancini", "Koundé"),
    ("Matías Soulé", "Raphinha"),
    ("Soulé (RW)", "Raphinha (LW)"),
    ("Soulé", "Raphinha"),
    ("Soule (RW)", "Raphinha (LW)"),
    ("Soule", "Raphinha"),
    ("Niccolò Pisilli", "Pedri"),
    ("Pisilli (CM)", "Pedri (CM)"),
    ("Pisilli", "Pedri"),
    ("Pierluigi Gollini", "Iñaki Peña"),
    ("Gollini (GK)", "Peña (GK)"),
    ("Gollini", "Peña"),
    ("Yan Diomande", "Rodrygo"),
    ("Diomande (RW)", "Rodrygo (RW)"),
    ("Diomande", "Rodrygo"),
    ("Mario Hermoso", "Iñigo Martínez"),
    ("Hermoso (CB)", "Iñigo Martínez (CB)"),
    ("Hermoso", "Iñigo Martínez"),
    ("Carlos Espí", "Endrick"),
    ("Carlos Espí (ST)", "Endrick (CF)"),
    ("Espí (ST)", "Endrick (CF)"),
    ("Espí", "Endrick"),
    ("Espi (ST)", "Endrick (CF)"),
    ("Espi", "Endrick"),
    ("Rodrigo Mora", "Fermín López"),
    ("Rodrigo Mora (AM)", "Fermín (AM)"),
    ("Mora (AM)", "Fermín (AM)"),
    ("Mora", "Fermín"),
    ("Koulierakis (CB)", "Cubarsí (CB)"),
    ("Koulierakis", "Cubarsí"),
]

SLUG_REPLACEMENTS = [
    ("ibrahima_konate", "aurelien_tchouameni"),
    ("konate", "tchouameni"),
    ("donyell_malen", "robert_lewandowski"),
    ("malen", "lewandowski"),
    ("marten_de_roon", "marc_casado"),
    ("de_roon", "marc_casado"),
    ("roon", "casado"),
    ("marc_cucurella", "ferland_mendy"),
    ("cucurella", "mendy"),
    ("denzel_dumfries", "lucas_vazquez"),
    ("dumfries", "lucas_vazquez"),
    ("paulo_dybala", "lamine_yamal"),
    ("dybala", "yamal"),
    ("evan_ndicka", "pau_cubarsi"),
    ("ndicka", "cubarsi"),
    ("gianluca_mancini", "jules_kounde"),
    ("mancini", "kounde"),
    ("matias_soule", "raphinha"),
    ("soule", "raphinha"),
    ("niccolo_pisilli", "pedri"),
    ("pisilli", "pedri"),
    ("pierluigi_gollini", "inaki_pena"),
    ("gollini", "inaki_pena"),
    ("yan_diomande", "rodrygo"),
    ("diomande", "rodrygo"),
    ("mario_hermoso", "inigo_martinez"),
    ("hermoso", "inigo_martinez"),
    ("carlos_espi", "endrick"),
    ("espi", "endrick"),
    ("rodrigo_mora", "fermin_lopez"),
    ("mora", "fermin"),
    ("koulierakis", "cubarsi"),
]

import re

def clean_str(val: str) -> str:
    if not isinstance(val, str) or not val:
        return val
    res = val
    for old_n, new_n in NAME_REPLACEMENTS:
        res = res.replace(old_n, new_n)
    for old_s, new_s in SLUG_REPLACEMENTS:
        res = re.sub(re.escape(old_s), new_s, res, flags=re.IGNORECASE)
    return res

def clean_track_dict(tr: dict, freeze_id_map: dict = None) -> None:
    # 1. Referee filtering
    is_ref = (
        str(tr.get("jersey_class", "")).upper() == "REFEREE" or
        str(tr.get("team", "")).lower() == "referee" or
        "referee" in str(tr.get("name", "")).lower()
    )
    if is_ref:
        tr["name"] = ""
        tr["short_name"] = ""
        tr["role"] = ""
        tr["label"] = ""
        tr["name_pos"] = ""
        tr["number"] = ""
        tr["showBeam"] = False
        tr["showRing"] = False
        tr["team"] = "referee"
        tr["jersey_class"] = "REFEREE"
        return

    tid = tr.get("track_id")
    # If this track ID has an authoritative assignment at freeze, adopt it
    if freeze_id_map and tid in freeze_id_map:
        auth = freeze_id_map[tid]
        tr["name"] = auth.get("name", tr.get("name", ""))
        tr["short_name"] = auth.get("short_name", tr.get("short_name", ""))
        tr["role"] = auth.get("role", tr.get("role", ""))
        tr["label"] = auth.get("label", tr.get("label", ""))
        tr["name_pos"] = auth.get("name_pos", tr.get("name_pos", ""))
        return

    # Clean existing string fields
    for field in ["name", "short_name", "role", "label", "name_pos"]:
        if field in tr and tr[field]:
            tr[field] = clean_str(tr[field])

def clean_deep(obj):
    if isinstance(obj, dict):
        return {k: clean_deep(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_deep(item) for item in obj]
    elif isinstance(obj, str):
        return clean_str(obj)
    return obj

def sanitize_json_file(file_path: Path):
    logger.info(f"Processing {file_path.name}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return

    if isinstance(data, list):
        data = clean_deep(data)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully updated list {file_path.name}")
        return

    # If it has tracks_per_frame
    t_map = data.get("tracks_per_frame")
    if isinstance(t_map, dict):
        K = str(data.get("cause_frame_idx", 60))
        freeze_tracks = t_map.get(K, [])
        freeze_id_map = {}
        for ft in freeze_tracks:
            tid = ft.get("track_id")
            clean_track_dict(ft)
            if tid is not None and ft.get("name"):
                freeze_id_map[tid] = {
                    "name": ft.get("name"),
                    "short_name": ft.get("short_name"),
                    "role": ft.get("role"),
                    "label": ft.get("label"),
                    "name_pos": ft.get("name_pos"),
                }

        # Propagate to all frames
        for f_key, f_tracks in t_map.items():
            if isinstance(f_tracks, list):
                for tr in f_tracks:
                    if isinstance(tr, dict):
                        clean_track_dict(tr, freeze_id_map)

    # Clean cause_telestration
    if "cause_telestration" in data and isinstance(data["cause_telestration"], dict):
        data["cause_telestration"] = clean_deep(data["cause_telestration"])

    # Clean beats
    if "beats" in data and isinstance(data["beats"], dict):
        data["beats"] = clean_deep(data["beats"])

    # Clean top-level fields
    for top_k in ["event_window", "active_roster", "thesis", "narrative_tone", "summary"]:
        if top_k in data:
            data[top_k] = clean_deep(data[top_k])

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Successfully updated {file_path.name}")

def main():
    target_files = sorted(list(CHECKPOINT_DIR.glob("*.json")))
    logger.info(f"Sanitizing all tracks across {len(target_files)} checkpoint files in {CHECKPOINT_DIR}")

    for f in target_files:
        sanitize_json_file(f)

    # Clear REMOTION_RENDERED in DB to guarantee a fresh full render
    import sqlite3
    db_path = Path(r"F:\DockerRenders\pipeline_state.db")
    if db_path.is_file():
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute("DELETE FROM pipeline_state WHERE match_id='barcelona_vs_real_madrid' AND stage='REMOTION_RENDERED'")
        conn.commit()
        logger.info("Cleared REMOTION_RENDERED from pipeline_state.db")
        conn.close()

if __name__ == "__main__":
    main()
