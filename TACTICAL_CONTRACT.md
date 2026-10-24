# Tactical Masterclass — Inter-Module Data Contract
# All modules MUST strictly produce and consume these exact JSON structures.

# =====================================================================
# 1. Event Window Schema 
# Produced by: src/scraper/
# Consumed by: src/ingestion/ and src/vision/
# =====================================================================
EventWindowSchema:
  {
    "clip_id": "clip_01_overload",
    "period": 2,
    "t_cause_clock": "83:56",
    "t_effect_clock": "84:12",
    "narrative_tone": "RUTHLESS_AUTOPSY",
    "thesis": "Defensive block was repeatedly stretched laterally due to aggressive flank overload",
    "crowd_evidence": "Top fan consensus: Flank space exploited by progressive overlapping run.",
    "active_roster": {
      "home": [{"num": 2, "name": "Karsdorp", "role": "RB"}],
      "away": [{"num": 7, "name": "Vinicius Jr", "role": "LW"}]
    }
  }

# =====================================================================
# 2. Synchronized Clip & Vision Schema
# Produced by: src/vision/
# Consumed by: src/rendering/ and remotion-server/
# =====================================================================
SynchronizedClipVisionSchema:
  {
    "clip_id": "clip_01_overload",
    "video_path": "renders/clips/clip_01_overload.mp4",
    "fps": 30,
    "cause_frame_idx": 90,
    "effect_frame_idx": 240,
    "team_colors": {"home_hex": "#8a1529", "away_hex": "#ffffff"},
    "tracks_per_frame": {
      "90": [{"track_id": 4, "team": "home", "x_pct": 78.4, "y_pct": 64.2, "bbox": [410, 620, 510, 810]}]
    },
    "cause_telestration": {
      "focal_track_ids": [4, 11],
      "backline_track_ids": [2, 4, 5, 3],
      "triangle_track_ids": [8, 10, 11],
      "vacated_zone_polygon": [[52.1, 41.0], [68.4, 43.2], [66.0, 58.5], [49.8, 56.1]],
      "vacated_label": "VACATED HALF-SPACE"
    },
    "beats": {
      "beat_1_setup": {"text": "...", "wav_path": "...", "duration_ms": 3800},
      "beat_2_cause": {"text": "...", "wav_path": "...", "duration_ms": 5400, "freeze_frames": 162},
      "beat_3_effect": {"text": "...", "wav_path": "...", "duration_ms": 4200}
    }
  }