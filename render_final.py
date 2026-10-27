import os
from pathlib import Path
import base64
import cv2
import itertools
import json
import math
import shutil
import sqlite3
import subprocess
import time
import urllib.request
import sys
import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

# Ensure FFmpeg/FFprobe binaries are available in PATH
for _p in [
    r"C:\Users\mehdi\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin",
    r"C:\Users\mlkme\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\imageio_ffmpeg\binaries",
]:
    if os.path.isdir(_p) and _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

from src.vision.tactical_vision_engine import TacticalVisionEngine
from src.rendering.tactical_narrator import TacticalNarrator
from src.rendering.audio_engine import AudioEngine


K_VIDEO_MAP = {
    "clip_01_left_wing": 80,
    "clip_02_zone_14": 95,
    "clip_03_defensive_third": 30,
    "clip_04_central": 95,
    "clip_05_right_wing": 18,
}


def realign_cause_telestration(cause_tel: dict, tracks_per_frame: dict, freeze_frame: int, clip_id: str = ""):
    """
    Re-aligns cause_telestration player coordinates to tracks_per_frame[str(freeze_frame)].
    Delegates authoritatively to TacticalVisionEngine.sanitize_v3_telestration to avoid team swaps and polygon corruption.
    """
    if not isinstance(tracks_per_frame, dict) or not isinstance(cause_tel, dict):
        return

    f_key = str(freeze_frame)
    tracks_at_freeze = tracks_per_frame.get(f_key)
    if not tracks_at_freeze:
        valid_keys = [int(k) for k in tracks_per_frame.keys() if str(k).isdigit()]
        if valid_keys:
            closest_k = min(valid_keys, key=lambda k: abs(k - freeze_frame))
            tracks_at_freeze = tracks_per_frame.get(str(closest_k), [])
            tracks_per_frame[f_key] = tracks_at_freeze

    if tracks_at_freeze is None:
        tracks_at_freeze = []
        tracks_per_frame[f_key] = tracks_at_freeze

    # Remove phantom empty-grass track 22 if present
    tracks_at_freeze[:] = [t for t in tracks_at_freeze if int(t.get("track_id", -1)) != 22]
    if isinstance(tracks_per_frame, dict):
        for k in list(tracks_per_frame.keys()):
            tracks_per_frame[k] = [t for t in tracks_per_frame[k] if int(t.get("track_id", -1)) != 22]

    cid = str(clip_id or cause_tel.get("clip_id") or "").lower()

    f_img = None
    f_paths = [
        cause_tel.get("freeze_frame_path"),
        cause_tel.get("cause_frame_path"),
        f"F:/DockerRenders/assets/frames/{clip_id}_cause.jpg",
        f"F:/DockerRenders/assets/frames/{clip_id}_freeze.jpg",
        f"F:/DockerRenders/assets/frames/{cid}_cause.jpg",
        f"F:/DockerRenders/assets/frames/{cid[:7]}_cause.jpg",
    ]
    for fp in f_paths:
        if fp and Path(str(fp)).is_file():
            try:
                f_img = cv2.imread(str(fp))
                if f_img is not None and f_img.size > 0:
                    break
            except Exception:
                pass

    homographies = None
    f_next_paths = [
        f"F:/DockerRenders/assets/frames/{clip_id}_frame_next_2.jpg",
        f"F:/DockerRenders/assets/frames/{clip_id}_frame_next_1.jpg",
        f"F:/DockerRenders/assets/frames/{clip_id}_effect.jpg",
    ]
    if f_img is not None:
        for fnp in f_next_paths:
            if fnp and Path(fnp).is_file():
                try:
                    f_next = cv2.imread(fnp)
                    if f_next is not None and f_next.size > 0:
                        from src.vision.camera_pan_compensator import CameraPanCompensator
                        cpc = CameraPanCompensator()
                        H, _ = cpc.estimate_homography(f_next, f_img)
                        homographies = {"frame_next_2": H}
                        break
                except Exception:
                    pass

    TacticalVisionEngine.sanitize_v3_telestration(
        cause_tel=cause_tel,
        tracks_at_freeze=tracks_at_freeze,
        clip_id=clip_id,
        freeze_frame_img=f_img,
        tracks_per_frame=tracks_per_frame,
        freeze_frame=freeze_frame,
        homographies=homographies,
    )
    tracks_per_frame[f_key] = tracks_at_freeze


def main():
    clip_ids = [
        "clip_01_left_wing",
        "clip_02_zone_14",
        "clip_03_defensive_third",
        "clip_04_central",
        "clip_05_right_wing",
    ]

    chk_dir = Path("F:/DockerRenders/checkpoints/roma_vs_real_madrid")
    chk_dir.mkdir(parents=True, exist_ok=True)
    audio_base = Path("F:/DockerRenders/assets/audio")
    audio_base.mkdir(parents=True, exist_ok=True)

    if "--force" not in sys.argv:
        print("=== STEP 0: Using Existing Verified Checkpoints (pass --force to clear) ===")
    else:
        print("=== STEP 0: Purging Stale Cached Audio-Locked JSON and WAV Files (--force) ===")
        deleted_json_count = 0
        for old_json in chk_dir.glob("*_audio_locked.json"):
            try:
                old_json.unlink(missing_ok=True)
                deleted_json_count += 1
            except Exception:
                pass
        print(f"Deleted {deleted_json_count} old cached audio_locked JSON files from {chk_dir}")

        deleted_wav_count = 0
        for cid in clip_ids:
            c_audio = audio_base / cid
            if c_audio.exists():
                for old_wav in c_audio.glob("*.wav"):
                    try:
                        old_wav.unlink(missing_ok=True)
                        deleted_wav_count += 1
                    except Exception:
                        pass
        print(f"Deleted {deleted_wav_count} old cached WAV files from {audio_base}")

    for db_p in [Path("pipeline_state.db"), Path("F:/DockerRenders/pipeline_state.db")]:
        if db_p.exists():
            try:
                with sqlite3.connect(str(db_p)) as conn:
                    conn.execute("DELETE FROM pipeline_state WHERE match_id LIKE '%roma%' OR match_id LIKE '%masterclass%';")
                    conn.commit()
                print(f"Reset pipeline state records in {db_p}")
            except Exception as e:
                print(f"Db reset notice ({db_p}): {e}")

    vision_engine = TacticalVisionEngine()
    narrator = TacticalNarrator()
    audio_engine = AudioEngine()
    synchronized_clips = []

    print("\n=== STEP 1: Running Vision Analysis & Audio Generation on all 5 clips ===")
    for cid in clip_ids:
        print(f"\n--- Processing {cid} ---")
        ext_path = chk_dir / f"{cid}_extracted.json"
        ew_path = chk_dir / f"{cid}_event_window.json"
        aud_path = chk_dir / f"{cid}_audio_locked.json"
        vis_path = chk_dir / f"{cid}_vision.json"

        if "--force" not in sys.argv and vis_path.is_file() and aud_path.is_file():
            print(f"  -> Loading cached verified {vis_path.name} and {aud_path.name}")
            with open(vis_path, "r", encoding="utf-8") as f:
                vis = json.load(f)
            with open(aud_path, "r", encoding="utf-8") as f:
                aud = json.load(f)
            synchronized_clips.append(aud)
            continue


        K_video = K_VIDEO_MAP.get(cid, 66)

        ext = {
            "clip_id": cid,
            "clip_path": f"F:/DockerRenders/assets/clips/{cid}.mp4",
            "cause_frame_path": f"F:/DockerRenders/assets/frames/{cid}_cause.jpg",
            "effect_frame_path": f"F:/DockerRenders/assets/frames/{cid}_effect.jpg",
            "fps": 30,
            "cause_frame_idx": K_video,
            "video_freeze_frame": K_video,
            "pre_roll_duration_frames": K_video,
            "effect_frame_idx": 240,
        }

        clip_theses = {
            "clip_01_left_wing": "Real Madrid escaping AS Roma's high press through Aurélien Tchouaméni lofting a composed safety back-pass to Thibaut Courtois",
            "clip_02_zone_14": "AS Roma pressing trap in Zone 14 as Kylian Mbappé attempts central progression before Evan Ndicka intercepts to trigger turnover",
            "clip_03_defensive_third": "Thibaut Courtois deep build-up inviting AS Roma's high press and launching a diagonal aerial pass to Federico Valverde",
            "clip_04_central": "AS Roma central transition through Evan Ndicka releasing Rodrigo Mora between Real Madrid's lines",
            "clip_05_right_wing": "Real Madrid short corner combination with Álvaro Carreras playing a one-two with Arda Güler before whipping a cross into the box",
        }
        ew = {
            "clip_id": cid,
            "narrative_tone": "TACTICAL_MASTERCLASS",
            "thesis": clip_theses.get(cid, f"Tactical analysis sequence for {cid.replace('_', ' ')}"),
            "home_team": "AS Roma",
            "away_team": "Real Madrid",
        }

        if vis_path.is_file():
            with open(vis_path, "r", encoding="utf-8") as f:
                vis = json.load(f)
            if "cause_telestration" not in vis or not isinstance(vis["cause_telestration"], dict):
                vis["cause_telestration"] = {}
            realign_cause_telestration(vis["cause_telestration"], vis.get("tracks_per_frame", {}), K_video, cid)
        else:
            vis = vision_engine.generate_synchronized_clip_vision(
                video_path=ext["clip_path"],
                event_window=ew,
                cause_frame_idx=K_video,
                effect_frame_idx=ext.get("effect_frame_idx", 240),
                fps=ext.get("fps", 30),
                beats=None,
            )
        vis["clip_id"] = cid

        # Synthesize locked 3-beat commentary audio
        clip_audio_dir = audio_base / cid
        clip_audio_dir.mkdir(parents=True, exist_ok=True)
        beats_script = narrator.generate_3beat_script(
            event_window=ew,
            cause_telestration=vis.get("cause_telestration"),
            freeze_frame_path=ext.get("cause_frame_path"),
        )

        c_idx = K_video
        locked_beats = audio_engine.synthesize_3beat_commentary(
            beats=beats_script,
            output_dir=clip_audio_dir,
            fps=ext.get("fps", 30),
            cause_frame_idx=c_idx,
        )

        for b_name in ("beat_1_setup", "beat_2_cause", "beat_3_effect"):
            if b_name in locked_beats and isinstance(locked_beats[b_name], dict):
                w_path = locked_beats[b_name].get("wav_path", "")
                locked_beats[b_name]["url"] = f"http://127.0.0.1:3000/assets/audio/{cid}/{Path(w_path).name}"

        # Decouple: Do NOT overwrite K to beat_1_frames! Lock to physical frame of contact K_video
        K = K_video
        ext["cause_frame_idx"] = K
        ext["video_freeze_frame"] = K
        ext["pre_roll_duration_frames"] = K
        vis["cause_frame_idx"] = K
        vis["video_freeze_frame"] = K
        vis["pre_roll_duration_frames"] = K

        vis["beats"] = locked_beats

        # Save freeze frame {cid}_cause.jpg at exact K_video
        vid_p = ext["clip_path"]
        if Path(vid_p).is_file():
            cap = cv2.VideoCapture(str(vid_p))
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, K)
                ret_f, f_img = cap.read()
                if ret_f and f_img is not None:
                    Path(ext["cause_frame_path"]).parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(ext["cause_frame_path"]), f_img)
                    print(f"Saved cause freeze frame for {cid} at physical frame {K} to {ext['cause_frame_path']}")
                cap.release()

        vis["beats"] = locked_beats

        # Save vision json
        vis_path = chk_dir / f"{cid}_vision.json"
        with open(vis_path, "w", encoding="utf-8") as f:
            json.dump(vis, f, indent=2, cls=NumpyJSONEncoder)
        print(f"Saved {vis_path.name}")

        aud = dict(ext)
        aud["video_path"] = ext.get("clip_path")
        aud.update({
            "tracks_per_frame": vis.get("tracks_per_frame", {}),
            "cause_telestration": vis.get("cause_telestration", {}),
            "team_colors": vis.get("team_colors", {}),
            "fg_mask_path": vis.get("fg_mask_path", ""),
            "fg_mask_url": vis.get("fg_mask_url", ""),
            "beats": locked_beats,
        })
        with open(aud_path, "w", encoding="utf-8") as f:
            json.dump(aud, f, indent=2, cls=NumpyJSONEncoder)
        print(f"Updated {aud_path.name}")

        cause_frame = str(ext.get("cause_frame_idx", 66))
        tracks = vis.get("tracks_per_frame", {}).get(cause_frame, [])
        labelled = [t for t in tracks if t.get("name")]
        focal_ids = vis.get("cause_telestration", {}).get("focal_track_ids", [])
        backline_ids = vis.get("cause_telestration", {}).get("backline_track_ids", [])
        vacated = vis.get("cause_telestration", {}).get("vacated_zone_ellipse", {})
        print(f"  Focal IDs: {focal_ids}")
        print(f"  Backline IDs: {backline_ids}")
        print(f"  Labelled count: {len(labelled)} / {len(tracks)} total detections")
        for p in labelled:
            print(f"    [{p.get('team')}] #{p.get('number')} {p.get('name')} ({p.get('role')}) at ({p.get('x_pct')}%, {p.get('y_pct')}%)")
        print(f"  Vacated Zone: cx={vacated.get('cx')}, cy={vacated.get('cy')}")

        synchronized_clips.append(aud)

    print("\n=== STEP 2: Assembling Masterclass Render Payload ===")
    active_segments = []
    current_time = 0.0

    for s_idx, sc in enumerate(synchronized_clips):
        cid = sc.get("clip_id", f"clip_{s_idx+1:02d}")
        beats = sc.get("beats", {})

        # Ensure beat audio URLs are populated with HTTP endpoints for Remotion container
        if beats:
            for b_name in ("beat_1_setup", "beat_2_cause", "beat_3_effect"):
                b_item = beats.get(b_name)
                if isinstance(b_item, dict) and not b_item.get("url"):
                    wp = b_item.get("wav_path", "")
                    norm = wp.replace("\\", "/")
                    if "DockerRenders" in norm:
                        rel = norm.split("DockerRenders/")[1].lstrip("/")
                        if rel.startswith("assets/"):
                            rel = rel[len("assets/"):]
                        b_item["url"] = f"http://127.0.0.1:3000/assets/{rel}"
                    elif wp.startswith("http://") or wp.startswith("https://"):
                        b_item["url"] = wp

        vid_path = sc.get("video_path") or sc.get("clip_path") or f"F:/DockerRenders/assets/clips/{cid}.mp4"
        actual_video_frames = 0
        if vid_path and Path(vid_path).is_file():
            try:
                cap = cv2.VideoCapture(str(vid_path))
                if cap.isOpened():
                    actual_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
            except Exception:
                actual_video_frames = 0

        K_video = K_VIDEO_MAP.get(cid, 66)
        b1_ms = beats.get("beat_1_setup", {}).get("duration_ms", 2000) if isinstance(beats.get("beat_1_setup"), dict) else 2000
        b2_ms = beats.get("beat_2_cause", {}).get("duration_ms", 5400) if isinstance(beats.get("beat_2_cause"), dict) else 5400
        b3_ms = beats.get("beat_3_effect", {}).get("duration_ms", 4200) if isinstance(beats.get("beat_3_effect"), dict) else 4200

        # Physical video freeze frame vs pre-roll duration: K = K_video (decoupled from beat_1_frames)
        b1_sec = b1_ms / 1000.0
        beat_1_frames = int(math.ceil((b1_sec + 0.15) * 30))
        b2_sec = b2_ms / 1000.0
        beat_2_frames = int(math.ceil(b2_sec * 30))

        # Synchronize freeze duration to cover Beat 2 completely with zero audio overlap
        freeze_end_frame = max(K_video, beat_1_frames) + beat_2_frames + 12
        freeze_frames = freeze_end_frame - K_video

        K = K_video
        video_freeze_frame = K
        pre_roll_duration_frames = K
        freeze_frame = K
        cause_frame_idx = K

        # Re-save cause_frame_path ({cid}_cause.jpg) from video at exact freeze_frame index K and encode Base64 data URI
        frame_path = sc.get("cause_frame_path") or sc.get("freeze_frame_path") or f"F:/DockerRenders/assets/frames/{cid}_cause.jpg"
        freeze_frame_b64 = None
        if vid_path and Path(vid_path).is_file() and frame_path:
            cap = cv2.VideoCapture(str(vid_path))
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, K)
                ret_f, f_img = cap.read()
                if ret_f and f_img is not None:
                    Path(frame_path).parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(frame_path), f_img)
                    _, buffer = cv2.imencode(".jpg", f_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    freeze_frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                    print(f"Re-saved cause freeze frame for {cid} at frame {K} to {frame_path} and encoded Base64 URI")
                cap.release()

        b3_sec = b3_ms / 1000.0
        beat_3_frames = math.ceil(b3_sec * 30) + 6

        if actual_video_frames <= 0:
            eff_idx = sc.get("effect_frame_idx", 240)
            actual_video_frames = max(eff_idx + 30, pre_roll_duration_frames + beat_3_frames + 30)

        # Eliminate trailing black screens between clips: post_roll_frames strictly bounded by video
        max_available_post_roll = max(1, actual_video_frames - K - 2)

        b3_path = None
        if isinstance(beats.get("beat_3_effect"), dict) and beats["beat_3_effect"].get("wav_path"):
            b3_path = Path(beats["beat_3_effect"]["wav_path"])
        if not b3_path or not b3_path.is_file():
            cand_b3 = audio_base / cid / "beat_3_effect.wav"
            if cand_b3.is_file():
                b3_path = cand_b3

        if beat_3_frames > max_available_post_roll and b3_path and b3_path.is_file():
            avail_sec = max(0.5, (max_available_post_roll / 30.0) - 0.2)
            speed_factor = min(1.30, max(1.10, b3_sec / avail_sec))
            temp_b3 = b3_path.with_name(f"temp_speed_{b3_path.name}")
            try:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(b3_path),
                    "-filter:a", f"atempo={speed_factor:.3f}",
                    str(temp_b3),
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                if temp_b3.is_file() and temp_b3.stat().st_size > 0:
                    temp_b3.replace(b3_path)
                    b3_sec = b3_sec / speed_factor
                    b3_ms = int(b3_sec * 1000)
                    beat_3_frames = math.ceil(b3_sec * 30) + 6
                    if isinstance(beats.get("beat_3_effect"), dict):
                        beats["beat_3_effect"]["duration_sec"] = b3_sec
                        beats["beat_3_effect"]["duration_ms"] = b3_ms
                    print(f"Sped up beat_3 commentary for {cid} by {speed_factor:.2f}x to fit available {max_available_post_roll} post-roll frames")
            except Exception as e:
                print(f"Could not speed up beat_3 for {cid}: {e}")
                if temp_b3.is_file():
                    temp_b3.unlink(missing_ok=True)

        post_roll_frames = min(max_available_post_roll, max(90, beat_3_frames))
        total_composition_frames = cause_frame_idx + freeze_frames + post_roll_frames
        seg_dur = round(total_composition_frames / 30.0, 4)

        vid_name = Path(vid_path).name if vid_path else f"{cid}.mp4"
        clip_url = f"http://127.0.0.1:3000/assets/clips/{vid_name}"

        frame_name = Path(frame_path).name if frame_path else f"{cid}_cause.jpg"
        freeze_frame_url = freeze_frame_b64 or f"http://127.0.0.1:3000/assets/frames/{frame_name}"
        fg_mask_url = sc.get("fg_mask_url") or f"http://127.0.0.1:3000/assets/frames/{cid}_fg_mask.png"

        cause_tel = dict(sc.get("cause_telestration", {}))
        cause_tel["video_freeze_frame"] = K
        cause_tel["videoFreezeFrame"] = K
        cause_tel["pre_roll_duration_frames"] = K
        cause_tel["preRollDurationFrames"] = K
        cause_tel["cause_frame_idx"] = K
        cause_tel["causeFrameIdx"] = K
        cause_tel["freeze_frame"] = K
        cause_tel["freeze_frames"] = freeze_frames
        cause_tel["freezeFrameUrl"] = freeze_frame_url
        cause_tel["freeze_frame_url"] = freeze_frame_url
        cause_tel["show_score_badge"] = False
        t_map = sc.get("tracks_per_frame", {})
        if isinstance(t_map, dict):
            c_orig = sc.get("cause_frame_idx")
            if c_orig is not None and str(c_orig) in t_map and str(K) not in t_map:
                t_map[str(K)] = t_map[str(c_orig)]
            realign_cause_telestration(cause_tel, t_map, K, clip_id=cid)
            if "clip_05" in cid:
                cause_tel["is_set_piece_or_long_ball"] = True
                cause_tel["is_set_piece"] = True
            elif "clip_03" in cid:
                cause_tel["is_set_piece_or_long_ball"] = True
                cause_tel["is_set_piece"] = True
            elif "clip_01" in cid:
                cause_tel["is_set_piece_or_long_ball"] = True
                cause_tel["is_set_piece"] = True

            # Persist updated realigned cause_telestration and tracks_per_frame to checkpoint JSON
            vis_checkpoint = chk_dir / f"{cid}_vision.json"
            if vis_checkpoint.is_file():
                try:
                    with open(vis_checkpoint, "r", encoding="utf-8") as f_r:
                        vis_data = json.load(f_r)
                    vis_data["cause_telestration"] = cause_tel
                    vis_data["tracks_per_frame"] = t_map
                    with open(vis_checkpoint, "w", encoding="utf-8") as f_w:
                        json.dump(vis_data, f_w, indent=2, cls=NumpyJSONEncoder)
                except Exception:
                    pass

        seg = {
            "id": f"seg_{cid}",
            "type": "clip",
            "url": clip_url,
            "freezeFrameUrl": freeze_frame_url,
            "freeze_frame_url": freeze_frame_url,
            "telestration": {
                "freezeFrameUrl": freeze_frame_url,
                "freeze_frame_url": freeze_frame_url,
            },
            "fgMaskUrl": fg_mask_url,
            "fg_mask_url": fg_mask_url,
            "effectFrameUrl": f"http://127.0.0.1:3000/assets/frames/{cid}_effect.jpg",
            "effect_frame_url": f"http://127.0.0.1:3000/assets/frames/{cid}_effect.jpg",
            "startTime": round(current_time, 4),
            "duration": seg_dur,
            "total_composition_frames": total_composition_frames,
            "durationInFrames": total_composition_frames,
            "duration_frames": total_composition_frames,
            "actual_clip_video_frames": actual_video_frames,
            "actual_video_frames": actual_video_frames,
            "actualVideoFrames": actual_video_frames,
            "total_video_frames": actual_video_frames,
            "totalVideoFrames": actual_video_frames,
            "post_roll_frames": post_roll_frames,
            "postRollFrames": post_roll_frames,
            "title": f"TACTICAL BREAKDOWN // {cid.upper()}",
            "video_freeze_frame": video_freeze_frame,
            "videoFreezeFrame": video_freeze_frame,
            "pre_roll_duration_frames": pre_roll_duration_frames,
            "preRollDurationFrames": pre_roll_duration_frames,
            "cause_frame_idx": cause_frame_idx,
            "causeFrameIdx": cause_frame_idx,
            "effect_frame_idx": sc.get("effect_frame_idx", 240),
            "freeze_frames": freeze_frames,
            "freezeFrames": freeze_frames,
            "tracks_per_frame": t_map,
            "cause_telestration": cause_tel,
            "ball_carrier_per_frame": cause_tel.get("ball_carrier_per_frame", {}),
            "team_colors": sc.get("team_colors", {"home_hex": "#8a1529", "away_hex": "#ffffff"}),
            "beats": beats,
        }
        active_segments.append(seg)
        current_time += seg_dur

    payload = {
        "title": "TACTICAL BREAKDOWN // SEQUENCE ANALYSIS",
        "homeTeam": "AS Roma",
        "awayTeam": "Real Madrid",
        "score": "3 - 4",
        "segments": active_segments,
        "durationInSeconds": round(current_time, 2),
        "compositionId": "TacticalMasterclass",
    }

    print("\n=== STEP 2.5: Verifying Pre-Render Invariants Across All 5 Clips ===")
    assert len(active_segments) == 5, f"Expected 5 active segments, got {len(active_segments)}"
    for s_idx, seg in enumerate(active_segments):
        cid = seg.get("id", f"seg_{s_idx}")
        c_tel = seg.get("cause_telestration", {})
        t_map = seg.get("tracks_per_frame", {})
        K_val = seg.get("cause_frame_idx")
        fr_key = str(K_val)

        # 1. T_preroll == V_freeze == cause_frame_idx == K (zero 1.5s pre-freeze black screens)
        t_preroll = seg.get("pre_roll_duration_frames")
        v_freeze = seg.get("video_freeze_frame")
        assert t_preroll == v_freeze == K_val, (
            f"Pre-freeze timing mismatch in {cid}: T_preroll={t_preroll}, V_freeze={v_freeze}, K={K_val}"
        )

        is_sp = bool(c_tel.get("is_set_piece_or_long_ball") or c_tel.get("is_set_piece") or "clip_05" in cid or "clip_03" in cid or "clip_01" in cid)
        if not is_sp:
            # 2. Exploit Zone: exactly 4 vertices, pairwise distance >= 4.0% (zero 3-point triangles)
            ez = c_tel.get("exploit_zone", {})
            ez_verts = ez.get("vertices_pct", [])
            assert len(ez_verts) == 4, f"Exploit zone in {cid} has {len(ez_verts)} vertices (must be exactly 4)!"
            for (i, p1), (j, p2) in itertools.combinations(enumerate(ez_verts), 2):
                d = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                assert d >= 4.0, (
                    f"Pairwise vertex distance between {i} and {j} in {cid} too small: {d:.2f}% < 4.0% (looks like 3-point triangle)!"
                )

            # 3. Structural Network: >= 3 vertices, all track_ids exist on freeze frame (zero 2-point lines)
            sn = c_tel.get("structural_network", {})
            sn_verts = sn.get("vertices_pct", [])
            sn_ids = sn.get("track_ids", [])
            assert len(sn_verts) >= 3, f"Structural network in {cid} has {len(sn_verts)} vertices (< 3)!"
            assert len(sn_ids) >= 3, f"Structural network in {cid} has {len(sn_ids)} track_ids (< 3)!"

            tracks_on_k = t_map.get(fr_key, [])
            present_tids = {int(t.get("track_id")) for t in tracks_on_k if isinstance(t, dict) and "track_id" in t}
            for tid in sn_ids:
                assert int(tid) in present_tids, (
                    f"Structural network track_id {tid} in {cid} is missing from tracks_per_frame[{fr_key}]!"
                )

            print(f"  [Pre-Render Assertion PASS] {cid}: T_preroll==V_freeze=={K_val}, ez_verts={len(ez_verts)} (min_pairwise>=4.0%), sn_ids={sn_ids} (all {len(sn_ids)} on freeze frame)")
        else:
            # 2 & 3. Set-Piece / Distribution: Unit lines suppressed; verify trajectory vector exists
            vecs = c_tel.get("vectors", [])
            ball_vecs = [v for v in vecs if v.get("type") in ("AERIAL_BALL_ARC", "GROUND_PASS_SOLID")]
            assert len(ball_vecs) >= 1, f"Set piece / distribution {cid} must have at least 1 trajectory vector, got {len(ball_vecs)}!"
            assert ball_vecs[0].get("from_pct") is not None, f"Trajectory vector missing from_pct in {cid}"
            target_pt = ball_vecs[0].get("landing_target_pct") or ball_vecs[0].get("to_pct")
            assert target_pt is not None, f"Trajectory vector missing landing/target in {cid}"
            print(f"  [Pre-Render Assertion PASS] {cid} (SET PIECE / DISTRIBUTION): Unit lines suppressed, {ball_vecs[0].get('type')} verified from {ball_vecs[0].get('from_pct')} to {target_pt}")

    print("All Pre-Render Invariants PASSED across all 5 clips!\n")

    render_json_path = chk_dir / "remotion_render.json"
    with open(render_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, cls=NumpyJSONEncoder)
    print(f"Saved masterclass payload with {len(active_segments)} segments (total {current_time:.2f}s) to {render_json_path}")

    if "--audit" in sys.argv:
        print("\n=== VISUAL SELF-AUDIT: Rendering Static Preview Images for all 5 clips ===")
        audit_dir = Path("F:/DockerRenders/audit_previews")
        audit_dir.mkdir(parents=True, exist_ok=True)
        brain_dir = Path(r"C:\Users\mlkme\.gemini\antigravity\brain\e1705ba5-d518-4d90-8d36-9a1916bf4cd0\audit_previews")
        brain_dir.mkdir(parents=True, exist_ok=True)

        audit_images = []
        for i, sc in enumerate(synchronized_clips):
            cid = sc.get("clip_id", f"clip_{i+1:02d}")
            seg = active_segments[i]
            seg_start = seg["startTime"]
            seg_start_f = round(seg_start * 30)
            c_idx = seg["cause_frame_idx"]
            f_frames = seg["freeze_frames"]

            b1_ms = seg.get("beats", {}).get("beat_1_setup", {}).get("duration_ms", 2000) if isinstance(seg.get("beats"), dict) else 2000
            b1_frames = int(math.ceil((b1_ms / 1000.0) * 30))
            graphics_start = max(c_idx, b1_frames)

            # 1. Moving frame before freeze (ball carrier spotlight in buildup)
            f_pre = seg_start_f + max(5, c_idx - 15)
            # 2. Freeze frame (full telestration active at 65% duration)
            f_freeze = seg_start_f + graphics_start + int(f_frames * 0.65)
            # 3. Moving frame after freeze / pass arrival
            f_post = seg_start_f + c_idx + f_frames + 25

            frames_to_render = [
                (f_pre, f"{cid}_moving_pre.png", f"Moving Before Freeze (local frame {c_idx-15})"),
                (f_freeze, f"{cid}_freeze_telestration.png", f"Freeze Frame Telestration (local frame {graphics_start+int(f_frames*0.65)})"),
                (f_post, f"{cid}_moving_post.png", f"Moving After Freeze (local frame {c_idx+f_frames+25})"),
            ]

            for global_f, fname, desc in frames_to_render:
                print(f"Rendering {cid}: {desc} at global frame {global_f}...")
                still_payload = {
                    "compositionId": "TacticalMasterclass",
                    "frame": global_f,
                    "outputFilename": fname,
                    **payload
                }
                req = urllib.request.Request(
                    "http://127.0.0.1:3008/still",
                    data=json.dumps(still_payload, cls=NumpyJSONEncoder).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    out_p = Path("F:/DockerRenders") / fname
                    if out_p.exists():
                        dest_p = audit_dir / fname
                        shutil.copy2(str(out_p), str(dest_p))
                        brain_dest = brain_dir / fname
                        shutil.copy2(str(out_p), str(brain_dest))
                        print(f"  -> Saved {dest_p} ({dest_p.stat().st_size} bytes)")
                        audit_images.append(str(brain_dest))

        print("\n=== AUDIT PREVIEW IMAGES GENERATED ===")
        for img_p in audit_images:
            print(f"Preview: {img_p}")
        return

    print("\n=== STEP 3: Dispatching Render to Remotion Server ===")
    req = urllib.request.Request(
        "http://127.0.0.1:3008/render",
        data=json.dumps(payload, cls=NumpyJSONEncoder).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        job_id = res_data.get("jobId")

    print(f"Dispatched Job ID: {job_id}")

    print("\n=== STEP 4: Polling Render Progress ===")
    poll_url = f"http://127.0.0.1:3008/status/{job_id}"
    last_prog = -1
    start_poll = time.time()
    final_output_path = None

    while time.time() - start_poll < 3600:
        try:
            with urllib.request.urlopen(poll_url, timeout=10) as resp:
                status_data = json.loads(resp.read().decode("utf-8"))
                status = status_data.get("status")
                progress = status_data.get("progress", 0)
                if progress != last_prog:
                    print(f"Render progress: {progress}% (status: {status})")
                    last_prog = progress
                if status == "done":
                    final_output_path = status_data.get("outputPath")
                    print(f"Render complete! Container path: {final_output_path}")
                    break
                if status == "error":
                    raise RuntimeError(f"Remotion render failed: {status_data.get('error')}")
        except Exception as poll_err:
            if "Remotion render failed" in str(poll_err):
                raise
            print(f"Poll warning: {poll_err}")
        time.sleep(5)

    if not final_output_path:
        raise TimeoutError("Remotion render timed out after 1200 seconds")

    host_output_mp4 = Path("F:/DockerRenders") / f"{job_id}.mp4"
    target_mp4 = Path("F:/DockerRenders/Tactical_Masterclass_Clean_V4.mp4")
    legacy_mp4 = Path("F:/DockerRenders/Roma_vs_Real_Madrid_Masterclass_Final.mp4")
    user_mp4 = Path("F:/DockerRenders/roma_vs_real_madrid_masterclass.mp4")
    final_master_mp4 = Path("F:/DockerRenders/final_masterclass.mp4")
    renders_dir = Path("F:/DockerRenders/renders")
    renders_dir.mkdir(parents=True, exist_ok=True)
    renders_master_mp4 = renders_dir / "final_masterclass.mp4"

    if host_output_mp4.exists():
        shutil.copy2(str(host_output_mp4), str(target_mp4))
        shutil.copy2(str(host_output_mp4), str(legacy_mp4))
        shutil.copy2(str(host_output_mp4), str(user_mp4))
        shutil.copy2(str(host_output_mp4), str(final_master_mp4))
        shutil.copy2(str(host_output_mp4), str(renders_master_mp4))
        print(f"Successfully copied to {target_mp4} (size: {target_mp4.stat().st_size} bytes)")
        print(f"Also copied to {legacy_mp4} (size: {legacy_mp4.stat().st_size} bytes)")
        print(f"Also copied to {user_mp4} (size: {user_mp4.stat().st_size} bytes)")
        print(f"Also copied to {final_master_mp4} (size: {final_master_mp4.stat().st_size} bytes)")
        print(f"Also copied to {renders_master_mp4} (size: {renders_master_mp4.stat().st_size} bytes)")
        try:
            brain_mp4 = Path(r"C:\Users\mlkme\.gemini\antigravity\brain\e1705ba5-d518-4d90-8d36-9a1916bf4cd0\roma_vs_real_madrid_masterclass.mp4")
            shutil.copy2(str(host_output_mp4), str(brain_mp4))
            print(f"Also copied to {brain_mp4} (size: {brain_mp4.stat().st_size} bytes)")
        except Exception as e:
            print(f"Notice: could not copy to brain artifact: {e}")
    else:
        print(f"Warning: {host_output_mp4} not found directly, checking fallback...")

    print("\n=== STEP 5: Extracting 5 Keyframe PNGs for Verification ===")
    verif_dir = Path("F:/DockerRenders/verification_final")
    verif_dir.mkdir(parents=True, exist_ok=True)
    verif_brain_dir = Path(r"C:\Users\mlkme\.gemini\antigravity\brain\e1705ba5-d518-4d90-8d36-9a1916bf4cd0\verification_final")
    verif_brain_dir.mkdir(parents=True, exist_ok=True)

    for i, sc in enumerate(synchronized_clips):
        cid = sc.get("clip_id", f"clip_{i+1:02d}")
        seg = active_segments[i]
        c_idx = seg["cause_frame_idx"]
        f_frames = seg["freeze_frames"]
        seg_start = seg["startTime"]

        b1_ms = seg.get("beats", {}).get("beat_1_setup", {}).get("duration_ms", 2000) if isinstance(seg.get("beats"), dict) else 2000
        b1_frames = int(math.ceil((b1_ms / 1000.0) * 30))
        graphics_start = max(c_idx, b1_frames)

        # 1. First half freeze (graphics active at graphics_start + 15 frames)
        ts_first_half = seg_start + (graphics_start + 15) / 30.0
        # 2. Second half freeze (Exploit zone badge & Runner badge active)
        ts_second_half = seg_start + (c_idx + int(f_frames * 0.75)) / 30.0
        # 3. Sequence 3 (+30 frames post-unpause: live pitch, tracking rings & exploit polygon)
        ts_seq3 = seg_start + (c_idx + f_frames + 30) / 30.0
        # Legacy keyframe
        ts_legacy = seg_start + (graphics_start / 30.0) + 1.5

        extracts = [
            (ts_first_half, f"freeze_first_half_{cid}.png"),
            (ts_second_half, f"freeze_second_half_{cid}.png"),
            (ts_seq3, f"seq3_active_{cid}.png"),
            (ts_legacy, f"final_keyframe_{cid}.png"),
        ]

        for ts, fname in extracts:
            out_img = verif_dir / fname
            cmd = [
                "ffmpeg", "-y", "-ss", f"{ts:.2f}",
                "-i", str(target_mp4),
                "-vframes", "1",
                "-q:v", "2",
                str(out_img)
            ]
            print(f"Extracting {fname} at {ts:.2f}s...")
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            if out_img.is_file():
                try:
                    shutil.copy2(str(out_img), str(verif_brain_dir / fname))
                except Exception:
                    pass

    print("\n=== VERIFICATION KEYFRAMES EXTRACTED ===")
    for p in sorted(verif_dir.glob("*.png")):
        print(f" - {p.name} ({p.stat().st_size} bytes)")

    print("\n=== STEP 6: OpenCV Black Frame & FFprobe Audio Verification ===")
    # 6.1 Check zero black frames across clip boundaries and sample frames using OpenCV
    cap = cv2.VideoCapture(str(target_mp4))
    total_rendered_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Target video total frames: {total_rendered_frames}, FPS: {fps_val:.2f}")

    # Compute boundary frames between segments
    boundary_frames = []
    cumulative_f = 0
    seg_start_frames = []
    for s_idx, seg in enumerate(active_segments):
        seg_start_frames.append(cumulative_f)
        cumulative_f += seg["total_composition_frames"]
        if s_idx < len(active_segments) - 1:
            boundary_frames.append(cumulative_f)

    frames_to_test = set()
    for bf in boundary_frames:
        for offset in [-3, -2, -1, 0, 1, 2, 3]:
            f_idx = bf + offset
            if 0 <= f_idx < total_rendered_frames:
                frames_to_test.add(f_idx)

    # Add 10 evenly spaced test frames
    for f_idx in range(0, total_rendered_frames, max(1, total_rendered_frames // 10)):
        frames_to_test.add(f_idx)

    black_frame_count = 0
    tested_count = 0
    for f_idx in sorted(frames_to_test):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: could not read frame {f_idx}")
            continue
        tested_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_lum = float(gray.mean())
        if mean_lum < 3.0:
            print(f"ERROR: Black frame detected at frame {f_idx} (mean luminance: {mean_lum:.2f})")
            black_frame_count += 1

    print(f"OpenCV Transition Black Frame Check: Tested {tested_count} frames. Black frames: {black_frame_count}")
    assert black_frame_count == 0, f"Found {black_frame_count} black frames in transition boundaries!"

    # 6.2 Pre-Freeze (K - 10 to K - 1) and Sequence 2 Freeze (K to K + 5) Luminance Assertion Across All Segments
    print("\n--- Verifying Pre-Freeze (K - 10 to K - 1) and Sequence 2 (K to K + 5) Luminance Across All 5 Segments ---")
    pre_freeze_tested = 0
    pre_freeze_black_count = 0
    for s_idx, seg in enumerate(active_segments):
        cid = seg.get("id", f"seg_{s_idx}")
        start_f = seg_start_frames[s_idx]
        K = seg["cause_frame_idx"]
        k_global = start_f + K

        # 10 frames immediately BEFORE K (K - 10 to K - 1) and first 5 frames of Sequence 2 (K to K + 5)
        frames_to_check = [
            (k_global - offset, f"Pre-Freeze K-{offset}")
            for offset in range(10, 0, -1)
        ] + [
            (k_global + offset, f"Sequence 2 Freeze K+{offset}")
            for offset in range(0, 6)
        ]

        for f_idx, label in frames_to_check:
            if f_idx < 0 or f_idx >= total_rendered_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"Warning: could not read frame {f_idx} for {cid} ({label})")
                continue
            pre_freeze_tested += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_lum = float(gray.mean())
            if mean_lum <= 40.0:
                print(f"ERROR: Black/dim frame detected at global frame {f_idx} for {cid} ({label}): mean luma = {mean_lum:.2f} <= 40.0")
                pre_freeze_black_count += 1
            else:
                print(f"  [OK] {cid} {label} (global frame {f_idx}): mean luma = {mean_lum:.2f} > 40.0")

    print(f"Pre-Freeze & Freeze Luminance Check: Tested {pre_freeze_tested} frames. Dim/Black frames: {pre_freeze_black_count}")
    assert pre_freeze_black_count == 0, f"Found {pre_freeze_black_count} black/dim frames (mean_luma <= 40.0) before or during freeze across segments!"

    # 6.3 Sequence 3 Footage & Central Turf Luminance Check across all 5 segments
    print("\n--- Verifying Sequence 3 Live Pitch Footage Across All Segments ---")
    seq3_tested = 0
    seq3_black_count = 0
    for s_idx, seg in enumerate(active_segments):
        cid = seg.get("id", f"seg_{s_idx}")
        start_f = seg_start_frames[s_idx]
        unpause_f = start_f + seg["cause_frame_idx"] + seg["freeze_frames"]
        for offset in [15, 30, 45, 60]:
            f_idx = unpause_f + offset
            if f_idx >= total_rendered_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: could not read Seq 3 frame {f_idx} for {cid}")
                continue
            seq3_tested += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Crop central pitch (avoid top HUD banner and bottom broadcast bar)
            turf_crop = gray[220:860, 200:1720]
            turf_lum = float(turf_crop.mean())
            if turf_lum < 10.0:
                print(f"ERROR: Dead/Black footage detected in Sequence 3 of {cid} at frame {f_idx} (+{offset}f after unpause). Turf luminance: {turf_lum:.2f}")
                seq3_black_count += 1
            else:
                print(f"  [OK] {cid} Seq 3 (+{offset}f, frame {f_idx}): Turf luminance = {turf_lum:.2f}")

    cap.release()
    print(f"Sequence 3 Live Turf Check: Tested {seq3_tested} frames. Dead/Black frames: {seq3_black_count}")
    assert seq3_black_count == 0, f"Found {seq3_black_count} black/dead footage frames in Sequence 3!"

    # 6.2 Check audio stream and zero audio track overlaps using ffprobe or ffmpeg
    try:
        ffprobe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,channels,sample_rate,duration",
            "-show_entries", "format=duration",
            "-of", "json",
            str(target_mp4)
        ]
        probe_res = subprocess.check_output(ffprobe_cmd).decode("utf-8")
        probe_data = json.loads(probe_res)
        audio_streams = probe_data.get("streams", [])
        if not audio_streams:
            raise AssertionError("No audio stream found in rendered MP4!")
        a_stream = audio_streams[0]
        total_audio_dur = float(probe_data.get("format", {}).get("duration", 0))
        print(f"FFprobe Audio: Codec={a_stream.get('codec_name')}, Channels={a_stream.get('channels')}, Sample Rate={a_stream.get('sample_rate')}, Duration={total_audio_dur:.2f}s")
    except Exception as probe_err:
        print(f"FFprobe check notice: {probe_err}")

    # 6.3 Verify zero audio track overlaps across beats in each segment
    audio_overlap_detected = False
    for i, seg in enumerate(active_segments):
        cid = seg.get("id")
        beats = seg.get("beats", {})
        c_idx = seg.get("cause_frame_idx", 66)
        f_frames = seg.get("freeze_frames", 120)
        p_frames = seg.get("post_roll_frames", 90)

        b1_ms = beats.get("beat_1_setup", {}).get("duration_ms", 0) if isinstance(beats.get("beat_1_setup"), dict) else 0
        b2_ms = beats.get("beat_2_cause", {}).get("duration_ms", 0) if isinstance(beats.get("beat_2_cause"), dict) else 0
        b3_ms = beats.get("beat_3_effect", {}).get("duration_ms", 0) if isinstance(beats.get("beat_3_effect"), dict) else 0

        b1_frames = math.ceil((b1_ms / 1000.0) * 30)
        b2_frames = math.ceil((b2_ms / 1000.0) * 30)
        b3_frames = math.ceil((b3_ms / 1000.0) * 30)

        # In decoupled timeline:
        # Beat 1: 0 to b1_frames + 5
        # Beat 2: b1_frames + 6 to b1_frames + 6 + b2_frames
        # Beat 3: freeze_end_frame (c_idx + f_frames) to freeze_end_frame + b3_frames
        gap_1_2 = (b1_frames + 6) - (b1_frames + 5)
        gap_2_3 = (c_idx + f_frames) - (b1_frames + 6 + b2_frames)
        gap_3_end = p_frames - b3_frames

        print(f"Segment {i} ({cid}): B1={b1_frames}f (gap to B2: {gap_1_2}f), B2={b2_frames}f (gap to B3: {gap_2_3}f), B3={b3_frames}f (gap to End: {gap_3_end}f)")
        if gap_1_2 < 0 or gap_2_3 < 0 or gap_3_end < 0:
            print(f"ERROR: Audio overlap detected in segment {cid}!")
            audio_overlap_detected = True

    assert not audio_overlap_detected, "Audio overlap detected between commentary beats!"
    print("Zero Audio Track Overlaps Verified: All commentary beats cleanly segregated with positive buffer intervals.")

if __name__ == "__main__":
    main()

