import json, urllib.request, shutil
from pathlib import Path

render_json = Path("F:/DockerRenders/checkpoints/roma_vs_real_madrid/remotion_render.json")
with open(render_json, "r") as f:
    payload = json.load(f)

# Update segment 0 (clip_01_left_wing) with updated cause_telestration vectors and visual_cue_tags
chk = Path("F:/DockerRenders/checkpoints/roma_vs_real_madrid/clip_01_left_wing_vision.json")
with open(chk, "r") as f:
    c01_data = json.load(f)

for seg in payload.get("segments", []):
    if "clip_01_left_wing" in seg.get("id", ""):
        seg["cause_telestration"]["vectors"] = c01_data["cause_telestration"]["vectors"]
        seg["cause_telestration"]["pressing_track_ids"] = c01_data["cause_telestration"].get("pressing_track_ids", [])
        
        # Update visualCueTags to ensure candidate_passes ends around frame 185 and executed_pass starts at 195
        f_start = seg.get("cause_frame_idx", 93)
        tags = seg.get("visualCueTags") or seg.get("visual_cue_tags") or []
        for t in tags:
            if t.get("type") in ("candidate_passes", "pass_options"):
                t["start_frame"] = f_start + 10 # 103
                t["duration_frames"] = 80
                t["end_frame"] = f_start + 90 # 183
        
        # Ensure executed_pass tag exists
        if not any(t.get("type") in ("executed_pass", "real_pass") for t in tags):
            tags.append({
                "type": "executed_pass",
                "target": "executed_pass",
                "start_frame": f_start + 96, # 189
                "duration_frames": 200,
                "end_frame": f_start + 308,
                "beat": "beat_2_cause"
            })
        seg["visualCueTags"] = tags
        seg["visual_cue_tags"] = tags
        print(f"Updated tags for {seg.get('id')}: {len(tags)} tags")

# Save updated remotion_render.json
with open(render_json, "w") as f:
    json.dump(payload, f, indent=2)

brain_dir = Path(r"C:/Users/mlkme/.gemini/antigravity/brain/99567d6e-95a0-4ce9-83c3-38f3d719fe29")

for test_frame, stage_name in [(130, "phase_a_options_frame130"), (240, "phase_b_executed_frame240")]:
    fname = f"verified_staged_{stage_name}.png"
    still_payload = {
        "compositionId": "TacticalMasterclass",
        "frame": test_frame,
        "outputFilename": fname,
        **payload
    }

    print(f"Requesting still render at frame {test_frame} ({stage_name})...")
    req = urllib.request.Request(
        "http://127.0.0.1:3008/still",
        data=json.dumps(still_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Server response:", res)

    out_p = Path("F:/DockerRenders") / fname
    if out_p.exists():
        brain_p = brain_dir / fname
        shutil.copy2(str(out_p), str(brain_p))
        print(f"Saved {stage_name} still to {brain_p} ({brain_p.stat().st_size} bytes)")
    else:
        print("Still file not found at", out_p)
