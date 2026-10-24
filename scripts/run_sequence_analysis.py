"""
Run Multi-Frame Sequence Analysis on Real Match Broadcast Video
Analyzes and renders 5 continuous keyframes (T0 to T4) from raw_bundesliga_2e57b9.mp4:
- Frame 290: T0 Setup & Initiation
- Frame 310: T1 Pass Release
- Frame 330: T2 Midfield Channel Trajectory
- Frame 350: T3 Target Reception & First Touch
- Frame 370: T4 Progressive Attack & Carry
Also produces a continuous video clip with live visual overlays across all 80 frames.
"""

import os
import sys
import shutil
import cv2
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vision_pipeline.multi_frame_tracker import MultiFrameTracker
from vision_pipeline.tactical_renderer import TacticalRenderer

ARTIFACT_DIR = r"C:\Users\Yacine\.gemini\antigravity\brain\81cf05f0-9be0-412d-a2c5-1dcb5e0b9048"
VIDEO_PATH = r"C:\Users\Yacine\AppData\Local\Temp\real_match_footage\raw_bundesliga_2e57b9.mp4"
TEMP_FRAME_DIR = r"C:\Users\Yacine\AppData\Local\Temp\real_match_footage\sequence_frames"
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")

os.makedirs(TEMP_FRAME_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run():
    print(f"=== Multi-Frame Tactical Sequence Analysis ($T_0 \to T_4$) ===")
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video file not found at {VIDEO_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    key_frames = [290, 310, 330, 350, 370]
    frame_items = []

    print(f"1. Extracting {len(key_frames)} keyframes from broadcast video...")
    for f_idx in key_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Could not read frame {f_idx} from {VIDEO_PATH}")
        fpath = os.path.join(TEMP_FRAME_DIR, f"frame_{f_idx}.jpg")
        cv2.imwrite(fpath, frame)
        frame_items.append((f_idx, fpath))
        print(f"   - Extracted frame {f_idx} -> {fpath}")

    # Initialize Tracker and Tactical Renderer
    print(f"\n2. Running MultiFrameTracker with sub-pixel turf grounding and persistent MOT...")
    tracker = MultiFrameTracker(fps=fps)
    renderer = TacticalRenderer(output_dir=OUTPUT_DIR)

    tracking_results = tracker.track_sequence(frame_items, match_name="1. FC Köln vs Hertha BSC")
    stages = tracking_results["stages"]
    print(f"   - Successfully tracked {len(stages)} stages across {tracking_results['total_frames_tracked']} frames.")
    print(f"   - Key Actors: Passer Track #{tracking_results['key_actors']['passer_track_id']}, Receiver Track #{tracking_results['key_actors']['receiver_track_id']}")

    rendered_images = []
    print(f"\n3. Rendering broadcast tactical overlays for each stage ($T_0 \dots T_4$)...")
    for stage_data in stages:
        stage_idx = stage_data["stage_idx"]
        f_idx = stage_data["metrics"]["frame_idx"]
        frame_path = os.path.join(TEMP_FRAME_DIR, f"frame_{f_idx}.jpg")
        out_name = f"sequence_bundesliga_t{stage_idx}.png"

        rendered_path = renderer.render_sequence_stage(
            image_path=frame_path,
            stage_data=stage_data,
            output_filename=out_name
        )
        print(f"   - Rendered Stage T{stage_idx} ({stage_data['phase_name']}): {rendered_path}")
        rendered_images.append(rendered_path)

        # Copy to artifact directory for UI display
        if os.path.exists(ARTIFACT_DIR):
            artifact_target = os.path.join(ARTIFACT_DIR, out_name)
            shutil.copy2(rendered_path, artifact_target)
            print(f"     -> Copied to artifact: {artifact_target}")

    # Generate continuous video clip from frame 290 to 370 with tracking overlays
    print(f"\n4. Generating continuous 25fps video clip (frames 290 to 370) with live tracking overlays...")
    video_out_path = os.path.join(OUTPUT_DIR, "sequence_bundesliga_tracked.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_out_path, fourcc, 25.0, (1920, 1080))

    # Dense tracking over 80 frames
    dense_frames = []
    for f in range(290, 371):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if ret:
            dense_frames.append((f, frame))

    cap.release()

    # Interpolate stage positions for dense video
    for f_idx, frame in dense_frames:
        # Determine nearest stage
        nearest_stage = min(stages, key=lambda s: abs(s["metrics"]["frame_idx"] - f_idx))
        # Draw live HUD and rings on frame
        annotated = frame.copy()

        # Draw ball
        b_pos = nearest_stage["ball"]["position_px"]
        if b_pos:
            cv2.circle(annotated, (int(b_pos[0]), int(b_pos[1])), 10, (0, 215, 255), 2)
            cv2.circle(annotated, (int(b_pos[0]), int(b_pos[1])), 4, (255, 255, 255), -1)

        # Draw top 12 tracked players
        for p in nearest_stage["players"][:12]:
            fx, fy = int(p["foot_pos_px"][0]), int(p["foot_pos_px"][1])
            is_white = "White" in p["kit_color"]
            ring_col = (255, 255, 255) if is_white else (200, 50, 0)
            cv2.ellipse(annotated, (fx, fy), (28, 9), 0, 0, 360, ring_col, 2)
            # Label
            tag = f"#{p['number']} {p['name'][:10]}"
            cv2.rectangle(annotated, (fx - 40, fy - 52), (fx + 40, fy - 36), (15, 20, 30), -1)
            cv2.rectangle(annotated, (fx - 40, fy - 52), (fx + 40, fy - 36), (0, 229, 255), 1)
            cv2.putText(annotated, tag, (fx - 36, fy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Top banner
        phase_str = f"LIVE TRACKING | FRAME {f_idx} | {nearest_stage['phase_name'].upper()}"
        cv2.rectangle(annotated, (560, 16), (1360, 50), (10, 14, 23), -1)
        cv2.rectangle(annotated, (560, 16), (1360, 50), (0, 229, 255), 1)
        cv2.putText(annotated, phase_str, (580, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_video.write(annotated)

    out_video.release()
    print(f"   - Continuous tracked video successfully saved to: {video_out_path}")
    if os.path.exists(ARTIFACT_DIR):
        artifact_vid = os.path.join(ARTIFACT_DIR, "sequence_bundesliga_tracked.mp4")
        shutil.copy2(video_out_path, artifact_vid)
        print(f"     -> Copied video to artifact: {artifact_vid}")

    print("\n=== Multi-Frame Sequence Complete! ===")
    return rendered_images

if __name__ == "__main__":
    run()
