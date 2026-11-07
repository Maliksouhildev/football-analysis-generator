"""
Generate a realistic 1080p broadcast video clip with simulated camera panning and a camera cut.
Uses actual El Clásico broadcast frames (46:43 and 80:58) to test:
- Camera Motion Compensation (GMC sparse optical flow)
- BoT-SORT tracking with stride=2 interpolation
- Shot Boundary Detection
- Pass 2 Multi-Angle Collage Batching & Identity Resolution
"""

import os
import cv2
import numpy as np

def generate_pan_video(
    frame1_path: str,
    frame2_path: str,
    output_path: str,
    fps: float = 30.0,
    shot1_duration_sec: float = 3.0,
    shot2_duration_sec: float = 3.0,
):
    f1 = cv2.imread(frame1_path)
    f2 = cv2.imread(frame2_path)
    if f1 is None or f2 is None:
        raise ValueError(f"Could not load input frames: {frame1_path}, {frame2_path}")

    # Standardize to 1080p
    f1 = cv2.resize(f1, (1920, 1080))
    f2 = cv2.resize(f2, (1920, 1080))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))

    shot1_frames = int(shot1_duration_sec * fps)
    shot2_frames = int(shot2_duration_sec * fps)

    # Shot 1: Simulated camera pan (shift left-to-right by 60 pixels)
    for i in range(shot1_frames):
        shift_x = int(60.0 * np.sin(i / 15.0))
        M = np.float32([[1, 0, shift_x], [0, 1, 0]])
        panned = cv2.warpAffine(f1, M, (1920, 1080), borderMode=cv2.BORDER_REFLECT)
        out.write(panned)

    # Shot 2: Camera cut to frame 2 with slow zoom-in (1.0 to 1.05)
    for i in range(shot2_frames):
        scale = 1.0 + 0.05 * (i / float(shot2_frames))
        cx, cy = 960, 540
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        zoomed = cv2.warpAffine(f2, M, (1920, 1080), borderMode=cv2.BORDER_REFLECT)
        out.write(zoomed)

    out.release()
    print(f"Generated synthetic test video: {output_path} ({shot1_frames + shot2_frames} frames)")

if __name__ == "__main__":
    os.makedirs("test_data", exist_ok=True)
    generate_pan_video(
        frame1_path="test_data/clasico_eval_dataset/frame_04_min46_kickoff_46m43s.jpg",
        frame2_path="test_data/clasico_eval_dataset/frame_06_min80_block_80m58s.jpg",
        output_path="test_data/synthetic_clasico_pan_test.mp4",
        shot1_duration_sec=3.0,
        shot2_duration_sec=3.0,
    )
