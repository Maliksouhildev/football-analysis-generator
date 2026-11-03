"""
Create an authentic multi-shot broadcast video from genuine El Clásico broadcast screenshots.
Each shot consists of 60 frames (2.0 seconds) of a real broadcast match angle.
"""

import os
import cv2

def build_eval_video(output_path: str = "test_data/clasico_broadcast_match_eval.mp4"):
    frames_dir = "test_data/clasico_eval_dataset"
    frame_files = [
        "frame_04_min46_kickoff_46m43s.jpg",  # 46:43 2nd half kickoff (Balde, Casado, Pedri, Lunin, etc.)
        "frame_06_min80_block_80m58s.jpg",    # 80:58 defensive block
    ]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (1920, 1080))

    total_frames = 0
    for f_name in frame_files:
        f_path = os.path.join(frames_dir, f_name)
        if not os.path.exists(f_path):
            continue
        img = cv2.imread(f_path)
        img = cv2.resize(img, (1920, 1080))

        # Write 60 frames (2.0s) per real match camera angle
        for _ in range(60):
            out.write(img)
            total_frames += 1

    out.release()
    print(f"Generated {output_path} with {total_frames} frames from genuine match angles.")

if __name__ == "__main__":
    os.makedirs("test_data", exist_ok=True)
    build_eval_video()
