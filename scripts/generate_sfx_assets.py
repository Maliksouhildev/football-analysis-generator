"""
scripts/generate_sfx_assets.py
Synthesizes and exports all broadcast procedural SFX assets,
including slow_focus_swell.wav, into local, Docker, and Remotion asset directories.
"""

import os
import shutil
from pathlib import Path
from src.rendering.sfx_generator import ProceduralSFXGenerator

def main():
    target_dirs = [
        Path("renders/assets/audio/sfx").resolve(),
        Path("F:/DockerRenders/assets/audio/sfx").resolve(),
        Path("C:/Users/mlkme/Desktop/cursor-mikow/remotion-server/public/assets/audio/sfx").resolve(),
        Path("C:/Users/mlkme/Desktop/cursor-mikow/remotion-server/public/sfx").resolve(),
    ]

    # Generate into the primary local folder first
    primary_dir = Path("renders/assets/audio/sfx").resolve()
    generated = ProceduralSFXGenerator.generate_all_sfx(primary_dir)
    print(f"Generated {len(generated)} SFX files in {primary_dir}")

    # Copy to all destination directories
    for dest_dir in target_dirs:
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            for name, src_path in generated.items():
                dest_file = dest_dir / Path(src_path).name
                shutil.copy2(src_path, dest_file)
            print(f"Synced SFX assets to: {dest_dir}")
        except Exception as e:
            print(f"Notice: Could not write to {dest_dir}: {e}")

    print("SFX generation and deployment complete.")

if __name__ == "__main__":
    main()
