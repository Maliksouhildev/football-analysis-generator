"""
src/ingestion/preprocess_frames.py
High-Fidelity Freeze-Frame Preprocessing Engine for Tactical Remotion Broadcasts.

Callable via n8n CLI Execute Command or imported directly.
1. FFmpeg exact frame extraction (1080p).
2. 25% padded bounding-box cropping.
3. 4x AI Super-Resolution via Real-ESRGAN Vulkan CLI.
4. Background removal via rembg (birefnet-general / isnet-general-use) + pitch chroma suppression.
5. Dual-cutout output: 4K tight cutout + full 1920x1080 transparent AR occlusion canvas.
6. Depth map generation via Depth-Anything-V2 Gradio Client (with graceful gradient fallback).
7. Manifest sync with timeline_manifest.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("preprocess_frames")


class FramePreprocessor:
    """Production preprocessing pipeline for tactical freeze-frames and player cutouts."""

    def __init__(
        self,
        output_dir: Union[str, Path] = r"F:\DockerRenders\assets\preprocessed",
        realesrgan_dir: Optional[Union[str, Path]] = None,
        rembg_model: str = "birefnet-general",
        docker_sync: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.docker_sync = docker_sync

        # Real-ESRGAN portable executable resolution
        if realesrgan_dir:
            self.realesrgan_bin = Path(realesrgan_dir) / "realesrgan-ncnn-vulkan.exe"
        else:
            default_tool = Path(__file__).resolve().parent.parent.parent / "tools" / "realesrgan" / "realesrgan-ncnn-vulkan.exe"
            if default_tool.exists():
                self.realesrgan_bin = default_tool
            else:
                self.realesrgan_bin = Path(r"C:\Users\mlkme\.antigravity\workflows\127-0-0-1-5678\football automation\tools\realesrgan\realesrgan-ncnn-vulkan.exe")

        self.rembg_model = rembg_model
        self._rembg_session = None

    def _get_rembg_session(self):
        if self._rembg_session is None:
            try:
                import rembg
                # Try preferred birefnet-general, then isnet-general-use, then u2net
                for model_candidate in [self.rembg_model, "isnet-general-use", "u2net"]:
                    try:
                        self._rembg_session = rembg.new_session(model_candidate)
                        logger.info("Initialized rembg session with model: %s", model_candidate)
                        break
                    except Exception as err:
                        logger.warning("Could not initialize rembg model '%s': %s", model_candidate, err)
                if self._rembg_session is None:
                    self._rembg_session = rembg.new_session()
            except Exception as e:
                logger.error("Failed to load rembg: %s", e)
        return self._rembg_session

    def extract_frame(
        self,
        video_path: Union[str, Path],
        timestamp_or_frame: Union[float, int, str],
        out_path: Path,
        fps: float = 30.0,
    ) -> Path:
        """Step A: Uses FFmpeg exact seeking to extract 1920x1080 frame."""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Source video not found: {video_path}")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(timestamp_or_frame, int) or (isinstance(timestamp_or_frame, str) and timestamp_or_frame.isdigit()):
            frame_num = int(timestamp_or_frame)
            time_sec = max(0.0, frame_num / fps)
        else:
            time_sec = float(timestamp_or_frame)

        # Fast and accurate seek with -ss before -i and 1 frame output
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", f"{time_sec:.4f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(out_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0 or not out_path.exists():
            # OpenCV fallback
            cap = cv2.VideoCapture(str(video_path))
            if isinstance(timestamp_or_frame, int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, timestamp_or_frame)
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                raise RuntimeError(f"FFmpeg and OpenCV both failed to extract frame from {video_path} at {timestamp_or_frame}")
            cv2.imwrite(str(out_path), frame)

        logger.info("Step A: Extracted base frame -> %s", out_path.name)
        return out_path

    def crop_player(
        self,
        frame_path: Path,
        bbox: Union[Dict[str, Any], List[Union[int, float]]],
        out_path: Path,
        padding_pct: float = 0.25,
    ) -> Tuple[Path, Tuple[int, int, int, int]]:
        """
        Step B: Crops bounding box with 25% padding margin.
        Returns: (cropped_file_path, (padded_x, padded_y, padded_w, padded_h))
        """
        img = Image.open(frame_path).convert("RGB")
        img_w, img_h = img.size

        # Parse bbox formats: {x, y, width, height} or [x, y, w, h] or {xmin, ymin, xmax, ymax}
        if isinstance(bbox, dict):
            if "width" in bbox and "height" in bbox:
                bx, by, bw, bh = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            elif "xmax" in bbox and "ymax" in bbox:
                bx, by = bbox["xmin"], bbox["ymin"]
                bw, bh = bbox["xmax"] - bx, bbox["ymax"] - by
            else:
                raise ValueError(f"Unrecognized bbox dictionary structure: {bbox}")
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            bx, by, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]
        else:
            raise ValueError(f"Invalid bbox format: {bbox}")

        # Convert normalized percentage (0..1 or 0..100) to absolute pixels if needed
        if 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0:
            bx *= img_w
            by *= img_h
            bw *= img_w
            bh *= img_h
        elif bw > 1.0 and bw <= 100.0 and bh > 1.0 and bh <= 100.0 and bx < 100.0 and by < 100.0 and img_w > 100:
            bx = (bx / 100.0) * img_w
            by = (by / 100.0) * img_h
            bw = (bw / 100.0) * img_w
            bh = (bh / 100.0) * img_h

        # Apply +25% padding margin
        pad_w = bw * padding_pct
        pad_h = bh * padding_pct

        crop_x1 = max(0, int(round(bx - pad_w)))
        crop_y1 = max(0, int(round(by - pad_h)))
        crop_x2 = min(img_w, int(round(bx + bw + pad_w)))
        crop_y2 = min(img_h, int(round(by + bh + pad_h)))

        cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(out_path, format="PNG")

        padded_box = (crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1)
        logger.info("Step B: Cropped bbox with 25%% padding (%dx%d) -> %s", cropped.width, cropped.height, out_path.name)
        return out_path, padded_box

    def upscale_4x(self, crop_path: Path, out_path: Path) -> Path:
        """Step C: Upscales cropped bounding box 4x using Real-ESRGAN Vulkan CLI."""
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if self.realesrgan_bin and self.realesrgan_bin.exists():
            cmd = [
                str(self.realesrgan_bin),
                "-i", str(crop_path),
                "-o", str(out_path),
                "-n", "realesrgan-x4plus",
                "-s", "4",
            ]
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90)
                if proc.returncode == 0 and out_path.exists():
                    logger.info("Step C: 4x AI Super-Resolution via Real-ESRGAN succeeded -> %s", out_path.name)
                    return out_path
            except Exception as e:
                logger.warning("Real-ESRGAN execution failed: %s; falling back to Lanczos-4", e)

        # High-quality Lanczos-4 fallback
        img = Image.open(crop_path)
        upscaled = img.resize((img.width * 4, img.height * 4), Image.Resampling.LANCZOS)
        upscaled.save(out_path, format="PNG")
        logger.info("Step C: 4x Lanczos-4 upscaling completed -> %s", out_path.name)
        return out_path

    def isolate_player(
        self,
        crop_4k_path: Path,
        crop_box_original: Tuple[int, int, int, int],
        canvas_dims: Tuple[int, int],
        out_4k_path: Path,
        out_full_path: Path,
    ) -> Tuple[Path, Path]:
        """
        Step D: Background removal via rembg + pitch chroma suppression on boots.
        Produces both:
        1. cutout_4k_path (tight transparent 4K cutout).
        2. cutout_full_path (full 1920x1080 transparent canvas).
        """
        import rembg

        session = self._get_rembg_session()
        with open(crop_4k_path, "rb") as f:
            input_bytes = f.read()

        # Direct neural network removal (BiRefNet outputs continuous soft alpha without pymatting memory explosion)
        output_bytes = rembg.remove(input_bytes, session=session)
        with open(out_4k_path, "wb") as f:
            f.write(output_bytes)

        # Green Grass Chroma Suppressor around boots
        cutout_img = Image.open(out_4k_path).convert("RGBA")
        np_arr = np.array(cutout_img)
        # Suppress spill: where alpha > 0 and Green is dominant over Red and Blue (grass fringe)
        r, g, b, a = np_arr[:, :, 0], np_arr[:, :, 1], np_arr[:, :, 2], np_arr[:, :, 3]
        green_mask = (g.astype(int) > r.astype(int) + 25) & (g.astype(int) > b.astype(int) + 25) & (a > 30)
        # Desaturate green fringe to neutral turf shadow
        np_arr[green_mask, 1] = np.maximum(r[green_mask], b[green_mask])
        cleaned_cutout = Image.fromarray(np_arr)
        cleaned_cutout.save(out_4k_path, format="PNG")

        # Composite cutout back into full 1920x1080 transparent canvas
        orig_x, orig_y, orig_w, orig_h = crop_box_original
        full_canvas = Image.new("RGBA", canvas_dims, (0, 0, 0, 0))
        # Downscale cutout back to exact crop box dimensions for the full 1080p canvas
        cutout_for_canvas = cleaned_cutout.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        full_canvas.paste(cutout_for_canvas, (orig_x, orig_y), cutout_for_canvas)
        full_canvas.save(out_full_path, format="PNG")

        logger.info("Step D: Isolated player cutouts generated -> %s and %s", out_4k_path.name, out_full_path.name)
        return out_4k_path, out_full_path

    def generate_depth_map(self, frame_path: Path, out_path: Path) -> Path:
        """Step E: Depth map generation via Depth-Anything-V2 Gradio client with pitch gradient fallback."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        depth_generated = False

        try:
            from gradio_client import Client, handle_file
            client = Client("depth-anything/Depth-Anything-V2")
            result = client.predict(
                image=handle_file(str(frame_path)),
                api_name="/on_submit"
            )
            depth_file = result[0] if isinstance(result, (list, tuple)) else result
            if depth_file and os.path.exists(depth_file):
                shutil.copy2(depth_file, out_path)
                depth_generated = True
                logger.info("Step E: Depth map generated via Depth-Anything-V2 -> %s", out_path.name)
        except Exception as e:
            logger.warning("Gradio Depth-Anything-V2 query skipped/failed (%s); applying pitch-gradient depth fallback", e)

        if not depth_generated:
            # Broadcast Pitch Depth Fallback: Linear vertical depth gradient (stands far = darker, foreground turf = brighter)
            img = cv2.imread(str(frame_path))
            h, w = img.shape[:2]
            gradient = np.linspace(30, 245, h, dtype=np.uint8)
            depth_map = np.tile(gradient[:, np.newaxis], (1, w))
            # Blur to soften edges
            depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)
            cv2.imwrite(str(out_path), depth_map)
            logger.info("Step E: Pitch depth gradient fallback generated -> %s", out_path.name)

        return out_path

    def sync_to_docker(self, local_path: Path, rel_subpath: str) -> None:
        """Syncs the preprocessed assets directly into the Remotion Docker container and public directory."""
        if not self.docker_sync or not local_path.exists():
            return
        local_public = Path(r"C:\Users\mlkme\Desktop\cursor-mikow\remotion-server\public\assets\preprocessed")
        try:
            local_public.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, local_public / local_path.name)
        except Exception:
            pass

        try:
            subprocess.run(
                ["docker", "cp", str(local_path), f"cursor-mikow-remotion:/tmp/assets/preprocessed/{local_path.name}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except Exception:
            pass

    def process_item(
        self,
        item: Dict[str, Any],
        video_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Executes end-to-end preprocessing for a single freeze event.
        item: {
            "id": "freeze_01",
            "timestamp": 14.5, // or "frame": 435
            "bbox": {"x": 640, "y": 320, "width": 180, "height": 340},
            "player_name": "Lamine Yamal"
        }
        """
        event_id = str(item.get("id") or "event_0")
        timestamp = item.get("timestamp") or item.get("time") or item.get("frame", 0)
        bbox = item.get("bbox") or item.get("bounding_box")

        frame_path = self.output_dir / f"frame_{event_id}.png"
        crop_path = self.output_dir / f"crop_{event_id}.png"
        crop_4k_path = self.output_dir / f"crop_{event_id}_4k.png"
        cutout_4k_path = self.output_dir / f"cutout_{event_id}_4k.png"
        cutout_full_path = self.output_dir / f"cutout_{event_id}_full.png"
        depth_path = self.output_dir / f"depth_{event_id}.png"

        # Step A: Exact frame extraction
        self.extract_frame(video_path, timestamp, frame_path)
        with Image.open(frame_path) as im:
            canvas_dims = im.size

        # Step B: 25% padded crop
        _, crop_box = self.crop_player(frame_path, bbox, crop_path, padding_pct=0.25)

        # Step C: 4x AI Super-Resolution
        self.upscale_4x(crop_path, crop_4k_path)

        # Step D: Background removal & full canvas composite
        self.isolate_player(crop_4k_path, crop_box, canvas_dims, cutout_4k_path, cutout_full_path)

        # Step E: Depth map
        self.generate_depth_map(frame_path, depth_path)

        # Step F: Sync assets
        for p in [frame_path, crop_path, crop_4k_path, cutout_4k_path, cutout_full_path, depth_path]:
            self.sync_to_docker(p, f"preprocessed/{p.name}")

        result = {
            "id": event_id,
            "status": "success",
            "baseFramePng": str(frame_path),
            "baseFrameUrl": f"http://127.0.0.1:3000/assets/preprocessed/{frame_path.name}",
            "cutout4kPng": str(cutout_4k_path),
            "cutout4kUrl": f"http://127.0.0.1:3000/assets/preprocessed/{cutout_4k_path.name}",
            "cutoutFullPng": str(cutout_full_path),
            "cutoutFullUrl": f"http://127.0.0.1:3000/assets/preprocessed/{cutout_full_path.name}",
            "depthMapPng": str(depth_path),
            "depthMapUrl": f"http://127.0.0.1:3000/assets/preprocessed/{depth_path.name}",
            "cropBox": {
                "x": crop_box[0],
                "y": crop_box[1],
                "width": crop_box[2],
                "height": crop_box[3],
            },
        }
        return result

    def process_manifest(
        self,
        manifest_path: Union[str, Path],
        video_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Updates timeline_manifest.json with preprocessed asset URLs."""
        manifest_file = Path(manifest_path)
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        actual_video = video_path or manifest_data.get("videoFile") or manifest_data.get("video_path")
        if not actual_video:
            raise ValueError("No video file specified in arguments or manifest")

        items = manifest_data.get("freezeEvents") or manifest_data.get("segments") or []
        processed_items = []
        for idx, item in enumerate(items):
            if "bbox" in item or "bounding_box" in item:
                if "id" not in item:
                    item["id"] = f"event_{idx+1}"
                res = self.process_item(item, actual_video)
                item["baseFrameUrl"] = res["baseFrameUrl"]
                item["cutoutUrl"] = res["cutoutFullUrl"]
                item["cutout4kUrl"] = res["cutout4kUrl"]
                item["depthMapUrl"] = res["depthMapUrl"]
                processed_items.append(res)

        # Write updated manifest back
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        return {
            "status": "success",
            "manifest": str(manifest_file),
            "processedCount": len(processed_items),
            "results": processed_items,
        }


def main():
    parser = argparse.ArgumentParser(description="Football Freeze-Frame 4K AI Preprocessor")
    parser.add_argument("--video", type=str, required=True, help="Path to source match video (.mp4)")
    parser.add_argument("--manifest", type=str, default=None, help="Path to timeline_manifest.json")
    parser.add_argument("--id", type=str, default="freeze_01", help="Event ID")
    parser.add_argument("--timestamp", type=str, default=None, help="Timestamp in seconds or HH:MM:SS.mmm")
    parser.add_argument("--frame", type=int, default=None, help="Frame index")
    parser.add_argument("--bbox", type=str, default=None, help="JSON bounding box string: '{\"x\":640,\"y\":320,\"width\":180,\"height\":340}'")
    parser.add_argument("--output-dir", type=str, default=r"F:\DockerRenders\assets\preprocessed", help="Output directory")
    args = parser.parse_args()

    preprocessor = FramePreprocessor(output_dir=args.output_dir)

    if args.manifest:
        result = preprocessor.process_manifest(args.manifest, video_path=args.video)
        print(json.dumps(result, indent=2))
        return

    # Single item processing
    if not args.bbox:
        print(json.dumps({"error": "Either --manifest or --bbox must be provided"}, indent=2))
        sys.exit(1)

    parsed_bbox = json.loads(args.bbox)
    item = {
        "id": args.id,
        "timestamp": float(args.timestamp) if args.timestamp else (args.frame or 0),
        "bbox": parsed_bbox,
    }
    result = preprocessor.process_item(item, video_path=args.video)
    # Output JSON for n8n consumption
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
