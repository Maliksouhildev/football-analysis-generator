"""
Montage Builder
Takes detected player bounding boxes from the local CV detector,
crops each player, and stitches them into a single labeled sprite sheet.
Enables single-call Vision LLM identification with zero rate-limit issues.
"""

import math
import os
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image, ImageDraw, ImageFont

class MontageBuilder:
    """Stitches player crops into a single labeled sprite sheet for Vision LLM ingestion."""

    def __init__(self, cell_width: int = 90, cell_height: int = 120, cols: int = 4):
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.cols = cols

    def create_montage(
        self,
        image_path: str,
        detected_players: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> Tuple[Image.Image, str]:
        """
        Generates a single labeled composite image of all player crops.
        Returns the PIL Image and the saved file path.
        """
        base_img = Image.open(image_path).convert("RGB")
        w_orig, h_orig = base_img.size

        num_players = len(detected_players)
        if num_players == 0:
            # Return blank placeholder
            blank = Image.new("RGB", (self.cell_width * 2, self.cell_height), color=(20, 20, 20))
            if output_path:
                blank.save(output_path, quality=90)
            return blank, output_path or ""

        cols = min(self.cols, num_players)
        rows = math.ceil(num_players / cols)

        pad = 8
        header_h = 24
        footer_h = 18

        total_cell_w = self.cell_width + pad * 2
        total_cell_h = self.cell_height + header_h + footer_h + pad * 2

        montage_w = cols * total_cell_w + pad
        montage_h = rows * total_cell_h + pad

        montage = Image.new("RGB", (montage_w, montage_h), color=(18, 22, 28))
        draw = ImageDraw.Draw(montage)

        for idx, p in enumerate(detected_players):
            col_idx = idx % cols
            row_idx = idx // cols

            cell_x = pad + col_idx * total_cell_w
            cell_y = pad + row_idx * total_cell_h

            # Draw cell card background
            draw.rounded_rectangle(
                [cell_x, cell_y, cell_x + total_cell_w - pad, cell_y + total_cell_h - pad],
                radius=4,
                fill=(28, 34, 44),
                outline=(50, 60, 75),
                width=1
            )

            # Crop player with 6px padding
            bbox = p.get("bbox_px")
            if not bbox and "bbox_norm" in p:
                bn = p["bbox_norm"]
                bbox = [bn[0] * h_orig / 1000.0, bn[1] * w_orig / 1000.0, bn[2] * h_orig / 1000.0, bn[3] * w_orig / 1000.0]

            y1, x1, y2, x2 = bbox
            pad_x = (x2 - x1) * 0.15
            pad_y = (y2 - y1) * 0.10

            crop_box = (
                max(0, int(x1 - pad_x)),
                max(0, int(y1 - pad_y)),
                min(w_orig, int(x2 + pad_x)),
                min(h_orig, int(y2 + pad_y))
            )

            crop = base_img.crop(crop_box)

            # Resize preserving aspect ratio to fit inside crop area
            target_crop_w = self.cell_width
            target_crop_h = self.cell_height
            crop.thumbnail((target_crop_w, target_crop_h), Image.Resampling.LANCZOS)
            cw, ch = crop.size

            # Center crop inside the cell
            paste_x = cell_x + pad + (target_crop_w - cw) // 2
            paste_y = cell_y + header_h + pad + (target_crop_h - ch) // 2
            montage.paste(crop, (paste_x, paste_y))

            # Draw Header ID Badge
            pid = p.get("id", f"P{idx+1}")
            draw.rounded_rectangle(
                [cell_x + pad, cell_y + 4, cell_x + total_cell_w - pad * 2, cell_y + header_h],
                radius=2,
                fill=(0, 229, 255, 230) if p.get("is_ball_carrier") else (40, 50, 65)
            )
            draw.text((cell_x + pad + 6, cell_y + 6), pid, fill=(10, 15, 20) if p.get("is_ball_carrier") else (255, 255, 255))

            # Draw Footer Kit Color Label
            kit_str = str(p.get("kit_color", ""))
            short_kit = kit_str.split("(")[0].strip()[:14]
            draw.text((cell_x + pad, cell_y + header_h + self.cell_height + pad + 2), short_kit, fill=(180, 190, 205))

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            montage.save(output_path, quality=92)

        return montage, output_path or ""
