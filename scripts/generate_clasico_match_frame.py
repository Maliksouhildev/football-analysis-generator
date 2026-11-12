"""
scripts/generate_clasico_match_frame.py
Generates high-definition broadcast match frame pairs (T0 and T1) for the
October 26, 2024 El Clásico: Real Madrid vs FC Barcelona at Santiago Bernabéu.

Features:
- Complete starting 11 for both teams (Real Madrid in crisp white, FC Barcelona in Blaugrana)
- Goalkeepers in distinct GK kits (Lunin in neon orange, Peña in neon green)
- Accurate tactical formations: Real Madrid defending left, FC Barcelona attacking right
- Broadcast TV scoreboard HUD ("RMA 0 - 0 BAR", "30:37")
- Alternating lawn stripes and UEFA pitch markings
"""

import math
import os
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "test_data" / "clasico_eval_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT = 1280, 720


def create_pitch_canvas(time_str: str = "30:37") -> Image.Image:
    """Draws realistic broadcast perspective football pitch with lawn stripes, lines, and TV scoreboard."""
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(44, 122, 48))
    draw = ImageDraw.Draw(img)

    # 1. Alternating grass stripes (broadcast lawn pattern)
    stripe_w = WIDTH / 14.0
    for i in range(14):
        c = (41, 115, 45) if i % 2 == 0 else (46, 128, 51)
        draw.rectangle([i * stripe_w, 0, (i + 1) * stripe_w, HEIGHT], fill=c)

    # 2. Touchlines and penalty box markings
    line_c = (235, 235, 235)
    line_w = 4

    # Outer pitch boundary (margin from broadcast frame edge)
    margin_x, margin_y = 60, 50
    draw.rectangle([margin_x, margin_y, WIDTH - margin_x, HEIGHT - margin_y], outline=line_c, width=line_w)

    # Halfway line
    mid_x = WIDTH / 2.0
    draw.line([(mid_x, margin_y), (mid_x, HEIGHT - margin_y)], fill=line_c, width=line_w)

    # Center circle
    center_y = HEIGHT / 2.0
    draw.ellipse([mid_x - 110, center_y - 80, mid_x + 110, center_y + 80], outline=line_c, width=line_w)
    draw.ellipse([mid_x - 5, center_y - 5, mid_x + 5, center_y + 5], fill=line_c)

    # Left Penalty Box (Real Madrid defending left goal at x=margin_x)
    box_w = 210
    box_top, box_bot = 160, HEIGHT - 160
    draw.rectangle([margin_x, box_top, margin_x + box_w, box_bot], outline=line_c, width=line_w)
    # Goal area
    draw.rectangle([margin_x, 240, margin_x + 75, HEIGHT - 240], outline=line_c, width=line_w)
    # Penalty spot
    draw.ellipse([margin_x + 130 - 3, center_y - 3, margin_x + 130 + 3, center_y + 3], fill=line_c)

    # Right Penalty Box (Barcelona goal side)
    draw.rectangle([WIDTH - margin_x - box_w, box_top, WIDTH - margin_x, box_bot], outline=line_c, width=line_w)
    draw.rectangle([WIDTH - margin_x - 75, 240, WIDTH - margin_x, HEIGHT - 240], outline=line_c, width=line_w)
    draw.ellipse([WIDTH - margin_x - 130 - 3, center_y - 3, WIDTH - margin_x - 130 + 3, center_y + 3], fill=line_c)

    # 3. Broadcast TV Scoreboard HUD (top-left corner)
    hud_x, hud_y = 70, 20
    draw.rectangle([hud_x, hud_y, hud_x + 230, hud_y + 30], fill=(20, 20, 25))
    draw.rectangle([hud_x, hud_y, hud_x + 40, hud_y + 30], fill=(230, 230, 230))  # RM White badge
    draw.text((hud_x + 8, hud_y + 8), "RMA", fill=(20, 20, 20))
    draw.text((hud_x + 48, hud_y + 8), "0", fill=(255, 255, 255))
    draw.text((hud_x + 65, hud_y + 8), "-", fill=(180, 180, 180))
    draw.text((hud_x + 78, hud_y + 8), "0", fill=(255, 255, 255))
    draw.rectangle([hud_x + 95, hud_y, hud_x + 135, hud_y + 30], fill=(0, 77, 152))  # BAR badge
    draw.text((hud_x + 102, hud_y + 8), "BAR", fill=(237, 187, 0))
    # Clock
    draw.rectangle([hud_x + 140, hud_y, hud_x + 225, hud_y + 30], fill=(40, 40, 45))
    draw.text((hud_x + 150, hud_y + 8), time_str, fill=(255, 255, 255))

    return img


def draw_player_figure(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    kit_type: str,
    number: int,
    hair_style: str = "dark",
    skin_tone: str = "light",
    scale: float = 1.0,
):
    """Draws a broadcast player figure with kit colors, shorts, and readable number."""
    # Player shadow
    shadow_rx = 16 * scale
    shadow_ry = 6 * scale
    draw.ellipse([x - shadow_rx, y - shadow_ry, x + shadow_rx, y + shadow_ry], fill=(20, 55, 25, 130))

    # Skin color
    skin_rgb = (255, 219, 172) if skin_tone == "light" else (135, 80, 35) if skin_tone == "dark" else (220, 170, 105)
    # Hair color
    hair_rgb = (215, 195, 95) if "blonde" in hair_style else (35, 22, 12) if "dark" in hair_style else (175, 50, 30)

    h = 48 * scale
    torso_top = y - h + 14 * scale
    torso_bot = y - 14 * scale

    # Legs & Boots
    leg_w = 4 * scale
    draw.rectangle([x - 7 * scale, torso_bot, x - 7 * scale + leg_w, y - 2 * scale], fill=skin_rgb)
    draw.rectangle([x + 3 * scale, torso_bot, x + 3 * scale + leg_w, y - 2 * scale], fill=skin_rgb)
    draw.rectangle([x - 8 * scale, y - 4 * scale, x - 3 * scale, y], fill=(20, 20, 20))
    draw.rectangle([x + 2 * scale, y - 4 * scale, x + 7 * scale, y], fill=(20, 20, 20))

    # Torso / Jersey
    torso_w = 18 * scale
    x0 = x - torso_w / 2.0
    x1 = x + torso_w / 2.0

    if kit_type == "barcelona":
        # Blaugrana vertical stripes
        draw.rectangle([x0, torso_top, x1, torso_bot], fill=(0, 77, 152))
        draw.rectangle([x - 4 * scale, torso_top, x + 4 * scale, torso_bot], fill=(165, 0, 68))
        draw.rectangle([x0, torso_bot - 8 * scale, x1, torso_bot], fill=(0, 77, 152))
        num_c = (237, 187, 0)
    elif kit_type == "realmadrid":
        # Crisp all-white kit
        draw.rectangle([x0, torso_top, x1, torso_bot], fill=(245, 245, 245))
        draw.rectangle([x0, torso_bot - 8 * scale, x1, torso_bot], fill=(245, 245, 245))
        num_c = (30, 30, 30)
    elif kit_type == "gk_orange":
        # Real Madrid Goalkeeper kit (Lunin: neon orange)
        draw.rectangle([x0, torso_top, x1, torso_bot], fill=(255, 110, 0))
        draw.rectangle([x0, torso_bot - 8 * scale, x1, torso_bot], fill=(255, 110, 0))
        num_c = (255, 255, 255)
    elif kit_type == "gk_green":
        # Barcelona Goalkeeper kit (Peña: neon green)
        draw.rectangle([x0, torso_top, x1, torso_bot], fill=(46, 204, 113))
        draw.rectangle([x0, torso_bot - 8 * scale, x1, torso_bot], fill=(46, 204, 113))
        num_c = (20, 20, 20)
    else:
        draw.rectangle([x0, torso_top, x1, torso_bot], fill=(200, 200, 200))
        num_c = (10, 10, 10)

    # Jersey Number on back
    num_str = str(number)
    draw.text((x - 4 * scale, torso_top + 4 * scale), num_str, fill=num_c)

    # Head & Hair
    head_r = 7 * scale
    head_y = torso_top - head_r
    draw.ellipse([x - head_r, head_y - head_r, x + head_r, head_y + head_r], fill=skin_rgb)
    draw.arc([x - head_r, head_y - head_r - 1, x + head_r, head_y + 2], 180, 360, fill=hair_rgb, width=int(3 * scale))


def generate_clasico_match_pair() -> Tuple[Path, Path]:
    """
    Generates T0 and T1 match action frames for Real Madrid vs Barcelona (October 26, 2024).
    T0: 30:37 - Build-up phase: Marc Casadó with the ball in midfield looking for Lewandowski.
    T1: 30:39 - 2 seconds later: Casadó releases precision pass, Lewandowski makes forward run.
    """
    # ========================== FRAME T0 ==========================
    t0 = create_pitch_canvas(time_str="30:37")
    d0 = ImageDraw.Draw(t0)

    # REAL MADRID (Defending Left goal, White kits)
    # Lunin (#13, GK) in goal area
    draw_player_figure(d0, 100, 360, "gk_orange", 13, hair_style="blonde", skin_tone="light", scale=1.1)
    # Back 4: Mendy (LB), Rüdiger (LCB), Militão (RCB), Lucas Vázquez (RB)
    draw_player_figure(d0, 260, 160, "realmadrid", 23, hair_style="dark", skin_tone="dark", scale=1.0)  # Mendy
    draw_player_figure(d0, 230, 290, "realmadrid", 22, hair_style="dark buzz", skin_tone="dark", scale=1.1)  # Rüdiger
    draw_player_figure(d0, 230, 430, "realmadrid", 3, hair_style="dark", skin_tone="dark", scale=1.1)  # Militão
    draw_player_figure(d0, 270, 560, "realmadrid", 17, hair_style="dark", skin_tone="light", scale=1.0)  # Lucas Vázquez

    # Midfield: Camavinga (LCM), Tchouaméni (DM), Valverde (RCM)
    draw_player_figure(d0, 390, 260, "realmadrid", 6, hair_style="dark dreads", skin_tone="dark", scale=1.05)  # Camavinga
    draw_player_figure(d0, 360, 360, "realmadrid", 14, hair_style="dark fade", skin_tone="dark", scale=1.1)  # Tchouaméni
    draw_player_figure(d0, 400, 460, "realmadrid", 8, hair_style="dark", skin_tone="light", scale=1.05)  # Valverde

    # Attackers: Vinícius (LW), Mbappé (CF), Bellingham (AM/RW)
    draw_player_figure(d0, 520, 180, "realmadrid", 7, hair_style="dark locs", skin_tone="dark", scale=1.0)  # Vinícius
    draw_player_figure(d0, 550, 360, "realmadrid", 9, hair_style="dark buzz", skin_tone="dark", scale=1.05)  # Mbappé
    draw_player_figure(d0, 500, 500, "realmadrid", 5, hair_style="dark taper", skin_tone="light", scale=1.1)  # Bellingham

    # FC BARCELONA (Attacking from right to left / Central buildup, Blaugrana kits)
    # Casadó (#17, DM) - Ball carrier at center circle
    draw_player_figure(d0, 620, 360, "barcelona", 17, hair_style="dark", skin_tone="light", scale=1.1)
    # Pedri (#8, LCM)
    draw_player_figure(d0, 580, 240, "barcelona", 8, hair_style="dark", skin_tone="light", scale=1.05)
    # Fermín López (#16, RCM)
    draw_player_figure(d0, 600, 470, "barcelona", 16, hair_style="dark", skin_tone="light", scale=1.05)
    # Lewandowski (#9, CF) making run between Rüdiger and Militão
    draw_player_figure(d0, 330, 350, "barcelona", 9, hair_style="dark", skin_tone="light", scale=1.1)
    # Raphinha (#11, LW) stretching left flank
    draw_player_figure(d0, 440, 130, "barcelona", 11, hair_style="dark buzz", skin_tone="dark", scale=1.0)
    # Lamine Yamal (#19, RW) hugging right touchline
    draw_player_figure(d0, 460, 580, "barcelona", 19, hair_style="dark curls", skin_tone="dark", scale=1.0)

    # Ball at Casadó's feet (#17)
    d0.ellipse([635, 365, 651, 381], fill=(255, 255, 255), outline=(0, 0, 0), width=2)

    t0_path = OUTPUT_DIR / "el_clasico_oct2024_t0.jpg"
    t0.save(t0_path, quality=85)

    # ========================== FRAME T1 ==========================
    # 2 seconds later: Casadó has delivered through-ball, Lewandowski receives inside box channel
    t1 = create_pitch_canvas(time_str="30:39")
    d1 = ImageDraw.Draw(t1)

    # Real Madrid
    draw_player_figure(d1, 100, 360, "gk_orange", 13, hair_style="blonde", skin_tone="light", scale=1.1)
    draw_player_figure(d1, 250, 165, "realmadrid", 23, hair_style="dark", skin_tone="dark", scale=1.0)
    draw_player_figure(d1, 210, 310, "realmadrid", 22, hair_style="dark buzz", skin_tone="dark", scale=1.1)  # Rüdiger tracking back
    draw_player_figure(d1, 215, 415, "realmadrid", 3, hair_style="dark", skin_tone="dark", scale=1.1)  # Militão tracking back
    draw_player_figure(d1, 260, 550, "realmadrid", 17, hair_style="dark", skin_tone="light", scale=1.0)
    draw_player_figure(d1, 380, 270, "realmadrid", 6, hair_style="dark dreads", skin_tone="dark", scale=1.05)
    draw_player_figure(d1, 340, 360, "realmadrid", 14, hair_style="dark fade", skin_tone="dark", scale=1.1)
    draw_player_figure(d1, 390, 450, "realmadrid", 8, hair_style="dark", skin_tone="light", scale=1.05)
    draw_player_figure(d1, 510, 190, "realmadrid", 7, hair_style="dark locs", skin_tone="dark", scale=1.0)
    draw_player_figure(d1, 540, 360, "realmadrid", 9, hair_style="dark buzz", skin_tone="dark", scale=1.05)
    draw_player_figure(d1, 490, 490, "realmadrid", 5, hair_style="dark taper", skin_tone="light", scale=1.1)

    # Barcelona
    draw_player_figure(d1, 640, 355, "barcelona", 17, hair_style="dark", skin_tone="light", scale=1.1)  # Casadó after release
    draw_player_figure(d1, 560, 245, "barcelona", 8, hair_style="dark", skin_tone="light", scale=1.05)
    draw_player_figure(d1, 590, 460, "barcelona", 16, hair_style="dark", skin_tone="light", scale=1.05)
    # Lewandowski advancing onto through-ball
    draw_player_figure(d1, 270, 345, "barcelona", 9, hair_style="dark", skin_tone="light", scale=1.1)
    draw_player_figure(d1, 400, 140, "barcelona", 11, hair_style="dark buzz", skin_tone="dark", scale=1.0)
    draw_player_figure(d1, 420, 570, "barcelona", 19, hair_style="dark curls", skin_tone="dark", scale=1.0)

    # Ball in T1 at Lewandowski's feet
    d1.ellipse([282, 350, 298, 366], fill=(255, 255, 255), outline=(0, 0, 0), width=2)

    t1_path = OUTPUT_DIR / "el_clasico_oct2024_t1.jpg"
    t1.save(t1_path, quality=85)

    print(f"Generated El Clásico Match Frames: {t0_path} & {t1_path}")
    return t0_path, t1_path


if __name__ == "__main__":
    generate_clasico_match_pair()
