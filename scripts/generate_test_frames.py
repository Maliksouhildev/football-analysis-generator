"""
Generates high-definition, broadcast-style tactical match frame pairs (T0 and T1)
with authentic pitch textures, field markings, team kits, jersey numbers, and tactical positioning.
"""

import math
import os
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WIDTH, HEIGHT = 1280, 720

def create_pitch_canvas() -> Image.Image:
    """Draws realistic broadcast perspective football pitch with alternating grass stripes and lines."""
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(46, 125, 50))
    draw = ImageDraw.Draw(img)

    # Alternating grass stripes (broadcast lawn pattern)
    stripe_w = WIDTH / 14.0
    for i in range(14):
        c = (43, 118, 47) if i % 2 == 0 else (48, 133, 53)
        draw.rectangle([i * stripe_w, 0, (i + 1) * stripe_w, HEIGHT], fill=c)

    # Touchlines and penalty box markings
    line_c = (235, 235, 235)
    line_w = 4

    # Outer pitch boundary
    margin_x, margin_y = 60, 50
    draw.rectangle([margin_x, margin_y, WIDTH - margin_x, HEIGHT - margin_y], outline=line_c, width=line_w)

    # Halfway line
    mid_x = WIDTH / 2.0
    draw.line([(mid_x, margin_y), (mid_x, HEIGHT - margin_y)], fill=line_c, width=line_w)

    # Center circle
    center_y = HEIGHT / 2.0
    draw.ellipse([mid_x - 110, center_y - 80, mid_x + 110, center_y + 80], outline=line_c, width=line_w)
    draw.ellipse([mid_x - 5, center_y - 5, mid_x + 5, center_y + 5], fill=line_c)

    # Left Penalty Box (Galatasaray defending / Barcelona attacking from left or right)
    box_w = 220
    box_top, box_bot = 160, HEIGHT - 160
    draw.rectangle([margin_x, box_top, margin_x + box_w, box_bot], outline=line_c, width=line_w)
    # Goal area
    draw.rectangle([margin_x, 240, margin_x + 80, HEIGHT - 240], outline=line_c, width=line_w)

    # Right Penalty Box
    draw.rectangle([WIDTH - margin_x - box_w, box_top, WIDTH - margin_x, box_bot], outline=line_c, width=line_w)
    draw.rectangle([WIDTH - margin_x - 80, 240, WIDTH - margin_x, HEIGHT - 240], outline=line_c, width=line_w)

    return img

def draw_player_figure(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    kit_type: str,
    number: int,
    hair_style: str = "dark",
    skin_tone: str = "light",
    scale: float = 1.0
):
    """Draws a high-fidelity broadcast player figure with kit stripes, shorts, and readable number."""
    # Player shadow
    shadow_rx = 16 * scale
    shadow_ry = 6 * scale
    draw.ellipse([x - shadow_rx, y - shadow_ry, x + shadow_rx, y + shadow_ry], fill=(20, 60, 25, 120))

    # Skin color
    skin_rgb = (255, 219, 172) if skin_tone == "light" else (141, 85, 36) if skin_tone == "dark" else (224, 172, 105)
    # Hair color
    hair_rgb = (220, 200, 100) if "blonde" in hair_style else (40, 25, 15) if "dark" in hair_style else (180, 50, 30)

    h = 48 * scale
    torso_top = y - h + 14 * scale
    torso_bot = y - 14 * scale

    # Legs & Boots
    leg_w = 4 * scale
    draw.rectangle([x - 7 * scale, torso_bot, x - 7 * scale + leg_w, y - 2 * scale], fill=skin_rgb)
    draw.rectangle([x + 3 * scale, torso_bot, x + 3 * scale + leg_w, y - 2 * scale], fill=skin_rgb)
    # Boots
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
        # Shorts (Blue)
        draw.rectangle([x0, torso_bot - 8 * scale, x1, torso_bot], fill=(0, 77, 152))
        num_c = (237, 187, 0)
    elif kit_type == "galatasaray":
        # Yellow and Red halves
        draw.rectangle([x0, torso_top, x, torso_bot], fill=(253, 185, 18))
        draw.rectangle([x, torso_top, x1, torso_bot], fill=(169, 4, 50))
        # Shorts (Red)
        draw.rectangle([x0, torso_bot - 8 * scale, x1, torso_bot], fill=(169, 4, 50))
        num_c = (255, 255, 255)
    elif kit_type == "realmadrid":
        draw.rectangle([x0, torso_top, x1, torso_bot], fill=(245, 245, 245))
        draw.rectangle([x0, torso_bot - 8 * scale, x1, torso_bot], fill=(245, 245, 245))
        num_c = (30, 30, 30)
    elif kit_type == "mancity":
        draw.rectangle([x0, torso_top, x1, torso_bot], fill=(108, 171, 221))
        draw.rectangle([x0, torso_bot - 8 * scale, x1, torso_bot], fill=(245, 245, 245))
        num_c = (28, 44, 91)
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
    # Hair
    draw.arc([x - head_r, head_y - head_r - 1, x + head_r, head_y + 2], 180, 360, fill=hair_rgb, width=int(3 * scale))

def generate_fixture_1():
    """Generates T0 and T1 for Galatasaray vs Barcelona: Pedri midfield through-ball to Ferran Torres."""
    # === FRAME T0: Buildup / Setup ===
    t0 = create_pitch_canvas()
    d0 = ImageDraw.Draw(t0)

    # Barcelona players in T0
    # Pedri (#16) has ball in midfield
    draw_player_figure(d0, 480, 360, "barcelona", 16, hair_style="dark", skin_tone="light", scale=1.1)
    # Busquets (#5) holding pivot
    draw_player_figure(d0, 340, 370, "barcelona", 5, hair_style="dark", skin_tone="light", scale=1.1)
    # Frenkie de Jong (#21) supporting midfield
    draw_player_figure(d0, 450, 480, "barcelona", 21, hair_style="blonde", skin_tone="light", scale=1.05)
    # Ferran Torres (#19) making forward diagonal run into box channel
    draw_player_figure(d0, 720, 260, "barcelona", 19, hair_style="dark", skin_tone="light", scale=1.0)
    # Aubameyang (#25) pinning center backs
    draw_player_figure(d0, 840, 360, "barcelona", 25, hair_style="dark mohawk", skin_tone="dark", scale=1.05)
    # Jordi Alba (#18) overlapping left flank
    draw_player_figure(d0, 520, 160, "barcelona", 18, hair_style="dark", skin_tone="light", scale=1.0)

    # Galatasaray defenders in T0
    # Nelsson (#25) CB
    draw_player_figure(d0, 800, 320, "galatasaray", 25, hair_style="blonde", skin_tone="light", scale=1.05)
    # Marcão (#45) CB
    draw_player_figure(d0, 810, 410, "galatasaray", 45, hair_style="dark", skin_tone="dark", scale=1.1)
    # Boey (#93) RB tracking wing
    draw_player_figure(d0, 750, 190, "galatasaray", 93, hair_style="blonde dreads", skin_tone="dark", scale=1.0)
    # Antalyalı (#4) DM pressing Pedri
    draw_player_figure(d0, 560, 350, "galatasaray", 4, hair_style="dark", skin_tone="olive", scale=1.05)
    # Kutlu (#22) CM covering center
    draw_player_figure(d0, 600, 420, "galatasaray", 22, hair_style="dark", skin_tone="light", scale=1.05)

    # Ball in T0 at Pedri's feet
    d0.ellipse([500, 365, 516, 381], fill=(255, 255, 255), outline=(0, 0, 0), width=2)

    t0_path = os.path.join(OUTPUT_DIR, "galatasaray_barcelona_t0.jpg")
    t0.save(t0_path, quality=95)

    # === FRAME T1: Outcome (2 seconds later) ===
    t1 = create_pitch_canvas()
    d1 = ImageDraw.Draw(t1)

    # Barcelona players in T1 (ball reached Ferran Torres inside the box)
    # Pedri (#16) after releasing pass
    draw_player_figure(d1, 510, 355, "barcelona", 16, hair_style="dark", skin_tone="light", scale=1.1)
    # Busquets (#5)
    draw_player_figure(d1, 360, 365, "barcelona", 5, hair_style="dark", skin_tone="light", scale=1.1)
    # Frenkie de Jong (#21) advancing
    draw_player_figure(d1, 500, 470, "barcelona", 21, hair_style="blonde", skin_tone="light", scale=1.05)
    # Ferran Torres (#19) receiving ball in space at edge of box
    draw_player_figure(d1, 860, 240, "barcelona", 19, hair_style="dark", skin_tone="light", scale=1.0)
    # Aubameyang (#25) making near-post run
    draw_player_figure(d1, 950, 330, "barcelona", 25, hair_style="dark mohawk", skin_tone="dark", scale=1.05)
    # Jordi Alba (#18)
    draw_player_figure(d1, 640, 150, "barcelona", 18, hair_style="dark", skin_tone="light", scale=1.0)

    # Galatasaray defenders in T1 scrambling back
    draw_player_figure(d1, 880, 310, "galatasaray", 25, hair_style="blonde", skin_tone="light", scale=1.05)
    draw_player_figure(d1, 890, 390, "galatasaray", 45, hair_style="dark", skin_tone="dark", scale=1.1)
    draw_player_figure(d1, 840, 200, "galatasaray", 93, hair_style="blonde dreads", skin_tone="dark", scale=1.0)
    draw_player_figure(d1, 620, 345, "galatasaray", 4, hair_style="dark", skin_tone="olive", scale=1.05)
    draw_player_figure(d1, 670, 410, "galatasaray", 22, hair_style="dark", skin_tone="light", scale=1.05)

    # Ball in T1 at Ferran Torres's feet
    d1.ellipse([872, 246, 888, 262], fill=(255, 255, 255), outline=(0, 0, 0), width=2)

    t1_path = os.path.join(OUTPUT_DIR, "galatasaray_barcelona_t1.jpg")
    t1.save(t1_path, quality=95)

    print(f"Generated Fixture 1: {t0_path} & {t1_path}")
    return t0_path, t1_path

def generate_fixture_2():
    """Generates T0 and T1 for Real Madrid vs Man City: Valverde switch of play to Vinicius Jr."""
    # === FRAME T0: Setup ===
    t0 = create_pitch_canvas()
    d0 = ImageDraw.Draw(t0)

    # Real Madrid (White)
    draw_player_figure(d0, 420, 420, "realmadrid", 15, hair_style="dark", skin_tone="light", scale=1.1) # Valverde (ball carrier)
    draw_player_figure(d0, 780, 160, "realmadrid", 7, hair_style="dark locs", skin_tone="dark", scale=1.0)  # Vini Jr (wing run)
    draw_player_figure(d0, 680, 360, "realmadrid", 5, hair_style="dark taper", skin_tone="light", scale=1.1) # Bellingham
    draw_player_figure(d0, 820, 340, "realmadrid", 9, hair_style="dark buzz", skin_tone="dark", scale=1.05) # Mbappe
    draw_player_figure(d0, 320, 380, "realmadrid", 18, hair_style="dark", skin_tone="dark", scale=1.1)  # Tchouameni

    # Man City (Sky Blue)
    draw_player_figure(d0, 520, 400, "mancity", 16, hair_style="dark", skin_tone="light", scale=1.1)     # Rodri
    draw_player_figure(d0, 760, 200, "mancity", 2, hair_style="dark bald", skin_tone="light", scale=1.05)  # Walker
    draw_player_figure(d0, 790, 330, "mancity", 3, hair_style="dark", skin_tone="light", scale=1.1)      # Ruben Dias
    draw_player_figure(d0, 620, 320, "mancity", 17, hair_style="blonde", skin_tone="light", scale=1.05)  # De Bruyne

    # Ball at Valverde's feet
    d0.ellipse([435, 425, 451, 441], fill=(255, 255, 255), outline=(0, 0, 0), width=2)

    t0_path = os.path.join(OUTPUT_DIR, "realmadrid_mancity_t0.jpg")
    t0.save(t0_path, quality=95)

    # === FRAME T1: Outcome ===
    t1 = create_pitch_canvas()
    d1 = ImageDraw.Draw(t1)

    # Real Madrid
    draw_player_figure(d1, 460, 410, "realmadrid", 15, hair_style="dark", skin_tone="light", scale=1.1)
    draw_player_figure(d1, 910, 180, "realmadrid", 7, hair_style="dark locs", skin_tone="dark", scale=1.0) # Vini receives on wing
    draw_player_figure(d1, 740, 340, "realmadrid", 5, hair_style="dark taper", skin_tone="light", scale=1.1)
    draw_player_figure(d1, 900, 320, "realmadrid", 9, hair_style="dark buzz", skin_tone="dark", scale=1.05)
    draw_player_figure(d1, 350, 370, "realmadrid", 18, hair_style="dark", skin_tone="dark", scale=1.1)

    # Man City
    draw_player_figure(d1, 560, 390, "mancity", 16, hair_style="dark", skin_tone="light", scale=1.1)
    draw_player_figure(d1, 880, 210, "mancity", 2, hair_style="dark bald", skin_tone="light", scale=1.05) # Walker chasing
    draw_player_figure(d1, 840, 320, "mancity", 3, hair_style="dark", skin_tone="light", scale=1.1)
    draw_player_figure(d1, 650, 310, "mancity", 17, hair_style="blonde", skin_tone="light", scale=1.05)

    # Ball at Vini Jr's feet
    d1.ellipse([920, 185, 936, 201], fill=(255, 255, 255), outline=(0, 0, 0), width=2)

    t1_path = os.path.join(OUTPUT_DIR, "realmadrid_mancity_t1.jpg")
    t1.save(t1_path, quality=95)

    print(f"Generated Fixture 2: {t0_path} & {t1_path}")
    return t0_path, t1_path

if __name__ == "__main__":
    generate_fixture_1()
    generate_fixture_2()
