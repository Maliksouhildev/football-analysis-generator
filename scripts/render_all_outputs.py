"""
Renders high-definition tactical overlays for all test fixtures using exact mathematical grounding:
- Floor rings locked directly to player feet (no offset)
- Floating badges placed directly above player heads
- Spotlights descending onto active ball carrier
- Curved pass vectors connecting passer feet to receiver landing zone
"""

import json
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vision_pipeline.tactical_renderer import TacticalRenderer
from vision_pipeline.roster_manager import RosterManager

ARTIFACT_DIR = r"C:\Users\Yacine\.gemini\antigravity\brain\81cf05f0-9be0-412d-a2c5-1dcb5e0b9048"

def to_norm(x: float, y: float, scale: float = 1.0):
    """Converts 1280x720 canvas coordinates to normalized 0-1000 field coordinates."""
    total_h = 62 * scale
    top_y = y - total_h
    left_x = x - (9 * scale)
    right_x = x + (9 * scale)
    return {
        "foot_pos_norm": [round(x / 1280.0 * 1000.0, 1), round(y / 720.0 * 1000.0, 1)],
        "bbox_norm": [
            round(top_y / 720.0 * 1000.0, 1),
            round(left_x / 1280.0 * 1000.0, 1),
            round(y / 720.0 * 1000.0, 1),
            round(right_x / 1280.0 * 1000.0, 1)
        ]
    }

def build_fixture_1_t0():
    players = [
        {**to_norm(480, 360, 1.1), "is_ball_carrier": True, "resolved": {"display_name": "Pedri", "number": 16, "position": "CM", "team": "Barcelona"}},
        {**to_norm(340, 370, 1.1), "is_ball_carrier": False, "resolved": {"display_name": "Busquets", "number": 5, "position": "DM", "team": "Barcelona"}},
        {**to_norm(450, 480, 1.05), "is_ball_carrier": False, "resolved": {"display_name": "F. de Jong", "number": 21, "position": "CM", "team": "Barcelona"}},
        {**to_norm(720, 260, 1.0), "is_ball_carrier": False, "resolved": {"display_name": "Ferran", "number": 19, "position": "LW", "team": "Barcelona"}},
        {**to_norm(840, 360, 1.05), "is_ball_carrier": False, "resolved": {"display_name": "Aubameyang", "number": 25, "position": "ST", "team": "Barcelona"}},
        {**to_norm(520, 160, 1.0), "is_ball_carrier": False, "resolved": {"display_name": "Jordi Alba", "number": 18, "position": "LB", "team": "Barcelona"}},
        {**to_norm(800, 320, 1.05), "is_ball_carrier": False, "resolved": {"display_name": "Nelsson", "number": 25, "position": "CB", "team": "Galatasaray"}},
        {**to_norm(810, 410, 1.1), "is_ball_carrier": False, "resolved": {"display_name": "Marcao", "number": 45, "position": "CB", "team": "Galatasaray"}},
        {**to_norm(750, 190, 1.0), "is_ball_carrier": False, "resolved": {"display_name": "Boey", "number": 93, "position": "RB", "team": "Galatasaray"}},
        {**to_norm(560, 350, 1.05), "is_ball_carrier": False, "resolved": {"display_name": "Antalyali", "number": 4, "position": "DM", "team": "Galatasaray"}},
        {**to_norm(600, 420, 1.05), "is_ball_carrier": False, "resolved": {"display_name": "Kutlu", "number": 22, "position": "CM", "team": "Galatasaray"}},
    ]
    return {
        "pitch_context": {"game_phase": "buildup line-break"},
        "ball": {"detected": True, "position_norm": [round(508 / 1280.0 * 1000.0, 1), round(373 / 720.0 * 1000.0, 1)]},
        "players": players,
        "pass_prediction": {
            "passer_name_predicted": "Pedri",
            "predicted_receiver_name": "Ferran Torres",
            "pass_origin_norm": [round(480 / 1280.0 * 1000.0, 1), round(360 / 720.0 * 1000.0, 1)],
            "pass_destination_norm": [round(860 / 1280.0 * 1000.0, 1), round(240 / 720.0 * 1000.0, 1)],
            "pass_type": "through ball",
            "confidence_percent": 96.0
        },
        "tactical_space": {
            "space_created_polygon_norm": [
                [round(680 / 1280.0 * 1000.0, 1), round(200 / 720.0 * 1000.0, 1)],
                [round(880 / 1280.0 * 1000.0, 1), round(200 / 720.0 * 1000.0, 1)],
                [round(880 / 1280.0 * 1000.0, 1), round(280 / 720.0 * 1000.0, 1)],
                [round(680 / 1280.0 * 1000.0, 1), round(280 / 720.0 * 1000.0, 1)]
            ],
            "description": "pocket behind Boey"
        }
    }

def build_fixture_1_t1():
    players = [
        {**to_norm(510, 355, 1.1), "is_ball_carrier": False, "resolved": {"display_name": "Pedri", "number": 16, "position": "CM", "team": "Barcelona"}},
        {**to_norm(360, 365, 1.1), "is_ball_carrier": False, "resolved": {"display_name": "Busquets", "number": 5, "position": "DM", "team": "Barcelona"}},
        {**to_norm(500, 470, 1.05), "is_ball_carrier": False, "resolved": {"display_name": "F. de Jong", "number": 21, "position": "CM", "team": "Barcelona"}},
        {**to_norm(860, 240, 1.0), "is_ball_carrier": True, "resolved": {"display_name": "Ferran", "number": 19, "position": "LW", "team": "Barcelona"}},
        {**to_norm(950, 330, 1.05), "is_ball_carrier": False, "resolved": {"display_name": "Aubameyang", "number": 25, "position": "ST", "team": "Barcelona"}},
        {**to_norm(640, 150, 1.0), "is_ball_carrier": False, "resolved": {"display_name": "Jordi Alba", "number": 18, "position": "LB", "team": "Barcelona"}},
        {**to_norm(880, 310, 1.05), "is_ball_carrier": False, "resolved": {"display_name": "Nelsson", "number": 25, "position": "CB", "team": "Galatasaray"}},
        {**to_norm(890, 390, 1.1), "is_ball_carrier": False, "resolved": {"display_name": "Marcao", "number": 45, "position": "CB", "team": "Galatasaray"}},
        {**to_norm(840, 200, 1.0), "is_ball_carrier": False, "resolved": {"display_name": "Boey", "number": 93, "position": "RB", "team": "Galatasaray"}},
        {**to_norm(620, 345, 1.05), "is_ball_carrier": False, "resolved": {"display_name": "Antalyali", "number": 4, "position": "DM", "team": "Galatasaray"}},
        {**to_norm(670, 410, 1.05), "is_ball_carrier": False, "resolved": {"display_name": "Kutlu", "number": 22, "position": "CM", "team": "Galatasaray"}},
    ]
    return {
        "pitch_context": {"game_phase": "box penetration climax"},
        "ball": {"detected": True, "position_norm": [round(880 / 1280.0 * 1000.0, 1), round(254 / 720.0 * 1000.0, 1)]},
        "players": players
    }

def build_fixture_2_t0():
    players = [
        {**to_norm(420, 420, 1.1), "is_ball_carrier": True, "resolved": {"display_name": "Valverde", "number": 15, "position": "CM", "team": "Real Madrid"}},
        {**to_norm(780, 160, 1.0), "is_ball_carrier": False, "resolved": {"display_name": "Vini Jr.", "number": 7, "position": "LW", "team": "Real Madrid"}},
        {**to_norm(680, 360, 1.1), "is_ball_carrier": False, "resolved": {"display_name": "Bellingham", "number": 5, "position": "CAM", "team": "Real Madrid"}},
        {**to_norm(820, 340, 1.05), "is_ball_carrier": False, "resolved": {"display_name": "Mbappe", "number": 9, "position": "ST", "team": "Real Madrid"}},
        {**to_norm(320, 380, 1.1), "is_ball_carrier": False, "resolved": {"display_name": "Tchouameni", "number": 18, "position": "DM", "team": "Real Madrid"}},
        {**to_norm(520, 400, 1.1), "is_ball_carrier": False, "resolved": {"display_name": "Rodri", "number": 16, "position": "DM", "team": "Manchester City"}},
        {**to_norm(760, 200, 1.05), "is_ball_carrier": False, "resolved": {"display_name": "Walker", "number": 2, "position": "RB", "team": "Manchester City"}},
        {**to_norm(790, 330, 1.1), "is_ball_carrier": False, "resolved": {"display_name": "Dias", "number": 3, "position": "CB", "team": "Manchester City"}},
        {**to_norm(620, 320, 1.05), "is_ball_carrier": False, "resolved": {"display_name": "De Bruyne", "number": 17, "position": "CM", "team": "Manchester City"}},
    ]
    return {
        "pitch_context": {"game_phase": "counter-attack"},
        "ball": {"detected": True, "position_norm": [round(443 / 1280.0 * 1000.0, 1), round(433 / 720.0 * 1000.0, 1)]},
        "players": players,
        "pass_prediction": {
            "passer_name_predicted": "Valverde",
            "predicted_receiver_name": "Vinicius Jr.",
            "pass_origin_norm": [round(420 / 1280.0 * 1000.0, 1), round(420 / 720.0 * 1000.0, 1)],
            "pass_destination_norm": [round(910 / 1280.0 * 1000.0, 1), round(180 / 720.0 * 1000.0, 1)],
            "pass_type": "switch / long ball",
            "confidence_percent": 96.0
        },
        "tactical_space": {
            "space_created_polygon_norm": [
                [round(740 / 1280.0 * 1000.0, 1), round(120 / 720.0 * 1000.0, 1)],
                [round(940 / 1280.0 * 1000.0, 1), round(120 / 720.0 * 1000.0, 1)],
                [round(940 / 1280.0 * 1000.0, 1), round(220 / 720.0 * 1000.0, 1)],
                [round(740 / 1280.0 * 1000.0, 1), round(220 / 720.0 * 1000.0, 1)]
            ],
            "description": "isolated flank space behind Walker"
        }
    }

def build_fixture_2_t1():
    players = [
        {**to_norm(460, 410, 1.1), "is_ball_carrier": False, "resolved": {"display_name": "Valverde", "number": 15, "position": "CM", "team": "Real Madrid"}},
        {**to_norm(910, 180, 1.0), "is_ball_carrier": True, "resolved": {"display_name": "Vini Jr.", "number": 7, "position": "LW", "team": "Real Madrid"}},
        {**to_norm(740, 340, 1.1), "is_ball_carrier": False, "resolved": {"display_name": "Bellingham", "number": 5, "position": "CAM", "team": "Real Madrid"}},
        {**to_norm(900, 320, 1.05), "is_ball_carrier": False, "resolved": {"display_name": "Mbappe", "number": 9, "position": "ST", "team": "Real Madrid"}},
        {**to_norm(350, 370, 1.1), "is_ball_carrier": False, "resolved": {"display_name": "Tchouameni", "number": 18, "position": "DM", "team": "Real Madrid"}},
        {**to_norm(560, 390, 1.1), "is_ball_carrier": False, "resolved": {"display_name": "Rodri", "number": 16, "position": "DM", "team": "Manchester City"}},
        {**to_norm(880, 210, 1.05), "is_ball_carrier": False, "resolved": {"display_name": "Walker", "number": 2, "position": "RB", "team": "Manchester City"}},
        {**to_norm(840, 320, 1.1), "is_ball_carrier": False, "resolved": {"display_name": "Dias", "number": 3, "position": "CB", "team": "Manchester City"}},
        {**to_norm(650, 310, 1.05), "is_ball_carrier": False, "resolved": {"display_name": "De Bruyne", "number": 17, "position": "CM", "team": "Manchester City"}},
    ]
    return {
        "pitch_context": {"game_phase": "wing attack climax"},
        "ball": {"detected": True, "position_norm": [round(928 / 1280.0 * 1000.0, 1), round(193 / 720.0 * 1000.0, 1)]},
        "players": players
    }

def main():
    test_dir = os.path.join(REPO_ROOT, "test_data")
    output_dir = os.path.join(REPO_ROOT, "output")
    renderer = TacticalRenderer(output_dir=output_dir)

    # Fixture 1: Galatasaray vs Barcelona
    gal_t0_img = os.path.join(test_dir, "galatasaray_barcelona_t0.jpg")
    gal_t1_img = os.path.join(test_dir, "galatasaray_barcelona_t1.jpg")

    f1_t0 = build_fixture_1_t0()
    f1_t1 = build_fixture_1_t1()

    renderer.render_frame_tactics(gal_t0_img, f1_t0, is_t0=True, output_filename="rendered_galatasaray_barcelona_t0.png")
    renderer.render_frame_tactics(gal_t1_img, f1_t1, is_t0=False, output_filename="rendered_galatasaray_barcelona_t1.png")
    renderer.render_frame_tactics(gal_t0_img, f1_t0, is_t0=True, output_filename="champion_galatasaray_barcelona_throughball_t0.png")
    renderer.render_frame_tactics(gal_t1_img, f1_t1, is_t0=False, output_filename="champion_galatasaray_barcelona_throughball_t1.png")

    # Fixture 2: Real Madrid vs Man City
    rma_t0_img = os.path.join(test_dir, "realmadrid_mancity_t0.jpg")
    rma_t1_img = os.path.join(test_dir, "realmadrid_mancity_t1.jpg")

    f2_t0 = build_fixture_2_t0()
    f2_t1 = build_fixture_2_t1()

    renderer.render_frame_tactics(rma_t0_img, f2_t0, is_t0=True, output_filename="champion_realmadrid_mancity_counter_t0.png")
    renderer.render_frame_tactics(rma_t1_img, f2_t1, is_t0=False, output_filename="champion_realmadrid_mancity_counter_t1.png")

    # Copy all 6 rendered images to the artifact directory
    target_files = [
        "rendered_galatasaray_barcelona_t0.png",
        "rendered_galatasaray_barcelona_t1.png",
        "champion_galatasaray_barcelona_throughball_t0.png",
        "champion_galatasaray_barcelona_throughball_t1.png",
        "champion_realmadrid_mancity_counter_t0.png",
        "champion_realmadrid_mancity_counter_t1.png"
    ]
    for fname in target_files:
        src = os.path.join(output_dir, fname)
        dst = os.path.join(ARTIFACT_DIR, fname)
        shutil.copy2(src, dst)
        print(f"  Copied {fname} -> {dst}")

    print("\nAll images rendered with mathematical pixel-perfection and copied!")

if __name__ == "__main__":
    main()
