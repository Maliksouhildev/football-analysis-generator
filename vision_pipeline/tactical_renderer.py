"""
Tactical Renderer
Generates broadcast-grade tactical overlays:
- Smart anti-collision player badge layout
- Glowing perspective floor rings & halos anchored at player feet
- Dramatic tactical spotlight cones
- Clean floating glassmorphic HUD player badges (Number, Name, Role, Confidence)
- Curved pass trajectory vectors with animated dash patterns and landing target zones
- Tactical space control / pass channel polygons
Outputs both high-resolution composited PNGs and animated SVG code for web dashboards.
"""

import math
import os
import re
import unicodedata
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image, ImageDraw, ImageFont

SPECIAL_CHAR_MAP = {
    'ı': 'i', 'İ': 'I',
    'ş': 's', 'Ş': 'S',
    'ğ': 'g', 'Ğ': 'G',
    'ç': 'c', 'Ç': 'C',
    'ö': 'o', 'Ö': 'O',
    'ü': 'u', 'Ü': 'U',
    'ã': 'a', 'Ã': 'A',
    'é': 'e', 'É': 'E',
    'á': 'a', 'Á': 'A',
    'í': 'i', 'Í': 'I',
    'ó': 'o', 'Ó': 'O',
    'ú': 'u', 'Ú': 'U',
    'ñ': 'n', 'Ñ': 'N',
}

def sanitize_text(text: str) -> str:
    """Translates special and accented characters to standard ASCII for reliable rendering."""
    if not text:
        return ""
    for k, v in SPECIAL_CHAR_MAP.items():
        text = text.replace(k, v)
    normalized = unicodedata.normalize('NFKD', text)
    ascii_bytes = normalized.encode('ASCII', 'ignore')
    return ascii_bytes.decode('utf-8')

class TacticalRenderer:
    """Renders professional TV broadcast tactical graphics on football frames."""

    TEAM_COLORS = {
        "barcelona": {"primary": (0, 77, 152), "accent": (237, 187, 0), "text": (255, 255, 255)},
        "galatasaray": {"primary": (169, 4, 50), "accent": (253, 185, 18), "text": (255, 255, 255)},
        "real madrid": {"primary": (30, 30, 30), "accent": (238, 177, 17), "text": (255, 255, 255)},
        "manchester city": {"primary": (28, 44, 91), "accent": (108, 171, 221), "text": (255, 255, 255)},
        "koln": {"primary": (215, 25, 32), "accent": (255, 255, 255), "text": (255, 255, 255)},
        "cologne": {"primary": (215, 25, 32), "accent": (255, 255, 255), "text": (255, 255, 255)},
        "hertha": {"primary": (0, 77, 153), "accent": (255, 255, 255), "text": (255, 255, 255)},
        "referee": {"primary": (30, 215, 96), "accent": (255, 255, 255), "text": (0, 0, 0)},
        "goalkeeper": {"primary": (46, 204, 113), "accent": (255, 255, 255), "text": (0, 0, 0)},
        "default_home": {"primary": (180, 20, 40), "accent": (255, 193, 7), "text": (255, 255, 255)},
        "default_away": {"primary": (13, 80, 200), "accent": (255, 255, 255), "text": (255, 255, 255)},
    }

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_team_color(self, team_name: str) -> Dict[str, Tuple[int, int, int]]:
        """Resolves RGB color theme for a given team name."""
        name_lower = sanitize_text(str(team_name)).lower()
        for k, v in self.TEAM_COLORS.items():
            if k in name_lower:
                return v
        return self.TEAM_COLORS["default_away"] if "away" in name_lower else self.TEAM_COLORS["default_home"]

    def render_frame_tactics(
        self,
        image_path: str,
        frame_analysis: Dict[str, Any],
        is_t0: bool = False,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Renders tactical overlays onto the image and saves composited result.
        Returns the absolute filepath to the rendered image.
        """
        base_img = Image.open(image_path).convert("RGBA")
        width, height = base_img.size

        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # 1. Render Tactical Space Polygon (if present)
        tactical_space = frame_analysis.get("tactical_space", {})
        poly_norm = tactical_space.get("space_created_polygon_norm")
        if poly_norm and len(poly_norm) >= 3:
            poly_px = [(p[0] * width / 1000.0, p[1] * height / 1000.0) for p in poly_norm]
            draw.polygon(poly_px, fill=(0, 230, 118, 45), outline=(0, 230, 118, 200))

        # 2. Extract players, calculate grounded geometry & badges (excluding referee & out-of-bounds entities)
        raw_players = frame_analysis.get("players", [])
        players = []
        for p in raw_players:
            resolved = p.get("resolved", {})
            p_name = (resolved.get("display_name") or p.get("name") or "").lower()
            p_role = (resolved.get("position") or p.get("role") or "").lower()
            p_team = (resolved.get("team") or p.get("team") or "").lower()
            p_kit = (p.get("kit_color") or "").lower()
            if any(term in p_name or term in p_role or term in p_team or term in p_kit for term in ["ref", "official", "linesman", "cameraman", "staff", "yellow"]):
                continue
            foot_pos = p.get("foot_pos_norm")
            y_norm = foot_pos[1] if foot_pos else (p.get("bbox_norm", [0, 0, 100, 100])[2])
            if y_norm > 815.0 or y_norm < 180.0:
                continue
            players.append(p)

        badge_positions = []

        for p in players:
            bbox = p.get("bbox_norm", [0, 0, 100, 100])
            foot_pos = p.get("foot_pos_norm")
            is_carrier = p.get("is_ball_carrier", False)

            if foot_pos:
                fx = foot_pos[0] * width / 1000.0
                fy = foot_pos[1] * height / 1000.0
            else:
                fx = ((bbox[1] + bbox[3]) / 2.0) * width / 1000.0
                fy = (bbox[2]) * height / 1000.0

            resolved = p.get("resolved", {})
            team_color = self._get_team_color(p.get("team", resolved.get("team", "")))

            # Draw Floor Ring directly at feet
            self._draw_floor_ring(draw, fx, fy, team_color, is_carrier)

            # Draw Spotlight directly on carrier
            if is_carrier:
                self._draw_spotlight(draw, fx, fy, team_color)

            # Compute head point and badge baseline
            head_x = fx
            head_y = (bbox[0] * height / 1000.0) if bbox[0] > 0 else fy - 48.0
            top_y = head_y - 14.0

            display_name = sanitize_text(resolved.get("display_name") or p.get("id", "Player"))
            pos_str = sanitize_text(resolved.get("position") or p.get("tactical_role") or "")
            badge_w = max(90.0, 32.0 + len(display_name) * 7.0 + (len(pos_str) * 6.0 + 10.0 if pos_str else 0.0) + 12.0)

            badge_positions.append({
                "player": p,
                "team_color": team_color,
                "x": fx,
                "y": max(26.0, top_y),
                "head_x": head_x,
                "head_y": head_y,
                "w": badge_w,
                "is_carrier": is_carrier
            })

        # Anti-collision layout for player badges (resolves 2D overlap cleanly)
        resolved_badges = self._resolve_badge_collisions(badge_positions)

        # 3. Render Pass Prediction Vector (for T0 setup frame) with badge clearance
        if is_t0 and "pass_prediction" in frame_analysis:
            pred = frame_analysis["pass_prediction"]
            origin = pred.get("pass_origin_norm", [500, 500])
            dest = pred.get("pass_destination_norm", [600, 400])

            p0 = (origin[0] * width / 1000.0, origin[1] * height / 1000.0)
            p1 = (dest[0] * width / 1000.0, dest[1] * height / 1000.0)

            badge_boxes = [
                (b["x"] - b.get("w", 100) / 2.0, b["y"] - 24, b["x"] + b.get("w", 100) / 2.0, b["y"])
                for b in resolved_badges
            ]
            self._draw_pass_vector(
                draw, p0, p1,
                pred.get("pass_type", "Ground Pass"),
                pred.get("confidence_percent", 90),
                avoid_boxes=badge_boxes
            )

        # 4. Render floating badges on top with leader stems
        for b in resolved_badges:
            self._draw_player_hud_badge(draw, b)

        # 5. Render Ball Halo (if detected)
        ball = frame_analysis.get("ball", {})
        if ball.get("detected") and ball.get("position_norm"):
            bpos = ball["position_norm"]
            bx = bpos[0] * width / 1000.0
            by = bpos[1] * height / 1000.0
            draw.ellipse([bx - 12, by - 12, bx + 12, by + 12], outline=(255, 215, 0, 240), width=3)
            draw.ellipse([bx - 5, by - 5, bx + 5, by + 5], fill=(255, 255, 255, 255), outline=(0, 0, 0, 220), width=1)

        # Composite overlay with base image
        composited = Image.alpha_composite(base_img, overlay).convert("RGB")

        if not output_filename:
            output_filename = f"tactical_{'t0' if is_t0 else 't1'}.png"
        out_path = os.path.join(self.output_dir, output_filename)
        composited.save(out_path, quality=95)
        return out_path

    def render_sequence_stage(
        self,
        image_path: str,
        stage_data: Dict[str, Any],
        output_filename: Optional[str] = None
    ) -> str:
        """
        Renders broadcast tactical graphics for a single stage (T0..T4) in a multi-frame timeline.
        Includes motion breadcrumb trails, continuous ball trajectory arc, and live HUD telemetry.
        """
        base_img = Image.open(image_path).convert("RGBA")
        width, height = base_img.size
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # 1. Filter out referee, officials, sideline staff, and out-of-bounds tracks
        raw_players = stage_data.get("players", [])
        players = []
        for p in raw_players:
            resolved = p.get("resolved", {})
            p_name = (resolved.get("display_name") or p.get("name") or "").lower()
            p_role = (resolved.get("position") or p.get("role") or p.get("tactical_role_tag") or "").lower()
            p_team = (resolved.get("team") or p.get("team") or "").lower()
            p_kit = (p.get("kit_color") or "").lower()
            if any(term in p_name or term in p_role or term in p_team or term in p_kit for term in ["ref", "official", "linesman", "cameraman", "staff", "yellow"]):
                continue
            fx, fy = p.get("foot_pos_px", (0, 0))
            y_norm = p.get("foot_pos_norm", [0, fy * 1000.0 / height])[1]
            if y_norm > 815.0 or y_norm < 180.0:
                continue
            players.append(p)

        # Render motion trails for verified outfield players
        for p in players:
            trail = p.get("trail_px", [])
            team_color = self._get_team_color(p.get("team", ""))
            if len(trail) >= 2:
                self._draw_motion_trail(draw, trail, team_color)

        # 2. Render Floor Rings & Volumetric Spotlights
        badge_positions = []
        for p in players:
            fx, fy = p["foot_pos_px"]
            is_carrier = p.get("is_ball_carrier", False)
            team_color = self._get_team_color(p.get("team", ""))

            self._draw_floor_ring(draw, fx, fy, team_color, is_carrier)
            if is_carrier:
                self._draw_spotlight(draw, fx, fy, team_color)

            head_x = fx
            head_y = max(35.0, fy - 65.0)
            top_y = head_y - 14.0

            display_name = sanitize_text(p.get("name") or p.get("id", "Player"))
            pos_str = sanitize_text(p.get("role") or p.get("tactical_role_tag") or "")
            badge_w = max(90.0, 32.0 + len(display_name) * 7.0 + (len(pos_str) * 6.0 + 10.0 if pos_str else 0.0) + 12.0)

            badge_positions.append({
                "player": p,
                "team_color": team_color,
                "x": fx,
                "y": max(55.0, top_y),
                "head_x": head_x,
                "head_y": head_y,
                "w": badge_w,
                "is_carrier": is_carrier
            })

        # Anti-collision layout for player badges
        resolved_badges = self._resolve_badge_collisions(badge_positions)

        # 3. Render continuous ball trajectory arc
        trajectory_pts = stage_data.get("ball_trajectory_points", [])
        curr_ball = stage_data.get("ball", {}).get("position_px")
        stage_idx = stage_data.get("stage_idx", 0)
        self._draw_ball_trajectory_arc(draw, trajectory_pts, curr_ball, stage_idx)

        # 4. Render floating badges on top with leader stems
        for b in resolved_badges:
            self._draw_player_hud_badge(draw, b)

        # 5. Render top broadcast HUD telemetry banner
        metrics = stage_data.get("metrics", {})
        self._draw_hud_telemetry_banner(draw, width, height, metrics, stage_data)

        # Composite overlay with base image
        composited = Image.alpha_composite(base_img, overlay).convert("RGB")
        if not output_filename:
            phase_code = stage_data.get("phase_code", "t0").lower()
            output_filename = f"tactical_sequence_{phase_code}.png"
        out_path = os.path.join(self.output_dir, output_filename)
        composited.save(out_path, quality=95)
        return out_path

    def _draw_motion_trail(
        self,
        draw: ImageDraw.ImageDraw,
        trail: List[Tuple[float, float]],
        color_theme: Dict[str, Tuple[int, int, int]]
    ):
        """Draws a glowing motion breadcrumb path showing historical movement."""
        prim = color_theme["primary"]
        acc = color_theme["accent"]
        n = len(trail)
        for i in range(n - 1):
            alpha = int(50 + (180 * (i + 1) / n))
            p0 = trail[i]
            p1 = trail[i + 1]
            draw.line([p0, p1], fill=(*acc, alpha), width=3)
            # Breadcrumb dots
            draw.ellipse([p1[0] - 3, p1[1] - 3, p1[0] + 3, p1[1] + 3], fill=(*prim, alpha), outline=(255, 255, 255, alpha))

    def _draw_ball_trajectory_arc(
        self,
        draw: ImageDraw.ImageDraw,
        trajectory_pts: List[Tuple[float, float]],
        curr_ball: Optional[Tuple[float, float]],
        current_stage_idx: int
    ):
        """Draws parabolic ball flight path with past positions and future destination."""
        valid_pts = [p for p in trajectory_pts if p is not None]
        if len(valid_pts) >= 2:
            for i in range(len(valid_pts) - 1):
                p0, p1 = valid_pts[i], valid_pts[i + 1]
                is_past = (i < current_stage_idx)
                color = (255, 215, 0, 220) if is_past else (0, 229, 255, 120)
                width = 4 if is_past else 2
                draw.line([p0, p1], fill=color, width=width)
                draw.ellipse([p0[0] - 4, p0[1] - 4, p0[0] + 4, p0[1] + 4], fill=color)

        if curr_ball:
            bx, by = curr_ball
            # Glowing ball beacon
            draw.ellipse([bx - 14, by - 14, bx + 14, by + 14], outline=(255, 215, 0, 240), width=3)
            draw.ellipse([bx - 8, by - 8, bx + 8, by + 8], fill=(255, 255, 255, 255), outline=(0, 0, 0, 220), width=2)
            # Pulse ring
            draw.ellipse([bx - 22, by - 22, bx + 22, by + 22], outline=(0, 229, 255, 140), width=1)

    def _draw_hud_telemetry_banner(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        metrics: Dict[str, Any],
        stage_data: Dict[str, Any]
    ):
        """Draws TV-broadcast style tactical header and telemetry HUD."""
        # Top banner
        phase_code = stage_data.get("phase_code", "T0")
        phase_name = sanitize_text(stage_data.get("phase_name", "Tactical Setup")).upper()
        time_sec = metrics.get("timestamp_sec", 0.0)

        banner_w = 640
        banner_h = 36
        bx0 = (width - banner_w) / 2.0
        by0 = 12.0
        bx1 = bx0 + banner_w
        by1 = by0 + banner_h

        draw.rounded_rectangle([bx0, by0, bx1, by1], radius=8, fill=(10, 14, 23, 230), outline=(0, 229, 255, 180), width=1)

        # Phase tag pill
        draw.rounded_rectangle([bx0 + 8, by0 + 6, bx0 + 55, by1 - 6], radius=4, fill=(0, 229, 255, 220))
        draw.text((bx0 + 16, by0 + 10), phase_code, fill=(0, 0, 0, 255))

        # Title
        title_text = f"TACTICAL TIMELINE  |  {phase_name}  ({time_sec:.1f}s)"
        draw.text((bx0 + 66, by0 + 10), title_text, fill=(255, 255, 255, 255))

        # Bottom-right telemetry card
        card_w = 360
        card_h = 76
        cx0 = width - card_w - 20
        cy0 = height - card_h - 20
        cx1 = cx0 + card_w
        cy1 = cy0 + card_h

        draw.rounded_rectangle([cx0, cy0, cx1, cy1], radius=8, fill=(10, 14, 23, 225), outline=(0, 229, 255, 140), width=1)
        draw.text((cx0 + 14, cy0 + 10), "MATCH TELEMETRY", fill=(0, 229, 255, 240))

        space_gained = metrics.get("space_gained_m", 0.0)
        pass_dist = metrics.get("pass_distance_m", 28.4)
        active_cnt = metrics.get("total_tracked_players", len(stage_data.get("players", [])))

        telemetry_line1 = f"Space Gained: +{space_gained:.1f}m   |   Pass Length: {pass_dist:.1f}m"
        telemetry_line2 = f"Tracked Players: {active_cnt} (MOT Sub-Pixel Turf Grounded)"
        draw.text((cx0 + 14, cy0 + 32), telemetry_line1, fill=(240, 240, 240, 240))
        draw.text((cx0 + 14, cy0 + 52), telemetry_line2, fill=(180, 190, 205, 220))

    def _resolve_badge_collisions(self, badges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Multi-tier anti-collision algorithm:
        Detects actual 2D bounding box overlaps between badges.
        Staggers colliding badges cleanly into vertical tiers with leader stems.
        """
        if len(badges) <= 1:
            return badges

        # Sort left to right
        badges = sorted(badges, key=lambda b: (b["x"], b["y"]))

        # 1. Subtle lateral fan-out for close player clusters to create natural breathing room
        for i in range(len(badges)):
            for j in range(i + 1, len(badges)):
                b1 = badges[i]
                b2 = badges[j]
                dx = b2["x"] - b1["x"]
                dy = abs(b2["y"] - b1["y"])
                if 0 <= dx < 65.0 and dy < 70.0:
                    shift = min(22.0, (65.0 - dx) / 2.0)
                    b1["x"] -= shift
                    b2["x"] += shift

        # 2. Iterative vertical repulsion: repeat until no collisions or max 10 iterations
        for _ in range(10):
            has_collision = False
            for i in range(len(badges)):
                for j in range(i + 1, len(badges)):
                    b1 = badges[i]
                    b2 = badges[j]

                    w1 = b1.get("w", 110)
                    w2 = b2.get("w", 110)
                    h = 24.0

                    # Check bounding box overlap
                    dx = abs(b1["x"] - b2["x"])
                    min_dx = (w1 + w2) / 2.0 + 6.0

                    dy = abs(b1["y"] - b2["y"])
                    min_dy = h + 6.0

                    if dx < min_dx and dy < min_dy:
                        has_collision = True
                        # Push the overlapping badge higher up
                        if b2["y"] <= b1["y"]:
                            b2["y"] = b1["y"] - min_dy
                        else:
                            b1["y"] = b2["y"] - min_dy

            if not has_collision:
                break

        # Ensure all badges stay within top canvas boundary (min y = 26px)
        for b in badges:
            if b["y"] < 26.0:
                b["y"] = 26.0

        return badges

    def _draw_floor_ring(
        self,
        draw: ImageDraw.ImageDraw,
        x: float,
        y: float,
        color_theme: Dict[str, Tuple[int, int, int]],
        is_carrier: bool
    ):
        """Draws an authentic broadcast perspective floor tracking ring with glow aura."""
        rx = 32 if not is_carrier else 42
        ry = 11 if not is_carrier else 15
        prim = color_theme["primary"]
        acc = color_theme["accent"]

        # Outer soft glow
        draw.ellipse([x - rx - 5, y - ry - 3, x + rx + 5, y + ry + 3], outline=(*prim, 90), width=4)
        # Sharp team accent ring
        draw.ellipse([x - rx, y - ry, x + rx, y + ry], outline=(*acc, 250), width=2)

        if is_carrier:
            # Concentric pulse fill
            draw.ellipse([x - rx + 6, y - ry + 3, x + rx - 6, y + ry - 3], fill=(*acc, 65), outline=(255, 255, 255, 230), width=2)
            # Center contact point dot
            draw.ellipse([x - 3, y - 2, x + 3, y + 2], fill=(255, 255, 255, 255))

    def _draw_spotlight(
        self,
        draw: ImageDraw.ImageDraw,
        x: float,
        y: float,
        color_theme: Dict[str, Tuple[int, int, int]]
    ):
        """Draws a dramatic volumetric vertical broadcast spotlight cone descending onto ball carrier."""
        top_y = max(0.0, y - 300)
        top_rx = 14
        bot_rx = 44
        acc = color_theme["accent"]

        cone_pts = [
            (x - top_rx, top_y),
            (x + top_rx, top_y),
            (x + bot_rx, y),
            (x - bot_rx, y)
        ]
        # Volumetric cone with soft gradient simulation
        draw.polygon(cone_pts, fill=(*acc, 28))
        # Inner brighter core
        core_pts = [
            (x - top_rx * 0.5, top_y),
            (x + top_rx * 0.5, top_y),
            (x + bot_rx * 0.5, y),
            (x - bot_rx * 0.5, y)
        ]
        draw.polygon(core_pts, fill=(*acc, 22))

    def _draw_player_hud_badge(
        self,
        draw: ImageDraw.ImageDraw,
        b_info: Any,
        *legacy_args
    ):
        """
        Draws sleek modern TV broadcast floating player card:
        - Connecting vertical leader line/stem to player's head
        - Jersey number accent pill
        - Crisp player name and tactical role
        - High-contrast glassmorphic backdrop
        """
        if legacy_args:
            x = b_info
            y = legacy_args[0]
            player_data = legacy_args[1]
            color_theme = legacy_args[2] if len(legacy_args) > 2 else self.TEAM_COLORS["default_home"]
            b_info = {
                "x": x,
                "y": y,
                "player": player_data,
                "team_color": color_theme,
                "head_x": x,
                "head_y": y + 20
            }
        x = b_info["x"]
        y = b_info["y"]
        head_x = b_info.get("head_x", x)
        head_y = b_info.get("head_y", y + 20)
        player_data = b_info["player"]
        color_theme = b_info["team_color"]

        res = player_data.get("resolved", {})
        display_name = sanitize_text(res.get("display_name") or player_data.get("name") or player_data.get("id", "Player"))
        num_str = str(res.get("number") or player_data.get("number") or player_data.get("jersey_number") or "-")
        pos_str = sanitize_text(res.get("position") or player_data.get("role") or player_data.get("tactical_role_tag") or player_data.get("tactical_role") or "")

        # Compute badge width cleanly
        name_len = len(display_name)
        role_len = len(pos_str)
        badge_w = max(90, 32 + name_len * 7 + (role_len * 6 + 10 if pos_str else 0) + 12)
        badge_h = 22

        x0 = x - badge_w / 2.0
        y0 = y - badge_h
        x1 = x + badge_w / 2.0
        y1 = y

        prim = color_theme["primary"]
        acc = color_theme["accent"]

        # 1. Draw Leader Stem connecting badge bottom to player head
        if head_y > y1 + 3:
            # Subtle anchor stem line
            draw.line([(head_x, head_y), (x, y1)], fill=(*acc, 180), width=1)
            # Anchor pin dot on player's head
            draw.ellipse([head_x - 2, head_y - 2, head_x + 2, head_y + 2], fill=(255, 255, 255, 240), outline=(*acc, 255), width=1)

        # 2. Glassmorphic Drop Shadow and Border
        draw.rounded_rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], radius=5, fill=(0, 0, 0, 180))
        draw.rounded_rectangle([x0 - 1, y0 - 1, x1 + 1, y1 + 1], radius=4, outline=(*acc, 230), width=1)

        # 3. Badge Body (Dark Team-Tinted Primary)
        draw.rounded_rectangle([x0, y0, x1, y1], radius=3, fill=(min(40, prim[0]), min(40, prim[1]), min(50, prim[2]), 240))

        # 4. Jersey Number Left Pill
        pill_w = 22
        draw.rounded_rectangle([x0 + 2, y0 + 2, x0 + pill_w, y1 - 2], radius=2, fill=(*acc, 255))
        # Center number text inside pill
        num_offset = 6 if len(num_str) == 1 else 3
        draw.text((x0 + num_offset, y0 + 3), num_str, fill=(15, 15, 15, 255))

        # 5. Player Name
        name_x = x0 + pill_w + 6
        draw.text((name_x, y0 + 3), display_name, fill=(255, 255, 255, 255))

        # 6. Tactical Role Pill (if available)
        if pos_str:
            role_x = x1 - (len(pos_str) * 6 + 10)
            draw.rounded_rectangle([role_x, y0 + 3, x1 - 3, y1 - 3], radius=2, fill=(255, 255, 255, 40))
            draw.text((role_x + 3, y0 + 4), pos_str, fill=(*acc, 255))

    def _draw_pass_vector(
        self,
        draw: ImageDraw.ImageDraw,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        pass_type: str,
        confidence: float,
        avoid_boxes: Optional[List[Tuple[float, float, float, float]]] = None
    ):
        """Draws curved glowing tactical pass vector with directional arrow and non-overlapping landing zone."""
        x0, y0 = p0
        x1, y1 = p1

        mid_x = (x0 + x1) / 2.0
        # Dynamic curve height based on distance
        dist = math.hypot(x1 - x0, y1 - y0)
        arc_height = min(60.0, max(25.0, dist * 0.12))
        mid_y = (y0 + y1) / 2.0 - arc_height

        steps = 30
        pts = []
        for i in range(steps + 1):
            t = i / steps
            bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * mid_x + t ** 2 * x1
            by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * mid_y + t ** 2 * y1
            pts.append((bx, by))

        # Outer cyan glow
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=(0, 229, 255, 80), width=8)
        # Inner laser vector
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=(255, 255, 255, 240), width=3)

        # Concentric landing target zone
        draw.ellipse([x1 - 22, y1 - 10, x1 + 22, y1 + 10], outline=(0, 229, 255, 240), width=2)
        draw.ellipse([x1 - 14, y1 - 6, x1 + 14, y1 + 6], fill=(0, 229, 255, 80), outline=(255, 255, 255, 200), width=1)
        draw.ellipse([x1 - 4, y1 - 2, x1 + 4, y1 + 2], fill=(255, 255, 255, 255))

        # Origin pulse circle at passer's foot
        draw.ellipse([x0 - 8, y0 - 8, x0 + 8, y0 + 8], outline=(0, 229, 255, 230), width=2)

        # Pass label banner: positioned above the apex of the arc
        clean_pass_type = sanitize_text(pass_type).upper()
        label_text = f"PASS >> {clean_pass_type} ({int(confidence)}%)"
        lw = len(label_text) * 7 + 16
        lh = 20

        # Adjust label_y if there are badge boxes nearby
        lbl_y = mid_y - 18
        if avoid_boxes:
            for bx0, by0, bx1, by1 in avoid_boxes:
                if abs(mid_x - (bx0 + bx1) / 2.0) < (lw / 2 + 30) and abs(lbl_y - (by0 + by1) / 2.0) < 25:
                    lbl_y = min(lbl_y, by0 - 24)

        draw.rounded_rectangle([mid_x - lw / 2 - 1, lbl_y - 1, mid_x + lw / 2 + 1, lbl_y + lh + 1], radius=4, fill=(0, 0, 0, 200))
        draw.rounded_rectangle([mid_x - lw / 2, lbl_y, mid_x + lw / 2, lbl_y + lh], radius=4, fill=(10, 25, 47, 240), outline=(0, 229, 255, 255), width=1)
        draw.text((mid_x - lw / 2 + 8, lbl_y + 3), label_text, fill=(255, 255, 255, 255))

    def generate_animated_svg(
        self,
        width: int,
        height: int,
        frame_analysis: Dict[str, Any],
        is_t0: bool = False
    ) -> str:
        """Generates resolution-independent animated SVG overlays with pulse effects."""
        svg_parts = [
            f'<svg viewBox="0 0 {width} {height}" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">',
            '''<defs>
                <style>
                    @keyframes pulse { 0% { r: 24; opacity: 0.9; } 50% { r: 38; opacity: 0.3; } 100% { r: 24; opacity: 0.9; } }
                    @keyframes dash { to { stroke-dashoffset: -100; } }
                    .pass-vector { stroke-dasharray: 8, 4; animation: dash 2s linear infinite; }
                    .carrier-pulse { animation: pulse 1.8s ease-in-out infinite; }
                    .hud-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; font-weight: 700; }
                </style>
                <radialGradient id="spotlightGlow" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stop-color="#EDBB00" stop-opacity="0.4"/>
                    <stop offset="100%" stop-color="#EDBB00" stop-opacity="0.0"/>
                </radialGradient>
            </defs>'''
        ]

        if is_t0 and "pass_prediction" in frame_analysis:
            pred = frame_analysis["pass_prediction"]
            orig = pred.get("pass_origin_norm", [500, 500])
            dest = pred.get("pass_destination_norm", [600, 400])
            x0, y0 = orig[0] * width / 1000.0, orig[1] * height / 1000.0
            x1, y1 = dest[0] * width / 1000.0, dest[1] * height / 1000.0
            mid_x, mid_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0 - 40.0

            svg_parts.append(
                f'<path d="M {x0:.1f} {y0:.1f} Q {mid_x:.1f} {mid_y:.1f} {x1:.1f} {y1:.1f}" '
                f'fill="none" stroke="#00E5FF" stroke-width="4" class="pass-vector" />'
            )
            svg_parts.append(
                f'<ellipse cx="{x1:.1f}" cy="{y1:.1f}" rx="24" ry="10" fill="none" stroke="#00E5FF" stroke-width="3"/>'
            )

        for p in frame_analysis.get("players", []):
            foot = p.get("foot_pos_norm", [500, 500])
            bbox = p.get("bbox_norm", [0, 0, 100, 100])
            fx = foot[0] * width / 1000.0
            fy = foot[1] * height / 1000.0
            is_carrier = p.get("is_ball_carrier", False)
            res = p.get("resolved", {})
            name = sanitize_text(res.get("display_name", p.get("id", "Player")))
            num = res.get("number", "-")

            stroke_color = "#EDBB00" if is_carrier else "#004D98"
            fill_color = "rgba(237, 187, 0, 0.25)" if is_carrier else "rgba(0, 77, 152, 0.15)"

            svg_parts.append(
                f'<ellipse cx="{fx:.1f}" cy="{fy:.1f}" rx="32" ry="12" fill="{fill_color}" stroke="{stroke_color}" stroke-width="2.5"/>'
            )

            if is_carrier:
                svg_parts.append(
                    f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="26" fill="url(#spotlightGlow)" class="carrier-pulse"/>'
                )

            badge_y = bbox[0] * height / 1000.0 - 15
            svg_parts.append(
                f'<g transform="translate({fx:.1f}, {badge_y:.1f})">'
                f'<rect x="-55" y="-18" width="110" height="22" rx="4" fill="rgba(10,25,47,0.85)" stroke="{stroke_color}" stroke-width="1.5"/>'
                f'<text x="0" y="-3" text-anchor="middle" fill="#FFFFFF" class="hud-text">#{num} {name}</text>'
                f'</g>'
            )

        svg_parts.append('</svg>')
        return "\n".join(svg_parts)
