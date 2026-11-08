"""
Smart Collage Batcher & Pass-2 Identity Resolver.
- Splits Pass 2 into TWO separate compact 2D Grid Collages (Home vs Away) to prevent resolution crushing.
- Upscales crops 2x with cv2.INTER_CUBIC + unsharp mask to maximize jersey number readability.
- Formats Track IDs strictly as T{id} (never '#' prefix) to prevent ID -> jersey number contamination.
- Removes synthetic full-pitch radar maps; relies strictly on annotated Context Frame.
- Enforces Hard Team Firewalls and Strict 1-to-1 Bijection (Hungarian algorithm on remaining slots).
"""

from dataclasses import dataclass, field
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.request
import base64

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .bot_sort_tracker import TrackletLifecycle, TrackletCrop
from .team_color_clusterer import TeamClusterResult
from ..pitch_geometry import PitchGeometryEngine


@dataclass
class TeamGridPayload:
    """Team-specific 2D grid payload for Pass-2 VLM Identity Resolution."""
    shot_id: int
    team_label: str  # "home" or "away"
    team_name: str   # "Real Madrid" or "Barcelona"
    track_ids: List[int]
    grid_img: np.ndarray
    context_frame_img: np.ndarray
    tracklet_metadata: List[Dict[str, Any]]
    active_roster: List[Dict[str, Any]]


class SmartCollageBatcher:
    """
    Assembles Pass-2 2D grid collages and executes team-firewalled, 1-to-1 unique identity assignment.
    """

    def __init__(
        self,
        crop_h: int = 150,
        crop_w: int = 80,
        cols_per_grid: int = 3,
        min_tracklet_duration_sec: float = 0.3,
        geom_engine: Optional[PitchGeometryEngine] = None,
    ):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.cols_per_grid = cols_per_grid
        self.min_tracklet_duration_sec = min_tracklet_duration_sec
        self.geom_engine = geom_engine or PitchGeometryEngine()

    def render_player_card(
        self,
        track_id: int,
        team_str: str,
        crops: List[TrackletCrop],
        rel_pos: Tuple[float, float],
    ) -> np.ndarray:
        """
        Render a 2D card for one tracked player:
        Header: Track: T{track_id} | Pos: ({rx:.2f}, {ry:.2f})
        Body: 3 crops (Early, Peak, Late) upscaled 2x and sharpened.
        """
        # Card dimensions: 3 crops of width crop_w + margins + card header
        card_w = self.crop_w * 3 + 24
        header_h = 36
        badge_h = 18
        card_h = header_h + self.crop_h + badge_h + 12

        card = np.full((card_h, card_w, 3), 36, dtype=np.uint8)

        # Team color banner
        if team_str == "home":
            badge_color = (245, 245, 245)
            text_color = (15, 15, 15)
        elif team_str == "away":
            badge_color = (140, 40, 40)
            text_color = (255, 255, 255)
        elif "gk" in team_str:
            badge_color = (0, 140, 255)
            text_color = (0, 0, 0)
        else:
            badge_color = (0, 220, 220)
            text_color = (0, 0, 0)

        # Header background
        cv2.rectangle(card, (2, 2), (card_w - 2, header_h), badge_color, -1)
        cv2.rectangle(card, (2, 2), (card_w - 2, card_h - 2), (90, 90, 90), 1)

        # Header Text: STRICTLY "T{id}" - NO '#'
        track_lbl = f"Track: T{track_id}"
        pos_lbl = f"Pos: ({rel_pos[0]:.2f}, {rel_pos[1]:.2f})"
        cv2.putText(card, track_lbl, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, text_color, 2)
        cv2.putText(card, pos_lbl, (card_w - 110, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1)

        # Draw 3 crops (Early, Peak, Late)
        start_y = header_h + 6
        for i, crop_obj in enumerate(crops[:3]):
            cx = 6 + i * (self.crop_w + 6)
            cy = start_y

            # 2x upscale with INTER_CUBIC
            raw_crop = crop_obj.crop_img
            resized = cv2.resize(raw_crop, (self.crop_w, self.crop_h), interpolation=cv2.INTER_CUBIC)

            # Unsharp mask to make jersey numbers pop
            gaussian = cv2.GaussianBlur(resized, (0, 0), 2.0)
            sharpened = cv2.addWeighted(resized, 1.4, gaussian, -0.4, 0)

            card[cy:cy + self.crop_h, cx:cx + self.crop_w] = sharpened
            cv2.rectangle(card, (cx, cy), (cx + self.crop_w, cy + self.crop_h), (140, 140, 140), 1)

            # Sub-badge (EARLY, PEAK, LATE)
            badge_txt = f"{crop_obj.label.upper()} {int(crop_obj.position_pct * 100)}%"
            cv2.putText(
                card, badge_txt, (cx + 2, cy + self.crop_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1
            )

        return card

    def render_team_2d_grid(
        self,
        cards: List[np.ndarray],
        cols: int = 3,
        pad: int = 12,
    ) -> np.ndarray:
        """
        Arrange player cards into a compact 2D grid (e.g. 3 columns x N rows).
        Keeps aspect ratio near 4:3 or 16:9 to avoid VLM downscale crushing.
        """
        if not cards:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        card_h, card_w = cards[0].shape[:2]
        n_cards = len(cards)
        rows = int(np.ceil(n_cards / float(cols)))

        grid_w = cols * card_w + (cols + 1) * pad
        grid_h = rows * card_h + (rows + 1) * pad
        grid = np.full((grid_h, grid_w, 3), 24, dtype=np.uint8)

        for idx, card in enumerate(cards):
            r = idx // cols
            c = idx % cols
            y = pad + r * (card_h + pad)
            x = pad + c * (card_w + pad)
            grid[y:y + card_h, x:x + card_w] = card

        return grid

    def render_context_frame(
        self,
        keyframe: np.ndarray,
        active_tracklets: List[Tuple[int, List[float], str]],  # (track_id, bbox, team)
        max_dim: int = 1280,
    ) -> np.ndarray:
        """
        Draw numbered bounding boxes with team colors on the Context Frame.
        FORMATS TRACK LABELS STRICTLY AS "T{id}" - NEVER "#".
        """
        ctx = keyframe.copy()
        h, w = ctx.shape[:2]

        for track_id, bbox, team in active_tracklets:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if team == "home":
                color = (255, 255, 255)
                text_col = (0, 0, 0)
            elif team == "away":
                color = (60, 60, 220)
                text_col = (255, 255, 255)
            elif "gk" in team:
                color = (0, 165, 255)
                text_col = (0, 0, 0)
            elif team == "referee":
                color = (0, 240, 240)
                text_col = (0, 0, 0)
            else:
                color = (0, 255, 0)
                text_col = (0, 0, 0)

            # Draw bounding box
            cv2.rectangle(ctx, (x1, y1), (x2, y2), color, 2)

            # STRICTLY "T{id}" - NEVER "#"
            lbl = f"T{track_id}"
            t_size = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
            cv2.rectangle(ctx, (x1, max(0, y1 - 22)), (x1 + t_size[0] + 6, y1), color, -1)
            cv2.putText(
                ctx, lbl, (x1 + 3, max(14, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_col, 2
            )

        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            ctx = cv2.resize(ctx, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        return ctx

    def build_team_grid_payloads(
        self,
        shot_id: int,
        lifecycles: Dict[int, TrackletLifecycle],
        team_assignments: Dict[int, TeamClusterResult],
        frame_cache: Dict[int, np.ndarray],
        active_roster: Dict[str, Any],
        fps: float = 30.0,
    ) -> Tuple[Optional[TeamGridPayload], Optional[TeamGridPayload], np.ndarray]:
        """
        Build TWO separate team payloads (Home vs Away) with 2D grids and a single Context Frame.
        """
        valid_lifecycles = [
            lc for lc in lifecycles.values()
            if lc.duration_seconds(fps) >= self.min_tracklet_duration_sec
        ]

        if not valid_lifecycles:
            return None, None, np.zeros((100, 100, 3), dtype=np.uint8)

        # Context keyframe: median frame
        cached_keys = sorted(frame_cache.keys())
        mid_frame_idx = cached_keys[len(cached_keys) // 2]
        keyframe = frame_cache[mid_frame_idx]

        # Gather active boxes at keyframe
        active_boxes_at_keyframe = []
        for lc in valid_lifecycles:
            if mid_frame_idx in lc.frame_indices:
                b_idx = lc.frame_indices.index(mid_frame_idx)
                bbox = lc.bboxes[b_idx]
                t_str = team_assignments.get(lc.track_id, TeamClusterResult(lc.track_id, "unknown", 0.5, False)).team
                active_boxes_at_keyframe.append((lc.track_id, bbox, t_str))

        context_frame = self.render_context_frame(keyframe, active_boxes_at_keyframe)

        # Separate lifecycles into Home vs Away groups
        home_lcs: List[TrackletLifecycle] = []
        away_lcs: List[TrackletLifecycle] = []

        # Derive camera zoom scale s_x from median player bounding box height (Bug 3 fix)
        # Wide tactical cam: h_med ~ 85px -> s_x ~ 0.50; Tighter zoom: h_med ~ 130px -> s_x ~ 0.32
        h_frame, w_frame = keyframe.shape[:2]
        all_box_heights = [
            (b[3] - b[1])
            for lc in valid_lifecycles
            for b in lc.bboxes
            if (b[3] - b[1]) > 10
        ]
        h_med = float(np.median(all_box_heights)) if all_box_heights else 85.0
        h_med_1080p = h_med * (1080.0 / max(1.0, float(h_frame)))
        s_x = float(np.clip(42.5 / max(1.0, h_med_1080p), 0.30, 0.75))

        # Pitch turf hull bounds for lateral Y mapping (Bug 4 fix)
        pitch_geom = self.geom_engine.analyze_frame(keyframe) if (self.geom_engine and keyframe is not None and keyframe.size > 0) else None

        for lc in valid_lifecycles:
            tid = lc.track_id
            t_res = team_assignments.get(tid)
            t_str = t_res.team if t_res else "unknown"

            if t_str in ["home", "gk_home"]:
                home_lcs.append(lc)
            elif t_str in ["away", "gk_away"]:
                away_lcs.append(lc)
            elif t_str == "referee":
                pass  # Handled below to enforce strictly Max 1 Referee

        # Enforce strictly Max 1 Match Referee on the pitch
        ref_lcs = [
            lc for lc in valid_lifecycles
            if team_assignments.get(lc.track_id, TeamClusterResult(lc.track_id, "unknown", 0.5, False)).team == "referee"
        ]
        if ref_lcs:
            best_ref = min(
                ref_lcs,
                key=lambda lc: ((lc.get_average_center()[0] - 0.5 * 1920) ** 2 + (lc.get_average_center()[1] - 0.5 * 1080) ** 2)
            )
            best_ref.assigned_name = "Match Referee"
            best_ref.assigned_number = 0
            best_ref.assigned_confidence = 0.95

        # Render Home Grid
        home_payload = None
        if home_lcs:
            home_lcs.sort(key=lambda lc: lc.get_average_center()[0])
            home_cards = []
            home_meta = []
            for lc in home_lcs:
                tid = lc.track_id
                t_res = team_assignments.get(tid)
                t_str = t_res.team if t_res else "home"
                crops = lc.extract_3_angle_crops(frame_cache)
                if not crops:
                    continue
                cx, cy = lc.get_average_center()
                # Normalized coordinate mapping without min-max stretch
                rx = float(np.clip(cx / float(w_frame), 0.0, 1.0))
                top_y, bot_y = PitchGeometryEngine.get_turf_hull_bounds(
                    pitch_geom=pitch_geom, frame_shape=(h_frame, w_frame), x_px=cx
                )
                y_span = max(0.10, bot_y - top_y)
                ry = float(np.clip((cy / float(h_frame) - top_y) / y_span, 0.0, 1.0))
                card = self.render_player_card(tid, t_str, crops, (rx, ry))
                home_cards.append(card)
                home_meta.append({
                    "track_label": f"T{tid}",
                    "track_id": tid,
                    "team": t_str,
                    "relative_x": round(rx, 3),
                    "relative_y": round(ry, 3),
                    "scale_sx": round(s_x, 3),
                })
            if home_cards:
                home_grid = self.render_team_2d_grid(home_cards, cols=self.cols_per_grid)
                home_payload = TeamGridPayload(
                    shot_id=shot_id,
                    team_label="home",
                    team_name=active_roster.get("home_team", "Real Madrid"),
                    track_ids=[m["track_id"] for m in home_meta],
                    grid_img=home_grid,
                    context_frame_img=context_frame,
                    tracklet_metadata=home_meta,
                    active_roster=active_roster.get("home_active_xi", []),
                )

        # Render Away Grid
        away_payload = None
        if away_lcs:
            away_lcs.sort(key=lambda lc: lc.get_average_center()[0])
            away_cards = []
            away_meta = []
            for lc in away_lcs:
                tid = lc.track_id
                t_res = team_assignments.get(tid)
                t_str = t_res.team if t_res else "away"
                crops = lc.extract_3_angle_crops(frame_cache)
                if not crops:
                    continue
                cx, cy = lc.get_average_center()
                rx = float(np.clip(cx / float(w_frame), 0.0, 1.0))
                top_y, bot_y = PitchGeometryEngine.get_turf_hull_bounds(
                    pitch_geom=pitch_geom, frame_shape=(h_frame, w_frame), x_px=cx
                )
                y_span = max(0.10, bot_y - top_y)
                ry = float(np.clip((cy / float(h_frame) - top_y) / y_span, 0.0, 1.0))
                card = self.render_player_card(tid, t_str, crops, (rx, ry))
                away_cards.append(card)
                away_meta.append({
                    "track_label": f"T{tid}",
                    "track_id": tid,
                    "team": t_str,
                    "relative_x": round(rx, 3),
                    "relative_y": round(ry, 3),
                    "scale_sx": round(s_x, 3),
                })
            if away_cards:
                away_grid = self.render_team_2d_grid(away_cards, cols=self.cols_per_grid)
                away_payload = TeamGridPayload(
                    shot_id=shot_id,
                    team_label="away",
                    team_name=active_roster.get("away_team", "Barcelona"),
                    track_ids=[m["track_id"] for m in away_meta],
                    grid_img=away_grid,
                    context_frame_img=context_frame,
                    tracklet_metadata=away_meta,
                    active_roster=active_roster.get("away_active_xi", []),
                )

        return home_payload, away_payload, context_frame

    def resolve_team_identities(
        self,
        payload: TeamGridPayload,
        output_dir: str,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Query Gemini Flash for ONE team's 2D grid collage against that team's strict roster.
        Applies Hard Team Firewall and Strict 1-to-1 Bijection (Hungarian Matching).
        """
        os.makedirs(output_dir, exist_ok=True)
        grid_path = os.path.join(output_dir, f"shot_{payload.shot_id:02d}_{payload.team_label}_grid.png")
        context_path = os.path.join(output_dir, f"shot_{payload.shot_id:02d}_context.png")

        cv2.imwrite(grid_path, payload.grid_img)
        cv2.imwrite(context_path, payload.context_frame_img)

        # Separate GK track from outfield tracks
        gk_track_id: Optional[int] = None
        for meta in payload.tracklet_metadata:
            if "gk" in meta["team"]:
                gk_track_id = meta["track_id"]

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        vlm_results: List[Dict[str, Any]] = []

        if api_key:
            vlm_results = self._query_gemini_for_team(payload, grid_path, context_path, api_key)

        # Enforce Hard Team Firewall & 1-to-1 Bijection in Python
        final_assignments = self._enforce_strict_1to1_bijection(
            payload=payload,
            vlm_results=vlm_results,
            gk_track_id=gk_track_id,
        )

        # Save diagnostics
        diag_path = os.path.join(output_dir, f"shot_{payload.shot_id:02d}_{payload.team_label}_identities.json")
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(final_assignments, f, indent=2)

        return final_assignments

    def _query_gemini_for_team(
        self,
        payload: TeamGridPayload,
        grid_path: str,
        context_path: str,
        api_key: str,
    ) -> List[Dict[str, Any]]:
        """Direct REST call to Gemini Flash for a single team's grid."""
        try:
            kit_desc = "pure WHITE" if payload.team_label == "home" else "BLAUGRANA (Blue/Red stripes)"
            roster_json = json.dumps(payload.active_roster, indent=2)

            if payload.team_label == "home":
                flank_guidance = """
TACTICAL FLANK ORIENTATION (Real Madrid attacking Left-to-Right):
- TOP of screen (y -> 0.0) is Real Madrid's LEFT FLANK: Left-Back (Ferland Mendy #23, Fran Garcia #20), Left-Wing (Vinicius Junior #7).
- BOTTOM of screen (y -> 1.0) is Real Madrid's RIGHT FLANK: Right-Back (Lucas Vazquez #17), Right-Midfield (Federico Valverde #8).
- Real Madrid Left-Back/Wing CANNOT be at the bottom of the screen!
"""
            else:
                flank_guidance = """
TACTICAL FLANK ORIENTATION (Barcelona attacking Right-to-Left):
- TOP of screen (y -> 0.0) is Barcelona's RIGHT FLANK: Right-Back (Jules Kounde #23), Right-Wing (Lamine Yamal #19).
- BOTTOM of screen (y -> 1.0) is Barcelona's LEFT FLANK: Left-Back (Alejandro Balde #3), Left-Wing (Raphinha #11).
- Barcelona Left-Back (Alejandro Balde #3) is at the BOTTOM of the screen!
"""

            prompt_text = f"""
You are a broadcast football tactical analyst identifying players for {payload.team_name} in El Clasico.

Images provided:
1. Context Frame (Image 1): Broadcast camera shot showing active {payload.team_name} players labeled as T{{id}} (e.g. T8, T2, T11).
2. 2D Player Grid (Image 2): High-resolution crops (Early, Peak Sharpness, Late) for each {payload.team_name} track.

CRITICAL READING CONSTRAINTS:
1. TRACK LABELS ARE NOT JERSEY NUMBERS: The label "T11" means Track #11, NOT Jersey #11! The label "T8" means Track #8, NOT Jersey #8! Never confuse T{{id}} with a shirt number.
2. READ PHYSICAL PRINTED DIGITS: Look closely at the player's back, shorts, and chest in the 3 crops for the actual printed digits (e.g. 17, 3, 8, 16, 19, 9, 7).
3. STRICT ROSTER RESTRICTION: You may ONLY pick players from this {payload.team_name} active roster:
{roster_json}
Kit Appearance: {payload.team_name} wears {kit_desc}. Goalkeeper wears orange/distinctive kit.
4. PARTIAL VISIBILITY: Notice only {len(payload.track_ids)} players of {payload.team_name} are visible in this camera view. Do NOT force all 11 players.
{flank_guidance}
Return a strict JSON array of objects mapping each visible track to their player name:
[
  {{
    "track_id": integer (e.g. 8),
    "track_label": "T8",
    "player_name": string (exact name from the active roster),
    "jersey_number": integer (printed digit read from shirt/shorts),
    "confidence": float between 0.50 and 0.99,
    "reasoning": string (e.g. "Visible #17 on shirt back, defensive midfield position")
  }}
]

Output valid JSON ONLY.
"""

            with open(context_path, "rb") as f:
                ctx_b64 = base64.b64encode(f.read()).decode("utf-8")
            with open(grid_path, "rb") as f:
                grid_b64 = base64.b64encode(f.read()).decode("utf-8")

            parts = [
                {"inline_data": {"mime_type": "image/png", "data": ctx_b64}},
                {"inline_data": {"mime_type": "image/png", "data": grid_b64}},
                {"text": prompt_text},
            ]

            req_payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                },
            }
            req_data = json.dumps(req_payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}

            models_to_try = ["gemini-2.5-flash", "gemini-flash-lite-latest", "gemini-1.5-flash"]
            raw_text = None

            for m_name in models_to_try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={api_key}"
                req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=18) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        candidates = data.get("candidates", [])
                        if candidates:
                            raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                            if raw_text:
                                break
                except Exception as ex:
                    continue

            if not raw_text:
                return []

            clean_text = raw_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:].strip()
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:].strip()
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3].strip()

            parsed = json.loads(clean_text)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict) and "tracks" in parsed:
                return parsed["tracks"]
            return []

        except Exception as e:
            print(f"[SmartCollageBatcher] VLM call failed for {payload.team_name}: {e}")
            return []

    def _enforce_strict_1to1_bijection(
        self,
        payload: TeamGridPayload,
        vlm_results: List[Dict[str, Any]],
        gk_track_id: Optional[int],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Hard Python Enforcement:
        1. Hard Team Firewall: Drop any player not in this team's active roster.
        2. Goalkeeper Lock: Bind GK track to team's goalkeeper with conf=0.99.
        3. Strict 1-to-1 Uniqueness: Greedy lock for visible jersey numbers (conf >= 0.90),
           then Hungarian algorithm (linear_sum_assignment) on remaining slots.
        """
        roster_by_name = {p["name"].lower(): p for p in payload.active_roster}
        roster_by_num = {p["number"]: p for p in payload.active_roster}

        final_assignments: Dict[int, Dict[str, Any]] = {}
        assigned_names: Set[str] = set()

        # Step 1: Hard Goalkeeper Lock
        gk_player = next((p for p in payload.active_roster if p.get("role") == "GK"), None)
        if gk_track_id is not None and gk_player:
            final_assignments[gk_track_id] = {
                "track_id": gk_track_id,
                "track_label": f"T{gk_track_id}",
                "player_name": gk_player["name"],
                "jersey_number": gk_player["number"],
                "team": f"gk_{payload.team_label}",
                "confidence": 0.99,
                "reasoning": "Goalkeeper kit and goal area spatial lock",
            }
            assigned_names.add(gk_player["name"].lower())

        # Step 2: Process VLM candidate assignments
        # Filter: Must be in team roster and not GK
        valid_candidates = []
        for item in vlm_results:
            tid = int(item.get("track_id", 0))
            if tid == gk_track_id:
                continue

            raw_name = str(item.get("player_name", "")).strip()
            raw_num = item.get("jersey_number")

            matched_p = None
            if raw_name.lower() in roster_by_name:
                matched_p = roster_by_name[raw_name.lower()]
            elif raw_num in roster_by_num:
                matched_p = roster_by_num[raw_num]

            if matched_p and matched_p.get("role") != "GK":
                conf = float(item.get("confidence", 0.80))
                reason = str(item.get("reasoning", "VLM identification"))
                valid_candidates.append((tid, matched_p, conf, reason))

        # Sort by confidence descending (high-confidence printed numbers first)
        valid_candidates.sort(key=lambda x: x[2], reverse=True)

        # Greedy lock on high-confidence unique players
        meta_by_id = {m["track_id"]: m for m in payload.tracklet_metadata}
        for tid, p, conf, reason in valid_candidates:
            if tid in final_assignments:
                continue
            p_name = p["name"]
            role = p.get("role", "CM")
            t_meta = meta_by_id.get(tid, {})
            t_ry = t_meta.get("relative_y", 0.5)

            # Geometric flank firewall (unless super high confidence visible number >= 0.95)
            if conf < 0.95:
                if payload.team_label == "home":
                    # Real Madrid: LB/LW must be top (t_ry < 0.60); RB/RW must be bottom (t_ry > 0.40)
                    if role in ["LB", "LW"] and t_ry > 0.60:
                        continue
                    if role in ["RB", "RW"] and t_ry < 0.40:
                        continue
                else:
                    # Barcelona: LB/LW (Balde) must be bottom (t_ry > 0.40); RB/RW (Yamal) must be top (t_ry < 0.60)
                    if role in ["LB", "LW"] and t_ry < 0.40:
                        continue
                    if role in ["RB", "RW"] and t_ry > 0.60:
                        continue

            if p_name.lower() not in assigned_names:
                final_assignments[tid] = {
                    "track_id": tid,
                    "track_label": f"T{tid}",
                    "player_name": p_name,
                    "jersey_number": p["number"],
                    "team": payload.team_label,
                    "confidence": conf,
                    "reasoning": reason,
                }
                assigned_names.add(p_name.lower())

        # Step 3: Hungarian Bipartite Matching for any remaining unassigned tracks
        unassigned_track_ids = [tid for tid in payload.track_ids if tid not in final_assignments]
        available_outfield = [
            p for p in payload.active_roster
            if p.get("role") != "GK" and p["name"].lower() not in assigned_names
        ]

        if unassigned_track_ids and available_outfield:
            # Formation nominal relative coordinates:
            # Home (Real Madrid defending left): LB is top/left, RB is bottom/left, CB central, CM mid, ST right
            # Away (Barcelona defending right): RB is top/right, LB is bottom/right, CB central, CM mid, ST left
            cost_matrix = np.zeros((len(unassigned_track_ids), len(available_outfield)), dtype=np.float32)

            # Map tracks to (rx, ry)
            # Map tracks to (rx, ry)
            meta_by_id = {m["track_id"]: m for m in payload.tracklet_metadata}

            # Determine camera translation t_x for affine mapping: x_pitch_hat = s_x * rx + t_x
            gk_m = next((m for m in payload.tracklet_metadata if m.get("team") in ("gk_home", "gk_away")), None)
            f_mid_x = 0.50
            unassigned_xs = [
                meta_by_id[tid].get("relative_x", 0.5)
                for tid in unassigned_track_ids
                if tid in meta_by_id
            ]
            screen_mid_x = float(np.median(unassigned_xs)) if unassigned_xs else 0.50
            global_sx = float(payload.tracklet_metadata[0].get("scale_sx", 0.50)) if payload.tracklet_metadata else 0.50

            if gk_m is not None:
                gk_pitch_x = 0.05 if payload.team_label == "home" else 0.95
                t_x = gk_pitch_x - global_sx * float(gk_m.get("relative_x", 0.5))
            else:
                t_x = f_mid_x - global_sx * screen_mid_x

            for i, tid in enumerate(unassigned_track_ids):
                m = meta_by_id.get(tid, {})
                rx = m.get("relative_x", 0.5)
                ry = m.get("relative_y", 0.5)
                player_sx = float(m.get("scale_sx", global_sx))

                # Module 4: Affine coordinate mapping x_pitch_hat = s_x * x_screen + t_x
                x_pitch_hat = float(np.clip(player_sx * rx + t_x, 0.0, 1.0))

                for j, p in enumerate(available_outfield):
                    role = p.get("role", "CM")
                    # Target nominal (fx, fy) based on role
                    if payload.team_label == "home":
                        # Defending left goal, attacking left-to-right (+X)
                        # TOP of screen (y -> 0.0) is LEFT flank (LB, LCB, LW)
                        # BOTTOM of screen (y -> 1.0) is RIGHT flank (RB, RCB, RW)
                        fx_map = {"GK": 0.05, "LB": 0.22, "LCB": 0.20, "CB": 0.20, "RCB": 0.20, "RB": 0.22, "DM": 0.40, "LCM": 0.50, "CM": 0.50, "RCM": 0.50, "AM": 0.65, "LW": 0.75, "CF": 0.85, "RW": 0.75}
                        fy_map = {"GK": 0.50, "LB": 0.20, "LCB": 0.38, "CB": 0.50, "RCB": 0.62, "RB": 0.80, "DM": 0.50, "LCM": 0.35, "CM": 0.50, "RCM": 0.65, "AM": 0.50, "LW": 0.20, "CF": 0.50, "RW": 0.80}
                    else:
                        # Defending right goal, attacking right-to-left (-X)
                        # TOP of screen (y -> 0.0) is RIGHT flank (RB, RCB, RW - Yamal #19, Kounde #23)
                        # BOTTOM of screen (y -> 1.0) is LEFT flank (LB, LCB, LW - Balde #3, Raphinha #11)
                        fx_map = {"GK": 0.95, "RB": 0.78, "RCB": 0.80, "CB": 0.80, "LCB": 0.80, "LB": 0.78, "DM": 0.60, "RCM": 0.50, "CM": 0.50, "LCM": 0.50, "AM": 0.35, "RW": 0.25, "CF": 0.15, "LW": 0.25}
                        fy_map = {"GK": 0.50, "RB": 0.20, "RCB": 0.38, "CB": 0.50, "LCB": 0.62, "LB": 0.80, "DM": 0.50, "RCM": 0.35, "CM": 0.50, "LCM": 0.65, "AM": 0.50, "RW": 0.20, "CF": 0.50, "LW": 0.80}

                    fx = fx_map.get(role, 0.5)
                    fy = fy_map.get(role, 0.5)

                    # True 2D Euclidean distance in pitch coordinate space
                    dx = x_pitch_hat - fx
                    dy = ry - fy
                    cost_matrix[i, j] = (dx ** 2 + dy ** 2) ** 0.5

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            swap_margins = compute_hungarian_2opt_swap_margins(cost_matrix, row_ind, col_ind)

            for r, c in zip(row_ind, col_ind):
                tid = unassigned_track_ids[r]
                p = available_outfield[c]
                margin = swap_margins.get(r, 1.0)
                is_ambiguous = (margin < 0.15)

                reason_str = f"Hungarian topological formation match ({p.get('role', 'CM')})"
                if is_ambiguous:
                    reason_str += f" [AMBIGUITY FLAG: 2-opt swap margin={margin:.1%} < 15%]"

                final_assignments[tid] = {
                    "track_id": tid,
                    "track_label": f"T{tid}",
                    "player_name": p["name"],
                    "jersey_number": p["number"],
                    "team": payload.team_label,
                    "confidence": 0.70 if is_ambiguous else 0.78,
                    "is_ambiguous": is_ambiguous,
                    "swap_margin": round(margin, 3),
                    "reasoning": reason_str,
                }
                assigned_names.add(p["name"].lower())

        return final_assignments


# =============================================================================
# Module 4 Algorithmic Margin & 2-Opt Pairwise Utilities
# =============================================================================

def compute_hungarian_2opt_swap_margins(
    cost_matrix: np.ndarray,
    row_ind: np.ndarray,
    col_ind: np.ndarray,
    epsilon: float = 1e-6,
) -> Dict[int, float]:
    """
    Computes the 2-opt pairwise swap margin for each assigned pair (i1, j1):
    Delta C_swap(i1, i2) = ((C[i1, j2] + C[i2, j1]) - (C[i1, j1] + C[i2, j2])) / (abs(C[i1, j1]) + abs(C[i2, j2]) + eps)
    Also evaluates 1-opt alternative match against unassigned candidate roles.

    By optimality of Hungarian assignment, numerator >= 0.
    Returns: dict mapping row_idx i1 -> min Delta C_swap(i1).
    Flag as ambiguous whenever margin < 0.15.
    """
    n_assigned = len(row_ind)
    margins: Dict[int, float] = {}

    if n_assigned < 1:
        return margins

    assignment_map = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    rows = list(assignment_map.keys())
    assigned_cols = set(assignment_map.values())
    total_cols = cost_matrix.shape[1]
    unassigned_cols = [c for c in range(total_cols) if c not in assigned_cols]

    for i1 in rows:
        j1 = assignment_map[i1]
        c11 = float(cost_matrix[i1, j1])
        min_delta = float("inf")

        # 1. 2-opt pairwise swap against all other assigned players
        for i2 in rows:
            if i1 == i2:
                continue
            j2 = assignment_map[i2]
            c22 = float(cost_matrix[i2, j2])
            c12 = float(cost_matrix[i1, j2])
            c21 = float(cost_matrix[i2, j1])

            numerator = (c12 + c21) - (c11 + c22)
            numerator = max(0.0, numerator)
            denominator = abs(c11) + abs(c22) + epsilon
            swap_margin = numerator / denominator
            if swap_margin < min_delta:
                min_delta = swap_margin

        # 2. 1-opt alternative match against unassigned roles
        for j_free in unassigned_cols:
            c_free = float(cost_matrix[i1, j_free])
            delta_free = max(0.0, c_free - c11)
            free_margin = delta_free / (abs(c11) + epsilon)
            if free_margin < min_delta:
                min_delta = free_margin

        margins[i1] = float(min_delta) if min_delta != float("inf") else 1.0

    return margins
