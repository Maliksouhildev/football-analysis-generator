"""
Smart Collage Batcher & Pass-2 Identity Resolver.
Generates multi-angle 3-crop tracklet strips (early, peak, late), a context frame with numbered boxes,
and a 2D radar minimap. Resolves all tracklet identities in ONE single Gemini Flash API call per camera shot.
"""

from dataclasses import dataclass, field
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from .bot_sort_tracker import TrackletLifecycle, TrackletCrop
from .team_color_clusterer import TeamClusterResult


@dataclass
class CollageBatchPayload:
    """Consolidated payload prepared for Pass-2 VLM Identity Resolution."""
    shot_id: int
    total_tracklets: int
    home_visible: int
    away_visible: int
    gk_visible: int
    ref_visible: int
    montage_img: np.ndarray
    context_frame_img: np.ndarray
    radar_minimap_img: np.ndarray
    tracklet_metadata: List[Dict[str, Any]]
    active_roster: Dict[str, Any]


class SmartCollageBatcher:
    """
    Assembles Pass-2 multi-angle visual strips and pitch context for 1-call VLM identity assignment.
    """

    def __init__(
        self,
        crop_target_h: int = 140,
        crop_target_w: int = 80,
        label_card_w: int = 180,
        min_tracklet_duration_sec: float = 1.0,
    ):
        self.crop_target_h = crop_target_h
        self.crop_target_w = crop_target_w
        self.label_card_w = label_card_w
        self.min_tracklet_duration_sec = min_tracklet_duration_sec

    def render_tracklet_row(
        self,
        track_id: int,
        team_info: Optional[TeamClusterResult],
        crops: List[TrackletCrop],
        rel_pos: Tuple[float, float],
    ) -> np.ndarray:
        """
        Render a single row in the collage:
        [ Label Card (ID, Team, Rel Pos) | Crop 1 (Early) | Crop 2 (Peak) | Crop 3 (Late) ]
        """
        row_h = self.crop_target_h + 30
        row_w = self.label_card_w + (self.crop_target_w + 10) * 3 + 20
        row_img = np.full((row_h, row_w, 3), 32, dtype=np.uint8)

        # Team color banner
        team_str = team_info.team if team_info else "unknown"
        if team_str == "home":
            badge_color = (240, 240, 240)  # White
            text_color = (20, 20, 20)
        elif team_str == "away":
            badge_color = (130, 40, 40)   # Blaugrana Red/Blue
            text_color = (255, 255, 255)
        elif "gk" in team_str:
            badge_color = (0, 140, 255)   # Orange
            text_color = (0, 0, 0)
        elif team_str == "referee":
            badge_color = (0, 220, 220)   # Yellow
            text_color = (0, 0, 0)
        else:
            badge_color = (120, 120, 120)
            text_color = (255, 255, 255)

        # Draw Label Card
        cv2.rectangle(row_img, (5, 5), (self.label_card_w - 5, row_h - 5), badge_color, -1)
        cv2.rectangle(row_img, (5, 5), (self.label_card_w - 5, row_h - 5), (80, 80, 80), 1)

        cv2.putText(
            row_img, f"Track #{track_id}", (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2
        )
        cv2.putText(
            row_img, f"Team: {team_str.upper()}", (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1
        )
        cv2.putText(
            row_img, f"Pos: ({rel_pos[0]:.2f}, {rel_pos[1]:.2f})", (12, 82),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1
        )

        # Draw 3 crops (Early, Peak, Late)
        start_x = self.label_card_w + 10
        for i, crop_obj in enumerate(crops[:3]):
            cx = start_x + i * (self.crop_target_w + 10)
            cy = 10

            # Resize crop
            resized = cv2.resize(
                crop_obj.crop_img,
                (self.crop_target_w, self.crop_target_h),
                interpolation=cv2.INTER_AREA
            )
            row_img[cy:cy + self.crop_target_h, cx:cx + self.crop_target_w] = resized

            # Border around crop
            cv2.rectangle(
                row_img, (cx, cy),
                (cx + self.crop_target_w, cy + self.crop_target_h),
                (180, 180, 180), 1
            )

            # Badge text at bottom
            badge_txt = f"{crop_obj.label.upper()} {int(crop_obj.position_pct*100)}%"
            cv2.putText(
                row_img, badge_txt, (cx + 2, cy + self.crop_target_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1
            )

        return row_img

    def render_context_frame(
        self,
        keyframe: np.ndarray,
        active_tracklets: List[Tuple[int, List[float], str]],  # (track_id, bbox, team)
        max_dim: int = 1280,
    ) -> np.ndarray:
        """
        Draw numbered bounding boxes with team color borders on the mid-shot context frame.
        """
        ctx = keyframe.copy()
        h, w = ctx.shape[:2]

        for track_id, bbox, team in active_tracklets:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if team == "home":
                color = (255, 255, 255)
            elif team == "away":
                color = (60, 60, 220)
            elif "gk" in team:
                color = (0, 165, 255)
            elif team == "referee":
                color = (0, 240, 240)
            else:
                color = (0, 255, 0)

            # Bounding box
            cv2.rectangle(ctx, (x1, y1), (x2, y2), color, 2)

            # Label banner
            lbl = f"#{track_id}"
            t_size = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(ctx, (x1, max(0, y1 - 20)), (x1 + t_size[0] + 6, y1), color, -1)
            cv2.putText(
                ctx, lbl, (x1 + 3, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
            )

        # Scale down if larger than max_dim
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            ctx = cv2.resize(ctx, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        return ctx

    def render_radar_minimap(
        self,
        tracklets_with_teams: List[Tuple[int, str, float, float]],  # (track_id, team, rel_x, rel_y)
        w: int = 640,
        h: int = 400,
    ) -> np.ndarray:
        """
        Render a 2D radar diagram showing relative locations of each tracked player on the pitch.
        """
        radar = np.full((h, w, 3), (35, 95, 35), dtype=np.uint8)  # Grass green

        # Pitch boundary & lines
        cv2.rectangle(radar, (30, 25), (w - 30, h - 25), (240, 240, 240), 2)
        cv2.line(radar, (w // 2, 25), (w // 2, h - 25), (240, 240, 240), 2)
        cv2.circle(radar, (w // 2, h // 2), 45, (240, 240, 240), 2)

        # Penalty boxes
        box_w = 80
        box_h = 160
        cv2.rectangle(radar, (30, h // 2 - box_h // 2), (30 + box_w, h // 2 + box_h // 2), (240, 240, 240), 2)
        cv2.rectangle(radar, (w - 30 - box_w, h // 2 - box_h // 2), (w - 30, h // 2 + box_h // 2), (240, 240, 240), 2)

        # Plot players
        field_w = w - 60
        field_h = h - 50

        for track_id, team, rel_x, rel_y in tracklets_with_teams:
            px = int(30 + np.clip(rel_x, 0.0, 1.0) * field_w)
            py = int(25 + np.clip(rel_y, 0.0, 1.0) * field_h)

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
                color = (180, 180, 180)
                text_col = (0, 0, 0)

            cv2.circle(radar, (px, py), 12, color, -1)
            cv2.circle(radar, (px, py), 12, (20, 20, 20), 1)

            cv2.putText(
                radar, str(track_id), (px - 6, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_col, 1
            )

        return radar

    def build_batch_payload(
        self,
        shot_id: int,
        lifecycles: Dict[int, TrackletLifecycle],
        team_assignments: Dict[int, TeamClusterResult],
        frame_cache: Dict[int, np.ndarray],
        active_roster: Dict[str, Any],
        fps: float = 30.0,
    ) -> Optional[CollageBatchPayload]:
        """
        Build the consolidated Pass-2 batch payload for the entire camera shot.
        """
        # Filter for valid tracklets alive for >= min_tracklet_duration_sec
        valid_lifecycles = [
            lc for lc in lifecycles.values()
            if lc.duration_seconds(fps) >= self.min_tracklet_duration_sec
        ]

        if not valid_lifecycles:
            return None

        # Sort tracklets by team and left-to-right position
        valid_lifecycles.sort(key=lambda lc: (
            team_assignments.get(lc.track_id, TeamClusterResult(lc.track_id, "unknown", 0.5, False)).team,
            lc.get_average_center()[0]
        ))

        # Extract crops and render rows
        rows: List[np.ndarray] = []
        tracklet_metadata: List[Dict[str, Any]] = []

        home_cnt, away_cnt, gk_cnt, ref_cnt = 0, 0, 0, 0

        # Calculate relative positions
        all_centers = [lc.get_average_center() for lc in valid_lifecycles]
        min_x = min(c[0] for c in all_centers)
        max_x = max(c[0] for c in all_centers)
        min_y = min(c[1] for c in all_centers)
        max_y = max(c[1] for c in all_centers)
        span_x = max(1.0, max_x - min_x)
        span_y = max(1.0, max_y - min_y)

        radar_records = []

        for lc in valid_lifecycles:
            tid = lc.track_id
            t_info = team_assignments.get(tid)
            team_str = t_info.team if t_info else "unknown"

            if team_str == "home":
                home_cnt += 1
            elif team_str == "away":
                away_cnt += 1
            elif "gk" in team_str:
                gk_cnt += 1
            elif team_str == "referee":
                ref_cnt += 1

            crops = lc.extract_3_angle_crops(frame_cache)
            if not crops:
                continue

            cx, cy = lc.get_average_center()
            rel_x = (cx - min_x) / span_x
            rel_y = (cy - min_y) / span_y
            radar_records.append((tid, team_str, rel_x, rel_y))

            row_img = self.render_tracklet_row(tid, t_info, crops, (rel_x, rel_y))
            rows.append(row_img)

            tracklet_metadata.append({
                "track_id": tid,
                "team": team_str,
                "relative_x": round(rel_x, 3),
                "relative_y": round(rel_y, 3),
                "duration_frames": lc.total_frames,
                "duration_sec": round(lc.duration_seconds(fps), 2),
            })

        if not rows:
            return None

        # Stack rows into single montage
        montage = np.vstack(rows)

        # Context frame: pick the median frame across all cached frames
        cached_keys = sorted(frame_cache.keys())
        mid_frame_idx = cached_keys[len(cached_keys) // 2]
        keyframe = frame_cache[mid_frame_idx]

        # Gather active boxes at median frame
        active_boxes_at_keyframe = []
        for lc in valid_lifecycles:
            if mid_frame_idx in lc.frame_indices:
                b_idx = lc.frame_indices.index(mid_frame_idx)
                bbox = lc.bboxes[b_idx]
                t_str = team_assignments.get(lc.track_id, TeamClusterResult(lc.track_id, "unknown", 0.5, False)).team
                active_boxes_at_keyframe.append((lc.track_id, bbox, t_str))

        context_frame = self.render_context_frame(keyframe, active_boxes_at_keyframe)
        radar_minimap = self.render_radar_minimap(radar_records)

        return CollageBatchPayload(
            shot_id=shot_id,
            total_tracklets=len(valid_lifecycles),
            home_visible=home_cnt,
            away_visible=away_cnt,
            gk_visible=gk_cnt,
            ref_visible=ref_cnt,
            montage_img=montage,
            context_frame_img=context_frame,
            radar_minimap_img=radar_minimap,
            tracklet_metadata=tracklet_metadata,
            active_roster=active_roster,
        )

    def resolve_identities_via_vlm(
        self,
        payload: CollageBatchPayload,
        output_dir: str,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Send the combined Context Frame, Radar Minimap, and 3-Angle Montage to Gemini Flash
        in ONE single API call to map all Track_IDs to confirmed player names.
        Uses direct standard-library urllib REST calls with GEMINI_API_KEY.
        """
        import base64
        import re
        import urllib.request

        os.makedirs(output_dir, exist_ok=True)

        montage_path = os.path.join(output_dir, f"shot_{payload.shot_id:02d}_montage.png")
        context_path = os.path.join(output_dir, f"shot_{payload.shot_id:02d}_context.png")
        radar_path = os.path.join(output_dir, f"shot_{payload.shot_id:02d}_radar.png")

        cv2.imwrite(montage_path, payload.montage_img)
        cv2.imwrite(context_path, payload.context_frame_img)
        cv2.imwrite(radar_path, payload.radar_minimap_img)

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return self._fallback_topological_resolution(payload)

        try:
            home_team = payload.active_roster.get("home_team", "Real Madrid")
            away_team = payload.active_roster.get("away_team", "Barcelona")
            home_players = payload.active_roster.get("home_active_xi", [])
            away_players = payload.active_roster.get("away_active_xi", [])

            prompt_text = f"""
You are an expert football broadcast tactical analyst performing player identification for El Clásico ({home_team} vs {away_team}).

We provide 3 images from ONE continuous camera shot:
1. Context Frame (Image 1): Broadcast camera view with numbered bounding boxes (Track #ID) showing exact pitch location.
2. 2D Radar Minimap (Image 2): Normalized pitch positions showing relative spatial formation.
3. Multi-Angle Crop Montage (Image 3): 3 temporal angle crops (Early, Peak Sharpness, Late) for every Track #ID.

CRITICAL TACTICAL CONSTRAINTS:
- PARTIAL VISIBILITY: Only {payload.home_visible} of 11 {home_team} players and {payload.away_visible} of 11 {away_team} players are visible in this camera shot! Do NOT force-fit all 11 roster players.
- Team Kits: {home_team} is in pure WHITE kits. {away_team} is in BLAUGRANA (Blue/Red) kits.
- Goalkeepers wear distinctive non-outfield kits (Orange/Green/Yellow).
- Jersey Numbers: Inspect the 3-crop strips carefully for visible kit numbers on backs, chests, or shorts.
- Roster Lineups:
  {home_team} Active Players: {json.dumps(home_players)}
  {away_team} Active Players: {json.dumps(away_players)}

Return a strict JSON array of objects mapping each Track #ID (integer) to:
[
  {{
    "track_id": integer,
    "player_name": string (exact name from the active roster),
    "jersey_number": integer,
    "team": "home" | "away" | "gk_home" | "gk_away" | "referee",
    "confidence": float between 0.50 and 0.99,
    "reasoning": string (e.g. "Visible #7 on back of white jersey", "Orange kit in penalty area -> Lunin", "Left flank relative position")
  }}
]

Output valid JSON ONLY.
"""

            # Encode 3 images to base64
            parts = []
            for img_p in [context_path, radar_path, montage_path]:
                with open(img_p, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("utf-8")
                parts.append({"inline_data": {"mime_type": "image/png", "data": b64_data}})

            parts.append({"text": prompt_text})

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

            for model_name in models_to_try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=18) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        candidates = data.get("candidates", [])
                        if candidates:
                            raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                            if raw_text:
                                break
                except Exception as ex_m:
                    print(f"[SmartCollageBatcher] Model {model_name} failed: {ex_m}. Trying next model...")
                    continue

            if not raw_text:
                print("[SmartCollageBatcher] All Gemini models failed. Falling back to topological resolution.")
                return self._fallback_topological_resolution(payload)

            # Strip markdown formatting
            clean_text = raw_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:].strip()
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:].strip()
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3].strip()

            parsed = json.loads(clean_text)

            assignments: Dict[int, Dict[str, Any]] = {}
            if isinstance(parsed, list):
                for item in parsed:
                    tid = int(item.get("track_id", 0))
                    if tid > 0:
                        assignments[tid] = item
            elif isinstance(parsed, dict):
                if "tracks" in parsed and isinstance(parsed["tracks"], list):
                    for item in parsed["tracks"]:
                        tid = int(item.get("track_id", 0))
                        if tid > 0:
                            assignments[tid] = item
                else:
                    for k, v in parsed.items():
                        if str(k).isdigit() and isinstance(v, dict):
                            v["track_id"] = int(k)
                            assignments[int(k)] = v

            # Save diagnostics
            diag_path = os.path.join(output_dir, f"shot_{payload.shot_id:02d}_vlm_identities.json")
            with open(diag_path, "w", encoding="utf-8") as f:
                json.dump(assignments, f, indent=2)

            print(f"[SmartCollageBatcher] Successfully resolved {len(assignments)} tracklet identities via Gemini Flash!")
            return assignments

        except Exception as e:
            print(f"[SmartCollageBatcher] VLM call failed ({e}). Falling back to topological resolution.")
            return self._fallback_topological_resolution(payload)

    def _fallback_topological_resolution(
        self,
        payload: CollageBatchPayload,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Offline deterministic topological resolution when VLM is disabled.
        """
        assignments: Dict[int, Dict[str, Any]] = {}

        home_players = payload.active_roster.get("home_active_xi", [])
        away_players = payload.active_roster.get("away_active_xi", [])

        # Assign GK
        for meta in payload.tracklet_metadata:
            tid = meta["track_id"]
            team = meta["team"]

            if "gk" in team:
                gk_name = "Andriy Lunin" if "home" in team else "Iñaki Peña"
                gk_num = 13 if "home" in team else 13
                assignments[tid] = {
                    "track_id": tid,
                    "player_name": gk_name,
                    "jersey_number": gk_num,
                    "team": team,
                    "confidence": 0.95,
                    "reasoning": "Goalkeeper kit and goal area spatial lock",
                }
            elif team == "referee":
                assignments[tid] = {
                    "track_id": tid,
                    "player_name": "Match Referee",
                    "jersey_number": 0,
                    "team": "referee",
                    "confidence": 0.90,
                    "reasoning": "Referee kit and central pitch position",
                }

        return assignments
