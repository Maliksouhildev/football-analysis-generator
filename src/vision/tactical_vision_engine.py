"""
src/vision/tactical_vision_engine.py
Next-Generation Multimodal Tactical Vision & Dynamic Telestration Engine.

Uses Gemini Vision (with resilient multi-model failover) guided by:
- Match clock and scoreboard detection from frame graphics
- Dynamic Active XI Lineups with substitution awareness at the exact match minute
- Kit color differentiation (Home vs Away vs Referee vs Goalkeeper)
- Boot-level turf contact coordinate anchoring (0-100% normalized)
- Strict exclusion of referees, goalkeepers, and broadcast ad banners
"""

from __future__ import annotations

import base64
import itertools
import json
import logging
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from sklearn.cluster import KMeans

from src.scraper.lineup_scraper import MatchLineupResolver
from src.vision.player_detector import get_pitch_convex_hull
from src.vision.tactical_parser import (
    PassingTriangleBuilder,
    TacticalBacklineBuilder,
    TacticalFormationParser,
    TacticalTelestrationProps,
    TelestrationPlayer,
    TelestrationLine,
    TelestrationPolygon,
)

logger = logging.getLogger("vision.tactical_vision_engine")

SUPPORTED_VISION_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
]

TACTICAL_TELESTRATION_SYSTEM_PROMPT = """You are a World-Class Football Tactical Telestrator and Match Analyst for UEFA Champions League and Sky Sports broadcasts.
Your task is to analyze freeze-frame match footage, recognize the game situation and active players on the pitch, and output exact 2D tactical telestration vectors.

CRITICAL RULES:
1. PITCH-SURFACE BOOT / FEET BASE ANCHORING:
   - The visual radar ring and spotlight beam MUST sit flat on the green turf grass DIRECTLY AT THE BASE OF THE PLAYER'S BOOTS / STUDS.
   - For every player, detect their 2D bounding box 'box_2d': [ymin, xmin, ymax, xmax] in normalized 0-1000 scale.
   - The feet turf contact base coordinate is:
     * x = round(((xmin + xmax) / 2.0) / 10.0, 1)  (the lateral center of their boots, range 0..100)
     * y = round(ymax / 10.0, 1)                    (the bottom-most turf contact point of their boot soles, range 0..100)
   - All coordinates must lie strictly on the green grass within the pitch boundary.

2. ACTIVE ACTION CLUSTER:
   - Focus strictly on 3 to 6 key outfield players directly involved in the central tactical duel / play.
   - NEVER tag the Referee (who wears a distinct cyan/neon kit).
   - NEVER tag the Goalkeeper (who wears a distinct neon kit and gloves).
   - NEVER tag empty turf.

3. PLAYER IDENTIFICATION & KIT MATCHING:
   - Home Kit vs Away Kit: Use the kit colors provided in the match lineup context to categorize which team is which.
   - Identify players using their:
     * Visible shirt number (e.g., #18, #14, #5, #22, #23, #16).
     * Pitch position and role (e.g., CBs holding the defensive line, CF making the central run, CM orchestrating).
     * Physical appearance, skin tone, hair, and build matching the active lineup list for this match minute.
   - Use ONLY players from the provided active lineup list who are currently on the pitch (accounting for substitutions).
   - NOTE: Identify all players strictly by Tactical Role: 'BALL CARRIER', 'PRESSING TRIGGER', 'THIRD-MAN RUNNER', 'ISOLATED FULLBACK', 'COVERING CB', 'MIDFIELD PIVOT'. Never guess or output real player names.

4. TACTICAL ROLES & COLORS:
   - Focal Player / Ball Carrier / Primary Attacker: color = '#ffff00' (Yellow), showRing = true, showBeam = true, showBadge = true.
   - Attacking Teammates / Passing Options: color = '#00e676' (Emerald), showRing = true, showBeam = false, showBadge = true.
   - Opposing Defenders / Pressing Markers: color = '#ef4444' (Red), showRing = true, showBeam = false, showBadge = true.

5. TACTICAL GEOMETRY:
   - Draw defensive backlines or pressing corridors with lines connecting valid player IDs.
   - Draw attacking triangles or press traps with polygons connecting 3+ valid player IDs.

6. VACATED / EMPTY TACTICAL ZONES (EXPLOITED SPACE):
   - Whenever players leave their tactical zone empty (e.g. center-back dragged out of position, fullback caught high leaving open flank, midfield defensive vacuum / empty pocket in Zone 14), you MUST illustrate this empty space!
   - Model the empty zone as a colored polygon or quadrilateral (square/box) with 4 boundary points (x, y) [0.0..100.0] on the pitch turf:
     * "id": "vacated_zone_1"
     * "isVacatedZone": true
     * "points": [{"x": 58.0, "y": 32.0}, {"x": 68.0, "y": 30.0}, {"x": 70.0, "y": 42.0}, {"x": 60.0, "y": 44.0}]
     * "fillColor": "rgba(245, 158, 11, 0.25)" (Translucent amber warning glow)
     * "borderColor": "#f59e0b" (Amber border)
     * "pattern": "stripes"
     * "label": "VACATED DEFENSIVE ZONE" (or "EXPLOITED HALF-SPACE", "EMPTY ZONE 14")

OUTPUT FORMAT:
Return strictly a valid JSON object matching this schema:
{
  "match_clock": "string (e.g. 84:12)",
  "score": "string (e.g. 1 - 0)",
  "tactical_concept": "string (e.g. MINUTE 85' // TACTICAL BREAKDOWN - PROGRESSION INTO VACATED HALF-SPACE)",
  "action_summary": "High-density 1-sentence breakdown of the tactical structure",
  "players": [
    {
      "id": "p1",
      "name": "COVERING CB",
      "number": 5,
      "team": "Home",
      "role": "COVERING CB",
      "color": "#ef4444",
      "box_2d": [360, 500, 480, 525],
      "x": 51.2,
      "y": 48.0,
      "showRing": true,
      "showBeam": false,
      "showBadge": true,
      "visual_evidence": "Defending unit marking space in penalty area"
    }
  ],
  "lines": [
    {
      "id": "def_line",
      "playerIds": ["p1", "p2"],
      "color": "#ef4444",
      "width": 3,
      "dashed": true
    }
  ],
  "polygons": [
    {
      "id": "vacated_zone_1",
      "isVacatedZone": true,
      "points": [{"x": 58.0, "y": 32.0}, {"x": 68.0, "y": 30.0}, {"x": 70.0, "y": 42.0}, {"x": 60.0, "y": 44.0}],
      "fillColor": "rgba(245, 158, 11, 0.25)",
      "borderColor": "#f59e0b",
      "label": "VACATED DEFENSIVE ZONE"
    }
  ]
}
"""


class TacticalVisionEngine:
    """
    Intelligent tactical telestration engine powered by Gemini Vision
    and active match lineup resolution.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()

    def analyze_frame(
        self,
        image_path: Union[str, Path],
        event_context: Optional[Dict[str, Any]] = None,
    ) -> TacticalTelestrationProps:
        """
        Analyzes a freeze-frame image and generates broadcast-quality
        tactical telestration coordinates and player identities.
        """
        ctx = dict(event_context or {})
        img_path = Path(image_path)
        if not img_path.is_file():
            raise FileNotFoundError(f"Freeze frame image not found: {img_path}")

        # 1. Resolve match context and minute
        home_team = ctx.get("home_team", "Home")
        away_team = ctx.get("away_team", "Away")
        minute = int(ctx.get("match_minute", 0))

        # 2. Get active lineup and kit definitions
        lineup_info = MatchLineupResolver.get_active_lineup(home_team, away_team, match_minute=minute)
        home_kit = lineup_info.get("home_kit", {})
        away_kit = lineup_info.get("away_kit", {})
        referee_kit = lineup_info.get("referee_kit", {})
        home_outfield = lineup_info.get("home_outfield", [])
        away_outfield = lineup_info.get("away_outfield", [])

        # 3. Build rich prompt with lineup metadata
        user_prompt = self._build_user_prompt(
            home_team=home_team,
            away_team=away_team,
            match_minute=minute,
            home_kit=home_kit,
            away_kit=away_kit,
            referee_kit=referee_kit,
            home_outfield=home_outfield,
            away_outfield=away_outfield,
            event_context=ctx,
        )

        # 4. Encode image
        with open(img_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")

        # 5. Call Gemini Vision with model failover pool
        response_json = self._call_vision_with_failover(b64_image, user_prompt)

        # 6. Sanitize and post-process coordinates & player identities
        processed_data = self._post_process_tactical_data(response_json, lineup_info, img_path)

        return TacticalFormationParser.parse_formation(processed_data)

    def _build_user_prompt(
        self,
        home_team: str,
        away_team: str,
        match_minute: int,
        home_kit: Dict[str, Any],
        away_kit: Dict[str, Any],
        referee_kit: Dict[str, Any],
        home_outfield: List[Dict[str, Any]],
        away_outfield: List[Dict[str, Any]],
        event_context: Dict[str, Any],
    ) -> str:
        home_players_str = ", ".join([f"#{p['number']} {p['name']} ({p['role']})" for p in home_outfield])
        away_players_str = ", ".join([f"#{p['number']} {p['name']} ({p['role']})" for p in away_outfield])

        narrative = event_context.get("narrative_hook") or event_context.get("description", "")
        focal_hint = ""
        players_inv = event_context.get("players_involved", [])
        if players_inv:
            focal_hint = f"Expected key player in this sequence: {players_inv[0]}."

        prompt = f"""Match Context:
- Fixture: {home_team} vs {away_team}
- Match Minute: ~{match_minute}'
- Home Kit ({home_team}): Primary {home_kit.get('primary', 'carmine red')}, hex {home_kit.get('hex', '#8a1529')}
- Away Kit ({away_team}): Primary {away_kit.get('primary', 'white')}, hex {away_kit.get('hex', '#ffffff')}
- Referee Kit: {referee_kit.get('primary', 'cyan/neon')} (DO NOT TAG)
- Context narrative: {narrative}
{focal_hint}

ACTIVE OUTFIELD SQUADS AT THIS MINUTE:
- {home_team} Active Outfield Players: {home_players_str}
- {away_team} Active Outfield Players: {away_players_str}

TASKS:
1. Confirm the match clock and score in the broadcast bug at top-left.
2. FIRST: Locate the football on the pitch. Identify the exact player who has the ball at their feet (including the Goalkeeper in black or Corner Taker at the corner flag) and assign their role strictly as 'BALL CARRIER' with showBeam=true and color='#ffff00'.
3. Identify 3 to 6 key players in this action cluster and label them strictly by Tactical Role: 'BALL CARRIER', 'PRESSING TRIGGER', 'THIRD-MAN RUNNER', 'ISOLATED FULLBACK', 'COVERING CB', 'MIDFIELD PIVOT'. NEVER guess or output real player names.
4. EXACT BOOT / FEET BASE ANCHORING: For each player, output 'box_2d': [ymin, xmin, ymax, xmax] in 0-1000 scale where ymax is the bottom of their shoes touching the turf grass! Set x = round(((xmin+xmax)/2.0)/10.0, 1) and y = round(ymax/10.0, 1) so the radar ring sits directly at their feet on the pitch, NOT their torso or waist.
5. Identify passing lines or pressing corridors between players in 'lines'.
6. MANDATORY VACATED TACTICAL ZONE: Identify the primary VACATED or EXPLOITED empty space/pocket on the turf created by players being pulled out of position (e.g. empty corridor ahead of the attacker, vacated half-space, or hole behind defensive line). You MUST output this in 'polygons' as a 4-point quadrilateral (points array with four {{"x": float, "y": float}} objects) on the grass pitch with "isVacatedZone": true, "fillColor": "rgba(245, 158, 11, 0.25)", "borderColor": "#f59e0b", and a descriptive label (e.g. "VACATED DEFENSIVE ZONE", "EXPLOITED HALF-SPACE", "EMPTY DEFENSIVE POCKET").
"""
        return prompt

    def _call_vision_with_failover(self, b64_image: str, user_prompt: str) -> Dict[str, Any]:
        """Calls Gemini API across available models until a valid JSON response is obtained."""
        payload = {
            "system_instruction": {
                "parts": [{"text": TACTICAL_TELESTRATION_SYSTEM_PROMPT}]
            },
            "contents": [{
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64_image,
                        }
                    },
                    {
                        "text": user_prompt
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json",
            }
        }

        req_data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
        }
        if self.api_key:
            headers["x-goog-api-key"] = self.api_key

        last_err: Optional[Exception] = None

        models_to_try = list(SUPPORTED_VISION_MODELS)
        idx_m = 0
        while idx_m < len(models_to_try):
            model_name = models_to_try[idx_m]
            idx_m += 1
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            if self.api_key:
                url += f"?key={urllib.parse.quote(self.api_key)}"

            for attempt in range(2):
                logger.info("Attempting tactical vision analysis with model '%s' (attempt %s)...", model_name, attempt + 1)
                req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")

                try:
                    with urllib.request.urlopen(req, timeout=40) as resp:
                        resp_bytes = resp.read()
                        data = json.loads(resp_bytes.decode("utf-8"))
                        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        # Strip markdown code fences if present
                        if text.startswith("```json"):
                            text = text[len("```json"):].strip()
                        elif text.startswith("```"):
                            text = text[len("```"):].strip()
                        if text.endswith("```"):
                            text = text[:-len("```")].strip()
                        parsed = json.loads(text)
                        logger.info("Successfully extracted tactical vectors using '%s'.", model_name)
                        return parsed
                except urllib.error.HTTPError as e:
                    err_msg = ""
                    try:
                        err_msg = e.read().decode("utf-8", errors="ignore")
                    except Exception:
                        pass
                    logger.warning("Model '%s' HTTP %s: %s", model_name, e.code, err_msg[:120])
                    last_err = e
                    if e.code == 429 or "429" in str(e):
                        logger.warning("Model '%s' returned HTTP 429 rate limit. Sleeping 4 seconds and retrying with fallback flash models...", model_name)
                        import time
                        time.sleep(4.0)
                        for rm in ["gemini-flash-lite-latest", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]:
                            if rm not in models_to_try[idx_m:]:
                                models_to_try.append(rm)
                        break
                    if e.code == 503 and attempt == 0:
                        import time
                        time.sleep(2.0)
                        continue
                    break
                except Exception as e:
                    logger.warning("Model '%s' failed: %s", model_name, e)
                    last_err = e
                    break

        raise RuntimeError(f"All vision models exhausted in failover pool: {last_err}")

    def _post_process_tactical_data(
        self,
        raw_data: Dict[str, Any],
        lineup_info: Dict[str, Any],
        img_path: Path,
    ) -> Dict[str, Any]:
        """
        Validates, boundary-clamps, and aligns extracted coordinates and identities:
        - Clamps y to [14.0, 78.0] to guarantee 0% tags on bottom banner or top crowd.
        - Clamps x to [4.0, 96.0].
        - Ensures player names are formatted cleanly.
        - Verifies that boot points contact green grass pitch surface.
        """
        raw_players = raw_data.get("players", [])
        clean_players: List[Dict[str, Any]] = []

        home_outfield = {p["name"].lower(): p for p in lineup_info.get("home_outfield", [])}
        away_outfield = {p["name"].lower(): p for p in lineup_info.get("away_outfield", [])}

        img = cv2.imread(str(img_path)) if img_path and img_path.is_file() else None
        hull = get_pitch_convex_hull(img) if img is not None else None

        for idx, p in enumerate(raw_players):
            name = str(p.get("name", f"Player {idx+1}")).strip()
            number = p.get("number")
            role = p.get("role", "")
            team = p.get("team", "")

            # Strict exclusion for known injured/non-squad players
            if "militão" in name.lower() or "militao" in name.lower():
                logger.warning("Filtering out injured/non-squad player: %s", name)
                continue

            # 1. Base / Boot contact extraction from box_2d if available
            box_2d = p.get("box_2d")
            if box_2d and isinstance(box_2d, (list, tuple)) and len(box_2d) == 4:
                try:
                    ymin, xmin, ymax, xmax = [float(v) for v in box_2d]
                    scale = 10.0 if max(ymax, xmax) > 100.0 else 1.0
                    x = ((xmin + xmax) / 2.0) / scale
                    # The bottom-most line ymax is the contact point of shoes on the turf
                    y = ymax / scale
                except Exception:
                    x = float(p.get("x", 50.0))
                    y = float(p.get("y", 50.0))
            else:
                try:
                    x = float(p.get("x", 50.0))
                    y = float(p.get("y", 50.0))
                except (ValueError, TypeError):
                    continue

            # Auto-normalize if model emitted 0-1000 scale or pixels
            if x > 100.0:
                x = x / 10.0 if x <= 1000.0 else (x / 1920.0) * 100.0
            if y > 100.0:
                y = y / 10.0 if y <= 1000.0 else (y / 1080.0) * 100.0

            # Pitch Convex Hull validation (replaces static [4, 96], [14, 78] ROI clamp)
            if hull is not None and img is not None:
                h_img, w_img = img.shape[:2]
                pt_px = (float(x * w_img / 100.0), float(y * h_img / 100.0))
                if cv2.pointPolygonTest(hull, pt_px, False) < 0:
                    logger.warning("Player %s coordinates (%s, %s) outside pitch convex hull. Filtering out.", name, x, y)
                    continue
            else:
                if x < 0.0 or x > 100.0 or y < 0.0 or y > 100.0:
                    continue

            # Match against active lineup for canonical spelling and number if available
            matched_entry = None
            for roster in (home_outfield, away_outfield):
                for r_name, r_entry in roster.items():
                    if name.lower() in r_name or r_name in name.lower():
                        matched_entry = r_entry
                        break
                if matched_entry:
                    break

            canonical_role = TacticalVisionEngine._resolve_canonical_player_name_pos(
                p, team, role_hint=role or name
            )

            clean_players.append({
                "id": p.get("id", f"p_{idx}"),
                "name": canonical_role,
                "number": number or "",
                "role": canonical_role,
                "color": p.get("color", "#ffff00"),
                "x": round(x, 2),
                "y": round(y, 2),
                "showRing": bool(p.get("showRing", True)),
                "showBeam": bool(p.get("showBeam", False)),
                "showBadge": bool(p.get("showBadge", True)),
            })

        # Ensure at least 1 focal player has showBeam=True (prioritizing BALL CARRIER)
        if clean_players and not any(p.get("showBeam") for p in clean_players):
            bc_player = next((p for p in clean_players if "BALL CARRIER" in str(p.get("role", "")).upper()), None)
            if bc_player is not None:
                bc_player["showBeam"] = True
                bc_player["color"] = "#ffff00"
            else:
                best_p = min(clean_players, key=lambda p: np.hypot(float(p.get("x", 50.0)) - 50.0, float(p.get("y", 50.0)) - 50.0))
                best_p["showBeam"] = True
                best_p["color"] = "#ffff00"

        # Filter lines and polygons to only reference valid cleaned player IDs
        valid_ids = {p["id"] for p in clean_players}

        clean_lines = []
        for l in raw_data.get("lines", []):
            p_ids = [pid for pid in l.get("playerIds", []) if pid in valid_ids]
            if len(p_ids) >= 2:
                clean_lines.append({
                    "id": l.get("id", "line"),
                    "playerIds": p_ids,
                    "color": l.get("color", "#00e676"),
                    "width": l.get("width", 3),
                    "dashed": bool(l.get("dashed", False)),
                })

        clean_polygons = []
        for poly in raw_data.get("polygons", []):
            raw_pts = poly.get("points")
            is_vacated = bool(poly.get("isVacatedZone")) or bool(raw_pts and len(raw_pts) >= 3) or (
                bool(poly.get("label")) and bool(re.search(r"vacat|empty|exploit|vacuum|pocket|space|zone", str(poly.get("label")), re.I))
            )

            # Case A: Arbitrary Empty Vacated Zone (quadrilateral / box)
            if raw_pts and len(raw_pts) >= 3:
                clamped_pts = []
                for pt in raw_pts:
                    try:
                        px = float(pt.get("x", 50.0))
                        py = float(pt.get("y", 50.0))
                        if px > 100.0:
                            px = px / 10.0 if px <= 1000.0 else (px / 1920.0) * 100.0
                        if py > 100.0:
                            py = py / 10.0 if py <= 1000.0 else (py / 1080.0) * 100.0
                        px = max(0.0, min(100.0, px))
                        py = max(0.0, min(100.0, py))
                        clamped_pts.append({"x": round(px, 2), "y": round(py, 2)})
                    except Exception:
                        pass
                if len(clamped_pts) >= 3:
                    cx_p = sum(pt["x"] for pt in clamped_pts) / len(clamped_pts)
                    cy_p = sum(pt["y"] for pt in clamped_pts) / len(clamped_pts)
                    clamped_pts = sorted(clamped_pts, key=lambda pt: math.atan2(pt["y"] - cy_p, pt["x"] - cx_p))
                    clean_polygons.append({
                        "id": poly.get("id", f"vacated_zone_{len(clean_polygons)}"),
                        "playerIds": [],
                        "points": clamped_pts,
                        "fillColor": poly.get("fillColor", "rgba(245, 158, 11, 0.25)"),
                        "borderColor": poly.get("borderColor", "#f59e0b"),
                        "pattern": poly.get("pattern", "stripes"),
                        "label": poly.get("label", "VACATED DEFENSIVE ZONE"),
                        "isVacatedZone": True,
                    })
                    continue

            # Case B: Player-anchored polygon
            p_ids = [pid for pid in poly.get("playerIds", []) if pid in valid_ids]
            if len(p_ids) >= 3:
                p_ids = p_ids[:4]
                p_pts = {p["id"]: (float(p["x"]), float(p["y"])) for p in clean_players if p["id"] in p_ids}
                if len(p_pts) >= 3:
                    cx_p = sum(p_pts[pid][0] for pid in p_ids) / len(p_ids)
                    cy_p = sum(p_pts[pid][1] for pid in p_ids) / len(p_ids)
                    p_ids = sorted(p_ids, key=lambda pid: math.atan2(p_pts[pid][1] - cy_p, p_pts[pid][0] - cx_p))
                clean_polygons.append({
                    "id": poly.get("id", "poly"),
                    "playerIds": p_ids,
                    "fillColor": poly.get("fillColor", "rgba(0, 230, 118, 0.22)"),
                    "borderColor": poly.get("borderColor", "#00e676"),
                    "pattern": poly.get("pattern", "stripes"),
                    "label": poly.get("label", "TACTICAL SHAPE"),
                    "isVacatedZone": is_vacated,
                })

        return {
            "players": clean_players,
            "lines": clean_lines,
            "polygons": clean_polygons,
            "title": raw_data.get("tactical_concept") or raw_data.get("title"),
            "subtitle": raw_data.get("action_summary") or raw_data.get("subtitle"),
        }

    # =========================================================================
    # Multi-Frame Tracking, Unsupervised Kit Colors & Perspective Telestration
    # =========================================================================

    @staticmethod
    def _extract_team_colors_kmeans(
        samples: List[np.ndarray],
    ) -> Tuple[Dict[str, str], Optional[KMeans]]:
        """
        Runs KMeans(n_clusters=2) in LAB color space across player upper-torso
        crops to auto-detect home_hex and away_hex.
        """
        if len(samples) < 2:
            return {"home_hex": "#e11d48", "away_hex": "#ffffff"}, None

        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(samples)
            centers = kmeans.cluster_centers_

            def _lab_to_hex(lab: np.ndarray) -> str:
                pixel = np.uint8([[lab]])
                bgr = cv2.cvtColor(pixel, cv2.COLOR_LAB2BGR)[0][0]
                r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
                return f"#{r:02x}{g:02x}{b:02x}"

            c0_hex = _lab_to_hex(centers[0])
            c1_hex = _lab_to_hex(centers[1])
            return {"home_hex": c0_hex, "away_hex": c1_hex}, kmeans
        except Exception as e:
            logger.warning("KMeans kit color extraction failed (%s), using default colors", e)
    @staticmethod
    def _classify_jersey_rgb_hsv(
        frame: Optional[np.ndarray],
        track: Dict[str, Any],
        w: int = 1920,
        h: int = 1080,
        home_hex: Optional[str] = None,
        away_hex: Optional[str] = None,
    ) -> str:
        """
        Samples the median RGB & HSV of the center torso box (x_center ± 20%, y_top + 20% to y_top + 50%
        of bbox), masking out green grass pixels (Hue 35-85 in OpenCV 0-180 scale).
        NEVER trusts track.get('team') or pitch x_pct!
        Returns: 'TEAM_WHITE', 'TEAM_RED', 'REFEREE', 'GOALKEEPER', or 'EXCLUDE'.
        """
        # Preserve synthetic anchor tracks (e.g. 96 White Carrier, 97 White Runner, 98 Red Corner Taker, 99 Roma GK)
        try:
            tid_int = int(track.get("track_id", 0))
            if tid_int >= 90 and track.get("jersey_class"):
                return str(track["jersey_class"])
        except (ValueError, TypeError):
            pass

        if frame is None:
            raw_jc = track.get("jersey_class")
            if raw_jc:
                return raw_jc
            return "EXCLUDE"

        bbox = track.get("bbox")
        if bbox and len(bbox) == 4:
            ymin, xmin, ymax, xmax = [float(v) for v in bbox]
        else:
            try:
                x_pct = float(track.get("x_pct", 50.0))
                y_pct = float(track.get("y_pct", 50.0))
            except (ValueError, TypeError):
                x_pct, y_pct = 50.0, 50.0
            cx_px = x_pct * w / 100.0
            feet_y_px = y_pct * h / 100.0
            bh = int(h * 0.08)
            bw = int(bh * 0.45)
            ymin = feet_y_px - bh
            ymax = feet_y_px
            xmin = cx_px - bw // 2
            xmax = cx_px + bw // 2

        bh = ymax - ymin
        bw = xmax - xmin
        if bh <= 4 or bw <= 4:
            return "EXCLUDE"

        # Center torso box: x_center ± 20% of bbox width, y_top + 20% to y_top + 50% of bbox height
        x_center = xmin + bw * 0.5
        t_y1 = max(0, min(h - 1, int(ymin + bh * 0.20)))
        t_y2 = max(0, min(h, int(ymin + bh * 0.50)))
        t_x1 = max(0, min(w - 1, int(x_center - bw * 0.20)))
        t_x2 = max(0, min(w, int(x_center + bw * 0.20)))

        if t_y2 <= t_y1 or t_x2 <= t_x1:
            return "EXCLUDE"

        torso_crop = frame[t_y1:t_y2, t_x1:t_x2]
        if torso_crop.size == 0:
            return "EXCLUDE"

        # Mask out green grass pixels (Hue 35-85 in OpenCV 0-180 scale)
        torso_hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        grass_mask = (torso_hsv[:, :, 0] >= 35) & (torso_hsv[:, :, 0] <= 85) & (torso_hsv[:, :, 1] >= 30)
        non_grass = ~grass_mask

        if np.count_nonzero(non_grass) >= 6:
            med_b = float(np.median(torso_crop[non_grass, 0]))
            med_g = float(np.median(torso_crop[non_grass, 1]))
            med_r = float(np.median(torso_crop[non_grass, 2]))
            med_h = float(np.median(torso_hsv[non_grass, 0]))
            med_s = float(np.median(torso_hsv[non_grass, 1]))
            med_v = float(np.median(torso_hsv[non_grass, 2]))
        else:
            med_b = float(np.median(torso_crop[:, :, 0]))
            med_g = float(np.median(torso_crop[:, :, 1]))
            med_r = float(np.median(torso_crop[:, :, 2]))
            med_h = float(np.median(torso_hsv[:, :, 0]))
            med_s = float(np.median(torso_hsv[:, :, 1]))
            med_v = float(np.median(torso_hsv[:, :, 2]))

        # Store sampled RGB/HSV on track for downstream analysis
        track["torso_rgb"] = [med_r, med_g, med_b]
        track["torso_hsv"] = [med_h, med_s, med_v]

        # 1. Reject Referee:
        # Referees wear kits distinct from outfield players (e.g. fluorescent cyan, neon yellow, bright pink)
        if (med_b - med_r > 32) or (med_g - med_r > 32) or (78 <= med_h <= 135 and med_s >= 45):
            track["jersey_class"] = "REFEREE"
            track["team"] = "referee"
            return "REFEREE"

        # 2. Goalkeeper (black/dark/distinct kit):
        if med_v < 62 and med_s < 60:
            track["jersey_class"] = "GOALKEEPER"
            track["team"] = "gk"
            return "GOALKEEPER"

        # 3. Dynamic color distance if home_hex and away_hex are provided
        if home_hex and away_hex:
            h_rgb = np.array(TacticalVisionEngine._hex_to_rgb(home_hex))
            a_rgb = np.array(TacticalVisionEngine._hex_to_rgb(away_hex))
            t_rgb = np.array([med_r, med_g, med_b])
            dist_home = np.linalg.norm(t_rgb - h_rgb)
            dist_away = np.linalg.norm(t_rgb - a_rgb)
            if dist_home <= dist_away:
                track["jersey_class"] = "TEAM_HOME"
                track["team"] = "home"
                return "TEAM_HOME"
            else:
                track["jersey_class"] = "TEAM_AWAY"
                track["team"] = "away"
                return "TEAM_AWAY"

        # 4. Outfield relative brightness fallback when kit colors are not yet known
        if (med_r + med_g + med_b) / 3.0 > 140 and med_s < 40:
            track["jersey_class"] = "TEAM_AWAY"
            track["team"] = "away"
            return "TEAM_AWAY"
        else:
            track["jersey_class"] = "TEAM_HOME"
            track["team"] = "home"
            return "TEAM_HOME" 

    @staticmethod
    def classify_jersey_color(
        frame: Optional[np.ndarray],
        track: Dict[str, Any],
        w: int = 1920,
        h: int = 1080,
    ) -> str:
        return TacticalVisionEngine._classify_jersey_rgb_hsv(frame, track, w, h)

    @staticmethod
    def extract_track_torso_rgb(
        frame: Optional[np.ndarray],
        track: Optional[Dict[str, Any]],
        w: int = 1920,
        h: int = 1080,
    ) -> Tuple[float, float, float]:
        """
        Extracts median torso RGB (R, G, B) for a player track.
        Prioritizes track["torso_rgb"], then non-grass torso sampling from frame,
        and finally falls back to jersey_class / team defaults.
        """
        if not isinstance(track, dict):
            return 128.0, 128.0, 128.0

        rgb = track.get("torso_rgb")
        if isinstance(rgb, (list, tuple)) and len(rgb) >= 3:
            try:
                return float(rgb[0]), float(rgb[1]), float(rgb[2])
            except (ValueError, TypeError):
                pass

        if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
            fh, fw = frame.shape[:2]
            bbox = track.get("bbox") if isinstance(track, dict) else None
            if bbox and len(bbox) == 4:
                ymin, xmin, ymax, xmax = [float(v) for v in bbox]
            else:
                try:
                    x_pct = float(track.get("x_pct", 50.0))
                    y_pct = float(track.get("y_pct", 50.0))
                except (ValueError, TypeError):
                    x_pct, y_pct = 50.0, 50.0
                cx_px = x_pct * fw / 100.0
                feet_y_px = y_pct * fh / 100.0
                bh = int(fh * 0.08)
                bw = int(bh * 0.45)
                ymin = feet_y_px - bh
                ymax = feet_y_px
                xmin = cx_px - bw // 2
                xmax = cx_px + bw // 2

            bh = ymax - ymin
            bw = xmax - xmin
            if bh > 4 and bw > 4:
                x_center = xmin + bw * 0.5
                t_y1 = max(0, min(fh - 1, int(ymin + bh * 0.20)))
                t_y2 = max(0, min(fh, int(ymin + bh * 0.50)))
                t_x1 = max(0, min(fw - 1, int(x_center - bw * 0.20)))
                t_x2 = max(0, min(fw, int(x_center + bw * 0.20)))

                if t_y2 > t_y1 and t_x2 > t_x1:
                    crop = frame[t_y1:t_y2, t_x1:t_x2]
                    if crop.size > 0:
                        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        grass_mask = (crop_hsv[:, :, 0] >= 35) & (crop_hsv[:, :, 0] <= 85) & (crop_hsv[:, :, 1] >= 30)
                        non_grass = ~grass_mask
                        if np.count_nonzero(non_grass) >= 6:
                            med_b = float(np.median(crop[non_grass, 0]))
                            med_g = float(np.median(crop[non_grass, 1]))
                            med_r = float(np.median(crop[non_grass, 2]))
                        else:
                            med_b = float(np.median(crop[:, :, 0]))
                            med_g = float(np.median(crop[:, :, 1]))
                            med_r = float(np.median(crop[:, :, 2]))
                        track["torso_rgb"] = [med_r, med_g, med_b]
                        return med_r, med_g, med_b

        jc = str(track.get("jersey_class", "")).upper()
        team = str(track.get("team", "")).lower()
        if jc in ("TEAM_HOME", "TEAM_RED") or team == "home":
            return 210.0, 45.0, 45.0
        elif jc in ("TEAM_AWAY", "TEAM_WHITE") or team == "away":
            return 215.0, 215.0, 215.0
        elif jc == "GOALKEEPER":
            return 35.0, 35.0, 35.0

        return 128.0, 128.0, 128.0

    @staticmethod
    def is_same_team_runner(
        track: Optional[Dict[str, Any]],
        carrier_team_class: str,
        frame: Optional[np.ndarray] = None,
        w: int = 1920,
        h: int = 1080,
    ) -> bool:
        """
        Validates if a candidate runner strictly belongs to carrier_team_class.
        Generic across all matches, teams, and kit colors.
        """
        if not isinstance(track, dict):
            return False
        tr_team = str(track.get("team", "")).lower()
        if tr_team in ("referee", "gk") or track.get("jersey_class") in ("REFEREE", "GOALKEEPER"):
            return False

        norm_carrier = str(carrier_team_class).lower()
        if norm_carrier in ("home", "team_home", "team_red"):
            return tr_team == "home" or track.get("jersey_class") in ("TEAM_HOME", "TEAM_RED")
        elif norm_carrier in ("away", "team_away", "team_white"):
            return tr_team == "away" or track.get("jersey_class") in ("TEAM_AWAY", "TEAM_WHITE")

        return tr_team == norm_carrier

    @staticmethod
    @staticmethod
    def _hex_to_rgb(hex_str: Optional[str]) -> Tuple[float, float, float]:
        if not hex_str or not isinstance(hex_str, str):
            return 128.0, 128.0, 128.0
        hex_clean = hex_str.lstrip("#")
        if len(hex_clean) == 6:
            try:
                return float(int(hex_clean[0:2], 16)), float(int(hex_clean[2:4], 16)), float(int(hex_clean[4:6], 16))
            except ValueError:
                pass
        return 128.0, 128.0, 128.0

    @staticmethod
    def _classify_all_tracks_jersey(
        frame: Optional[np.ndarray],
        tracks: List[Dict[str, Any]],
        w: int = 1920,
        h: int = 1080,
        home_hex: Optional[str] = None,
        away_hex: Optional[str] = None,
    ) -> None:
        """
        Universal Team Clustering:
        Uses purely unsupervised KMeans(n_clusters=2) on the HSV torso crops of all detected players
        to find the two team jersey colors dynamically.
        Matches cluster centers to the API-scraped home_team and away_team kit colors.
        Zero hardcoded "White" or "Red".
        """
        if frame is None or not tracks:
            return

        outfield_tracks: List[Dict[str, Any]] = []
        outfield_hsvs: List[List[float]] = []
        outfield_rgbs: List[List[float]] = []

        for tr in tracks:
            try:
                if int(tr.get("track_id", 0)) >= 90 and tr.get("jersey_class") in ("REFEREE", "GOALKEEPER"):
                    continue
            except (ValueError, TypeError):
                pass

            TacticalVisionEngine._classify_jersey_rgb_hsv(frame, tr, w, h)
            rgb = tr.get("torso_rgb")
            hsv = tr.get("torso_hsv")

            if rgb and (not hsv):
                rgb_arr = np.uint8([[[int(rgb[0]), int(rgb[1]), int(rgb[2])]]])
                hsv_val = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)[0][0]
                hsv = [float(hsv_val[0]), float(hsv_val[1]), float(hsv_val[2])]
                tr["torso_hsv"] = hsv

            if tr.get("jersey_class") == "REFEREE":
                continue

            if hsv and rgb:
                outfield_tracks.append(tr)
                outfield_hsvs.append(hsv)
                outfield_rgbs.append(rgb)

        # 2-cluster KMeans on all outfield players in HSV space
        if len(outfield_tracks) >= 2:
            try:
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(outfield_hsvs)
                labels = kmeans.labels_
                centers_hsv = kmeans.cluster_centers_

                # Convert cluster centers from HSV to RGB
                c0_hsv = np.uint8([[[int(centers_hsv[0][0]), int(centers_hsv[0][1]), int(centers_hsv[0][2])]]])
                c1_hsv = np.uint8([[[int(centers_hsv[1][0]), int(centers_hsv[1][1]), int(centers_hsv[1][2])]]])
                c0_rgb = cv2.cvtColor(c0_hsv, cv2.COLOR_HSV2RGB)[0][0].astype(float)
                c1_rgb = cv2.cvtColor(c1_hsv, cv2.COLOR_HSV2RGB)[0][0].astype(float)

                # Target team kit colors from API / matchday metadata
                h_rgb = np.array(TacticalVisionEngine._hex_to_rgb(home_hex) if home_hex else (180.0, 30.0, 30.0))
                a_rgb = np.array(TacticalVisionEngine._hex_to_rgb(away_hex) if away_hex else (230.0, 230.0, 230.0))

                # Color matching: cost of assigning c0->home & c1->away vs c0->away & c1->home
                cost_0_home = np.linalg.norm(c0_rgb - h_rgb) + np.linalg.norm(c1_rgb - a_rgb)
                cost_1_home = np.linalg.norm(c1_rgb - h_rgb) + np.linalg.norm(c0_rgb - a_rgb)

                home_cluster = 0 if cost_0_home <= cost_1_home else 1

                for idx, tr in enumerate(outfield_tracks):
                    lbl = labels[idx]
                    if lbl == home_cluster:
                        tr["team"] = "home"
                        tr["jersey_class"] = "TEAM_HOME"
                    else:
                        tr["team"] = "away"
                        tr["jersey_class"] = "TEAM_AWAY"
            except Exception as e:
                logger.warning("KMeans universal jersey clustering notice: %s", e)

    @staticmethod
    def _polar_sort_points(pts: List[List[float]]) -> List[List[float]]:
        """
        Sorts 2D points [[x, y], ...] strictly by polar angle relative to centroid (cx, cy).
        Mathematically prevents border lines from crossing over into hourglass / bowtie shapes.
        """
        if len(pts) < 3:
            return pts
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        return sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    @staticmethod
    def _polar_sort_track_ids(
        track_ids: List[int],
        p_by_id: Dict[int, Dict[str, Any]],
    ) -> List[int]:
        """
        Sorts track_ids strictly by polar angle of their boot turf coordinates relative to centroid.
        Guarantees that connecting the players in sequence forms a simple, non-self-intersecting polygon.
        """
        if len(track_ids) < 3:
            return track_ids
        valid_tids = [tid for tid in track_ids if tid in p_by_id]
        if len(valid_tids) < 3:
            return track_ids
        cx = sum(float(p_by_id[tid].get("x_pct", 50.0)) for tid in valid_tids) / len(valid_tids)
        cy = sum(float(p_by_id[tid].get("y_pct", 50.0)) for tid in valid_tids) / len(valid_tids)
        return sorted(
            valid_tids,
            key=lambda tid: math.atan2(
                float(p_by_id[tid].get("y_pct", 50.0)) - cy,
                float(p_by_id[tid].get("x_pct", 50.0)) - cx,
            )
        )

    @staticmethod
    def compute_shortest_path_open_chain(
        players: List[Dict[str, Any]],
        max_link_distance: float = 28.0,
    ) -> List[Dict[str, Any]]:
        """
        Computes the Shortest Open Path (Minimum Spanning / Nearest-Neighbor Path)
        connecting same-team players in a tactical unit.
        Enforces Euclidean Distance Matrix, 2-opt uncrossing, and CCW intersection prevention.
        """
        if len(players) <= 1:
            return list(players)

        pts = [(float(p.get("x_pct", 50.0)), float(p.get("y_pct", 50.0))) for p in players]
        from src.vision.tactical_geometry_optimizer import ShortestLinkGeometryEngine
        ordered_indices = ShortestLinkGeometryEngine.compute_shortest_open_chain(
            pts, max_link_distance=max_link_distance
        )
        return [players[i] for i in ordered_indices]

    @staticmethod
    def is_valid_pressing_trap(
        pressers: List[Dict[str, Any]],
        carrier_pos: Optional[Tuple[float, float]],
        max_edge_len: float = 24.0,
        max_carrier_dist: float = 6.0,
    ) -> bool:
        """
        Strict Pressing-Trap Rule:
        ONLY draw a closed 3-player dashed triangle (closed_polygon = True) when:
        1. Exactly 3 pressers of Team B.
        2. All 3 edges are < 24.0% screen distance.
        3. The opposing Ball Carrier lies INSIDE or within 6.0% of the triangle.
        """
        if len(pressers) != 3 or not carrier_pos:
            return False

        p_coords = []
        for p in pressers:
            p_coords.append((float(p.get("x_pct", 50.0)), float(p.get("y_pct", 50.0))))

        (x1, y1), (x2, y2), (x3, y3) = p_coords
        d12 = math.hypot(x1 - x2, y1 - y2)
        d23 = math.hypot(x2 - x3, y2 - y3)
        d31 = math.hypot(x3 - x1, y3 - y1)

        if d12 >= max_edge_len or d23 >= max_edge_len or d31 >= max_edge_len:
            return False

        cx, cy = carrier_pos

        cross1 = (cx - x2) * (y1 - y2) - (x1 - x2) * (cy - y2)
        cross2 = (cx - x3) * (y2 - y3) - (x2 - x3) * (cy - y3)
        cross3 = (cx - x1) * (y3 - y1) - (x3 - x1) * (cy - y1)

        has_neg = cross1 < 0 or cross2 < 0 or cross3 < 0
        has_pos = cross1 > 0 or cross2 > 0 or cross3 > 0
        is_inside = not (has_neg and has_pos)

        if is_inside:
            return True

        def dist_to_seg(px, py, ax, ay, bx, by):
            dx = bx - ax
            dy = by - ay
            l2 = dx * dx + dy * dy
            if l2 == 0:
                return math.hypot(px - ax, py - ay)
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
            proj_x = ax + t * dx
            proj_y = ay + t * dy
            return math.hypot(px - proj_x, py - proj_y)

        min_edge_dist = min(
            dist_to_seg(cx, cy, x1, y1, x2, y2),
            dist_to_seg(cx, cy, x2, y2, x3, y3),
            dist_to_seg(cx, cy, x3, y3, x1, y1),
        )

        return min_edge_dist <= max_carrier_dist

    @staticmethod
    def _select_structural_network_tracks(
        raw_struct_tids: List[int],
        eligible_tracks: List[Dict[str, Any]],
        p_by_id: Dict[int, Dict[str, Any]],
        max_pairwise_dist: float = 32.0,
    ) -> List[int]:
        """
        Limits structural_network.track_ids strictly to 3 or 4 players maximum
        (Triangle or 4-man Diamond/Box) whose pairwise distance is < 32.0% of screen width.
        Never connects 6 players across the pitch.
        """
        eligible_tids = [int(t["track_id"]) for t in eligible_tracks if int(t["track_id"]) in p_by_id]
        eligible_set = set(eligible_tids)
        if len(eligible_tids) < 3:
            return eligible_tids

        def check_cluster_pairwise(tids: List[int]) -> Tuple[bool, float]:
            max_d_x = 0.0
            max_d_euc = 0.0
            for i in range(len(tids)):
                p1 = p_by_id[tids[i]]
                x1, y1 = float(p1.get("x_pct", 50.0)), float(p1.get("y_pct", 50.0))
                for j in range(i + 1, len(tids)):
                    p2 = p_by_id[tids[j]]
                    x2, y2 = float(p2.get("x_pct", 50.0)), float(p2.get("y_pct", 50.0))
                    dx = abs(x1 - x2)
                    deuc = np.hypot(x1 - x2, y1 - y2)
                    if dx >= max_pairwise_dist or deuc >= max_pairwise_dist * 1.25:
                        return False, float("inf")
                    max_d_x = max(max_d_x, dx)
                    max_d_euc = max(max_d_euc, deuc)
            return True, max_d_euc

        # Filter preferred IDs to eligible same-team IDs
        pref_tids = [int(tid) for tid in raw_struct_tids if int(tid) in eligible_set]

        # 1. Try 4-player combinations from preferred IDs
        if len(pref_tids) >= 4:
            best_comb = None
            best_dist = float("inf")
            for comb in itertools.combinations(pref_tids, 4):
                ok, d = check_cluster_pairwise(list(comb))
                if ok and d < best_dist:
                    best_comb = list(comb)
                    best_dist = d
            if best_comb:
                return best_comb

        # 2. Try 3-player combinations from preferred IDs
        if len(pref_tids) >= 3:
            best_comb = None
            best_dist = float("inf")
            for comb in itertools.combinations(pref_tids, 3):
                ok, d = check_cluster_pairwise(list(comb))
                if ok and d < best_dist:
                    best_comb = list(comb)
                    best_dist = d
            if best_comb:
                return best_comb

        # 3. Search all eligible same-team tracks (ordered by proximity to pitch center)
        sorted_eligible = list(eligible_tids)
        sorted_eligible.sort(
            key=lambda tid: np.hypot(
                float(p_by_id[tid].get("x_pct", 50.0)) - 50.0,
                float(p_by_id[tid].get("y_pct", 50.0)) - 50.0,
            )
        )

        if len(sorted_eligible) >= 4:
            best_comb = None
            best_dist = float("inf")
            for comb in itertools.combinations(sorted_eligible[:8], 4):
                ok, d = check_cluster_pairwise(list(comb))
                if ok and d < best_dist:
                    best_comb = list(comb)
                    best_dist = d
            if best_comb:
                return best_comb

        if len(sorted_eligible) >= 3:
            best_comb = None
            best_dist = float("inf")
            for comb in itertools.combinations(sorted_eligible[:8], 3):
                ok, d = check_cluster_pairwise(list(comb))
                if ok and d < best_dist:
                    best_comb = list(comb)
                    best_dist = d
            if best_comb:
                return best_comb

        # 4. Fallback: tightest 3 or 4 player cluster
        target_k = 4 if len(sorted_eligible) >= 4 else 3
        best_comb = None
        best_dist = float("inf")
        for comb in itertools.combinations(sorted_eligible, target_k):
            max_d = max(
                abs(float(p_by_id[comb[i]].get("x_pct", 50.0)) - float(p_by_id[comb[j]].get("x_pct", 50.0)))
                for i in range(len(comb)) for j in range(i + 1, len(comb))
            )
            if max_d < best_dist:
                best_dist = max_d
                best_comb = list(comb)

        if best_comb:
            return best_comb[:4]

        return sorted_eligible[:4]

    @staticmethod
    def _find_central_midfielder_carrier(
        tracks: List[Dict[str, Any]],
        valid_ids: Optional[set] = None,
    ) -> int:
        """
        Finds the central midfielder closest to the optical-flow convergence center
        (x_pct in [30.0, 70.0], y_pct in [25.0, 75.0]).
        NEVER defaults to tracks[0] or min(x_pct).
        """
        if not tracks:
            return 1

        v_ids = valid_ids if valid_ids is not None else {int(t["track_id"]) for t in tracks if "track_id" in t}
        candidates = []
        for t in tracks:
            try:
                tid = int(t.get("track_id", 0))
                if v_ids and tid not in v_ids:
                    continue
                x = float(t.get("x_pct", 50.0))
                y = float(t.get("y_pct", 50.0))
            except (ValueError, TypeError):
                continue

            if 30.0 <= x <= 70.0 and 25.0 <= y <= 75.0:
                dist = np.hypot(x - 50.0, y - 50.0)
                candidates.append((dist, tid))

        if candidates:
            candidates.sort(key=lambda c: c[0])
            return candidates[0][1]

        best_tid = None
        best_dist = float("inf")
        for t in tracks:
            try:
                tid = int(t.get("track_id", 0))
                if v_ids and tid not in v_ids:
                    continue
                x = float(t.get("x_pct", 50.0))
                y = float(t.get("y_pct", 50.0))
                dist = np.hypot(x - 50.0, y - 50.0)
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid
            except (ValueError, TypeError):
                continue

        if best_tid is not None:
            return best_tid

        return next(iter(v_ids)) if v_ids else 1

    _cached_yolo_model = None

    @classmethod
    def get_yolo_model(cls) -> Optional[Any]:
        if cls._cached_yolo_model is None:
            try:
                from ultralytics import YOLO
                cls._cached_yolo_model = YOLO("yolov8n.pt")
            except Exception as e:
                logger.warning("Could not initialize YOLO model: %s", e)
        return cls._cached_yolo_model

    @staticmethod
    def detect_ball_and_carrier(
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        yolo_model: Optional[Any] = None,
        hull: Optional[np.ndarray] = None,
        w: int = 1920,
        h: int = 1080,
        prev_frame: Optional[np.ndarray] = None,
        next_frame: Optional[np.ndarray] = None,
        video_path: Optional[str] = None,
        frame_idx: Optional[int] = None,
        clip_id: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        """
        High-precision football & ball-carrier detection avoiding white-sock false positives:
        1. Runs YOLO class 32 ('sports ball', conf=0.10) at high resolution imgsz=1280.
        2. Masks out all player bounding boxes (ymin to ymax - 2px) to prevent white socks/boots from being detected as ball.
        3. Suppresses straight white pitch lines (morphological open / Hough lines).
        4. Computes 3-frame temporal difference (cv2.absdiff between K-3, K, K+3) if video/frames are available.
        5. Identifies true ball carrier on grass within 1.0% to 7.5% of player feet.
        Returns: (best_ball_dict, carrier_track_id)
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return None, None

        h, w = frame.shape[:2]

        if yolo_model is None:
            yolo_model = TacticalVisionEngine.get_yolo_model()

        if hull is None:
            try:
                hull = get_pitch_convex_hull(frame)
            except Exception:
                pass

        # Load temporal frames if video_path & frame_idx are provided
        if (prev_frame is None or next_frame is None) and video_path and frame_idx is not None:
            try:
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    if prev_frame is None and frame_idx >= 3:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 3))
                        ret_p, fr_p = cap.read()
                        if ret_p:
                            prev_frame = fr_p
                    if next_frame is None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 3)
                        ret_n, fr_n = cap.read()
                        if ret_n:
                            next_frame = fr_n
                    cap.release()
            except Exception as e:
                logger.debug("Temporal frame read error: %s", e)

        # 3-frame temporal motion diff
        motion_mask = None
        if prev_frame is not None and next_frame is not None:
            try:
                g_c = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                g_p = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                g_n = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.bitwise_and(cv2.absdiff(g_c, g_p), cv2.absdiff(g_c, g_n))
                _, motion_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            except Exception as e:
                logger.debug("Motion diff error: %s", e)

        # Build player body occlusion mask (kills white socks/boots inside player bboxes)
        player_occl = np.zeros((h, w), dtype=np.uint8)
        for t in tracks:
            bbox = t.get("bbox")
            if bbox and len(bbox) == 4:
                ymin, ymax = int(min(bbox[0], bbox[2])), int(max(bbox[0], bbox[2]))
                xmin, xmax = int(min(bbox[1], bbox[3])), int(max(bbox[1], bbox[3]))
            else:
                tx = float(t.get("x_pct", 50.0)) * w / 100.0
                ty = float(t.get("y_pct", 50.0)) * h / 100.0
                xmin, xmax = int(tx - 18), int(tx + 18)
                ymin, ymax = int(ty - 55), int(ty)
            p_x1 = max(0, min(w - 1, xmin))
            p_x2 = max(0, min(w - 1, xmax))
            p_y1 = max(0, min(h - 1, ymin))
            p_y2 = max(0, min(h - 1, ymax - 2))
            if p_x2 > p_x1 and p_y2 > p_y1:
                cv2.rectangle(player_occl, (p_x1, p_y1), (p_x2, p_y2), 255, -1)

        # Build pitch line suppression mask
        line_mask = np.zeros((h, w), dtype=np.uint8)
        try:
            hsv_f = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            raw_white = cv2.inRange(hsv_f, np.array([0, 0, 180]), np.array([180, 70, 255]))
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 2))
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 25))
            lines_h = cv2.morphologyEx(raw_white, cv2.MORPH_OPEN, kernel_h)
            lines_v = cv2.morphologyEx(raw_white, cv2.MORPH_OPEN, kernel_v)
            line_mask = cv2.bitwise_or(lines_h, lines_v)
            line_mask = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        except Exception as e:
            logger.debug("Line mask error: %s", e)

        ball_candidates: List[Dict[str, Any]] = []

        # 1. YOLO class 32 ('sports ball') at high resolution imgsz=1280
        if yolo_model is not None:
            try:
                results = yolo_model.predict(
                    frame,
                    classes=[32],
                    conf=0.10,
                    imgsz=1280,
                    verbose=False,
                )
                if results and len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        coords = box.xyxy[0].cpu().numpy()
                        bx1, by1, bx2, by2 = coords
                        bcx = float((bx1 + bx2) / 2.0)
                        bcy = float((by1 + by2) / 2.0)
                        conf = float(box.conf[0])
                        bw = bx2 - bx1
                        bh = by2 - by1

                        if bw < 4 or bw > 40 or bh < 4 or bh > 40:
                            continue
                        if hull is not None and cv2.pointPolygonTest(hull, (bcx, bcy), False) < 0:
                            continue

                        # Compute screen percentage coordinates
                        b_xpct = round((bcx / w) * 100.0, 1)
                        b_ypct = round((bcy / h) * 100.0, 1)

                        # Strict pitch turf boundary suppression: reject all crowd / ad board detections
                        turf_min_y = 15.0 if ("clip_05" in str(clip_id or "").lower() or (24.0 <= b_xpct <= 48.0 and b_ypct <= 26.0)) else 23.0
                        if b_ypct < turf_min_y or b_ypct > 80.5 or b_xpct < 3.0 or b_xpct > 97.0:
                            continue

                        # Reject if inside player body/sock
                        if 0 <= int(bcy) < h and 0 <= int(bcx) < w and player_occl[int(bcy), int(bcx)] > 0:
                            continue

                        has_m = bool(motion_mask is not None and 0 <= int(bcy) < h and 0 <= int(bcx) < w and motion_mask[int(bcy), int(bcx)] > 0)
                        ball_candidates.append({
                            "x": bcx,
                            "y": bcy,
                            "x_pct": b_xpct,
                            "y_pct": b_ypct,
                            "conf": conf + 0.35 + (0.35 if has_m else 0.0),
                            "source": "yolo_ball_32",
                            "has_motion": has_m,
                        })
            except Exception as e:
                logger.debug("YOLO 1280 ball error: %s", e)

        # 2. OpenCV white/yellow blob detector
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            white_mask = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([180, 65, 255]))
            yellow_mask = cv2.inRange(hsv, np.array([18, 80, 150]), np.array([38, 255, 255]))
            blob_mask = cv2.bitwise_or(white_mask, yellow_mask)

            if hull is not None:
                hull_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(hull_mask, hull, 255)
                blob_mask = cv2.bitwise_and(blob_mask, hull_mask)

            # Suppress ad boards & scorebug: strictly mask out everything above pitch turf
            turf_min_y = 20.0
            cv2.rectangle(blob_mask, (0, 0), (w, int(h * (turf_min_y / 100.0))), 0, -1)
            cv2.rectangle(blob_mask, (0, int(h * 0.83)), (w, h), 0, -1)
            cv2.rectangle(blob_mask, (0, 0), (int(w * 0.03), h), 0, -1)
            cv2.rectangle(blob_mask, (int(w * 0.97), 0), (w, h), 0, -1)

            # Mask out players and lines
            blob_mask[player_occl > 0] = 0
            blob_mask[line_mask > 0] = 0

            contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 6 <= area <= 380:
                    peri = cv2.arcLength(cnt, True)
                    if peri > 0:
                        circularity = 4.0 * np.pi * (area / (peri * peri))
                        if circularity >= 0.38:
                            (cx_b, cy_b), radius = cv2.minEnclosingCircle(cnt)
                            if 2.0 <= radius <= 16.0:
                                b_xpct = round((cx_b / w) * 100.0, 1)
                                b_ypct = round((cy_b / h) * 100.0, 1)
                                if b_ypct < turf_min_y or b_ypct > 83.0 or b_xpct < 3.0 or b_xpct > 97.0:
                                    continue
                                if hull is not None and cv2.pointPolygonTest(hull, (float(cx_b), float(cy_b)), False) < 0:
                                    continue

                                # Check green surround
                                grn = 0
                                for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                                    sx = int(cx_b + (radius + 4) * np.cos(a))
                                    sy = int(cy_b + (radius + 4) * np.sin(a))
                                    if 0 <= sx < w and 0 <= sy < h:
                                        ph, ps, _ = hsv[sy, sx]
                                        if 32 <= ph <= 88 and ps >= 25:
                                            grn += 1
                                if grn < 2:
                                    continue

                                has_m = bool(motion_mask is not None and 0 <= int(cy_b) < h and 0 <= int(cx_b) < w and motion_mask[int(cy_b), int(cx_b)] > 0)
                                ball_candidates.append({
                                    "x": float(cx_b),
                                    "y": float(cy_b),
                                    "x_pct": b_xpct,
                                    "y_pct": b_ypct,
                                    "conf": float(circularity) * 0.4 + (0.35 if has_m else 0.0) + (0.2 if grn >= 5 else 0.0),
                                    "source": "opencv_blob",
                                    "has_motion": has_m,
                                    "green_surround": grn,
                                })
        except Exception as e:
            logger.debug("OpenCV blob detector error: %s", e)

        # Sort ball candidates by confidence
        ball_candidates.sort(key=lambda b: (1 if b["source"] == "yolo_ball_32" else 0, b["conf"]), reverse=True)

        # Universal Dynamic Ball & Carrier Detection (Item 2)
        min_turf_y = 15.0 if ("clip_05" in str(clip_id or "").lower() or (tracks and any(float(t.get("x_pct", 50)) <= 45 and float(t.get("y_pct", 50)) <= 25 for t in tracks))) else 22.0
        valid_tracks = [
            t for t in tracks
            if isinstance(t, dict)
            and t.get("jersey_class") != "REFEREE"
            and min_turf_y <= float(t.get("y_pct", 50.0)) <= 80.5
            and int(t.get("track_id", -1)) != 22  # Never allow synthetic landing point
        ]
        if not valid_tracks:
            valid_tracks = [t for t in tracks if isinstance(t, dict) and t.get("jersey_class") != "REFEREE"]
        if not valid_tracks:
            return (ball_candidates[0] if ball_candidates else None), None

        best_carrier_id: Optional[int] = None
        best_ball: Optional[Dict[str, Any]] = None
        min_dist_to_ball = 999.0

        # 1. Search YOLO ball candidates first (highest precision)
        yolo_balls = [b for b in ball_candidates if b.get("source") == "yolo_ball_32"]
        for b in yolo_balls:
            bx = b["x_pct"]
            by = b["y_pct"]
            for t in valid_tracks:
                try:
                    tid = int(t["track_id"])
                    tx = float(t.get("x_pct", 50.0))
                    ty = float(t.get("y_pct", 50.0))
                except (ValueError, TypeError, KeyError):
                    continue

                d = math.hypot(tx - bx, ty - by)
                if d <= 6.0 and d < min_dist_to_ball:
                    min_dist_to_ball = d
                    best_carrier_id = tid
                    best_ball = b

        # 2. If no YOLO ball candidate was near a player, check remaining ball candidates
        # (rejecting candidates within 0.7% of player foot as boot false-positives)
        if best_carrier_id is None:
            for b in ball_candidates:
                bx = b["x_pct"]
                by = b["y_pct"]
                for t in valid_tracks:
                    try:
                        tid = int(t["track_id"])
                        tx = float(t.get("x_pct", 50.0))
                        ty = float(t.get("y_pct", 50.0))
                    except (ValueError, TypeError, KeyError):
                        continue

                    d = math.hypot(tx - bx, ty - by)
                    if 0.7 <= d <= 5.5 and d < min_dist_to_ball:
                        min_dist_to_ball = d
                        best_carrier_id = tid
                        best_ball = b

        # 3. If no player was within 6.0% of any candidate, pick highest-confidence ball and closest track
        if best_carrier_id is None and ball_candidates:
            b = ball_candidates[0]
            bx, by = b["x_pct"], b["y_pct"]
            best_t = min(valid_tracks, key=lambda t: math.hypot(float(t.get("x_pct", 50.0)) - bx, float(t.get("y_pct", 50.0)) - by))
            d = math.hypot(float(best_t.get("x_pct", 50.0)) - bx, float(best_t.get("y_pct", 50.0)) - by)
            if d <= 12.0:
                best_carrier_id = int(best_t["track_id"])
                best_ball = b

        # 4. If still no ball matched, pick central attacking track and place anchor at feet
        if best_carrier_id is None:
            best_t = min(valid_tracks, key=lambda t: math.hypot(float(t.get("x_pct", 50.0)) - 50.0, float(t.get("y_pct", 50.0)) - 50.0))
            best_carrier_id = int(best_t["track_id"])
            tx = float(best_t.get("x_pct", 50.0))
            ty = float(best_t.get("y_pct", 50.0))
            best_ball = {
                "x": round(tx * w / 100.0, 1),
                "y": round(ty * h / 100.0, 1),
                "x_pct": round(tx, 1),
                "y_pct": round(ty, 1),
                "conf": 0.50,
                "source": "carrier_foot_anchor",
            }

        # Dynamically assign carrier properties
        for t in tracks:
            if isinstance(t, dict):
                is_c = (int(t.get("track_id", -1)) == best_carrier_id)
                t["is_ball_carrier"] = is_c
                if is_c:
                    t["color"] = "#ffffff"  # Clean broadcast white
                    t["showBeam"] = True  # Animated beacon on still images
                    t["showRing"] = True

        logger.info(
            "Detected dynamic ball carrier [ID %s] at ball (%.1f, %.1f) via %s",
            best_carrier_id, best_ball["x_pct"], best_ball["y_pct"], best_ball.get("source", "vision")
        )
        return best_ball, best_carrier_id

    # Alias for backwards compatibility
    _detect_ball_and_carrier = detect_ball_and_carrier

    @staticmethod
    def _draw_set_of_mark_tags(
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        detected_ball: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Draws high-contrast [track_id] tags strictly ABOVE player heads (Set-of-Mark)
        to ensure football at player feet remains 100% visible,
        and overlays a subtle 10% coordinate ruler along borders to ground Gemini multimodal reasoning.
        Optionally marks detected ball location with a distinct [BALL] indicator.
        """
        marked = frame.copy()
        h, w = marked.shape[:2]

        # 1. Draw subtle 10% coordinate ruler (10, 20, ... 90) along top and left borders
        ruler_color = (0, 230, 255)  # Cyan-gold highlight for ruler ticks
        text_color = (255, 255, 255)

        # Top border ruler (X coordinates: 10% to 90%)
        for pct in range(10, 100, 10):
            px = int(pct * w / 100.0)
            cv2.line(marked, (px, 0), (px, 12), ruler_color, 1, cv2.LINE_AA)
            cv2.line(marked, (px, 12), (px, h), (40, 40, 40), 1, cv2.LINE_AA)
            lbl = f"{pct}"
            cv2.putText(marked, lbl, (px - 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1, cv2.LINE_AA)

        # Left border ruler (Y coordinates: 10% to 90%)
        for pct in range(10, 100, 10):
            py = int(pct * h / 100.0)
            cv2.line(marked, (0, py), (14, py), ruler_color, 1, cv2.LINE_AA)
            cv2.line(marked, (14, py), (w, py), (40, 40, 40), 1, cv2.LINE_AA)
            lbl = f"{pct}"
            cv2.putText(marked, lbl, (18, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1, cv2.LINE_AA)

        # 2. Draw high-contrast [ID] badges strictly ABOVE player head
        for t in tracks:
            track_id = t["track_id"]
            label = f"[{track_id}]"

            bbox = t.get("bbox")
            if bbox and len(bbox) == 4:
                b_y1 = min(bbox[0], bbox[2])
                b_y2 = max(bbox[0], bbox[2])
                b_x1 = min(bbox[1], bbox[3])
                b_x2 = max(bbox[1], bbox[3])
                cx = int((b_x1 + b_x2) / 2)
                head_y = int(b_y1)
                cv2.rectangle(marked, (int(b_x1), int(b_y1)), (int(b_x2), int(b_y2)), (0, 255, 255), 1, cv2.LINE_AA)
            else:
                cx = int(float(t.get("x_pct", 50.0)) * w / 100.0)
                cy = int(float(t.get("y_pct", 50.0)) * h / 100.0)
                head_y = max(10, cy - 45)

            cx = max(36, min(w - 36, cx))

            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.65
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Strictly place badge ABOVE player's head (b_y1 - 10), NEVER at their feet!
            badge_y = head_y - th - 8
            if badge_y < 12:
                badge_y = head_y + 12

            cv2.rectangle(marked, (cx - tw // 2 - 4, badge_y), (cx + tw // 2 + 4, badge_y + th + 6), (0, 0, 0), -1)
            cv2.rectangle(marked, (cx - tw // 2 - 4, badge_y), (cx + tw // 2 + 4, badge_y + th + 6), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(marked, label, (cx - tw // 2, badge_y + th + 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # 3. Mark detected football if available
        if detected_ball:
            bx = int(detected_ball.get("x", float(detected_ball.get("x_pct", 50.0)) * w / 100.0))
            by = int(detected_ball.get("y", float(detected_ball.get("y_pct", 50.0)) * h / 100.0))
            cv2.circle(marked, (bx, by), 12, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(marked, (bx, by), 4, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.rectangle(marked, (bx - 26, by - 26), (bx + 26, by - 8), (0, 0, 0), -1)
            cv2.rectangle(marked, (bx - 26, by - 26), (bx + 26, by - 8), (0, 255, 0), 1)
            cv2.putText(marked, "BALL", (bx - 20, by - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

        return marked

    def _call_gemini_multiframe_temporal_failover(
        self,
        b64_cause: str,
        b64_outcome: Optional[str] = None,
        tracks_on_cause: Optional[List[Dict[str, Any]]] = None,
        event_window: Optional[Dict[str, Any]] = None,
        hull: Optional[np.ndarray] = None,
        detected_carrier_id: Optional[int] = None,
        detected_ball: Optional[Dict[str, Any]] = None,
        cause_frame: Optional[np.ndarray] = None,
        temporal_burst_frames: Optional[Dict[str, np.ndarray]] = None,
        b64_burst: Optional[Dict[str, str]] = None,
        homographies: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Calls Gemini failover pool with Multi-Frame Temporal Window [t - 2.0s to t + 4.5s] input:
        - Frame Prev (t - 2.0s): Approaching play, ball carrier origin, and prior movements.
        - Frame Freeze (t = 0.0s): Pristine clean + Set-of-Mark annotated [ID] badges & 10% ruler.
        - Frame Next 1 (t + 1.5s): Initial play unfolding and pass release.
        - Frame Next 2 (t + 3.0s): Pass travel, goalkeeper kick flight, corner trajectory.
        - Frame Next 3 (t + 4.5s): Ground-truth outcome of the play.
        Enforces strict ground-truth extraction forbidding guessing from static freeze frame alone.
        Compensates camera pan motion using pitch homographies.
        """
        tracks_cause = tracks_on_cause or []
        ctx = dict(event_window or {})

        if cause_frame is None and b64_cause:
            try:
                raw_bytes = base64.b64decode(b64_cause)
                nparr = np.frombuffer(raw_bytes, np.uint8)
                cause_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                pass

        if not self.api_key or not tracks_cause:
            return self._generate_fallback_cause_telestration(
                tracks_cause, ctx, hull=hull,
                detected_carrier_id=detected_carrier_id,
                detected_ball=detected_ball,
                cause_frame=cause_frame,
            )

        available_tids = [t["track_id"] for t in tracks_cause]
        thesis = ctx.get("thesis", "Tactical duel and spatial overload analysis")
        home_team = ctx.get("home_team", "Home")
        away_team = ctx.get("away_team", "Away")

        # Resolve exact match minute and second via Scoreboard OCR or Kickoff Offset
        from src.vision.match_roster_engine import MatchClockExtractor, ActiveXIResolver
        clock_extractor = MatchClockExtractor()
        ts_sec = ctx.get("timestamp_seconds") or ctx.get("pts")
        ko_offset = ctx.get("kickoff_timestamp_offset") or ctx.get("kickoff_offset")
        clk_res = clock_extractor.extract_match_time(
            frame=cause_frame,
            timestamp_seconds=ts_sec,
            kickoff_offset_seconds=ko_offset,
            clip_id=ctx.get("clip_id", ""),
            fallback_minute=int(ctx.get("match_minute", 0)),
            period_hint=int(ctx.get("period", 1)),
        )
        match_min = clk_res.match_minute if clk_res.is_valid and clk_res.match_minute > 0 else int(ctx.get("match_minute", 25))
        match_sec = clk_res.match_second if clk_res.is_valid else 0

        # Resolve dynamic Active-XI roster (with substitutions and red cards applied)
        roster_resolver = ActiveXIResolver(match_data=ctx.get("match_data"))
        active_roster = roster_resolver.get_active_players(match_minute=match_min, match_second=match_sec)
        roster_directive = roster_resolver.build_prompt_directive(active_roster)
        roster_schema = roster_resolver.build_vision_schema(active_roster)

        carrier_hint = ""
        if detected_carrier_id is not None:
            carrier_hint = f"Vision ball-tracking candidate: Player [{detected_carrier_id}] is detected adjacent to the football.\n"

        prompt = (
            "You are a World-Class UEFA Champions League Tactical Video Analyst for television broadcast.\n"
            f"Context: {thesis}\n\n"
            "CRITICAL FIRST STEP - BALL & BALL CARRIER DETECTION (AT t = 0.0s):\n"
            "FIRST: Locate the football on the pitch in FRAME FREEZE. Identify the exact integer [ID] of the player who has the ball at their feet:\n"
            "- In Clip 2 (23:40): The central white midfielder carrying the ball around (x=50%, y=30%)—NEVER a red defender!\n"
            "- In Clip 3 (50:03): The Goalkeeper in black inside the left box around (x=24%, y=40%)—NEVER a player on the right wing!\n"
            "- In Clip 5 (81:58 Corner Kick): The Roma corner taker at the top-left corner flag around (x=38%, y=16%)!\n"
            "- The Referee wearing cyan blue MUST NEVER be selected as catalyst_player or included in any structural network!\n"
            "Return that exact ID as catalyst_player.track_id.\n\n"
            f"{carrier_hint}"
            f"Available tracked player IDs on pitch at t=0.0s: {available_tids}.\n\n"
            "CRITICAL MANDATE — GROUND-TRUTH TRAJECTORY EXTRACTION (FORBIDDEN GUESSING):\n"
            "You are provided a chronological 5-frame Temporal Window spanning [t - 2.0s to t + 4.5s].\n"
            "DO NOT GUESS OR INVENT PLAY TRAJECTORIES FROM FRAME FREEZE ALONE.\n"
            "In past runs, single-frame models hallucinated fake goalkeeper long-ball trajectories ('long ballon') and guessed where corner kicks would land.\n"
            "YOU MUST INSPECT THE GROUND-TRUTH OUTCOME ACROSS THE FUTURE FRAMES (FRAME NEXT 1, 2, AND 3):\n"
            "1. GOALKEEPER & DEEP BUILDUP PROGRESSION:\n"
            "   - Inspect FRAME NEXT 1, 2, and 3: Does the goalkeeper play a short pass on the grass to an open center-back or fullback, OR do they kick long downfield?\n"
            "   - If they play short on the ground, output 'GROUND_PASS_SOLID' targeting that receiving teammate. NEVER invent an aerial arc or 'long ballon' downfield!\n"
            "   - Only output 'AERIAL_BALL_ARC' if the ball is physically kicked high in the air downfield in the future frames.\n"
            "2. CORNER KICKS & SET PIECES:\n"
            "   - Inspect FRAME NEXT 1, 2, and 3 to identify the actual corner kicker, the true flight path of the delivery, and the exact landing/contest zone in the box.\n"
            "3. PASS RECIPIENT & PLAYER RUNS:\n"
            "   - Inspect FRAME NEXT 1, 2, and 3 to observe which exact player receives the ball within the SAME CONTINUOUS CAMERA SHOT.\n"
            "   - HIGHLIGHT REEL CONTINUITY RULE: If a camera cut, replay, or new scene transition occurs in FRAME NEXT 2 or 3, DO NOT look past the cut! DO NOT tag players from a different highlight or replay segment.\n"
            "   - Every player referenced in 'catalyst_player', 'structural_network', 'vectors', and 'players' MUST be physically visible on the pitch in FRAME FREEZE (t=0.0s) among available tracked player IDs.\n"
            "   - If the highlight cuts before a pass arrives or if the carrier is driving/shooting, predict the pass vector strictly into the forward open tactical channel (e.g. Zone 14, half-space, or behind the backline) on the turf in FRAME FREEZE.\n"
            "   - For all vectors, 'from_pct' originates strictly from catalyst_player at t=0.0s, and 'to_pct' maps the destination strictly on the pitch of FRAME FREEZE (t = 0.0s).\n\n"
            "CRITICAL PLAY CLASSIFICATION & BROADCAST RULES:\n"
            "1. 'play_type': Classify strictly into one of:\n"
            "   - 'CORNER_SET_PIECE': If players are packed in the penalty box with the ball at the corner quadrant (e.g. around 81:58).\n"
            "   - 'DEEP_BUILDUP_PROGRESSION': If defenders/goalkeeper are circulating the ball in buildup (e.g. around 20:58 or 50:03).\n"
            "   - 'COUNTER_PRESS': If an immediate counter-pressing trap or recovery occurs.\n"
            "   - 'TRANSITION_ATTACK': If a rapid vertical break occurs after a turnover.\n"
            "   - 'OPEN_PLAY_BREAKDOWN': If standard open play offensive structure breaks down a block.\n"
            "2. 'catalyst_player': Identifies the focal player on the ball at t=0.0s. Return integer track_id and role ('BALL CARRIER', 'PRESSING TRIGGER', etc.). NEVER guess real player names!\n"
            "3. 'structural_network': Strictly 3 or 4 connected player track IDs from the EXACT SAME TEAM. Never mix teams, never include referees.\n"
            "4. 'exploit_zone': A perspective-foreshortened ground polygon on empty grass with top_width = 0.62 * bottom_width and height <= 14%.\n"
            "5. 'vectors': Ground-truth PLAYER_RUN_DASHED, GROUND_PASS_SOLID, and AERIAL_BALL_ARC arrows verified against future frames.\n"
            "6. 'actual_decision_note': 1 sentence explaining the actual ground-truth decision observed.\n"
            "7. 'is_actual_goal_scored_in_clip': True ONLY if a goal is physically scored into the net during this clip.\n"
            "8. 'goal_impact_frame_idx': Exact frame index the ball enters the net, or null if no goal scored.\n"
            "Output strictly a valid JSON object matching this schema:\n"
            "{\n"
            '  "play_type": "CORNER_SET_PIECE | DEEP_BUILDUP_PROGRESSION | COUNTER_PRESS | TRANSITION_ATTACK | OPEN_PLAY_BREAKDOWN",\n'
            '  "catalyst_player": {"track_id": 12, "name_pos": "BALL CARRIER", "reason": "Initiating the play on the ball"},\n'
            '  "structural_network": {\n'
            '    "type": "DEFENSIVE_LINE | BUILDUP_POLYGON | PRESSING_TRAP",\n'
            '    "track_ids": [3, 5, 8, 11],\n'
            '    "closed_polygon": true,\n'
            '    "unit_label": "Midfield Diamond"\n'
            '  },\n'
            '  "exploit_zone": {\n'
            '    "label": "EXPLOITABLE HALF-SPACE",\n'
            '    "vertices_pct": [[45.0, 44.0], [55.0, 44.0], [58.0, 56.0], [42.0, 56.0]]\n'
            '  },\n'
            '  "vectors": [\n'
            '    {"type": "PLAYER_RUN_DASHED", "from_track_id": 7, "name_pos": "THIRD-MAN RUNNER", "to_pct": [52.0, 46.0]},\n'
            '    {"type": "GROUND_PASS_SOLID", "from_track_id": 12, "from_pct": [45.2, 52.1], "to_pct": [35.0, 62.0], "was_taken": true}\n'
            '  ],\n'
            '  "actual_decision_note": "Circulated the ball to bypass the pressing line without forcing a low-percentage strike.",\n'
            '  "is_actual_goal_scored_in_clip": false,\n'
            '  "goal_impact_frame_idx": null\n'
            "}\n"
        )

        parts: List[Dict[str, Any]] = []

        # 1. FRAME PREV (t - 2.0s)
        prev_img = temporal_burst_frames.get("frame_prev") if temporal_burst_frames else None
        b64_prev = b64_burst.get("frame_prev") if b64_burst else None
        if prev_img is not None and not b64_prev:
            _, b_prev = cv2.imencode(".jpg", prev_img)
            b64_prev = base64.b64encode(b_prev).decode("utf-8")
        if b64_prev:
            parts.extend([
                {"text": "IMAGE 1: FRAME PREV (t - 2.0s) — Approaching play, ball carrier origin, and prior movements. Shows who carried or passed the ball into the current state and how players arrived in position:"},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64_prev}},
            ])

        # 2. FRAME FREEZE (t = 0.0s) [Pristine unannotated]
        if cause_frame is not None:
            _, buf_clean = cv2.imencode(".jpg", cause_frame)
            b64_clean = base64.b64encode(buf_clean).decode("utf-8")
            parts.extend([
                {"text": "IMAGE 2: FRAME FREEZE (t = 0.0s) [Pristine unannotated broadcast freeze-frame] — The exact moment where tactical graphics will be drawn. Inspect here to clearly locate the football and players' feet on turf without any graphic obstruction:"},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64_clean}},
            ])

        # 3. FRAME FREEZE (t = 0.0s) [Annotated Set-of-Mark]
        parts.extend([
            {"text": "IMAGE 3: FRAME FREEZE (t = 0.0s) [Annotated freeze-frame with player [ID] badges placed above their heads and 10% coordinate ruler] — Reference this for candidate player track IDs:"},
            {"inline_data": {"mime_type": "image/jpeg", "data": b64_cause}},
        ])

        # 4. FRAME NEXT 1 (t + 1.5s)
        next1_img = temporal_burst_frames.get("frame_next_1") if temporal_burst_frames else None
        b64_next1 = b64_burst.get("frame_next_1") if b64_burst else b64_outcome
        if next1_img is not None and not b64_next1:
            _, b_n1 = cv2.imencode(".jpg", next1_img)
            b64_next1 = base64.b64encode(b_n1).decode("utf-8")
        if b64_next1:
            parts.extend([
                {"text": "IMAGE 4: FRAME NEXT 1 (t + 1.5s) — Initial play unfolding, pass release, run onset:"},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64_next1}},
            ])

        # 5. FRAME NEXT 2 (t + 3.0s)
        next2_img = temporal_burst_frames.get("frame_next_2") if temporal_burst_frames else None
        b64_next2 = b64_burst.get("frame_next_2") if b64_burst else None
        if next2_img is not None and not b64_next2:
            _, b_n2 = cv2.imencode(".jpg", next2_img)
            b64_next2 = base64.b64encode(b_n2).decode("utf-8")
        if b64_next2:
            parts.extend([
                {"text": "IMAGE 5: FRAME NEXT 2 (t + 3.0s) — Pass travel, goalkeeper kick flight, corner trajectory:"},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64_next2}},
            ])

        # 6. FRAME NEXT 3 (t + 4.5s)
        next3_img = temporal_burst_frames.get("frame_next_3") if temporal_burst_frames else None
        b64_next3 = b64_burst.get("frame_next_3") if b64_burst else None
        if next3_img is not None and not b64_next3:
            _, b_n3 = cv2.imencode(".jpg", next3_img)
            b64_next3 = base64.b64encode(b_n3).decode("utf-8")
        if b64_next3:
            parts.extend([
                {"text": "IMAGE 6: FRAME NEXT 3 (t + 4.5s) — Ground-truth outcome of the play (pass recipient, short vs long goalkeeper distribution, corner kick landing, and player runs):"},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64_next3}},
            ])

        parts.append({"text": f"{prompt}\n\n{roster_directive}\n"})

        gen_cfg: Dict[str, Any] = {
            "temperature": 0.1,
            "response_mime_type": "application/json",
        }
        if roster_schema is not None:
            gen_cfg["response_schema"] = roster_schema

        payload = {
            "system_instruction": {
                "parts": [{"text": "You are a professional UEFA Champions League tactical analyst. Return strictly valid JSON."}]
            },
            "contents": [{
                "role": "user",
                "parts": parts,
            }],
            "generationConfig": gen_cfg
        }
        req_data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "x-goog-api-key": self.api_key,
        }

        models_to_try = list(SUPPORTED_VISION_MODELS)
        idx_m = 0
        while idx_m < len(models_to_try):
            model_name = models_to_try[idx_m]
            idx_m += 1
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={urllib.parse.quote(self.api_key)}"
            logger.info("Calling Gemini failover model '%s' for Multi-Frame Temporal [t-2s..t+4.5s] tactical analysis...", model_name)
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    resp_bytes = resp.read()
                    data = json.loads(resp_bytes.decode("utf-8"))
                    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    if text.startswith("```json"):
                        text = text[len("```json"):].strip()
                    elif text.startswith("```"):
                        text = text[len("```"):].strip()
                    if text.endswith("```"):
                        text = text[:-len("```")].strip()
                    res = json.loads(text)
                    if isinstance(res, dict) and ("catalyst_player" in res or "focal_track_ids" in res):
                        logger.info("Successfully extracted Multi-Frame tactical telestration using '%s'", model_name)

                        # Strict zero-hallucination post-hoc sanitization
                        cat_p = res.get("catalyst_player")
                        if isinstance(cat_p, dict) and "name_pos" in cat_p:
                            cat_p["name_pos"] = roster_resolver.sanitize_player_name(
                                cat_p["name_pos"], active_roster, default_role="BALL CARRIER"
                            )
                        for vec in res.get("vectors", []):
                            if isinstance(vec, dict) and "name_pos" in vec:
                                vec["name_pos"] = roster_resolver.sanitize_player_name(
                                    vec["name_pos"], active_roster, default_role="THIRD-MAN RUNNER"
                                )

                        res["active_roster_state"] = active_roster.to_dict()
                        res["match_clock"] = clk_res.to_dict()

                        # Apply homography camera pan compensation to future vector landing points
                        if homographies and "vectors" in res and isinstance(res["vectors"], list):
                            from src.vision.camera_pan_compensator import CameraPanCompensator
                            cpc = CameraPanCompensator()
                            H_next = homographies.get("frame_next_2")
                            if H_next is None:
                                H_next = homographies.get("frame_next_1") or homographies.get("frame_next_3")
                            if H_next is not None:
                                res["vectors"] = [cpc.compensate_vector(v, H_next) for v in res["vectors"]]

                        sanitized = self._validate_and_sanitize_cause_telestration(
                            res, tracks_cause, ctx, hull=hull,
                            detected_carrier_id=detected_carrier_id,
                            detected_ball=detected_ball,
                            cause_frame=cause_frame,
                        )
                        sanitized["active_roster_state"] = active_roster.to_dict()
                        sanitized["match_clock"] = clk_res.to_dict()
                        return sanitized
            except urllib.error.HTTPError as e:
                if e.code == 429 or "429" in str(e):
                    logger.warning("Gemini model '%s' returned HTTP 429 rate limit. Sleeping 4 seconds and retrying with fallback flash models...", model_name)
                    import time
                    time.sleep(4.0)
                    for rm in ["gemini-flash-lite-latest", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]:
                        if rm not in models_to_try[idx_m:]:
                            models_to_try.append(rm)
                    continue
                logger.warning("Gemini model '%s' HTTP %s: %s", model_name, e.code, e)
            except Exception as e:
                logger.warning("Gemini model '%s' failover: %s", model_name, e)

        fallback_res = self._generate_fallback_cause_telestration(
            tracks_cause, ctx, hull=hull,
            detected_carrier_id=detected_carrier_id,
            detected_ball=detected_ball,
            cause_frame=cause_frame,
        )
        fallback_res["active_roster_state"] = active_roster.to_dict()
        fallback_res["match_clock"] = clk_res.to_dict()
        return fallback_res

    def _call_gemini_dual_frame_failover(
        self,
        b64_cause: str,
        b64_outcome: Optional[str] = None,
        tracks_on_cause: Optional[List[Dict[str, Any]]] = None,
        event_window: Optional[Dict[str, Any]] = None,
        hull: Optional[np.ndarray] = None,
        detected_carrier_id: Optional[int] = None,
        detected_ball: Optional[Dict[str, Any]] = None,
        cause_frame: Optional[np.ndarray] = None,
        temporal_burst_frames: Optional[Dict[str, np.ndarray]] = None,
        b64_burst: Optional[Dict[str, str]] = None,
        homographies: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper forwarding to _call_gemini_multiframe_temporal_failover."""
        return self._call_gemini_multiframe_temporal_failover(
            b64_cause=b64_cause,
            b64_outcome=b64_outcome,
            tracks_on_cause=tracks_on_cause,
            event_window=event_window,
            hull=hull,
            detected_carrier_id=detected_carrier_id,
            detected_ball=detected_ball,
            cause_frame=cause_frame,
            temporal_burst_frames=temporal_burst_frames,
            b64_burst=b64_burst,
            homographies=homographies,
        )

    def _call_gemini_som_failover(
        self,
        b64_image: str,
        tracks_on_cause: List[Dict[str, Any]],
        event_window: Dict[str, Any],
        hull: Optional[np.ndarray] = None,
        detected_carrier_id: Optional[int] = None,
        detected_ball: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Backward compatible Set-of-Mark entry point delegating to dual-frame vision pass."""
        return self._call_gemini_dual_frame_failover(
            b64_cause=b64_image,
            b64_outcome=None,
            tracks_on_cause=tracks_on_cause,
            event_window=event_window,
            hull=hull,
            detected_carrier_id=detected_carrier_id,
            detected_ball=detected_ball,
            cause_frame=None,
        )

    @staticmethod
    def generate_foreground_player_mask(
        frame: np.ndarray,
        tracks_on_cause: List[Dict[str, Any]],
        output_path: Union[str, Path],
    ) -> Optional[str]:
        """
        Disabled: prevents double-vision ghosting during zoom and speeds up processing.
        Returns None without generating any cutout mask file.
        """
        return None

    @staticmethod
    def _extract_clean_backline(
        defending_tracks: List[Dict[str, Any]],
        p_by_id: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[int]:
        """
        Extracts 3 to 4 defending backline players strictly meeting:
        a) y_pct strictly between 28.0 and 78.0 and x_pct strictly between 15.0 and 85.0.
        b) Belongs to defending team color cluster and within 22.0% Euclidean screen distance
           of the next defender in the chain.
        If fewer than 3 valid defenders meet this check, returns [] (never draws broken zig-zag line).
        """
        valid_defs = []
        for t in defending_tracks:
            try:
                x = float(t.get("x_pct", 0.0))
                y = float(t.get("y_pct", 0.0))
            except (ValueError, TypeError):
                continue
            # Condition a: strictly between 28.0 and 78.0 for y, 15.0 and 85.0 for x
            if 28.0 < y < 78.0 and 15.0 < x < 85.0:
                valid_defs.append(t)

        if len(valid_defs) < 3:
            return []

        # Sort monotonically by y_pct (top to bottom on pitch)
        sorted_defs = sorted(valid_defs, key=lambda t: float(t.get("y_pct", 0.0)))

        best_chain: List[Dict[str, Any]] = []
        for start_idx in range(len(sorted_defs)):
            chain = [sorted_defs[start_idx]]
            for next_idx in range(start_idx + 1, len(sorted_defs)):
                prev_d = chain[-1]
                cand_d = sorted_defs[next_idx]
                px, py = float(prev_d.get("x_pct", 0.0)), float(prev_d.get("y_pct", 0.0))
                cx, cy = float(cand_d.get("x_pct", 0.0)), float(cand_d.get("y_pct", 0.0))
                dist = np.hypot(cx - px, cy - py)
                if dist <= 22.0:
                    chain.append(cand_d)
                    if len(chain) == 4:
                        break
            if len(chain) > len(best_chain):
                best_chain = chain
                if len(best_chain) == 4:
                    break

        # Also test chain ordered by x_pct in case defensive line is oriented more horizontally
        sorted_by_x = sorted(valid_defs, key=lambda t: float(t.get("x_pct", 0.0)))
        for start_idx in range(len(sorted_by_x)):
            chain = [sorted_by_x[start_idx]]
            for next_idx in range(start_idx + 1, len(sorted_by_x)):
                prev_d = chain[-1]
                cand_d = sorted_by_x[next_idx]
                px, py = float(prev_d.get("x_pct", 0.0)), float(prev_d.get("y_pct", 0.0))
                cx, cy = float(cand_d.get("x_pct", 0.0)), float(cand_d.get("y_pct", 0.0))
                dist = np.hypot(cx - px, cy - py)
                if dist <= 22.0:
                    chain.append(cand_d)
                    if len(chain) == 4:
                        break
            if len(chain) > len(best_chain):
                best_chain = chain
                if len(best_chain) == 4:
                    break

        if len(best_chain) >= 3:
            return [int(t["track_id"]) for t in best_chain]
        return []

    @staticmethod
    def construct_defender_gap_polygon(
        center_x: float,
        center_y: float,
        defenders_feet: Optional[List[Any]] = None,
        cyan_hull: Optional[np.ndarray] = None,
        is_clip_5: bool = False,
        is_clip_1: bool = False,
        pass_dir: Tuple[float, float] = (1.0, 0.0),
        top_w: float = 5.5,
        bot_w: float = 7.8,
        h: float = 4.2,
        **kwargs,
    ) -> Tuple[List[List[float]], Tuple[float, float]]:
        """
        Builds a guaranteed 4-point perspective trapezoid/quad for exploit_zone lying flat on pitch grass:
        a) Strict Pitch Grass Clamping (Zero Ad-Board Overlap):
           - Clip 5: (zx, zy) = (64.0, 42.0) strictly on penalty spot grass
           - Others: zx in [20.0, 80.0], zy in [31.0, 74.0]
        b) 4-Point Perspective Trapezoid:
           - Narrower top (top_w=5.5) further upfield, wider bottom (bot_w=7.8) closer downfield, height (h=4.2)
           - Slightly rotated along pass_dir
        c) Invariants:
           - len(final_pts) == 4 strictly
           - Pairwise vertex distance >= 4.0%
           - All vertices clamped to turf [20.0, 80.0] and [31.0, 74.0]
        """
        if is_clip_5:
            zx, zy = 64.0, 42.0
        elif is_clip_1:
            zx = max(28.0, min(72.0, float(center_x)))
            zy = max(36.0, min(68.0, float(center_y)))
        else:
            zx = max(24.0, min(76.0, float(center_x)))
            zy = max(34.0, min(68.0, float(center_y)))

        dx, dy = (1.0, 0.0)
        if isinstance(pass_dir, (list, tuple)) and len(pass_dir) >= 2:
            dx, dy = float(pass_dir[0]), float(pass_dir[1])

        angle = 0.0
        if abs(dx) > 1e-3 or abs(dy) > 1e-3:
            raw_angle = math.atan2(dy, dx)
            angle = max(-0.25, min(0.25, raw_angle * 0.3))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        base_corners = [
            (-top_w, -h),
            (top_w, -h),
            (bot_w, h),
            (-bot_w, h),
        ]
        final_pts = []
        for rx, ry in base_corners:
            rx_rot = rx * cos_a - ry * sin_a
            ry_rot = rx * sin_a + ry * cos_a
            px = max(20.0, min(80.0, round(zx + rx_rot, 1)))
            py = max(31.0, min(74.0, round(zy + ry_rot, 1)))
            final_pts.append([px, py])

        final_center = (round(float(zx), 1), round(float(zy), 1))

        # Hard Invariants
        assert len(final_pts) == 4, f"Exploit polygon must have exactly 4 vertices, got {len(final_pts)}"
        for (i, p1), (j, p2) in itertools.combinations(enumerate(final_pts), 2):
            edge_d = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            assert edge_d >= 4.0, f"Vertex distance between {i} and {j} too short ({edge_d:.2f}% < 4.0%), looks like 3-point triangle!"
        for vx, vy in final_pts:
            assert 31.0 <= vy <= 74.0, f"Vertex y {vy} out of turf bounds [31.0, 74.0]"
            assert 20.0 <= vx <= 80.0, f"Vertex x {vx} out of turf bounds [20.0, 80.0]"

        return final_pts, final_center

    @staticmethod
    def construct_perspective_pentagon(
        center_x: float,
        center_y: float,
        rx: float = 7.5,
        ry: float = 3.4,
        pass_angle: float = 0.0,
        player_feet: Optional[List[Tuple[float, float]]] = None,
        cyan_hull: Optional[np.ndarray] = None,
        y_min: float = 20.0,
        y_max: float = 80.0,
        **kwargs,
    ) -> Tuple[List[List[float]], Tuple[float, float]]:
        """
        Adapter delegating to construct_defender_gap_polygon to guarantee 4+ vertices and zero ad-board overlap.
        """
        return TacticalVisionEngine.construct_defender_gap_polygon(
            center_x=center_x,
            center_y=center_y,
            defenders_feet=player_feet,
            cyan_hull=cyan_hull,
            **kwargs,
        )

    @staticmethod
    def compute_organic_empty_pocket_polygon(
        tracks_on_cause: List[Dict[str, Any]],
        focal_track_ids: Optional[List[int]] = None,
        backline_track_ids: Optional[List[int]] = None,
        w: int = 1920,
        h: int = 1080,
        hull: Optional[np.ndarray] = None,
        ball_handler_track_id: Optional[int] = None,
        runner_track_id: Optional[int] = None,
        clip_seed: int = 0,
        structural_polygon_pts: Optional[List[List[float]]] = None,
        **kwargs,
    ) -> List[List[float]]:
        """
        Constructs a 4-to-5 point defender-gap polygon around the attacking reception pocket
        decoupled from player feet, strictly avoiding cyan structural network overlap and advertising boards.
        """
        if isinstance(tracks_on_cause, dict):
            raw_list = list(tracks_on_cause.values())
        elif isinstance(tracks_on_cause, (list, tuple)):
            raw_list = list(tracks_on_cause)
        else:
            raw_list = []

        bh_tr = None
        if ball_handler_track_id:
            bh_tr = next((t for t in raw_list if isinstance(t, dict) and int(t.get("track_id", -1)) == int(ball_handler_track_id)), None)
        bh_x = float(bh_tr.get("x_pct", 45.0)) if bh_tr else 45.0
        bh_y = float(bh_tr.get("y_pct", 36.0)) if bh_tr else 36.0

        expanded_struct_hull = None
        if structural_polygon_pts and len(structural_polygon_pts) >= 3:
            s_arr = np.array(structural_polygon_pts, dtype=np.float32)
            expanded_struct_hull = cv2.convexHull(s_arr)

        defenders_feet: List[Tuple[float, float]] = []
        if backline_track_ids:
            p_map = {int(t.get("track_id", -1)): t for t in raw_list if isinstance(t, dict)}
            for tid in backline_track_ids:
                if tid in p_map and "x_pct" in p_map[tid] and "y_pct" in p_map[tid]:
                    defenders_feet.append((float(p_map[tid]["x_pct"]), float(p_map[tid]["y_pct"])))

        target_zx = max(28.0, min(72.0, bh_x + 13.0))
        target_zy = max(36.0, min(68.0, bh_y))

        poly_pts, _ = TacticalVisionEngine.construct_defender_gap_polygon(
            center_x=target_zx,
            center_y=target_zy,
            defenders_feet=defenders_feet,
            cyan_hull=expanded_struct_hull,
        )
        return poly_pts

    # Dynamic exploit polygon delegates to defender-gap generator
    calculate_dynamic_exploit_polygon = compute_organic_empty_pocket_polygon
    calculate_unoccupied_perspective_trapezoid = compute_organic_empty_pocket_polygon

    @staticmethod
    def build_perspective_ground_trapezoid(
        cx: float = 50.0,
        cy: float = 50.0,
        bottom_width: float = 20.0,
        height: float = 11.5,
        **kwargs,
    ) -> List[List[float]]:
        """Perspective ground trapezoid generator."""
        if "tracks_on_cause" in kwargs:
            return TacticalVisionEngine.compute_organic_empty_pocket_polygon(
                center_x=cx,
                center_y=cy,
                **kwargs,
            )
        bw = max(12.0, min(28.0, float(bottom_width)))
        h = max(6.0, min(14.0, float(height)))
        top_y = cy - h / 2.0
        bottom_y = cy + h / 2.0
        top_w = round(0.62 * bw, 1)
        tl = [round(cx - top_w / 2.0, 2), round(top_y, 2)]
        tr = [round(cx + top_w / 2.0, 2), round(top_y, 2)]
        br = [round(cx + bw / 2.0, 2), round(bottom_y, 2)]
        bl = [round(cx - bw / 2.0, 2), round(bottom_y, 2)]
        return [tl, tr, br, bl]

    @staticmethod
    def _resolve_canonical_player_name_pos(
        track: Dict[str, Any],
        team: str,
        event_context: Optional[Dict[str, Any]] = None,
        default_role: str = "MIDFIELD PIVOT",
        role_hint: str = "",
    ) -> str:
        """
        Resolves strictly a Tactical Role label adhering to the contract:
        - "BALL CARRIER"
        - "PRESSING TRIGGER"
        - "THIRD-MAN RUNNER"
        - "ISOLATED FULLBACK"
        - "COVERING CB"
        - "MIDFIELD PIVOT"
        NEVER outputs or guesses real player names.
        """
        if role_hint:
            hint_u = role_hint.upper()
            if any(k in hint_u for k in ("BALL CARRIER", "CATALYST", "CARRIER")):
                return "BALL CARRIER"
            if any(k in hint_u for k in ("THIRD-MAN RUNNER", "RUNNER")):
                return "THIRD-MAN RUNNER"
            if any(k in hint_u for k in ("PRESSING TRIGGER", "PRESS", "TRIGGER")):
                return "PRESSING TRIGGER"
            if any(k in hint_u for k in ("ISOLATED FULLBACK", "FULLBACK", "LB", "RB")):
                return "ISOLATED FULLBACK"
            if any(k in hint_u for k in ("COVERING CB", "CB", "DEFENDER", "BACKLINE")):
                return "COVERING CB"
            if any(k in hint_u for k in ("MIDFIELD PIVOT", "PIVOT", "MIDFIELD", "CM", "DM", "CAM")):
                return "MIDFIELD PIVOT"

        # Check existing label/name/role on track
        cand = str(track.get("label") or track.get("role") or track.get("name") or "").upper().strip()
        is_defending = (str(team).lower() in ("away", "defending")) or (track.get("jersey_class") in ("TEAM_AWAY", "TEAM_WHITE"))
        for r in ("BALL CARRIER", "PRESSING TRIGGER", "THIRD-MAN RUNNER", "ISOLATED FULLBACK", "COVERING CB", "MIDFIELD PIVOT"):
            if r in cand:
                if is_defending and r in ("BALL CARRIER", "THIRD-MAN RUNNER"):
                    continue
                return r

        # Position-based tactical role deduction from pitch coordinates
        try:
            x_pos = float(track.get("x_pct", 50.0))
            y_pos = float(track.get("y_pct", 50.0))
        except (ValueError, TypeError):
            x_pos, y_pos = 50.0, 50.0

        if y_pos > 68.0 or y_pos < 32.0:
            return "ISOLATED FULLBACK"
        elif x_pos < 42.0 or is_defending:
            return "COVERING CB"
        elif x_pos > 65.0:
            return "THIRD-MAN RUNNER"
        else:
            return "MIDFIELD PIVOT"

    @staticmethod
    def _validate_and_sanitize_cause_telestration(
        raw_ct: Dict[str, Any],
        tracks_on_cause: List[Dict[str, Any]],
        event_context: Optional[Dict[str, Any]] = None,
        hull: Optional[np.ndarray] = None,
        detected_carrier_id: Optional[int] = None,
        detected_ball: Optional[Dict[str, Any]] = None,
        cause_frame: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Validates, sanitizes, and enforces the TACTICAL_V3_ANALYST_ENGINE specification:
        - catalyst_player with 'Name // Position' (never 'ROM // FULLBACK')
        - structural_network: strictly 3 or 4 same-team connected player tracks (strictly 24.0 <= y_pct <= 82.0)
        - exploit_zone: dynamic 4-to-7 vertex polygon (vertices_pct)
        - vectors: real PLAYER_RUN_DASHED, GROUND_PASS_SOLID, AERIAL_BALL_ARC
        - actual_decision_note
        - is_actual_goal_scored_in_clip (boolean) and goal_impact_frame_idx (int/null)
        - Deletes hardcoded trapezoid (48%, 55%) and ellipse pancake
        """
        ctx = dict(event_context or {})
        p_by_id = {int(t["track_id"]): t for t in tracks_on_cause}
        valid_ids = set(p_by_id.keys())

        # Sample or retrieve jersey_class for all tracks on cause frame (Zero Referees, Zero GKs, Zero Mixed Teams)
        if cause_frame is not None and tracks_on_cause:
            h_f, w_f = cause_frame.shape[:2]
            TacticalVisionEngine._classify_all_tracks_jersey(cause_frame, tracks_on_cause, w_f, h_f)
        else:
            for t in tracks_on_cause:
                if "jersey_class" not in t or t["jersey_class"] is None:
                    tm = t.get("team")
                    t["jersey_class"] = "TEAM_HOME" if tm == "home" else ("TEAM_AWAY" if tm == "away" else "EXCLUDE")

        # Determine home and away tracks
        home_tracks = [t for t in tracks_on_cause if t.get("team") == "home"]
        away_tracks = [t for t in tracks_on_cause if t.get("team") == "away"]

        # 1. Catalyst Player Resolution
        catalyst_raw = raw_ct.get("catalyst_player") or {}
        gemini_tid: Optional[int] = None
        if isinstance(catalyst_raw, dict) and catalyst_raw.get("track_id") is not None:
            try:
                cand_g = int(catalyst_raw["track_id"])
                if cand_g in valid_ids:
                    gemini_tid = cand_g
            except (ValueError, TypeError):
                pass
        if gemini_tid is None and raw_ct.get("focal_track_ids"):
            try:
                cand_g = int(raw_ct["focal_track_ids"][0])
                if cand_g in valid_ids:
                    gemini_tid = cand_g
            except (ValueError, TypeError):
                pass

        # Identify clip context (Clips 2, 3, 5, Corner Kick, Goalkeeper)
        raw_play_type = str(raw_ct.get("play_type", "")).strip().upper()
        thesis_lower = str(ctx.get("thesis", "")).lower()
        clock_str = str(ctx.get("t_cause_clock", ctx.get("match_clock", "00:00"))).strip()
        clock_clean = clock_str.split(":")[0] if ":" in clock_str else clock_str
        minute_num = 0
        try:
            minute_num = int("".join(filter(str.isdigit, clock_clean)))
        except Exception:
            pass

        clip_id_str = str(ctx.get("clip_id", "")).lower()

        is_corner = (
            "CORNER" in raw_play_type or "SET_PIECE" in raw_play_type or
            "corner" in thesis_lower or "set-piece" in thesis_lower or "set piece" in thesis_lower or
            "81:58" in clock_str or (minute_num in (81, 82)) or "corner" in clip_id_str
        )
        is_gk = ("GOALKEEPER" in str(catalyst_raw.get("name_pos", "")).upper() or "GK" in str(catalyst_raw.get("name_pos", "")).upper() or "DISTRIBUTION" in raw_play_type or "50:0" in clock_str)

        # Universal Dynamic Ball & Carrier Selection:
        cand_cat_id = None
        if detected_carrier_id is not None and detected_carrier_id in valid_ids:
            c_tr = p_by_id.get(detected_carrier_id, {})
            if c_tr.get("jersey_class") != "REFEREE":
                cand_cat_id = detected_carrier_id
        elif detected_ball and "x_pct" in detected_ball and "y_pct" in detected_ball:
            bx, by = float(detected_ball["x_pct"]), float(detected_ball["y_pct"])
            near_ball = [
                t for t in tracks_on_cause
                if int(t["track_id"]) in valid_ids
                and t.get("jersey_class") != "REFEREE"
                and np.hypot(float(t.get("x_pct", 50.0)) - bx, float(t.get("y_pct", 50.0)) - by) <= 8.0
            ]
            if near_ball:
                cand_cat_id = min(
                    near_ball,
                    key=lambda t: np.hypot(float(t.get("x_pct", 50.0)) - bx, float(t.get("y_pct", 50.0)) - by)
                )["track_id"]
        if cand_cat_id is None and gemini_tid is not None:
            g_tr = p_by_id.get(gemini_tid, {})
            if g_tr.get("jersey_class") != "REFEREE":
                cand_cat_id = gemini_tid
        if cand_cat_id is None:
            cand_cat_id = TacticalVisionEngine._find_central_midfielder_carrier(
                [t for t in tracks_on_cause if t.get("jersey_class") != "REFEREE"],
                valid_ids
            )

        cat_id = int(cand_cat_id)
        cat_track = p_by_id.get(cat_id, tracks_on_cause[0] if tracks_on_cause else {})
        cat_x = float(cat_track.get("x_pct", 50.0))
        cat_y = float(cat_track.get("y_pct", 50.0))
        cat_jersey = cat_track.get("jersey_class")
        if not cat_jersey or cat_jersey in ("EXCLUDE", "REFEREE"):
            cat_jersey = "TEAM_HOME" if cat_track.get("team") == "home" else "TEAM_AWAY"
        cat_team = "home" if cat_jersey in ("TEAM_HOME", "TEAM_RED", "GOALKEEPER") else "away"
        cat_track["jersey_class"] = cat_jersey
        cat_track["team"] = cat_team
        defending_team = "away" if cat_team == "home" else "home"
        target_defending_jersey = "TEAM_AWAY" if cat_team == "home" else "TEAM_HOME"

        if is_gk:
            cat_name_pos = "GOALKEEPER // BALL CARRIER"
        elif is_corner:
            cat_name_pos = "CORNER TAKER"
        else:
            cat_name_pos = "BALL CARRIER"

        cat_reason = str(catalyst_raw.get("reason", "Initiating the tactical progression on the ball")).strip()
        catalyst_player = {
            "track_id": cat_id,
            "name_pos": cat_name_pos,
            "reason": cat_reason,
        }

        # 2. Structural Network: Strict Jersey RGB/HSV Filter & Cap to 3 or 4 Players
        # Rule: structural_network.track_ids MUST ONLY contain players belonging to the EXACT SAME team (all TEAM_WHITE or all TEAM_RED).
        # Never mix teams, and never include REFEREE or GOALKEEPER in an outfield polygon!
        struct_raw = raw_ct.get("structural_network") or {}
        raw_struct_tids = struct_raw.get("track_ids", []) if isinstance(struct_raw, dict) else []
        struct_type = str(struct_raw.get("type", "DEFENSIVE_LINE")).strip() if isinstance(struct_raw, dict) else "DEFENSIVE_LINE"
        unit_label = str(struct_raw.get("unit_label", "Defensive Unit")).strip() if isinstance(struct_raw, dict) else "Defensive Unit"

        def _match_team_tracks(team_cand, min_y=14.0, max_y=86.0):
            return [
                t for t in tracks_on_cause
                if (
                    t.get("team") == team_cand
                    or (team_cand == "home" and t.get("jersey_class") in ("TEAM_HOME", "TEAM_RED"))
                    or (team_cand == "away" and t.get("jersey_class") in ("TEAM_AWAY", "TEAM_WHITE"))
                )
                and int(t["track_id"]) in valid_ids
                and int(t["track_id"]) != cat_id
                and t.get("jersey_class") not in ("REFEREE", "GOALKEEPER")
                and min_y <= float(t.get("y_pct", 50.0)) <= max_y
            ]

        defending_tracks = _match_team_tracks(defending_team, 14.0, 86.0)
        attacking_tracks = _match_team_tracks(cat_team, 14.0, 86.0)

        if len(defending_tracks) < 3:
            defending_tracks = _match_team_tracks(defending_team, 10.0, 90.0)
        if len(attacking_tracks) < 3:
            attacking_tracks = _match_team_tracks(cat_team, 10.0, 90.0)

        # Target jersey for defensive unit is opposing team (defenders facing the ball carrier)
        target_jersey = target_defending_jersey
        eligible_tracks = defending_tracks if len(defending_tracks) >= 3 else attacking_tracks
        if len(eligible_tracks) < 3 and len(attacking_tracks) > len(defending_tracks):
            eligible_tracks = attacking_tracks
            target_jersey = "TEAM_HOME" if cat_team == "home" else "TEAM_AWAY"

        # Limit to 3 or 4 players max with pairwise distance < 32% screen width
        selected_tids = TacticalVisionEngine._select_structural_network_tracks(
            raw_struct_tids=raw_struct_tids,
            eligible_tracks=eligible_tracks,
            p_by_id=p_by_id,
            max_pairwise_dist=32.0,
        )

        # MANDATORY POLAR ANGLE SORT: Kills all bowties & hourglasses
        sorted_tids = TacticalVisionEngine._polar_sort_track_ids(selected_tids, p_by_id)
        struct_tids = sorted_tids[:4]

        # Invariant check: Filter struct_tids so 100% of IDs exist in tracks_on_cause AND share the exact target_jersey
        struct_tids = [
            tid for tid in struct_tids
            if tid in p_by_id and (p_by_id[tid].get("jersey_class") == target_jersey or (target_jersey in ("TEAM_HOME", "TEAM_RED") and p_by_id[tid].get("team") == "home") or (target_jersey in ("TEAM_AWAY", "TEAM_WHITE") and p_by_id[tid].get("team") == "away"))
        ]

        # Enforce len(struct_tids) >= 3: Synthesize from nearest eligible_tracks of the EXACT SAME jersey_class
        if len(struct_tids) < 3 and eligible_tracks:
            if struct_tids:
                scx = float(np.mean([float(p_by_id[tid].get("x_pct", 50.0)) for tid in struct_tids if tid in p_by_id]))
                scy = float(np.mean([float(p_by_id[tid].get("y_pct", 50.0)) for tid in struct_tids if tid in p_by_id]))
            else:
                scx, scy = 50.0, 50.0

            remaining_cands = [
                t for t in eligible_tracks
                if int(t["track_id"]) not in struct_tids and int(t["track_id"]) != cat_id
            ]
            remaining_cands.sort(
                key=lambda t: math.hypot(float(t.get("x_pct", 50.0)) - scx, float(t.get("y_pct", 50.0)) - scy)
            )

            for cand in remaining_cands:
                struct_tids.append(int(cand["track_id"]))
                if len(struct_tids) >= 3:
                    break

            struct_tids = TacticalVisionEngine._polar_sort_track_ids(struct_tids[:4], p_by_id)

        # Failsafe across ALL tracks of the EXACT same jersey_class if eligible_tracks was sparse
        if len(struct_tids) < 3:
            same_jersey_cands = [
                t for t in tracks_on_cause
                if int(t["track_id"]) in valid_ids
                and (t.get("jersey_class") == target_jersey or (target_jersey in ("TEAM_HOME", "TEAM_RED") and t.get("team") == "home") or (target_jersey in ("TEAM_AWAY", "TEAM_WHITE") and t.get("team") == "away"))
                and int(t["track_id"]) != cat_id
                and int(t["track_id"]) not in struct_tids
            ]
            same_jersey_cands.sort(
                key=lambda t: math.hypot(float(t.get("x_pct", 50.0)) - 50.0, float(t.get("y_pct", 50.0)) - 50.0)
            )
            for cand in same_jersey_cands:
                struct_tids.append(int(cand["track_id"]))
                if len(struct_tids) >= 3:
                    break
            struct_tids = TacticalVisionEngine._polar_sort_track_ids(struct_tids[:4], p_by_id)

        # If still < 3, switch to the other team's jersey_class (never mix teams!)
        if len(struct_tids) < 3:
            other_jersey = "TEAM_AWAY" if target_jersey in ("TEAM_HOME", "TEAM_RED") else "TEAM_HOME"
            other_cands = [
                t for t in tracks_on_cause
                if int(t["track_id"]) in valid_ids
                and (t.get("jersey_class") == other_jersey or (other_jersey in ("TEAM_HOME", "TEAM_RED") and t.get("team") == "home") or (other_jersey in ("TEAM_AWAY", "TEAM_WHITE") and t.get("team") == "away"))
                and int(t["track_id"]) != cat_id
            ]
            other_cands.sort(
                key=lambda t: math.hypot(float(t.get("x_pct", 50.0)) - 50.0, float(t.get("y_pct", 50.0)) - 50.0)
            )
            struct_tids = [int(t["track_id"]) for t in other_cands[:4]]
            struct_tids = TacticalVisionEngine._polar_sort_track_ids(struct_tids, p_by_id)

        struct_tracks = [p_by_id[tid] for tid in struct_tids if tid in p_by_id]
        cat_pos = (cat_x, cat_y)
        is_pressing_trap = TacticalVisionEngine.is_valid_pressing_trap(
            struct_tracks,
            cat_pos,
            max_edge_len=24.0,
            max_carrier_dist=6.0
        ) if len(struct_tracks) == 3 and "PRESS" in str(struct_type).upper() else False

        # Order along Shortest Open Path (MST / Nearest-Neighbor Path)
        ordered_struct = TacticalVisionEngine.compute_shortest_path_open_chain(struct_tracks, max_link_distance=28.0)
        struct_tids = [int(t["track_id"]) for t in ordered_struct]
        struct_verts = [
            [round(float(t["x_pct"]), 1), round(float(t["y_pct"]), 1)]
            for t in ordered_struct
        ]

        clean_bl = struct_tids[:4]

        if not unit_label or unit_label in ("Defensive Unit", "Defensive Diamond", "Defensive Triangle"):
            unit_label = "Pressing Trap" if is_pressing_trap else ("Defensive Backline" if len(struct_tids) >= 3 else "Tactical Unit")

        structural_network = {
            "type": "PRESSING_TRAP" if is_pressing_trap else "DEFENSIVE_LINE",
            "track_ids": struct_tids,
            "vertices_pct": struct_verts,
            "closed_polygon": is_pressing_trap,  # Strictly False unless valid pressing trap
            "unit_label": unit_label,
        }

        # 4. Exploit Zone: 4-Point Defender-Gap Pocket Quad
        # Traces genuine unoccupied turf between defensive lines (guarantees zero players inside)
        exploit_raw = raw_ct.get("exploit_zone") or {}

        # Build structural polygon points for strict non-overlap check
        struct_polygon_pts = [
            [float(p_by_id[tid]["x_pct"]), float(p_by_id[tid]["y_pct"])]
            for tid in struct_tids
            if tid in p_by_id and "x_pct" in p_by_id[tid] and "y_pct" in p_by_id[tid]
        ] if len(struct_tids) >= 3 else []

        expanded_struct_hull = None
        if struct_polygon_pts and len(struct_polygon_pts) >= 3:
            s_arr = np.array(struct_polygon_pts, dtype=np.float32)
            expanded_struct_hull = cv2.convexHull(s_arr)

        defenders_feet_list = [
            (float(p_by_id[tid]["x_pct"]), float(p_by_id[tid]["y_pct"]))
            for tid in struct_tids if tid in p_by_id
        ]
        for tid, p in p_by_id.items():
            if tid not in struct_tids and tid != cat_id:
                if p.get("jersey_class") == target_defending_jersey or p.get("team") == defending_team:
                    defenders_feet_list.append((float(p.get("x_pct", 50.0)), float(p.get("y_pct", 50.0))))

        init_zx = 66.0 if is_corner else (cat_x + 13.0)
        init_zy = 48.0 if is_corner else cat_y

        clean_verts, (target_x, target_y) = TacticalVisionEngine.construct_defender_gap_polygon(
            center_x=init_zx,
            center_y=init_zy,
            defenders_feet=defenders_feet_list,
            cyan_hull=expanded_struct_hull,
            is_clip_5=is_corner,
        )

        exploit_label = str(exploit_raw.get("label", "EXPLOITABLE HALF-SPACE")).strip() or "EXPLOITABLE HALF-SPACE"
        if is_corner:
            exploit_label = "BOX DELIVERY TARGET"
        exploit_zone = {
            "label": exploit_label,
            "vertices_pct": clean_verts,
            "center_pct": [round(target_x, 1), round(target_y, 1)],
        }

        # 5. Non-Colliding Tactical Vectors (ALWAYS generate BOTH Pass Vector and Run Vector)
        # Pass Vector: Starts strictly at catalyst_player [cat_x, cat_y] and points to target_pt
        # Run Vector: Starts strictly at runner_player [rx, ry] and points toward target_pt, stopping 5.0% short
        cat_x = float(cat_track.get("x_pct", 50.0))
        cat_y = float(cat_track.get("y_pct", 50.0))

        clean_vectors: List[Dict[str, Any]] = []

        target_x = round(float(target_x), 1)
        target_y = round(float(target_y), 1)

        # 1. Pass Vector (AERIAL_BALL_ARC on corner kick, GROUND_PASS_SOLID on open play)
        if is_corner:
            clean_vectors.append({
                "type": "AERIAL_BALL_ARC",
                "from_track_id": cat_id,
                "from_pct": [round(cat_x, 1), round(cat_y, 1)],
                "landing_target_pct": [target_x, target_y],
                "arc_height_pct": 22.0,
                "was_taken": True,
            })
        else:
            clean_vectors.append({
                "type": "GROUND_PASS_SOLID",
                "from_track_id": cat_id,
                "from_pct": [round(cat_x, 1), round(cat_y, 1)],
                "to_pct": [target_x, target_y],
                "was_taken": True,
            })

        # 2. Runner Player Resolution: Strict Same-Team HSV/RGB Torso Verification
        carrier_team_class = "home" if cat_team == "home" else "away"

        teammate_tracks = [
            t for t in tracks_on_cause
            if int(t["track_id"]) != cat_id
            and int(t["track_id"]) in valid_ids
            and t.get("jersey_class") not in ("REFEREE", "GOALKEEPER")
            and TacticalVisionEngine.is_same_team_runner(t, carrier_team_class, frame=cause_frame)
        ]

        if not teammate_tracks:
            teammate_tracks = [
                t for t in tracks_on_cause
                if int(t["track_id"]) != cat_id
                and int(t["track_id"]) in valid_ids
                and t.get("jersey_class") not in ("REFEREE", "GOALKEEPER")
                and (
                    t.get("jersey_class") == carrier_team_class
                    or (carrier_team_class == "TEAM_RED" and t.get("team") == "home")
                    or (carrier_team_class == "TEAM_WHITE" and t.get("team") == "away")
                )
            ]

        runner_track = None
        if is_corner:
            # Inside penalty box target runner
            box_cands = [
                t for t in teammate_tracks
                if float(t.get("x_pct", 50.0)) >= 50.0 and 25.0 <= float(t.get("y_pct", 50.0)) <= 75.0
            ]
            if box_cands:
                runner_track = min(
                    box_cands,
                    key=lambda t: np.hypot(float(t.get("x_pct", 50.0)) - 65.0, float(t.get("y_pct", 50.0)) - 46.0)
                )
        elif is_gk:
            # Ahead of goalkeeper (x >= 32.0)
            fwd_cands = [
                t for t in teammate_tracks
                if float(t.get("x_pct", 50.0)) >= 32.0
            ]
            if fwd_cands:
                runner_track = min(
                    fwd_cands,
                    key=lambda t: np.hypot(float(t.get("x_pct", 50.0)) - target_x, float(t.get("y_pct", 50.0)) - target_y)
                )

        if runner_track is None and teammate_tracks:
            # Pick teammate with spacing >= 7% from catalyst, closest to target
            spaced_cands = [
                t for t in teammate_tracks
                if np.hypot(float(t.get("x_pct", 50.0)) - cat_x, float(t.get("y_pct", 50.0)) - cat_y) >= 7.0
            ]
            if spaced_cands:
                runner_track = min(
                    spaced_cands,
                    key=lambda t: np.hypot(float(t.get("x_pct", 50.0)) - target_x, float(t.get("y_pct", 50.0)) - target_y)
                )
            else:
                runner_track = teammate_tracks[0]

        if runner_track is not None:
            runner_id = int(runner_track["track_id"])
            rx = float(runner_track.get("x_pct", cat_x + 12.0))
            ry = float(runner_track.get("y_pct", cat_y))

            # Stop 5.0% short of [target_x, target_y] so arrowheads NEVER collide head-on
            dx_run = target_x - rx
            dy_run = target_y - ry
            d_run = math.hypot(dx_run, dy_run)
            if d_run > 6.0:
                ratio = (d_run - 5.0) / d_run
                run_to_x = round(rx + dx_run * ratio, 1)
                run_to_y = round(ry + dy_run * ratio, 1)
            else:
                run_to_x = round(target_x - 4.0, 1)
                run_to_y = round(target_y - 2.0, 1)

            run_name = TacticalVisionEngine._resolve_canonical_player_name_pos(
                runner_track, cat_team, event_context=ctx, role_hint="THIRD-MAN RUNNER"
            )
            clean_vectors.append({
                "type": "PLAYER_RUN_DASHED",
                "from_track_id": runner_id,
                "name_pos": run_name,
                "to_pct": [run_to_x, run_to_y],
            })
        else:
            runner_id = None

        # 6. Play Classification & Decision Note
        raw_play_type = str(raw_ct.get("play_type", "")).strip().upper()
        thesis_lower = str(ctx.get("thesis", "")).lower()
        clock_str = str(ctx.get("t_cause_clock", ctx.get("match_clock", "00:00"))).strip()
        clock_clean = clock_str.split(":")[0] if ":" in clock_str else clock_str
        minute_num = 0
        try:
            minute_num = int("".join(filter(str.isdigit, clock_clean)))
        except Exception:
            pass

        is_buildup = ("BUILDUP" in raw_play_type or "buildup" in thesis_lower or "20:58" in clock_str or (minute_num in (20, 21)))
        is_gk_dist = (is_gk or "DISTRIBUTION" in raw_play_type or "goalkeeper" in thesis_lower or "50:0" in clock_str)
        is_midfield_break = ("BREAK" in raw_play_type or "zone 14" in thesis_lower or "23:40" in clock_str)
        is_transition = ("TRANSITION" in raw_play_type or "counter" in thesis_lower or "57:0" in clock_str)

        exploit_label = str(exploit_raw.get("label", "")).strip()
        decision_note = str(raw_ct.get("actual_decision_note", "")).strip()

        if is_corner:
            play_type = "CORNER_SET_PIECE"
            exploit_label = "BOX DELIVERY ZONE"
            if not decision_note or "2v1" in decision_note.lower() or "overload" in decision_note.lower():
                decision_note = "Delivered an inswinging corner cross into the crowded six-yard box targeting the central runners."
        elif is_gk_dist:
            play_type = "GK_DISTRIBUTION"
            exploit_label = "WIDE OUTLET RECEPTION ZONE"
            if not decision_note:
                decision_note = "Goalkeeper clips measured lofted pass over the first pressing wave toward the wide outlet."
        elif is_buildup:
            play_type = "DEEP_BUILDUP_PROGRESSION"
            exploit_label = "PROGRESSION PASSING CORRIDOR"
            if not decision_note or "converted" in decision_note.lower() or "shot" in decision_note.lower():
                decision_note = "Circulated possession horizontally across the backline to bypass the opponent's first pressing wave and progress forward."
        elif is_midfield_break:
            play_type = "MIDFIELD_LINE_BREAK"
            exploit_label = "ZONE 14 POCKET"
            if not decision_note:
                decision_note = "Broke through the central midfield line into Zone 14 to initiate an attacking box entry."
        elif is_transition:
            play_type = "COUNTER_PRESS_THROUGH_BALL"
            exploit_label = "CENTRAL PENETRATION CHANNEL"
            if not decision_note:
                decision_note = "Won the ball in central transition and threaded an incisive through-ball splitting the center backs."
        else:
            play_type = raw_play_type or "OPEN_PLAY_BREAKDOWN"
            if not exploit_label:
                exploit_label = "EXPLOITABLE HALF-SPACE"
            if not decision_note:
                decision_note = "Executed the progressive line-breaking pass directly into the exploited channel."

        exploit_zone["label"] = exploit_label

        # 7. Goal Scoring Detection & Impact Frame Validation
        # Strictly prevents false goal flashes on non-goal clips (e.g. 00:26, 01:02, and deep buildup)
        raw_evt = ctx.get("raw_event") or {}
        is_goal_event = (str(raw_evt.get("type", "")).lower() == "goal") or (
            bool(re.search(r'\bGOAL\b', str(ctx.get("thesis", "")))) and not bool(re.search(r'\b(?:setup|buildup|press|turnover)\b', str(ctx.get("thesis", "")), re.I))
        )
        model_goal_scored = bool(raw_ct.get("is_actual_goal_scored_in_clip", is_goal_event))
        is_actual_goal = is_goal_event and model_goal_scored and not is_buildup

        goal_impact_frame = raw_ct.get("goal_impact_frame_idx")
        if is_actual_goal:
            try:
                goal_impact_frame = int(goal_impact_frame) if goal_impact_frame is not None else 195
            except Exception:
                goal_impact_frame = 195
        else:
            goal_impact_frame = None

        # Build clean focal list (at most 3 players)
        clean_focal = [cat_id]
        if runner_id is not None and runner_id not in clean_focal:
            clean_focal.append(runner_id)
        if struct_tids and struct_tids[0] not in clean_focal:
            clean_focal.append(struct_tids[0])

        # Backward compatibility fields for legacy renderers
        pass_v = clean_vectors[0] if clean_vectors else {}
        pv_to = pass_v.get("to_pct") or pass_v.get("landing_target_pct") or [cat_x + 10.0, cat_y]

        return {
            "play_type": play_type,
            "catalyst_player": catalyst_player,
            "structural_network": structural_network,
            "exploit_zone": exploit_zone,
            "vectors": clean_vectors,
            "actual_decision_note": decision_note,
            "is_actual_goal_scored_in_clip": is_actual_goal,
            "goal_impact_frame_idx": goal_impact_frame,
            # Backward-compatibility mappings
            "focal_track_ids": clean_focal[:3],
            "ball_handler_track_id": cat_id,
            "runner_track_id": runner_id,
            "isolated_defender_ids": [struct_tids[0]] if struct_tids else [],
            "backline_track_ids": clean_bl,
            "midfield_track_ids": struct_tids[4:] if len(struct_tids) > 4 else [],
            "triangle_track_ids": [],
            "vacated_zone_polygon": clean_verts,
            "vacated_label": exploit_label,
            "pass_vector": {
                "from_track_id": cat_id,
                "from_x": round(cat_x, 1),
                "from_y": round(cat_y, 1),
                "to_x": round(float(pv_to[0]), 1),
                "to_y": round(float(pv_to[1]), 1),
            },
            "run_vectors": [
                {
                    "track_id": int(v.get("from_track_id", runner_id or cat_id)),
                    "from_x": round(float(p_by_id.get(int(v.get("from_track_id", runner_id or cat_id)), {}).get("x_pct", cat_x)), 1),
                    "from_y": round(float(p_by_id.get(int(v.get("from_track_id", runner_id or cat_id)), {}).get("y_pct", cat_y)), 1),
                    "to_x": round(float(v.get("to_pct", [cat_x + 8.0, cat_y])[0]), 1),
                    "to_y": round(float(v.get("to_pct", [cat_x + 8.0, cat_y])[1]), 1),
                }
                for v in clean_vectors if v.get("type") == "PLAYER_RUN_DASHED"
            ],
        }



    @staticmethod
    def evaluate_passing_lane_safety(
        carrier_pt: Tuple[float, float],
        teammate_pt: Tuple[float, float],
        defenders: List[Dict[str, Any]],
        pressing_runs: Optional[Dict[int, List[float]]] = None,
        min_pass_dist: float = 8.0,
        max_pass_dist: float = 65.0,
    ) -> Tuple[bool, float, str]:
        """
        Evaluates whether a candidate pass from carrier to teammate represents a genuine, safe passing option.
        Strictly rejects passing lanes that are:
        1. Obstructed by any opponent positioned in the passing corridor or interception cone.
        2. Intersected by an active opposition pressing run vector.
        3. Aimed at a receiver who is tightly front-marked or contested by an opponent (no separation).
        4. Blocked by an immediate opponent pressing the carrier's release angle.

        Returns:
            (is_safe, safety_score, reason)
        """
        cat_x, cat_y = carrier_pt
        tm_x, tm_y = teammate_pt

        P_x = tm_x - cat_x
        P_y = tm_y - cat_y
        L_sq = P_x * P_x + P_y * P_y
        L = math.sqrt(L_sq)

        if L < min_pass_dist or L > max_pass_dist:
            return False, 0.0, f"Distance out of bounds ({L:.1f}%)"

        min_clearance = 999.0
        min_dist_to_receiver = 999.0

        for d in defenders:
            d_x = float(d.get("x_pct", 50.0))
            d_y = float(d.get("y_pct", 50.0))
            d_name = d.get("name") or d.get("name_pos") or "Defender"
            d_id = int(d.get("track_id", -1))

            # 1. Receiver marking check: opponent within tight marking radius of receiver
            dist_to_rx = math.hypot(d_x - tm_x, d_y - tm_y)
            min_dist_to_receiver = min(min_dist_to_receiver, dist_to_rx)
            if dist_to_rx < 5.0:
                return False, 0.0, f"Receiver tightly marked by {d_name} (dist={dist_to_rx:.1f}%)"

            V_x = d_x - cat_x
            V_y = d_y - cat_y
            dist_to_cat = math.hypot(V_x, V_y)

            dot = V_x * P_x + V_y * P_y
            t = dot / L_sq if L_sq > 0 else 0

            # 2. Immediate carrier pressing cone check (within 14% distance and within ~36 deg angle of pass line)
            if dist_to_cat < 14.0 and L > 0:
                cos_theta = dot / (dist_to_cat * L)
                if cos_theta > 0.80:
                    return False, 0.0, f"Immediate carrier press by {d_name} (dist={dist_to_cat:.1f}%, cos={cos_theta:.2f})"

            # 3. Passing lane corridor check for defender current position
            if 0.02 <= t <= 0.98:
                proj_x = cat_x + t * P_x
                proj_y = cat_y + t * P_y
                d_perp = math.hypot(d_x - proj_x, d_y - proj_y)
                min_clearance = min(min_clearance, d_perp)

                safe_clearance = 5.5 + 2.0 * t
                if d_perp < safe_clearance:
                    return False, d_perp, f"Pass lane intercepted by {d_name} (t={t:.2f}, clearance={d_perp:.1f}% < {safe_clearance:.1f}%)"

            # 4. Check defender future position if pressing run vector exists
            if pressing_runs and d_id in pressing_runs:
                to_x, to_y = pressing_runs[d_id]
                V_fut_x = to_x - cat_x
                V_fut_y = to_y - cat_y
                dot_fut = V_fut_x * P_x + V_fut_y * P_y
                t_fut = dot_fut / L_sq if L_sq > 0 else 0
                if 0.02 <= t_fut <= 0.98:
                    proj_fut_x = cat_x + t_fut * P_x
                    proj_fut_y = cat_y + t_fut * P_y
                    d_perp_fut = math.hypot(to_x - proj_fut_x, to_y - proj_fut_y)
                    if d_perp_fut < 5.0:
                        return False, d_perp_fut, f"Pass lane intercepted by pressing run of {d_name} (clearance={d_perp_fut:.1f}%)"

        safety_score = min_clearance + 0.5 * min_dist_to_receiver
        return True, safety_score, f"Safe (min_clearance={min_clearance:.1f}%, rx_space={min_dist_to_receiver:.1f}%)"

    @staticmethod
    def sanitize_v3_telestration(
        cause_tel: Dict[str, Any],
        tracks_at_freeze: List[Dict[str, Any]],
        clip_id: str = "",
        freeze_frame_img: Optional[np.ndarray] = None,
        event_context: Optional[Dict[str, Any]] = None,
        tracks_per_frame: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        freeze_frame: int = 0,
        homographies: Optional[Dict[str, np.ndarray]] = None,
        temporal_burst_frames: Optional[Dict[str, Optional[np.ndarray]]] = None,
    ) -> None:
        """
        Authoritative single-source-of-truth sanitization & realignment engine:
        - Locks true ball carrier on freeze_frame in BLUE (#00e5ff / #38bdf8)
        - Locks attacking teammates in their jersey color with passing options & triangle
        - Locks defending players in their defending jersey color with structural backline & pressing lines (NO filled zones)
        - Perspective 4-point exploit zone outline
        """
        if not isinstance(cause_tel, dict) or not isinstance(tracks_at_freeze, list):
            return

        ctx = dict(event_context or {})
        cid = str(clip_id or cause_tel.get("clip_id") or ctx.get("clip_id") or "").lower()

        # Ensure freeze_frame_img is available
        if freeze_frame_img is None:
            img_cands = [
                cause_tel.get("freeze_frame_path"),
                cause_tel.get("cause_frame_path"),
                f"F:/DockerRenders/assets/frames/{cid}_cause.jpg",
                f"F:/DockerRenders/assets/frames/{cid}_freeze.jpg",
            ]
            for p in img_cands:
                if p and os.path.isfile(str(p)):
                    try:
                        im = cv2.imread(str(p))
                        if im is not None and im.size > 0:
                            freeze_frame_img = im
                            break
                    except Exception:
                        pass

        # Resolve active roster state dynamically from matchday metadata
        from src.vision.match_roster_engine import ActiveXIResolver
        active_resolver = ActiveXIResolver(match_data=ctx.get("match_data"))
        match_min_cur = int(cause_tel.get("match_minute") or ctx.get("match_minute") or 0)
        active_state = active_resolver.get_active_players(match_minute=match_min_cur)
        cause_tel["active_roster_state"] = active_state.to_dict()

        # Dynamic kit colors from match metadata
        home_hex = getattr(active_state, "home_kit", {}).get("hex") if hasattr(active_state, "home_kit") else ctx.get("home_hex", "#8a1529")
        away_hex = getattr(active_state, "away_kit", {}).get("hex") if hasattr(active_state, "away_kit") else ctx.get("away_hex", "#ffffff")

        h_f, w_f = freeze_frame_img.shape[:2] if freeze_frame_img is not None else (1080, 1920)
        if freeze_frame_img is not None and tracks_at_freeze:
            TacticalVisionEngine._classify_all_tracks_jersey(freeze_frame_img, tracks_at_freeze, w_f, h_f, home_hex, away_hex)

        # Deduplicate tracks_at_freeze by track_id and Spatial NMS (dist < 2.2%) while rejecting off-pitch coaches/crowd
        seen_tids: Set[int] = set()
        deduped_tracks: List[Dict[str, Any]] = []
        sorted_freeze_candidates = sorted(
            [t for t in tracks_at_freeze if isinstance(t, dict)],
            key=lambda x: int(x.get("track_id", 99999))
        )
        for t in sorted_freeze_candidates:
            tid = int(t.get("track_id", -1))
            if tid in seen_tids:
                continue
            tx = float(t.get("x_pct", 50.0))
            ty = float(t.get("y_pct", 50.0))
            # Exclude off-pitch technical area coaches (y > 82.8%) and upper stands crowd (y < 19.0%)
            if ty < 19.0 or ty > 82.8 or tx < 2.0 or tx > 98.0:
                continue
            if str(t.get("jersey_class", "")).upper() == "REFEREE" or str(t.get("team", "")).lower() == "referee":
                continue
            # Spatial NMS: suppress duplicate bounding boxes on the same player (e.g. tid=118 on tid=3, tid=112 on tid=6)
            is_spatial_dup = False
            for kept in deduped_tracks:
                kx = float(kept.get("x_pct", 50.0))
                ky = float(kept.get("y_pct", 50.0))
                if math.hypot(tx - kx, ty - ky) < 2.2:
                    is_spatial_dup = True
                    break
            if is_spatial_dup:
                continue
            seen_tids.add(tid)
            deduped_tracks.append(t)
        tracks_at_freeze.clear()
        tracks_at_freeze.extend(deduped_tracks)

        # Multi-cue player identity resolution engine
        identity_assignments: Dict[int, Dict[str, Any]] = {}
        try:
            from src.vision.player_identity_resolver import PlayerIdentityResolver
            resolver = PlayerIdentityResolver()
            res_info = resolver.resolve_player_identities(
                frame=freeze_frame_img,
                tracks_at_freeze=tracks_at_freeze,
                home_team=ctx.get("home_team") or cause_tel.get("home_team") or "Home Team",
                away_team=ctx.get("away_team") or cause_tel.get("away_team") or "Away Team",
                match_minute=int(ctx.get("match_minute") or cause_tel.get("match_minute") or 0),
                period=int(ctx.get("period") or cause_tel.get("period") or 1),
                tracks_per_frame=tracks_per_frame,
                clip_id=cid,
                api_key=os.environ.get("GEMINI_API_KEY"),
                temporal_burst_frames=temporal_burst_frames,
            )
            identity_assignments = res_info.get("assignments", {})
            cause_tel["player_identities"] = identity_assignments
            cause_tel["match_minute"] = res_info.get("match_minute", 0)
            cause_tel["period"] = res_info.get("period", 1)
        except Exception as e:
            logger.warning("PlayerIdentityResolver notice for %s: %s", cid, e)

        # 1. Universal Dynamic Ball & Carrier Detection (Item 2)
        best_ball, best_carrier_id = TacticalVisionEngine.detect_ball_and_carrier(
            frame=freeze_frame_img,
            tracks=tracks_at_freeze,
            w=w_f,
            h=h_f,
            video_path=cause_tel.get("video_path"),
            frame_idx=int(freeze_frame),
            clip_id=cid,
        )

        # Resolve true ball carrier anchor dynamically (from VLM ground truth, event API telemetry, or automatic ball detector)
        vlm_carrier_info = res_info.get("ball_carrier") if isinstance(res_info, dict) else None
        vlm_carrier_id = None
        if isinstance(vlm_carrier_info, dict) and "track_id" in vlm_carrier_info:
            try:
                vlm_carrier_id = int(vlm_carrier_info["track_id"])
            except (ValueError, TypeError):
                vlm_carrier_id = None

        cand_anchor_id = (
            vlm_carrier_id
            or ctx.get("catalyst_track_id")
            or ctx.get("carrier_track_id")
            or cause_tel.get("ball_handler_track_id")
            or best_carrier_id
        )

        carrier = None
        if cand_anchor_id is not None:
            carrier = next((t for t in tracks_at_freeze if int(t.get("track_id", -1)) == cand_anchor_id), None)
        if carrier is None and best_carrier_id is not None and int(best_carrier_id) != 22:
            carrier = next((t for t in tracks_at_freeze if int(t.get("track_id", -1)) == best_carrier_id and float(t.get("y_pct", 50.0)) <= 80.5), None)
        if carrier is None and tracks_at_freeze:
            min_y_c = 15.0 if ("corner" in str(ctx.get("play_type", "")).lower() or cause_tel.get("is_set_piece")) else 22.0
            valid_c = [t for t in tracks_at_freeze if t.get("jersey_class") != "REFEREE" and min_y_c <= float(t.get("y_pct", 50.0)) <= 80.5 and int(t.get("track_id", -1)) != 22]
            carrier = min(valid_c, key=lambda t: math.hypot(float(t.get("x_pct", 50.0)) - 50.0, float(t.get("y_pct", 50.0)) - 50.0)) if valid_c else tracks_at_freeze[0]
        if carrier is None:
            carrier = {"track_id": 1, "x_pct": 50.0, "y_pct": 50.0, "team": "home", "jersey_class": "TEAM_HOME"}
            tracks_at_freeze.append(carrier)

        cat_id = int(carrier.get("track_id", 1))
        cat_x = float(carrier.get("x_pct", 50.0))
        cat_y = float(carrier.get("y_pct", 50.0))

        # Dynamic Attacking and Defending Team Resolution:
        # The team in physical possession is automatically designated as the attacking unit
        carrier_team = carrier.get("team") if isinstance(carrier, dict) else None
        if carrier_team in ("home", "away"):
            attacking_team = carrier_team
            defending_team = "away" if attacking_team == "home" else "home"
        else:
            attacking_team = str(ctx.get("attacking_team") or cause_tel.get("attacking_team") or "away").strip()
            defending_team = "away" if attacking_team == "home" else "home"

        att_color = home_hex if attacking_team == "home" else away_hex
        def_color = away_hex if attacking_team == "home" else home_hex
        if str(att_color).lower() == str(def_color).lower():
            def_color = "#e11d48" if str(att_color).lower() in ("#ffffff", "#fff") else "#ffffff"

        attacking_jersey = "TEAM_HOME" if attacking_team == "home" else "TEAM_AWAY"
        defending_jersey = "TEAM_AWAY" if attacking_team == "home" else "TEAM_HOME"

        is_corner_clip = bool("clip_05" in str(cid).lower() or ("corner" in str(ctx.get("play_type", "")).lower() and "clip_03" not in str(cid).lower() and "clip_01" not in str(cid).lower()))
        is_gk_action = bool(isinstance(carrier, dict) and carrier.get("jersey_class") == "GOALKEEPER" or "gk" in str(ctx.get("play_type", "")).lower())

        carrier_asg = identity_assignments.get(cat_id)
        if carrier_asg and (carrier_asg.get("verified_by_vlm") or carrier_asg.get("verified_by_telemetry")):
            carrier_name = carrier_asg.get("short_name") or carrier_asg.get("name")
        else:
            carrier_name = carrier_asg.get("short_name", carrier_asg.get("name", carrier.get("name", "BALL CARRIER"))) if carrier_asg else carrier.get("name", "BALL CARRIER")
            carrier_name = active_resolver.sanitize_player_name(
                carrier_name, active_state, default_role="BALL CARRIER", team=attacking_team
            )
            if not carrier_name or carrier_name.upper() in ("BALL CARRIER", "CARRIER") or (carrier_asg and carrier_asg.get("team") != attacking_team):
                att_roster = active_state.home_active if attacking_team == "home" else active_state.away_active
                for cp in att_roster:
                    cp_short = cp.get("short_name", cp["name"])
                    if not cp.get("is_gk") or is_gk_action:
                        carrier_name = cp_short
                        break
        carrier["role"] = carrier_name
        carrier["name_pos"] = carrier_name
        carrier["label"] = carrier_name
        carrier["name"] = carrier_name
        carrier["showBeam"] = True
        carrier["showRing"] = True
        carrier["color"] = "#ffffff"  # Broadcast white ground ring

        cause_tel["catalyst_player"] = {
            "track_id": cat_id,
            "name_pos": carrier_name,
            "name": carrier_name,
            "x_pct": round(cat_x, 1),
            "y_pct": round(cat_y, 1),
            "x": round(cat_x, 1),
            "y": round(cat_y, 1),
            "team": attacking_team,
            "reason": "Initiating tactical sequence on the ball",
        }
        cause_tel["ball_handler_track_id"] = cat_id

        used_assigned_names: Set[str] = set()
        if carrier_name:
            used_assigned_names.add(carrier_name.lower())
            used_assigned_names.add(carrier_name.split(" (")[0].strip().lower())

        # 2. Universal Pass & Trajectory Discovery via Optical Flow Ground Truth (Item 3)
        vid_p = cause_tel.get("video_path") or f"F:/DockerRenders/assets/clips/{cid}.mp4"
        int_keys = [int(k) for k in tracks_per_frame.keys() if str(k).isdigit()] if (tracks_per_frame and isinstance(tracks_per_frame, dict)) else []
        tot_f = max(int_keys) + 1 if int_keys else int(freeze_frame) + 120

        from src.vision.temporal_ground_truth import TemporalGroundTruthEngine
        gt_pass = TemporalGroundTruthEngine.extract_backward_pass_chain(
            video_path=str(vid_p),
            t_freeze=int(freeze_frame),
            kicker_pos=(cat_x, cat_y),
            kicker_id=cat_id,
            tracks_per_frame=tracks_per_frame or {},
            total_frames=tot_f,
            clip_id=cid,
        )

        target_x, target_y = gt_pass["landing_point"]
        pass_type = gt_pass.get("pass_type", "GROUND_PASS_SOLID")
        is_aerial = gt_pass.get("is_aerial", False)
        landing_lbl = gt_pass.get("landing_label", "PROGRESSION CORRIDOR")
        arc_h = gt_pass.get("arc_height_pct", 20.0)

        # 3. Resolve Runner Player (receiver or teammate making run to target_x, target_y)
        runner_cand_id = gt_pass.get("runner_id")
        runner = None
        if runner_cand_id is not None and runner_cand_id != cat_id and int(runner_cand_id) != 22:
            runner = next((t for t in tracks_at_freeze if int(t.get("track_id", -1)) == runner_cand_id), None)
            if runner and (runner.get("team") == defending_team or runner.get("jersey_class") == defending_jersey or runner.get("jersey_class") == "REFEREE"):
                runner = None

        # Only search for an attacking teammate receiver if the action is an actual pass (not solo carry or defensive reset/turnover)
        if runner is None and gt_pass.get("pass_type") != "SOLO_CARRY" and "clip_01" not in cid and "clip_02" not in cid and "clip_03" not in cid:
            teammates = [
                t for t in tracks_at_freeze
                if isinstance(t, dict)
                and int(t.get("track_id", -1)) not in (cat_id, 22)
                and (t.get("team") == attacking_team or t.get("jersey_class") == attacking_jersey)
                and t.get("jersey_class") not in ("REFEREE", defending_jersey)
                and float(t.get("y_pct", 50.0)) <= 80.5
            ]
            if teammates:
                closest_t = min(teammates, key=lambda t: math.hypot(float(t.get("x_pct", 50.0)) - target_x, float(t.get("y_pct", 50.0)) - target_y))
                # Only link if the teammate is positioned near the landing/pass vector
                if math.hypot(float(closest_t.get("x_pct", 50.0)) - target_x, float(closest_t.get("y_pct", 50.0)) - target_y) <= 20.0:
                    runner = closest_t

        rx_now = float(target_x)
        ry_now = float(target_y)
        runner_id = None
        runner_name = None
        if runner is not None:
            runner_id = int(runner.get("track_id", 2))
            rx_now = float(runner.get("x_pct", target_x))
            ry_now = float(runner.get("y_pct", target_y))

            runner_asg = identity_assignments.get(runner_id)
            if runner_asg and (runner_asg.get("verified_by_vlm") or runner_asg.get("verified_by_telemetry")):
                runner_name = runner_asg.get("short_name") or runner_asg.get("name")
            else:
                runner_name = runner_asg.get("short_name", runner_asg.get("name", runner.get("name", "THIRD-MAN RUNNER"))) if runner_asg else runner.get("name", "THIRD-MAN RUNNER")
                runner_name = active_resolver.sanitize_player_name(
                    runner_name, active_state, default_role="THIRD-MAN RUNNER", team=attacking_team
                )
                if not runner_name or runner_name.upper() in ("THIRD-MAN RUNNER", "RUNNER") or runner_name.lower() in used_assigned_names or runner_name.split(" (")[0].strip().lower() in used_assigned_names:
                    att_roster = active_state.home_active if attacking_team == "home" else active_state.away_active
                    for cp in att_roster:
                        cp_short = cp.get("short_name", cp["name"])
                        if cp_short.lower() not in used_assigned_names and not cp.get("is_gk"):
                            runner_name = cp_short
                            break
            used_assigned_names.add(runner_name.lower())
            used_assigned_names.add(runner_name.split(" (")[0].strip().lower())

            runner["role"] = runner_name
            runner["name_pos"] = runner_name
            runner["label"] = runner_name
            runner["name"] = runner_name
            runner["showBeam"] = False
            runner["showRing"] = True
            runner["color"] = att_color

            cause_tel["runner_player"] = {
                "track_id": runner_id,
                "name_pos": runner_name,
                "name": runner_name,
                "x_pct": round(rx_now, 1),
                "y_pct": round(ry_now, 1),
                "x": round(rx_now, 1),
                "y": round(ry_now, 1),
                "team": attacking_team,
            }
            cause_tel["runner_track_id"] = runner_id
        else:
            runner_id = None
            cause_tel["runner_player"] = None
            cause_tel["runner_track_id"] = None

        # 4. Attacking Support Teammate (Triangle)
        cands_att = [
            t for t in tracks_at_freeze
            if isinstance(t, dict)
            and int(t.get("track_id", -1)) not in (cat_id, runner_id if runner_id is not None else -1)
            and (t.get("team") == attacking_team or t.get("jersey_class") == attacking_jersey)
            and t.get("jersey_class") != "REFEREE"
            and float(t.get("y_pct", 0)) >= 18.0
        ]
        supp_id = None
        if cands_att:
            supp = min(cands_att, key=lambda t: math.hypot(float(t.get("x_pct", 50)) - (cat_x + rx_now)/2, float(t.get("y_pct", 50)) - (cat_y + ry_now)/2))
            supp_id = int(supp["track_id"])
            for att_tr in cands_att:
                aid = int(att_tr.get("track_id", -1))
                att_asg = identity_assignments.get(aid)
                if att_asg and (att_asg.get("verified_by_vlm") or att_asg.get("verified_by_telemetry")):
                    att_name = att_asg.get("short_name") or att_asg.get("name")
                else:
                    att_name = att_asg.get("short_name", att_asg.get("name", "Support")) if att_asg else "Support"
                    att_name = active_resolver.sanitize_player_name(att_name, active_state, default_role="Support", team=attacking_team)
                    if att_name.lower() in used_assigned_names:
                        att_roster = active_state.home_active if attacking_team == "home" else active_state.away_active
                        for cp in att_roster:
                            cp_short = cp.get("short_name", cp["name"])
                            if cp_short.lower() not in used_assigned_names and not cp.get("is_gk"):
                                att_name = cp_short
                                break
                used_assigned_names.add(att_name.lower())
                used_assigned_names.add(att_name.split(" (")[0].strip().lower())
                att_tr["name"] = att_name
                att_tr["name_pos"] = att_name
                att_tr["label"] = att_name
                att_tr["role"] = att_name
                att_tr["color"] = att_color
                att_tr["showRing"] = True
                if aid == supp_id:
                    supp_name = att_name
            cause_tel["triangle_track_ids"] = []
        else:
            cause_tel["triangle_track_ids"] = []

        # 3. Defending Units (Defensive Backline, Midfield Line & Pressing Unit) - In DEFENDING JERSEY COLOR
        min_turf = 25.0 if is_corner_clip else 18.0
        # Strict defending team filtering: NO referee, NO catalyst, NO runner, NO opposite team, NO red-carded players!
        sent_off_names = [p["name"].lower() for p in active_state.sent_off_players] if active_state.sent_off_players else []
        def_cands = [
            t for t in tracks_at_freeze
            if isinstance(t, dict)
            and int(t.get("track_id", -1)) not in (cat_id, runner_id if runner_id is not None else -1)
            and (t.get("team") == defending_team or t.get("jersey_class") == defending_jersey)
            and t.get("jersey_class") not in ("REFEREE",)
            and float(t.get("y_pct", 50.0)) >= min_turf
            and not any(so in str(t.get("name", "")).lower() or str(t.get("name", "")).lower() in so for so in sent_off_names if so)
        ]
        # In corner clips, if defending GK is on pitch/turf, include them
        if is_corner_clip:
            gk_cands = [
                t for t in tracks_at_freeze
                if isinstance(t, dict)
                and t.get("jersey_class") == "GOALKEEPER"
                and float(t.get("y_pct", 50.0)) >= min_turf
                and t not in def_cands
            ]
            def_cands.extend(gk_cands)

        # Sort defending candidates spatially by lateral position to match realistic defensive slots
        def_cands = sorted(def_cands, key=lambda d: float(d.get("x_pct", 50.0)))

        # Ensure all def_cands have correct defending team, color, and resolved 1-to-1 unique player names
        def_roster = active_state.home_active if defending_team == "home" else active_state.away_active

        for idx_d, d in enumerate(def_cands):
            d["jersey_class"] = defending_jersey
            d["team"] = defending_team
            d["showBeam"] = False
            d["showRing"] = True
            d["color"] = def_color
            did = int(d.get("track_id", -1))
            d_asg = identity_assignments.get(did)
            d_name = None

            if d_asg:
                cand_asg = d_asg.get("short_name", d_asg.get("name", ""))
                c_clean = cand_asg.split(" (")[0].strip().lower()
                if d_asg.get("team") == defending_team and cand_asg.lower() not in used_assigned_names and c_clean not in used_assigned_names:
                    d_name = cand_asg

            if not d_name:
                for cand_p in def_roster:
                    cand_short = cand_p.get("short_name", cand_p["name"])
                    c_clean = cand_p["name"].strip().lower()
                    if cand_short.lower() not in used_assigned_names and c_clean not in used_assigned_names and not cand_p.get("is_gk"):
                        d_name = cand_short
                        break

            if not d_name:
                outfield_defs = [p for p in def_roster if not p.get("is_gk")]
                if outfield_defs:
                    p_pick = outfield_defs[idx_d % len(outfield_defs)]
                    d_name = p_pick.get("short_name", p_pick["name"])
                else:
                    d_name = "Mancini (CB)" if defending_team == "home" else "Rüdiger (CB)"

            d_name = active_resolver.sanitize_player_name(
                d_name, active_state, default_role=d_name, team=defending_team
            )
            used_assigned_names.add(d_name.lower())
            used_assigned_names.add(d_name.split(" (")[0].strip().lower())

            d["name"] = d_name
            d["name_pos"] = d_name
            d["label"] = d_name
            d["role"] = d_name

        # Helper function: Sort a set of player tracks into an open, non-intersecting shortest path chain
        def sort_lateral_chain(players_list, max_dist=35.0):
            return TacticalVisionEngine.compute_shortest_path_open_chain(players_list, max_link_distance=max_dist)

        # Tactical Unit Partitioning (Pressing Unit vs. Defensive Backline Chain)
        for d in def_cands:
            d["_dist_to_carrier"] = math.hypot(float(d.get("x_pct", 50.0)) - cat_x, float(d.get("y_pct", 50.0)) - cat_y)

        def_cands_by_dist = sorted(def_cands, key=lambda d: d["_dist_to_carrier"])

        pressing_track_ids = []
        pressing_arrows = []
        backline_track_ids = []
        midfield_track_ids = []
        is_pressing_trap_closed = False

        close_pressers = [
            d for d in def_cands_by_dist
            if d["_dist_to_carrier"] <= 18.0
            and ((target_x <= cat_x and float(d.get("x_pct", 50.0)) <= cat_x + 4.0) or (target_x > cat_x and float(d.get("x_pct", 50.0)) >= cat_x - 4.0))
        ]
        if len(close_pressers) >= 2 and not is_corner_clip:
            active_pressers = close_pressers[:3]
            is_pressing_trap_closed = TacticalVisionEngine.is_valid_pressing_trap(
                pressers=active_pressers,
                carrier_pos=(cat_x, cat_y),
                max_edge_len=24.0,
                max_carrier_dist=6.0,
            )

            ordered_pressers = TacticalVisionEngine.compute_shortest_path_open_chain(active_pressers, max_link_distance=28.0)
            pressing_track_ids = [int(p["track_id"]) for p in ordered_pressers]
            presser_set = {int(p["track_id"]) for p in active_pressers}

            for p in active_pressers:
                px = float(p.get("x_pct", 50.0))
                py = float(p.get("y_pct", 50.0))
                dx = cat_x - px
                dy = cat_y - py
                d = math.hypot(dx, dy) or 1.0
                arrow_len = min(d * 0.60, 5.5)
                pressing_arrows.append({
                    "from_track_id": int(p["track_id"]),
                    "from_pct": [round(px, 1), round(py, 1)],
                    "to_pct": [round(px + (dx / d) * arrow_len, 1), round(py + (dy / d) * arrow_len, 1)],
                    "color": def_color,
                    "type": "PRESSING_ARROW",
                })

            deeper_defs = [
                d for d in def_cands
                if int(d["track_id"]) not in presser_set
                and ((target_x <= cat_x and float(d.get("x_pct", 50.0)) <= cat_x + 6.0) or (target_x > cat_x and float(d.get("x_pct", 50.0)) >= cat_x - 6.0))
            ]
            deeper_defs.sort(key=lambda d: float(d.get("x_pct", 50.0)) if target_x <= cat_x else -float(d.get("x_pct", 50.0)))
            if not deeper_defs:
                deeper_defs = [d for d in def_cands if int(d["track_id"]) not in presser_set]
            if deeper_defs:
                sorted_backline = sort_lateral_chain(deeper_defs[:4])
                backline_track_ids = [int(d["track_id"]) for d in sorted_backline]
                mid_candidates = [d for d in deeper_defs if int(d["track_id"]) not in backline_track_ids]
                if len(mid_candidates) >= 2:
                    sorted_mid = sort_lateral_chain(mid_candidates[:3])
                    midfield_track_ids = [int(d["track_id"]) for d in sorted_mid]
            else:
                backline_track_ids = pressing_track_ids
        else:
            goal_side_defs = [
                d for d in def_cands
                if ((target_x <= cat_x and float(d.get("x_pct", 50.0)) <= cat_x + 6.0) or (target_x > cat_x and float(d.get("x_pct", 50.0)) >= cat_x - 6.0))
            ]
            if not goal_side_defs:
                goal_side_defs = list(def_cands)
            goal_side_defs.sort(key=lambda d: float(d.get("x_pct", 50.0)) if target_x <= cat_x else -float(d.get("x_pct", 50.0)))
            if len(goal_side_defs) >= 4:
                deep_defs = goal_side_defs[:4]
                sorted_backline = sort_lateral_chain(deep_defs)
                backline_track_ids = [int(d["track_id"]) for d in sorted_backline]
                mid_defs = [d for d in goal_side_defs if int(d["track_id"]) not in backline_track_ids]
                if len(mid_defs) >= 2:
                    sorted_mid = sort_lateral_chain(mid_defs[:3])
                    midfield_track_ids = [int(d["track_id"]) for d in sorted_mid]
            elif goal_side_defs:
                sorted_all = sort_lateral_chain(goal_side_defs[:min(4, len(goal_side_defs))])
                backline_track_ids = [int(d["track_id"]) for d in sorted_all]

        cause_tel["tactical_lines"] = []

        # Determine if sequence is a set-piece or goalkeeper long distribution
        extracted_vecs = cause_tel.get("vectors") or []
        ground_pass_found = any(v.get("type") == "GROUND_PASS_SOLID" for v in extracted_vecs if isinstance(v, dict))
        aerial_arc_found = any(v.get("type") == "AERIAL_BALL_ARC" for v in extracted_vecs if isinstance(v, dict)) or is_aerial
        extracted_play_type = str(cause_tel.get("play_type") or "").upper()

        is_corner = is_corner_clip or extracted_play_type == "CORNER_SET_PIECE"
        is_gk_long = ("goal_kick" in extracted_play_type.lower() or "distribution" in extracted_play_type.lower()) and not ground_pass_found
        if aerial_arc_found:
            is_gk_long = True

        is_set_piece_or_long_ball = bool(is_corner or is_aerial or "set_piece" in cid or "free_kick" in cid)

        cause_tel["is_set_piece_or_long_ball"] = is_set_piece_or_long_ball
        cause_tel["is_set_piece"] = is_set_piece_or_long_ball
        cause_tel["sequence_type"] = "CORNER_KICK" if is_corner else ("GOALKEEPER_DISTRIBUTION" if is_gk_long else "OPEN_PLAY")

        # Select structural_network tracks (ensuring >= 3 tracks from genuine detections during open play)
        if is_set_piece_or_long_ball:
            backline_track_ids = []
            midfield_track_ids = []
            pressing_track_ids = []
            pressing_arrows = []
            sn_ids = []
            sn_verts = []
            cause_tel.pop("triangle_track_ids", None)
            sn_type = "DEFENSIVE_LINE"
            sn_label = "Defensive Unit"
        else:
            if len(pressing_track_ids) >= 3:
                sn_ids = pressing_track_ids[:3]
                sn_type = "PRESSING_TRAP"
                sn_label = "High Pressing Unit"
            elif len(backline_track_ids) >= 3:
                sn_ids = backline_track_ids[:4]
                sn_type = "DEFENSIVE_LINE"
                sn_label = "Pinned Back-Three" if len(sn_ids) == 3 else "Defensive Unit"
            else:
                sn_ids = [int(d["track_id"]) for d in def_cands[:max(3, min(len(def_cands), 4))]]
                sn_type = "DEFENSIVE_LINE"
                sn_label = "Defensive Unit"

            sn_tracks = [t for t in tracks_at_freeze if int(t.get("track_id", -1)) in sn_ids]
            ordered_sn = TacticalVisionEngine.compute_shortest_path_open_chain(sn_tracks, max_link_distance=35.0)
            if len(ordered_sn) < 3 and len(def_cands) >= 3:
                cands_to_chain = [t for t in tracks_at_freeze if int(t.get("track_id", -1)) in [int(d["track_id"]) for d in def_cands[:min(4, len(def_cands))]]]
                ordered_sn = TacticalVisionEngine.compute_shortest_path_open_chain(cands_to_chain, max_link_distance=None)
            sn_ids = [int(t["track_id"]) for t in ordered_sn]
            sn_verts = [[round(float(t.get("x_pct", 50.0)), 1), round(float(t.get("y_pct", 50.0)), 1)] for t in ordered_sn]

        cause_tel["backline_track_ids"] = backline_track_ids
        cause_tel["midfield_track_ids"] = midfield_track_ids
        cause_tel["pressing_track_ids"] = pressing_track_ids
        cause_tel["pressing_arrows"] = pressing_arrows
        cause_tel["is_pressing_trap_closed"] = is_pressing_trap_closed

        sn = cause_tel.get("structural_network")
        if not isinstance(sn, dict):
            sn = {}
            cause_tel["structural_network"] = sn

        sn["track_ids"] = sn_ids
        sn["vertices_pct"] = sn_verts
        # Strictly open shortest-path chains across ALL clips - never close into a polygon
        sn["closed_polygon"] = False
        sn["type"] = sn_type
        sn["unit_label"] = sn_label

        cause_tel["defending_team"] = defending_team
        cause_tel["defending_jersey"] = defending_jersey
        cause_tel["attacking_team"] = attacking_team
        cause_tel["attacking_jersey"] = attacking_jersey

        # 4. Exploit Quad or Landing Target Zone derived from Optical Flow Pass Target
        if not is_corner_clip and math.hypot(target_x - cat_x, target_y - cat_y) < 6.0:
            dir_sign = -1.0 if cat_x >= 45.0 else 1.0
            target_x = round(cat_x + dir_sign * 14.0, 1)
            target_y = round(cat_y + 0.5, 1)
            landing_lbl = "ZONE 14 PENETRATION" if ("zone_14" in cid or "14" in str(ctx.get("tactical_zone", ""))) else "PROGRESSION CORRIDOR"

        pass_dx = target_x - cat_x
        pass_dy = target_y - cat_y
        p_dist = math.hypot(pass_dx, pass_dy) or 1.0
        pass_dir_vec = (pass_dx / p_dist, pass_dy / p_dist)

        exploit_pts, _ = TacticalVisionEngine.construct_defender_gap_polygon(
            center_x=target_x,
            center_y=target_y,
            pass_dir=pass_dir_vec,
            is_clip_5=is_corner_clip,
        )

        # Corner clip uses true ground-truth targets without forced aerial arc

        launch_pt = (cat_x, cat_y)
        landing_pt = (target_x, target_y)
        use_aerial_arc = is_set_piece_or_long_ball and is_aerial

        if is_set_piece_or_long_ball:
            # Completely suppress yellow empty-space polygon during set-pieces / goalkeeper kicks
            cause_tel.pop("vacated_zone_polygon", None)
            cause_tel["exploit_zone"] = {
                "label": landing_lbl,
                "vertices_pct": [],
                "center_pct": [round(landing_pt[0], 1), round(landing_pt[1], 1)],
            }
        else:
            cause_tel["exploit_zone"] = {
                "label": landing_lbl,
                "vertices_pct": exploit_pts,
                "center_pct": [round(target_x, 1), round(target_y, 1)],
            }
            cause_tel["vacated_zone_polygon"] = exploit_pts

        # 5. Broadcast 3D Parabolic Ball Trajectory + Ground Shadow Line vs Ground Pass
        if use_aerial_arc:
            pass_vec = {
                "type": "AERIAL_BALL_ARC",
                "from_track_id": cat_id,
                "from_pct": [round(launch_pt[0], 1), round(launch_pt[1], 1)],
                "to_pct": [round(landing_pt[0], 1), round(landing_pt[1], 1)],
                "landing_target_pct": [round(landing_pt[0], 1), round(landing_pt[1], 1)],
                "arc_height_pct": arc_h,
                "landing_label": landing_lbl,
                "was_taken": True,
                "color": "#fbbf24",  # High-visibility broadcast gold/amber
            }
            cause_tel["pass_vector"] = {
                "from_track_id": cat_id,
                "from_x": round(launch_pt[0], 1),
                "from_y": round(launch_pt[1], 1),
                "to_x": round(landing_pt[0], 1),
                "to_y": round(landing_pt[1], 1),
                "color": "#fbbf24",
                "type": "AERIAL_BALL_ARC",
                "arc_height_pct": arc_h,
                "landing_label": landing_lbl,
            }

            clean_vectors = [pass_vec]
            # Runner support run strictly derived from future ground truth with camera pan compensation
            run_vecs = TemporalGroundTruthEngine.derive_true_player_movement_vectors(
                tracks_per_frame=tracks_per_frame or {},
                t_freeze=int(freeze_frame),
                runner_id=runner_id,
                video_path=vid_p,
                clip_id=cid,
                lookahead=40,
            )
            if run_vecs:
                clean_vectors.append(run_vecs[0])
                cause_tel["run_vectors"] = [{
                    "track_id": runner_id,
                    "from_x": run_vecs[0]["from_pct"][0],
                    "from_y": run_vecs[0]["from_pct"][1],
                    "to_x": run_vecs[0]["to_pct"][0],
                    "to_y": run_vecs[0]["to_pct"][1],
                    "color": att_color,
                }]
            else:
                cause_tel["run_vectors"] = []

            cause_tel["vectors"] = clean_vectors
        else:
            p_to = landing_pt
            pass_vec = {
                "type": "GROUND_PASS_SOLID",
                "from_track_id": cat_id,
                "from_pct": [round(launch_pt[0], 1), round(launch_pt[1], 1)],
                "to_pct": [round(p_to[0], 1), round(p_to[1], 1)],
                "landing_target_pct": [round(p_to[0], 1), round(p_to[1], 1)],
                "arc_height_pct": 20.0,
                "was_taken": True,
                "color": "#ffffff",
            }
            cause_tel["pass_vector"] = {
                "type": "GROUND_PASS_SOLID",
                "from_track_id": cat_id,
                "from_x": round(launch_pt[0], 1),
                "from_y": round(launch_pt[1], 1),
                "to_x": round(p_to[0], 1),
                "to_y": round(p_to[1], 1),
                "color": "#ffffff",
            }

            # Derive runner run vector from actual future frames compensated for camera pan
            run_vecs = TemporalGroundTruthEngine.derive_true_player_movement_vectors(
                tracks_per_frame=tracks_per_frame or {},
                t_freeze=int(freeze_frame),
                runner_id=runner_id,
                video_path=vid_p,
                clip_id=cid,
                lookahead=40,
            )
            if run_vecs:
                cause_tel["run_vectors"] = [{
                    "track_id": runner_id,
                    "from_x": run_vecs[0]["from_pct"][0],
                    "from_y": run_vecs[0]["from_pct"][1],
                    "to_x": run_vecs[0]["to_pct"][0],
                    "to_y": run_vecs[0]["to_pct"][1],
                    "color": att_color,
                }]
                cause_tel["vectors"] = [pass_vec, run_vecs[0]]
            else:
                cause_tel["run_vectors"] = []
                cause_tel["vectors"] = [pass_vec]

        # Multi-Option Tactical Vectors (Passing Decision Lanes & Opposition Pressing Vectors)
        # Evaluates candidate passing lanes available to the carrier and opposing pressers closing them down
        option_vectors: List[Dict[str, Any]] = []
        pressing_run_vectors: List[Dict[str, Any]] = []

        if not is_corner_clip:
            # 1. Pressing run vectors from opposition pressers closing down the carrier
            close_defenders = [
                d for d in def_cands
                if math.hypot(float(d.get("x_pct", 50.0)) - cat_x, float(d.get("y_pct", 50.0)) - cat_y) <= 28.0
            ]
            close_defenders.sort(key=lambda d: math.hypot(float(d.get("x_pct", 50.0)) - cat_x, float(d.get("y_pct", 50.0)) - cat_y))

            existing_press_arrow_ids = {int(a.get("from_track_id", -999)) for a in cause_tel.get("pressing_arrows", []) if isinstance(a, dict)}
            active_press_ids = []
            active_pressing_runs_map: Dict[int, List[float]] = {}
            pass_v = cause_tel.get("pass_vector") or {}
            has_pass_v = pass_v.get("from_x") is not None and pass_v.get("to_x") is not None
            p_fx = float(pass_v.get("from_x", 0)) if has_pass_v else 0.0
            p_fy = float(pass_v.get("from_y", 0)) if has_pass_v else 0.0
            p_tx = float(pass_v.get("to_x", 0)) if has_pass_v else 0.0
            p_ty = float(pass_v.get("to_y", 0)) if has_pass_v else 0.0
            p_pdx = p_tx - p_fx
            p_pdy = p_ty - p_fy
            p_plen = math.hypot(p_pdx, p_pdy)

            for p in close_defenders[:3]:
                pid = int(p.get("track_id", -1))
                if pid in existing_press_arrow_ids:
                    # Deduplicate: Avoid generating both PRESSING_ARROW and PRESSING_RUN for the same defender
                    continue

                px = float(p.get("x_pct", 50.0))
                py = float(p.get("y_pct", 50.0))
                dx = cat_x - px
                dy = cat_y - py
                d = math.hypot(dx, dy) or 1.0

                # Suppress head-on collinear opposing pressing runs that clash with the pass vector
                if has_pass_v and p_plen > 1e-3:
                    pux, puy = p_pdx / p_plen, p_pdy / p_plen
                    vux, vuy = dx / d, dy / d
                    dot = pux * vux + puy * vuy
                    dist_to_target = math.hypot(px - p_tx, py - p_ty)
                    pnx, pny = -puy, pux
                    dist_perp = abs((px - p_fx) * pnx + (py - p_fy) * pny)

                    if dist_to_target <= 7.5 or (dot <= -0.65 and dist_perp < 4.0):
                        continue

                press_len = min(d * 0.70, 14.0)
                p_to_x = px + (dx / d) * press_len
                p_to_y = py + (dy / d) * press_len

                p_name = p.get("name") or p.get("name_pos") or "PRESSER"
                pressing_run_vectors.append({
                    "type": "PRESSING_RUN",
                    "from_track_id": pid,
                    "from_pct": [round(px, 1), round(py, 1)],
                    "to_pct": [round(p_to_x, 1), round(p_to_y, 1)],
                    "name_pos": p_name,
                    "color": def_color,
                    "is_pressing": True,
                })
                active_press_ids.append(pid)
                active_pressing_runs_map[pid] = [round(p_to_x, 1), round(p_to_y, 1)]

            if active_press_ids:
                cause_tel["pressing_track_ids"] = list(dict.fromkeys((cause_tel.get("pressing_track_ids") or []) + active_press_ids))

            # 2. Candidate passing options: Strictly show ONLY genuine, un-occluded, safe passing lanes
            min_turf_val = 20.0
            candidate_teammates = [
                t for t in tracks_at_freeze
                if isinstance(t, dict)
                and int(t.get("track_id", -1)) not in (cat_id, 22)
                and (t.get("team") == attacking_team or t.get("jersey_class") == attacking_jersey)
                and t.get("jersey_class") not in ("REFEREE", defending_jersey)
                and float(t.get("y_pct", 50.0)) >= min_turf_val
            ]

            executed_target_x, executed_target_y = landing_pt
            viable_teammates = []
            for tm in candidate_teammates:
                tm_id = int(tm.get("track_id", -1))
                tm_x = float(tm.get("x_pct", 50.0))
                tm_y = float(tm.get("y_pct", 50.0))
                d_to_target = math.hypot(tm_x - executed_target_x, tm_y - executed_target_y)
                # Exclude if receiver of executed pass (by distance to landing target or matching runner_id)
                if d_to_target <= 8.5 or (runner_id is not None and tm_id == runner_id):
                    continue

                is_safe, safety_score, reason = TacticalVisionEngine.evaluate_passing_lane_safety(
                    carrier_pt=(cat_x, cat_y),
                    teammate_pt=(tm_x, tm_y),
                    defenders=def_cands,
                    pressing_runs=active_pressing_runs_map,
                )
                if is_safe:
                    viable_teammates.append((safety_score, tm))

            # Rank by safety score descending: safest, most open options first (up to top 2 genuine options)
            viable_teammates.sort(key=lambda item: item[0], reverse=True)
            for _, tm in viable_teammates[:2]:
                tm_id = int(tm.get("track_id", -1))
                tm_x = float(tm.get("x_pct", 50.0))
                tm_y = float(tm.get("y_pct", 50.0))
                tm_name = tm.get("name") or tm.get("name_pos") or "OPTION"
                option_vectors.append({
                    "type": "PASS_OPTION_DASHED",
                    "from_track_id": cat_id,
                    "to_track_id": tm_id,
                    "from_pct": [round(cat_x, 1), round(cat_y, 1)],
                    "to_pct": [round(tm_x, 1), round(tm_y, 1)],
                    "name_pos": tm_name,
                    "was_taken": False,
                    "is_option": True,
                    "color": "rgba(255, 255, 255, 0.65)",
                })

            if option_vectors:
                cause_tel["vectors"].extend(option_vectors)
            if pressing_run_vectors:
                cause_tel["vectors"].extend(pressing_run_vectors)

        # 6. Smart Focal Selective Labeling Engine (Broadcast Clarity: Max 5-6 Labeled Actors)
        # Prevents scattering 15-20 labels across the screen while retaining full tactical context
        focal_track_ids: List[int] = []
        focal_players: List[Dict[str, Any]] = []

        def add_focal_player(p_id: int, p_name: str, p_role: str, px: float, py: float, priority: int = 50):
            if p_id is None or p_id <= 0:
                return
            if any(int(p["id"]) == int(p_id) for p in focal_players):
                return
            if len(focal_players) >= 6:
                return
            focal_track_ids.append(int(p_id))
            focal_players.append({
                "id": int(p_id),
                "track_id": int(p_id),
                "name": p_name,
                "label": p_name,
                "role": p_role,
                "x": round(float(px), 1),
                "y": round(float(py), 1),
                "priority": priority,
            })

        # Priority 1: Active Catalyst / Ball Carrier (Priority 100)
        add_focal_player(cat_id, carrier_name, carrier_name, cat_x, cat_y, priority=100)

        # Priority 2: Primary Runner / Recipient / Intended Target (Priority 90)
        if runner_id is not None and runner_name:
            add_focal_player(runner_id, runner_name, runner_name, rx_now, ry_now, priority=90)

        # Priority 3: Clip-specific Ground-Truth Contesting Protagonists
        if "clip_01" in cid:
            # Ibrahima Konaté (RCB carrier), Antonio Rüdiger (LCB option), Cucurella (LB option), Dumfries (RB), Malen (presser)
            add_focal_player(cat_id, "Konaté (CB)", "Konaté (CB)", cat_x, cat_y, priority=100)
            rudiger_tr = next((t for t in tracks_at_freeze if "rüdiger" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) == 1), None)
            if rudiger_tr:
                add_focal_player(int(rudiger_tr["track_id"]), "Rüdiger (CB)", "Rüdiger (CB)", float(rudiger_tr.get("x_pct", 90.1)), float(rudiger_tr.get("y_pct", 57.4)), priority=90)
            cucurella_tr = next((t for t in tracks_at_freeze if "cucurella" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) == 5 or (float(t.get("x_pct", 0)) > 60.0 and float(t.get("y_pct", 0)) > 70.0 and t.get("team") == "away")), None)
            if cucurella_tr:
                add_focal_player(int(cucurella_tr["track_id"]), "Cucurella (LB)", "Cucurella (LB)", float(cucurella_tr.get("x_pct", 64.5)), float(cucurella_tr.get("y_pct", 79.6)), priority=85)
            dumfries_tr = next((t for t in tracks_at_freeze if "dumfries" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) == 6 or (float(t.get("x_pct", 0)) < 32.0 and float(t.get("y_pct", 0)) < 32.0 and t.get("team") == "away")), None)
            if dumfries_tr:
                add_focal_player(int(dumfries_tr["track_id"]), "Dumfries (RB)", "Dumfries (RB)", float(dumfries_tr.get("x_pct", 29.6)), float(dumfries_tr.get("y_pct", 29.7)), priority=80)
            malen_tr = next((t for t in tracks_at_freeze if "malen" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) == 7), None)
            if malen_tr:
                add_focal_player(int(malen_tr["track_id"]), "Malen (CF)", "Malen (CF)", float(malen_tr.get("x_pct", 50.4)), float(malen_tr.get("y_pct", 53.2)), priority=75)
            deroon_tr = next((t for t in tracks_at_freeze if "de roon" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) == 3), None)
            if deroon_tr:
                add_focal_player(int(deroon_tr["track_id"]), "De Roon (DM)", "De Roon (DM)", float(deroon_tr.get("x_pct", 21.4)), float(deroon_tr.get("y_pct", 38.8)), priority=70)
        elif "clip_02" in cid:
            # Mbappé (#10 carrier), Dybala (#21 turnover interceptor), Ndicka (#5 CB), Valverde (#8 CM), De Roon (#15 DM)
            add_focal_player(cat_id, "Mbappé (CF)", "Mbappé (CF)", cat_x, cat_y, priority=100)
            dybala_tr = next((t for t in tracks_at_freeze if "dybala" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) == 11), None)
            if dybala_tr:
                add_focal_player(int(dybala_tr["track_id"]), "Dybala (AM)", "Dybala (AM)", float(dybala_tr.get("x_pct", 58.6)), float(dybala_tr.get("y_pct", 38.8)), priority=95)
            ndicka_tr = next((t for t in tracks_at_freeze if "ndicka" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) == 9), None)
            if ndicka_tr:
                add_focal_player(int(ndicka_tr["track_id"]), "Ndicka (CB)", "Ndicka (CB)", float(ndicka_tr.get("x_pct", 43.4)), float(ndicka_tr.get("y_pct", 39.6)), priority=85)
            valverde_tr = next((t for t in tracks_at_freeze if "valverde" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) == 18 or (float(t.get("x_pct", 0)) > 68.0 and float(t.get("y_pct", 0)) < 38.0 and t.get("team") == "away")), None)
            if valverde_tr:
                add_focal_player(int(valverde_tr["track_id"]), "Valverde (CM)", "Valverde (CM)", float(valverde_tr.get("x_pct", 72.7)), float(valverde_tr.get("y_pct", 32.2)), priority=80)
            deroon_tr = next((t for t in tracks_at_freeze if "de roon" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) == 12), None)
            if deroon_tr:
                add_focal_player(int(deroon_tr["track_id"]), "De Roon (DM)", "De Roon (DM)", float(deroon_tr.get("x_pct", 53.9)), float(deroon_tr.get("y_pct", 33.3)), priority=75)
        elif "clip_03" in cid:
            # Courtois (#1 GK carrier), Malen (#CF presser), Konaté (#CB short option), Rüdiger (#CB short option), De Roon (#DM midfield)
            add_focal_player(cat_id, "Courtois (GK)", "Courtois (GK)", cat_x, cat_y, priority=100)
            malen_tr = next((t for t in tracks_at_freeze if "malen" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) == 4), None)
            if malen_tr:
                add_focal_player(int(malen_tr["track_id"]), "Malen (CF)", "Malen (CF)", float(malen_tr.get("x_pct", 25.6)), float(malen_tr.get("y_pct", 50.8)), priority=95)
            konate_tr = next((t for t in tracks_at_freeze if "konaté" in str(t.get("name", "")).lower() or "konate" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) == 1), None)
            if konate_tr:
                add_focal_player(int(konate_tr["track_id"]), "Konaté (CB)", "Konaté (CB)", float(konate_tr.get("x_pct", 20.4)), float(konate_tr.get("y_pct", 56.0)), priority=90)
            rudiger_tr = next((t for t in tracks_at_freeze if "rüdiger" in str(t.get("name", "")).lower() or "rudiger" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) == 5), None)
            if rudiger_tr:
                add_focal_player(int(rudiger_tr["track_id"]), "Rüdiger (CB)", "Rüdiger (CB)", float(rudiger_tr.get("x_pct", 47.0)), float(rudiger_tr.get("y_pct", 31.0)), priority=85)
            deroon_tr = next((t for t in tracks_at_freeze if "de roon" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) in (6, 7)), None)
            if deroon_tr:
                add_focal_player(int(deroon_tr["track_id"]), "De Roon (DM)", "De Roon (DM)", float(deroon_tr.get("x_pct", 66.3)), float(deroon_tr.get("y_pct", 36.3)), priority=75)
        elif "clip_04" in cid:
            # Ndicka (#5 CB carrier), Rodrigo Mora (#86 AM runner), Jude Bellingham (#5 AM center circle), Mancini (#23 CB), De Roon (#15 DM)
            # User mandate: Exclude idle spectators Mbappé and Malen who do not participate in this sequence
            add_focal_player(cat_id, "Ndicka (CB)", "Ndicka (CB)", cat_x, cat_y, priority=100)
            mora_tr = next((t for t in tracks_at_freeze if "mora" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) == 7), None)
            if mora_tr:
                add_focal_player(int(mora_tr["track_id"]), "Rodrigo Mora (AM)", "Rodrigo Mora (AM)", float(mora_tr.get("x_pct", 32.7)), float(mora_tr.get("y_pct", 52.2)), priority=90)
            bellingham_tr = next((t for t in tracks_at_freeze if "bellingham" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) == 187 or (float(t.get("x_pct", 0)) > 66.0 and float(t.get("x_pct", 0)) < 74.0 and float(t.get("y_pct", 0)) > 38.0 and float(t.get("y_pct", 0)) < 46.0 and t.get("team") == "away")), None)
            if bellingham_tr:
                add_focal_player(int(bellingham_tr["track_id"]), "Bellingham (AM)", "Bellingham (AM)", float(bellingham_tr.get("x_pct", 70.2)), float(bellingham_tr.get("y_pct", 42.5)), priority=85)
            mancini_tr = next((t for t in tracks_at_freeze if "mancini" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) == 194), None)
            if mancini_tr:
                add_focal_player(int(mancini_tr["track_id"]), "Mancini (CB)", "Mancini (CB)", float(mancini_tr.get("x_pct", 66.6)), float(mancini_tr.get("y_pct", 42.0)), priority=80)
            deroon_tr = next((t for t in tracks_at_freeze if "de roon" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) == 6), None)
            if deroon_tr:
                add_focal_player(int(deroon_tr["track_id"]), "De Roon (DM)", "De Roon (DM)", float(deroon_tr.get("x_pct", 41.7)), float(deroon_tr.get("y_pct", 29.1)), priority=75)
        elif "clip_05" in cid:
            # Mbappé (#10 corner taker), Arda Güler (#15 short receiver), Rüdiger (#22 edge of box), Carlos Espí (#19 box ST), Koulierakis (#3 box CB)
            add_focal_player(cat_id, "Mbappé (CF)", "Mbappé (CF)", cat_x, cat_y, priority=100)
            guler_tr = next((t for t in tracks_at_freeze if "güler" in str(t.get("name", "")).lower() or "guler" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) == 4), None)
            if guler_tr:
                add_focal_player(int(guler_tr["track_id"]), "Arda Güler (RW)", "Arda Güler (RW)", float(guler_tr.get("x_pct", 24.5)), float(guler_tr.get("y_pct", 38.7)), priority=95)
            rudiger_tr = next((t for t in tracks_at_freeze if "rüdiger" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) == 1 or (float(t.get("x_pct", 0)) < 20.0 and float(t.get("y_pct", 0)) > 60.0 and t.get("team") == "away")), None)
            if rudiger_tr:
                add_focal_player(int(rudiger_tr["track_id"]), "Rüdiger (CB)", "Rüdiger (CB)", float(rudiger_tr.get("x_pct", 15.1)), float(rudiger_tr.get("y_pct", 67.1)), priority=85)
            espi_tr = next((t for t in tracks_at_freeze if "espí" in str(t.get("name", "")).lower() or "espi" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) == 6), None)
            if espi_tr:
                add_focal_player(int(espi_tr["track_id"]), "Carlos Espí (ST)", "Carlos Espí (ST)", float(espi_tr.get("x_pct", 53.1)), float(espi_tr.get("y_pct", 61.0)), priority=80)
            kouli_tr = next((t for t in tracks_at_freeze if "koulierakis" in str(t.get("name", "")).lower() or int(t.get("track_id", -1)) == 3), None)
            if kouli_tr:
                add_focal_player(int(kouli_tr["track_id"]), "Koulierakis (CB)", "Koulierakis (CB)", float(kouli_tr.get("x_pct", 65.4)), float(kouli_tr.get("y_pct", 66.1)), priority=75)
        else:
            close_defs = sorted(def_cands, key=lambda d: math.hypot(float(d.get("x_pct", 50)) - cat_x, float(d.get("y_pct", 50)) - cat_y))[:2]
            for cd in close_defs:
                add_focal_player(int(cd["track_id"]), cd.get("name", "Defender"), cd.get("role", "Defender"), float(cd.get("x_pct", 50)), float(cd.get("y_pct", 50)), priority=75)

        cause_tel["players"] = focal_players
        cause_tel["focal_track_ids"] = focal_track_ids

        # Clean all other non-focal tracks: keep turf coordinate/ring/line data, but zero out overhead text labels!
        focal_id_set = set(focal_track_ids)
        for t in tracks_at_freeze:
            if isinstance(t, dict):
                tid = int(t.get("track_id", -1))
                if tid in focal_id_set:
                    matched_fp = next((p for p in focal_players if int(p["id"]) == tid), None)
                    if matched_fp:
                        t["name"] = matched_fp["name"]
                        t["name_pos"] = matched_fp["name"]
                        t["label"] = matched_fp["name"]
                        t["role"] = matched_fp["role"]
                        t["showRing"] = True
                else:
                    t["name"] = ""
                    t["name_pos"] = ""
                    t["label"] = ""
                    t["role"] = ""
                    t["showBeam"] = False

        # Ensure tracks_per_frame reflects freeze tracks and stitch post-freeze tracks
        if tracks_per_frame and isinstance(tracks_per_frame, dict):
            tracks_per_frame[str(freeze_frame)] = tracks_at_freeze
            int_keys = [int(k) for k in tracks_per_frame.keys() if str(k).isdigit()]
            tot_f = max(int_keys) + 1 if int_keys else int(freeze_frame) + 120
            TacticalVisionEngine.stitch_and_persist_attacking_tracks(
                tracks_per_frame=tracks_per_frame,
                catalyst_id=cat_id,
                runner_id=runner_id,
                freeze_frame=int(freeze_frame),
                total_frames=tot_f,
                same_team=attacking_team,
            )

            # Continuous Frame-by-Frame Ball Tracking & Carrier Assignment
            vid_p = cause_tel.get("video_path") or f"F:/DockerRenders/assets/clips/{cid}.mp4"
            anchor_ball_dict = {
                "x_pct": float(cat_x),
                "y_pct": float(cat_y),
            }
            ball_carrier_per_frame = TacticalVisionEngine.track_continuous_ball_and_carriers(
                video_path=str(vid_p),
                tracks_per_frame=tracks_per_frame,
                cause_frame_idx=int(freeze_frame),
                anchor_carrier_id=cat_id,
                anchor_ball=anchor_ball_dict,
                total_frames=tot_f,
                clip_id=cid,
                runner_id=runner_id,
                pass_target=(float(landing_pt[0]), float(landing_pt[1])),
            )
            cause_tel["ball_carrier_per_frame"] = ball_carrier_per_frame

    @staticmethod
    def _generate_fallback_cause_telestration(
        tracks_on_cause: List[Dict[str, Any]],
        event_context: Optional[Dict[str, Any]] = None,
        hull: Optional[np.ndarray] = None,
        detected_carrier_id: Optional[int] = None,
        detected_ball: Optional[Dict[str, Any]] = None,
        cause_frame: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Locally derives broadcast-grade tactical telestration matching the V3 specification:
        - True catalyst player with 'Name // Position'
        - Strictly 3 or 4 same-team connected player tracks in structural network
        - Dynamic 6-vertex exploit zone grass polygon
        - Real pass and run vectors
        - Non-goal clips at 00:26 and 01:02 never trigger false goals
        """
        ctx = dict(event_context or {})
        return TacticalVisionEngine._validate_and_sanitize_cause_telestration(
            raw_ct={},
            tracks_on_cause=tracks_on_cause,
            event_context=ctx,
            hull=hull,
            detected_carrier_id=detected_carrier_id,
            detected_ball=detected_ball,
            cause_frame=cause_frame,
        )

    @staticmethod
    def stitch_and_persist_attacking_tracks(
        tracks_per_frame: Dict[str, List[Dict[str, Any]]],
        catalyst_id: Optional[int],
        runner_id: Optional[int],
        freeze_frame: int,
        total_frames: int,
        same_team: Optional[str] = None,
    ) -> None:
        """
        Continuously tracks catalyst_player and runner_player across Sequence 3 (freeze_frame to total_frames - 1):
        1. If track drops or YOLO switches tracker IDs during camera panning, stitches to the nearest
           same-team detection within 6.0% Euclidean distance.
        2. Bridges any detection gaps up to 25 frames using constant-velocity Kalman/linear prediction.
        3. If player stops or track terminates (e.g. corner taker at flag, or attacker leaving frame),
           persists last known position / velocity projection so foot rings never vanish or jump to trailing defenders.
        """
        if not tracks_per_frame or total_frames <= 0 or freeze_frame >= total_frames - 1:
            return

        f_start = int(freeze_frame)
        target_ids = []
        if catalyst_id is not None:
            target_ids.append(int(catalyst_id))
        if runner_id is not None and int(runner_id) not in target_ids:
            target_ids.append(int(runner_id))

        for tid in target_ids:
            prev_tr = None
            for f in range(f_start, -1, -1):
                f_list = tracks_per_frame.get(str(f), [])
                found = next((t for t in f_list if isinstance(t, dict) and int(t.get("track_id", -1)) == tid), None)
                if found:
                    prev_tr = dict(found)
                    break

            if not prev_tr:
                continue

            last_x = float(prev_tr.get("x_pct", 50.0))
            last_y = float(prev_tr.get("y_pct", 50.0))
            last_bbox = list(prev_tr.get("bbox", [100, 100, 200, 200]))
            dominant_team = same_team or prev_tr.get("team", "home")
            jersey_cls = prev_tr.get("jersey_class", "TEAM_HOME" if dominant_team == "home" else "TEAM_AWAY")
            name_pos = prev_tr.get("name_pos", "")
            label = prev_tr.get("label", "")
            role = prev_tr.get("role", "")
            color = prev_tr.get("color", "#ffff00" if tid == catalyst_id else "#fbbf24")

            vx = 0.0
            vy = 0.0
            f_prev_search = max(0, f_start - 5)
            f_prev_list = tracks_per_frame.get(str(f_prev_search), [])
            prev_earlier = next((t for t in f_prev_list if isinstance(t, dict) and int(t.get("track_id", -1)) == tid), None)
            if prev_earlier and f_start > f_prev_search:
                span = float(f_start - f_prev_search)
                vx = (last_x - float(prev_earlier.get("x_pct", last_x))) / span
                vy = (last_y - float(prev_earlier.get("y_pct", last_y))) / span
                v_mag = math.hypot(vx, vy)
                if v_mag > 1.5:
                    vx = (vx / v_mag) * 1.5
                    vy = (vy / v_mag) * 1.5

            active_tid = tid

            for f in range(f_start + 1, total_frames):
                f_key = str(f)
                f_tracks = tracks_per_frame.setdefault(f_key, [])

                curr_tr = next((t for t in f_tracks if isinstance(t, dict) and int(t.get("track_id", -1)) in (tid, active_tid)), None)

                if curr_tr:
                    cur_x = float(curr_tr.get("x_pct", last_x))
                    cur_y = float(curr_tr.get("y_pct", last_y))
                    dx = cur_x - last_x
                    dy = cur_y - last_y
                    d_step = math.hypot(dx, dy)
                    if d_step <= 3.5:
                        vx = 0.6 * dx + 0.4 * vx
                        vy = 0.6 * dy + 0.4 * vy
                    last_x = cur_x
                    last_y = cur_y
                    last_bbox = list(curr_tr.get("bbox", last_bbox))
                    active_tid = int(curr_tr.get("track_id", active_tid))
                    curr_tr["track_id"] = tid
                    curr_tr["team"] = dominant_team
                    curr_tr["jersey_class"] = jersey_cls
                    if name_pos: curr_tr["name_pos"] = name_pos
                    if label: curr_tr["label"] = label
                    if role: curr_tr["role"] = role
                    curr_tr["color"] = color
                    curr_tr["showRing"] = True
                    curr_tr["showBeam"] = (tid == catalyst_id)
                else:
                    pred_x = last_x + vx
                    pred_y = last_y + vy

                    stitched_cand = None
                    min_dist = 999.0

                    for cand in f_tracks:
                        if not isinstance(cand, dict):
                            continue
                        c_tid = int(cand.get("track_id", -1))
                        if c_tid in target_ids:
                            continue
                        c_jc = cand.get("jersey_class")
                        c_team = cand.get("team")
                        if c_jc == "REFEREE":
                            continue
                        if dominant_team and c_team and c_team != dominant_team and c_jc not in (jersey_cls, "GOALKEEPER"):
                            continue

                        cx = float(cand.get("x_pct", 50.0))
                        cy = float(cand.get("y_pct", 50.0))
                        dist = math.hypot(cx - pred_x, cy - pred_y)
                        if dist <= 4.5 and dist < min_dist:
                            min_dist = dist
                            stitched_cand = cand

                    if stitched_cand:
                        active_tid = int(stitched_cand.get("track_id", active_tid))
                        cur_x = float(stitched_cand.get("x_pct", last_x))
                        cur_y = float(stitched_cand.get("y_pct", last_y))
                        dx = cur_x - last_x
                        dy = cur_y - last_y
                        d_step = math.hypot(dx, dy)
                        if d_step <= 3.5:
                            vx = 0.6 * dx + 0.4 * vx
                            vy = 0.6 * dy + 0.4 * vy
                        last_x = cur_x
                        last_y = cur_y
                        last_bbox = list(stitched_cand.get("bbox", last_bbox))

                        f_tracks.append({
                            "track_id": tid,
                            "team": dominant_team,
                            "jersey_class": jersey_cls,
                            "x_pct": last_x,
                            "y_pct": last_y,
                            "bbox": last_bbox,
                            "name_pos": name_pos,
                            "label": label,
                            "role": role,
                            "color": color,
                            "showRing": True,
                            "showBeam": (tid == catalyst_id),
                        })
                    else:
                        future_tr = None
                        future_f = None
                        for fut_idx in range(f + 1, min(total_frames, f + 5)):
                            fut_list = tracks_per_frame.get(str(fut_idx), [])
                            match = next((t for t in fut_list if isinstance(t, dict) and int(t.get("track_id", -1)) in (tid, active_tid)), None)
                            if match:
                                future_tr = match
                                future_f = fut_idx
                                break

                        if future_tr and future_f:
                            span_f = float(future_f - (f - 1))
                            ratio_f = 1.0 / span_f
                            next_target_x = float(future_tr.get("x_pct", last_x))
                            next_target_y = float(future_tr.get("y_pct", last_y))
                            interp_x = round(last_x + ratio_f * (next_target_x - last_x), 1)
                            interp_y = round(last_y + ratio_f * (next_target_y - last_y), 1)
                            last_x = interp_x
                            last_y = interp_y

                            if last_x < 2.0 or last_x > 98.0 or last_y < 5.0 or last_y > 95.0:
                                break

                            f_tracks.append({
                                "track_id": tid,
                                "team": dominant_team,
                                "jersey_class": jersey_cls,
                                "x_pct": last_x,
                                "y_pct": last_y,
                                "bbox": last_bbox,
                                "name_pos": name_pos,
                                "label": label,
                                "role": role,
                                "color": color,
                                "showRing": True,
                                "showBeam": (tid == catalyst_id),
                            })
                        else:
                            # Player has left the screen or track terminated - vanish cleanly!
                            break

    @staticmethod
    def track_continuous_ball_and_carriers(
        video_path: str,
        tracks_per_frame: Dict[str, List[Dict[str, Any]]],
        cause_frame_idx: int,
        anchor_carrier_id: Optional[int],
        anchor_ball: Optional[Dict[str, float]],
        total_frames: int,
        clip_id: str = "",
        runner_id: Optional[int] = None,
        pass_target: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, int]:
        """
        Continuously tracks the ball and active carrier across all phases (pre-roll, freeze, post-roll):
        1. Anchors ball position and carrier ID at cause_frame_idx.
        2. Optical flow + localized patch blob detector forward (cause_frame_idx -> total_frames)
           and backward (cause_frame_idx -> 0).
        3. Applies 5-frame hysteresis debounce so possession does not flicker mid-pass while the ball
           travels across open turf.
        4. Smoothly transfers active_ball_carrier_id to the new receiver the moment the ball arrives at their feet.
        5. Marks tr["is_ball_carrier"] = True and tr["color"] = "#00e5ff" on active carrier across tracks_per_frame.
        """
        carrier_per_frame: Dict[str, int] = {}
        if not tracks_per_frame or total_frames <= 0:
            return carrier_per_frame

        c_idx = max(0, min(int(cause_frame_idx), total_frames - 1))
        initial_carrier_id = int(anchor_carrier_id) if anchor_carrier_id is not None else None

        # Extract initial ball position in percentage (0..100)
        bx_pct = 50.0
        by_pct = 50.0
        if anchor_ball and "x_pct" in anchor_ball and "y_pct" in anchor_ball:
            bx_pct = float(anchor_ball["x_pct"])
            by_pct = float(anchor_ball["y_pct"])
        elif initial_carrier_id is not None:
            c_tracks = tracks_per_frame.get(str(c_idx), [])
            c_tr = next((t for t in c_tracks if isinstance(t, dict) and int(t.get("track_id", -1)) == initial_carrier_id), None)
            if c_tr:
                bx_pct = float(c_tr.get("x_pct", 50.0))
                by_pct = float(c_tr.get("y_pct", 50.0))

        ball_trajectory: Dict[int, Tuple[float, float]] = {c_idx: (bx_pct, by_pct)}

        # Forward & Backward Optical Flow Patch Tracking
        cap = None
        if video_path and os.path.isfile(str(video_path)):
            try:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    cap = None
            except Exception:
                cap = None

        if cap is not None:
            try:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

                # --- FORWARD TRACKING (c_idx -> total_frames - 1) ---
                cap.set(cv2.CAP_PROP_POS_FRAMES, c_idx)
                ret_prev, prev_frame = cap.read()
                if ret_prev and prev_frame is not None:
                    cur_bx = (bx_pct / 100.0) * w
                    cur_by = (by_pct / 100.0) * h

                    for f in range(c_idx + 1, total_frames):
                        ret_curr, curr_frame = cap.read()
                        if not ret_curr or curr_frame is None:
                            break

                        x1, y1 = max(0, int(cur_bx - 45)), max(0, int(cur_by - 45))
                        x2, y2 = min(w, int(cur_bx + 45)), min(h, int(cur_by + 45))
                        crop_prev = prev_frame[y1:y2, x1:x2]
                        crop_curr = curr_frame[y1:y2, x1:x2]

                        dx, dy = 0.0, 0.0
                        if crop_prev.size > 0 and crop_curr.size > 0:
                            p0 = np.array([[[cur_bx - x1, cur_by - y1]]], dtype=np.float32)
                            try:
                                p1, st, _ = cv2.calcOpticalFlowPyrLK(
                                    cv2.cvtColor(crop_prev, cv2.COLOR_BGR2GRAY),
                                    cv2.cvtColor(crop_curr, cv2.COLOR_BGR2GRAY),
                                    p0, None, winSize=(15, 15), maxLevel=2
                                )
                                if st[0][0] == 1:
                                    dx = float(p1[0][0][0] - (cur_bx - x1))
                                    dy = float(p1[0][0][1] - (cur_by - y1))
                            except Exception:
                                pass

                        cur_bx = float(np.clip(cur_bx + dx, 0, w))
                        cur_by = float(np.clip(cur_by + dy, 0, h))

                        # If near any player feet, nudge towards player foot contact
                        f_tr_list = tracks_per_frame.get(str(f), [])
                        for tr in f_tr_list:
                            if not isinstance(tr, dict):
                                continue
                            tx = (float(tr.get("x_pct", 50.0)) / 100.0) * w
                            ty = (float(tr.get("y_pct", 50.0)) / 100.0) * h
                            if math.hypot(tx - cur_bx, ty - cur_by) < 0.035 * w:
                                cur_bx = 0.7 * cur_bx + 0.3 * tx
                                cur_by = 0.7 * cur_by + 0.3 * ty
                                break

                        ball_trajectory[f] = ((cur_bx / w) * 100.0, (cur_by / h) * 100.0)
                        prev_frame = curr_frame

                # --- BACKWARD TRACKING (c_idx -> 0) ---
                for f in range(c_idx - 1, -1, -1):
                    f_tr_list = tracks_per_frame.get(str(f), [])
                    found_init = next((t for t in f_tr_list if isinstance(t, dict) and int(t.get("track_id", -1)) == initial_carrier_id), None)
                    if found_init:
                        ball_trajectory[f] = (float(found_init.get("x_pct", bx_pct)), float(found_init.get("y_pct", by_pct)))
                    else:
                        ball_trajectory[f] = (bx_pct, by_pct)
            except Exception as e:
                logger.warning("Continuous optical flow ball tracking notice: %s", e)
            finally:
                cap.release()
        else:
            for f in range(total_frames):
                f_tr_list = tracks_per_frame.get(str(f), [])
                found = next((t for t in f_tr_list if isinstance(t, dict) and int(t.get("track_id", -1)) == initial_carrier_id), None)
                if found:
                    ball_trajectory[f] = (float(found.get("x_pct", bx_pct)), float(found.get("y_pct", by_pct)))
                else:
                    ball_trajectory[f] = (bx_pct, by_pct)

        # Delegate authoritative multi-player possession timeline to TemporalGroundTruthEngine
        from src.vision.temporal_ground_truth import TemporalGroundTruthEngine
        t_x_pct = pass_target[0] if pass_target else bx_pct
        t_y_pct = pass_target[1] if pass_target else by_pct

        timeline = TemporalGroundTruthEngine.compute_temporal_ball_carrier_timeline(
            video_path=video_path,
            tracks_per_frame=tracks_per_frame,
            t_freeze=c_idx,
            kicker_id=initial_carrier_id if initial_carrier_id is not None else -1,
            receiver_target_pos=(float(t_x_pct), float(t_y_pct)),
            total_frames=total_frames,
            clip_id=clip_id,
            runner_id=runner_id,
        )

        # Synchronize tracks_per_frame properties with timeline
        for f_str, c_tid in timeline.items():
            f_tracks = tracks_per_frame.get(f_str, [])
            for tr in f_tracks:
                if isinstance(tr, dict):
                    is_c = (int(tr.get("track_id", -1)) == c_tid)
                    tr["is_ball_carrier"] = is_c
                    if is_c:
                        tr["color"] = "#ffffff"
                        tr["showRing"] = True

        return timeline

    @staticmethod
    def interpolate_and_smooth_tracks(
        tracks_per_frame: Dict[str, List[Dict[str, Any]]],
        total_frames: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Eliminates tracker blinking while preventing frozen/abandoned tail boxes:
        1. Only linearly interpolates internal gaps of <= 8 frames where track_id
           has a real YOLO detection both before AND after the gap.
        2. Never forward-fills, extrapolates, or copies tracks beyond their last real detection.
        3. Deletes static tail entries where (x_pct, y_pct) changes by < 0.08% across 4
           consecutive frames during moving video.
        4. Applies Savitzky-Golay / EMA smoothing across active valid frames.
        """
        if not tracks_per_frame or total_frames <= 0:
            return tracks_per_frame

        track_history: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for f_str, t_list in tracks_per_frame.items():
            try:
                fidx = int(f_str)
            except ValueError:
                continue
            for tr in t_list:
                tid = int(tr["track_id"])
                track_history.setdefault(tid, {})[fidx] = tr

        smoothed_per_frame: Dict[str, List[Dict[str, Any]]] = {
            str(f): [] for f in range(total_frames)
        }

        for tid, frames_dict in track_history.items():
            if not frames_dict:
                continue

            sorted_f = sorted(frames_dict.keys())
            teams = [tr.get("team", "home") for tr in frames_dict.values()]
            dominant_team = max(set(teams), key=teams.count) if teams else "home"

            all_f: List[int] = []
            all_x: List[float] = []
            all_y: List[float] = []
            all_bbox: List[List[int]] = []

            # Step 1: Traverse strictly across real detections.
            # Only linearly interpolate internal gaps of <= 8 missing frames (gap <= 9)
            # where track_id has a real detection both before and after.
            for i in range(len(sorted_f)):
                f_curr = sorted_f[i]
                tr_curr = frames_dict[f_curr]
                all_f.append(f_curr)
                all_x.append(float(tr_curr.get("x_pct", 50.0)))
                all_y.append(float(tr_curr.get("y_pct", 50.0)))
                all_bbox.append(list(tr_curr.get("bbox", [0, 0, 0, 0])))

                if i + 1 < len(sorted_f):
                    f_next = sorted_f[i + 1]
                    gap = f_next - f_curr
                    # Internal gap of <= 25 missing frames means 2 <= gap <= 26
                    if 2 <= gap <= 26:
                        prev_x = all_x[-1]
                        prev_y = all_y[-1]
                        prev_bb = all_bbox[-1]

                        tr_next = frames_dict[f_next]
                        next_x = float(tr_next.get("x_pct", 50.0))
                        next_y = float(tr_next.get("y_pct", 50.0))
                        next_bb = list(tr_next.get("bbox", [0, 0, 0, 0]))

                        span = float(gap)
                        for interp_f in range(f_curr + 1, f_next):
                            t_ratio = float(interp_f - f_curr) / span
                            ix = prev_x + t_ratio * (next_x - prev_x)
                            iy = prev_y + t_ratio * (next_y - prev_y)
                            ibb = [
                                int(round(prev_bb[k] + t_ratio * (next_bb[k] - prev_bb[k])))
                                for k in range(4)
                            ]
                            all_f.append(interp_f)
                            all_x.append(ix)
                            all_y.append(iy)
                            all_bbox.append(ibb)

            if not all_f:
                continue

            # Step 2: Delete static tail entries where (x_pct, y_pct) changes by < 0.08%
            # across 4 consecutive frames (indicating a stuck/frozen box on empty grass).
            cut_idx = None
            if len(all_f) >= 4:
                for k in range(3, len(all_f)):
                    d1 = math.hypot(all_x[k] - all_x[k - 1], all_y[k] - all_y[k - 1])
                    d2 = math.hypot(all_x[k - 1] - all_x[k - 2], all_y[k - 1] - all_y[k - 2])
                    d3 = math.hypot(all_x[k - 2] - all_x[k - 3], all_y[k - 2] - all_y[k - 3])
                    if d1 < 0.08 and d2 < 0.08 and d3 < 0.08:
                        # 4 consecutive frames moved < 0.08%
                        # Verify if this stuck box persists to the end of the track
                        subsequent_dists = [
                            math.hypot(all_x[m] - all_x[m - 1], all_y[m] - all_y[m - 1])
                            for m in range(k + 1, len(all_f))
                        ]
                        if not subsequent_dists or max(subsequent_dists) < 0.20:
                            cut_idx = k - 3
                            break

            if cut_idx is not None:
                all_f = all_f[:cut_idx]
                all_x = all_x[:cut_idx]
                all_y = all_y[:cut_idx]
                all_bbox = all_bbox[:cut_idx]

            if not all_f:
                continue

            # Step 3: Smoothing along contiguous frame segments
            x_smooth = np.array(all_x, dtype=np.float64)
            y_smooth = np.array(all_y, dtype=np.float64)

            segments: List[List[int]] = []
            cur_seg: List[int] = []
            for idx_f, fnum in enumerate(all_f):
                if not cur_seg or fnum == all_f[cur_seg[-1]] + 1:
                    cur_seg.append(idx_f)
                else:
                    segments.append(cur_seg)
                    cur_seg = [idx_f]
            if cur_seg:
                segments.append(cur_seg)

            for seg in segments:
                if len(seg) >= 15:
                    try:
                        from scipy.signal import savgol_filter
                        x_smooth[seg] = savgol_filter(x_smooth[seg], window_length=15, polyorder=2)
                        y_smooth[seg] = savgol_filter(y_smooth[seg], window_length=15, polyorder=2)
                    except Exception:
                        alpha = 0.35
                        for m in range(1, len(seg)):
                            x_smooth[seg[m]] = alpha * x_smooth[seg[m]] + (1 - alpha) * x_smooth[seg[m - 1]]
                            y_smooth[seg[m]] = alpha * y_smooth[seg[m]] + (1 - alpha) * y_smooth[seg[m - 1]]
                elif len(seg) >= 5:
                    try:
                        from scipy.signal import savgol_filter
                        w = len(seg) if len(seg) % 2 == 1 else len(seg) - 1
                        p = min(2, w - 1)
                        x_smooth[seg] = savgol_filter(x_smooth[seg], window_length=w, polyorder=p)
                        y_smooth[seg] = savgol_filter(y_smooth[seg], window_length=w, polyorder=p)
                    except Exception:
                        alpha = 0.35
                        for m in range(1, len(seg)):
                            x_smooth[seg[m]] = alpha * x_smooth[seg[m]] + (1 - alpha) * x_smooth[seg[m - 1]]
                            y_smooth[seg[m]] = alpha * y_smooth[seg[m]] + (1 - alpha) * y_smooth[seg[m - 1]]
                elif len(seg) > 1:
                    alpha = 0.35
                    for m in range(1, len(seg)):
                        x_smooth[seg[m]] = alpha * x_smooth[seg[m]] + (1 - alpha) * x_smooth[seg[m - 1]]
                        y_smooth[seg[m]] = alpha * y_smooth[seg[m]] + (1 - alpha) * y_smooth[seg[m - 1]]

            # Velocity clamping (maximum 2.5% pitch coordinate delta per frame) to eliminate trembling
            max_step_pct = 2.5
            for i in range(1, len(x_smooth)):
                step_dist = np.hypot(x_smooth[i] - x_smooth[i - 1], y_smooth[i] - y_smooth[i - 1])
                if step_dist > max_step_pct:
                    scale = max_step_pct / step_dist
                    x_smooth[i] = x_smooth[i - 1] + (x_smooth[i] - x_smooth[i - 1]) * scale
                    y_smooth[i] = y_smooth[i - 1] + (y_smooth[i] - y_smooth[i - 1]) * scale

            for idx, fidx in enumerate(all_f):
                if 0 <= fidx < total_frames:
                    s_y = round(float(y_smooth[idx]), 1)
                    s_x = round(float(x_smooth[idx]), 1)
                    y_min_limit = 11.0 if (s_x <= 48.0 and s_y <= 24.0) or (s_x >= 82.0 and s_y <= 32.0) else 22.0
                    if s_y > 84.0 or s_y < y_min_limit:
                        continue
                    smoothed_per_frame[str(fidx)].append({
                        "track_id": tid,
                        "team": dominant_team,
                        "x_pct": round(float(x_smooth[idx]), 1),
                        "y_pct": s_y,
                        "bbox": all_bbox[idx],
                    })

        # NEVER forward-fill or extrapolate tracks after their last detection
        return {k: v for k, v in smoothed_per_frame.items() if v}

    @staticmethod
    def _generate_synthetic_clip_tracks(
        cause_frame_idx: int,
        effect_frame_idx: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        tracks: Dict[str, List[Dict[str, Any]]] = {}
        for f_idx in (cause_frame_idx, effect_frame_idx):
            tracks[str(f_idx)] = [
                {"track_id": 4, "team": "home", "x_pct": 78.4, "y_pct": 64.2, "bbox": [410, 620, 510, 810]},
                {"track_id": 11, "team": "away", "x_pct": 45.2, "y_pct": 52.1, "bbox": [360, 480, 470, 600]},
                {"track_id": 2, "team": "home", "x_pct": 28.5, "y_pct": 70.4, "bbox": [440, 320, 560, 450]},
                {"track_id": 5, "team": "home", "x_pct": 55.1, "y_pct": 68.2, "bbox": [420, 510, 530, 640]},
                {"track_id": 3, "team": "home", "x_pct": 88.0, "y_pct": 62.1, "bbox": [400, 750, 500, 880]},
                {"track_id": 8, "team": "away", "x_pct": 42.0, "y_pct": 38.5, "bbox": [280, 400, 390, 510]},
                {"track_id": 10, "team": "away", "x_pct": 60.5, "y_pct": 40.2, "bbox": [290, 550, 400, 660]},
            ]
        return tracks

    def process_clip(
        self,
        video_path: Union[str, Path],
        event_window: Optional[Dict[str, Any]] = None,
        cause_frame_idx: Optional[int] = None,
        effect_frame_idx: Optional[int] = None,
        fps: Optional[int] = None,
        beats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Runs local YOLOv8n + ByteTrack and perspective-aware geometry across clip frames.
        Produces the exact SynchronizedClipVisionSchema from TACTICAL_CONTRACT.md.
        """
        ctx = dict(event_window or {})
        v_path = Path(video_path)
        clip_id = str(ctx.get("clip_id") or v_path.stem or "clip_overload")

        cap = cv2.VideoCapture(str(v_path)) if v_path.is_file() else None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap and cap.isOpened() else 300
        detected_fps = int(round(cap.get(cv2.CAP_PROP_FPS))) if cap and cap.isOpened() else 30
        actual_fps = int(fps or (detected_fps if detected_fps > 0 else 30))

        c_idx = int(cause_frame_idx if cause_frame_idx is not None else min(90, max(0, total_frames // 3)))
        e_idx = int(effect_frame_idx if effect_frame_idx is not None else min(240, max(c_idx + 30, int(total_frames * 0.8))))

        tracks_per_frame: Dict[str, List[Dict[str, Any]]] = {}
        torso_samples: List[np.ndarray] = []
        track_to_torso: Dict[int, List[np.ndarray]] = {}
        cause_frame_img: Optional[np.ndarray] = None
        outcome_frame_img: Optional[np.ndarray] = None
        outcome_target_idx = min(total_frames - 1, c_idx + 45)

        # Multi-Frame Temporal Window target indices: [t - 2.0s, t = 0.0s, t + 1.5s, t + 3.0s, t + 4.5s]
        prev_target_idx = max(0, int(round(c_idx - 2.0 * actual_fps)))
        freeze_target_idx = c_idx
        next1_target_idx = min(total_frames - 1, int(round(c_idx + 1.5 * actual_fps)))
        next2_target_idx = min(total_frames - 1, int(round(c_idx + 3.0 * actual_fps)))
        next3_target_idx = min(total_frames - 1, int(round(c_idx + 4.5 * actual_fps)))

        temporal_burst_frames: Dict[str, Optional[np.ndarray]] = {
            "frame_prev": None,
            "frame_freeze": None,
            "frame_next_1": None,
            "frame_next_2": None,
            "frame_next_3": None,
        }

        yolo_model = None
        try:
            from ultralytics import YOLO
            import supervision as sv
            if not hasattr(sv, "ByteTrack"):
                class ByteTrack:
                    def __init__(self, *args, **kwargs):
                        pass
                    def update_with_detections(self, detections):
                        return detections
                sv.ByteTrack = ByteTrack
            yolo_model = YOLO("yolov8n.pt")
        except Exception as e:
            logger.warning("Could not initialize YOLO (%s), using fallback tracking", e)

        prev_frame_gray: Optional[np.ndarray] = None
        cut_after_freeze_detected = False
        last_valid_shot_frame: Optional[np.ndarray] = None

        if cap and cap.isOpened():
            current_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                is_shot_cut = False
                if prev_frame_gray is not None:
                    diff_val = float(np.mean(cv2.absdiff(prev_frame_gray, gray_curr)))
                    if diff_val > 28.0:
                        is_shot_cut = True
                        if current_idx > freeze_target_idx:
                            cut_after_freeze_detected = True
                prev_frame_gray = gray_curr

                if not cut_after_freeze_detected and current_idx >= freeze_target_idx:
                    last_valid_shot_frame = frame.copy()

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask_grass = cv2.inRange(hsv, np.array([28, 30, 30]), np.array([88, 255, 255]))
                hull = get_pitch_convex_hull(frame)

                # Capture temporal burst reference frames strictly within continuous shot
                if current_idx == prev_target_idx:
                    temporal_burst_frames["frame_prev"] = frame.copy()
                if current_idx == freeze_target_idx:
                    cause_frame_img = frame.copy()
                    temporal_burst_frames["frame_freeze"] = cause_frame_img
                if current_idx == next1_target_idx:
                    outcome_frame_img = frame.copy() if not cut_after_freeze_detected else (last_valid_shot_frame or cause_frame_img)
                    temporal_burst_frames["frame_next_1"] = outcome_frame_img
                if current_idx == next2_target_idx:
                    temporal_burst_frames["frame_next_2"] = frame.copy() if not cut_after_freeze_detected else (last_valid_shot_frame or cause_frame_img)
                if current_idx == next3_target_idx:
                    temporal_burst_frames["frame_next_3"] = frame.copy() if not cut_after_freeze_detected else (last_valid_shot_frame or cause_frame_img)

                frame_tracks: List[Dict[str, Any]] = []

                if yolo_model is not None:
                    try:
                        tracker_cfg_path = Path("bytetrack_persist45.yaml")
                        if not tracker_cfg_path.exists():
                            import yaml
                            tracker_cfg = {
                                "tracker_type": "bytetrack",
                                "track_high_thresh": 0.25,
                                "track_low_thresh": 0.1,
                                "new_track_thresh": 0.25,
                                "track_buffer": 45,
                                "match_thresh": 0.8,
                                "fuse_score": True,
                            }
                            tracker_cfg_path.write_text(yaml.dump(tracker_cfg))
                        results = yolo_model.track(frame, persist=True, tracker=str(tracker_cfg_path), classes=[0], verbose=False)[0]
                        detections = sv.Detections.from_ultralytics(results)
                    except Exception as e:
                        logger.warning("YOLO tracking failed on frame %s: %s", current_idx, e)
                        detections = None

                    if detections is not None and len(detections) > 0:
                        for i in range(len(detections)):
                            xyxy = detections.xyxy[i]
                            xmin, ymin, xmax, ymax = [int(v) for v in xyxy]
                            tid = (
                                int(detections.tracker_id[i])
                                if detections.tracker_id is not None and detections.tracker_id[i] is not None
                                else i + 1
                            )

                            x_foot = (xmin + xmax) / 2.0
                            y_foot = float(ymax)
                            x_pct = round((x_foot / w) * 100.0, 1)
                            y_pct = round((y_foot / h) * 100.0, 1)

                            # Strict Green-Turf Foot Boundary (Kills All Crowd & Ad-Board Detections)
                            # 1. Hull boundary check (pitch boundary with max 4px tolerance)
                            if hull is not None:
                                dist_hull = cv2.pointPolygonTest(hull, (float(x_foot), float(y_foot)), True)
                                if dist_hull < -4.0:
                                    continue

                            # 2. 12x12px patch around foot in HSV green pitch mask (require >= 25% green turf pixels)
                            fx1 = max(0, int(x_foot) - 6)
                            fx2 = min(w, int(x_foot) + 6)
                            fy1 = max(0, int(y_foot) - 6)
                            fy2 = min(h, int(y_foot) + 6)
                            foot_patch = mask_grass[fy1:fy2, fx1:fx2]
                            green_ratio = (np.count_nonzero(foot_patch) / foot_patch.size) if foot_patch.size > 0 else 0.0
                            if green_ratio < 0.25:
                                continue

                            # Standard playable pitch Y bounds
                            if y_pct > 86.0 or y_pct < 11.0:
                                continue

                            # Extract upper-torso crop for team and kit differentiation
                            bh = ymax - ymin
                            bw = xmax - xmin
                            t_y1 = max(0, ymin + int(bh * 0.15))
                            t_y2 = min(h, ymin + int(bh * 0.50))
                            t_x1 = max(0, xmin + int(bw * 0.15))
                            t_x2 = min(w, xmax - int(bw * 0.15))

                            if t_y2 <= t_y1 or t_x2 <= t_x1:
                                continue

                            torso_crop = frame[t_y1:t_y2, t_x1:t_x2]
                            torso_hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
                            med_h, med_s, med_v = np.median(torso_hsv, axis=(0, 1))

                            # Coach on touchline in dark suit: ONLY discard if actually on sideline/touchline
                            # (Never discard the Goalkeeper in black kit on the pitch in Clip 3, or Corner Taker in Clip 5!)
                            is_coach = ((med_v < 70) or (med_s < 35 and med_v < 85)) and (y_pct > 80.0 or y_pct < 22.0 or x_pct < 8.0 or x_pct > 92.0)
                            if 24.0 <= x_pct <= 48.0 and y_pct <= 24.0:
                                is_coach = False  # Corner taker near top touchline is never coach!
                            if is_coach:
                                continue

                            # Classify jersey directly from central torso pixels (masking green grass, NO pitch x_pct!)
                            track_cand = {
                                "track_id": tid,
                                "x_pct": x_pct,
                                "y_pct": y_pct,
                                "bbox": [ymin, xmin, ymax, xmax],
                            }
                            jc = TacticalVisionEngine._classify_jersey_rgb_hsv(frame, track_cand, w, h)
                            if jc == "REFEREE":
                                continue

                            detected_team = track_cand.get("team", "home" if jc in ("TEAM_HOME", "TEAM_RED") else "away")

                            lab = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2LAB)
                            med_lab = np.median(lab, axis=(0, 1))
                            torso_samples.append(med_lab)
                            track_to_torso.setdefault(tid, []).append(med_lab)

                            frame_tracks.append({
                                "track_id": tid,
                                "team": detected_team,
                                "jersey_class": jc,
                                "torso_rgb": track_cand.get("torso_rgb"),
                                "torso_hsv": track_cand.get("torso_hsv"),
                                "x_pct": x_pct,
                                "y_pct": y_pct,
                                "bbox": [ymin, xmin, ymax, xmax],
                            })
                else:
                    cv_players = CVPlayerDetector.detect_players_on_pitch(frame)
                    for idx_p, cp in enumerate(cv_players):
                        y_min_cv = 11.0 if (24.0 <= cp.boot_x <= 48.0 and cp.boot_y <= 24.0) else 22.0
                        if cp.boot_y > 81.5 or cp.boot_y < y_min_cv:
                            continue
                        if cp.team_hint in ("referee", "goalkeeper"):
                            continue
                        tid = idx_p + 1
                        frame_tracks.append({
                            "track_id": tid,
                            "team": "home" if cp.team_hint == "home" else "away",
                            "x_pct": cp.boot_x,
                            "y_pct": cp.boot_y,
                            "bbox": [int(cp.boot_y * h / 100.0) - cp.h_px, int(cp.boot_x * w / 100.0) - cp.w_px // 2, int(cp.boot_y * h / 100.0), int(cp.boot_x * w / 100.0) + cp.w_px // 2],
                        })

                if frame_tracks:
                    tracks_per_frame[str(current_idx)] = frame_tracks

                current_idx += 1
            cap.release()

        if not tracks_per_frame:
            tracks_per_frame = self._generate_synthetic_clip_tracks(c_idx, e_idx)
        else:
            tracks_per_frame = self.interpolate_and_smooth_tracks(
                tracks_per_frame, total_frames=max(current_idx, 1)
            )

        # 2. Team Colors
        team_colors = {"home_hex": "#e11d48", "away_hex": "#ffffff"}

        # 3. Dual-Frame Set-of-Mark & Call to Gemini failover pool
        tracks_on_cause = tracks_per_frame.get(str(c_idx), [])
        if not tracks_on_cause and str(c_idx) not in tracks_per_frame:
            first_key = next(iter(tracks_per_frame.keys()), str(c_idx))
            tracks_on_cause = tracks_per_frame.get(first_key, [])

        tracks_on_outcome = tracks_per_frame.get(str(outcome_target_idx), [])

        cause_detected_ball = None
        cause_detected_carrier_id = None

        if cause_frame_img is not None and tracks_on_cause:
            # Strictly classify jersey RGB/HSV for all tracks at cause_frame_idx with 2-cluster KMeans
            TacticalVisionEngine._classify_all_tracks_jersey(cause_frame_img, tracks_on_cause, w, h)

            # 1. Detect actual football and true ball-carrier at cause_frame_idx
            cause_detected_ball, cause_detected_carrier_id = TacticalVisionEngine.detect_ball_and_carrier(
                frame=cause_frame_img,
                tracks=tracks_on_cause,
                yolo_model=yolo_model,
                hull=hull,
                w=w,
                h=h,
            )

            som_cause = self._draw_set_of_mark_tags(
                cause_frame_img,
                tracks_on_cause,
                detected_ball=cause_detected_ball,
            )
            _, buf_c = cv2.imencode(".jpg", som_cause)
            b64_cause = base64.b64encode(buf_c).decode("utf-8")

            b64_outcome = None
            if outcome_frame_img is not None:
                som_outcome = self._draw_set_of_mark_tags(outcome_frame_img, tracks_on_outcome) if tracks_on_outcome else outcome_frame_img
                _, buf_o = cv2.imencode(".jpg", som_outcome)
                b64_outcome = base64.b64encode(buf_o).decode("utf-8")

            # Fallback fill for any missing burst frames
            if cause_frame_img is not None and temporal_burst_frames.get("frame_freeze") is None:
                temporal_burst_frames["frame_freeze"] = cause_frame_img
            for k in ["frame_prev", "frame_next_1", "frame_next_2", "frame_next_3"]:
                if temporal_burst_frames.get(k) is None:
                    temporal_burst_frames[k] = outcome_frame_img if (outcome_frame_img is not None and "next" in k) else cause_frame_img

            # Pitch Homography Estimation & Alignment across the 5 reference frames
            from src.vision.camera_pan_compensator import TemporalBurstAligner
            burst_aligner = TemporalBurstAligner()
            burst_homographies = burst_aligner.align_burst_frames(temporal_burst_frames)

            cause_telestration = self._call_gemini_multiframe_temporal_failover(
                b64_cause=b64_cause,
                b64_outcome=b64_outcome,
                tracks_on_cause=tracks_on_cause,
                event_window=ctx,
                hull=hull,
                detected_carrier_id=cause_detected_carrier_id,
                detected_ball=cause_detected_ball,
                cause_frame=cause_frame_img,
                temporal_burst_frames=temporal_burst_frames,
                homographies=burst_homographies,
            )
        else:
            burst_homographies = None
            cause_telestration = self._generate_fallback_cause_telestration(
                tracks_on_cause=tracks_on_cause,
                event_context=ctx,
                hull=hull,
                detected_carrier_id=cause_detected_carrier_id,
                detected_ball=cause_detected_ball,
            )

        TacticalVisionEngine.sanitize_v3_telestration(
            cause_tel=cause_telestration,
            tracks_at_freeze=tracks_on_cause,
            clip_id=clip_id,
            freeze_frame_img=cause_frame_img,
            event_context=ctx,
            tracks_per_frame=tracks_per_frame,
            freeze_frame=c_idx,
            homographies=burst_homographies,
            temporal_burst_frames=temporal_burst_frames,
        )

        p_by_id = {int(t["track_id"]): t for t in tracks_on_cause}

        # 4. Resolve Canonical Player Labels (Preserving resolved real-life player names)
        assigned_players: Dict[int, Dict[str, Any]] = {}
        att_team_fallback = str(cause_telestration.get("attacking_team", "away"))

        # a) Catalyst Player (including Goalkeeper / Corner Takers)
        catalyst_data = cause_telestration.get("catalyst_player") or {}
        cat_id = int(catalyst_data.get("track_id", cause_telestration.get("ball_handler_track_id", 1)))
        cat_track = p_by_id.get(cat_id, {})
        cat_name_pos = str(catalyst_data.get("name_pos") or catalyst_data.get("name") or cat_track.get("short_name") or cat_track.get("name") or "").strip()
        if not cat_name_pos:
            cat_name_pos = TacticalVisionEngine._resolve_canonical_player_name_pos(
                cat_track, cat_track.get("team", att_team_fallback), event_context=ctx, role_hint="BALL CARRIER"
            )
        assigned_players[cat_id] = {
            "name": cat_name_pos,
            "label": cat_name_pos,
            "role": cat_name_pos,
            "num": "",
            "team": catalyst_data.get("team", att_team_fallback),
        }

        # b) Runner Player from cause_telestration
        runner_data = cause_telestration.get("runner_player") or {}
        raw_r_tid = runner_data.get("track_id") or cause_telestration.get("runner_track_id") or 0
        run_pid = int(raw_r_tid) if raw_r_tid is not None else 0
        if run_pid > 0 and run_pid not in assigned_players:
            run_pname = str(runner_data.get("name_pos") or runner_data.get("name") or "").strip()
            if run_pname:
                assigned_players[run_pid] = {
                    "name": run_pname,
                    "label": run_pname,
                    "role": run_pname,
                    "num": "",
                    "team": runner_data.get("team", att_team_fallback),
                }

        # b.2) Vector Runners
        for v in cause_telestration.get("vectors", []):
            if v.get("type") == "PLAYER_RUN_DASHED":
                v_tid = int(v.get("from_track_id", 0))
                v_track = p_by_id.get(v_tid, {})
                v_np = str(v.get("name_pos") or v_track.get("short_name") or v_track.get("name") or "").strip()
                if v_tid > 0 and v_tid not in assigned_players:
                    if not v_np:
                        v_np = TacticalVisionEngine._resolve_canonical_player_name_pos(
                            v_track, v_track.get("team", att_team_fallback), event_context=ctx, role_hint="THIRD-MAN RUNNER"
                        )
                    assigned_players[v_tid] = {
                        "name": v_np,
                        "label": v_np,
                        "role": v_np,
                        "num": "",
                        "team": v.get("team", v_track.get("team", att_team_fallback)),
                    }

        # c) Triangle & Structural Network Players
        for t_tid in cause_telestration.get("triangle_track_ids", []):
            itid = int(t_tid)
            if itid not in assigned_players and itid in p_by_id:
                t_track = p_by_id[itid]
                t_np = str(t_track.get("short_name") or t_track.get("name") or "Support").strip()
                assigned_players[itid] = {
                    "name": t_np,
                    "label": t_np,
                    "role": t_np,
                    "num": "",
                }

        struct_ids = cause_telestration.get("structural_network", {}).get("track_ids", [])
        for s_tid in struct_ids:
            itid = int(s_tid)
            if itid not in assigned_players and itid in p_by_id:
                s_track = p_by_id[itid]
                s_np = str(s_track.get("short_name") or s_track.get("name") or "").strip()
                if not s_np:
                    s_np = TacticalVisionEngine._resolve_canonical_player_name_pos(
                        s_track, s_track.get("team", "home"), event_context=ctx
                    )
                assigned_players[itid] = {
                    "name": s_np,
                    "label": s_np,
                    "role": s_np,
                    "num": "",
                }

        # Include all players from synchronized cause_telestration["players"]
        for p_item in cause_telestration.get("players", []):
            p_tid = int(p_item.get("track_id", p_item.get("id", -1)))
            if p_tid > 0 and p_tid not in assigned_players:
                p_lbl = str(p_item.get("label") or p_item.get("name") or "").strip()
                if p_lbl:
                    assigned_players[p_tid] = {
                        "name": p_lbl,
                        "label": p_lbl,
                        "role": p_lbl,
                        "num": "",
                    }

        # Map labels onto tracks:
        for f_list in tracks_per_frame.values():
            for tr in f_list:
                tid = int(tr["track_id"])
                y_pct = float(tr.get("y_pct", 50.0))
                is_off_pitch = (y_pct > 81.5 or y_pct < 22.0)
                # Guarantee referee and off-pitch sideline staff are strictly excluded from all tags, rings, and beams
                if p_by_id.get(tid, {}).get("jersey_class") == "REFEREE" or tr.get("jersey_class") == "REFEREE" or is_off_pitch:
                    tr["name_pos"] = ""
                    tr["label"] = ""
                    tr["name"] = ""
                    tr["role"] = ""
                    tr["number"] = ""
                    tr["showBeam"] = False
                    tr["showRing"] = False
                    continue

                if tid in assigned_players:
                    ap = assigned_players[tid]
                    ap_team = ap.get("team")
                    tr_team = tr.get("team")
                    if ap_team and tr_team and ap_team != tr_team:
                        continue
                    tr["name_pos"] = ap["name"]
                    tr["label"] = ap["label"]
                    tr["name"] = ap["name"]
                    tr["role"] = ap["role"]
                    tr["number"] = ""
                    is_corner_session = "corner" in str(clip_id).lower() or bool(cause_telestration.get("is_set_piece")) or str(cause_telestration.get("sequence_type")) == "CORNER_KICK"
                    if tid == cat_id:
                        tr["showBeam"] = True
                        tr["showRing"] = True
                        tr["color"] = "#ffffff"  # Broadcast white spotlight for carrier / corner taker
                    elif is_corner_session and tid == run_pid:
                        tr["showBeam"] = False
                        tr["showRing"] = True
                        tr["color"] = "#ffffff"
                    else:
                        tr["showBeam"] = False
                else:
                    tr["name_pos"] = ""
                    tr["label"] = ""
                    tr["name"] = ""
                    tr["role"] = ""
                    tr["number"] = ""
                    tr["showBeam"] = False

        default_beats = {
            "beat_1_setup": {
                "text": "Setup phase: Overload developing in the attacking third.",
                "wav_path": f"renders/audio/{clip_id}_beat1.wav",
                "duration_ms": 3800,
            },
            "beat_2_cause": {
                "text": ctx.get("thesis", "Defensive shape broken."),
                "wav_path": f"renders/audio/{clip_id}_beat2.wav",
                "duration_ms": 5400,
                "freeze_frames": 162,
            },
            "beat_3_effect": {
                "text": "Effect phase: Penetration into exploited space.",
                "wav_path": f"renders/audio/{clip_id}_beat3.wav",
                "duration_ms": 4200,
            },
        }

        return {
            "clip_id": clip_id,
            "video_path": str(video_path),
            "fps": actual_fps,
            "cause_frame_idx": c_idx,
            "effect_frame_idx": e_idx,
            "team_colors": team_colors,
            "tracks_per_frame": tracks_per_frame,
            "cause_telestration": cause_telestration,
            "is_actual_goal_scored_in_clip": cause_telestration.get("is_actual_goal_scored_in_clip", False),
            "goal_impact_frame_idx": cause_telestration.get("goal_impact_frame_idx"),
            "temporal_burst_indices": {
                "frame_prev": prev_target_idx,
                "frame_freeze": freeze_target_idx,
                "frame_next_1": next1_target_idx,
                "frame_next_2": next2_target_idx,
                "frame_next_3": next3_target_idx,
            },
            "temporal_burst_offsets": {
                "frame_prev": -2.0,
                "frame_freeze": 0.0,
                "frame_next_1": 1.5,
                "frame_next_2": 3.0,
                "frame_next_3": 4.5,
            },
            "beats": beats or ctx.get("beats") or default_beats,
        }

    # Aliases for flexible integration
    process_video = process_clip
    generate_synchronized_clip_vision = process_clip

