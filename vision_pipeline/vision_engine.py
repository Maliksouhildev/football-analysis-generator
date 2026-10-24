"""
Vision Engine for Tactical Football Analysis
Integrates Gemini Multimodal Vision with structured tactical extraction,
reverse-timeline paired frame processing (T1 -> T0), and dynamic model fallbacks.
"""

import base64
import json
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image
import requests

from vision_pipeline.roster_manager import RosterManager

# Supported and tested Gemini model fallback ladder
DEFAULT_MODELS = [
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash"
]

class VisionEngine:
    """Multi-modal Vision AI engine for football player identification & tactical analysis."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.5-flash",
        temperature: float = 0.1,
        roster_mgr: Optional[RosterManager] = None
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment or constructor.")
        self.model_name = model_name
        self.temperature = temperature
        self.roster_mgr = roster_mgr or RosterManager()

    def _encode_image(self, image_path: str) -> Tuple[str, str]:
        """Loads and encodes an image into base64 with mime type."""
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        with open(image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        return b64_data, mime

    def _call_gemini_vision(
        self,
        image_path: str,
        prompt: str,
        system_instruction: Optional[str] = None,
        model_override: Optional[str] = None,
        temp_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calls Gemini Vision API with structured JSON output and automated fallback ladder."""
        b64_data, mime = self._encode_image(image_path)
        temp = self.temperature if temp_override is None else temp_override
        models_to_try = [model_override] if model_override else [self.model_name] + [m for m in DEFAULT_MODELS if m != self.model_name]

        last_error = None
        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            payload: Dict[str, Any] = {
                "contents": [
                    {
                        "parts": [
                            {"inlineData": {"mimeType": mime, "data": b64_data}},
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": temp,
                    "maxOutputTokens": 4096
                }
            }
            if system_instruction:
                payload["systemInstruction"] = {
                    "parts": [{"text": system_instruction}]
                }

            try:
                resp = requests.post(url, json=payload, timeout=35)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            raw_text = parts[0]["text"].strip()
                            return json.loads(raw_text)
                elif resp.status_code in [404, 429, 503]:
                    last_error = f"Model {model} returned HTTP {resp.status_code}: {resp.text[:120]}"
                    time.sleep(1.0)
                    continue
                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as e:
                last_error = str(e)
                time.sleep(1.0)

        raise RuntimeError(f"All Gemini Vision models failed. Last error: {last_error}")

    def analyze_frame_t1(
        self,
        t1_image_path: str,
        fixture: Dict[str, Any],
        prompt_style: str = "chain_of_thought"
    ) -> Dict[str, Any]:
        """
        Analyzes outcome frame T1: identifies all players, kit colors,
        numbers, physical traits, pitch zones, ball position, and receiver/carrier.
        """
        home_name = fixture["home_team"]["name"]
        away_name = fixture["away_team"]["name"]
        home_kit = fixture["home_team"]["kit"]["shirt"]
        away_kit = fixture["away_team"]["kit"]["shirt"]

        prompt = f"""You are an elite tactical football vision analyst.
This is frame T1 (the climax/receiving frame) of a match between {home_name} (Kit: {home_kit}) and {away_name} (Kit: {away_kit}).

Analyze every player and the ball on the pitch with surgical precision.

Return a JSON object with this exact structure:
{{
  "pitch_context": {{
    "camera_angle": "broadcast / tactical high / close-up",
    "attacking_direction_home": "left-to-right / right-to-left"
  }},
  "ball": {{
    "detected": true/false,
    "position_norm": [x, y],  // 0 to 1000 scale (0,0 is top-left, 1000,1000 is bottom-right)
    "carrier_player_id": "P1" or null
  }},
  "players": [
    {{
      "id": "P1",
      "bbox_norm": [ymin, xmin, ymax, xmax],  // 0 to 1000 scale
      "foot_pos_norm": [x, y],               // Ground contact point for tactical ring
      "team": "{home_name}" or "{away_name}" or "Referee" or "Goalkeeper",
      "kit_color": "shirt color description",
      "jersey_number": 8 or null,            // Extracted digit on back/shorts or null if obscured
      "tactical_role": "GK/CB/LB/RB/DM/CM/CAM/LW/RW/ST",
      "physical_traits": "hair color/style, skin tone, build (e.g. blonde ponytail tall, dark fade agile)",
      "is_ball_carrier": true/false,
      "tactical_action": "receiving pass / pressing / defending / making run"
    }}
  ]
}}"""

        raw_result = self._call_gemini_vision(t1_image_path, prompt)

        # Enhance player detections with RosterManager resolution
        resolved_players = []
        for p in raw_result.get("players", []):
            if p.get("team") == "Referee":
                p["resolved"] = {"name": "Referee", "display_name": "Referee", "number": None, "confidence": 100.0}
            else:
                resolution = self.roster_mgr.resolve_player(p, fixture)
                p["resolved"] = resolution
            resolved_players.append(p)

        raw_result["players"] = resolved_players
        return raw_result

    def analyze_frame_t0(
        self,
        t0_image_path: str,
        t1_analysis: Dict[str, Any],
        fixture: Dict[str, Any],
        prompt_style: str = "chain_of_thought"
    ) -> Dict[str, Any]:
        """
        Analyzes setup frame T0 (preceding T1 in match timeline):
        Identifies current ball carrier in T0 and predicts where the ball will be passed,
        the pass trajectory vector, and intended recipient.
        """
        home_name = fixture["home_team"]["name"]
        away_name = fixture["away_team"]["name"]
        home_kit = fixture["home_team"]["kit"]["shirt"]
        away_kit = fixture["away_team"]["kit"]["shirt"]

        # Context from confirmed T1 outcome
        t1_ball = t1_analysis.get("ball", {})
        t1_carrier = None
        for p in t1_analysis.get("players", []):
            if p.get("is_ball_carrier"):
                t1_carrier = p
                break

        t1_context_note = ""
        if t1_carrier:
            res = t1_carrier.get("resolved", {})
            t1_context_note = (
                f"TACTICAL GROUND TRUTH: In the subsequent frame T1, the ball reaches {res.get('name', 'player')} "
                f"({res.get('position', '')}) at normalized pitch coordinates {t1_ball.get('position_norm', [500, 500])}."
            )

        prompt = f"""You are an elite tactical football vision analyst.
This is frame T0 (the setup/buildup frame, occurring 1-2 seconds BEFORE frame T1) between {home_name} and {away_name}.
{t1_context_note}

Analyze this setup frame T0:
1. Locate all active players and the ball.
2. Identify the CURRENT ball carrier in T0 (who has possession or is striking the ball).
3. Predict the pass trajectory vector: where is the ball being played, to whom, and through which tactical channel?
4. Identify open passing space / defensive gaps.

Return a JSON object with this exact structure:
{{
  "pitch_context": {{
    "game_phase": "buildup / counter-attack / attacking transition / defensive block"
  }},
  "ball": {{
    "detected": true/false,
    "position_norm": [x, y],
    "carrier_player_id": "P1" or null
  }},
  "players": [
    {{
      "id": "P1",
      "bbox_norm": [ymin, xmin, ymax, xmax],
      "foot_pos_norm": [x, y],
      "team": "{home_name}" or "{away_name}" or "Referee" or "Goalkeeper",
      "kit_color": "shirt color description",
      "jersey_number": 16 or null,
      "tactical_role": "GK/CB/LB/RB/DM/CM/CAM/LW/RW/ST",
      "physical_traits": "hair style, skin tone, build",
      "is_ball_carrier": true/false,
      "tactical_action": "passing / dribbling / scanning / overlapping run"
    }}
  ],
  "pass_prediction": {{
    "passer_player_id": "P1",
    "passer_name_predicted": "Pedri",
    "predicted_receiver_id": "P2",
    "predicted_receiver_name": "Ferran Torres",
    "pass_origin_norm": [x, y],
    "pass_destination_norm": [x, y],
    "pass_type": "through ball / ground pass / lofted switch / cutback",
    "tactical_intent": "breaking defensive line / finding space between lines",
    "confidence_percent": 94.0
  }},
  "tactical_space": {{
    "space_created_polygon_norm": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    "description": "pocket of space behind defensive line"
  }}
}}"""

        raw_result = self._call_gemini_vision(t0_image_path, prompt)

        # Enhance player detections with RosterManager resolution
        resolved_players = []
        for p in raw_result.get("players", []):
            if p.get("team") == "Referee":
                p["resolved"] = {"name": "Referee", "display_name": "Referee", "number": None, "confidence": 100.0}
            else:
                resolution = self.roster_mgr.resolve_player(p, fixture)
                p["resolved"] = resolution
            resolved_players.append(p)

        raw_result["players"] = resolved_players
        return raw_result

    def analyze_pair(
        self,
        t0_image_path: str,
        t1_image_path: str,
        match_name: str = "Galatasaray vs Barcelona",
        model_override: Optional[str] = None,
        temp_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes reverse-timeline analysis:
        1. Analyzes T1 (outcome)
        2. Analyzes T0 (preceding setup with pass prediction)
        3. Computes cross-frame tracking and prediction metrics
        """
        fixture = self.roster_mgr.get_fixture_data(match_name)

        t1_start = time.time()
        t1_analysis = self.analyze_frame_t1(t1_image_path, fixture)
        t1_duration = time.time() - t1_start

        t0_start = time.time()
        t0_analysis = self.analyze_frame_t0(t0_image_path, t1_analysis, fixture)
        t0_duration = time.time() - t0_start

        # Compute validation metrics
        metrics = self._evaluate_reverse_timeline_accuracy(t0_analysis, t1_analysis)

        return {
            "match": match_name,
            "t1_outcome_analysis": t1_analysis,
            "t0_setup_analysis": t0_analysis,
            "metrics": metrics,
            "timing": {
                "t1_duration_s": round(t1_duration, 2),
                "t0_duration_s": round(t0_duration, 2),
                "total_duration_s": round(t1_duration + t0_duration, 2)
            }
        }

    def _evaluate_reverse_timeline_accuracy(
        self,
        t0_analysis: Dict[str, Any],
        t1_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validates pass prediction and player identification consistency:
        - Compares predicted pass destination in T0 against actual T1 ball position.
        - Calculates Euclidean distance error in normalized 0-1000 space.
        - Computes player identification confidence distribution.
        """
        pass_pred = t0_analysis.get("pass_prediction", {})
        pred_dest = pass_pred.get("pass_destination_norm", [500, 500])

        t1_ball = t1_analysis.get("ball", {})
        actual_dest = t1_ball.get("position_norm", [500, 500])

        dx = pred_dest[0] - actual_dest[0]
        dy = pred_dest[1] - actual_dest[1]
        distance_error = (dx * dx + dy * dy) ** 0.5

        # Maximum field diagonal is sqrt(1000^2 + 1000^2) = 1414
        normalized_error = min(1.0, distance_error / 1414.2)
        spatial_pass_accuracy = max(0.0, round((1.0 - normalized_error) * 100.0, 1))

        # Check player confidence scores
        confidences = [
            p.get("resolved", {}).get("confidence", 50.0)
            for p in t0_analysis.get("players", []) + t1_analysis.get("players", [])
            if p.get("resolved")
        ]
        avg_confidence = round(sum(confidences) / max(1, len(confidences)), 1)

        return {
            "spatial_pass_accuracy_percent": spatial_pass_accuracy,
            "euclidean_distance_error_norm": round(distance_error, 1),
            "average_player_id_confidence": avg_confidence,
            "t0_players_detected": len(t0_analysis.get("players", [])),
            "t1_players_detected": len(t1_analysis.get("players", [])),
            "pass_type": pass_pred.get("pass_type", "ground pass")
        }
