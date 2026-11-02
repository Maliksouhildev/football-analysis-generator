"""
src/rendering/tactical_narrator.py
AI Tactical Script Generator for 3-Beat TTS Lockstep Commentary.

Generates 3 locked beats per tactical clip:
- beat_1_setup: Moving pre-roll (frame 0 to cause_frame_idx)
- beat_2_cause: Spoken while video FREEZES at cause_frame_idx, explicitly referencing visual telestrations on screen
- beat_3_effect: Spoken as video UNFREEZES and consequence unfolds (cause_frame_idx to effect_frame_idx + post-roll)
Strictly adheres to narrative_tone ('RUTHLESS_AUTOPSY' vs 'TACTICAL_MASTERCLASS') and thesis.
"""

from __future__ import annotations
import copy

import base64
import json
import logging
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger("rendering.tactical_narrator")

SpeedRegimeType = Literal["FAST_TRANSIT", "NORMAL_FLOW", "SLOW_FOCUS", "FREEZE_HOLD"]


def _normalize_name(name: str) -> str:
    if not name:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_str = nfkd.encode("ASCII", "ignore").decode("utf-8")
    return ascii_str.lower().strip()


KNOWN_CLUB_ROSTERS: Dict[str, Dict[str, Any]] = {
    "FC Barcelona": {
        "aliases": ["barcelona", "fc barcelona", "barca", "barça"],
        "players": [
            {"name": "Robert Lewandowski", "short": "Lewandowski", "number": 9, "role": "ST"},
            {"name": "Lamine Yamal", "short": "Lamine Yamal", "number": 19, "role": "RW"},
            {"name": "Raphinha", "short": "Raphinha", "number": 11, "role": "LW"},
            {"name": "Pedri", "short": "Pedri", "number": 8, "role": "CM"},
            {"name": "Fermín López", "short": "Fermín López", "number": 16, "role": "AM"},
            {"name": "Marc Casadó", "short": "Casadó", "number": 17, "role": "DM"},
            {"name": "Pau Cubarsí", "short": "Cubarsí", "number": 2, "role": "CB"},
            {"name": "Iñigo Martínez", "short": "Iñigo Martínez", "number": 5, "role": "CB"},
            {"name": "Jules Koundé", "short": "Koundé", "number": 23, "role": "RB"},
            {"name": "Alejandro Balde", "short": "Balde", "number": 3, "role": "LB"},
            {"name": "Iñaki Peña", "short": "Iñaki Peña", "number": 13, "role": "GK"},
            {"name": "Frenkie de Jong", "short": "De Jong", "number": 21, "role": "CM"},
            {"name": "Gavi", "short": "Gavi", "number": 6, "role": "CM"},
            {"name": "Dani Olmo", "short": "Dani Olmo", "number": 20, "role": "AM"},
            {"name": "Ferran Torres", "short": "Ferran Torres", "number": 7, "role": "FW"},
            {"name": "Ansu Fati", "short": "Ansu Fati", "number": 10, "role": "LW"},
        ],
        "default_starters": ["Lewandowski", "Lamine Yamal", "Raphinha", "Pedri", "Fermín López", "Casadó", "Cubarsí", "Iñigo Martínez", "Koundé", "Balde", "Iñaki Peña"],
        "star_attacker": "Lewandowski",
        "star_creator": "Pedri",
        "goalkeeper": "Iñaki Peña",
    },
    "Real Madrid": {
        "aliases": ["real madrid", "madrid", "real", "los blancos"],
        "players": [
            {"name": "Andriy Lunin", "short": "Lunin", "number": 13, "role": "GK"},
            {"name": "Thibaut Courtois", "short": "Courtois", "number": 1, "role": "GK"},
            {"name": "Kylian Mbappé", "short": "Mbappé", "number": 9, "role": "ST"},
            {"name": "Vinícius Júnior", "short": "Vinícius", "number": 7, "role": "LW"},
            {"name": "Jude Bellingham", "short": "Bellingham", "number": 5, "role": "AM"},
            {"name": "Federico Valverde", "short": "Valverde", "number": 8, "role": "CM"},
            {"name": "Aurélien Tchouaméni", "short": "Tchouaméni", "number": 14, "role": "DM"},
            {"name": "Eduardo Camavinga", "short": "Camavinga", "number": 6, "role": "CM"},
            {"name": "Antonio Rüdiger", "short": "Rüdiger", "number": 22, "role": "CB"},
            {"name": "Éder Militão", "short": "Militão", "number": 3, "role": "CB"},
            {"name": "Lucas Vázquez", "short": "Lucas Vázquez", "number": 17, "role": "RB"},
            {"name": "Ferland Mendy", "short": "Mendy", "number": 23, "role": "LB"},
            {"name": "Luka Modrić", "short": "Modrić", "number": 10, "role": "CM"},
            {"name": "Rodrygo", "short": "Rodrygo", "number": 11, "role": "RW"},
            {"name": "Brahim Díaz", "short": "Brahim", "number": 21, "role": "AM"},
            {"name": "Dani Carvajal", "short": "Carvajal", "number": 2, "role": "RB"},
        ],
        "default_starters": ["Mbappé", "Vinícius", "Bellingham", "Valverde", "Tchouaméni", "Camavinga", "Rüdiger", "Militão", "Lucas Vázquez", "Mendy", "Lunin"],
        "star_attacker": "Mbappé",
        "star_creator": "Vinícius",
        "goalkeeper": "Lunin",
    },
}

PHANTOM_REPLACEMENTS: Dict[str, Dict[str, str]] = {
    "pisilli": {"Real Madrid": "Camavinga", "FC Barcelona": "Casadó"},
    "malen": {"Real Madrid": "Rodrygo", "FC Barcelona": "Ferran Torres"},
    "dumfries": {"Real Madrid": "Lucas Vázquez", "FC Barcelona": "Koundé"},
    "konaté": {"Real Madrid": "Rüdiger", "FC Barcelona": "Cubarsí"},
    "konate": {"Real Madrid": "Rüdiger", "FC Barcelona": "Cubarsí"},
    "ndicka": {"Real Madrid": "Militão", "FC Barcelona": "Iñigo Martínez"},
    "cucurella": {"Real Madrid": "Mendy", "FC Barcelona": "Balde"},
    "gordon": {"Real Madrid": "Rodrygo", "FC Barcelona": "Raphinha"},
    "cancelo": {"Real Madrid": "Lucas Vázquez", "FC Barcelona": "Balde"},
    "adeyemi": {"Real Madrid": "Mbappé", "FC Barcelona": "Lamine Yamal"},
    "jesus": {"Real Madrid": "Mbappé", "FC Barcelona": "Lewandowski"},
    "huijsen": {"Real Madrid": "Militão", "FC Barcelona": "Cubarsí"},
    "silva": {"Real Madrid": "Modrić", "FC Barcelona": "Pedri"},
    "bernardo": {"Real Madrid": "Modrić", "FC Barcelona": "Pedri"},
}

JARGON_ALTERNATIVES: Dict[str, List[str]] = {
    "flank corridor": ["wide flank", "touchline channel", "flank space"],
    "corridor": ["channel", "half-space", "pocket", "inside lane"],
    "spatial overload": ["numerical advantage", "passing combination", "two-versus-one combination"],
    "overload": ["numerical advantage", "combination play"],
    "defensive block": ["defensive unit", "compact shape", "backline structure"],
    "tactical void": ["open turf", "vacated space", "unattended pocket"],
    "structural collapse": ["defensive fracture", "marking breakdown", "split in the backline"],
    "structural decay": ["breakdown in spacing", "defensive misalignment", "lapse in recovery"],
}

CLICHE_REPLACEMENTS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'\bClinical finish buries past the helpless keeper\b', re.I), "A clinical finish beats the goalkeeper into the corner"),
    (re.compile(r'\bClinical finish buries the chance past helpless goalkeeper\b', re.I), "A precise strike finds the corner of the net"),
    (re.compile(r'\bLate collapse exposes\b', re.I), "The defensive unit is caught out of position"),
    (re.compile(r'\bTotal defensive breakdown\b', re.I), "A momentary lapse in defensive shape"),
    (re.compile(r'\bfatal structural neglect\b', re.I), "a momentary fracture in the backline"),
    (re.compile(r'\bTotal structural decay\b', re.I), "A breakdown in team spacing"),
    (re.compile(r'\bFatal right flank structural defensive collapse\b', re.I), "The defensive unit is stretched wide across the right flank"),
    (re.compile(r'\bEntire defense exposed by basic flank overloads\b', re.I), "The defensive line is stretched across the pitch"),
    (re.compile(r'\bPrimary tactical receiver\b', re.I), "the attacking runner"),
    (re.compile(r'\bUnfreezing into motion,?\s*', re.I), "In real time, "),
    (re.compile(r'\bUnfreezing,?\s*', re.I), "As play resumes, "),
    (re.compile(r'\bReleasing into motion,?\s*', re.I), "In real time, "),
    (re.compile(r'\busing precise telemetry,?\s*', re.I), ""),
    (re.compile(r'\bwhile commands? high ratings?\b', re.I), "with sharp composure"),
    (re.compile(r'\bbacked by\s+\d+(\.\d+)?\s+rating\s+and\s+', re.I), "registering "),
    (re.compile(r'\bbacked by\s+\d+(\.\d+)?\s+rating\b', re.I), "showing high tactical efficiency"),
    (re.compile(r'\b\d+(\.\d+)?\s+rating\b', re.I), "commanding form"),
    (re.compile(r'\bMatch Telemetry (?:&|and)?\s*Heatmap\b', re.I), ""),
    (re.compile(r'\bTelemetry Heatmap\b', re.I), ""),
    (re.compile(r'\bMatch Telemetry:?\s*', re.I), ""),
    (re.compile(r'\bas seen on the heatmap\b', re.I), ""),
    (re.compile(r'\baccording to the (?:rating box|telemetry)\b', re.I), ""),
    (re.compile(r'\bPHASE:\s*[A-Z\s_-]+\b', re.I), ""),
]



class SpeedSegment(BaseModel):
    segment_type: SpeedRegimeType = Field(..., description="Classification: FAST_TRANSIT, NORMAL_FLOW, SLOW_FOCUS, or FREEZE_HOLD")
    rate: float = Field(..., description="Playback rate: 1.25 to 1.35 for FAST_TRANSIT, 1.00 for NORMAL_FLOW, 0.60 to 0.75 for SLOW_FOCUS, 0.00 for FREEZE_HOLD")
    source_start_sec: float = Field(..., description="Source video start timestamp in seconds")
    source_end_sec: float = Field(..., description="Source video end timestamp in seconds")
    duration_sec: Optional[float] = Field(None, description="Output duration in seconds")
    start_frame: Optional[int] = Field(None, description="Timeline start frame index")
    end_frame: Optional[int] = Field(None, description="Timeline end frame index")
    startFrame: Optional[int] = Field(None, description="Timeline start frame index (camelCase)")
    endFrame: Optional[int] = Field(None, description="Timeline end frame index (camelCase)")
    output_frames: Optional[int] = Field(None, description="Total output frames")
    reason: Optional[str] = Field(None, description="Tactical rationale for speed ramping")


class TacticalStoryboardSchema(BaseModel):
    beat_1_setup: str = Field(..., description="Spoken voiceover for moving pre-roll (~1.8 to 2.2s)")
    beat_2_cause: str = Field(..., description="Spoken voiceover for freeze frame (~6.0 to 7.5s)")
    beat_2_steps: Optional[Dict[str, str]] = Field(None, description="4-step analyst breakdown")
    beat_3_effect: str = Field(..., description="Spoken voiceover for unfreezing post-roll (~2.5s)")
    speed_segments: List[SpeedSegment] = Field(default_factory=list, description="Continuous speed ramping segments covering the clip")


PROMPT_3BEAT_TEMPLATE = """You are a World-Class UEFA Champions League Tactical Video Analyst for television broadcast (in the style of Jamie Carragher and Thierry Henry on CBS Sports).
Analyze this tactical freeze-frame for {home_team} vs {away_team}.

=== AUTHORITATIVE MATCH ROSTER DICTIONARY ===
{roster_firewall_context}

Event Telemetry Context:
- Tone: {narrative_tone} ({tone_instruction})
- Tactical Zone: {tactical_zone}
- Core Thesis: {thesis}
- Crowd / Match Evidence: {crowd_evidence}
- Match Clock: {clock}
- Telestration Elements Visible On Screen:
  * Exploit Zone / Vacated Space: {vacated_label}
  * Defensive Backline / Structural Shape: {backline_info}
  * Attacking Shape / Overload: {triangle_info}
  * Key Players In Spotlight: {focal_info}
  * Ball Carrier / Catalyst In Focus: {catalyst_info} ({catalyst_team_name})
  * Target / Supporting Teammate In Focus: {runner_info}
  * Opponent Presser / Challenger In Focus: {opponent_info}
  * Verified Players On Pitch: {available_players_info}
- Empirical Match Actions & Form (NO STATIC RATINGS):
{player_telemetry_context}

Speed Regimes for Dynamic Camera & Time-Mapping:
Between freeze-frames, divide the continuous clip timeline into speed_segments classified into 4 subtle speed regimes:
- "FAST_TRANSIT" (rate = 1.25 to 1.35): Ball traveling across long pass, uncontested turf, or low-information transitions.
- "NORMAL_FLOW" (rate = 1.00): Standard live broadcast speed.
- "SLOW_FOCUS" (rate = 0.60 to 0.75): Subtle cinematic deceleration as player executes press-breaking turn, tight dribble, crucial defensive mistake, or final deflection/shot.
- "FREEZE_HOLD" (rate = 0.00, duration_sec = 1.5 to 2.5): Full tactical pause where spotlights and HUD badges animate in at cause_frame_idx.

TASK:
Write exactly 3 distinct spoken voiceover commentary beats and continuous speed_segments in valid JSON format.
During the video FREEZE (beat_2_cause), you MUST follow the 4-Step Analyst Framework:
- Step 1 (Catalyst): Identify who has the ball or triggers the action.
- Step 2 (Geometric Advantage / Shape): Explain the linked defensive line or numerical advantage shape.
- Step 3 (Exploit Zone): Explain the dangerous vacated space that opened up on the turf.
- Step 4 (Vectors & Decision): Explain the pass/run vectors and explicitly state the actual decision taken.

Output schema:
{{
  "beat_1_setup": "<ultra-punchy 5 to 7 words MAX spoken during moving pre-roll (~1.8 to 2.2s max), tailored to {tactical_zone} and minute {clock}>",
  "beat_2_cause": "<18 to 24 words MAXIMUM TOTAL (~6.0 to 7.5 seconds total across all 4 steps combined—just 1 or 2 crisp, fast-paced sentences covering Step 1 -> Step 2 -> Step 3 -> Step 4>",
  "beat_2_steps": {{
    "step_1": "<Step 1: 4 to 6 words identifying catalyst player>",
    "step_2": "<Step 2: 4 to 6 words explaining structure or shape>",
    "step_3": "<Step 3: 4 to 6 words explaining exploitable space>",
    "step_4": "<Step 4: 4 to 6 words explaining vectors and decision taken>"
  }},
  "beat_3_effect": "<8 to 12 words maximum spoken as video UNFREEZES (~2.5s)>",
  "speed_segments": [
    {{
      "segment_type": "FAST_TRANSIT",
      "rate": 1.30,
      "source_start_sec": 0.0,
      "source_end_sec": 2.2,
      "duration_sec": 1.69,
      "reason": "Long ball transit into attacking half-space"
    }}
  ]
}}

CRITICAL BROADCAST & TACTICAL INTEGRITY DIRECTIVES:

1. STRICT TEAM FIREWALL & ZERO CROSS-TEAM PASSING (MANDATORY):
   - You are strictly bound by the team affiliations in the Authoritative Match Roster Dictionary above.
   - An attacking player can ONLY pass, cross, combine with, or assist teammates of their OWN club!
   - Under NO circumstances can a player pass to or assist an opposing player (e.g. Real Madrid's Courtois or Tchouaméni can NEVER pass to Barcelona's Lewandowski; Barcelona's Pedri can NEVER pass to Real Madrid's Mbappé).
   - If the ball transitions between teams, it is EXCLUSIVELY an interception, tackle, defensive block, clearance, turnover, or save—NEVER a pass or assist!
   - Zero self-contradiction: Never claim a player assisted a goal if they were committing a foul or making a tackle.

2. ZERO PHANTOM PLAYERS:
   - ONLY name players who appear in the Authoritative Match Roster Dictionary above.
   - NEVER hallucinate or mention players who do not play for either club in this fixture (strictly forbidden: Pisilli, Malen, Dumfries, Konaté, Ndicka unless explicitly listed in the roster above).

3. BAN "BACKED BY..." AND ROBOTIC MECHANICAL CLICHÉS:
   - BANNED OPENERS: Never start any clause or sentence with "Backed by...", "backed by...", or "backed by [stat]".
   - BANNED MECHANICAL PHRASING:
     * "Unfreezing into motion" / "Unfreezing," / "Releasing into motion"
     * "using precise telemetry"
     * "Clinical finish buries past the helpless keeper"
     * "Late collapse exposes..." / "Total defensive breakdown" / "fatal structural neglect"
     * "while commands high ratings"
   - NATURAL STATISTICAL INTEGRATION:
     Weave statistics and player form seamlessly into physical football actions:
     * "Tchouaméni, who rarely misplaces a pass with ninety-three percent accuracy, spots the opening..."
     * "Fermín López, winning three defensive recoveries with aggressive pressing, snaps into the challenge..."
     * "Mbappé isolates his marker with electric acceleration..."
   - NEVER read out static video game ratings (e.g. "7.2 rating", "8.5 rating", "75 rating"). Translate form into dynamic pitch performance (e.g., "commanding the midfield tempo", "dominant in aerial duels", "pinpoint distribution").

4. TACTICAL JARGON BUDGET (MAXIMUM 1 PER CLIP):
   - Limit tactical buzzwords ("structural collapse", "structural decay", "tactical void", "spatial overload", "corridor", "flank corridor", "defensive block", "overload") to a MAXIMUM of ONE occurrence across the entire clip.
   - Use diverse physical football terminology:
     * Spatial terms: channel, half-space, pocket, inside lane, touchline, wide flank, dangerous pocket.
     * Technical actions: weighted through-ball, first-time delivery, disguised pass, line-breaking ball, chipped cross, sharp cutback.
     * Tactical movement: blindside run, decoy sprint, cover shadow, recovery run, pressing angle, compact defensive shape.

5. SPATIAL ACCURACY & ZERO GRAPHIC READING:
   - NEVER read UI overlays aloud as if they were physical match events.
     Do NOT speak "Match Telemetry Heatmap", "Heatmap", "PHASE: DEEP BUILD-UP CIRCULATION", or HUD banners.
   - Visual cue tags like [match_heatmap:Player] and [highlight:Player] are invisible markers for the video engine. The voiceover text around them must describe physical football action on the pitch.
   - Respect spatial reality: If defenders are in close proximity, describe tight marking and intense pressure—NEVER claim there is a "vacant corridor" when markers are right next to the carrier.

6. STRICT WORD BUDGETS:
   - beat_1_setup: ultra-punchy 5 to 7 words MAX (~1.8 to 2.2 seconds max).
   - beat_2_cause (freeze-frame narration): 18 to 24 words MAX total across all 4 steps combined (~6.0 to 7.5s, 1-2 crisp sentences). A tactical freeze-frame must NEVER exceed 6 to 8 seconds!
   - beat_3_effect: 8 to 12 words max (~2.5 seconds).
7. Every clip must receive a completely distinct, non-repetitive sentence structure and opener tailored to its tactical zone ({tactical_zone}) and match minute ({clock}). Never repeat openers across clips.
8. No emojis, no hashtags, no conversational filler (no "Here we see", no "Notice how").
9. Output ONLY the valid JSON object.
"""

CANONICAL_PROMPT_TEMPLATE = """You are a World-Class UEFA Champions League Tactical Video Analyst for live television broadcast (in the style of Jamie Carragher and Thierry Henry on CBS Sports).
Analyze this exact match freeze-frame keyframe for Minute {minute}' of {home_team} vs {away_team}.

Event Telemetry Context:
- Team in Focus: {team}
- Key Player: {player}
- Tactical Phase: {event_type}
- Match Situation: {description}

TASK:
Write a punchy, broadcast-ready 1-2 sentence tactical analysis (18 to 25 words maximum) explaining:
1. What happened at this exact moment (e.g. who made an error, lost possession, broke the press, or scored).
2. The tactical breakdown (broken defensive shape, midfield turnover, blindside run, passing lane cut).
3. The immediate consequence leading to the chance or goal.

CRITICAL BROADCAST RULES:
- Never say "Backed by...", "Unfreezing into motion", "using precise telemetry", or read static ratings aloud.
- Real player names only from this fixture; never invent phantom players.
- Teammates only pass to teammates; opposition actions are tackles, turnovers, or blocks.
- Do NOT use emojis, hashtags, or conversational padding (no "Here we see", no "Notice how").
- Speak with high-intensity television commentary authority.
- Output ONLY the spoken voiceover commentary text.
"""


class TacticalNarrator:
    """Generates 3-beat broadcast commentary lockstep scripts and keyframe tactical analysis."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent"
        self._circuit_open = False
        self._circuit_opened_at = 0.0
        self._cooldown = 120.0

    @staticmethod
    def extract_cue_tags(raw_text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extracts tactical visual cue tags like [candidate_passes], [highlight:Konaté], [stat_card:Konaté].
        Returns (cleaned_spoken_text, cue_tags_list).
        For [highlight:Player], if the player name is used directly as the grammatical noun in place of
        the tag (i.e. not already spoken in the surrounding sentence or upcoming clause), it is preserved
        so voiceover commentary remains grammatically complete without stuttering or name duplication.
        """
        if not raw_text or not isinstance(raw_text, str):
            return "", []

        tag_pattern = re.compile(r'\[([a-zA-Z0-9_]+)(?::([^\]]+))?\]')
        tags: List[Dict[str, Any]] = []

        last_end = 0
        clean_parts: List[str] = []
        current_char_offset = 0

        raw_matches = list(tag_pattern.finditer(raw_text))

        for idx, match in enumerate(raw_matches):
            preceding_text = raw_text[last_end:match.start()]
            following_text = raw_text[match.end():(raw_matches[idx + 1].start() if idx + 1 < len(raw_matches) else len(raw_text))]
            clean_parts.append(preceding_text)
            current_char_offset += len(preceding_text)

            tag_type = match.group(1).lower().strip()
            tag_param = match.group(2).strip() if match.group(2) else ""

            words_before = len("".join(clean_parts).split())

            tags.append({
                "type": tag_type,
                "target": tag_param,
                "char_offset": current_char_offset,
                "word_index": words_before,
                "raw": match.group(0),
            })

            # For [highlight:Player], preserve player name ONLY if not already present in surrounding context
            if tag_type in ("highlight", "player") and tag_param:
                # Do not insert pure numbers (#16) or role abbreviations (CB) as spoken nouns
                is_number_or_role = bool(re.match(r'^#?\d+$', tag_param)) or (len(tag_param) <= 3 and tag_param.upper() in ("CB", "LB", "RB", "GK", "CDM", "CM", "CAM", "RW", "LW", "ST", "CF"))
                if not is_number_or_role:
                    clean_target = re.sub(r'[^a-zA-Z0-9\s]', '', tag_param.lower())
                    target_words = [w for w in clean_target.split() if len(w) >= 3]

                    # Check preceding 10 words
                    prev_words = [re.sub(r'[^a-zA-Z0-9]', '', w.lower()) for w in preceding_text.split()[-10:]]
                    already_in_prev = any(tw in prev_words for tw in target_words) if target_words else False

                    # Check following 6 words
                    foll_words = [re.sub(r'[^a-zA-Z0-9]', '', w.lower()) for w in following_text.split()[:6]]
                    already_in_foll = any(tw in foll_words for tw in target_words) if target_words else False

                    if not already_in_prev and not already_in_foll:
                        clean_parts.append(tag_param)
                        current_char_offset += len(tag_param)

            last_end = match.end()

        clean_parts.append(raw_text[last_end:])
        cleaned = re.sub(r'\s+([.,!?;:])', r'\1', "".join(clean_parts))
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        final_words = cleaned.split()
        total_words = len(final_words)
        for t in tags:
            t["total_words"] = total_words
            t["word_index"] = min(t["word_index"], total_words)

        # Auto-detect tactical mistake keywords if not already tagged
        negative_keywords = {
            "wrong", "dangerous", "danger", "stupid", "fail", "fails", "failed", "failing", "failure",
            "mistake", "mistakes", "mistaken", "error", "errors", "poor", "poorer", "poorly",
            "costly", "disaster", "disastrous", "blunder", "blunders", "disjointed",
            "punished", "punish", "punishes", "punishing", "exposed", "exposes", "exposing", "exposure",
            "vulnerable", "vulnerability", "misplaced", "misplace", "misplaces",
            "turnover", "lost", "loses", "losing", "struggle", "struggles", "struggling",
            "trapped", "trap", "trapping", "caught", "hesitant", "hesitated", "hesitation",
            "sloppy", "fatal", "unforgivable", "collapse", "collapses", "collapsing", "collapsed",
            "crumbles", "crumbling", "crumbled", "broken", "breakdown", "fragmented", "isolated",
        }
        existing_mistake_indices = {t["word_index"] for t in tags if t.get("type") in ("mistake_flash", "red_flash", "mistake")}
        last_tagged_w_idx = -99
        for w_idx, w in enumerate(final_words):
            clean_w = re.sub(r'[^a-zA-Z0-9]', '', w.lower())
            if clean_w in negative_keywords:
                # Minimum spacing of 4 words between auto-detected mistake tags
                if w_idx not in existing_mistake_indices and (w_idx - last_tagged_w_idx) >= 4:
                    tags.append({
                        "type": "mistake_flash",
                        "target": clean_w,
                        "word_index": w_idx,
                        "total_words": total_words,
                        "raw": f"[mistake_flash:{clean_w}]",
                    })
                    last_tagged_w_idx = w_idx

        return cleaned, tags

    def generate_3beat_script(
        self,
        event_window: Dict[str, Any],
        cause_telestration: Optional[Dict[str, Any]] = None,
        freeze_frame_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generates 3 distinct script beats per clip matching TACTICAL_CONTRACT.md:
        - beat_1_setup: Moving pre-roll (0 to cause_frame_idx)
        - beat_2_cause: Spoken while video FREEZES at cause_frame_idx (referencing on-screen graphics)
        - beat_3_effect: Spoken as video UNFREEZES and consequence unfolds
        """
        tone = str(event_window.get("narrative_tone", "RUTHLESS_AUTOPSY")).strip().upper()
        if "MASTERCLASS" in tone:
            tone = "TACTICAL_MASTERCLASS"
        else:
            tone = "RUTHLESS_AUTOPSY"

        thesis = str(event_window.get("thesis", "")).strip()
        crowd_evidence = str(event_window.get("crowd_evidence", "")).strip()
        
        # Robust extraction of match clock without defaulting to hardcoded 84:00
        t_cause_clock = ""
        for ck in ("t_cause_clock", "clock", "timestamp", "match_clock", "time"):
            val = event_window.get(ck)
            if val and str(val).strip():
                t_cause_clock = str(val).strip()
                break
        if not t_cause_clock:
            if "minute" in event_window or "match_minute" in event_window:
                m_val = event_window.get("minute", event_window.get("match_minute"))
                try:
                    t_cause_clock = f"{int(m_val):02d}:00"
                except (ValueError, TypeError):
                    t_cause_clock = ""
        if not t_cause_clock and "t_effect_clock" in event_window:
            t_cause_clock = str(event_window.get("t_effect_clock", "")).strip()
        active_roster = event_window.get("active_roster", {})

        tel = cause_telestration or {}
        vacated_label = str(tel.get("vacated_label", "VACATED ZONE")).strip().upper()
        tactical_zone = self._resolve_tactical_zone(event_window, tel)
        clip_id_str = str(event_window.get("clip_id", "") or tel.get("clip_id", "")).strip()

        # Resolve authoritative home and away team identities
        home_team = str(event_window.get("home_team") or tel.get("home_team") or "").strip()
        away_team = str(event_window.get("away_team") or tel.get("away_team") or "").strip()
        if not home_team or home_team.lower() in ("home", "home team"):
            combined_t = f"{clip_id_str} {thesis} {event_window.get('match_id', '')}".lower()
            if any(k in combined_t for k in ("barca", "barcelona", "clasico", "real madrid")):
                home_team = "Barcelona"
                away_team = "Real Madrid"
            else:
                home_team = "Home Club"
                away_team = "Away Club"
        if not away_team or away_team.lower() in ("away", "away team"):
            away_team = "Real Madrid" if "barca" in home_team.lower() else "Away Club"

        # Dynamic commentary: If custom script beats provided in event_window, use them
        custom_beats = event_window.get("script") or event_window.get("commentary_beats")
        if custom_beats and isinstance(custom_beats, dict) and all(k in custom_beats for k in ("beat_1_setup", "beat_2_cause", "beat_3_effect")):
            c_copy = copy.deepcopy(custom_beats)
            if "speed_segments" not in c_copy or not c_copy["speed_segments"]:
                c_copy["speed_segments"] = self._create_default_speed_segments(tel)
            injected = self._inject_real_names(c_copy, tel)
            return self._post_process_beats(injected, home_team=home_team, away_team=away_team, active_roster=active_roster)

        # Check circuit breaker
        now = time.time()
        if self._circuit_open:
            if now - self._circuit_opened_at > self._cooldown:
                self._circuit_open = False
            else:
                res = self._generate_fallback_3beat_script(
                    tone=tone,
                    thesis=thesis,
                    t_cause_clock=t_cause_clock,
                    vacated_label=vacated_label,
                    active_roster=active_roster,
                    crowd_evidence=crowd_evidence,
                    cause_telestration=tel,
                    tactical_zone=tactical_zone,
                    clip_id=clip_id_str,
                    home_team=home_team,
                    away_team=away_team,
                )
                injected = self._inject_real_names(res, tel)
                return self._post_process_beats(injected, home_team=home_team, away_team=away_team, active_roster=active_roster)

        # Attempt Gemini Vision / LLM multimodal generation if configured
        if self.api_key:
            try:
                beats = self._call_gemini_3beat(
                    event_window=event_window,
                    cause_telestration=tel,
                    freeze_frame_path=freeze_frame_path,
                    tone=tone,
                    thesis=thesis,
                    t_cause_clock=t_cause_clock,
                    vacated_label=vacated_label,
                    tactical_zone=tactical_zone,
                    home_team=home_team,
                    away_team=away_team,
                    active_roster=active_roster,
                )
                if beats and all(k in beats for k in ("beat_1_setup", "beat_2_cause", "beat_3_effect")):
                    logger.info("Successfully generated 3-beat script via Gemini 3.8 Flash.")
                    all_text = f"{beats.get('beat_1_setup', '')} {beats.get('beat_2_cause', '')} {beats.get('beat_3_effect', '')}"
                    if "[" not in all_text or "]" not in all_text:
                        cat_info = tel.get("catalyst_player") or {}
                        cat_n = cat_info.get("name") or cat_info.get("name_pos") or "The ball carrier"
                        cat_clean = re.sub(r"\s*\([^)]*\)", "", str(cat_n)).strip()
                        beats["beat_2_cause"] = f"[highlight:{cat_clean}] {beats['beat_2_cause']} [candidate_passes] [run_vectors] [stat_card:{cat_clean}]"
                    if "speed_segments" not in beats or not beats["speed_segments"]:
                        beats["speed_segments"] = self._create_default_speed_segments(tel)
                    injected = self._inject_real_names(beats, tel)
                    return self._post_process_beats(injected, home_team=home_team, away_team=away_team, active_roster=active_roster)
            except Exception as e:
                logger.warning(f"Live Gemini 3-beat generation failed: {e}. Tripping circuit breaker.")
                self._circuit_open = True
                self._circuit_opened_at = time.time()

        # Deterministic broadcast-grade fallback
        res = self._generate_fallback_3beat_script(
            tone=tone,
            thesis=thesis,
            t_cause_clock=t_cause_clock,
            vacated_label=vacated_label,
            active_roster=active_roster,
            crowd_evidence=crowd_evidence,
            cause_telestration=tel,
            tactical_zone=tactical_zone,
            clip_id=clip_id_str,
            home_team=home_team,
            away_team=away_team,
        )
        injected = self._inject_real_names(res, tel)
        return self._post_process_beats(injected, home_team=home_team, away_team=away_team, active_roster=active_roster)

    @staticmethod
    def _resolve_tactical_zone(
        event_window: Dict[str, Any],
        cause_telestration: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Resolves one of the 5 canonical tactical zones:
        'left_wing', 'zone_14', 'defensive_third', 'central', 'right_wing'.
        """
        canonical_zones = ["left_wing", "zone_14", "defensive_third", "central", "right_wing"]

        # 1. Direct key inspection in event_window and telestration
        sources = [event_window, cause_telestration or {}]
        for src in sources:
            for k in ("tactical_zone", "zone", "zone_name", "tactical_area", "clip_id", "id"):
                val = src.get(k)
                if val and isinstance(val, str):
                    val_lower = val.strip().lower()
                    if "14" in val_lower or "fourteen" in val_lower:
                        return "zone_14"
                    if "left" in val_lower or "lw" in val_lower:
                        return "left_wing"
                    if "right" in val_lower or "rw" in val_lower:
                        return "right_wing"
                    if "defens" in val_lower or "third" in val_lower or "own half" in val_lower:
                        return "defensive_third"
                    if "centr" in val_lower or "midfield" in val_lower or "middle" in val_lower or "pivot" in val_lower:
                        return "central"

        # 2. Check vacated_label in telestration
        if cause_telestration:
            v_label = str(cause_telestration.get("vacated_label", "")).strip().lower()
            if "14" in v_label:
                return "zone_14"
            if "left" in v_label:
                return "left_wing"
            if "right" in v_label:
                return "right_wing"
            if "defens" in v_label or "third" in v_label:
                return "defensive_third"
            if "centr" in v_label or "midfield" in v_label:
                return "central"

        # 3. Clip index mapping if provided
        for k in ("clip_idx", "clip_index", "index", "clip_num", "event_idx", "segment_idx"):
            if k in event_window and event_window[k] is not None:
                try:
                    c_idx = int(event_window[k])
                    return canonical_zones[c_idx % len(canonical_zones)]
                except (ValueError, TypeError):
                    pass

        # 4. Deterministic mapping based on match clock
        clock_raw = str(event_window.get("t_cause_clock", event_window.get("match_minute", "0"))).strip()
        clock_clean = clock_raw.split(":")[0] if ":" in clock_raw else clock_raw
        digits = "".join(filter(str.isdigit, clock_clean))
        if digits:
            return canonical_zones[int(digits) % len(canonical_zones)]

        # 5. Fallback hash
        thesis = str(event_window.get("thesis", ""))
        return canonical_zones[abs(hash(thesis or clock_raw)) % len(canonical_zones)]

    @classmethod
    def _resolve_match_roster(
        cls,
        home_team: str,
        away_team: str,
        active_roster: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Builds the Authoritative Match Roster Dictionary and Team Firewall context.
        Strictly partitions players into their verified clubs to eliminate
        cross-team passing hallucinations and phantom player mentions.
        """
        home_norm = _normalize_name(home_team)
        away_norm = _normalize_name(away_team)

        home_club_key = None
        away_club_key = None
        for club, data in KNOWN_CLUB_ROSTERS.items():
            for alias in data.get("aliases", []):
                a_norm = _normalize_name(alias)
                if a_norm in home_norm or home_norm in a_norm:
                    home_club_key = club
                if a_norm in away_norm or away_norm in a_norm:
                    away_club_key = club

        if not home_club_key and any(k in home_norm for k in ("barca", "barcelona", "catalan")):
            home_club_key = "FC Barcelona"
        if not away_club_key and any(k in away_norm for k in ("madrid", "real", "los blancos")):
            away_club_key = "Real Madrid"

        if not home_club_key and not away_club_key:
            home_club_key = "FC Barcelona"
            away_club_key = "Real Madrid"
        elif not home_club_key:
            home_club_key = "FC Barcelona" if away_club_key == "Real Madrid" else "Real Madrid"
        elif not away_club_key:
            away_club_key = "Real Madrid" if home_club_key == "FC Barcelona" else "FC Barcelona"

        home_data = KNOWN_CLUB_ROSTERS.get(home_club_key, {})
        away_data = KNOWN_CLUB_ROSTERS.get(away_club_key, {})

        all_players_by_team: Dict[str, str] = {}
        home_players_list: List[str] = []
        away_players_list: List[str] = []

        for p in home_data.get("players", []):
            p_name = p["name"]
            p_short = p.get("short", p_name)
            p_num = p.get("number", "")
            p_role = p.get("role", "")
            home_players_list.append(f"#{p_num} {p_name} ({p_role})")
            all_players_by_team[_normalize_name(p_name)] = home_club_key
            all_players_by_team[_normalize_name(p_short)] = home_club_key
            for part in p_name.split():
                if len(part) >= 4:
                    all_players_by_team[_normalize_name(part)] = home_club_key

        for p in away_data.get("players", []):
            p_name = p["name"]
            p_short = p.get("short", p_name)
            p_num = p.get("number", "")
            p_role = p.get("role", "")
            away_players_list.append(f"#{p_num} {p_name} ({p_role})")
            all_players_by_team[_normalize_name(p_name)] = away_club_key
            all_players_by_team[_normalize_name(p_short)] = away_club_key
            for part in p_name.split():
                if len(part) >= 4:
                    all_players_by_team[_normalize_name(part)] = away_club_key

        if active_roster and isinstance(active_roster, dict):
            for side, team_key in (("home", home_club_key), ("away", away_club_key)):
                s_list = active_roster.get(side) or active_roster.get(f"{side}_players") or []
                if isinstance(s_list, list):
                    for item in s_list:
                        n = item.get("name") if isinstance(item, dict) else str(item)
                        if n:
                            all_players_by_team[_normalize_name(n)] = team_key

        firewall_text = (
            f"[HOME CLUB: {home_club_key}]\n"
            + "\n".join(f"  * {line}" for line in home_players_list)
            + f"\n\n[AWAY CLUB: {away_club_key}]\n"
            + "\n".join(f"  * {line}" for line in away_players_list)
            + f"\n\nCRITICAL FIREWALL RULE: {home_club_key} players can ONLY pass, cross, combine with, or assist teammates of {home_club_key}. "
            f"{away_club_key} players can ONLY pass, cross, combine with, or assist teammates of {away_club_key}. "
            f"Cross-team interactions can ONLY be tackles, blocks, duels, saves, or interceptions!"
        )

        return {
            "home_club": home_club_key,
            "away_club": away_club_key,
            "home_data": home_data,
            "away_data": away_data,
            "all_players_by_team": all_players_by_team,
            "firewall_text": firewall_text,
        }

    @classmethod
    def _get_player_club(cls, player_name: str, roster_info: Dict[str, Any]) -> Optional[str]:
        if not player_name or not roster_info:
            return None
        norm = _normalize_name(player_name)
        lookup = roster_info.get("all_players_by_team", {})
        if norm in lookup:
            return lookup[norm]
        for part in norm.split():
            if len(part) >= 4 and part in lookup:
                return lookup[part]
        return None

    def _call_gemini_3beat(
        self,
        event_window: Dict[str, Any],
        cause_telestration: Dict[str, Any],
        freeze_frame_path: Optional[str],
        tone: str,
        thesis: str,
        t_cause_clock: str,
        vacated_label: str,
        tactical_zone: str = "central",
        home_team: Optional[str] = None,
        away_team: Optional[str] = None,
        active_roster: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, str]]:
        home = str(home_team or event_window.get("home_team") or "FC Barcelona").strip()
        away = str(away_team or event_window.get("away_team") or "Real Madrid").strip()
        roster_info = self._resolve_match_roster(home, away, active_roster)
        roster_firewall_context = roster_info["firewall_text"]

        tone_instruction = (
            "Expose structural negligence, failure to track runners, and tactical collapse"
            if tone == "RUTHLESS_AUTOPSY"
            else "Praise coordinated manipulation of space, spatial dominance, and lethal execution"
        )

        backline_info = "Yes (defensive unit linked)" if cause_telestration.get("backline_track_ids") else "None"
        triangle_info = "Yes (attacking overload triangle)" if cause_telestration.get("triangle_track_ids") else "None"
        focal_info = f"Tracks {cause_telestration.get('focal_track_ids', [])}"
        cat_p = cause_telestration.get("catalyst_player") or {}
        raw_cat = str(cat_p.get("name") or cat_p.get("name_pos") or "The ball carrier").strip()
        cleaned_cat = re.sub(r'[\(\[\/].*?[\)\]\/]', '', raw_cat).strip()
        catalyst_info = cleaned_cat if cleaned_cat and cleaned_cat.upper() not in ("BALL CARRIER", "CARRIER") else ""

        catalyst_club = self._get_player_club(catalyst_info, roster_info)
        if not catalyst_club:
            catalyst_club = roster_info["home_club"]
            if not catalyst_info:
                catalyst_info = roster_info["home_data"].get("star_creator", "Catalyst")

        catalyst_team_name = catalyst_club
        opposing_club = roster_info["away_club"] if catalyst_club == roster_info["home_club"] else roster_info["home_club"]
        opp_data = roster_info["away_data"] if catalyst_club == roster_info["home_club"] else roster_info["home_data"]
        team_data = roster_info["home_data"] if catalyst_club == roster_info["home_club"] else roster_info["away_data"]

        pls = cause_telestration.get("players") or []
        avail_names = []
        for p in pls:
            p_n = str(p.get("name") or "").strip()
            p_clean = re.sub(r'[\(\[\/].*?[\)\]\/]', '', p_n).strip()
            if p_clean and "CARRIER" not in p_clean.upper() and "RUNNER" not in p_clean.upper() and "DEFENDER" not in p_clean.upper():
                avail_names.append(p_clean)

        runner_p = cause_telestration.get("runner_player") or {}
        raw_runner = str(runner_p.get("name") or runner_p.get("name_pos") or "").strip()
        cleaned_runner = re.sub(r'[\(\[\/].*?[\)\]\/]', '', raw_runner).strip()
        runner_club = self._get_player_club(cleaned_runner, roster_info)

        runner_info = ""
        opponent_info = ""

        if cleaned_runner:
            if runner_club == catalyst_club:
                runner_info = cleaned_runner
            elif runner_club == opposing_club:
                opponent_info = cleaned_runner

        if not runner_info and avail_names:
            for cand in avail_names:
                c_club = self._get_player_club(cand, roster_info)
                if c_club == catalyst_club and cand.lower() != catalyst_info.lower():
                    runner_info = cand
                    break
                elif c_club == opposing_club and not opponent_info:
                    opponent_info = cand

        if not runner_info:
            runner_info = team_data.get("star_attacker", "Supporting teammate")
            if runner_info.lower() == catalyst_info.lower():
                runner_info = team_data.get("star_creator", "Supporting teammate")

        if not opponent_info:
            opponent_info = opp_data.get("star_creator", "Defending marker")

        available_players_info = ", ".join(avail_names) if avail_names else ", ".join(team_data.get("default_starters", [])[:4] + opp_data.get("default_starters", [])[:4])

        from src.rendering.match_stats_provider import MatchStatsProvider
        stats_provider = MatchStatsProvider()

        telemetry_lines: List[str] = []
        if catalyst_info:
            cat_phrase = self._get_player_telemetry_phrase(stats_provider, catalyst_info)
            telemetry_lines.append(f"  * {catalyst_info} ({catalyst_club}): {cat_phrase}")
        if runner_info:
            run_phrase = self._get_player_telemetry_phrase(stats_provider, runner_info)
            telemetry_lines.append(f"  * {runner_info} ({catalyst_club}): {run_phrase}")
        if opponent_info:
            opp_phrase = self._get_player_telemetry_phrase(stats_provider, opponent_info)
            telemetry_lines.append(f"  * {opponent_info} ({opposing_club}): {opp_phrase}")

        player_telemetry_context = "\n".join(telemetry_lines) if telemetry_lines else "  * Dynamic match actions and empirical form available."

        prompt = PROMPT_3BEAT_TEMPLATE.format(
            home_team=home,
            away_team=away,
            roster_firewall_context=roster_firewall_context,
            narrative_tone=tone,
            tone_instruction=tone_instruction,
            tactical_zone=tactical_zone,
            thesis=thesis or "Tactical breakdown",
            crowd_evidence=event_window.get("crowd_evidence", "Consensus breakdown"),
            clock=t_cause_clock,
            vacated_label=vacated_label,
            backline_info=backline_info,
            triangle_info=triangle_info,
            focal_info=focal_info,
            catalyst_info=catalyst_info,
            catalyst_team_name=catalyst_team_name,
            runner_info=runner_info,
            opponent_info=opponent_info,
            available_players_info=available_players_info,
            player_telemetry_context=player_telemetry_context,
        )

        parts: list[dict[str, Any]] = []
        if freeze_frame_path and Path(freeze_frame_path).is_file():
            with open(freeze_frame_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})

        parts.append({"text": prompt})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.3,
                "response_mime_type": "application/json",
            },
        }
        req_data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        models_to_try = ["gemini-flash-lite-latest", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
        last_exc = None
        idx_m = 0

        while idx_m < len(models_to_try):
            model_name = models_to_try[idx_m]
            idx_m += 1
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")

            try:
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if candidates:
                        raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                        m = re.search(r'\{.*\}', raw_text, re.DOTALL)
                        clean_text = m.group(0) if m else raw_text
                        if clean_text.startswith("```json"):
                            clean_text = clean_text[7:].strip()
                        elif clean_text.startswith("```"):
                            clean_text = clean_text[3:].strip()
                        if clean_text.endswith("```"):
                            clean_text = clean_text[:-3].strip()
                        parsed = json.loads(clean_text)
                        b2_text = str(parsed.get("beat_2_cause", "")).strip()
                        b2_steps = parsed.get("beat_2_steps") or {}
                        if not isinstance(b2_steps, dict) or not all(k in b2_steps for k in ("step_1", "step_2", "step_3", "step_4")):
                            s_list = [s.strip() for s in re.split(r'(?<=[.!?])\s+', b2_text) if s.strip()]
                            if len(s_list) >= 4:
                                b2_steps = {
                                    "step_1": s_list[0],
                                    "step_2": s_list[1],
                                    "step_3": s_list[2],
                                    "step_4": " ".join(s_list[3:]),
                                }
                            else:
                                b2_steps = {
                                    "step_1": b2_text,
                                    "step_2": b2_text,
                                    "step_3": b2_text,
                                    "step_4": b2_text,
                                }
                        logger.info("Successfully generated 3-beat script via %s.", model_name)
                        b1_str = str(parsed.get("beat_1_setup", "")).strip()
                        if not b1_str.endswith((".", "!", "?")):
                            b1_str += "."

                        if not b2_text.endswith((".", "!", "?")):
                            b2_text += "."

                        b3_str = str(parsed.get("beat_3_effect", "")).strip()
                        if not b3_str.endswith((".", "!", "?")):
                            b3_str += "."

                        raw_segs = parsed.get("speed_segments") or []
                        speed_segs: List[Dict[str, Any]] = []
                        if isinstance(raw_segs, list) and raw_segs:
                            for s in raw_segs:
                                if isinstance(s, dict) and "segment_type" in s and "rate" in s:
                                    try:
                                        m = SpeedSegment(**s)
                                        speed_segs.append(m.model_dump() if hasattr(m, "model_dump") else m.dict())
                                    except Exception:
                                        pass
                        if not speed_segs or not any(s.get("segment_type") == "FREEZE_HOLD" for s in speed_segs):
                            speed_segs = self._create_default_speed_segments(cause_telestration)

                        return {
                            "beat_1_setup": b1_str,
                            "beat_2_cause": b2_text,
                            "beat_2_steps": b2_steps,
                            "beat_3_effect": b3_str,
                            "speed_segments": speed_segs,
                        }
            except urllib.error.HTTPError as e:
                last_exc = e
                if e.code == 429 or "429" in str(e):
                    logger.warning("Gemini model '%s' returned HTTP 429 rate limit. Sleeping 4 seconds and retrying with fallback flash models...", model_name)
                    time.sleep(4.0)
                    for rm in ["gemini-flash-lite-latest", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]:
                        if rm not in models_to_try[idx_m:]:
                            models_to_try.append(rm)
                    continue
                logger.warning("Gemini model '%s' HTTP %s: %s", model_name, e.code, e)
            except Exception as e:
                logger.warning("Gemini model '%s' failed for 3-beat script: %s", model_name, e)
                last_exc = e

        if last_exc:
            raise last_exc
        return None

    @staticmethod
    def _inject_real_names(beats: Dict[str, Any], cause_tel: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Replaces generic placeholder roles (like 'BALL CARRIER') with the actual verified player names
        so both TTS audio commentary and onscreen captions speak the player's real name out loud.
        """
        if not beats or not isinstance(beats, dict) or not cause_tel or not isinstance(cause_tel, dict):
            return beats

        def _clean_spoken_name(raw: str) -> str:
            if not raw:
                return ""
            s = re.sub(r'[\(\[\/].*?[\)\]\/]', '', raw).strip()
            return s if s.upper() not in ("BALL CARRIER", "CARRIER", "PLAYER", "CATALYST", "") else ""

        cat_player = cause_tel.get("catalyst_player") or {}
        cat_spoken = _clean_spoken_name(cat_player.get("name") or cat_player.get("name_pos") or "")

        runner_player = cause_tel.get("runner_player") or {}
        run_spoken = _clean_spoken_name(runner_player.get("name") or runner_player.get("name_pos") or "")

        def _sub(text: str) -> str:
            if not text or not isinstance(text, str):
                return text
            t = text
            if cat_spoken:
                t = re.sub(r'\b(?:the\s+)?ball\s+carrier\b', cat_spoken, t, flags=re.IGNORECASE)
                t = re.sub(r'\b(?:the\s+)?central\s+playmaker\b', cat_spoken, t, flags=re.IGNORECASE)
                t = re.sub(r'\b(?:the\s+)?wide\s+playmaker\b', cat_spoken, t, flags=re.IGNORECASE)
                t = re.sub(r'\b(?:the\s+)?central\s+catalyst\b', cat_spoken, t, flags=re.IGNORECASE)
            if run_spoken:
                t = re.sub(r'\b(?:the\s+)?third[- ]man\s+runner\b', run_spoken, t, flags=re.IGNORECASE)
                t = re.sub(r'\b(?:the\s+)?primary\s+runner\b', run_spoken, t, flags=re.IGNORECASE)
            return t

        res = copy.deepcopy(beats)
        for k in ("beat_1_setup", "beat_2_cause", "beat_3_effect"):
            if k in res and isinstance(res[k], str):
                res[k] = _sub(res[k])
            elif k in res and isinstance(res[k], dict) and "text" in res[k]:
                res[k]["text"] = _sub(res[k]["text"])

        if "beat_2_steps" in res and isinstance(res["beat_2_steps"], dict):
            for sk in res["beat_2_steps"]:
                res["beat_2_steps"][sk] = _sub(str(res["beat_2_steps"][sk]))

        # Ensure visual cue heatmap tag in Beat 2 without adding robotic "Backed by..." prefix
        b2_str = str(res.get("beat_2_cause", ""))
        has_data_tag = any(k in b2_str for k in ("[match_heatmap", "[heatmap", "[average_position", "[data_card"))
        if not has_data_tag and cat_spoken:
            # Prepend silent cue tag at first mention of player or at the start
            if f"[highlight:{cat_spoken}]" in b2_str:
                res["beat_2_cause"] = b2_str.replace(f"[highlight:{cat_spoken}]", f"[match_heatmap:{cat_spoken}] [highlight:{cat_spoken}]", 1)
            else:
                res["beat_2_cause"] = f"[match_heatmap:{cat_spoken}] {b2_str}"
            if "beat_2_steps" in res and isinstance(res["beat_2_steps"], dict):
                s1 = res["beat_2_steps"].get("step_1", "")
                if s1 and f"[match_heatmap:{cat_spoken}]" not in s1:
                    res["beat_2_steps"]["step_1"] = f"[match_heatmap:{cat_spoken}] {s1}"

        return res

    @staticmethod
    def _get_player_telemetry_phrase(stats_provider: Any, player_name: str, default_stat: str = "completing ninety percent of passes") -> str:
        """
        Returns an empirical, television-grade spoken telemetry phrase for a player.
        e.g. "winning ninety percent of duels with ninety-three percent pass accuracy"
        Weaves naturally into speech without robotic "backed by" or video-game rating numbers.
        """
        if not player_name or any(w in player_name.lower() for w in ("carrier", "runner", "pass", "option", "defender")):
            return default_stat
        try:
            p_stats = stats_provider.get_player_stats(player_name)
            if not p_stats:
                return default_stat
            stats_dict = p_stats.get("stats", {})

            # Look for pass accuracy
            pass_phrase = ""
            for k, v in stats_dict.items():
                if "pass" in k.lower():
                    m = re.search(r'\(?(\d+)%\)?', str(v))
                    if m:
                        pct = int(m.group(1))
                        pass_phrase = f"completing {pct} percent of his passes"
                    break

            # Look for duels won
            duel_phrase = ""
            for k, v in stats_dict.items():
                if "duel" in k.lower():
                    m = re.search(r'\(?(\d+)%\)?', str(v))
                    if m:
                        pct = int(m.group(1))
                        duel_phrase = f"winning {pct} percent of duels"
                    elif "/" in str(v):
                        duel_phrase = f"winning {str(v).strip()} duels"
                    break

            # Look for defensive actions
            def_phrase = ""
            for k, v in stats_dict.items():
                if any(w in k.lower() for w in ("interception", "tackle", "clearance", "recover")):
                    m = re.search(r'(\d+)', str(v))
                    if m and int(m.group(1)) > 0:
                        def_phrase = f"registering {m.group(1)} defensive recoveries"
                    break

            candidates = [c for c in (pass_phrase, duel_phrase, def_phrase) if c]
            if candidates:
                if len(candidates) >= 2:
                    return f"{candidates[0]} and {candidates[1]}"
                return candidates[0]
        except Exception:
            pass
        return "commanding the tempo with technical composure"

    def _generate_fallback_3beat_script(
        self,
        tone: str,
        thesis: str,
        t_cause_clock: str,
        vacated_label: str,
        active_roster: Dict[str, Any],
        crowd_evidence: str,
        cause_telestration: Dict[str, Any],
        tactical_zone: str = "central",
        clip_id: Optional[str] = None,
        home_team: Optional[str] = None,
        away_team: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Deterministic, television-grade Carragher & Henry commentary lockstep script.
        Follows the 4-Step Analyst Framework:
        - Step 1: Catalyst identification with naturally woven performance
        - Step 2: Linked defensive structure or shape
        - Step 3: Exploit zone / vacated space
        - Step 4: Decision taken & pass/run vector
        Guarantees zero cross-team passing, zero phantom players, zero 'Backed by...' openers,
        and zero repetitive robotic cliches.
        """
        clock_clean = t_cause_clock.split(":")[0].strip() if ":" in t_cause_clock else str(t_cause_clock).strip()
        minute_digits = "".join(filter(str.isdigit, clock_clean))
        min_prefix = f"Minute {int(minute_digits)}: " if minute_digits and int(minute_digits) > 0 else ""

        tel_dict = cause_telestration if isinstance(cause_telestration, dict) else {}
        home_team_name = str(home_team or tel_dict.get("home_team") or "FC Barcelona").strip()
        away_team_name = str(away_team or tel_dict.get("away_team") or "Real Madrid").strip()

        roster_info = self._resolve_match_roster(home_team_name, away_team_name, active_roster)
        home_club = roster_info["home_club"]
        away_club = roster_info["away_club"]

        # Catalyst resolution
        cat_player = tel_dict.get("catalyst_player") or {}
        raw_cat_name = str(cat_player.get("name") or cat_player.get("name_pos") or "").strip()
        cat_name = re.sub(r'[\(\[\/].*?[\)\]\/]', '', raw_cat_name).strip()
        cat_club = self._get_player_club(cat_name, roster_info)
        if not cat_club:
            cat_club = home_club
            if not cat_name:
                cat_name = roster_info["home_data"].get("star_creator", "Catalyst")

        att_team = cat_club
        def_team = away_club if att_team == home_club else home_club
        att_data = roster_info["home_data"] if att_team == home_club else roster_info["away_data"]
        def_data = roster_info["away_data"] if att_team == home_club else roster_info["home_data"]

        # Runner / teammate resolution (strictly from att_team)
        run_player = tel_dict.get("runner_player") or {}
        raw_run_name = str(run_player.get("name") or run_player.get("name_pos") or "").strip()
        cleaned_run_name = re.sub(r'[\(\[\/].*?[\)\]\/]', '', raw_run_name).strip()
        run_club = self._get_player_club(cleaned_run_name, roster_info)

        if cleaned_run_name and run_club == att_team and cleaned_run_name.lower() != cat_name.lower():
            run_name = cleaned_run_name
        else:
            run_name = att_data.get("star_attacker", "Supporting teammate")
            if run_name.lower() == cat_name.lower():
                run_name = att_data.get("star_creator", "Supporting teammate")

        from src.rendering.match_stats_provider import MatchStatsProvider
        stats_provider = MatchStatsProvider()
        cat_telemetry = self._get_player_telemetry_phrase(stats_provider, cat_name)

        zone_label = tactical_zone.replace("_", " ").title()
        play_type = str(tel_dict.get("play_type", "")).strip().upper()
        thesis_lower = (thesis or "").lower()
        is_corner = (
            play_type in ("CORNER_SET_PIECE", "CORNER", "SET_PIECE")
            or "corner" in thesis_lower
            or "set piece" in thesis_lower
            or "set-piece" in thesis_lower
        )
        is_gk_buildup = ("gk" in play_type.lower() or "goalkeeper" in play_type.lower() or (tactical_zone == "defensive_third" and "gk" in thesis_lower))

        if is_corner:
            b1 = f"{att_team} work a structured corner routine through [highlight:{cat_name}]."
            b2_cause = f"[highlight:{cat_name}], {cat_telemetry}, addresses the delivery targeting [highlight:{run_name}]. [candidate_passes] [run_vectors] Space develops across the six-yard box as [match_heatmap:{cat_name}] {cat_name} creates a dangerous crossing angle."
            b2_steps = {
                "step_1": f"[highlight:{cat_name}] addresses the corner delivery.",
                "step_2": f"{def_team}'s zonal marking holds position.",
                "step_3": f"[candidate_passes] [run_vectors] Space opens near the six-yard box.",
                "step_4": f"[match_heatmap:{cat_name}] {cat_name} delivers accurately to [highlight:{run_name}].",
            }
            b3 = f"As play resumes, the whipped delivery tests {def_team}'s aerial organization."
        elif is_gk_buildup or tactical_zone == "defensive_third":
            b1 = f"{att_team} build patiently from deep under pressure."
            b2_cause = f"[highlight:{cat_name}], {cat_telemetry}, invites the initial press. [candidate_passes] [run_vectors] A clean progression lane opens toward [highlight:{run_name}] as [match_heatmap:{cat_name}] {cat_name} lifts a measured forward distribution."
            b2_steps = {
                "step_1": f"[highlight:{cat_name}] surveys passing options from deep.",
                "step_2": f"{def_team}'s forward pressing unit steps up.",
                "step_3": f"[candidate_passes] [run_vectors] Progression lane opens in midfield.",
                "step_4": f"[match_heatmap:{cat_name}] {cat_name} clips a measured pass to [highlight:{run_name}].",
            }
            b3 = f"In real time, the distribution cleanly transitions possession into the middle third."
        elif tactical_zone == "zone_14":
            b1 = f"{att_team} penetrate the central half-space with pace."
            b2_cause = f"[highlight:{cat_name}], {cat_telemetry}, spots [highlight:{run_name}] between the defensive lines. [candidate_passes] [run_vectors] The central shape fractures as [match_heatmap:{cat_name}] {cat_name} slides an incisive through-ball."
            b2_steps = {
                "step_1": f"[highlight:{cat_name}] receives centrally facing goal.",
                "step_2": f"{def_team}'s midfield line backpedals under duress.",
                "step_3": f"[candidate_passes] [run_vectors] An exploitable gap opens in Zone 14.",
                "step_4": f"[match_heatmap:{cat_name}] {cat_name} threads the release ball to [highlight:{run_name}].",
            }
            b3 = f"As play resumes, [highlight:{run_name}] receives in stride to produce a dangerous opportunity."
        elif tactical_zone in ("left_wing", "right_wing"):
            flank_side = "left" if tactical_zone == "left_wing" else "right"
            b1 = f"{att_team} isolate the {flank_side} touchline through [highlight:{cat_name}]."
            b2_cause = f"[highlight:{cat_name}], {cat_telemetry}, engages his defender out wide. [candidate_passes] [run_vectors] Inside space opens as [match_heatmap:{cat_name}] [highlight:{run_name}] attacks the channel."
            b2_steps = {
                "step_1": f"[highlight:{cat_name}] controls possession along the {flank_side} flank.",
                "step_2": f"{def_team}'s defensive unit shifts toward the ball.",
                "step_3": f"[candidate_passes] [run_vectors] Channel opens behind the advancing marker.",
                "step_4": f"[match_heatmap:{cat_name}] {cat_name} releases [highlight:{run_name}] into the box.",
            }
            b3 = f"In real time, the progressive delivery breaches the penalty box with pace."
        else:
            b1 = f"{att_team} dictate the midfield tempo through [highlight:{cat_name}]."
            b2_cause = f"[highlight:{cat_name}], {cat_telemetry}, orchestrates the transition. [candidate_passes] [run_vectors] Passing angles develop as [match_heatmap:{cat_name}] {cat_name} releases [highlight:{run_name}] through the middle."
            b2_steps = {
                "step_1": f"[highlight:{cat_name}] controls centrally in transition.",
                "step_2": f"{def_team}'s defensive structure holds its line.",
                "step_3": f"[candidate_passes] [run_vectors] Passing angles develop across the pitch.",
                "step_4": f"[match_heatmap:{cat_name}] {cat_name} releases [highlight:{run_name}] forward.",
            }
            b3 = f"As play resumes, {att_team} sustain forward momentum into the attacking half."

        speed_segments = self._create_default_speed_segments(tel_dict)
        return {
            "beat_1_setup": b1,
            "beat_2_cause": b2_cause,
            "beat_2_steps": b2_steps,
            "beat_3_effect": b3,
            "speed_segments": speed_segments,
        }


    @staticmethod
    def _create_default_speed_segments(cause_telestration: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        tel = cause_telestration or {}
        c_idx = tel.get("cause_frame_idx") or tel.get("video_freeze_frame") or 66
        try:
            c_idx_int = int(c_idx)
        except Exception:
            c_idx_int = 66
        t_cause_sec = round(max(0.6, float(c_idx_int) / 30.0), 2)
        total_sec = round(t_cause_sec + 4.5, 2)

        speed_segments: List[Dict[str, Any]] = []
        if t_cause_sec > 1.2:
            transit_end = round(max(0.6, t_cause_sec - 0.6), 2)
            speed_segments.append({
                "segment_type": "FAST_TRANSIT",
                "rate": 1.30,
                "source_start_sec": 0.0,
                "source_end_sec": transit_end,
                "duration_sec": round(transit_end / 1.30, 2),
                "reason": "Ball traveling in transition across open turf",
            })
            speed_segments.append({
                "segment_type": "SLOW_FOCUS",
                "rate": 0.70,
                "source_start_sec": transit_end,
                "source_end_sec": t_cause_sec,
                "duration_sec": round((t_cause_sec - transit_end) / 0.70, 2),
                "reason": "Deceleration as carrier engages defender prior to freeze",
            })
        else:
            speed_segments.append({
                "segment_type": "NORMAL_FLOW",
                "rate": 1.00,
                "source_start_sec": 0.0,
                "source_end_sec": t_cause_sec,
                "duration_sec": t_cause_sec,
                "reason": "Initial tactical phase setup",
            })

        # Freeze hold
        speed_segments.append({
            "segment_type": "FREEZE_HOLD",
            "rate": 0.00,
            "source_start_sec": t_cause_sec,
            "source_end_sec": t_cause_sec,
            "duration_sec": 2.0,
            "reason": "Tactical freeze pause for telestration reveal",
        })

        # Unfreeze micro-technique slow focus
        focus_end = round(t_cause_sec + 1.2, 2)
        speed_segments.append({
            "segment_type": "SLOW_FOCUS",
            "rate": 0.65,
            "source_start_sec": t_cause_sec,
            "source_end_sec": focus_end,
            "duration_sec": round(1.2 / 0.65, 2),
            "reason": "Decisive press-breaking turn and key action execution",
        })

        # Final consequence flow
        speed_segments.append({
            "segment_type": "NORMAL_FLOW",
            "rate": 1.00,
            "source_start_sec": focus_end,
            "source_end_sec": total_sec,
            "duration_sec": round(total_sec - focus_end, 2),
            "reason": "Consequence unfolds into box or clearance",
        })
        return speed_segments

    @classmethod
    def _post_process_beats(
        cls,
        beats: Dict[str, Any],
        home_team: str = "Home",
        away_team: str = "Away",
        active_roster: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Multi-stage television broadcast sanitizer ensuring 100% compliance with directives:
        1. Strips banned mechanical openers ('Backed by...', 'Unfreezing into motion', 'using precise telemetry')
        2. Enforces Team Firewall & Zero Cross-Team Passing
        3. Eradicates phantom players (Pisilli, Malen, Dumfries, Konaté, Ndicka, Cucurella, Gordon, Cancelo)
        4. Caps tactical jargon to at most 1 occurrence per buzzword per clip
        5. Removes UI overlay narrations (Heatmap, Telemetry) while keeping visual tags intact
        """
        if not beats or not isinstance(beats, dict):
            return beats

        roster_info = cls._resolve_match_roster(home_team, away_team, active_roster)
        home_club = roster_info["home_club"]
        away_club = roster_info["away_club"]
        player_lookup = roster_info["all_players_by_team"]

        seen_jargon: Dict[str, int] = {}
        jargon_alt_indices: Dict[str, int] = {}

        def _clean_field(text: str, is_b2: bool = False) -> str:
            if not text or not isinstance(text, str):
                return text

            # 1. Eradicate phantom players
            for phantom, rep_dict in PHANTOM_REPLACEMENTS.items():
                if re.search(r'\b' + re.escape(phantom) + r'\b', text, re.I):
                    rep = rep_dict.get(home_club, rep_dict.get(away_club, "teammate"))
                    text = re.sub(r'\b' + re.escape(phantom) + r'\b', rep, text, flags=re.I)

            # 2. Ban "Backed by..." openers & rewrite naturally
            m_lead = re.match(
                r'^(?:(\[[a-zA-Z0-9_:]+\]\s*)*)[Bb]acked by\s+([^,]+),\s*([A-Z\u00c0-\u017f][a-z\u00c0-\u017f]+(?:\s+[A-Z\u00c0-\u017f][a-z\u00c0-\u017f]+)?)\s+(.*)',
                text,
            )
            if m_lead:
                tag_prefix = m_lead.group(1) or ""
                stat = m_lead.group(2).strip()
                player = m_lead.group(3).strip()
                rest = m_lead.group(4).strip()
                stat_clean = re.sub(r'\b\d+(\.\d+)?\s+rating\b', 'technical composure', stat)
                if any(w in stat_clean.lower() for w in ('pass', 'accuracy', 'recovery', 'duel', 'composure', 'distribution', 'percent')):
                    text = f"{tag_prefix}{player}, showing {stat_clean}, {rest}"
                else:
                    text = f"{tag_prefix}{player} {rest}"

            text = re.sub(r',\s*backed by\s+\d+(\.\d+)?\s+rating\s+and\s+', ', registering ', text, flags=re.I)
            text = re.sub(r',\s*backed by\s+[^,.:;]+(?=[,.:;]|$)', '', text, flags=re.I)
            text = re.sub(r'\b[Bb]acked by\s+[^,.:;]+[,.:;]?\s*', '', text, flags=re.I)

            # 3. Apply Cliche and Robotic Replacements
            for pattern, repl in CLICHE_REPLACEMENTS:
                text = pattern.sub(repl, text)

            # 4. Enforce Team Firewall & Zero Cross-Team Passing
            if 'courtois' in text.lower() and 'lewandowski' in text.lower():
                text = re.sub(
                    r'Courtois delivers under intense pressure,\s*while Lewandowski exploits',
                    'Courtois is pressed into a hurried clearance as Lewandowski challenges in the',
                    text,
                    flags=re.I,
                )

            m_exp = re.search(r'exploiting\s+(Vin[ií]cius(?:\s+Jr)?)\'s\s+(?:spatial\s+)?overload', text, re.I)
            if m_exp:
                text = text[:m_exp.start(1)] + 'Raphinha' + text[m_exp.end(1):]
                text = re.sub(r'\boverload\b', 'channel run', text, flags=re.I)

            pass_pat = re.compile(
                r'(\b(?:passes?|passing|passed|slides?|sliding|slid|threads?|threading|threaded|'
                r'plays?|playing|played|finds?|finding|found|crosses?|crossing|crossed|'
                r'releases?|releasing|released|assists?|assisting|assisted|feeds?|feeding|fed)\s+'
                r'(?:a\s+[^,.:;]+?\s+)?(?:to|for|into)\s+)(?:\[highlight:)?([A-Z\u00c0-\u017f][a-z\u00c0-\u017f]+(?:\s+[A-Z\u00c0-\u017f][a-z\u00c0-\u017f]+)?)(?:\])?',
                re.I,
            )
            for m in list(pass_pat.finditer(text)):
                target_name = m.group(2).strip()
                target_norm = _normalize_name(target_name)
                target_club = player_lookup.get(target_norm)
                if not target_club:
                    for part in target_norm.split():
                        if len(part) >= 4 and part in player_lookup:
                            target_club = player_lookup[part]
                            break

                prior_text = text[:m.start()]
                subject_club = None
                for norm_p, c_name in player_lookup.items():
                    if re.search(r'\b' + re.escape(norm_p) + r'\b', _normalize_name(prior_text)):
                        subject_club = c_name

                if subject_club and target_club and subject_club != target_club:
                    star_teammate = (
                        roster_info["home_data"].get("star_attacker", "his teammate")
                        if subject_club == home_club
                        else roster_info["away_data"].get("star_attacker", "his teammate")
                    )
                    text = text[:m.start(2)] + star_teammate + text[m.end(2):]

            # 5. Cap tactical buzzwords (Max 1 per clip across all beats)
            for term, alts in JARGON_ALTERNATIVES.items():
                pat = re.compile(r'\b' + re.escape(term) + r'\b', re.I)
                matches = list(pat.finditer(text))
                if not matches:
                    continue
                for m in reversed(matches):
                    count = seen_jargon.get(term, 0)
                    if count >= 1:
                        idx = jargon_alt_indices.get(term, 0)
                        replacement = alts[idx % len(alts)]
                        jargon_alt_indices[term] = idx + 1
                        text = text[:m.start()] + replacement + text[m.end():]
                    seen_jargon[term] = count + 1

            # 6. Clean spacing and punctuation
            text = re.sub(r'\s{2,}', ' ', text).strip()
            if text and not text.endswith(('.', '!', '?')):
                text += '.'
            return text

        res = copy.deepcopy(beats)
        if "beat_1_setup" in res:
            res["beat_1_setup"] = _clean_field(res["beat_1_setup"])
        if "beat_2_cause" in res:
            res["beat_2_cause"] = _clean_field(res["beat_2_cause"], is_b2=True)
        if "beat_2_steps" in res and isinstance(res["beat_2_steps"], dict):
            for sk in res["beat_2_steps"]:
                res["beat_2_steps"][sk] = _clean_field(res["beat_2_steps"][sk])
        if "beat_3_effect" in res:
            res["beat_3_effect"] = _clean_field(res["beat_3_effect"])

        return res


    # Backward compatibility with single-sentence freeze-frame commentary
    def generate_commentary(
        self,
        freeze_frame_path: Optional[str],
        event_context: Dict[str, Any],
    ) -> str:
        minute = event_context.get("match_minute", 0)
        team = event_context.get("team", "Unknown Team")
        home = event_context.get("home_team", "Home")
        away = event_context.get("away_team", "Away")
        raw_player = event_context.get("player") or (
            event_context.get("players_involved", ["BALL CARRIER"])[0]
            if event_context.get("players_involved")
            else "BALL CARRIER"
        )
        if isinstance(raw_player, dict):
            raw_player = raw_player.get("name", "BALL CARRIER")

        event_type = event_context.get("event_type", "action")
        description = event_context.get("narrative_hook") or event_context.get("description", "")

        player = ""
        if isinstance(raw_player, str):
            rp_u = raw_player.upper()
            for r in ("BALL CARRIER", "PRESSING TRIGGER", "THIRD-MAN RUNNER", "ISOLATED FULLBACK", "COVERING CB", "MIDFIELD PIVOT"):
                if r in rp_u:
                    player = r
                    break

        if not player:
            ev_lower = str(event_type).lower()
            if "press" in ev_lower or "turnover" in ev_lower:
                player = "PRESSING TRIGGER"
            elif "transition" in ev_lower or "breakout" in ev_lower or "counter" in ev_lower or "run" in ev_lower:
                player = "THIRD-MAN RUNNER"
            elif "defen" in ev_lower or "block" in ev_lower or "tackle" in ev_lower:
                player = "COVERING CB"
            elif "pivot" in ev_lower or "midfield" in ev_lower:
                player = "MIDFIELD PIVOT"
            elif "flank" in ev_lower or "wing" in ev_lower or "overlap" in ev_lower:
                player = "ISOLATED FULLBACK"
            else:
                player = "BALL CARRIER"

        # Check circuit breaker
        now = time.time()
        if self._circuit_open:
            if now - self._circuit_opened_at > self._cooldown:
                self._circuit_open = False
            else:
                return self._generate_fallback_commentary(minute, team, player, event_type, description)

        if self.api_key and freeze_frame_path and Path(freeze_frame_path).is_file():
            try:
                commentary = self._analyze_image_with_gemini(
                    freeze_frame_path=freeze_frame_path,
                    minute=minute,
                    home=home,
                    away=away,
                    team=team,
                    player=player,
                    event_type=event_type,
                    description=description,
                )
                if commentary and len(commentary.split()) >= 6:
                    return commentary
            except Exception as e:
                logger.warning(f"Live Gemini commentary analysis failed: {e}. Tripping circuit breaker.")
                self._circuit_open = True
                self._circuit_opened_at = time.time()

        return self._generate_fallback_commentary(minute, team, player, event_type, description)

    def _analyze_image_with_gemini(
        self,
        freeze_frame_path: str,
        minute: int,
        home: str,
        away: str,
        team: str,
        player: str,
        event_type: str,
        description: str,
    ) -> str:
        with open(freeze_frame_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = CANONICAL_PROMPT_TEMPLATE.format(
            minute=minute,
            home_team=home,
            away_team=away,
            team=team,
            player=player,
            event_type=event_type,
            description=description,
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_b64,
                            }
                        },
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 100,
            },
        }

        models_to_try = ["gemini-flash-lite-latest", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
        last_exc = None
        idx_m = 0
        while idx_m < len(models_to_try):
            model_name = models_to_try[idx_m]
            idx_m += 1
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if candidates:
                        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        cleaned = re.sub(r'[\r\n]+', ' ', text).strip()
                        return cleaned.strip('"').strip("'")
            except urllib.error.HTTPError as e:
                last_exc = e
                if e.code == 429 or "429" in str(e):
                    logger.warning("Gemini commentary model '%s' returned HTTP 429 rate limit. Sleeping 4 seconds and retrying with fallback flash models...", model_name)
                    time.sleep(4.0)
                    for rm in ["gemini-flash-lite-latest", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]:
                        if rm not in models_to_try[idx_m:]:
                            models_to_try.append(rm)
                    continue
                logger.warning("Gemini commentary model '%s' HTTP %s: %s", model_name, e.code, e)
            except Exception as e:
                logger.warning("Gemini commentary model '%s' failed: %s", model_name, e)
                last_exc = e

        raise ValueError(f"Empty response from Gemini Vision API failover: {last_exc}")

    def _generate_fallback_commentary(
        self,
        minute: int,
        team: str,
        player: str,
        event_type: str,
        description: str,
    ) -> str:
        norm_type = event_type.lower()
        tactical_role = "BALL CARRIER"
        if isinstance(player, str):
            p_upper = player.upper()
            for r in ("BALL CARRIER", "PRESSING TRIGGER", "THIRD-MAN RUNNER", "ISOLATED FULLBACK", "COVERING CB", "MIDFIELD PIVOT"):
                if r in p_upper:
                    tactical_role = r
                    break
            else:
                if "press" in norm_type or "turnover" in norm_type:
                    tactical_role = "PRESSING TRIGGER"
                elif "transition" in norm_type or "breakout" in norm_type or "counter" in norm_type:
                    tactical_role = "THIRD-MAN RUNNER"
                elif "defen" in norm_type or "block" in norm_type:
                    tactical_role = "COVERING CB"
                elif "pivot" in norm_type or "midfield" in norm_type:
                    tactical_role = "MIDFIELD PIVOT"
                elif "flank" in norm_type or "wing" in norm_type:
                    tactical_role = "ISOLATED FULLBACK"
                else:
                    tactical_role = "BALL CARRIER"

        role_str = f"the {tactical_role}"
        if "goal" in norm_type:
            return f"Minute {minute}: A fatal lapse in defensive recovery leaves {role_str} unmarked, ruthlessly punishing the goalkeeper with a clinical strike."
        elif "press" in norm_type or "turnover" in norm_type:
            return f"Minute {minute}: {team} springs an aggressive pressing trap centrally; {role_str} forces the turnover to ignite an immediate transition."
        elif "transition" in norm_type or "breakout" in norm_type:
            return f"Minute {minute}: A line-breaking pass from deep unlocks the half-space, allowing {role_str} to isolate the backline in open turf."
        elif "sub" in norm_type:
            return f"Minute {minute}: A major tactical reorganization as fresh personnel enter, shifting team structure into a high-tempo offensive press."
        else:
            return f"Minute {minute}: {role_str} exploits space between the lines, executing a decisive action to breach the opposition block."
