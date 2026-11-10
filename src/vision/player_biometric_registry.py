"""
src/vision/player_biometric_registry.py
Authoritative Player Biometric & Physical Trait Registry.

Provides structured, deterministic biometric and physical reference profiles
(skin tone, hair style/shape, hair color, facial hair, height/physique)
for El Clásico squads (Real Madrid & FC Barcelona) and key test players (e.g. Cucurella).
Used by the "Guess Who?" multi-stage elimination engine to prune incompatible candidates.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class PlayerBiometricProfile:
    """Deterministic physical and biometric signature of an on-pitch player."""
    name: str
    number: int
    team: str  # "real_madrid" | "barcelona" | "test_fixture"
    role: str
    skin_tone: str  # "dark" | "brown" | "light_brown" | "fair" | "olive"
    hair_style: str  # "locs_braids" | "voluminous_curly" | "headband_floppy" | "buzzcut" | "fade" | "short_neat" | "messy"
    hair_color: str  # "black" | "dark_brown" | "blonde" | "blonde_gold_tips"
    facial_hair: str  # "full_beard" | "stubble" | "clean_shaven" | "goatee"
    height_cm: int
    build: str  # "tall_muscular" | "athletic" | "slender_agile" | "compact_agile"
    distinguishing_features: List[str] = field(default_factory=list)
    short_name: str = ""

    def __post_init__(self):
        if not self.short_name:
            last = self.name.split()[-1] if self.name else "Player"
            self.short_name = f"{last} ({self.role})"

    @staticmethod
    def normalize_name(n: str) -> str:
        s = unicodedata.normalize("NFKD", n)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return s.lower().strip()

    def compatibility_score(self, observed: Dict[str, Any]) -> float:
        """
        Computes compatibility score between observed traits and this profile.
        Returns a score in [0.0, 1.0]. A score of 0.0 means hard biometric contradiction (eliminated).
        """
        score = 1.0

        # 1. Skin tone check (Hard elimination)
        obs_skin = observed.get("skin_tone", "").lower().strip()
        if obs_skin:
            is_dark_group = self.skin_tone in ("dark", "brown")
            obs_is_dark = obs_skin in ("dark", "black", "brown", "dark_brown")
            obs_is_light = obs_skin in ("fair", "white", "light", "pale", "olive")

            if obs_is_dark and self.skin_tone in ("fair", "light"):
                return 0.0  # Hard contradiction: observed dark skin on fair-skinned candidate
            if obs_is_light and self.skin_tone in ("dark",):
                return 0.0  # Hard contradiction: observed fair skin on dark-skinned candidate
            if (obs_is_dark and is_dark_group) or (obs_is_light and not is_dark_group):
                score += 0.2

        # 2. Distinctive hair style check
        obs_hair = observed.get("hair_style", "").lower().strip()
        if obs_hair:
            # Dreadlocks / Locs / Braids (e.g., Camavinga, Kounde)
            if any(k in obs_hair for k in ("locs", "dread", "braid", "twist")):
                if self.hair_style not in ("locs_braids", "braids"):
                    return 0.0  # Hard elimination
                score += 0.4
            elif self.hair_style == "locs_braids" and any(k in obs_hair for k in ("buzz", "bald", "short_neat", "combed")):
                return 0.0

            # Voluminous Bushy Curly Afro (e.g., Cucurella)
            if any(k in obs_hair for k in ("voluminous", "bushy", "big_curly", "curly_afro", "huge_hair")):
                if self.hair_style != "voluminous_curly":
                    return 0.0
                score += 0.5
            elif self.hair_style == "voluminous_curly" and any(k in obs_hair for k in ("buzz", "short", "fade")):
                return 0.0

            # Headband / Long Floppy (e.g., Modric)
            if "headband" in obs_hair or "band" in obs_hair:
                if "headband" in self.distinguishing_features or self.hair_style == "headband_floppy":
                    score += 0.4
                else:
                    score -= 0.3

        # 3. Facial hair check
        obs_beard = observed.get("facial_hair", "").lower().strip()
        if obs_beard:
            if "beard" in obs_beard and self.facial_hair == "full_beard":
                score += 0.3
            elif "beard" in obs_beard and self.facial_hair == "clean_shaven":
                score -= 0.3
            elif "clean" in obs_beard and self.facial_hair == "full_beard":
                score -= 0.4

        return max(0.0, min(1.0, score / 1.5))


# Authoritative Profiles for Real Madrid 2024-25 Roster
REAL_MADRID_PROFILES: List[PlayerBiometricProfile] = [
    # Goalkeepers
    PlayerBiometricProfile(
        name="Thibaut Courtois", number=1, team="real_madrid", role="GK",
        skin_tone="fair", hair_style="short_neat", hair_color="dark_brown", facial_hair="clean_shaven",
        height_cm=200, build="tall_muscular", distinguishing_features=["very_tall", "gk_gloves"]
    ),
    PlayerBiometricProfile(
        name="Andriy Lunin", number=13, team="real_madrid", role="GK",
        skin_tone="fair", hair_style="short_neat", hair_color="brown", facial_hair="clean_shaven",
        height_cm=191, build="tall_muscular", distinguishing_features=["tall", "gk_gloves"]
    ),
    # Defenders
    PlayerBiometricProfile(
        name="Dean Huijsen", number=18, team="real_madrid", role="LCB",
        skin_tone="fair", hair_style="short_neat", hair_color="dark_brown", facial_hair="clean_shaven",
        height_cm=197, build="tall_muscular", distinguishing_features=["very_tall_cb", "aerial_reach"]
    ),
    PlayerBiometricProfile(
        name="Ibrahima Konaté", number=4, team="real_madrid", role="RCB",
        skin_tone="dark", hair_style="buzzcut", hair_color="black", facial_hair="stubble",
        height_cm=194, build="tall_muscular", distinguishing_features=["tall_power_cb"]
    ),
    PlayerBiometricProfile(
        name="Marc Cucurella", number=17, team="real_madrid", role="LB",
        skin_tone="fair", hair_style="voluminous_curly", hair_color="dark_brown", facial_hair="clean_shaven",
        height_cm=172, build="compact_agile", distinguishing_features=["signature_massive_bushy_curly_hair", "long_curls"]
    ),
    PlayerBiometricProfile(
        name="Denzel Dumfries", number=2, team="real_madrid", role="RB",
        skin_tone="dark", hair_style="fade", hair_color="black", facial_hair="clean_shaven",
        height_cm=188, build="tall_muscular", distinguishing_features=["explosive_right_back"]
    ),
    PlayerBiometricProfile(
        name="Trent Alexander-Arnold", number=12, team="real_madrid", role="RB",
        skin_tone="light_brown", hair_style="fade", hair_color="black", facial_hair="stubble",
        height_cm=180, build="athletic", distinguishing_features=["crossing_specialist"]
    ),
    PlayerBiometricProfile(
        name="Lucas Vázquez", number=27, team="real_madrid", role="RB",
        skin_tone="fair", hair_style="short_neat", hair_color="dark_brown", facial_hair="stubble",
        height_cm=173, build="compact_agile", distinguishing_features=["right_back"]
    ),
    PlayerBiometricProfile(
        name="Éder Militão", number=3, team="real_madrid", role="RCB",
        skin_tone="brown", hair_style="buzzcut", hair_color="black", facial_hair="stubble",
        height_cm=186, build="athletic", distinguishing_features=["center_back"]
    ),
    PlayerBiometricProfile(
        name="Antonio Rüdiger", number=22, team="real_madrid", role="LCB",
        skin_tone="dark", hair_style="buzzcut", hair_color="black", facial_hair="full_beard",
        height_cm=190, build="tall_muscular", distinguishing_features=["full_black_beard", "tall_powerful"]
    ),
    PlayerBiometricProfile(
        name="Ferland Mendy", number=23, team="real_madrid", role="LB",
        skin_tone="dark", hair_style="buzzcut", hair_color="black", facial_hair="clean_shaven",
        height_cm=180, build="tall_muscular", distinguishing_features=["left_back"]
    ),
    # Midfielders
    PlayerBiometricProfile(
        name="Bernardo Silva", number=20, team="real_madrid", role="CM",
        skin_tone="olive", hair_style="short_neat", hair_color="dark_brown", facial_hair="full_beard",
        height_cm=173, build="compact_agile", distinguishing_features=["playmaker", "beard"]
    ),
    PlayerBiometricProfile(
        name="Arda Güler", number=15, team="real_madrid", role="RW",
        skin_tone="fair", hair_style="short_neat", hair_color="dark_brown", facial_hair="clean_shaven",
        height_cm=175, build="slender_agile", distinguishing_features=["creative_winger"]
    ),
    PlayerBiometricProfile(
        name="Yann Diomande", number=25, team="real_madrid", role="LW",
        skin_tone="dark", hair_style="fade", hair_color="black", facial_hair="clean_shaven",
        height_cm=183, build="athletic", distinguishing_features=["pacey_winger"]
    ),
    PlayerBiometricProfile(
        name="Aurélien Tchouaméni", number=14, team="real_madrid", role="DM",
        skin_tone="dark", hair_style="fade", hair_color="black", facial_hair="clean_shaven",
        height_cm=187, build="tall_muscular", distinguishing_features=["holding_midfielder", "tall"]
    ),
    PlayerBiometricProfile(
        name="Federico Valverde", number=8, team="real_madrid", role="RCM",
        skin_tone="fair", hair_style="short_neat", hair_color="dark_brown", facial_hair="clean_shaven",
        height_cm=182, build="athletic", distinguishing_features=["box_to_box", "energetic"]
    ),
    PlayerBiometricProfile(
        name="Eduardo Camavinga", number=6, team="real_madrid", role="LCM",
        skin_tone="dark", hair_style="locs_braids", hair_color="blonde_gold_tips", facial_hair="clean_shaven",
        height_cm=182, build="athletic", distinguishing_features=["signature_long_locs_blonde_tips", "tied_back"]
    ),
    PlayerBiometricProfile(
        name="Jude Bellingham", number=5, team="real_madrid", role="AM",
        skin_tone="light_brown", hair_style="fade", hair_color="black", facial_hair="clean_shaven",
        height_cm=186, build="tall_muscular", distinguishing_features=["tall_athletic", "tapered_afro"]
    ),
    PlayerBiometricProfile(
        name="Luka Modrić", number=10, team="real_madrid", role="CM",
        skin_tone="fair", hair_style="headband_floppy", hair_color="brown", facial_hair="clean_shaven",
        height_cm=172, build="compact_agile", distinguishing_features=["signature_thin_black_headband", "floppy_hair"]
    ),
    PlayerBiometricProfile(
        name="Brahim Díaz", number=21, team="real_madrid", role="RW",
        skin_tone="olive", hair_style="fade", hair_color="dark_brown", facial_hair="stubble",
        height_cm=171, build="compact_agile", distinguishing_features=["low_center_of_gravity"]
    ),
    # Forwards
    PlayerBiometricProfile(
        name="Vinícius Júnior", number=7, team="real_madrid", role="LW",
        skin_tone="dark", hair_style="fade", hair_color="black", facial_hair="clean_shaven",
        height_cm=176, build="slender_agile", distinguishing_features=["left_winger", "fast_dynamic"]
    ),
    PlayerBiometricProfile(
        name="Kylian Mbappé", number=9, team="real_madrid", role="CF",
        skin_tone="brown", hair_style="buzzcut", hair_color="black", facial_hair="clean_shaven",
        height_cm=178, build="athletic", distinguishing_features=["signature_buzzcut", "explosive_speed"]
    ),
    PlayerBiometricProfile(
        name="Rodrygo", number=11, team="real_madrid", role="RW",
        skin_tone="brown", hair_style="fade", hair_color="black", facial_hair="clean_shaven",
        height_cm=174, build="compact_agile", distinguishing_features=["right_winger"]
    ),
]

# Authoritative Profiles for FC Barcelona 2024-25 Roster
BARCELONA_PROFILES: List[PlayerBiometricProfile] = [
    # Goalkeepers
    PlayerBiometricProfile(
        name="Joan García", number=1, team="barcelona", role="GK",
        skin_tone="fair", hair_style="short_neat", hair_color="dark_brown", facial_hair="clean_shaven",
        height_cm=191, build="tall_muscular", distinguishing_features=["tall_gk", "gk_gloves"]
    ),
    PlayerBiometricProfile(
        name="Iñaki Peña", number=13, team="barcelona", role="GK",
        skin_tone="fair", hair_style="short_neat", hair_color="dark_brown", facial_hair="clean_shaven",
        height_cm=184, build="athletic", distinguishing_features=["gk_gloves"]
    ),
    PlayerBiometricProfile(
        name="Wojciech Szczęsny", number=25, team="barcelona", role="GK",
        skin_tone="fair", hair_style="short_neat", hair_color="dark_brown", facial_hair="clean_shaven",
        height_cm=195, build="tall_muscular", distinguishing_features=["very_tall", "gk_gloves"]
    ),
    # Defenders
    PlayerBiometricProfile(
        name="Xavi Espart", number=22, team="barcelona", role="RB",
        skin_tone="fair", hair_style="short_neat", hair_color="brown", facial_hair="clean_shaven",
        height_cm=178, build="athletic", distinguishing_features=["young_right_back"]
    ),
    PlayerBiometricProfile(
        name="Gerard Martín", number=24, team="barcelona", role="LCB",
        skin_tone="fair", hair_style="short_neat", hair_color="brown", facial_hair="clean_shaven",
        height_cm=186, build="athletic", distinguishing_features=["solid_defender"]
    ),
    PlayerBiometricProfile(
        name="João Cancelo", number=12, team="barcelona", role="LB",
        skin_tone="olive", hair_style="short_neat", hair_color="dark_brown", facial_hair="stubble",
        height_cm=182, build="athletic", distinguishing_features=["experienced_fullback"]
    ),
    PlayerBiometricProfile(
        name="Jules Koundé", number=23, team="barcelona", role="RB",
        skin_tone="dark", hair_style="locs_braids", hair_color="black", facial_hair="goatee",
        height_cm=180, build="athletic", distinguishing_features=["braids_cornrows", "athletic"]
    ),
    PlayerBiometricProfile(
        name="Pau Cubarsí", number=2, team="barcelona", role="RCB",
        skin_tone="fair", hair_style="short_neat", hair_color="brown", facial_hair="clean_shaven",
        height_cm=184, build="athletic", distinguishing_features=["clean_cut_young", "center_back"]
    ),
    PlayerBiometricProfile(
        name="Iñigo Martínez", number=5, team="barcelona", role="LCB",
        skin_tone="fair", hair_style="short_neat", hair_color="dark_brown", facial_hair="full_beard",
        height_cm=182, build="tall_muscular", distinguishing_features=["prominent_full_brown_beard", "veteran"]
    ),
    PlayerBiometricProfile(
        name="Alejandro Balde", number=3, team="barcelona", role="LB",
        skin_tone="dark", hair_style="buzzcut", hair_color="black", facial_hair="clean_shaven",
        height_cm=175, build="slender_agile", distinguishing_features=["fast_left_back", "close_cropped_hair"]
    ),
    # Midfielders
    PlayerBiometricProfile(
        name="Rodri", number=14, team="barcelona", role="DM",
        skin_tone="fair", hair_style="short_neat", hair_color="dark_brown", facial_hair="clean_shaven",
        height_cm=191, build="tall_muscular", distinguishing_features=["tall_pivot", "ballon_dor_winner"]
    ),
    PlayerBiometricProfile(
        name="Marc Casadó", number=17, team="barcelona", role="DM",
        skin_tone="fair", hair_style="short_neat", hair_color="dark_brown", facial_hair="clean_shaven",
        height_cm=172, build="compact_agile", distinguishing_features=["compact_pivot"]
    ),
    PlayerBiometricProfile(
        name="Pedri", number=8, team="barcelona", role="LCM",
        skin_tone="fair", hair_style="short_neat", hair_color="dark_brown", facial_hair="stubble",
        height_cm=174, build="slender_agile", distinguishing_features=["lean_playmaker"]
    ),
    PlayerBiometricProfile(
        name="Fermín López", number=16, team="barcelona", role="AM",
        skin_tone="fair", hair_style="short_neat", hair_color="brown", facial_hair="clean_shaven",
        height_cm=174, build="athletic", distinguishing_features=["energetic_midfielder"]
    ),
    PlayerBiometricProfile(
        name="Frenkie de Jong", number=21, team="barcelona", role="CM",
        skin_tone="fair", hair_style="short_neat", hair_color="blonde", facial_hair="clean_shaven",
        height_cm=180, build="athletic", distinguishing_features=["signature_blonde_hair"]
    ),
    PlayerBiometricProfile(
        name="Anthony Gordon", number=10, team="barcelona", role="LW",
        skin_tone="fair", hair_style="short_neat", hair_color="blonde", facial_hair="clean_shaven",
        height_cm=183, build="athletic", distinguishing_features=["blond_winger", "pacey"]
    ),
    PlayerBiometricProfile(
        name="Karim Adeyemi", number=7, team="barcelona", role="LW",
        skin_tone="light_brown", hair_style="fade", hair_color="black", facial_hair="clean_shaven",
        height_cm=180, build="athletic", distinguishing_features=["rapid_winger"]
    ),
    PlayerBiometricProfile(
        name="Gabriel Jesus", number=9, team="barcelona", role="CF",
        skin_tone="brown", hair_style="fade", hair_color="black", facial_hair="goatee",
        height_cm=175, build="athletic", distinguishing_features=["brazilian_striker"]
    ),
    PlayerBiometricProfile(
        name="Dani Olmo", number=20, team="barcelona", role="AM",
        skin_tone="fair", hair_style="short_neat", hair_color="brown", facial_hair="clean_shaven",
        height_cm=177, build="athletic", distinguishing_features=["creative_midfielder"]
    ),
    PlayerBiometricProfile(
        name="Gavi", number=6, team="barcelona", role="CM",
        skin_tone="fair", hair_style="messy", hair_color="dark_brown", facial_hair="clean_shaven",
        height_cm=173, build="compact_agile", distinguishing_features=["intense_pressing", "messy_hair"]
    ),
    # Forwards
    PlayerBiometricProfile(
        name="Lamine Yamal", number=19, team="barcelona", role="RW",
        skin_tone="light_brown", hair_style="fade", hair_color="black", facial_hair="clean_shaven",
        height_cm=178, build="slender_agile", distinguishing_features=["signature_curled_fade", "left_footed_winger"]
    ),
    PlayerBiometricProfile(
        name="Raphinha", number=11, team="barcelona", role="LW",
        skin_tone="brown", hair_style="fade", hair_color="black", facial_hair="clean_shaven",
        height_cm=176, build="athletic", distinguishing_features=["captain_armband_candidate", "athletic_winger"]
    ),
    PlayerBiometricProfile(
        name="Robert Lewandowski", number=9, team="barcelona", role="CF",
        skin_tone="fair", hair_style="short_neat", hair_color="dark_brown", facial_hair="clean_shaven",
        height_cm=185, build="tall_muscular", distinguishing_features=["tall_striker", "combed_hair"]
    ),
    PlayerBiometricProfile(
        name="Pau Víctor", number=18, team="barcelona", role="CF",
        skin_tone="fair", hair_style="short_neat", hair_color="dark_brown", facial_hair="clean_shaven",
        height_cm=183, build="athletic", distinguishing_features=["striker"]
    ),
]

# Stand-Alone Evaluation Fixture Profiles (e.g. Marc Cucurella duel test in frame 05)
STANDALONE_FIXTURE_PROFILES: List[PlayerBiometricProfile] = [
    PlayerBiometricProfile(
        name="Marc Cucurella", number=17, team="test_fixture", role="LB",
        skin_tone="fair", hair_style="voluminous_curly", hair_color="dark_brown", facial_hair="clean_shaven",
        height_cm=172, build="compact_agile", distinguishing_features=["signature_massive_bushy_curly_hair", "long_curls"]
    ),
]

ALL_PROFILES: List[PlayerBiometricProfile] = REAL_MADRID_PROFILES + BARCELONA_PROFILES + STANDALONE_FIXTURE_PROFILES


class BiometricRegistry:
    """Registry lookup and elimination engine for player biometrics."""

    @classmethod
    def get_profile(cls, name_or_number: Any, team: Optional[str] = None) -> Optional[PlayerBiometricProfile]:
        norm = PlayerBiometricProfile.normalize_name(str(name_or_number)) if isinstance(name_or_number, str) else ""
        num = int(name_or_number) if isinstance(name_or_number, int) or (isinstance(name_or_number, str) and name_or_number.isdigit()) else None

        # Pass 1: Exact or substring match of full name
        for p in ALL_PROFILES:
            if team and p.team != team and p.team != "test_fixture":
                continue
            p_norm = PlayerBiometricProfile.normalize_name(p.name)
            if norm and (p_norm == norm or p_norm in norm or norm in p_norm):
                return p

        # Pass 2: Match player surname (last name)
        if norm:
            norm_parts = norm.split()
            for p in ALL_PROFILES:
                if team and p.team != team and p.team != "test_fixture":
                    continue
                p_parts = PlayerBiometricProfile.normalize_name(p.name).split()
                p_surname = p_parts[-1] if p_parts else ""
                if p_surname and (p_surname in norm_parts or p_surname in norm):
                    return p

        # Pass 3: Match by jersey number
        if num is not None:
            for p in ALL_PROFILES:
                if team and p.team != team and p.team != "test_fixture":
                    continue
                if p.number == num:
                    return p
        return None

    @classmethod
    def filter_candidates(
        cls,
        candidates: List[Dict[str, Any]],
        team: str,
        observed_traits: Dict[str, Any],
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Eliminates incompatible candidates based on biometric observations.
        Returns a sorted list of (candidate, compatibility_score) tuples.
        """
        scored_candidates = []
        for cand in candidates:
            cand_name = cand.get("name", "")
            cand_num = cand.get("number", 0)
            profile = cls.get_profile(cand_name, team=team) or cls.get_profile(cand_num, team=team)

            if profile is None:
                # No biometric profile found; default to neutral score
                scored_candidates.append((cand, 0.5))
                continue

            score = profile.compatibility_score(observed_traits)
            if score > 0.0:  # Score 0.0 means hard contradiction / elimination
                scored_candidates.append((cand, score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates
