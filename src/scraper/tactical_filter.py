"""
src/scraper/tactical_filter.py
Tactical Significance Filter & Pitch Zone Evaluation Module.

Evaluates raw match play-by-play events using a 0-100 tactical significance rubric,
prunes routine ball circulation (< 40 pts), classifies 2D pitch coordinates into
tactical zones (defensive third, middle third, attacking third, Zone 14), and emits
structured tactical events and audit reports.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("scraper.tactical_filter")


class PitchThird(str, Enum):
    DEFENSIVE = "defensive_third"
    MIDDLE = "middle_third"
    ATTACKING = "attacking_third"
    UNKNOWN = "unknown"


class PitchChannel(str, Enum):
    LEFT_WING = "left_wing"
    LEFT_HALF_SPACE = "left_half_space"
    CENTRAL = "central_channel"
    RIGHT_HALF_SPACE = "right_half_space"
    RIGHT_WING = "right_wing"
    UNKNOWN = "unknown"


class TacticalCategory(str, Enum):
    GOAL = "GOAL"
    BIG_CHANCE = "BIG_CHANCE"
    DEFENSIVE_TURNOVER = "DEFENSIVE_TURNOVER"
    BROKEN_PRESS = "BROKEN_PRESS"
    TACTICAL_SUB = "TACTICAL_SUB"
    MIDDLE_TURNOVER = "MIDDLE_TURNOVER"
    TACTICAL_FOUL = "TACTICAL_FOUL"
    ROUTINE_CIRCULATION = "ROUTINE_CIRCULATION"
    UNKNOWN = "UNKNOWN"


@dataclass
class TacticalFilterConfig:
    """Configurable scoring, zone, and pruning parameters."""
    retention_threshold: float = 75.0
    prune_threshold: float = 40.0

    # Category scoring weights
    goal_score: float = 100.0
    big_chance_score: float = 90.0
    defensive_third_turnover_score: float = 85.0
    broken_press_score: float = 80.0
    tactical_sub_score: float = 75.0
    middle_third_turnover_score: float = 65.0
    yellow_card_score: float = 40.0
    routine_pass_score: float = 15.0
    default_fallback_score: float = 20.0

    # Coordinate and metric boundaries
    xg_big_chance_threshold: float = 0.15
    defensive_third_x_limit: float = 35.0
    middle_third_x_limit: float = 65.0

    # Dynamic stalemate rebalancing
    auto_rebalance_stalemate: bool = True
    stalemate_target_min: int = 10
    stalemate_min_score: float = 60.0


@dataclass
class PitchCoordinateEvaluator:
    """Evaluates 2D pitch coordinates into tactical zones and spatial corridors."""

    @staticmethod
    def classify_third(pitch_x: Optional[float], def_limit: float = 35.0, mid_limit: float = 65.0) -> PitchThird:
        if pitch_x is None:
            return PitchThird.UNKNOWN
        if pitch_x <= def_limit:
            return PitchThird.DEFENSIVE
        if pitch_x <= mid_limit:
            return PitchThird.MIDDLE
        return PitchThird.ATTACKING

    @staticmethod
    def classify_channel(pitch_y: Optional[float]) -> PitchChannel:
        if pitch_y is None:
            return PitchChannel.UNKNOWN
        if pitch_y <= 20.0:
            return PitchChannel.LEFT_WING
        if pitch_y <= 40.0:
            return PitchChannel.LEFT_HALF_SPACE
        if pitch_y <= 60.0:
            return PitchChannel.CENTRAL
        if pitch_y <= 80.0:
            return PitchChannel.RIGHT_HALF_SPACE
        return PitchChannel.RIGHT_WING

    @staticmethod
    def identify_sub_zone(pitch_x: Optional[float], pitch_y: Optional[float]) -> Optional[str]:
        if pitch_x is None or pitch_y is None:
            return None
        # Own penalty box (defending third)
        if pitch_x <= 16.5 and 21.0 <= pitch_y <= 79.0:
            return "OWN_PENALTY_BOX"
        # Opponent penalty box (attacking third)
        if pitch_x >= 83.5 and 21.0 <= pitch_y <= 79.0:
            return "OPPONENT_PENALTY_BOX"
        # Zone 14 ("The Golden Square" central corridor outside attacking box)
        if 65.0 <= pitch_x <= 83.5 and 35.0 <= pitch_y <= 65.0:
            return "ZONE_14"
        return None

    @staticmethod
    def normalize_orientation(pitch_x: float, pitch_y: float, attacks_right: bool = True) -> Tuple[float, float]:
        """Normalizes coordinates so acting team attacks from 0 -> 100."""
        if attacks_right:
            return pitch_x, pitch_y
        return max(0.0, min(100.0, 100.0 - pitch_x)), max(0.0, min(100.0, 100.0 - pitch_y))


class TacticalSignificanceScorer:
    """
    Calculates tactical significance score (0 to 100) based on domain rubric.
    Guarantees 100% backward-compatibility with tests and downstream modules.
    """

    @staticmethod
    def score_event(event: Dict[str, Any], config: Optional[TacticalFilterConfig] = None) -> float:
        cfg = config or TacticalFilterConfig()
        event_type = str(event.get("type", "")).lower()
        xg = float(event.get("xg", 0.0) or 0.0)
        pitch_x = event.get("pitch_x")
        pitch_x_val = float(pitch_x) if pitch_x is not None else 50.0
        is_big_chance = bool(event.get("is_big_chance", False))

        # 1. Goals (Tier 1: 100 pts)
        if event_type in ("goal", "penalty_goal", "own_goal"):
            return cfg.goal_score

        # 2. Big Chances / High xG / VAR Overturns (Tier 2: 90 pts)
        if event_type == "var_overturn" or is_big_chance or xg >= cfg.xg_big_chance_threshold:
            return cfg.big_chance_score

        # 3. Defensive Third Turnovers (Tier 3: 85 pts if pitch_x <= 35.0, else 65)
        if event_type in ("turnover", "dispossessed", "ball_loss"):
            if pitch_x_val <= cfg.defensive_third_x_limit:
                return cfg.defensive_third_turnover_score
            return cfg.middle_third_turnover_score

        # 4. Broken Presses / Key Recoveries / Progressive Carries (Tier 4: 80 pts)
        if event_type in ("broken_press", "key_recovery", "progressive_carry", "line_break"):
            return cfg.broken_press_score

        # 5. Tactical Substitutions & Formation Deltas (Tier 5: 75 pts)
        if event_type in ("tactical_sub", "formation_change", "tactical_substitution"):
            return cfg.tactical_sub_score

        # 6. Yellow Cards without direct scoring chance (Tier 7: 40 pts)
        if event_type in ("yellow_card", "booking", "foul"):
            return cfg.yellow_card_score

        # 7. Routine passes & circulation (Tier 8: 15 pts -> PRUNED)
        if event_type in ("pass", "throw_in", "clearance", "touch"):
            return cfg.routine_pass_score

        # Safe fallback for missing type or unrecognized event payload (20 pts)
        return cfg.default_fallback_score

    @classmethod
    def filter_high_impact_events(
        cls, events: List[Dict[str, Any]], threshold: float = 75.0, config: Optional[TacticalFilterConfig] = None
    ) -> List[Dict[str, Any]]:
        """Filters out routine events below threshold; attaches significance_score."""
        cfg = config or TacticalFilterConfig(retention_threshold=threshold)
        retained = []
        for evt in events:
            score = cls.score_event(evt, cfg)
            evt_copy = dict(evt)
            evt_copy["significance_score"] = score
            if score >= threshold:
                retained.append(evt_copy)
        return retained


@dataclass
class TacticalFilterReport:
    """Detailed summary and audit metrics from a tactical filtering run."""
    match_id: str
    total_events_ingested: int
    total_events_retained: int
    total_events_pruned: int
    prune_rate_pct: float
    retention_threshold: float
    category_counts: Dict[str, int]
    zone_distribution: Dict[str, int]
    retained_events: List[Dict[str, Any]]
    pruned_summary: Dict[str, int]
    narrative_summary: str
    filtering_duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "total_events_ingested": self.total_events_ingested,
            "total_events_retained": self.total_events_retained,
            "total_events_pruned": self.total_events_pruned,
            "prune_rate_pct": round(self.prune_rate_pct, 2),
            "retention_threshold": self.retention_threshold,
            "category_counts": self.category_counts,
            "zone_distribution": self.zone_distribution,
            "retained_events": self.retained_events,
            "pruned_summary": self.pruned_summary,
            "narrative_summary": self.narrative_summary,
            "filtering_duration_ms": round(self.filtering_duration_ms, 2),
        }


class TacticalSignificanceFilter:
    """
    Enterprise Tactical Filter instance.
    Handles coordinate enrichment, category labeling, aggressive pruning (80-85% reduction),
    stalemate rebalancing, and report generation.
    """

    def __init__(self, config: Optional[TacticalFilterConfig] = None):
        self.config = config or TacticalFilterConfig()
        self.coord_evaluator = PitchCoordinateEvaluator()

    def score_event(self, event: Dict[str, Any]) -> float:
        return TacticalSignificanceScorer.score_event(event, self.config)

    def filter_high_impact_events(self, events: List[Dict[str, Any]], threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        t = threshold if threshold is not None else self.config.retention_threshold
        return TacticalSignificanceScorer.filter_high_impact_events(events, threshold=t, config=self.config)

    def categorize_event(self, event_type: str, score: float, xg: float, pitch_x: Optional[float]) -> TacticalCategory:
        evt_type = event_type.lower()
        if score >= 100.0 or evt_type in ("goal", "penalty_goal", "own_goal"):
            return TacticalCategory.GOAL
        if score >= 90.0 or evt_type == "var_overturn" or xg >= self.config.xg_big_chance_threshold:
            return TacticalCategory.BIG_CHANCE
        if evt_type in ("turnover", "dispossessed", "ball_loss"):
            if pitch_x is not None and pitch_x <= self.config.defensive_third_x_limit:
                return TacticalCategory.DEFENSIVE_TURNOVER
            return TacticalCategory.MIDDLE_TURNOVER
        if evt_type in ("broken_press", "key_recovery", "progressive_carry", "line_break"):
            return TacticalCategory.BROKEN_PRESS
        if evt_type in ("tactical_sub", "formation_change"):
            return TacticalCategory.TACTICAL_SUB
        if evt_type in ("yellow_card", "booking", "foul"):
            return TacticalCategory.TACTICAL_FOUL
        if evt_type in ("pass", "throw_in", "clearance", "touch"):
            return TacticalCategory.ROUTINE_CIRCULATION
        return TacticalCategory.UNKNOWN

    def generate_narrative_hook(self, event: Dict[str, Any], category: TacticalCategory, zone: PitchThird) -> str:
        minute = event.get("minute", 0)
        team = event.get("team", "Attacking Team")
        player = event.get("player") or event.get("player_in") or "Key Player"
        desc = event.get("description", "")

        if category == TacticalCategory.GOAL:
            return f"{minute}' GOAL: {player} ({team}) converts decisive chance."
        if category == TacticalCategory.BIG_CHANCE:
            xg = event.get("xg", 0.0)
            return f"{minute}' BIG CHANCE ({xg:.2f} xG): {player} penetrates defensive line."
        if category == TacticalCategory.DEFENSIVE_TURNOVER:
            return f"{minute}' HIGH-PRESS TURNOVER: {player} dispossessed in defensive third."
        if category == TacticalCategory.BROKEN_PRESS:
            return f"{minute}' PRESS BREAK: {player} executes line-breaking carry through midfield."
        if category == TacticalCategory.TACTICAL_SUB:
            delta = event.get("formation_delta", "tactical adjustment")
            return f"{minute}' TACTICAL SUB: {player} introduced ({delta})."
        return f"{minute}' {category.value}: {desc or 'Tactical action on pitch.'}"

    def filter_feed(self, events: List[Dict[str, Any]], match_id: str = "match_001") -> Tuple[List[Dict[str, Any]], TacticalFilterReport]:
        t_start = time.perf_counter()
        if not events:
            report = TacticalFilterReport(
                match_id=match_id,
                total_events_ingested=0,
                total_events_retained=0,
                total_events_pruned=0,
                prune_rate_pct=0.0,
                retention_threshold=self.config.retention_threshold,
                category_counts={},
                zone_distribution={},
                retained_events=[],
                pruned_summary={},
                narrative_summary="No events ingested.",
                filtering_duration_ms=(time.perf_counter() - t_start) * 1000.0,
            )
            return [], report

        scored_events: List[Dict[str, Any]] = []
        pruned_summary: Dict[str, int] = {}
        category_counts: Dict[str, int] = {}
        zone_counts: Dict[str, int] = {
            PitchThird.DEFENSIVE.value: 0,
            PitchThird.MIDDLE.value: 0,
            PitchThird.ATTACKING.value: 0,
        }

        for evt in events:
            score = TacticalSignificanceScorer.score_event(evt, self.config)
            pitch_x = evt.get("pitch_x")
            pitch_y = evt.get("pitch_y")
            pitch_x_val = float(pitch_x) if pitch_x is not None else None
            pitch_y_val = float(pitch_y) if pitch_y is not None else None

            third = self.coord_evaluator.classify_third(pitch_x_val, self.config.defensive_third_x_limit, self.config.middle_third_x_limit)
            channel = self.coord_evaluator.classify_channel(pitch_y_val)
            sub_zone = self.coord_evaluator.identify_sub_zone(pitch_x_val, pitch_y_val)
            category = self.categorize_event(evt.get("type", ""), score, float(evt.get("xg", 0.0) or 0.0), pitch_x_val)

            evt_enriched = dict(evt)
            evt_enriched["significance_score"] = score
            evt_enriched["category"] = category.value
            evt_enriched["pitch_third"] = third.value
            evt_enriched["pitch_channel"] = channel.value
            if sub_zone:
                evt_enriched["sub_zone"] = sub_zone
            evt_enriched["narrative_hook"] = self.generate_narrative_hook(evt, category, third)

            # Strict prune below prune_threshold (e.g. routine passes < 40)
            if score < self.config.prune_threshold:
                p_key = evt.get("type", "routine_pass").upper()
                pruned_summary[p_key] = pruned_summary.get(p_key, 0) + 1
            else:
                scored_events.append(evt_enriched)

        # Retain events >= retention_threshold (75.0)
        retained = [e for e in scored_events if e["significance_score"] >= self.config.retention_threshold]

        # Dynamic stalemate rebalancing if retained < stalemate_target_min
        if self.config.auto_rebalance_stalemate and len(retained) < self.config.stalemate_target_min:
            additional = [e for e in scored_events if e["significance_score"] < self.config.retention_threshold]
            additional.sort(key=lambda x: x["significance_score"], reverse=True)
            for cand in additional:
                if cand["significance_score"] >= self.config.stalemate_min_score:
                    if cand not in retained:
                        retained.append(cand)
                    if len(retained) >= self.config.stalemate_target_min:
                        break

        # Re-sort retained chronologically by minute and second
        retained.sort(key=lambda x: (x.get("minute", 0), x.get("second", 0)))

        # Update metrics and distribution counts
        for r_evt in retained:
            cat = r_evt["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
            z = r_evt["pitch_third"]
            if z in zone_counts:
                zone_counts[z] += 1

        total_ingested = len(events)
        total_retained = len(retained)
        total_pruned = total_ingested - total_retained
        prune_rate = (total_pruned / total_ingested * 100.0) if total_ingested > 0 else 0.0

        cat_summary_str = ", ".join(f"{count} {cat.replace('_', ' ').title()}s" for cat, count in category_counts.items())
        narrative = (
            f"Ingested {total_ingested} raw events. Pruned {total_pruned} low-significance events ({prune_rate:.1f}% prune rate). "
            f"Retained {total_retained} high-impact tactical actions: {cat_summary_str}. "
            f"Pitch zone distribution: {zone_counts.get('defensive_third', 0)} Defensive, "
            f"{zone_counts.get('middle_third', 0)} Middle, {zone_counts.get('attacking_third', 0)} Attacking Thirds."
        )

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        report = TacticalFilterReport(
            match_id=match_id,
            total_events_ingested=total_ingested,
            total_events_retained=total_retained,
            total_events_pruned=total_pruned,
            prune_rate_pct=prune_rate,
            retention_threshold=self.config.retention_threshold,
            category_counts=category_counts,
            zone_distribution=zone_counts,
            retained_events=retained,
            pruned_summary=pruned_summary,
            narrative_summary=narrative,
            filtering_duration_ms=t_elapsed_ms,
        )

        return retained, report


# Convenient Alias
TacticalFilter = TacticalSignificanceFilter
