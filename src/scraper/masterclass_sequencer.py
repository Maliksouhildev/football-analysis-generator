"""
src/scraper/masterclass_sequencer.py
Masterclass Event Sequencer & Remotion Adapter Module.

Selects 10–14 high-impact tactical actions chronologically distributed across
Minute 1' to 90'+, enforces anti-clustering narrative dispersion, generates
commentary directives and lower thirds, and exports canonical TacticalEvent and
TacticalSegment interfaces for Remotion rendering and n8n orchestration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("scraper.masterclass_sequencer")


# ---------------------------------------------------------------------------
# 1. Canonical Domain Contracts & Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CommentaryDirective:
    """Directives guiding the downstream LLM scriptwriter and Voice Director AI."""
    analytical_focus: str
    narration_tone: str
    cadence_cue: str
    key_stats_citation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analytical_focus": self.analytical_focus,
            "narration_tone": self.narration_tone,
            "cadence_cue": self.cadence_cue,
            "key_stats_citation": self.key_stats_citation,
        }


@dataclass
class TacticalEvent:
    """
    Canonical Tactical Event Contract defined in PROJECT.md and spec_r1_r2.md §3.3.
    """
    id: str
    match_minute: int
    stoppage_minute: int
    event_type: str
    tactical_score: float
    team: str
    players_involved: List[str]
    narrative_hook: str
    second: int = 0
    period: int = 1
    pitch_x: float = 50.0
    pitch_y: float = 50.0
    xg: float = 0.0
    is_big_chance: bool = False
    description: str = ""
    formation_delta: Optional[str] = None
    player_in: Optional[str] = None
    player_out: Optional[str] = None
    commentary_directive: Optional[CommentaryDirective] = None
    lower_third: Optional[Dict[str, str]] = None
    telestration: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "match_minute": self.match_minute,
            "stoppage_minute": self.stoppage_minute,
            "event_type": self.event_type,
            "tactical_score": self.tactical_score,
            "team": self.team,
            "players_involved": self.players_involved,
            "narrative_hook": self.narrative_hook,
            "second": self.second,
            "period": self.period,
            "pitch_x": self.pitch_x,
            "pitch_y": self.pitch_y,
            "xg": self.xg,
            "is_big_chance": self.is_big_chance,
            "description": self.description,
            "formation_delta": self.formation_delta,
            "player_in": self.player_in,
            "player_out": self.player_out,
            "commentary_directive": self.commentary_directive.to_dict() if self.commentary_directive else None,
            "lower_third": self.lower_third,
            "telestration": self.telestration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TacticalEvent:
        players = data.get("players_involved") or []
        if not players and data.get("player"):
            players = [str(data["player"])]
        if data.get("player_in") and data["player_in"] not in players:
            players.append(str(data["player_in"]))
        if data.get("player_out") and data["player_out"] not in players:
            players.append(str(data["player_out"]))

        cd_data = data.get("commentary_directive")
        cd = CommentaryDirective(**cd_data) if isinstance(cd_data, dict) else None

        minute = int(data.get("minute") or data.get("match_minute") or 0)
        default_period = 1 if minute <= 45 else 2
        period = int(data.get("period") or default_period)
        stoppage = int(data.get("stoppage_minute") or 0)
        second = int(data.get("second") or 0)

        raw_px = data.get("pitch_x")
        raw_py = data.get("pitch_y")
        px = min(100.0, max(0.0, float(raw_px if raw_px is not None else 50.0)))
        py = min(100.0, max(0.0, float(raw_py if raw_py is not None else 50.0)))
        xg = float(data.get("xg") or 0.0)
        score = float(data.get("significance_score") or data.get("tactical_score") or 0.0)

        return cls(
            id=str(data.get("id") or ""),
            match_minute=minute,
            stoppage_minute=stoppage,
            event_type=str(data.get("type") or data.get("event_type") or "pass"),
            tactical_score=score,
            team=str(data.get("team") or ""),
            players_involved=players,
            narrative_hook=str(data.get("narrative_hook") or ""),
            second=second,
            period=period,
            pitch_x=px,
            pitch_y=py,
            xg=xg,
            is_big_chance=bool(data.get("is_big_chance") or False),
            description=str(data.get("description") or ""),
            formation_delta=data.get("formation_delta"),
            player_in=data.get("player_in"),
            player_out=data.get("player_out"),
            commentary_directive=cd,
            lower_third=data.get("lower_third"),
            telestration=data.get("telestration"),
        )


@dataclass
class TacticalSegment:
    """Remotion Composition Segment matching TacticalMasterclass.tsx:20-35."""
    id: str
    type: str                  # 'clip' | 'freeze_frame'
    url: str
    start_time: float
    duration: float
    title: str
    subtitle: str
    match_minute: int
    lower_third: Dict[str, str]
    telestration: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "url": self.url,
            "startTime": round(self.start_time, 2),
            "duration": round(self.duration, 2),
            "title": self.title,
            "subtitle": self.subtitle,
            "matchMinute": self.match_minute,
            "lowerThird": self.lower_third,
            "telestration": self.telestration,
        }


# ---------------------------------------------------------------------------
# 2. Domain Helper Generators (Directives, Hooks, Lower Thirds)
# ---------------------------------------------------------------------------

class TacticalMetadataSynthesizer:
    """Generates narrative hooks, commentary directives, and lower-third banners."""

    @staticmethod
    def generate_narrative_hook(event: Dict[str, Any]) -> str:
        evt_type = str(event.get("type") or "").lower()
        minute = int(event.get("minute") or 0)
        stoppage = int(event.get("stoppage_minute") or 0)
        clock_str = f"{minute}'+{stoppage}'" if stoppage > 0 else f"{minute}'"
        player = event.get("player") or event.get("player_in") or "Key Player"
        team = event.get("team") or "Team"
        xg = float(event.get("xg") or 0.0)
        desc = event.get("description") or ""
        px = float(event.get("pitch_x") if event.get("pitch_x") is not None else 50.0)

        if evt_type in ("goal", "penalty_goal"):
            detail = f"Decisive {xg:.2f} xG finish by {player} ({team})"
            if desc:
                detail += f" following {str(desc).lower()}"
            return f"Minute {clock_str}: {detail}."
        if evt_type == "var_overturn":
            return f"Minute {clock_str}: VAR review overturns goal by {player} ({team}) due to offside."
        if evt_type == "big_chance":
            return f"Minute {clock_str}: High-value scoring chance ({xg:.2f} xG) for {player} ({team}) penetrating the box."
        if evt_type == "turnover":
            zone = "defensive third" if px <= 35.0 else "midfield"
            return f"Minute {clock_str}: Critical ball loss by {player} ({team}) in {zone} under high-intensity press."
        if evt_type in ("broken_press", "progressive_carry"):
            return f"Minute {clock_str}: Line-breaking progression by {player} ({team}) beating opponent defensive corridor."
        if evt_type == "tactical_sub":
            delta = event.get("formation_delta") or "tactical adjustment"
            p_in = event.get("player_in") or "Attacking sub"
            p_out = event.get("player_out") or "Starter"
            return f"Minute {clock_str}: Tactical shift by {team}: {p_in} on for {p_out} ({delta})."

        return f"Minute {clock_str}: Tactical sequence involving {player} ({team}) — {desc or 'Key action'}."

    @staticmethod
    def generate_commentary_directive(event: Dict[str, Any]) -> CommentaryDirective:
        evt_type = str(event.get("type") or "").lower()
        player = event.get("player") or "Player"
        xg = float(event.get("xg") or 0.0)
        pitch_x = float(event.get("pitch_x") if event.get("pitch_x") is not None else 50.0)

        if evt_type in ("goal", "penalty_goal"):
            return CommentaryDirective(
                analytical_focus="Defensive backline disorganization and striker spatial execution.",
                narration_tone="urgent-analytical",
                cadence_cue="[PAUSE 0.4s] Notice the defender tracking the decoy run, [BREATH] leaving the corridor exposed.",
                key_stats_citation=f"xG: {xg:.2f} | Pitch X: {pitch_x:.1f} | Goal Scored"
            )
        if evt_type == "var_overturn":
            return CommentaryDirective(
                analytical_focus="Defensive line coordination, offside trap execution, and VAR review impact.",
                narration_tone="dramatic-authoritative",
                cadence_cue="[PAUSE 0.5s] The initial euphoria fades [BREATH] as VAR detects the shoulder offside.",
                key_stats_citation="VAR Overturn // Disallowed Goal"
            )
        if evt_type == "turnover":
            return CommentaryDirective(
                analytical_focus="Pressing trigger activation, cover shadows, and counter-attack acceleration.",
                narration_tone="high-tempo-critical",
                cadence_cue="[PAUSE 0.3s] The trap snaps shut in the defensive third [BREATH] leading to immediate danger.",
                key_stats_citation=f"Turnover Zone: {'Defensive 3rd' if pitch_x <= 35 else 'Midfield'} (X={pitch_x:.1f})"
            )
        if evt_type in ("broken_press", "progressive_carry"):
            return CommentaryDirective(
                analytical_focus="Press evasion, body feinting, and third-man forward progression.",
                narration_tone="measured-appreciative",
                cadence_cue="[PAUSE 0.4s] Watch how one progressive touch eliminates three pressing midfielders.",
                key_stats_citation=f"Line-Breaking Progressive Carry | Player: {player}"
            )
        if evt_type == "tactical_sub":
            delta = event.get("formation_delta") or "Formation Shift"
            return CommentaryDirective(
                analytical_focus=f"Tactical re-alignment to {delta}, altering structural width and pressing engagement.",
                narration_tone="tactical-reflective",
                cadence_cue="[PAUSE 0.6s] A calculated structural change from the bench [BREATH] shifting momentum.",
                key_stats_citation=f"Tactical Sub // Delta: {delta}"
            )

        return CommentaryDirective(
            analytical_focus="Positional structure and passing lane occupancy.",
            narration_tone="measured-explanatory",
            cadence_cue="[PAUSE 0.4s] Tracking team shape across the pitch.",
            key_stats_citation=f"Minute {event.get('minute') or 0} Action"
        )

    @staticmethod
    def format_lower_third(event: Dict[str, Any]) -> Dict[str, str]:
        minute = int(event.get("minute") or 0)
        stoppage = int(event.get("stoppage_minute") or 0)
        clock_tag = f"{minute}'+{stoppage}'" if stoppage > 0 else f"{minute}'"
        player = event.get("player") or event.get("player_in") or "TACTICAL KEY"
        team = event.get("team") or "MATCH ANALYSIS"
        evt_type = str(event.get("type") or "").upper()
        xg = float(event.get("xg") or 0.0)
        px = float(event.get("pitch_x") if event.get("pitch_x") is not None else 50.0)

        line1 = f"{player.upper()} ({team.upper()})"

        if evt_type in ("GOAL", "PENALTY_GOAL"):
            line2 = f"GOAL SCORED // {xg:.2f} xG // HIGH TACTICAL IMPACT"
        elif evt_type == "VAR_OVERTURN":
            line2 = "DISALLOWED GOAL // VAR OFFSIDE OVERTURN"
        elif evt_type == "BIG_CHANCE":
            line2 = f"BIG CHANCE // {xg:.2f} xG // BOX PENETRATION"
        elif evt_type == "TURNOVER":
            zone = "DEFENSIVE THIRD" if px <= 35.0 else "MIDFIELD"
            line2 = f"BALL LOSS // {zone} PRESS EXPLOIT"
        elif evt_type in ("BROKEN_PRESS", "PROGRESSIVE_CARRY"):
            line2 = "PRESS EVASION // LINE-BREAKING TRANSITION"
        elif evt_type == "TACTICAL_SUB":
            delta = str(event.get("formation_delta") or "TACTICAL SHIFT").upper()
            line2 = f"FORMATION SHIFT // {delta}"
        else:
            line2 = str(event.get("description") or "TACTICAL SEQUENCE").upper()

        return {
            "line1": line1,
            "line2": line2,
            "tag": f"TACTICAL INFLECTION // {clock_tag}",
        }


# ---------------------------------------------------------------------------
# 3. Masterclass Event Sequencer
# ---------------------------------------------------------------------------

class MasterclassEventSequencer:
    """
    Selects 10–14 chronological tactical events distributed across Min 1' to 90'+.
    Adheres strictly to the test suite and domain specifications.
    """

    @staticmethod
    def _score_event(event: Dict[str, Any]) -> float:
        """Calculates tactical significance score if event is not pre-scored."""
        if "significance_score" in event and event["significance_score"] is not None:
            return float(event["significance_score"])
        from src.scraper.tactical_filter import TacticalSignificanceScorer
        return TacticalSignificanceScorer.score_event(event)

    @classmethod
    def _chronological_sort_key(cls, event: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """
        Calculates monotonic chronological sort key with stoppage time and priority.
        Tuple: (period, minute, effective_second, type_priority)
        """
        minute = int(event.get("minute") or 0)
        second = int(event.get("second") or 0)
        stoppage = int(event.get("stoppage_minute") or 0)

        # Determine period
        period = event.get("period")
        if period is None:
            period = 1 if minute <= 45 else 2
        period = int(period)

        # Stoppage time offset
        effective_second = second + (stoppage * 60)

        # Sub-minute tie-break hierarchy
        type_prio = {
            "goal": 0,
            "penalty_goal": 0,
            "var_overturn": 1,
            "big_chance": 2,
            "turnover": 3,
            "broken_press": 4,
            "progressive_carry": 5,
            "key_recovery": 6,
            "tactical_sub": 7,
            "yellow_card": 8,
            "pass": 9,
        }.get(str(event.get("type", "")).lower(), 10)

        return (period, minute, effective_second, type_prio)

    @classmethod
    def sequence_events(
        cls,
        events: List[Dict[str, Any]],
        target_min: int = 10,
        target_max: int = 14,
        enforce_temporal_spread: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Sequences match events into a 10–14 event masterclass narrative.

        Args:
            events: Raw or scored play-by-play events feed.
            target_min: Minimum segments desired (default 10).
            target_max: Maximum segments desired (default 14).
            enforce_temporal_spread: Whether to enforce anti-clustering.

        Returns:
            List of sequenced, enriched event dictionaries matching test assertions.
        """
        if not events:
            return []

        # 1. Score all events and synthesize metadata
        scored_events: List[Dict[str, Any]] = []
        for evt in events:
            s_evt = dict(evt)
            score = cls._score_event(evt)
            s_evt["significance_score"] = score
            if "narrative_hook" not in s_evt:
                s_evt["narrative_hook"] = TacticalMetadataSynthesizer.generate_narrative_hook(s_evt)
            if "lower_third" not in s_evt:
                s_evt["lower_third"] = TacticalMetadataSynthesizer.format_lower_third(s_evt)
            scored_events.append(s_evt)

        # Filter out routine passes (< 40.0) from consideration
        valid_events = [
            e for e in scored_events
            if e["significance_score"] >= 40.0 and str(e.get("type", "")).lower() != "pass"
        ]

        if not valid_events:
            return []

        valid_events.sort(key=cls._chronological_sort_key)

        # Boundary check: If total valid events <= target_min, return all valid events
        if len(valid_events) <= target_min:
            return valid_events

        # Identify goals
        goals = [
            e for e in valid_events
            if e["significance_score"] >= 100.0 or str(e.get("type", "")).lower() in ("goal", "penalty_goal", "own_goal")
        ]

        # Case 1: Total goals exceed target_max (e.g. 20 goals)
        if len(goals) >= target_max:
            n = len(goals)
            k = target_max
            # Uniform Bresenham decimation across the entire match
            indices = [int(i * (n - 1) / (k - 1)) for i in range(k)]
            high_impact = [goals[idx] for idx in indices]
        else:
            # Case 2: Goals fit within target_max, preserve ALL goals
            high_impact = list(goals)
            remaining_slots = target_max - len(high_impact)

            non_goals = [e for e in valid_events if e not in goals]

            # Pool candidates by priority: essential (>=90), high (>=75), medium (>=40)
            t90 = [e for e in non_goals if e["significance_score"] >= 90.0]
            t75 = [e for e in non_goals if 75.0 <= e["significance_score"] < 90.0]
            t40 = [e for e in non_goals if 40.0 <= e["significance_score"] < 75.0]

            if len(t90) >= remaining_slots:
                n = len(t90)
                k = remaining_slots
                if k == 1:
                    selected_t90 = [t90[n // 2]]
                else:
                    indices = [int(i * (n - 1) / (k - 1)) for i in range(k)]
                    selected_t90 = [t90[idx] for idx in indices]
                high_impact.extend(selected_t90)
            else:
                high_impact.extend(t90)
                remaining_slots = target_max - len(high_impact)

                candidates_next = t75 + (t40 if len(high_impact) + len(t75) < target_min else [])
                if candidates_next and remaining_slots > 0:
                    if len(candidates_next) <= remaining_slots:
                        if len(high_impact) < target_min:
                            candidates_next.sort(key=lambda x: x["significance_score"], reverse=True)
                            for cand in candidates_next:
                                if cand not in high_impact:
                                    high_impact.append(cand)
                                if len(high_impact) >= target_min:
                                    break
                        else:
                            high_impact.extend(candidates_next[:remaining_slots])
                    else:
                        n = len(candidates_next)
                        k = remaining_slots
                        if k == 1:
                            selected_next = [candidates_next[n // 2]]
                        else:
                            indices = [int(i * (n - 1) / (k - 1)) for i in range(k)]
                            selected_next = [candidates_next[idx] for idx in indices]
                        high_impact.extend(selected_next)

        # Underflow rebalancing if len(high_impact) < target_min
        if len(high_impact) < target_min:
            lower_pool = [
                e for e in valid_events
                if e not in high_impact and e["significance_score"] >= 40.0 and str(e.get("type", "")).lower() != "pass"
            ]
            lower_pool.sort(key=lambda x: x["significance_score"], reverse=True)
            for cand in lower_pool:
                high_impact.append(cand)
                if len(high_impact) >= target_min:
                    break

        high_impact.sort(key=cls._chronological_sort_key)

        # 6. Half Dispersion Enforcement (>= 3 in H1 <= 45', >= 3 in H2 > 45')
        first_half = [e for e in high_impact if int(e.get("minute") or 0) <= 45]
        second_half = [e for e in high_impact if int(e.get("minute") or 0) > 45]

        feed_h1 = [
            e for e in valid_events
            if int(e.get("minute") or 0) <= 45 and e not in high_impact
            and e["significance_score"] >= 40.0 and str(e.get("type", "")).lower() != "pass"
        ]
        feed_h2 = [
            e for e in valid_events
            if int(e.get("minute") or 0) > 45 and e not in high_impact
            and e["significance_score"] >= 40.0 and str(e.get("type", "")).lower() != "pass"
        ]

        # Balance second half if < 3
        while len(second_half) < 3 and feed_h2 and len(high_impact) >= target_min:
            feed_h2.sort(key=lambda x: x["significance_score"], reverse=True)
            new_h2 = feed_h2.pop(0)

            if len(high_impact) >= target_max:
                h1_non_goals = [
                    e for e in high_impact
                    if int(e.get("minute") or 0) <= 45 and str(e.get("type", "")).lower() not in ("goal", "penalty_goal", "own_goal")
                ]
                if h1_non_goals:
                    h1_non_goals.sort(key=lambda x: x["significance_score"])
                    high_impact.remove(h1_non_goals[0])
                else:
                    h1_goals = [e for e in high_impact if int(e.get("minute") or 0) <= 45]
                    if len(h1_goals) > 3:
                        high_impact.remove(h1_goals[1])
                    else:
                        break

            high_impact.append(new_h2)
            first_half = [e for e in high_impact if int(e.get("minute") or 0) <= 45]
            second_half = [e for e in high_impact if int(e.get("minute") or 0) > 45]

        # Balance first half if < 3
        while len(first_half) < 3 and feed_h1 and len(high_impact) >= target_min:
            feed_h1.sort(key=lambda x: x["significance_score"], reverse=True)
            new_h1 = feed_h1.pop(0)

            if len(high_impact) >= target_max:
                h2_non_goals = [
                    e for e in high_impact
                    if int(e.get("minute") or 0) > 45 and str(e.get("type", "")).lower() not in ("goal", "penalty_goal", "own_goal")
                ]
                if h2_non_goals:
                    h2_non_goals.sort(key=lambda x: x["significance_score"])
                    high_impact.remove(h2_non_goals[0])
                else:
                    h2_goals = [e for e in high_impact if int(e.get("minute") or 0) > 45]
                    if len(h2_goals) > 3:
                        high_impact.remove(h2_goals[1])
                    else:
                        break

            high_impact.append(new_h1)
            first_half = [e for e in high_impact if int(e.get("minute") or 0) <= 45]
            second_half = [e for e in high_impact if int(e.get("minute") or 0) > 45]

        high_impact.sort(key=cls._chronological_sort_key)
        return high_impact

    @classmethod
    def sequence_to_canonical(
        cls,
        events: List[Dict[str, Any]],
        target_min: int = 10,
        target_max: int = 14
    ) -> List[TacticalEvent]:
        """Sequences raw events and returns typed TacticalEvent dataclass instances."""
        dict_events = cls.sequence_events(events, target_min=target_min, target_max=target_max)
        canonical_events: List[TacticalEvent] = []
        for e in dict_events:
            event_obj = TacticalEvent.from_dict(e)
            event_obj.commentary_directive = TacticalMetadataSynthesizer.generate_commentary_directive(e)
            canonical_events.append(event_obj)
        return canonical_events


# ---------------------------------------------------------------------------
# 4. Remotion Masterclass Payload Adapter
# ---------------------------------------------------------------------------

class RemotionMasterclassAdapter:
    """Adapts sequenced tactical events into Remotion video composition payloads."""

    @classmethod
    def assemble_segments(
        cls,
        events: List[Dict[str, Any]],
        total_duration: float = 540.0,
        assets_base_url: str = "http://127.0.0.1:3000/assets",
        dual_segment_mode: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Assembles segments array matching TacticalMasterclass.tsx:20-35.
        Generates paired (freeze_frame first with tactical overlay, then clean clip) segments.
        """
        num_events = len(events)
        if num_events == 0:
            return []

        segments: List[Dict[str, Any]] = []
        per_event_time = total_duration / num_events
        current_time = 0.0

        for idx, evt in enumerate(events):
            minute = evt.get("minute", 0)
            evt_id = evt.get("id", f"evt_{idx}")
            lower_third = evt.get("lower_third") or TacticalMetadataSynthesizer.format_lower_third(evt)
            title = f"MINUTE {minute}' ACTION"
            subtitle = evt.get("description") or evt.get("narrative_hook") or "Tactical Breakdown"

            if dual_segment_mode:
                # 1. Tactical Setup Freeze-Frame FIRST (3.2s analysis with telestration)
                freeze_duration = min(4.0, max(2.5, per_event_time * 0.4))
                # 2. Pure Broadcast Action Clip SECOND (execution with zero overlay clutter)
                clip_duration = max(3.0, per_event_time - freeze_duration)

                segments.append({
                    "id": f"seg_{idx}_freeze_{minute}m",
                    "type": "freeze_frame",
                    "url": f"{assets_base_url}/frames/frame_{evt_id}_{minute}m.jpg",
                    "startTime": round(current_time, 2),
                    "duration": round(freeze_duration, 2),
                    "title": f"MINUTE {minute}' TACTICAL BREAKDOWN",
                    "subtitle": subtitle,
                    "matchMinute": minute,
                    "lowerThird": lower_third,
                    "telestration": evt.get("telestration"),
                    "video_pts": evt.get("video_pts"),
                })
                current_time += freeze_duration

                segments.append({
                    "id": f"seg_{idx}_clip_{minute}m",
                    "type": "clip",
                    "url": f"{assets_base_url}/clips/clip_{evt_id}_{minute}m.mp4",
                    "startTime": round(current_time, 2),
                    "duration": round(clip_duration, 2),
                    "title": title,
                    "subtitle": subtitle,
                    "matchMinute": minute,
                    "lowerThird": lower_third,
                    "telestration": None,  # Strictly clean broadcast action
                    "video_pts": evt.get("video_pts"),
                })
                current_time += clip_duration
            else:
                seg_type = "clip" if idx % 2 == 0 else "freeze_frame"
                asset_url = (
                    f"{assets_base_url}/clips/clip_{evt_id}_{minute}m.mp4"
                    if seg_type == "clip"
                    else f"{assets_base_url}/frames/frame_{evt_id}_{minute}m.jpg"
                )
                segments.append({
                    "id": f"seg_{idx}_{minute}m",
                    "type": seg_type,
                    "url": asset_url,
                    "startTime": round(current_time, 2),
                    "duration": round(per_event_time, 2),
                    "title": title,
                    "subtitle": subtitle,
                    "matchMinute": minute,
                    "lowerThird": lower_third,
                    "telestration": evt.get("telestration"),
                })
                current_time += per_event_time

        return segments

    @classmethod
    def build_render_payload(
        cls,
        events: List[Dict[str, Any]],
        match_title: str = "TACTICAL MASTERCLASS: MATCH BREAKDOWN",
        home_team: str = "HOME TEAM",
        away_team: str = "AWAY TEAM",
        score: str = "0 - 0",
        audio_base64: Optional[str] = None,
        bg_music_url: str = "http://127.0.0.1:3000/assets/audio/bg_music.mp3",
        webhook_url: Optional[str] = None,
        total_duration: float = 540.0
    ) -> Dict[str, Any]:
        """
        Builds complete payload for POST http://remotion-server:3000/render.
        """
        segments = cls.assemble_segments(events, total_duration=total_duration)
        return {
            "compositionId": "TacticalMasterclass",
            "audioBase64": audio_base64 or "",
            "musicUrl": bg_music_url,
            "homeTeam": home_team,
            "awayTeam": away_team,
            "score": score,
            "title": match_title,
            "segments": segments,
            "durationInSeconds": round(total_duration, 2),
            "webhookUrl": webhook_url or "",
        }
