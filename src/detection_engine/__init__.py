"""
src/detection_engine/__init__.py
================================================================================
Public API for the modular Player Detection, Tracking, Roster Sync & Identity Subsystem.
================================================================================
"""

from src.detection_engine.pitch_geometry import PitchGeometryEngine
from src.detection_engine.player_id_pipeline import PlayerIDPipeline
from src.detection_engine.types import (
    AffinityScoreBreakdown,
    CandidateBox,
    DetectedPitchActor,
    DetectionResult,
    PipelineConfig,
)
from src.vision.match_roster_engine import CANONICAL_CLASICO_ROSTER, ActiveRosterState, ActiveXIResolver

__all__ = [
    "PlayerIDPipeline",
    "PipelineConfig",
    "DetectionResult",
    "DetectedPitchActor",
    "CandidateBox",
    "AffinityScoreBreakdown",
    "PitchGeometryEngine",
    "ActiveXIResolver",
    "ActiveRosterState",
    "CANONICAL_CLASICO_ROSTER",
]
