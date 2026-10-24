"""
src/vision/__init__.py
Milestone 3: Vision-Powered Tactical Telestrator.

Public API exporting Gemini 2.5 Flash Vision client, coordinate normalizer,
tactical formation parser, Remotion telestration adapter, and deterministic
synthetic tactical fixture generator.
"""

from src.vision.tactical_parser import (
    BACKLINE_ROLES,
    CLUB_NEON_PALETTES,
    NeonColorManager,
    PassingTriangleBuilder,
    PressingLineBuilder,
    ROLE_NORMALIZATION_MAP,
    TacticalBacklineBuilder,
    TacticalFormationParser,
    TacticalGeometryExtractor,
    TacticalRole,
    TacticalTelestrationProps,
    TelestrationLine,
    TelestrationPlayer,
    TelestrationPolygon,
)
from src.vision.gemini_client import (
    CANONICAL_SYSTEM_INSTRUCTION,
    GeminiVisionClient,
    OfflineTacticalExtractor,
    VisionPromptBuilder,
)
from src.vision.player_detector import (
    CVPlayerDetector,
    DetectedPitchPlayer,
    get_pitch_convex_hull,
)
from src.vision.camera_pan_compensator import (
    CameraPanCompensator,
    TemporalBurstAligner,
)
from src.vision.pitch_homography_role_mapper import (
    LiveSpatialRoleMapper,
    PitchHomographyProjector,
    PlayerVisualFeatures,
    SignalAVisualExtractor,
    SignalBFormationTopology,
    SignalCSetPieceLookup,
)
from src.vision.temporal_ground_truth import TemporalGroundTruthEngine
from src.vision.match_roster_engine import (
    ActiveRosterState,
    ActiveXIResolver,
    MatchClockExtractor,
    MatchClockResult,
    MatchRosterStateEngine,
    PlayerProfile,
)
from src.vision.tactical_geometry_optimizer import (
    LineSegment,
    Point2D,
    ShortestLinkGeometryEngine,
    TacticalLineStyle,
    TacticalLineType,
)
from src.vision.tactical_vision_engine import TacticalVisionEngine
from src.vision.telestration_adapter import (
    CANONICAL_TACTICAL_SCENARIOS,
    CoordinateNormalizer,
    RemotionTelestrationAdapter,
    SyntheticTacticalGenerator,
    extract_tactical_telestration,
)

__all__ = [
    # Tactical entities & schema
    "TacticalRole",
    "TelestrationPlayer",
    "TelestrationLine",
    "TelestrationPolygon",
    "TacticalTelestrationProps",
    # Color & taxonomy
    "ROLE_NORMALIZATION_MAP",
    "BACKLINE_ROLES",
    "CLUB_NEON_PALETTES",
    "NeonColorManager",
    # Builders & parsers
    "TacticalBacklineBuilder",
    "PressingLineBuilder",
    "PassingTriangleBuilder",
    "TacticalFormationParser",
    "TacticalGeometryExtractor",
    # Coordinate normalization
    "CoordinateNormalizer",
    # Camera Pan Compensation & Homography
    "CameraPanCompensator",
    "TemporalBurstAligner",
    # 2D Pitch Homography & Live Spatial Role Mapping
    "PitchHomographyProjector",
    "LiveSpatialRoleMapper",
    "PlayerVisualFeatures",
    "SignalAVisualExtractor",
    "SignalBFormationTopology",
    "SignalCSetPieceLookup",
    # Timestamp-Synced Match Roster & Active XI Resolver
    "MatchClockExtractor",
    "MatchClockResult",
    "PlayerProfile",
    "ActiveRosterState",
    "ActiveXIResolver",
    "MatchRosterStateEngine",
    # Gemini client & prompt
    "CANONICAL_SYSTEM_INSTRUCTION",
    "VisionPromptBuilder",
    "OfflineTacticalExtractor",
    "GeminiVisionClient",
    # Synthetic generator & adapter
    "CANONICAL_TACTICAL_SCENARIOS",
    "SyntheticTacticalGenerator",
    "RemotionTelestrationAdapter",
    "extract_tactical_telestration",
    "TacticalVisionEngine",
    "TemporalGroundTruthEngine",
    "get_pitch_convex_hull",
    "CVPlayerDetector",
    "DetectedPitchPlayer",
    # Shortest-Link Geometry Optimizer
    "ShortestLinkGeometryEngine",
    "TacticalLineStyle",
    "TacticalLineType",
    "LineSegment",
    "Point2D",
]

