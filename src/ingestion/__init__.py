"""
src/ingestion
Live Match Stream Ingestion, Scoreboard Clock OCR, PTS Indexing, and Clip Slicing.
"""

from .stream_buffer import (
    CircularBufferManager,
    SegmentMetadata,
    StreamBufferConfig,
    StreamBufferManager,
)
from .clock_ocr import (
    BROADCAST_PRESETS,
    ClockOCRParser,
    ClockReading,
    MonotonicClockFilter,
    PurePythonDigitOCR,
    ScoreboardBannerDetector,
    ScoreboardClockEngine,
    ScoreboardClockOCR,
    ScoreboardROI,
    binary_closing_3x3,
    compute_otsu_threshold,
)
from .time_indexer import (
    MatchTime,
    MatchTimeIndexer,
    TimeAnchor,
    TimeIndexer,
)
from .clip_extractor import (
    ClipExtractor,
    LaplacianSharpnessCalculator,
    calculate_sharpness_score,
    is_wide_tactical_cam,
)
from .replay_ingestor import (
    ReplayIngestor,
    DEFAULT_REPLAY_DIR,
    DEFAULT_INCOMING_DIR,
)

__all__ = [
    # Replay Ingestor
    "ReplayIngestor",
    "DEFAULT_REPLAY_DIR",
    "DEFAULT_INCOMING_DIR",
    # Stream Buffer
    "StreamBufferManager",
    "CircularBufferManager",
    "StreamBufferConfig",
    "SegmentMetadata",
    # Clock OCR
    "ScoreboardClockOCR",
    "ScoreboardClockEngine",
    "ScoreboardROI",
    "ClockReading",
    "ClockOCRParser",
    "MonotonicClockFilter",
    "ScoreboardBannerDetector",
    "PurePythonDigitOCR",
    "compute_otsu_threshold",
    "binary_closing_3x3",
    "BROADCAST_PRESETS",
    # Time Indexer
    "MatchTimeIndexer",
    "TimeIndexer",
    "MatchTime",
    "TimeAnchor",
    # Clip Extractor
    "ClipExtractor",
    "calculate_sharpness_score",
    "LaplacianSharpnessCalculator",
    "is_wide_tactical_cam",
]
