"""
Tracking and Video Evaluation Subsystem.
Implements Shot Boundary Detection, BoT-SORT with Camera Motion Compensation,
Team/GK Color Clustering (k=2 + Outlier Isolation), and Smart Collage Batching.
"""

from .shot_detector import ShotBoundaryDetector, ShotSegment
from .bot_sort_tracker import BoTSORTTracker, TrackedObject, TrackletLifecycle
from .team_color_clusterer import TeamColorClusterer, TeamClusterResult
from .smart_collage_batcher import SmartCollageBatcher, CollageBatchPayload

__all__ = [
    "ShotBoundaryDetector",
    "ShotSegment",
    "BoTSORTTracker",
    "TrackedObject",
    "TrackletLifecycle",
    "TeamColorClusterer",
    "TeamClusterResult",
    "SmartCollageBatcher",
    "CollageBatchPayload",
]
