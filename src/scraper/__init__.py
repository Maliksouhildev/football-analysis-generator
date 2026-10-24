"""
src/scraper
Play-by-Play Football Match Event Scraper, Tactical Significance Filter,
Masterclass Event Sequencer, Free Crowd Consensus Scraper, and Tactical Evidence Matcher.
"""

from .event_scraper import (
    BaseProviderClient,
    CanonicalEventType,
    CanonicalMatchEvent,
    CircuitBreaker,
    CircuitState,
    EspnPublicClient,
    MultiProviderEventScraper,
    MultiProviderScraper,
    MultiProviderScraperSimulator,
    PlayerInfo,
    SofascoreClient,
    SyntheticFixtureClient,
    TeamInfo,
    TokenBucketRateLimiter,
    ApiFootballClient,
)
from .tactical_filter import (
    PitchChannel,
    PitchCoordinateEvaluator,
    PitchThird,
    TacticalCategory,
    TacticalFilter,
    TacticalFilterConfig,
    TacticalFilterReport,
    TacticalSignificanceFilter,
    TacticalSignificanceScorer,
)
from .masterclass_sequencer import (
    CommentaryDirective,
    MasterclassEventSequencer,
    RemotionMasterclassAdapter,
    TacticalEvent,
    TacticalMetadataSynthesizer,
    TacticalSegment,
)
from .lineup_scraper import MatchLineupResolver
from .crowd_consensus import CrowdConsensusScraper, CrowdConsensusResult
from .evidence_matcher import EvidenceMatcher, GeminiFailoverPool, TacticalThesis

__all__ = [
    # Event Scraper
    "CanonicalEventType",
    "PlayerInfo",
    "TeamInfo",
    "CanonicalMatchEvent",
    "CircuitState",
    "CircuitBreaker",
    "TokenBucketRateLimiter",
    "BaseProviderClient",
    "SofascoreClient",
    "ApiFootballClient",
    "EspnPublicClient",
    "SyntheticFixtureClient",
    "MultiProviderScraper",
    "MultiProviderEventScraper",
    "MultiProviderScraperSimulator",
    # Tactical Filter
    "PitchThird",
    "PitchChannel",
    "TacticalCategory",
    "TacticalFilterConfig",
    "PitchCoordinateEvaluator",
    "TacticalSignificanceScorer",
    "TacticalSignificanceFilter",
    "TacticalFilter",
    "TacticalFilterReport",
    # Masterclass Sequencer
    "CommentaryDirective",
    "TacticalEvent",
    "TacticalSegment",
    "TacticalMetadataSynthesizer",
    "MasterclassEventSequencer",
    "RemotionMasterclassAdapter",
    # Lineup Scraper & Crowd Consensus & Evidence Matcher
    "MatchLineupResolver",
    "CrowdConsensusScraper",
    "CrowdConsensusResult",
    "EvidenceMatcher",
    "GeminiFailoverPool",
    "TacticalThesis",
]
