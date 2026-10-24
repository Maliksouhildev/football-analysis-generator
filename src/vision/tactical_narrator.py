"""
src/vision/tactical_narrator.py
AI Tactical Picture Analysis & Broadcast Commentary Generator.
Re-exports TacticalNarrator from src.rendering.tactical_narrator with full 3-beat script generation.
"""

from src.rendering.tactical_narrator import (
    TacticalNarrator,
    CANONICAL_PROMPT_TEMPLATE,
    PROMPT_3BEAT_TEMPLATE,
)

__all__ = ["TacticalNarrator", "CANONICAL_PROMPT_TEMPLATE", "PROMPT_3BEAT_TEMPLATE"]
