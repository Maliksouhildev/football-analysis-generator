import math
import pytest
from src.vision.tactical_vision_engine import TacticalVisionEngine

class TestTelestrationVectors:
    def test_ground_pass_color_is_white(self):
        """Verifies that ground pass vectors default to broadcast white (#ffffff) instead of cyan (#00e5ff)."""
        cause_tel = {}
        tracks = [
            {"track_id": 1, "x_pct": 50.0, "y_pct": 50.0, "team": "home"},
            {"track_id": 2, "x_pct": 40.0, "y_pct": 50.0, "team": "home"},
        ]
        ctx = {"clip_id": "clip_01_test"}
        # Check sanitize_v3_telestration logic on ground pass
        TacticalVisionEngine.sanitize_v3_telestration(
            cause_tel=cause_tel,
            tracks_at_freeze=tracks,
            clip_id="clip_01_test",
            event_context=ctx,
        )
        for v in cause_tel.get("vectors", []):
            if v.get("type") == "GROUND_PASS_SOLID":
                assert v.get("color") != "#00e5ff", "Ground pass must not be cyan #00e5ff"
                assert v.get("color") == "#ffffff", f"Ground pass should be #ffffff, got {v.get('color')}"

    def test_pressing_vectors_deduplication(self):
        """Verifies that defending pressers are not duplicated between pressing_arrows and pressing_run_vectors."""
        cause_tel = {
            "pressing_arrows": [
                {"from_track_id": 9, "from_pct": [43.4, 39.6], "to_pct": [48.9, 39.4], "color": "#8a1529", "type": "PRESSING_ARROW"}
            ]
        }
        tracks = [
            {"track_id": 3, "x_pct": 53.4, "y_pct": 39.3, "team": "away"},
            {"track_id": 9, "x_pct": 43.4, "y_pct": 39.6, "team": "home"},
            {"track_id": 12, "x_pct": 53.9, "y_pct": 33.3, "team": "home"},
        ]
        ctx = {"clip_id": "clip_02_zone_14"}
        TacticalVisionEngine.sanitize_v3_telestration(
            cause_tel=cause_tel,
            tracks_at_freeze=tracks,
            clip_id="clip_02_zone_14",
            event_context=ctx,
        )
        # Track 9 already has a PRESSING_ARROW, so vectors should not have a duplicate PRESSING_RUN for track 9
        pressing_runs = [v for v in cause_tel.get("vectors", []) if v.get("type") == "PRESSING_RUN" and int(v.get("from_track_id", -1)) == 9]
        assert len(pressing_runs) == 0, "Track 9 should not have a duplicate PRESSING_RUN when PRESSING_ARROW exists"

    def test_collinear_offset_preserves_separation(self):
        """Verifies collinear pressing vectors are offset laterally away from the pass vector."""
        cause_tel = {}
        tracks = [
            {"track_id": 3, "x_pct": 53.4, "y_pct": 39.3, "team": "away"}, # Carrier
            {"track_id": 9, "x_pct": 43.4, "y_pct": 39.6, "team": "home"}, # Collinear opponent
            {"track_id": 12, "x_pct": 53.9, "y_pct": 33.3, "team": "home"},
        ]
        ctx = {"clip_id": "clip_02_zone_14"}
        TacticalVisionEngine.sanitize_v3_telestration(
            cause_tel=cause_tel,
            tracks_at_freeze=tracks,
            clip_id="clip_02_zone_14",
            event_context=ctx,
        )
        pass_v = cause_tel.get("pass_vector", {})
        if pass_v and pass_v.get("from_x") is not None:
            p_fy = float(pass_v["from_y"])
            # If any pressing run exists near the pass line, ensure it is not directly at the same y
            for v in cause_tel.get("vectors", []):
                if v.get("type") == "PRESSING_RUN":
                    from_y = float(v["from_pct"][1])
                    to_y = float(v["to_pct"][1])
                    assert not (abs(from_y - p_fy) < 0.2 and abs(to_y - p_fy) < 0.2), "Pressing vector must not be exactly stacked on pass line"
