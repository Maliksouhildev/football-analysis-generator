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

    def test_pass_option_deduplicated_for_pass_receiver(self):
        """Verifies that candidate pass options do NOT include the teammate who receives the executed pass."""
        cause_tel = {}
        tracks = [
            {"track_id": 10, "x_pct": 72.0, "y_pct": 39.5, "team": "away"}, # Carrier
            {"track_id": 1, "x_pct": 92.0, "y_pct": 50.0, "team": "away"},  # Receiver
            {"track_id": 2, "x_pct": 32.3, "y_pct": 47.7, "team": "away"},  # Alternative option
        ]
        ctx = {"clip_id": "clip_01_left_wing"}
        TacticalVisionEngine.sanitize_v3_telestration(
            cause_tel=cause_tel,
            tracks_at_freeze=tracks,
            clip_id="clip_01_left_wing",
            event_context=ctx,
        )
        pass_options = [v for v in cause_tel.get("vectors", []) if v.get("type") == "PASS_OPTION_DASHED"]
        target_ids = {int(v.get("to_track_id", -1)) for v in pass_options}
        assert 1 not in target_ids, "Receiver of executed pass (track 1) must not have a duplicate candidate pass option"

    def test_head_on_opposing_pressing_run_suppressed(self):
        """Verifies that an opponent standing at the pass destination does not generate a colliding head-on opposing pressing run."""
        cause_tel = {
            "pass_vector": {
                "from_x": 53.4,
                "from_y": 39.3,
                "to_x": 43.4,
                "to_y": 39.6,
            }
        }
        tracks = [
            {"track_id": 3, "x_pct": 53.4, "y_pct": 39.3, "team": "away"}, # Carrier
            {"track_id": 9, "x_pct": 43.4, "y_pct": 39.6, "team": "home"}, # Pass destination opponent
            {"track_id": 12, "x_pct": 53.9, "y_pct": 33.3, "team": "home"}, # Flank presser
        ]
        ctx = {"clip_id": "clip_02_zone_14"}
        TacticalVisionEngine.sanitize_v3_telestration(
            cause_tel=cause_tel,
            tracks_at_freeze=tracks,
            clip_id="clip_02_zone_14",
            event_context=ctx,
        )
        pressing_runs = [v for v in cause_tel.get("vectors", []) if v.get("type") == "PRESSING_RUN"]
        presser_ids = {int(v.get("from_track_id", -1)) for v in pressing_runs}
        assert 9 not in presser_ids, "Opponent at pass target (track 9) must not generate a colliding opposing pressing run"

    def test_pressing_vector_anchored_at_player_position(self):
        """Verifies that pressing run start coordinates strictly match the player's true track coordinates without artificial detachment."""
        cause_tel = {}
        tracks = [
            {"track_id": 3, "x_pct": 53.4, "y_pct": 39.3, "team": "away"}, # Carrier
            {"track_id": 12, "x_pct": 53.9, "y_pct": 33.3, "team": "home"}, # Presser
        ]
        ctx = {"clip_id": "clip_02_zone_14"}
        TacticalVisionEngine.sanitize_v3_telestration(
            cause_tel=cause_tel,
            tracks_at_freeze=tracks,
            clip_id="clip_02_zone_14",
            event_context=ctx,
        )
        for v in cause_tel.get("vectors", []):
            if v.get("type") == "PRESSING_RUN" and int(v.get("from_track_id", -1)) == 12:
                assert abs(v["from_pct"][0] - 53.9) < 0.1, f"Expected from_pct X 53.9, got {v['from_pct'][0]}"
                assert abs(v["from_pct"][1] - 33.3) < 0.1, f"Expected from_pct Y 33.3, got {v['from_pct'][1]}"
