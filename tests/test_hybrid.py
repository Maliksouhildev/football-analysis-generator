"""
Unit tests for the Hybrid Local CV + Vision LLM Pipeline:
- LocalPlayerDetector sub-pixel localization & kit analysis
- MontageBuilder sprite sheet composition
- HybridTacticalEngine reverse-timeline pass prediction
"""

import os
import sys
import unittest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vision_pipeline.local_detector import LocalPlayerDetector
from vision_pipeline.montage_builder import MontageBuilder
from vision_pipeline.hybrid_engine import HybridTacticalEngine
from vision_pipeline.roster_manager import RosterManager

class TestHybridPipeline(unittest.TestCase):
    """Test suite for local detector, montage builder, and hybrid engine."""

    def setUp(self):
        self.test_data_dir = os.path.join(REPO_ROOT, "test_data")
        self.output_dir = os.path.join(REPO_ROOT, "output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.t0_img = os.path.join(self.test_data_dir, "galatasaray_barcelona_t0.jpg")
        self.t1_img = os.path.join(self.test_data_dir, "galatasaray_barcelona_t1.jpg")
        self.rma_t0 = os.path.join(self.test_data_dir, "realmadrid_mancity_t0.jpg")
        self.rma_t1 = os.path.join(self.test_data_dir, "realmadrid_mancity_t1.jpg")

        self.detector = LocalPlayerDetector()
        self.montage_builder = MontageBuilder(cell_width=80, cell_height=110, cols=4)

    def test_local_detector_detection_count(self):
        """Verifies local CV detector isolates all players on pitch."""
        det = self.detector.detect(self.t0_img)
        players = det["players"]
        self.assertGreaterEqual(len(players), 8, f"Expected at least 8 players, got {len(players)}")

        for p in players:
            self.assertIn("bbox_px", p)
            self.assertIn("foot_pos_px", p)
            self.assertIn("foot_pos_norm", p)
            self.assertIn("kit_color", p)
            # Foot contact point y should be equal to bottom of bounding box
            self.assertAlmostEqual(p["foot_pos_px"][1], p["bbox_px"][2], delta=1.5)

    def test_local_detector_kit_classification(self):
        """Verifies kit color extractor separates opposing teams."""
        det = self.detector.detect(self.rma_t0)
        kits = [p["kit_color"] for p in det["players"]]
        # Should detect white or blue kit themes
        has_white_or_blue = any("White" in k or "Blue" in k for k in kits)
        self.assertTrue(has_white_or_blue, f"Kits detected: {kits}")

    def test_montage_builder_generation(self):
        """Verifies MontageBuilder generates a valid sprite sheet image."""
        det = self.detector.detect(self.t0_img)
        montage_path = os.path.join(self.output_dir, "test_montage.jpg")
        montage, path = self.montage_builder.create_montage(self.t0_img, det["players"], montage_path)

        self.assertTrue(os.path.exists(montage_path))
        self.assertGreater(montage.width, 200)
        self.assertGreater(montage.height, 100)

    def test_hybrid_engine_reverse_timeline(self):
        """Verifies HybridTacticalEngine executes paired T1 -> T0 reverse timeline."""
        engine = HybridTacticalEngine(
            api_key="MOCK_TEST_KEY",
            detector=self.detector,
            montage_builder=self.montage_builder
        )

        res = engine.analyze_pair(self.t0_img, self.t1_img, match_name="Galatasaray vs Barcelona")
        self.assertEqual(res["pipeline"], "hybrid_onnx_vlm")
        self.assertIn("t1_outcome_analysis", res)
        self.assertIn("t0_setup_analysis", res)

        t0_data = res["t0_setup_analysis"]
        self.assertIn("pass_prediction", t0_data)
        pred = t0_data["pass_prediction"]

        # Pass origin must match carrier's foot position exactly (0% coordinate drift)
        carrier_foot = None
        for p in t0_data["players"]:
            if p.get("is_ball_carrier"):
                carrier_foot = p["foot_pos_norm"]
                break

        self.assertIsNotNone(carrier_foot)
        self.assertEqual(pred["pass_origin_norm"], carrier_foot)
        self.assertEqual(res["metrics"]["coordinate_drift_px"], 0.0)

    def test_multi_frame_tracker_sequence(self):
        """Verifies MultiFrameTracker maintains persistent track IDs and motion trails."""
        from vision_pipeline.multi_frame_tracker import MultiFrameTracker
        tracker = MultiFrameTracker()
        frame_items = [(0, self.t0_img), (1, self.t1_img)]
        res = tracker.track_sequence(frame_items, match_name="Galatasaray vs Barcelona")

        self.assertEqual(len(res["stages"]), 2)
        stage0 = res["stages"][0]
        stage1 = res["stages"][1]
        self.assertIn("players", stage0)
        self.assertIn("players", stage1)
        self.assertGreater(len(stage0["players"]), 5)

        # Check persistent track IDs exist
        t0_ids = {p["track_id"] for p in stage0["players"]}
        t1_ids = {p["track_id"] for p in stage1["players"]}
        overlap = t0_ids.intersection(t1_ids)
        self.assertGreaterEqual(len(overlap), 3, f"Expected overlapping tracks, got {overlap}")

    def test_tactical_renderer_sequence_stage(self):
        """Verifies TacticalRenderer generates multi-frame sequence stage with breadcrumbs."""
        from vision_pipeline.tactical_renderer import TacticalRenderer
        from vision_pipeline.multi_frame_tracker import MultiFrameTracker

        tracker = MultiFrameTracker()
        renderer = TacticalRenderer(output_dir=self.output_dir)

        frame_items = [(0, self.t0_img), (1, self.t1_img)]
        seq_data = tracker.track_sequence(frame_items, match_name="Galatasaray vs Barcelona")

        out_img = renderer.render_sequence_stage(
            image_path=self.t0_img,
            stage_data=seq_data["stages"][0],
            output_filename="test_sequence_render_t0.png"
        )
        self.assertTrue(os.path.exists(out_img))
        img = Image.open(out_img)
        self.assertEqual(img.size, (1280, 720))

if __name__ == "__main__":
    unittest.main()
