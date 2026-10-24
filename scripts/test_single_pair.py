"""
Test runner for a single match frame pair using VisionEngine and TacticalRenderer.
"""

import json
import os
import sys

# Support UTF-8 in Windows console
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vision_pipeline.vision_engine import VisionEngine
from vision_pipeline.roster_manager import RosterManager
from vision_pipeline.tactical_renderer import TacticalRenderer

def main():
    test_dir = os.path.join(REPO_ROOT, "test_data")
    t0_path = os.path.join(test_dir, "galatasaray_barcelona_t0.jpg")
    t1_path = os.path.join(test_dir, "galatasaray_barcelona_t1.jpg")

    print(f"Testing Reverse-Timeline Vision Pipeline on:")
    print(f"  T0: {t0_path}")
    print(f"  T1: {t1_path}")

    engine = VisionEngine(model_name="gemini-3.5-flash", temperature=0.1)
    result = engine.analyze_pair(t0_path, t1_path, match_name="Galatasaray vs Barcelona")

    print("\n=== PIPELINE EXECUTION SUCCESSFUL ===")
    print("Timing:", result["timing"])
    print("Metrics:", json.dumps(result["metrics"], indent=2))

    t1_analysis = result["t1_outcome_analysis"]
    t0_analysis = result["t0_setup_analysis"]

    print("\n--- T1 Detected Players ---")
    for p in t1_analysis.get("players", []):
        res = p.get("resolved", {})
        print(f"ID {p.get('id')}: #{res.get('number')} {res.get('name')} ({res.get('position')}) - Confidence: {res.get('confidence')}% - Carrier: {p.get('is_ball_carrier')}")

    print("\n--- T0 Pass Prediction ---")
    pred = t0_analysis.get("pass_prediction", {})
    print(f"Passer: {pred.get('passer_name_predicted')}")
    print(f"Target Receiver: {pred.get('predicted_receiver_name')}")
    print(f"Pass Type: {pred.get('pass_type')}")
    print(f"Predicted Destination Norm: {pred.get('pass_destination_norm')}")
    print(f"Actual T1 Ball Norm: {t1_analysis.get('ball', {}).get('position_norm')}")
    print(f"Spatial Pass Accuracy: {result['metrics']['spatial_pass_accuracy_percent']}%")

    # Render tactical graphics
    renderer = TacticalRenderer()
    out_t1 = renderer.render_frame_tactics(t1_path, t1_analysis, is_t0=False, output_filename="rendered_galatasaray_barcelona_t1.png")
    out_t0 = renderer.render_frame_tactics(t0_path, t0_analysis, is_t0=True, output_filename="rendered_galatasaray_barcelona_t0.png")

    print(f"\nRendered Tactical Overlays:")
    print(f"  T1 Annotated Image: {out_t1}")
    print(f"  T0 Annotated Image: {out_t0}")

    # Also save the complete JSON results for the dashboard
    json_out = os.path.join(renderer.output_dir, "analysis_results.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Analysis JSON: {json_out}")

if __name__ == "__main__":
    main()
