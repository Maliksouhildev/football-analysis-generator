import os
import sys
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_pipeline.hybrid_engine import HybridTacticalEngine
from vision_pipeline.tactical_renderer import TacticalRenderer
from vision_pipeline.local_detector import LocalPlayerDetector
from vision_pipeline.montage_builder import MontageBuilder
from vision_pipeline.roster_manager import RosterManager

def run_real_footage_test():
    print("=== STARTING REAL BROADCAST FOOTAGE ANALYSIS ===")
    
    t0_path = r"C:\Users\Yacine\AppData\Local\Temp\real_match_footage\raw_frames\frame_0310.jpg"
    t1_path = r"C:\Users\Yacine\AppData\Local\Temp\real_match_footage\raw_frames\frame_0350.jpg"
    
    if not os.path.exists(t0_path) or not os.path.exists(t1_path):
        raise FileNotFoundError("Raw match frames not found in temp directory.")
        
    artifact_dir = r"C:\Users\Yacine\.gemini\antigravity\brain\81cf05f0-9be0-412d-a2c5-1dcb5e0b9048"
    os.makedirs(artifact_dir, exist_ok=True)
    
    detector = LocalPlayerDetector(confidence_threshold=0.15)
    montage_builder = MontageBuilder(cell_width=95, cell_height=125, cols=4)
    roster_mgr = RosterManager()
    
    engine = HybridTacticalEngine(
        detector=detector,
        montage_builder=montage_builder,
        roster_mgr=roster_mgr,
        cache_dir=r"C:\Users\Yacine\AppData\Local\Temp\real_match_footage\montages"
    )
    
    match_name = "1. FC Koln vs Hertha BSC"
    print(f"Executing reverse-timeline analysis for real match: {match_name}...")
    result = engine.analyze_pair(t0_path, t1_path, match_name=match_name)
    
    t1_analysis = result["t1_outcome_analysis"]
    t0_analysis = result["t0_setup_analysis"]
    
    print("\n--- T1 OUTCOME ANALYSIS ---")
    print(f"Game phase: {t1_analysis.get('pitch_context', {}).get('game_phase')}")
    print(f"Total players detected: {len(t1_analysis.get('players', []))}")
    print(f"Ball detected: {t1_analysis.get('ball')}")
    carrier_t1 = next((p for p in t1_analysis.get("players", []) if p.get("is_ball_carrier")), None)
    if carrier_t1:
        res = carrier_t1.get("resolved", {})
        print(f"Ball Carrier at T1: {res.get('display_name')} (#{res.get('number')}) - Foot: {carrier_t1.get('foot_pos_px')}")
    
    print("\n--- T0 SETUP ANALYSIS ---")
    print(f"Total players detected: {len(t0_analysis.get('players', []))}")
    pass_pred = t0_analysis.get("pass_prediction", {})
    print(f"Passer ID: {pass_pred.get('passer_id')} -> Receiver ID: {pass_pred.get('receiver_id')}")
    print(f"Pass type: {pass_pred.get('pass_type')}, Confidence: {pass_pred.get('confidence')}%")
    print(f"Key insight: {t0_analysis.get('key_tactical_insight')}")
    
    # Render broadcast graphics
    print("\nRendering broadcast-grade tactical overlays...")
    renderer = TacticalRenderer(output_dir=artifact_dir)
    
    out_t1_path = os.path.join(artifact_dir, "real_koln_bundesliga_t1.png")
    out_t0_path = os.path.join(artifact_dir, "real_koln_bundesliga_t0.png")
    
    renderer.render_frame_tactics(t1_path, t1_analysis, is_t0=False, output_filename=out_t1_path)
    renderer.render_frame_tactics(t0_path, t0_analysis, is_t0=True, output_filename=out_t0_path)
    
    print(f"Rendered T1: {out_t1_path}")
    print(f"Rendered T0: {out_t0_path}")
    
    # Copy montages to artifacts for easy inspection
    t0_montage = result.get("t0_montage_path")
    t1_montage = result.get("t1_montage_path")
    if t0_montage and os.path.exists(t0_montage):
        shutil.copy(t0_montage, os.path.join(artifact_dir, "montage_koln_bundesliga_t0.jpg"))
    if t1_montage and os.path.exists(t1_montage):
        shutil.copy(t1_montage, os.path.join(artifact_dir, "montage_koln_bundesliga_t1.jpg"))
        
    print("=== ANALYSIS COMPLETE & ARTIFACTS GENERATED ===")
    return result

if __name__ == "__main__":
    run_real_footage_test()
