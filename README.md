# Football Analysis Generator — Tactical Masterclass & Vision AI

Automated system for tactical football (soccer) match analysis, reverse-timeline player tracking, pass prediction, and broadcast-grade tactical overlays. Combines an n8n workflow orchestrator with an autonomous multimodal Vision AI pipeline and interactive tactical control room.

---

## Architecture Overview

```
                      ┌──────────────────────────────┐
                      │      n8n Orchestrator        │
                      │       (workflow.ts)          │
                      └──────────────┬───────────────┘
                                     │ HTTP (port 5055)
                                     ▼
                      ┌──────────────────────────────┐
                      │      Tactical API Server     │
                      │  (vision_pipeline/api_server)│
                      └──────────────┬───────────────┘
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
┌──────────────────────────────┐               ┌──────────────────────────────┐
│       Vision AI Engine       │               │        Roster Manager        │
│ (vision_pipeline/vision_eng) │               │(vision_pipeline/roster_mgr)  │
│ - Reverse Timeline (T1 → T0) │◄─────────────►│ - Non-Hardcoded Resolution   │
│ - Multimodal Gemini Vision   │               │ - Kit, Number, Traits, Roles │
│ - Ball & Pass Prediction     │               │ - SkillCorner / Free Datasets│
└──────────────┬───────────────┘               └──────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐               ┌──────────────────────────────┐
│      Tactical Renderer       │               │   Evaluation & Tuning Loop   │
│ (vision_pipeline/tactical_ren│               │  (vision_pipeline/eval_loop) │
│ - Floor Rings & Spotlights   │               │ - Prompt & Model Sweeps      │
│ - Floating Glassmorphic Badges│              │ - Precision & Spatial Metric │
│ - Aerodynamic Pass Vectors   │               │ - Leaderboard Benchmarking   │
└──────────────────────────────┘               └──────────────────────────────┘
```

---

## Key Modules

| Module | Location | Description |
|:-------|:---------|:------------|
| **Multi-Frame Tracker** | [`vision_pipeline/multi_frame_tracker.py`](vision_pipeline/multi_frame_tracker.py) | Continuous multi-object tracking (MOT) across sequential match frames ($T_0 \dots T_4$) with persistent track IDs, velocity estimation (km/h), motion breadcrumbs, and ball flight trajectory splines. |
| **Local Detector** | [`vision_pipeline/local_detector.py`](vision_pipeline/local_detector.py) | Sub-pixel turf-grounded player & ball detection running locally on CPU with zero coordinate drift, turf mask filtering, and kit color extraction. |
| **Hybrid Tactical Engine** | [`vision_pipeline/hybrid_engine.py`](vision_pipeline/hybrid_engine.py) | Couples local detection with single-call sprite montage VLM reasoning, eliminating API rate limits and hallucinated grass coordinates. |
| **Roster Manager** | [`vision_pipeline/roster_manager.py`](vision_pipeline/roster_manager.py) | Dynamic, non-hardcoded player identification using multi-cue matching (jersey numbers, kit colors, tactical positions, physical traits). |
| **Tactical Renderer**| [`vision_pipeline/tactical_renderer.py`](vision_pipeline/tactical_renderer.py) | Renders TV broadcast tactical indicators: perspective floor rings, volumetric spotlights, motion breadcrumb trails, continuous ball arcs, and glassmorphic HUD badges. |
| **API Server** | [`vision_pipeline/api_server.py`](vision_pipeline/api_server.py) | Lightweight HTTP server listening on port `5055` supporting `/analyze`, `/analyze-hybrid`, and `/analyze-sequence` for n8n workflow and web dashboard. |
| **Control Dashboard** | [`dashboard.html`](dashboard.html) | Interactive web dashboard featuring 5-stage timeline scrubbing ($T_0 \dots T_4$), live auto-play sequence playback, and continuous video tracking mode. |

---

## Getting Started

### 1. Prerequisites
- Python 3.12+ (with `requests`, `pillow`, `numpy`, `google-genai`)
- FFmpeg (for video frame extraction)
- `GEMINI_API_KEY` environment variable set

### 2. Run the Evaluation & Tuning Loop
To run the automated prompt and model optimization suite across test fixtures:
```bash
python vision_pipeline/eval_loop.py
```
Outputs leaderboard to `output/eval_leaderboard.json` and `output/eval_summary.md`.

### 3. Run Unit Tests
```bash
python -m unittest tests/test_pipeline.py
```

### 4. Start the Vision API Server (Port 5055)
```bash
python vision_pipeline/api_server.py
```
Test endpoints:
- `GET http://localhost:5055/health`
- `POST http://localhost:5055/search_telegram`
- `POST http://localhost:5055/analyze`

### 5. Launch the Tactical Control Room
Open [`dashboard.html`](dashboard.html) in any modern browser to inspect match sequence pairs, scrubber between buildup and outcome frames, and toggle tactical indicators.
