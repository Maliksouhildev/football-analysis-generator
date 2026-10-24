# Football Video Automation System — Test Infrastructure & Plan (TEST_INFRA.md)

**System**: Automated Football Match Tactical Analysis & Masterclass Video Pipeline  
**Target Root**: `C:\Users\mlkme\Desktop\cursor-mikow`  
**Test Suite Directory**: `tests/e2e/`  
**Author**: `test_writer_1`  
**Date**: 2026-10-15  
**Version**: 1.0.0  

---

## 1. Test Philosophy & Design Principles

1. **Opaque-Box & Requirement-Driven**:
   Tests validate behavior against authoritative specifications (`ORIGINAL_REQUEST.md`, `PROJECT.md`, `spec_r1_r2.md`, `spec_r3_r4_r5.md`) through published interfaces and contracts, without relying on internal private states.
2. **Determinism & Speed**:
   Live broadcasts, external web scrapers, Gemini cloud APIs, and Docker Remotion rendering cannot be queried during automated regression tests without introducing latency, network flakiness, rate limits, and quota consumption. The test infrastructure provides high-fidelity, self-contained mock fixtures (synthetic video stream manifests, scoreboard banner crops, match event JSONs, Gemini Vision responses, and Remotion composition payload schemas) allowing the entire test suite (~180+ tests) to execute in seconds with 100% determinism.
3. **Progressive Testability & Zero-Dependency Execution**:
   The test suite supports standalone execution via standard Python (`python tests/e2e/run_tests.py`) as well as `pytest` (`pytest tests/e2e`). It requires no proprietary test runners and provides contract mock harnesses that allow tests to run cleanly before, during, and after module implementations.
4. **Adversarial & Boundary Verification**:
   Special attention is given to edge cases: broadcast scoreboard occlusions during close-up replays, non-monotonic OCR digit noise, extra-time/stoppage-time clock formats (`45'+3'`), Cloudflare 403 fallbacks, zero-goal low-xG matches, degenerate polygons (<3 points), single defender backlines (<2 players), and Docker container host resolutions.

---

## 2. Feature Inventory Mapping

The 18 features defined in `PROJECT.md` are rigorously tested across 4 Tiers:

| Feature # | Feature Name | Primary Spec Source | Test Module | Tier 1 (Coverage) | Tier 2 (Boundaries) |
|---|---|---|---|---|---|
| **F1** | Stream Circular Buffer | R1 / `spec_r1_r2.md` §2.1 | `test_tier1_features.py`, `test_tier2_boundaries.py` | 5 tests | 5 tests |
| **F2** | Scoreboard ROI Detection | R1 / `spec_r1_r2.md` §2.2 | `test_tier1_features.py`, `test_tier2_boundaries.py` | 5 tests | 5 tests |
| **F3** | Clock Image Preprocessing | R1 / `spec_r1_r2.md` §2.3 | `test_tier1_features.py`, `test_tier2_boundaries.py` | 5 tests | 5 tests |
| **F4** | Clock OCR Engine | R1 / `spec_r1_r2.md` §2.3-2.4 | `test_tier1_features.py`, `test_tier2_boundaries.py` | 5 tests | 5 tests |
| **F5** | Monotonic Clock Filter | R1 / `spec_r1_r2.md` §2.4 | `test_tier1_features.py`, `test_tier2_boundaries.py` | 5 tests | 5 tests |
| **F6** | Match Time Indexer | R1 / `spec_r1_r2.md` §2.5 | `test_tier1_features.py`, `test_tier2_boundaries.py` | 5 tests | 5 tests |
| **F7** | Clip & Frame Cutter | R1 / `spec_r1_r2.md` §2.6 | `test_tier1_features.py`, `test_tier2_boundaries.py` | 5 tests | 5 tests |
| **F8** | Multi-Provider Scraper | R2 / `spec_r1_r2.md` §3.1 | `test_tier1_features.py`, `test_tier2_boundaries.py` | 5 tests | 5 tests |
| **F9** | Tactical Significance Filter | R2 / `spec_r1_r2.md` §3.2 | `test_tier1_features.py`, `test_tier2_boundaries.py` | 5 tests | 5 tests |
| **F10** | Masterclass Event Sequencer | R2 / `spec_r1_r2.md` §3.2.1 | `test_tier1_features.py`, `test_tier2_boundaries.py` | 5 tests | 5 tests |
| **F11** | Vision Prompt & Client | R3 / `spec_r3_r4_r5.md` §4 | `test_tier1_features.py`, `test_tier2_boundaries.py` | 5 tests | 5 tests |
| **F12** | 2D Foot Coordinate Anchoring | R3 / `spec_r3_r4_r5.md` §4 | `test_tier1_features.py`, `test_tier2_boundaries.py` | 5 tests | 5 tests |
| **F13** | Tactical Geometry Extractor | R3 / `spec_r3_r4_r5.md` §4 | `test_tier1_features.py`, `test_tier2_boundaries.py` | 5 tests | 5 tests |
| **F14** | Broadcast Telestration Styling | R4 / `spec_r3_r4_r5.md` §5 | `test_tier1_features.py`, `test_tier2_boundaries.py` | 5 tests | 5 tests |
| **F15** | Masterclass Video Composition | R4 / `spec_r3_r4_r5.md` §5 | `test_tier1_features.py`, `test_tier2_boundaries.py` | 5 tests | 5 tests |
| **F16** | Synchronized Voiceover | R4 / `spec_r3_r4_r5.md` §5.3 | `test_tier1_features.py`, `test_tier2_boundaries.py` | 5 tests | 5 tests |
| **F17** | End-to-End n8n Workflow | R5 / `spec_r3_r4_r5.md` §6 | `test_tier1_features.py`, `test_tier2_boundaries.py` | 5 tests | 5 tests |
| **F18** | E2E Opaque-Box Test Suite | PROJECT.md § Architecture | `test_tier1_features.py`, `test_tier2_boundaries.py` | 5 tests | 5 tests |

---

## 3. Test Architecture & Directory Layout

```
tests/e2e/
├── __init__.py
├── conftest.py                       # Pytest configuration and shared fixtures
├── run_tests.py                      # Standalone CLI test runner (zero external dependencies)
├── fixtures/                         # Authoritative mock fixtures
│   ├── __init__.py
│   ├── stream_fixtures.py            # HLS segments, synthetic video buffers, mock frames
│   ├── scoreboard_fixtures.py        # Scoreboard banner ROIs, clock digit images, OCR strings
│   ├── match_feed_fixtures.py        # Sofascore, API-Football, ESPN sample feeds
│   ├── vision_fixtures.py            # Gemini 2.5 Flash request/response mocks & coordinates
│   └── remotion_fixtures.py          # Masterclass props, segment arrays, audio timestamps
├── harnesses/                        # Contract validation and simulation harnesses
│   ├── __init__.py
│   ├── ingestion_harness.py          # Buffer, OCR, and cutter contract simulators
│   ├── scraper_harness.py            # Provider fallback and significance scoring simulators
│   ├── vision_harness.py             # Prompt builder, coordinate normalizer, geometry parser
│   ├── remotion_harness.py           # Payload validator, timeline calculator, schema linter
│   └── n8n_harness.py                # Single-trigger pipeline simulator and resume resolver
├── test_tier1_features.py            # Tier 1: Feature Coverage (90+ test cases)
├── test_tier2_boundaries.py          # Tier 2: Boundary & Corner Cases (90+ test cases)
├── test_tier3_combinations.py        # Tier 3: Cross-Feature Pairwise Combinations (12+ test cases)
└── test_tier4_scenarios.py           # Tier 4: Real-World Match Breakdowns (9+ end-to-end scenarios)
```

---

## 4. Test Tiers Breakdown

### Tier 1: Feature Coverage (Minimum 5 tests per feature = 90 tests)
Verifies nominal, happy-path functionality for all 18 features:
- F1: Circular buffer writing, segment naming, window retention, manifest generation, PTS continuity.
- F2: Preset selection for Premier League, La Liga, UCL, MLS, temporal variance localization.
- F3: Lanczos 4x scaling, HSV luminance masking, Otsu binarization, inverted output, white border padding.
- F4: Standard MM:SS recognition, colon detection, single-digit minute handling, 90+ min recognition, confidence scores.
- F5: Monotonic increment validation, replay skip detection, dead-reckoning advance, period transitions, half-time pause.
- F6: Forward PTS to match time, reverse match time to PTS, linear interpolation, stoppage offset, second-half recalibration.
- F7: 8-second clip cutting, 1080p frame extraction, discrete Laplacian variance computation, highest sharpness frame selection, fast-start metadata.
- F8: Sofascore incidents scraping, API-Football fixture events, ESPN zero-auth fallback, rate-limit fallback trigger, multi-provider normalization.
- F9: Goal scoring (100 pts), big chance scoring (90 pts), defensive turnover (85 pts), broken press (80 pts), routine pass filtering (<30 pts).
- F10: Chronological sorting, 10-14 segment count, temporal dispersion across match blocks, goal preservation, lower third synthesis.
- F11: Gemini 2.5 prompt construction, system instructions injection, base64 freeze frame encoding, temperature 0.1 enforcement, JSON response parsing.
- F12: Boot coordinate anchoring, 0-100% normalization, pixel scale mapping, pitch boundary clamping, role assignment.
- F13: Defensive backline chain sequence, pressing line dashed paths, passing triangle formation, overload polygon, label assignment.
- F14: Ellipse aspect ratio (2.33:1), pulsing reticle animation formula, vertical conical beam geometry, 45° diagonal hatched pattern, HUD badge glassmorphism.
- F15: Masterclass 8-10 min timeline calculation, clip and freeze-frame alternating sequences, lower-third timing, audio ducking, zero bundle error compliance.
- F16: Edge TTS timestamp parsing, speech narration alignment to segment cuts, loudnorm EBU R128 audio filtering, background music loop, outro fadeout.
- F17: 21-node orchestration flow, webhook resume URL formatting, async render POST, completion webhook resume, MP4 download and archive.
- F18: Test runner execution, test suite discovery, fixture isolation, deterministic output, multi-platform compatibility.

### Tier 2: Boundary & Corner Cases (Minimum 5 tests per feature = 90 tests)
Verifies adversarial edge cases, stress limits, and boundary conditions:
- Stoppage time overflow notations (`45'+7'`, `90'+12'`).
- Scoreboard occlusions during goal replays and VAR reviews (up to 30 seconds dead reckoning).
- Semi-transparent glassmorphism scoreboards with rapid grass movement.
- Network stream disconnects, reconnection PTS epoch reset, and corrupt packet discarding.
- Empty event feeds, matches with 0 goals and 0 big chances (0-0 tactical stalemate).
- Simultaneous overlapping events (minute 62 goal + 3 substitutions + yellow card).
- Extreme coordinates (player off-pitch at x=105%, y=-4%).
- Degenerate geometry (backline with 1 defender, triangle with 2 players, missing player IDs).
- Extended match duration (120 minutes + penalties = 18,000+ frames).
- Missing audio bed, TTS synthesis failure, Docker container localhost networking rewriting.

### Tier 3: Cross-Feature Combinations (Pairwise & Subsystem Interactions)
Verifies integration contracts between adjacent subsystems:
1. Ingestion (F1-F7) ↔ Scraper (F8-F10): PTS indexer correctly retrieves clips for high-scoring events.
2. Scraper (F9-F10) ↔ Vision (F11-F13): Pruned tactical events supply context for Gemini prompt generation.
3. Vision (F12-F13) ↔ Remotion Telestration (F14-F15): 2D boot coordinates and geometries render cleanly in `TacticalTelestrationOverlay`.
4. Ingestion Clips (F7) ↔ Remotion Composition (F15): Video clips and freeze frames alternate seamlessly in chronological order.
5. Voiceover (F16) ↔ Remotion Timeline (F15): Masterclass segment durations scale dynamically to match spoken audio timestamps.
6. Remotion Server (F15) ↔ n8n Automation (F17): Async render request, webhook callback, and MP4 retrieval flow end-to-end.

### Tier 4: Real-World Match Breakdown Scenarios (9 Complete Workflows)
1. **Scenario 1: UEFA Champions League High-Press Turnover Masterclass**
   - Minute 23' high press turnover leading to 0.42 xG goal. 8s clip, 1080p frame, 4-man pressing trap polygon, Edge TTS voiceover.
2. **Scenario 2: Premier League Stoppage-Time Comeback Thriller**
   - Minute 45'+3' and 90'+5' goals. Monotonic clock handles '+' syntax; clips sliced with stoppage time offsets.
3. **Scenario 3: Low-Block Counter-Attack Masterclass**
   - Compact 5-man defensive backline, 85-point defensive third recovery, fast counter pass into final third.
4. **Scenario 4: 0-0 Tactical Stalemate (Zero Goal Dynamic Re-balancing)**
   - No goals or big chances; filter dynamically lowers threshold to select top 12 pressing duels and tactical subs.
5. **Scenario 5: VAR Overturned Goal with Sub-Minute Flurry**
   - Minute 67' goal disallowed by VAR, followed by 3 substitutions; sequencer tracks VAR incident and maintains chronological stability.
6. **Scenario 6: Broadcast Scoreboard Occlusion During 25-Second Celebration**
   - Clock graphic removed during celebrations; dead-reckoning engine tracks match minutes flawlessly.
7. **Scenario 7: Full 90-Minute Chronological Breakdown (12 Key Events)**
   - 12 segments spanning 1' to 89', total video duration 520 seconds (8m 40s), 100% audio and video alignment.
8. **Scenario 8: Extra Time & Penalty Shootout Tactical Transition**
   - Minute 105' and 118' extra-time segments, state machine period advancement, shootout summary.
9. **Scenario 9: Multi-Provider Failover under Cloudflare 403 Block**
   - Sofascore returns 403; scraper seamlessly falls back to API-Football and ESPN zero-auth feed with zero lost events.

---

## 5. Coverage & Quality Thresholds

| Metric | Target Requirement | Suite Commitment |
|---|---|---|
| Total Test Cases | >= 180 tests | >= 195 tests |
| Tier 1 Feature Coverage | >= 90 tests (5 per feature x 18) | 90 tests |
| Tier 2 Boundary Cases | >= 90 tests (5 per feature x 18) | 90 tests |
| Tier 3 Combinations | Pairwise across 5 subsystems | 12 tests |
| Tier 4 Scenarios | >= 9 tactical workflows | 9 tests |
| Execution Speed | < 10 seconds for full suite | ~2-4 seconds |
| Flakiness / External Network Calls | 0% (100% deterministic via fixtures) | 0% |
| Supported Runners | Python unittest & Pytest | Both supported |

---

## 6. Execution Instructions

### Standalone Runner (No external packages required):
```bash
python tests/e2e/run_tests.py
```

### Pytest Runner:
```bash
pytest tests/e2e/ -v
```

### Tier-Specific Execution:
```bash
python tests/e2e/run_tests.py --tier 1
python tests/e2e/run_tests.py --tier 2
python tests/e2e/run_tests.py --tier 3
python tests/e2e/run_tests.py --tier 4
```
