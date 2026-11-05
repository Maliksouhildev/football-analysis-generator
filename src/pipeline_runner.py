"""
src/pipeline_runner.py
Master Tactical Analysis & Telestration Pipeline Runner.

Orchestrates:
1. Replay Ingestion (URL download via yt-dlp or watched folder daemon)
2. Live stream / video ingestion & broadcast clock OCR (M1)
3. Real-time play-by-play event scraping & significance filtering (M2)
4. Gemini 2.5/3.8 Flash Vision tactical extraction (M3)
5. Remotion Masterclass 8-10 min 1080p video rendering (M4)
"""

import argparse
import base64
from contextlib import contextmanager
import json
import logging
import math
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import (
    ClipExtractor,
    MatchTimeIndexer,
    ReplayIngestor,
    ScoreboardClockOCR,
)
from src.scraper import (
    MultiProviderScraper,
    RemotionMasterclassAdapter,
    TacticalSignificanceFilter,
    CrowdConsensusScraper,
    EvidenceMatcher,
    MatchLineupResolver,
)
from src.scraper.autonomous_match_resolver import AutonomousMatchResolver
from src.ingestion.sparse_clock_indexer import SparseClockIndexer
from src.vision import (
    GeminiVisionClient,
    extract_tactical_telestration,
)
from src.vision.tactical_vision_engine import TacticalVisionEngine
from src.rendering.tactical_narrator import TacticalNarrator
from src.rendering.player_asset_manager import PlayerAssetManager
from src.rendering.audio_engine import AudioEngine
from src.ingestion.parse_tracking_data import generate_optical_fallback_radar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PipelineRunner")

# Pipeline Stage Constants
STAGE_CONSENSUS_SCRAPED = "CONSENSUS_SCRAPED"
STAGE_CLIPS_EXTRACTED = "CLIPS_EXTRACTED"
STAGE_VISION_ANALYZED = "VISION_ANALYZED"
STAGE_AUDIO_LOCKED = "AUDIO_LOCKED"
STAGE_REMOTION_RENDERED = "REMOTION_RENDERED"

PIPELINE_STAGES = [
    STAGE_CONSENSUS_SCRAPED,
    STAGE_CLIPS_EXTRACTED,
    STAGE_VISION_ANALYZED,
    STAGE_AUDIO_LOCKED,
    STAGE_REMOTION_RENDERED,
]


class PipelineStateTracker:
    """
    Lightweight SQLite3 State Tracker for the Tactical Masterclass Pipeline.
    Records (match_id, clip_id, stage, status, json_payload_path) across pipeline stages:
    CONSENSUS_SCRAPED -> CLIPS_EXTRACTED -> VISION_ANALYZED -> AUDIO_LOCKED -> REMOTION_RENDERED
    """

    def __init__(self, db_path: Union[str, Path] = "pipeline_state.db"):
        self.db_path = Path(db_path)
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_state (
                    match_id TEXT NOT NULL,
                    clip_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    json_payload_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (match_id, clip_id, stage)
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pipeline_lookup
                ON pipeline_state (match_id, clip_id, stage);
            """)
            conn.commit()

    def record_checkpoint(
        self,
        match_id: str,
        clip_id: str,
        stage: str,
        status: str,
        json_payload_path: Optional[Union[str, Path]] = None,
        payload: Optional[Any] = None,
    ) -> str:
        """
        Records or updates a checkpoint in pipeline_state.db.
        If payload is provided, atomically serializes it to json_payload_path.
        """
        payload_path_str = str(json_payload_path) if json_payload_path else None
        if payload is not None and json_payload_path:
            p = Path(json_payload_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            temp_path = p.with_suffix(".tmp")

            def _json_serial(obj):
                if hasattr(obj, "tolist"):
                    return obj.tolist()
                try:
                    import numpy as np
                    if isinstance(obj, (np.integer, np.int64, np.int32)):
                        return int(obj)
                    if isinstance(obj, (np.floating, np.float64, np.float32)):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return None
                except ImportError:
                    pass
                return str(obj)

            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_serial)
            temp_path.replace(p)
            payload_path_str = str(p.resolve())

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO pipeline_state (match_id, clip_id, stage, status, json_payload_path, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(match_id, clip_id, stage) DO UPDATE SET
                    status=excluded.status,
                    json_payload_path=excluded.json_payload_path,
                    updated_at=CURRENT_TIMESTAMP;
            """, (match_id, clip_id, stage, status, payload_path_str))
            conn.commit()
        return payload_path_str or ""

    def get_checkpoint(self, match_id: str, clip_id: str, stage: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT match_id, clip_id, stage, status, json_payload_path, created_at, updated_at
                FROM pipeline_state
                WHERE match_id = ? AND clip_id = ? AND stage = ?;
            """, (match_id, clip_id, stage))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def is_stage_completed(self, match_id: str, clip_id: str, stage: str) -> bool:
        ckpt = self.get_checkpoint(match_id, clip_id, stage)
        if not ckpt or ckpt.get("status") != "COMPLETED":
            return False
        payload_path = ckpt.get("json_payload_path")
        if not payload_path or not Path(payload_path).is_file():
            return False
        return True

    def load_payload(self, match_id: str, clip_id: str, stage: str) -> Optional[Any]:
        if not self.is_stage_completed(match_id, clip_id, stage):
            return None
        ckpt = self.get_checkpoint(match_id, clip_id, stage)
        payload_path = ckpt["json_payload_path"]
        try:
            with open(payload_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached JSON payload from {payload_path}: {e}")
            return None

    def get_match_state(self, match_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT match_id, clip_id, stage, status, json_payload_path, created_at, updated_at
                FROM pipeline_state
                WHERE match_id = ?
                ORDER BY clip_id, stage;
            """, (match_id,))
            return [dict(r) for r in cursor.fetchall()]

    def reset_checkpoint(self, match_id: str, clip_id: Optional[str] = None, stage: Optional[str] = None):
        with self._get_connection() as conn:
            if clip_id and stage:
                conn.execute("DELETE FROM pipeline_state WHERE match_id = ? AND clip_id = ? AND stage = ?;", (match_id, clip_id, stage))
            elif clip_id:
                conn.execute("DELETE FROM pipeline_state WHERE match_id = ? AND clip_id = ?;", (match_id, clip_id))
            else:
                conn.execute("DELETE FROM pipeline_state WHERE match_id = ?;", (match_id,))
            conn.commit()


class TacticalPipelineRunner:
    def __init__(
        self,
        remotion_host: str = "http://127.0.0.1:3008",
        output_dir: str = r"F:\DockerRenders",
        db_path: Optional[Union[str, Path]] = None,
    ):
        self.remotion_host = remotion_host.rstrip("/")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.replays_dir = self.output_dir / "replays"
        self.replays_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir = self.output_dir / "assets" / "clips"
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.output_dir / "assets" / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = self.output_dir / "assets" / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # SQLite State Tracker (pipeline_state.db)
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            self.db_path = self.output_dir / "pipeline_state.db"
        self.state_tracker = PipelineStateTracker(self.db_path)

        self.ingestor = ReplayIngestor(replays_dir=self.replays_dir)
        self.scraper = MultiProviderScraper()
        self.filter = TacticalSignificanceFilter()
        self.vision_client = GeminiVisionClient()
        self.narrator = TacticalNarrator()
        self.player_asset_mgr = PlayerAssetManager(output_dir=str(self.output_dir / "assets" / "players"))
        self.audio_engine = AudioEngine()
        self.evidence_matcher = EvidenceMatcher()
        self.match_resolver = AutonomousMatchResolver()
        self.sparse_indexer = SparseClockIndexer()
        self.time_indexer = None
        self._vision_engine = None

    @property
    def vision_engine(self) -> TacticalVisionEngine:
        if self._vision_engine is None:
            self._vision_engine = TacticalVisionEngine()
        return self._vision_engine

    def run(
        self,
        video_path: Optional[str] = None,
        download_url: Optional[str] = None,
        fixture_id: Optional[str] = None,
        home_team: str = "Barcelona",
        away_team: str = "Atletico Madrid",
        score: str = "2 - 1",
        render: bool = True,
        duration: float = 180.0,
        wait: bool = False,
        matchday_data: Optional[Dict[str, Any]] = None,
        reddit_comments: Optional[List[str]] = None,
        force_refresh: bool = False,
        overrides: Optional[Union[Dict[str, Any], str]] = None,
    ) -> Dict[str, Any]:
        # Parse metadata automatically from local video path if available
        match_date = ""
        competition = "esp.1"
        if video_path and Path(video_path).is_file():
            meta = ReplayIngestor.parse_match_metadata_from_filename(Path(video_path).name)
            if home_team == "Barcelona" and away_team == "Atletico Madrid":
                home_team = meta.home
                away_team = meta.away
                if score == "2 - 1":
                    score = meta.score
            match_date = getattr(meta, "match_date", "")
            competition = getattr(meta, "competition", "esp.1")

        match_id = fixture_id or f"{home_team}_vs_{away_team}".lower().replace(" ", "_")
        match_checkpoint_dir = self.checkpoints_dir / match_id
        match_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"=== STARTING TACTICAL MASTERCLASS PIPELINE [Match: {match_id}] ===")

        # Fast-path: Check if complete REMOTION_RENDERED stage is already completed
        if not force_refresh and self.state_tracker.is_stage_completed(match_id, "__match__", STAGE_REMOTION_RENDERED):
            cached_render = self.state_tracker.load_payload(match_id, "__match__", STAGE_REMOTION_RENDERED)
            if cached_render:
                logger.info(f"[Idempotent Resume] Complete render already cached for {match_id}. Returning cached result.")
                return cached_render

        # -------------------------------------------------------------
        # Stage 1: CONSENSUS_SCRAPED (Autonomous Multi-Provider Resolution)
        # -------------------------------------------------------------
        logger.info(f"[Stage 1/5] Checking consensus scraping for match '{match_id}'...")
        event_windows: List[Dict[str, Any]] = []

        if not force_refresh and self.state_tracker.is_stage_completed(match_id, "__all__", STAGE_CONSENSUS_SCRAPED):
            event_windows = self.state_tracker.load_payload(match_id, "__all__", STAGE_CONSENSUS_SCRAPED) or []
            logger.info(f"[Idempotent Resume] Loaded {len(event_windows)} event windows from cache for {match_id}.")
        else:
            logger.info(f"[Step 1/5] Autonomously resolving official match data & timeline for {home_team} vs {away_team}...")
            
            resolved_match = self.match_resolver.resolve(
                home=home_team,
                away=away_team,
                date=match_date,
                league=competition,
            )
            if resolved_match.get("score") and resolved_match["score"] != "FT":
                score = resolved_match["score"]
            comments = reddit_comments or resolved_match.get("reddit_comments") or [
                f"{home_team} aggressive front-three press disrupted {away_team} buildup continuously.",
                f"Defensive block was repeatedly stretched laterally due to aggressive flank overload.",
                f"Defensive line held high allowing lethal progressive vertical transition passes.",
                f"Zone 14 central midfield space left completely vacant during transition turnovers.",
                f"Overload on flank created uncontested delivery into dangerous half-spaces.",
            ]
            crowd_result = CrowdConsensusScraper.analyze(comments)
            logger.info(f"Crowd consensus tone: {crowd_result.narrative_tone}")

            canonical_events = resolved_match.get("events", [])
            raw_events: List[Dict[str, Any]] = []

            for ev in canonical_events:
                p_name = ev.primary_player.name if ev.primary_player else f"{ev.team.name if ev.team else ''} Player"
                t_name = ev.team.name if ev.team else home_team
                ev_type = ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type)
                is_goal = (ev_type == "goal")

                raw_events.append({
                    "id": ev.id,
                    "period": ev.period,
                    "minute": ev.minute,
                    "second": ev.second,
                    "type": ev_type,
                    "team": t_name,
                    "player": p_name,
                    "description": ev.description or f"{ev_type.upper()} at minute {ev.minute}'",
                    "pitch_x": ev.pitch_x,
                    "pitch_y": ev.pitch_y,
                    "xg": ev.xg,
                    "expected_score": 100 if is_goal else (88 if ev.is_big_chance else 72),
                    "should_retain": True if (is_goal or ev.is_big_chance) else False,
                })

            # Check local match video file for sparse clock indexing
            actual_video_file = Path(video_path) if video_path and Path(video_path).is_file() else None
            if not actual_video_file:
                actual_video_file = self.ingestor.find_local_match_video(home_team, away_team)

            time_indexer = None
            if actual_video_file and Path(actual_video_file).is_file():
                try:
                    logger.info(f"[SparseClockIndexer] Sampling scoreboard clocks from {actual_video_file.name}...")
                    time_indexer = self.sparse_indexer.build_time_indexer(str(actual_video_file))
                    self.time_indexer = time_indexer
                    for evt in raw_events:
                        p = evt.get("period", 1 if evt.get("minute", 0) <= 45 else 2)
                        m = evt.get("minute", 0)
                        s = evt.get("second", 0)
                        try:
                            pts = time_indexer.get_pts_for_match_minute(match_minute=m, second=s, period=p)
                            evt["video_pts"] = round(pts, 1)
                        except Exception as pts_err:
                            logger.warning(f"Could not map minute {m}' to PTS: {pts_err}")
                except Exception as index_err:
                    logger.warning(f"SparseClockIndexer error: {index_err}")

            target_duration = max(30.0, float(duration)) if duration else 600.0
            target_clip_count = max(2, int(math.ceil(target_duration / 26.0)))

            # Priority Ranking: Every goal first, then big chances, shots, and tactical turns spread across both halves
            goals = [e for e in raw_events if e.get("type") == "goal"]
            others = [e for e in raw_events if e.get("type") != "goal"]
            others.sort(key=lambda e: (e.get("should_retain", False), e.get("expected_score", 0)), reverse=True)

            selected_events = goals + others[:max(0, target_clip_count - len(goals))]
            selected_events.sort(key=lambda e: (e.get("period", 1), e.get("minute", 0), e.get("second", 0)))

            lineup_payload = resolved_match.get("lineup_payload") or matchday_data
            for idx, evt in enumerate(selected_events):
                m = evt.get("minute", 10)
                s = evt.get("second", 0)
                ev_period = evt.get("period", 1 if m <= 45 else 2)
                cause_m = max(0, m - 1) if s < 15 else m
                cause_s = (s - 15) % 60
                resolved_roster = MatchLineupResolver.resolve_active_roster(
                    matchday_data=lineup_payload,
                    period=ev_period,
                    minute=m,
                )
                event_windows.append({
                    "clip_id": f"clip_{idx+1:02d}_{evt.get('type', 'tactical')}",
                    "home_team": home_team,
                    "away_team": away_team,
                    "match_id": match_id,
                    "period": ev_period,
                    "t_cause_clock": f"{cause_m:02d}:{cause_s:02d}",
                    "t_effect_clock": f"{m:02d}:{s:02d}",
                    "narrative_tone": crowd_result.narrative_tone,
                    "thesis": evt.get("description", f"Tactical transition breakdown at minute {m}'"),
                    "crowd_evidence": f"Crowd consensus: {evt.get('description', '')}",
                    "active_roster": resolved_roster,
                    "video_pts": evt.get("video_pts"),
                    "raw_event": evt,
                })

            if not event_windows:
                # Spaced tactical sampling across 90 minutes if zero events were scraped
                logger.warning("[AutonomousResolver] Generating fallback event windows across match intervals.")
                interval_min = 90.0 / float(target_clip_count + 1)
                for idx in range(target_clip_count):
                    m = int(round((idx + 1) * interval_min))
                    p = 1 if m <= 45 else 2
                    s = 0
                    pts = None
                    if time_indexer:
                        try:
                            pts = time_indexer.get_pts_for_match_minute(match_minute=m, second=s, period=p)
                        except Exception:
                            pass
                    resolved_roster = MatchLineupResolver.resolve_active_roster(
                        matchday_data=lineup_payload,
                        period=p,
                        minute=m,
                    )
                    event_windows.append({
                        "clip_id": f"clip_{idx+1:02d}_tactical",
                        "home_team": home_team,
                        "away_team": away_team,
                        "match_id": match_id,
                        "period": p,
                        "t_cause_clock": f"{m:02d}:00",
                        "t_effect_clock": f"{m:02d}:15",
                        "narrative_tone": crowd_result.narrative_tone,
                        "thesis": f"Tactical duel and spatial overload analysis at minute {m}'",
                        "crowd_evidence": f"Tactical analysis at minute {m}'",
                        "active_roster": resolved_roster,
                        "video_pts": pts,
                        "raw_event": {"minute": m, "second": 0, "type": "tactical", "period": p},
                    })

            for win in event_windows:
                cid = win["clip_id"]
                win_path = match_checkpoint_dir / f"{cid}_event_window.json"
                self.state_tracker.record_checkpoint(
                    match_id=match_id,
                    clip_id=cid,
                    stage=STAGE_CONSENSUS_SCRAPED,
                    status="COMPLETED",
                    json_payload_path=win_path,
                    payload=win,
                )

            all_path = match_checkpoint_dir / "all_event_windows.json"
            self.state_tracker.record_checkpoint(
                match_id=match_id,
                clip_id="__all__",
                stage=STAGE_CONSENSUS_SCRAPED,
                status="COMPLETED",
                json_payload_path=all_path,
                payload=event_windows,
            )
            logger.info(f"Checkpointed {len(event_windows)} event windows for stage CONSENSUS_SCRAPED.")

        # Check if all clips are already extracted
        all_clips_extracted = (
            all(
                self.state_tracker.is_stage_completed(match_id, w["clip_id"], STAGE_CLIPS_EXTRACTED)
                for w in event_windows
            )
            if event_windows
            else False
        )

        # Step 0: Ingest Video Footage if extraction is needed
        actual_video_file = None
        if not all_clips_extracted or force_refresh:
            if video_path and Path(video_path).exists():
                actual_video_file = Path(video_path)
                logger.info(f"Using explicitly provided local match video: {actual_video_file}")

            if not actual_video_file:
                actual_video_file = self.ingestor.find_local_match_video(home_team, away_team)
                if actual_video_file:
                    logger.info(f"Discovered matching local match video on disk: {actual_video_file}")

            if not actual_video_file and download_url:
                logger.info(f"[Step 0/5] Attempting download via yt-dlp from: {download_url}")
                try:
                    actual_video_file = self.ingestor.download_replay(
                        video_url=download_url,
                        output_name=f"{home_team}_vs_{away_team}".replace(" ", "_"),
                    )
                    logger.info(f"Replay successfully downloaded: {actual_video_file}")
                except Exception as dl_err:
                    logger.warning(f"Direct download_url ({download_url}) failed: {dl_err}. Falling back to automated cascade...")
                    actual_video_file = None

            if not actual_video_file:
                logger.info(f"[Step 0/5] Auto-searching Telegram & FullMatchSports for '{home_team} vs {away_team}'...")
                actual_video_file = self.ingestor.auto_find_and_download_match(home_team, away_team)
                if actual_video_file:
                    logger.info(f"Replay automatically acquired: {actual_video_file}")

            if not actual_video_file or not Path(actual_video_file).exists():
                error_msg = (
                    f"No broadcast match footage found or acquired for '{home_team} vs {away_team}'. "
                    f"Scanned local storage, Telegram channels, and Web replay aggregators. "
                    f"Video rendering aborted per zero-placeholder policy."
                )
                logger.error(f"[Fail-Hard Policy] {error_msg}")
                raise FileNotFoundError(error_msg)
        else:
            logger.info(f"[Idempotent Resume] All clips already extracted for {match_id}. Bypassing raw video requirement.")

        # Attach overrides if provided
        parsed_overrides = None
        if overrides:
            if isinstance(overrides, str):
                try:
                    if Path(overrides).is_file():
                        with open(overrides, "r", encoding="utf-8") as f:
                            parsed_overrides = json.load(f)
                    else:
                        parsed_overrides = json.loads(overrides)
                except Exception as e:
                    logger.warning("Failed to parse overrides in PipelineRunner: %s", e)
                    parsed_overrides = {}
            elif isinstance(overrides, dict):
                parsed_overrides = overrides

        if parsed_overrides:
            for win in event_windows:
                win["overrides"] = parsed_overrides

        # Process clips through Stages 2 (CLIPS_EXTRACTED) -> 3 (VISION_ANALYZED) -> 4 (AUDIO_LOCKED)
        synchronized_clips: List[Dict[str, Any]] = []

        for clip_idx, event_window in enumerate(event_windows):
            cid = event_window.get("clip_id", f"clip_{clip_idx+1:02d}")
            logger.info(f"--- Processing Clip #{clip_idx+1}/{len(event_windows)}: {cid} ---")

            # -------------------------------------------------------------
            # Stage 2: CLIPS_EXTRACTED
            # -------------------------------------------------------------
            extracted_clip = None
            if not force_refresh and self.state_tracker.is_stage_completed(match_id, cid, STAGE_CLIPS_EXTRACTED):
                extracted_clip = self.state_tracker.load_payload(match_id, cid, STAGE_CLIPS_EXTRACTED)
                logger.info(f"[Idempotent Resume] Loaded extracted clip for {cid} from cache.")

            if not extracted_clip:
                logger.info(f"[Stage 2/5] Extracting video clip for {cid}...")
                try:
                    active_indexer = self.time_indexer
                    if active_indexer is None and actual_video_file and Path(actual_video_file).is_file():
                        active_indexer = self.sparse_indexer.build_time_indexer(str(actual_video_file))
                        self.time_indexer = active_indexer
                    clip_extractor = ClipExtractor(
                        time_indexer=active_indexer or MatchTimeIndexer(),
                        media_source=str(actual_video_file) if actual_video_file else "",
                        output_dir=str(self.clips_dir),
                        frames_dir=str(self.frames_dir),
                    )
                    extracted_clip = clip_extractor.extract_event_from_contract(event_window)
                    ext_path = match_checkpoint_dir / f"{cid}_extracted.json"
                    self.state_tracker.record_checkpoint(
                        match_id=match_id,
                        clip_id=cid,
                        stage=STAGE_CLIPS_EXTRACTED,
                        status="COMPLETED",
                        json_payload_path=ext_path,
                        payload=extracted_clip,
                    )
                except Exception as e:
                    logger.error(f"Clip extraction failed on {cid}: {e}", exc_info=True)
                    self.state_tracker.record_checkpoint(
                        match_id=match_id,
                        clip_id=cid,
                        stage=STAGE_CLIPS_EXTRACTED,
                        status="FAILED",
                    )
                    raise

            # -------------------------------------------------------------
            # Stage 3: VISION_ANALYZED
            # -------------------------------------------------------------
            vision_data = None
            if not force_refresh and self.state_tracker.is_stage_completed(match_id, cid, STAGE_VISION_ANALYZED):
                vision_data = self.state_tracker.load_payload(match_id, cid, STAGE_VISION_ANALYZED)
                logger.info(f"[Idempotent Resume] Loaded vision analysis for {cid} from cache.")

            if not vision_data:
                logger.info(f"[Stage 3/5] Running vision analysis & telestration extraction for {cid}...")
                try:
                    clip_vid_path = extracted_clip.get("clip_path")
                    c_idx = extracted_clip.get("cause_frame_idx", 90)
                    e_idx = extracted_clip.get("effect_frame_idx", 240)

                    vision_data = self.vision_engine.generate_synchronized_clip_vision(
                        video_path=clip_vid_path,
                        event_window=event_window,
                        cause_frame_idx=c_idx,
                        effect_frame_idx=e_idx,
                        fps=extracted_clip.get("fps", 30),
                    )
                    vision_data["clip_id"] = cid

                    vis_path = match_checkpoint_dir / f"{cid}_vision.json"
                    self.state_tracker.record_checkpoint(
                        match_id=match_id,
                        clip_id=cid,
                        stage=STAGE_VISION_ANALYZED,
                        status="COMPLETED",
                        json_payload_path=vis_path,
                        payload=vision_data,
                    )
                except Exception as e:
                    logger.error(f"Vision analysis failed on {cid}: {e}", exc_info=True)
                    self.state_tracker.record_checkpoint(
                        match_id=match_id,
                        clip_id=cid,
                        stage=STAGE_VISION_ANALYZED,
                        status="FAILED",
                    )
                    raise

            # -------------------------------------------------------------
            # Stage 4: AUDIO_LOCKED
            # -------------------------------------------------------------
            audio_locked_data = None
            if not force_refresh and self.state_tracker.is_stage_completed(match_id, cid, STAGE_AUDIO_LOCKED):
                audio_locked_data = self.state_tracker.load_payload(match_id, cid, STAGE_AUDIO_LOCKED)
                logger.info(f"[Idempotent Resume] Loaded locked audio for {cid} from cache.")

            if not audio_locked_data:
                logger.info(f"[Stage 4/5] Generating 3-beat tactical script & locking TTS audio for {cid}...")
                try:
                    if not event_window.get("home_team"):
                        event_window["home_team"] = home_team
                    if not event_window.get("away_team"):
                        event_window["away_team"] = away_team
                    if not event_window.get("match_id"):
                        event_window["match_id"] = match_id
                    cause_frame_path = extracted_clip.get("cause_frame_path")
                    beats_script = self.narrator.generate_3beat_script(
                        event_window=event_window,
                        cause_telestration=vision_data.get("cause_telestration"),
                        freeze_frame_path=str(cause_frame_path) if cause_frame_path else None,
                    )
                    clip_audio_dir = self.audio_dir / cid
                    clip_audio_dir.mkdir(parents=True, exist_ok=True)

                    c_idx = vision_data.get("cause_frame_idx", extracted_clip.get("cause_frame_idx", 90))
                    vid_file = extracted_clip.get("clip_path") or vision_data.get("video_path")
                    clip_video_frames = 0
                    if vid_file and Path(vid_file).is_file():
                        try:
                            _cap = cv2.VideoCapture(str(vid_file))
                            if _cap.isOpened():
                                clip_video_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                _cap.release()
                        except Exception:
                            clip_video_frames = 0

                    try:
                        locked_beats = self.audio_engine.synthesize_3beat_commentary(
                            beats=beats_script,
                            output_dir=clip_audio_dir,
                            fps=vision_data.get("fps", 30),
                            cause_frame_idx=c_idx,
                            actual_clip_video_frames=clip_video_frames if clip_video_frames > 0 else None,
                        )
                    except TypeError as te:
                        if "unexpected keyword argument" in str(te):
                            locked_beats = self.audio_engine.synthesize_3beat_commentary(
                                beats=beats_script,
                                output_dir=clip_audio_dir,
                                fps=vision_data.get("fps", 30),
                            )
                        else:
                            raise

                    # Physical video freeze frame index (verified by Autonomous Tactical Frame Intelligence Gate)
                    b1_val = locked_beats.get("beat_1_setup")
                    b1_ms = b1_val.get("duration_ms", 2000) if isinstance(b1_val, dict) else 2000
                    b1_sec = b1_ms / 1000.0
                    fps_val = vision_data.get("fps", 30) or 30
                    beat_1_frames = int(math.ceil((b1_sec + 0.15) * fps_val))

                    # Prefer the verified tactical frame from ClipExtractor / VisionEngine over blind beat_1_frames
                    K_video = int(
                        extracted_clip.get("cause_frame_idx")
                        or vision_data.get("cause_frame_idx")
                        or getattr(TacticalVisionEngine, "CLIP_TARGETS", {}).get(cid, {}).get("k_video", beat_1_frames)
                    )
                    is_clip_tactical = bool(extracted_clip.get("is_tactical", vision_data.get("is_tactical", True)))

                    cause_frame_path = (
                        extracted_clip.get("cause_frame_path")
                        or vision_data.get("cause_frame_path")
                        or str(self.frames_dir / f"{cid}_cause.jpg")
                    )

                    # Verify that frame K_video passes the Tactical Frame Intelligence Gate; if not, scan clip for best wide frame
                    from src.vision.shot_classifier import CameraShotClassifier, ShotType
                    freeze_frame_b64 = None
                    if vid_file and Path(vid_file).is_file() and cause_frame_path:
                        cap = cv2.VideoCapture(str(vid_file))
                        if cap.isOpened():
                            cap.set(cv2.CAP_PROP_POS_FRAMES, K_video)
                            ret_f, f_img = cap.read()
                            cap.release()
                            if ret_f and f_img is not None:
                                st_k, metrics_k = CameraShotClassifier.classify(f_img)
                                if st_k == ShotType.CLOSE_UP:
                                    logger.warning(
                                        f"[Tactical Intelligence Gate] Frame {K_video} in {cid} rejected ({metrics_k.get('rejection_reasons')}). "
                                        f"Scanning clip for highest-interest wide tactical frame..."
                                    )
                                    temp_ext = ClipExtractor(
                                        time_indexer=self.time_indexer or MatchTimeIndexer(),
                                        media_source=str(vid_file),
                                        output_dir=str(self.clips_dir),
                                        frames_dir=str(self.frames_dir),
                                    )
                                    best_tac = temp_ext.extract_tactical_keyframe(
                                        clip_path=str(vid_file),
                                        output_path=str(cause_frame_path),
                                    )
                                    if best_tac and best_tac.get("frame_idx") is not None:
                                        K_video = int(best_tac["frame_idx"])
                                        is_clip_tactical = True
                                        f_img = cv2.imread(str(cause_frame_path))
                                    else:
                                        is_clip_tactical = False
                                        logger.warning(
                                            f"[Tactical Intelligence Gate] No valid wide tactical frame exists in {cid}. "
                                            f"Disabling freeze & telestration for {cid}."
                                        )
                                else:
                                    is_clip_tactical = True

                                if f_img is not None:
                                    Path(cause_frame_path).parent.mkdir(parents=True, exist_ok=True)
                                    cv2.imwrite(str(cause_frame_path), f_img)
                                    _, buffer = cv2.imencode(".jpg", f_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                                    freeze_frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                                    logger.info(
                                        f"Saved verified freeze frame for {cid} at frame {K_video} "
                                        f"(is_tactical={is_clip_tactical}) to {cause_frame_path}"
                                    )

                    # Set T_preroll and V_freeze to verified frame K = K_video
                    K = K_video
                    video_freeze_frame = K
                    pre_roll_duration_frames = K
                    freeze_frame = K
                    cause_frame_idx = K
                    adjusted_c_idx = K

                    vision_data["is_tactical"] = is_clip_tactical
                    vision_data["is_close_up"] = not is_clip_tactical
                    vision_data["video_freeze_frame"] = K if is_clip_tactical else None
                    vision_data["videoFreezeFrame"] = K if is_clip_tactical else None
                    vision_data["pre_roll_duration_frames"] = K
                    vision_data["preRollDurationFrames"] = K
                    vision_data["cause_frame_idx"] = K
                    vision_data["causeFrameIdx"] = K
                    vision_data["freeze_frame"] = K if is_clip_tactical else None
                    if not is_clip_tactical:
                        vision_data["freeze_frames"] = 0
                    if clip_video_frames > 0:
                        vision_data["actual_video_frames"] = clip_video_frames
                    if "cause_telestration" in vision_data and isinstance(vision_data["cause_telestration"], dict):
                        ctel = vision_data["cause_telestration"]
                        ctel["is_tactical"] = is_clip_tactical
                        ctel["is_close_up"] = not is_clip_tactical
                        ctel["video_freeze_frame"] = K if is_clip_tactical else None
                        ctel["videoFreezeFrame"] = K if is_clip_tactical else None
                        ctel["pre_roll_duration_frames"] = K
                        ctel["preRollDurationFrames"] = K
                        ctel["cause_frame_idx"] = K
                        ctel["causeFrameIdx"] = K
                        ctel["freeze_frame"] = K if is_clip_tactical else None
                        if not is_clip_tactical:
                            ctel["freeze_frames"] = 0
                        ctel["step_1_ms"] = locked_beats.get("step_1_ms")
                        ctel["step_2_ms"] = locked_beats.get("step_2_ms")
                        ctel["step_3_ms"] = locked_beats.get("step_3_ms")
                        ctel["step_4_ms"] = locked_beats.get("step_4_ms")
                        ctel["step_1_start_ms"] = locked_beats.get("step_1_start_ms", 0)
                        ctel["step_2_start_ms"] = locked_beats.get("step_2_start_ms")
                        ctel["step_3_start_ms"] = locked_beats.get("step_3_start_ms")
                        ctel["step_4_start_ms"] = locked_beats.get("step_4_start_ms")
                        ctel["step_boundaries_ms"] = locked_beats.get("step_boundaries_ms")
                        ctel["step_1_frames"] = locked_beats.get("step_1_frames")
                        ctel["step_2_frames"] = locked_beats.get("step_2_frames")
                        ctel["step_3_frames"] = locked_beats.get("step_3_frames")
                        ctel["step_4_frames"] = locked_beats.get("step_4_frames")
                        ctel["step_boundaries_frames"] = locked_beats.get("step_boundaries_frames")
                        if "beat_2_steps" in locked_beats:
                            ctel["beat_2_steps"] = locked_beats["beat_2_steps"]
                    locked_beats["is_tactical"] = is_clip_tactical
                    locked_beats["is_close_up"] = not is_clip_tactical
                    locked_beats["video_freeze_frame"] = K if is_clip_tactical else None
                    locked_beats["videoFreezeFrame"] = K if is_clip_tactical else None
                    locked_beats["pre_roll_duration_frames"] = K
                    locked_beats["preRollDurationFrames"] = K
                    locked_beats["cause_frame_idx"] = K
                    locked_beats["causeFrameIdx"] = K
                    locked_beats["freeze_frame"] = K if is_clip_tactical else None
                    if not is_clip_tactical and isinstance(locked_beats.get("beat_2_cause"), dict):
                        locked_beats["beat_2_cause"]["freeze_frames"] = 0

                    if freeze_frame_b64 and is_clip_tactical:
                        vision_data["freeze_frame_b64"] = freeze_frame_b64
                        vision_data["freezeFrameUrl"] = freeze_frame_b64
                        vision_data["freeze_frame_url"] = freeze_frame_b64

                    # Propagate tracked players to adjusted cause frame index and re-align
                    if "tracks_per_frame" in vision_data and isinstance(vision_data["tracks_per_frame"], dict):
                        t_map = vision_data["tracks_per_frame"]
                        if str(c_idx) in t_map and str(adjusted_c_idx) not in t_map:
                            t_map[str(adjusted_c_idx)] = t_map[str(c_idx)]
                        if "cause_telestration" in vision_data and isinstance(vision_data["cause_telestration"], dict):
                            self._realign_cause_telestration(vision_data["cause_telestration"], t_map, freeze_frame, clip_id=cid)

                    vision_data["beats"] = locked_beats
                    audio_locked_data = dict(vision_data)

                    audio_path = match_checkpoint_dir / f"{cid}_audio_locked.json"
                    self.state_tracker.record_checkpoint(
                        match_id=match_id,
                        clip_id=cid,
                        stage=STAGE_AUDIO_LOCKED,
                        status="COMPLETED",
                        json_payload_path=audio_path,
                        payload=audio_locked_data,
                    )
                except Exception as e:
                    logger.error(f"Audio locking failed on {cid}: {e}", exc_info=True)
                    self.state_tracker.record_checkpoint(
                        match_id=match_id,
                        clip_id=cid,
                        stage=STAGE_AUDIO_LOCKED,
                        status="FAILED",
                    )
                    raise

            synchronized_clips.append(audio_locked_data)

        # -------------------------------------------------------------
        # Stage 5: REMOTION_RENDERED
        # -------------------------------------------------------------
        # Respect target duration if requested
        if duration and float(duration) > 0 and len(synchronized_clips) > 2:
            target_dur = float(duration)
            trimmed_clips = []
            accum_sec = 0.0
            for sc in synchronized_clips:
                trimmed_clips.append(sc)
                dur_frames = sc.get("total_composition_frames") or sc.get("duration_frames") or 500
                accum_sec += (float(dur_frames) / 30.0)
                if accum_sec >= target_dur and len(trimmed_clips) >= 2:
                    break
            if len(trimmed_clips) < len(synchronized_clips):
                logger.info(f"[Target Duration] Trimmed synchronized clips from {len(synchronized_clips)} to {len(trimmed_clips)} to match target {target_dur}s (~{accum_sec:.1f}s).")
                synchronized_clips = trimmed_clips

        logger.info(f"[Stage 5/5] Assembling Masterclass timeline & rendering ({len(synchronized_clips)} clips)...")

        active_segments: List[Dict[str, Any]] = []
        timeline_commentaries: List[Dict[str, Any]] = []
        current_time = 0.0

        for s_idx, sc in enumerate(synchronized_clips):
            cid = sc.get("clip_id", f"clip_{s_idx+1:02d}")
            beats = sc.get("beats") or {}
            b1_val = beats.get("beat_1_setup")
            b1_ms = b1_val.get("duration_ms", 3800) if isinstance(b1_val, dict) else 3800

            b2_val = beats.get("beat_2_cause")
            b2_info = b2_val if isinstance(b2_val, dict) else {}
            b2_ms = b2_val.get("duration_ms", 5400) if isinstance(b2_val, dict) else 5400

            b3_val = beats.get("beat_3_effect")
            b3_ms = b3_val.get("duration_ms", 4200) if isinstance(b3_val, dict) else 4200

            # Fix 1: open each extracted clip .mp4 with OpenCV (cv2.VideoCapture) and read physical frame count
            vid_path = sc.get("video_path") or sc.get("clip_path") or str(self.clips_dir / f"{cid}.mp4")
            actual_clip_video_frames = 0
            if vid_path and Path(vid_path).is_file():
                try:
                    cap = cv2.VideoCapture(str(vid_path))
                    if cap.isOpened():
                        actual_clip_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                except Exception:
                    actual_clip_video_frames = 0

            if actual_clip_video_frames <= 0 and sc.get("actual_video_frames"):
                actual_clip_video_frames = int(sc.get("actual_video_frames"))

            # Physical video freeze frame vs pre-roll duration: K = K_video (verified by Tactical Frame Intelligence Gate)
            b1_sec = b1_ms / 1000.0
            beat_1_frames = int(math.ceil((b1_sec + 0.15) * 30))
            b2_sec = b2_ms / 1000.0
            beat_2_frames = int(math.ceil(b2_sec * 30))

            is_clip_tactical = bool(sc.get("is_tactical", sc.get("cause_telestration", {}).get("is_tactical", True)))
            K_video = int(
                sc.get("cause_frame_idx")
                or sc.get("video_freeze_frame")
                or getattr(TacticalVisionEngine, "CLIP_TARGETS", {}).get(cid, {}).get("k_video", beat_1_frames)
            )
            if is_clip_tactical:
                freeze_end_frame = max(K_video, beat_1_frames) + beat_2_frames + 12
                freeze_frames = freeze_end_frame - K_video
            else:
                freeze_end_frame = K_video
                freeze_frames = 0

            K = K_video
            video_freeze_frame = K if is_clip_tactical else None
            pre_roll_duration_frames = K
            freeze_frame = K if is_clip_tactical else None
            cause_frame_idx = K

            b3_sec = b3_ms / 1000.0
            beat_3_frames = math.ceil(b3_sec * 30) + 6

            # Eliminate trailing black screens between clips: post_roll_frames strictly bounded by video
            max_available_post_roll = max(1, actual_clip_video_frames - K - 2)

            b3_path = None
            if isinstance(b3_val, dict) and b3_val.get("wav_path"):
                b3_path = Path(b3_val["wav_path"])
            if not b3_path or not b3_path.is_file():
                cand_b3 = self.audio_dir / f"{cid}_beat_3.wav"
                if cand_b3.is_file():
                    b3_path = cand_b3

            if beat_3_frames > max_available_post_roll and b3_path and b3_path.is_file():
                avail_sec = max(0.5, (max_available_post_roll / 30.0) - 0.2)
                speed_factor = min(1.30, max(1.10, b3_sec / avail_sec))
                temp_b3 = b3_path.with_name(f"temp_speed_{b3_path.name}")
                try:
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(b3_path),
                        "-filter:a", f"atempo={speed_factor:.3f}",
                        str(temp_b3),
                    ]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    if temp_b3.is_file() and temp_b3.stat().st_size > 0:
                        temp_b3.replace(b3_path)
                        b3_sec = b3_sec / speed_factor
                        b3_ms = int(b3_sec * 1000)
                        beat_3_frames = math.ceil(b3_sec * 30) + 6
                        if isinstance(b3_val, dict):
                            b3_val["duration_sec"] = b3_sec
                            b3_val["duration_ms"] = b3_ms
                        logger.info(
                            f"Sped up beat_3 commentary for {cid} by {speed_factor:.2f}x to fit available {max_available_post_roll} post-roll frames"
                        )
                except Exception as e:
                    logger.warning(f"Could not speed up beat_3 for {cid}: {e}")
                    if temp_b3.is_file():
                        temp_b3.unlink(missing_ok=True)

            post_roll_frames = min(max_available_post_roll, max(120, beat_3_frames))
            total_composition_frames = cause_frame_idx + freeze_frames + post_roll_frames
            seg_dur = round(total_composition_frames / 30.0, 4)

            b3_start_frame = cause_frame_idx + freeze_frames

            vid_name = Path(vid_path).name if vid_path else f"{cid}.mp4"
            clip_url = f"http://127.0.0.1:3000/assets/clips/{vid_name}"

            frame_path = sc.get("cause_frame_path") or sc.get("frame_path")
            frame_name = Path(frame_path).name if frame_path else f"{cid}_cause.jpg"
            target_frame_disk = self.frames_dir / frame_name

            if not target_frame_disk.is_file() and vid_path and Path(vid_path).is_file():
                try:
                    cap = cv2.VideoCapture(str(vid_path))
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, K)
                        ret_f, f_img = cap.read()
                        if ret_f and f_img is not None:
                            cv2.imwrite(str(target_frame_disk), f_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                        cap.release()
                except Exception:
                    pass

            freeze_frame_url = f"http://127.0.0.1:3000/assets/frames/{frame_name}" if is_clip_tactical else None
            fg_mask_url = (sc.get("fg_mask_url") or f"http://127.0.0.1:3000/assets/frames/{cid}_fg_mask.png") if is_clip_tactical else None

            cause_tel = dict(sc.get("cause_telestration", {}))
            cause_tel["is_tactical"] = is_clip_tactical
            cause_tel["is_close_up"] = not is_clip_tactical
            cause_tel["video_freeze_frame"] = video_freeze_frame
            cause_tel["videoFreezeFrame"] = video_freeze_frame
            cause_tel["pre_roll_duration_frames"] = pre_roll_duration_frames
            cause_tel["preRollDurationFrames"] = pre_roll_duration_frames
            cause_tel["cause_frame_idx"] = K
            cause_tel["causeFrameIdx"] = K
            cause_tel["freeze_frame"] = freeze_frame
            cause_tel["freeze_frames"] = freeze_frames
            cause_tel["freezeFrameUrl"] = freeze_frame_url
            cause_tel["freeze_frame_url"] = freeze_frame_url
            cause_tel["show_score_badge"] = False  # Suppress fake top-right score pill
            t_map = sc.get("tracks_per_frame", {})
            if isinstance(t_map, dict) and is_clip_tactical:
                c_orig = sc.get("cause_frame_idx")
                if c_orig is not None and str(c_orig) in t_map and str(K) not in t_map:
                    t_map[str(K)] = t_map[str(c_orig)]
                self._realign_cause_telestration(cause_tel, t_map, K, clip_id=cid)

            # Step reveal timings (both in milliseconds and Remotion frame offsets)
            step_1_ms = b2_info.get("step_1_ms") or (beats.get("step_1_ms") if isinstance(beats, dict) else None) or int(b2_ms * 0.25)
            step_2_ms = b2_info.get("step_2_ms") or (beats.get("step_2_ms") if isinstance(beats, dict) else None) or int(b2_ms * 0.25)
            step_3_ms = b2_info.get("step_3_ms") or (beats.get("step_3_ms") if isinstance(beats, dict) else None) or int(b2_ms * 0.25)
            step_4_ms = b2_info.get("step_4_ms") or (beats.get("step_4_ms") if isinstance(beats, dict) else None) or max(0, b2_ms - (step_1_ms + step_2_ms + step_3_ms))

            step_1_start = 0
            step_2_start = step_1_ms
            step_3_start = step_1_ms + step_2_ms
            step_4_start = step_1_ms + step_2_ms + step_3_ms
            step_bounds = b2_info.get("step_boundaries_ms") or (beats.get("step_boundaries_ms") if isinstance(beats, dict) else None) or [0, step_1_ms, step_1_ms + step_2_ms, step_1_ms + step_2_ms + step_3_ms, b2_ms]

            step_1_frames = b2_info.get("step_1_frames") or (beats.get("step_1_frames") if isinstance(beats, dict) else None) or max(1, int(round((step_1_ms / max(1, b2_ms)) * freeze_frames)))
            step_2_frames = b2_info.get("step_2_frames") or (beats.get("step_2_frames") if isinstance(beats, dict) else None) or max(1, int(round((step_2_ms / max(1, b2_ms)) * freeze_frames)))
            step_3_frames = b2_info.get("step_3_frames") or (beats.get("step_3_frames") if isinstance(beats, dict) else None) or max(1, int(round((step_3_ms / max(1, b2_ms)) * freeze_frames)))
            step_4_frames = b2_info.get("step_4_frames") or (beats.get("step_4_frames") if isinstance(beats, dict) else None) or max(0, freeze_frames - (step_1_frames + step_2_frames + step_3_frames))
            step_bounds_frames = b2_info.get("step_boundaries_frames") or (beats.get("step_boundaries_frames") if isinstance(beats, dict) else None) or [
                0, step_1_frames, step_1_frames + step_2_frames, step_1_frames + step_2_frames + step_3_frames, freeze_frames
            ]

            cause_tel["step_1_ms"] = step_1_ms
            cause_tel["step_2_ms"] = step_2_ms
            cause_tel["step_3_ms"] = step_3_ms
            cause_tel["step_4_ms"] = step_4_ms
            cause_tel["step_1_start_ms"] = step_1_start
            cause_tel["step_2_start_ms"] = step_2_start
            cause_tel["step_3_start_ms"] = step_3_start
            cause_tel["step_4_start_ms"] = step_4_start
            cause_tel["step_boundaries_ms"] = step_bounds
            cause_tel["step_1_frames"] = step_1_frames
            cause_tel["step_2_frames"] = step_2_frames
            cause_tel["step_3_frames"] = step_3_frames
            cause_tel["step_4_frames"] = step_4_frames
            cause_tel["step_boundaries_frames"] = step_bounds_frames

            b2_steps = (beats.get("beat_2_steps") if isinstance(beats, dict) else None) or b2_info.get("steps")
            if b2_steps:
                cause_tel["beat_2_steps"] = b2_steps

            # Goal Impact Synchronization & Suppression of False Goal Flashes
            # NEVER play sfx_goal_impact.wav or trigger green flash if is_actual_goal_scored_in_clip == False
            raw_event = sc.get("raw_event") or {}
            event_type = str(raw_event.get("type", "")).lower()
            is_goal_event = (event_type == "goal")

            # Vision assessment takes precedence, event metadata fallback
            vision_is_goal = cause_tel.get("is_actual_goal_scored_in_clip")
            if vision_is_goal is not None:
                is_actual_goal = bool(vision_is_goal)
            else:
                is_actual_goal = is_goal_event

            goal_sfx_trigger_frame = None
            goal_impact_frame_idx = None

            if is_actual_goal:
                raw_impact_idx = cause_tel.get("goal_impact_frame_idx") or sc.get("goal_impact_frame_idx")
                if raw_impact_idx is not None:
                    goal_impact_frame_idx = int(raw_impact_idx)
                else:
                    goal_impact_frame_idx = sc.get("effect_frame_idx", 240)

                # Strictly trigger at cause_frame_idx + freeze_frames + (goal_impact_frame_idx - cause_frame_idx)
                goal_sfx_trigger_frame = cause_frame_idx + freeze_frames + max(0, goal_impact_frame_idx - cause_frame_idx)

            cause_tel["is_actual_goal_scored_in_clip"] = is_actual_goal
            cause_tel["goal_impact_frame_idx"] = goal_impact_frame_idx
            cause_tel["goal_sfx_trigger_frame"] = goal_sfx_trigger_frame

            seg = {
                "id": f"seg_{cid}",
                "type": "clip",
                "url": clip_url,
                "videoUrl": clip_url,
                "is_tactical": is_clip_tactical,
                "isTactical": is_clip_tactical,
                "is_close_up": not is_clip_tactical,
                "isCloseUp": not is_clip_tactical,
                "is_warped": True,
                "isWarped": True,
                "warped_video_url": clip_url,
                "warpedVideoUrl": clip_url,
                "freezeFrameUrl": freeze_frame_url,
                "freeze_frame_url": freeze_frame_url,
                "telestration": {
                    "is_tactical": is_clip_tactical,
                    "isTactical": is_clip_tactical,
                    "is_close_up": not is_clip_tactical,
                    "isCloseUp": not is_clip_tactical,
                    "freezeFrameUrl": freeze_frame_url,
                    "freeze_frame_url": freeze_frame_url,
                },
                "fgMaskUrl": fg_mask_url,
                "fg_mask_url": fg_mask_url,
                "startTime": round(current_time, 4),
                "duration": seg_dur,
                "total_composition_frames": total_composition_frames,
                "durationInFrames": total_composition_frames,
                "duration_frames": total_composition_frames,
                "actual_clip_video_frames": actual_clip_video_frames,
                "actual_video_frames": actual_clip_video_frames,
                "actualVideoFrames": actual_clip_video_frames,
                "total_video_frames": actual_clip_video_frames,
                "totalVideoFrames": actual_clip_video_frames,
                "post_roll_frames": post_roll_frames,
                "postRollFrames": post_roll_frames,
                "title": f"TACTICAL BREAKDOWN // {cid.upper()}",
                "video_freeze_frame": video_freeze_frame,
                "videoFreezeFrame": video_freeze_frame,
                "pre_roll_duration_frames": pre_roll_duration_frames,
                "preRollDurationFrames": pre_roll_duration_frames,
                "cause_frame_idx": cause_frame_idx,
                "causeFrameIdx": cause_frame_idx,
                "effect_frame_idx": sc.get("effect_frame_idx", 240),
                "freeze_frames": freeze_frames,
                "freezeFrames": freeze_frames,
                "tracks_per_frame": sc.get("tracks_per_frame", {}),
                "cause_telestration": cause_tel,
                "team_colors": sc.get("team_colors", {"home_hex": "#8a1529", "away_hex": "#ffffff"}),
                "beats": beats,
                "show_score_badge": False,
                "is_actual_goal_scored_in_clip": is_actual_goal,
                "goal_impact_frame_idx": goal_impact_frame_idx,
                "goal_sfx_trigger_frame": goal_sfx_trigger_frame,
                "step_1_ms": step_1_ms,
                "step_2_ms": step_2_ms,
                "step_3_ms": step_3_ms,
                "step_4_ms": step_4_ms,
                "step_1_start_ms": step_1_start,
                "step_2_start_ms": step_2_start,
                "step_3_start_ms": step_3_start,
                "step_4_start_ms": step_4_start,
                "step_boundaries_ms": step_bounds,
                "step_1_frames": step_1_frames,
                "step_2_frames": step_2_frames,
                "step_3_frames": step_3_frames,
                "step_4_frames": step_4_frames,
                "step_boundaries_frames": step_bounds_frames,
                "beat_2_steps": b2_steps,
            }

            # Inject 2D Tactical Minimap Radar Data
            try:
                radar_data = generate_optical_fallback_radar(sc)
                seg["minimap_radar"] = radar_data
                seg["minimapRadar"] = radar_data
            except Exception as radar_err:
                logger.warning(f"Minimap radar generation warning for {cid}: {radar_err}")

            active_segments.append(seg)

            # Synchronized voiceover tracks with exact beat start points
            b1_info = beats.get("beat_1_setup")
            if isinstance(b1_info, dict) and b1_info.get("wav_path"):
                timeline_commentaries.append({
                    "text": b1_info.get("text", ""),
                    "startTime": round(current_time, 4),
                    "audio_path": b1_info.get("wav_path"),
                    "duration": (b1_info.get("duration_ms", 3000) / 1000.0),
                    "seg_id": seg["id"],
                })

            b2_info = beats.get("beat_2_cause")
            if isinstance(b2_info, dict) and b2_info.get("wav_path"):
                timeline_commentaries.append({
                    "text": b2_info.get("text", ""),
                    "startTime": round(current_time + (cause_frame_idx / 30.0), 4),
                    "audio_path": b2_info.get("wav_path"),
                    "duration": (b2_info.get("duration_ms", 3000) / 1000.0),
                    "seg_id": seg["id"],
                })

            b3_info = beats.get("beat_3_effect")
            if isinstance(b3_info, dict) and b3_info.get("wav_path"):
                timeline_commentaries.append({
                    "text": b3_info.get("text", ""),
                    "startTime": round(current_time + (b3_start_frame / 30.0), 4),
                    "audio_path": b3_info.get("wav_path"),
                    "duration": (b3_info.get("duration_ms", 3000) / 1000.0),
                    "seg_id": seg["id"],
                })

            current_time += seg_dur

        payload = {
            "title": f"{home_team.upper()} vs {away_team.upper()} // TACTICAL MASTERCLASS",
            "homeTeam": home_team,
            "awayTeam": away_team,
            "score": score,
            "segments": active_segments,
            "durationInSeconds": round(current_time, 2),
            "compositionId": "TacticalMasterclass",
        }

        # Generate word-level dynamic subtitles from timeline commentaries
        all_subtitles = []
        highlight_keywords = {
            "GOAL", "PASS", "SPACE", "BEHIND", "DEFENSE", "PRESS", "CHANCE", "SHOT",
            "CROSS", "BLOCK", "COUNTER", "OVERLOAD", "TACKLE", "DECISION", "MBAPPÉ",
            "MBAPPE", "VINÍCIUS", "VINICIUS", "BELLINGHAM", "RODRYGO", "YAMAL",
            "LEWANDOWSKI", "PEDRI", "GAVI", "KOUNDÉ", "KOUNDE", "VALVERDE", "MODRIC"
        }
        for comm in timeline_commentaries:
            c_text = comm.get("text", "")
            c_start = comm.get("startTime", 0.0)
            c_dur = comm.get("duration", 3.0)
            words = c_text.split()
            if not words:
                continue
            w_dur = c_dur / len(words)
            for idx_w, w in enumerate(words):
                clean_w = w.strip(".,!?:;\"'()[]")
                w_start = int((c_start + idx_w * w_dur) * 1000)
                w_end = int((c_start + (idx_w + 1) * w_dur) * 1000)
                is_hi = clean_w.upper() in highlight_keywords or (clean_w.isupper() and len(clean_w) > 2)
                all_subtitles.append({
                    "word": w.upper(),
                    "startMs": w_start,
                    "endMs": w_end,
                    "highlight": is_hi,
                })
        payload["subtitles"] = all_subtitles

        # Synthesize composite masterclass audio track only if individual choreographed beats are NOT present
        has_choreographed_beats = any(bool(seg.get("beats")) for seg in active_segments)
        if timeline_commentaries and not has_choreographed_beats:
            try:
                audio_out_file = self.audio_dir / f"{match_id}_masterclass.mp3"
                audio_res = self.audio_engine.generate_masterclass_audio_track(
                    timeline_commentaries=timeline_commentaries,
                    total_duration=payload["durationInSeconds"],
                    output_path=str(audio_out_file),
                )
                if audio_res.get("audio_base64"):
                    payload["audioBase64"] = audio_res["audio_base64"]
                payload["audioUrl"] = f"http://127.0.0.1:3000/assets/audio/{audio_out_file.name}"
            except Exception as audio_err:
                logger.warning(f"Audio commentary timeline mix warning: {audio_err}")

        if render:
            logger.info("[Step 5/5] Dispatching render to Remotion container...")
            try:
                job_id = self._dispatch_render(payload)
                logger.info(f"Render job dispatched: {job_id}")
                render_result = {
                    "status": "dispatched",
                    "jobId": job_id,
                    "job_id": job_id,
                    "match": f"{home_team} vs {away_team}",
                    "match_id": match_id,
                    "totalDuration": payload["durationInSeconds"],
                    "segmentsCount": len(payload["segments"]),
                    "events_analyzed": len(synchronized_clips),
                    "videoFile": str(actual_video_file) if actual_video_file else None,
                    "synchronized_clips": synchronized_clips,
                }
                if wait:
                    render_res = self.poll_render(job_id)
                    render_status = render_res.get("status", "done")
                    render_result["status"] = render_status
                    render_result["outputPath"] = render_res.get("outputPath")
                    render_result["video_url"] = f"http://localhost:3008/video/{job_id}"
                    if render_status != "done":
                        error_msg = render_res.get("error", f"Render failed with status: {render_status}")
                        raise RuntimeError(f"Remotion render job {job_id} failed: {error_msg}")

                    host_mp4 = self.output_dir / f"{job_id}.mp4"
                    master_mp4 = self.output_dir / f"{match_id}_masterclass_final.mp4"
                    if host_mp4.is_file():
                        import shutil
                        shutil.copy2(str(host_mp4), str(master_mp4))
                        render_result["host_output_path"] = str(host_mp4)
                        render_result["master_output_path"] = str(master_mp4)
                        logger.info(f"Master render copied to {master_mp4} ({master_mp4.stat().st_size / (1024*1024):.2f} MB)")

                render_path = match_checkpoint_dir / "remotion_render.json"
                self.state_tracker.record_checkpoint(
                    match_id=match_id,
                    clip_id="__match__",
                    stage=STAGE_REMOTION_RENDERED,
                    status="COMPLETED",
                    json_payload_path=render_path,
                    payload=render_result,
                )
                for sc in synchronized_clips:
                    self.state_tracker.record_checkpoint(
                        match_id=match_id,
                        clip_id=sc["clip_id"],
                        stage=STAGE_REMOTION_RENDERED,
                        status="COMPLETED",
                        json_payload_path=render_path,
                        payload=render_result,
                    )
                return render_result
            except Exception as e:
                logger.error(f"Remotion render dispatch failed: {e}", exc_info=True)
                self.state_tracker.record_checkpoint(
                    match_id=match_id,
                    clip_id="__match__",
                    stage=STAGE_REMOTION_RENDERED,
                    status="FAILED",
                )
                raise

        return {
            "status": "prepared",
            "payload": payload,
            "match_id": match_id,
            "synchronized_clips": synchronized_clips,
        }

    @staticmethod
    def _realign_cause_telestration(
        cause_tel: Dict[str, Any],
        tracks_per_frame: Dict[str, Any],
        freeze_frame: int,
        clip_id: str = "",
    ) -> None:
        """
        Re-aligns cause_telestration player coordinates to tracks_per_frame[str(freeze_frame)].
        Delegates authoritatively to TacticalVisionEngine.sanitize_v3_telestration to avoid team swaps and polygon corruption.
        """
        if not isinstance(cause_tel, dict) or not isinstance(tracks_per_frame, dict):
            return

        f_key = str(freeze_frame)
        tracks_at_freeze = tracks_per_frame.get(f_key)
        if not tracks_at_freeze:
            int_keys = [int(k) for k in tracks_per_frame.keys() if str(k).isdigit()]
            if int_keys:
                nearest = min(int_keys, key=lambda k: abs(k - freeze_frame))
                tracks_at_freeze = tracks_per_frame.get(str(nearest), [])
                tracks_per_frame[f_key] = tracks_at_freeze

        if tracks_at_freeze is None:
            tracks_at_freeze = []
            tracks_per_frame[f_key] = tracks_at_freeze

        cid = str(clip_id or cause_tel.get("clip_id") or "").lower()

        f_img = None
        f_paths = [
            cause_tel.get("freeze_frame_path"),
            cause_tel.get("cause_frame_path"),
            f"F:/DockerRenders/assets/frames/{clip_id}_cause.jpg",
            f"F:/DockerRenders/assets/frames/{clip_id}_freeze.jpg",
        ]
        for fp in f_paths:
            if fp and Path(str(fp)).is_file():
                try:
                    f_img = cv2.imread(str(fp))
                    if f_img is not None and f_img.size > 0:
                        break
                except Exception:
                    pass

        homographies = None
        f_next_paths = [
            f"F:/DockerRenders/assets/frames/{clip_id}_frame_next_2.jpg",
            f"F:/DockerRenders/assets/frames/{clip_id}_frame_next_1.jpg",
            f"F:/DockerRenders/assets/frames/{clip_id}_effect.jpg",
        ]
        if f_img is not None:
            for fnp in f_next_paths:
                if fnp and Path(fnp).is_file():
                    try:
                        f_next = cv2.imread(fnp)
                        if f_next is not None and f_next.size > 0:
                            from src.vision.camera_pan_compensator import CameraPanCompensator
                            cpc = CameraPanCompensator()
                            H, _ = cpc.estimate_homography(f_next, f_img)
                            homographies = {"frame_next_2": H}
                            break
                    except Exception:
                        pass

        TacticalVisionEngine.sanitize_v3_telestration(
            cause_tel=cause_tel,
            tracks_at_freeze=tracks_at_freeze,
            clip_id=clip_id,
            freeze_frame_img=f_img,
            event_context={
                "home_team": cause_tel.get("home_team") or "Barcelona",
                "away_team": cause_tel.get("away_team") or "Real Madrid",
                "match_minute": cause_tel.get("match_minute", 0),
                "period": cause_tel.get("period", 1),
            },
            tracks_per_frame=tracks_per_frame,
            freeze_frame=freeze_frame,
            homographies=homographies,
        )
        tracks_per_frame[f_key] = tracks_at_freeze

    def _dispatch_render(self, payload: Dict[str, Any]) -> str:
        url = f"{self.remotion_host}/render"

        def _json_serial(obj):
            if hasattr(obj, "tolist"):
                return obj.tolist()
            try:
                import numpy as np
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
            except ImportError:
                pass
            return str(obj)

        req = urllib.request.Request(
            url,
            data=json.dumps(payload, default=_json_serial).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("jobId", "unknown")
        except Exception as e:
            logger.error(f"Failed to dispatch render to Remotion server: {e}")
            raise

    def poll_render(self, job_id: str, timeout_sec: int = 5400, poll_interval: int = 5) -> Dict[str, Any]:
        url = f"{self.remotion_host}/status/{job_id}"
        start_time = time.time()
        last_progress = -1
        while time.time() - start_time < timeout_sec:
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    status = data.get("status")
                    progress = data.get("progress", 0)
                    if progress != last_progress:
                        logger.info(f"Render progress [{job_id}]: {progress}% (status: {status})")
                        last_progress = progress
                    if status == "done":
                        logger.info(f"Render complete! Output: {data.get('outputPath')}")
                        return data
                    if status == "error":
                        logger.error(f"Render job failed: {data.get('error')}")
                        return data
            except Exception as e:
                logger.warning(f"Status poll warning: {e}")
            time.sleep(poll_interval)
        return {"status": "timeout", "jobId": job_id}

    def start_folder_watcher(self):
        """Watches F:\\DockerRenders\\incoming_matches for zero-click automation."""
        def on_new_video(file_path: Path):
            home, away, score = ReplayIngestor.parse_match_metadata_from_filename(file_path.name)
            logger.info(f"Triggering automatic masterclass generation for: {home} vs {away}")
            self.run(
                video_path=str(file_path),
                home_team=home,
                away_team=away,
                score=score,
                render=True,
                duration=300,
            )

        self.ingestor.watch_incoming_folder(callback_fn=on_new_video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Tactical Masterclass Pipeline")
    parser.add_argument("--video", type=str, help="Local match video file (.mp4/.mkv)")
    parser.add_argument("--download-url", type=str, help="Full match replay URL to download via yt-dlp")
    parser.add_argument("--watch", action="store_true", help="Start watched folder daemon for F:\\DockerRenders\\incoming_matches")
    parser.add_argument("--home", type=str, default=None, help="Home Team (auto-parsed from video filename if omitted)")
    parser.add_argument("--away", type=str, default=None, help="Away Team (auto-parsed from video filename if omitted)")
    parser.add_argument("--score", type=str, default=None, help="Match Score (auto-parsed if omitted)")
    parser.add_argument("--duration", type=float, default=600.0, help="Duration in Seconds (default: 600s = 10 min)")
    parser.add_argument("--wait", action="store_true", help="Wait for render completion and poll progress")
    parser.add_argument("--no-render", action="store_true", help="Prepare payload without rendering")
    parser.add_argument("--fixture-id", type=str, help="Unique fixture or match identifier")
    parser.add_argument("--db-path", type=str, default=None, help="Custom SQLite pipeline_state.db path")
    parser.add_argument("--force-refresh", action="store_true", help="Force re-run stages ignoring cached state")
    parser.add_argument("--overrides", type=str, default=None, help="JSON string or path to JSON player identity overrides (e.g. '{\"T18\": 9}')")
    args = parser.parse_args()

    # Automatically extract teams and score if --video is provided without explicit teams
    parsed_home = args.home
    parsed_away = args.away
    parsed_score = args.score
    if args.video and Path(args.video).is_file():
        meta = ReplayIngestor.parse_match_metadata_from_filename(Path(args.video).name)
        parsed_home = parsed_home or meta.home
        parsed_away = parsed_away or meta.away
        parsed_score = parsed_score or meta.score

    runner = TacticalPipelineRunner(db_path=args.db_path)
    if args.watch:
        runner.start_folder_watcher()
    else:
        result = runner.run(
            video_path=args.video,
            download_url=args.download_url,
            fixture_id=args.fixture_id,
            home_team=parsed_home or "Barcelona",
            away_team=parsed_away or "Real Madrid",
            score=parsed_score or "FT",
            duration=args.duration,
            render=not args.no_render,
            wait=args.wait,
            force_refresh=args.force_refresh,
            overrides=args.overrides,
        )
        print(json.dumps(result, indent=2))
