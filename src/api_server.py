"""
src/api_server.py
Tactical Masterclass Fast HTTP API Server.
Connects n8n workflow container (via host.docker.internal:5055) to the Python Tactical Pipeline.
"""

import json
import logging
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

from flask import Flask, request, jsonify

from src.pipeline_runner import TacticalPipelineRunner
from src.ingestion.telegram_scraper import TelegramReplayScraper
from src.scraper.lineup_scraper import fetch_live_match_data, MatchLineupResolver
from src.vision.match_roster_engine import ActiveXIResolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TacticalApiServer")

app = Flask("TacticalMasterclassAPI")
runner = TacticalPipelineRunner()
tg_scraper = TelegramReplayScraper()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "Tactical Masterclass Pipeline API",
        "remotion_host": runner.remotion_host,
        "output_dir": str(runner.output_dir),
        "db_path": str(runner.db_path),
    })


@app.route("/api/step1_lineup", methods=["POST", "GET"])
def step1_lineup():
    """
    Step 1: Universal Lineup & Matchday Ingestion Endpoint.
    Payload:
    {
        "match_url": "https://www.fotmob.com/... or https://www.sofascore.com/...",
        "video_path": "..." (optional)
    }
    Dynamically scrapes starting XI, tactical formations, substitutions, and red cards.
    Initializes ActiveXIResolver with zero hardcoded rosters.
    """
    data = request.get_json(force=True, silent=True) or request.args.to_dict() or {}
    match_url = data.get("match_url") or data.get("url")
    video_path = data.get("video_path")

    if not match_url:
        return jsonify({
            "status": "error",
            "error": "Missing mandatory parameter: 'match_url' (FotMob or Sofascore fixture URL required)."
        }), 400

    logger.info(f"Scraping live match data for match_url: {match_url}")

    try:
        match_data = fetch_live_match_data(match_url)
        MatchLineupResolver.register_matchday_data(match_data)
        resolver = ActiveXIResolver(match_data=match_data)
        initial_roster = resolver.get_active_players(match_minute=0)

        home_cfg = match_data.get("home_team", {})
        away_cfg = match_data.get("away_team", {})

        return jsonify({
            "status": "success",
            "match_url": match_url,
            "video_path": video_path,
            "match_id": match_data.get("match_id", "live_fixture"),
            "home_team": home_cfg.get("name", "Home Team"),
            "away_team": away_cfg.get("name", "Away Team"),
            "home_formation": home_cfg.get("formation", "4-3-3"),
            "away_formation": away_cfg.get("formation", "4-3-3"),
            "home_starters_count": len(home_cfg.get("starting_xi", [])),
            "away_starters_count": len(away_cfg.get("starting_xi", [])),
            "substitutions_count": len(match_data.get("substitutions", [])),
            "incidents_count": len(match_data.get("incidents", [])),
            "active_on_pitch_starters": initial_roster.active_on_pitch_players,
            "whitelist_enum": initial_roster.whitelist_enum,
            "match_data": match_data,
        })
    except Exception as e:
        logger.error(f"Live match scraping failed for {match_url}: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "match_url": match_url
        }), 500


@app.route("/pipeline_state/<match_id>", methods=["GET"])
def pipeline_state(match_id):
    checkpoints = runner.state_tracker.get_match_state(match_id)
    return jsonify({
        "status": "success",
        "match_id": match_id,
        "count": len(checkpoints),
        "checkpoints": checkpoints,
    })


@app.route("/pipeline_state/<match_id>/reset", methods=["POST"])
def reset_pipeline_state(match_id):
    runner.state_tracker.reset_checkpoint(match_id)
    return jsonify({
        "status": "success",
        "message": f"Reset checkpoints for match '{match_id}'",
        "match_id": match_id,
    })


@app.route("/search_telegram", methods=["POST", "GET"])
def search_telegram():
    data = request.get_json(force=True, silent=True) or request.args.to_dict() or {}
    query = data.get("query")
    home = data.get("home_team", "")
    away = data.get("away_team", "")
    if not query and home and away:
        query = f"{home} vs {away}"
    elif not query:
        query = "football match"

    logger.info(f"Received Telegram search request: '{query}'")

    # Check local disk first
    if home and away:
        local_vid = runner.ingestor.find_local_match_video(home, away)
        if local_vid:
            return jsonify({
                "status": "found",
                "query": query,
                "resolved_url": "",
                "video_path": str(local_vid),
                "message": f"Local broadcast footage found on disk: {local_vid.name}"
            })

    resolved = tg_scraper.search_public_channels(query, dry_run=True)
    if resolved:
        clean_url = str(resolved).replace("\\", "/")
        if clean_url.startswith("https:/") and not clean_url.startswith("https://"):
            clean_url = clean_url.replace("https:/", "https://")
        elif clean_url.startswith("http:/") and not clean_url.startswith("http://"):
            clean_url = clean_url.replace("http:/", "http://")
        return jsonify({
            "status": "found",
            "query": query,
            "resolved_url": clean_url,
            "channel": tg_scraper.channels
        })
    return jsonify({
        "status": "not_found",
        "query": query,
        "resolved_url": "",
        "message": "Match not yet available on public Telegram channels."
    }), 200


@app.route("/analyze", methods=["POST"])
def analyze_match():
    """
    Master tactical analysis and telestration endpoint called by n8n.
    Payload:
    {
        "match_url": "https://www.fotmob.com/... or https://www.sofascore.com/...", (optional/recommended)
        "video_path": "F:\\DockerRenders\\incoming_matches\\...", (optional)
        "home_team": "...", (optional, auto-derived from match_url)
        "away_team": "...", (optional, auto-derived from match_url)
        "duration": 540,
        "download_url": "..." (optional),
        "fixture_id": "...", (optional)
        "match_id": "...", (optional)
        "force_refresh": false (optional)
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    match_url = data.get("match_url") or data.get("url")
    video_path = data.get("video_path")
    download_url = data.get("download_url")
    if download_url:
        download_url = download_url.replace("\\", "/")

    home_team = data.get("home_team")
    away_team = data.get("away_team")
    score = data.get("score", "FT")
    duration = float(data.get("duration", 540.0))
    fixture_id = data.get("fixture_id") or data.get("match_id")
    force_refresh = bool(data.get("force_refresh", False))

    if match_url:
        logger.info(f"Dynamically scraping live match data for match_url: {match_url}")
        try:
            match_data = fetch_live_match_data(match_url)
            MatchLineupResolver.register_matchday_data(match_data)
            if not home_team:
                home_team = match_data.get("home_team", {}).get("name")
            if not away_team:
                away_team = match_data.get("away_team", {}).get("name")
            if not fixture_id:
                fixture_id = match_data.get("match_id")
        except Exception as e:
            logger.warning(f"Could not scrape match_url '{match_url}': {e}. Continuing with provided parameters.")

    home_team = home_team or "Home"
    away_team = away_team or "Away"

    logger.info(f"n8n triggered Masterclass: {home_team} vs {away_team} (duration: {duration}s)")

    try:
        result = runner.run(
            video_path=video_path,
            download_url=download_url,
            fixture_id=fixture_id,
            home_team=home_team,
            away_team=away_team,
            score=score,
            render=True,
            duration=duration,
            force_refresh=force_refresh,
        )
        job_id = result.get("job_id") or result.get("jobId")
        video_file = result.get("video_file") or result.get("videoFile")
        video_url = result.get("video_url") or (f"http://localhost:3008/video/{job_id}" if job_id else None)
        events_analyzed = result.get("events_analyzed") or result.get("segmentsCount") or len(result.get("events", []))
        match_id = result.get("match_id") or fixture_id or f"{home_team}_vs_{away_team}".lower().replace(" ", "_")
        return jsonify({
            "status": "success",
            "match": f"{home_team} vs {away_team}",
            "match_id": match_id,
            "job_id": job_id,
            "video_url": video_url,
            "video_file": str(video_file) if video_file else None,
            "events_analyzed": events_analyzed,
            "composition": "TacticalMasterclass",
            "resolution": "1920x1080 @ 30fps"
        })
    except FileNotFoundError as e:
        logger.warning(f"No match footage found: {e}")
        return jsonify({
            "status": "no_footage_found",
            "error": str(e),
            "match": f"{home_team} vs {away_team}",
            "rendered": False
        }), 422
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "match": f"{home_team} vs {away_team}",
            "rendered": False
        }), 500


if __name__ == "__main__":
    logger.info("Starting Tactical Masterclass API on 0.0.0.0:5055...")
    app.run(host="0.0.0.0", port=5055, threaded=True)
