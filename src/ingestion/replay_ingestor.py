"""
src/ingestion/replay_ingestor.py
Automated Full Match Replay Ingestion Engine.

Features:
1. Direct high-speed 1080p download of full match replays via yt-dlp.
2. Watched folder daemon for automated zero-click ingestion (F:\\DockerRenders\\incoming_matches).
3. Automatic handoff to ClipExtractor and ScoreboardClockOCR.
"""

import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger("ingestion.replay_ingestor")

from .fullmatch_scraper import FullMatchWebScraper
from .telegram_scraper import TelegramReplayScraper

DEFAULT_REPLAY_DIR = Path(r"F:\DockerRenders\replays")
DEFAULT_INCOMING_DIR = Path(r"F:\DockerRenders\incoming_matches")


class ReplayIngestor:
    def __init__(
        self,
        replays_dir: Path = DEFAULT_REPLAY_DIR,
        incoming_dir: Path = DEFAULT_INCOMING_DIR,
    ):
        self.replays_dir = Path(replays_dir)
        self.incoming_dir = Path(incoming_dir)
        self.replays_dir.mkdir(parents=True, exist_ok=True)
        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        self.telegram = TelegramReplayScraper(output_dir=self.incoming_dir)
        self.web_scraper = FullMatchWebScraper()

    def find_local_match_video(self, home_team: str, away_team: str) -> Optional[Path]:
        r"""
        [Tier 0] Searches local disk for pre-existing or extracted match video files.
        Checks incoming_matches, F:\Downloads (including subdirectories), and replays.
        """
        search_dirs = [
            self.incoming_dir,
            Path(r"F:\Downloads"),
            self.replays_dir,
        ]
        terms = [t.lower() for t in (home_team, away_team) if len(t) > 2]
        abbrevs = [t[:3].lower() for t in (home_team, away_team) if len(t) >= 3]

        for sdir in search_dirs:
            if not sdir.exists():
                continue
            file_list = list(sdir.glob("*.*")) + list(sdir.glob("*/*.*")) + list(sdir.glob("*/*/*.*"))
            for f in file_list:
                if f.suffix.lower() in (".mp4", ".mkv", ".ts", ".avi", ".mov"):
                    fname = f.name.lower()
                    if any(t in fname for t in terms) or (len(abbrevs) >= 2 and all(a in fname for a in abbrevs)):
                        logger.info(f"[Tier 0] Found local broadcast match video on disk: {f}")
                        return f
        return None

    def download_youtube_broadcast(self, home_team: str, away_team: str) -> Optional[Path]:
        """
        [Tier 3] Searches YouTube broadcast archives for official extended 1080p highlights/replays.
        """
        query = f"{home_team} vs {away_team} match highlights"
        logger.info(f"[Tier 3] Searching YouTube 1080p broadcast feeds for '{query}'...")
        cmd = [
            "yt-dlp",
            "--print", "%(id)s\t%(duration)s\t%(title)s",
            f"ytsearch4:{query}"
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            candidates = []
            for line in res.stdout.strip().split("\n"):
                if not line.strip() or "\t" not in line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    vid_id = parts[0].strip()
                    try:
                        dur = float(parts[1].strip())
                    except ValueError:
                        dur = 0
                    title = parts[2].strip()
                    t_low = title.lower()
                    h_ok = any(w in t_low for w in home_team.lower().split() if len(w) > 2)
                    a_ok = any(w in t_low for w in away_team.lower().split() if len(w) > 2)
                    if h_ok and a_ok and dur >= 120:
                        candidates.append((dur, vid_id, title))

            if not candidates:
                logger.info(f"No matching YouTube broadcast footage found for '{query}'.")
                return None

            # Sort by longest duration for fullest match coverage
            candidates.sort(key=lambda c: c[0], reverse=True)
            best_dur, best_id, best_title = candidates[0]
            logger.info(f"Selected best broadcast footage: '{best_title}' ({best_dur//60:.0f}m{best_dur%60:.0f}s). Downloading in 1080p...")

            yt_url = f"https://www.youtube.com/watch?v={best_id}"
            output_name = f"{home_team}_vs_{away_team}".replace(" ", "_")
            return self.download_replay(video_url=yt_url, output_name=output_name)
        except Exception as e:
            logger.warn(f"YouTube broadcast search/download failed: {e}")
            return None

    def auto_find_and_download_match(self, home_team: str, away_team: str) -> Optional[Path]:
        r"""
        Automated multi-tier fallback cascade:
        Tier 0: Local disk search (incoming_matches, F:\Downloads, replays).
        Tier 1: Telegram channels (direct CDN MP4 or supported stream hosts).
        Tier 2: Web Replay Aggregators (FullMatchSports & FullMatchShows).
        Tier 3: YouTube Official 1080p broadcast feeds.
        Tier 4: Fail Hard (returns None -> strict fail-hard zero-placeholder policy).
        """
        query = f"{home_team} vs {away_team}"

        # Tier 0: Check local storage first
        local_vid = self.find_local_match_video(home_team, away_team)
        if local_vid:
            return local_vid

        # Tier 1: Telegram Channels
        logger.info(f"[Tier 1] Checking Telegram channels for '{query}'...")
        tele_path = self.telegram.search_and_download_sync(query)
        if tele_path and tele_path.exists():
            logger.info(f"Successfully obtained match video from Telegram: {tele_path}")
            return tele_path

        # Tier 2: Web Replay Aggregators
        logger.info(f"[Tier 2] Telegram not available or match not found. Scraping FullMatchSports for '{query}'...")
        page_url = self.web_scraper.search_match_replay(home_team, away_team)
        if page_url:
            links = self.web_scraper.extract_download_links_from_page(page_url)
            for link_obj in links:
                try:
                    logger.info(f"Attempting download via {link_obj['host']} ({link_obj['url']})...")
                    dest_file = self.download_replay(
                        video_url=link_obj['url'],
                        output_name=f"{home_team}_vs_{away_team}".replace(" ", "_"),
                    )
                    if dest_file and dest_file.exists():
                        return dest_file
                except Exception as dl_err:
                    logger.warn(f"Download attempt failed for {link_obj['host']}: {dl_err}")

        # Tier 3: YouTube Broadcast Archive
        logger.info(f"[Tier 3] Checking YouTube broadcast feeds for '{query}'...")
        yt_vid = self.download_youtube_broadcast(home_team, away_team)
        if yt_vid and yt_vid.exists():
            return yt_vid

        logger.warn(f"Auto-finder exhausted all tiers for '{query}'. No broadcast footage found.")
        return None

    def download_replay(
        self,
        video_url: str,
        output_name: Optional[str] = None,
        max_height: int = 1080,
    ) -> Path:
        """
        Downloads a full match replay or clip using yt-dlp in up to 1080p.
        Supports YouTube, OK.ru, Streamwish, Dailymotion, direct MP4/m3u8, etc.
        """
        if not output_name:
            clean_timestamp = int(time.time())
            output_name = f"match_replay_{clean_timestamp}"

        output_template = str(self.replays_dir / f"{output_name}.%(ext)s")
        format_selector = f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best"

        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--retries", "5",
            "--fragment-retries", "5",
            "-f", format_selector,
            "--merge-output-format", "mp4",
            "-o", output_template,
            video_url,
        ]

        logger.info(f"Downloading replay via yt-dlp: {video_url} -> {output_template}")
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Download completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"yt-dlp download failed: {e.stderr}")
            raise RuntimeError(f"Replay download failed: {e.stderr}")

        # Find the output mp4
        candidates = list(self.replays_dir.glob(f"{output_name}.*"))
        if not candidates:
            raise FileNotFoundError(f"Expected download file {output_template} was not found.")
        return candidates[0]

    def watch_incoming_folder(self, callback_fn, poll_interval_sec: int = 5):
        """
        Continuously monitors F:\\DockerRenders\\incoming_matches.
        When a new video appears and finishes copying, triggers the analysis pipeline.
        """
        logger.info(f"Watching for incoming match video files in {self.incoming_dir}...")
        processed_files = set()

        while True:
            try:
                for file_path in self.incoming_dir.glob("*.*"):
                    if file_path.suffix.lower() not in (".mp4", ".mkv", ".ts", ".avi", ".mov"):
                        continue
                    if file_path in processed_files:
                        continue

                    # Ensure file is completely written (stable size for 3 seconds)
                    size1 = file_path.stat().st_size
                    time.sleep(3)
                    size2 = file_path.stat().st_size
                    if size1 != size2 or size1 == 0:
                        continue

                    logger.info(f"New match video detected: {file_path.name} ({size2 / 1024 / 1024:.1f} MB)")
                    processed_files.add(file_path)

                    # Trigger processing
                    try:
                        callback_fn(file_path)
                    except Exception as err:
                        logger.error(f"Error processing {file_path.name}: {err}")

                time.sleep(poll_interval_sec)
            except KeyboardInterrupt:
                logger.info("Folder watch stopped.")
                break
            except Exception as e:
                logger.error(f"Folder watch error: {e}")
                time.sleep(poll_interval_sec)

    @staticmethod
    def parse_match_metadata_from_filename(filename: str) -> Tuple[str, str, str]:
        """
        Extracts home team, away team, and score/date from filename.
        e.g. 'Barcelona_vs_Real_Madrid_3-2.mp4' -> ('Barcelona', 'Real Madrid', '3 - 2')
        """
        clean_name = Path(filename).stem.replace("_", " ").replace("-", " ")
        vs_match = re.search(r"([A-Za-z\s]+)\s+(?:vs|v)\s+([A-Za-z\s]+)", clean_name, re.I)
        if vs_match:
            home = vs_match.group(1).strip()
            away = vs_match.group(2).strip()
            return home, away, "FT"
        return "HOME TEAM", "AWAY TEAM", "FT"
