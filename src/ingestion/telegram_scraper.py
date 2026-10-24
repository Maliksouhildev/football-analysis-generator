"""
src/ingestion/telegram_scraper.py
Automated High-Speed Telegram Football Replay Scraper & Downloader.

Dual Architecture:
1. Public Web Scraper (Zero Auth / Zero Key): Scrapes live public Telegram channels (e.g. @replaysfootball,
   @footballreplays, @footballmatchreplay) for instant post-whistle match uploads and direct CDN/Storage links.
2. Telethon Native Client: If TELEGRAM_API_ID & TELEGRAM_API_HASH are configured in environment,
   authenticates via MTProto to search private channels, VIP replay groups, and download 1080p 50fps broadcast feeds.
"""

import asyncio
import logging
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("ingestion.telegram_scraper")

DEFAULT_INCOMING_DIR = Path(r"F:\DockerRenders\incoming_matches")

# Verified active Telegram channels updated immediately after match whistle
DEFAULT_CHANNELS = [
    "replaysfootball",       # FullMatch-Replay instant feed (UCL, EPL, La Liga, Serie A)
    "footballreplays",        # Direct Telegram CDN MP4 video uploads
    "footballmatchreplay",    # Full match replays & high-speed links
    "fullmatchfootball",      # Full 90-minute archives
]


class TelegramReplayScraper:
    def __init__(
        self,
        api_id: Optional[int] = None,
        api_hash: Optional[str] = None,
        session_name: str = "football_replay_bot",
        output_dir: Path = DEFAULT_INCOMING_DIR,
    ):
        self.api_id = api_id or os.environ.get("TELEGRAM_API_ID")
        self.api_hash = api_hash or os.environ.get("TELEGRAM_API_HASH")
        self.session_name = session_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = None

        # Custom channels from environment
        env_channels = os.environ.get("TELEGRAM_CHANNELS")
        if env_channels:
            custom_list = [c.strip().lstrip("@") for c in env_channels.split(",") if c.strip()]
            self.channels = custom_list + [c for c in DEFAULT_CHANNELS if c not in custom_list]
        else:
            self.channels = DEFAULT_CHANNELS

    def is_telethon_configured(self) -> bool:
        """Returns True if Telegram API credentials are provided for Telethon client."""
        return bool(self.api_id and self.api_hash)

    # ---------------------------------------------------------
    # TIER 1: PUBLIC TELEGRAM WEB SCRAPING (Zero API Key Needed)
    # ---------------------------------------------------------
    def search_public_channels(self, query: str, dry_run: bool = False) -> Optional[Path]:
        """
        Scrapes public Telegram channel web previews (t.me/s/{channel}) for the match.
        Extracts direct video files (telesco.pe CDN) or direct high-speed cloud mirrors.
        If dry_run is True, returns resolved URL as Path without downloading.
        """
        query_terms = [t.lower() for t in re.split(r"\s+(?:vs|v|-)\s+|\s+", query) if len(t) > 2]
        if not query_terms:
            logger.warn(f"Invalid search query: '{query}'")
            return None

        logger.info(f"[Telegram Public Search] Scanning channels {self.channels} for '{query}'...")

        for ch in self.channels:
            url = f"https://t.me/s/{ch}"
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")

                # Extract individual message blocks
                blocks = re.findall(
                    r'<div class="tgme_widget_message_wrap[^"]*">(.*?)</div>\s*</div>\s*</div>',
                    html,
                    re.DOTALL
                )

                for block in blocks:
                    # Clean text and check query match
                    text_m = re.search(r'class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', block, re.DOTALL)
                    text = text_m.group(1) if text_m else ""
                    clean_text = ' '.join(re.sub(r'<[^>]+>', ' ', text).split()).lower()

                    if all(term in clean_text for term in query_terms):
                        logger.info(f"Match found in @{ch}: '{clean_text[:80]}'")

                        # Option A: Direct Telegram video embedded in post
                        vid_m = re.search(r'<video\s+src="([^"]+)"', block)
                        if vid_m:
                            direct_vid = vid_m.group(1)
                            logger.info(f"Direct Telegram CDN video discovered: {direct_vid[:60]}...")
                            if dry_run:
                                return direct_vid
                            return self._download_direct_stream(direct_vid, query)

                        # Option B: External high-speed storage or replay mirror
                        links = re.findall(r'href="([^"]+)"', block)
                        ext_links = [l for l in links if "t.me" not in l and not l.startswith("#")]
                        for link in ext_links:
                            resolved = self._resolve_external_replay_link(link)
                            if resolved:
                                logger.info(f"Discovered replay link from @{ch}: {resolved}")
                                if dry_run:
                                    return resolved
                                return self._download_file_or_stream(resolved, query)

            except Exception as e:
                logger.warn(f"Failed scanning public Telegram channel @{ch}: {e}")

        logger.info(f"No match replay found across public Telegram channels for '{query}'.")
        return None

    def _resolve_external_replay_link(self, link: str) -> Optional[str]:
        """Resolves replay hub landing page to direct fast download link."""
        try:
            if "fullmatch-replay.org" in link:
                req = urllib.request.Request(link, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=7) as resp:
                    m_html = resp.read().decode("utf-8", errors="ignore")

                # Extract all <a> tags with inner text
                all_anchors = re.findall(r'<a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a>', m_html, re.DOTALL)
                server_links = []
                SUPPORTED_HOSTS = ["streamwish", "filelions", "voe.sx", "doodstream", "ok.ru", "telesco.pe"]
                BLOCKED_HOSTS = ["bysefujedu", "vixeo", "fbetlixgee", "bysekoze", "playmogo", "playmate", "terabox"]

                for href, inner in all_anchors:
                    if "/category/" in href or "/tag/" in href or href.startswith("#"):
                        continue
                    if any(b in href.lower() for b in BLOCKED_HOSTS):
                        continue
                    if any(s in href.lower() for s in SUPPORTED_HOSTS) or any(href.lower().endswith(ext) for ext in [".mp4", ".m3u8", ".ts"]):
                        server_links.append(href)

                if server_links:
                    return server_links[0]
            elif any(k in link.lower() for k in ["streamwish", "filelions", "ok.ru", "telesco.pe"]):
                return link
        except Exception as err:
            logger.warn(f"Error resolving external replay link {link}: {err}")
        return None

    def _download_direct_stream(self, video_url: str, query: str) -> Path:
        """Streams direct MP4 video from Telegram CDN (telesco.pe) to disk."""
        safe_name = re.sub(r"[^\w\-_.]", "_", query) + ".mp4"
        dest_path = self.output_dir / safe_name

        logger.info(f"Streaming Telegram CDN video to: {dest_path}...")
        req = urllib.request.Request(video_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as out_file:
            total_bytes = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = resp.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_bytes > 0:
                    pct = (downloaded / total_bytes) * 100
                    if downloaded % (10 * 1024 * 1024) < 1024 * 1024:
                        logger.info(f"Telegram Download Progress: {pct:.1f}% ({downloaded / 1024 / 1024:.1f} MB)")

        logger.info(f"Telegram direct stream download complete: {dest_path}")
        return dest_path

    def _download_file_or_stream(self, url: str, query: str) -> Path:
        """Downloads file via direct HTTP or yt-dlp fallback."""
        safe_name = re.sub(r"[^\w\-_.]", "_", query)
        dest_path = self.output_dir / f"{safe_name}.mp4"

        # Check if direct fast-download HTTP link
        if "fast-download" in url or url.endswith(".mp4") or "cdn" in url:
            try:
                logger.info(f"Downloading high-speed direct file from {url}...")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as out_file:
                    downloaded = 0
                    while True:
                        chunk = resp.read(2 * 1024 * 1024)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (20 * 1024 * 1024) < 2 * 1024 * 1024:
                            logger.info(f"Downloaded {downloaded / 1024 / 1024:.1f} MB...")
                logger.info(f"Direct file download complete: {dest_path}")
                return dest_path
            except Exception as dl_err:
                logger.warn(f"Direct stream download failed, falling back to yt-dlp: {dl_err}")

        # Fallback to yt-dlp
        output_template = str(self.output_dir / f"{safe_name}.%(ext)s")
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--retries", "5",
            "--fragment-retries", "5",
            "-o", output_template,
            url,
        ]
        logger.info(f"Invoking yt-dlp for Telegram-extracted link: {url}")
        subprocess.run(cmd, check=True, capture_output=True)

        candidates = list(self.output_dir.glob(f"{safe_name}.*"))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"Failed to find downloaded replay for {query}")

    # ---------------------------------------------------------
    # TIER 2: TELETHON CLIENT (Authenticated Private Channels)
    # ---------------------------------------------------------
    async def _init_telethon_client(self):
        if not self.is_telethon_configured():
            return
        try:
            from telethon import TelegramClient
            self.client = TelegramClient(self.session_name, int(self.api_id), str(self.api_hash))
            await self.client.start()
        except Exception as e:
            logger.error(f"Failed to start Telethon client: {e}")
            self.client = None

    async def search_telethon(self, query: str) -> Optional[Path]:
        """Searches user's private Telegram channels and dialogs using Telethon."""
        if not self.is_telethon_configured():
            return None

        if not self.client:
            await self._init_telethon_client()
        if not self.client:
            return None

        query_terms = [t.lower() for t in re.split(r"\s+(?:vs|v|-)\s+|\s+", query) if len(t) > 2]
        logger.info(f"[Telethon MTProto Search] Searching dialogs for '{query}'...")

        try:
            async for dialog in self.client.iter_dialogs():
                if dialog.is_channel or dialog.is_group:
                    async for message in self.client.iter_messages(dialog, limit=40):
                        text = (message.text or message.caption or "").lower()
                        if all(term in text for term in query_terms):
                            if message.video or (message.document and "video" in getattr(message.document, "mime_type", "")):
                                safe_name = re.sub(r"[^\w\-_.]", "_", query) + ".mp4"
                                target_path = self.output_dir / safe_name
                                logger.info(f"Found match in Telethon dialog '{dialog.name}'. Downloading media to {target_path}...")
                                await message.download_media(file=str(target_path))
                                return target_path
        except Exception as e:
            logger.error(f"Telethon search encountered error: {e}")

        return None

    # ---------------------------------------------------------
    # MASTER ENTRYPOINT: Automatic Telegram Cascade
    # ---------------------------------------------------------
    def search_and_download(self, query: str, dry_run: bool = False) -> Optional[Path]:
        """
        Master search method:
        1. Checks public Telegram channels (zero configuration required).
        2. If not found and Telethon is configured, checks private Telethon dialogs.
        """
        # Tier 1: Public Telegram Channels
        res = self.search_public_channels(query, dry_run=dry_run)
        if res and (dry_run or res.exists()):
            return res

        # Tier 2: Telethon Client
        if self.is_telethon_configured() and not dry_run:
            try:
                res_tele = asyncio.run(self.search_telethon(query))
                if res_tele and res_tele.exists():
                    return res_tele
            except Exception as e:
                logger.warn(f"Telethon execution error: {e}")

        return None

    def search_and_download_sync(self, query: str, dry_run: bool = False) -> Optional[Path]:
        """Synchronous wrapper for search_and_download."""
        return self.search_and_download(query, dry_run=dry_run)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Telegram Football Replay Scraper")
    parser.add_argument("--query", required=True, help="Match query, e.g. 'Galatasaray vs Barcelona'")
    parser.add_argument("--outdir", default=str(DEFAULT_INCOMING_DIR), help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Search and resolve link without downloading")
    args = parser.parse_args()

    scraper = TelegramReplayScraper(output_dir=Path(args.outdir))
    result = scraper.search_and_download(args.query, dry_run=args.dry_run)
    if result:
        print(f"\n[SUCCESS] Replay resolved/downloaded: {result}")
    else:
        print(f"\n[FAILED] Could not locate replay for '{args.query}' on Telegram.")
        sys.exit(1)
