"""
src/rendering/player_asset_manager.py
Player Portrait & Cutout Asset Manager for Remotion Broadcast Overlays.

Resolves player names to high-resolution official portraits & cutouts:
1. Checks local cache (F:\\DockerRenders\\assets\\players\\{slug}.png).
2. Upgrades any legacy low-res/placeholder dummy badges (< 8KB) to real official photos.
3. Queries Wikipedia PageImages API (direct title & generator search) with 500px resolution.
4. Queries TheSportsDB API as secondary fallback.
5. Downloads, caches, and syncs directly into the Docker Remotion assets container.
6. Generates a stylized broadcast SVG badge with club colors and player initials if no online photo exists.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("rendering.player_asset_manager")


class PlayerAssetManager:
    """Manages player illustration cutouts and portraits for broadcast overlays."""

    def __init__(self, output_dir: str = r"F:\DockerRenders\assets\players"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time = 0.0

    @staticmethod
    def clean_name(name: str) -> str:
        """Strips role tags, numbers, and tactical suffixes: 'Konaté (CB)' -> 'Konaté'."""
        cleaned = re.sub(r"\s*\(.*?\)", "", name)
        cleaned = re.sub(r"#\d+", "", cleaned)
        cleaned = re.sub(r"\s+(CB|LB|RB|LWB|RWB|CDM|DM|CM|CAM|AM|LW|RW|ST|CF|GK)$", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def get_slug(name: str) -> str:
        clean = PlayerAssetManager.clean_name(name).lower()
        clean = clean.replace('é', 'e').replace('è', 'e').replace('ü', 'u').replace('ñ', 'n').replace('í', 'i').replace('ó', 'o').replace('á', 'a').replace('ç', 'c')
        return re.sub(r'[^a-z0-9]', '_', clean).strip('_')

    def is_valid_real_photo(self, file_path: Path) -> bool:
        """Verifies if the file exists and is a genuine photo (not a legacy placeholder dummy circle)."""
        if not file_path.exists():
            return False
        size = file_path.stat().st_size
        # Legacy placeholder circles are small files (< 6KB)
        return size >= 8000

    def get_player_portrait(self, player_name: str, team: str = "Football Club") -> Optional[str]:
        """
        Returns the Remotion-accessible HTTP URL for the player's portrait cutout.
        Guarantees local caching so Remotion never stalls on external web requests.
        """
        if not player_name or not player_name.strip():
            return None

        clean_player = self.clean_name(player_name)
        slug = self.get_slug(clean_player)
        cached_file = self.output_dir / f"{slug}.png"

        # 1. Check local cache for genuine photo
        if self.is_valid_real_photo(cached_file):
            self._sync_to_docker(cached_file, f"{slug}.png")
            return f"http://127.0.0.1:3000/assets/players/{slug}.png"

        # 2. Query Wikipedia PageImages (covers virtually all professional players in world football)
        photo_url = self._lookup_wikipedia(clean_player)
        if photo_url:
            success = self._download_and_cache(photo_url, cached_file)
            if success:
                logger.info("Acquired official Wikipedia photo for '%s' -> %s (%d bytes)", clean_player, cached_file.name, cached_file.stat().st_size)
                self._sync_to_docker(cached_file, f"{slug}.png")
                return f"http://127.0.0.1:3000/assets/players/{slug}.png"

        # 3. Query TheSportsDB
        cutout_url = self._lookup_thesportsdb(clean_player)
        if cutout_url:
            success = self._download_and_cache(cutout_url, cached_file)
            if success:
                logger.info("Acquired official TheSportsDB cutout for '%s' -> %s", clean_player, cached_file.name)
                self._sync_to_docker(cached_file, f"{slug}.png")
                return f"http://127.0.0.1:3000/assets/players/{slug}.png"

        # 4. Generate high-res stylized SVG/PNG fallback badge if no online photo found
        if not cached_file.exists() or cached_file.stat().st_size < 100:
            badge_path = self._generate_fallback_badge(clean_player, team, cached_file)
            if badge_path and badge_path.exists():
                self._sync_to_docker(cached_file, f"{slug}.png")
                return f"http://127.0.0.1:3000/assets/players/{slug}.png"

        self._sync_to_docker(cached_file, f"{slug}.png")
        return f"http://127.0.0.1:3000/assets/players/{slug}.png"

    def _lookup_wikipedia(self, player_name: str) -> Optional[str]:
        """Queries Wikipedia PageImages API with title lookup, falling back to footballer generator search."""
        # A) Direct Title Lookup
        title = urllib.parse.quote(player_name.replace(" ", "_"))
        url = f"https://en.wikipedia.org/w/api.php?action=query&titles={title}&prop=pageimages&format=json&pithumbsize=500"
        headers = {"User-Agent": "FootballDocBot/1.0 (contact@footballdoc.org)"}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                for _, page in pages.items():
                    thumb = page.get("thumbnail", {}).get("source")
                    if thumb:
                        return thumb
        except Exception as e:
            logger.debug("Wikipedia direct title lookup for '%s' failed: %s", player_name, e)

        # B) Generator Search Lookup (e.g. "Rodrigo Mora footballer")
        q = urllib.parse.quote(f"{player_name} footballer")
        surl = f"https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch={q}&gsrlimit=3&prop=pageimages&pithumbsize=500&format=json"
        try:
            req2 = urllib.request.Request(surl, headers=headers)
            with urllib.request.urlopen(req2, timeout=4) as resp2:
                data2 = json.loads(resp2.read().decode("utf-8"))
                pages2 = data2.get("query", {}).get("pages", {})
                for _, page in pages2.items():
                    thumb = page.get("thumbnail", {}).get("source")
                    if thumb:
                        return thumb
        except Exception as e:
            logger.debug("Wikipedia search lookup for '%s' failed: %s", player_name, e)

        return None

    def _lookup_thesportsdb(self, player_name: str) -> Optional[str]:
        clean_key = player_name.lower().replace('é', 'e').replace('è', 'e').replace('ü', 'u').replace('ñ', 'n')
        search_query = re.sub(r'[^a-zA-Z0-9]', '_', clean_key).strip('_')

        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self._last_request_time = time.time()

        url = f"https://www.thesportsdb.com/api/v1/json/3/searchplayers.php?p={search_query}"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                players = data.get("player") or []
                if players:
                    return players[0].get("strCutout") or players[0].get("strThumb") or players[0].get("strRender")
        except Exception as e:
            logger.debug("TheSportsDB lookup for '%s' failed: %s", player_name, e)
        return None

    def _download_and_cache(self, url: str, dest: Path) -> bool:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "FootballDocBot/1.0 (contact@footballdoc.org)"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                if len(data) > 1000:
                    with open(dest, "wb") as f:
                        f.write(data)
                    return True
        except Exception as e:
            logger.warning("Failed to download player cutout from %s: %s", url, e)
        return False

    def _sync_to_docker(self, local_file: Path, filename: str) -> None:
        """Copies the player portrait into Remotion public assets and Docker container asset directories."""
        if not local_file.exists():
            return
        
        # 1. Sync to local remotion-server public directory
        local_public = Path(r"C:\Users\mlkme\Desktop\cursor-mikow\remotion-server\public\assets\players")
        try:
            local_public.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_file, local_public / filename)
        except Exception as e:
            logger.debug("Failed copying portrait to local remotion public: %s", e)

        # 2. Sync to Docker container public and tmp directories
        try:
            subprocess.run(
                ["docker", "cp", str(local_file), f"cursor-mikow-remotion:/app/public/assets/players/{filename}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            subprocess.run(
                ["docker", "cp", str(local_file), f"cursor-mikow-remotion:/tmp/assets/players/{filename}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except Exception:
            pass

    def _generate_fallback_badge(self, player_name: str, team: str, dest: Path) -> Path:
        """Generates a high-quality stylized broadcast player portrait badge."""
        initials = "".join([part[0].upper() for part in player_name.split() if part])[:2] or "FC"
        bg_color = "#1e3a2b" if "roma" in team.lower() else "#0f2042"
        border_color = "#f0b429" if "roma" in team.lower() else "#00e5ff"
        text_color = "#ffffff"

        svg_content = f"""<svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="grad" cx="50%" cy="40%" r="50%">
      <stop offset="0%" stop-color="{border_color}" stop-opacity="0.3" />
      <stop offset="100%" stop-color="{bg_color}" stop-opacity="0.95" />
    </radialGradient>
  </defs>
  <circle cx="100" cy="100" r="90" fill="url(#grad)" stroke="{border_color}" stroke-width="6" />
  <text x="100" y="118" font-family="system-ui, -apple-system, sans-serif" font-size="52" font-weight="900" fill="{text_color}" text-anchor="middle" letter-spacing="2">{initials}</text>
  <circle cx="100" cy="100" r="82" fill="none" stroke="{border_color}" stroke-width="1.5" stroke-dasharray="8 6" opacity="0.6"/>
</svg>"""
        try:
            dest.write_text(svg_content, encoding="utf-8")
            return dest
        except Exception:
            return dest
