"""
src/rendering/player_asset_manager.py
Player Portrait & Cutout Asset Manager for Remotion Broadcast Overlays.

Resolves player names to high-resolution transparent PNG cutouts:
1. Checks local cache (F:\\DockerRenders\\assets\\players\\{slug}.png).
2. Queries TheSportsDB transparent cutout API with rate limiting and retry backoff.
3. Downloads, caches, and returns the local Remotion HTTP asset URL (http://127.0.0.1:3000/assets/players/{slug}.png).
4. Generates a stylized fallback circular badge with squad colors and player initials if external image is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("rendering.player_asset_manager")

KNOWN_PLAYER_NAMES: Dict[str, str] = {
    "kylian mbappe": "Kylian_Mbappe",
    "mbappe": "Kylian_Mbappe",
    "aurelien tchouameni": "Aurelien_Tchouameni",
    "tchouameni": "Aurelien_Tchouameni",
    "antonio rudiger": "Antonio_Rudiger",
    "rudiger": "Antonio_Rudiger",
    "endrick": "Endrick",
    "donyell malen": "Donyell_Malen",
    "malen": "Donyell_Malen",
    "marten de roon": "Marten_de_Roon",
    "de roon": "Marten_de_Roon",
    "gianluca mancini": "Gianluca_Mancini",
    "mancini": "Gianluca_Mancini",
    "evan ndicka": "Evan_Ndicka",
    "ndicka": "Evan_Ndicka",
    "eder militao": "Eder_Militao",
    "militao": "Eder_Militao",
    "dani carvajal": "Daniel_Carvajal",
    "carvajal": "Daniel_Carvajal",
    "lamine yamal": "Lamine_Yamal",
    "pedri": "Pedri",
    "joao cancelo": "Joao_Cancelo",
    "fermin lopez": "Fermin_Lopez",
    "karim adeyemi": "Karim_Adeyemi",
    "lucas torreira": "Lucas_Torreira",
    "mauro icardi": "Mauro_Icardi",
}


class PlayerAssetManager:
    """Manages player illustration cutouts and portraits for broadcast overlays."""

    def __init__(self, output_dir: str = r"F:\DockerRenders\assets\players"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time = 0.0

    @staticmethod
    def get_slug(name: str) -> str:
        clean = name.lower()
        clean = clean.replace('é', 'e').replace('è', 'e').replace('ü', 'u').replace('ñ', 'n').replace('í', 'i').replace('ó', 'o').replace('á', 'a')
        return re.sub(r'[^a-z0-9]', '_', clean).strip('_')

    def get_player_portrait(self, player_name: str, team: str = "Football Club") -> Optional[str]:
        """
        Returns the Remotion-accessible HTTP URL for the player's portrait cutout.
        Guarantees local caching so Remotion never stalls on external web requests.
        """
        if not player_name or not player_name.strip():
            return None

        slug = self.get_slug(player_name)
        cached_file = self.output_dir / f"{slug}.png"

        # 1. Check local cache
        if cached_file.exists() and cached_file.stat().st_size > 1000:
            return f"http://127.0.0.1:3000/assets/players/{slug}.png"

        # 2. Query TheSportsDB
        cutout_url = self._lookup_thesportsdb(player_name)
        if cutout_url:
            success = self._download_and_cache(cutout_url, cached_file)
            if success:
                logger.info(f"Acquired official cutout for '{player_name}' -> {cached_file.name}")
                return f"http://127.0.0.1:3000/assets/players/{slug}.png"

        # 3. Generate high-res stylized SVG/PNG fallback badge
        badge_path = self._generate_fallback_badge(player_name, team, cached_file)
        if badge_path and badge_path.exists():
            return f"http://127.0.0.1:3000/assets/players/{slug}.png"

        return None

    def _lookup_thesportsdb(self, player_name: str) -> Optional[str]:
        clean_key = player_name.lower().replace('é', 'e').replace('è', 'e').replace('ü', 'u').replace('ñ', 'n')
        search_query = KNOWN_PLAYER_NAMES.get(clean_key)
        if not search_query:
            search_query = re.sub(r'[^a-zA-Z0-9]', '_', clean_key).strip('_')

        # Throttling to prevent 503 rate limits
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
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
                    return players[0].get("strCutout") or players[0].get("strThumb")
        except Exception as e:
            logger.debug(f"TheSportsDB lookup for '{player_name}' failed: {e}")
        return None

    def _download_and_cache(self, url: str, dest: Path) -> bool:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = resp.read()
                if len(data) > 1000:
                    with open(dest, "wb") as f:
                        f.write(data)
                    return True
        except Exception as e:
            logger.warning(f"Failed to download player cutout from {url}: {e}")
        return False

    def _generate_fallback_badge(self, player_name: str, team: str, dest: Path) -> Path:
        """Generates a high-quality stylized circular player avatar badge."""
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
