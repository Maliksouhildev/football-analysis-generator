"""
src/ingestion/fullmatch_scraper.py
Automated Replay Web Scraper for FullMatchSports & Replay Aggregators.

Extracts direct high-speed hosting links (Gofile, Mega, Streamwish, FileLions)
from match pages (?tab=links) and triggers yt-dlp download.
"""

import logging
import re
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

logger = logging.getLogger("ingestion.fullmatch_scraper")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


class FullMatchWebScraper:
    def __init__(self):
        self.headers = {"User-Agent": USER_AGENT}

    def extract_download_links_from_page(self, page_url: str) -> List[Dict[str, str]]:
        """
        Extracts all direct file-hosting links (Gofile, Mega, Streamwish, FileLions)
        from a FullMatchSports match page across all available tabs.
        """
        clean_base = page_url.split("?")[0].rstrip("/")
        tabs_to_try = [
            f"{clean_base}/?tab=links",
            f"{clean_base}/?tab=download",
            f"{clean_base}/?tab=full-match",
            clean_base,
        ]
        extracted = []
        seen_urls = set()

        for tab_url in tabs_to_try:
            logger.info(f"Scraping download links from: {tab_url}")
            try:
                req = urllib.request.Request(tab_url, headers=self.headers)
                with urllib.request.urlopen(req, timeout=12) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")

                # Match anchor links
                raw_links = re.findall(r'<a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
                for url, label in raw_links:
                    label_clean = re.sub(r'<[^>]+>', ' ', label).strip()
                    url_clean = url.strip()
                    if url_clean in seen_urls:
                        continue
                    # Match major direct download hosts
                    for host in ["gofile.io", "mega.nz", "streamwish", "filelions", "1fichier", "doodstream", "voe.sx"]:
                        if host in url_clean.lower():
                            seen_urls.add(url_clean)
                            extracted.append({
                                "host": host,
                                "label": label_clean,
                                "url": url_clean,
                            })
                            break

                # Also match embedded player iframes (Streamwish, Ok.ru, etc.)
                iframes = re.findall(r'<iframe\s+[^>]*src=[\x27\x22]([^\x27\x22]+)[\x27\x22]', html, re.IGNORECASE)
                for ifr in iframes:
                    for host in ["streamwish", "filelions", "ok.ru", "doodstream", "voe.sx"]:
                        if host in ifr.lower() and ifr not in seen_urls:
                            seen_urls.add(ifr)
                            extracted.append({
                                "host": host,
                                "label": f"Embedded Stream ({host})",
                                "url": ifr,
                            })

                if extracted:
                    logger.info(f"Discovered {len(extracted)} valid download/stream hosts from {tab_url}.")
                    return extracted
            except Exception as e:
                logger.warn(f"Failed to scrape replay links from {tab_url}: {e}")

        logger.info(f"Found {len(extracted)} valid download hosts across all tabs.")
        return extracted

    def search_match_replay(self, home_team: str, away_team: str) -> Optional[str]:
        """
        Searches FullMatchSports for a given match and returns the best matching page URL.
        """
        query = f"{home_team} vs {away_team}"
        search_url = f"https://fullmatchsports.cc/?s={urllib.parse.quote(query)}"
        logger.info(f"Searching FullMatchSports for '{query}'...")
        
        req = urllib.request.Request(search_url, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
                
            # Find matching article URLs
            post_links = re.findall(r'<h2[^>]*class="[^"]*entry-title[^"]*"[^>]*><a\s+href="([^"]+)"', html)
            if not post_links:
                post_links = re.findall(r'href="(https://fullmatchsports\.cc/[a-z0-9\-]+-full-match[^"]*/)"', html)
                
            if post_links:
                logger.info(f"Found match page: {post_links[0]}")
                return post_links[0]
        except Exception as e:
            logger.error(f"Search failed for '{query}': {e}")
            
        return None
