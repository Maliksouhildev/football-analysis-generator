import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.fullmatch_scraper import FullMatchWebScraper

scraper = FullMatchWebScraper()
page = "https://fullmatchsports.cc/atletico-madrid-vs-real-madrid-full-match-2026-27/"
links = scraper.extract_download_links_from_page(page)
print(f"Extracted links for Atletico vs Real Madrid:")
for l in links:
    print(" ", l)

page2 = "https://fullmatchsports.cc/real-madrid-vs-inter-full-match-ucl-2026-27/"
links2 = scraper.extract_download_links_from_page(page2)
print(f"Extracted links for Real Madrid vs Inter:")
for l in links2:
    print(" ", l)
