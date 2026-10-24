"""
Searches and downloads authentic high-definition broadcast match action photos from Wikimedia Commons.
"""

import json
import os
import requests

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {"User-Agent": "TacticalMasterclassFootballBot/2.0 (research@tactics.org)"}

QUERIES = [
    "incategory:Association_football_match_actions",
    "incategory:Association_football_players_in_action",
    "incategory:Real_Madrid_CF_matches",
    "incategory:FC_Barcelona_matches",
    "football match inplay broadcast action pitch"
]

def search_wikimedia_images():
    api_url = "https://commons.wikimedia.org/w/api.php"
    found_images = []
    seen_urls = set()

    for q in QUERIES:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": q,
            "gsrnamespace": 6,
            "gsrlimit": 10,
            "prop": "imageinfo",
            "iiprop": "url|size|mime",
            "format": "json"
        }
        try:
            r = requests.get(api_url, params=params, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                pages = r.json().get("query", {}).get("pages", {})
                for pid, p in pages.items():
                    info = p.get("imageinfo", [{}])[0]
                    w = info.get("width", 0)
                    h = info.get("height", 0)
                    url = info.get("url", "")
                    mime = info.get("mime", "")
                    if w >= 1200 and h >= 700 and "jpeg" in mime.lower() and url not in seen_urls:
                        seen_urls.add(url)
                        found_images.append({
                            "title": p.get("title", ""),
                            "width": w,
                            "height": h,
                            "url": url
                        })
        except Exception as e:
            print(f"Query {q} error: {e}")

    print(f"Total candidate images found: {len(found_images)}")
    return found_images

def download_images(imgs):
    downloaded = []
    for idx, img in enumerate(imgs[:4], 1):
        target_path = os.path.join(OUTPUT_DIR, f"real_match_broadcast_{idx}.jpg")
        print(f"Downloading {img['title']} -> {target_path}...")
        try:
            r = requests.get(img["url"], headers=HEADERS, timeout=20)
            if r.status_code == 200:
                with open(target_path, "wb") as f:
                    f.write(r.content)
                downloaded.append(target_path)
                print(f"  [OK] Saved {target_path} ({len(r.content) // 1024} KB)")
        except Exception as e:
            print(f"  [FAIL] {e}")
    return downloaded

if __name__ == "__main__":
    imgs = search_wikimedia_images()
    for img in imgs[:8]:
        print(f"  {img['title']} ({img['width']}x{img['height']}) -> {img['url']}")
    download_images(imgs)
