import urllib.request
import json
import re
import os

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def get_wikimedia_api_url(filename):
    api_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles=File:{filename}&prop=imageinfo&iiprop=url&format=json"
    req = urllib.request.Request(api_url, headers=HEADERS)
    data = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
    pages = data.get("query", {}).get("pages", {})
    for pid, pinfo in pages.items():
        if "imageinfo" in pinfo:
            return pinfo["imageinfo"][0]["url"]
    return None

import shutil

dest_dir = r"C:\Users\Yacine\AppData\Local\Temp\real_match_footage"
os.makedirs(dest_dir, exist_ok=True)

files = [
    ("Manchester_City_v_Real_Madrid_21_November_2012.jpg", "real_mancity_realmadrid_1.jpg"),
    ("Real_Madrid_Manchester_City_September_2012.JPG", "real_mancity_realmadrid_2.jpg"),
    ("Forcejeo_Real_Madrid_-_FC_Barcelona.jpg", "real_realmadrid_barcelona.jpg")
]

for wiki_name, local_name in files:
    direct = get_wikimedia_api_url(wiki_name)
    if direct:
        local_path = os.path.join(dest_dir, local_name)
        print(f"Downloading {wiki_name} -> {local_path}...")
        req = urllib.request.Request(direct, headers=HEADERS)
        with urllib.request.urlopen(req) as resp, open(local_path, "wb") as f:
            f.write(resp.read())
        size_kb = os.path.getsize(local_path) / 1024
        print(f"  Downloaded: {size_kb:.1f} KB")
