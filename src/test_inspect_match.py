import urllib.request
import re

sites = [
    ("FootyFull", "https://footyfull.com/"),
    ("FullMatchesAndShows", "https://fullmatchesandshows.com/"),
    ("SoccerCatch", "https://soccercatch.com/"),
]

for name, url in sites:
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            # find match titles
            titles = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>([^<]*(?:vs|v|-)[^<]*Full Match[^<]*)</a>', html, re.I)
            if not titles:
                titles = re.findall(r'<h[23][^>]*><a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>', html)
            print(f"[{name}] Found {len(titles)} posts:")
            for t in titles[:3]:
                print(f"  - {t[1].strip()}: {t[0]}")
    except Exception as e:
        print(f"[{name}] Error:", e)
