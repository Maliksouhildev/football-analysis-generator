import urllib.request
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

url = "https://fullmatchshows.com/roma-vs-real-madrid-full-match-replay-14-oct-2026/"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", errors="ignore")

print("--- All Links on fullmatchshows post ---")
anchors = re.findall(r'<a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
for href, text in anchors:
    clean = " ".join(re.sub(r'<[^>]+>', " ", text).split())
    if any(k in href.lower() or k in clean.lower() for k in ["download", "server", "stream", "play", "1st", "2nd", "half", "video", "file"]):
        print(f"  {clean} -> {href}")

print("\n--- Iframes ---")
for ifr in re.findall(r'<iframe[^>]*src="([^"]+)"', html):
    print("  IFRAME:", ifr)
