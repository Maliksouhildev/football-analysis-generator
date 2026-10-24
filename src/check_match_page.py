import urllib.request
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

url = "https://www.fullmatch-replay.org/roma-vs-real-madrid-14-oct-2026/"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
try:
    html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", errors="ignore")
    print("Page Title:", re.findall(r'<title>(.*?)</title>', html))
    print("\n--- Anchors ---")
    anchors = re.findall(r'<a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
    for href, text in anchors:
        clean = " ".join(re.sub(r'<[^>]+>', " ", text).split())
        print(f"  {clean[:30]} -> {href}")
    print("\n--- Iframes ---")
    for ifr in re.findall(r'<iframe[^>]*src="([^"]+)"', html):
        print("  IFRAME:", ifr)
except Exception as e:
    print(f"Error fetching {url}: {e}")
