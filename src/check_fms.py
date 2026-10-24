import urllib.request
import re

url = "https://www.fullmatchshows.com/?s=Real+Madrid"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
try:
    html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", errors="ignore")
    print("Page length:", len(html))
    matches = re.findall(r'<a\s+[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', html)
    for href, text in matches[:20]:
        if any(k in href.lower() for k in ["roma", "madrid", "ucl", "full-match"]):
            print(f"  {text.strip()} -> {href}")
except Exception as e:
    print(f"Error: {e}")
