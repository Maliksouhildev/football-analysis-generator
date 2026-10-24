import urllib.request
import re

url = "https://fullmatchsports.cc/galatasaray-vs-barcelona-full-match-ucl-2026-27/?tab=links"
req = urllib.request.Request(url, headers={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})

try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
        links = re.findall(r'<a\s+[^>]*href="([^"]+)"[^>]*>([^<]*)</a>', html)
        print(f"Total links found: {len(links)}")
        found_hosts = []
        for l, text in links:
            if any(h in l.lower() for h in ['mega', 'gofile', 'streamwish', 'filelions', '1fichier', 'dood', 'voe', 'download', 'drive']):
                print(f"Host: {text.strip()} -> {l}")
                found_hosts.append((text.strip(), l))
        if not found_hosts:
            # Look for iframe embeds
            iframes = re.findall(r'<iframe[^>]+src="([^">]+)"', html)
            print("Iframes:", iframes)
except Exception as e:
    print("Error parsing page:", e)
