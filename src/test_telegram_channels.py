import urllib.request
import re

channels = [
    "football_replays",
    "fullmatchreplays",
    "soccerreplays",
    "footballdownload"
]

for ch in channels:
    url = f"https://t.me/s/{ch}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            # Look for message texts
            messages = re.findall(r'<div[^>]*class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', html, re.S)
            print(f"[@{ch}] Channel active! Found {len(messages)} recent messages.")
            for msg in messages[-3:]:
                clean_text = re.sub(r'<[^>]+>', ' ', msg).strip()
                if any(x in clean_text.lower() for x in ['vs', 'v', 'league', 'ucl', 'half', '1080p']):
                    print(f"  Match: {clean_text[:80]}...")
    except Exception as e:
        print(f"[@{ch}] Error:", e)
