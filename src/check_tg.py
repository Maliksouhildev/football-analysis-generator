import urllib.request
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

channels = ["replaysfootball", "footballreplays", "footballmatchreplay", "fullmatchfootball"]
for ch in channels:
    url = f"https://t.me/s/{ch}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        html = urllib.request.urlopen(req, timeout=8).read().decode("utf-8", errors="ignore")
        # Extract full message wraps
        blocks = re.findall(r'<div class="tgme_widget_message_wrap[^"]*">(.*?)</div>\s*</div>\s*</div>', html, re.DOTALL)
        print(f"\n================ Channel @{ch}: {len(blocks)} message blocks ================")
        for b in blocks[-6:]:
            text_m = re.search(r'class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', b, re.DOTALL)
            text = text_m.group(1) if text_m else ""
            clean_text = " ".join(re.sub(r'<[^>]+>', " ", text).split())
            links = re.findall(r'href="([^"]+)"', b)
            vids = re.findall(r'<video[^>]*src="([^"]+)"', b)
            print(f"- Text: {clean_text[:100]}")
            if vids:
                print(f"  VIDEO: {vids}")
            if links:
                print(f"  LINKS: {links}")
    except Exception as e:
        print(f"Channel @{ch} error: {e}")
