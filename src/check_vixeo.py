import urllib.request
import re

url = 'https://vixeo.io/e/DmsluD7M5Y7e'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'Referer': 'https://www.fullmatch-replay.org/'})
html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8', errors='ignore')

print("Occurrences of m3u8:", len(re.findall(r'm3u8', html)))
print("Occurrences of mp4:", len(re.findall(r'mp4', html)))
print("Occurrences of http in script:", len(re.findall(r'https?://[^\s"\'<>]+', html)))

urls = re.findall(r'https?://[^\s"\'<>]+', html)
for u in urls[:20]:
    print("Found URL:", u)

# Let's search for video, player, hls, source tags
for tag in re.findall(r'<(?:video|source|iframe)[^>]+>', html):
    print("Tag:", tag)
