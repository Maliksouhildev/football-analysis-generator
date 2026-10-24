import urllib.request
import re

url = "https://raw.githubusercontent.com/roboflow/sports/main/README.md"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
content = urllib.request.urlopen(req).read().decode("utf-8")
urls = re.findall(r"https?://[^\s\)\"\>]+", content)
for u in urls:
    print(u)
