import urllib.request
import re

def search_soccercatch(query="Barcelona"):
    url = f"https://soccercatch.com/search?q={urllib.parse.quote(query)}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            # Look for match links
            links = re.findall(r'href="(https://soccercatch\.com/matches/[^"]+)"', html)
            unique_links = list(dict.fromkeys(links))
            print(f"Soccercatch found {len(unique_links)} match links for query '{query}':")
            for link in unique_links[:5]:
                print(" -", link)
            return unique_links
    except Exception as e:
        print("Soccercatch search error:", e)
        return []

def search_fullmatchsports(query="Barcelona"):
    url = f"https://fullmatchsports.cc/?s={urllib.parse.quote(query)}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            links = re.findall(r'href="(https://fullmatchsports\.cc/[^"]+/)"', html)
            filtered = [l for l in links if not any(x in l for x in ['category', 'tag', 'page', 'author', 'contact', 'dmca'])]
            unique_links = list(dict.fromkeys(filtered))
            print(f"FullMatchSports found {len(unique_links)} match links for query '{query}':")
            for link in unique_links[:5]:
                print(" -", link)
            return unique_links
    except Exception as e:
        print("FullMatchSports search error:", e)
        return []

if __name__ == "__main__":
    search_soccercatch("Barcelona")
    search_fullmatchsports("Barcelona")
