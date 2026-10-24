import subprocess
import json
import logging
from pathlib import Path

logger = logging.getLogger("test_yt")

def search_youtube_replay(home_team: str, away_team: str):
    search_queries = [
        f"{home_team} vs {away_team} full match",
        f"{home_team} vs {away_team} extended highlights",
        f"{home_team} vs {away_team} match highlights"
    ]
    
    candidates = []
    for q in search_queries:
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-playlist",
            f"ytsearch5:{q}"
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            for line in res.stdout.strip().split("\n"):
                if not line.strip(): continue
                data = json.loads(line)
                dur = data.get("duration", 0)
                title = data.get("title", "")
                url = data.get("webpage_url")
                
                # Check team keywords in title
                t_low = title.lower()
                h_match = any(w in t_low for w in home_team.lower().split() if len(w) > 2)
                a_match = any(w in t_low for w in away_team.lower().split() if len(w) > 2)
                
                if h_match and a_match and dur > 180: # At least 3 minutes, prefer longer
                    candidates.append({
                        "title": title,
                        "duration": dur,
                        "url": url,
                        "uploader": data.get("uploader", "")
                    })
        except Exception as e:
            print(f"Error querying '{q}': {e}")
            
    # Sort by duration descending (we want the fullest broadcast footage)
    candidates.sort(key=lambda x: x["duration"], reverse=True)
    return candidates

if __name__ == "__main__":
    results = search_youtube_replay("Roma", "Real Madrid")
    print(f"Found {len(results)} valid candidates:")
    for r in results[:5]:
        print(f"  {r['duration']//60}m{r['duration']%60}s | {r['title']} | {r['url']}")
