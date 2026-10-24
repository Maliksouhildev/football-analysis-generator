import urllib.request
import json

def get_gofile_download_url(content_id):
    # Gofile API requires getting a guest account token first or querying contents
    # Step 1: create guest account
    req = urllib.request.Request(
        "https://api.gofile.io/accounts",
        data=b"{}",
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            acc_data = json.loads(resp.read().decode())
            token = acc_data.get("data", {}).get("token")
            print("Gofile token obtained:", token[:10] + "...")
    except Exception as e:
        print(f"Failed to create gofile guest account: {e}")
        token = None

    # Step 2: query contents
    url = f"https://api.gofile.io/contents/{content_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req2 = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req2, timeout=10) as resp2:
            data = json.loads(resp2.read().decode())
            print("Gofile response status:", data.get("status"))
            children = data.get("data", {}).get("children", {})
            for child_id, file_info in children.items():
                print("File found:", file_info.get("name"), "Size:", file_info.get("size"), "Link:", file_info.get("link"))
                return file_info.get("link")
    except Exception as e:
        print(f"Failed to get gofile content: {e}")
    return None

if __name__ == "__main__":
    get_gofile_download_url("1iHZJ88w")
