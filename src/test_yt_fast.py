import subprocess
import time

t0 = time.time()
query = "Champions League match highlights tactical analysis"
cmd = [
    "yt-dlp",
    "--print", "%(id)s\t%(duration)s\t%(title)s",
    f"ytsearch3:{query}"
]
res = subprocess.run(cmd, capture_output=True, text=True)
print(f"Elapsed: {time.time() - t0:.2f}s")
print("Output:")
print(res.stdout)
