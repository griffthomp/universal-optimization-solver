import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"
OUTPUT_DIR = "./tsplib_downloads"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fetch and parse HTML
print("Fetching TSPLIB index...")
res = requests.get(BASE_URL)
soup = BeautifulSoup(res.content, "html.parser")

# Get all file links
links = soup.find_all("a")
filenames = sorted(set(a["href"] for a in links if a.get("href", "").endswith((".tsp", ".tsp.gz"))))

# Normalize to tsp base names (avoid double download)
base_names = sorted(set(f.replace(".tsp.gz", "").replace(".tsp", "") for f in filenames))

# Try to download .tsp.gz, else fallback to .tsp
for name in base_names:
    for ext in [".tsp.gz", ".tsp"]:
        file_url = urljoin(BASE_URL, name + ext)
        filename = name + ext
        output_path = os.path.join(OUTPUT_DIR, filename)
        print(f"Trying: {filename}...")

        try:
            r = requests.get(file_url, stream=True)
            if r.status_code == 200:
                with open(output_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✅ Downloaded: {filename}")
                break  # Stop trying other formats once successful
            else:
                print(f"❌ Not found: {filename}")
        except Exception as e:
            print(f"⚠️ Error downloading {filename}: {e}")

