"""
Extract all Hindi article URLs from Vikaspedia Sitemap
"""

import requests
import xml.etree.ElementTree as ET
from pathlib import Path

# =====================================================
# SETTINGS
# =====================================================

SITEMAP_URL = "https://agriculture.vikaspedia.in/sitemap.xml"

OUTPUT_FOLDER = Path("data/urls")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_FOLDER / "hindi_urls.txt"

# =====================================================
# DOWNLOAD SITEMAP
# =====================================================

print("=" * 60)
print("Downloading Sitemap...")
print("=" * 60)

response = requests.get(
    SITEMAP_URL,
    headers={"User-Agent": "Mozilla/5.0"}
)

response.raise_for_status()

print("Downloaded Successfully!")

# =====================================================
# PARSE XML
# =====================================================

root = ET.fromstring(response.content)

namespace = {
    "ns": "http://www.sitemaps.org/schemas/sitemap/0.9"
}

urls = set()

for url in root.findall("ns:url", namespace):

    loc = url.find("ns:loc", namespace)

    if loc is None:
        continue

    link = loc.text

    if ("?lgn=hi" in link and "/viewcontent/" in link):
        urls.add(link)

# =====================================================
# SAVE
# =====================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for url in urls:
        f.write(url + "\n")

print("=" * 60)
print("Done!")
print("=" * 60)

print(f"Total Hindi URLs : {len(urls)}")
print(f"Saved to : {OUTPUT_FILE}")

