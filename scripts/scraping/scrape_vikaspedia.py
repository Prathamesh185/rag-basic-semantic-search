import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# PATHS
URL_FILE = Path("data/urls/hindi_urls.txt")

OUTPUT_DIR = Path("data/scraped/hindi")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FAILED_FILE = Path("data/urls/failed_urls.txt")

# HEADERS

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    )
}

# HELPERS

def slugify(url):
    
    """
    Generate filename from URL slug.
    """

    path = urlparse(url).path.strip("/")

    if not path:
        return "home"

    slug = path.split("/")[-1]

    slug = re.sub(
        r"[^a-zA-Z0-9\u0900-\u097F_-]",
        "_",
        slug
    )

    return slug[:100]


def clean_text(text):

    # Clean extracted article text.

    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n +", "\n", text)

    return text.strip()


# LOAD URLS
with open(URL_FILE, "r", encoding="utf-8") as f:
    urls = [line.strip() for line in f if line.strip()]

# Remove duplicate URLs while preserving order
urls = list(dict.fromkeys(urls))

TOTAL = len(urls)

print("=" * 70)
print(f"Total URLs : {TOTAL}")
print("=" * 70)

saved = 0
skipped = 0
failed = 0

# SCRAPER

for index, url in enumerate(urls, start=1):

    filename = OUTPUT_DIR / f"{slugify(url)}.json"

    # Resume support

    if filename.exists():
        skipped += 1
        print(f"[{index}/{TOTAL}] Skipped")
        continue

    success = False

    # Retry up to 3 times
    for attempt in range(3):

        try:

            response = requests.get(
                url,
                headers=HEADERS,
                timeout=60
            )

            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # TITLE
            title = ""

            if soup.title:
                title = soup.title.get_text(strip=True)

            # SUMMARY

            summary = ""

            meta = soup.find(
                "meta",
                attrs={"name": "description"}
            )

            if meta:
                summary = meta.get("content", "")

            # CONTENT
            content = ""

            next_data = soup.find(
                "script",
                id="__NEXT_DATA__"
            )

            if next_data and next_data.string:

                try:

                    data = json.loads(next_data.string)

                    page_props = (
                        data.get("props", {})
                        .get("pageProps", {})
                    )

                    ssr = page_props.get("ssrPageContent")

                    if ssr and ssr.get("content"):

                        html = ssr["content"]

                        article = BeautifulSoup(
                            html,
                            "lxml"
                        )

                        content = article.get_text(
                            separator="\n",
                            strip=True
                        )

                except Exception:
                    pass

            # Fallback
            if len(content.strip()) < 100:

                content = soup.get_text(
                    separator="\n",
                    strip=True
                )

            # Clean text

            content = clean_text(content)

            # Skip tiny pages
            if len(content) < 300:

                print(
                    f"[{index}/{TOTAL}] Too little content, skipped"
                )

                success = True
                break

            # Save JSON
            article = {
                "title": title,
                "summary": summary,
                "content": content,
                "language": "hi",
                "source": "Vikaspedia",
                "url": url,
                "scraped_at": datetime.now().isoformat(),
                "word_count": len(content.split())
            }

            with open(
                filename,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    article,
                    f,
                    ensure_ascii=False,
                    indent=4
                )

            saved += 1
            success = True

            print(f"[{index}/{TOTAL}] Saved")

            # Random polite delay
            sleep_time = random.uniform(1, 2)

            print(
                f"Sleeping {sleep_time:.1f} sec..."
            )

            time.sleep(sleep_time)

            break

        except Exception as e:

            print(
                f"[{index}/{TOTAL}] Attempt {attempt + 1}/3 failed"
            )

            print(e)

            if attempt < 2:

                wait = random.randint(8, 15)

                print(
                    f"Retrying in {wait} sec..."
                )

                time.sleep(wait)

    # Permanent failure
    if not success:

        failed += 1

        print(f"[{index}/{TOTAL}] FAILED")

        with open(
            FAILED_FILE,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(url + "\n")

        # Extra delay after failure
        time.sleep(20)

# SUMMARY
print()
print("=" * 70)
print("SCRAPING FINISHED")
print("=" * 70)

print(f"Saved   : {saved}")
print(f"Skipped : {skipped}")
print(f"Failed  : {failed}")