"""
DCR RAG, step 1: download every source in dcr_sources.txt and save it as clean text.

Usage (from the repo root, with the venv active):
    python dcr-rag/ingest.py

Output: dcr-rag/data/docs/<name>.txt, one file per source. Each file starts with
"SOURCE: <url>" and "TITLE: <title>" so later steps can cite where text came from.
"""

import io
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

HERE = Path(__file__).parent
SOURCES = HERE / "dcr_sources.txt"
OUT_DIR = HERE / "data" / "docs"
HEADERS = {"User-Agent": "Mozilla/5.0 (personal learning project)"}
MIN_CHARS = 500   # less than this usually means the page needs JavaScript or blocked us


def read_sources():
    """URLs from dcr_sources.txt, skipping blank lines and # comments."""
    lines = (line.strip() for line in SOURCES.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def file_name(url):
    """https://docs.snowflake.com/en/user-guide/key-pair-auth -> docs-snowflake-com-en-user-guide-key-pair-auth.txt"""
    parts = urlparse(url)
    slug = re.sub(r"[^a-z0-9]+", "-", (parts.netloc + parts.path).lower()).strip("-")
    return slug[:120] + ".txt"


def clean(text):
    text = unicodedata.normalize("NFKC", text)  # PDF ligatures like 'ﬀ' 'ﬁ' 'ﬂ' -> 'ff' 'fi' 'fl'
    text = re.sub(r"[ \t]+", " ", text)          # squeeze runs of spaces
    text = re.sub(r"\n\s*\n+", "\n\n", text)     # squeeze runs of blank lines
    return text.strip()


def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    # Remove page furniture that would pollute search results
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    return title, clean(main.get_text("\n"))


def pdf_to_text(data):
    reader = PdfReader(io.BytesIO(data))
    title = reader.metadata.title if reader.metadata and reader.metadata.title else ""
    # Keep page markers so answers can later cite "page 42"
    pages = [f"[page {i}]\n{page.extract_text() or ''}" for i, page in enumerate(reader.pages, 1)]
    return title, clean("\n\n".join(pages))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved, failed = [], []

    for url in read_sources():
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            is_pdf = url.lower().endswith(".pdf") or "pdf" in resp.headers.get("content-type", "")
            title, text = pdf_to_text(resp.content) if is_pdf else html_to_text(resp.text)
            if len(text) < MIN_CHARS:
                raise ValueError(f"only {len(text)} characters of text (needs JavaScript or blocked?)")

            path = OUT_DIR / file_name(url)
            path.write_text(f"SOURCE: {url}\nTITLE: {title}\n\n{text}", encoding="utf-8")
            saved.append(len(text))
            print(f"OK    ~{len(text) // 4:>8,} tokens  {path.name}")
        except Exception as e:
            failed.append(url)
            print(f"FAIL  {url}\n      {e}")

    print(f"\n{len(saved)} saved, {len(failed)} failed, about {sum(saved) // 4:,} tokens in total.")


if __name__ == "__main__":
    main()