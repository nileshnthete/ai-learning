"""
DCR RAG, step 2: split every document into overlapping chunks.

Usage (from the repo root, with the venv active):
    python dcr-rag/chunk.py

Reads:  dcr-rag/data/docs/*.txt        (from ingest.py)
Writes: dcr-rag/data/chunks.jsonl      (one JSON object per chunk)

Each chunk keeps its source URL, title and (for the PDF) page number,
so answers can cite exactly where information came from.
"""

import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
DOCS_DIR = HERE / "data" / "docs"
OUT_FILE = HERE / "data" / "chunks.jsonl"

TARGET_WORDS = 300    # aim for chunks of about this size (~400 tokens)
OVERLAP_WORDS = 50    # repeat up to this many words from the end of the previous chunk


def read_doc(path):
    """Split a saved file into its header (SOURCE, TITLE) and body text."""
    header, _, body = path.read_text(encoding="utf-8").partition("\n\n")
    meta = dict(line.split(": ", 1) for line in header.splitlines() if ": " in line)
    return meta.get("SOURCE", ""), meta.get("TITLE", ""), body


def split_long(text, max_words):
    """Break a paragraph that is longer than a whole chunk into chunk-sized pieces."""
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def paragraphs(body):
    """Yield (page, paragraph) pairs. Page is set by '[page N]' markers in PDFs, else None."""
    page = None
    for block in body.split("\n\n"):
        block = block.strip()
        marker = re.match(r"\[page (\d+)\]\s*", block)
        if marker:
            page = int(marker.group(1))
            block = block[marker.end():].strip()
        if block:
            for piece in split_long(block, TARGET_WORDS):
                yield page, piece


def chunk_document(body):
    """Group paragraphs into ~TARGET_WORDS chunks, carrying a small overlap forward."""
    chunks, current, count = [], [], 0

    def emit(parts):
        pages = [p for p, _ in parts if p is not None]
        chunks.append({"page": pages[0] if pages else None,
                       "text": "\n\n".join(t for _, t in parts)})

    for page, para in paragraphs(body):
        n = len(para.split())
        if current and count + n > TARGET_WORDS:
            emit(current)
            # Overlap: carry the last OVERLAP_WORDS words of this chunk into the next one,
            # so text that spans the boundary appears in both chunks. Whole paragraphs
            # are carried while they fit; then the tail of the next one fills the rest.
            carry, carried = [], 0
            for part_page, part_text in reversed(current):
                words = part_text.split()
                room = OVERLAP_WORDS - carried
                if room <= 0:
                    break
                if len(words) <= room:
                    carry.insert(0, (part_page, part_text))
                    carried += len(words)
                else:
                    carry.insert(0, (part_page, " ".join(words[-room:])))
                    carried += room
                    break
            current, count = carry, carried
        current.append((page, para))
        count += n

    if current:
        emit(current)
    return chunks


def main():
    files = sorted(DOCS_DIR.glob("*.txt"))
    per_doc, total_words = Counter(), 0

    with OUT_FILE.open("w", encoding="utf-8") as out:
        for path in files:
            source, title, body = read_doc(path)
            for i, chunk in enumerate(chunk_document(body)):
                record = {
                    "id": f"{path.stem}-{i:04d}",
                    "source": source,
                    "title": title,
                    "page": chunk["page"],
                    "text": chunk["text"],
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                per_doc[path.stem] += 1
                total_words += len(chunk["text"].split())

    total = sum(per_doc.values())
    print(f"{total} chunks from {len(files)} documents "
          f"(average {total_words // max(total, 1)} words per chunk)\n")
    for name, n in per_doc.most_common():
        print(f"{n:>6}  {n / total:>5.1%}  {name}")
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()