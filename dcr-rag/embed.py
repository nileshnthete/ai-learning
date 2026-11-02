"""
DCR RAG, step 3a: create an embedding (dense vector) for every chunk with Voyage.

Usage (from the repo root, with the venv active and VOYAGE_API_KEY set):
    python dcr-rag/embed.py

Reads:  dcr-rag/data/chunks.jsonl
Writes: dcr-rag/data/embeddings.jsonl   ({"id": ..., "embedding": [...]} per line)

Resumable: chunks that already have an embedding are skipped, so if a rate limit
or network error stops the run, just run it again and it carries on.
"""

import json
import time
from pathlib import Path

import voyageai

HERE = Path(__file__).parent
CHUNKS_FILE = HERE / "data" / "chunks.jsonl"
OUT_FILE = HERE / "data" / "embeddings.jsonl"

MODEL = "voyage-4"
BATCH_SIZE = 32     # chunks per API call
MAX_RETRIES = 6


def load_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def embed_with_retry(vo, texts):
    """Call Voyage; on errors such as rate limits, wait and retry with growing delays."""
    for attempt in range(MAX_RETRIES):
        try:
            return vo.embed(texts, model=MODEL, input_type="document").embeddings
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt * 5          # 5s, 10s, 20s, 40s, 80s
            print(f"  {type(e).__name__}: {e} -> waiting {wait}s and retrying")
            time.sleep(wait)


def main():
    chunks = load_jsonl(CHUNKS_FILE)
    done = {row["id"] for row in load_jsonl(OUT_FILE)}
    todo = [c for c in chunks if c["id"] not in done]
    print(f"{len(chunks)} chunks, {len(done)} already embedded, {len(todo)} to go")

    vo = voyageai.Client()   # reads VOYAGE_API_KEY
    with OUT_FILE.open("a", encoding="utf-8") as out:
        for start in range(0, len(todo), BATCH_SIZE):
            batch = todo[start:start + BATCH_SIZE]
            # Embed the title together with the text: a cheap way to give each chunk
            # a little context about which document it came from.
            texts = [f"{c['title']}\n\n{c['text']}" for c in batch]
            vectors = embed_with_retry(vo, texts)
            for chunk, vector in zip(batch, vectors):
                out.write(json.dumps({"id": chunk["id"], "embedding": vector}) + "\n")
            out.flush()   # save progress after every batch
            print(f"  embedded {min(start + BATCH_SIZE, len(todo))}/{len(todo)}")

    print(f"Done. Embeddings saved to {OUT_FILE}")


if __name__ == "__main__":
    main()