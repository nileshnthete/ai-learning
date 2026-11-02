"""
DCR RAG, step 4: hybrid search = keyword (BM25) + meaning (embeddings), merged with rank fusion.

Usage (from the repo root, with the venv active and VOYAGE_API_KEY set):
    python dcr-rag/search.py "How do I rotate a Snowflake key pair?"

Prints the top results and, for each, where it ranked in the keyword list and
the embedding list, so you can see what each kind of search contributes.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

HERE = Path(__file__).parent
CHUNKS_FILE = HERE / "data" / "chunks.jsonl"
EMB_FILE = HERE / "data" / "embeddings.jsonl"

MODEL = "voyage-4"
CANDIDATES = 50   # how many results to take from each search before fusing
RRF_K = 60        # standard constant for Reciprocal Rank Fusion


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize(text):
    """Lowercase words; keeps hyphenated terms like 'key-pair' together."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


class HybridSearch:
    def __init__(self, embed_query):
        """embed_query: function that turns a question into a vector (Voyage in real use)."""
        self.embed_query = embed_query
        self.chunks = load_jsonl(CHUNKS_FILE)
        vectors = {row["id"]: row["embedding"] for row in load_jsonl(EMB_FILE)}
        missing = [c["id"] for c in self.chunks if c["id"] not in vectors]
        if missing:
            raise SystemExit(f"{len(missing)} chunks have no embedding yet - run embed.py first")
        self.matrix = np.array([vectors[c["id"]] for c in self.chunks])          # dense index
        self.bm25 = BM25Okapi([tokenize(f"{c['title']} {c['text']}") for c in self.chunks])  # sparse index

    def keyword_ranking(self, question):
        scores = self.bm25.get_scores(tokenize(question))
        ranked = np.argsort(-scores)[:CANDIDATES]
        return [i for i in ranked if scores[i] > 0]   # only chunks that share at least one word

    def embedding_ranking(self, question):
        scores = self.matrix @ np.array(self.embed_query(question))   # dot product = similarity
        return list(np.argsort(-scores)[:CANDIDATES])

    def search(self, question, top_k=5):
        keyword = self.keyword_ranking(question)
        dense = self.embedding_ranking(question)

        # Reciprocal Rank Fusion: each list adds 1 / (RRF_K + rank) to a chunk's score.
        # A chunk found by both searches gets two contributions and rises to the top.
        fused = {}
        for ranking in (keyword, dense):
            for rank, idx in enumerate(ranking, 1):
                fused[idx] = fused.get(idx, 0) + 1 / (RRF_K + rank)

        best = sorted(fused, key=fused.get, reverse=True)[:top_k]
        results = []
        for idx in best:
            chunk = dict(self.chunks[idx])
            chunk["score"] = round(fused[idx], 4)
            chunk["keyword_rank"] = keyword.index(idx) + 1 if idx in keyword else None
            chunk["embedding_rank"] = dense.index(idx) + 1 if idx in dense else None
            results.append(chunk)
        return results


def voyage_query_embedder():
    import voyageai
    vo = voyageai.Client()
    return lambda q: vo.embed([q], model=MODEL, input_type="query").embeddings[0]


def main():
    question = " ".join(sys.argv[1:]) or "How do I rotate a Snowflake key pair?"
    searcher = HybridSearch(voyage_query_embedder())
    print(f"Question: {question}\n")
    for i, r in enumerate(searcher.search(question), 1):
        page = f", page {r['page']}" if r["page"] else ""
        kw = r["keyword_rank"] or "-"
        em = r["embedding_rank"] or "-"
        print(f"{i}. [{r['score']}] keyword #{kw}, embedding #{em}  {r['id']}{page}")
        print(f"   {r['source']}")
        print(f"   {r['text'][:200].replace(chr(10), ' ')}...\n")


if __name__ == "__main__":
    main()