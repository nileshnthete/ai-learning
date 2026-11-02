"""
DCR RAG, step 5: retrieve the best chunks, then have Claude answer from them with citations.

Usage (from the repo root, with the venv active, VOYAGE_API_KEY and ANTHROPIC_API_KEY set):
    python dcr-rag/answer.py "How can a backend service log in to Snowflake without a password?"
"""

import sys

import anthropic

from search import HybridSearch, voyage_query_embedder

MODEL = "claude-sonnet-5-5"
TOP_K = 5   # how many chunks Claude gets to read

SYSTEM = """<role>
You answer questions about data clean rooms and the platforms around them
(Snowflake, Databricks, AWS Clean Rooms, Meta, Google) using ONLY the sources provided.
</role>

<rules>
1. Use only facts stated in the sources. Do not add knowledge from outside them,
   even if you know it, because users need answers they can verify.
2. Cite every factual sentence with the number of the source it comes from, like [2].
3. If the sources do not contain the answer, say so plainly and stop. A wrong answer
   is worse than no answer.
4. If sources disagree or cover different products, say which product each statement applies to.
</rules>

<output_format>
A short, direct answer (a few sentences or a short list), with citations.
</output_format>"""


def format_sources(chunks):
    """Number each chunk and wrap it in tags, so Claude can cite it as [n]."""
    parts = []
    for n, c in enumerate(chunks, 1):
        page = f' page="{c["page"]}"' if c["page"] else ""
        parts.append(f'<source id="{n}" url="{c["source"]}"{page}>\n{c["text"]}\n</source>')
    return "<sources>\n" + "\n".join(parts) + "\n</sources>"


def answer(question, searcher, client):
    chunks = searcher.search(question, top_k=TOP_K)
    prompt = f"{format_sources(chunks)}\n\nQuestion: {question}"
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return text, chunks


def main():
    question = " ".join(sys.argv[1:]) or "How can a backend service log in to Snowflake without a password?"
    searcher = HybridSearch(voyage_query_embedder())
    client = anthropic.Anthropic()

    text, chunks = answer(question, searcher, client)
    print(f"Question: {question}\n\n{text}\n\nSources:")
    for n, c in enumerate(chunks, 1):
        page = f", page {c['page']}" if c["page"] else ""
        print(f"  [{n}] {c['source']}{page}")


if __name__ == "__main__":
    main()