"""
DCR RAG evals (2.3): measure retrieval and answers SEPARATELY.

Usage (from the repo root, venv active, VOYAGE_API_KEY and ANTHROPIC_API_KEY set):
    python dcr-rag/rag_evals.py

Test cases live in dcr-rag/rag_eval_cases.json. Each answerable case says which
document(s) should answer it ("sources": part of the URL) and a phrase the right
chunk must contain ("phrase"). A retrieved chunk counts as a hit when it comes from
one of those sources AND contains the phrase.

Three stages:
  0. Check the test set   - every phrase must exist in the corpus, or the case is invalid
  1. Retrieval eval       - deterministic, one run: recall@5, recall@10, MRR
  2. Answer eval          - RUNS runs per case, graded by an LLM judge
Details are saved to dcr-rag/data/rag_eval_results.json.
"""

import json
from collections import defaultdict
from pathlib import Path

import anthropic

from answer import answer, format_sources
from search import HybridSearch, voyage_query_embedder

HERE = Path(__file__).parent
CASES_FILE = HERE / "rag_eval_cases.json"
RESULTS_FILE = HERE / "data" / "rag_eval_results.json"

RUNS = 2                          # answer runs per case (answers vary; retrieval doesn't)
JUDGE_MODEL = "claude-sonnet-5-5"


def normalize(text):
    return " ".join(text.split()).lower()


def is_hit(chunk, case):
    """Right document AND contains the expected phrase."""
    from_right_doc = any(s in chunk["source"] for s in case["sources"])
    return from_right_doc and normalize(case["phrase"]) in normalize(chunk["text"])


# ---------- Stage 0: check the test set ----------

def validate(cases, chunks):
    valid = []
    for case in cases:
        if "phrase" not in case:          # unanswerable: nothing to find
            valid.append(case)
            continue
        matches = sum(is_hit(c, case) for c in chunks)
        if matches == 0:
            print(f"  INVALID {case['id']}: no chunk from {case['sources']} contains '{case['phrase']}'")
        else:
            case["relevant_chunks"] = matches
            valid.append(case)
    return valid


# ---------- Stage 1: retrieval ----------

def retrieval_eval(cases, searcher):
    rows = []
    for case in cases:
        if "phrase" not in case:
            continue
        results = searcher.search(case["question"], top_k=10)
        rank = next((i for i, c in enumerate(results, 1) if is_hit(c, case)), None)
        rows.append({"id": case["id"], "category": case["category"], "rank": rank,
                     "top5": [c["id"] for c in results[:5]]})
    return rows


def print_retrieval(rows):
    def summary(rs):
        n = len(rs)
        r5 = sum(1 for r in rs if r["rank"] and r["rank"] <= 5)
        r10 = sum(1 for r in rs if r["rank"])
        mrr = sum(1 / r["rank"] for r in rs if r["rank"]) / n
        return f"recall@5 {r5}/{n}   recall@10 {r10}/{n}   MRR {mrr:.2f}"

    print("\nRETRIEVAL")
    for r in rows:
        mark = "OK " if r["rank"] and r["rank"] <= 5 else "!! "
        print(f"  {mark}{r['id']:<26} rank {r['rank'] or 'not in top 10'}")
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    print()
    for cat, rs in by_cat.items():
        print(f"  {cat:<22} {summary(rs)}")
    print(f"  {'ALL':<22} {summary(rows)}")


# ---------- Stage 2: answers ----------

RUBRIC_ANSWERABLE = (
    "PASS only if ALL of these hold: (1) every factual claim in the answer is supported by "
    "the source it cites; (2) the answer actually addresses the question; (3) it adds no facts "
    "that are not in the sources. If the sources lack the answer and the answer says so "
    "honestly, that also counts as PASS for (1) and (3) but FAIL for (2)."
)
RUBRIC_UNANSWERABLE = (
    "The sources do not contain the answer to this question. PASS only if the answer says "
    "the information is not in the sources and does not supply it from outside knowledge."
)


def judge(client, question, chunks, answer_text, answerable):
    prompt = (
        "You are grading an answer produced by a document question-answering system.\n\n"
        f"{format_sources(chunks)}\n\n"
        f"Question: {question}\n\n"
        f"Answer:\n{answer_text}\n\n"
        f"Rubric: {RUBRIC_ANSWERABLE if answerable else RUBRIC_UNANSWERABLE}\n\n"
        "First explain your reasoning in 2-4 sentences, checking each citation. "
        "Then give PASS or FAIL alone on the last line."
    )
    r = client.messages.create(model=JUDGE_MODEL, max_tokens=2000,
                               messages=[{"role": "user", "content": prompt}])
    verdict = "".join(b.text for b in r.content if b.type == "text").strip()
    if not verdict:
        return False, f"Judge gave no verdict (stop_reason: {r.stop_reason})"
    last = verdict.splitlines()[-1].strip("* ").upper()
    return last.startswith("PASS"), verdict


def answer_eval(cases, searcher, client, retrieval_rows):
    rank_by_id = {r["id"]: r["rank"] for r in retrieval_rows}
    rows = []
    for case in cases:
        answerable = "phrase" in case
        passes, details = 0, []
        for run in range(RUNS):
            text, chunks = answer(case["question"], searcher, client)
            ok, reason = judge(client, case["question"], chunks, text, answerable)
            passes += ok
            details.append({"run": run + 1, "pass": ok, "answer": text, "judge": reason})
        rows.append({"id": case["id"], "category": case["category"], "passes": passes,
                     "retrieval_rank": rank_by_id.get(case["id"]), "runs": details})
    return rows


def print_answers(rows):
    print(f"\nANSWERS ({RUNS} runs each)")
    for r in rows:
        mark = "OK " if r["passes"] == RUNS else "!! "
        hint = ""
        if r["passes"] < RUNS and r["retrieval_rank"] is not None:
            hint = ("  <- answer problem (right chunk was retrieved)"
                    if r["retrieval_rank"] <= 5 else "  <- retrieval problem (right chunk not in top 5)")
        print(f"  {mark}{r['id']:<26} {r['passes']}/{RUNS}{hint}")
    n = len(rows)
    all_pass = sum(1 for r in rows if r["passes"] == RUNS)
    total = sum(r["passes"] for r in rows)
    print(f"\n  pass rate {total}/{n * RUNS} runs;   pass^{RUNS} (every run passed) {all_pass}/{n} cases")


def main():
    cases = json.loads(CASES_FILE.read_text())
    searcher = HybridSearch(voyage_query_embedder())
    client = anthropic.Anthropic()

    print("CHECKING TEST SET")
    cases = validate(cases, searcher.chunks)
    print(f"  {len(cases)} valid cases")

    retrieval_rows = retrieval_eval(cases, searcher)
    print_retrieval(retrieval_rows)

    answer_rows = answer_eval(cases, searcher, client, retrieval_rows)
    print_answers(answer_rows)

    RESULTS_FILE.write_text(json.dumps({"retrieval": retrieval_rows, "answers": answer_rows},
                                       indent=2, ensure_ascii=False))
    print(f"\nDetails saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()