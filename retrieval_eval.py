"""
Phase 6: Retrieval Evaluation Harness

Answers the question every RAG demo dodges: "how do you know it works?"

Runs a fixed question set (eval_questions.json) against a small purpose-built
fixture repo (eval_repo/) with known call-chain relationships (data loading ->
class-imbalance handling -> training -> evaluation -> inference -> API), then
sweeps top_k and one-hop graph expansion on/off to measure:

  - primary_recall:  did the directly-relevant chunk show up at all?
  - neighbor_recall:  for "flow" questions, did the one-hop-related chunk
                       (caller/callee) also show up? This is the metric that
                       specifically tests whether graph expansion (Phase 4)
                       is pulling its weight, or just adding noise.

Why a dedicated eval_repo instead of test_repo/: test_repo/ (a Streamlit churn
app) is nearly all module-level script code with almost no cross-function call
graph, so it can't exercise graph expansion at all -- the README's own
flagship example ("where is class imbalance handled?" -> undersample() ->
prepare_training_data()) doesn't actually exist in test_repo/. eval_repo/
reproduces that exact shape on purpose, so the eval is reproducible regardless
of whatever real project test_repo/ happens to point at.

Uses an isolated Chroma path (./chroma_db_eval) so running this never touches
the real index used by phase5_api.py.

HOW TO RUN:
    python3 phase6_eval.py
"""
import json

from ast_parser import RepoParser
from call_graph import CallGraphBuilder
from code_store import CodeStore
from retriever import GraphExpandedRetriever

EVAL_REPO = "./eval_repo"
EVAL_DB_PATH = "./chroma_db_eval"
QUESTIONS_PATH = "./eval_questions.json"


def build_index() -> CodeStore:
    store = CodeStore(persist_path=EVAL_DB_PATH)

    parser = RepoParser(EVAL_REPO)
    chunks = parser.parse_repo(embedding_provider=store.provider)
    graph = CallGraphBuilder(chunks)
    resolved_chunks = graph.build()

    store.index_chunks(resolved_chunks)
    print(f"[eval] indexed {len(resolved_chunks)} chunks from {EVAL_REPO} into {EVAL_DB_PATH}")
    return store


def load_questions() -> list:
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def score_question(retriever: GraphExpandedRetriever, q: dict, top_k: int, force_expand: bool) -> dict:
    results = retriever.retrieve(q["question"], top_k=top_k, force_expand=force_expand)
    all_ids = {r["id"] for r in results}
    direct_ids = {r["id"] for r in results if r["relation"] == "direct_match"}

    primary_ids = set(q["primary_ids"])
    neighbor_ids = set(q["neighbor_ids"])

    return {
        "primary_hit": bool(primary_ids & all_ids) if primary_ids else None,
        "primary_hit_direct": bool(primary_ids & direct_ids) if primary_ids else None,
        "neighbor_hit": bool(neighbor_ids & all_ids) if neighbor_ids else None,
        "top_result": results[0]["id"] if results else None,
    }


def run_config(retriever: GraphExpandedRetriever, questions: list, top_k: int, force_expand: bool) -> dict:
    scored = [score_question(retriever, q, top_k, force_expand) for q in questions if q["type"] != "none"]

    primary_scores = [s["primary_hit"] for s in scored if s["primary_hit"] is not None]
    neighbor_scores = [s["neighbor_hit"] for s in scored if s["neighbor_hit"] is not None]

    return {
        "top_k": top_k,
        "force_expand": force_expand,
        "primary_recall": sum(primary_scores) / len(primary_scores) if primary_scores else 0.0,
        "neighbor_recall": sum(neighbor_scores) / len(neighbor_scores) if neighbor_scores else 0.0,
        "n_primary": len(primary_scores),
        "n_neighbor": len(neighbor_scores),
    }


def print_table(rows: list):
    header = f"{'top_k':>5} | {'expand':>6} | {'primary_recall':>14} | {'neighbor_recall':>15} | n_primary | n_neighbor"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['top_k']:>5} | {str(r['force_expand']):>6} | "
            f"{r['primary_recall']*100:>13.1f}% | {r['neighbor_recall']*100:>14.1f}% | "
            f"{r['n_primary']:>9} | {r['n_neighbor']:>10}"
        )


def check_none_questions(retriever: GraphExpandedRetriever, questions: list, top_k: int = 3):
    """Qualitative check for out-of-scope questions: the system has no correct
    answer to give, so there's no recall metric -- just log the top match so a
    human can eyeball whether it's confidently returning something irrelevant."""
    none_qs = [q for q in questions if q["type"] == "none"]
    if not none_qs:
        return
    print("\nOut-of-scope questions (no correct answer exists in eval_repo/) -- eyeball check:")
    for q in none_qs:
        results = retriever.retrieve(q["question"], top_k=top_k)
        top = results[0]["id"] if results else "(no results)"
        print(f"  Q: {q['question']!r}\n     top match returned: {top}")


def main():
    store = build_index()
    retriever = GraphExpandedRetriever(store)
    questions = load_questions()

    configs = [
        (3, False),  # baseline: semantic search only, no graph expansion
        (3, True),   # one-hop expansion on (current default behavior)
        (1, True),
        (5, True),
    ]

    rows = [run_config(retriever, questions, top_k, expand) for top_k, expand in configs]

    print(f"\nEval set: {len(questions)} questions ({len(questions) - 1} scored, 1 out-of-scope)\n")
    print_table(rows)

    baseline = next(r for r in rows if not r["force_expand"] and r["top_k"] == 3)
    expanded = next(r for r in rows if r["force_expand"] and r["top_k"] == 3)
    lift = (expanded["neighbor_recall"] - baseline["neighbor_recall"]) * 100
    print(
        f"\nGraph expansion effect @top_k=3: neighbor_recall {baseline['neighbor_recall']*100:.1f}% "
        f"(no expansion) -> {expanded['neighbor_recall']*100:.1f}% (one-hop expansion), "
        f"a {lift:+.1f} point change."
    )

    check_none_questions(retriever, questions)


if __name__ == "__main__":
    main()
