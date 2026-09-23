"""
Phase 6b: Answer-Quality Evaluation

phase6_eval.py answers "did the right code reach the LLM's context?" (retrieval
quality). It does NOT check whether the LLM's final natural-language answer is
actually correct -- a question could retrieve the perfect context and still get
summarized into a vague or wrong answer. This script closes that gap.

Method: for each question in eval_questions.json, run the real pipeline end to
end (retrieval -> Ollama-generated answer, same as phase5_api.py's /ask), then
check whether the generated answer text actually names the correct function(s)
-- not an LLM-as-judge score, just a deterministic substring check against
identifiers derived from the question's known-correct chunk ids. This keeps
scoring reproducible and free (no second LLM call to judge the first one).

Two metrics, mirroring phase6_eval.py's primary/neighbor split so the two
evaluations tell one connected story:
  - answer_primary_recall:  does the answer name the directly-relevant function?
  - answer_neighbor_recall: for "flow" questions, does the answer ALSO mention
                             the related (graph-expanded) function -- i.e. did
                             the one-hop context that phase6_eval proved gets
                             RETRIEVED actually make it into the final ANSWER?

Requires Ollama running locally with the model configured in
phase5_answer_providers.py (default: llama3.2:3b). Each call takes ~15-25s on
CPU, so a 20-question run takes several minutes -- this is real LLM inference,
not a mock.

HOW TO RUN:
    python3 phase6b_answer_eval.py
"""
from retriever import GraphExpandedRetriever
from answer_providers import get_answer_provider
from retrieval_eval import build_index, load_questions


def _format_context(results: list) -> str:
    parts = []
    for r in results:
        tag = "DIRECT MATCH" if r["relation"] == "direct_match" else f"RELATED (via {r['matched_via']})"
        parts.append(f"[{tag}] {r['id']}\nDocstring: {r['docstring']}\n```\n{r['source']}\n```")
    return "\n\n".join(parts)


def expected_terms(chunk_id: str) -> list:
    """Derive plain-text terms a correct answer should mention from a chunk id
    like 'model.py::ModelTrainer.train' -> ['ModelTrainer', 'train',
    'ModelTrainer.train']."""
    qualified = chunk_id.split("::")[-1]
    parts = [p for p in qualified.split(".") if p and p != "__module__"]
    terms = set(parts)
    if len(parts) > 1:
        terms.add(qualified)
    return list(terms)


def score_answer(retriever, answer_provider, q: dict, top_k: int = 3) -> dict:
    results = retriever.retrieve(q["question"], top_k=top_k)
    context = _format_context(results)
    answer = answer_provider.generate(q["question"], context)
    answer_lower = answer.lower()

    primary_terms = [t for cid in q["primary_ids"] for t in expected_terms(cid)]
    neighbor_terms = [t for cid in q["neighbor_ids"] for t in expected_terms(cid)]

    return {
        "question": q["question"],
        "answer": answer,
        "primary_hit": any(t.lower() in answer_lower for t in primary_terms) if primary_terms else None,
        "neighbor_hit": any(t.lower() in answer_lower for t in neighbor_terms) if neighbor_terms else None,
    }


def main():
    store = build_index()
    retriever = GraphExpandedRetriever(store)
    answer_provider = get_answer_provider(backend="ollama")
    questions = load_questions()
    scored_questions = [q for q in questions if q["type"] != "none"]

    print(f"\n[answer-eval] running {len(scored_questions)} questions through real Ollama generation "
          f"(~15-25s each, this will take a few minutes)...\n")

    results = []
    for i, q in enumerate(scored_questions, start=1):
        r = score_answer(retriever, answer_provider, q)
        results.append(r)
        mark_p = "hit" if r["primary_hit"] else "MISS"
        mark_n = "" if r["neighbor_hit"] is None else (" | neighbor: hit" if r["neighbor_hit"] else " | neighbor: MISS")
        print(f"[{i}/{len(scored_questions)}] primary: {mark_p}{mark_n}  -- {q['question']}")

    primary_scores = [r["primary_hit"] for r in results if r["primary_hit"] is not None]
    neighbor_scores = [r["neighbor_hit"] for r in results if r["neighbor_hit"] is not None]

    answer_primary_recall = sum(primary_scores) / len(primary_scores) if primary_scores else 0.0
    answer_neighbor_recall = sum(neighbor_scores) / len(neighbor_scores) if neighbor_scores else 0.0

    print("\n" + "=" * 70)
    print("ANSWER-QUALITY RESULTS (does the final LLM answer name the right function?)")
    print("=" * 70)
    print(f"answer_primary_recall:  {answer_primary_recall*100:.1f}%  ({sum(primary_scores)}/{len(primary_scores)})")
    print(f"answer_neighbor_recall: {answer_neighbor_recall*100:.1f}%  ({sum(neighbor_scores)}/{len(neighbor_scores)})")

    print("\nSample answers (first 3):")
    for r in results[:3]:
        print(f"\nQ: {r['question']}")
        print(f"A: {r['answer'][:300]}")

    return results


if __name__ == "__main__":
    main()
