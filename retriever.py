"""
Phase 4: Graph-Expanded Retrieval

Plain semantic search (Phase 3) finds functions that *sound* related to a
question. But often the real answer needs a function's neighbors too --
e.g. asking "where is class imbalance handled?" matches `undersample()`,
but the full picture also needs `prepare_training_data()` (which calls it)
to make sense.

This phase:
1. Runs normal semantic search to get the top-k direct matches
2. For each match, reads its `calls` / `called_by` metadata (already resolved
   in Phase 2, stored in Chroma in Phase 3)
3. Fetches those neighbor chunks too (one hop only -- two hops tends to pull
   in too much unrelated code and drowns out the actual answer)
4. Returns a combined, deduplicated context: direct matches clearly labeled
   separately from graph-expanded neighbors

HOW TO RUN:
    python3 phase4_retriever.py
(requires phase1_ast_parser.py, phase2_call_graph.py,
phase3_embedding_providers.py, phase3_code_store.py, and a populated
test_repo/ folder in the same directory -- run phase3_code_store.py at least
once first so the Chroma DB exists)
"""
from ast_parser import RepoParser
from call_graph import CallGraphBuilder
from code_store import CodeStore


class GraphExpandedRetriever:
    def __init__(self, store: CodeStore):
        self.store = store

    def _should_expand_graph(self, question: str) -> bool:
        """Use graph expansion for flow questions, but keep fact/list queries focused."""
        normalized = question.lower()
        focused_terms = (
            "what are",
            "which",
            "list",
            "input",
            "inputs",
            "variable",
            "variables",
            "feature",
            "features",
            "column",
            "columns",
            "parameter",
            "parameters",
        )
        return not any(term in normalized for term in focused_terms)

    def retrieve(self, question: str, top_k: int = 3, force_expand: bool | None = None):
        """Returns a list of dicts, each with:
            - id, docstring, source, file_path
            - relation: "direct_match" or "graph_neighbor"
            - matched_via: for neighbors, which direct match pulled it in

        force_expand: override the focused-query heuristic. True/False forces
        graph expansion on/off regardless of question wording -- used by the
        eval harness (phase6_eval.py) to A/B one-hop expansion vs none on the
        same question set. Leave as None for normal heuristic-driven behavior.
        """
        semantic_results = self.store.query(question, top_k=top_k)
        expand_graph = self._should_expand_graph(question) if force_expand is None else force_expand

        direct_ids = semantic_results["ids"][0]
        direct_metadatas = semantic_results["metadatas"][0]
        direct_documents = semantic_results["documents"][0]

        results = []
        seen_ids = set()
        neighbor_ids_to_fetch = set()
        neighbor_source_map = {}  # neighbor_id -> which direct match pulled it in

        # Step 1: collect direct matches
        for doc_id, metadata, document in zip(direct_ids, direct_metadatas, direct_documents):
            results.append({
                "id": doc_id,
                "docstring": metadata["docstring"],
                "source": document,
                "file_path": metadata["file_path"],
                "relation": "direct_match",
                "matched_via": None,
            })
            seen_ids.add(doc_id)

            # Gather one-hop neighbors from the resolved graph metadata
            if not expand_graph:
                continue

            calls = metadata["calls"].split(",") if metadata["calls"] else []
            called_by = metadata["called_by"].split(",") if metadata["called_by"] else []
            for neighbor_id in calls + called_by:
                if neighbor_id and neighbor_id not in seen_ids:
                    neighbor_ids_to_fetch.add(neighbor_id)
                    neighbor_source_map[neighbor_id] = doc_id

        # Step 2: fetch the actual neighbor chunks from Chroma by id
        if neighbor_ids_to_fetch:
            fetched = self.store.collection.get(ids=list(neighbor_ids_to_fetch))
            for doc_id, metadata, document in zip(
                fetched["ids"], fetched["metadatas"], fetched["documents"]
            ):
                if doc_id in seen_ids:
                    continue  # don't duplicate something already a direct match
                results.append({
                    "id": doc_id,
                    "docstring": metadata["docstring"],
                    "source": document,
                    "file_path": metadata["file_path"],
                    "relation": "graph_neighbor",
                    "matched_via": neighbor_source_map[doc_id],
                })
                seen_ids.add(doc_id)

        return results


if __name__ == "__main__":
    # Rebuild the index fresh (safe to re-run, upsert handles re-indexing)
    parser = RepoParser("./test_repo")
    chunks = parser.parse_repo()
    graph = CallGraphBuilder(chunks)
    resolved_chunks = graph.build()

    store = CodeStore()
    store.index_chunks(resolved_chunks)

    retriever = GraphExpandedRetriever(store)

    question = "where is class imbalance handled?"
    results = retriever.retrieve(question, top_k=1)

    print(f"Query: {question!r}\n")
    for r in results:
        tag = "[DIRECT MATCH]" if r["relation"] == "direct_match" else f"[PULLED IN via {r['matched_via']}]"
        print(f"{tag} {r['id']}")
        print(f"  docstring: {r['docstring']}")
        print()
