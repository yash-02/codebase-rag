"""
Phase 2: Call-Graph Builder

Phase 1 gave every chunk a list of raw names it calls (e.g. "prepare_training_data",
but also noise like "read_csv" from pandas). This phase:

1. Resolves each raw call name to an actual chunk ID, if one exists in the repo
   (e.g. "prepare_training_data" -> "data_utils.py::prepare_training_data")
2. Drops anything that doesn't resolve (library calls, builtins) — we only keep
   edges between functions we actually parsed
3. Builds the reverse edge: called_by, so every chunk knows who calls it

This resolved graph is what makes retrieval smarter than plain vector search:
at query time (Phase 4) we can pull in one-hop neighbors of a matched chunk.

Known limitation (worth stating plainly in interviews): this is name-based
resolution, not real static analysis. If two different classes both have a
method called "train", a call to "self.train()" could resolve ambiguously.
We handle the common case (unique names across the repo) and fall back to
"first match" otherwise -- a real implementation would need type inference
to resolve this precisely, which is out of scope here.

HOW TO RUN:
    python3 call_graph.py
(requires phase1_ast_parser.py in the same folder — it imports RepoParser from it)
"""
from ast_parser import CodeChunk, RepoParser


class CallGraphBuilder:
    def __init__(self, chunks: list[CodeChunk]):
        self.chunks = chunks
        # Index chunks by short name (e.g. "train") -> list of chunk ids
        # (a name can map to multiple chunks, e.g. two classes with the same method name)
        self.name_index: dict[str, list[str]] = {}
        for chunk in chunks:
            self.name_index.setdefault(chunk.name, []).append(chunk.id)

        self.by_id: dict[str, CodeChunk] = {c.id: c for c in chunks}

    def build(self) -> list[CodeChunk]:
        """Resolve calls -> resolved_calls, and fill in called_by. Returns the same
        chunk objects, mutated in place, for simplicity."""
        for chunk in self.chunks:
            resolved = []
            for raw_name in chunk.calls:
                matches = self.name_index.get(raw_name)
                if not matches:
                    continue  # library/builtin call, not in our repo -> drop
                for target_id in matches:
                    if target_id != chunk.id:
                        resolved.append(target_id)
            chunk.calls = sorted(set(resolved))  # overwrite raw names with resolved chunk ids

        # Now build called_by as the reverse of calls
        for chunk in self.chunks:
            for target_id in chunk.calls:
                target_chunk = self.by_id[target_id]
                if chunk.id not in target_chunk.called_by:
                    target_chunk.called_by.append(chunk.id)

        return self.chunks

    def get_neighbors(self, chunk_id: str) -> list[str]:
        """One-hop neighbors: everything this chunk calls + everything that calls it."""
        chunk = self.by_id[chunk_id]
        return sorted(set(chunk.calls + chunk.called_by))


if __name__ == "__main__":
    parser = RepoParser("./test_repo")
    chunks = parser.parse_repo()

    graph = CallGraphBuilder(chunks)
    resolved_chunks = graph.build()

    for c in resolved_chunks:
        print(f"\n--- {c.id} ---")
        print(f"  calls (resolved):     {c.calls}")
        print(f"  called_by (resolved): {c.called_by}")

    # Example: check one-hop neighbors for a specific function.
    # Change this to any chunk id printed above.
    if resolved_chunks:
        example_id = resolved_chunks[0].id
        print(f"\n--- One-hop neighbor test for '{example_id}' ---")
        print(graph.get_neighbors(example_id))
