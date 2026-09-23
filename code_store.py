"""
Phase 3b: Embeddings + Storage (CodeStore)

Turns each CodeChunk into an embedding and stores it in a local Chroma
collection, along with the graph metadata (calls/called_by) as retrievable
metadata -- this is what lets Phase 4 do one-hop expansion without re-parsing
anything.

Cost: $0. Storage is local Chroma (a file on disk), embeddings come from
the pluggable provider in phase3_embedding_providers.py.

HOW TO RUN:
    pip install chromadb sentence-transformers scikit-learn
    python3 phase3_code_store.py
(requires phase1_ast_parser.py, phase2_call_graph.py, and
phase3_embedding_providers.py in the same folder, plus a test_repo/ folder
with some .py files in it)
"""
import chromadb

from ast_parser import CodeChunk, RepoParser
from call_graph import CallGraphBuilder
from embedding_providers import get_embedding_provider


class CodeStore:
    def __init__(self, persist_path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(name="code_chunks")
        # Pluggable backend: tries sentence-transformers first, falls back to
        # TF-IDF only if that can't load. See phase3_embedding_providers.py.
        self.provider = get_embedding_provider()

    def embed_text(self, texts):
        return self.provider.embed(texts)

    def index_chunks(self, chunks):
        """Embed and upsert every chunk. Upsert = safe to re-run (re-indexing),
        since Chroma will overwrite existing ids rather than duplicate them --
        this is the 'diff-aware re-indexing' behavior useful for a live repo."""
        if not chunks:
            return

        # What we embed: qualified name + docstring + full source. Docstring
        # first because it's often the cleanest description of intent.
        texts_to_embed = [
            f"{c.qualified_name}\n{c.docstring}\n{c.source}" for c in chunks
        ]
        embeddings = self.embed_text(texts_to_embed)

        ids = [c.id for c in chunks]
        documents = [c.source for c in chunks]
        metadatas = [
            {
                "file_path": c.file_path,
                "qualified_name": c.qualified_name,
                "docstring": c.docstring,
                "start_line": c.start_line,
                "end_line": c.end_line,
                # Chroma metadata values must be str/int/float/bool, so join lists
                "calls": ",".join(c.calls),
                "called_by": ",".join(c.called_by),
            }
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query(self, question, top_k=3):
        """Plain semantic search -- no graph expansion yet, that's Phase 4."""
        query_embedding = self.embed_text([question])[0]
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results


if __name__ == "__main__":
    # End-to-end test: parse -> resolve graph -> embed -> store -> query
    store = CodeStore()

    parser = RepoParser("./test_repo")
    chunks = parser.parse_repo(embedding_provider=store.provider)

    graph = CallGraphBuilder(chunks)
    resolved_chunks = graph.build()

    store.index_chunks(resolved_chunks)
    print(f"Indexed {len(resolved_chunks)} chunks into Chroma.\n")

    # Try any natural-language question about your repo here
    question = "where is class imbalance handled?"
    results = store.query(question, top_k=2)

    print(f"Query: {question!r}\n")
    for i, (doc_id, metadata) in enumerate(zip(results["ids"][0], results["metadatas"][0])):
        print(f"Match {i+1}: {doc_id}")
        print(f"  docstring: {metadata['docstring']}")
        print(f"  calls: {metadata['calls']}")
        print(f"  called_by: {metadata['called_by']}")
        print()
