"""
Phase 7: MCP Server

Exposes this project's existing pipeline (Phases 1-5, untouched) as MCP tools,
so any MCP-compatible client -- Claude Desktop, Claude Code, or a custom agent
-- can index a repo and ask questions about it as first-class tool calls,
instead of only through the REST API in phase5_api.py.

This file adds no new retrieval/answer logic. It only wires the same
RepoParser -> CallGraphBuilder -> CodeStore -> GraphExpandedRetriever pipeline
into MCP tool functions, which is why it's a separate ~100-line file rather
than a rewrite: the layering in phases 1-5 made this additive.

Three tools, deliberately kept separate rather than collapsed into one:
  - index_repo:    parse + embed + store a repo path
  - search_code:   raw graph-expanded retrieval, no LLM -- lets an agent (e.g.
                   Claude Code itself) reason directly over the retrieved
                   source instead of getting a pre-summarized answer
  - ask_question:  retrieval + LLM-generated natural-language answer with
                   citations, for callers that want a finished answer

HOW TO RUN (stdio transport, for Claude Desktop / Claude Code):
    python3 phase7_mcp_server.py

Register it with an MCP client by pointing at this file, e.g. in
claude_desktop_config.json:
    {
      "mcpServers": {
        "codebase-rag": {
          "command": "python3",
          "args": ["/absolute/path/to/phase7_mcp_server.py"]
        }
      }
    }
"""
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from ast_parser import RepoParser
from call_graph import CallGraphBuilder
from code_store import CodeStore
from retriever import GraphExpandedRetriever
from answer_providers import get_answer_provider

mcp = FastMCP("codebase-rag")

# Shared state, same pattern as phase5_api.py -- initialized once per process.
_store = CodeStore()
_retriever = GraphExpandedRetriever(_store)
_answer_provider = get_answer_provider(backend="ollama")


def _format_context(results: list) -> str:
    parts = []
    for r in results:
        tag = "DIRECT MATCH" if r["relation"] == "direct_match" else f"RELATED (via {r['matched_via']})"
        parts.append(f"[{tag}] {r['id']}\nDocstring: {r['docstring']}\n```\n{r['source']}\n```")
    return "\n\n".join(parts)


@mcp.tool()
def index_repo(repo_path: str) -> dict:
    """Parse a Python repo (.py files and .ipynb notebooks), build its call
    graph, and embed+store every function/method/module chunk for retrieval.

    Args:
        repo_path: filesystem path to the repo to index (relative paths are
            resolved against this server's working directory).

    Returns a dict with the number of chunks indexed, or an error message if
    the path doesn't exist.
    """
    resolved = Path(repo_path)
    if not resolved.exists():
        return {"status": "error", "message": f"repo_path does not exist: {repo_path}"}

    parser = RepoParser(str(resolved))
    chunks = parser.parse_repo(embedding_provider=_store.provider)
    graph = CallGraphBuilder(chunks)
    resolved_chunks = graph.build()
    _store.index_chunks(resolved_chunks)
    return {"status": "ok", "chunks_indexed": len(resolved_chunks)}


@mcp.tool()
def search_code(question: str, top_k: int = 3) -> dict:
    """Retrieve the raw code chunks most relevant to a natural-language
    question about the currently indexed repo, expanded one hop through the
    call graph. Returns source code directly, without an LLM-generated
    summary -- use this when the calling agent wants to read and reason over
    the code itself.

    Args:
        question: natural-language question about the indexed codebase.
        top_k: number of direct semantic matches to retrieve before graph
            expansion (default 3).
    """
    if not question.strip():
        return {"status": "error", "message": "question must not be empty"}

    results = _retriever.retrieve(question, top_k=top_k)
    return {
        "status": "ok",
        "results": [
            {
                "id": r["id"],
                "file_path": r["file_path"],
                "relation": r["relation"],
                "matched_via": r["matched_via"],
                "docstring": r["docstring"],
                "source": r["source"],
            }
            for r in results
        ],
    }


@mcp.tool()
def ask_question(question: str, top_k: int = 3) -> dict:
    """Ask a natural-language question about the currently indexed repo and
    get back an LLM-generated answer with citations, using graph-expanded
    retrieval for context. Requires an answer backend to be configured (see
    this file's _answer_provider) -- defaults to Ollama running locally.

    Args:
        question: natural-language question about the indexed codebase.
        top_k: number of direct semantic matches to retrieve before graph
            expansion (default 3).
    """
    if not question.strip():
        return {"status": "error", "message": "question must not be empty"}

    results = _retriever.retrieve(question, top_k=top_k)
    context = _format_context(results)
    answer = _answer_provider.generate(question, context)
    return {
        "status": "ok",
        "answer": answer,
        "sources": [r["id"] for r in results],
    }


if __name__ == "__main__":
    mcp.run()
