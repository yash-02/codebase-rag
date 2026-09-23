"""
Phase 5b: FastAPI App

Ties together everything from Phases 1-4 (parse -> call graph -> embed/store ->
graph-expanded retrieval) plus Phase 5a (answer generation) into one HTTP API.

Endpoints:
    GET  /        -- browser UI
    POST /index   -- parse and index a repo
    POST /ask     -- ask a question, get an answer with citations
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ast_parser import RepoParser
from call_graph import CallGraphBuilder
from code_store import CodeStore
from retriever import GraphExpandedRetriever
from answer_providers import get_answer_provider

BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"

app = FastAPI(title="Codebase-Aware RAG")
app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")

# Shared state -- initialized once at startup
store = CodeStore()
retriever = GraphExpandedRetriever(store)

# CHANGE THIS ONE LINE to switch answer backend: "stub" (no setup) -> "ollama"
# (local, free, needs Ollama running) -> "anthropic" (paid API, needs API key)
answer_provider = get_answer_provider(backend="ollama")


class IndexRequest(BaseModel):
    repo_path: str


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


@app.get("/", response_class=HTMLResponse)
def ui():
    """Serve the browser UI."""
    provider_name = answer_provider.__class__.__name__.replace("Provider", "")
    html = (UI_DIR / "index.html").read_text(encoding="utf-8")
    return html.replace("__ANSWER_PROVIDER__", provider_name)


@app.post("/index")
def index_repo(req: IndexRequest):
    """Parse a repo, build its call graph, and embed+store every chunk."""
    parser = RepoParser(req.repo_path)
    chunks = parser.parse_repo(embedding_provider=store.provider)

    graph = CallGraphBuilder(chunks)
    resolved_chunks = graph.build()

    store.index_chunks(resolved_chunks)
    return {"status": "ok", "chunks_indexed": len(resolved_chunks)}


def _run_retrieval_and_answer(question: str, top_k: int):
    """Shared logic used by both /ask (JSON) and /ask/plain (readable text)."""
    results = retriever.retrieve(question, top_k=top_k)

    context_parts = []
    for r in results:
        tag = "DIRECT MATCH" if r["relation"] == "direct_match" else f"RELATED (via {r['matched_via']})"
        context_parts.append(
            f"[{tag}] {r['id']}\n"
            f"Docstring: {r['docstring']}\n"
            f"```\n{r['source']}\n```"
        )
    context = "\n\n".join(context_parts)

    answer = answer_provider.generate(question, context)
    sources = [r["id"] for r in results]
    return answer, sources


@app.post("/ask")
def ask(req: AskRequest):
    """JSON version for scripts, frontends, and other API callers."""
    answer, sources = _run_retrieval_and_answer(req.question, req.top_k)
    return {
        "question": req.question,
        "answer": answer,
        "sources": sources,
    }


@app.post("/ask/plain", response_class=PlainTextResponse)
def ask_plain(req: AskRequest):
    """Plain-text version for terminals."""
    answer, sources = _run_retrieval_and_answer(req.question, req.top_k)
    sources_list = "\n".join(f"  - {s}" for s in sources)
    return (
        f"Question: {req.question}\n"
        f"{'=' * 60}\n\n"
        f"{answer}\n\n"
        f"{'=' * 60}\n"
        f"Sources used:\n{sources_list}\n"
    )


if __name__ == "__main__":
    import uvicorn

    host = "127.0.0.1"
    port = 8000
    print(f"Open the browser UI at http://localhost:{port}/")
    uvicorn.run(app, host=host, port=port)
