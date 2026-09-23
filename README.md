# Codebase-Aware RAG

A Retrieval-Augmented Generation system that answers natural-language questions about a codebase — not by naively chunking files into fixed-size text blocks, but by parsing real function/class structure and using the call graph between functions to pull in relevant context automatically.

**Example:** ask *"where is class imbalance handled?"* and it doesn't just find `undersample()` — it also pulls in `prepare_training_data()`, the function that calls it, because the two are connected in the call graph.

---

## Why this is different from a basic "chat with your code" tool

Most simple implementations embed every file as raw text and do vector search. That breaks in predictable ways: a 200-line function gets cut in half by fixed-size chunking, and retrieval only ever returns the one matched snippet with no surrounding context.

This project instead:
- Parses `.py` files and Python code cells inside `.ipynb` notebooks with Python's `ast` module, so chunks are always whole functions/methods, never cut mid-logic
- Builds a resolved call graph (who calls whom), filtering out library/builtin noise
- Expands retrieval one hop across that graph, so an LLM answering a question sees a function *and* its immediate caller/callee, not an isolated snippet

---

## Architecture

```
Python source files
        │
        ▼
Phase 1: AST Parser          →  extracts every function/method + module-level code
        │
        ▼
Phase 2: Call Graph Builder  →  resolves raw call names into real links, drops noise
        │
        ▼
Phase 3: Embedding + Storage →  embeds each chunk, stores in local Chroma DB
        │
        ▼
Phase 4: Graph-Expanded      →  semantic search + one-hop call graph expansion
         Retrieval
        │
        ▼
Phase 5: FastAPI + LLM       →  /index and /ask endpoints, answer generation
         Answer Generation      with citations
        │
        ▼
Phase 6: Eval Harness        →  measures retrieval quality against a known
                                 answer set, quantifies graph-expansion value
        │
        ▼
Phase 7: MCP Server          →  same pipeline, exposed as MCP tools for any
                                 MCP-compatible agent (Claude Desktop, Claude
                                 Code, etc.)
```

Every phase is provider-agnostic where it matters:
- **Embeddings:** `sentence-transformers` (local, free) is primary, with automatic fallback to TF-IDF if the model can't be downloaded
- **Answer generation:** swap between `stub` (no setup), `ollama` (local, free), or `anthropic` (paid API) with one config line

---

## File structure

```
codebase_rag/
├── ast_parser.py           # AST parsing, function/module chunk extraction
├── call_graph.py           # Call graph resolution (calls / called_by)
├── embedding_providers.py  # Pluggable embedding backends
├── code_store.py           # Chroma storage + semantic search
├── retriever.py            # Graph-expanded retrieval
├── answer_providers.py     # Pluggable LLM backends (stub / ollama / anthropic / openai)
├── server.py               # FastAPI app (UI, /index, /ask)
├── retrieval_eval.py       # Retrieval eval harness (recall metrics, top_k/hop A/B)
├── answer_eval.py          # Answer-quality eval harness (does the final answer name the right function?)
├── mcp_server.py           # MCP server exposing the pipeline as agent tools
├── eval_questions.json     # Question set with known-correct chunk ids
├── eval_repo/              # Small fixture repo with real call chains, used only by retrieval_eval.py
├── ui/
│   ├── index.html          # Browser UI markup
│   ├── styles.css          # Browser UI styling
│   └── app.js               # Browser UI behavior
├── README.md
└── test_repo/               # Point this at any Python codebase you want to query
```

All files import flat (`from ast_parser import ...`) — keep them in one folder, no package structure needed.

---

## Setup

Requires **Python 3.10+**. Steps differ slightly by OS because of how virtual environments are created/activated — everything else (the commands you actually run day to day) is identical.

### Windows (PowerShell or Command Prompt)

```powershell
git clone https://github.com/yash-02/codebase-rag.git
cd codebase-rag
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks the activation script with an execution-policy error, run this once first, then retry `.venv\Scripts\activate`:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### macOS / Linux (bash/zsh)

```bash
git clone https://github.com/yash-02/codebase-rag.git
cd codebase-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

First run downloads the embedding model (~80MB, one-time, needs internet). Every run after that works fully offline.

### Optional: local LLM for real answers (recommended, free)

1. Install [Ollama](https://ollama.com/download) — native installers for Windows, macOS, and Linux are all on that page.
2. Pull a model:
   ```bash
   ollama pull llama3.1:8b
   ```
   (Same command on Windows, macOS, and Linux — run it in PowerShell/Terminal after installing Ollama.)
3. In `server.py`, set:
   ```python
   answer_provider = get_answer_provider(backend="ollama")
   ```

### Optional: Claude API for higher-quality answers (paid, ~cents per query)

```bash
pip install anthropic
```
Set `ANTHROPIC_API_KEY` as an environment variable, then in `server.py`:
```python
answer_provider = get_answer_provider(backend="anthropic")
```

Restart the server after changing this line — it's read once at startup, not per-request.

### Optional: OpenAI API for higher-quality answers (paid, ~cents per query)

```bash
pip install openai
```
Set `OPENAI_API_KEY` as an environment variable, then in `server.py`:
```python
answer_provider = get_answer_provider(backend="openai")
```

Restart the server after changing this line — it's read once at startup, not per-request.

---

## Running it

**1. Point `test_repo/` at the code you want to query** (copy in your own project's `.py` and `.ipynb` files).

**2. Start the server** (make sure your virtual environment is activated first — see Setup above):

Windows:
```powershell
python server.py
```

macOS / Linux:
```bash
python3 server.py
```

Leave this running — it prints `Open the browser UI at http://localhost:8000/` and serves requests from this terminal. Open a **second** terminal for the next steps (remember to activate the same `.venv` there too if you use the `curl`/CLI steps below).

**3. Index the repo** (run once, or again whenever the code changes):

macOS / Linux / Windows with `curl` (Windows 10+ ships `curl.exe` natively, works in PowerShell too):
```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "./test_repo"}'
```

Windows PowerShell (native alternative, no `curl` needed):
```powershell
$body = @{ repo_path = "./test_repo" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/index" -Method Post -ContentType "application/json" -Body $body
```

**4. Ask questions:**

macOS / Linux / Windows with `curl`:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "where is class imbalance handled?"}'
```

Windows PowerShell:
```powershell
$body = @{ question = "where is class imbalance handled?" } | ConvertTo-Json
$response = Invoke-RestMethod -Uri "http://localhost:8000/ask" -Method Post -ContentType "application/json" -Body $body
$response.answer
```

**Or skip the terminal entirely — use the browser (same on every OS):** with the server running, open `http://localhost:8000/` for the app UI. You can index a repo, ask a question, choose how many retrieved results to use, and see the answer sources in one page.

The FastAPI docs are still available at `http://localhost:8000/docs` if you want to test the raw endpoints.

### Notebook support

Jupyter notebooks are supported for Python code cells. During indexing, each `.ipynb` file is read as notebook JSON, code cells are extracted, and each cell is parsed with the same AST-based function/class extraction used for `.py` files.

Notebook source IDs include the cell number, for example:

```text
analysis.ipynb::cell_3::train_model
analysis.ipynb::cell_5::__module__
```

Common notebook-only commands such as `%matplotlib inline`, `!pip install ...`, and `?helper` are skipped during parsing so the remaining Python code in the cell can still be indexed.

---

## Evaluation

`retrieval_eval.py` measures whether graph-expanded retrieval actually helps, rather than assuming it.

It indexes a small purpose-built fixture repo, `eval_repo/`, into an isolated Chroma DB (`chroma_db_eval/`, separate from your real index) and runs a fixed 21-question set (`eval_questions.json`) against it. `eval_repo/` mirrors this README's own flagship example — a `data_utils.py::undersample()` function that handles class imbalance, called from `prepare_training_data()` — because the real `test_repo/` (a Streamlit churn app) turned out to be almost all module-level script code with no meaningful call graph, and couldn't actually exercise graph expansion at all.

Each question is tagged as one of:
- `flow` — has a known "neighbor" chunk (a caller/callee) that one-hop expansion should surface
- `focused` — a fact/list-style question where expansion should stay off
- `none` — an out-of-scope question with no correct answer, logged qualitatively rather than scored

The script sweeps `top_k` and forces graph expansion on/off (via a `force_expand` override added to `GraphExpandedRetriever.retrieve`) to report `primary_recall` (did the right chunk show up at all?) and `neighbor_recall` (did expansion surface the related chunk?) per configuration:

```
top_k | expand | primary_recall | neighbor_recall | n_primary | n_neighbor
--------------------------------------------------------------------------
    3 |  False |          80.0% |           50.0% |        20 |         12
    3 |   True |          90.0% |           83.3% |        20 |         12
    1 |   True |          80.0% |           83.3% |        20 |         12
    5 |   True |         100.0% |          100.0% |        20 |         12
```

At `top_k=3`, one-hop graph expansion lifts `neighbor_recall` from 50% to 83.3% — a measured answer to "does the call-graph expansion in Phase 4 actually help," not just an assumption.

```bash
python3 retrieval_eval.py        # macOS/Linux
python retrieval_eval.py         # Windows
```

### Answer-quality evaluation (Phase 6b)

`retrieval_eval.py` only proves the right *code* reaches the LLM's context — it says nothing about whether the LLM's final natural-language *answer* is actually correct. `answer_eval.py` closes that gap: it runs the same 20 questions through the real pipeline end to end (retrieval → Ollama-generated answer, same as `/ask`), then checks whether the generated answer text actually names the correct function(s) — a deterministic substring check against identifiers derived from each question's known-correct chunk id, not a second LLM-as-judge call.

```
answer_primary_recall:  75.0%  (15/20)
answer_neighbor_recall: 58.3%  (7/12)
```

Compared against the retrieval-level numbers above (`primary_recall` 90.0%, `neighbor_recall` 83.3% at `top_k=3`), there's a real, measured drop between "the right context was retrieved" and "the final answer actually said it" — a 15-point drop for direct matches, and a larger 25-point drop for graph-expanded neighbors. The most likely cause: the prompt template's own instruction to "ignore nearby code that is not required for the answer" (`answer_providers.py`) appears to sometimes suppress the local 3B model from mentioning a correctly-retrieved neighbor chunk, even when it's sitting right there in context.

This is the honest, useful finding this evaluation was built to surface: **good retrieval does not automatically mean a good final answer** — they are separate failure points and need separate measurement. (Caveat: scoring is a strict identifier-name substring match, so an answer that correctly describes a function without naming it verbatim is scored as a miss — the reported numbers are a lower bound on true answer quality, not an exact one.)

```bash
python3 answer_eval.py        # macOS/Linux
python answer_eval.py         # Windows
```

---

## Semantic Module Chunking

Phase 1's module-level fallback (see "Known limitations") originally captured a whole file's top-level code — everything not inside a function or class — as a single `__module__` chunk, no matter how long or how many unrelated things it mixed together. `test_repo/train.py` is a real example: 147 lines covering data cleaning, feature engineering, a preprocessing pipeline, model training, *and* a separate KMeans customer-segmentation step, all as one chunk with one blurry embedding.

`RepoParser.parse_repo()` now accepts an optional `embedding_provider`. When given, module-level code is first broken into one unit per top-level statement, then merged back together using **the same embedding model already used for retrieval** — no dependency on comment conventions or blank-line formatting. A running centroid of the current group is compared against each new statement's embedding (`SEMANTIC_SPLIT_THRESHOLD`, `ast_parser.py`); similarity below the threshold starts a new chunk. A minimum-lines floor (`SEMANTIC_MIN_GROUP_LINES`) stops single short statements (e.g. `model.fit(X, y)`) from becoming their own low-context chunk before enough of a section has accumulated to judge a real topic change.

The threshold (0.55) was chosen by sweeping 0.50–0.80 against `train.py` and picking the value that produced the most sensible boundaries — same "measure, don't assume" approach as the retrieval eval above, not a guessed constant.

**Result on `test_repo/train.py`:**

```
Before: 1 chunk  (train.py::__module__, lines 14-160, 147 lines)

After:  8 chunks (train.py::__module__::part1 .. part8, 1-43 lines each)
        part1  lines 14-22   setup (paths, load CSV, mkdir)
        part2  lines 24-52   churn label cleanup + feature engineering
        part3  lines 54-96   train/test column split + preprocessing pipeline
        part4  lines 98-110  model pipeline definition
        part5  lines 112-128 fit + save model
        part6  lines 130-140 segmentation column prep
        part7  lines 142-158 KMeans clustering + save
        part8  lines 160-160 final print
```

Every call site (`code_store.py`, `server.py`'s `/index`, `retrieval_eval.py`, `mcp_server.py`'s `index_repo`) now passes its `CodeStore`'s embedding provider through, so this is on by default wherever the project actually indexes a repo. Passing no provider (as `ast_parser.py`'s own `__main__` demo still does) preserves the original single-chunk behavior — this is an additive, backward-compatible change, not a rewrite.

---

## MCP Server

`mcp_server.py` exposes the exact same pipeline (Phases 1-5, unmodified) as an [MCP](https://modelcontextprotocol.io/) server, so any MCP-compatible client — Claude Desktop, Claude Code, or a custom agent — can index a repo and ask questions about it as tool calls, instead of only through the REST API.

Three tools:
- `index_repo(repo_path)` — parse, build the call graph, embed, and store a repo
- `search_code(question, top_k=3)` — raw graph-expanded retrieval, no LLM step; returns source directly so a calling agent can reason over it itself
- `ask_question(question, top_k=3)` — retrieval + LLM-generated answer with citations (same as `/ask`)

Run it directly (stdio transport):
```bash
python3 mcp_server.py        # macOS/Linux
python mcp_server.py         # Windows
```

Register it with an MCP client, e.g. in `claude_desktop_config.json`. Use the Python interpreter **inside this project's virtual environment** so the right dependencies are on the path — MCP clients don't activate `.venv` for you.

macOS / Linux:
```json
{
  "mcpServers": {
    "codebase-rag": {
      "command": "/absolute/path/to/codebase-rag/.venv/bin/python3",
      "args": ["/absolute/path/to/codebase-rag/mcp_server.py"]
    }
  }
}
```

Windows:
```json
{
  "mcpServers": {
    "codebase-rag": {
      "command": "C:\\absolute\\path\\to\\codebase-rag\\.venv\\Scripts\\python.exe",
      "args": ["C:\\absolute\\path\\to\\codebase-rag\\mcp_server.py"]
    }
  }
}
```

---

## Cost

| Component | Backend used | Cost |
|---|---|---|
| Parsing, call graph, storage | Local Python + Chroma | $0 |
| Embeddings | sentence-transformers (local) | $0 (one-time model download) |
| Answer generation | Ollama (local) | $0 |
| Answer generation | Claude/OpenAI API | ~$0.01–0.05 per query |

Running this entirely on the local stack (sentence-transformers + Ollama) costs nothing, indefinitely.

---

## Known limitations

- **Python only.** The parser uses Python's `ast` module, which can parse `.py` files and Python cells in `.ipynb` notebooks, but can't parse C++, Java, or JS. Adding another language means writing a separate Tree-sitter–based parser for it; everything downstream (call graph, storage, retrieval, API) stays the same.
- **Notebook support is code-cell based.** Markdown cells, outputs, images, and non-Python cell magics are not indexed as knowledge sources.
- **Call resolution is name-based, not true static analysis.** If two classes both define a method called `train`, `self.train()` may resolve ambiguously. Real disambiguation would need type inference, which is out of scope here.
- **One-hop graph expansion only.** Two hops tends to pull in loosely-related code and drowns out the actual answer — one hop is a deliberate trade-off between completeness and noise.
- **Module-level code is captured as a single `__module__` chunk per file**, not split further. Fine for typical scripts; a very large script mixing many unrelated concerns at module level would benefit from smarter splitting.
- **TF-IDF fallback (if sentence-transformers can't load) is keyword-based, not semantic.** It works for queries that share vocabulary with docstrings/code, but weaker for paraphrased questions.

---

## Possible extensions

- Tree-sitter–based parser for a second language (C++ is a natural fit given systems background)
- Two-hop or weighted graph expansion instead of fixed one-hop (`retrieval_eval.py` gives a baseline to measure this against)
- Hybrid retrieval (BM25/lexical + semantic) for identifier-exact queries like "which function is called `X`"
- Diff-aware re-indexing on file save (watch `test_repo/` and auto re-index changed files only)
