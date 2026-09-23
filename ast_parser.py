"""
Phase 1: AST Parser

Walks a Python repo and extracts every function and method as a structured
"chunk" — not raw text split by token count. Each chunk keeps its full source,
signature, docstring, and the raw list of names it calls (used in Phase 2 to
build the cross-file call graph).

Why AST instead of regex/text splitting:
Regex-based splitting breaks on nested functions, decorators, multi-line
signatures, etc. The `ast` module gives us Python's own parse tree, so we get
exact boundaries for every function/class regardless of formatting.

HOW TO RUN:
    python3 ast_parser.py
(edit the repo_path at the bottom to point at any Python repo you want to test)
"""
import ast
import json
import math
import os
from dataclasses import dataclass, field

# Similarity threshold for semantic module-level splitting (see
# _group_units_by_similarity). Compared against the running centroid of the
# CURRENT group, not just the previous statement -- comparing to a single
# neighbor was too noisy (single-line statements like `model.fit(X, y)` read
# as "dissimilar" to their neighbors purely from being short, even when they
# belong to the same logical section). Chosen by sweeping 0.50-0.80 against
# test_repo/train.py and picking the value that produced the most sensible
# chunk boundaries (147-line single chunk -> 8 focused chunks of ~5-40 lines,
# roughly matching the file's own logical sections) rather than guessed
# outright -- same "measure, don't assume" approach as phase6_eval.py.
SEMANTIC_SPLIT_THRESHOLD = 0.55

# Never cut a new section until the current group has accumulated at least
# this many source lines -- stops single short statements from becoming their
# own isolated, low-context chunk.
SEMANTIC_MIN_GROUP_LINES = 8


@dataclass
class CodeChunk:
    """One retrievable unit: a single function or method."""
    id: str                  # unique id, e.g. "model.py::ModelTrainer.train"
    name: str                # e.g. "train"
    qualified_name: str      # e.g. "ModelTrainer.train"
    file_path: str
    start_line: int
    end_line: int
    source: str
    docstring: str
    calls: list = field(default_factory=list)   # names this function calls
    called_by: list = field(default_factory=list)  # filled in later, Phase 2


class RepoParser:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def parse_repo(self, embedding_provider=None) -> list[CodeChunk]:
        """Parse every .py and .ipynb file in the repo, return CodeChunks.

        embedding_provider: optional object with an `.embed(texts) -> list[vector]`
        method (see phase3_embedding_providers.py). When given, module-level
        code with no enclosing function (see _make_module_level_chunks) is
        split into multiple semantically coherent chunks instead of one giant
        blob, using embedding similarity between consecutive statements to
        detect topic changes. When omitted, behavior is unchanged: one
        `__module__` chunk per file.
        """
        chunks = []
        for root, _, files in os.walk(self.repo_path):
            for fname in files:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, self.repo_path)
                if fname.endswith(".py"):
                    chunks.extend(self._parse_file(full_path, rel_path, embedding_provider))
                elif fname.endswith(".ipynb"):
                    chunks.extend(self._parse_notebook(full_path, rel_path, embedding_provider))
        return chunks

    def _parse_file(self, full_path: str, rel_path: str, embedding_provider=None) -> list[CodeChunk]:
        with open(full_path, "r", encoding="utf-8") as f:
            source_text = f.read()

        return self._parse_python_source(source_text, rel_path, embedding_provider)

    def _parse_notebook(self, full_path: str, rel_path: str, embedding_provider=None) -> list[CodeChunk]:
        """Extract Python code cells from a Jupyter notebook and parse each
        cell as if it were a small Python source file."""
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                notebook = json.load(f)
        except (OSError, json.JSONDecodeError):
            return []

        chunks = []
        for cell_number, cell in enumerate(notebook.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue

            source = cell.get("source", "")
            if isinstance(source, list):
                source_text = "".join(source)
            else:
                source_text = source

            if not source_text.strip():
                continue

            cell_path = f"{rel_path}::cell_{cell_number}"
            source_text = self._prepare_notebook_source(source_text)
            chunks.extend(self._parse_python_source(source_text, cell_path, embedding_provider))

        return chunks

    def _prepare_notebook_source(self, source_text: str) -> str:
        """Comment out common notebook-only commands while preserving line
        numbers, so the remaining Python code can still be parsed with AST."""
        cleaned_lines = []
        for line in source_text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("%", "!", "?")):
                cleaned_lines.append("# Notebook command skipped by parser: " + line)
            else:
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _parse_python_source(self, source_text: str, rel_path: str, embedding_provider=None) -> list[CodeChunk]:
        try:
            tree = ast.parse(source_text, filename=rel_path)
        except SyntaxError:
            # Skip files/cells that don't parse (e.g. shell magic, Python 2)
            return []

        source_lines = source_text.splitlines()
        chunks = []

        # Walk top level: capture both free functions and class methods
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        chunks.append(
                            self._make_chunk(sub, source_lines, rel_path, parent_class=node.name)
                        )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Only top-level functions here; methods handled above via ClassDef
                if self._is_top_level(node, tree):
                    chunks.append(
                        self._make_chunk(node, source_lines, rel_path, parent_class=None)
                    )

        # NEW: also capture top-level code that isn't inside any function/class.
        # Streamlit apps, notebooks-turned-scripts, and straight-line training
        # scripts often put most of their logic at module level -- without this,
        # such files would silently produce zero chunks.
        chunks.extend(
            self._make_module_level_chunks(tree, source_lines, rel_path, embedding_provider)
        )

        return chunks

    def _collect_module_statement_units(self, tree, source_lines, rel_path):
        """Break module-scope code (not def/class bodies, not imports) into one
        unit per top-level statement -- the raw material both the single-chunk
        path and the semantic-splitting path build from."""
        units = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue

            start = node.lineno
            end = getattr(node, "end_lineno", start)
            source = "\n".join(source_lines[start - 1:end])

            calls = []
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    if isinstance(sub.func, ast.Name):
                        calls.append(sub.func.id)
                    elif isinstance(sub.func, ast.Attribute):
                        calls.append(sub.func.attr)

            units.append({"start": start, "end": end, "source": source, "calls": calls})

        return units

    def _cosine_similarity(self, a, b) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _average_vector(self, vectors):
        length = len(vectors[0])
        return [sum(v[i] for v in vectors) / len(vectors) for i in range(length)]

    def _group_units_by_similarity(self, units, embedding_provider):
        """Embed each top-level statement and cut a new group whenever the
        next statement drifts too far from the CURRENT GROUP'S centroid --
        this is the semantic-chunking alternative to guessing boundaries from
        comment headers or blank lines: it reuses the same embedding model
        already used for retrieval to detect an actual topic change in the
        code, rather than a text convention. A minimum-lines floor
        (SEMANTIC_MIN_GROUP_LINES) stops short individual statements from
        becoming their own low-context chunk before enough of the section has
        accumulated to judge a real topic change."""
        embeddings = embedding_provider.embed([u["source"] for u in units])

        groups = [[units[0]]]
        group_embeddings = [[embeddings[0]]]

        for i in range(1, len(units)):
            current_group = groups[-1]
            current_lines = sum(u["end"] - u["start"] + 1 for u in current_group)
            centroid = self._average_vector(group_embeddings[-1])
            similarity = self._cosine_similarity(centroid, embeddings[i])

            if similarity >= SEMANTIC_SPLIT_THRESHOLD or current_lines < SEMANTIC_MIN_GROUP_LINES:
                current_group.append(units[i])
                group_embeddings[-1].append(embeddings[i])
            else:
                groups.append([units[i]])
                group_embeddings.append([embeddings[i]])

        return groups

    def _merge_units_into_chunk(self, group, rel_path, module_docstring, section_index, total_sections, source_lines):
        lines = set()
        calls = []
        for unit in group:
            for line_no in range(unit["start"], unit["end"] + 1):
                lines.add(line_no)
            calls.extend(unit["calls"])

        sorted_lines = sorted(lines)
        source = "\n".join(source_lines[i - 1] for i in sorted_lines)

        if total_sections == 1:
            chunk_id = f"{rel_path}::__module__"
            name = qualified_name = "__module__"
            docstring = module_docstring or "(top-level script code, not inside any function)"
        else:
            chunk_id = f"{rel_path}::__module__::part{section_index + 1}"
            name = qualified_name = f"__module__.part{section_index + 1}"
            docstring = (
                f"(top-level script code, section {section_index + 1} of {total_sections}, "
                f"split from the rest of the module by semantic similarity)"
            )
            if section_index == 0 and module_docstring:
                docstring = f"{module_docstring}\n{docstring}"

        return CodeChunk(
            id=chunk_id,
            name=name,
            qualified_name=qualified_name,
            file_path=rel_path,
            start_line=sorted_lines[0],
            end_line=sorted_lines[-1],
            source=source,
            docstring=docstring,
            calls=sorted(set(calls)),
        )

    def _make_module_level_chunks(self, tree, source_lines, rel_path, embedding_provider=None) -> list:
        """Collect statements at module scope into one or more chunks
        representing 'the script itself'. Returns [] if there's nothing
        meaningful (e.g. a file that's only imports, or only functions).

        Without an embedding_provider: exactly one __module__ chunk (original
        behavior). With one: statements are grouped by embedding similarity
        first, so a file mixing unrelated concerns at module level (e.g. a
        training script that also does clustering) is split into multiple
        smaller, more focused chunks instead of one blurry one -- see
        SEMANTIC_SPLIT_THRESHOLD above and README's "Semantic module chunking"
        section for the measured effect."""
        units = self._collect_module_statement_units(tree, source_lines, rel_path)
        if not units:
            return []

        module_docstring = ast.get_docstring(tree) or ""

        if embedding_provider is None or len(units) == 1:
            groups = [units]
        else:
            groups = self._group_units_by_similarity(units, embedding_provider)

        return [
            self._merge_units_into_chunk(
                group, rel_path, module_docstring, i, len(groups), source_lines
            )
            for i, group in enumerate(groups)
        ]

    def _is_top_level(self, func_node, tree) -> bool:
        """Check this function isn't nested inside a class (avoids double-counting methods)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if func_node in ast.walk(node) and func_node is not node:
                    if func_node in node.body:
                        return False
        return True

    def _make_chunk(self, func_node, source_lines, rel_path, parent_class) -> CodeChunk:
        qualified_name = f"{parent_class}.{func_node.name}" if parent_class else func_node.name
        chunk_id = f"{rel_path}::{qualified_name}"

        start = func_node.lineno
        end = func_node.end_lineno
        source = "\n".join(source_lines[start - 1:end])

        docstring = ast.get_docstring(func_node) or ""

        calls = self._extract_calls(func_node)

        return CodeChunk(
            id=chunk_id,
            name=func_node.name,
            qualified_name=qualified_name,
            file_path=rel_path,
            start_line=start,
            end_line=end,
            source=source,
            docstring=docstring,
            calls=calls,
        )

    def _extract_calls(self, func_node) -> list[str]:
        """Find every function/method name called inside this function body."""
        calls = []
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
        return sorted(set(calls))


if __name__ == "__main__":
    # Point this at any repo you want to test (start with a small one of your own)
    parser = RepoParser("./test_repo")
    chunks = parser.parse_repo()
    for c in chunks:
        print(f"\n--- {c.id} (lines {c.start_line}-{c.end_line}) ---")
        print(f"docstring: {c.docstring}")
        print(f"calls: {c.calls}")
