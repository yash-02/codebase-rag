"""
Phase 5a: Answer Provider Interface

Same pattern as the embedding providers in Phase 3: one interface, swappable
backends. The FastAPI app never needs to know which one is active.

PRIMARY (recommended to start): OllamaProvider
    - Fully local, free, no API key. Requires Ollama installed and running
      (https://ollama.com), with a model pulled, e.g.:
          ollama pull llama3.1:8b
    - Works fully offline once the model is downloaded.

PAID API option: AnthropicProvider
    - Higher quality answers, costs a small amount per query (see cost
      breakdown discussed earlier -- a few cents for typical usage).
    - Requires: pip install anthropic, and an ANTHROPIC_API_KEY env var.

PAID API option: OpenAIProvider
    - Same trade-offs as AnthropicProvider (higher quality, small per-query
      cost) -- an alternative provider for teams already on OpenAI.
    - Requires: pip install openai, and an OPENAI_API_KEY env var.

STUB (no dependencies, no setup): StubProvider
    - Doesn't call any LLM at all -- just formats the retrieved context
      directly. Useful to confirm the whole pipeline (retrieval + API)
      works end-to-end before you set up Ollama or an API key.

Swap which one is active by changing ONE line in phase5_api.py.
"""
from abc import ABC, abstractmethod


class AnswerProvider(ABC):
    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        """Given a question and retrieved code context, return a natural
        language answer."""
        ...


PROMPT_TEMPLATE = """You are a precise codebase assistant.
Use ONLY the provided code context. Do not guess or use outside knowledge.

Answer rules:
- Answer only the user's exact question.
- Prefer exact identifiers, variable names, column names, functions, and file ids
  from the context.
- Ignore nearby code that is not required for the answer.
- If the question asks "what are", "which", "list", "inputs", "variables",
  "features", "columns", or "parameters", return only the relevant names in
  bullets, grouped by file/page/function when useful.
- Do not explain what the model, clustering, or surrounding app does unless the
  user specifically asks.
- If the context does not contain the answer, say what is missing in one short
  sentence.
- Keep the answer concise: maximum 8 bullets or 120 words.
- Use Markdown, but do not add fixed template headings.

Code context:
{context}

Question: {question}

Answer:"""


class OllamaProvider(AnswerProvider):
    """Local, free. Requires Ollama running at localhost:11434."""

    def __init__(self, model: str = "llama3.2:3b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def generate(self, question: str, context: str) -> str:
        import requests
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        response = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()


class AnthropicProvider(AnswerProvider):
    """Paid API. Higher quality, costs a small amount per query."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.model = model

    def generate(self, question: str, context: str) -> str:
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


class OpenAIProvider(AnswerProvider):
    """Paid API. Alternative to AnthropicProvider, same trade-offs (higher
    quality, small cost per query). Requires: pip install openai, and an
    OPENAI_API_KEY env var."""

    def __init__(self, model: str = "gpt-4o-mini"):
        import openai
        self.client = openai.OpenAI()  # reads OPENAI_API_KEY from env
        self.model = model

    def generate(self, question: str, context: str) -> str:
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()


class StubProvider(AnswerProvider):
    """No dependencies, no setup. Doesn't call any LLM -- just presents the
    retrieved context in a readable way, so you can confirm retrieval is
    working correctly before wiring up a real LLM."""

    def generate(self, question: str, context: str) -> str:
        return (
            f"## Short Answer\n"
            f"- No LLM is connected, so this is the retrieved context instead of a generated answer.\n"
            f"- Connect Ollama or Claude to get a plain-English answer for: \"{question}\"\n\n"
            f"## Evidence\n"
            f"- The sections below are the exact code chunks that would be sent to the LLM.\n\n"
            f"## How It Works\n"
            f"1. The retriever found direct semantic matches for the question.\n"
            f"2. It expanded those matches through the call graph where related callers or callees were available.\n"
            f"3. The answer provider would normally summarize this context into the structured sections shown here.\n\n"
            f"## Limitations\n"
            f"- This stub output does not interpret the code. It only displays retrieved context.\n\n"
            f"## Sources\n"
            f"- See retrieved context ids below.\n\n"
            f"## Retrieved Context\n"
            f"```text\n{context}\n```"
        )


def get_answer_provider(backend: str = "stub") -> AnswerProvider:
    """backend: 'ollama', 'anthropic', 'openai', or 'stub'"""
    if backend == "ollama":
        return OllamaProvider()
    elif backend == "anthropic":
        return AnthropicProvider()
    elif backend == "openai":
        return OpenAIProvider()
    else:
        return StubProvider()
