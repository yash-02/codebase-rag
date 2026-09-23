"""
Phase 3a: Embedding Provider Interface

Defines one interface, multiple backends. CodeStore (phase3_code_store.py)
never needs to know which backend is active -- it just calls provider.embed(texts).

PRIMARY backend: sentence-transformers (all-MiniLM-L6-v2)
    - Local, free, runs on CPU. Downloads once (~80MB) from huggingface.co
      the first time you run it, then works fully offline forever after that.
    - Install: pip install sentence-transformers

FALLBACK backend: TF-IDF (scikit-learn)
    - Fully offline, zero downloads, but weaker (keyword-overlap, not deep
      semantic similarity). Only used automatically if sentence-transformers
      can't load (e.g. you're offline the very first time you run this and
      it can't download the model yet).
    - Install: pip install scikit-learn

To add a paid API backend later (OpenAI, Cohere, etc.), write one more class
implementing `embed()` and add it to `get_embedding_provider()`. Nothing else
in the project needs to change.
"""
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts):
        """Turn a list of strings into a list of embedding vectors."""
        ...


class SentenceTransformerProvider(EmbeddingProvider):
    """Primary backend. Local transformer model, free, no API key."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()


class TfidfProvider(EmbeddingProvider):
    """Offline fallback. Zero downloads, weaker semantic quality."""

    def __init__(self, max_features=384):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self._fitted = False

    def embed(self, texts):
        if not self._fitted:
            vectors = self.vectorizer.fit_transform(texts)
            self._fitted = True
        else:
            vectors = self.vectorizer.transform(texts)
        return vectors.toarray().tolist()


def get_embedding_provider():
    """Try the primary backend first. Fall back automatically if it can't load
    (e.g. no internet yet to download the model on first run)."""
    try:
        provider = SentenceTransformerProvider()
        print("[embedding] using sentence-transformers (primary)")
        return provider
    except Exception as e:
        print(f"[embedding] sentence-transformers unavailable ({e}); falling back to TF-IDF")
        return TfidfProvider()
