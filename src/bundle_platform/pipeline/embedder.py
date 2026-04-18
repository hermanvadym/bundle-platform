# pipeline/embedder.py
"""
Embedding wrapper for diagent RAG pipeline.

Uses fastembed (ONNX Runtime, no PyTorch) with BAAI/bge-small-en-v1.5.
Model is ~33 MB and downloaded to ~/.cache/fastembed/ on first use.

The Embedder is instantiated once per session and shared across the
preprocessing and retrieval steps. Model loading is deferred to the
first embed call so startup is fast even when RAG is not used.
"""

from fastembed import TextEmbedding

from bundle_platform.pipeline.exceptions import RagUnavailable

# Model name and output dimension — update both together if switching models.
MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384  # output dimension of bge-small-en-v1.5


class Embedder:
    """
    Lazy-loading text embedder backed by fastembed.

    Usage:
        embedder = Embedder()
        vectors = embedder.embed_texts(["line one", "line two"])
        query_vec = embedder.embed_query("what caused the crash?")
    """

    def __init__(self, batch_size: int = 64) -> None:
        self._batch_size = batch_size
        self._model: TextEmbedding | None = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns one vector per text."""
        model = self._load_model()
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            vectors.extend([list(map(float, v)) for v in model.embed(batch)])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([text])[0]

    def _load_model(self) -> TextEmbedding:
        if self._model is None:
            try:
                self._model = TextEmbedding(MODEL_NAME)
            except Exception as exc:
                raise RagUnavailable(
                    f"fastembed model '{MODEL_NAME}' unavailable: {exc}"
                ) from exc
        return self._model
