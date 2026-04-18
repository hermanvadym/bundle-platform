# src/bundle_platform/pipeline/reranker.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CrossEncoderReranker:
    """Re-scores retrieved chunks against a query using a cross-encoder model.

    A cross-encoder reads query+document together (unlike bi-encoders that
    embed them separately), so it captures query-document interaction directly.
    This produces more accurate relevance scores at the cost of running one
    forward pass per candidate — acceptable for re-ranking a small top-K.

    The model is loaded lazily on first use so tests that don't call rerank()
    pay no import cost. This matters because the model download is ~85 MB.

    Why ms-marco-MiniLM-L-6-v2?
    It was trained on MS MARCO passage ranking — a large dataset of web search
    queries — and generalises well to log retrieval. It is small enough to run
    on CPU in reasonable time (~200ms for 20 candidates).
    """

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    _model: object = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, chunks: list[dict], top_n: int) -> list[dict]:
        """Re-rank chunks by relevance to query, return top_n.

        Each chunk must have a "text" key. A "rerank_score" key is added to
        each returned chunk so downstream code can inspect the raw score.

        Args:
            query:   The investigation question.
            chunks:  List of dicts, each with at minimum a "text" key.
            top_n:   How many top-ranked chunks to return.

        Returns:
            The top_n chunks sorted by descending rerank_score.
            Returns [] if chunks is empty.
        """
        if not chunks:
            return []
        self._load()
        model = self._model  # type: ignore[assignment]
        pairs = [(query, chunk.get("text", "")) for chunk in chunks]
        raw = model.predict(pairs)
        scores: list[float] = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        scored = [
            {**chunk, "rerank_score": float(score)}
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        scored.sort(key=lambda c: c["rerank_score"], reverse=True)
        return scored[:top_n]
