"""Cross-encoder reranking for retrieval results (Phase 3, fully local).

First-stage retrievers (BM25, dense, hybrid RRF) are fast but coarse. A reranker
re-scores the top candidates with a model that reads the *query and document
together*, then a **relevance threshold** drops weak matches so only genuinely
relevant context reaches the model — which keeps the prompt (and therefore the
KV-cache VRAM tax) small.

Everything here is local:

* :class:`CrossEncoderReranker` uses a local ``sentence-transformers`` cross
  encoder (e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``, ~80 MB, CPU-friendly).
* :class:`LexicalReranker` is a dependency-free BM25-lite fallback so reranking
  works (and is testable) even without the model — and never reaches the network.

:class:`RerankingMemory` wraps *any* :class:`MemoryBackend`: it over-fetches from
the base retriever and reranks, so it composes with sqlite/FAISS/hybrid alike.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from openjarvis.core.registry import MemoryRegistry
from openjarvis.tools.storage._stubs import MemoryBackend, RetrievalResult

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by для for from has have he her his i in is it its "
    "me my no not of on or that the their them they this to was we were what when "
    "where which who why will with you your".split()
)


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class Reranker(ABC):
    """Scores how relevant each document is to a query (higher = better)."""

    reranker_id: str = "base"

    @abstractmethod
    def score(self, query: str, documents: List[str]) -> List[float]:
        """Return one relevance score per document, aligned with *documents*."""


class LexicalReranker(Reranker):
    """Dependency-free BM25-lite reranker — the offline/no-model fallback.

    Scores by query-term coverage plus a saturating term-frequency bonus, so it
    meaningfully reorders candidates without any model, network, or extra deps.
    Scores are normalised to ``[0, 1]``.
    """

    reranker_id = "lexical"

    def score(self, query: str, documents: List[str]) -> List[float]:
        q_terms = [t for t in dict.fromkeys(_tokens(query)) if t not in _STOPWORDS]
        if not q_terms:
            return [0.0] * len(documents)
        out: List[float] = []
        for doc in documents:
            doc_tokens = _tokens(doc)
            counts: Dict[str, int] = {}
            for t in doc_tokens:
                counts[t] = counts.get(t, 0) + 1
            matched = sum(1 for t in q_terms if t in counts)
            coverage = matched / len(q_terms)
            # Saturating TF (BM25-style) so repeated hits help but plateau.
            tf_bonus = sum(
                counts.get(t, 0) / (counts.get(t, 0) + 1.5) for t in q_terms
            ) / len(q_terms)
            out.append(round(0.7 * coverage + 0.3 * tf_bonus, 6))
        return out


class CrossEncoderReranker(Reranker):
    """Local cross-encoder reranker via ``sentence-transformers``.

    Lazily imports the library and raises a clear, actionable error if it is
    missing (mirrors the embedder/backends). Scores are squashed through a
    sigmoid into ``[0, 1]`` so a single relevance threshold is model-agnostic.
    """

    reranker_id = "cross-encoder"

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with: uv sync --extra memory-faiss  (or pip install "
                "sentence-transformers). The LexicalReranker needs no deps."
            ) from exc
        self._model = CrossEncoder(model_name)

    def score(self, query: str, documents: List[str]) -> List[float]:
        if not documents:
            return []
        raw = self._model.predict([(query, doc) for doc in documents])
        return [1.0 / (1.0 + math.exp(-float(s))) for s in raw]


def get_reranker(name: str = "auto", *, model: str = "") -> Reranker:
    """Build a reranker. ``auto`` tries the cross-encoder, falls back to lexical."""
    key = (name or "auto").strip().lower()
    if key in ("lexical", "bm25"):
        return LexicalReranker()
    if key in ("cross-encoder", "cross_encoder", "ce"):
        return CrossEncoderReranker(model) if model else CrossEncoderReranker()
    # auto
    try:
        return CrossEncoderReranker(model) if model else CrossEncoderReranker()
    except ImportError:
        return LexicalReranker()


def rerank(
    query: str,
    results: List[RetrievalResult],
    *,
    reranker: Reranker,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
) -> List[RetrievalResult]:
    """Re-score *results* with *reranker*, threshold, and truncate.

    The first-stage score is preserved in ``metadata['retrieval_score']`` and
    each result's ``score`` becomes the rerank score. Results below *min_score*
    are dropped (the relevance threshold), then the top *top_k* are returned.
    """
    if not results:
        return []
    scores = reranker.score(query, [r.content for r in results])
    reranked: List[RetrievalResult] = []
    for r, s in zip(results, scores):
        meta = dict(r.metadata)
        meta["retrieval_score"] = r.score
        meta["reranker"] = reranker.reranker_id
        reranked.append(
            RetrievalResult(
                content=r.content, score=float(s), source=r.source, metadata=meta
            )
        )
    reranked.sort(key=lambda r: r.score, reverse=True)
    if min_score is not None:
        reranked = [r for r in reranked if r.score >= min_score]
    if top_k is not None:
        reranked = reranked[:top_k]
    return reranked


@MemoryRegistry.register("reranking")
class RerankingMemory(MemoryBackend):
    """Wraps any base backend with a cross-encoder rerank stage.

    Over-fetches ``top_k * fetch_multiplier`` candidates from *base*, reranks
    them, applies the relevance threshold, and returns the top_k. store / delete
    / clear delegate straight to the base backend.
    """

    backend_id: str = "reranking"

    def __init__(
        self,
        *,
        base: MemoryBackend,
        reranker: Optional[Reranker] = None,
        fetch_multiplier: int = 4,
        min_score: Optional[float] = None,
    ) -> None:
        self._base = base
        self._reranker = reranker or get_reranker("auto")
        self._fetch_multiplier = max(1, fetch_multiplier)
        self._min_score = min_score

    def store(
        self,
        content: str,
        *,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self._base.store(content, source=source, metadata=metadata)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        candidates = self._base.retrieve(query, top_k=top_k * self._fetch_multiplier)
        return rerank(
            query,
            candidates,
            reranker=self._reranker,
            top_k=top_k,
            min_score=self._min_score,
        )

    def delete(self, doc_id: str) -> bool:
        return self._base.delete(doc_id)

    def clear(self) -> None:
        self._base.clear()


__all__ = [
    "CrossEncoderReranker",
    "LexicalReranker",
    "RerankingMemory",
    "Reranker",
    "get_reranker",
    "rerank",
]
