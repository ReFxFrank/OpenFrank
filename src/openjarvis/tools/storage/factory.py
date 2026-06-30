"""Build a configured memory backend, optionally wrapped with reranking.

A single, tested entry point so the CLI/server get the same composition: the
base retriever named by ``[tools.storage] default_backend`` plus, when
``rerank_enabled`` is set, a cross-encoder rerank stage with a relevance
threshold (see :mod:`openjarvis.tools.storage.rerank`).
"""

from __future__ import annotations

from typing import Optional

from openjarvis.core.registry import MemoryRegistry
from openjarvis.tools.storage._stubs import MemoryBackend


def _create(key: str, db_path: Optional[str]) -> MemoryBackend:
    """Instantiate a registered backend, passing db_path only if accepted."""
    if db_path is not None:
        try:
            return MemoryRegistry.create(key, db_path=db_path)
        except TypeError:
            pass  # backend doesn't take db_path (e.g. faiss/hybrid)
    return MemoryRegistry.create(key)


def build_backend(
    config,  # noqa: ANN001 — JarvisConfig
    *,
    backend: Optional[str] = None,
    db_path: Optional[str] = None,
) -> MemoryBackend:
    """Build the configured memory backend, composing reranking when enabled.

    Returns the bare base backend unless ``[tools.storage] rerank_enabled`` is
    true, in which case it is wrapped in :class:`RerankingMemory` with the
    configured reranker, over-fetch factor, and relevance threshold.
    """
    storage = config.tools.storage
    key = backend or storage.default_backend
    base = _create(key, db_path if db_path is not None else storage.db_path)

    if not getattr(storage, "rerank_enabled", False):
        return base

    from openjarvis.tools.storage.rerank import RerankingMemory, get_reranker

    reranker = get_reranker(storage.rerank_backend, model=storage.rerank_model)
    return RerankingMemory(
        base=base,
        reranker=reranker,
        fetch_multiplier=storage.rerank_fetch_multiplier,
        min_score=storage.rerank_min_score,
    )


__all__ = ["build_backend"]
