"""Tests for cross-encoder reranking (Phase 3)."""

from __future__ import annotations

from typing import List

from openjarvis.core.config import JarvisConfig
from openjarvis.tools.storage._stubs import MemoryBackend, RetrievalResult
from openjarvis.tools.storage.factory import build_backend
from openjarvis.tools.storage.rerank import (
    CrossEncoderReranker,
    LexicalReranker,
    RerankingMemory,
    get_reranker,
    rerank,
)


class _ListBackend(MemoryBackend):
    """Minimal base backend that returns docs in INSERTION order (not relevance)."""

    backend_id = "list"

    def __init__(self, db_path: str | None = None) -> None:
        self._docs: List[RetrievalResult] = []

    def store(self, content, *, source="", metadata=None):  # noqa: ANN001
        self._docs.append(
            RetrievalResult(content=content, source=source, metadata=metadata or {})
        )
        return str(len(self._docs))

    def retrieve(self, query, *, top_k=5, **kwargs):  # noqa: ANN001
        # Deliberately return in insertion order with flat scores so the only
        # thing that can reorder by relevance is the reranker.
        out = [
            RetrievalResult(
                content=d.content, score=1.0, source=d.source, metadata=dict(d.metadata)
            )
            for d in self._docs
        ]
        return out[:top_k]

    def delete(self, doc_id):  # noqa: ANN001
        return False

    def clear(self):
        self._docs.clear()


# --------------------------------------------------------------------------
# LexicalReranker
# --------------------------------------------------------------------------


def test_lexical_reranker_scores_relevant_higher():
    r = LexicalReranker()
    docs = [
        "the cat sat on the mat",
        "quantum chromodynamics and gluon fields",
        "a feline rested on a rug",  # synonyms, fewer exact terms
    ]
    scores = r.score("where did the cat sit", docs)
    assert scores[0] == max(scores)  # exact-term doc wins
    assert scores[0] > scores[1]


def test_lexical_reranker_empty_query():
    assert LexicalReranker().score("the a of", ["anything"]) == [0.0]


def test_lexical_reranker_scores_in_unit_range():
    scores = LexicalReranker().score("cat mat", ["the cat sat on the mat"])
    assert 0.0 <= scores[0] <= 1.0


# --------------------------------------------------------------------------
# rerank() — reorder, threshold, truncate, metadata
# --------------------------------------------------------------------------


def _results(*contents):
    return [RetrievalResult(content=c, score=1.0) for c in contents]


def test_rerank_reorders_by_relevance():
    results = _results(
        "completely unrelated text about gardening",
        "the python programming language and its syntax",
    )
    out = rerank("python programming syntax", results, reranker=LexicalReranker())
    assert out[0].content.startswith("the python")


def test_rerank_preserves_first_stage_score_in_metadata():
    results = [RetrievalResult(content="python syntax", score=0.42)]
    out = rerank("python", results, reranker=LexicalReranker())
    assert out[0].metadata["retrieval_score"] == 0.42
    assert out[0].metadata["reranker"] == "lexical"


def test_rerank_applies_min_score_threshold():
    results = _results("python programming", "unrelated gardening tips")
    out = rerank(
        "python programming", results, reranker=LexicalReranker(), min_score=0.5
    )
    # Only the relevant doc clears the threshold.
    assert all(r.score >= 0.5 for r in out)
    assert any("python" in r.content for r in out)


def test_rerank_truncates_top_k():
    results = _results("python a", "python b", "python c")
    out = rerank("python", results, reranker=LexicalReranker(), top_k=2)
    assert len(out) == 2


def test_rerank_empty_results():
    assert rerank("q", [], reranker=LexicalReranker()) == []


# --------------------------------------------------------------------------
# RerankingMemory wrapper
# --------------------------------------------------------------------------


def test_reranking_memory_reorders_base_results():
    base = _ListBackend()
    base.store("irrelevant chatter about the weather")
    base.store("the capital of France is Paris")
    wrapped = RerankingMemory(base=base, reranker=LexicalReranker())
    out = wrapped.retrieve("what is the capital of France", top_k=2)
    assert out[0].content == "the capital of France is Paris"


def test_reranking_memory_threshold_filters():
    base = _ListBackend()
    base.store("totally unrelated content")
    base.store("Paris France capital city")
    wrapped = RerankingMemory(base=base, reranker=LexicalReranker(), min_score=0.4)
    out = wrapped.retrieve("Paris France capital", top_k=5)
    assert all(r.score >= 0.4 for r in out)
    assert out and "Paris" in out[0].content


def test_reranking_memory_delegates_store_delete_clear():
    base = _ListBackend()
    wrapped = RerankingMemory(base=base, reranker=LexicalReranker())
    wrapped.store("x")
    assert len(base._docs) == 1
    wrapped.clear()
    assert len(base._docs) == 0


# --------------------------------------------------------------------------
# get_reranker factory + build_backend composition
# --------------------------------------------------------------------------


def test_get_reranker_lexical_and_auto_fallback():
    assert isinstance(get_reranker("lexical"), LexicalReranker)
    # Without sentence-transformers installed, "auto" falls back to lexical.
    assert isinstance(get_reranker("auto"), (LexicalReranker, CrossEncoderReranker))


def test_build_backend_wraps_when_rerank_enabled():
    from openjarvis.core.registry import MemoryRegistry

    MemoryRegistry.register_value("list", _ListBackend)
    cfg = JarvisConfig()
    cfg.tools.storage.default_backend = "list"
    cfg.tools.storage.rerank_enabled = True
    cfg.tools.storage.rerank_backend = "lexical"
    backend = build_backend(cfg)
    assert isinstance(backend, RerankingMemory)


def test_build_backend_bare_when_rerank_disabled():
    from openjarvis.core.registry import MemoryRegistry

    MemoryRegistry.register_value("list", _ListBackend)
    cfg = JarvisConfig()
    cfg.tools.storage.default_backend = "list"
    cfg.tools.storage.rerank_enabled = False
    backend = build_backend(cfg)
    assert isinstance(backend, _ListBackend)
