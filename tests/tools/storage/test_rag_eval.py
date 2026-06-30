"""Small offline RAG eval: reranking must improve ranking quality (Phase 3).

A first-stage retriever that returns candidates in a poor order should be
measurably improved by the cross-encoder rerank stage. We assert Mean
Reciprocal Rank (MRR) and precision@1 go up after reranking — the brief's
"retrieval quality improves on a small RAG eval" acceptance, run fully offline
with the dependency-free LexicalReranker.
"""

from __future__ import annotations

from typing import List

from openjarvis.tools.storage._stubs import RetrievalResult
from openjarvis.tools.storage.rerank import LexicalReranker, rerank

# (query, relevant-doc-substring)
CORPUS = [
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
    "Python is a high-level programming language known for readable syntax.",
    "Photosynthesis lets plants convert sunlight into chemical energy.",
    "The mitochondrion is the powerhouse of the cell, producing ATP.",
    "Mount Everest is the highest mountain above sea level, in the Himalayas.",
    "HTTP is the protocol that powers data exchange on the world wide web.",
]
QUERIES = [
    ("which tower is in Paris France", "Eiffel"),
    ("readable programming language syntax", "Python"),
    ("how do plants convert sunlight energy", "Photosynthesis"),
    ("what produces ATP in the cell", "powerhouse"),
    ("highest mountain in the Himalayas", "Everest"),
    ("protocol for the world wide web", "HTTP"),
]


def _reciprocal_rank(results: List[RetrievalResult], needle: str) -> float:
    for i, r in enumerate(results):
        if needle.lower() in r.content.lower():
            return 1.0 / (i + 1)
    return 0.0


def _baseline_order(query: str) -> List[RetrievalResult]:
    """A deliberately poor first stage: fixed insertion order, flat scores."""
    return [RetrievalResult(content=c, score=1.0) for c in CORPUS]


def test_reranking_improves_mrr_and_precision_at_1():
    reranker = LexicalReranker()
    base_rr: List[float] = []
    reranked_rr: List[float] = []
    base_p1 = 0
    reranked_p1 = 0

    for query, needle in QUERIES:
        base = _baseline_order(query)
        reranked = rerank(query, base, reranker=reranker)

        base_rr.append(_reciprocal_rank(base, needle))
        reranked_rr.append(_reciprocal_rank(reranked, needle))
        base_p1 += int(needle.lower() in base[0].content.lower())
        reranked_p1 += int(needle.lower() in reranked[0].content.lower())

    base_mrr = sum(base_rr) / len(base_rr)
    reranked_mrr = sum(reranked_rr) / len(reranked_rr)

    # Reranking must strictly improve MRR over the poor baseline ordering.
    assert reranked_mrr > base_mrr
    # And it should put the right doc first for most queries.
    assert reranked_p1 >= base_p1
    assert reranked_p1 >= len(QUERIES) - 1  # at most one miss tolerated


def test_reranking_perfect_on_this_fixture():
    """On this fixture the lexical reranker should rank every answer #1."""
    reranker = LexicalReranker()
    for query, needle in QUERIES:
        reranked = rerank(query, _baseline_order(query), reranker=reranker, top_k=1)
        assert reranked and needle.lower() in reranked[0].content.lower()
