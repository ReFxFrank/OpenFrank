"""Tests for the persistent sqlite-vec memory backend (Phase 3).

Uses a deterministic hashing embedder so the tests run fully offline (no Ollama).
Skipped if the sqlite-vec extension is not installed.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip(
    "sqlite_vec", reason="sqlite-vec not installed (uv sync --extra memory-sqlite-vec)"
)
np = pytest.importorskip("numpy")

from openjarvis.tools.storage.embeddings import Embedder  # noqa: E402
from openjarvis.tools.storage.sqlite_vec_backend import SqliteVecMemory  # noqa: E402

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder — similar text → similar vec."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts):  # noqa: ANN001
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _TOKEN.findall(t.lower()):
                out[i, hash(tok) % self._dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return out / norms

    def dim(self) -> int:
        return self._dim


@pytest.fixture
def mem():
    return SqliteVecMemory(db_path=":memory:", embedder=HashingEmbedder())


def test_store_and_retrieve_relevant_first(mem):
    mem.store("the capital of France is Paris", source="geo")
    mem.store("photosynthesis converts light into chemical energy", source="bio")
    out = mem.retrieve("what is the capital of France", top_k=2)
    assert out
    assert out[0].content == "the capital of France is Paris"
    assert out[0].source == "geo"
    assert out[0].score > 0


def test_metadata_roundtrips(mem):
    mem.store("hello world", metadata={"tag": "greeting", "n": 3})
    out = mem.retrieve("hello", top_k=1)
    assert out[0].metadata["tag"] == "greeting"
    assert out[0].metadata["n"] == 3


def test_delete(mem):
    doc_id = mem.store("deletable content here")
    assert mem.delete(doc_id) is True
    assert mem.delete(doc_id) is False
    assert mem.retrieve("deletable content", top_k=5) == []


def test_clear(mem):
    mem.store("one")
    mem.store("two")
    mem.clear()
    assert mem.retrieve("one", top_k=5) == []


def test_retrieve_empty_store(mem):
    assert mem.retrieve("anything", top_k=5) == []


def test_persistence_survives_restart(tmp_path):
    db = str(tmp_path / "mem.db")
    m1 = SqliteVecMemory(db_path=db, embedder=HashingEmbedder())
    m1.store("the mitochondria is the powerhouse of the cell", source="bio")
    m1.close()

    # Reopen the same file — content must still be retrievable.
    m2 = SqliteVecMemory(db_path=db, embedder=HashingEmbedder())
    out = m2.retrieve("mitochondria powerhouse cell", top_k=1)
    assert out
    assert "mitochondria" in out[0].content
    m2.close()
