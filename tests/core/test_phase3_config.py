"""Tests for Phase 3 config: reranking + local embedding settings."""

from __future__ import annotations

import textwrap

from openjarvis.core.config import (
    StorageConfig,
    load_config,
    validate_config_key,
)


def _reload(monkeypatch, tmp_path, toml: str):
    monkeypatch.delenv("OPENJARVIS_LOCAL_ONLY", raising=False)
    load_config.cache_clear()
    p = tmp_path / "config.toml"
    p.write_text(textwrap.dedent(toml))
    return load_config(path=p)


def test_storage_rerank_defaults():
    s = StorageConfig()
    assert s.rerank_enabled is False  # opt-in
    assert s.rerank_backend == "auto"
    assert s.rerank_min_score == 0.0
    assert s.rerank_fetch_multiplier >= 1


def test_embedding_defaults_are_local():
    s = StorageConfig()
    # Local Ollama embeddings by default — never a hosted embedding API.
    assert s.embedding_engine == "ollama"
    assert s.embedding_model == "nomic-embed-text"


def test_rerank_toml_overlay(monkeypatch, tmp_path):
    cfg = _reload(
        monkeypatch,
        tmp_path,
        """
        [tools.storage]
        default_backend = "sqlite_vec"
        rerank_enabled = true
        rerank_backend = "cross-encoder"
        rerank_min_score = 0.35
        """,
    )
    assert cfg.tools.storage.default_backend == "sqlite_vec"
    assert cfg.tools.storage.rerank_enabled is True
    assert cfg.tools.storage.rerank_backend == "cross-encoder"
    assert cfg.tools.storage.rerank_min_score == 0.35


def test_rerank_keys_are_settable():
    assert validate_config_key("tools.storage.rerank_enabled") is bool
    assert validate_config_key("tools.storage.rerank_min_score") is float
    assert validate_config_key("tools.storage.embedding_model") is str


def test_ollama_embedder_default_is_loopback():
    # The default embedder must target loopback so it passes the local_only
    # egress guard and never reaches a hosted endpoint.
    from openjarvis.tools.storage.embeddings import OllamaEmbedder

    emb = OllamaEmbedder()
    assert emb._base_url.startswith("http://localhost") or emb._base_url.startswith(
        "http://127.0.0.1"
    )
