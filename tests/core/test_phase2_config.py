"""Tests for Phase 2 config: [offload] and [router] sections."""

from __future__ import annotations

import textwrap

from openjarvis.core.config import (
    JarvisConfig,
    OffloadConfig,
    RouterConfig,
    generate_default_toml,
    load_config,
    validate_config_key,
)


def _reload(monkeypatch, tmp_path, toml: str):
    monkeypatch.delenv("OPENJARVIS_LOCAL_ONLY", raising=False)
    load_config.cache_clear()
    p = tmp_path / "config.toml"
    p.write_text(textwrap.dedent(toml))
    return load_config(path=p)


def test_offload_defaults():
    o = OffloadConfig()
    assert o.profile == "auto"
    assert o.flash_attention is True
    assert o.kv_cache_quant == "q8"
    assert o.resident_reserve_gb > 0


def test_router_defaults():
    r = RouterConfig()
    assert r.enabled is False  # opt-in, never silently overrides -m
    assert r.fast_model and r.balanced_model and r.deep_model
    assert r.fast_max_score < r.deep_min_score


def test_jarvis_config_has_offload_and_router():
    cfg = JarvisConfig()
    assert isinstance(cfg.offload, OffloadConfig)
    assert isinstance(cfg.router, RouterConfig)


def test_offload_router_toml_overlay(monkeypatch, tmp_path):
    cfg = _reload(
        monkeypatch,
        tmp_path,
        """
        [offload]
        profile = "gaming"
        multitask_budget_gb = 8.5

        [router]
        enabled = true
        deep_model = "llama4-scout:17b"
        """,
    )
    assert cfg.offload.profile == "gaming"
    assert cfg.offload.multitask_budget_gb == 8.5
    assert cfg.router.enabled is True
    assert cfg.router.deep_model == "llama4-scout:17b"


def test_offload_router_keys_are_settable():
    assert validate_config_key("offload.profile") is str
    assert validate_config_key("router.enabled") is bool
    assert validate_config_key("offload.safety_margin_gb") is float


def test_generated_config_includes_offload_and_router():
    toml = generate_default_toml(JarvisConfig().hardware)
    assert "[offload]" in toml
    assert "[router]" in toml
    assert 'profile = "auto"' in toml
