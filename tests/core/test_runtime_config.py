"""Tests for the [runtime] local_only configuration (Phase 1 lockdown)."""

from __future__ import annotations

import textwrap

import pytest

from openjarvis.core.config import (
    JarvisConfig,
    RuntimeConfig,
    generate_default_toml,
    generate_minimal_toml,
    load_config,
)


def _reload(monkeypatch, tmp_path, toml: str | None = None, env: dict | None = None):
    """Load config from a temp file with a clean lru_cache + env."""
    monkeypatch.delenv("OPENJARVIS_LOCAL_ONLY", raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    load_config.cache_clear()
    if toml is None:
        return load_config(path=tmp_path / "missing.toml")
    p = tmp_path / "config.toml"
    p.write_text(textwrap.dedent(toml))
    return load_config(path=p)


def test_default_is_local_only():
    """local_only defaults to True — fully-local is the hard default."""
    assert RuntimeConfig().local_only is True
    assert RuntimeConfig().enforce_egress_guard is True
    assert JarvisConfig().runtime.local_only is True


def test_load_config_default_local_only(monkeypatch, tmp_path):
    cfg = _reload(monkeypatch, tmp_path)
    assert cfg.runtime.local_only is True


def test_toml_can_disable_local_only(monkeypatch, tmp_path):
    cfg = _reload(
        monkeypatch,
        tmp_path,
        """
        [runtime]
        local_only = false
        egress_allowlist = "ollama.lan:11434, 10.0.0.5"
        """,
    )
    assert cfg.runtime.local_only is False
    assert "ollama.lan" in cfg.runtime.egress_allowlist


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "FALSE", "Off"])
def test_env_override_disables(monkeypatch, tmp_path, val):
    """Env var overrides the config-file value (and the default)."""
    cfg = _reload(
        monkeypatch,
        tmp_path,
        "[runtime]\nlocal_only = true\n",
        env={"OPENJARVIS_LOCAL_ONLY": val},
    )
    assert cfg.runtime.local_only is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE"])
def test_env_override_enables(monkeypatch, tmp_path, val):
    cfg = _reload(
        monkeypatch,
        tmp_path,
        "[runtime]\nlocal_only = false\n",
        env={"OPENJARVIS_LOCAL_ONLY": val},
    )
    assert cfg.runtime.local_only is True


def test_env_override_unknown_value_ignored(monkeypatch, tmp_path):
    cfg = _reload(
        monkeypatch,
        tmp_path,
        "[runtime]\nlocal_only = false\n",
        env={"OPENJARVIS_LOCAL_ONLY": "maybe"},
    )
    # Unrecognised env value → config-file value stands.
    assert cfg.runtime.local_only is False


def test_generated_default_config_sets_local_only():
    """The config `jarvis init` writes opts into the local guarantee."""
    hw = JarvisConfig().hardware
    for toml in (generate_default_toml(hw), generate_minimal_toml(hw)):
        assert "[runtime]" in toml
        assert "local_only = true" in toml


def test_generated_default_config_has_no_api_keys():
    """No API-key fields or enabled cloud engine in the default config."""
    hw = JarvisConfig().hardware
    toml = generate_default_toml(hw).lower()
    assert "api_key" not in toml
    assert "sk-" not in toml
    # No *active* [engine.cloud] block (commented provider docs are fine).
    assert "\n[engine.cloud]" not in toml


def test_runtime_is_settable_section():
    """`jarvis config set runtime.local_only false` must validate."""
    from openjarvis.core.config import validate_config_key

    assert validate_config_key("runtime.local_only") is bool
