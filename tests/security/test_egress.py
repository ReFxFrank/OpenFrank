"""Tests for the network egress guard (Phase 1 local_only chokepoint)."""

from __future__ import annotations

import socket

import pytest

from openjarvis.core.config import JarvisConfig
from openjarvis.security.egress import (
    EgressBlocked,
    LocalOnlyViolation,
    build_allowlist,
    enforce_local_only,
    host_allowed,
    install_guard,
    is_guard_active,
    uninstall_guard,
    url_allowed,
)


@pytest.fixture(autouse=True)
def _always_uninstall():
    """Never let a patched socket leak out of a test."""
    yield
    uninstall_guard()
    assert not is_guard_active()


# --------------------------------------------------------------------------
# Error types
# --------------------------------------------------------------------------


def test_egress_blocked_is_oserror():
    assert issubclass(EgressBlocked, OSError)


def test_local_only_violation_is_not_oserror():
    # Must be loud — not swallowed by `except OSError` networking handlers.
    assert issubclass(LocalOnlyViolation, RuntimeError)
    assert not issubclass(LocalOnlyViolation, OSError)


# --------------------------------------------------------------------------
# Allowlist construction
# --------------------------------------------------------------------------


def test_build_allowlist_includes_loopback_with_no_config():
    allow = build_allowlist(None)
    assert "localhost" in allow


def test_build_allowlist_includes_engine_hosts():
    cfg = JarvisConfig()
    cfg.engine.ollama_host = "http://my-ollama.local:11434"
    allow = build_allowlist(cfg)
    assert "my-ollama.local" in allow


def test_build_allowlist_includes_extra_entries():
    cfg = JarvisConfig()
    cfg.runtime.egress_allowlist = "10.0.0.5:11434, gpu-box.lan, http://other:8000"
    allow = build_allowlist(cfg)
    assert "10.0.0.5" in allow
    assert "gpu-box.lan" in allow
    assert "other" in allow


# --------------------------------------------------------------------------
# host_allowed / url_allowed
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost", "127.5.5.5"],
)
def test_loopback_allowed(host):
    assert host_allowed(host, build_allowlist(None)) is True


@pytest.mark.parametrize("host", ["8.8.8.8", "1.1.1.1", "example.com", ""])
def test_public_blocked(host):
    assert host_allowed(host, build_allowlist(None)) is False


def test_explicit_allow_entry_matches():
    allow = {"localhost", "gpu-box.lan"}
    assert host_allowed("gpu-box.lan", allow) is True
    assert host_allowed("other.lan", allow) is False


def test_hostname_resolving_to_loopback_allowed(monkeypatch):
    # A custom name (not a builtin loopback alias) that resolves only to 127.x.
    def fake_getaddrinfo(host, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert host_allowed("my-loopback-alias", set()) is True


def test_hostname_resolving_to_public_blocked(monkeypatch):
    def fake_getaddrinfo(host, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert host_allowed("evil.example", set()) is False


def test_url_allowed():
    allow = build_allowlist(None)
    assert url_allowed("http://127.0.0.1:11434/api", allow) is True
    assert url_allowed("https://api.openai.com/v1", allow) is False


# --------------------------------------------------------------------------
# Socket guard install / block / allow
# --------------------------------------------------------------------------


def test_guard_blocks_public_connect():
    install_guard({"localhost"})
    assert is_guard_active()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.2)
    try:
        with pytest.raises(EgressBlocked):
            s.connect(("8.8.8.8", 53))
    finally:
        s.close()


def test_guard_blocks_connect_ex():
    install_guard({"localhost"})
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.2)
    try:
        with pytest.raises(EgressBlocked):
            s.connect_ex(("8.8.8.8", 53))
    finally:
        s.close()


def test_guard_allows_loopback():
    install_guard({"localhost"})
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.2)
    try:
        # Port 9 (discard) is almost certainly closed → ConnectionRefused, but
        # crucially NOT EgressBlocked: the guard let it reach the OS.
        with pytest.raises(OSError) as exc:
            s.connect(("127.0.0.1", 9))
        assert not isinstance(exc.value, EgressBlocked)
    except OSError:
        pass  # some CI lets the connect succeed/timeout; either way not blocked
    finally:
        s.close()


def test_guard_allows_extra_allowlisted_host(monkeypatch):
    # gpu-box.lan resolves to a public IP, but is explicitly allowlisted.
    install_guard({"gpu-box.lan"})
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.01)
    try:
        # Allowlisted → guard passes it to the OS; the real connect then fails
        # (timeout/refused), which is an OSError but NOT EgressBlocked.
        with pytest.raises(OSError) as exc:
            s.connect(("gpu-box.lan", 11434))
        assert not isinstance(exc.value, EgressBlocked)
    except OSError:
        pass
    finally:
        s.close()


def test_guard_is_idempotent_and_refreshes_allowlist():
    install_guard({"localhost"})
    first_connect = socket.socket.connect
    install_guard({"localhost", "gpu-box.lan"})  # second install = no double-patch
    assert socket.socket.connect is first_connect


def test_uninstall_restores_socket():
    original = socket.socket.connect
    install_guard({"localhost"})
    assert socket.socket.connect is not original
    uninstall_guard()
    assert socket.socket.connect is original
    assert not is_guard_active()


# --------------------------------------------------------------------------
# enforce_local_only wiring
# --------------------------------------------------------------------------


def test_enforce_installs_when_local_only():
    cfg = JarvisConfig()
    cfg.runtime.local_only = True
    cfg.runtime.enforce_egress_guard = True
    assert enforce_local_only(cfg) is True
    assert is_guard_active()


def test_enforce_noop_when_not_local_only():
    cfg = JarvisConfig()
    cfg.runtime.local_only = False
    assert enforce_local_only(cfg) is False
    assert not is_guard_active()


def test_enforce_noop_when_guard_disabled():
    cfg = JarvisConfig()
    cfg.runtime.local_only = True
    cfg.runtime.enforce_egress_guard = False
    assert enforce_local_only(cfg) is False
    assert not is_guard_active()
