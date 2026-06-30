"""Tests for network-capability classification (Phase 4 local_only gating)."""

from __future__ import annotations

from openjarvis.security.capabilities import (
    DEFAULT_TOOL_CAPABILITIES,
    NETWORK_CAPABILITIES,
    Capability,
    requires_network,
)


def test_network_capabilities_set():
    assert "network:fetch" in NETWORK_CAPABILITIES
    assert Capability.NETWORK_FETCH.value in NETWORK_CAPABILITIES
    assert "channel:send" in NETWORK_CAPABILITIES
    # File/code/memory capabilities are NOT network.
    assert "file:read" not in NETWORK_CAPABILITIES
    assert "code:execute" not in NETWORK_CAPABILITIES


def test_requires_network():
    assert requires_network(["network:fetch"]) is True
    assert requires_network(["file:read", "network:fetch"]) is True
    assert requires_network([Capability.NETWORK_FETCH]) is True  # str-enum member
    assert requires_network(["file:read", "code:execute"]) is False
    assert requires_network([]) is False
    assert requires_network(None) is False


def test_default_tool_capabilities_flag_network_tools():
    for tool in ("web_search", "http_request", "browser"):
        caps = DEFAULT_TOOL_CAPABILITIES[tool]
        assert requires_network([c.value for c in caps])
    # Local tools are not network.
    assert not requires_network(
        [c.value for c in DEFAULT_TOOL_CAPABILITIES["file_read"]]
    )
