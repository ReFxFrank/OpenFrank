"""ToolExecutor airgap gate — network tools blocked in local_only (Phase 4)."""

from __future__ import annotations

from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import ToolCall, ToolResult
from openjarvis.tools._stubs import BaseTool, ToolExecutor, ToolSpec


class _NetworkTool(BaseTool):
    tool_id = "net"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="net_tool",
            description="reaches the network",
            required_capabilities=["network:fetch"],
        )

    def execute(self, **params) -> ToolResult:  # noqa: ANN003
        return ToolResult(tool_name="net_tool", content="fetched", success=True)


class _LocalTool(BaseTool):
    tool_id = "loc"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="local_tool",
            description="local only",
            required_capabilities=["file:read"],
        )

    def execute(self, **params) -> ToolResult:  # noqa: ANN003
        return ToolResult(tool_name="local_tool", content="read", success=True)


def _call(name: str) -> ToolCall:
    return ToolCall(id="1", name=name, arguments="{}")


def test_network_tool_blocked_in_local_only_and_logged():
    bus = EventBus(record_history=True)
    ex = ToolExecutor([_NetworkTool()], bus, local_only=True)
    result = ex.execute(_call("net_tool"))
    assert result.success is False
    assert "local_only" in result.content
    # The block is logged on the event bus.
    blocks = [e for e in bus.history if e.event_type == EventType.SECURITY_BLOCK]
    assert blocks
    assert blocks[0].data["reason"] == "local_only"
    assert blocks[0].data["tool"] == "net_tool"
    assert "network:fetch" in blocks[0].data["capabilities"]


def test_local_tool_allowed_in_local_only():
    ex = ToolExecutor([_LocalTool()], local_only=True)
    result = ex.execute(_call("local_tool"))
    assert result.success is True
    assert result.content == "read"


def test_network_tool_allowed_when_not_local_only():
    ex = ToolExecutor([_NetworkTool()], local_only=False)
    result = ex.execute(_call("net_tool"))
    assert result.success is True
    assert result.content == "fetched"


def test_default_executor_is_not_local_only():
    # Backwards-compatible default: no gating unless explicitly enabled.
    ex = ToolExecutor([_NetworkTool()])
    assert ex.execute(_call("net_tool")).success is True
