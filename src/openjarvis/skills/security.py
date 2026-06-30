"""Skill security — capability validation and trust tiers."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from openjarvis.security.capabilities import NETWORK_CAPABILITIES, requires_network
from openjarvis.skills.types import SkillManifest

DANGEROUS_CAPABILITIES: frozenset[str] = frozenset(
    {"shell:execute", "network:listen", "filesystem:write"}
)


class TrustTier(str, Enum):
    """Trust tier for a skill, ordered from most to least trusted."""

    BUNDLED = "bundled"
    INDEXED = "indexed"
    WORKSPACE = "workspace"
    UNREVIEWED = "unreviewed"


def classify_trust_tier(
    *,
    is_bundled: bool = False,
    is_workspace: bool = False,
    has_signature: bool = False,
    in_index: bool = False,
) -> TrustTier:
    """Return the trust tier for a skill based on its provenance.

    Priority (highest to lowest): bundled > workspace > indexed > unreviewed.
    """
    if is_bundled:
        return TrustTier.BUNDLED
    if is_workspace:
        return TrustTier.WORKSPACE
    if has_signature and in_index:
        return TrustTier.INDEXED
    return TrustTier.UNREVIEWED


def validate_capabilities(manifest: SkillManifest, allowed: Set[str]) -> List[str]:
    """Return a list of capabilities required by *manifest* that are not in *allowed*.

    An empty list means the manifest is fully authorized.
    """
    return [cap for cap in manifest.required_capabilities if cap not in allowed]


def has_dangerous_capabilities(manifest: SkillManifest) -> List[str]:
    """Return the subset of *manifest*'s required capabilities that are dangerous."""
    return [
        cap for cap in manifest.required_capabilities if cap in DANGEROUS_CAPABILITIES
    ]


def _tool_capability_map(
    overrides: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[str]]:
    """Map tool name → required capabilities.

    Prefers live ``ToolSpec.required_capabilities`` from the registry, falling
    back to the static ``DEFAULT_TOOL_CAPABILITIES`` table so classification
    works even before tools are instantiated (and in tests). *overrides* win.
    """
    from openjarvis.security.capabilities import DEFAULT_TOOL_CAPABILITIES

    # Normalise to the capability *value* string ("network:fetch"); Capability is
    # a str-Enum whose str() is "Capability.NETWORK_FETCH", so use .value.
    def _cap_str(c: object) -> str:
        return str(getattr(c, "value", c))

    caps: Dict[str, List[str]] = {
        name: [_cap_str(c) for c in cap_list]
        for name, cap_list in DEFAULT_TOOL_CAPABILITIES.items()
    }
    try:
        from openjarvis.core.registry import ToolRegistry

        for name, entry in ToolRegistry.items():
            spec = getattr(entry, "spec", None)
            req = getattr(spec, "required_capabilities", None)
            if req and not isinstance(req, property):
                caps[name] = [_cap_str(c) for c in req]
    except Exception:  # registry not populated / import edge — fall back to defaults
        pass
    if overrides:
        caps.update(overrides)
    return caps


def skill_network_capabilities(
    manifest: SkillManifest,
    tool_capabilities: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """Return the network capabilities a skill needs (declared or via its tools).

    A skill rarely declares ``required_capabilities`` directly; its network
    surface comes from the tools its steps invoke (e.g. ``web_search`` →
    ``network:fetch``). This inspects both.
    """
    caps_map = (
        tool_capabilities if tool_capabilities is not None else _tool_capability_map()
    )
    found: List[str] = []
    for cap in manifest.required_capabilities:
        if cap in NETWORK_CAPABILITIES and cap not in found:
            found.append(cap)
    for step in manifest.steps:
        for cap in caps_map.get(step.tool_name, []):
            if cap in NETWORK_CAPABILITIES and cap not in found:
                found.append(cap)
    return found


def skill_requires_network(
    manifest: SkillManifest,
    tool_capabilities: Optional[Dict[str, List[str]]] = None,
) -> bool:
    """True if running *manifest* would reach the network (declared or via tools)."""
    if requires_network(manifest.required_capabilities):
        return True
    return bool(skill_network_capabilities(manifest, tool_capabilities))


def partition_local_safe(
    manifests: List[SkillManifest],
    tool_capabilities: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[SkillManifest], List[SkillManifest]]:
    """Split skills into ``(local_safe, network_requiring)``.

    In ``local_only`` mode the network-requiring set is disabled by default; the
    caller surfaces only the local-safe set to the agent.
    """
    caps_map = (
        tool_capabilities if tool_capabilities is not None else _tool_capability_map()
    )
    local_safe: List[SkillManifest] = []
    network: List[SkillManifest] = []
    for m in manifests:
        (network if skill_requires_network(m, caps_map) else local_safe).append(m)
    return local_safe, network


__all__ = [
    "DANGEROUS_CAPABILITIES",
    "TrustTier",
    "classify_trust_tier",
    "validate_capabilities",
    "has_dangerous_capabilities",
    "partition_local_safe",
    "skill_network_capabilities",
    "skill_requires_network",
]
