"""Tests for local_only skill classification (Phase 4)."""

from __future__ import annotations

import glob
from pathlib import Path

from openjarvis.skills.loader import load_skill
from openjarvis.skills.security import (
    partition_local_safe,
    skill_network_capabilities,
    skill_requires_network,
)
from openjarvis.skills.types import SkillManifest, SkillStep

_DATA = Path(__file__).resolve().parents[2] / "src" / "openjarvis" / "skills" / "data"

# Static tool→capability map so the test doesn't depend on registry population.
_TOOL_CAPS = {
    "web_search": ["network:fetch"],
    "http_request": ["network:fetch"],
    "file_read": ["file:read"],
    "think": [],
}


def _skill(*tool_names, caps=None):
    return SkillManifest(
        name="t",
        steps=[SkillStep(tool_name=tn) for tn in tool_names],
        required_capabilities=caps or [],
    )


def test_skill_with_network_tool_requires_network():
    s = _skill("file_read", "http_request", "think")
    assert skill_requires_network(s, _TOOL_CAPS) is True
    assert "network:fetch" in skill_network_capabilities(s, _TOOL_CAPS)


def test_skill_with_only_local_tools_is_local_safe():
    s = _skill("file_read", "think")
    assert skill_requires_network(s, _TOOL_CAPS) is False
    assert skill_network_capabilities(s, _TOOL_CAPS) == []


def test_declared_network_capability_counts():
    s = _skill("think", caps=["network:fetch"])
    assert skill_requires_network(s, _TOOL_CAPS) is True


def test_partition_local_safe():
    skills = [
        _skill("file_read"),
        _skill("web_search"),
        _skill("think"),
    ]
    local_safe, network = partition_local_safe(skills, _TOOL_CAPS)
    assert len(local_safe) == 2
    assert len(network) == 1


def test_bundled_skills_classified():
    """The real bundled skills split into local-safe + a few network ones."""
    mans = [load_skill(p) for p in sorted(glob.glob(str(_DATA / "*.toml")))]
    assert mans, "no bundled skills found"
    local_safe, network = partition_local_safe(mans, _TOOL_CAPS)
    names = {m.name for m in network}
    # web-summarize uses http_request; topic-research uses web_search.
    assert "web-summarize" in names
    assert "topic-research" in names
    # Most bundled skills are local-safe.
    assert len(local_safe) > len(network)


def test_bundled_local_skills_are_discoverable_and_parse():
    """Curated local skills load and expose steps (agent-discoverable)."""
    local_only_safe = []
    for p in sorted(glob.glob(str(_DATA / "*.toml"))):
        m = load_skill(p)
        if not skill_requires_network(m, _TOOL_CAPS):
            local_only_safe.append(m)
    assert len(local_only_safe) >= 10
    for m in local_only_safe:
        assert m.name and m.steps  # parsed with a name and at least one step
