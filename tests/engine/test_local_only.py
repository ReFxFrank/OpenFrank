"""Engine-factory fail-closed behaviour under [runtime] local_only (Phase 1).

The autouse conftest fixture wipes EngineRegistry between tests, so these tests
register their own fake engines to exercise the is_cloud gating logic directly —
independent of which real engine SDKs happen to be importable.
"""

from __future__ import annotations

import pytest

from openjarvis.core.config import JarvisConfig
from openjarvis.core.registry import EngineRegistry
from openjarvis.engine._base import InferenceEngine
from openjarvis.engine._discovery import _make_engine, discover_engines, get_engine
from openjarvis.security.egress import LocalOnlyViolation


class _FakeCloud(InferenceEngine):
    engine_id = "fake_cloud"
    is_cloud = True

    def generate(self, messages, *, model, **kwargs):  # noqa: ANN001, ANN003
        return {"content": "", "usage": {}}

    async def stream(self, messages, *, model, **kwargs):  # noqa: ANN001, ANN003
        yield ""

    def list_models(self):
        return ["cloud-model"]

    def health(self) -> bool:
        return True


class _FakeLocal(InferenceEngine):
    engine_id = "fake_local"
    is_cloud = False

    def generate(self, messages, *, model, **kwargs):  # noqa: ANN001, ANN003
        return {"content": "", "usage": {}}

    async def stream(self, messages, *, model, **kwargs):  # noqa: ANN001, ANN003
        yield ""

    def list_models(self):
        return ["local-model"]

    def health(self) -> bool:
        return True


@pytest.fixture
def engines():
    EngineRegistry.register_value("fake_cloud", _FakeCloud)
    EngineRegistry.register_value("fake_local", _FakeLocal)
    yield


def _config(local_only: bool) -> JarvisConfig:
    cfg = JarvisConfig()
    cfg.runtime.local_only = local_only
    return cfg


def test_real_cloud_engine_declares_is_cloud():
    # Sanity: the canonical lockdown signal is set on the shipped adapters.
    from openjarvis.engine.cloud import CloudEngine

    assert CloudEngine.is_cloud is True


def test_make_engine_cloud_fails_closed_in_local_only(engines):
    with pytest.raises(LocalOnlyViolation):
        _make_engine("fake_cloud", _config(True))


def test_make_engine_cloud_allowed_when_local_only_off(engines):
    engine = _make_engine("fake_cloud", _config(False))
    assert engine.is_cloud is True


def test_make_engine_local_backend_allowed_in_local_only(engines):
    engine = _make_engine("fake_local", _config(True))
    assert engine.is_cloud is False


def test_get_engine_explicit_cloud_fails_closed(engines):
    with pytest.raises(LocalOnlyViolation):
        get_engine(_config(True), engine_key="fake_cloud")


def test_get_engine_explicit_cloud_allowed_when_off(engines):
    result = get_engine(_config(False), engine_key="fake_cloud")
    assert result is not None
    assert result[0] == "fake_cloud"


def test_discover_engines_excludes_cloud_in_local_only(engines):
    discovered = {k for k, _ in discover_engines(_config(True))}
    assert "fake_cloud" not in discovered
    assert "fake_local" in discovered


def test_discover_engines_includes_cloud_when_off(engines):
    discovered = {k for k, _ in discover_engines(_config(False))}
    assert "fake_cloud" in discovered


def test_discover_engines_does_not_raise_with_cloud_default(engines):
    cfg = _config(True)
    cfg.engine.default = "fake_cloud"
    discovered = discover_engines(cfg)  # must not raise
    assert all(
        not getattr(EngineRegistry.get(k), "is_cloud", False) for k, _ in discovered
    )
