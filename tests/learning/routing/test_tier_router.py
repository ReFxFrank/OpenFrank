"""Tests for the tier router (Phase 2 smarter routing)."""

from __future__ import annotations

from openjarvis.core.config import JarvisConfig
from openjarvis.engine.offload import OffloadProfile, VramStatus
from openjarvis.learning.routing.tier_router import (
    RouteDecision,
    Tier,
    classify_tier,
    model_param_count_b,
    route,
    tier_model,
)


def _cfg(**router_overrides):
    cfg = JarvisConfig()
    cfg.router.enabled = True
    for k, v in router_overrides.items():
        setattr(cfg.router, k, v)
    return cfg


def _gpu(total=16.0, free=10.0):
    return VramStatus(
        available=True,
        total_gb=total,
        free_gb=free,
        used_gb=total - free,
        source="test",
    )


# A query that reliably scores in the "very_complex" band (> deep_min_score).
DEEP_QUERY = (
    "Analyze and compare these approaches in depth and explain step by step why "
    "each works. First derive the integral and prove the theorem about the matrix "
    "eigenvalues; then next additionally evaluate the trade-offs and pros and cons. "
    "Why is option A faster? How does option B scale? What about option C? Which "
    "wins? Write code: ```python\ndef solve(x):\n    return x\n``` and design a "
    "complete system. Consider the probability and compute the sum and the "
    "derivative as well, reasoning carefully about each requirement below.\n"
    "1. requirement one\n2. requirement two\n3. requirement three\n4. requirement four"
)


# --------------------------------------------------------------------------
# Classification + tier→model
# --------------------------------------------------------------------------


def test_classify_tier_boundaries():
    cfg = _cfg().router
    assert classify_tier(0.0, cfg) is Tier.FAST
    assert classify_tier(0.29, cfg) is Tier.FAST
    assert classify_tier(0.30, cfg) is Tier.BALANCED
    assert classify_tier(0.79, cfg) is Tier.BALANCED
    assert classify_tier(0.80, cfg) is Tier.DEEP
    assert classify_tier(1.0, cfg) is Tier.DEEP


def test_tier_model_mapping():
    cfg = _cfg().router
    assert tier_model(Tier.FAST, cfg)[0] == cfg.fast_model
    assert tier_model(Tier.BALANCED, cfg)[0] == cfg.balanced_model
    assert tier_model(Tier.DEEP, cfg)[0] == cfg.deep_model


def test_model_param_count_from_tag():
    assert model_param_count_b("qwen3:14b") == 14.0
    assert model_param_count_b("gpt-oss:20b") == 20.0
    assert model_param_count_b("qwen3:0.6b") == 0.6
    assert model_param_count_b("mystery-model") == 0.0


def test_model_param_count_override_wins():
    assert model_param_count_b("x", {"x": 7.0}) == 7.0


# --------------------------------------------------------------------------
# Routing decisions
# --------------------------------------------------------------------------


def test_route_simple_query_is_fast():
    d = route("hi", _cfg(), vram_status=_gpu(free=15))
    assert isinstance(d, RouteDecision)
    assert d.tier is Tier.FAST
    assert d.model == _cfg().router.fast_model


def test_route_complex_query_is_deep():
    d = route(DEEP_QUERY, _cfg(), vram_status=_gpu(free=15))
    assert d.tier is Tier.DEEP
    assert d.model == _cfg().router.deep_model


def test_route_sets_offload_plan_and_profile():
    d = route("hi", _cfg(), vram_status=_gpu(total=16, free=15))
    assert d.profile is OffloadProfile.IDLE  # GPU mostly free
    assert d.offload.num_gpu > 0  # fast model fits on GPU
    assert d.offload.cpu_only is False


def test_route_cpu_only_when_no_gpu():
    d = route("hi", _cfg(), vram_status=VramStatus(available=False))
    assert d.profile is OffloadProfile.CPU_ONLY
    assert d.offload.cpu_only is True
    assert d.offload.num_gpu == 0


def test_route_downgrades_when_tier_cannot_fit_any_layer():
    # Deep query, but a tiny pinned GPU budget so the big deep model can't get
    # even one GPU layer → downgrade to a lighter tier that can.
    cfg = _cfg(
        deep_model="qwen3:32b",
        balanced_model="qwen3:14b",
        fast_model="qwen3:8b",
        allow_downgrade=True,
    )
    cfg.offload.profile = "gaming"
    cfg.offload.gaming_budget_gb = 0.3  # < one 32B layer, ≥ one 14B layer
    cfg.offload.resident_reserve_gb = 0.0
    d = route(DEEP_QUERY, cfg, vram_status=_gpu(total=16, free=15))
    assert d.requested_tier is Tier.DEEP
    assert d.downgraded is True
    assert d.tier in (Tier.BALANCED, Tier.FAST)
    assert d.offload.cpu_only is False


def test_route_no_downgrade_when_disabled():
    cfg = _cfg(deep_model="qwen3:32b", allow_downgrade=False)
    cfg.offload.profile = "gaming"
    cfg.offload.gaming_budget_gb = 0.3
    cfg.offload.resident_reserve_gb = 0.0
    d = route(DEEP_QUERY, cfg, vram_status=_gpu(total=16, free=15))
    assert d.tier is Tier.DEEP
    assert d.downgraded is False
    assert d.offload.cpu_only is True  # honoured the requested tier, on CPU


def test_route_self_verify_only_for_deep():
    deep = route(DEEP_QUERY, _cfg(self_verify=True), vram_status=_gpu(free=15))
    fast = route("hi", _cfg(self_verify=True), vram_status=_gpu(free=15))
    assert deep.self_verify is True
    assert fast.self_verify is False


def test_route_trace_dict_is_complete():
    d = route("hi", _cfg(), vram_status=_gpu(free=15))
    trace = d.to_trace()
    for key in ("tier", "model", "profile", "num_gpu", "cpu_only", "complexity_score"):
        assert key in trace


def test_route_respects_custom_budget_override():
    # Pin multitask but cap its budget to a sliver → fast model won't fully fit.
    cfg = _cfg()
    cfg.offload.profile = "multitask"
    cfg.offload.multitask_budget_gb = 1.0
    cfg.offload.resident_reserve_gb = 0.0
    d = route("hi", cfg, vram_status=_gpu(total=16, free=15))
    # 8B (~5 GB) under a 1 GB budget → partial or cpu, never fully fitting.
    assert d.offload.fits_fully is False
