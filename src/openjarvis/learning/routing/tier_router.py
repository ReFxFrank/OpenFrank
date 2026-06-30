"""Tier router — classify a query and route it to fast / balanced / deep.

Combines the existing complexity analyzer (``complexity.score_complexity``) with
the VRAM-aware offload planner (``engine.offload``) to produce a single,
traceable :class:`RouteDecision`: which tier, which model+engine, and how to
split that model across GPU/CPU under the active offload profile.

This is the Phase 2 "smarter routing + VRAM-aware model management" layer. It is
a pure decision function — it never loads a model or calls an engine — so it is
fully unit-testable with a mocked :class:`~openjarvis.engine.offload.VramStatus`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from openjarvis.engine.offload import (
    DEFAULT_PROFILE_BUDGETS_GB,
    OffloadPlan,
    OffloadProfile,
    VramStatus,
    estimate_model_size_gb,
    plan_offload,
    read_vram,
    resolve_profile,
)
from openjarvis.learning.routing.complexity import (
    adjust_tokens_for_model,
    score_complexity,
)

logger = logging.getLogger(__name__)

_PARAMS_FROM_TAG = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)


class Tier(str, Enum):
    """Routing tiers, lightest → heaviest."""

    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"


# Lightest → heaviest, used for the downgrade walk.
_TIER_ORDER = (Tier.FAST, Tier.BALANCED, Tier.DEEP)


@dataclass(frozen=True)
class RouteDecision:
    """The full routing + offload decision for one query (traceable)."""

    tier: Tier
    model: str
    engine: str  # "" → use the active / default engine
    complexity_score: float
    complexity_tier: str
    max_tokens: int
    profile: OffloadProfile
    offload: OffloadPlan
    self_verify: bool
    requested_tier: Tier
    downgraded: bool
    reason: str

    def to_trace(self) -> Dict[str, object]:
        """Compact dict for telemetry / trace logging."""
        return {
            "tier": self.tier.value,
            "requested_tier": self.requested_tier.value,
            "downgraded": self.downgraded,
            "model": self.model,
            "engine": self.engine or "(default)",
            "complexity_score": self.complexity_score,
            "complexity_tier": self.complexity_tier,
            "max_tokens": self.max_tokens,
            "profile": self.profile.value,
            "num_gpu": self.offload.num_gpu,
            "gpu_fraction": self.offload.gpu_fraction,
            "cpu_only": self.offload.cpu_only,
            "budget_gb": self.offload.budget_gb,
            "model_size_gb": self.offload.model_size_gb,
            "self_verify": self.self_verify,
            "reason": self.reason,
        }


def classify_tier(score: float, router_cfg) -> Tier:  # noqa: ANN001
    """Map a complexity score (0–1) to a tier using configured boundaries."""
    if score < router_cfg.fast_max_score:
        return Tier.FAST
    if score >= router_cfg.deep_min_score:
        return Tier.DEEP
    return Tier.BALANCED


def tier_model(tier: Tier, router_cfg) -> tuple[str, str]:  # noqa: ANN001
    """Return ``(model, engine)`` configured for *tier*."""
    if tier is Tier.FAST:
        return router_cfg.fast_model, router_cfg.fast_engine
    if tier is Tier.DEEP:
        return router_cfg.deep_model, router_cfg.deep_engine
    return router_cfg.balanced_model, router_cfg.balanced_engine


def model_param_count_b(
    model: str, overrides: Optional[Dict[str, float]] = None
) -> float:
    """Best-effort parameter count for a model tag.

    Order: explicit override → ModelRegistry spec → parse from the tag
    (``qwen3:14b`` → 14.0) → 0.0. The registry is wiped between tests, so the
    tag-parsing fallback keeps routing deterministic without it.
    """
    if overrides and model in overrides:
        return overrides[model]
    try:
        from openjarvis.core.registry import ModelRegistry

        spec = ModelRegistry.get(model)
        if getattr(spec, "parameter_count_b", 0):
            return float(spec.parameter_count_b)
    except (KeyError, AttributeError, ImportError):
        pass
    m = _PARAMS_FROM_TAG.search(model)
    return float(m.group(1)) if m else 0.0


def _custom_budgets(offload_cfg) -> Dict[OffloadProfile, float]:  # noqa: ANN001
    out = dict(DEFAULT_PROFILE_BUDGETS_GB)
    if getattr(offload_cfg, "idle_budget_gb", 0) > 0:
        out[OffloadProfile.IDLE] = offload_cfg.idle_budget_gb
    if getattr(offload_cfg, "multitask_budget_gb", 0) > 0:
        out[OffloadProfile.MULTITASK] = offload_cfg.multitask_budget_gb
    if getattr(offload_cfg, "gaming_budget_gb", 0) > 0:
        out[OffloadProfile.GAMING] = offload_cfg.gaming_budget_gb
    return out


def _plan_for(
    tier: Tier,
    router_cfg,  # noqa: ANN001
    offload_cfg,  # noqa: ANN001
    profile: OffloadProfile,
    status: VramStatus,
    overrides: Optional[Dict[str, float]],
) -> tuple[str, str, OffloadPlan]:
    model, engine = tier_model(tier, router_cfg)
    params = model_param_count_b(model, overrides)
    quant = "q4"
    size = estimate_model_size_gb(params, quant)
    plan = plan_offload(
        size,
        profile=profile,
        status=status,
        parameter_count_b=params,
        safety_margin_gb=offload_cfg.safety_margin_gb,
        resident_reserve_gb=offload_cfg.resident_reserve_gb,
        custom_budgets=_custom_budgets(offload_cfg),
    )
    return model, engine, plan


def route(
    query: str,
    config,  # noqa: ANN001 — JarvisConfig
    *,
    vram_status: Optional[VramStatus] = None,
    model_param_counts: Optional[Dict[str, float]] = None,
) -> RouteDecision:
    """Decide tier + model + offload split for *query*.

    Reads live VRAM (unless *vram_status* is supplied — tests inject it),
    resolves the offload profile, scores complexity, picks a tier, and plans the
    GPU/CPU split. If the chosen tier can't get any GPU layers and downgrade is
    enabled (and a GPU is present), it falls to a lighter tier rather than
    running a heavy model entirely on CPU.
    """
    offload_cfg = config.offload
    router_cfg = config.router

    if vram_status is None:
        vram_status = read_vram(offload_cfg.gpu_index)
    profile = resolve_profile(offload_cfg.profile, vram_status)

    result = score_complexity(query)
    requested = classify_tier(result.score, router_cfg)

    # Candidate tiers: the requested one, then progressively lighter ones.
    idx = _TIER_ORDER.index(requested)
    candidates = list(reversed(_TIER_ORDER[: idx + 1]))  # requested → lighter

    planned = [
        _plan_for(t, router_cfg, offload_cfg, profile, vram_status, model_param_counts)
        for t in candidates
    ]

    # Downgrade only when a GPU is actually present and not pinned cpu_only —
    # otherwise honour the requested tier and run it on CPU as asked.
    allow_dg = (
        router_cfg.allow_downgrade
        and vram_status.available
        and profile is not OffloadProfile.CPU_ONLY
    )
    chosen_i = 0
    if allow_dg:
        for i, (_m, _e, plan) in enumerate(planned):
            if not plan.cpu_only:
                chosen_i = i
                break

    chosen_tier = candidates[chosen_i]
    model, engine, plan = planned[chosen_i]
    downgraded = chosen_tier is not requested

    max_tokens = adjust_tokens_for_model(result.suggested_max_tokens, model)
    self_verify = bool(router_cfg.self_verify) and chosen_tier is Tier.DEEP

    reason = f"complexity={result.score:.2f}({result.tier}) → {requested.value}"
    if downgraded:
        reason += f" → downgraded to {chosen_tier.value} ({plan.reason})"
    else:
        reason += f"; {plan.reason}"

    decision = RouteDecision(
        tier=chosen_tier,
        model=model,
        engine=engine,
        complexity_score=result.score,
        complexity_tier=result.tier,
        max_tokens=max_tokens,
        profile=profile,
        offload=plan,
        self_verify=self_verify,
        requested_tier=requested,
        downgraded=downgraded,
        reason=reason,
    )
    logger.debug("tier_router decision: %s", decision.to_trace())
    return decision


__all__ = [
    "RouteDecision",
    "Tier",
    "classify_tier",
    "model_param_count_b",
    "route",
    "tier_model",
]
