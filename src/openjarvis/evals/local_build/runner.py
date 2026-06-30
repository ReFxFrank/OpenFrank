"""Eval runner for the local-build before/after report (Phase 7).

Runs the personal-assistant suite and aggregates per-tier metrics: task success,
tokens/sec, latency, free VRAM, energy. The per-prompt execution (`run_one`) is
**injected** — on the rig it routes (Phase 2) + calls the engine + reads
telemetry; in tests it's a deterministic fake — so the scoring/aggregation logic
is fully verifiable offline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from openjarvis.evals.local_build.suite import SUITE, EvalPrompt


@dataclass
class RunOutput:
    """What a single prompt execution reports back."""

    text: str
    tier: str = ""  # actual routed tier (may differ from the expected tier)
    latency_s: float = 0.0
    completion_tokens: int = 0
    tok_per_s: float = 0.0
    energy_j: float = 0.0
    free_vram_gb: Optional[float] = None


@dataclass
class PromptResult:
    id: str
    category: str
    tier: str
    success: bool
    latency_s: float
    tok_per_s: float
    energy_j: float
    free_vram_gb: Optional[float]


def default_score(prompt: EvalPrompt, text: str) -> bool:
    """Task success: any expected substring present (case-insensitive); else
    a non-empty response counts (open-ended chat/tool prompts)."""
    if not text or not text.strip():
        return False
    if prompt.expected:
        low = text.lower()
        return any(exp.lower() in low for exp in prompt.expected)
    return True


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else 0.0


@dataclass
class EvalResults:
    results: List[PromptResult] = field(default_factory=list)
    label: str = ""

    def _bucket(self, items: List[PromptResult]) -> Dict[str, object]:
        if not items:
            return {
                "count": 0,
                "task_success": 0.0,
                "tok_per_s": 0.0,
                "latency_s": 0.0,
                "energy_j": 0.0,
                "free_vram_gb": None,
            }
        free = [r.free_vram_gb for r in items if r.free_vram_gb is not None]
        return {
            "count": len(items),
            "task_success": round(sum(r.success for r in items) / len(items), 4),
            "tok_per_s": _mean([r.tok_per_s for r in items]),
            "latency_s": _mean([r.latency_s for r in items]),
            "energy_j": _mean([r.energy_j for r in items]),
            "free_vram_gb": _mean(free) if free else None,
        }

    def aggregate(self) -> Dict[str, object]:
        tiers: Dict[str, object] = {}
        for tier in ("fast", "balanced", "deep"):
            tiers[tier] = self._bucket([r for r in self.results if r.tier == tier])
        categories: Dict[str, object] = {}
        for cat in ("chat", "reasoning", "tool_use", "rag"):
            categories[cat] = self._bucket(
                [r for r in self.results if r.category == cat]
            )
        return {
            "label": self.label,
            "overall": self._bucket(self.results),
            "tiers": tiers,
            "categories": categories,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "results": [asdict(r) for r in self.results],
            "aggregate": self.aggregate(),
        }


def run_eval(
    run_one: Callable[[EvalPrompt], RunOutput],
    *,
    suite: Optional[List[EvalPrompt]] = None,
    score_fn: Callable[[EvalPrompt, str], bool] = default_score,
    label: str = "",
) -> EvalResults:
    """Run *suite* through *run_one*, scoring success and bucketing by tier."""
    suite = suite if suite is not None else SUITE
    results: List[PromptResult] = []
    for prompt in suite:
        out = run_one(prompt)
        results.append(
            PromptResult(
                id=prompt.id,
                category=prompt.category,
                tier=out.tier or prompt.tier,
                success=score_fn(prompt, out.text),
                latency_s=out.latency_s,
                tok_per_s=out.tok_per_s,
                energy_j=out.energy_j,
                free_vram_gb=out.free_vram_gb,
            )
        )
    return EvalResults(results=results, label=label)


__all__ = [
    "EvalResults",
    "PromptResult",
    "RunOutput",
    "default_score",
    "run_eval",
]
