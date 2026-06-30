"""Per-stage cost attribution for a turn (Phase 7 telemetry).

The Phase 0 telemetry records one row per *inference call*. Phase 2/3 added
pipeline stages around inference — routing, memory/retrieval, self-verification —
whose cost should be auditable separately so "smarter" can be attributed to a
stage rather than hidden in the total. This adds a light stage timer plus a
free-VRAM sampler, so the headroom guarantee is auditable per turn.

Dependency-free (``time.perf_counter``); the free-VRAM sample reuses the Phase 2
reader and degrades to ``None`` on a GPU-less box.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

# Canonical stage names so reports line up across runs.
ROUTING = "routing"
MEMORY = "memory"
VERIFICATION = "verification"
GENERATION = "generation"
TOOLS = "tools"
OTHER = "other"


@dataclass
class StageBreakdown:
    """Per-stage seconds for one turn, plus the tightest free-VRAM seen."""

    stages: Dict[str, float] = field(default_factory=dict)
    total_seconds: float = 0.0
    min_free_vram_gb: Optional[float] = None

    def fraction(self, name: str) -> float:
        """Fraction of total time spent in *name* (0 if no time recorded)."""
        if self.total_seconds <= 0:
            return 0.0
        return self.stages.get(name, 0.0) / self.total_seconds

    def to_dict(self) -> Dict[str, object]:
        return {
            "stages": dict(self.stages),
            "total_seconds": round(self.total_seconds, 6),
            "min_free_vram_gb": self.min_free_vram_gb,
            "fractions": {k: round(self.fraction(k), 4) for k in self.stages},
        }


class StageTimer:
    """Accumulate per-stage durations and free-VRAM samples for one turn.

    Usage::

        timer = StageTimer(sample_vram=True)
        with timer.stage("routing"):
            ...
        with timer.stage("generation"):
            ...
        breakdown = timer.breakdown()
    """

    def __init__(self, *, sample_vram: bool = False, clock=time.perf_counter) -> None:
        self._stages: Dict[str, float] = {}
        self._sample_vram = sample_vram
        self._clock = clock
        self._min_free_vram: Optional[float] = None
        if sample_vram:
            self.sample_vram()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block and attribute it to stage *name* (sums on re-entry)."""
        start = self._clock()
        try:
            yield
        finally:
            self.add(name, self._clock() - start)
            if self._sample_vram:
                self.sample_vram()

    def add(self, name: str, seconds: float) -> None:
        """Add *seconds* to stage *name*."""
        if seconds < 0:
            seconds = 0.0
        self._stages[name] = self._stages.get(name, 0.0) + seconds

    def sample_vram(self) -> Optional[float]:
        """Record current free VRAM, tracking the minimum (tightest headroom)."""
        try:
            from openjarvis.engine.offload import read_vram

            status = read_vram()
        except Exception:
            return None
        if not status.available:
            return None
        free = status.free_gb
        if self._min_free_vram is None or free < self._min_free_vram:
            self._min_free_vram = free
        return free

    def breakdown(self) -> StageBreakdown:
        return StageBreakdown(
            stages=dict(self._stages),
            total_seconds=sum(self._stages.values()),
            min_free_vram_gb=self._min_free_vram,
        )


__all__ = [
    "GENERATION",
    "MEMORY",
    "OTHER",
    "ROUTING",
    "TOOLS",
    "VERIFICATION",
    "StageBreakdown",
    "StageTimer",
]
