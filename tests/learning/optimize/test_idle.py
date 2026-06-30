"""Tests for GPU idle detection used to schedule optimization (Phase 5)."""

from __future__ import annotations

from openjarvis.engine.offload import VramStatus
from openjarvis.learning.optimize.idle import gpu_is_idle


def _gpu(total=16.0, free=10.0):
    return VramStatus(
        available=True,
        total_gb=total,
        free_gb=free,
        used_gb=total - free,
        source="test",
    )


def test_idle_when_gpu_mostly_free():
    assert gpu_is_idle(_gpu(total=16, free=15)) is True


def test_not_idle_when_gpu_busy():
    assert gpu_is_idle(_gpu(total=16, free=4)) is False


def test_no_gpu_is_treated_as_idle():
    # CPU-only run won't fight the user for a GPU.
    assert gpu_is_idle(VramStatus(available=False)) is True


def test_threshold_is_configurable():
    status = _gpu(total=16, free=8)  # 50% free
    assert gpu_is_idle(status, min_free_fraction=0.4) is True
    assert gpu_is_idle(status, min_free_fraction=0.9) is False
