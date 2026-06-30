"""Tests for per-stage cost attribution (Phase 7 telemetry)."""

from __future__ import annotations

from openjarvis.telemetry.stage_metrics import (
    GENERATION,
    ROUTING,
    StageBreakdown,
    StageTimer,
)


def test_stage_timer_attributes_with_fake_clock():
    ticks = iter([0.0, 0.5, 0.5, 2.5])  # routing 0.5s, generation 2.0s
    timer = StageTimer(clock=lambda: next(ticks))
    with timer.stage(ROUTING):
        pass
    with timer.stage(GENERATION):
        pass
    b = timer.breakdown()
    assert b.stages[ROUTING] == 0.5
    assert b.stages[GENERATION] == 2.0
    assert b.total_seconds == 2.5
    assert b.fraction(GENERATION) == 0.8


def test_stage_timer_sums_on_reentry():
    ticks = iter([0.0, 1.0, 5.0, 6.0])
    timer = StageTimer(clock=lambda: next(ticks))
    with timer.stage("tools"):
        pass
    with timer.stage("tools"):
        pass
    assert timer.breakdown().stages["tools"] == 2.0


def test_add_negative_clamped():
    timer = StageTimer()
    timer.add("x", -5.0)
    assert timer.breakdown().stages["x"] == 0.0


def test_breakdown_to_dict():
    ticks = iter([0.0, 1.0])
    timer = StageTimer(clock=lambda: next(ticks))
    with timer.stage(ROUTING):
        pass
    d = timer.breakdown().to_dict()
    assert d["stages"][ROUTING] == 1.0
    assert d["total_seconds"] == 1.0
    assert d["fractions"][ROUTING] == 1.0


def test_empty_breakdown_fraction_is_zero():
    b = StageBreakdown()
    assert b.fraction("anything") == 0.0
    assert b.total_seconds == 0.0


def test_sample_vram_no_gpu_is_none(monkeypatch):
    from openjarvis.engine import offload

    monkeypatch.setattr(offload, "_read_vram_pynvml", lambda i: None)
    monkeypatch.setattr(offload, "_read_vram_nvidia_smi", lambda i: None)
    timer = StageTimer(sample_vram=True)
    assert timer.breakdown().min_free_vram_gb is None


def test_sample_vram_tracks_minimum(monkeypatch):
    from openjarvis.engine import offload
    from openjarvis.engine.offload import VramStatus

    frees = iter([10.0, 6.0, 8.0])

    def fake(_i):
        return VramStatus(available=True, total_gb=16, free_gb=next(frees), source="t")

    monkeypatch.setattr(offload, "_read_vram_pynvml", fake)
    timer = StageTimer(sample_vram=True)  # samples 10.0
    timer.sample_vram()  # 6.0
    timer.sample_vram()  # 8.0
    assert timer.breakdown().min_free_vram_gb == 6.0
