"""Tests for VRAM-aware offload profiles + planning (Phase 2)."""

from __future__ import annotations

import pytest

from openjarvis.engine import offload
from openjarvis.engine.offload import (
    OffloadProfile,
    VramStatus,
    auto_select_profile,
    effective_budget_gb,
    estimate_gpu_layers,
    estimate_model_size_gb,
    estimate_total_layers,
    plan_offload,
    read_vram,
    resolve_profile,
)


def _gpu(total=16.0, free=10.0):
    return VramStatus(
        available=True,
        total_gb=total,
        free_gb=free,
        used_gb=total - free,
        source="test",
    )


# --------------------------------------------------------------------------
# VramStatus + reads
# --------------------------------------------------------------------------


def test_vram_status_fractions():
    s = _gpu(total=16, free=4)
    assert s.used_fraction == pytest.approx(0.75)
    assert s.free_fraction == pytest.approx(0.25)


def test_unavailable_status_fractions_are_zero():
    s = VramStatus(available=False)
    assert s.used_fraction == 0.0
    assert s.free_fraction == 0.0


def test_read_vram_falls_back_to_unavailable(monkeypatch):
    # Force both backends to report nothing → CPU-only machine path.
    monkeypatch.setattr(offload, "_read_vram_pynvml", lambda i: None)
    monkeypatch.setattr(offload, "_read_vram_nvidia_smi", lambda i: None)
    status = read_vram()
    assert status.available is False


# --------------------------------------------------------------------------
# Profile auto-selection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("free", "expected"),
    [
        (15.0, OffloadProfile.IDLE),  # 94% free
        (9.0, OffloadProfile.MULTITASK),  # 56% free
        (4.0, OffloadProfile.GAMING),  # 25% free
        (1.0, OffloadProfile.CPU_ONLY),  # 6% free → GPU busy
    ],
)
def test_auto_select_profile(free, expected):
    assert auto_select_profile(_gpu(total=16, free=free)) is expected


def test_auto_select_no_gpu_is_cpu_only():
    assert auto_select_profile(VramStatus(available=False)) is OffloadProfile.CPU_ONLY


def test_resolve_profile_auto_and_explicit():
    assert resolve_profile("auto", _gpu(free=15)) is OffloadProfile.IDLE
    assert resolve_profile("gaming", _gpu(free=15)) is OffloadProfile.GAMING
    assert (
        resolve_profile("bogus", _gpu(free=9)) is OffloadProfile.MULTITASK
    )  # falls back


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


def test_effective_budget_capped_by_profile():
    # idle cap is 14; free is plenty → budget ~= cap - reserve.
    b = effective_budget_gb(
        OffloadProfile.IDLE, _gpu(total=16, free=15), safety_margin_gb=0.5
    )
    assert b == pytest.approx(14.0)


def test_effective_budget_capped_by_free_vram():
    # multitask cap is 9, but only 6 free → budget limited by free - margin.
    b = effective_budget_gb(
        OffloadProfile.MULTITASK, _gpu(total=16, free=6), safety_margin_gb=0.5
    )
    assert b == pytest.approx(5.5)


def test_effective_budget_reserves_resident_models():
    b = effective_budget_gb(
        OffloadProfile.IDLE,
        _gpu(free=15),
        safety_margin_gb=0.5,
        resident_reserve_gb=2.0,
    )
    assert b == pytest.approx(12.0)


def test_cpu_only_profile_has_zero_budget():
    assert effective_budget_gb(OffloadProfile.CPU_ONLY, _gpu(free=15)) == 0.0


def test_no_gpu_has_zero_budget():
    assert effective_budget_gb(OffloadProfile.IDLE, VramStatus(available=False)) == 0.0


# --------------------------------------------------------------------------
# Footprint estimation
# --------------------------------------------------------------------------


def test_estimate_model_size_q4_ballpark():
    # Approximate Q4 footprints (real GB/B is mildly non-linear). The brief's
    # targets (8B≈5-6, 14B≈9-10, 32B≈19-20) are starting points to verify with
    # nvidia-smi on the rig; here we just assert a sane band + ordering.
    assert 4.5 <= estimate_model_size_gb(8, "q4") <= 7.0
    assert 8.0 <= estimate_model_size_gb(14, "q4") <= 11.0
    assert 18.0 <= estimate_model_size_gb(32, "q4") <= 22.0
    # Quantization ordering: q4 < q8 < fp16.
    assert (
        estimate_model_size_gb(8, "q4")
        < estimate_model_size_gb(8, "q8")
        < estimate_model_size_gb(8, "fp16")
    )


def test_estimate_total_layers_monotonic():
    assert estimate_total_layers(8) < estimate_total_layers(32)


def test_estimate_gpu_layers_partial():
    # 10 GB model over 40 layers = 0.25 GB/layer; 5 GB budget → 20 layers.
    assert estimate_gpu_layers(10.0, 40, 5.0) == 20


def test_estimate_gpu_layers_zero_when_no_budget():
    assert estimate_gpu_layers(10.0, 40, 0.0) == 0


# --------------------------------------------------------------------------
# Planning — the headroom guarantee
# --------------------------------------------------------------------------


def test_plan_fits_fully_on_idle():
    # 8B (~5 GB) under idle (14 GB budget) → all on GPU.
    plan = plan_offload(
        estimate_model_size_gb(8, "q4"),
        profile=OffloadProfile.IDLE,
        status=_gpu(total=16, free=15),
        parameter_count_b=8,
        resident_reserve_gb=0.0,
    )
    assert plan.fits_fully is True
    assert plan.cpu_only is False
    assert plan.gpu_fraction == 1.0
    assert plan.num_gpu == plan.total_layers


def test_plan_hybrid_split_when_partial():
    # 14B (~9 GB) under gaming (3 GB budget) → partial offload, not cpu-only.
    plan = plan_offload(
        estimate_model_size_gb(14, "q4"),
        profile=OffloadProfile.GAMING,
        status=_gpu(total=16, free=10),
        parameter_count_b=14,
        resident_reserve_gb=0.0,
    )
    assert plan.cpu_only is False
    assert 0 < plan.num_gpu < plan.total_layers
    assert 0.0 < plan.gpu_fraction < 1.0


def test_plan_shifts_to_cpu_when_budget_too_small():
    # 32B (~20 GB) but only ~2 GB free → cannot fit even partially → CPU, no OOM.
    plan = plan_offload(
        estimate_model_size_gb(32, "q4"),
        profile=OffloadProfile.GAMING,
        status=_gpu(total=16, free=2.0),
        parameter_count_b=32,
        resident_reserve_gb=1.5,
    )
    assert plan.cpu_only is True
    assert plan.num_gpu == 0


def test_plan_cpu_only_profile():
    plan = plan_offload(
        estimate_model_size_gb(8, "q4"),
        profile=OffloadProfile.CPU_ONLY,
        status=_gpu(free=15),
        parameter_count_b=8,
    )
    assert plan.cpu_only is True
    assert plan.num_gpu == 0


def test_plan_no_gpu_is_cpu_only():
    plan = plan_offload(
        estimate_model_size_gb(8, "q4"),
        profile=OffloadProfile.MULTITASK,
        status=VramStatus(available=False),
        parameter_count_b=8,
    )
    assert plan.cpu_only is True


def test_plan_never_exceeds_free_vram():
    # Budget must respect free VRAM, never the (larger) profile cap.
    status = _gpu(total=16, free=4.0)
    plan = plan_offload(
        estimate_model_size_gb(14, "q4"),
        profile=OffloadProfile.IDLE,  # cap 14, but only 4 free
        status=status,
        parameter_count_b=14,
        safety_margin_gb=0.5,
        resident_reserve_gb=0.0,
    )
    assert plan.budget_gb <= status.free_gb


def test_plan_as_engine_kwargs():
    plan = plan_offload(
        estimate_model_size_gb(8, "q4"),
        profile=OffloadProfile.IDLE,
        status=_gpu(free=15),
        parameter_count_b=8,
    )
    assert plan.as_engine_kwargs() == {"num_gpu": plan.num_gpu}
