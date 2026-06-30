"""Tests for the guarded optimization run (Phase 5 acceptance)."""

from __future__ import annotations

from openjarvis.learning.optimize.guarded import run_guarded_optimization


def _dirs(tmp_path):
    return tmp_path / "ov", tmp_path / "sn"


def test_skips_when_not_idle(tmp_path):
    ov, sn = _dirs(tmp_path)
    calls = []
    rep = run_guarded_optimization(
        lambda: calls.append("ran") or {"x": 1},
        overlay_root=ov,
        snapshots_dir=sn,
        require_idle=True,
        idle_check=lambda: False,
    )
    assert rep.ran is False
    assert "busy" in rep.skipped_reason.lower()
    assert calls == []  # optimizer never invoked


def test_no_op_when_optimizer_finds_nothing(tmp_path):
    ov, sn = _dirs(tmp_path)
    rep = run_guarded_optimization(
        lambda: {},  # empty result → nothing to do
        bench_fn=lambda: 1.0,
        overlay_root=ov,
        snapshots_dir=sn,
        require_idle=False,
    )
    assert rep.ran is True
    assert rep.no_op is True
    assert rep.kept is True
    assert rep.snapshot_id  # still snapshotted


def test_keeps_when_benchmark_improves(tmp_path):
    ov, sn = _dirs(tmp_path)
    scores = iter([0.50, 0.80])  # before, after
    rep = run_guarded_optimization(
        lambda: {"skill": "ok"},
        bench_fn=lambda: next(scores),
        overlay_root=ov,
        snapshots_dir=sn,
        require_idle=False,
    )
    assert rep.ran and rep.kept and rep.improved
    assert rep.rolled_back is False
    assert rep.score_before == 0.50 and rep.score_after == 0.80


def test_rolls_back_when_benchmark_regresses(tmp_path):
    ov, sn = _dirs(tmp_path)
    (ov / "s").mkdir(parents=True)
    (ov / "s" / "optimized.toml").write_text("ORIGINAL")

    scores = iter([0.80, 0.50])  # before, after (regression)

    def optimize():
        # Simulate the optimizer changing an overlay.
        (ov / "s" / "optimized.toml").write_text("CHANGED")
        return {"s": "changed"}

    rep = run_guarded_optimization(
        optimize,
        bench_fn=lambda: next(scores),
        overlay_root=ov,
        snapshots_dir=sn,
        require_idle=False,
    )
    assert rep.ran and rep.rolled_back and not rep.kept
    assert rep.improved is False
    # The overlay was restored to its pre-run content.
    assert (ov / "s" / "optimized.toml").read_text() == "ORIGINAL"


def test_kept_without_benchmark(tmp_path):
    ov, sn = _dirs(tmp_path)
    rep = run_guarded_optimization(
        lambda: {"s": "x"},
        bench_fn=None,
        overlay_root=ov,
        snapshots_dir=sn,
        require_idle=False,
    )
    assert rep.ran and rep.kept and not rep.rolled_back
    assert rep.snapshot_id  # rollback still available


def test_keep_threshold_requires_meaningful_gain(tmp_path):
    ov, sn = _dirs(tmp_path)
    scores = iter([0.50, 0.51])  # tiny gain, below threshold
    rep = run_guarded_optimization(
        lambda: {"s": "x"},
        bench_fn=lambda: next(scores),
        overlay_root=ov,
        snapshots_dir=sn,
        require_idle=False,
        keep_threshold=0.05,
    )
    assert rep.rolled_back is True  # 0.01 gain < 0.05 threshold


def test_summary_strings(tmp_path):
    ov, sn = _dirs(tmp_path)
    rep = run_guarded_optimization(
        lambda: {"s": "x"},
        overlay_root=ov,
        snapshots_dir=sn,
        require_idle=True,
        idle_check=lambda: False,
    )
    assert "skipped" in rep.summary()
