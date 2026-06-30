"""Guarded optimization run — snapshot, benchmark, keep-or-rollback (Phase 5).

Wraps any optimization step with the local-build guardrails:

1. **Idle gate** — skip when the GPU is busy (don't fight the user for it).
2. **Snapshot** — record the overlay tree before changing anything.
3. **Benchmark before/after** — run a scoring function on each side.
4. **Keep or roll back** — keep the new overlays only if the score improved by
   at least ``keep_threshold``; otherwise restore the snapshot. With no
   benchmark, changes are kept (nothing to compare) but still snapshotted, so a
   manual rollback is always possible.

``optimize_fn`` and ``bench_fn`` are injected, so the whole control flow is unit
-testable offline without DSPy or a running engine. A run that finds nothing to
do (``optimize_fn`` returns a falsy/empty result) **cleanly no-ops with a
report** rather than failing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from openjarvis.learning.optimize.snapshot import (
    create_snapshot,
    default_overlay_root,
    default_snapshots_dir,
    rollback,
)


@dataclass
class GuardedRunReport:
    """Outcome of a guarded optimization run."""

    ran: bool
    skipped_reason: str = ""
    snapshot_id: Optional[str] = None
    score_before: Optional[float] = None
    score_after: Optional[float] = None
    improved: bool = False
    kept: bool = False
    rolled_back: bool = False
    no_op: bool = False
    detail: str = ""

    def summary(self) -> str:
        if not self.ran:
            return f"skipped: {self.skipped_reason}"
        if self.no_op:
            return f"no-op: {self.detail}"
        parts = []
        if self.score_before is not None and self.score_after is not None:
            parts.append(f"{self.score_before:.4f} → {self.score_after:.4f}")
        parts.append("kept" if self.kept else "rolled back")
        if self.snapshot_id:
            parts.append(f"snapshot={self.snapshot_id}")
        return ", ".join(parts)


def run_guarded_optimization(
    optimize_fn: Callable[[], object],
    *,
    bench_fn: Optional[Callable[[], float]] = None,
    overlay_root: Optional[Path] = None,
    snapshots_dir: Optional[Path] = None,
    require_idle: bool = True,
    idle_check: Optional[Callable[[], bool]] = None,
    auto_rollback: bool = True,
    keep_threshold: float = 0.0,
    label: str = "optimize",
) -> GuardedRunReport:
    """Run *optimize_fn* with snapshot + benchmark + keep-or-rollback guardrails."""
    overlay_root = Path(overlay_root) if overlay_root else default_overlay_root()
    snapshots_dir = Path(snapshots_dir) if snapshots_dir else default_snapshots_dir()

    # 1. Idle gate.
    if require_idle:
        check = idle_check or _default_idle_check
        if not check():
            return GuardedRunReport(
                ran=False, skipped_reason="GPU busy (not idle); deferring optimization"
            )

    # 2. Snapshot before touching anything.
    snap = create_snapshot(
        overlay_root=overlay_root, snapshots_dir=snapshots_dir, label=label
    )

    # 3. Benchmark before.
    score_before = bench_fn() if bench_fn else None

    # 4. Optimize.
    result = optimize_fn()
    if not result:
        return GuardedRunReport(
            ran=True,
            no_op=True,
            snapshot_id=snap.snapshot_id,
            kept=True,
            score_before=score_before,
            detail="optimizer found nothing to do",
        )

    # 5. Benchmark after + decide.
    if bench_fn is None:
        return GuardedRunReport(
            ran=True,
            snapshot_id=snap.snapshot_id,
            score_before=score_before,
            kept=True,
            detail="no benchmark configured — kept; rollback available",
        )

    score_after = bench_fn()
    improved = score_after >= (score_before or 0.0) + keep_threshold
    if improved or not auto_rollback:
        return GuardedRunReport(
            ran=True,
            snapshot_id=snap.snapshot_id,
            score_before=score_before,
            score_after=score_after,
            improved=improved,
            kept=True,
            detail="kept (improved)" if improved else "kept (auto_rollback off)",
        )

    # Regressed → restore the snapshot.
    rollback(snap.snapshot_id, overlay_root=overlay_root, snapshots_dir=snapshots_dir)
    return GuardedRunReport(
        ran=True,
        snapshot_id=snap.snapshot_id,
        score_before=score_before,
        score_after=score_after,
        improved=False,
        kept=False,
        rolled_back=True,
        detail=(
            f"rolled back: {score_after:.4f} < "
            f"{(score_before or 0.0) + keep_threshold:.4f}"
        ),
    )


def _default_idle_check() -> bool:
    from openjarvis.learning.optimize.idle import gpu_is_idle

    return gpu_is_idle()


__all__ = ["GuardedRunReport", "run_guarded_optimization"]
