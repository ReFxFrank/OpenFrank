"""GPU idle detection for scheduling optimization (Phase 5).

Local optimization runs are GPU-hungry, so they should run when the GPU is free
(e.g. nightly), never while the user is gaming/working. This reuses the Phase 2
live VRAM reader so "idle" means the same thing as the offload profiler's
``idle`` profile.
"""

from __future__ import annotations

from typing import Optional


def gpu_is_idle(
    status: Optional[object] = None, *, min_free_fraction: float = 0.8
) -> bool:
    """Return True when it's safe to run a GPU-heavy optimization.

    Idle = at least *min_free_fraction* of VRAM is free. A machine with **no
    GPU** is treated as idle (a CPU-only run won't fight the user for the card).
    """
    from openjarvis.engine.offload import VramStatus, read_vram

    s = status if isinstance(status, VramStatus) else read_vram()
    if not s.available:
        return True
    return s.free_fraction >= min_free_fraction


__all__ = ["gpu_is_idle"]
