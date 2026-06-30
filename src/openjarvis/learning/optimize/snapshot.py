"""Snapshot / rollback for skill-optimization overlays (Phase 5 reversibility).

Every optimization run must be reversible. Before a run we snapshot the skill
overlay tree (``~/.openjarvis/learning/skills/<skill>/optimized.toml``); if the
run regresses (or the user changes their mind) we restore it. Snapshots are
plain copies on disk under ``~/.openjarvis/learning/snapshots/<id>/`` with a
``meta.json`` — fully local, no external state.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from openjarvis.core.paths import get_config_dir

_OVERLAY_SUBDIR = "overlays"
_META_NAME = "meta.json"


@dataclass(frozen=True)
class OverlaySnapshot:
    """Metadata for one overlay snapshot."""

    snapshot_id: str
    label: str
    created_at: str
    file_count: int
    path: str  # snapshot directory


def default_overlay_root() -> Path:
    """Resolve the skill-overlay root (config override → default tree)."""
    try:
        from openjarvis.core.config import load_config

        cfg = load_config()
        cfg_dir = getattr(getattr(cfg.learning, "skills", None), "overlay_dir", None)
        if cfg_dir:
            return Path(cfg_dir).expanduser()
    except Exception:
        pass
    return get_config_dir() / "learning" / "skills"


def default_snapshots_dir() -> Path:
    """Where snapshots are stored."""
    return get_config_dir() / "learning" / "snapshots"


def _count_files(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file()) if root.exists() else 0


def create_snapshot(
    *,
    overlay_root: Optional[Path] = None,
    snapshots_dir: Optional[Path] = None,
    label: str = "",
    snapshot_id: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> OverlaySnapshot:
    """Snapshot the current overlay tree. Returns the :class:`OverlaySnapshot`.

    An empty/absent overlay root snapshots cleanly (``file_count == 0``); rolling
    that back later removes any overlays created since — so the *absence* of
    overlays is itself a restorable state.
    """
    overlay_root = Path(overlay_root) if overlay_root else default_overlay_root()
    snapshots_dir = Path(snapshots_dir) if snapshots_dir else default_snapshots_dir()
    sid = snapshot_id or uuid.uuid4().hex[:12]
    created = timestamp or _dt.datetime.now(_dt.timezone.utc).isoformat()

    snap_dir = snapshots_dir / sid
    snap_overlays = snap_dir / _OVERLAY_SUBDIR
    if snap_dir.exists():
        shutil.rmtree(snap_dir)
    snap_overlays.parent.mkdir(parents=True, exist_ok=True)

    if overlay_root.exists():
        shutil.copytree(overlay_root, snap_overlays)
    else:
        snap_overlays.mkdir(parents=True, exist_ok=True)

    snap = OverlaySnapshot(
        snapshot_id=sid,
        label=label,
        created_at=created,
        file_count=_count_files(snap_overlays),
        path=str(snap_dir),
    )
    (snap_dir / _META_NAME).write_text(json.dumps(asdict(snap), indent=2))
    return snap


def list_snapshots(snapshots_dir: Optional[Path] = None) -> List[OverlaySnapshot]:
    """List snapshots, newest first."""
    snapshots_dir = Path(snapshots_dir) if snapshots_dir else default_snapshots_dir()
    if not snapshots_dir.exists():
        return []
    out: List[OverlaySnapshot] = []
    for child in snapshots_dir.iterdir():
        meta = child / _META_NAME
        if meta.is_file():
            try:
                out.append(OverlaySnapshot(**json.loads(meta.read_text())))
            except (json.JSONDecodeError, TypeError):
                continue
    out.sort(key=lambda s: s.created_at, reverse=True)
    return out


def get_snapshot(
    snapshot_id: str, snapshots_dir: Optional[Path] = None
) -> Optional[OverlaySnapshot]:
    """Return the snapshot with *snapshot_id*, or None."""
    snapshots_dir = Path(snapshots_dir) if snapshots_dir else default_snapshots_dir()
    meta = snapshots_dir / snapshot_id / _META_NAME
    if not meta.is_file():
        return None
    try:
        return OverlaySnapshot(**json.loads(meta.read_text()))
    except (json.JSONDecodeError, TypeError):
        return None


def rollback(
    snapshot_id: str,
    *,
    overlay_root: Optional[Path] = None,
    snapshots_dir: Optional[Path] = None,
) -> OverlaySnapshot:
    """Restore the overlay tree from *snapshot_id*. Raises if it doesn't exist."""
    overlay_root = Path(overlay_root) if overlay_root else default_overlay_root()
    snapshots_dir = Path(snapshots_dir) if snapshots_dir else default_snapshots_dir()
    snap = get_snapshot(snapshot_id, snapshots_dir)
    if snap is None:
        raise FileNotFoundError(f"No snapshot {snapshot_id!r} in {snapshots_dir}")

    snap_overlays = snapshots_dir / snapshot_id / _OVERLAY_SUBDIR
    if overlay_root.exists():
        shutil.rmtree(overlay_root)
    overlay_root.parent.mkdir(parents=True, exist_ok=True)
    if snap_overlays.exists():
        shutil.copytree(snap_overlays, overlay_root)
    else:
        overlay_root.mkdir(parents=True, exist_ok=True)
    return snap


__all__ = [
    "OverlaySnapshot",
    "create_snapshot",
    "default_overlay_root",
    "default_snapshots_dir",
    "get_snapshot",
    "list_snapshots",
    "rollback",
]
