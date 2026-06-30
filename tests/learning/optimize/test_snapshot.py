"""Tests for overlay snapshot / rollback (Phase 5 reversibility)."""

from __future__ import annotations

from pathlib import Path

from openjarvis.learning.optimize.snapshot import (
    create_snapshot,
    get_snapshot,
    list_snapshots,
    rollback,
)


def _overlay(root: Path, skill: str, content: str) -> None:
    d = root / skill
    d.mkdir(parents=True, exist_ok=True)
    (d / "optimized.toml").write_text(content)


def test_create_snapshot_copies_overlays(tmp_path):
    ov, sn = tmp_path / "ov", tmp_path / "sn"
    _overlay(ov, "web-summarize", "skill_name = 'web-summarize'")
    snap = create_snapshot(overlay_root=ov, snapshots_dir=sn, label="v1")
    assert snap.file_count == 1
    assert snap.label == "v1"
    assert (Path(snap.path) / "overlays" / "web-summarize" / "optimized.toml").exists()


def test_rollback_restores_modified_overlay(tmp_path):
    ov, sn = tmp_path / "ov", tmp_path / "sn"
    _overlay(ov, "s", "ORIGINAL")
    snap = create_snapshot(overlay_root=ov, snapshots_dir=sn)

    (ov / "s" / "optimized.toml").write_text("MODIFIED")
    rollback(snap.snapshot_id, overlay_root=ov, snapshots_dir=sn)
    assert (ov / "s" / "optimized.toml").read_text() == "ORIGINAL"


def test_rollback_removes_overlays_created_after_snapshot(tmp_path):
    ov, sn = tmp_path / "ov", tmp_path / "sn"
    _overlay(ov, "existing", "keep")
    snap = create_snapshot(overlay_root=ov, snapshots_dir=sn)

    # A new overlay added after the snapshot must be removed on rollback.
    _overlay(ov, "added-later", "new")
    rollback(snap.snapshot_id, overlay_root=ov, snapshots_dir=sn)
    assert (ov / "existing" / "optimized.toml").exists()
    assert not (ov / "added-later").exists()


def test_snapshot_of_empty_root_then_rollback_clears(tmp_path):
    ov, sn = tmp_path / "ov", tmp_path / "sn"
    # No overlays yet → snapshot the empty state.
    snap = create_snapshot(overlay_root=ov, snapshots_dir=sn)
    assert snap.file_count == 0
    # Create one, then roll back to the empty snapshot.
    _overlay(ov, "s", "x")
    rollback(snap.snapshot_id, overlay_root=ov, snapshots_dir=sn)
    assert not (ov / "s").exists()


def test_list_and_get_snapshots(tmp_path):
    ov, sn = tmp_path / "ov", tmp_path / "sn"
    _overlay(ov, "s", "x")
    a = create_snapshot(
        overlay_root=ov,
        snapshots_dir=sn,
        snapshot_id="aaa",
        timestamp="2026-01-01T00:00:00",
    )
    b = create_snapshot(
        overlay_root=ov,
        snapshots_dir=sn,
        snapshot_id="bbb",
        timestamp="2026-02-01T00:00:00",
    )
    snaps = list_snapshots(sn)
    assert [s.snapshot_id for s in snaps] == ["bbb", "aaa"]  # newest first
    assert get_snapshot("aaa", sn).snapshot_id == a.snapshot_id
    assert get_snapshot(b.snapshot_id, sn) is not None
    assert get_snapshot("missing", sn) is None


def test_rollback_unknown_snapshot_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        rollback("nope", overlay_root=tmp_path / "ov", snapshots_dir=tmp_path / "sn")
