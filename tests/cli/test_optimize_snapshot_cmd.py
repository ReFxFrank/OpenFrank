"""CLI tests for `jarvis optimize snapshot|snapshots|rollback` (Phase 5)."""

from __future__ import annotations

from click.testing import CliRunner

from openjarvis.cli.optimize_cmd import optimize_group


def _overlay(home, skill, content):
    d = home / "learning" / "skills" / skill
    d.mkdir(parents=True, exist_ok=True)
    (d / "optimized.toml").write_text(content)


def test_snapshot_then_list_then_rollback(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    # default_overlay_root() reads the lru_cached load_config(); clear it so the
    # command resolves paths under this test's OPENJARVIS_HOME, not a cached one.
    from openjarvis.core.config import load_config

    load_config.cache_clear()
    _overlay(tmp_path, "web-summarize", "ORIGINAL")

    runner = CliRunner()

    # 1. snapshot
    r = runner.invoke(optimize_group, ["snapshot", "--label", "before-test"])
    assert r.exit_code == 0, r.output
    assert "Snapshot created" in r.output

    # 2. list shows it
    r = runner.invoke(optimize_group, ["snapshots"])
    assert r.exit_code == 0
    assert "before-test" in r.output

    # 3. modify the overlay, then roll back to the snapshot
    overlay = tmp_path / "learning" / "skills" / "web-summarize" / "optimized.toml"
    overlay.write_text("MODIFIED")

    from openjarvis.learning.optimize.snapshot import list_snapshots

    sid = list_snapshots(tmp_path / "learning" / "snapshots")[0].snapshot_id
    r = runner.invoke(optimize_group, ["rollback", sid])
    assert r.exit_code == 0, r.output
    assert "Rolled back" in r.output
    assert overlay.read_text() == "ORIGINAL"


def test_rollback_unknown_id_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    # default_overlay_root() reads the lru_cached load_config(); clear it so the
    # command resolves paths under this test's OPENJARVIS_HOME, not a cached one.
    from openjarvis.core.config import load_config

    load_config.cache_clear()
    r = CliRunner().invoke(optimize_group, ["rollback", "does-not-exist"])
    assert r.exit_code == 1


def test_snapshots_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    # default_overlay_root() reads the lru_cached load_config(); clear it so the
    # command resolves paths under this test's OPENJARVIS_HOME, not a cached one.
    from openjarvis.core.config import load_config

    load_config.cache_clear()
    r = CliRunner().invoke(optimize_group, ["snapshots"])
    assert r.exit_code == 0
    assert "No snapshots yet" in r.output
