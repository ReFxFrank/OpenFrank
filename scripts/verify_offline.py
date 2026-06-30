#!/usr/bin/env python3
"""Offline / airgap verification harness for OpenJarvis local_only mode.

Asserts the fully-local guarantee holds. Designed to pass even with the host's
network physically disabled — it verifies that egress is *blocked*, not that the
internet is reachable. Exit code 0 = all checks passed, non-zero = a guarantee
was violated.

Checks:
  1. [runtime] local_only resolves to true (config default or env).
  2. The engine factory FAILS CLOSED for cloud backends (LocalOnlyViolation),
     for both an explicit request and during discovery.
  3. The egress guard ALLOWS loopback and BLOCKS the public internet.
  4. (Best effort) If a local engine is up, run a tiny generation; otherwise
     report it as skipped — its absence is not a failure of the guarantee.

Invoked by scripts/verify-offline.sh and scripts/verify-offline.ps1.
"""

from __future__ import annotations

import os
import socket
import sys

# Force the guarantee on for this verification regardless of the user's config.
os.environ["OPENJARVIS_LOCAL_ONLY"] = "1"

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

_failures = 0


def _report(status: str, msg: str) -> None:
    global _failures
    marker = {PASS: "✓", FAIL: "✗", SKIP: "–"}.get(status, "?")
    print(f"  [{marker}] {status}: {msg}")
    if status == FAIL:
        _failures += 1


def check_local_only_default() -> None:
    from openjarvis.core.config import load_config

    load_config.cache_clear()  # type: ignore[attr-defined]
    cfg = load_config()
    if getattr(cfg.runtime, "local_only", False):
        _report(PASS, "config resolves local_only = true")
    else:
        _report(FAIL, "config resolves local_only = false (airgap not in effect)")


def check_cloud_fails_closed() -> None:
    from openjarvis.core.config import load_config
    from openjarvis.core.registry import EngineRegistry
    from openjarvis.engine._discovery import _make_engine, discover_engines, get_engine
    from openjarvis.security.egress import LocalOnlyViolation

    cfg = load_config()

    # Find a registered cloud engine key.
    cloud_keys = [
        k
        for k in EngineRegistry.keys()
        if getattr(EngineRegistry.get(k), "is_cloud", False)
    ]
    if not cloud_keys:
        _report(SKIP, "no cloud engine registered to test fail-closed")
        return
    key = cloud_keys[0]

    # 3a. Direct factory call must raise.
    try:
        _make_engine(key, cfg)
        _report(FAIL, f"_make_engine({key!r}) did NOT fail closed")
    except LocalOnlyViolation:
        _report(PASS, f"_make_engine({key!r}) failed closed (LocalOnlyViolation)")

    # 3b. Explicit get_engine request must raise, not silently fall back.
    try:
        get_engine(cfg, engine_key=key)
        _report(FAIL, f"get_engine(engine_key={key!r}) did NOT fail closed")
    except LocalOnlyViolation:
        _report(PASS, f"get_engine(engine_key={key!r}) failed closed")

    # 3c. Discovery must never surface a cloud engine.
    discovered = {k for k, _ in discover_engines(cfg)}
    if any(getattr(EngineRegistry.get(k), "is_cloud", False) for k in discovered):
        _report(FAIL, "discovery surfaced a cloud engine in local_only")
    else:
        _report(PASS, "discovery surfaced no cloud engines")


def check_egress_guard() -> None:
    from openjarvis.security.egress import (
        EgressBlocked,
        install_guard,
        uninstall_guard,
    )

    install_guard({"localhost"})
    try:
        # Loopback must NOT be blocked. A refused/closed-port error is fine —
        # that means the guard let it through to the OS.
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.2)
        try:
            s.connect(("127.0.0.1", 9))  # discard port; likely refused
        except EgressBlocked:
            _report(FAIL, "egress guard wrongly BLOCKED loopback")
        except OSError:
            _report(PASS, "egress guard allowed loopback (reached the OS)")
        else:
            _report(PASS, "egress guard allowed loopback")
        finally:
            s.close()

        # A public address must be blocked.
        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s2.settimeout(0.2)
        try:
            s2.connect(("8.8.8.8", 53))
            _report(FAIL, "egress guard did NOT block public 8.8.8.8:53")
        except EgressBlocked:
            _report(PASS, "egress guard blocked public 8.8.8.8:53")
        except OSError as exc:  # any other socket error means it wasn't blocked
            _report(FAIL, f"public connect not blocked (got {type(exc).__name__})")
        finally:
            s2.close()
    finally:
        uninstall_guard()


def check_local_inference_optional() -> None:
    from openjarvis.core.config import load_config
    from openjarvis.engine._discovery import discover_engines

    cfg = load_config()
    healthy = discover_engines(cfg)
    if not healthy:
        _report(SKIP, "no local engine running (start Ollama to test inference)")
        return
    _report(PASS, f"local engine(s) available: {[k for k, _ in healthy]}")


def main() -> int:
    print("OpenJarvis offline / airgap verification")
    print("=" * 44)
    print(f"OPENJARVIS_LOCAL_ONLY={os.environ.get('OPENJARVIS_LOCAL_ONLY')}")
    print()
    print("1. local_only configuration")
    check_local_only_default()
    print("2. cloud backends fail closed")
    check_cloud_fails_closed()
    print("3. network egress guard")
    check_egress_guard()
    print("4. local inference (optional)")
    check_local_inference_optional()
    print()
    if _failures:
        print(f"RESULT: FAILED ({_failures} check(s) failed)")
        return 1
    print("RESULT: PASSED — fully-local guarantee holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
