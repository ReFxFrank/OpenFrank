"""Regression: malformed signature must not panic the Rust path (Phase 6 DoS).

The Rust skill verifier decoded the manifest's hex ``signature`` field by slicing
the &str by byte index (`s[i..i+2]`). An attacker-controlled signature containing
a multi-byte UTF-8 codepoint (e.g. "aéb") made an index land inside a codepoint,
panicking with "byte index N is not a char boundary". Across the PyO3 boundary
that surfaces as ``pyo3_runtime.PanicException`` — a ``BaseException`` ordinary
``except Exception`` handlers miss — i.e. a denial-of-service vector.

These tests exercise the exposed ``openjarvis_rust.load_skill(...).verify_signature``
path with a valid verifying key so decode_hex is actually reached.
"""

from __future__ import annotations

import pytest

rust = pytest.importorskip(
    "openjarvis_rust",
    reason="Rust extension not built (run: maturin develop)",
)

# RFC 8032 Test-1 Ed25519 public key — a known-valid verifying key, so
# verify_signature gets past key validation and reaches the hex decode of the
# (attacker-controlled) signature field.
VALID_PUBKEY_HEX = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"

_MANIFEST_TEMPLATE = """
name = "evil"
version = "0"
description = ""
author = ""
required_capabilities = []
signature = "{signature}"

[[steps]]
tool_name = "x"
arguments_template = "{{}}"
output_key = "o"
"""


def _manifest(signature: str):
    return rust.load_skill(_MANIFEST_TEMPLATE.format(signature=signature))


@pytest.mark.parametrize(
    "bad_signature",
    [
        "aéb",  # 4 bytes, index splits the 'é'
        "é",  # 2 bytes
        "🦀🦀🦀🦀",  # 16 bytes, all multi-byte
        "aa🦀",  # 6 bytes
        "zz",  # even length, non-hex ASCII
        "abc",  # odd length
    ],
)
def test_malformed_signature_returns_false_without_panic(bad_signature):
    # Must return a plain False — never raise (a PanicException would be a
    # BaseException, so we assert no exception of any kind by not catching).
    result = _manifest(bad_signature).verify_signature(VALID_PUBKEY_HEX)
    assert result is False


def test_valid_length_wrong_signature_reaches_verification():
    # 64 valid hex bytes: decode_hex succeeds, verification runs and fails.
    # Confirms the valid key let us reach the decode path the bug lived in.
    result = _manifest("00" * 64).verify_signature(VALID_PUBKEY_HEX)
    assert result is False


def test_empty_signature_is_handled():
    result = _manifest("").verify_signature(VALID_PUBKEY_HEX)
    assert result is False
