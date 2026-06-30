"""Tests that the Ollama engine plumbs the offload split into request options."""

from __future__ import annotations

from openjarvis.engine.ollama import _build_options, _default_num_ctx


def test_options_default_without_num_gpu():
    opts = _build_options(0.7, 256, {})
    assert opts["temperature"] == 0.7
    assert opts["num_predict"] == 256
    assert opts["num_ctx"] == _default_num_ctx()
    # No num_gpu → Ollama auto-derives the split (unchanged behaviour).
    assert "num_gpu" not in opts


def test_options_includes_num_gpu_when_provided():
    opts = _build_options(0.7, 256, {"num_gpu": 12})
    assert opts["num_gpu"] == 12


def test_num_gpu_zero_is_cpu_only_and_preserved():
    # 0 is meaningful (cpu-only) and must not be dropped as falsy.
    opts = _build_options(0.7, 256, {"num_gpu": 0})
    assert opts["num_gpu"] == 0


def test_options_includes_main_gpu():
    opts = _build_options(0.7, 256, {"num_gpu": 8, "main_gpu": 1})
    assert opts["main_gpu"] == 1


def test_num_ctx_override():
    opts = _build_options(0.7, 256, {"num_ctx": 4096})
    assert opts["num_ctx"] == 4096
