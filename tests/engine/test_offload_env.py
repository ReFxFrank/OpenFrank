"""Tests for the Ollama KV-cache / flash-attention env helper (Phase 3)."""

from __future__ import annotations

from openjarvis.engine.offload import ollama_runtime_env


def test_flash_attention_and_q8_kv():
    env = ollama_runtime_env(flash_attention=True, kv_cache_quant="q8")
    assert env["OLLAMA_FLASH_ATTENTION"] == "1"
    assert env["OLLAMA_KV_CACHE_TYPE"] == "q8_0"


def test_q4_kv_maps():
    env = ollama_runtime_env(flash_attention=True, kv_cache_quant="q4")
    assert env["OLLAMA_KV_CACHE_TYPE"] == "q4_0"


def test_kv_quant_requires_flash_attention():
    # KV-cache quant needs flash attention; without it, no KV type is emitted.
    env = ollama_runtime_env(flash_attention=False, kv_cache_quant="q8")
    assert "OLLAMA_KV_CACHE_TYPE" not in env
    assert "OLLAMA_FLASH_ATTENTION" not in env


def test_no_kv_quant_when_blank():
    env = ollama_runtime_env(flash_attention=True, kv_cache_quant="")
    assert env["OLLAMA_FLASH_ATTENTION"] == "1"
    assert "OLLAMA_KV_CACHE_TYPE" not in env
