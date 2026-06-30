"""VRAM-aware offload profiles — the headroom guarantee for the local build.

Hybrid CPU+GPU is the **default** execution model, not a fallback: each model is
split across the GPU's VRAM and system RAM, sized by the active *offload profile*
so the assistant never starves the rest of the machine (OS, browser, games).

The engine places as many layers in VRAM as the active profile's budget allows
and runs the rest on the CPU/RAM. This module turns a profile + a live free-VRAM
reading into a concrete :class:`OffloadPlan` — primarily a GPU layer count
(``num_gpu`` for Ollama, ``--n-gpu-layers`` for llama.cpp). Token generation is
memory-bandwidth-bound, so the rule is: keep the hot path on the GPU, push the
overflow to RAM, and accept a speed cost proportional to how much spills over.

Critically, if the budget cannot fit even one layer, the plan **shifts to CPU**
rather than OOMing or evicting the user's other GPU apps.

VRAM is read live via ``pynvml`` (preferred) or ``nvidia-smi`` (fallback); on a
machine with no NVIDIA GPU both fail gracefully and the planner returns a
CPU-only plan. This module has no hard dependency on a GPU being present.
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class OffloadProfile(str, Enum):
    """How much VRAM the assistant may use, leaving the rest for other apps."""

    IDLE = "idle"  # GPU free → models mostly/entirely on GPU (fastest)
    MULTITASK = "multitask"  # browser/IDE/work (DEFAULT) → light partial offload
    GAMING = "gaming"  # GPU busy → heavy offload, keep the assistant alive
    CPU_ONLY = "cpu_only"  # no GPU layers at all (escape hatch / no GPU)


# Per-profile VRAM budget caps in GB, tuned for a ~16 GB card (RTX 5080). These
# are *caps*; the real budget is min(cap, live_free_vram - safety_margin), so on
# a smaller card the planner still respects what is actually free.
DEFAULT_PROFILE_BUDGETS_GB: Dict[OffloadProfile, float] = {
    OffloadProfile.IDLE: 14.0,
    OffloadProfile.MULTITASK: 9.0,
    OffloadProfile.GAMING: 3.0,
    OffloadProfile.CPU_ONLY: 0.0,
}

# Rough on-disk / VRAM footprint per 1B params at common quantizations (GB/B).
# Calibrated to the brief's confirmed sizing (8B≈5-6, 14B≈9-10, 32B≈19-20 @Q4).
_QUANT_GB_PER_B: Dict[str, float] = {
    "none": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "q8": 1.06,
    "gguf_q8": 1.06,
    "int4": 0.62,
    "q4": 0.62,
    "gguf_q4": 0.62,
    "nvfp4": 0.55,
    "fp4": 0.55,
}


@dataclass(frozen=True)
class VramStatus:
    """A live VRAM reading for one GPU (or 'unavailable' on a CPU-only box)."""

    available: bool
    total_gb: float = 0.0
    free_gb: float = 0.0
    used_gb: float = 0.0
    device_name: str = ""
    source: str = "none"  # "pynvml" | "nvidia-smi" | "none"

    @property
    def used_fraction(self) -> float:
        if not self.available or self.total_gb <= 0:
            return 0.0
        return max(0.0, min(1.0, self.used_gb / self.total_gb))

    @property
    def free_fraction(self) -> float:
        if not self.available or self.total_gb <= 0:
            return 0.0
        return max(0.0, min(1.0, self.free_gb / self.total_gb))


@dataclass(frozen=True)
class OffloadPlan:
    """A concrete GPU/CPU split for loading a model under a profile."""

    profile: OffloadProfile
    budget_gb: float
    model_size_gb: float
    total_layers: int
    num_gpu: int  # layers to place on GPU (Ollama num_gpu / llama.cpp n-gpu-layers)
    gpu_fraction: float  # 0.0 (all CPU) .. 1.0 (all GPU)
    cpu_only: bool
    fits_fully: bool  # whole model fits in the budget
    reason: str

    def as_engine_kwargs(self) -> Dict[str, int]:
        """Engine kwargs to realise this plan (consumed by the Ollama engine)."""
        return {"num_gpu": self.num_gpu}


# ---------------------------------------------------------------------------
# Live VRAM reads
# ---------------------------------------------------------------------------


def _read_vram_pynvml(index: int) -> Optional[VramStatus]:
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The pynvml package is deprecated.*",
                category=FutureWarning,
            )
            import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            if index >= count:
                return None
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            gib = 1024**3
            return VramStatus(
                available=True,
                total_gb=mem.total / gib,
                free_gb=mem.free / gib,
                used_gb=mem.used / gib,
                device_name=name,
                source="pynvml",
            )
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # noqa: BLE001 — any failure → fall through
        logger.debug("pynvml VRAM read failed: %s", exc)
        return None


def _read_vram_nvidia_smi(index: int) -> Optional[VramStatus]:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=memory.total,memory.free,memory.used,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("nvidia-smi VRAM read failed: %s", exc)
        return None
    if not out:
        return None
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    if len(parts) < 4:
        return None
    try:
        total_mb, free_mb, used_mb = (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return None
    return VramStatus(
        available=True,
        total_gb=total_mb / 1024.0,
        free_gb=free_mb / 1024.0,
        used_gb=used_mb / 1024.0,
        device_name=parts[3],
        source="nvidia-smi",
    )


def read_vram(index: int = 0) -> VramStatus:
    """Read live VRAM for GPU *index*: pynvml → nvidia-smi → unavailable.

    Never raises; on a machine with no NVIDIA GPU returns
    ``VramStatus(available=False)`` so callers transparently get a CPU-only plan.
    """
    status = _read_vram_pynvml(index)
    if status is not None:
        return status
    status = _read_vram_nvidia_smi(index)
    if status is not None:
        return status
    return VramStatus(available=False)


# ---------------------------------------------------------------------------
# Profile selection + budget
# ---------------------------------------------------------------------------


def auto_select_profile(status: VramStatus) -> OffloadProfile:
    """Pick a profile from current GPU usage.

    Heavily-loaded GPU (a game running) → gaming/cpu-only; mostly-free GPU →
    idle; in between → multitask (the default). Based on the *free* fraction so
    it adapts to whatever else is using the card right now.
    """
    if not status.available:
        return OffloadProfile.CPU_ONLY
    free = status.free_fraction
    if free >= 0.80:
        return OffloadProfile.IDLE
    if free >= 0.45:
        return OffloadProfile.MULTITASK
    if free >= 0.18:
        return OffloadProfile.GAMING
    return OffloadProfile.CPU_ONLY


def effective_budget_gb(
    profile: OffloadProfile,
    status: VramStatus,
    *,
    safety_margin_gb: float = 0.5,
    resident_reserve_gb: float = 0.0,
    custom_budgets: Optional[Dict[OffloadProfile, float]] = None,
) -> float:
    """The VRAM budget actually usable for model layers under *profile*.

    ``min(profile cap, live_free - safety_margin) - resident_reserve``, floored
    at 0. The resident reserve keeps room for the always-resident embedding +
    reranker models. Never exceeds what is physically free, so we cannot evict
    the user's other GPU apps.
    """
    if profile == OffloadProfile.CPU_ONLY or not status.available:
        return 0.0
    budgets = custom_budgets or DEFAULT_PROFILE_BUDGETS_GB
    cap = budgets.get(profile, DEFAULT_PROFILE_BUDGETS_GB[profile])
    usable_free = max(0.0, status.free_gb - safety_margin_gb)
    return max(0.0, min(cap, usable_free) - resident_reserve_gb)


# ---------------------------------------------------------------------------
# Model footprint estimation
# ---------------------------------------------------------------------------


def estimate_model_size_gb(parameter_count_b: float, quantization: str = "q4") -> float:
    """Approximate on-disk / VRAM footprint of a model at a quantization."""
    per_b = _QUANT_GB_PER_B.get(quantization.lower().replace("-", "_"), 0.62)
    return max(0.0, parameter_count_b) * per_b


def estimate_total_layers(parameter_count_b: float) -> int:
    """Rough transformer layer count from parameter size (a planning hint)."""
    p = parameter_count_b
    if p <= 0:
        return 32
    if p < 2:
        return 24
    if p < 5:
        return 28
    if p < 10:
        return 32
    if p < 20:
        return 40
    if p < 40:
        return 48
    return 64


def estimate_kv_cache_gb(
    context_length: int,
    parameter_count_b: float,
    *,
    kv_quant_bytes: float = 1.0,
) -> float:
    """Very rough KV-cache footprint. Context is a VRAM tax; keep it modest.

    Scales with context and model size. ``kv_quant_bytes`` ≈ 1.0 models Q8 KV
    (half of fp16's 2 bytes) — enabling KV-cache quantization roughly halves
    this. Coarse on purpose; the goal is to bias the split, not bill exactly.
    """
    if context_length <= 0 or parameter_count_b <= 0:
        return 0.0
    layers = estimate_total_layers(parameter_count_b)
    # heads*dim grows ~with sqrt(params); approximate hidden bytes per token.
    hidden_factor = 2.0 * math.sqrt(parameter_count_b)  # KiB/token/layer-ish
    bytes_per_token = layers * hidden_factor * 1024 * kv_quant_bytes
    return (context_length * bytes_per_token) / (1024**3)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def estimate_gpu_layers(
    model_size_gb: float,
    total_layers: int,
    budget_gb: float,
    *,
    kv_cache_gb: float = 0.0,
) -> int:
    """How many of *total_layers* fit in *budget_gb* (after the KV cache)."""
    if total_layers <= 0 or model_size_gb <= 0:
        return 0
    usable = budget_gb - kv_cache_gb
    if usable <= 0:
        return 0
    per_layer = model_size_gb / total_layers
    if per_layer <= 0:
        return total_layers
    layers = int(math.floor(usable / per_layer))
    return max(0, min(total_layers, layers))


def plan_offload(
    model_size_gb: float,
    *,
    profile: OffloadProfile,
    status: VramStatus,
    total_layers: Optional[int] = None,
    parameter_count_b: float = 0.0,
    safety_margin_gb: float = 0.5,
    resident_reserve_gb: float = 0.0,
    kv_cache_gb: float = 0.0,
    custom_budgets: Optional[Dict[OffloadProfile, float]] = None,
) -> OffloadPlan:
    """Turn a profile + live VRAM reading into a concrete GPU/CPU split.

    Returns a CPU-only plan (``num_gpu=0``) whenever the budget cannot fit even
    one layer — shifting the split toward CPU instead of OOMing.
    """
    if total_layers is None:
        total_layers = estimate_total_layers(parameter_count_b)

    budget = effective_budget_gb(
        profile,
        status,
        safety_margin_gb=safety_margin_gb,
        resident_reserve_gb=resident_reserve_gb,
        custom_budgets=custom_budgets,
    )

    def _cpu_only(reason: str) -> OffloadPlan:
        return OffloadPlan(
            profile=profile,
            budget_gb=round(budget, 2),
            model_size_gb=round(model_size_gb, 2),
            total_layers=total_layers,
            num_gpu=0,
            gpu_fraction=0.0,
            cpu_only=True,
            fits_fully=False,
            reason=reason,
        )

    if profile == OffloadProfile.CPU_ONLY:
        return _cpu_only("profile=cpu_only — all layers on CPU/RAM")
    if not status.available:
        return _cpu_only("no GPU detected — running fully on CPU/RAM")
    if budget <= 0:
        return _cpu_only(
            f"VRAM budget {budget:.1f} GB after margins — shifting fully to CPU"
        )

    layers = estimate_gpu_layers(
        model_size_gb, total_layers, budget, kv_cache_gb=kv_cache_gb
    )
    if layers <= 0:
        return _cpu_only(
            f"model {model_size_gb:.1f} GB does not fit budget "
            f"{budget:.1f} GB even partially — CPU fallback (no OOM)"
        )

    if layers >= total_layers:
        return OffloadPlan(
            profile=profile,
            budget_gb=round(budget, 2),
            model_size_gb=round(model_size_gb, 2),
            total_layers=total_layers,
            num_gpu=total_layers,
            gpu_fraction=1.0,
            cpu_only=False,
            fits_fully=True,
            reason=(
                f"fits fully on GPU ({model_size_gb:.1f} GB ≤ {budget:.1f} GB budget)"
            ),
        )

    return OffloadPlan(
        profile=profile,
        budget_gb=round(budget, 2),
        model_size_gb=round(model_size_gb, 2),
        total_layers=total_layers,
        num_gpu=layers,
        gpu_fraction=round(layers / total_layers, 3),
        cpu_only=False,
        fits_fully=False,
        reason=(
            f"hybrid split: {layers}/{total_layers} layers on GPU "
            f"({model_size_gb:.1f} GB > {budget:.1f} GB budget; rest in RAM)"
        ),
    )


def ollama_runtime_env(
    flash_attention: bool = True, kv_cache_quant: str = "q8"
) -> Dict[str, str]:
    """Server-level Ollama env vars that keep long context within the VRAM budget.

    Flash attention and KV-cache quantization are configured at ``ollama serve``
    startup (they are not per-request options), so set these in the environment
    *before* launching Ollama. KV-cache quant (Q8 ≈ half the fp16 cache) requires
    flash attention to be on. Context is a VRAM tax — these roughly halve it.
    """
    env: Dict[str, str] = {}
    if flash_attention:
        env["OLLAMA_FLASH_ATTENTION"] = "1"
    mapping = {
        "q8": "q8_0",
        "q8_0": "q8_0",
        "q4": "q4_0",
        "q4_0": "q4_0",
        "f16": "f16",
        "fp16": "f16",
    }
    cache_type = mapping.get((kv_cache_quant or "").strip().lower())
    if cache_type and flash_attention:
        env["OLLAMA_KV_CACHE_TYPE"] = cache_type
    return env


def resolve_profile(name: str, status: VramStatus) -> OffloadProfile:
    """Resolve a configured profile name, honouring ``"auto"``."""
    key = (name or "").strip().lower()
    if key in ("", "auto"):
        return auto_select_profile(status)
    try:
        return OffloadProfile(key)
    except ValueError:
        logger.debug("unknown offload profile %r — auto-selecting", name)
        return auto_select_profile(status)


__all__ = [
    "DEFAULT_PROFILE_BUDGETS_GB",
    "OffloadPlan",
    "OffloadProfile",
    "VramStatus",
    "auto_select_profile",
    "effective_budget_gb",
    "estimate_gpu_layers",
    "estimate_kv_cache_gb",
    "estimate_model_size_gb",
    "estimate_total_layers",
    "ollama_runtime_env",
    "plan_offload",
    "read_vram",
    "resolve_profile",
]
