"""Before/after report renderer for the local-build eval (Phase 7).

Turns two aggregate dicts (from :meth:`EvalResults.aggregate`) into a markdown
report comparing task success, tokens/sec, latency, free VRAM, and energy per
tier — so "smarter" is numbers, not vibes. ``after=None`` renders a
before-only report with the after column marked pending (for the rig run).
"""

from __future__ import annotations

from typing import Dict, Optional

_METRICS = [
    ("task_success", "Task success", "{:.0%}", True),
    ("tok_per_s", "Tokens/sec", "{:.1f}", True),
    ("latency_s", "Latency (s)", "{:.2f}", False),
    ("free_vram_gb", "Free VRAM (GB)", "{:.1f}", True),
    ("energy_j", "Energy (J)", "{:.1f}", False),
]


def _fmt(value: object, fmt: str) -> str:
    if value is None:
        return "—"
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def _delta(before: object, after: object, higher_is_better: bool) -> str:
    if before is None or after is None:
        return "—"
    try:
        d = float(after) - float(before)
    except (ValueError, TypeError):
        return "—"
    if abs(d) < 1e-9:
        return "±0"
    good = (d > 0) == higher_is_better
    arrow = "▲" if d > 0 else "▼"
    mark = "" if good else " ⚠"
    return f"{arrow}{abs(d):.2f}{mark}"


def _section(
    name: str, before: Dict[str, object], after: Optional[Dict[str, object]]
) -> str:
    lines = [f"### {name}", ""]
    if after is None:
        lines.append("| Metric | Before | After |")
        lines.append("|---|---|---|")
        for key, label, fmt, _ in _METRICS:
            lines.append(f"| {label} | {_fmt(before.get(key), fmt)} | _pending_ |")
    else:
        lines.append("| Metric | Before | After | Δ |")
        lines.append("|---|---|---|---|")
        for key, label, fmt, hib in _METRICS:
            b, a = before.get(key), after.get(key)
            lines.append(
                f"| {label} | {_fmt(b, fmt)} | {_fmt(a, fmt)} | {_delta(b, a, hib)} |"
            )
    lines.append("")
    return "\n".join(lines)


def render_before_after(
    before: Dict[str, object],
    after: Optional[Dict[str, object]] = None,
    *,
    title: str = "Local-Build Eval — Before / After",
    note: str = "",
) -> str:
    """Render a markdown before/after report from two aggregate dicts."""
    out = [f"# {title}", ""]
    if note:
        out += [note, ""]
    if after is None:
        out += [
            "> **After column pending** — run `scripts/run_local_eval.py` on the "
            "RTX 5080 rig (with Ollama + the per-tier models pulled) to fill it. "
            "The before numbers are the Phase 0 targets from `baseline.json`.",
            "",
        ]

    out.append("## Overall")
    out.append("")
    out.append(
        _section(
            "All prompts",
            before.get("overall", {}),
            (after or {}).get("overall") if after else None,
        )
    )

    out.append("## Per tier")
    out.append("")
    b_tiers = before.get("tiers", {}) or {}
    a_tiers = (after or {}).get("tiers", {}) if after else {}
    for tier in ("fast", "balanced", "deep"):
        if tier in b_tiers:
            out.append(
                _section(tier, b_tiers[tier], a_tiers.get(tier) if after else None)
            )

    cats = before.get("categories", {}) or {}
    if cats:
        out.append("## Per category")
        out.append("")
        a_cats = (after or {}).get("categories", {}) if after else {}
        for cat in ("chat", "reasoning", "tool_use", "rag"):
            if cat in cats:
                out.append(_section(cat, cats[cat], a_cats.get(cat) if after else None))

    return "\n".join(out).rstrip() + "\n"


__all__ = ["render_before_after"]
