#!/usr/bin/env python3
"""Run the local-build personal-assistant eval on the rig and write results JSON.

Routes each prompt (Phase 2 tier router), runs it on the local engine, and
records per-tier task success / tokens-per-sec / latency / free VRAM / energy.
Intended for the RTX 5080 + Ollama; on a machine with no engine it exits with a
clear message. Output feeds ``scripts/gen_eval_report.py``.

Usage:
    uv run python scripts/run_local_eval.py --label after -o docs/local-build/after.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--label", default="after", help="Label for this run (before/after)."
    )
    ap.add_argument("-o", "--output", default="docs/local-build/after.json")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    from openjarvis.core.config import load_config
    from openjarvis.core.types import Message, Role
    from openjarvis.engine import get_engine
    from openjarvis.engine.offload import read_vram
    from openjarvis.evals.local_build.runner import RunOutput, run_eval
    from openjarvis.learning.routing.tier_router import route

    cfg = load_config()
    resolved = get_engine(cfg)
    if resolved is None:
        print("No local engine available (start Ollama + pull the tier models).")
        return 1
    _engine_key, engine = resolved

    def run_one(prompt) -> RunOutput:  # noqa: ANN001
        # Route to a tier + model (falls back to the default model if disabled).
        model = cfg.intelligence.default_model
        tier = prompt.tier
        engine_kwargs = {}
        if cfg.router.enabled:
            decision = route(prompt.prompt, cfg)
            model, tier = decision.model, decision.tier.value
            engine_kwargs["num_gpu"] = decision.offload.num_gpu

        text = prompt.prompt
        if prompt.context:
            text = f"Context:\n{prompt.context}\n\nQuestion: {prompt.prompt}"
        messages = [Message(role=Role.USER, content=text)]

        free_before = read_vram().free_gb
        t0 = time.perf_counter()
        result = engine.generate(
            messages, model=model, max_tokens=args.max_tokens, **engine_kwargs
        )
        latency = time.perf_counter() - t0
        free_after = read_vram().free_gb

        usage = result.get("usage", {}) if isinstance(result, dict) else {}
        out_tokens = int(usage.get("completion_tokens", 0))
        energy = float(usage.get("energy_joules", 0.0))
        return RunOutput(
            text=result.get("content", "") if isinstance(result, dict) else str(result),
            tier=tier,
            latency_s=latency,
            completion_tokens=out_tokens,
            tok_per_s=(out_tokens / latency) if latency > 0 else 0.0,
            energy_j=energy,
            free_vram_gb=min(free_before, free_after) if free_before else None,
        )

    results = run_eval(run_one, label=args.label)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results.to_dict(), indent=2))
    agg = results.aggregate()["overall"]
    print(f"Wrote {out_path}")
    print(
        f"Overall: success={agg['task_success']:.0%} "
        f"tok/s={agg['tok_per_s']:.1f} latency={agg['latency_s']:.2f}s "
        f"free_vram={agg['free_vram_gb']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
