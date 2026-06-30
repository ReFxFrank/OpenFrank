"""Tests for the local-build eval suite, runner, and report (Phase 7)."""

from __future__ import annotations

from openjarvis.evals.local_build.report import render_before_after
from openjarvis.evals.local_build.runner import (
    EvalResults,
    RunOutput,
    default_score,
    run_eval,
)
from openjarvis.evals.local_build.suite import SUITE, EvalPrompt, by_category, by_tier

# --------------------------------------------------------------------------
# Suite
# --------------------------------------------------------------------------


def test_suite_size_and_coverage():
    assert len(SUITE) >= 45
    cats = by_category()
    for cat in ("chat", "reasoning", "tool_use", "rag"):
        assert len(cats[cat]) >= 8
    tiers = by_tier()
    assert set(tiers) <= {"fast", "balanced", "deep"}


def test_suite_ids_unique():
    ids = [p.id for p in SUITE]
    assert len(ids) == len(set(ids))


def test_rag_prompts_have_context():
    rag = [p for p in SUITE if p.category == "rag"]
    assert rag and all(p.context for p in rag)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_default_score_substring_match():
    p = EvalPrompt("x", "chat", "capital of France?", expected=["Paris"])
    assert default_score(p, "It's Paris.") is True
    assert default_score(p, "I don't know") is False


def test_default_score_open_ended_nonempty():
    p = EvalPrompt("x", "chat", "say hi")  # no expected
    assert default_score(p, "Hello there") is True
    assert default_score(p, "   ") is False


# --------------------------------------------------------------------------
# Runner + aggregation
# --------------------------------------------------------------------------


def _engine(answer_correctly: bool, toks=80, lat=2.0):
    def run_one(p: EvalPrompt) -> RunOutput:
        text = (p.expected[0] if p.expected else "ok") if answer_correctly else "nope"
        return RunOutput(
            text=text,
            tier=p.tier,
            latency_s=lat,
            completion_tokens=toks,
            tok_per_s=toks / lat,
            energy_j=lat * 100,
            free_vram_gb=7.0,
        )

    return run_one


def test_run_eval_aggregates_per_tier():
    res = run_eval(_engine(True), label="after")
    agg = res.aggregate()
    assert agg["overall"]["task_success"] == 1.0
    assert agg["overall"]["tok_per_s"] == 40.0
    assert "fast" in agg["tiers"] and "deep" in agg["tiers"]
    assert agg["overall"]["count"] == len(SUITE)


def test_run_eval_low_success_when_wrong():
    res = run_eval(_engine(False))
    # Open-ended prompts (no expected) still pass on a non-empty answer; prompts
    # with expected substrings fail — so success is strictly between 0 and 1.
    s = res.aggregate()["overall"]["task_success"]
    assert 0.0 < s < 1.0


def test_results_to_dict_roundtrip_shape():
    res = run_eval(_engine(True))
    d = res.to_dict()
    assert "results" in d and "aggregate" in d
    assert len(d["results"]) == len(SUITE)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def test_report_before_only_marks_pending():
    before = run_eval(_engine(False)).aggregate()
    md = render_before_after(before, None)
    assert "pending" in md.lower()
    assert "Task success" in md


def test_report_before_after_shows_deltas():
    before = run_eval(_engine(False, toks=60, lat=3.0)).aggregate()
    after = run_eval(_engine(True, toks=90, lat=1.5)).aggregate()
    md = render_before_after(before, after)
    assert "Δ" in md
    assert "▲" in md  # tok/s and success went up
    # Latency improved (down) — rendered with a down arrow.
    assert "▼" in md


def test_report_handles_empty_aggregates():
    empty = EvalResults().aggregate()
    md = render_before_after(empty, None)
    assert "Before" in md
