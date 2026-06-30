"""Personal-assistant eval suite for the local build (Phase 0 / Phase 7).

~50 representative prompts across four categories — chat, multi-step reasoning,
tool-use, and RAG — used to measure the before/after intelligence lift. Each
entry carries an expected-substring list where the answer is deterministic
(used for task-success scoring); open-ended chat prompts score on a non-empty,
on-topic response. ``tier`` is the expected routing tier (Phase 2), so results
can be bucketed per tier.

Kept as plain data so the same suite drives the rig run and the offline harness
tests. RAG prompts ship the supporting context inline so retrieval can be graded
without external corpora.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class EvalPrompt:
    id: str
    category: str  # chat | reasoning | tool_use | rag
    prompt: str
    tier: str = "fast"  # expected routing tier: fast | balanced | deep
    expected: List[str] = field(default_factory=list)  # success substrings (any-of)
    context: Optional[str] = None  # inline RAG context, if any


SUITE: List[EvalPrompt] = [
    # --- chat (12) ---------------------------------------------------------
    EvalPrompt("chat-01", "chat", "Hello! Who are you?", "fast"),
    EvalPrompt("chat-02", "chat", "What's the capital of France?", "fast", ["Paris"]),
    EvalPrompt("chat-03", "chat", "Give me a fun fact about octopuses.", "fast"),
    EvalPrompt("chat-04", "chat", "How many days are in a leap year?", "fast", ["366"]),
    EvalPrompt(
        "chat-05",
        "chat",
        "What color do you get mixing blue and yellow?",
        "fast",
        ["green"],
    ),
    EvalPrompt(
        "chat-06",
        "chat",
        "Translate 'good morning' to Spanish.",
        "fast",
        ["buenos d", "buenos di"],
    ),
    EvalPrompt(
        "chat-07", "chat", "What's the chemical symbol for gold?", "fast", ["Au"]
    ),
    EvalPrompt("chat-08", "chat", "Suggest a name for a pet turtle.", "fast"),
    EvalPrompt(
        "chat-09", "chat", "Is a tomato a fruit or a vegetable?", "fast", ["fruit"]
    ),
    EvalPrompt(
        "chat-10", "chat", "What planet is known as the Red Planet?", "fast", ["Mars"]
    ),
    EvalPrompt(
        "chat-11", "chat", "Round 3.14159 to two decimal places.", "fast", ["3.14"]
    ),
    EvalPrompt(
        "chat-12",
        "chat",
        "What's the opposite of 'ephemeral'?",
        "fast",
        ["permanent", "lasting", "enduring"],
    ),
    # --- reasoning (14) ----------------------------------------------------
    EvalPrompt(
        "reason-01",
        "reasoning",
        "If a train travels 60 km in 1.5 hours, what is its average speed in km/h?",
        "balanced",
        ["40"],
    ),
    EvalPrompt(
        "reason-02",
        "reasoning",
        "I have 3 apples and buy 2 bags of 4 apples each. How many apples total?",
        "balanced",
        ["11"],
    ),
    EvalPrompt(
        "reason-03",
        "reasoning",
        "Explain step by step why the sky appears blue.",
        "balanced",
        ["scatter"],
    ),
    EvalPrompt("reason-04", "reasoning", "What is 15% of 240?", "balanced", ["36"]),
    EvalPrompt(
        "reason-05",
        "reasoning",
        "A is taller than B, B is taller than C. Who is shortest?",
        "balanced",
        ["C"],
    ),
    EvalPrompt(
        "reason-06",
        "reasoning",
        "Compare the trade-offs of SQLite vs PostgreSQL for a single-user desktop app.",
        "balanced",
    ),
    EvalPrompt(
        "reason-07",
        "reasoning",
        "If today is Wednesday, what day is it 10 days from now?",
        "balanced",
        ["Saturday"],
    ),
    EvalPrompt(
        "reason-08",
        "reasoning",
        "Analyze the pros and cons of remote work, then give a recommendation.",
        "deep",
    ),
    EvalPrompt(
        "reason-09",
        "reasoning",
        "Prove that the sum of two even numbers is even.",
        "deep",
        ["even"],
    ),
    EvalPrompt(
        "reason-10",
        "reasoning",
        "Plan a 3-step approach to learn a new language in 3 months.",
        "balanced",
    ),
    EvalPrompt(
        "reason-11",
        "reasoning",
        "What is the next number in the sequence 2, 4, 8, 16, ...?",
        "balanced",
        ["32"],
    ),
    EvalPrompt(
        "reason-12",
        "reasoning",
        "Compute the derivative of x^2 + 3x and explain each step.",
        "deep",
        ["2x + 3", "2x+3"],
    ),
    EvalPrompt(
        "reason-13",
        "reasoning",
        "Design and justify a caching strategy for a read-heavy API; weigh the trade-offs step by step.",
        "deep",
    ),
    EvalPrompt(
        "reason-14",
        "reasoning",
        "If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?",
        "balanced",
        ["5"],
    ),
    # --- tool_use (12) -----------------------------------------------------
    EvalPrompt(
        "tool-01",
        "tool_use",
        "List the files in the current directory.",
        "balanced",
        ["file"],
    ),
    EvalPrompt(
        "tool-02",
        "tool_use",
        "Read README.md and summarize it in 3 bullet points.",
        "balanced",
    ),
    EvalPrompt(
        "tool-03",
        "tool_use",
        "Calculate 1234 * 5678 using a tool.",
        "balanced",
        ["7006652"],
    ),
    EvalPrompt(
        "tool-04",
        "tool_use",
        "Write a Python function that reverses a string, then run it on 'hello'.",
        "deep",
        ["olleh"],
    ),
    EvalPrompt(
        "tool-05",
        "tool_use",
        "Search my notes for anything about the Q3 budget.",
        "balanced",
    ),
    EvalPrompt("tool-06", "tool_use", "Find all .py files modified today.", "balanced"),
    EvalPrompt(
        "tool-07",
        "tool_use",
        "Count the number of lines in pyproject.toml.",
        "balanced",
    ),
    EvalPrompt(
        "tool-08",
        "tool_use",
        "Create a reminder for tomorrow at 9am to call the dentist.",
        "balanced",
    ),
    EvalPrompt(
        "tool-09",
        "tool_use",
        "Run a Python snippet that prints the first 5 Fibonacci numbers.",
        "deep",
        ["0, 1, 1, 2, 3", "0 1 1 2 3"],
    ),
    EvalPrompt(
        "tool-10",
        "tool_use",
        "Summarize the largest function in this codebase.",
        "deep",
    ),
    EvalPrompt(
        "tool-11",
        "tool_use",
        "What's the sum of 10, 20, 30, and 40?",
        "balanced",
        ["100"],
    ),
    EvalPrompt(
        "tool-12",
        "tool_use",
        "Generate a unit test for a function that adds two numbers.",
        "deep",
        ["assert", "def test"],
    ),
    # --- rag (12) ----------------------------------------------------------
    EvalPrompt(
        "rag-01",
        "rag",
        "According to the notes, what time is the standup?",
        "balanced",
        ["9:30", "9.30"],
        context="Team notes: Daily standup is at 9:30am in the main room. Retro is Friday.",
    ),
    EvalPrompt(
        "rag-02",
        "rag",
        "What database does the project use, per the docs?",
        "balanced",
        ["sqlite", "SQLite"],
        context="Architecture: the project stores data in a local SQLite database with FTS5 search.",
    ),
    EvalPrompt(
        "rag-03",
        "rag",
        "From the policy, how many vacation days do new hires get?",
        "balanced",
        ["15"],
        context="HR policy: New hires receive 15 vacation days per year, rising to 20 after three years.",
    ),
    EvalPrompt(
        "rag-04",
        "rag",
        "Per the recipe, how long do you bake the bread?",
        "balanced",
        ["35", "40"],
        context="Recipe: Knead for 10 minutes, proof for 1 hour, then bake at 220C for 35-40 minutes.",
    ),
    EvalPrompt(
        "rag-05",
        "rag",
        "What is the project's default port, from the config notes?",
        "balanced",
        ["8000"],
        context="Config: the server listens on port 8000 by default; override with --port.",
    ),
    EvalPrompt(
        "rag-06",
        "rag",
        "According to the changelog, what was fixed in v1.2?",
        "balanced",
        ["memory leak", "leak"],
        context="Changelog v1.2: Fixed a memory leak in the cache layer and improved startup time.",
    ),
    EvalPrompt(
        "rag-07",
        "rag",
        "From the meeting notes, who owns the API migration?",
        "balanced",
        ["Priya"],
        context="Meeting notes: Priya owns the API migration; Sam handles the frontend.",
    ),
    EvalPrompt(
        "rag-08",
        "rag",
        "Per the manual, what voltage does the device require?",
        "balanced",
        ["5V", "5 V"],
        context="Manual: The device requires a 5V USB-C power supply rated at least 2A.",
    ),
    EvalPrompt(
        "rag-09",
        "rag",
        "What's the refund window according to the terms?",
        "balanced",
        ["30 day", "30-day", "30 days"],
        context="Terms: Customers may request a full refund within 30 days of purchase.",
    ),
    EvalPrompt(
        "rag-10",
        "rag",
        "From the spec, what format are timestamps in?",
        "balanced",
        ["ISO 8601", "ISO-8601", "ISO8601"],
        context="API spec: All timestamps are returned in ISO 8601 format, UTC.",
    ),
    EvalPrompt(
        "rag-11",
        "rag",
        "Per the onboarding doc, what's the first step for a new dev?",
        "balanced",
        ["clone"],
        context="Onboarding: Step 1, clone the repo. Step 2, run the setup script. Step 3, request access.",
    ),
    EvalPrompt(
        "rag-12",
        "rag",
        "Synthesize the two notes: what's the combined deadline and owner?",
        "deep",
        ["Friday", "Priya"],
        context="Note A: The migration deadline is Friday. Note B: Priya owns the migration.",
    ),
]


def by_tier() -> dict:
    """Group prompt ids by expected tier."""
    out: dict = {}
    for p in SUITE:
        out.setdefault(p.tier, []).append(p.id)
    return out


def by_category() -> dict:
    out: dict = {}
    for p in SUITE:
        out.setdefault(p.category, []).append(p.id)
    return out


__all__ = ["EvalPrompt", "SUITE", "by_category", "by_tier"]
