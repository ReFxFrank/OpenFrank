# Running OpenJarvis Fully Local (RTX 5080 + 64 GB DDR5)

A reproducible guide to running OpenJarvis 100% on your own machine — no cloud,
GPU headroom preserved for games/work, with persistent local memory/RAG.

This guide reflects the local-build fork: Phase 1 (airgap lockdown), Phase 2
(VRAM-aware routing), Phase 3 (offline memory/RAG). Hardware-dependent numbers
(tokens/sec, exact VRAM split) must be confirmed on your rig with `nvidia-smi`
and `ollama ps`; see `baseline.json` for the targets.

---

## 0. Quickstart (WSL2 — two scripts)

If you're on WSL2/Ubuntu, the whole setup is two commands. Clone into the **WSL
home** (not `/mnt/c` — building on the Windows mount is slow and OneDrive locks
build files):

```bash
cd ~ && git clone https://github.com/refxfrank/openfrank.git && cd openfrank
bash scripts/install/setup-wsl.sh      # installs toolchain + builds + pulls models
bash scripts/start.sh                   # launches the assistant (interactive chat)
```

`start.sh` also takes `ask "question"`, `serve`, or `doctor`. The manual
step-by-step below explains what those scripts do and how to configure things;
read on if you want the details or are on native Windows.

---

## 1. Install

Pick **one** path and stick with it.

### Option A — Native Windows (recommended for a gaming rig)
Ollama runs natively on Windows with CUDA, so the assistant and your games share
the same driver stack with the least overhead.

```powershell
# 1. Install Ollama for Windows (https://ollama.com/download) — runs as a service.
# 2. Install OpenJarvis:
irm https://open-jarvis.github.io/OpenJarvis/install.ps1 | iex
```

### Option B — WSL2 (Ubuntu) with CUDA passthrough
```bash
curl -fsSL https://open-jarvis.github.io/OpenJarvis/install.sh | bash
```
WSL2 supports CUDA passthrough; install the NVIDIA CUDA-on-WSL driver on Windows
and a recent Ollama inside WSL.

**Recommendation:** native Windows. One driver stack, no WSL memory ceiling to
tune, and `nvidia-smi` / `ollama ps` report the same numbers your games see.

Verify the GPU is actually doing the work (not silently on CPU):
```
nvidia-smi              # VRAM occupied, util > 0 during a query
ollama ps               # shows the loaded model + GPU/CPU split
```
If util is 0%, fix the CUDA/driver routing before continuing.

---

## 2. Turn on the fully-local guarantee

`local_only` is **on by default**. Confirm (or force) it:

```toml
# ~/.openjarvis/config.toml
[runtime]
local_only = true          # no cloud; cloud engines fail closed; egress blocked
```
or per-process: `set OPENJARVIS_LOCAL_ONLY=1` (PowerShell: `$env:OPENJARVIS_LOCAL_ONLY=1`).

In this mode the engine factory refuses cloud backends and the egress guard
blocks every outbound connection except loopback + your local engines. To opt
back into cloud (not recommended for this build), see
`configs/openjarvis/cloud.example.toml`.

---

## 3. Set the VRAM cap (offload profile)

Hybrid CPU+GPU is the default. The **offload profile** caps how much VRAM the
assistant may use so the rest stays free for the OS, browser, and games.

```toml
[offload]
profile = "auto"          # auto | idle | multitask | gaming | cpu_only
safety_margin_gb = 0.5
resident_reserve_gb = 1.5 # room for the embedding + reranker models
flash_attention = true    # keep long context cheap
kv_cache_quant = "q8"     # "" | q8 | q4  (halves the KV-cache VRAM tax)
# multitask_budget_gb = 9.0   # override the per-profile cap if you like
```

| Profile | VRAM budget | When | Speed |
|---|---|---|---|
| `idle` | ~14 GB | GPU free | fastest |
| `multitask` (default) | ~8–10 GB | browser/IDE/work | fast |
| `gaming` | ~2–4 GB | a game owns the card | slower, stays alive |
| `cpu_only` | 0 GB | escape hatch / no GPU | slowest, always available |

`auto` selects the profile from current free VRAM. The planner sets Ollama's
`num_gpu` (GPU layer count) from the budget and never exceeds physically-free
VRAM — if even one layer won't fit it shifts fully to CPU instead of OOMing.

### Flash-attention + KV-cache quant (set before launching Ollama)
These are server-level Ollama settings, not per-request. Set them once:

```powershell
setx OLLAMA_FLASH_ATTENTION 1
setx OLLAMA_KV_CACHE_TYPE q8_0     # q4_0 for an even smaller KV cache
# then restart the Ollama service
```
(`openjarvis.engine.offload.ollama_runtime_env()` returns exactly these from
your `[offload]` config.)

---

## 4. Pull the per-tier models

The tier router (off by default; `[router] enabled = true` to use it) maps query
complexity → fast / balanced / deep. Pull the models you want resident:

```bash
ollama pull qwen3:8b           # fast tier   (~5-6 GB @ Q4)
ollama pull qwen3:14b          # balanced    (~9-10 GB @ Q4)  ← daily driver
ollama pull gpt-oss:20b        # deep (MoE)  (~13-15 GB; cheap to offload)
ollama pull nomic-embed-text   # embeddings  (<1 GB, kept resident)
```

```toml
[router]
enabled = true
fast_model = "qwen3:8b"
balanced_model = "qwen3:14b"
deep_model = "gpt-oss:20b"
```

**Context is a VRAM tax.** The KV cache grows with context length and competes
with model layers for the budget. Rough guidance per tier under `multitask`
(confirm with `nvidia-smi`):

| Tier | Model | Keep context… | Why |
|---|---|---|---|
| fast | 8B | generous (16k+) | model is small, lots of headroom |
| balanced | 14B | moderate (8–16k) | model already uses most of the budget |
| deep | 20B MoE | modest (4–8k) | leave room for experts spilling to RAM |

Enable `flash_attention` + `kv_cache_quant = "q8"` to roughly halve the KV cache
and push the safe context length up. Only raise `num_ctx` when the budget allows.

---

## 5. Local memory & RAG

Persistent vector store with local embeddings — nothing leaves the machine.

```toml
[tools.storage]
default_backend = "sqlite_vec"   # one on-disk file, survives restarts
embedding_engine = "ollama"      # nomic-embed-text via local Ollama
embedding_model = "nomic-embed-text"
rerank_enabled = true            # cross-encoder rerank + relevance threshold
rerank_backend = "auto"          # cross-encoder if installed, else lexical
rerank_min_score = 0.2           # drop weak matches (keeps the prompt small)
```

Install the optional bits:
```bash
uv sync --extra memory-sqlite-vec   # persistent vector store
uv sync --extra memory-faiss        # adds sentence-transformers for the cross-encoder reranker
```

Ingest your own docs (PDFs/notes/folders), fully on disk:
```bash
jarvis memory index ~/Documents/notes        # a folder, or a single PDF/markdown file
```

Long-term memory (durable facts/preferences across sessions) survives restarts;
only the relevant facts are loaded per turn so context — and the VRAM budget —
stays cheap.

---

## 6. Verify it's airtight

```bash
# Linux / WSL
bash scripts/verify-offline.sh
# Windows
powershell -File scripts\verify-offline.ps1
```
For the strongest proof, disable networking first (the script asserts that
egress is *blocked* and cloud engines *fail closed*, so it passes offline).

Then a real smoke test:
```bash
jarvis doctor                  # should be green
jarvis ask "summarize my notes on X"   # routed, local, GPU-backed
```

---

## 7. Self-improvement (local learning loop)

OpenJarvis can optimize its own skill prompts from your local traces — fully
local, GPU-idle-scheduled, and reversible.

```bash
# On-demand (snapshots overlays first, so it's undoable):
jarvis optimize skills --policy dspy

# Scheduled / nightly: only runs when the GPU is idle (won't fight your games):
jarvis optimize skills --policy dspy --require-idle

# Inspect and undo:
jarvis optimize snapshots                 # list overlay snapshots
jarvis optimize rollback <snapshot_id>    # restore a previous state
```

```toml
[optimize]
snapshot = true          # snapshot overlays before every run (reversible)
require_idle = true      # only run when the GPU is free
auto_rollback = true     # undo a run that regresses the benchmark
keep_threshold = 0.0     # min bench gain required to keep a run
local_optimizer_model = "qwen3:14b"   # local optimizer LM (cloud would fail closed)
```

**Nightly schedule (idle time).** Run it when you're asleep and the GPU is free.
Native Windows (Task Scheduler):
```powershell
schtasks /create /tn "jarvis-optimize" /tr "jarvis optimize skills --require-idle" /sc daily /st 03:00
```
WSL2 / Linux (cron): `0 3 * * * jarvis optimize skills --require-idle`.

Prove the gain with the benchmark, before and after:
```bash
jarvis bench skills --condition skills_on --max-samples 50
jarvis bench skills --condition skills_optimized_dspy --max-samples 50
```
With `auto_rollback`, a run that doesn't beat the baseline is reverted
automatically; either way a snapshot is kept so you can roll back by hand.

---

## 8. Measure the before/after lift

Prove "smarter" with numbers on your own rig:

```bash
# Run the ~50-prompt personal-assistant suite (chat, reasoning, tool-use, RAG):
uv run python scripts/run_local_eval.py --label after -o docs/local-build/after.json

# Render the before/after report (before = Phase 0 targets in baseline.json):
uv run python scripts/gen_eval_report.py --after docs/local-build/after.json
#   → docs/local-build/EVAL-REPORT.md  (task success, tok/s, latency, free VRAM, energy)
```

Per-turn stage costs (routing / memory / verification / generation) and the
tightest free-VRAM seen are captured by `openjarvis.telemetry.StageTimer`, so the
headroom guarantee is auditable, not just asserted.

---

## 9. Keep the GPU free while gaming

- Set `[offload] profile = "gaming"` (or `cpu_only`) before launching a game, or
  leave `profile = "auto"` and the planner will detect the busy GPU and shrink
  the budget automatically.
- In `cpu_only` the assistant runs entirely on the 64 GB of DDR5 — slower, but it
  never touches the card a game is using and never OOMs it.

---

## 10. Voice — speak replies aloud (local TTS)

The assistant can speak its replies with a **fully-local** voice (Kokoro), so it
works under `local_only` with no cloud TTS:

```bash
uv pip install kokoro soundfile          # or: uv sync --extra speech-tts
bash scripts/start.sh chat --speak       # or: uv run jarvis chat --speak
```

- `--tts-backend kokoro` (default, local) · `cartesia` / `openai_tts` are cloud
  and are blocked under `local_only`.
- `--voice af_bella` picks a voice (Kokoro: `af_heart` default, `af_bella`,
  `am_michael`, …); blank uses the backend default.
- **WSL2 audio:** playback needs an audio bridge. On Windows 11, WSLg provides
  PulseAudio automatically — install the player with
  `sudo apt-get install -y pulseaudio-utils` (gives `paplay`). If no player is
  found, the reply is still synthesized to a `.wav` and its path is logged, so
  nothing breaks.
