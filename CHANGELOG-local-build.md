# CHANGELOG — Local-Build Fork

Changes that turn OpenJarvis into a fully-local, RTX-5080-tuned assistant, with
the rationale for each. Companion to `docs/local-build/`. Newest phase first.

The recon/CI machine for this fork is a **Linux CPU-only container** (no NVIDIA
GPU, no Ollama). Code-level guarantees (config, engine factory, egress guard,
the Rust DoS fix) are fully verified here; GPU-dependent numbers (tokens/sec,
VRAM split) are recorded as targets to re-measure on the real rig — see
`docs/local-build/baseline.json`.

---

## Phase 3 — Memory & RAG upgrade (all offline)

**Goal:** add a persistent local vector store, a local reranker with a relevance
threshold, and keep long context within the VRAM budget — all 100% on disk/local.

> The existing stack already had dense/FAISS/hybrid/BM25/chunking/ingest, so this
> phase **extends** it rather than rebuilding. The real gaps were a reranker, the
> brief's preferred sqlite-vec store, and KV-cache/flash-attn wiring.

- **Persistent local vector store (`tools/storage/sqlite_vec_backend.py`).** A
  `MemoryBackend` (`sqlite_vec`) built on the sqlite-vec extension: vectors +
  content + metadata live in **one on-disk SQLite file**, writes are
  transactional, and **memory survives restarts** (the FAISS backend is in-RAM).
  Embeds via the local `OllamaEmbedder` (`nomic-embed-text`) by default — never a
  hosted API. Clear errors if the extension/loader is missing. New optional extra
  `memory-sqlite-vec`. FAISS stays available for large in-memory indexes; sqlite-vec
  is the default *persistent* choice (justified in the module docstring).

- **Local cross-encoder reranker (`tools/storage/rerank.py`).** A `Reranker`
  abstraction with `CrossEncoderReranker` (local `sentence-transformers`, lazy) and
  a dependency-free `LexicalReranker` (BM25-lite) fallback that never reaches the
  network. `rerank()` re-scores, applies a **relevance threshold**, and truncates;
  `RerankingMemory` wraps *any* base backend (over-fetch → rerank → threshold), so
  it composes with sqlite/FAISS/hybrid. `get_reranker("auto")` uses the
  cross-encoder if installed, else lexical.

- **Backend factory (`tools/storage/factory.py`).** `build_backend(config)` — one
  tested entry point that builds the configured base backend and composes the
  rerank stage when `rerank_enabled`, so CLI/server get identical wiring.

- **Config.** `StorageConfig` gains local-embedding settings (`embedding_engine`
  = ollama, `embedding_model` = nomic-embed-text) and rerank settings
  (`rerank_enabled` (opt-in), `rerank_backend`, `rerank_model`, `rerank_min_score`,
  `rerank_fetch_multiplier`).

- **Long context within budget.** `engine/offload.ollama_runtime_env()` derives
  the server-level `OLLAMA_FLASH_ATTENTION` / `OLLAMA_KV_CACHE_TYPE` env from the
  `[offload]` config (KV quant requires flash-attn); the context-vs-free-VRAM
  tradeoff per tier is documented in `RUNNING-OFFLINE.md`.

- **Docs.** New `docs/local-build/RUNNING-OFFLINE.md`: install (native Windows vs
  WSL2), `local_only` + VRAM cap, per-tier model pulls, flash-attn/KV-quant, local
  memory/RAG + doc ingestion, the verify script, and gaming/CPU-fallback.

- **Tests (31):** `tests/tools/storage/test_rerank.py`,
  `test_sqlite_vec_backend.py` (offline via a deterministic hashing embedder;
  persistence-across-restart), `test_rag_eval.py` (a small RAG eval — reranking
  strictly improves MRR + precision@1), `tests/core/test_phase3_config.py`,
  `tests/engine/test_offload_env.py`.

**Verification:** tools + core + engine + memory = 1555 passed, 0 new failures
(the lone web-search failure is the pre-existing missing-key environmental one);
ruff clean; sqlite-vec round-trips and survives restart; reranking improves the
RAG eval.

---

## Phase 2 — Smarter routing & VRAM-aware model management

**Goal:** route each query to the right tier (fast/balanced/deep) and make hybrid
CPU+GPU the default — split each model across VRAM + RAM sized by an offload
profile so the assistant never starves the rest of the PC and never OOMs.

- **Offload profiles + VRAM budget (the headroom guarantee, critical).** New
  `engine/offload.py`: `OffloadProfile` (idle ~14 GB / multitask ~9 GB / gaming
  ~3 GB / cpu_only), live VRAM reads via `pynvml` → `nvidia-smi` → graceful
  CPU-only fallback (`read_vram`), `auto_select_profile` from current GPU usage,
  and `plan_offload` → a concrete GPU layer count (`num_gpu`). The budget is
  `min(profile cap, live_free − margin) − resident_reserve`, so it never exceeds
  what's physically free (can't evict the user's other GPU apps). If the budget
  can't fit even one layer it **shifts to CPU** (`num_gpu=0`) rather than OOMing.
  Includes Q4/Q8/fp16 footprint + KV-cache estimators (context is a VRAM tax).

- **Tier router.** New `learning/routing/tier_router.py`: reuses the existing
  complexity analyzer, maps score → fast/balanced/deep (configurable
  boundaries), resolves tier → model+engine, and builds the offload plan into a
  single traceable `RouteDecision` (`to_trace()`). If the chosen tier can't get
  any GPU layers and a GPU is present, it **drops a tier** rather than running a
  heavy model fully on CPU. Pure decision function — unit-testable with a mocked
  `VramStatus`.

- **Config.** `[offload]` (profile/auto, margins, resident reserve, per-profile
  budget overrides, flash-attn, KV-cache quant) and `[router]` (enabled,
  tier→model+engine, score boundaries, downgrade, self-verify) in `core/config.py`,
  both in the default config generator. `router.enabled` defaults **false** so an
  explicit `-m` is never silently overridden.

- **Engine plumbing.** `engine/ollama.py` now threads `num_gpu` (+ `main_gpu`)
  into the Ollama request `options` via a new `_build_options` helper, so an
  `OffloadPlan` actually enforces the GPU/CPU split (`num_gpu=0` = cpu-only).
  Absent → Ollama auto-derives the split (unchanged behaviour).

- **Request-path integration (opt-in, traceable).** `cli/ask.py`: when
  `[router] enabled` and the user didn't pin `-m`, it routes per-query, logs the
  chosen tier/model/profile/num_gpu **every turn**, threads `num_gpu` to the
  engine, and surfaces the tier + GPU/CPU split + VRAM budget in the inference
  profile panel.

- **Tests (49):** `tests/engine/test_offload.py`, `tests/engine/test_ollama_offload_options.py`,
  `tests/learning/routing/test_tier_router.py`, `tests/core/test_phase2_config.py` —
  profile auto-select, budget capping by free VRAM, fits/partial/CPU-shift
  planning, tier classification, tier downgrade, num_gpu plumbing, config overlay.

**Verification:** engine + learning/routing + core = 791 passed, 0 failed; ruff
clean; existing `tests/cli/test_ask_router.py` still passes. Real tokens/sec,
VRAM split and the concurrent-load (assistant + game) smoke test require the RTX
5080 rig + Ollama — `baseline.json` records the targets to confirm there.

---

## Phase 6 — Security hardening (malformed-hex / signature panic DoS)

**Goal:** fix the known malformed-hex / signature-verification panic in the Rust
path — validate input and return an error instead of panicking; add a regression
test with malformed input.

- **Root cause.** `rust/crates/openjarvis-skills/src/lib.rs::decode_hex` decoded
  the manifest's hex `signature` by slicing the `&str` by byte index
  (`s[i..i + 2]`). A signature containing a multi-byte UTF-8 codepoint (e.g.
  `"aéb"`, 4 bytes) makes an index land *inside* a codepoint, so Rust panics
  with "byte index N is not a char boundary". The `signature` field is
  attacker-controlled (it comes straight from a loaded skill manifest), and the
  panic crosses the PyO3 boundary as `pyo3_runtime.PanicException` — a
  `BaseException` that ordinary `except Exception` handlers do **not** catch.
  That is a denial-of-service vector reachable from the exposed
  `openjarvis_rust.load_skill(...).verify_signature(...)` API. (The sibling
  `parse_public_key_hex` in `skills.rs` was already hardened; `decode_hex` was
  the remaining hole, and `security/signing.py` uses base64, so it is unaffected.)

- **Fix.** `decode_hex` now operates on the byte slice and never slices the
  `&str`: it rejects non-ASCII input up front and decodes via `chunks_exact(2)` +
  `to_digit(16)`, returning `Err` for odd-length / non-ASCII / non-hex input. No
  input can panic. Public behaviour is unchanged for valid hex.

- **Tests.**
  - Rust (`openjarvis-skills`): `decode_hex` valid round-trip, odd/non-hex
    rejection, and **multi-byte UTF-8 rejection without panic**; plus an
    end-to-end `verify_signature` test with a *valid* key and a malformed
    multi-byte signature (so the decode path the bug lived in is exercised).
  - Python (`tests/skills/test_signature_dos.py`): drives the PyO3
    `verify_signature` path with the RFC 8032 Test-1 public key and a battery of
    malformed signatures, asserting `False` and no exception.

**Verification:** `cargo test -p openjarvis-skills` → 8 passed; `cargo fmt`
clean; rebuilt `openjarvis_rust`; `tests/skills` → 277 passed, DoS regression →
8 passed.

---

## Phase 1 — Fully-local lockdown (airgap mode)

**Goal:** make "fully local" a *hard guarantee* instead of a default — no cloud
code path is reachable in `local_only`, and cloud engines fail closed with a
clear error rather than silently falling back.

- **`[runtime] local_only` config (default `true`).** New `RuntimeConfig`
  dataclass (`local_only`, `enforce_egress_guard`, `egress_allowlist`) on
  `JarvisConfig`; `core/config.py`. Overridable at runtime via the
  `OPENJARVIS_LOCAL_ONLY` env var (precedence over TOML). `runtime` is a
  first-class settable section (`jarvis config set runtime.local_only false`).
  *Why:* a single, discoverable switch for the whole guarantee.

- **Engine factory fails closed.** `engine/_discovery.py`: `_make_engine`
  refuses to instantiate any `is_cloud` backend in `local_only`
  (`LocalOnlyViolation`); `get_engine` raises loudly when a cloud engine is
  *explicitly* requested instead of silently falling back; `discover_engines`
  never surfaces cloud engines. *Why:* `is_cloud` is the canonical lockdown
  signal already on every adapter (`CloudEngine`, `LiteLLMEngine`); gating the
  one chokepoint covers all of them and the public API is unchanged.

- **Network egress guard.** New `security/egress.py` — a process-wide socket
  guard (patches `socket.connect`/`connect_ex`) that allows loopback + the
  configured local engine hosts + `egress_allowlist`, and blocks everything
  else with `EgressBlocked` (an `OSError` subclass, so HTTP clients surface it
  cleanly). AF_UNIX/local IPC always allowed. Opt-in by call (`enforce_local_only`,
  wired into the CLI group), never installed at import time — so importing
  `openjarvis` never changes socket behaviour. It is the inverse policy of
  `security/ssrf.py` (which blocks *private* IPs); the two are complementary.
  *Why:* a real, enforced chokepoint that every tool/engine passes through.

- **Cloud removed from the default config; opt-in file added.** Both default
  TOML generators (`generate_default_toml`, `generate_minimal_toml`) now emit
  `[runtime] local_only = true` and carry no API-key fields or active cloud
  engine. Cloud setup moved to `configs/openjarvis/cloud.example.toml` (not
  loaded automatically; documents the deliberate opt-out + env-var keys).
  *Why:* the option survives but is off and out of everyone's live config.

- **Offline verification.** `scripts/verify-offline.sh` / `.ps1` (+ shared
  `scripts/verify_offline.py`) assert: `local_only` resolves true, cloud
  backends fail closed, egress allows loopback and blocks the public internet,
  and (best effort) a local engine is reachable. Passes with the host network
  physically disabled because it checks that egress is *blocked*, not reachable.

- **Tests.** `tests/core/test_runtime_config.py`, `tests/security/test_egress.py`,
  `tests/engine/test_local_only.py` (54 tests): defaults, TOML overlay, env
  override, allowlist construction, guard block/allow/idempotency/uninstall,
  and factory fail-closed for both explicit and discovery paths.

**Verification:** `tests/core tests/engine tests/security` → 988 passed, 0
failed; `scripts/verify_offline.py` → all checks pass; ruff clean.

---

## Phase 0 — Recon, bring-up & baseline

- `docs/local-build/ARCHITECTURE.md` — maps the five primitives to real files,
  traces the CLI→engine→agent request path, and records each phase's hook
  points and deviations from the brief (e.g. `is_cloud` as the lockdown signal;
  the pre-existing `learning/routing/` subsystem; `nvidia-ml-py` already a dep).
- `docs/local-build/baseline.json` — measured CPU-container baseline
  (6771 passed; remaining failures are missing-optional-dependency /
  environmental) plus the per-tier GPU targets flagged for re-measurement on
  the rig.
- Confirmed bring-up: `uv sync --extra dev`, `maturin develop` builds the
  `openjarvis_rust` extension (~112 s), Rust-bridge tests pass after build.
