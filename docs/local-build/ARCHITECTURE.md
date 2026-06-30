# OpenJarvis — Local-Build Architecture Map

> Phase 0 recon for the "fully-local, RTX-5080-tuned" fork. This document maps
> the **real** code as checked out (package name `openjarvis`, repo `OpenFrank`),
> not the build brief. Where the brief and the code disagree, the code wins and
> the deviation is noted here.
>
> Everything below was verified by reading `src/openjarvis/` directly. File and
> line references are `path:line`.

## 1. Repository facts (verified)

| Brief claim | Reality in this checkout | Note |
|---|---|---|
| Package `src/openjarvis/` | ✅ `src/openjarvis/` | Python core |
| Rust ext via `maturin` (PyO3) | ✅ `rust/`, module `openjarvis_rust` | crate `rust/crates/openjarvis-python` builds the cdylib; `maturin develop --manifest-path rust/crates/openjarvis-python/Cargo.toml` |
| `configs/openjarvis/`, runtime `~/.openjarvis/config.toml` | ✅ | `core/paths.py` resolves the config dir (env-aware) |
| `tests/` pytest | ✅ ~6.8k tests | see `baseline.json` |
| Five primitives | ✅ `intelligence/`, `engine/`, `agents/`, `tools/`+`memory/`, `learning/` | |
| Engines behind one `InferenceEngine` | ✅ `engine/_stubs.py:54` | ABC + `EngineRegistry` |
| Eight built-in agents | ⚠️ **more than eight** registered | see §3 |
| `pyproject.toml` + `uv.lock`, managed with `uv` | ✅ | build backend `hatchling`; Rust ext built separately by `maturin` |

**Deviations from the brief worth recording:**

- The repo is the **`OpenFrank` fork of OpenJarvis**; the importable package is
  still `openjarvis` and the CLI entrypoint is still `jarvis`. All internal
  paths use `openjarvis`.
- `nvidia-ml-py` is already a top-level dependency (`pyproject.toml`), so VRAM
  introspection for the Phase 2 headroom work does **not** need a new dep.
- Agent count is **>8**: beyond the brief's list, the registry also carries
  `simple`, `orchestrator`, `native_react` (+ `react` alias), `native_openhands`,
  `operative`, `monitor_operative`, `deep_research`, `morning_digest`,
  `claude_code`, `opencode`, `openhands`, `proactive`, `rlm`
  (`agents/*.py`, `@AgentRegistry.register(...)`).
- A routing subsystem **already exists** under `learning/routing/`
  (`router.py`, `complexity.py`, `heuristic_policy.py`, `learned_router.py`,
  `RouterPolicyRegistry`). Phase 2's "smarter routing" should **extend** this,
  not invent a parallel one.

## 2. The five primitives → real files

### Intelligence — `src/openjarvis/intelligence/`
Model *definitions* and the catalog, not inference itself.
- `model_catalog.py` — `BUILTIN_MODELS`, `register_builtin_models()`,
  `merge_discovered_models()` (`intelligence/__init__.py:5`).
- Model metadata + per-model config lives in `core/config.py`
  (`IntelligenceConfig`, `core/config.py:577`).

### Engine — `src/openjarvis/engine/`  *(primary Phase 1 surface)*
The `InferenceEngine` ABC and all backend adapters.
- `_stubs.py:54` — `InferenceEngine(ABC)`. Key contract: `generate()`,
  `stream()`, `stream_full()`, `list_models()`, `health()`, `can_serve()`,
  `close()`, `prepare()`. **Class attribute `is_cloud: bool = False`** —
  cloud adapters set it `True` (`cloud.py:317`, `litellm.py:30`). This flag is
  the canonical lockdown signal used in Phase 1.
- `__init__.py` — imports `ollama` + `openai_compat_engines` eagerly, then
  best-effort imports the optional `cloud`, `litellm`, `gemma_cpp` modules
  (registers them only if their SDK deps are present).
- `_discovery.py` — the **engine factory**: `_make_engine(key, config)`
  (`:34`), `discover_engines(config)` (`:104`, concurrent health probes),
  `get_engine(config, engine_key, model)` (`:158`, ordered try + fallback).
  This is where Phase 1 fences off cloud adapters.
- `core/registry.py` — `EngineRegistry` (decorator registry,
  `@EngineRegistry.register("name")`).
- Registered engine keys (verified):
  - Local: `ollama` (default; `ollama.py`), `vllm`, `sglang`, `llamacpp`,
    `mlx`, `lmstudio`, `exo`, `nexa`, `uzu`, `apple_fm`, `lemonade`
    (data-driven in `openai_compat_engines.py`), `gemma_cpp`.
  - **Cloud (fenced in `local_only`): `cloud`** (OpenAI/Anthropic/Google/
    OpenRouter/MiniMax/DeepSeek/Codex, `cloud.py:312`) and **`litellm`**
    (`litellm.py`, can proxy to cloud). Both expose `is_cloud = True`.

### Agents — `src/openjarvis/agents/`
Orchestration strategies over the engine + tools.
- `_stubs.py` — `BaseAgent` ABC; `AgentRegistry`.
- One module per agent (see §1 for the registered set). The brief's
  "orchestrator upgrade" (Phase 2) extends `orchestrator.py` / `native_react.py`.

### Tools & Memory — `src/openjarvis/tools/`, `src/openjarvis/memory/`
- `tools/` — tool specs + `ToolRegistry`; `tools/storage/` holds the memory
  backends (`StorageConfig` lives here; `JarvisConfig.memory` is a
  backward-compat alias for `tools.storage`, `core/config.py:1607`).
- `memory/` — `FactStore` + retrieval (`FactStoreRegistry`, `MemoryRegistry`).
- `tools/storage/embeddings.py` — embedding integration (Phase 3 surface).

### Learning — `src/openjarvis/learning/`
- `optimize/` — DSPy/GEPA/ACE optimizers (`DSPyOptimizerConfig` etc. in
  `core/config.py`). Drives `jarvis optimize skills` (Phase 5).
- `routing/` — the existing model router (Phase 2 surface).
- `training/`, `spec_search/`, `intelligence/`, `agents/` — supporting policies.

### Cross-cutting (not a primitive, but central)
- **Config** — `core/config.py` (~2.2k lines). `JarvisConfig` (`:1573`) is a
  nested dataclass tree; `load_config()` (`:1810`, `lru_cache`d) detects
  hardware, builds defaults, overlays `~/.openjarvis/config.toml`. TOML overlay
  via `_apply_toml_section()` (`:1700`); valid sections are derived from the
  dataclass fields (`_SETTABLE_SECTIONS`, `:1624`). Default config is rendered
  by `generate_minimal_toml()` / `generate_default_toml()` (`:1902` / `:1939`).
- **Security** — `core/registry.py` plus `security/` (`ssrf.py`,
  `injection_scanner.py`, `rate_limiter.py`, `audit.py`, `scanner.py`,
  `signing.py`, `subprocess_sandbox.py`). Note `ssrf.py` *blocks private IPs*
  (anti-SSRF) — that is the **inverse** of the `local_only` egress policy
  (which *allows* loopback and blocks the public internet); the two are
  complementary, see Phase 1.
- **Rust extension** — `rust/crates/*`; the cdylib `openjarvis_rust` is the
  mandatory backend for security-critical paths (`_rust_bridge.py`,
  `security/rate_limiter.py`, scanners). Phase 6's DoS fix lives in
  `rust/crates/openjarvis-skills/src/lib.rs` (`decode_hex` / `verify_signature`).
- **Telemetry** — `telemetry/` (`InstrumentedEngine`, `TelemetryStore`) wraps
  the engine so energy/FLOPs/latency/cost are recorded per turn. This is where
  Phase 2/7 attribute per-stage cost and free-VRAM.

## 3. Request path (CLI/API → response)

Traced from `cli/ask.py`:

```
jarvis ask "..."                         cli/ask.py
  └─ load_config()                       core/config.py:1810  (hardware + TOML)
  └─ register_builtin_models()           intelligence/model_catalog.py
  └─ discover_engines(config)            engine/_discovery.py:104  (health probes)
     └─ get_engine(config, key, model)   engine/_discovery.py:158  (select backend)
        └─ _make_engine(key, config)     engine/_discovery.py:34   ◄── Phase 1 gate
  └─ InstrumentedEngine(engine)          telemetry/instrumented_engine.py
  └─ agent = AgentRegistry.create(...)   agents/  (simple / orchestrator / react / …)
     └─ agent.run(messages)
        ├─ (optional) router policy       learning/routing/  ◄── Phase 2 tier choice
        ├─ tool calls                     tools/  +  memory/   ◄── Phase 3 RAG
        └─ engine.generate()/.stream()    engine/<backend>.py  → Ollama/vLLM/…
  └─ telemetry recorded                   telemetry/store.py
  └─ response rendered                    rich Console
```

The server path (`cli/serve.py` → `server/`) is the same below `get_engine`;
it adds an HTTP/auth layer (`server/auth_middleware.py`, needs `starlette`).

## 4. Where each local-build phase hooks in

| Phase | Primary files | Mechanism |
|---|---|---|
| **0** recon/baseline | `docs/local-build/*` | this doc + `baseline.json` |
| **1** local lockdown | `core/config.py` (`RuntimeConfig`), `engine/_discovery.py`, new `security/egress.py`, `configs/openjarvis/cloud.example.toml`, `scripts/verify-offline.*` | `local_only` flag → factory refuses `is_cloud` engines (fail closed); egress guard allowlists loopback + local engine hosts |
| **2** routing + VRAM | `learning/routing/*`, new offload-profile module, `core/config.py` (hardware/`GpuInfo`), `nvidia-ml-py` | tier router + VRAM-budget → Ollama `num_gpu` / llama.cpp `--n-gpu-layers` |
| **3** memory + RAG | `tools/storage/*`, `memory/*` | `sqlite-vec`, local embeddings (`nomic-embed-text` via Ollama), reranker, KV-quant docs |
| **4** skills + tools | `skills/*`, `tools/*`, Phase-1 egress guard | curated local skills; egress-gated |
| **5** learning loop | `learning/optimize/*`, `cli/optimize_cmd.py`, `cli/bench_cmd.py` | DSPy on local traces, scheduled at idle, snapshot+rollback |
| **6** security | `rust/crates/openjarvis-skills/src/lib.rs` | harden `decode_hex` (DoS), regression tests; `docs/local-build/SECURITY.md` |
| **7** eval/telemetry/docs | `telemetry/*`, `evals/*`, `docs/local-build/RUNNING-OFFLINE.md` | before/after report, per-stage + VRAM attribution |

## 5. Environment note (recon machine ≠ target machine)

Recon/CI for this fork ran in a **Linux CPU-only container** (no NVIDIA GPU, no
`nvidia-smi`, no Ollama installed; `cargo` present, `maturin` installed via
`uv sync --extra dev`). Consequences:

- The Rust extension **builds and loads** here (`maturin develop` →
  `openjarvis_rust` importable); Phase 6 is fully verifiable.
- Phase 1 (config/factory/egress) is fully verifiable here — it is pure
  Python + sockets and needs no GPU.
- Hardware-dependent numbers in the brief's per-tier table (tokens/sec, real
  VRAM split, `ollama ps`) **cannot** be measured on this box. Those are
  recorded as *targets* in `baseline.json` and must be re-measured on the
  actual RTX 5080 rig (`nvidia-smi`, `ollama ps`) before being trusted — see
  `RUNNING-OFFLINE.md` (Phase 7).

The green-test contract (`uv run pytest tests/ -v`) holds with the caveat that
some suites need **optional** extras not installed in the recon container
(`starlette` for the server, `polars` for the framework-comparison table,
`tavily`/web-search keys, energy backends). Those pre-existing failures are
environmental, enumerated in `baseline.json`, and unrelated to local-build code.
