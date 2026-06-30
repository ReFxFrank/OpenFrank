# CHANGELOG — Local-Build Fork

Changes that turn OpenJarvis into a fully-local, RTX-5080-tuned assistant, with
the rationale for each. Companion to `docs/local-build/`. Newest phase first.

The recon/CI machine for this fork is a **Linux CPU-only container** (no NVIDIA
GPU, no Ollama). Code-level guarantees (config, engine factory, egress guard,
the Rust DoS fix) are fully verified here; GPU-dependent numbers (tokens/sec,
VRAM split) are recorded as targets to re-measure on the real rig — see
`docs/local-build/baseline.json`.

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
