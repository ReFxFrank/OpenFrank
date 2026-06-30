# Local-Build Security Notes & Threat Model

Scope: the security posture of the fully-local fork. Companion to
`ARCHITECTURE.md`. This is a short threat-model note, not a formal audit.

## Assets we protect
- **The user's data and machine.** "Fully local" means data must not leave the
  box. The headline guarantee.
- **Process availability.** The assistant must not be crashable by malformed
  input (a DoS would also undermine "always available while gaming").
- **Local credentials/config** (`~/.openjarvis/credentials.toml`, config).

## Trust boundaries
- **Untrusted: all model-facing content** — user prompts, retrieved RAG chunks,
  ingested files, tool outputs, skill manifests from third-party sources. Treat
  retrieved/file content as data, never as instructions that can change
  tool-permission boundaries (prompt-injection awareness).
- **Semi-trusted: bundled skills/tools** (shipped in-repo).
- **Trusted: the local engine** on loopback (Ollama/llama.cpp/vLLM).

## Controls in this build (by phase)

### Network egress / airgap (Phase 1)
- `[runtime] local_only = true` (default). The engine factory **fails closed**
  on any cloud backend (`LocalOnlyViolation`) — no silent cloud fallback.
- A process-wide **socket egress guard** (`security/egress.py`) allows only
  loopback + configured local engine hosts + an explicit allowlist; everything
  else raises `EgressBlocked`. It is the inverse-but-complementary policy to
  `security/ssrf.py` (which blocks private IPs to stop SSRF).
- Verify with `scripts/verify-offline.{sh,ps1}` (passes with the NIC disabled).
- **Residual risk:** a localhost proxy could forward outbound; don't run one in
  `local_only`. The guard is installed by the CLI bootstrap, not at import.

### VRAM / availability (Phase 2)
- The offload planner never exceeds physically-free VRAM and shifts to CPU
  rather than OOM, so the assistant can't crash the GPU or evict other apps.

### Memory / RAG (Phase 3)
- Embeddings and reranking are local (Ollama / sentence-transformers); no hosted
  embedding API is reachable in `local_only`. Retrieved chunks are untrusted
  context (see prompt-injection above).

### Skills & tools (Phase 4)
- Capability model: tools declare `required_capabilities`; the `ToolExecutor`
  enforces an RBAC policy **and**, in `local_only`, blocks any tool requiring a
  network capability and logs a `SECURITY_BLOCK` event. Skills run through the
  dispatcher, so a network-requiring skill is blocked there too.
- Cloud/network-requiring skills are classified (`skill_requires_network`) and
  disabled by default in `local_only`.
- Skill loading is hardened: `skills/parser.py` validates required fields,
  length, and naming; the step template renderer is a `\{(\w+)\}` regex
  substitution — **no `eval`/`exec`/format injection** — so a malicious manifest
  cannot execute arbitrary code outside the intended tool runner.
- Code execution tools run via the sandbox/CodeAct runner, not the host shell by
  default.

### Learning loop (Phase 5)
- Optimization is reversible (snapshot/rollback of overlays) and bench-gated
  (auto-rollback on regression), so a bad run can't silently degrade behaviour.
- Runs are local; a hosted optimizer LM would fail closed under `local_only`.

### DoS hardening (Phase 6)
- **Fixed:** the Rust skill verifier's `decode_hex` sliced the attacker-controlled
  hex `signature` field by byte index, so a multi-byte UTF-8 signature (e.g.
  `"aéb"`) caused a char-boundary panic that crossed the PyO3 boundary as an
  uncatchable `PanicException` (a DoS). It now decodes on bytes and rejects
  non-ASCII/odd/non-hex input. Regression tests in the Rust crate and
  `tests/skills/test_signature_dos.py`.

## Secrets & logs
- Credentials live in `credentials.toml` (0600). The secret scanner
  (`security/scanner.py`) and redaction guardrails keep secrets out of
  traces/logs; the audit log is a hash-chained SQLite table.
- The routing/telemetry logs added in this build record tiers, models, VRAM, and
  stage timings — **no prompt content or secrets**.

## Known residual risks / future work
- Prompt-injection defenses are best-effort; a determined injection in retrieved
  content could still mislead the model within its granted capabilities (it
  cannot, however, exceed the capability/egress boundaries above).
- The egress guard is socket-level; a tool using a non-socket IPC to a local
  proxy that egresses is out of scope.
- Server bind host defaults to `0.0.0.0`; for a single-user local rig prefer
  `127.0.0.1` (see `[server] host`).
