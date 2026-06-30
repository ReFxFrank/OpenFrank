#!/usr/bin/env bash
# OpenFrank one-shot setup for WSL2 / Ubuntu (Debian-family).
#
# Installs the toolchain (system packages, uv, Rust 1.88, Ollama), builds the
# mandatory Rust extension, writes a local-only config, and pulls the per-tier
# models. Safe to re-run — each step is skipped if already done.
#
#   bash scripts/install/setup-wsl.sh              # full setup (+ models)
#   bash scripts/install/setup-wsl.sh --no-models  # skip the model download
#
# After it finishes:  bash scripts/start.sh
set -euo pipefail

log() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die() {
  printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PULL_MODELS=1
for arg in "$@"; do
  case "$arg" in
    --no-models) PULL_MODELS=0 ;;
    -h | --help)
      sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) die "unknown argument: $arg (try --help)" ;;
  esac
done

# --- pre-flight ------------------------------------------------------------
command -v apt-get >/dev/null 2>&1 || die \
  "This script targets Debian/Ubuntu (WSL2). On another distro, install
   build-essential/pkg-config/zstd/git/curl, uv, rustup (1.88+) and Ollama
   yourself, then run: bash scripts/start.sh"

case "$REPO_ROOT" in
  /mnt/*)
    warn "The repo lives on the Windows drive: $REPO_ROOT"
    warn "Building under /mnt is slow and OneDrive can lock build files."
    if [ "${ALLOW_WINDOWS_MOUNT:-0}" != "1" ]; then
      warn "Recommended instead:"
      warn "  cd ~ && git clone https://github.com/refxfrank/openfrank.git"
      warn "  cd openfrank && bash scripts/install/setup-wsl.sh"
      die "To build here anyway, re-run with: ALLOW_WINDOWS_MOUNT=1 bash $0"
    fi
    warn "Proceeding on the Windows mount (ALLOW_WINDOWS_MOUNT=1 set)."
    ;;
esac

# --- 1. system packages ----------------------------------------------------
log "Installing system packages (needs sudo): build tools, zstd, git..."
sudo apt-get update -y
sudo apt-get install -y build-essential pkg-config zstd git curl ca-certificates

# --- 2. uv (Python + project manager) --------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# shellcheck disable=SC1091
[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || die "uv still not on PATH; restart the shell and re-run."

# --- 3. Rust (repo pins 1.88 via rust-toolchain.toml) ----------------------
if ! command -v rustup >/dev/null 2>&1; then
  log "Installing rustup (Rust toolchain manager)..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
export PATH="$HOME/.cargo/bin:$PATH"
log "Ensuring Rust 1.88 is available (the repo requires it)..."
rustup toolchain install 1.88 >/dev/null 2>&1 || warn "could not pre-install 1.88; cargo will fetch it on build."

# --- 4. Ollama (local inference engine) ------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
  log "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

# --- 5. build OpenFrank ----------------------------------------------------
cd "$REPO_ROOT"
log "Installing Python dependencies (uv sync)..."
uv sync --extra dev --extra memory-sqlite-vec --extra memory-faiss --extra gpu-metrics
log "Building the native Rust extension with --release (takes a few minutes)..."
uv run maturin develop --release \
  --manifest-path rust/crates/openjarvis-python/Cargo.toml

# --- 6. config (local-only by default) -------------------------------------
CONFIG="${OPENJARVIS_HOME:-$HOME/.openjarvis}/config.toml"
if [ ! -f "$CONFIG" ]; then
  log "Writing default config ([runtime] local_only = true)..."
  uv run jarvis init || warn "jarvis init failed; configure $CONFIG manually (see docs/local-build/RUNNING-OFFLINE.md)."
fi

# --- 7. GPU sanity (soft) --------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  log "NVIDIA GPU visible in WSL — CUDA path OK."
else
  warn "No GPU visible in WSL (nvidia-smi failed). Install the 'CUDA on WSL'"
  warn "driver on the Windows side, or the assistant will run CPU-only."
fi

# --- 8. models -------------------------------------------------------------
if [ "$PULL_MODELS" = "1" ]; then
  log "Pulling the per-tier models (large, one-time; --no-models to skip)..."
  bash "$REPO_ROOT/scripts/start.sh" --pull-only ||
    warn "Model pull had issues; you can pull later with e.g. 'ollama pull qwen3:14b'."
fi

log "Setup complete. Launch the assistant with:  bash scripts/start.sh"
