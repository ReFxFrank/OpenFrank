#!/usr/bin/env bash
# OpenFrank launcher — the easy way to run the assistant.
#
# Ensures Ollama is running and the models are present, then starts jarvis.
#
#   bash scripts/start.sh                 # interactive chat (default)
#   bash scripts/start.sh ask "question" # one-shot question
#   bash scripts/start.sh serve          # run the HTTP server / API
#   bash scripts/start.sh doctor         # health check
#   bash scripts/start.sh --pull-only    # just ensure Ollama + models, then exit
#
# Override the model set with: OPENFRANK_MODELS="qwen3:14b nomic-embed-text"
set -euo pipefail

log() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Make uv / cargo available in a non-login shell.
# shellcheck disable=SC1091
[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

command -v uv >/dev/null 2>&1 ||
  { warn "uv not found — run: bash scripts/install/setup-wsl.sh"; exit 1; }

# The native extension is mandatory (security paths). Fail early with a hint.
if ! uv run python -c "import openjarvis_rust" >/dev/null 2>&1; then
  warn "Native extension not built — run: bash scripts/install/setup-wsl.sh"
  exit 1
fi

OLLAMA_URL="${OLLAMA_HOST:-http://127.0.0.1:11434}"
# Normalise a bare host:port (OLLAMA_HOST style) into a URL for curl.
case "$OLLAMA_URL" in http://* | https://*) ;; *) OLLAMA_URL="http://$OLLAMA_URL" ;; esac
# `read` returns nonzero at end-of-input on a here-string; `|| true` keeps
# `set -e` from aborting after the array is populated.
read -r -a MODELS <<<"${OPENFRANK_MODELS:-qwen3:8b qwen3:14b gpt-oss:20b nomic-embed-text}" || true

ensure_ollama() {
  if curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then return 0; fi
  command -v ollama >/dev/null 2>&1 ||
    { warn "Ollama not installed — run: bash scripts/install/setup-wsl.sh"; return 1; }
  log "Starting Ollama in the background..."
  mkdir -p "$HOME/.openjarvis"
  nohup ollama serve >"$HOME/.openjarvis/ollama.log" 2>&1 &
  for _ in $(seq 1 30); do
    if curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  warn "Ollama did not become ready; see $HOME/.openjarvis/ollama.log"
  return 1
}

ensure_models() {
  local installed
  installed="$(ollama list 2>/dev/null || true)"
  for m in "${MODELS[@]}"; do
    if printf '%s\n' "$installed" | grep -qF "$m"; then continue; fi
    log "Pulling $m (one-time download)..."
    ollama pull "$m" || warn "could not pull $m (deep/balanced queries may fail until it's present)"
  done
}

ensure_ollama || exit 1

action="${1:-chat}"
case "$action" in
  --pull-only)
    ensure_models
    log "Models ready."
    ;;
  chat)
    ensure_models
    log "Launching chat (Ctrl-C to exit)..."
    exec uv run jarvis chat
    ;;
  serve)
    ensure_models
    log "Starting server (default http://127.0.0.1:8000)..."
    exec uv run jarvis serve
    ;;
  ask)
    shift
    ensure_models
    exec uv run jarvis ask "$@"
    ;;
  *)
    # Pass any other subcommand straight through to the CLI.
    exec uv run jarvis "$@"
    ;;
esac
