#!/usr/bin/env bash
# Build the OpenFrank web UI (one-time) so `jarvis serve` can serve it.
#
# Vite is configured (frontend/vite.config.ts) to emit the production build
# straight into src/openjarvis/server/static/, which the API server mounts at
# http://127.0.0.1:8000 — so there's no separate dev server to run.
#
#   bash scripts/build-ui.sh           # install Node (if needed) + build
#   bash scripts/build-ui.sh --clean   # also wipe node_modules first
#
# The common WSL2 trap this guards against: with no Linux Node installed, the
# `node`/`npm` on PATH is the *Windows* binary (…/mnt/c/…), which then runs
# against \\wsl.localhost\… UNC paths and dies with
# "UNC paths are not supported". We detect that and install Node *inside* WSL.
set -euo pipefail

log() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die() {
  printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND="$REPO_ROOT/frontend"
CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    -h | --help)
      sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) die "unknown argument: $arg (try --help)" ;;
  esac
done

[ -d "$FRONTEND" ] || die "frontend/ not found at $FRONTEND"

# Make an nvm-installed node visible in this non-login shell, if present.
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
# shellcheck disable=SC1091
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" >/dev/null 2>&1 || true

# A "Linux" node is one that does NOT resolve under the Windows mount (/mnt/...).
# The Windows node leaking in via PATH is exactly what breaks `npm install`.
node_is_linux() {
  local p
  p="$(command -v node 2>/dev/null || true)"
  [ -n "$p" ] || return 1
  case "$p" in /mnt/* | /c/*) return 1 ;; esac
  return 0
}

install_node_linux() {
  log "Installing Node.js inside WSL (NodeSource LTS)..."
  if ! command -v sudo >/dev/null 2>&1; then
    die "sudo not available — install Node 20+ inside WSL manually, then re-run."
  fi
  if command -v apt-get >/dev/null 2>&1; then
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - ||
      die "NodeSource setup failed; install Node 20+ inside WSL and re-run."
    sudo apt-get install -y nodejs ||
      die "apt-get install nodejs failed; install Node 20+ inside WSL and re-run."
  else
    die "Not a Debian/Ubuntu system. Install Node 20+ inside this environment and re-run."
  fi
}

if node_is_linux; then
  log "Using Linux Node: $(command -v node) ($(node --version))"
else
  if command -v node >/dev/null 2>&1; then
    warn "The 'node' on PATH is the Windows build ($(command -v node))."
    warn "Running npm against WSL files with it fails ('UNC paths are not supported')."
  else
    warn "No Node.js found inside WSL."
  fi
  install_node_linux
  node_is_linux || die "Node still resolves to a non-Linux path ($(command -v node)). Open a new WSL shell and re-run."
  log "Installed Linux Node: $(command -v node) ($(node --version))"
fi

cd "$FRONTEND"

# A node_modules left behind by Windows npm is half-written (EPERM on cleanup);
# wipe it so the Linux install starts clean. --clean forces this regardless.
# Files written by Windows npm onto the WSL FS can carry perms the Linux user
# can't clear, so fall back to sudo when a plain rm is denied.
if [ "$CLEAN" = "1" ] || [ -d node_modules ]; then
  log "Removing existing node_modules (stale / Windows-written)..."
  rm -rf node_modules 2>/dev/null || sudo rm -rf node_modules ||
    die "could not remove node_modules — run: sudo rm -rf '$FRONTEND/node_modules'"
fi

log "Installing JS dependencies (npm)..."
# Prefer a reproducible install from the lockfile; fall back if it's out of sync.
if [ -f package-lock.json ]; then
  npm ci || { warn "npm ci failed (lockfile drift?); falling back to npm install."; npm install; }
else
  npm install
fi

log "Building the web UI (outputs to src/openjarvis/server/static/)..."
npm run build

STATIC="$REPO_ROOT/src/openjarvis/server/static"
if [ -f "$STATIC/index.html" ]; then
  log "Web UI built → $STATIC"
  log "Launch it with:  bash scripts/start.sh serve   (then open http://127.0.0.1:8000)"
else
  die "Build finished but $STATIC/index.html is missing — check the npm output above."
fi
