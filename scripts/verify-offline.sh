#!/usr/bin/env bash
# Verify OpenJarvis's fully-local guarantee (airgap mode).
#
# For the strongest proof, disable networking first, e.g.:
#   Linux:  sudo ip link set <iface> down      (or run inside `unshare -rn`)
#   macOS:  sudo ifconfig en0 down
# then run this script. It asserts that local_only is on, cloud engines fail
# closed, and outbound egress is blocked — none of which need the network up.
#
# Exit code 0 = guarantee holds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export OPENJARVIS_LOCAL_ONLY=1

cd "${REPO_ROOT}"

# Prefer `uv run` (resolves the project venv); fall back to plain python.
if command -v uv >/dev/null 2>&1; then
  uv run python scripts/verify_offline.py
else
  python3 scripts/verify_offline.py
fi
