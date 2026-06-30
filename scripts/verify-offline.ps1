# Verify OpenJarvis's fully-local guarantee (airgap mode) on Windows.
#
# For the strongest proof, disable networking first, e.g. (elevated PowerShell):
#   Disable-NetAdapter -Name "Wi-Fi" -Confirm:$false
# then run this script. It asserts that local_only is on, cloud engines fail
# closed, and outbound egress is blocked — none of which need the network up.
# Re-enable with: Enable-NetAdapter -Name "Wi-Fi"
#
# Exit code 0 = guarantee holds.
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Split-Path -Parent $ScriptDir
$env:OPENJARVIS_LOCAL_ONLY = "1"

Set-Location $RepoRoot

# Prefer `uv run` (resolves the project venv); fall back to plain python.
if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv run python scripts/verify_offline.py
} else {
    python scripts/verify_offline.py
}
exit $LASTEXITCODE
