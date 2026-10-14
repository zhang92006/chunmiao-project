param([Parameter(Mandatory=$true)][string]$Python39)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$replicationPython = Join-Path (Get-Location) '.venv_replication/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $replicationPython)) {
    & $Python39 -m venv .venv_replication
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create isolated environment' }
}
& $replicationPython -m pip install --disable-pip-version-check -r configs/requirements-d2rl-baseline.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $replicationPython -m scenario_reconstruction.run_published_d2rl
if ($LASTEXITCODE -ne 0) { throw 'Asset preparation failed' }
