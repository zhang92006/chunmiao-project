param(
    [string]$Python = '',
    [string]$Assets = '',
    [ValidateRange(1, 100000)][int]$Repeats = 5,
    [int]$Seed = 1000,
    [string]$OutputRoot = 'outputs/dynamic_pairs_seed1000_x5_v1'
)
$ErrorActionPreference = 'Stop'
$dynamicRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $dynamicRoot
try {
    if ([string]::IsNullOrWhiteSpace($Python)) { $Python = (Get-Command python -ErrorAction Stop).Source }
    if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
    if (Test-Path -LiteralPath $OutputRoot) { throw 'Output already exists; choose a fresh OutputRoot.' }
    New-Item -ItemType Directory -Force 'outputs/logs' | Out-Null
    $dynamicLog = Join-Path 'outputs/logs' ((Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_dynamic_pairs.log')
    $dynamicArgs = @('-u', '-m', 'scenario_reconstruction.highd_dynamic_pairs_collect',
        '--output', $OutputRoot, '--repeats', $Repeats, '--seed', $Seed, '--prepare')
    if (-not [string]::IsNullOrWhiteSpace($Assets)) { $dynamicArgs += @('--assets', $Assets) }
    # SUMO collision warnings on stderr are expected, not Python failures.
    $ErrorActionPreference = 'Continue'
    & $Python @dynamicArgs 2>&1 | Tee-Object -FilePath $dynamicLog
    $dynamicExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($dynamicExit -ne 0) { throw "Dynamic-pair collection/audit failed; inspect $dynamicLog" }
    Write-Output "Finished: $OutputRoot/collection_summary.json"
    Write-Output "Sequence audit: $OutputRoot/sequence_audit/audit_summary.json"
}
finally {
    Pop-Location
}
