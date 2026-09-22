param(
    [Parameter(Mandatory=$true)][string]$SourceRoot,
    [Parameter(Mandatory=$true)][string]$Python,
    [string]$OutputRoot = 'data_analysis/raw_data/highd_native_validation_frozen_v1',
    [ValidateRange(1,100)][int]$Repeats = 5
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Python executable is missing' }
if (Test-Path -LiteralPath $OutputRoot) { throw 'Output already exists; use a fresh OutputRoot' }
$longModel = 'data_analysis/raw_data/highd_longitudinal_calibration_v2/highd_longitudinal_ndd_calibrated_v2.npz'
$lateralModel = 'data_analysis/raw_data/highd_lane_change_context_free_flow_v1/highd_lane_change_context_v1.npz'
$lateralConfig = 'configs/highd_lane_change_context_free_flow_v1.json'
$receipt = Get-Content -Raw -LiteralPath 'data_analysis/raw_data/highd_longitudinal_calibration_v2/longitudinal_calibration_summary.json' | ConvertFrom-Json
if ((Get-FileHash -LiteralPath $longModel -Algorithm SHA256).Hash -ne $receipt.selected_model_sha256) {
    throw 'Longitudinal model differs from the frozen calibration result'
}
if ((Get-FileHash -LiteralPath $lateralModel -Algorithm SHA256).Hash -ne '11f2fe0a9d7b7355eccb9d3467a5c4749bbac68ea94eafa265d9ddf7813bbbde') {
    throw 'Free-flow lateral model differs from the calibrated candidate'
}
if ((Get-FileHash -LiteralPath $lateralConfig -Algorithm SHA256).Hash -ne '239c0ec5be529bf1484684e83511c36b9292452897b67a6688537febcaa07645') {
    throw 'Free-flow lateral config differs from the calibrated candidate'
}
$templates = Join-Path $OutputRoot 'templates'
$rollouts = Join-Path $OutputRoot 'rollouts'
& $Python -m scenario_reconstruction.highd_naturalistic_reference --source_root $SourceRoot --split_manifest 'data/processed/highd_stratified_v2/manifest.json' --config 'configs/highd_naturalistic_validation_v1.json' --output $templates
if ($LASTEXITCODE -ne 0) { throw 'Native validation export failed' }
$manifest = Join-Path $templates 'native_manifest.json'
& $Python -m scenario_reconstruction.highd_naturalistic_rollout $manifest --split validation --limit 60 --repeats $Repeats --seed 7 --longitudinal_model $longModel --context_model $lateralModel --context_config $lateralConfig --output $rollouts
if ($LASTEXITCODE -ne 0) { throw 'Native closed-loop validation failed' }
& $Python -m scenario_reconstruction.highd_naturalistic_compare $manifest $rollouts --output (Join-Path $OutputRoot 'native_comparison.json')
if ($LASTEXITCODE -ne 0) { throw 'Native distribution comparison failed' }
