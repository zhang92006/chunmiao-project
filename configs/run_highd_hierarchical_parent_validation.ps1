param(
    [Parameter(Mandatory=$true)][string]$SourceRoot,
    [Parameter(Mandatory=$true)][string]$Python,
    [string]$OutputRoot = 'data_analysis/raw_data/highd_hierarchical_parent_validation_v1',
    [ValidateRange(1,100)][int]$Repeats = 5
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Python executable is missing' }
if (Test-Path -LiteralPath $OutputRoot) { throw 'Output already exists; use a fresh OutputRoot' }

$longModel = 'data_analysis/raw_data/highd_longitudinal_tail_calibration_v1/highd_longitudinal_gap_candidate.npz'
$lateralModel = 'data_analysis/raw_data/highd_lane_change_context_free_flow_v1/highd_lane_change_context_v1.npz'
$lateralConfig = 'configs/highd_lane_change_context_free_flow_v1.json'
$frozen = Get-Content -Raw -LiteralPath 'configs/highd_hierarchical_parent_v1.json' | ConvertFrom-Json
if ((Get-FileHash -LiteralPath $longModel -Algorithm SHA256).Hash.ToLower() -ne $frozen.longitudinal_model_sha256) {
    throw 'Longitudinal model differs from the calibration-selected candidate'
}

$templates = Join-Path $OutputRoot 'templates'
$hard = Join-Path $OutputRoot 'hard_fallback_rollouts'
$parent = Join-Path $OutputRoot 'hierarchical_parent_rollouts'
& $Python -m scenario_reconstruction.highd_naturalistic_reference --source_root $SourceRoot --split_manifest 'data/processed/highd_stratified_v2/manifest.json' --config 'configs/highd_naturalistic_validation_v1.json' --output $templates
if ($LASTEXITCODE -ne 0) { throw 'Validation template export failed' }
$manifest = Join-Path $templates 'native_manifest.json'

& $Python -m scenario_reconstruction.highd_naturalistic_rollout $manifest --split validation --limit 60 --repeats $Repeats --seed 7 --longitudinal_model $longModel --context_model $lateralModel --context_config $lateralConfig --output $hard --original_gap_mode measured_gap --empty_state_mode original_fallback
if ($LASTEXITCODE -ne 0) { throw 'Hard-fallback validation failed' }
& $Python -m scenario_reconstruction.highd_naturalistic_compare $manifest $hard --output (Join-Path $OutputRoot 'hard_fallback_comparison.json')
if ($LASTEXITCODE -ne 0) { throw 'Hard-fallback comparison failed' }

& $Python -m scenario_reconstruction.highd_naturalistic_rollout $manifest --split validation --limit 60 --repeats $Repeats --seed 7 --longitudinal_model $longModel --context_model $lateralModel --context_config $lateralConfig --output $parent --original_gap_mode measured_gap --empty_state_mode hierarchical_parent
if ($LASTEXITCODE -ne 0) { throw 'Hierarchical-parent validation failed' }
& $Python -m scenario_reconstruction.highd_naturalistic_compare $manifest $parent --output (Join-Path $OutputRoot 'hierarchical_parent_comparison.json')
if ($LASTEXITCODE -ne 0) { throw 'Hierarchical-parent comparison failed' }

Write-Host "Paired highD validation finished: $OutputRoot"
