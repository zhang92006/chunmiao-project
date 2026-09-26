param(
    [string]$Python = '',
    [string]$SequenceManifest = 'data_analysis\raw_data\highd_cav_first_sequences_train80_v1\sequence_manifest.json',
    [string]$OutputRoot = 'data_analysis\raw_data\highd_cav_first_ppo_meanprecision_seed7_v1',
    [ValidateRange(1, 100000)][int]$Iterations = 20,
    [int]$Seed = 7,
    [string]$PpoConfig = 'configs\highd_dual_bv_meanprecision_ppo_v1.json'
)
$ErrorActionPreference = 'Stop'
$sequenceTrainRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $sequenceTrainRoot
$savedOpenblasThreads = $env:OPENBLAS_NUM_THREADS
$savedMklThreads = $env:MKL_NUM_THREADS
$savedOmpThreads = $env:OMP_NUM_THREADS
try {
    if ([string]::IsNullOrWhiteSpace($Python)) { $Python = (Get-Command python -ErrorAction Stop).Source }
    if (Test-Path -LiteralPath $OutputRoot) { throw 'Output already exists; use a fresh OutputRoot.' }
    if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
    if (-not (Test-Path -LiteralPath $SequenceManifest)) { throw "Sequence manifest not found: $SequenceManifest" }
    if (-not (Test-Path -LiteralPath $PpoConfig)) { throw "PPO config not found: $PpoConfig" }
    $sequenceManifestData = Get-Content -Raw -LiteralPath $SequenceManifest | ConvertFrom-Json
    if ($sequenceManifestData.summary.contract -ne 'highd_first_collision_critical_sequence_v1') {
        throw 'This entry requires the explicit CAV-first event dataset, not full-horizon or audit-only data.'
    }
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:OMP_NUM_THREADS = '1'
    & $Python -u -m scenario_reconstruction.highd_critical_sequence_train --sequence_manifest $SequenceManifest --output $OutputRoot --iterations $Iterations --seed $Seed --config $PpoConfig
    if ($LASTEXITCODE -ne 0) { throw "Sequence training failed. Existing output preserved: $OutputRoot" }
    Write-Output "CAV-first sequence training finished: $OutputRoot\training_summary.json"
}
finally {
    $env:OPENBLAS_NUM_THREADS = $savedOpenblasThreads
    $env:MKL_NUM_THREADS = $savedMklThreads
    $env:OMP_NUM_THREADS = $savedOmpThreads
    Pop-Location
}
