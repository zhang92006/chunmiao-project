# Run from the repository root. Only BV_context is likelihood-ratio guarded.
param([string]$SimulationPython = 'python')
$ErrorActionPreference = 'Continue'
$ablationManifest = 'data_analysis\raw_data\shrp2_multibv_sumo_templates_adaptivecritical_context_v2\bridge_summary.json'
$ablationPolicy = 'data_analysis\raw_data\shrp2_online_policy_audited5_seed19_v1\policy.pt'
$ablationLimits = @(100, 200, 400)
if (-not (Get-Command $SimulationPython -ErrorAction SilentlyContinue)) { throw 'SUMO Python environment not found' }
if (-not (Test-Path -LiteralPath $ablationManifest)) { throw 'Run this script from the repository root' }
if (-not (Test-Path -LiteralPath $ablationPolicy)) { throw 'Export and verify the audited Seed19 policy first' }
New-Item -ItemType Directory -Force 'data_analysis\logs' | Out-Null

foreach ($limit in $ablationLimits) {
    $ablationOutput = "data_analysis\raw_data\shrp2_closed_loop_validation14_x10_audited5_seed19_context_lr${limit}_budget10_v1"
    if (Test-Path -LiteralPath $ablationOutput) {
        throw "Output already exists: $ablationOutput. Review prior results before rerunning."
    }
}

foreach ($limit in $ablationLimits) {
    $ablationOutput = "data_analysis\raw_data\shrp2_closed_loop_validation14_x10_audited5_seed19_context_lr${limit}_budget10_v1"
    $ablationLog = "data_analysis\logs\shrp2_closed_loop_validation14_x10_audited5_seed19_context_lr${limit}_budget10_v1.log"
    $arguments = @(
        '-m', 'scenario_reconstruction.run_template_manifest', $ablationManifest,
        '--split', 'validation', '--repeats', '10',
        '--proposal_mode', 'factorized', '--epsilon', '0.5',
        '--frozen_epsilon_source', 'runtime', '--simulation_seed', '52000',
        '--online_policy', $ablationPolicy, '--online_intervention_budget', '10',
        '--online_max_proposal_likelihood_ratio', "$limit",
        '--online_likelihood_ratio_guard_actor', 'BV_context',
        '--experiment_path', $ablationOutput
    )
    & $SimulationPython @arguments 2>&1 | Tee-Object -FilePath $ablationLog
    if ($LASTEXITCODE -ne 0) { throw "Simulation command failed: $ablationOutput" }
    $summary = Get-Content -LiteralPath (Join-Path $ablationOutput 'manifest_run_summary.json') -Raw | ConvertFrom-Json
    if ($summary.attempted -ne 140 -or $summary.successful_runs -ne 140) {
        throw "Expected 140 successful paired runs: $ablationOutput"
    }
    & $SimulationPython -m scenario_reconstruction.d2rl_closed_loop_audit $ablationOutput
    if ($LASTEXITCODE -ne 0) { throw "Audit failed: $ablationOutput" }
}
