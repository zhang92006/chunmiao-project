# Run from the scenario_reconstruction repository root.
param([string]$SimulationPython = 'python')
$ErrorActionPreference = 'Continue' # SUMO warnings are written to native stderr.

$manifest = 'data_analysis\raw_data\shrp2_multibv_sumo_templates_adaptivecritical_context_v2\bridge_summary.json'
$policy = 'data_analysis\raw_data\shrp2_online_policy_audited5_seed19_v1\policy.pt'
$allocation = 'data_analysis\raw_data\shrp2_closed_loop_three_seed_stratified_allocation_v1.json'
$output = 'data_analysis\raw_data\shrp2_closed_loop_validation14_stratified140_audited5_seed19_budget10_v1'
$log = 'data_analysis\logs\shrp2_closed_loop_validation14_stratified140_audited5_seed19_budget10_v1.log'

if (-not (Get-Command $SimulationPython -ErrorAction SilentlyContinue)) { throw 'SUMO Python environment not found' }
foreach ($required in @($manifest, $policy, $allocation)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing required input: $required" }
}
if (Test-Path -LiteralPath $output) { throw "Output already exists: $output" }
New-Item -ItemType Directory -Force 'data_analysis\logs' | Out-Null

$arguments = @(
    '-m', 'scenario_reconstruction.run_template_manifest', $manifest,
    '--split', 'validation',
    '--proposal_mode', 'factorized',
    '--epsilon', '0.5',
    '--frozen_epsilon_source', 'runtime',
    '--simulation_seed', '83000',
    '--online_policy', $policy,
    '--online_intervention_budget', '10',
    '--stratified_allocation', $allocation,
    '--experiment_path', $output
)
& $SimulationPython @arguments 2>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) { throw "Stratified simulation failed: $output" }

$summary = Get-Content -LiteralPath (Join-Path $output 'manifest_run_summary.json') -Raw | ConvertFrom-Json
if ($summary.attempted -ne 140 -or $summary.successful_runs -ne 140) {
    throw 'Expected exactly 140 successful stratified rollouts'
}
if ($null -eq $summary.stratified_allocation) {
    throw 'Run summary is missing stratified allocation metadata'
}

& $SimulationPython -m scenario_reconstruction.d2rl_closed_loop_audit $output
if ($LASTEXITCODE -ne 0) { throw "Audit failed: $output. Inspect closed_loop_audit.json" }
