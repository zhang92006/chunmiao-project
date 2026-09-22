# Run from the repository root. Existing unlimited-budget v2 pilot is the baseline.
param([string]$SimulationPython = 'python')
$ErrorActionPreference = 'Continue'
$ablationManifest = 'data_analysis\raw_data\shrp2_multibv_sumo_templates_adaptivecritical_context_v2\bridge_summary.json'
$ablationPolicy = 'data_analysis\raw_data\shrp2_online_policy_seed19_v1\policy.pt'
$baselineAudit = 'data_analysis\raw_data\shrp2_closed_loop_validation14_x10_policy_seed19_v2\closed_loop_audit.json'
if (-not (Get-Command $SimulationPython -ErrorAction SilentlyContinue)) { throw 'SUMO Python environment not found' }
if (-not (Test-Path -LiteralPath $ablationManifest)) { throw 'Run this script from the repository root' }
if (-not (Test-Path -LiteralPath $ablationPolicy)) { throw 'Export and verify the Seed19 policy first' }
if (-not (Test-Path -LiteralPath $baselineAudit)) { throw 'Audited unlimited-budget v2 baseline is required' }
$baseline = Get-Content -LiteralPath $baselineAudit -Raw | ConvertFrom-Json
if (-not $baseline.audit_passed) { throw 'Unlimited-budget baseline did not pass audit' }
New-Item -ItemType Directory -Force 'data_analysis\logs' | Out-Null
$budgets = @(1, 5, 10)
foreach ($budget in $budgets) {
    $output = "data_analysis\raw_data\shrp2_closed_loop_validation14_x10_policy_seed19_budget${budget}_v1"
    if (Test-Path -LiteralPath $output) { throw "Output already exists: $output. Review prior results before rerunning." }
}
foreach ($budget in $budgets) {
    $output = "data_analysis\raw_data\shrp2_closed_loop_validation14_x10_policy_seed19_budget${budget}_v1"
    $log = "data_analysis\logs\shrp2_closed_loop_policy_seed19_budget${budget}_v1.log"
    $ablationArgs = @('-m', 'scenario_reconstruction.run_template_manifest', $ablationManifest,
        '--split', 'validation', '--repeats', '10', '--proposal_mode', 'factorized',
        '--epsilon', '0.5', '--frozen_epsilon_source', 'runtime', '--simulation_seed', '52000',
        '--online_policy', $ablationPolicy, '--online_intervention_budget', "$budget",
        '--experiment_path', $output)
    & $SimulationPython @ablationArgs 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "Simulation command failed: $output" }
    $summary = Get-Content -LiteralPath (Join-Path $output 'manifest_run_summary.json') -Raw | ConvertFrom-Json
    if ($summary.attempted -ne 140) { throw 'Expected exactly 14 templates x 10 repeats' }
    if ($summary.online_intervention_budget -ne $budget) { throw 'Manifest summary budget mismatch' }
    & $SimulationPython -m scenario_reconstruction.d2rl_closed_loop_audit $output
    if ($LASTEXITCODE -ne 0) { throw "Audit failed: $output" }
}
