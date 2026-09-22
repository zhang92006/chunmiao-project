# Run from the scenario_reconstruction repository root.
param(
    [string]$SimulationPython = 'python',
    [ValidateRange(1, 10)][int]$BatchCount = 3
)
$ErrorActionPreference = 'Continue' # SUMO warnings are written to native stderr.

$manifest = 'data_analysis\raw_data\shrp2_multibv_sumo_templates_adaptivecritical_context_v2\bridge_summary.json'
$policy = 'data_analysis\raw_data\shrp2_online_policy_audited5_seed19_v1\policy.pt'
$stratifiedPlan = 'data_analysis\raw_data\shrp2_closed_loop_three_seed_stratified_allocation_v1.json'
$equalPlan = 'data_analysis\raw_data\shrp2_closed_loop_equal14_allocation140_v1.json'
$comparison = 'data_analysis\raw_data\shrp2_closed_loop_stratified_replication_comparison_v1.json'

if (-not (Get-Command $SimulationPython -ErrorAction SilentlyContinue)) { throw 'SUMO Python environment not found' }
foreach ($required in @($manifest, $policy, $stratifiedPlan)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing required input: $required" }
}
New-Item -ItemType Directory -Force 'data_analysis\logs' | Out-Null

& $SimulationPython -m scenario_reconstruction.d2rl_equal_template_allocation `
    --reference $stratifiedPlan --budget 140 --output $equalPlan
if ($LASTEXITCODE -ne 0) { throw 'Failed to create the equal-template allocation' }

$arms = @(
    @{ Name = 'uniform'; Plan = $equalPlan; SeedOffset = 0 },
    @{ Name = 'stratified'; Plan = $stratifiedPlan; SeedOffset = 10000 }
)
$completed = @()
for ($batch = 1; $batch -le $BatchCount; $batch++) {
    foreach ($arm in $arms) {
        $output = "data_analysis\raw_data\shrp2_closed_loop_validation14_$($arm.Name)140_rep${batch}_audited5_seed19_budget10_v1"
        $log = "data_analysis\logs\shrp2_closed_loop_validation14_$($arm.Name)140_rep${batch}_audited5_seed19_budget10_v1.log"
        $audit = Join-Path $output 'closed_loop_audit.json'
        if (Test-Path -LiteralPath $audit) {
            $existingAudit = Get-Content -LiteralPath $audit -Raw | ConvertFrom-Json
            if (-not $existingAudit.audit_passed) { throw "Existing batch failed audit: $output" }
            $completed += @{ Name = $arm.Name; Path = $output }
            continue
        }
        if (Test-Path -LiteralPath $output) {
            throw "Incomplete output exists; inspect it before resuming: $output"
        }
        $simulationSeed = 93000 + (($batch - 1) * 20000) + $arm.SeedOffset
        $arguments = @(
            '-m', 'scenario_reconstruction.run_template_manifest', $manifest,
            '--split', 'validation',
            '--proposal_mode', 'factorized',
            '--epsilon', '0.5',
            '--frozen_epsilon_source', 'runtime',
            '--simulation_seed', [string]$simulationSeed,
            '--online_policy', $policy,
            '--online_intervention_budget', '10',
            '--stratified_allocation', $arm.Plan,
            '--experiment_path', $output
        )
        & $SimulationPython @arguments 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { throw "Simulation failed: $output" }
        $summary = Get-Content -LiteralPath (Join-Path $output 'manifest_run_summary.json') -Raw | ConvertFrom-Json
        if ($summary.attempted -ne 140 -or $summary.successful_runs -ne 140) {
            throw "Expected 140 successful rollouts: $output"
        }
        & $SimulationPython -m scenario_reconstruction.d2rl_closed_loop_audit $output
        if ($LASTEXITCODE -ne 0) { throw "Audit failed: $output" }
        $completed += @{ Name = $arm.Name; Path = $output }
    }
}

$comparisonArguments = @('-m', 'scenario_reconstruction.d2rl_stratified_replication_compare')
foreach ($item in $completed) {
    $comparisonArguments += @('--arm', "$($item.Name)=$($item.Path)")
}
$comparisonArguments += @('--output', $comparison)
& $SimulationPython @comparisonArguments
if ($LASTEXITCODE -ne 0) { throw 'Repeated-validation comparison failed' }
