# Run from the repository root. All arms use 14 validation templates x 10 repeats.
param([string]$SimulationPython = 'python')
$ErrorActionPreference = 'Continue' # Windows PowerShell treats native stderr warnings as errors.
$pilotManifest = 'data_analysis\raw_data\shrp2_multibv_sumo_templates_adaptivecritical_context_v2\bridge_summary.json'
$pilotPolicy = 'data_analysis\raw_data\shrp2_online_policy_seed19_v1\policy.pt'
if (-not (Get-Command $SimulationPython -ErrorAction SilentlyContinue)) { throw 'SUMO Python environment not found' }
if (-not (Test-Path -LiteralPath $pilotManifest)) { throw 'Run this script from the repository root' }
if (-not (Test-Path -LiteralPath $pilotPolicy)) { throw 'Export and verify the Seed19 policy first' }
New-Item -ItemType Directory -Force 'data_analysis\logs' | Out-Null
$pilotArms = @(
    @{ Name = 'policy_seed19'; Epsilon = '0.5'; Online = $true },
    @{ Name = 'fixed05'; Epsilon = '0.5'; Online = $false },
    @{ Name = 'fixed0001'; Epsilon = '0.0001'; Online = $false }
)
foreach ($pilotArm in $pilotArms) {
    $pilotOutput = 'data_analysis\raw_data\shrp2_closed_loop_validation14_x10_' + $pilotArm.Name + '_v2'
    if (Test-Path -LiteralPath $pilotOutput) { throw "Output already exists: $pilotOutput. Review prior results before rerunning." }
}
foreach ($pilotArm in $pilotArms) {
    $pilotOutput = 'data_analysis\raw_data\shrp2_closed_loop_validation14_x10_' + $pilotArm.Name + '_v2'
    $pilotLog = 'data_analysis\logs\shrp2_closed_loop_' + $pilotArm.Name + '_v2.log'
    $pilotArgs = @('-m', 'scenario_reconstruction.run_template_manifest', $pilotManifest,
        '--split', 'validation', '--repeats', '10', '--proposal_mode', 'factorized',
        '--epsilon', $pilotArm.Epsilon, '--frozen_epsilon_source', 'runtime',
        '--simulation_seed', '52000', '--experiment_path', $pilotOutput)
    if ($pilotArm.Online) { $pilotArgs += @('--online_policy', $pilotPolicy) }
    & $simulationPython @pilotArgs 2>&1 | Tee-Object -FilePath $pilotLog
    if ($LASTEXITCODE -ne 0) { throw "Simulation command failed: $pilotOutput" }
    $pilotSummary = Get-Content -LiteralPath (Join-Path $pilotOutput 'manifest_run_summary.json') -Raw | ConvertFrom-Json
    if ($pilotSummary.attempted -ne 140) { throw 'Expected exactly 14 templates x 10 repeats; inspect the manifest' }
    & $simulationPython -m scenario_reconstruction.d2rl_closed_loop_audit $pilotOutput
    if ($LASTEXITCODE -ne 0) { throw "Audit failed: $pilotOutput. Inspect closed_loop_audit.json" }
}
