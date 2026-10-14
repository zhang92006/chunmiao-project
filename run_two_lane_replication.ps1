param(
    [switch]$Run,
    [ValidateRange(1,32)][int]$Workers = 10,
    [ValidateRange(1,10000000)][int]$EpisodesPerWorker = 20001,
    [string]$Name = 'published_d2rl_repro_long'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$replicationPython = Join-Path $PSScriptRoot '.venv_replication/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $replicationPython)) {
    throw 'Run tools/setup_published_d2rl.ps1 with a Python 3.9 executable first.'
}
$replicationArgs = @('-m','scenario_reconstruction.run_published_d2rl','--workers',"$Workers",'--episodes-per-worker',"$EpisodesPerWorker",'--name',$Name)
if ($Run) { $replicationArgs += '--run' }
& $replicationPython @replicationArgs
if ($LASTEXITCODE -ne 0) { throw "Replication command failed with exit code $LASTEXITCODE" }
