param(
    [string]$Template = "data_analysis\raw_data\SimpleMultiBV25\templates\simple_multibv_brake_and_cut_in\simple_multibv_brake_and_cut_in_base.json",
    [int]$Episode = 10,
    [string]$ExperimentPath = "data_analysis\raw_data\SimpleMultiBV25\gui_debug",
    [string]$CondaEnvPath = "D:\Anaconda3\envs\D2RL",
    [string]$SumoHome = "",
    [int]$GuiDelay = 300
)

$ErrorActionPreference = "Stop"

$pythonRoot = $env:CONDA_PREFIX
if (-not $pythonRoot -or -not (Test-Path (Join-Path $pythonRoot "python.exe"))) {
    $pythonRoot = $CondaEnvPath
}

$pythonExe = Join-Path $pythonRoot "python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Cannot find D2RL python.exe. Activate D2RL or pass -CondaEnvPath <path-to-D2RL-env>."
}

$sumoHome = $SumoHome
if (-not $sumoHome) {
    $sumoHome = Join-Path $pythonRoot "Lib\site-packages\sumo"
}
if (-not (Test-Path (Join-Path $sumoHome "tools"))) {
    throw "SUMO_HOME candidate does not contain tools: $sumoHome"
}
$sumoBin = Join-Path $sumoHome "bin"
if (-not (Test-Path (Join-Path $sumoBin "sumo-gui.exe"))) {
    throw "Cannot find real sumo-gui.exe in SUMO_HOME bin: $sumoBin"
}

$env:SUMO_HOME = $sumoHome
$env:CONDA_PREFIX = $pythonRoot
$env:PATH = "$sumoBin;$(Join-Path $pythonRoot 'Scripts');$(Join-Path $pythonRoot 'Library\bin');$pythonRoot;$env:PATH"

& $pythonExe -m scenario_reconstruction.run_template `
    $Template `
    --episode $Episode `
    --experiment_path $ExperimentPath `
    --gui `
    --gui_delay $GuiDelay
