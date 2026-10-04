param(
    [ValidateSet('verify','smoke','train','collect','prepare','evaluate','compare')][string]$Task = 'verify',
    [string]$Python = '',
    [string[]]$ExtraArgs = @()
)
$ErrorActionPreference = 'Stop'
$dynamicHandoffRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $dynamicHandoffRoot
try {
    if ([string]::IsNullOrWhiteSpace($Python)) { $Python = (Get-Command python -ErrorAction Stop).Source }
    function Invoke-DynamicPython([string[]]$TaskArgs) {
        & $Python -m scenario_reconstruction.highd_dynamic_handoff @TaskArgs
        if ($LASTEXITCODE -ne 0) { throw 'Dynamic handoff command failed; existing outputs retained.' }
    }
    switch ($Task) {
        'verify' { Invoke-DynamicPython (@('verify','--sumo') + $ExtraArgs) }
        'smoke' {
            Invoke-DynamicPython @('verify','--sumo')
            Invoke-DynamicPython (@('train','--output','outputs/dynamic_local_smoke','--iterations','2') + $ExtraArgs)
            Invoke-DynamicPython @('evaluate','--training-root','outputs/dynamic_local_smoke','--output','outputs/dynamic_local_online_smoke','--scenarios','adjacent_approach_pair','--seed','2000','--smoke')
        }
        'compare' {
            Invoke-DynamicPython (@('collect','--output','outputs/dynamic_compare_controls','--seed','2000') + $ExtraArgs)
            Invoke-DynamicPython @('prepare','--collection','outputs/dynamic_compare_controls','--output','outputs/dynamic_compare_controls_audit')
            Invoke-DynamicPython (@('evaluate','--output','outputs/dynamic_compare_policy','--seed','2000') + $ExtraArgs)
        }
        default { Invoke-DynamicPython (@($Task) + $ExtraArgs) }
    }
}
finally { Pop-Location }
