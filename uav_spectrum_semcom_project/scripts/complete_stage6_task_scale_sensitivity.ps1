param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Runner = Join-Path $Project "scripts\run_stage6_task_scale_sensitivity.py"
$Analyzer = Join-Path $Project "scripts\analyze_stage6_task_scale_sensitivity.py"
$LogDir = Join-Path $Project "results\stage6\task_scale_sensitivity_grouped_v1\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Processes = @()
foreach ($N in @(8, 16, 32, 64)) {
    $Stdout = Join-Path $LogDir "n${N}.stdout.log"
    $Stderr = Join-Path $LogDir "n${N}.stderr.log"
    $Processes += Start-Process -FilePath $Python `
        -ArgumentList @($Runner, "--run-n", "$N") `
        -WorkingDirectory $Project `
        -RedirectStandardOutput $Stdout `
        -RedirectStandardError $Stderr `
        -WindowStyle Hidden `
        -PassThru
}

$Processes | Wait-Process
$Failed = @()
foreach ($Process in $Processes) {
    $Process.Refresh()
    if ($Process.ExitCode -ne 0) {
        $Failed += $Process.Id
    }
}
if ($Failed.Count -gt 0) {
    throw "S6.7c shards failed without retry: $($Failed -join ', ')"
}

& $Python $Runner --merge
if ($LASTEXITCODE -ne 0) {
    throw "S6.7c merge failed"
}
& $Python $Analyzer
if ($LASTEXITCODE -ne 0) {
    throw "S6.7c analysis failed"
}
