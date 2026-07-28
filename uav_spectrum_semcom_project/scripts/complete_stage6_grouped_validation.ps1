param(
    [Parameter(Mandatory = $true)]
    [string]$InitialRunnerPids
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Runner = Join-Path $ProjectDir "scripts\run_stage6_grouped_validation.py"
$Analyzer = Join-Path $ProjectDir "scripts\analyze_stage6_grouped_validation.py"
$CheckpointDir = Join-Path $ProjectDir "results\stage6\matched_reliability_grouped_validation_v1\checkpoints"
$Result = Join-Path $ProjectDir "results\stage6\matched_reliability_grouped_validation_v1\result.json"
$Analysis = Join-Path $ProjectDir "results\stage6\matched_reliability_grouped_analysis_v1\analysis.json"
$Python = (Get-Command python).Source
$Pids = @(
    $InitialRunnerPids.Split(",") |
        ForEach-Object { [int]$_.Trim() } |
        Where-Object { $_ -gt 0 }
)

Set-Location -LiteralPath $ProjectDir

foreach ($ProcessId in $Pids) {
    Wait-Process -Id $ProcessId -ErrorAction SilentlyContinue
}

foreach ($N in 8, 16, 32, 64) {
    $Checkpoint = Join-Path $CheckpointDir "n$N.json"
    $Complete = $false
    if (Test-Path -LiteralPath $Checkpoint) {
        $Value = Get-Content -LiteralPath $Checkpoint -Raw -Encoding UTF8 |
            ConvertFrom-Json
        $Complete = $Value.status -eq "complete"
    }
    if (-not $Complete) {
        & $Python $Runner --n-channels $N
        if ($LASTEXITCODE -ne 0) {
            throw "Grouped validation resume failed for N=$N"
        }
    }
}

if (-not (Test-Path -LiteralPath $Result)) {
    & $Python $Runner --merge
    if ($LASTEXITCODE -ne 0) {
        throw "Grouped validation merge failed"
    }
}

if (-not (Test-Path -LiteralPath $Analysis)) {
    & $Python $Analyzer
    if ($LASTEXITCODE -ne 0) {
        throw "Grouped validation analysis failed"
    }
}

Write-Output "grouped_validation_and_analysis_complete"
Write-Output $Result
Write-Output $Analysis
