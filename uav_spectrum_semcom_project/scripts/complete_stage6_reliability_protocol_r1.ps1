param(
    [Parameter(Mandatory = $true)]
    [string]$InitialRunnerPids
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Runner = Join-Path $ProjectDir "scripts\run_stage6_reliability_protocol_r1_grouped.py"
$Analyzer = Join-Path $ProjectDir "scripts\analyze_stage6_reliability_protocol_r1_grouped.py"
$CheckpointDir = Join-Path $ProjectDir "results\stage6\reliability_protocol_r1_grouped_v1\checkpoints"
$Result = Join-Path $ProjectDir "results\stage6\reliability_protocol_r1_grouped_v1\result.json"
$Analysis = Join-Path $ProjectDir "results\stage6\reliability_protocol_r1_analysis_v1\analysis.json"
$Python = (Get-Command python).Source
$RunnerPids = @(
    $InitialRunnerPids.Split(",") |
        ForEach-Object { [int]$_.Trim() } |
        Where-Object { $_ -gt 0 }
)

Set-Location -LiteralPath $ProjectDir

foreach ($ProcessId in $RunnerPids) {
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
            throw "R1 grouped validation resume failed for N=$N"
        }
    }
}

if (-not (Test-Path -LiteralPath $Result)) {
    & $Python $Runner --merge
    if ($LASTEXITCODE -ne 0) {
        throw "R1 grouped merge failed"
    }
}

if (-not (Test-Path -LiteralPath $Analysis)) {
    & $Python $Analyzer
    if ($LASTEXITCODE -ne 0) {
        throw "R1 grouped analysis failed"
    }
}

Write-Output "r1_grouped_validation_and_analysis_complete"
Write-Output $Result
Write-Output $Analysis
