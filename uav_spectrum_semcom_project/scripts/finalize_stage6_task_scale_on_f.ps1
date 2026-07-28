param(
    [string]$Python = "python",
    [int[]]$ShardProcessIds = @(6188, 26236, 12860, 14356),
    [string]$LargeResultRoot = "F:\uav_spectrum_semcom_large_results"
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Source = Join-Path $Project "results\stage6\task_scale_sensitivity_grouped_v1"
$TargetParent = Join-Path $LargeResultRoot "stage6"
$Target = Join-Path $TargetParent "task_scale_sensitivity_grouped_v1"

$ShardProcesses = @(
    $ShardProcessIds | ForEach-Object {
        Get-Process -Id $_ -ErrorAction Stop
    }
)
$ShardProcesses | Wait-Process
$Failed = @()
foreach ($Process in $ShardProcesses) {
    $Process.Refresh()
    if ($Process.ExitCode -ne 0) {
        $Failed += $Process.Id
    }
}
if ($Failed.Count -gt 0) {
    throw "S6.7c shards failed without retry: $($Failed -join ', ')"
}

$ResolvedProject = [IO.Path]::GetFullPath($Project)
$ResolvedSource = [IO.Path]::GetFullPath($Source)
$ResolvedTarget = [IO.Path]::GetFullPath($Target)
if (-not $ResolvedSource.StartsWith($ResolvedProject, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to move a source outside the project"
}
if (-not $ResolvedTarget.StartsWith("F:\", [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to move results outside the registered F-drive root"
}
if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
    throw "S6.7c source result directory is missing"
}
if (Test-Path -LiteralPath $Target) {
    throw "S6.7c F-drive target already exists"
}

New-Item -ItemType Directory -Force -Path $TargetParent | Out-Null
Move-Item -LiteralPath $Source -Destination $Target
New-Item -ItemType Junction -Path $Source -Target $Target | Out-Null

& $Python (Join-Path $Project "scripts\run_stage6_task_scale_sensitivity.py") --merge
if ($LASTEXITCODE -ne 0) {
    throw "S6.7c merge failed"
}
& $Python (Join-Path $Project "scripts\analyze_stage6_task_scale_sensitivity.py")
if ($LASTEXITCODE -ne 0) {
    throw "S6.7c analysis failed"
}
