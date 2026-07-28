param(
    [string]$ProjectRoot = "",
    [string]$DataTargetRoot = "F:\uav_spectrum_semcom_data",
    [string]$ResultTargetRoot = "F:\uav_spectrum_semcom_large_results\stage6"
)

$ErrorActionPreference = "Stop"

function Get-FullPath([string]$Path) {
    return [System.IO.Path]::GetFullPath($Path).TrimEnd("\")
}

function Assert-PathUnder([string]$Path, [string]$Root, [string]$Label) {
    $fullPath = Get-FullPath $Path
    $fullRoot = Get-FullPath $Root
    if (-not ($fullPath -eq $fullRoot -or $fullPath.StartsWith($fullRoot + "\", [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "$Label path is outside the allowed root: $fullPath is not under $fullRoot."
    }
}

function Get-TreeMetrics([string]$Path) {
    $measure = Get-ChildItem -LiteralPath $Path -File -Recurse -Force -ErrorAction Stop |
        Measure-Object -Property Length -Sum
    $sum = if ($null -eq $measure.Sum) { 0L } else { [int64]$measure.Sum }
    return [pscustomobject]@{
        files = [int64]$measure.Count
        bytes = $sum
    }
}

function Test-SameMetrics($Left, $Right) {
    return ($Left.files -eq $Right.files -and $Left.bytes -eq $Right.bytes)
}

function Move-TreeToJunction([string]$Source, [string]$Target, [string]$AllowedTargetRoot) {
    Assert-PathUnder $Source $ProjectRoot "Source"
    Assert-PathUnder $Target $AllowedTargetRoot "Target"

    $sourceFull = Get-FullPath $Source
    $targetFull = Get-FullPath $Target
    $sourceItem = Get-Item -LiteralPath $sourceFull -Force

    if ($sourceItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
        $actualTarget = @($sourceItem.Target) | Select-Object -First 1
        if ((Get-FullPath $actualTarget) -ne $targetFull) {
            throw "Existing junction has an unexpected target: $sourceFull -> $actualTarget"
        }
        $metrics = Get-TreeMetrics $sourceFull
        Write-Host "SKIP already migrated: $sourceFull -> $targetFull"
        return [pscustomobject]@{
            source = $sourceFull
            target = $targetFull
            files = $metrics.files
            bytes = $metrics.bytes
            status = "already_junction"
        }
    }

    if (-not $sourceItem.PSIsContainer) {
        throw "Source is not a directory: $sourceFull"
    }

    New-Item -ItemType Directory -Path (Split-Path -Parent $targetFull) -Force | Out-Null
    $sourceMetrics = Get-TreeMetrics $sourceFull
    Write-Host "COPY $sourceFull -> $targetFull ($($sourceMetrics.files) files, $($sourceMetrics.bytes) bytes)"

    $null = & robocopy.exe $sourceFull $targetFull /E /COPY:DAT /DCOPY:DAT /R:2 /W:2 /MT:16 /XJ /NP /NFL /NDL
    $robocopyCode = $LASTEXITCODE
    if ($robocopyCode -ge 8) {
        throw "Robocopy failed with exit code $robocopyCode."
    }

    $targetMetrics = Get-TreeMetrics $targetFull
    if (-not (Test-SameMetrics $sourceMetrics $targetMetrics)) {
        throw "Copy verification failed: source $($sourceMetrics.files)/$($sourceMetrics.bytes), target $($targetMetrics.files)/$($targetMetrics.bytes)."
    }

    $backup = $sourceFull + ".c_drive_backup_20260728"
    Assert-PathUnder $backup $ProjectRoot "Backup"
    if (Test-Path -LiteralPath $backup) {
        throw "Backup path already exists; refusing to overwrite: $backup"
    }

    Move-Item -LiteralPath $sourceFull -Destination $backup
    New-Item -ItemType Junction -Path $sourceFull -Target $targetFull | Out-Null

    $junctionItem = Get-Item -LiteralPath $sourceFull -Force
    if (-not ($junctionItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
        throw "Failed to create junction: $sourceFull"
    }
    $junctionTarget = @($junctionItem.Target) | Select-Object -First 1
    if ((Get-FullPath $junctionTarget) -ne $targetFull) {
        throw "Junction target verification failed: $junctionTarget"
    }

    $junctionMetrics = Get-TreeMetrics $sourceFull
    if (-not (Test-SameMetrics $sourceMetrics $junctionMetrics)) {
        throw "Junction content verification failed; keeping the C-drive backup: $backup"
    }

    $expectedSuffix = ".c_drive_backup_20260728"
    if (-not $backup.EndsWith($expectedSuffix, [System.StringComparison]::Ordinal)) {
        throw "Backup name does not satisfy the deletion guard: $backup"
    }
    Assert-PathUnder $backup $ProjectRoot "Backup pending deletion"
    Remove-Item -LiteralPath $backup -Recurse -Force

    Write-Host "DONE $sourceFull -> $targetFull"
    return [pscustomobject]@{
        source = $sourceFull
        target = $targetFull
        files = $sourceMetrics.files
        bytes = $sourceMetrics.bytes
        status = "migrated_and_linked"
    }
}

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$ProjectRoot = Get-FullPath $ProjectRoot
$DataTargetRoot = Get-FullPath $DataTargetRoot
$ResultTargetRoot = Get-FullPath $ResultTargetRoot

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "Project directory does not exist: $ProjectRoot"
}
if ((Split-Path -Qualifier $DataTargetRoot) -ne "F:" -or (Split-Path -Qualifier $ResultTargetRoot) -ne "F:") {
    throw "All target directories must be on drive F."
}

$mappings = @(
    @{
        source = Join-Path $ProjectRoot "data\raw"
        target = Join-Path $DataTargetRoot "raw"
        allowed = $DataTargetRoot
    },
    @{
        source = Join-Path $ProjectRoot "data\processed"
        target = Join-Path $DataTargetRoot "processed"
        allowed = $DataTargetRoot
    },
    @{
        source = Join-Path $ProjectRoot "results\stage6\matched_reliability_coarse_grid_v1"
        target = Join-Path $ResultTargetRoot "matched_reliability_coarse_grid_v1"
        allowed = $ResultTargetRoot
    },
    @{
        source = Join-Path $ProjectRoot "results\stage6\matched_reliability_grouped_validation_v1"
        target = Join-Path $ResultTargetRoot "matched_reliability_grouped_validation_v1"
        allowed = $ResultTargetRoot
    },
    @{
        source = Join-Path $ProjectRoot "results\stage6\reliability_protocol_r1_grouped_v1"
        target = Join-Path $ResultTargetRoot "reliability_protocol_r1_grouped_v1"
        allowed = $ResultTargetRoot
    },
    @{
        source = Join-Path $ProjectRoot "results\stage6\task_scale_sensitivity_grouped_v1"
        target = Join-Path $ResultTargetRoot "task_scale_sensitivity_grouped_v1"
        allowed = $ResultTargetRoot
    }
)

$beforeC = Get-PSDrive -Name C
$beforeF = Get-PSDrive -Name F
$records = @()

foreach ($mapping in $mappings) {
    $records += Move-TreeToJunction -Source $mapping.source -Target $mapping.target -AllowedTargetRoot $mapping.allowed
}

$afterC = Get-PSDrive -Name C
$afterF = Get-PSDrive -Name F
$manifest = [ordered]@{
    schema_version = "uav-spectrum-semcom-storage-migration-v1"
    generated_at = (Get-Date).ToString("o")
    project_root = $ProjectRoot
    policy = "Code, documents, Git metadata and compact summaries remain on C; bulk datasets and large experiment artifacts reside on F and are exposed through NTFS junctions."
    final_data_accesses_added = 0
    c_free_bytes_before = [int64]$beforeC.Free
    c_free_bytes_after = [int64]$afterC.Free
    f_free_bytes_before = [int64]$beforeF.Free
    f_free_bytes_after = [int64]$afterF.Free
    total_bulk_files_on_f = [int64](($records | Measure-Object -Property files -Sum).Sum)
    total_bulk_bytes_on_f = [int64](($records | Measure-Object -Property bytes -Sum).Sum)
    mappings = $records
}

$manifestPath = Join-Path $ProjectRoot "results\stage6\storage_migration_to_f_v1.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
Write-Host "MANIFEST $manifestPath"
Write-Host "C free: $([math]::Round($afterC.Free / 1GB, 2)) GB; F free: $([math]::Round($afterF.Free / 1GB, 2)) GB"
