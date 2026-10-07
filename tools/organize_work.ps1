<#!
Archive generated local test folders without deleting or changing their contents.
Default: report the exact plan. -Apply performs the reviewed local moves.
This does not connect to a server or move CT/checkpoints used by experiments.
#>
[CmdletBinding()]
param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$workspaceRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$workRoot = Join-Path $workspaceRoot 'work'
$archiveRoot = Join-Path $workRoot 'archive\tests'
$receiptRoot = Join-Path $workRoot 'archive\organization'
$groups = [ordered]@{
    'fingerprint_fixture' = 'regions\fingerprint'
    'fixed_region_fixture' = 'regions\data'
    'missing_signature' = 'regions\missing-signature'
    'resident_fixture' = 'regions\resident'
    'valid_signature' = 'regions\signature'
    'region_resume_contract' = 'regions\resume'
    'half_a_entry_archived_CPU_UNIT' = 'v1\half-a'
    'half_b_entry_CPU_UNIT' = 'v1\half-b'
    'roi_probe_parser_UNIT' = 'v1\roi-parser'
    'roi_recovery_suite_UNIT' = 'v1\roi-recovery'
    'roi_reuse_UNIT' = 'v1\roi-reuse'
    'scope_training_entry_UNIT' = 'v1\scope-entry'
    'scope_UNIT' = 'v1\scope-copy'
    'transition_D_parallel_UNIT_complete' = 'transition\parallel-pass'
    'transition_D_parallel_UNIT_failure' = 'transition\parallel-failure'
    'transition_D_preparation_UNIT' = 'transition\prepare'
    'transition_storage_UNIT' = 'transition\storage'
    'v14_version_UNIT' = 'v1\version'
    'v17_collector_metadata_UNIT' = 'transition\metadata'
    'v1_bounded_controller_UNIT' = 'v1\scope-controller'
    'v1_bounded_scope_UNIT' = 'v1\scope'
    'v1_snapshot_bytecode_UNIT' = 'v1\bytecode'
    'v1_telemetry_entry_metadata_UNIT' = 'v1\telemetry'
    'v1x_runtime_metadata_UNIT' = 'v1\runtime'
    'v1x_serialization_metadata_UNIT' = 'v1\serialization'
}
$temporaryFamilies = @('fingerprint_fixture', 'fixed_region_fixture', 'missing_signature', 'resident_fixture', 'valid_signature', 'region_resume_contract')

function CheckedPath([string]$path) {
    $absolute = [IO.Path]::GetFullPath($path)
    if (-not $absolute.StartsWith($workRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Outside this workspace work directory: $absolute"
    }
    # Check every existing parent as well, so a junction cannot redirect a move.
    $cursor = $absolute
    while ($cursor.Length -gt $workRoot.Length) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Linked paths are not archive targets: $cursor"
            }
        }
        $cursor = Split-Path -Parent $cursor
    }
    return $absolute
}

$plan = [Collections.Generic.List[object]]::new()
$counters = @{}
foreach ($folder in Get-ChildItem -LiteralPath $workRoot -Directory -Force | Sort-Object Name) {
    foreach ($prefix in $groups.Keys) {
        $suffix = if ($temporaryFamilies -contains $prefix) { '[a-z0-9_]{8}' } else { '[0-9a-f]{32}' }
        if ($folder.Name -notmatch ('^' + [regex]::Escape($prefix) + '_' + $suffix + '$')) { continue }
        $source = CheckedPath $folder.FullName
        $groupRoot = CheckedPath (Join-Path $archiveRoot $groups[$prefix])
        if (-not $counters.ContainsKey($prefix)) { $counters[$prefix] = 0 }
        do {
            $counters[$prefix]++
            $destination = CheckedPath (Join-Path $groupRoot ('{0:d3}' -f $counters[$prefix]))
        } while (Test-Path -LiteralPath $destination)
        $plan.Add([pscustomobject]@{ source=$source; destination=$destination; group=$groups[$prefix] })
        break
    }
}
if (-not $Apply) {
    [pscustomobject]@{ mode='plan'; directories=$plan.Count; deleted_files=0; groups=@($plan | Group-Object group | Select-Object Name,Count); moves=$plan } | ConvertTo-Json -Depth 6
} elseif ($plan.Count -eq 0) {
    [pscustomobject]@{ mode='apply'; moved_directories=0; deleted_files=0; message='No matching generated fixtures remain.' } | ConvertTo-Json
} else {
    $receipt = CheckedPath (Join-Path $receiptRoot ('tests-' + (Get-Date -Format 'yyyyMMdd-HHmmss')))
    if (Test-Path -LiteralPath $receipt) { throw "Receipt already exists: $receipt" }
    [IO.Directory]::CreateDirectory($receipt) | Out-Null
    $plan | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $receipt 'plan.json') -Encoding utf8
    $before = [Collections.Generic.List[object]]::new()
    foreach ($move in $plan) {
        foreach ($item in Get-ChildItem -LiteralPath $move.source -Recurse -Force) {
            CheckedPath $item.FullName | Out-Null
            if ($item.PSIsContainer) { continue }
            $relative = $item.FullName.Substring($move.source.Length + 1)
            $before.Add([pscustomobject]@{
                original=$item.FullName; current=(Join-Path $move.destination $relative)
                bytes=$item.Length; sha256=(Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            })
        }
    }
    $before | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $receipt 'files.json') -Encoding utf8
    Write-Output "Verified pre-move inventory: $($plan.Count) folders, $($before.Count) files."
    foreach ($move in $plan) {
        CheckedPath $move.source | Out-Null
        CheckedPath $move.destination | Out-Null
        if (Test-Path -LiteralPath $move.destination) { throw "Refuse overwrite: $($move.destination)" }
        [IO.Directory]::CreateDirectory((Split-Path -Parent $move.destination)) | Out-Null
        Move-Item -LiteralPath $move.source -Destination $move.destination
    }
    foreach ($entry in $before) {
        $item = Get-Item -LiteralPath $entry.current
        $hash = (Get-FileHash -LiteralPath $entry.current -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($item.Length -ne $entry.bytes -or $hash -ne $entry.sha256) { throw "Moved file differs: $($entry.current)" }
    }
    $summary = [pscustomobject]@{
        moved_directories=$plan.Count; moved_files=$before.Count
        bytes=($before | Measure-Object -Property bytes -Sum).Sum
        sha256_and_size_verified=$true; deleted_files=0; reclaimed_bytes=0
        training_started=$false; server_changed=$false; receipt=$receipt
    }
    $summary | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $receipt 'summary.json') -Encoding utf8
    $summary | ConvertTo-Json -Depth 4
}
