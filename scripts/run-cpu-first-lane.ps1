<#
.SYNOPSIS
    Run the CPU-only lane: build, synthetic loader/gather gates, and - when pinned model shards are supplied
    explicitly - real-file reality-check gates against those files.

.DESCRIPTION
    Three modes share one gate table so a dry-run states exactly what an executed run would do:

      DryRun   (default) print every planned command and each lane's resource requirements. Nothing is executed,
               no device is queried for confirmation, and nothing is written to disk.
      Synthetic execute the CPU-only gates that need no target model file. These are substitute measurements only:
               they can prove loader/plan/gather math on synthetic containers, never real-file parity or reality-check
               evidence against the pinned shards.
      RealFile  validate the explicitly declared pinned shards and this host's working-set/RAM/disk headroom first, then
               execute the CPU-only gates plus bench commands that read those files through zero-copy mmap spans. A
               missing shard path, an unreadable resource reading, or insufficient headroom fails naming what is expected;
               it never falls back to a synthetic fixture.

    Shards are validated and consumed in place: no lane merges shards into one file (that filled the disk in an earlier
    attempt), copies them, downloads anything, writes committed fixtures, touches bench reports by default, or stages Git
    changes. A report is written only when -ReportPath is given on an executed mode that passed validation.

.PARAMETER Mode
    DryRun | Synthetic | RealFile. Defaults to DryRun so the lane never executes anything by accident.

.PARAMETER ShardPath
    Absolute or repository-relative paths of the pinned model shards, repeated once per shard (for example the two
    -00001-of-00002 / -00002-of-00002 files). Required for RealFile; whenever supplied in any mode they are validated
    first and the real-file gate commands preview or run against those exact files. A declared path that does not exist
    fails naming it: no synthetic fixture is substituted.

.PARAMETER PleShardPath
    One of -ShardPath that streams from disk instead of staying resident (the pageable-table continuation shard). It is
    excluded from the derived resident working set and gets its own streamed-gather gate line instead of a GEMV pass.

.PARAMETER WorkingSetMB
    Resident weight-tier size in MB. 0 derives it from the declared shards minus -PleShardPath, which is what the pinned
    pair needs when every non-streamed shard stays resident; set it explicitly for a posture you have measured.

.PARAMETER HeadroomMB
    KV/prefill/scratch headroom added on top of the working set before free RAM counts as sufficient. Default 8192.

.PARAMETER MinFreeDiskMB
    Free space required on each shard's volume: room for harness scratch and report output, deliberately not room sized
    like a second copy of the model. Default 16384.

.PARAMETER Threads
    Bench thread quota (0 uses the bench command's default thread quota).

.PARAMETER ReportPath
    Write a plain-text lane log here. Only valid on Synthetic or RealFile after validation passes; DryRun refuses it so
    a preview can never leave artifacts behind. Default: no report at all.

.PARAMETER IncludeSmallModelSuites
    Also run the CPU suites that open a small pinned model from models/ (serve, parity, hybrid posture). Requires those
    files to be present in the repository working tree.

.PARAMETER IncludeQwen4ExpForward
    Opt-in bounded qwen4exp forward gate: one 'qwen4exp-forward-file' bench command over the declared shards on this lane's
    thread quota. It is not part of the default RealFile gate table, so a lane that does not ask for it behaves exactly as
    before. The command reports prefill/decode tokens-per-second or exits non-zero with one status=refused line naming which
    tensor role/posture this build cannot open against that file - a refusal is recorded as a gate failure rather than being
    swallowed, because the refusal text is itself the evidence the lane is asked to collect.

.EXAMPLE
    ./scripts/run-cpu-first-lane.ps1 -DryRun
.EXAMPLE
    ./scripts/run-cpu-first-lane.ps1 -Mode Synthetic -ReportPath artifacts/lane-runs/cpu-synthetic.txt
#>
[CmdletBinding()]
param(
    [ValidateSet('DryRun', 'Synthetic', 'RealFile')]
    [string]$Mode = 'DryRun',

    [string[]]$ShardPath = @(),

    [string]$PleShardPath = '',

    [int]$WorkingSetMB = 0,

    [int]$HeadroomMB = 8192,

    [int]$MinFreeDiskMB = 16384,

    [int]$Threads = 0,

    [string]$ReportPath = '',

    [switch]$IncludeSmallModelSuites,

    [switch]$IncludeQwen4ExpForward
)

. (Join-Path $PSScriptRoot 'lane-gates.common.ps1')

# A lane that cannot load its shared gates stops before printing a half-plan: every refusal here names the missing helper.
foreach ($required in @('Get-LaneHostSnapshot', 'Write-LaneResourceLine', 'Write-LaneGatePlan', 'Invoke-LaneGate', 'Write-LaneReport')) {
    if (-not (Test-Path -Path "Function:$required")) {
        Write-Host "lane_refusal=shared_helper_missing function=$required helper=$(Join-Path $PSScriptRoot 'lane-gates.common.ps1')"

        exit 3
    }
}

$RepoRoot = Split-Path -Parent $PSScriptRoot

# Declared shards may arrive as an array or as one comma-joined argument; both spellings name the same files.
$declaredShards = Expand-LaneDeclarationList -Values $ShardPath

# The pinned Flash-Next pair the loader gate observes, recorded here only so a refusal can name where the shards live
# instead of silently reaching for a synthetic stand-in.
$PinnedShardHint = @(
    'C:\source\github.com\Strata-data\models\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf',
    'C:\source\github.com\Strata-data\models\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf'
)

function Resolve-LanePath {
    param([string]$Path)

    if ([System.IO.Path]::IsPathRooted($Path)) { return [System.IO.Path]::GetFullPath($Path) }

    [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $Path))
}

function New-LaneGateList {
    param([string]$Mode, [object[]]$Shards, [pscustomobject]$Resources)

    $gates = [System.Collections.Generic.List[object]]::new()

    function Add-Gate {
        param($Name, $Executable, $Arguments, $Requires, $Status = 'planned')

        $gates.Add([pscustomobject]@{
            Name       = $Name
            Executable = $Executable
            Arguments  = @($Arguments)
            Requires   = $Requires
            Status     = $Status
        })
    }

    Add-Gate 'build' 'dotnet' @('build', '-c', 'Release', '--nologo', '-p:SkipWebUi=true') `
        'none (no CUDA device, no model files, no web UI toolchain dependency)'

    # CPU-only synthetic gates: these run with no target file at all.
    Add-Gate 'loader-plan-gates-synthetic' 'dotnet' @(
        'test', 'tests/Lair.Tests', '-c', 'Release', '--no-build', '--nologo',
        '--filter', 'Category=Loader&FullyQualifiedName!~Qwen4Exp',
        '--logger', 'console;verbosity=normal'
    ) 'none (synthetic GGUF containers built by the suites themselves)'

    Add-Gate 'gather-math-gates-synthetic' 'dotnet' @(
        'test', 'tests/Lair.Tests', '-c', 'Release', '--no-build', '--nologo',
        '--filter', 'Category=Moe',
        '--logger', 'console;verbosity=normal'
    ) 'none (synthetic single-layer MoE models, dense-equivalent reference math)'

    Add-Gate 'ple-gates-synthetic' 'dotnet' @() `
        'no committed PLE synthetic suite yet: this lane reports it unrunnable rather than substituting another category' `
        'no_committed_gate'

    Add-Gate 'mtp-gates-synthetic' 'dotnet' @() `
        'no committed MTP drafting suite yet: this lane reports it unrunnable rather than substituting another category' `
        'no_committed_gate'

    Add-Gate 'bench-gemv-synthetic' 'dotnet' @(
        'run', '--project', 'bench/Lair.Bench', '-c', 'Release', '--no-build', '--', 'gemv'
    ) 'none (in-process 4096x4096 Q4_K weights, no file opened)'

    if ($IncludeSmallModelSuites) {
        Add-Gate 'model-loaded-cpu-suites' 'dotnet' @(
            'test', 'tests/Lair.Tests', '-c', 'Release', '--no-build', '--nologo',
            '--filter', 'Category=Serve|Parity',
            '--logger', 'console;verbosity=normal'
        ) "models/SmolLM-135M-Instruct-F16.gguf and the other pinned small fixtures present in $RepoRoot"
    }

    # Real-file gates exist only for a validated shard declaration, so a dry-run with declared shards previews the exact
    # commands against those same files instead of previewing something an executed run would not do.
    if (@($Shards).Count -gt 0) {
        foreach ($shard in $Shards) {
            Add-Gate "reality-check-header[$([System.IO.Path]::GetFileName($shard.Path))]" 'dotnet' @(
                'run', '--project', 'bench/Lair.Bench', '-c', 'Release', '--no-build', '--', 'gguf-dump', $shard.Path
            ) "pinned shard present: header/tensor-table parse only, no payload copy"
        }

        foreach ($shard in @($Shards | Where-Object { -not $_.Streamed })) {
            $gemvArgs = @('run', '--project', 'bench/Lair.Bench', '-c', 'Release', '--no-build', '--', 'gemv-file', $shard.Path)
            if ($Threads -gt 0) { $gemvArgs += @('--threads', ([string]$Threads)) }
            Add-Gate "reality-check-gemv-file[$([System.IO.Path]::GetFileName($shard.Path))]" 'dotnet' $gemvArgs "resident working set within the declared RAM headroom: kernels read zero-copy mmap spans"

            $prefillArgs = @('run', '--project', 'bench/Lair.Bench', '-c', 'Release', '--no-build', '--', 'prefill-profile', $shard.Path)
            if ($Threads -gt 0) { $prefillArgs += @('--threads', ([string]$Threads)) }
            Add-Gate "reality-check-prefill-profile[$([System.IO.Path]::GetFileName($shard.Path))]" 'dotnet' $prefillArgs "resident weight tier plus prefill activation buffers within the declared RAM headroom"
        }

        foreach ($shard in @($Shards | Where-Object { $_.Streamed })) {
            Add-Gate "ple-disk-gather-reality-check[$([System.IO.Path]::GetFileName($shard.Path))]" 'dotnet' @(
                'run', '--project', 'bench/Lair.Bench', '-c', 'Release', '--no-build', '--', 'gguf-dump', $shard.Path
            ) "streamed shard stays on disk: header parse now, indexed row gather once the PLE seam exists"
        }

        if ($IncludeQwen4ExpForward) {
            # Any one declared shard is a valid entry point: the loader assembles its -NNNNN-of-NNNNN siblings automatically, and
            # the bounded qwen4exp seam reads only packed weight views for blocks it composes (never the PLE table span).
            $entryShard = @($Shards)[0]
            $forwardArgs = @('run', '--project', 'bench/Lair.Bench', '-c', 'Release', '--no-build', '--', 'qwen4exp-forward-file', $entryShard.Path)
            if ($Threads -gt 0) { $forwardArgs += @('--threads', ([string]$Threads)) }

            Add-Gate "reality-check-qwen4exp-forward-file[$([System.IO.Path]::GetFileName($entryShard.Path))]" 'dotnet' $forwardArgs `
                'assembled pinned shards resident within the declared RAM headroom: bounded chunk forwards over short synthetic windows; a seam refusal exits non-zero naming the unsupported role/posture rather than half-running the target'
        }
    }

    @($gates)
}

$hostSnapshot = Get-LaneHostSnapshot
$executing = ($Mode -ne 'DryRun')

Write-Host "lane=cpu-first mode=$Mode repo_root=$RepoRoot"
Write-Host 'lane_policy=no_model_download no_shard_merge_or_copy no_git_write'
if (-not $executing) { Write-Host 'lane_policy=dry_run_executes_nothing writes_no_report' }

# A report request is only meaningful for an executed, validated mode. Refusing it in dry-run keeps previews artifact-free.
if (-not [string]::IsNullOrWhiteSpace($ReportPath)) {
    if ($Mode -eq 'DryRun') {
        Write-Host "lane_refusal=dry_run_writes_no_report report_path=$ReportPath"
        exit 3
    }

    $reportFullPath = Resolve-LanePath $ReportPath
} else {
    $reportFullPath = ''
}

Write-LaneResourceLine -Prefix 'host' -Host_ $hostSnapshot
Write-Host 'lane_resources=cpu_only_gates_require_no_cuda_device no_gpu_idleness_confirmation_needed'
Write-Host 'lane_scope_note=synthetic_loader_and_gather_gates_run_without_the_target_file real_file_parity_and_reality_check_need_declared_shards_never_synthetic_substitutes'

if ($Mode -ne 'RealFile') {
    Write-Host ("real_file_requirements=pinned_shards_declared_explicitly working_set_source=auto_derived_from_declared_shards headroom_mb={0} min_free_disk_mb={1}" -f $HeadroomMB, $MinFreeDiskMB)
}

# ---------------------------------------------------------------- real-file resource validation (before any gate runs)

$shardRecords = @()
$validationFailures = [System.Collections.Generic.List[string]]::new()
$resourceRequirements = [pscustomobject]@{
    WorkingSetMB        = 0
    WorkingSetSource    = 'not_declared'
    HeadroomMB          = $HeadroomMB
    RequiredFreeRamMB   = 0
    MinFreeDiskMB       = $MinFreeDiskMB
}

if ($Mode -eq 'RealFile' -or @($declaredShards).Count -gt 0) {
    Write-Host "real_file_mode=explicit_shard_paths_required mode=$Mode pinned_shards_hint=$($PinnedShardHint -join ', ')"

    if (@($declaredShards).Count -eq 0) {
        $validationFailures.Add(
            '-Mode RealFile needs explicit -ShardPath entries for the pinned shards (expected: ' + ($PinnedShardHint -join '; ') + `
            '). Refusing to substitute synthetic fixtures for real-file reality-check evidence.')
    }

    $pleFull = if ([string]::IsNullOrWhiteSpace($PleShardPath)) { '' } else { Resolve-LanePath $PleShardPath }

    foreach ($declared in @($declaredShards)) {
        $full = Resolve-LanePath $declared
        if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
            $validationFailures.Add("missing pinned shard: $full")

            continue
        }

        $length = (Get-Item -LiteralPath $full).Length
        $volume = Get-LaneVolumeFreeMB -Path $full
        $shardRecords += [pscustomobject]@{
            Path     = $full
            Name     = [System.IO.Path]::GetFileName($full)
            Bytes    = $length
            Streamed = ($pleFull -ne '' -and [string]::Equals($pleFull, $full, [StringComparison]::OrdinalIgnoreCase))
            Volume   = $volume.Volume
            FreeMB   = $volume.FreeMB
        }

        Write-Host ("shard name={0} path={1} size={2} volume={3} volume_free_mb={4} residency={5}" -f `
            [System.IO.Path]::GetFileName($full), $full, (Format-LaneBytes $length), $volume.Volume, `
            $(if ($null -eq $volume.FreeMB) { 'unavailable' } else { $volume.FreeMB }), `
            $(if ($pleFull -ne '' -and [string]::Equals($pleFull, $full, [StringComparison]::OrdinalIgnoreCase)) { 'streamed-from-disk' } else { 'resident' }))

        if (-not $volume.Ready -or $null -eq $volume.FreeMB) {
            $validationFailures.Add("disk headroom unreadable for shard $($full): volume '$($volume.Volume)' reports no readable free space; expected at least $MinFreeDiskMB MB of scratch/report room")

            continue
        }

        if ($volume.FreeMB -lt $MinFreeDiskMB) {
            $validationFailures.Add(
                "insufficient disk headroom for shard $($full): volume '$($volume.Volume)' has $([int]$volume.FreeMB) MB free but the lane requires $MinFreeDiskMB MB (scratch and report output only: this lane never copies or merges shards)")
        }
    }

    if ($pleFull -ne '' -and @($shardRecords | Where-Object { $_.Streamed }).Count -eq 0) {
        $validationFailures.Add("-PleShardPath '$pleFull' is not one of the declared -ShardPath entries, so its streamed residency cannot be accounted")
    }

    if (@($shardRecords).Count -gt 0) {
        $residentBytes = [long](@($shardRecords | Where-Object { -not $_.Streamed }) | Measure-Object -Property Bytes -Sum).Sum
        $declaredWorkingSetMB = if ($WorkingSetMB -gt 0) { $WorkingSetMB } else { [int][math]::Ceiling($residentBytes / 1MB) }
        $resourceRequirements.WorkingSetMB = $declaredWorkingSetMB
        $resourceRequirements.WorkingSetSource = $(if ($WorkingSetMB -gt 0) { 'operator-declared' } else { "auto-derived from $($shardRecords.Count) declared shard(s), resident bytes $(Format-LaneBytes $residentBytes)" })
        $resourceRequirements.RequiredFreeRamMB = $declaredWorkingSetMB + $HeadroomMB

        Write-Host ("working_set_mb={0} working_set_source={1} headroom_mb={2} required_free_ram_mb={3}" -f `
            $declaredWorkingSetMB, $resourceRequirements.WorkingSetSource, $HeadroomMB, $resourceRequirements.RequiredFreeRamMB)

        if ($null -eq $hostSnapshot.FreeRamMB) {
            $validationFailures.Add(
                "RAM headroom unreadable on this host (source=$($hostSnapshot.RamSource)): the lane requires free_ram_mb>=$($resourceRequirements.RequiredFreeRamMB) for a working set of $declaredWorkingSetMB MB plus $HeadroomMB MB KV/prefill headroom; declare -WorkingSetMB explicitly only if you have measured that posture")
        } elseif ($hostSnapshot.FreeRamMB -lt $resourceRequirements.RequiredFreeRamMB) {
            $validationFailures.Add(
                "insufficient RAM headroom on this host: free_ram_mb=$($hostSnapshot.FreeRamMB) but the lane requires free_ram_mb>=$($resourceRequirements.RequiredFreeRamMB) (working set $declaredWorkingSetMB MB + KV/prefill headroom $HeadroomMB MB). Real-file reality-check evidence is not claimed from synthetic substitutes, so this lane refuses instead of running a smaller fixture")
        }
    }

    if (@($validationFailures).Count -gt 0) {
        Write-Host 'lane_refusal=real_file_resources_not_satisfied fallback=synthetic_fixtures_forbidden'
        foreach ($failure in $validationFailures) { Write-Host "  refusal: $failure" }

        Write-Host ("  requirement: pinned shards present; working_set_mb derived from the declared shards unless -WorkingSetMB states it (headroom_mb={0}); free_ram_mb>={1}; disk_free_mb>={2} on each shard volume" -f `
            $HeadroomMB, [Math]::Max($resourceRequirements.RequiredFreeRamMB, $HeadroomMB), $MinFreeDiskMB)
        exit 2
    }
}

# ---------------------------------------------------------------- gate table + execution

$gates = New-LaneGateList -Mode $Mode -Shards $shardRecords -Resources $resourceRequirements

Write-Host "lane_gates=$(($gates | ForEach-Object { $_.Name }) -join ',')"
foreach ($gate in $gates) {
    $planStatus = if ($gate.Status -eq 'no_committed_gate') { 'no_committed_gate' } elseif ($executing) { 'queued' } else { 'planned' }

    Write-LaneGatePlan -Name $gate.Name -Executable $gate.Executable -Arguments $gate.Arguments `
        -Requires $gate.Requires -Status $planStatus
}

$results = @()
$failures = [System.Collections.Generic.List[string]]::new()

foreach ($gate in $gates) {
    if ($gate.Status -eq 'no_committed_gate') {
        Write-Host ("gate={0} status=no_committed_gate detail={1}" -f $gate.Name, $gate.Requires)

        continue
    }

    if (-not $executing) { continue }

    # The test gates need the build gate's output (--no-build), so a failed or skipped build stops them instead of
    # reporting an unrelated missing-assets failure.
    if ($results | Where-Object { $_.Name -eq 'build' -and $_.ExitCode -ne 0 }) {
        Write-Host ("gate={0} status=skipped reason=build_gate_failed" -f $gate.Name)

        continue
    }

    $result = Invoke-LaneGate -Name $gate.Name -RepoRoot $RepoRoot -Executable $gate.Executable `
        -Arguments $gate.Arguments -Requires $gate.Requires
    $results += $result

    if ($result.ExitCode -ne 0) {
        $failures.Add("gate $($gate.Name) exited with code $($result.ExitCode): $($result.Command)")
    }
}

if (-not [string]::IsNullOrWhiteSpace($reportFullPath)) {
    $lines = @()
    if ($Mode -eq 'RealFile') {
        foreach ($shard in $shardRecords) {
            $lines += ("shard name={0} path={1} bytes={2} residency={3} volume='{4}' volume_free_mb={5}" -f `
                $shard.Name, $shard.Path, $shard.Bytes, $(if ($shard.Streamed) { 'streamed-from-disk' } else { 'resident' }), `
                $shard.Volume, $shard.FreeMB)
        }

        $lines += ("working_set_mb={0} working_set_source={1} headroom_mb={2} required_free_ram_mb={3} free_ram_mb={4}" -f `
            $resourceRequirements.WorkingSetMB, $resourceRequirements.WorkingSetSource, $HeadroomMB, `
            $resourceRequirements.RequiredFreeRamMB, $hostSnapshot.FreeRamMB)
    }

    foreach ($result in $results) {
        $lines += ("gate={0} exit_code={1} command={2}" -f $result.Name, $result.ExitCode, $result.Command)
    }

    if (@($failures).Count -gt 0) { $lines += @($failures | ForEach-Object { "failure=$_" }) }

    Write-LaneReport -Path $reportFullPath -LaneName 'cpu-first' -Host_ $hostSnapshot -Lines $lines
}

if ($Mode -eq 'Synthetic') {
    Write-Host 'lane_note=synthetic_substitutes_only real_file_parity_and_reality_check_cannot_be_claimed_from_these_gates'
} elseif ($Mode -eq 'RealFile') {
    Write-Host "lane_note=real_file_evidence_against_declared_shards shards=$(($shardRecords | ForEach-Object { $_.Name }) -join ',')"
} else {
    Write-Host 'lane_note=dry_run_nothing_executed'
}

if (@($failures).Count -gt 0) {
    foreach ($failure in $failures) { Write-Host "gate_failure: $failure" }

    exit 1
}

Write-Host "lane_result=ok mode=$Mode gates_ran=$(@($results).Count)"
exit 0
