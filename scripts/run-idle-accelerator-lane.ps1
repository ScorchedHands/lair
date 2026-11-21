<#
.SYNOPSIS
    Run the dedicated idle-accelerator lane: CUDA capability gates, CUDA kernel/prefill measurements, and two-model
    monitor measurements - only on an accelerator nobody else is using.

.DESCRIPTION
    This lane refuses before it executes anything unless you state that the device is free:

      * without -IdleGpuConfirmed it exits immediately naming the missing idle-accelerator confirmation. No probe, no
        dotnet command, no CUDA workload is launched, because a measurement taken beside another model (for example
        Strata on the dev box's only practical 4090) records that contention instead of LAIR behavior;
      * with -IdleGpuConfirmed it reads nvidia-smi utilization, device memory, and resident compute processes as
        evidence. A busy device or a process holding context on it fails naming what was observed and stops there;
      * a device whose VRAM envelope cannot hold the flagship working set is refused too: an older small accelerator may
        prove tiny capability gates with -CapabilityOnly, but its envelope would distort residency, gather, KV, and
        prefill evidence, so it never stands in for Flash-Next flagship proof.

    Nothing destructive happens in any mode: no downloads, no model copies, no Git writes. A report is written only when
    -ReportPath is supplied on an executed mode that passed every gate check; DryRun refuses a report path outright.

.PARAMETER IdleGpuConfirmed
    Required to proceed at all. It states that the accelerator is dedicated and idle for this run.

.PARAMETER Mode
    DryRun prints the planned CUDA commands after the confirmation and idleness checks pass, executing nothing. Measure
    executes them. Defaults to Measure so an unconfirmed call fails rather than previewing silently.

.PARAMETER CapabilityOnly
    Restrict the lane to tiny capability gates on a device whose VRAM cannot hold the flagship working set, with output
    stating plainly that these are not Flash-Next measurements. Without it, an insufficient envelope is refused.

.PARAMETER MinDeviceVramMiB
    Device memory required for the flagship gates (default 16384 MiB: a 4090/5090-class card). Compared against each
    device nvidia-smi reports; refusal names every observed device and its total.

.PARAMETER MaxUtilPct / MaxIdleVramMiB
    Idleness thresholds for the probe (default 5 percent utilization, 1024 MiB in use per device). A device above either
    threshold is refused even when no process on it can be named.

.PARAMETER CompetingProcessPattern
    Process-name pattern treated as an external model workload holding the device even at idle numbers (default covers
    strata, llama, lair, ollama, python). Desktop composition processes are reported as evidence but block only through
    the thresholds; pass an empty string to name-match nothing.

.PARAMETER SkipProbe
    Explicit opt-in for hosts where nvidia-smi is unavailable: proceed on the confirmation alone and record that no
    independent idleness evidence exists. Without it, an unreadable probe is a refusal.

.PARAMETER ShardPath
    Pinned model shards this lane would read for real-file CUDA reality-check work (repeat once per shard). Declared
    paths are validated before any gate runs; a missing file fails naming it instead of failing mid-measurement.

.PARAMETER ReportPath
    Plain-text lane log. Only valid with -Mode Measure; DryRun refuses it so previews leave no artifacts behind.

.EXAMPLE
    ./scripts/run-idle-accelerator-lane.ps1            # refusal: confirmation flag missing (exit 3)
.EXAMPLE
    ./scripts/run-idle-accelerator-lane.ps1 -IdleGpuConfirmed -Mode DryRun
#>
[CmdletBinding()]
param(
    [switch]$IdleGpuConfirmed,

    [ValidateSet('DryRun', 'Measure')]
    [string]$Mode = 'Measure',

    [switch]$CapabilityOnly,

    [int]$MinDeviceVramMiB = 16384,

    [int]$MaxUtilPct = 5,

    [long]$MaxIdleVramMiB = 1024,

    [string]$CompetingProcessPattern = 'strata|llama|lair|ollama|python',

    [switch]$SkipProbe,

    [string[]]$ShardPath = @(),

    [string]$ReportPath = ''
)

. (Join-Path $PSScriptRoot 'lane-gates.common.ps1')

# A lane that cannot load its shared gates stops before touching a device: every refusal here names the missing helper.
foreach ($required in @('Get-LaneHostSnapshot', 'Write-GpuIdleEvidence', 'Get-GpuIdleEvidence', 'Write-LaneGatePlan', 'Invoke-LaneGate', 'Write-LaneReport')) {
    if (-not (Test-Path -Path "Function:$required")) {
        Write-Host "lane_refusal=shared_helper_missing function=$required helper=$(Join-Path $PSScriptRoot 'lane-gates.common.ps1')"

        exit 3
    }
}

$RepoRoot = Split-Path -Parent $PSScriptRoot

# Declared shards may arrive as an array or as one comma-joined argument; both spellings name the same files.
$declaredShards = Expand-LaneDeclarationList -Values $ShardPath

function Resolve-LanePath {
    param([string]$Path)

    if ([System.IO.Path]::IsPathRooted($Path)) { return [System.IO.Path]::GetFullPath($Path) }

    [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $Path))
}

Write-Host "lane=idle-accelerator mode=$Mode repo_root=$RepoRoot"
Write-Host 'lane_policy=no_model_download no_shard_copy no_git_write'
if ($Mode -eq 'DryRun') { Write-Host 'lane_policy=dry_run_executes_nothing writes_no_report' }

$reportFullPath = ''
if (-not [string]::IsNullOrWhiteSpace($ReportPath)) {
    if ($Mode -eq 'DryRun') {
        Write-Host "lane_refusal=dry_run_writes_no_report report_path=$ReportPath"
        exit 3
    }

    $reportFullPath = Resolve-LanePath $ReportPath
}

# ---------------------------------------------------------------- confirmation gate: nothing else runs before it

if (-not $IdleGpuConfirmed) {
    Write-Host 'lane_refusal=idle_accelerator_not_confirmed'
    Write-Host '  refusal: this lane measures CUDA behavior, so it needs a dedicated idle accelerator. Pass -IdleGpuConfirmed only after the device is free: clear any external GPU workload first (on this dev box that means Strata holding the 4090), or run these gates on another 4090/5090-class card.'
    Write-Host '  refusal: no nvidia-smi probe, dotnet build, CUDA bench, or two-model monitor measurement was launched before this refusal.'
    Write-Host '  hint: preview the planned commands with -IdleGpuConfirmed -Mode DryRun once the device is genuinely idle.'
    exit 3
}

Write-Host 'lane_confirmation=idle_gpu_confirmed_by_operator'

# ---------------------------------------------------------------- declared shards are validated before any gate runs

$shardRecords = @()
foreach ($declared in @($declaredShards)) {
    $full = Resolve-LanePath $declared
    if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
        Write-Host "lane_refusal=missing_pinned_shard path=$full"
        Write-Host '  refusal: real-file CUDA reality-check work needs the declared shard present; this lane does not substitute a synthetic fixture for it.'

        exit 2
    }

    $shardRecords += [pscustomobject]@{ Path = $full; Name = [System.IO.Path]::GetFileName($full); Bytes = (Get-Item -LiteralPath $full).Length }
}

# ---------------------------------------------------------------- idleness evidence gate: still no dotnet command run

$evidence = $null
if ($SkipProbe) {
    Write-Host 'gpu_probe=skipped detail=nvidia-smi probe declined by operator; confirmation alone is the only idleness assertion and no independent evidence exists'
} else {
    $evidence = Get-GpuIdleEvidence -MaxUtilPct $MaxUtilPct -MaxIdleVramMiB $MaxIdleVramMiB -CompetingProcessPattern $CompetingProcessPattern
    Write-GpuIdleEvidence -Evidence $evidence -MaxUtilPct $MaxUtilPct -MaxIdleVramMiB $MaxIdleVramMiB -CompetingProcessPattern $CompetingProcessPattern

    if (-not $evidence.ProbeAvailable) {
        Write-Host 'lane_refusal=gpu_probe_unavailable detail=nvidia-smi not found: the lane cannot evidence that no other accelerator workload is active'
        Write-Host '  refusal: pass -SkipProbe to proceed on your confirmation alone, or run this lane where nvidia-smi can read the device.'

        exit 4
    }

    if (@($evidence.CompetingWorkloads).Count -gt 0) {
        Write-Host 'lane_refusal=competing_accelerator_workload_active'
        foreach ($workload in $evidence.CompetingWorkloads) {
            Write-Host ("  refusal: accelerator is held by pid={0} process='{1}' used_mib={2} gpu='{3}'" -f `
                $workload.Pid, $workload.Process, $workload.UsedMiB, $workload.Gpu)
        }

        Write-Host '  refusal: an idle accelerator is required for CUDA Flash-Next and two-model monitor measurements - clear the external GPU workload (on this dev box that means Strata holding the 4090) before launching any dotnet or CUDA command from this lane.'

        exit 4
    }

    $busyDevices = @($evidence.BusyDevices)
    if (@($busyDevices).Count -gt 0) {
        Write-Host 'lane_refusal=accelerator_not_idle'
        foreach ($device in $busyDevices) {
            Write-Host ("  refusal: device index={0} name='{1}' util_pct={2} (max_util_pct={3}) mem_used_mib={4} (max_idle_vram_mib={5})" -f `
                $device.Index, $device.Name, $(if ($device.UtilPct -lt 0) { 'unreadable' } else { $device.UtilPct }), `
                $MaxUtilPct, $(if ($device.UsedMiB -lt 0) { 'unreadable' } else { $device.UsedMiB }), $MaxIdleVramMiB)
        }

        Write-Host ("  refusal: wait for the device to report idle utilization and memory before this lane launches CUDA measurements (device_attached_processes_recorded={0})" -f @($evidence.ContextHolders).Count)

        exit 4
    }

    if (@($evidence.Devices).Count -eq 0) {
        Write-Host 'lane_refusal=no_accelerator_device detail=nvidia-smi reported no accelerator devices'
        exit 4
    }

    $flagshipDevices = @($evidence.Devices | Where-Object { $_.TotalMiB -ge $MinDeviceVramMiB })
    if (@($flagshipDevices).Count -eq 0) {
        Write-Host ("accelerator_envelope detail=no device reports >={0} MiB; a small old GPU can prove capability gates only, not Flash-Next flagship residency/gather/KV/prefill evidence" -f $MinDeviceVramMiB)
        foreach ($device in $evidence.Devices) {
            Write-Host ("  envelope: device index={0} name='{1}' mem_total_mib={2}" -f `
                $device.Index, $device.Name, $(if ($device.TotalMiB -lt 0) { 'unreadable' } else { $device.TotalMiB }))
        }

        if (-not $CapabilityOnly) {
            Write-Host 'lane_refusal=accelerator_envelope_insufficient detail=flagship CUDA gates need a dedicated idle 4090/5090-class accelerator'
            Write-Host '  hint: run the narrow capability gates with -CapabilityOnly, or target a dedicated idle flagship card.'

            exit 4
        }

        Write-Host 'lane_scope_note=flagship_cuda_gates_unavailable_on_this_device capability_only_lane_continues_with_tiny_capability_gates'
    }
}

# ---------------------------------------------------------------- gate table

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

Add-Gate 'build' 'dotnet' @('build', '-c', 'Release', '--nologo') `
    'idle accelerator confirmed and evidenced as free before this lane reaches any CUDA gate'

if ($CapabilityOnly) {
    Write-Host 'lane_scope=capability_only note=these_gates_answer_tiny_capability_questions_not_flash_next_flagship_residency_gather_kv_or_prefill_evidence'

    Add-Gate 'cuda-capability-gates' 'dotnet' @(
        'test', 'tests/Lair.Tests', '-c', 'Release', '--no-build', '--nologo',
        '--filter', 'Category=Cuda',
        '--logger', 'console;verbosity=normal'
    ) 'any accelerator with the driver: capability acceptance only, never a Flash-Next measurement target'

    Add-Gate 'cuda-gemv-capability' 'dotnet' @(
        'run', '--project', 'bench/Lair.Bench', '-c', 'Release', '--no-build', '--', 'cuda-gemv',
        '--rows', '256', '--cols', '512', '--batch', '8', '--reps', '3'
    ) 'device able to hold one small pinned synthetic weight pass'

    Add-Gate 'flash-next-cuda-flagship-gates' 'dotnet' @() `
        'refused on a capability-only lane: flagship CUDA evidence needs a dedicated idle 4090/5090-class accelerator and the pinned shards' `
        'no_committed_gate'
} else {
    Add-Gate 'cuda-capability-gates' 'dotnet' @(
        'test', 'tests/Lair.Tests', '-c', 'Release', '--no-build', '--nologo',
        '--filter', 'Category=Cuda',
        '--logger', 'console;verbosity=normal'
    ) 'dedicated idle 4090-class accelerator with the driver, no other compute process on it'

    Add-Gate 'cuda-kernel-measurements' 'dotnet' @(
        'run', '--project', 'bench/Lair.Bench', '-c', 'Release', '--no-build', '--', 'cuda-gemv'
    ) 'device holding the pinned synthetic weight fixtures with no competing context: per-format CUDA-vs-CPU GEMV/GEMM evidence'

    Add-Gate 'cuda-prefill-measurement' 'dotnet' @(
        'run', '--project', 'bench/Lair.Bench', '-c', 'Release', '--no-build', '--', 'cuda-prefill'
    ) 'device holding the pinned dense fixture plus its KV/pool working set: composed prefill evidence'

    Add-Gate 'two-model-monitor-measurements' 'dotnet' @(
        'run', '--project', 'bench/Lair.Bench', '-c', 'Release', '--no-build', '--', 'cuda-concurrency'
    ) 'device able to hold two independent weight passes at once: overlapped-stream payoff plus serialized-baseline comparison on one primary context'

    Add-Gate 'two-model-lane-tests' 'dotnet' @(
        'test', 'tests/Lair.Tests', '-c', 'Release', '--no-build', '--nologo',
        '--filter', 'Category=Streams|Concurrency',
        '--logger', 'console;verbosity=normal'
    ) 'idle accelerator: per-model stream lanes and co-scheduling gates run against the device'

    Add-Gate 'flash-next-cuda-reality-check' 'dotnet' @() `
        $(if (@($shardRecords).Count -gt 0) { "declared pinned shards present ($(($shardRecords | ForEach-Object { $_.Name }) -join ', ')) but no committed CUDA bench seam reads them yet: reported unrunnable rather than approximated on synthetic fixtures" } else { 'needs both an idle flagship accelerator and explicit -ShardPath entries for the pinned files; reported unrunnable rather than substituted with a smaller fixture' }) `
        'no_committed_gate'

    Add-Gate 'two-model-monitor-serve' 'dotnet' @() `
        'manual gate: two GPU models served live on one idle device with monitor evidence - scripted measurement lanes cannot claim it' `
        'no_committed_gate'
}

$executing = ($Mode -eq 'Measure')

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

    if ($results | Where-Object { $_.Name -eq 'build' -and $_.ExitCode -ne 0 }) {
        Write-Host ("gate={0} status=skipped reason=build_gate_failed" -f $gate.Name)

        continue
    }

    # Re-check idleness immediately before the first device-touching gate: a workload that started between the probe and
    # this moment must not slip into a measurement.
    if ($gate.Name -ne 'build' -and -not $SkipProbe) {
        $recheck = Get-GpuIdleEvidence -MaxUtilPct $MaxUtilPct -MaxIdleVramMiB $MaxIdleVramMiB -CompetingProcessPattern $CompetingProcessPattern
        if ($recheck.ProbeAvailable -and (@($recheck.CompetingWorkloads).Count -gt 0 -or @($recheck.BusyDevices).Count -gt 0)) {
            Write-Host 'lane_refusal=competing_accelerator_workload_active reason=device_stopped_being_idle_after_first_probe'
            foreach ($workload in $recheck.CompetingWorkloads) {
                Write-Host ("  refusal: accelerator is held by pid={0} process='{1}' used_mib={2} gpu='{3}'" -f `
                    $workload.Pid, $workload.Process, $workload.UsedMiB, $workload.Gpu)
            }

            foreach ($device in $recheck.BusyDevices) {
                Write-Host ("  refusal: device index={0} name='{1}' util_pct={2} mem_used_mib={3}" -f `
                    $device.Index, $device.Name, $(if ($device.UtilPct -lt 0) { 'unreadable' } else { $device.UtilPct }), `
                    $(if ($device.UsedMiB -lt 0) { 'unreadable' } else { $device.UsedMiB }))
            }

            $failures.Add('accelerator stopped being idle before the CUDA gates ran; remaining device measurements were not launched')

            break
        }
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
    foreach ($shard in $shardRecords) { $lines += ("shard name={0} path={1} bytes={2}" -f $shard.Name, $shard.Path, $shard.Bytes) }

    if ($null -ne $evidence) {
        foreach ($device in $evidence.Devices) {
            $lines += ("gpu index={0} name='{1}' util_pct={2} mem_used_mib={3} mem_total_mib={4}" -f `
                $device.Index, $device.Name, $device.UtilPct, $device.UsedMiB, $device.TotalMiB)
        }

        foreach ($workload in $evidence.CompetingWorkloads) {
            $lines += ("gpu_competing_workload pid={0} process='{1}' used_mib={2}" -f $workload.Pid, $workload.Process, $workload.UsedMiB)
        }
    } else {
        $lines += 'gpu_probe=skipped_by_operator'
    }

    foreach ($result in $results) {
        $lines += ("gate={0} exit_code={1} command={2}" -f $result.Name, $result.ExitCode, $result.Command)
    }

    if (@($failures).Count -gt 0) { $lines += @($failures | ForEach-Object { "failure=$_" }) }

    Write-LaneReport -Path $reportFullPath -LaneName 'idle-accelerator' -Host_ (Get-LaneHostSnapshot) -Lines $lines
}

if ($CapabilityOnly) {
    Write-Host 'lane_note=capability_only_evidence flagship_flash_next_claims_not_made_from_this_lane'
} elseif (@($shardRecords).Count -gt 0) {
    Write-Host "lane_note=idle_accelerator_measurements declared_shards=$(($shardRecords | ForEach-Object { $_.Name }) -join ',')"
} else {
    Write-Host 'lane_note=idle_accelerator_measurements_on_pinned_synthetic_cuda_fixtures'
}

if (@($failures).Count -gt 0) {
    foreach ($failure in $failures) { Write-Host "gate_failure: $failure" }

    exit 1
}

Write-Host "lane_result=ok mode=$Mode gates_ran=$(@($results).Count)"
exit 0
