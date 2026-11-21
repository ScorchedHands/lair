# Shared gates for the committed lane runner scripts.
#
# Read-only by construction: nothing in this file writes a report, copies a model shard, merges shards, downloads
# anything, or mutates Git state. Writers live in the runners and only run after their resource validation passes.
# Every host reading degrades to an explicit "not readable here" value rather than guessing, so a gate failure always
# names what was expected instead of inventing numbers the operator has not asked for.

$ErrorActionPreference = 'Stop'

function Format-LaneBytes {
    param([long]$Bytes)

    if ($Bytes -lt 1MB) { return ('{0:N2} MB ({1} B)' -f [math]::Round($Bytes / 1MB, 2), $Bytes) }

    '{0:N1} GiB' -f [math]::Round($Bytes / 1GB, 1)
}

function Expand-LaneDeclarationList {
    # Accepts either a bound array or one comma-joined argument (how a native caller reaches an array parameter), and
    # returns the individual declarations so a lane never mistakes "a,b" for one file name.
    param([string[]]$Values)

    $expanded = [System.Collections.Generic.List[string]]::new()
    foreach ($entry in @($Values)) {
        foreach ($piece in ([string]$entry -split ',')) {
            $trimmed = $piece.Trim().Trim('"').Trim("'")
            if (-not [string]::IsNullOrWhiteSpace($trimmed)) { $expanded.Add($trimmed) }
        }
    }

    @($expanded)
}

function Get-LaneHostSnapshot {
    # Free physical RAM as reported by the host. Windows reads the OS counter; Linux reads /proc/meminfo. Any other
    # posture reports null so a real-file gate fails naming its requirement instead of asserting an unreadable number.
    $freeRamMB = $null
    $ramSource = 'unavailable'

    if ($IsWindows) {
        try {
            $os = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop
            $freeRamKB = [long]$os.FreePhysicalMemory
            $freeRamMB = [int][math]::Floor($freeRamKB / 1KB)
            $ramSource = 'Win32_OperatingSystem.FreePhysicalMemory'
        } catch {
            $ramSource = "unreadable ($($_.Exception.Message))"
        }
    } elseif ($IsLinux -and (Test-Path '/proc/meminfo')) {
        try {
            foreach ($line in Get-Content '/proc/meminfo') {
                if ($line -match '^MemAvailable:\s+(\d+) kB$') {
                    $freeRamMB = [int][math]::Floor([long]$Matches[1] / 1KB)
                    $ramSource = '/proc/meminfo MemAvailable'
                    break
                }
            }

            if ($null -eq $freeRamMB) { $ramSource = 'unreadable (/proc/meminfo has no MemAvailable row)' }
        } catch {
            $ramSource = "unreadable ($($_.Exception.Message))"
        }
    } else {
        $ramSource = 'unsupported posture'
    }

    [pscustomobject]@{
        Os                = if ($IsWindows) { 'windows' } elseif ($IsLinux) { 'linux' } else { 'other' }
        FreeRamMB         = $freeRamMB
        RamSource         = $ramSource
        LogicalProcessors = [Environment]::ProcessorCount
    }
}

function Get-LaneVolumeFreeMB {
    # Free space on the volume that actually holds a path. Used for scratch/headroom checks only: no lane copies or
    # merges model shards, so this never asks for room sized like a second copy of the model.
    param([string]$Path)

    $root = [System.IO.Path]::GetFullPath($Path)

    if ($IsLinux -or $IsMacOS) {
        try {
            $dfOutput = & df -Pk "$root" 2>&1 | ForEach-Object { ([string]$_).Trim() } | Where-Object { $_ -ne '' }
            $candidates = @()

            foreach ($line in @($dfOutput)) {
                if ($line -match '^(\S+)\s+(\d+)\s+\d+\s+(\d+)\s+([0-9]+%)\s+(.+)$') {
                    $candidates += [pscustomobject]@{ Mount = $Matches[5]; AvailableBlocks = [long]$Matches[3] }
                }
            }

            foreach ($candidate in @($candidates | Sort-Object -Property @{Expression={$_.Mount.Length}} -Descending)) {
                if ($root.StartsWith($candidate.Mount, [StringComparison]::Ordinal)) {
                    return [pscustomobject]@{ Volume = $candidate.Mount; FreeMB = [int][math]::Floor(($candidate.AvailableBlocks * 1KB) / 1MB); Ready = $true }
                }
            }
        } catch {
            # Fall through to the runtime DriveInfo posture below.
        }
    }

    try {
        # DriveInfo resolves the drive that contains any absolute path, file or directory.
        $drive = New-Object System.IO.DriveInfo($root)
    } catch {
        return [pscustomobject]@{ Volume = (Split-Path -Parent $root); FreeMB = $null; Ready = $false }
    }

    # DriveInfo names its free-space property differently across runtimes, so read whichever one this host exposes.
    $freeBytes = $null
    foreach ($propertyName in @('TotalFreeBytes', 'TotalFreeSize', 'TotalFreeSpace')) {
        try {
            $candidateValue = $drive.($propertyName)
            if ($null -ne $candidateValue) {
                $freeBytes = [long]$candidateValue

                break
            }
        } catch {
            # Property absent on this runtime: try the next name.
        }
    }

    [pscustomobject]@{
        Volume = $drive.Name
        FreeMB = if ($null -eq $freeBytes) { $null } else { [int][math]::Floor($freeBytes / 1MB) }
        Ready  = [bool]$drive.IsReady
    }
}

function Write-LaneResourceLine {
    param([string]$Prefix, [pscustomobject]$Host_)

    $freeRamText = if ($null -eq $Host_.FreeRamMB) { 'unavailable' } else { "$($Host_.FreeRamMB) MB" }
    Write-Host ("{0} os={1} free_ram_mb={2} ram_source={3} logical_processors={4}" -f `
        $Prefix, $Host_.Os, $freeRamText, $Host_.RamSource, $Host_.LogicalProcessors)
}

function Format-LaneCommand {
    param([string]$Executable, [string[]]$Arguments = @())

    $parts = @(if ([string]::IsNullOrWhiteSpace($Executable)) { 'dotnet' } else { $Executable })
    foreach ($arg in @($Arguments)) {
        if ($null -eq $arg) { continue }

        # Quote anything a shell would otherwise split or interpret, so the printed line is exactly what was run.
        $text = [string]$arg
        if ($text -match '[\s;&|<>()$*!@#]') {
            $parts += "'" + $text.Replace("'", "''") + "'"
        } else {
            $parts += $text
        }
    }

    $parts -join ' '
}

function Write-LaneGatePlan {
    # Prints one planned gate exactly as the runner would invoke it, plus what that gate needs from the host.
    param(
        [string]$Name,
        [string]$Executable = 'dotnet',
        [string[]]$Arguments = @(),
        [string]$Requires = 'none',
        [string]$Status = 'planned'
    )

    Write-Host ("gate={0} status={1} requires: {2}" -f $Name, $Status, $Requires)
    if ($Status -ne 'no_committed_gate') {
        Write-Host ("  command: {0}" -f (Format-LaneCommand -Executable $Executable -Arguments $Arguments))
    }
}

function Invoke-LaneGate {
    # Runs one gate from the repository root, streams its output live, and returns a result record. Dry-run never
    # reaches this function; the runner prints the plan instead.
    param(
        [string]$Name,
        [string]$RepoRoot,
        [string]$Executable = 'dotnet',
        [string[]]$Arguments = @(),
        [string]$Requires = 'none'
    )

    Write-LaneGatePlan -Name $Name -Executable $Executable -Arguments $Arguments -Requires $Requires -Status 'running'

    Push-Location $RepoRoot
    try {
        try {
            & $Executable @($Arguments) 2>&1 | ForEach-Object { Write-Host ([string]$_) }
            $exitCode = $LASTEXITCODE
        } catch {
            # A tool that is not on PATH fails the gate naming what could not be invoked, instead of aborting the lane.
            Write-Host ("  gate_error={0} detail={1}" -f $Executable, $_.Exception.Message)
            $exitCode = 127
        }
    } finally {
        Pop-Location
    }

    if ($null -eq $exitCode) { $exitCode = 0 }

    [pscustomobject]@{ Name = $Name; ExitCode = $exitCode; Command = (Format-LaneCommand -Executable $Executable -Arguments $Arguments) }
}

function Get-GpuIdleEvidence {
    # Reads accelerator utilization and resident compute processes through nvidia-smi when the tool is present.
    # Returns evidence only: it never launches a CUDA workload, so an unconfirmed lane can refuse before any device work.
    param(
        [int]$MaxUtilPct = 5,
        [long]$MaxIdleVramMiB = 1024,
        # Model-serving process names that hold the device even while momentarily idle. Desktop composition processes
        # appear on Windows adapters too; they are recorded as context holders rather than treated as lane blockers.
        [string]$CompetingProcessPattern = 'strata|llama|lair|ollama|python'
    )

    $tool = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue
    if ($null -eq $tool) {
        return [pscustomobject]@{
            ProbeAvailable   = $false
            Tool             = 'nvidia-smi'
            Devices          = @()
            CompetingWorkloads = @()
            ContextHolders   = @()
            Idle             = $null
            Notes            = @('nvidia-smi not found on PATH: accelerator idleness cannot be evidenced from this host')
        }
    }

    $notes = [System.Collections.Generic.List[string]]::new()
    $devices = @()
    try {
        $rows = & nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total `
            --format=csv,noheader,nounits 2>&1 | ForEach-Object { ([string]$_).Trim() } | Where-Object { $_ -ne '' }

        foreach ($row in $rows) {
            $cells = @($row -split ',') | ForEach-Object { $_.Trim() }
            if ($cells.Count -lt 5) { continue }

            # The driver reports "[N/A]" for some fields on certain adapters; record those as unknown rather than failing the probe.
            $utilValue = 0L
            $usedValue = 0L
            $totalValue = 0L
            $notesSeen = @()
            if (-not [long]::TryParse($cells[2], [ref]$utilValue)) { $utilValue = -1; $notesSeen += 'utilization_unreadable' }
            if (-not [long]::TryParse($cells[3], [ref]$usedValue)) { $usedValue = -1; $notesSeen += 'memory_used_unreadable' }
            if (-not [long]::TryParse($cells[4], [ref]$totalValue)) { $totalValue = -1; $notesSeen += 'memory_total_unreadable' }

            foreach ($note in $notesSeen) { $notes.Add("device $($cells[0]): $note") }

            $devices += [pscustomobject]@{
                Index       = $cells[0]
                Name        = $cells[1]
                UtilPct     = $utilValue
                UsedMiB     = $usedValue
                TotalMiB    = $totalValue
            }
        }

        if ($devices.Count -eq 0) { $notes.Add('nvidia-smi reported no accelerator devices') }
    } catch {
        $notes.Add("device query failed: $($_.Exception.Message)")
    }

    $competing = @()
    $contextHolders = @()
    try {
        # Compute processes are the competing-workload signal: a device idle at 0% can still hold another model's context.
        $rows = & nvidia-smi --query-compute-apps=pid,used_memory,gpu_name `
            --format=csv,noheader,nounits 2>&1 | ForEach-Object { ([string]$_).Trim() } | Where-Object { $_ -ne '' }

        foreach ($row in $rows) {
            if ($row -match '^No running processes found') { continue }

            $cells = @($row -split ',') | ForEach-Object { $_.Trim() }
            if ($cells.Count -lt 2) { continue }

            $pidText = $cells[0]
            # nvidia-smi reports "[N/A]" for some processes' memory; an unreadable footprint is still a held device.
            $usedMiBValue = 0L
            $readableMemory = [long]::TryParse($cells[1], [ref]$usedMiBValue)

            $procName = "pid:$pidText"
            try {
                $parsedPid = [int]$pidText
                $named = Get-Process -Id $parsedPid -ErrorAction Stop
                if ($null -ne $named) { $procName = "$($named.ProcessName) (pid $parsedPid)" }
            } catch {
                # A process name the current user cannot read is still a competing workload; report the pid.
            }

            $record = [pscustomobject]@{
                Pid     = $pidText
                Process = $procName
                UsedMiB = $(if ($readableMemory) { $usedMiBValue } else { 'unreadable' })
                Gpu     = if ($cells.Count -ge 3) { $cells[2] } else { 'unknown' }
            }

            if (-not [string]::IsNullOrWhiteSpace($CompetingProcessPattern) -and $procName -match $CompetingProcessPattern) {
                $competing += $record
            } else {
                $contextHolders += $record
            }
        }
    } catch {
        $notes.Add("compute-process query failed: $($_.Exception.Message)")
    }

    # A device reporting busy utilization, or holding more memory than the idle ceiling, is not an idleness target even
    # when no model-named process can be identified by name.
    $busyDevices = @($devices | Where-Object { ($_.UtilPct -ge 0 -and $_.UtilPct -gt $MaxUtilPct) -or ($_.UsedMiB -ge 0 -and $_.UsedMiB -gt $MaxIdleVramMiB) })
    $idle = if (@($devices).Count -eq 0) { $false } else { (@($competing).Count -eq 0 -and @($busyDevices).Count -eq 0) }

    [pscustomobject]@{
        ProbeAvailable     = $true
        Tool               = 'nvidia-smi'
        Devices            = @($devices)
        CompetingWorkloads = @($competing)
        ContextHolders     = @($contextHolders)
        BusyDevices        = @($busyDevices)
        Idle               = $idle
        Notes              = @($notes)
    }
}

function Write-GpuIdleEvidence {
    param([pscustomobject]$Evidence, [int]$MaxUtilPct, [long]$MaxIdleVramMiB, [string]$CompetingProcessPattern = '')

    if (-not $Evidence.ProbeAvailable) {
        foreach ($note in $Evidence.Notes) { Write-Host "gpu_probe=unavailable detail=$note" }

        return
    }

    Write-Host ("gpu_probe={0} max_util_pct={1} max_idle_vram_mib={2} competing_process_pattern='{3}'" -f `
        $Evidence.Tool, $MaxUtilPct, $MaxIdleVramMiB, $CompetingProcessPattern)
    foreach ($device in $Evidence.Devices) {
        Write-Host ("  gpu index={0} name='{1}' util_pct={2} mem_used_mib={3} mem_total_mib={4}" -f `
            $device.Index, $device.Name, $(if ($device.UtilPct -lt 0) { 'unreadable' } else { $device.UtilPct }), `
            $(if ($device.UsedMiB -lt 0) { 'unreadable' } else { $device.UsedMiB }), `
            $(if ($device.TotalMiB -lt 0) { 'unreadable' } else { $device.TotalMiB }))
    }

    foreach ($note in $Evidence.Notes) { Write-Host "  gpu_probe_note=$note" }

    if (@($Evidence.CompetingWorkloads).Count -eq 0) {
        Write-Host '  gpu_competing_workload=none'
    } else {
        foreach ($workload in $Evidence.CompetingWorkloads) {
            Write-Host ("  gpu_competing_workload pid={0} process='{1}' used_mib={2} gpu='{3}'" -f `
                $workload.Pid, $workload.Process, $workload.UsedMiB, $workload.Gpu)
        }
    }

    if (@($Evidence.ContextHolders).Count -gt 0) {
        Write-Host ("  gpu_context_holders count={0} note=device_attached_processes_recorded_as_evidence; they block the lane through the utilization/memory thresholds rather than by name" -f @($Evidence.ContextHolders).Count)
    }

    if (@($Evidence.BusyDevices).Count -gt 0) {
        foreach ($device in $Evidence.BusyDevices) {
            Write-Host ("  gpu_busy_device index={0} util_pct={1} mem_used_mib={2}" -f `
                $device.Index, $(if ($device.UtilPct -lt 0) { 'unreadable' } else { $device.UtilPct }), `
                $(if ($device.UsedMiB -lt 0) { 'unreadable' } else { $device.UsedMiB }))
        }
    }
}

function Write-LaneReport {
    # Writes a plain-text lane log. Callers only reach this from an executed mode whose resource validation passed and
    # that asked for a report path explicitly; dry-runs never call it.
    param(
        [string]$Path,
        [string]$LaneName,
        [pscustomobject]$Host_,
        [object[]]$Lines = @()
    )

    $parent = Split-Path -Parent ([System.IO.Path]::GetFullPath($Path))
    if (-not [string]::IsNullOrWhiteSpace($parent) -and -not (Test-Path $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    $header = @(
        "lane=$LaneName",
        ("generated_utc={0}" -f (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')),
        'policy=no_model_download no_shard_merge_or_copy no_git_write'
    )

    $body = @($Lines | ForEach-Object { [string]$_ })
    Set-Content -LiteralPath $Path -Encoding utf8NoBOM -Value (@($header) + $body)

    Write-Host "report_written=$Path"
}
