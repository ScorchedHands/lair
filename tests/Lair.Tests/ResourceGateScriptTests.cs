using System.Diagnostics;
using System.Text.RegularExpressions;

namespace Lair.Tests;

/// <summary>
/// Resource-gate gates for the committed lane runner scripts: the CPU-first lane and the idle-accelerator lane, plus the
/// shared helper they dot-source. Every gate here runs a script through PowerShell in its refusal or preview posture only -
/// no CUDA device is required, no model file is opened, no Xeon host is needed, nothing heavy executes, and no report is
/// written. What each gate asserts is wording: the CPU lane must name its resource requirements and refuse to pass off a
/// synthetic container as real-file evidence, and the accelerator lane must stop before it launches anything until an idle
/// device is confirmed. The whole category skips cleanly when PowerShell 7 is not installed on the host (the scripts are
/// PowerShell artifacts, so pwsh is their runtime), following the repository's skip-if-absent convention.
/// </summary>
[Trait("Category", "Scripts")]
public sealed class ResourceGateScriptTests : IDisposable
{
    private const string CpuLaneScript = "scripts/run-cpu-first-lane.ps1";
    private const string IdleLaneScript = "scripts/run-idle-accelerator-lane.ps1";
    private const string SharedHelperScript = "scripts/lane-gates.common.ps1";

    // Operations these lanes must never perform: no Git state changes, no file copies or deletions of model data, and no downloads.
    private static readonly string[] ForbiddenOperations =
    [
        @"\bgit\s+(add|commit|push|reset|stash|merge|checkout|rebase)\b",
        @"\bCopy-Item\b",
        @"\bMove-Item\b",
        @"\bRemove-Item\b",
        @"\bInvoke-WebRequest\b",
        @"\bInvoke-RestMethod\b",
        @"\bStart-BitsTransfer\b",
        @"\bcurl\b",
        @"\bwget\b",
    ];

    // Paths a lane must never write to: committed fixtures and committed bench reports are not runner output.
    private static readonly string[] ForbiddenWriteTargets =
    [
        @"bench/reports",
        @"bench\reports",
        @"tests/fixtures",
        @"tests\fixtures",
    ];

    private static readonly string? PwshPath = ResolvePwsh();

    private readonly List<string> _tempPaths = [];

    public void Dispose()
    {
        foreach (string path in _tempPaths)
        {
            try
            {
                if (File.Exists(path))
                {
                    File.Delete(path);
                }
            } catch (IOException)
            {
                // Best-effort cleanup of scratch stand-in paths.
            }
        }
    }

    // ---------------------------------------------------------------- existence + parseability through pwsh itself

    [Theory]
    [InlineData(SharedHelperScript)]
    [InlineData(CpuLaneScript)]
    [InlineData(IdleLaneScript)]
    public void LaneScripts_AreCommitted_AndParseThroughPwsh(string relativePath)
    {
        string script = RequireRepoRelative(relativePath);

        Assert.True(File.Exists(script), $"expected committed lane script at {script}");
        Assert.NotEmpty(File.ReadAllText(script));

        if (PwshPath is null)
        {
            return; // pwsh absent on this host: the scripts are PowerShell artifacts and cannot be exercised here.
        }

        string[] scriptArguments = relativePath == CpuLaneScript ? ["-Mode", "DryRun"] : [];

        // Running the artifact is the parse check: a syntax error would fail before any gate logic could print its marker.
        (int exit, string output) = RunScript(relativePath, scriptArguments);

        Assert.True(File.Exists(script), $"expected committed lane script at {script}");
        Assert.NotEmpty(File.ReadAllText(script));

        switch (relativePath)
        {
            case SharedHelperScript:
                // The helper defines gates only: executing it standalone must succeed and launch nothing.
                Assert.Equal(0, exit);
                Assert.DoesNotContain("dotnet", output, StringComparison.Ordinal);
                break;
            case CpuLaneScript:
                Assert.Contains("lane=cpu-first mode=DryRun", output, StringComparison.Ordinal);
                Assert.Equal(0, exit);
                break;
            default:
                // The accelerator lane defaults to executing gates, so an unconfirmed call must refuse with its own code.
                Assert.Equal(3, exit);
                Assert.Contains("lane_refusal=idle_accelerator_not_confirmed", output, StringComparison.Ordinal);
                break;
        }
    }

    [Theory]
    [InlineData(CpuLaneScript)]
    [InlineData(IdleLaneScript)]
    [InlineData(SharedHelperScript)]
    public void LaneScripts_ContainNoDestructiveOrProtectedPathOperations(string relativePath)
    {
        string script = RequireRepoRelative(relativePath);

        if (PwshPath is null)
        {
            return;
        }

        string source = File.ReadAllText(script);

        foreach (string forbidden in ForbiddenOperations)
        {
            Assert.False(Regex.IsMatch(source, forbidden), $"{relativePath} contains a forbidden operation matching {forbidden}");
        }

        foreach (string target in ForbiddenWriteTargets)
        {
            Assert.False(source.Contains(target, StringComparison.OrdinalIgnoreCase), $"{relativePath} references protected report/fixture path {target}");
        }

        // Each runner states its own no-download / no-shard-copy / no-Git-write policy out loud.
        if (relativePath != SharedHelperScript)
        {
            Assert.Contains("no_model_download", source, StringComparison.Ordinal);
            Assert.Contains("no_git_write", source, StringComparison.Ordinal);
        }
    }

    // ---------------------------------------------------------------- CPU-first lane preview and resource wording

    [Fact]
    public void CpuFirstLane_DryRun_NamesResourceRequirements_AndNeedsNoCudaDevice()
    {
        (int exit, string output) = RunLaneScript(["-Mode", "DryRun"]);

        if (PwshPath is null)
        {
            return;
        }

        Assert.Equal(0, exit);
        Assert.Contains("lane=cpu-first mode=DryRun", output, StringComparison.Ordinal);
        Assert.Contains("dry_run_executes_nothing writes_no_report", output, StringComparison.Ordinal);

        // CPU-only gates: no CUDA device and no GPU idleness confirmation is demanded anywhere in the preview.
        Assert.Contains("cpu_only_gates_require_no_cuda_device", output, StringComparison.Ordinal);
        Assert.DoesNotContain("-IdleGpuConfirmed", output, StringComparison.Ordinal);

        // Named resource expectations for the real-file posture of the same lane.
        Assert.Contains("headroom_mb=", output, StringComparison.Ordinal);
        Assert.Contains("min_free_disk_mb=", output, StringComparison.Ordinal);
        Assert.Contains("pinned_shards_declared_explicitly", output, StringComparison.Ordinal);

        // Targeted test filters and bench commands are printed exactly as the executed lane would run them.
        Assert.Contains("--filter Category=Moe", output, StringComparison.Ordinal);
        Assert.Contains("Category=Loader&FullyQualifiedName!~Qwen4Exp", output, StringComparison.Ordinal);
        Assert.Contains("bench/Lair.Bench", output, StringComparison.Ordinal);

        // Synthetic gates are labeled as such, and suites that do not exist yet are reported unrunnable rather than faked.
        Assert.Contains("synthetic_loader_and_gather_gates_run_without_the_target_file", output, StringComparison.Ordinal);
        Assert.Contains("real_file_parity_and_reality_check_need_declared_shards_never_synthetic_substitutes", output, StringComparison.Ordinal);
        Assert.Contains("gate=ple-gates-synthetic status=no_committed_gate", output, StringComparison.Ordinal);
        Assert.Contains("gate=mtp-gates-synthetic status=no_committed_gate", output, StringComparison.Ordinal);

        // A dry-run launches nothing.
        Assert.DoesNotContain("status=running", output, StringComparison.Ordinal);
    }

    [Fact]
    public void CpuFirstLane_DryRun_RefusesAReportPath_AndWritesNoArtifact()
    {
        string reportPath = TempScratchPath("cpu-lane-dryrun-report.txt");

        (int exit, string output) = RunLaneScript(["-Mode", "DryRun", "-ReportPath", reportPath]);

        if (PwshPath is null)
        {
            return;
        }

        Assert.Equal(3, exit);
        Assert.Contains("lane_refusal=dry_run_writes_no_report", output, StringComparison.Ordinal);
        Assert.False(File.Exists(reportPath), "a dry-run must not write a lane report");
    }

    [Fact]
    public void CpuFirstLane_RealFile_WithoutDeclaredShards_FailsNamingPinnedPathsAndHeadroom()
    {
        (int exit, string output) = RunLaneScript(["-Mode", "RealFile"]);

        if (PwshPath is null)
        {
            return;
        }

        Assert.Equal(2, exit);
        Assert.Contains("lane_refusal=real_file_resources_not_satisfied", output, StringComparison.Ordinal);
        Assert.Contains("fallback=synthetic_fixtures_forbidden", output, StringComparison.Ordinal);
        Assert.Contains("needs explicit -ShardPath", output, StringComparison.Ordinal);

        // The refusal names the pinned shard location and every headroom requirement instead of guessing a smaller run.
        Assert.Contains("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf", output, StringComparison.Ordinal);
        Assert.Contains("requirement: pinned shards present", output, StringComparison.Ordinal);
        Assert.Contains("headroom_mb=", output, StringComparison.Ordinal);
        Assert.Contains("free_ram_mb>=", output, StringComparison.Ordinal);
        Assert.Contains("disk_free_mb>=", output, StringComparison.Ordinal);

        // Nothing executed on the way to that refusal.
        Assert.DoesNotContain("status=running", output, StringComparison.Ordinal);
    }

    [Fact]
    public void CpuFirstLane_RealFile_MissingDeclaredShard_FailsNamingPath_NotSyntheticSubstitute()
    {
        string missing = TempScratchPath("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf");

        (int exit, string output) = RunLaneScript(["-Mode", "RealFile", "-ShardPath", missing]);

        if (PwshPath is null)
        {
            return;
        }

        Assert.Equal(2, exit);
        Assert.Contains("missing pinned shard:", output, StringComparison.Ordinal);
        Assert.Contains(missing, output, StringComparison.Ordinal);
        Assert.Contains("fallback=synthetic_fixtures_forbidden", output, StringComparison.Ordinal);

        // A declared path that is absent produces no real-file gate at all: the lane refuses instead of measuring a stand-in.
        Assert.DoesNotContain("reality-check", output, StringComparison.Ordinal);
        Assert.DoesNotContain("status=running", output, StringComparison.Ordinal);
    }

    [Fact]
    public void CpuFirstLane_RealFile_InsufficientHeadroom_FailsNamingWorkingSetAndRamNumbers()
    {
        string standIn = TempScratchPath("lane-stand-in-shard.gguf");
        File.WriteAllText(standIn, new string('a', 4096)); // a present file so the failure under test is headroom, not absence

        // A declared working set no realistic host can hold: the refusal must name its own numbers.
        (int exit, string output) = RunLaneScript(["-Mode", "RealFile", "-ShardPath", standIn, "-WorkingSetMB", "1", "-HeadroomMB", "2097152"]);

        if (PwshPath is null)
        {
            return;
        }

        Assert.Equal(2, exit);
        Assert.Contains("insufficient RAM headroom", output, StringComparison.Ordinal);
        Assert.Contains("working_set_mb=1 working_set_source=operator-declared", output, StringComparison.Ordinal);
        Assert.Contains("required_free_ram_mb=", output, StringComparison.Ordinal);
        Assert.Contains("refuses instead of running a smaller fixture", output, StringComparison.Ordinal);

        (int diskExit, string diskOutput) = RunLaneScript(["-Mode", "RealFile", "-ShardPath", standIn, "-MinFreeDiskMB", "999999999"]);
        Assert.Equal(2, diskExit);
        Assert.Contains("insufficient disk headroom", diskOutput, StringComparison.Ordinal);
        Assert.Contains("never copies or merges shards", diskOutput, StringComparison.Ordinal);

        // With a posture any host satisfies, the declared-shard preview names the exact real-file commands against those paths.
        (int previewExit, string previewOutput) = RunLaneScript(["-Mode", "DryRun", "-ShardPath", standIn, "-WorkingSetMB", "1", "-HeadroomMB", "0", "-MinFreeDiskMB", "1"]);
        Assert.Equal(0, previewExit);
        Assert.Contains("reality-check-gemv-file[", previewOutput, StringComparison.Ordinal);
        Assert.Contains(Path.GetFileName(standIn), previewOutput, StringComparison.Ordinal);
        Assert.Contains(standIn, previewOutput, StringComparison.Ordinal); // the command names the declared file itself
        Assert.Contains("gguf-dump", previewOutput, StringComparison.Ordinal);
    }

    // ---------------------------------------------------------------- idle-accelerator lane refusal

    [Fact]
    public void IdleAcceleratorLane_WithoutConfirmation_RefusesBeforeLaunchingAnyCommand()
    {
        (int exit, string output) = RunScript(IdleLaneScript, []);

        if (PwshPath is null)
        {
            return;
        }

        Assert.Equal(3, exit);
        Assert.Contains("lane_refusal=idle_accelerator_not_confirmed", output, StringComparison.Ordinal);
        Assert.Contains("needs a dedicated idle accelerator", output, StringComparison.Ordinal);
        Assert.Contains("clear any external GPU workload", output, StringComparison.Ordinal);
        Assert.Contains("Strata", output, StringComparison.Ordinal);

        // The refusal happens before the gate table exists: no build, bench, or CUDA command was launched.
        Assert.DoesNotContain("lane_gates=", output, StringComparison.Ordinal);
        Assert.DoesNotContain("status=queued", output, StringComparison.Ordinal);
        Assert.DoesNotContain("status=running", output, StringComparison.Ordinal);
    }

    [Fact]
    public void IdleAcceleratorLane_MissingDeclaredShard_FailsBeforeAnyGateRuns()
    {
        string missing = TempScratchPath("idle-lane-missing-shard.gguf");

        (int exit, string output) = RunScript(IdleLaneScript, ["-IdleGpuConfirmed", "-Mode", "DryRun", "-ShardPath", missing]);

        if (PwshPath is null)
        {
            return;
        }

        Assert.Equal(2, exit);
        Assert.Contains("lane_refusal=missing_pinned_shard", output, StringComparison.Ordinal);
        Assert.Contains(missing, output, StringComparison.Ordinal);
        Assert.Contains("does not substitute a synthetic fixture", output, StringComparison.Ordinal);
        Assert.DoesNotContain("status=running", output, StringComparison.Ordinal);
    }

    [Fact]
    public void IdleAcceleratorLane_WithConfirmation_EvidencesDeviceIdleness_BeforeAnyGateRuns()
    {
        (int exit, string output) = RunScript(IdleLaneScript, ["-IdleGpuConfirmed", "-Mode", "DryRun"]);

        if (PwshPath is null)
        {
            return;
        }

        Assert.Contains("lane_confirmation=idle_gpu_confirmed_by_operator", output, StringComparison.Ordinal);

        bool probeReadable = output.Contains("gpu_probe=nvidia-smi", StringComparison.Ordinal);
        if (!probeReadable)
        {
            // A host that cannot read the device must say so and refuse rather than assume idleness.
            Assert.Contains("lane_refusal=gpu_probe_unavailable", output, StringComparison.Ordinal);
            Assert.Equal(4, exit);

            return;
        }

        Assert.Contains("gpu index=", output, StringComparison.Ordinal);

        if (exit == 0)
        {
            // Genuinely idle device: the preview names CUDA gates without launching them.
            Assert.Contains("lane_gates=", output, StringComparison.Ordinal);
            Assert.DoesNotContain("status=running", output, StringComparison.Ordinal);

            return;
        }

        // Occupied device (the dev box case while Strata runs): refusal names what holds it and launches nothing.
        Assert.Equal(4, exit);
        Assert.True(
            output.Contains("lane_refusal=competing_accelerator_workload_active", StringComparison.Ordinal) ||
            output.Contains("lane_refusal=accelerator_not_idle", StringComparison.Ordinal),
            $"expected an idleness refusal, got: {output}");

        if (output.Contains("gpu_competing_workload pid=", StringComparison.Ordinal))
        {
            Assert.Contains("clear the external GPU workload", output, StringComparison.Ordinal);
        } else
        {
            Assert.Contains("wait for the device to report idle utilization and memory", output, StringComparison.Ordinal);
        }

        Assert.DoesNotContain("status=running", output, StringComparison.Ordinal);
    }

    // ---------------------------------------------------------------- helpers

    private static string RequireRepoRelative(string relativePath)
    {
        string? path = ModelFixture.ResolveRepoRelative(relativePath.Replace('/', Path.DirectorySeparatorChar));

        Assert.NotNull(path);

        return path!;
    }

    private string TempScratchPath(string fileName)
    {
        string directory = Path.Combine(Path.GetTempPath(), "lair-lane-gate-tests");
        Directory.CreateDirectory(directory);

        string path = Path.Combine(directory, $"{Guid.NewGuid():N}-{fileName}");
        _tempPaths.Add(path);

        return path;
    }

    private static (int ExitCode, string Output) RunLaneScript(string[] laneArguments) => RunScript(CpuLaneScript, laneArguments);

    private static (int ExitCode, string Output) RunScript(string relativeScript, string[] arguments)
    {
        if (PwshPath is null)
        {
            return (0, string.Empty); // pwsh absent: the gate that called this returns before asserting.
        }

        string script = RequireRepoRelative(relativeScript);

        var startInfo = new ProcessStartInfo
        {
            FileName = PwshPath,
            WorkingDirectory = Directory.GetCurrentDirectory(),
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };

        foreach (string argument in new[] { "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", script })
        {
            startInfo.ArgumentList.Add(argument);
        }

        foreach (string argument in arguments)
        {
            if (!string.IsNullOrEmpty(argument))
            {
                startInfo.ArgumentList.Add(argument);
            }
        }

        using Process process = Process.Start(startInfo) ?? throw new InvalidOperationException($"could not launch {PwshPath}");

        var lines = new List<string>();
        object lineLock = new();

        void Capture(object? sender, DataReceivedEventArgs args)
        {
            if (args.Data is null)
            {
                return;
            }

            lock (lineLock)
            {
                lines.Add(args.Data);
            }
        }

        process.OutputDataReceived += Capture;
        process.ErrorDataReceived += Capture;
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        if (!process.WaitForExit(120_000))
        {
            try
            {
                process.Kill(entireProcessTree: true);
            } catch (InvalidOperationException)
            {
                // Already gone.
            }

            throw new InvalidOperationException($"lane script {relativeScript} did not finish within 120s");
        }

        process.WaitForExit(); // flushes the asynchronous output readers

        lock (lineLock)
        {
            return (process.ExitCode, string.Join("\n", lines));
        }
    }

    private static string? ResolvePwsh()
    {
        foreach (string pathEntry in (Environment.GetEnvironmentVariable("PATH") ?? string.Empty).Split(Path.PathSeparator))
        {
            if (string.IsNullOrWhiteSpace(pathEntry))
            {
                continue;
            }

            try
            {
                string candidate = Path.Combine(pathEntry.Trim(), OperatingSystem.IsWindows() ? "pwsh.exe" : "pwsh");
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            } catch (ArgumentException)
            {
                // Malformed PATH entry: keep looking.
            }
        }

        string[] fallbacks = OperatingSystem.IsWindows()
            ? [@"C:\Program Files\PowerShell\7\pwsh.exe", Path.Combine(Path.GetTempPath(), "pwsh.exe")]
            : ["/usr/local/bin/pwsh", "/usr/bin/pwsh"];

        foreach (string candidate in fallbacks)
        {
            try
            {
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            } catch (ArgumentException)
            {
                // Unusable path shape: keep looking.
            }
        }

        return null;
    }
}
