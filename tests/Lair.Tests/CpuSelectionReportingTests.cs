using System.Diagnostics;
using System.Runtime.Intrinsics.X86;
using System.Text.Json;

using Lair.Backends.Cpu;
using Lair.Host;
using Lair.Host.Config;
using Lair.Http;
using Lair.Http.Monitoring;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// Selection reporting for the resolved <c>(isa, gather-mode)</c> pair an opened CPU backend holds. The pair is reporting only:
/// <see cref="CpuBackend.ResolvedSelection"/> states the ISA axis as declared (<c>auto</c>, <c>avx2</c>, <c>avx-vnni</c>), the
/// integer-dot encoding it resolved to (<c>scalar</c>, <c>avx2</c>, <c>vnni</c>), and the grid staging that encoding feeds
/// (<c>scalar</c>, <c>gather</c>), and nothing here is a knob — a stated posture is the same programmatic seam
/// <c>engine.cpu.isa</c> refusals are proven through. At the serve seam the pair rides the load log in the style of
/// <c>prefill_kernel</c>, the monitor facts, the snapshot JSON, and the CLI model line, so a forced <c>avx2</c> run and an
/// <c>auto</c> run of one model diff on the ISA field alone. Gather values are asserted only as the vocabulary they may take:
/// which encoding a host's own probe measures is host posture, never something these gates pin.
/// </summary>
[Trait("Category", "Config")]
[Trait("Category", "Metrics")]
public sealed class CpuSelectionReportingTests : IDisposable
{
    private static readonly string[] AllowedResolvedIsas = ["scalar", "avx2", "vnni"];
    private static readonly string[] AllowedGatherModes = ["scalar", "gather"];

    private readonly List<string> _tempFiles = [];

    public void Dispose()
    {
        foreach (string path in _tempFiles)
        {
            try
            {
                File.Delete(path);
            }
            catch (IOException)
            {
                // A model still holding its mmap view keeps the file on Windows; the temp directory absorbs it.
            }
        }

        _tempFiles.Clear();
    }

    // ---------------------------------------------------------------- the reporting record on an opened backend

    [Theory]
    [InlineData("auto", true, false, "auto", "avx2")]
    [InlineData("avx2", true, false, "avx2", "avx2")]
    [InlineData("auto", false, false, "auto", "scalar")]
    public void BackendOpen_StatedPosture_ReportsDeclaredAndResolvedIsaInReportingNames(string declaredIsa, bool avx2Available, bool vnniAvailable, string expectedRequested, string expectedResolved)
    {
        using CpuBackend backend = new(threads: 1, requestedIsa: Parse(declaredIsa), cpuFeatures: new CpuFeaturePosture(avx2Available, vnniAvailable));

        CpuSelectionReport report = backend.ResolvedSelection;

        Assert.Equal(expectedRequested, report.RequestedIsa); // the axis exactly as declared, in the vocabulary a host may state
        Assert.Equal(expectedResolved, report.ResolvedIsa); // beside the encoding that declaration resolved to against this posture
        Assert.Contains(report.GatherMode, AllowedGatherModes); // and the staging that feeds it, stated in its own vocabulary
    }

    [Theory]
    [InlineData("auto", "auto", "vnni")]
    [InlineData("avx2", "avx2", "avx2")]
    public void BackendOpen_VnniCapablePosture_ReportsVnniForAutoAndAvx2ForTheForcedAxis(string declaredIsa, string expectedRequested, string expectedResolved)
    {
        if (!AvxVnni.IsSupported)
        {
            return; // a posture stating vpdpbusd is only assertable on a host that reports it: the pair is a fact of that posture.
        }

        using CpuBackend backend = new(threads: 1, requestedIsa: Parse(declaredIsa), cpuFeatures: new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: true));

        Assert.Equal(expectedRequested, backend.ResolvedSelection.RequestedIsa);
        Assert.Equal(expectedResolved, backend.ResolvedSelection.ResolvedIsa);
        Assert.Contains(backend.ResolvedSelection.GatherMode, AllowedGatherModes);
    }

    [Fact]
    public void BackendOpen_NoVectorPosture_ReportsScalarPair_WithoutInventingGather()
    {
        // The scalar-oracle posture (the test knob, never a config axis) resolves scalar dots and scalar staging: a report never
        // names a gather encoding this backend cannot issue.
        using CpuBackend backend = new(threads: 1, allowAvx2: false);

        Assert.Equal("auto", backend.ResolvedSelection.RequestedIsa);
        Assert.Equal("scalar", backend.ResolvedSelection.ResolvedIsa);
        Assert.Equal("scalar", backend.ResolvedSelection.GatherMode);
    }

    [Fact]
    public void BackendOpen_ScalarPosture_ReportsScalarPair_WithoutInventingGather()
    {
        using CpuBackend backend = new(threads: 1, cpuFeatures: new CpuFeaturePosture(Avx2Available: false, AvxVnniAvailable: false));

        Assert.Equal("scalar", backend.ResolvedSelection.ResolvedIsa);
        Assert.Equal("scalar", backend.ResolvedSelection.GatherMode);
    }

    // ---------------------------------------------------------------- the pair at the serve seam

    [Fact]
    public void Serve_AutoAgainstVnniCapablePosture_LogsThePairAtLoad_AndReportsItOnMonitorFacts()
    {
        if (!AvxVnni.IsSupported)
        {
            return; // the auto-over-VNNI posture is only assertable where the host reports vpdpbusd.
        }

        CapturingLogger log = new();
        using GenerationService engine = OpenEngine("cpu-sel-auto", isa: null, cpuFeatures: VnniCapablePosture(), log);

        Assert.Contains(
            log.Messages(LogLevel.Information),
            line => line.StartsWith($"cpu_selection model='cpu-sel-auto' isa=vnni gather=", StringComparison.Ordinal) && AllowedGatherModes.Contains(TailValue(line, "gather=")));

        MetricsSnapshot snapshot = NewestSnapshotAfterLoad(engine);

        Assert.Equal("vnni", snapshot.CpuIsa);
        Assert.Contains(snapshot.CpuGather, AllowedGatherModes);

        engine.Dispose();
    }

    [Fact]
    public void Serve_ForceAvx2AgainstVnniCapablePosture_ReportsAvx2Isa_AndDiffersFromAutoOnTheIsaFieldAlone()
    {
        if (!AvxVnni.IsSupported)
        {
            return; // the forced-vs-auto comparison is only meaningful on a posture that reports vpdpbusd.
        }

        CpuFeaturePosture posture = VnniCapablePosture();

        CapturingLogger autoLog = new();
        using GenerationService auto = OpenEngine("cpu-sel-diff-auto", isa: null, cpuFeatures: posture, autoLog);

        CapturingLogger forcedLog = new();
        using GenerationService forced = OpenEngine("cpu-sel-diff-avx2", isa: "avx2", cpuFeatures: posture, forcedLog);

        Assert.Contains(forcedLog.Messages(LogLevel.Information), line => line.Contains("cpu_selection model='cpu-sel-diff-avx2' isa=avx2 gather=", StringComparison.Ordinal));

        MetricsSnapshot autoSnapshot = NewestSnapshotAfterLoad(auto);
        MetricsSnapshot forcedSnapshot = NewestSnapshotAfterLoad(forced);

        Assert.Equal("vnni", autoSnapshot.CpuIsa); // the same posture, resolved two ways: the ISA field is where the two runs diff.
        Assert.Equal("avx2", forcedSnapshot.CpuIsa);
        Assert.NotEqual(autoSnapshot.CpuIsa, forcedSnapshot.CpuIsa);
        Assert.Equal(autoSnapshot.CpuGather, forcedSnapshot.CpuGather); // one host posture resolves one grid encoding, whichever dot axis was asked for.

        auto.Dispose();
        forced.Dispose();
    }

    [Fact]
    public void Serve_AutoAgainstScalarPosture_ReportsScalarIsaAndScalarGatherAtLoadAndOnMonitor()
    {
        CapturingLogger log = new();
        using GenerationService engine = OpenEngine(
            "cpu-sel-scalar",
            isa: null,
            cpuFeatures: new CpuFeaturePosture(Avx2Available: false, AvxVnniAvailable: false),
            log);

        Assert.Contains(
            log.Messages(LogLevel.Information),
            line => line == $"cpu_selection model='cpu-sel-scalar' isa=scalar gather=scalar");

        MetricsSnapshot snapshot = NewestSnapshotAfterLoad(engine);

        Assert.Equal("scalar", snapshot.CpuIsa);
        Assert.Equal("scalar", snapshot.CpuGather); // a posture with no vector path never reports a gather encoding it cannot issue.

        engine.Dispose();
    }

    [Fact]
    public void SnapshotJson_CarriesCpuSelectionFields_AndKeepsEveryFieldAnExistingConsumerReads()
    {
        using GenerationService engine = OpenEngine("cpu-sel-json", isa: "avx2", cpuFeatures: new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: false), new CapturingLogger());
        MetricsSnapshot snapshot = NewestSnapshotAfterLoad(engine);

        using JsonDocument document = JsonDocument.Parse(snapshot.ToJson());
        JsonElement root = document.RootElement;

        Assert.Equal(JsonValueKind.String, root.GetProperty("metrics_cpu_isa").ValueKind);
        Assert.Contains(root.GetProperty("metrics_cpu_isa").GetString(), AllowedResolvedIsas);
        Assert.Equal(JsonValueKind.String, root.GetProperty("metrics_cpu_gather").ValueKind);
        Assert.Contains(root.GetProperty("metrics_cpu_gather").GetString(), AllowedGatherModes);

        // Every field an existing consumer already reads is still there with the same meaning.
        Assert.Equal("cpu-sel-json", root.GetProperty("metrics_model").GetString());
        Assert.True(root.TryGetProperty("metrics_worker_threads", out _));
        Assert.True(root.TryGetProperty("metrics_prefill_kernel", out _));
        Assert.True(root.TryGetProperty("metrics_kv_residency", out _));
        Assert.True(root.TryGetProperty("metrics_compute_requested_threads", out _));
        Assert.True(root.TryGetProperty("metrics_persistent_vram_bytes", out JsonElement persistentVram));
        Assert.Equal(JsonValueKind.Null, persistentVram.ValueKind); // a CPU-only serve still carries nothing in the accelerator slots.

        engine.Dispose();
    }

    // ---------------------------------------------------------------- the CLI model line

    [Fact]
    public async Task MonitorCommand_RendersCpuSelectionTail_AndCpuOnlyColumnsStayIntact()
    {
        string path = SyntheticModel("cpu-sel-monitor");
        LairConfig config = ConfigFor(path, "cpu-sel-monitor", isa: null) with
        {
            Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },
        };

        using GenerationService engine = GenerationService.Open(config, NullLogger.Instance);
        WebApplication app = ApiApp.Build(config, engine);
        await ((IHost)app).StartAsync(CancellationToken.None);
        try
        {
            Task<GenerationResult> load = Task.Run(() => Generate(engine, maxTokens: 8));

            StringWriter output = new();
            using CancellationTokenSource cancel = new(TimeSpan.FromSeconds(30));
            Task<int> run = MonitorCommand.RunAsync(new Uri(app.Urls.Single()), output, cancel.Token);

            Stopwatch clock = Stopwatch.StartNew();
            bool sawRow = false;
            while (clock.Elapsed < TimeSpan.FromSeconds(25) && !sawRow)
            {
                sawRow = DataRows(output.ToString()).Any(row => row.Contains("cpu isa="));
                if (!sawRow)
                {
                    await Task.Delay(200);
                }
            }

            cancel.Cancel();
            int exitCode = await run;
            Assert.NotEmpty((await load).Text); // the host's own engine completed its greedy run

            Assert.True(sawRow, $"the monitor never rendered a row with the resolved CPU selection:{Environment.NewLine}{output}");
            Assert.Equal(0, exitCode);

            List<string> rows = DataRows(output.ToString());
            Assert.Contains(rows, row => row.Contains("cpu-sel-monitor")); // the classic columns still render.
            Assert.Contains(rows, row => row.Contains("cpu isa=") && row.Contains("gather="));

            string row = rows.Single(r => r.Contains("cpu-sel-monitor") && r.Contains("cpu isa="));
            Assert.Contains(ValueAfter(row, "cpu isa="), AllowedResolvedIsas);
            Assert.Contains(ValueAfter(row, "gather="), AllowedGatherModes);

            Assert.DoesNotContain(rows, row => row.Contains(" gpu ", StringComparison.Ordinal)); // the accelerator tail stays absent on a CPU-only serve.
            Assert.DoesNotContain(rows, row => row.Contains("persistent-vram", StringComparison.Ordinal));
        }
        finally
        {
            await ((IHost)app).StopAsync(TimeSpan.FromSeconds(5));
            await app.DisposeAsync();
        }
    }

    // ---------------------------------------------------------------- helpers

    private static CpuIsa Parse(string isa) => CpuIsaNames.TryParse(isa, out CpuIsa parsed) ? parsed : throw new InvalidOperationException($"unreachable test ISA '{isa}'");

    private static CpuFeaturePosture VnniCapablePosture() => new(Avx2Available: true, AvxVnniAvailable: true);

    private GenerationService OpenEngine(string modelName, string? isa, CpuFeaturePosture cpuFeatures, CapturingLogger logger)
    {
        GenerationService engine = GenerationService.Open(
            ConfigFor(SyntheticModel(modelName), modelName, isa),
            logger,
            ledger: null,
            compute: null,
            transport: null,
            cpuFeatures: cpuFeatures);

        engine.AttachMetrics(NullLoggerFactory.Instance, TimeSpan.FromMilliseconds(50));
        return engine;
    }

    private static LairConfig ConfigFor(string modelPath, string modelName, string? isa) => new()
    {
        Server = new ServerConfig(),
        Engine = new EngineConfig { Threads = 1, Cpu = isa is null ? new EngineCpuConfig() : new EngineCpuConfig { Isa = isa } },
        Models =
        [
            new ModelConfig
            {
                Name = modelName,
                Path = modelPath,
                MaxContextTokens = 512,
                KvBlocks = 8,
            },
        ],
    };

    private static GenerationResult Generate(GenerationService engine, int maxTokens) =>
        engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = "hello world",
            MaxTokens = maxTokens,
            Temperature = 0f,
        }, CancellationToken.None).GetAwaiter().GetResult();

    private static MetricsSnapshot NewestSnapshotAfterLoad(GenerationService engine)
    {
        _ = Generate(engine, maxTokens: 4);

        EngineMetricsBus bus = engine.Metrics!;
        for (int attempt = 0; attempt < 40; attempt++)
        {
            IReadOnlyList<MetricsSnapshot> ring = bus.SnapshotRing();
            if (ring.Count > 0)
            {
                return ring[0];
            }

            Thread.Sleep(50);
        }

        throw new InvalidOperationException("the bus published no snapshots after a completed generation");
    }

    // The value one key carries to the end of a load log line.
    private static string TailValue(string line, string key)
    {
        int start = line.IndexOf(key, StringComparison.Ordinal);
        Assert.True(start >= 0, $"log line '{line}' carries no '{key}' field");
        return line[(start + key.Length)..].Trim();
    }

    // The value one field carries up to the next space in a rendered row.
    private static string ValueAfter(string row, string key)
    {
        int start = row.IndexOf(key, StringComparison.Ordinal);
        Assert.True(start >= 0, $"row '{row}' carries no '{key}' field");
        string rest = row[(start + key.Length)..];
        int space = rest.IndexOf(' ');
        return space < 0 ? rest.Trim() : rest[..space].Trim();
    }

    private static List<string> DataRows(string text) =>
        [.. text.Split('\n', StringSplitOptions.RemoveEmptyEntries).Where(l => !l.StartsWith("model", StringComparison.Ordinal))];

    private string SyntheticModel(string modelName)
    {
        string path = SyntheticLlamaGguf.Write(modelName);
        _tempFiles.Add(path);
        return path;
    }

    private sealed class CapturingLogger : ILogger
    {
        private readonly List<(LogLevel Level, string Message)> _entries = [];

        public IReadOnlyList<string> Messages(LogLevel level) =>
            [.. _entries.Where(entry => entry.Level == level).Select(entry => entry.Message)];

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
        {
            ArgumentNullException.ThrowIfNull(formatter);
            _entries.Add((logLevel, formatter(state, exception)));
        }
    }
}
