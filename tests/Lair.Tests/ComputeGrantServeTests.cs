using System.Text.Json;
using Lair.Host.Config;
using Lair.Http;
using Lair.Http.Monitoring;
using Lair.Models;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// Compute grants at the serve seam: a model opened against accounting an earlier seat already filled runs on what that seat left —
/// its backend participates below what it asked to run with, and the load reports requested against granted rather than refusing the
/// open or displacing the standing claim. A posture fully claimed still serves every later model beside what stands there, reporting
/// the overlap; monitor facts carry requested beside granted so a clamped seat reads that way on the same surfaces worker counts do.
/// Declared tier totals read against the reported posture as warn-and-accept lines rather than refusals. The compute seam is programmatic:
/// an injected ledger carries what its caller reported about nodes, and every open without one derives this process's posture once.
/// </summary>
[Trait("Category", "Scheduling")]
[Trait("Category", "Affinity")]
public sealed class ComputeGrantServeTests : IDisposable
{
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

    [Fact]
    public void LaterOversubscribedModel_ServesOnItsPartialGrant_AndLoadReportsRequestedAgainstGranted()
    {
        string path = SyntheticModel("grant-clamped");
        ComputeCapacityLedger compute = new(SyntheticTwoNodePosture(processorsPerNode: 3, memoryMb: null)); // six logical processors reported.

        CapturingLogger firstLog = new();
        using GenerationService first = OpenEngine(path, "grant-first", threads: 4, logger: firstLog, compute: compute);

        Assert.Contains(
            "numa_placement model='grant-first' node=0 spread=auto→auto",
            firstLog.Messages(LogLevel.Information)); // an unpinned auto posture seats on the least-loaded node, lowest index first.
        Assert.Contains(
            "compute_grant model='grant-first' requested=4 granted=3 exclusive_cores=3 shared_cores=False node_allocations=0:3 nodes=2 logical_processors=6",
            firstLog.Messages(LogLevel.Information)); // a claim seats on its chosen node only: node 0 reported three processors, so four asked for clamps to three.
        Assert.Equal(3, first.EffectiveWorkerCount); // participation rides inside what its chosen node holds, not past it.

        CapturingLogger secondLog = new();
        using GenerationService second = OpenEngine(path, "grant-second", threads: 4, logger: secondLog, compute: compute);

        Assert.Contains(
            "numa_placement model='grant-second' node=1 spread=auto→auto",
            secondLog.Messages(LogLevel.Information)); // node 0 stands loaded, so the least-loaded node is node 1.
        Assert.Contains(
            "compute_grant model='grant-second' requested=4 granted=3 exclusive_cores=3 shared_cores=False node_allocations=1:3 nodes=2 logical_processors=6",
            secondLog.Messages(LogLevel.Information)); // the clamp is reported at load, not inferred from a worker count.
        Assert.Equal(3, second.EffectiveWorkerCount); // participation rides inside what its own chosen node left, below this model's own choice.

        ComputeGrant standing = compute.GrantFor("grant-first")!;
        Assert.Equal(3, standing.GrantedCores); // the earlier seat keeps everything it was granted: partial grants never displace it.
        Assert.Equal(new Dictionary<int, int> { [0] = 3 }, standing.NodeAllocations); // its seat is recorded on the node it seated on.

        _ = Generate(first, "hello world");
        _ = Generate(second, "hello world");

        MetricsSnapshot snapshot = NewestSnapshotAfterLoad(second);
        Assert.Equal(3, snapshot.WorkerThreads);          // the effective count already on the monitor surface.
        Assert.Equal(4, snapshot.ComputeRequestedThreads); // beside it: what this model asked to run with.
        Assert.Equal(3, snapshot.ComputeGrantedThreads);   // against what its chosen node seated for it.
        Assert.Equal(1, snapshot.PlacementNode);           // the node this engine seated on.
        Assert.Equal("auto", snapshot.PlacementSpread);    // beside the posture placement applied.
        Assert.Equal("1:3", snapshot.PlacementNodeAllocations); // and the per-node share its grant seated.
        Assert.Null(snapshot.CoreOvercommit);              // a clamped share is not an overlap: nothing opted in here.
        Assert.Null(snapshot.SharedCoreModels);            // and its own cores stay reported as unshared.

        first.Dispose();
        second.Dispose();
    }

    [Fact]
    public void ExhaustedPosture_StillServesEveryLaterModel_AndReportsTheOverlapItRunsOn()
    {
        string path = SyntheticModel("grant-exhaust");
        ComputeCapacityLedger compute = new(SyntheticTwoNodePosture(processorsPerNode: 2, memoryMb: null)); // four logical processors reported.

        using GenerationService first = OpenEngine(path, "grant-full", threads: 4, logger: new CapturingLogger(), compute: compute);
        using GenerationService second = OpenEngine(path, "grant-second", threads: 4, logger: new CapturingLogger(), compute: compute);
        Assert.Equal(0, compute.RemainingCores); // every reported node is held by the two seats ahead of this claim.

        CapturingLogger thirdLog = new();
        using GenerationService third = OpenEngine(path, "grant-after", threads: 3, logger: thirdLog, compute: compute);

        Assert.Contains(
            "compute_grant model='grant-after' requested=3 granted=1 exclusive_cores=0 shared_cores=True node_allocations= nodes=2 logical_processors=4",
            thirdLog.Messages(LogLevel.Information)); // a claim whose chosen nodes hold nothing left is admitted beside what stands there.
        Assert.Equal(1, third.EffectiveWorkerCount);

        _ = Generate(first, "hello world");   // the standing models keep serving after later claims seat.
        _ = Generate(second, "hello world");
        _ = Generate(third, "hello world");

        MetricsSnapshot snapshot = NewestSnapshotAfterLoad(third);
        Assert.Equal(3, snapshot.ComputeRequestedThreads);
        Assert.Equal(1, snapshot.ComputeGrantedThreads);
        Assert.Contains("grant-after", snapshot.SharedCoreModels!); // the overlap is named on the same surface worker counts use.

        first.Dispose();
        second.Dispose();
        third.Dispose();
    }

    [Fact]
    public void DeclaredTierTotals_AndNodeOverrides_LogWarnAndAcceptAgainstTheReportedPosture()
    {
        string path = SyntheticModel("grant-declared");
        ComputeCapacityLedger compute = new(SyntheticTwoNodePosture(processorsPerNode: 3, memoryMb: [4096, 8192])); // six processors / twelve GB reported.

        CapturingLogger log = new();
        using GenerationService engine = OpenEngine(
            path, "grant-mismatch", threads: 1, logger: log, compute: compute,
            tiers: new TierCapacityConfig { DramCores = 32, DramMb = 65536, Nodes = [new NodeCapacityOverride { Index = 9, Mb = 4096 }] });

        IReadOnlyList<string> info = log.Messages(LogLevel.Information);
        Assert.Contains(info, line => line.StartsWith("compute_topology model='grant-mismatch' nodes=2 logical_processors=6 node_mb_total=12288", StringComparison.Ordinal)); // the posture is reported once per open against its source.

        IReadOnlyList<string> warnings = log.Messages(LogLevel.Warning);
        Assert.Contains(warnings, line => line.Contains("declared cores 32 against reported 6") && line.Contains("accepted against reported topology"));
        Assert.Contains(warnings, line => line.Contains("declared megabytes 65536 against reported 12288"));
        Assert.Contains(warnings, line => line.Contains("node index 9 this posture does not report") && line.Contains("accepted as an override of detection"));

        engine.Dispose();
    }

    [Fact]
    public void DeclaredTierTotals_ThatMatchTheReportedPosture_LogNoCapacityWarning()
    {
        string path = SyntheticModel("grant-matching");
        ComputeCapacityLedger compute = new(SyntheticTwoNodePosture(processorsPerNode: 3, memoryMb: [4096, 8192]));

        CapturingLogger log = new();
        using GenerationService engine = OpenEngine(
            path, "grant-matching", threads: 1, logger: log, compute: compute,
            tiers: new TierCapacityConfig { DramCores = 6, DramMb = 12288 });

        Assert.DoesNotContain(log.Messages(LogLevel.Warning), line => line.Contains("tiers_capacity")); // agreement needs no warning; the posture line still states its source.
        Assert.Contains(log.Messages(LogLevel.Information), line => line.StartsWith("compute_topology model='grant-matching'", StringComparison.Ordinal));

        engine.Dispose();
    }

    [Fact]
    public void OpenWithoutAnInjectedLedger_ServesAgainstThisHostsOwnReportedPosture()
    {
        string path = SyntheticModel("grant-host-default"); // no compute ledger supplied: this open derives the process posture once.

        CapturingLogger log = new();
        GenerationService engine = GenerationService.Open(Config(path, "grant-host", threads: 2), log); // no compute ledger supplied at this call either.
        engine.AttachMetrics(NullLoggerFactory.Instance, TimeSpan.FromMilliseconds(50));

        HostTopology posture = HostTopologyProbe.Detect();
        Assert.Contains(
            log.Messages(LogLevel.Information),
            line => line.Contains($"compute_topology model='grant-host' nodes={posture.Nodes.Count} logical_processors={Environment.ProcessorCount}"));

        Assert.Equal(2, engine.EffectiveWorkerCount); // a host with capacity ahead of the claim leaves today's construction untouched.

        _ = Generate(engine, "hello world");

        MetricsSnapshot snapshot = NewestSnapshotAfterLoad(engine);
        Assert.Equal(2, snapshot.ComputeRequestedThreads);                                        // what this open asked to run with.
        Assert.Equal(2, snapshot.ComputeGrantedThreads);                                          // seated whole against a posture that has room for it.
        Assert.True(snapshot.ComputeGrantedThreads <= posture.TotalLogicalProcessors);           // no grant reaches past what the host reported.

        engine.Dispose();
    }

    [Fact]
    public void SnapshotJson_CarriesTheNewComputeFactsAsTrailingFields_AndKeepsEveryFieldAnExistingConsumerReads()
    {
        string path = SyntheticModel("grant-json");
        ComputeCapacityLedger compute = new(SyntheticTwoNodePosture(processorsPerNode: 2, memoryMb: null));

        using GenerationService engine = OpenEngine(path, "grant-json", threads: 3, logger: new CapturingLogger(), compute: compute);
        MetricsSnapshot snapshot = NewestSnapshotAfterLoad(engine);

        using JsonDocument document = JsonDocument.Parse(snapshot.ToJson());
        JsonElement root = document.RootElement;

        Assert.Equal(JsonValueKind.Number, root.GetProperty("metrics_compute_requested_threads").ValueKind);
        Assert.Equal(2, root.GetProperty("metrics_compute_granted_threads").GetInt32()); // the claim seats on its chosen node, which reports two processors.
        Assert.Equal(0, root.GetProperty("metrics_placement_node").GetInt32());          // the node placement chose for this engine.
        Assert.Equal("auto", root.GetProperty("metrics_placement_spread").GetString());  // the posture it applied, beside the one it applied it through.
        Assert.Equal("0:2", root.GetProperty("metrics_compute_node_allocations").GetString()); // and the per-node share the grant seated.
        Assert.True(root.TryGetProperty("metrics_core_overcommit", out JsonElement overcommit)); // trailing optional fields serialize inert like their neighbors.
        Assert.Equal(JsonValueKind.Null, overcommit.ValueKind);

        // Every field an existing consumer already reads is still there with the same meaning.
        Assert.Equal("grant-json", root.GetProperty("metrics_model").GetString());
        Assert.True(root.TryGetProperty("metrics_worker_threads", out _));
        Assert.True(root.TryGetProperty("metrics_thread_quota", out _));
        Assert.True(root.TryGetProperty("metrics_ledger_occupancy", out _));

        engine.Dispose();
    }

    private GenerationService OpenEngine(
        string modelPath,
        string modelName,
        int threads,
        CapturingLogger logger,
        ComputeCapacityLedger compute,
        TierCapacityConfig? tiers = null)
    {
        GenerationService engine = GenerationService.Open(Config(modelPath, modelName, threads, tiers), logger, ledger: null, compute);
        engine.AttachMetrics(NullLoggerFactory.Instance, TimeSpan.FromMilliseconds(50)); // the same facts path a serving host wires through its monitor build.
        return engine;
    }

    private static LairConfig Config(string modelPath, string modelName, int threads, TierCapacityConfig? tiers = null) => new()
    {
        Server = new ServerConfig(),
        Engine = new EngineConfig { Threads = threads },
        Tiers = tiers,
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

    private static GenerationResult Generate(GenerationService engine, string prompt) =>
        engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = prompt,
            MaxTokens = 8,
            Temperature = 0f,
        }, CancellationToken.None).GetAwaiter().GetResult();

    private static MetricsSnapshot NewestSnapshotAfterLoad(GenerationService engine)
    {
        _ = Generate(engine, "hello world");

        EngineMetricsBus bus = engine.Metrics!; // the attached bus samples this engine's counters and carries its model facts.
        for (int attempt = 0; attempt < 40; attempt++)
        {
            IReadOnlyList<MetricsSnapshot> ring = bus.SnapshotRing();
            if (ring.Count > 0)
            {
                return ring[0]; // newest first: the closing tick after a completed load publishes even once the engine goes quiet.
            }

            Thread.Sleep(50);
        }

        throw new InvalidOperationException("the bus published no snapshots after a completed generation");
    }

    private static HostTopology SyntheticTwoNodePosture(int processorsPerNode, long?[]? memoryMb)
    {
        memoryMb ??= [null, null]; // a posture that states nothing about bytes leaves both nodes unreported.

        return new()
        {
            Nodes =
            [
                new HostNodeCapacity
                {
                    Index = 0,
                    LogicalProcessors = processorsPerNode,
                    MemoryMb = memoryMb[0],
                    CoreIds = [.. Enumerable.Range(0, processorsPerNode)],
                },
                new HostNodeCapacity
                {
                    Index = 1,
                    LogicalProcessors = processorsPerNode,
                    MemoryMb = memoryMb[1],
                    CoreIds = [.. Enumerable.Range(processorsPerNode, processorsPerNode)],
                },
            ],
            DerivedFromReportedTopology = true, // synthetic posture stands in for what a real host would have reported.
        };
    }

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
