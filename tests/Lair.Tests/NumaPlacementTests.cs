using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Host.Config;
using Lair.Http;
using Lair.Http.Monitoring;
using Lair.Models;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// NUMA placement derivation over a reported posture: the resolver is pure and deterministic — the same posture and the
/// same ordered claims produce the same placement map on every run. Auto seats on the least-loaded node with the lowest
/// node index as tiebreak, local maps a pinned core set to the node that reports carrying it, and interleave records a
/// spread posture across the reported nodes. A pinned set beside a spread posture, and a pinned set spanning reported
/// nodes beside a single-node posture, are conflicts naming model, key, and conflict; a posture that reports no core ids
/// carries a pin as an advisory rather than emulating a mapping. The ledger seats grants on the nodes placement chose,
/// records the per-node share in every grant, and clamps per node without displacing an earlier seat. At the serve seam
/// the placement is logged and rides the monitor facts beside the grant it seated on.
/// </summary>
[Trait("Category", "Affinity")]
[Trait("Category", "Config")]
[Trait("Category", "Scheduling")]
public sealed class NumaPlacementTests : IDisposable
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
    public void Resolver_SamePostureAndClaimOrder_RepeatRunsProduceTheSamePlacementMap()
    {
        HostTopology posture = SyntheticTwoNodePosture(processorsPerNode: 3);
        IReadOnlyList<NumaClaim> claims =
        [
            new NumaClaim { Model = "alpha", RequestedCores = 2 },
            new NumaClaim { Model = "beta", RequestedCores = 2, Spread = NumaSpreadPreference.Interleave },
            new NumaClaim { Model = "gamma", RequestedCores = 1, Spread = NumaSpreadPreference.Local, CoreSet = new HashSet<int>([4, 5]) },
        ];

        IReadOnlyList<NumaPlacement> first = NumaPlacementResolver.Resolve(posture, claims);
        IReadOnlyList<NumaPlacement> second = NumaPlacementResolver.Resolve(posture, claims);

        Assert.Equal(first.Select(placement => placement.Describe()), second.Select(placement => placement.Describe())); // every placement is a value record: the same inputs resolve to the same map.
        Assert.Equal(["alpha", "beta", "gamma"], [.. first.Select(placement => placement.Model)]); // the map keeps claim order.
        Assert.Equal([0], first[0].NodeIndices);
        Assert.Equal([0, 1], first[1].NodeIndices);
        Assert.Equal([1], first[2].NodeIndices);
    }

    [Fact]
    public void AutoPosture_ChoosesLeastLoadedNode_ThenLowestNodeIndexAsTiebreak()
    {
        HostTopology posture = SyntheticTwoNodePosture(processorsPerNode: 4);

        NumaPlacement fresh = NumaPlacementResolver.Resolve(posture, [new NumaClaim { Model = "first", RequestedCores = 1 }])[0];
        Assert.Equal([0], fresh.NodeIndices); // nothing loaded: the lowest node index wins.

        NumaPlacement seeded = NumaPlacementResolver.Resolve(
            posture,
            [new NumaClaim { Model = "second", RequestedCores = 1 }],
            initialNodeLoads: new Dictionary<int, int> { [0] = 3 })[0];
        Assert.Equal([1], seeded.NodeIndices); // node 0 stands loaded: the least-loaded node is node 1.

        NumaPlacement tie = NumaPlacementResolver.Resolve(
            posture,
            [new NumaClaim { Model = "fourth", RequestedCores = 2 }],
            initialNodeLoads: new Dictionary<int, int> { [0] = 1, [1] = 1 })[0];
        Assert.Equal([0], tie.NodeIndices); // equal load: the tiebreak is the lowest node index.
    }

    [Fact]
    public void AutoPosture_LoadFollowsTheWorkerSharesEarlierClaimsInTheRunSeated()
    {
        HostTopology posture = SyntheticTwoNodePosture(processorsPerNode: 4);

        IReadOnlyList<NumaPlacement> map = NumaPlacementResolver.Resolve(posture, [
            new NumaClaim { Model = "wide", RequestedCores = 3 },
            new NumaClaim { Model = "narrow", RequestedCores = 1 },
        ]);

        Assert.Equal([0], map[0].NodeIndices);   // the first claim loads node 0 with its share.
        Assert.Equal([1], map[1].NodeIndices);   // the next claim sees that load and takes the other node.
    }

    [Fact]
    public void LocalPosture_MapsPinnedCoreSetToTheNodeReportingIt_AndAutoReadsBackAsTheLocalPostureThePinBecame()
    {
        HostTopology posture = SyntheticTwoNodePosture(processorsPerNode: 3); // node 1 reports core ids 3, 4, 5.

        NumaPlacement local = NumaPlacementResolver.Resolve(posture, [
            new NumaClaim { Model = "pinned", RequestedCores = 3, Spread = NumaSpreadPreference.Local, CoreSet = new HashSet<int>([4, 5]) },
        ])[0];

        Assert.Equal([1], local.NodeIndices);
        Assert.Equal(NumaSpreadPreference.Local, local.Requested);
        Assert.Equal(NumaSpreadPreference.Local, local.Applied);
        Assert.Equal([4, 5], local.CoreIds); // the declared ids ride onto the node that reports carrying them.
        Assert.Null(local.Advisory);          // the posture honoured the pin whole.

        NumaPlacement auto = NumaPlacementResolver.Resolve(posture, [
            new NumaClaim { Model = "auto-pinned", RequestedCores = 3, CoreSet = new HashSet<int>([4, 5]) },
        ])[0];

        Assert.Equal([1], auto.NodeIndices);
        Assert.Equal(NumaSpreadPreference.Auto, auto.Requested);
        Assert.Equal(NumaSpreadPreference.Local, auto.Applied); // a pin localized an auto choice: the map reports the posture it applied.
    }

    [Fact]
    public void InterleaveBesideAPinnedCoreSet_RefusesNamingModelKeyAndConflict()
    {
        HostTopology posture = SyntheticTwoNodePosture(processorsPerNode: 3);

        ModelException error = Assert.Throws<ModelException>(() => NumaPlacementResolver.Resolve(posture, [
            new NumaClaim { Model = "conflicting", RequestedCores = 2, Spread = NumaSpreadPreference.Interleave, CoreSet = new HashSet<int>([0, 1]) },
        ]));

        Assert.Contains("'conflicting'", error.Message);
        Assert.Contains("placement.numa_spread 'interleave'", error.Message);
        Assert.Contains("placement.core_set", error.Message); // the refusal names both keys in conflict.
    }

    [Fact]
    public void PinnedCoreSetSpanningReportedNodes_RefusesAgainstAPostureReportingCoreIds()
    {
        HostTopology posture = SyntheticTwoNodePosture(processorsPerNode: 3); // core 2 sits on node 0, core 3 on node 1.

        ModelException local = Assert.Throws<ModelException>(() => NumaPlacementResolver.Resolve(posture, [
            new NumaClaim { Model = "spanning", RequestedCores = 2, Spread = NumaSpreadPreference.Local, CoreSet = new HashSet<int>([2, 3]) },
        ]));

        Assert.Contains("'spanning'", local.Message);
        Assert.Contains("placement.core_set", local.Message);
        Assert.Contains("spans reported nodes 0, 1", local.Message);

        ModelException auto = Assert.Throws<ModelException>(() => NumaPlacementResolver.Resolve(posture, [
            new NumaClaim { Model = "spaning-auto", RequestedCores = 2, CoreSet = new HashSet<int>([0, 5]) },
        ]));

        Assert.Contains("'spaning-auto'", auto.Message);
        Assert.Contains("placement.numa_spread 'auto'", auto.Message);
    }

    [Fact]
    public void PinnedPostureAgainstAPostureWithoutCoreIds_ReportsAdvisoryRatherThanEmulatingAMapping()
    {
        HostTopology posture = new()
        {
            Nodes =
            [
                new HostNodeCapacity { Index = 0, LogicalProcessors = 4 },
                new HostNodeCapacity { Index = 1, LogicalProcessors = 4 },
            ],
            DerivedFromReportedTopology = false, // the documented fallback: nodes sized, no core ids stated.
        };

        NumaPlacement placement = NumaPlacementResolver.Resolve(posture, [
            new NumaClaim { Model = "advisory", RequestedCores = 2, Spread = NumaSpreadPreference.Local, CoreSet = new HashSet<int>([1, 2]) },
        ])[0];

        Assert.Equal([0], placement.NodeIndices);        // a node is still chosen — the least-loaded one.
        Assert.NotNull(placement.Advisory);              // and the mapping it could not state is reported, not invented.
        Assert.Contains("no core ids", placement.Advisory);
        Assert.Equal([1, 2], placement.CoreIds);         // the pin itself rides along for the backend to carry.
    }

    [Fact]
    public void InterleavePosture_RecordsASpreadAcrossEveryReportedNode()
    {
        HostTopology posture = SyntheticTwoNodePosture(processorsPerNode: 2);

        NumaPlacement placement = NumaPlacementResolver.Resolve(posture, [
            new NumaClaim { Model = "spread", RequestedCores = 5, Spread = NumaSpreadPreference.Interleave },
        ])[0];

        Assert.Equal([0, 1], placement.NodeIndices);
        Assert.True(placement.IsSpread);
        Assert.Equal(NumaSpreadPreference.Interleave, placement.Applied);
        Assert.Empty(placement.CoreIds); // a spread posture names nodes, never cores.
    }

    [Fact]
    public void TotalsOnlyClaims_StillSeatPerNodeFromTheReportedPosture_AndRecordTheShareInEveryGrant()
    {
        ComputeCapacityLedger ledger = new(SyntheticTwoNodePosture(processorsPerNode: 4)); // eight logical processors reported, no placement supplied.

        ComputeGrant first = ledger.RegisterClaim("first", requestedCores: 5);
        Assert.Equal(5, first.GrantedCores);
        Assert.Equal(new Dictionary<int, int> { [0] = 4, [1] = 1 }, first.NodeAllocations); // the host-total seat reads node by node.

        ComputeGrant second = ledger.RegisterClaim("second", requestedCores: 6);
        Assert.Equal(3, second.GrantedCores); // the same clamping this ledger did before per-node seats existed.
        Assert.Equal(new Dictionary<int, int> { [1] = 3 }, second.NodeAllocations); // node 0 ran out; this seat lands on what node 1 left.

        Assert.Equal(0, ledger.RemainingCoresForNode(0));
        Assert.Equal(0, ledger.RemainingCoresForNode(1));
        Assert.Equal(new Dictionary<int, int> { [0] = 4, [1] = 4 }, ledger.NodeAllocations); // the host view placement reads as load.
        Assert.Contains("[0:4,1:1]", first.Describe()); // the per-node share rides the grant's own diagnostic text.
    }

    [Fact]
    public void PlacedClaims_ClampPerNode_AndANodeFullClaimStillRunsBesideStandingGrants()
    {
        ComputeCapacityLedger ledger = new(SyntheticTwoNodePosture(processorsPerNode: 4));

        ComputeGrant first = ledger.RegisterClaim("first", 3, new NumaPlacement
        {
            Model = "first",
            Requested = NumaSpreadPreference.Auto,
            Applied = NumaSpreadPreference.Auto,
            NodeIndices = [0],
        });
        Assert.Equal(3, first.GrantedCores);
        Assert.Equal(new Dictionary<int, int> { [0] = 3 }, first.NodeAllocations);

        ComputeGrant second = ledger.RegisterClaim("second", 5, new NumaPlacement
        {
            Model = "second",
            Requested = NumaSpreadPreference.Interleave,
            Applied = NumaSpreadPreference.Interleave,
            NodeIndices = [0, 1],
        });
        Assert.Equal(5, second.GrantedCores); // node 0 left one processor and node 1 all four: the spread seats on both.
        Assert.Equal(new Dictionary<int, int> { [0] = 1, [1] = 4 }, second.NodeAllocations);

        ComputeGrant third = ledger.RegisterClaim("third", 2, new NumaPlacement
        {
            Model = "third",
            Requested = NumaSpreadPreference.Auto,
            Applied = NumaSpreadPreference.Auto,
            NodeIndices = [1], // a posture whose chosen node is full: it clamps there rather than moving.
        });
        Assert.Equal(1, third.GrantedCores);
        Assert.Equal(0, third.ExclusiveCores);
        Assert.True(third.SharesHeldCores);
        Assert.Empty(third.NodeAllocations); // it seated nothing as its own — the overlap says so.

        Assert.Equal(0, ledger.RemainingCoresForNode(0));
        Assert.Equal(0, ledger.RemainingCoresForNode(1));

        ModelException wrongModel = Assert.Throws<ModelException>(() => ledger.RegisterClaim("fourth", 1, new NumaPlacement
        {
            Model = "elsewhere",
            Requested = NumaSpreadPreference.Auto,
            Applied = NumaSpreadPreference.Auto,
            NodeIndices = [0],
        }));
        Assert.Contains("'fourth'", wrongModel.Message);
        Assert.Contains("'elsewhere'", wrongModel.Message); // a claim seats on the placement resolved for it.
    }

    [Fact]
    public void OpenedEngine_LogsItsPlacement_AndCarriesNodeSpreadAndPerNodeGrantOnMonitorFacts()
    {
        string path = SyntheticModel("placement-serve");
        ComputeCapacityLedger compute = new(SyntheticTwoNodePosture(processorsPerNode: 3)); // node 1 reports core ids 3, 4, 5.

        CapturingLogger log = new();
        using GenerationService engine = OpenEngine(
            path, "placement-serve", threads: 4, logger: log, compute: compute,
            placement: new PlacementConfig { NumaSpread = "local", CoreSet = new HashSet<int>([4, 5]) });

        IReadOnlyList<string> info = log.Messages(LogLevel.Information);
        Assert.Contains(
            "numa_placement model='placement-serve' node=1 spread=local→local core_set='4-5'",
            info); // one line naming the model, the node, and the posture it resolved through.
        Assert.Contains(
            info,
            line => line.Contains("compute_grant model='placement-serve' requested=4 granted=3") && line.Contains("node_allocations=1:3"));

        Assert.Equal([1], engine.ResolvedPlacement!.NodeIndices);
        Assert.Equal(3, engine.EffectiveWorkerCount); // participation rides inside the node its pin chose.

        MetricsSnapshot snapshot = NewestSnapshotAfterLoad(engine);
        Assert.Equal(1, snapshot.PlacementNode);
        Assert.Equal("1", snapshot.PlacementNodes);
        Assert.Equal("local", snapshot.PlacementSpread);
        Assert.Equal("local", snapshot.PlacementSpreadRequested);
        Assert.Equal("1:3", snapshot.PlacementNodeAllocations); // the per-node share the grant seated, on the same surface worker counts use.
        Assert.Null(snapshot.PlacementAdvisory);                 // the posture honoured this pin whole.
    }

    [Fact]
    public void UnpinnedAutoEngine_LogsOnePlacementLineNamingTheNodeItSeatedOn()
    {
        string path = SyntheticModel("placement-auto");
        ComputeCapacityLedger compute = new(SyntheticTwoNodePosture(processorsPerNode: 2));

        CapturingLogger log = new();
        using GenerationService engine = OpenEngine(path, "placement-auto", threads: 2, logger: log, compute: compute);

        Assert.Contains(
            "numa_placement model='placement-auto' node=0 spread=auto→auto",
            log.Messages(LogLevel.Information)); // an unpinned auto posture states the node it chose, not a mapping it did not make.

        MetricsSnapshot snapshot = NewestSnapshotAfterLoad(engine);
        Assert.Equal(0, snapshot.PlacementNode);
        Assert.Equal("auto", snapshot.PlacementSpread);
        Assert.Equal("0:2", snapshot.PlacementNodeAllocations);
    }

    [Fact]
    public void DedicatedLanePosture_SplitsItsLanesInsideTheChosenNode_WhenTheModelPinsNothing()
    {
        if (!CpuBackend.SupportsCorePinning)
        {
            return; // a node-derived lane mask only applies where the host has a real affinity seam
        }

        string path = SyntheticModel("placement-lane");
        ComputeCapacityLedger compute = new(SyntheticTwoNodePosture(processorsPerNode: 4)); // node 0 reports core ids 0-3.

        using GenerationService engine = OpenEngine(
            path, "placement-lane", threads: 8, logger: new CapturingLogger(), compute: compute,
            placement: new PlacementConfig { PrefillLane = PlacementPrefillLaneValues.Dedicated });

        IReadOnlyList<int> nodeCoreIds = compute.Topology.Nodes[0].CoreIds;
        Assert.Equal([0], engine.ResolvedPlacement!.NodeIndices);

        PrefillLaneLayout lane = engine.DedicatedPrefillLanes!;
        Assert.All(lane.PrefillCoreIds, core => Assert.Contains(core, nodeCoreIds)); // each lane claims its side of the chosen node's ids.
        Assert.All(lane.DecodeCoreIds, core => Assert.Contains(core, nodeCoreIds));
        Assert.Empty(lane.PrefillCoreIds.Intersect(lane.DecodeCoreIds));             // and the two sides stay disjoint inside it.
    }

    [Fact]
    public void NodeDerivedAffinity_KeepsWorkerExecutionsInsideTheChosenNode_WhereTheHostCanPin()
    {
        HostTopology posture = SyntheticTwoNodePosture(processorsPerNode: 3);
        NumaPlacement placement = NumaPlacementResolver.Resolve(posture, [new NumaClaim { Model = "placed", RequestedCores = 3 }])[0];
        HostNodeCapacity chosen = posture.Nodes.First(node => node.Index == placement.NodeIndices[0]);

        if (!CpuBackend.SupportsCorePinning)
        {
            // Off-Windows hosts assert the named advisory instead of page residency: this build neither prefaults the
            // model file nor moves pages, and byte placement follows the workers pinned to the chosen node.
            Assert.Contains(NumaPlacementResolver.PagePlacementAdvisory, "first-touch discipline only");
            Assert.Contains(NumaPlacementResolver.PagePlacementAdvisory, "neither prefaults the model file nor moves pages");
            return;
        }

        using CpuBackend backend = new(threads: 3, coreSet: new HashSet<int>(chosen.CoreIds)); // the mask placement derives from the chosen node

        (byte[] weights, float[] x, float[] output) = Workload();
        backend.EnableExecutionSampling();

        for (int call = 0; call < 8; call++)
        {
            backend.Gemv(x, new WeightRef(weights, new TensorShape([16384, 576]), QuantFormat.F32), output);
        }

        IReadOnlyList<int> samples = backend.ExecutionSamples;
        Assert.True(samples.Count >= 100, $"expected at least 100 sampled worker executions but got {samples.Count}.");

        foreach (int processor in samples)
        {
            Assert.Contains(processor, chosen.CoreIds); // every sampled execution ran inside the node placement chose
        }
    }

    private GenerationService OpenEngine(
        string modelPath,
        string modelName,
        int threads,
        CapturingLogger logger,
        ComputeCapacityLedger compute,
        PlacementConfig? placement = null)
    {
        GenerationService engine = GenerationService.Open(
            new LairConfig
            {
                Server = new ServerConfig(),
                Engine = new EngineConfig { Threads = threads },
                Models =
                [
                    new ModelConfig
                    {
                        Name = modelName,
                        Path = modelPath,
                        MaxContextTokens = 512,
                        KvBlocks = 8,
                        Placement = placement ?? new PlacementConfig(),
                    },
                ],
            },
            logger,
            ledger: null,
            compute);

        engine.AttachMetrics(NullLoggerFactory.Instance, TimeSpan.FromMilliseconds(50)); // the same facts path a serving host wires through its monitor build.
        return engine;
    }

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

    private static (byte[] Weights, float[] X, float[] Output) Workload()
    {
        const int Rows = 16384;
        const int Cols = 576;

        byte[] weights = new byte[Rows * Cols * sizeof(float)];
        Random random = new(20261115);
        foreach (ref float value in System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(weights.AsSpan()))
        {
            value = (float)(random.NextDouble() - 0.5) * 0.01f; // small weights keep every output finite
        }

        float[] x = new float[Cols];
        foreach (ref float value in x.AsSpan())
        {
            value = (float)(random.NextDouble() - 0.5);
        }

        return (weights, x, new float[Rows]);
    }

    private static HostTopology SyntheticTwoNodePosture(int processorsPerNode) => new()
    {
        Nodes =
        [
            new HostNodeCapacity
            {
                Index = 0,
                LogicalProcessors = processorsPerNode,
                MemoryMb = null,
                CoreIds = [.. Enumerable.Range(0, processorsPerNode)],
            },
            new HostNodeCapacity
            {
                Index = 1,
                LogicalProcessors = processorsPerNode,
                MemoryMb = null,
                CoreIds = [.. Enumerable.Range(processorsPerNode, processorsPerNode)],
            },
        ],
        DerivedFromReportedTopology = true, // synthetic posture stands in for what a real host would have reported.
    };

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
