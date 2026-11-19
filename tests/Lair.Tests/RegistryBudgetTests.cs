using System.Diagnostics;

using Lair.Host.Config;
using Lair.Http;
using Lair.Models;
using Microsoft.Extensions.Logging;

namespace Lair.Tests;

/// <summary>
/// Device budgets across a multi-model serve: every served instance claims against one host accounting in open order, and an over-commit
/// moves each resource down its own chain rather than turning work away while a lower position is free — the model that claimed first keeps
/// its tier whole. A final copy-holding position with nothing left refuses naming model, resource, tier, and what every consulted position
/// holds; at startup that refusal fails the whole declared posture (and leaves no claim standing under a name no engine serves), while in a
/// live reload it refuses only that addition behind everything already held. A budget refusal never disturbs an admitted sequence: running
/// work finishes exactly as it started, and only new admissions answer with the exhaustion code.
/// </summary>
[Trait("Category", "Budget")]
public sealed class RegistryBudgetTests : IDisposable
{
    private readonly List<string> _tempPaths = [];

    public void Dispose()
    {
        foreach (string path in _tempPaths)
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

        _tempPaths.Clear();
    }

    [Fact]
    public void Open_TwoDeclaredModels_OverCommitSystemMemory_SpillsEachResourceDownItsOwnChain_FirstHolderKeepsItsTier()
    {
        string alphaPath = SyntheticModel();
        string betaPath = SyntheticModel();

        // One megabyte of system-memory weight backing across the whole host: exactly one model's working copy fits it. The KV seam is roomy
        // enough for both reservations side by side, so only the resource that actually ran over commits moves.
        DeviceLedger ledger = new([new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.Weights, Capacity = 1 }, new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.Kv, Capacity = 8 }]);

        using ModelRegistry registry = ModelRegistry.Open(
            ConfigWith([WeightedModel("alpha", alphaPath), WeightedModel("beta", betaPath)]),
            logger: null,
            transport: null,
            ledger);

        Assert.Equal(["alpha", "beta"], registry.AdvertisedIds()); // declared order is advertised order.

        List<DeviceLedgerClaim> weights = [.. ledger.Claims.Where(claim => claim.Resource == BudgetResource.Weights)];
        Assert.Equal(2, weights.Count);

        Assert.Equal("alpha", weights[0].Model);
        Assert.Equal(BudgetTier.Dram, weights[0].Tier);   // the first holder keeps its working-copy tier whole.
        Assert.False(weights[0].Demoted);

        Assert.Equal("beta", weights[1].Model);           // the over-commit lands on the next position of its own chain...
        Assert.Equal(BudgetTier.Disk, weights[1].Tier);   // ...the origin store this read-side resource already reads from.
        Assert.True(weights[1].AtOriginStore);
        Assert.True(weights[1].Demoted);                  // a demotion of the chain's own making, never a displacement of an earlier claim.
        Assert.True(weights[1].Sequence > weights[0].Sequence);

        IReadOnlyList<DeviceLedgerClaim> kvClaims = ledger.ClaimsAt(BudgetTier.Dram, BudgetResource.Kv);
        Assert.Equal(["alpha", "beta"], [.. kvClaims.Select(claim => claim.Model)]); // the resource with room left stayed put: chains decide per resource.

        using ModelRegistry.Lease serving = registry.BeginRequest("beta");
        Assert.True(Generate(serving.Engine, maxTokens: 4).CompletionTokens > 0); // a demoted seat serves what its chain says it reads from.
    }

    [Fact]
    public void Open_SecondDeclaredModelWeightChainExhausted_FailsWholeServe_NamingModelResourceTierAndBreakdown_AndLeavesNothingStanding()
    {
        string alphaPath = SyntheticModel();
        string betaPath = SyntheticModel();

        DeviceLedger ledger = new([new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.Weights, Capacity = 1 }]); // one seat total; the second model declares no lower position.

        LairConfig candidate = ConfigWith([SystemMemoryOnlyWeights("alpha", alphaPath), SystemMemoryOnlyWeights("beta", betaPath)]);

        BudgetExhaustedException loadFailure = Assert.Throws<BudgetExhaustedException>(
            () => ModelRegistry.Open(candidate, logger: null, transport: null, ledger)); // a declared posture opens every model it declares or none.

        Assert.Equal("beta", loadFailure.Model);
        Assert.Equal(BudgetResource.Weights, loadFailure.Resource);
        Assert.Equal(BudgetTier.Dram, loadFailure.Tier);
        Assert.Contains("'dram' holds 1 of 1 megabytes ('alpha' 1 megabytes)", loadFailure.Breakdown); // actionable: the model, its resource, the tier, and what stands there.

        Assert.Empty(ledger.Claims); // nothing served from this posture, so nothing may keep holding seats in host accounting under names no engine answers for.
    }

    [Fact]
    public void ApplyConfiguration_AddModelOverCommittingSystemMemory_SpillsDownItsOwnChain_BehindEveryClaimAlreadyHeld()
    {
        string alphaPath = SyntheticModel();
        string betaPath = SyntheticModel();

        DeviceLedger ledger = new([new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.Weights, Capacity = 1 }, new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.Kv, Capacity = 8 }]); // the live model already holds the only system-memory weight seat.

        LairConfig start = ConfigWith([WeightedModel("alpha", alphaPath)]);
        using ModelRegistry registry = ModelRegistry.Open(start, logger: null, transport: null, ledger);

        DeviceLedgerClaim heldFirst = Assert.Single(ledger.ClaimsAt(BudgetTier.Dram, BudgetResource.Weights)); // what the live posture holds is what a later addition must seat behind.
        Assert.Equal("alpha", heldFirst.Model);

        IReadOnlyList<FieldResult> results = registry.ApplyConfiguration(ConfigWith([WeightedModel("alpha", alphaPath), WeightedModel("beta", betaPath)]));

        FieldResult addition = Assert.Single(results, result => result.Change.Field == "models[1]");
        Assert.Equal(ConfigApplyOutcome.Applied, addition.Outcome); // the addition is accepted into the posture and loads in the background behind what stands there.
        Assert.Contains("'beta' added", addition.Detail);

        WaitUntil(() => registry.AdvertisedIds().Contains("beta"), "'beta' to advertise after its background open");

        List<DeviceLedgerClaim> weights = [.. ledger.Claims.Where(claim => claim.Resource == BudgetResource.Weights)];
        Assert.Equal(["alpha", "beta"], [.. weights.Select(claim => claim.Model)]); // registration order is the order claims took: the addition lands last.
        Assert.True(weights[1].AtOriginStore && weights[1].Demoted); // it spilled down its own chain instead of displacing the seated holder.
        Assert.Equal(BudgetTier.Disk, weights[1].Tier);

        using ModelRegistry.Lease serving = registry.BeginRequest("alpha"); // and everything already served kept its tier whole.
        Assert.True(Generate(serving.Engine, maxTokens: 4).CompletionTokens > 0);
    }

    [Fact]
    public void ApplyConfiguration_AddModelWhoseFinalTierIsExhausted_RefusesActionably_WithoutDisturbingServedModels_AndSeatsNothing()
    {
        string alphaPath = SyntheticModel();
        string betaPath = SyntheticModel();
        CapturingLogger logger = new();

        DeviceLedger ledger = new([new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.Kv, Capacity = 8 }]); // the live model's reservation fills this tier end to end.

        LairConfig start = ConfigWith([KvBudgetedModel("alpha", alphaPath, kvBlocks: 8)]);
        using ModelRegistry registry = ModelRegistry.Open(start, logger, transport: null, ledger);

        IReadOnlyList<FieldResult> results = registry.ApplyConfiguration(ConfigWith([KvBudgetedModel("alpha", alphaPath, kvBlocks: 8), KvBudgetedModel("beta", betaPath, kvBlocks: 4)]));

        FieldResult addition = Assert.Single(results, result => result.Change.Field == "models[1]");
        Assert.Equal(ConfigApplyOutcome.Applied, addition.Outcome); // the attempt is reported where it was made; its refusal lands when the open runs behind what already holds.

        WaitUntil(() => logger.Messages(LogLevel.Error).Any(line => line.Contains("model_load_failed model='beta'", StringComparison.Ordinal)), "the unseat-able addition to report its refusal");

        string refusal = Assert.Single(logger.Messages(LogLevel.Error), line => line.Contains("model_load_failed model='beta'", StringComparison.Ordinal));
        Assert.Contains("outcome=refused", refusal);
        Assert.Contains("resource='kv'", refusal);   // actionable by the same vocabulary every other exhaustion answers with: model, resource, tier...
        Assert.Contains("tier='dram'", refusal);
        Assert.Contains("'dram' holds 8 of 8 KV blocks ('alpha' 8 KV blocks)", refusal); // ...beside the ledger breakdown of what stands at that position.

        Assert.Equal(["alpha"], registry.AdvertisedIds()); // nothing half-opened advertised, and no claim or grant was left standing under the refused name.
        Assert.DoesNotContain(registry.Ledger!.Claims, claim => claim.Model == "beta");
        Assert.Null(registry.Compute?.GrantFor("beta"));

        using ModelRegistry.Lease serving = registry.BeginRequest("alpha"); // every model that was already served kept serving untouched.
        Assert.True(Generate(serving.Engine, maxTokens: 4).CompletionTokens > 0);
    }

    [Fact]
    public async Task AdmissionRefusalUnderCrossModelOverCommit_NamesBudgetExhaustion_WithoutPreemptingAnAdmittedSequence()
    {
        string alphaPath = SyntheticModel();
        string betaPath = SyntheticModel();

        DeviceLedger ledger = new([new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.Kv, Capacity = 16 }]); // both reservations seat side by side.

        using ModelRegistry registry = ModelRegistry.Open(
            ConfigWith([KvBudgetedModel("alpha", alphaPath, kvBlocks: 8), KvBudgetedModel("beta", betaPath, kvBlocks: 8)]),
            logger: null,
            transport: null,
            ledger);

        // One sequence admitted on the first model runs while accounting tightens around it.
        using ModelRegistry.Lease runningLease = registry.BeginRequest("alpha");
        Task<GenerationResult> inFlight = Task.Run(() => Generate(runningLease.Engine, maxTokens: 100));
        WaitUntil(() => runningLease.Engine.ServedScheduler.Stats.SequencesAdmitted >= 1, "'alpha' to admit its sequence before the posture tightens");

        ledger.UpdateCapacity(BudgetTier.Dram, BudgetResource.Kv, capacity: 8); // the host re-reports a smaller total than what already stands there — seated claims stay whole.

        using ModelRegistry.Lease refused = registry.BeginRequest("beta"); // the refusal releases its slot and leaves no pool or ledger state behind.
        EngineBusyException refusal = Assert.Throws<EngineBusyException>(() => Generate(refused.Engine, maxTokens: 40));
        Assert.Equal("budget_exhausted", refusal.Code); // budget-chain exhaustion answers as budget exhaustion, not overload or context length.
        Assert.Contains("'beta'", refusal.Message);
        Assert.Contains("'dram' holds 8 of 8 KV blocks ('alpha' 8 KV blocks, 'beta' 8 KV blocks)", refusal.Message);

        GenerationResult completed = await inFlight; // the admitted sequence was never preempted by a later refusal.
        Assert.True(completed.CompletionTokens > 0);

        IReadOnlyList<DeviceLedgerClaim> standing = ledger.ClaimsAt(BudgetTier.Dram, BudgetResource.Kv);
        Assert.Equal(["alpha", "beta"], [.. standing.Select(claim => claim.Model)]);
        Assert.Equal([8L, 8L], [.. standing.Select(claim => claim.Amount)]); // both reservations stayed seated exactly as registered.

        ledger.UpdateCapacity(BudgetTier.Dram, BudgetResource.Kv, capacity: 16); // the host restates its total: what the refusal turned away was room at a tier, not state either engine held.
        using ModelRegistry.Lease admittedAfter = registry.BeginRequest("beta");
        Assert.True(Generate(admittedAfter.Engine, maxTokens: 4).CompletionTokens > 0); // the refused model's own pool and pipeline came back untouched and serves again.
    }

    [Fact]
    public void Open_TwoDeclaredModels_OverCommitWorkerClaims_ClampsWithoutDisturbingEarlierSeats_AndBothKeepServing()
    {
        string alphaPath = SyntheticModel();
        string betaPath = SyntheticModel();

        // A two-processor posture is all the host accounting has to seat worker claims against: the first claim takes it whole.
        ComputeCapacityLedger compute = new(new HostTopology
        {
            Nodes = [new HostNodeCapacity { Index = 0, LogicalProcessors = 2, CoreIds = [0, 1] }],
        });

        using ModelRegistry registry = ModelRegistry.Open(
            ConfigWith([QuotaModel("alpha", alphaPath), QuotaModel("beta", betaPath)], threads: 2), // both models ask to run with two workers against a posture that carries two.
            logger: null,
            transport: null,
            ledger: null,
            compute);

        Assert.NotNull(compute.GrantFor("alpha")); // both claims seated in open order against one host posture.
        ComputeGrant heldFirst = compute.GrantFor("alpha")!;
        Assert.Equal(2, heldFirst.RequestedCores);
        Assert.Equal(2, heldFirst.ExclusiveCores);   // the earlier seat keeps everything it claimed.
        Assert.False(heldFirst.SharesHeldCores);

        Assert.NotNull(compute.GrantFor("beta")); // an oversubscribed claim is clamped to what its predecessors left — never a refusal, never a displacement.
        ComputeGrant seatedLast = compute.GrantFor("beta")!;
        Assert.Equal(2, seatedLast.RequestedCores);
        Assert.Equal(1, seatedLast.GrantedCores);
        Assert.True(seatedLast.SharesHeldCores);

        using ModelRegistry.Lease first = registry.BeginRequest("alpha");
        Assert.True(Generate(first.Engine, maxTokens: 4).CompletionTokens > 0);

        using ModelRegistry.Lease second = registry.BeginRequest("beta"); // a partial share still runs the model it was seated for.
        Assert.True(Generate(second.Engine, maxTokens: 4).CompletionTokens > 0);
    }

    private string SyntheticModel()
    {
        string path = SyntheticLlamaGguf.Write("registry-budget-model");
        _tempPaths.Add(path);
        return path;
    }

    // One tiny synthetic model's weight set rounds to a single megabyte, so one seat of weight backing is exactly one model's working copy.
    // Its KV reservation carries its own chain too: the resource with room beside what stands there must stay put while weights spill.
    private static ModelConfig WeightedModel(string name, string path) => new()
    {
        Name = name,
        Path = path,
        MaxContextTokens = 512,
        KvBlocks = 4,
        Budgets =
        [
            new BudgetChain { Resource = BudgetResource.Weights, Order = [BudgetTier.Dram, BudgetTier.Disk], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 64 } },
            new BudgetChain { Resource = BudgetResource.Kv, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 4 } },
        ],
    };

    private static ModelConfig SystemMemoryOnlyWeights(string name, string path) => new()
    {
        Name = name,
        Path = path,
        MaxContextTokens = 512,
        KvBlocks = 4,
        Budgets = [new BudgetChain { Resource = BudgetResource.Weights, Order = [BudgetTier.Dram] }], // no lower position: an exhausted system-memory seat has nowhere to spill.
    };

    private static ModelConfig KvBudgetedModel(string name, string path, int kvBlocks) => new()
    {
        Name = name,
        Path = path,
        MaxContextTokens = 512,
        KvBlocks = kvBlocks,
        Budgets = [new BudgetChain { Resource = BudgetResource.Kv, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = kvBlocks } }],
    };

    // A model that asks to run with two workers: what the host accounting seats for it is this ledger's answer, not its own request.
    private static ModelConfig QuotaModel(string name, string path) => new()
    {
        Name = name,
        Path = path,
        MaxContextTokens = 512,
        KvBlocks = 4,
        Placement = new PlacementConfig { ThreadQuota = 2 },
    };

    private static LairConfig ConfigWith(IReadOnlyList<ModelConfig> models, int threads = 1) => new()
    {
        Server = new ServerConfig(),
        Engine = new EngineConfig { Threads = threads },
        Models = [.. models],
        Host = null, // these gates state no host totals at all: the accounting they seat against is supplied through the seams.
    };

    private static GenerationResult Generate(GenerationService engine, string prompt = "hello world", int maxTokens = 8) =>
        engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = prompt,
            MaxTokens = maxTokens,
            Temperature = 0f,
        }, CancellationToken.None).GetAwaiter().GetResult();

    private static void WaitUntil(Func<bool> condition, string what)
    {
        Stopwatch clock = Stopwatch.StartNew();
        while (!condition())
        {
            if (clock.Elapsed > TimeSpan.FromSeconds(30))
            {
                throw new TimeoutException($"timed out waiting for {what}");
            }

            Thread.Sleep(50);
        }
    }

    private sealed class CapturingLogger : ILogger
    {
        private readonly object _gate = new();
        private readonly List<(LogLevel Level, string Message)> _entries = [];

        public IReadOnlyList<string> Messages(LogLevel level)
        {
            lock (_gate)
            {
                return [.. _entries.Where(entry => entry.Level == level).Select(entry => entry.Message)];
            }
        }

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
        {
            ArgumentNullException.ThrowIfNull(formatter);

            lock (_gate)
            {
                _entries.Add((logLevel, formatter(state, exception))); // registry loads report from background threads.
            }
        }
    }
}
