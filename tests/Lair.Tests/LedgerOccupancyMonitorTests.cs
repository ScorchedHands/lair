using System.Text.Json;
using Lair.Host.Config;
using Lair.Http;
using Lair.Http.Monitoring;
using Lair.Models;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// Additive ledger occupancy on the monitor surface: snapshots carry one entry per tier position a seated claim holds,
/// with the opened seam's capacity posture and contested holders only where several claimants share that tier — while an
/// engine whose ledger seated no claims publishes exactly the snapshot shape it published before the ledger existed. The
/// fields ride the same attached-facts path every other additive operator view uses, so existing stream consumers read a
/// superset of what they read today and never a different value for any field they already knew.
/// </summary>
[Trait("Category", "Metrics")]
[Trait("Category", "Budget")]
public sealed class LedgerOccupancyMonitorTests : IDisposable
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
    public void SnapshotCarriesLedgerOccupancy_WithContestedHolders_WhenSharedClaimantsShareATier()
    {
        string path = SyntheticModel("monitor-contended");

        DeviceLedger ledger = new([new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.Kv, Capacity = 8 }]);

        using GenerationService first = OpenWithBus(path, "occupancy-a", kvBlocks: 4, ledger);
        using GenerationService second = OpenWithBus(path, "occupancy-b", kvBlocks: 4, ledger); // both engines' reservations share one tier on the shared ledger

        MetricsSnapshot snapshot = NewestSnapshotAfterLoad(second); // the facts a bus carries describe what had seated when it attached — attach after both engines' claims registered to see the contention

        IReadOnlyList<LedgerOccupancyEntry> occupancy = snapshot.LedgerOccupancy!;
        Assert.NotEmpty(occupancy); // a bus attached to an engine whose claims seated reports them additively

        LedgerOccupancyEntry kvDram = Assert.Single(occupancy, entry => entry.Resource == "kv" && entry.Tier == "dram");
        Assert.Equal("KV block", kvDram.Unit);
        Assert.Equal(8L, kvDram.Capacity!.Value); // the opened seam's declared posture rides along
        Assert.True(kvDram.Contested);           // more than one claimant holds this tier on the shared ledger

        IReadOnlyList<LedgerHolderEntry> holders = kvDram.Holders!;
        Assert.Equal(["occupancy-a", "occupancy-b"], [.. holders.Select(holder => holder.Model)]); // contested holders in registration order
        Assert.Equal(8L, occupancy.Sum(entry => entry.Occupied));                                // four blocks each against the eight-block seam

        using GenerationService solo = OpenWithBus(path, "occupancy-solo", kvBlocks: 2, new DeviceLedger([new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.Kv, Capacity = 8 }]));
        LedgerOccupancyEntry uncontested = Assert.Single(NewestSnapshotAfterLoad(solo).LedgerOccupancy!);
        Assert.False(uncontested.Contested); // a single claimant shares nothing: the holder list stays absent
        Assert.Null(uncontested.Holders);

        first.Dispose();
        second.Dispose();
    }

    [Fact]
    public void Snapshot_WithoutSeatedClaims_AndStreamJson_KeepTheirPreLedgerShape()
    {
        string path = SyntheticModel("monitor-baseline"); // this model declares no budgets: nothing ever seats a claim on its ledger

        using GenerationService engine = OpenWithBus(path, "occupancy-none", kvBlocks: 8);

        MetricsSnapshot snapshot = NewestSnapshotAfterLoad(engine);
        Assert.Null(snapshot.LedgerOccupancy); // the additive slot stays inert when an engine claims nothing

        using JsonDocument document = JsonDocument.Parse(snapshot.ToJson());
        JsonElement root = document.RootElement;

        Assert.True(root.TryGetProperty("metrics_ledger_occupancy", out JsonElement occupancy)); // a trailing optional field, serialized null like its neighbors
        Assert.Equal(JsonValueKind.Null, occupancy.ValueKind);

        // Every operator-view field an existing consumer already reads is still there with the same meaning.
        Assert.Equal("occupancy-none", root.GetProperty("metrics_model").GetString());
        Assert.True(root.TryGetProperty("metrics_kv_used_blocks", out _));
        Assert.True(root.TryGetProperty("metrics_kv_budget_blocks", out _));
        Assert.True(root.TryGetProperty("metrics_admission_refusals_total", out JsonElement refusals));
        Assert.Equal(JsonValueKind.Number, refusals.ValueKind);
    }

    [Fact]
    public void Snapshot_RefreshesLedgerOccupancy_AfterLiveKvCeilingEdit()
    {
        string path = SyntheticModel("monitor-live-kv");
        DeviceLedger ledger = new([new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.Kv, Capacity = 16 }]);

        using GenerationService engine = OpenWithBus(path, "occupancy-live", kvBlocks: 4, ledger);
        MetricsSnapshot before = NewestSnapshotAfterLoad(engine);
        LedgerOccupancyEntry beforeEntry = Assert.Single(before.LedgerOccupancy!, entry => entry.Resource == "kv" && entry.Tier == "dram");
        Assert.Equal(4L, beforeEntry.Occupied);

        LairConfig current = engine.EffectiveConfiguration;
        ModelConfig model = current.Models[0];
        LairConfig candidate = current with
        {
            Models =
            [
                model with
                {
                    KvBlocks = 2,
                    Budgets =
                    [
                        new BudgetChain
                        {
                            Resource = BudgetResource.Kv,
                            Order = [BudgetTier.Dram],
                            Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 2 },
                        },
                    ],
                },
            ],
        };

        IReadOnlyList<FieldResult> results = engine.ApplyConfiguration(candidate);
        FieldResult applied = Assert.Single(results, result => result.Change.Field == "models[0].budgets.kv.caps.dram_blocks");
        Assert.Equal(ConfigApplyOutcome.Applied, applied.Outcome);
        Assert.Contains("KV budget 4 -> 2 blocks", applied.Detail);
        Assert.Equal(2L, Assert.Single(ledger.ClaimsAt(BudgetTier.Dram, BudgetResource.Kv)).Amount); // the standing claim follows the live engine seam without moving to another tier.

        MetricsSnapshot after = SnapshotWithLedgerOccupancy(engine, "kv", "dram", 2L);
        LedgerOccupancyEntry afterEntry = Assert.Single(after.LedgerOccupancy!, entry => entry.Resource == "kv" && entry.Tier == "dram");
        Assert.Equal(2L, afterEntry.Occupied); // the refreshed operator view follows the same standing claim.
    }

    private GenerationService OpenWithBus(string modelPath, string modelName, int kvBlocks, DeviceLedger? ledger = null)
    {
        LairConfig config = new()
        {
            Server = new ServerConfig(),
            Engine = new EngineConfig { Threads = 1 },
            Models =
            [
                new ModelConfig
                {
                    Name = modelName,
                    Path = modelPath,
                    MaxContextTokens = 512,
                    KvBlocks = kvBlocks,
                    Budgets = ledger is not null
                        ? [new BudgetChain { Resource = BudgetResource.Kv, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = kvBlocks } }]
                        : null,
                },
            ],
        };

        GenerationService engine = GenerationService.Open(config, logger: null, ledger);
        engine.AttachMetrics(NullLoggerFactory.Instance, TimeSpan.FromMilliseconds(50)); // the same facts path a serving host wires through ApiApp.Build
        return engine;
    }

    private static MetricsSnapshot NewestSnapshotAfterLoad(GenerationService engine)
    {
        _ = engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = "hello world",
            MaxTokens = 8,
            Temperature = 0f,
        }, CancellationToken.None).GetAwaiter().GetResult();

        EngineMetricsBus bus = engine.Metrics!; // the attached bus samples this engine's counters and carries its model facts
        for (int attempt = 0; attempt < 40; attempt++)
        {
            IReadOnlyList<MetricsSnapshot> ring = bus.SnapshotRing();
            if (ring.Count > 0)
            {
                return ring[0]; // newest first: the closing tick after a completed load publishes even once the engine goes quiet
            }

            Thread.Sleep(50);
        }

        throw new InvalidOperationException("the bus published no snapshots after a completed generation");
    }

    private static MetricsSnapshot SnapshotWithLedgerOccupancy(GenerationService engine, string resource, string tier, long expectedOccupied)
    {
        _ = engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = "hello world",
            MaxTokens = 4,
            Temperature = 0f,
        }, CancellationToken.None).GetAwaiter().GetResult();

        EngineMetricsBus bus = engine.Metrics!;
        for (int attempt = 0; attempt < 40; attempt++)
        {
            foreach (MetricsSnapshot snapshot in bus.SnapshotRing())
            {
                if (snapshot.LedgerOccupancy is { } entries)
                {
                    LedgerOccupancyEntry? entry = entries.FirstOrDefault(candidate => candidate.Resource == resource && candidate.Tier == tier);
                    if (entry?.Occupied == expectedOccupied)
                    {
                        return snapshot;
                    }
                }
            }

            Thread.Sleep(50);
        }

        throw new InvalidOperationException($"the bus published no snapshot carrying {expectedOccupied} {resource} occupancy at tier '{tier}'");
    }

    private string SyntheticModel(string modelName)
    {
        string path = SyntheticLlamaGguf.Write(modelName);
        _tempFiles.Add(path);
        return path;
    }
}
