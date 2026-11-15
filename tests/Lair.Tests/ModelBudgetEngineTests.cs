using Lair.Host.Config;
using Lair.Http;
using Lair.Models;
using Microsoft.Extensions.Logging;

namespace Lair.Tests;

/// <summary>
/// Engine lanes for budget ceilings: a system-memory KV ceiling applies through the same live cap every other KV edit uses, weight and
/// buffer ceilings report their own postures instead of being applied or dropped quietly, and a ceiling whose chain was never opened at
/// that tier is refused as what opening established. These run against a synthetic model so no fixture gates the pipeline itself.
/// </summary>
[Trait("Category", "HotReload")]
public sealed class ModelBudgetApplyTests : IDisposable
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
    public void ApplyConfiguration_KvCeiling_AppliesThroughTheLiveKvSeam_AndLeavesSiblingChainsAlone()
    {
        string path = SyntheticModel();

        using GenerationService engine = GenerationService.Open(ConfigWith(
            path,
            kvBlocks: 32,
            budgets:
            [
                new BudgetChain { Resource = BudgetResource.Weights, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 1024 } },
                new BudgetChain { Resource = BudgetResource.Kv, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 32 } },
            ]));

        IReadOnlyList<FieldResult> results = engine.ApplyConfiguration(ConfigWith(
            path,
            kvBlocks: 16, // the canonical block count budgets state for this model's system-memory tier
            budgets:
            [
                new BudgetChain { Resource = BudgetResource.Weights, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 1024 } },
                new BudgetChain { Resource = BudgetResource.Kv, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 16 } },
            ]));

        FieldResult applied = Assert.Single(results);
        Assert.Equal("models[0].budgets.kv.caps.dram_blocks", applied.Change.Field);
        Assert.Equal(ConfigApplyOutcome.Applied, applied.Outcome);
        Assert.Contains("KV budget 32 -> 16 blocks within the 32-block physical reservation", applied.Detail);

        ModelConfig effective = engine.EffectiveConfiguration.Models[0];
        Assert.Equal(16, effective.KvBlocks);
        Assert.Equal(16L, ChainOf(effective.Budgets!, BudgetResource.Kv).Caps[BudgetTier.Dram]);

        // The merge carried exactly the ceiling that applied: a sibling chain in the same candidate stayed at what load resolved.
        Assert.Equal(1024L, ChainOf(effective.Budgets!, BudgetResource.Weights).Caps[BudgetTier.Dram]);
    }

    [Fact]
    public void ApplyConfiguration_WeightCeiling_RefusedNamingSlabMigration_EffectiveUnchanged()
    {
        string path = SyntheticModel();

        using GenerationService engine = GenerationService.Open(ConfigWith(
            path,
            kvBlocks: 32,
            budgets: [new BudgetChain { Resource = BudgetResource.Weights, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 1024 } }]));

        IReadOnlyList<FieldResult> results = engine.ApplyConfiguration(ConfigWith(
            path,
            kvBlocks: 32,
            budgets: [new BudgetChain { Resource = BudgetResource.Weights, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 512 } }]));

        FieldResult refused = Assert.Single(results);
        Assert.Equal("models[0].budgets.weights.caps.dram_mb", refused.Change.Field);
        Assert.Equal(ConfigMutationClass.RestartOnly, refused.Change.MutationClass);
        Assert.Equal(ConfigApplyOutcome.Refused, refused.Outcome);
        Assert.Contains("re-reading slabs from the model file", refused.Detail);

        Assert.Equal(1024L, ChainOf(engine.EffectiveConfiguration.Models[0].Budgets!, BudgetResource.Weights).Caps[BudgetTier.Dram]);
    }

    [Fact]
    public void ApplyConfiguration_PrefillBufferCeiling_ClassifiesNewRequests_RefusedBecauseOpenedWindowsStayEstablished()
    {
        string path = SyntheticModel();

        using GenerationService engine = GenerationService.Open(ConfigWith(
            path,
            kvBlocks: 32,
            budgets: [new BudgetChain { Resource = BudgetResource.PrefillBuffers, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 64 } }]));

        IReadOnlyList<FieldResult> results = engine.ApplyConfiguration(ConfigWith(
            path,
            kvBlocks: 32,
            budgets: [new BudgetChain { Resource = BudgetResource.PrefillBuffers, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 128 } }]));

        FieldResult refused = Assert.Single(results);
        Assert.Equal("models[0].budgets.prefill_buffers.caps.dram_mb", refused.Change.Field);
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, refused.Change.MutationClass); // a class is what the field belongs to, not an enforcement promise.
        Assert.Equal(ConfigApplyOutcome.Refused, refused.Outcome);
        Assert.Contains("the physical clamp is where this value lands", refused.Detail); // the ceiling binds when the engine opens its row window; a live edit re-resolves neither it nor opened buffers

        Assert.Equal(64L, ChainOf(engine.EffectiveConfiguration.Models[0].Budgets!, BudgetResource.PrefillBuffers).Caps[BudgetTier.Dram]); // nothing in force changed.
    }

    [Fact]
    public void ApplyConfiguration_CeilingForAChainOpenedWithoutThatTier_RefusedAsEstablishedAtOpen()
    {
        string path = SyntheticModel();

        using GenerationService engine = GenerationService.Open(ConfigWith(path, kvBlocks: 32, budgets: null)); // this model opened with no budget chains at all.

        IReadOnlyList<FieldResult> results = engine.ApplyConfiguration(ConfigWith(
            path,
            kvBlocks: 32,
            budgets: [new BudgetChain { Resource = BudgetResource.Kv, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 16 } }]));

        string[] fields = [.. results.Select(result => result.Change.Field)];
        Assert.Contains("models[0].budgets.kv.order", fields); // establishing a chain is structural: resolved when the engine opened.
        Assert.Contains("models[0].budgets.kv.caps.dram_blocks", fields);

        FieldResult ceiling = Assert.Single(results, result => result.Change.Field == "models[0].budgets.kv.caps.dram_blocks");
        Assert.Equal(ConfigMutationClass.Live, ceiling.Change.MutationClass); // the class stands; this engine simply has no chain holding copies at that tier.
        Assert.Equal(ConfigApplyOutcome.Refused, ceiling.Outcome);
        Assert.Contains("established when the engine opened", ceiling.Detail);

        Assert.Null(engine.EffectiveConfiguration.Models[0].Budgets);
    }

    private string SyntheticModel()
    {
        string path = SyntheticLlamaGguf.Write("budget-apply-model");
        _tempFiles.Add(path);
        return path;
    }

    private static LairConfig ConfigWith(string modelPath, int kvBlocks, IReadOnlyList<BudgetChain>? budgets) => new()
    {
        Server = new ServerConfig(),
        Engine = new EngineConfig(),
        Models =
        [
            new ModelConfig
            {
                Name = "budget-apply-model",
                Path = modelPath,
                MaxContextTokens = 512,
                KvBlocks = kvBlocks,
                Budgets = budgets,
            },
        ],
    };

    private static BudgetChain ChainOf(IReadOnlyList<BudgetChain> chains, BudgetResource resource)
    {
        foreach (BudgetChain chain in chains)
        {
            if (chain.Resource == resource)
            {
                return chain;
            }
        }

        throw new InvalidOperationException($"the configuration in force carries no chain for '{resource}'");
    }
}

/// <summary>Model-open posture of the two spellings of a system-memory KV ceiling: agreement announces once, budgets alone announce nothing.</summary>
[Trait("Category", "Config")]
public sealed class ModelBudgetOpenAnnouncementTests : IDisposable
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
    public void Open_BothSpellingsAgree_LogsOneDeprecationLine_AndSizesStartupAsBefore()
    {
        CapturingLogger logger = new();

        using GenerationService engine = GenerationService.Open(ConfigWith(legacy: true), logger);

        string[] deprecations = [.. logger.Messages(LogLevel.Information).Where(line => line.Contains("deprecated", StringComparison.Ordinal))];
        Assert.Single(deprecations);
        Assert.Contains("budget-announce-model", deprecations[0]);
        Assert.Contains("'budgets.kv.caps.dram_blocks' states the same ceiling", deprecations[0]);

        // One announcement about spelling, not a second enforcement path: startup sizing stayed what today's key sized it to.
        Assert.Equal(32, engine.EffectiveConfiguration.Models[0].KvBlocks);
    }

    [Fact]
    public void Open_BudgetsAlone_AnnounceNothingAboutTheLegacyKey()
    {
        CapturingLogger logger = new();

        using GenerationService engine = GenerationService.Open(ConfigWith(legacy: false), logger);

        Assert.DoesNotContain(logger.Messages(LogLevel.Information), line => line.Contains("deprecated", StringComparison.Ordinal));
        Assert.Equal(32, engine.EffectiveConfiguration.Models[0].KvBlocks); // budgets alone carry the reservation exactly as 'kv_blocks' did.
    }

    private LairConfig ConfigWith(bool legacy)
    {
        string path = SyntheticLlamaGguf.Write("budget-announce-model");
        _tempFiles.Add(path);

        return new LairConfig
        {
            Server = new ServerConfig(),
            Engine = new EngineConfig(),
            Models =
            [
                new ModelConfig
                {
                    Name = "budget-announce-model",
                    Path = path,
                    MaxContextTokens = 512,
                    KvBlocks = 32, // both spellings state this ceiling; the loader refuses any other pairing of them.
                    LegacyKvBlocksDeclared = legacy,
                    Budgets = [new BudgetChain { Resource = BudgetResource.Kv, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 32 } }],
                },
            ],
        };
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
