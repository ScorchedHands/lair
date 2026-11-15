using Lair.Host.Config;
using Lair.Models;

namespace Lair.Tests;

[Trait("Category", "HotReload")]
public sealed class ModelBudgetHotReloadTests : IDisposable
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
                // Best-effort cleanup of temp fixtures.
            }
        }
    }

    [Fact]
    public void Classify_BudgetCeilingsCarryTheirOwnClasses()
    {
        Assert.Equal(ConfigMutationClass.Live, ConfigMutationTable.Classify("models[0].budgets.kv.caps.dram_blocks"));
        Assert.Equal(ConfigMutationClass.RestartOnly, ConfigMutationTable.Classify("models[0].budgets.weights.caps.dram_mb"));
        Assert.Equal(ConfigMutationClass.RestartOnly, ConfigMutationTable.Classify("models[0].budgets.weights.caps.vram_mb"));
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, ConfigMutationTable.Classify("models[0].budgets.prefill_buffers.caps.dram_mb"));

        // A ceiling beyond the tier this build actually holds copies at retargets nothing live: it classifies with its resource's structural posture.
        Assert.Equal(ConfigMutationClass.RestartOnly, ConfigMutationTable.Classify("models[0].budgets.kv.caps.vram_blocks"));

        // An order names structure resolved when the engine opened, and a key this table does not recognize never falls through unclassified.
        Assert.Equal(ConfigMutationClass.RestartOnly, ConfigMutationTable.Classify("models[0].budgets.kv.order"));
        Assert.Equal(ConfigMutationClass.RestartOnly, ConfigMutationTable.Classify("models[0].budgets.kv.caps.blocks"));
    }

    [Fact]
    public void Diff_BudgetedCeilingChange_ReportsTheBudgetFieldNotTheLegacyKey()
    {
        LairConfig current = ConfigWith(new ModelConfig
        {
            Name = "qwen-test",
            Path = "a.gguf",
            KvBlocks = 256,
            Budgets = [new BudgetChain { Resource = BudgetResource.Kv, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 256 } }],
        });

        LairConfig candidate = ConfigWith(new ModelConfig
        {
            Name = "qwen-test",
            Path = "a.gguf",
            KvBlocks = 384, // the loader canonizes a budgeted ceiling onto this count; budgets alone own what the reservation and live cap hold.
            Budgets = [new BudgetChain { Resource = BudgetResource.Kv, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 384 } }],
        });

        IReadOnlyList<FieldChange> changes = ConfigMutationTable.Diff(current, candidate);

        FieldChange ceiling = Assert.Single(changes, change => change.Field == "models[0].budgets.kv.caps.dram_blocks");
        Assert.Equal("256", ceiling.OldValue);
        Assert.Equal("384", ceiling.NewValue);
        Assert.Equal(ConfigMutationClass.Live, ceiling.MutationClass);

        // One edited key reports once: with no legacy spelling in either configuration, the canonical block count carries no separate entry.
        Assert.DoesNotContain(changes, change => change.Field == "models[0].kv_blocks");
    }

    [Fact]
    public void Diff_LegacyCeilingChange_StillReportsTheLegacyKey()
    {
        LairConfig current = ConfigWith(new ModelConfig { Name = "qwen-test", Path = "a.gguf", KvBlocks = 256, LegacyKvBlocksDeclared = true });
        LairConfig candidate = ConfigWith(new ModelConfig { Name = "qwen-test", Path = "a.gguf", KvBlocks = 320, LegacyKvBlocksDeclared = true });

        IReadOnlyList<FieldChange> changes = ConfigMutationTable.Diff(current, candidate);

        FieldChange ceiling = Assert.Single(changes, change => change.Field == "models[0].kv_blocks");
        Assert.Equal(ConfigMutationClass.Live, ceiling.MutationClass);
    }

    [Fact]
    public void Diff_WeightAndBufferCeilings_ClassifyWithTheirOwnPostures()
    {
        LairConfig current = ConfigWith(new ModelConfig
        {
            Name = "qwen-test",
            Path = "a.gguf",
            Budgets = [new BudgetChain { Resource = BudgetResource.Weights, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 1024 } }],
        });

        LairConfig candidate = ConfigWith(new ModelConfig
        {
            Name = "qwen-test",
            Path = "a.gguf",
            Budgets =
            [
                new BudgetChain { Resource = BudgetResource.Weights, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 2048 } },
                new BudgetChain { Resource = BudgetResource.PrefillBuffers, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 64 } },
            ],
        });

        IReadOnlyList<FieldChange> changes = ConfigMutationTable.Diff(current, candidate);

        FieldChange weights = Assert.Single(changes, change => change.Field == "models[0].budgets.weights.caps.dram_mb");
        Assert.Equal("1024", weights.OldValue);
        Assert.Equal("2048", weights.NewValue);
        Assert.Equal(ConfigMutationClass.RestartOnly, weights.MutationClass);

        FieldChange buffers = Assert.Single(changes, change => change.Field == "models[0].budgets.prefill_buffers.caps.dram_blocks" || change.Field == "models[0].budgets.prefill_buffers.caps.dram_mb");
        Assert.Equal("(unset)", buffers.OldValue); // a ceiling nothing declared before reads back as absent, not as some default.
        Assert.Equal("64", buffers.NewValue);
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, buffers.MutationClass);

        FieldChange chain = Assert.Single(changes, change => change.Field == "models[0].budgets.prefill_buffers.order");
        Assert.Equal("(unset)", chain.OldValue);
        Assert.Equal("dram", chain.NewValue);
        Assert.Equal(ConfigMutationClass.RestartOnly, chain.MutationClass); // establishing a chain is structural: resolved when the engine opened.
    }

    [Fact]
    public void Diff_NoBudgetsOnEitherSide_EmitsNoBudgetFields()
    {
        LairConfig current = ConfigWith(new ModelConfig { Name = "qwen-test", Path = "a.gguf" });
        LairConfig candidate = ConfigWith(new ModelConfig { Name = "qwen-test", Path = "a.gguf", KvBlocks = 512, LegacyKvBlocksDeclared = true });

        IReadOnlyList<FieldChange> changes = ConfigMutationTable.Diff(current, candidate);

        Assert.DoesNotContain(changes, change => change.Field.StartsWith("models[0].budgets.", StringComparison.Ordinal));
    }

    [Fact]
    public void ExtractEdits_RejectedCandidate_StillNamesEachBudgetFieldAndClass()
    {
        LairConfig current = ConfigWith(new ModelConfig
        {
            Name = "qwen-test",
            Path = "a.gguf",
            KvBlocks = 512,
            Budgets =
            [
                new BudgetChain { Resource = BudgetResource.Kv, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 512 } },
                new BudgetChain { Resource = BudgetResource.Weights, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 1024 } },
            ],
        });

        // This candidate validates nowhere — a KV ceiling written as text and a weight ceiling below nothing's floor — yet each edited field still names itself with its class in the reload summary.
        string path = WriteFile("""
            models:
              - name: qwen-test
                path: 'a.gguf'
                budgets:
                  kv:
                    order: [dram]
                    caps: { dram_blocks: notanumber }
                  weights:
                    order: [dram]
                    caps: { dram_mb: 4096 }

            """);

        IReadOnlyList<FieldChange> edits = ConfigMutationTable.ExtractEdits(current, path);

        FieldChange kvEdit = Assert.Single(edits, change => change.Field == "models[0].budgets.kv.caps.dram_blocks");
        Assert.Equal("512", kvEdit.OldValue);
        Assert.Equal("notanumber", kvEdit.NewValue);
        Assert.Equal(ConfigMutationClass.Live, kvEdit.MutationClass);

        FieldChange weightsEdit = Assert.Single(edits, change => change.Field == "models[0].budgets.weights.caps.dram_mb");
        Assert.Equal("1024", weightsEdit.OldValue);
        Assert.Equal("4096", weightsEdit.NewValue);
        Assert.Equal(ConfigMutationClass.RestartOnly, weightsEdit.MutationClass);

        // Chain order lists carry no single canonical text to compare against raw scalars — semantic chain edits surface through the full diff of a file that validates.
        Assert.DoesNotContain(edits, change => change.Field.EndsWith(".order", StringComparison.Ordinal));
    }

    private static LairConfig ConfigWith(ModelConfig model) => new()
    {
        Server = new ServerConfig(),
        Engine = new EngineConfig(),
        Models = [model],
        Ui = new UiConfig(),
    };

    private string WriteFile(string contents)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-budget-reload-{Guid.NewGuid():N}.yaml");
        File.WriteAllText(path, contents);
        _tempPaths.Add(path);
        return path;
    }
}
