using Lair.Host.Config;
using Lair.Models;

namespace Lair.Tests;

[Trait("Category", "Config")]
public sealed class ModelBudgetConfigTests : IDisposable
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
    public void Load_BudgetChains_ParseDeclaredOrdersAndCeilings()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                budgets:
                  weights:
                    order: [dram, disk]
                    caps:
                      dram_mb: 2048
                  kv:
                    order: [dram]
                    caps:
                      dram_blocks: 512
                  prefill_buffers:
                    order: [dram]

            """;

        LairConfig config = YamlConfigLoader.Load(WriteConfig(yaml));
        ModelConfig model = Assert.Single(config.Models);

        Assert.NotNull(model.Budgets);
        Assert.Equal(3, model.Budgets.Count);

        BudgetChain weights = ChainOf(model.Budgets, BudgetResource.Weights);
        Assert.Equal([BudgetTier.Dram, BudgetTier.Disk], weights.Order);
        Assert.Equal(2048L, weights.Caps[BudgetTier.Dram]);
        Assert.False(weights.Caps.ContainsKey(BudgetTier.Disk)); // an uncapped position states no ceiling rather than some default one.

        BudgetChain kv = ChainOf(model.Budgets, BudgetResource.Kv);
        Assert.Equal([BudgetTier.Dram], kv.Order);
        Assert.Equal(512L, kv.Caps[BudgetTier.Dram]);

        BudgetChain buffers = ChainOf(model.Budgets, BudgetResource.PrefillBuffers);
        Assert.Equal([BudgetTier.Dram], buffers.Order);
        Assert.Empty(buffers.Caps);

        // The declared system-memory KV ceiling is what sizes the reservation here: budgets alone carry it exactly as 'kv_blocks' did.
        Assert.Equal(512, model.KvBlocks);
        Assert.False(model.LegacyKvBlocksDeclared);
    }

    [Fact]
    public void Load_BothSpellingsAgree_AcceptsAndRecordsLegacyPresence()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                kv_blocks: 512
                budgets:
                  kv:
                    order: [dram]
                    caps:
                      dram_blocks: 512

            """;

        LairConfig config = YamlConfigLoader.Load(WriteConfig(yaml));
        ModelConfig model = Assert.Single(config.Models);

        Assert.Equal(512, model.KvBlocks); // startup sizing stays exactly what it was today.
        Assert.True(model.LegacyKvBlocksDeclared);
    }

    [Fact]
    public void Load_BothSpellingsDisagree_FailsNamingModelAndEachPathValue()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                kv_blocks: 640
                budgets:
                  kv:
                    order: [dram]
                    caps:
                      dram_blocks: 512

            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("model 'qwen-test'", error.Message);
        Assert.Contains("models[0].kv_blocks 640", error.Message);
        Assert.Contains("models[0].budgets.kv.caps.dram_blocks 512", error.Message);
    }

    [Fact]
    public void Load_LegacyKeyAlone_BehavesExactlyAsBefore()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                kv_blocks: 320
            """;

        LairConfig config = YamlConfigLoader.Load(WriteConfig(yaml));
        ModelConfig model = Assert.Single(config.Models);

        Assert.Equal(320, model.KvBlocks);
        Assert.True(model.LegacyKvBlocksDeclared);
        Assert.Null(model.Budgets); // no budgets block declares nothing: today's behavior is the empty posture.
    }

    [Fact]
    public void Load_BudgetedCeilingBelowLegacyFloor_FailsWithTheSameFloorAsKvBlocks()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                budgets:
                  kv:
                    order: [dram]
                    caps:
                      dram_blocks: 4

            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("models[0].budgets.kv.caps.dram_blocks 4", error.Message);
        Assert.Contains("8 or greater", error.Message);
    }

    [Fact]
    public void Load_UnknownResourceKey_FailsNamingPathAndAcceptedResources()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                budgets:
                  layers:
                    order: [dram]

            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("Unknown key 'layers' under models[0].budgets", error.Message);
        Assert.Contains("weights, kv, prefill_buffers, ple", error.Message);
    }

    [Fact]
    public void Load_CapKeyInWrongUnit_FailsNamingResourceVocabulary()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                budgets:
                  weights:
                    order: [dram]
                    caps:
                      dram_blocks: 256

            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("models[0].budgets.weights.caps.dram_blocks", error.Message);
        Assert.Contains("'weights' ceilings are named 'dram_mb', 'vram_mb', 'disk_mb'", error.Message);
    }

    [Fact]
    public void Load_UnknownTierInOrder_FailsNamingValueAndAcceptedTiers()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                budgets:
                  kv:
                    order: [ssd]

            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("models[0].budgets.kv.order 'ssd' is invalid for model 'qwen-test'", error.Message);
        Assert.Contains("'dram', 'vram', 'disk'", error.Message);
    }

    [Fact]
    public void Load_OldTierNameInOrder_FailsNamingValueAndAcceptedTiers()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                budgets:
                  kv:
                    order: [dram, nvme]

            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("models[0].budgets.kv.order 'nvme' is invalid for model 'qwen-test'", error.Message); // the retired spelling names no tier: it fails as a value, not as an alias.
        Assert.Contains("'dram', 'vram', 'disk'", error.Message);
    }

    [Fact]
    public void Load_OutOfOrderChain_FailsNamingTheOffendingChain()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                budgets:
                  weights:
                    order: [disk, dram]

            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("models[0].budgets.weights.order 'disk', 'dram' is out of order for model 'qwen-test'", error.Message);
    }

    [Fact]
    public void Load_RepeatedTierInOrder_Fails()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                budgets:
                  kv:
                    order: [dram, vram, dram]

            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("names each tier exactly once", error.Message);
    }

    [Fact]
    public void Load_EmptyOrder_FailsNamingResourceToOmit()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                budgets:
                  prefill_buffers:
                    order: []

            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("models[0].budgets.prefill_buffers.order is empty for model 'qwen-test'", error.Message);
        Assert.Contains("omit 'prefill_buffers' entirely", error.Message);
    }

    [Fact]
    public void Load_OrderKeyMissing_FailsNamingSubmapping()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                budgets:
                  kv:
                    caps:
                      dram_blocks: 256

            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("Missing required key 'order' under models[0].budgets.kv", error.Message);
    }

    [Fact]
    public void Load_NonPositiveCeiling_FailsNamingPathAndValue()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                budgets:
                  prefill_buffers:
                    order: [dram]
                    caps:
                      dram_mb: 0

            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("models[0].budgets.prefill_buffers.caps.dram_mb 0", error.Message);
        Assert.Contains("must be 1 or greater", error.Message);
    }

    [Fact]
    public void Load_CeilingBindsTierChainDoesNotName_FailsPointingAtOrder()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                budgets:
                  weights:
                    order: [dram]
                    caps:
                      vram_mb: 512

            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("models[0].budgets.weights.caps.vram_mb", error.Message);
        Assert.Contains("declare 'vram' there to cap it", error.Message);
    }

    [Fact]
    public void Load_WeightCeilingAtOriginStore_RefusedAsNotYetMeaningful()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                budgets:
                  weights:
                    order: [dram, disk]
                    caps:
                      dram_mb: 1024
                      disk_mb: 2048

            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("models[0].budgets.weights.caps.disk_mb", error.Message);
        Assert.Contains("no model-level ceiling describes that position", error.Message); // the byte resource already sits whole on its origin store.
    }

    [Fact]
    public void Load_PleChainParsesItsOwnOrderAndCeilings()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                budgets:
                  weights:
                    order: [dram, disk]
                    caps:
                      dram_mb: 2048
                  ple:
                    order: [dram, vram, disk]
                    caps:
                      dram_mb: 4096
                      vram_mb: 512

            """;

        LairConfig config = YamlConfigLoader.Load(WriteConfig(yaml));
        ModelConfig model = Assert.Single(config.Models);

        Assert.NotNull(model.Budgets);

        BudgetChain ple = ChainOf(model.Budgets, BudgetResource.Ple);
        Assert.Equal([BudgetTier.Dram, BudgetTier.Vram, BudgetTier.Disk], ple.Order);
        Assert.Equal(512L, ple.Caps[BudgetTier.Vram]);
        Assert.Equal(4096L, ple.Caps[BudgetTier.Dram]);
        Assert.False(ple.Caps.ContainsKey(BudgetTier.Disk)); // the model file already holds the whole table there, so only working-copy tiers carry a ceiling of ours.

        // The PLE chain declares itself beside the weight chain rather than riding on it: each resource keeps its own order and ceilings.
        BudgetChain weights = ChainOf(model.Budgets, BudgetResource.Weights);
        Assert.Equal([BudgetTier.Dram, BudgetTier.Disk], weights.Order);
        Assert.Equal(2048L, weights.Caps[BudgetTier.Dram]);
    }

    [Fact]
    public void Load_PleDiskCap_FailsNamingQualifiedKeyAndModel()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                budgets:
                  ple:
                    order: [dram, disk]
                    caps:
                        disk_mb: 2048

            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("models[0].budgets.ple.caps.disk_mb", error.Message);
        Assert.Contains("no model-level ceiling describes that position", error.Message); // the gathered rows read their complete table from disk storage already, exactly as weights do.
    }

    [Fact]
    public void Load_OlderCapKeySpellings_FailNamingPathAndAcceptedCeilings()
    {
        ConfigException weights = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                budgets:
                  weights:
                    order: [dram, disk]
                    caps:
                      nvme_mb: 2048

            """)));
        Assert.Contains("models[0].budgets.weights.caps.nvme_mb", weights.Message); // the retired key spelling is not rewritten to the new one.
        Assert.Contains("'weights' ceilings are named 'dram_mb', 'vram_mb', 'disk_mb'", weights.Message);

        ConfigException kv = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                budgets:
                  kv:
                    order: [dram]
                    caps:
                      nvme_blocks: 256

            """)));
        Assert.Contains("models[0].budgets.kv.caps.nvme_blocks", kv.Message);
        Assert.Contains("'kv' ceilings are named 'dram_blocks', 'vram_blocks', 'disk_blocks'", kv.Message);
    }

    [Fact]
    public void Load_FutureShapedChain_AcceptsStructurallyWhereNothingNeedsItYet()
    {
        // A declared accelerator tier is vocabulary before anything resident sits there — the posture refusal belongs to resolution against opened pools, not to parsing a declaration.
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                budgets:
                  kv:
                    order: [dram, vram]

            """;

        LairConfig config = YamlConfigLoader.Load(WriteConfig(yaml));
        ModelConfig model = Assert.Single(config.Models);

        BudgetChain kv = ChainOf(model.Budgets!, BudgetResource.Kv);
        Assert.Equal([BudgetTier.Dram, BudgetTier.Vram], kv.Order);
    }

    [Fact]
    public void Load_DiskKvChain_AcceptsStructurallyWhereNoWritingExistsYet()
    {
        // The write-side disk tier is chain vocabulary before anything can land there — the posture refusal belongs to resolution against opened pools, not to parsing a declaration.
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                budgets:
                  kv:
                    order: [dram, disk]

            """;

        LairConfig config = YamlConfigLoader.Load(WriteConfig(yaml));
        ModelConfig model = Assert.Single(config.Models);

        BudgetChain kv = ChainOf(model.Budgets!, BudgetResource.Kv);
        Assert.Equal([BudgetTier.Dram, BudgetTier.Disk], kv.Order);
    }

    private static BudgetChain ChainOf(IReadOnlyList<BudgetChain> chains, BudgetResource resource)
    {
        foreach (BudgetChain chain in chains)
        {
            if (chain.Resource == resource)
            {
                return chain;
            }
        }

        throw new InvalidOperationException($"the configuration declared no chain for '{resource}'");
    }

    private string WriteConfig(string yaml)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-budget-config-{Guid.NewGuid():N}.yaml");
        File.WriteAllText(path, yaml);
        _tempPaths.Add(path);
        return path;
    }

    private string TempModelFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-budget-model-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, [0x47, 0x47, 0x55, 0x46]);
        _tempPaths.Add(path);
        return path;
    }
}
