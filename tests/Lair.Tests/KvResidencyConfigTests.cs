using Lair.Host.Config;
using Lair.Models;

namespace Lair.Tests;

/// <summary>
/// The KV residency placement vocabulary and the block ceiling that bounds an accelerator-resident reservation: every
/// declared residency mode parses without consulting what any opened backend can back, an absent key keeps today's
/// host-resident posture and sizing exactly as they were, a value outside the declared set fails naming the served model,
/// the key, and the accepted set, and the accelerator-tier KV ceiling is a finite block count bound to the chain tier it
/// names — declared posture at load, never a claim a loader resolves against a pool.
/// </summary>
[Trait("Category", "Config")]
public sealed class KvResidencyConfigTests : IDisposable
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

    [Theory]
    [InlineData("auto")]
    [InlineData("dram")]
    [InlineData("vram")]
    [InlineData("split")]
    public void Load_KvResidency_AcceptsEveryDeclaredMode(string residency)
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                placement:
                  kv_residency: {residency}

            """));

        ModelConfig model = Assert.Single(config.Models);
        Assert.Equal(residency, model.Placement.KvResidency);

        // Accepting the vocabulary sizes nothing and opens nothing: the reservation stays what today's keys size, and the
        // posture that can hold it is what route resolution answers when the engine opens.
        Assert.Equal(256, model.KvBlocks);
        Assert.Null(model.Budgets);
    }

    [Fact]
    public void Load_KvResidencyAbsent_KeepsHostResidency()
    {
        LairConfig declared = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                placement:
                  device: cpu

            """));
        LairConfig absent = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                placement:
                  device: cpu
                  kv_residency: dram

            """));

        Assert.Equal("dram", declared.Models[0].Placement.KvResidency);
        Assert.Equal("dram", absent.Models[0].Placement.KvResidency);
    }

    [Theory]
    [InlineData("gpu")]
    [InlineData("cuda")]
    [InlineData("host")]
    [InlineData("DRAM")]
    public void Load_KvResidencyOutsideTheDeclaredSet_FailsNamingModelKeyAndAcceptedSet(string residency)
    {
        ConfigException error = Assert.Throws<ConfigException>(
            () => YamlConfigLoader.Load(WriteConfig($"""
                models:
                  - name: qwen-test
                    path: '{TempModelFile()}'
                    placement:
                      kv_residency: {residency}

                """)));

        Assert.Contains($"models[0].placement.kv_residency '{residency}' is invalid for model 'qwen-test'", error.Message);
        Assert.Contains("'kv_residency' accepts only 'auto', 'dram', 'vram', 'split'", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_VramKvCapacity_ParsesFiniteBlockCeilingBoundToItsChainTier()
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                placement:
                  kv_residency: split
                budgets:
                  kv:
                    order: [dram, vram]
                    caps:
                      dram_blocks: 512
                      vram_blocks: 64

            """));

        ModelConfig model = Assert.Single(config.Models);
        BudgetChain kv = ChainOf(model.Budgets!, BudgetResource.Kv);

        Assert.Equal([BudgetTier.Dram, BudgetTier.Vram], kv.Order);
        Assert.Equal(512L, kv.Caps[BudgetTier.Dram]);
        Assert.Equal(64L, kv.Caps[BudgetTier.Vram]); // a finite block count, the unit this resource counts ceilings in
        Assert.Equal(512, model.KvBlocks); // the host ceiling is what sizes the reservation; the accelerator ceiling bounds blocks, not the arena
        Assert.Equal("split", model.Placement.KvResidency);
    }

    [Fact]
    public void Load_VramKvCapacityAlone_LeavesTheHostReservationAtTodaysSizing()
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                budgets:
                  kv:
                    order: [dram, vram]
                    caps:
                      vram_blocks: 128

            """));

        ModelConfig model = Assert.Single(config.Models);
        BudgetChain kv = ChainOf(model.Budgets!, BudgetResource.Kv);

        Assert.Equal(128L, kv.Caps[BudgetTier.Vram]);
        Assert.False(kv.Caps.ContainsKey(BudgetTier.Dram)); // an uncapped position states no ceiling rather than some default one
        Assert.Equal(256, model.KvBlocks); // no host ceiling was declared, so today's sizing stands
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-3)]
    public void Load_VramKvCapacityNotFinite_FailsNamingPathAndValue(int ceiling)
    {
        ConfigException error = Assert.Throws<ConfigException>(
            () => YamlConfigLoader.Load(WriteConfig($"""
                models:
                  - name: qwen-test
                    path: '{TempModelFile()}'
                    budgets:
                      kv:
                        order: [dram, vram]
                        caps:
                          vram_blocks: {ceiling}

                """)));

        Assert.Contains($"models[0].budgets.kv.caps.vram_blocks {ceiling}", error.Message);
        Assert.Contains("must be 1 or greater", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_VramKvCapacityWithoutItsChainTier_FailsPointingAtOrder()
    {
        ConfigException error = Assert.Throws<ConfigException>(
            () => YamlConfigLoader.Load(WriteConfig($"""
                models:
                  - name: qwen-test
                    path: '{TempModelFile()}'
                    budgets:
                      kv:
                        order: [dram]
                        caps:
                          vram_blocks: 64

                """)));

        Assert.Contains("models[0].budgets.kv.caps.vram_blocks", error.Message);
        Assert.Contains("declare 'vram' there to cap it", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_VramKvCapacityNonInteger_FailsNamingPath()
    {
        ConfigException error = Assert.Throws<ConfigException>(
            () => YamlConfigLoader.Load(WriteConfig($"""
                models:
                  - name: qwen-test
                    path: '{TempModelFile()}'
                    budgets:
                      kv:
                        order: [dram, vram]
                        caps:
                          vram_blocks: plenty

                """)));

        Assert.Contains("models[0].budgets.kv.caps.vram_blocks 'plenty' is invalid for model 'qwen-test'", error.Message);
        Assert.Contains("'vram_blocks' must state an integer ceiling.", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_AcceleratorKvCeilingInWrongUnit_FailsNamingTheKvCeilingVocabulary()
    {
        ConfigException error = Assert.Throws<ConfigException>(
            () => YamlConfigLoader.Load(WriteConfig($"""
                models:
                  - name: qwen-test
                    path: '{TempModelFile()}'
                    budgets:
                      kv:
                        order: [dram, vram]
                        caps:
                          vram_mb: 512

                """)));

        Assert.Contains("models[0].budgets.kv.caps.vram_mb", error.Message);
        Assert.Contains("'kv' ceilings are named 'dram_blocks', 'vram_blocks', 'disk_blocks'", error.Message, StringComparison.Ordinal);
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
        string path = Path.Combine(Path.GetTempPath(), $"lair-kv-residency-config-{Guid.NewGuid():N}.yaml");
        File.WriteAllText(path, yaml);
        _tempPaths.Add(path);
        return path;
    }

    private string TempModelFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-kv-residency-model-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, [0x47, 0x47, 0x55, 0x46]);
        _tempPaths.Add(path);
        return path;
    }
}
