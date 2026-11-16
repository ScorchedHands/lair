using Lair.Host.Config;

namespace Lair.Tests;

/// <summary>
/// Declared host capacity and the core-set posture built on it: a top-level tiers block parses its flat resident totals, its
/// presence-only disk path, and per-node overrides keyed by node position; malformed or non-positive values fail naming key and value
/// (naming the index too for an override); unknown keys under any of those blocks are refused with the accepted vocabulary. Pinned core
/// sets keep today's disjoint-partition refusal naming both models by default, and a model that opts into core overcommit relaxes that
/// rule for its own claim only.
/// </summary>
[Trait("Category", "Config")]
public sealed class TierCapacityTests : IDisposable
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

        _tempPaths.Clear();
    }

    [Fact]
    public void Load_TiersBlock_ParsesResidentTotalsDiskPathAndNodeOverrides()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            tiers:
              dram_mb: 65536
              dram_cores: 32
              nvme_path: models
              nodes:
                - index: 0
                  mb: 32768
                  cores: 16
                - index: 1
                  mb: 32768
        """;

        LairConfig config = YamlConfigLoader.Load(WriteConfig(yaml));

        TierCapacityConfig tiers = config.Tiers!;
        Assert.Equal(65536L, tiers.DramMb);
        Assert.Null(tiers.VramMb); // absent accelerator posture stays undeclared, not zero
        Assert.Equal(32, tiers.DramCores);
        Assert.Equal("models", tiers.NvmePath);

        Assert.Equal(2, tiers.Nodes.Count);
        NodeCapacityOverride first = tiers.Nodes[0];
        Assert.Equal(0, first.Index);
        Assert.Equal(32768L, first.Mb);
        Assert.Equal(16, first.Cores); // an override may state cores alone about a node

        NodeCapacityOverride second = tiers.Nodes[1];
        Assert.Equal(1, second.Index);
        Assert.Null(second.Cores); // what an entry does not declare stays undeclared rather than zero
    }

    [Fact]
    public void Load_AbsentTiersBlock_DeclaresNoPublicTotals()
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            """));

        Assert.Null(config.Tiers); // today's posture states no capacity, exactly as before this key existed.
    }

    [Fact]
    public void Load_TiersBlock_ParsesAcceleratorTotal()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            tiers:
              vram_mb: 24576
        """;

        LairConfig config = YamlConfigLoader.Load(WriteConfig(yaml));

        TierCapacityConfig tiers = config.Tiers!;
        Assert.Null(tiers.DramMb);
        Assert.Equal(24576L, tiers.VramMb);
    }

    [Fact]
    public void Load_NonPositiveAcceleratorTotal_FailsNamingKeyAndValue()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            tiers:
              vram_mb: 0
        """)));

        Assert.Contains("tiers.vram_mb 0 is invalid", error.Message);
    }

    [Fact]
    public void Load_UnknownKeyUnderTiers_FailsNamingTheOffendingKey()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            tiers:
              ddr_mb: 1024
        """)));

        Assert.Contains("Unknown key 'ddr_mb' under 'tiers'", error.Message);
        Assert.Contains("dram_mb, vram_mb, dram_cores, nvme_path, nodes", error.Message);
    }

    [Fact]
    public void Load_NonPositiveTierTotal_FailsNamingKeyAndValue()
    {
        ConfigException cores = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            tiers:
              dram_cores: 0
        """)));
        Assert.Contains("tiers.dram_cores 0 is invalid", cores.Message);

        ConfigException bytes = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            tiers:
              dram_mb: -5
        """)));
        Assert.Contains("tiers.dram_mb -5 is invalid", bytes.Message);
    }

    [Fact]
    public void Load_MalformedTierValue_FailsNamingTheKey()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            tiers:
              dram_mb: lots-of-memory
        """)));

        Assert.Contains("tiers.dram_mb must be an integer value.", error.Message);
    }

    [Fact]
    public void Load_MalformedNodeOverride_FailsNamingKeyValueAndIndex()
    {
        ConfigException bytes = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            tiers:
              nodes:
                - index: 1
                  mb: 0
        """)));

        Assert.Contains("mb 0 is invalid for node index 1", bytes.Message);

        ConfigException cores = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            tiers:
              nodes:
                - index: 2
                  cores: not-a-number
        """)));

        Assert.Contains("cores must be an integer value.", cores.Message);
    }

    [Fact]
    public void Load_NodeOverrideWithoutIndex_FailsNamingTheEntry()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            tiers:
              nodes:
                - mb: 4096
        """)));

        Assert.Contains("Missing required key 'index' under", error.Message);
    }

    [Fact]
    public void Load_NodeOverridesNamingTheSameIndex_Fails()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            tiers:
              nodes:
                - index: 0
                  cores: 8
                - index: 0
                  mb: 4096
        """)));

        Assert.Contains("names node index 0 more than once", error.Message);
    }

    [Fact]
    public void Load_EmptyTiersBlock_FailsAsADeclarationOfNothing()
    {
        // The braces spell an empty mapping, which needs a plain string here: an interpolated raw literal would read them as interpolation holes.
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(
            "models:\n  - name: qwen-test\n    path: '" + TempModelFile() + "'\ntiers: {}")));

        Assert.Contains("'tiers' declares nothing", error.Message);
    }

    [Fact]
    public void Load_CoreOvercommitPlacementKey_ParsesBothPosturesAndRefusesANonBoolean()
    {
        LairConfig optedIn = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                placement:
                  core_set: "0-3"
                  core_overcommit: true
            """));

        Assert.True(optedIn.Models[0].Placement.CoreOvercommit);

        LairConfig defaultPosture = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                placement:
                  core_set: "0-3"
            """));

        Assert.False(defaultPosture.Models[0].Placement.CoreOvercommit); // absent keeps today's disjoint posture.

        ConfigException malformed = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                placement:
                  core_overcommit: sometimes
            """)));

        Assert.Contains("models[0].placement.core_overcommit must be a true or false value.", malformed.Message);
    }

    [Fact]
    public void ValidateCoreSetsDisjoint_OverlappingModelsStillNameBothModels_ByDefault()
    {
        List<ModelConfig> models =
        [
            new ModelConfig { Name = "first", Path = "a.gguf", Placement = new PlacementConfig { CoreSet = new HashSet<int>([0, 1, 2]) } },
            new ModelConfig { Name = "second", Path = "b.gguf", Placement = new PlacementConfig { CoreSet = new HashSet<int>([2, 3]) } },
        ];

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.ValidateCoreSetsDisjoint(models));

        Assert.Contains("both pin core 2", error.Message);
        Assert.Contains("'first'", error.Message);
        Assert.Contains("'second'", error.Message);
    }

    [Fact]
    public void ValidateCoreSetsDisjoint_OptedInClaimMayShareHeldCores_WhileOtherPairsStillRefuse()
    {
        List<ModelConfig> sharing =
        [
            new ModelConfig { Name = "first", Path = "a.gguf", Placement = new PlacementConfig { CoreSet = new HashSet<int>([0, 1, 2]) } },
            new ModelConfig { Name = "second", Path = "b.gguf", Placement = new PlacementConfig { CoreSet = new HashSet<int>([2, 3]), CoreOvercommit = true } },
        ];

        YamlConfigLoader.ValidateCoreSetsDisjoint(sharing); // the opted-in claim shares core 2 rather than being refused.

        List<ModelConfig> thirdModelWithoutOptIn =
        [
            .. sharing,
            new ModelConfig { Name = "third", Path = "c.gguf", Placement = new PlacementConfig { CoreSet = new HashSet<int>([3]) } },
        ];

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.ValidateCoreSetsDisjoint(thirdModelWithoutOptIn));

        Assert.Contains("both pin core 3", error.Message); // the relaxation belongs to the claim that opted in, not to every pair.
    }

    private string WriteConfig(string yaml)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-tiers-{Guid.NewGuid():N}.yaml");
        File.WriteAllText(path, yaml);
        _tempPaths.Add(path);
        return path;
    }

    private string TempModelFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-model-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, [0x47, 0x46]); // only the key shape is under test here: a stub file satisfies presence.
        _tempPaths.Add(path);
        return path;
    }
}
