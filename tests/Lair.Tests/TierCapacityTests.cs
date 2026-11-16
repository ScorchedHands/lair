using Lair.Host.Config;

namespace Lair.Tests;

/// <summary>
/// Declared host capacity and the core-set posture built on it: a top-level host block parses its flat resident totals, its
/// presence-only disk path, and per-node overrides keyed by node position under its numa nesting; malformed or non-positive values fail
/// naming key and value (naming the index too for an override); unknown keys under any of those blocks are refused with the accepted
/// vocabulary, and a block spelled in the old top-level name is refused as an unknown key. Pinned core sets keep today's
/// disjoint-partition refusal naming both models by default, and a model that opts into core overcommit relaxes that rule for its own
/// claim only.
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
    public void Load_HostBlock_ParsesResidentTotalsDiskPathAndNodeOverrides()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            host:
              dram_mb: 65536
              cpu_cores: 32
              disk_path: models
              numa:
                nodes:
                  - index: 0
                    mb: 32768
                    cores: 16
                  - index: 1
                    mb: 32768
        """;

        LairConfig config = YamlConfigLoader.Load(WriteConfig(yaml));

        HostCapacityConfig host = config.Host!;
        Assert.Equal(65536L, host.DramMb);
        Assert.Null(host.VramMb); // absent accelerator posture stays undeclared, not zero
        Assert.Equal(32, host.CpuCores);
        Assert.Equal("models", host.DiskPath);

        Assert.Equal(2, host.Nodes.Count);
        NodeCapacityOverride first = host.Nodes[0];
        Assert.Equal(0, first.Index);
        Assert.Equal(32768L, first.Mb);
        Assert.Equal(16, first.Cores); // an override may state cores alone about a node

        NodeCapacityOverride second = host.Nodes[1];
        Assert.Equal(1, second.Index);
        Assert.Null(second.Cores); // what an entry does not declare stays undeclared rather than zero
    }

    [Fact]
    public void Load_AbsentHostBlock_DeclaresNoPublicTotals()
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            """));

        Assert.Null(config.Host); // today's posture states no capacity, exactly as before this key existed.
    }

    [Fact]
    public void Load_HostBlock_ParsesAcceleratorTotal()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            host:
              vram_mb: 24576
        """;

        LairConfig config = YamlConfigLoader.Load(WriteConfig(yaml));

        HostCapacityConfig host = config.Host!;
        Assert.Null(host.DramMb);
        Assert.Equal(24576L, host.VramMb);
    }

    [Fact]
    public void Load_NonPositiveAcceleratorTotal_FailsNamingKeyAndValue()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            host:
              vram_mb: 0
        """)));

        Assert.Contains("host.vram_mb 0 is invalid", error.Message);
    }

    [Fact]
    public void Load_UnknownKeyUnderHost_FailsNamingTheOffendingKey()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            host:
              ddr_mb: 1024
        """)));

        Assert.Contains("Unknown key 'ddr_mb' under 'host'", error.Message);
        Assert.Contains("dram_mb, vram_mb, cpu_cores, disk_path, numa", error.Message);
    }

    [Fact]
    public void Load_OldTopLevelBlockSpelling_FailsAsAnUnknownKeyNamingTheAcceptedVocabulary()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            tiers:
              dram_cores: 8
        """)));

        // The old spelling is refused outright, not aliased: the refusal names it and lists where capacity belongs now.
        Assert.Contains("Unknown key 'tiers' under", error.Message);
        Assert.Contains("server, engine, models, ui, host", error.Message);
    }

    [Fact]
    public void Load_OldPostureKeySpellings_FailAsUnknownKeysUnderHost()
    {
        ConfigException cores = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            host:
              dram_cores: 8
        """)));
        Assert.Contains("Unknown key 'dram_cores' under 'host'", cores.Message);

        ConfigException bytes = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            host:
              nvme_path: models
        """)));
        Assert.Contains("Unknown key 'nvme_path' under 'host'", bytes.Message);

        ConfigException nodes = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            host:
              dram_mb: 8192
              nodes: []
        """)));
        Assert.Contains("Unknown key 'nodes' under 'host'", nodes.Message); // per-node overrides moved into the numa nesting; the flat spelling is not accepted.
    }

    [Fact]
    public void Load_NonPositiveHostTotal_FailsNamingKeyAndValue()
    {
        ConfigException cores = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            host:
              cpu_cores: 0
        """)));
        Assert.Contains("host.cpu_cores 0 is invalid", cores.Message);

        ConfigException bytes = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            host:
              dram_mb: -5
        """)));
        Assert.Contains("host.dram_mb -5 is invalid", bytes.Message);
    }

    [Fact]
    public void Load_MalformedHostValue_FailsNamingTheKey()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            host:
              dram_mb: lots-of-memory
        """)));

        Assert.Contains("host.dram_mb must be an integer value.", error.Message);
    }

    [Fact]
    public void Load_MalformedDiskPath_FailsNamingTheKeyWithItsPresenceOnlyContract()
    {
        ConfigException blank = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            host:
              disk_path: '   '
        """)));

        Assert.Contains("host.disk_path must name a non-empty path this host reads from", blank.Message);

        ConfigException nested = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            host:
              disk_path:
                drive: D
        """)));

        Assert.Contains("host.disk_path must be a text value.", nested.Message); // presence and path only — no ceiling may ride the posture key.
    }

    [Fact]
    public void Load_MalformedNumaNesting_FailsNamingTheNestAndItsAcceptedKeys()
    {
        ConfigException notMapping = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            host:
              numa: 0
        """)));

        Assert.Contains("'host.numa' must be a mapping of key/value pairs.", notMapping.Message);

        ConfigException unknownNestKey = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            host:
              numa:
                spread: local
        """)));

        Assert.Contains("Unknown key 'spread' under 'host.numa'", unknownNestKey.Message); // placement posture lives on the model entry, not in declared capacity.

        ConfigException nodesNotList = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            host:
              numa:
                nodes: 0
        """)));

        Assert.Contains("'host.numa.nodes' must be a list of per-node overrides", nodesNotList.Message);
    }

    [Fact]
    public void Load_MalformedNodeOverride_FailsNamingKeyValueAndIndex()
    {
        ConfigException bytes = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            host:
              numa:
                nodes:
                  - index: 1
                    mb: 0
        """)));

        Assert.Contains("mb 0 is invalid for node index 1", bytes.Message);

        ConfigException cores = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            host:
              numa:
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
            host:
              numa:
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
            host:
              numa:
                nodes:
                  - index: 0
                    cores: 8
                  - index: 0
                    mb: 4096
        """)));

        Assert.Contains("names node index 0 more than once", error.Message);
    }

    [Fact]
    public void Load_EmptyHostBlock_FailsAsADeclarationOfNothing()
    {
        // The braces spell an empty mapping, which needs a plain string here: an interpolated raw literal would read them as interpolation holes.
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(
            "models:\n  - name: qwen-test\n    path: '" + TempModelFile() + "'\nhost: {}")));

        Assert.Contains("'host' declares nothing", error.Message);
    }

    [Fact]
    public void Load_HostBlockNamingOnlyAnEmptyNumaNesting_FailsAsADeclarationOfNothing()
    {
        // Same brace caution as above: the empty mapping needs a plain string, not an interpolated raw literal.
        ConfigException emptyNest = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(
            "models:\n  - name: qwen-test\n    path: '" + TempModelFile() + "'\nhost:\n  numa: {}")));

        Assert.Contains("'host' declares nothing", emptyNest.Message); // an empty nesting states no override, so the block as a whole still says nothing.

        ConfigException emptyList = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            host:
              numa:
                nodes: []
        """)));

        Assert.Contains("'host' declares nothing", emptyList.Message); // same answer for an explicitly empty override list.
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
        string path = Path.Combine(Path.GetTempPath(), $"lair-host-{Guid.NewGuid():N}.yaml");
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
