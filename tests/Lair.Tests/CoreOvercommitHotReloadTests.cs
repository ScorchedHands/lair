using Lair.Host.Config;
using Lair.Http;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// The keys that bind what a running engine contends for: core overcommit relaxes the disjoint core-set rule only when the engine opens,
/// and declared host totals are compared against the posture detection reported at open — so both classify as restart-only in the reload
/// table, name old against new in the diff (per-node overrides through one canonical field, like every other structural list), and refuse a
/// live attempt with their mutation class while the engine keeps serving what it opened with. The raw-scalar extraction that names changed
/// keys when a candidate fails validation covers both plain scalars.
/// </summary>
[Trait("Category", "HotReload")]
[Trait("Category", "Config")]
public sealed class CoreOvercommitHotReloadTests : IDisposable
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
    public void CoreOvercommit_AndHostTotalEdits_ClassifyAsRestartOnly_NamingOldAgainstNew()
    {
        string modelPath = TempModelFile();

        LairConfig current = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
                placement:
                  core_set: "0-1"
                  core_overcommit: false
            host:
              cpu_cores: 8
        """));

        LairConfig candidate = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
                placement:
                  core_set: "0-1"
                  core_overcommit: true
            host:
              cpu_cores: 16
        """));

        IReadOnlyList<FieldChange> changes = ConfigMutationTable.Diff(current, candidate);

        FieldChange overcommit = Assert.Single(changes, change => change.Field == "models[0].placement.core_overcommit");
        Assert.Equal(ConfigMutationClass.RestartOnly, overcommit.MutationClass); // relaxing a disjoint partition is established when the engine opens.
        Assert.Equal("false", overcommit.OldValue);
        Assert.Equal("true", overcommit.NewValue); // the loader's own posture text, in the same form raw extraction produces.

        FieldChange cores = Assert.Single(changes, change => change.Field == "host.cpu_cores");
        Assert.Equal(ConfigMutationClass.RestartOnly, cores.MutationClass); // declared totals bind accounting at open.
        Assert.Equal("8", cores.OldValue);
        Assert.Equal("16", cores.NewValue);

        Assert.Equal(ConfigMutationClass.RestartOnly, ConfigMutationTable.Classify("host.dram_mb"));
        Assert.Equal(ConfigMutationClass.RestartOnly, ConfigMutationTable.Classify("host.vram_mb"));
        Assert.Equal(ConfigMutationClass.RestartOnly, ConfigMutationTable.Classify("host.cpu_cores"));
        Assert.Equal(ConfigMutationClass.RestartOnly, ConfigMutationTable.Classify("host.disk_path"));
    }

    [Fact]
    public void NodeOverrideEdits_ReportThroughOneCanonicalField_AndUnchangedPosturesReportNoHostChanges()
    {
        string modelPath = TempModelFile();

        LairConfig current = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
            host:
              dram_mb: 8192
              numa:
                nodes:
                  - index: 0
                    mb: 4096
        """));

        LairConfig candidate = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
            host:
              dram_mb: 8192
              numa:
                nodes:
                  - index: 0
                    mb: 4096
                    cores: 4
        """));

        FieldChange nodes = Assert.Single(ConfigMutationTable.Diff(current, candidate)); // the flat totals stand; only the override posture moved.
        Assert.Equal("host.numa.nodes", nodes.Field);
        Assert.Equal(ConfigMutationClass.RestartOnly, nodes.MutationClass);
        Assert.Contains("index=0:mb=4096:cores=unset", nodes.OldValue);
        Assert.Contains("index=0:mb=4096:cores=4", nodes.NewValue);

        LairConfig samePosture = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
            host:
              dram_mb: 8192
              numa:
                nodes:
                  - index: 0
                    cores: 4
                    mb: 4096
        """));

        Assert.Empty(ConfigMutationTable.Diff(candidate, samePosture)); // the same declared posture written in a different key order is no change at all.
    }

    [Fact]
    public void AcceleratorTotalEdits_ClassifyAsRestartOnly_NamingOldAgainstNew()
    {
        string modelPath = TempModelFile();

        LairConfig current = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
            host:
              vram_mb: 1024
        """));

        LairConfig candidate = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
            host:
              vram_mb: 2048
        """));

        FieldChange bytes = Assert.Single(ConfigMutationTable.Diff(current, candidate)); // the accelerator ceiling binds accounting at open.
        Assert.Equal("host.vram_mb", bytes.Field);
        Assert.Equal(ConfigMutationClass.RestartOnly, bytes.MutationClass);
        Assert.Equal("1024", bytes.OldValue);
        Assert.Equal("2048", bytes.NewValue);

        LairConfig samePosture = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
            host:
              vram_mb: 1024
        """));

        Assert.Empty(ConfigMutationTable.Diff(current, samePosture)); // the same declared ceiling is no change.
    }

    [Fact]
    public async Task LiveEditAttemptOnAnOpenedEngine_RefusesCoreOvercommitAndHostTotals_AndKeepsServingOpenedValues()
    {
        string modelPath = SyntheticModel();

        LairConfig opened = new()
        {
            Server = new ServerConfig(),
            Engine = new EngineConfig { Threads = 1 },
            Host = new HostCapacityConfig { CpuCores = 8, Nodes = [new NodeCapacityOverride { Index = 0, Mb = 4096 }] },
            Models =
            [
                new ModelConfig
                {
                    Name = "reload-core",
                    Path = modelPath,
                    MaxContextTokens = 512,
                    KvBlocks = 8,
                    Placement = new PlacementConfig { CoreSet = new HashSet<int>([0]) },
                },
            ],
        };

        using GenerationService engine = GenerationService.Open(opened, NullLogger.Instance);

        LairConfig candidate = opened with
        {
            Host = opened.Host! with { CpuCores = 16 },
            Models =
            [
                opened.Models[0] with { Placement = opened.Models[0].Placement with { CoreOvercommit = true } },
            ],
        };

        IReadOnlyList<FieldResult> results = engine.ApplyConfiguration(candidate);

        FieldResult overcommit = Assert.Single(results, result => result.Change.Field == "models[0].placement.core_overcommit");
        Assert.Equal(ConfigApplyOutcome.Refused, overcommit.Outcome);
        Assert.Contains("requires restart (restart-only)", overcommit.Detail); // the actionable refusal naming field and class.

        FieldResult cores = Assert.Single(results, result => result.Change.Field == "host.cpu_cores");
        Assert.Equal(ConfigApplyOutcome.Refused, cores.Outcome);
        Assert.Contains("8 -> 16", cores.Detail); // old against new in the refusal text itself.

        Assert.False(engine.EffectiveConfiguration.Models[0].Placement.CoreOvercommit); // what opened stands: overlap is not switched under a running engine.
        Assert.Equal(8, engine.EffectiveConfiguration.Host!.CpuCores);                   // declared totals stay as they were compared against at open.
        Assert.Single(engine.EffectiveConfiguration.Host.Nodes);

        GenerationResult served = await engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = "hello world",
            MaxTokens = 8,
            Temperature = 0f,
        }, CancellationToken.None);

        Assert.False(string.IsNullOrEmpty(served.Text)); // the refusal left a serving engine serving.
    }

    [Fact]
    public void MalformedCandidateStillNamesCoreOvercommitAndHostScalars_ThroughRawExtraction()
    {
        string modelPath = TempModelFile();

        LairConfig current = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
                placement:
                  core_overcommit: false
            host:
              cpu_cores: 8
        """));

        // A candidate the loader must refuse (a non-integer host total) still names every changed key it can read, each with its class.
        string malformed = WriteConfig("""
            models:
              - name: qwen-test
                path: 'x.gguf'
                placement:
                  core_overcommit: true
            host:
              cpu_cores: plenty
        """);

        IReadOnlyList<FieldChange> edits = ConfigMutationTable.ExtractEdits(current, malformed);

        FieldChange overcommit = Assert.Single(edits, edit => edit.Field == "models[0].placement.core_overcommit");
        Assert.Equal("false", overcommit.OldValue); // the value the engine opened with reads back in the same text form raw extraction produces.
        Assert.Equal("true", overcommit.NewValue);  // and the candidate's own scalar arrives as written, not as a parsed posture.
        Assert.Equal(ConfigMutationClass.RestartOnly, overcommit.MutationClass);

        FieldChange cores = Assert.Single(edits, edit => edit.Field == "host.cpu_cores");
        Assert.Equal("8", cores.OldValue);
        Assert.Equal("plenty", cores.NewValue); // a value that cannot validate is still named with its class: nothing here applies.
        Assert.Equal(ConfigMutationClass.RestartOnly, cores.MutationClass);
    }

    private string WriteConfig(string yaml)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-reload-{Guid.NewGuid():N}.yaml");
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

    private static string SyntheticModel() => SyntheticLlamaGguf.Write("reload-core");
}
