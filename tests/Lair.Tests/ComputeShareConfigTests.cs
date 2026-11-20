using Lair.Host.Config;
using Lair.Http;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// `models[].compute_share` as a public config key and as a hot-reload mutation: absent declares no share and leaves step composition
/// bounded by segment caps alone, a present value parses into the model config, an out-of-range or non-integer value fails naming the model
/// entry, the model, the key, the offending value, and the accepted range, and the key classifies Live — future compositions re-resolve
/// against the new share while nothing already launched is preempted. On a model whose resolved route names no accelerator-backed segment
/// the posture applies and records against no window at all, the same warn-and-accept shape its load reported.
/// </summary>
[Trait("Category", "Config")]
public sealed class ComputeShareConfigTests : IDisposable
{
    private readonly List<string> _tempPaths = [];

    [Fact]
    public void Load_ComputeShare_ParsesTheDeclaredShare_AndAbsentKeepsNoShare()
    {
        LairConfig declared = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                compute_share: 30
            """));

        Assert.Equal(30, declared.Models[0].ComputeShare);

        LairConfig absent = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            """));

        Assert.Null(absent.Models[0].ComputeShare); // absent declares no share: composition bounds this model by its segment caps and quota headroom alone
    }

    [Fact]
    public void Load_ComputeShare_OutOfRange_FailsNamingModelKeyModelValueAndAcceptedRange()
    {
        ConfigException low = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                compute_share: 0
            """)));

        Assert.Contains("models[0].compute_share 0 is invalid for model 'qwen-test'", low.Message, StringComparison.Ordinal);
        Assert.Contains("'compute_share' must be an integer from 1 to 100", low.Message, StringComparison.Ordinal);

        ConfigException high = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                compute_share: 101
            """)));

        Assert.Contains("models[0].compute_share 101", high.Message, StringComparison.Ordinal);

        ConfigException nonInteger = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                compute_share: half
            """)));

        Assert.Contains("models[0].compute_share must be an integer value.", nonInteger.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ComputeShare_ClassifiesLive_AndDiffReportsIt_AsOneLiveChange()
    {
        Assert.Equal(ConfigMutationClass.Live, ConfigMutationTable.Classify("models[0].compute_share"));

        LairConfig current = ConfigWithShare(30);
        LairConfig candidate = ConfigWithShare(60);

        FieldChange change = Assert.Single(ConfigMutationTable.Diff(current, candidate));
        Assert.Equal("models[0].compute_share", change.Field);
        Assert.Equal(ConfigMutationClass.Live, change.MutationClass); // future boundaries re-resolve; in-flight sequences keep their window slots

        Assert.Empty(ConfigMutationTable.Diff(current, ConfigWithShare(30))); // the same declared share is no change at all
    }

    [Fact]
    public void ApplyConfiguration_ComputeShare_AppliesThroughTheLiveSeam_AndRecordsAgainstNoWindow_WhenNoDeviceSegmentResolves()
    {
        string path = SyntheticModelFile();

        using GenerationService engine = GenerationService.Open(ConfigWithShare(30, path), NullLogger.Instance);

        Assert.Single(engine.ResolvedRoute.Segments); // this build's CPU-only posture names no accelerator-backed segment
        Assert.Null(engine.HostCoScheduler); // no device window bounds this model's scheduler

        IReadOnlyList<FieldResult> results = engine.ApplyConfiguration(ConfigWithShare(60, path));

        FieldResult applied = Assert.Single(results);
        Assert.Equal("models[0].compute_share", applied.Change.Field);
        Assert.Equal(ConfigMutationClass.Live, applied.Change.MutationClass);
        Assert.Equal(ConfigApplyOutcome.Applied, applied.Outcome);
        Assert.Contains("device fair-share 30 -> 60", applied.Detail, StringComparison.Ordinal);
        Assert.Contains("no accelerator-backed segment", applied.Detail, StringComparison.Ordinal); // the posture records for a placement that opens a window
        Assert.Contains("nothing in flight is preempted", applied.Detail, StringComparison.Ordinal);

        Assert.Equal(60, engine.EffectiveConfiguration.Models[0].ComputeShare); // the merge carried exactly the share that applied
    }

    // ---------------------------------------------------------------- helpers

    private static LairConfig ConfigWithShare(int share) => ConfigWithShare(share, modelPath: null);

    private static LairConfig ConfigWithShare(int share, string? modelPath) => new()
    {
        Server = new ServerConfig(),
        Engine = new EngineConfig { Threads = 1 },
        Models =
        [
            new ModelConfig
            {
                Name = "compute-share-test",
                Path = modelPath ?? "model.gguf",
                MaxContextTokens = 512,
                KvBlocks = 8,
                ComputeShare = share,
            },
        ],
    };

    private string SyntheticModelFile()
    {
        string path = SyntheticLlamaGguf.Write("compute-share-serve");
        _tempPaths.Add(path);
        return path;
    }

    private string WriteConfig(string yaml)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-config-{Guid.NewGuid():N}.yaml");
        File.WriteAllText(path, yaml);
        _tempPaths.Add(path);
        return path;
    }

    private string TempModelFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-model-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, [0x47, 0x47, 0x55, 0x46]);
        _tempPaths.Add(path);
        return path;
    }

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
}
