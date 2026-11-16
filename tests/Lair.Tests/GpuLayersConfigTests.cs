using Lair.Host.Config;
using Lair.Http;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// The keys that decide which compute backends and route segments an open builds: gpu_layers parses as a plain placement integer beside the device posture it splits, refuses a negative count naming model, key, and value exactly like every other out-of-range placement number here, classifies as restart-only in the reload table (an edit cannot change which backends already opened), refuses a live attempt against an opened engine while that engine keeps serving what it opened with, and still names itself through raw extraction when a candidate fails validation outright.
/// </summary>
[Trait("Category", "HotReload")]
[Trait("Category", "Config")]
public sealed class GpuLayersConfigTests : IDisposable
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

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(7)]
    public void Load_GpuLayers_ParsesAsDeclaredAndDefaultsToZero(int declared)
    {
        string modelPath = TempModelFile();

        LairConfig loaded = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
                placement:
                  gpu_layers: {declared}
            """));

        Assert.Equal(declared, loaded.Models[0].Placement.GpuLayers); // the split lands as declared — where an engine open resolves it is decided there, not here.

        LairConfig absent = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
                placement:
                  thread_quota: 2
            """));

        Assert.Equal(0, absent.Models[0].Placement.GpuLayers); // an absent key keeps today's CPU-only posture.
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(-8)]
    public void Load_NegativeGpuLayers_RefusesNamingModelKeyAndValue(int declared)
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                placement:
                  gpu_layers: {declared}
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));

        Assert.Contains($"models[0].placement.gpu_layers {declared}", error.Message);
        Assert.Contains("is invalid for model 'qwen-test'", error.Message); // the served model is named beside the key and its value, like every other placement refusal.
        Assert.Contains("'gpu_layers' must be a non-negative integer", error.Message);
    }

    [Fact]
    public void Load_NonIntegerGpuLayers_RefusesAsTheOtherIntegersDo()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                placement:
                  gpu_layers: some layers
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));

        Assert.Contains("models[0].placement.gpu_layers must be an integer value", error.Message);
    }

    [Fact]
    public void GpuLayersEdits_ClassifyAsRestartOnly_NamingOldAgainstNew()
    {
        string modelPath = TempModelFile();

        LairConfig current = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
                placement:
                  gpu_layers: 0
            """));

        LairConfig candidate = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
                placement:
                  gpu_layers: 2
            """));

        FieldChange split = Assert.Single(ConfigMutationTable.Diff(current, candidate)); // the posture moved and nothing else did.
        Assert.Equal("models[0].placement.gpu_layers", split.Field);
        Assert.Equal(ConfigMutationClass.RestartOnly, split.MutationClass); // which backends an open builds is established when it opens; a live edit cannot rebuild them.
        Assert.Equal("0", split.OldValue);
        Assert.Equal("2", split.NewValue);

        LairConfig samePosture = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
                placement:
                  gpu_layers: 0
            """));

        Assert.Empty(ConfigMutationTable.Diff(current, samePosture)); // an explicit zero is the posture itself, not a change to it.
    }

    [Fact]
    public async Task LiveEditAttemptOnAnOpenedEngine_RefusesGpuLayers_AndKeepsServingOpenedValues()
    {
        string modelPath = SyntheticModel();

        LairConfig opened = new()
        {
            Server = new ServerConfig(),
            Engine = new EngineConfig { Threads = 1 },
            Models =
            [
                new ModelConfig
                {
                    Name = "reload-split",
                    Path = modelPath,
                    MaxContextTokens = 512,
                    KvBlocks = 8,
                },
            ],
        };

        using GenerationService engine = GenerationService.Open(opened, NullLogger.Instance);

        LairConfig candidate = opened with
        {
            Models = [opened.Models[0] with { Placement = new PlacementConfig { GpuLayers = 1 } }],
        };

        IReadOnlyList<FieldResult> results = engine.ApplyConfiguration(candidate);

        FieldResult split = Assert.Single(results, result => result.Change.Field == "models[0].placement.gpu_layers");
        Assert.Equal(ConfigApplyOutcome.Refused, split.Outcome);
        Assert.Contains("requires restart (restart-only)", split.Detail); // the actionable refusal naming field and class.
        Assert.Contains("0 -> 1", split.Detail);

        Assert.Equal(0, engine.EffectiveConfiguration.Models[0].Placement.GpuLayers); // what opened stands: no backend appears under a running engine.
        Assert.Single(engine.ResolvedRoute.Segments);

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
    public void MalformedCandidateStillNamesGpuLayers_ThroughRawExtraction()
    {
        string modelPath = TempModelFile();

        LairConfig current = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
                placement:
                  gpu_layers: 0
            """));

        // A candidate the loader must refuse (a non-integer placement value) still names every changed key it can read, each with its class.
        string malformed = WriteConfig("""
            models:
              - name: qwen-test
                path: 'x.gguf'
                placement:
                  gpu_layers: plenty
            """);

        IReadOnlyList<FieldChange> edits = ConfigMutationTable.ExtractEdits(current, malformed);

        FieldChange split = Assert.Single(edits, edit => edit.Field == "models[0].placement.gpu_layers");
        Assert.Equal("0", split.OldValue);   // the value the engine opened with reads back in the same text form raw extraction produces.
        Assert.Equal("plenty", split.NewValue); // and the candidate's own scalar arrives as written, not as a parsed posture.
        Assert.Equal(ConfigMutationClass.RestartOnly, split.MutationClass);
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

    private static string SyntheticModel() => SyntheticLlamaGguf.Write("reload-split");
}
