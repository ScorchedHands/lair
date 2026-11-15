using Lair.Host.Config;
using Lair.Http;
using Lair.Models;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// The spread posture a model's placement declares: <c>placement.numa_spread</c> parses its three accepted values with
/// <c>auto</c> as the posture an absent key resolves to, and an unrecognized value fails naming the served model, the
/// key, and the accepted set. In the reload table the key classifies as new-requests-only — the preference rides later
/// placement decisions — and a live edit on an opened engine records it while the node the engine opened on and the
/// posture its worker pool and weight pages established stay as the open resolved them. The raw-scalar extraction that
/// names changed keys when a candidate fails validation covers the plain scalar.
/// </summary>
[Trait("Category", "Config")]
[Trait("Category", "HotReload")]
public sealed class NumaSpreadConfigTests : IDisposable
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
    public void Load_NumaSpreadKey_ParsesEachPostureAndDefaultsToAuto()
    {
        LairConfig local = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                placement:
                  numa_spread: local
            """));
        Assert.Equal("local", local.Models[0].Placement.NumaSpread);

        LairConfig spread = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                placement:
                  numa_spread: interleave
            """));
        Assert.Equal("interleave", spread.Models[0].Placement.NumaSpread);

        LairConfig defaultPosture = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                placement:
                  core_set: "0-3"
            """));
        Assert.Equal("auto", defaultPosture.Models[0].Placement.NumaSpread); // absent keeps today's posture: placement chooses.
    }

    [Fact]
    public void Load_UnknownNumaSpread_FailsNamingModelKeyAndAcceptedSet()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                placement:
                  numa_spread: everywhere
            """)));

        Assert.Contains("models[0].placement.numa_spread 'everywhere' is invalid for model 'qwen-test'", error.Message);
        Assert.Contains("'auto', 'local', or 'interleave'", error.Message);
    }

    [Fact]
    public void NumaSpreadEdits_ClassifyAsNewRequestsOnly_NamingOldAgainstNew()
    {
        string modelPath = TempModelFile();

        LairConfig current = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
                placement:
                  numa_spread: auto
            """));

        LairConfig candidate = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
                placement:
                  numa_spread: interleave
            """));

        FieldChange spread = Assert.Single(ConfigMutationTable.Diff(current, candidate));
        Assert.Equal("models[0].placement.numa_spread", spread.Field);
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, spread.MutationClass); // the preference rides later placement decisions.
        Assert.Equal("auto", spread.OldValue);
        Assert.Equal("interleave", spread.NewValue);

        Assert.Equal(ConfigMutationClass.NewRequestsOnly, ConfigMutationTable.Classify("models[0].placement.numa_spread"));
        Assert.Empty(ConfigMutationTable.Diff(candidate, candidate)); // the same posture against itself is no change at all.
    }

    [Fact]
    public async Task LiveNumaSpreadEdit_RecordsThePreference_AndReportsTheOpenedPostureAsEstablished()
    {
        string modelPath = SyntheticModel("reload-spread");

        LairConfig opened = new()
        {
            Server = new ServerConfig(),
            Engine = new EngineConfig { Threads = 1 },
            Models =
            [
                new ModelConfig
                {
                    Name = "reload-spread",
                    Path = modelPath,
                    MaxContextTokens = 512,
                    KvBlocks = 8,
                    Placement = new PlacementConfig { NumaSpread = "auto" },
                },
            ],
        };

        using GenerationService engine = GenerationService.Open(opened, NullLogger.Instance);
        IReadOnlyList<int> openedNodes = engine.ResolvedPlacement!.NodeIndices; // the node posture the engine opened on

        LairConfig candidate = opened with
        {
            Models = [opened.Models[0] with { Placement = opened.Models[0].Placement with { NumaSpread = "interleave" } }],
        };

        FieldResult spread = Assert.Single(engine.ApplyConfiguration(candidate));
        Assert.Equal("models[0].placement.numa_spread", spread.Change.Field);
        Assert.Equal(ConfigApplyOutcome.Applied, spread.Outcome);
        Assert.Contains("later placement decisions", spread.Detail);
        Assert.Contains("remain as the open resolved them", spread.Detail); // the honest reading: an opened posture is not re-bound live.

        Assert.Equal("interleave", engine.EffectiveConfiguration.Models[0].Placement.NumaSpread); // the preference is in force for later decisions.
        Assert.Equal(openedNodes, engine.ResolvedPlacement!.NodeIndices);                         // and the opened placement stands untouched.

        GenerationResult served = await engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = "hello world",
            MaxTokens = 8,
            Temperature = 0f,
        }, CancellationToken.None);

        Assert.False(string.IsNullOrEmpty(served.Text)); // the engine keeps serving on the posture it opened with.
    }

    [Fact]
    public void MalformedCandidateStillNamesNumaSpread_ThroughRawExtraction()
    {
        string modelPath = TempModelFile();

        LairConfig current = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
                placement:
                  numa_spread: auto
            """));

        // A candidate the loader must refuse (a posture outside the accepted set) still names the changed key with its class.
        string malformed = WriteConfig("""
            models:
              - name: qwen-test
                path: 'x.gguf'
                placement:
                  numa_spread: sideways
            """);

        IReadOnlyList<FieldChange> edits = ConfigMutationTable.ExtractEdits(current, malformed);

        FieldChange spread = Assert.Single(edits, edit => edit.Field == "models[0].placement.numa_spread");
        Assert.Equal("auto", spread.OldValue);
        Assert.Equal("sideways", spread.NewValue); // a value that cannot validate is still named with its class: nothing here applies.
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, spread.MutationClass);
    }

    [Fact]
    public void ResolverVocabulary_MatchesTheKeysTheLoaderAccepts()
    {
        Assert.Equal(["auto", "local", "interleave"], NumaSpreadNames.DeclaredNames);
        Assert.True(NumaSpreadNames.TryParse("interleave", out NumaSpreadPreference parsed));
        Assert.Equal(NumaSpreadPreference.Interleave, parsed);
        Assert.False(NumaSpreadNames.TryParse("everywhere", out _));
    }

    private string WriteConfig(string yaml)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-spread-{Guid.NewGuid():N}.yaml");
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

    private static string SyntheticModel(string modelName) => SyntheticLlamaGguf.Write(modelName);
}
