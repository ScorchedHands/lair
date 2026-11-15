using Lair.Host.Config;
using Lair.Http;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// The published source a model entry pins and the pack manifest that records its bytes: the <c>source:</c> block parses
/// repo, revision as declared, and exactly one of <c>file</c> or <c>files</c> — each entry a plain path/name string or a
/// mapping stating the path beside its pinned sha256 and declared size — and every malformed value fails config load
/// naming the qualified config path and the offending value like every other block does. An absent block preserves the
/// local-path posture exactly. The pack's <c>manifest.json</c> round-trips repo, resolved revision, per-file sha256,
/// size, and fetched_at through the store, and survives a hot-reload cycle untouched: a valid live edit applies without
/// rewriting the pin, and an edit of the pin itself classifies as restart-only because loadability depends on the
/// manifest and fetch behavior established when the model opened.
/// </summary>
[Trait("Category", "Config")]
[Trait("Category", "HotReload")]
public sealed class ModelSourceConfigTests : IDisposable
{
    private const string ValidSha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    private const string ValidSha2 = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    private readonly List<string> _tempPaths = [];
    private readonly List<string> _tempDirs = [];

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

        foreach (string dir in _tempDirs)
        {
            try
            {
                Directory.Delete(dir, recursive: true);
            }
            catch (IOException)
            {
            }
            catch (UnauthorizedAccessException)
            {
            }
        }
    }

    [Fact]
    public void Load_SingleFileSource_ParsesRepoRevisionAsDeclaredAndFile()
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                source:
                  repo: bartowski/Qwen3-Fixture
                  revision: main
                  file: model.gguf
            """));

        ModelSourceConfig source = config.Models[0].Source!;
        Assert.NotNull(source);
        Assert.Equal("bartowski/Qwen3-Fixture", source.Repo);
        Assert.Equal("main", source.Revision); // stored as declared: a branch becomes a sha only when fetch resolves it.

        SourceFileConfig file = Assert.Single(source.Files);
        Assert.Equal("model.gguf", file.Path);
        Assert.Null(file.Sha256); // a bare path entry pins no hash yet.
        Assert.Null(file.Size);
    }

    [Fact]
    public void Load_MultiFileSource_ParsesPlainAndMappingEntries()
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                source:
                  repo: bartowski/Llama-Fixture
                  revision: {ValidSha}
                  files:
                    - shard-00001-of-00002.gguf
                    - path: shard-00002-of-00002.gguf
                      sha256: {ValidSha2}
                      size: 1024
            """));

        ModelSourceConfig source = config.Models[0].Source!;
        Assert.Equal(ValidSha, source.Revision);

        Assert.Equal(2, source.Files.Count);

        SourceFileConfig plain = source.Files[0];
        Assert.Equal("shard-00001-of-00002.gguf", plain.Path);
        Assert.Null(plain.Sha256);
        Assert.Null(plain.Size);

        SourceFileConfig pinned = source.Files[1];
        Assert.Equal("shard-00002-of-00002.gguf", pinned.Path);
        Assert.Equal(ValidSha2, pinned.Sha256);
        Assert.Equal(1024L, pinned.Size);
    }

    [Fact]
    public void Load_SingleFileAsMapping_ParsesHashAndSize()
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                source:
                  repo: bartowski/Qwen3-Fixture
                  revision: {ValidSha}
                  file:
                    path: model.gguf
                    sha256: {ValidSha}
                    size: 0
            """));

        SourceFileConfig file = Assert.Single(config.Models[0].Source!.Files);
        Assert.Equal("model.gguf", file.Path);
        Assert.Equal(ValidSha, file.Sha256);
        Assert.Equal(0L, file.Size); // a declared zero is a non-negative size like any other.
    }

    [Fact]
    public void Load_AbsentSource_PreservesLocalModelBehaviorExactly()
    {
        string modelPath = TempModelFile();

        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
            """));

        Assert.Null(config.Models[0].Source); // no source block: the declared local path is served as it always was.
        Assert.Equal(modelPath, config.Models[0].Path);
    }

    [Fact]
    public void Load_EmptyRepo_FailsNamingPathAndValue()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                source:
                  repo: ""
                  revision: main
                  file: model.gguf
            """)));

        Assert.Contains("models[0].source.repo", error.Message);
        Assert.Contains("non-empty", error.Message);
    }

    [Fact]
    public void Load_EmptyRevision_FailsNamingPathAndValue()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                source:
                  repo: bartowski/Qwen3-Fixture
                  revision: ""
                  file: model.gguf
            """)));

        Assert.Contains("models[0].source.revision", error.Message);
        Assert.Contains("commit sha or branch", error.Message);
    }

    [Fact]
    public void Load_EmptyFilesList_FailsNamingPathAndValue()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                source:
                  repo: bartowski/Qwen3-Fixture
                  revision: main
                  files: []
            """)));

        Assert.Contains("models[0].source.files is empty for model 'qwen-test'", error.Message);
        Assert.Contains("at least one model file", error.Message);
    }

    [Fact]
    public void Load_BothFileAndFiles_FailsNamingBothKeys()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                source:
                  repo: bartowski/Qwen3-Fixture
                  revision: main
                  file: model.gguf
                  files: [other.gguf]
            """)));

        Assert.Contains("models[0].source declares both 'file' and 'files'", error.Message);
        Assert.Contains("exactly one of them", error.Message);
    }

    [Fact]
    public void Load_MissingFileSpellings_FailsNamingTheSourceBlock()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                source:
                  repo: bartowski/Qwen3-Fixture
                  revision: main
            """)));

        Assert.Contains("models[0].source declares no model file", error.Message);
        Assert.Contains("exactly one of 'file' or 'files'", error.Message);
    }

    [Fact]
    public void Load_InvalidSha256_FailsNamingEntryPathAndValue()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                source:
                  repo: bartowski/Qwen3-Fixture
                  revision: main
                  files:
                    - path: model.gguf
                      sha256: DEADBEEF
            """)));

        Assert.Contains("models[0].source.files[0].sha256 'DEADBEEF' is invalid for model 'qwen-test'", error.Message);
        Assert.Contains("64 lowercase hex characters", error.Message);
    }

    [Fact]
    public void Load_UppercaseSha256_FailsNamingEntryPathAndValue()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                source:
                  repo: bartowski/Qwen3-Fixture
                  revision: main
                  file:
                    path: model.gguf
                    sha256: {ValidSha.ToUpperInvariant()}
            """)));

        Assert.Contains($"models[0].source.file.sha256 '{ValidSha.ToUpperInvariant()}' is invalid", error.Message);
        Assert.Contains("64 lowercase hex characters", error.Message);
    }

    [Fact]
    public void Load_NegativeSize_FailsNamingEntryPathAndValue()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                source:
                  repo: bartowski/Qwen3-Fixture
                  revision: main
                  files:
                    - path: model.gguf
                      size: -1
            """)));

        Assert.Contains("models[0].source.files[0].size -1 is invalid for model 'qwen-test'", error.Message);
        Assert.Contains("non-negative integer", error.Message);
    }

    [Fact]
    public void Load_UnknownSourceKey_FailsNamingAllowedKeys()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                source:
                  repo: bartowski/Qwen3-Fixture
                  revision: main
                  file: model.gguf
                  branch: main
            """)));

        Assert.Contains("Unknown key 'branch' under models[0].source", error.Message);
        Assert.Contains("repo, revision, file, files", error.Message);
    }

    [Fact]
    public void Manifest_SaveAndLoad_RoundTripsRepoResolvedRevisionHashSizeAndFetchedAt()
    {
        string packDir = TempPackDir();

        DateTimeOffset fetchedAt = new(2026, 11, 15, 9, 0, 0, TimeSpan.Zero);
        ModelManifest manifest = new()
        {
            Models =
            [
                new ModelManifestEntry
                {
                    Model = "qwen-test",
                    Repo = "bartowski/Qwen3-Fixture",
                    ResolvedRevision = ValidSha,
                    Files =
                    [
                        new ModelManifestFile { Path = "model.gguf", Sha256 = ValidSha2, Size = 1234567, FetchedAt = fetchedAt },
                    ],
                },
            ],
        };

        ModelManifestStore.Save(packDir, manifest);
        Assert.True(File.Exists(Path.Combine(packDir, ModelManifestStore.FileName)));

        ModelManifest reloaded = ModelManifestStore.Load(packDir)!;
        ModelManifestEntry entry = Assert.Single(reloaded.Models);
        Assert.Equal("qwen-test", entry.Model);
        Assert.Equal("bartowski/Qwen3-Fixture", entry.Repo);
        Assert.Equal(ValidSha, entry.ResolvedRevision); // the resolved sha is the pin, whatever the config declared.

        ModelManifestFile file = Assert.Single(entry.Files);
        Assert.Equal("model.gguf", file.Path);
        Assert.Equal(ValidSha2, file.Sha256);
        Assert.Equal(1234567L, file.Size);
        Assert.Equal(fetchedAt, file.FetchedAt);

        // A pack that carries no manifest yet loads as no manifest rather than failing.
        string emptyDir = TempPackDir();
        Assert.Null(ModelManifestStore.Load(emptyDir));
    }

    [Fact]
    public void SourceEdits_ClassifyAsRestartOnly_NamingOldAgainstNew()
    {
        string modelPath = TempModelFile();

        LairConfig current = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
                source:
                  repo: bartowski/Qwen3-Fixture
                  revision: main
                  files:
                    - path: model.gguf
                      sha256: {ValidSha}
                      size: 4096
            """));

        LairConfig candidate = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
                source:
                  repo: bartowski/Qwen3-Fixture
                  revision: {ValidSha2}
                  files:
                    - path: model.gguf
                      sha256: {ValidSha}
                      size: 4096
            """));

        FieldChange source = Assert.Single(ConfigMutationTable.Diff(current, candidate));
        Assert.Equal("models[0].source", source.Field);
        Assert.Equal(ConfigMutationClass.RestartOnly, source.MutationClass); // loadability depends on the manifest and fetch behavior established at open.
        Assert.Contains("revision=main", source.OldValue);
        Assert.Contains($"revision={ValidSha2}", source.NewValue);

        Assert.Equal(ConfigMutationClass.RestartOnly, ConfigMutationTable.Classify("models[0].source"));
        Assert.Empty(ConfigMutationTable.Diff(candidate, candidate)); // the same declared pin against itself is no change at all.
    }

    [Fact]
    public async Task ManifestSurvivesHotReloadCycle_AndSourceEditsAreRefusedOnAnOpenedEngine()
    {
        string modelPath = SyntheticLlamaGguf.Write("reload-source");
        string packDir = TempPackDir();

        DateTimeOffset fetchedAt = new(2026, 11, 15, 9, 0, 0, TimeSpan.Zero);
        ModelManifest manifest = new()
        {
            Models =
            [
                new ModelManifestEntry
                {
                    Model = "reload-source",
                    Repo = "bartowski/Qwen3-Fixture",
                    ResolvedRevision = ValidSha,
                    Files = [new ModelManifestFile { Path = "model.gguf", Sha256 = ValidSha2, Size = 4096, FetchedAt = fetchedAt }],
                },
            ],
        };
        ModelManifestStore.Save(packDir, manifest);

        string configPath = WriteConfig($"""
            models:
              - name: reload-source
                path: '{modelPath}'
                max_context_tokens: 512
                kv_blocks: 8
                source:
                  repo: bartowski/Qwen3-Fixture
                  revision: main
                  file: model.gguf
            """);

        LairConfig opened = YamlConfigLoader.Load(configPath);
        using GenerationService engine = GenerationService.Open(opened, NullLogger.Instance);

        // A valid hot-reload change applies while the pack manifest stands untouched: a live edit never rewrites a pin.
        FieldResult ui = Assert.Single(engine.ApplyConfiguration(engine.EffectiveConfiguration with { Ui = new UiConfig { RecentRequests = 40 } }));
        Assert.Equal("ui.recent_requests", ui.Change.Field);
        Assert.Equal(ConfigApplyOutcome.Applied, ui.Outcome);

        ModelManifest afterLiveEdit = ModelManifestStore.Load(packDir)!;
        AssertManifestIntact(afterLiveEdit, fetchedAt);

        // An edit of the pin itself is refused naming field and class, the opened pin stands, and the manifest still stands.
        LairConfig candidate = engine.EffectiveConfiguration with
        {
            Models = [engine.EffectiveConfiguration.Models[0] with { Source = engine.EffectiveConfiguration.Models[0].Source! with { Revision = ValidSha2 } }],
        };

        FieldResult source = Assert.Single(engine.ApplyConfiguration(candidate));
        Assert.Equal("models[0].source", source.Change.Field);
        Assert.Equal(ConfigMutationClass.RestartOnly, source.Change.MutationClass);
        Assert.Equal(ConfigApplyOutcome.Refused, source.Outcome);
        Assert.Contains("requires restart (restart-only)", source.Detail);

        Assert.Equal("main", engine.EffectiveConfiguration.Models[0].Source!.Revision); // the pin the engine opened with stands.

        ModelManifest afterRefusedEdit = ModelManifestStore.Load(packDir)!;
        AssertManifestIntact(afterRefusedEdit, fetchedAt);

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

    private static void AssertManifestIntact(ModelManifest manifest, DateTimeOffset fetchedAt)
    {
        ModelManifestEntry entry = Assert.Single(manifest.Models);
        Assert.Equal("reload-source", entry.Model);
        Assert.Equal("bartowski/Qwen3-Fixture", entry.Repo);
        Assert.Equal(ValidSha, entry.ResolvedRevision);

        ModelManifestFile file = Assert.Single(entry.Files);
        Assert.Equal("model.gguf", file.Path);
        Assert.Equal(ValidSha2, file.Sha256);
        Assert.Equal(4096L, file.Size);
        Assert.Equal(fetchedAt, file.FetchedAt);
    }

    private string WriteConfig(string yaml)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-source-{Guid.NewGuid():N}.yaml");
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

    private string TempPackDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"lair-pack-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        _tempDirs.Add(dir);
        return dir;
    }
}
