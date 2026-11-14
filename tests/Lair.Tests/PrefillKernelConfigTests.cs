using Lair.Host.Config;

namespace Lair.Tests;

/// <summary>
/// The <c>engine.prefill_kernel</c> config key: the loader accepts <c>gemv</c>, <c>tiled</c>, and <c>gemm</c> and
/// defaults to <c>gemv</c>; an unrecognized value fails naming the served model and the offending value with the
/// accepted set; and the hot-reload table classifies the field new-requests-only — the same class the admission-side
/// choice keys carry — so a reload names it with its class in both the full diff and the best-effort raw-scalar
/// extraction. The selection itself is engine-wide because this build's engine serves exactly one model, which is
/// why the load-time failure can name that model alongside the value.
/// </summary>
[Trait("Category", "Config")]
public sealed class PrefillKernelConfigTests : IDisposable
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
    [InlineData("gemv")]
    [InlineData("tiled")]
    [InlineData("gemm")]
    public void Load_PrefillKernel_AcceptsEachSelection(string prefillKernel)
    {
        // The schema accepts every selection: whether one can run against the served file is the forward engine's
        // answer at model open — a dense-matrix selection over formats with no blocked matmul is refused there, naming
        // them — and never the loader's.
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            engine:
              prefill_kernel: {prefillKernel}
            """));

        Assert.Equal(prefillKernel, config.Engine.PrefillKernel);
    }

    [Fact]
    public void Load_PrefillKernel_AbsentDefaultsToGemv()
    {
        LairConfig absent = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            """));

        Assert.Equal("gemv", absent.Engine.PrefillKernel);

        LairConfig emptyEngine = YamlConfigLoader.Load(WriteConfig($"""
            engine:
              threads: 2
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            """));

        Assert.Equal("gemv", emptyEngine.Engine.PrefillKernel);
    }

    [Theory]
    [InlineData("quantized")]
    [InlineData("GemV")]
    [InlineData("tiled-gemm")]
    [InlineData("")]
    public void Load_UnknownPrefillKernel_FailsNamingModelAndValue(string prefillKernel)
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            engine:
              prefill_kernel: '{prefillKernel}'
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains($"engine.prefill_kernel '{prefillKernel}'", error.Message);
        Assert.Contains("model 'qwen-test'", error.Message); // the failure names the model the selection governs
        Assert.Contains("'prefill_kernel' accepts only 'gemv', 'tiled', or 'gemm'", error.Message);
    }

    [Fact]
    public void Load_PrefillKernel_NonScalarValue_FailsAsTextValue()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            engine:
              prefill_kernel:
                choice: gemv
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("engine.prefill_kernel must be a text value", error.Message);
    }

    [Fact]
    public void ReloadTable_PrefillKernel_IsClassifiedNewRequestsOnly()
    {
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, ConfigMutationTable.Classify("engine.prefill_kernel"));

        string modelPath = TempModelFile();
        LairConfig current = LoadConfig(modelPath, "gemv");
        LairConfig candidate = LoadConfig(modelPath, "tiled");

        FieldChange change = Assert.Single(ConfigMutationTable.Diff(current, candidate));
        Assert.Equal("engine.prefill_kernel", change.Field);
        Assert.Equal("gemv", change.OldValue);
        Assert.Equal("tiled", change.NewValue);
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, change.MutationClass);
    }

    [Fact]
    public void ReloadTable_PrefillKernel_EqualSelection_ProducesNoChange()
    {
        string modelPath = TempModelFile();
        LairConfig current = LoadConfig(modelPath, "gemv");
        LairConfig candidate = LoadConfig(modelPath, "gemv");

        Assert.Empty(ConfigMutationTable.Diff(current, candidate));
    }

    [Fact]
    public void ReloadTable_PrefillKernel_ExtractEdits_NamesFieldAndClass_OnInvalidCandidate()
    {
        string modelPath = TempModelFile();
        LairConfig current = LoadConfig(modelPath, "gemv");

        // An unrecognized selection is rejected by the loader outright; the best-effort extraction still names the
        // edited field with its mutation class so a refused edit reports as new-requests-only rather than falling
        // through unclassified.
        string path = WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
            engine:
              prefill_kernel: quantized
            """);

        Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(path));

        FieldChange edit = Assert.Single(ConfigMutationTable.ExtractEdits(current, path));
        Assert.Equal("engine.prefill_kernel", edit.Field);
        Assert.Equal("gemv", edit.OldValue);
        Assert.Equal("quantized", edit.NewValue);
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, edit.MutationClass);
    }

    // ---------------------------------------------------------------- fixture plumbing

    private LairConfig LoadConfig(string modelPath, string prefillKernel) =>
        YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
            engine:
              prefill_kernel: {prefillKernel}
            """));

    private string WriteConfig(string yaml)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-prefill-kernel-config-{Guid.NewGuid():N}.yaml");
        File.WriteAllText(path, yaml);
        _tempPaths.Add(path);
        return path;
    }

    private string TempModelFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-prefill-kernel-model-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, [0x47, 0x47, 0x55, 0x46]);
        _tempPaths.Add(path);
        return path;
    }
}
