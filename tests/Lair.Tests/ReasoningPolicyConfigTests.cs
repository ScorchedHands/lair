using Lair.Host.Config;

namespace Lair.Tests;

/// <summary>
/// The <c>models[].reasoning</c> config key: the loader accepts <c>keep</c> and <c>strip</c>, defaults to <c>keep</c> when
/// absent, an unrecognized value fails naming the model entry and the key with the accepted set, and the hot-reload table
/// classifies the field new-requests-only — the same class the other request-side output keys carry — so a reload names it
/// with its class in both the diff and the best-effort raw-scalar extraction.
/// </summary>
[Trait("Category", "Config")]
public sealed class ReasoningPolicyConfigTests : IDisposable
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
    [InlineData("keep")]
    [InlineData("strip")]
    public void Load_Reasoning_AcceptsEachPolicy(string reasoning)
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                reasoning: {reasoning}
            """));

        Assert.Equal(reasoning, config.Models[0].Reasoning);
    }

    [Fact]
    public void Load_Reasoning_AbsentDefaultsToKeep()
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            """));

        Assert.Equal("keep", config.Models[0].Reasoning);
    }

    [Theory]
    [InlineData("hide")]
    [InlineData("Keep")]
    [InlineData("remove reasoning")]
    public void Load_Reasoning_InvalidValue_FailsNamingModelAndKey(string reasoning)
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                reasoning: '{reasoning}'
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains($"models[0].reasoning '{reasoning}'", error.Message);
        Assert.Contains("'reasoning' accepts only 'keep' or 'strip'", error.Message);
    }

    [Fact]
    public void Load_Reasoning_NonScalarValue_FailsAsTextValue()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                reasoning:
                  policy: strip
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("models[0].reasoning must be a text value", error.Message);
    }

    [Fact]
    public void ReloadTable_Reasoning_IsClassifiedNewRequestsOnly()
    {
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, ConfigMutationTable.Classify("models[0].reasoning"));

        string modelPath = TempModelFile();
        LairConfig current = LoadConfig(modelPath, "keep");
        LairConfig candidate = LoadConfig(modelPath, "strip");

        FieldChange change = Assert.Single(ConfigMutationTable.Diff(current, candidate));
        Assert.Equal("models[0].reasoning", change.Field);
        Assert.Equal("keep", change.OldValue);
        Assert.Equal("strip", change.NewValue);
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, change.MutationClass);
    }

    [Fact]
    public void ReloadTable_Reasoning_EqualPolicy_ProducesNoChange()
    {
        string modelPath = TempModelFile();
        LairConfig current = LoadConfig(modelPath, "strip");
        LairConfig candidate = LoadConfig(modelPath, "strip");

        Assert.Empty(ConfigMutationTable.Diff(current, candidate));
    }

    [Fact]
    public void ReloadTable_Reasoning_ExtractEdits_NamesFieldAndClass_OnInvalidCandidate()
    {
        string modelPath = TempModelFile();
        LairConfig current = LoadConfig(modelPath, "keep");

        // The candidate carries an invalid policy, so the loader rejects it whole; the raw-scalar extraction still names
        // the changed key with its mutation class, exactly as it does for every other known key.
        string path = WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
                reasoning: hide
            """);

        Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(path));

        FieldChange edit = Assert.Single(ConfigMutationTable.ExtractEdits(current, path));
        Assert.Equal("models[0].reasoning", edit.Field);
        Assert.Equal("keep", edit.OldValue);
        Assert.Equal("hide", edit.NewValue);
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, edit.MutationClass);
    }

    // ---------------------------------------------------------------- fixture plumbing

    private LairConfig LoadConfig(string modelPath, string reasoning) =>
        YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
                reasoning: {reasoning}
            """));

    private string WriteConfig(string yaml)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-reasoning-config-{Guid.NewGuid():N}.yaml");
        File.WriteAllText(path, yaml);
        _tempPaths.Add(path);
        return path;
    }

    private string TempModelFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-reasoning-model-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, [0x47, 0x47, 0x55, 0x46]);
        _tempPaths.Add(path);
        return path;
    }
}
