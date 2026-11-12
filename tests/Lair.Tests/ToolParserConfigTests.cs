using Lair.Host.Config;

namespace Lair.Tests;

/// <summary>
/// The <c>models[].tool_parser</c> config key: the loader accepts the four choices and defaults to <c>auto</c>, an
/// unrecognized value fails naming the model entry and the key with the accepted set, and the hot-reload table
/// classifies the field new-requests-only — the same class the other admission-side tool knobs carry — so a reload
/// names it with its class in both the diff and the best-effort raw-scalar extraction.
/// </summary>
[Trait("Category", "Config")]
public sealed class ToolParserConfigTests : IDisposable
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
    [InlineData("template")]
    [InlineData("fenced")]
    [InlineData("none")]
    public void Load_ToolParser_AccceptsEachChoice(string toolParser)
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                tool_parser: {toolParser}
            """));

        Assert.Equal(toolParser, config.Models[0].ToolParser);
    }

    [Fact]
    public void Load_ToolParser_AbsentDefaultsToAuto()
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            """));

        Assert.Equal("auto", config.Models[0].ToolParser);
    }

    [Theory]
    [InlineData("template-only")]
    [InlineData("Auto")]
    [InlineData("")]
    public void Load_ToolParser_InvalidValue_FailsNamingModelAndKey(string toolParser)
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                tool_parser: '{toolParser}'
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains($"models[0].tool_parser '{toolParser}'", error.Message);
        Assert.Contains("'tool_parser' accepts only 'auto', 'template', 'fenced', or 'none'", error.Message);
    }

    [Fact]
    public void Load_ToolParser_NonScalarValue_FailsAsTextValue()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                tool_parser:
                  choice: fenced
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("models[0].tool_parser must be a text value", error.Message);
    }

    [Fact]
    public void ReloadTable_ToolParser_IsClassifiedNewRequestsOnly()
    {
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, ConfigMutationTable.Classify("models[0].tool_parser"));

        string modelPath = TempModelFile();
        LairConfig current = LoadConfig(modelPath, "auto");
        LairConfig candidate = LoadConfig(modelPath, "none");

        FieldChange change = Assert.Single(ConfigMutationTable.Diff(current, candidate));
        Assert.Equal("models[0].tool_parser", change.Field);
        Assert.Equal("auto", change.OldValue);
        Assert.Equal("none", change.NewValue);
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, change.MutationClass);
    }

    [Fact]
    public void ReloadTable_ToolParser_EqualChoice_ProducesNoChange()
    {
        string modelPath = TempModelFile();
        LairConfig current = LoadConfig(modelPath, "fenced");
        LairConfig candidate = LoadConfig(modelPath, "fenced");

        Assert.Empty(ConfigMutationTable.Diff(current, candidate));
    }

    [Fact]
    public void ReloadTable_ToolParser_ExtractEdits_NamesFieldAndClass_OnInvalidCandidate()
    {
        string modelPath = TempModelFile();
        LairConfig current = LoadConfig(modelPath, "auto");

        // The candidate carries an invalid choice, so the loader rejects it whole; the raw-scalar extraction still
        // names the changed key with its mutation class, exactly as it does for every other known key. The unchanged
        // path scalar is not reported, so the tool_parser edit stands alone.
        string path = WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
                tool_parser: bogus
            """);

        Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(path));

        FieldChange edit = Assert.Single(ConfigMutationTable.ExtractEdits(current, path));
        Assert.Equal("models[0].tool_parser", edit.Field);
        Assert.Equal("auto", edit.OldValue);
        Assert.Equal("bogus", edit.NewValue);
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, edit.MutationClass);
    }

    // ---------------------------------------------------------------- fixture plumbing

    private LairConfig LoadConfig(string modelPath, string toolParser) =>
        YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
                tool_parser: {toolParser}
            """));

    private string WriteConfig(string yaml)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-tool-parser-config-{Guid.NewGuid():N}.yaml");
        File.WriteAllText(path, yaml);
        _tempPaths.Add(path);
        return path;
    }

    private string TempModelFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-tool-parser-model-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, [0x47, 0x47, 0x55, 0x46]);
        _tempPaths.Add(path);
        return path;
    }
}
