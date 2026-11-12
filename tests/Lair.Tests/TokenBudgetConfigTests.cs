using Lair.Host.Config;

namespace Lair.Tests;

/// <summary>
/// The <c>models[].max_reasoning_tokens</c> and <c>models[].max_response_tokens</c> config keys: the loader carries each
/// budget on the model entry, an absent key leaves that side of generation unbudgeted with every other default intact, a
/// value below 1 fails naming the model entry, key, and offending number exactly as the other sized keys do, and the
/// hot-reload table classifies both fields new-requests-only — the same class the per-model request-policy keys carry —
/// so a reload names them with their class in both the diff and the best-effort raw-scalar extraction.
/// </summary>
[Trait("Category", "Config")]
public sealed class TokenBudgetConfigTests : IDisposable
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

    [Fact]
    public void Load_BothBudgets_ParseFromModelEntry()
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                max_reasoning_tokens: 1024
                max_response_tokens: 512
            """));

        ModelConfig model = Assert.Single(config.Models);
        Assert.Equal(1024, model.MaxReasoningTokens);
        Assert.Equal(512, model.MaxResponseTokens);
    }

    [Fact]
    public void Load_BudgetsAbsent_RemainUnbudgeted_AndOtherDefaultsStand()
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            """));

        ModelConfig model = Assert.Single(config.Models);
        Assert.Null(model.MaxReasoningTokens);
        Assert.Null(model.MaxResponseTokens);
        Assert.Equal(4096, model.MaxContextTokens);
        Assert.Equal(256, model.KvBlocks);
    }

    [Fact]
    public void Load_OnlyOneBudget_Parses_AndTheOtherStaysUnbudgeted()
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                max_response_tokens: 256
            """));

        Assert.Null(config.Models[0].MaxReasoningTokens);
        Assert.Equal(256, config.Models[0].MaxResponseTokens);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(0)]
    public void Load_MaxReasoningTokensTooSmall_FailsNamingModelKeyAndValue(int budget)
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                max_reasoning_tokens: {budget}
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains($"models[0].max_reasoning_tokens {budget}", error.Message);
        Assert.Contains("'max_reasoning_tokens' must be 1 or greater", error.Message);
    }

    [Theory]
    [InlineData(-7)]
    [InlineData(0)]
    public void Load_MaxResponseTokensTooSmall_FailsNamingModelKeyAndValue(int budget)
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                max_response_tokens: {budget}
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains($"models[0].max_response_tokens {budget}", error.Message);
        Assert.Contains("'max_response_tokens' must be 1 or greater", error.Message);
    }

    [Theory]
    [InlineData("max_reasoning_tokens")]
    [InlineData("max_response_tokens")]
    public void Load_BudgetNonIntegerValue_FailsAsIntegerValue(string key)
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                {key}: 'lots'
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains($"models[0].{key} must be an integer value", error.Message);
    }

    [Fact]
    public void ReloadTable_BothBudgets_AreClassifiedNewRequestsOnly()
    {
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, ConfigMutationTable.Classify("models[0].max_reasoning_tokens"));
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, ConfigMutationTable.Classify("models[0].max_response_tokens"));
    }

    [Fact]
    public void ReloadTable_MaxReasoningTokens_Diff_NamesFieldOldNewAndClass()
    {
        string modelPath = TempModelFile();
        LairConfig current = LoadWithBudgets(modelPath, reasoning: null, response: 512);
        LairConfig candidate = LoadWithBudgets(modelPath, reasoning: 1024, response: 512);

        FieldChange change = Assert.Single(ConfigMutationTable.Diff(current, candidate));
        Assert.Equal("models[0].max_reasoning_tokens", change.Field);
        Assert.Equal("(unset)", change.OldValue);
        Assert.Equal("1024", change.NewValue);
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, change.MutationClass);
    }

    [Fact]
    public void ReloadTable_MaxResponseTokens_Diff_NamesRaisedBudgetBeingLowered()
    {
        string modelPath = TempModelFile();
        LairConfig current = LoadWithBudgets(modelPath, reasoning: 2048, response: 1024);
        LairConfig candidate = LoadWithBudgets(modelPath, reasoning: 2048, response: null);

        FieldChange change = Assert.Single(ConfigMutationTable.Diff(current, candidate));
        Assert.Equal("models[0].max_response_tokens", change.Field);
        Assert.Equal("1024", change.OldValue);
        Assert.Equal("(unset)", change.NewValue);
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, change.MutationClass);
    }

    [Fact]
    public void ReloadTable_EqualBudgets_ProduceNoChange()
    {
        string modelPath = TempModelFile();
        LairConfig current = LoadWithBudgets(modelPath, reasoning: 1024, response: 512);
        LairConfig candidate = LoadWithBudgets(modelPath, reasoning: 1024, response: 512);

        Assert.Empty(ConfigMutationTable.Diff(current, candidate));
    }

    [Fact]
    public void ReloadTable_Budget_ExtractEdits_NamesFieldAndClass_OnInvalidCandidate()
    {
        string modelPath = TempModelFile();
        LairConfig current = LoadWithBudgets(modelPath, reasoning: null, response: null);

        // The candidate carries a negative budget, so the loader rejects it whole; the raw-scalar extraction still names
        // the changed key with its mutation class, exactly as it does for every other known key.
        string path = WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
                max_response_tokens: -7
            """);

        Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(path));

        FieldChange edit = Assert.Single(ConfigMutationTable.ExtractEdits(current, path));
        Assert.Equal("models[0].max_response_tokens", edit.Field);
        Assert.Equal("(unset)", edit.OldValue);
        Assert.Equal("-7", edit.NewValue);
        Assert.Equal(ConfigMutationClass.NewRequestsOnly, edit.MutationClass);
    }

    // ---------------------------------------------------------------- fixture plumbing

    private LairConfig LoadWithBudgets(string modelPath, int? reasoning, int? response) =>
        YamlConfigLoader.Load(WriteConfig(BudgetYaml(modelPath, reasoning, response)));

    private static string BudgetYaml(string modelPath, int? reasoning, int? response)
    {
        List<string> lines =
        [
            "models:",
            "  - name: qwen-test",
            $"    path: '{modelPath}'",
        ];

        if (reasoning is not null)
        {
            lines.Add($"    max_reasoning_tokens: {reasoning}");
        }

        if (response is not null)
        {
            lines.Add($"    max_response_tokens: {response}");
        }

        return string.Join('\n', lines);
    }

    private string WriteConfig(string yaml)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-token-budget-config-{Guid.NewGuid():N}.yaml");
        File.WriteAllText(path, yaml);
        _tempPaths.Add(path);
        return path;
    }

    private string TempModelFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-token-budget-model-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, [0x47, 0x47, 0x55, 0x46]);
        _tempPaths.Add(path);
        return path;
    }
}
