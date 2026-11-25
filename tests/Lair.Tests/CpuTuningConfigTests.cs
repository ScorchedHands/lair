using Lair.Host.Config;
using Lair.Http;

namespace Lair.Tests;

/// <summary>
/// The <c>engine.cpu.tuning</c> axis end to end below model open: the loader accepts the four integer keys under a nested
/// <c>cpu.tuning</c> block and defaults every absent one to null — posture-resolved, not fixed to a value — while an out-of-span
/// or non-integer value fails naming the qualified key, its value, and the accepted span, the same refusal shape the backend seam
/// applies. The hot-reload table classifies every one restart-only because each pin resolves once against detected topology and
/// effective worker count when a backend opens, beside the ISA it shares a block with.
/// </summary>
[Trait("Category", "Config")]
public sealed class CpuTuningConfigTests : IDisposable
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

    // ---------------------------------------------------------------- loader acceptance and defaults

    [Fact]
    public void Load_CpuTuning_AcceptsEveryKeyWithStatedValues()
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            engine:
              cpu:
                tuning:
                  row_block: 16
                  parallel_row_threshold: 256
                  chain_group: 8
                  max_scratch_cols: 16384
            """));

        Assert.Equal(16, config.Engine.Cpu.Tuning.RowBlock);
        Assert.Equal(256, config.Engine.Cpu.Tuning.ParallelRowThreshold);
        Assert.Equal(8, config.Engine.Cpu.Tuning.ChainGroup);
        Assert.Equal(16384, config.Engine.Cpu.Tuning.MaxScratchCols);
    }

    [Fact]
    public void Load_CpuTuning_AbsentKeysDeclareNothing()
    {
        // No tuning block at all is today's posture: every knob stays posture-resolved at model open rather than pinned to one machine's measurement.
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            """));

        Assert.NotNull(config.Engine.Cpu.Tuning);
        Assert.Null(config.Engine.Cpu.Tuning.RowBlock);
        Assert.Null(config.Engine.Cpu.Tuning.ParallelRowThreshold);
        Assert.Null(config.Engine.Cpu.Tuning.ChainGroup);
        Assert.Null(config.Engine.Cpu.Tuning.MaxScratchCols);
    }

    [Fact]
    public void Load_CpuTuning_PartialBlockPinsOnlyStatedKnobs()
    {
        // A stated row_block beside an omitted threshold is a legal posture: the pin applies and the sibling keeps resolving from detected posture.
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            engine:
              cpu:
                tuning:
                  row_block: 4
            """));

        Assert.Equal(4, config.Engine.Cpu.Tuning.RowBlock);
        Assert.Null(config.Engine.Cpu.Tuning.ParallelRowThreshold);
    }

    [Fact]
    public void Load_CpuTuning_UnknownKey_FailsNamingTheBlock()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            engine:
              cpu:
                tuning:
                  rowblocks: 16
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("Unknown key 'rowblocks'", error.Message);
        Assert.Contains("'engine.cpu.tuning'", error.Message); // the tuning block owns its own vocabulary, like every other nesting here
    }

    [Fact]
    public void Load_CpuTuning_NonScalarValue_FailsAsIntegerValue()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            engine:
              cpu:
                tuning:
                  row_block:
                    size: 16
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("engine.cpu.tuning.row_block must be an integer value", error.Message);
    }

    [Theory]
    [InlineData("row_block: 0", "engine.cpu.tuning.row_block 0 is invalid: 'cpu.tuning.row_block' must be between 1 and 64.")]
    [InlineData("row_block: 65", "engine.cpu.tuning.row_block 65 is invalid: 'cpu.tuning.row_block' must be between 1 and 64.")]
    [InlineData("parallel_row_threshold: 0", "engine.cpu.tuning.parallel_row_threshold 0 is invalid: 'cpu.tuning.parallel_row_threshold' must be 1 or greater.")]
    [InlineData("chain_group: 9", "engine.cpu.tuning.chain_group 9 is invalid: 'cpu.tuning.chain_group' must be between 1 and 8 — a chain group wider than eight spills accumulators past the vector posture's register budget.")]
    [InlineData("max_scratch_cols: 128", "engine.cpu.tuning.max_scratch_cols 128 is invalid: 'cpu.tuning.max_scratch_cols' must be between 512 and 65536 — wider staging no longer stays cache-resident beside the packed row it replaces.")]
    [InlineData("max_scratch_cols: 70000", "engine.cpu.tuning.max_scratch_cols 70000 is invalid: 'cpu.tuning.max_scratch_cols' must be between 512 and 65536 — wider staging no longer stays cache-resident beside the packed row it replaces.")]
    public void Load_CpuTuning_ValueOutsideItsSpan_FailsNamingQualifiedKeyAndSpan(string tuningLine, string expectedMessage)
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            engine:
              cpu:
                tuning:
                  {tuningLine}
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains(expectedMessage, error.Message);
    }

    [Fact]
    public void Load_CpuTuning_NonIntegerText_FailsAsIntegerValue()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            engine:
              cpu:
                tuning:
                  chain_group: wide
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("engine.cpu.tuning.chain_group must be an integer value", error.Message);
    }

    // ---------------------------------------------------------------- hot-reload classification

    [Theory]
    [InlineData("engine.cpu.tuning.row_block")]
    [InlineData("engine.cpu.tuning.parallel_row_threshold")]
    [InlineData("engine.cpu.tuning.chain_group")]
    [InlineData("engine.cpu.tuning.max_scratch_cols")]
    public void ReloadTable_CpuTuning_IsClassifiedRestartOnly(string field)
    {
        Assert.Equal(ConfigMutationClass.RestartOnly, ConfigMutationTable.Classify(field));
    }

    [Fact]
    public void ReloadTable_CpuTuning_DiffNamesFieldWithOldAndNewValues()
    {
        string modelPath = TempModelFile();
        LairConfig current = LoadTuningConfig(modelPath, "row_block: 16");
        LairConfig candidate = LoadTuningConfig(modelPath, "row_block: 8");

        FieldChange change = Assert.Single(ConfigMutationTable.Diff(current, candidate));
        Assert.Equal("engine.cpu.tuning.row_block", change.Field);
        Assert.Equal("16", change.OldValue);
        Assert.Equal("8", change.NewValue);
        Assert.Equal(ConfigMutationClass.RestartOnly, change.MutationClass); // a live backend never re-reads a pin: only an engine open resolves it against topology again
    }

    [Fact]
    public void ReloadTable_CpuTuning_AbsentAgainstDeclared_IsAChange()
    {
        // Unlike the ISA, an absent tuning key carries no value at all — resolving from posture is a different posture than pinning the same number a measurement would have produced.
        string modelPath = TempModelFile();
        LairConfig current = LoadTuningConfig(modelPath, null);
        LairConfig candidate = LoadTuningConfig(modelPath, "row_block: 16");

        FieldChange change = Assert.Single(ConfigMutationTable.Diff(current, candidate));
        Assert.Equal("engine.cpu.tuning.row_block", change.Field);
        Assert.Equal("(unset)", change.OldValue);
        Assert.Equal("16", change.NewValue);
    }

    [Fact]
    public void ReloadTable_CpuTuning_InvalidValue_IsExtractedAsRestartOnly()
    {
        // A refused pin is rejected by the loader outright; the best-effort extraction still names the edited field with its mutation class so a refused edit reports as restart-only rather than falling through unclassified.
        string modelPath = TempModelFile();
        LairConfig current = LoadTuningConfig(modelPath, "row_block: 16");

        // The refused edit changes only the row block line: every other declared key matches what is running, so the extraction names exactly one field.
        string path = WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
            engine:
              cpu:
                tuning:
                  row_block: 0
            """);
        Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(path));

        FieldChange edit = Assert.Single(ConfigMutationTable.ExtractEdits(current, path));
        Assert.Equal("engine.cpu.tuning.row_block", edit.Field);
        Assert.Equal("16", edit.OldValue);
        Assert.Equal("0", edit.NewValue);
        Assert.Equal(ConfigMutationClass.RestartOnly, edit.MutationClass);
    }

    // ---------------------------------------------------------------- fixture plumbing

    private LairConfig LoadTuningConfig(string modelPath, string? tuningLine)
    {
        // A null tuning line declares the whole block absent; a stated line pins one knob beside the rest resolving.
        string engineBlock = tuningLine is null ? string.Empty : $"\nengine:\n  cpu:\n    tuning:\n      {tuningLine}";

        return YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{modelPath}'
            {engineBlock.TrimStart('\n')}
            """));
    }

    private string WriteConfig(string yaml)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-cpu-tuning-config-{Guid.NewGuid():N}.yaml");
        File.WriteAllText(path, yaml);
        _tempPaths.Add(path);
        return path;
    }

    private string TempModelFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-cpu-tuning-model-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, [0x47, 0x47, 0x55, 0x46]);
        _tempPaths.Add(path);
        return path;
    }
}
