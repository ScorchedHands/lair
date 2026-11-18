using Lair.Backends.Cpu;
using Lair.Host.Config;
using Lair.Http;
using Lair.Kv;
using Lair.Models;
using Microsoft.Extensions.Logging;

namespace Lair.Tests;

/// <summary>
/// The cold-path KV format axis: config resolves the pair, reload classifies each side as restart-only, direct pool
/// sizing proves the footprint formulas, and model open refuses a quantized pair its closed attention route cannot
/// address — an unsupported pair or a head geometry the quant kernels cannot slice — before the pool is constructed.
/// </summary>
[Trait("Category", "Config")]
[Trait("Category", "Kv")]
public sealed class KvFormatDescriptorTests : IDisposable
{
    private readonly List<string> _tempPaths = [];
    private string? _modelPath;

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

    [Fact]
    public void Load_KvQuantAbsent_DefaultsCanonicalF32_AndKeepsExistingKvBlocksSizing()
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: kv-format-test
                path: '{ModelPath()}'
                max_context_tokens: 512
                kv_blocks: 8

            """));

        ModelConfig model = Assert.Single(config.Models);

        Assert.True(model.KvQuant.IsFloatStorage);
        Assert.Equal(KvFormatPair.Default, model.KvQuant);
        Assert.Equal("f32", KvSideFormatNames.Of(model.KvQuant.K));
        Assert.Equal("f32", KvSideFormatNames.Of(model.KvQuant.V));
        Assert.Equal(8, model.KvBlocks);
    }

    [Theory]
    [InlineData("f32", "fp32", "f32", "f32")]
    [InlineData("f16", "fp16", "f16", "f16")]
    [InlineData("q8", "int8", "q8_0", "q8_0")]
    [InlineData("q8_0", "q4_0", "q8_0", "q4_0")]
    public void Load_KvQuantNamedPairs_ResolveCanonicalSides(string declaredK, string declaredV, string expectedK, string expectedV)
    {
        LairConfig config = LoadQuantConfig(declaredK, declaredV);
        ModelConfig model = Assert.Single(config.Models);

        Assert.Equal(expectedK, KvSideFormatNames.Of(model.KvQuant.K));
        Assert.Equal(expectedV, KvSideFormatNames.Of(model.KvQuant.V));
    }

    [Fact]
    public void Load_KvQuantPartialPair_DefaultsTheMissingSide()
    {
        LairConfig config = LoadQuantConfig("f16", "f32");
        ModelConfig model = Assert.Single(config.Models);

        Assert.Equal("f16", KvSideFormatNames.Of(model.KvQuant.K));
        Assert.Equal("f32", KvSideFormatNames.Of(model.KvQuant.V));
    }

    [Theory]
    [InlineData("q9")]
    [InlineData("Q8_0")]
    [InlineData("int7")]
    public void Load_KvQuantInvalidSide_FailsNamingQualifiedPathValueAndAcceptedVocabulary(string declaredK)
    {
        ConfigException error = Assert.Throws<ConfigException>(() => LoadQuantConfig(declaredK, "f32"));

        Assert.Contains($"models[0].kv.quant.k '{declaredK}' is invalid", error.Message);
        Assert.Contains("'k' accepts only 'f32', 'fp32', 'f16', 'fp16', 'q8_0', 'q8', 'int8', 'q4_0'", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_KvQuantDroppedQ4Pair_RefusesNamingBothQualifiedSides()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => LoadQuantConfig("q4_0", "q4_0"));

        Assert.Contains("models[0].kv.quant.k declares 'q4_0'", error.Message);
        Assert.Contains("models[0].kv.quant.v declares 'q4_0'", error.Message);
        Assert.Contains("q4_0/q4_0 is the dropped KV pair", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_UnknownKeyUnderKv_FailsQualifiedKvPath()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: kv-format-test
                path: '{ModelPath()}'
                max_context_tokens: 512
                kv_blocks: 8
                kv:
                  bad: x

            """)));

        Assert.Contains("Unknown key 'bad' under models[0].kv", error.Message);
        Assert.Contains("allowed keys are: quant", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_UnknownKeyUnderKvQuant_FailsQualifiedQuantPath()
    {
        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: kv-format-test
                path: '{ModelPath()}'
                max_context_tokens: 512
                kv_blocks: 8
                kv:
                  quant:
                    k: f32
                    v: f32
                    bad: x

            """)));

        Assert.Contains("Unknown key 'bad' under models[0].kv.quant", error.Message);
        Assert.Contains("allowed keys are: k, v", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ReloadTable_KvQuantSides_ClassifyRestartOnly()
    {
        Assert.Equal(ConfigMutationClass.RestartOnly, ConfigMutationTable.Classify("models[0].kv.quant.k"));
        Assert.Equal(ConfigMutationClass.RestartOnly, ConfigMutationTable.Classify("models[0].kv.quant.v"));
    }

    [Fact]
    public void ReloadDiff_KvQuantSides_ReportCanonicalRestartOnlyChanges()
    {
        LairConfig current = LoadQuantConfig("f32", "f32");
        LairConfig candidate = LoadQuantConfig("q8_0", "q4_0");

        IReadOnlyList<FieldChange> changes = ConfigMutationTable.Diff(current, candidate);

        FieldChange k = SingleField(changes, "models[0].kv.quant.k");
        Assert.Equal("f32", k.OldValue);
        Assert.Equal("q8_0", k.NewValue);
        Assert.Equal(ConfigMutationClass.RestartOnly, k.MutationClass);

        FieldChange v = SingleField(changes, "models[0].kv.quant.v");
        Assert.Equal("f32", v.OldValue);
        Assert.Equal("q4_0", v.NewValue);
        Assert.Equal(ConfigMutationClass.RestartOnly, v.MutationClass);
    }

    [Fact]
    public void ReloadRawEdits_KvQuantSides_NameRestartOnlyChangesFromCandidateText()
    {
        LairConfig current = LoadQuantConfig("f32", "f32");
        string candidatePath = WriteQuantYaml("q8_0", "q4_0");

        IReadOnlyList<FieldChange> edits = ConfigMutationTable.ExtractEdits(current, candidatePath);

        FieldChange k = SingleField(edits, "models[0].kv.quant.k");
        Assert.Equal("f32", k.OldValue);
        Assert.Equal("q8_0", k.NewValue);
        Assert.Equal(ConfigMutationClass.RestartOnly, k.MutationClass);

        FieldChange v = SingleField(edits, "models[0].kv.quant.v");
        Assert.Equal("f32", v.OldValue);
        Assert.Equal("q4_0", v.NewValue);
        Assert.Equal(ConfigMutationClass.RestartOnly, v.MutationClass);
    }

    [Fact]
    public void Pool_Footprint_DefaultFp32_KeepsExistingConstructorFormula()
    {
        using CpuBackend backend = new();
        using BlockPool legacy = new(
            blockTokens: 4,
            layerCount: 2,
            kvHeads: 2,
            headDim: 3,
            capacityBlocks: 1,
            backend: backend,
            residency: BlockResidencyMode.Dram,
            deviceBackedBlocks: 0,
            mirrorOwner: null);
        using BlockPool explicitDefault = new(
            blockTokens: 4,
            layerCount: 2,
            kvHeads: 2,
            headDim: 3,
            capacityBlocks: 1,
            backend: backend,
            formats: KvFormatPair.Default);

        Assert.Equal(checked(2L * 2 * 4 * 6 * sizeof(float)), legacy.BytesPerBlock);
        Assert.Equal(legacy.BytesPerBlock, explicitDefault.BytesPerBlock);
        Assert.True(explicitDefault.Formats.IsFloatStorage);

        int id = explicitDefault.Allocate();
        Assert.Equal(96, explicitDefault.BlockView(id).Length);
        explicitDefault.Release(id);
    }

    [Fact]
    public void Pool_Footprint_Fp16_UsesHalfElementBytes_WithoutScalePadding()
    {
        using CpuBackend backend = new();
        using BlockPool pool = new(
            blockTokens: 4,
            layerCount: 2,
            kvHeads: 2,
            headDim: 3,
            capacityBlocks: 1,
            backend: backend,
            formats: new KvFormatPair(KvSideFormat.F16, KvSideFormat.F16));

        Assert.Equal(checked(2L * 2 * 4 * 6 * 2), pool.BytesPerBlock);

        int id = pool.Allocate();
        pool.Release(id);

        Assert.Throws<InvalidOperationException>(() => pool.BlockView(0));
        Assert.Throws<InvalidOperationException>(() => pool.MutableBlockSpan(0));
        Assert.Throws<InvalidOperationException>(() => pool.SlotSpan(0, 0, isV: false, offsetWithinBlock: 0));
    }

    [Fact]
    public void Pool_Footprint_Q8_0_UsesWholeGroupPayloadAndScaleBytes()
    {
        using CpuBackend backend = new();
        using BlockPool pool = new(
            blockTokens: 4,
            layerCount: 2,
            kvHeads: 2,
            headDim: 3,
            capacityBlocks: 1,
            backend: backend,
            formats: new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q8_0));

        Assert.Equal(checked(2L * 2 * 1 * 34), pool.BytesPerBlock);

        int id = pool.Allocate();
        pool.Release(id);

        InvalidOperationException blockView = Assert.Throws<InvalidOperationException>(() => pool.BlockView(0));
        Assert.Contains("k='q8_0' v='q8_0'", blockView.Message);
        Assert.Contains("no quantized-KV storage or attention execution path", blockView.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Pool_Footprint_AsymmetricK8V4_SumsEachSideWithItsOwnGroupBytes()
    {
        using CpuBackend backend = new();
        using BlockPool pool = new(
            blockTokens: 4,
            layerCount: 2,
            kvHeads: 2,
            headDim: 3,
            capacityBlocks: 1,
            backend: backend,
            formats: new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q4_0));

        Assert.Equal(checked(2L * (1 * 34 + 1 * 18)), pool.BytesPerBlock);

        int id = pool.Allocate();
        pool.Release(id);

        InvalidOperationException slotSpan = Assert.Throws<InvalidOperationException>(() => pool.SlotSpan(0, 0, isV: true, offsetWithinBlock: 0));
        Assert.Contains("k='q8_0' v='q4_0'", slotSpan.Message);
    }

    [Fact]
    public void Pool_Footprint_QuantizedGroups_RoundEachSideRegionUpToWhole32ValueGroups()
    {
        using CpuBackend backend = new();
        using BlockPool pool = new(
            blockTokens: 1,
            layerCount: 1,
            kvHeads: 1,
            headDim: 33,
            capacityBlocks: 1,
            backend: backend,
            formats: new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q4_0));

        Assert.Equal(checked(1L * (2 * 34 + 2 * 18)), pool.BytesPerBlock);
    }

    [Fact]
    public void EngineOpen_DefaultKvQuant_LogsCanonicalPair_AndOpensSuccessfully()
    {
        CapturingLogger logger = new();
        using GenerationService engine = GenerationService.Open(ConfigWith("kv-format-default", KvFormatPair.Default), logger);

        Assert.Contains(
            logger.Messages(LogLevel.Information),
            line => line.Contains("kv_quant model='kv-format-default' k=f32 v=f32", StringComparison.Ordinal));
    }

    [Fact]
    public void EngineOpen_QuantizedPairOverUnaddressableGeometry_RefusesNamingModelPairAndHeadDimensionBeforePoolConstruction()
    {
        CapturingLogger logger = new();

        // The synthetic fixture's head dimension is not a whole number of quant groups, so the closed quantized route
        // cannot address its head regions: the refusal names the served model, the pair, and the geometry it cannot
        // carry — before any pool is constructed, and never as a silent fall back to the FP32 posture.
        ModelException error = Assert.Throws<ModelException>(
            () => GenerationService.Open(ConfigWith("kv-format-k8v4", new KvFormatPair(KvSideFormat.Q8_0, KvSideFormat.Q4_0)), logger));

        Assert.Contains("model 'kv-format-k8v4'", error.Message);
        Assert.Contains("k='q8_0' v='q4_0'", error.Message);
        Assert.Contains("head dimension", error.Message);
        Assert.Contains("whole 32-value groups per head region", error.Message);

        Assert.Contains(
            logger.Messages(LogLevel.Information),
            line => line.Contains("kv_quant model='kv-format-k8v4' k=q8_0 v=q4_0", StringComparison.Ordinal));
    }

    private LairConfig LoadQuantConfig(string k, string v) => YamlConfigLoader.Load(WriteQuantYaml(k, v));

    private string WriteQuantYaml(string k, string v) => WriteConfig($"""
        models:
          - name: kv-format-test
            path: '{ModelPath()}'
            max_context_tokens: 512
            kv_blocks: 8
            kv:
              quant:
                k: {k}
                v: {v}

        """);

    private LairConfig ConfigWith(string modelName, KvFormatPair formats)
    {
        string modelPath = SyntheticLlamaGguf.Write(modelName);
        _tempPaths.Add(modelPath);

        return new LairConfig
        {
            Server = new ServerConfig(),
            Engine = new EngineConfig { Threads = 1 },
            Models =
            [
                new ModelConfig
                {
                    Name = modelName,
                    Path = modelPath,
                    MaxContextTokens = 512,
                    KvBlocks = 8,
                    KvQuant = formats,
                },
            ],
        };
    }

    private static FieldChange SingleField(IReadOnlyList<FieldChange> changes, string field)
    {
        foreach (FieldChange change in changes)
        {
            if (change.Field == field)
            {
                return change;
            }
        }

        throw new InvalidOperationException($"the reload table reported no change for '{field}'");
    }

    private string WriteConfig(string yaml)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-kv-format-config-{Guid.NewGuid():N}.yaml");
        File.WriteAllText(path, yaml);
        _tempPaths.Add(path);
        return path;
    }

    private string ModelPath()
    {
        if (_modelPath is null)
        {
            _modelPath = SyntheticLlamaGguf.Write("kv-format-test");
            _tempPaths.Add(_modelPath);
        }

        return _modelPath;
    }

    private sealed class CapturingLogger : ILogger
    {
        private readonly List<(LogLevel Level, string Message)> _entries = [];

        public IReadOnlyList<string> Messages(LogLevel level) =>
            [.. _entries.Where(entry => entry.Level == level).Select(entry => entry.Message)];

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
        {
            ArgumentNullException.ThrowIfNull(formatter);
            _entries.Add((logLevel, formatter(state, exception)));
        }
    }
}
