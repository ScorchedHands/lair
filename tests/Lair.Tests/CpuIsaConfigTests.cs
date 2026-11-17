using System.Runtime.Intrinsics.X86;
using Lair.Backends.Cpu;
using Lair.Host.Config;
using Lair.Http;
using Lair.Models;

namespace Lair.Tests;

/// <summary>
/// The <c>engine.cpu.isa</c> axis end to end below model open: the loader accepts <c>auto</c>, <c>avx2</c>, and
/// <c>avx-vnni</c> under a nested <c>cpu</c> block and defaults to <c>auto</c>; an unrecognized value fails naming the
/// served model, the key as declared, and the accepted set; the hot-reload table classifies the field restart-only because
/// the selection is compiled into the kernels of an opened backend. The resolution rule itself — auto takes the highest path a posture reports, a forced path resolves to itself or not at all — and what each outcome means for an opening backend are asserted against stated postures rather than this machine's CPU, so every gate here answers the same way on any host.
/// </summary>
[Trait("Category", "Config")]
public sealed class CpuIsaConfigTests : IDisposable
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
    [InlineData("avx2")]
    [InlineData("avx-vnni")]
    public void Load_CpuIsa_AcceptsEachSelection(string cpuIsa)
    {
        // The schema accepts every selection: which one this host can run is the backend's answer when it opens, and a forced
        // ISA that host cannot report fails there naming what is missing — never here in the loader.
        LairConfig config = YamlConfigLoader.Load(WriteConfig(BuildYaml(cpuIsa)));

        Assert.Equal(cpuIsa, config.Engine.Cpu.Isa);
    }

    [Fact]
    public void Load_CpuIsa_AbsentDefaultsToAuto()
    {
        // No cpu block at all is today's posture: the highest path this host reports, which is what every unpinned engine has run.
        LairConfig config = YamlConfigLoader.Load(WriteConfig(BuildYaml(null)));

        Assert.Equal("auto", config.Engine.Cpu.Isa);
    }

    [Theory]
    [InlineData("avx512")]
    [InlineData("vnni")]
    [InlineData("AVX-VNNI")] // the vocabulary is exact-case, like every other choice key here
    [InlineData("")]
    public void Load_UnknownCpuIsa_FailsNamingModelAndValue(string cpuIsa)
    {
        string yaml = BuildYaml(cpuIsa);

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains($"engine.cpu.isa '{cpuIsa}'", error.Message);
        Assert.Contains("model 'qwen-test'", error.Message); // the failure names the model the selection governs
        Assert.Contains("'cpu.isa' accepts only 'auto', 'avx2', 'avx-vnni'", error.Message);
    }

    [Fact]
    public void Load_CpuIsa_NonScalarValue_FailsAsTextValue()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            engine:
              cpu:
                isa:
                  choice: auto
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("engine.cpu.isa must be a text value", error.Message);
    }

    [Fact]
    public void Load_CpuIsa_UnknownNestedKey_FailsNamingTheBlock()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            engine:
              cpu:
                quantization: avx2
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("Unknown key 'quantization'", error.Message);
        Assert.Contains("'engine.cpu'", error.Message); // the cpu block owns its own vocabulary, like every other nesting here
    }

    [Fact]
    public void ReloadTable_CpuIsa_IsClassifiedRestartOnly()
    {
        Assert.Equal(ConfigMutationClass.RestartOnly, ConfigMutationTable.Classify("engine.cpu.isa"));

        string modelPath = TempModelFile();
        LairConfig current = LoadCpuConfig(modelPath, "auto");
        LairConfig candidate = LoadCpuConfig(modelPath, "avx2");

        FieldChange change = Assert.Single(ConfigMutationTable.Diff(current, candidate));
        Assert.Equal("engine.cpu.isa", change.Field);
        Assert.Equal("auto", change.OldValue);
        Assert.Equal("avx2", change.NewValue);
        Assert.Equal(ConfigMutationClass.RestartOnly, change.MutationClass); // a live backend never re-reads the axis: only an engine open resolves it again
    }

    [Fact]
    public void ReloadTable_CpuIsa_DeclaredAutoAgainstAbsentKey_ProducesNoChange()
    {
        // Stating 'auto' is the posture an absent key already has, so a reload that spells the default changes nothing.
        string modelPath = TempModelFile();
        LairConfig current = LoadCpuConfig(modelPath, null);
        LairConfig candidate = LoadCpuConfig(modelPath, "auto");

        Assert.Empty(ConfigMutationTable.Diff(current, candidate));
    }

    [Fact]
    public void ReloadTable_CpuIsa_UnknownValue_IsExtractedAsRestartOnly()
    {
        // An unrecognized selection is rejected by the loader outright; the best-effort extraction still names the edited field
        // with its mutation class so a refused edit reports as restart-only rather than falling through unclassified.
        string modelPath = TempModelFile();
        LairConfig current = LoadCpuConfig(modelPath, "auto");

        // The refused edit changes only the ISA line: every other declared key matches what is running, so the extraction names exactly one field.
        string path = WriteConfig(BuildYaml("quantized", modelPath));
        Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(path));

        FieldChange edit = Assert.Single(ConfigMutationTable.ExtractEdits(current, path));
        Assert.Equal("engine.cpu.isa", edit.Field);
        Assert.Equal("auto", edit.OldValue);
        Assert.Equal("quantized", edit.NewValue);
        Assert.Equal(ConfigMutationClass.RestartOnly, edit.MutationClass);
    }

    // ---------------------------------------------------------------- resolution rule over stated postures

    [Fact]
    public void Probe_AutoTakesTheHighestPathAPostureReports()
    {
        Assert.True(CpuIsaProbe.TryResolve(CpuIsa.Auto, new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: true), out IntegerDotPath bothPaths, out string? missingBoth));
        Assert.Equal(IntegerDotPath.Vnni, bothPaths); // AVX-VNNI over AVX2 whenever a posture reports it beside the baseline
        Assert.Null(missingBoth);

        Assert.True(CpuIsaProbe.TryResolve(CpuIsa.Auto, new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: false), out IntegerDotPath vectorOnly, out string? missingVector));
        Assert.Equal(IntegerDotPath.Avx2, vectorOnly); // the pair-then-quadruple dot when no vpdpbusd is reported
        Assert.Null(missingVector);

        Assert.True(CpuIsaProbe.TryResolve(CpuIsa.Auto, new CpuFeaturePosture(Avx2Available: false, AvxVnniAvailable: false), out IntegerDotPath neitherPath, out string? missingNeither));
        Assert.Equal(IntegerDotPath.Scalar, neitherPath); // nothing vector reported: the reference dots run, as they always have
        Assert.Null(missingNeither);

        // A posture that states vpdpbusd without stating the baseline its staging and sign fold use resolves to no vector path at all.
        Assert.True(CpuIsaProbe.TryResolve(CpuIsa.Auto, new CpuFeaturePosture(Avx2Available: false, AvxVnniAvailable: true), out IntegerDotPath inconsistentPath, out _));
        Assert.Equal(IntegerDotPath.Scalar, inconsistentPath);
    }

    [Theory]
    [InlineData(CpuIsa.AvxVnni)]
    [InlineData(CpuIsa.Avx2)]
    public void Probe_ForcedPathResolvesToItselfOrNotAtAll(CpuIsa requested)
    {
        // A posture that can run a forced selection resolves it to itself...
        CpuFeaturePosture capable = new(Avx2Available: true, AvxVnniAvailable: requested == CpuIsa.AvxVnni);

        Assert.True(CpuIsaProbe.TryResolve(requested, capable, out IntegerDotPath runnable, out string? missingRunnable));
        Assert.Equal(requested == CpuIsa.AvxVnni ? IntegerDotPath.Vnni : IntegerDotPath.Avx2, runnable);
        Assert.Null(missingRunnable);

        // ...and a posture that reports neither vector path refuses it naming the CPU feature needed. Nothing here answers an explicit choice with some other encoding that happens to be available; only the auto branch may land on scalar dots.
        Assert.False(CpuIsaProbe.TryResolve(requested, new CpuFeaturePosture(Avx2Available: false, AvxVnniAvailable: false), out _, out string? missing));

        Assert.Contains(CpuFeaturePosture.FeatureNameFor(requested), missing); // a forced vpdpbusd with no baseline at all names the baseline beside it, which its own dedicated gate below pins
    }

    [Fact]
    public void Probe_ForcedVnniWithoutBaselineNamesTheBaselineToo()
    {
        Assert.False(CpuIsaProbe.TryResolve(CpuIsa.AvxVnni, new CpuFeaturePosture(Avx2Available: false, AvxVnniAvailable: true), out _, out string? missing));

        // vpdpbusd staging and sign folding still run on the AVX2 baseline this backend has always required, so a posture
        // claiming only VNNI names both.
        Assert.Equal("AVX-VNNI (and the AVX2 baseline its staging and sign fold use)", missing);
    }

    [Fact]
    public void Probe_DetectNeverClaimsVnniWithoutItsBaseline()
    {
        CpuFeaturePosture detected = CpuIsaProbe.Detect();

        // The posture this process reports is the only one a default-opened backend resolves against: it can never state an
        // integer-dot path whose staging intrinsics are missing here.
        Assert.True(!detected.AvxVnniAvailable || (detected.Avx2Available && Avx2.IsSupported && Fma.IsSupported && Ssse3.IsSupported));

        if (AvxVnni.IsSupported)
        {
            Assert.True(detected.AvxVnniAvailable); // a host reporting vpdpbusd reports it to the axis too, not only to the runtime class
        }
    }

    // ---------------------------------------------------------------- what each outcome means for an opening backend

    [Theory]
    [InlineData(true, true, IntegerDotPath.Vnni)]
    [InlineData(true, false, IntegerDotPath.Avx2)]
    [InlineData(false, false, IntegerDotPath.Scalar)]
    public void BackendOpen_AutoResolvesTheHighestPathThePostureReports(bool avx2Available, bool vnniAvailable, IntegerDotPath expected)
    {
        using CpuBackend backend = new(threads: 1, cpuFeatures: new CpuFeaturePosture(avx2Available, vnniAvailable));

        Assert.Equal(CpuIsa.Auto, backend.RequestedCpuIsa); // the axis as declared
        Assert.Equal(expected, backend.ResolvedIntegerDotPath); // and exactly one encoding resolved from it for this backend's whole lifetime
    }

    [Fact]
    public void BackendOpen_ForceUnavailableRefusesLoadNamingMissingFeature()
    {
        CpuIsaUnavailableException vnniEx = Assert.Throws<CpuIsaUnavailableException>(
            () => new CpuBackend(threads: 1, requestedIsa: CpuIsa.AvxVnni, cpuFeatures: new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: false)));

        Assert.Equal(CpuIsa.AvxVnni, vnniEx.Requested);
        Assert.Equal("AVX-VNNI", vnniEx.MissingFeature); // the failure names what this CPU does not report instead of substituting AVX2 or scalar
        Assert.Contains("engine.cpu.isa 'avx-vnni' is unavailable on this host: the CPU does not report AVX-VNNI", vnniEx.Message);

        CpuIsaUnavailableException avx2Ex = Assert.Throws<CpuIsaUnavailableException>(
            () => new CpuBackend(threads: 1, requestedIsa: CpuIsa.Avx2, cpuFeatures: new CpuFeaturePosture(Avx2Available: false, AvxVnniAvailable: true)));

        Assert.Equal("AVX2", avx2Ex.MissingFeature);
    }

    [Fact]
    public void BackendOpen_ForceOnDisabledVectorPosture_Refuses()
    {
        // Forcing an ISA asks for a vector integer dot; the no-vector posture (the scalar-oracle knob) never runs one, so the two
        // declarations contradict each other and the open refuses instead of resolving some third path.
        ArgumentException ex = Assert.Throws<ArgumentException>(
            () => new CpuBackend(threads: 1, allowAvx2: false, requestedIsa: CpuIsa.AvxVnni, cpuFeatures: new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: true)));

        Assert.Contains("cannot be forced on a backend opened with its vector posture disabled", ex.Message);
    }

    [Fact]
    public void BackendOpen_ScalarOracleKnob_RunsReferenceDotsWhateverAutoResolvedTo()
    {
        // allowAvx2:false is the test knob for reference-dot comparison: it keeps running scalar dots on a host whose auto
        // selection would otherwise land on vpdpbusd, without pretending that posture was an ISA declaration.
        using CpuBackend backend = new(threads: 1, allowAvx2: false);

        Assert.Equal(CpuIsa.Auto, backend.RequestedCpuIsa);
        Assert.Equal(IntegerDotPath.Scalar, backend.ResolvedIntegerDotPath);
    }

    [Fact]
    public void BackendOpen_AutoOnThisHostResolvesWhatTheProbeReports()
    {
        // The default-open posture is what every unpinned engine runs today: the highest path this process reports, read once.
        using CpuBackend backend = new(threads: 1);

        IntegerDotPath expected = CpuIsaProbe.Detect().AvxVnniAvailable ? IntegerDotPath.Vnni : IntegerDotPath.Avx2;
        Assert.Equal(expected, backend.ResolvedIntegerDotPath);
    }

    [Fact]
    public void EngineOpen_ForceUnavailableRefusesLoadNamingMissingFeature()
    {
        string modelPath = SyntheticLlamaGguf.Write("isa-load-refusal");
        _tempPaths.Add(modelPath);

        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: isa-load-refusal
                path: '{modelPath}'
            engine:
              cpu:
                isa: avx-vnni
            """));

        ModelException ex = Assert.Throws<ModelException>(
            () => GenerationService.Open(
                config,
                logger: null,
                ledger: null,
                compute: null,
                transport: null,
                cpuFeatures: new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: false)));

        Assert.Contains("isa-load-refusal", ex.Message);
        Assert.Contains("'engine.cpu.isa' 'avx-vnni'", ex.Message); // the failure names both the declared key and its value, as load refusals do here
        Assert.Contains("AVX-VNNI", ex.Message); // the model open refuses before any backend can substitute another integer-dot path
    }

    // ---------------------------------------------------------------- fixture plumbing

    private LairConfig LoadCpuConfig(string modelPath, string? cpuIsa) => YamlConfigLoader.Load(WriteConfig(BuildYaml(cpuIsa, modelPath)));

    private string BuildYaml(string? cpuIsa, string? modelPath = null)
    {
        // The baseline posture declares no engine key at all — today's default selection — while every compared run states it. A test that only reads config values gets a real (stub) file so the loader reaches its own key validation.
        string engineBlock = cpuIsa is null ? string.Empty : $"\nengine:\n  cpu:\n    isa: {cpuIsa}";

        return $"""
            models:
              - name: qwen-test
                path: '{modelPath ?? TempModelFile()}'
            {engineBlock.TrimStart()}
            """;
    }

    private string WriteConfig(string yaml)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-cpu-isa-config-{Guid.NewGuid():N}.yaml");
        File.WriteAllText(path, yaml);
        _tempPaths.Add(path);
        return path;
    }

    private string TempModelFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-cpu-isa-model-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, [0x47, 0x47, 0x55, 0x46]);
        _tempPaths.Add(path);
        return path;
    }
}
