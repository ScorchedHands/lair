using Lair.Backends.Cpu;
using Lair.Host.Config;
using Lair.Http;
using Lair.Http.Monitoring;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// What a resolved <c>engine.prefill_kernel</c> selection does at model open on the pinned SmolLM fixture: the two
/// reserved window shapes are refused naming the value and the kernel the forward engine lacks — never fallen back to
/// the route that happens to work — while an explicit <c>gemv</c> opens, forwards, and serves exactly what the absent
/// key does today. The resolved selection also reaches the monitor snapshot's configuration-backed facts so a host can
/// read which route an engine actually runs. Runs inside the serialized real-model lane with the parity suites.
/// </summary>
[Trait("Category", "Config")]
[Collection("model-loaded")] // shares one serialized lane with the other real-model suites: two engines mapping 270 MB at once is not what this box should be doing
public sealed class PrefillKernelServeTests
{
    private const string ModelName = "smollm-prefill-kernel";

    [Theory]
    [InlineData("tiled")]
    [InlineData("gemm")]
    public async Task GenerationServiceOpen_ReservedSelection_RefusesAtModelOpenNamingValueAndMissingKernel(string selection)
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        ModelException error = await Assert.ThrowsAsync<ModelException>(() => Task.Run(() => OpenEngine(selection)));

        Assert.Contains($"prefill kernel '{selection}' is not implemented by the forward engine", error.Message);
        Assert.Contains($"no {selection} attention/weight path exists in this build", error.Message); // names what is missing, not just the value
        Assert.Contains("only gemv executes today", error.Message);
    }

    [Theory]
    [InlineData(PrefillKernel.Tiled)]
    [InlineData(PrefillKernel.Gemm)]
    public void TransformerModelOpen_ReservedSelection_RefusesBeforeAnyTensorMaps(PrefillKernel selection)
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(ModelFixture.Resolve()!);

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, backend, maxContextTokens: 128, logLine: null, prefillKernel: selection));
        Assert.Contains($"prefill kernel '{PrefillKernelNames.Of(selection)}' is not implemented by the forward engine", error.Message);

        // The refusal is a capability report, never a tensor or geometry complaint.
        Assert.DoesNotContain("tensor", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task Prefill_ExplicitGemvSelection_ServesTheSameTokenStreamAsTheDefault()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        string prompt = "Describe a quiet harbor at dawn in one short paragraph.";

        GenerationResult defaultRun;
        GenerationResult explicitRun;
        MetricsSnapshot snapshotWithFacts;

        using (GenerationService plain = OpenEngine(kernelSelection: null))
        {
            defaultRun = await plain.GenerateAsync(Completion(prompt, 16), CancellationToken.None);
        }

        using (GenerationService gemv = OpenEngine("gemv"))
        {
            explicitRun = await gemv.GenerateAsync(Completion(prompt, 16), CancellationToken.None);

            EngineMetricsBus bus = gemv.AttachMetrics(NullLoggerFactory.Instance, TimeSpan.FromMilliseconds(50));
            GenerationResult measured = await gemv.GenerateAsync(Completion(prompt + " The gulls stayed quiet.", 8), CancellationToken.None);
            Assert.False(string.IsNullOrEmpty(measured.Text));

            for (int attempt = 0; attempt < 40 && bus.RingCount == 0; attempt++)
            {
                await Task.Delay(25); // the bus publishes on its own cadence; wait for a tick rather than assume one already ran
            }

            Assert.True(bus.RingCount > 0, "the metrics bus published no snapshot carrying the resolved selection");
            snapshotWithFacts = bus.SnapshotRing()[0];
        }

        // The gemv route is byte-identical to today's default behavior: same tokens, same budget accounting.
        Assert.Equal(defaultRun.Text, explicitRun.Text);
        Assert.Equal(defaultRun.CompletionTokens, explicitRun.CompletionTokens);
        Assert.Equal(defaultRun.FinishReason, explicitRun.FinishReason);

        // And the resolved value reaches a monitoring surface that reports configuration-backed facts.
        Assert.Equal("gemv", snapshotWithFacts.PrefillKernel);
    }

    [Fact]
    public void ApplyPrefillKernel_ReservedSelection_Refuses_AndAGemvRepointLandsOnTheVisibleRoute()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(ModelFixture.Resolve()!);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 128);

        // The default selection is what an engine opened without a stated kernel runs on.
        Assert.Equal(PrefillKernel.GemV, model.ResolvedPrefillKernel);

        ModelException error = Assert.Throws<ModelException>(() => model.ApplyPrefillKernel(PrefillKernel.Gemm));
        Assert.Contains("prefill kernel 'gemm' is not implemented by the forward engine", error.Message);
        Assert.Equal(PrefillKernel.GemV, model.ResolvedPrefillKernel); // a refused re-point leaves the running route untouched

        model.ApplyPrefillKernel(PrefillKernel.GemV);
        Assert.Equal(PrefillKernel.GemV, model.ResolvedPrefillKernel);
    }

    private static GenerationRequest Completion(string prompt, int maxTokens) => new()
    {
        Model = ModelName,
        IsChat = false,
        Prompt = prompt,
        MaxTokens = maxTokens,
        Temperature = 0f, // greedy: byte-comparable across engines and selections
    };

    private static GenerationService OpenEngine(string? kernelSelection) =>
        GenerationService.Open(YamlConfigLoader.Load(WriteConfig(BuildYaml(kernelSelection))));

    private static string BuildYaml(string? kernelSelection)
    {
        string modelPath = ModelFixture.Resolve()!;
        string kernelLine = kernelSelection is null ? string.Empty : $"\n  prefill_kernel: {kernelSelection}";

        return $"""
            engine:
              threads: 4
              max_batch_tokens: 8{kernelLine}
            models:
              - name: {ModelName}
                path: '{modelPath}'
                type: chat
                max_context_tokens: 512
                kv_blocks: 24
            """;
    }

    private static string WriteConfig(string yaml)
    {
        string dir = Path.Combine(Path.GetTempPath(), $"lair-prefill-kernel-serve-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, "lair.yaml");
        File.WriteAllText(path, yaml);

        AppDomain.CurrentDomain.ProcessExit += (_, _) => TryDeleteTree(dir);
        return path;
    }

    private static void TryDeleteTree(string dir)
    {
        try
        {
            Directory.Delete(dir, recursive: true);
        }
        catch (IOException)
        {
            // Temp scratch under the test user's path; a locked leftover is noise, not a failure signal.
        }
        catch (UnauthorizedAccessException)
        {
        }
    }
}
