using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Host.Config;
using Lair.Http;
using Lair.Http.Monitoring;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// What a resolved <c>engine.prefill_kernel</c> selection does at model open on the pinned SmolLM fixture: the dense-matrix
/// window is refused for this file because its weights carry no format with a blocked matmul, and the refusal names the
/// served model, the value, and that file's own formats — never fallen back to the route that happens to work — while an
/// explicit <c>gemv</c> or <c>tiled</c> opens, forwards, and serves exactly what the absent key does today: the row-tiled
/// window is bit-identical with the batched-row one. The resolved selection also reaches the monitor snapshot's
/// configuration-backed facts so a host can read which route an engine actually runs. Runs inside the serialized
/// real-model lane with the parity suites.
/// </summary>
[Trait("Category", "Config")]
[Collection("model-loaded")] // shares one serialized lane with the other real-model suites: two engines mapping 270 MB at once is not what this box should be doing
public sealed class PrefillKernelServeTests
{
    private const string ModelName = "smollm-prefill-kernel";

    [Theory]
    [InlineData("gemm")]
    public async Task GenerationServiceOpen_SelectionWithoutAPathForThisFile_RefusesAtModelOpenNamingModelFormatAndMissingKernel(string selection)
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        ModelException error = await Assert.ThrowsAsync<ModelException>(() => Task.Run(() => OpenEngine(selection)));

        Assert.Contains($"prefill kernel '{selection}' is not implemented by the forward engine", error.Message);
        Assert.Contains($"no {selection} attention/weight path exists in this build", error.Message); // names what is missing, not just the value
        Assert.Contains("model 'smollm-prefill-kernel'", error.Message); // names the served model the refusal is about
        Assert.Contains("format(s) F16", error.Message); // names this file's own weight format as the reason the selection cannot run here
    }

    [Theory]
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

    [Theory]
    [InlineData(PrefillKernel.GemV)]
    [InlineData(PrefillKernel.Tiled)]
    public void TransformerModelOpen_KernelBearingSelection_ForwardsBitIdenticalToTheDefaultRoute(PrefillKernel selection)
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(ModelFixture.Resolve()!);
        using TransformerModel defaultRoute = TransformerModel.Open(gguf, backend, maxContextTokens: 128);
        using TransformerModel selected = TransformerModel.Open(gguf, backend, maxContextTokens: 128, logLine: null, prefillKernel: selection);

        Assert.Equal(selection, selected.ResolvedPrefillKernel); // the engine opens on what was stated and reports it back

        int[] prompt = [.. Enumerable.Repeat(7, 20)];
        using BlockPool pool = new(blockTokens: 16, layerCount: defaultRoute.Desc.LayerCount, kvHeads: defaultRoute.Desc.KvHeadCount, headDim: defaultRoute.Desc.HeadDim, capacityBlocks: 8, backend);
        using SequenceKv defaultKv = new(pool);
        using SequenceKv selectedKv = new(pool);

        float[] defaultLogits = [.. defaultRoute.Prefill(prompt, defaultKv)];
        float[] selectedLogits = [.. selected.Prefill(prompt, selectedKv)];

        AssertBitsEqual(defaultLogits, selectedLogits, $"a 20-token window on the '{PrefillKernelNames.Of(selection)}' route against the default");
        Assert.Equal(ArgMax(defaultLogits), ArgMax(selectedLogits));
        AssertKvBytesEqual(defaultKv, selectedKv, defaultRoute.Desc.LayerCount, prompt.Length);

        // The decode steps after such a prefill continue identically too: same first token, then byte-identical logits.
        int nextDefault = ArgMax(defaultLogits);
        int nextSelected = ArgMax(selectedLogits);
        for (int step = 0; step < 4; step++)
        {
            float[] defaultStep = [.. defaultRoute.DecodeNext(nextDefault, defaultKv)];
            float[] selectedStep = [.. selected.DecodeNext(nextSelected, selectedKv)];
            AssertBitsEqual(defaultStep, selectedStep, $"decode step {step} after a '{PrefillKernelNames.Of(selection)}' prefill");
            nextDefault = ArgMax(defaultStep);
            nextSelected = ArgMax(selectedStep);
        }

        Assert.Equal(nextDefault, nextSelected); // the sampled tokens stay identical all the way through
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
    public void ApplyPrefillKernel_ReservedSelection_Refuses_AndTiledAndGemvRepointsLandOnVisibleRoutes()
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

        // The row-tiled selection has a kernel: an in-place re-point lands on it and its windows match what the
        // default route forwarded — same logits bytes, same KV bytes.
        int[] prompt = [.. Enumerable.Repeat(7, 18)];
        using BlockPool pool = new(blockTokens: 16, layerCount: model.Desc.LayerCount, kvHeads: model.Desc.KvHeadCount, headDim: model.Desc.HeadDim, capacityBlocks: 8, backend);
        using SequenceKv gemvKv = new(pool);
        float[] gemvLogits = [.. model.Prefill(prompt, gemvKv)];

        model.ApplyPrefillKernel(PrefillKernel.Tiled);
        Assert.Equal(PrefillKernel.Tiled, model.ResolvedPrefillKernel);

        using SequenceKv tiledKv = new(pool);
        float[] tiledLogits = [.. model.Prefill(prompt, tiledKv)];

        AssertBitsEqual(gemvLogits, tiledLogits, "a window re-pointed onto the row-tiled route against the gemv route");
        Assert.Equal(ArgMax(gemvLogits), ArgMax(tiledLogits));
        AssertKvBytesEqual(gemvKv, tiledKv, model.Desc.LayerCount, prompt.Length);

        model.ApplyPrefillKernel(PrefillKernel.GemV);
        Assert.Equal(PrefillKernel.GemV, model.ResolvedPrefillKernel); // the re-point back lands on the other visible route
    }

    // ---------------------------------------------------------------- helpers

    /// <summary>Compares two float windows on the raw bits of the floats: NaN payloads and signed zeros must match too.</summary>
    private static void AssertBitsEqual(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string context)
    {
        Assert.Equal(expected.Length, actual.Length);

        ReadOnlySpan<byte> expectedBytes = MemoryMarshal.AsBytes(expected);
        ReadOnlySpan<byte> actualBytes = MemoryMarshal.AsBytes(actual);

        for (int i = 0; i < expectedBytes.Length; i++)
        {
            Assert.True(
                expectedBytes[i] == actualBytes[i],
                $"{context}: first differing byte at {i} (float {i / 4}): expected 0x{expectedBytes[i]:X2}, got 0x{actualBytes[i]:X2}");
        }
    }

    // Every position and every layer: the whole KV both routes wrote, compared slot by slot on raw bytes.
    private static void AssertKvBytesEqual(SequenceKv reference, SequenceKv other, int layerCount, int positions)
    {
        Assert.Equal(positions, other.Length);

        for (int position = 0; position < positions; position++)
        {
            for (int layer = 0; layer < layerCount; layer++)
            {
                AssertBitsEqual(reference.ReadK(position, layer), other.ReadK(position, layer), $"K at position {position} layer {layer}");
                AssertBitsEqual(reference.ReadV(position, layer), other.ReadV(position, layer), $"V at position {position} layer {layer}");
            }
        }
    }

    /// <summary>Picks the top id by descending logit with a strict "greater than" scan: equal logits keep the lower id.</summary>
    private static int ArgMax(ReadOnlySpan<float> logits)
    {
        float best = float.NegativeInfinity;
        int bestIdx = -1;

        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > best)
            {
                best = logits[i];
                bestIdx = i;
            }
        }

        Assert.True(bestIdx >= 0, "logit vector had no selectable entry");
        return bestIdx;
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
