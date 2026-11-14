using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Host.Config;
using Lair.Http;
using Lair.Http.Monitoring;
using Lair.Kv;
using Lair.Loader;
using Lair.Contracts;
using Lair.Models;
using Lair.Models.Engine;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// The dense-matrix prefill route against a file every one of its weight formats has kernels for: the selection opens,
/// reports itself back, and forwards a prompt window byte-for-byte on the batched-row reference it replaces — same logit
/// bytes, same KV bytes, same continuation after that window. A re-point between the two routes on one engine lands on
/// identical arithmetic too, so selecting the route is a traversal choice rather than a second forward math. The refused
/// side of the same gate is pinned on a file whose formats have no such path: it names each format that lacks one instead
/// of running some other route. Cases skip when their fixture file is absent, like the other real-model suites, and share
/// the serialized <c>model-loaded</c> lane.
/// </summary>
[Trait("Category", "Kernel")]
[Collection("model-loaded")] // shares one serialized lane with the other real-model suites: two engines mapping weights at once is not what this box should be doing
public sealed class GemmPrefillIdentityTests
{
    private const string ProvenFixtureRelativePath = "models/smollm-135m-instruct-q8_0.gguf";
    private const string MixedFormatFixtureRelativePath = "models/smollm-135m-instruct-q4_k_m.gguf";

    [Fact]
    public void TransformerModelOpen_EveryMatmulFormatHasADenseMatrixPath_AcceptsTheSelectionAndReportsItsFormats()
    {
        if (ModelFixture.ResolveRepoRelative(ProvenFixtureRelativePath) is not string fixturePath)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(fixturePath);
        using TransformerModel selected = TransformerModel.Open(gguf, backend, maxContextTokens: 128, logLine: null, prefillKernel: PrefillKernel.Gemm);

        Assert.Equal(PrefillKernel.Gemm, selected.ResolvedPrefillKernel); // the engine opens on what was stated and reports it back
        Assert.Contains(QuantFormat.Q8_0, selected.MatmulWeightFormats);

        foreach (QuantFormat format in selected.MatmulWeightFormats)
        {
            Assert.True(backend.SupportsGemm(format), $"the selection opened over a weight format ({format}) this backend has no blocked matmul for");
        }
    }

    [Fact]
    public void PrefillWindow_DenseMatrixRoute_IsRawByteIdenticalToTheBatchedRowAndTiledRoutes()
    {
        if (ModelFixture.ResolveRepoRelative(ProvenFixtureRelativePath) is not string fixturePath)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(fixturePath);
        using TransformerModel reference = TransformerModel.Open(gguf, backend, maxContextTokens: 128);
        using TransformerModel gemv = TransformerModel.Open(gguf, backend, maxContextTokens: 128, logLine: null, prefillKernel: PrefillKernel.GemV);
        using TransformerModel tiled = TransformerModel.Open(gguf, backend, maxContextTokens: 128, logLine: null, prefillKernel: PrefillKernel.Tiled);
        using TransformerModel gemm = TransformerModel.Open(gguf, backend, maxContextTokens: 128, logLine: null, prefillKernel: PrefillKernel.Gemm);

        Assert.Equal(PrefillKernel.GemV, reference.ResolvedPrefillKernel); // an engine opened without a stated selection is the GEMV reference
        Assert.Equal(PrefillKernel.Gemm, gemm.ResolvedPrefillKernel);

        int[] prompt = [.. Enumerable.Repeat(7, 40)]; // past one default sub-batch, so several windows and both routes' column walks run
        using BlockPool pool = new(blockTokens: 16, layerCount: reference.Desc.LayerCount, kvHeads: reference.Desc.KvHeadCount, headDim: reference.Desc.HeadDim, capacityBlocks: 8, backend);

        float[] referenceLogits = Prefill(reference, pool, prompt);
        foreach (TransformerModel route in new[] { gemv, tiled, gemm })
        {
            using SequenceKv kv = new(pool);
            float[] logits = [.. route.Prefill(prompt, kv)];

            AssertBitsEqual(referenceLogits, logits, $"a 40-token window on the '{PrefillKernelNames.Of(route.ResolvedPrefillKernel)}' route against the default");
            Assert.Equal(ArgMax(referenceLogits), ArgMax(logits));
        }

        // The continuation after such a prefill continues identically too: each route is paired with its own reference
        // engine walk from the same prompt, so every step compares like for like.
        foreach (TransformerModel route in new[] { gemv, tiled, gemm })
        {
            using SequenceKv referenceKv = new(pool);
            float[] referenceStepLogits = [.. reference.Prefill(prompt, referenceKv)];
            int nextReference = ArgMax(referenceStepLogits);

            using SequenceKv kv = new(pool);
            float[] logits = [.. route.Prefill(prompt, kv)];
            int next = ArgMax(logits);
            Assert.Equal(nextReference, next);

            for (int step = 0; step < 4; step++)
            {
                float[] referenceStep = [.. reference.DecodeNext(nextReference, referenceKv)];
                float[] routeStep = [.. route.DecodeNext(next, kv)];
                AssertBitsEqual(referenceStep, routeStep, $"decode step {step} after a '{PrefillKernelNames.Of(route.ResolvedPrefillKernel)}' prefill");
                nextReference = ArgMax(referenceStep);
                next = ArgMax(routeStep);
            }

            Assert.Equal(nextReference, next); // the sampled tokens stay identical all the way through
        }
    }

    [Fact]
    public void ApplyPrefillKernel_OnAFileWhoseFormatsAllHaveADenseMatrixPath_RepointsBetweenRoutesIdentically()
    {
        if (ModelFixture.ResolveRepoRelative(ProvenFixtureRelativePath) is not string fixturePath)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(fixturePath);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 128, logLine: null, prefillKernel: PrefillKernel.Gemm);

        int[] prompt = [.. Enumerable.Repeat(7, 30)];
        using BlockPool pool = new(blockTokens: 16, layerCount: model.Desc.LayerCount, kvHeads: model.Desc.KvHeadCount, headDim: model.Desc.HeadDim, capacityBlocks: 8, backend);

        using SequenceKv gemmKv = new(pool);
        float[] gemmLogits = [.. model.Prefill(prompt, gemmKv)];

        model.ApplyPrefillKernel(PrefillKernel.GemV);
        Assert.Equal(PrefillKernel.GemV, model.ResolvedPrefillKernel); // the re-point lands on the other route this file can run

        using SequenceKv gemvKv = new(pool);
        float[] gemvLogits = [.. model.Prefill(prompt, gemvKv)];

        AssertBitsEqual(gemmLogits, gemvLogits, "a window re-pointed onto the batched-row route against the dense-matrix route");
        Assert.Equal(ArgMax(gemmLogits), ArgMax(gemvLogits));
    }

    [Fact]
    public void TransformerModelOpen_AWeightFormatWithoutADenseMatrixPath_RefusesNamingThatFileAndItsFormats()
    {
        if (ModelFixture.ResolveRepoRelative(MixedFormatFixtureRelativePath) is not string fixturePath)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(fixturePath);

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, backend, maxContextTokens: 128, logLine: null, prefillKernel: PrefillKernel.Gemm));

        Assert.Contains("prefill kernel 'gemm' is not implemented by the forward engine", error.Message);

        // The refusal names this file's own dense weight formats as the reason: a family with one matrix that has no
        // blocked matmul never opens on a route it cannot run, and a format it does run is not reported as missing.
        Assert.Contains("carries dense weight matrices of format(s)", error.Message);
        Assert.Contains("no gemm attention/weight path exists in this build for them", error.Message);
        Assert.DoesNotContain("format(s) Q4_K", error.Message);
    }

    [Fact]
    public async Task GenerationServiceOpen_DenseMatrixSelectionOnAProvenFile_ServesTheSameStreamAndReportsBothFacts()
    {
        if (ModelFixture.ResolveRepoRelative(ProvenFixtureRelativePath) is not string fixturePath)
        {
            return; // real-model tests skip when the fixture is absent
        }

        const string model = "smollm-gemm-prefill";
        string prompt = "Describe a quiet harbor at dawn in one short paragraph.";

        GenerationResult defaultRun;
        GenerationResult gemmRun;
        MetricsSnapshot snapshotWithFacts;

        using (GenerationService plain = OpenEngine(fixturePath, model, kernelSelection: null))
        {
            defaultRun = await plain.GenerateAsync(Completion(model, prompt, 16), CancellationToken.None);
        }

        using (GenerationService gemm = OpenEngine(fixturePath, model, "gemm"))
        {
            gemmRun = await gemm.GenerateAsync(Completion(model, prompt, 16), CancellationToken.None);

            EngineMetricsBus bus = gemm.AttachMetrics(NullLoggerFactory.Instance, TimeSpan.FromMilliseconds(50));
            GenerationResult measured = await gemm.GenerateAsync(Completion(model, prompt + " The gulls stayed quiet.", 8), CancellationToken.None);
            Assert.False(string.IsNullOrEmpty(measured.Text));

            for (int attempt = 0; attempt < 40 && bus.RingCount == 0; attempt++)
            {
                await Task.Delay(25); // the bus publishes on its own cadence; wait for a tick rather than assume one already ran
            }

            Assert.True(bus.RingCount > 0, "the metrics bus published no snapshot carrying the resolved selection");
            snapshotWithFacts = bus.SnapshotRing()[0];
        }

        // The dense-matrix route serves what the default route served: same tokens, same budget accounting.
        Assert.Equal(defaultRun.Text, gemmRun.Text);
        Assert.Equal(defaultRun.CompletionTokens, gemmRun.CompletionTokens);
        Assert.Equal(defaultRun.FinishReason, gemmRun.FinishReason);

        // And both facts reach the monitoring surface: which route ran, and the weight formats it ran over.
        Assert.Equal("gemm", snapshotWithFacts.PrefillKernel);
        Assert.Contains(QuantFormat.Q8_0.ToString(), snapshotWithFacts.MatmulWeightFormats);
    }

    // ---------------------------------------------------------------- helpers

    private static float[] Prefill(TransformerModel model, BlockPool pool, int[] prompt)
    {
        using SequenceKv kv = new(pool);
        return [.. model.Prefill(prompt, kv)];
    }

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

    private static GenerationRequest Completion(string model, string prompt, int maxTokens) => new()
    {
        Model = model,
        IsChat = false,
        Prompt = prompt,
        MaxTokens = maxTokens,
        Temperature = 0f, // greedy: byte-comparable across engines and selections
    };

    private static GenerationService OpenEngine(string fixturePath, string modelName, string? kernelSelection) =>
        GenerationService.Open(YamlConfigLoader.Load(WriteConfig(BuildYaml(fixturePath, modelName, kernelSelection))));

    private static string BuildYaml(string fixturePath, string modelName, string? kernelSelection)
    {
        string kernelLine = kernelSelection is null ? string.Empty : $"\n  prefill_kernel: {kernelSelection}";

        return $"""
            engine:
              threads: 4
              max_batch_tokens: 8{kernelLine}
            models:
              - name: {modelName}
                path: '{fixturePath}'
                type: chat
                max_context_tokens: 512
                kv_blocks: 24
            """;
    }

    private static string WriteConfig(string yaml)
    {
        string dir = Path.Combine(Path.GetTempPath(), $"lair-gemm-prefill-{Guid.NewGuid():N}");
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
