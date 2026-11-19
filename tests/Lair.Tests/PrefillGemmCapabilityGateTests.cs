using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// The CPU posture of the same dense-matrix capability gate, pinned on synthetic files so it never skips: formats this
/// build multiplies in bulk (<c>Q8_0</c>, <c>Q4_K</c>, <c>IQ4_XS</c>) select cleanly at open and re-point to the dense
/// route byte-for-byte; a format with no blocked matmul refuses naming itself and the backend that reports none, at open
/// and on a hot re-point alike, and never falls back to the window shape that happens to work. No CPU GEMM behavior is
/// changed by these lanes — they read the same capability answer the kernels resolve from. Runs in the serialized
/// <c>model-loaded</c> lane with the CUDA posture of this gate and the other prefill-kernel suites.
/// </summary>
[Trait("Category", "Loader")]
[Trait("Category", "Config")]
[Collection("model-loaded")] // shares one serialized lane with the other model-opening suites on this box
public sealed class PrefillGemmCapabilityGateTests : IDisposable
{
    private static readonly SyntheticLlamaGguf.Geometry QuantizedShape = new(
        Layers: 2,
        HiddenSize: 256,
        HeadCount: 2,
        KvHeadCount: 2,
        FeedForwardLength: 512);

    private readonly List<string> _tempFiles = [];

    public void Dispose()
    {
        foreach (string path in _tempFiles)
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

        _tempFiles.Clear();
    }

    [Theory]
    [InlineData(QuantFormat.Q8_0)]
    [InlineData(QuantFormat.Q4_K)]
    public void CpuRoute_EveryWindowFormatHasABlockedMatmul_OpenOnGemmAndReportIt(QuantFormat format)
    {
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices($"gemm-cpu-{format}", format, QuantizedShape);
        _tempFiles.Add(path);

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend cpu = new(threads: 2);

        Assert.True(cpu.SupportsGemm(format), $"the CPU backend stopped reporting its {format} blocked matmul");

        using TransformerModel model = TransformerModel.Open(gguf, cpu, maxContextTokens: 32, logLine: null, PrefillKernel.Gemm);

        Assert.Equal(PrefillKernel.Gemm, model.ResolvedPrefillKernel);
        Assert.Contains(format, model.MatmulWeightFormats);

        using BlockPool pool = NewPool(model, cpu);
        using SequenceKv kv = new(pool);
        float[] logits = [.. model.Prefill([1, 2, 3], kv)]; // the selection runs: its windows multiply their own sub-batch against each matrix
        Assert.NotEmpty(logits);
    }

    [Theory]
    [InlineData(QuantFormat.Q8_0, true)]
    [InlineData(QuantFormat.Q4_K, true)]
    [InlineData(QuantFormat.IQ4_XS, true)] // the capability fact is pinned straight from the backend: this suite's synthetic writer ships no block geometry for it
    [InlineData(QuantFormat.Q5_K, false)]
    [InlineData(QuantFormat.IQ3_S, false)]
    public void CpuBlockedMatmulCapability_CoversExactlyTheFormatsThisBuildMultipliesInBulk(QuantFormat format, bool supported)
    {
        using CpuBackend cpu = new(threads: 1);

        Assert.Equal(supported, cpu.SupportsGemm(format));
    }

    [Theory]
    [InlineData(QuantFormat.Q5_K)]
    [InlineData(QuantFormat.IQ3_S)]
    public void CpuRoute_FormatWithoutABlockedMatmul_RefusesAtOpenNamingFormatAndBackend(QuantFormat format)
    {
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices($"gemm-cpu-missing-{format}", format, QuantizedShape);
        _tempFiles.Add(path);

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend cpu = new(threads: 2);

        Assert.False(cpu.SupportsGemm(format), $"the CPU backend claimed a {format} blocked matmul it does not carry");

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, cpu, maxContextTokens: 32, logLine: null, PrefillKernel.Gemm));

        Assert.Contains("prefill kernel 'gemm'", error.Message, StringComparison.Ordinal);
        Assert.Contains(format.ToString(), error.Message, StringComparison.Ordinal);
        Assert.Contains("'cpu'", error.Message, StringComparison.Ordinal); // names the backend that reports no blocked matmul for it
    }

    [Fact]
    public void CpuEngine_OpenedOnGemV_RepointToGemmWhereEveryFormatHasOne_LandsOnTheDenseRouteIdentically()
    {
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices("gemm-cpu-repoint", QuantFormat.Q8_0, QuantizedShape);
        _tempFiles.Add(path);

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend cpu = new(threads: 2);

        using TransformerModel model = TransformerModel.Open(gguf, cpu, maxContextTokens: 32, logLine: null, PrefillKernel.GemV);
        Assert.Equal(PrefillKernel.GemV, model.ResolvedPrefillKernel);

        int[] prompt = [1, 2, 3, 4];
        using BlockPool pool = NewPool(model, cpu);

        using SequenceKv gemvKv = new(pool);
        float[] gemvLogits = [.. model.Prefill(prompt, gemvKv)];

        model.ApplyPrefillKernel(PrefillKernel.Gemm);
        Assert.Equal(PrefillKernel.Gemm, model.ResolvedPrefillKernel); // the re-point lands on the dense route this file can run

        using SequenceKv gemmKv = new(pool);
        float[] gemmLogits = [.. model.Prefill(prompt, gemmKv)];

        AssertBitsEqual(gemvLogits, gemmLogits, "a window re-pointed onto the dense-matrix route against the batched-row route");
    }

    [Fact]
    public void CpuEngine_RepointToGemmWithoutABlockedMatmul_RefusesAndLeavesRunningRouteUntouched()
    {
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices("gemm-cpu-repoint-missing", QuantFormat.Q5_K, QuantizedShape);
        _tempFiles.Add(path);

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend cpu = new(threads: 2);

        using TransformerModel model = TransformerModel.Open(gguf, cpu, maxContextTokens: 32, logLine: null, PrefillKernel.GemV);
        Assert.Equal(PrefillKernel.GemV, model.ResolvedPrefillKernel);

        ModelException error = Assert.Throws<ModelException>(() => model.ApplyPrefillKernel(PrefillKernel.Gemm));

        Assert.Contains("prefill kernel 'gemm'", error.Message, StringComparison.Ordinal);
        Assert.Contains("Q5_K", error.Message, StringComparison.Ordinal);
        Assert.Equal(PrefillKernel.GemV, model.ResolvedPrefillKernel); // a refused re-point leaves the running route untouched — no window runs some other kernel behind it
    }

    // ---------------------------------------------------------------- helpers

    private static BlockPool NewPool(TransformerModel model, IBackend poolOwner) =>
        new(blockTokens: 16, layerCount: model.Desc.LayerCount, kvHeads: model.Desc.KvHeadCount, headDim: model.Desc.HeadDim, capacityBlocks: 8, poolOwner);

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
}
