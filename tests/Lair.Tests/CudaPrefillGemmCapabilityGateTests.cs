using System.Buffers;

using Lair.Backends.Cpu;
using Lair.Backends.Cuda;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Load-time and re-point capability gate for the dense-matrix prefill selection: which compute backends need blocked
/// matmul kernels is decided from the execution spans a resolved route actually runs each prompt-window range on, not
/// from a blanket claim about one backend and never from a per-layer throw. A whole-model route whose formats every
/// window-running backend multiplies in bulk opens on <c>gemm</c>, reports that selection back, and its windows run the
/// blocked matmul for real; a route naming a format some range's own backend has no kernel for refuses at model open —
/// naming the selection, the offending format(s), the segment, and the backend answering it — with no fall back to a
/// window shape that happens to work. The capability facts these lanes read are the CUDA seam's static coverage
/// (<see cref="CudaBackend.HasGemmKernel"/> / <see cref="CudaBackend.HasGemvKernel"/>) through a device-<c>cuda</c>
/// stand-in whose kernels stay host-side, so no test here opens a driver; refusal lanes name formats the real seam
/// genuinely carries no kernel for (the sub-4-bit i-quants), not chosen-away facts. The
/// CPU posture's own honesty — supported formats select cleanly, unsupported ones refuse naming them — is pinned in the
/// sibling <see cref="PrefillGemmCapabilityGateTests"/>. All of these share one serialized lane with the other suites
/// that open engines on this box.
/// </summary>
[Trait("Category", "Loader")]
[Trait("Category", "Config")]
[Trait("Category", "Cuda")]
[Collection("model-loaded")] // shares one serialized lane with the other model-opening suites: two engines mapping weights at once is not what this box should be doing
public sealed class CudaPrefillGemmCapabilityGateTests : IDisposable
{
    private static readonly SyntheticLlamaGguf.Geometry QuantizedShape = new(
        Layers: 2,
        HiddenSize: 256,
        HeadCount: 2,
        KvHeadCount: 2,
        FeedForwardLength: 512);

    // The synthetic fixture ships two layers, so a split at this boundary hands layer 0 to the accelerator-backed first span and layer 1 to the CPU tail.
    private const int FirstSpanLayers = 1;

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

    // ---------------------------------------------------------------- whole-model cuda route: selection lands

    [Theory]
    [InlineData(QuantFormat.Q8_0)]
    [InlineData(QuantFormat.Q5_0)]
    [InlineData(QuantFormat.Q2_0)]
    [InlineData(QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q5_K)]
    [InlineData(QuantFormat.Q2_K)]
    [InlineData(QuantFormat.Q3_K)]
    public void WholeModelCudaRoute_EveryWindowFormatHasABlockedMatmul_OpenOnGemmAndReportsIt(QuantFormat format)
    {
        Assert.True(CudaBackend.HasGemmKernel(format), $"the CUDA seam stopped reporting its {format} blocked matmul");

        string path = SyntheticLlamaGguf.WriteQuantizedMatrices($"gemm-cuda-{format}", format, QuantizedShape);
        _tempFiles.Add(path);

        ModelRoute route = WholeModelCudaRoute($"gemm-cuda-{format}");

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend hostKernels = new(threads: 2);
        CountingCudaCapabilityStandIn cuda = new(hostKernels);

        using TransformerModel model = TransformerModel.Open(gguf, cuda, maxContextTokens: 32, logLine: null, PrefillKernel.Gemm, route);

        Assert.Equal(PrefillKernel.Gemm, model.ResolvedPrefillKernel); // the engine opens on what was stated and reports it back
        Assert.Contains(format, model.MatmulWeightFormats);            // and names the formats its windows multiply in bulk
        Assert.Single(model.Plan.Route.Segments);
        Assert.Equal(CudaDeviceId, model.Plan.Route.Segments[0].DeviceId);
    }

    [Fact]
    public void WholeModelCudaRoute_GemmWindow_RunsTheBlockedMatmulWithNoUnsupportedThrow()
    {
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices("gemm-cuda-forward", QuantFormat.Q8_0, QuantizedShape);
        _tempFiles.Add(path);

        ModelRoute route = WholeModelCudaRoute("gemm-cuda-forward");

        float[] cudaLogits;
        int gemmPasses;
        int gemvPasses;

        using (GgufFile gguf = GgufFile.Open(path))
        {
            using CpuBackend hostKernels = new(threads: 2);
            CountingCudaCapabilityStandIn cuda = new(hostKernels);

            using TransformerModel model = TransformerModel.Open(gguf, cuda, maxContextTokens: 32, logLine: null, PrefillKernel.Gemm, route);
            using BlockPool pool = NewPool(model, hostKernels);
            using SequenceKv kv = new(pool);

            cudaLogits = [.. model.Prefill([1, 2, 3, 4], kv)]; // every dense weight pass of the window answers its own blocked matmul: no per-layer Unsupported surfaces mid-window

            gemmPasses = cuda.GemmBatchCalls;
            gemvPasses = cuda.GemvBatchCalls;
        }

        Assert.True(gemmPasses > 0, "the dense-matrix selection opened but never ran a blocked matmul");
        Assert.Equal(0, gemvPasses); // no weight pass fell back to the batched-row kernel behind the selection that was resolved

        using (GgufFile gguf = GgufFile.Open(path))
        {
            using CpuBackend referenceHost = new(threads: 2);
            using TransformerModel reference = TransformerModel.Open(gguf, referenceHost, maxContextTokens: 32, logLine: null, PrefillKernel.GemV);
            using BlockPool referencePool = NewPool(reference, referenceHost);
            using SequenceKv referenceKv = new(referencePool);

            float[] referenceLogits = [.. reference.Prefill([1, 2, 3, 4], referenceKv)];

            AssertBitsEqual(referenceLogits, cudaLogits, "a window on the whole-model CUDA dense-matrix route against the batched-row reference");
        }
    }

    // ---------------------------------------------------------------- whole-model cuda route: refusal at open

    [Theory]
    [InlineData(QuantFormat.IQ3_S)]
    public void WholeModelCudaRoute_FormatWithoutABlockedMatmul_RefusesAtOpenNamingSelectionFormatAndBackend(QuantFormat format)
    {
        Assert.False(CudaBackend.HasGemmKernel(format), $"the CUDA seam claimed a {format} blocked matmul its kernel does not exist for");

        string path = SyntheticLlamaGguf.WriteQuantizedMatrices($"gemm-cuda-missing-{format}", format, QuantizedShape);
        _tempFiles.Add(path);

        ModelRoute route = WholeModelCudaRoute($"gemm-cuda-missing-{format}");

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend hostKernels = new(threads: 2);
        CountingCudaCapabilityStandIn cuda = new(hostKernels);

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, cuda, maxContextTokens: 32, logLine: null, PrefillKernel.Gemm, route));

        Assert.Contains("prefill kernel 'gemm'", error.Message, StringComparison.Ordinal);
        Assert.Contains(format.ToString(), error.Message, StringComparison.Ordinal);   // names the file's own format as the reason
        Assert.Contains($"'{CudaDeviceId}'", error.Message, StringComparison.Ordinal);  // and the backend that reports no blocked matmul for it
        Assert.DoesNotContain("no GEMV kernel", error.Message, StringComparison.Ordinal); // this is the dense-matrix gate, not the row-pass gate
    }

    [Theory]
    [InlineData(QuantFormat.F32)]
    [InlineData(QuantFormat.F16)]
    [InlineData(QuantFormat.BF16)]
    public void WholeModelCudaRoute_UnquantizedDenseFormat_OpensOnGemmAndReportsIt(QuantFormat format)
    {
        // The unquantized trio now carries its own windowed dense matmul on this seam, so the selection is what a whole-model 'cuda' route resolves for them.
        Assert.True(CudaBackend.HasGemmKernel(format), $"the CUDA seam stopped reporting its {format} blocked matmul");

        string path = SyntheticLlamaGguf.Write($"gemm-cuda-{format}", withReasoningMarkers: true, format, QuantizedShape);
        _tempFiles.Add(path);

        ModelRoute route = WholeModelCudaRoute($"gemm-cuda-{format}");

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend hostKernels = new(threads: 2);
        CountingCudaCapabilityStandIn cuda = new(hostKernels);

        using TransformerModel model = TransformerModel.Open(gguf, cuda, maxContextTokens: 32, logLine: null, PrefillKernel.Gemm, route);

        Assert.Equal(PrefillKernel.Gemm, model.ResolvedPrefillKernel); // no load-time refusal naming a format this seam multiplies in bulk
        Assert.Contains(format, model.MatmulWeightFormats);
    }

    [Theory]
    [InlineData(QuantFormat.Q8_0, true)]
    [InlineData(QuantFormat.Q5_0, true)]
    [InlineData(QuantFormat.Q2_0, true)]
    [InlineData(QuantFormat.Q4_K, true)]
    [InlineData(QuantFormat.Q6_K, true)]
    [InlineData(QuantFormat.Q5_K, true)]
    [InlineData(QuantFormat.Q2_K, true)]
    [InlineData(QuantFormat.Q3_K, true)]
    [InlineData(QuantFormat.IQ4_NL, true)] // the IQ4 pair carries a windowed kernel even though this suite's synthetic writer ships no block geometry for it
    [InlineData(QuantFormat.IQ4_XS, true)]
    [InlineData(QuantFormat.F32, true)] // the unquantized trio carries windowed dense matmul here today.
    [InlineData(QuantFormat.F16, true)]
    [InlineData(QuantFormat.BF16, true)]
    [InlineData(QuantFormat.IQ2_XXS, false)] // the sub-4-bit i-quants stay refused until their dense path lands
    [InlineData(QuantFormat.IQ2_XS, false)]
    [InlineData(QuantFormat.IQ2_S, false)]
    [InlineData(QuantFormat.IQ3_XXS, false)]
    [InlineData(QuantFormat.IQ3_S, false)]
    public void CudaBlockedMatmulCapability_CoversExactlyTheFormatsTheSeamCarries(QuantFormat format, bool supported) =>
        Assert.Equal(supported, CudaBackend.HasGemmKernel(format));

    [Fact]
    public void WholeModelCudaRoute_SameFileOnGemv_OpensUnchanged_BecauseBatchedRowsCarryEveryFormat()
    {
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices("gemm-cuda-gemv", QuantFormat.IQ3_S, QuantizedShape);
        _tempFiles.Add(path);

        ModelRoute route = WholeModelCudaRoute("gemm-cuda-gemv");

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend hostKernels = new(threads: 2);
        CountingCudaCapabilityStandIn cuda = new(hostKernels);

        Assert.False(CudaBackend.HasGemmKernel(QuantFormat.IQ3_S), "the seam claimed a blocked matmul for the format this lane pins as missing");

        using TransformerModel model = TransformerModel.Open(gguf, cuda, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, route);
        Assert.Equal(PrefillKernel.GemV, model.ResolvedPrefillKernel); // the refusal belongs to the dense-matrix selection alone; every other window shape still runs this file

        using BlockPool pool = NewPool(model, hostKernels);
        using SequenceKv kv = new(pool);
        Assert.Equal(QuantizedShape.Layers, model.Desc.LayerCount);
        float[] logits = [.. model.Prefill([1, 2, 3], kv)]; // the batched-row window forwards this file end to end
        Assert.NotEmpty(logits);
    }

    // ---------------------------------------------------------------- offloaded prompt side: decided from the backend that answers the windows

    [Fact]
    public void OffloadedCudaPromptBackend_FormatOnlyThatDeviceMultipliesInBulk_OpensOnGemmAndReportsIt()
    {
        // Q5_K is the format this build's CUDA seam multiplies in bulk while the CPU backend has no blocked matmul for it.
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices("gemm-offload-q5k", QuantFormat.Q5_K, QuantizedShape);
        _tempFiles.Add(path);

        ModelRoute route = SingleCpuSegmentRoute("gemm-offload-q5k");

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend decodeSide = new(threads: 2);
        CountingCudaCapabilityStandIn cuda = new(decodeSide);

        Assert.False(decodeSide.SupportsGemm(QuantFormat.Q5_K), "the CPU backend claimed a blocked matmul it does not carry");
        Assert.True(CudaBackend.HasGemvKernel(QuantFormat.Q5_K), "the CUDA seam stopped reporting its Q5_K GEMV kernel: the decode side still dots this file's rows");

        using TransformerModel model = TransformerModel.Open(
            gguf, decodeSide, maxContextTokens: 32, logLine: null, PrefillKernel.Gemm, route, prefillsBackend: cuda);

        // The selection is decided from the backend that actually runs each window range: this engine's own CPU backend multiplies nothing in bulk here and never blocks the open.
        Assert.Equal(PrefillKernel.Gemm, model.ResolvedPrefillKernel);
        Assert.Contains(QuantFormat.Q5_K, model.MatmulWeightFormats);
    }

    [Fact]
    public void OffloadedCudaPromptBackend_NamingAFormatItCannotMultiplyInBulk_RefusesAtOpenNamingSegmentAndThatBackend()
    {
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices("gemm-offload-missing", QuantFormat.IQ3_S, QuantizedShape);
        _tempFiles.Add(path);

        ModelRoute route = SingleCpuSegmentRoute("gemm-offload-missing");

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend decodeSide = new(threads: 2);
        CountingCudaCapabilityStandIn cuda = new(decodeSide);

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, decodeSide, maxContextTokens: 32, logLine: null, PrefillKernel.Gemm, route, prefillsBackend: cuda));

        Assert.Contains("model 'gemm-offload-missing'", error.Message, StringComparison.Ordinal);
        Assert.Contains("prefill kernel 'gemm'", error.Message, StringComparison.Ordinal);
        Assert.Contains("segment 0", error.Message, StringComparison.Ordinal);          // the window range whose own backend lacks the kernel
        Assert.Contains($"'{CudaDeviceId}'", error.Message, StringComparison.Ordinal);   // and that offloaded prompt backend by device
        Assert.Contains("IQ3_S", error.Message, StringComparison.Ordinal);               // names what it cannot multiply in bulk
    }

    [Fact]
    public void OffloadedCudaPromptBackend_HotRepointToGemmMissingCapability_RefusesAndLeavesRunningRouteUntouched()
    {
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices("gemm-offload-repoint", QuantFormat.IQ3_S, QuantizedShape);
        _tempFiles.Add(path);

        ModelRoute route = SingleCpuSegmentRoute("gemm-offload-repoint");

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend decodeSide = new(threads: 2);
        CountingCudaCapabilityStandIn cuda = new(decodeSide);

        using TransformerModel model = TransformerModel.Open(gguf, decodeSide, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, route, prefillsBackend: cuda);
        Assert.Equal(PrefillKernel.GemV, model.ResolvedPrefillKernel);

        ModelException error = Assert.Throws<ModelException>(() => model.ApplyPrefillKernel(PrefillKernel.Gemm));

        Assert.Contains("prefill kernel 'gemm'", error.Message, StringComparison.Ordinal);
        Assert.Contains($"'{CudaDeviceId}'", error.Message, StringComparison.Ordinal);
        Assert.Equal(PrefillKernel.GemV, model.ResolvedPrefillKernel); // a refused re-point leaves the route this engine opened on in force — no window ever runs some other kernel behind it
    }

    // ---------------------------------------------------------------- hybrid split: each range answers for its own backend

    [Fact]
    public void HybridSplitRoute_WhereEachRangeMultipliesItsOwnFormatsOnItsOwnBackend_OpenOnGemmAndReportsIt()
    {
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices("gemm-split-q8", QuantFormat.Q8_0, QuantizedShape);
        _tempFiles.Add(path);

        ModelRoute route = CudaFirstSplitRoute("gemm-split-q8");

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend decodeSide = new(threads: 2);
        CountingCudaCapabilityStandIn cuda = new(decodeSide);

        Assert.True(CpuBackendSupportsGemm(QuantFormat.Q8_0), "the CPU backend stopped carrying its Q8_0 blocked matmul");

        using TransformerModel model = TransformerModel.Open(
            gguf, decodeSide, maxContextTokens: 32, logLine: null, PrefillKernel.Gemm, route, [cuda, decodeSide]);

        Assert.Equal(PrefillKernel.Gemm, model.ResolvedPrefillKernel); // every resolved range names formats its own backend multiplies in bulk
        Assert.Equal(2, model.Plan.Route.Segments.Count);
        Assert.Equal(CudaDeviceId, model.Plan.Route.Segments[0].DeviceId);
        Assert.Equal(ModelRoute.CpuDeviceId, model.Plan.Route.Segments[1].DeviceId);

        using BlockPool pool = NewPool(model, decodeSide);
        using SequenceKv kv = new(pool);
        float[] splitLogits = [.. model.Prefill([1, 2, 3], kv)]; // and the window crosses: both spans ran their own dense passes
        Assert.NotEmpty(splitLogits);
    }

    [Fact]
    public void HybridSplitRoute_RangeNamingAFormatItsOwnBackendCannotMultiplyInBulk_RefusesAtOpenNamingSelectionAndFormat()
    {
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices("gemm-split-q5k", QuantFormat.Q5_K, QuantizedShape);
        _tempFiles.Add(path);

        ModelRoute route = CudaFirstSplitRoute("gemm-split-q5k");

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend decodeSide = new(threads: 2);
        CountingCudaCapabilityStandIn cuda = new(decodeSide);

        Assert.True(CudaBackend.HasGemmKernel(QuantFormat.Q5_K), "the CUDA seam stopped reporting its Q5_K blocked matmul");
        Assert.False(decodeSide.SupportsGemm(QuantFormat.Q5_K), "the CPU backend claimed a blocked matmul it does not carry");

        // The accelerator-backed first span could run this format in bulk; the CPU tail's own range cannot, so the route is refused rather than split across kernel shapes mid-window.
        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, decodeSide, maxContextTokens: 32, logLine: null, PrefillKernel.Gemm, route, [cuda, decodeSide]));

        Assert.Contains("prefill kernel 'gemm'", error.Message, StringComparison.Ordinal);
        Assert.Contains("Q5_K", error.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("no GEMV kernel", error.Message, StringComparison.Ordinal);
    }

    // ---------------------------------------------------------------- helpers

    private const string CudaDeviceId = CudaBackend.DeviceIdValue;

    private static ModelRoute WholeModelCudaRoute(string modelName) =>
        ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = modelName,
            Stages = [new RouteStage(ModelRoutes.PrefillStage, CudaDeviceId), new RouteStage(ModelRoutes.DecodeStage, CudaDeviceId)],
            AvailableDevices = [ModelRoute.CpuDeviceId, CudaDeviceId],
        });

    private static ModelRoute SingleCpuSegmentRoute(string modelName) =>
        ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = modelName,
            Stages = [new RouteStage(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId)],
            AvailableDevices = [ModelRoute.CpuDeviceId, CudaDeviceId],
        });

    private static ModelRoute CudaFirstSplitRoute(string modelName) =>
        ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = modelName,
            Stages =
            [
                new RouteStage(ModelRoutes.PrefillStage, CudaDeviceId) { SplitPosition = 0 },
                new RouteStage(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId) { SplitPosition = FirstSpanLayers },
            ],
            AvailableDevices = [ModelRoute.CpuDeviceId, CudaDeviceId],
        });

    private static BlockPool NewPool(TransformerModel model, IBackend poolOwner) =>
        new(blockTokens: 16, layerCount: model.Desc.LayerCount, kvHeads: model.Desc.KvHeadCount, headDim: model.Desc.HeadDim, capacityBlocks: 8, poolOwner);

    private static bool CpuBackendSupportsGemm(QuantFormat format)
    {
        using CpuBackend cpu = new(threads: 1);
        return cpu.SupportsGemm(format);
    }

    /// <summary>Compares two float windows on the raw bits of the floats: NaN payloads and signed zeros must match too.</summary>
    private static void AssertBitsEqual(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string context)
    {
        Assert.Equal(expected.Length, actual.Length);

        ReadOnlySpan<byte> expectedBytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(expected);
        ReadOnlySpan<byte> actualBytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(actual);

        for (int i = 0; i < expectedBytes.Length; i++)
        {
            Assert.True(
                expectedBytes[i] == actualBytes[i],
                $"{context}: first differing byte at {i} (float {i / 4}): expected 0x{expectedBytes[i]:X2}, got 0x{actualBytes[i]:X2}");
        }
    }

    /// <summary>
    /// A compute backend answering device <c>cuda</c> with the CUDA seam's own static capability facts — its GEMV and blocked-matmul coverage read straight from
    /// <see cref="CudaBackend.HasGemvKernel"/> and <see cref="CudaBackend.HasGemmKernel"/> — while every kernel stays host-side behind the wrapped CPU backend. The load-time gate decides on those capability answers and the tensor table alone, so these lanes never open a driver, and the formats they report as missing are exactly the ones
    /// the real seam carries no kernel for. The counters make which weight passes this engine's own spans executed visible: a dense-matrix window that ran its blocked matmul is distinguishable from one that fell back to batched rows.
    /// </summary>
    private sealed class CountingCudaCapabilityStandIn : IBackend
    {
        private readonly CpuBackend _inner;

        public CountingCudaCapabilityStandIn(CpuBackend inner)
        {
            ArgumentNullException.ThrowIfNull(inner);
            _inner = inner;
        }

        /// <summary>Blocked-matmul passes this backend ran.</summary>
        public int GemmBatchCalls { get; private set; }

        /// <summary>Batched-row weight passes this backend ran.</summary>
        public int GemvBatchCalls { get; private set; }

        public string DeviceId => CudaBackend.DeviceIdValue;

        public Lair.Contracts.WeightTier WeightTier => Lair.Contracts.WeightTier.Dram;

        public bool SupportsGemv(QuantFormat format) => CudaBackend.HasGemvKernel(format);

        public bool SupportsGemm(QuantFormat format) => CudaBackend.HasGemmKernel(format);

        public MemoryHandle AllocateAligned(long bytes, int alignment) => _inner.AllocateAligned(bytes, alignment);

        public void FreeAligned(MemoryHandle handle) => _inner.FreeAligned(handle);

        public IKvDeviceMirror OpenKvMirror(KvMirrorRequest request) => _inner.OpenKvMirror(request);

        public void Gemv(ReadOnlySpan<float> x, WeightRef w, Span<float> output) => _inner.Gemv(x, w, output);

        public void GemvBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs)
        {
            GemvBatchCalls++;
            _inner.GemvBatch(inputs, count, w, outputs);
        }

        public void GemmBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs)
        {
            if (!CudaBackend.HasGemmKernel(w.Format))
            {
                throw new NotSupportedException($"this stand-in reports no blocked matmul for {w.Format}, exactly as the CUDA seam refuses it");
            }

            GemmBatchCalls++;
            _inner.GemmBatch(inputs, count, w, outputs); // host-side arithmetic: what a window ran through this backend's own dense-matrix path.
        }

        public void DequantRow(WeightRef w, int row, Span<float> output) => _inner.DequantRow(w, row, output);

        public float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b) => _inner.Dot(a, b);

        public void Axpy(float alpha, ReadOnlySpan<float> x, Span<float> y) => _inner.Axpy(alpha, x, y);

        public void Fill(Span<float> y, float v) => _inner.Fill(y, v);

        public void Scale(Span<float> y, float s) => _inner.Scale(y, s);

        public void Dispose()
        {
            // The CPU backend this stand-in reads through belongs to the test that opened it.
        }
    }
}
