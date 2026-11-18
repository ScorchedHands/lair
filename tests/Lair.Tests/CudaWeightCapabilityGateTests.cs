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
/// Load-time CUDA capability gate: a resolved route whose segment names a compute backend with no GEMV kernel for one of
/// the weight formats that segment runs refuses at model open, naming the model, the offending format, the segment, and
/// that backend — the posture every other unimplemented capability takes, and the replacement for the mid-decode
/// <c>Unsupported</c> throw that used to surface the same fact after work had dispatched. The gate consults the backend's
/// own capability answer, so the CUDA formats the seam executes today (<c>F32</c>, <c>F16</c>, <c>BF16</c>, <c>Q4_K</c>,
/// <c>Q6_K</c>) still resolve and open unchanged, a CPU-only route is untouched, and nothing falls back to CPU for a
/// device-named segment. The CUDA capability facts are read through the backend's static kernel coverage, so no test here
/// opens a device: the refusal is a load-time decision over the tensor table.
/// </summary>
[Trait("Category", "Loader")]
[Trait("Category", "Config")]
[Collection("model-loaded")] // shares one serialized lane with the other suites that open engines on this box
public sealed class CudaWeightCapabilityGateTests : IDisposable
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

    // ---------------------------------------------------------------- refusal at load

    [Fact]
    public void CudaRoute_NamingUnsupportedQuantFormat_RefusesAtOpen_NamingModelFormatAndBackend()
    {
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices("gate-cuda-iq3s", QuantFormat.IQ3_S, QuantizedShape);
        _tempFiles.Add(path);

        ModelRoute route = ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "gate-cuda-iq3s",
            Stages = [new RouteStage(ModelRoutes.PrefillStage, CudaDeviceId), new RouteStage(ModelRoutes.DecodeStage, CudaDeviceId)],
            AvailableDevices = [ModelRoute.CpuDeviceId, CudaDeviceId],
        });

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend hostKernels = new(threads: 2);
        CudaCapabilityStandIn cuda = new(hostKernels);

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, cuda, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, route));

        Assert.Contains("model 'gate-cuda-iq3s'", error.Message, StringComparison.Ordinal);
        Assert.Contains("IQ3_S", error.Message, StringComparison.Ordinal);
        Assert.Contains($"segment 0 ('{CudaDeviceId}'", error.Message, StringComparison.Ordinal);
        Assert.Contains($"'{CudaDeviceId}'", error.Message, StringComparison.Ordinal);
        Assert.Contains("no GEMV kernel", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void HybridSplitCudaSpan_NamingUnsupportedQuantFormat_RefusesAtOpen_NamingThatSegmentAndBackend()
    {
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices("gate-hybrid-iq3s", QuantFormat.IQ3_S, QuantizedShape);
        _tempFiles.Add(path);

        ModelRoute route = ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "gate-hybrid-iq3s",
            Stages =
            [
                new RouteStage(ModelRoutes.PrefillStage, CudaDeviceId) { SplitPosition = 0 },
                new RouteStage(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId) { SplitPosition = 1 },
            ],
            AvailableDevices = [ModelRoute.CpuDeviceId, CudaDeviceId],
        });

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend cpu = new(threads: 2);
        CudaCapabilityStandIn cuda = new(cpu);

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, cpu, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, route, [cuda, cpu]));

        Assert.Contains("model 'gate-hybrid-iq3s'", error.Message, StringComparison.Ordinal);
        Assert.Contains("IQ3_S", error.Message, StringComparison.Ordinal);
        Assert.Contains($"segment 0 ('{CudaDeviceId}'", error.Message, StringComparison.Ordinal);
        Assert.Contains("no GEMV kernel", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void HybridSplitCudaTail_NamingUnsupportedTiedHeadFormat_RefusesAtOpen_NamingTheHeadItsBackendRuns()
    {
        // Every transformer layer runs on cpu; the tail span carries no layers and its backend dots the tied output head —
        // the one weight pass a layerless cuda span still owns, so its format is what the gate names.
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices("gate-hybrid-head", QuantFormat.IQ3_S, QuantizedShape);
        _tempFiles.Add(path);

        ModelRoute route = ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "gate-hybrid-head",
            Stages =
            [
                new RouteStage(ModelRoutes.PrefillStage, ModelRoute.CpuDeviceId) { SplitPosition = 0 },
                new RouteStage(ModelRoutes.DecodeStage, CudaDeviceId) { SplitPosition = QuantizedShape.Layers },
            ],
            AvailableDevices = [ModelRoute.CpuDeviceId, CudaDeviceId],
        });

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend cpu = new(threads: 2);
        CudaCapabilityStandIn cuda = new(cpu);

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, cpu, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, route, [cpu, cuda]));

        Assert.Contains("model 'gate-hybrid-head'", error.Message, StringComparison.Ordinal);
        Assert.Contains("IQ3_S", error.Message, StringComparison.Ordinal);
        Assert.Contains($"segment 1 ('{CudaDeviceId}'", error.Message, StringComparison.Ordinal);
        Assert.Contains("no GEMV kernel", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void OffloadedCudaPrefill_NamingUnsupportedQuantFormat_RefusesAtOpen_NamingThatBackend()
    {
        // A prefill offload posture names the cuda device for prompt windows over the single DRAM-resident span: the
        // format that window would dot is refused at open, not raised when the first window dispatches.
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices("gate-offload-iq3s", QuantFormat.IQ3_S, QuantizedShape);
        _tempFiles.Add(path);

        ModelRoute route = ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "gate-offload-iq3s",
            Stages = [new RouteStage(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId)],
            AvailableDevices = [ModelRoute.CpuDeviceId, CudaDeviceId],
        });

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend cpu = new(threads: 2);
        CudaCapabilityStandIn cuda = new(cpu);

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, cpu, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, route, prefillsBackend: cuda));

        Assert.Contains("model 'gate-offload-iq3s'", error.Message, StringComparison.Ordinal);
        Assert.Contains("IQ3_S", error.Message, StringComparison.Ordinal);
        Assert.Contains($"'{CudaDeviceId}'", error.Message, StringComparison.Ordinal);
        Assert.Contains("no GEMV kernel", error.Message, StringComparison.Ordinal);
    }

    // ---------------------------------------------------------------- supported postures unchanged

    [Theory]
    [InlineData(QuantFormat.F32)]
    [InlineData(QuantFormat.F16)]
    [InlineData(QuantFormat.BF16)]
    [InlineData(QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q6_K)]
    public void CudaRoute_NamingFormatTheSeamExecutes_ResolvesAndOpensUnchanged(QuantFormat format)
    {
        Assert.True(CudaBackend.HasGemvKernel(format), $"the CUDA seam stopped reporting its {format} GEMV kernel");

        string path = SyntheticLlamaGguf.WriteQuantizedMatrices($"gate-cuda-{format}", format, QuantizedShape);
        _tempFiles.Add(path);

        ModelRoute route = ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = $"gate-cuda-{format}",
            Stages = [new RouteStage(ModelRoutes.PrefillStage, CudaDeviceId), new RouteStage(ModelRoutes.DecodeStage, CudaDeviceId)],
            AvailableDevices = [ModelRoute.CpuDeviceId, CudaDeviceId],
        });

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend hostKernels = new(threads: 2);
        CudaCapabilityStandIn cuda = new(hostKernels);

        using TransformerModel model = TransformerModel.Open(gguf, cuda, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, route);

        Assert.Equal(QuantizedShape.Layers, model.Desc.LayerCount);
        Assert.Single(model.Plan.Route.Segments);
        Assert.Equal(CudaDeviceId, model.Plan.Route.Segments[0].DeviceId);
    }

    [Fact]
    public void CpuOnlyRoute_NamingFormatCudaLacks_OpensUnchanged_AndItsBackendClaimsEveryFormat()
    {
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices("gate-cpu-iq3s", QuantFormat.IQ3_S, QuantizedShape);
        _tempFiles.Add(path);

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend cpu = new(threads: 2);

        Assert.True(cpu.SupportsGemv(QuantFormat.IQ3_S), "the CPU backend stopped claiming a GEMV kernel for a format it executes");

        using TransformerModel defaultRoute = TransformerModel.Open(gguf, cpu, maxContextTokens: 32, logLine: null, PrefillKernel.GemV);
        Assert.Single(defaultRoute.Plan.Route.Segments);
        Assert.Equal(ModelRoute.CpuDeviceId, defaultRoute.Plan.Route.Segments[0].DeviceId);

        ModelRoute cpuRoute = ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "gate-cpu-iq3s",
            Stages = [new RouteStage(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId)],
            AvailableDevices = [ModelRoute.CpuDeviceId],
        });

        using TransformerModel declaredRoute = TransformerModel.Open(gguf, cpu, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, cpuRoute);
        Assert.Equal(QuantizedShape.Layers, declaredRoute.Desc.LayerCount);
        Assert.Contains(QuantFormat.IQ3_S, declaredRoute.MatmulWeightFormats);
    }

    // ---------------------------------------------------------------- capability facts pinned

    [Theory]
    [InlineData(QuantFormat.F32, true)]
    [InlineData(QuantFormat.F16, true)]
    [InlineData(QuantFormat.BF16, true)]
    [InlineData(QuantFormat.Q4_K, true)]
    [InlineData(QuantFormat.Q6_K, true)]
    [InlineData(QuantFormat.Q8_0, false)]
    [InlineData(QuantFormat.Q5_0, false)]
    [InlineData(QuantFormat.Q5_K, false)]
    [InlineData(QuantFormat.Q2_K, false)]
    [InlineData(QuantFormat.Q3_K, false)]
    [InlineData(QuantFormat.IQ2_XXS, false)]
    [InlineData(QuantFormat.IQ2_XS, false)]
    [InlineData(QuantFormat.IQ2_S, false)]
    [InlineData(QuantFormat.IQ3_XXS, false)]
    [InlineData(QuantFormat.IQ3_S, false)]
    [InlineData(QuantFormat.IQ4_NL, false)]
    [InlineData(QuantFormat.IQ4_XS, false)]
    [InlineData(QuantFormat.Q2_0, false)]
    public void CudaGemvCapability_CoversExactlyTheFormatsTheSeamCarries(QuantFormat format, bool supported) =>
        Assert.Equal(supported, CudaBackend.HasGemvKernel(format));

    // ---------------------------------------------------------------- helpers

    private const string CudaDeviceId = CudaBackend.DeviceIdValue;

    /// <summary>
    /// A compute backend answering device <c>cuda</c> with the CUDA backend's own GEMV capability facts while its kernels
    /// stay host-side: the load-time gate decides on the capability answer and the tensor table alone, so the refusal these
    /// tests pin never reaches a kernel and no test opens a driver. The wrapped CPU backend stays owned by the test.
    /// </summary>
    private sealed class CudaCapabilityStandIn : IBackend
    {
        private readonly CpuBackend _inner;

        public CudaCapabilityStandIn(CpuBackend inner)
        {
            ArgumentNullException.ThrowIfNull(inner);
            _inner = inner;
        }

        public string DeviceId => CudaBackend.DeviceIdValue;

        public Lair.Contracts.WeightTier WeightTier => Lair.Contracts.WeightTier.Dram;

        public bool SupportsGemv(QuantFormat format) => CudaBackend.HasGemvKernel(format);

        public bool SupportsGemm(QuantFormat format) => false;

        public MemoryHandle AllocateAligned(long bytes, int alignment) => _inner.AllocateAligned(bytes, alignment);

        public void FreeAligned(MemoryHandle handle) => _inner.FreeAligned(handle);

        public IKvDeviceMirror OpenKvMirror(KvMirrorRequest request) => _inner.OpenKvMirror(request);

        public void Gemv(ReadOnlySpan<float> x, WeightRef w, Span<float> output) => _inner.Gemv(x, w, output);

        public void GemvBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs) => _inner.GemvBatch(inputs, count, w, outputs);

        public void GemmBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs) => throw new NotSupportedException("this stand-in carries no blocked matmul");

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
