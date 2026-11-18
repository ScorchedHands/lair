using System.Buffers;

using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Backends.Cuda;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// CUDA parity over the synthetic F16 llama fixture: one model opened whole on CpuBackend, the same file opened whole
/// over CudaBackend as a single 'cuda' segment at DRAM residency, and a split route whose first span runs layer 0 on CPU
/// while its receiving span takes every later layer plus the output head onto CUDA. Greedy token streams must agree
/// across all three — argmax agreement is what a device crossing guarantees, not raw logit bytes between devices — and
/// each weight tensor reaches device memory exactly once per backend instance. Weight GEMV runs on the device for every format this seam
    /// accepts today, blocked matmul runs there too for the block-quant formats and the IQ4 pair (F32, F16, BF16, and the Q8_K i-quants still refuse it naming that
    /// backend), and a cuda-named route whose compute backend reports a weight

/// format with no GEMV kernel refuses at model open — the load-time capability gate — rather than after a weight pass has dispatched.
/// </summary>
[Trait("Category", "Parity")]
[Trait("Category", "Cuda")]
public sealed class CudaBackendParityTests : IDisposable
{
    private const int DecodeSteps = 4;

    // The synthetic fixture ships two layers, so the split hands layer 0 to CPU and every later layer plus the head to CUDA.
    private const int FirstSpanLayers = 1;

    private static readonly int[] PromptTokens = [1, 2, 3];

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
                // a temp file this run could not remove is no one's problem but its own
            }
        }
    }

    [Fact]
    public void CudaParity_F16Transformer_CpuOnlyCudaOnlyAndSplitRoute_DecodeSameTokens()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam runs on Windows against nvcuda.dll.");
        }

        string path = SyntheticLlamaGguf.Write("parity-cuda-f16", withReasoningMarkers: true, QuantFormat.F16);
        _tempFiles.Add(path);

        List<int> cpuTokens;
        List<int> cudaTokens;
        int deviceWeightCopies;
        int expectedDeviceWeights;
        List<int> hybridCpuFirstTokens;
        List<int> hybridCudaFirstTokens;

        try
        {
            using GgufFile gguf = GgufFile.Open(path);

            // CPU-only baseline: the default single-CPU-segment route every unsplit open resolves today.
            using (CpuBackend cpu = new())
            {
                using TransformerModel model = TransformerModel.Open(gguf, cpu, maxContextTokens: 32, logLine: null, PrefillKernel.GemV);
                expectedDeviceWeights = DenseMatrixCount(model) + 1; // every per-layer dense matrix plus the tied output head over token-embedding rows
                cpuTokens = GreedyDecode(model, cpu);
            }

            // CUDA-only model: one segment naming 'cuda' at DRAM residency opened whole over CudaBackend.
            using (CudaBackend cuda = new())
            {
                ModelRoute route = CudaWholeModelRoute("parity-cuda-whole");
                using TransformerModel model = TransformerModel.Open(gguf, cuda, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, route);
                cudaTokens = GreedyDecode(model, cuda);
                deviceWeightCopies = cuda.DeviceWeightCopies; // one copy per dense matrix and the head across every decode step taken here
            }

            hybridCpuFirstTokens = RunSplitRoute(gguf, cpuFirst: true);
            hybridCudaFirstTokens = RunSplitRoute(gguf, cpuFirst: false);
        }
        catch (EntryPointNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point.", ex);
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API.", ex);
        }

        Assert.Equal(cpuTokens, cudaTokens);
        Assert.Equal(cpuTokens, hybridCpuFirstTokens);
        Assert.Equal(cpuTokens, hybridCudaFirstTokens);
        Assert.Equal(expectedDeviceWeights, deviceWeightCopies); // weights crossed to device once per tensor for the whole run, never per GEMV call
    }

    // The capability facts this file pins today: every format CudaBackend accepts resolves a GEMV function handle, so weight GEMV runs on the
    // device for all of them; its windowed blocked-matmul kernels cover the block-quant formats and the IQ4 pair, while the unquantized F32/F16/BF16 trio
    // and the Q8_K i-quant family refuse it. The load-time refusal lane therefore runs against a test-local stand-in that answers 'cuda' while reporting one
    // chosen format as missing — what stays pinned is the gate's decision, wording, and no-CPU-fallback posture, not a coverage gap this seam no longer has.
    [Fact]
    public void CudaParity_SplitRouteRefusesCrossingWithoutItsBackend_AndFormatsAStandInCudaLacksRefuseNamingIt()
    {
        string path = SyntheticLlamaGguf.Write("parity-cuda-refusal", withReasoningMarkers: true, QuantFormat.F16);
        _tempFiles.Add(path);

        ModelRoute split = SplitRoute("parity-cuda-refusal");

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend cpu = new();

        // A crossing to a device no opened backend reports stays the load-time refusal it has always been.
        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, cpu, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, split));
        Assert.Contains("segment 1 ('cuda' at 'Dram') names a device this engine has no backend for", error.Message);
        Assert.Contains("the opened compute backend reports 'cpu'", error.Message);

        using CudaBackend cuda = new();

        // Formats the seam executes reach this backend and run there; a packed quant claims its own windowed blocked matmul, while an unquantized matrix is refused naming model-format-backend.
        Assert.True(cuda.SupportsGemv(QuantFormat.IQ4_XS), "the CUDA seam stopped executing its IQ4_XS GEMV");
        Assert.True(cuda.SupportsGemm(QuantFormat.IQ4_XS), "the CUDA seam stopped claiming its windowed IQ4_XS blocked matmul");

        TensorShape f16Shape = new([2, 256]);
        byte[] packedF16 = new byte[2 * (int)f16Shape.RowStrideBytes(QuantFormat.F16)]; // two rows of F16 packing for shape [2, 256]

        Assert.False(cuda.SupportsGemm(QuantFormat.F16), "the CUDA seam claimed a blocked matmul for an unquantized format it carries no kernel for");
        NotSupportedException gemmError = Assert.Throws<NotSupportedException>(
            () => cuda.GemmBatch(new float[256], 1, new WeightRef(packedF16, f16Shape, QuantFormat.F16), new float[2]));
        Assert.Contains("CudaBackend ('cuda')", gemmError.Message);
        Assert.Contains("no blocked matmul kernel for format F16", gemmError.Message, StringComparison.Ordinal);

        ModelException kernelRefusal = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, cuda, maxContextTokens: 32, logLine: null, PrefillKernel.Gemm, CudaWholeModelRoute("parity-cuda-gemm")));
        Assert.Contains("gemm", kernelRefusal.Message);

        // The same capability fact against a quantized file: a cuda-named route naming a format its compute backend reports no GEMV kernel for is
        // refused at model open — naming the model, the format, and this backend — before any weight pass dispatches. This seam carries an IQ3_S
        // kernel today (asserted just below), so the refusal runs through the stand-in's chosen answer: what stays pinned here is that a route can
        // no longer reach the mid-decode refusal, while nothing falls back to CPU for a device-named segment.
        string quantizedPath = SyntheticLlamaGguf.WriteQuantizedMatrices(
            "parity-cuda-iq3s", QuantFormat.IQ3_S, new SyntheticLlamaGguf.Geometry(Layers: 2, HiddenSize: 256, HeadCount: 2, KvHeadCount: 2, FeedForwardLength: 512));
        _tempFiles.Add(quantizedPath);

        Assert.True(CudaBackend.HasGemvKernel(QuantFormat.IQ3_S), "the CUDA seam stopped reporting its IQ3_S GEMV kernel"); // the refusal below is a stand-in answer, not this backend's own fact.

        using GgufFile quantizedGguf = GgufFile.Open(quantizedPath);
        using CpuBackend hostKernels = new();

        ModelException loadRefusal = Assert.Throws<ModelException>(
            () => TransformerModel.Open(quantizedGguf, new CudaCapabilityStandIn(hostKernels, QuantFormat.IQ3_S), maxContextTokens: 32, logLine: null, PrefillKernel.GemV, CudaWholeModelRoute("parity-cuda-iq3s")));
        Assert.Contains("model 'parity-cuda-iq3s'", loadRefusal.Message);
        Assert.Contains("IQ3_S", loadRefusal.Message);
        Assert.Contains("no GEMV kernel", loadRefusal.Message);
    }

    // ---------------------------------------------------------------- helpers

    private List<int> RunSplitRoute(GgufFile gguf, bool cpuFirst)
    {
        string modelName = cpuFirst ? "parity-cuda-split" : "parity-cuda-reversed";
        ModelRoute route = SplitRoute(modelName);

        using CpuBackend firstSpan = new();
        using CudaBackend receiving = new();

        if (cpuFirst)
        {
            // prefill stage starts the split on CPU at layer 0; decode hands every later layer to CUDA.
            using TransformerModel cpuFirstModel = TransformerModel.Open(gguf, firstSpan, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, route, [receiving]);
            return GreedyDecode(cpuFirstModel, firstSpan);
        }

        // Reversed posture: the opened backend is CUDA and CPU answers the crossing position after it.
        ModelRoute reversed = ReversedSplitRoute(modelName);
        using TransformerModel cudaFirstModel = TransformerModel.Open(gguf, receiving, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, reversed, [firstSpan]);
        return GreedyDecode(cudaFirstModel, receiving);
    }

    private static List<int> GreedyDecode(TransformerModel model, IBackend poolOwner)
    {
        using BlockPool pool = new(
            blockTokens: 16,
            layerCount: model.Desc.LayerCount,
            kvHeads: model.Desc.KvHeadCount,
            headDim: model.Desc.HeadDim,
            capacityBlocks: 16,
            poolOwner);

        using SequenceKv kv = new(pool);

        int token = ArgMax(model.Prefill(PromptTokens, kv));
        List<int> sampled = [token];

        for (int step = 0; step < DecodeSteps; step++)
        {
            token = ArgMax(model.DecodeNext(token, kv));
            sampled.Add(token);
        }

        return sampled;
    }

    private static int DenseMatrixCount(TransformerModel model) => model.Desc.LayerCount * 7; // q,k,v,o,gate,up,down per layer

    private static ModelRoute CudaWholeModelRoute(string modelName) =>
        ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = modelName,
            Stages = [new(ModelRoutes.PrefillStage, CudaBackend.DeviceIdValue), new(ModelRoutes.DecodeStage, CudaBackend.DeviceIdValue)],
            AvailableDevices = [ModelRoute.CpuDeviceId, CudaBackend.DeviceIdValue],
        });

    private static ModelRoute SplitRoute(string modelName) =>
        ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = modelName,
            Stages =
            [
                new RouteStage(ModelRoutes.PrefillStage, ModelRoute.CpuDeviceId) { SplitPosition = 0 },
                new RouteStage(ModelRoutes.DecodeStage, CudaBackend.DeviceIdValue) { SplitPosition = FirstSpanLayers },
            ],
            AvailableDevices = [ModelRoute.CpuDeviceId, CudaBackend.DeviceIdValue],
        });

    private static ModelRoute ReversedSplitRoute(string modelName) =>
        ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = modelName,
            Stages =
            [
                new RouteStage(ModelRoutes.PrefillStage, CudaBackend.DeviceIdValue) { SplitPosition = 0 },
                new RouteStage(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId) { SplitPosition = FirstSpanLayers },
            ],
            AvailableDevices = [ModelRoute.CpuDeviceId, CudaBackend.DeviceIdValue],
        });

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

    /// <summary>
    /// A compute backend answering device <c>cuda</c> with the CUDA backend's own GEMV capability facts while its kernels stay host-side, so a
    /// refusal lane can pin the load-time gate against a format this seam actually carries today. Formats named in <c>cudaMissingFormats</c>
    /// report as unsupported here even though the real seam resolves a kernel for every accepted format — that is how the refusal lanes keep
    /// exercising gate machinery over a capability answer this stand-in chooses to give, without production claiming a gap it no longer has. The
    /// wrapped CPU backend stays owned by the test.
    /// </summary>
    private sealed class CudaCapabilityStandIn : IBackend
    {
        private readonly CpuBackend _inner;
        private readonly QuantFormat[] _cudaMissingFormats;

        public CudaCapabilityStandIn(CpuBackend inner, params QuantFormat[] cudaMissingFormats)
        {
            ArgumentNullException.ThrowIfNull(inner);
            _inner = inner;
            _cudaMissingFormats = cudaMissingFormats ?? [];
        }

        public string DeviceId => CudaBackend.DeviceIdValue;

        public Lair.Contracts.WeightTier WeightTier => Lair.Contracts.WeightTier.Dram;

        public bool SupportsGemv(QuantFormat format) => Array.IndexOf(_cudaMissingFormats, format) < 0 && CudaBackend.HasGemvKernel(format);

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

