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
/// each weight tensor reaches device memory exactly once per backend instance. Formats with no CUDA kernel refuse naming
/// that backend instead of running some other route's math.
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

    [Fact]
    public void CudaParity_SplitRouteRefusesCrossingWithoutItsBackend_AndQuantizedFormatsRefuseNamingCuda()
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

        // Quantized weight formats reach this backend only to be refused naming it — never a silent CPU fallback.
        byte[] packedQ4K = new byte[2 * 144]; // two rows of Q4_K packing for shape [2, 256]

        NotSupportedException gemvError = Assert.Throws<NotSupportedException>(
            () => cuda.Gemv(new float[256], new WeightRef(packedQ4K, new TensorShape([2, 256]), QuantFormat.Q4_K), new float[2]));
        Assert.Contains("CudaBackend ('cuda')", gemvError.Message);
        Assert.Contains("Q4_K", gemvError.Message);

        NotSupportedException gemmError = Assert.Throws<NotSupportedException>(
            () => cuda.GemmBatch(new float[256], 1, new WeightRef(packedQ4K, new TensorShape([2, 256]), QuantFormat.Q4_K), new float[2]));
        Assert.Contains("CudaBackend ('cuda')", gemmError.Message);
        Assert.False(cuda.SupportsGemm(QuantFormat.F16)); // the blocked matmul has no CUDA path yet at all

        ModelException kernelRefusal = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, cuda, maxContextTokens: 32, logLine: null, PrefillKernel.Gemm, CudaWholeModelRoute("parity-cuda-gemm")));
        Assert.Contains("gemm", kernelRefusal.Message);
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
}

