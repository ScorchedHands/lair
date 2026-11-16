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
/// Prefill offload correctness gate over the synthetic F16 llama fixture: a CPU-only baseline opened on its default
/// single-CPU DRAM segment decodes greedily, and the same file reopened over that exact route with the main backend a
/// CpuBackend and the prompt-window side alone a CudaBackend — no lane layout behind it. Every token of the offloaded
/// engine's greedy decode must match the baseline argmax for argmax, while the CUDA instance reports every dense matrix
/// plus the tied output head copied to device exactly once (what only the prompt windows crossed) and the opened CPU
/// backend still answers every per-token weight pass. Posture refusals keep their wording: an offload backend over a
/// split route refuses naming what lands short, and a dedicated lane layout without its execution backend is still half
/// of a pair — nothing runs somewhere else behind the posture that happens to work.
/// </summary>
[Trait("Category", "Parity")]
[Trait("Category", "Cuda")]
public sealed class CudaPrefillOffloadTests : IDisposable
{
    private const int DecodeSteps = 4;

    // The fixture ships two layers, so a split refusal names boundaries inside its own layer range.
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
    public void PrefillOffloadedToCuda_DecodeStaysOnCpu_MatchesBaselineTokensAndCopiesCounts()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam runs on Windows against nvcuda.dll.");
        }

        string path = SyntheticLlamaGguf.Write("parity-cuda-prefilloffload", withReasoningMarkers: true, QuantFormat.F16);
        _tempFiles.Add(path);

        ModelRoute baselineRoute;
        List<int> cpuTokens;
        int expectedLaneCopies;
        List<int> offloadedTokens;
        int laneWeightCopies;
        int decodeSideGemvCalls;

        try
        {
            using GgufFile gguf = GgufFile.Open(path);

            // CPU-only baseline: the default single-CPU DRAM segment every unsplit open resolves, greedy tokens recorded.
            using (CpuBackend cpu = new())
            {
                using TransformerModel baseline = TransformerModel.Open(gguf, cpu, maxContextTokens: 32, logLine: null, PrefillKernel.GemV);

                // The resolved route this open landed on — the same value set the offloaded engine reopens over unchanged.
                baselineRoute = baseline.Plan.Route;
                Assert.Single(baselineRoute.Segments);
                Assert.Equal(ModelRoute.CpuDeviceId, baselineRoute.Segments[0].DeviceId);

                expectedLaneCopies = DenseMatrixCount(baseline) + 1; // every per-layer dense matrix plus the tied output head over token-embedding rows
                cpuTokens = GreedyDecode(baseline, cpu);
            }

            using (CpuBackend decodeSideInner = new())
            {
                CountingDecodeBackend decodeSide = new(decodeSideInner);
                using CudaBackend prefillsLane = new();

                // The offload posture: CPU answers every decode-side span of this route, and the CUDA backend alone carries
                // prompt windows over the same DRAM-resident weights — no lane layout behind it.
                using TransformerModel offloaded = TransformerModel.Open(
                    gguf, decodeSide, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, baselineRoute, prefillsBackend: prefillsLane);

                Assert.Equal(PrefillKernel.GemV, offloaded.ResolvedPrefillKernel); // the batched-row window shape; CUDA carries no blocked matmul
                Assert.Same(baselineRoute, offloaded.Plan.Route);                  // decode-side posture is exactly what the baseline resolved

                offloadedTokens = GreedyDecode(offloaded, decodeSide);

                laneWeightCopies = prefillsLane.DeviceWeightCopies; // every dense matrix and the head crossed to device once across the prompt windows taken here
                decodeSideGemvCalls = decodeSide.GemvCalls + decodeSide.GemvBatchCalls; // per-token weight passes ran on this engine's own CPU backend, not through the lane
            }
        }
        catch (EntryPointNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point.", ex);
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API.", ex);
        }

        Assert.Equal(cpuTokens, offloadedTokens); // token-identical decode across postures — GPU prompt windows beside CPU/hybrid decode change no argmax
        Assert.True(decodeSideGemvCalls > 0, "the opened CPU backend answered no weight pass: the engine's decode side never ran on it");
        Assert.Equal(expectedLaneCopies, laneWeightCopies); // prompt windows crossed every dense matrix to CUDA exactly once per tensor — none of them fell back through some other route
    }

    [Fact]
    public void PrefillOffloadBackendOverSplitRoute_RefusesNamingWhatItDoesNotCarry()
    {
        string path = SyntheticLlamaGguf.Write("prefilloffload-split-refusal"); // the posture check lands before any weight pass, so no CUDA device is consulted

        ModelRoute route = TwoCpuSegmentSplitRoute("prefilloffload-split");

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend decodeSide = new(threads: 2);
        using CpuBackend lane = new(threads: 1);

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, decodeSide, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, route, prefillsBackend: lane));

        Assert.Contains("offloaded prefill backend on device 'cpu' arrives with no lane layout over a route of 2 segments", error.Message);
        Assert.Contains("one DRAM-resident span", error.Message);
    }

    [Fact]
    public void DedicatedLaneLayoutWithoutItsBackend_StillRefusesHalfOfThePair()
    {
        string path = SyntheticLlamaGguf.Write("prefilloffload-halfpair-refusal");

        ModelRoute route = SingleCpuSegmentRoute("prefilloffload-halfpair");
        PrefillLaneLayout layout = ModelPrefillLanes.ResolveDedicated(
            "prefilloffload-halfpair", engineThreadsChoice: 4, threadQuota: null, coreSet: new HashSet<int>());

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend main = new();

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, main, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, route, prefillsLane: layout));

        Assert.Contains("arrives with only half of its pair", error.Message);
        Assert.Contains("no execution backend for the lane to run on", error.Message);
    }

    // ---------------------------------------------------------------- helpers

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

    private static ModelRoute SingleCpuSegmentRoute(string modelName) =>
        ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = modelName,
            Stages = [new(ModelRoutes.PrefillStage, ModelRoute.CpuDeviceId), new(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId)],
            AvailableDevices = [ModelRoute.CpuDeviceId],
        });

    private static ModelRoute TwoCpuSegmentSplitRoute(string modelName) =>
        ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = modelName,
            Stages =
            [
                new RouteStage(ModelRoutes.PrefillStage, ModelRoute.CpuDeviceId) { SplitPosition = 0 },
                new RouteStage(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId) { SplitPosition = FirstSpanLayers },
            ],
            AvailableDevices = [ModelRoute.CpuDeviceId],
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

    /// <summary>A CPU compute backend that counts the weight passes it was asked to run: what this engine's own spans execute is visible as calls against its own kernels.</summary>
    private sealed class CountingDecodeBackend : IBackend
    {
        private readonly CpuBackend _inner;

        public CountingDecodeBackend(CpuBackend inner)
        {
            ArgumentNullException.ThrowIfNull(inner);
            _inner = inner;
        }

        /// <summary>Single-token weight passes this backend ran.</summary>
        public int GemvCalls { get; private set; }

        /// <summary>Batched-row weight passes this backend ran.</summary>
        public int GemvBatchCalls { get; private set; }

        public string DeviceId => _inner.DeviceId;

        public Lair.Contracts.WeightTier WeightTier => Lair.Contracts.WeightTier.Dram;

        public MemoryHandle AllocateAligned(long bytes, int alignment) => _inner.AllocateAligned(bytes, alignment);

        public void FreeAligned(MemoryHandle handle) => _inner.FreeAligned(handle);

        public void Gemv(ReadOnlySpan<float> x, WeightRef w, Span<float> output)
        {
            GemvCalls++;
            _inner.Gemv(x, w, output);
        }

        public void GemvBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs)
        {
            GemvBatchCalls++;
            _inner.GemvBatch(inputs, count, w, outputs);
        }

        public void GemmBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs) => _inner.GemmBatch(inputs, count, w, outputs);

        public bool SupportsGemm(QuantFormat format) => _inner.SupportsGemm(format);

        public void DequantRow(WeightRef w, int row, Span<float> output) => _inner.DequantRow(w, row, output);

        public float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b) => _inner.Dot(a, b);

        public void Axpy(float alpha, ReadOnlySpan<float> x, Span<float> y) => _inner.Axpy(alpha, x, y);

        public void Fill(Span<float> y, float v) => _inner.Fill(y, v);

        public void Scale(Span<float> y, float s) => _inner.Scale(y, s);

        public void Dispose() => _inner.Dispose();
    }
}
