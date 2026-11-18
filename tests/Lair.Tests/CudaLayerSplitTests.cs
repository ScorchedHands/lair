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
    /// Layer-split parity over the synthetic F16 llama fixture: one model opened CPU-only on its default route decodes greedily, and the same file reopened at every split boundary names a hybrid route whose first span runs layers 0 through that boundary on device 'cuda' while every later layer — including the output head's zero-layer tail position — stays on cpu. The host posture is what an opened engine carries: this service-side CPU backend answers the pool and the decode-side facts, and an ordered execution set of [cuda, cpu] answers each resolved segment in order so a CUDA-first span runs even when the engine itself was opened over CPU. Every boundary must land token-identical to the CPU baseline for prompt window and continued steps alike — argmax agreement across a device crossing, not raw logit bytes between devices — while the CUDA instance reports the CUDA-resident span's transformer matrices crossed to device exactly once per reopened model, while the cpu-side tail answers its own range and the output head from CPU-side storage. The staging facts name one cpu-side tail span from each boundary through this file's last layer, and an opened CPU wrapper proves that tail actually ran there: it saw weight passes below the whole-model baseline rather than leaving some part of the split unrun. The full-layer split is proven as a contract too: with the final boundary at the layer count, every transformer layer runs on CUDA while the tail span — no layers, only the output head — still executes on the opened CPU backend. A boundary outside this file's own range stays the load-time refusal it has always been, and an execution set that covers neither posture refuses naming what lands short rather than running some part of the model somewhere else.

/// </summary>
[Trait("Category", "Parity")]
[Trait("Category", "Cuda")]
public sealed class CudaLayerSplitTests : IDisposable
{
    private const int DecodeSteps = 4;

    // The synthetic fixture ships two layers, so the sweep covers boundary 1 (one layer on CUDA) and boundary 2 (every transformer layer on CUDA).
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
    public void LayerSplit_EveryBoundaryUpToLayerCount_MatchesCpuOnlyBaseline_AndEachSpanRunsOnItsOwnBackend()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            throw new InvalidOperationException("The CUDA driver seam runs on Windows against nvcuda.dll.");
        }

        string path = SyntheticLlamaGguf.Write("split-gpu-layers-f16", withReasoningMarkers: true, QuantFormat.F16);
        _tempFiles.Add(path);

        List<int> baselineTokens;
        int matricesPerLayer;
        int baselineCpuPasses;

        using GgufFile gguf = GgufFile.Open(path);

        // CPU-only baseline: the default single-CPU-segment route every unsplit open resolves today.
        try
        {
            using CpuBackend cpuOnlyInner = new();
            CountingDecodeBackend countingBaseline = new(cpuOnlyInner); // the whole-model CPU posture, visible as every weight pass it answers.
            using TransformerModel baseline = TransformerModel.Open(gguf, countingBaseline, maxContextTokens: 32, logLine: null, PrefillKernel.GemV);

            Assert.Single(baseline.Plan.Route.Segments);
            Assert.Equal(ModelRoute.CpuDeviceId, baseline.Plan.Route.Segments[0].DeviceId);

            matricesPerLayer = baseline.Desc.LayerCount == 0 ? 7 : DenseMatrixCount(baseline) / baseline.Desc.LayerCount; // the dense matrix families each transformer layer carries — q,k,v,o,gate,up,down in this fixture.
            baselineTokens = GreedyDecode(baseline, countingBaseline);
            baselineCpuPasses = countingBaseline.GemvCalls + countingBaseline.GemvBatchCalls; // every counted pass the whole model took on its opened CPU backend.
        }
        catch (EntryPointNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point.", ex);
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API.", ex);
        }

        int layerCount = ModelLayerCount(path);

        for (int boundary = 1; boundary <= layerCount; boundary++)
        {
            List<int> hybridTokens;
            int cudaWeightCopies;
            int cpuTailGemvCalls;
            TransformerModel.ExecutionCrossing[] stagingFacts;

            using CpuBackend decodeSideInner = new();
            CountingDecodeBackend countingDecodeSide = new(decodeSideInner); // the service-side CPU backend, visible as it answers every span its route resolved onto it.
            IBackend decodeSide = countingDecodeSide;

            try
            {
                using CudaBackend cudaFirst = new();

                ModelRoute hybrid = HybridSplitRoute($"split-gpu-layers-{boundary}", boundary);

                // The host posture: this engine's own opened backend stays CPU — it answers the pool and every decode-side fact — while an ordered execution set names each resolved segment in order, including the first CUDA span a hybrid route puts ahead of it.
                using TransformerModel hybridModel = TransformerModel.Open(
                    gguf, decodeSide, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, hybrid, [(IBackend)cudaFirst, (IBackend)decodeSide]); // both resolved positions are answered explicitly.

                Assert.Equal(2, hybrid.Segments.Count);
                Assert.Equal(CudaBackend.DeviceIdValue, hybrid.Segments[0].DeviceId); // the first segment names device 'cuda' from layer 0.
                Assert.Equal(ModelRoute.CpuDeviceId, hybrid.Segments[1].DeviceId);    // every later layer — and the head at this tail position — runs on cpu.

                stagingFacts = [.. hybridModel.ExecutionStaging];
                hybridTokens = GreedyDecode(hybridModel, decodeSide);

                cudaWeightCopies = cudaFirst.DeviceWeightCopies; // each tensor crossed to device once for this model's whole run, never per weight pass
                cpuTailGemvCalls = countingDecodeSide.GemvCalls + countingDecodeSide.GemvBatchCalls;
            }
            catch (EntryPointNotFoundException ex)
            {
                throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded or lacks a required entry point.", ex);
            }
            catch (DllNotFoundException ex)
            {
                throw new InvalidOperationException("CUDA driver seam unavailable: nvcuda.dll could not be loaded. Install or enable an NVIDIA driver exposing the CUDA Driver API.", ex);
            }

            Assert.Equal(baselineTokens, hybridTokens); // token-identical prompt and continued steps at boundary {boundary} — argmax agreement across the crossing changes nothing else about the run

            TransformerModel.ExecutionCrossing tail = Assert.Single(stagingFacts); // one crossing into the cpu-side tail span
            Assert.Equal(1, tail.SegmentIndex);
            Assert.Equal(boundary, tail.LayerStart);
            Assert.Equal(layerCount, tail.LayerEnd); // at a boundary equal to this file's layer count the range is empty: every transformer layer ran earlier and only the output head remains on cpu.
            Assert.Equal(ModelRoute.CpuDeviceId, tail.StagingOwnerDeviceId);

            Assert.True(cpuTailGemvCalls > 0, "the opened CPU backend answered no weight pass at this split — its tail span never executed there");
            Assert.True(cpuTailGemvCalls < baselineCpuPasses, $"boundary {boundary} moved layers to CUDA but the opened CPU backend still ran every counted pass this model took");

            Assert.Equal(boundary * matricesPerLayer, cudaWeightCopies); // the CUDA-resident span's transformer matrices crossed to device once for this model's run — the cpu-side tail answers its own range and the output head from CPU-side storage.
        }
    }

    [Fact]
    public void LayerSplit_BoundaryPastTheLayerCount_RefusesNamingWhatRunsNothing()
    {
        string path = SyntheticLlamaGguf.Write("split-gpu-layers-past"); // the shape check lands before any weight pass, so no CUDA device is consulted.

        ModelRoute route = ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "split-gpu-layers-past",
            Stages =
            [
                new RouteStage(ModelRoutes.PrefillStage, CudaBackend.DeviceIdValue) { SplitPosition = 0 },
                new RouteStage(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId) { SplitPosition = 5 }, // the fixture ships two layers: a span starting past them runs nothing
            ],
            AvailableDevices = [ModelRoute.CpuDeviceId, CudaBackend.DeviceIdValue],
        });

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend decodeSide = new(threads: 2);
        using CudaBackend cudaAnswer = new(); // a real CUDA-backed first span so the refusal under test is about the boundary, not an absent device.

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, decodeSide, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, route, [cudaAnswer, (IBackend)decodeSide]));

        Assert.Contains("starts at boundary '5', past this file's last layer", error.Message);
    }

    [Fact]
    public void LayerSplit_ExecutionSetCoveringNeitherPosture_RefusesNamingWhatLandsShort()
    {
        string path = SyntheticLlamaGguf.Write("split-gpu-layers-partial"); // the count check lands before any device is consulted.

        ModelRoute route = TwoCpuSegmentHybridRoute("split-gpu-layers-partial", boundary: 2); // supplied one backend where a two-segment split wants either every position after the first or both resolved positions — three answers disagree with each other, and none covers this shape.

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend decodeSide = new(threads: 2);

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, decodeSide, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, route, [])); // an empty execution set matches neither exactly one crossing nor every resolved position in order — this open can cover no shape it could run.

        Assert.Contains("resolves a route of 2 segment(s)", error.Message);
        Assert.Contains("needs one execution backend for every position after the first", error.Message);
    }

    // ---------------------------------------------------------------- helpers

    private static int ModelLayerCount(string path)
    {
        using GgufFile probe = GgufFile.Open(path);
        return ModelPlans.Resolve(probe, logLine: null).LayerCount;
    }

    /// <summary>The hybrid route a host's layer split resolves to: device 'cuda' from layer 0 through the boundary, cpu from the boundary on — at the layer count that tail span carries no transformer layers and only the output head.</summary>
    private static ModelRoute HybridSplitRoute(string modelName, int boundary) =>
        ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = modelName,
            Stages =
            [
                new RouteStage(ModelRoutes.PrefillStage, CudaBackend.DeviceIdValue) { SplitPosition = 0 },
                new RouteStage(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId) { SplitPosition = boundary },
            ],
            AvailableDevices = [ModelRoute.CpuDeviceId, CudaBackend.DeviceIdValue],
        });

    /// <summary>A two-segment split whose spans both name cpu — the shape a device answer cannot pre-empt, so refusal tests can inspect resolution order without consulting any driver.</summary>
    private static ModelRoute TwoCpuSegmentHybridRoute(string modelName, int boundary) =>
        ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = modelName,
            Stages =
            [
                new RouteStage(ModelRoutes.PrefillStage, ModelRoute.CpuDeviceId) { SplitPosition = 0 },
                new RouteStage(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId) { SplitPosition = boundary },
            ],
            AvailableDevices = [ModelRoute.CpuDeviceId],
        });

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

    /// <summary>A CPU compute backend that counts the weight passes it was asked to run: what a span resolved onto this engine's own device executes is visible as calls against its kernels.</summary>
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

        public IKvDeviceMirror OpenKvMirror(KvMirrorRequest request) => _inner.OpenKvMirror(request);

        public void Gemv(ReadOnlySpan<float> x, WeightRef w, Span<float> output)
        {
            GemvCalls++;
            _inner.Gemv(x, w, output);
        }

        public void GemvBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs)
        {
            GemvBatchCalls++;
            _inner.GemvBatch(inputs, count, w, outputs); // a batched-row weight pass through this engine's own kernels; the CPU backend answers it bit-identically to its single-vector path.
        }

        public void GemmBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs) => _inner.GemmBatch(inputs, count, w, outputs);

        public bool SupportsGemv(QuantFormat format) => _inner.SupportsGemv(format);

        public bool SupportsGemm(QuantFormat format) => _inner.SupportsGemm(format);

        public void DequantRow(WeightRef w, int row, Span<float> output) => _inner.DequantRow(w, row, output);

        public float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b) => _inner.Dot(a, b);

        public void Axpy(float alpha, ReadOnlySpan<float> x, Span<float> y) => _inner.Axpy(alpha, x, y);

        public void Fill(Span<float> y, float v) => _inner.Fill(y, v);

        public void Scale(Span<float> y, float s) => _inner.Scale(y, s);

        public void Dispose() => _inner.Dispose();
    }
}
