using System.Buffers;
using System.Runtime.InteropServices;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Split-route execution tests over the synthetic two-layer llama fixture: a route whose ordered segments agree on device and residency but name distinct boundaries resolves to two spans of layers, runs each span's weight passes through its own compute backend, and hands only the residual rows between them — staged in arenas the receiving backend allocated at open. Every forward across such a split lands bit-for-bit on the same model run whole-model on one backend: logits raw bytes, sampled-token streams continued step by step, merged-batch rows, and every KV byte of every position and layer are identical to the single-backend reference, while the engine's own staging diagnostics name which backend owns each crossing arena. An ordinary CPU placement still collapses prefill and decode into exactly one segment and stages nothing at all: two spans that agree on device, residency, and boundary remain one span, and a route this build cannot run — an unsupplied crossing position, a dedicated lane posture over a split, or boundaries that fail to cover the file's layers once — refuses naming what lands short rather than running some part of the model somewhere else.
/// </summary>
[Trait("Category", "Scheduling")]
[Trait("Category", "Loader")]
public sealed class SplitRouteExecutionTests : IDisposable
{
    private readonly List<string> _tempFiles = [];

    // The synthetic fixture ships two layers, so a split at boundary 1 hands layer 0 to the first backend and layer 1 to the receiving one.
    private const int FirstSpanLayers = 1;

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

    /// <summary>The kernels this build executes, so both window routes are proven across the crossing.</summary>
    public static TheoryData<PrefillKernel> PrefillKernels() => [PrefillKernel.GemV, PrefillKernel.Tiled];

    // ---------------------------------------------------------------- split identity and staging ownership

    [Fact]
    public void OrderedCpuStages_NamingDistinctBoundaries_ResolveToTwoSpans_OnOneDeviceAndTier()
    {
        ModelRoute route = ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "split-identity",
            Stages =
            [
                new RouteStage(ModelRoutes.PrefillStage, ModelRoute.CpuDeviceId) { SplitPosition = 0 },
                new RouteStage(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId) { SplitPosition = FirstSpanLayers },
            ],
            AvailableDevices = [ModelRoute.CpuDeviceId],
        });

        Assert.Equal(2, route.Segments.Count);
        foreach (DeviceSegment segment in route.Segments)
        {
            Assert.Equal(ModelRoute.CpuDeviceId, segment.DeviceId);
            Assert.Equal(WeightTier.Dram, segment.WeightTier);
        }

        Assert.Equal(0, route.Segments[0].SplitPosition);
        Assert.Equal(FirstSpanLayers, route.Segments[1].SplitPosition);
    }

    [Fact]
    public void OrdinaryCpuStages_StillCollapseToOneSegment_AndStageNothingAtOpen()
    {
        string path = SyntheticLlamaGguf.Write("split-collapse");
        _tempFiles.Add(path);

        ModelRoute route = ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "split-collapse",
            Stages = [new(ModelRoutes.PrefillStage, ModelRoute.CpuDeviceId), new(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId)],
            AvailableDevices = [ModelRoute.CpuDeviceId],
        });

        DeviceSegment segment = Assert.Single(route.Segments);
        Assert.Equal(ModelRoute.WholeModelSplitPosition, segment.SplitPosition); // an unsplit route carries the whole-model position and nothing else

        using GgufFile gguf = GgufFile.Open(path);
        CountingBackend backend = new(new CpuBackend(threads: 2));
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, route, []);

        Assert.Empty(model.ExecutionStaging); // one span crosses nothing: today's activation arrays remain the only mutable streams any forward touches
    }

    [Theory]
    [MemberData(nameof(PrefillKernels))]
    public void SplitRoute_StagesOneArenaPerCrossing_OwnedByTheReceivingBackend_AndFreedWithTheModel(PrefillKernel kernel)
    {
        string path = SyntheticLlamaGguf.Write("split-staging");
        _tempFiles.Add(path);

        ModelRoute route = TwoCpuSegmentSplitRoute("split-staging");

        using GgufFile gguf = GgufFile.Open(path);
        CountingBackend firstSpan = new(new CpuBackend(threads: 2));
        CountingBackend receiving = new(new CpuBackend(threads: 2));

        // The BlockPool arena belongs to the device the model runs on — one resident pool, whatever spans execute layers.
        using TransformerModel model = TransformerModel.Open(gguf, firstSpan, maxContextTokens: 32, logLine: null, kernel, route, [receiving]);
        using BlockPool pool = NewPool(model, firstSpan, capacityBlocks: 4);

        TransformerModel.ExecutionCrossing crossing = Assert.Single(model.ExecutionStaging);
        Assert.Equal(1, crossing.SegmentIndex); // the resolved position after the first is what stages
        Assert.Equal(FirstSpanLayers, crossing.LayerStart);
        Assert.Equal(model.Desc.LayerCount, crossing.LayerEnd);
        Assert.Equal(ModelRoute.CpuDeviceId, crossing.StagingOwnerDeviceId);

        long expectedBytes = (long)Math.Max(TransformerModel.MaxBatchSequences, model.Plan.PrefillRowWidth) * model.Desc.HiddenSize * sizeof(float);
        Assert.Equal(expectedBytes, crossing.StagingBytes); // sized once at open to this engine's whole row capacity

        using SequenceKv kv = new(pool);
        model.Prefill([1, 2, 3], kv); // a window crosses: the receiving backend's arena is what it lands in

        Assert.Equal(1, receiving.AllocateCalls); // exactly one arena on the receiving side — no second KV pool behind it
        Assert.Equal(0, receiving.FreeCalls); // still held while this engine lives
        Assert.Equal(1, firstSpan.AllocateCalls); // its own allocation is the single resident BlockPool arena

        model.Dispose();

        Assert.Equal(1, receiving.FreeCalls); // the staging arena returns to the backend that allocated it before any host disposes backends
    }

    [Fact]
    public void SplitRoute_RunsNoSecondKvArena_AndEveryLayerWritesIntoTheOneResidentPool()
    {
        string path = SyntheticLlamaGguf.Write("split-kv-arena");
        _tempFiles.Add(path);

        ModelRoute route = TwoCpuSegmentSplitRoute("split-kv-arena");

        using GgufFile gguf = GgufFile.Open(path);
        CountingBackend firstSpan = new(new CpuBackend(threads: 2));
        CountingBackend receiving = new(new CpuBackend(threads: 2));

        float[] splitLogits;
        int usedAfterSplitForward;

        {
            using TransformerModel split = TransformerModel.Open(gguf, firstSpan, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, route, [receiving]);
            using BlockPool pool = NewPool(split, firstSpan, capacityBlocks: 4);

            // Both spans write K/V through one SequenceKv over this engine's single resident pool — the layer executed on the receiving backend still lands in that arena.
            using SequenceKv splitKv = new(pool);
            splitLogits = [.. split.Prefill([1, 2], splitKv)];

            usedAfterSplitForward = pool.UsedBlocks; // a forward that advanced two positions holds its blocks on the one pool
        }

        Assert.True(usedAfterSplitForward > 0, "the split forward reserved nothing in the resident pool");

        CountingBackend referenceSpan = new(new CpuBackend(threads: 2));
        using TransformerModel whole = TransformerModel.Open(gguf, referenceSpan, maxContextTokens: 32);
        using BlockPool referencePool = NewPool(whole, referenceSpan, capacityBlocks: 4);
        using SequenceKv wholeKv = new(referencePool);

        float[] wholeLogits = [.. whole.Prefill([1, 2], wholeKv)];

        AssertBitsEqual(wholeLogits, splitLogits, "a window across the two-span route against the same model run on one backend");
        Assert.Equal(1, receiving.AllocateCalls); // only its staging arena was ever allocated there — no pool of its own exists behind a span
    }

    // ---------------------------------------------------------------- execution identity across the crossing

    [Theory]
    [MemberData(nameof(PrefillKernels))]
    public void SplitRoute_PrefillAndContinuedSteps_MatchWholeModelForward_BitForBit(PrefillKernel kernel)
    {
        string path = SyntheticLlamaGguf.Write("split-identity-forward");
        _tempFiles.Add(path);

        ModelRoute route = TwoCpuSegmentSplitRoute("split-identity-forward");

        using GgufFile gguf = GgufFile.Open(path);

        CountingBackend referenceBackend = new(new CpuBackend(threads: 2));
        using TransformerModel whole = TransformerModel.Open(gguf, referenceBackend, maxContextTokens: 64, logLine: null, kernel);
        using BlockPool referencePool = NewPool(whole, referenceBackend, capacityBlocks: 8);

        CountingBackend firstSpan = new(new CpuBackend(threads: 2));
        CountingBackend receiving = new(new CpuBackend(threads: 2));
        using TransformerModel split = TransformerModel.Open(gguf, firstSpan, maxContextTokens: 64, logLine: null, kernel, route, [receiving]);
        using BlockPool splitPool = NewPool(split, firstSpan, capacityBlocks: 8);

        int[] prompt = [1, 2, 3, 4]; // ids inside the synthetic vocabulary; a window whose rows ride through both spans together

        using SequenceKv wholeWindow = new(referencePool);
        using SequenceKv splitWindow = new(splitPool);

        float[] wholeLogits = [.. whole.Prefill(prompt, wholeWindow)];
        float[] splitLogits = [.. split.Prefill(prompt, splitWindow)];

        AssertBitsEqual(wholeLogits, splitLogits, $"prefill of a {prompt.Length}-token window across the two-span route on the '{PrefillKernelNames.Of(kernel)}' kernel");
        Assert.Equal(ArgMax(wholeLogits), ArgMax(splitLogits)); // the same token is what either engine samples next

        // The sampled token rides into the next forward on both engines, and each step's logits must keep landing byte-identical across every crossing.
        int wholeNext = ArgMax(wholeLogits);
        int splitNext = ArgMax(splitLogits);
        using SequenceKv wholeContinuation = new(referencePool);
        using SequenceKv splitContinuation = new(splitPool);

        // Both continuations replay the same prompt first, so their KV starts identical and only the stepped forwards differ.
        float[] wholeSeed = [.. whole.Prefill(prompt, wholeContinuation)];
        float[] splitSeed = [.. split.Prefill(prompt, splitContinuation)];
        AssertBitsEqual(wholeSeed, splitSeed, "the continuation seed on both engines");

        for (int step = 0; step < 4; step++)
        {
            float[] wholeStep = [.. whole.DecodeNext(wholeNext, wholeContinuation)];
            float[] splitStep = [.. split.DecodeNext(splitNext, splitContinuation)];

            AssertBitsEqual(wholeStep, splitStep, $"decode step {step} across the two-span route on the '{PrefillKernelNames.Of(kernel)}' kernel");
            wholeNext = ArgMax(wholeStep);
            splitNext = ArgMax(splitStep);
        }

        // Every position of every layer: the KV both engines wrote through their own pools lands byte-identical, including the layers executed on the receiving backend.
        Assert.Equal(wholeContinuation.Length, splitContinuation.Length);
        for (int position = 0; position < wholeContinuation.Length; position++)
        {
            for (int layer = 0; layer < whole.Desc.LayerCount; layer++)
            {
                AssertBitsEqual(wholeContinuation.ReadK(position, layer), splitContinuation.ReadK(position, layer), $"K at position {position} layer {layer}");
                AssertBitsEqual(wholeContinuation.ReadV(position, layer), splitContinuation.ReadV(position, layer), $"V at position {position} layer {layer}");
            }
        }
    }

    [Theory]
    [MemberData(nameof(PrefillKernels))]
    public void SplitRoute_MergedBatchRowsMatchWholeModelForward_BitForBit(PrefillKernel kernel)
    {
        string path = SyntheticLlamaGguf.Write("split-identity-batch");
        _tempFiles.Add(path);

        ModelRoute route = TwoCpuSegmentSplitRoute("split-identity-batch");

        using GgufFile gguf = GgufFile.Open(path);

        CountingBackend referenceBackend = new(new CpuBackend(threads: 2));
        using TransformerModel whole = TransformerModel.Open(gguf, referenceBackend, maxContextTokens: 64, logLine: null, kernel);
        using BlockPool referencePool = NewPool(whole, referenceBackend, capacityBlocks: 8);

        CountingBackend firstSpan = new(new CpuBackend(threads: 2));
        CountingBackend receiving = new(new CpuBackend(threads: 2));
        using TransformerModel split = TransformerModel.Open(gguf, firstSpan, maxContextTokens: 64, logLine: null, kernel, route, [receiving]);
        using BlockPool splitPool = NewPool(split, firstSpan, capacityBlocks: 8);

        // Two sequences advance in one merged step on each engine — the rows cross spans together exactly as a scheduled batch would.
        SequenceKv wholeA = new(referencePool);
        SequenceKv wholeB = new(referencePool);
        SequenceKv splitA = new(splitPool);
        SequenceKv splitB = new(splitPool);

        float[] wholePrefillA = [.. whole.Prefill([1, 2], wholeA)];
        float[] wholePrefillB = [.. whole.Prefill([3, 4], wholeB)];
        float[] splitPrefillA = [.. split.Prefill([1, 2], splitA)];
        float[] splitPrefillB = [.. split.Prefill([3, 4], splitB)];

        AssertBitsEqual(wholePrefillA, splitPrefillA, "sequence A's prefill across the two-span route");
        AssertBitsEqual(wholePrefillB, splitPrefillB, "sequence B's prefill across the two-span route");

        float[] wholeStep = [.. whole.DecodeBatch([ArgMax(wholePrefillA), ArgMax(wholePrefillB)], [wholeA, wholeB])];
        float[] splitStep = [.. split.DecodeBatch([ArgMax(splitPrefillA), ArgMax(splitPrefillB)], [splitA, splitB])];

        AssertBitsEqual(wholeStep, splitStep, "a merged two-sequence step across the two-span route");

        for (int position = 0; position < wholeA.Length; position++)
        {
            for (int layer = 0; layer < whole.Desc.LayerCount; layer++)
            {
                AssertBitsEqual(wholeA.ReadK(position, layer), splitA.ReadK(position, layer), $"sequence A K at position {position} layer {layer}");
                AssertBitsEqual(wholeB.ReadV(position, layer), splitB.ReadV(position, layer), $"sequence B V at position {position} layer {layer}");
            }
        }

        wholeA.Dispose();
        wholeB.Dispose();
        splitA.Dispose();
        splitB.Dispose();
    }

    // ---------------------------------------------------------------- refusals naming what this build does not carry

    [Fact]
    public void SplitRoute_UnsuppliedCrossingPosition_RefusesNamingWhatIsMissing()
    {
        string path = SyntheticLlamaGguf.Write("split-unsupplied");
        _tempFiles.Add(path);

        ModelRoute route = TwoCpuSegmentSplitRoute("split-unsupplied");

        using GgufFile gguf = GgufFile.Open(path);
        CountingBackend backend = new(new CpuBackend(threads: 2));

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, backend, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, route)); // the public seam supplies no crossing backends at all

        Assert.Contains("resolves a route of 2 segment(s)", error.Message);
        Assert.Contains("needs one execution backend for every position after the first", error.Message);
    }

    [Fact]
    public void SplitRoute_DedicatedLanePosture_RefusesNamingTheStageSideCrossingItCannotRun()
    {
        string path = SyntheticLlamaGguf.Write("split-lane");
        _tempFiles.Add(path);

        ModelRoute route = TwoCpuSegmentSplitRoute("split-lane");
        PrefillLaneLayout layout = ModelPrefillLanes.ResolveDedicated("split-lane", engineThreadsChoice: 2, threadQuota: null, coreSet: new HashSet<int>());

        using GgufFile gguf = GgufFile.Open(path);
        CountingBackend decodeSide = new(new CpuBackend(threads: 1));
        CountingBackend stageSide = new(new CpuBackend(threads: 1));

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, decodeSide, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, route, stageSide, layout));

        Assert.Contains("opens a dedicated prefill lane over a route of 2 segments", error.Message);
        Assert.Contains("prompt windows have no crossing to run on", error.Message);
    }

    [Fact]
    public void SplitRoute_FirstSpanMustStartAtLayerZero_RefusesNamingTheBoundary()
    {
        string path = SyntheticLlamaGguf.Write("split-first-span");
        _tempFiles.Add(path);

        ModelRoute route = ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "split-first-span",
            Stages =
            [
                new RouteStage(ModelRoutes.PrefillStage, ModelRoute.CpuDeviceId) { SplitPosition = 1 },
                new RouteStage(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId) { SplitPosition = 2 },
            ],
            AvailableDevices = [ModelRoute.CpuDeviceId],
        });

        using GgufFile gguf = GgufFile.Open(path);
        CountingBackend firstSpan = new(new CpuBackend(threads: 2));
        CountingBackend receiving = new(new CpuBackend(threads: 2));

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, firstSpan, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, route, [receiving]));

        Assert.Contains("starts the split at boundary '1'", error.Message);
        Assert.Contains("a route's first span covers this model from layer 0", error.Message);
    }

    [Fact]
    public void SplitRoute_BoundaryPastTheLastLayer_RefusesNamingWhatRunsNothing()
    {
        string path = SyntheticLlamaGguf.Write("split-past-layers");
        _tempFiles.Add(path);

        ModelRoute route = ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = "split-past-layers",
            Stages =
            [
                new RouteStage(ModelRoutes.PrefillStage, ModelRoute.CpuDeviceId) { SplitPosition = 0 },
                new RouteStage(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId) { SplitPosition = 5 }, // the fixture ships two layers: a span starting past them runs nothing
            ],
            AvailableDevices = [ModelRoute.CpuDeviceId],
        });

        using GgufFile gguf = GgufFile.Open(path);
        CountingBackend firstSpan = new(new CpuBackend(threads: 2));
        CountingBackend receiving = new(new CpuBackend(threads: 2));

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, firstSpan, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, route, [receiving]));

        Assert.Contains("starts at boundary '5', past this file's last layer", error.Message);
    }

    [Fact]
    public void SplitRoute_EncoderForward_RefusesNamingWhatItDoesNotCarry()
    {
        string path = SyntheticLlamaGguf.Write("split-encode");
        _tempFiles.Add(path);

        ModelRoute route = TwoCpuSegmentSplitRoute("split-encode");

        using GgufFile gguf = GgufFile.Open(path);
        CountingBackend firstSpan = new(new CpuBackend(threads: 2));
        CountingBackend receiving = new(new CpuBackend(threads: 2));

        using TransformerModel split = TransformerModel.Open(gguf, firstSpan, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, route, [receiving]);
        using BlockPool pool = NewPool(split, firstSpan, capacityBlocks: 4);
        using SequenceKv kv = new(pool);

        ModelException error = Assert.Throws<ModelException>(() => split.Encode([1, 2], kv));

        Assert.Contains("encoder forward rents streams past the row capacity", error.Message);
    }

    // ---------------------------------------------------------------- helpers

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

    private static BlockPool NewPool(TransformerModel model, IBackend backend, int capacityBlocks) =>
        new(blockTokens: 16, layerCount: model.Desc.LayerCount, kvHeads: model.Desc.KvHeadCount, headDim: model.Desc.HeadDim, capacityBlocks, backend);

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

    /// <summary>A compute backend that counts the aligned arenas it was asked to hold: what a span stages in is visible as calls against its own allocator.</summary>
    private sealed class CountingBackend : IBackend
    {
        private readonly CpuBackend _inner;

        public CountingBackend(CpuBackend inner)
        {
            ArgumentNullException.ThrowIfNull(inner);
            _inner = inner;
        }

        /// <summary>Arenas this backend allocated.</summary>
        public int AllocateCalls { get; private set; }

        /// <summary>Arenas released back to this backend's allocator.</summary>
        public int FreeCalls { get; private set; }

        public string DeviceId => _inner.DeviceId;

        public MemoryHandle AllocateAligned(long bytes, int alignment)
        {
            AllocateCalls++;
            return _inner.AllocateAligned(bytes, alignment);
        }

        public void FreeAligned(MemoryHandle handle)
        {
            FreeCalls++;
            _inner.FreeAligned(handle);
        }

        public void Gemv(ReadOnlySpan<float> x, WeightRef w, Span<float> output) => _inner.Gemv(x, w, output);

        public void GemvBatch(ReadOnlySpan<float> inputs, int count, WeightRef w, Span<float> outputs) => _inner.GemvBatch(inputs, count, w, outputs);

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
