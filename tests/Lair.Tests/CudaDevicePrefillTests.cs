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
/// Composition gate for the device prefill posture: every prompt window this engine forwards under an opened device step
/// must answer what its host route answers, within this seam's recorded CUDA band — the composed walk runs the same causal
/// selection over the same KV bytes while scores, softmax state, and K/V reads stay on the far side of the seam, so drift
/// lands where the committed attention kernels' numerics land (ex2.approx against MathF.Exp), not in composition. The
/// pinned dense fixture carries a grouped-query shape the committed shift mapping cannot divide directly — nine query
/// heads over three kv-heads — so its composed windows run through padded head staging, and matching the host reference on
/// that file is what proves the padding maps heads rather than merely walking them. Device to device the claim stays
/// bit-for-bit: one window walked twice answers identical bytes. Posture refusals (geometry facts beside this plan's own
/// values, a step narrower than the resolved row width, any selection but the dense-matrix one, an offload side that moves
/// windows onto another backend) are checked through stand-in steps with no driver opened at all. All lanes share one
/// serialized lane with the other model-opening suites.
/// </summary>
[Trait("Category", "Parity")]
[Trait("Category", "Prefill")]
[Trait("Category", "Cuda")]
[Collection("model-loaded")] // shares one serialized lane with the other model-opening suites: two engines mapping weights at once is not what this box should be doing
public sealed class CudaDevicePrefillTests : IDisposable
{
    // Recorded ceiling for composed logits against the host reference. Measured worst max-abs-logit drift across these lanes on this box — the pinned fixture's composed windows at lengths 32/64/128 and both synthetic shapes: 2.019E-003 (at a logit magnitude near +5). The gate sits about four times above that measured worst; no existing CPU or CUDA GEMV/blocked-matmul/attention band moves here, and nothing in this suite asserts bit equality across paths — only device-to-device determinism does.
    private const float LogitDriftCeiling = 4e-3f; // Recorded ceiling for composed logits against the host reference — a new composition-only band (DN-M5.4-3), added rather than widened: no existing CPU or CUDA GEMV/blocked-matmul/attention band moves here, and nothing in this suite asserts bit equality across paths (only device-to-device determinism does). Measured worst max-abs-logit drift per lane on this box: pinned SmolLM-F16 composed window at 128 tokens `1.945E-003` at index 22381 (reference -2.448534, composed -2.446590); synthetic direct grouped shape `4.244E-005`; synthetic padded non-power-of-two grouped shape `2.480E-005`. The gate sits about twice above the measured worst — the same rounding posture MVP5.3 recorded for its F16 matmul band, and composition adds no numerics of its own beyond those committed kernels' answers.

    // The dense shape every synthetic lane resolves: two layers at a head dimension the step kernels walk (16), grouped-query 2:1 for the direct shift mapping, and one deliberately non-power-of-two sibling.
    private static readonly SyntheticLlamaGguf.Geometry DirectShape = new(
        Layers: 2,
        HiddenSize: 64,
        HeadCount: 4,
        KvHeadCount: 2,
        FeedForwardLength: 128);

    // Nine query heads over three kv-heads at head dimension 8: the same grouped-query ratio the pinned fixture ships in miniature, so padded staging composition is gated on a file these lanes always have.
    private static readonly SyntheticLlamaGguf.Geometry PaddedShape = new(
        Layers: 2,
        HiddenSize: 72,
        HeadCount: 9,
        KvHeadCount: 3,
        FeedForwardLength: 144);

    private const int BlockTokens = 16;

    // The window every synthetic lane prefills: ids stay inside these fixtures' own tiny vocabulary (the synthetic writer ships eight tokens), matching what every other suite on this box feeds them. Decode continuation advances four greedy steps past it.
    private static readonly int[] SyntheticPrompt = [1, 2, 3];

    private const int DecodeSteps = 4;

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
    }

    // ---------------------------------------------------------------- composed logits against the host reference

    [Fact]
    public void DevicePrefill_DirectGroupedShape_LogitsInsideRecordedBand_ContinuationMatchesCpuBaseline()
    {
        using CudaBackend cuda = OpenCuda();

        string path = SyntheticLlamaGguf.Write("prefill-device-direct", withReasoningMarkers: true, QuantFormat.F16, DirectShape);
        _tempFiles.Add(path);

        int maxContextTokens = 32;
        List<int> baselineTokens;
        float[] referenceLogits;

        using (GgufFile gguf = GgufFile.Open(path))
        {
            using CpuBackend cpu = new(threads: 2);
            using TransformerModel baseline = TransformerModel.Open(gguf, cpu, maxContextTokens, logLine: null, PrefillKernel.GemV);

            Assert.False(baseline.DevicePrefillActive); // every default posture composes nothing device-side

            using BlockPool pool = NewHostPool(baseline.Desc.LayerCount, baseline.Desc.KvHeadCount, baseline.Desc.HeadDim, capacityBlocks: 4, cpu);
            using SequenceKv kv = new(pool);

            referenceLogits = [.. baseline.Prefill(SyntheticPrompt, kv)];
            baselineTokens = GreedyDecode(baseline, kv);
        }

        float[] deviceLogits;
        List<int> deviceTokens;

        using (GgufFile gguf = GgufFile.Open(path))
        {
            using TransformerModel composed = OpenComposed(gguf, cuda, maxContextTokens, DirectShape);
            Assert.True(composed.DevicePrefillActive);
            Assert.Equal("private arena", composed.DevicePrefillKvResidency);

            using BlockPool pool = NewHostPool(composed.Desc.LayerCount, composed.Desc.KvHeadCount, composed.Desc.HeadDim, capacityBlocks: 4, cuda);
            using SequenceKv kv = new(pool);

            deviceLogits = [.. composed.Prefill(SyntheticPrompt, kv)];
            deviceTokens = GreedyDecode(composed, kv); // the decode side stays this engine's attention over its own KV bytes — composition lands on prompt windows only
        }

        AssertWithinRecordedBand(referenceLogits, deviceLogits, "composed window against the batched-row reference");
        Assert.Equal(baselineTokens, deviceTokens); // token-identical continuation across postures — a composed prefill leaves nothing downstream reading different bytes
    }

    [Fact]
    public void DevicePrefill_NonPow2GroupedShape_PaddedStagingMatchesHostReference_WithinRecordedBand()
    {
        using CudaBackend cuda = OpenCuda();

        string path = SyntheticLlamaGguf.Write("prefill-device-padded", withReasoningMarkers: true, QuantFormat.F16, PaddedShape);
        _tempFiles.Add(path);

        int maxContextTokens = 32; // nine heads over three kv-heads — a group the committed shift mapping cannot divide, so this open composes through padded staging end to end
        float[] referenceLogits;
        int baselineArgmax;

        using (GgufFile gguf = GgufFile.Open(path))
        {
            using CpuBackend cpu = new(threads: 2);
            using TransformerModel baseline = TransformerModel.Open(gguf, cpu, maxContextTokens, logLine: null, PrefillKernel.GemV);

            Assert.Equal(9, baseline.Desc.HeadCount);
            Assert.Equal(3, baseline.Desc.KvHeadCount);

            using BlockPool pool = NewHostPool(baseline.Desc.LayerCount, baseline.Desc.KvHeadCount, baseline.Desc.HeadDim, capacityBlocks: 4, cpu);
            using SequenceKv kv = new(pool);

            referenceLogits = [.. baseline.Prefill(SyntheticPrompt, kv)];
            baselineArgmax = ArgMax(referenceLogits);
        }

        float[] deviceLogits;

        using (GgufFile gguf = GgufFile.Open(path))
        {
            using TransformerModel composed = OpenComposed(gguf, cuda, maxContextTokens, PaddedShape);
            Assert.Equal(9, composed.Desc.HeadCount); // the composition answers this plan's own head count, not whatever staging its walk pads to

            using BlockPool pool = NewHostPool(composed.Desc.LayerCount, composed.Desc.KvHeadCount, composed.Desc.HeadDim, capacityBlocks: 4, cuda);
            using SequenceKv kv = new(pool);

            deviceLogits = [.. composed.Prefill(SyntheticPrompt, kv)];
        }

        AssertWithinRecordedBand(referenceLogits, deviceLogits, "padded grouped-query composition against the batched-row reference");
        Assert.Equal(baselineArgmax, ArgMax(deviceLogits)); // every real head's answer lands in its own region — a mapping that shuffled heads would move this value
    }

    [Fact]
    public void DevicePrefill_PinnedDenseFixture_ComposesWithinRecordedBand_AndReportsStoreBackedResidency()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        using CudaBackend cuda = OpenCuda();

        int maxLength = 128;
        int capacityBlocks = (maxLength + BlockTokens - 1) / BlockTokens + 4;
        int storePositions = checked(capacityBlocks * BlockTokens);

        float[] referenceLogits;

        using (GgufFile gguf = GgufFile.Open(modelPath))
        {
            ModelPlan geometry = ModelPlans.Resolve(gguf);

            Assert.Equal(9, geometry.HeadCount); // the pinned fixture's own grouped-query ratio: nine heads over three kv-heads
            Assert.Equal(3, geometry.KvHeadCount);

            using CpuBackend cpu = new();
            using TransformerModel baseline = TransformerModel.Open(gguf, cpu, maxLength, logLine: null, PrefillKernel.GemV);

            using BlockPool pool = NewHostPool(geometry.LayerCount, geometry.KvHeadCount, geometry.HeadDim, capacityBlocks, cpu);
            using SequenceKv kv = new(pool);

            referenceLogits = [.. baseline.Prefill(BuildWindow(maxLength), kv)];
        }

        float[] deviceLogits;

        using (GgufFile gguf = GgufFile.Open(modelPath))
        {
            ModelPlan geometry = ModelPlans.Resolve(gguf);

            // The store-bound posture: this pool's whole declared range carries a persistent FP32 copy through the CUDA mirror in the device-step execution, and the composed step writes its KV into exactly those blocks. Host KV bytes stay authoritative for anything decoded afterward.
            using BlockPool pool = new(
                BlockTokens, geometry.LayerCount, geometry.KvHeadCount, geometry.HeadDim, capacityBlocks, cuda,
                BlockResidencyMode.Vram, deviceBackedBlocks: capacityBlocks, mirrorOwner: null, KvFormatPair.Default, KvMirrorExecution.DeviceStep);

            using IDeviceAttentionStep step = cuda.OpenAttentionStep(
                new CudaDeviceStepRequest
                {
                    MaxPositions = storePositions, // a bound store addresses exactly the range it declares — positions come from block count x block tokens on both sides
                    LayerCount = geometry.LayerCount,
                    HeadCount = geometry.HeadCount,
                    KvHeadCount = geometry.KvHeadCount,
                    HeadDim = geometry.HeadDim,
                    MaxWindowRows = Math.Max(CudaDeviceStepRequest.DefaultMaxWindowRows, geometry.PrefillRowWidth),
                },
                pool.DeviceStepMirror!);

            using TransformerModel composed = OpenComposedWith(gguf, cuda, maxLength, WholeModelCudaRoute("device-prefill-fixture"), step);
            Assert.Equal("pool device store", composed.DevicePrefillKvResidency);

            using SequenceKv kv = new(pool);
            deviceLogits = [.. composed.Prefill(BuildWindow(maxLength), kv)];

            Assert.True(pool.DeviceWrittenBlocks > 0, "a composed window wrote no KV on the device: the step never touched its bound store");
        }

        AssertWithinRecordedBand(referenceLogits, deviceLogits, "composed dense prefill against the batched-row reference on the pinned fixture");
        Assert.Equal(ArgMax(referenceLogits), ArgMax(deviceLogits));
    }

    // ---------------------------------------------------------------- determinism across composed runs

    [Fact]
    public void DevicePrefill_SameWindowTwice_LogitsAreBitIdentical()
    {
        using CudaBackend cuda = OpenCuda();

        string path = SyntheticLlamaGguf.Write("prefill-device-deterministic", withReasoningMarkers: true, QuantFormat.F16, DirectShape);
        _tempFiles.Add(path);

        float[] first;
        float[] second;

        using GgufFile gguf = GgufFile.Open(path);

        using (TransformerModel composed = OpenComposed(gguf, cuda, maxContextTokens: 32, DirectShape))
        {
            using BlockPool pool = NewHostPool(composed.Desc.LayerCount, composed.Desc.KvHeadCount, composed.Desc.HeadDim, capacityBlocks: 4, cuda);

            using SequenceKv kvA = new(pool);
            first = [.. composed.Prefill(SyntheticPrompt, kvA)];
        }

        // The same window walked again on a fresh sequence and a fresh engine over the same opened step family — the device walk answers bytes, not a reroll.
        using (TransformerModel composed = OpenComposed(gguf, cuda, maxContextTokens: 32, DirectShape))
        {
            using BlockPool pool = NewHostPool(composed.Desc.LayerCount, composed.Desc.KvHeadCount, composed.Desc.HeadDim, capacityBlocks: 4, cuda);

            using SequenceKv kvB = new(pool);
            second = [.. composed.Prefill(SyntheticPrompt, kvB)];
        }

        AssertBitsEqual(first, second, "the same window through two opened compositions");
    }

    // ---------------------------------------------------------------- posture refusals, no driver consulted

    [Fact]
    public void DevicePrefill_StepGeometryDisagreesWithPlan_RefusesNamingBothSides()
    {
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices("prefill-refusal-geometry", QuantFormat.Q8_0, DirectShape);
        _tempFiles.Add(path);

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend cpu = new(threads: 2);

        // A step opened for four kv-heads over a plan that resolves two — every geometry fact is checked against the plan, and the refusal names both values.
        StandInAttentionStep wrongGeometry = new(DirectShape.Layers, DirectShape.HeadCount, kvHeads: 4, DirectShape.HiddenSize / DirectShape.HeadCount);

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, cpu, maxContextTokens: 32, logLine: null, PrefillKernel.Gemm, SingleCpuRoute("prefill-refusal-geometry"), wrongGeometry));

        Assert.Contains("kv-head count", error.Message, StringComparison.Ordinal);
        Assert.Contains("resolves kv-head count 2", error.Message, StringComparison.Ordinal);
        Assert.Contains("declares 4", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DevicePrefill_StepStagesNarrowerThanResolvedRowWidth_RefusesNamingBothSides()
    {
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices("prefill-refusal-width", QuantFormat.Q8_0, DirectShape);
        _tempFiles.Add(path);

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend cpu = new(threads: 2);

        StandInAttentionStep narrowWindow = new(DirectShape.Layers, DirectShape.HeadCount, DirectShape.KvHeadCount, DirectShape.HiddenSize / DirectShape.HeadCount)
        { MaxRows = 1 };

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, cpu, maxContextTokens: 32, logLine: null, PrefillKernel.Gemm, SingleCpuRoute("prefill-refusal-width"), narrowWindow));

        Assert.Contains(TransformerModel.DefaultPrefillRowWidth.ToString(), error.Message, StringComparison.Ordinal);
        Assert.Contains("stages at most 1 rows", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DevicePrefill_AnySelectionButDenseMatrix_RefusesAtOpenNamingTheComposition()
    {
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices("prefill-refusal-kernel", QuantFormat.Q8_0, DirectShape);
        _tempFiles.Add(path);

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend cpu = new(threads: 2);

        StandInAttentionStep step = StepFor(DirectShape);

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, cpu, maxContextTokens: 32, logLine: null, PrefillKernel.GemV, SingleCpuRoute("prefill-refusal-kernel"), step));

        Assert.Contains("device attention step composes prompt windows under prefill kernel 'gemm'", error.Message, StringComparison.Ordinal);
        Assert.Contains("'gemv'", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DevicePrefill_HotRepointOffDenseMatrix_RefusesAndLeavesRunningSelectionInForce()
    {
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices("prefill-repoint", QuantFormat.Q8_0, DirectShape); // a format this CPU backend multiplies in bulk, so the dense-matrix selection opens cleanly over it
        _tempFiles.Add(path);

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend cpu = new(threads: 2);

        StandInAttentionStep step = StepFor(DirectShape);

        using TransformerModel composed = TransformerModel.Open(gguf, cpu, maxContextTokens: 32, logLine: null, PrefillKernel.Gemm, SingleCpuRoute("prefill-repoint"), step);
        Assert.Equal(PrefillKernel.Gemm, composed.ResolvedPrefillKernel);

        ModelException error = Assert.Throws<ModelException>(() => composed.ApplyPrefillKernel(PrefillKernel.GemV));

        Assert.Contains("a re-point away from it is refused rather than run behind an opened composition", error.Message, StringComparison.Ordinal);
        Assert.Equal(PrefillKernel.Gemm, composed.ResolvedPrefillKernel); // the selection this engine opened on stays in force — no window ever runs some other route behind a composition
    }

    [Fact]
    public void DevicePrefill_OffloadedPromptSide_RefusesNamingThePostureItDoesNotCarry()
    {
        string path = SyntheticLlamaGguf.WriteQuantizedMatrices("prefill-refusal-offload", QuantFormat.Q8_0, DirectShape);
        _tempFiles.Add(path);

        using GgufFile gguf = GgufFile.Open(path);
        using CpuBackend engineSide = new(threads: 2);
        using CpuBackend stageSide = new(threads: 1);

        StandInAttentionStep step = StepFor(DirectShape);

        ModelException error = Assert.Throws<ModelException>(
            () => TransformerModel.Open(gguf, engineSide, maxContextTokens: 32, logLine: null, PrefillKernel.Gemm, SingleCpuRoute("prefill-refusal-offload"), step, prefillsBackend: stageSide));

        Assert.Contains("composes prompt windows inside this engine's own resolved spans", error.Message, StringComparison.Ordinal);
        Assert.Contains("an offloaded prefill backend", error.Message, StringComparison.Ordinal);
    }

    // ---------------------------------------------------------------- helpers

    private static CudaBackend OpenCuda() => new(CudaWeightResidency.DeviceCopy);

    private static StandInAttentionStep StepFor(SyntheticLlamaGguf.Geometry geometry) =>
        new(geometry.Layers, geometry.HeadCount, geometry.KvHeadCount, geometry.HiddenSize / geometry.HeadCount);

    // A composed open over a private-arena step: the whole-model accelerator route the composition composes inside. The caller owns what this opens and disposes it in its own order; model disposal never releases it.
    private static TransformerModel OpenComposed(GgufFile gguf, CudaBackend cuda, int maxContextTokens, SyntheticLlamaGguf.Geometry geometry)
    {
        IDeviceAttentionStep step = cuda.OpenAttentionStep(new CudaDeviceStepRequest
        {
            MaxPositions = Math.Max(maxContextTokens, BlockTokens),
            LayerCount = geometry.Layers,
            HeadCount = geometry.HeadCount,
            KvHeadCount = geometry.KvHeadCount,
            HeadDim = geometry.HiddenSize / geometry.HeadCount,
        });

        return OpenComposedWith(gguf, cuda, maxContextTokens, WholeModelCudaRoute($"device-prefill-{geometry.HeadCount}x{geometry.KvHeadCount}"), step);
    }

    private static TransformerModel OpenComposedWith(GgufFile gguf, CudaBackend cuda, int maxContextTokens, ModelRoute route, IDeviceAttentionStep step) =>
        TransformerModel.Open(gguf, cuda, maxContextTokens, logLine: null, PrefillKernel.Gemm, route, step);

    private static BlockPool NewHostPool(int layers, int kvHeads, int headDim, int capacityBlocks, IBackend owner) =>
        new(BlockTokens, layers, kvHeads, headDim, capacityBlocks, owner);

    private static ModelRoute WholeModelCudaRoute(string modelName) =>
        ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = modelName,
            Stages = [new RouteStage(ModelRoutes.PrefillStage, CudaBackend.DeviceIdValue), new RouteStage(ModelRoutes.DecodeStage, CudaBackend.DeviceIdValue)],
            AvailableDevices = [ModelRoute.CpuDeviceId, CudaBackend.DeviceIdValue],
        });

    private static ModelRoute SingleCpuRoute(string modelName) =>
        ModelRoutes.Resolve(new RouteRequest
        {
            ModelName = modelName,
            Stages = [new RouteStage(ModelRoutes.DecodeStage, ModelRoute.CpuDeviceId)],
            AvailableDevices = [ModelRoute.CpuDeviceId],
        });

    private static int[] BuildWindow(int tokens)
    {
        int[] ids = new int[tokens];
        for (int i = 0; i < tokens; i++)
        {
            ids[i] = 1 + i % 250; // in-vocabulary on every fixture here: token values drive embedding lookups, and both arms read the same window
        }

        return ids;
    }

    private static List<int> GreedyDecode(TransformerModel model, SequenceKv kv)
    {
        List<int> sampled = [ArgMax(model.LastLogits())];

        for (int step = 0; step < DecodeSteps; step++)
        {
            sampled.Add(ArgMax(model.DecodeNext(sampled[^1], kv)));
        }

        return sampled;
    }

    private static int ArgMax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > logits[best])
            {
                best = i;
            }
        }

        return best;
    }

    // The seam's band posture in the shape every CUDA parity claim on this engine already uses: a ceiling over max absolute drift, with the measured worst stated in any failure. Composition adds no numerics of its own — only where the committed kernels' answers land against the host chain matters here.
    private static void AssertWithinRecordedBand(ReadOnlySpan<float> reference, ReadOnlySpan<float> actual, string context)
    {
        Assert.Equal(reference.Length, actual.Length);

        double worst = 0;
        int worstAt = -1;

        for (int i = 0; i < reference.Length; i++)
        {
            double drift = Math.Abs(reference[i] - actual[i]);
            if (drift > worst)
            {
                worst = drift;
                worstAt = i;
            }
        }

        Assert.True(
            worst <= LogitDriftCeiling,
            $"{context}: max logit drift {worst:E3} at index {worstAt} (reference {reference[worstAt]:E6}, composed {actual[worstAt]:E6}) exceeds the recorded ceiling {LogitDriftCeiling:E1}");
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

    /// <summary>
    /// A stand-in for the opened composition: it answers posture-check geometry facts with whatever values a refusal lane is
    /// checking and refuses to run anything, so every refusal here lands at open against the plan — no driver, no device work.
    /// </summary>
    private sealed class StandInAttentionStep : IDeviceAttentionStep
    {
        public StandInAttentionStep(int layers, int heads, int kvHeads, int headDim)
        {
            Layers = layers;
            Heads = heads;
            KvHeads = kvHeads;
            Dim = headDim;
        }

        private int Layers { get; }

        private int Heads { get; }

        private int KvHeads { get; }

        private int Dim { get; }

        public int MaxRows { get; init; } = CudaDeviceStepRequest.DefaultMaxWindowRows;

        public bool StoreBacked { get; init; }

        public int MaxPositions => 32;

        public int LayerCount => Layers;

        public int HeadCount => Heads;

        public int KvHeadCount => KvHeads;

        public int HeadDim => Dim;

        public int MaxWindowRows => MaxRows;

        public bool KvStoreBacked => StoreBacked;

        public void RunAttentionWindow(
            int layer, int positionBase, ReadOnlySpan<float> qRows, ReadOnlySpan<float> kRows, ReadOnlySpan<float> vRows,
            float attnScale, float softcap, int window, Span<float> output) =>
            throw new InvalidOperationException("a stand-in composition never runs a window — refusal lanes check posture at open and stop there");

        public void Dispose()
        {
            // Nothing was opened device-side.
        }
    }
}
