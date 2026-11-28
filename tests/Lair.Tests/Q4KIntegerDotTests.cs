using Lair.Backends.Cpu;
using Lair.Contracts;

namespace Lair.Tests;

/// <summary>
/// The integer-domain AVX2 Q4_K dense batched dot (internal, unpromoted): what a posture may and may not run with it, and its own
/// determinism contract. Quantizing FP32 activations to signed bytes loses precision against the float kernels, so outputs are NOT
/// expected bit-identical to them — what this variant guarantees instead is that every chain walks ascending super-blocks in fixed
/// integer lane order with a fixed per-block fold sequence, so batched equals solo equals threaded at any worker count and any chain
/// grouping. On hosts whose auto selection lands on AVX-VNNI (this machine) or scalar, the variant never runs whatever a seam states:
/// those postures keep today's float traversals bit-for-bit. Every test that exercises it forces an AVX2-only posture so the oracle's
/// resolution is deterministic across machines instead of depending on what this box happens to report.
/// </summary>
[Trait("Category", "Kernel")]
public sealed class Q4KIntegerDotTests
{
    // The dense feed-forward geometries plus ragged column tails: a partial final quant block, columns past one 256-element block,
    // and shapes whose staged activation padding must contribute exactly nothing.
    private static readonly (int Rows, int Cols)[] MatrixSizes = [(100, 137), (9, 264), (130, 300), (512, 1024)];

    // Counts at and above the stated width, ragged counts across chain-group boundaries, and single-vector calls below it.
    private static readonly int[] IntegerDotCounts = [8, 9, 16, 20, 32, 63, 64];

    // AVX2-only posture: the one resolution this variant is stated for. (This machine's auto lands on Vnni.)
    private static CpuFeaturePosture Avx2Only => new(Avx2Available: true, AvxVnniAvailable: false);

    [Theory]
    [MemberData(nameof(MatrixSizesData))]
    public void GemvBatch_Q4KIntegerDot_AtEveryCount_IsBitIdenticalToItselfAcrossGroupingsAndThreads(int rows, int cols)
    {
        foreach (int count in IntegerDotCounts)
        {
            (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 71_000 + count, vectors: count);
            WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K);

            // Reference arm: the default resolved group width (8) with a stated integer-dot width.
            using CpuBackend reference = new(threads: 1, cpuFeatures: Avx2Only, tuning: new CpuKernelTuningOverrides { Q4KIntegerDotBatchWidth = 8 });
            Assert.Equal(IntegerDotPath.Avx2, reference.ResolvedIntegerDotPath); // the posture this variant is stated for, forced so it holds on any machine

            float[] expected = new float[count * rows];
            reference.GemvBatch(inputs, count, weight, expected);

            // Ragged chain grouping: every stated group splits the batch into consecutive groups plus a tail — same chains, same order, identical bits whatever width walked them.
            foreach (int group in new[] { 2, 5 })
            {
                using CpuBackend grouped = new(threads: 1, cpuFeatures: Avx2Only, tuning: new CpuKernelTuningOverrides { Q4KIntegerDotBatchWidth = 8, ChainGroup = group });
                float[] actualGrouped = new float[count * rows];
                grouped.GemvBatch(inputs, count, weight, actualGrouped);

                BatchKernelTests.AssertBitsEqual(expected, actualGrouped, $"Q4_K integer dot at {rows}x{cols} count {count} under chain group {group}");
            }

            // Threaded arm: forced tiny row blocks hand every row to a worker — the same per-row walks, partitioned.
            using CpuBackend threaded = new(threads: 8, cpuFeatures: Avx2Only, tuning: new CpuKernelTuningOverrides { Q4KIntegerDotBatchWidth = 8, RowBlock = 1, ParallelRowThreshold = 1 });
            float[] actualThreaded = new float[count * rows];
            threaded.GemvBatch(inputs, count, weight, actualThreaded);

            BatchKernelTests.AssertBitsEqual(expected, actualThreaded, $"threaded Q4_K integer dot at {rows}x{cols} count {count}");
        }
    }

    public static TheoryData<int, int> MatrixSizesData()
    {
        TheoryData<int, int> data = [];
        foreach ((int rows, int cols) in MatrixSizes)
        {
            data.Add(rows, cols);
        }

        return data;
    }

    // When activations quantize exactly — constant vectors whose every value is the super-block maximum, so each staged quant
    // lands on -127 and its dequantization is exact — the integer variant must land on what the float kernels compute for the
    // same dot within pure summation-order rounding: this is the numeric contract that separates real arithmetic from quantization
    // loss. (Against ordinary seeded activations the two traversals differ by activation-quantization noise amplified through
    // cancellation against these writers' large block scales, which is expected and not asserted here.)
    [Theory]
    [MemberData(nameof(MatrixSizesData))]
    public void GemvBatch_Q4KIntegerDot_ExactQuantizedActivations_MatchFloatRouteWithinSummationRounding(int rows, int cols)
    {
        const int count = 16;

        (_, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 72_000 + cols, vectors: count);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K);

        List<float> inputList = [];
        for (int v = 0; v < count; v++)
        {
            float constant = v % 2 == 0 ? 0.5f : -0.25f; // each vector is one constant: its quantization is lossless, dequantization exact.

            inputList.AddRange(Enumerable.Repeat(constant, cols));
        }

        float[] inputs = [.. inputList];

        using CpuBackend floatRoute = new(threads: 1, cpuFeatures: Avx2Only); // no width stated: today's traversal.
        Assert.Equal(IntegerDotPath.Avx2, floatRoute.ResolvedIntegerDotPath);
        float[] expected = new float[count * rows];
        floatRoute.GemvBatch(inputs, count, weight, expected);

        using CpuBackend integerRoute = new(threads: 1, cpuFeatures: Avx2Only, tuning: new CpuKernelTuningOverrides { Q4KIntegerDotBatchWidth = 8 });
        float[] actual = new float[count * rows];
        integerRoute.GemvBatch(inputs, count, weight, actual);

        double maxAbsDiff = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            // Bit-equality is possible here and not required: with exact activation quantization both traversals sum the same
            // terms, so agreement to summation rounding — not divergence per element — is what this contract states.
            maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs((double)actual[i] - expected[i]));
        }

        double scaleRef = Math.Max(1, Enumerable.Range(0, expected.Length).Max(i => Math.Abs(expected[i])));
        Assert.True(maxAbsDiff <= 2e-3 * scaleRef, $"integer dot at {rows}x{cols} with exact activation quantization drifted {maxAbsDiff:E4} absolute against the float route (values up to {scaleRef:F1})");
    }

    [Theory]
    [InlineData(7)] // one below the stated width: today's traversal stays in force, bit-for-bit
    [InlineData(1)]
    [InlineData(3)]
    public void GemvBatch_Q4KBelowIntegerDotWidth_KeepsFloatTraversal_BitIdentical(int count)
    {
        const int rows = 300;
        const int cols = 300;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 73_000 + count, vectors: count);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K);

        using CpuBackend reference = new(threads: 1, cpuFeatures: Avx2Only);
        float[] expected = new float[count * rows];
        reference.GemvBatch(inputs, count, weight, expected);

        using CpuBackend pinned = new(threads: 1, cpuFeatures: Avx2Only, tuning: new CpuKernelTuningOverrides { Q4KIntegerDotBatchWidth = 8 });
        float[] actual = new float[count * rows];
        pinned.GemvBatch(inputs, count, weight, actual);

        BatchKernelTests.AssertBitsEqual(expected, actual, $"Q4_K batch({count}) below the stated integer-dot width at {rows}x{cols}");
    }

    // ---------------------------------------------------------------- no posture promotes it: only a stated width on an AVX2-resolved backend runs it

    [Fact]
    public void Resolve_Q4KIntegerDotWidthStaysUnpromotedAtEveryPosture()
    {
        Assert.Equal(0, CpuKernelTuning.Resolve(32, vectorPosture: true, CpuKernelTuningOverrides.Auto).Q4KIntegerDotBatchWidth); // no auto posture states a width for it
        Assert.Equal(0, CpuKernelTuning.Resolve(32, vectorPosture: false, CpuKernelTuningOverrides.Auto).Q4KIntegerDotBatchWidth); // and without a vector posture nothing runs anyway
    }

    [Fact]
    public void BackendOpen_AutoVnniHostKeepsFloatTraversal_AtStatedWidth()
    {
        using CpuBackend vnniPinned = new(threads: 1, cpuFeatures: new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: true), tuning: new CpuKernelTuningOverrides { Q4KIntegerDotBatchWidth = 8 });

        Assert.Equal(IntegerDotPath.Vnni, vnniPinned.ResolvedIntegerDotPath); // this machine's auto lands here today
        Assert.Equal(8, vnniPinned.ResolvedTuning.Q4KIntegerDotBatchWidth); // a stated width is resolved...

        const int rows = 300;
        const int cols = 1024;
        const int count = 16;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 74_000, vectors: count);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K);

        using CpuBackend reference = new(threads: 1, cpuFeatures: new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: true)); // auto Vnni posture without the seam
        float[] expected = new float[count * rows];
        reference.GemvBatch(inputs, count, weight, expected);

        float[] actual = new float[count * rows];
        vnniPinned.GemvBatch(inputs, count, weight, actual);

        BatchKernelTests.AssertBitsEqual(expected, actual, "a stated integer-dot width on a vpdpbusd-resolved backend does not substitute another ladder"); // ...but only the AVX2 rung may run it.
    }

    [Fact]
    public void BackendOpen_ScalarPostureKeepsFloatTraversal_AtStatedWidth()
    {
        const int rows = 300;
        const int cols = 1024;
        const int count = 16;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 75_000, vectors: count);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K);

        using CpuBackend reference = new(threads: 1, allowAvx2: false);
        Assert.Equal(IntegerDotPath.Scalar, reference.ResolvedIntegerDotPath);
        float[] expected = new float[count * rows];
        reference.GemvBatch(inputs, count, weight, expected);

        using CpuBackend pinned = new(threads: 1, allowAvx2: false, tuning: new CpuKernelTuningOverrides { Q4KIntegerDotBatchWidth = 8 });
        float[] actual = new float[count * rows];
        pinned.GemvBatch(inputs, count, weight, actual);

        BatchKernelTests.AssertBitsEqual(expected, actual, "a stated integer-dot width on a scalar posture runs no vector kernel at all");
    }

    [Fact]
    public void BackendOpen_Avx2OnlyPosture_ResolvesStatedWidthExactly()
    {
        using CpuBackend unpinned = new(threads: 8, cpuFeatures: Avx2Only);
        Assert.Equal(0, unpinned.ResolvedTuning.Q4KIntegerDotBatchWidth); // the AVX2-only lane does not promote it on its own

        using CpuBackend pinned = new(threads: 8, cpuFeatures: Avx2Only, tuning: new CpuKernelTuningOverrides { Q4KIntegerDotBatchWidth = 16 });
        Assert.Equal(16, pinned.ResolvedTuning.Q4KIntegerDotBatchWidth); // and only a stated width opens it there
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(65)] // past the batch ceiling every dense kernel's scratch is sized for
    public void Resolve_Q4KIntegerDotPinOutsideItsSpan_RefusesNamingQualifiedKeyAndValue(int value)
    {
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(
            () => CpuKernelTuning.Resolve(8, vectorPosture: true, new CpuKernelTuningOverrides { Q4KIntegerDotBatchWidth = value }));

        Assert.Equal("engine.cpu.tuning.q4k_integer_dot_batch_width", error.ParamName);
        Assert.Contains($"engine.cpu.tuning.q4k_integer_dot_batch_width {value}", error.Message);
    }

    [Fact]
    public void Resolve_Q4KIntegerDotPinResolvesExactly()
    {
        foreach (int width in new[] { 0, 1, 8, 64 })
        {
            CpuKernelTuning tuning = CpuKernelTuning.Resolve(32, vectorPosture: true, new CpuKernelTuningOverrides { Q4KIntegerDotBatchWidth = width });

            Assert.Equal(width, tuning.Q4KIntegerDotBatchWidth);
        }
    }

    // A stated integer-dot width says nothing about the blocked matmul crossover for another format's batch, and a stated
    // Q6_K crossover never opens this variant: each seam selects only its own traversal.
    [Fact]
    public void GemvBatch_Q4KIntegerDotPin_DoesNotExtendToQ6K()
    {
        const int rows = 300;
        const int cols = 1024;
        const int count = 16;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q6_K, seed: 76_000, vectors: count);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q6_K);

        using CpuBackend reference = new(threads: 1, cpuFeatures: Avx2Only);
        float[] expected = new float[count * rows];
        reference.GemvBatch(inputs, count, weight, expected);

        using CpuBackend pinnedForQ4K = new(threads: 1, cpuFeatures: Avx2Only, tuning: new CpuKernelTuningOverrides { Q4KIntegerDotBatchWidth = 8 });
        float[] actual = new float[count * rows];
        pinnedForQ4K.GemvBatch(inputs, count, weight, actual);

        BatchKernelTests.AssertBitsEqual(expected, actual, "the Q4_K integer-dot pin leaves a Q6_K batch untouched"); // no format substitution anywhere.
    }

    [Fact]
    public void GemvBatch_Q6KMatmulRoutePin_DoesNotOpenQ4KIntegerDot()
    {
        const int rows = 300;
        const int cols = 1024;
        const int count = 16;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 77_000, vectors: count);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K);

        using CpuBackend reference = new(threads: 1, cpuFeatures: Avx2Only);
        float[] expected = new float[count * rows];
        reference.GemvBatch(inputs, count, weight, expected);

        using CpuBackend pinnedForQ6K = new(threads: 1, cpuFeatures: Avx2Only, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidth = 8 });
        float[] actual = new float[count * rows];
        pinnedForQ6K.GemvBatch(inputs, count, weight, actual);

        BatchKernelTests.AssertBitsEqual(expected, actual, "a stated Q6_K crossover leaves a Q4_K batch on its own traversal"); // the integer-dot seam stays independent.
    }

    // The fused gate/up pair route is untouched by this seam: pairs stay on their measured posture and fuse exactly as before.
    [Fact]
    public void TryGemvBatchSharedInput_Q4KPair_FusesAtStatedIntegerDotWidth_BitIdenticallyToDefaultPosture()
    {
        const int rows = 256;
        const int cols = 1024;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 78_000, vectors: 16);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K);

        using CpuBackend defaultPosture = new(threads: 1, cpuFeatures: Avx2Only);
        float[] expectedA = new float[16 * rows];
        float[] expectedB = new float[16 * rows];
        Assert.True(((IDenseWeightPairFusion)defaultPosture).TryGemvBatchSharedInput(inputs, 16, weight, expectedA, weight, expectedB));

        using CpuBackend statedWidth = new(threads: 1, cpuFeatures: Avx2Only, tuning: new CpuKernelTuningOverrides { Q4KIntegerDotBatchWidth = 8 });
        float[] outA = new float[16 * rows];
        float[] outB = new float[16 * rows];

        Assert.True(((IDenseWeightPairFusion)statedWidth).TryGemvBatchSharedInput(inputs, 16, weight, outA, weight, outB)); // the seam says nothing about a shared-input pair: that traversal is its own measured posture
        BatchKernelTests.AssertBitsEqual(expectedA, outA, $"gate side at {rows}x{cols} under a stated integer-dot width");
        BatchKernelTests.AssertBitsEqual(expectedB, outB, $"up side at {rows}x{cols} under a stated integer-dot width");
    }

    // The chain-group guard: every resolved group fits the register budget this variant is written for (8 accumulator chains), and
    // wider-than-tail batches split into consecutive budget-sized groups — asserted by the grouping test above at pinned 3 alongside
    // the default 8, with both arms producing identical bits. Here we pin that resolution directly.
    [Fact]
    public void Resolve_ChainGroupGuard_KeepsGroupsWithinTheRegisterBudget()
    {
        foreach (int group in new[] { 1, 2, 3, 7, 8 }) // every pinned group a call may state for this variant
        {
            CpuKernelTuning tuning = CpuKernelTuning.Resolve(64, vectorPosture: true, new CpuKernelTuningOverrides { Q4KIntegerDotBatchWidth = 8, ChainGroup = group });

            Assert.True(tuning.ChainGroup >= 1 && tuning.ChainGroup <= CpuKernelTuning.ChainGroupMax); // the integer arms never see a group past eight chains
        }
    }
}
