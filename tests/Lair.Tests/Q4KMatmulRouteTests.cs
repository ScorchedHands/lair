using Lair.Backends.Cpu;
using Lair.Contracts;

namespace Lair.Tests;

/// <summary>
/// The Q4_K crossover of the hybrid dense-matrix route, and what a posture may and may not decide with it. No auto posture promotes
/// Q4_K to the blocked matmul tile executor: this build measures that executor as the cheaper traversal for its Q6_K feed-forward pass
/// while local stage-profile measurements show this file's single-matrix Q4_K passes taking that route cheaper without changing fused pairs, so stating it remains an internal seam rather than a posture decision — pending target-host validation for promotion while remaining
/// measurable and refusable rather than promoted. Whatever route a call takes, every output element lands bit-for-bit on what the other
/// route writes for it: the tile executor walks that row's groups and chunks in the same ascending order with the same FMA association
/// and scalar-tail cleanup as its batched-row kernel, and its accumulator chains stay live in registers instead of round-tripping through
/// stack scratch per multiply. The crossover is a statement about single-matrix batched passes only: the fused gate/up pair route already shares one activation walk across two accumulator chains, so it stays on its own measured posture at any stated width and every
/// scratch-eligible pair fuses exactly as it does before that seam exists.
/// </summary>
[Trait("Category", "Kernel")]
public sealed class Q4KMatmulRouteTests
{
    // The dense feed-forward geometries this route exists for (a Qwen-shaped gate/up pair 1536x2048 and down 2048x1536), plus shapes that
    // straddle the tile boundary and carry ragged column tails a tiled walk must keep byte-identical: a partial final quant block, a
    // column count past one 256-element column block, and fewer rows than one tile group.
    private static readonly (int Rows, int Cols)[] MatrixSizes = [(100, 137), (9, 264), (130, 300), (512, 1024), (1536, 2048), (2048, 1536)];

    // Counts at and above the crossover width, plus ragged counts that exercise the tile kernel's chain tails.
    private static readonly int[] RouteCounts = [8, 9, 16, 32, 64];

    [Theory]
    [MemberData(nameof(MatrixSizesData))]
    public void GemvBatch_Q4KMatmulRoute_IsBitIdenticalToTheBatchedRowRoute(int rows, int cols)
    {
        foreach (int count in RouteCounts)
        {
            (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 61_000 + count, vectors: count);
            WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K);

            using CpuBackend rowRoute = new(threads: 1, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidthQ4K = 0 });
            float[] expected = new float[count * rows];
            rowRoute.GemvBatch(inputs, count, weight, expected);

            using CpuBackend hybrid = new(threads: 1, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidthQ4K = 8 });
            float[] actual = new float[count * rows];
            hybrid.GemvBatch(inputs, count, weight, actual);

            BatchKernelTests.AssertBitsEqual(expected, actual, $"Q4_K hybrid route at {rows}x{cols} count {count}");
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

    [Theory]
    [InlineData(7)] // one below the stated width: the batched-row route stays in force
    [InlineData(1)]
    [InlineData(3)]
    public void GemvBatch_Q4KBelowRouteWidth_KeepsBatchedRowRoute_BitIdentical(int count)
    {
        const int rows = 300;
        const int cols = 300;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 62_000 + count, vectors: count);
        WeightRef weight = new(weights, new TensorShape([rows, cols]), QuantFormat.Q4_K);

        using CpuBackend reference = new(threads: 1);
        float[] expected = new float[count * rows];
        reference.GemvBatch(inputs, count, weight, expected);

        using CpuBackend pinned = new(threads: 1, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidthQ4K = 8 });
        float[] actual = new float[count * rows];
        pinned.GemvBatch(inputs, count, weight, actual);

        BatchKernelTests.AssertBitsEqual(expected, actual, $"Q4_K batch({count}) below the stated width at {rows}x{cols}");
    }

    [Fact]
    public void GemvBatch_Q4KMatmulRoute_Threaded_IsBitIdenticalToSingleThreaded()
    {
        const int Rows = 512;
        const int Cols = 1024;
        const int Count = 32;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(Rows, Cols, QuantFormat.Q4_K, seed: 63_000, vectors: Count);
        WeightRef weight = new(weights, new TensorShape([Rows, Cols]), QuantFormat.Q4_K);

        using CpuBackend single = new(threads: 1, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidthQ4K = 8 });
        float[] expected = new float[Count * Rows];
        single.GemvBatch(inputs, Count, weight, expected);

        // Forced tiny row blocks past the threshold hand every row to a worker: the same tile walks, partitioned.
        using CpuBackend threaded = new(threads: 8, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidthQ4K = 8, RowBlock = 3, ParallelRowThreshold = 1 });
        float[] actual = new float[Count * Rows];
        threaded.GemvBatch(inputs, Count, weight, actual);

        BatchKernelTests.AssertBitsEqual(expected, actual, $"threaded Q4_K hybrid route at {Rows}x{Cols}");
    }

    [Fact]
    public void GemvBatch_Q4KMatmulRoute_ScalarPosture_IsBitIdenticalToScalarBatchedRowRoute()
    {
        const int Rows = 130;
        const int Cols = 300;
        const int Count = 8;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(Rows, Cols, QuantFormat.Q4_K, seed: 64_000, vectors: Count);
        WeightRef weight = new(weights, new TensorShape([Rows, Cols]), QuantFormat.Q4_K);

        using CpuBackend reference = new(threads: 1, allowAvx2: false, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidthQ4K = 0 });
        float[] expected = new float[Count * Rows];
        reference.GemvBatch(inputs, Count, weight, expected);

        using CpuBackend hybrid = new(threads: 1, allowAvx2: false, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidthQ4K = 8 });
        float[] actual = new float[Count * Rows];
        hybrid.GemvBatch(inputs, Count, weight, actual);

        BatchKernelTests.AssertBitsEqual(expected, actual, $"scalar Q4_K hybrid route at {Rows}x{Cols}");
    }

    // The blocked matmul itself: the register-resident tile walk lands bit-for-bit on the independent single-vector reference for a row,
    // which is what keeps both the explicit dense-matrix selection and this crossover free of any numeric tolerance.
    [Fact]
    public void GemmBatch_Q4KRegisterResidentTile_IsBitIdenticalToSingleVectorGemv()
    {
        const int Rows = 130;
        const int Cols = 300; // a ragged final quant block inside the tile walk

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(Rows, Cols, QuantFormat.Q4_K, seed: 65_000, vectors: 1);
        WeightRef weight = new(weights, new TensorShape([Rows, Cols]), QuantFormat.Q4_K);

        using CpuBackend backend = new(threads: 1);
        float[] viaGemv = new float[Rows];
        backend.Gemv(inputs, weight, viaGemv);

        float[] viaGemm = new float[Rows];
        backend.GemmBatch(inputs, 1, weight, viaGemm);

        BatchKernelTests.AssertBitsEqual(viaGemv, viaGemm, $"Q4_K blocked matmul vs single-vector GEMV at {Rows}x{Cols}");
    }

    // ---------------------------------------------------------------- no posture promotes Q4_K to that route

    [Fact]
    public void Resolve_Q4KMatmulRouteWidthStaysUnpromotedAtEveryPosture()
    {
        Assert.Equal(0, CpuKernelTuning.Resolve(32, vectorPosture: true, CpuKernelTuningOverrides.Auto).MatmulRouteBatchWidthQ4K); // the AVX2 posture keeps Q4_K on its batched-row kernels whatever it promotes for Q6_K
        Assert.Equal(8, CpuKernelTuning.Resolve(32, vectorPosture: true, CpuKernelTuningOverrides.Auto).MatmulRouteBatchWidth); // that promotion is still measured and in force for the format it was measured on
        Assert.Equal(0, CpuKernelTuning.Resolve(32, vectorPosture: false, CpuKernelTuningOverrides.Auto).MatmulRouteBatchWidthQ4K); // without a vector posture no tile kernel runs anyway
    }

    [Fact]
    public void BackendOpen_Avx2OnlyXeonLanePosture_DoesNotPromoteQ4KToTheBlockedMatmulRoute()
    {
        using CpuBackend xeonLane = new(threads: 8, cpuFeatures: new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: false));

        Assert.Equal(0, xeonLane.ResolvedTuning.MatmulRouteBatchWidthQ4K);
        Assert.Equal(8, xeonLane.ResolvedTuning.MatmulRouteBatchWidth); // the measured Q6_K crossover stays as it is; nothing here widens it to another format

        using CpuBackend pinned = new(threads: 8, cpuFeatures: new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: false), tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidthQ4K = 8 });
        Assert.Equal(8, pinned.ResolvedTuning.MatmulRouteBatchWidthQ4K); // a stated width is what opens that door on this lane, and only one
    }

    [Theory]
    [InlineData(0)] // stating 0 restates today's traversal explicitly
    [InlineData(8)]
    [InlineData(64)]
    public void Resolve_Q4KMatmulRouteWidthPinResolvesExactly(int width)
    {
        CpuKernelTuning tuning = CpuKernelTuning.Resolve(32, vectorPosture: true, new CpuKernelTuningOverrides { MatmulRouteBatchWidthQ4K = width });

        Assert.Equal(width, tuning.MatmulRouteBatchWidthQ4K);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(65)] // past the batch ceiling every dense kernel's scratch is sized for
    public void Resolve_Q4KMatmulRouteWidthPinOutsideItsSpan_RefusesNamingQualifiedKeyAndValue(int value)
    {
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(
            () => CpuKernelTuning.Resolve(8, vectorPosture: true, new CpuKernelTuningOverrides { MatmulRouteBatchWidthQ4K = value }));

        Assert.Equal("engine.cpu.tuning.matmul_route_batch_width_q4k", error.ParamName);
        Assert.Contains($"engine.cpu.tuning.matmul_route_batch_width_q4k {value}", error.Message);
    }

    [Fact]
    public void GemvBatch_Q6KMatmulRoutePin_DoesNotExtendToQ4K()
    {
        const int Rows = 300;
        const int Cols = 1024; // block-aligned, so the scratch batch-dot route is eligible and stays in force here
        const int Count = 16;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(Rows, Cols, QuantFormat.Q4_K, seed: 66_000, vectors: Count);
        WeightRef weight = new(weights, new TensorShape([Rows, Cols]), QuantFormat.Q4_K);

        using CpuBackend reference = new(threads: 1);
        float[] expected = new float[Count * Rows];
        reference.GemvBatch(inputs, count: Count, weight, expected);

        using CpuBackend pinnedForQ6K = new(threads: 1, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidth = 8 });
        float[] actual = new float[Count * Rows];
        pinnedForQ6K.GemvBatch(inputs, Count, weight, actual);

        BatchKernelTests.AssertBitsEqual(expected, actual, $"the Q6_K route pin leaves a Q4_K batch({Count}) at {Rows}x{Cols} on its own traversal");
    }

    // ---------------------------------------------------------------- the fused gate/up pair route at that width

    [Fact]
    public void TryGemvBatchSharedInput_Q4KPair_FusesAtTheDefaultPosture()
    {
        const int Rows = 256;
        const int Cols = 1024; // block-aligned for both matrices, as a Qwen feed-forward gate/up pair is

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(Rows, Cols, QuantFormat.Q4_K, seed: 67_000, vectors: 16);
        WeightRef weight = new(weights, new TensorShape([Rows, Cols]), QuantFormat.Q4_K);

        using CpuBackend backend = new(threads: 8, cpuFeatures: new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: false));
        float[] outA = new float[16 * Rows];
        float[] outB = new float[16 * Rows];

        bool fused = ((IDenseWeightPairFusion)backend).TryGemvBatchSharedInput(inputs, 16, weight, outA, weight, outB);

        Assert.True(fused); // today's posture keeps gate/up walking as one shared-input pair: the Q4_K crossover is not promoted by it
    }

    [Fact]
    public void TryGemvBatchSharedInput_Q4KPair_FusesAtStatedQ4KWidth_BitIdenticallyToDefaultPosture()
    {
        const int Rows = 256;
        const int Cols = 1024; // block-aligned, so the scratch route fuses this pair at every posture

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(Rows, Cols, QuantFormat.Q4_K, seed: 68_000, vectors: 16);
        WeightRef weight = new(weights, new TensorShape([Rows, Cols]), QuantFormat.Q4_K);

        using CpuBackend defaultPosture = new(threads: 1);
        float[] expectedA = new float[16 * Rows];
        float[] expectedB = new float[16 * Rows];
        Assert.True(((IDenseWeightPairFusion)defaultPosture).TryGemvBatchSharedInput(inputs, 16, weight, expectedA, weight, expectedB));

        using CpuBackend statedWidth = new(threads: 1, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidthQ4K = 8 });
        float[] outA = new float[16 * Rows];
        float[] outB = new float[16 * Rows];

        Assert.True(((IDenseWeightPairFusion)statedWidth).TryGemvBatchSharedInput(inputs, 16, weight, outA, weight, outB)); // the crossover says nothing about a shared-input pair: this traversal is its own measured posture
        BatchKernelTests.AssertBitsEqual(expectedA, outA, $"gate side at {Rows}x{Cols} under a stated Q4_K width");
        BatchKernelTests.AssertBitsEqual(expectedB, outB, $"up side at {Rows}x{Cols} under a stated Q4_K width");
    }

    [Fact]
    public void TryGemvBatchSharedInput_Q6KPair_IsUnaffectedByAStatedQ4KWidth()
    {
        const int Rows = 256;
        const int Cols = 1024;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(Rows, Cols, QuantFormat.Q6_K, seed: 69_000, vectors: 16);
        WeightRef weight = new(weights, new TensorShape([Rows, Cols]), QuantFormat.Q6_K);

        using CpuBackend backend = new(threads: 1, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidthQ4K = 8 });
        float[] outA = new float[16 * Rows];
        float[] outB = new float[16 * Rows];

        Assert.True(((IDenseWeightPairFusion)backend).TryGemvBatchSharedInput(inputs, 16, weight, outA, weight, outB)); // a stated crossover says nothing about any fused scratch-eligible pair route
    }

    [Fact]
    public void TryGemvBatchSharedInput_MixedFormatPair_FusesAtStatedQ4KWidth_BitIdenticallyToDefaultPosture()
    {
        const int Rows = 256;
        const int Cols = 1024;

        (float[] inputs, byte[] q4Weights) = BatchKernelTests.CreateCase(Rows, Cols, QuantFormat.Q4_K, seed: 70_000, vectors: 16);
        (_, byte[] q6Weights) = BatchKernelTests.CreateCase(Rows, Cols, QuantFormat.Q6_K, seed: 70_001, vectors: 16);
        WeightRef gate = new(q4Weights, new TensorShape([Rows, Cols]), QuantFormat.Q4_K);
        WeightRef up = new(q6Weights, new TensorShape([Rows, Cols]), QuantFormat.Q6_K);

        using CpuBackend defaultPosture = new(threads: 1);
        float[] expectedA = new float[16 * Rows];
        float[] expectedB = new float[16 * Rows];
        Assert.True(((IDenseWeightPairFusion)defaultPosture).TryGemvBatchSharedInput(inputs, 16, gate, expectedA, up, expectedB));

        using CpuBackend statedWidth = new(threads: 1, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidthQ4K = 8 });
        float[] outA = new float[16 * Rows];
        float[] outB = new float[16 * Rows];

        Assert.True(((IDenseWeightPairFusion)statedWidth).TryGemvBatchSharedInput(inputs, 16, gate, outA, up, outB)); // a stated crossover for one scratch-eligible format does not split the pair route apart
        BatchKernelTests.AssertBitsEqual(expectedA, outA, $"mixed-format gate side at {Rows}x{Cols} under a stated Q4_K width");
        BatchKernelTests.AssertBitsEqual(expectedB, outB, $"mixed-format up side at {Rows}x{Cols} under a stated Q4_K width");
    }
}
