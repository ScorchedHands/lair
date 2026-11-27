using Lair.Backends.Cpu;
using Lair.Contracts;

namespace Lair.Tests;

/// <summary>
/// The stated crossover that lets a shared-input fused pair hand each matrix's rows to the blocked matmul tile executor instead of walking its two staged rows once per chain group (the seam's own traversal re-reads the activation batch for every weight row, where the tile executor amortizes one activation-vector load across a tile group's eight rows' chains). No resolved posture states this width: an auto open keeps today's pair traversal exactly as it is, and only an internal seam (backend tuning overrides or a bench pin) opens the other route for measurement. Whatever route a fused call takes, every output element lands bit-for-bit on what that matrix's independent batched pass writes — each format's tile walk carries its own ascending chunk order, FMA association, and scalar-tail cleanup contract from the dense-matrix crossover tests.
/// </summary>
[Trait("Category", "Kernel")]
public sealed class FusedPairMatmulRouteTests
{
    // The feed-forward geometry this seam exists for (a Qwen-shaped gate/up pair 1536x2048) plus shapes straddling the tile boundary: a column count past one 256-element column block, fewer rows than one tile group. Ragged quant-group columns ride their own cases below, because fusion's alignment rules decline them for Q4_K pairs outright.
    private static readonly (int Rows, int Cols)[] MatrixSizes = [(9, 288), (130, 512), (512, 768), (1536, 2048)];

    private static readonly int[] RouteCounts = [8, 9, 16, 32, 64];

    public static TheoryData<int, int> MatrixSizesData()
    {
        TheoryData<int, int> data = [];
        foreach ((int rows, int cols) in MatrixSizes)
        {
            data.Add(rows, cols);
        }

        return data;
    }

    // ---------------------------------------------------------------- no posture states the pair crossover

    [Fact]
    public void Resolve_PairMatmulRouteWidthStaysUnstatedAtEveryPosture()
    {
        Assert.Equal(0, CpuKernelTuning.Resolve(32, vectorPosture: true, CpuKernelTuningOverrides.Auto).PairMatmulRouteBatchWidth); // the AVX2 posture keeps every fused pair on its staged traversal whatever it promotes for single-matrix passes
        Assert.Equal(8, CpuKernelTuning.Resolve(32, vectorPosture: true, CpuKernelTuningOverrides.Auto).MatmulRouteBatchWidth); // that measured Q6_K crossover stays exactly as it is
        Assert.Equal(0, CpuKernelTuning.Resolve(1, vectorPosture: false, CpuKernelTuningOverrides.Auto).PairMatmulRouteBatchWidth); // without a vector posture no tile kernel runs anyway
    }

    [Fact]
    public void BackendOpen_Avx2OnlyXeonLanePosture_DoesNotStateThePairCrossover()
    {
        using CpuBackend xeonLane = new(threads: 8, cpuFeatures: new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: false));

        Assert.Equal(0, xeonLane.ResolvedTuning.PairMatmulRouteBatchWidth); // an auto open keeps fused gate/up on today's traversal
        Assert.Equal(8, xeonLane.ResolvedTuning.MatmulRouteBatchWidth); // the single-matrix crossover it was measured at stays untouched

        using CpuBackend pinned = new(threads: 8, cpuFeatures: new CpuFeaturePosture(Avx2Available: true, AvxVnniAvailable: false), tuning: new CpuKernelTuningOverrides { PairMatmulRouteBatchWidth = 8 });
        Assert.Equal(8, pinned.ResolvedTuning.PairMatmulRouteBatchWidth); // a stated width is what opens that door on this lane, and only one
    }

    [Theory]
    [InlineData(0)] // stating 0 restates today's traversal explicitly
    [InlineData(8)]
    [InlineData(64)]
    public void Resolve_PairMatmulRouteWidthPinResolvesExactly(int width)
    {
        CpuKernelTuning tuning = CpuKernelTuning.Resolve(32, vectorPosture: true, new CpuKernelTuningOverrides { PairMatmulRouteBatchWidth = width });

        Assert.Equal(width, tuning.PairMatmulRouteBatchWidth);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(65)] // past the batch ceiling every dense kernel's scratch is sized for
    public void Resolve_PairMatmulRouteWidthPinOutsideItsSpan_RefusesNamingQualifiedKeyAndValue(int value)
    {
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(
            () => CpuKernelTuning.Resolve(8, vectorPosture: true, new CpuKernelTuningOverrides { PairMatmulRouteBatchWidth = value }));

        Assert.Equal("engine.cpu.tuning.pair_matmul_route_batch_width", error.ParamName);
        Assert.Contains($"engine.cpu.tuning.pair_matmul_route_batch_width {value}", error.Message);
    }

    // ---------------------------------------------------------------- the staged traversal stays in force by default

    [Theory]
    [MemberData(nameof(MatrixSizesData))]
    public void FusedPair_DefaultPosture_KeepsStagedTraversal_BitIdenticalToIndependentCalls(int rows, int cols)
    {
        foreach (int count in RouteCounts)
        {
            using CpuBackend backend = new(threads: 1);

            (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q4_K, seed: 21_000 + count, vectors: count);
            (_, byte[] weightsB) = BatchKernelTests.CreateCase(rows, cols, QuantFormat.Q6_K, seed: 22_000 + count, vectors: count);

            WeightRef gate = new(weightsA, new TensorShape([rows, cols]), QuantFormat.Q4_K);
            WeightRef up = new(weightsB, new TensorShape([rows, cols]), QuantFormat.Q6_K);

            float[] expectedA = new float[count * rows];
            float[] expectedB = new float[count * rows];
            backend.GemvBatch(inputs, count, gate, expectedA);
            backend.GemvBatch(inputs, count, up, expectedB);

            Assert.Equal(0, backend.ResolvedTuning.PairMatmulRouteBatchWidth); // the auto posture states nothing here

            float[] actualA = new float[count * rows];
            float[] actualB = new float[count * rows];
            bool ran = ((IDenseWeightPairFusion)backend).TryGemvBatchSharedInput(inputs, count, gate, actualA, up, actualB);

            Assert.Equal(count >= 8, ran); // fusion engages from one full chain group of batch vectors up, exactly as before this seam existed
            if (!ran)
            {
                continue;
            }

            BatchKernelTests.AssertBitsEqual(expectedA, actualA, $"default staged traversal A {rows}x{cols} count({count})");
            BatchKernelTests.AssertBitsEqual(expectedB, actualB, $"default staged traversal B {rows}x{cols} count({count})");
        }
    }

    [Fact]
    public void FusedPair_StatedQ4KCrossover_DoesNotMoveFusedPairsOffTheirTraversal()
    {
        // The single-matrix crossover remains a statement about single-matrix passes: stating it must not switch any fused pair to the tile executor.
        const int Rows = 256;
        const int Cols = 1024;

        (float[] inputs, byte[] weights) = BatchKernelTests.CreateCase(Rows, Cols, QuantFormat.Q4_K, seed: 23_000, vectors: 16);
        WeightRef weight = new(weights, new TensorShape([Rows, Cols]), QuantFormat.Q4_K);

        using CpuBackend defaultPosture = new(threads: 1);
        float[] expectedA = new float[16 * Rows];
        float[] expectedB = new float[16 * Rows];
        Assert.True(((IDenseWeightPairFusion)defaultPosture).TryGemvBatchSharedInput(inputs, 16, weight, expectedA, weight, expectedB));

        using CpuBackend statedQ4K = new(threads: 1, tuning: new CpuKernelTuningOverrides { MatmulRouteBatchWidthQ4K = 8 });
        Assert.Equal(0, statedQ4K.ResolvedTuning.PairMatmulRouteBatchWidth); // stating the single-matrix crossover states nothing about pairs

        float[] outA = new float[16 * Rows];
        float[] outB = new float[16 * Rows];
        Assert.True(((IDenseWeightPairFusion)statedQ4K).TryGemvBatchSharedInput(inputs, 16, weight, outA, weight, outB)); // the pair stays on its own measured traversal
        BatchKernelTests.AssertBitsEqual(expectedA, outA, $"gate side under a stated Q4_K crossover at {Rows}x{Cols}");
        BatchKernelTests.AssertBitsEqual(expectedB, outB, $"up side under a stated Q4_K crossover at {Rows}x{Cols}");
    }

    // ---------------------------------------------------------------- the stated pair crossover: bit-identical tile traversal

    [Theory]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q6_K, QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K)] // the q4_k_m feed-forward posture: Q4_K gate over a Q6_K up.
    public void FusedPair_PairMatmulRoute_IsBitIdenticalToStagedTraversalAndIndependentCalls(QuantFormat formatA, QuantFormat formatB)
    {
        foreach ((int rows, int cols) in MatrixSizes)
        {
            foreach (int count in RouteCounts)
            {
                (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rows, cols, formatA, seed: 31_000 + count, vectors: count);
                (_, byte[] weightsB) = BatchKernelTests.CreateCase(rows, cols, formatB, seed: 42_000 + count, vectors: count);

                WeightRef gate = new(weightsA, new TensorShape([rows, cols]), formatA);
                WeightRef up = new(weightsB, new TensorShape([rows, cols]), formatB);

                using CpuBackend independent = new(threads: 1);
                float[] expectedA = new float[count * rows];
                float[] expectedB = new float[count * rows];
                independent.GemvBatch(inputs, count, gate, expectedA);
                independent.GemvBatch(inputs, count, up, expectedB);

                using CpuBackend traversalRoute = new(threads: 1); // default posture: the staged pair traversal runs today
                float[] traversalA = new float[count * rows];
                float[] traversalB = new float[count * rows];
                Assert.True(traversalRoute.TryGemvBatchSharedInput(inputs, count, gate, traversalA, up, traversalB), $"{formatA}+{formatB} pair traversal refused at the default posture");

                using CpuBackend tileRoute = new(threads: 1, tuning: new CpuKernelTuningOverrides { PairMatmulRouteBatchWidth = 8 });
                Assert.Equal(8, tileRoute.ResolvedTuning.PairMatmulRouteBatchWidth);
                float[] tileA = new float[count * rows];
                float[] tileB = new float[count * rows];
                Assert.True(tileRoute.TryGemvBatchSharedInput(inputs, count, gate, tileA, up, tileB), $"{formatA}+{formatB} fused pair refused the stated crossover at {rows}x{cols}");

                BatchKernelTests.AssertBitsEqual(expectedA, traversalA, $"traversal A {formatA}/{formatB} batch({count}) at {rows}x{cols}");
                BatchKernelTests.AssertBitsEqual(expectedA, tileA, $"tile-routed pair A {formatA}/{formatB} batch({count}) at {rows}x{cols}");
                BatchKernelTests.AssertBitsEqual(traversalA, tileA, $"pair traversal vs tile route A {formatA}/{formatB} batch({count})");
                BatchKernelTests.AssertBitsEqual(expectedB, tileB, $"tile-routed pair B {formatA}/{formatB} batch({count}) at {rows}x{cols}");
                BatchKernelTests.AssertBitsEqual(traversalB, tileB, $"pair traversal vs tile route B {formatA}/{formatB} batch({count})");
            }
        }
    }

    [Theory]
    [InlineData(8)] // below a stated sixteen: the staged pair traversal stays in force
    [InlineData(9)] // ragged beside it too — full groups plus one-chain tails, still on today's route
    public void FusedPair_BelowStatedWidth_KeepsStagedTraversal_BitIdentical(int count)
    {
        const int Rows = 300;
        const int Cols = 512;

        (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(Rows, Cols, QuantFormat.Q4_K, seed: 51_000 + count, vectors: count);
        (_, byte[] weightsB) = BatchKernelTests.CreateCase(Rows, Cols, QuantFormat.Q6_K, seed: 52_000 + count, vectors: count);

        WeightRef gate = new(weightsA, new TensorShape([Rows, Cols]), QuantFormat.Q4_K);
        WeightRef up = new(weightsB, new TensorShape([Rows, Cols]), QuantFormat.Q6_K);

        using CpuBackend reference = new(threads: 1); // the default posture's own fused traversal is this case's expected value
        float[] expectedA = new float[count * Rows];
        float[] expectedB = new float[count * Rows];
        Assert.True(reference.TryGemvBatchSharedInput(inputs, count, gate, expectedA, up, expectedB));

        using CpuBackend pinnedWide = new(threads: 1, tuning: new CpuKernelTuningOverrides { PairMatmulRouteBatchWidth = 16 }); // stated at sixteen: this batch keeps today's traversal
        float[] actualA = new float[count * Rows];
        float[] actualB = new float[count * Rows];
        Assert.True(pinnedWide.TryGemvBatchSharedInput(inputs, count, gate, actualA, up, actualB), $"pair fusion refused below the stated width at batch({count})");

        BatchKernelTests.AssertBitsEqual(expectedA, actualA, $"below-width pair A at {Rows}x{Cols} batch({count})");
        BatchKernelTests.AssertBitsEqual(expectedB, actualB, $"below-width pair B at {Rows}x{Cols} batch({count})");
    }

    [Fact]
    public void FusedPair_Q6KOnlyPair_RaggedQuantGroupColumns_TileRouteIsBitIdentical()
    {
        // 272 columns is block-aligned for Q6_K (sixteen-element groups) but not for Q4_K: a pure-Q6_K pair fuses at this shape today, and its stated crossover must land on the same bytes.
        const int Rows = 130;
        const int Cols = 272;
        const int Count = 9;

        (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(Rows, Cols, QuantFormat.Q6_K, seed: 53_000, vectors: Count);
        (_, byte[] weightsB) = BatchKernelTests.CreateCase(Rows, Cols, QuantFormat.Q6_K, seed: 54_000, vectors: Count);

        WeightRef gate = new(weightsA, new TensorShape([Rows, Cols]), QuantFormat.Q6_K);
        WeightRef up = new(weightsB, new TensorShape([Rows, Cols]), QuantFormat.Q6_K);

        using CpuBackend traversalRoute = new(threads: 1);
        float[] expectedA = new float[Count * Rows];
        float[] expectedB = new float[Count * Rows];
        Assert.True(traversalRoute.TryGemvBatchSharedInput(inputs, Count, gate, expectedA, up, expectedB), "Q6_K pair traversal refused at a Q6_K-aligned ragged column count");

        using CpuBackend tileRoute = new(threads: 1, tuning: new CpuKernelTuningOverrides { PairMatmulRouteBatchWidth = 8 });
        float[] actualA = new float[Count * Rows];
        float[] actualB = new float[Count * Rows];
        Assert.True(tileRoute.TryGemvBatchSharedInput(inputs, Count, gate, actualA, up, actualB), "Q6_K tile-routed pair refused at a ragged column count");

        BatchKernelTests.AssertBitsEqual(expectedA, actualA, $"ragged-column Q6_K tile route A at {Rows}x{Cols}");
        BatchKernelTests.AssertBitsEqual(expectedB, actualB, $"ragged-column Q6_K tile route B at {Rows}x{Cols}");
    }

    [Fact]
    public void FusedPair_PairMatmulRoute_Threaded_IsBitIdenticalToSingleThreadedTileRoute()
    {
        const int Rows = 1536;
        const int Cols = 2048; // a Qwen-shaped gate/up pair: block-aligned for both formats and wide enough to span several column blocks.

        (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(Rows, Cols, QuantFormat.Q4_K, seed: 61_000, vectors: 32);
        (_, byte[] weightsB) = BatchKernelTests.CreateCase(Rows, Cols, QuantFormat.Q6_K, seed: 62_000, vectors: 32);

        WeightRef gate = new(weightsA, new TensorShape([Rows, Cols]), QuantFormat.Q4_K);
        WeightRef up = new(weightsB, new TensorShape([Rows, Cols]), QuantFormat.Q6_K);

        using CpuBackend single = new(threads: 1, tuning: new CpuKernelTuningOverrides { PairMatmulRouteBatchWidth = 8 });
        float[] expectedA = new float[32 * Rows];
        float[] expectedB = new float[32 * Rows];
        Assert.True(single.TryGemvBatchSharedInput(inputs, 32, gate, expectedA, up, expectedB));

        // Forced tiny row blocks past the threshold hand every tile-row group to a worker: rows are independent, so any assignment lands on the same bytes.
        using CpuBackend threaded = new(threads: 4, tuning: new CpuKernelTuningOverrides { PairMatmulRouteBatchWidth = 8, RowBlock = 3, ParallelRowThreshold = 1 });
        float[] actualA = new float[32 * Rows];
        float[] actualB = new float[32 * Rows];
        Assert.True(threaded.TryGemvBatchSharedInput(inputs, 32, gate, actualA, up, actualB), "threaded tile-routed pair refused");

        BatchKernelTests.AssertBitsEqual(expectedA, actualA, $"threaded tile-routed pair A at {Rows}x{Cols}");
        BatchKernelTests.AssertBitsEqual(expectedB, actualB, $"threaded tile-routed pair B at {Rows}x{Cols}");
    }

    [Fact]
    public void FusedPair_PairMatmulRoute_ScalarPosture_IsBitIdenticalToIndependentCalls()
    {
        const int Rows = 130; // fewer rows than one tile group, so the scalar tiles run their partial row walk.
        const int Cols = 544; // block-aligned for both scratch formats (2 x 272-element column blocks: past one super-block wide tile) while a ragged final quant group stays inside it.

        (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(Rows, Cols, QuantFormat.Q4_K, seed: 71_000, vectors: 8);
        (_, byte[] weightsB) = BatchKernelTests.CreateCase(Rows, Cols, QuantFormat.Q6_K, seed: 72_000, vectors: 8);

        WeightRef gate = new(weightsA, new TensorShape([Rows, Cols]), QuantFormat.Q4_K);
        WeightRef up = new(weightsB, new TensorShape([Rows, Cols]), QuantFormat.Q6_K);

        using CpuBackend independent = new(threads: 1, allowAvx2: false);
        float[] expectedA = new float[8 * Rows];
        float[] expectedB = new float[8 * Rows];
        independent.GemvBatch(inputs, 8, gate, expectedA);
        independent.GemvBatch(inputs, 8, up, expectedB);

        using CpuBackend tileRoute = new(threads: 1, allowAvx2: false, tuning: new CpuKernelTuningOverrides { PairMatmulRouteBatchWidth = 8 });
        float[] actualA = new float[8 * Rows];
        float[] actualB = new float[8 * Rows];
        Assert.True(tileRoute.TryGemvBatchSharedInput(inputs, 8, gate, actualA, up, actualB), "scalar tile-routed pair refused");

        BatchKernelTests.AssertBitsEqual(expectedA, actualA, $"scalar tile-routed pair A at {Rows}x{Cols}");
        BatchKernelTests.AssertBitsEqual(expectedB, actualB, $"scalar tile-routed pair B at {Rows}x{Cols}");
    }

    // ---------------------------------------------------------------- the seam does not widen fusion's scope

    [Theory]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K)]
    public void FusedPair_OutsideScratchPostures_StillRefuseWithTheCrossoverStated(QuantFormat formatA, QuantFormat formatB)
    {
        int rows = 100;
        int cols = 137; // ragged for every block layout: the fused route declines rather than approximate, stated crossover or not.

        foreach (int count in new[] { 8 })
        {
            using CpuBackend backend = new(threads: 1, tuning: new CpuKernelTuningOverrides { PairMatmulRouteBatchWidth = 8 });

            (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rows, cols, formatA, seed: 81_000 + count, vectors: count);
            (_, byte[] weightsB) = BatchKernelTests.CreateCase(rows, cols, formatB, seed: 82_000 + count, vectors: count);

            TensorShape shape = new([rows, cols]);

            float[] outputsA = [.. Enumerable.Repeat(float.NaN, count * rows)];
            float[] outputsB = [.. Enumerable.Repeat(float.NaN, count * rows)];

            bool ran = backend.TryGemvBatchSharedInput(inputs, count, new WeightRef(weightsA, shape, formatA), outputsA, new WeightRef(weightsB, shape, formatB), outputsB);

            Assert.False(ran, $"{formatA}+{formatB} at {rows}x{cols} must refuse shared-input fusion even with the pair crossover stated");
            Assert.All(outputsA, value => Assert.True(float.IsNaN(value)));
            Assert.All(outputsB, value => Assert.True(float.IsNaN(value)));
        }
    }

    [Theory]
    [InlineData(QuantFormat.IQ4_XS, QuantFormat.IQ4_XS)] // an integer-dot i-quant: outside the scratch formats this seam routes at any posture.
    public void FusedPair_NonScratchFormats_StillRefuseWithTheCrossoverStated(QuantFormat formatA, QuantFormat formatB)
    {
        int rows = 100;
        int cols = 512; // block-aligned: the refusal here is about the format set, not column alignment.

        using CpuBackend backend = new(threads: 1, tuning: new CpuKernelTuningOverrides { PairMatmulRouteBatchWidth = 8 });

        (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rows, cols, formatA, seed: 91_000, vectors: 8);
        (_, byte[] weightsB) = BatchKernelTests.CreateCase(rows, cols, formatB, seed: 92_000, vectors: 8);

        TensorShape shape = new([rows, cols]);

        float[] outputsA = [.. Enumerable.Repeat(float.NaN, 8 * rows)];
        float[] outputsB = [.. Enumerable.Repeat(float.NaN, 8 * rows)];

        bool ran = backend.TryGemvBatchSharedInput(inputs, 8, new WeightRef(weightsA, shape, formatA), outputsA, new WeightRef(weightsB, shape, formatB), outputsB);

        Assert.False(ran, $"{formatA}+{formatB} must refuse shared-input fusion even with the pair crossover stated");
        Assert.All(outputsA, value => Assert.True(float.IsNaN(value)));
        Assert.All(outputsB, value => Assert.True(float.IsNaN(value)));
    }
}
