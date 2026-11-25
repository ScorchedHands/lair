using Lair.Backends.Cpu;
using Lair.Contracts;

namespace Lair.Tests;

/// <summary>
/// Shared-input dense pair fusion (the CPU backend's IDenseWeightPairFusion seam): every fused output element must match the value
/// an independent GemvBatch call for that matrix writes, bit-for-bit, single-threaded and threaded — same accumulator order per
/// chain, same row partitioning under threading. Postures outside the fused format set refuse with false and leave outputs untouched,
/// so callers keep their independent calls there.
/// </summary>
[Trait("Category", "Kernel")]
public class DenseWeightFusionTests
{
    // The engine's own dense matrix shapes (block-aligned columns) plus a ragged-column case that must refuse fusion outright.
    private static readonly (int Rows, int Cols)[] MatrixSizes = [(256, 384), (1024, 512)];

    private static readonly int[] BatchCounts = [1, 2, 7, 8, 9, 16, 32, 64];

    [Theory]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q6_K, QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K)] // the q4_k_m feed-forward posture: Q4_K gate over a Q6_K up.
    public void FusedPair_IsBitIdenticalToIndependentGemvBatchCalls(QuantFormat formatA, QuantFormat formatB)
    {
        foreach ((int rows, int cols) in MatrixSizes)
        {
            foreach (int count in BatchCounts)
            {
                using CpuBackend backend = new(threads: 1);

                (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rows, cols, formatA, seed: 1234 + count, vectors: count);
                (_, byte[] weightsB) = BatchKernelTests.CreateCase(rows, cols, formatB, seed: 9021 + count, vectors: count);

                TensorShape shapeA = new([rows, cols]);
                TensorShape shapeB = new([rows, cols]);

                float[] expectedA = new float[count * rows];
                float[] expectedB = new float[count * rows];
                backend.GemvBatch(inputs, count, new WeightRef(weightsA, shapeA, formatA), expectedA);
                backend.GemvBatch(inputs, count, new WeightRef(weightsB, shapeB, formatB), expectedB);

                float[] actualA = new float[count * rows];
                float[] actualB = new float[count * rows];

                IDenseWeightPairFusion fusion = backend;
                bool ran = fusion.TryGemvBatchSharedInput(inputs, count, new WeightRef(weightsA, shapeA, formatA), actualA, new WeightRef(weightsB, shapeB, formatB), actualB);

                // Fusion engages from one full register group of chains up; below that the backend declines and the caller keeps its
                // independent calls. Either route must land bit-identical to those independent calls.
                Assert.Equal(count >= 8, ran);
                if (!ran)
                {
                    continue; // refused postures leave the outputs untouched — the caller's independent GemvBatch calls are what run there.
                }

                BatchKernelTests.AssertBitsEqual(expectedA, actualA, $"fused A {formatA}/{formatB} batch({count}) at {rows}x{cols}");
                BatchKernelTests.AssertBitsEqual(expectedB, actualB, $"fused B {formatA}/{formatB} batch({count}) at {rows}x{cols}");
            }
        }
    }

    [Theory]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q6_K, QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K)]
    public void FusedPair_Threaded_IsBitIdenticalToSingleThreadedAndIndependentCalls(QuantFormat formatA, QuantFormat formatB)
    {
        // Row counts past the backend's parallel threshold so both routes really thread; row independence must keep every element
        // identical no matter which worker claims which block.
        foreach (int count in new[] { 8, 32 })
        {
            using CpuBackend single = new(threads: 1);
            using CpuBackend threaded = new(threads: 4);

            int rows = 512;
            int cols = 768;

            (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rows, cols, formatA, seed: 30_000 + count, vectors: count);
            (_, byte[] weightsB) = BatchKernelTests.CreateCase(rows, cols, formatB, seed: 41_000 + count, vectors: count);

            TensorShape shape = new([rows, cols]);
            WeightRef weightA = new(weightsA, shape, formatA);
            WeightRef weightB = new(weightsB, shape, formatB);

            float[] expectedSingleA = new float[count * rows];
            float[] expectedSingleB = new float[count * rows];
            single.GemvBatch(inputs, count, weightA, expectedSingleA);
            single.GemvBatch(inputs, count, weightB, expectedSingleB);

            float[] threadedIndependentA = new float[count * rows];
            float[] threadedIndependentB = new float[count * rows];
            threaded.GemvBatch(inputs, count, weightA, threadedIndependentA);
            threaded.GemvBatch(inputs, count, weightB, threadedIndependentB);

            float[] fusedSingleA = new float[count * rows];
            float[] fusedSingleB = new float[count * rows];
            Assert.True(single.TryGemvBatchSharedInput(inputs, count, weightA, fusedSingleA, weightB, fusedSingleB));

            float[] fusedThreadedA = new float[count * rows];
            float[] fusedThreadedB = new float[count * rows];
            Assert.True(threaded.TryGemvBatchSharedInput(inputs, count, weightA, fusedThreadedA, weightB, fusedThreadedB), "threaded fusion refused");

            BatchKernelTests.AssertBitsEqual(expectedSingleA, fusedSingleA, $"single-thread fused A {formatA}/{formatB} batch({count})");
            BatchKernelTests.AssertBitsEqual(expectedSingleB, fusedSingleB, $"single-thread fused B {formatA}/{formatB} batch({count})");
            BatchKernelTests.AssertBitsEqual(expectedSingleA, threadedIndependentA, $"threaded independent A {formatA}/{formatB} batch({count})");
            BatchKernelTests.AssertBitsEqual(expectedSingleA, fusedThreadedA, $"threaded fused A {formatA}/{formatB} batch({count})");
            BatchKernelTests.AssertBitsEqual(expectedSingleB, fusedThreadedB, $"threaded fused B {formatA}/{formatB} batch({count})");
        }
    }

    [Theory]
    // Formats outside the scratch route (integer-dot i-quants included), and a block-unaligned column count on an eligible format.
    [InlineData(QuantFormat.Q8_0, QuantFormat.Q8_0)]
    [InlineData(QuantFormat.IQ4_XS, QuantFormat.IQ4_XS)]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K)]
    public void FusedPair_OutsideFusedPostures_RefusesWithoutTouchingOutputs(QuantFormat formatA, QuantFormat formatB)
    {
        int rows = 100;
        int cols = 137; // ragged for every block layout: the fused route must decline rather than approximate.

        foreach (int count in new[] { 8 })
        {
            using CpuBackend backend = new(threads: 1);

            (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rows, cols, formatA, seed: 555 + count, vectors: count);
            (_, byte[] weightsB) = BatchKernelTests.CreateCase(rows, cols, formatB, seed: 666 + count, vectors: count);

            TensorShape shape = new([rows, cols]);

            float[] outputsA = [.. Enumerable.Repeat(float.NaN, count * rows)];
            float[] outputsB = [.. Enumerable.Repeat(float.NaN, count * rows)];

            bool ran = backend.TryGemvBatchSharedInput(inputs, count, new WeightRef(weightsA, shape, formatA), outputsA, new WeightRef(weightsB, shape, formatB), outputsB);

            Assert.False(ran, $"{formatA}+{formatB} at {rows}x{cols} must refuse shared-input fusion");
            Assert.All(outputsA, value => Assert.True(float.IsNaN(value)));
            Assert.All(outputsB, value => Assert.True(float.IsNaN(value)));
        }
    }

    [Theory]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q6_K, QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K)]
    public void FusedPair_PairRouteAndFallbackRoute_AreBitIdenticalToEachOtherAndToIndependentCalls(QuantFormat formatA, QuantFormat formatB)
    {
        // The pair scratch ceiling is the seam: at the default posture the pair route (two staged rows, one shared batch walk) runs;
        // with the ceiling pinned between one row and a pair of rows the same call keeps fused execution on the back-to-back
        // single-row fallback. Both routes must land element-for-element on each other and on two independent batch calls.
        int rows = 512;
        int cols = 768;

        foreach (int count in new[] { 8, 9, 16, 32 }) // ragged counts exercise pair-group tails in both routes
        {
            (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rows, cols, formatA, seed: 71_000 + count, vectors: count);
            (_, byte[] weightsB) = BatchKernelTests.CreateCase(rows, cols, formatB, seed: 82_000 + count, vectors: count);

            TensorShape shape = new([rows, cols]);
            WeightRef weightA = new(weightsA, shape, formatA);
            WeightRef weightB = new(weightsB, shape, formatB);

            using CpuBackend independent = new(threads: 1);
            float[] expectedA = new float[count * rows];
            float[] expectedB = new float[count * rows];
            independent.GemvBatch(inputs, count, weightA, expectedA);
            independent.GemvBatch(inputs, count, weightB, expectedB);

            using CpuBackend pairRoute = new(threads: 1); // default posture: 2 x 768 columns stay under the auto scratch ceiling
            Assert.Equal(CpuKernelTuning.PairChainGroupMax, pairRoute.ResolvedTuning.PairChainGroup);
            float[] pairA = new float[count * rows];
            float[] pairB = new float[count * rows];
            Assert.True(pairRoute.TryGemvBatchSharedInput(inputs, count, weightA, pairA, weightB, pairB), $"{formatA}+{formatB} pair route refused at the default posture");

            using CpuBackend fallbackRoute = new(threads: 1, tuning: new CpuKernelTuningOverrides { MaxScratchCols = 1024 }); // one row stageable, a pair of rows not: the fused call keeps the single-row fallback
            float[] fallbackA = new float[count * rows];
            float[] fallbackB = new float[count * rows];
            Assert.True(fallbackRoute.TryGemvBatchSharedInput(inputs, count, weightA, fallbackA, weightB, fallbackB), $"{formatA}+{formatB} fused fallback refused at a pinned scratch ceiling");

            BatchKernelTests.AssertBitsEqual(expectedA, pairA, $"pair route A {formatA}/{formatB} batch({count}) at {rows}x{cols}");
            BatchKernelTests.AssertBitsEqual(expectedB, pairB, $"pair route B {formatA}/{formatB} batch({count}) at {rows}x{cols}");
            BatchKernelTests.AssertBitsEqual(expectedA, fallbackA, $"fallback route A {formatA}/{formatB} batch({count}) at {rows}x{cols}");
            BatchKernelTests.AssertBitsEqual(expectedB, fallbackB, $"fallback route B {formatA}/{formatB} batch({count}) at {rows}x{cols}");
            BatchKernelTests.AssertBitsEqual(pairA, fallbackA, $"pair route vs fallback A {formatA}/{formatB} batch({count})");
            BatchKernelTests.AssertBitsEqual(pairB, fallbackB, $"pair route vs fallback B {formatA}/{formatB} batch({count})");
        }
    }

    [Theory]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q6_K, QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K)]
    public void FusedPair_PinnedNarrowerPairGroup_IsBitIdenticalToIndependentCalls(QuantFormat formatA, QuantFormat formatB)
    {
        // A pinned chain group narrows the pair group derived from it (six chains pair at three, two at one), so the pair dot runs
        // through its stack-staged route. Pair grouping only chooses which batch vectors share one pass over the pair — each chain's
        // element order and reduction stay the same, so the bytes must not move.
        int rows = 512;
        int cols = 768;

        foreach (int chainGroup in new[] { 6, 2 })
        {
            foreach (int count in new[] { 8, 9, 16 })
            {
                (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rows, cols, formatA, seed: 93_000 + count, vectors: count);
                (_, byte[] weightsB) = BatchKernelTests.CreateCase(rows, cols, formatB, seed: 104_000 + count, vectors: count);

                TensorShape shape = new([rows, cols]);
                WeightRef weightA = new(weightsA, shape, formatA);
                WeightRef weightB = new(weightsB, shape, formatB);

                using CpuBackend independent = new(threads: 1);
                float[] expectedA = new float[count * rows];
                float[] expectedB = new float[count * rows];
                independent.GemvBatch(inputs, count, weightA, expectedA);
                independent.GemvBatch(inputs, count, weightB, expectedB);

                using CpuBackend pinned = new(threads: 1, tuning: new CpuKernelTuningOverrides { ChainGroup = chainGroup });
                Assert.Equal(chainGroup / 2, pinned.ResolvedTuning.PairChainGroup);
                float[] actualA = new float[count * rows];
                float[] actualB = new float[count * rows];
                Assert.True(pinned.TryGemvBatchSharedInput(inputs, count, weightA, actualA, weightB, actualB), $"{formatA}+{formatB} fused pair refused at a pinned chain group");

                BatchKernelTests.AssertBitsEqual(expectedA, actualA, $"narrow pair group {chainGroup} A {formatA}/{formatB} batch({count}) at {rows}x{cols}");
                BatchKernelTests.AssertBitsEqual(expectedB, actualB, $"narrow pair group {chainGroup} B {formatA}/{formatB} batch({count}) at {rows}x{cols}");
            }
        }
    }

    [Theory]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q6_K, QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K)]
    public void FusedPair_PairRoute_Threaded_IsBitIdenticalToSingleThreadedPairRoute(QuantFormat formatA, QuantFormat formatB)
    {
        // Row counts past the threaded posture's dispatch threshold so the pair route really threads; rows are independent, so any
        // worker assignment of pair rows must land on the single-threaded pair route's elements exactly.
        int rows = 512;
        int cols = 768;

        foreach (int count in new[] { 8, 9, 32 })
        {
            (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rows, cols, formatA, seed: 115_000 + count, vectors: count);
            (_, byte[] weightsB) = BatchKernelTests.CreateCase(rows, cols, formatB, seed: 126_000 + count, vectors: count);

            TensorShape shape = new([rows, cols]);
            WeightRef weightA = new(weightsA, shape, formatA);
            WeightRef weightB = new(weightsB, shape, formatB);

            using CpuBackend single = new(threads: 1);
            float[] singleA = new float[count * rows];
            float[] singleB = new float[count * rows];
            Assert.True(single.TryGemvBatchSharedInput(inputs, count, weightA, singleA, weightB, singleB), "single-threaded pair route refused");

            using CpuBackend threaded = new(threads: 4, tuning: new CpuKernelTuningOverrides { RowBlock = 3, ParallelRowThreshold = 1 });
            float[] threadedA = new float[count * rows];
            float[] threadedB = new float[count * rows];
            Assert.True(threaded.TryGemvBatchSharedInput(inputs, count, weightA, threadedA, weightB, threadedB), "threaded pair route refused");

            BatchKernelTests.AssertBitsEqual(singleA, threadedA, $"threaded pair route A {formatA}/{formatB} batch({count})");
            BatchKernelTests.AssertBitsEqual(singleB, threadedB, $"threaded pair route B {formatA}/{formatB} batch({count})");
        }
    }

    [Fact]
    public void FusedPair_MismatchedColumnCounts_ThrowsClearMessage()
    {
        using CpuBackend backend = new(threads: 1);

        (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(256, 384, QuantFormat.Q4_K, seed: 7, vectors: 8);
        (_, byte[] weightsB) = BatchKernelTests.CreateCase(256, 512, QuantFormat.Q6_K, seed: 8, vectors: 8);

        float[] outputsA = new float[8 * 256];
        float[] outputsB = new float[8 * 256];

        ArgumentException error = Assert.Throws<ArgumentException>(() => backend.TryGemvBatchSharedInput(
            inputs, 8, new WeightRef(weightsA, new TensorShape([256, 384]), QuantFormat.Q4_K), outputsA,
            new WeightRef(weightsB, new TensorShape([256, 512]), QuantFormat.Q6_K), outputsB));

        Assert.Contains("equal column counts", error.Message);
    }

    // ---------------------------------------------------------------- batched triple fusion (the attention Q/K/V seam)

    // The engine's own QKV row geometry (query region wider than the kv region) plus an all-equal-row case and an asymmetric one: the fused
    // union walk must give every matrix exactly its own rows and its own output stride.
    private static readonly (int RowsA, int RowsB, int RowsC)[] TripleRowShapes = [(256, 256, 256), (512, 384, 256)];
    private static readonly (int RowsA, int RowsB, int RowsC)[] QkvRowShapes = [(2048, 256, 256)];

    [Theory]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q4_K, QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q6_K, QuantFormat.Q6_K, QuantFormat.Q6_K)]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K, QuantFormat.Q4_K)] // a Q4_K query over Q6_K kv halves
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q4_K, QuantFormat.Q6_K)]
    public void FusedTriple_IsBitIdenticalToThreeIndependentGemvBatchCalls(QuantFormat formatA, QuantFormat formatB, QuantFormat formatC)
    {
        foreach ((int rowsA, int rowsB, int rowsC) in TripleRowShapes)
        {
            foreach (int count in BatchCounts)
            {
                using CpuBackend backend = new(threads: 1);

                int cols = 768;
                (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rowsA, cols, formatA, seed: 1234 + count, vectors: count);
                (_, byte[] weightsB) = BatchKernelTests.CreateCase(rowsB, cols, formatB, seed: 9021 + count, vectors: count);
                (_, byte[] weightsC) = BatchKernelTests.CreateCase(rowsC, cols, formatC, seed: 5566 + count, vectors: count);

                WeightRef weightA = new(weightsA, new TensorShape([rowsA, cols]), formatA);
                WeightRef weightB = new(weightsB, new TensorShape([rowsB, cols]), formatB);
                WeightRef weightC = new(weightsC, new TensorShape([rowsC, cols]), formatC);

                float[] expectedA = new float[count * rowsA];
                float[] expectedB = new float[count * rowsB];
                float[] expectedC = new float[count * rowsC];
                backend.GemvBatch(inputs, count, weightA, expectedA);
                backend.GemvBatch(inputs, count, weightB, expectedB);
                backend.GemvBatch(inputs, count, weightC, expectedC);

                float[] actualA = new float[count * rowsA];
                float[] actualB = new float[count * rowsB];
                float[] actualC = new float[count * rowsC];

                IDenseWeightTripleFusion fusion = backend;
                bool ran = fusion.TryGemvBatchSharedInput(inputs, count, weightA, actualA, weightB, actualB, weightC, actualC);

                // Fusion engages from one full chain group of batch vectors up; below that the backend declines and the caller keeps its
                // independent calls. Either route must land bit-identical to those independent calls.
                Assert.Equal(count >= 8, ran);
                if (!ran)
                {
                    continue; // refused postures leave the outputs untouched — the caller's independent GemvBatch calls are what run there.
                }

                BatchKernelTests.AssertBitsEqual(expectedA, actualA, $"fused triple A {formatA}/{formatB}/{formatC} batch({count}) at {rowsA}/{rowsB}/{rowsC}x{cols}");
                BatchKernelTests.AssertBitsEqual(expectedB, actualB, $"fused triple B {formatA}/{formatB}/{formatC} batch({count}) at {rowsA}/{rowsB}/{rowsC}x{cols}");
                BatchKernelTests.AssertBitsEqual(expectedC, actualC, $"fused triple C {formatA}/{formatB}/{formatC} batch({count}) at {rowsA}/{rowsB}/{rowsC}x{cols}");
            }
        }
    }

    [Theory]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q4_K, QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K, QuantFormat.Q6_K)]
    public void FusedTriple_RaggedBatchCounts_ExerciseTripleGroupTailsBitIdentically(QuantFormat formatA, QuantFormat formatB, QuantFormat formatC)
    {
        // The auto posture triples at three chains: counts 10 and 11 leave one- and two-chain ragged tails, and 13/17 leave tails beside a
        // ragged pair/single segment (rows 512/384/256). Each tail chain must advance exactly like the full-group chains and like the
        // independent calls' own chains. The Qwen-shaped 2048/256/256 trio is included: its wide query region and narrow kv regions split
        // the union walk into a triple segment plus single-row segments.
        int cols = 768;
        foreach (int count in new[] { 10, 11, 13, 17 })
        {
            foreach ((int rowsA, int rowsB, int rowsC) in TripleRowShapes.Concat(QkvRowShapes))
            {
                using CpuBackend backend = new(threads: 1);

                (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rowsA, cols, formatA, seed: 61_000 + count, vectors: count);
                (_, byte[] weightsB) = BatchKernelTests.CreateCase(rowsB, cols, formatB, seed: 62_000 + count, vectors: count);
                (_, byte[] weightsC) = BatchKernelTests.CreateCase(rowsC, cols, formatC, seed: 63_000 + count, vectors: count);

                WeightRef weightA = new(weightsA, new TensorShape([rowsA, cols]), formatA);
                WeightRef weightB = new(weightsB, new TensorShape([rowsB, cols]), formatB);
                WeightRef weightC = new(weightsC, new TensorShape([rowsC, cols]), formatC);

                float[] expectedA = new float[count * rowsA];
                float[] expectedB = new float[count * rowsB];
                float[] expectedC = new float[count * rowsC];
                backend.GemvBatch(inputs, count, weightA, expectedA);
                backend.GemvBatch(inputs, count, weightB, expectedB);
                backend.GemvBatch(inputs, count, weightC, expectedC);

                float[] actualA = new float[count * rowsA];
                float[] actualB = new float[count * rowsB];
                float[] actualC = new float[count * rowsC];
                Assert.True(backend.TryGemvBatchSharedInput(inputs, count, weightA, actualA, weightB, actualB, weightC, actualC), "triple fusion refused at a ragged batch count");

                BatchKernelTests.AssertBitsEqual(expectedA, actualA, $"triple tail A {formatA}/{formatB}/{formatC} batch({count}) at {rowsA}/{rowsB}/{rowsC}x{cols}");
                BatchKernelTests.AssertBitsEqual(expectedB, actualB, $"triple tail B {formatA}/{formatB}/{formatC} batch({count}) at {rowsA}/{rowsB}/{rowsC}x{cols}");
                BatchKernelTests.AssertBitsEqual(expectedC, actualC, $"triple tail C {formatA}/{formatB}/{formatC} batch({count}) at {rowsA}/{rowsB}/{rowsC}x{cols}");
            }
        }
    }

    [Theory]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q4_K, QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K, QuantFormat.Q4_K)]
    public void FusedTriple_Threaded_IsBitIdenticalToSingleThreadedAndIndependentCalls(QuantFormat formatA, QuantFormat formatB, QuantFormat formatC)
    {
        // Row counts past the threaded posture's dispatch threshold so the union walk really threads; rows are independent, so any worker
        // assignment of triple rows must land on the single-threaded triple route's elements exactly — including the differing-row segments.
        int cols = 768;
        foreach (int count in new[] { 8, 9, 32 })
        {
            foreach ((int rowsA, int rowsB, int rowsC) in TripleRowShapes)
            {
                (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rowsA, cols, formatA, seed: 71_000 + count, vectors: count);
                (_, byte[] weightsB) = BatchKernelTests.CreateCase(rowsB, cols, formatB, seed: 72_000 + count, vectors: count);
                (_, byte[] weightsC) = BatchKernelTests.CreateCase(rowsC, cols, formatC, seed: 73_000 + count, vectors: count);

                WeightRef weightA = new(weightsA, new TensorShape([rowsA, cols]), formatA);
                WeightRef weightB = new(weightsB, new TensorShape([rowsB, cols]), formatB);
                WeightRef weightC = new(weightsC, new TensorShape([rowsC, cols]), formatC);

                using CpuBackend single = new(threads: 1);
                float[] singleA = new float[count * rowsA];
                float[] singleB = new float[count * rowsB];
                float[] singleC = new float[count * rowsC];
                Assert.True(single.TryGemvBatchSharedInput(inputs, count, weightA, singleA, weightB, singleB, weightC, singleC), "single-threaded triple route refused");

                using CpuBackend threaded = new(threads: 4, tuning: new CpuKernelTuningOverrides { RowBlock = 3, ParallelRowThreshold = 1 });
                float[] independentA = new float[count * rowsA];
                float[] independentB = new float[count * rowsB];
                float[] independentC = new float[count * rowsC];
                threaded.GemvBatch(inputs, count, weightA, independentA);
                threaded.GemvBatch(inputs, count, weightB, independentB);
                threaded.GemvBatch(inputs, count, weightC, independentC);

                float[] threadedA = new float[count * rowsA];
                float[] threadedB = new float[count * rowsB];
                float[] threadedC = new float[count * rowsC];
                Assert.True(threaded.TryGemvBatchSharedInput(inputs, count, weightA, threadedA, weightB, threadedB, weightC, threadedC), "threaded triple route refused");

                BatchKernelTests.AssertBitsEqual(singleA, threadedA, $"threaded triple route A {formatA}/{formatB}/{formatC} batch({count}) at {rowsA}/{rowsB}/{rowsC}x{cols}");
                BatchKernelTests.AssertBitsEqual(singleB, threadedB, $"threaded triple route B {formatA}/{formatB}/{formatC} batch({count}) at {rowsA}/{rowsB}/{rowsC}x{cols}");
                BatchKernelTests.AssertBitsEqual(singleC, threadedC, $"threaded triple route C {formatA}/{formatB}/{formatC} batch({count}) at {rowsA}/{rowsB}/{rowsC}x{cols}");
                BatchKernelTests.AssertBitsEqual(singleA, independentA, $"threaded independent A {formatA}/{formatB}/{formatC} batch({count})");
                BatchKernelTests.AssertBitsEqual(singleB, independentB, $"threaded independent B {formatA}/{formatB}/{formatC} batch({count})");
                BatchKernelTests.AssertBitsEqual(singleC, independentC, $"threaded independent C {formatA}/{formatB}/{formatC} batch({count})");
            }
        }
    }

    [Theory]
    // Formats outside the scratch route (integer-dot i-quants included), and a block-unaligned column count on an eligible format.
    [InlineData(QuantFormat.Q8_0, QuantFormat.Q8_0, QuantFormat.Q8_0)]
    [InlineData(QuantFormat.IQ4_XS, QuantFormat.IQ4_XS, QuantFormat.IQ4_XS)]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K, QuantFormat.Q4_K)]
    public void FusedTriple_OutsideFusedPostures_RefusesWithoutTouchingOutputs(QuantFormat formatA, QuantFormat formatB, QuantFormat formatC)
    {
        int cols = 137; // ragged for every block layout: the fused route must decline rather than approximate.

        foreach (int count in new[] { 8 })
        {
            using CpuBackend backend = new(threads: 1);

            (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(256, cols, formatA, seed: 555 + count, vectors: count);
            (_, byte[] weightsB) = BatchKernelTests.CreateCase(256, cols, formatB, seed: 666 + count, vectors: count);
            (_, byte[] weightsC) = BatchKernelTests.CreateCase(256, cols, formatC, seed: 777 + count, vectors: count);

            float[] outputsA = [.. Enumerable.Repeat(float.NaN, count * 256)];
            float[] outputsB = [.. Enumerable.Repeat(float.NaN, count * 256)];
            float[] outputsC = [.. Enumerable.Repeat(float.NaN, count * 256)];

            bool ran = backend.TryGemvBatchSharedInput(
                inputs, count,
                new WeightRef(weightsA, new TensorShape([256, cols]), formatA), outputsA,
                new WeightRef(weightsB, new TensorShape([256, cols]), formatB), outputsB,
                new WeightRef(weightsC, new TensorShape([256, cols]), formatC), outputsC);

            Assert.False(ran, $"{formatA}+{formatB}+{formatC} at {cols} columns must refuse shared-input triple fusion");
            Assert.All(outputsA, value => Assert.True(float.IsNaN(value)));
            Assert.All(outputsB, value => Assert.True(float.IsNaN(value)));
            Assert.All(outputsC, value => Assert.True(float.IsNaN(value)));
        }
    }

    [Theory]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q4_K, QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K, QuantFormat.Q6_K)]
    public void FusedTriple_ScratchCeilingDegradation_IsBitIdenticalToIndependentCalls(QuantFormat formatA, QuantFormat formatB, QuantFormat formatC)
    {
        // The triple scratch ceiling is the seam: at the default posture the triple route (three staged rows, one shared batch walk) runs;
        // with the ceiling between a pair of rows and a triple of rows the same call keeps fused execution on the pair route for two matrices
        // and the single-row scratch route for the third; with the ceiling below a pair the fused call degrades to three single-row scratch
        // passes. All three routes must land element-for-element on each other and on three independent batch calls.
        int rows = 512;
        int cols = 768;

        foreach (int count in new[] { 8, 9, 16, 32 }) // ragged counts exercise group tails in every degraded route
        {
            (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rows, cols, formatA, seed: 81_000 + count, vectors: count);
            (_, byte[] weightsB) = BatchKernelTests.CreateCase(rows, cols, formatB, seed: 82_000 + count, vectors: count);
            (_, byte[] weightsC) = BatchKernelTests.CreateCase(rows, cols, formatC, seed: 83_000 + count, vectors: count);

            WeightRef weightA = new(weightsA, new TensorShape([rows, cols]), formatA);
            WeightRef weightB = new(weightsB, new TensorShape([rows, cols]), formatB);
            WeightRef weightC = new(weightsC, new TensorShape([rows, cols]), formatC);

            using CpuBackend independent = new(threads: 1);
            float[] expectedA = new float[count * rows];
            float[] expectedB = new float[count * rows];
            float[] expectedC = new float[count * rows];
            independent.GemvBatch(inputs, count, weightA, expectedA);
            independent.GemvBatch(inputs, count, weightB, expectedB);
            independent.GemvBatch(inputs, count, weightC, expectedC);

            using CpuBackend tripleRoute = new(threads: 1); // default posture: 3 x 768 columns stay under the auto scratch ceiling
            Assert.Equal(CpuKernelTuning.TripleChainGroupMax, tripleRoute.ResolvedTuning.TripleChainGroup);
            float[] tripleA = new float[count * rows];
            float[] tripleB = new float[count * rows];
            float[] tripleC = new float[count * rows];
            Assert.True(tripleRoute.TryGemvBatchSharedInput(inputs, count, weightA, tripleA, weightB, tripleB, weightC, tripleC), $"{formatA}/{formatB}/{formatC} triple route refused at the default posture");

            using CpuBackend pairDegraded = new(threads: 1, tuning: new CpuKernelTuningOverrides { MaxScratchCols = 1536 }); // a pair of rows stageable, a triple of rows not: pair route for two, single-row scratch for the third
            float[] pairA = new float[count * rows];
            float[] pairB = new float[count * rows];
            float[] pairC = new float[count * rows];
            Assert.True(pairDegraded.TryGemvBatchSharedInput(inputs, count, weightA, pairA, weightB, pairB, weightC, pairC), $"{formatA}/{formatB}/{formatC} pair-degraded triple fusion refused at a pinned scratch ceiling");

            using CpuBackend singleDegraded = new(threads: 1, tuning: new CpuKernelTuningOverrides { MaxScratchCols = 1024 }); // one row stageable, a pair of rows not: three single-row scratch passes
            float[] singleA = new float[count * rows];
            float[] singleB = new float[count * rows];
            float[] singleC = new float[count * rows];
            Assert.True(singleDegraded.TryGemvBatchSharedInput(inputs, count, weightA, singleA, weightB, singleB, weightC, singleC), $"{formatA}/{formatB}/{formatC} single-degraded triple fusion refused at a pinned scratch ceiling");

            BatchKernelTests.AssertBitsEqual(expectedA, tripleA, $"triple route A {formatA}/{formatB}/{formatC} batch({count}) at {rows}x{cols}");
            BatchKernelTests.AssertBitsEqual(expectedB, tripleB, $"triple route B {formatA}/{formatB}/{formatC} batch({count}) at {rows}x{cols}");
            BatchKernelTests.AssertBitsEqual(expectedC, tripleC, $"triple route C {formatA}/{formatB}/{formatC} batch({count}) at {rows}x{cols}");
            BatchKernelTests.AssertBitsEqual(expectedA, pairA, $"pair-degraded A {formatA}/{formatB}/{formatC} batch({count})");
            BatchKernelTests.AssertBitsEqual(expectedB, pairB, $"pair-degraded B {formatA}/{formatB}/{formatC} batch({count})");
            BatchKernelTests.AssertBitsEqual(expectedC, pairC, $"pair-degraded C {formatA}/{formatB}/{formatC} batch({count})");
            BatchKernelTests.AssertBitsEqual(expectedA, singleA, $"single-degraded A {formatA}/{formatB}/{formatC} batch({count})");
            BatchKernelTests.AssertBitsEqual(expectedB, singleB, $"single-degraded B {formatA}/{formatB}/{formatC} batch({count})");
            BatchKernelTests.AssertBitsEqual(expectedC, singleC, $"single-degraded C {formatA}/{formatB}/{formatC} batch({count})");
        }
    }

    [Theory]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q4_K, QuantFormat.Q4_K)]
    [InlineData(QuantFormat.Q4_K, QuantFormat.Q6_K, QuantFormat.Q6_K)]
    public void FusedTriple_PinnedNarrowerTripleGroup_IsBitIdenticalToIndependentCalls(QuantFormat formatA, QuantFormat formatB, QuantFormat formatC)
    {
        // A pinned chain group narrows the triple group derived from it (two chains triple at two, one at one), so the triple dot runs
        // through its stack-staged route. Triple grouping only chooses which batch vectors share one pass over the triple — each chain's
        // element order and reduction stay the same, so the bytes must not move.
        int rows = 512;
        int cols = 768;

        foreach (int chainGroup in new[] { 2, 1 })
        {
            foreach (int count in new[] { 8, 9, 10 })
            {
                (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(rows, cols, formatA, seed: 91_000 + count, vectors: count);
                (_, byte[] weightsB) = BatchKernelTests.CreateCase(rows, cols, formatB, seed: 92_000 + count, vectors: count);
                (_, byte[] weightsC) = BatchKernelTests.CreateCase(rows, cols, formatC, seed: 93_000 + count, vectors: count);

                WeightRef weightA = new(weightsA, new TensorShape([rows, cols]), formatA);
                WeightRef weightB = new(weightsB, new TensorShape([rows, cols]), formatB);
                WeightRef weightC = new(weightsC, new TensorShape([rows, cols]), formatC);

                using CpuBackend independent = new(threads: 1);
                float[] expectedA = new float[count * rows];
                float[] expectedB = new float[count * rows];
                float[] expectedC = new float[count * rows];
                independent.GemvBatch(inputs, count, weightA, expectedA);
                independent.GemvBatch(inputs, count, weightB, expectedB);
                independent.GemvBatch(inputs, count, weightC, expectedC);

                using CpuBackend pinned = new(threads: 1, tuning: new CpuKernelTuningOverrides { ChainGroup = chainGroup });
                Assert.Equal(chainGroup, pinned.ResolvedTuning.TripleChainGroup); // a resolved group narrower than the budget caps the triple group to it
                float[] actualA = new float[count * rows];
                float[] actualB = new float[count * rows];
                float[] actualC = new float[count * rows];
                Assert.True(pinned.TryGemvBatchSharedInput(inputs, count, weightA, actualA, weightB, actualB, weightC, actualC), $"{formatA}/{formatB}/{formatC} fused triple refused at a pinned chain group");

                BatchKernelTests.AssertBitsEqual(expectedA, actualA, $"narrow triple group {chainGroup} A {formatA}/{formatB}/{formatC} batch({count}) at {rows}x{cols}");
                BatchKernelTests.AssertBitsEqual(expectedB, actualB, $"narrow triple group {chainGroup} B {formatA}/{formatB}/{formatC} batch({count}) at {rows}x{cols}");
                BatchKernelTests.AssertBitsEqual(expectedC, actualC, $"narrow triple group {chainGroup} C {formatA}/{formatB}/{formatC} batch({count}) at {rows}x{cols}");
            }
        }
    }

    [Fact]
    public void FusedTriple_MismatchedColumnCounts_ThrowsClearMessage()
    {
        using CpuBackend backend = new(threads: 1);

        (float[] inputs, byte[] weightsA) = BatchKernelTests.CreateCase(256, 384, QuantFormat.Q4_K, seed: 7, vectors: 8);
        (_, byte[] weightsB) = BatchKernelTests.CreateCase(256, 512, QuantFormat.Q6_K, seed: 8, vectors: 8);
        (_, byte[] weightsC) = BatchKernelTests.CreateCase(256, 384, QuantFormat.Q4_K, seed: 9, vectors: 8);

        float[] outputsA = new float[8 * 256];
        float[] outputsB = new float[8 * 256];
        float[] outputsC = new float[8 * 256];

        ArgumentException error = Assert.Throws<ArgumentException>(() => backend.TryGemvBatchSharedInput(
            inputs, 8, new WeightRef(weightsA, new TensorShape([256, 384]), QuantFormat.Q4_K), outputsA,
            new WeightRef(weightsB, new TensorShape([256, 512]), QuantFormat.Q6_K), outputsB,
            new WeightRef(weightsC, new TensorShape([256, 384]), QuantFormat.Q4_K), outputsC));

        Assert.Contains("equal column counts", error.Message);
    }
}
